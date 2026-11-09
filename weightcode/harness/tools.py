"""The editor tools behind the MCP server (HARNESS_SPEC section 4), independent of the JSON-RPC transport.

``ToolServer`` holds the session's sandbox kernel, the check ledger (the counted patch-check oracle with the real
model as base, see ``weightcode.harness.ledger``) and the arm rules. Calls are handled one at a time by the caller's single
thread, so the query budget and the cell counter cannot race.

Information rules: results carry no per-example check-set information, no host path and no content of any file
outside /work/patches. A patch that fails validation consumes no query. A patch that passes validation but whose
patched model produces non-finite logits on the check set is charged one query, because that outcome depends on the
check set (the oracle itself does not count it).

No result carries a number computed from the parameter values a patch was applied to (review finding A,
2026-11-03; the same in every arm): a rejection names the failed limit and its threshold, never the measured value
(``weightcode.harness.applying.EditorRejection``), and a patch-check result omits the oracle's ``child_sha256`` and
``n_changed``. Both, and the executor's full rejection text, stay in the host logs (``attempts/*/attempt.json``,
``logs/checks.jsonl``, ``logs/tools.jsonl`` metadata and session_result.json).

Every patch-check, predict and declare_final call first copies the patch and its .npy files into a private attempt
directory (``weightcode.harness.snapshot``) and works on that copy, so what was checked is exactly what is stored.

In an arm without ``patch_check`` (M2NL) the kept patch is the one of the last valid declaration
(``last_valid_declaration``, contract 5.2 with a budget of zero), so ``status`` and ``declare_final`` describe that
rule (``NO_CHECK_DESCRIPTIONS``) and report ``kept_patch_file``, with no query or check fields.

In a program arm (LEARN) ``patch_check`` and ``declare_final`` take ``program_file``, a .py file in /work/patches.
``patch_check`` snapshots the program, runs it once in the program sandbox on the session's own student and experience
(``weightcode.harness.program``), validates the deltas, writes them as the derived add_dense patch into the attempt directory
and applies that patch under the arm rules, so a limit failure reads exactly as on the patch path. A run or validation
failure consumes no query and returns its problems verbatim (the run only saw editor-visible inputs). A valid update is
one joint query; the editor sees ``score_new``, ``score_old`` and ``score`` (S_joint) only, with the program's own
stdout and stderr. ``declare_final`` snapshots the program and records whether it is byte-identical to a checked one; it
does not run it.

LEARN_BLIND (PREREG A4a, ``ArmSpec.blind``): a program runs on the private copies of the real weights and of the whole
LEARN experience (session.json ``blind``, hash-checked here), since the trial has neither. The run is a blind run
(``weightcode.harness.program.run_update`` ``blind``): a valid check returns ``score_new``, ``score_old``, ``score``,
``exit_status`` (``ok``), ``query_index`` and ``queries_remaining``, with no stdout, stderr, run time or tensor list,
and a refused run returns its exit status and the stage at which it failed in fixed wording, naming no tensor. A limit
failure gives its kinds with their thresholds (``weightcode.harness.applying.blind_rejection_messages``). The host logs keep
the full text.

The information ladder (PREREG; session.json ``e11_variant``, ``weightcode.harness.e11``). U (``unlabelled``) changes no
tool. A no-data session (K0b ``named``, K0a ``named_nocheck``) has no trial experience file: a program runs on the
session's private empty experience (session.json ``e11_view``, hash-checked here), and ``patch_check`` runs it "on
this classifier" (``named_check_description``). K0a has no ``patch_check``: ``declare_final`` snapshots the program,
runs it once and validates and applies its update within the limits exactly as a check would, and scores nothing; a
program that fails to run or fails a limit is not a valid declaration and gets its problems. ``status`` gives the cells
and the last declaration. Neither says "check" (``NOCHECK_PROGRAM_DESCRIPTIONS``); the last valid declaration is
checked once after the session (``weightcode.harness.finalize``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import shutil
import time

import numpy as np

from weightcode.evalkit.evidence import evidence_content_sha256
from weightcode.evalkit.patch_check import BudgetExceeded
from weightcode.harness.applying import REDACTION_BLIND, EditorRejection, SessionConfigError, blind_rejection_messages
from weightcode.harness.arms import PATCH_CHECK_TOOL, PREDICT_TOOL, arm_spec
from weightcode.harness.jsonio import append_jsonl, file_sha256, read_json, read_jsonl, utc_now, write_json
from weightcode.harness.launch import LaunchStartError, read_launch_start
from weightcode.harness import fscil, loop
from weightcode.harness.seqskill import SCAN_RULE
from weightcode.harness.ledger import CHECK_RECORD_SCHEMA, CheckLedger
from weightcode.harness.retention import session_floor
from weightcode.harness.session import SessionPaths
from weightcode.harness.program import (
    DELTAS_FILE,
    DERIVED_PATCH,
    MAX_PROGRAM_BYTES,
    RUN_DIR,
    RUN_RECORD,
    ProgramProblem,
    Student,
    program_name,
    run_update,
    write_derived_patch,
)
from weightcode.harness.snapshot import SnapshotProblem, patch_name, snapshot_file, snapshot_patch
from weightcode.patch import PatchError, PatchResult
from weightcode.sandbox import SandboxConfig, SandboxKernel
from weightcode.sandbox.program import ProgramSandbox

DECLARED_SCHEMA = "weightcode.harness.declared.v0"
ATTEMPT_SCHEMA = "weightcode.harness.attempt.v0"
MAX_RATIONALE_CHARS = 20_000
PREDICT_SUBSETS = ("fail", "rand", "all")
PREDICT_PARTS = {"fail": ("x_fail",), "rand": ("x_rand",), "all": ("x_fail", "x_rand")}
LOGIT_DECIMALS = 4
NON_FINITE_REASON = "the patched model produces non-finite logits on the check set"
EDITOR_HIDDEN_CHECK_FIELDS = ("child_sha256", "n_changed")  # computed on the real child: host logs only

TOOL_DESCRIPTIONS: dict[str, str] = {
    "python": (
        "Run Python code in the persistent kernel. Variables, imports and definitions persist between calls. numpy "
        "and the standard library are available; there is no network. The value of a final expression is shown. "
        "/work/trial is read-only; write patches and the .npy files they reference to /work/patches and other files "
        "to /work/scratch. Returns stdout, stderr, error, duration_s and cells_remaining."
    ),
    "patch_check": (
        "Check one patch. patch_file is the bare file name of a patch JSON in /work/patches. The patch is validated "
        "and applied, and the patched model is evaluated on the held-out check set. Returns aggregate numbers only: "
        "score (S = fixed - broken), fixed, broken, n (check-set size), n_base_errors, the accuracy of the original "
        "(base_acc) and of the patched model (child_acc), edit_size and edit_frac (numbers the patch specifies, and "
        "their fraction of all parameter entries) and queries_remaining. A successful check uses one query; a patch "
        "that fails validation uses none and its problems are returned (a failed size limit is named with its "
        "threshold). Each file name can be checked once."
    ),
    "status": "Queries remaining, Python cells remaining, every checked patch with its S, and your last declaration.",
    "declare_final": (
        "Record the patch you want kept: patch_file is a file name in /work/patches, or null for the original model; "
        "rationale says why. The declaration is recorded as your own choice; the kept patch is still the checked "
        "patch with the highest S (the original model if none has S > 0). It does not end the session; the last "
        "declaration counts."
    ),
    PREDICT_TOOL: (
        "Logits of the original model (patch_file null), or of the original model with the patch patch_file from "
        "/work/patches applied, on the evidence examples: subset 'fail' (x_fail), 'rand' (x_rand) or 'all' (x_fail "
        "then x_rand), in evidence.npz order, rounded to 4 decimals. Not counted against any budget."
    ),
}

NO_CHECK_DESCRIPTIONS: dict[str, str] = {
    "status": "Python cells remaining, your last declaration, and kept_patch_file: the patch kept if the session "
              "ended now (null for the original model).",
    "declare_final": (
        "Declare the patch you want kept: patch_file is a file name in /work/patches, or null for the original model; "
        "rationale says why. The patch is validated and applied within the limits, but it is not scored. A patch "
        "that fails validation is not accepted and its problems are returned (a failed size limit is named with its "
        "threshold). The last valid declaration is kept at the end. It does not end the session."
    ),
}
NO_CHECK_NOTE = "the last valid declaration is kept at the end; a declaration that fails validation changes nothing"

# The two check sets of a program arm's patch_check: evidence-v1 (a condition and its clean companion) and fewshot-v1
# (PREREG E2: V_check cut by class). Only this phrase differs between the two descriptions.
PROGRAM_CHECK_SETS = "one of new-condition images and one of clean images"
FEWSHOT_CHECK_SETS = "one of images of the new classes and one of images of the old classes"
# seqskill-v1 (PREREG E5: V_check cut by part) also names the model, which is not a classifier, and the row unit.
SEQSKILL_CHECK_SETS = ("one of rows of the new operator alone and one of old-operator rows (a row is right when every "
                       "output slot is right)")
# scanskill-v1 (PREREG E13, the seqskill variant scan_rule): target commands and held-out commands of the training
# distribution, with the model words and the row unit of seqskill-v1.
SCAN_CHECK_SETS = ("one of target commands and one of commands of the training distribution that were never in the "
                   "model's training rows (a row is right when every output slot is right, END included)")
# graphnew-v1 (PREREG E7: V_check cut by class, as nodes of one graph) names nodes where fewshot-v1 names images.
GRAPHNEW_CHECK_SETS = "one of nodes of the new classes and one of nodes of the old classes"
# fscil-v1 (PREREG E9: one incremental session) names the classes of this session and the classes seen before it.
FSCIL_CHECK_SETS = ("one of images of the classes added in this session and one of images of the classes seen before "
                    "(the base classes and those of the earlier sessions), both under this session's prediction rule")
SEQSKILL_MODEL_WORDS = (("on this classifier and its experience", "on this model and its experience"),
                        ("the updated classifier is evaluated", "the updated model is evaluated"))
# policyskill-v1 (PREREG E14: V_check is environment seeds, run as greedy episodes) names the policy and the episodes.
POLICYSKILL_CHECK_SETS = ("one of episodes of the new mission and one of episodes of the trained mission (each "
                          "episode run greedily in its own room, right when its mission is completed)")
POLICYSKILL_MODEL_WORDS = (("on this classifier and its experience", "on this policy and its experience"),
                           ("the updated classifier is evaluated", "the updated policy is evaluated"))
PROGRAM_DESCRIPTIONS: dict[str, str] = {
    "patch_check": (
        "Check one program. program_file is the bare name of a .py file in /work/patches that defines "
        "update(student, experience) and the strings HYPOTHESIS and PREDICTED_EFFECT. The program is run once, in a "
        "fresh sandbox, on this classifier and its experience; its deltas are validated and applied within the limits, "
        "and the updated classifier is evaluated on two held-out check sets, " + PROGRAM_CHECK_SETS + ". Returns "
        "score_new and score_old (fixed - broken on each set), score = score_new + score_old "
        "(S_joint), the tensors the deltas change, the program's stdout and stderr, duration_s and queries_remaining. "
        "A successful check uses one query; a program that fails to run, returns invalid deltas or fails a limit uses "
        "none, and its problems are returned (a failed limit is named with its threshold). Each file name can be "
        "checked once."
    ),
    "status": "Queries remaining, Python cells remaining, every checked program with its score_new, score_old and "
              "score, and your last declaration.",
    "declare_final": (
        "Record the program you want kept: program_file is the name of a .py file in /work/patches, or null for the "
        "original model; rationale says why. The declaration is recorded as your own choice; the kept update is "
        "still that of the checked program with the highest score (the original model if none has score > 0). It "
        "does not run the program and does not end the session; the last declaration counts."
    ),
}
# LEARN_BLIND (PREREG A4a): what patch_check returns, in place of this part of the program arm's description.
PROGRAM_RETURNS = ("(S_joint), the tensors the deltas change, the program's stdout and stderr, duration_s and "
                   "queries_remaining. A successful check uses one query; a program that fails to run, returns invalid "
                   "deltas or fails a limit uses none, and its problems are returned (a failed limit is named with its "
                   "threshold).")
BLIND_RETURNS = ("(S_joint), exit_status and queries_remaining. Nothing the program prints, raises or writes is "
                 "returned. exit_status is ok (the program returned its deltas), error (it raised an exception), "
                 "timeout (it did not return within the wall limit) or died (it ended without a result, for example at "
                 "a memory or file-size limit or by exiting). A successful check uses one query; a program that fails "
                 "to run, returns invalid deltas or fails a limit uses none, and its exit status and the stage at "
                 "which it failed are returned, naming no tensor (a failed limit is named with its threshold).")
# The no-data views of the information ladder (module docstring): what patch_check runs on (K0b), and K0a's two tools.
NAMED_CHECK_WORDS = ("on this classifier and its experience;", "on this classifier;")
NOCHECK_PROGRAM_DESCRIPTIONS: dict[str, str] = {
    "status": "Python cells remaining, and your last declaration with whether it is valid. Nothing is scored during "
              "the session.",
    "declare_final": (
        "Declare the program you want kept: program_file is the name of a .py file in /work/patches, or null for the "
        "original model; rationale says why. The program is run once, in a fresh sandbox, on this classifier; its "
        "deltas are validated and applied within the limits, but the updated classifier is not scored. Returns "
        "whether the declaration is valid, the tensors the deltas change, the program's stdout and stderr and "
        "duration_s; a program that fails to run, returns invalid deltas or fails a limit is not a valid declaration, "
        "and its problems are returned (a failed limit is named with its threshold). After the session the update of "
        "your last valid declaration is scored once on held-out images and kept only if its score_old is at least "
        "{floor} and its score > 0. It does not end the session; the last valid declaration counts."
    ),
}
NOCHECK_DECLARE_NOTE = "the last valid declaration is the one scored after the session; one that is not valid " \
    "changes " \
                       "nothing"
NOCHECK_REFUSED = "Nothing was applied, and the declaration is not valid."
NOCHECK_NOT_RUN = "Nothing was run, and the declaration is not valid."


def named_check_description() -> str:
    """The program arm's ``patch_check`` description for a no-data session (K0b): it runs on this classifier."""
    text = PROGRAM_DESCRIPTIONS["patch_check"]
    if text.count(NAMED_CHECK_WORDS[0]) != 1:  # pragma: no cover - both strings are fixed above
        raise RuntimeError("the program patch_check description no longer runs on the experience once")
    return text.replace(*NAMED_CHECK_WORDS)


# The retention kept rule (weightcode.harness.retention) replaces this phrase of declare_final; nothing else changes.
PROGRAM_KEPT_PHRASE = "the checked program with the highest score (the original model if none has score > 0)"
_NULLABLE_PROGRAM = {"type": ["string", "null"], "description": "bare name of a .py file in /work/patches, or null"}
PROGRAM_SCHEMAS: dict[str, dict[str, Any]] = {
    "patch_check": {
        "type": "object",
        "properties": {"program_file": {"type": "string", "description": "bare name of a .py file in /work/patches"}},
        "required": ["program_file"],
        "additionalProperties": False,
    },
    "declare_final": {
        "type": "object",
        "properties": {"program_file": _NULLABLE_PROGRAM, "rationale": {"type": "string"}},
        "required": ["program_file", "rationale"],
        "additionalProperties": False,
    },
}
PROGRAM_SHOWN_CHECK_FIELDS = ("score_new", "score_old", "score")

_NULLABLE_FILE = {"type": ["string", "null"], "description": "bare file name in /work/patches, or null"}
INPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "python": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to run in the kernel"}},
        "required": ["code"],
        "additionalProperties": False,
    },
    "patch_check": {
        "type": "object",
        "properties": {"patch_file": {"type": "string", "description": "bare file name in /work/patches"}},
        "required": ["patch_file"],
        "additionalProperties": False,
    },
    "status": {"type": "object", "properties": {}, "additionalProperties": False},
    "declare_final": {
        "type": "object",
        "properties": {"patch_file": _NULLABLE_FILE, "rationale": {"type": "string"}},
        "required": ["patch_file", "rationale"],
        "additionalProperties": False,
    },
    PREDICT_TOOL: {
        "type": "object",
        "properties": {"patch_file": _NULLABLE_FILE, "subset": {"type": "string", "enum": list(PREDICT_SUBSETS)}},
        "required": ["subset"],
        "additionalProperties": False,
    },
}


class ToolError(Exception):
    """A request the editor can fix; it becomes an ``isError`` result. Nothing was consumed unless it says so.

    ``extra`` goes into the editor-facing payload; ``meta`` only into the host tool log.
    """

    def __init__(self, messages: list[str] | str, *, meta: Mapping[str, Any] | None = None, **extra: Any) -> None:
        self.messages = [messages] if isinstance(messages, str) else list(messages)
        self.extra = extra
        self.meta = dict(meta or {})
        super().__init__("\n".join(self.messages))


class UnknownTool(KeyError):
    """The tool name is not one of this session's tools."""


@dataclass
class ToolOutcome:
    """Editor-facing payload, error flag, and host-only metadata for the tool log."""

    payload: dict[str, Any]
    is_error: bool
    meta: dict[str, Any] = field(default_factory=dict)


def fewshot_check_description() -> str:
    """The program arm's ``patch_check`` description with the fewshot-v1 check sets."""
    text = PROGRAM_DESCRIPTIONS["patch_check"]
    if text.count(PROGRAM_CHECK_SETS) != 1:  # pragma: no cover - both strings are fixed above
        raise RuntimeError("the program patch_check description no longer names its check sets once")
    return text.replace(PROGRAM_CHECK_SETS, FEWSHOT_CHECK_SETS)


def graphnew_check_description() -> str:
    """The program arm's ``patch_check`` description with the graphnew-v1 check sets (fewshot's, as nodes)."""
    text = PROGRAM_DESCRIPTIONS["patch_check"]
    if text.count(PROGRAM_CHECK_SETS) != 1:  # pragma: no cover - both strings are fixed above
        raise RuntimeError("the program patch_check description no longer names its check sets once")
    return text.replace(PROGRAM_CHECK_SETS, GRAPHNEW_CHECK_SETS)


def fscil_check_description() -> str:
    """The program arm's ``patch_check`` description with the fscil-v1 check sets."""
    text = PROGRAM_DESCRIPTIONS["patch_check"]
    if text.count(PROGRAM_CHECK_SETS) != 1:  # pragma: no cover - both strings are fixed above
        raise RuntimeError("the program patch_check description no longer names its check sets once")
    return text.replace(PROGRAM_CHECK_SETS, FSCIL_CHECK_SETS)


# E9 v2 (kept rule fscil-retain-v2 on check score fscil-cb-v1): patch_check, status and declare_final state the v2
# scores; every other sentence of the fscil-v1 descriptions is kept. This is the part of patch_check they replace.
FSCIL_V1_RETURNS = ("and the updated classifier is evaluated on two held-out check sets, " + FSCIL_CHECK_SETS
                    + ". Returns score_new and score_old (fixed - broken on each set), score = score_new + score_old "
                    "(S_joint), ")
FSCIL_V2_STATUS = ("Queries remaining, Python cells remaining, every checked program with its S, S_new, S_prev, "
                   "S_base, B_c, d_c, floor_c and eligible, and your last declaration.")
FSCIL_V2_KEPT_PHRASE = ("the eligible checked program (B_c >= floor_c) with the highest S (the original model if none "
                        "of them has S > 0)")


def fscil_v2_descriptions(rule: Mapping[str, Any]) -> dict[str, str]:
    """``patch_check``, ``status`` and ``declare_final`` of an E9 v2 session under ``rule`` (a
    ``weightcode.harness.fscil.kept_rule_v2`` record): the fscil-v1 texts with the v2 scores and kept rule."""
    from weightcode.harness.fscil import v2_floor_formula, v2_numbers

    check = fscil_check_description()
    declare = PROGRAM_DESCRIPTIONS["declare_final"]
    if check.count(FSCIL_V1_RETURNS) != 1 or declare.count(PROGRAM_KEPT_PHRASE) != 1:
        raise RuntimeError("the fscil-v1 descriptions no longer state the v1 scores once")
    n_c = v2_numbers(rule)["n_c"]
    returns = ("and the updated classifier is evaluated on held-out check images of every class seen so far, under "
               "this session's prediction rule. Returns, in percentage points, S (the change of the class-balanced "
               "accuracy over the classes seen so far), its parts S_new, S_prev and S_base (from the classes added in "
               "this session, the classes of the earlier sessions and the base classes; they sum to S), B_c (the "
               f"change of the base classes' class-balanced accuracy), d_c (the program's own base flip rate: fixed + "
               f"broken over the {n_c} check images of the base classes, a fraction), floor_c = "
               f"{v2_floor_formula(rule)} (the program's own floor on B_c) and eligible (B_c >= floor_c), ")
    return {"patch_check": check.replace(FSCIL_V1_RETURNS, returns), "status": FSCIL_V2_STATUS,
            "declare_final": declare.replace(PROGRAM_KEPT_PHRASE, FSCIL_V2_KEPT_PHRASE)}


def blind_check_description() -> str:
    """The program arm's ``patch_check`` description for LEARN_BLIND: what a check returns (``BLIND_RETURNS``)."""
    text = PROGRAM_DESCRIPTIONS["patch_check"]
    if text.count(PROGRAM_RETURNS) != 1:  # pragma: no cover - both strings are fixed above
        raise RuntimeError("the program patch_check description no longer states what it returns once")
    return text.replace(PROGRAM_RETURNS, BLIND_RETURNS)


def seqskill_check_description(scan: bool = False) -> str:
    """The program arm's ``patch_check`` description for a seqskill-v1 instance: the model words and the check sets
    (``SCAN_CHECK_SETS`` for a scanskill-v1 instance, PREREG E13)."""
    text = PROGRAM_DESCRIPTIONS["patch_check"]
    for old, new in ((PROGRAM_CHECK_SETS, SCAN_CHECK_SETS if scan else SEQSKILL_CHECK_SETS), *SEQSKILL_MODEL_WORDS):
        if text.count(old) != 1:  # pragma: no cover - every string is fixed above
            raise RuntimeError(f"the program patch_check description no longer says {old!r} once")
        text = text.replace(old, new)
    return text


def policyskill_check_description() -> str:
    """The program arm's ``patch_check`` description for a policyskill-v1 instance: the policy and the episodes."""
    text = PROGRAM_DESCRIPTIONS["patch_check"]
    for old, new in ((PROGRAM_CHECK_SETS, POLICYSKILL_CHECK_SETS), *POLICYSKILL_MODEL_WORDS):
        if text.count(old) != 1:  # pragma: no cover - every string is fixed above
            raise RuntimeError(f"the program patch_check description no longer says {old!r} once")
        text = text.replace(old, new)
    return text


def tool_schemas(program: bool = False) -> dict[str, dict[str, Any]]:
    """The input schemas of the tools: a program arm's ``patch_check`` and ``declare_final`` take ``program_file``."""
    return INPUT_SCHEMAS | PROGRAM_SCHEMAS if program else dict(INPUT_SCHEMAS)


def retention_declare_description(s_old_floor: int) -> str:
    """The program arm's ``declare_final`` description under the retention kept rule with floor ``s_old_floor``."""
    text = PROGRAM_DESCRIPTIONS["declare_final"]
    if text.count(PROGRAM_KEPT_PHRASE) != 1:
        raise ValueError("declare_final does not state the v1 kept rule")
    return text.replace(PROGRAM_KEPT_PHRASE, (
        f"the checked program with the highest score among those with score_old of at least {s_old_floor} (the "
        f"original model if none of them has score > 0)"))


def tool_specs(tools: tuple[str, ...] | list[str], program: bool = False, fewshot: bool = False,
               s_old_floor: int | None = None, loop_rule: Mapping[str, Any] | None = None,
               seqskill: bool = False, graphnew: bool = False, fscil: bool = False,
               fscil_v2: Mapping[str, Any] | None = None, blind: bool = False,
               e11_variant: str | None = None, scan: bool = False,
               policyskill: bool = False) -> list[dict[str, Any]]:
    """The MCP ``tools/list`` entries of one arm; without ``patch_check`` the declared-patch rule's descriptions, in a
    program arm the program tools' descriptions; ``fewshot`` (a fewshot-v1 LEARN session, PREREG E2) names the two
    check sets of ``patch_check`` as the images of the new classes and of the old classes; ``s_old_floor`` (the
    retention kept rule, ``weightcode.harness.retention``) states that rule in ``declare_final``; ``loop_rule`` (a loop
    round, ``weightcode.harness.loop``) names the loop's check sets and scores in ``patch_check`` and ``status`` and states
    its two floors in ``declare_final``; ``seqskill`` (a seqskill-v1 LEARN session, PREREG E5) names the model and
    the two check sets of ``patch_check`` as rows of the new operator alone and old-operator rows; ``graphnew`` (a
    graphnew-v1 LEARN session, PREREG E7) names them as the nodes of the new classes and of the old classes; ``fscil``
    (an fscil-v1 LEARN session, PREREG E9) as the images of the classes added in this session and of the classes seen
    before; ``fscil_v2`` (with ``fscil``: the session's ``kept_rule_v2`` record, PREREG E9 v2) states the v2 scores and
    kept rule in ``patch_check``, ``status`` and ``declare_final`` (``fscil_v2_descriptions``); ``blind`` (a LEARN_BLIND
    session on an evidence-v1 instance, PREREG A4a) states what its ``patch_check`` returns (``blind_check_description``).
    ``e11_variant`` (the information ladder, module docstring): "named" (K0b) runs ``patch_check`` on this classifier
    alone (``named_check_description``); "named_nocheck" (K0a, no ``patch_check``, a retention floor required) gives
    ``NOCHECK_PROGRAM_DESCRIPTIONS`` with the floor; "unlabelled" changes nothing. ``scan`` (with ``seqskill``: a
    scanskill-v1 session, PREREG E13) names the check sets as target commands and held-out commands of the training
    distribution (``SCAN_CHECK_SETS``). ``policyskill`` (a policyskill-v1 LEARN session, PREREG E14) names the policy
    and the two check sets of ``patch_check`` as episodes of the new mission and of the trained mission
    (``policyskill_check_description``).
    """
    from weightcode.harness.e11 import NAMED, NAMED_NOCHECK, check_variant

    variant = check_variant(e11_variant)
    rule = fewshot or seqskill or graphnew or fscil or policyskill
    if rule and not program:
        raise ValueError("a fewshot-v1, seqskill-v1, graphnew-v1, fscil-v1 or policyskill-v1 session runs in a "
                         "program arm only")
    if blind and (not program or rule or loop_rule is not None):
        raise ValueError("a blind session is a program arm on an evidence-v1 instance, not a loop round")
    if fewshot + seqskill + graphnew + fscil + policyskill > 1:
        raise ValueError("an instance follows one of fewshot-v1, seqskill-v1, graphnew-v1, fscil-v1 and "
                         "policyskill-v1")
    if scan and not seqskill:
        raise ValueError("a scanskill-v1 session is a seqskill session of variant scan_rule")
    if (s_old_floor is not None or loop_rule is not None) and not program:
        raise ValueError("the retention kept rule applies to a program arm only")
    if loop_rule is not None and (rule or s_old_floor is not None):
        raise ValueError("a loop round is an evidence-v1 session with the loop rule only")
    if fscil_v2 is not None and (not fscil or s_old_floor is not None):
        raise ValueError("the E9 v2 kept rule is an fscil-v1 session's only kept rule")
    descriptions = TOOL_DESCRIPTIONS if PATCH_CHECK_TOOL in tools else TOOL_DESCRIPTIONS | NO_CHECK_DESCRIPTIONS
    if program:
        descriptions = descriptions | PROGRAM_DESCRIPTIONS
    if fewshot:
        descriptions = descriptions | {"patch_check": fewshot_check_description()}
    if seqskill:
        descriptions = descriptions | {"patch_check": seqskill_check_description(scan)}
    if graphnew:
        descriptions = descriptions | {"patch_check": graphnew_check_description()}
    if policyskill:
        descriptions = descriptions | {"patch_check": policyskill_check_description()}
    if fscil:
        descriptions = descriptions | {"patch_check": fscil_check_description()}
    if fscil_v2 is not None:
        descriptions = descriptions | fscil_v2_descriptions(fscil_v2)
    if blind:
        descriptions = descriptions | {"patch_check": blind_check_description()}
    if s_old_floor is not None:
        descriptions = descriptions | {"declare_final": retention_declare_description(s_old_floor)}
    if loop_rule is not None:
        descriptions = descriptions | {
            "patch_check": loop.loop_check_description(PROGRAM_DESCRIPTIONS["patch_check"], PROGRAM_CHECK_SETS),
            "status": loop.LOOP_STATUS_DESCRIPTION,
            "declare_final": loop.loop_declare_description(PROGRAM_DESCRIPTIONS["declare_final"], PROGRAM_KEPT_PHRASE,
                                                           loop_rule),
        }
    if variant in (NAMED, NAMED_NOCHECK) and (not program or blind or rule or loop_rule is not None):
        raise ValueError("a no-data E11 session is a LEARN session on an evidence-v1 instance, not a loop round")
    if variant == NAMED:
        descriptions = descriptions | {"patch_check": named_check_description()}
    if variant == NAMED_NOCHECK:
        if PATCH_CHECK_TOOL in tools or s_old_floor is None:
            raise ValueError("the K0a session has no patch_check and keeps under a retention floor")
        descriptions = descriptions | {
            "status": NOCHECK_PROGRAM_DESCRIPTIONS["status"],
            "declare_final": NOCHECK_PROGRAM_DESCRIPTIONS["declare_final"].format(floor=s_old_floor)}
    schemas = tool_schemas(program)
    return [{"name": name, "description": descriptions[name], "inputSchema": schemas[name]} for name in tools]


def last_valid_declaration(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The last ``declared.jsonl`` record that passed validation (a patch or null), or None.

    With a patch-check budget of zero this declaration is the final (contract 5.2); an invalid declaration changes
    nothing, like a patch that fails validation in ``patch_check``.
    """
    valid = [record for record in records if record.get("valid")]
    return valid[-1] if valid else None


def sandbox_config(session: Mapping[str, Any], paths: SessionPaths) -> SandboxConfig:
    """The session's SandboxConfig, from session.json."""
    sandbox = dict(session["sandbox"])
    return SandboxConfig(
        trial_dir=paths.trial,
        patches_dir=paths.patches,
        scratch_dir=paths.scratch,
        log_path=paths.cells_log,
        python=Path(str(sandbox["python"])),
        cell_timeout_s=float(sandbox["cell_timeout_s"]),
        max_cells=int(sandbox["max_cells"]),
        memory_bytes=int(sandbox["memory_bytes"]),
        threads=int(sandbox["threads"]),
        max_output_chars=int(sandbox["max_output_chars"]),
    )


def _editor_name(raw: object) -> str:
    try:
        return patch_name(raw)
    except SnapshotProblem as problem:
        raise ToolError(problem.messages) from None


def _program_file(raw: object) -> str:
    try:
        return program_name(raw)
    except SnapshotProblem as problem:
        raise ToolError([message.replace("patch_file", "program_file") for message in problem.messages]) from None


class ToolServer:
    """The tools of one session. Not thread-safe by design: one caller thread handles one call at a time."""

    def __init__(self, session: Mapping[str, Any]) -> None:
        self.session = dict(session)
        self.paths = SessionPaths(Path(str(session["paths"]["root"])))
        self.tools = tuple(str(name) for name in session["tools"])
        unknown = [name for name in self.tools if name not in TOOL_DESCRIPTIONS]
        if unknown:
            raise SessionConfigError(f"session.json lists unknown tools {unknown}")
        self.checks = PATCH_CHECK_TOOL in self.tools
        self.program = arm_spec(str(session["arm"])).program
        self.blind = arm_spec(str(session["arm"])).blind  # LEARN_BLIND (module docstring)
        if self.blind != ("blind" in session):
            raise SessionConfigError("session.json has a blind section exactly when its arm is blind")
        self.schemas = tool_schemas(self.program)
        from weightcode.harness.e11 import NAMED_NOCHECK, NAMED_VARIANTS, check_variant

        self.e11_variant = check_variant(session.get("e11_variant"))  # the information ladder (module docstring)
        self.no_data = self.e11_variant in NAMED_VARIANTS
        self.hidden_check = self.e11_variant == NAMED_NOCHECK
        if self.no_data and (not self.program or self.blind or "e11_view" not in session):
            raise SessionConfigError("a no-data E11 session is a LEARN session with an e11_view section")
        if self.hidden_check and (self.checks or session["e11_view"].get("hidden_check") is None):
            raise SessionConfigError("a K0a session has no patch_check, and its e11_view names the hidden check")
        self.ledger = CheckLedger(session, self.paths)
        self.applier = self.ledger.applier
        self.loop_rule = loop.session_loop_rule(session)
        self.fscil_v2 = self.ledger.fscil_v2  # PREREG E9 v2: the kept rule fscil-retain-v2, or None
        self.shown_fields = (loop.shown_check_fields(self.ledger.n_prev) if self.loop_rule is not None
                             else fscil.V2_SHOWN_FIELDS if self.fscil_v2 is not None else PROGRAM_SHOWN_CHECK_FIELDS)
        self.evidence: dict[str, np.ndarray] | None = self._load_evidence() if PREDICT_TOOL in self.tools else None
        self.kernel = SandboxKernel(sandbox_config(session, self.paths))
        self._attempts = sum(1 for _ in self.paths.attempts.iterdir())
        self.wall_timeout_s = float(session["budget"]["wall_timeout_s"])
        try:  # written by the launcher before the editor, which starts this server, was started
            launch = read_launch_start(self.paths)
        except LaunchStartError as exc:
            raise SessionConfigError(str(exc)) from None
        if float(launch["wall_timeout_s"]) != self.wall_timeout_s:
            raise SessionConfigError("logs/launch_start.json states another wall-clock limit than session.json")
        self.launch_unix = float(launch["started_unix"])
        if self.program:
            self._load_student()

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the sandbox kernel; call it from the thread that serves requests (bwrap dies with that thread)."""
        self.kernel.start()

    def close(self) -> None:
        self.kernel.close()

    def tool_list(self) -> list[dict[str, Any]]:
        return tool_specs(self.tools, self.program, fewshot="fewshot" in self.session,
                          s_old_floor=(None if self.loop_rule is not None or self.fscil_v2 is not None
                                       else session_floor(self.session)),
                          loop_rule=self.loop_rule, seqskill="seqskill" in self.session,
                          graphnew="graphnew" in self.session, fscil="fscil" in self.session, fscil_v2=self.fscil_v2,
                          blind=self.blind, e11_variant=self.session.get("e11_variant"),
                          scan=self.session.get("seqskill", {}).get("rule") == SCAN_RULE,
                          policyskill="policyskill" in self.session)

    @property
    def queries_remaining(self) -> int:
        return self.ledger.queries_remaining

    def clock(self, now: float | None = None) -> dict[str, float]:
        """``wall_s_elapsed`` and ``wall_s_remaining``: seconds since the launcher started the editor
        (logs/launch_start.json) and seconds left of budget.wall_timeout_s (never below 0), to 0.1 s."""
        elapsed = max(0.0, (time.time() if now is None else now) - self.launch_unix)
        remaining = max(0.0, self.wall_timeout_s - elapsed)
        return {"wall_s_elapsed": round(elapsed, 1), "wall_s_remaining": round(remaining, 1)}

    # ------------------------------------------------------------------ dispatch

    def call(self, name: str, arguments: Mapping[str, Any] | None) -> ToolOutcome:
        """Run one tool. ``ToolError`` becomes an error result; any other exception is a harness fault. Every
        result, an error result too, ends with the session clock (``clock``), taken when the call returns."""
        outcome = self._call(name, arguments)
        outcome.payload.update(self.clock())
        return outcome

    def _call(self, name: str, arguments: Mapping[str, Any] | None) -> ToolOutcome:
        if name not in self.tools:
            raise UnknownTool(name)
        args = {} if arguments is None else arguments
        if not isinstance(args, Mapping):
            return ToolOutcome({"error": ["arguments must be a JSON object"]}, True)
        allowed = set(self.schemas[name]["properties"])
        extra = sorted(str(key) for key in set(args) - allowed)
        if extra:
            return ToolOutcome({"error": [f"unknown arguments {extra}; this tool takes {sorted(allowed)}"]}, True)
        handler = {
            "python": self._python,
            "patch_check": self._program_check if self.program else self._patch_check,
            "status": self._status,
            "declare_final": (self._nocheck_declare if self.hidden_check else self._program_declare if self.program
                              else self._declare_final),
            PREDICT_TOOL: self._predict,
        }[name]
        try:
            return handler(dict(args))
        except ToolError as problem:
            payload: dict[str, Any] = {"error": problem.messages}
            payload.update(problem.extra)
            return ToolOutcome(payload, True, problem.meta)

    # ------------------------------------------------------------------ helpers

    def _load_evidence(self) -> dict[str, np.ndarray]:
        path = Path(str(self.session["evidence_path"]))
        if file_sha256(path) != self.session["trial_files_sha256"].get("evidence.npz"):
            raise SessionConfigError("trial evidence.npz changed after assembly")
        with np.load(path, allow_pickle=False) as archive:
            evidence = {key: archive[key] for key in archive.files}
        if evidence_content_sha256(evidence) != self.session["evidence_content_sha256"]:
            raise SessionConfigError("trial evidence.npz content differs from private.json evidence_content_sha256")
        return evidence

    def _load_student(self) -> None:
        """The student and experience a program run is given: the trial's own files, hash-checked; in a blind arm
        the weights and the whole experience come from the session's private copies (session.json ``blind``)."""
        trial = self.paths.trial
        shown_files = ("model_shapes.json" if self.blind else "model.npz", "model_spec.json",
                       *(() if self.no_data else ("experience.npz",)), "forward.py")
        for name in shown_files:
            if file_sha256(trial / name) != self.session["trial_files_sha256"].get(name):
                raise SessionConfigError(f"trial {name} changed after assembly")
        params_file, self.experience_file = trial / "model.npz", trial / "experience.npz"
        if self.no_data:  # no trial experience: the private empty one (session.json e11_view)
            view = self.session["e11_view"]
            self.experience_file = Path(str(view["program_experience_path"]))
            if "experience.npz" in self.session["trial_files_sha256"] or file_sha256(self.experience_file) != view[
                    "program_experience_file_sha256"]:
                raise SessionConfigError("the no-data view's private experience changed after assembly, or the trial "
                                         "has an experience file")
        if self.blind:
            blind = self.session["blind"]
            params_file, self.experience_file = Path(str(blind["params_path"])), Path(str(blind["experience_path"]))
            for path, key in ((params_file, "params_file_sha256"), (self.experience_file, "experience_file_sha256")):
                if file_sha256(path) != blind[key]:
                    raise SessionConfigError(f"the blind view's private {path.name} changed after assembly")
            if self.applier.real_sha256 != blind["params_sha256"]:
                raise SessionConfigError("the blind view's private model.npz is not the real model")
        inputs = [str(name) for name in self.session["trial"]["input_files"]]
        self.student = Student(params=self.applier.real, params_file=params_file,
                               forward_file=trial / "forward.py", input_files={name: trial / name for name in inputs},
                               layout=read_json(trial / "model_spec.json"))
        self.program_sandbox = ProgramSandbox.from_config(self.kernel.config)
        self.program_wall_s = float(self.session["program"]["wall_s"])

    def _new_attempt(self) -> tuple[str, Path]:
        attempt_id = f"a{self._attempts:04d}"
        self._attempts += 1
        return attempt_id, self.paths.attempts / attempt_id

    def _snapshot_program(self, tool: str, name: str, nothing: str = "Nothing was run and no query was used."
                          ) -> tuple[str, Path, dict[str, Any]]:
        """Snapshot program ``name`` into a new attempt directory; a refused snapshot is recorded and raised, with
        ``nothing`` after its problems."""
        attempt_id, attempt_dir = self._new_attempt()
        record: dict[str, Any] = {"schema": ATTEMPT_SCHEMA, "attempt": attempt_id, "tool": tool, "program_file": name,
                                  "time": utc_now()}
        try:
            snapshot_file(self.paths.patches, name, attempt_dir, MAX_PROGRAM_BYTES, f"program file {name!r}")
        except SnapshotProblem as problem:
            record.update(outcome="invalid", stage="snapshot", messages=problem.messages,
                          editor_messages=problem.messages, redaction="snapshot")
            if not attempt_dir.is_dir():
                attempt_dir.mkdir(mode=0o700)
            write_json(attempt_dir / "attempt.json", record)
            raise ToolError(problem.messages + [nothing], program_file=name, meta={"attempt": attempt_id}) from None
        record["program_sha256"] = file_sha256(attempt_dir / name)
        return attempt_id, attempt_dir, record

    def _snapshot_and_apply(self, tool: str, name: str) -> tuple[str, Path, PatchResult, dict[str, Any]]:
        """Snapshot patch ``name`` into a new attempt directory and apply it under the arm rules."""
        attempt_id = f"a{self._attempts:04d}"
        self._attempts += 1
        attempt_dir = self.paths.attempts / attempt_id
        record: dict[str, Any] = {"schema": ATTEMPT_SCHEMA, "attempt": attempt_id, "tool": tool, "patch_file": name,
                                  "time": utc_now()}
        try:
            patch, copied = snapshot_patch(self.paths.patches, name, attempt_dir, self.applier.limits.max_file_bytes)
            record["files_sha256"] = {file: file_sha256(attempt_dir / file) for file in copied}
            result = self.applier.apply_for_editor(patch, attempt_dir)
        except (SnapshotProblem, PatchError) as problem:
            shown = list(problem.messages)  # SnapshotProblem: about the editor's own files, shown verbatim
            executor = problem.executor_messages if isinstance(problem, EditorRejection) else shown
            redaction = problem.redaction if isinstance(problem, EditorRejection) else "snapshot"
            record.update(outcome="invalid", messages=executor, editor_messages=shown, redaction=redaction)
            if not attempt_dir.is_dir():
                attempt_dir.mkdir(mode=0o700)
            write_json(attempt_dir / "attempt.json", record)
            raise ToolError(shown + ["The patch is invalid, so nothing was applied."], patch_file=name,
                            meta={"attempt": attempt_id, "executor_messages": executor, "redaction": redaction}
                            ) from None
        record.update(outcome="valid", child_sha256=result.child_sha256, executor=result.summary())
        write_json(attempt_dir / "attempt.json", record)
        return attempt_id, attempt_dir, result, record

    def _spent(self) -> ToolError:
        return ToolError(f"the patch-check budget of {self.ledger.budget} queries is spent; no query was used",
                         queries_remaining=0)

    # ------------------------------------------------------------------ tools

    def _python(self, args: dict[str, Any]) -> ToolOutcome:
        code = args.get("code")
        if not isinstance(code, str):
            raise ToolError("code must be a string")
        result = self.kernel.run(code)
        payload = {
            "cell_index": result.index,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
            "duration_s": round(result.duration_s, 3),
            "timed_out": result.timed_out,
            "truncated": result.truncated,
            "restarted": result.restarted,
            "refused": result.refused,
            "cells_remaining": self.kernel.cells_remaining,
        }
        return ToolOutcome(payload, result.error is not None or result.refused, {"cell_index": result.index})

    def _patch_check(self, args: dict[str, Any]) -> ToolOutcome:
        name = _editor_name(args.get("patch_file"))
        if self.ledger.queries_remaining <= 0:
            raise self._spent()
        attempt_id, _, result, record = self._snapshot_and_apply("patch_check", name)
        base_record = {"schema": CHECK_RECORD_SCHEMA, "time": utc_now(), "patch_file": name, "attempt": attempt_id,
                       "apply_mode": self.applier.apply_mode, "files_sha256": record.get("files_sha256", {})}
        try:
            entry = self.ledger.oracle.query(result.child, name)
        except BudgetExceeded:
            raise self._spent() from None
        except FloatingPointError:
            charge = dict(base_record, charged_failure=True, patch_id=name, reason=NON_FINITE_REASON,
                          child_sha256=result.child_sha256, executor=result.summary())
            append_jsonl(self.paths.checks_log, charge)
            self.ledger.record(charge)
            raise ToolError(f"{NON_FINITE_REASON}; this outcome depends on the check set, so it used one query",
                            queries_remaining=self.ledger.queries_remaining) from None
        except ValueError as problem:  # duplicate patch id or a child the oracle rejects: nothing consumed
            raise ToolError([str(problem), "No query was used."],
                            queries_remaining=self.ledger.queries_remaining) from None
        if entry["child_sha256"] != result.child_sha256:
            raise RuntimeError("the oracle hashed the child differently from the executor")
        logged = dict(base_record, **entry, executor=result.summary())
        append_jsonl(self.paths.checks_log, logged)
        self.ledger.record(logged)
        payload = {key: value for key, value in entry.items() if key not in EDITOR_HIDDEN_CHECK_FIELDS}
        payload["queries_remaining"] = self.ledger.queries_remaining
        payload["edit_size"] = result.edit_size
        payload["edit_frac"] = result.edit_frac
        return ToolOutcome(payload, False, {"attempt": attempt_id, "query_index": entry["query_index"],
                                            "child_sha256": entry["child_sha256"], "n_changed": entry["n_changed"]})

    def _run_and_apply(self, name: str, attempt_id: str, attempt_dir: Path, record: dict[str, Any], nothing: str,
                       **error_extra: Any) -> tuple[Any, PatchResult, dict[str, str]]:
        """Run the snapshot of program ``name`` once on the session's student, write its derived patch and apply it
        under the arm rules; the attempt record is completed as valid, or written as invalid and raised as a
        ``ToolError`` with ``nothing`` and ``error_extra`` after the problems."""

        def refuse(stage: str, shown: list[str], executor: list[str], redaction: str, run: Any) -> ToolError:
            record.update(outcome="invalid", stage=stage, messages=executor, editor_messages=shown,
                          redaction=redaction, run=run)
            write_json(attempt_dir / "attempt.json", record)
            extra = {} if not run else dict(run.get("shown", {}))
            return ToolError(shown + [nothing], program_file=name, **extra, **error_extra,
                             meta={"attempt": attempt_id, "executor_messages": executor, "redaction": redaction})

        try:
            update = run_update(self.student, self.experience_file, attempt_dir / name, attempt_dir / RUN_DIR,
                                self.program_sandbox, self.program_wall_s, blind=self.blind)
        except ProgramProblem as problem:
            raise refuse("program", problem.messages, problem.host_messages,
                         REDACTION_BLIND if self.blind else "verbatim", problem.run) from None
        shutil.copyfile(update.deltas_path, attempt_dir / DELTAS_FILE)
        patch = write_derived_patch(update, self.applier.real, self.applier.expected_parent_sha256, attempt_dir)
        write_json(attempt_dir / RUN_RECORD, update.record)
        try:
            result = self.applier.apply_for_editor(patch, attempt_dir)
        except PatchError as problem:
            executor = problem.executor_messages if isinstance(problem, EditorRejection) else list(problem.messages)
            redaction = problem.redaction if isinstance(problem, EditorRejection) else "verbatim"
            shown = list(problem.messages)
            if self.blind:  # kinds of failure only, never a tensor or a count (weightcode.harness.applying)
                shown, redaction = blind_rejection_messages(shown, self.applier.limits), REDACTION_BLIND
            raise refuse("limits", shown, executor, redaction, update.record) from None
        files = {item.name: file_sha256(item) for item in sorted(attempt_dir.iterdir()) if item.is_file()}
        record.update(outcome="valid", child_sha256=result.child_sha256, executor=result.summary(),
                      deltas_sha256=update.record["deltas_sha256"], tensors=update.tensors, files_sha256=files)
        write_json(attempt_dir / "attempt.json", record)
        return update, result, files

    def _program_check(self, args: dict[str, Any]) -> ToolOutcome:
        name = _program_file(args.get("program_file"))
        if self.ledger.queries_remaining <= 0:
            raise self._spent()
        attempt_id, attempt_dir, record = self._snapshot_program("patch_check", name)
        update, result, files = self._run_and_apply(name, attempt_id, attempt_dir, record,
                                                    "Nothing was applied and no query was used.",
                                                    queries_remaining=self.ledger.queries_remaining)
        base_record = {"schema": CHECK_RECORD_SCHEMA, "time": utc_now(), "patch_file": DERIVED_PATCH,
                       "program_file": name, "program_sha256": record["program_sha256"],
                       "deltas_sha256": record["deltas_sha256"], "attempt": attempt_id,
                       "apply_mode": self.applier.apply_mode, "files_sha256": files}
        try:
            entry = self.ledger.oracle.query(result.child, name)
        except BudgetExceeded:
            raise self._spent() from None
        except FloatingPointError:
            charge = dict(base_record, charged_failure=True, patch_id=name, reason=NON_FINITE_REASON,
                          child_sha256=result.child_sha256, executor=result.summary())
            append_jsonl(self.paths.checks_log, charge)
            self.ledger.record(charge)
            raise ToolError(f"{NON_FINITE_REASON}; this outcome depends on the check set, so it used one query",
                            queries_remaining=self.ledger.queries_remaining) from None
        except ValueError as problem:  # duplicate program name or a child the oracle rejects: nothing consumed
            raise ToolError([str(problem), "No query was used."],
                            queries_remaining=self.ledger.queries_remaining) from None
        if entry["child_sha256"] != result.child_sha256:
            raise RuntimeError("the oracle hashed the child differently from the executor")
        logged = dict(base_record, **entry, executor=result.summary())
        append_jsonl(self.paths.checks_log, logged)
        self.ledger.record(logged)
        payload: dict[str, Any] = {"program_file": name, "query_index": entry["query_index"],
                                   **{key: entry[key] for key in self.shown_fields},
                                   "queries_remaining": self.ledger.queries_remaining,
                                   **({} if self.blind else {"tensors": update.tensors}),
                                   **update.record["shown"]}
        return ToolOutcome(payload, False, {"attempt": attempt_id, "query_index": entry["query_index"],
                                            "child_sha256": entry["child_sha256"], "n_changed": entry["n_changed"]})

    def _nocheck_declare(self, args: dict[str, Any]) -> ToolOutcome:
        """K0a's ``declare_final`` (module docstring): run and validate the program, score nothing."""
        rationale = args.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ToolError("rationale must be a non-empty string")
        if len(rationale) > MAX_RATIONALE_CHARS:
            raise ToolError(f"rationale is longer than {MAX_RATIONALE_CHARS} characters")
        if "program_file" not in args:
            raise ToolError("program_file is required: a .py file name in /work/patches, or null for the original model")
        raw = args["program_file"]
        record: dict[str, Any] = {"schema": DECLARED_SCHEMA, "time": utc_now(), "rationale": rationale}
        messages: list[str] = []
        shown: dict[str, Any] = {}
        if raw is None:
            record.update(patch_file=None, program_file=None, kind="original", valid=True, checked=False,
                          query_index=None, attempt=None, child_sha256=self.applier.real_sha256)
        else:
            name = _program_file(raw)
            record.update(patch_file=name, program_file=name, kind="program", checked=False, query_index=None)
            try:
                attempt_id, attempt_dir, snapshot = self._snapshot_program("declare_final", name, NOCHECK_NOT_RUN)
                update, result, files = self._run_and_apply(name, attempt_id, attempt_dir, snapshot, NOCHECK_REFUSED)
            except ToolError as problem:
                messages = problem.messages
                record.update(valid=False, messages=messages, attempt=problem.meta.get("attempt"), child_sha256=None,
                              executor_messages=problem.meta.get("executor_messages"))
                shown = {key: value for key, value in problem.extra.items() if key != "program_file"}
            else:
                record.update(valid=True, attempt=attempt_id, program_sha256=snapshot["program_sha256"],
                              deltas_sha256=update.record["deltas_sha256"], child_sha256=result.child_sha256,
                              tensors=update.tensors, files_sha256=files)
                shown = {"tensors": update.tensors, **update.record["shown"]}
        append_jsonl(self.paths.declared_log, record)
        payload: dict[str, Any] = {"recorded": True, "program_file": record["program_file"], "valid": record["valid"],
                                   **shown, "note": NOCHECK_DECLARE_NOTE}
        if messages:
            payload["error"] = messages
        return ToolOutcome(payload, not record["valid"], {"attempt": record.get("attempt")})

    def _program_declare(self, args: dict[str, Any]) -> ToolOutcome:
        rationale = args.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ToolError("rationale must be a non-empty string")
        if len(rationale) > MAX_RATIONALE_CHARS:
            raise ToolError(f"rationale is longer than {MAX_RATIONALE_CHARS} characters")
        if "program_file" not in args:
            raise ToolError("program_file is required: a .py file name in /work/patches, or null for the original model")
        raw = args["program_file"]
        record: dict[str, Any] = {"schema": DECLARED_SCHEMA, "time": utc_now(), "rationale": rationale}
        messages: list[str] = []
        if raw is None:
            record.update(patch_file=None, program_file=None, kind="original", valid=True, checked=False,
                          query_index=None, attempt=None, child_sha256=self.applier.real_sha256)
        else:
            name = _program_file(raw)
            checked = [entry for entry in self.ledger.checked() if entry["patch_id"] == name]
            record.update(patch_file=name, program_file=name, kind="program", checked=bool(checked),
                          query_index=checked[0]["query_index"] if checked else None)
            try:
                attempt_id, _, snapshot = self._snapshot_program("declare_final", name)
            except ToolError as problem:
                messages = problem.messages
                record.update(valid=False, messages=messages, attempt=problem.meta.get("attempt"), child_sha256=None)
            else:
                write_json(self.paths.attempts / attempt_id / "attempt.json", dict(snapshot, outcome="snapshot"))
                same = bool(checked) and snapshot["program_sha256"] == checked[0]["program_sha256"]
                record.update(valid=True, attempt=attempt_id, program_sha256=snapshot["program_sha256"],
                              child_sha256=checked[0]["child_sha256"] if same else None)
            if checked:
                record["same_as_checked"] = record.get("program_sha256") == checked[0]["program_sha256"]
        append_jsonl(self.paths.declared_log, record)
        payload: dict[str, Any] = {"recorded": True, "program_file": record["program_file"], "valid": record["valid"],
                                   "checked": record["checked"],
                                   "note": "recorded as your own choice; the kept update is still chosen by the "
                                           "final-selection rule"}
        if messages:
            payload["error"] = messages
        return ToolOutcome(payload, not record["valid"], {"attempt": record.get("attempt")})

    def _kept_patch_file(self) -> str | None:
        kept = last_valid_declaration(read_jsonl(self.paths.declared_log))
        return None if kept is None else kept["patch_file"]

    def _status(self, args: dict[str, Any]) -> ToolOutcome:
        declared = read_jsonl(self.paths.declared_log)
        last = declared[-1] if declared else None
        last_shown = None if last is None else {"patch_file": last["patch_file"], "valid": last["valid"]}
        if self.hidden_check:  # K0a: no query and no score (module docstring)
            valid = last_valid_declaration(declared)
            return ToolOutcome({"cells_remaining": self.kernel.cells_remaining,
                                "cells_budget": self.kernel.config.max_cells,
                                "declared_final": None if last is None else {"program_file": last["program_file"],
                                                                             "valid": last["valid"]},
                                "last_valid_program_file": None if valid is None else valid["program_file"]}, False)
        if not self.checks:
            return ToolOutcome({"cells_remaining": self.kernel.cells_remaining,
                                "cells_budget": self.kernel.config.max_cells, "declared_final": last_shown,
                                "kept_patch_file": self._kept_patch_file()}, False)
        if self.program:
            checked = [{"program_file": entry["patch_id"], "query_index": entry["query_index"],
                        **{key: entry[key] for key in self.shown_fields}} for entry in self.ledger.checked()]
            if last is not None:
                last_shown = {"program_file": last["program_file"], "valid": last["valid"]}
        else:
            checked = [{"patch_file": entry["patch_id"], "query_index": entry["query_index"], "S": entry["score"]}
                       for entry in self.ledger.checked()]
        payload = {
            "queries_remaining": self.ledger.queries_remaining,
            "queries_budget": self.ledger.budget,
            "cells_remaining": self.kernel.cells_remaining,
            "cells_budget": self.kernel.config.max_cells,
            "checked": checked,
            "charged_failures": [{"program_file" if self.program else "patch_file": entry["patch_id"],
                                  "reason": entry["reason"]} for entry in self.ledger.charged],
            "declared_final": last_shown,
        }
        return ToolOutcome(payload, False)

    def _declare_final(self, args: dict[str, Any]) -> ToolOutcome:
        rationale = args.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ToolError("rationale must be a non-empty string")
        if len(rationale) > MAX_RATIONALE_CHARS:
            raise ToolError(f"rationale is longer than {MAX_RATIONALE_CHARS} characters")
        if "patch_file" not in args:
            raise ToolError("patch_file is required: a file name in /work/patches, or null for the original model")
        raw = args["patch_file"]
        record: dict[str, Any] = {"schema": DECLARED_SCHEMA, "time": utc_now(), "rationale": rationale}
        messages: list[str] = []
        if raw is None:
            record.update(patch_file=None, kind="original", valid=True, checked=False, query_index=None,
                          attempt=None, child_sha256=self.applier.real_sha256)
        else:
            name = _editor_name(raw)
            checked = [entry for entry in self.ledger.checked() if entry["patch_id"] == name]
            record.update(patch_file=name, kind="patch", checked=bool(checked),
                          query_index=checked[0]["query_index"] if checked else None)
            try:
                attempt_id, _, result, _ = self._snapshot_and_apply("declare_final", name)
            except ToolError as problem:
                messages = problem.messages
                record.update(valid=False, messages=messages, attempt=problem.meta.get("attempt"), child_sha256=None,
                              executor_messages=problem.meta.get("executor_messages"))
            else:
                record.update(valid=True, attempt=attempt_id, child_sha256=result.child_sha256)
            if checked:
                record["same_as_checked"] = record["child_sha256"] == checked[0]["child_sha256"]
        append_jsonl(self.paths.declared_log, record)
        payload: dict[str, Any] = {"recorded": True, "patch_file": record["patch_file"], "valid": record["valid"]}
        if self.checks:
            payload.update(checked=record["checked"],
                           note="recorded as your own choice; the kept patch is still chosen by the final-selection "
                                "rule")
        else:
            payload.update(note=NO_CHECK_NOTE, kept_patch_file=self._kept_patch_file())
        if messages:
            payload["error"] = messages
        return ToolOutcome(payload, not record["valid"], {"attempt": record.get("attempt")})

    def _predict(self, args: dict[str, Any]) -> ToolOutcome:
        if self.evidence is None:
            raise RuntimeError("predict is listed for this session but its evidence was not loaded")
        subset = args.get("subset")
        if subset not in PREDICT_SUBSETS:
            raise ToolError(f"subset must be one of {list(PREDICT_SUBSETS)}")
        raw = args.get("patch_file")
        if raw is None:
            params, child_sha, attempt_id, shown = self.applier.real, self.applier.real_sha256, None, None
        else:
            shown = _editor_name(raw)
            attempt_id, _, result, _ = self._snapshot_and_apply("predict", shown)
            params, child_sha = result.child, result.child_sha256
        parts = PREDICT_PARTS[str(subset)]
        x = np.concatenate([self.evidence[key] for key in parts], axis=0)
        logits = np.asarray(self.ledger.forward(dict(params), x))
        if logits.ndim != 2 or logits.shape[0] != len(x):
            raise RuntimeError(f"forward returned logits of shape {logits.shape}")
        if not np.all(np.isfinite(logits)):
            raise ToolError("the model produces non-finite logits on these examples")
        payload = {
            "subset": subset,
            "patch_file": shown,
            "n": int(len(x)),
            "order": " then ".join(parts) + ", in evidence.npz order",
            "logits": np.round(logits.astype(np.float64), LOGIT_DECIMALS).tolist(),
        }
        return ToolOutcome(payload, False, {"attempt": attempt_id, "child_sha256": child_sha})


__all__ = ["NAMED_CHECK_WORDS", "NOCHECK_DECLARE_NOTE", "NOCHECK_NOT_RUN", "NOCHECK_PROGRAM_DESCRIPTIONS",
           "NOCHECK_REFUSED", "named_check_description", "BLIND_RETURNS", "EDITOR_HIDDEN_CHECK_FIELDS",
           "FEWSHOT_CHECK_SETS", "FSCIL_CHECK_SETS",
           "GRAPHNEW_CHECK_SETS", "INPUT_SCHEMAS", "PROGRAM_RETURNS", "blind_check_description",
           "NO_CHECK_DESCRIPTIONS", "PROGRAM_CHECK_SETS", "PROGRAM_DESCRIPTIONS", "SCAN_CHECK_SETS", "SEQSKILL_CHECK_SETS",
           "fewshot_check_description", "fscil_check_description", "graphnew_check_description",
           "policyskill_check_description", "seqskill_check_description",
           "PROGRAM_SCHEMAS", "PROGRAM_SHOWN_CHECK_FIELDS", "TOOL_DESCRIPTIONS", "ToolError", "ToolOutcome",
           "ToolServer", "UnknownTool", "last_valid_declaration", "sandbox_config", "tool_schemas", "tool_specs"]
