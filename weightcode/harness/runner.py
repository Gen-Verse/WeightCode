"""One editor session from instance id to records (HARNESS_SPEC sections 5 and 10).

``run_session`` checks the instance before anything is written (evidence-v1 only; not superseded, listed active and
current under ``weightcode.evalkit.instances.check_task_current``; main instances only with ``allow_main``), then draws an
opaque session id, assembles the arm's trial view, renders the prompts, scans every editor-visible file, both
prompts and the tool descriptions for forbidden strings, writes ``session.json`` and ``mcp.json``, runs the sandbox
isolation self-test with the instance's forbidden strings, launches the editor (``claude -p`` or a scripted
stand-in), and finalises (``weightcode.harness.finalize``). A leak or a failed self-test aborts before the launch and is
recorded; in an arm without labels (M2NL) a label array or label key anywhere in the trial view is such a leak
(``weightcode.harness.leaks.scan_label_arrays``). An arm without ``patch_check`` gets a patch-check budget of zero. A program
arm (LEARN) is refused before any write unless its replay rows and the clean companion of V_check exist
(``weightcode.harness.clean.preflight``), and its MCP tool timeout must exceed the program wall limit. A fewshot-v1 instance
(PREREG E2, ``weightcode.harness.fewshot``) runs in a program arm only, with the fewshot LEARN system prompt and
``patch_check`` description, its experience from its own evidence file and V_check cut by class. A seqskill-v1
instance (PREREG E5, ``weightcode.harness.seqskill``) likewise, with the seqskill LEARN prompts and V_check cut by part; its
variant rule_only (PREREG E8, ``seqskill_variant``) gives the new operator's rule in model_spec.json and no row of it
in the experience, with the rule_only prompts and leak words, and is otherwise the same session. A seqskill-v1 LEARN
session may run with the per-tensor rel-Frob limit lifted (``lift_rel_frob``, the PREREG E8 amendment). A graphnew-v1
instance (PREREG E7, ``weightcode.harness.graphnew``) likewise, with the graphnew LEARN prompts, V_check cut by class,
graph.npz as the trial's one input file, the graphnew leak words in the text scan (``graphnew_scan``) and the kept rule
joint-retain-v1 with its floor given in rows (``retention_floor_rows``; a dry run may mark a stand-in floor as a
placeholder in its floor source, which no real session accepts). An fscil-v1 instance (PREREG E9,
``weightcode.harness.fscil``) likewise, with the fscil LEARN prompts, the check cut by session under the session mask, the
fscil leak words and the instance chain's ids in the forbidden strings, the kept rule joint-retain-v1 with the session's
floor in rows (or, with ``fscil_version`` "v2", PREREG E9 v2's kept rule fscil-retain-v2 with no floor given), and,
given ``library_sessions``, the lineage's kept programs of earlier sessions as its library. A policyskill-v1
instance (PREREG E14, ``weightcode.harness.policyskill``) likewise, with the policyskill LEARN prompts, V_check as
environment seeds scored by greedy rollouts, its recorded host paths forbidden, the E14 leak tokens in the text scan
only, E10's MCP floors and the kept rule joint-retain-v1 with its floor given in rows. The LEARN_BLIND arm
(PREREG A4a, ``weightcode.harness.arms``) runs on an evidence-v1 instance's own student only, with no loop context, library
or prompt override; ``max_checks`` (PREREG A4b) lowers the patch-check budget of any arm with ``patch_check``. The E11
rule view (PREREG K1, ``e11_variant="rule"``) runs in the LEARN arm on an E11 instance whose condition has a stated
rule, with no loop context, library or prompt override: its trial shows the clean replay rows alone and states the
condition's rule (``weightcode.harness.trial.assemble``), and its leak words, checks, kept rule and limits are E11's. The
information ladder's views (PREREG; ``e11_variant`` "unlabelled", "named", "named_nocheck"; ``weightcode.harness.e11``) run
in the LEARN arm on the one ladder instance only, with no loop context, library, prompt override or ``max_checks``:
U is E11's session without y_new; K0b gives no data and names the capability, whose sentence is the one text exempt
from the leak words and hint words (``ForbiddenStrings.sentence_exempt``; it must occur exactly once in task.json and
once in the task prompt, and nowhere else); K0a is K0b without ``patch_check``, under a retention floor, with a
budget of one query that only finalisation spends (``weightcode.harness.finalize``). A dry
run stops after the self-test, under ``$WEIGHTCODE_RUNS/dry_runs``, and writes no run.json and no manifest line. The runner
never reads ``$WEIGHTCODE_SEALED`` and computes no sealed readout.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import load_params
from weightcode.common.paths import PRIVATE_VAR, SEALED_VAR, private_dir, runs_dir, sealed_dir
from weightcode.data.conditions import CONDITIONS
from weightcode.evalkit.instances import scan_for_tokens
from weightcode.harness import clean, e10, e11, e12, fewshot, fscil, graphnew, loop, policyskill, seqskill, t04
from weightcode.harness.applying import ArmApplier
from weightcode.harness.arms import MCP_SERVER_NAME, PATCH_CHECK_TOOL, SHOWN_DECOY, arm_spec, patch_check_budget
from weightcode.harness.experience import EXPERIENCE_KEYS
from weightcode.harness.finalize import (
    STATUS_ABORTED_LEAK,
    STATUS_ABORTED_SELFTEST,
    STATUS_DRY_RUN,
    aborted_result,
    base_result,
    finalize_session,
)
from weightcode.harness.instance import InstanceRecord, InstanceRefused, read_instance_record
from weightcode.harness.jsonio import file_sha256, text_sha256, utc_now, write_json
from weightcode.harness.launch import ClaudeLauncher, LaunchOutcome, Launcher, editor_environment
from weightcode.harness.leaks import (
    ForbiddenStrings,
    Scanner,
    forbidden_strings,
    scan_hint_words,
    exempt_counts,
    scan_label_arrays,
    scan_texts,
    scan_trial,
    trial_texts,
)
from weightcode.harness.library import (
    LIBRARY_INDEX_FILE,
    LIBRARY_PROGRAM_TEMPLATE,
    LibraryError,
    library_entries,
    write_library,
)
from weightcode.harness.program import MAX_PROGRAM_BYTES
from weightcode.harness.prompts import Budgets, fscil_library_files_section, library_files_section, render_system, render_task
from weightcode.harness.retention import (
    RetentionRuleError,
    check_floor_rows,
    is_placeholder,
    kept_rule_record,
    kept_rule_record_rows,
)
from weightcode.harness.provenance import REPO_ROOT, claude_version, code_changed, collect
from weightcode.harness.session import SESSION_SCHEMA, SessionPaths, new_session_id
from weightcode.harness.tools import sandbox_config, tool_specs
from weightcode.harness.trial import StudentOverride, TrialError, assemble, check_max_checks, decoy_evidence_source
from weightcode.sandbox import isolation_selftest
from weightcode.sandbox.program import PROGRAM_CLOCK, PROGRAM_WALL_S, driver_sha256

DEFAULT_MODEL = "claude-opus-5-5"  # the PREREG editor model
DEFAULT_THREADS = 4
DEFAULT_CELL_TIMEOUT_S = 120.0
DEFAULT_MAX_CELLS = 150
DEFAULT_MEMORY_BYTES = 8 * 2**30
DEFAULT_MAX_OUTPUT_CHARS = 30_000
DEFAULT_MCP_TOOL_TIMEOUT_MS = 600_000
DEFAULT_MCP_CONNECT_TIMEOUT_MS = 120_000
PROGRAM_TOOL_MARGIN_S = 60.0  # a program check runs the program, then applies and scores its update
SYSTEM_PROMPT_MODES = ("replace", "append")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
SESSIONS_DIR = "sessions"
DRY_RUNS_DIR = "dry_runs"
SERVER_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SANDBOX_PYTHON = REPO_ROOT / ".venv-sandbox" / "bin" / "python"


@dataclass(frozen=True)
class SessionResult:
    """What ``run_session`` returns: the session id, its status, its directory and session_result.json."""

    session_id: str
    status: str
    root: Path
    record: dict[str, Any]


def session_forbidden_strings(record: InstanceRecord, extra_paths: tuple[str, ...] = (),
                              seqskill_variant: str | None = None,
                              extra_tokens: Sequence[str] = ()) -> ForbiddenStrings:
    """The instance's forbidden strings plus its host paths, the private and sealed roots and every condition name.

    A fewshot-v1 instance has no decoy; its recorded host paths (V_check, V_select, V_evidence, the replay and the
    student's source file, ``weightcode.harness.fewshot.private_paths``) are forbidden instead. A seqskill-v1 instance
    likewise, and its condition's leak words too (the dataset tokens, the condition name, its rule kind and the
    words that would name that kind, ``weightcode.harness.seqskill.leak_words``); its rule_only ``seqskill_variant`` keeps
    the condition name and drops the words that name the kind of rule, which the trial now gives. A graphnew-v1
    instance forbids its recorded host paths (``weightcode.harness.graphnew.private_paths``) and adds the graphnew leak words
    (``weightcode.harness.graphnew.leak_words``: E6's list and the 40 category codes) to the text scan only, which matches them
    with ``graphnew_scan``; the self-test, a substring match over mounts, environment and listing, keeps the base list,
    since a year such as "2018" can be part of a mount id there. An fscil-v1 instance forbids its recorded host paths
    (``weightcode.harness.fscil.private_paths``), names its store's base student as the model id, forbids ``extra_tokens``
    (the ids of its instance chain, its base model and its library's sources, ``weightcode.harness.fscil.session_extra_tokens``)
    like a condition name, and adds the fscil leak tokens (``weightcode.harness.fscil.leak_words``: the dataset's tokens and
    words and the store roots) to the text scan only, since the runs root is a prefix of the session's own mounts.
    An E10 instance (PREREG E10, ``weightcode.harness.e10``) forbids ``extra_tokens`` (``e10.self_test_tokens``: the E10
    condition and corruption names with an underscore, every E10 instance id and the library's source sessions) like a
    condition name, and adds the E10 leak words (``e10.leak_words``: every E10 corruption's condition leak tokens, the
    family names and the benchmark's names) to the text scan only. The program files of an E10 loop round's library
    are scanned, and withheld, on the list without those words (``ForbiddenStrings.library_text``, E4-loop's
    standard; ``weightcode.harness.leaks``). An E11 instance (PREREG E11, ``weightcode.harness.e11``: E10's conditions with a ViT
    student) is treated exactly so, with ``e11.self_test_tokens`` (which add the E11 instance ids) and
    ``e11.leak_words`` (E10's). So is an E12 instance (PREREG E12, ``weightcode.harness.e12``: E10's conditions with an
    ImageNet-pretrained ResNet-18), with ``e12.self_test_tokens`` and ``e12.leak_words`` (E10's). So is a T04
    instance (PREREG T04, ``weightcode.harness.t04``: ImageNet-C conditions on the native ImageNet ResNet-18), with
    ``t04.self_test_tokens`` and ``t04.leak_words`` (its own: E10's words "snow" and "digital" occur in ImageNet class
    names). A policyskill-v1
    instance (PREREG E14) forbids its recorded host paths (``weightcode.harness.policyskill.private_paths``) and adds the
    E14 leak tokens (``weightcode.harness.policyskill.leak_words``: the dataset tokens and the level, environment and
    benchmark names) to the text scan only, since short tokens such as "seed" can occur in the environment.
    """
    is_e11 = e11.is_e11(record.private)
    is_e12 = e12.is_e12(record.private)
    is_t04 = t04.is_t04(record.private)
    # E11 and E12 share E10's conditions, leak words and library standard; T04 shares the standard with its own words
    is_e10 = e10.is_e10(record.private) or is_e11 or is_e12 or is_t04
    if seqskill.check_variant(seqskill_variant) != seqskill.STANDARD and not record.is_seqskill:
        raise InstanceRefused(f"instance {record.instance_id}: a seqskill variant applies to a seqskill-v1 instance")
    if extra_tokens and not (record.is_fscil or is_e10):
        raise InstanceRefused(f"instance {record.instance_id}: extra forbidden ids apply to an fscil-v1 or an E10 "
                              "instance, or an E11, E12 or T04 one")
    paths = [str(record.private["vcheck_path"]), str(record.instance_dir / "private.json"),
             str(record.instance_dir), *extra_paths]
    conditions = list(CONDITIONS)
    private: Mapping[str, Any] = record.private
    if record.is_fewshot:
        paths += fewshot.private_paths(record.private)
    elif record.is_seqskill:
        paths += seqskill.private_paths(record.private)
        conditions += seqskill.leak_words(record.private, seqskill_variant)
    elif record.is_graphnew:
        paths += graphnew.private_paths(record.private)
    elif record.is_policyskill:
        paths += policyskill.private_paths(record.private)
    elif record.is_fscil:
        paths += fscil.private_paths(record.private)
        conditions += [str(token) for token in extra_tokens]
        private = {**record.private, "model_id": record.private["base_model_id"]}  # the store's base student
    else:
        paths.append(str(record.private["decoy_model_path"]))
        conditions += [str(token) for token in extra_tokens]  # empty unless E10 or E11
    for variable, resolve in ((PRIVATE_VAR, private_dir), (SEALED_VAR, sealed_dir)):
        paths += [os.path.abspath(os.path.expanduser(os.environ[variable])), str(resolve())]
    forbidden = forbidden_strings(private, record.instance_id, paths=paths, conditions=conditions)
    if record.is_graphnew:
        forbidden = dataclasses.replace(forbidden, text=tuple(sorted({*forbidden.text, *graphnew.leak_words()})))
    if record.is_policyskill:
        forbidden = dataclasses.replace(forbidden, text=tuple(sorted({*forbidden.text, *policyskill.leak_words()})))
    if record.is_fscil:
        forbidden = dataclasses.replace(forbidden, text=tuple(sorted({*forbidden.text,
                                                                      *fscil.leak_words(extra_tokens)})))
    if is_e10:  # the library programs keep E4-loop's standard: the list before the E10 leak words
        words = (e11.leak_words() if is_e11 else e12.leak_words() if is_e12 else t04.leak_words() if is_t04 else
                 e10.leak_words())
        forbidden = dataclasses.replace(forbidden, text=tuple(sorted({*forbidden.text, *words})),
                                        library_text=forbidden.text)
    return forbidden


SENTENCE_EXEMPT_PLACES = ("trial/task.json", "prompts/task.md")
"""Where an exempt sentence (K0a, K0b) must occur, once each; it must occur in no other scanned text."""


def leak_scan(paths: SessionPaths, prompts: Mapping[str, str], tools: list[dict[str, Any]],
              forbidden: ForbiddenStrings, *, labels: bool = True, scanner: Scanner = scan_for_tokens) -> dict[str, Any]:
    """Forbidden strings in the trial files, the prompts and the tool descriptions (matched by ``scanner``,
    ``graphnew_scan`` for a graphnew-v1 session; a round library's program files on ``forbidden.library_text`` when it
    is set, E10); hint words in the last two; and, when the arm shows no labels, every label array or label key in the
    trial files (``label_arrays``, else None). ``forbidden.sentence_exempt`` (K0a, K0b) is removed from every text
    before both scans and must occur exactly once in each of ``SENTENCE_EXEMPT_PLACES`` and in no other text
    (``sentence_placed``), or the scan fails."""
    texts = dict(prompts)
    texts["tools"] = json.dumps(tools, sort_keys=True)
    exempt = forbidden.sentence_exempt
    hits = scan_trial(paths.trial, forbidden.text, scanner, library_tokens=forbidden.library_text,
                      sentence_exempt=exempt)
    hits.update(scan_texts(texts, forbidden.text, scanner, sentence_exempt=exempt))
    hints = scan_hint_words(texts, sentence_exempt=exempt)
    label_arrays = None if labels else scan_label_arrays(paths.trial)
    placed: dict[str, Any] = {}
    if exempt:  # each exempt sentence once in trial/task.json and once in the task prompt, nowhere else
        counts = exempt_counts({**trial_texts(paths.trial), **texts}, exempt)
        expected = {name: {sentence: int(name in SENTENCE_EXEMPT_PLACES) for sentence in exempt} for name in counts}
        placed = {"sentence_exempt": list(exempt), "sentence_counts": counts,
                  "sentence_placed": counts == expected}
    return {"hits": hits, "hint_words": hints, "label_arrays": label_arrays,
            "passed": not hits and not hints and not label_arrays and placed.get("sentence_placed", True),
            "tokens": list(forbidden.text), "text_exempt": list(forbidden.text_exempt),
            **({} if forbidden.library_text is None else {"library_tokens": list(forbidden.library_text)}),
            **placed}


def _check_arguments(arm: str, repeat: int, model: str, effort: str, wall_timeout_s: float, threads: int,
                     system_prompt_mode: str, mcp_tool_timeout_ms: int, cell_timeout_s: float) -> None:
    arm_spec(arm)
    if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 0:
        raise ValueError(f"repeat must be a non-negative int, got {repeat!r}")
    if not model:
        raise ValueError("model must be named")
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
    if not wall_timeout_s > 0:
        raise ValueError("wall_timeout_s must be positive")
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive int")
    if system_prompt_mode not in SYSTEM_PROMPT_MODES:
        raise ValueError(f"system_prompt_mode must be one of {SYSTEM_PROMPT_MODES}")
    if mcp_tool_timeout_ms <= 1000 * cell_timeout_s:
        raise ValueError("mcp_tool_timeout_ms must exceed the cell timeout, or a long cell would be cut by Claude Code")
    if arm_spec(arm).program and mcp_tool_timeout_ms <= 1000 * (PROGRAM_WALL_S + PROGRAM_TOOL_MARGIN_S):
        raise ValueError(f"mcp_tool_timeout_ms must exceed the program wall limit of {PROGRAM_WALL_S:g} s plus "
                         f"{PROGRAM_TOOL_MARGIN_S:g} s, or a program check would be cut by Claude Code")


def mcp_config(session_json: Path, threads: int) -> dict[str, Any]:
    """mcp.json: the session's one stdio server, with the BLAS thread variables in its environment."""
    value = str(threads)
    return {"mcpServers": {MCP_SERVER_NAME: {
        "type": "stdio",
        "command": str(SERVER_PYTHON),
        "args": ["-m", "weightcode.harness.mcp_server", "--session", str(session_json)],
        "env": {"OMP_NUM_THREADS": value, "OPENBLAS_NUM_THREADS": value, "MKL_NUM_THREADS": value},
    }}}


def run_session(
    instance_id: str,
    arm: str,
    repeat: int,
    *,
    model: str = DEFAULT_MODEL,
    effort: str,
    wall_timeout_s: float,
    allow_main: bool = False,
    threads: int = DEFAULT_THREADS,
    cell_timeout_s: float = DEFAULT_CELL_TIMEOUT_S,
    max_cells: int = DEFAULT_MAX_CELLS,
    memory_bytes: int = DEFAULT_MEMORY_BYTES,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    mcp_tool_timeout_ms: int | None = None,
    mcp_connect_timeout_ms: int | None = None,
    system_prompt_mode: str = "replace",
    task_prompt_override: str | None = None,
    launcher: Launcher | None = None,
    claude_bin: str | None = None,
    skip_audit: bool = False,
    dry_run: bool = False,
    sandbox_python: Path = SANDBOX_PYTHON,
    retention_floor_frac: float | None = None,
    library_sessions: Sequence[str] | None = None,
    loop_context: Path | None = None,
    seqskill_variant: str | None = None,
    lift_rel_frob: bool = False,
    retention_floor_rows: int | None = None,
    retention_floor_source: Mapping[str, Any] | None = None,
    fscil_version: str | None = None,
    max_checks: int | None = None,
    blind_predictions: bool = False,
    e11_variant: str | None = None,
    api_key_file: Path | None = None,
    oauth_token_file: Path | None = None,
) -> SessionResult:
    """Run one editor session (see the module docstring); every refusal of the instance comes before any write.

    ``loop_context`` (a program arm on an evidence-v1 dev instance, PREREG E4-loop, ``weightcode.harness.loop``) runs the
    session as round r of a loop stream: the stream's current student replaces the instance's base model (model.npz,
    the oracle base and the final's parent), the experience is drawn on that student and holds the earlier rounds'
    rows, the trial holds the round library, the check adds one part per earlier condition and the kept rule is
    loop-retain-v1, or loop-retain-v2 when the context states the stream's clean budget (``loop.context_kept_rule``).
    It excludes ``retention_floor_frac``, ``library_sessions`` and a prompt override. A context with an edit log
    (PREREG T16 arm L, ``weightcode.harness.loop``) adds the trial file edit_log.json and one system-prompt sentence, and
    nothing else; session.json ``loop.edit_log`` records the arm either way. A context that also states
    ``protect_logged`` (T16 follow-up M) adds the protected-entry limit, applied with the session's other limits
    (``weightcode.harness.applying``), and one more sentence after the edit log's; session.json ``loop.protect_logged``
    records it, and a session without the limit has no such key.

    ``retention_floor_frac`` (a program arm, dev only) runs the session under the retention kept rule
    (``weightcode.harness.retention``): the floor is that fraction of the V_check rows behind score_old, it is stated in the
    system prompt and in ``declare_final``, recorded as session.json ``kept_rule`` and applied by the final-patch rule.
    None is the v1 rule, and then nothing in the session differs from a session without the argument.

    ``library_sessions`` (a program arm on an evidence-v1 instance, dev only, LEARN v3) puts the kept programs of
    those earlier sessions and their check scores into the trial (``weightcode.harness.library``), adds one bullet for them
    to the task prompt's file list and records session.json ``library``. None leaves the session as without it.

    ``seqskill_variant`` (a seqskill-v1 instance only): None or "standard" is the PREREG E5 session, unchanged, and
    "rule_only" the PREREG E8 one (``weightcode.harness.trial.assemble``, the rule_only prompts and leak words). The
    variant is recorded as session.json ``seqskill.variant``; a standard session has no such key.

    ``lift_rel_frob`` (a seqskill-v1 instance in the LEARN arm only; PREREG E8 amendment of 2026-11-04 15:37) runs the
    session with ``max_rel_frob`` None (``weightcode.harness.trial.lifted_limits``): in the trial task.json limits and the
    task prompt rendered from them, in session.json ``limits`` (which every check, declaration, final and readout
    replay applies) and as the session.json flag ``lift_rel_frob``. False leaves the session as without it, with no
    such key. Anything else is refused before any write.

    ``retention_floor_rows`` (a program arm, an int <= 0; required on a graphnew-v1 instance, PREREG E7) is the same
    retention kept rule with its floor on score_old given as a row count of the check set behind score_old, computed
    outside the harness (for E7 from the PREREG formula) and recorded as session.json ``kept_rule`` with
    ``floor_kind: "rows"`` (``weightcode.harness.retention.kept_rule_record_rows``); it excludes ``retention_floor_frac``.
    ``retention_floor_source`` (a JSON object, with ``retention_floor_rows`` only) says where the number comes from
    and is copied as ``kept_rule.floor_source``; one with ``"placeholder": true`` marks a stand-in floor and is
    accepted in a dry run only.

    On an fscil-v1 instance (PREREG E9) ``retention_floor_rows`` is required (the floor of the instance's session) and
    ``library_sessions`` names the lineage's earlier sessions whose kept programs form the library
    (``weightcode.harness.fscil.library_entries``: sessions on the instance's chain that kept a checked program); the
    library's bullet is worded for the earlier sessions of this classifier.

    ``fscil_version`` (an fscil-v1 instance only): None or "v1" is that PREREG E9 session, unchanged; "v2" is the
    PREREG E9 v2 session, which takes no retention floor and runs under kept rule fscil-retain-v2 on check score
    fscil-cb-v1 (``weightcode.harness.fscil.v2_rule_for``: R = -1.0, the store's base-row counts), recorded as session.json
    ``kept_rule``, stated in the system prompt, the task.json readout and final selection and the tool descriptions,
    reported by every check and applied by the final-patch rule; its library sources are v2 sessions.

    ``mcp_tool_timeout_ms`` and ``mcp_connect_timeout_ms`` (Claude Code's MCP_TOOL_TIMEOUT and MCP_TIMEOUT): None is
    the instance's default, ``DEFAULT_MCP_TOOL_TIMEOUT_MS`` and ``DEFAULT_MCP_CONNECT_TIMEOUT_MS``, except on an E10
    instance (PREREG E10), whose ResNet-20 check predictions need ``weightcode.harness.e10.MCP_TOOL_TIMEOUT_MS`` and
    ``MCP_CONNECT_TIMEOUT_MS`` and which refuses smaller values, and an E11 instance (PREREG E11, a ViT-Lite student
    on E10's data), an E12 instance (PREREG E12, a pretrained ResNet-18 on E10's data), a T04 instance (PREREG T04,
    the native ImageNet ResNet-18) and a policyskill-v1 instance (PREREG E14, rollout checks), which have the same
    floors. The values used are recorded in session.json ``mcp``.

    A policyskill-v1 instance (PREREG E14) runs in the LEARN arm under the kept rule joint-retain-v1 with its floor
    in rows (``retention_floor_rows``, old-mission check episodes), and is refused before any write otherwise.

    An E11 instance is refused before any write unless its student passes ``weightcode.harness.e11.check_student`` (no
    BatchNorm statistics; one class token and one positional embedding, each of rank >= 2), and an E12 instance
    unless its student passes ``weightcode.harness.e12.check_student`` (exactly the resnet18_up tensors), and a T04
    instance unless its student passes ``weightcode.harness.t04.check_student`` (the resnet18 tensors, 1000-class head).

    ``max_checks`` (an arm with ``patch_check``, PREREG A4b: 2 in the LEARN arm) is the session's patch-check budget in
    place of the instance's, which it may only lower: in the trial task.json, the task prompt, session.json ``budget``
    (which the check ledger enforces, so the next check after the last is refused) and session.json ``check_cap``.
    None leaves the session as without it. A loop round takes none.

    The LEARN_BLIND arm (PREREG A4a) runs on an evidence-v1 instance without a loop context, library sessions or a
    prompt override. Its trial holds no parameter value and, unless ``blind_predictions``, no prediction of the student
    (``weightcode.harness.trial.assemble``); its programs run on the session's private copies of the LEARN inputs, recorded as
    session.json ``blind``, and its checks return no program output (``weightcode.harness.tools``). ``blind_predictions`` is
    for that arm only.

    ``e11_variant`` (PREREG K1): None or "standard" is the E11 session, unchanged; "rule" is the rule view, in the
    LEARN arm on an E11 evidence-v1 instance whose condition record carries a stated rule
    (``weightcode.harness.e11.condition_rule``), with no loop context, library sessions or prompt override; anything else is
    refused before any write. The rule and the condition record's hash are recorded as session.json ``e11_rule``, and
    session.json, session_result.json, run.json and the manifest line carry ``e11_variant: "rule"`` (absent for the
    standard view), so a rule-view session is never read as the standard session of the same instance, arm and repeat.
    The ladder variants "unlabelled" (U), "named" (K0b) and "named_nocheck" (K0a; it needs a retention floor) run in the
    LEARN arm on the one ladder instance (``weightcode.harness.e11.ladder_view``) with no loop context, library sessions,
    prompt override or ``max_checks``, are refused before any write otherwise, and record ``e11_variant`` and
    ``e11_view`` in session.json and ``e11_variant`` in every other record (module docstring).
    """
    _check_arguments(arm, repeat, model, effort, wall_timeout_s, threads, system_prompt_mode,
                     DEFAULT_MCP_TOOL_TIMEOUT_MS if mcp_tool_timeout_ms is None else mcp_tool_timeout_ms,
                     cell_timeout_s)
    spec = arm_spec(arm)
    try:
        variant = seqskill.check_variant(seqskill_variant)
    except seqskill.SeqskillHarnessError as exc:
        raise ValueError(str(exc)) from None
    rule_only = variant == seqskill.RULE_ONLY
    if not isinstance(lift_rel_frob, bool):
        raise ValueError(f"lift_rel_frob must be a bool, got {lift_rel_frob!r}")
    if lift_rel_frob and arm != "LEARN":
        raise ValueError(f"the rel-Frob limit is lifted in the LEARN arm only, not {arm}")
    if lift_rel_frob and loop_context is not None:
        raise ValueError("a loop round runs on an evidence-v1 instance; the rel-Frob limit is lifted for seqskill-v1")
    if rule_only and loop_context is not None:
        raise ValueError("a loop round runs on an evidence-v1 instance; the rule_only variant is for seqskill-v1")
    if retention_floor_frac is not None:
        if not spec.program:
            raise ValueError(f"the retention kept rule applies to a program arm only, not {arm}")
        if not 0.0 < float(retention_floor_frac) < 1.0:
            raise ValueError(f"retention_floor_frac must lie in (0, 1), got {retention_floor_frac!r}")
    if retention_floor_rows is not None:
        if not spec.program:
            raise ValueError(f"the retention kept rule applies to a program arm only, not {arm}")
        if retention_floor_frac is not None:
            raise ValueError("a retention floor is given as a fraction or in rows, not both")
        if loop_context is not None:
            raise ValueError("a loop round has its own kept rule; it takes no retention floor")
        try:
            check_floor_rows(retention_floor_rows)
        except RetentionRuleError as exc:
            raise ValueError(str(exc)) from None
    try:
        version = fscil.check_version(fscil_version)
    except fscil.FscilHarnessError as exc:
        raise ValueError(str(exc)) from None
    fscil_v2 = version == fscil.FSCIL_V2  # PREREG E9 v2: kept rule fscil-retain-v2, no floor given
    if fscil_v2:
        if not spec.program:
            raise ValueError(f"an E9 v2 session runs in a program arm only, not {arm}")
        if retention_floor_frac is not None or retention_floor_rows is not None or retention_floor_source is not None:
            raise ValueError("an E9 v2 session keeps by fscil-retain-v2, whose floor is per candidate; it takes no "
                             "retention floor")
        if loop_context is not None:
            raise ValueError("a loop round runs on an evidence-v1 instance; the E9 v2 kept rule is for fscil-v1")
    if retention_floor_source is not None:
        if retention_floor_rows is None:
            raise ValueError("a floor source goes with a floor in rows (retention_floor_rows)")
        if not isinstance(retention_floor_source, Mapping):
            raise ValueError(f"a floor source is a JSON object, not {type(retention_floor_source).__name__}")
        if is_placeholder(retention_floor_source) and not dry_run:
            raise ValueError("a placeholder floor is a stand-in floor in rows for a dry run only")
    if library_sessions is not None:
        if not spec.program:
            raise ValueError(f"a program library applies to a program arm only, not {arm}")
        if task_prompt_override is not None:
            raise ValueError("a program library is stated in the rendered task prompt; it takes no prompt override")
    try:
        check_max_checks(max_checks, arm)
    except TrialError as exc:
        raise ValueError(str(exc)) from None
    if max_checks is not None and loop_context is not None:
        raise ValueError("a loop round has the check budget of its context; it takes no max_checks")
    if not isinstance(blind_predictions, bool):
        raise ValueError(f"blind_predictions must be a bool, got {blind_predictions!r}")
    if blind_predictions and not spec.blind:
        raise ValueError(f"blind_predictions is for the LEARN_BLIND arm, not {arm}")
    if spec.blind and (loop_context is not None or library_sessions is not None or task_prompt_override is not None):
        raise ValueError(f"{arm} runs on an instance's own student with its own prompt; it takes no loop context, "
                         "library sessions or prompt override")
    e11_name = e11.check_variant(e11_variant)  # ValueError for an unknown variant
    e11_rule_view = e11_name == e11.RULE
    ladder = e11_name if e11_name in e11.LADDER_VARIANTS else None
    if ladder is not None:  # the information ladder: E11's LEARN session with only the editor's information changed
        if arm != "LEARN":
            raise ValueError(f"the E11 {ladder} view runs in the LEARN arm only, not {arm}")
        if loop_context is not None:
            raise ValueError(f"the E11 {ladder} view is refused in a loop round")
        if library_sessions is not None or task_prompt_override is not None:
            raise ValueError(f"the E11 {ladder} view takes no library sessions or prompt override")
        if max_checks is not None:
            raise ValueError(f"the E11 {ladder} view has E11's checks; it takes no max_checks")
        if ladder == e11.NAMED_NOCHECK and retention_floor_frac is None and retention_floor_rows is None:
            raise ValueError("the K0a view keeps its one post-session check under joint-retain-v1; give its retention "
                             "floor")
    if e11_rule_view:  # PREREG K1: the LEARN arm, the instance's own student, the rule-view prompt, nothing added
        if arm != "LEARN":
            raise ValueError(f"the E11 rule view runs in the LEARN arm only, not {arm}")
        if loop_context is not None:
            raise ValueError("the E11 rule view is refused in a loop round")
        if library_sessions is not None or task_prompt_override is not None:
            raise ValueError("the E11 rule view gives the editor the network, the clean rows and the rule only; it "
                             "takes no library sessions or prompt override")
    if loop_context is not None:
        if not spec.program:
            raise ValueError(f"a loop round runs in a program arm only, not {arm}")
        if task_prompt_override is not None or retention_floor_frac is not None or library_sessions is not None:
            raise ValueError("a loop round has its own prompt, kept rule and library; it takes no prompt override, "
                             "retention floor or library sessions")
        if allow_main:
            raise ValueError("a loop round runs on dev instances only")
    record = read_instance_record(instance_id, allow_main=allow_main)
    timeouts = (e11.mcp_timeouts if e11.is_e11(record.private) else
                e12.mcp_timeouts if e12.is_e12(record.private) else
                t04.mcp_timeouts if t04.is_t04(record.private) else
                policyskill.mcp_timeouts if record.is_policyskill else e10.mcp_timeouts)
    try:  # None: the instance's default (E10's on an E10 or E11 instance, which refuses smaller values)
        mcp_tool_timeout_ms, mcp_connect_timeout_ms = timeouts(
            record.private, mcp_tool_timeout_ms, mcp_connect_timeout_ms,
            (DEFAULT_MCP_TOOL_TIMEOUT_MS, DEFAULT_MCP_CONNECT_TIMEOUT_MS))
    except ValueError as exc:
        raise InstanceRefused(f"instance {instance_id} refused: {exc}") from None
    if e11.is_e11(record.private):  # PREREG E11: no BatchNorm; class token and positions under rel-Frob
        try:
            e11.check_student(load_params(record.instance_dir / "model.npz"))
        except e11.E11HarnessError as exc:
            raise InstanceRefused(f"instance {instance_id} refused: {exc}") from None
    if e12.is_e12(record.private):  # PREREG E12: the resnet18_up tensors, BatchNorm statistics as 1-D parameters
        try:
            e12.check_student(load_params(record.instance_dir / "model.npz"))
        except e12.E12HarnessError as exc:
            raise InstanceRefused(f"instance {instance_id} refused: {exc}") from None
    if t04.is_t04(record.private):  # PREREG T04: the resnet18 tensors with the 1000-class head
        try:
            t04.check_student(load_params(record.instance_dir / "model.npz"))
        except t04.T04HarnessError as exc:
            raise InstanceRefused(f"instance {instance_id} refused: {exc}") from None
    if e11_rule_view:  # PREREG K1: an E11 instance whose condition record carries the stated rule
        try:
            e11.condition_rule(record.private)
        except (e11.E11HarnessError, OSError, ValueError) as exc:
            raise InstanceRefused(f"instance {instance_id} refused for the E11 rule view: {exc}") from None
    if ladder is not None:  # the information ladder: the one PREREG instance
        try:
            e11.ladder_view(record.private, ladder)
        except e11.E11HarnessError as exc:
            raise InstanceRefused(f"instance {instance_id} refused for the E11 {ladder} view: {exc}") from None
    if record.has_own_experience and not spec.program:
        raise InstanceRefused(f"instance {instance_id} is a {record.private['evidence_rule']} instance; it runs in a "
                              f"program arm (LEARN) only, not {arm}")
    if record.has_own_experience and spec.blind:
        raise InstanceRefused(f"instance {instance_id} is a {record.private['evidence_rule']} instance; {arm} runs on "
                              "an evidence-v1 instance only")
    if max_checks is not None and max_checks > int(record.task["patch_check"]["budget_queries"]):
        raise InstanceRefused(f"instance {instance_id}: max_checks {max_checks} is above its patch-check budget of "
                              f"{record.task['patch_check']['budget_queries']}; it can only lower it")
    if rule_only and not record.is_seqskill:
        raise InstanceRefused(f"instance {instance_id} is not a seqskill-v1 instance; the rule_only variant is for "
                              "one")
    if lift_rel_frob and not record.is_seqskill:
        raise InstanceRefused(f"instance {instance_id} is not a seqskill-v1 instance; the rel-Frob limit is lifted "
                              "for one only")
    if fscil_v2 and not record.is_fscil:
        raise InstanceRefused(f"instance {instance_id} is not an fscil-v1 instance; the E9 v2 session is for one")
    graphnew_vcheck_sha256: str | None = None
    if record.is_graphnew:  # PREREG E7: the floor is a number of old-class check rows, stated before the session
        if retention_floor_frac is not None:
            raise InstanceRefused(f"instance {instance_id}: the PREREG E7 floor is a row count, not a fraction")
        if retention_floor_rows is None:
            raise InstanceRefused(f"instance {instance_id} is a graphnew-v1 instance; its session runs under the kept "
                                  "rule joint-retain-v1 with the PREREG E7 floor in rows (retention_floor_rows)")
        try:
            graphnew_vcheck_sha256 = graphnew.vcheck_file_sha256(record.private)  # content hash checked first
        except (graphnew.GraphnewHarnessError, OSError) as exc:
            raise InstanceRefused(f"instance {instance_id} refused: {exc}") from None
    if record.is_policyskill:  # PREREG E14: the floor is a number of old-mission check episodes, stated before
        if retention_floor_frac is not None:
            raise InstanceRefused(f"instance {instance_id}: the PREREG E14 floor is an episode count, not a fraction")
        if retention_floor_rows is None:
            raise InstanceRefused(f"instance {instance_id} is a policyskill-v1 instance; its session runs under the "
                                  "kept rule joint-retain-v1 with the PREREG E14 floor in episodes "
                                  "(retention_floor_rows)")
    fscil_n_vcheck: int | None = None
    if record.is_fscil:  # PREREG E9: the session's floor is a number of old-part check rows, stated before the session
        if retention_floor_frac is not None:
            raise InstanceRefused(f"instance {instance_id}: the PREREG E9 floor is a row count, not a fraction")
        if retention_floor_rows is None and not fscil_v2:
            raise InstanceRefused(f"instance {instance_id} is an fscil-v1 instance; its session runs under the kept "
                                  "rule joint-retain-v1 with the PREREG E9 floor of its session in rows "
                                  "(retention_floor_rows)")
        try:
            fscil_n_vcheck = fscil.vcheck_rows(record.private)  # file and content hash checked first
        except (fscil.FscilHarnessError, OSError, KeyError) as exc:
            raise InstanceRefused(f"instance {instance_id} refused: {exc}") from None
    fscil_kept_v2: dict[str, Any] | None = None
    if fscil_v2:  # PREREG E9 v2: the kept rule and its base-row counts, read before any write
        try:
            fscil_kept_v2 = fscil.v2_rule_for(record.private)
        except (fscil.FscilHarnessError, fscil.stale_error(), OSError, KeyError) as exc:
            raise InstanceRefused(f"instance {instance_id} refused for an E9 v2 session: {exc}") from None
    if retention_floor_rows is not None:  # the rows behind score_old, known before any write
        n_old_rows = record.private["n_vcheck_old"] if record.has_own_experience else record.private["n_vcheck"]
        try:
            check_floor_rows(retention_floor_rows, int(n_old_rows))
        except RetentionRuleError as exc:
            raise InstanceRefused(f"instance {instance_id} refused: {exc}") from None
    prepared: loop.PreparedRound | None = None
    if loop_context is not None:  # the whole round is read and checked before anything is written
        if record.has_own_experience or record.split_role != "dev":
            raise InstanceRefused(f"instance {instance_id}: a loop round runs on an evidence-v1 dev instance")
        try:
            prepared = loop.prepare_round(Path(loop_context), instance_id, record.private)
        except loop.LoopError as exc:
            raise InstanceRefused(f"instance {instance_id} refused for a loop round: {exc}") from None
    library_sources: list[dict[str, Any]] | None = None
    if library_sessions is not None and record.is_fscil:  # the lineage's kept programs, checked before any write
        try:
            library_sources = fscil.library_entries(library_sessions, runs_dir() / SESSIONS_DIR,
                                                    target_instance_id=instance_id, version=version)
        except (fscil.FscilHarnessError, fscil.stale_error(), OSError, KeyError, ValueError) as exc:
            raise InstanceRefused(f"instance {instance_id} refused for a library session: {exc}") from None
    elif library_sessions is not None:  # every library source is read and checked before anything is written
        if record.has_own_experience:
            raise InstanceRefused(f"instance {instance_id} is a {record.private['evidence_rule']} instance; a program "
                                  f"library is for evidence-v1 instances")
        try:
            _, target_clean = clean.load_clean_check(load_instance(record.instance_dir))
            library_sources = library_entries(library_sessions, runs_dir() / SESSIONS_DIR,
                                              target_instance_id=instance_id,
                                              target_model_id=str(record.private["model_id"]),
                                              target_clean_sha256=str(target_clean["clean_sha256"]))
        except (LibraryError, clean.CleanRowsUnavailable) as exc:
            raise InstanceRefused(f"instance {instance_id} refused for a library session: {exc}") from None
    extra_paths: tuple[str, ...] = ()
    if spec.evidence and spec.shown == SHOWN_DECOY:  # review finding B: refused before anything is written
        try:
            decoy_source, _ = decoy_evidence_source(record.instance_dir)
        except TrialError as exc:
            raise InstanceRefused(f"instance {instance_id} refused for {arm}: {exc}") from None
        extra_paths = (str(decoy_source), str(decoy_source.parent))
    if spec.program and not record.has_own_experience:  # the replay rows and the clean V_check companion exist
        try:
            clean.preflight(load_instance(record.instance_dir))
        except clean.CleanRowsUnavailable as exc:
            raise InstanceRefused(f"instance {instance_id} refused for {arm}: {exc}") from None
    if launcher is None and not dry_run:
        launcher = ClaudeLauncher(claude_bin, api_key_file=api_key_file, oauth_token_file=oauth_token_file)
    if not SERVER_PYTHON.is_file():
        raise FileNotFoundError(f"MCP server interpreter {SERVER_PYTHON} is missing")
    if prepared is not None:
        extra_paths = (*extra_paths, *prepared.extra_forbidden)
    extra_tokens: list[str] = []
    if record.is_fscil:
        try:
            extra_tokens = fscil.session_extra_tokens(
                record.private, fscil.library_ids(library_sources) if library_sources is not None else ())
        except (fscil.FscilHarnessError, fscil.stale_error()) as exc:
            raise InstanceRefused(f"instance {instance_id} refused: {exc}") from None
    elif (e10.is_e10(record.private) or e11.is_e11(record.private) or e12.is_e12(record.private)
          or t04.is_t04(record.private)):
        tokens_of = (e11.self_test_tokens if e11.is_e11(record.private) else
                     e12.self_test_tokens if e12.is_e12(record.private) else
                     t04.self_test_tokens if t04.is_t04(record.private) else e10.self_test_tokens)
        extra_tokens = tokens_of(() if prepared is None else e10.library_session_ids(prepared.context["library"]))
    forbidden = session_forbidden_strings(record, extra_paths, seqskill_variant=variant, extra_tokens=extra_tokens)
    if ladder in e11.NAMED_VARIANTS:  # K0a, K0b: the capability sentence is the information of the rung
        forbidden = dataclasses.replace(
            forbidden, sentence_exempt=(e11.CAPABILITY_SENTENCES[str(record.private["condition"])],))

    parent = runs_dir() / (DRY_RUNS_DIR if dry_run else SESSIONS_DIR)
    parent.mkdir(exist_ok=True)
    session_id = new_session_id(forbidden.text, parent)
    paths = SessionPaths(parent / session_id)
    paths.create()
    real_model_path = record.instance_dir / "model.npz"
    loop_record: dict[str, Any] | None = None
    edit_log_files: dict[str, str] | None = None  # PREREG T16 arm L only
    if prepared is None:
        view = assemble(record.instance_dir, arm, paths.root, seqskill_variant=variant, lift_rel_frob=lift_rel_frob,
                        fscil_kept_rule=fscil_kept_v2, max_checks=max_checks, blind_predictions=blind_predictions,
                        e11_variant=e11_variant)
        if library_sources is not None and record.is_fscil:  # a program with a forbidden string is listed unshown
            library_sources = fscil.withhold(library_sources, forbidden.text)
            library = write_library(paths.trial, library_sources, fscil.library_index(library_sources, version))
        else:
            library = None if library_sources is None else write_library(paths.trial, library_sources)
    else:
        real_model_path, prev_checks = loop.write_round_private(prepared, paths.private)
        view = assemble(record.instance_dir, arm, paths.root, override=StudentOverride(
            params=prepared.student, experience=prepared.experience, record=prepared.experience_record,
            keys=(*EXPERIENCE_KEYS, *loop.PREV_KEYS), task=prepared.task_rewrite))
        withheld = loop.withheld_tokens(prepared.context["library"], forbidden.library_scan)
        library = loop.write_loop_library(paths.trial, prepared.context["library"], int(record.private["n_vcheck"]),
                                          withheld)
        edit_log_files = None if prepared.edit_log is None else loop.write_edit_log(paths.trial, prepared.edit_log)
        context = prepared.context
        loop_record = {"schema": loop.LOOP_RECORD_SCHEMA, "stream_id": context["stream_id"], "round": prepared.round,
                       "context_path": str(prepared.context_path), "context_sha256": file_sha256(prepared.context_path),
                       "student_sha256": context["student"]["sha256"], "student_source": context["student"]["path"],
                       "base_params_sha256": context["base_params_sha256"], "prev_checks": prev_checks,
                       "prev_instances": [item["instance_id"] for item in prev_checks],
                       "edit_log": loop.session_edit_log(prepared, edit_log_files),
                       **({} if prepared.protect is None else {"protect_logged": prepared.protect})}
    task = record.task
    task_budget = patch_check_budget(int(task["patch_check"]["budget_queries"]), arm)
    hidden_check = ladder == e11.NAMED_NOCHECK  # K0a: one query, spent by finalisation alone
    budgets = Budgets(patch_checks=1 if hidden_check else task_budget if max_checks is None else max_checks,
                      python_cells=max_cells, cell_timeout_s=cell_timeout_s, wall_timeout_s=wall_timeout_s)
    session_tools = tuple(tool for tool in spec.tools if not (hidden_check and tool == PATCH_CHECK_TOOL))
    allowed_tools = tuple(name for name, tool in zip(spec.allowed_tools, spec.tools) if tool in session_tools)
    if hidden_check and view.e11_view is not None:
        view.e11_view.update(hidden_check_budget=1, editor_tools=list(session_tools))
    kept_rule = None
    if retention_floor_frac is not None or retention_floor_rows is not None:  # the rows behind score_old: old-class
        # (fewshot-v1, graphnew-v1), old-operator (seqskill-v1), old-mission (policyskill-v1) or clean (evidence-v1)
        if view.fewshot is not None:
            n_old, check_set = int(view.fewshot["n_vcheck_old"]), "old-class"
        elif view.seqskill is not None:
            n_old, check_set = int(view.seqskill["n_vcheck_old"]), seqskill.check_set_name(view.seqskill)
        elif view.graphnew is not None:
            n_old, check_set = int(view.graphnew["n_vcheck_old"]), graphnew.CHECK_SET
        elif view.policyskill is not None:
            n_old, check_set = int(view.policyskill["n_vcheck_old"]), policyskill.CHECK_SET
        elif view.fscil is not None:
            n_old, check_set = int(view.fscil["n_vcheck_old"]), fscil.CHECK_SET
        else:
            n_old, check_set = int(view.clean_check["n"]), "clean"
        kept_rule = (kept_rule_record(retention_floor_frac, n_old, check_set) if retention_floor_frac is not None
                     else kept_rule_record_rows(int(retention_floor_rows), n_old, check_set, retention_floor_source))
    if fscil_kept_v2 is not None:
        kept_rule = fscil_kept_v2
    if prepared is not None:
        kept_rule = prepared.kept_rule
        system_prompt = loop.loop_kept_section(render_system(arm), kept_rule)
        if prepared.edit_log is not None:  # PREREG T16 arm L: the one added sentence
            system_prompt = loop.edit_log_section(system_prompt)
        if prepared.protect is not None:  # T16 follow-up M: the limit's one sentence, after the edit log's
            system_prompt = loop.protect_section(system_prompt)
    else:
        system_prompt = render_system(arm, fewshot=record.is_fewshot, kept_rule=kept_rule,
                                      seqskill=record.is_seqskill, seqskill_variant=variant,
                                      graphnew=record.is_graphnew, fscil=record.is_fscil, e11_variant=e11_variant,
                                      policyskill=record.is_policyskill)
    task_prompt = (task_prompt_override if task_prompt_override is not None
                   else render_task(arm, view.task, view.limits, budgets, view.input_files,
                                    seqskill=record.is_seqskill, seqskill_variant=variant,
                                    graphnew=record.is_graphnew, fscil=record.is_fscil, e11_variant=e11_variant,
                                    policyskill=record.is_policyskill))
    if prepared is not None and library is not None:
        n_files = sum(1 for name in library["files"] if name != loop.LOOP_LIBRARY_INDEX)
        task_prompt = loop.loop_files_section(task_prompt, prepared.n_prev, n_files + 1)
    elif library is not None and record.is_fscil and library_sources is not None:
        task_prompt = fscil_library_files_section(task_prompt,
                                                  fscil.library_bullet(library_sources, LIBRARY_INDEX_FILE, version))
    elif library is not None:
        task_prompt = library_files_section(task_prompt, len(library["entries"]), LIBRARY_INDEX_FILE,
                                            LIBRARY_PROGRAM_TEMPLATE)
    paths.system_prompt.write_text(system_prompt, encoding="utf-8")
    paths.task_prompt.write_text(task_prompt, encoding="utf-8")
    tools = tool_specs(session_tools, spec.program, fewshot=record.is_fewshot,
                       s_old_floor=(None if kept_rule is None or prepared is not None or fscil_kept_v2 is not None
                                    else int(kept_rule["s_old_floor"])),
                       loop_rule=None if prepared is None else kept_rule, seqskill=record.is_seqskill,
                       graphnew=record.is_graphnew, fscil=record.is_fscil, fscil_v2=fscil_kept_v2, blind=spec.blind,
                       e11_variant=e11_variant, scan=seqskill.is_scan(record.private),
                       policyskill=record.is_policyskill)
    scan = leak_scan(paths, {"prompts/system.md": system_prompt, "prompts/task.md": task_prompt}, tools, forbidden,
                     labels=spec.labels, scanner=graphnew.scan if record.is_graphnew else scan_for_tokens)

    claude_info: dict[str, Any] | None = None
    if isinstance(launcher, ClaudeLauncher):
        env, removed = editor_environment(os.environ, {"mcp": {"tool_timeout_ms": mcp_tool_timeout_ms,
                                                              "connect_timeout_ms": mcp_connect_timeout_ms}})
        claude_info = {"bin": launcher.claude, "realpath": str(Path(launcher.claude).resolve()),
                       "version": claude_version(launcher.claude, env), "env_removed": removed,
                       "billing": dict(launcher.billing)}
    launcher_name = "none (dry run)" if launcher is None else launcher.name
    session: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "session_id": session_id,
        "arm": arm,
        "repeat": repeat,
        "instance_id": instance_id,
        "split_role": record.split_role,
        "dataset": record.private["dataset"],
        "condition": record.private.get("condition"),  # None for a fewshot-v1 instance
        "evidence_rule": record.private["evidence_rule"],
        "model_id": record.private["base_model_id"] if record.is_fscil else record.private["model_id"],
        "decoy_model_id": record.private.get("decoy_model_id"),
        "seed": record.private.get("seed"),
        "dry_run": dry_run,
        "paths": {"root": str(paths.root), "trial": str(paths.trial), "patches": str(paths.patches),
                  "scratch": str(paths.scratch), "cwd": str(paths.cwd), "logs": str(paths.logs)},
        "instance": {"dir": str(record.instance_dir), "task_sha256": record.private["task_sha256"],
                     "task_json_sha256": file_sha256(record.instance_dir / "task.json"),
                     "private_json_sha256": file_sha256(record.instance_dir / "private.json"),
                     "index_row": record.index_row},
        "real_model_path": str(real_model_path),
        "real_model_sha256": view.real_model_sha256,
        "shown": view.shown,
        "shown_model_sha256": view.shown_model_sha256,
        "decoy_model_sha256": view.decoy_model_sha256,
        "apply_mode": view.apply_mode,
        "edit_limits_enforced": spec.size_limits,
        "forward_path": str(view.private_forward_path),
        "forward_sha256": record.private["forward_sha256"],
        "vcheck_path": record.private["vcheck_path"],
        "vcheck_sha256": (graphnew_vcheck_sha256 if record.is_graphnew  # private.json records its content hash
                          else record.private["vcheck_sha256"]),
        # the V_check file's rows: a scanskill-v1 instance's holds the new part only (old_check is its own file)
        "n_vcheck": (fscil_n_vcheck if record.is_fscil else record.private["n_vcheck_new"]
                     if seqskill.is_scan(record.private) else record.private["n_vcheck"]),
        "n_base_errors_vcheck": record.private.get("n_base_errors_vcheck"),  # fewshot: per part in "fewshot"
        "budget": {"patch_checks": budgets.patch_checks, "python_cells": max_cells, "cell_timeout_s": cell_timeout_s,
                   "wall_timeout_s": wall_timeout_s},
        "limits": view.limits,
        "tools": list(session_tools),
        "allowed_tools": list(allowed_tools),
        "editor": {"model": model, "effort": effort, "launcher": launcher_name,
                   "system_prompt_mode": system_prompt_mode,
                   "claude_version": None if claude_info is None else claude_info["version"]},
        "sandbox": {"python": str(sandbox_python), "cell_timeout_s": cell_timeout_s, "max_cells": max_cells,
                    "memory_bytes": memory_bytes, "threads": threads, "max_output_chars": max_output_chars},
        "server_threads": threads,
        "server_python": str(SERVER_PYTHON),
        "mcp": {"config_path": str(paths.mcp_json), "tool_timeout_ms": mcp_tool_timeout_ms,
                "connect_timeout_ms": mcp_connect_timeout_ms},
        "trial": view.as_dict(),
        **({"experience": view.experience, "clean_check": view.clean_check,
            "program": {"wall_s": PROGRAM_WALL_S, "clock": PROGRAM_CLOCK, "driver_sha256": driver_sha256(),
                        "max_program_bytes": MAX_PROGRAM_BYTES, "tool_margin_s": PROGRAM_TOOL_MARGIN_S}}
           if spec.program else {}),
        **({"fewshot": view.fewshot} if view.fewshot is not None else {}),
        **({"seqskill": view.seqskill} if view.seqskill is not None else {}),
        **({"lift_rel_frob": True} if lift_rel_frob else {}),
        **({"graphnew": view.graphnew} if view.graphnew is not None else {}),
        **({"fscil": view.fscil} if view.fscil is not None else {}),
        **({"policyskill": view.policyskill} if view.policyskill is not None else {}),
        **({"kept_rule": kept_rule} if kept_rule is not None else {}),
        **({"library": library} if library is not None else {}),
        **({"loop": loop_record} if loop_record is not None else {}),
        **({"blind": view.blind} if view.blind is not None else {}),
        **({"check_cap": {"max_checks": max_checks, "task_budget_queries": task_budget}}
           if max_checks is not None else {}),
        **({"e11_variant": e11.RULE, "e11_rule": view.e11_rule} if view.e11_rule is not None else {}),
        **({"e11_variant": str(view.e11_view["variant"]), "e11_view": view.e11_view}
           if view.e11_view is not None else {}),
        "trial_files_sha256": {**view.files, **({} if library is None else library["files"]),
                               **({} if edit_log_files is None else edit_log_files)},
        "evidence_path": None if view.evidence_path is None else str(view.evidence_path),
        "evidence_content_sha256": view.evidence_content_sha256,
        "evidence_source": view.evidence_source,
        "forbidden_strings": forbidden.as_dict(),
        "leak_scan": scan,
        "prompts": {"system_path": str(paths.system_prompt), "task_path": str(paths.task_prompt),
                    "task_prompt_override": task_prompt_override is not None,
                    "system_sha256": text_sha256(system_prompt), "task_sha256": text_sha256(task_prompt)},
        "claude": claude_info,
        "created_at": utc_now(),
    }
    session["provenance"] = collect(sandbox_python=sandbox_python,
                                    prompts={"system": system_prompt, "task": task_prompt}, claude=claude_info)
    if prepared is not None and prepared.protect is not None:  # follow-up M: the applier the tools and the final use
        protected = ArmApplier(json.loads(json.dumps(session))).protected  # sets the limit up from these records
        if protected is None or loop.protected_counts(protected) != prepared.protect["per_tensor"]:
            raise loop.LoopError("the session's applier does not hold the protected-entry limit it records")
    write_json(paths.session_json, session)
    write_json(paths.mcp_json, mcp_config(paths.session_json, threads))
    real = load_params(Path(session["real_model_path"]))

    if not scan["passed"]:
        result = aborted_result(paths, session, STATUS_ABORTED_LEAK, real, dry_run=dry_run, leak_scan=scan)
        return SessionResult(session_id, STATUS_ABORTED_LEAK, paths.root, result)
    selftest = isolation_selftest(sandbox_config(session, paths), log_path=paths.selftest_cells,
                                  forbidden_strings=forbidden.selftest)
    write_json(paths.selftest_json, selftest)
    selftest_summary = {"status": selftest["status"], "failed": selftest["failed"], "path": "logs/selftest.json",
                        "duration_s": selftest["duration_s"]}
    if selftest["status"] != "pass":
        result = aborted_result(paths, session, STATUS_ABORTED_SELFTEST, real, dry_run=dry_run,
                                selftest=selftest_summary, leak_scan=scan)
        return SessionResult(session_id, STATUS_ABORTED_SELFTEST, paths.root, result)
    if dry_run or launcher is None:
        result = base_result(session, STATUS_DRY_RUN, selftest=selftest_summary, leak_scan=scan,
                             trial_files=sorted(view.files), tools=[tool["name"] for tool in tools],
                             audit={"status": "not_run", "reason": "dry run"})
        write_json(paths.result_json, result)
        return SessionResult(session_id, STATUS_DRY_RUN, paths.root, result)

    try:
        outcome = launcher.launch(session, paths, task_prompt)
    except Exception as exc:  # recorded as a harness exception (INFRA) by finalisation
        outcome = LaunchOutcome(launcher=launcher.name, stream_expected=launcher.stream_expected,
                                started_at=session["created_at"], ended_at=utc_now(),
                                error=f"launcher: {type(exc).__name__}: {exc}")
    after = collect(sandbox_python=sandbox_python, prompts={"system": system_prompt, "task": task_prompt},
                    claude=claude_info)
    result = finalize_session(paths, session, outcome, skip_audit=skip_audit, provenance_after=after,
                              code_changes=code_changed(session["provenance"], after),
                              extra={"selftest": selftest_summary, "leak_scan": scan})
    return SessionResult(session_id, str(result["status"]), paths.root, result)


__all__ = ["DEFAULT_MODEL", "DRY_RUNS_DIR", "SESSIONS_DIR", "SessionResult", "leak_scan", "mcp_config",
           "run_session", "session_forbidden_strings"]
