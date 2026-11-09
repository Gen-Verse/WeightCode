"""Trial assembly: the arm-specific editor view of one instance (contract 5.4, HARNESS_SPEC section 5).

``assemble(instance_dir, arm, session_dir)`` writes ``<session_dir>/trial/`` (bound read-only at /work/trial) and
the private copies the oracle uses (``<session_dir>/private/forward.py`` plus the instance's input files), checking
every hash against private.json and recording the hash of every file it writes.

Per arm (HARNESS_SPEC section 6): M2 and DECOY show ``model.npz`` (real or decoy weights), M1 the real weights
without ``evidence.npz``, NOWEIGHT ``model_shapes.json`` instead of ``model.npz``, REWRITE a ``model.npz`` of
zeros. Every arm gets ``model_spec.json`` (without its ``model_id`` key), ``forward.py``, ``task.json`` and the
files listed in ``model_spec.json["input_files"]``. The arm's task.json gets ``parent_sha256``, the
``params_sha256`` of the model shown in that arm (the real hash in NOWEIGHT); in M1 it loses the evidence section
and the statement's sentence about the evidence file; in REWRITE its limits are the ones applied (size limits
disabled). Nothing else differs between arms.

Evidence (review finding B, 2026-11-03): every arm with evidence gets the instance's own evidence.npz, except DECOY,
which gets the evidence drawn by the evidence-v1 rule from the decoy model's own errors
(``weightcode.evalkit.instances.decoy_evidence``), so that its ``pred_*`` arrays match a forward pass of the shown decoy
weights. A DECOY trial is refused when that evidence is not available; it never falls back to the real evidence.

M2NL (label-free M2): the real weights as in M2 and an ``evidence.npz`` written from the instance's evidence with only
the random part and no labels: ``x_rand``, ``pred_rand`` and the random rows' ``split_ids``; ``x_fail``, ``y_fail``,
``pred_fail`` and ``y_rand`` are dropped. Its task.json describes only that sample: the evidence section lists the
random part without labels, the statement's evidence sentence is replaced by one about random examples and the
classifier's predictions, and the ``patch_check`` section (a tool the arm does not have) is replaced by
``final_selection``, the declared-patch rule. Its evidence hash is ``params_sha256`` of the three arrays.

LEARN (update programs): the real weights and, instead of ``evidence.npz``, ``experience.npz`` with the instance's
evidence as the new-condition part and the dataset's replay rows (clean train rows, rule replay-v1, through
``weightcode.harness.clean``) as the old part (``weightcode.harness.experience``); its task.json is ``learn_task`` (experience,
program and joint-check sections, the limits a program is held to; no evidence or patch-format section and no condition
name). No clean companion row reaches the trial: the clean companion of V_check is copied to
``private/clean_vcheck.npz`` (outside every sandbox mount, hashed into session.json) for the joint oracle of the MCP
server, which runs without the store variables (``weightcode.harness.clean``).

LEARN on a fewshot-v1 instance (PREREG E2, ``weightcode.harness.fewshot``; no other arm): the real weights and
``experience.npz`` = the instance's evidence.npz, which already has the LEARN keys (labelled examples of the new
classes and the old-class replay rows), with every ``pred_*`` re-checked; its task.json is ``learn_task_fewshot``.
There is no decoy and no clean companion: the view records the ``fewshot`` section (class lists and V_check part
sizes) from which the session ledger cuts V_check by class.

LEARN on a seqskill-v1 instance (PREREG E5, ``weightcode.harness.seqskill``; no other arm) is assembled the same way: the
real weights and ``experience.npz`` = the instance's evidence.npz (the new operator's rows and the old-operator replay
rows), with every ``pred_*`` re-checked; its task.json is ``learn_task_seqskill``. The view records the ``seqskill``
section (the new operator and the V_check part sizes) from which the session ledger cuts V_check by part.

Its variant rule_only (PREREG E8, ``seqskill_variant="rule_only"``) is assembled from the same instance with the same
checks; three trial files differ. ``experience.npz`` holds the old part alone (``x_old``, ``y_old``, ``pred_old``,
``split_ids_old``), cut after the whole evidence was re-checked (``weightcode.harness.seqskill.rule_only_experience``).
``model_spec.json`` states the new operator's rule, read from the condition record whose hash private.json records
(``weightcode.harness.seqskill.new_operator_rule`` and ``rule_only_model_spec``); the instance's own model_spec.json is
still checked against ``model_spec_sha256`` before it is rewritten. task.json is ``learn_task_seqskill`` of that
variant. The counts check reads the old part alone, and the ``seqskill`` section records the variant and the rule.

Its variant scan_rule (PREREG E13, ``seqskill_variant="scan_rule"``) is the view of a scanskill-v1 instance and of no
other: the interpretation rules are in the instance's own model_spec.json, the experience is the instance's evidence
(the primitive row when the condition has one, and the replay rows; ``weightcode.harness.seqskill.scan_experience``, every
``pred_*`` re-checked), task.json is ``learn_task_scan``, the counts check reads the primitive and old parts and no new
part, and the ``seqskill`` section records the old_check file the ledger loads beside V_check. It runs with the
rel-Frob limit lifted, as PREREG E13 states.

With ``lift_rel_frob`` (PREREG E8 amendment of 2026-11-04 15:37; a seqskill-v1 instance in the LEARN arm only) the
arm's limits are ``applied_limits`` with ``max_rel_frob`` None (``lifted_limits``). The one ``Limits`` value gives both
the trial task.json limits and their definitions (``program_limits_view``) and the view's ``limits``, which session.json
records and every check, declaration, final and readout replay applies (``weightcode.harness.applying.ArmApplier``,
``weightcode.evalkit.devreadout.run_limits``). The changed-entry cap and every other limit are unchanged.

LEARN on a graphnew-v1 instance (PREREG E7, ``weightcode.harness.graphnew``; no other arm) is assembled as on a fewshot-v1
instance: the real weights and ``experience.npz`` = the instance's evidence.npz (labelled nodes of the new classes and
the old-class replay nodes, as int64 node ids), with every ``pred_*`` re-checked in one forward; its task.json is
``learn_task_graphnew`` and its limits are ``graphnew.program_limits`` (equal to the LEARN arm's applied limits).
model_spec.json must list exactly ``graph.npz`` as input file; the trial and private copies of it must hash to
private.json ``graph_file_sha256``, since forward.py reads the copy beside itself. The view records the ``graphnew``
section (class lists, V_check part sizes and content hash, the graph's hash) from which the session ledger cuts V_check
by class.

LEARN on an fscil-v1 instance (PREREG E9, ``weightcode.harness.fscil``; no other arm) is assembled as on a fewshot-v1
instance: the instance's student (the base at session 2, a lineage's output after that) and ``experience.npz`` = the
instance's evidence.npz in three parts (session t's shots, the base replay rows and the shots of the earlier sessions
with their session), with every ``pred_*`` re-checked under the session mask; its task.json is ``learn_task_fscil``
and its limits are ``fscil.program_limits`` (equal to the LEARN arm's applied limits). model_spec.json, with the
session's prediction rule, is shown as the instance writes it. The view records the ``fscil`` section (the session, its
class lists, the check part sizes and old_check's path and hashes) from which the session ledger cuts the check.

LEARN on a policyskill-v1 instance (PREREG E14, ``weightcode.harness.policyskill``; no other arm) is assembled as on a
seqskill-v1 instance: the real weights and ``experience.npz`` = the instance's evidence.npz (every step of the new-mission
demonstrations and the old-mission replay steps, observation rows with the bot's action and the student's logits), with
every logit row re-checked against forward.py (``weightcode.harness.policyskill.experience``); its task.json is
``learn_task_policyskill`` and its limits are the LEARN arm's applied limits (``weightcode.harness.arms.applied_limits``,
equal to ``weightcode.evalkit.e14.e14_program_limits``). The view records the ``policyskill`` section (the V_check part sizes,
content and seed hashes and the base's per-episode success hashes) from which the session ledger builds the rollout
check.

LEARN_BLIND (PREREG A4a; an evidence-v1 instance without a student override only) is assembled as LEARN, with three
trial files different: ``model_shapes.json`` (NOWEIGHT's file: every tensor's name, shape and dtype) in place of
``model.npz``; ``experience.npz`` without the student's predictions ``pred_new`` and ``pred_old``
(``weightcode.harness.experience.blind_trial_experience``), unless ``blind_predictions`` keeps them; and task.json
``learn_task_blind``. What a program run gets is LEARN's: ``private/model.npz`` holds the real weights written exactly as
LEARN's trial ``model.npz``, and ``private/experience.npz`` the whole LEARN experience, both outside every editor mount
and recorded with their hashes in the view's ``blind`` section, from which the tool server stages a run.

``max_checks`` (PREREG A4b; any arm with ``patch_check``) lowers task.json ``patch_check.budget_queries`` to that number;
the runner gives the session the same budget. None leaves the task.json as without it.

The E11 rule view (PREREG K1, ``e11_variant="rule"``; the LEARN arm on an E11 evidence-v1 instance of a condition with a
stated rule, without a student override) is assembled as LEARN, with two trial files different: ``experience.npz``
holds the old part alone (``x_old``, ``y_old``, ``pred_old``, ``split_ids_old``), cut after ``build_experience`` checked
the whole experience (``weightcode.harness.experience.rule_experience``), and task.json is ``learn_task_rule`` of the LEARN
task.json, which states the condition's rule (``weightcode.harness.e11.condition_rule``, read and checked from the
condition record). The weights, model_spec.json, forward.py, the limits and the clean V_check companion are LEARN's.
The view records the rule and the condition record's hash (``e11_rule``); a program run gets the trial's experience,
so it never sees a row of the condition either.

The information ladder (PREREG; ``e11_variant`` "unlabelled", "named" or "named_nocheck"; the LEARN arm on the one
ladder instance, ``weightcode.harness.e11.ladder_view``, without a student override) is assembled as LEARN with these files
different. U: ``experience.npz`` without ``y_new`` (``weightcode.harness.experience.unlabelled_experience``), its rows and
their order E11's, and task.json ``learn_task_unlabelled``, which does not say which condition rows are errors. K0b
and K0a: no ``experience.npz`` in the trial and no array of the experience anywhere the editor can reach; a program run gets an empty experience from the session's private ``NAMED_PROGRAM_EXPERIENCE_FILE``
(an .npz without members, hash-recorded in ``e11_view``), and task.json is ``learn_task_named`` (K0a without a
``patch_check`` section). Every view cuts its experience after ``build_experience`` checked the whole LEARN
experience. The weights, model_spec.json, forward.py, the limits and the clean V_check companion are LEARN's. The view
records ``e11_view`` (the variant, the condition and seed, the capability sentence of K0a and K0b, K0a's hidden check
and the private experience file).
"""

from __future__ import annotations

import copy
import dataclasses
import os
import re
import shutil
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import Params, load_params, params_sha256, save_params
from weightcode.evalkit.evidence import evidence_content_sha256
from weightcode.evalkit.instances import InstanceConflict, decoy_evidence
from weightcode.evalkit.patch_check import ForwardFn, predict
from weightcode.harness.arms import SHOWN_DECOY, SHOWN_REAL, SHOWN_SHAPES, SHOWN_ZEROS, applied_limits, arm_spec, limits_dict
from weightcode.patch import Limits
from weightcode.patch.limits import dense_delta_bytes
from weightcode.harness.clean import load_clean_check, source_name
from weightcode.harness import e11, fewshot, fscil, graphnew, policyskill, seqskill
from weightcode.harness.experience import (
    EXPERIENCE_FILE,
    EXPERIENCE_KEYS,
    EXPERIENCE_SOURCE,
    NAMED_EXPERIENCE_SOURCE,
    RULE_EXPERIENCE_KEYS,
    RULE_EXPERIENCE_SOURCE,
    UNLABELLED_EXPERIENCE_KEYS,
    UNLABELLED_EXPERIENCE_SOURCE,
    ExperienceError,
    blind_trial_experience,
    build_experience,
    learn_task,
    learn_task_blind,
    learn_task_fewshot,
    learn_task_graphnew,
    learn_task_named,
    learn_task_policyskill,
    learn_task_rule,
    learn_task_scan,
    learn_task_seqskill,
    learn_task_unlabelled,
    named_experience,
    rule_experience,
    unlabelled_experience,
)
from weightcode.harness.instance import REQUIRED_PRIVATE_FIELDS, refuse_v0
from weightcode.harness.jsonio import file_sha256, json_text, read_json

SHAPES_SCHEMA = "weightcode.model_shapes.v0"
NAMED_PROGRAM_EXPERIENCE_FILE = "program_experience.npz"
"""A no-data session's private experience file: an .npz without members, what a program run gets (module docstring)."""
EVIDENCE_FROM_INSTANCE = "instance"
EVIDENCE_FROM_DECOY = "decoy_model_instance"
EVIDENCE_FROM_INSTANCE_RANDOM = "instance_random_unlabelled"
EVIDENCE_PARTS = ("fail", "rand")
UNLABELLED_EVIDENCE_KEYS = ("x_rand", "pred_rand", "split_ids")
UNLABELLED_SENTENCE = (
    "The evidence file holds random examples from the deployment data, with the classifier's predictions."
)
UNLABELLED_EVIDENCE_SECTION_KEYS = {"random": ["x_rand", "y_rand", "pred_rand"], "ids": ["split_ids"]}
UNLABELLED_RANDOM_DESCRIPTION = "examples drawn at random from the deployment data, with the classifier's predictions"
UNLABELLED_IDS_DESCRIPTION = "one opaque id per example, in x_rand order"
DECLARED_FINAL_SELECTION = (
    "kept at the end: the patch named in the last valid declare_final call (a patch that passes validation within "
    "the limits, or null for the unedited classifier); the unedited classifier if no valid declaration was made; no "
    "patch is checked or scored during the session"
)
PREDICT_BATCH = 1024
STANDARD_FILES = ("model.npz", "model_spec.json", "forward.py", "evidence.npz", "task.json", "model_shapes.json",
                  EXPERIENCE_FILE)
CLEAN_VCHECK_FILE = "clean_vcheck.npz"
BLIND_PARAMS_FILE = "model.npz"  # in the session's private directory: the weights a LEARN_BLIND program run gets
BLIND_EXPERIENCE_FILE = EXPERIENCE_FILE  # likewise: the whole LEARN experience a LEARN_BLIND program run gets
RESERVED_NAMES = frozenset({*STANDARD_FILES, "private.json"})
INPUT_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class TrialError(RuntimeError):
    """The instance files are inconsistent, so no trial view can be assembled."""


def lifted_limits(limits: Limits) -> Limits:
    """``limits`` without the per-tensor rel-Frob limit (``max_rel_frob`` None); every other field is kept."""
    if limits.max_rel_frob is None:
        raise TrialError("the rel-Frob limit is already lifted")
    return dataclasses.replace(limits, max_rel_frob=None)


LOOP_EXPERIENCE_SOURCE = "loop_stream_experience_v1"


@dataclass(frozen=True)
class StudentOverride:
    """A program-arm trial on another student than the instance's base model (PREREG E4-loop,
    ``weightcode.harness.loop``): the student's parameters (same layout as the base), its experience arrays and host
    record, the experience keys, and the rewrite of the LEARN task.json for that round."""

    params: Mapping[str, np.ndarray]
    experience: Mapping[str, np.ndarray]
    record: Mapping[str, Any]
    keys: tuple[str, ...]
    task: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class TrialView:
    """The assembled view of one arm and the private files the oracle uses."""

    arm: str
    trial_dir: Path
    files: dict[str, str]
    shown: str
    shown_model_sha256: str
    real_model_sha256: str
    decoy_model_sha256: str | None  # None for a fewshot-v1, seqskill-v1 or graphnew-v1 instance (no decoy)
    apply_mode: str
    limits: dict[str, Any]
    task: dict[str, Any]
    private_forward_path: Path
    private_files: dict[str, str]
    input_files: tuple[str, ...]
    tensor_shapes: dict[str, list[int]]
    n_parameters: int
    classes: list[str]
    evidence_path: Path | None
    evidence_content_sha256: str | None
    evidence_source: str | None = None
    evidence_source_path: Path | None = None
    notes: list[str] = field(default_factory=list)
    experience: dict[str, Any] | None = None
    clean_check: dict[str, Any] | None = None
    fewshot: dict[str, Any] | None = None
    seqskill: dict[str, Any] | None = None
    lift_rel_frob: bool = False
    graphnew: dict[str, Any] | None = None
    fscil: dict[str, Any] | None = None
    blind: dict[str, Any] | None = None  # LEARN_BLIND: the trial experience and the private program inputs
    max_checks: int | None = None  # the lowered patch-check budget of task.json, or None
    e11_rule: dict[str, Any] | None = None  # the E11 rule view (PREREG K1): the stated rule and its provenance
    e11_view: dict[str, Any] | None = None  # an information-ladder view (module docstring)
    policyskill: dict[str, Any] | None = None  # PREREG E14: what the ledger needs to build the rollout check

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready record for session.json."""
        return {
            "arm": self.arm,
            "trial_dir": str(self.trial_dir),
            "files_sha256": dict(self.files),
            "shown": self.shown,
            "shown_model_sha256": self.shown_model_sha256,
            "real_model_sha256": self.real_model_sha256,
            "decoy_model_sha256": self.decoy_model_sha256,
            "apply_mode": self.apply_mode,
            "limits": dict(self.limits),
            "private_forward_path": str(self.private_forward_path),
            "private_files_sha256": dict(self.private_files),
            "input_files": list(self.input_files),
            "n_parameters": self.n_parameters,
            "evidence_path": None if self.evidence_path is None else str(self.evidence_path),
            "evidence_content_sha256": self.evidence_content_sha256,
            "evidence_source": self.evidence_source,
            "evidence_source_path": None if self.evidence_source_path is None else str(self.evidence_source_path),
            "notes": list(self.notes),
            **({} if self.experience is None else {"experience": dict(self.experience)}),
            **({} if self.clean_check is None else {"clean_check": dict(self.clean_check)}),
            **({} if self.fewshot is None else {"fewshot": dict(self.fewshot)}),
            **({} if self.seqskill is None else {"seqskill": dict(self.seqskill)}),
            **({"lift_rel_frob": True} if self.lift_rel_frob else {}),
            **({} if self.graphnew is None else {"graphnew": dict(self.graphnew)}),
            **({} if self.fscil is None else {"fscil": dict(self.fscil)}),
            **({} if self.blind is None else {"blind": dict(self.blind)}),
            **({} if self.max_checks is None else {"max_checks": self.max_checks}),
            **({} if self.e11_rule is None else {"e11_rule": dict(self.e11_rule)}),
            **({} if self.e11_view is None else {"e11_view": dict(self.e11_view)}),
            **({} if self.policyskill is None else {"policyskill": dict(self.policyskill)}),
        }


def _replace_evidence_sentence(statement: str, replacement: str | None, what: str) -> str:
    sentences = [part for part in _SENTENCE_SPLIT.split(statement.strip()) if part]
    found = [index for index, sentence in enumerate(sentences) if "evidence" in sentence.lower()]
    if len(found) != 1:
        raise TrialError(
            f"cannot derive the {what} statement: expected exactly one sentence of task.json 'statement' to "
            f"mention the evidence file, found {len(found)}"
        )
    if replacement is None:
        del sentences[found[0]]
    else:
        sentences[found[0]] = replacement
    return " ".join(sentences)


def statement_without_evidence(statement: str) -> str:
    """The task statement minus its one sentence about the evidence file (the M1 view has no evidence file)."""
    return _replace_evidence_sentence(statement, None, "no-evidence")


def statement_unlabelled(statement: str) -> str:
    """The task statement with its one evidence sentence replaced by ``UNLABELLED_SENTENCE`` (the M2NL view)."""
    return _replace_evidence_sentence(statement, UNLABELLED_SENTENCE, "unlabelled-evidence")


def unlabelled_evidence_section(section: Mapping[str, Any]) -> dict[str, Any]:
    """The task.json evidence section of the M2NL view: the random part without labels, and its ids.

    The instance's section must list the evidence-v1 keys, so the derived section describes the file actually written.
    """
    if not isinstance(section, Mapping):
        raise TrialError("task.json evidence must be an object")
    for part, keys in UNLABELLED_EVIDENCE_SECTION_KEYS.items():
        entry = section.get(part)
        if not isinstance(entry, Mapping) or entry.get("keys") != keys:
            raise TrialError(f"task.json evidence.{part} does not list the evidence-v1 keys {keys}")
    return {
        "random": {"keys": ["x_rand", "pred_rand"], "count": section["random"]["count"],
                   "description": UNLABELLED_RANDOM_DESCRIPTION},
        "ids": {"keys": ["split_ids"], "description": UNLABELLED_IDS_DESCRIPTION},
    }


def unlabelled_evidence(evidence: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The random part of an evidence-v1 sample without labels: ``x_rand``, ``pred_rand`` and the random rows' ids.

    ``split_ids`` lists the fail rows first, then the random rows (``weightcode.evalkit.evidence``), so the random rows' ids
    are the last ``len(x_rand)`` entries.
    """
    n_fail, n_rand = len(evidence["x_fail"]), len(evidence["x_rand"])
    ids = np.asarray(evidence["split_ids"])
    if len(ids) != n_fail + n_rand or len(evidence["pred_rand"]) != n_rand:
        raise TrialError(f"evidence has {len(ids)} split_ids and {len(evidence['pred_rand'])} pred_rand rows for "
                         f"{n_fail} fail and {n_rand} random rows")
    return {"x_rand": np.ascontiguousarray(evidence["x_rand"]),
            "pred_rand": np.ascontiguousarray(evidence["pred_rand"]),
            "split_ids": np.ascontiguousarray(ids[n_fail:])}


def arm_task(task: Mapping[str, Any], arm: str, shown_sha256: str, n_old: int | None = None,
             blind_predictions: bool = False) -> dict[str, Any]:
    """The task.json written into the arm's trial view (``n_old``: the replay row count of a program arm;
    ``blind_predictions``: a LEARN_BLIND view that keeps the student's predictions in its experience)."""
    spec = arm_spec(arm)
    result = copy.deepcopy(dict(task))
    if "parent_sha256" in result:
        raise TrialError("the instance task.json already has parent_sha256; the harness sets it per arm")
    if blind_predictions and not spec.blind:
        raise TrialError(f"blind_predictions is for the LEARN_BLIND view, not {arm}")
    if spec.program:
        evidence = task.get("evidence")
        if not isinstance(evidence, Mapping):
            raise TrialError("task.json has no evidence section to give the LEARN experience counts")
        if n_old is None:
            raise TrialError("the LEARN view needs the replay row count")
        n_fail, n_rand = int(evidence["fail"]["count"]), int(evidence["random"]["count"])
        limits = applied_limits(task["limits"], arm)
        if spec.blind:
            return learn_task_blind(task, shown_sha256, limits, n_fail + n_rand, n_fail, n_old, blind_predictions)
        return learn_task(task, shown_sha256, limits, n_fail + n_rand, n_fail, n_old)
    result["parent_sha256"] = shown_sha256
    if not spec.evidence:
        if "evidence" not in result:
            raise TrialError("task.json has no evidence section to remove for the no-evidence view")
        del result["evidence"]
        result["statement"] = statement_without_evidence(str(result["statement"]))
    elif not spec.labels:
        if "evidence" not in result:
            raise TrialError("task.json has no evidence section for the unlabelled view to replace")
        result["evidence"] = unlabelled_evidence_section(result["evidence"])
        result["statement"] = statement_unlabelled(str(result["statement"]))
    if not spec.checks:
        if "patch_check" not in result or "final_selection" in result:
            raise TrialError("the view without patch_check needs task.json's patch_check section and no "
                             "final_selection key, which the harness sets")
        del result["patch_check"]
        result["final_selection"] = DECLARED_FINAL_SELECTION
    if not spec.size_limits:
        result["limits"] = limits_dict(applied_limits(task["limits"], arm))
    return result


def _check_layout(reference: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray], what: str) -> None:
    if set(reference) != set(other):
        raise TrialError(f"{what}: keys differ from the real model: {sorted(set(reference) ^ set(other))}")
    for key, value in reference.items():
        if other[key].shape != value.shape or other[key].dtype != value.dtype:
            raise TrialError(f"{what}: {key} is {other[key].dtype}{other[key].shape}, real {value.dtype}{value.shape}")


def _ordered(params: Mapping[str, np.ndarray], order: list[str]) -> Params:
    return {key: np.ascontiguousarray(params[key]) for key in order}


def _regular_file(path: Path, what: str) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise TrialError(f"{what} {path.name} is missing from the instance") from None
    if not stat.S_ISREG(info.st_mode):
        raise TrialError(f"{what} {path.name} is not a regular file")


def input_file_names(spec: Mapping[str, Any]) -> tuple[str, ...]:
    """``model_spec.json["input_files"]``, validated: bare, distinct names that no standard file uses."""
    names = spec.get("input_files", [])
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise TrialError("model_spec.json input_files must be a list of file names")
    for name in names:
        if INPUT_FILE_RE.fullmatch(name) is None or name in RESERVED_NAMES:
            raise TrialError(f"model_spec.json input_files has an invalid name {name!r}")
    if len(set(names)) != len(names):
        raise TrialError(f"model_spec.json input_files repeats a name: {names}")
    return tuple(names)


def load_decoy(private: Mapping[str, Any], real: Mapping[str, np.ndarray]) -> Params:
    """The decoy weights from ``private.json["decoy_model_path"]``, hash- and layout-checked."""
    decoy = load_params(Path(str(private["decoy_model_path"])))
    digest = params_sha256(decoy)
    if digest != private["decoy_model_sha256"]:
        raise TrialError(f"decoy model hash {digest} != private.json decoy_model_sha256")
    if digest == private["model_sha256"]:
        raise TrialError("the decoy model is the real model")
    _check_layout(real, decoy, "decoy model")
    return decoy


def decoy_evidence_source(instance_dir: str | Path) -> tuple[Path, str]:
    """The DECOY view's evidence file and content hash (``weightcode.evalkit.instances.decoy_evidence``); ``TrialError``
    when it is not available, so a DECOY view never falls back to the real model's evidence."""
    try:
        return decoy_evidence(instance_dir)
    except (InstanceConflict, OSError, KeyError, ValueError) as exc:
        raise TrialError(f"no DECOY view: the evidence drawn from the decoy model's own errors is not available: "
                         f"{exc}") from None


def check_evidence_predictions(forward: ForwardFn, params: Mapping[str, np.ndarray],
                               evidence: Mapping[str, np.ndarray], what: str,
                               parts: tuple[str, ...] = EVIDENCE_PARTS) -> None:
    """``pred_<part>`` must be the predictions of ``params`` (the weights the editor is shown) for every part."""
    for part in parts:
        x, shown = evidence[f"x_{part}"], evidence[f"pred_{part}"]
        if len(x) == 0:
            continue
        pred = predict(forward, params, x, PREDICT_BATCH)
        disagree = int(np.count_nonzero(pred != shown))
        if disagree:
            raise TrialError(f"{what}: pred_{part} disagrees with the shown weights on {disagree} of {len(x)} rows")


def zero_params(real: Mapping[str, np.ndarray]) -> Params:
    """All-zero parameters with the real model's keys, shapes and dtypes."""
    return {key: np.zeros_like(value) for key, value in real.items()}


def check_max_checks(max_checks: int | None, arm: str) -> None:
    """``max_checks`` is None, or a positive int in an arm with ``patch_check``; anything else is a ``TrialError``."""
    if max_checks is None:
        return
    if isinstance(max_checks, bool) or not isinstance(max_checks, int) or max_checks < 1:
        raise TrialError(f"max_checks must be a positive int, got {max_checks!r}")
    if not arm_spec(arm).checks:
        raise TrialError(f"max_checks lowers the patch-check budget; {arm} has no patch_check")


def assemble(instance_dir: str | Path, arm: str, session_dir: str | Path,
             override: StudentOverride | None = None, seqskill_variant: str | None = None,
             lift_rel_frob: bool = False, fscil_kept_rule: Mapping[str, Any] | None = None,
             max_checks: int | None = None, blind_predictions: bool = False,
             e11_variant: str | None = None) -> TrialView:
    """Write the trial view of ``arm`` into ``<session_dir>/trial`` and the private forward copy; return its record.

    ``override`` (a program arm on an evidence-v1 instance only) shows and uses its student instead of the instance's
    base model, with its experience and task.json rewrite; everything else is assembled as without it.

    ``seqskill_variant`` (a seqskill-v1 instance only): None or "standard" is the PREREG E5 view, "rule_only" the
    PREREG E8 view (module docstring). ``lift_rel_frob`` (a seqskill-v1 instance in the LEARN arm only) assembles the
    view under ``lifted_limits`` (module docstring). ``fscil_kept_rule`` (an fscil-v1 instance only), a
    ``weightcode.harness.fscil.kept_rule_v2`` record, gives the task.json the E9 v2 readout and final selection
    (``weightcode.harness.fscil.learn_task_fscil``); None is E9's view, unchanged.

    ``max_checks`` (an arm with ``patch_check``, at most the instance's budget) lowers task.json
    ``patch_check.budget_queries`` (PREREG A4b); ``blind_predictions`` (LEARN_BLIND only) keeps ``pred_new`` and
    ``pred_old`` in the blind trial's experience (module docstring). Their defaults leave the view as without them.

    ``e11_variant`` (an E11 instance in the LEARN arm, without ``override``): None or "standard" is the E11 view,
    unchanged; "rule" the PREREG K1 rule view; "unlabelled", "named" and "named_nocheck" the information ladder's U,
    K0b and K0a views (module docstring), which take no ``max_checks``.
    """
    spec = arm_spec(arm)
    try:
        e11_name = e11.check_variant(e11_variant)
    except ValueError as exc:
        raise TrialError(str(exc)) from None
    e11_rule_view = e11_name == e11.RULE
    ladder = e11_name if e11_name in e11.LADDER_VARIANTS else None
    named = ladder in e11.NAMED_VARIANTS
    if e11_rule_view and (arm != "LEARN" or override is not None):
        raise TrialError(f"the E11 rule view is assembled in the LEARN arm on the instance's own student, not {arm}"
                         f"{' with a student override (a loop round)' if override is not None else ''}")
    if ladder is not None and (arm != "LEARN" or override is not None or max_checks is not None):
        overridden = " with a student override (a loop round)" if override is not None else ""
        raise TrialError(f"the E11 {ladder} view is assembled in the LEARN arm on the instance's own student with "
                         f"E11's checks, not {arm}{overridden}{' with max_checks' if max_checks is not None else ''}")
    try:
        variant = seqskill.check_variant(seqskill_variant)
    except seqskill.SeqskillHarnessError as exc:
        raise TrialError(str(exc)) from None
    rule_only = variant == seqskill.RULE_ONLY
    scan_view = variant == seqskill.SCAN
    if not isinstance(lift_rel_frob, bool):
        raise TrialError(f"lift_rel_frob must be a bool, got {lift_rel_frob!r}")
    if lift_rel_frob and arm != "LEARN":
        raise TrialError(f"the rel-Frob limit is lifted in the LEARN arm only, not {arm}")
    check_max_checks(max_checks, arm)
    if not isinstance(blind_predictions, bool):
        raise TrialError(f"blind_predictions must be a bool, got {blind_predictions!r}")
    if blind_predictions and not spec.blind:
        raise TrialError(f"blind_predictions is for the LEARN_BLIND view, not {arm}")
    if spec.blind and override is not None:
        raise TrialError(f"{arm} is assembled on the instance's own student; it takes no student override")
    instance_dir = Path(instance_dir)
    session_dir = Path(session_dir)
    trial_dir = session_dir / "trial"
    private_dir = session_dir / "private"
    if trial_dir.exists():
        raise TrialError(f"{trial_dir} already exists")
    if not private_dir.is_dir():
        raise TrialError(f"{private_dir} must exist before assembly")

    instance = load_instance(instance_dir)  # hash-checks model.npz and forward.py against private.json
    task, private = instance.task, instance.private
    few = fewshot.is_fewshot(private)  # PREREG E2: no decoy, no condition, the experience is the evidence file
    seq = seqskill.is_seqskill(private)  # PREREG E5: the same, with token rows and the V_check cut by part
    gnew = graphnew.is_graphnew(private)  # PREREG E7: fewshot's view on node ids of one graph, graph.npz an input
    fsc = fscil.is_fscil(private)  # PREREG E9: fewshot's view of one session, three experience parts, the session mask
    pol = policyskill.is_policyskill(private)  # PREREG E14: seqskill's view for a policy, checked by rollouts
    own = few or seq or gnew or fsc or pol  # the evidence file is the experience; no decoy and no clean companion
    if fscil_kept_rule is not None and not fsc:
        raise TrialError("an E9 v2 kept rule is for an fscil-v1 instance")
    if spec.blind and own:
        raise TrialError(f"{arm} is assembled on an evidence-v1 instance only, not {instance.instance_id}")
    if few:
        if not spec.program:
            raise TrialError(f"a fewshot-v1 instance is assembled for a program arm only, not {arm}")
        try:
            fewshot.check_current(instance_dir)  # task.json, model_spec.json and evidence.npz as the code writes them
        except fewshot.stale_error() as exc:
            raise TrialError(f"the fewshot instance is not current: {exc}") from None
    elif seq:
        if not spec.program:
            raise TrialError(f"a seqskill-v1 instance is assembled for a program arm only, not {arm}")
        try:
            seqskill.check_current(instance_dir)  # task.json, model_spec.json and evidence.npz as the code writes them
            seqskill.check_instance_variant(private, variant)  # scan_rule on a scanskill-v1 instance and no other
        except seqskill.stale_error() as exc:
            raise TrialError(f"the seqskill instance is not current: {exc}") from None
        except seqskill.SeqskillHarnessError as exc:
            raise TrialError(str(exc)) from None
        if scan_view and not lift_rel_frob:
            raise TrialError("a scan_rule session runs with the rel-Frob limit lifted (PREREG E13); pass "
                             "lift_rel_frob")
    elif gnew:
        if not spec.program:
            raise TrialError(f"a graphnew-v1 instance is assembled for a program arm only, not {arm}")
        try:
            graphnew.check_current(instance_dir)  # task.json, model_spec.json, graph.npz and evidence.npz as written
        except graphnew.stale_error() as exc:
            raise TrialError(f"the graphnew instance is not current: {exc}") from None
    elif fsc:
        if not spec.program:
            raise TrialError(f"an fscil-v1 instance is assembled for a program arm only, not {arm}")
        try:
            fscil.check_current(instance_dir)  # task.json, model_spec.json and evidence.npz as the code writes them
        except fscil.stale_error() as exc:
            raise TrialError(f"the fscil instance is not current: {exc}") from None
    elif pol:
        if not spec.program:
            raise TrialError(f"a policyskill-v1 instance is assembled for a program arm only, not {arm}")
        try:
            policyskill.check_current(instance_dir)  # task.json, model_spec.json and evidence.npz as the code writes
        except policyskill.stale_error() as exc:
            raise TrialError(f"the policyskill instance is not current: {exc}") from None
    else:
        refuse_v0(instance.instance_id, task, private)
    if (rule_only or scan_view) and not seq:
        raise TrialError(f"the {variant} view is defined for a seqskill-v1 or scanskill-v1 instance, not "
                         f"{instance.instance_id}")
    if lift_rel_frob and not seq:
        raise TrialError(f"the rel-Frob limit is lifted for a seqskill-v1 instance only, not {instance.instance_id}")
    stated_rule: dict[str, Any] | None = None
    if e11_rule_view:  # PREREG K1: an E11 instance whose condition record carries the rule the view states
        if own:
            raise TrialError(f"the E11 rule view is for an E11 evidence-v1 instance, not {instance.instance_id}")
        try:
            stated_rule = e11.condition_rule(private)
        except (e11.E11HarnessError, OSError, ValueError) as exc:
            raise TrialError(f"no E11 rule view of {instance.instance_id}: {exc}") from None
    ladder_record: dict[str, Any] | None = None
    if ladder is not None:  # the information ladder: the one PREREG instance
        if own:
            raise TrialError(f"the E11 {ladder} view is for an E11 evidence-v1 instance, not {instance.instance_id}")
        try:
            ladder_record = e11.ladder_view(private, ladder)
        except e11.E11HarnessError as exc:
            raise TrialError(f"no E11 {ladder} view of {instance.instance_id}: {exc}") from None
    required = (fewshot.REQUIRED_PRIVATE_FIELDS if few else seqskill.required_fields(private) if seq
                else graphnew.REQUIRED_PRIVATE_FIELDS if gnew else fscil.REQUIRED_PRIVATE_FIELDS if fsc
                else policyskill.REQUIRED_PRIVATE_FIELDS if pol else REQUIRED_PRIVATE_FIELDS)
    missing = [name for name in required if name not in private]
    if missing:
        raise TrialError(f"private.json lacks {missing}")
    if file_sha256(instance_dir / "task.json") != private["task_sha256"]:
        raise TrialError("task.json bytes differ from private.json task_sha256")
    if file_sha256(instance_dir / "model_spec.json") != private["model_spec_sha256"]:
        raise TrialError("model_spec.json bytes differ from private.json model_spec_sha256")
    # the fewshot and seqskill evidence hashes (weightcode.evalkit) were checked by check_current, and again by experience
    evidence_sha = private["evidence_content_sha256"] if own else evidence_content_sha256(instance.evidence)
    if evidence_sha != private["evidence_content_sha256"]:
        raise TrialError("evidence.npz content differs from private.json evidence_content_sha256")

    real = instance.params
    order = list(real)
    real_sha = instance.params_sha256
    arm_limits = applied_limits(task["limits"], arm, len(real), dense_delta_bytes(v.shape for v in real.values()))
    if lift_rel_frob:  # the PREREG E8 amendment: no per-tensor rel-Frob limit, everything else as applied
        arm_limits = lifted_limits(arm_limits)
    if override is not None:
        if not spec.program or own:
            raise TrialError("a student override is for a program arm on an evidence-v1 instance")
        _check_layout(real, override.params, "the override student")
        real = _ordered(override.params, order)
        real_sha = params_sha256(real)  # may be the base model: a loop round after rounds that all kept the base
    decoy_sha = None if own else str(private["decoy_model_sha256"])
    model_spec = read_json(instance_dir / "model_spec.json")
    if not isinstance(model_spec, dict):
        raise TrialError("model_spec.json is not a JSON object")
    new_rule: dict[str, Any] | None = None
    if rule_only:  # PREREG E8: the new operator's rule, from the condition record whose hash private.json records
        try:
            new_rule = seqskill.new_operator_rule(private)
            model_spec = seqskill.rule_only_model_spec(model_spec, private, new_rule)
        except (seqskill.SeqskillHarnessError, OSError, ValueError) as exc:
            raise TrialError(f"no rule_only view of {instance.instance_id}: {exc}") from None
    if "classes" in model_spec and model_spec["classes"] != task.get("classes"):
        raise TrialError("model_spec.json classes differ from task.json classes")
    inputs = input_file_names(model_spec)
    if gnew and inputs != (graphnew.GRAPH_FILE,):
        raise TrialError(f"a graphnew-v1 model_spec.json lists input_files {list(inputs)}, not [{graphnew.GRAPH_FILE!r}]")
    if gnew:
        try:
            graph_limits = graphnew.program_limits(task["limits"], len(real))  # PREREG E7: the evalkit rule's limits
            head = graphnew.head_tensors(model_spec)
        except (graphnew.GraphnewHarnessError, graphnew.stale_error()) as exc:
            raise TrialError(f"no LEARN limits for {instance.instance_id}: {exc}") from None
        if list(graph_limits.rel_frob_exempt) != head or list(private["head_tensors"]) != head:
            raise TrialError(f"the head {head} is not the rel-Frob exemption {list(graph_limits.rel_frob_exempt)} and "
                             f"private.json head_tensors {private['head_tensors']}")
    if fsc:
        if inputs:
            raise TrialError(f"an fscil-v1 model_spec.json lists input_files {list(inputs)}; it has none")
        try:
            fscil_limits = fscil.program_limits(task["limits"], len(real))  # PREREG E9: max_ops is the tensor count
            head = fscil.head_tensors(model_spec)
        except (fscil.FscilHarnessError, fscil.stale_error()) as exc:
            raise TrialError(f"no LEARN limits for {instance.instance_id}: {exc}") from None
        if list(fscil_limits.rel_frob_exempt) != head or list(private["head_tensors"]) != head:
            raise TrialError(f"the head {head} is not the rel-Frob exemption {list(fscil_limits.rel_frob_exempt)} and "
                             f"private.json head_tensors {private['head_tensors']}")
    evidence_source: str | None = None
    evidence_source_path: Path | None = None
    shown_evidence_sha: str | None = None
    unlabelled: dict[str, np.ndarray] | None = None
    experience: dict[str, np.ndarray] | None = None
    experience_record: dict[str, Any] | None = None
    program_experience: dict[str, np.ndarray] | None = None  # LEARN_BLIND: what a program run gets
    if spec.program:
        if few:
            experience, experience_record = fewshot.experience(instance)
            evidence_source = fewshot.EXPERIENCE_SOURCE
        elif gnew:
            experience, experience_record = graphnew.experience(instance)
            evidence_source = graphnew.EXPERIENCE_SOURCE
        elif fsc:
            try:
                experience, experience_record = fscil.experience(instance)
            except (fscil.FscilHarnessError, fscil.stale_error()) as exc:
                raise TrialError(f"no LEARN experience for {instance.instance_id}: {exc}") from None
            evidence_source = fscil.EXPERIENCE_SOURCE
        elif pol:
            try:
                experience, experience_record = policyskill.experience(instance)
            except (policyskill.PolicyskillHarnessError, policyskill.stale_error()) as exc:
                raise TrialError(f"no LEARN experience for {instance.instance_id}: {exc}") from None
            evidence_source = policyskill.EXPERIENCE_SOURCE
        elif seq:
            experience, experience_record = seqskill.experience(instance)
            evidence_source = seqskill.SCAN_EXPERIENCE_SOURCE if scan_view else seqskill.EXPERIENCE_SOURCE
            if rule_only:  # PREREG E8: the old part alone, cut from the checked evidence
                experience, experience_record = seqskill.rule_only_experience(experience, experience_record)
                evidence_source = seqskill.RULE_ONLY_EXPERIENCE_SOURCE
        elif override is not None:
            experience, experience_record = dict(override.experience), dict(override.record)
            evidence_source = LOOP_EXPERIENCE_SOURCE
            parts = tuple(part for part in ("new", "old", "prev") if f"x_{part}" in experience)
            check_evidence_predictions(instance.forward, real, experience, "the override experience", parts)
        else:
            experience, experience_record = build_experience(instance)
            evidence_source = EXPERIENCE_SOURCE
            if e11_rule_view:  # PREREG K1: the old part alone, cut from the checked experience
                try:
                    experience, experience_record = rule_experience(experience, experience_record)
                except ExperienceError as exc:
                    raise TrialError(f"no E11 rule view of {instance.instance_id}: {exc}") from None
                evidence_source = RULE_EXPERIENCE_SOURCE
            elif ladder is not None:  # the information ladder: cut from the checked experience
                try:
                    experience, experience_record = (named_experience(experience_record, ladder) if named
                                                     else unlabelled_experience(experience, experience_record))
                except ExperienceError as exc:
                    raise TrialError(f"no E11 {ladder} view of {instance.instance_id}: {exc}") from None
                evidence_source = NAMED_EXPERIENCE_SOURCE if named else UNLABELLED_EXPERIENCE_SOURCE
        shown_evidence_sha = experience_record["content_sha256"]
        evidence_source_path = None if override is not None or named else instance_dir / "evidence.npz"
        if spec.blind:  # the trial shows the rows and labels; a program run still gets the whole experience
            program_experience = experience
            experience = blind_trial_experience(program_experience, blind_predictions)
            shown_evidence_sha = params_sha256(experience)
    elif spec.evidence and not spec.labels:
        if spec.shown != SHOWN_REAL:  # pragma: no cover - ArmSpec values are fixed in arms.py
            raise TrialError("unlabelled evidence is defined for the real weights only")
        unlabelled = unlabelled_evidence(instance.evidence)
        evidence_source, shown_evidence_sha = EVIDENCE_FROM_INSTANCE_RANDOM, params_sha256(unlabelled)
        evidence_source_path = instance_dir / "evidence.npz"
    elif spec.evidence and spec.shown == SHOWN_DECOY:
        evidence_source = EVIDENCE_FROM_DECOY
        evidence_source_path, shown_evidence_sha = decoy_evidence_source(instance_dir)
        if shown_evidence_sha == evidence_sha:
            raise TrialError("the decoy model's evidence is the real model's evidence")
    elif spec.evidence:
        evidence_source, shown_evidence_sha = EVIDENCE_FROM_INSTANCE, evidence_sha
        evidence_source_path = instance_dir / "evidence.npz"

    if spec.shown == SHOWN_REAL or spec.shown == SHOWN_SHAPES:
        shown_params, shown_sha = real, real_sha
    elif spec.shown == SHOWN_DECOY:
        shown_params = _ordered(load_decoy(private, real), order)
        shown_sha = decoy_sha
    elif spec.shown == SHOWN_ZEROS:
        shown_params = zero_params(real)
        shown_sha = params_sha256(shown_params)
        if shown_sha in (real_sha, decoy_sha):
            raise TrialError("the zero model hashes like a trained model")
    else:  # pragma: no cover - ArmSpec values are fixed above
        raise TrialError(f"unknown shown model {spec.shown!r}")

    trial_dir.mkdir(mode=0o700)
    files: dict[str, str] = {}

    def write_text(name: str, text: str) -> None:
        (trial_dir / name).write_text(text, encoding="utf-8")
        files[name] = file_sha256(trial_dir / name)

    visible_spec = {key: value for key, value in model_spec.items() if key != "model_id"}
    write_text("model_spec.json", json_text(visible_spec))
    shutil.copyfile(instance_dir / "forward.py", trial_dir / "forward.py")
    files["forward.py"] = file_sha256(trial_dir / "forward.py")
    if files["forward.py"] != private["forward_sha256"]:
        raise TrialError("trial forward.py copy differs from private.json forward_sha256")
    if few and experience is not None and experience_record is not None:
        trial_task = learn_task_fewshot(task, shown_sha,
                                        applied_limits(task["limits"], arm, len(real),
                                                       dense_delta_bytes(v.shape for v in real.values())),
                                        experience["y_new"], experience_record["n_old"])
    elif gnew and experience is not None and experience_record is not None:
        trial_task = learn_task_graphnew(task, shown_sha, graph_limits, experience["y_new"], experience_record["n_old"])
    elif fsc and experience_record is not None:
        trial_task = fscil.learn_task_fscil(task, shown_sha, fscil_limits, experience_record, fscil_kept_rule)
    elif seq and scan_view and experience_record is not None:
        trial_task = learn_task_scan(task, shown_sha, arm_limits, experience_record)
    elif seq and experience_record is not None:
        trial_task = learn_task_seqskill(task, shown_sha, arm_limits, experience_record, variant)
    elif pol and experience_record is not None:
        try:
            trial_task = learn_task_policyskill(task, shown_sha, arm_limits, experience_record)
        except ExperienceError as exc:
            raise TrialError(f"no LEARN task view of {instance.instance_id}: {exc}") from None
    else:
        trial_task = arm_task(task, arm, shown_sha, None if experience_record is None else experience_record["n_old"],
                              blind_predictions=blind_predictions)
        if stated_rule is not None and experience_record is not None:  # PREREG K1: the rule in place of the rows
            try:
                trial_task = learn_task_rule(trial_task, str(stated_rule["sentence"]), int(experience_record["n_old"]))
            except ExperienceError as exc:
                raise TrialError(f"no E11 rule view of {instance.instance_id}: {exc}") from None
        if ladder_record is not None and experience_record is not None:  # the information ladder
            try:
                trial_task = (learn_task_named(trial_task, str(ladder_record["capability_sentence"]),
                                               checks=ladder == e11.NAMED) if named
                              else learn_task_unlabelled(trial_task))
            except ExperienceError as exc:
                raise TrialError(f"no E11 {ladder} view of {instance.instance_id}: {exc}") from None
    if override is not None:
        trial_task = override.task(trial_task)
        if trial_task.get("parent_sha256") != shown_sha:
            raise TrialError("the override task.json does not name the override student as its parent")
    if max_checks is not None:  # PREREG A4b: the arm's own task.json, with a lower patch-check budget
        check = trial_task.get("patch_check")
        if not isinstance(check, dict) or not isinstance(check.get("budget_queries"), int):
            raise TrialError("max_checks needs a task.json patch_check section with an int budget_queries")
        if max_checks > check["budget_queries"]:
            raise TrialError(f"max_checks {max_checks} is above the task's patch-check budget of "
                             f"{check['budget_queries']}; it can only lower it")
        check["budget_queries"] = max_checks
    if experience_record is not None:
        counts = trial_task.get("experience", {})
        if named:  # no experience section and no row
            described = ("experience" in trial_task, experience_record["n_new"], experience_record["n_old"])
            expected = (False, 0, 0)
        elif few or gnew:
            described = (counts["new"]["count"], counts["new"]["per_class"], counts["old"]["count"])
            expected = tuple(experience_record[key] for key in ("n_new", "per_class", "n_old"))
        elif fsc:
            described = (counts["new"]["count"], counts["new"]["per_class"], counts["old"]["count"],
                         counts["prev"]["count"])
            expected = tuple(experience_record[key] for key in ("n_new", "per_class", "n_old", "n_prev"))
        elif seq and scan_view:  # the primitive part (when the condition has one) and the old part, no new part
            described = ("new" in counts, counts["prim"]["count"] if "prim" in counts else 0, counts["old"]["count"])
            expected = (False, experience_record["n_prim"], experience_record["n_old"])
        elif (seq and rule_only) or e11_rule_view:  # the old part alone: no new section, no new row
            described = ("new" in counts, counts["old"]["count"], experience_record["n_new"])
            expected = (False, experience_record["n_old"], 0)
        elif seq:
            described = (counts["new"]["count"], counts["old"]["count"])
            expected = tuple(experience_record[key] for key in ("n_new", "n_old"))
        elif pol:
            described = (counts["new"]["count"], counts["new"]["demonstrations"], counts["old"]["count"])
            expected = tuple(experience_record[key] for key in ("n_new", "n_new_demonstrations", "n_old"))
        elif ladder == e11.UNLABELLED:  # U: the selection of the rows is not stated (no n_from_errors)
            described = (counts["new"]["count"], "n_from_errors" in counts["new"], counts["old"]["count"])
            expected = (experience_record["n_new"], False, experience_record["n_old"])
        else:
            described = (counts["new"]["count"], counts["new"]["n_from_errors"], counts["old"]["count"])
            expected = tuple(experience_record[key] for key in ("n_new", "n_new_fail", "n_old"))
        if described != expected:
            raise TrialError(f"the LEARN task.json counts {described} differ from the experience rows {expected}")
    write_text("task.json", json_text(trial_task))
    if spec.shown == SHOWN_SHAPES:
        shapes = {
            "schema": SHAPES_SCHEMA,
            "n_parameters": int(sum(value.size for value in real.values())),
            "tensors": [{"name": key, "shape": list(real[key].shape), "dtype": str(real[key].dtype)} for key in order],
        }
        write_text("model_shapes.json", json_text(shapes))
    else:
        saved = save_params(trial_dir / "model.npz", _ordered(shown_params, order))
        if saved != shown_sha:
            raise TrialError("the shown model.npz does not hash to the shown model")
        files["model.npz"] = file_sha256(trial_dir / "model.npz")
    evidence_path = None
    clean_check: dict[str, Any] | None = None
    fewshot_record: dict[str, Any] | None = None
    seqskill_record: dict[str, Any] | None = None
    graphnew_record: dict[str, Any] | None = None
    fscil_record: dict[str, Any] | None = None
    policyskill_record: dict[str, Any] | None = None
    blind_record: dict[str, Any] | None = None
    if experience is not None:
        if named and ladder_record is not None:  # no trial experience file: the private empty one (module docstring)
            empty_path = private_dir / NAMED_PROGRAM_EXPERIENCE_FILE
            np.savez(empty_path)
            with np.load(empty_path, allow_pickle=False) as archive:
                copied = {key: archive[key] for key in archive.files}
            ladder_record.update(program_experience_path=str(empty_path),
                                 program_experience_file_sha256=file_sha256(empty_path), program_experience_keys=[])
        else:
            np.savez(trial_dir / EXPERIENCE_FILE, **experience)
            files[EXPERIENCE_FILE] = file_sha256(trial_dir / EXPERIENCE_FILE)
            with np.load(trial_dir / EXPERIENCE_FILE, allow_pickle=False) as archive:
                copied = {key: archive[key] for key in archive.files}
        if named:  # K0b, K0a: no array
            keys = ()
        elif override is not None:
            keys = override.keys
        elif fsc:  # the three parts of an incremental session
            keys = fscil.EXPERIENCE_KEYS
        elif pol:  # the evidence keys: steps with logits, demonstration and step numbers
            keys = policyskill.EXPERIENCE_KEYS
        elif spec.blind:  # the LEARN keys, without the predictions unless the session keeps them
            keys = tuple(experience)
        elif e11_rule_view:  # PREREG K1: the old part alone
            keys = RULE_EXPERIENCE_KEYS
        elif ladder == e11.UNLABELLED:  # U: without y_new
            keys = UNLABELLED_EXPERIENCE_KEYS
        else:
            keys = (seqskill.experience_keys(private, variant) if scan_view
                    else seqskill.RULE_ONLY_EXPERIENCE_KEYS if rule_only else EXPERIENCE_KEYS)
        if named:  # an empty archive, and nothing shown
            if copied or experience or shown_evidence_sha is not None:
                raise TrialError("the no-data view's program experience is not empty")
        elif sorted(copied) != sorted(keys) or params_sha256(copied) != shown_evidence_sha:
            raise TrialError("the trial experience.npz differs from the assembled experience")
        if few:  # the ledger cuts V_check by class; there is no clean companion to copy
            fewshot_record = fewshot.session_record(private)
        elif seq:  # the ledger cuts V_check by part; there is no clean companion to copy
            seqskill_record = seqskill.session_record(private, variant, new_rule)
        elif gnew:  # the ledger cuts V_check by class; there is no clean companion to copy
            graphnew_record = graphnew.session_record(private)
        elif fsc:  # the ledger cuts V_check and old_check by session; there is no clean companion to copy
            fscil_record = fscil.session_record(private)
        elif pol:  # the ledger reads the V_check seeds by mission; there is no clean companion to copy
            policyskill_record = policyskill.session_record(private)
        else:
            rows, hashes = load_clean_check(instance)
            if len(rows["y"]) != int(private["n_vcheck"]):
                raise TrialError(f"the clean V_check rows number {len(rows['y'])}, V_check {private['n_vcheck']}")
            np.savez(private_dir / CLEAN_VCHECK_FILE, **rows)
            clean_check = {"split": "V_check", "path": str(private_dir / CLEAN_VCHECK_FILE),
                           "sha256": file_sha256(private_dir / CLEAN_VCHECK_FILE), "n": int(len(rows["y"])),
                           **hashes, "source": source_name()}
        if program_experience is not None and experience_record is not None:
            blind_record = _write_blind_inputs(private_dir, _ordered(real, order), real_sha, program_experience,
                                               experience_record, experience, blind_predictions)
    elif evidence_source_path is not None:
        evidence_path = trial_dir / "evidence.npz"
        if unlabelled is None:
            shutil.copyfile(evidence_source_path, evidence_path)
        else:
            np.savez(evidence_path, **unlabelled)
        files["evidence.npz"] = file_sha256(evidence_path)
        with np.load(evidence_path, allow_pickle=False) as archive:
            copied = {key: archive[key] for key in archive.files}
        if unlabelled is None:
            copied_sha = evidence_content_sha256(copied)
        elif sorted(copied) != sorted(UNLABELLED_EVIDENCE_KEYS):
            raise TrialError(f"the unlabelled evidence.npz has keys {sorted(copied)}, not "
                             f"{list(UNLABELLED_EVIDENCE_KEYS)}")
        else:
            copied_sha = params_sha256(copied)
            if len(copied["x_rand"]) != trial_task["evidence"]["random"]["count"]:
                raise TrialError("the unlabelled evidence.npz row count differs from its task.json count")
        if copied_sha != shown_evidence_sha:
            raise TrialError(f"trial evidence.npz copy differs from its source ({evidence_source})")
        # pred_* describe the decoy in DECOY and the real model in every other arm (REWRITE shows zeros, NOWEIGHT
        # shapes); in DECOY this is what keeps the evidence consistent with the weights the editor can run.
        described = shown_params if spec.shown == SHOWN_DECOY else real
        parts = EVIDENCE_PARTS if unlabelled is None else ("rand",)
        check_evidence_predictions(instance.forward, described, copied, f"{arm} evidence.npz", parts)

    private_files: dict[str, str] = {}
    shutil.copyfile(instance_dir / "forward.py", private_dir / "forward.py")
    private_files["forward.py"] = file_sha256(private_dir / "forward.py")
    if private_files["forward.py"] != private["forward_sha256"]:
        raise TrialError("private forward.py copy differs from private.json forward_sha256")
    for name in inputs:
        source = instance_dir / name
        _regular_file(source, "input file")
        for target_dir, record in ((trial_dir, files), (private_dir, private_files)):
            shutil.copyfile(source, target_dir / name)
            record[name] = file_sha256(target_dir / name)
        if files[name] != private_files[name]:
            raise TrialError(f"input file {name} changed while it was copied")
        if gnew and files[name] != private["graph_file_sha256"]:  # forward.py reads this copy beside itself
            raise TrialError(f"the {name} copy differs from private.json graph_file_sha256")
    if ladder_record is not None and named:
        private_files[NAMED_PROGRAM_EXPERIENCE_FILE] = str(ladder_record["program_experience_file_sha256"])
    if blind_record is not None:
        private_files[BLIND_PARAMS_FILE] = blind_record["params_file_sha256"]
        private_files[BLIND_EXPERIENCE_FILE] = blind_record["experience_file_sha256"]

    limits = limits_dict(graph_limits if gnew else fscil_limits if fsc else arm_limits)
    return TrialView(
        arm=arm,
        trial_dir=trial_dir,
        files=files,
        shown=spec.shown,
        shown_model_sha256=shown_sha,
        real_model_sha256=real_sha,
        decoy_model_sha256=decoy_sha,
        apply_mode=spec.apply_mode,
        limits=limits,
        task=trial_task,
        private_forward_path=private_dir / "forward.py",
        private_files=private_files,
        input_files=inputs,
        tensor_shapes={key: list(real[key].shape) for key in order},
        n_parameters=int(sum(value.size for value in real.values())),
        classes=list(task.get("classes", [])),
        evidence_path=evidence_path,
        evidence_content_sha256=shown_evidence_sha,
        evidence_source=evidence_source,
        evidence_source_path=evidence_source_path,
        experience=experience_record,
        clean_check=clean_check,
        fewshot=fewshot_record,
        seqskill=seqskill_record,
        lift_rel_frob=lift_rel_frob,
        graphnew=graphnew_record,
        fscil=fscil_record,
        blind=blind_record,
        max_checks=max_checks,
        e11_rule=stated_rule,
        e11_view=ladder_record,
        policyskill=policyskill_record,
    )


def _write_blind_inputs(private_dir: Path, params: Params, params_sha: str, program_experience: Mapping[str, np.ndarray],
                        experience_record: Mapping[str, Any], trial_experience: Mapping[str, np.ndarray],
                        predictions: bool) -> dict[str, Any]:
    """Write a LEARN_BLIND session's program inputs into its private directory: the weights as LEARN's trial
    ``model.npz`` writes them, and the whole LEARN experience; return the view's ``blind`` record."""
    params_path, experience_path = private_dir / BLIND_PARAMS_FILE, private_dir / BLIND_EXPERIENCE_FILE
    if save_params(params_path, params) != params_sha:
        raise TrialError("the private model.npz of a blind view does not hash to the student")
    np.savez(experience_path, **program_experience)
    with np.load(experience_path, allow_pickle=False) as archive:
        written = {key: archive[key] for key in archive.files}
    if sorted(written) != sorted(EXPERIENCE_KEYS) or params_sha256(written) != experience_record["content_sha256"]:
        raise TrialError("the private experience.npz of a blind view differs from the LEARN experience")
    return {
        "schema": "weightcode.harness.blind_view.v1",
        "predictions_shown": predictions,
        "trial_experience_keys": list(trial_experience),
        "trial_experience_content_sha256": params_sha256(trial_experience),
        "params_path": str(params_path),
        "params_file_sha256": file_sha256(params_path),
        "params_sha256": params_sha,
        "experience_path": str(experience_path),
        "experience_file_sha256": file_sha256(experience_path),
        "experience_content_sha256": str(experience_record["content_sha256"]),
    }
