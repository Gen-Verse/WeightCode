"""Shared run plumbing: evidence, the dev/main guard, output layout, the counted check loop and run.json.

A run is one method on one instance with one run seed, written to
``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/[<ref_label>/]<run_seed>/``. All candidates are generated from
editor-visible information before the first patch-check query, so no candidate depends on a V_check readout. Each
valid candidate is written to ``q<slot>/`` (as its parameter delta with ``write_delta_patch``, or as the explicit ops
it carries), re-applied with the shared executor, and checked once; the final patch is ``PatchCheckOracle.final()``
(contract 5.2), and the base model when nothing was checked. The score is S = fixed - broken over all V_check images;
an evidence-v0 instance adds its target-pair readouts.

Joint runs (``joint=True``, PREREG E1: add a capability without replacing one) append the dataset's replay rows
(``weightcode.evalkit.companions.load_replay``: clean train images, rule replay-v1) to the evidence and score every
candidate with the joint oracle (``make_joint_oracle``: the condition's V_check plus its clean companion,
S_joint = S_new + S_old). ``replay_in_fit`` says whether the generator sees the replay rows: the label fits do, the
fixed rules keep their definition on the deployment rows.

Fewshot runs (``fewshot=True``, PREREG E2: add classes a student never trained on) take a fewshot-v1 instance
(``weightcode.evalkit.fewshot``): the evidence is its new-class rows followed by its replay rows of the train classes, the
joint oracle scores the new-class and the train-class rows of V_check, and the limits are
``fewshot_program_limits``, the ones an E2 update program is held to (head exempt from max_rel_frob, at most 20% of
the entries changed, no edit-size limit).

E10 runs (``e10=True`` with ``joint=True``, PREREG E10: E1 on CIFAR-10-C with ResNet-20) are joint runs held to
``weightcode.evalkit.e10.e10_program_limits`` (rel-Frob at most 0.5 on tensors of rank 2 or more, at most 20% of the entries
changed with the BatchNorm running statistics counted, max_ops at least the tensor count, no edit-size limit), the
limits of an E10 update program, instead of E1's patch limits.

Seqskill runs (``seqskill=True``, PREREG E5: teach a sequence student a new operator) take a seqskill-v1 instance
(``weightcode.evalkit.seqskill``): the evidence is its new-operator rows followed by its replay rows of the old operators,
the joint oracle scores part 1 (the new operator alone) and part 4 (old operators) of the condition's V_check, and the
limits are ``seqskill_program_limits`` (rel-Frob at most 0.5 on tensors of rank 2 or more, at most 20% of the entries
changed, no edit-size limit).

Graphnew runs (``graphnew=True``, PREREG E7: add classes a graph network never trained on) take a graphnew-v1
instance (``weightcode.evalkit.graphnew``): the evidence is its new-class nodes followed by its replay nodes of the train
classes (int64 node ids into the instance's graph.npz), the joint oracle scores the new-class and the train-class rows
of V_check (one full-graph forward per part), and the limits are ``graphnew_program_limits`` (head exempt from
max_rel_frob, rel-Frob at most 0.5 on the other tensors of rank 2 or more, at most 20% of the entries changed, no
edit-size limit). forward.py runs the whole graph on every call, so every prediction here is one call
(``predict_nodes``).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import weightcode
from weightcode.baselines.instance_io import DEFAULT_QUERY_BUDGET, Instance, make_oracle, write_delta_patch
from weightcode.common.identity import Params, count_changed, params_sha256
from weightcode.common.paths import runs_dir
from weightcode.data import seqops as S
from weightcode.evalkit.companions import REPLAY_RULE, load_replay, make_joint_oracle
from weightcode.evalkit.e10 import e10_program_limits
from weightcode.evalkit.evidence import EVIDENCE_RULE, EVIDENCE_RULE_V1
from weightcode.evalkit.fewshot import FEWSHOT_RULE, check_fewshot_current, fewshot_program_limits, make_fewshot_oracle
from weightcode.evalkit.fewshot import REPLAY_RULE as FEWSHOT_REPLAY_RULE
from weightcode.evalkit.graphnew import (
    GRAPHNEW_RULE,
    check_graphnew_current,
    graphnew_program_limits,
    make_graphnew_oracle,
    predict_nodes,
)
from weightcode.evalkit.graphnew import REPLAY_RULE as GRAPHNEW_REPLAY_RULE
from weightcode.evalkit.metrics import BASE_CANDIDATE, Candidate, summarize
from weightcode.evalkit.patch_check import JOINT_PARTS, JointPatchCheckOracle, PatchCheckOracle, predict
from weightcode.evalkit.scanskill import SCAN_LIMITS_SOURCE, SCAN_RULE, check_scan_current, make_scan_oracle, scan_program_limits
from weightcode.evalkit.scanskill import evidence_keys as scan_evidence_keys
from weightcode.evalkit.scanskill import seq_predictions as scan_predictions
from weightcode.evalkit.seqskill import REPLAY_RULE as SEQSKILL_REPLAY_RULE
from weightcode.evalkit.seqskill import (
    SEQSKILL_RULE,
    check_seqskill_current,
    make_seqskill_oracle,
    seq_predictions,
    seqskill_program_limits,
)
from weightcode.evalkit.targets import DEPLOY_CONDITION
from weightcode.patch import Limits, PatchError, task_limits
from weightcode.patch.limits import dense_delta_bytes

from .edits import (
    MAX_CHANGED_FRAC,
    MAX_EDIT_FRAC,
    MAX_REL_FROB,
    PATCH_APPLIER,
    REL_FROB_MIN_NDIM,
    apply_patch_result,
    edit_stats_under,
    write_ops_patch,
)

RUN_SCHEMA = "weightcode.baseline_run.v0"
EVIDENCE_BATCH = 256
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_TRACKED_PACKAGES = ("baselines", "patch", "evalkit", "common")
FEWSHOT_EVIDENCE_FIELDS = ("x_new", "y_new", "pred_new", "x_old", "y_old", "pred_old")
FEWSHOT_LIMITS_SOURCE = "fewshot_program_limits(task.json limits): PREREG E2, the limits of an E2 update program"
SEQSKILL_LIMITS_SOURCE = ("seqskill_program_limits(task.json limits, n_tensors): PREREG E5, the limits of an E5 update "
                          "program (max_ops raised to the student's tensor count)")
E10_LIMITS_SOURCE = ("e10_program_limits(task.json limits, n_tensors, dense_bytes): PREREG E10, the limits of an E10 "
                     "update program")
E14_LIMITS_SOURCE = ("e14_program_limits(task.json limits, n_tensors, dense_bytes): PREREG E14, the limits of an E14 "
                     "update program (rel-Frob lifted, at most 20% of the entries changed)")
GRAPHNEW_LIMITS_SOURCE = ("graphnew_program_limits(task.json limits, n_tensors): PREREG E7, the limits of an E7 update "
                          "program (head exempt from max_rel_frob, max_ops raised to the student's tensor count)")


class SplitRoleError(RuntimeError):
    """Raised when a run would touch a main instance without explicit permission, or the role is unknown."""


class NoValidCandidate(RuntimeError):
    """Every candidate of a run that requires a valid one was invalid; the run is recorded as failed."""


@dataclass(frozen=True)
class Evidence:
    """The 64 visible evidence images in a fixed order: the k_fail failures, then the k_rand random ones.

    Under evidence-v0 the failures are target-pair failures (y = a predicted as b); under evidence-v1 they are base
    errors of any class. The random images are a plain sample of the deployment data in both rules. In a joint run
    the ``n_replay`` replay rows (clean train images, ``load_joint_evidence``) follow them. ``sha256`` always
    identifies the instance's own evidence draw and ``replay_sha256`` the replay content.

    A fewshot-v1 instance (``load_fewshot_evidence``) has no failure part: its new-class rows are the ``n_rand`` rows,
    drawn without regard to the base predictions, and its replay rows of the train classes follow them. A seqskill-v1
    instance (``load_seqskill_evidence``) has the same shape: its new-operator token rows, then its replay rows of the
    old operators; ``y`` and ``base_pred`` are then [N, L] with IGNORE (-1) off the answer positions. A graphnew-v1
    instance (``load_graphnew_evidence``) has the fewshot shape with int64 node ids as ``x``.
    """

    x: np.ndarray
    y: np.ndarray
    n_fail: int
    n_rand: int
    sha256: str
    base_pred: np.ndarray
    rule: str
    n_replay: int = 0
    replay_sha256: str | None = None
    replay_rule: str = REPLAY_RULE

    @property
    def n_deployment(self) -> int:
        """Rows drawn from the deployment condition: the failures and the random ones."""
        return self.n_fail + self.n_rand

    def deployment(self) -> Evidence:
        """The deployment rows only, without replay rows (the evidence of a non-joint run)."""
        if self.n_replay == 0:
            return self
        k = self.n_deployment
        return Evidence(
            x=self.x[:k], y=self.y[:k], n_fail=self.n_fail, n_rand=self.n_rand, sha256=self.sha256,
            base_pred=self.base_pred[:k], rule=self.rule,
        )

    def describe(self) -> dict[str, Any]:
        """JSON-ready summary for run records."""
        if self.rule in (FEWSHOT_RULE, SEQSKILL_RULE, GRAPHNEW_RULE, SCAN_RULE):
            summary: dict[str, Any] = {"rule": self.rule, "n_new": self.n_rand, "sha256": self.sha256}
        else:
            summary = {"rule": self.rule, "n_fail": self.n_fail, "n_rand": self.n_rand, "sha256": self.sha256}
        if self.n_replay:
            summary.update({"n_replay": self.n_replay, "replay_rule": self.replay_rule, "replay_sha256": self.replay_sha256})
        return summary

    def composition(self) -> str:
        """The rows in words, for patch hypotheses."""
        if self.rule == FEWSHOT_RULE:
            return f"{self.n_rand} new-class rows, {self.n_replay} replay rows of the train classes"
        if self.rule == SEQSKILL_RULE:
            return f"{self.n_rand} new-operator rows, {self.n_replay} replay rows of the old operators"
        if self.rule == GRAPHNEW_RULE:
            return f"{self.n_rand} new-class nodes, {self.n_replay} replay nodes of the train classes"
        if self.rule == SCAN_RULE:
            return f"{self.n_rand} primitive rows, {self.n_replay} replay rows of old commands"
        replay = f", {self.n_replay} clean replay" if self.n_replay else ""
        return f"{self.n_fail} failures, {self.n_rand} random{replay}"


@dataclass
class CandidatePatch:
    """One candidate child model, generated before any query, with its configuration and diagnostics.

    ``ops`` None: the patch is the delta ``child - base`` written by ``write_delta_patch``. Otherwise the patch is these
    explicit ``weightcode.patch.v0`` ops and ``child`` is the result the generator expects from them, which the executor's
    output is compared against.
    """

    slot: int
    child: Params
    config: dict[str, Any]
    hypothesis: str
    predicted_effect: str
    notes: dict[str, Any] = field(default_factory=dict)
    ops: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class MethodSpec:
    """Fixed description of a method as recorded in run.json."""

    method: str
    label: str
    uses_labels_for_fit: bool
    edit_limits_enforced: bool


Generator = Callable[[Evidence], tuple[list[CandidatePatch], dict[str, Any]]]


def check_split_role(instance: Instance, allow_main: bool) -> str:
    """Return the instance's split role; refuse main instances unless ``allow_main`` and refuse unknown roles.

    PREREG section 2: non-LLM arms run on dev instances until their code and parameters are frozen.
    """
    role = instance.private.get("split_role")
    if role == "dev":
        return role
    if role == "main":
        if allow_main:
            return role
        raise SplitRoleError(
            f"{instance.instance_id} is a main instance; non-LLM arms run on main instances only after the protocol "
            "freeze (PREREG section 2). Pass allow_main explicitly once frozen."
        )
    raise SplitRoleError(f"{instance.instance_id}: private.json split_role is {role!r}, expected 'dev' or 'main'")


def safe_name(value: str, what: str) -> str:
    """Return ``value`` if it is a plain path component (letters, digits, '_', '.', '-'), else raise."""
    if not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{what} {value!r} must match {_SAFE_NAME.pattern} and not be '.' or '..'")
    return value


def baseline_run_dir(instance_id: str, method: str, run_seed: int, ref_label: str | None = None) -> Path:
    """``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/[<ref_label>/]<run_seed>``."""
    path = runs_dir() / "baselines" / safe_name(instance_id, "instance id") / safe_name(method, "method")
    if ref_label is not None:
        path = path / safe_name(ref_label, "reference label")
    return path / str(int(run_seed))


def control_run_dir(instance_id: str, kind: str, arm: str, run_index: int) -> Path:
    """``$WEIGHTCODE_RUNS/baselines/<instance_id>/<kind>/<arm>/r<run_index>``: one B1 control run (PREREG layout)."""
    if run_index < 0:
        raise ValueError(f"run_index must be >= 0, got {run_index}")
    root = runs_dir() / "baselines" / safe_name(instance_id, "instance id") / safe_name(kind, "control kind")
    return root / safe_name(arm, "arm") / f"r{int(run_index)}"


def read_model_spec(instance: Instance) -> dict[str, Any]:
    """The editor-visible model_spec.json of an instance."""
    return json.loads((Path(instance.instance_dir) / "model_spec.json").read_text())


def evidence_rule(instance: Instance) -> str:
    """The instance's evidence rule from private.json, checked against the presence of a target pair."""
    rule = instance.private.get("evidence_rule")
    if rule == EVIDENCE_RULE and instance.target is None:
        raise ValueError(f"{instance.instance_id}: an {EVIDENCE_RULE} instance needs a target pair in private.json")
    if rule == EVIDENCE_RULE_V1 and instance.target is not None:
        raise ValueError(f"{instance.instance_id}: an {EVIDENCE_RULE_V1} instance has no target pair, found one")
    if rule not in (EVIDENCE_RULE, EVIDENCE_RULE_V1):
        raise ValueError(f"{instance.instance_id}: unknown evidence_rule {rule!r} in private.json")
    return str(rule)


def load_evidence(instance: Instance) -> Evidence:
    """Concatenate x_fail and x_rand with labels and check the failures against forward.py.

    evidence-v0: every failure is a target-pair failure (y = a, predicted b). evidence-v1: every failure is a base
    error (prediction != y), and pred_fail and pred_rand equal the base predictions.
    """
    rule = evidence_rule(instance)
    data = instance.evidence
    required = ("x_fail", "y_fail", "x_rand", "y_rand")
    if rule == EVIDENCE_RULE_V1:
        required += ("pred_fail", "pred_rand")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"evidence.npz lacks {missing}")
    x = np.concatenate([data["x_fail"], data["x_rand"]])
    y = np.concatenate([data["y_fail"], data["y_rand"]]).astype(np.int64)
    n_fail, n_rand = len(data["x_fail"]), len(data["x_rand"])
    if x.dtype != np.uint8 or len(y) != len(x) or len(data["y_fail"]) != n_fail:
        raise ValueError(f"evidence images must be uint8 with one label each, got {x.dtype}, {len(x)} vs {len(y)}")
    if rule == EVIDENCE_RULE:
        a, b = instance.target  # type: ignore[misc]
        if not np.all(data["y_fail"] == a):
            raise ValueError(f"y_fail must all equal the target class a = {a}")
    base_pred = predict(instance.forward, instance.params, x, EVIDENCE_BATCH)
    if rule == EVIDENCE_RULE:
        off_target = int(np.count_nonzero(base_pred[:n_fail] != b))
        if off_target:
            raise ValueError(f"{off_target} of {n_fail} x_fail images are not predicted as b = {b} by forward.py")
    else:
        correct = int(np.count_nonzero(base_pred[:n_fail] == y[:n_fail]))
        if correct:
            raise ValueError(f"{correct} of {n_fail} x_fail images are classified correctly by forward.py")
        stored = np.concatenate([data["pred_fail"], data["pred_rand"]]).astype(np.int64)
        disagree = int(np.count_nonzero(stored != base_pred))
        if disagree:
            raise ValueError(f"pred_fail/pred_rand differ from the forward.py predictions on {disagree} images")
    digest = params_sha256({key: np.asarray(data[key]) for key in ("x_fail", "y_fail", "x_rand", "y_rand")})
    return Evidence(x=x, y=y, n_fail=n_fail, n_rand=n_rand, sha256=digest, base_pred=base_pred, rule=rule)


def load_joint_evidence(instance: Instance) -> Evidence:
    """The instance's evidence followed by its dataset's replay rows (replay-v1), for joint runs.

    The replay rows are clean train images, one fixed draw per dataset (``weightcode.evalkit.companions``), so none of
    them shares its underlying image with an evidence row.
    """
    evidence = load_evidence(instance)
    replay = load_replay(str(instance.private["dataset"]))
    x_replay = replay["x"]
    if x_replay.dtype != evidence.x.dtype or x_replay.shape[1:] != evidence.x.shape[1:]:
        raise ValueError(
            f"replay images are {x_replay.dtype}{x_replay.shape[1:]}, evidence {evidence.x.dtype}{evidence.x.shape[1:]}"
        )
    base_pred = predict(instance.forward, instance.params, x_replay, EVIDENCE_BATCH)
    return Evidence(
        x=np.concatenate([evidence.x, x_replay]),
        y=np.concatenate([evidence.y, replay["y"].astype(np.int64)]),
        n_fail=evidence.n_fail,
        n_rand=evidence.n_rand,
        sha256=evidence.sha256,
        base_pred=np.concatenate([evidence.base_pred, base_pred]),
        rule=evidence.rule,
        n_replay=int(len(x_replay)),
        replay_sha256=params_sha256(replay),
    )


def fewshot_rule(instance: Instance) -> str:
    """FEWSHOT_RULE for a current fewshot-v1 instance without a target pair; refuses anything else.

    ``check_fewshot_current`` refuses an instance that is not active or whose files the current code would not write.
    """
    rule = instance.private.get("evidence_rule")
    if rule != FEWSHOT_RULE or instance.target is not None:
        raise ValueError(f"{instance.instance_id}: a fewshot run needs a {FEWSHOT_RULE} instance, found rule {rule!r}")
    check_fewshot_current(instance.instance_dir)
    return FEWSHOT_RULE


def load_fewshot_evidence(instance: Instance) -> Evidence:
    """The new-class rows of a fewshot-v1 instance followed by its replay rows of the train classes (PREREG E2).

    Every stored prediction must equal forward.py's, every new-class row must carry a new class and every replay row a
    train class. ``sha256`` identifies the new-class part and ``replay_sha256`` the replay part (rule
    fewshot-replay-v1).
    """
    data = instance.evidence
    missing = [key for key in FEWSHOT_EVIDENCE_FIELDS if key not in data]
    if missing:
        raise ValueError(f"evidence.npz lacks {missing}")
    x = np.concatenate([data["x_new"], data["x_old"]])
    y = np.concatenate([data["y_new"], data["y_old"]]).astype(np.int64)
    n_new, n_old = len(data["x_new"]), len(data["x_old"])
    if x.dtype != np.uint8 or len(y) != len(x) or len(data["y_new"]) != n_new:
        raise ValueError(f"evidence images must be uint8 with one label each, got {x.dtype}, {len(x)} vs {len(y)}")
    new_classes = [int(c) for c in instance.private["new_classes"]]
    if not np.all(np.isin(data["y_new"], new_classes)) or np.any(np.isin(data["y_old"], new_classes)):
        raise ValueError(f"the new-class rows must carry classes {new_classes} and the replay rows none of them")
    base_pred = predict(instance.forward, instance.params, x, EVIDENCE_BATCH)
    stored = np.concatenate([data["pred_new"], data["pred_old"]]).astype(np.int64)
    disagree = int(np.count_nonzero(stored != base_pred))
    if disagree:
        raise ValueError(f"pred_new/pred_old differ from the forward.py predictions on {disagree} images")
    return Evidence(
        x=x,
        y=y,
        n_fail=0,
        n_rand=n_new,
        sha256=params_sha256({key: np.asarray(data[key]) for key in ("x_new", "y_new")}),
        base_pred=base_pred,
        rule=FEWSHOT_RULE,
        n_replay=n_old,
        replay_sha256=params_sha256({key: np.asarray(data[key]) for key in ("x_old", "y_old")}),
        replay_rule=FEWSHOT_REPLAY_RULE,
    )


def seqskill_rule(instance: Instance) -> str:
    """SEQSKILL_RULE for a current seqskill-v1 instance without a target pair; refuses anything else.

    ``check_seqskill_current`` refuses an instance that is not active or whose files the current code would not write.
    """
    rule = instance.private.get("evidence_rule")
    if rule != SEQSKILL_RULE or instance.target is not None:
        raise ValueError(f"{instance.instance_id}: a seqskill run needs a {SEQSKILL_RULE} instance, found rule {rule!r}")
    check_seqskill_current(instance.instance_dir)
    return SEQSKILL_RULE


def load_seqskill_evidence(instance: Instance) -> Evidence:
    """The new-operator rows of a seqskill-v1 instance followed by its replay rows of the old operators (PREREG E5).

    Rows are int64 token rows [N, L] and targets [N, L] with IGNORE off the answer positions. Every stored prediction
    must equal ``seq_predictions`` of forward.py, every new row must hold the instance's new operator alone (OPA the
    new operator, OPB NONE) and no replay row a reserved operator. ``sha256`` identifies the new part and
    ``replay_sha256`` the replay part (rule seqskill-replay-v1).
    """
    data = instance.evidence
    missing = [key for key in FEWSHOT_EVIDENCE_FIELDS if key not in data]
    if missing:
        raise ValueError(f"evidence.npz lacks {missing}")
    x = np.concatenate([data["x_new"], data["x_old"]])
    y = np.concatenate([data["y_new"], data["y_old"]]).astype(np.int64)
    n_new, n_old = len(data["x_new"]), len(data["x_old"])
    if x.dtype != np.int64 or x.ndim != 2 or y.shape != x.shape:
        raise ValueError(f"seqskill evidence must be int64 token rows with one target per position, got {x.dtype} "
                         f"{x.shape} and targets {y.shape}")
    new_id = int(instance.private["new_operator_id"])
    if not np.all(data["x_new"][:, 1] == new_id) or not np.all(data["x_new"][:, 2] == S.NONE):
        raise ValueError(f"every new row must hold operator {new_id} alone (OPA {new_id}, OPB NONE)")
    if np.isin(data["x_old"][:, 1:3], S.NEW_OP_IDS).any():
        raise ValueError("a replay row holds a reserved operator")
    base_pred = seq_predictions(instance.forward, instance.params, x, y, EVIDENCE_BATCH)
    stored = np.concatenate([data["pred_new"], data["pred_old"]]).astype(np.int64)
    disagree = int(np.count_nonzero(stored != base_pred))
    if disagree:
        raise ValueError(f"pred_new/pred_old differ from the forward.py predictions at {disagree} positions")
    return Evidence(
        x=x,
        y=y,
        n_fail=0,
        n_rand=n_new,
        sha256=params_sha256({key: np.asarray(data[key]) for key in ("x_new", "y_new")}),
        base_pred=base_pred,
        rule=SEQSKILL_RULE,
        n_replay=n_old,
        replay_sha256=params_sha256({key: np.asarray(data[key]) for key in ("x_old", "y_old")}),
        replay_rule=SEQSKILL_REPLAY_RULE,
    )


def scan_rule(instance: Instance) -> str:
    """SCAN_RULE for a current scanskill-v1 instance without a target pair (PREREG E13); refuses anything else.

    ``check_scan_current`` refuses an instance that is not active or whose files the current code would not write.
    """
    rule = instance.private.get("evidence_rule")
    if rule != SCAN_RULE or instance.target is not None:
        raise ValueError(f"{instance.instance_id}: a scan run needs a {SCAN_RULE} instance, found rule {rule!r}")
    check_scan_current(instance.instance_dir)
    return SCAN_RULE


def load_scan_evidence(instance: Instance) -> Evidence:
    """The experience rows of a scanskill-v1 instance (PREREG E13): its primitive row (addprim_jump: "jump" as in
    training; none on around_right), then its 64 replay rows of old commands.

    Rows are int64 token rows [N, L] and targets [N, L] with IGNORE off the OUT positions. Every stored prediction must
    equal ``seq_predictions`` of forward.py. ``n_rand`` is the number of primitive rows, ``sha256`` the instance's
    evidence content hash and ``replay_sha256`` the replay part (rule seqskill-replay-v1).
    """
    data = instance.evidence
    keys = scan_evidence_keys(str(instance.private["condition"]))
    if sorted(data) != sorted(keys):
        raise ValueError(f"evidence.npz keys {sorted(data)} are not the scanskill keys {sorted(keys)}")
    parts = ("prim", "old") if "x_prim" in data else ("old",)
    x = np.concatenate([data[f"x_{part}"] for part in parts])
    y = np.concatenate([data[f"y_{part}"] for part in parts]).astype(np.int64)
    if x.dtype != np.int64 or x.ndim != 2 or y.shape != x.shape:
        raise ValueError(f"scan evidence must be int64 token rows with one target per position, got {x.dtype} "
                         f"{x.shape} and targets {y.shape}")
    base_pred = scan_predictions(instance.forward, instance.params, x, y, EVIDENCE_BATCH)
    stored = np.concatenate([data[f"pred_{part}"] for part in parts]).astype(np.int64)
    disagree = int(np.count_nonzero(stored != base_pred))
    if disagree:
        raise ValueError(f"the stored predictions differ from the forward.py predictions at {disagree} positions")
    n_prim = len(data["x_prim"]) if "x_prim" in data else 0
    if n_prim != int(instance.private["n_prim"]) or len(data["x_old"]) != int(instance.private["n_replay"]):
        raise ValueError("the experience part sizes differ from the instance record")
    return Evidence(
        x=x,
        y=y,
        n_fail=0,
        n_rand=n_prim,
        sha256=str(instance.private["evidence_content_sha256"]),
        base_pred=base_pred,
        rule=SCAN_RULE,
        n_replay=len(data["x_old"]),
        replay_sha256=params_sha256({key: np.asarray(data[key]) for key in ("x_old", "y_old")}),
        replay_rule=SEQSKILL_REPLAY_RULE,
    )


def load_e14_evidence(instance: Instance) -> Evidence:
    """The new-demonstration steps of a policyskill-v1 instance followed by its 64 replay steps (PREREG E14).

    Rows are int64 observation rows [N, 197] and targets the planner's action indices. Every stored logit must equal
    forward.py's (``weightcode.evalkit.e14.policy_logits``), every new row must hold the new mission's verbs and no replay
    row a reserved word. ``base_pred`` is the argmax of the logits. ``sha256`` identifies the new part and
    ``replay_sha256`` the replay part (rule policyskill-replay-v1)."""
    from weightcode.data import e14 as D
    from weightcode.evalkit import e14 as E

    data = instance.evidence
    missing = [key for key in E.EVIDENCE_KEYS if key not in data]
    if missing or instance.private.get("evidence_rule") != E.POLICYSKILL_RULE:
        raise ValueError(f"not a {E.POLICYSKILL_RULE} evidence file (missing {missing})")
    D.check_rows(np.asarray(data["x_new"]), D.mission_verbs("new"))
    D.check_rows(np.asarray(data["x_old"]), D.mission_verbs("old"))
    x = np.concatenate([data["x_new"], data["x_old"]]).astype(np.int64)
    y = np.concatenate([data["y_new"], data["y_old"]]).astype(np.int64)
    logits = E.policy_logits(instance.forward, instance.params, x)
    stored = np.concatenate([data["logits_new"], data["logits_old"]])
    if not np.array_equal(logits, stored):
        raise ValueError("logits_new/logits_old differ from forward.py's logits")
    return Evidence(
        x=x,
        y=y,
        n_fail=0,
        n_rand=int(len(data["x_new"])),
        sha256=params_sha256({key: np.asarray(data[key]) for key in ("x_new", "y_new")}),
        base_pred=logits.argmax(1).astype(np.int64),
        rule=E.POLICYSKILL_RULE,
        n_replay=int(len(data["x_old"])),
        replay_sha256=params_sha256({key: np.asarray(data[key]) for key in ("x_old", "y_old")}),
        replay_rule=E.REPLAY_RULE,
    )


def e14_rule(instance: Instance) -> str:
    """POLICYSKILL_RULE for a current policyskill-v1 instance; refuses anything else."""
    from weightcode.evalkit import e14 as E

    rule = instance.private.get("evidence_rule")
    if rule != E.POLICYSKILL_RULE:
        raise ValueError(f"{instance.instance_id}: evidence rule {rule!r} is not {E.POLICYSKILL_RULE}")
    E.check_e14_current(instance.instance_dir)
    return str(rule)


def graphnew_rule(instance: Instance) -> str:
    """GRAPHNEW_RULE for a current graphnew-v1 instance without a target pair; refuses anything else.

    ``check_graphnew_current`` refuses an instance that is not active or whose files the current code would not write.
    """
    rule = instance.private.get("evidence_rule")
    if rule != GRAPHNEW_RULE or instance.target is not None:
        raise ValueError(f"{instance.instance_id}: a graphnew run needs a {GRAPHNEW_RULE} instance, found rule {rule!r}")
    check_graphnew_current(instance.instance_dir)
    return GRAPHNEW_RULE


def load_graphnew_evidence(instance: Instance) -> Evidence:
    """The new-class nodes of a graphnew-v1 instance followed by its replay nodes of the train classes (PREREG E7).

    ``x`` holds int64 node ids into the instance's graph.npz. Every stored prediction must equal forward.py's (one
    call over all 144 nodes, ``predict_nodes``), every new row must carry a new class and every replay row a train
    class. ``sha256`` identifies the new part and ``replay_sha256`` the replay part (rule graphnew-replay-v1).
    """
    data = instance.evidence
    missing = [key for key in FEWSHOT_EVIDENCE_FIELDS if key not in data]
    if missing:
        raise ValueError(f"evidence.npz lacks {missing}")
    x = np.concatenate([data["x_new"], data["x_old"]])
    y = np.concatenate([data["y_new"], data["y_old"]]).astype(np.int64)
    n_new, n_old = len(data["x_new"]), len(data["x_old"])
    if x.dtype != np.int64 or x.ndim != 1 or len(y) != len(x) or len(data["y_new"]) != n_new:
        raise ValueError(f"graphnew evidence must be int64 node ids [N] with one label each, got {x.dtype} {x.shape}, "
                         f"{len(y)} labels")
    if len(np.unique(x)) != len(x):
        raise ValueError("graphnew evidence repeats a node")
    new_classes = [int(c) for c in instance.private["new_classes"]]
    if not np.all(np.isin(data["y_new"], new_classes)) or np.any(np.isin(data["y_old"], new_classes)):
        raise ValueError(f"the new-class rows must carry classes {new_classes} and the replay rows none of them")
    (base_pred,) = predict_nodes(instance.forward, instance.params, x)
    stored = np.concatenate([data["pred_new"], data["pred_old"]]).astype(np.int64)
    disagree = int(np.count_nonzero(stored != base_pred))
    if disagree:
        raise ValueError(f"pred_new/pred_old differ from the forward.py predictions on {disagree} nodes")
    return Evidence(
        x=x,
        y=y,
        n_fail=0,
        n_rand=n_new,
        sha256=params_sha256({key: np.asarray(data[key]) for key in ("x_new", "y_new")}),
        base_pred=base_pred,
        rule=GRAPHNEW_RULE,
        n_replay=n_old,
        replay_sha256=params_sha256({key: np.asarray(data[key]) for key in ("x_old", "y_old")}),
        replay_rule=GRAPHNEW_REPLAY_RULE,
    )


def jsonable(value: Any) -> Any:
    """Convert numpy scalars, arrays, tuples and paths for JSON; non-finite floats become strings."""
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return number
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"cannot serialise {type(value).__name__}")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict JSON (no NaN or infinity literals) through a temporary file and a rename."""
    text = json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def code_fingerprint() -> dict[str, str]:
    """sha256 of every source file a run depends on: the fitting package, the executor, instance I/O and evalkit."""
    package_root = Path(weightcode.__file__).resolve().parent
    files = sorted((package_root / "baselines" / "fit").glob("*.py")) + sorted((package_root / "patch").glob("*.py"))
    files += [
        package_root / "baselines" / "instance_io.py",
        package_root / "evalkit" / "companions.py",
        package_root / "evalkit" / "evidence.py",
        package_root / "evalkit" / "fewshot.py",
        package_root / "evalkit" / "graphnew.py",
        package_root / "evalkit" / "metrics.py",
        package_root / "evalkit" / "patch_check.py",
        package_root / "common" / "identity.py",
    ]
    return {
        str(path.relative_to(package_root.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }


def git_state() -> dict[str, Any]:
    """HEAD commit and uncommitted paths under the code this run uses; the git error text if git is unavailable."""
    repo = Path(weightcode.__file__).resolve().parents[1]
    command = ["git", "-C", str(repo)]
    try:
        commit = subprocess.run([*command, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        status = subprocess.run(
            [*command, "status", "--porcelain", "--", *(f"weightcode/{part}" for part in _TRACKED_PACKAGES)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return {"commit": commit.stdout.strip(), "uncommitted": status.stdout.splitlines()}


def provenance() -> dict[str, Any]:
    """Software versions, code hashes and git state for a run record."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "code_sha256": code_fingerprint(),
        "git": git_state(),
    }


def invalid_reason(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray]) -> str | None:
    """Why ``child`` cannot be a child of ``base`` (keys, shapes, dtypes, non-finite values), or None."""
    if set(child) != set(base):
        return f"key mismatch: {sorted(set(child) ^ set(base))}"
    for key, parent in base.items():
        value = np.asarray(child[key])
        if value.shape != parent.shape or value.dtype != parent.dtype:
            return f"{key}: {value.dtype}{value.shape} != {parent.dtype}{parent.shape}"
        if not np.all(np.isfinite(value)):
            return f"{key} has non-finite values"
    return None


def _final_record(
    instance: Instance, final: Candidate, log: list[dict[str, Any]], checked: list[dict[str, Any]], joint: bool = False
) -> dict[str, Any]:
    """The chosen patch with its V_check readout: S, fixed and broken over all images, and the accuracy change.

    The base no-op has S = fixed = broken = 0 and edit size 0. Accuracies need at least one consumed query (every
    log entry carries the same base accuracy and n); without one they are None. ``fixed_target`` is None on an
    instance without a target pair. A joint run's numbers are pooled over both check sets (S = S_joint), and each
    part's numbers follow with the suffixes ``_new`` (the condition's V_check) and ``_old`` (its clean companion).
    """
    entry = next((item for item in log if item["patch_id"] == final.patch_id), None)
    if final.patch_id is not None and entry is None:
        raise RuntimeError(f"final patch {final.patch_id} is missing from the oracle log")
    has_target = instance.target is not None
    if entry is not None:
        fixed, broken = int(entry["fixed"]), int(entry["broken"])
        fixed_target = int(entry["fixed_target"]) if has_target else None
        candidate = next(item for item in checked if item["patch_id"] == final.patch_id)
        edit_size = int(candidate["edit"]["edit_size"])
    else:
        fixed, broken, edit_size = 0, 0, 0
        fixed_target = 0 if has_target else None
    if fixed - broken != final.score:
        raise RuntimeError(f"final score {final.score} != fixed {fixed} - broken {broken}")
    n_check = int(log[0]["n"]) if log else None
    base_acc = float(log[0]["base_acc"]) if log else None
    parts: dict[str, Any] = {}
    if joint:
        for part in JOINT_PARTS:
            n_part = int(log[0][f"n_{part}"]) if log else None
            base_part = float(log[0][f"base_acc_{part}"]) if log else None
            score_part = int(entry[f"score_{part}"]) if entry is not None else 0
            parts.update({
                f"score_{part}": score_part,
                f"fixed_{part}": int(entry[f"fixed_{part}"]) if entry is not None else 0,
                f"broken_{part}": int(entry[f"broken_{part}"]) if entry is not None else 0,
                f"n_vcheck_{part}": n_part,
                f"base_acc_{part}": base_part,
                f"child_acc_{part}": float(entry[f"child_acc_{part}"]) if entry is not None else base_part,
                f"delta_acc_{part}": score_part / n_part if n_part else None,
            })
        if parts["score_new"] + parts["score_old"] != final.score:
            raise RuntimeError(f"final score {final.score} != S_new {parts['score_new']} + S_old {parts['score_old']}")
    return {
        "patch_id": final.patch_id,
        "is_base_noop": final.patch_id is None,
        "dir": final.patch_id,
        "query_index": final.query_index,
        "score": final.score,
        "fixed": fixed,
        "broken": broken,
        "fixed_target": fixed_target,
        "n_vcheck": n_check,
        "base_acc": base_acc,
        "child_acc": float(entry["child_acc"]) if entry is not None else base_acc,
        "delta_acc": final.score / n_check if n_check else None,
        "n_changed": final.n_changed,
        "edit_size": edit_size,
        "child_sha256": str(entry["child_sha256"]) if entry is not None else instance.params_sha256,
        **parts,
    }


def execute_run(
    *,
    instance: Instance,
    spec: MethodSpec,
    run_dir: Path,
    run_seed: int,
    config: Mapping[str, Any],
    generate: Generator,
    allow_main: bool = False,
    budget: int = DEFAULT_QUERY_BUDGET,
    require_valid_candidate: bool = False,
    joint: bool = False,
    replay_in_fit: bool = True,
    fewshot: bool = False,
    seqskill: bool = False,
    graphnew: bool = False,
    e10: bool = False,
    scan: bool = False,
    e14: bool = False,
    evidence_override: Evidence | None = None,
) -> dict[str, Any]:
    """Generate all candidates, write and re-apply each, spend one query per valid candidate, record run.json.

    A candidate is invalid, and consumes no query, if its parameters are malformed or the executor rejects its patch
    (with the run's limits); the reason and the executor's messages are recorded. With ``require_valid_candidate``, a
    run that generated candidates but could check none of them fails with ``NoValidCandidate`` instead of ending as
    the base no-op (a B1 control whose draws are all rejected has not measured anything). The run directory must not
    exist yet. A failed run still writes run.json with status "failed" and the oracle log so far, then re-raises.

    With ``joint`` (evidence-v1 instances of a named condition only) the replay rows are appended to the evidence and
    the joint oracle scores every candidate; ``replay_in_fit`` False gives the generator the deployment rows only.
    Each checked candidate then records its evidence summary on the deployment rows and on the replay rows.

    With ``fewshot`` (fewshot-v1 instances only, not combined with ``joint``) the evidence is the instance's new-class
    rows plus its replay rows, always in the fit, the fewshot joint oracle scores the candidates and every patch is
    applied and measured under ``fewshot_program_limits``.

    With ``seqskill`` (seqskill-v1 instances only, alone) the evidence is the instance's new-operator rows plus its
    replay rows, always in the fit, the seqskill joint oracle (V_check parts 1 and 4) scores the candidates and every
    patch is applied and measured under ``seqskill_program_limits``.

    With ``graphnew`` (graphnew-v1 instances only, alone) the evidence is the instance's new-class nodes plus its
    replay nodes, always in the fit, the graphnew joint oracle (V_check cut by class) scores the candidates, every
    patch is applied and measured under ``graphnew_program_limits`` and every evidence prediction is one forward call.

    With ``scan`` (scanskill-v1 instances only, alone; PREREG E13) the evidence is the instance's experience rows (its
    primitive row on addprim_jump, then its 64 replay rows; ``load_scan_evidence``), the scan joint oracle (new V_check
    and the 1000 old V_check rows) scores the candidates and every patch is applied and measured under
    ``scan_program_limits`` (E5's program limits with the rel-Frob limit lifted). The generator may fit other rows
    (the rule-generated references); the record then says which.

    With ``e14`` (policyskill-v1 instances only, alone) the evidence is the instance's new-demonstration steps plus its
    replay steps, always in the fit, the E14 rollout oracle (``weightcode.evalkit.e14.make_e14_oracle``: 500 new and 500 old
    V_check episodes) scores the candidates and every patch is applied and measured under ``e14_program_limits``.
    ``evidence_override`` (E14: the label-equivalent runs; E10-limit joint runs: PREREG T04's large-sample
    fine-tuning diagnostic, ``scripts/t04/large_ft.py``) gives the generator other rows than the instance's; the
    oracle and the limits are unchanged, and run.json records the override's description.
    """
    role = check_split_role(instance, allow_main)
    if scan and (joint or fewshot or seqskill or graphnew or e10 or e14):
        raise ValueError("a scan run is scored by the scan joint oracle; pass scan alone")
    if evidence_override is not None and not (e14 or (e10 and joint)):
        raise ValueError("evidence_override is for an E14 run or an E10-limit joint run")
    if e14 and (joint or fewshot or seqskill or graphnew or e10):
        raise ValueError("an E14 run is scored by the E14 rollout oracle; pass e14 alone")
    if e10 and (fewshot or seqskill or graphnew):
        raise ValueError("an E10 run is a joint evidence-v1 run; it does not combine with fewshot, seqskill or graphnew")
    if scan:
        if not replay_in_fit:
            raise ValueError("an E13 run gets the instance's experience rows; replay_in_fit must be True")
        rule = scan_rule(instance)
        limits = scan_program_limits(instance.task["limits"], len(instance.params))
    elif e14:
        from weightcode.evalkit.e14 import e14_program_limits

        rule = e14_rule(instance)  # replay_in_fit False: F1's ft_full_noreplay (the oracle still scores both parts)
        limits = e14_program_limits(instance.task["limits"], len(instance.params),
                                    dense_delta_bytes(v.shape for v in instance.params.values()))
    elif graphnew:
        if joint or fewshot or seqskill:
            raise ValueError("a graphnew run is scored by the graphnew joint oracle; pass graphnew alone")
        if not replay_in_fit:
            raise ValueError("every E7 reference fits the same new-class and replay nodes; replay_in_fit must be True")
        rule = graphnew_rule(instance)
        limits = graphnew_program_limits(instance.task["limits"], len(instance.params))
    elif seqskill:
        if joint or fewshot:
            raise ValueError("a seqskill run is scored by the seqskill joint oracle; pass seqskill alone")
        if not replay_in_fit:
            raise ValueError("every E5 reference fits the same new and replay rows; replay_in_fit must be True")
        rule = seqskill_rule(instance)
        limits = seqskill_program_limits(instance.task["limits"], len(instance.params))
    elif fewshot:
        if joint:
            raise ValueError("a fewshot run is scored by the fewshot joint oracle; pass fewshot without joint")
        if not replay_in_fit:
            raise ValueError("every E2 reference fits the same new-class and replay rows; replay_in_fit must be True")
        rule = fewshot_rule(instance)
        limits = fewshot_program_limits(instance.task["limits"])
    elif e10:
        if not joint:
            raise ValueError("an E10 run is a joint run; pass e10 with joint")
        rule = evidence_rule(instance)
        limits = e10_program_limits(instance.task["limits"], len(instance.params),
                                    dense_delta_bytes(v.shape for v in instance.params.values()))
    else:
        rule = evidence_rule(instance)
        limits = Limits()
    if joint:
        if rule != EVIDENCE_RULE_V1 or instance.target is not None:
            raise ValueError(f"{instance.instance_id}: joint runs need an {EVIDENCE_RULE_V1} instance without a target pair")
        if instance.private.get("condition") in (None, DEPLOY_CONDITION):
            raise ValueError(
                f"{instance.instance_id}: joint runs need a named condition; {DEPLOY_CONDITION!r} has no clean companion"
            )
    run_dir.mkdir(parents=True, exist_ok=False)
    target = instance.target
    a, b = target if target is not None else (None, None)
    started = time.perf_counter()
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "method": spec.method,
        "label": spec.label,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "instance": {
            "instance_id": instance.instance_id,
            "split_role": role,
            "allow_main": allow_main,
            "params_sha256": instance.params_sha256,
            "forward_sha256": instance.forward_sha256,
            "evidence_rule": rule,
            "target": list(target) if target is not None else None,
        },
        "run_seed": int(run_seed),
        "budget": budget,
        "config": dict(config),
        "flags": {
            "uses_labels_for_fit": spec.uses_labels_for_fit,
            "edit_limits_enforced": spec.edit_limits_enforced,
            "respects_edit_limits": None,
        },
        "limits": {
            "max_changed_frac": MAX_CHANGED_FRAC,
            "max_edit_frac": MAX_EDIT_FRAC,
            "max_rel_frob": MAX_REL_FROB,
            "rel_frob_min_ndim": REL_FROB_MIN_NDIM,
        },
        "provenance": provenance(),
    }
    if fewshot:
        record["limits"] = task_limits(limits)
        record["limits_source"] = FEWSHOT_LIMITS_SOURCE
        record["fewshot"] = {
            "rule": FEWSHOT_RULE,
            "train_classes": list(instance.private["train_classes"]),
            "new_classes": list(instance.private["new_classes"]),
            "head_tensors": list(instance.private["head_tensors"]),
        }
        record["joint"] = {
            "oracle": "JointPatchCheckOracle",
            "parts": {"new": "V_check rows of the new classes", "old": "V_check rows of the train classes"},
            "score": "S_joint = S_new + S_old",
            "replay_rule": FEWSHOT_REPLAY_RULE,
            "replay_in_fit": True,
        }
    if seqskill:
        record["limits"] = task_limits(limits)
        record["limits_source"] = SEQSKILL_LIMITS_SOURCE
        executor = limits if spec.edit_limits_enforced else limits.without_edit_limits(len(instance.params))
        record["executor_limits"] = task_limits(executor)
        record["seqskill"] = {
            "rule": SEQSKILL_RULE,
            "condition": instance.private["condition"],
            "new_operator": instance.private["new_operator"],
            "new_operator_id": int(instance.private["new_operator_id"]),
        }
        record["joint"] = {
            "oracle": "JointPatchCheckOracle",
            "parts": {"new": "condition V_check part 1: the new operator alone, the lengths of the evidence",
                      "old": "condition V_check part 4: old-operator rows, single and composed"},
            "score": "S_joint = S_new + S_old",
            "replay_rule": SEQSKILL_REPLAY_RULE,
            "replay_in_fit": True,
        }
    if graphnew:
        record["limits"] = task_limits(limits)
        record["limits_source"] = GRAPHNEW_LIMITS_SOURCE
        executor = limits if spec.edit_limits_enforced else limits.without_edit_limits(len(instance.params))
        record["executor_limits"] = task_limits(executor)
        record["graphnew"] = {
            "rule": GRAPHNEW_RULE,
            "train_classes": list(instance.private["train_classes"]),
            "new_classes": list(instance.private["new_classes"]),
            "head_tensors": list(instance.private["head_tensors"]),
            "n_nodes": int(instance.private["n_nodes"]),
            "graph_file_sha256": instance.private["graph_file_sha256"],
        }
        record["joint"] = {
            "oracle": "JointPatchCheckOracle",
            "parts": {"new": "V_check rows of the new classes", "old": "V_check rows of the train classes"},
            "score": "S_joint = S_new + S_old",
            "replay_rule": GRAPHNEW_REPLAY_RULE,
            "replay_in_fit": True,
        }
    if scan:
        record["limits"] = task_limits(limits)
        record["limits_source"] = SCAN_LIMITS_SOURCE
        executor = limits if spec.edit_limits_enforced else limits.without_edit_limits(len(instance.params))
        record["executor_limits"] = task_limits(executor)
        record["scan"] = {
            "rule": SCAN_RULE,
            "condition": instance.private["condition"],
            "primitive": list(instance.private["primitive"]),
            "primitive_row": instance.private["primitive_row"],
        }
        record["joint"] = {
            "oracle": "JointPatchCheckOracle",
            "parts": {"new": "the condition's new V_check: held-out commands of the official test file",
                      "old": "the condition's old V_check: 1000 held-out commands of the official training file"},
            "score": "S_joint = S_new + S_old",
            "replay_rule": SEQSKILL_REPLAY_RULE,
            "replay_in_fit": True,
        }
    if e14:
        record["limits"] = task_limits(limits)
        record["limits_source"] = E14_LIMITS_SOURCE
        executor = limits if spec.edit_limits_enforced else limits.without_edit_limits(len(instance.params))
        record["executor_limits"] = task_limits(executor)
        record["e14"] = {"rule": rule, "model_id": instance.private["model_id"],
                         "n_new_demonstrations": int(instance.private["n_new_demonstrations"])}
        record["joint"] = {
            "oracle": "RolloutJointOracle",
            "parts": {"new": "500 new V_check episodes, greedy rollouts", "old": "500 old V_check episodes"},
            "score": "S_joint = S_new + S_old, fixed minus broken episodes",
            "replay_rule": instance.private["replay_rule"],
            "replay_in_fit": replay_in_fit,
        }
    if e10:
        record["limits"] = task_limits(limits)
        record["limits_source"] = E10_LIMITS_SOURCE
        executor = limits if spec.edit_limits_enforced else limits.without_edit_limits(len(instance.params))
        record["executor_limits"] = task_limits(executor)
    if joint:
        record["joint"] = {
            "oracle": "JointPatchCheckOracle",
            "parts": {"new": "condition V_check", "old": "clean companion of V_check"},
            "score": "S_joint = S_new + S_old",
            "replay_rule": REPLAY_RULE,
            "replay_in_fit": replay_in_fit,
        }
    oracle: PatchCheckOracle | JointPatchCheckOracle | None = None
    timing: dict[str, float] = {}
    try:
        if scan:
            evidence = load_scan_evidence(instance)
        elif e14:
            evidence = load_e14_evidence(instance) if evidence_override is None else evidence_override
        elif graphnew:
            evidence = load_graphnew_evidence(instance)
        elif seqskill:
            evidence = load_seqskill_evidence(instance)
        elif fewshot:
            evidence = load_fewshot_evidence(instance)
        elif evidence_override is not None:  # an E10-limit joint run with its own fit rows (PREREG T04)
            evidence = evidence_override
        else:
            evidence = load_joint_evidence(instance) if joint else load_evidence(instance)
        fit_evidence = evidence if replay_in_fit else evidence.deployment()
        record["evidence"] = fit_evidence.describe()
        if joint or fewshot or seqskill or graphnew or scan or e14:
            record["joint"]["evidence"] = evidence.describe()
        candidates, generation = generate(fit_evidence)
        timing["generate_s"] = time.perf_counter() - started
        record["generation"] = generation
        record["deviations"] = list(generation.get("deviations", []))
        slots = [candidate.slot for candidate in candidates]
        if len(set(slots)) != len(slots) or len(candidates) > budget:
            raise ValueError(f"need at most {budget} candidates with distinct slots, got slots {slots}")
        record["patch_applier"] = PATCH_APPLIER
        checked: list[dict[str, Any]] = []
        record["candidates"] = checked
        check_started = time.perf_counter()
        if candidates and scan:
            oracle = make_scan_oracle(instance, budget=budget)
        elif candidates and e14:
            from weightcode.evalkit.e14 import make_e14_oracle

            oracle = make_e14_oracle(instance, budget=budget)
        elif candidates and graphnew:
            oracle = make_graphnew_oracle(instance, budget=budget)
        elif candidates and seqskill:
            oracle = make_seqskill_oracle(instance, budget=budget)
        elif candidates and fewshot:
            oracle = make_fewshot_oracle(instance, budget=budget)
        elif candidates:
            oracle = make_joint_oracle(instance, budget=budget) if joint else make_oracle(instance, budget=budget)
        for candidate in candidates:
            entry: dict[str, Any] = {
                "slot": candidate.slot,
                "patch_id": f"q{candidate.slot}",
                "config": candidate.config,
                "notes": candidate.notes,
            }
            checked.append(entry)
            problem = invalid_reason(instance.params, candidate.child)
            if problem is not None:
                entry["invalid"] = problem
                entry["invalid_stage"] = "precheck"
                continue
            patch_dir = run_dir / entry["patch_id"]
            if candidate.ops is None:
                entry["patch_form"] = "delta"
                write_delta_patch(
                    patch_dir,
                    instance.params_sha256,
                    instance.params,
                    candidate.child,
                    candidate.hypothesis,
                    candidate.predicted_effect,
                )
            else:
                entry["patch_form"] = "ops"
                write_ops_patch(
                    patch_dir, instance.params_sha256, candidate.ops, candidate.hypothesis, candidate.predicted_effect
                )
            try:
                result = apply_patch_result(
                    instance.params,
                    patch_dir / "patch.json",
                    instance.params_sha256,
                    enforce_limits=spec.edit_limits_enforced,
                    limits=limits,
                )
            except PatchError as rejection:
                # As for an editor patch (contract 5.2): a patch the executor rejects consumes no query.
                entry["invalid"] = str(rejection)
                entry["invalid_stage"] = "executor"
                entry["executor_messages"] = list(rejection.messages)
                continue
            applied = {key: np.ascontiguousarray(value) for key, value in result.child.items()}
            # The edit size of record is the executor's count for the file as written (the delta writer can move an
            # index-form tensor to add_dense to stay under the JSON cap), never one predicted from the delta.
            exact = result.child_sha256 == params_sha256(candidate.child)
            entry["roundtrip_exact"] = exact
            if not exact:
                entry["roundtrip_entries_differing"] = count_changed(candidate.child, applied)
            entry["edit"] = edit_stats_under(instance.params, applied, limits, edit_size=result.edit_size)
            if graphnew:  # forward.py runs the whole graph on every call: one call for all evidence nodes
                (child_pred,) = predict_nodes(instance.forward, applied, evidence.x)
            else:
                child_pred = predict(instance.forward, applied, evidence.x, EVIDENCE_BATCH)
            k = evidence.n_deployment
            if k:  # an E13 around_right instance has no primitive row, so its experience has no deployment part
                entry["evidence_summary"] = summarize(evidence.y[:k], evidence.base_pred[:k], child_pred[:k], a,
                                                      b).as_dict()
            if evidence.n_replay:
                entry["replay_summary"] = summarize(evidence.y[k:], evidence.base_pred[k:], child_pred[k:]).as_dict()
            assert oracle is not None
            result = oracle.query(applied, entry["patch_id"])
            entry["query_index"] = result["query_index"]
            entry["check"] = result
        timing["check_s"] = time.perf_counter() - check_started
        if require_valid_candidate and candidates and not any("check" in entry for entry in checked):
            first = checked[0]
            raise NoValidCandidate(
                f"all {len(checked)} candidates were invalid (stages "
                f"{sorted({str(entry.get('invalid_stage')) for entry in checked})}); "
                f"{first['patch_id']}: {first.get('invalid')}"
            )
        final = oracle.final() if oracle is not None else BASE_CANDIDATE
        log = oracle.log if oracle is not None else []
        record["final"] = _final_record(instance, final, log, checked,
                                        joint=joint or fewshot or seqskill or graphnew or scan or e14)
        record["queries_used"] = len(log)
        record["n_invalid"] = sum(1 for entry in checked if "invalid" in entry)
        valid = [entry for entry in checked if "edit" in entry]
        record["flags"]["respects_edit_limits"] = all(entry["edit"]["within_limits"] for entry in valid)
        record["status"] = "complete"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if oracle is not None:
            record["oracle_log"] = oracle.log
        timing["wall_s"] = time.perf_counter() - started
        record["timing"] = timing
        write_json_atomic(run_dir / "run.json", record)
    return record


def read_run_record(path: str | Path) -> dict[str, Any]:
    """Read a run.json written by ``execute_run`` and check its schema."""
    record = json.loads(Path(path).read_text())
    if record.get("schema") != RUN_SCHEMA:
        raise ValueError(f"{path}: schema {record.get('schema')!r} != {RUN_SCHEMA!r}")
    return record
