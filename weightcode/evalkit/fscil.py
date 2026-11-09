"""Rule fscil-v1 (PREREG E9): few-shot class-incremental sessions on CIFAR-100 (the TOPIC/CEC protocol).

The student is a ResNet-20 with a cosine head over all 100 outputs, trained on the base classes 0-59 only
(``fscil_base`` in the fewshot model store of ``cifar100_fscil``). Session t (2 to 9) adds the classes
``session_classes(t)``, 5 classes with 5 labelled shots each. Predictions are the argmax over the classes seen through
session t, the outputs ``0 .. n_seen(t) - 1`` (``masked_forward``); forward.py itself returns the raw 100 logits.

An instance is one session on one current student. The builder takes its parent: the base student for session 2,
else a lineage's output at session t - 1 together with the instance it was made on. The instance content depends only
on (session, parent parameters), so that pair is its identity: lineages that reach the same student at the same
session share one instance, and every use is logged::

    $WEIGHTCODE_PRIVATE/fscil_instances/
        index.json                       {"schema", "instances": [active rows], "superseded": [rows]}
        changes.jsonl                    one line per created or used instance (lineage, parent instance, run id)
        c<8 hex>/
            model.npz                    the parent parameters (byte copy)
            model_spec.json              the base student's spec without model_id, plus the session's prediction rule
            forward.py                   byte copy of the dataset forward (raw 100 logits)
            evidence.npz                 the evidence, keys EVIDENCE_KEYS
            task.json                    schema weightcode.task.incremental.v1
            private.json                 schema weightcode.instance_private.fscil.v1, never editor-visible

Evidence of rule fscil-v1 (no draw depends on the student, so every lineage sees the same rows at a session):

- new: the 25 listed shots of session t (V_evidence, stored in session order then list order);
- old: the replay of rule fscil-replay-v1, N_REPLAY rows of old_replay (held out of base training) drawn uniformly
  without replacement with ``default_rng([SPLIT_SEED, REPLAY_STREAM])``, one draw shared by every session;
- prev: the shots of sessions 2 .. t - 1 with their session (PREREG E9: the learner's own labelled history);
- pred_*: the current student's masked argmax; split ids: ``e`` + 12 hex from a seed of (parent, session, rule).

Check (``make_fscil_oracle``): S_new over the V_check rows of session t's classes, S_old over old_check (the base
classes) plus the V_check rows of the classes of sessions 2 .. t - 1, under the session-t mask and against the current
student (``JointPatchCheckOracle``). Readouts (``fscil_readout``, ``fscil_accuracy``) on V_select for the incremental
classes and on old_select for the base classes, pooled and class-balanced, under the session-t mask.

Limits: task.json states the default ``weightcode.patch.Limits`` with the head (``fc.weight``) exempt from max_rel_frob;
1-D tensors (every BatchNorm tensor) are exempt by rank. Programs and references are held to
``fscil_program_limits``: the edit-size limit off and at most 20% of the entries changed, counted over every tensor.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import Params, load_params, params_sha256
from weightcode.common.paths import private_dir, runs_dir, sealed_dir
from weightcode.data.registry import SPLIT_SEED
from weightcode.evalkit.evidence import SPLIT_ID_HEX_DIGITS, _seed_from_text
from weightcode.evalkit.instances import (
    _FINAL_SELECTION,
    QUERY_BUDGET,
    TRAINING_FREE_RULE,
    leak_tokens,
    limit_definitions,
    npz_text,
    patch_limits,
    scan_for_tokens,
)
from weightcode.evalkit.patch_check import ForwardFn, JointPatchCheckOracle, predict, validate_child
from weightcode.evalkit.targets import (
    class_names,
    file_sha256,
    json_text,
    jsonable,
    load_checked_split,
    load_dataset_forward,
    no_bytecode_cache,
    read_dataset_meta,
    read_json,
    split_path,
    utc_now,
    write_json_atomic,
)
from weightcode.models.export import fewshot_models_dir, read_fewshot_index
from weightcode.patch import OP_NAMES, Limits
from weightcode.patch import SCHEMA as PATCH_SCHEMA
from weightcode.patch.limits import SIZE_LIMIT_FIELDS

DATASET = "cifar100_fscil"
FSCIL_RULE = "fscil-v1"
REPLAY_RULE = "fscil-replay-v1"
REPLAY_STREAM = 709
N_REPLAY = 64
N_CLASSES = 100
N_BASE_CLASSES = 60
N_WAYS = 5
N_SHOTS = 5
FIRST_SESSION = 2
LAST_SESSION = 9
SESSIONS: tuple[int, ...] = tuple(range(FIRST_SESSION, LAST_SESSION + 1))
TASK_SCHEMA = "weightcode.task.incremental.v1"
"""Editor-visible, so it may not contain "fscil" (a part of the dataset name, a leak token)."""
PRIVATE_SCHEMA = "weightcode.instance_private.fscil.v1"
INDEX_SCHEMA = "weightcode.fscil_instances_index.v1"
ROOT_NAME = "fscil_instances"
CHANGES_LOG_NAME = "changes.jsonl"
INSTANCE_ID_RE = re.compile(r"^c[0-9a-f]{8}$")
LINEAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
EVIDENCE_KEYS: tuple[str, ...] = (
    "x_new", "y_new", "pred_new", "split_ids_new",
    "x_old", "y_old", "pred_old", "split_ids_old",
    "x_prev", "y_prev", "pred_prev", "session_prev", "split_ids_prev",
)
EDITOR_VISIBLE_FILES: tuple[str, ...] = ("model.npz", "model_spec.json", "forward.py", "evidence.npz", "task.json")
SPLITS: tuple[str, ...] = ("V_evidence", "V_check", "V_select", "old_check", "old_select", "old_replay")
HEAD_OP = "cosine_linear"
HEAD_TENSOR = "fc.weight"
FSCIL_MAX_CHANGED_FRAC = 0.20
"""PREREG E9 (E2's limits): an update may change at most 20% of the parameter entries, counted over every tensor."""
BASE_ROLE = "fscil_base"
ALLCLASS_ROLE = "fscil_allclass"
ARCH = "resnet20_cosine"
DEV_SEED = 100
EVALUATION_BATCH = 256
"""Forward batch of the build, the oracle and the readouts (numpy float64 ResNet-20: about 0.6 GB at 256)."""
READOUT_PARTS: tuple[str, ...] = ("new", "prev", "old", "seen")
DATASET_WORDS: tuple[str, ...] = ("cifar", "cifar100", "cifar-100", "cifar_100", "fscil", "toronto", "krizhevsky")
"""Words naming the dataset or its source; with leak_tokens(DATASET) and the store roots they form the leak scan."""
_FLOOR_CLAMP = -20
_FLOOR_CLAMP_R = -1.0
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_PROGRAM_SIZE_FIELDS = ("max_rel_frob", "max_changed_frac")


class FscilError(RuntimeError):
    """An fscil instance, its store or its inputs were refused; nothing is overwritten."""


class StaleFscilInstance(FscilError):
    """An instance that is not active or whose files the current code would not write; it is not run."""


# ---------------------------------------------------------------------------------------------------------------
# Session arithmetic (pure)


def _check_session(session: int, first: int = 1) -> int:
    t = int(session)
    if not first <= t <= LAST_SESSION or t != session:
        raise FscilError(f"session must be an integer in {first}..{LAST_SESSION}, got {session!r}")
    return t


def session_classes(session: int) -> list[int]:
    """Classes added in a session: 0-59 in session 1 (the base), 60 + 5(t - 2) .. 64 + 5(t - 2) in session t >= 2."""
    t = _check_session(session)
    if t == 1:
        return list(range(N_BASE_CLASSES))
    start = N_BASE_CLASSES + N_WAYS * (t - 2)
    return list(range(start, start + N_WAYS))


def n_seen(session: int) -> int:
    """Number of classes seen through a session, 60 + 5(t - 1); the mask keeps outputs 0 .. n_seen - 1."""
    return N_BASE_CLASSES + N_WAYS * (_check_session(session) - 1)


def seen_classes(session: int) -> list[int]:
    """Classes seen through a session, in label order (a prefix of the 100 labels)."""
    return list(range(n_seen(session)))


def prev_classes(session: int) -> list[int]:
    """Incremental classes of the sessions before ``session`` (2 .. t - 1); empty for sessions 1 and 2."""
    t = _check_session(session)
    return list(range(N_BASE_CLASSES, N_BASE_CLASSES + N_WAYS * max(t - 2, 0)))


def session_of_class(label: int) -> int:
    """The session that adds class ``label``."""
    c = int(label)
    if not 0 <= c < N_CLASSES:
        raise FscilError(f"class {label!r} outside 0..{N_CLASSES - 1}")
    return 1 if c < N_BASE_CLASSES else 2 + (c - N_BASE_CLASSES) // N_WAYS


def masked_forward(forward: ForwardFn, session: int) -> ForwardFn:
    """The forward under the session's prediction rule: the logits of the seen classes only.

    The seen classes are the prefix 0 .. n_seen - 1, so the argmax index of the masked logits is the label, and the
    logits stay finite (``predict`` refuses non-finite logits, so masking by -inf is not an option).
    """
    k = n_seen(session)

    def forward_masked(params: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
        logits = np.asarray(forward(params, x))
        if logits.ndim != 2 or logits.shape[1] != N_CLASSES:
            raise FscilError(f"forward returned logits of shape {logits.shape}, expected [N, {N_CLASSES}]")
        return logits[:, :k]

    return forward_masked


def old_part_sizes(session: int, meta: Mapping[str, Any] | None = None) -> tuple[int, int]:
    """(n_c, n_s): sizes of the old part of the session's check (old_check plus the V_check rows of the earlier
    incremental classes) and of its V_select readout (old_select plus the V_select rows of those classes).

    From dataset.json class counts (``meta``, read from the store when None): 1800 + 500(t - 2) and 1800 + 1000(t - 2)
    on the real store.
    """
    t = _check_session(session, FIRST_SESSION)
    meta = read_dataset_meta(DATASET) if meta is None else meta
    prev = prev_classes(t)
    counts = {split: np.asarray(meta["splits"][split]["class_counts"], dtype=np.int64)
              for split in ("old_check", "old_select", "V_check", "V_select")}
    n_c = int(counts["old_check"].sum() + counts["V_check"][prev].sum())
    n_s = int(counts["old_select"].sum() + counts["V_select"][prev].sum())
    return n_c, n_s


def joint_retain_floor(n_c: int, n_s: int, r_points: float, d: float) -> int:
    """The S_old floor of kept rule joint-retain-v1 (PREREG E7 with the amendment of 2026-11-04 14:46; E9 per session).

    floor = nearest integer (halves away from zero) of (n_c / 100) (R - 0.11 + 164.5 sqrt(d / n_c + d / n_s)), never above 0. The lower clamp at -20 rows
    applies only when R = -1.0 (R >= -1.0); when R comes from V3 (R < -1.0) there is no lower clamp.
    """
    if n_c <= 0 or n_s <= 0 or not math.isfinite(r_points) or not (math.isfinite(d) and d >= 0):
        raise FscilError(f"invalid floor inputs n_c={n_c}, n_s={n_s}, R={r_points}, d={d}")
    raw = (n_c / 100.0) * (r_points - 0.11 + 164.5 * math.sqrt(d / n_c + d / n_s))
    value = int(math.floor(abs(raw) + 0.5)) * (1 if raw >= 0 else -1)  # nearest, halves away from zero (E7)
    if r_points >= _FLOOR_CLAMP_R:
        value = max(value, _FLOOR_CLAMP)
    return int(min(value, 0))


# ---------------------------------------------------------------------------------------------------------------
# E9 v2: class-balanced check score fscil-cb-v1 and kept rule fscil-retain-v2 (PREREG "E9 v2", 2026-11-04)

CHECK_SCORE_RULE = "fscil-cb-v1"
KEPT_RULE_V2 = "fscil-retain-v2"
CB_PARTS = ("new", "prev", "old")
"""Row labels of a check under fscil-cb-v1, as ``readout_rows`` names them: session t's classes, the incremental classes
of sessions 2 .. t - 1, and the base classes."""
V2_BASE_ROWS = 1800
"""Base rows of V_check (old_check) and of V_select (old_select) on the real store, 30 per base class: fscil-retain-v2's
n_c and n_s."""
V2_TIE_POINTS = 1e-9
"""Two check scores (points) within this distance are a tie under fscil-retain-v2 (S is a float mean)."""


def cb_check_rows(y_new: np.ndarray, y_old: np.ndarray, session: int) -> tuple[np.ndarray, np.ndarray]:
    """(y, part) of a session's check in the joint oracle's row order: ``fscil_check_parts``' new rows, then its old
    rows (old_check's base rows, then the V_check rows of the classes of sessions 2 .. t - 1), labelled as in
    ``CB_PARTS`` (pure)."""
    t = _check_session(session, FIRST_SESSION)
    y_new = np.asarray(y_new).astype(np.int64)
    y_old = np.asarray(y_old).astype(np.int64)
    if not np.all(np.isin(y_new, session_classes(t))):
        raise FscilError(f"a new row of the check is not of a class of session {t}")
    base = y_old < N_BASE_CLASSES
    if not np.all(base | np.isin(y_old, prev_classes(t))):
        raise FscilError(f"an old row of the check is neither a base class nor a class of sessions 2 .. {t - 1}")
    part = np.concatenate([np.full(len(y_new), "new"), np.where(base, "old", "prev")])
    return np.concatenate([y_new, y_old]), part


def cb_check_score(
    base_pred: np.ndarray, child_pred: np.ndarray, y: np.ndarray, part: np.ndarray, session: int
) -> dict[str, Any]:
    """Check score fscil-cb-v1 of one child against the session's student on V_check under the session-t mask (pure).

    For each seen class c, delta_c = (fixed_c - broken_c) / n_c over that class's rows. ``S`` = 100 x the mean of
    delta_c over the seen classes (the change of all-seen class-balanced accuracy, in points); ``S_new``, ``S_prev``
    and ``S_base`` are the contributions of session t's classes, the earlier incremental classes and the base classes
    (each 100 x their delta_c summed over the seen-class count), so they sum to S. ``B_c`` = 100 x the mean of delta_c
    over the base classes (the base-class change in points); ``d_c`` = (fixed + broken) / n over the base rows, the
    child's own base flip rate. ``part`` labels each row as in ``CB_PARTS`` and must agree with ``y``.
    """
    t = _check_session(session, FIRST_SESSION)
    y = np.asarray(y).astype(np.int64)
    base_pred, child_pred, part = np.asarray(base_pred), np.asarray(child_pred), np.asarray(part)
    if not (y.shape == base_pred.shape == child_pred.shape == part.shape) or y.ndim != 1:
        raise FscilError(f"check arrays differ in shape: y {y.shape}, base {base_pred.shape}, child "
                         f"{child_pred.shape}, part {part.shape}")
    expected = np.where(np.isin(y, session_classes(t)), "new",
                        np.where(y < N_BASE_CLASSES, "old", np.where(np.isin(y, prev_classes(t)), "prev", "")))
    if np.any(expected == ""):
        raise FscilError(f"a check row is of a class not seen at session {t}")
    if not np.array_equal(part.astype(str), expected):
        raise FscilError(f"part labels disagree with y at session {t} (labels {CB_PARTS})")
    base_right, child_right = base_pred == y, child_pred == y
    fixed_rows, broken_rows = ~base_right & child_right, base_right & ~child_right
    seen = seen_classes(t)
    groups = {"new": set(session_classes(t)), "prev": set(prev_classes(t))}
    per_class: dict[str, dict[str, Any]] = {}
    sums = {"new": 0.0, "prev": 0.0, "old": 0.0}
    for c in seen:
        rows = y == c
        n_c = int(rows.sum())
        if n_c == 0:
            raise FscilError(f"the check has no row of seen class {c} at session {t}")
        fixed_c, broken_c = int(np.count_nonzero(fixed_rows & rows)), int(np.count_nonzero(broken_rows & rows))
        delta_c = (fixed_c - broken_c) / n_c
        per_class[str(c)] = {"n": n_c, "fixed": fixed_c, "broken": broken_c, "delta": delta_c}
        group = "new" if c in groups["new"] else "prev" if c in groups["prev"] else "old"
        sums[group] += delta_c
    k = len(seen)
    base_rows = part == "old"
    fixed_base, broken_base = int(np.count_nonzero(fixed_rows & base_rows)), int(np.count_nonzero(broken_rows & base_rows))
    n_base = int(base_rows.sum())
    s_new, s_prev, s_base = (100.0 * sums[g] / k for g in ("new", "prev", "old"))
    return {
        "rule": CHECK_SCORE_RULE, "session": t, "n_seen_classes": k, "n": int(len(y)),
        "S": 100.0 * (sums["new"] + sums["prev"] + sums["old"]) / k, "S_new": s_new, "S_prev": s_prev, "S_base": s_base,
        "B_c": 100.0 * sums["old"] / N_BASE_CLASSES, "d_c": (fixed_base + broken_base) / n_base,
        "fixed_base": fixed_base, "broken_base": broken_base, "n_base": n_base,
        "fixed": int(np.count_nonzero(fixed_rows)), "broken": int(np.count_nonzero(broken_rows)),
        "per_class": per_class,
    }


def retain_v2_floor(d_c: float, r_points: float = -1.0, n_c: int = V2_BASE_ROWS, n_s: int = V2_BASE_ROWS) -> float:
    """floor_c of kept rule fscil-retain-v2, in points: min(0, R - 0.11 + 164.5 sqrt(d_c / n_c + d_c / n_s)).

    E7's formula in points with the candidate's own base flip rate d_c; it lies between R - 0.11 and 0. Not rounded.
    """
    if n_c <= 0 or n_s <= 0 or not math.isfinite(r_points) or not (math.isfinite(d_c) and 0.0 <= d_c <= 1.0):
        raise FscilError(f"invalid floor inputs d_c={d_c}, R={r_points}, n_c={n_c}, n_s={n_s}")
    return min(0.0, r_points - 0.11 + 164.5 * math.sqrt(d_c / n_c + d_c / n_s))


def retain_v2_pick(
    scores: Sequence[Mapping[str, Any]], r_points: float = -1.0, n_c: int = V2_BASE_ROWS, n_s: int = V2_BASE_ROWS
) -> int | None:
    """Kept rule fscil-retain-v2: the index of the argmax S among the checked programs with B_c >= floor_c, or None to
    keep the session's student.

    The student is in the pool at S = 0 and B_c = 0 (it always passes, floor_c <= 0), so a program is kept only when
    its S exceeds 0 by more than ``V2_TIE_POINTS``; among programs, S within ``V2_TIE_POINTS`` of the best is a tie,
    broken by the lowest index. This is joint-retain-v1's semantics (WEIGHTCODE-main, 2026-11-04, before any v2 code).
    ``scores`` are ``cb_check_score`` dicts (S, B_c, d_c) in query order.
    """
    eligible = [(i, float(s["S"])) for i, s in enumerate(scores)
                if float(s["B_c"]) >= retain_v2_floor(float(s["d_c"]), r_points, n_c, n_s)]
    if not eligible:
        return None
    best = max(score for _, score in eligible)
    if best <= V2_TIE_POINTS:
        return None
    return min(i for i, score in eligible if score >= best - V2_TIE_POINTS)


# ---------------------------------------------------------------------------------------------------------------
# Limits


def head_tensors(spec: Mapping[str, Any]) -> list[str]:
    """The head of a ResNet-20 student: the last layer of model_spec.json, which must be the cosine head."""
    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers:
        raise FscilError("model_spec.json has no layers")
    last = layers[-1]
    if last.get("op") != HEAD_OP or last.get("weight") != HEAD_TENSOR:
        raise FscilError(f"model_spec.json must end in {{op: {HEAD_OP}, weight: {HEAD_TENSOR}}}, got {last}")
    return [HEAD_TENSOR]


def fscil_limits(spec: Mapping[str, Any]) -> Limits:
    """The default limits with the head exempt from max_rel_frob (PREREG E9, as E2)."""
    return Limits(rel_frob_exempt=tuple(head_tensors(spec)))


def fscil_program_limits(task_limits_json: Mapping[str, Any]) -> Limits:
    """The limits of an E9 update and every E9 reference: task.json's limits with the edit-size limit off and at most
    ``FSCIL_MAX_CHANGED_FRAC`` of the entries changed (E2's ``fewshot_program_limits``). max_ops is the harness's."""
    if not isinstance(task_limits_json, Mapping):
        raise FscilError(f"task.json limits must be an object, got {type(task_limits_json).__name__}")
    limits = Limits(**dict(task_limits_json))
    disabled = {name: None for name in SIZE_LIMIT_FIELDS if name not in _PROGRAM_SIZE_FIELDS}
    return dataclasses.replace(limits, **disabled, max_changed_frac=FSCIL_MAX_CHANGED_FRAC)


# ---------------------------------------------------------------------------------------------------------------
# Rows (pure)


def draw_replay_rows(n_old_replay: int) -> np.ndarray:
    """Rule fscil-replay-v1: N_REPLAY rows of old_replay, uniform without replacement, one draw for every session."""
    if n_old_replay < N_REPLAY:
        raise FscilError(f"old_replay has {n_old_replay} rows; the replay needs {N_REPLAY}")
    rng = np.random.default_rng([SPLIT_SEED, REPLAY_STREAM])
    return rng.choice(n_old_replay, size=N_REPLAY, replace=False).astype(np.int64)


def shot_rows(y_evidence: np.ndarray, sessions: Sequence[int]) -> np.ndarray:
    """Rows of V_evidence whose class belongs to one of ``sessions``, in stored order (session order, then list order).

    Refuses a session whose shots are not N_WAYS classes of N_SHOTS rows each.
    """
    y = np.asarray(y_evidence).astype(np.int64)
    rows = []
    for t in sessions:
        classes = session_classes(_check_session(t, FIRST_SESSION))
        hit = np.flatnonzero(np.isin(y, classes))
        counts = np.bincount(y[hit] - classes[0], minlength=N_WAYS)
        if counts.shape != (N_WAYS,) or not np.all(counts == N_SHOTS):
            raise FscilError(f"V_evidence holds {counts.tolist()} shots of the classes of session {t}")
        rows.append(hit)
    return np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, dtype=np.int64)


def fscil_check_parts(
    v_check: Mapping[str, np.ndarray], old_check: Mapping[str, np.ndarray], session: int
) -> dict[str, np.ndarray]:
    """The session's check: new = V_check rows of session t's classes; old = old_check (base classes) followed by the
    V_check rows of the classes of sessions 2 .. t - 1. Each part keeps its split's row order (pure)."""
    t = _check_session(session, FIRST_SESSION)
    x_ch, y_ch = np.asarray(v_check["x"]), np.asarray(v_check["y"]).astype(np.int64)
    x_oc, y_oc = np.asarray(old_check["x"]), np.asarray(old_check["y"]).astype(np.int64)
    if np.any(y_oc >= N_BASE_CLASSES):
        raise FscilError("old_check holds a row of an incremental class")
    new = np.isin(y_ch, session_classes(t))
    prev = np.isin(y_ch, prev_classes(t))
    if not new.any():
        raise FscilError(f"V_check has no row of the classes of session {t}")
    return {
        "x_new": np.ascontiguousarray(x_ch[new]),
        "y_new": y_ch[new],
        "x_old": np.ascontiguousarray(np.concatenate([x_oc, x_ch[prev]])),
        "y_old": np.concatenate([y_oc, y_ch[prev]]),
    }


def _ids(seed: int, n: int) -> list[str]:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16**SPLIT_ID_HEX_DIGITS, size=n, dtype=np.int64)
    if np.unique(codes).size != codes.size:
        raise FscilError(f"split id collision for seed {seed}; the id width must grow (rule change)")
    return [f"e{int(code):0{SPLIT_ID_HEX_DIGITS}x}" for code in codes]


def fscil_seed(parent_sha256: str, session: int) -> int:
    """Seed of the split ids of one instance (the rows themselves are fixed by the rule)."""
    if not _SHA_RE.fullmatch(parent_sha256):
        raise ValueError(f"not a sha256 hex digest: {parent_sha256!r}")
    return _seed_from_text(f"{parent_sha256}|{int(session)}|{DATASET}|{FSCIL_RULE}")


# ---------------------------------------------------------------------------------------------------------------
# Task and leak scan


def _class_list(names: Sequence[str], classes: Sequence[int]) -> str:
    return ", ".join(f"{c} ({names[c]})" for c in classes)


def build_fscil_task(names: Sequence[str], session: int, limits: Limits) -> dict[str, Any]:
    """Editor-visible task record of rule fscil-v1 (schema weightcode.task.incremental.v1)."""
    t = _check_session(session, FIRST_SESSION)
    if len(names) != N_CLASSES:
        raise FscilError(f"expected {N_CLASSES} class names, got {len(names)}")
    new, prev, k = session_classes(t), prev_classes(t), n_seen(t)
    n_prev = len(prev) * N_SHOTS
    earlier = "session 2" if t == 3 else f"sessions 2 to {t - 1}"
    history = (
        f" The classes {_class_list(names, prev)} were added in {earlier}; the classifier given here is "
        "the one that came out of those sessions."
        if prev else ""
    )
    prev_text = (f", the {N_SHOTS} labelled images of each class added in the earlier sessions ({n_prev} images, "
                 "with their session)" if prev else "")
    stated = patch_limits(limits)
    return {
        "schema": TASK_SCHEMA,
        "statement": (
            f"This image classifier has {N_CLASSES} outputs, one per class. Its prediction is the highest-scoring "
            f"output among the classes seen so far, outputs 0 to {k - 1}; the other outputs are ignored. It was "
            f"first trained only on images of the base classes 0 to {N_BASE_CLASSES - 1}. New classes arrive in "
            f"sessions of {N_WAYS} classes with {N_SHOTS} labelled images each. This is session {t}: the classes "
            f"{_class_list(names, new)} are added now.{history} The evidence file holds the {N_SHOTS} labelled images "
            f"of each new class, {N_REPLAY} labelled images of the base classes{prev_text}, with the classifier's "
            "predictions under this rule. Change the classifier's weights directly, without training it, so that it "
            "also recognises the new classes on new images, while the images of the classes seen before that it "
            "classifies correctly stay correct."
        ),
        "classes": list(names),
        "session": t,
        "n_seen": k,
        "base_classes": list(range(N_BASE_CLASSES)),
        "prev_classes": prev,
        "new_classes": new,
        "seen_classes": seen_classes(t),
        "prediction_rule": {
            "rule": f"argmax over outputs 0 .. {k - 1} (the classes seen through this session); forward.py returns "
            f"all {N_CLASSES} logits",
            "n_seen": k,
        },
        "evidence": {
            "new": {
                "keys": ["x_new", "y_new", "pred_new"],
                "count": len(new) * N_SHOTS,
                "per_class": N_SHOTS,
                "description": "the labelled images of the classes added in this session, in class order",
            },
            "old": {
                "keys": ["x_old", "y_old", "pred_old"],
                "count": N_REPLAY,
                "description": "labelled images of the base classes, a fixed random draw from held-out images of "
                "those classes, the same in every session",
            },
            "prev": {
                "keys": ["x_prev", "y_prev", "pred_prev", "session_prev"],
                "count": n_prev,
                "per_class": N_SHOTS,
                "description": "the labelled images given in the earlier sessions, in session order, with the session "
                "of each (empty in session 2)",
            },
            "ids": {"keys": ["split_ids_new", "split_ids_old", "split_ids_prev"],
                    "description": "one opaque id per example"},
        },
        "patch_check": {
            "budget_queries": QUERY_BUDGET,
            "readout": "each query applies one patch and returns aggregate numbers on a held-out check split, each "
            "against the classifier given here and under the prediction rule above, for two parts, the images of the "
            "classes added in this session and the images of the classes seen before (the base classes and the "
            "classes of the earlier sessions): accuracy, fixed (images the given classifier got wrong that the "
            "patched one gets right), broken (images the given classifier got right that the patched one gets "
            "wrong), the changed-parameter count, S_new and S_old (fixed - broken on each part) and S = S_new + "
            "S_old; no per-image information",
            "invalid_patches": "a patch that fails validation consumes no query",
            "final_selection": _FINAL_SELECTION,
        },
        "patch_format": {"schema": PATCH_SCHEMA, "allowed_ops": list(OP_NAMES)},
        "limits": stated,
        "limit_definitions": limit_definitions(stated),
        "training_free_rule": TRAINING_FREE_RULE,
    }


def fscil_leak_tokens(extra: Sequence[str] = ()) -> list[str]:
    """Lower-case tokens no editor-visible file may contain: ``leak_tokens(DATASET)``, ``DATASET_WORDS``, the store
    roots and ``extra`` (instance, session and run ids)."""
    roots = [str(path).lower() for path in (private_dir(), runs_dir(), sealed_dir())]
    return sorted(set(leak_tokens(DATASET)) | set(DATASET_WORDS) | set(roots) | {str(e).lower() for e in extra if e})


def fscil_scan(text: str, extra: Sequence[str] = ()) -> list[str]:
    """Leak tokens found in ``text`` (``scan_for_tokens``)."""
    return scan_for_tokens(text, fscil_leak_tokens(extra))


def _scan_visible(tokens: Sequence[str], **texts: str) -> None:
    for where, text in texts.items():
        found = scan_for_tokens(text, tokens)
        if found:
            raise FscilError(f"{where} contains {found}; no editor-visible file may")


# ---------------------------------------------------------------------------------------------------------------
# Store: index.json, changes.jsonl, lock


def fscil_root() -> Path:
    """``$WEIGHTCODE_PRIVATE/fscil_instances`` (created if missing)."""
    root = private_dir() / ROOT_NAME
    root.mkdir(exist_ok=True)
    return root


@contextmanager
def _exclusive_lock(root: Path) -> Iterator[None]:
    descriptor = os.open(root / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def read_fscil_instance_index() -> dict[str, Any]:
    """The fscil instance index; an empty one before the first build."""
    path = fscil_root() / "index.json"
    if not path.exists():
        return {"schema": INDEX_SCHEMA, "instances": [], "superseded": []}
    index = read_json(path)
    if not isinstance(index, dict) or index.get("schema") != INDEX_SCHEMA:
        raise FscilError(f"{path} is not a {INDEX_SCHEMA} index")
    return index


def active_fscil_ids(session: int | None = None) -> list[str]:
    """Ids of the active fscil instances, optionally of one session, in index order."""
    rows = read_fscil_instance_index()["instances"]
    return [str(row["instance_id"]) for row in rows if session is None or int(row["session"]) == int(session)]


def fscil_instance_dir(instance_id: str) -> Path:
    """Directory of one fscil instance; refuses an id of another format."""
    if not INSTANCE_ID_RE.fullmatch(str(instance_id)):
        raise FscilError(f"{instance_id!r} is not an fscil instance id (c + 8 hex)")
    return fscil_root() / str(instance_id)


def load_fscil_instance(instance: Instance | str) -> Instance:
    """An Instance, or the fscil instance with this id loaded and hash-checked (``load_instance``)."""
    return instance if isinstance(instance, Instance) else load_instance(fscil_instance_dir(instance))


def _active_row(instance_id: str) -> dict[str, Any]:
    index = read_fscil_instance_index()
    rows = [row for row in index["instances"] if row["instance_id"] == instance_id]
    if not rows:
        retired = {row["instance_id"] for row in index["superseded"]}
        status = "superseded" if instance_id in retired else "not in the index"
        raise StaleFscilInstance(f"fscil instance {instance_id} is {status}; only active instances are run")
    return rows[0]


def instance_chain(instance_id: str) -> list[str]:
    """The instance and its ancestors through parent_instance links, newest first, ending at a session-2 instance."""
    chain = [str(instance_id)]
    while True:
        row = _active_row(chain[-1])
        parent = row.get("parent_instance")
        if parent is None:
            return chain
        if parent in chain:
            raise FscilError(f"parent_instance cycle at {parent}")
        chain.append(str(parent))


_INDEX_KEYS = ("instance_id", "dataset", "session", "n_seen", "parent_kind", "parent_params_sha256", "parent_instance",
               "base_model_id", "evidence_rule", "created_at")


def _index_row(private: Mapping[str, Any]) -> dict[str, Any]:
    return {key: private[key] for key in _INDEX_KEYS}


def _new_instance_id(root: Path) -> str:
    while True:
        instance_id = "c" + secrets.token_hex(4)
        if not (root / instance_id).exists() and not (root / f".tmp-{instance_id}").exists():
            return instance_id


def _log_change(root: Path, payload: Mapping[str, Any]) -> None:
    with open(root / CHANGES_LOG_NAME, "a", encoding="utf-8") as log:
        log.write(json.dumps(jsonable(dict(payload)), sort_keys=True) + "\n")


# ---------------------------------------------------------------------------------------------------------------
# Build


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    if tuple(sorted(arrays)) != tuple(sorted(EVIDENCE_KEYS)):
        raise FscilError(f"evidence keys {sorted(arrays)} != {sorted(EVIDENCE_KEYS)}")
    return params_sha256(arrays)


def _save_evidence(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    expected = _evidence_content_sha256(arrays)
    np.savez(path, **{key: np.ascontiguousarray(arrays[key]) for key in EVIDENCE_KEYS})
    with np.load(path, allow_pickle=False) as archive:
        reloaded = {key: archive[key] for key in archive.files}
    if _evidence_content_sha256(reloaded) != expected:
        raise FscilError(f"{path}: evidence content hash changed on reload")
    return expected


def _load_splits(meta: Mapping[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
    data, record = {}, {}
    for split in SPLITS:
        path = split_path(DATASET, split)
        expected = meta["splits"][split]["sha256"]
        data[split] = load_checked_split(path, expected)
        record[split] = {"path": str(path), "sha256": file_sha256(path), "content_sha256": expected}
    return data, record


def base_student(seed: int = DEV_SEED) -> dict[str, Any]:
    """The fewshot index row of the session-1 student (role fscil_base) of one seed; exactly one must exist."""
    rows = [row for row in read_fewshot_index(DATASET)
            if row.get("role") == BASE_ROLE and row.get("arch") == ARCH and int(row["seed"]) == int(seed)]
    if len(rows) != 1:
        raise FscilError(f"{len(rows)} {BASE_ROLE} students of {DATASET} with seed {seed}; exactly one is needed")
    return rows[0]


def model_record(row: Mapping[str, Any], role: str) -> tuple[Path, Params, dict[str, Any], str]:
    """(model dir, params, private.json, model_spec.json text) of a fewshot-store student, hash-checked."""
    source = fewshot_models_dir(DATASET) / str(row["model_id"])
    params = load_params(source / "model.npz")
    if params_sha256(params) != row["params_sha256"]:
        raise FscilError(f"{source}/model.npz hash differs from the fewshot index")
    private = read_json(source / "private.json")
    if private.get("role") != role:
        raise FscilError(f"{source}/private.json has role {private.get('role')!r}, expected {role!r}")
    return source, params, private, (source / "model_spec.json").read_text(encoding="utf-8")


def _instance_spec(base_spec_text: str, session: int) -> str:
    """The instance's model_spec.json: the base student's spec without its ``model_id`` (a store id, kept in
    private.json as base_model_id; the harness drops it the same way) plus the session's prediction block."""
    spec = json.loads(base_spec_text)
    spec.pop("model_id", None)
    if "prediction" in spec:
        raise FscilError("the base model_spec.json already has a prediction block")
    spec["prediction"] = {
        "rule": f"argmax over outputs 0 .. {n_seen(session) - 1} (the classes seen through session {session})",
        "n_seen": n_seen(session),
        "session": int(session),
    }
    return json_text(spec)


def _plan(
    session: int,
    parent_params: Params,
    parent_kind: str,
    parent_instance: str | None,
    base_row: Mapping[str, Any],
    base_source: Path,
    base_spec_text: str,
    names: Sequence[str],
    data: Mapping[str, Mapping[str, np.ndarray]],
    split_record: Mapping[str, Mapping[str, Any]],
    forward: ForwardFn,
    forward_sha: str,
    tokens: Sequence[str],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str, dict[str, Any]]:
    """Evidence arrays, task, instance spec text and private record (without id and time) of one instance."""
    t = session
    spec_text = _instance_spec(base_spec_text, t)
    limits = fscil_limits(json.loads(spec_text))
    if HEAD_TENSOR not in parent_params:
        raise FscilError(f"{HEAD_TENSOR} is not in the parent parameters")
    digest = params_sha256(parent_params)
    masked = masked_forward(forward, t)

    v_ev, replay_pool = data["V_evidence"], data["old_replay"]
    new_rows = shot_rows(v_ev["y"], [t])
    prev_rows = shot_rows(v_ev["y"], list(range(FIRST_SESSION, t)))
    replay_rows = draw_replay_rows(len(replay_pool["y"]))
    seed = fscil_seed(digest, t)
    ids = _ids(seed, len(new_rows) + N_REPLAY + len(prev_rows))
    ids_new, ids_old, ids_prev = (ids[: len(new_rows)], ids[len(new_rows) : len(new_rows) + N_REPLAY],
                                  ids[len(new_rows) + N_REPLAY :])

    def part(source: Mapping[str, np.ndarray], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.ascontiguousarray(source["x"][rows])
        y = np.asarray(source["y"][rows]).astype(np.int64)
        pred = (predict(masked, parent_params, x, batch_size) if len(rows)
                else np.zeros(0, dtype=np.int64)).astype(np.int64)
        return x, y, pred

    x_new, y_new, pred_new = part(v_ev, new_rows)
    x_old, y_old, pred_old = part(replay_pool, replay_rows)
    x_prev, y_prev, pred_prev = part(v_ev, prev_rows)
    if np.any(y_old >= N_BASE_CLASSES):
        raise FscilError("old_replay holds a row of an incremental class")
    arrays = {
        "x_new": x_new, "y_new": y_new, "pred_new": pred_new, "split_ids_new": np.array(ids_new, dtype=f"<U{1 + SPLIT_ID_HEX_DIGITS}"),
        "x_old": x_old, "y_old": y_old, "pred_old": pred_old, "split_ids_old": np.array(ids_old, dtype=f"<U{1 + SPLIT_ID_HEX_DIGITS}"),
        "x_prev": x_prev, "y_prev": y_prev, "pred_prev": pred_prev,
        "session_prev": np.array([session_of_class(c) for c in y_prev], dtype=np.int64),
        "split_ids_prev": np.array(ids_prev, dtype=f"<U{1 + SPLIT_ID_HEX_DIGITS}"),
    }
    task = build_fscil_task(names, t, limits)
    task_text = json_text(task)
    visible_tokens = sorted(set(tokens) | {str(base_row["model_id"]).lower()}
                            | ({str(parent_instance).lower()} if parent_instance else set()))
    _scan_visible(visible_tokens, **{"task.json": task_text, "evidence.npz": npz_text(arrays),
                                     "model_spec.json": spec_text, "model.npz names": npz_text(parent_params)})

    parts = fscil_check_parts(data["V_check"], data["old_check"], t)
    pred_new_ch = predict(masked, parent_params, parts["x_new"], batch_size)
    pred_old_ch = predict(masked, parent_params, parts["x_old"], batch_size)
    v_sel, o_sel = data["V_select"], data["old_select"]
    private = {
        "schema": PRIVATE_SCHEMA,
        "evidence_rule": FSCIL_RULE,
        "dataset": DATASET,
        "session": t,
        "n_seen": n_seen(t),
        "new_classes": session_classes(t),
        "prev_classes": prev_classes(t),
        "parent_kind": parent_kind,
        "parent_params_sha256": digest,
        "parent_instance": parent_instance,
        "base_model_id": base_row["model_id"],
        "base_params_sha256": base_row["params_sha256"],
        "seed": int(base_row["seed"]),
        "arch": base_row["arch"],
        "role": "dev",
        "split_role": "dev",
        "model_sha256": digest,
        "model_spec_sha256": _text_sha256(spec_text),
        "base_spec_sha256": _text_sha256(base_spec_text),
        "base_model_source_path": str(base_source / "model.npz"),
        "forward_sha256": forward_sha,
        "head_tensors": list(limits.rel_frob_exempt),
        **{f"{key}_{field}": split_record[split][field]
           for split, key in (("V_evidence", "vevidence"), ("V_check", "vcheck"), ("V_select", "vselect"),
                              ("old_check", "oldcheck"), ("old_select", "oldselect"), ("old_replay", "oldreplay"))
           for field in ("path", "sha256", "content_sha256")},
        "replay_rule": REPLAY_RULE,
        "replay_stream": [SPLIT_SEED, REPLAY_STREAM],
        "replay_rows": [int(r) for r in replay_rows],
        "replay_source_idx": [int(i) for i in replay_pool["idx"][replay_rows]],
        "evidence_seed": seed,
        "evidence_rows_new": [int(r) for r in new_rows],
        "evidence_source_idx_new": [int(i) for i in v_ev["idx"][new_rows]],
        "evidence_rows_prev": [int(r) for r in prev_rows],
        "evidence_source_idx_prev": [int(i) for i in v_ev["idx"][prev_rows]],
        "evidence_split_ids_new": ids_new,
        "evidence_split_ids_old": ids_old,
        "evidence_split_ids_prev": ids_prev,
        "evidence_content_sha256": _evidence_content_sha256(arrays),
        "n_evidence_new": int(len(new_rows)),
        "n_evidence_old": N_REPLAY,
        "n_evidence_prev": int(len(prev_rows)),
        "n_base_correct_evidence_new": int(np.count_nonzero(pred_new == y_new)),
        "n_base_errors_evidence_old": int(np.count_nonzero(pred_old != y_old)),
        "n_base_errors_evidence_prev": int(np.count_nonzero(pred_prev != y_prev)),
        "n_vcheck_new": int(len(parts["y_new"])),
        "n_vcheck_old": int(len(parts["y_old"])),
        "n_base_errors_vcheck_new": int(np.count_nonzero(pred_new_ch != parts["y_new"])),
        "n_base_errors_vcheck_old": int(np.count_nonzero(pred_old_ch != parts["y_old"])),
        "n_vselect_new": int(np.count_nonzero(np.isin(v_sel["y"], session_classes(t)))),
        "n_vselect_prev": int(np.count_nonzero(np.isin(v_sel["y"], prev_classes(t)))),
        "n_oldselect": int(len(o_sel["y"])),
        "task_sha256": _text_sha256(task_text),
    }
    return arrays, task, spec_text, private


_STABLE_KEYS = ("evidence_content_sha256", "task_sha256", "model_sha256", "forward_sha256", "model_spec_sha256",
                "vcheck_content_sha256", "vselect_content_sha256", "oldcheck_content_sha256",
                "oldselect_content_sha256", "replay_rows", "evidence_rows_new", "evidence_rows_prev", "base_model_id")


def _same_build(instance_dir: Path, task: Mapping[str, Any], core: Mapping[str, Any]) -> None:
    private = read_json(instance_dir / "private.json")
    changed = [key for key in _STABLE_KEYS if private.get(key) != core.get(key)]
    if changed or json.loads((instance_dir / "task.json").read_text(encoding="utf-8")) != json.loads(json_text(task)):
        raise FscilError(f"{instance_dir}: differs from a rebuild in {changed or ['task.json']}; nothing overwritten")


def _create(root: Path, arrays: Mapping[str, np.ndarray], task: Mapping[str, Any], spec_text: str,
            core: Mapping[str, Any], params_path: Path, forward_path: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    instance_id = _new_instance_id(root)
    staging = root / f".tmp-{instance_id}"
    staging.mkdir()
    shutil.copyfile(params_path, staging / "model.npz")
    if params_sha256(load_params(staging / "model.npz")) != core["model_sha256"]:
        raise FscilError(f"{staging}/model.npz: copy hash mismatch")
    (staging / "model_spec.json").write_text(spec_text, encoding="utf-8")
    shutil.copyfile(forward_path, staging / "forward.py")
    if file_sha256(staging / "forward.py") != core["forward_sha256"]:
        raise FscilError(f"{staging}/forward.py: copy hash mismatch")
    if _save_evidence(staging / "evidence.npz", arrays) != core["evidence_content_sha256"]:
        raise FscilError(f"{staging}/evidence.npz: content hash mismatch")
    (staging / "task.json").write_text(json_text(task), encoding="utf-8")
    private = dict(core, instance_id=instance_id, created_at=utc_now(), numpy_version=np.__version__,
                   created_by=dict(provenance))
    (staging / "private.json").write_text(json_text(private), encoding="utf-8")
    with no_bytecode_cache():
        load_instance(staging)
    os.rename(staging, root / instance_id)
    return jsonable(private)


def build_fscil_instance(
    session: int,
    *,
    lineage: str,
    parent_params: str | Path | None = None,
    parent_instance: str | None = None,
    run_id: str | None = None,
    base_seed: int = DEV_SEED,
    batch_size: int = EVALUATION_BATCH,
) -> dict[str, Any]:
    """Build (or reuse) the instance of ``session`` on a parent student; returns its index row with ``status``.

    Session 2 takes no parent arguments: the parent is the fscil_base student of ``base_seed``. Session t >= 3 takes
    ``parent_params`` (a model.npz: the lineage's output at t - 1) and ``parent_instance`` (the active session t - 1
    instance that output was made on); the parameters must have the base's names, shapes and dtypes and be finite.
    An active instance of the same (session, parent parameters) is reused when a rebuild gives the same files, and
    refused otherwise. Every call appends a created or used line with the lineage, parent instance and run id.
    """
    t = _check_session(session, FIRST_SESSION)
    if not LINEAGE_RE.fullmatch(str(lineage)):
        raise FscilError(f"lineage {lineage!r} must match {LINEAGE_RE.pattern}")
    meta = read_dataset_meta(DATASET)
    names = class_names(meta)
    forward, forward_sha, forward_path = load_dataset_forward(DATASET, meta)
    base_row = base_student(base_seed)
    base_source, base_params, base_private, base_spec_text = model_record(base_row, BASE_ROLE)
    if base_private.get("train_classes") != list(range(N_BASE_CLASSES)):
        raise FscilError(f"{base_source}/private.json train_classes are not the base classes 0..{N_BASE_CLASSES - 1}")
    if t == FIRST_SESSION:
        if parent_params is not None or parent_instance is not None:
            raise FscilError("session 2 is built on the base student; it takes no parent")
        params, params_path, kind = base_params, base_source / "model.npz", "base"
    else:
        if parent_params is None or parent_instance is None:
            raise FscilError(f"session {t} needs parent_params and parent_instance (the session {t - 1} instance)")
        row = _active_row(str(parent_instance))
        if int(row["session"]) != t - 1 or row["base_model_id"] != base_row["model_id"]:
            raise FscilError(f"parent instance {parent_instance} is session {row['session']} of base "
                             f"{row['base_model_id']}, not session {t - 1} of {base_row['model_id']}")
        params_path = Path(parent_params)
        try:
            params = validate_child(base_params, load_params(params_path))
        except ValueError as error:
            raise FscilError(f"{params_path}: {error}") from error
        kind = "lineage"
    tokens = fscil_leak_tokens()
    _scan_visible(tokens, **{"forward.py": forward_path.read_text(encoding="utf-8")})
    data, split_record = _load_splits(meta)
    arrays, task, spec_text, core = _plan(t, params, kind, parent_instance, base_row, base_source, base_spec_text,
                                          names, data, split_record, forward, forward_sha, tokens, batch_size)
    provenance = {"lineage": str(lineage), "parent_instance": parent_instance, "run_id": run_id,
                  "parent_params_path": str(params_path)}
    root = fscil_root()
    with _exclusive_lock(root):
        index = read_fscil_instance_index()
        existing = [r for r in index["instances"]
                    if int(r["session"]) == t and r["parent_params_sha256"] == core["parent_params_sha256"]]
        if existing:
            (match,) = existing
            _same_build(root / match["instance_id"], task, core)
            _log_change(root, {"change": "used", **_index_row(match), **provenance, "at": utc_now()})
            return {**match, "status": "reused"}
        private = _create(root, arrays, task, spec_text, core, params_path, forward_path, provenance)
        index["instances"].append(_index_row(private))
        write_json_atomic(root / "index.json", index)
        _log_change(root, {"change": "created", **_index_row(private), **provenance, "at": utc_now()})
        return {**_index_row(private), "status": "created"}


# ---------------------------------------------------------------------------------------------------------------
# Running an instance: stale check, oracle, readouts


def check_fscil_current(instance_dir: str | Path) -> dict[str, Any]:
    """Refuse an instance that is not active or whose files the current code would not write; else return its task.

    Checks the directory name, the index, the rule, task.json against private.json and a fresh ``build_fscil_task``,
    and the hashes of model.npz, model_spec.json, forward.py and evidence.npz.
    """
    root = Path(instance_dir)
    private = read_json(root / "private.json")
    instance_id = str(private.get("instance_id"))
    if root.name != instance_id:
        raise StaleFscilInstance(f"{root}: directory name differs from private.json instance_id {instance_id!r}")
    _active_row(instance_id)
    if private.get("evidence_rule") != FSCIL_RULE:
        raise StaleFscilInstance(f"{instance_id} follows {private.get('evidence_rule')!r}, not {FSCIL_RULE}")
    task_text = (root / "task.json").read_text(encoding="utf-8")
    if _text_sha256(task_text) != private.get("task_sha256"):
        raise StaleFscilInstance(f"{root / 'task.json'}: bytes do not match private.json task_sha256")
    spec_text = (root / "model_spec.json").read_text(encoding="utf-8")
    if _text_sha256(spec_text) != private.get("model_spec_sha256"):
        raise StaleFscilInstance(f"{root / 'model_spec.json'}: hash differs from private.json")
    task = json.loads(task_text)
    expected = json.loads(json_text(build_fscil_task(task.get("classes", []), private["session"],
                                                     fscil_limits(json.loads(spec_text)))))
    if task != expected:
        changed = sorted(key for key in set(task) | set(expected) if task.get(key) != expected.get(key))
        raise StaleFscilInstance(f"{instance_id}: task.json differs from what the current code writes in {changed}")
    instance = load_instance(root)
    if _evidence_content_sha256(instance.evidence) != private.get("evidence_content_sha256"):
        raise StaleFscilInstance(f"{root / 'evidence.npz'}: content hash differs from private.json")
    return task


def _load_recorded(private: Mapping[str, Any], key: str) -> dict[str, np.ndarray]:
    path = Path(str(private[f"{key}_path"]))
    return load_checked_split(path, str(private[f"{key}_content_sha256"]))


def make_fscil_oracle(
    instance: Instance | str, budget: int = QUERY_BUDGET, batch_size: int = EVALUATION_BATCH
) -> JointPatchCheckOracle:
    """Joint counted oracle of an fscil instance: its check parts under the session mask, against its student."""
    resolved = load_fscil_instance(instance)
    private = resolved.private
    parts = fscil_check_parts(_load_recorded(private, "vcheck"), _load_recorded(private, "oldcheck"),
                              int(private["session"]))
    if (len(parts["y_new"]), len(parts["y_old"])) != (private["n_vcheck_new"], private["n_vcheck_old"]):
        raise FscilError(f"{resolved.instance_id}: check parts differ in size from the instance record")
    return JointPatchCheckOracle(parts["x_new"], parts["y_new"], parts["x_old"], parts["y_old"],
                                 masked_forward(resolved.forward, int(private["session"])), resolved.params, budget,
                                 batch_size)


def readout_rows(
    v_select: Mapping[str, np.ndarray], old_select: Mapping[str, np.ndarray], session: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, part) of a session's readout: old_select (part "old"), then the V_select rows of the classes of sessions
    2 .. t - 1 ("prev") and of session t ("new"). Session 1 reads the base classes only (pure)."""
    t = _check_session(session)
    y_sel = np.asarray(v_select["y"]).astype(np.int64)
    y_old = np.asarray(old_select["y"]).astype(np.int64)
    if np.any(y_old >= N_BASE_CLASSES):
        raise FscilError("old_select holds a row of an incremental class")
    prev = np.isin(y_sel, prev_classes(t))
    new = np.isin(y_sel, session_classes(t)) if t >= FIRST_SESSION else np.zeros_like(prev)
    x = np.concatenate([old_select["x"], v_select["x"][prev], v_select["x"][new]])
    y = np.concatenate([y_old, y_sel[prev], y_sel[new]])
    part = np.array(["old"] * len(y_old) + ["prev"] * int(prev.sum()) + ["new"] * int(new.sum()))
    return np.ascontiguousarray(x), y, part


def _acc(correct: np.ndarray) -> float | None:
    return float(np.mean(correct)) if correct.size else None


def _balanced(y: np.ndarray, correct: np.ndarray) -> float | None:
    if not y.size:
        return None
    return float(np.mean([np.mean(correct[y == label]) for label in np.unique(y)]))


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def _part_masks(part: np.ndarray) -> dict[str, np.ndarray]:
    masks = {name: part == name for name in ("new", "prev", "old")}
    masks["seen"] = np.ones(part.shape, dtype=bool)
    return masks


def accuracy_summary(pred: np.ndarray, y: np.ndarray, part: np.ndarray, session: int) -> dict[str, Any]:
    """Absolute readout of one model's masked predictions on ``readout_rows`` (pure)."""
    t = _check_session(session)
    right = np.asarray(pred) == y
    result: dict[str, Any] = {"session": t, "n_seen_classes": n_seen(t)}
    for name, mask in _part_masks(part).items():
        result[f"n_{name}"] = int(mask.sum())
        result[f"acc_{name}"] = _acc(right[mask])
        result[f"bal_{name}"] = _balanced(y[mask], right[mask])
    result["per_class"] = {str(c): {"n": int((y == c).sum()), "acc": _acc(right[y == c])} for c in seen_classes(t)}
    result["per_session"] = {
        str(s): {"n": int(np.isin(y, session_classes(s)).sum()),
                 "bal": _balanced(y[np.isin(y, session_classes(s))], right[np.isin(y, session_classes(s))])}
        for s in range(1, t + 1)
    }
    result["pred_in_new"] = int(np.count_nonzero(np.isin(pred, session_classes(t)))) if t >= FIRST_SESSION else 0
    return result


def readout_summary(base: np.ndarray, child: np.ndarray, y: np.ndarray, part: np.ndarray, session: int) -> dict[str, Any]:
    """Child against base on ``readout_rows`` (pure). "_base" is the instance's current student."""
    t = _check_session(session)
    base_right, child_right = np.asarray(base) == y, np.asarray(child) == y
    result: dict[str, Any] = {"session": t, "n_seen_classes": n_seen(t)}
    for name, mask in _part_masks(part).items():
        result[f"n_{name}"] = int(mask.sum())
        for model, right in (("base", base_right), ("child", child_right)):
            result[f"acc_{name}_{model}"] = _acc(right[mask])
            result[f"bal_{name}_{model}"] = _balanced(y[mask], right[mask])
        result[f"delta_{name}"] = _delta(result[f"acc_{name}_base"], result[f"acc_{name}_child"])
        result[f"delta_bal_{name}"] = _delta(result[f"bal_{name}_base"], result[f"bal_{name}_child"])
        result[f"fixed_{name}"] = int(np.count_nonzero(mask & ~base_right & child_right))
        result[f"broken_{name}"] = int(np.count_nonzero(mask & base_right & ~child_right))
    per_class = {}
    for c in seen_classes(t):
        rows = y == c
        per_class[str(c)] = {"n": int(rows.sum()), "acc_base": _acc(base_right[rows]),
                             "acc_child": _acc(child_right[rows])}
        per_class[str(c)]["delta"] = _delta(per_class[str(c)]["acc_base"], per_class[str(c)]["acc_child"])
    result["per_class"] = per_class
    result["per_session"] = {}
    for s in range(1, t + 1):
        rows = np.isin(y, session_classes(s))
        result["per_session"][str(s)] = {"n": int(rows.sum()), "bal_base": _balanced(y[rows], base_right[rows]),
                                         "bal_child": _balanced(y[rows], child_right[rows])}
    result["child_pred_in_new"] = int(np.count_nonzero(np.isin(child, session_classes(t))))
    return result


def fscil_accuracy(
    params: Params, session: int, *, forward: ForwardFn | None = None, batch_size: int = EVALUATION_BATCH
) -> dict[str, Any]:
    """Absolute masked accuracies of one model at a session on the store's V_select and old_select (dev readout).

    ``forward`` defaults to the dataset forward checked against dataset.json. Used for the base, the all-class
    reference, imprint_plain and the lineage summaries (session 1 reads the base classes only).
    """
    meta = read_dataset_meta(DATASET)
    if forward is None:
        forward, _, _ = load_dataset_forward(DATASET, meta)
    v_sel = load_checked_split(split_path(DATASET, "V_select"), meta["splits"]["V_select"]["sha256"])
    o_sel = load_checked_split(split_path(DATASET, "old_select"), meta["splits"]["old_select"]["sha256"])
    x, y, part = readout_rows(v_sel, o_sel, session)
    pred = predict(masked_forward(forward, session), params, x, batch_size)
    return accuracy_summary(pred, y, part, session)


def fscil_readout(instance: Instance | str, child: Params, batch_size: int = EVALUATION_BATCH) -> dict[str, Any]:
    """Masked readout of ``child`` against the instance's student at the instance's session (PREREG E9 readouts)."""
    resolved = load_fscil_instance(instance)
    private = resolved.private
    t = int(private["session"])
    x, y, part = readout_rows(_load_recorded(private, "vselect"), _load_recorded(private, "oldselect"), t)
    masked = masked_forward(resolved.forward, t)
    base = predict(masked, resolved.params, x, batch_size)
    pred = predict(masked, validate_child(resolved.params, child), x, batch_size)
    return readout_summary(base, pred, y, part, t)


ReadoutFn = Callable[[Instance | str, Params], dict[str, Any]]

__all__ = [
    "ALLCLASS_ROLE", "ARCH", "BASE_ROLE", "CB_PARTS", "CHECK_SCORE_RULE", "DATASET", "DATASET_WORDS", "DEV_SEED", "EDITOR_VISIBLE_FILES",
    "EVALUATION_BATCH", "EVIDENCE_KEYS", "FIRST_SESSION", "FSCIL_MAX_CHANGED_FRAC", "FSCIL_RULE", "FscilError",
    "HEAD_OP", "HEAD_TENSOR", "INDEX_SCHEMA", "INSTANCE_ID_RE", "KEPT_RULE_V2", "LAST_SESSION", "N_BASE_CLASSES", "N_CLASSES",
    "N_REPLAY", "N_SHOTS", "N_WAYS", "PRIVATE_SCHEMA", "READOUT_PARTS", "REPLAY_RULE", "REPLAY_STREAM", "ROOT_NAME",
    "SESSIONS", "StaleFscilInstance", "TASK_SCHEMA", "V2_BASE_ROWS", "V2_TIE_POINTS", "accuracy_summary", "active_fscil_ids", "base_student",
    "build_fscil_instance", "build_fscil_task", "cb_check_rows", "cb_check_score", "check_fscil_current", "draw_replay_rows", "fscil_accuracy",
    "fscil_check_parts", "fscil_instance_dir", "fscil_leak_tokens", "fscil_limits", "fscil_program_limits",
    "fscil_readout", "fscil_root", "fscil_scan", "fscil_seed", "head_tensors", "instance_chain", "joint_retain_floor",
    "load_fscil_instance", "make_fscil_oracle", "masked_forward", "model_record", "n_seen", "old_part_sizes",
    "prev_classes", "read_fscil_instance_index", "readout_rows", "readout_summary", "retain_v2_floor", "retain_v2_pick", "seen_classes", "session_classes",
    "session_of_class", "shot_rows",
]
