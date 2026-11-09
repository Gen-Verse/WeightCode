"""Rule scanskill-v1 (PREREG E13): instances that ask an update to make a SCAN student run a known primitive, or a
known phrase, inside the old compositions.

A SCAN base student (``weightcode.models.scan_export``, role base: exact match >= 0.95 on the 1000 old V_select rows) was
trained on the official training file of its condition without the held-out commands (``weightcode.data.scan``):

- addprim_jump: every command without jump, and "jump" alone 1,467 times; the new part is the commands that compose
  jump;
- around_right: every command without the phrase "around right"; the new part is the commands that hold it.

There is one instance per (condition, student). The layout is the evidence-v1 layout, so
``weightcode.baselines.instance_io.load_instance`` loads it unchanged. The instances live under their own root with their own
index, apart from the E5 store::

    $WEIGHTCODE_PRIVATE/scan_instances/
        index.json                       {"schema", "instances": [active rows], "superseded": [rows]}
        changes.jsonl                    one line per created or superseded instance
        n<8 hex>/
            model.npz, model_spec.json   byte copies of the student's files
            forward.py                   byte copy of the dataset forward
            evidence.npz                 the LEARN experience as it is, keys ``evidence_keys(condition)``
            task.json                    schema weightcode.task.composeskill.v1: data facts only (the harness words the task)
            private.json                 schema weightcode.instance_private.scanskill.v1, never editor-visible

What the teacher gets is the rule level (PREREG E13): the student's own model_spec.json states every token's meaning
(SCAN's interpretation function as published) and the grammar, and the evidence holds no row of the new part:

- old part: rule seqskill-replay-v1 as E5 defines it, N_REPLAY rows drawn uniformly without replacement with
  ``default_rng([SPLIT_SEED, REPLAY_STREAM])``, here from the student's training rows of old commands (part 2; for
  addprim_jump the "jump" rows are not old rows);
- primitive part (addprim_jump only): the "jump" row once, as it appears in training (part 3);
- y_*: the class at each OUT position (the six actions, END after the last) and IGNORE (-1) elsewhere; pred_*: the
  student's numpy-forward argmax at each OUT position and IGNORE elsewhere (``seq_predictions``);
- split ids: one ``e`` + 12 hex id per row from ``default_rng(scan_seed(params_sha256, condition))``, primitive
  first.

The student must leave headroom: exact match at most HEADROOM_EM_MAX on the new V_select rows, or no instance is
built (PREREG E13). Check (``make_scan_oracle``): S_new = fixed - broken by exact match on the condition's new
V_check, S_old the same on its 1000 old V_check rows (old_check), S = S_new + S_old (``JointPatchCheckOracle``).
Readout (``scan_readout``): exact match on new V_select (with token accuracy and the breakdown by construction) and on
old V_select, child against base. Limits (``scan_program_limits``): E5's program limits (at most 20% of the entries
changed, token embedding rows counted with tok_emb.weight) with the per-tensor rel-Frob limit lifted, as in E8's
amendment of 2026-11-04 15:37.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import Params, load_params, params_sha256
from weightcode.common.paths import private_dir
from weightcode.data import scan as S
from weightcode.data.registry import SPLIT_SEED
from weightcode.data.store import dataset_json_path, forward_path, models_dir, read_dataset_json, split_path
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
from weightcode.evalkit.metrics import IGNORE, rows_right, token_accuracy
from weightcode.evalkit.patch_check import ForwardFn, JointPatchCheckOracle, load_forward, predict
from weightcode.evalkit.seqskill import REPLAY_RULE, REPLAY_STREAM, seqskill_program_limits
from weightcode.evalkit.targets import file_sha256, json_text, jsonable, no_bytecode_cache, read_json, utc_now, write_json_atomic
from weightcode.patch import OP_NAMES, Limits
from weightcode.patch import SCHEMA as PATCH_SCHEMA

SCAN_RULE = "scanskill-v1"
N_REPLAY = 64
HEADROOM_EM_MAX = 0.50
"""PREREG E13 headroom: a base's exact match on the new V_select rows is at most this, or the condition is not run."""
GATE_EM = 0.95
"""PREREG E13 gate (``weightcode.models.train_scan.GATE_EM``): a base's exact match on the 1000 old V_select rows."""
TASK_SCHEMA = "weightcode.task.composeskill.v1"
"""Editor-visible, so it avoids the dataset name (the session forbids it)."""
PRIVATE_SCHEMA = "weightcode.instance_private.scanskill.v1"
INDEX_SCHEMA = "weightcode.scan_instances_index.v1"
CHANGES_LOG_NAME = "changes.jsonl"
ROOT_NAME = "scan_instances"
INSTANCE_ID_RE = re.compile(r"^n[0-9a-f]{8}$")
OLD_KEYS: tuple[str, ...] = ("x_old", "y_old", "pred_old", "split_ids_old")
PRIM_KEYS: tuple[str, ...] = ("x_prim", "y_prim", "pred_prim", "split_ids_prim")
EDITOR_VISIBLE_FILES: tuple[str, ...] = ("model.npz", "model_spec.json", "forward.py", "evidence.npz", "task.json")
EVALUATION_BATCH = 1024
SCAN_MAX_CHANGED_FRAC = 0.20
"""PREREG E13: at most 20% of the entries changed against the base, the token embedding rows counted with
tok_emb.weight (E5's cap, ``weightcode.evalkit.seqskill.SEQSKILL_MAX_CHANGED_FRAC``)."""
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

LEAK_WORDS: tuple[str, ...] = tuple(sorted({
    *S.CONDITIONS, "addprim", "template", "around_right",
    *(Path(name).name.lower() for files in S.CONDITION_FILES.values() for name in files.values()),
    *(Path(name).parent.name.lower() for files in S.CONDITION_FILES.values() for name in files.values()),
    "tasks_train", "tasks_test",
}))
"""PREREG E13 leak words: the condition names, "addprim", "template", "around_right" and the official file names (and
their directories). With the dataset's own leak tokens (``leak_tokens("scan")``: its name and "seed"), none may appear
in an editor-visible file."""

TARGET_COMMANDS: Mapping[str, str] = {
    "addprim_jump": "commands that hold jump together with other words: jump with a direction, opposite or around, "
    "repeated by twice or thrice, and joined to other commands by and or after",
    "around_right": "commands that hold u around right for a primitive u (walk, look, run or jump), alone, repeated by "
    "twice or thrice, and joined to other commands by and or after",
}
"""What the new part is, as the official test file of each condition defines it (checked on every new row at build)."""
TRAINING_FACTS: Mapping[str, str] = {
    "addprim_jump": "the model was trained on commands without jump and on the command jump alone; it was never trained "
    "on a command that holds jump with another word",
    "around_right": "the model was trained on commands without the words around right next to each other; it was "
    "trained on around with left and on right in every other construction, and never on a command that holds "
    "around right",
}


SCAN_LIMITS_SOURCE = ("scan_program_limits(task.json limits, n_tensors): PREREG E13, E5's program limits (at most 20% "
                      "of the entries changed) with the per-tensor rel-Frob limit lifted")


def scan_program_limits(task_limits_json: Mapping[str, Any], n_tensors: int | None = None) -> Limits:
    """The limits that bind an E13 update and every E13 reference run within limits: E5's program limits
    (``seqskill_program_limits``: every size limit but max_rel_frob and max_changed_frac off, at most 20% of the entries
    changed, max_ops at least the tensor count) with max_rel_frob lifted (None).

    They equal ``weightcode.harness.trial.lifted_limits(applied_limits(task_limits_json, "LEARN", n_tensors))``, the limits a
    LEARN session applies with ``--lift-rel-frob`` (a test checks it)."""
    limits = seqskill_program_limits(task_limits_json, n_tensors)
    if limits.max_changed_frac != SCAN_MAX_CHANGED_FRAC:  # pragma: no cover - both constants are 0.20
        raise ScanskillError(f"E5's changed-entry cap is {limits.max_changed_frac}, not {SCAN_MAX_CHANGED_FRAC}")
    return dataclasses.replace(limits, max_rel_frob=None)


class ScanskillError(RuntimeError):
    """A scanskill instance, its store or its inputs were refused; nothing is overwritten."""


class StaleScanskillInstance(ScanskillError):
    """An instance that is not active or whose files the current code would not write; it is not run."""


# ---------------------------------------------------------------------------------------------------------------
# Pure pieces


def evidence_keys(condition: str) -> tuple[str, ...]:
    """The evidence keys of a condition: the primitive part first when the condition has a primitive row."""
    if condition not in S.CONDITIONS:
        raise ScanskillError(f"unknown SCAN condition {condition!r}; one of {S.CONDITIONS}")
    return (*PRIM_KEYS, *OLD_KEYS) if S.PRIMITIVE_ROW[condition] else OLD_KEYS


def scan_leak_tokens() -> list[str]:
    """Lower-case tokens no editor-visible file may contain: the dataset's leak tokens and ``LEAK_WORDS``."""
    return sorted(set(leak_tokens(S.DATASET)) | set(LEAK_WORDS))


def seq_predictions(forward: ForwardFn, params: Params, x: np.ndarray, y: np.ndarray,
                    batch_size: int = EVALUATION_BATCH) -> np.ndarray:
    """The student's argmax at each labelled (OUT) position of ``x`` and IGNORE elsewhere, int64 [N, L]: the one
    definition of ``pred_*`` in a scanskill evidence file (E5's ``seq_predictions``); the harness re-checks it."""
    y = np.asarray(y)
    pred = predict(forward, params, np.ascontiguousarray(x), batch_size).astype(np.int64)
    if pred.shape != y.shape:
        raise ScanskillError(f"predictions {pred.shape} do not match targets {y.shape}")
    pred[y == IGNORE] = IGNORE
    return pred


def scan_seed(params_sha256_hex: str, condition: str) -> int:
    """Seed of the scanskill-v1 split ids of one student under one condition."""
    if not _SHA_RE.fullmatch(params_sha256_hex):
        raise ValueError(f"not a sha256 hex digest: {params_sha256_hex!r}")
    return _seed_from_text(f"{params_sha256_hex}|{S.DATASET}|{condition}|{SCAN_RULE}")


def old_train_rows(train: Mapping[str, np.ndarray]) -> np.ndarray:
    """Row indices of the training rows of old commands (part 2), in file order."""
    part = np.asarray(train["part"])
    if not np.isin(part, [S.PART_OLD, S.PART_PRIM]).all():
        raise ScanskillError("a training split holds a part other than old commands and the primitive row")
    return np.flatnonzero(part == S.PART_OLD).astype(np.int64)


def draw_replay_rows(train: Mapping[str, np.ndarray]) -> np.ndarray:
    """Rule seqskill-replay-v1 on the student's training rows of old commands: N_REPLAY row indices of train.npz, drawn
    uniformly without replacement from the part-2 rows with ``default_rng([SPLIT_SEED, REPLAY_STREAM])``."""
    pool = old_train_rows(train)
    if len(pool) < N_REPLAY:
        raise ScanskillError(f"the training rows hold {len(pool)} old rows; the replay needs {N_REPLAY}")
    rng = np.random.default_rng([SPLIT_SEED, REPLAY_STREAM])
    return pool[rng.choice(len(pool), size=N_REPLAY, replace=False)].astype(np.int64)


def primitive_train_row(train: Mapping[str, np.ndarray], condition: str) -> int | None:
    """Index of the first primitive row (part 3) of train.npz, or None for a condition without one; every part-3 row
    must be the same row, the primitive alone."""
    word = S.PRIMITIVE_ROW[condition]
    rows = np.flatnonzero(np.asarray(train["part"]) == S.PART_PRIM)
    if word is None:
        if rows.size:
            raise ScanskillError(f"{condition} has no primitive row, yet its training rows hold {rows.size}")
        return None
    if rows.size == 0:
        raise ScanskillError(f"{condition}: the training rows hold no {word!r} row")
    x = np.asarray(train["x"])
    if not (x[rows] == x[rows[0]]).all() or S.decode(x[rows[:1]]) != [word]:
        raise ScanskillError(f"{condition}: the part-3 training rows are not all the command {word!r}")
    return int(rows[0])


def split_ids(seed: int, n: int) -> list[str]:
    """``n`` unique opaque ids (``e`` + 12 hex) from ``default_rng(seed)``."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16**SPLIT_ID_HEX_DIGITS, size=n, dtype=np.int64)
    if np.unique(codes).size != codes.size:
        raise ScanskillError(f"split id collision for seed {seed}; the id width must grow (rule change)")
    return [f"e{int(code):0{SPLIT_ID_HEX_DIGITS}x}" for code in codes]


def check_new_commands(condition: str, commands: Sequence[str]) -> None:
    """Every new command holds the condition's primitive words next to each other, and, for around_right, after a
    primitive u (``TARGET_COMMANDS``); raises otherwise."""
    for command in commands:
        words = command.split()
        if not S.has_primitive(words, condition):
            raise ScanskillError(f"{condition}: new command {command!r} lacks {S.PRIMITIVE[condition]}")
        if condition == "addprim_jump" and len(words) < 2:
            raise ScanskillError(f"{condition}: new command {command!r} is jump alone")
        if condition == "around_right":
            for k in range(len(words) - 1):
                if words[k:k + 2] == ["around", "right"] and (k == 0 or words[k - 1] not in S.VERBS):
                    raise ScanskillError(f"{condition}: {command!r} holds around right after {words[k - 1:k]}")


def check_old_commands(condition: str, commands: Sequence[str]) -> None:
    """No old command holds the condition's primitive words next to each other (for addprim_jump: no jump)."""
    for command in commands:
        if S.has_primitive(command, condition):
            raise ScanskillError(f"{condition}: old command {command!r} holds {S.PRIMITIVE[condition]}")


def build_scan_task(condition: str, limits: Limits) -> dict[str, Any]:
    """Editor-visible data facts of a scanskill-v1 instance (schema ``TASK_SCHEMA``); the harness words the task."""
    evidence_keys(condition)
    stated = patch_limits(limits)
    evidence: dict[str, Any] = {}
    if S.PRIMITIVE_ROW[condition]:
        word = str(S.PRIMITIVE_ROW[condition])
        evidence["prim"] = {
            "keys": ["x_prim", "y_prim", "pred_prim"],
            "count": 1,
            "description": f"the command {word} alone, the one training command that holds {word}, as it appears in "
            "training, with its answer",
        }
    evidence["old"] = {
        "keys": ["x_old", "y_old", "pred_old"],
        "count": N_REPLAY,
        "description": "rows of training commands, a fixed random draw from the model's training rows, with their "
        "answers",
    }
    prim = S.PRIMITIVE_ROW[condition] is not None
    evidence["predictions"] = (("pred_prim and pred_old hold " if prim else "pred_old holds ")
                               + "the model's argmax at each OUT position and -1 elsewhere")
    evidence["ids"] = {"keys": ["split_ids_prim", "split_ids_old"] if prim else ["split_ids_old"],
                       "description": "one opaque id per example"}
    return {
        "schema": TASK_SCHEMA,
        "classes": list(S.CLASS_NAMES),
        "input": "token rows int64 [N, seq_len]; model_spec.json input.tokens gives every token's meaning (the "
        "interpretation rules) and input.layout the row format and the grammar",
        "target": {"commands": TARGET_COMMANDS[condition], "training": TRAINING_FACTS[condition]},
        "rows": {
            "layout": S.INPUT_LAYOUT,
            "targets": f"y holds the class index (0 to {S.N_CLASSES - 1}: the six actions, then END) at each OUT "
            "position and -1 elsewhere; a row is right when every OUT position is right, END included",
        },
        "evidence": evidence,
        "patch_check": {
            "budget_queries": QUERY_BUDGET,
            "parts": {
                "new": f"held-out {TARGET_COMMANDS[condition]}",
                "old": "held-out commands of the training distribution, never in the model's training rows",
            },
            "readout": "each query applies one patch and returns aggregate numbers on a held-out check split, each "
            "against the unedited model, for the two parts: exact-match accuracy, fixed (rows the unedited model got "
            "wrong that the patched one gets right), broken (rows the unedited model got right that the patched one "
            "gets wrong), the changed-parameter count, S_new and S_old (fixed - broken on each part) and S = S_new + "
            "S_old; no per-row information",
            "invalid_patches": "a patch that fails validation consumes no query",
            "final_selection": _FINAL_SELECTION,
        },
        "patch_format": {"schema": PATCH_SCHEMA, "allowed_ops": list(OP_NAMES)},
        "limits": stated,
        "limit_definitions": limit_definitions(stated),
        "training_free_rule": TRAINING_FREE_RULE,
    }


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence_content_sha256(arrays: Mapping[str, np.ndarray], condition: str) -> str:
    if tuple(sorted(arrays)) != tuple(sorted(evidence_keys(condition))):
        raise ScanskillError(f"evidence keys {sorted(arrays)} != {sorted(evidence_keys(condition))}")
    return params_sha256(arrays)


def _save_evidence(path: Path, arrays: Mapping[str, np.ndarray], condition: str) -> str:
    expected = _evidence_content_sha256(arrays, condition)
    np.savez(path, **{key: np.ascontiguousarray(arrays[key]) for key in evidence_keys(condition)})
    with np.load(path, allow_pickle=False) as archive:
        reloaded = {key: archive[key] for key in archive.files}
    if _evidence_content_sha256(reloaded, condition) != expected:
        raise ScanskillError(f"{path}: evidence content hash changed on reload")
    return expected


# ---------------------------------------------------------------------------------------------------------------
# Store: index.json, changes.jsonl, lock


def scan_instances_root() -> Path:
    """``$WEIGHTCODE_PRIVATE/scan_instances`` (created if missing)."""
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


def read_scan_instance_index() -> dict[str, Any]:
    """The scanskill instance index; an empty one before the first build."""
    path = private_dir() / ROOT_NAME / "index.json"
    if not path.exists():
        return {"schema": INDEX_SCHEMA, "instances": [], "superseded": []}
    index = read_json(path)
    if not isinstance(index, dict) or index.get("schema") != INDEX_SCHEMA:
        raise ScanskillError(f"{path} is not a {INDEX_SCHEMA} index")
    return index


def active_scan_ids(condition: str | None = None) -> list[str]:
    """Ids of the active scanskill instances, optionally of one condition, in index order."""
    rows = read_scan_instance_index()["instances"]
    return [str(row["instance_id"]) for row in rows if condition is None or row["condition"] == condition]


def scan_instance_dir(instance_id: str) -> Path:
    """Directory of one scanskill instance (computed without creating the root); refuses an id of another format."""
    if not INSTANCE_ID_RE.fullmatch(str(instance_id)):
        raise ScanskillError(f"{instance_id!r} is not a scanskill instance id (n + 8 hex)")
    return private_dir() / ROOT_NAME / str(instance_id)


def load_scan_instance(instance: Instance | str) -> Instance:
    """An Instance, or the scanskill instance with this id loaded and hash-checked (``load_instance``)."""
    return instance if isinstance(instance, Instance) else load_instance(scan_instance_dir(instance))


_INDEX_KEYS = ("instance_id", "dataset", "condition", "arch", "seed", "model_id", "split_role", "primitive",
               "evidence_rule", "created_at")


def _index_row(private: Mapping[str, Any]) -> dict[str, Any]:
    return {key: private[key] for key in _INDEX_KEYS}


def _new_instance_id(root: Path) -> str:
    while True:
        instance_id = "n" + secrets.token_hex(4)
        if not (root / instance_id).exists() and not (root / f".tmp-{instance_id}").exists():
            return instance_id


# ---------------------------------------------------------------------------------------------------------------
# Build

SPLITS: tuple[str, ...] = ("train", "old_check", "old_select", "V_check", "V_select")
SPLIT_KEYS: Mapping[str, str] = {"train": "train", "old_check": "oldcheck", "old_select": "oldselect",
                                 "V_check": "vcheck", "V_select": "vselect"}
"""private.json key prefix of each split's path, file sha256 and content sha256."""


def _load(path: Path, expected: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    if params_sha256(arrays) != expected:
        raise ScanskillError(f"{path}: content hash differs from its record")
    return arrays


def _load_inputs(condition: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, str]]]:
    meta = read_dataset_json(S.DATASET, condition)
    data, record = {}, {}
    for split in SPLITS:
        path = split_path(S.DATASET, split, condition)
        expected = meta["splits"][split]["sha256"]
        data[split] = _load(path, expected)
        record[split] = {"path": str(path), "sha256": file_sha256(path), "content_sha256": expected}
    return data, record


def _scan_visible(tokens: Sequence[str], **texts: str) -> None:
    for where, text in texts.items():
        found = scan_for_tokens(text, tokens)
        if found:
            raise ScanskillError(f"{where} contains {found}; no editor-visible file may")


def _parts_of(data: Mapping[str, Mapping[str, np.ndarray]], condition: str) -> None:
    """Every split holds the parts it should and the commands the condition says (new: the target; old: none)."""
    expected = {"old_check": S.PART_OLD, "old_select": S.PART_OLD, "V_check": S.PART_NEW, "V_select": S.PART_NEW}
    for split, part in expected.items():
        found = sorted({int(p) for p in np.unique(data[split]["part"])})
        if found != [part]:
            raise ScanskillError(f"{condition}/{split} holds parts {found}, not [{part}]")
    for split in ("V_check", "V_select"):
        check_new_commands(condition, S.decode(data[split]["x"]))
    for split in ("old_check", "old_select"):
        check_old_commands(condition, S.decode(data[split]["x"]))
    train = data["train"]
    check_old_commands(condition, S.decode(train["x"][old_train_rows(train)]))


def _plan(
    condition: str,
    row: Mapping[str, Any],
    data: Mapping[str, Mapping[str, np.ndarray]],
    split_record: Mapping[str, Mapping[str, str]],
    forward: ForwardFn,
    forward_sha: str,
    tokens: Sequence[str],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Evidence arrays, task and private record (without id and time) of one student's instance."""
    source = models_dir(S.DATASET, condition) / str(row["model_id"])
    params = load_params(source / "model.npz")
    digest = params_sha256(params)
    if digest != row["params_sha256"]:
        raise ScanskillError(f"{source}/model.npz hash {digest} != the scan index {row['params_sha256']}")
    model_private = read_json(source / "private.json")
    if model_private.get("role") != "base" or row.get("role") != "base":
        raise ScanskillError(f"{source}: not a base student")
    gate_em = float(model_private["final_vselect_acc"])
    if gate_em < GATE_EM:
        raise ScanskillError(f"{source}: old V_select exact match {gate_em} is below the gate {GATE_EM}")
    spec_text = (source / "model_spec.json").read_text(encoding="utf-8")
    if json.loads(spec_text).get("classes") != list(S.CLASS_NAMES):
        raise ScanskillError(f"{source}/model_spec.json class list differs from SCAN's")

    em = {}
    for split in ("V_select", "old_select"):
        pred = predict(forward, params, data[split]["x"], batch_size)
        em[split] = float(np.mean(rows_right(data[split]["y"], pred)))
    if abs(em["old_select"] - gate_em) > 0.0:
        raise ScanskillError(f"{source}: old V_select exact match {em['old_select']} differs from the export {gate_em}")
    if em["V_select"] > HEADROOM_EM_MAX:
        raise ScanskillError(f"{row['model_id']} answers {em['V_select']:.4f} of the new V_select rows; a scanskill base "
                             f"needs at most {HEADROOM_EM_MAX} (PREREG E13 headroom)")

    train = data["train"]
    replay_rows = draw_replay_rows(train)
    prim_row = primitive_train_row(train, condition)
    n_prim = 0 if prim_row is None else 1
    seed = scan_seed(digest, condition)
    ids = split_ids(seed, n_prim + N_REPLAY)
    arrays: dict[str, np.ndarray] = {}
    if prim_row is not None:
        x_prim = np.ascontiguousarray(train["x"][[prim_row]])
        y_prim = np.ascontiguousarray(train["y"][[prim_row]]).astype(np.int64)
        arrays.update(x_prim=x_prim, y_prim=y_prim, pred_prim=seq_predictions(forward, params, x_prim, y_prim),
                      split_ids_prim=np.array(ids[:n_prim]))
    x_old = np.ascontiguousarray(train["x"][replay_rows])
    y_old = np.ascontiguousarray(train["y"][replay_rows]).astype(np.int64)
    arrays.update(x_old=x_old, y_old=y_old, pred_old=seq_predictions(forward, params, x_old, y_old),
                  split_ids_old=np.array(ids[n_prim:]))
    arrays = {key: arrays[key] for key in evidence_keys(condition)}
    task = build_scan_task(condition, Limits())
    task_text = json_text(task)
    _scan_visible(sorted(set(tokens) | {str(row["model_id"]).lower()}),
                  **{"task.json": task_text, "evidence.npz": npz_text(arrays)})
    _scan_visible(tokens, **{"model_spec.json": spec_text, "model.npz names": npz_text(params)})

    base_errors = {}
    for split in ("V_check", "old_check"):
        pred = predict(forward, params, data[split]["x"], batch_size)
        base_errors[split] = int(np.count_nonzero(~rows_right(data[split]["y"], pred)))
    private: dict[str, Any] = {
        "schema": PRIVATE_SCHEMA,
        "evidence_rule": SCAN_RULE,
        "dataset": S.DATASET,
        "condition": condition,
        "primitive": list(S.PRIMITIVE[condition]),
        "primitive_row": S.PRIMITIVE_ROW[condition],
        "primitive_token_id": (None if S.PRIMITIVE_ROW[condition] is None
                               else int(S.TOKEN_ID[str(S.PRIMITIVE_ROW[condition])])),
        "arch": row["arch"],
        "seed": int(row["seed"]),
        "model_id": row["model_id"],
        "role": "base",
        "split_role": "dev",
        "model_sha256": digest,
        "model_source_path": str(source / "model.npz"),
        "model_spec_sha256": file_sha256(source / "model_spec.json"),
        "forward_sha256": forward_sha,
        "base_size_step": int(model_private["size_step"]),
        "base_gate_em": gate_em,
        "gate_em": GATE_EM,
        "base_vselect_em": {"new": em["V_select"], "old": em["old_select"]},
        "headroom_em_max": HEADROOM_EM_MAX,
        "condition_dataset_json_sha256": file_sha256(dataset_json_path(S.DATASET, condition)),
        "replay_rule": REPLAY_RULE,
        "replay_stream": [SPLIT_SEED, REPLAY_STREAM],
        "replay_pool": "train.npz rows of part 2 (old commands), in file order",
        "replay_train_rows": [int(r) for r in replay_rows],
        "replay_source_idx": [int(i) for i in train["idx"][replay_rows]],
        "prim_train_row": prim_row,
        "evidence_seed": seed,
        "evidence_split_ids_prim": ids[:n_prim],
        "evidence_split_ids_old": ids[n_prim:],
        "evidence_content_sha256": _evidence_content_sha256(arrays, condition),
        "n_prim": n_prim,
        "n_replay": N_REPLAY,
        "n_base_correct_evidence_prim": (None if prim_row is None
                                         else int(np.count_nonzero(rows_right(arrays["y_prim"], arrays["pred_prim"])))),
        "n_base_errors_evidence_old": int(np.count_nonzero(~rows_right(y_old, arrays["pred_old"]))),
        "n_vcheck": int(len(data["V_check"]["y"]) + len(data["old_check"]["y"])),
        "n_vcheck_new": int(len(data["V_check"]["y"])),
        "n_vcheck_old": int(len(data["old_check"]["y"])),
        "n_base_errors_vcheck_new": base_errors["V_check"],
        "n_base_errors_vcheck_old": base_errors["old_check"],
        "n_vselect_new": int(len(data["V_select"]["y"])),
        "n_vselect_old": int(len(data["old_select"]["y"])),
        "task_sha256": _text_sha256(task_text),
    }
    for split, key in SPLIT_KEYS.items():
        private.update({f"{key}_path": split_record[split]["path"], f"{key}_sha256": split_record[split]["sha256"],
                        f"{key}_content_sha256": split_record[split]["content_sha256"]})
    return arrays, task, private


def _create(root: Path, arrays: Mapping[str, np.ndarray], task: Mapping[str, Any], core: Mapping[str, Any],
            forward_file: Path) -> dict[str, Any]:
    instance_id = _new_instance_id(root)
    staging = root / f".tmp-{instance_id}"
    staging.mkdir()
    source = Path(core["model_source_path"])
    shutil.copyfile(source, staging / "model.npz")
    if params_sha256(load_params(staging / "model.npz")) != core["model_sha256"]:
        raise ScanskillError(f"{staging}/model.npz: copy hash mismatch")
    shutil.copyfile(source.with_name("model_spec.json"), staging / "model_spec.json")
    if file_sha256(staging / "model_spec.json") != core["model_spec_sha256"]:
        raise ScanskillError(f"{staging}/model_spec.json: copy hash mismatch")
    shutil.copyfile(forward_file, staging / "forward.py")
    if file_sha256(staging / "forward.py") != core["forward_sha256"]:
        raise ScanskillError(f"{staging}/forward.py: copy hash mismatch")
    if _save_evidence(staging / "evidence.npz", arrays, str(core["condition"])) != core["evidence_content_sha256"]:
        raise ScanskillError(f"{staging}/evidence.npz: content hash mismatch")
    (staging / "task.json").write_text(json_text(task), encoding="utf-8")
    private = dict(core, instance_id=instance_id, created_at=utc_now(), numpy_version=np.__version__)
    (staging / "private.json").write_text(json_text(private), encoding="utf-8")
    with no_bytecode_cache():
        load_instance(staging)
    os.rename(staging, root / instance_id)
    return jsonable(private)


_STABLE = ("evidence_content_sha256", "task_sha256", "model_sha256", "forward_sha256", "model_spec_sha256",
           "vcheck_content_sha256", "oldcheck_content_sha256", "vselect_content_sha256", "oldselect_content_sha256",
           "replay_train_rows", "prim_train_row", "condition_dataset_json_sha256")


def _same_build(instance_dir: Path, task: Mapping[str, Any], core: Mapping[str, Any]) -> None:
    private = read_json(instance_dir / "private.json")
    changed = [key for key in _STABLE if private.get(key) != core.get(key)]
    if changed or json.loads((instance_dir / "task.json").read_text(encoding="utf-8")) != json.loads(json_text(task)):
        raise ScanskillError(f"{instance_dir}: differs from a rebuild in {changed or ['task.json']}; nothing overwritten")


def build_scan_instances(condition: str, *, seeds: Sequence[int] | None = None,
                         batch_size: int = EVALUATION_BATCH) -> list[dict[str, Any]]:
    """Build (or reuse) one scanskill-v1 instance per base student of the condition; returns one row per student.

    Students are the ``base`` rows of the condition's model index (``weightcode.models.export.read_index``), optionally only
    some ``seeds``. An active instance of the same (condition, model id) is reused when a rebuild gives the same
    evidence, task and inputs, and refused otherwise. A student without headroom is refused (no instance).
    """
    from weightcode.models.export import read_index

    evidence_keys(condition)
    students = [row for row in read_index(S.DATASET, condition)
                if row["role"] == "base" and (seeds is None or int(row["seed"]) in {int(s) for s in seeds})]
    if not students:
        raise ScanskillError(f"no {condition} base students for seeds {seeds}")
    base_meta = read_dataset_json(S.DATASET)
    cond_meta = read_dataset_json(S.DATASET, condition)
    if cond_meta["forward_sha256"] != base_meta["forward_sha256"]:
        raise ScanskillError(f"{condition}: forward.py hash differs from the dataset's")
    forward_file = forward_path(S.DATASET)
    with no_bytecode_cache():
        forward, forward_sha = load_forward(forward_file)
    if forward_sha != base_meta["forward_sha256"]:
        raise ScanskillError(f"{forward_file}: sha256 {forward_sha} != dataset.json")
    tokens = scan_leak_tokens()
    _scan_visible(tokens, **{"forward.py": forward_file.read_text(encoding="utf-8")})
    data, split_record = _load_inputs(condition)
    _parts_of(data, condition)
    root = scan_instances_root()
    outcomes = []
    with _exclusive_lock(root):
        index = read_scan_instance_index()
        for row in sorted(students, key=lambda r: (r["arch"], int(r["seed"]))):
            arrays, task, core = _plan(condition, row, data, split_record, forward, forward_sha, tokens, batch_size)
            existing = [r for r in index["instances"] if r["condition"] == condition and r["model_id"] == row["model_id"]]
            if existing:
                (match,) = existing
                _same_build(root / match["instance_id"], task, core)
                outcomes.append({**match, "status": "reused"})
                continue
            private = _create(root, arrays, task, core, forward_file)
            index["instances"].append(_index_row(private))
            write_json_atomic(root / "index.json", index)
            with open(root / CHANGES_LOG_NAME, "a", encoding="utf-8") as log:
                log.write(json.dumps({"change": "created", **_index_row(private)}, sort_keys=True) + "\n")
            outcomes.append({**_index_row(private), "status": "created"})
    return outcomes


def supersede_scan_instance(instance_id: str, reason: str) -> dict[str, Any]:
    """Move an active scanskill instance to ``superseded`` in the index and return the moved row; its files stay."""
    if not isinstance(reason, str) or not reason.strip():
        raise ScanskillError("superseding an instance needs a reason")
    root = scan_instances_root()
    with _exclusive_lock(root):
        index = read_scan_instance_index()
        hits = [row for row in index["instances"] if row["instance_id"] == instance_id]
        if len(hits) != 1:
            raise ScanskillError(f"{instance_id} is not an active scanskill instance")
        moved = {**hits[0], "superseded_at": utc_now(), "reason": reason}
        index["instances"] = [row for row in index["instances"] if row["instance_id"] != instance_id]
        index["superseded"].append(moved)
        write_json_atomic(root / "index.json", index)
        with open(root / CHANGES_LOG_NAME, "a", encoding="utf-8") as log:
            log.write(json.dumps({"change": "superseded", **moved}, sort_keys=True) + "\n")
    return moved


# ---------------------------------------------------------------------------------------------------------------
# Running an instance: stale check, oracle, readout


def check_scan_current(instance_dir: str | Path) -> dict[str, Any]:
    """Refuse an instance that is not active or whose files the current code would not write; else return its task.

    Checks the directory name, the index, the rule, the task.json bytes against private.json, the task against a fresh
    ``build_scan_task`` and the hashes of model.npz, model_spec.json, forward.py and evidence.npz.
    """
    root = Path(instance_dir)
    private = read_json(root / "private.json")
    instance_id = str(private.get("instance_id"))
    if root.name != instance_id:
        raise StaleScanskillInstance(f"{root}: directory name differs from private.json instance_id {instance_id!r}")
    index = read_scan_instance_index()
    if instance_id not in {row["instance_id"] for row in index["instances"]}:
        retired = {row["instance_id"] for row in index["superseded"]}
        status = "superseded" if instance_id in retired else "not in the index"
        raise StaleScanskillInstance(f"scanskill instance {instance_id} is {status}; only active instances are run")
    if private.get("evidence_rule") != SCAN_RULE:
        raise StaleScanskillInstance(f"{instance_id} follows {private.get('evidence_rule')!r}, not {SCAN_RULE}")
    task_text = (root / "task.json").read_text(encoding="utf-8")
    if _text_sha256(task_text) != private.get("task_sha256"):
        raise StaleScanskillInstance(f"{root / 'task.json'}: bytes do not match private.json task_sha256")
    task = json.loads(task_text)
    expected = json.loads(json_text(build_scan_task(str(private["condition"]), Limits())))
    if task != expected:
        changed = sorted(key for key in set(task) | set(expected) if task.get(key) != expected.get(key))
        raise StaleScanskillInstance(f"{instance_id}: task.json differs from what the current code writes in {changed}")
    if file_sha256(root / "model_spec.json") != private.get("model_spec_sha256"):
        raise StaleScanskillInstance(f"{root / 'model_spec.json'}: hash differs from private.json")
    instance = load_instance(root)  # model.npz and forward.py hashes against private.json
    if _evidence_content_sha256(instance.evidence, str(private["condition"])) != private.get("evidence_content_sha256"):
        raise StaleScanskillInstance(f"{root / 'evidence.npz'}: content hash differs from private.json")
    return task


def load_recorded(private: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    """One of the instance's recorded splits (``SPLIT_KEYS``), refused unless its content hash is the recorded one."""
    if split not in SPLIT_KEYS:
        raise ScanskillError(f"unknown split {split!r}; one of {list(SPLIT_KEYS)}")
    key = SPLIT_KEYS[split]
    recorded = private.get(f"{key}_content_sha256")
    if recorded is None:
        raise ScanskillError(f"the instance record has no {key}_content_sha256")
    return _load(Path(str(private[f"{key}_path"])), str(recorded))


def check_parts(new: Mapping[str, np.ndarray], old: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The two check parts: new V_check (part 1 only) and old V_check (old_check, part 2 only), in row order."""
    for name, arrays, part in (("new V_check", new, S.PART_NEW), ("old V_check", old, S.PART_OLD)):
        found = sorted({int(p) for p in np.unique(arrays["part"])})
        if found != [part]:
            raise ScanskillError(f"{name} holds parts {found}, not [{part}]")
    return {"x_new": np.ascontiguousarray(new["x"]), "y_new": np.ascontiguousarray(new["y"]).astype(np.int64),
            "x_old": np.ascontiguousarray(old["x"]), "y_old": np.ascontiguousarray(old["y"]).astype(np.int64)}


def make_scan_oracle(instance: Instance | str, budget: int = QUERY_BUDGET,
                     batch_size: int = EVALUATION_BATCH) -> JointPatchCheckOracle:
    """Joint counted oracle of a scanskill instance: new V_check and old V_check (``check_parts``)."""
    resolved = load_scan_instance(instance)
    private = resolved.private
    parts = check_parts(load_recorded(private, "V_check"), load_recorded(private, "old_check"))
    if (len(parts["y_new"]), len(parts["y_old"])) != (private["n_vcheck_new"], private["n_vcheck_old"]):
        raise ScanskillError(f"{resolved.instance_id}: the check parts differ in size from the instance record")
    return JointPatchCheckOracle(parts["x_new"], parts["y_new"], parts["x_old"], parts["y_old"], resolved.forward,
                                 resolved.params, budget, batch_size)


def part_summary(y: np.ndarray, base: np.ndarray, child: np.ndarray) -> dict[str, Any]:
    """n, base and child exact match, delta, fixed, broken and token accuracies of one set of rows."""
    rb, rc = rows_right(y, base), rows_right(y, child)
    em_base, em_child = float(np.mean(rb)), float(np.mean(rc))
    return {"n": int(len(y)), "em_base": em_base, "em_child": em_child, "delta": em_child - em_base,
            "fixed": int(np.count_nonzero(~rb & rc)), "broken": int(np.count_nonzero(rb & ~rc)),
            "token_acc_base": token_accuracy(y, base), "token_acc_child": token_accuracy(y, child)}


def construction_masks(x: np.ndarray) -> dict[str, np.ndarray]:
    """For each construction of ``S.CONSTRUCTIONS`` the rows whose command holds any of its words, and ``none`` the
    rows that hold none of them (rows can be in several constructions)."""
    words = [set(command.split()) for command in S.decode(x)]
    masks = {name: np.array([bool(set(group) & held) for held in words], dtype=bool)
             for name, group in S.CONSTRUCTIONS.items()}
    masks["none"] = ~np.any(np.stack(list(masks.values())), axis=0)
    return masks


def scan_readout_arrays(new: Mapping[str, np.ndarray], old: Mapping[str, np.ndarray], forward: ForwardFn,
                        base_params: Params, child: Params, batch_size: int = EVALUATION_BATCH,
                        base_pred: Mapping[str, np.ndarray] | None = None,
                        fit_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """The PREREG E13 readout of ``child`` against ``base_params`` on new V_select (``new``) and old V_select
    (``old``): each part's ``part_summary``, the new part by construction, and points (100 x delta exact match)
    with joint = gain + clean change. ``base_pred`` (keys new, old) reuses the base predictions.

    ``fit_ids`` (the command ids a rule-generated reference fitted) adds ``fit_coverage``: how many new V_select
    commands the fit set holds, and the new part read on the commands it holds and on those it does not hold (a
    diagnostic beside the reading, not the bar; WEIGHTCODE-main, 2026-11-06)."""
    result: dict[str, Any] = {}
    bases: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    for name, arrays in (("new", new), ("old", old)):
        bases[name] = predict(forward, base_params, arrays["x"], batch_size) if base_pred is None else base_pred[name]
        preds[name] = predict(forward, child, arrays["x"], batch_size)
        result[name] = part_summary(np.asarray(arrays["y"]), bases[name], preds[name])
    y_new = np.asarray(new["y"])
    result["new_by_construction"] = {
        name: part_summary(y_new[rows], bases["new"][rows], preds["new"][rows]) if rows.any() else {"n": 0}
        for name, rows in construction_masks(np.asarray(new["x"])).items()}
    if fit_ids is not None:
        held = np.isin(np.asarray(new["idx"]), np.asarray(list(fit_ids), dtype=np.int64))
        result["fit_coverage"] = {
            "n_new": int(len(y_new)), "n_in_fit_set": int(held.sum()), "n_not_in_fit_set": int((~held).sum()),
            **{label: part_summary(y_new[rows], bases["new"][rows], preds["new"][rows]) if rows.any() else {"n": 0}
               for label, rows in (("in_fit_set", held), ("not_in_fit_set", ~held))}}
    result["gain_points"] = 100.0 * result["new"]["delta"]
    result["clean_points"] = 100.0 * result["old"]["delta"]
    result["joint_points"] = result["gain_points"] + result["clean_points"]
    return result


def scan_readout(instance: Instance | str, child: Params, batch_size: int = EVALUATION_BATCH,
                 fit_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """``scan_readout_arrays`` of ``child`` against the instance's base on the condition's new and old V_select."""
    resolved = load_scan_instance(instance)
    private = resolved.private
    return scan_readout_arrays(load_recorded(private, "V_select"), load_recorded(private, "old_select"),
                               resolved.forward, resolved.params, child, batch_size, fit_ids=fit_ids)


__all__ = [
    "EDITOR_VISIBLE_FILES", "GATE_EM", "HEADROOM_EM_MAX", "INSTANCE_ID_RE", "LEAK_WORDS", "N_REPLAY", "OLD_KEYS",
    "PRIM_KEYS", "PRIVATE_SCHEMA", "SCAN_LIMITS_SOURCE", "SCAN_MAX_CHANGED_FRAC", "SCAN_RULE", "SPLIT_KEYS",
    "TARGET_COMMANDS", "TASK_SCHEMA", "TRAINING_FACTS", "ScanskillError", "StaleScanskillInstance", "active_scan_ids",
    "build_scan_instances", "build_scan_task", "check_new_commands", "check_old_commands", "check_parts",
    "check_scan_current", "construction_masks", "draw_replay_rows", "evidence_keys", "load_recorded",
    "load_scan_instance", "make_scan_oracle", "old_train_rows", "part_summary", "primitive_train_row",
    "read_scan_instance_index", "scan_instance_dir", "scan_instances_root", "scan_leak_tokens", "scan_program_limits",
    "scan_readout", "scan_readout_arrays", "scan_seed", "seq_predictions", "split_ids", "supersede_scan_instance",
]


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m weightcode.evalkit.scanskill ids --condition C``: print the condition's one active instance id."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="python -m weightcode.evalkit.scanskill")
    sub = parser.add_subparsers(dest="command", required=True)
    ids = sub.add_parser("ids", help="print the one active scanskill instance id of a condition")
    ids.add_argument("--condition", required=True, choices=S.CONDITIONS)
    args = parser.parse_args(argv)
    found = active_scan_ids(args.condition)
    if len(found) != 1:
        print(f"{args.condition}: {len(found)} active scanskill instances, expected one", file=sys.stderr)
        return 1
    print(found[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
