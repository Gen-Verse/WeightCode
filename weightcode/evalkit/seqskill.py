"""Rule seqskill-v1 (PREREG E5): instances that ask an update to teach a sequence student a new operator.

A seqops base student (``weightcode.models.seq_export``) was trained on old-operator rows only: the reserved operator token
OP_NEW<k> never occurred in its training, so its embedding row is untrained. The condition new<k>_<kind>
(``weightcode.data.seqops_build``) binds the token to a rule drawn after training. There is one instance per (condition,
student). The layout is the evidence-v1 layout, so ``weightcode.baselines.instance_io.load_instance`` loads it unchanged.
The instances live under their own root with their own index::

    $WEIGHTCODE_PRIVATE/seqskill_instances/
        index.json                       {"schema", "instances": [active rows], "superseded": [rows]}
        changes.jsonl                    one line per created instance
        q<8 hex>/
            model.npz, model_spec.json   byte copies of the student's files
            forward.py                   byte copy of the dataset forward
            evidence.npz                 the evidence, keys EVIDENCE_KEYS; it is the LEARN experience as it is
            task.json                    schema weightcode.task.seqskill.v1: data facts only (the harness words the task)
            private.json                 schema weightcode.instance_private.seqskill.v1, never editor-visible

Evidence of rule seqskill-v1:

- new part: N_NEW rows drawn uniformly without replacement from the condition's V_evidence (part 1: the new operator
  alone, n 4 to 8) with ``default_rng(seqskill_seed(params_sha256, condition))``. Together they must show every
  symbol in their inputs, or no instance is built (the draw is not repeated);
- old part: rule seqskill-replay-v1, N_REPLAY rows drawn uniformly without replacement from the dataset's train
  sample with ``default_rng([SPLIT_SEED, REPLAY_STREAM])``, one draw shared by every student;
- y_*: the answer symbol at each OUT position and IGNORE (-1) elsewhere. pred_*: the student's numpy-forward argmax at
  each OUT position and IGNORE elsewhere (``seq_predictions``);
- split ids: N_NEW + N_REPLAY unique ``e`` + 12 hex ids from the instance generator after the row draw, new first.

The student must lack the new skill: its exact match on each new part of the condition's V_select (parts 1 to 3) is
below BASE_EM_MAX, and its reserved embedding rows are untouched, or no instance is built.

Check (``make_seqskill_oracle``): the condition's V_check cut by part (``seqskill_check_parts``). S_new is over part 1
(the new operator alone, n 4 to 8), S_old over part 4 (old operators) and S = S_new + S_old (``JointPatchCheckOracle``).
Parts 2 and 3 are never in a check. Readout (``seqskill_readout``): exact match on the condition's V_select per part,
child against base, with breakdowns by length and by operator pair.

Limits: task.json states the default ``weightcode.patch.Limits`` without exemptions. A LEARN session applies them through
``weightcode.harness.arms.applied_limits``: rel-Frob at most 0.5 on tensors of rank 2 or more and at most 20% of the entries
changed (PREREG E5).
"""

from __future__ import annotations

import dataclasses
import fcntl
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
from weightcode.data import seqops as S
from weightcode.data.registry import SPLIT_SEED
from weightcode.data.seqops_build import CLASS_NAMES, parse_condition
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
from weightcode.evalkit.targets import file_sha256, json_text, jsonable, no_bytecode_cache, read_json, utc_now, write_json_atomic
from weightcode.patch import OP_NAMES, Limits
from weightcode.patch import SCHEMA as PATCH_SCHEMA
from weightcode.patch.limits import SIZE_LIMIT_FIELDS

SEQSKILL_RULE = "seqskill-v1"
REPLAY_RULE = "seqskill-replay-v1"
REPLAY_STREAM = 1170
"""Generator stream of the seqskill replay, inside the seqops range 1100-1199 and clear of its other streams."""
N_NEW = 64
N_REPLAY = 64
BASE_EM_MAX = 0.01
"""A student is a valid base of a new-skill instance only if its exact match on every new part is below this."""
TASK_SCHEMA = "weightcode.task.seqskill.v1"
PRIVATE_SCHEMA = "weightcode.instance_private.seqskill.v1"
INDEX_SCHEMA = "weightcode.seqskill_instances_index.v1"
CHANGES_LOG_NAME = "changes.jsonl"
ROOT_NAME = "seqskill_instances"
INSTANCE_ID_RE = re.compile(r"^q[0-9a-f]{8}$")
EVIDENCE_KEYS: tuple[str, ...] = (
    "x_new", "y_new", "pred_new", "split_ids_new", "x_old", "y_old", "pred_old", "split_ids_old",
)
EDITOR_VISIBLE_FILES: tuple[str, ...] = ("model.npz", "model_spec.json", "forward.py", "evidence.npz", "task.json")
CHECK_PART_KEYS: tuple[str, ...] = ("x_new", "y_new", "x_old", "y_old")
CHECK_PARTS: tuple[int, int] = (S.PART_NEW_SHORT, S.PART_OLD)
READOUT_PARTS: Mapping[str, int] = {"i": S.PART_NEW_SHORT, "ii": S.PART_NEW_LONG, "iii": S.PART_NEW_COMPOSED,
                                    "iv": S.PART_OLD}
NEW_PARTS: tuple[int, ...] = (S.PART_NEW_SHORT, S.PART_NEW_LONG, S.PART_NEW_COMPOSED)
MAP_KINDS: Mapping[str, tuple[int, ...]] = {
    "position": tuple(S.TOKEN_ID[f"OP_{name}"] for name in S.POSITION_OPS),
    "symbol": tuple(S.TOKEN_ID[f"OP_{name}"] for name in S.SYMBOL_OPS),
}
"""Old operators by what they map: positions (REV, ROTL1, ROTR1, SWAP2) or symbols (INC, DEC, COMP, SUBA)."""
CONDITION_LEAK_WORDS: tuple[str, ...] = ("substitution", "permutation")
"""Words that would tell an editor what kind of rule the new operator is; with the condition name and the dataset
tokens, none may appear in an editor-visible file."""
EVALUATION_BATCH = 1024
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


SEQSKILL_MAX_CHANGED_FRAC = 0.20
"""PREREG E5: an update may change at most 20% of the parameter entries, the changed-entry cap of every program arm
(``weightcode.harness.arms.PROGRAM_MAX_CHANGED_FRAC``; a test holds the two equal)."""
_PROGRAM_SIZE_FIELDS = ("max_rel_frob", "max_changed_frac")


def seqskill_program_limits(task_limits_json: Mapping[str, Any], n_tensors: int | None = None) -> Limits:
    """The limits that bind an E5 update and every E5 reference run within limits: task.json's limits with every size
    limit but max_rel_frob off and at most ``SEQSKILL_MAX_CHANGED_FRAC`` of the entries changed.

    ``n_tensors``: the student's tensor count; ``max_ops`` becomes ``max(max_ops, n_tensors)``, the most ops a patch
    derived from a whole update can have (one per tensor). They equal
    ``weightcode.harness.arms.applied_limits(task_limits_json, "LEARN", n_tensors)`` (a test checks it).
    """
    if not isinstance(task_limits_json, Mapping):
        raise SeqskillError(f"task.json limits must be an object, got {type(task_limits_json).__name__}")
    if n_tensors is not None and (isinstance(n_tensors, bool) or not isinstance(n_tensors, int) or n_tensors < 1):
        raise SeqskillError(f"n_tensors must be an int >= 1, got {n_tensors!r}")
    limits = Limits(**dict(task_limits_json))
    disabled = {name: None for name in SIZE_LIMIT_FIELDS if name not in _PROGRAM_SIZE_FIELDS}
    max_ops = limits.max_ops if n_tensors is None else max(limits.max_ops, n_tensors)
    return dataclasses.replace(limits, **disabled, max_changed_frac=SEQSKILL_MAX_CHANGED_FRAC, max_ops=max_ops)


class SeqskillError(RuntimeError):
    """A seqskill instance, its store or its inputs were refused; nothing is overwritten."""


class StaleSeqskillInstance(SeqskillError):
    """An instance that is not active or whose files the current code would not write; it is not run."""


# ---------------------------------------------------------------------------------------------------------------
# Pure pieces: predictions, check parts, seeds, task


def seq_predictions(forward: ForwardFn, params: Params, x: np.ndarray, y: np.ndarray,
                    batch_size: int = EVALUATION_BATCH) -> np.ndarray:
    """The student's argmax at each labelled (OUT) position of ``x`` and IGNORE elsewhere, int64 [N, L].

    The one definition of ``pred_*`` in a seqskill evidence file; the harness re-checks the file with it.
    """
    y = np.asarray(y)
    pred = predict(forward, params, np.ascontiguousarray(x), batch_size).astype(np.int64)
    if pred.shape != y.shape:
        raise SeqskillError(f"predictions {pred.shape} do not match targets {y.shape}")
    pred[y == IGNORE] = IGNORE
    return pred


def seqskill_check_parts(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A condition's V_check cut by part: part 1 (new part) and part 4 (old part), each in the split's row order.

    Pure: returns x_new, y_new, x_old, y_old. Refuses a split holding any other part, so the longer-length and the
    composed rows can never reach a check.
    """
    x, y, part = np.asarray(arrays["x"]), np.asarray(arrays["y"]), np.asarray(arrays["part"])
    if not (len(x) == len(y) == len(part)) or y.ndim != 2:
        raise SeqskillError(f"check arrays x {x.shape}, y {y.shape} and part {part.shape} do not align")
    found = sorted({int(p) for p in np.unique(part)})
    if found != sorted(CHECK_PARTS):
        raise SeqskillError(f"a seqskill check split holds parts {list(CHECK_PARTS)} only, found {found}")
    new, old = part == S.PART_NEW_SHORT, part == S.PART_OLD
    return {
        "x_new": np.ascontiguousarray(x[new]),
        "y_new": np.ascontiguousarray(y[new]).astype(np.int64),
        "x_old": np.ascontiguousarray(x[old]),
        "y_old": np.ascontiguousarray(y[old]).astype(np.int64),
    }


def seqskill_seed(params_sha256_hex: str, condition: str) -> int:
    """Seed of the seqskill-v1 evidence draw of one student under one condition."""
    if not _SHA_RE.fullmatch(params_sha256_hex):
        raise ValueError(f"not a sha256 hex digest: {params_sha256_hex!r}")
    return _seed_from_text(f"{params_sha256_hex}|{S.DATASET}|{condition}|{SEQSKILL_RULE}")


def draw_replay_rows(n_train: int) -> np.ndarray:
    """Rule seqskill-replay-v1: N_REPLAY rows of the train sample, uniform, without replacement."""
    if n_train < N_REPLAY:
        raise SeqskillError(f"the train sample has {n_train} rows; the replay needs {N_REPLAY}")
    rng = np.random.default_rng([SPLIT_SEED, REPLAY_STREAM])
    return rng.choice(n_train, size=N_REPLAY, replace=False).astype(np.int64)


def draw_new_rows(n_pool: int, seed: int, n_old: int = N_REPLAY) -> tuple[np.ndarray, list[str], list[str]]:
    """Rule seqskill-v1: N_NEW rows of the condition's V_evidence, then the opaque ids of both parts."""
    if n_pool < N_NEW:
        raise SeqskillError(f"V_evidence has {n_pool} rows; the rule needs {N_NEW}")
    rng = np.random.default_rng(seed)
    new_rows = rng.choice(n_pool, size=N_NEW, replace=False).astype(np.int64)
    codes = rng.integers(0, 16**SPLIT_ID_HEX_DIGITS, size=N_NEW + n_old, dtype=np.int64)
    if np.unique(codes).size != codes.size:
        raise SeqskillError(f"split id collision for seed {seed}; the id width must grow (rule change)")
    ids = [f"e{int(code):0{SPLIT_ID_HEX_DIGITS}x}" for code in codes]
    return new_rows, ids[:N_NEW], ids[N_NEW:]


def seqskill_leak_tokens(condition: str) -> list[str]:
    """Lower-case tokens no editor-visible file may contain: the dataset tokens, the condition name, its rule kind."""
    index, kind = parse_condition(condition)
    return sorted(set(leak_tokens(S.DATASET)) | set(CONDITION_LEAK_WORDS) | {condition.lower(), kind.lower()})


def build_seqskill_task(operator: str, limits: Limits) -> dict[str, Any]:
    """Editor-visible data facts of a seqskill-v1 instance (schema weightcode.task.seqskill.v1); the harness words the task."""
    if operator not in {S.TOKENS[t] for t in S.NEW_OP_IDS}:
        raise SeqskillError(f"{operator!r} is not a reserved operator token")
    stated = patch_limits(limits)
    lo, hi = S.MIN_LEN, S.MAX_LEN
    short_lo, short_hi = S.NEW_SHORT_LENGTHS
    return {
        "schema": TASK_SCHEMA,
        "classes": list(CLASS_NAMES),
        "input": "token rows int64 [N, seq_len]; model_spec.json input.tokens gives every token's meaning and "
        "input.layout the row format",
        "new_operator": {
            "token": operator,
            "id": int(S.TOKEN_ID[operator]),
            "facts": "the model was never trained on a row holding this token: its embedding row is untrained and the "
            "model does not compute the operator. What the operator does is not given; the new evidence rows show it",
        },
        "old_operators": [S.TOKENS[t] for t in S.OLD_OP_IDS],
        "rows": {
            "layout": S.INPUT_LAYOUT,
            "lengths": [lo, hi],
            "new_operator_rows": f"the new operator alone (OPB = NONE) at lengths {lo} to {hi}, and composed with each "
            "old operator in either slot (OPA or OPB is the new operator) at the lengths of the new evidence",
            "old_operator_rows": f"single and composed old-operator rows at lengths {lo} to {hi}, as in training",
            "targets": "y holds the answer symbol index (0 to 15) at each OUT position and -1 elsewhere; a row is right "
            "when every OUT position is right",
        },
        "evidence": {
            "new": {
                "keys": ["x_new", "y_new", "pred_new"],
                "count": N_NEW,
                "description": f"rows of the new operator alone, lengths {short_lo} to {short_hi}, drawn at random "
                "from held-out data, with their answers",
            },
            "old": {
                "keys": ["x_old", "y_old", "pred_old"],
                "count": N_REPLAY,
                "description": "old-operator rows, a fixed random draw from the training distribution, with their "
                "answers",
            },
            "predictions": "pred_new and pred_old hold the model's argmax at each OUT position and -1 elsewhere",
            "ids": {"keys": ["split_ids_new", "split_ids_old"], "description": "one opaque id per example"},
        },
        "patch_check": {
            "budget_queries": QUERY_BUDGET,
            "parts": {
                "new": f"held-out rows of the new operator alone, lengths {short_lo} to {short_hi}",
                "old": f"held-out old-operator rows, single and composed, lengths {lo} to {hi}",
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
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    if tuple(sorted(arrays)) != tuple(sorted(EVIDENCE_KEYS)):
        raise SeqskillError(f"evidence keys {sorted(arrays)} != {sorted(EVIDENCE_KEYS)}")
    return params_sha256(arrays)


def _save_evidence(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    expected = _evidence_content_sha256(arrays)
    np.savez(path, **{key: np.ascontiguousarray(arrays[key]) for key in EVIDENCE_KEYS})
    with np.load(path, allow_pickle=False) as archive:
        reloaded = {key: archive[key] for key in archive.files}
    if _evidence_content_sha256(reloaded) != expected:
        raise SeqskillError(f"{path}: evidence content hash changed on reload")
    return expected


# ---------------------------------------------------------------------------------------------------------------
# Store: index.json, changes.jsonl, lock


def seqskill_root() -> Path:
    """``$WEIGHTCODE_PRIVATE/seqskill_instances`` (created if missing)."""
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


def read_seqskill_instance_index() -> dict[str, Any]:
    """The seqskill instance index; an empty one before the first build."""
    path = private_dir() / ROOT_NAME / "index.json"
    if not path.exists():
        return {"schema": INDEX_SCHEMA, "instances": [], "superseded": []}
    index = read_json(path)
    if not isinstance(index, dict) or index.get("schema") != INDEX_SCHEMA:
        raise SeqskillError(f"{path} is not a {INDEX_SCHEMA} index")
    return index


def active_seqskill_ids(condition: str | None = None) -> list[str]:
    """Ids of the active seqskill instances, optionally of one condition, in index order."""
    rows = read_seqskill_instance_index()["instances"]
    return [str(row["instance_id"]) for row in rows if condition is None or row["condition"] == condition]


def seqskill_instance_dir(instance_id: str) -> Path:
    """Directory of one seqskill instance (computed without creating the root); refuses an id of another format."""
    if not INSTANCE_ID_RE.fullmatch(str(instance_id)):
        raise SeqskillError(f"{instance_id!r} is not a seqskill instance id (q + 8 hex)")
    return private_dir() / ROOT_NAME / str(instance_id)


def load_seqskill_instance(instance: Instance | str) -> Instance:
    """An Instance, or the seqskill instance with this id loaded and hash-checked (``load_instance``)."""
    return instance if isinstance(instance, Instance) else load_instance(seqskill_instance_dir(instance))


_INDEX_KEYS = ("instance_id", "dataset", "condition", "arch", "seed", "model_id", "split_role", "new_operator",
               "evidence_rule", "created_at")


def _index_row(private: Mapping[str, Any]) -> dict[str, Any]:
    return {key: private[key] for key in _INDEX_KEYS}


def _new_instance_id(root: Path) -> str:
    while True:
        instance_id = "q" + secrets.token_hex(4)
        if not (root / instance_id).exists() and not (root / f".tmp-{instance_id}").exists():
            return instance_id


# ---------------------------------------------------------------------------------------------------------------
# Build


def _load(path: Path, expected: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    if params_sha256(arrays) != expected:
        raise SeqskillError(f"{path}: content hash differs from its dataset.json")
    return arrays


def _load_inputs(condition: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, str]]]:
    base_meta = read_dataset_json(S.DATASET)
    cond_meta = read_dataset_json(S.DATASET, condition)
    data, record = {}, {}
    sources = {"train": (None, base_meta), **{split: (condition, cond_meta) for split in ("V_evidence", "V_check",
                                                                                         "V_select")}}
    for split, (where, meta) in sources.items():
        path = split_path(S.DATASET, split, where)
        expected = meta["splits"][split]["sha256"]
        data[split] = _load(path, expected)
        record[split] = {"path": str(path), "sha256": file_sha256(path), "content_sha256": expected}
    return data, record


def _scan_visible(tokens: Sequence[str], **texts: str) -> None:
    for where, text in texts.items():
        found = scan_for_tokens(text, tokens)
        if found:
            raise SeqskillError(f"{where} contains {found}; no editor-visible file may")


def _base_em_by_part(forward: ForwardFn, params: Params, v_sel: Mapping[str, np.ndarray], batch_size: int) -> dict[str, float]:
    pred = predict(forward, params, v_sel["x"], batch_size)
    right = rows_right(v_sel["y"], pred)
    return {name: float(np.mean(right[v_sel["part"] == part])) for name, part in READOUT_PARTS.items()}


def _plan(
    condition: str,
    row: Mapping[str, Any],
    cond_meta: Mapping[str, Any],
    data: Mapping[str, Mapping[str, np.ndarray]],
    split_record: Mapping[str, Mapping[str, str]],
    replay_rows: np.ndarray,
    forward: ForwardFn,
    forward_sha: str,
    tokens: Sequence[str],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Evidence arrays, task and private record (without id and time) of one student's instance."""
    source = models_dir(S.DATASET) / str(row["model_id"])
    params = load_params(source / "model.npz")
    digest = params_sha256(params)
    if digest != row["params_sha256"]:
        raise SeqskillError(f"{source}/model.npz hash {digest} != seqops index {row['params_sha256']}")
    model_private = read_json(source / "private.json")
    if model_private.get("role") != "base" or not model_private.get("reserved_rows_unchanged"):
        raise SeqskillError(f"{source}: not a base student with untouched reserved embedding rows")
    spec_text = (source / "model_spec.json").read_text(encoding="utf-8")
    if json.loads(spec_text).get("classes") != list(CLASS_NAMES):
        raise SeqskillError(f"{source}/model_spec.json class list differs from the seqops symbols")
    operator = str(cond_meta["condition"]["operator"])

    v_sel = data["V_select"]
    base_em = _base_em_by_part(forward, params, v_sel, batch_size)
    lacking = {name: base_em[name] for name in ("i", "ii", "iii")}
    if any(value >= BASE_EM_MAX for value in lacking.values()):
        raise SeqskillError(f"{row['model_id']} already answers new-operator rows (exact match {lacking}); "
                            f"a seqskill base needs every new part below {BASE_EM_MAX}")

    v_ev, train = data["V_evidence"], data["train"]
    if set(np.unique(v_ev["part"]).tolist()) != {S.PART_NEW_SHORT}:
        raise SeqskillError("the condition's V_evidence must hold part 1 only")
    seed = seqskill_seed(digest, condition)
    new_rows, ids_new, ids_old = draw_new_rows(len(v_ev["y"]), seed, len(replay_rows))
    x_new, y_new = np.ascontiguousarray(v_ev["x"][new_rows]), np.ascontiguousarray(v_ev["y"][new_rows])
    coverage = S.symbol_coverage(x_new)
    if coverage != S.N_SYMBOLS:
        raise SeqskillError(f"the {N_NEW} new rows show {coverage} of {S.N_SYMBOLS} symbols; rule seqskill-v1 needs all")
    x_old, y_old = np.ascontiguousarray(train["x"][replay_rows]), np.ascontiguousarray(train["y"][replay_rows])
    arrays = {
        "x_new": x_new,
        "y_new": y_new.astype(np.int64),
        "pred_new": seq_predictions(forward, params, x_new, y_new, batch_size),
        "split_ids_new": np.array(ids_new),
        "x_old": x_old,
        "y_old": y_old.astype(np.int64),
        "pred_old": seq_predictions(forward, params, x_old, y_old, batch_size),
        "split_ids_old": np.array(ids_old),
    }
    task = build_seqskill_task(operator, Limits())
    task_text = json_text(task)
    _scan_visible(sorted(set(tokens) | {str(row["model_id"]).lower()}),
                  **{"task.json": task_text, "evidence.npz": npz_text(arrays)})
    _scan_visible(tokens, **{"model_spec.json": spec_text, "model.npz names": npz_text(params)})

    parts = seqskill_check_parts(data["V_check"])
    pred_new_ch = predict(forward, params, parts["x_new"], batch_size)
    pred_old_ch = predict(forward, params, parts["x_old"], batch_size)
    sel_counts = {name: int(np.count_nonzero(v_sel["part"] == part)) for name, part in READOUT_PARTS.items()}
    private = {
        "schema": PRIVATE_SCHEMA,
        "evidence_rule": SEQSKILL_RULE,
        "dataset": S.DATASET,
        "condition": condition,
        "new_operator": operator,
        "new_operator_id": int(S.TOKEN_ID[operator]),
        "rule_agreement": cond_meta["condition"]["rule_agreement"],
        "arch": row["arch"],
        "seed": int(row["seed"]),
        "model_id": row["model_id"],
        "role": "base",
        "split_role": "dev",
        "model_sha256": digest,
        "model_source_path": str(source / "model.npz"),
        "model_spec_sha256": file_sha256(source / "model_spec.json"),
        "forward_sha256": forward_sha,
        "base_size_check_em": model_private["size_check"]["em"],
        "base_vselect_em_by_part": base_em,
        "base_em_max": BASE_EM_MAX,
        "condition_dataset_json_sha256": file_sha256(dataset_json_path(S.DATASET, condition)),
        "vevidence_path": split_record["V_evidence"]["path"],
        "vevidence_sha256": split_record["V_evidence"]["sha256"],
        "vevidence_content_sha256": split_record["V_evidence"]["content_sha256"],
        "n_vevidence_new_pool": int(len(v_ev["y"])),
        "replay_rule": REPLAY_RULE,
        "replay_stream": [SPLIT_SEED, REPLAY_STREAM],
        "replay_path": split_record["train"]["path"],
        "replay_train_content_sha256": split_record["train"]["content_sha256"],
        "replay_train_rows": [int(r) for r in replay_rows],
        "replay_source_idx": [int(i) for i in train["idx"][replay_rows]],
        "evidence_seed": seed,
        "evidence_row_ids_new": [int(r) for r in new_rows],
        "evidence_source_idx_new": [int(i) for i in v_ev["idx"][new_rows]],
        "evidence_split_ids_new": ids_new,
        "evidence_split_ids_old": ids_old,
        "evidence_content_sha256": _evidence_content_sha256(arrays),
        "evidence_symbol_coverage": coverage,
        "evidence_new_lengths": [int(n) for n in v_ev["n"][new_rows]],
        "n_base_correct_evidence_new": int(np.count_nonzero(rows_right(y_new, arrays["pred_new"]))),
        "n_base_errors_evidence_old": int(np.count_nonzero(~rows_right(y_old, arrays["pred_old"]))),
        "vcheck_path": split_record["V_check"]["path"],
        "vcheck_sha256": split_record["V_check"]["sha256"],
        "vcheck_content_sha256": split_record["V_check"]["content_sha256"],
        "n_vcheck": int(len(data["V_check"]["y"])),
        "n_vcheck_new": int(len(parts["y_new"])),
        "n_vcheck_old": int(len(parts["y_old"])),
        "n_base_errors_vcheck_new": int(np.count_nonzero(~rows_right(parts["y_new"], pred_new_ch))),
        "n_base_errors_vcheck_old": int(np.count_nonzero(~rows_right(parts["y_old"], pred_old_ch))),
        "vselect_path": split_record["V_select"]["path"],
        "vselect_sha256": split_record["V_select"]["sha256"],
        "vselect_content_sha256": split_record["V_select"]["content_sha256"],
        "n_vselect": int(len(v_sel["y"])),
        "n_vselect_parts": sel_counts,
        "n_new": N_NEW,
        "n_replay": int(len(replay_rows)),
        "task_sha256": _text_sha256(task_text),
    }
    return arrays, task, private


def _create(root: Path, arrays: Mapping[str, np.ndarray], task: Mapping[str, Any], core: Mapping[str, Any],
            forward_file: Path) -> dict[str, Any]:
    instance_id = _new_instance_id(root)
    staging = root / f".tmp-{instance_id}"
    staging.mkdir()
    source = Path(core["model_source_path"])
    shutil.copyfile(source, staging / "model.npz")
    if params_sha256(load_params(staging / "model.npz")) != core["model_sha256"]:
        raise SeqskillError(f"{staging}/model.npz: copy hash mismatch")
    shutil.copyfile(source.with_name("model_spec.json"), staging / "model_spec.json")
    if file_sha256(staging / "model_spec.json") != core["model_spec_sha256"]:
        raise SeqskillError(f"{staging}/model_spec.json: copy hash mismatch")
    shutil.copyfile(forward_file, staging / "forward.py")
    if file_sha256(staging / "forward.py") != core["forward_sha256"]:
        raise SeqskillError(f"{staging}/forward.py: copy hash mismatch")
    if _save_evidence(staging / "evidence.npz", arrays) != core["evidence_content_sha256"]:
        raise SeqskillError(f"{staging}/evidence.npz: content hash mismatch")
    (staging / "task.json").write_text(json_text(task), encoding="utf-8")
    private = dict(core, instance_id=instance_id, created_at=utc_now(), numpy_version=np.__version__)
    (staging / "private.json").write_text(json_text(private), encoding="utf-8")
    with no_bytecode_cache():
        load_instance(staging)
    os.rename(staging, root / instance_id)
    return jsonable(private)


def _same_build(instance_dir: Path, task: Mapping[str, Any], core: Mapping[str, Any]) -> None:
    private = read_json(instance_dir / "private.json")
    stable = ("evidence_content_sha256", "task_sha256", "model_sha256", "forward_sha256", "model_spec_sha256",
              "vcheck_content_sha256", "vselect_content_sha256", "replay_train_rows", "condition_dataset_json_sha256")
    changed = [key for key in stable if private.get(key) != core.get(key)]
    if changed or json.loads((instance_dir / "task.json").read_text(encoding="utf-8")) != json.loads(json_text(task)):
        raise SeqskillError(f"{instance_dir}: differs from a rebuild in {changed or ['task.json']}; nothing overwritten")


def build_seqskill_instances(
    condition: str, *, seeds: Sequence[int] | None = None, batch_size: int = EVALUATION_BATCH
) -> list[dict[str, Any]]:
    """Build (or reuse) one seqskill-v1 instance per selected base student of seqops; returns one row per student.

    Students are the ``base`` rows of the seqops model index, optionally only some ``seeds``. An active instance of
    the same (condition, model id) is reused when a rebuild gives the same evidence, task and inputs, and refused
    otherwise.
    """
    from weightcode.models.export import read_index

    parse_condition(condition)
    students = [row for row in read_index(S.DATASET)
                if row["role"] == "base" and (seeds is None or int(row["seed"]) in {int(s) for s in seeds})]
    if not students:
        raise SeqskillError(f"no seqops base students for seeds {seeds}")
    base_meta = read_dataset_json(S.DATASET)
    cond_meta = read_dataset_json(S.DATASET, condition)
    if cond_meta["forward_sha256"] != base_meta["forward_sha256"]:
        raise SeqskillError(f"{condition}: forward.py hash differs from the dataset's")
    forward_file = forward_path(S.DATASET)
    with no_bytecode_cache():
        forward, forward_sha = load_forward(forward_file)
    if forward_sha != base_meta["forward_sha256"]:
        raise SeqskillError(f"{forward_file}: sha256 {forward_sha} != dataset.json")
    tokens = seqskill_leak_tokens(condition)
    _scan_visible(tokens, **{"forward.py": forward_file.read_text(encoding="utf-8")})
    data, split_record = _load_inputs(condition)
    replay_rows = draw_replay_rows(len(data["train"]["y"]))
    root = seqskill_root()
    outcomes = []
    with _exclusive_lock(root):
        index = read_seqskill_instance_index()
        for row in sorted(students, key=lambda r: (r["arch"], int(r["seed"]))):
            arrays, task, core = _plan(condition, row, cond_meta, data, split_record, replay_rows, forward,
                                       forward_sha, tokens, batch_size)
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


def supersede_seqskill_instance(instance_id: str, reason: str) -> dict[str, Any]:
    """Move an active seqskill instance to ``superseded`` in the index and return the moved row; its files stay.

    ``check_seqskill_current`` refuses a superseded instance, so nothing runs on it again. The row gains
    ``superseded_at`` and ``reason``, and changes.jsonl records the change.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise SeqskillError("superseding an instance needs a reason")
    root = seqskill_root()
    with _exclusive_lock(root):
        index = read_seqskill_instance_index()
        hits = [row for row in index["instances"] if row["instance_id"] == instance_id]
        if len(hits) != 1:
            raise SeqskillError(f"{instance_id} is not an active seqskill instance")
        moved = {**hits[0], "superseded_at": utc_now(), "reason": reason}
        index["instances"] = [row for row in index["instances"] if row["instance_id"] != instance_id]
        index["superseded"].append(moved)
        write_json_atomic(root / "index.json", index)
        with open(root / CHANGES_LOG_NAME, "a", encoding="utf-8") as log:
            log.write(json.dumps({"change": "superseded", **moved}, sort_keys=True) + "\n")
    return moved


# ---------------------------------------------------------------------------------------------------------------
# Running an instance: stale check, oracle, readout


def check_seqskill_current(instance_dir: str | Path) -> dict[str, Any]:
    """Refuse an instance that is not active or whose files the current code would not write; else return its task.

    Checks the directory name, the index, the rule, the task.json bytes against private.json, the task against a
    fresh ``build_seqskill_task`` and the hashes of model.npz, model_spec.json, forward.py and evidence.npz.
    """
    root = Path(instance_dir)
    private = read_json(root / "private.json")
    instance_id = str(private.get("instance_id"))
    if root.name != instance_id:
        raise StaleSeqskillInstance(f"{root}: directory name differs from private.json instance_id {instance_id!r}")
    index = read_seqskill_instance_index()
    if instance_id not in {row["instance_id"] for row in index["instances"]}:
        retired = {row["instance_id"] for row in index["superseded"]}
        status = "superseded" if instance_id in retired else "not in the index"
        raise StaleSeqskillInstance(f"seqskill instance {instance_id} is {status}; only active instances are run")
    if private.get("evidence_rule") != SEQSKILL_RULE:
        raise StaleSeqskillInstance(f"{instance_id} follows {private.get('evidence_rule')!r}, not {SEQSKILL_RULE}")
    task_text = (root / "task.json").read_text(encoding="utf-8")
    if _text_sha256(task_text) != private.get("task_sha256"):
        raise StaleSeqskillInstance(f"{root / 'task.json'}: bytes do not match private.json task_sha256")
    task = json.loads(task_text)
    expected = json.loads(json_text(build_seqskill_task(str(private["new_operator"]), Limits())))
    if task != expected:
        changed = sorted(key for key in set(task) | set(expected) if task.get(key) != expected.get(key))
        raise StaleSeqskillInstance(f"{instance_id}: task.json differs from what the current code writes in {changed}")
    if file_sha256(root / "model_spec.json") != private.get("model_spec_sha256"):
        raise StaleSeqskillInstance(f"{root / 'model_spec.json'}: hash differs from private.json")
    instance = load_instance(root)  # model.npz and forward.py hashes against private.json
    if _evidence_content_sha256(instance.evidence) != private.get("evidence_content_sha256"):
        raise StaleSeqskillInstance(f"{root / 'evidence.npz'}: content hash differs from private.json")
    return task


def _load_recorded(private: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    key = {"V_check": "vcheck", "V_select": "vselect"}[split]
    path, recorded = Path(str(private[f"{key}_path"])), private.get(f"{key}_content_sha256")
    if recorded is None:
        raise SeqskillError(f"the instance record has no {key}_content_sha256")
    return _load(path, str(recorded))


def make_seqskill_oracle(
    instance: Instance | str, budget: int = QUERY_BUDGET, batch_size: int = EVALUATION_BATCH
) -> JointPatchCheckOracle:
    """Joint counted oracle of a seqskill instance: its condition's V_check cut by part (``seqskill_check_parts``)."""
    resolved = load_seqskill_instance(instance)
    private = resolved.private
    parts = seqskill_check_parts(_load_recorded(private, "V_check"))
    if (len(parts["y_new"]), len(parts["y_old"])) != (private["n_vcheck_new"], private["n_vcheck_old"]):
        raise SeqskillError(f"{resolved.instance_id}: V_check parts differ in size from the instance record")
    return JointPatchCheckOracle(parts["x_new"], parts["y_new"], parts["x_old"], parts["y_old"], resolved.forward,
                                 resolved.params, budget, batch_size)


def _part_summary(y: np.ndarray, base: np.ndarray, child: np.ndarray) -> dict[str, Any]:
    rb, rc = rows_right(y, base), rows_right(y, child)
    em_base, em_child = float(np.mean(rb)), float(np.mean(rc))
    return {"n": int(len(y)), "em_base": em_base, "em_child": em_child, "delta": em_child - em_base,
            "fixed": int(np.count_nonzero(~rb & rc)), "broken": int(np.count_nonzero(rb & ~rc)),
            "token_acc_base": token_accuracy(y, base), "token_acc_child": token_accuracy(y, child)}


def _breakdown(arrays: Mapping[str, np.ndarray], mask: np.ndarray, base: np.ndarray, child: np.ndarray,
               key: str) -> dict[str, Any]:
    y = arrays["y"]
    out: dict[str, Any] = {}
    if key == "length":
        groups = {str(int(n)): (arrays["n"] == n) for n in np.unique(arrays["n"][mask])}
    else:
        pairs = sorted({(int(a), int(b)) for a, b in zip(arrays["opa"][mask], arrays["opb"][mask])})
        groups = {f"{S.TOKENS[a]}+{S.TOKENS[b]}": (arrays["opa"] == a) & (arrays["opb"] == b) for a, b in pairs}
    for name, rows in groups.items():
        rows = rows & mask
        out[name] = {"n": int(rows.sum()), "em_base": float(np.mean(rows_right(y[rows], base[rows]))),
                     "em_child": float(np.mean(rows_right(y[rows], child[rows])))}
    return out


def seqskill_readout(instance: Instance | str, child: Params, batch_size: int = EVALUATION_BATCH) -> dict[str, Any]:
    """Exact match of ``child`` against the instance's base on the condition's V_select, by part (PREREG E5).

    Parts: i (the new operator alone, the lengths of the evidence), ii (alone, longer lengths), iii (composed with
    each old operator, both orders), iv (old operators). ``new`` pools parts i to iii and ``all`` every row. Each part
    gives n, base and child exact match, delta, fixed, broken and token accuracies; ``*_points`` is 100 * delta.
    Breakdowns: part i and ii by length, part iii by operator pair and by the kind of its old operator
    (``iii_by_map_kind``: position or symbol map, ``MAP_KINDS``), and part iv into single and composed old-operator
    rows (``iv_by_kind``); each kind has the fields of a part.
    """
    resolved = load_seqskill_instance(instance)
    arrays = _load_recorded(resolved.private, "V_select")
    y = np.asarray(arrays["y"])
    base = predict(resolved.forward, resolved.params, arrays["x"], batch_size)
    pred = predict(resolved.forward, child, arrays["x"], batch_size)
    part = np.asarray(arrays["part"])
    result: dict[str, Any] = {}
    for name, code in READOUT_PARTS.items():
        rows = part == code
        result[name] = _part_summary(y[rows], base[rows], pred[rows])
    new = np.isin(part, NEW_PARTS)
    result["new"] = _part_summary(y[new], base[new], pred[new])
    result["all"] = _part_summary(y, base, pred)
    for name in (*READOUT_PARTS, "new", "all"):
        result[f"{name}_points"] = 100.0 * result[name]["delta"]
    result["i_by_length"] = _breakdown(arrays, part == S.PART_NEW_SHORT, base, pred, "length")
    result["ii_by_length"] = _breakdown(arrays, part == S.PART_NEW_LONG, base, pred, "length")
    result["iii_by_pair"] = _breakdown(arrays, part == S.PART_NEW_COMPOSED, base, pred, "pair")
    composed_new = part == S.PART_NEW_COMPOSED
    opa, opb = np.asarray(arrays["opa"]), np.asarray(arrays["opb"])
    old_of_pair = np.where(np.isin(opa, S.NEW_OP_IDS), opb, opa)
    result["iii_by_map_kind"] = {}
    for kind, ids in MAP_KINDS.items():
        rows = composed_new & np.isin(old_of_pair, ids)
        result["iii_by_map_kind"][kind] = _part_summary(y[rows], base[rows], pred[rows])
    if sum(entry["n"] for entry in result["iii_by_map_kind"].values()) != result["iii"]["n"]:
        raise SeqskillError("part iii rows do not split into position-map and symbol-map pairs")
    old_rows = part == S.PART_OLD
    kinds = {"single": old_rows & (opb == S.NONE), "composed": old_rows & (opb != S.NONE)}
    result["iv_by_kind"] = {kind: _part_summary(y[rows], base[rows], pred[rows]) for kind, rows in kinds.items()}
    return result


__all__ = [
    "BASE_EM_MAX", "CHECK_PART_KEYS", "CHECK_PARTS", "EDITOR_VISIBLE_FILES", "EVIDENCE_KEYS", "INSTANCE_ID_RE",
    "MAP_KINDS", "N_NEW", "N_REPLAY", "PRIVATE_SCHEMA", "READOUT_PARTS", "REPLAY_RULE", "REPLAY_STREAM", "SEQSKILL_RULE",
    "SEQSKILL_MAX_CHANGED_FRAC", "TASK_SCHEMA", "SeqskillError", "StaleSeqskillInstance", "active_seqskill_ids", "build_seqskill_instances",
    "build_seqskill_task", "check_seqskill_current", "draw_new_rows", "draw_replay_rows", "load_seqskill_instance",
    "make_seqskill_oracle", "read_seqskill_instance_index", "seq_predictions", "seqskill_check_parts",
    "seqskill_instance_dir", "seqskill_leak_tokens", "seqskill_program_limits", "seqskill_readout", "seqskill_root", "seqskill_seed",
    "supersede_seqskill_instance",
]
