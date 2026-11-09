"""Rule policyskill-v1 (PREREG E14): instances that ask an update to teach a trained policy a new skill.

A film_policy student (``weightcode.models.e14_policy``, imported by ``weightcode.models.e14_export``) was trained by imitation on
the old mission ("go to {the|a} {color} {type}") only: the instruction words "pick" and "up" never occurred in its
training, so their embedding rows are untrained, and the pickup action was never a target. The new mission is "pick up
{the|a} {color} {type}" in the same rooms (``weightcode.data.e14``). There is one instance per base student. The layout is
the evidence-v1 layout, so ``weightcode.baselines.instance_io.load_instance`` loads it unchanged::

    $WEIGHTCODE_PRIVATE/e14_instances/
        index.json                       {"schema", "instances": [active rows], "superseded": [rows]}
        changes.jsonl                    one line per created or superseded instance
        p<8 hex>/
            model.npz, model_spec.json   byte copies of the student's files
            forward.py                   byte copy of the dataset forward
            evidence.npz                 the evidence, keys EVIDENCE_KEYS; it is the LEARN experience as it is
            task.json                    schema weightcode.task.policyskill.v1: data facts only (the harness words the task)
            private.json                 schema weightcode.instance_private.policyskill.v1, never editor-visible

Evidence of rule policyskill-v1:

- new part: every step of the first ``K_NEW`` usable new V_evidence demonstrations in environment order (the bot's
  episodes on the new mission; PREREG E14 as built), with the bot's action, the demonstration number (0 to 7) and the
  step number;
- old part: rule policyskill-replay-v1, ``N_REPLAY`` steps drawn uniformly without replacement from the student's own
  training rows (``weightcode.data.e14.train_rows`` at its episode count) with ``default_rng([SPLIT_SEED, REPLAY_STREAM])``;
- logits_*: the student's float32 action logits from forward.py; the action is their argmax;
- split ids: one ``e`` + 12 hex id per row from the instance generator, new rows first.

The student must be a gated base (old V_select success >= 0.95, new V_select success <= 0.20, read at the import) with
untouched reserved word rows, or no instance is built.

Check (``make_e14_oracle``, ``RolloutJointOracle``): greedy forward.py rollouts on the 500 new and the 500 old V_check
episodes; per episode 1 on success; S_new and S_old are fixed minus broken episodes, child against base, and
S = S_new + S_old (the fields of ``JointPatchCheckOracle``). Readout (``e14_readout``): the same on the 1,000 new and
the 1,000 old V_select episodes, with lengths, pickups, the breakdown by article and object type, and the split by
whether the episode's (color, type) occurs among the 8 demonstrations. ``e14_wrapper_readout`` reads PREREG E14's
hand-written wrapper (the base, with pickup when the cell in front matches the instruction) the same way.

V_check and V_select are environment seed ranges in the code (``weightcode.data.e14.SEED_RANGES``). The workstation keeps
one file of the V_check seeds (``vcheck_file``: x = seeds, y = 1, mission = 1 new and 0 old) for the editor harness's
session ledger; every reader checks the seeds against the instance record's content hash.

Limits (PREREG E14: as E13): task.json states ``weightcode.patch.Limits`` with max_rel_frob None (the per-tensor rel-Frob limit
lifted); a LEARN session applies them through ``weightcode.harness.arms.applied_limits``, which leaves at most 20% of the
entries changed (embedding rows counted with their table) and max_ops at least the tensor count
(``e14_program_limits`` is the same rule for the references).
"""

from __future__ import annotations

import dataclasses
import fcntl
import functools
import hashlib
import json
import os
import re
import secrets
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import Params, load_params, params_sha256
from weightcode.common.paths import private_dir
from weightcode.data import e14 as D
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
from weightcode.evalkit.patch_check import ForwardFn, JointPatchCheckOracle, load_forward
from weightcode.evalkit.targets import file_sha256, json_text, jsonable, no_bytecode_cache, read_json, utc_now, write_json_atomic
from weightcode.patch import OP_NAMES, Limits
from weightcode.patch import SCHEMA as PATCH_SCHEMA
from weightcode.patch.limits import SIZE_LIMIT_FIELDS, with_dense_file_cap

POLICYSKILL_RULE = "policyskill-v1"
REPLAY_RULE = "policyskill-replay-v1"
REPLAY_STREAM = 1400
"""Generator stream of the E14 replay draw (range 1400-1499 is E14's)."""
K_NEW = 8
N_REPLAY = 64
TASK_SCHEMA = "weightcode.task.policyskill.v1"
PRIVATE_SCHEMA = "weightcode.instance_private.policyskill.v1"
INDEX_SCHEMA = "weightcode.policyskill_instances_index.v1"
CHANGES_LOG_NAME = "changes.jsonl"
ROOT_NAME = "e14_instances"
INSTANCE_ID_RE = re.compile(r"^p[0-9a-f]{8}$")
EVIDENCE_KEYS: tuple[str, ...] = (
    "x_new", "y_new", "logits_new", "episode_new", "step_new", "split_ids_new",
    "x_old", "y_old", "logits_old", "split_ids_old",
)
EDITOR_VISIBLE_FILES: tuple[str, ...] = ("model.npz", "model_spec.json", "forward.py", "evidence.npz", "task.json")
LEAK_WORDS: tuple[str, ...] = D.LEAK_WORDS
"""PREREG E14 as built (WEIGHTCODE-main decision b): the three level names, pickupinstr, babyai and minigrid."""
CHECK_RANGES = {"new": "V_check_new", "old": "V_check_old"}
SELECT_RANGES = {"new": "V_select_new", "old": "V_select_old"}
VCHECK_FILE = "V_check.npz"
GATE = 0.95
HEADROOM = 0.20
E14_MAX_CHANGED_FRAC = 0.20
"""PREREG E14: an update may change at most 20% of the parameter entries (``weightcode.harness.arms.PROGRAM_MAX_CHANGED_FRAC``;
a test holds the two equal)."""
_PROGRAM_SIZE_FIELDS = ("max_rel_frob", "max_changed_frac")
TASK_LIMITS = Limits(max_rel_frob=None)
"""The limits task.json states: the defaults with the per-tensor rel-Frob limit lifted (PREREG E14, as E13)."""


class E14InstanceError(RuntimeError):
    """An E14 instance, its store or its inputs were refused; nothing is overwritten."""


class StaleE14Instance(E14InstanceError):
    """An instance that is not active or whose files the current code would not write; it is not run."""


def e14_program_limits(task_limits_json: Mapping[str, Any], n_tensors: int | None = None,
                       dense_bytes: int | None = None) -> Limits:
    """The limits that bind an E14 update and every E14 reference run within limits: task.json's limits with every
    size limit but max_rel_frob (None in an E14 task.json) off, at most ``E14_MAX_CHANGED_FRAC`` of the entries
    changed, max_ops = max(max_ops, n_tensors) and the dense file cap (``with_dense_file_cap``). They equal
    ``weightcode.harness.arms.applied_limits(task_limits_json, "LEARN", n_tensors, dense_bytes)`` (a test checks it)."""
    if not isinstance(task_limits_json, Mapping):
        raise E14InstanceError(f"task.json limits must be an object, got {type(task_limits_json).__name__}")
    if n_tensors is not None and (isinstance(n_tensors, bool) or not isinstance(n_tensors, int) or n_tensors < 1):
        raise E14InstanceError(f"n_tensors must be an int >= 1, got {n_tensors!r}")
    limits = Limits(**dict(task_limits_json))
    disabled = {name: None for name in SIZE_LIMIT_FIELDS if name not in _PROGRAM_SIZE_FIELDS}
    max_ops = limits.max_ops if n_tensors is None else max(limits.max_ops, n_tensors)
    return with_dense_file_cap(
        dataclasses.replace(limits, **disabled, max_changed_frac=E14_MAX_CHANGED_FRAC, max_ops=max_ops), dense_bytes)


# ---------------------------------------------------------------------------------------------------------------
# Pure pieces: logits, seeds, draws, the V_check arrays, the task


def policy_logits(forward: ForwardFn, params: Params, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """forward.py's action logits on rows ``x``, float32 [N, 7]: the one definition of ``logits_*``."""
    x = np.asarray(x)
    if x.dtype != np.int64 or x.ndim != 2 or x.shape[1] != D.ROW_LEN:
        raise E14InstanceError(f"rows must be int64 [N, {D.ROW_LEN}], got {x.dtype}{x.shape}")
    parts = [np.asarray(forward(params, x[start:start + batch_size])) for start in range(0, len(x), batch_size)]
    logits = np.concatenate(parts) if parts else np.zeros((0, len(D.ACTIONS)), dtype=np.float32)
    if logits.dtype != np.float32 or logits.shape != (len(x), len(D.ACTIONS)) or not np.all(np.isfinite(logits)):
        raise E14InstanceError(f"forward.py returned {logits.dtype}{logits.shape} or non-finite logits")
    return logits


def logits_fn(forward: ForwardFn, params: Params) -> D.LogitsFn:
    """``rows -> logits`` of forward.py under ``params``, for ``weightcode.data.e14.rollouts``."""
    frozen = {key: np.ascontiguousarray(value) for key, value in params.items()}

    def run(x: np.ndarray) -> np.ndarray:
        return np.asarray(forward(frozen, x))
    return run


def instance_seed(params_sha256_hex: str) -> int:
    """Seed of the policyskill-v1 id draw of one student."""
    if not re.fullmatch(r"[0-9a-f]{64}", params_sha256_hex):
        raise ValueError(f"not a sha256 hex digest: {params_sha256_hex!r}")
    return _seed_from_text(f"{params_sha256_hex}|{D.DATASET}|{POLICYSKILL_RULE}")


def draw_replay_rows(n_rows: int) -> np.ndarray:
    """Rule policyskill-replay-v1: ``N_REPLAY`` of the student's ``n_rows`` training rows, uniform, no replacement."""
    if n_rows < N_REPLAY:
        raise E14InstanceError(f"the training rows number {n_rows}; the replay needs {N_REPLAY}")
    rng = np.random.default_rng([SPLIT_SEED, REPLAY_STREAM])
    return np.sort(rng.choice(n_rows, size=N_REPLAY, replace=False)).astype(np.int64)


def draw_split_ids(seed: int, n_new: int, n_old: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 16**SPLIT_ID_HEX_DIGITS, size=n_new + n_old, dtype=np.int64)
    if np.unique(codes).size != codes.size:
        raise E14InstanceError(f"split id collision for seed {seed}; the id width must grow (rule change)")
    ids = [f"e{int(code):0{SPLIT_ID_HEX_DIGITS}x}" for code in codes]
    return ids[:n_new], ids[n_new:]


def vcheck_arrays() -> dict[str, np.ndarray]:
    """The V_check seeds as one file's arrays: x = seeds (new then old), y = 1 (success), mission = 1 new, 0 old."""
    new, old = (D.SEED_RANGES[CHECK_RANGES[part]].seeds for part in ("new", "old"))
    seeds = np.concatenate([new, old]).astype(np.int64)
    mission = np.concatenate([np.ones(len(new), np.int64), np.zeros(len(old), np.int64)])
    return {"x": seeds, "y": np.ones(len(seeds), np.int64), "mission": mission}


def check_seeds(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The new and old V_check seeds of a V_check file's arrays; they must be the code's ranges."""
    expected = vcheck_arrays()
    if sorted(arrays) != sorted(expected) or not all(np.array_equal(arrays[k], expected[k]) for k in expected):
        raise E14InstanceError("the V_check arrays are not the code's V_check seed ranges")
    mission = np.asarray(arrays["mission"])
    return {"new": np.asarray(arrays["x"])[mission == 1], "old": np.asarray(arrays["x"])[mission == 0]}


def seeds_sha256(seeds: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(seeds, dtype="<i8").tobytes()).hexdigest()


def success_sha256(success: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(success, dtype=np.uint8).tobytes()).hexdigest()


def e14_leak_tokens() -> list[str]:
    """Lower-case tokens no editor-visible file may contain: the dataset tokens and ``LEAK_WORDS``."""
    return sorted(set(leak_tokens(D.DATASET)) | set(LEAK_WORDS))


def build_e14_task(n_new_rows: int, limits: Limits = TASK_LIMITS) -> dict[str, Any]:
    """Editor-visible data facts of a policyskill-v1 instance (schema weightcode.task.policyskill.v1); the harness words the
    task. ``n_new_rows``: the steps of the ``K_NEW`` demonstrations."""
    stated = patch_limits(limits)
    n_check = {part: D.SEED_RANGES[name].n for part, name in CHECK_RANGES.items()}
    return {
        "schema": TASK_SCHEMA,
        "input": f"observation rows int64 [N, {D.ROW_LEN}]: the grid and the instruction; model_spec.json input gives "
                 "the layout, the cell encoding and every instruction word's meaning",
        "actions": list(D.ACTIONS),
        "trained_mission": "go to {the|a} {color} {type}",
        "new_mission": {
            "instruction": "pick up {the|a} {color} {type}",
            "words": [{"word": word, "id": int(D.WORD_ID[word])} for word in D.RESERVED_WORDS],
            "facts": "the model was never trained on a row holding these words: their embedding rows are untrained, "
                     f"and the action {D.ACTIONS[D.PICKUP]} (index {D.PICKUP}) was never a training target, so the "
                     "model does not perform the new mission. The rooms, objects and colors are those of the trained "
                     f"mission; an episode ends when the mission is done or after {D.MAX_STEPS} steps. What completes "
                     "the new mission is not stated; the demonstrations show it",
        },
        "rows": {
            "layout": D.encoding_record()["row"],
            "targets": "y holds the planner's action index (0 to 6) at that step",
        },
        "evidence": {
            "new": {
                "keys": ["x_new", "y_new", "logits_new", "episode_new", "step_new"],
                "count": int(n_new_rows),
                "demonstrations": K_NEW,
                "description": f"every step of {K_NEW} successful demonstrations of the new mission by a planner that "
                               "sees the whole grid, in held-out rooms; episode_new numbers the demonstration (0 to "
                               f"{K_NEW - 1}) and step_new the step in it (0 is the first)",
            },
            "old": {
                "keys": ["x_old", "y_old", "logits_old"],
                "count": N_REPLAY,
                "description": "single steps of the trained mission, a fixed random draw from the model's training "
                               "demonstrations, with the planner's action",
            },
            "logits": "logits_new and logits_old hold the model's action logits (float32 [N, 7]) on each row; the model "
                      "takes the argmax action",
            "ids": {"keys": ["split_ids_new", "split_ids_old"], "description": "one opaque id per row"},
        },
        "patch_check": {
            "budget_queries": QUERY_BUDGET,
            "parts": {
                "new": f"{n_check['new']} held-out episodes of the new mission",
                "old": f"{n_check['old']} held-out episodes of the trained mission",
            },
            "readout": "each query applies one patch and runs the patched model greedily (argmax action at every "
                       f"step, at most {D.MAX_STEPS} steps) on every held-out episode of both parts, and returns "
                       "aggregate numbers, each against the unedited model: the success rate, fixed (episodes the "
                       "unedited model failed that the patched one completes), broken (episodes the unedited model "
                       "completed that the patched one fails), the changed-parameter count, S_new and S_old (fixed - "
                       "broken on each part) and S = S_new + S_old; no per-episode information",
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


def evidence_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    if tuple(sorted(arrays)) != tuple(sorted(EVIDENCE_KEYS)):
        raise E14InstanceError(f"evidence keys {sorted(arrays)} != {sorted(EVIDENCE_KEYS)}")
    return params_sha256(arrays)


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    expected = params_sha256(arrays)
    np.savez(path, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
    with np.load(path, allow_pickle=False) as archive:
        reloaded = {key: archive[key] for key in archive.files}
    if params_sha256(reloaded) != expected:
        raise E14InstanceError(f"{path}: content hash changed on reload")
    return expected


# ---------------------------------------------------------------------------------------------------------------
# Store: index.json, changes.jsonl, lock, the V_check file


def e14_root() -> Path:
    """``$WEIGHTCODE_PRIVATE/e14_instances`` (created if missing)."""
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


def read_e14_instance_index() -> dict[str, Any]:
    """The E14 instance index; an empty one before the first build."""
    path = private_dir() / ROOT_NAME / "index.json"
    if not path.exists():
        return {"schema": INDEX_SCHEMA, "instances": [], "superseded": []}
    index = read_json(path)
    if not isinstance(index, dict) or index.get("schema") != INDEX_SCHEMA:
        raise E14InstanceError(f"{path} is not a {INDEX_SCHEMA} index")
    return index


def active_e14_ids() -> list[str]:
    return [str(row["instance_id"]) for row in read_e14_instance_index()["instances"]]


def e14_instance_dir(instance_id: str) -> Path:
    """Directory of one E14 instance (computed without creating the root); refuses an id of another form."""
    if not INSTANCE_ID_RE.fullmatch(str(instance_id)):
        raise E14InstanceError(f"{instance_id!r} is not an E14 instance id (p + 8 hex)")
    return private_dir() / ROOT_NAME / str(instance_id)


def load_e14_instance(instance: Instance | str) -> Instance:
    return instance if isinstance(instance, Instance) else load_instance(e14_instance_dir(instance))


def vcheck_file() -> Path:
    """``$WEIGHTCODE_PRIVATE/e14_gotolocal/splits/V_check.npz``, the V_check seeds for the session ledger."""
    return D.store_dir() / "splits" / VCHECK_FILE


def write_vcheck_file() -> dict[str, str]:
    """Write the V_check seed file once (refusing a different existing one); returns its path and hashes."""
    path = vcheck_file()
    arrays = vcheck_arrays()
    content = params_sha256(arrays)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if params_sha256({key: archive[key] for key in archive.files}) != content:
                raise E14InstanceError(f"{path} exists with other content; nothing overwritten")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        _save_npz(temporary, arrays)
        temporary.replace(path)
    return {"path": str(path), "sha256": file_sha256(path), "content_sha256": content}


_INDEX_KEYS = ("instance_id", "dataset", "arch", "seed", "model_id", "split_role", "evidence_rule", "created_at")


def _index_row(private: Mapping[str, Any]) -> dict[str, Any]:
    return {key: private[key] for key in _INDEX_KEYS}


def _new_instance_id(root: Path) -> str:
    while True:
        instance_id = "p" + secrets.token_hex(4)
        if not (root / instance_id).exists() and not (root / f".tmp-{instance_id}").exists():
            return instance_id


# ---------------------------------------------------------------------------------------------------------------
# Rollout check


class RolloutJointOracle(JointPatchCheckOracle):
    """``JointPatchCheckOracle`` scored by greedy forward.py rollouts: part ``new`` runs the new mission and part
    ``old`` the trained mission from their environment seeds; an episode is right (1) when it succeeds. Every field of
    a query (score, fixed, broken, n, n_base_errors, base_acc, child_acc, with ``_new`` and ``_old`` suffixes) is the
    joint oracle's, counted in episodes."""

    def __init__(self, seeds_new: np.ndarray, seeds_old: np.ndarray, forward: ForwardFn,
                 base_params: Mapping[str, np.ndarray], budget: int,
                 expected_base_success_sha256: Mapping[str, str] | None = None) -> None:
        self.last_results: dict[str, D.RolloutResult] = {}
        seeds_new = np.asarray(seeds_new, dtype=np.int64)
        seeds_old = np.asarray(seeds_old, dtype=np.int64)
        super().__init__(seeds_new, np.ones(len(seeds_new), np.int64), seeds_old, np.ones(len(seeds_old), np.int64),
                         forward, base_params, budget)
        self.base_results = dict(self.last_results)
        if expected_base_success_sha256 is not None:
            got = {part: success_sha256(result.success) for part, result in self.base_results.items()}
            if got != dict(expected_base_success_sha256):
                raise E14InstanceError(f"the base's V_check episodes differ from the instance record: {got}")

    def _predict_part(self, part: str, params: Mapping[str, np.ndarray]) -> np.ndarray:
        result = D.rollouts(logits_fn(self._forward, params), part, self._x[part])
        self.last_results[part] = result
        return result.success.astype(np.int64)


def _base_record(models_row: Mapping[str, Any], model_private: Mapping[str, Any]) -> dict[str, Any]:
    readouts = model_private["readouts"]
    if not (models_row.get("gate_passed") and readouts["gate"]["passed"] and readouts["headroom"]["passed"]):
        raise E14InstanceError(f"{models_row['model_id']} is not a gated base: gate {readouts['gate']}, headroom "
                               f"{readouts['headroom']}")
    if not model_private.get("reserved_rows_unchanged"):
        raise E14InstanceError(f"{models_row['model_id']}: the reserved word rows changed in training")
    return {"base_vselect_success_old": readouts["old"]["success"],
            "base_vselect_success_new": readouts["new"]["success"],
            "base_vselect_success_sha256": {m: readouts[m]["success_sha256"] for m in D.MISSIONS}}


def _plan(models_row: Mapping[str, Any], forward: ForwardFn, forward_sha: str, tokens: list[str],
          vcheck: Mapping[str, str]) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Evidence arrays, task and private record (without id and time) of one student's instance."""
    from weightcode.models.e14_export import models_dir

    source = models_dir() / str(models_row["model_id"])
    params = load_params(source / "model.npz")
    digest = params_sha256(params)
    if digest != models_row["params_sha256"]:
        raise E14InstanceError(f"{source}/model.npz hash {digest} != the models index {models_row['params_sha256']}")
    model_private = read_json(source / "private.json")
    base = _base_record(models_row, model_private)
    spec_text = (source / "model_spec.json").read_text(encoding="utf-8")

    manifest = read_json(D.demos_dir() / D.DEMOS_MANIFEST)
    new_demos = D.load_demos("V_evidence_new", manifest["sets"]["V_evidence_new"]["sha256"])
    train = D.load_demos("train", manifest["sets"]["train"]["sha256"])
    episodes = D.usable_episodes(new_demos, K_NEW)
    rows_new = np.flatnonzero(np.isin(new_demos["episode"], episodes))
    x_new = new_demos["x"][rows_new].astype(np.int64)
    y_new = new_demos["y"][rows_new].astype(np.int64)
    episode_new = np.searchsorted(episodes, new_demos["episode"][rows_new]).astype(np.int64)
    step_new = new_demos["step"][rows_new].astype(np.int64)
    D.check_rows(x_new, D.mission_verbs("new"))
    if not (np.isin(y_new, D.NEW_ACTIONS).all() and np.count_nonzero(y_new == D.PICKUP) == K_NEW):
        raise E14InstanceError("the new demonstrations must use left, right, forward and one pickup each")
    x_train, y_train = D.train_rows(train, int(models_row["n_episodes"]))
    replay_rows = draw_replay_rows(len(x_train))
    x_old, y_old = x_train[replay_rows], y_train[replay_rows]
    seed = instance_seed(digest)
    ids_new, ids_old = draw_split_ids(seed, len(x_new), N_REPLAY)
    arrays = {
        "x_new": x_new, "y_new": y_new, "logits_new": policy_logits(forward, params, x_new),
        "episode_new": episode_new, "step_new": step_new, "split_ids_new": np.array(ids_new),
        "x_old": x_old, "y_old": y_old, "logits_old": policy_logits(forward, params, x_old),
        "split_ids_old": np.array(ids_old),
    }
    task = build_e14_task(len(x_new))
    task_text = json_text(task)
    visible = sorted(set(tokens) | {str(models_row["model_id"]).lower()})
    _scan_visible(visible, **{"task.json": task_text, "evidence.npz": npz_text(arrays)})
    _scan_visible(tokens, **{"model_spec.json": spec_text, "model.npz names": npz_text(params)})

    oracle = RolloutJointOracle(*(D.SEED_RANGES[CHECK_RANGES[p]].seeds for p in ("new", "old")), forward, params, 0)
    base_check = oracle.base_results
    ep_seeds = new_demos["ep_seed"][episodes]
    private = {
        "schema": PRIVATE_SCHEMA,
        "evidence_rule": POLICYSKILL_RULE,
        "dataset": D.DATASET,
        "arch": models_row["arch"],
        "seed": int(models_row["seed"]),
        "model_id": models_row["model_id"],
        "size": models_row["size"],
        "n_train_episodes": int(models_row["n_episodes"]),
        "role": "base",
        "split_role": "dev",
        "model_sha256": digest,
        "model_source_path": str(source / "model.npz"),
        "model_spec_sha256": file_sha256(source / "model_spec.json"),
        "forward_sha256": forward_sha,
        "n_parameters": int(sum(v.size for v in params.values())),
        "n_tensors": len(params),
        **base,
        "gate": GATE,
        "headroom": HEADROOM,
        "reserved_word_ids": [int(i) for i in D.RESERVED_IDS],
        "pickup_action": int(D.PICKUP),
        "demos_manifest_sha256": file_sha256(D.demos_dir() / D.DEMOS_MANIFEST),
        "vevidence_demos_sha256": manifest["sets"]["V_evidence_new"]["sha256"],
        "vevidence_path": str(D.demos_dir() / D.DEMO_FILES["V_evidence_new"]),
        "evidence_episodes": [int(e) for e in episodes],
        "evidence_episode_env_ids": [int(s) for s in ep_seeds],
        "n_new_rows": int(len(x_new)),
        "n_new_demonstrations": K_NEW,
        "replay_rule": REPLAY_RULE,
        "replay_stream": [SPLIT_SEED, REPLAY_STREAM],
        "replay_path": str(D.demos_dir() / D.DEMO_FILES["train"]),
        "replay_train_demos_sha256": manifest["sets"]["train"]["sha256"],
        "replay_train_rows": [int(r) for r in replay_rows],
        "n_replay": N_REPLAY,
        "evidence_seed": seed,
        "evidence_split_ids_new": ids_new,
        "evidence_split_ids_old": ids_old,
        "evidence_content_sha256": evidence_content_sha256(arrays),
        "n_base_right_evidence_new": int(np.count_nonzero(arrays["logits_new"].argmax(1) == y_new)),
        "n_base_right_evidence_old": int(np.count_nonzero(arrays["logits_old"].argmax(1) == y_old)),
        "vcheck_path": vcheck["path"],
        "vcheck_sha256": vcheck["sha256"],
        "vcheck_content_sha256": vcheck["content_sha256"],
        "n_vcheck": int(len(base_check["new"].seeds) + len(base_check["old"].seeds)),
        "n_vcheck_new": int(len(base_check["new"].seeds)),
        "n_vcheck_old": int(len(base_check["old"].seeds)),
        "vcheck_seeds_sha256": {p: seeds_sha256(base_check[p].seeds) for p in ("new", "old")},
        "n_base_errors_vcheck_new": int(np.count_nonzero(~base_check["new"].success)),
        "n_base_errors_vcheck_old": int(np.count_nonzero(~base_check["old"].success)),
        "base_vcheck_success_sha256": {p: success_sha256(base_check[p].success) for p in ("new", "old")},
        "vselect_seeds_sha256": {p: seeds_sha256(D.SEED_RANGES[SELECT_RANGES[p]].seeds) for p in ("new", "old")},
        "n_vselect_new": D.SEED_RANGES[SELECT_RANGES["new"]].n,
        "n_vselect_old": D.SEED_RANGES[SELECT_RANGES["old"]].n,
        "environment_versions": D.environment_versions(),
        "task_sha256": _text_sha256(task_text),
    }
    return arrays, task, private


def _scan_visible(tokens: list[str], **texts: str) -> None:
    for where, text in texts.items():
        found = scan_for_tokens(text, tokens)
        if found:
            raise E14InstanceError(f"{where} contains {found}; no editor-visible file may")


def _create(root: Path, arrays: Mapping[str, np.ndarray], task: Mapping[str, Any], core: Mapping[str, Any],
            forward_file: Path) -> dict[str, Any]:
    instance_id = _new_instance_id(root)
    staging = root / f".tmp-{instance_id}"
    staging.mkdir()
    source = Path(core["model_source_path"])
    shutil.copyfile(source, staging / "model.npz")
    if params_sha256(load_params(staging / "model.npz")) != core["model_sha256"]:
        raise E14InstanceError(f"{staging}/model.npz: copy hash mismatch")
    shutil.copyfile(source.with_name("model_spec.json"), staging / "model_spec.json")
    if file_sha256(staging / "model_spec.json") != core["model_spec_sha256"]:
        raise E14InstanceError(f"{staging}/model_spec.json: copy hash mismatch")
    shutil.copyfile(forward_file, staging / "forward.py")
    if file_sha256(staging / "forward.py") != core["forward_sha256"]:
        raise E14InstanceError(f"{staging}/forward.py: copy hash mismatch")
    if _save_npz(staging / "evidence.npz", arrays) != core["evidence_content_sha256"]:
        raise E14InstanceError(f"{staging}/evidence.npz: content hash mismatch")
    (staging / "task.json").write_text(json_text(task), encoding="utf-8")
    private = dict(core, instance_id=instance_id, created_at=utc_now(), numpy_version=np.__version__)
    (staging / "private.json").write_text(json_text(private), encoding="utf-8")
    with no_bytecode_cache():
        load_instance(staging)
    os.rename(staging, root / instance_id)
    return jsonable(private)


def build_e14_instance(model_id: str) -> dict[str, Any]:
    """Build (or reuse) the policyskill-v1 instance of the gated base ``model_id``; returns its index row and status.

    An active instance of the same model id is reused when a rebuild gives the same evidence, task and inputs, and
    refused otherwise."""
    from weightcode.models.e14_export import load_store_forward, read_models_index

    rows = [row for row in read_models_index() if row["model_id"] == model_id]
    if len(rows) != 1:
        raise E14InstanceError(f"{model_id} is not one row of the E14 models index")
    forward, forward_sha = load_store_forward()
    tokens = e14_leak_tokens()
    forward_file = D.store_dir() / "forward.py"
    _scan_visible(tokens, **{"forward.py": forward_file.read_text(encoding="utf-8")})
    vcheck = write_vcheck_file()
    root = e14_root()
    with _exclusive_lock(root):
        index = read_e14_instance_index()
        arrays, task, core = _plan(rows[0], forward, forward_sha, tokens, vcheck)
        existing = [r for r in index["instances"] if r["model_id"] == model_id]
        if existing:
            (match,) = existing
            private = read_json(root / match["instance_id"] / "private.json")
            stable = ("evidence_content_sha256", "task_sha256", "model_sha256", "forward_sha256", "model_spec_sha256",
                      "vcheck_content_sha256", "base_vcheck_success_sha256", "replay_train_rows")
            changed = [key for key in stable if private.get(key) != core.get(key)]
            if changed:
                raise E14InstanceError(f"{match['instance_id']}: differs from a rebuild in {changed}; nothing "
                                       "overwritten")
            return {**match, "status": "reused"}
        private = _create(root, arrays, task, core, forward_file)
        index["instances"].append(_index_row(private))
        write_json_atomic(root / "index.json", index)
        with open(root / CHANGES_LOG_NAME, "a", encoding="utf-8") as log:
            log.write(json.dumps({"change": "created", **_index_row(private)}, sort_keys=True) + "\n")
    return {**_index_row(private), "status": "created"}


def supersede_e14_instance(instance_id: str, reason: str) -> dict[str, Any]:
    """Move an active E14 instance to ``superseded`` (its files stay); ``check_e14_current`` then refuses it."""
    if not isinstance(reason, str) or not reason.strip():
        raise E14InstanceError("superseding an instance needs a reason")
    root = e14_root()
    with _exclusive_lock(root):
        index = read_e14_instance_index()
        hits = [row for row in index["instances"] if row["instance_id"] == instance_id]
        if len(hits) != 1:
            raise E14InstanceError(f"{instance_id} is not an active E14 instance")
        moved = {**hits[0], "superseded_at": utc_now(), "reason": reason}
        index["instances"] = [row for row in index["instances"] if row["instance_id"] != instance_id]
        index["superseded"].append(moved)
        write_json_atomic(root / "index.json", index)
        with open(root / CHANGES_LOG_NAME, "a", encoding="utf-8") as log:
            log.write(json.dumps({"change": "superseded", **moved}, sort_keys=True) + "\n")
    return moved


# ---------------------------------------------------------------------------------------------------------------
# Running an instance: stale check, oracle, readout


def check_e14_current(instance_dir: str | Path) -> dict[str, Any]:
    """Refuse an instance that is not active or whose files the current code would not write; else return its task.

    Checks the directory name, the index, the rule, the task.json bytes against private.json, the task against a fresh
    ``build_e14_task`` and the hashes of model.npz, model_spec.json, forward.py and evidence.npz."""
    root = Path(instance_dir)
    private = read_json(root / "private.json")
    instance_id = str(private.get("instance_id"))
    if root.name != instance_id:
        raise StaleE14Instance(f"{root}: directory name differs from private.json instance_id {instance_id!r}")
    index = read_e14_instance_index()
    if instance_id not in {row["instance_id"] for row in index["instances"]}:
        status = "superseded" if instance_id in {row["instance_id"] for row in index["superseded"]} else "not indexed"
        raise StaleE14Instance(f"E14 instance {instance_id} is {status}; only active instances are run")
    if private.get("evidence_rule") != POLICYSKILL_RULE:
        raise StaleE14Instance(f"{instance_id} follows {private.get('evidence_rule')!r}, not {POLICYSKILL_RULE}")
    task_text = (root / "task.json").read_text(encoding="utf-8")
    if _text_sha256(task_text) != private.get("task_sha256"):
        raise StaleE14Instance(f"{root / 'task.json'}: bytes do not match private.json task_sha256")
    task = json.loads(task_text)
    expected = json.loads(json_text(build_e14_task(int(private["n_new_rows"]))))
    if task != expected:
        changed = sorted(key for key in set(task) | set(expected) if task.get(key) != expected.get(key))
        raise StaleE14Instance(f"{instance_id}: task.json differs from what the current code writes in {changed}")
    if file_sha256(root / "model_spec.json") != private.get("model_spec_sha256"):
        raise StaleE14Instance(f"{root / 'model_spec.json'}: hash differs from private.json")
    instance = load_instance(root)
    if evidence_content_sha256(instance.evidence) != private.get("evidence_content_sha256"):
        raise StaleE14Instance(f"{root / 'evidence.npz'}: content hash differs from private.json")
    return task


def check_seed_parts(private: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The new and old V_check seeds from the code's ranges, checked against the instance record."""
    arrays = vcheck_arrays()
    if params_sha256(arrays) != private["vcheck_content_sha256"]:
        raise E14InstanceError("the code's V_check seeds differ from the instance record")
    parts = check_seeds(arrays)
    if {p: seeds_sha256(s) for p, s in parts.items()} != dict(private["vcheck_seeds_sha256"]):
        raise E14InstanceError("the V_check parts differ from the instance record")
    return parts


def make_e14_oracle(instance: Instance | str, budget: int = QUERY_BUDGET) -> RolloutJointOracle:
    """Joint counted rollout oracle of an E14 instance on V_check (``RolloutJointOracle``), its base's episodes checked
    against the instance record."""
    resolved = load_e14_instance(instance)
    private = resolved.private
    parts = check_seed_parts(private)
    return RolloutJointOracle(parts["new"], parts["old"], resolved.forward, resolved.params, budget,
                              dict(private["base_vcheck_success_sha256"]))


def _summary(base: D.RolloutResult, child: D.RolloutResult) -> dict[str, Any]:
    b, c = base.success, child.success
    out: dict[str, Any] = {
        "n": int(len(b)), "success_base": float(b.mean()), "success_child": float(c.mean()),
        "delta": float(c.mean() - b.mean()), "fixed": int(np.count_nonzero(~b & c)),
        "broken": int(np.count_nonzero(b & ~c)),
        "mean_length_base": float(base.length.mean()), "mean_length_child": float(child.length.mean()),
        "mean_length_child_success": float(child.length[c].mean()) if c.any() else None,
        "episodes_with_pickup_base": int(np.count_nonzero(base.n_pickups)),
        "episodes_with_pickup_child": int(np.count_nonzero(child.n_pickups)),
        "success_sha256_child": success_sha256(c),
    }
    out["points"] = 100.0 * out["delta"]
    for key, values, names in (("by_article", child.article, D.ARTICLES), ("by_type", child.obj_type, D.OLD_TYPES)):
        if not np.array_equal(values, base.article if key == "by_article" else base.obj_type):
            raise E14InstanceError("base and child episodes differ in their instructions")
        out[key] = {name: {"n": int((values == i).sum()),
                           "success_base": float(b[values == i].mean()) if (values == i).any() else None,
                           "success_child": float(c[values == i].mean()) if (values == i).any() else None}
                    for i, name in enumerate(names)}
    return out


def demonstrated_descriptors(evidence: Mapping[str, np.ndarray]) -> set[tuple[int, int]]:
    """The (color index, object type index into ``weightcode.data.e14.OLD_TYPES``) pairs of the instance's new
    demonstrations, read from their instruction words."""
    _, instr = D.split_rows(np.asarray(evidence["x_new"]))
    pairs = set()
    for color_id, type_id in {(int(c), int(t)) for c, t in instr[:, 3:5]}:
        pairs.add((D.COLORS.index(D.VOCAB[color_id]), D.OLD_TYPES.index(D.VOCAB[type_id])))
    return pairs


def _seen_split(base: D.RolloutResult, child: D.RolloutResult, seen_pairs: set[tuple[int, int]]) -> dict[str, Any]:
    """Success of base and child split by whether the episode's (color, type) is among ``seen_pairs`` (PREREG E14
    addition, main d744142)."""
    seen = np.array([(int(c), int(t)) in seen_pairs for c, t in zip(child.obj_color, child.obj_type)], dtype=bool)
    if not (np.array_equal(base.obj_color, child.obj_color) and np.array_equal(base.obj_type, child.obj_type)):
        raise E14InstanceError("base and child episodes differ in their instructions")
    out = {}
    for name, rows in (("demonstrated", seen), ("not_demonstrated", ~seen)):
        out[name] = {"n": int(rows.sum()),
                     "success_base": float(base.success[rows].mean()) if rows.any() else None,
                     "success_child": float(child.success[rows].mean()) if rows.any() else None}
    return out


_DIRECTION_STEP = ((1, 0), (0, 1), (-1, 0), (0, -1))
"""Cell in front of the agent for direction 0 east, 1 south, 2 west, 3 north (model_spec.json input.grid)."""


def wrapper_logits_fn(forward: ForwardFn, params: Params) -> D.LogitsFn:
    """PREREG E14's hand-written wrapper (main d744142), beside the bar and not a weight program: the base policy acts
    as on the old mission, and when the cell in front of the agent holds an object of the instruction's color and type
    the action is pickup. Returns logits whose argmax is that action (the base's logits, with pickup raised above the
    maximum on a trigger row)."""
    base = logits_fn(forward, params)
    agent = D.OBJECTS.index("agent")

    def run(x: np.ndarray) -> np.ndarray:
        out = np.array(base(x), dtype=np.float32, copy=True)
        grid = x[:, :D.OBS_LEN].reshape(len(x), D.GRID, D.GRID, D.N_CHANNELS)
        instr = x[:, D.OBS_LEN:]
        for i in range(len(x)):
            ax, ay = np.argwhere(grid[i, :, :, 0] == agent)[0]
            dx, dy = _DIRECTION_STEP[int(grid[i, ax, ay, 2])]
            fx, fy = int(ax) + dx, int(ay) + dy
            if not (0 <= fx < D.GRID and 0 <= fy < D.GRID):
                continue
            want_color = D.COLORS.index(D.VOCAB[int(instr[i, 3])])
            want_type = D.OBJECTS.index(D.VOCAB[int(instr[i, 4])])
            if int(grid[i, fx, fy, 0]) == want_type and int(grid[i, fx, fy, 1]) == want_color:
                out[i, D.PICKUP] = out[i].max() + 1.0
        return out
    return run


def e14_wrapper_readout(instance: Instance | str) -> dict[str, Any]:
    """The hand-written wrapper (``wrapper_logits_fn`` on the base) against the base on V_select, new and old, in the
    fields of ``e14_readout``. Not a candidate and not in the bar."""
    resolved = load_e14_instance(instance)
    base = _base_vselect(str(resolved.instance_dir), resolved.params_sha256)
    fn = wrapper_logits_fn(resolved.forward, resolved.params)
    seen_pairs = demonstrated_descriptors(resolved.evidence)
    result: dict[str, Any] = {}
    for part, name in SELECT_RANGES.items():
        child = D.rollouts(fn, part, D.SEED_RANGES[name].seeds)
        result[part] = _summary(base[part], child)
        result[part]["by_demonstrated_descriptor"] = _seen_split(base[part], child, seen_pairs)
    result["gain"] = result["new"]["points"]
    result["clean"] = result["old"]["points"]
    result["joint"] = result["gain"] + result["clean"]
    return result


# ---------------------------------------------------------------------------------------------------------------
# T11 controls (PREREG E14, entry "T11 BabyAI controls"): diagnostics beside the bar. External rules with the final
# program's dead-end handling, the readout of any logits function, per-episode rollout traces and the episode
# categories. ``wrapper_logits_fn`` and ``e14_wrapper_readout`` above stay the wrapper of record.

_FORWARD = D.ACTIONS.index("forward")
_DROP = D.ACTIONS.index("drop")
_EMPTY = D.OBJECTS.index("empty")
_WALL = D.OBJECTS.index("wall")
_AGENT = D.OBJECTS.index("agent")
_OBJECT_IDS = np.array([D.OBJECTS.index(name) for name in D.OLD_TYPES], dtype=np.int64)
_FACEABLE = np.array([_EMPTY, _WALL, *_OBJECT_IDS.tolist()], dtype=np.int64)
_STEP = np.array(_DIRECTION_STEP, dtype=np.int64)
_COLOR_OF_WORD = np.full(len(D.VOCAB), -1, dtype=np.int64)
_TYPE_OF_WORD = np.full(len(D.VOCAB), -1, dtype=np.int64)
for _name in D.COLORS:
    _COLOR_OF_WORD[D.WORD_ID[_name]] = D.COLORS.index(_name)
for _name in D.OLD_TYPES:
    _TYPE_OF_WORD[D.WORD_ID[_name]] = D.OBJECTS.index(_name)
_GOTO_VERBS = {D.WORD_ID["pick"]: D.WORD_ID["go"], D.WORD_ID["up"]: D.WORD_ID["to"]}


@dataclasses.dataclass(frozen=True)
class FacedCells:
    """Per row: the object and color index of the cell in front of the agent, and the instruction's target (object
    index of its type, color index)."""

    obj: np.ndarray
    color: np.ndarray
    want_type: np.ndarray
    want_color: np.ndarray

    @property
    def trigger(self) -> np.ndarray:
        """The faced cell has the instruction's color and type (``wrapper_logits_fn``'s trigger)."""
        return (self.obj == self.want_type) & (self.color == self.want_color)

    @property
    def blocked(self) -> np.ndarray:
        """The faced cell is not empty (a wall or an object): the final program's dead-end detector."""
        return self.obj != _EMPTY

    @property
    def is_object(self) -> np.ndarray:
        return np.isin(self.obj, _OBJECT_IDS)

    @property
    def nontarget_object(self) -> np.ndarray:
        """The faced cell holds an object that is not the target."""
        return self.is_object & ~self.trigger

    @property
    def wrong_target(self) -> np.ndarray:
        """The faced cell holds an object that matches exactly one of the target's color and type."""
        return self.is_object & ((self.obj == self.want_type) != (self.color == self.want_color))


def faced_cells(x: np.ndarray) -> FacedCells:
    """``FacedCells`` of observation rows ``x`` (int64 [N, ROW_LEN]); refuses a row without exactly one agent cell, a
    faced cell outside the grid or a faced cell that is not empty, a wall or a key, ball or box."""
    grid, instr = D.split_rows(np.asarray(x))
    agent = grid[..., 0] == _AGENT
    if not np.all(agent.sum(axis=(1, 2)) == 1):
        raise E14InstanceError("a row does not have exactly one agent cell")
    rows, ax, ay = np.nonzero(agent)  # C order: one hit per row, rows 0..N-1 in order
    direction = grid[rows, ax, ay, 2]
    fx, fy = ax + _STEP[direction, 0], ay + _STEP[direction, 1]
    if not (np.all((fx >= 0) & (fx < D.GRID)) and np.all((fy >= 0) & (fy < D.GRID))):
        raise E14InstanceError("an agent faces a cell outside the grid")
    obj, color = grid[rows, fx, fy, 0], grid[rows, fx, fy, 1]
    if not np.isin(obj, _FACEABLE).all():
        raise E14InstanceError(f"a faced cell holds object index {sorted(set(obj[~np.isin(obj, _FACEABLE)].tolist()))}; "
                               "GoToLocal rooms hold only empty cells, walls, keys, balls and boxes")
    want_type, want_color = _TYPE_OF_WORD[instr[:, 4]], _COLOR_OF_WORD[instr[:, 3]]
    if (want_type < 0).any() or (want_color < 0).any():
        raise E14InstanceError("an instruction does not name a color and a type in positions 3 and 4")
    return FacedCells(obj, color, want_type, want_color)


def goto_rows(x: np.ndarray) -> np.ndarray:
    """A copy of rows ``x`` with the verbs pick and up replaced by go and to (the final program's verb copy, step 1,
    done on the input instead of the embedding rows); rows of the old mission are unchanged."""
    out = np.array(x, dtype=np.int64, copy=True)
    for position in (0, 1):
        column = out[:, D.OBS_LEN + position]
        for new_id, old_id in _GOTO_VERBS.items():
            column[column == new_id] = old_id
    return out


T11_RULES: dict[str, dict[str, bool]] = {
    "wrapper_v2": {"trigger": True, "deadend": True, "goto": False},
    "deadend_only": {"trigger": False, "deadend": True, "goto": False},
    "wrapper_v1_goto": {"trigger": True, "deadend": False, "goto": True},
    "wrapper_v2_goto": {"trigger": True, "deadend": True, "goto": True},
}
"""The external-rule arms of PREREG E14 T11 beside the wrapper of record (``wrapper_logits_fn``)."""


def control_logits_fn(forward: ForwardFn, params: Params, *, trigger: bool, deadend: bool, goto: bool) -> D.LogitsFn:
    """An external rule around the base policy (PREREG E14 T11). ``goto``: the base reads go and to in place of pick and
    up (``goto_rows``). ``deadend``: on a row whose faced cell is not empty, forward's logit is set to the row's minimum
    minus 1, so forward is never the argmax; nothing else changes. ``trigger``: then, on a row whose faced cell has the
    instruction's color and type, pickup is set to the row's maximum plus 1 (``wrapper_logits_fn``'s trigger). The
    rules read the original row."""
    base = logits_fn(forward, params)

    def run(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        out = np.array(base(goto_rows(x) if goto else x), dtype=np.float32, copy=True)
        cells = faced_cells(x)
        if deadend:
            rows = cells.blocked
            out[rows, _FORWARD] = out[rows].min(axis=1) - 1.0
        if trigger:
            rows = cells.trigger
            out[rows, D.PICKUP] = out[rows].max(axis=1) + 1.0
        return out
    return run


def wrapper_v2_logits_fn(forward: ForwardFn, params: Params) -> D.LogitsFn:
    """The wrapper with the final program's dead-end handling: base, the dead-end rule, then the trigger."""
    return control_logits_fn(forward, params, **T11_RULES["wrapper_v2"])


def deadend_only_logits_fn(forward: ForwardFn, params: Params) -> D.LogitsFn:
    """The base with the dead-end rule and no trigger."""
    return control_logits_fn(forward, params, **T11_RULES["deadend_only"])


def e14_fn_readout(instance: Instance | str, fn: D.LogitsFn) -> tuple[dict[str, Any], dict[str, D.RolloutResult]]:
    """Greedy rollouts of the logits function ``fn`` against the instance's base on the 1,000 new and 1,000 old V_select
    episodes, in the fields of ``e14_wrapper_readout`` (``e14_readout`` without ``demonstrated_descriptors``), and the
    child's per-part ``RolloutResult``. New part first, then old."""
    resolved = load_e14_instance(instance)
    for part, name in SELECT_RANGES.items():
        if seeds_sha256(D.SEED_RANGES[name].seeds) != resolved.private["vselect_seeds_sha256"][part]:
            raise E14InstanceError(f"the code's {name} seeds differ from the instance record")
    base = _base_vselect(str(resolved.instance_dir), resolved.params_sha256)
    seen_pairs = demonstrated_descriptors(resolved.evidence)
    result: dict[str, Any] = {}
    children: dict[str, D.RolloutResult] = {}
    for part, name in SELECT_RANGES.items():
        children[part] = D.rollouts(fn, part, D.SEED_RANGES[name].seeds)
        result[part] = _summary(base[part], children[part])
        result[part]["by_demonstrated_descriptor"] = _seen_split(base[part], children[part], seen_pairs)
    result["gain"] = result["new"]["points"]
    result["clean"] = result["old"]["points"]
    result["joint"] = result["gain"] + result["clean"]
    return result, children


class RolloutTrace:
    """A logits function that runs ``fn`` and records, for every row of every call, the faced cell, the instruction's
    target and the argmax action (the action ``weightcode.data.e14.rollouts`` takes). ``steps`` assigns the recorded rows to
    episodes from the rollout's lengths."""

    def __init__(self, fn: D.LogitsFn) -> None:
        self.fn = fn
        self.calls: list[dict[str, np.ndarray]] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(self.fn(x))
        cells = faced_cells(x)
        self.calls.append({"obj": cells.obj, "color": cells.color, "want_type": cells.want_type,
                           "want_color": cells.want_color, "action": np.argmax(out, axis=1)})
        return out

    def steps(self, lengths: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        """The recorded calls cut into consecutive rollout calls with episode lengths ``lengths`` (one array per
        ``rollouts`` call, in call order) and assigned to episodes: per call, arrays [n_episodes, MAX_STEPS] of the
        faced object and color and the action (-1 after an episode's end), and the target per episode. In a rollout
        call, step t runs the episodes with length > t, in seed order."""
        out, position = [], 0
        for length in lengths:
            length = np.asarray(length, dtype=np.int64)
            n = len(length)
            arrays = {key: np.full((n, D.MAX_STEPS), -1, dtype=np.int64) for key in ("obj", "color", "action")}
            want = {"want_type": np.full(n, -1, dtype=np.int64), "want_color": np.full(n, -1, dtype=np.int64)}
            for t in range(int(length.max()) if n else 0):
                if position >= len(self.calls):
                    raise E14InstanceError("the trace holds fewer calls than the rollout lengths need")
                call = self.calls[position]
                position += 1
                running = np.flatnonzero(length > t)
                if len(call["action"]) != len(running):
                    raise E14InstanceError(f"step {t}: {len(call['action'])} rows recorded, {len(running)} episodes "
                                           "running")
                for key in arrays:
                    arrays[key][running, t] = call[key]
                for key in want:
                    if t == 0:
                        want[key][running] = call[key]
                    elif not np.array_equal(want[key][running], call[key]):
                        raise E14InstanceError(f"step {t}: an episode's instruction changed")
            out.append({**arrays, **want, "length": length})
        if position != len(self.calls):
            raise E14InstanceError(f"the trace holds {len(self.calls)} calls, the rollout lengths account for {position}")
        return out


def trajectory_features(steps: Mapping[str, np.ndarray], success: np.ndarray) -> dict[str, np.ndarray]:
    """Per-episode facts of one traced rollout call (``RolloutTrace.steps``) with its per-episode ``success``.

    ``faced_target``: some observed row faces the target; ``before_*`` over the observed steps before the first such
    row (all steps when there is none): ``before_blocked`` faces an object that is not the target, ``before_wrong``
    faces an object matching exactly one of the target's color and type, ``before_stuck`` takes forward while the
    faced cell is not empty; ``no_target``: no observed row faces the target and the episode did not succeed (on the
    old mission the target is faced only after the last action, which ends the episode with success). Over the whole
    episode: ``forward_blocked_steps``, the steps that take forward into a cell that is not empty;
    ``carried_nontarget``, the first pickup facing an object picks up an object that is not the target; ``drops``, the
    drop actions."""
    obj, color, action = (np.asarray(steps[key]) for key in ("obj", "color", "action"))
    want_type, want_color = np.asarray(steps["want_type"])[:, None], np.asarray(steps["want_color"])[:, None]
    length = np.asarray(steps["length"])
    n, width = obj.shape
    t = np.arange(width)[None, :]
    valid = t < length[:, None]
    is_object = np.isin(obj, _OBJECT_IDS) & valid
    target = is_object & (obj == want_type) & (color == want_color)
    faced_target = target.any(axis=1)
    first = np.where(faced_target, target.argmax(axis=1), length)
    before = valid & (t < first[:, None])
    blocked_cell = valid & (obj != _EMPTY)
    wrong = is_object & ((obj == want_type) != (color == want_color))
    pick_on_object = is_object & (action == D.PICKUP)
    has_pick = pick_on_object.any(axis=1)
    first_pick = pick_on_object.argmax(axis=1)
    rows = np.arange(n)
    return {
        "faced_target": faced_target,
        "before_blocked": (before & is_object & ~target).any(axis=1),
        "before_wrong": (before & wrong).any(axis=1),
        "before_stuck": (before & blocked_cell & (action == _FORWARD)).any(axis=1),
        "no_target": ~faced_target & ~np.asarray(success, dtype=bool),
        "forward_blocked_steps": (blocked_cell & (action == _FORWARD)).sum(axis=1),
        "carried_nontarget": has_pick & ~target[rows, first_pick],
        "drops": (valid & (action == _DROP)).sum(axis=1),
    }


T11_CATEGORIES = ("blocked", "wrong_target", "stuck", "no_target", "plain")
"""PREREG E14 T11 episode categories, from the base policy's own trajectory (``episode_categories``)."""


def episode_categories(base_features: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The T11 categories of one mission's episodes from the base's ``trajectory_features``: fixed boolean masks."""
    blocked = np.asarray(base_features["before_blocked"], dtype=bool)
    stuck = np.asarray(base_features["before_stuck"], dtype=bool)
    no_target = np.asarray(base_features["no_target"], dtype=bool)
    return {"blocked": blocked, "wrong_target": np.asarray(base_features["before_wrong"], dtype=bool), "stuck": stuck,
            "no_target": no_target, "plain": ~(blocked | stuck | no_target)}


@functools.lru_cache(maxsize=4)
def _base_vselect(instance_dir: str, params_sha: str) -> dict[str, D.RolloutResult]:
    resolved = load_e14_instance(Path(instance_dir).name)
    if resolved.params_sha256 != params_sha:
        raise E14InstanceError("the instance's model changed")
    fn = logits_fn(resolved.forward, resolved.params)
    out = {part: D.rollouts(fn, part, D.SEED_RANGES[name].seeds) for part, name in SELECT_RANGES.items()}
    expected = dict(resolved.private["base_vselect_success_sha256"])
    got = {part: success_sha256(result.success) for part, result in out.items()}
    if got != expected:
        raise E14InstanceError(f"the base's V_select episodes differ from its import readout: {got} != {expected}")
    return out


def e14_readout(instance: Instance | str, child: Params) -> dict[str, Any]:
    """Greedy rollouts of ``child`` against the instance's base on the 1,000 new and 1,000 old V_select episodes
    (PREREG E14): per part n, success of base and child, delta, fixed, broken, mean lengths, pickups, success by
    article and by object type, and success split by whether the episode's (color, type) is among the demonstrations'
    (``by_demonstrated_descriptor``, main d744142); ``gain`` = 100 x delta new, ``clean`` = 100 x delta old,
    ``joint`` = gain + clean. The base's episodes are read once per process and checked against its import readout."""
    resolved = load_e14_instance(instance)
    for part, name in SELECT_RANGES.items():
        if seeds_sha256(D.SEED_RANGES[name].seeds) != resolved.private["vselect_seeds_sha256"][part]:
            raise E14InstanceError(f"the code's {name} seeds differ from the instance record")
    base = _base_vselect(str(resolved.instance_dir), resolved.params_sha256)
    fn = logits_fn(resolved.forward, child)
    seen_pairs = demonstrated_descriptors(resolved.evidence)
    result: dict[str, Any] = {"demonstrated_descriptors": sorted([D.COLORS[c], D.OLD_TYPES[t]] for c, t in seen_pairs)}
    for part, name in SELECT_RANGES.items():
        child_result = D.rollouts(fn, part, D.SEED_RANGES[name].seeds)
        result[part] = _summary(base[part], child_result)
        result[part]["by_demonstrated_descriptor"] = _seen_split(base[part], child_result, seen_pairs)
    result["gain"] = result["new"]["points"]
    result["clean"] = result["old"]["points"]
    result["joint"] = result["gain"] + result["clean"]
    return result


__all__ = [
    "CHECK_RANGES", "E14InstanceError", "E14_MAX_CHANGED_FRAC", "EDITOR_VISIBLE_FILES", "EVIDENCE_KEYS", "GATE",
    "HEADROOM", "INSTANCE_ID_RE", "K_NEW", "LEAK_WORDS", "N_REPLAY", "POLICYSKILL_RULE", "PRIVATE_SCHEMA",
    "REPLAY_RULE", "REPLAY_STREAM", "ROOT_NAME", "RolloutJointOracle", "SELECT_RANGES", "StaleE14Instance",
    "TASK_LIMITS", "TASK_SCHEMA", "VCHECK_FILE", "active_e14_ids", "build_e14_instance", "build_e14_task",
    "check_e14_current", "check_seed_parts", "check_seeds", "demonstrated_descriptors", "draw_replay_rows",
    "draw_split_ids", "e14_instance_dir", "e14_wrapper_readout", "wrapper_logits_fn",
    "e14_leak_tokens", "e14_program_limits", "e14_readout", "e14_root", "evidence_content_sha256", "instance_seed",
    "load_e14_instance", "logits_fn", "make_e14_oracle", "policy_logits", "read_e14_instance_index", "seeds_sha256",
    "success_sha256", "supersede_e14_instance", "vcheck_arrays", "vcheck_file", "write_vcheck_file",
    "FacedCells", "RolloutTrace", "T11_CATEGORIES", "T11_RULES", "control_logits_fn", "deadend_only_logits_fn",
    "e14_fn_readout", "episode_categories", "faced_cells", "goto_rows", "trajectory_features", "wrapper_v2_logits_fn",
]
