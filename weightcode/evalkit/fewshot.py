"""Rule fewshot-v1 (PREREG E2): instances that ask an edit to add classes a student never trained on.

A fewshot student (``weightcode.models.export.fewshot_models_dir``) was trained on the train-split rows of its train
classes (0-4) with every output kept, so the outputs of its new classes (5-9) never saw an image of their class. There
is one instance per (dataset, student). The layout is the evidence-v1 layout, so
``weightcode.baselines.instance_io.load_instance`` loads it unchanged. The instances live under their own root with their
own index, so the evidence-v1 store, its rule check and its builders never see them::

    $WEIGHTCODE_PRIVATE/fewshot_instances/
        index.json                       {"schema", "instances": [active rows], "superseded": [rows]}
        changes.jsonl                    one line per created instance
        f<8 hex>/
            model.npz, model_spec.json   byte copies of the student's files
            forward.py                   byte copy of the dataset forward (it reads the architecture from the names)
            evidence.npz                 the evidence, keys EVIDENCE_KEYS
            task.json                    schema weightcode.task.fewshot.v1
            private.json                 schema weightcode.instance_private.fewshot.v1, never editor-visible

Evidence of rule fewshot-v1:

- new part: N_PER_NEW_CLASS rows of each new class, in class order, drawn without replacement from the clean
  V_evidence rows of that class with ``default_rng(fewshot_seed(params_sha256, dataset))``;
- old part: the replay of rule fewshot-replay-v1, N_REPLAY rows drawn uniformly without replacement from the clean
  train rows of the train classes with ``default_rng([SPLIT_SEED, REPLAY_STREAM])``. It is one draw per (dataset,
  train classes), shared by every student of it, and the rows are images those students were trained on;
- pred_*: the student's numpy-forward argmax on those rows;
- split ids: N_NEW + N_REPLAY unique ``e`` + 12 hex ids from the instance generator after the row draw, new first.

Check (``make_fewshot_oracle``): the clean V_check split by class (``fewshot_check_parts``), S_new over the rows of
the new classes, S_old over the rest and S = S_new + S_old (``JointPatchCheckOracle``). Readout (``fewshot_readout``):
10-way accuracy on the clean V_select over all rows, the new-class rows and the old-class rows, child against base,
with per-class changes and the row counts.

Limits: task.json states the default ``weightcode.patch.Limits`` with the head (the last linear layer of model_spec.json)
exempt from max_rel_frob, because the head rows of the new classes are new capacity (PREREG E2). An update program
and every E2 reference are held to ``fewshot_program_limits``: the same with the edit-size limit off and at most 20%
of the entries changed, counted over every tensor.
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
from weightcode.data.registry import SPLIT_SEED

# One seed scheme for every evidence rule: the evidence-v0 and evidence-v1 seeds come from the same function.
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
from weightcode.evalkit.patch_check import JointPatchCheckOracle, predict
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
from weightcode.patch import OP_NAMES, Limits
from weightcode.patch import SCHEMA as PATCH_SCHEMA
from weightcode.patch.limits import SIZE_LIMIT_FIELDS

FEWSHOT_RULE = "fewshot-v1"
REPLAY_RULE = "fewshot-replay-v1"
REPLAY_STREAM = 9002
"""Generator stream of the fewshot replay; replay-v1 of PREREG E1 uses 9001."""
N_PER_NEW_CLASS = 10
N_REPLAY = 64
DEFAULT_TRAIN_CLASSES: tuple[int, ...] = (0, 1, 2, 3, 4)
TASK_SCHEMA = "weightcode.task.fewshot.v1"
PRIVATE_SCHEMA = "weightcode.instance_private.fewshot.v1"
INDEX_SCHEMA = "weightcode.fewshot_instances_index.v1"
CHANGES_LOG_NAME = "changes.jsonl"
INSTANCE_ID_RE = re.compile(r"^f[0-9a-f]{8}$")
EVIDENCE_KEYS: tuple[str, ...] = (
    "x_new", "y_new", "pred_new", "split_ids_new", "x_old", "y_old", "pred_old", "split_ids_old",
)
EDITOR_VISIBLE_FILES: tuple[str, ...] = ("model.npz", "model_spec.json", "forward.py", "evidence.npz", "task.json")
CHECK_PART_KEYS: tuple[str, ...] = ("x_new", "y_new", "x_old", "y_old")
SPLITS: tuple[str, ...] = ("train", "V_evidence", "V_check", "V_select")
EVALUATION_BATCH = 1024
"""Forward batch size of the build, the oracle and the readout."""
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class FewshotError(RuntimeError):
    """A fewshot instance, its store or its inputs were refused; nothing is overwritten."""


class StaleFewshotInstance(FewshotError):
    """An instance that is not active or whose files the current code would not write; it is not run."""


def fewshot_root() -> Path:
    """``$WEIGHTCODE_PRIVATE/fewshot_instances`` (created if missing)."""
    root = private_dir() / "fewshot_instances"
    root.mkdir(exist_ok=True)
    return root


def fewshot_seed(params_sha256_hex: str, dataset: str) -> int:
    """Seed of the fewshot-v1 draw of one student."""
    if not _SHA_RE.fullmatch(params_sha256_hex):
        raise ValueError(f"not a sha256 hex digest: {params_sha256_hex!r}")
    return _seed_from_text(f"{params_sha256_hex}|{dataset}|{FEWSHOT_RULE}")


def head_tensors(spec: Mapping[str, Any]) -> list[str]:
    """Weight and bias names of the head: the last layer of model_spec.json, which must be linear."""
    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers or layers[-1].get("op") != "linear":
        raise FewshotError("model_spec.json must end in a linear layer (the head)")
    return [str(layers[-1]["weight"]), str(layers[-1]["bias"])]


def fewshot_limits(spec: Mapping[str, Any]) -> Limits:
    """The default limits with the head exempt from max_rel_frob (PREREG E2)."""
    return Limits(rel_frob_exempt=tuple(head_tensors(spec)))


FEWSHOT_MAX_CHANGED_FRAC = 0.20
"""PREREG E2: an update may change at most 20% of the parameter entries. It is the changed-entry cap of every program
arm, ``weightcode.harness.arms.PROGRAM_MAX_CHANGED_FRAC``; a test holds the two equal. It is not imported from there, so
evalkit and the fitting references do not load the harness package."""

_PROGRAM_SIZE_FIELDS = ("max_rel_frob", "max_changed_frac")
"""The size limits a program's deltas are checked against, as in ``weightcode.harness.arms.PROGRAM_LIMIT_FIELDS``."""


def fewshot_program_limits(task_limits_json: Mapping[str, Any]) -> Limits:
    """The limits that bind an E2 update and every E2 reference: task.json's limits, the edit-size limit off and at
    most ``FEWSHOT_MAX_CHANGED_FRAC`` of the entries changed, with the head still exempt from max_rel_frob.

    They equal ``weightcode.harness.arms.applied_limits(task_limits_json, "LEARN")``, the limits a LEARN session applies to
    the same task (a test checks it). Dense deltas are counted per changed entry; M2's edit size is counted by
    information and has no per-entry equivalent (PREREG E1, LEARN limits).
    """
    if not isinstance(task_limits_json, Mapping):
        raise FewshotError(f"task.json limits must be an object, got {type(task_limits_json).__name__}")
    limits = Limits(**dict(task_limits_json))
    disabled = {name: None for name in SIZE_LIMIT_FIELDS if name not in _PROGRAM_SIZE_FIELDS}
    return dataclasses.replace(limits, **disabled, max_changed_frac=FEWSHOT_MAX_CHANGED_FRAC)


def new_classes_of(n_classes: int, train_classes: Sequence[int]) -> list[int]:
    """The labels a student with ``train_classes`` never trained on, in label order."""
    train = [int(c) for c in train_classes]
    if not train or train != sorted(set(train)) or train[0] < 0 or train[-1] >= n_classes or len(train) == n_classes:
        raise FewshotError(f"train classes must be a proper sorted subset of 0..{n_classes - 1}, got {train}")
    return [c for c in range(n_classes) if c not in train]


def fewshot_check_parts(arrays: Mapping[str, np.ndarray], new_classes: Sequence[int]) -> dict[str, np.ndarray]:
    """The check split cut by class: rows whose label is in ``new_classes`` (new part) and the rest (old part).

    Pure: returns x_new, y_new, x_old, y_old, each part in the split's row order. The one place where the class split
    of PREREG E2 is defined; the oracle of a session ledger builds JointPatchCheckOracle from these arrays.
    """
    new = [int(c) for c in new_classes]
    if not new:
        raise FewshotError("new_classes is empty")
    x, y = np.asarray(arrays["x"]), np.asarray(arrays["y"])
    if y.ndim != 1 or len(x) != len(y):
        raise FewshotError(f"check arrays x {x.shape} and y {y.shape} do not align")
    in_new = np.isin(y, new)
    if not in_new.any() or in_new.all():
        raise FewshotError(f"the check split has {int(in_new.sum())} of {len(y)} rows in the new classes {new}")
    return {
        "x_new": np.ascontiguousarray(x[in_new]),
        "y_new": y[in_new].astype(np.int64),
        "x_old": np.ascontiguousarray(x[~in_new]),
        "y_old": y[~in_new].astype(np.int64),
    }


def draw_replay_rows(y_train: np.ndarray, train_classes: Sequence[int]) -> np.ndarray:
    """Rule fewshot-replay-v1: N_REPLAY train rows of the train classes, uniform, without replacement."""
    pool = np.flatnonzero(np.isin(np.asarray(y_train), [int(c) for c in train_classes]))
    if pool.size < N_REPLAY:
        raise FewshotError(f"only {pool.size} train rows in the train classes; the replay needs {N_REPLAY}")
    rng = np.random.default_rng([SPLIT_SEED, REPLAY_STREAM])
    return rng.choice(pool, size=N_REPLAY, replace=False).astype(np.int64)


def draw_new_rows(
    y_evidence: np.ndarray, new_classes: Sequence[int], seed: int, n_old: int = N_REPLAY
) -> tuple[np.ndarray, list[str], list[str]]:
    """Rule fewshot-v1: N_PER_NEW_CLASS V_evidence rows per new class (class order), then the opaque ids.

    Returns the rows and the split ids of the new part and of the old part (``n_old`` ids), all from one generator.
    """
    y = np.asarray(y_evidence)
    rng = np.random.default_rng(seed)
    rows = []
    for label in new_classes:
        pool = np.flatnonzero(y == int(label))
        if pool.size < N_PER_NEW_CLASS:
            raise FewshotError(f"V_evidence has {pool.size} rows of class {label}; the rule needs {N_PER_NEW_CLASS}")
        rows.append(rng.choice(pool, size=N_PER_NEW_CLASS, replace=False))
    new_rows = np.concatenate(rows).astype(np.int64)
    codes = rng.integers(0, 16**SPLIT_ID_HEX_DIGITS, size=new_rows.size + n_old, dtype=np.int64)
    if np.unique(codes).size != codes.size:
        raise FewshotError(f"split id collision for seed {seed}; the id width must grow (rule change)")
    ids = [f"e{int(code):0{SPLIT_ID_HEX_DIGITS}x}" for code in codes]
    return new_rows, ids[: new_rows.size], ids[new_rows.size :]


def build_fewshot_task(
    names: Sequence[str], train_classes: Sequence[int], new_classes: Sequence[int], limits: Limits
) -> dict[str, Any]:
    """Editor-visible task record of rule fewshot-v1 (schema weightcode.task.fewshot.v1)."""
    trained = ", ".join(f"{c} ({names[c]})" for c in train_classes)
    added = ", ".join(f"{c} ({names[c]})" for c in new_classes)
    stated = patch_limits(limits)
    return {
        "schema": TASK_SCHEMA,
        "statement": (
            f"This image classifier has {len(names)} outputs, one per class. It was trained only on images of the "
            f"classes {trained}. The outputs of the classes {added} never saw a training image of their class, so "
            f"the classifier does not predict them. The evidence file holds {N_PER_NEW_CLASS} labelled images of "
            f"each of these new classes and {N_REPLAY} labelled images of the classes it was trained on, with the "
            "classifier's predictions. Change the classifier's weights directly, without training it, so that it "
            "also recognises the new classes on new images, while the images of the trained classes that it "
            "classifies correctly stay correct."
        ),
        "classes": list(names),
        "train_classes": [int(c) for c in train_classes],
        "new_classes": [int(c) for c in new_classes],
        "evidence": {
            "new": {
                "keys": ["x_new", "y_new", "pred_new"],
                "count": N_PER_NEW_CLASS * len(new_classes),
                "per_class": N_PER_NEW_CLASS,
                "description": "labelled images of the new classes, in class order, drawn at random from held-out "
                "data; the classifier has never seen these classes",
            },
            "old": {
                "keys": ["x_old", "y_old", "pred_old"],
                "count": N_REPLAY,
                "description": "labelled images of the classes the classifier was trained on, a fixed random draw "
                "from its training images",
            },
            "ids": {"keys": ["split_ids_new", "split_ids_old"], "description": "one opaque id per example"},
        },
        "patch_check": {
            "budget_queries": QUERY_BUDGET,
            "readout": "each query applies one patch and returns aggregate numbers on a held-out check split, each "
            "against the unedited classifier, for two parts, the images of the new classes and the images of the "
            "trained classes: accuracy, fixed (images the unedited classifier got wrong that the patched one gets "
            "right), broken (images the unedited classifier got right that the patched one gets wrong), the "
            "changed-parameter count, S_new and S_old (fixed - broken on each part) and S = S_new + S_old; no "
            "per-image information",
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
        raise FewshotError(f"evidence keys {sorted(arrays)} != {sorted(EVIDENCE_KEYS)}")
    return params_sha256(arrays)


def _save_evidence(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    expected = _evidence_content_sha256(arrays)
    np.savez(path, **{key: np.ascontiguousarray(arrays[key]) for key in EVIDENCE_KEYS})
    with np.load(path, allow_pickle=False) as archive:
        reloaded = {key: archive[key] for key in archive.files}
    if _evidence_content_sha256(reloaded) != expected:
        raise FewshotError(f"{path}: evidence content hash changed on reload")
    return expected


# ---------------------------------------------------------------------------------------------------------------
# Store: index.json, changes.jsonl, lock


@contextmanager
def _exclusive_lock(root: Path) -> Iterator[None]:
    descriptor = os.open(root / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def read_fewshot_instance_index() -> dict[str, Any]:
    """The fewshot instance index; an empty one before the first build."""
    path = fewshot_root() / "index.json"
    if not path.exists():
        return {"schema": INDEX_SCHEMA, "instances": [], "superseded": []}
    index = read_json(path)
    if not isinstance(index, dict) or index.get("schema") != INDEX_SCHEMA:
        raise FewshotError(f"{path} is not a {INDEX_SCHEMA} index")
    return index


def active_fewshot_ids(dataset: str | None = None) -> list[str]:
    """Ids of the active fewshot instances, optionally of one dataset, in index order."""
    rows = read_fewshot_instance_index()["instances"]
    return [str(row["instance_id"]) for row in rows if dataset is None or row["dataset"] == dataset]


def fewshot_instance_dir(instance_id: str) -> Path:
    """Directory of one fewshot instance; refuses an id of another format."""
    if not INSTANCE_ID_RE.fullmatch(str(instance_id)):
        raise FewshotError(f"{instance_id!r} is not a fewshot instance id (f + 8 hex)")
    return fewshot_root() / str(instance_id)


def load_fewshot_instance(instance: Instance | str) -> Instance:
    """An Instance, or the fewshot instance with this id loaded and hash-checked (``load_instance``)."""
    return instance if isinstance(instance, Instance) else load_instance(fewshot_instance_dir(instance))


def _index_row(private: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("instance_id", "dataset", "arch", "seed", "model_id", "split_role", "train_classes", "new_classes",
            "evidence_rule", "created_at")
    return {key: private[key] for key in keys}


def _new_instance_id(root: Path) -> str:
    while True:
        instance_id = "f" + secrets.token_hex(4)
        if not (root / instance_id).exists() and not (root / f".tmp-{instance_id}").exists():
            return instance_id


# ---------------------------------------------------------------------------------------------------------------
# Build


def _load_splits(dataset: str, meta: Mapping[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    data, record = {}, {}
    for split in SPLITS:
        path = split_path(dataset, split)
        expected = meta["splits"][split]["sha256"]
        data[split] = load_checked_split(path, expected)
        record[split] = {"path": str(path), "sha256": file_sha256(path), "content_sha256": expected}
    return data, record


def _scan_visible(tokens: Sequence[str], **texts: str) -> None:
    for where, text in texts.items():
        found = scan_for_tokens(text, tokens)
        if found:
            raise FewshotError(f"{where} contains {found}; no editor-visible file may")


def _plan(
    dataset: str,
    row: Mapping[str, Any],
    names: Sequence[str],
    data: Mapping[str, Mapping[str, np.ndarray]],
    split_record: Mapping[str, Mapping[str, Any]],
    replay_rows: np.ndarray,
    forward: Any,
    forward_sha: str,
    tokens: Sequence[str],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Evidence arrays, task and private record (without id and time) of one student's instance."""
    from weightcode.models.export import fewshot_models_dir  # imports torch; kept out of module import (CODE_AUDIT section 4)

    train_classes = [int(c) for c in row["train_classes"]]
    new_classes = new_classes_of(len(names), train_classes)
    source = fewshot_models_dir(dataset) / str(row["model_id"])
    params = load_params(source / "model.npz")
    digest = params_sha256(params)
    if digest != row["params_sha256"]:
        raise FewshotError(f"{source}/model.npz hash {digest} != fewshot index {row['params_sha256']}")
    model_private = read_json(source / "private.json")
    if model_private.get("train_classes") != train_classes or model_private.get("role") != "dev":
        raise FewshotError(f"{source}/private.json does not describe a dev student of classes {train_classes}")
    spec_text = (source / "model_spec.json").read_text(encoding="utf-8")
    spec = json.loads(spec_text)
    if spec.get("classes") != list(names):
        raise FewshotError(f"{source}/model_spec.json class list differs from dataset.json")
    limits = fewshot_limits(spec)
    missing = sorted(set(limits.rel_frob_exempt) - set(params))
    if missing:
        raise FewshotError(f"head tensors {missing} are not in {source}/model.npz")

    v_ev, train = data["V_evidence"], data["train"]
    new_rows, ids_new, ids_old = draw_new_rows(v_ev["y"], new_classes, fewshot_seed(digest, dataset), len(replay_rows))
    x_new, y_new = np.ascontiguousarray(v_ev["x"][new_rows]), v_ev["y"][new_rows].astype(np.int64)
    x_old, y_old = np.ascontiguousarray(train["x"][replay_rows]), train["y"][replay_rows].astype(np.int64)
    arrays = {
        "x_new": x_new,
        "y_new": y_new,
        "pred_new": predict(forward, params, x_new, batch_size).astype(np.int64),
        "split_ids_new": np.array(ids_new),
        "x_old": x_old,
        "y_old": y_old,
        "pred_old": predict(forward, params, x_old, batch_size).astype(np.int64),
        "split_ids_old": np.array(ids_old),
    }
    task = build_fewshot_task(names, train_classes, new_classes, limits)
    task_text = json_text(task)
    _scan_visible(
        sorted(set(tokens) | {str(row["model_id"]).lower()}),
        **{"task.json": task_text, "evidence.npz": npz_text(arrays)},
    )
    _scan_visible(tokens, **{"model_spec.json": spec_text, "model.npz names": npz_text(params)})

    v_ch, v_sel = data["V_check"], data["V_select"]
    parts = fewshot_check_parts(v_ch, new_classes)
    pred_new_ch = predict(forward, params, parts["x_new"], batch_size)
    pred_old_ch = predict(forward, params, parts["x_old"], batch_size)
    in_new_sel = np.isin(v_sel["y"], new_classes)
    private = {
        "schema": PRIVATE_SCHEMA,
        "evidence_rule": FEWSHOT_RULE,
        "dataset": dataset,
        "arch": row["arch"],
        "seed": int(row["seed"]),
        "model_id": row["model_id"],
        "role": "dev",
        "split_role": "dev",
        "train_classes": train_classes,
        "new_classes": new_classes,
        "model_sha256": digest,
        "model_source_path": str(source / "model.npz"),
        "model_spec_sha256": file_sha256(source / "model_spec.json"),
        "forward_sha256": forward_sha,
        "head_tensors": list(limits.rel_frob_exempt),
        "vevidence_path": split_record["V_evidence"]["path"],
        "vevidence_sha256": split_record["V_evidence"]["sha256"],
        "vevidence_content_sha256": split_record["V_evidence"]["content_sha256"],
        "n_vevidence_new_pool": int(np.count_nonzero(np.isin(v_ev["y"], new_classes))),
        "replay_rule": REPLAY_RULE,
        "replay_stream": [SPLIT_SEED, REPLAY_STREAM],
        "replay_path": split_record["train"]["path"],
        "replay_train_content_sha256": split_record["train"]["content_sha256"],
        "replay_train_rows": [int(r) for r in replay_rows],
        "replay_source_idx": [int(i) for i in train["idx"][replay_rows]],
        "evidence_seed": fewshot_seed(digest, dataset),
        "evidence_row_ids_new": [int(r) for r in new_rows],
        "evidence_source_idx_new": [int(i) for i in v_ev["idx"][new_rows]],
        "evidence_split_ids_new": ids_new,
        "evidence_split_ids_old": ids_old,
        "evidence_content_sha256": _evidence_content_sha256(arrays),
        "n_base_correct_evidence_new": int(np.count_nonzero(arrays["pred_new"] == y_new)),
        "n_base_errors_evidence_old": int(np.count_nonzero(arrays["pred_old"] != y_old)),
        "vcheck_path": split_record["V_check"]["path"],
        "vcheck_sha256": split_record["V_check"]["sha256"],
        "vcheck_content_sha256": split_record["V_check"]["content_sha256"],
        "n_vcheck": int(len(v_ch["y"])),
        "n_vcheck_new": int(len(parts["y_new"])),
        "n_vcheck_old": int(len(parts["y_old"])),
        "n_base_errors_vcheck_new": int(np.count_nonzero(pred_new_ch != parts["y_new"])),
        "n_base_errors_vcheck_old": int(np.count_nonzero(pred_old_ch != parts["y_old"])),
        "n_base_pred_in_new_vcheck": int(np.count_nonzero(np.isin(np.concatenate([pred_new_ch, pred_old_ch]), new_classes))),
        "vselect_path": split_record["V_select"]["path"],
        "vselect_sha256": split_record["V_select"]["sha256"],
        "vselect_content_sha256": split_record["V_select"]["content_sha256"],
        "n_vselect": int(len(v_sel["y"])),
        "n_vselect_new": int(np.count_nonzero(in_new_sel)),
        "n_vselect_old": int(np.count_nonzero(~in_new_sel)),
        "n_per_new_class": N_PER_NEW_CLASS,
        "n_replay": int(len(replay_rows)),
        "task_sha256": _text_sha256(task_text),
    }
    return arrays, task, private


def _create(root: Path, arrays: Mapping[str, np.ndarray], task: Mapping[str, Any], core: Mapping[str, Any],
            forward_path: Path) -> dict[str, Any]:
    instance_id = _new_instance_id(root)
    staging = root / f".tmp-{instance_id}"
    staging.mkdir()
    source = Path(core["model_source_path"])
    shutil.copyfile(source, staging / "model.npz")
    if params_sha256(load_params(staging / "model.npz")) != core["model_sha256"]:
        raise FewshotError(f"{staging}/model.npz: copy hash mismatch")
    shutil.copyfile(source.with_name("model_spec.json"), staging / "model_spec.json")
    if file_sha256(staging / "model_spec.json") != core["model_spec_sha256"]:
        raise FewshotError(f"{staging}/model_spec.json: copy hash mismatch")
    shutil.copyfile(forward_path, staging / "forward.py")
    if file_sha256(staging / "forward.py") != core["forward_sha256"]:
        raise FewshotError(f"{staging}/forward.py: copy hash mismatch")
    if _save_evidence(staging / "evidence.npz", arrays) != core["evidence_content_sha256"]:
        raise FewshotError(f"{staging}/evidence.npz: content hash mismatch")
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
              "vcheck_content_sha256", "vselect_content_sha256", "replay_train_rows")
    changed = [key for key in stable if private.get(key) != core.get(key)]
    if changed or json.loads((instance_dir / "task.json").read_text(encoding="utf-8")) != json.loads(json_text(task)):
        raise FewshotError(f"{instance_dir}: differs from a rebuild in {changed or ['task.json']}; nothing overwritten")


def build_fewshot_instances(
    dataset: str,
    *,
    arch: str | None = None,
    seeds: Sequence[int] | None = None,
    train_classes: Sequence[int] = DEFAULT_TRAIN_CLASSES,
    batch_size: int = EVALUATION_BATCH,
) -> list[dict[str, Any]]:
    """Build (or reuse) one fewshot-v1 instance per selected fewshot student; returns one row per student.

    Students come from the fewshot index of ``dataset`` with the given train classes, optionally one ``arch`` and
    some ``seeds``. An active instance of the same (dataset, model id, train classes) is reused when a rebuild gives
    the same evidence, task and inputs, and refused otherwise.
    """
    from weightcode.models.export import read_fewshot_index  # imports torch; kept out of module import (CODE_AUDIT section 4)

    train = [int(c) for c in train_classes]
    students = [
        row
        for row in read_fewshot_index(dataset)
        if row["train_classes"] == train and (arch is None or row["arch"] == arch)
        and (seeds is None or int(row["seed"]) in {int(s) for s in seeds})
    ]
    if not students:
        raise FewshotError(f"no fewshot students of {dataset} for classes {train}, arch {arch}, seeds {seeds}")
    meta = read_dataset_meta(dataset)
    names = class_names(meta)
    forward, forward_sha, forward_path = load_dataset_forward(dataset, meta)
    tokens = leak_tokens(dataset)
    _scan_visible(tokens, **{"forward.py": forward_path.read_text(encoding="utf-8")})
    data, split_record = _load_splits(dataset, meta)
    replay_rows = draw_replay_rows(data["train"]["y"], train)
    root = fewshot_root()
    outcomes = []
    with _exclusive_lock(root):
        index = read_fewshot_instance_index()
        for row in sorted(students, key=lambda r: (r["arch"], int(r["seed"]))):
            arrays, task, core = _plan(dataset, row, names, data, split_record, replay_rows, forward, forward_sha,
                                       tokens, batch_size)
            existing = [r for r in index["instances"] if r["dataset"] == dataset and r["model_id"] == row["model_id"]
                        and r["train_classes"] == train]
            if existing:
                (match,) = existing
                _same_build(root / match["instance_id"], task, core)
                outcomes.append({**match, "status": "reused"})
                continue
            private = _create(root, arrays, task, core, forward_path)
            index["instances"].append(_index_row(private))
            write_json_atomic(root / "index.json", index)
            with open(root / CHANGES_LOG_NAME, "a", encoding="utf-8") as log:
                log.write(json.dumps({"change": "created", **_index_row(private)}, sort_keys=True) + "\n")
            outcomes.append({**_index_row(private), "status": "created"})
    return outcomes


# ---------------------------------------------------------------------------------------------------------------
# Running an instance: stale check, oracle, readout


def check_fewshot_current(instance_dir: str | Path) -> dict[str, Any]:
    """Refuse an instance that is not active or whose files the current code would not write; else return its task.

    Checks the directory name, the index, the rule, the task.json bytes against private.json, the task against a
    fresh ``build_fewshot_task`` (statement, limits with the head exemption, ops) and the hashes of model.npz,
    model_spec.json, forward.py and evidence.npz. The runner calls it before a session or a baseline.
    """
    root = Path(instance_dir)
    private = read_json(root / "private.json")
    instance_id = str(private.get("instance_id"))
    if root.name != instance_id:
        raise StaleFewshotInstance(f"{root}: directory name differs from private.json instance_id {instance_id!r}")
    index = read_fewshot_instance_index()
    if instance_id not in {row["instance_id"] for row in index["instances"]}:
        retired = {row["instance_id"] for row in index["superseded"]}
        status = "superseded" if instance_id in retired else "not in the index"
        raise StaleFewshotInstance(f"fewshot instance {instance_id} is {status}; only active instances are run")
    if private.get("evidence_rule") != FEWSHOT_RULE:
        raise StaleFewshotInstance(f"{instance_id} follows {private.get('evidence_rule')!r}, not {FEWSHOT_RULE}")
    task_text = (root / "task.json").read_text(encoding="utf-8")
    if _text_sha256(task_text) != private.get("task_sha256"):
        raise StaleFewshotInstance(f"{root / 'task.json'}: bytes do not match private.json task_sha256")
    task = json.loads(task_text)
    spec = read_json(root / "model_spec.json")
    expected = json.loads(json_text(build_fewshot_task(
        task.get("classes", []), private["train_classes"], private["new_classes"], fewshot_limits(spec))))
    if task != expected:
        changed = sorted(key for key in set(task) | set(expected) if task.get(key) != expected.get(key))
        raise StaleFewshotInstance(f"{instance_id}: task.json differs from what the current code writes in {changed}")
    if file_sha256(root / "model_spec.json") != private.get("model_spec_sha256"):
        raise StaleFewshotInstance(f"{root / 'model_spec.json'}: hash differs from private.json")
    instance = load_instance(root)  # model.npz and forward.py hashes against private.json
    if _evidence_content_sha256(instance.evidence) != private.get("evidence_content_sha256"):
        raise StaleFewshotInstance(f"{root / 'evidence.npz'}: content hash differs from private.json")
    return task


def _load_recorded(private: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    key = {"V_check": "vcheck", "V_select": "vselect"}[split]
    path, recorded = Path(str(private[f"{key}_path"])), private.get(f"{key}_content_sha256")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if recorded is None or params_sha256(arrays) != recorded:
        raise FewshotError(f"{path}: content hash differs from the instance record")
    return arrays


def make_fewshot_oracle(
    instance: Instance | str, budget: int = QUERY_BUDGET, batch_size: int = EVALUATION_BATCH
) -> JointPatchCheckOracle:
    """Joint counted oracle of a fewshot instance: its V_check cut by class (``fewshot_check_parts``)."""
    resolved = load_fewshot_instance(instance)
    private = resolved.private
    parts = fewshot_check_parts(_load_recorded(private, "V_check"), private["new_classes"])
    if (len(parts["y_new"]), len(parts["y_old"])) != (private["n_vcheck_new"], private["n_vcheck_old"]):
        raise FewshotError(f"{resolved.instance_id}: V_check parts differ in size from the instance record")
    return JointPatchCheckOracle(parts["x_new"], parts["y_new"], parts["x_old"], parts["y_old"], resolved.forward,
                                 resolved.params, budget, batch_size)


def _accuracy(correct: np.ndarray) -> float:
    return float(np.mean(correct)) if correct.size else float("nan")


def fewshot_readout(
    instance: Instance | str, child: Params, batch_size: int = EVALUATION_BATCH
) -> dict[str, Any]:
    """10-way accuracy of ``child`` against the instance's base on the clean V_select (PREREG E2 readouts).

    Returns the accuracy of base and child and its change over all rows, the new-class rows and the old-class rows,
    the per-class changes, the worst old-class change and the row counts. joint_points is 100 * (delta_new +
    delta_old); all_points (100 * delta_all) is the quantity the PREREG E2 gate reads.
    """
    resolved = load_fewshot_instance(instance)
    private = resolved.private
    arrays = _load_recorded(private, "V_select")
    x, y = arrays["x"], np.asarray(arrays["y"]).astype(np.int64)
    base = predict(resolved.forward, resolved.params, x, batch_size)
    pred = predict(resolved.forward, child, x, batch_size)
    return class_split_readout(base, pred, y, private["train_classes"], private["new_classes"])


def class_split_readout(
    base: np.ndarray, pred: np.ndarray, y: np.ndarray, train_classes: Sequence[int], new_classes: Sequence[int]
) -> dict[str, Any]:
    """The readout of ``fewshot_readout`` from base and child predictions on labelled rows (pure).

    Accuracy of base and child and its change over all rows, the new-class rows and the old-class rows, with fixed
    and broken counts, per-class accuracies over every class, the worst old-class change, joint_points, all_points
    and how many child predictions fall in a new class. ``weightcode.evalkit.graphnew`` reads its V_select with it.
    """
    y = np.asarray(y).astype(np.int64)
    base, pred = np.asarray(base), np.asarray(pred)
    if not base.shape == pred.shape == y.shape:
        raise FewshotError(f"predictions {base.shape} / {pred.shape} do not align with labels {y.shape}")
    new_classes = [int(c) for c in new_classes]
    in_new = np.isin(y, new_classes)
    n_classes = len(train_classes) + len(new_classes)
    result: dict[str, Any] = {"n_all": int(len(y)), "n_new": int(in_new.sum()), "n_old": int((~in_new).sum())}
    for part, mask in (("all", np.ones_like(in_new)), ("new", in_new), ("old", ~in_new)):
        base_acc, child_acc = _accuracy(base[mask] == y[mask]), _accuracy(pred[mask] == y[mask])
        result[f"acc_{part}_base"], result[f"acc_{part}_child"] = base_acc, child_acc
        result[f"delta_{part}"] = child_acc - base_acc
        result[f"fixed_{part}"] = int(np.count_nonzero((base[mask] != y[mask]) & (pred[mask] == y[mask])))
        result[f"broken_{part}"] = int(np.count_nonzero((base[mask] == y[mask]) & (pred[mask] != y[mask])))
    per_class = {}
    for label in range(n_classes):
        rows = y == label
        per_class[str(label)] = {
            "n": int(rows.sum()),
            "acc_base": _accuracy(base[rows] == label),
            "acc_child": _accuracy(pred[rows] == label),
        }
        per_class[str(label)]["delta"] = per_class[str(label)]["acc_child"] - per_class[str(label)]["acc_base"]
    result["per_class"] = per_class
    old_deltas = {label: per_class[str(label)]["delta"] for label in train_classes}
    worst = min(old_deltas, key=lambda label: (old_deltas[label], label))
    result["worst_old_class"] = int(worst)
    result["worst_old_class_delta"] = float(old_deltas[worst])
    result["joint_points"] = 100.0 * (result["delta_new"] + result["delta_old"])
    result["all_points"] = 100.0 * result["delta_all"]
    result["child_pred_in_new_classes"] = int(np.count_nonzero(np.isin(pred, new_classes)))
    return result


__all__ = [
    "CHECK_PART_KEYS", "DEFAULT_TRAIN_CLASSES", "EDITOR_VISIBLE_FILES", "EVIDENCE_KEYS", "FEWSHOT_RULE",
    "N_PER_NEW_CLASS", "N_REPLAY", "PRIVATE_SCHEMA", "REPLAY_RULE", "TASK_SCHEMA", "FewshotError",
    "StaleFewshotInstance", "active_fewshot_ids", "build_fewshot_instances", "build_fewshot_task",
    "check_fewshot_current", "class_split_readout", "draw_new_rows", "draw_replay_rows", "fewshot_check_parts", "fewshot_instance_dir",
    "fewshot_limits", "fewshot_readout", "fewshot_root", "fewshot_seed", "head_tensors", "load_fewshot_instance",
    "make_fewshot_oracle", "new_classes_of", "read_fewshot_instance_index",
]
