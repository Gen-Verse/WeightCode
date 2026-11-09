"""Export and import of the E9 ResNet-20 students (arch ``resnet20_cosine``, dataset ``cifar100_fscil``).

Roles (``FSCIL_ROLES``, PREREG E9):

* ``fscil_base``: the session-1 student, trained on the ``train`` split (430 images of each base class 0-59) with
  cross-entropy over all 100 outputs, so the outputs of classes 60-99 only ever see negative gradient;
* ``fscil_allclass``: the all-class reference, the same recipe on ``train`` plus ``ref_pool`` and ``V_evidence``
  (195 + 5 images of every new class), all 100 classes. A reference point only; it is never edited.

Both live in the fewshot store of the dataset (``weightcode.models.export.fewshot_models_dir``) with
``FEWSHOT_INDEX_KEYS`` rows.

A training job (``scripts/train_fscil.py --group train``) reads only its training splits and dataset.json, never a
readout split. ``export_fscil_model`` writes each student to the run's pending directory (``pending_models_dir``,
outside the fewshot store): model.npz, model_spec.json (``weightcode.models.resnet20.build_resnet20_spec``) and
private.json, with the training log, the clean training loss and accuracy and the training device's logits on the
first ``DEVICE_CHECK_ROWS`` training rows. Its index row carries ``None`` for both accuracies.

``import_fscil_models`` brings the students of a fetched run into the fewshot store. Before anything is copied it
re-checks every student with the local files:

* the parameter hash, the provenance in private.json (training split hashes and the forward.py hash must be the
  local ones), the public spec rebuilt from the parameters;
* parity of the local forward.py on the first ``PARITY_N`` V_select images against CPU float64 and float32 torch
  rebuilds (``weightcode.models.parity.check_logit_parity``);
* the local forward.py against the training device's recorded logits (``DEVICE_CHECK_FACTOR`` times the logit
  scale).

It then computes the readouts with forward.py (100-way argmax, no mask):

* ``vselect_acc``: on V_select (classes 60-99, 200 per class);
* ``vselect_acc_train_classes``: on the rows of the student's train classes in old_select (30 per base class) and
  V_select together. V_select holds no base-class row, so for the base this is its old_select accuracy.

``replace_base_fc`` then applies CEC's last base-stage step (``HEAD_RULE``, user decision 2026-11-04): the head rows
of each student's train classes become the class means of forward.py's features over its training rows. The replaced
students take new model ids and the fewshot index rows; the trained-row students stay in the store, logged in
``REPLACEMENTS_LOG``.

The masked per-session readouts of PREREG E9 belong to ``weightcode.evalkit.fscil``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightcode.common.identity import Params, load_params, params_sha256, save_params
from weightcode.data import cifar100_fscil as C
from weightcode.data.store import (
    dataset_dir,
    forward_path,
    load_split,
    read_dataset_json,
    read_json,
    split_sha256,
    write_json,
)
from weightcode.evalkit.patch_check import ForwardFn
from weightcode.evalkit.targets import utc_now
from weightcode.models import resnet20 as R
from weightcode.models.export import (
    FEWSHOT_INDEX_KEYS,
    PARITY_N,
    ParityError,
    all_model_ids,
    fewshot_models_dir,
    new_model_id,
    read_fewshot_index,
    write_fewshot_index,
)
from weightcode.models.importer import MODEL_FILES, annotate_import, checked_forward, run_record
from weightcode.models.parity import PARITY_RULE, check_logit_parity, logit_scale

DATASET = C.DATASET
FSCIL_PRIVATE_SCHEMA = "weightcode.model_private.fscil.v1"
FSCIL_ROLES: Mapping[str, tuple[str, ...]] = {
    "fscil_base": ("train",),
    "fscil_allclass": ("train", "ref_pool", "V_evidence"),
}
PENDING_DIR = "pending"
PARITY_SPLIT = "V_select"
READOUT_SPLITS: tuple[str, ...] = ("old_select", "V_select")
DEVICE_CHECK_ROWS = 64
DEVICE_CHECK_FACTOR = 1e-3
"""Bound on |forward.py - training-device logits| on the recorded rows, times the logit scale: the device computes in
float32 with its own convolution algorithms, so the bound is looser than the CPU float32 parity bound."""
READOUT_BATCH = 256


def role_train_classes(role: str) -> list[int]:
    """The classes a role trains on: the 60 base classes, or all 100."""
    if role not in FSCIL_ROLES:
        raise ValueError(f"role must be one of {sorted(FSCIL_ROLES)}, got {role!r}")
    return list(range(C.N_BASE_CLASSES)) if role == "fscil_base" else list(range(C.N_CLASSES))


def pending_models_dir() -> Path:
    """``$WEIGHTCODE_PRIVATE/cifar100_fscil/fewshot/pending``: students exported by a training job, not yet imported."""
    return dataset_dir(DATASET) / "fewshot" / PENDING_DIR


def normalization(manifest: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """(mean, std) of dataset.json, refused unless they are the protocol's constants."""
    block = manifest["normalization"]
    mean, std = [float(v) for v in block["mean"]], [float(v) for v in block["std"]]
    if tuple(mean) != C.MEAN or tuple(std) != C.STD:
        raise ValueError(f"dataset.json normalization {mean}/{std} differs from {C.MEAN}/{C.STD}")
    return mean, std


def checked_split(name: str, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """A stored split, refused unless it hashes as dataset.json records."""
    arrays = load_split(DATASET, name)
    if split_sha256(arrays) != manifest["splits"][name]["sha256"]:
        raise RuntimeError(f"{DATASET}/{name} does not match its dataset.json hash")
    return arrays


def training_rows(role: str, manifest: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Images, labels and split hashes of a role's training set (its splits concatenated in ``FSCIL_ROLES`` order)."""
    classes = role_train_classes(role)
    parts = [checked_split(name, manifest) for name in FSCIL_ROLES[role]]
    x = np.concatenate([part["x"] for part in parts])
    y = np.concatenate([part["y"] for part in parts]).astype(np.int64)
    idx = np.concatenate([part["idx"] for part in parts])
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{role}: training splits overlap")
    if sorted(set(y.tolist())) != classes:
        raise ValueError(f"{role}: training labels cover {sorted(set(y.tolist()))[:5]}..., expected {classes[:5]}...")
    return x, y, {name: manifest["splits"][name]["sha256"] for name in FSCIL_ROLES[role]}


def torch_logits(params: Params, x: np.ndarray, mean: Sequence[float], std: Sequence[float],
                 dtype: torch.dtype) -> np.ndarray:
    """CPU torch eval logits of a rebuild in ``dtype`` (float64 copy)."""
    return R.eval_logits(R.build_resnet20(C.N_CLASSES, mean, std, params, dtype), x)


def resnet_parity(params: Params, x: np.ndarray, forward: ForwardFn, mean: Sequence[float],
                  std: Sequence[float]) -> dict[str, float]:
    """``check_logit_parity`` of forward.py on images ``x`` against CPU float64 and float32 torch rebuilds."""
    mine = np.asarray(forward(dict(params), x), dtype=np.float64)
    return check_logit_parity(mine, torch_logits(params, x, mean, std, torch.float64),
                              torch_logits(params, x, mean, std, torch.float32))


def batched_logits(forward: ForwardFn, params: Params, x: np.ndarray, batch_size: int = READOUT_BATCH) -> np.ndarray:
    """forward.py logits of ``x`` in batches (float64 copy)."""
    parts = [np.asarray(forward(dict(params), x[start:start + batch_size]), dtype=np.float64)
             for start in range(0, len(x), batch_size)]
    return np.concatenate(parts)


def readout_accuracies(pred: Mapping[str, np.ndarray], splits: Mapping[str, Mapping[str, np.ndarray]],
                       train_classes: Sequence[int]) -> dict[str, Any]:
    """100-way accuracy per readout split and on the train-class rows of all of them (module docstring)."""
    result: dict[str, Any] = {}
    hits, total = 0, 0
    for name in READOUT_SPLITS:
        y = np.asarray(splits[name]["y"])
        right = pred[name] == y
        result[name] = float(np.mean(right))
        keep = np.isin(y, list(train_classes))
        hits += int(np.count_nonzero(right[keep]))
        total += int(np.count_nonzero(keep))
    result["train_class_rows"] = hits / total
    result["n_train_class_rows"] = total
    result["definition"] = ("100-way argmax of forward.py, no mask: accuracy on old_select (classes 0-59, 30 each) "
                            "and on V_select (classes 60-99, 200 each); train_class_rows over the rows of the "
                            "student's train classes in both")
    return result


def export_fscil_model(
    *,
    manifest: Mapping[str, Any],
    role: str,
    seed: int,
    config: R.FscilTrainConfig,
    result: R.FscilTrainResult,
    training_splits: Mapping[str, str],
    x_train: np.ndarray,
    determinism: Mapping[str, Any],
    run_info: Mapping[str, Any],
    models_root: Path | None = None,
    extra_private: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one student to the pending directory (or ``models_root``) and return its pending index row.

    ``x_train`` is the role's training images in ``training_rows`` order; the first ``DEVICE_CHECK_ROWS`` of them are
    the rows of the device check. Nothing here reads a readout split (module docstring).
    """
    classes = role_train_classes(role)
    if dict(training_splits) != {name: manifest["splits"][name]["sha256"] for name in FSCIL_ROLES[role]}:
        raise ValueError(f"role {role!r} must train on {FSCIL_ROLES[role]} as dataset.json records them")
    mean, std = normalization(manifest)
    root = pending_models_dir() if models_root is None else Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    model_id = new_model_id(all_model_ids(DATASET) | {path.name.removeprefix(".staging-")
                                                      for path in root.iterdir() if path.is_dir()})
    staging = root / f".staging-{model_id}"
    staging.mkdir()
    params = R.export_params(result.model)
    digest = save_params(staging / "model.npz", params)
    reloaded = load_params(staging / "model.npz")
    if params_sha256(reloaded) != digest:
        raise RuntimeError("reloaded parameters hash differently")
    write_json(staging / "model_spec.json",
               R.build_resnet20_spec(model_id, manifest["class_names"], mean, std, reloaded))
    rows = np.ascontiguousarray(x_train[:DEVICE_CHECK_ROWS])
    device_logits = R.eval_logits(result.model, rows)
    device = next(result.model.parameters()).device
    private = {
        "schema": FSCIL_PRIVATE_SCHEMA,
        "model_id": model_id,
        "dataset": DATASET,
        "condition": None,
        "seed": int(seed),
        "arch": R.ARCH,
        "role": role,
        "train_classes": classes,
        "model_config": {"n_classes": C.N_CLASSES, "scale": R.COSINE_SCALE, "bn_eps": R.BN_EPS,
                         "norm_eps": R.NORM_EPS, "mean": mean, "std": std},
        "training_config": config.as_dict(),
        "recipe": R.RECIPE,
        "train_data": {
            "splits": dict(training_splits),
            "order": list(FSCIL_ROLES[role]),
            "n_train": int(len(x_train)),
            "n_per_class": {str(c): int(n) for c, n in enumerate(
                sum(np.asarray(manifest["splits"][name]["class_counts"]) for name in FSCIL_ROLES[role]))},
        },
        "split_seed": manifest["split_seed"],
        "determinism": dict(determinism),
        "train_log": result.log,
        "final_train": dict(result.final),
        "params_sha256": digest,
        "n_parameters": int(sum(value.size for value in reloaded.values())),
        "n_tensors": len(reloaded),
        "layer_norms": R.layer_norms(reloaded),
        "forward_sha256": manifest["forward_sha256"],
        "device_check": {
            "rows": f"the first {len(rows)} training rows in train_data.order",
            "device": str(device),
            "logits": [[float(v) for v in row] for row in device_logits],
        },
        "wall_time_s": result.wall_time_s,
        "readouts": None,
        "readouts_note": "computed at import with the local forward.py (weightcode.models.fscil_export.import_fscil_models)",
        **dict(run_info),
        **(dict(extra_private) if extra_private else {}),
    }
    write_json(staging / "private.json", private)
    staging.rename(root / model_id)
    return {"model_id": model_id, "seed": int(seed), "arch": R.ARCH, "role": role, "params_sha256": digest,
            "vselect_acc": None, "train_classes": classes, "vselect_acc_train_classes": None}


def json_roundtrip(payload: Any) -> Any:
    """``payload`` as JSON reads it back (tuples become lists)."""
    return json.loads(json.dumps(payload))


def _check_one(row: Mapping[str, Any], source: Path, manifest: Mapping[str, Any], forward: ForwardFn,
               forward_sha: str, parity_x: np.ndarray, readouts: Mapping[str, Mapping[str, np.ndarray]]
               ) -> tuple[dict[str, Any], dict[str, Any]]:
    if tuple(row) != FEWSHOT_INDEX_KEYS:
        raise ValueError(f"index row keys {tuple(row)} != {FEWSHOT_INDEX_KEYS}")
    if row["vselect_acc"] is not None or row["vselect_acc_train_classes"] is not None:
        raise ValueError(f"{row['model_id']}: a pending row must carry no accuracy")
    model_id = str(row["model_id"])
    classes = role_train_classes(row["role"])
    folder = source / model_id
    files = sorted(path.name for path in folder.iterdir())
    if files != sorted(MODEL_FILES):
        raise ValueError(f"{folder}: files {files} != {sorted(MODEL_FILES)}")
    params = load_params(folder / "model.npz")
    private = read_json(folder / "private.json")
    expected = {"schema": FSCIL_PRIVATE_SCHEMA, "model_id": model_id, "dataset": DATASET, "condition": None,
                "seed": row["seed"], "arch": R.ARCH, "role": row["role"], "params_sha256": row["params_sha256"],
                "forward_sha256": forward_sha, "train_classes": classes}
    found = {key: private.get(key) for key in expected}
    if params_sha256(params) != row["params_sha256"] or found != expected or row["train_classes"] != classes:
        raise ValueError(f"{model_id}: private.json or model.npz disagrees with the index ({found} vs {expected})")
    if private.get("smoke"):
        raise ValueError(f"{model_id} is a smoke-test model and is never imported")
    role_splits = {name: manifest["splits"][name]["sha256"] for name in FSCIL_ROLES[row["role"]]}
    if private["train_data"]["splits"] != role_splits:
        raise ValueError(f"{model_id}: trained on splits that are not the local ones")
    R.check_params(params, C.N_CLASSES)
    mean, std = normalization(manifest)
    spec = read_json(folder / "model_spec.json")
    if spec != json_roundtrip(R.build_resnet20_spec(model_id, manifest["class_names"], mean, std, params)):
        raise ValueError(f"{model_id}: model_spec.json differs from the spec rebuilt from its parameters")
    started = time.perf_counter()
    try:
        parity = resnet_parity(params, parity_x, forward, mean, std)
    except ParityError as error:
        raise ValueError(f"{model_id}: local parity: {error}") from error
    parity_time = time.perf_counter() - started
    x_train, _, _ = training_rows(row["role"], manifest)
    recorded = np.asarray(private["device_check"]["logits"], dtype=np.float64)
    local = batched_logits(forward, params, np.ascontiguousarray(x_train[:len(recorded)]))
    device_diff = float(np.max(np.abs(local - recorded)))
    device_bound = DEVICE_CHECK_FACTOR * logit_scale(recorded)
    if not device_diff <= device_bound:
        raise ValueError(f"{model_id}: forward.py differs from the training device's logits by {device_diff:.3g} > "
                         f"{device_bound:.3g}")
    pred = {name: np.argmax(batched_logits(forward, params, arrays["x"]), axis=1) for name, arrays in readouts.items()}
    accuracy = readout_accuracies(pred, readouts, classes)
    filled = {**dict(row), "vselect_acc": accuracy["V_select"],
              "vselect_acc_train_classes": accuracy["train_class_rows"]}
    check = {
        "model_id": model_id,
        "parity": {"split": PARITY_SPLIT, "n": int(len(parity_x)), "rule": dict(PARITY_RULE), **parity,
                   "time_s": parity_time},
        "device_check": {"n_rows": int(len(recorded)), "max_abs_diff": device_diff, "bound": device_bound,
                         "argmax_disagreements": int(np.count_nonzero(local.argmax(1) != recorded.argmax(1)))},
        "readouts": accuracy,
    }
    return filled, check


def import_fscil_models(run_dir: Path, note: str | None = None) -> list[dict[str, Any]]:
    """Check every student of a fetched run, copy them into the fewshot store and index them (module docstring).

    Each imported private.json gains an ``import`` block: the run record, the training environment, the local checks
    and readouts, and ``note``. Refuses a model id already used under the dataset and an (arch, train classes, seed)
    already in the local fewshot index. Nothing is copied unless every student passes.
    """
    record = run_record(Path(run_dir))
    source = Path(run_dir) / "private" / DATASET / "fewshot" / PENDING_DIR
    rows = read_json(source / "index.json")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source}/index.json lists no students")
    manifest = read_dataset_json(DATASET)
    forward, forward_sha = checked_forward(DATASET, manifest)
    parity_x = np.ascontiguousarray(checked_split(PARITY_SPLIT, manifest)["x"][:PARITY_N])
    readouts = {name: checked_split(name, manifest) for name in READOUT_SPLITS}
    local = read_fewshot_index(DATASET)
    used_ids = all_model_ids(DATASET)
    taken = {(entry["arch"], tuple(entry["train_classes"]), entry["seed"]) for entry in local}
    results = []
    for row in rows:
        key = (row["arch"], tuple(row["train_classes"]), row["seed"])
        if row["model_id"] in used_ids:
            raise ValueError(f"model id {row['model_id']} is already used under {DATASET}")
        if key in taken:
            raise ValueError(f"{DATASET}: {key} is already in the local fewshot index")
        taken.add(key)
        results.append(_check_one(row, source, manifest, forward, forward_sha, parity_x, readouts))
    root = fewshot_models_dir(DATASET)
    root.mkdir(parents=True, exist_ok=True)
    for row, (_, check) in zip(rows, results):
        staging = root / f".staging-{row['model_id']}"
        shutil.copytree(source / row["model_id"], staging)
        private = read_json(staging / "private.json")
        annotate_import(staging, {**record, "trained_with": {"versions": private["versions"],
                                                             "device": private["device"]},
                                  "local_checks": check, "note": note})
        staging.rename(root / row["model_id"])
    write_fewshot_index(DATASET, local + [filled for filled, _ in results])
    return [check for _, check in results]


# ---------------------------------------------------------------------------------------------------------------
# CEC's replace_base_fc (PREREG E9, user decision 2026-11-04 about 16:05 Beijing)

HEAD_RULE = "cec-replace_base_fc-v1"
HEAD_RULE_DEFINITION = (
    "CEC's replace_base_fc (icoz69/CEC-CVPR2021 at 3a5d335, models/base/fscil_trainer.py, run at the end of the base "
    "session by the README pretrain line, which passes no -not_data_init; models/base/helper.py replace_base_fc): the "
    "fc.weight rows of the student's train classes are set to the mean of the pooled encoder features over the "
    "student's training rows, in eval mode with no augmentation. Here the features are forward.py's features() "
    "(float64, BatchNorm in eval form), the means are taken in float64 and stored as float32; every other tensor is "
    "unchanged. The all-class reference gets the same rule over its own training rows (all 100 rows)."
)
REPLACEMENTS_LOG = "replacements.jsonl"
"""Append-only log, beside the fewshot index, of every student superseded by a head replacement."""
_SOURCE_KEYS = ("import", "device_check", "readouts", "readouts_note", "layer_norms")

FeatureFn = Any  # Callable[[Params, np.ndarray], np.ndarray]: float64 features [N, 64] of raw uint8 images


def checked_features(dataset: str, manifest: Mapping[str, Any]) -> FeatureFn:
    """``features(params, x)`` of the dataset's forward.py, refused unless its sha256 is the one ``manifest`` records."""
    path = forward_path(dataset)
    text = path.read_bytes()
    digest = hashlib.sha256(text).hexdigest()
    if digest != manifest["forward_sha256"]:
        raise RuntimeError(f"{dataset}/forward.py sha256 {digest} != dataset.json {manifest['forward_sha256']}")
    namespace: dict[str, Any] = {"__name__": "weightcode_fscil_forward_features"}
    exec(compile(text, str(path), "exec"), namespace)  # noqa: S102 (the dataset's own hash-checked forward)
    return namespace["features"]


def class_mean_rows(features: FeatureFn, params: Params, x: np.ndarray, y: np.ndarray,
                    classes: Sequence[int]) -> tuple[np.ndarray, list[int]]:
    """float64 [len(classes), 64] mean feature of each class over the rows of ``x``, and the row count per class."""
    f = np.asarray(features(dict(params), x), dtype=np.float64)
    y = np.asarray(y)
    if f.shape != (len(x), R.N_FEATURES) or not np.all(np.isfinite(f)):
        raise ValueError(f"features returned {f.shape} or non-finite values for {len(x)} rows")
    counts = [int(np.count_nonzero(y == c)) for c in classes]
    if min(counts) == 0:
        raise ValueError(f"class {list(classes)[counts.index(0)]} has no training row")
    return np.stack([f[y == c].mean(axis=0) for c in classes]), counts


def replace_fc_rows(params: Params, classes: Sequence[int], rows: np.ndarray) -> Params:
    """``params`` with fc.weight[classes] set to ``rows`` (cast to the head's dtype); every other tensor shared."""
    child = dict(params)
    head = np.array(params[R.HEAD], copy=True)
    head[np.asarray(list(classes))] = np.asarray(rows, dtype=np.float64).astype(head.dtype)
    child[R.HEAD] = head
    return child


def replace_base_fc(seed: int, *, note: str, features: FeatureFn | None = None) -> list[dict[str, Any]]:
    """Apply ``HEAD_RULE`` to the fscil_base and fscil_allclass students of ``seed`` and index the results.

    Each new student gets a new model id, a model_spec.json rebuilt from its parameters, parity of forward.py
    against the torch rebuilds and the import readouts recomputed, and a private.json that copies the source's
    with a ``head_rule`` block; the source's own import, device check, readouts and layer norms move under
    ``head_rule.source``. The new rows replace the source rows in the fewshot index (one row per arch, train classes
    and seed). The source folders stay in the store untouched, and a line per replacement goes to
    ``REPLACEMENTS_LOG`` with ``note``. Refuses a source that already carries a head rule; nothing is written unless
    both students pass. ``features`` defaults to the dataset's forward.py ``features`` (serial).
    """
    if not str(note).strip():
        raise ValueError("a replacement needs a note naming the decision")
    manifest = read_dataset_json(DATASET)
    forward, forward_sha = checked_forward(DATASET, manifest)
    features = checked_features(DATASET, manifest) if features is None else features
    mean, std = normalization(manifest)
    parity_x = np.ascontiguousarray(checked_split(PARITY_SPLIT, manifest)["x"][:PARITY_N])
    readouts = {name: checked_split(name, manifest) for name in READOUT_SPLITS}
    local = read_fewshot_index(DATASET)
    root = fewshot_models_dir(DATASET)
    used_ids = all_model_ids(DATASET)
    planned = []
    for role in FSCIL_ROLES:
        sources = [row for row in local if row["role"] == role and row["arch"] == R.ARCH and row["seed"] == seed]
        if len(sources) != 1:
            raise ValueError(f"expected one {role} student of seed {seed} in the fewshot index, found {len(sources)}")
        (source_row,) = sources
        folder = root / str(source_row["model_id"])
        params = load_params(folder / "model.npz")
        private = read_json(folder / "private.json")
        if params_sha256(params) != source_row["params_sha256"] or private.get("role") != role:
            raise ValueError(f"{folder}: model.npz or private.json disagrees with the fewshot index")
        if "head_rule" in private:
            raise ValueError(f"{folder}: already carries head rule {private['head_rule'].get('name')!r}")
        classes = role_train_classes(role)
        x_train, y_train, split_hashes = training_rows(role, manifest)
        started = time.perf_counter()
        rows, counts = class_mean_rows(features, params, x_train, y_train, classes)
        feature_s = time.perf_counter() - started
        child = replace_fc_rows(params, classes, rows)
        R.check_params(child, C.N_CLASSES)
        changed = sorted(name for name in params if not np.array_equal(params[name], child[name]))
        if changed != [R.HEAD]:
            raise ValueError(f"the replacement changed {changed}, not only {R.HEAD}")
        model_id = new_model_id(used_ids)
        used_ids.add(model_id)
        try:
            parity = resnet_parity(child, parity_x, forward, mean, std)
        except ParityError as error:
            raise ValueError(f"{model_id}: local parity: {error}") from error
        pred = {name: np.argmax(batched_logits(forward, child, arrays["x"]), axis=1)
                for name, arrays in readouts.items()}
        accuracy = readout_accuracies(pred, readouts, classes)
        digest = params_sha256(child)
        head = params[R.HEAD].astype(np.float64)
        head_rule = {
            "name": HEAD_RULE,
            "definition": HEAD_RULE_DEFINITION,
            "rows_replaced": classes,
            "rows_per_class": counts,
            "training_splits": split_hashes,
            "n_training_rows": int(len(y_train)),
            "features": "forward.py features()",
            "forward_sha256": forward_sha,
            "feature_time_s": feature_s,
            "head_rel_change": float(np.linalg.norm(child[R.HEAD].astype(np.float64) - head) / np.linalg.norm(head)),
            "note": str(note),
            "source": {"model_id": source_row["model_id"], "params_sha256": source_row["params_sha256"],
                       **{key: private.get(key) for key in _SOURCE_KEYS}},
            "local_checks": {"parity": {"split": PARITY_SPLIT, "n": int(len(parity_x)), "rule": dict(PARITY_RULE),
                                        **parity},
                             "readouts": accuracy},
        }
        new_private = {key: value for key, value in private.items() if key not in _SOURCE_KEYS}
        new_private.update(model_id=model_id, params_sha256=digest, layer_norms=R.layer_norms(child),
                           readouts=accuracy, readouts_note="recomputed after the head rule (head_rule.local_checks)",
                           head_rule=head_rule)
        index_row = {**source_row, "model_id": model_id, "params_sha256": digest,
                     "vselect_acc": accuracy["V_select"], "vselect_acc_train_classes": accuracy["train_class_rows"]}
        index_row = {key: index_row[key] for key in FEWSHOT_INDEX_KEYS}
        spec = R.build_resnet20_spec(model_id, manifest["class_names"], mean, std, child)
        planned.append((source_row, index_row, child, new_private, spec))
    records = []
    for source_row, index_row, child, new_private, spec in planned:
        model_id = index_row["model_id"]
        staging = root / f".staging-{model_id}"
        staging.mkdir()
        save_params(staging / "model.npz", child)
        write_json(staging / "model_spec.json", spec)
        write_json(staging / "private.json", new_private)
        staging.rename(root / model_id)
        records.append({"change": "head_rule", "rule": HEAD_RULE, "role": source_row["role"], "seed": seed,
                        "superseded": {"model_id": source_row["model_id"],
                                       "params_sha256": source_row["params_sha256"]},
                        "by": {"model_id": model_id, "params_sha256": index_row["params_sha256"]},
                        "readouts": new_private["readouts"], "note": str(note), "at": utc_now()})
    superseded = {source_row["model_id"] for source_row, *_ in planned}
    write_fewshot_index(DATASET, [row for row in local if row["model_id"] not in superseded]
                        + [index_row for _, index_row, *_ in planned])
    with open(root / REPLACEMENTS_LOG, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records


__all__ = [
    "DATASET", "DEVICE_CHECK_FACTOR", "DEVICE_CHECK_ROWS", "FSCIL_PRIVATE_SCHEMA", "FSCIL_ROLES", "HEAD_RULE",
    "HEAD_RULE_DEFINITION", "PENDING_DIR", "READOUT_SPLITS", "REPLACEMENTS_LOG", "batched_logits", "checked_features",
    "checked_split", "class_mean_rows", "export_fscil_model", "import_fscil_models", "normalization",
    "pending_models_dir", "readout_accuracies", "replace_base_fc", "replace_fc_rows", "resnet_parity",
    "role_train_classes", "torch_logits", "training_rows",
]
