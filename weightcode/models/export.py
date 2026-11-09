"""Export of a trained model: model.npz, model_spec.json, private.json, staged and parity-checked (brief item 5).

Every model is written to ``models/.staging-<model_id>/`` first. The parameters are reloaded from the npz and the
dataset's ``forward.py`` (loaded from its file, sha256 checked against dataset.json) must reproduce the CPU
float32 torch eval logits of the reloaded parameters to max abs diff <= ``PARITY_TOL`` on the first
``PARITY_N`` V_select images. Only then is the directory renamed to ``models/<model_id>/``. A failed check raises
and leaves the staging directory for inspection.
"""

from __future__ import annotations

import secrets
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.data.store import dataset_dir, forward_path, list_conditions, models_dir, read_json, write_json
from weightcode.evalkit.patch_check import ForwardFn, load_forward, predict
from weightcode.models.arch import build_model, export_params, load_params_into
from weightcode.models.spec import build_model_spec
from weightcode.models.train import TrainConfig, TrainResult, predict_torch

PRIVATE_SCHEMA = "weightcode.model_private.v0"
PARITY_TOL = 1e-4
PARITY_N = 1000
PARITY_SPLIT = "V_select"
ROLES: tuple[str, ...] = ("base", "ceiling", "dev")
INDEX_KEYS: tuple[str, ...] = ("model_id", "seed", "arch", "role", "params_sha256", "vselect_acc")
FEWSHOT_INDEX_KEYS: tuple[str, ...] = (*INDEX_KEYS, "train_classes", "vselect_acc_train_classes")
"""Index row of a fewshot student: ``vselect_acc`` over all V_select rows, the second over its train classes."""


class ParityError(RuntimeError):
    """The numpy reference forward does not reproduce the torch logits within tolerance."""


def load_forward_no_cache(path: Path) -> tuple[ForwardFn, str]:
    """``load_forward`` without writing ``__pycache__`` beside the dataset's forward.py."""
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return load_forward(path)
    finally:
        sys.dont_write_bytecode = previous


def new_model_id(existing: set[str]) -> str:
    """Opaque random id ``m`` + 8 hex digits, unique among ``existing``."""
    while True:
        model_id = "m" + secrets.token_hex(4)
        if model_id not in existing:
            return model_id


def index_path(dataset: str, condition: str | None = None) -> Path:
    """Path of the model index of a dataset, or of one deployment condition of it (its ceiling models)."""
    return models_dir(dataset, condition) / "index.json"


def read_index(dataset: str, condition: str | None = None) -> list[dict[str, Any]]:
    """Index entries of the exported, active models of ``dataset`` (or its ``condition``); empty if none yet."""
    path = index_path(dataset, condition)
    if not path.exists():
        return []
    entries = read_json(path)
    if not isinstance(entries, list):
        raise ValueError(f"{path} must hold a JSON list")
    return entries


def write_index(dataset: str, entries: list[dict[str, Any]], condition: str | None = None) -> None:
    """Rewrite the model index atomically, ordered by role then seed."""
    order = {role: position for position, role in enumerate(ROLES)}
    for entry in entries:
        if tuple(entry) != INDEX_KEYS:
            raise ValueError(f"index entry keys {tuple(entry)} != {INDEX_KEYS}")
    write_json(index_path(dataset, condition), sorted(entries, key=lambda e: (order[e["role"]], e["seed"])))


def fewshot_models_dir(dataset: str) -> Path:
    """``$WEIGHTCODE_PRIVATE/<dataset>/fewshot/models``: the students trained on a subset of the classes (PREREG E2).

    They have their own index, so the dataset's model index, its roles and every builder that reads it never see
    them.
    """
    return dataset_dir(dataset) / "fewshot" / "models"


def read_fewshot_index(dataset: str) -> list[dict[str, Any]]:
    """Index rows of the fewshot students of ``dataset``; empty if none yet."""
    path = fewshot_models_dir(dataset) / "index.json"
    if not path.exists():
        return []
    entries = read_json(path)
    if not isinstance(entries, list):
        raise ValueError(f"{path} must hold a JSON list")
    return entries


def write_fewshot_index(dataset: str, entries: list[dict[str, Any]]) -> None:
    """Rewrite the fewshot index atomically, ordered by arch, train classes and seed; one row per such triple."""
    keys = set()
    for entry in entries:
        if tuple(entry) != FEWSHOT_INDEX_KEYS:
            raise ValueError(f"fewshot index entry keys {tuple(entry)} != {FEWSHOT_INDEX_KEYS}")
        key = (entry["arch"], tuple(entry["train_classes"]), entry["seed"])
        if key in keys:
            raise ValueError(f"fewshot index holds {key} twice")
        keys.add(key)
    ordered = sorted(entries, key=lambda e: (e["arch"], tuple(e["train_classes"]), e["seed"]))
    write_json(fewshot_models_dir(dataset) / "index.json", ordered)


def all_model_ids(dataset: str) -> set[str]:
    """Every model id used under a dataset, its conditions and fewshot students included (unique per dataset)."""
    roots = [models_dir(dataset)] + [models_dir(dataset, name) for name in list_conditions(dataset)]
    roots.append(fewshot_models_dir(dataset))
    ids: set[str] = set()
    for root in roots:
        if root.is_dir():
            ids |= {path.name.removeprefix(".staging-") for path in root.iterdir() if path.is_dir()}
    return ids


def confusion_matrix(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts [true, predicted]."""
    return np.bincount(y * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def top_confusions(confusion: np.ndarray, k: int) -> list[tuple[int, int, int]]:
    """The ``k`` largest off-diagonal cells as (true, predicted, count); ties by (true, predicted)."""
    cells = [(int(confusion[a, b]), a, b) for a in range(confusion.shape[0]) for b in range(confusion.shape[1]) if a != b]
    cells.sort(key=lambda cell: (-cell[0], cell[1], cell[2]))
    return [(a, b, count) for count, a, b in cells[:k]]


def export_model(
    *,
    dataset: str,
    manifest: Mapping[str, Any],
    role: str,
    seed: int,
    config: TrainConfig,
    train_split: str,
    train_split_sha256: str,
    result: TrainResult,
    determinism: Mapping[str, Any],
    run_info: Mapping[str, Any],
    vselect: Mapping[str, np.ndarray],
    extra_private: Mapping[str, Any] | None = None,
    condition: str | None = None,
    models_root: Path | None = None,
) -> dict[str, Any]:
    """Write one model's files, verify parity and return its index entry.

    With ``condition``, the model goes to the condition's models directory, ``manifest`` and ``vselect`` are the
    condition's, and the forward is the dataset's shared forward.py. ``models_root`` puts a clean-data model in
    another directory (the fewshot students, ``fewshot_models_dir``); it excludes ``condition``.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if models_root is not None and condition is not None:
        raise ValueError("models_root is for clean-data models; it cannot be combined with a condition")
    root = models_dir(dataset, condition) if models_root is None else Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    model_id = new_model_id(all_model_ids(dataset))
    staging = root / f".staging-{model_id}"
    staging.mkdir()

    params = export_params(result.model)
    digest = save_params(staging / "model.npz", params)
    reloaded = load_params(staging / "model.npz")
    if params_sha256(reloaded) != digest:
        raise RuntimeError("reloaded parameters hash differently")

    forward, forward_sha = load_forward_no_cache(forward_path(dataset))
    if forward_sha != manifest["forward_sha256"]:
        raise RuntimeError(f"forward.py sha256 {forward_sha} != dataset.json {manifest['forward_sha256']}")

    input_shape = manifest["image_shape"]
    mean, std = manifest["norm"]["mean"], manifest["norm"]["std"]
    n_classes = len(manifest["class_names"])
    x_val, y_val = vselect["x"], vselect["y"]
    x_par = x_val[:PARITY_N]
    cpu_model = load_params_into(build_model(config.arch, input_shape, n_classes, mean, std), reloaded).eval()
    with torch.no_grad():
        torch_cpu = cpu_model(torch.from_numpy(x_par)).numpy().astype(np.float64)
    device = next(result.model.parameters()).device
    torch_dev = predict_torch(result.model, torch.from_numpy(x_par).to(device)).cpu().numpy().astype(np.float64)
    started = time.perf_counter()
    numpy_logits = forward(dict(reloaded), x_par).astype(np.float64)
    forward_time = time.perf_counter() - started
    parity = float(np.max(np.abs(numpy_logits - torch_cpu)))
    parity_dev = float(np.max(np.abs(numpy_logits - torch_dev)))
    if not parity <= PARITY_TOL:
        raise ParityError(f"{dataset} {role} seed {seed}: numpy vs torch cpu max abs diff {parity} > {PARITY_TOL}")

    pred = predict(forward, reloaded, x_val, batch_size=1024)
    pred_torch = predict_torch(result.model, torch.from_numpy(x_val).to(device)).argmax(dim=1).cpu().numpy()
    vselect_acc = float(np.mean(pred == y_val))
    confusion = confusion_matrix(y_val, pred, n_classes)

    spec = build_model_spec(model_id, config.arch, input_shape, mean, std, manifest["class_names"], reloaded)
    write_json(staging / "model_spec.json", spec)
    private = {
        "schema": PRIVATE_SCHEMA,
        "model_id": model_id,
        "dataset": dataset,
        "condition": condition,
        "seed": seed,
        "arch": config.arch,
        "role": role,
        "train_split": train_split,
        "train_split_sha256": train_split_sha256,
        "split_seed": manifest["split_seed"],
        "training_config": config.as_dict(),
        "determinism": dict(determinism),
        "epoch_log": result.epoch_log,
        "final_vselect_acc": vselect_acc,
        "final_vselect_acc_definition": "numpy forward.py argmax on all V_select images (canonical)",
        "final_vselect_acc_torch_device": float(np.mean(pred_torch == y_val)),
        "vselect_argmax_disagreements_numpy_vs_torch_device": int(np.count_nonzero(pred != pred_torch)),
        "vselect_confusion": confusion.tolist(),
        "params_sha256": digest,
        "n_parameters": int(sum(value.size for value in reloaded.values())),
        "forward_sha256": forward_sha,
        "parity": {
            "split": PARITY_SPLIT,
            "n_images": int(x_par.shape[0]),
            "images": f"first {int(x_par.shape[0])} V_select images in stored order",
            "reference": "torch eval logits, CPU float32, parameters reloaded from model.npz",
            "max_abs_diff": parity,
            "tolerance": PARITY_TOL,
            "max_abs_diff_vs_training_device": parity_dev,
            "numpy_forward_time_s": forward_time,
        },
        "wall_time_s": result.wall_time_s,
        **dict(run_info),
        **(dict(extra_private) if extra_private else {}),
    }
    write_json(staging / "private.json", private)
    final_dir = root / model_id
    staging.rename(final_dir)
    return {
        "model_id": model_id,
        "seed": seed,
        "arch": config.arch,
        "role": role,
        "params_sha256": digest,
        "vselect_acc": vselect_acc,
    }
