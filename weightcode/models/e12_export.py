"""PREREG E12 students: export from the training job (pending) and the data they train on.

The students (arch ``resnet18_up``, ``weightcode.models.resnet18up``) start from torchvision's IMAGENET1K_V1 ResNet-18
(``$WEIGHTCODE_PRIVATE/pretrained/resnet18-f37072fd.pth``, sha256-checked) and fine-tune on E1's 40,000 CIFAR-10 train rows
(``$WEIGHTCODE_PRIVATE/cifar10/splits/train.npz``, hash-checked against cifar10's dataset.json). A training job writes each
one to ``$WEIGHTCODE_PRIVATE/cifar10/e12_pending/<model_id>/`` (model.npz and private.json, with the job's device logits on
the first ``DEVICE_CHECK_ROWS`` training rows and the module's agreement with torchvision's resnet18) and a pending
index; nothing there is a store model. The import (parity against forward.py, the device check and the clean V_select
accuracy) runs on the workstation and writes the E12 store; the job reads no readout split.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.common.paths import private_dir
from weightcode.data.store import dataset_dir, load_split, read_dataset_json, split_sha256, write_json
from weightcode.models import resnet18up as M
from weightcode.models.export import all_model_ids, new_model_id

SOURCE_DATASET = "cifar10"
N_CLASSES = 10
PENDING_DIR = "e12_pending"
PRIVATE_SCHEMA = "weightcode.e12_model_private.v1"
DEVICE_CHECK_ROWS = 64
SOURCE_SEED = 100
DECOY_SEED = 101
SEEDS: tuple[int, ...] = (SOURCE_SEED, DECOY_SEED)
"""Seed 100 is PREREG E12's student; seed 101 is the instance builder's decoy for it (and seed 100 for 101)."""
ROLE = "e12_base"
WEIGHTS_RELPATH = f"pretrained/{M.WEIGHTS_FILE}"
"""The ImageNet weights under $WEIGHTCODE_PRIVATE (the training job's input)."""


def weights_path() -> Path:
    return private_dir() / WEIGHTS_RELPATH


def pending_models_dir() -> Path:
    """``$WEIGHTCODE_PRIVATE/cifar10/e12_pending``: students exported by a training job, not yet imported."""
    return dataset_dir(SOURCE_DATASET) / PENDING_DIR


def training_rows() -> tuple[np.ndarray, np.ndarray, dict[str, str], dict[str, Any]]:
    """E1's train split (images, labels, {split: sha256}) and cifar10's dataset.json, hash-checked."""
    manifest = read_dataset_json(SOURCE_DATASET)
    arrays = load_split(SOURCE_DATASET, "train")
    digest = split_sha256(arrays)
    if digest != manifest["splits"]["train"]["sha256"]:
        raise RuntimeError("cifar10/train does not match its dataset.json hash")
    return np.ascontiguousarray(arrays["x"]), arrays["y"].astype(np.int64), {"train": digest}, manifest


def export_pending(
    *,
    manifest: Mapping[str, Any],
    seed: int,
    config: M.E12TrainConfig,
    model_config: M.ResNet18UpConfig,
    result: M.E12TrainResult,
    training_splits: Mapping[str, str],
    x_train: np.ndarray,
    determinism: Mapping[str, Any],
    run_info: Mapping[str, Any],
    weights: Mapping[str, Any],
    torchvision_check: Mapping[str, Any],
    models_root: Path | None = None,
    extra_private: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one student to the pending directory (or ``models_root``) and return its pending index row."""
    if dict(training_splits) != {"train": manifest["splits"]["train"]["sha256"]}:
        raise ValueError("E12 students train on cifar10's train split as dataset.json records it")
    root = pending_models_dir() if models_root is None else Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    taken = {path.name.removeprefix(".staging-") for path in root.iterdir() if path.is_dir()}
    model_id = new_model_id(all_model_ids(SOURCE_DATASET) | taken)
    staging = root / f".staging-{model_id}"
    staging.mkdir()
    params = M.export_params(result.model)
    digest = save_params(staging / "model.npz", params)
    reloaded = load_params(staging / "model.npz")
    if params_sha256(reloaded) != digest:
        raise RuntimeError("reloaded parameters hash differently")
    rows = np.ascontiguousarray(x_train[:DEVICE_CHECK_ROWS])
    device_logits = M.eval_logits(result.model, rows)
    device = next(result.model.parameters()).device
    private = {
        "schema": PRIVATE_SCHEMA,
        "model_id": model_id,
        "source_dataset": SOURCE_DATASET,
        "seed": int(seed),
        "arch": M.ARCH,
        "role": ROLE,
        "model_config": {**model_config.as_dict(), "bn_eps": M.BN_EPS, "mean": list(M.IMAGENET_MEAN),
                         "std": list(M.IMAGENET_STD)},
        "pretrained": dict(weights),
        "training_config": config.as_dict(),
        "recipe": M.RECIPE,
        "train_data": {"splits": dict(training_splits), "n_train": int(len(x_train)),
                       "class_counts": manifest["splits"]["train"]["class_counts"]},
        "split_seed": manifest["split_seed"],
        "determinism": dict(determinism),
        "train_log": result.log,
        "final_train": dict(result.final),
        "params_sha256": digest,
        "n_parameters": int(sum(value.size for value in reloaded.values())),
        "n_tensors": len(reloaded),
        "device_check": {
            "rows": f"the first {len(rows)} rows of cifar10/train",
            "device": str(device),
            "logits": [[float(v) for v in row] for row in device_logits],
        },
        "torchvision_check": dict(torchvision_check),
        "wall_time_s": result.wall_time_s,
        "readouts": None,
        "readouts_note": "computed at import with the local forward.py",
        **dict(run_info),
        **(dict(extra_private) if extra_private else {}),
    }
    write_json(staging / "private.json", private)
    final = root / model_id
    staging.rename(final)
    return {"model_id": model_id, "seed": int(seed), "arch": M.ARCH, "role": ROLE, "params_sha256": digest,
            "path": str(final)}


PARITY_ROWS = 1000
DEVICE_MAX_ABS_LOGIT_DIFF = 1e-3


def import_e12_models(run_dir: Path, note: str) -> list[dict[str, Any]]:
    """Import the pending students of a fetched training run into the E12 store (``weightcode.data.e12.DATASET``).

    For each pending row: the params hash; the refusal of smoke students and of a student trained at another R than
    the store's forward.py; parity of the store's forward.py against CPU torch rebuilds on the first ``PARITY_ROWS``
    clean V_select rows (``weightcode.models.parity.check_logit_parity``: float64 within one float32 step of max(1, |logit|),
    float32 within PARITY_TOL times the logit scale; the float32 argmax disagreements are recorded); the device
    check (forward.py logits on the first ``DEVICE_CHECK_ROWS`` train rows against the job's device logits: max abs
    difference at most ``DEVICE_MAX_ABS_LOGIT_DIFF`` and no argmax disagreement); the clean V_select accuracy (the
    index's ``vselect_acc``). Then model.npz, model_spec.json and private.json go to ``models/<model_id>/`` and a row to
    ``models/index.json`` (role "dev", one per seed). Refuses a seed or id already in the index.
    """
    import torch

    from weightcode.data import e12 as D
    from weightcode.data.store import models_dir, read_json
    from weightcode.evalkit.targets import load_dataset_forward, read_dataset_meta
    from weightcode.models.parity import check_logit_parity

    if not note or not note.strip():
        raise ValueError("an import needs a note")
    pending = Path(run_dir) / "private" / SOURCE_DATASET / PENDING_DIR
    rows = read_json(pending / "index.json")
    meta = read_dataset_meta(D.DATASET)
    forward, forward_sha, _ = load_dataset_forward(D.DATASET, meta)
    store_resize = int(meta["student_preprocessing"]["resize"])
    vselect = load_split(D.DATASET, "V_select")
    train = load_split(D.DATASET, "train")
    if split_sha256(vselect) != meta["splits"]["V_select"]["sha256"] or split_sha256(train) != meta["splits"]["train"][
            "sha256"]:
        raise RuntimeError("the E12 store's splits do not match its dataset.json")
    root = models_dir(D.DATASET)
    root.mkdir(parents=True, exist_ok=True)
    index_file = root / "index.json"
    index = read_json(index_file) if index_file.exists() else []
    imported = []
    for row in rows:
        source = pending / row["model_id"]
        params = load_params(source / "model.npz")
        if params_sha256(params) != row["params_sha256"]:
            raise RuntimeError(f"{source}/model.npz hash differs from the pending index")
        private = read_json(source / "private.json")
        if private.get("smoke"):
            raise RuntimeError(f"{source} is a smoke student")
        config = M.ResNet18UpConfig(n_classes=int(private["model_config"]["n_classes"]),
                                    resize=int(private["model_config"]["resize"]))
        if config.resize != store_resize:
            raise RuntimeError(f"{row['model_id']} was trained at R {config.resize}; the store's forward.py has R "
                               f"{store_resize}")
        if any(r["model_id"] == row["model_id"] or (r["seed"] == row["seed"] and r["role"] == "dev") for r in index):
            raise RuntimeError(f"seed {row['seed']} or id {row['model_id']} is already in {index_file}")
        x_par = np.ascontiguousarray(vselect["x"][:PARITY_ROWS])
        ours = np.asarray(forward(params, x_par), dtype=np.float64)
        ref64 = M.eval_logits(M.build_resnet18up(config, params, torch.float64), x_par)
        ref32 = M.eval_logits(M.build_resnet18up(config, params, torch.float32), x_par)
        parity = {"rows": f"the first {PARITY_ROWS} clean V_select rows", **check_logit_parity(ours, ref64, ref32),
                  "float32_argmax_disagreements": int(np.count_nonzero(ours.argmax(1) != ref32.argmax(1)))}
        device_logits = np.asarray(private["device_check"]["logits"], dtype=np.float64)
        local = np.asarray(forward(params, np.ascontiguousarray(train["x"][:DEVICE_CHECK_ROWS])), dtype=np.float64)
        device = {"rows": private["device_check"]["rows"], "device": private["device_check"]["device"],
                  "max_abs_diff": float(np.max(np.abs(local - device_logits))),
                  "argmax_disagreements": int(np.count_nonzero(local.argmax(1) != device_logits.argmax(1)))}
        if device["max_abs_diff"] > DEVICE_MAX_ABS_LOGIT_DIFF or device["argmax_disagreements"]:
            raise RuntimeError(f"{row['model_id']}: device check {device}")
        pred = np.asarray(forward(params, vselect["x"])).argmax(1)
        acc = float(np.mean(pred == vselect["y"]))
        target = root / row["model_id"]
        staging = root / f".staging-{row['model_id']}"
        staging.mkdir()
        save_params(staging / "model.npz", params)
        write_json(staging / "model_spec.json",
                   M.build_resnet18up_spec(row["model_id"], meta["class_names"], config, params))
        write_json(staging / "private.json", {
            **{k: v for k, v in private.items() if k not in ("readouts", "readouts_note")},
            "dataset": D.DATASET, "condition": None, "store_role": "dev", "split_role": "dev",
            "import": {"run_dir": str(run_dir), "note": note, "forward_sha256": forward_sha,
                       "local_checks": {"parity": parity, "device_check": device}},
            "readouts": {"clean_vselect_acc": acc, "n_vselect": int(len(pred))},
        })
        staging.rename(target)
        entry = {"model_id": row["model_id"], "seed": int(row["seed"]), "arch": M.ARCH, "role": "dev",
                 "params_sha256": row["params_sha256"], "vselect_acc": acc}
        index.append(entry)
        write_json(index_file, index)
        imported.append({**entry, "parity": parity, "device_check": device})
    return imported


__all__ = ["DECOY_SEED", "DEVICE_CHECK_ROWS", "N_CLASSES", "PENDING_DIR", "PRIVATE_SCHEMA", "ROLE", "SEEDS",
           "SOURCE_DATASET", "SOURCE_SEED", "WEIGHTS_RELPATH", "export_pending", "import_e12_models",
           "pending_models_dir", "training_rows", "weights_path"]
