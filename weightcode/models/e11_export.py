"""PREREG E11 students: export from the training job (pending) and the data they train on.

The six students (arch ``vit_lite_7_4``, ``weightcode.models.vitlite``; seeds 100 to 105) train on E1's 40,000 CIFAR-10
train rows with E1's per-channel normalisation (``weightcode.models.e10_export.training_rows`` and ``normalization``, the
same rows and constants as E10's students). A training job writes each one to
``$WEIGHTCODE_PRIVATE/cifar10/e11_pending/<model_id>/`` (model.npz and private.json, with the job's device logits on the
first ``DEVICE_CHECK_ROWS`` training rows) and a pending index; nothing there is a store model. The import (parity
against forward.py, the device check and the readouts) runs on the workstation; the job reads no readout split.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.data.store import dataset_dir, load_split, split_sha256, write_json
from weightcode.models import vitlite as V
from weightcode.models.e10_export import DEVICE_CHECK_ROWS, N_CLASSES, SOURCE_DATASET, normalization, training_rows
from weightcode.models.export import all_model_ids, new_model_id

PENDING_DIR = "e11_pending"
PRIVATE_SCHEMA = "weightcode.e11_model_private.v1"
SEEDS: tuple[int, ...] = (100, 101, 102, 103, 104, 105)
ROLE = "e11_base"


def pending_models_dir() -> Path:
    """``$WEIGHTCODE_PRIVATE/cifar10/e11_pending``: students exported by a training job, not yet imported."""
    return dataset_dir(SOURCE_DATASET) / PENDING_DIR


def export_pending(
    *,
    manifest: Mapping[str, Any],
    seed: int,
    config: V.CCTTrainConfig,
    model_config: V.ViTLiteConfig,
    result: V.CCTTrainResult,
    training_splits: Mapping[str, str],
    x_train: np.ndarray,
    determinism: Mapping[str, Any],
    run_info: Mapping[str, Any],
    models_root: Path | None = None,
    extra_private: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one student to the pending directory (or ``models_root``) and return its pending index row."""
    if dict(training_splits) != {"train": manifest["splits"]["train"]["sha256"]}:
        raise ValueError("E11 students train on cifar10's train split as dataset.json records it")
    mean, std = normalization(manifest)
    root = pending_models_dir() if models_root is None else Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    taken = {path.name.removeprefix(".staging-") for path in root.iterdir() if path.is_dir()}
    model_id = new_model_id(all_model_ids(SOURCE_DATASET) | taken)
    staging = root / f".staging-{model_id}"
    staging.mkdir()  # an id taken concurrently raises here; nothing is overwritten
    params = V.export_params(result.model)
    digest = save_params(staging / "model.npz", params)
    reloaded = load_params(staging / "model.npz")
    if params_sha256(reloaded) != digest:
        raise RuntimeError("reloaded parameters hash differently")
    rows = np.ascontiguousarray(x_train[:DEVICE_CHECK_ROWS])
    device_logits = V.eval_logits(result.model, rows)
    device = next(result.model.parameters()).device
    private = {
        "schema": PRIVATE_SCHEMA,
        "model_id": model_id,
        "source_dataset": SOURCE_DATASET,
        "seed": int(seed),
        "arch": V.ARCH,
        "role": ROLE,
        "model_config": {**model_config.as_dict(), "ln_eps": V.LN_EPS, "mean": mean, "std": std},
        "training_config": config.as_dict(),
        "recipe": V.RECIPE,
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
        "wall_time_s": result.wall_time_s,
        "readouts": None,
        "readouts_note": "computed at import with the local forward.py",
        **dict(run_info),
        **(dict(extra_private) if extra_private else {}),
    }
    write_json(staging / "private.json", private)
    final = root / model_id
    staging.rename(final)
    return {"model_id": model_id, "seed": int(seed), "arch": V.ARCH, "role": ROLE, "params_sha256": digest,
            "path": str(final)}


PARITY_ROWS = 1000
PARITY_FLOAT64_REL_TOL = 1e-6  # forward.py casts its float64 logits to float32 once
DEVICE_MAX_ABS_LOGIT_DIFF = 1e-3


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def _as_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _check_model_config(private: Mapping[str, Any], mean: list[float], std: list[float]) -> None:
    expected = _as_json({**V.ViTLiteConfig().as_dict(), "ln_eps": V.LN_EPS, "mean": mean, "std": std})
    if private.get("model_config") != expected:
        raise RuntimeError(f"{private.get('model_id')}: model_config {private.get('model_config')} is not PREREG "
                           f"E11's {expected}")
    if private.get("training_config") != _as_json(V.CCTTrainConfig().as_dict()):
        raise RuntimeError(f"{private.get('model_id')}: training_config differs from the pinned recipe")


def import_e11_models(run_dir: Path, note: str) -> list[dict[str, Any]]:
    """Import the pending students of a fetched training run into the E11 store (``weightcode.data.e11.DATASET``).

    For each pending row: the params hash; the refusal of smoke students and of any model or training config other
    than PREREG E11's (``ViTLiteConfig()``, ``CCTTrainConfig()``); parity of the store's forward.py against a CPU
    torch float64 rebuild (relative logit difference at most ``PARITY_FLOAT64_REL_TOL``) and against float32 (max
    abs difference and argmax disagreements, recorded) on the first ``PARITY_ROWS`` clean V_select rows; the device
    check (forward.py logits on the first ``DEVICE_CHECK_ROWS`` train rows against the job's device logits: max abs
    difference at most ``DEVICE_MAX_ABS_LOGIT_DIFF`` and no argmax disagreement); the clean V_select accuracy (the
    index's ``vselect_acc``). Then model.npz, model_spec.json and private.json go to ``models/<model_id>/`` and a row
    to ``models/index.json`` (role "dev", one per seed). Refuses a seed or id already in the index.
    """
    import torch

    from weightcode.data import e11 as D
    from weightcode.data.store import models_dir, read_json
    from weightcode.evalkit.targets import load_dataset_forward, read_dataset_meta

    if not note or not note.strip():
        raise ValueError("an import needs a note")
    pending = Path(run_dir) / "private" / SOURCE_DATASET / PENDING_DIR
    rows = read_json(pending / "index.json")
    meta = read_dataset_meta(D.DATASET)
    forward, forward_sha, _ = load_dataset_forward(D.DATASET, meta)
    mean, std = normalization(meta)
    vselect = load_split(D.DATASET, "V_select")
    train = load_split(D.DATASET, "train")
    if split_sha256(vselect) != meta["splits"]["V_select"]["sha256"] or split_sha256(train) != meta["splits"]["train"][
            "sha256"]:
        raise RuntimeError("the E11 store's splits do not match its dataset.json")
    root = models_dir(D.DATASET)
    root.mkdir(parents=True, exist_ok=True)
    index_file = root / "index.json"
    index = read_json(index_file) if index_file.exists() else []
    config = V.ViTLiteConfig()
    imported = []
    for row in rows:
        source = pending / row["model_id"]
        params = load_params(source / "model.npz")
        if params_sha256(params) != row["params_sha256"]:
            raise RuntimeError(f"{source}/model.npz hash differs from the pending index")
        private = read_json(source / "private.json")
        if private.get("smoke"):
            raise RuntimeError(f"{source} is a smoke student")
        if private.get("arch") != V.ARCH or private.get("role") != ROLE:
            raise RuntimeError(f"{source}: arch {private.get('arch')!r}, role {private.get('role')!r}")
        _check_model_config(private, mean, std)
        if any(r["model_id"] == row["model_id"] or (r["seed"] == row["seed"] and r["role"] == "dev") for r in index):
            raise RuntimeError(f"seed {row['seed']} or id {row['model_id']} is already in {index_file}")
        x_par = np.ascontiguousarray(vselect["x"][:PARITY_ROWS])
        ours = np.asarray(forward(params, x_par), dtype=np.float64)
        ref64 = V.eval_logits(V.build_vitlite(config, mean, std, params, torch.float64), x_par)
        ref32 = V.eval_logits(V.build_vitlite(config, mean, std, params, torch.float32), x_par)
        parity = {"rows": f"the first {PARITY_ROWS} clean V_select rows",
                  "float64_max_rel_diff": _relative(ours, ref64), "float64_tolerance": PARITY_FLOAT64_REL_TOL,
                  "float32_max_abs_diff": float(np.max(np.abs(ours - ref32))),
                  "float32_argmax_disagreements": int(np.count_nonzero(ours.argmax(1) != ref32.argmax(1)))}
        if parity["float64_max_rel_diff"] > PARITY_FLOAT64_REL_TOL:
            raise RuntimeError(f"{row['model_id']}: forward.py parity {parity}")
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
                   V.build_vitlite_spec(row["model_id"], meta["class_names"], mean, std, params))
        write_json(staging / "private.json", {
            **{k: v for k, v in private.items() if k not in ("readouts", "readouts_note")},
            "dataset": D.DATASET, "condition": None, "store_role": "dev", "split_role": "dev",
            "import": {"run_dir": str(run_dir), "note": note, "forward_sha256": forward_sha,
                       "local_checks": {"parity": parity, "device_check": device}},
            "readouts": {"clean_vselect_acc": acc, "n_vselect": int(len(pred))},
        })
        staging.rename(target)
        entry = {"model_id": row["model_id"], "seed": int(row["seed"]), "arch": V.ARCH, "role": "dev",
                 "params_sha256": row["params_sha256"], "vselect_acc": acc}
        index.append(entry)
        write_json(index_file, index)
        imported.append({**entry, "parity": parity, "device_check": device})
    return imported


__all__ = ["DEVICE_CHECK_ROWS", "N_CLASSES", "PENDING_DIR", "PRIVATE_SCHEMA", "ROLE", "SEEDS", "SOURCE_DATASET",
           "export_pending", "import_e11_models", "normalization", "pending_models_dir", "training_rows"]
