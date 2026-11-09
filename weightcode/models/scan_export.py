"""Export and import of the E13 SCAN students (arch ``tfm``) in the store layout of ``weightcode.models.export``.

``export_scan_model`` writes one student under ``models_dir("scan", condition)`` (or ``models_root``): model.npz,
model_spec.json (``weightcode.models.transformer.build_seq_model_spec`` with SCAN's token meanings, the grammar as
published) and private.json, after the numpy forward.py has passed ``seq_parity`` (``weightcode.models.seq_export``,
``weightcode.models.parity.PARITY_RULE``) at every position of the 1000 old V_select rows. Its index row has
``INDEX_KEYS``; ``vselect_acc`` is the gate number of PREREG E13: forward.py's exact match on the 1000 old V_select
rows (old commands held out of training). The role is ``base`` when it reaches ``GATE_EM`` and ``dev`` otherwise.
No new-part row is read here: headroom (exact match on new V_select) is read locally after the import.

``import_scan_models`` brings the students of a fetched NSCC run into the local store. Before anything is copied it
re-checks every model with the local files: the parameter hash, the provenance (the training split's hash must be the
local one), the public spec rebuilt from the parameters, parity, and vselect_acc recomputed with the local forward.py,
which must equal the exported value.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightcode.common.identity import Params, load_params, params_sha256, save_params
from weightcode.data import scan as S
from weightcode.data.scan_build import FORWARD_N_HEADS, checked_split
from weightcode.data.store import forward_path, models_dir, read_dataset_json, read_json, write_json
from weightcode.evalkit.metrics import rows_right
from weightcode.evalkit.patch_check import ForwardFn, predict
from weightcode.models import transformer as T
from weightcode.models.export import (
    INDEX_KEYS,
    ParityError,
    all_model_ids,
    load_forward_no_cache,
    new_model_id,
    read_index,
    write_index,
)
from weightcode.models.importer import MODEL_FILES, annotate_import, models_source, run_record
from weightcode.models.parity import PARITY_RULE
from weightcode.models.seq_export import seq_parity
from weightcode.models.train_scan import GATE_EM, ScanTrainConfig, ScanTrainResult, em_summary, torch_predict

SCAN_PRIVATE_SCHEMA = "weightcode.model_private.scan.v1"
SCAN_ROLES: tuple[str, ...] = ("base", "dev")
GATE_SPLIT = "old_select"
EVAL_BATCH = 1024
VSELECT_ACC_DEFINITION = (
    "numpy forward.py, row exact match over the OUT positions (every OUT position right, END included), on the 1000 "
    "old V_select rows (old commands held out of training); the PREREG E13 gate"
)


def build_spec(model_id: str, config: T.TfmConfig, params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The public model_spec.json of an E13 student: SCAN's grammar as every token's meaning."""
    return T.build_seq_model_spec(model_id, config, S.token_table(), list(S.CLASS_NAMES), S.INPUT_LAYOUT, params)


def check_model_config(config: T.TfmConfig) -> None:
    """A student must fit the dataset's forward.py: vocabulary, length, classes, pad id and head count."""
    expected = (S.VOCAB_SIZE, S.SEQ_LEN, S.N_CLASSES, S.PAD, FORWARD_N_HEADS)
    found = (config.vocab_size, config.seq_len, config.n_out, config.pad_id, config.n_heads)
    if found != expected:
        raise ValueError(f"student config {found} does not fit the SCAN forward.py {expected}")


def gate_accuracy(forward: ForwardFn, params: Params, old_select: Mapping[str, np.ndarray]) -> tuple[float, np.ndarray]:
    """forward.py's exact match over the old V_select rows, and the per-position predictions."""
    pred = predict(forward, params, old_select["x"], batch_size=EVAL_BATCH)
    return float(np.mean(rows_right(old_select["y"], pred))), pred


def gate_record(accuracy: float) -> dict[str, Any]:
    return {"split": GATE_SPLIT, "n": S.N_OLD_HELD, "threshold_em": GATE_EM, "em": accuracy,
            "passed": accuracy >= GATE_EM, "definition": VSELECT_ACC_DEFINITION}


def export_scan_model(
    *,
    condition: str,
    manifest: Mapping[str, Any],
    base_manifest: Mapping[str, Any],
    seed: int,
    size_step: int,
    config: ScanTrainConfig,
    result: ScanTrainResult,
    determinism: Mapping[str, Any],
    run_info: Mapping[str, Any],
    old_select: Mapping[str, np.ndarray],
    models_root: Path | None = None,
    extra_private: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one student's files, verify parity, read the gate and return its index row (module docstring)."""
    model_config = config.model_config()
    check_model_config(model_config)
    root = models_dir(S.DATASET, condition) if models_root is None else Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    model_id = new_model_id(all_model_ids(S.DATASET) | {p.name for p in root.iterdir() if p.is_dir()})
    staging = root / f".staging-{model_id}"
    staging.mkdir()

    params = T.export_seq_params(result.model)
    digest = save_params(staging / "model.npz", params)
    reloaded = load_params(staging / "model.npz")
    if params_sha256(reloaded) != digest:
        raise RuntimeError("reloaded parameters hash differently")
    forward, forward_sha = load_forward_no_cache(forward_path(S.DATASET))
    if forward_sha != base_manifest["forward_sha256"] or forward_sha != manifest["forward_sha256"]:
        raise RuntimeError(f"forward.py sha256 {forward_sha} != dataset.json {base_manifest['forward_sha256']}")

    x_par = np.ascontiguousarray(old_select["x"])
    device = next(result.model.parameters()).device
    result.model.eval()
    with torch.no_grad():
        torch_dev = result.model(torch.from_numpy(x_par).to(device)).cpu().numpy().astype(np.float64)
    started = time.perf_counter()
    numpy_logits = forward(dict(reloaded), x_par).astype(np.float64)
    forward_time = time.perf_counter() - started
    try:
        parity = seq_parity(model_config, reloaded, x_par, forward, numpy_logits)
    except ParityError as error:
        raise ParityError(f"scan {condition} seed {seed} size step {size_step}: {error}") from error
    parity_dev = float(np.max(np.abs(numpy_logits - torch_dev)))

    accuracy, pred = gate_accuracy(forward, reloaded, old_select)
    gate = gate_record(accuracy)
    role = "base" if gate["passed"] else "dev"
    pred_torch = torch_predict(result.model, x_par, device)
    labelled = old_select["y"] != S.IGNORE
    spec = build_spec(model_id, model_config, reloaded)
    write_json(staging / "model_spec.json", spec)
    train_record = manifest["splits"]["train"]
    private = {
        "schema": SCAN_PRIVATE_SCHEMA,
        "model_id": model_id,
        "dataset": S.DATASET,
        "condition": condition,
        "seed": seed,
        "arch": config.arch,
        "role": role,
        "size_step": size_step,
        "model_config": asdict(model_config),
        "train_data": {
            "split": "train",
            "sha256": train_record["sha256"],
            "n_rows": train_record["n"],
            "batches": "default_rng([split_seed, 1310, seed, epoch]).permutation(n) per epoch, batches of "
            "batch_size in that order",
            "rows_seen": int(result.rows_seen),
            "steps": int(result.steps),
        },
        "split_seed": manifest["split_seed"],
        "training_config": config.as_dict(),
        "determinism": dict(determinism),
        "train_log": result.log,
        "gate": gate,
        "final_vselect_acc": accuracy,
        "final_vselect_acc_definition": VSELECT_ACC_DEFINITION,
        "final_vselect_acc_torch_device": float(np.mean(rows_right(old_select["y"], pred_torch))),
        "vselect_position_disagreements_numpy_vs_torch_device": int(np.count_nonzero((pred != pred_torch) & labelled)),
        "old_vselect_summary": em_summary(old_select, pred),
        "params_sha256": digest,
        "n_parameters": int(sum(value.size for value in reloaded.values())),
        "forward_sha256": forward_sha,
        "parity": {
            "split": GATE_SPLIT,
            "n_rows": int(x_par.shape[0]),
            "rows": f"all {int(x_par.shape[0])} old V_select rows, every position",
            "reference": "torch eval logits, CPU float32, parameters reloaded from model.npz",
            "max_abs_diff": parity["max_abs_diff"],
            "tolerance": parity["tolerance"],
            "logit_scale": parity["logit_scale"],
            "rule": dict(PARITY_RULE),
            "float64": {"reference": "torch eval logits, CPU float64, parameters reloaded from model.npz",
                        "max_abs_diff": parity["float64_max_abs_diff"],
                        "max_rel_diff": parity["float64_max_rel_diff"],
                        "tolerance": parity["float64_tolerance"]},
            "max_abs_diff_vs_training_device": parity_dev,
            "numpy_forward_time_s": forward_time,
        },
        "wall_time_s": result.wall_time_s,
        **dict(run_info),
        **(dict(extra_private) if extra_private else {}),
    }
    write_json(staging / "private.json", private)
    staging.rename(root / model_id)
    return {"model_id": model_id, "seed": seed, "arch": config.arch, "role": role, "params_sha256": digest,
            "vselect_acc": accuracy}


def _json_roundtrip(payload: Any) -> Any:
    return json.loads(json.dumps(payload))


def _check_one(row: Mapping[str, Any], source: Path, condition: str, manifest: Mapping[str, Any], forward: ForwardFn,
               forward_sha: str, old_select: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if tuple(row) != INDEX_KEYS:
        raise ValueError(f"index row keys {tuple(row)} != {INDEX_KEYS}")
    model_id = row["model_id"]
    folder = source / model_id
    files = sorted(path.name for path in folder.iterdir())
    if files != sorted(MODEL_FILES):
        raise ValueError(f"{folder}: files {files} != {sorted(MODEL_FILES)}")
    params = load_params(folder / "model.npz")
    private = read_json(folder / "private.json")
    expected = {"model_id": model_id, "dataset": S.DATASET, "condition": condition, "seed": row["seed"],
                "arch": row["arch"], "role": row["role"], "params_sha256": row["params_sha256"],
                "forward_sha256": forward_sha, "schema": SCAN_PRIVATE_SCHEMA}
    found = {key: private.get(key) for key in expected}
    if params_sha256(params) != row["params_sha256"] or found != expected:
        raise ValueError(f"{model_id}: private.json or model.npz disagrees with the index ({found} vs {expected})")
    if private.get("smoke"):
        raise ValueError(f"{model_id} is a smoke-test model and is never imported")
    if private["train_data"]["sha256"] != manifest["splits"]["train"]["sha256"]:
        raise ValueError(f"{model_id}: trained on a train split that is not the local one")
    config = T.TfmConfig(**private["model_config"])
    check_model_config(config)
    spec = read_json(folder / "model_spec.json")
    if spec != _json_roundtrip(build_spec(model_id, config, params)):
        raise ValueError(f"{model_id}: model_spec.json differs from the spec rebuilt from its parameters")
    try:
        parity = seq_parity(config, params, np.ascontiguousarray(old_select["x"]), forward)
    except ParityError as error:
        raise ValueError(f"{model_id}: local parity: {error}") from error
    accuracy, _ = gate_accuracy(forward, params, old_select)
    if accuracy != row["vselect_acc"] or accuracy != private["final_vselect_acc"]:
        raise ValueError(f"{model_id}: local old V_select exact match {accuracy} != exported {row['vselect_acc']}")
    if (row["role"] == "base") != (accuracy >= GATE_EM):
        raise ValueError(f"{model_id}: role {row['role']} does not follow the gate at {accuracy}")
    return {"model_id": model_id, "local_parity_max_abs_diff": parity["max_abs_diff"],
            "local_parity_tolerance": parity["tolerance"],
            "local_parity_float64_max_rel_diff": parity["float64_max_rel_diff"], "local_vselect_acc": accuracy}


def import_scan_models(run_dir: Path, condition: str, note: str | None = None) -> list[dict[str, Any]]:
    """Check every student of one condition in a fetched NSCC run, copy them into the local store and index them.

    Refuses a model id already used under scan, a (role, seed) already in the condition's index (so a second base of
    one seed), a smoke model and a role that does not follow the gate. Nothing is copied unless every model passes.
    """
    record = run_record(Path(run_dir))
    source = models_source(Path(run_dir), S.DATASET, condition)
    manifest = read_dataset_json(S.DATASET, condition)
    base = read_dataset_json(S.DATASET)
    forward, forward_sha = load_forward_no_cache(forward_path(S.DATASET))
    if forward_sha != base["forward_sha256"]:
        raise ValueError(f"local forward.py sha256 {forward_sha} != dataset.json {base['forward_sha256']}")
    old_select = checked_split(condition, GATE_SPLIT)
    rows = read_json(source / "index.json")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source}/index.json lists no models")
    local = read_index(S.DATASET, condition)
    used_ids = all_model_ids(S.DATASET)
    taken = {(entry["role"], entry["seed"]) for entry in local}
    checks = []
    for row in rows:
        if row["model_id"] in used_ids:
            raise ValueError(f"model id {row['model_id']} is already used under {S.DATASET}")
        if (row["role"], row["seed"]) in taken:
            raise ValueError(f"{S.DATASET}/{condition}: {row['role']} seed {row['seed']} is already in the index")
        taken.add((row["role"], row["seed"]))
        checks.append(_check_one(row, source, condition, manifest, forward, forward_sha, old_select))
    root = models_dir(S.DATASET, condition)
    root.mkdir(parents=True, exist_ok=True)
    for row, check in zip(rows, checks):
        staging = root / f".staging-{row['model_id']}"
        shutil.copytree(source / row["model_id"], staging)
        private = read_json(staging / "private.json")
        annotate_import(staging, {**record, "trained_with": {"versions": private["versions"],
                                                             "device": private["device"]},
                                  "local_checks": check, "note": note})
        staging.rename(root / row["model_id"])
    write_index(S.DATASET, local + [dict(row) for row in rows], condition)
    return checks


__all__ = ["GATE_SPLIT", "SCAN_PRIVATE_SCHEMA", "SCAN_ROLES", "build_spec", "check_model_config",
           "export_scan_model", "gate_accuracy", "gate_record", "import_scan_models"]
