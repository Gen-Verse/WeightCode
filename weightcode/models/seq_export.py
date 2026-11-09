"""Export and import of the seqops students (arch ``tfm``) in the store layout of ``weightcode.models.export``.

``export_seq_model`` writes one student under ``models_dir("seqops")`` (or ``models_root``): model.npz,
model_spec.json (``weightcode.models.transformer.build_seq_model_spec``) and private.json, after the numpy forward.py has
passed ``seq_parity`` at every position of the first PARITY_N base V_select rows: against a CPU float64 torch rebuild
within one float32 rounding step of each logit (forward.py computes in float64 and casts the logits once to float32,
so this is an exactness test of the implementation), and against the CPU float32 torch model within PARITY_TOL times
the logit scale max(1, max |logit|). Float32 rounding grows with the size of the values summed, so an absolute bound
fails for a correct forward once the logits are large (a student with a common logit offset near -400 differs from
float32 torch by 2.4e-4, about 8 float32 steps at that magnitude).
Its index row has ``INDEX_KEYS``; ``vselect_acc`` is the row exact match of forward.py over every base V_select row
(old operators).

``import_seq_models`` brings the students of a fetched NSCC run into the local store. Before anything is copied it
re-checks every model with the local files: the parameter hash, the provenance in private.json (the held-out keys it
refused must be the local ones), the public spec rebuilt from the parameters, parity, and vselect_acc recomputed
with the local forward.py, which must equal the exported value.
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
from weightcode.data import seqops as S
from weightcode.data.seqops_build import CLASS_NAMES, FORWARD_N_HEADS
from weightcode.data.store import forward_path, models_dir, read_dataset_json, read_json, write_json
from weightcode.evalkit.metrics import rows_right
from weightcode.evalkit.patch_check import ForwardFn, predict
from weightcode.models import transformer as T
from weightcode.models.export import (
    INDEX_KEYS,
    PARITY_N,
    PARITY_SPLIT,
    PARITY_TOL,
    ParityError,
    all_model_ids,
    load_forward_no_cache,
    new_model_id,
    read_index,
    write_index,
)
from weightcode.models.importer import (
    MODEL_FILES,
    _checked_vselect,
    annotate_import,
    checked_forward,
    models_source,
    run_record,
)
from weightcode.models.parity import FLOAT32_STEP, PARITY_RULE, check_logit_parity, float32_tolerance, logit_scale
from weightcode.models.train_seq import SeqTrainConfig, SeqTrainResult, em_by_operator, torch_predict

SEQ_PRIVATE_SCHEMA = "weightcode.model_private.seqops.v1"
SEQ_ROLES: tuple[str, ...] = ("base", "dev")
EVAL_BATCH = 1024
VSELECT_ACC_DEFINITION = (
    "numpy forward.py, row exact match over the OUT positions (every labelled position right), on all base V_select "
    "rows (old operators, single and composed, n 4-12)"
)


def build_spec(model_id: str, config: T.TfmConfig, params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The public model_spec.json of a seqops student."""
    return T.build_seq_model_spec(model_id, config, S.token_meanings(), list(CLASS_NAMES), S.INPUT_LAYOUT, params)


def torch_cpu_logits(config: T.TfmConfig, params: Params, x: np.ndarray, dtype: torch.dtype = torch.float32) -> np.ndarray:
    """CPU torch eval logits of the student with ``params`` on token rows ``x`` in ``dtype`` (float64 copy)."""
    model = T.load_seq_params(T.build_seq_model(config), params).to(dtype).eval()
    with torch.no_grad():
        return model(torch.from_numpy(np.ascontiguousarray(x))).numpy().astype(np.float64)


def seq_parity(config: T.TfmConfig, params: Params, x: np.ndarray, forward: ForwardFn,
               numpy_logits: np.ndarray | None = None) -> dict[str, float]:
    """forward.py against CPU torch float64 and float32 rebuilds at every position of ``x`` (``PARITY_RULE``).

    Raises ParityError if either bound fails. ``numpy_logits`` reuses forward.py's logits when the caller has them.
    """
    mine = forward(dict(params), x).astype(np.float64) if numpy_logits is None else np.asarray(numpy_logits, np.float64)
    exact = torch_cpu_logits(config, params, x, torch.float64)
    return check_logit_parity(mine, exact, torch_cpu_logits(config, params, x, torch.float32))


def vselect_accuracy(forward: ForwardFn, params: Params, vselect: Mapping[str, np.ndarray]) -> tuple[float, np.ndarray]:
    """Row exact match of forward.py over all V_select rows, and the per-position predictions."""
    pred = predict(forward, params, vselect["x"], batch_size=EVAL_BATCH)
    return float(np.mean(rows_right(vselect["y"], pred))), pred


def check_model_config(config: T.TfmConfig) -> None:
    """A student must fit the dataset's forward.py: vocabulary, length, outputs, pad id and head count."""
    expected = (S.VOCAB_SIZE, S.SEQ_LEN, S.N_SYMBOLS, S.PAD, FORWARD_N_HEADS)
    found = (config.vocab_size, config.seq_len, config.n_out, config.pad_id, config.n_heads)
    if found != expected:
        raise ValueError(f"student config {found} does not fit the seqops forward.py {expected}")


def export_seq_model(
    *,
    manifest: Mapping[str, Any],
    role: str,
    seed: int,
    config: SeqTrainConfig,
    result: SeqTrainResult,
    size_check: Mapping[str, Any],
    determinism: Mapping[str, Any],
    run_info: Mapping[str, Any],
    vselect: Mapping[str, np.ndarray],
    models_root: Path | None = None,
    extra_private: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one student's files, verify parity and return its index row (module docstring)."""
    if role not in SEQ_ROLES:
        raise ValueError(f"role must be one of {SEQ_ROLES}")
    model_config = config.model_config()
    check_model_config(model_config)
    root = models_dir(S.DATASET) if models_root is None else Path(models_root)
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
    if forward_sha != manifest["forward_sha256"]:
        raise RuntimeError(f"forward.py sha256 {forward_sha} != dataset.json {manifest['forward_sha256']}")

    x_par = np.ascontiguousarray(vselect["x"][:PARITY_N])
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
        raise ParityError(f"seqops {role} seed {seed}: {error}") from error
    parity_dev = float(np.max(np.abs(numpy_logits - torch_dev)))

    vselect_acc, pred = vselect_accuracy(forward, reloaded, vselect)
    pred_torch = torch_predict(result.model, np.ascontiguousarray(vselect["x"]), device)
    labelled = vselect["y"] != S.IGNORE
    spec = build_spec(model_id, model_config, reloaded)
    write_json(staging / "model_spec.json", spec)
    private = {
        "schema": SEQ_PRIVATE_SCHEMA,
        "model_id": model_id,
        "dataset": S.DATASET,
        "condition": None,
        "seed": seed,
        "arch": config.arch,
        "role": role,
        "model_config": asdict(model_config),
        "train_data": {
            "source": "weightcode.data.seqops.TrainStream: a fresh batch of old-operator rows per step",
            "batches": "default_rng([split_seed, 1100, seed, step]); rows whose key is held out are refused and redrawn",
            "held_out": dict(manifest["held_out"]),
            "n_refused_rows": int(result.n_refused),
            "rows_seen": int(config.steps * config.batch_size),
        },
        "split_seed": manifest["split_seed"],
        "training_config": config.as_dict(),
        "determinism": dict(determinism),
        "train_log": result.log,
        "size_check": dict(size_check),
        "reserved_rows_unchanged": bool(result.reserved_rows_unchanged),
        "final_vselect_acc": vselect_acc,
        "final_vselect_acc_definition": VSELECT_ACC_DEFINITION,
        "final_vselect_acc_torch_device": float(np.mean(rows_right(vselect["y"], pred_torch))),
        "vselect_position_disagreements_numpy_vs_torch_device": int(np.count_nonzero((pred != pred_torch) & labelled)),
        "vselect_by_operator": em_by_operator(vselect, pred),
        "params_sha256": digest,
        "n_parameters": int(sum(value.size for value in reloaded.values())),
        "forward_sha256": forward_sha,
        "parity": {
            "split": PARITY_SPLIT,
            "n_rows": int(x_par.shape[0]),
            "rows": f"first {int(x_par.shape[0])} base V_select rows in stored order, every position",
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
            "vselect_acc": vselect_acc}


def _check_one(
    row: Mapping[str, Any], source: Path, manifest: Mapping[str, Any], forward: ForwardFn, forward_sha: str,
    vselect: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if tuple(row) != INDEX_KEYS:
        raise ValueError(f"index row keys {tuple(row)} != {INDEX_KEYS}")
    model_id = row["model_id"]
    folder = source / model_id
    files = sorted(path.name for path in folder.iterdir())
    if files != sorted(MODEL_FILES):
        raise ValueError(f"{folder}: files {files} != {sorted(MODEL_FILES)}")
    params = load_params(folder / "model.npz")
    private = read_json(folder / "private.json")
    expected = {"model_id": model_id, "dataset": S.DATASET, "condition": None, "seed": row["seed"], "arch": row["arch"],
                "role": row["role"], "params_sha256": row["params_sha256"], "forward_sha256": forward_sha,
                "schema": SEQ_PRIVATE_SCHEMA}
    found = {key: private.get(key) for key in expected}
    if params_sha256(params) != row["params_sha256"] or found != expected:
        raise ValueError(f"{model_id}: private.json or model.npz disagrees with the index ({found} vs {expected})")
    if private.get("smoke"):
        raise ValueError(f"{model_id} is a smoke-test model and is never imported")
    if private["train_data"]["held_out"] != manifest["held_out"]:
        raise ValueError(f"{model_id}: trained against held-out keys that are not the local ones")
    config = T.TfmConfig(**private["model_config"])
    check_model_config(config)
    spec = read_json(folder / "model_spec.json")
    if spec != json_roundtrip(build_spec(model_id, config, params)):
        raise ValueError(f"{model_id}: model_spec.json differs from the spec rebuilt from its parameters")
    try:
        parity = seq_parity(config, params, np.ascontiguousarray(vselect["x"][:PARITY_N]), forward)
    except ParityError as error:
        raise ValueError(f"{model_id}: local parity: {error}") from error
    accuracy, _ = vselect_accuracy(forward, params, vselect)
    if accuracy != row["vselect_acc"] or accuracy != private["final_vselect_acc"]:
        raise ValueError(f"{model_id}: local V_select exact match {accuracy} != exported {row['vselect_acc']} "
                         f"(private {private['final_vselect_acc']})")
    return {"model_id": model_id, "local_parity_max_abs_diff": parity["max_abs_diff"],
            "local_parity_tolerance": parity["tolerance"],
            "local_parity_float64_max_rel_diff": parity["float64_max_rel_diff"], "local_vselect_acc": accuracy}


def json_roundtrip(payload: Any) -> Any:
    import json

    return json.loads(json.dumps(payload))


RETIRED_LOG = "retired.jsonl"
"""Log of students removed from the seqops model index, under the models directory: one JSON line each."""


def retire_seq_model(model_id: str, reason: str) -> dict[str, Any]:
    """Remove a student from the seqops model index and log why; its directory stays, so its id is never reused.

    Builders read only the index, so no new instance is built on a retired student. The index row, the reason and
    the time are appended to ``models/retired.jsonl`` before the index is rewritten.
    """
    import datetime

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("retiring a student needs a reason")
    rows = read_index(S.DATASET)
    hits = [row for row in rows if row["model_id"] == model_id]
    if len(hits) != 1:
        raise ValueError(f"{model_id} is not in the seqops model index")
    record = {"model_id": model_id, "row": hits[0], "reason": reason,
              "retired_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}
    with open(models_dir(S.DATASET) / RETIRED_LOG, "a", encoding="utf-8") as log:
        log.write(json.dumps(record, sort_keys=True) + "\n")
    write_index(S.DATASET, [row for row in rows if row["model_id"] != model_id])
    return record


def import_seq_models(run_dir: Path, note: str | None = None) -> list[dict[str, Any]]:
    """Check every student of a fetched NSCC run, copy them into the local store and index them.

    Each imported private.json gains an ``import`` block: the run record (job, request, container image and hash),
    the training environment recorded at export, the local re-checks and ``note``. Refuses a model id already used
    under seqops and a (role, seed) already in the local index. Nothing is copied unless every model passes.
    """
    record = run_record(Path(run_dir))
    source = models_source(Path(run_dir), S.DATASET, None)
    manifest = read_dataset_json(S.DATASET)
    forward, forward_sha = checked_forward(S.DATASET, manifest)
    vselect = _checked_vselect(S.DATASET, None, manifest)
    rows = read_json(source / "index.json")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source}/index.json lists no models")
    local = read_index(S.DATASET)
    used_ids = all_model_ids(S.DATASET)
    taken = {(entry["role"], entry["seed"]) for entry in local}
    checks = []
    for row in rows:
        if row["model_id"] in used_ids:
            raise ValueError(f"model id {row['model_id']} is already used under {S.DATASET}")
        if (row["role"], row["seed"]) in taken:
            raise ValueError(f"{S.DATASET}: {row['role']} seed {row['seed']} is already in the local index")
        taken.add((row["role"], row["seed"]))
        checks.append(_check_one(row, source, manifest, forward, forward_sha, vselect))
    root = models_dir(S.DATASET)
    root.mkdir(parents=True, exist_ok=True)
    for row, check in zip(rows, checks):
        staging = root / f".staging-{row['model_id']}"
        shutil.copytree(source / row["model_id"], staging)
        private = read_json(staging / "private.json")
        annotate_import(staging, {**record, "trained_with": {"versions": private["versions"],
                                                             "device": private["device"]},
                                  "local_checks": check, "note": note})
        staging.rename(root / row["model_id"])
    write_index(S.DATASET, local + [dict(row) for row in rows])
    return checks
