"""Models trained elsewhere and clean models read under a deployment condition.

``import_models`` brings models exported by an NSCC run into the local store. Every file is re-checked here before
anything is copied: the parameter hash, the provenance in private.json, the public spec (rebuilt from the
parameters), numpy-versus-torch parity and the V_select accuracy recomputed with the local forward.py, which must
equal the exported value.

``evaluate_on_condition`` reads the dataset's clean models on a condition's V_select (accuracy only) and writes
``<condition>/models/base_eval.json``. ``checked_forward``, ``load_indexed_models`` and ``EVAL_BATCH_SIZE`` are that
evaluation path, shared with the severity calibration of ``weightcode.models.calibration``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightcode.common.identity import Params, load_params, params_sha256
from weightcode.data.store import (
    forward_path,
    load_split,
    models_dir,
    read_dataset_json,
    read_json,
    split_sha256,
    write_json,
)
from weightcode.evalkit.patch_check import ForwardFn, predict
from weightcode.evalkit.targets import file_sha256
from weightcode.models.arch import build_model, load_params_into
from weightcode.models.export import (
    INDEX_KEYS,
    PARITY_N,
    PARITY_TOL,
    all_model_ids,
    load_forward_no_cache,
    read_index,
    write_index,
)
from weightcode.models.spec import build_model_spec, check_public

MODEL_FILES: tuple[str, ...] = ("model.npz", "model_spec.json", "private.json")
BASE_EVAL_SCHEMA = "weightcode.base_eval.v0"
BASE_EVAL_FILE = "base_eval.json"
EVAL_BATCH_SIZE = 1024
"""Batch size of ``predict`` in the clean-model evaluations (it bounds memory and changes no prediction)."""


def _checked_vselect(dataset: str, condition: str | None, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays = load_split(dataset, "V_select", condition)
    if split_sha256(arrays) != manifest["splits"]["V_select"]["sha256"]:
        raise RuntimeError(f"{dataset}/{condition or ''} V_select does not match its dataset.json hash")
    return arrays


def checked_forward(dataset: str, manifest: Mapping[str, Any]) -> tuple[ForwardFn, str]:
    """The dataset's numpy forward.py, refused unless its sha256 is the one ``manifest`` records."""
    forward, digest = load_forward_no_cache(forward_path(dataset))
    if digest != manifest["forward_sha256"]:
        raise RuntimeError(f"{dataset}/forward.py sha256 {digest} != dataset.json {manifest['forward_sha256']}")
    return forward, digest


def numpy_torch_parity(arch: str, manifest: Mapping[str, Any], params: Params, x: np.ndarray, forward: ForwardFn) -> float:
    """Max abs difference between the numpy forward and CPU float32 torch eval logits on images ``x``."""
    model = build_model(
        arch, manifest["image_shape"], len(manifest["class_names"]), manifest["norm"]["mean"], manifest["norm"]["std"]
    )
    model = load_params_into(model, params).eval()
    with torch.no_grad():
        reference = model(torch.from_numpy(x)).numpy().astype(np.float64)
    return float(np.max(np.abs(forward(dict(params), x).astype(np.float64) - reference)))


def _check_one(
    row: Mapping[str, Any],
    source: Path,
    *,
    dataset: str,
    condition: str | None,
    manifest: Mapping[str, Any],
    forward: ForwardFn,
    forward_sha: str,
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
    digest = params_sha256(params)
    private = read_json(folder / "private.json")
    expected = {
        "model_id": model_id,
        "dataset": dataset,
        "condition": condition,
        "seed": row["seed"],
        "arch": row["arch"],
        "role": row["role"],
        "params_sha256": row["params_sha256"],
        "forward_sha256": forward_sha,
    }
    found = {key: private.get(key) for key in expected}
    if digest != row["params_sha256"] or found != expected:
        raise ValueError(f"{model_id}: private.json or model.npz disagrees with the index ({found} vs {expected})")
    if private.get("smoke"):
        raise ValueError(f"{model_id} is a smoke-test model and is never imported")
    train_split = private["train_split"]
    if private["train_split_sha256"] != manifest["splits"][train_split]["sha256"]:
        raise ValueError(f"{model_id}: trained on a {train_split} split that is not the local one")
    spec = read_json(folder / "model_spec.json")
    rebuilt = build_model_spec(
        model_id,
        row["arch"],
        manifest["image_shape"],
        manifest["norm"]["mean"],
        manifest["norm"]["std"],
        manifest["class_names"],
        params,
    )
    check_public(spec)
    if spec != json.loads(json.dumps(rebuilt)):
        raise ValueError(f"{model_id}: model_spec.json differs from the spec rebuilt from its parameters")
    parity = numpy_torch_parity(row["arch"], manifest, params, vselect["x"][:PARITY_N], forward)
    if not parity <= PARITY_TOL:
        raise ValueError(f"{model_id}: local numpy vs torch parity {parity} > {PARITY_TOL}")
    pred = predict(forward, params, vselect["x"], batch_size=1024)
    accuracy = float(np.mean(pred == vselect["y"]))
    if accuracy != row["vselect_acc"] or accuracy != private["final_vselect_acc"]:
        raise ValueError(
            f"{model_id}: local V_select accuracy {accuracy} != exported {row['vselect_acc']} "
            f"(private {private['final_vselect_acc']})"
        )
    return {"model_id": model_id, "local_parity_max_abs_diff": parity, "local_vselect_acc": accuracy}


GPU4090_HOST = "4090"
"""``meta.json`` host of a run on the 4090 fallback (``scripts/data_jobs/run_4090.sh``); NSCC runs have no host."""
_GPU4090_ENVIRONMENT_KEYS = ("venv", "python_version", "venv_freeze_sha256")
_RECORDED_PACKAGES = ("torch", "numpy")


def run_record(run_dir: Path) -> dict[str, Any]:
    """Where and how a fetched run trained its models: an NSCC run (job, request, container) or a 4090 fallback run
    (``gpu4090_record``), with its exit status."""
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "meta.json")
    exit_record = read_json(run_dir / "exit.json")
    if exit_record["exit_code"] != 0:
        raise ValueError(f"{run_dir.name} exited with {exit_record['exit_code']}; its models are not imported")
    if meta.get("host") is not None:
        if meta["host"] != GPU4090_HOST:
            raise ValueError(f"{run_dir}/meta.json: unknown host {meta['host']!r}")
        return gpu4090_record(run_dir, meta, exit_record)
    container = None
    for line in (run_dir / "environment.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64 and parts[1].endswith(".sif"):
            container = {"image": parts[1], "sha256": parts[0]}
    if container is None:
        raise ValueError(f"{run_dir}/environment.txt does not record the container image hash")
    return {
        "nscc_run": meta["run_name"],
        "pbs_job_id": (run_dir / "job_id.txt").read_text().strip(),
        "request": meta["request"],
        "resources": {"select": meta["select"], "walltime": meta["walltime"]},
        "elapsed_s": exit_record["elapsed_s"],
        "container": container,
        "code_digest": meta["code_digest"],
    }


def gpu4090_record(run_dir: Path, meta: Mapping[str, Any], exit_record: Mapping[str, Any]) -> dict[str, Any]:
    """The run record of a 4090 fallback run: run name, the reason NUS did not serve it, the venv and its packages.

    environment.txt states the venv, its Python version and the sha256 of venv_freeze.txt (the venv's full
    ``pip freeze``), and the file must match it. The reason must be named: the fallback is used only when NUS cannot
    serve a job.
    """
    found: dict[str, str] = {}
    for line in (Path(run_dir) / "environment.txt").read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0] in _GPU4090_ENVIRONMENT_KEYS:
            found[parts[0]] = parts[1].strip()
    missing = [key for key in _GPU4090_ENVIRONMENT_KEYS if key not in found]
    if missing:
        raise ValueError(f"{run_dir}/environment.txt does not record {missing}")
    freeze_path = Path(run_dir) / "venv_freeze.txt"
    if file_sha256(freeze_path) != found["venv_freeze_sha256"]:
        raise ValueError(f"{freeze_path}: sha256 differs from environment.txt")
    reason = meta.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{run_dir}/meta.json names no reason for the 4090 fallback")
    packages = {}
    for line in freeze_path.read_text().splitlines():
        name, _, version = line.partition("==")
        if name.strip().lower() in _RECORDED_PACKAGES:
            packages[name.strip().lower()] = version.strip()
    return {
        "gpu4090_run": meta["run_name"],
        "host": GPU4090_HOST,
        "fallback_reason": reason,
        "request": None,
        "elapsed_s": exit_record["elapsed_s"],
        "environment": {**found, "packages": packages},
        "code_digest": meta["code_digest"],
    }


def annotate_import(folder: Path, block: Mapping[str, Any]) -> None:
    """Add an ``import`` block (training environment and import checks) to a model's private.json."""
    path = Path(folder) / "private.json"
    private = read_json(path)
    if "import" in private:
        raise ValueError(f"{path} already has an import block")
    write_json(path, {**private, "import": dict(block)})


def models_source(run_dir: Path, dataset: str, condition: str | None) -> Path:
    """Models directory of a fetched run's private store."""
    root = Path(run_dir) / "private" / dataset
    return (root if condition is None else root / "conditions" / condition) / "models"


def import_models(
    dataset: str, condition: str | None, run_dir: Path, note: str | None = None
) -> list[dict[str, Any]]:
    """Check every model of a fetched NSCC run and copy them into the local store, then index them.

    The models are read from ``models_source(run_dir, dataset, condition)``. Each imported private.json gains an
    ``import`` block: the run record (job, request, container image and hash), the training environment recorded at
    export (library versions, device), the local re-checks and ``note``. Refuses a model id already used under the
    dataset and a (role, seed) already in the local index. Nothing is copied unless every model passes.
    """
    record = run_record(run_dir)
    source = models_source(run_dir, dataset, condition)
    manifest = read_dataset_json(dataset, condition)
    forward, forward_sha = checked_forward(dataset, manifest)
    vselect = _checked_vselect(dataset, condition, manifest)
    rows = read_json(source / "index.json")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source}/index.json lists no models")
    local = read_index(dataset, condition)
    used_ids = all_model_ids(dataset)
    taken = {(entry["role"], entry["seed"]) for entry in local}
    checks = []
    for row in rows:
        if row["model_id"] in used_ids:
            raise ValueError(f"model id {row['model_id']} is already used under {dataset}")
        if (row["role"], row["seed"]) in taken:
            raise ValueError(f"{dataset}/{condition}: {row['role']} seed {row['seed']} is already in the local index")
        taken.add((row["role"], row["seed"]))
        checks.append(
            _check_one(
                row,
                source,
                dataset=dataset,
                condition=condition,
                manifest=manifest,
                forward=forward,
                forward_sha=forward_sha,
                vselect=vselect,
            )
        )
    root = models_dir(dataset, condition)
    root.mkdir(parents=True, exist_ok=True)
    for row, check in zip(rows, checks):
        staging = root / f".staging-{row['model_id']}"
        shutil.copytree(source / row["model_id"], staging)
        private = read_json(staging / "private.json")
        annotate_import(
            staging,
            {
                **record,
                "trained_with": {"versions": private["versions"], "device": private["device"]},
                "local_checks": check,
                "note": note,
            },
        )
        staging.rename(root / row["model_id"])
    write_index(dataset, local + [dict(row) for row in rows], condition)
    return checks


def load_indexed_models(
    dataset: str, selection: Mapping[str, tuple[int, ...]]
) -> list[tuple[dict[str, Any], Params]]:
    """The clean models of ``dataset`` named by ``selection`` (role -> seeds), each checked against its index hash.

    Rows are read from the dataset's models/index.json (never written here) and returned sorted by (role, seed);
    a requested (role, seed) without an index row is refused.
    """
    wanted = {(role, seed) for role, seeds in selection.items() for seed in seeds}
    entries = [entry for entry in read_index(dataset) if (entry["role"], entry["seed"]) in wanted]
    missing = wanted - {(entry["role"], entry["seed"]) for entry in entries}
    if missing:
        raise ValueError(f"{dataset}: no clean model for {sorted(missing)}")
    models = []
    for entry in sorted(entries, key=lambda e: (e["role"], e["seed"])):
        params = load_params(models_dir(dataset) / entry["model_id"] / "model.npz")
        if params_sha256(params) != entry["params_sha256"]:
            raise ValueError(f"{entry['model_id']}: parameters do not match the index")
        models.append((entry, params))
    return models


def evaluate_on_condition(
    dataset: str,
    condition: str,
    selection: Mapping[str, tuple[int, ...]],
    run_info: Mapping[str, Any],
) -> dict[str, Any]:
    """V_select accuracy of the dataset's clean models (``selection``: role -> seeds) under ``condition``.

    Writes ``models/base_eval.json`` of the condition. A rerun must reproduce the recorded rows exactly.
    """
    clean = read_dataset_json(dataset)
    manifest = read_dataset_json(dataset, condition)
    for key in ("forward_sha256", "class_names", "image_shape", "norm"):
        clean_value = clean[key] if key != "norm" else {k: clean["norm"][k] for k in ("mean", "std")}
        value = manifest[key] if key != "norm" else {k: manifest["norm"][k] for k in ("mean", "std")}
        if value != clean_value:
            raise ValueError(f"{condition}: {key} differs from the clean dataset")
    forward, forward_sha = checked_forward(dataset, manifest)
    vselect = _checked_vselect(dataset, condition, manifest)
    rows = []
    for entry, params in load_indexed_models(dataset, selection):
        pred = predict(forward, params, vselect["x"], batch_size=EVAL_BATCH_SIZE)
        rows.append(
            {
                **{key: entry[key] for key in INDEX_KEYS if key != "vselect_acc"},
                "vselect_acc": float(np.mean(pred == vselect["y"])),
                "clean_vselect_acc": entry["vselect_acc"],
                "parity_max_abs_diff": numpy_torch_parity(
                    entry["arch"], manifest, params, vselect["x"][:PARITY_N], forward
                ),
            }
        )
    payload = {
        "schema": BASE_EVAL_SCHEMA,
        "dataset": dataset,
        "condition": condition,
        "split": "V_select",
        "split_sha256": manifest["splits"]["V_select"]["sha256"],
        "forward_sha256": forward_sha,
        "definition": "numpy forward.py argmax accuracy on all V_select images of the condition; parity on the first "
        f"{PARITY_N} against CPU float32 torch",
        "models": rows,
    }
    path = models_dir(dataset, condition) / BASE_EVAL_FILE
    if path.exists():
        recorded = read_json(path)
        if _identity(recorded) != _identity(json.loads(json.dumps(payload))):
            raise RuntimeError(f"{path} exists and differs from a fresh evaluation")
        return recorded
    write_json(path, {**payload, "created_by": dict(run_info)})
    return payload


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The fields a rerun must reproduce exactly (parity values are float diagnostics and may vary in the last bits)."""
    rows = [{key: value for key, value in row.items() if key != "parity_max_abs_diff"} for row in payload["models"]]
    return {**{key: value for key, value in payload.items() if key not in ("created_by", "models")}, "models": rows}
