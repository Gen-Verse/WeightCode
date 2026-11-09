"""PREREG E14 students: the store's forward.py and dataset.json, export from the training job, the gate and the import.

A training job (``scripts/train_e14.py``) writes each student to ``$WEIGHTCODE_PRIVATE/e14_gotolocal/e14_pending/<model_id>/``
(model.npz and private.json, with the job's gate readings and its float32 torch logits on the first
``DEVICE_CHECK_ROWS`` training rows) and a pending index; nothing there is a store model. The import runs on the
workstation: params hash, parity of the store's forward.py against CPU torch rebuilds (float64 and float32) on
``parity_rows``, the device check, then the gate and the headroom read again with forward.py (greedy rollouts on
V_select old and new), and writes ``models/<model_id>/`` (model.npz, model_spec.json, private.json) and
``models/index.json``.

Gate (PREREG E14): success on the 1,000 old V_select episodes at least ``GATE`` (0.95). Headroom: success on the
1,000 new V_select episodes at most ``HEADROOM`` (0.20). Both are pre-checks, read before any instance.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.data import e14 as D
from weightcode.data.store import read_json, write_json
from weightcode.models import e14_policy as P
from weightcode.models.export import all_model_ids, new_model_id

PENDING_DIR = "e14_pending"
PRIVATE_SCHEMA = "weightcode.e14_model_private.v1"
DATASET_SCHEMA = "weightcode.e14_dataset.v1"
DEVICE_CHECK_ROWS = 64
DEVICE_MAX_ABS_LOGIT_DIFF = 1e-3
SOURCE_SEED = 100
ROLE = "e14_base"
GATE = 0.95
HEADROOM = 0.20
CHAIN = (("base", 50_000), ("base", 100_000), ("next", 100_000))
"""PREREG E14's order: 50,000 episodes; otherwise 100,000; then the next size (at 100,000). The first that passes the
gate is the base; if none passes, E14 is reported and not run."""
PARITY_TRAIN_ROWS = 1000


def pending_models_dir() -> Path:
    return D.store_dir() / PENDING_DIR


def forward_path() -> Path:
    return D.store_dir() / "forward.py"


def write_store_forward() -> str:
    """Write the store's forward.py (refuses one with other bytes); returns its sha256."""
    source = P.forward_source().encode("utf-8")
    digest = hashlib.sha256(source).hexdigest()
    path = forward_path()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"{path} exists with other bytes")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source)
    return digest


def load_store_forward() -> tuple[Any, str]:
    from weightcode.models.export import load_forward_no_cache

    return load_forward_no_cache(forward_path())


def wheels_record(wheel_dir: Path) -> dict[str, Any]:
    """sha256 of every pinned wheel in ``wheel_dir``, checked against ``weightcode.data.e14.WHEELS``."""
    found = {}
    for name, expected in D.WHEELS.items():
        digest = hashlib.sha256((wheel_dir / name).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"{name} hashes to {digest}, not the PyPI sha256 {expected}")
        found[name] = digest
    return found


def readout(forward: Any, params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Gate and headroom: greedy rollouts of forward.py under ``params`` on V_select old and new, with success, mean
    episode length and success by article ("the": one object matches; "a": several) and by object type."""
    fn = P.numpy_logits_fn(forward, params)
    out: dict[str, Any] = {}
    for mission in D.MISSIONS:
        seeds = D.SEED_RANGES[f"V_select_{mission}"].seeds
        result = D.rollouts(fn, mission, seeds)
        out[mission] = summarize_rollouts(result)
        out[mission]["success_sha256"] = hashlib.sha256(result.success.astype(np.uint8).tobytes()).hexdigest()
    out["gate"] = {"rule": f"old V_select success >= {GATE}", "value": out["old"]["success"],
                   "passed": out["old"]["success"] >= GATE}
    out["headroom"] = {"rule": f"new V_select success <= {HEADROOM}", "value": out["new"]["success"],
                       "passed": out["new"]["success"] <= HEADROOM}
    return out


def summarize_rollouts(result: D.RolloutResult) -> dict[str, Any]:
    """Success, n, mean length (all and successful), pickups, and success by article and by object type."""
    s = result.success
    out: dict[str, Any] = {"n": int(len(s)), "n_success": int(s.sum()), "success": float(s.mean()),
                           "mean_length": float(result.length.mean()),
                           "mean_length_success": float(result.length[s].mean()) if s.any() else None,
                           "episodes_with_pickup": int(np.count_nonzero(result.n_pickups))}
    out["by_article"] = {a: {"n": int((result.article == i).sum()),
                             "success": float(s[result.article == i].mean()) if (result.article == i).any() else None}
                         for i, a in enumerate(D.ARTICLES)}
    out["by_type"] = {t: {"n": int((result.obj_type == i).sum()),
                          "success": float(s[result.obj_type == i].mean()) if (result.obj_type == i).any() else None}
                      for i, t in enumerate(D.OLD_TYPES)}
    return out


def export_pending(*, seed: int, size: str, n_episodes: int, config: P.E14TrainConfig, result: P.E14TrainResult,
                   demos: Mapping[str, Any], x_train: np.ndarray, gate: Mapping[str, Any],
                   determinism: Mapping[str, Any], run_info: Mapping[str, Any], root: Path | None = None
                   ) -> dict[str, Any]:
    """Write one trained student to the pending directory and return its pending index row."""
    root = pending_models_dir() if root is None else Path(root)
    root.mkdir(parents=True, exist_ok=True)
    taken = {path.name.removeprefix(".staging-") for path in root.iterdir() if path.is_dir()}
    model_id = new_model_id(all_model_ids(D.DATASET) | taken)
    staging = root / f".staging-{model_id}"
    staging.mkdir()
    params = P.export_params(result.model)
    digest = save_params(staging / "model.npz", params)
    if params_sha256(load_params(staging / "model.npz")) != digest:
        raise RuntimeError("reloaded parameters hash differently")
    rows = np.ascontiguousarray(x_train[:DEVICE_CHECK_ROWS])
    private = {
        "schema": PRIVATE_SCHEMA, "model_id": model_id, "dataset": D.DATASET, "seed": int(seed), "arch": P.ARCH,
        "role": ROLE, "size": size, "model_config": P.SIZES[size].as_dict(), "training_config": config.as_dict(),
        "train_data": {"n_episodes": int(n_episodes), "n_rows": int(len(x_train)), **dict(demos)},
        "determinism": dict(determinism), "train_log": result.log, "final_train": dict(result.final),
        "reserved_rows_unchanged": result.reserved_rows_unchanged, "params_sha256": digest,
        "n_parameters": int(sum(v.size for v in params.values())), "n_tensors": len(params),
        "device_check": {"rows": f"the first {len(rows)} training rows", "device": "cpu",
                         "logits": [[float(v) for v in row] for row in P.eval_logits(result.model, rows)]},
        "job_gate": dict(gate), "wall_time_s": result.wall_time_s, **dict(run_info),
    }
    write_json(staging / "private.json", private)
    staging.rename(root / model_id)
    return {"model_id": model_id, "seed": int(seed), "size": size, "n_episodes": int(n_episodes),
            "params_sha256": digest, "job_gate_passed": bool(gate["gate"]["passed"])}


def parity_rows() -> np.ndarray:
    """The first ``PARITY_TRAIN_ROWS`` training rows and every row of the first 64 usable new V_evidence episodes
    (the rows that hold the reserved words)."""
    train = D.load_demos("train")
    new = D.load_demos("V_evidence_new")
    rows_new = new["x"][D.first_usable(new, 64)]
    return np.concatenate([train["x"][:PARITY_TRAIN_ROWS], rows_new]).astype(np.int64)


def models_dir() -> Path:
    return D.store_dir() / "models"


def read_models_index() -> list[dict[str, Any]]:
    path = models_dir() / "index.json"
    return list(read_json(path)) if path.exists() else []


def import_e14_models(run_dir: Path, note: str) -> list[dict[str, Any]]:
    """Import the pending students of a fetched training run (module docstring). Refuses a smoke student, an id
    already in the index, a params hash that differs from the pending index, a parity or device-check failure."""
    import torch

    from weightcode.models.parity import check_logit_parity

    if not note or not note.strip():
        raise ValueError("an import needs a note")
    pending = Path(run_dir) / "private" / D.DATASET / PENDING_DIR
    rows = read_json(pending / "index.json")
    forward, forward_sha = load_store_forward()
    x_par = parity_rows()
    train = D.load_demos("train")
    index = read_models_index()
    models_dir().mkdir(parents=True, exist_ok=True)
    imported = []
    for row in rows:
        source = pending / row["model_id"]
        params = load_params(source / "model.npz")
        if params_sha256(params) != row["params_sha256"]:
            raise RuntimeError(f"{source}/model.npz hash differs from the pending index")
        private = read_json(source / "private.json")
        if private.get("smoke"):
            raise RuntimeError(f"{source} is a smoke student")
        if any(r["model_id"] == row["model_id"] for r in index):
            raise RuntimeError(f"{row['model_id']} is already in the models index")
        config = P.config_from_params(params)
        ours = np.asarray(forward(params, x_par), dtype=np.float64)
        ref64 = P.eval_logits(P.build_policy(config, params, torch.float64), x_par)
        ref32 = P.eval_logits(P.build_policy(config, params, torch.float32), x_par)
        parity = {"rows": f"the first {PARITY_TRAIN_ROWS} training rows and the rows of the first 64 usable new "
                          f"V_evidence episodes ({len(x_par)} rows)", **check_logit_parity(ours, ref64, ref32),
                  "float32_argmax_disagreements": int(np.count_nonzero(ours.argmax(1) != ref32.argmax(1)))}
        device_logits = np.asarray(private["device_check"]["logits"], dtype=np.float64)
        local = np.asarray(forward(params, train["x"][:DEVICE_CHECK_ROWS].astype(np.int64)), dtype=np.float64)
        device = {"rows": private["device_check"]["rows"], "max_abs_diff": float(np.max(np.abs(local - device_logits))),
                  "argmax_disagreements": int(np.count_nonzero(local.argmax(1) != device_logits.argmax(1)))}
        if device["max_abs_diff"] > DEVICE_MAX_ABS_LOGIT_DIFF or device["argmax_disagreements"]:
            raise RuntimeError(f"{row['model_id']}: device check {device}")
        local_readout = readout(forward, params)
        job = private.get("job_gate", {})
        agreement = {m: job.get(m, {}).get("success_sha256") == local_readout[m]["success_sha256"] for m in D.MISSIONS}
        target = models_dir() / row["model_id"]
        staging = models_dir() / f".staging-{row['model_id']}"
        staging.mkdir()
        save_params(staging / "model.npz", params)
        write_json(staging / "model_spec.json", P.build_spec(row["model_id"], params))
        write_json(staging / "private.json", {
            **private, "store_role": "dev", "split_role": "dev",
            "import": {"run_dir": str(run_dir), "note": note, "forward_sha256": forward_sha,
                       "local_checks": {"parity": parity, "device_check": device}},
            "readouts": local_readout, "job_local_episode_agreement": agreement,
        })
        staging.rename(target)
        entry = {"model_id": row["model_id"], "seed": int(row["seed"]), "arch": P.ARCH, "size": row["size"],
                 "n_episodes": int(row["n_episodes"]), "role": "dev", "params_sha256": row["params_sha256"],
                 "gate_old_vselect": local_readout["old"]["success"], "gate_passed": local_readout["gate"]["passed"],
                 "headroom_new_vselect": local_readout["new"]["success"],
                 "headroom_passed": local_readout["headroom"]["passed"]}
        index.append(entry)
        write_json(models_dir() / "index.json", index)
        imported.append({**entry, "parity": parity, "device_check": device, "job_local_episode_agreement": agreement})
    return imported


def dataset_record(wheel_dir: Path, sealed: Mapping[str, Any]) -> dict[str, Any]:
    """dataset.json: the pinned environment and wheels, the levels, the encoding, the seed ranges and the sealed
    record's hash, the bot rule and the forward's hash."""
    return {
        "schema": DATASET_SCHEMA, "dataset": D.DATASET, "prereg": "PREREG E14 (main 9df26e9)",
        "environment": {"pinned": dict(D.PINNED), "installed": D.environment_versions(),
                        "wheels_sha256": wheels_record(wheel_dir)},
        "levels": {"old": {"name": D.OLD_LEVEL, "class": "minigrid.envs.babyai.goto.GoToLocal",
                           "kwargs": {"room_size": 8, "num_dists": 8}},
                   "new": {"name": D.NEW_LEVEL, "class": "weightcode.data.e14 GoToLocalPickup(GoToLocal)",
                           "gen_mission": "GoToLocal.gen_mission with PickupInstr(ObjDesc(obj.type, obj.color)) "
                                          "(strict False) in place of GoToInstr"},
                   "max_steps": D.MAX_STEPS},
        "encoding": D.encoding_record(),
        "seed_ranges": D.ranges_record(),
        "sealed_record": dict(sealed),
        "bot": "minigrid.utils.baby_ai_bot.BabyAIBot with vis_mask set True everywhere at the episode start",
        "forward_sha256": hashlib.sha256(P.forward_source().encode("utf-8")).hexdigest(),
    }


__all__ = ["CHAIN", "DATASET_SCHEMA", "DEVICE_CHECK_ROWS", "GATE", "HEADROOM", "PENDING_DIR", "PRIVATE_SCHEMA",
           "ROLE", "SOURCE_SEED", "dataset_record", "export_pending", "forward_path", "import_e14_models",
           "load_store_forward", "models_dir", "parity_rows", "pending_models_dir", "read_models_index", "readout",
           "summarize_rollouts", "wheels_record", "write_store_forward"]
