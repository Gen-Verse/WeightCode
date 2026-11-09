"""Train, import and report the E9 ResNet-20 students (``weightcode.models.resnet20``, ``weightcode.models.fscil_export``).

Groups:

* ``train``: for each seed, ``--role fscil_base`` (the train split, classes 0-59, cross-entropy over all 100 outputs)
  or ``--role fscil_allclass`` (train + ref_pool + V_evidence, all 100 classes), under CEC's base-stage recipe
  (``FscilTrainConfig``). The job needs dataset.json and the role's training splits only. Each student goes to the
  pending directory (``pending_models_dir``) with a pending index row; nothing enters the fewshot store here.
* ``import``: the students of a fetched NSCC run (``import_fscil_models``: parity, device check and readouts with the
  local forward.py, then the fewshot store and its index).
* ``report``: the local fewshot index with each student's import readouts, parity and final training numbers.

``--smoke-epochs N`` (with ``--smoke-rows M``, the first M training rows) trains into ``--models-root`` (required),
marks every student smoke (the importer refuses them) and never writes the pending index of the store.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/train_fscil.py --group train --role fscil_base --seeds 100 --device cuda
    .venv/bin/python scripts/train_fscil.py --group import --run-dir $WEIGHTCODE_PRIVATE/nscc_runs/RUN --note "..."
    .venv/bin/python scripts/train_fscil.py --group report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch

from weightcode.data import cifar100_fscil as C
from weightcode.data.store import read_dataset_json, read_json, write_json
from weightcode.models import resnet20 as R
from weightcode.models.export import fewshot_models_dir, read_fewshot_index
from weightcode.models.fscil_export import (
    FSCIL_ROLES,
    export_fscil_model,
    import_fscil_models,
    normalization,
    pending_models_dir,
    training_rows,
)
from weightcode.models.runinfo import run_info
from weightcode.models.train import configure_determinism

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_EVERY = 10


def train_group(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available")
    smoke = args.smoke_epochs is not None
    if smoke and args.models_root is None:
        raise SystemExit("--smoke-epochs needs --models-root (smoke students never enter the store)")
    if args.smoke_rows is not None and not smoke:
        raise SystemExit("--smoke-rows is for smoke runs only")
    manifest = read_dataset_json(C.DATASET)
    mean, std = normalization(manifest)
    x, y, used = training_rows(args.role, manifest)
    if args.smoke_rows is not None:
        x, y = x[: args.smoke_rows], y[: args.smoke_rows]
    info = run_info(REPO_ROOT, args.git_state_file)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    info["device"] = f"{name} ({device})"
    config = R.FscilTrainConfig()
    if smoke:
        config = dataclasses.replace(config, epochs=args.smoke_epochs)
    root = Path(args.models_root) if args.models_root is not None else pending_models_dir()
    index_file = root / "index.json"
    previous = read_json(index_file) if index_file.exists() else []
    entries: list[dict[str, Any]] = []
    for seed in args.seeds:
        determinism = configure_determinism(seed)

        def log(entry: dict[str, Any], seed: int = seed) -> None:
            if entry["epoch"] % LOG_EVERY == 0 or entry["epoch"] in (1, config.epochs):
                print(json.dumps({"role": args.role, "seed": seed, **entry}), flush=True)

        result = R.train_resnet20(config, seed, x, y, C.N_CLASSES, mean, std, device, log_fn=log)
        extra: dict[str, Any] = {"smoke": True, "smoke_rows": args.smoke_rows} if smoke else {}
        entry = export_fscil_model(manifest=manifest, role=args.role, seed=seed, config=config, result=result,
                                   training_splits=used, x_train=x, determinism=determinism, run_info=info,
                                   models_root=root, extra_private=extra)
        entries.append(entry)
        write_json(index_file, previous + entries)
        print(json.dumps({"role": args.role, "seed": seed, "model_id": entry["model_id"],
                          "params_sha256": entry["params_sha256"], **result.final,
                          "final_train_loss_augmented": result.log[-1]["train_loss"],
                          "wall_time_s": round(result.wall_time_s, 1)}), flush=True)
    return 0


def import_group(args: argparse.Namespace) -> int:
    print(json.dumps(import_fscil_models(args.run_dir, args.note), indent=2))
    return 0


def report_group(args: argparse.Namespace) -> int:
    rows = []
    for entry in read_fewshot_index(C.DATASET):
        private = read_json(fewshot_models_dir(C.DATASET) / entry["model_id"] / "private.json")
        checks = private["import"]["local_checks"]
        rows.append({**entry, "final_train": private["final_train"], "readouts": checks["readouts"],
                     "parity_float64_max_rel_diff": checks["parity"]["float64_max_rel_diff"],
                     "parity_float32_max_abs_diff": checks["parity"]["max_abs_diff"],
                     "device_check": checks["device_check"], "wall_time_s": private["wall_time_s"],
                     "layer_norms": private["layer_norms"]})
    print(json.dumps(rows, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=("train", "import", "report"), required=True)
    parser.add_argument("--role", choices=tuple(FSCIL_ROLES), default="fscil_base")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--git-state-file", type=Path, help="git_state.json of a job's code copy (run_info)")
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--smoke-rows", type=int)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--note")
    args = parser.parse_args()
    if args.group == "import" and args.run_dir is None:
        raise SystemExit("--group import needs --run-dir")
    return {"train": train_group, "import": import_group, "report": report_group}[args.group](args)


if __name__ == "__main__":
    sys.exit(main())
