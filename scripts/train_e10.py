"""Train the PREREG E10 ResNet-20 students (``weightcode.models.resnet20a``, ``weightcode.models.e10_export``).

Group ``train``: for each seed, He et al.'s CIFAR recipe (``HeTrainConfig``) on E1's 40,000 cifar10 train rows; each
student goes to the pending directory with a pending index row. The job needs cifar10's dataset.json and
splits/train.npz only.

``--smoke-epochs N`` (with ``--smoke-rows M``, the first M training rows) trains into ``--models-root`` (required)
and marks every student smoke; the import refuses them.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/train_e10.py --group train --seeds 100 101 102 103 104 105 --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch

from weightcode.data.store import read_json, write_json
from weightcode.models import e10_export as E
from weightcode.models import resnet20a as A
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
    x, y, used, manifest = E.training_rows()
    mean, std = E.normalization(manifest)
    if args.smoke_rows is not None:
        x, y = x[: args.smoke_rows], y[: args.smoke_rows]
    info = run_info(REPO_ROOT, args.git_state_file)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    info["device"] = f"{name} ({device})"
    config = A.HeTrainConfig()
    if smoke:
        config = dataclasses.replace(config, epochs=args.smoke_epochs)
    root = Path(args.models_root) if args.models_root is not None else E.pending_models_dir()
    index_file = root / "index.json"
    previous = read_json(index_file) if index_file.exists() else []
    entries: list[dict[str, Any]] = []
    for seed in args.seeds:
        determinism = configure_determinism(seed)

        def log(entry: dict[str, Any], seed: int = seed) -> None:
            if entry["epoch"] % LOG_EVERY == 0 or entry["epoch"] in (1, config.epochs):
                print(json.dumps({"seed": seed, **entry}), flush=True)

        result = A.train_resnet20a(config, seed, x, y, E.N_CLASSES, mean, std, device, log_fn=log)
        extra: dict[str, Any] = {"smoke": True, "smoke_rows": args.smoke_rows} if smoke else {}
        entry = E.export_pending(manifest=manifest, seed=seed, config=config, result=result, training_splits=used,
                                 x_train=x, determinism=determinism, run_info=info, models_root=root,
                                 extra_private=extra)
        entries.append(entry)
        write_json(index_file, previous + entries)
        print(json.dumps({"seed": seed, "model_id": entry["model_id"], "params_sha256": entry["params_sha256"],
                          **result.final, "final_train_loss_augmented": result.log[-1]["train_loss"],
                          "wall_time_s": round(result.wall_time_s, 1)}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=("train",), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(E.SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--git-state-file", type=Path, help="git_state.json of a job's code copy (run_info)")
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--smoke-rows", type=int)
    parser.add_argument("--models-root", type=Path)
    args = parser.parse_args()
    return {"train": train_group}[args.group](args)


if __name__ == "__main__":
    sys.exit(main())
