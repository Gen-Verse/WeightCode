"""Train the PREREG E12 ResNet-18 students (``weightcode.models.resnet18up``, ``weightcode.models.e12_export``).

Group ``train``: for each seed, PREREG E12's fine-tuning recipe (``E12TrainConfig``) from torchvision's IMAGENET1K_V1
ResNet-18 (fc replaced, seeded) on E1's 40,000 cifar10 train rows at the upsampled size ``--resize``; each student
goes to the pending directory with a pending index row, with its agreement with ``torchvision.models.resnet18`` on
the first training rows. The job needs cifar10's dataset.json, splits/train.npz and pretrained/resnet18-f37072fd.pth
only.

``--concurrent`` trains the seeds in parallel child processes on the same device (one per seed); each child's log
goes to ``<root>/logs/``, and the parent writes the pending index once every child has exited 0. ``--duplicate-seed S`` (with ``--concurrent``) adds one more child that trains seed S
again into ``<root>/duplicate/`` and compares its params sha256 with the first: the determinism check of the timing
smoke.

``--smoke-epochs N`` (with ``--smoke-rows M``, the first M training rows) trains into ``--models-root`` (required)
and marks every student smoke; the import refuses them.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/train_e12.py --group train --resize 128 --seeds 100 101 --concurrent --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from weightcode.data.store import read_json, write_json
from weightcode.models import e12_export as E
from weightcode.models import resnet18up as M
from weightcode.models.runinfo import run_info
from weightcode.models.train import configure_determinism

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_EVERY = 1
TORCHVISION_CHECK_ROWS = 64
TORCHVISION_MAX_ABS_DIFF = 1e-4
"""The student module must equal ``torchvision.models.resnet18`` holding its tensors (fp32 on the device)."""


def _root(args: argparse.Namespace) -> Path:
    return Path(args.models_root) if args.models_root is not None else E.pending_models_dir()


def train_one(args: argparse.Namespace, seed: int, root: Path) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available")
    smoke = args.smoke_epochs is not None
    x, y, used, manifest = E.training_rows()
    if args.smoke_rows is not None:
        x, y = x[: args.smoke_rows], y[: args.smoke_rows]
    info = run_info(REPO_ROOT, args.git_state_file)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    info["device"] = f"{name} ({device})"
    config = M.E12TrainConfig()
    if smoke:
        config = dataclasses.replace(config, epochs=args.smoke_epochs)
    model_config = M.ResNet18UpConfig(resize=args.resize)
    weights_path = E.weights_path()
    weights = {"name": M.WEIGHTS_NAME, "file": M.WEIGHTS_FILE, "url": M.WEIGHTS_URL,
               "sha256": M.file_sha256(weights_path), "store_path": E.WEIGHTS_RELPATH}
    if weights["sha256"] != M.WEIGHTS_SHA256:
        raise SystemExit(f"{weights_path}: sha256 {weights['sha256']}, expected {M.WEIGHTS_SHA256}")
    determinism = configure_determinism(seed)

    def log(entry: dict[str, Any]) -> None:
        if entry["epoch"] % LOG_EVERY == 0 or entry["epoch"] in (1, config.epochs) or smoke:
            print(json.dumps({"seed": seed, **entry}), flush=True)

    result = M.train_resnet18up(config, model_config, seed, x, y, weights_path, device, log_fn=log)
    check = M.torchvision_reference_diff(result.model, x[:TORCHVISION_CHECK_ROWS])
    check["rows"] = f"the first {TORCHVISION_CHECK_ROWS} training rows"
    if check["max_abs_diff"] is None or not check["max_abs_diff"] <= TORCHVISION_MAX_ABS_DIFF:
        raise SystemExit(f"the student differs from torchvision resnet18: {check}")
    extra: dict[str, Any] = {"smoke": True, "smoke_rows": args.smoke_rows} if smoke else {}
    entry = E.export_pending(manifest=manifest, seed=seed, config=config, model_config=model_config, result=result,
                             training_splits=used, x_train=x, determinism=determinism, run_info=info,
                             weights=weights, torchvision_check=check, models_root=root, extra_private=extra)
    print(json.dumps({"seed": seed, "model_id": entry["model_id"], "params_sha256": entry["params_sha256"],
                      "resize": model_config.resize, **result.final,
                      "final_train_loss_augmented": result.log[-1]["train_loss"],
                      "mean_epoch_time_s": sum(e["epoch_time_s"] for e in result.log) / len(result.log),
                      "torchvision_max_abs_diff": check["max_abs_diff"],
                      "wall_time_s": round(result.wall_time_s, 1)}), flush=True)
    return entry


def _child_command(args: argparse.Namespace, seed: int, root: Path, row_file: Path) -> list[str]:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--group", "train", "--seeds", str(seed),
           "--resize", str(args.resize), "--device", args.device, "--models-root", str(root), "--row-file",
           str(row_file)]
    if args.git_state_file is not None:
        cmd += ["--git-state-file", str(args.git_state_file)]
    if args.smoke_epochs is not None:
        cmd += ["--smoke-epochs", str(args.smoke_epochs)]
    if args.smoke_rows is not None:
        cmd += ["--smoke-rows", str(args.smoke_rows)]
    return cmd


def train_concurrent(args: argparse.Namespace) -> int:
    root = _root(args)
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    rows_dir = root / "rows"
    logs.mkdir(exist_ok=True)
    rows_dir.mkdir(exist_ok=True)
    jobs = [(seed, root, rows_dir / f"{seed}.json", logs / f"train_{seed}.log") for seed in args.seeds]
    if args.duplicate_seed is not None:
        if args.duplicate_seed not in args.seeds:
            raise SystemExit("--duplicate-seed must be one of --seeds")
        dup = root / "duplicate"
        dup.mkdir(exist_ok=True)
        jobs.append((args.duplicate_seed, dup, rows_dir / f"{args.duplicate_seed}_duplicate.json",
                     logs / f"train_{args.duplicate_seed}_duplicate.log"))
    for _, _, row_file, _ in jobs:
        if row_file.exists():
            raise SystemExit(f"{row_file} exists; nothing started")
    started = time.time()
    procs = []
    for seed, target, row_file, log_file in jobs:
        handle = open(log_file, "w", encoding="utf-8")  # closed after the child exits
        procs.append((seed, row_file, log_file, handle,
                      subprocess.Popen(_child_command(args, seed, target, row_file), stdout=handle,
                                       stderr=subprocess.STDOUT, cwd=REPO_ROOT)))
    print(json.dumps({"concurrent": len(procs), "seeds": [p[0] for p in procs], "started": started}), flush=True)
    codes = []
    for seed, row_file, log_file, handle, proc in procs:
        code = proc.wait()
        handle.close()
        codes.append(code)
        tail = log_file.read_text(encoding="utf-8").strip().splitlines()[-1:] or [""]
        print(json.dumps({"seed": seed, "exit": code, "elapsed_s": round(time.time() - started, 1),
                          "log": str(log_file), "last_line": tail[0][:400]}), flush=True)
    if any(codes):
        print(f"children failed: {codes}; no index written", file=sys.stderr)
        return 1
    entries = [read_json(rows_dir / f"{seed}.json") for seed in args.seeds]
    index_file = root / "index.json"
    previous = read_json(index_file) if index_file.exists() else []
    write_json(index_file, previous + entries)
    if args.duplicate_seed is not None:
        first = next(e for e in entries if e["seed"] == args.duplicate_seed)
        second = read_json(rows_dir / f"{args.duplicate_seed}_duplicate.json")
        same = first["params_sha256"] == second["params_sha256"]
        print(json.dumps({"determinism_check": {"seed": args.duplicate_seed, "first": first["params_sha256"],
                                                "duplicate": second["params_sha256"], "equal": same}}), flush=True)
        if not same:
            print("the duplicate run differs: training is not deterministic under concurrency", file=sys.stderr)
            return 2
    print(json.dumps({"index": str(index_file), "n": len(entries), "elapsed_s": round(time.time() - started, 1)}),
          flush=True)
    return 0


def train_group(args: argparse.Namespace) -> int:
    smoke = args.smoke_epochs is not None
    if smoke and args.models_root is None:
        raise SystemExit("--smoke-epochs needs --models-root (smoke students never enter the store)")
    if args.smoke_rows is not None and not smoke:
        raise SystemExit("--smoke-rows is for smoke runs only")
    if args.concurrent:
        return train_concurrent(args)
    if args.duplicate_seed is not None:
        raise SystemExit("--duplicate-seed needs --concurrent")
    root = _root(args)
    if args.row_file is not None:  # a child of --concurrent: one seed, its row to a file, no index
        if len(args.seeds) != 1:
            raise SystemExit("--row-file takes exactly one seed")
        entry = train_one(args, args.seeds[0], root)
        with open(args.row_file, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return 0
    index_file = root / "index.json"
    previous = read_json(index_file) if index_file.exists() else []
    entries: list[dict[str, Any]] = []
    for seed in args.seeds:
        entries.append(train_one(args, seed, root))
        write_json(index_file, previous + entries)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=("train",), required=True)
    parser.add_argument("--resize", type=int, required=True, choices=M.RESIZES,
                        help="the upsampled size R chosen by PREREG E12's forward.py timing")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(E.SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument("--duplicate-seed", type=int)
    parser.add_argument("--git-state-file", type=Path, help="git_state.json of a job's code copy (run_info)")
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--smoke-rows", type=int)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--row-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    return {"train": train_group}[args.group](args)


if __name__ == "__main__":
    sys.exit(main())
