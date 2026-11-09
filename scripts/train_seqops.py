"""Train, export, import and report the seqops students (PREREG E5; ``weightcode.models.train_seq``).

Groups:

* ``train``: for each seed, the default size (d 64) under the fixed recipe, then the size check (exact match on old
  operators over the train-dev rows, at least SIZE_CHECK_EM). A student that passes is exported as role ``base``.
  One that misses is exported as role ``dev`` with its size check, and the next pre-registered size (d 128) is
  trained and exported as ``base`` whatever its own check says (the size steps up once only). ``--size-step``
  trains one step alone, so each size can run as its own job. The held-out keys are
  re-derived from the code and checked against dataset.json (``weightcode.data.seqops_build.checked_exclusion``), so the
  job needs only dataset.json, forward.py and V_select.
* ``import``: the students of a fetched NSCC run (``weightcode.models.seq_export.import_seq_models``).
* ``report``: the local index with each student's V_select exact match, size check and weakest operator pair.

``--smoke-steps N`` trains N steps per size into ``--models-root`` (required with it), marks every model smoke (the
importer refuses them) and never writes the store's index.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/train_seqops.py --group train --seeds 100 --device cuda
    .venv/bin/python scripts/train_seqops.py --group import --run-dir $WEIGHTCODE_PRIVATE/nscc_runs/RUN_NAME --note "..."
    .venv/bin/python scripts/train_seqops.py --group report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch

from weightcode.data import seqops as S
from weightcode.data.seqops_build import checked_exclusion
from weightcode.data.store import read_dataset_json, write_json
from weightcode.models.export import read_index, write_index
from weightcode.models.importer import _checked_vselect
from weightcode.models.runinfo import run_info
from weightcode.models.seq_export import export_seq_model, import_seq_models
from weightcode.models.train import configure_determinism
from weightcode.models.train_seq import (
    SIZE_CHECK_EM,
    SIZE_STEPS,
    SeqTrainConfig,
    em_by_operator,
    torch_predict,
    train_dev_rows,
    train_seq,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def size_check(model: torch.nn.Module, rows: dict, device: torch.device) -> dict[str, Any]:
    """Exact match of the student on the train-dev rows (PREREG E5 size check)."""
    readout = em_by_operator(rows, torch_predict(model, rows["x"], device))
    return {"rows": "train-dev rows, default_rng([split_seed, 1106]), held-out keys refused", "n": int(len(rows["y"])),
            "threshold_em": SIZE_CHECK_EM, "passed": readout["em"] >= SIZE_CHECK_EM, **readout}


def train_group(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available")
    if args.smoke_steps is not None and args.models_root is None:
        raise SystemExit("--smoke-steps needs --models-root (smoke models never enter the store)")
    manifest = read_dataset_json(S.DATASET)
    exclude = checked_exclusion(manifest)
    vselect = _checked_vselect(S.DATASET, None, manifest)
    rows = train_dev_rows(exclude)
    info = run_info(REPO_ROOT, args.git_state_file)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    info["device"] = f"{name} ({device})"
    entries = []
    steps = range(len(SIZE_STEPS)) if args.size_step is None else [args.size_step]
    for seed in args.seeds:
        for step_index in steps:
            config = SIZE_STEPS[step_index]
            if args.smoke_steps is not None:
                config = dataclasses.replace(config, steps=args.smoke_steps, warmup_steps=min(config.warmup_steps,
                                             max(1, args.smoke_steps // 10)), log_every=max(1, args.smoke_steps // 4))
            determinism = configure_determinism(seed)

            def monitor(model: torch.nn.Module, _rows: dict = rows) -> dict[str, Any]:
                readout = em_by_operator(_rows, torch_predict(model, _rows["x"], device))
                return {"train_dev_em": readout["em"], "train_dev_min_pair_em": readout["min_pair_em"]}

            result = train_seq(config, seed, exclude, device, monitor=monitor)
            check = size_check(result.model, rows, device)
            last_step = step_index == len(SIZE_STEPS) - 1
            role = "base" if check["passed"] or last_step else "dev"
            extra: dict[str, Any] = {"size_step": step_index, "size_steps": [c.as_dict() for c in SIZE_STEPS]}
            if args.smoke_steps is not None:
                extra["smoke"] = True
            entry = export_seq_model(manifest=manifest, role=role, seed=seed, config=config, result=result,
                                     size_check=check, determinism=determinism, run_info=info, vselect=vselect,
                                     models_root=args.models_root, extra_private=extra)
            entries.append(entry)
            print(json.dumps({"seed": seed, "size_step": step_index, "d_model": config.d_model, "role": role,
                              "size_check_em": check["em"], "min_pair_em": check["min_pair_em"],
                              "vselect_acc": entry["vselect_acc"], "model_id": entry["model_id"],
                              "reserved_rows_unchanged": result.reserved_rows_unchanged,
                              "wall_time_s": round(result.wall_time_s, 1)}), flush=True)
            if check["passed"]:
                break
    if args.models_root is not None:
        index_file = Path(args.models_root) / "index.json"
        previous = json.loads(index_file.read_text()) if index_file.exists() else []
        write_json(index_file, previous + entries)
    else:
        write_index(S.DATASET, read_index(S.DATASET) + entries)
    return 0


def import_group(args: argparse.Namespace) -> int:
    checks = import_seq_models(args.run_dir, args.note)
    print(json.dumps(checks, indent=2))
    return 0


def report_group(_: argparse.Namespace) -> int:
    from weightcode.data.store import models_dir, read_json

    rows = []
    for entry in read_index(S.DATASET):
        private = read_json(models_dir(S.DATASET) / entry["model_id"] / "private.json")
        check = private["size_check"]
        rows.append({**entry, "d_model": private["model_config"]["d_model"], "size_check_em": check["em"],
                     "size_check_passed": check["passed"], "vselect_min_pair_em": private["vselect_by_operator"]["min_pair_em"],
                     "parity": private["parity"]["max_abs_diff"], "reserved_rows_unchanged": private["reserved_rows_unchanged"]})
    print(json.dumps(rows, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=("train", "import", "report"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--git-state-file", type=Path, help="git state recorded at submission (code copy without .git)")
    parser.add_argument("--models-root", type=Path, help="write models here instead of the store (smoke runs)")
    parser.add_argument("--smoke-steps", type=int, help="train only this many steps per size (smoke run)")
    parser.add_argument("--size-step", type=int, choices=range(len(SIZE_STEPS)),
                        help="train only this size step: 0 (d 64; base if it passes the size check, else dev) or 1 "
                        "(d 128, base); default: 0, then 1 if 0 misses")
    parser.add_argument("--run-dir", type=Path, help="fetched NSCC run directory (--group import)")
    parser.add_argument("--note", help="note stored in each imported model's import block")
    args = parser.parse_args()
    if args.group == "train":
        return train_group(args)
    if args.group == "import":
        if args.run_dir is None:
            raise SystemExit("--group import needs --run-dir")
        return import_group(args)
    return report_group(args)


if __name__ == "__main__":
    sys.exit(main())
