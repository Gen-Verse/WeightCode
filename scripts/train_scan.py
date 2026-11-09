"""Train, export, import and report the E13 SCAN students (PREREG E13; ``weightcode.models.train_scan``).

Groups:

* ``train``: per condition and seed, size step 0 (d 128, 4 layers, MLP 512) under the fixed recipe, then the gate
  (exact match on the 1000 old V_select rows, at least GATE_EM). A student that passes is exported as role ``base``
  and the condition stops there. One that misses is exported as role ``dev``, and size step 1 (d 256, 6 layers,
  MLP 1024) is trained and exported, ``base`` only if it passes the gate (PREREG E13: if neither passes, the
  condition is reported and not run). ``--size-step`` trains one step alone. The job reads the condition's
  dataset.json, the dataset's dataset.json and forward.py, train.npz and old_select.npz, each hash-checked; no
  new-part row is read.
* ``import``: the students of one condition of a fetched NSCC run (``weightcode.models.scan_export.import_scan_models``).
* ``report``: each condition's index with the gate, parity and, for a base, headroom (exact match on new V_select,
  at most HEADROOM_EM_MAX), read locally.

``--smoke-epochs N`` trains N epochs per size into ``--models-root`` (required with it), marks every model smoke (the
importer refuses them) and never writes the store's index.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/train_scan.py --group train --condition addprim_jump --seeds 100 --device cuda
    .venv/bin/python scripts/train_scan.py --group import --condition addprim_jump --run-dir RUN --note "..."
    .venv/bin/python scripts/train_scan.py --group report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch

from weightcode.data import scan as S
from weightcode.data.scan_build import checked_split
from weightcode.data.store import read_dataset_json, write_json
from weightcode.models.export import read_index, write_index
from weightcode.models.runinfo import run_info
from weightcode.models.scan_export import export_scan_model, import_scan_models
from weightcode.models.train import configure_determinism
from weightcode.models.train_scan import GATE_EM, HEADROOM_EM_MAX, SIZE_STEPS, em_summary, monitor_rows, torch_predict, train_scan

REPO_ROOT = Path(__file__).resolve().parents[1]


def train_group(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available")
    if args.smoke_epochs is not None and args.models_root is None:
        raise SystemExit("--smoke-epochs needs --models-root (smoke models never enter the store)")
    base_manifest = read_dataset_json(S.DATASET)
    info = run_info(REPO_ROOT, args.git_state_file)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    info["device"] = f"{name} ({device})"
    for condition in args.condition:
        manifest = read_dataset_json(S.DATASET, condition)
        train = checked_split(condition, "train")
        old_select = checked_split(condition, "old_select")
        watch = {key: value[monitor_rows(len(train["y"]))] for key, value in train.items()}
        entries = []
        steps = range(len(SIZE_STEPS)) if args.size_step is None else [args.size_step]
        for seed in args.seeds:
            for step_index in steps:
                config = SIZE_STEPS[step_index]
                if args.smoke_epochs is not None:
                    config = dataclasses.replace(config, epochs=args.smoke_epochs, log_every_epochs=1,
                                                 warmup_steps=min(config.warmup_steps, 10))
                determinism = configure_determinism(seed)

                def monitor(model: torch.nn.Module, _rows: dict = watch) -> dict[str, Any]:
                    right = S.row_correct(_rows["y"], torch_predict(model, _rows["x"], device))
                    return {"train_monitor_em": float(right.mean())}

                result = train_scan(config, seed, train, device, monitor=monitor)
                extra: dict[str, Any] = {"size_steps": [c.as_dict() for c in SIZE_STEPS],
                                         "train_monitor": f"exact match on {len(watch['y'])} fixed training rows "
                                         "(default_rng([split_seed, 1311]))"}
                if args.smoke_epochs is not None:
                    extra["smoke"] = True
                root = None if args.models_root is None else Path(args.models_root) / condition
                entry = export_scan_model(condition=condition, manifest=manifest, base_manifest=base_manifest,
                                          seed=seed, size_step=step_index, config=config, result=result,
                                          determinism=determinism, run_info=info, old_select=old_select,
                                          models_root=root, extra_private=extra)
                entries.append(entry)
                print(json.dumps({"condition": condition, "seed": seed, "size_step": step_index,
                                  "d_model": config.d_model, "n_layers": config.n_layers, "role": entry["role"],
                                  "old_vselect_em": entry["vselect_acc"], "model_id": entry["model_id"],
                                  "steps": result.steps, "wall_time_s": round(result.wall_time_s, 1)}), flush=True)
                if entry["role"] == "base":
                    break
        if args.models_root is not None:
            index_file = Path(args.models_root) / condition / "index.json"
            previous = json.loads(index_file.read_text()) if index_file.exists() else []
            write_json(index_file, previous + entries)
        else:
            write_index(S.DATASET, read_index(S.DATASET, condition) + entries, condition)
    return 0


def import_group(args: argparse.Namespace) -> int:
    if len(args.condition) != 1:
        raise SystemExit("--group import takes one --condition")
    print(json.dumps(import_scan_models(args.run_dir, args.condition[0], args.note), indent=2))
    return 0


def report_group(args: argparse.Namespace) -> int:
    from weightcode.data.store import forward_path, models_dir, read_json
    from weightcode.common.identity import load_params
    from weightcode.evalkit.patch_check import predict
    from weightcode.models.export import load_forward_no_cache

    forward, _ = load_forward_no_cache(forward_path(S.DATASET))
    rows = []
    for condition in args.condition:
        new_select = checked_split(condition, "V_select")
        for entry in read_index(S.DATASET, condition):
            folder = models_dir(S.DATASET, condition) / entry["model_id"]
            private = read_json(folder / "private.json")
            row = {"condition": condition, **entry, "size_step": private["size_step"],
                   "d_model": private["model_config"]["d_model"], "n_layers": private["model_config"]["n_layers"],
                   "gate_em": private["gate"]["em"], "gate_passed": private["gate"]["passed"],
                   "parity_max_abs_diff": private["parity"]["max_abs_diff"],
                   "parity_float64_max_rel_diff": private["parity"]["float64"]["max_rel_diff"]}
            if entry["role"] == "base":
                pred = predict(forward, load_params(folder / "model.npz"), new_select["x"], 1024)
                summary = em_summary(new_select, pred)
                row["headroom"] = {"new_vselect_em": summary["em"], "new_vselect_token_acc": summary["token_acc"],
                                   "threshold_em_max": HEADROOM_EM_MAX,
                                   "passed": summary["em"] <= HEADROOM_EM_MAX, "n": summary["n"],
                                   "by_construction": summary["by_construction"]}
            rows.append(row)
    print(json.dumps(rows, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=("train", "import", "report"), required=True)
    parser.add_argument("--condition", choices=S.CONDITIONS, action="append")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--git-state-file", type=Path, help="git state recorded at submission (code copy without .git)")
    parser.add_argument("--models-root", type=Path, help="write models here instead of the store (smoke runs)")
    parser.add_argument("--smoke-epochs", type=int, help="train only this many epochs per size (smoke run)")
    parser.add_argument("--size-step", type=int, choices=range(len(SIZE_STEPS)),
                        help=f"train only this size step (0: d 128; 1: d 256, 6 layers); default 0, then 1 if 0 "
                        f"misses the gate (exact match {GATE_EM} on old V_select)")
    parser.add_argument("--run-dir", type=Path, help="fetched NSCC run directory (--group import)")
    parser.add_argument("--note", help="note stored in each imported model's import block")
    args = parser.parse_args()
    args.condition = list(dict.fromkeys(args.condition or S.CONDITIONS))
    if args.group == "train":
        return train_group(args)
    if args.group == "import":
        if args.run_dir is None:
            raise SystemExit("--group import needs --run-dir")
        return import_group(args)
    return report_group(args)


if __name__ == "__main__":
    sys.exit(main())
