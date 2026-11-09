"""Download, split and store the WeightCode datasets (contract section 2).

For each dataset: download the official sets with torchvision into $WEIGHTCODE_PRIVATE/raw (sha256 of every raw file in
raw/provenance.json), split the official train set with split seed 20261103, write
$WEIGHTCODE_PRIVATE/<dataset>/splits/{train,V_select,V_evidence,V_check}.npz (Colored-MNIST adds train_uniform.npz),
the official test set to $WEIGHTCODE_SEALED/<dataset>/test.npz only, the numpy reference forward.py and dataset.json.

Deployment conditions (weightcode.data.conditions, CIFAR-10-C corruptions) are built from the stored clean splits and the
raw official test set: $WEIGHTCODE_PRIVATE/<dataset>/conditions/<condition>/{dataset.json,splits/*.npz} and
$WEIGHTCODE_SEALED/<dataset>/conditions/<condition>/test.npz.

The drawn-parameter conditions of contract section 2.1 (colormix_d1, blur5_d1, gamma_d1) are calibrated first, on
the train split only (--calibrate writes conditions/<condition>/calibration.json). The chosen severity is then
entered as the condition's constant in weightcode.data.conditions, and the normal build writes the splits. Every build
checks the calibration log against the registered parameters and severity.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/prepare_data.py --dataset fashion_mnist
    .venv/bin/python scripts/prepare_data.py --dataset fashion_mnist --verify   # re-derive and compare hashes
    .venv/bin/python scripts/prepare_data.py --dataset cifar10 --condition all  # contrast_s3, gaussian_noise_s3
    .venv/bin/python scripts/prepare_data.py --dataset cifar10 --condition all --verify
    .venv/bin/python scripts/prepare_data.py --dataset cifar10 --condition colormix_d1 --calibrate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from weightcode.data import build as B
from weightcode.data import conditions as C
from weightcode.data.colored import RHO_PRIMARY
from weightcode.data.registry import DATASETS, N_CLASSES
from weightcode.data.sources import download_and_load, record_provenance
from weightcode.data.store import dataset_json_path, forward_path, read_dataset_json
from weightcode.models.calibration import calibrate, write_calibration
from weightcode.models.forward_gen import render_forward, write_forward
from weightcode.models.runinfo import run_info

REPO_ROOT = Path(__file__).resolve().parents[1]


def prepare(name: str, fetch_note: str | None) -> None:
    """Build and write one dataset; refuses if its dataset.json already exists."""
    if dataset_json_path(name).exists():
        raise FileExistsError(f"{dataset_json_path(name)} exists; run with --verify to check it")
    raw = download_and_load(DATASETS[name].source)
    provenance = record_provenance(raw.source, fetch_note)
    built = B.build(name, raw)
    config = DATASETS[name]
    source = render_forward(config.image_shape, built.manifest["norm"]["mean"], built.manifest["norm"]["std"], N_CLASSES)
    forward_sha = write_forward(forward_path(name), source)
    manifest = B.write(built, forward_sha, provenance, run_info(REPO_ROOT))
    summary = {
        "dataset": name,
        "split_sizes": manifest["split_sizes"],
        "norm": {"mean": manifest["norm"]["mean"], "std": manifest["norm"]["std"]},
        "forward_sha256": forward_sha,
    }
    if "colored" in manifest:
        summary["train_conflict_fraction"] = manifest["colored"]["train_conflict_fraction"]
    print(json.dumps(summary, indent=2))


def verify(name: str) -> bool:
    """Re-derive every split from the raw files and compare with dataset.json and the files on disk."""
    manifest = read_dataset_json(name)
    raw = download_and_load(DATASETS[name].source)
    rho = manifest["colored"]["rho"] if "colored" in manifest else RHO_PRIMARY
    report = B.verify(B.build(name, raw, rho=rho))
    print(json.dumps({name: {key: value["match"] for key, value in report.items()}}, indent=2))
    return all(value["match"] for value in report.values())


def _raw_test(dataset: str) -> dict[str, np.ndarray]:
    """Official test arrays re-read from the raw source (the sealed file is never opened)."""
    raw = download_and_load(DATASETS[dataset].source)
    return {"x": raw.x_test, "y": raw.y_test, "idx": np.arange(raw.x_test.shape[0], dtype=np.int64)}


def prepare_condition(name: str, raw_test: dict[str, np.ndarray]) -> None:
    """Build and write one deployment condition; refuses an existing one."""
    condition = C.get_condition(name)
    if dataset_json_path(condition.dataset, name).exists():
        raise FileExistsError(f"{dataset_json_path(condition.dataset, name)} exists; run with --verify to check it")
    manifest = C.write_condition(C.build_from_store(name, raw_test), run_info(REPO_ROOT))
    summary = {
        "condition": name,
        "splits": {split: entry["sha256"] for split, entry in manifest["splits"].items()},
        "test_sha256": manifest["test"]["sha256"],
        "corrupted_train_channel_stats": manifest["corrupted_train_channel_stats"],
    }
    print(json.dumps(summary, indent=2))


def verify_condition(name: str, raw_test: dict[str, np.ndarray]) -> bool:
    """Re-derive one condition from the clean store and the raw test set and compare every hash."""
    report = C.verify_condition(C.build_from_store(name, raw_test))
    print(json.dumps({name: {key: value["match"] for key, value in report.items()}}, indent=2))
    return all(value["match"] for value in report.values())


def calibrate_condition(name: str) -> None:
    """Calibrate one drawn condition's severity on the train split and write its calibration log."""
    log = calibrate(name, run_info(REPO_ROOT))
    path = write_calibration(log)
    chosen = log["chosen"]
    print(json.dumps({"condition": name, "calibration": str(path), "target_drop": log["target_drop"]["value"], **chosen}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=[*DATASETS, "all"])
    parser.add_argument("--condition", help=f"a deployment condition ({sorted(C.CONDITIONS)}) or 'all'")
    parser.add_argument("--verify", action="store_true", help="re-derive and compare instead of writing")
    parser.add_argument("--fetch-note", help="how the raw archives were fetched, if not by torchvision")
    parser.add_argument(
        "--calibrate", action="store_true", help="calibrate a drawn condition's severity on the train split"
    )
    args = parser.parse_args()
    if args.calibrate:
        if args.verify or args.condition not in C.DRAWN_CONDITIONS:
            raise SystemExit(f"--calibrate needs one drawn --condition ({sorted(C.DRAWN_CONDITIONS)}), without --verify")
        if C.DRAWN_CONDITIONS[args.condition].dataset != args.dataset:
            raise SystemExit(f"{args.condition} is not a condition of {args.dataset}")
        calibrate_condition(args.condition)
        return 0
    if args.condition is not None:
        if args.dataset == "all":
            raise SystemExit("--condition needs one --dataset")
        known = [name for name, cond in C.CONDITIONS.items() if cond.dataset == args.dataset]
        names = known if args.condition == "all" else [args.condition]
        if not names or any(name not in known for name in names):
            raise SystemExit(f"conditions of {args.dataset}: {known}")
        raw_test = _raw_test(args.dataset)
        ok = True
        for name in names:
            if args.verify:
                ok = verify_condition(name, raw_test) and ok
            else:
                prepare_condition(name, raw_test)
        return 0 if ok else 1
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    ok = True
    for name in names:
        if args.verify:
            ok = verify(name) and ok
        else:
            prepare(name, args.fetch_note)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
