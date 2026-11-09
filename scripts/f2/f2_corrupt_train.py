"""PREREG F2 data: the pinned official CIFAR-10-C code at severity 5 on E1's 40,000 CIFAR-10 train rows.

    .venv/bin/python -I scripts/f2/f2_corrupt_train.py \\
        --official $WEIGHTCODE_PRIVATE/external/robustness_8190fe3/make_cifar_c.py \\
        --clean-dir $WEIGHTCODE_PRIVATE/cifar10_vl7/splits --dev-rows $WEIGHTCODE_PRIVATE/e10_staging/dev_rows_20261104/dev_rows.json \\
        --out-dir $WEIGHTCODE_PRIVATE/cifar10_vl7/f2_train/corrupt_20261105

Runs in E10's pinned environment (``scripts/e10/cifar10c_official.py``: the official make_cifar_c.py at 8190fe3, sha256
f865ae91..., with the library versions of the E10 fidelity record), not the project's.

1. Reproduction check, before any train row is corrupted: for each F2 condition the clean V_evidence images are
   corrupted again with E10's dev seed for (condition, V_evidence), and the sha256 of the result must equal the
   ``x_sha256`` that E10's dev-rows record holds for those rows. This shows that this machine and environment still
   produce E10's fidelity-checked rows bit for bit. Only V_evidence's x array is read; no label, no model.
2. The 40,000 train rows (``train.npz``, x only), in row order, are corrupted with one recorded seed per condition
   (``TRAIN_SEEDS``, fixed here before any train row was corrupted; contrast is deterministic and draws nothing) and
   written to ``<out-dir>/<condition>/train_x.npz`` (the single array ``x``, NCHW uint8). A record ``corrupt_train.json``
   beside them states the inputs' sha256, the seeds, the outputs' sha256 and the reproduction check. Refuses to
   overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e10"))
import cifar10c_official as O  # noqa: E402

CONDITIONS = ("gaussian_noise", "contrast")
TRAIN_SEEDS = {"gaussian_noise": 1004, "contrast": 5004}
"""One generator seed per condition for the 40,000 train rows (E10's dev splits used 1001-1003 and 5001-5003)."""
DEV_SEEDS_V_EVIDENCE = {"gaussian_noise": 1001, "contrast": 5001}
"""E10's seeds for (condition, V_evidence) (``scripts/e10/cifar10c_make_dev.py`` DEV_SEEDS)."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True, help="$WEIGHTCODE_PRIVATE/cifar10_vl7/splits")
    parser.add_argument("--dev-rows", type=Path, required=True, help="E10's dev_rows.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} exists; nothing written")
    official = O.load_official(args.official)
    dev = json.loads(args.dev_rows.read_text(encoding="utf-8"))
    if dev["official"]["sha256"] != O.OFFICIAL_SHA256 or dev["libraries"] != O.library_versions():
        raise SystemExit(f"this environment differs from E10's dev rows: {O.library_versions()} vs {dev['libraries']}")
    record = {"schema": "weightcode.f2.corrupt_train.v1", "severity": O.SEVERITY,
              "official": {"commit": O.OFFICIAL_COMMIT, "file": O.OFFICIAL_FILE, "sha256": O.OFFICIAL_SHA256},
              "libraries": O.library_versions(),
              "dev_rows": {"path": str(args.dev_rows), "sha256": O.file_sha256(args.dev_rows)},
              "reproduction": {}, "rows": {}}
    evidence_path = args.clean_dir / "V_evidence.npz"
    with np.load(evidence_path, allow_pickle=False) as arrays:
        x_evidence = np.ascontiguousarray(arrays["x"])
    for condition in CONDITIONS:
        row = dev["rows"][f"{condition}/V_evidence"]
        if row["seed"] != DEV_SEEDS_V_EVIDENCE[condition] or row["clean_file_sha256"] != O.file_sha256(evidence_path):
            raise SystemExit(f"{condition}/V_evidence: seed or clean file differs from E10's dev rows")
        again = O.corrupt(official, condition, x_evidence, row["seed"])
        digest = hashlib.sha256(again.tobytes()).hexdigest()
        same = digest == row["x_sha256"]
        record["reproduction"][condition] = {"split": "V_evidence", "seed": row["seed"], "n": int(len(again)),
                                             "x_sha256": digest, "e10_x_sha256": row["x_sha256"], "equal": same}
        print(f"reproduction {condition}/V_evidence seed {row['seed']}: equal {same}", flush=True)
        if not same:
            raise SystemExit(f"{condition}: the pinned code no longer reproduces E10's V_evidence rows; nothing written")
    train_path = args.clean_dir / "train.npz"
    with np.load(train_path, allow_pickle=False) as arrays:
        x = np.ascontiguousarray(arrays["x"])
    staging = args.out_dir.with_name(args.out_dir.name + ".partial")
    staging.mkdir(parents=True)
    for condition in CONDITIONS:
        seed = TRAIN_SEEDS[condition]
        started = time.time()
        out = O.corrupt(official, condition, x, seed)
        (staging / condition).mkdir()
        target = staging / condition / "train_x.npz"
        np.savez(target, x=out)
        mad = np.abs(out.astype(np.int16) - x.astype(np.int16)).reshape(len(x), -1).mean(axis=1)
        record["rows"][condition] = {
            "split": "train", "seed": seed, "n": int(len(x)), "clean_file_sha256": O.file_sha256(train_path),
            "out_file_sha256": O.file_sha256(target), "x_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
            "median_mad_from_clean": float(np.median(mad)), "seconds": round(time.time() - started, 1),
        }
        print(f"{condition}/train: {len(x)} rows, median MAD {np.median(mad):.3f}", flush=True)
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    (staging / "corrupt_train.json").write_text(text, encoding="utf-8")
    staging.rename(args.out_dir)
    print(f"-> {args.out_dir}/corrupt_train.json {hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
