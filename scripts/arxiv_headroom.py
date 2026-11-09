"""PREREG E6 headroom report of one ogbn-arxiv protocol (``weightcode.analysis.arxiv_headroom``), dev seed 100.

Writes $WEIGHTCODE_RUNS/devreadouts/e6_headroom_<protocol>_s<seed>_dev_<date>.json (refuses an existing file) and prints
the gate and the diagnostics' headline numbers with the file's sha256.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/arxiv_headroom.py --protocol inductive --date 20261104
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from weightcode.analysis.arxiv_headroom import headroom_report
from weightcode.common.paths import runs_dir
from weightcode.data import arxiv as A


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=tuple(A.PROTOCOLS), required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--date", required=True, help="YYYYmmdd stamp of the readout file")
    args = parser.parse_args()
    out = runs_dir() / "devreadouts" / f"e6_headroom_{args.protocol}_s{args.seed}_dev_{args.date}.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    report = headroom_report(args.protocol, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, indent=2) + "\n").encode("utf-8")
    out.write_bytes(data)
    d = report["diagnostics"]
    print(json.dumps({
        "file": str(out), "sha256": hashlib.sha256(data).hexdigest(), "accuracy": report["accuracy"],
        "G_new": {k: report["G_new"][k] for k in ("diff_points", "paired_se_points", "only_b_right", "only_a_right")},
        "G_old": {k: report["G_old"][k] for k in ("diff_points", "paired_se_points", "only_b_right", "only_a_right")},
        "G_shift": report["G_shift"], "gate": report["gate"],
        "prior_oracle": {k: d["prior_oracle"][k] for k in ("gain_points", "old_change_points", "recovered_fraction")},
        "last_many_best_joint": d["last_many"]["best_joint"],
        "separability": {k: d["separability"][k] for k in ("degree_auc", "logistic_auc", "degree_median")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
