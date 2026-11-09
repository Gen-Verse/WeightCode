"""PREREG E7 validity gate (``weightcode.analysis.graphnew_validity``), dev seed 100.

Writes $WEIGHTCODE_RUNS/devreadouts/e7_validity_s<seed>_dev_<date>.json (refuses an existing file) and prints V1 to V3, R
and the file's sha256.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/graphnew_validity.py --date 20261104
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from weightcode.analysis.graphnew_validity import validity_report
from weightcode.common.paths import runs_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--date", required=True, help="YYYYmmdd stamp of the readout file")
    args = parser.parse_args()
    out = runs_dir() / "devreadouts" / f"e7_validity_s{args.seed}_dev_{args.date}.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    report = validity_report(args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, indent=2) + "\n").encode("utf-8")
    out.write_bytes(data)
    keys = ("diff_points", "paired_se_points", "only_b_right", "only_a_right", "acc_a", "acc_b")
    print(json.dumps({"file": str(out), "sha256": hashlib.sha256(data).hexdigest(), "gate": report["gate"],
                      "V1": report["V1"], "V2": {k: report["V2"][k] for k in keys},
                      "V3": {k: report["V3"][k] for k in keys} | {"R": report["V3"]["R"]},
                      "reference_new_class": {k: report["reference_new_class"][k]
                                              for k in ("acc_pooled", "acc_balanced", "acc_base_pooled")},
                      "old_select": report["old_select"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
