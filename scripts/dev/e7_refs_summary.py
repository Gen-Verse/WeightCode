"""PREREG E7 reference summary: the bars, the flip rate d and the kept-rule floor, from the reference runs and readouts.

Reads the readout files of ``scripts/dev/e7_readouts.sh`` (each reference's own final, and the kept-rule pick at
S_old >= -20), the run.json files they name, and the validity gate record (R and the all-class reference accuracy).
Writes one JSON record (refuses to overwrite) and prints the table:

- per reference: final slot and config, V_check S_new / S_old, V_select all-class, new-class and old-class change,
  within limits or not; the same for the kept pick;
- the best reference by V_select all-class accuracy among the gate set (every reference but imprint_retain), the bar
  (best + 2 points) and whether the 2-point criterion is not applicable (best within 1 point of the reference);
- d = (fixed + broken) / n_c on the V_check old part, the median over every reference's checked candidates, and the
  floor round((n_c / 100) (R - 0.11 + 164.5 sqrt(d / n_c + d / n_s))) clamped to [-20, 0] (PREREG E7 kept rule).

    .venv/bin/python scripts/dev/e7_refs_summary.py --refs REFS.jsonl --kept KEPT.jsonl \\
        --validity $WEIGHTCODE_RUNS/devreadouts/e7_validity_s100_dev_20261104.json --out SUMMARY.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

GATE_EXCLUDED = ("imprint_retain",)
BAR_POINTS = 2.0
NOT_APPLICABLE_POINTS = 1.0
FLOOR_LOOSEST = -20
FLOOR_TIGHTEST = 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floor_value(d: float, n_c: int, n_s: int, r_points: float) -> float:
    """(n_c / 100) x (R - 0.11 + 164.5 x sqrt(d / n_c + d / n_s)), before rounding and the clamp."""
    return (n_c / 100) * (r_points - 0.11 + 164.5 * math.sqrt(d / n_c + d / n_s))


def kept_floor(value: float) -> int:
    """The nearest integer to ``value`` (halves away from zero), never looser than -20 and never above 0."""
    nearest = int(math.floor(abs(value) + 0.5)) * (1 if value >= 0 else -1)
    return max(FLOOR_LOOSEST, min(FLOOR_TIGHTEST, nearest))


def row(record: dict[str, Any]) -> dict[str, Any]:
    r = record["readouts_graphnew"]
    return {
        "method": record["method"],
        "final": record.get("final_patch_id") or "base",
        "config": record.get("final_config"),
        "S": record["vcheck"]["score"],
        "S_new": record["vcheck"]["score_new"],
        "S_old": record["vcheck"]["score_old"],
        "vselect_all_acc": 100 * r["acc_all_child"],
        "vselect_new_acc": 100 * r["acc_new_child"],
        "vselect_new_acc_balanced": 100 * r["acc_new_balanced_child"],
        "vselect_old_change_points": 100 * r["delta_old"],
        "vselect_all_points": r["all_points"],
        "worst_old_class_delta_points": 100 * r["worst_old_class_delta"],
        "respects_edit_limits": record.get("respects_edit_limits"),
        "is_base_noop": record["is_base_noop"],
        "run": record["run"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refs", type=Path, required=True)
    parser.add_argument("--kept", type=Path, required=True)
    parser.add_argument("--validity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"{args.out} exists; nothing overwritten", file=sys.stderr)
        return 3
    refs = [json.loads(line) for line in args.refs.read_text().splitlines() if line.strip()]
    kept = [json.loads(line) for line in args.kept.read_text().splitlines() if line.strip()]
    validity = json.loads(args.validity.read_text())
    if [r["method"] for r in refs] != [k["method"] for k in kept] or len({r["instance_id"] for r in refs}) != 1:
        raise SystemExit("the two readout files must name the same runs of one instance, in the same order")
    r_points = float(validity["V3"]["R"])
    reference_points = 100 * float(validity["V2"]["acc_b"])
    flips, n_c = [], None
    for record in refs:
        run = json.loads(Path(record["run"]).read_text())
        for entry in run["oracle_log"]:
            n_old = int(entry["n_old"])
            if n_c is not None and n_old != n_c:
                raise SystemExit(f"V_check old part sizes differ: {n_old} vs {n_c}")
            n_c = n_old
            flips.append((int(entry["fixed_old"]) + int(entry["broken_old"])) / n_old)
    n_s = int(refs[0]["readouts_graphnew"]["n_old"])
    d = statistics.median(flips)
    value = floor_value(d, n_c, n_s, r_points)
    rows = [row(r) for r in refs]
    kept_rows = [{**row(k), "selection": k["selection"]} for k in kept]
    gate = [x for x in rows if x["method"] not in GATE_EXCLUDED]
    best = max(gate, key=lambda x: x["vselect_all_acc"])
    summary = {
        "schema": "weightcode.e7_refs_summary.v1",
        "instance_id": refs[0]["instance_id"],
        "inputs": {str(p): sha256(p) for p in (args.refs, args.kept, args.validity)},
        "R_points": r_points,
        "all_class_reference_vselect_acc": reference_points,
        "references": rows,
        "kept20": kept_rows,
        "gate_set": [x["method"] for x in gate],
        "best_reference": {"method": best["method"], "vselect_all_acc": best["vselect_all_acc"]},
        "bar_all_class_acc": best["vselect_all_acc"] + BAR_POINTS,
        "bar_old_change_points": r_points,
        "not_applicable": best["vselect_all_acc"] >= reference_points - NOT_APPLICABLE_POINTS,
        "not_applicable_threshold_acc": reference_points - NOT_APPLICABLE_POINTS,
        "flip_rate": {"rule": "median over every reference's checked candidates of (fixed_old + broken_old) / n_c",
                      "n_checks": len(flips), "d": d, "min": min(flips), "max": max(flips), "n_c": n_c, "n_s": n_s},
        "floor": {"formula": "round((n_c / 100) x (R - 0.11 + 164.5 x sqrt(d / n_c + d / n_s))), clamped to [-20, 0]",
                  "value_at_d": value, "floor": kept_floor(value)},
    }
    text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as handle:
        handle.write(text)
    for title, table in (("own final", rows), ("kept20", kept_rows)):
        print(f"== {title}")
        for x in table:
            print(f"{x['method']:<20} {x['final']:<5} {json.dumps(x['config']):<38} S_new {x['S_new']:>5} S_old "
                  f"{x['S_old']:>5} | all {x['vselect_all_acc']:6.2f}% new {x['vselect_new_acc']:6.2f}% old "
                  f"{x['vselect_old_change_points']:+6.2f} | limits {x['respects_edit_limits']}")
    print(f"best {best['method']} {best['vselect_all_acc']:.2f}%; bar {summary['bar_all_class_acc']:.2f}%; not "
          f"applicable {summary['not_applicable']} (threshold {summary['not_applicable_threshold_acc']:.2f}%)")
    print(f"d {d:.5f} over {len(flips)} checks; floor value {value:.2f} -> {summary['floor']['floor']}")
    print(f"-> {args.out} {sha256(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
