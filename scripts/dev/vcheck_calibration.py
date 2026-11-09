"""How well a check's clean change on V_check predicts the same program's clean change on V_select (dev only).

Reads the session_info records that scripts/dev/post_learn.sh wrote ($WEIGHTCODE_RUNS/devreadouts/learn/<sid>/info.json)
and pairs, for every checked program of an evidence-v1 session, score_old / n_old on V_check with score / n on the
clean V_select, both in accuracy points against the base model. Prints the spread of the difference and the V_check
margin under the floor that keeps the V_select change at or above the floor with a given one-sided probability under
a normal approximation of that difference. No V_check or V_select row is read here, only the recorded counts.

Usage: scripts/dev/vcheck_calibration.py [--sessions SID ...] [--floor-points 1.0] [--coverage 0.95]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

Z_ONE_SIDED = {0.9: 1.2816, 0.95: 1.6449, 0.975: 1.96, 0.99: 2.3263}


def pairs(root: Path, sessions: list[str] | None) -> list[dict[str, Any]]:
    """One record per checked program of the evidence-v1 sessions under ``root`` (all, or ``sessions``)."""
    paths = [root / sid / "info.json" for sid in sessions] if sessions else sorted(root.glob("*/info.json"))
    out = []
    for path in paths:
        info = json.loads(path.read_text(encoding="utf-8"))
        for row in info["trajectory"]:
            vcheck, vselect = row.get("vcheck"), row.get("vselect_clean")
            if not vcheck or not vselect or "readouts_fewshot" in row:
                continue
            out.append({"session": info["session"], "condition": info["condition"], "query": row["query_index"],
                        "vcheck": 100.0 * vcheck["score_old"] / vcheck["n_old"],
                        "vselect": 100.0 * vselect["score"] / vselect["n"],
                        "n_old": vcheck["n_old"], "n_select": vselect["n"]})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--floor-points", type=float, default=1.0)
    parser.add_argument("--coverage", type=float, default=0.95, choices=sorted(Z_ONE_SIDED))
    args = parser.parse_args(argv)
    root = Path(os.environ["WEIGHTCODE_RUNS"]) / "devreadouts" / "learn"
    records = pairs(root, args.sessions)
    if len(records) < 3:
        print(f"{len(records)} checked programs; too few", file=sys.stderr)
        return 1
    diff = [r["vselect"] - r["vcheck"] for r in records]
    xs, ys = [r["vcheck"] for r in records], [r["vselect"] for r in records]
    mean, sd = statistics.mean(diff), statistics.stdev(diff)
    corr = statistics.correlation(xs, ys)
    margin = Z_ONE_SIDED[args.coverage] * sd - mean
    at_floor = [r for r in records if -args.floor_points <= r["vcheck"] <= 0.0]
    print(f"checked programs {len(records)} in {len({r['session'] for r in records})} sessions; rows per set "
          f"{sorted({r['n_old'] for r in records})} (V_check clean) and {sorted({r['n_select'] for r in records})} "
          f"(V_select clean)")
    print(f"vselect - vcheck (points): mean {mean:+.3f}, sd {sd:.3f}, min {min(diff):+.2f}, max {max(diff):+.2f}; "
          f"correlation {corr:.3f}")
    print(f"checks with vcheck in [-{args.floor_points:g}, 0]: {len(at_floor)}, of which vselect < "
          f"-{args.floor_points:g}: {sum(r['vselect'] < -args.floor_points for r in at_floor)}")
    print(f"V_check margin for P(vselect >= -{args.floor_points:g}) >= {args.coverage:g} at the floor (normal "
          f"approximation): {margin:.2f} points, a V_check floor of {-args.floor_points + margin:+.2f} points")
    for r in sorted(records, key=lambda r: (r["condition"], r["session"], r["query"])):
        print(f"  {r['condition']:18s} {r['session']} q{r['query']} vcheck {r['vcheck']:+.2f} vselect "
              f"{r['vselect']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
