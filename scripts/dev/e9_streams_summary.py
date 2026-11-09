"""PREREG E9 reference streams: the table of one session from its ``session_<t>.json`` (``run_fscil_streams.py``).

    .venv/bin/python scripts/dev/e9_streams_summary.py RUN_DIR SESSION --validity FILE [--json OUT]

Prints and (with --json, refusing to overwrite) writes:
- the floor of the session: n_c, n_s, d_t over every checked candidate, its range, and floor_t;
- per method, the flip rates (fixed_old + broken_old) / n_c of its checks: n, quartiles and median;
- per stream, the kept check (config, S_new, S_old, S, changed entries, limits) and its V_select readout under the
  session mask, class-balanced: all seen, the session's classes, earlier incremental classes, base classes, and the
  base-class change against the session-1 base; with the no-floor argmax as context;
- per checked stream, the supplement (never gating): the per-candidate-floor pick, its S, S_old and own floor_c,
  and its V_select readout;
- the best reference of the gate set (the checked streams but imprint_retain; imprint_plain is an instrument), the
  bar (best + 2 points of all-seen class-balanced accuracy) and the not-applicable test (best within 1 point of the
  all-class reference's all-seen class-balanced accuracy at that session, from the validity record).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

GATE_EXCLUDED = ("imprint_retain",)
INSTRUMENTS = ("imprint_plain",)
BAR_POINTS = 2.0
NOT_APPLICABLE_POINTS = 1.0


def _pts(x: float | None) -> float | None:
    return None if x is None else 100.0 * x


def summarize(record: dict[str, Any], validity: dict[str, Any]) -> dict[str, Any]:
    t = int(record["session"])
    floor = record["floor"]
    per_method = {}
    rows = []
    for method, stream in record["streams"].items():
        checks = stream["checks"] or []
        flips = [(c["fixed_old"] + c["broken_old"]) / c["n_old"] for c in checks]
        if flips:
            q1, med, q3 = np.quantile(flips, [0.25, 0.5, 0.75])
            per_method[method] = {"n": len(flips), "q1": float(q1), "median": float(med), "q3": float(q3),
                                  "min": min(flips), "max": max(flips)}
        kept = stream["kept"]
        check = kept["check"] or {}
        config = next((c["config"] for c in stream["candidates"] if c["patch_id"] == kept["patch_id"]), None)
        edit = next((c["notes"]["edit"] for c in stream["candidates"] if c["patch_id"] == kept["patch_id"]), None)
        a, vb = stream["readout"]["absolute"], stream["readout"]["vs_base"]
        free = stream["unconstrained_pick"]
        free_check = next((c for c in checks if free and c["patch_id"] == free["patch_id"]), None)
        sup = stream.get("supplement_per_candidate_floor")
        supplement = None
        if sup is not None:
            i = next((k for k, c in enumerate(checks) if c["patch_id"] == sup["pick"]), None)
            sa = sup["readout"]["absolute"]
            supplement = {"pick": sup["pick"] or "current", "S": sup["score"],
                          "S_old": None if i is None else checks[i]["score_old"],
                          "floor_c": None if i is None else sup["floors"][i],
                          "floors_min_max": [min(sup["floors"]), max(sup["floors"])] if sup["floors"] else None,
                          "bal_seen": _pts(sa["bal_seen"]), "bal_new": _pts(sa["bal_new"]),
                          "bal_prev": _pts(sa["bal_prev"]), "bal_old": _pts(sa["bal_old"]),
                          "old_change_vs_base": _pts(sup["readout"]["vs_base"]["delta_bal_old"])}
        rows.append({
            "method": method,
            "supplement": supplement,
            "kept": kept["patch_id"] or "current",
            "config": config,
            "S_new": check.get("score_new"), "S_old": check.get("score_old"), "S": check.get("score"),
            "n_changed": check.get("n_changed", edit["n_changed"] if edit else 0),
            "within_limits": None if edit is None else edit["within_limits"],
            "bal_seen": _pts(a["bal_seen"]), "bal_new": _pts(a["bal_new"]), "bal_prev": _pts(a["bal_prev"]),
            "bal_old": _pts(a["bal_old"]), "old_change_vs_base": _pts(vb["delta_bal_old"]),
            "acc_seen": _pts(a["acc_seen"]),
            "no_floor_pick": None if free is None else {
                "patch_id": free["patch_id"] or "current", "S": free["score"],
                "S_old": None if free_check is None else free_check["score_old"]},
        })
    gate = [r for r in rows if r["method"] not in GATE_EXCLUDED + INSTRUMENTS]
    best = max(gate, key=lambda r: r["bal_seen"])
    reference = 100.0 * float(validity["per_session"][str(t)]["reference"]["bal_seen"])
    return {
        "session": t, "floor": floor, "flip_rate_by_method": per_method, "streams": rows,
        "gate_set": [r["method"] for r in gate],
        "best_reference": {"method": best["method"], "bal_seen": best["bal_seen"]},
        "bar_bal_seen": best["bal_seen"] + BAR_POINTS,
        "allclass_bal_seen": reference,
        "not_applicable": best["bal_seen"] >= reference - NOT_APPLICABLE_POINTS,
        "R_points": float(validity["V3"]["R"]),
    }


def _f(x: Any, spec: str = "6.2f") -> str:
    return "  n/a " if x is None else format(x, spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("session", type=int)
    parser.add_argument("--validity", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    path = args.run_dir / f"session_{args.session}.json"
    record_bytes = path.read_bytes()
    validity_bytes = args.validity.read_bytes()
    summary = summarize(json.loads(record_bytes), json.loads(validity_bytes))
    summary["inputs"] = {str(path): hashlib.sha256(record_bytes).hexdigest(),
                         str(args.validity): hashlib.sha256(validity_bytes).hexdigest()}
    fl = summary["floor"]
    print(f"session {summary['session']}: n_c {fl['n_c']} n_s {fl['n_s']}; d {fl['d']:.5f} over {fl['n_checks']} "
          f"checks (min {fl['min']:.4f}, max {fl['max']:.4f}); floor {fl['floor']}")
    print("flip rate by method: n  q1  median  q3  (min max)")
    for m, q in summary["flip_rate_by_method"].items():
        print(f"  {m:<20} {q['n']:>2} {q['q1']:.4f} {q['median']:.4f} {q['q3']:.4f} ({q['min']:.4f} {q['max']:.4f})")
    print(f"{'stream':<20} {'kept':<24} {'config':<26} {'S_new':>5} {'S_old':>5} {'S':>5} {'chg':>6} lim | "
          f"{'seen':>6} {'new':>6} {'prev':>6} {'old':>6} {'dOld':>6} | no-floor pick")
    for r in summary["streams"]:
        cfg = json.dumps(r["config"]) if r["config"] else "-"
        nf = r["no_floor_pick"]
        print(f"{r['method']:<20} {r['kept']:<24} {cfg[:26]:<26} {_f(r['S_new'], '5d')} {_f(r['S_old'], '5d')} "
              f"{_f(r['S'], '5d')} {r['n_changed']:>6} {'-' if r['within_limits'] is None else str(r['within_limits'])[0]} | {_f(r['bal_seen'])} "
              f"{_f(r['bal_new'])} {_f(r['bal_prev'])} {_f(r['bal_old'])} {_f(r['old_change_vs_base'], '+6.2f')} | "
              f"{'' if nf is None else nf['patch_id'] + ' S ' + str(nf['S']) + ' S_old ' + str(nf['S_old'])}")
    print("supplement (per-candidate floor, never gating): pick  S  S_old  floor_c  (floors min max) | seen new prev "
          "old dOld")
    for r in summary["streams"]:
        u = r["supplement"]
        if u is not None:
            print(f"  {r['method']:<20} {u['pick']:<24} {_f(u['S'], '5d')} {_f(u['S_old'], '5d')} "
                  f"{_f(u['floor_c'], '5d')} ({u['floors_min_max']}) | {_f(u['bal_seen'])} {_f(u['bal_new'])} "
                  f"{_f(u['bal_prev'])} {_f(u['bal_old'])} {_f(u['old_change_vs_base'], '+6.2f')}")
    print(f"best reference {summary['best_reference']['method']} {summary['best_reference']['bal_seen']:.2f}; bar "
          f"{summary['bar_bal_seen']:.2f}; all-class {summary['allclass_bal_seen']:.2f}; not applicable "
          f"{summary['not_applicable']}; R {summary['R_points']}")
    if args.json:
        if args.json.exists():
            print(f"{args.json} exists; nothing written", file=sys.stderr)
            return 3
        text = json.dumps(summary, indent=1, sort_keys=True, allow_nan=False) + "\n"
        with open(args.json, "x", encoding="utf-8") as handle:
            handle.write(text)
        print(f"-> {args.json} {hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
