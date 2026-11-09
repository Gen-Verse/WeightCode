"""Readings of list v3 (PREREG Sealed read, "List v3" and its "Composition") from its read's result records.

List v3 has no paper cell map. This script evaluates, on the records it is given, the readings that DEV_RESULTS "T01
and T02" and "T08" fixed on V_select, so the test numbers sit beside the dev numbers they replace:

- B for each of the eight T01 and T02 finals: gain >= 10, clean >= -1.0, joint >= the instance's reference joint + 1.
  The seed-100 references (and the finals of record, read beside them) come from an earlier read's records (list v2):
  E11 gaussian ref_E11_gaussian_bar_grad_sparse19_sal, contrast ref_E11_contrast_bar_grad_sparse19_traj. The seed-101
  and seed-102 references are list v3's own (the bars of PREREG 92a96af). The instance's best kept F1 pick is read
  beside each final (no criterion);
- met / launched over the eight, and with the two finals of record on seed 100;
- T08 per source over its five non-own students: own gain (the source final on its own student), mean gain, reuse
  share (mean gain / own gain), clean >= -1.0 count, C in the E3 form (share >= 0.7 and clean >= -1.0 on at least 4
  of 5), and the per-target form (ratio >= 0.7 and clean >= -1.0, count of 5).

Every record is checked against its list entry (instance and child sha256), every earlier record against the earlier
list. A reading whose entry has no record is "not readable", never filled from another source.

Usage: python scripts/sealed/make_test_table_v3.py --results <v3 jsonl> --list scripts/sealed/finals_v3.json
       --earlier-results <v2 jsonl> --earlier-list scripts/sealed/finals_v2.json --out <json>
       [--mode sealed|dev] [--list-sha <v3 sha256>] [--earlier-list-sha <v2 sha256>]

In dev mode, several result files may be given per option (one dry-run file per entry) and the checks against the
list still apply; the dev table must reproduce DEV_RESULTS.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

B_GAIN, B_CLEAN, B_MARGIN = 10.0, -1.0, 1.0
C_RATIO, C_CLEAN, C_MIN_STUDENTS = 0.7, -1.0, 4
SEEDS = (100, 101, 102, 103, 104, 105)

FINALS = (
    # name, condition, seed, bar reference, F1 kept pick; a reference in EARLIER is read from the earlier records
    ("T02_B_gaussian_s100_r1", "gaussian", 100, "ref_E11_gaussian_bar_grad_sparse19_sal", "ref_E11_gaussian_F1_lora_r4"),
    ("T02_B_gaussian_s100_r2", "gaussian", 100, "ref_E11_gaussian_bar_grad_sparse19_sal", "ref_E11_gaussian_F1_lora_r4"),
    ("T01_B_gaussian_s101", "gaussian", 101, "ref_E11_gaussian_s101_bar_interp", "ref_E11_gaussian_s101_F1_lora_r16"),
    ("T01_B_gaussian_s102", "gaussian", 102, "ref_E11_gaussian_s102_bar_interp_e11g2",
     "ref_E11_gaussian_s102_F1_grad_last_e11g2"),
    ("T02_B_contrast_s100_r1", "contrast", 100, "ref_E11_contrast_bar_grad_sparse19_traj",
     "ref_E11_contrast_F1_ft_full_l2sp"),
    ("T02_B_contrast_s100_r2", "contrast", 100, "ref_E11_contrast_bar_grad_sparse19_traj",
     "ref_E11_contrast_F1_ft_full_l2sp"),
    ("T01_B_contrast_s101", "contrast", 101, "ref_E11_contrast_s101_bar_grad_sparse19_sal_e11g2",
     "ref_E11_contrast_s101_F1_ft_full_lwf"),
    ("T01_B_contrast_s102", "contrast", 102, "ref_E11_contrast_s102_bar_grad_full_e11g2",
     "ref_E11_contrast_s102_F1_lora_r16"),
)
RECORD_FINALS = {"gaussian": ("E11_B_gaussian_s100", "ref_E11_gaussian_bar_grad_sparse19_sal"),
                 "contrast": ("E11_B_contrast_s100", "ref_E11_contrast_bar_grad_sparse19_traj")}
EARLIER = ("ref_E11_gaussian_bar_grad_sparse19_sal", "ref_E11_gaussian_F1_lora_r4",
           "ref_E11_contrast_bar_grad_sparse19_traj", "ref_E11_contrast_F1_ft_full_l2sp",
           "E11_B_gaussian_s100", "E11_B_contrast_s100")


class TableError(RuntimeError):
    pass


def entries(list_path: Path) -> dict[str, dict[str, Any]]:
    parsed = json.loads(list_path.read_text(encoding="utf-8"))
    return {e["name"]: e for block in ("finals", "references") for e in parsed[block]}


def load(paths: Sequence[Path], mode: str, list_sha: str | None, listed: Mapping[str, Mapping[str, Any]],
         wanted: set[str] | None) -> dict[str, dict[str, Any]]:
    """Records by entry name, each checked against its list entry; ``wanted`` limits the names kept (earlier read)."""
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            name = record.get("name")
            if wanted is not None and name not in wanted:
                continue
            where = f"{path}:{number}: {name}"
            if record.get("mode") != mode:
                raise TableError(f"{where}: mode {record.get('mode')!r}, expected {mode!r}")
            if list_sha is not None and record.get("list_sha256") != list_sha:
                raise TableError(f"{where}: list_sha256 {record.get('list_sha256')!r}, expected {list_sha}")
            if name not in listed:
                raise TableError(f"{where}: not on the list")
            entry = listed[name]
            if record.get("instance_id") != entry["instance_id"] or record.get("child_sha256") != entry["child_sha256"]:
                raise TableError(f"{where}: instance or child differs from the list entry")
            for key in ("gain", "clean", "joint"):
                if not isinstance(record.get(key), (int, float)) or not math.isfinite(record[key]):
                    raise TableError(f"{where}: {key} {record.get(key)!r}")
            if abs(record["gain"] + record["clean"] - record["joint"]) > 1e-9:
                raise TableError(f"{where}: joint != gain + clean")
            if mode == "sealed" and name in records:
                raise TableError(f"{where}: appears twice in a sealed read")
            records[name] = record
    return records


def field(records: Mapping[str, Mapping[str, Any]], name: str, key: str) -> float | None:
    record = records.get(name)
    return None if record is None else float(record[key])


def reading(checks: Mapping[str, bool | None]) -> str:
    if any(v is None for v in checks.values()):
        return "not readable (an entry has no record)"
    return "met" if all(checks.values()) else "missed"


def b(records: Mapping[str, Mapping[str, Any]], final: str, ref: str, f1: str | None) -> dict[str, Any]:
    gain, clean, joint = (field(records, final, k) for k in ("gain", "clean", "joint"))
    ref_joint = field(records, ref, "joint")
    bar = None if ref_joint is None else ref_joint + B_MARGIN
    checks = {"gain >= 10": None if gain is None else gain >= B_GAIN,
              "clean >= -1.0": None if clean is None else clean >= B_CLEAN,
              "joint >= reference joint + 1": None if joint is None or bar is None else joint >= bar}
    out = {"final": final, "gain": gain, "clean": clean, "joint": joint, "reference": ref,
           "reference_joint": ref_joint, "bar": bar, "checks": checks, "reading": reading(checks)}
    if f1 is not None:
        out.update(f1_kept=f1, f1_kept_joint=field(records, f1, "joint"))
    return out


def reuse(records: Mapping[str, Mapping[str, Any]], final: str, seed: int) -> dict[str, Any]:
    short = final.replace("_B_", "_")
    own = field(records, final, "gain")
    cells = {t: {"gain": field(records, f"T08_{short}_on_s{t}", "gain"),
                 "clean": field(records, f"T08_{short}_on_s{t}", "clean")} for t in SEEDS if t != seed}
    gains = [c["gain"] for c in cells.values()]
    cleans = [c["clean"] for c in cells.values()]
    if None in gains or None in cleans or own is None:
        return {"own_gain": own, "cells": cells, "reading": "not readable (an entry has no record)"}
    mean_gain = statistics.fmean(gains)  # type: ignore[arg-type]
    share = mean_gain / own if own else None
    n_clean = sum(1 for c in cleans if c >= C_CLEAN)  # type: ignore[operator]
    per_target = sum(1 for g, c in zip(gains, cleans) if own and g / own >= C_RATIO and c >= C_CLEAN)  # type: ignore[operator]
    checks = {"reuse share >= 0.7": None if share is None else share >= C_RATIO,
              "clean >= -1.0 on >= 4 of 5": n_clean >= C_MIN_STUDENTS}
    return {"own_gain": own, "own_clean": field(records, final, "clean"), "mean_gain": mean_gain,
            "mean_clean": statistics.fmean(cleans), "share": share, "n_clean_ok": n_clean,
            "per_target_ok": per_target, "cells": cells, "checks": checks, "reading": reading(checks)}


def build(v3: Mapping[str, Mapping[str, Any]], earlier: Mapping[str, Mapping[str, Any]],
          listed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    both = {**earlier, **v3}
    if set(earlier) & set(v3):
        raise TableError(f"names in both reads: {sorted(set(earlier) & set(v3))}")
    finals = {name: {"condition": cond, "seed": seed, **b(both, name, ref, f1)}
              for name, cond, seed, ref, f1 in FINALS}
    for name, cond, seed, ref, _ in FINALS:
        if ref not in EARLIER and listed[ref]["instance_id"] != listed[name]["instance_id"]:
            raise TableError(f"{name}: reference {ref} is on another instance")
    records = {cond: b(both, final, ref, None) for cond, (final, ref) in RECORD_FINALS.items()}
    count: dict[str, Any] = {}
    for cond in ("gaussian", "contrast"):
        new = [f for f in finals.values() if f["condition"] == cond]
        met = sum(1 for f in new if f["reading"] == "met")
        s100 = [f for f in new if f["seed"] == 100]
        count[cond] = {"new_met": met, "new_launched": len(new),
                       "s100_with_record_met": sum(1 for f in s100 if f["reading"] == "met")
                       + (records[cond]["reading"] == "met"), "s100_with_record_launched": len(s100) + 1,
                       "unreadable": [f["final"] for f in new if f["reading"].startswith("not readable")]}
    count["all_new"] = {"met": sum(1 for f in finals.values() if f["reading"] == "met"), "launched": len(finals)}
    t08 = {name: {"condition": cond, "seed": seed, **reuse(v3, name, seed)} for name, cond, seed, _, _ in FINALS}
    missing = sorted(set(listed) - set(v3))
    return {"finals": finals, "finals_of_record": records, "met_over_launched": count, "t08": t08,
            "missing": missing}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--earlier-results", nargs="+", type=Path, required=True)
    parser.add_argument("--earlier-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("sealed", "dev"), default="sealed")
    parser.add_argument("--list-sha", default=None)
    parser.add_argument("--earlier-list-sha", default=None)
    args = parser.parse_args(argv)
    if args.mode == "sealed" and (args.list_sha is None or args.earlier_list_sha is None):
        parser.error("--list-sha and --earlier-list-sha are required in sealed mode")
    listed = entries(args.list)
    v3 = load(args.results, args.mode, args.list_sha, listed, None)
    earlier = load(args.earlier_results, args.mode, args.earlier_list_sha, entries(args.earlier_list), set(EARLIER))
    if set(EARLIER) - set(earlier):
        raise TableError(f"earlier records missing: {sorted(set(EARLIER) - set(earlier))}")
    result = build(v3, earlier, listed)
    result["inputs"] = {"results": [str(p) for p in args.results], "list": str(args.list),
                        "earlier_results": [str(p) for p in args.earlier_results],
                        "earlier_list": str(args.earlier_list), "mode": args.mode, "list_sha256": args.list_sha,
                        "earlier_list_sha256": args.earlier_list_sha, "n_records": len(v3)}
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"records {len(v3)} of {len(listed)}; missing {result['missing']}")
    for name, f in result["finals"].items():
        print(f"B {name}: {f['gain']:+.3f} / {f['clean']:+.3f} / {f['joint']:+.3f} vs bar {f['bar']:+.3f}: "
              f"{f['reading']}" if f["joint"] is not None and f["bar"] is not None else f"B {name}: {f['reading']}")
    for cond, f in result["finals_of_record"].items():
        print(f"record {cond} {f['final']}: {f['reading']}")
    print("met / launched:", json.dumps(result["met_over_launched"]))
    for name, r in result["t08"].items():
        if "share" in r:
            print(f"T08 {name}: own {r['own_gain']:+.3f}, mean {r['mean_gain']:+.3f}, share {r['share']:.3f}, "
                  f"clean ok {r['n_clean_ok']}/5, per-target {r['per_target_ok']}/5: C {r['reading']}")
        else:
            print(f"T08 {name}: {r['reading']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
