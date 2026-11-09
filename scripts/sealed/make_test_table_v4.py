"""Readings of list v4 (PREREG T16 "List v4") from its read's result records: the T16 stream criteria on the test sets.

For each of the six T16 streams (arm N: N1 the session of record, N2, N3; arm L: L1, L2, L3) the readings DEV_RESULTS
"T16" fixed on V_select, evaluated on the records given:

- the marginal-gain condition: contrast W1 to W2 >= 10 (W2 and W1 read on i3eb3fc78), gaussian W1 to W2 >= -3 (W1 on
  i00802425 is list v2's E11_B_gaussian_s100, the same child on the same instance), clean W0 to W2 >= -2.0;
- the formal stream criteria: mean gain over the two conditions after round 2 >= 10, clean >= -2.0, forgetting (gaussian
  W1 minus W2) <= 3; and the stream joint (mean gain plus clean);
- met / launched per arm, and T16's reading (L met in at least 2 of 3 and N in at most 1 of 3, or otherwise).

The clean change of a W2 must be the same on both instances (one student, one clean set); a difference stops the build.
Every record is checked against its list entry (instance and child sha256), the earlier record against list v2. A
reading whose entry has no record is "not readable", never filled from another source.

Usage: python scripts/sealed/make_test_table_v4.py --results <v4 jsonl> --list scripts/sealed/finals_v4.json
       --earlier-results <v2 jsonl> --earlier-list scripts/sealed/finals_v2.json --out <json>
       [--mode sealed|dev] [--list-sha <v4 sha256>] [--earlier-list-sha <v2 sha256>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

MARGINAL_GAIN, FORGET_MAX, CLEAN_MIN, MEAN_GAIN_MIN = 10.0, 3.0, -2.0, 10.0
STREAMS = (("N1", "N"), ("L1", "L"), ("N2", "N"), ("L2", "L"), ("N3", "N"), ("L3", "L"))
W1_GAUSSIAN = "E11_B_gaussian_s100"  # list v2: W1 = e40c2078... on i00802425
W1_CONTRAST = "E11_stream_W1_contrast"
CLEAN_TOL = 1e-9


class TableError(RuntimeError):
    pass


def entries(list_path: Path) -> dict[str, dict[str, Any]]:
    parsed = json.loads(list_path.read_text(encoding="utf-8"))
    return {e["name"]: e for block in ("finals", "references") for e in parsed[block]}


def load(paths: Sequence[Path], mode: str, list_sha: str | None, listed: Mapping[str, Mapping[str, Any]],
         wanted: set[str] | None) -> dict[str, dict[str, Any]]:
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
            if mode == "sealed" and name in records:
                raise TableError(f"{where}: appears twice in a sealed read")
            records[name] = record
    return records


def value(records: Mapping[str, Mapping[str, Any]], name: str, key: str) -> float | None:
    record = records.get(name)
    return None if record is None else float(record[key])


def met(checks: Mapping[str, bool | None]) -> str:
    if any(v is None for v in checks.values()):
        return "not readable (an entry has no record)"
    return "met" if all(checks.values()) else "missed"


def stream(records: Mapping[str, Mapping[str, Any]], label: str) -> dict[str, Any]:
    g2 = value(records, f"E11_stream_T16_{label}_W2_gaussian", "gain")
    c2 = value(records, f"E11_stream_T16_{label}_W2_contrast", "gain")
    clean_g = value(records, f"E11_stream_T16_{label}_W2_gaussian", "clean")
    clean_c = value(records, f"E11_stream_T16_{label}_W2_contrast", "clean")
    if clean_g is not None and clean_c is not None and abs(clean_g - clean_c) > CLEAN_TOL:
        raise TableError(f"{label}: W2's clean change differs between the instances ({clean_g}, {clean_c})")
    clean = clean_g if clean_g is not None else clean_c
    g1, c1 = value(records, W1_GAUSSIAN, "gain"), value(records, W1_CONTRAST, "gain")
    new = None if c2 is None or c1 is None else c2 - c1
    old = None if g2 is None or g1 is None else g2 - g1
    marginal = {"contrast W1 to W2 >= 10": None if new is None else new >= MARGINAL_GAIN,
                "gaussian W1 to W2 >= -3": None if old is None else old >= -FORGET_MAX,
                "clean W0 to W2 >= -2.0": None if clean is None else clean >= CLEAN_MIN}
    mean_gain = None if g2 is None or c2 is None else (g2 + c2) / 2
    forgetting = None if old is None else -old
    formal = {"mean gain >= 10": None if mean_gain is None else mean_gain >= MEAN_GAIN_MIN,
              "clean >= -2.0": None if clean is None else clean >= CLEAN_MIN,
              "forgetting <= 3": None if forgetting is None else forgetting <= FORGET_MAX}
    return {"W2": {"gaussian": g2, "contrast": c2, "clean": clean}, "contrast_W1_to_W2": new,
            "gaussian_W1_to_W2": old, "marginal": marginal, "marginal_reading": met(marginal),
            "mean_gain": mean_gain, "max_forgetting": forgetting, "formal": formal, "formal_reading": met(formal),
            "stream_joint": None if mean_gain is None or clean is None else mean_gain + clean}


def build(v4: Mapping[str, Mapping[str, Any]], earlier: Mapping[str, Mapping[str, Any]],
          listed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    both = {**earlier, **v4}
    w1 = {"gaussian": value(both, W1_GAUSSIAN, "gain"), "contrast": value(both, W1_CONTRAST, "gain"),
          "clean": value(both, W1_CONTRAST, "clean"), "clean_on_i00802425": value(both, W1_GAUSSIAN, "clean")}
    if w1["clean"] is not None and w1["clean_on_i00802425"] is not None and \
            abs(w1["clean"] - w1["clean_on_i00802425"]) > CLEAN_TOL:
        raise TableError(f"W1's clean change differs between the instances ({w1})")
    streams = {label: {"arm": arm, **stream(both, label)} for label, arm in STREAMS}
    arms: dict[str, Any] = {}
    for arm in ("L", "N"):
        rows = [s for s in streams.values() if s["arm"] == arm]
        arms[arm] = {"met": sum(1 for s in rows if s["marginal_reading"] == "met"), "launched": len(rows),
                     "unreadable": sum(1 for s in rows if s["marginal_reading"].startswith("not readable"))}
    if arms["L"]["unreadable"] or arms["N"]["unreadable"]:
        t16 = "not readable (an entry has no record)"
    elif arms["L"]["met"] >= 2 and arms["N"]["met"] <= 1:
        t16 = "L met in at least 2 of 3 and N in at most 1 of 3: the edit history changes the outcome"
    else:
        t16 = "otherwise: the log in this form does not change the outcome at this budget"
    return {"W1": w1, "streams": streams, "arms": arms, "t16_reading": t16,
            "all_met": sum(1 for s in streams.values() if s["marginal_reading"] == "met"),
            "missing": sorted(set(listed) - set(v4))}


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
    v4 = load(args.results, args.mode, args.list_sha, listed, None)
    earlier = load(args.earlier_results, args.mode, args.earlier_list_sha, entries(args.earlier_list), {W1_GAUSSIAN})
    if W1_GAUSSIAN not in earlier:
        raise TableError(f"no earlier record of {W1_GAUSSIAN}")
    result = build(v4, earlier, listed)
    result["inputs"] = {"results": [str(p) for p in args.results], "list": str(args.list),
                        "earlier_results": [str(p) for p in args.earlier_results],
                        "earlier_list": str(args.earlier_list), "mode": args.mode, "list_sha256": args.list_sha,
                        "earlier_list_sha256": args.earlier_list_sha, "n_records": len(v4)}
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"records {len(v4)} of {len(listed)}; missing {result['missing']}; W1 {result['W1']}")
    for label, s in result["streams"].items():
        if s["contrast_W1_to_W2"] is None:
            print(f"{label} ({s['arm']}): {s['marginal_reading']}")
            continue
        print(f"{label} ({s['arm']}): contrast W1->W2 {s['contrast_W1_to_W2']:+.3f}, gaussian W1->W2 "
              f"{s['gaussian_W1_to_W2']:+.3f}, clean {s['W2']['clean']:+.3f}: marginal {s['marginal_reading']}; "
              f"formal {s['formal_reading']} (mean {s['mean_gain']:+.3f}, joint {s['stream_joint']:+.3f})")
    print("arms:", json.dumps(result["arms"]), "|", result["t16_reading"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
