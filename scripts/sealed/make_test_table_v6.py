"""The T04 test table and readings from the one sealed read of list v6 (PREREG "T04", sealed test, "Readings of list
v6"), or from the dev check records of the same entries.

Written and tested before the read. Every reading is computed mechanically from the records:

- per entry, against the instance's base: gain, clean and joint in points, fixed and broken on both sets, and (sealed
  mode, where the records carry per-image correctness) a paired 95% interval of each: the normal approximation of the
  per-image difference, 100 x 1.96 x sd(d, ddof=1) / sqrt(n), as T05 (scripts/t05/t05_readout.py). For gain d is the
  child's minus the base's correctness on the condition row, for clean on its clean companion, for joint their sum;
- per LEARN final, B at test level: gain >= G, clean >= -1.0 and joint >= the bar source's test joint + 1, the bar's
  definition applied to test numbers as for the earlier lists (make_test_table.py); the dev bar is reported beside;
- per LEARN final, the comparison with fine-tuning at matched information: its joint against every arm-3 kept pick of
  record (an arm-3 pick that is the base counts as 0), and the paired interval of the final minus the main control;
- the secondary readout (sealed mode): each named network's gain per set, points only, descriptive.

Checks that stop the table: every record matches its list entry (name, instance, child sha256, readout), and in
sealed mode its mode and the list's canonical sha256; the primary n is 38,000 (sealed) or 4,000 (dev) everywhere;
every entry on an instance has the same base correctness; a record's per-image bits reproduce its gain and clean.

    python scripts/sealed/make_test_table_v6.py --list scripts/sealed/finals_v6.json \\
        --results $WEIGHTCODE_SEALED/results/<sha>.jsonl --out <table.json> [--dev-records <dev jsonl> ...]
    python scripts/sealed/make_test_table_v6.py --list scripts/sealed/finals_v6.json --mode dev \\
        --dev-records <dev jsonl> ... --out <table.json>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit.sealed import manifest_sha256
from weightcode.evalkit.sealed_range import decode_bits

SCHEMA = "weightcode.t04_test_table_v6.v1"
ROLES_FILE = "finals_v6_roles.json"
G = {"gaussian_noise_s3": 7.0, "contrast_s4": 8.5}
"""PREREG T04: G = 0.15 x the base's V_select headroom, rounded to 0.5 (DEV_RESULTS "T04 build")."""
DEV_BAR = {"gaussian_noise_s3": 9.0, "contrast_s4": 6.325}
"""The bars of record on V_select after the edge extension (DEV_RESULTS "T04 references"), reported beside."""
B_CLEAN, B_MARGIN = -1.0, 1.0
Z95 = 1.96
N_PRIMARY = {"sealed": 38_000, "dev": 4_000}
TOL = 1e-9


def at_least(value: float, bound: float) -> bool:
    """value >= bound up to float rounding of sums of points (points are multiples of 100 / n)."""
    return value >= bound - TOL


class TableError(RuntimeError):
    """A check that stops the table."""


def interval(d: np.ndarray) -> dict[str, float]:
    """Point and paired 95% normal interval of mean(d), in points."""
    d = np.asarray(d, dtype=np.float64)
    point = 100.0 * float(d.mean())
    half = 100.0 * Z95 * float(d.std(ddof=1)) / math.sqrt(len(d))
    return {"point": point, "lo": point - half, "hi": point + half}


def load_roles(list_path: Path, list_sha: str) -> list[dict[str, Any]]:
    """The role map beside the list; refused unless it names the list's entries."""
    roles = json.loads((list_path.parent / ROLES_FILE).read_text(encoding="utf-8"))
    if roles.get("schema") != "weightcode.t04_v6_roles.v1":
        raise TableError(f"{ROLES_FILE}: unexpected schema {roles.get('schema')!r}")
    return list(roles["roles"])


def read_records(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Records by entry name; a name twice is refused."""
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["name"] in out:
                raise TableError(f"{record['name']}: two records")
            out[record["name"]] = record
    return out


def check_records(records: Mapping[str, Mapping[str, Any]], entries: Sequence[SF.Final], mode: str,
                  list_sha: str) -> None:
    """The checks that stop the table (module docstring), except the per-image ones."""
    names = {e.name for e in entries}
    if set(records) != names:
        raise TableError(f"records {sorted(set(records) ^ names)} do not match the list")
    for entry in entries:
        record = records[entry.name]
        for key, ours in (("instance_id", entry.instance_id), ("child_sha256", entry.child_sha256),
                          ("readout", entry.readout)):
            if record[key] != ours:
                raise TableError(f"{entry.name}: {key} {record[key]!r} != list {ours!r}")
        if record["mode"] != mode:
            raise TableError(f"{entry.name}: mode {record['mode']!r}, expected {mode!r}")
        if mode == "sealed" and record["list_sha256"] != list_sha:
            raise TableError(f"{entry.name}: list sha256 {record['list_sha256']} != {list_sha}")
        for part in ("readouts", "readouts_clean"):
            if int(record[part]["n"]) != N_PRIMARY[mode]:
                raise TableError(f"{entry.name}: {part} n {record[part]['n']} != {N_PRIMARY[mode]}")


def correctness(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Per-image correctness of base and child on the condition and clean rows, checked against the readouts."""
    out = {}
    for part, bits_key, points_key in (("cond", "per_image", "gain"), ("clean", "per_image_clean", "clean")):
        bits = record[bits_key]
        n = int(bits["n"])
        base, child = decode_bits(bits["base_correct"], n), decode_bits(bits["child_correct"], n)
        point = 100.0 * float(np.mean(child.astype(np.int8) - base.astype(np.int8)))
        if abs(point - float(record[points_key])) > TOL:
            raise TableError(f"{record['name']}: per-image {part} {point} != record {record[points_key]}")
        out[f"base_{part}"], out[f"child_{part}"] = base, child
    return out


def entry_row(record: Mapping[str, Any], bits: Mapping[str, np.ndarray] | None) -> dict[str, Any]:
    """One entry's row of the table."""
    row: dict[str, Any] = {
        "name": record["name"], "instance_id": record["instance_id"], "condition": record["condition"],
        "child_sha256": record["child_sha256"], "gain": float(record["gain"]), "clean": float(record["clean"]),
        "joint": float(record["joint"]),
        "fixed": int(record["readouts"]["fixed"]), "broken": int(record["readouts"]["broken"]),
        "fixed_clean": int(record["readouts_clean"]["fixed"]), "broken_clean": int(record["readouts_clean"]["broken"]),
        "base_acc": float(record["readouts"]["base_acc"]), "child_acc": float(record["readouts"]["child_acc"]),
        "base_acc_clean": float(record["readouts_clean"]["base_acc"]),
        "child_acc_clean": float(record["readouts_clean"]["child_acc"]),
    }
    if bits is not None:
        d_gain = bits["child_cond"].astype(np.int8) - bits["base_cond"].astype(np.int8)
        d_clean = bits["child_clean"].astype(np.int8) - bits["base_clean"].astype(np.int8)
        row["intervals"] = {"gain": interval(d_gain), "clean": interval(d_clean), "joint": interval(d_gain + d_clean)}
    if record.get("secondary"):
        row["secondary_gain"] = {s: float(part["gain"]) for s, part in sorted(record["secondary"].items())}
    return row


def readings(rows: Mapping[str, Mapping[str, Any]], roles: Sequence[Mapping[str, Any]],
             bits: Mapping[str, Mapping[str, np.ndarray]] | None, dev_rows: Mapping[str, Mapping[str, Any]] | None
             ) -> dict[str, Any]:
    """B at test level and the comparison with fine-tuning, per LEARN final (module docstring)."""
    out: dict[str, Any] = {}
    for condition in G:
        mine = [r for r in roles if r["condition"] == condition]
        finals = [r for r in mine if r["key"].startswith("LEARN_")]
        bar_roles = [r for r in mine if r["key"].startswith("bar_")]
        main_roles = [r for r in mine if r["key"].startswith("main_")]
        arm3 = [r for r in mine if r.get("arm") == "3"]
        if len(finals) != 2 or len(bar_roles) != 1 or len(main_roles) != 1:
            raise TableError(f"{condition}: roles {len(finals)} finals, {len(bar_roles)} bar, {len(main_roles)} main")
        bar_source, main = rows[bar_roles[0]["entry"]], rows[main_roles[0]["entry"]]
        bar = bar_source["joint"] + B_MARGIN
        ft = {r["key"]: rows[r["entry"]]["joint"] for r in arm3}
        best_ft_key = max(ft, key=lambda k: ft[k])
        per_final = {}
        for role in finals:
            row = rows[role["entry"]]
            checks = {"gain": at_least(row["gain"], G[condition]), "clean": at_least(row["clean"], B_CLEAN),
                      "joint": at_least(row["joint"], bar)}
            reading: dict[str, Any] = {
                "entry": role["entry"], "gain": row["gain"], "clean": row["clean"], "joint": row["joint"],
                "B_checks": checks, "B_met": all(checks.values()),
                "joint_minus_best_ft": row["joint"] - ft[best_ft_key], "best_ft": best_ft_key,
                "above_every_ft": all(row["joint"] > v for v in ft.values()),
                "joint_minus_main": row["joint"] - main["joint"],
            }
            if bits is not None:
                f, m = bits[role["entry"]], bits[main_roles[0]["entry"]]
                d = (f["child_cond"].astype(np.int8) - m["child_cond"].astype(np.int8)) \
                    + (f["child_clean"].astype(np.int8) - m["child_clean"].astype(np.int8))
                diff = interval(d)
                if abs(diff["point"] - reading["joint_minus_main"]) > 1e-6:
                    raise TableError(f"{role['entry']}: paired point {diff['point']} != {reading['joint_minus_main']}")
                reading["joint_minus_main_interval"] = diff
            if dev_rows is not None:
                dev = dev_rows[role["entry"]]
                dev_bar = dev_rows[bar_roles[0]["entry"]]["joint"] + B_MARGIN
                dev_met = (at_least(dev["gain"], G[condition]) and at_least(dev["clean"], B_CLEAN)
                           and at_least(dev["joint"], dev_bar))
                reading["dev"] = {"gain": dev["gain"], "clean": dev["clean"], "joint": dev["joint"], "bar": dev_bar,
                                  "B_met": dev_met}
                reading["dev_test_agree"] = dev_met == reading["B_met"]
            per_final[role["key"]] = reading
        out[condition] = {"G": G[condition], "bar": bar, "bar_source": bar_roles[0]["entry"],
                          "bar_source_joint": bar_source["joint"], "dev_bar_of_record": DEV_BAR[condition],
                          "main_control": main_roles[0]["entry"], "main_joint": main["joint"],
                          "ft_joints": ft, "finals": per_final,
                          "B_met": f"{sum(r['B_met'] for r in per_final.values())} of {len(per_final)}"}
    return out


def build(list_path: Path, mode: str, results: Path | None, dev_records: Sequence[Path]) -> dict[str, Any]:
    """The table (module docstring)."""
    parsed, entries = SF.load_list(list_path)
    list_sha = manifest_sha256(parsed)
    if mode == "sealed" and parsed["status"] != SF.STATUS_APPROVED:
        raise TableError(f"{list_path}: a sealed table needs the approved list")
    roles = load_roles(list_path, list_sha)
    role_entries = {r["entry"] for r in roles}
    if role_entries != {e.name for e in entries}:
        raise TableError(f"roles name {sorted(role_entries ^ {e.name for e in entries})} outside the list")
    dev = read_records(dev_records) if dev_records else None
    if dev is not None:
        check_records(dev, entries, "dev", list_sha)
    if mode == "sealed":
        if results is None:
            raise TableError("sealed mode needs --results")
        records = read_records([results])
        check_records(records, entries, "sealed", list_sha)
        bits = {name: correctness(record) for name, record in records.items()}
        by_instance: dict[str, set[tuple[str, str]]] = {}
        for name, record in records.items():
            key = (record["per_image"]["base_correct_sha256"], record["per_image_clean"]["base_correct_sha256"])
            by_instance.setdefault(record["instance_id"], set()).add(key)
        if any(len(keys) != 1 for keys in by_instance.values()):
            raise TableError("entries on one instance differ in base correctness")
    else:
        if dev is None:
            raise TableError("dev mode needs --dev-records")
        records, bits = dev, None
    rows = {name: entry_row(record, bits[name] if bits else None) for name, record in records.items()}
    dev_rows = {name: entry_row(record, None) for name, record in dev.items()} if dev is not None else None
    return {"schema": SCHEMA, "mode": mode, "list": str(list_path), "list_sha256": list_sha,
            "list_status": parsed["status"], "results": str(results) if results else None,
            "dev_records": [str(p) for p in dev_records], "n_primary": N_PRIMARY[mode],
            "interval": "paired 95% normal approximation of the per-image difference (T05's rule)",
            "entries": [rows[e.name] for e in entries], "readings": readings(rows, roles, bits, dev_rows)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--mode", choices=("sealed", "dev"), default="sealed")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--dev-records", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        table = build(args.list, args.mode, args.results, args.dev_records)
    except (TableError, SF.SealedFinalsError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as handle:
        json.dump(table, handle, indent=1, sort_keys=True, allow_nan=False)
    print(f"table: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
