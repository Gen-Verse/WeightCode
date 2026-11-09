"""Paper table cells and bar readings from sealed-read records (or dev dry-run records of the same entries).

Inputs: the read's result records (``weightcode.sealed_finals_result.v1`` lines), the approved list and its cell map
(``finals_v2_cells.json``). Output: one JSON file with

- ``cells``: every test-swapped cell of Paper/manuscript/tables/table_data.json (tab:main, tab:examples, tab:reuse,
  tab:stack), keyed by the paper's row ids, in points (shares as ratios);
- ``entries``: gain, clean and joint of every list entry, and the entries without a record (failed, spent);
- ``bars``: each final's prespecified reading (PREREG) evaluated on these numbers, met or missed.

Cell definitions follow the dev numbers they replace (DEV_RESULTS): net = joint = gain + clean; fine-tuning and
reference cells are joints, except ``ft_free_clean``; a reuse share is the mean gain over seeds 101 to 105 divided by
the seed-100 gain of the same method (the program's B final, or the F1 kept pick for the recipe and the
re-selection); a stream score is the mean gain over the four conditions plus the mean clean change after that round;
max forgetting is the largest (gain after the condition's own round) minus (gain after the last round) over the
conditions seen before the last round, signed, so a negative value means every earlier gain held or rose (the
definition of ``weightcode.harness.program_stream.stream_summary``). A cell whose child is a base no-op (cell map text
"base no-op" or "unchanged network") is 0 by construction. A cell whose entry has no record is null, never filled
from another source.

Usage: python scripts/sealed/make_test_table.py --results <jsonl> [<jsonl> ...] --list scripts/sealed/finals_v2.json
       --cells scripts/sealed/finals_v2_cells.json --out <json> [--mode sealed|dev] [--list-sha <canonical sha256>]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

NOOP_PREFIXES = ("base no-op", "unchanged network")
MAIN_ROWS = {"vit_gaussian": "vit_gauss", "vit_contrast": "vit_contrast", "r18_gaussian": "r18_gauss",
             "r18_contrast": "r18_contrast", "babyai": "babyai_pickup"}
STREAM_CONDITIONS = ("gaussian", "defocus", "snow", "contrast")  # stream order: round r learns condition r
STREAM_ROUNDS = ("W1", "W2", "W3", "W4")
SEEDS = ("101", "102", "103", "104", "105")
REUSE_SOURCE = {"gaussian": ("E11_B_gaussian_s100", "vit_gaussian"), "contrast": ("E11_B_contrast_s100", "vit_contrast")}
CLEAN_SPREAD_TOL = 1e-9

# Prespecified readings (PREREG; DEV_RESULTS gives each final's reading). Points.
B_GAIN, B_CLEAN, B_MARGIN = 10.0, -1.0, 1.0
C_RATIO, C_CLEAN, C_MIN_STUDENTS = 0.7, -1.0, 4
E13_EM, E13_CLEAN = 0.5, -1.0
STREAM_MEAN_GAIN, STREAM_CLEAN, STREAM_FORGET = 10.0, -2.0, 3.0
K1_CONTROL_JOINT = 0.0  # k1_ft_rule's kept pick is the base (DEV_RESULTS K1 control)


class TableError(RuntimeError):
    pass


def load_records(paths: Sequence[Path], mode: str, list_sha: str | None) -> dict[str, dict[str, Any]]:
    """The latest record of each entry name across ``paths``; every record must be of ``mode`` (and of ``list_sha``)."""
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            schemas = ("weightcode.sealed_finals_result.v1",) if mode == "sealed" else (
                "weightcode.sealed_finals_result.v0", "weightcode.sealed_finals_result.v1")  # v0: the first dev dry run
            if record.get("schema") not in schemas:
                raise TableError(f"{path}:{number}: schema {record.get('schema')!r}")
            if record.get("mode") != mode:
                raise TableError(f"{path}:{number}: mode {record.get('mode')!r}, expected {mode!r}")
            if list_sha is not None and record.get("list_sha256") != list_sha:
                raise TableError(f"{path}:{number}: list_sha256 {record.get('list_sha256')!r}, expected {list_sha}")
            for key in ("gain", "clean", "joint"):
                if not isinstance(record.get(key), (int, float)) or not math.isfinite(record[key]):
                    raise TableError(f"{path}:{number}: {record.get('name')}: {key} {record.get(key)!r}")
            if abs(record["gain"] + record["clean"] - record["joint"]) > 1e-9:
                raise TableError(f"{path}:{number}: {record['name']}: joint != gain + clean")
            if mode == "sealed" and record["name"] in records:
                raise TableError(f"{path}:{number}: {record['name']} appears twice in a sealed read")
            records[record["name"]] = record
    return records


def list_entries(list_path: Path) -> list[dict[str, Any]]:
    parsed = json.loads(list_path.read_text(encoding="utf-8"))
    return [dict(entry, block=block) for block in ("finals", "references") for entry in parsed[block]]


def is_noop(ref: str) -> bool:
    return any(ref.startswith(prefix) for prefix in NOOP_PREFIXES)


class Reader:
    """Looks up a cell-map reference (an entry name or a no-op text) in the records."""

    def __init__(self, records: Mapping[str, Mapping[str, Any]], names: set[str]) -> None:
        self.records, self.names, self.used = records, names, set()

    def get(self, ref: str) -> Mapping[str, Any] | None:
        if is_noop(ref):
            return {"gain": 0.0, "clean": 0.0, "joint": 0.0, "noop": True}
        if ref not in self.names:
            raise TableError(f"cell map names {ref!r}, which is not on the list")
        self.used.add(ref)
        return self.records.get(ref)

    def field(self, ref: str, key: str) -> float | None:
        record = self.get(ref)
        return None if record is None else float(record[key])


def mean_or_none(values: Sequence[float | None]) -> float | None:
    return None if any(v is None for v in values) else statistics.fmean(values)  # type: ignore[arg-type]


def ratio(num: float | None, den: float | None) -> float | None:
    return None if num is None or den is None or den == 0 else num / den


def main_cells(cmap: Mapping[str, Any], rd: Reader) -> dict[str, Any]:
    out = {}
    for key, row in cmap.items():
        program = row["program"]
        out[MAIN_ROWS[key]] = {
            "gain": rd.field(program, "gain"), "clean": rd.field(program, "clean"), "net": rd.field(program, "joint"),
            "ft_kept": rd.field(row["f1_kept"], "joint"), "ft_free": rd.field(row["f1_own"], "joint"),
            "ft_free_clean": rd.field(row["f1_own"], "clean"), "ref_kept": rd.field(row["fixed"], "joint")}
    return out


def example_cells(cmap: Mapping[str, Any], rd: Reader) -> dict[str, Any]:
    prefix = {"gaussian": "g", "contrast": "c"}
    out: dict[str, dict[str, float | None]] = {}
    for cond, rows in cmap.items():
        p = prefix[cond]
        for row_id, ref in rows.items():
            cells = out.setdefault(row_id, {})
            cells[f"{p}_gain"] = rd.field(ref, "gain")
            cells[f"{p}_clean"] = rd.field(ref, "clean")
            cells[f"{p}_net"] = rd.field(ref, "joint")
    return out


def reuse_cells(cmap: Mapping[str, Any], main_map: Mapping[str, Any], rd: Reader) -> tuple[dict[str, Any], dict]:
    out: dict[str, Any] = {}
    per_student: dict[str, Any] = {}
    for cond, seeds in cmap.items():
        source_final, main_key = REUSE_SOURCE[cond]
        own = {"program": rd.field(source_final, "gain"), "recipe": rd.field(main_map[main_key]["f1_kept"], "gain"),
               "reselect": rd.field(main_map[main_key]["f1_kept"], "gain")}
        out[cond], per_student[cond] = {}, {}
        for method in ("program", "recipe", "reselect"):
            gains = [rd.field(seeds[s][method], "gain") for s in SEEDS]
            cleans = [rd.field(seeds[s][method], "clean") for s in SEEDS]
            mean_gain = mean_or_none(gains)
            ratios = [ratio(g, own[method]) for g in gains]
            meets = None if any(r is None for r in ratios) or any(c is None for c in cleans) else sum(
                1 for r, c in zip(ratios, cleans) if r >= C_RATIO and c >= C_CLEAN)  # type: ignore[operator]
            out[cond][method] = {
                "gain": mean_gain, "share": ratio(mean_gain, own[method]),
                "clean_min": None if None in cleans else min(cleans),  # type: ignore[type-var]
                "clean_max": None if None in cleans else max(cleans),  # type: ignore[type-var]
                "n_clean_ok": None if None in cleans else sum(1 for c in cleans if c >= C_CLEAN),  # type: ignore[operator]
                "n_ratio_and_clean_ok": meets, "own_gain_s100": own[method]}
            per_student[cond][method] = {s: {"gain": g, "clean": c} for s, g, c in zip(SEEDS, gains, cleans)}
    out["seed_gains"] = [per_student["gaussian"]["program"][s]["gain"] for s in SEEDS]
    return out, per_student


def stream_cells(cmap: Mapping[str, Any], rd: Reader) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm, rounds in cmap.items():
        per_round = []
        for w in STREAM_ROUNDS:
            gains = [rd.field(rounds[w][c], "gain") for c in STREAM_CONDITIONS]
            cleans = [rd.field(rounds[w][c], "clean") for c in STREAM_CONDITIONS]
            spread = None if None in cleans else max(cleans) - min(cleans)  # type: ignore[type-var]
            if spread is not None and spread > CLEAN_SPREAD_TOL:
                raise TableError(f"stream {arm} {w}: the clean change differs across the four reads ({cleans})")
            mean_gain, clean = mean_or_none(gains), mean_or_none(cleans)
            per_round.append({"gains": dict(zip(STREAM_CONDITIONS, gains)), "clean": clean, "mean_gain": mean_gain,
                              "score": None if mean_gain is None or clean is None else mean_gain + clean})
        forgetting = {}
        for index, cond in enumerate(STREAM_CONDITIONS[:-1]):
            own, last = per_round[index]["gains"][cond], per_round[-1]["gains"][cond]
            forgetting[cond] = None if own is None or last is None else own - last
        values = list(forgetting.values())
        out[arm] = {"rounds": dict(zip(STREAM_ROUNDS, per_round)), "scores": [r["score"] for r in per_round],
                    "mean_gain": per_round[-1]["mean_gain"], "clean": per_round[-1]["clean"],
                    "forgetting": forgetting, "max_forgetting": None if None in values else max(values)}  # type: ignore[type-var]
    return out


def met(conditions: Mapping[str, bool | None]) -> str:
    if any(v is None for v in conditions.values()):
        return "not readable (an entry has no record)"
    return "met" if all(conditions.values()) else "missed"


def b_reading(rd: Reader, final: str, reference: str | None, reference_joint: float | None = None) -> dict[str, Any]:
    gain, clean, joint = (rd.field(final, k) for k in ("gain", "clean", "joint"))
    ref_joint = rd.field(reference, "joint") if reference is not None else reference_joint
    checks = {"gain >= 10": None if gain is None else gain >= B_GAIN,
              "clean >= -1.0": None if clean is None else clean >= B_CLEAN,
              "joint >= reference joint + 1": None if joint is None or ref_joint is None
              else joint >= ref_joint + B_MARGIN}
    return {"final": final, "reference": reference, "gain": gain, "clean": clean, "joint": joint,
            "reference_joint": ref_joint, "checks": checks, "reading": met(checks)}


def bars(cells_map: Mapping[str, Any], rd: Reader, reuse: Mapping[str, Any], stream: Mapping[str, Any]
         ) -> dict[str, Any]:
    main_map = cells_map["main"]
    out: dict[str, Any] = {}
    for key in ("vit_gaussian", "vit_contrast", "r18_gaussian", "r18_contrast", "babyai"):
        row = main_map[key]
        reading = b_reading(rd, row["program"], row["fixed"])
        kept = rd.field(row["f1_kept"], "joint")
        reading["F1 kept joint"] = kept
        reading["joint minus F1 kept joint"] = None if kept is None or reading["joint"] is None else reading["joint"] - kept
        out[f"B {row['program']}"] = reading
    e11_bar = main_map["vit_gaussian"]["fixed"]
    out["ladder U (B-style against the E11 gaussian bar)"] = b_reading(rd, "ladder_U_gaussian_s100", e11_bar)
    k1 = b_reading(rd, "ladder_K1_gaussian_s100", None, reference_joint=K1_CONTROL_JOINT)
    k1["beside: E11 gaussian bar (reference joint + 1)"] = (None if rd.field(e11_bar, "joint") is None
                                                            else rd.field(e11_bar, "joint") + B_MARGIN)
    out["ladder K1 (against k1_ft_rule's kept pick, the base)"] = k1
    for split in ("addprim_jump", "around_right"):
        record = rd.get(f"E13_{split}_s100")
        ref = rd.get(f"ref_E13_{split}_bar_rulegen_mixed_sparse19_sal")
        em = None if record is None else float(record["readouts_scan"]["new"]["em_child"])
        clean = None if record is None else float(record["clean"])
        checks = {"new EM >= 50%": None if em is None else em >= E13_EM,
                  "clean >= -1.0": None if clean is None else clean >= E13_CLEAN}
        out[f"E13 {split} graded"] = {
            "new_em_child": em, "new_em_base": None if record is None else record["readouts_scan"]["new"]["em_base"],
            "gain": None if record is None else record["gain"], "clean": clean,
            "reference_gain": None if ref is None else ref["gain"], "reference_clean": None if ref is None else ref["clean"],
            "reference_new_em_child": None if ref is None else ref["readouts_scan"]["new"]["em_child"],
            "checks": checks, "reading": met(checks)}
    for cond in ("gaussian", "contrast"):
        program = reuse[cond]["program"]
        n = program["n_ratio_and_clean_ok"]
        checks = {"reuse share >= 0.7": None if program["share"] is None else program["share"] >= C_RATIO,
                  "ratio >= 0.7 and clean >= -1.0 on >= 4 of 5": None if n is None else n >= C_MIN_STUDENTS}
        out[f"C {cond} (E11 C, seeds 101 to 105)"] = {
            "share": program["share"], "mean_gain": program["gain"], "n_students_ok": n,
            "recipe_share": reuse[cond]["recipe"]["share"], "recipe_gain": reuse[cond]["recipe"]["gain"],
            "program_minus_recipe_gain": None if program["gain"] is None or reuse[cond]["recipe"]["gain"] is None
            else program["gain"] - reuse[cond]["recipe"]["gain"], "checks": checks, "reading": met(checks)}
    teacher, ft = stream["program"], stream["ft"]
    checks = {"mean gain >= 10": None if teacher["mean_gain"] is None else teacher["mean_gain"] >= STREAM_MEAN_GAIN,
              "clean >= -2.0": None if teacher["clean"] is None else teacher["clean"] >= STREAM_CLEAN,
              "max forgetting <= 3": None if teacher["max_forgetting"] is None
              else teacher["max_forgetting"] <= STREAM_FORGET,
              "stream score above the F4 ft_replay stream": None if teacher["scores"][-1] is None
              or ft["scores"][-1] is None else teacher["scores"][-1] > ft["scores"][-1]}
    out["E10 stream v2 (absolute criteria and the F4 comparison)"] = {
        "score_W4": teacher["scores"][-1], "ft_score_W4": ft["scores"][-1],
        "score_minus_ft": None if teacher["scores"][-1] is None or ft["scores"][-1] is None
        else teacher["scores"][-1] - ft["scores"][-1], "checks": checks, "reading": met(checks)}
    return out


def build(records: Mapping[str, Mapping[str, Any]], entries: Sequence[Mapping[str, Any]],
          cells_map: Mapping[str, Any]) -> dict[str, Any]:
    names = {e["name"] for e in entries}
    rd = Reader(records, names)
    tables = cells_map["tables"]
    reuse, per_student = reuse_cells(tables["reuse"], tables["main"], rd)
    stream = stream_cells(tables["stack"], rd)
    cells = {"main": main_cells(tables["main"], rd), "examples": example_cells(tables["examples"], rd),
             "reuse": reuse, "stack": {"program": stream["program"]["scores"], "ft": stream["ft"]["scores"],
                                       "mean_gain": stream["program"]["mean_gain"],
                                       "clean": stream["program"]["clean"],
                                       "max_forgetting": stream["program"]["max_forgetting"]}}
    readings = bars(tables, rd, reuse, stream)
    listed = [{"name": e["name"], "block": e["block"], "readout": e.get("readout"), "instance_id": e.get("instance_id"),
               **({k: records[e["name"]][k] for k in ("gain", "clean", "joint")} if e["name"] in records
                  else {"gain": None, "clean": None, "joint": None, "missing": True})} for e in entries]
    return {"cells": cells, "bars": readings, "reuse_per_student": per_student, "stream": stream, "entries": listed,
            "missing": [e["name"] for e in entries if e["name"] not in records],
            "not_used_by_any_cell": sorted(names - rd.used)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("sealed", "dev"), default="sealed")
    parser.add_argument("--list-sha", default=None, help="the approved list's canonical sha256 (sealed mode)")
    args = parser.parse_args(argv)
    if args.mode == "sealed" and args.list_sha is None:
        parser.error("--list-sha is required in sealed mode")
    records = load_records(args.results, args.mode, args.list_sha)
    entries = list_entries(args.list)
    cells_map = json.loads(args.cells.read_text(encoding="utf-8"))
    result = build(records, entries, cells_map)
    result["inputs"] = {"results": [str(p) for p in args.results], "list": str(args.list), "cells": str(args.cells),
                        "mode": args.mode, "list_sha256": args.list_sha, "n_records": len(records)}
    args.out.write_text(json.dumps(result, indent=1, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"records {len(records)}, entries {len(entries)}, missing {len(result['missing'])}: {result['missing']}")
    for name, reading in result["bars"].items():
        print(f"{name}: {reading['reading']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
