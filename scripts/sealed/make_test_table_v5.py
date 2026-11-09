"""Readings of list v5 (PREREG "T07 sealed read (list v5)"): T07's two tables at test level.

Inputs: the v5 records (the vision_range entries), the records of the earlier lists for the cells the reuse map
(``finals_v5_reused.json``) takes from them, and T07's dev table of record. Per architecture, network and set, with
T07's cell rule (``scripts/t07/t07_range.py``: change against the base in points, image-level paired bootstrap of
1,000 resamples from numpy default_rng(20261107), one index matrix of the cell size n for every contrast, percentile
95%, help / harm / neutral labels, the base's loss from clean and the recovered share):

- a new cell (a v5 entry) is computed from the record's per-image correctness of base and child;
- a reused cell (an earlier list's read) has counts only: its interval is the bootstrap of the canonical vector
  (``fixed`` rows +1, then ``broken`` rows -1, the rest 0) under the same index matrix. The point, fixed, broken and
  accuracies are the record's; the interval differs from a per-image one by Monte Carlo noise only, and the row says
  which kind it is;
- final against fine-tuning (``llm_minus_ft``): the paired bootstrap of the two networks' per-image correctness where
  both cells are new, otherwise the point (difference of the two changes) with no interval.

Checks before any reading: every record matches its list entry (instance, child; sealed mode: the list's canonical
sha256), every cell has the same n, a new cell's bootstrap point equals its readout's change, every network of an
architecture has the same base correctness on a set (new cells: the per-image sha256; reused cells: base accuracy
within 1e-12), and in dev mode every v5 record's reference check matched. Readings: the test label of each cell, the
help and harm sets per network (T07's applicability range), and per network the number of the 26 sets whose test
label equals the dev label of T07's table (and the help/harm reversals). In dev mode the script also checks that it
reproduces T07's dev table: every field of a new cell and of a paired llm_minus_ft row within 1e-9, the point fields of
the rest.

    .venv/bin/python scripts/sealed/make_test_table_v5.py --mode sealed|dev \
        --list scripts/sealed/finals_v5.json --reuse scripts/sealed/finals_v5_reused.json --results <v5 jsonl|dir> \
        --earlier-list scripts/sealed/finals_v2.json scripts/sealed/finals_v4.json \
        --earlier-results <v2 jsonl|dir> <v4 jsonl|dir> --t07 <runs>/t07/<run>/t07_table.json --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t07"))
import t07_range as T  # noqa: E402

from weightcode.evalkit.sealed import manifest_sha256  # noqa: E402
from weightcode.evalkit.sealed_range import decode_bits  # noqa: E402

SCHEMA = "weightcode.t07_v5_table.v1"
T07_RECORD_SHA256 = "55982f0da04b35140b494022768e64e4f0aa2115f7aff4084fbd2453265fa556"
ARCHS = ("vit", "r18")
FINALS = ("llm_gaussian_noise", "llm_contrast")
NETWORKS = ("llm_gaussian_noise", "llm_contrast", "ft_gaussian_noise", "ft_contrast")
CONDITIONS = ("gaussian_noise", "contrast")
MODES = ("sealed", "dev")
POINTS_TOL, ACC_TOL = 1e-9, 1e-12
CELL_FIELDS = ("delta_points", "ci95", "boot_sd", "label", "fixed", "broken", "worst_class_delta_points", "base_acc",
               "child_acc", "n", "base_loss_from_clean_points", "recovered_share")
POINT_FIELDS = ("delta_points", "label", "fixed", "broken", "worst_class_delta_points", "base_acc", "child_acc", "n",
                "base_loss_from_clean_points", "recovered_share")


class TableError(RuntimeError):
    pass


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _files(paths: Sequence[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        out.extend(sorted(path.glob("*.jsonl")) if path.is_dir() else [path])
    return out


def entries_of(list_path: Path) -> dict[str, dict[str, Any]]:
    parsed = _json(list_path)
    return {e["name"]: e for block in ("finals", "references") for e in parsed[block]}


def load_records(paths: Sequence[Path], mode: str, lists: Mapping[str, Path], wanted: Mapping[str, str]
                 ) -> dict[str, dict[str, Any]]:
    """Records by name for the names in ``wanted`` (name -> list file name), each checked against its list entry."""
    listed = {name: entries_of(path) for name, path in lists.items()}
    shas = {name: manifest_sha256(_json(path)) for name, path in lists.items()}
    records: dict[str, dict[str, Any]] = {}
    for path in _files(paths):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            name = record.get("name")
            if name not in wanted:
                continue
            where = f"{path}:{number}: {name}"
            list_name = wanted[name]
            entry = listed[list_name].get(name)
            if entry is None:
                raise TableError(f"{where}: not on {list_name}")
            if record.get("mode") != mode:
                raise TableError(f"{where}: mode {record.get('mode')!r}, expected {mode!r}")
            if mode == "sealed" and record.get("list_sha256") != shas[list_name]:
                raise TableError(f"{where}: list_sha256 {record.get('list_sha256')!r}, {list_name} is {shas[list_name]}")
            if record.get("instance_id") != entry["instance_id"] or record.get("child_sha256") != entry["child_sha256"]:
                raise TableError(f"{where}: instance or child differs from the {list_name} entry")
            if name in records and records[name] != record:
                raise TableError(f"{where}: two different records")
            records[name] = record
    missing = sorted(set(wanted) - set(records))
    if missing:
        raise TableError(f"no record of {missing}; a cell is never filled from another source")
    return records


def cell_data(cell: Mapping[str, Any], record: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """The readout of one cell, and its per-image correctness (new) or its canonical vector (reused)."""
    readouts = record[cell["field"]]
    n = int(readouts["n"])
    if cell["status"] == "new":
        if record.get("readout") != "vision_range" or record["source"].get("set") != cell["set"]:
            raise TableError(f"{cell['entry']}: not the vision_range read of {cell['set']}")
        if mode == "dev" and not (record.get("reference") or {}).get("match"):
            raise TableError(f"{cell['entry']}: its dev check did not match its T07 record")
        bits = record["per_image"]
        if int(bits["n"]) != n:
            raise TableError(f"{cell['entry']}: per_image n {bits['n']} != readout n {n}")
        for key in ("base_correct", "child_correct"):
            if hashlib.sha256(bits[key].encode("ascii")).hexdigest() != bits[f"{key}_sha256"]:
                raise TableError(f"{cell['entry']}: {key} does not hash to its record")
        base, child = decode_bits(bits["base_correct"], n), decode_bits(bits["child_correct"], n)
        if int(np.sum(child & ~base)) != int(readouts["fixed"]) or int(np.sum(base & ~child)) != int(readouts["broken"]):
            raise TableError(f"{cell['entry']}: per-image bits do not give the readout's fixed and broken")
        values = child.astype(np.int64) - base.astype(np.int64)
        return {"readouts": readouts, "n": n, "values": values, "base": base, "child": child,
                "base_sha256": bits["base_correct_sha256"], "interval": "paired image bootstrap (per-image record)"}
    fixed, broken = int(readouts["fixed"]), int(readouts["broken"])
    values = np.zeros(n, np.int64)
    values[:fixed] = 1
    values[fixed:fixed + broken] = -1
    return {"readouts": readouts, "n": n, "values": values, "base": None, "child": None, "base_sha256": None,
            "interval": "bootstrap of the canonical vector (counts-only record)"}


def build(reuse: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], t07: Mapping[str, Any], mode: str
          ) -> dict[str, Any]:
    sets = list(t07["sets"])
    cells = {(c["arch"], c["network"], c["set"]): c for c in reuse["cells"]}
    if sorted(cells) != sorted((a, nw, s) for a in ARCHS for nw in NETWORKS for s in sets):
        raise TableError("the reuse map does not hold one cell per (network, set) of T07's table")
    data = {key: cell_data(c, records[c["entry"]], mode) for key, c in cells.items()}
    sizes = {d["n"] for d in data.values()}
    if len(sizes) != 1:
        raise TableError(f"cells of different sizes {sorted(sizes)}")
    (n,) = sizes
    index = T.boot_index(n)
    base_acc: dict[tuple[str, str], float] = {}
    for arch in ARCHS:
        for s in sets:
            group = [data[(arch, nw, s)] for nw in NETWORKS]
            hashes = {d["base_sha256"] for d in group if d["base_sha256"] is not None}
            if len(hashes) > 1:
                raise TableError(f"{arch} {s}: the networks' bases are not correct on the same images")
            accs = [float(d["readouts"]["base_acc"]) for d in group]
            if max(accs) - min(accs) > ACC_TOL:
                raise TableError(f"{arch} {s}: base accuracy differs between networks {accs}")
            base_acc[(arch, s)] = accs[0]
    rows: list[dict[str, Any]] = []
    for arch in ARCHS:
        for nw in NETWORKS:
            for s in sets:
                d, c = data[(arch, nw, s)], cells[(arch, nw, s)]
                ro = d["readouts"]
                boot = T.boot_contrast(d["values"], index)
                if not math.isclose(boot["point"], 100.0 * float(ro["delta_acc"]), rel_tol=0.0, abs_tol=POINTS_TOL):
                    raise TableError(f"{arch} {nw} {s}: bootstrap point {boot['point']} != readout change")
                headroom, share = T.recovered_share(boot["point"], base_acc[(arch, "clean")], base_acc[(arch, s)])
                rows.append({"arch": arch, "network": nw, "set": s, "group": T.C.group_of(s),
                             "severity": T.C.severity_of(s), "delta_points": boot["point"], "ci95": boot["ci95"],
                             "boot_sd": boot["boot_sd"], "label": T.label(boot["point"], boot["ci95"]),
                             "fixed": int(ro["fixed"]), "broken": int(ro["broken"]),
                             "worst_class_delta_points": 100.0 * float(ro["worst_class_delta"]),
                             "base_acc": float(ro["base_acc"]), "child_acc": float(ro["child_acc"]), "n": int(ro["n"]),
                             "base_loss_from_clean_points": headroom, "recovered_share": share,
                             "source": {k: c[k] for k in ("status", "list", "entry", "field")},
                             "interval": d["interval"]})
    llm_vs_ft: list[dict[str, Any]] = []
    for arch in ARCHS:
        for condition in CONDITIONS:
            for s in sets:
                a, b = data[(arch, f"llm_{condition}", s)], data[(arch, f"ft_{condition}", s)]
                row: dict[str, Any] = {"arch": arch, "condition": condition, "set": s, "group": T.C.group_of(s), "n": n}
                if a["child"] is not None and b["child"] is not None:
                    boot = T.boot_contrast(a["child"].astype(np.int64) - b["child"].astype(np.int64), index)
                    row.update(llm_minus_ft_points=boot["point"], ci95=boot["ci95"], paired=True)
                else:
                    point = 100.0 * (float(a["readouts"]["child_acc"]) - float(b["readouts"]["child_acc"]))
                    row.update(llm_minus_ft_points=point, ci95=None, paired=False,
                               note="point only: a counts-only earlier record has no per-image correctness")
                llm_vs_ft.append(row)
    return {"n": n, "sets": sets, "rows": rows, "llm_minus_ft": llm_vs_ft}


def readings(table: Mapping[str, Any], t07: Mapping[str, Any]) -> dict[str, Any]:
    """Applicability range per network and dev-test label agreement against T07's table."""
    dev = {(r["arch"], r["network"], r["set"]): r for r in t07["rows"]}
    out: dict[str, Any] = {}
    for arch in ARCHS:
        for nw in NETWORKS:
            rows = [r for r in table["rows"] if (r["arch"], r["network"]) == (arch, nw)]
            changed = [{"set": r["set"], "dev": dev[(arch, nw, r["set"])]["label"], "test": r["label"]}
                       for r in rows if dev[(arch, nw, r["set"])]["label"] != r["label"]]
            out[f"{arch} {nw}"] = {
                "help_sets": [r["set"] for r in rows if r["label"] == "help"],
                "harm_sets": [r["set"] for r in rows if r["label"] == "harm"],
                "n_sets": len(rows), "n_same_label_as_dev": len(rows) - len(changed),
                "n_reversed": sum(1 for c in changed if {c["dev"], c["test"]} == {"help", "harm"}),
                "changed": changed}
    return out


def reproduction(table: Mapping[str, Any], t07: Mapping[str, Any]) -> list[str]:
    """Dev mode: where the table differs from T07's dev table (every field of new cells and paired rows)."""
    def same(a: Any, b: Any) -> bool:
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, float) or isinstance(b, float):
            return a is not None and b is not None and math.isclose(a, b, rel_tol=0.0, abs_tol=POINTS_TOL)
        return a == b

    problems: list[str] = []
    dev = {(r["arch"], r["network"], r["set"]): r for r in t07["rows"]}
    for r in table["rows"]:
        fields = CELL_FIELDS if r["source"]["status"] == "new" else POINT_FIELDS
        if r["source"]["status"] != "new" and r["label"] != dev[(r["arch"], r["network"], r["set"])]["label"]:
            fields = tuple(f for f in POINT_FIELDS if f != "label")  # a canonical interval may move a label
            problems.append(f"{r['arch']} {r['network']} {r['set']}: label {r['label']} (canonical interval) vs dev "
                            f"{dev[(r['arch'], r['network'], r['set'])]['label']} (reused cell; reported, not an error)")
        problems += [f"{r['arch']} {r['network']} {r['set']} {f}: {r[f]} vs {dev[(r['arch'], r['network'], r['set'])][f]}"
                     for f in fields if not same(r[f], dev[(r["arch"], r["network"], r["set"])][f])]
    dev_ft = {(r["arch"], r["condition"], r["set"]): r for r in t07["llm_minus_ft"]}
    for r in table["llm_minus_ft"]:
        ref = dev_ft[(r["arch"], r["condition"], r["set"])]
        keys = ("llm_minus_ft_points", "ci95") if r["paired"] else ("llm_minus_ft_points",)
        problems += [f"llm_minus_ft {r['arch']} {r['condition']} {r['set']} {k}: {r[k]} vs {ref[k]}"
                     for k in keys if not same(r[k], ref[k])]
    return problems


def print_tables(table: Mapping[str, Any], read: Mapping[str, Any], mode: str) -> None:
    for arch in ARCHS:
        print(f"\n{arch} ({mode}): change in points vs base (95% CI) [label]; n = {table['n']} per set; "
              "* = interval from a counts-only earlier record")
        print("set".ljust(22) + "".join(nw.rjust(32) for nw in NETWORKS))
        for s in table["sets"]:
            cells = []
            for nw in NETWORKS:
                (r,) = [r for r in table["rows"] if (r["arch"], r["network"], r["set"]) == (arch, nw, s)]
                mark = "" if r["source"]["status"] == "new" else "*"
                cells.append(f"{r['delta_points']:+.2f} ({r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}) "
                             f"{r['label'][:4]}{mark}")
            print(s.ljust(22) + "".join(c.rjust(32) for c in cells))
        print(f"\n{arch} ({mode}): final minus fine-tuning, points (paired 95% CI; point only where a side is reused)")
        for s in table["sets"]:
            cells = []
            for condition in CONDITIONS:
                (r,) = [r for r in table["llm_minus_ft"] if (r["arch"], r["condition"], r["set"]) == (arch, condition, s)]
                ci = f"({r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f})" if r["ci95"] else "(point only)"
                cells.append(f"{r['llm_minus_ft_points']:+.2f} {ci}")
            print(s.ljust(22) + "".join(c.rjust(32) for c in cells))
    print("\ndev-test label agreement per network (T07 dev table vs this table):")
    for key, r in read.items():
        print(f"  {key:<24} same {r['n_same_label_as_dev']}/{r['n_sets']}, reversed {r['n_reversed']}; help "
              f"{len(r['help_sets'])}, harm {len(r['harm_sets'])}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--list", type=Path, required=True, help="scripts/sealed/finals_v5.json")
    parser.add_argument("--reuse", type=Path, required=True, help="scripts/sealed/finals_v5_reused.json")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="v5 records: jsonl files or folders")
    parser.add_argument("--earlier-list", type=Path, nargs="+", required=True)
    parser.add_argument("--earlier-results", type=Path, nargs="+", required=True)
    parser.add_argument("--t07", type=Path, required=True, help="T07's t07_table.json of record")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if hashlib.sha256(args.t07.read_bytes()).hexdigest() != T07_RECORD_SHA256:
        raise TableError(f"{args.t07}: not T07's table of record")
    reuse, t07 = _json(args.reuse), _json(args.t07)
    if reuse.get("t07_record_sha256") != T07_RECORD_SHA256 or reuse.get("list") != args.list.name:
        raise TableError(f"{args.reuse}: not the reuse map of {args.list.name} and T07's table of record")
    earlier = {p.name: p for p in args.earlier_list}
    new_wanted = {c["entry"]: args.list.name for c in reuse["cells"] if c["status"] == "new"}
    old_wanted = {c["entry"]: c["list"] for c in reuse["cells"] if c["status"] == "reused"}
    if not set(old_wanted.values()) <= set(earlier):
        raise TableError(f"the reuse map needs {sorted(set(old_wanted.values()))}; given {sorted(earlier)}")
    records = {**load_records(args.results, args.mode, {args.list.name: args.list}, new_wanted),
               **load_records(args.earlier_results, args.mode, earlier, old_wanted)}
    table = build(reuse, records, t07, args.mode)
    read = readings(table, t07)
    out: dict[str, Any] = {
        "schema": SCHEMA, "mode": args.mode, "list": str(args.list), "list_sha256": manifest_sha256(_json(args.list)),
        "reuse": str(args.reuse), "t07_record_sha256": T07_RECORD_SHA256,
        "bootstrap": {"seed": T.BOOT_SEED, "generator": "numpy default_rng", "n_resamples": T.BOOT_RESAMPLES,
                      "n_images": table["n"], "interval": "percentile 2.5 and 97.5 (numpy linear)"},
        "rules": {"help": f">= +{T.HELP_POINTS} point and CI above 0", "harm": f"<= -{T.HELP_POINTS} and CI below 0",
                  "recovered_share": f"change / (base clean acc - base set acc), loss >= {T.HEADROOM_MIN_POINTS}"},
        **table, "readings": read}
    status = 0
    if args.mode == "dev":
        problems = reproduction(table, t07)
        errors = [p for p in problems if "not an error" not in p]
        out["reproduces_t07"] = {"equal": not errors, "differences": problems}
        status = 0 if not errors else 1
    text = json.dumps(out, indent=1) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print_tables(table, read, args.mode)
    if args.mode == "dev":
        print(f"\nreproduces T07's dev table: {out['reproduces_t07']['equal']} "
              f"({len(out['reproduces_t07']['differences'])} notes)")
        for p in out["reproduces_t07"]["differences"]:
            print(f"  {p}")
    print(f"-> {args.out} sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
