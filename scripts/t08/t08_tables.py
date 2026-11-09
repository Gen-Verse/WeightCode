"""PREREG T08: the reuse matrix table (``python -m weightcode.harness.reuse_matrix table``) as markdown.

Per condition and arm, the 5 x 6 matrix of V_select gain / clean / joint in points (status when not ok; the
diagonal program cell marked *), the diagonal gates, the per-source readings, the per-target means and the E3 and F6
cross-checks. Numbers are copied from the table JSON, nothing is recomputed.

    python scripts/t08/t08_tables.py $WEIGHTCODE_RUNS/t08/t08_matrix_<date>.json > tables.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LABELS = {"sc918e11dceef637a": "record", "s4253319a365b126a": "T02 rep 1", "scb92c3eb5d5226cc": "T02 rep 2",
          "scf16f7fb35855c03": "T01", "sf45990dee41ac233": "T01", "s760cb52ce82fb937": "record",
          "s7ae7d105d7655ed5": "T02 rep 1", "sfeb1cbc77d6a6042": "T02 rep 2", "s3db14093bc6c3538": "T01",
          "s037ead942a4cc8aa": "T01"}


def _f(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:+.3f}"
    return str(value)


def _name(table: dict[str, Any], sid: str) -> str:
    return f"{sid[:8]} ({LABELS.get(sid, '?')}, seed {table['sources'][sid]['seed']})"


def render(table: dict[str, Any]) -> str:
    out: list[str] = []
    for condition, c in table["conditions"].items():
        targets = c["targets"]
        seeds = [r["seed"] for r in c["target_readings"]]
        for arm in ("program", "copy"):
            out += ["", f"### {condition}, {arm} arm: gain / clean / joint (points; * own instance)", "",
                    "| source | " + " | ".join(f"{t} ({s})" for t, s in zip(targets, seeds)) + " |",
                    "|---|" + "---|" * len(targets)]
            for sid, row in c["rows"].items():
                if "program" not in row:
                    out.append(f"| {_name(table, sid)} | {row['status']} |")
                    continue
                own = table["sources"][sid]["instance_id"]
                cells = []
                for target in targets:
                    cell = row[arm].get(target)
                    if cell is None:
                        cells.append("-")
                        continue
                    text = f"{_f(cell['gain'])} / {_f(cell['clean'])} / {_f(cell['joint'])}"
                    if cell["status"] != "ok":
                        text += f" ({cell['status']})"
                    cells.append(("*" if target == own else "") + text)
                out.append(f"| {_name(table, sid)} | " + " | ".join(cells) + " |")
        out += ["", f"### {condition}: diagonal gates (child sha256 equal; deltas recorded)", ""]
        for sid, gate in c["diagonal_gates"].items():
            out.append(f"- {_name(table, sid)}: passed {_f(gate['passed'])}, same child {_f(gate['same_child'])}, "
                       f"same float64 deltas {_f(gate['same_deltas'])}")
        out += ["", f"### {condition}: per source (non-own targets: 5)", "",
                "| source | own gain / clean | program mean gain / clean / joint | reuse share | C, E3 form "
                "| C, per target | copy mean gain / clean / joint | program - copy: gain / joint |",
                "|---|---|---|---|---|---|---|---|"]
        for r in c["source_readings"]:
            if not r["read"]:
                out.append(f"| {_name(table, r['session_id'])} | not read: {r['reason']} | | | | | | |")
                continue
            base = [t for arm in ("program", "copy") for t in r["counted_as_base"][arm]]
            out.append(
                f"| {_name(table, r['session_id'])} | {_f(r['own_gain'])} / {_f(r['own_clean'])} | "
                f"{_f(r['program_mean_gain'])} / {_f(r['program_mean_clean'])} / {_f(r['program_mean_joint'])} | "
                f"{_f(r['reuse_share'])} | {_f(r['c_e3_form'])} (clean ok {r['program_clean_ok']} of 5) | "
                f"{_f(r['c_per_target'])} ({r['c_per_target_n']} of 5) | "
                f"{_f(r['copy_mean_gain'])} / {_f(r['copy_mean_clean'])} / {_f(r['copy_mean_joint'])} | "
                f"{_f(r['program_minus_copy_gain'])} / {_f(r['program_minus_copy_joint'])}"
                + (f" (as base: {base})" if base else "") + " |")
        out += ["", f"### {condition}: per target, mean over sources (n)", "",
                "| target | program, all sources | program, non-own sources | copy, non-own sources |",
                "|---|---|---|---|"]
        for r in c["target_readings"]:
            parts = [r["program_all_sources"], r["program_non_own"], r["copy_non_own"]]
            out.append(f"| {r['instance_id']} ({r['seed']}) | " + " | ".join(
                f"({p['n']}) {_f(p['gain'])} / {_f(p['clean'])} / {_f(p['joint'])}" for p in parts) + " |")
    out += ["", "### cross-checks", ""]
    for sid, checks in table.get("cross_check", {}).items():
        for name, check in checks.items():
            out.append(f"- {_name(table, sid)} {name}: {check['n']} targets, all fields equal {_f(check['all_equal'])}")
            for target, row in check["targets"].items():
                if not row["equal"]:
                    out.append(f"  - {target}: differs in {row['differ']}; ours {row['ours']}, reference "
                               f"{row['reference']}")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("table", type=Path)
    args = parser.parse_args()
    print(render(json.loads(args.table.read_text(encoding="utf-8"))), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
