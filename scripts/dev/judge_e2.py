"""E2 numbers of one fewshot LEARN session against PREREG section 0 E2 (dev, seed 100). Reads the outputs of
scripts/dev/post_learn.sh ($WEIGHTCODE_RUNS/devreadouts/learn/<sid>/info.json from session_info and
transfer/transfer_*.json next to it), the session's own
logs/checks.jsonl, and the references' devreadout records.

Usage: judge_e2.py SESSION_ID [--best-ref-all POINTS] [--refs FILE ...]

Points are 100 x the change against the base model on V_select (10-way argmax): all rows (the gate), the new-class
rows and the old-class rows, with the worst old class. Supported (PREREG E2): all-class change >= the best of the four
references + 2 points, and old-class change >= -1 point. --best-ref-all is the best reference's all-class change in
points, read from the references' own record before this session's readout; without it no verdict is printed.

Supplementary, not the gate (PREREG E2, recorded before any E2 LEARN check): the equal-retention pick. Among the
session's checks with S_old >= -20 on V_check plus the base no-op, the highest S_joint, ties to fewer changed
parameters, then the earlier query (choose_final on a filtered pool, as weightcode.evalkit.devreadout.retention_choice).
Its V_select row comes from the session_info trajectory. --refs lists reference records (jsonl, one run per line,
each with instance_id, method and readouts_fewshot) whose rows for this instance are printed next to it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

LEARN_ROOT = Path(os.environ["WEIGHTCODE_RUNS"]) / "devreadouts" / "learn"
S_OLD_FLOOR = -20


def pts(value: float | None) -> str:
    return "  n/a " if value is None else f"{100.0 * value:+6.2f}"


def line(tag: str, full: dict) -> str:
    return (f"{tag} all {pts(full['delta_all'])} new {pts(full['delta_new'])} old {pts(full['delta_old'])}  worst old "
            f"class {full['worst_old_class']} {pts(full['worst_old_class_delta'])}  acc all "
            f"{100.0 * full['acc_all_child']:.2f}% new {100.0 * full['acc_new_child']:.2f}%  n all/new/old "
            f"{full['n_all']}/{full['n_new']}/{full['n_old']}")


def retention_pick(checks: list[dict], floor: int = S_OLD_FLOOR) -> tuple[dict | None, int]:
    """The equal-retention pick over a session's checks, and the number of eligible checks."""
    eligible = []
    for check in checks:
        if int(check["score"]) != int(check["score_new"]) + int(check["score_old"]):
            raise SystemExit(f"check {check['query_index']}: S_joint != S_new + S_old")
        if int(check["score_old"]) >= floor:
            eligible.append(check)
    base = {"score": 0, "n_changed": 0, "query_index": -1, "patch_id": None}
    chosen = min([base, *eligible], key=lambda c: (-int(c["score"]), int(c["n_changed"]), int(c["query_index"])))
    return (None if chosen["patch_id"] is None else chosen), len(eligible)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    parser.add_argument("--best-ref-all", type=float, default=None)
    parser.add_argument("--refs", nargs="*", default=[])
    args = parser.parse_args()
    out = LEARN_ROOT / args.session_id
    info = json.loads((out / "info.json").read_text())
    final = info["final"]
    print(f"== {args.session_id}  instance {info['instance_id']}  ({info['dataset']})")
    print(f"   final: noop {final.get('is_base_noop')}  query {final.get('query_index')}  program {final.get('patch_id')}")
    if final.get("vcheck"):
        vc = final["vcheck"]
        print(f"   V_check: score_new {vc.get('score_new')}  score_old {vc.get('score_old')}  S_joint {vc.get('score')}")
    if final.get("readouts_fewshot"):
        full = final["readouts_fewshot"]
        print(line("   E2 V_select:", full))
        if args.best_ref_all is not None:
            checks = {"all>=best_ref+2": 100.0 * full["delta_all"] >= args.best_ref_all + 2.0,
                      "old>=-1.0": 100.0 * full["delta_old"] >= -1.0}
            print(f"   E2 {'SUPPORTED' if all(checks.values()) else 'not supported'} (best reference all "
                  f"{args.best_ref_all:+.2f}): {checks}")
    session_checks = Path(os.environ["WEIGHTCODE_RUNS"]) / "sessions" / args.session_id / "logs" / "checks.jsonl"
    checks = [json.loads(raw) for raw in session_checks.read_text().splitlines() if raw.strip()]
    chosen, n_eligible = retention_pick(checks)
    if chosen is None:
        print(f"   supplement (S_old >= {S_OLD_FLOOR}): base ({n_eligible}/{len(checks)} eligible, none with S_joint > 0)")
    else:
        rows = [row for row in info["trajectory"] if int(row["query_index"]) == int(chosen["query_index"])]
        if len(rows) != 1:
            raise SystemExit(f"query {chosen['query_index']}: {len(rows)} trajectory rows")
        print(f"   supplement (S_old >= {S_OLD_FLOOR}): q{chosen['query_index']} {chosen.get('program_file')} "
              f"({n_eligible}/{len(checks)} eligible)  V_check S_new {chosen['score_new']} S_old {chosen['score_old']}")
        print(line("   supplement V_select:", rows[0]["readouts_fewshot"]))
    for ref_file in args.refs:
        for raw in Path(ref_file).read_text().splitlines():
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("instance_id") != info["instance_id"]:
                continue
            sel = record.get("selection") or {}
            tag = f"   ref {Path(ref_file).stem[-24:]} {record.get('method')}"
            if sel:
                tag += f" pick {sel.get('patch_id')} ({sel.get('n_eligible')}/{sel.get('n_checks')})"
            print(line(tag + ":", record["readouts_fewshot"]))
    for path in sorted((out / "transfer").glob("transfer_*.json")):
        record = json.loads(path.read_text())
        target = record["target"]["instance_id"]
        if record.get("status") != "ok":
            print(f"   transfer {target}: {record.get('status')} {record.get('messages')}")
            continue
        print(line(f"   transfer {target} within_limits {record.get('within_limits')}:", record["readouts_fewshot"]))


if __name__ == "__main__":
    main()
