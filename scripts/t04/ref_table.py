"""The T04 reference table of record from the kept readouts (PREREG T04 arms 3 to 6, with the edge-extension amendment).

Per condition: every kept readout (arm, method, run label, kept pick and its config, primary pick, eligible checks,
V_check S_new / S_old, V_select gain / clean / joint in points); the bar and the main control before the extension
(labels t04, t04g2, f1) and of record over all grids (with t04g3); the determinism check of the extension (every
configuration an extension grid repeats from its source must give the source's V_check S_new and S_old exactly); and
the arm-6 frontier readouts (descriptive). gain = 100 x readouts.delta_acc on the condition's V_select, clean = 100 x
readouts_clean.delta_acc on the clean V_select, joint = gain + clean. The bar is the best joint of arms 4 and 5 plus 1;
the main control is the arm-3 kept pick with the best joint.

Stdlib only, for the NSCC login node over the T04 store:

    python3 scripts/t04/ref_table.py --store $BASE/t04/store --pairs $BASE/t04/amend/edge_pairs.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weightcode.common.paths import relocate_run_path
CONDITIONS = ("gaussian_noise_s3", "contrast_s4")
ARM4 = ("bn_adapt", "tent")
ARM5_T04 = ("grad_last", "grad_full", "ridge_last", "prior_shift", "tie_inputs", "input_stats", "grad_sparse19_sal",
            "grad_sparse19_traj", "interp")
ARM5_GRIDS = ("grad_sparse19_sal", "grad_sparse19_traj", "interp")
GRID_LABELS = ("t04g2", "t04g3")
EXTENSION = "t04g3"


def arm_of(method: str, label: str) -> str:
    """PREREG T04's arm of a run: 4 (bn_adapt, tent), 5 (E10's other references and their grids), 3 (F1's methods,
    and grad_full and grad_last on the second grid and its extension), 6 (ft_large)."""
    if method == "ft_large":
        return "6"
    if method in ARM4 and label in ("t04", EXTENSION):
        return "4"
    if (label == "t04" and method in ARM5_T04) or (label in GRID_LABELS and method in ARM5_GRIDS):
        return "5"
    if label in ("f1", EXTENSION) or (label in GRID_LABELS and method in ("grad_full", "grad_last")):
        return "3"
    raise ValueError(f"no arm for {method}/{label}")


def points(record: dict[str, Any]) -> tuple[float, float, float]:
    gain, clean = 100 * record["readouts"]["delta_acc"], 100 * record["readouts_clean"]["delta_acc"]
    return gain, clean, gain + clean


def load(store: Path, condition: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((store / "runs" / "devreadouts" / "t04" / condition).glob("kept_*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            run_path = relocate_run_path(r["run"], store)
            run = json.loads(run_path.read_text())
            config = {c["patch_id"]: c["config"] for c in run["candidates"]}
            sel = r["selection"]
            label = run_path.parts[-3]
            rows.append({"arm": arm_of(r["method"], label), "method": r["method"], "label": label,
                         "kept": sel["patch_id"], "kept_config": config.get(sel["patch_id"]),
                         "primary": sel.get("primary_patch_id"), "primary_config": config.get(sel.get("primary_patch_id")),
                         "eligible": f"{sel['n_eligible']}/{sel['n_checks']}",
                         "s_new": r["vcheck"].get("score_new"), "s_old": r["vcheck"].get("score_old"),
                         "points": points(r), "file": path.name, "run": run})
    return rows


def best(rows: list[dict[str, Any]], arms: Sequence[str]) -> dict[str, Any]:
    """The best joint among ``arms``; on a tie the run before the extension (a repeated configuration is the same
    network) is named."""
    return max((x for x in rows if x["arm"] in arms), key=lambda x: (x["points"][2], x["label"] != EXTENSION))


def cfg(config: Any) -> str:
    return "base" if config is None else json.dumps(config, sort_keys=True)


def line(x: dict[str, Any]) -> str:
    g, c, j = x["points"]
    return f"{x['method']}/{x['label']} {x['kept'] or 'base'} {cfg(x['kept_config'])} {g:+.3f} / {c:+.3f} / {j:+.3f}"


def determinism(rows: list[dict[str, Any]], decisions: list[dict[str, Any]], condition: str) -> tuple[int, list[str]]:
    """The repeated configurations of each extension grid against its source run's V_check log."""
    by = {(x["method"], x["label"]): x["run"] for x in rows}
    n, bad = 0, []

    def log(run: dict[str, Any]) -> dict[str, tuple[int, int]]:
        key = {c["patch_id"]: json.dumps(c["config"], sort_keys=True) for c in run["candidates"]}
        return {key[e["patch_id"]]: (int(e["score_new"]), int(e["score_old"])) for e in run["oracle_log"]}

    for d in decisions:
        if not (d["extended"] and d["condition"] == condition):
            continue
        source, ext = by.get((d["method"], d["label"])), by.get((d["method"], EXTENSION))
        if source is None or ext is None:
            bad.append(f"{d['method']}: missing source or {EXTENSION} run")
            continue
        if [c["config"] for c in ext["candidates"]] != d["configs"]:
            bad.append(f"{d['method']}: {EXTENSION} configurations differ from the pair list")
        s_log, e_log = log(source), log(ext)
        for i in d["repeats_of_source"]:
            key = json.dumps(d["configs"][i], sort_keys=True)
            n += 1
            if s_log.get(key) != e_log.get(key):
                bad.append(f"{d['method']} {key}: source {s_log.get(key)}, {EXTENSION} {e_log.get(key)}")
    return n, bad


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", required=True, type=Path, help="the T04 store root")
    parser.add_argument("--pairs", required=True, type=Path, help="edge_extension.py's output (edge_pairs.json)")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = parser.parse_args(argv)
    decisions = json.loads(args.pairs.read_text())["decisions"]
    for condition in args.conditions:
        rows = load(args.store, condition)
        print(f"== {condition}: {len(rows)} kept readouts")
        print("arm | method / label | kept pick (config) | primary | eligible | V_check S_new / S_old | gain | clean | "
              "joint")
        for x in sorted(rows, key=lambda x: (x["arm"], -x["points"][2], x["method"], x["label"])):
            g, c, j = x["points"]
            print(f"{x['arm']} | {x['method']} / {x['label']} | {x['kept'] or 'base'} {cfg(x['kept_config'])} | "
                  f"{x['primary'] or 'base'} {cfg(x['primary_config'])} | {x['eligible']} | {x['s_new']} / "
                  f"{x['s_old']} | {g:+.3f} | {c:+.3f} | {j:+.3f}")
        pre = [x for x in rows if x["label"] != EXTENSION]
        bar0, bar1 = best(pre, ("4", "5")), best(rows, ("4", "5"))
        ctl0, ctl1 = best(pre, ("3",)), best(rows, ("3",))
        print(f"bar: before the extension {bar0['points'][2] + 1:+.3f} ({line(bar0)}); of record "
              f"{bar1['points'][2] + 1:+.3f} ({line(bar1)})")
        print(f"main control: before the extension {line(ctl0)}; of record {line(ctl1)}")
        n, bad = determinism(rows, decisions, condition)
        print(f"determinism: {n} repeated configurations, {len(bad)} mismatches" + "".join(f"\n  {b}" for b in bad))
        for path in sorted((args.store / "runs" / "devreadouts" / "t04" / condition).glob("frontier_*.jsonl")):
            print(f"arm-6 frontier ({path.name}; descriptive, selects nothing):")
            for r in map(json.loads, path.read_text().splitlines()):
                g, c, j = points(r)
                print(f"  {r['method']} {r['patch_id']} {cfg(r['config'])} | V_check {r['vcheck']['score_new']} / "
                      f"{r['vcheck']['score_old']} | {g:+.3f} | {c:+.3f} | {j:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
