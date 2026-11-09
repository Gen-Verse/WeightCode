"""The (method, condition) pairs that PREREG T04's edge-extension amendment extends, computed from the kept readouts.

PREREG T04, "Amendment, edge extension of the reference grids" (WEIGHTCODE-main, main 9f0baa9, 2026-11-07 about 16:00
Beijing). A grid-based reference of arms 3, 4 or 5 is extended on a condition when its primary pick (``choose_final``
over its whole V_check log, E1's original rule; ``selection.primary_patch_id``) or its kept pick
(``selection.patch_id``) lies on the conservative edge of its grid: the smallest learning rate, the smallest interp
alpha, or the largest bn_adapt prior strength. The extension grid is the run's own 8 configurations with every
learning rate divided by 10, every alpha divided by 8, or every prior strength multiplied by 16, in the same order,
everything else unchanged. Not extended: prior_shift, ridge_last, tie_inputs, input_stats (no such edge), arm 6
(ft_large), and runs whose primary and kept picks are both off the edge or both the base.

Stdlib only, so it runs on the NSCC login node with the system python3 over the T04 store:

    python3 scripts/t04/edge_extension.py --devreadouts $STORE/runs/devreadouts/t04 --store $STORE --out pairs.json

Each kept record's ``run`` is a path under the store's mount point inside the job; it is
read under ``--store``. The output lists every kept record with its decision, and the extended pairs with their
trigger and their 8 new configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weightcode.common.paths import relocate_run_path
NO_EDGE = ("prior_shift", "ridge_last", "tie_inputs", "input_stats")
"""References without a conservative edge (the amendment's list)."""
DIAGNOSTIC = ("ft_large",)
"""Arm 6, never extended."""
EDGES: dict[str, tuple[str, str, float]] = {
    # config key: (edge, scale rule, factor)
    "lr": ("smallest", "divided by 10", 0.1),
    "alpha": ("smallest", "divided by 8", 0.125),
    "N": ("largest", "multiplied by 16", 16.0),
}
EXTENSION_LABEL = "t04g3"


class EdgeError(ValueError):
    """A kept record or run that the rule cannot read (unknown grid, missing pick)."""


def grid_key(configs: Sequence[Mapping[str, Any]], method: str) -> str:
    """The config key that carries the grid's conservative edge: lr, alpha (interp) or N (bn_adapt)."""
    keys = [k for k in EDGES if all(k in c for c in configs)]
    if len(keys) != 1:
        raise EdgeError(f"{method}: no single edge key among {sorted(EDGES)} in {list(configs)}")
    return keys[0]


def scaled(value: float, factor: float, key: str) -> float | int:
    """``value`` times ``factor``, written with 12 significant digits (1e-05 / 10 is 1e-06, not 1.0000000000000002e-06);
    bn_adapt's prior strengths stay integers."""
    if key == "N":
        return int(round(value * factor))
    return float(f"{value * factor:.12g}")


def decide(record: Mapping[str, Any], run: Mapping[str, Any], label: str) -> dict[str, Any]:
    """The amendment's decision for one kept record (its run's directory label ``label``, e.g. t04g2): extended or
    not, the trigger, and the extension grid."""
    method = record["method"]
    sel = record["selection"]
    out: dict[str, Any] = {"condition": record["condition"], "instance": record["instance_id"], "method": method,
                           "label": label, "primary": sel.get("primary_patch_id"), "kept": sel.get("patch_id")}
    if method in NO_EDGE or method in DIAGNOSTIC:
        return out | {"extended": False, "reason": "no conservative edge (not extended by the amendment)"}
    configs = {c["patch_id"]: c["config"] for c in run["candidates"]}
    order = [c["config"] for c in run["candidates"]]
    key = grid_key(order, method)
    edge_kind, rule, factor = EDGES[key]
    values = [c[key] for c in order]
    edge = min(values) if edge_kind == "smallest" else max(values)
    triggers = []
    for which in ("primary", "kept"):
        pid = out[which]
        if pid is None:
            continue
        if pid not in configs:
            raise EdgeError(f"{method}/{label}: {which} pick {pid} is not a candidate of the run")
        if configs[pid][key] == edge:
            triggers.append(f"{which} {pid} {json.dumps(configs[pid], sort_keys=True)}")
    out |= {"edge": f"{edge_kind} {key} = {edge:g}", "edge_key": key}
    if not triggers:
        both_base = out["primary"] is None and out["kept"] is None
        return out | {"extended": False,
                      "reason": "primary and kept are the base" if both_base else "primary and kept off the edge"}
    new = [dict(c) | {key: scaled(c[key], factor, key)} for c in order]
    old = [json.dumps(c, sort_keys=True) for c in order]
    repeats = [i for i, c in enumerate(new) if json.dumps(c, sort_keys=True) in old]
    return out | {"extended": True, "trigger": triggers, "rule": f"{key} {rule}", "extension_label": EXTENSION_LABEL,
                  "configs": new, "repeats_of_source": repeats}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--devreadouts", required=True, type=Path, help="runs/devreadouts/t04 of the T04 store")
    parser.add_argument("--store", required=True, type=Path, help="the T04 store root")
    parser.add_argument("--conditions", nargs="+", default=["gaussian_noise_s3", "contrast_s4"])
    parser.add_argument("--out", type=Path, help="write the decisions as JSON here")
    args = parser.parse_args(argv)
    decisions, files = [], []
    for condition in args.conditions:
        kept = sorted((args.devreadouts / condition).glob("kept_*.jsonl"))
        for path in kept:
            files.append({"file": f"{condition}/{path.name}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if record["condition"] != condition:
                    raise EdgeError(f"{path}: condition {record['condition']} in the {condition} directory")
                run_path = relocate_run_path(record["run"], args.store)
                if run_path.name != "run.json" or run_path.parts[-4] != record["method"]:
                    raise EdgeError(f"{path}: run {run_path} is not <method>/<label>/<seed>/run.json")
                decisions.append(decide(record, json.loads(run_path.read_text()), run_path.parts[-3]))
    extended = [d for d in decisions if d["extended"]]
    result = {"rule": "PREREG T04 amendment, edge extension of the reference grids (main 9f0baa9)",
              "n_kept_files": len(files), "n_records": len(decisions), "n_extended": len(extended),
              "kept_files": files, "decisions": decisions}
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"{len(files)} kept files, {len(decisions)} records, {len(extended)} extended pairs")
    for d in decisions:
        head = f"{d['condition']} {d['method']}/{d['label']}"
        if not d["extended"]:
            print(f"  not extended  {head}: {d['reason']}" + (f" ({d['edge']})" if "edge" in d else ""))
    for d in extended:
        print(f"  EXTENDED      {d['condition']} {d['method']}/{d['label']} -> {d['method']}/{EXTENSION_LABEL}: "
              f"{d['edge']}; trigger {'; '.join(d['trigger'])}; {d['rule']}; repeats of the source grid at "
              f"{d['repeats_of_source']}")
        print(f"                new configs {json.dumps(d['configs'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
