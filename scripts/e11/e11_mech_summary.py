"""Tables and the pre-registered reading of the E11 mechanism records ($WEIGHTCODE_RUNS/devreadouts/e11_mech/*.jsonl).

Reads the records ``scripts/e11/e11_mech.py`` wrote, prints markdown tables (points, V_select) and applies the reading
fixed in PREREG (d83f0fc, "Mechanism of the E11 gaussian final"): "most of the gain" is >= 75% of the final's joint at
clean >= -1.0; T0 first, then T1. With ``--write`` the reading is stored as ``summary.jsonl`` (never overwritten).

    source scripts/env.sh
    .venv/bin/python -B scripts/e11/e11_mech_summary.py [--write]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

READING_FRACTION = 0.75
READING_CLEAN_FLOOR = -1.0
GATES = (2, 4, 6)


def folder() -> Path:
    return Path(os.environ["WEIGHTCODE_RUNS"]) / "devreadouts" / "e11_mech"


def load(name: str) -> list[dict[str, Any]]:
    path = folder() / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def f(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_RANKS: dict[str, int] = {}


def ranks() -> dict[str, int]:
    """Tensor ranks of the instance's student (the limit applies to rank >= 2 only)."""
    if not _RANKS:
        import numpy as np

        path = Path(os.environ["WEIGHTCODE_PRIVATE"]) / "instances" / "i00802425" / "model.npz"
        with np.load(path, allow_pickle=False) as archive:
            _RANKS.update({key: archive[key].ndim for key in archive.files})
    return _RANKS


def max_rel_frob(edit: dict[str, Any] | None) -> str:
    """The largest rel-Frob over the changed tensors of rank >= 2 (the ones the 0.5 limit applies to)."""
    values = [v for k, v in ((edit or {}).get("per_tensor_rel_frob") or {}).items() if ranks()[k] >= 2]
    return f"{max(values):.3f}" if values else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="store the reading as summary.jsonl")
    args = parser.parse_args()
    final_joint = None
    print("## Reproduction")
    for r in load("repro"):
        if r["name"] == "final_reapplied":
            final_joint = r["joint"]
            print(f"final re-applied: child {r['child_sha256'][:12]}, gain {f(r['gain'])}, clean {f(r['clean'])}, joint "
                  f"{f(r['joint'])}, worst-class clean {f(r['worst_class_clean'])}, reproduced {r['reproduced']}")
        else:
            print(f"program rerun: deltas {r['run']['deltas_sha256'][:12]}, child {r['edit']['child_sha256'][:12]}, "
                  f"within limits {r['edit']['within_limits']}, reproduced {r['reproduced']}")

    final_edit = next((r["edit"] for r in load("repro") if r["name"] == "final_program_rerun"), None)
    a1 = {r["name"]: r for r in load("a1")}
    if a1:
        print("\n## A1 (subset = the gates ablated)")
        print("| variant | ablated | gain | clean | joint | worst-class clean | entries changed | max rel-Frob | within |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in a1.values():
            e = r.get("edit") or final_edit
            print(f"| {r['config']['variant'] or 'none (final)'} | {r['config']['subset']} | {f(r['gain'])} | "
                  f"{f(r['clean'])} | {f(r['joint'])} | {f(r['worst_class_clean'])} | "
                  f"{e['n_changed'] if e else 'n/a'} | {max_rel_frob(e)} | {e['within_limits'] if e else 'n/a'} |")
        on = a1["all_on"]
        print("\n| marginal (all on minus all but that gate) | gain | clean | joint |\n|---|---|---|---|")
        for variant in ("off", "revert"):
            for b in GATES:
                r = a1[f"{variant}_g{b}"]
                print(f"| g{b}, {variant} | {f(on['gain'] - r['gain'])} | {f(on['clean'] - r['clean'])} | "
                      f"{f(on['joint'] - r['joint'])} |")

    a2 = load("a2")
    if a2:
        print("\n## A2 (V_select, 4000 condition rows and their clean companions, final model)")
        print("| gate | tokens | rows | frac above t | frac fully open | mean opening | mean output norm |")
        print("|---|---|---|---|---|---|---|")
        for r in a2:
            b, gated = r["config"]["block"], r["config"]["gated_tokens"]
            for tok in (gated, "cls" if gated == "patch" else "patch"):
                for split in ("cond", "clean"):
                    s = r["selectivity"][f"{split}_{tok}"]
                    print(f"| g{b} | {tok}{' (gated)' if tok == gated else ''} | {split} | "
                          f"{s['frac_above_threshold']:.4f} | {s['frac_fully_open']:.4f} | {s['mean_opening']:.4f} | "
                          f"{s['mean_gate_output_norm']:.4f} |")
        print("\n| gate block output | tokens | base: norm of mean offset | final: norm of mean offset | base: mean paired "
              "norm | final: mean paired norm | cos(base offset, program offset) | program offset norm |")
        print("|---|---|---|---|---|---|---|---|")
        for r in a2:
            b, gated = r["config"]["block"], r["config"]["gated_tokens"]
            o = r["offsets"]
            for tok in ("patch", "cls"):
                cos = o["base"].get(f"cos_program_offset_{tok}")
                print(f"| {b} | {tok} | {o['base'][f'{tok}_mean_offset_norm']:.3f} | "
                      f"{o['final'][f'{tok}_mean_offset_norm']:.3f} | {o['base'][f'{tok}_mean_paired_norm']:.3f} | "
                      f"{o['final'][f'{tok}_mean_paired_norm']:.3f} | {'' if cos is None else f'{cos:.3f}'} | "
                      f"{r['gate']['offset_norm']:.3f} |")
        print("\n| gate block output | edit moves condition rows (cls, patch) | edit moves clean rows (cls, patch) |")
        print("|---|---|---|")
        for r in a2:
            m = r["edit_moves_block_output"]
            print(f"| {r['config']['block']} | {m['cond']['cls_mean_norm']:.4f}, {m['cond']['patch_mean_norm']:.4f} | "
                  f"{m['clean']['cls_mean_norm']:.4f}, {m['clean']['patch_mean_norm']:.4f} |")

    picks: dict[str, dict[str, Any]] = {}
    for name in ("t0", "t1"):
        rec = load(name)
        if not rec:
            continue
        r = rec[0]
        picks[name.upper()] = r
        print(f"\n## {name.upper()} checks (V_check, joint-retain-v1, S_old >= -20)")
        print("| check | config | S_joint | S_new | S_old | eligible | entries changed | max rel-Frob | within limits |")
        print("|---|---|---|---|---|---|---|---|---|")
        for c in r["checks"]:
            cfg = {k: v for k, v in c["config"].items() if k != "name"}
            e = c.get("edit")
            if c.get("invalid"):
                print(f"| {c['patch_id']} | {c['config']['name']} {cfg} | invalid ({c['invalid_stage']}): "
                      f"{str(c['invalid'])[:120]} | | | | | | |")
                continue
            print(f"| {c['patch_id']} | {c['config']['name']} | {c['score']} | {c['score_new']} | {c['score_old']} | "
                  f"{c['eligible']} | {e['n_changed']} | {max_rel_frob(e)} | {e['within_limits']} |")
        sel = r["selection"]
        print(f"pick {sel.get('patch_id') or 'base'} ({sel.get('n_eligible')} of {sel.get('n_checks')} eligible): "
              f"V_select gain {f(r['gain'])}, clean {f(r['clean'])}, joint {f(r['joint'])}, worst-class clean "
              f"{f(r['worst_class_clean'])}")

    t2 = load("t2")
    if t2:
        print("\n## T2 (random gate units, not checked)")
        print("| draw | units (block: j1, j2) | gain | clean | joint | worst-class clean | within limits |")
        print("|---|---|---|---|---|---|---|")
        for r in t2:
            if r["name"] == "t2_mean":
                print(f"| mean of 5 | | {f(r['gain'])} | {f(r['clean'])} | {f(r['joint'])} | "
                      f"{f(r['worst_class_clean'])} | |")
                continue
            units = re.findall(r"T2 draw \d+: units (\d+),(\d+) at block (\d+)", r["run"]["stdout"])
            blocks = re.findall(r"gate (patch|cls)\s+block (\d+)", r["run"]["stdout"])
            text = "; ".join(f"{b}: {j1},{j2}" for j1, j2, b in units)
            print(f"| {r['config']['draw']} | {text} (gates at {[int(b) for _, b in blocks]}) | {f(r['gain'])} | "
                  f"{f(r['clean'])} | {f(r['joint'])} | {f(r['worst_class_clean'])} | {r['edit']['within_limits']} |")

    t3 = load("t3")
    if t3:
        r = t3[0]
        print(f"\n## T3 (gates always open, not checked)\ngain {f(r['gain'])}, clean {f(r['clean'])}, joint "
              f"{f(r['joint'])}, worst-class clean {f(r['worst_class_clean'])}, within limits {r['edit']['within_limits']}")

    if final_joint is not None and "T0" in picks and "T1" in picks:
        bar = READING_FRACTION * final_joint
        reach = {k: bool(v["joint"] >= bar - 1e-9 and v["clean"] >= READING_CLEAN_FLOOR - 1e-9) for k, v in picks.items()}
        if reach["T0"]:
            branch = "T0 reaches it: the gain is generic feature alignment; the claim for the teacher narrows to finding it"
        elif reach["T1"]:
            branch = "T1 reaches it but T0 does not: the gate carries the gain and the corruption model is not needed"
        else:
            branch = "neither reaches it: both the inferred corruption model and the gate matter"
        reading = {"schema": "weightcode.e11.mech.reading.v1", "final_joint": final_joint, "bar": bar,
                   "clean_floor": READING_CLEAN_FLOOR, "reaches": reach, "branch": branch,
                   "picks": {k: {key: v[key] for key in ("gain", "clean", "joint", "worst_class_clean")} | {
                       "patch_id": v["selection"].get("patch_id")} for k, v in picks.items()},
                   "records": {p.name: sha(p) for p in sorted(folder().glob("*.jsonl")) if p.name != "summary.jsonl"}}
        print("\n## Reading\n" + json.dumps(reading, indent=1))
        if args.write:
            path = folder() / "summary.jsonl"
            if path.exists():
                raise SystemExit(f"{path} exists; not overwritten")
            path.write_text(json.dumps(reading, sort_keys=True) + "\n")
            (folder() / "summary.jsonl.sha256").write_text(f"{sha(path)}  {path.name}\n")
            print(f"wrote {path} sha256 {sha(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
