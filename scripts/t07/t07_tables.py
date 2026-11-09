"""Markdown tables of the T07 and M05 phase 1 records, as written into DEV_RESULTS.

Usage: python scripts/t07/t07_tables.py T07_TABLE_JSON M05_SUMMARY_JSON
Reads only the two record files (no network, no image) and prints, per architecture, the 26-set table of change
[paired 95% interval] and label for the four networks, label counts per group, recovered shares, the LLM minus FT
table, the T07-G gate table with its Spearman, the clean per-class gate rates and the M05 tables.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

NETS = ["llm_gaussian_noise", "llm_contrast", "ft_gaussian_noise", "ft_contrast"]
MARK = {"help": "+", "harm": "-", "neutral": "0"}
GROUPS = ["clean", "own_gaussian_noise", "own_contrast", "e10_other", "unseen"]


def t07_tables(t07: dict[str, Any]) -> None:
    """Print the per-architecture tables, counts, recovered shares, LLM minus FT and T07-G."""
    rows = {(r["arch"], r["network"], r["set"]): r for r in t07["rows"]}
    print("checks all equal:", t07["all_checks_equal"], len(t07["checks"]))
    for arch in ("vit", "r18"):
        print(f"\n{arch}: change in points vs base [label: + help, - harm, 0 neutral]; base acc; n = 4000")
        print("| set | base acc | LLM gaussian | LLM contrast | FT gaussian | FT contrast |")
        print("|---|---|---|---|---|---|")
        for s in t07["sets"]:
            base = rows[(arch, NETS[0], s)]["base_acc"]
            cells = [f"{rows[(arch, n, s)]['delta_points']:+.2f} [{rows[(arch, n, s)]['ci95'][0]:+.2f}, "
                     f"{rows[(arch, n, s)]['ci95'][1]:+.2f}] {MARK[rows[(arch, n, s)]['label']]}" for n in NETS]
            print(f"| {s} | {100 * base:.2f} | " + " | ".join(cells) + " |")
        print("\nlabel counts per network and group (help/neutral/harm):")
        for n in NETS:
            c = Counter((r["group"], r["label"]) for r in t07["rows"] if r["arch"] == arch and r["network"] == n)
            print(n, {g: f"{c[(g, 'help')]}/{c[(g, 'neutral')]}/{c[(g, 'harm')]}" for g in GROUPS})
        print("\nrecovered share (loss >= 2 points) on own s1..s5:")
        for n in NETS[:2]:
            cond = n.split("_", 1)[1]
            vals = []
            for sev in range(1, 6):
                r = rows[(arch, n, f"{cond}_s{sev}")]
                share = "NA" if r["recovered_share"] is None else round(r["recovered_share"], 3)
                vals.append(f"s{sev} {r['base_loss_from_clean_points']:.1f} -> {share}")
            print(n, vals)
    lm = {(r["arch"], r["condition"], r["set"]): r for r in t07["llm_minus_ft"]}
    print("\nLLM minus FT (points [paired 95% interval]):")
    print("| set | ViT gaussian | ViT contrast | R18 gaussian | R18 contrast |")
    print("|---|---|---|---|---|")
    for s in t07["sets"]:
        cells = [f"{lm[(a, c, s)]['llm_minus_ft_points']:+.2f} [{lm[(a, c, s)]['ci95'][0]:+.2f}, "
                 f"{lm[(a, c, s)]['ci95'][1]:+.2f}]" for a in ("vit", "r18") for c in ("gaussian_noise", "contrast")]
        print(f"| {s} | " + " | ".join(cells) + " |")
    g = {(x["final"], x["set"]): x for x in t07["gates"]}
    print("\nT07-G main gate rate per set and the final's change:")
    print("| set | gaussian final: patch gate rate | change | contrast final: gate unit rate | change |")
    print("|---|---|---|---|---|")
    for s in t07["sets"]:
        a, b = g[("gaussian_noise", s)], g[("contrast", s)]
        print(f"| {s} | {a['main_gate_rate']:.4f} | {a['delta_points']:+.2f} | {b['main_gate_rate']:.4f} | "
              f"{b['delta_points']:+.2f} |")
    for final in ("gaussian_noise", "contrast"):
        print(final, "spearman:", t07["gate_associations"][final])
    print("\nT07-G clean per-class main-gate rate:")
    for x in t07["gates"]:
        if x["set"] != "clean":
            continue
        st = x["stats"]
        if x["final"] == "gaussian_noise":
            for k in st:
                rates = [round(v, 4) for v in st[k]["per_class_gated_rate"]]
                extra = f" cls rate {round(st[k]['cls']['frac_above_threshold'], 4)}" if k.startswith("cls") else ""
                print(x["final"], k + extra, rates)
        else:
            print(x["final"], [round(v, 4) for v in st["gate"]["per_class_patch_open_rate"]])


def m05_tables(m05: dict[str, Any]) -> None:
    """Print the M05 phase 1 table per program with its descriptive Spearman."""
    print("\nM05:")
    for p, d in m05["programs"].items():
        print(f"\n{p}: same relation {d['same_relation_count']}/{d['n_targets']}")
        print("| target | gain | clean | cond rate | clean rate | correction norm (cond) | base offset | edited offset "
              "| residual share | clean edit move | same relation |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in d["rows"]:
            print(f"| {r['target']}{' *' if r['own'] else ''} | {r['gain_points']:+.3f} | {r['clean_points']:+.3f} | "
                  f"{r['cond_rate']:.4f} | {r['clean_rate']:.4f} | {r['correction_norm_cond_patch']:.3f} | "
                  f"{r['base_offset_patch']:.3f} | {r['edited_offset_patch']:.3f} | {r['residual_share']:.3f} | "
                  f"{r['edit_move_clean_patch']:.4f} | {r['same_relation']} |")
        print("spearman non-own:", d["spearman_non_own"])
    print("references reproduced:", m05["references_reproduced"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("t07_table", type=Path)
    ap.add_argument("m05_summary", type=Path)
    args = ap.parse_args()
    t07_tables(json.loads(args.t07_table.read_text(encoding="utf-8")))
    m05_tables(json.loads(args.m05_summary.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
