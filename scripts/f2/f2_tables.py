"""PREREG F2 report tables (Markdown) from the summary JSON that ``f2_cli.py summary`` writes: every configuration
per N with its V_check scores and V_select gain, clean, joint and worst-class clean; the kept pick (joint-retain-v1)
and the run's own no-floor pick are marked; then the picks and N*.

Usage: ``python scripts/f2/f2_tables.py SUMMARY.json ...``.
"""

import json
import sys

HEADER = ("| N | lr | steps | V_check S_joint | S_new | S_old | V_select gain | clean | joint | worst-class clean "
          "| pick |")


def tables(path: str) -> list[str]:
    s = json.load(open(path))
    order = sorted(s["groups"], key=lambda g: (g == "ceiling", s["groups"][g]["n"]))
    out = [f"\n### {s['condition']} ({s['instance_id']}), teacher V_select joint {s['teacher_joint']}", HEADER,
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for group in order:
        t = s["groups"][group]
        kept, own = t["kept"]["patch_id"], t["own"]["patch_id"]
        n = f"{t['n']}" + (" (ceiling)" if group == "ceiling" else "")
        for r in t["configs"]:
            c, v = r["config"], r["vcheck"]
            mark = ("kept" if r["patch_id"] == kept else "") + (" own" if r["patch_id"] == own else "")
            out.append(f"| {n} | {c['lr']:g} | {c['steps']} | {v['score']} | {v['score_new']} | {v['score_old']} | "
                       f"{r['gain']:+.3f} | {r['clean']:+.3f} | {r['joint']:+.3f} | {r['worst_class_clean']:+.3f} | "
                       f"{mark.strip()} |")
    out.append("\nPicks (joint-retain-v1, read on V_select):")
    for group in order:
        k = s["groups"][group]["kept"]
        if k["is_base"]:
            cfg = "base (no configuration passes S_old >= -20)"
        else:
            cfg = f"lr {k['config']['lr']:g}, {k['config']['steps']} steps"
        out.append(f"- {group}: {cfg}; eligible {k['n_eligible']}/{k['n_checks']}; gain {k['gain']:+.3f}, clean "
                   f"{k['clean']:+.3f}, joint {k['joint']:+.3f}, worst-class clean {k['worst_class_clean']:+.3f}")
    ns = s["n_star"]
    out.append(f"\nN*: {json.dumps(ns) if ns else 'no teacher joint'}")
    return out


if __name__ == "__main__":
    for summary in sys.argv[1:]:
        print("\n".join(tables(summary)))
