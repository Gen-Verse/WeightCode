"""R21-e11-rep (PREREG "Replication preparation"): reference and F1 tables from devreadout records (read-only).

usage: python -I scripts/store/e11rep_tables.py refs  FLOOR20_E11 FLOOR20_E11G2   -> kept picks of both grids, bar
       python -I scripts/store/e11rep_tables.py f1    F1_FLOOR20 F1_OWN F1_FRONTIER E11G2_FLOOR20 E11G2_OWN \
           E11G2_FRONTIER_F1                                                             -> F1 kept, own and frontier
       python -I scripts/store/e11rep_tables.py xcheck RECORD_JSONL NEW_JSONL      -> per-row equality of two
                                                                                            readouts of the same runs
       python -I scripts/store/e11rep_tables.py checks RECORD_ROOT NEW_ROOT INSTANCE -> per-check V_check equality
                                                                                            of two executions of runs
Gain, clean and joint are V_select points: 100 x delta_acc on the condition split, on its clean companion, and their
sum. The configuration of a pick comes from its run.json candidate (patch_id). The bar: gain >= 10, clean >= -1.0 and
joint >= the best kept reference of both grids + 1 (the base when every kept pick is the base).
"""
import json
import sys
from pathlib import Path


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def config_of(r):
    """(patch_id, config) of a record: a frontier row's own; a kept pick's selection; an own final's run final."""
    if "config" in r and "patch_id" in r:
        return r["patch_id"], r["config"]
    if r.get("is_base_noop"):
        return "base", None
    run = json.loads(Path(r["run"]).read_text())
    if "selection" in r:
        pid = r["selection"].get("patch_id")
    else:
        pid = run["final"]["patch_id"]
    if pid is None:
        return "base", None
    cand = {c["patch_id"]: c for c in run["candidates"]}
    return pid, cand[pid]["config"] if pid in cand else None


def pts(r):
    g = 100 * r["readouts"]["delta_acc"]
    c = 100 * r["readouts_clean"]["delta_acc"]
    return g, c, g + c


def fmt_cfg(cfg):
    if cfg is None:
        return "-"
    return ", ".join(f"{k} {v}" for k, v in cfg.items())


def line(tag, r):
    g, c, j = pts(r)
    pid, cfg = config_of(r)
    v = r["vcheck"]
    lim = r.get("within_program_limits")
    return (f"{tag:24s} {r['method']:20s} {g:+8.3f} {c:+8.3f} {j:+8.3f}   V_check S {v['score']:5d} new {v.get('score_new', '-'):>5} "
            f"old {v.get('score_old', '-'):>4}   {pid} [{fmt_cfg(cfg)}]" + ("" if lim is None else f" limits {lim}"))


def refs(f_e11, f_e11g2):
    best = None
    print(f"{'grid':24s} {'method':20s} {'gain':>8s} {'clean':>8s} {'joint':>8s}")
    for tag, f in (("e11 kept", f_e11), ("e11g2 kept", f_e11g2)):
        for r in rows(f):
            print(line(tag, r))
            if not r.get("is_base_noop"):
                j = pts(r)[2]
                if best is None or j > best[0]:
                    best = (j, tag, r["method"])
    if best is None:
        print("every kept pick is the base; bar joint >= +1.000")
    else:
        print(f"best kept reference: {best[2]} ({best[1]}) joint {best[0]:+.3f}; bar: gain >= 10, clean >= -1.0, "
              f"joint >= {best[0] + 1:+.3f}")


def f1(f_floor, f_own, f_front, g2_floor, g2_own, g2_front):
    print(f"{'pick':24s} {'method':20s} {'gain':>8s} {'clean':>8s} {'joint':>8s}")
    kept = {r["method"]: r for r in rows(f_floor)}
    own = {r["method"]: r for r in rows(f_own)}
    for r in rows(g2_floor):
        if r["method"] in ("grad_full", "grad_last"):
            kept[r["method"] + " (e11g2)"] = r
    for r in rows(g2_own):
        if r["method"] in ("grad_full", "grad_last"):
            own[r["method"] + " (e11g2)"] = r
    front = rows(f_front) + rows(g2_front)
    for m in kept:
        print(line("kept (floor -20)", kept[m]))
        print(line("own (argmax S_joint)", own[m]))
        base_m = m.split(" ")[0]
        fr = [r for r in front if r["method"] == base_m]
        if fr:
            top = max(fr, key=lambda r: pts(r)[2])
            print(line("frontier max (descr.)", top))
            ok = [r for r in fr if pts(r)[1] >= -1.0]
            if ok:
                print(line("frontier max clean>=-1", max(ok, key=lambda r: pts(r)[2])))
    allf = [r for r in front]
    if allf:
        print(line("ALL frontier max", max(allf, key=lambda r: pts(r)[2])))
        ok = [r for r in allf if pts(r)[1] >= -1.0]
        if ok:
            print(line("ALL frontier clean>=-1", max(ok, key=lambda r: pts(r)[2])))


def xcheck(f_rec, f_new):
    a, b = rows(f_rec), rows(f_new)
    print(f"rows {len(a)} vs {len(b)}")
    for x, y in zip(a, b):
        assert x["method"] == y["method"]
        px, py = config_of(x)[0], config_of(y)[0]
        gx, gy = pts(x), pts(y)
        same_pick = px == py
        same_num = all(abs(u - v) < 1e-9 for u, v in zip(gx, gy))
        vx, vy = x["vcheck"], y["vcheck"]
        print(f"{x['method']:20s} pick {px}/{py} {'=' if same_pick else '!='}  4090 {gx[0]:+.3f}/{gx[1]:+.3f}/{gx[2]:+.3f}"
              f"  NSCC {gy[0]:+.3f}/{gy[1]:+.3f}/{gy[2]:+.3f}  {'equal' if same_pick and same_num else 'NOT equal'}"
              f"  V_check S {vx['score']}/{vy['score']}")


E11 = ["grad_last", "grad_full", "ridge_last", "prior_shift", "tie_inputs", "input_stats", "grad_sparse19_sal",
       "grad_sparse19_traj", "interp", "tent_ln", "sar"]
E11G2 = ["grad_last", "grad_full", "grad_sparse19_sal", "grad_sparse19_traj", "interp"]


def checks(rec_root, new_root, instance):
    """Per run (both grids) of INSTANCE under two run roots (<root>/baselines/<instance>/<method>/<label>/0/run.json):
    checks, checks whose child sha256 differs, checks whose S, S_new or S_old differs, and the run's own final."""
    total = [0, 0, 0]
    for label, methods in (("e11", E11), ("e11g2", E11G2)):
        for m in methods:
            a = json.loads((Path(rec_root) / "baselines" / instance / m / label / "0" / "run.json").read_text())
            b = json.loads((Path(new_root) / "baselines" / instance / m / label / "0" / "run.json").read_text())
            la, lb = a.get("oracle_log") or [], b.get("oracle_log") or []
            assert [e["patch_id"] for e in la] == [e["patch_id"] for e in lb], (label, m)
            sha = sum(x["child_sha256"] != y["child_sha256"] for x, y in zip(la, lb))
            diff = [(x["patch_id"], x["score"], y["score"], x["score_old"], y["score_old"]) for x, y in zip(la, lb)
                    if (x["score"], x["score_new"], x["score_old"]) != (y["score"], y["score_new"], y["score_old"])]
            total[0] += len(la)
            total[1] += sha
            total[2] += len(diff)
            fa, fb = a["final"], b["final"]
            print(f"{label:5s} {m:20s} checks {len(la)}  child sha differ {sha}  S differ {len(diff)}  final "
                  f"{fa['patch_id']} S {fa['score']} / {fb['patch_id']} S {fb['score']}"
                  + (f"  (patch, S, S, S_old, S_old) {diff}" if diff else ""))
    print(f"total checks {total[0]}, child sha256 differ {total[1]}, S/S_new/S_old differ {total[2]}")


if __name__ == "__main__":
    {"refs": refs, "f1": f1, "xcheck": xcheck, "checks": checks}[sys.argv[1]](*sys.argv[2:])
