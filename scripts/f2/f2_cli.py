"""PREREG F2 (the label-equivalent amount) outside the training job: data, import, checks and the summary.

Subcommands (``weightcode.baselines.fit.labeleq``):

* ``data --corrupt-dir DIR``: subsets.npz and f2_data.json from the corrupted train arrays (workstation).
* ``import --run-dir DIR``: hash and device checks of a fetched training run; writes ``DIR/f2_import.json``.
* ``check --instance I --run-dir DIR [--groups G ...]``: one baseline run per check group (``n<N>``, ``ceiling``) on
  the E11 instance I, the references' joint oracle and executor without size limits, written to
  ``$WEIGHTCODE_RUNS/baselines/<I>/ft_labeleq/f2_<group>/<seed>/`` (4090 CPU, inside the store-copy bwrap).
* ``summary --instance I --kept FILE --frontier FILE [--teacher-joint J] --out FILE``: the table per N (kept pick,
  the run's own no-floor pick and every configuration), the ceiling and N* from the devreadout records. The kept
  pick's readout must equal its frontier record (two readouts of one child); the own pick is read from the frontier.

Readouts are the project's (``python -m weightcode.evalkit.devreadout RUN ... --retention-floor -20`` for the kept picks,
``--all-candidates`` for the frontier).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
REPO_ROOT = Path(__file__).resolve().parents[2]


def cmd_data(args: argparse.Namespace) -> int:
    from weightcode.baselines.fit import labeleq as F
    from weightcode.models.runinfo import run_info

    payload = F.build_f2_data(args.corrupt_dir, run_info(REPO_ROOT))
    print(json.dumps({k: payload[k] for k in ("pool", "subsets")}, indent=1))
    print(f"-> {F.f2_dir() / F.F2_DATA_JSON}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from weightcode.baselines.fit import labeleq as F

    payload = F.import_f2_run(args.run_dir, args.models_subdir)
    print(json.dumps({k: payload[k] for k in ("n_checkpoints", "device_max_abs_diff", "index_sha256")}))
    print(f"-> {Path(args.run_dir) / F.F2_IMPORT_JSON}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(args.threads)
    from weightcode.baselines.fit import labeleq as F
    from weightcode.baselines.fit.cli import load_checked_instances, summary_line
    from weightcode.baselines.fit.runner import baseline_run_dir

    (instance,) = load_checked_instances([args.instance], allow_main=False)
    F.condition_of(instance)
    record = json.loads((Path(args.run_dir) / F.F2_IMPORT_JSON).read_text())
    for group in args.groups or F.groups():
        run_dir = baseline_run_dir(instance.instance_id, F.F2_METHOD, args.seed, f"f2_{group}")
        result = F.run_f2(instance, record, group, run_dir, run_seed=args.seed)
        print(summary_line(result), flush=True)
        print(f"  -> {run_dir}", flush=True)
    return 0


def _points(record: dict[str, Any]) -> dict[str, float]:
    ro, rc = record["readouts"], record["readouts_clean"]
    gain, clean = 100 * ro["delta_acc"], 100 * rc["delta_acc"]
    return {"gain": gain, "clean": clean, "joint": gain + clean, "worst_class_clean": 100 * rc["worst_class_delta"]}


def _own(pick: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The run's own pick (argmax S_joint, no floor; the kept record's ``primary_patch_id``) from the frontier."""
    patch = pick["selection"]["primary_patch_id"]
    if patch is None:
        return {"patch_id": None, "is_base": True, "gain": 0.0, "clean": 0.0, "joint": 0.0, "worst_class_clean": 0.0}
    (row,) = [r for r in rows if r["patch_id"] == patch]
    return {"patch_id": patch, "is_base": False, "config": row["config"], "vcheck": row["vcheck"], **_points(row)}


def _check_kept_matches_frontier(group: str, pick: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    patch = pick["selection"]["patch_id"]
    if patch is None:
        if pick["readouts"]["delta_acc"] != 0 or pick["readouts_clean"]["delta_acc"] != 0:
            raise SystemExit(f"{group}: a base pick with nonzero deltas")
        return
    (row,) = [r for r in rows if r["patch_id"] == patch]
    for key in ("readouts", "readouts_clean"):
        if any(pick[key][f] != row[key][f] for f in ("delta_acc", "fixed", "broken", "worst_class_delta")):
            raise SystemExit(f"{group}: the kept pick's {key} differ from its frontier record")
    if pick["vcheck"]["score"] != row["vcheck"]["score"]:
        raise SystemExit(f"{group}: the kept pick's V_check score differs from its frontier record")


def cmd_summary(args: argparse.Namespace) -> int:
    from weightcode.baselines.fit import labeleq as F

    def read(path: Path | None) -> list[dict[str, Any]]:
        return [] if path is None else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def group_of(record: dict[str, Any]) -> str:
        return Path(record["run"]).parent.parent.name.removeprefix("f2_")

    kept, frontier = read(args.kept), read(args.frontier)
    for rec in kept + frontier:
        if rec["instance_id"] != args.instance:
            raise SystemExit(f"a record of {rec['instance_id']}, expected {args.instance}")
    condition = (kept or frontier)[0]["condition"]
    table: dict[str, Any] = {}
    for group in F.groups():
        n, configs = F.group_rows(group)
        rows = sorted([r for r in frontier if group_of(r) == group], key=lambda r: r["query_index"])
        picks = [r for r in kept if group_of(r) == group]
        if not rows and not picks:
            continue
        if len(picks) != 1:
            raise SystemExit(f"{group}: {len(picks)} kept-pick records, expected 1")
        pick = picks[0]
        if rows:
            _check_kept_matches_frontier(group, pick, rows)
        table[group] = {
            "n": n,
            "configs": [{"config": r["config"], "patch_id": r["patch_id"], "vcheck": r["vcheck"],
                         "within_program_limits": r["within_program_limits"], **_points(r)} for r in rows],
            "kept": {"patch_id": pick["selection"]["patch_id"], "is_base": pick["selection"]["patch_id"] is None,
                     "n_eligible": pick["selection"]["n_eligible"], "n_checks": pick["selection"]["n_checks"],
                     "config": next((r["config"] for r in rows if r["patch_id"] == pick["selection"]["patch_id"]),
                                    None),
                     "vcheck": pick["vcheck"], **_points(pick)},
            "own": _own(pick, rows),
        }
        if rows and len(rows) != len(configs):
            raise SystemExit(f"{group}: {len(rows)} frontier records, expected {len(configs)}")
    teacher = args.teacher_joint if args.teacher_joint is not None else F.F2_TEACHER_JOINT.get(condition)
    points = [{"n": t["n"], "joint": t["kept"]["joint"], "clean": t["kept"]["clean"]}
              for g, t in table.items() if g != F.CEILING]
    payload = {"schema": "weightcode.f2.summary.v1", "prereg": F.PREREG_F2, "instance_id": args.instance,
               "condition": condition, "limits": F.LIMITS_NOTE, "teacher_joint": teacher,
               "n_star": F.n_star(points, teacher) if teacher is not None else None,
               "inputs": {"kept": str(args.kept), "frontier": str(args.frontier)},
               "groups": table}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    for group, t in table.items():
        k = t["kept"]
        print(f"{group}: kept {k['patch_id'] or 'base'} {k['config']} V_check S {k['vcheck']['score']} "
              f"(new {k['vcheck']['score_new']}, old {k['vcheck']['score_old']}); V_select gain {k['gain']:+.3f} "
              f"clean {k['clean']:+.3f} joint {k['joint']:+.3f} worst {k['worst_class_clean']:+.3f}")
    print(f"N*: {payload['n_star']['text'] if payload['n_star'] else 'teacher joint not given'}")
    print(f"-> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("data")
    p.add_argument("--corrupt-dir", type=Path, required=True)
    p = sub.add_parser("import")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--models-subdir", default="private/f2_models")
    p = sub.add_parser("check")
    p.add_argument("--instance", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--groups", nargs="+")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("summary")
    p.add_argument("--instance", required=True)
    p.add_argument("--kept", type=Path, required=True)
    p.add_argument("--frontier", type=Path, required=True)
    p.add_argument("--teacher-joint", type=float)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
            os.environ[variable] = str(args.threads)
        sys.dont_write_bytecode = True  # importing an instance's forward.py must not write __pycache__
    return {"data": cmd_data, "import": cmd_import, "check": cmd_check, "summary": cmd_summary}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
