"""Train PREREG F2's fine-tuning trajectories on the GPU (``weightcode.baselines.fit.labeleq``; NSCC request R9).

``--plan main``: every trajectory of the given conditions (``f2_trajectories``: 6 N x 4 lr per condition, to 1000
steps with a checkpoint at 100, and to 5000 steps at N = 40,000), longest first. ``--plan smoke``: the timing smoke
(``smoke_trajectories``), which checks determinism (a duplicate trajectory) and that the 100-step checkpoint of a
longer trajectory equals a 100-step trajectory, bit for bit.

The parent runs ``--concurrent`` child processes at a time on the one device, each one trajectory with
``--num-workers`` loader workers, and starts the next when one exits. Each child writes its checkpoints under
``--out-root`` (model.npz and record.json per checkpoint, ``weightcode.baselines.fit.labeleq.checkpoint_dir``) and its
index rows to a row file; the parent writes ``<out-root>/index.json`` once every child exited 0, and otherwise
``failures.json`` and exit status 1 (the finished checkpoints stay). The job reads cifar10_vl7's dataset.json, its
train split, the student's model.npz and the F2 files (f2_data.json, subsets.npz, the corrupted train arrays); no
V_evidence, V_check, V_select or test file.

Usage (inside the NSCC container, from the run's code copy):
    python -u scripts/train_f2.py --plan main --out-root {RUN_DIR}/private/f2_models --concurrent 6 --num-workers 3 \\
        --device cuda --git-state-file {RUN_DIR}/git_state.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _plan(name: str, conditions: list[str]) -> list[Any]:
    from weightcode.baselines.fit import labeleq as F

    if name == "smoke":
        return F.smoke_trajectories()
    return F.f2_trajectories(conditions)


def run_child(args: argparse.Namespace) -> int:
    import torch

    from weightcode.baselines.fit import labeleq as F
    from weightcode.models.runinfo import run_info
    from weightcode.models.train import configure_determinism

    (trajectory,) = [t for t in _plan(args.plan, args.conditions) if t.name == args.child]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but CUDA is not available")
    determinism = configure_determinism(args.seed)
    data = F.load_f2_data([trajectory.condition])
    info = run_info(REPO_ROOT, args.git_state_file)
    info["device"] = f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'} ({device})"
    info["determinism"] = determinism
    info["pbs_jobid"] = os.environ.get("PBS_JOBID")
    info["torch_cuda"] = torch.version.cuda
    info["plan"] = args.plan

    def log(entry: dict[str, Any]) -> None:
        if entry["step"] % 500 == 0 or entry["step"] in (1, 100):
            print(json.dumps(entry), flush=True)

    rows = F.train_trajectory(trajectory, data, device=device, seed=args.seed, num_workers=args.num_workers,
                              out_root=args.out_root, provenance=info, log_fn=log)
    with open(args.row_file, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(rows) + "\n")
    print(json.dumps({"trajectory": trajectory.name, "done": True, "rows": rows}), flush=True)
    return 0


def _command(args: argparse.Namespace, name: str, row_file: Path) -> list[str]:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--plan", args.plan, "--out-root", str(args.out_root),
           "--device", args.device, "--num-workers", str(args.num_workers), "--seed", str(args.seed),
           "--child", name, "--row-file", str(row_file), "--conditions", *args.conditions]
    if args.git_state_file is not None:
        cmd += ["--git-state-file", str(args.git_state_file)]
    return cmd


def run_parent(args: argparse.Namespace) -> int:
    from weightcode.baselines.fit import labeleq as F

    trajectories = _plan(args.plan, args.conditions)
    root = Path(args.out_root)
    if (root / "index.json").exists() or (root / "failures.json").exists():
        raise SystemExit(f"{root} already holds an index or failures; nothing started")
    logs, rows_dir = root / "logs", root / "rows"
    logs.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(exist_ok=True)
    F.load_f2_data(sorted({t.condition for t in trajectories}))  # hash checks once, before any child starts
    pending = list(trajectories)
    running: dict[str, tuple[subprocess.Popen, Any, float]] = {}
    results: dict[str, dict[str, Any]] = {}
    started = time.time()
    print(json.dumps({"plan": args.plan, "trajectories": len(pending), "concurrent": args.concurrent,
                      "num_workers": args.num_workers, "started": started}), flush=True)
    while pending or running:
        while pending and len(running) < args.concurrent:
            t = pending.pop(0)
            row_file = rows_dir / f"{t.name}.json"
            if row_file.exists():
                raise SystemExit(f"{row_file} exists")
            handle = open(logs / f"{t.name}.log", "w", encoding="utf-8")  # closed when the child exits
            proc = subprocess.Popen(_command(args, t.name, row_file), stdout=handle, stderr=subprocess.STDOUT,
                                    cwd=REPO_ROOT)
            running[t.name] = (proc, handle, time.time())
        time.sleep(2)
        for name in list(running):
            proc, handle, t0 = running[name]
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            del running[name]
            tail = (logs / f"{name}.log").read_text(encoding="utf-8").strip().splitlines()[-1:] or [""]
            results[name] = {"exit": code, "wall_s": round(time.time() - t0, 1), "last_line": tail[0][:300]}
            print(json.dumps({"trajectory": name, "exit": code, "wall_s": results[name]["wall_s"],
                              "elapsed_s": round(time.time() - started, 1), "running": len(running),
                              "pending": len(pending)}), flush=True)
    failed = {name: r for name, r in results.items() if r["exit"] != 0}
    if failed:
        (root / "failures.json").write_text(json.dumps({"failed": failed, "results": results}, indent=1) + "\n")
        print(f"{len(failed)} trajectories failed: {sorted(failed)}; no index written", file=sys.stderr)
        return 1
    rows = [row for t in trajectories for row in json.loads((rows_dir / f"{t.name}.json").read_text())]
    index = {"schema": "weightcode.f2.index.v1", "plan": args.plan, "conditions": args.conditions, "seed": args.seed,
             "concurrent": args.concurrent, "num_workers": args.num_workers, "pbs_jobid": os.environ.get("PBS_JOBID"),
             "trajectories": [t.as_dict() for t in trajectories], "results": results, "rows": rows,
             "elapsed_s": round(time.time() - started, 1)}
    if args.plan == "smoke":
        by = {(r["trajectory"], r["steps"]): r["params_sha256"] for r in rows}
        at = {(r["trajectory"], r["steps"]): r["elapsed_s"] for r in rows}
        a = "gaussian_noise_n1024_lr0.0001"
        index["smoke_checks"] = {
            "duplicate_equal_100": by[(a, 100)] == by[(a + "_dup", 100)],
            "duplicate_equal_200": by[(a, 200)] == by[(a + "_dup", 200)],
            "checkpoint_equals_short_run_100": by[(a, 100)] == by[(a + "_short", 100)],
            "steps_per_s_100_to_200": {t.name: round(100 / (at[(t.name, 200)] - at[(t.name, 100)]), 2)
                                       for t in trajectories if (t.name, 200) in at},
            "s_to_step_100": {t.name: at[(t.name, 100)] for t in trajectories},
        }
        print(json.dumps({"smoke_checks": index["smoke_checks"]}), flush=True)
    (root / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(json.dumps({"index": str(root / "index.json"), "n_checkpoints": len(rows),
                      "elapsed_s": index["elapsed_s"]}), flush=True)
    if args.plan == "smoke" and not all(v for k, v in index["smoke_checks"].items() if k.startswith(("dup", "check"))):
        print("smoke check failed: training is not deterministic or a checkpoint differs from the short run",
              file=sys.stderr)
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", choices=("main", "smoke"), required=True)
    parser.add_argument("--conditions", nargs="+", default=["gaussian_noise", "contrast"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--concurrent", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="run seed (PREREG F2: 0)")
    parser.add_argument("--git-state-file", type=Path, help="git_state.json of a job's code copy (run_info)")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--row-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child is not None:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    sys.exit(main())
