"""NSCC R22-e13 smoke for PREREG E13: the store path remap, the files the references open, timings and code paths.

Inside a store job, at one CPU thread count (set before numpy and torch are
imported, as run_scan_refs.py and the readouts set it) and one fit device (cuda as configured by
``scanrefs.configure_device``, or cpu), for the seed-100 scanskill-v1 instance of one condition. Two modes:

* ``time``: load and hash-check the instance through the mounted store (``check_scan_current``), build its experience
  rows and its joint oracle (the base predictions on new V_check and the 1000 old V_check rows), time one check of the
  base child (the cost of every check), time one readout (new and old V_select, base and child), and time full-batch
  Adam steps of full fine-tuning on the experience rows and on the two rule-generated fit sets (2000 and 4000 generated
  rows plus the experience rows) after two warm-up steps. Every store file opened is listed (an audit hook).
* ``paths``: run one reference end to end (``run_reference``, 8 checks by the oracle) on the device with the step grid
  cut to (1, 2, 3, 4), the regularized grid to (1, 2) and the trajectory mask to 3 steps, into ``--out-root``, so every
  code path of the main job runs once on the device. These runs are smoke runs, never references.

Nothing is written to the store outside ``--out-root``; one JSON line per measurement goes to stdout. Not a result:
the timings set the main job's layout and walltime.

    python -B scripts/store/e13_smoke.py time --condition addprim_jump --threads 1 --device cuda --steps 20
    python -B scripts/store/e13_smoke.py paths --condition addprim_jump --method lora_r4 --threads 1 \\
        --device cuda --out-root "$WEIGHTCODE_RUNS/smoke_paths"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SMOKE_STEPS = (1, 2, 3, 4)
SMOKE_REG_STEPS = (1, 2)
SMOKE_TRAJ_STEPS = 3
WARMUP_STEPS = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("time", "paths"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--steps", type=int, default=20, help="time: timed Adam steps per fit set")
    parser.add_argument("--method", help="paths: the reference to run")
    parser.add_argument("--out-root", type=Path, help="paths: smoke run directories go under here")
    args = parser.parse_args(argv)
    opened: set[str] = set()
    store_roots = tuple(os.path.realpath(os.environ[name]) + os.sep
                        for name in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED", "WEIGHTCODE_RUNS"))

    def hook(event: str, hook_args: tuple) -> None:
        if event == "open" and hook_args and isinstance(hook_args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(hook_args[0])
            if os.path.realpath(path).startswith(store_roots):
                opened.add(os.path.normpath(path))

    sys.addaudithook(hook)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    if args.device == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.dont_write_bytecode = True
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    from weightcode.baselines.fit import scanrefs as R
    from weightcode.baselines.fit import seqgrad as SG
    from weightcode.baselines.fit.cli import summary_line
    from weightcode.baselines.fit.runner import load_scan_evidence
    from weightcode.evalkit import scanskill as K

    def emit(**record: object) -> None:
        print(json.dumps(record), flush=True)

    settings = R.configure_device(args.device)
    emit(kind="env", mode=args.mode, threads=args.threads, python=sys.version.split()[0], numpy=np.__version__,
         torch=torch.__version__, torch_threads=torch.get_num_threads(), nproc=os.cpu_count(), device=settings,
         weightcode_private=os.environ.get("WEIGHTCODE_PRIVATE"), weightcode_runs=os.environ.get("WEIGHTCODE_RUNS"),
         sealed_entries=len(os.listdir(os.environ["WEIGHTCODE_SEALED"])))
    (instance_id,) = K.active_scan_ids(args.condition)

    if args.mode == "paths":
        if not args.method or args.out_root is None:
            raise SystemExit("paths needs --method and --out-root")
        R.STEP_CHECKPOINTS, R.REG_STEPS, R.TRAJ_STEPS = SMOKE_STEPS, SMOKE_REG_STEPS, SMOKE_TRAJ_STEPS
        instance = K.load_scan_instance(instance_id)
        fit_set = None
        if args.method in R.RULEGEN_METHODS:
            fit_set = R.rulegen_fit_set(args.condition, R.RULEGEN_FITS[R.rulegen_parts(args.method)[0]])
        start = time.perf_counter()
        record = R.run_reference(instance, args.method, fit_set=fit_set, device=args.device,
                                 run_dir=args.out_root / instance_id / args.method)
        emit(kind="paths", instance=instance_id, condition=args.condition, method=args.method,
             status=record.get("status"), seconds=round(time.perf_counter() - start, 1), summary=summary_line(record))
        return 0 if record.get("status") == "complete" else 1

    t0 = time.perf_counter()
    instance = K.load_scan_instance(instance_id)
    K.check_scan_current(instance.instance_dir)
    evidence = load_scan_evidence(instance)
    t1 = time.perf_counter()
    oracle = K.make_scan_oracle(instance, budget=1)
    t2 = time.perf_counter()
    entry = oracle.query(dict(instance.params), "smoke")
    t3 = time.perf_counter()
    K.scan_readout(instance, dict(instance.params))
    t4 = time.perf_counter()
    emit(kind="load", instance=instance_id, condition=args.condition, threads=args.threads,
         rows=int(len(evidence.y)), n_prim=int(evidence.n_rand), n_replay=int(evidence.n_replay),
         load_s=round(t1 - t0, 2), oracle_base_s=round(t2 - t1, 2), check_s=round(t3 - t2, 2),
         check_score=int(entry["score"]), check_rows=int(entry["n"]), readout_s=round(t4 - t3, 2))
    fits = {"experience": (evidence.x, evidence.y)}
    for fit in ("full", "mixed"):
        generated = R.rulegen_fit_set(args.condition, fit)
        fits[fit] = (np.concatenate([generated["x"], evidence.x]), np.concatenate([generated["y"], evidence.y]))
    for name, (x, y) in fits.items():
        x_t, y_t = torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))
        SG.adam_checkpoints(instance.params, x_t, y_t, trainable=sorted(instance.params), lr=1e-3,
                            checkpoints=(WARMUP_STEPS,), seed=0, device=args.device)
        start = time.perf_counter()
        SG.adam_checkpoints(instance.params, x_t, y_t, trainable=sorted(instance.params), lr=1e-3,
                            checkpoints=(args.steps,), seed=0, device=args.device)
        emit(kind="adam_full", fit=name, rows=int(len(y)), threads=args.threads, device=args.device,
             steps=args.steps, s_per_step=round((time.perf_counter() - start) / args.steps, 4))
    if args.device == "cuda":
        emit(kind="cuda_memory", max_allocated_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    emit(kind="opened", threads=args.threads, n=len(opened), files=sorted(opened))
    return 0


if __name__ == "__main__":
    sys.exit(main())
