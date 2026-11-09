"""NSCC R10 smoke for PREREG F1 on a CPU node: the store path remap, the files F1 opens, and timings.

Inside a store job, at one thread count (set before numpy and torch are
imported, as run_ft_compare.py and the readouts set it), for each instance: load and hash-check it through the mounted
store (``load_checked_instances``), build its joint evidence (condition plus replay rows) and its joint oracle (the
base predictions on both V_check parts: the cost of one check), load its V_select pair, time the numpy forward
(forward.py, the check and readout path) on the clean V_check rows, and time full-batch Adam steps of full
fine-tuning on the 128 rows. Every store file opened is listed (an audit hook), so the inputs list can be checked
against what F1 reads. Nothing is written to the store; one JSON line per measurement goes to stdout. Not a result:
the timings set the main job's thread layout and walltime.

    python -B scripts/store/f1_smoke.py --instances i3eb3fc78 i00802425 --threads 16 --steps 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instances", nargs="+", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--steps", type=int, default=5)
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
    sys.dont_write_bytecode = True
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    from weightcode.baselines.fit.cli import load_checked_instances
    from weightcode.baselines.fit.grad import adam_checkpoints
    from weightcode.baselines.fit.runner import load_joint_evidence, read_model_spec
    from weightcode.baselines.fit.torch_forward import architecture_from, images_tensor, trainable_tensors
    from weightcode.evalkit import devreadout
    from weightcode.evalkit.companions import clean_companion, make_joint_oracle
    from weightcode.evalkit.patch_check import predict

    def emit(**record: object) -> None:
        print(json.dumps(record), flush=True)

    emit(kind="env", threads=args.threads, python=sys.version.split()[0], numpy=np.__version__,
         torch=torch.__version__, torch_threads=torch.get_num_threads(), nproc=os.cpu_count(),
         weightcode_private=os.environ.get("WEIGHTCODE_PRIVATE"), weightcode_runs=os.environ.get("WEIGHTCODE_RUNS"),
         sealed_entries=len(os.listdir(os.environ["WEIGHTCODE_SEALED"])))
    for instance_id in args.instances:
        t0 = time.perf_counter()
        instance = load_checked_instances([instance_id], allow_main=False)[0]
        evidence = load_joint_evidence(instance)
        t1 = time.perf_counter()
        make_joint_oracle(instance, budget=8)
        t2 = time.perf_counter()
        devreadout.readout_split(instance)
        dataset, condition = str(instance.private["dataset"]), str(instance.private["condition"])
        clean_companion(dataset, condition, "V_select")
        clean_check = clean_companion(dataset, condition, "V_check")
        t3 = time.perf_counter()
        predict(instance.forward, instance.params, clean_check.x, 1024)
        t4 = time.perf_counter()
        emit(kind="load", instance=instance_id, condition=condition, threads=args.threads,
             rows=int(len(evidence.y)), n_replay=int(evidence.n_replay), load_s=round(t1 - t0, 2),
             oracle_base_s=round(t2 - t1, 2), split_load_s=round(t3 - t2, 2),
             predict_clean_vcheck_s=round(t4 - t3, 2), clean_vcheck_rows=int(len(clean_check.y)))
        arch = architecture_from(instance.params, read_model_spec(instance))
        x = images_tensor(evidence.x)
        y = torch.from_numpy(evidence.y.astype(np.int64))
        trainable = trainable_tensors(arch, instance.params)
        start = time.perf_counter()
        adam_checkpoints(arch, instance.params, x, y, trainable=trainable, lr=1e-5, checkpoints=(args.steps,), seed=0)
        emit(kind="adam_full", instance=instance_id, threads=args.threads, steps=args.steps,
             s_per_step=round((time.perf_counter() - start) / args.steps, 3))
    emit(kind="opened", threads=args.threads, n=len(opened), files=sorted(opened))
    return 0


if __name__ == "__main__":
    sys.exit(main())
