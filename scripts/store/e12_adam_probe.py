"""NSCC R17-e12 diagnostic: full fine-tuning Adam steps on an E12 instance's 128 evidence rows, nothing else.

The R17 CPU smoke 236488 segfaulted (exit 139) in the 8-thread process at its full fine-tuning step, while the
16-thread process finished. This probe runs only that step (``adam_checkpoints`` on ``trainable_tensors``, lr 1e-5,
as f1_smoke.py), at one thread count set before numpy and torch are imported, with ``-X faulthandler`` expected on
the command line so a crash prints the Python stack. Optionally oneDNN off (``--no-mkldnn``), to tell a oneDNN
convolution fault from any other. Nothing is written to the store; one JSON line per repeat goes to stdout. Not a
result.

    python -X faulthandler -B scripts/store/e12_adam_probe.py --instance i26d23d3e --threads 8 --steps 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-mkldnn", action="store_true")
    args = parser.parse_args(argv)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    if args.no_mkldnn:
        torch.backends.mkldnn.enabled = False
    from weightcode.baselines.fit.cli import load_checked_instances
    from weightcode.baselines.fit.grad import adam_checkpoints
    from weightcode.baselines.fit.runner import load_joint_evidence, read_model_spec
    from weightcode.baselines.fit.torch_forward import architecture_from, images_tensor, trainable_tensors

    instance = load_checked_instances([args.instance], allow_main=False)[0]
    evidence = load_joint_evidence(instance)
    arch = architecture_from(instance.params, read_model_spec(instance))
    x = images_tensor(evidence.x)
    y = torch.from_numpy(evidence.y.astype(np.int64))
    trainable = trainable_tensors(arch, instance.params)
    for repeat in range(args.repeats):
        start = time.perf_counter()
        adam_checkpoints(arch, instance.params, x, y, trainable=trainable, lr=1e-5, checkpoints=(args.steps,), seed=0)
        print(json.dumps({"kind": "adam_full", "instance": args.instance, "threads": args.threads,
                          "mkldnn": torch.backends.mkldnn.enabled, "repeat": repeat, "steps": args.steps,
                          "s_per_step": round((time.perf_counter() - start) / args.steps, 3)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
