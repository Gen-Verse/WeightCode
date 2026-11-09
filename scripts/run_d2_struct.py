"""Run PREREG D2 (``weightcode.baselines.fit.d2struct``): the LLM final's structure trained by gradient, on an E11 instance.

Methods: d2_struct_base (start (i), the base) and d2_struct_final (start (ii), the final's child), 8 configurations
each, one joint check per configuration, on the instance's 64 condition rows and 64 clean replay rows. The trainable
entries are those the final changes, rebuilt from its program-transfer record on this instance (``--transfer``).
Runs go to ``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/d2/<seed>/``; an existing run directory is never
overwritten. Main instances are refused (exit status 2, nothing written).

Read the runs as F1's are read: ``python -m weightcode.evalkit.devreadout RUN --clean`` (own pick), ``--retention-floor
-20`` (the kept pick) and ``--all-candidates`` (every configuration).

Example::

    source scripts/env.sh
    .venv/bin/python scripts/run_d2_struct.py --instance i00802425 \\
        --transfer $WEIGHTCODE_RUNS/devreadouts/learn/sc918e11dceef637a/transfer/transfer_3c1cebc35ef0_i00802425.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
METHODS = ("d2_struct_base", "d2_struct_final")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", required=True, help="E11 dev instance id")
    parser.add_argument("--transfer", type=Path, required=True,
                        help="the final's program-transfer record on this instance (transfer_<program>_<id>.json)")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0, help="run seed (directory name)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch and BLAS")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
        os.environ[variable] = str(args.threads)
    sys.dont_write_bytecode = True  # importing an instance's forward.py must not write __pycache__ into $WEIGHTCODE_PRIVATE
    import torch

    torch.set_num_threads(args.threads)
    from weightcode.baselines.fit.cli import EXIT_REFUSED, load_checked_instances, refusal_message, summary_line
    from weightcode.baselines.fit.d2struct import D2_LABEL, D2_METHODS, D2Error, run_d2
    from weightcode.baselines.fit.ftcompare import F1_DATASET
    from weightcode.baselines.fit.runner import SplitRoleError, baseline_run_dir

    if set(METHODS) != set(D2_METHODS):
        raise RuntimeError(f"script methods {METHODS} differ from the library's {D2_METHODS}")
    try:
        instance = load_checked_instances([args.instance], allow_main=False)[0]
        if instance.private.get("dataset") != F1_DATASET:
            raise D2Error(f"D2 runs on {F1_DATASET} instances; {instance.instance_id} is not")
        if not args.transfer.is_file():
            raise D2Error(f"no transfer record {args.transfer}")
    except (SplitRoleError, D2Error) as error:
        print(refusal_message(error), file=sys.stderr)
        return EXIT_REFUSED
    print(f"instance: {instance.instance_id}; transfer: {args.transfer}", flush=True)
    for method in args.methods:
        run_dir = baseline_run_dir(instance.instance_id, method, args.seed, D2_LABEL)
        record = run_d2(instance, method, args.transfer, run_dir, run_seed=args.seed)
        print(summary_line(record), flush=True)
        print(f"  -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
