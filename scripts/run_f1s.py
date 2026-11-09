"""Run PREREG F1s (``weightcode.baselines.fit.f1s``): stronger fine-tuning arms s1 to s5 on E11 dev instances.

Runs go to ``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/f1s/<seed>/``; an existing run directory is never
overwritten. Read the runs as F1's: ``python -m weightcode.evalkit.devreadout RUN --clean`` (own), ``--retention-floor -20``
(kept) and ``--all-candidates`` (frontier). ``f1s_s5_oracle_prior`` needs ``--s5-rows`` (the record written by
``scripts/f1s/f1s_s5_rows.py``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
METHODS = ("f1s_s1_gentle", "f1s_s2_aug", "f1s_s3_lora_aug", "f1s_s4_est_prior", "f1s_s5_oracle_prior",
           "k1_ft_rule")
ROWS_METHODS = ("f1s_s5_oracle_prior", "k1_ft_rule")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", action="append", required=True, help="E11 dev instance id (repeatable)")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0, help="run seed (LoRA's A, minibatch and augmentation draws)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch and BLAS")
    parser.add_argument("--s5-rows", type=Path, help="rows record: f1s_s5_oracle_prior's s5_rows.json, or k1_ft_rule's "
                                                       "k1_rule_rows.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
        os.environ[variable] = str(args.threads)
    sys.dont_write_bytecode = True
    import torch

    torch.set_num_threads(args.threads)
    from weightcode.baselines.fit.cli import EXIT_REFUSED, load_checked_instances, refusal_message, summary_line
    from weightcode.baselines.fit.f1s import F1S_LABEL, F1S_METHODS, F1sError, run_f1s
    from weightcode.baselines.fit.ftcompare import F1_DATASET
    from weightcode.baselines.fit.runner import SplitRoleError, baseline_run_dir

    if set(METHODS) != set(F1S_METHODS):
        raise RuntimeError(f"script methods {METHODS} differ from the library's {F1S_METHODS}")
    try:
        instances = load_checked_instances(list(dict.fromkeys(args.instance)), allow_main=False)
        wrong = [i.instance_id for i in instances if i.private.get("dataset") != F1_DATASET]
        if wrong:
            raise F1sError(f"F1s runs on {F1_DATASET} instances; {wrong} are not")
        if len({m in ROWS_METHODS for m in args.methods}) > 1:
            raise F1sError("--methods mixes arms that need --s5-rows with arms that do not; run them separately")
        if any(m in ROWS_METHODS for m in args.methods) and args.s5_rows is None:
            raise F1sError(f"{args.methods} need --s5-rows")
    except (SplitRoleError, F1sError) as error:
        print(refusal_message(error), file=sys.stderr)
        return EXIT_REFUSED
    print(f"instances: {[instance.instance_id for instance in instances]}", flush=True)
    for instance in instances:
        for method in args.methods:
            run_dir = baseline_run_dir(instance.instance_id, method, args.seed, F1S_LABEL)
            rows = args.s5_rows if method in ROWS_METHODS else None
            record = run_f1s(instance, method, run_dir, run_seed=args.seed, s5_rows=rows)
            print(summary_line(record), flush=True)
            print(f"  -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
