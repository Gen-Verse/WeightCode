"""Run PREREG F1's fine-tuning methods on E11, E12 and T04 instances (``weightcode.baselines.fit.ftcompare``).

Methods: ft_ln, lora_r4, lora_r16, ft_full_noreplay, ft_full_lwf and ft_full_l2sp (all six by default), 8
configurations each, one joint check per configuration, on the instance's 64 condition rows and 64 clean replay rows
(ft_full_noreplay on the condition rows only). Runs go to ``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/f1/<seed>/``;
an existing run directory is never overwritten. Main instances are refused (exit status 2, nothing written). F1's
full and last-layer methods are the E11 second-grid runs (``run_fit_baselines.py --e11 --e11-grid2``), not rerun here.

On an E12 instance (cifar10_r18, the ImageNet-pretrained ResNet-18) the methods are ft_bn (for ft_ln), lora_r4,
lora_r16 (every convolution and fc), ft_full_noreplay, ft_full_lwf and ft_full_l2sp, all six by default; F1's full and
last layer there are ``run_fit_baselines.py --e12 --e12-grid2``. A PREREG T04 instance (in1k_t04, the native
ImageNet ResNet-18) runs E12's six; its full and last layer are ``run_fit_baselines.py --t04 --t04-grid2``. One
invocation takes instances of one dataset.

``--t04-edge`` (T04 instances only; PREREG T04's edge-extension amendment, main 9f0baa9): each method's grid with every
learning rate divided by 10, run label ``t04g3``; the pairs to run are scripts/t04/edge_extension.py's.

Read the runs as the references are read: ``python -m weightcode.evalkit.devreadout RUN --clean`` (own pick),
``--retention-floor -20`` (the kept pick) and ``--all-candidates`` (every configuration, the descriptive frontier).

Example::

    source scripts/env.sh
    .venv/bin/python scripts/run_ft_compare.py --instance i00802425 --threads 3 --methods lora_r4 lora_r16
"""

from __future__ import annotations

import argparse
import os
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
METHODS = ("ft_ln", "lora_r4", "lora_r16", "ft_full_noreplay", "ft_full_lwf", "ft_full_l2sp")
E12_METHODS = ("ft_bn", "lora_r4", "lora_r16", "ft_full_noreplay", "ft_full_lwf", "ft_full_l2sp")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", action="append", required=True,
                        help="E11, E12 or T04 dev instance id (repeatable, one dataset per invocation)")
    parser.add_argument("--methods", nargs="+", choices=sorted(set(METHODS) | set(E12_METHODS)),
                        help="default: the six of the instances' dataset")
    parser.add_argument("--seed", type=int, default=0, help="run seed (LoRA's A draw; directory name)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch and BLAS")
    parser.add_argument("--t04-edge", action="store_true",
                        help="T04 edge extension (every lr divided by 10), run label t04g3; needs --methods")
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
    from weightcode.baselines.fit.cli import T04_EDGE_LABEL
    from weightcode.baselines.fit.ftcompare import (
        F1_DATASET_METHODS,
        F1_E12_DATASET,
        F1_LABEL,
        F1_METHODS,
        F1_T04_DATASET,
        F1Error,
        run_f1,
    )
    from weightcode.baselines.fit.runner import SplitRoleError, baseline_run_dir

    if set(METHODS) != set(F1_METHODS) or set(E12_METHODS) != set(F1_DATASET_METHODS[F1_E12_DATASET]):
        raise RuntimeError(f"script methods {METHODS}, {E12_METHODS} differ from the library's {F1_DATASET_METHODS}")
    try:
        instances = load_checked_instances(list(dict.fromkeys(args.instance)), allow_main=False)
        datasets = {i.private.get("dataset") for i in instances}
        wrong = [i.instance_id for i in instances if i.private.get("dataset") not in F1_DATASET_METHODS]
        if wrong:
            raise F1Error(f"F1 runs on {sorted(F1_DATASET_METHODS)} instances; {wrong} are not")
        if len(datasets) != 1:
            raise F1Error(f"one invocation takes instances of one dataset, got {sorted(datasets)}")
        (dataset,) = datasets
        if args.t04_edge and (dataset != F1_T04_DATASET or not args.methods):
            raise F1Error(f"--t04-edge runs named methods on {F1_T04_DATASET} instances (got {dataset}, "
                          f"methods {args.methods})")
        methods = list(args.methods) if args.methods else list(F1_DATASET_METHODS[dataset])
        unfit = [m for m in methods if m not in F1_DATASET_METHODS[dataset]]
        if unfit:
            raise F1Error(f"{unfit} are not F1 methods on {dataset}; expected some of {F1_DATASET_METHODS[dataset]}")
    except (SplitRoleError, F1Error) as error:
        print(refusal_message(error), file=sys.stderr)
        return EXIT_REFUSED
    print(f"instances: {[instance.instance_id for instance in instances]}", flush=True)
    for instance in instances:
        for method in methods:
            run_dir = baseline_run_dir(instance.instance_id, method, args.seed,
                                       T04_EDGE_LABEL if args.t04_edge else F1_LABEL)
            record = run_f1(instance, method, run_dir, run_seed=args.seed, edge=args.t04_edge)
            print(summary_line(record), flush=True)
            print(f"  -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
