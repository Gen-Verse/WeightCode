"""Run the fitting references and the fixed-rule arms on repair instances.

Fitting references: grad_last, grad_sparse, grad_full, ridge_last (bounded) and ridge_last_free (unconstrained, like
grad_full). Fixed-rule arms: prior_shift (label-shift correction of the final-layer bias), tie_inputs (first-layer
input weights tied across input channels) and input_stats (label-free per-channel input statistics); see
``weightcode.baselines.fit.rules``.

Instances: the given ``--instance`` ids, or by default every instance in ``$WEIGHTCODE_PRIVATE/instances/index.json`` with
split_role "dev" (optionally one ``--dataset``). Every selected instance's private.json split_role is checked before
anything runs: a main instance is refused (exit status 2, nothing written) unless ``--allow-main`` is given, which is
reserved for runs after the pre-registration freeze (PREREG section 2).

Each method on each instance gets its own counted oracle with 8 patch-check queries and writes
``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/<seed>/`` (one ``q<k>/`` patch per checked candidate plus run.json).
An existing run directory is never overwritten.

``--joint`` (PREREG E1, evidence-v1 instances of a named condition): the label fits train on the 64 evidence rows
plus the dataset's 64 replay rows (clean train images, replay-v1), the fixed rules keep the 64 evidence rows, and
every candidate is scored by the joint oracle (condition V_check plus its clean companion, S_joint = S_new + S_old).
Runs go to ``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/e1joint/<seed>/``.

``--fewshot`` (PREREG E2, fewshot-v1 instances: ``--instance f<8 hex>`` or every active one in
``$WEIGHTCODE_PRIVATE/fewshot_instances/index.json``, optionally one ``--dataset``): the four E2 references by default
(imprint, ridge_margin, grad_last, grad_full), each fitted on the instance's 50 new-class rows plus its 64 replay
rows, scored by the fewshot joint oracle (V_check rows of the new classes and of the train classes) and held to the
E2 program limits (head exempt from max_rel_frob, at most 20% of the entries changed). Runs go to
``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/e2/<seed>/``.

``--e10`` (PREREG E10, evidence-v1 instances of the cifar10_r20 conditions; implies ``--joint``): the E10 references
by default (E1's six; grad_sparse19_sal, grad_sparse19_traj and interp; bn_adapt and tent, which read the 64
condition inputs only), scored by the joint oracle and held to the E10 program limits (rel-Frob 0.5 on tensors of rank
2 or more, at most 20% of the entries changed, max_ops at least the tensor count). Runs go to
``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/e10/<seed>/``.

``--e11`` (PREREG E11, evidence-v1 instances of the cifar10_vl7 conditions, a ViT-Lite-7/4 student; implies
``--e10``'s limits and oracle): the E11 references by default (E1's six adapted; grad_sparse19_sal, grad_sparse19_traj
and interp; tent_ln and sar, which read the 64 condition inputs only). Runs go to
``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/e11/<seed>/``.

``--e11-grid2`` (with ``--e11``; PREREG E11's second reference grid, amended 2026-11-05): grad_last, grad_full,
grad_sparse19_sal and grad_sparse19_traj over Adam lr {1e-5, 1e-4} x steps {10, 30, 100, 300}, and interp at alpha
k/64 (k = 1..8) toward the first grid's theta_full; these five by default and only these. Runs go to
``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/e11g2/<seed>/``, never into the first grid's ``e11`` directories.

``--e12`` (PREREG E12, evidence-v1 instances of the cifar10_r18 conditions, the ImageNet-pretrained ResNet-18
student; implies ``--e10``'s limits and oracle): E10's eleven references, adapted to ResNet-18 (bn_adapt and tent
included). Runs go to ``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/e12/<seed>/``. ``--e12-grid2`` (with ``--e12``):
the second grid as ``--e11-grid2`` defines it, run label ``e12g2``; its grad_full and grad_last are F1's "full" and
"last layer" on E12. ``--t04`` and ``--t04-grid2`` (PREREG T04, in1k_t04 instances, the native ImageNet ResNet-18)
run exactly as ``--e12`` and ``--e12-grid2``, with run labels ``t04`` and ``t04g2``. ``--t04-edge`` (with ``--t04``;
PREREG T04's edge-extension amendment, main 9f0baa9) runs the extension grid of a second-grid method (lr divided by
10, interp alpha divided by 8) or of bn_adapt (prior strengths multiplied by 16), run label ``t04g3``; the pairs to
run are scripts/t04/edge_extension.py's.

Example::

    source scripts/env.sh
    .venv/bin/python scripts/run_fit_baselines.py --dataset fashion_mnist
    .venv/bin/python scripts/run_fit_baselines.py --instance <id> --methods grad_last ridge_last
    .venv/bin/python scripts/run_fit_baselines.py --instance <id> --joint --methods grad_last grad_full ridge_last
    .venv/bin/python scripts/run_fit_baselines.py --fewshot --dataset cifar10
    .venv/bin/python scripts/run_fit_baselines.py --e10 --instance <id>
    .venv/bin/python scripts/run_fit_baselines.py --e11 --instance <id>
    .venv/bin/python scripts/run_fit_baselines.py --e11 --e11-grid2 --instance <id>
    .venv/bin/python scripts/run_fit_baselines.py --e12 --instance <id>
    .venv/bin/python scripts/run_fit_baselines.py --e12 --e12-grid2 --instance <id>
    .venv/bin/python scripts/run_fit_baselines.py --t04 --instance <id>
    .venv/bin/python scripts/run_fit_baselines.py --t04 --t04-grid2 --instance <id>
"""

from __future__ import annotations

import argparse
import os
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
DEFAULT_METHODS = (
    "grad_last",
    "grad_sparse",
    "grad_full",
    "ridge_last",
    "ridge_last_free",
    "prior_shift",
    "tie_inputs",
    "input_stats",
)
"""Default methods on repair instances (the E1 set)."""
METHODS = (*DEFAULT_METHODS, "ridge_margin", "imprint", "imprint_retain", "grad_sparse19_sal", "grad_sparse19_traj",
           "interp", "bn_adapt", "tent", "tent_ln", "sar")
FEWSHOT_ONLY = ("imprint", "imprint_retain")
"""Methods that add classes and run with --fewshot only; imprint_retain belongs to the E2 supplement, not the gate."""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", action="append", help="instance id (repeatable); default: select from index")
    parser.add_argument("--split-role", choices=("dev", "main"), default="dev", help="index selection (default dev)")
    parser.add_argument("--dataset", help="restrict the index selection to one dataset")
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        help="default: the E1 set, or with --fewshot the four E2 references")
    parser.add_argument("--seed", type=int, default=0, help="run seed (torch seed for grad_*; directory name)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch and BLAS")
    parser.add_argument("--allow-main", action="store_true", help="permit main instances (only after the freeze)")
    parser.add_argument("--joint", action="store_true", help="E1: replay rows in the label fits, joint oracle")
    parser.add_argument("--e10", action="store_true",
                        help="E10: joint runs under the E10 program limits, the E10 references by default")
    parser.add_argument("--e11", action="store_true",
                        help="E11: --e10's limits and oracle on cifar10_vl7, the E11 references by default")
    parser.add_argument("--e11-grid2", action="store_true",
                        help="with --e11: PREREG E11's second grid (lr 1e-5/1e-4, interp alpha k/64), run label e11g2")
    parser.add_argument("--e12", action="store_true",
                        help="E12: --e10's limits and oracle on cifar10_r18, E10's references by default")
    parser.add_argument("--e12-grid2", action="store_true",
                        help="with --e12: the second grid on the E12 student, run label e12g2")
    parser.add_argument("--t04", action="store_true",
                        help="T04: --e10's limits and oracle on in1k_t04 (native ImageNet), E12's references by default")
    parser.add_argument("--t04-grid2", action="store_true",
                        help="with --t04: the second grid on the native network, run label t04g2")
    parser.add_argument("--t04-edge", action="store_true",
                        help="with --t04: the edge-extension grid (PREREG T04 amendment), run label t04g3")
    parser.add_argument("--fewshot", action="store_true",
                        help="E2: fewshot-v1 instances, new-class plus replay rows, fewshot joint oracle")
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
    from weightcode.baselines.fit.cli import (
        E10_METHODS,
        E10_ONLY,
        E10_REF_LABEL,
        E11_GRID2_LABEL,
        E11_GRID2_METHODS,
        E11_METHODS,
        E11_ONLY,
        E11_REF_LABEL,
        E12_GRID2_LABEL,
        E12_GRID2_METHODS,
        E12_METHODS,
        E12_REF_LABEL,
        EXIT_REFUSED,
        T04_EDGE_LABEL,
        T04_EDGE_METHODS,
        T04_GRID2_LABEL,
        T04_GRID2_METHODS,
        T04_METHODS,
        T04_REF_LABEL,
        FEWSHOT_CAPABLE,
        FEWSHOT_METHODS,
        FEWSHOT_REF_LABEL,
        FIT_METHODS,
        JOINT_REF_LABEL,
        SelectionError,
        load_checked_fewshot_instances,
        load_checked_instances,
        refusal_message,
        run_method,
        select_fewshot_ids,
        select_instance_ids,
        summary_line,
    )
    from weightcode.baselines.fit.runner import SplitRoleError, baseline_run_dir

    if set(METHODS) != set(FIT_METHODS):
        raise RuntimeError(f"script methods {METHODS} differ from the library's {FIT_METHODS}")
    if args.e11_grid2 and not args.e11:
        raise SystemExit("--e11-grid2 is PREREG E11's second grid; pass it with --e11")
    if args.e12_grid2 and not args.e12:
        raise SystemExit("--e12-grid2 is the E12 second grid; pass it with --e12")
    if args.t04_grid2 and not args.t04:
        raise SystemExit("--t04-grid2 is the T04 second grid; pass it with --t04")
    if args.t04_edge and (not args.t04 or args.t04_grid2):
        raise SystemExit("--t04-edge is the T04 edge extension; pass it with --t04 and without --t04-grid2")
    if int(args.e11) + int(args.e12) + int(args.t04) > 1:
        raise SystemExit("--e11, --e12 and --t04 do not combine")
    if args.e11 or args.e12 or args.t04:
        if args.e10:
            raise SystemExit("--e11, --e12 and --t04 imply the E10 limits; pass them alone")
        args.e10 = True
    if args.e10:
        if args.fewshot:
            raise SystemExit("--e10 and --fewshot do not combine")
        args.joint = True
    grid2 = args.e11_grid2 or args.e12_grid2 or args.t04_grid2
    if args.e11_grid2:
        default = E11_GRID2_METHODS
    elif args.e12_grid2:
        default = E12_GRID2_METHODS
    elif args.t04_grid2:
        default = T04_GRID2_METHODS
    elif args.t04_edge:
        default = None  # the pairs are edge_extension.py's; --methods is required
    elif args.e11:
        default = E11_METHODS
    elif args.e12:
        default = E12_METHODS
    elif args.t04:
        default = T04_METHODS
    else:
        default = FEWSHOT_METHODS if args.fewshot else (E10_METHODS if args.e10 else DEFAULT_METHODS)
    if default is None and not args.methods:
        raise SystemExit("--t04-edge needs --methods (the extended pairs of scripts/t04/edge_extension.py)")
    methods = list(args.methods) if args.methods else list(default)

    try:
        if args.fewshot:
            if args.joint or args.allow_main or args.split_role != "dev":
                raise SelectionError("--fewshot runs dev fewshot instances only; drop --joint, --allow-main, --split-role")
            unfit = [method for method in methods if method not in FEWSHOT_CAPABLE]
            if unfit:
                raise SelectionError(f"methods {unfit} do not run on fewshot instances")
            ids = select_fewshot_ids(args.instance, dataset=args.dataset)
            instances = load_checked_fewshot_instances(ids)
        else:
            fewshot_only = [method for method in methods if method in FEWSHOT_ONLY]
            if fewshot_only:
                raise SelectionError(f"{fewshot_only} add classes and run with --fewshot only")
            e10_only = [method for method in methods if method in E10_ONLY and (not args.e10 or args.e11)]
            if e10_only:
                raise SelectionError(f"{e10_only} are BatchNorm references of PREREG E10 and run with --e10 only")
            e11_only = [method for method in methods if method in E11_ONLY and not args.e11]
            if e11_only:
                raise SelectionError(f"{e11_only} are LayerNorm references of PREREG E11 and run with --e11 only")
            no_edge = [method for method in methods if args.t04_edge and method not in T04_EDGE_METHODS]
            if no_edge:
                raise SelectionError(f"{no_edge} have no edge-extension run; expected some of {T04_EDGE_METHODS}")
            no_grid2 = [method for method in methods if grid2 and method not in E11_GRID2_METHODS]
            if no_grid2:
                raise SelectionError(f"{no_grid2} have no second-grid run; expected some of {E11_GRID2_METHODS}")
            ids = select_instance_ids(args.instance, split_role=args.split_role, dataset=args.dataset)
            instances = load_checked_instances(ids, allow_main=args.allow_main)
    except (SplitRoleError, SelectionError) as error:
        print(refusal_message(error), file=sys.stderr)
        return EXIT_REFUSED
    print(f"instances: {[instance.instance_id for instance in instances]}", flush=True)
    if args.fewshot:
        ref_label: str | None = FEWSHOT_REF_LABEL
    else:
        ref_label = (E11_GRID2_LABEL if args.e11_grid2 else E12_GRID2_LABEL if args.e12_grid2 else
                     T04_GRID2_LABEL if args.t04_grid2 else T04_EDGE_LABEL if args.t04_edge else
                     E11_REF_LABEL if args.e11 else
                     E12_REF_LABEL if args.e12 else T04_REF_LABEL if args.t04 else
                     E10_REF_LABEL if args.e10 else (JOINT_REF_LABEL if args.joint else None))
    for instance in instances:
        for method in methods:
            run_dir = baseline_run_dir(instance.instance_id, method, args.seed, ref_label)
            record = run_method(
                instance,
                method,
                run_dir,
                run_seed=args.seed,
                allow_main=args.allow_main,
                joint=args.joint,
                fewshot=args.fewshot,
                e10=args.e10,
                e11=args.e11,
                grid2=grid2,
                e12=args.e12,
                t04=args.t04,
                edge=args.t04_edge,
            )
            print(summary_line(record), flush=True)
            print(f"  -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
