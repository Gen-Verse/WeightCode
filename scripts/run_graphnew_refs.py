"""Run the PREREG E7 references on graphnew-v1 instances.

References (``weightcode.baselines.fit.graphrefs``): grad_full and interp (edit limits ignored), grad_last,
grad_sparse19_sal, grad_sparse19_traj, ridge_margin, imprint and imprint_retain (E7 program limits enforced). Each is
fitted on the instance's 80 new-class nodes plus its 64 replay nodes, with a full-graph forward on the instance's
graph.npz, and scored by the graphnew joint oracle (V_check cut by class), 8 patch-check queries per run. Runs go to
``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/e7/<seed>/``; an existing run directory is never overwritten.

Instances: the given ``--instance`` ids, or every active instance in ``$WEIGHTCODE_PRIVATE/graphnew_instances/index.json``.
Graphnew instances are dev only; any other role is refused (exit status 2, nothing written). Dev readouts are a
separate step (``scripts/dev/e7_readouts.sh``)::

    source scripts/env.sh
    .venv/bin/python scripts/run_graphnew_refs.py --instance g06de3fa0 --threads 8
"""

from __future__ import annotations

import argparse
import os
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
METHODS = ("grad_full", "grad_last", "grad_sparse19_sal", "grad_sparse19_traj", "interp", "ridge_margin", "imprint",
           "imprint_retain")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", action="append", help="graphnew instance id g<8 hex> (repeatable)")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0, help="run seed (torch seed; directory name)")
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
    from weightcode.baselines.fit.cli import EXIT_REFUSED, summary_line
    from weightcode.baselines.fit.graphrefs import GRAPH_REF_METHODS, GRAPHNEW_REF_LABEL, run_graph_ref
    from weightcode.baselines.fit.runner import SplitRoleError, baseline_run_dir, check_split_role
    from weightcode.evalkit.graphnew import active_graphnew_ids, load_graphnew_instance

    if set(METHODS) != set(GRAPH_REF_METHODS):
        raise RuntimeError(f"script methods {METHODS} differ from the library's {GRAPH_REF_METHODS}")
    ids = list(dict.fromkeys(args.instance)) if args.instance else active_graphnew_ids()
    if not ids:
        print("refused: no active graphnew instance\nNothing was run.", file=sys.stderr)
        return EXIT_REFUSED
    instances = [load_graphnew_instance(instance_id) for instance_id in ids]
    try:
        for instance in instances:
            check_split_role(instance, allow_main=False)
    except SplitRoleError as error:
        print(f"refused: {error}\nNothing was run.", file=sys.stderr)
        return EXIT_REFUSED
    print(f"instances: {[instance.instance_id for instance in instances]}", flush=True)
    for instance in instances:
        for method in args.methods:
            run_dir = baseline_run_dir(instance.instance_id, method, args.seed, GRAPHNEW_REF_LABEL)
            record = run_graph_ref(instance, method, run_dir, run_seed=args.seed)
            print(summary_line(record), f"wall {record['timing']['wall_s']:.0f}s", flush=True)
            print(f"  -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
