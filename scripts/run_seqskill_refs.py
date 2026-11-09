"""Run the PREREG E5 gradient references on seqskill-v1 instances.

References (``weightcode.baselines.fit.seqgrad``): seq_grad_full (all parameters, edit limits ignored: the unconstrained
reference of the E5 bar), seq_grad_emb_head (the token embedding and the head) and seq_grad_new_row (the new
operator's embedding row only), the last two under the E5 program limits. Each is fitted on the instance's 64
new-operator rows plus its 64 replay rows and scored by the seqskill joint oracle (V_check parts 1 and 4).

Instances: the given ``--instance`` ids, or every active instance in ``$WEIGHTCODE_PRIVATE/seqskill_instances/index.json``
(optionally of one ``--condition``). Seqskill instances are dev only; any other role is refused (exit status 2,
nothing written). Each method on each instance gets its own counted oracle with 8 patch-check queries and writes
``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/e5/<seed>/``; an existing run directory is never overwritten.

Dev readouts are a separate step, the run's final and the kept rule v2 pick (S_old floor -40 of the 4000 old-operator
V_check rows)::

    source scripts/env.sh
    .venv/bin/python scripts/run_seqskill_refs.py --condition new0_substitution
    .venv/bin/python -m weightcode.evalkit.devreadout RUN.json [RUN.json ...] --out FILE
    .venv/bin/python -m weightcode.evalkit.devreadout RUN.json [RUN.json ...] --retention-floor -40 --out FILE
"""

from __future__ import annotations

import argparse
import os
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
METHODS = ("seq_grad_full", "seq_grad_emb_head", "seq_grad_new_row")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", action="append", help="seqskill instance id q<8 hex> (repeatable)")
    parser.add_argument("--condition", help="restrict the index selection to one condition, e.g. new0_substitution")
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
    from weightcode.baselines.fit.runner import SplitRoleError, baseline_run_dir, check_split_role
    from weightcode.baselines.fit.seqgrad import SEQ_GRAD_METHODS, SEQSKILL_REF_LABEL, run_seq_grad
    from weightcode.evalkit.seqskill import active_seqskill_ids, load_seqskill_instance

    if set(METHODS) != set(SEQ_GRAD_METHODS):
        raise RuntimeError(f"script methods {METHODS} differ from the library's {SEQ_GRAD_METHODS}")
    ids = list(dict.fromkeys(args.instance)) if args.instance else active_seqskill_ids(args.condition)
    if not ids:
        print(f"refused: no active seqskill instance of condition {args.condition!r}\nNothing was run.", file=sys.stderr)
        return EXIT_REFUSED
    instances = [load_seqskill_instance(instance_id) for instance_id in ids]
    try:
        for instance in instances:
            check_split_role(instance, allow_main=False)
    except SplitRoleError as error:
        print(f"refused: {error}\nNothing was run.", file=sys.stderr)
        return EXIT_REFUSED
    print(f"instances: {[instance.instance_id for instance in instances]}", flush=True)
    for instance in instances:
        for method in args.methods:
            run_dir = baseline_run_dir(instance.instance_id, method, args.seed, SEQSKILL_REF_LABEL)
            record = run_seq_grad(instance, method, run_dir, run_seed=args.seed)
            print(summary_line(record), flush=True)
            print(f"  -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
