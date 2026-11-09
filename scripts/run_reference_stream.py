"""Run the E4 sequential label-fit references on one stream of dev instances (PREREG E4).

Each (method, variant) pair is one reference stream (``weightcode.baselines.fit.reference_stream``). The round's
configuration is frozen from the method's E1 joint run: on that round's own instance, or with ``--frozen-from`` on the
instance of the same condition among those ids (seeds 101 to 105 take seed 100's choices). The rounds, experience,
readouts and summary are those of the program stream. Records go to ``--out`` (default
``$WEIGHTCODE_RUNS/reference_streams``) and are never overwritten.

Examples (the E4 stream order: noise, contrast, blur, gamma, colormix)::

    source scripts/env.sh
    .venv/bin/python scripts/run_reference_stream.py --instances ibb1f3fc6 i7346ae21 if9258f2a i4f9f5974 i984a5043
    .venv/bin/python scripts/run_reference_stream.py --instances ia83ba9b5 i3c156868 i6f53c57a ic1ea9383 i0f355ea0 \\
        --frozen-from ibb1f3fc6 i7346ae21 if9258f2a i4f9f5974 i984a5043
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
EXIT_REFUSED = 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instances", nargs="+", required=True, help="one dev instance per round, in round order")
    parser.add_argument("--frozen-from", nargs="+", metavar="ID",
                        help="take each round's configuration from the E1 joint run on the instance of its condition "
                             "among these (default: the round's own instance)")
    parser.add_argument("--methods", nargs="+", default=["grad_last", "ridge_last", "grad_full"])
    parser.add_argument("--variants", nargs="+", default=["replay", "cumulative"])
    parser.add_argument("--out", type=Path, help="directory for the records (default $WEIGHTCODE_RUNS/reference_streams)")
    parser.add_argument("--seed", type=int, default=0, help="run seed of the E1 joint runs and of the fits")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch and BLAS")
    return parser.parse_args(argv)


def _points(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.2f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
        os.environ[variable] = str(args.threads)
    sys.dont_write_bytecode = True  # importing an instance's forward.py must not write __pycache__ into $WEIGHTCODE_PRIVATE
    import torch

    torch.set_num_threads(args.threads)
    from weightcode.baselines.fit.reference_stream import METHODS, VARIANTS, run_reference_stream
    from weightcode.common.paths import runs_dir

    unknown = sorted(set(args.methods) - set(METHODS)) + sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        print(f"refused: unknown methods or variants {unknown}; methods {METHODS}, variants {VARIANTS}",
              file=sys.stderr)
        return EXIT_REFUSED
    out = args.out if args.out is not None else runs_dir() / "reference_streams"
    for method in args.methods:
        for variant in args.variants:
            path = run_reference_stream(method, variant, args.instances, out, frozen_from=args.frozen_from,
                                        run_seed=args.seed)
            summary = json.loads(path.read_text())["summary"]
            gains = {name: _points(row["gain_after_last_round"]) for name, row in summary["conditions"].items()}
            print(f"{method} {variant}: mean gain {_points(summary['mean_gain_after_last_round'])} points, clean "
                  f"{_points(summary['clean_delta_acc_mean_after_last_round'])}, max forgetting "
                  f"{_points(summary['max_forgetting'])}, rounds ok {summary['n_rounds_ok']}; gains {gains}",
                  flush=True)
            print(f"  -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
