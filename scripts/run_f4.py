"""Run one PREREG F4 arm (``weightcode.baselines.fit.f4``): sequential fine-tuning over a stream, no LLM call.

The arm runs round after round on its own current student through the control (b) stream
(``weightcode.baselines.fit.checked_reference_stream``): the rows, the multi-part check, the kept rule and the readouts are
control (b)'s. Records go to ``--out`` (default ``$WEIGHTCODE_RUNS/f4``) and are never overwritten.

Example (E11 stream, both conditions in order)::

    python -B scripts/run_f4.py --arm f4_lwf --instances i00802425 i3eb3fc78 --kept-rule loop-retain-v2 --allow-e11
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
ARMS = ("f4_ft_replay", "f4_lora", "f4_er", "f4_lwf")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--instances", nargs="+", required=True, help="the stream's instance ids, in round order")
    parser.add_argument("--out", type=Path, help="directory for the record (default $WEIGHTCODE_RUNS/f4)")
    parser.add_argument("--seed", type=int, default=0, help="run seed (LoRA's A draw)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch and BLAS")
    parser.add_argument("--kept-rule", default="loop-retain-v2", choices=("loop-retain-v1", "loop-retain-v2"))
    parser.add_argument("--allow-e11", action="store_true", help="the stream is an E11 (ViT-Lite) stream")
    return parser.parse_args(argv)


def _points(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:+.2f}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(args.threads)
    sys.dont_write_bytecode = True
    import torch

    torch.set_num_threads(args.threads)
    from weightcode.baselines.fit import f4
    from weightcode.baselines.fit.checked_reference_stream import CheckedReferenceError, run_checked_reference_stream
    from weightcode.common.paths import runs_dir

    started = time.perf_counter()
    out = args.out if args.out is not None else runs_dir() / "f4"
    try:
        path = run_checked_reference_stream(args.arm, args.instances, out, run_seed=args.seed, kept_rule=args.kept_rule,
                                            make_generator=f4.make_generator, allow_e11=args.allow_e11)
    except (CheckedReferenceError, f4.F4Error) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    record = json.loads(path.read_text())
    for row in record["rounds"]:
        kept = row.get("kept_check") or {}
        print(f"round {row['round']} {row['condition']}: {row['status']} kept {row['kept']} {row['kept_config']} "
              f"score {kept.get('score')} (checks {row['queries_used']}, eligible {row['n_eligible']})", flush=True)
    summary = record["summary"]
    gains = {name: _points(item["gain_after_last_round"]) for name, item in summary["conditions"].items()}
    print(f"{args.arm}: mean gain {_points(summary['mean_gain_after_last_round'])} points, clean "
          f"{_points(summary['clean_delta_acc_mean_after_last_round'])}, max forgetting "
          f"{_points(summary['max_forgetting'])}, rounds ok {summary['n_rounds_ok']}; gains {gains}; "
          f"wall {time.perf_counter() - started:.0f} s", flush=True)
    print(f"  -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
