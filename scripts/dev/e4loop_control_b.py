"""Control (b) of PREREG "E4 with the teacher in the loop": the equal-feedback label-fit references (dev only).

Each method (grad_last, ridge_last, grad_full) is one checked reference stream
(``weightcode.baselines.fit.checked_reference_stream``): every round re-selects the fit over its E1 configurations by the
loop's check and kept rule loop-retain-v1, on its own current student, with variant (b)'s rows. No LLM call.

Before the stream, the loop records given with ``--loop-record`` are rebuilt round by round on their own students
(``loop_round_rows``): the run stops unless every rebuilt experience hashes to what the loop recorded. The first one
is then the loop the stream's rows are compared with, round by round (``compare_rows``). Records go to ``--out``
(default ``$WEIGHTCODE_RUNS/loop_control_b``) and are never overwritten.

Example (seed 100, the E4 order: noise, contrast, blur, gamma, colormix)::

    source scripts/env.sh
    .venv/bin/python scripts/dev/e4loop_control_b.py --method grad_last --threads 6 \\
        --loop-record $WEIGHTCODE_RUNS/loop_streams/loop_teacher_ccc1fe9fe3ac_d5bef5b0f9b3.json \\
        $WEIGHTCODE_RUNS/loop_streams/loop_checked_selection_ded0afcd73b5_d5bef5b0f9b3.json

PREREG E10's control (b) is the same command on the E10 stream (seed 100, conditions in the E10 order), with the
methods grad_last, ridge_last, grad_full, bn_adapt and tent; on E10 instances every candidate is applied under the E10
program limits, and bn_adapt and tent fit the round's 64 condition inputs only::

    .venv/bin/python scripts/dev/e4loop_control_b.py --method bn_adapt --threads 6 \\
        --instances $(.venv/bin/python -m weightcode.harness.e10 ids) --out $WEIGHTCODE_RUNS/e10/control_b \\
        --loop-record <E10 teacher loop record> <E10 control (a) record>

``--kept-rule loop-retain-v2`` re-selects each round under the stream's clean budget instead
(``weightcode.harness.loop.loop_kept_rule_v2``: -60 of 4000 clean V_check rows over the stream, less this stream's own kept
score_old of the earlier rounds); give it the loop records of the v2 teacher stream and its v2 control (a).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
E4_SEED100 = ("ibb1f3fc6", "i7346ae21", "if9258f2a", "i4f9f5974", "i984a5043")
EXIT_REFUSED = 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", required=True,
                        help="grad_last, ridge_last or grad_full; on E10 instances also bn_adapt or tent")
    parser.add_argument("--instances", nargs="+", default=list(E4_SEED100),
                        help="one dev instance per round, in round order (default: the seed-100 E4 stream)")
    parser.add_argument("--loop-record", nargs="*", type=Path, default=[],
                        help="loop stream records to rebuild and check first; the first is compared row by row")
    parser.add_argument("--out", type=Path, help="directory for the record (default $WEIGHTCODE_RUNS/loop_control_b)")
    parser.add_argument("--seed", type=int, default=0, help="run seed of the fits (E1's)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch and BLAS")
    parser.add_argument("--kept-rule", default="loop-retain-v1", choices=("loop-retain-v1", "loop-retain-v2"),
                        help="the loop's kept rule the rounds re-select by (weightcode.harness.loop.KEPT_RULES)")
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
    from weightcode.baselines.fit.checked_reference_stream import METHODS, loop_round_rows, run_checked_reference_stream
    from weightcode.common.paths import runs_dir

    if args.method not in METHODS:
        print(f"refused: unknown method {args.method!r}; methods {METHODS}", file=sys.stderr)
        return EXIT_REFUSED
    started = time.perf_counter()
    checked = []
    for record in args.loop_record:
        rows = loop_round_rows(record)
        checked.append(rows)
        print(f"loop rows rebuilt and matched: {record} "
              f"({[row['content_sha256'][:12] for row in rows]})", flush=True)
    out = args.out if args.out is not None else runs_dir() / "loop_control_b"
    path = run_checked_reference_stream(args.method, args.instances, out,
                                        loop_rows=checked[0] if checked else None,
                                        loop_record=args.loop_record[0] if args.loop_record else None,
                                        run_seed=args.seed, kept_rule=args.kept_rule)
    record = json.loads(path.read_text())
    for row in record["rounds"]:
        kept = row.get("kept_check") or {}
        print(f"round {row['round']} {row['condition']}: {row['status']} kept {row['kept']} {row['kept_config']} "
              f"score {kept.get('score')} (checks {row['queries_used']}, eligible {row['n_eligible']})", flush=True)
    summary = record["summary"]
    gains = {name: _points(item["gain_after_last_round"]) for name, item in summary["conditions"].items()}
    print(f"{args.method}: mean gain {_points(summary['mean_gain_after_last_round'])} points, clean "
          f"{_points(summary['clean_delta_acc_mean_after_last_round'])}, max forgetting "
          f"{_points(summary['max_forgetting'])}, rounds ok {summary['n_rounds_ok']}; gains {gains}; "
          f"wall {time.perf_counter() - started:.0f} s", flush=True)
    if "kept_rule" in record:
        rule = record["kept_rule"]
        print(f"{rule['name']}: clean floors {rule['s_old_floor']}, kept score_old "
              f"{[item['s_old'] for item in rule['clean_used']]} (total {rule['clean_used_total']} of "
              f"{rule['clean_budget_floor']})", flush=True)
    print(f"  -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
