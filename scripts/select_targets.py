"""Apply the pre-registered target rule (targets-v0, see weightcode.evalkit.targets) to one dataset's selected models.

Target pairs feed only evidence-v0 instances, which evidence-v1 superseded (user decision 2026-11-03). Writes
$WEIGHTCODE_PRIVATE/<dataset>/models/<model_id>/targets.json for every selected dev and/or main model. An identical
existing file is kept; a different one is refused. The seed defaults are the development plan (main 0 and 1, dev
100 and 101). Run after `source scripts/env.sh`.

    .venv/bin/python scripts/select_targets.py --dataset fashion_mnist --split-role dev
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from weightcode.evalkit.targets import (
    DEFAULT_DEV_SEEDS,
    DEFAULT_MAIN_SEEDS,
    SPLIT_ROLES,
    SeedSelection,
    parse_seeds,
    run_target_selection,
)


def _seeds(text: str) -> tuple[int, ...]:
    try:
        return parse_seeds(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run target selection and print one line per model and chosen pair."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset directory name under $WEIGHTCODE_PRIVATE")
    parser.add_argument("--split-role", choices=("dev", "main", "all"), default="all")
    parser.add_argument("--main-seeds", type=_seeds, default=",".join(map(str, DEFAULT_MAIN_SEEDS)))
    parser.add_argument("--dev-seeds", type=_seeds, default=",".join(map(str, DEFAULT_DEV_SEEDS)))
    parser.add_argument("--batch-size", type=int, default=1024, help="forward batch size")
    args = parser.parse_args(argv)
    selection = SeedSelection(main=args.main_seeds, dev=args.dev_seeds)
    roles = SPLIT_ROLES if args.split_role == "all" else (args.split_role,)
    print(f"{args.dataset}: main seeds {list(selection.main)}, dev seeds {list(selection.dev)}")
    outcomes = run_target_selection(args.dataset, selection=selection, split_roles=roles, batch_size=args.batch_size)
    for outcome in outcomes:
        state = "written" if outcome.written else "unchanged"
        print(f"{outcome.split_role} {outcome.model_id} seed={outcome.seed} {state} shortfall={outcome.shortfall}")
        for pair in outcome.chosen:
            print(
                f"  rank {pair['rank']}: ({pair['a']} {pair['a_name']} -> {pair['b']} {pair['b_name']}) "
                f"V_select {pair['vselect_count']}/{pair['vselect_n_a']}, V_evidence {pair['vevidence_count']}, "
                f"V_check {pair['vcheck_count']}"
            )
    print(f"{len(outcomes)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
