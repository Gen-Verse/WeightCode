"""Build (or reuse) the instances of rules scanskill-v1 (E13, SCAN) and policyskill-v1 (E14, BabyAI).

A thin command-line entry over the builders that wrote the instances of record; it adds no logic of its own
(docs/maintenance/CODE_AUDIT.md section 4). The vision rules (evidence-v0, evidence-v1) stay in build_instances.py.

    scanskill-v1    weightcode.evalkit.scanskill.build_scan_instances(condition, seeds=..., batch_size=...)
    policyskill-v1  weightcode.evalkit.e14.build_e14_instance(model_id)

The builds of record (2026-11-06; read from $WEIGHTCODE_PRIVATE/scan_instances/index.json, e14_instances/index.json and
the instances' private.json):

    .venv/bin/python scripts/build_rule_instances.py --rule scanskill-v1 --condition addprim_jump   # ncc5c260f, ma78dbcbb
    .venv/bin/python scripts/build_rule_instances.py --rule scanskill-v1 --condition around_right   # n8826c49e, m896cb576
    .venv/bin/python scripts/build_rule_instances.py --rule policyskill-v1 --model-id mdb8464d0     # p6ce8ad67

Each SCAN condition's model index has one base student (seed 100), so the default seed selection (every base student)
is the one of record. The forward batch size is not recorded in the instances; the builder's default (1024) is used.
An active instance with the same key is verified and reused; a rebuild that differs from it is refused by the builder
and nothing is overwritten. Run after `source scripts/env.sh`. Prints one JSON line per instance.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

SCAN_RULE = "scanskill-v1"  # weightcode.evalkit.scanskill.SCAN_RULE; the test holds the two equal
POLICY_RULE = "policyskill-v1"  # weightcode.evalkit.e14.POLICYSKILL_RULE, likewise
SCAN_DEFAULT_BATCH = 1024  # weightcode.evalkit.scanskill.EVALUATION_BATCH, likewise


def _seeds(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"--seeds takes comma-separated integers: {text!r}") from error


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Call the rule's builder with the parsed arguments; the modules are imported here, so --help needs neither."""
    if args.rule == SCAN_RULE:
        from weightcode.evalkit import scanskill as K

        return K.build_scan_instances(args.condition, seeds=args.seeds, batch_size=args.batch_size)
    from weightcode.evalkit import e14 as E

    return [E.build_e14_instance(args.model_id)]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, check that each option belongs to the chosen rule, build or reuse, print the rows."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rule", required=True, choices=(SCAN_RULE, POLICY_RULE))
    parser.add_argument("--condition", help=f"{SCAN_RULE}: the SCAN condition (addprim_jump or around_right)")
    parser.add_argument("--seeds", type=_seeds, help=f"{SCAN_RULE}: only these base seeds (default: every base student)")
    parser.add_argument("--batch-size", type=int, help=f"{SCAN_RULE}: forward batch size (default {SCAN_DEFAULT_BATCH})")
    parser.add_argument("--model-id", help=f"{POLICY_RULE}: the gated base student of the E14 models index")
    args = parser.parse_args(argv)
    if args.rule == SCAN_RULE:
        if not args.condition:
            parser.error(f"{SCAN_RULE} needs --condition")
        if args.model_id is not None:
            parser.error(f"--model-id applies to {POLICY_RULE} only")
        if args.batch_size is None:
            args.batch_size = SCAN_DEFAULT_BATCH
        elif args.batch_size < 1:
            parser.error("--batch-size must be at least 1")
    else:
        if not args.model_id:
            parser.error(f"{POLICY_RULE} needs --model-id")
        if args.condition is not None or args.seeds is not None or args.batch_size is not None:
            parser.error(f"--condition, --seeds and --batch-size apply to {SCAN_RULE} only")
    rows = build(args)
    for row in rows:
        print(json.dumps(row, sort_keys=True, default=str))
    print(f"{len(rows)} {args.rule} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
