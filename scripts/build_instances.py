"""Build the repair instances of one dataset (see weightcode.evalkit.instances).

Rule evidence-v1 (default): one instance per selected model for a deployment condition (default "deploy").
Rule evidence-v0: one instance per chosen target pair from the models' targets.json (run select_targets.py first).

Writes $WEIGHTCODE_PRIVATE/instances/<instance_id>/ and $WEIGHTCODE_PRIVATE/instances/index.json. Existing active instances with
the same key are verified and reused; a difference is refused. --supersede-rule first retires every active instance
of the dataset built under that rule, for evidence-v1 only those of the --condition being built (rows move to
"superseded" in index.json, directories stay in place, reason logged to changes.jsonl). Dev models are built before
main models. Run after `source scripts/env.sh`.

    .venv/bin/python scripts/build_instances.py --dataset colored_mnist \\
        --supersede-rule evidence-v0 --reason "target pairs replaced by evidence-v1 (user decision 2026-11-03)"
    .venv/bin/python scripts/build_instances.py --dataset cifar10 --condition contrast_s3 --split-role dev

Seeds and decoys are inputs; the defaults are the development plan (main seeds 0 and 1, dev seeds 100 and 101,
decoys 0 <-> 1 and 100 <-> 101). Five main seeds would be --main-seeds 0,1,2,3,4 with
--decoys 0:1,1:0,2:3,3:4,4:2,100:101,101:100.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import TypeVar

from weightcode.evalkit.evidence import EVIDENCE_RULE, EVIDENCE_RULE_V1
from weightcode.evalkit.instances import (
    DEFAULT_DECOYS,
    active_instance_ids,
    build_dataset_instances,
    build_deployment_instances,
    format_decoys,
    parse_decoys,
    supersede_instances,
)
from weightcode.evalkit.targets import (
    DEFAULT_DEV_SEEDS,
    DEFAULT_MAIN_SEEDS,
    DEPLOY_CONDITION,
    SPLIT_ROLES,
    SeedSelection,
    parse_seeds,
)

T = TypeVar("T")


def _argument(parse: Callable[[str], T]) -> Callable[[str], T]:
    def convert(text: str) -> T:
        try:
            return parse(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    return convert


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, supersede if asked, build or reuse instances and print one line per instance."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset directory name under $WEIGHTCODE_PRIVATE")
    parser.add_argument("--evidence-rule", choices=(EVIDENCE_RULE_V1, EVIDENCE_RULE), default=EVIDENCE_RULE_V1)
    parser.add_argument("--condition", help=f"deployment condition, evidence-v1 only (default {DEPLOY_CONDITION})")
    parser.add_argument("--split-role", choices=("dev", "main", "all"), default="all")
    parser.add_argument("--main-seeds", type=_argument(parse_seeds), default=",".join(map(str, DEFAULT_MAIN_SEEDS)))
    parser.add_argument("--dev-seeds", type=_argument(parse_seeds), default=",".join(map(str, DEFAULT_DEV_SEEDS)))
    parser.add_argument("--decoys", type=_argument(parse_decoys), default=format_decoys(DEFAULT_DECOYS),
                        help="seed:decoy_seed pairs covering exactly the selected seeds")
    parser.add_argument("--supersede-rule", choices=(EVIDENCE_RULE, EVIDENCE_RULE_V1),
                        help="first supersede every active instance of --dataset built under this rule "
                             "(evidence-v1: of --condition only)")
    parser.add_argument("--reason", help="reason logged with --supersede-rule (required with it)")
    parser.add_argument("--batch-size", type=int, default=1024, help="forward batch size")
    args = parser.parse_args(argv)
    if args.condition is not None and args.evidence_rule != EVIDENCE_RULE_V1:
        parser.error("--condition applies to evidence-v1 only")
    if args.supersede_rule is not None and not (args.reason and args.reason.strip()):
        parser.error("--supersede-rule needs --reason")
    selection = SeedSelection(main=args.main_seeds, dev=args.dev_seeds)
    roles = SPLIT_ROLES if args.split_role == "all" else (args.split_role,)
    condition = args.condition or DEPLOY_CONDITION
    print(
        f"{args.dataset}: rule {args.evidence_rule}"
        + (f", condition {condition}" if args.evidence_rule == EVIDENCE_RULE_V1 else "")
        + f", main seeds {list(selection.main)}, dev seeds {list(selection.dev)}, decoys {format_decoys(args.decoys)}"
    )
    if args.supersede_rule is not None:
        scope = condition if args.supersede_rule == EVIDENCE_RULE_V1 else None
        ids = active_instance_ids(dataset=args.dataset, evidence_rule=args.supersede_rule, condition=scope)
        for row in supersede_instances(ids, args.reason) if ids else []:
            print(f"{row['instance_id']} superseded {row['split_role']} model={row['model_id']} seed={row['seed']}")
        print(f"{len(ids)} {args.supersede_rule} instances superseded")
    if args.evidence_rule == EVIDENCE_RULE_V1:
        outcomes = build_deployment_instances(
            args.dataset, condition, selection=selection, decoys=args.decoys, split_roles=roles,
            batch_size=args.batch_size,
        )
    else:
        outcomes = build_dataset_instances(
            args.dataset, selection=selection, decoys=args.decoys, split_roles=roles, batch_size=args.batch_size
        )
    for o in outcomes:
        if o.target is not None and o.target_names is not None:
            what = (f"target=({o.target[0]} {o.target_names[0]} -> {o.target[1]} {o.target_names[1]}) "
                    f"V_select {o.counts['n_class_a_predicted_b']}/{o.counts['n_class_a']}")
        else:
            what = (f"condition={o.condition} base errors V_evidence {o.counts['n_base_errors_vevidence']}/"
                    f"{o.counts['n_vevidence']}, V_check {o.counts['n_base_errors_vcheck']}/{o.counts['n_vcheck']}")
        print(f"{o.instance_id} {o.status} {o.split_role} model={o.model_id} seed={o.seed} "
              f"decoy=seed {o.decoy_seed} ({o.decoy_model_id}) {what}")
        for file_name, tokens in o.visible_file_findings.items():
            if tokens:
                print(f"  editor-visible {file_name} mentions {tokens}")
    print(f"{len(outcomes)} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
