"""Run one editor session, or list the instances a session may run on (docs/HARNESS_SPEC.md section 5).

Examples::

    source scripts/env.sh
    .venv/bin/python scripts/run_editor_session.py --list --dataset colored_mnist --seed 101
    .venv/bin/python scripts/run_editor_session.py --instance <id> --arm M2 --repeat 0 --effort max \\
        --wall-timeout-min 90
    .venv/bin/python scripts/run_editor_session.py --instance <id> --arm DECOY --repeat 0 --effort max \\
        --wall-timeout-min 90 --dry-run
    .venv/bin/python scripts/run_editor_session.py --instance <seqskill id> --arm LEARN --repeat 0 --effort max \\
        --wall-timeout-min 90 --retention-floor-frac 0.005 --seqskill-variant rule_only --lift-rel-frob --dry-run
    .venv/bin/python scripts/run_editor_session.py --instance <graphnew id> --arm LEARN --repeat 0 --effort max \
        --wall-timeout-min 90 --retention-floor-rows <PREREG E7 floor> --retention-floor-source floor.json --dry-run
    .venv/bin/python scripts/run_editor_session.py --instance <policyskill id> --arm LEARN --repeat 0 --effort max \\
        --wall-timeout-min 120 --mcp-tool-timeout-ms 1200000 --retention-floor-rows -3 \\
        --retention-floor-source floor.json --dry-run
    .venv/bin/python scripts/run_editor_session.py --instance <E10 id> --arm LEARN --repeat 0 --effort max \\
        --wall-timeout-min 90 --retention-floor-frac 0.005 --dry-run
    .venv/bin/python scripts/run_editor_session.py --instance $(.venv/bin/python -m weightcode.harness.e11 ids \\
        --condition <condition>) --arm LEARN --repeat 0 --effort max --wall-timeout-min 90 \\
        --retention-floor-frac 0.005 --dry-run
    .venv/bin/python scripts/run_editor_session.py --instance $(.venv/bin/python -m weightcode.harness.e11 ids \\
        --condition <condition>) --arm LEARN_BLIND --repeat 0 --effort max --wall-timeout-min 120 \\
        --retention-floor-frac 0.005 --dry-run
    .venv/bin/python scripts/run_editor_session.py --instance $(.venv/bin/python -m weightcode.harness.e11 ids \\
        --condition <condition>) --arm LEARN --repeat 0 --effort max --wall-timeout-min 120 \\
        --retention-floor-frac 0.005 --max-checks 2 --dry-run
    .venv/bin/python scripts/run_editor_session.py --instance $(.venv/bin/python -m weightcode.harness.e11 ids \\
        --condition gaussian_noise_s5) --arm LEARN --repeat 0 --effort max --wall-timeout-min 120 \\
        --retention-floor-frac 0.005 --e11-variant rule --dry-run
    .venv/bin/python scripts/run_editor_session.py --instance $(.venv/bin/python -m weightcode.harness.e11 ids \\
        --condition gaussian_noise_s5) --arm LEARN --repeat 0 --effort max --wall-timeout-min 120 \\
        --retention-floor-frac 0.005 --e11-variant unlabelled --dry-run
    .venv/bin/python scripts/run_editor_session.py --instance $(.venv/bin/python -m weightcode.evalkit.scanskill ids \\
        --condition <condition>) --arm LEARN --repeat 0 --effort max --wall-timeout-min 120 \\
        --retention-floor-frac 0.005 --seqskill-variant scan_rule --lift-rel-frob --dry-run

The last line is PREREG E13's B session on a scanskill-v1 instance (condition addprim_jump or around_right); its
session root and code path must not contain the leak word of the variant, which the isolation self-test searches the
mount table for. The second and third of the E11 lines are PREREG A4a (the blind arm) and A4b (the B arm with at most 2 checks), the
fourth PREREG K1 (the E11 rule view: the condition's rule as a formula, no image of the condition) and the last the
information ladder's U (E11 without y_new); ``--e11-variant named`` (K0b: no data, the capability named) and
``named_nocheck`` (K0a: K0b without patch_check, one check after the session) run the same way on the same instance.

A dry run assembles the trial, renders and leak-scans the prompts and runs the isolation self-test, without starting
Claude Code; it writes under ``$WEIGHTCODE_RUNS/dry_runs``. Instances are found by dataset, seed and split role
(``--list``), never by a remembered id. Main instances are refused unless ``--allow-main`` is given, which stays
unused until the PREREG is frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from weightcode.harness.arms import ARMS
from weightcode.harness.e11 import VARIANTS as E11_VARIANTS
from weightcode.harness.instance import list_instances
from weightcode.harness.seqskill import VARIANTS as SEQSKILL_VARIANTS
from weightcode.harness.runner import (
    DEFAULT_CELL_TIMEOUT_S,
    DEFAULT_MAX_CELLS,
    DEFAULT_MODEL,
    DEFAULT_THREADS,
    EFFORTS,
    SYSTEM_PROMPT_MODES,
    run_session,
)

LIST_FIELDS = ("instance_id", "dataset", "condition", "split_role", "seed", "model_id", "n_base_errors_vcheck")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="list active evidence-v1 instances and exit")
    parser.add_argument("--dataset", help="--list: only this dataset")
    parser.add_argument("--seed", type=int, help="--list: only this model seed")
    parser.add_argument("--split-role", default="dev", choices=("dev", "main", "all"), help="--list: role (dev)")
    parser.add_argument("--instance", help="instance id")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", choices=EFFORTS)
    parser.add_argument("--wall-timeout-min", type=float, help="wall-clock limit of the editor, in minutes")
    parser.add_argument("--allow-main", action="store_true", help="allow a main instance (only after PREREG freeze)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="BLAS threads of the server and sandbox")
    parser.add_argument("--cell-timeout-s", type=float, default=DEFAULT_CELL_TIMEOUT_S)
    parser.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS)
    parser.add_argument("--mcp-tool-timeout-ms", type=int, default=None,
                        help="Claude Code's MCP tool-call timeout (default: the instance's, 600000; E10 1200000)")
    parser.add_argument("--mcp-connect-timeout-ms", type=int, default=None,
                        help="Claude Code's MCP server startup timeout (default: the instance's, 120000; E10 900000)")
    parser.add_argument("--system-prompt-mode", choices=SYSTEM_PROMPT_MODES, default="replace")
    parser.add_argument("--task-prompt-file", type=Path, help="replace the rendered task prompt (smoke tests)")
    parser.add_argument("--dry-run", action="store_true", help="prepare and self-test only; do not start Claude Code")
    billing = parser.add_mutually_exclusive_group()
    billing.add_argument("--api-key-file", type=Path, default=None,
                         help="bill the editor to the Anthropic API key in this file (mode 600, one line); "
                              "session.json records the path, never the key. Default: the account's Claude Code login")
    billing.add_argument("--oauth-token-file", type=Path, default=None,
                         help="bill the editor to the subscription token in this file (`claude setup-token`; mode "
                              "600, one line); session.json records the path, never the token")
    parser.add_argument("--retention-floor-frac", type=float, default=None,
                        help="program arm, dev only: keep only checked programs with score_old >= -round(frac x the "
                             "rows behind score_old) (kept rule joint-retain-v1, weightcode.harness.retention); "
                             "default: the v1 rule")
    parser.add_argument("--library-sessions", nargs="+", default=None, metavar="SESSION_ID",
                        help="program arm on an evidence-v1 instance, dev only (LEARN v3): put the kept programs "
                             "of these earlier sessions and their check scores into the trial "
                             "(weightcode.harness.library); default: no library")
    parser.add_argument("--retention-floor-rows", type=int, default=None, metavar="S_OLD",
                        help="program arm (required on a graphnew-v1 instance, PREREG E7, and on a policyskill-v1 "
                             "instance, PREREG E14, in episodes): keep only checked "
                             "programs with score_old >= S_OLD, an int <= 0 in rows of the check set behind "
                             "score_old (kept rule joint-retain-v1, floor_kind rows, weightcode.harness.retention); "
                             "excludes --retention-floor-frac")
    parser.add_argument("--retention-floor-source", type=Path, default=None, metavar="PATH",
                        help="with --retention-floor-rows: a JSON object saying where the floor comes from (for E7 "
                             "the formula's inputs), copied into session.json kept_rule.floor_source; one with "
                             "\"placeholder\": true marks a stand-in floor and is accepted with --dry-run only")
    parser.add_argument("--seqskill-variant", choices=SEQSKILL_VARIANTS, default=None,
                        help="seqskill-v1 instance only: 'standard' is the PREREG E5 view; 'rule_only' (PREREG E8) "
                             "states the new operator's rule in model_spec.json and shows no row of it in the "
                             "experience (weightcode.harness.seqskill); 'scan_rule' (PREREG E13) is the view of a "
                             "scanskill-v1 instance (an n id) and of no other, and needs --lift-rel-frob; default: "
                             "standard")
    parser.add_argument("--lift-rel-frob", action="store_true",
                        help="seqskill-v1 instance in the LEARN arm only (PREREG E8 amendment): no per-tensor rel-Frob "
                             "limit (max_rel_frob null) in task.json, the prompts, every check, the final and the "
                             "readout; every other limit unchanged; default: the limit applies")
    parser.add_argument("--max-checks", type=int, default=None, metavar="N",
                        help="an arm with patch_check (PREREG A4b): at most N checks in the session, N >= 1 and at "
                             "most the instance's patch-check budget, in task.json, the prompt and the check ledger; "
                             "default: the instance's budget")
    parser.add_argument("--blind-show-predictions", action="store_true",
                        help="LEARN_BLIND only (PREREG A4a): keep the student's predictions pred_new and pred_old in "
                             "the trial's experience.npz; default: they are left out (a program run gets them either "
                             "way)")
    parser.add_argument("--e11-variant", choices=E11_VARIANTS, default=None,
                        help="E11 instance in the LEARN arm only: 'standard' is the PREREG E11 view; 'rule' (PREREG "
                             "K1) states the condition's rule as a formula in the task and shows no image of the "
                             "condition, only the clean replay rows; the information ladder on its one instance: "
                             "'unlabelled' (U) withholds y_new, 'named' (K0b) gives no data and names the capability, "
                             "'named_nocheck' (K0a) is K0b without patch_check, checked once after the session "
                             "(weightcode.harness.e11); default: standard")
    args = parser.parse_args(argv)
    if not args.list:
        missing = [flag for flag, value in (("--instance", args.instance), ("--arm", args.arm),
                                            ("--repeat", args.repeat), ("--effort", args.effort),
                                            ("--wall-timeout-min", args.wall_timeout_min)) if value is None]
        if missing:
            parser.error(f"missing {', '.join(missing)}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        role = None if args.split_role == "all" else args.split_role
        for row in list_instances(split_role=role, dataset=args.dataset, seed=args.seed):
            print(json.dumps({key: row.get(key) for key in LIST_FIELDS}, sort_keys=True))
        return 0
    override = None if args.task_prompt_file is None else args.task_prompt_file.read_text(encoding="utf-8")
    floor_source = None
    if args.retention_floor_source is not None:
        floor_source = json.loads(args.retention_floor_source.read_text(encoding="utf-8"))
        if not isinstance(floor_source, dict):
            raise SystemExit(f"{args.retention_floor_source} must hold a JSON object")
    result = run_session(
        args.instance,
        args.arm,
        args.repeat,
        model=args.model,
        effort=args.effort,
        wall_timeout_s=60.0 * args.wall_timeout_min,
        allow_main=args.allow_main,
        threads=args.threads,
        cell_timeout_s=args.cell_timeout_s,
        max_cells=args.max_cells,
        mcp_tool_timeout_ms=args.mcp_tool_timeout_ms,
        mcp_connect_timeout_ms=args.mcp_connect_timeout_ms,
        system_prompt_mode=args.system_prompt_mode,
        task_prompt_override=override,
        dry_run=args.dry_run,
        retention_floor_frac=args.retention_floor_frac,
        library_sessions=args.library_sessions,
        seqskill_variant=args.seqskill_variant,
        lift_rel_frob=args.lift_rel_frob,
        retention_floor_rows=args.retention_floor_rows,
        retention_floor_source=floor_source,
        max_checks=args.max_checks,
        blind_predictions=args.blind_show_predictions,
        e11_variant=args.e11_variant,
        api_key_file=args.api_key_file,
        oauth_token_file=args.oauth_token_file,
    )
    record = result.record
    final = record.get("final") or {}
    print(json.dumps({
        "session_id": result.session_id,
        "status": result.status,
        "root": str(result.root),
        "infra": record.get("infra"),
        "infra_reasons": record.get("infra_reasons"),
        "final_is_base_noop": final.get("is_base_noop"),
        "final_score": final.get("score"),
        "audit": record.get("audit"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
