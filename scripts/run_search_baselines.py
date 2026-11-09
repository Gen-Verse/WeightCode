"""Run the search baselines (Arachne-style DE, CARE-style PSO) on repair instances.

Instances come from $WEIGHTCODE_PRIVATE/instances/index.json: by default every entry with split_role "dev", optionally
narrowed by --dataset, or exactly the ids given with --instance. Before anything runs, the split_role in every
selected instance's private.json is read; a "main" instance is refused (exit status 2, nothing run) unless
--allow-main is given, which is reserved for runs after the pre-registration freeze (PREREG section 2). Each method
on each instance gets its own counted oracle with 8 patch-check queries and writes
$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/<seed>/ (q<k>/ patches and run.json). An existing run directory is
never overwritten.

Example::

    source scripts/env.sh
    .venv/bin/python scripts/run_search_baselines.py --instance ifee5b7da --instance ib338573e --methods arachne care --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
METHODS = ("arachne", "care")
SPLIT_ROLES = ("dev", "main")
SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
EXIT_REFUSED = 2


class Refusal(Exception):
    """The requested selection may not run; the message says why."""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line options (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", action="append", default=[], help="instance id from index.json (repeatable)")
    parser.add_argument("--dataset", help="only index entries of this dataset (not with --instance)")
    parser.add_argument(
        "--split-role", choices=SPLIT_ROLES, help="index entries of this role, default dev (not with --instance)"
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0, help="run seed (search randomness; directory name)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for BLAS")
    parser.add_argument("--allow-main", action="store_true", help="permit main instances (after the freeze only)")
    parser.add_argument("--dry-run", action="store_true", help="print the checked selection and exit")
    return parser.parse_args(argv)


def select_entries(index: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Index entries to run: the --instance ids in order, else the entries of --split-role (and --dataset)."""
    entries = index.get("instances")
    if not isinstance(entries, list):
        raise Refusal("index.json has no 'instances' list")
    by_id = {str(entry["instance_id"]): entry for entry in entries}
    if args.instance:
        if args.dataset is not None or args.split_role is not None:
            raise Refusal("--dataset and --split-role select from the index and cannot be combined with --instance")
        bad = [value for value in args.instance if not SAFE_ID.fullmatch(value) or value in {".", ".."}]
        missing = [value for value in args.instance if value not in by_id]
        if bad or missing:
            raise Refusal(f"unusable instance ids {bad}, ids not in index.json {missing}")
        return [by_id[value] for value in dict.fromkeys(args.instance)]
    role = args.split_role or "dev"
    if role == "main" and not args.allow_main:
        raise Refusal("--split-role main needs --allow-main, which is reserved for runs after the pre-registration freeze")
    return [
        entry
        for entry in entries
        if entry.get("split_role") == role and (args.dataset is None or entry.get("dataset") == args.dataset)
    ]


def check_roles(entries: list[dict[str, Any]], instances_root: Path, allow_main: bool) -> list[Path]:
    """Read each private.json split_role; refuse main (unless allowed), unknown roles and index mismatches."""
    if not entries:
        raise Refusal("no instance matches the selection")
    dirs, main, problems = [], [], []
    for entry in entries:
        instance_id = str(entry["instance_id"])
        instance_dir = instances_root / instance_id
        if not (instance_dir / "private.json").is_file():
            problems.append(f"{instance_id}: {instance_dir / 'private.json'} does not exist")
            continue
        role = json.loads((instance_dir / "private.json").read_text()).get("split_role")
        if role not in SPLIT_ROLES:
            problems.append(f"{instance_id}: private.json split_role {role!r} is not 'dev' or 'main'")
        elif role != entry.get("split_role"):
            problems.append(f"{instance_id}: private.json split_role {role!r} != index.json {entry.get('split_role')!r}")
        elif role == "main":
            main.append(instance_id)
        dirs.append(instance_dir)
    if main and not allow_main:
        problems.append(
            f"main instances {main}: search baselines run on main instances only after the pre-registration freeze "
            "(PREREG section 2); pass --allow-main then"
        )
    if problems:
        raise Refusal("; ".join(problems))
    return dirs


def main(argv: list[str] | None = None) -> int:
    """Check the selection, then run every method on every selected instance; returns the exit status."""
    args = parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
        os.environ[variable] = str(args.threads)
    from weightcode.common.paths import private_dir

    instances_root = private_dir() / "instances"
    try:
        entries = select_entries(json.loads((instances_root / "index.json").read_text()), args)
        instance_dirs = check_roles(entries, instances_root, args.allow_main)
    except Refusal as refusal:
        print(f"refused, nothing was run: {refusal}", file=sys.stderr)
        return EXIT_REFUSED
    for entry in entries:
        target = "" if entry.get("target") is None else f" target {entry['target']}"
        print(f"{entry['instance_id']} {entry.get('dataset')} {entry.get('arch')} {entry['split_role']} seed "
              f"{entry.get('seed')} {entry.get('evidence_rule')}{target}", flush=True)
    if args.dry_run:
        return 0

    from weightcode.baselines.search.runner import baseline_run_dir, run_search_baseline, summary_line

    for instance_dir in instance_dirs:
        for method in args.methods:
            record = run_search_baseline(instance_dir, method, args.seed, allow_main=args.allow_main)
            print(summary_line(record), flush=True)
            print(f"  -> {baseline_run_dir(instance_dir.name, method, args.seed)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
