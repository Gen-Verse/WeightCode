"""Run (or resume) one PREREG E9 lineage, T, F or C, or one E9 v2 lineage, T2, F2 or C2, through session 9.

    .venv/bin/python scripts/run_fscil_lineage.py --lineage T --refs DIR [--until 9]
    .venv/bin/python scripts/run_fscil_lineage.py --lineage T --floors FILE [--until 9]
    .venv/bin/python scripts/run_fscil_lineage.py --lineage F
    .venv/bin/python scripts/run_fscil_lineage.py --lineage T --dry-run [--refs DIR | --placeholder-floor N]
    .venv/bin/python scripts/run_fscil_lineage.py --lineage T2 [--until 9]
    .venv/bin/python scripts/run_fscil_lineage.py --lineage T2 --dry-run

T2, F2 and C2 (PREREG E9 v2, ``weightcode.harness.fscil_lineage``) keep by fscil-retain-v2, whose floor is per candidate, so
they take no ``--refs``, ``--floors`` or ``--placeholder-floor``.

The floors come from the reference streams' run directory (``--refs``, ``scripts/run_fscil_streams.py --out``) or
from a floors file (``--floors``); T and C need one, F none. T's sessions 2 to 4 are LEARN sessions (effort max, 90
minutes, 4 threads), so in production the driver runs as a systemd unit that oomd leaves alone::

    systemd-run --user --collect --property=ManagedOOMPreference=avoid --unit weightcode-e9-T \\
        --working-directory=<worktree> --setenv=WEIGHTCODE_PRIVATE=... --setenv=WEIGHTCODE_SEALED=... --setenv=WEIGHTCODE_RUNS=... \\
        <worktree>/.venv/bin/python scripts/run_fscil_lineage.py --lineage T --refs DIR

The worktree path must hold no forbidden string (no "fscil", "cifar" or "seed"): the sandbox self-test reads the
session's mounts, which include the repo. ``--dry-run`` builds (or reuses) T's session-2 instance and runs that
session as a dry run (no editor launched); it leaves the lineage state untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lineage", choices=("T", "F", "C", "T2", "F2", "C2"), required=True)
    floors = parser.add_mutually_exclusive_group()
    floors.add_argument("--refs", type=Path, help="the reference streams' run directory")
    floors.add_argument("--floors", type=Path, help="a floors file (weightcode.fscil_floors.v1)")
    parser.add_argument("--until", type=int, default=9)
    parser.add_argument("--base-seed", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--placeholder-floor", type=int, default=None,
                        help="a dry run's stand-in floor when no floor is stated yet")
    args = parser.parse_args(argv)
    from weightcode.evalkit.fscil import base_student
    from weightcode.harness import fscil_lineage as L

    base = base_student(args.base_seed)
    if args.lineage in L.LINEAGES_V2 and (args.refs is not None or args.floors is not None
                                         or args.placeholder_floor is not None):
        raise SystemExit(f"lineage {args.lineage} keeps by fscil-retain-v2 and reads no floors")
    table = None
    if args.refs is not None:
        table = L.load_reference_floors(args.refs, str(base["params_sha256"]))
    elif args.floors is not None:
        table = L.load_floors(args.floors)
    if args.dry_run:
        if args.lineage not in ("T", "T2"):
            raise SystemExit("only T's and T2's session 2 have a dry run")
        result = L.dry_run(floors=table, placeholder_floor=args.placeholder_floor, base_seed=args.base_seed,
                           lineage=args.lineage)
        print(json.dumps({"session_id": result.session_id, "status": result.status, "root": str(result.root),
                          "selftest": result.record.get("selftest"), "trial_files": result.record.get("trial_files")},
                         indent=1))
        return 0 if result.status == "dry_run" else 1
    if args.placeholder_floor is not None:
        raise SystemExit("--placeholder-floor is for a dry run only")
    driver = L.LineageDriver(args.lineage, floors=table, base_seed=args.base_seed)
    state = driver.run(args.until)
    print(json.dumps({"state": str(driver.state_path), "summary": state["summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
