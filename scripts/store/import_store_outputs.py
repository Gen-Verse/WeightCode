"""Place the outputs of a fetched NSCC store job into the local store (``scripts/store/fetch_store.sh``).

Every output must lie under ``runs/`` (relative to the store root), must match ``outputs.sha256`` and must not exist
locally yet: an existing run record is never overwritten, so a collision refuses the whole import before anything is
copied. A job that changed one of its inputs (``modified.txt`` not empty) or wrote under ``private/`` is refused.
The import record ``import.json`` (files, sha256, destination) is written into the fetched run directory.

A job with a non-zero exit code is refused unless ``--accept-failed-exit REASON`` names why its finished part stands
(for example: some lanes crashed and are rerun by another job). Then every run directory among the outputs (a
``runs/baselines/.../<seed>/`` directory) must hold its ``run.json``, so a run cut off mid-way is never imported, and
the exit code and the reason go into ``import.json``.

    source scripts/env.sh
    python -I scripts/store/import_store_outputs.py $WEIGHTCODE_PRIVATE/nscc_runs/RUN_NAME [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_outputs(run_dir: Path) -> list[tuple[str, str]]:
    """(sha256, store-relative path) per line of outputs.sha256."""
    rows = []
    for line in (run_dir / "outputs.sha256").read_text().splitlines():
        if line.strip():
            digest, path = line[:64], line[66:]
            rows.append((digest, path))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="$WEIGHTCODE_PRIVATE/nscc_runs/RUN_NAME")
    parser.add_argument("--dry-run", action="store_true", help="check and list, copy nothing")
    parser.add_argument("--accept-failed-exit", metavar="REASON",
                        help="import the complete runs of a job whose exit code is not 0, recording this reason")
    args = parser.parse_args(argv)
    runs_root = Path(os.environ["WEIGHTCODE_RUNS"]).resolve()
    run_dir = args.run_dir.resolve()
    if (run_dir / "import.json").exists():
        print(f"{run_dir} was imported already (import.json)", file=sys.stderr)
        return 3
    exit_record = json.loads((run_dir / "exit.json").read_text())
    if exit_record["exit_code"] != 0 and not args.accept_failed_exit:
        print(f"job exit code {exit_record['exit_code']}; import refused (see --accept-failed-exit)", file=sys.stderr)
        return 3
    modified = [line for line in (run_dir / "modified.txt").read_text().splitlines() if line.strip()]
    if modified:
        print(f"the job changed {len(modified)} inputs, e.g. {modified[:3]}; import refused", file=sys.stderr)
        return 3
    rows = read_outputs(run_dir)
    plan, problems = [], []
    for digest, rel in rows:
        if not rel.startswith("runs/") or ".." in Path(rel).parts:
            problems.append(f"output outside runs/: {rel}")
            continue
        source = run_dir / "store" / rel
        target = runs_root / rel[len("runs/"):]
        if sha256_file(source) != digest:
            problems.append(f"sha256 mismatch: {rel}")
        if target.exists():
            problems.append(f"exists locally: {target}")
        plan.append((rel, source, target, digest))
    if exit_record["exit_code"] != 0:
        rels = {rel for _, rel in rows}
        run_dirs = {"/".join(Path(rel).parts[:6]) for rel in rels
                    if Path(rel).parts[:2] == ("runs", "baselines") and len(Path(rel).parts) > 6}
        problems += [f"run without run.json (cut off?): {d}" for d in sorted(run_dirs) if f"{d}/run.json" not in rels]
    if problems:
        print("import refused:", *problems[:20], sep="\n  ", file=sys.stderr)
        return 3
    for rel, _, target, _ in plan:
        print(f"{rel} -> {target}")
    if args.dry_run:
        print(f"[dry run] {len(plan)} files, nothing copied")
        return 0
    for _, source, target, _ in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    record = {"imported": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "runs_root": str(runs_root), "job_exit_code": exit_record["exit_code"],
              "accepted_failed_exit": args.accept_failed_exit,
              "files": [{"path": rel, "sha256": digest, "target": str(target)} for rel, _, target, digest in plan]}
    (run_dir / "import.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"imported {len(plan)} files into {runs_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
