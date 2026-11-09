"""List the store files a --dev run of a sealed list opens, for a store job's --inputs file (no forward pass).

The dev path of ``weightcode.evalkit.sealed_finals`` runs for every entry of the list (or the ``--only`` names) with the
forward pass replaced by a stub (``devreadout.predict`` returns class 0 for every row), under an audit hook that
records every file opened under the store root. Everything else of the readout of record runs, its instance checks
included (``check_task_current`` reads the instance index), so each child is recovered and hash-checked, the dev rows
and the private index entries are loaded, and each entry's dev readout of record is found, as on NSCC, but no image is
forwarded and the comparison is not a result (the stub's records are written to a scratch file). The patch executor
opens a patch's deltas relative to its directory (dir_fd), which the hook sees without a directory, so the directory
of every patch.json or derived_patch.json opened is listed whole. The run fails if any path under the store's sealed/
is opened or listed.

    .venv/bin/python scripts/sealed/trace_dev_inputs.py --list LIST.json --scratch OUT.jsonl \
        [--only NAME ...] > paths.txt

Paths are printed relative to the store root ($WEIGHTCODE_PRIVATE/..), sorted, one per line.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.paths import private_dir, sealed_dir
from weightcode.evalkit import devreadout
from weightcode.evalkit import sealed_finals as SF

PATCH_FILES = ("patch.json", "derived_patch.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True, help="where the stub records go (never a result)")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    store = private_dir().parent.resolve()
    sealed = sealed_dir().resolve()
    opened: set[Path] = set()
    breaches: list[str] = []

    def hook(event: str, hook_args: tuple[Any, ...]) -> None:
        if event not in ("open", "os.listdir", "os.scandir") or not hook_args:
            return
        target = hook_args[0]
        if not isinstance(target, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(target))
        if not path.is_absolute():
            return
        resolved = path.resolve()
        if resolved == sealed or sealed in resolved.parents:
            breaches.append(f"{event} {resolved}")
        elif event == "open" and store in resolved.parents:
            opened.add(resolved)

    def no_forward(forward: Any, params: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
        return np.zeros(len(x), dtype=np.int64)

    devreadout.predict = no_forward  # type: ignore[assignment]
    sys.addaudithook(hook)
    SF.run_dev(args.list, args.scratch, only=args.only)
    if breaches:
        raise SystemExit(f"the dev path touched the sealed directory: {breaches}")
    paths: set[str] = set()
    for path in opened:
        if path.name in PATCH_FILES:
            paths.add(str(path.parent.relative_to(store)))
        else:
            paths.add(str(path.relative_to(store)))
    covered = {p for p in paths if any(p.startswith(d + "/") for d in paths)}
    for path in sorted(paths - covered):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
