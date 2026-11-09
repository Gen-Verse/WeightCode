"""Data and run locations, resolved from environment variables only.

There are no defaults: a missing variable is an error, so no process silently writes sealed material into
the repository or into a directory mounted into the editor sandbox.
"""

from __future__ import annotations

import os
from pathlib import Path

PRIVATE_VAR = "WEIGHTCODE_PRIVATE"
SEALED_VAR = "WEIGHTCODE_SEALED"
RUNS_VAR = "WEIGHTCODE_RUNS"


def _require_dir(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"environment variable {variable} is not set (see README.md: Data and saved runs)")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{variable}={value} is not an existing directory")
    return path


def private_dir() -> Path:
    """Root of train/val data, split indices, base checkpoints and private manifests."""
    return _require_dir(PRIVATE_VAR)


def sealed_dir() -> Path:
    """Root of official test sets, the sealed evaluator state and sealed results."""
    return _require_dir(SEALED_VAR)


def runs_dir() -> Path:
    """Root of trial directories and run logs."""
    return _require_dir(RUNS_VAR)


def relocate_run_path(recorded: str, store: Path) -> Path:
    """Resolve a recorded runs/... path under an explicitly supplied store root."""
    parts = Path(recorded).parts
    if parts.count("runs") != 1 or ".." in parts:
        raise ValueError(f"not a store runs path: {recorded!r}")
    relative = Path(*parts[parts.index("runs") + 1:])
    root = (Path(store) / "runs").resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"recorded path leaves the runs directory: {recorded!r}")
    return resolved
