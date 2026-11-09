"""Provenance of a data or training run: repository state, source-file hashes and library versions."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

CODE_GLOBS: tuple[str, ...] = (
    "weightcode/common/identity.py",
    "weightcode/evalkit/patch_check.py",
    "weightcode/data/*.py",
    "weightcode/models/*.py",
    "weightcode/models/templates/*.tmpl",
    "scripts/prepare_data.py",
    "scripts/train_base.py",
)
GIT_STATE_KEYS: tuple[str, ...] = ("commit", "dirty")


def git_state(repo_root: Path) -> dict[str, Any]:
    """HEAD commit and whether the working tree has uncommitted changes (read-only git calls)."""
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def code_sha256(repo_root: Path, globs: Sequence[str] = CODE_GLOBS) -> dict[str, str]:
    """sha256 of every file matched by ``globs`` (relative to the repo root), so a dirty tree stays auditable."""
    hashes: dict[str, str] = {}
    for pattern in globs:
        matches = sorted(repo_root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no file matches {pattern} under {repo_root}")
        for path in matches:
            hashes[str(path.relative_to(repo_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def library_versions() -> dict[str, Any]:
    """Python, numpy and, if importable, torch/torchvision/CUDA versions and the GPU name."""
    versions: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__}
    import torch
    import torchvision

    versions.update(
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
    )
    return versions


def read_git_state_file(path: Path) -> dict[str, Any]:
    """Git state recorded at submission for a code copy without ``.git`` (NSCC runs): ``{"commit", "dirty"}``."""
    state = json.loads(Path(path).read_text())
    if not isinstance(state, dict) or tuple(sorted(state)) != tuple(sorted(GIT_STATE_KEYS)):
        raise ValueError(f"{path}: expected keys {GIT_STATE_KEYS}, got {state!r}")
    if not isinstance(state["commit"], str) or len(state["commit"]) != 40 or not isinstance(state["dirty"], bool):
        raise ValueError(f"{path}: commit must be a 40-character hash and dirty a bool")
    return {"commit": state["commit"], "dirty": state["dirty"], "recorded_at_submission": str(path.name)}


def run_info(repo_root: Path, git_state_file: Path | None = None) -> dict[str, Any]:
    """Combined provenance block for manifests.

    ``git_state_file`` replaces the git call for a code copy that has no ``.git`` (an NSCC run directory); the
    per-file ``code_sha256`` is always computed from the files that actually run.
    """
    git = git_state(repo_root) if git_state_file is None else read_git_state_file(git_state_file)
    return {"git": git, "code_sha256": code_sha256(repo_root), "versions": library_versions()}
