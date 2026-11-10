"""Provenance of one session: code, prompts, sandbox and editor versions (HARNESS_SPEC section 5).

Recorded in session.json before the launch and again in session_result.json after it, so a code change during a
session is visible. Code identity is the git HEAD plus one hash of the working-tree state (``git status
--porcelain=v1`` output followed by ``git diff`` output) plus the sha256 of every .py file under the packages a
session runs (untracked files are not in ``git diff``, so the file hashes carry them). Only read-only git commands
are used, with ``--no-optional-locks``.

The host CPU (``cpu``: model name and SIMD flags from ``/proc/cpuinfo``) was added on 2026-11-07, because some program
runs reproduce their child hash bit for bit only on the same CPU class (docs/maintenance/CODE_AUDIT.md section 9).
It is ``None`` where ``/proc/cpuinfo`` does not exist (not Linux). Records written before that date have no ``cpu``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import weightcode
from weightcode.harness.jsonio import text_sha256
from weightcode.harness.prompts import template_sha256s
from weightcode.sandbox import DEFAULT_BWRAP, bwrap_version, kernel_sha256

REPO_ROOT = Path(weightcode.__file__).resolve().parents[1]
HASHED_PACKAGES = ("harness", "patch", "sandbox", "evalkit", "common", "audit")
COMMAND_TIMEOUT_S = 60.0
CPUINFO_PATH = Path("/proc/cpuinfo")
SIMD_FLAG_RE = re.compile(r"^(mmx|sse\d\w*|sse|ssse3|avx\w*|fma\w*|f16c|amx\w*|asimd\w*|neon|sve\w*)$")
"""Flags kept in ``cpu.simd_flags``: x86 MMX, SSE, AVX (AVX2, AVX-512, AVX-VNNI), FMA, F16C, AMX; Arm ASIMD, NEON,
SVE. Arm SME is left out because x86 uses ``sme`` for AMD memory encryption. The full flag list is not stored;
``cpu.flags_sha256`` hashes it so any other difference is visible."""


class ProvenanceError(RuntimeError):
    """A provenance command failed; the runner never records a guessed value."""


def _run(argv: list[str], *, env: Mapping[str, str] | None = None) -> str:
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S, check=False,
                                   env=None if env is None else dict(env), stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProvenanceError(f"{argv[0]} failed: {exc}") from exc
    if completed.returncode != 0:
        raise ProvenanceError(f"{' '.join(argv)} exited {completed.returncode}: {completed.stderr.strip()[:500]}")
    return completed.stdout


def _git_env() -> dict[str, str]:
    """The caller's environment without GIT_* variables, so no override can point git at another repository."""
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """HEAD and one sha256 over ``git status --porcelain=v1`` output followed by ``git diff`` output."""
    base = ["git", "--no-optional-locks", "-C", str(repo)]
    env = _git_env()
    head = _run([*base, "rev-parse", "HEAD"], env=env).strip()
    status = _run([*base, "status", "--porcelain=v1"], env=env)
    diff = _run([*base, "diff"], env=env)
    return {
        "head": head,
        "status_diff_sha256": text_sha256(status + diff),
        "dirty": bool(status.strip()),
        "n_status_lines": len([line for line in status.splitlines() if line.strip()]),
    }


def source_hashes(repo: Path = REPO_ROOT) -> dict[str, str]:
    """sha256 of every .py file under ``weightcode/<package>`` for the hashed packages, by repository-relative path."""
    hashes: dict[str, str] = {}
    for package in HASHED_PACKAGES:
        root = repo / "weightcode" / package
        if not root.is_dir():
            raise ProvenanceError(f"package directory {root} is missing")
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            hashes[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def digest_of(mapping: Mapping[str, str]) -> str:
    """One sha256 over a sorted ``{name: hash}`` mapping."""
    return text_sha256(json.dumps(dict(sorted(mapping.items())), sort_keys=True))


def sandbox_versions(python: Path) -> dict[str, str]:
    """Python and numpy versions of the sandbox interpreter, run isolated with an empty environment."""
    code = "import json, sys, numpy; print(json.dumps({'python': sys.version, 'numpy': numpy.__version__}))"
    output = _run([str(python), "-I", "-c", code], env={})
    versions = json.loads(output)
    if not isinstance(versions, dict):
        raise ProvenanceError("the sandbox interpreter printed no version record")
    return {"python": str(versions["python"]), "numpy": str(versions["numpy"])}


def cpu_info(path: Path = CPUINFO_PATH) -> dict[str, Any] | None:
    """Model name and SIMD flags of the host CPU from ``/proc/cpuinfo`` (Linux); ``None`` where the file is absent.

    The first processor block that names them is read: ``model name`` (x86 and some Arm kernels) and ``flags`` (x86)
    or ``Features`` (Arm). A field the file does not have is recorded as ``None``, never guessed. ``n_logical`` counts
    the ``processor`` entries.
    """
    if not path.exists():
        return None
    model: str | None = None
    flags: list[str] | None = None
    n_logical = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        if key == "processor":
            n_logical += 1
        elif key == "model name" and model is None:
            model = value.strip()
        elif key in ("flags", "Features") and flags is None:
            flags = sorted(set(value.split()))
    return {
        "source": str(path),
        "model": model,
        "n_logical": n_logical,
        "simd_flags": None if flags is None else [flag for flag in flags if SIMD_FLAG_RE.match(flag)],
        "flags_sha256": None if flags is None else text_sha256(" ".join(flags)),
    }


def claude_version(claude_bin: str, env: Mapping[str, str]) -> str:
    """``claude --version`` output."""
    return _run([claude_bin, "--version"], env=env).strip()


def collect(
    *,
    sandbox_python: Path,
    bwrap: Path = DEFAULT_BWRAP,
    prompts: Mapping[str, str],
    claude: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The provenance record: code, prompts (templates and rendered texts), sandbox and editor versions."""
    sources = source_hashes()
    return {
        "git": git_state(),
        "source_sha256": sources,
        "source_tree_sha256": digest_of(sources),
        "prompt_templates_sha256": template_sha256s(),
        "rendered_prompts_sha256": {name: text_sha256(text) for name, text in prompts.items()},
        "kernel_sha256": kernel_sha256(),
        "bwrap_version": bwrap_version(bwrap),
        "sandbox": sandbox_versions(sandbox_python),
        "host": {"python": sys.version, "numpy": np.__version__},
        "claude": None if claude is None else dict(claude),
        "cpu": cpu_info(CPUINFO_PATH),
    }


def code_changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """Provenance fields that differ between the record before the launch and the one after it."""
    keys = ("git", "source_tree_sha256", "prompt_templates_sha256", "kernel_sha256")
    return [key for key in keys if before.get(key) != after.get(key)]


__all__ = ["CPUINFO_PATH", "HASHED_PACKAGES", "REPO_ROOT", "SIMD_FLAG_RE", "ProvenanceError", "claude_version",
           "code_changed", "collect", "cpu_info", "digest_of", "git_state", "sandbox_versions", "source_hashes"]
