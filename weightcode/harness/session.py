"""Session identity and layout (HARNESS_SPEC sections 1 and 5).

A session id is ``s`` plus 16 random hex digits from ``secrets``. The sandbox can read the host path of every bind
source in /proc/self/mountinfo, so neither the session directory name nor any bound path carries the arm, instance,
dataset, condition, model, seed or repeat. The mapping from session id to those lives only in session.json and in
the runs manifest, both outside the sandbox.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from weightcode.common.paths import runs_dir

SESSION_SCHEMA = "weightcode.session.v0"
SESSION_RESULT_SCHEMA = "weightcode.session_result.v0"
MANIFEST_SCHEMA = "weightcode.runs_manifest_line.v0"
SESSION_ID_RE = re.compile(r"^s[0-9a-f]{16}$")
SESSION_ID_HEX_BYTES = 8
MAX_ID_DRAWS = 256


def sessions_root() -> Path:
    """``$WEIGHTCODE_RUNS/sessions`` (created when a session is created, never by a refusal path)."""
    return runs_dir() / "sessions"


def manifest_path() -> Path:
    """``$WEIGHTCODE_RUNS/manifest.jsonl``: one line per session, appended under an exclusive lock."""
    return runs_dir() / "manifest.jsonl"


def new_session_id(forbidden: Iterable[str], root: Path) -> str:
    """Draw ``s`` + 16 hex digits until the id contains no forbidden string (case-insensitive) and is unused.

    Random hex can spell a forbidden word (``s`` + ``eed...`` is "seed"); such draws are discarded, which keeps
    the id uniform over the allowed ids.
    """
    lowered = [token.lower() for token in forbidden if token]
    for _ in range(MAX_ID_DRAWS):
        candidate = "s" + secrets.token_hex(SESSION_ID_HEX_BYTES)
        if any(token in candidate for token in lowered):
            continue
        if (root / candidate).exists():
            continue
        return candidate
    raise RuntimeError("could not draw an unused session id free of forbidden strings")


@dataclass(frozen=True)
class SessionPaths:
    """Every path of one session directory."""

    root: Path

    @property
    def session_json(self) -> Path:
        return self.root / "session.json"

    @property
    def mcp_json(self) -> Path:
        return self.root / "mcp.json"

    @property
    def result_json(self) -> Path:
        return self.root / "session_result.json"

    @property
    def trial(self) -> Path:
        return self.root / "trial"

    @property
    def patches(self) -> Path:
        return self.root / "patches"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def cwd(self) -> Path:
        return self.root / "cwd"

    @property
    def private(self) -> Path:
        return self.root / "private"

    @property
    def attempts(self) -> Path:
        return self.root / "attempts"

    @property
    def prompts(self) -> Path:
        return self.root / "prompts"

    @property
    def system_prompt(self) -> Path:
        return self.prompts / "system.md"

    @property
    def task_prompt(self) -> Path:
        return self.prompts / "task.md"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def cells_log(self) -> Path:
        return self.logs / "cells.jsonl"

    @property
    def tools_log(self) -> Path:
        return self.logs / "tools.jsonl"

    @property
    def checks_log(self) -> Path:
        return self.logs / "checks.jsonl"

    @property
    def declared_log(self) -> Path:
        return self.logs / "declared.jsonl"

    @property
    def server_log(self) -> Path:
        return self.logs / "server.jsonl"

    @property
    def server_lock(self) -> Path:
        return self.logs / "server.lock"

    @property
    def server_pid(self) -> Path:
        return self.logs / "server.pid"

    @property
    def server_stderr(self) -> Path:
        return self.logs / "server_stderr.txt"

    @property
    def transcript(self) -> Path:
        return self.logs / "transcript.jsonl"

    @property
    def claude_stderr(self) -> Path:
        return self.logs / "claude_stderr.txt"

    @property
    def launch_start(self) -> Path:
        """Written by the launcher just before the editor starts: the origin of the session's wall clock."""
        return self.logs / "launch_start.json"

    @property
    def selftest_json(self) -> Path:
        return self.logs / "selftest.json"

    @property
    def selftest_cells(self) -> Path:
        return self.logs / "selftest_cells.jsonl"

    @property
    def final(self) -> Path:
        return self.root / "final"

    @property
    def final_model(self) -> Path:
        return self.final / "model.npz"

    @property
    def selection_json(self) -> Path:
        return self.final / "selection.json"

    @property
    def audit_json(self) -> Path:
        return self.root / "audit.json"

    def bound_dirs(self) -> tuple[Path, Path, Path]:
        """The three directories bound into the sandbox."""
        return self.trial, self.patches, self.scratch

    def create(self) -> None:
        """Create the session directory and its empty subdirectories; the root must not exist yet."""
        self.root.mkdir(mode=0o700)
        for directory in (self.patches, self.scratch, self.cwd, self.private, self.attempts, self.prompts,
                          self.logs, self.final):
            directory.mkdir(mode=0o700)
        self.cells_log.touch(mode=0o600)
