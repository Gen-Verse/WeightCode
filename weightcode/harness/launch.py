"""Launching the editor: the Claude Code command line, its environment and the wall-clock limit (HARNESS_SPEC 5).

``ClaudeLauncher`` runs::

    claude -p <task prompt> --model <model> --effort <effort> --setting-sources "" --tools "" \\
      --disable-slash-commands --strict-mcp-config --mcp-config <mcp.json> --allowedTools <the arm's tools> \\
      --system-prompt-file <prompts/system.md> --output-format stream-json --verbose --no-session-persistence

with stdin from /dev/null, ``cwd=<session>/cwd``, stdout to ``logs/transcript.jsonl`` and stderr to
``logs/claude_stderr.txt``, in its own process group, which is killed (SIGTERM, then SIGKILL) at the wall-clock
limit. Just before the editor process starts, the launcher writes ``logs/launch_start.json``
(``record_launch_start``), the origin of that limit, from which the MCP server reports ``wall_s_elapsed`` and
``wall_s_remaining`` in every tool result (``weightcode.harness.tools.ToolServer.clock``). ``--append-system-prompt-file``
replaces ``--system-prompt-file`` when the session's system prompt mode is
"append" (the spec's fallback if replacing the system prompt drops the MCP tools).

The environment is the caller's minus every variable that couples the child to the parent Claude Code session or to
this repository (``CLAUDE*``, ``MCP_*``, ``ANTHROPIC_*``, ``PYTHON*``, ``WEIGHTCODE_*``, editor-integration variables and
the BLAS thread variables), plus ``MCP_TOOL_TIMEOUT`` and ``MCP_TIMEOUT`` from the session. The names of removed
variables are recorded, never their values.

Billing. By default the editor uses the Claude Code login of the account it runs under (the subscription). With an
API key file (``ClaudeLauncher(api_key_file=...)``, Execution Plan v1.1, user 2026-11-06), the launcher reads the key
from that file just before the editor starts and sets ``ANTHROPIC_API_KEY`` in the editor's environment only. The file
must be a regular file of the current user that no one else can read or write (mode 0600 or stricter) and hold one
line. The key itself is never returned, logged or recorded: session.json records only the billing mode and the
file's path (``billing_record``).

OAuth token file (``ClaudeLauncher(oauth_token_file=...)``, user 2026-11-07, T04 on NSCC): the same rules for a
long-lived subscription token made with ``claude setup-token``. The launcher sets ``CLAUDE_CODE_OAUTH_TOKEN`` in the
editor's environment only, after the ``CLAUDE*`` variables are removed, so batch sessions on compute nodes bill the
subscription of the account that made the token without a login refresh. A launcher takes at most one of the two
files.
"""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from weightcode.harness.jsonio import read_json, utc_now, write_json
from weightcode.harness.session import SessionPaths

ENV_REMOVE_PREFIXES = ("CLAUDE", "MCP_", "ANTHROPIC_", "PYTHON", "WEIGHTCODE_", "VSCODE_", "COPILOT_", "GIT_")
ENV_REMOVE_NAMES = frozenset({"AI_AGENT", "VIRTUAL_ENV", "ELECTRON_RUN_AS_NODE", "OMP_NUM_THREADS",
                              "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"})
SYSTEM_PROMPT_FLAGS = {"replace": "--system-prompt-file", "append": "--append-system-prompt-file"}
KILL_REASON_WALL = "wall_timeout"
DEFAULT_KILL_GRACE_S = 10.0
LAUNCH_START_SCHEMA = "weightcode.harness.launch_start.v0"
BILLING_SUBSCRIPTION = "subscription"
BILLING_API_KEY_FILE = "api_key_file"
BILLING_OAUTH_TOKEN_FILE = "oauth_token_file"
API_KEY_ENV = "ANTHROPIC_API_KEY"
OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"


class CredentialFileError(RuntimeError):
    """A credential file is missing, not private to the current user, or not one non-empty line."""


class ApiKeyFileError(CredentialFileError):
    """The API key file is missing, not private to the current user, or not one non-empty line."""


class OAuthTokenFileError(CredentialFileError):
    """The OAuth token file is missing, not private to the current user, or not one non-empty line."""


def _read_private_line(path: str | Path, what: str, error: type[CredentialFileError]) -> str:
    """The one line in ``path``, a regular file of the current user with mode 600 or stricter; every failure names
    the file, never its content."""
    file = Path(path)
    try:
        info = file.stat()
    except FileNotFoundError:
        raise error(f"{what} {file} does not exist") from None
    if not stat.S_ISREG(info.st_mode):
        raise error(f"{what} {file} is not a regular file")
    if info.st_uid != os.getuid():
        raise error(f"{what} {file} is not owned by the current user")
    if info.st_mode & 0o077:
        raise error(f"{what} {file} has mode {stat.S_IMODE(info.st_mode):o}; it must be 600 or stricter")
    lines = [line.strip() for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1 or any(ch.isspace() for ch in lines[0]):
        raise error(f"{what} {file} must hold exactly one line with the credential")
    return lines[0]


def read_api_key(path: str | Path) -> str:
    """The key in ``path`` (see the module docstring); every failure names the file, never its content."""
    return _read_private_line(path, "API key file", ApiKeyFileError)


def read_oauth_token(path: str | Path) -> str:
    """The token in ``path`` (see the module docstring); every failure names the file, never its content."""
    return _read_private_line(path, "OAuth token file", OAuthTokenFileError)


def billing_record(api_key_file: str | Path | None, oauth_token_file: str | Path | None = None) -> dict[str, str]:
    """What session.json records about billing: the mode and, for a credential file, its absolute path (never the
    credential)."""
    if api_key_file is not None and oauth_token_file is not None:
        raise ValueError("bill the editor to an API key file or an OAuth token file, not both")
    if api_key_file is not None:
        return {"mode": BILLING_API_KEY_FILE, "api_key_file": str(Path(api_key_file).absolute())}
    if oauth_token_file is not None:
        return {"mode": BILLING_OAUTH_TOKEN_FILE, "oauth_token_file": str(Path(oauth_token_file).absolute())}
    return {"mode": BILLING_SUBSCRIPTION}


class LaunchStartError(RuntimeError):
    """logs/launch_start.json is missing or malformed: the session's wall clock has no origin."""


def record_launch_start(paths: SessionPaths, session: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``logs/launch_start.json`` (refused if it exists): the time the editor is started, in UTC and as Unix
    seconds, with the session's wall-clock limit; returns the record. Every launcher calls it just before it starts
    the editor, so the MCP server, which the editor starts, can read it."""
    if paths.launch_start.exists():
        raise LaunchStartError(f"{paths.launch_start} exists; a session is launched once")
    record = {"schema": LAUNCH_START_SCHEMA, "started_at": utc_now(), "started_unix": time.time(),
              "wall_timeout_s": float(session["budget"]["wall_timeout_s"])}
    write_json(paths.launch_start, record)
    return record


def read_launch_start(paths: SessionPaths) -> dict[str, Any]:
    """The session's launch record (``record_launch_start``), checked."""
    if not paths.launch_start.is_file():
        raise LaunchStartError(f"{paths.launch_start} is missing: the session was not started by a launcher")
    record = read_json(paths.launch_start)
    if not isinstance(record, dict) or record.get("schema") != LAUNCH_START_SCHEMA or not isinstance(
            record.get("started_unix"), (int, float)) or not isinstance(record.get("wall_timeout_s"), (int, float)):
        raise LaunchStartError(f"{paths.launch_start} is not a {LAUNCH_START_SCHEMA} record")
    return record


@dataclass
class LaunchOutcome:
    """How the editor process (or the scripted stand-in) ran and ended."""

    launcher: str
    stream_expected: bool
    started_at: str
    ended_at: str | None = None
    wall_s: float | None = None
    argv: list[str] | None = None
    exit_code: int | None = None
    killed: bool = False
    kill_reason: str | None = None
    error: str | None = None
    env_removed: list[str] = field(default_factory=list)
    env_set: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Launcher(Protocol):
    """Runs the editor of one prepared session and reports how it ended."""

    name: str
    stream_expected: bool

    def launch(self, session: Mapping[str, Any], paths: SessionPaths, task_prompt: str) -> LaunchOutcome: ...


def editor_environment(base: Mapping[str, str], session: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    """The environment of the editor process and the names removed from ``base``."""
    removed = sorted(key for key in base if key in ENV_REMOVE_NAMES or key.startswith(ENV_REMOVE_PREFIXES))
    env = {key: value for key, value in base.items() if key not in removed}
    env.update(editor_env_settings(session))
    return env, removed


def editor_env_settings(session: Mapping[str, Any]) -> dict[str, str]:
    """Variables the harness sets for Claude Code: the MCP tool-call and server-startup timeouts."""
    mcp = session["mcp"]
    return {"MCP_TOOL_TIMEOUT": str(int(mcp["tool_timeout_ms"])), "MCP_TIMEOUT": str(int(mcp["connect_timeout_ms"]))}


def find_claude(claude_bin: str | None = None) -> str:
    """The Claude Code executable: ``claude_bin`` or ``claude`` on PATH; a missing one is an error."""
    found = shutil.which(claude_bin or "claude")
    if found is None:
        raise FileNotFoundError(f"Claude Code executable {claude_bin or 'claude'!r} not found")
    return found


def claude_argv(claude: str, session: Mapping[str, Any], paths: SessionPaths, task_prompt: str) -> list[str]:
    """The Claude Code command line of one session (HARNESS_SPEC section 5)."""
    editor = session["editor"]
    flag = SYSTEM_PROMPT_FLAGS[str(editor["system_prompt_mode"])]
    return [
        claude, "-p", task_prompt,
        "--model", str(editor["model"]),
        "--effort", str(editor["effort"]),
        "--setting-sources", "",
        "--tools", "",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--mcp-config", str(paths.mcp_json),
        "--allowedTools", ",".join(session["allowed_tools"]),
        flag, str(paths.system_prompt),
        "--output-format", "stream-json",
        "--verbose",
        "--no-session-persistence",
    ]  # fmt: skip


def _kill_group(proc: subprocess.Popen[bytes], grace_s: float) -> None:
    """SIGTERM to the process group, then SIGKILL after ``grace_s``; reaps the leader."""
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, grace_s)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue
    proc.wait()


class ClaudeLauncher:
    """Runs ``claude -p`` for a prepared session under the wall-clock limit."""

    name = "claude"
    stream_expected = True

    def __init__(self, claude_bin: str | None = None, *, kill_grace_s: float = DEFAULT_KILL_GRACE_S,
                 api_key_file: str | Path | None = None, oauth_token_file: str | Path | None = None) -> None:
        self.claude = find_claude(claude_bin)
        self.kill_grace_s = kill_grace_s
        self.api_key_file = None if api_key_file is None else Path(api_key_file)
        self.oauth_token_file = None if oauth_token_file is None else Path(oauth_token_file)
        self.billing = billing_record(self.api_key_file, self.oauth_token_file)  # refuses both at once
        # refuse a bad file before any session is prepared
        if self.api_key_file is not None:
            read_api_key(self.api_key_file)
        if self.oauth_token_file is not None:
            read_oauth_token(self.oauth_token_file)

    def environment(self, session: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
        env, removed = editor_environment(os.environ, session)
        if self.api_key_file is not None:
            env[API_KEY_ENV] = read_api_key(self.api_key_file)
        if self.oauth_token_file is not None:
            env[OAUTH_TOKEN_ENV] = read_oauth_token(self.oauth_token_file)
        return env, removed

    def launch(self, session: Mapping[str, Any], paths: SessionPaths, task_prompt: str) -> LaunchOutcome:
        env, removed = self.environment(session)
        argv = claude_argv(self.claude, session, paths, task_prompt)
        outcome = LaunchOutcome(launcher=self.name, stream_expected=True, started_at=utc_now(), argv=argv,
                                env_removed=removed, env_set=editor_env_settings(session))
        wall_timeout_s = float(session["budget"]["wall_timeout_s"])
        record_launch_start(paths, session)
        began = time.monotonic()
        with open(paths.transcript, "wb") as out, open(paths.claude_stderr, "wb") as err:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, cwd=paths.cwd, env=env,
                                    start_new_session=True)
            try:
                outcome.exit_code = proc.wait(timeout=wall_timeout_s)
            except subprocess.TimeoutExpired:
                outcome.killed = True
                outcome.kill_reason = KILL_REASON_WALL
                _kill_group(proc, self.kill_grace_s)
                outcome.exit_code = proc.returncode
            except BaseException:
                _kill_group(proc, self.kill_grace_s)
                raise
        outcome.wall_s = round(time.monotonic() - began, 3)
        outcome.ended_at = utc_now()
        return outcome


__all__ = ["API_KEY_ENV", "ApiKeyFileError", "BILLING_API_KEY_FILE", "BILLING_OAUTH_TOKEN_FILE", "BILLING_SUBSCRIPTION",
           "ClaudeLauncher", "CredentialFileError", "OAUTH_TOKEN_ENV", "OAuthTokenFileError", "billing_record",
           "read_api_key", "read_oauth_token", "KILL_REASON_WALL", "LAUNCH_START_SCHEMA", "LaunchOutcome", "LaunchStartError", "Launcher",
           "SYSTEM_PROMPT_FLAGS", "claude_argv", "editor_env_settings", "editor_environment", "find_claude",
           "read_launch_start", "record_launch_start"]
