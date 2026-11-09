"""A scripted stand-in for the editor, for tests and the no-LLM integration check (HARNESS_SPEC section 10).

``McpStdioClient`` starts the session's real MCP server exactly as ``mcp.json`` tells Claude Code to (same command,
arguments and server environment, the same scrubbed parent environment, ``cwd=<session>/cwd``) and speaks
newline-delimited JSON-RPC 2.0 to it over stdio. ``ScriptedLauncher`` runs a script of tool calls through such a
client in place of ``claude -p``. A scripted session has no Claude Code process and writes no stream-json
transcript; its record says ``launcher: "scripted"``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from weightcode.harness.arms import MCP_SERVER_NAME
from weightcode.harness.jsonio import read_json, utc_now
from weightcode.harness.launch import LaunchOutcome, editor_environment, record_launch_start
from weightcode.harness.session import SessionPaths

DEFAULT_PROTOCOL_VERSION = "2025-06-18"
DEFAULT_REPLY_TIMEOUT_S = 600.0


class McpError(RuntimeError):
    """A JSON-RPC error reply, or a server that stopped answering."""

    def __init__(self, message: str, code: int | None = None) -> None:
        self.code = code
        super().__init__(message)


@dataclass
class ToolReply:
    """One ``tools/call`` result as the editor would see it."""

    name: str
    arguments: dict[str, Any]
    payload: Any
    is_error: bool


class McpStdioClient:
    """A minimal MCP client for one stdio server process."""

    def __init__(self, argv: list[str], env: Mapping[str, str], cwd: Path, stderr_path: Path,
                 reply_timeout_s: float = DEFAULT_REPLY_TIMEOUT_S) -> None:
        self.argv = list(argv)
        self._stderr = open(stderr_path, "ab")
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                                     cwd=cwd, env=dict(env))
        self.reply_timeout_s = reply_timeout_s
        self._next_id = 0

    @classmethod
    def for_session(cls, session: Mapping[str, Any], paths: SessionPaths) -> McpStdioClient:
        """The session's server as mcp.json describes it, with the editor environment Claude Code would pass."""
        config = read_json(paths.mcp_json)["mcpServers"][MCP_SERVER_NAME]
        env, _ = editor_environment(os.environ, session)
        env.update({str(key): str(value) for key, value in config.get("env", {}).items()})
        return cls([str(config["command"]), *map(str, config["args"])], env, paths.cwd,
                   paths.logs / "scripted_client_stderr.txt")

    def _send(self, message: Mapping[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def _read(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise McpError(f"the server closed its output (exit status {self.proc.poll()})")
        message = json.loads(line)
        if not isinstance(message, dict):
            raise McpError(f"the server sent a non-object message: {message!r}")
        return message

    def request(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        """Send one request and return its ``result``; a JSON-RPC error raises ``McpError``."""
        self._next_id += 1
        request_id = self._next_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = dict(params)
        self._send(message)
        reply = self._read()
        if reply.get("id") != request_id or reply.get("jsonrpc") != "2.0":
            raise McpError(f"reply {reply!r} does not answer request {request_id}")
        if "error" in reply:
            raise McpError(str(reply["error"].get("message")), reply["error"].get("code"))
        return reply["result"]

    def raw(self, line: bytes) -> dict[str, Any]:
        """Send raw bytes as one line and read one reply (protocol tests)."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(line.rstrip(b"\n") + b"\n")
        self.proc.stdin.flush()
        return self._read()

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = dict(params)
        self._send(message)

    def initialize(self, protocol_version: str = DEFAULT_PROTOCOL_VERSION) -> dict[str, Any]:
        result = self.request("initialize", {"protocolVersion": protocol_version, "capabilities": {},
                                             "clientInfo": {"name": "weightcode-scripted", "version": "0"}})
        self.notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self.request("tools/list")["tools"])

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolReply:
        args = {} if arguments is None else dict(arguments)
        result = self.request("tools/call", {"name": name, "arguments": args})
        content = result["content"]
        if len(content) != 1 or content[0].get("type") != "text":
            raise McpError(f"unexpected tool result content {content!r}")
        return ToolReply(name, args, json.loads(content[0]["text"]), bool(result["isError"]))

    def close(self, timeout_s: float = 60.0) -> int:
        """Close the server's input (end of file ends it) and return its exit status."""
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            status = self.proc.wait(timeout=timeout_s)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            if self.proc.stdout is not None:
                self.proc.stdout.close()
            self._stderr.close()
        return status


Script = Callable[[McpStdioClient, list[ToolReply]], None]


def call_and_record(client: McpStdioClient, replies: list[ToolReply], name: str,
                    arguments: Mapping[str, Any] | None = None) -> ToolReply:
    """Call one tool and append its reply to ``replies``."""
    reply = client.call(name, arguments)
    replies.append(reply)
    return reply


@dataclass
class ScriptedLauncher:
    """Runs ``script(client, replies)`` against the session's real MCP server instead of ``claude -p``."""

    script: Script
    name: str = "scripted"
    stream_expected: bool = False
    replies: list[ToolReply] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)

    def launch(self, session: Mapping[str, Any], paths: SessionPaths, task_prompt: str) -> LaunchOutcome:
        outcome = LaunchOutcome(launcher=self.name, stream_expected=False, started_at=utc_now())
        record_launch_start(paths, session)
        began = time.monotonic()
        client = McpStdioClient.for_session(session, paths)
        outcome.argv = client.argv
        try:
            client.initialize()
            self.tool_names = [tool["name"] for tool in client.list_tools()]
            self.script(client, self.replies)
        except Exception as exc:
            outcome.error = f"scripted editor: {type(exc).__name__}: {exc}"
        finally:
            outcome.exit_code = client.close()
        outcome.wall_s = round(time.monotonic() - began, 3)
        outcome.ended_at = utc_now()
        return outcome


__all__ = ["McpError", "McpStdioClient", "ScriptedLauncher", "ToolReply", "call_and_record"]
