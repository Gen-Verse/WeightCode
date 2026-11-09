"""Stdio MCP server of one editor session (HARNESS_SPEC section 4).

JSON-RPC 2.0, one JSON object per line, no third-party dependency. Launched by Claude Code as
``<repo>/.venv/bin/python -m weightcode.harness.mcp_server --session <session.json>``.

Protocol (verified on Claude Code 2.1.311): ``initialize`` echoes the client's ``protocolVersion`` and advertises
``{"tools": {}}``; ``notifications/*`` get no reply; ``tools/list``, ``tools/call`` and ``ping`` are served; every other
request gets error -32601. Tool results are ``{"content": [{"type": "text", "text": <JSON>}], "isError": bool}``.

The server is single-threaded and serial: it reads one request, answers it and only then reads the next, so the
query budget and the cell counter cannot race. Only one server may run per session (an exclusive lock on
``logs/server.lock``, which finalisation also waits on). The protocol output goes to a private duplicate of the
original stdout; fd 1 and fd 2 are pointed at ``logs/server_stderr.txt``, so a stray print can never corrupt the
protocol. Every tool call is logged to ``logs/tools.jsonl`` and lifecycle events to ``logs/server.jsonl``. An
unexpected exception inside a tool is a harness fault: it is logged with its traceback and the server exits, which
the runner classifies as INFRA (PREREG section 5); the editor never sees the exception text.

SIGTERM and SIGINT end the server between requests; one that arrives during a request is deferred until the request
is answered and logged, so a consumed query is never lost from the log.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

from weightcode.harness.jsonio import append_jsonl, read_json, utc_now
from weightcode.harness.session import SESSION_SCHEMA, SessionPaths
from weightcode.harness.tools import ToolServer, UnknownTool

SERVER_NAME = "weightcode"
SERVER_VERSION = "0"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
EXIT_FAULT = 70
EXIT_LOCKED = 75
MAX_LINE_BYTES = 32 * 2**20


class ServerStop(Exception):
    """Raised by the signal handler to end the serve loop between requests."""


class HarnessFault(RuntimeError):
    """An unexpected exception inside a tool; the server logs it and exits."""


def thread_problems(session: Mapping[str, Any], environ: Mapping[str, str]) -> list[str]:
    """Differences between the server's thread variables and session.json ``server_threads``."""
    expected = str(int(session["server_threads"]))
    return [f"{name}={environ.get(name)!r}, expected {expected!r}" for name in THREAD_VARIABLES
            if environ.get(name) != expected]


def _response(request_id: Any, *, result: Any = None, error: tuple[int, str] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is None:
        message["result"] = result
    else:
        message["error"] = {"code": error[0], "message": error[1]}
    return message


class McpSession:
    """JSON-RPC dispatch for one ``ToolServer``; logs every tool call."""

    def __init__(self, tools: ToolServer, paths: SessionPaths) -> None:
        self.tools = tools
        self.paths = paths
        self.busy = False

    def log_event(self, event: str, **fields: Any) -> None:
        append_jsonl(self.paths.server_log, {"time": utc_now(), "event": event, "pid": os.getpid(), **fields})

    def handle_line(self, line: bytes) -> dict[str, Any] | None:
        """Answer one input line; None for a notification."""
        try:
            message = json.loads(line)
        except ValueError:
            return _response(None, error=(PARSE_ERROR, "parse error"))
        if not isinstance(message, dict) or not isinstance(message.get("method"), str):
            request_id = message.get("id") if isinstance(message, dict) else None
            return _response(request_id, error=(INVALID_REQUEST, "invalid request"))
        if "id" not in message:
            self.log_event("notification", method=message["method"])
            return None
        return self.handle_request(message["id"], message["method"], message.get("params"))

    def handle_request(self, request_id: Any, method: str, params: Any) -> dict[str, Any]:
        if method == "initialize":
            version = params.get("protocolVersion") if isinstance(params, dict) else None
            if not isinstance(version, str):
                return _response(request_id, error=(INVALID_PARAMS, "initialize needs params.protocolVersion"))
            self.log_event("initialize", protocol_version=version,
                           client=params.get("clientInfo") if isinstance(params, dict) else None)
            return _response(request_id, result={
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })
        if method == "ping":
            return _response(request_id, result={})
        if method == "tools/list":
            return _response(request_id, result={"tools": self.tools.tool_list()})
        if method == "tools/call":
            return self.call_tool(request_id, params)
        self.log_event("unknown_method", method=method)
        return _response(request_id, error=(METHOD_NOT_FOUND, f"method not found: {method}"))

    def call_tool(self, request_id: Any, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            return _response(request_id, error=(INVALID_PARAMS, "tools/call needs params.name"))
        name = params["name"]
        arguments = params.get("arguments")
        started = time.monotonic()
        try:
            outcome = self.tools.call(name, arguments)
        except UnknownTool:
            append_jsonl(self.paths.tools_log, {"time": utc_now(), "request_id": request_id, "tool": name,
                                                "arguments": arguments, "unknown_tool": True})
            return _response(request_id, error=(INVALID_PARAMS, f"unknown tool: {name}"))
        except Exception as exc:
            append_jsonl(self.paths.tools_log, {"time": utc_now(), "request_id": request_id, "tool": name,
                                                "arguments": arguments, "fault": True})
            self.log_event("fault", tool=name, request_id=request_id, error=f"{type(exc).__name__}: {exc}",
                           traceback=traceback.format_exc())
            raise HarnessFault(f"tool {name} raised {type(exc).__name__}") from exc
        text = json.dumps(outcome.payload, sort_keys=True, allow_nan=False, ensure_ascii=True)
        append_jsonl(self.paths.tools_log, {
            "time": utc_now(),
            "request_id": request_id,
            "tool": name,
            "arguments": arguments,
            "is_error": outcome.is_error,
            "result": outcome.payload,
            "duration_s": round(time.monotonic() - started, 6),
            "meta": outcome.meta,
        })
        return _response(request_id, result={"content": [{"type": "text", "text": text}], "isError": outcome.is_error})


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def serve(session: McpSession, source: BinaryIO, out_fd: int, stop: dict[str, bool]) -> str:
    """Serve requests until end of input or a stop signal; return why the loop ended."""
    while True:
        if stop["pending"]:
            return "signal"
        try:
            session.busy = False
            line = source.readline(MAX_LINE_BYTES + 1)
        except ServerStop:
            return "signal"
        session.busy = True
        if not line:
            return "eof"
        if len(line) > MAX_LINE_BYTES:
            while line and not line.endswith(b"\n"):  # drop the rest of the oversized line, a bounded chunk at a time
                line = source.readline(MAX_LINE_BYTES)
            reply: dict[str, Any] | None = _response(None, error=(INVALID_REQUEST, "request line too long"))
        elif not line.strip():
            continue
        else:
            reply = session.handle_line(line)
        if reply is not None:
            _write_all(out_fd, (json.dumps(reply, allow_nan=False, ensure_ascii=True) + "\n").encode("ascii"))


def _lock(paths: SessionPaths) -> int:
    fd = os.open(paths.server_lock, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise
    return fd


def main(argv: list[str] | None = None) -> int:
    """Run the server for one session.json; the exit status is 0 on end of input or a stop signal."""
    parser = argparse.ArgumentParser(prog="python -m weightcode.harness.mcp_server")
    parser.add_argument("--session", required=True, type=Path, help="session.json of the session")
    args = parser.parse_args(argv)
    session = read_json(args.session)
    if not isinstance(session, dict) or session.get("schema") != SESSION_SCHEMA:
        raise SystemExit(f"{args.session} is not a {SESSION_SCHEMA} file")
    paths = SessionPaths(Path(str(session["paths"]["root"])))
    if paths.session_json.resolve() != args.session.resolve():
        raise SystemExit(f"{args.session} is not the session.json of the session it names")
    try:
        lock_fd = _lock(paths)
    except BlockingIOError:
        append_jsonl(paths.server_log, {"time": utc_now(), "event": "refused_second_server", "pid": os.getpid()})
        return EXIT_LOCKED
    protocol_out = os.dup(1)
    os.set_inheritable(protocol_out, False)
    log_fd = os.open(paths.server_stderr, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    sys.stdout = open(1, "w", encoding="utf-8", errors="replace", closefd=False, buffering=1)
    sys.stderr = open(2, "w", encoding="utf-8", errors="replace", closefd=False, buffering=1)
    paths.server_pid.write_text(f"{os.getpid()}\n", encoding="ascii")
    stop = {"pending": False}
    tools: ToolServer | None = None
    mcp: McpSession | None = None

    def on_signal(signum: int, _frame: object) -> None:
        stop["pending"] = True
        if mcp is None or not mcp.busy:
            raise ServerStop(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    status = 0
    reason = "unknown"
    try:
        problems = thread_problems(session, os.environ)
        if problems:
            append_jsonl(paths.server_log, {"time": utc_now(), "event": "fault", "pid": os.getpid(),
                                            "error": "thread variables differ from session.json", "detail": problems})
            return EXIT_FAULT
        tools = ToolServer(session)
        tools.start()
        mcp = McpSession(tools, paths)
        mcp.log_event("start", tools=list(tools.tools), replayed_checks=len(tools.ledger.entries),
                      queries_remaining=tools.queries_remaining, cells_remaining=tools.kernel.cells_remaining,
                      threads={name: os.environ.get(name) for name in THREAD_VARIABLES})
        reason = serve(mcp, sys.stdin.buffer, protocol_out, stop)
    except ServerStop:
        reason = "signal"
    except HarnessFault as exc:
        reason = f"fault: {exc}"
        status = EXIT_FAULT
    except Exception as exc:
        reason = f"fault: {type(exc).__name__}: {exc}"
        status = EXIT_FAULT
        append_jsonl(paths.server_log, {"time": utc_now(), "event": "fault", "pid": os.getpid(),
                                        "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if tools is not None:
            tools.close()
        append_jsonl(paths.server_log, {"time": utc_now(), "event": "stop", "pid": os.getpid(), "reason": reason,
                                        "status": status})
        os.close(protocol_out)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    return status


if __name__ == "__main__":
    sys.exit(main())
