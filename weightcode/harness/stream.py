"""Reading the editor's stream-json transcript (HARNESS_SPEC section 5).

``parse_stream`` extracts what session_result.json records: the init message (model id, Claude Code version, tools,
MCP server status), the result message (exit subtype, API error status, wall and API time, token usage), the first
and last ``rate_limit_event`` (five-hour and seven-day utilization and status) and the tool calls in order. The
transcript of a killed process can end in a partial line; such lines are counted, never parsed by guesswork.

``infra_reasons`` classifies a session as INFRA (PREREG section 5: rerun once) from the stream and the harness's own
state, never from patch-check numbers.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from weightcode.harness.arms import MCP_SERVER_NAME
from weightcode.harness.jsonio import iter_lines

RATE_LIMIT_REJECTED = "rejected"
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


@dataclass
class StreamSummary:
    """What one stream-json transcript says about the session."""

    n_lines: int = 0
    n_unparsed: int = 0
    init: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    rate_limit_first: dict[str, Any] | None = None
    rate_limit_last: dict[str, Any] | None = None
    n_rate_limit_events: int = 0
    tool_calls: list[str] = field(default_factory=list)
    n_tool_errors: int = 0
    n_assistant_messages: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_lines": self.n_lines,
            "n_unparsed": self.n_unparsed,
            "init": init_view(self.init),
            "result": result_view(self.result),
            "rate_limit_first": rate_limit_view(self.rate_limit_first),
            "rate_limit_last": rate_limit_view(self.rate_limit_last),
            "n_rate_limit_events": self.n_rate_limit_events,
            "tool_calls": dict(sorted(Counter(self.tool_calls).items())),
            "tool_call_order": list(self.tool_calls),
            "n_tool_calls": len(self.tool_calls),
            "n_tool_errors": self.n_tool_errors,
            "n_assistant_messages": self.n_assistant_messages,
        }


def parse_stream(path: str | Path) -> StreamSummary:
    """Summarise a stream-json transcript; a missing file is an empty transcript."""
    summary = StreamSummary()
    for line in iter_lines(path):
        if not line.strip():
            continue
        summary.n_lines += 1
        try:
            message = json.loads(line)
        except ValueError:
            summary.n_unparsed += 1
            continue
        if not isinstance(message, dict):
            summary.n_unparsed += 1
            continue
        kind = message.get("type")
        if kind == "system" and message.get("subtype") == "init" and summary.init is None:
            summary.init = message
        elif kind == "result":
            summary.result = message
        elif kind == "rate_limit_event":
            summary.n_rate_limit_events += 1
            if summary.rate_limit_first is None:
                summary.rate_limit_first = message
            summary.rate_limit_last = message
        elif kind == "assistant":
            summary.n_assistant_messages += 1
            for item in _content(message):
                if item.get("type") == "tool_use" and isinstance(item.get("name"), str):
                    summary.tool_calls.append(item["name"])
        elif kind == "user":
            for item in _content(message):
                if item.get("type") == "tool_result" and item.get("is_error") is True:
                    summary.n_tool_errors += 1
    return summary


def _content(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    body = message.get("message")
    content = body.get("content") if isinstance(body, dict) else None
    return [item for item in content if isinstance(item, dict)] if isinstance(content, list) else []


def init_view(init: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Fields of the init message that session_result.json keeps."""
    if init is None:
        return None
    return {key: init.get(key) for key in ("model", "claude_code_version", "tools", "mcp_servers", "permissionMode",
                                           "apiKeySource", "session_id", "cwd")}


def result_view(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Fields of the result message that session_result.json keeps."""
    if result is None:
        return None
    view = {key: result.get(key) for key in ("subtype", "is_error", "api_error_status", "terminal_reason",
                                             "stop_reason", "duration_ms", "duration_api_ms", "num_turns",
                                             "total_cost_usd", "session_id")}
    view["usage"] = usage_view(result.get("usage"))
    view["model_usage"] = result.get("modelUsage")
    denials = result.get("permission_denials")
    view["n_permission_denials"] = len(denials) if isinstance(denials, list) else None
    return view


def usage_view(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    view = {key: usage.get(key) for key in USAGE_KEYS}
    details = usage.get("output_tokens_details")
    view["thinking_tokens"] = details.get("thinking_tokens") if isinstance(details, dict) else None
    return view


def rate_limit_view(event: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Status and five-hour and seven-day utilization of one rate_limit_event."""
    if event is None:
        return None
    info = event.get("rate_limit_info")
    info = info if isinstance(info, dict) else {}
    windows = info.get("unifiedWindows")
    windows = windows if isinstance(windows, dict) else {}
    view: dict[str, Any] = {key: info.get(key) for key in ("status", "rateLimitType", "resetsAt", "isUsingOverage",
                                                           "overageStatus")}
    for name in ("five_hour", "seven_day"):
        window = windows.get(name)
        view[name] = ({key: window.get(key) for key in ("utilization", "status", "resetsAt")}
                      if isinstance(window, dict) else None)
    return view


def rate_limit_rejected(summary: StreamSummary) -> bool:
    """True when the last rate-limit event of the stream is a rejection."""
    view = rate_limit_view(summary.rate_limit_last)
    return view is not None and view["status"] == RATE_LIMIT_REJECTED


def infra_reasons(
    summary: StreamSummary,
    *,
    harness_errors: Sequence[str] = (),
    stream_expected: bool = True,
    killed_by_harness: bool = False,
    exit_code: int | None = None,
) -> list[str]:
    """Why a session is INFRA (PREREG section 5), or an empty list.

    INFRA when the editor process ends on a subscription rate-limit rejection, a non-null ``api_error_status``, or a
    harness crash outside the editor's control: a harness exception (``harness_errors``: launcher errors and MCP
    server faults), an MCP server that was not connected at startup, or a Claude Code process that ended without a
    result message although the harness did not kill it. ``stream_expected`` is False for a scripted session, which
    has no Claude Code process and no transcript; only harness errors apply to it.
    """
    reasons = [f"harness_exception: {error}" for error in harness_errors]
    if not stream_expected:
        return reasons
    if rate_limit_rejected(summary):
        reasons.append("rate_limit_rejected")
    if summary.result is not None and summary.result.get("api_error_status") is not None:
        reasons.append(f"api_error_status: {summary.result.get('api_error_status')}")
    if summary.init is not None:
        servers = summary.init.get("mcp_servers")
        states = {item.get("name"): item.get("status") for item in servers if isinstance(item, dict)} \
            if isinstance(servers, list) else {}
        if states.get(MCP_SERVER_NAME) != "connected":
            reasons.append(f"mcp_server_not_connected: {states.get(MCP_SERVER_NAME)!r}")
    if summary.result is None and not killed_by_harness:
        reasons.append(f"claude_exit_without_result: exit code {exit_code}")
    return reasons


__all__ = ["StreamSummary", "infra_reasons", "init_view", "parse_stream", "rate_limit_rejected", "rate_limit_view",
           "result_view"]
