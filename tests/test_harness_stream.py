"""Parsing Claude Code stream-json transcripts: rate limits, INFRA reasons and the session status (PREREG section 5).

``RECORDED`` is a trimmed copy of the transcript of the 2026-11-03 M2 smoke session (Claude Code 2.1.311): the init
line, two tool calls and one tool result with their content removed, the one rate-limit event and the result line.
Session ids were dropped; nothing else was edited.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from weightcode.harness.finalize import (KEEPS_FINAL_PATCH, STATUS_COMPLETE, STATUS_CRASHED, STATUS_INFRA, STATUS_TIMED_OUT,
                                   classify)
from weightcode.harness.launch import KILL_REASON_WALL, LaunchOutcome
from weightcode.harness.stream import infra_reasons, parse_stream

RECORDED: list[dict[str, Any]] = [
    {"type": "system", "subtype": "init", "model": "claude-opus-5-5", "claude_code_version": "2.1.311",
     "tools": ["mcp__weightcode__declare_final", "mcp__weightcode__patch_check", "mcp__weightcode__python", "mcp__weightcode__status"],
     "mcp_servers": [{"name": "weightcode", "status": "connected", "source": "dynamic"}], "permissionMode": "default",
     "apiKeySource": "none"},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "toolu_01JymJcc3JEtbajCBzfKA3wZ",
                                                   "name": "mcp__weightcode__python", "input": {}}]}},
    {"type": "rate_limit_event", "rate_limit_info": {
        "status": "allowed_warning", "resetsAt": 1793736000, "rateLimitType": "seven_day", "utilization": 0.82,
        "overageStatus": "rejected", "overageDisabledReason": "org_level_disabled", "isUsingOverage": False,
        "surpassedThreshold": 0.75,
        "unifiedWindows": {"seven_day": {"utilization": 0.82, "resetsAt": 1793736000, "status": "allowed_warning",
                                         "surpassedThreshold": 0.75}},
        "overageEnabled": False}},
    {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_01JymJcc3JEtbajCBzfKA3wZ",
                                              "is_error": None, "content": "trimmed"}]}},
    {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "toolu_0168Xu3gyTMycSWMjiCtJfdS",
                                                   "name": "mcp__weightcode__python", "input": {}}]}},
    {"type": "result", "subtype": "success", "is_error": False, "api_error_status": None, "duration_ms": 44420,
     "duration_api_ms": 39505, "num_turns": 8, "stop_reason": "end_turn", "total_cost_usd": 0.157107,
     "usage": {"input_tokens": 16, "cache_creation_input_tokens": 11217, "cache_read_input_tokens": 65235,
               "output_tokens": 2713, "output_tokens_details": {"thinking_tokens": 506},
               "service_tier": "standard"},
     "result": "trimmed"},
]


def _write(tmp_path: Path, messages: list[dict[str, Any]], extra_lines: tuple[str, ...] = ()) -> Path:
    path = tmp_path / "transcript.jsonl"
    path.write_text("".join(json.dumps(m) + "\n" for m in messages) + "".join(f"{line}\n" for line in extra_lines))
    return path


def _launch(**fields: Any) -> LaunchOutcome:
    return LaunchOutcome(launcher="claude", stream_expected=True, started_at="2026-11-03T00:00:00+00:00",
                         **{"exit_code": 0, **fields})


def test_the_recorded_transcript_parses_and_is_a_complete_session(tmp_path: Path) -> None:
    summary = parse_stream(_write(tmp_path, RECORDED, ("not json",)))
    view = summary.as_dict()
    assert view["n_lines"] == 7 and view["n_unparsed"] == 1
    assert view["init"]["model"] == "claude-opus-5-5" and view["init"]["claude_code_version"] == "2.1.311"
    assert view["tool_calls"] == {"mcp__weightcode__python": 2} and view["n_tool_errors"] == 0
    assert view["n_rate_limit_events"] == 1
    rate = view["rate_limit_last"]
    assert rate["status"] == "allowed_warning" and rate["seven_day"]["utilization"] == 0.82 and rate["five_hour"] is None
    assert view["result"]["usage"]["thinking_tokens"] == 506 and view["result"]["usage"]["output_tokens"] == 2713
    assert infra_reasons(summary, exit_code=0) == []
    assert classify(_launch(), summary, []) == (STATUS_COMPLETE, [])


def _variant(change: str) -> list[dict[str, Any]]:
    messages = copy.deepcopy(RECORDED)
    if change == "rejected":
        messages[2]["rate_limit_info"]["status"] = "rejected"
    elif change == "api_error":
        messages[5]["api_error_status"] = 529
        messages[5]["is_error"] = True
    elif change == "server_failed":
        messages[0]["mcp_servers"][0]["status"] = "failed"
    elif change == "no_result":
        messages = messages[:5]
    return messages


def test_infra_reasons_of_each_kind(tmp_path: Path) -> None:
    expected = {
        "rejected": "rate_limit_rejected",
        "api_error": "api_error_status: 529",
        "server_failed": "mcp_server_not_connected: 'failed'",
        "no_result": "claude_exit_without_result: exit code 1",
    }
    for change, reason in expected.items():
        summary = parse_stream(_write(tmp_path, _variant(change)))
        status, reasons = classify(_launch(exit_code=1 if change == "no_result" else 0), summary, [])
        assert status == STATUS_INFRA and reason in reasons, (change, reasons)


def test_a_harness_fault_is_infra_and_a_wall_timeout_is_not(tmp_path: Path) -> None:
    summary = parse_stream(_write(tmp_path, _variant("no_result")))
    status, reasons = classify(_launch(exit_code=-15, killed=True, kill_reason=KILL_REASON_WALL), summary, [])
    assert (status, reasons) == (STATUS_TIMED_OUT, [])
    status, reasons = classify(_launch(), parse_stream(_write(tmp_path, RECORDED)), ["mcp server: ValueError: x"])
    assert status == STATUS_INFRA and reasons == ["harness_exception: mcp server: ValueError: x"]


def test_an_error_result_without_an_api_error_is_a_crash(tmp_path: Path) -> None:
    messages = copy.deepcopy(RECORDED)
    messages[5].update(subtype="error_during_execution", is_error=True)
    assert classify(_launch(exit_code=1), parse_stream(_write(tmp_path, messages)), []) == (STATUS_CRASHED, [])


def test_a_missing_transcript_is_empty(tmp_path: Path) -> None:
    summary = parse_stream(tmp_path / "absent.jsonl")
    assert summary.n_lines == 0 and summary.result is None
    assert infra_reasons(summary, stream_expected=False) == []


def test_complete_timed_out_and_crashed_sessions_keep_the_rule_final() -> None:
    """PREREG v0.3 section 5 and contract 5.2: only an INFRA session (rerun once) keeps the base model."""
    assert KEEPS_FINAL_PATCH == {STATUS_COMPLETE, STATUS_TIMED_OUT, STATUS_CRASHED}
    assert STATUS_INFRA not in KEEPS_FINAL_PATCH
