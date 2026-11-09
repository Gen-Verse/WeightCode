"""T00 session table (scripts/t00/session_table.py) on a synthetic store with known answers.

Two sessions: SID_DONE ended normally (CLI result with modelUsage, a learn readout, a final program), and
SID_ABORT has no session_result.json (transcript only, with a message id repeated across content blocks).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "t00", Path(__file__).resolve().parents[1] / "scripts" / "t00" / "session_table.py")
t00 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t00)

SID_DONE = "s00000000000000d1"
SID_ABORT = "s00000000000000a1"
MODEL = "claude-opus-5-5"
HEADING = "## TOY, a synthetic experiment"
# Token counts of a real-sized session; at list price ($4 in, $20 out, $0.20 cache read, $8 cache write 1 h per
# MTok): 142*4 + 262158*20 + 11972915*0.20 + 313274*8 = 10144503 micro-dollars.
DONE_TOKENS = {"input": 142, "output": 262158, "cache_read": 11972915, "cache_write_5m": 0, "cache_write_1h": 313274}
DONE_USD = 10.144503


def _w(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _wl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _assistant(mid: str, inp: int, cr: int, cw1h: int, out: int) -> dict:
    return {"type": "assistant", "message": {"id": mid, "model": MODEL, "usage": {
        "input_tokens": inp, "cache_read_input_tokens": cr, "cache_creation_input_tokens": cw1h,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": cw1h},
        "output_tokens": out}}}


def _session_json(created: str) -> dict:
    return {"created_at": created, "instance_id": "itoy", "condition": "toy_s3", "arm": "LEARN",
            "model_id": "mtoy", "seed": 100, "real_model_sha256": "r" * 64,
            "budget": {"patch_checks": 8, "python_cells": 150, "wall_timeout_s": 5400},
            "program": {"clock": "virtual-v1", "driver_sha256": "d" * 64},
            "editor": {"model": MODEL, "effort": "high"}, "sandbox": {"memory_bytes": 8589934592},
            "provenance": {"git": {"head": "abc1234", "dirty": False}}}


@pytest.fixture()
def store(tmp_path: Path) -> dict[str, Path]:
    runs, private = tmp_path / "runs", tmp_path / "private"
    # completed session
    d = runs / "sessions" / SID_DONE
    _w(d / "session.json", _session_json("2026-11-01T00:00:00+00:00"))
    usage = {MODEL: {"inputTokens": DONE_TOKENS["input"], "outputTokens": DONE_TOKENS["output"],
                     "cacheReadInputTokens": DONE_TOKENS["cache_read"],
                     "cacheCreationInputTokens": DONE_TOKENS["cache_write_1h"],
                     "cacheCreationEphemeral5mInputTokens": 0,
                     "cacheCreationEphemeral1hInputTokens": DONE_TOKENS["cache_write_1h"],
                     "thinkingTokens": 200000, "costBasis": "list"}}
    _w(d / "session_result.json", {
        "status": "complete", "infra": False, "launch": {"exit_code": 0}, "queries": {"used": 3},
        "cells": {"used": 40, "by_status": {"ok": 40}, "restarts": 0}, "wall_s": 1800.0,
        "final": {"program": {"sha256": "p" * 64}, "child_sha256": "c" * 64, "is_base_noop": False,
                  "n_changed": 5007, "kind": "program"},
        "audit": {"verdict": "CLEAN"}, "code_changed_during_session": [],
        "stream": {"result": {"model_usage": usage, "total_cost_usd": DONE_USD, "duration_api_ms": 1500000,
                              "num_turns": 60}}})
    _wl(d / "logs" / "transcript.jsonl", [
        _assistant("m1", 100, 10000000, 300000, 10),
        _assistant("m1", 100, 10000000, 300000, 200000),
        _assistant("m2", 42, 1972915, 13274, 62158),
        {"type": "result", "modelUsage": usage, "total_cost_usd": DONE_USD}])
    _wl(d / "logs" / "checks.jsonl", [{"i": 1}, {"i": 2}, {"i": 3}])
    _w(d / "audit.json", {"verdict": "CLEAN"})
    _w(d / "final" / "program_run.json", {"wall_s": 600, "run": {"duration_s": 12.5,
                                                                  "result": {"update_s": 10.0, "clock_reads": 0}}})
    w = {"conv1.weight": np.zeros((4, 3, 3, 3), dtype=np.float32), "fc.bias": np.zeros(10, dtype=np.float32)}
    (d / "trial").mkdir(parents=True)
    np.savez(d / "trial" / "model.npz", **w)
    np.savez(d / "final" / "model.npz", **w)
    _w(runs / "devreadouts" / "learn" / SID_DONE / "info.json", {"final": {
        "vselect_new": {"base_acc": 0.4475, "child_acc": 0.7165},
        "vselect_clean": {"base_acc": 0.7838, "child_acc": 0.7835}}})
    # aborted session: transcript only; m1 appears on two lines (two content blocks), output partial on the first
    a = runs / "sessions" / SID_ABORT
    _w(a / "session.json", _session_json("2026-11-02T00:00:00+00:00"))
    _wl(a / "logs" / "transcript.jsonl", [
        _assistant("m1", 10, 100, 50, 5),
        _assistant("m1", 10, 100, 50, 20),
        {"type": "user", "message": {"content": "tool result"}},
        _assistant("m2", 3, 200, 0, 7)])
    _wl(a / "logs" / "cells.jsonl", [{"index": 1, "timestamp": "2026-11-02T00:10:00+00:00"}])
    _w(a / "logs" / "cells.jsonl.inflight", {"index": 2, "timestamp": "2026-11-02T00:25:00+00:00"})
    # store-level records
    _wl(runs / "manifest.jsonl", [{"session_id": SID_DONE, "status": "complete", "audit_verdict": "CLEAN"}])
    _w(private / "instances" / "index.json", {"instances": [{"instance_id": "itoy"}], "superseded": []})
    dev = tmp_path / "DEV_RESULTS.md"
    dev.write_text(f"# Results\n\n{HEADING}\n\nSession {SID_DONE}: supported. Session {SID_ABORT}: oomd kill.\n")
    readings = tmp_path / "session_readings.json"
    _w(readings, {"schema": "weightcode.t00.session_readings.v1", "sessions": {
        SID_DONE: {"experiment": "TOY", "condition": "toy_s3", "bar": "joint >= 10", "met": True,
                   "met_scope": "session", "reading": "supported", "dev_results_heading": HEADING},
        SID_ABORT: {"experiment": "TOY", "condition": "toy_s3", "bar": None, "met": None,
                    "null_kind": "not_applicable", "met_scope": None, "reading": "killed before any check",
                    "reason": "Not a launched session: a stop before launch.", "dev_results_heading": HEADING}}})
    return {"runs": runs, "private": private, "dev": dev, "readings": readings, "root": tmp_path}


def _rows(store: dict[str, Path]) -> dict[str, dict]:
    rows, _, _, _ = t00.build(store["runs"], store["private"], store["readings"], store["dev"])
    return {r["session_id"]: r for r in rows}


def test_list_price_arithmetic() -> None:
    assert t00.list_price_usd(DONE_TOKENS, MODEL) == pytest.approx(DONE_USD, abs=1e-9)
    one_each = {"input": 10**6, "output": 10**6, "cache_read": 10**6, "cache_write_5m": 10**6, "cache_write_1h": 10**6}
    assert t00.list_price_usd(one_each, MODEL) == pytest.approx(4 + 20 + 0.20 + 5 + 8, abs=1e-9)
    with pytest.raises(t00.T00Error):
        t00.list_price_usd(DONE_TOKENS, "some-other-model")


def test_transcript_counts_each_message_once(store: dict[str, Path]) -> None:
    tr = t00.transcript_usage(store["runs"] / "sessions" / SID_ABORT / "logs" / "transcript.jsonl")
    assert tr["n_messages"] == 2
    assert tr["models"] == [MODEL]
    assert tr["result"] is None
    # input and cache counted once per message id; output is the largest value seen per id
    assert tr["sums"] == {"input": 13, "cache_read": 300, "cache_write_5m": 0, "cache_write_1h": 50,
                          "cache_write_total": 50, "output": 27}


def test_model_usage_tokens() -> None:
    mu = {MODEL: {"inputTokens": 1, "outputTokens": 2, "cacheReadInputTokens": 3, "cacheCreationInputTokens": 9,
                  "cacheCreationEphemeral5mInputTokens": 4, "cacheCreationEphemeral1hInputTokens": 5,
                  "thinkingTokens": 1}}
    assert t00.tokens_from_model_usage(mu)[MODEL] == {"input": 1, "output": 2, "cache_read": 3, "cache_write_5m": 4,
                                                      "cache_write_1h": 5, "thinking": 1}
    mu[MODEL]["cacheCreationInputTokens"] = 10
    with pytest.raises(t00.T00Error):
        t00.tokens_from_model_usage(mu)


def test_completed_session_row(store: dict[str, Path]) -> None:
    r = _rows(store)[SID_DONE]
    assert r["status"] == "complete"
    assert r["llm_tokens"] == {**DONE_TOKENS, "thinking": 200000}
    assert r["api_equiv_cost_usd"] == pytest.approx(DONE_USD, abs=1e-9)
    assert r["cli_total_cost_usd"] == DONE_USD
    assert r["cost_detail"]["cli_matches_list_price"] is True
    assert r["cost_detail"]["transcript_input_cache_sums_match"] is True
    assert r["checks_used"] == 3 and r["checks_budget"] == 8
    assert r["cells_used"] == 40 and r["cells_budget"] == 150
    assert r["n_changed"] == 5007
    assert (r["target_before"], r["target_after"]) == (0.4475, 0.7165)
    assert (r["old_before"], r["old_after"]) == (0.7838, 0.7835)
    assert r["success_by_frozen_rule"] is True and r["dev_results_line"] == 3
    assert r["program_clock"] == "virtual-v1" and r["program_clock_reads"] == 0
    assert r["execution_time_s"] == 12.5 and r["generation_time_s"] == 1800.0
    assert r["inference_cost"]["equal_to_base"] is True and r["inference_cost"]["n_parameters"] == 118
    # a field the records do not hold is null with a reason, not a guess
    assert r["peak_memory"] is None and "8589934592" in r["missing"]["peak_memory"]
    assert r["api_equiv_cost_usd_lower_bound"] is None and "api_equiv_cost_usd_lower_bound" in r["not_applicable"]


def test_aborted_session_row(store: dict[str, Path]) -> None:
    r = _rows(store)[SID_ABORT]
    assert r["status"] == "aborted"
    assert "no result event in logs/transcript.jsonl" in r["failure_reason"]
    assert "cell 2" in r["failure_reason"]
    assert r["llm_tokens"] == {"input": 13, "output": 27, "cache_read": 300, "cache_write_5m": 0,
                               "cache_write_1h": 50, "thinking": None}
    assert r["cost_detail"]["output_is_lower_bound"] is True
    assert r["api_equiv_cost_usd"] is None and "lower bound" in r["missing"]["api_equiv_cost_usd"]
    assert r["api_equiv_cost_usd_lower_bound"] == pytest.approx((13 * 4 + 27 * 20 + 300 * 0.2 + 50 * 8) / 1e6)
    assert "tokens.thinking" in r["missing"]
    assert r["cli_total_cost_usd"] is None and "cli_total_cost_usd" in r["missing"]
    assert r["generation_time_s"] is None and "generation_time_s" in r["missing"]
    assert r["generation_time_s_lower_bound"] == 1500.0
    for k in ("program_hash", "n_changed", "audit_verdict", "target_after", "old_after", "inference_cost"):
        assert r[k] is None and k in r["missing"], k
    assert r["checks_used"] == 0
    # no criterion applies to a stop before launch: null, recorded as not applicable
    assert r["success_by_frozen_rule"] is None and "success_by_frozen_rule" in r["not_applicable"]
    assert r["manifest"]["present"] is False


def test_every_null_has_one_reason(store: dict[str, Path]) -> None:
    for r in _rows(store).values():
        for k, v in r.items():
            if k in ("missing", "not_applicable"):
                continue
            if v is None:
                assert (k in r["missing"]) != (k in r["not_applicable"]), (r["session_id"], k)
            else:
                assert k not in r["missing"] and k not in r["not_applicable"], (r["session_id"], k)


def test_row_refuses_null_without_reason() -> None:
    row = t00.Row()
    with pytest.raises(t00.T00Error):
        row.set("x", None)
    row.set("x", None, "not recorded")
    row.set("x", 1)
    assert row.as_dict() == {"x": 1, "missing": {}, "not_applicable": {}}


def test_readings_must_match_sessions(store: dict[str, Path]) -> None:
    doc = json.loads(store["readings"].read_text())
    del doc["sessions"][SID_ABORT]
    store["readings"].write_text(json.dumps(doc))
    with pytest.raises(t00.T00Error, match="does not match"):
        _rows(store)


def test_null_reading_needs_kind(store: dict[str, Path]) -> None:
    doc = json.loads(store["readings"].read_text())
    del doc["sessions"][SID_ABORT]["null_kind"]
    store["readings"].write_text(json.dumps(doc))
    with pytest.raises(t00.T00Error, match="null_kind"):
        _rows(store)


def test_main_writes_both_outputs(store: dict[str, Path]) -> None:
    out = store["root"] / "out"
    assert t00.main(["--runs", str(store["runs"]), "--private", str(store["private"]), "--readings",
                     str(store["readings"]), "--dev-results", str(store["dev"]), "--out-dir", str(out)]) == 0
    lines = (out / "session_table.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(ln)["schema"] == t00.SCHEMA for ln in lines)
    md = (out / "session_table.md").read_text()
    assert "API-equivalent at list price, not spend" in md
    assert "| TOY | 2 | 1 / 1 |" in md
    assert chr(0x2014) not in md  # no em dash in the generated table
