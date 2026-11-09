"""Tests of scripts/run_session_chain.py with a fake runner (no Claude Code, no store)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_session_chain.py"
spec = importlib.util.spec_from_file_location("run_session_chain", SCRIPT)
assert spec is not None and spec.loader is not None
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)

FAKE_RUNNER = '''import json, os, sys, uuid
from pathlib import Path
queue = Path(os.environ["FAKE_QUEUE"])
outcomes = json.loads(queue.read_text())
outcome = outcomes.pop(0)
queue.write_text(json.dumps(outcomes))
args = sys.argv[1:]
Path(os.environ["FAKE_CALLS"]).open("a").write(" ".join(args) + "\\n")
if outcome.get("no_dir"):
    sys.exit(3)
root = Path(os.environ["WEIGHTCODE_RUNS"]) / "sessions" / ("s" + uuid.uuid4().hex[:16])
root.mkdir(parents=True)
inst, arm, rep = args[args.index("--instance") + 1], args[args.index("--arm") + 1], int(args[args.index("--repeat") + 1])
(root / "session.json").write_text(json.dumps({"instance_id": inst, "arm": arm, "repeat": rep}))
if outcome.get("no_result"):
    sys.exit(1)
event = {"seven_day": {"utilization": outcome["util"]}, "five_hour": None}
result = {"status": outcome.get("status", "complete"), "infra": outcome.get("infra", False), "infra_reasons": [],
          "final": {"score": 5, "is_base_noop": False}, "wall_s": 1.0, "tokens": {"output_tokens": 1},
          "stream": {"rate_limit_last": event}}
(root / "session_result.json").write_text(json.dumps(result))
sys.exit(outcome.get("exit", 0))
'''


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcomes: list[dict], specs: list[str],
         *extra: str) -> tuple[int, list[dict], list[str]]:
    runner = tmp_path / "fake_runner.py"
    runner.write_text(FAKE_RUNNER)
    queue, calls = tmp_path / "queue.json", tmp_path / "calls.txt"
    queue.write_text(json.dumps(outcomes))
    calls.write_text("")
    monkeypatch.setenv("WEIGHTCODE_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("FAKE_QUEUE", str(queue))
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    log = tmp_path / "chain" / "log.jsonl"
    code = chain.main([*specs, "--log", str(log), "--effort", "max", "--wall-timeout-min", "90",
                       "--runner", str(runner), *extra])
    records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return code, records, [line for line in calls.read_text().splitlines() if line]


def test_all_sessions_run_when_nothing_stops_the_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, records, calls = _run(tmp_path, monkeypatch, [{"util": 0.5}, {"util": 0.6}], ["ia:M2", "ib:DECOY:1"])
    assert code == 0 and [r["spec"] for r in records] == ["ia:M2:0", "ib:DECOY:1"]
    assert all(r["stop_reason"] is None and r["status"] == "complete" for r in records)
    assert calls[1].split() == ["--instance", "ib", "--arm", "DECOY", "--repeat", "1", "--effort", "max",
                                "--wall-timeout-min", "90.0"]
    assert records[1]["utilization"] == 0.6 and Path(records[0]["output"]).exists()


@pytest.mark.parametrize("first, reason", [({"util": 0.95}, "utilization 0.95 >= --stop-at 0.95"),
                                           ({"util": 0.2, "infra": True, "status": "infra"}, "INFRA"),
                                           ({"no_result": True}, "no session_result.json"),
                                           ({"no_dir": True}, "expected one new session directory for this spec")])
def test_the_chain_stops_before_the_next_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: dict,
                                                reason: str) -> None:
    code, records, calls = _run(tmp_path, monkeypatch, [first, {"util": 0.1}], ["ia:M2", "ib:M2"])
    assert code == 2 and len(records) == 1 and len(calls) == 1
    assert records[0]["stop_reason"].startswith(reason)


def test_a_failed_audit_exit_code_with_a_finalised_session_does_not_stop(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    code, records, calls = _run(tmp_path, monkeypatch, [{"util": 0.3, "exit": 1}, {"util": 0.4}], ["ia:M2", "ib:M2"])
    assert code == 0 and len(calls) == 2 and records[0]["exit_code"] == 1 and records[0]["stop_reason"] is None


def test_a_bad_spec_is_refused() -> None:
    with pytest.raises(SystemExit):
        chain.main(["ia", "--log", "x.jsonl", "--effort", "max", "--wall-timeout-min", "1"])
