"""The LLM-judge step (weightcode/audit/judge.py, scripts/audit_judge.py) with a stand-in executable.

The real claude CLI is never run here: every test that reaches the CLI passes the path of a small stand-in script
that records its arguments and stdin and replies in the CLI's JSON output format.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from weightcode.audit import judge as J
from weightcode.audit import report as R
from weightcode.audit.session import audit_session, read_cells, write_report

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_judge.py"

LOAD = '''import numpy as np
ev = np.load("/work/trial/evidence.npz")
xr, yr = ev["x_rand"], ev["y_rand"]
H = xr.reshape(len(xr), -1) / 255.0
'''
TORCH = "try:\n    import torch\nexcept ImportError:\n    torch = None\n"
LSTSQ = "W, *_ = np.linalg.lstsq(H, np.eye(10)[yr], rcond=None)\n"

STAND_IN = r'''#!{python}
import json, os, re, sys, time
from pathlib import Path

here = Path(__file__).resolve().parent
args = sys.argv[1:]
if args == ["--version"]:
    print("0.0.0 (stand-in for the claude CLI)")
    sys.exit(0)
(here / "argv.json").write_text(json.dumps(args))
null, stdin = os.stat(os.devnull), os.fstat(0)
(here / "stdin.txt").write_text("devnull" if (stdin.st_rdev, stdin.st_ino) == (null.st_rdev, null.st_ino) else "other")
mode = (here / "mode.txt").read_text().strip()
prompt = args[args.index("-p") + 1]
listing = prompt.split("## SOFT flags to decide", 1)[1].split("## Executed cells", 1)[0]
ids = [flag["flag_id"] for flag in json.loads(listing)]
if mode == "exit3":
    sys.exit(3)
if mode == "sleep":
    time.sleep(30)
decision = "VIOLATION" if mode == "violation" else "ALLOWED"
reply = json.dumps({{"decisions": [{{"flag_id": i, "decision": decision, "confidence": "high",
                                    "clause": "imports", "reason": "stand-in"}} for i in ids]}})
if mode == "garbage":
    reply = "I think this is fine."
if mode == "fenced":
    reply = "```json\n" + reply + "\n```"
envelope = {{"type": "result", "subtype": "success", "is_error": mode == "cli_error", "num_turns": 1,
            "result": reply, "modelUsage": {{"stand-in-model": {{}}}}, "session_id": "s"}}
print(json.dumps(envelope))
'''


@pytest.fixture
def stand_in(tmp_path: Path) -> Path:
    directory = tmp_path / "bin"
    directory.mkdir()
    path = directory / "claude"
    path.write_text(STAND_IN.format(python=sys.executable), encoding="utf-8")
    path.chmod(0o755)
    (directory / "mode.txt").write_text("allowed", encoding="utf-8")
    return path


def set_mode(stand_in: Path, mode: str) -> None:
    (stand_in.parent / "mode.txt").write_text(mode, encoding="utf-8")


def make_session(root: Path, *cells: str) -> Path:
    logs = root / "logs"
    logs.mkdir(parents=True)
    lines = [json.dumps({"index": i, "code": code}) for i, code in enumerate(cells)]
    (logs / "cells.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    write_report(audit_session(logs / "cells.jsonl"), logs / "audit.json")
    return root


def load_report(session: Path) -> dict[str, object]:
    return json.loads((session / "logs" / "audit.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------- prompt and parsing


def test_judge_command_has_no_tools_settings_or_mcp() -> None:
    assert J.judge_command("claude", "PROMPT") == [
        "claude", "-p", "PROMPT", "--tools", "", "--setting-sources", "", "--strict-mcp-config",
        "--disable-slash-commands", "--no-session-persistence", "--output-format", "json"]
    assert J.judge_command("c", "P", model="m", effort="high")[-4:] == ["--model", "m", "--effort", "high"]


def test_template_has_each_placeholder_once_and_no_superseded_pair() -> None:
    text = J.template_text()
    for placeholder in J.PLACEHOLDERS:
        assert text.count(placeholder) == 1
    assert "class A" not in text and "predicts as B" not in text
    assert "y_fail" in text and "pred_" in text and "tie" in text


def test_build_prompt_shows_flags_and_cells(tmp_path: Path) -> None:
    session = make_session(tmp_path / "s", LOAD, TORCH)
    report = load_report(session)
    prompt, shown = J.build_prompt(report, read_cells(session / "logs" / "cells.jsonl"))
    assert not any(placeholder in prompt for placeholder in J.PLACEHOLDERS)
    assert '"flag_id": "F001"' in prompt and R.S_IMPORT in prompt
    assert "### cell 1 (status None)" in prompt and "import torch" in prompt
    assert shown == {"whole": [0, 1], "clipped": [], "omitted": []}


def test_build_prompt_clips_a_long_flagged_cell_and_omits_others(tmp_path: Path) -> None:
    filler = "".join(f"a{i} = {i}\n" for i in range(3000))
    session = make_session(tmp_path / "s", LOAD + filler, filler + TORCH + filler)
    report = load_report(session)
    cells = read_cells(session / "logs" / "cells.jsonl")
    base = len(J.build_prompt(report, cells[:0])[0].encode())
    prompt, shown = J.build_prompt(report, cells, byte_limit=base + 9000)
    assert shown == {"whole": [], "clipped": [1], "omitted": [0]}
    assert "import torch" in prompt and "[lines omitted]" in prompt and "Cells omitted for length: [0]" in prompt
    with pytest.raises(J.JudgeError, match="exceed"):
        J.build_prompt(report, cells, byte_limit=1000)


@pytest.mark.parametrize("reply, message", [
    ("not json", "not one JSON object"),
    ("[1]", "not a JSON object"),
    ('{"x": 1}', "no decisions list"),
    ('{"decisions": [1]}', "not a JSON object"),
    ('{"decisions": [{"flag_id": "F009", "decision": "ALLOWED", "confidence": "high"}]}', "unknown flag id"),
    ('{"decisions": [{"flag_id": "F001", "decision": "OK", "confidence": "high"}]}', "is not one of"),
    ('{"decisions": [{"flag_id": "F001", "decision": "ALLOWED", "confidence": "sure"}]}', "is not one of"),
    ('{"decisions": [{"flag_id": "F001", "decision": "ALLOWED", "confidence": "high", "reason": 3}]}', "strings"),
    ('{"decisions": [{"flag_id": "F001", "decision": "ALLOWED", "confidence": "high"},'
     ' {"flag_id": "F001", "decision": "ALLOWED", "confidence": "high"}]}', "two decisions"),
    ('{"decisions": [{"flag_id": "F001", "decision": "ALLOWED", "confidence": "high"}]}', "no decision for"),
])
def test_parse_decisions_rejects_malformed_replies(reply: str, message: str) -> None:
    with pytest.raises(J.JudgeError, match=message):
        J.parse_decisions(reply, ["F001", "F002"])


def test_parse_decisions_accepts_a_fenced_object_and_keeps_flag_order() -> None:
    reply = ('```json\n{"decisions": [{"flag_id": "F002", "decision": "VIOLATION", "confidence": "low"},'
             ' {"flag_id": "F001", "decision": "ALLOWED", "confidence": "high", "clause": "c", "reason": "r"}]}\n```')
    decisions = J.parse_decisions(reply, ["F001", "F002"])
    assert [(d["flag_id"], d["decision"]) for d in decisions] == [("F001", "ALLOWED"), ("F002", "VIOLATION")]


# --------------------------------------------------------------------------------------------- run_judge


def test_clean_report_needs_no_judge(tmp_path: Path) -> None:
    session = make_session(tmp_path / "s", LOAD)
    record = J.run_judge(session, claude_bin=str(tmp_path / "missing-claude"))
    assert record["status"] == "not_needed" and record["final_verdict"] == R.CLEAN
    assert json.loads((session / "logs" / J.JUDGE_FILENAME).read_text())["status"] == "not_needed"


def test_soft_flags_resolved_allowed(tmp_path: Path, stand_in: Path) -> None:
    session = make_session(tmp_path / "s", LOAD, TORCH)
    record = J.run_judge(session, claude_bin=str(stand_in), model="judge-model")
    assert record["status"] == "ok" and record["judge_verdict"] == "ALLOWED" and record["final_verdict"] == "ALLOWED"
    assert [d["flag_id"] for d in record["decisions"]] == ["F001"]  # type: ignore[index]
    argv = json.loads((stand_in.parent / "argv.json").read_text())
    assert argv[0] == "-p" and argv[2:] == ["--tools", "", "--setting-sources", "", "--strict-mcp-config",
                                            "--disable-slash-commands", "--no-session-persistence",
                                            "--output-format", "json", "--model", "judge-model"]
    assert (stand_in.parent / "stdin.txt").read_text() == "devnull"
    prompt = (session / "logs" / J.PROMPT_FILENAME).read_text(encoding="utf-8")
    assert argv[1] == prompt and "<prompt>" in record["command"]  # type: ignore[operator]
    assert record["claude_version"] == "0.0.0 (stand-in for the claude CLI)"
    assert record["models_used"] == ["stand-in-model"]
    assert record["template_sha256"] == J.template_sha256()


def test_judge_violation_decision_makes_the_final_verdict_violation(tmp_path: Path, stand_in: Path) -> None:
    session = make_session(tmp_path / "s", LOAD, TORCH)
    set_mode(stand_in, "violation")
    record = J.run_judge(session, claude_bin=str(stand_in))
    assert record["judge_verdict"] == R.VIOLATION and record["final_verdict"] == R.VIOLATION


def test_static_violation_stays_final(tmp_path: Path, stand_in: Path) -> None:
    session = make_session(tmp_path / "s", LOAD, TORCH, LSTSQ)
    assert load_report(session)["verdict"] == R.VIOLATION
    record = J.run_judge(session, claude_bin=str(stand_in))
    assert record["status"] == "not_needed" and record["final_verdict"] == R.VIOLATION
    assert not (stand_in.parent / "argv.json").exists()
    record = J.run_judge(session, claude_bin=str(stand_in), force=True)
    assert record["status"] == "ok" and record["judge_verdict"] == "ALLOWED"
    assert record["final_verdict"] == R.VIOLATION


@pytest.mark.parametrize("mode, message", [
    ("garbage", "not one JSON object"),
    ("exit3", "exited with status 3"),
    ("cli_error", "reported an error"),
])
def test_failed_judge_writes_a_failed_record(tmp_path: Path, stand_in: Path, mode: str, message: str) -> None:
    session = make_session(tmp_path / "s", LOAD, TORCH)
    set_mode(stand_in, mode)
    with pytest.raises(J.JudgeFailed, match=message) as raised:
        J.run_judge(session, claude_bin=str(stand_in))
    record = json.loads(raised.value.record_path.read_text())
    assert record["status"] == "failed" and record["final_verdict"] is None and message in record["error"]


def test_fenced_reply_is_accepted(tmp_path: Path, stand_in: Path) -> None:
    session = make_session(tmp_path / "s", LOAD, TORCH)
    set_mode(stand_in, "fenced")
    assert J.run_judge(session, claude_bin=str(stand_in))["status"] == "ok"


def test_judge_timeout_kills_the_process(tmp_path: Path, stand_in: Path) -> None:
    session = make_session(tmp_path / "s", LOAD, TORCH)
    set_mode(stand_in, "sleep")
    with pytest.raises(J.JudgeFailed, match="did not finish"):
        J.run_judge(session, claude_bin=str(stand_in), timeout_s=1.0)
    assert json.loads((session / "logs" / J.JUDGE_FILENAME).read_text())["timed_out"] is True


def test_missing_cli_writes_a_failed_record(tmp_path: Path) -> None:
    session = make_session(tmp_path / "s", LOAD, TORCH)
    with pytest.raises(J.JudgeFailed, match="--version failed"):
        J.run_judge(session, claude_bin=str(tmp_path / "missing-claude"))


def test_stale_or_foreign_report_is_refused_without_a_record(tmp_path: Path, stand_in: Path) -> None:
    session = make_session(tmp_path / "s", LOAD, TORCH)
    cells = session / "logs" / "cells.jsonl"
    cells.write_text(cells.read_text() + json.dumps({"index": 2, "code": "x = 1\n"}) + "\n")
    with pytest.raises(J.JudgeError, match="does not match the current"):
        J.run_judge(session, claude_bin=str(stand_in))
    other = make_session(tmp_path / "t", LOAD, TORCH)
    report = load_report(other)
    report["auditor_sha256"] = "0" * 64
    (other / "logs" / "audit.json").write_text(json.dumps(report))
    with pytest.raises(J.JudgeError, match="different audit implementation"):
        J.run_judge(other, claude_bin=str(stand_in))
    assert not (session / "logs" / J.JUDGE_FILENAME).exists() and not (other / "logs" / J.JUDGE_FILENAME).exists()
    with pytest.raises(J.JudgeError, match="does not exist"):
        J.run_judge(tmp_path / "nowhere", claude_bin=str(stand_in))


# ---------------------------------------------------------------------------------------------------- script


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("WEIGHTCODE_")}
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120, env=env,
                          check=False)


def test_script_exit_codes(tmp_path: Path, stand_in: Path) -> None:
    assert run_script(str(tmp_path / "none"), "--claude-bin", str(stand_in)).returncode == 2
    session = make_session(tmp_path / "s", LOAD, TORCH)
    completed = run_script(str(session), "--claude-bin", str(stand_in), "--model", "m", "--effort", "low")
    assert completed.returncode == 0, completed.stderr
    assert "final verdict ALLOWED" in completed.stdout
    assert json.loads((session / "logs" / J.JUDGE_FILENAME).read_text())["status"] == "ok"
    set_mode(stand_in, "garbage")
    completed = run_script(str(session), "--claude-bin", str(stand_in))
    assert completed.returncode == 1 and "judge failed" in completed.stderr


def test_script_report_option(tmp_path: Path, stand_in: Path) -> None:
    session = make_session(tmp_path / "s", LOAD, TORCH)
    moved = session / "audit.json"
    (session / "logs" / "audit.json").rename(moved)
    completed = run_script(str(session), "--claude-bin", str(stand_in), "--report", str(moved))
    assert completed.returncode == 0, completed.stderr
    assert (session / J.JUDGE_FILENAME).is_file() and not (session / "logs" / J.JUDGE_FILENAME).exists()
