"""Cell-log reading, report fields and the audit_session.py script (HARNESS_SPEC section 7).

All logs are synthetic and written under pytest's tmp_path; nothing touches $WEIGHTCODE_*.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from weightcode.audit import report as R
from weightcode.audit.session import AuditInputError, CellRecord, audit_cells, audit_session, read_cells, write_report
from weightcode.sandbox.host import INFLIGHT_SCHEMA, INFLIGHT_SUFFIX, _cell_record

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_session.py"

LOAD = '''import numpy as np
ev = np.load("/work/trial/evidence.npz")
xr, yr = ev["x_rand"], ev["y_rand"]
H = xr.reshape(len(xr), -1) / 255.0
'''
LSTSQ = '''Y = np.eye(10)[yr]
W, *_ = np.linalg.lstsq(H, Y, rcond=None)
m = np.load("/work/trial/model.npz")
params = {k: m[k] for k in m.files}
params["fc3.weight"] = W.T
'''
# A probe trained from scratch by gradient descent: an iterative update under a label-dependent objective.
GD = '''w = np.zeros((10, H.shape[1]))
for _ in range(50):
    z = H @ w.T
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    w -= 0.1 * (p - np.eye(10)[yr]).T @ H / len(yr)
'''
SCALES = '''m = np.load("/work/trial/model.npz")
p = {k: m[k] for k in m.files}
for s in (0.5, 0.8, 1.2):
    z = (H @ (p["fc3.weight"] * s).T).argmax(1)
    print((z == yr).mean())
'''


def record(index: int, code: str, *, status: str = "ok", executed: bool = True, refused: bool = False,
           error: str | None = None, generation: int = 0) -> dict[str, object]:
    """A cell-log line as weightcode.sandbox.host writes it."""
    return _cell_record(index=index, timestamp="2026-11-03T00:00:00+00:00", status=status, code=code,
                        executed=executed, refused=refused, error=error, kernel_generation=generation)


def write_log(path: Path, lines: list[dict[str, object] | str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join((line if isinstance(line, str) else json.dumps(line)) + "\n" for line in lines)
    path.write_text(text, encoding="utf-8")
    return path


def rules(report: R.AuditReport) -> set[str]:
    return {flag.rule for flag in report.flags}


# ------------------------------------------------------------------------------------------------- reading


def test_realistic_log_with_refused_sandbox_error_and_interrupted_cells(tmp_path: Path) -> None:
    log = write_log(tmp_path / "logs" / "cells.jsonl", [
        record(0, LOAD),
        record(1, GD, status="refused", executed=False, refused=True, error="Cell refused: the budget is used up."),
        record(2, GD, status="sandbox_error", executed=False, error="The sandbox kernel could not be restarted."),
        record(3, LSTSQ, status="interrupted",
               error="The host process stopped while this cell was running; its output and result are unknown."),
    ])
    report = audit_session(log)
    assert report.verdict == R.VIOLATION
    assert rules(report) == {R.V_SOLVER}
    assert {flag.cell_index for flag in report.flags} == {3}
    assert report.n_cells == 4 and report.n_analysed == 2
    assert report.not_executed_cells == (1, 2)
    assert [note.cell_index for note in report.notes if note.code == "N-NOT-EXECUTED"] == [1, 2]
    assert report.cells_sha256 == hashlib.sha256(log.read_bytes()).hexdigest()
    assert report.label_access_cells == (0,), "cells that read a label array from the evidence file"
    # Audit amendment A2 (2026-11-04): the default label keys include the route-v1 experience labels.
    assert report.evidence_label_keys == ("y_fail", "y_rand", "y_new", "y_old", "y_prev")


def test_refused_cells_alone_are_clean(tmp_path: Path) -> None:
    log = write_log(tmp_path / "cells.jsonl", [
        record(0, LOAD), record(1, GD, status="refused", executed=False, refused=True)])
    assert audit_session(log).verdict == R.CLEAN


def test_lines_with_only_index_and_code(tmp_path: Path) -> None:
    log = write_log(tmp_path / "cells.jsonl", [{"index": 0, "code": LOAD}, {"index": 1, "code": GD}])
    cells = read_cells(log)
    assert cells == [CellRecord(0, LOAD, True, None, None, None), CellRecord(1, GD, True, None, None, None)]
    report = audit_session(log)
    assert report.verdict == R.VIOLATION and rules(report) == {R.V_ITERATIVE}


def test_unknown_fields_are_ignored_and_absent_executed_follows_status(tmp_path: Path) -> None:
    log = write_log(tmp_path / "cells.jsonl", [
        {"index": 0, "code": LOAD, "future_field": {"x": 1}},
        {"index": 1, "code": GD, "status": "refused"},
        {"index": 2, "code": GD, "refused": True},
    ])
    cells = read_cells(log)
    assert [cell.executed for cell in cells] == [True, False, False]
    assert audit_session(log).verdict == R.CLEAN


@pytest.mark.parametrize("line, message", [
    ("not json", "not valid JSON"),
    ("[1, 2]", "not a JSON object"),
    (json.dumps({"index": 1, "code": "x = 1"}), "cell 0 was expected"),
    (json.dumps({"index": True, "code": "x = 1"}), "cell 0 was expected"),
    (json.dumps({"index": 0}), "no code string"),
    (json.dumps({"index": 0, "code": "x = 1", "schema": "other.v0"}), "expected"),
    (json.dumps({"index": 0, "code": "x = 1", "code_sha256": "0" * 64}), "code_sha256 does not match"),
    (json.dumps({"index": 0, "code": "x = 1", "executed": "yes"}), "malformed executed"),
    (json.dumps({"index": 0, "code": "x = 1", "kernel_generation": True}), "malformed kernel_generation"),
    (json.dumps({"index": 0, "code": "x = 1", "executed": True, "refused": True}), "marked executed"),
    (json.dumps({"index": 0, "code": "x = 1", "executed": True, "status": "sandbox_error"}), "marked executed"),
])
def test_malformed_lines_raise(tmp_path: Path, line: str, message: str) -> None:
    log = write_log(tmp_path / "cells.jsonl", [line])
    with pytest.raises(AuditInputError, match=message):
        audit_session(log)


def test_blank_line_inside_the_log_raises(tmp_path: Path) -> None:
    log = tmp_path / "cells.jsonl"
    log.write_text(json.dumps({"index": 0, "code": LOAD}) + "\n\n", encoding="utf-8")
    with pytest.raises(AuditInputError, match="not valid JSON"):
        audit_session(log)


def test_inflight_file_refuses_the_audit(tmp_path: Path) -> None:
    log = write_log(tmp_path / "cells.jsonl", [record(0, LOAD)])
    log.with_name(log.name + INFLIGHT_SUFFIX).write_text("{}", encoding="utf-8")
    with pytest.raises(AuditInputError, match="in flight"):
        audit_session(log)


def test_inflight_cell_recovered_in_memory(tmp_path: Path) -> None:
    """A2 (2026-11-04): a re-audit of a stopped session audits the in-flight cell and leaves the logs unchanged."""
    log = write_log(tmp_path / "cells.jsonl", [record(0, LOAD)])
    inflight = log.with_name(log.name + INFLIGHT_SUFFIX)
    pending = {"schema": INFLIGHT_SCHEMA, "index": 1, "timestamp": "2026-11-03T00:00:00+00:00", "code": LSTSQ,
               "kernel_generation": 0}
    inflight.write_text(json.dumps(pending), encoding="utf-8")
    before = log.read_bytes()
    report = audit_session(log, recover_inflight_in_memory=True)
    assert report.verdict == R.VIOLATION and R.V_SOLVER in rules(report)
    assert report.n_cells == 2
    assert log.read_bytes() == before and inflight.exists()
    inflight.write_text(json.dumps({**pending, "index": 0}), encoding="utf-8")  # result logged, cleanup missed
    assert audit_session(log, recover_inflight_in_memory=True).n_cells == 1
    inflight.write_text(json.dumps({**pending, "index": 5}), encoding="utf-8")
    with pytest.raises(AuditInputError, match="is for cell 5"):
        audit_session(log, recover_inflight_in_memory=True)
    inflight.write_text("{}", encoding="utf-8")
    with pytest.raises(AuditInputError, match="is not a"):
        audit_session(log, recover_inflight_in_memory=True)


def test_empty_log_is_clean(tmp_path: Path) -> None:
    log = tmp_path / "cells.jsonl"
    log.write_text("", encoding="utf-8")
    report = audit_session(log)
    assert report.verdict == R.CLEAN and report.n_cells == 0


# ------------------------------------------------------------------------------------- cells that did not run


def test_cell_that_does_not_parse_and_recorded_a_syntax_error_never_ran(tmp_path: Path) -> None:
    bad = "W = np.linalg.lstsq(H, np.eye(10)[yr]\nparams['fc3.weight'] = W\n"
    error = '  File "<cell-1>", line 1\n    W = np.linalg.lstsq(H, np.eye(10)[yr]\n                       ^\n' \
            "SyntaxError: '(' was never closed\n"
    log = write_log(tmp_path / "cells.jsonl", [record(0, LOAD), record(1, bad, status="error", error=error)])
    report = audit_session(log)
    assert report.verdict == R.CLEAN
    assert [note.code for note in report.notes] == ["N-UNPARSED-NOT-RUN"]
    assert report.unparsed_cells == (1,)


def test_cell_that_parses_but_does_not_compile_never_ran(tmp_path: Path) -> None:
    code = LSTSQ + "return W\n"
    error = '  File "<cell-1>", line 6\nSyntaxError: \'return\' outside function\n'
    log = write_log(tmp_path / "cells.jsonl", [record(0, LOAD), record(1, code, status="error", error=error)])
    report = audit_session(log)
    assert report.verdict == R.CLEAN
    assert [note.code for note in report.notes] == ["N-NOT-COMPILED"]


def test_compile_failure_without_a_recorded_syntax_error_is_analysed(tmp_path: Path) -> None:
    code = LSTSQ + "return W\n"
    log = write_log(tmp_path / "cells.jsonl", [record(0, LOAD), record(1, code)])
    report = audit_session(log)
    assert report.verdict == R.VIOLATION and R.V_SOLVER in rules(report)
    assert "N-COMPILE-MISMATCH" in {note.code for note in report.notes}


def test_unparsable_cell_that_may_have_run_gets_a_soft_flag_and_regex_fallback(tmp_path: Path) -> None:
    code = "clf.fit(H, yr)\nx = (\n"
    log = write_log(tmp_path / "cells.jsonl", [record(0, LOAD), record(1, code)])
    report = audit_session(log)
    assert R.S_UNPARSED in rules(report) and R.V_FIT in rules(report)
    assert report.verdict == R.VIOLATION


def test_runtime_error_cell_is_analysed_in_full(tmp_path: Path) -> None:
    code = LSTSQ + "raise RuntimeError('stop')\n"
    error = 'Traceback (most recent call last):\n  File "<cell-1>", line 6, in <module>\nRuntimeError: stop\n'
    log = write_log(tmp_path / "cells.jsonl", [record(0, LOAD), record(1, code, status="error", error=error)])
    assert R.V_SOLVER in rules(audit_session(log))


def test_kernel_restart_is_noted_and_bindings_are_kept(tmp_path: Path) -> None:
    log = write_log(tmp_path / "cells.jsonl", [
        record(0, LOAD, generation=0),
        record(1, "Y = np.eye(10)[yr]\n", generation=0),
        record(2, "W, *_ = np.linalg.lstsq(H, Y, rcond=None)\n", generation=1),
    ])
    report = audit_session(log)
    assert [(note.code, note.cell_index) for note in report.notes] == [("N-KERNEL-RESTART", 2)]
    assert report.verdict == R.VIOLATION, "the audit keeps bindings across a restart (conservative)"


# ------------------------------------------------------------------------------------------- report fields


def test_flag_fields_point_at_the_cell_line(tmp_path: Path) -> None:
    report = audit_cells([LOAD, LSTSQ])
    (flag,) = report.flags
    assert (flag.flag_id, flag.rule, flag.severity, flag.cell_index, flag.line) == ("F001", R.V_SOLVER, R.VIOLATION, 1, 2)
    assert flag.excerpt == "W, *_ = np.linalg.lstsq(H, Y, rcond=None)"
    assert "label" in flag.reason


def test_to_dict_is_json_serialisable_and_complete(tmp_path: Path) -> None:
    report = audit_cells([LOAD, SCALES, LSTSQ, GD])
    data = json.loads(json.dumps(report.to_dict()))
    assert data["schema"] == R.SCHEMA and data["verdict"] == R.VIOLATION
    assert data["candidate_count"] == report.candidate_count == 3
    assert data["candidate_settings"][0]["count"] == 3 and data["candidate_settings"][0]["kind"] == "loop"
    assert {"flag_id", "rule", "severity", "cell_index", "line", "excerpt", "reason"} <= set(data["flags"][0])
    assert data["rule_catalogue"][R.V_SOLVER]["severity"] == R.VIOLATION
    assert data["rule_counts"] == {R.V_ITERATIVE: 1, R.V_SOLVER: 1}
    assert len(data["auditor_sha256"]) == 64 and data["cells_sha256"] is None
    path = write_report(report, tmp_path / "out" / "audit.json")
    assert json.loads(path.read_text(encoding="utf-8")) == data


def test_plain_strings_and_records_mix_and_must_be_in_order() -> None:
    assert audit_cells([LOAD, CellRecord(1, LSTSQ)]).verdict == R.VIOLATION
    with pytest.raises(AuditInputError):
        audit_cells([LOAD, CellRecord(2, LSTSQ)])


def test_candidate_limit_and_label_keys_are_parameters() -> None:
    assert audit_cells([LOAD, SCALES]).verdict == R.CLEAN
    assert audit_cells([LOAD, SCALES], candidate_limit=2).verdict == R.VIOLATION
    renamed = LOAD.replace('ev["y_rand"]', 'ev["targets"]') + "W, *_ = np.linalg.lstsq(H, np.eye(10)[yr], rcond=None)\n"
    assert audit_cells([renamed]).verdict == R.CLEAN
    report = audit_cells([renamed], ("targets",))
    assert report.verdict == R.VIOLATION and report.evidence_label_keys == ("targets",)


def test_auditor_hash_covers_the_judge_prompt() -> None:
    assert R.auditor_sha256() == R.auditor_sha256()
    assert len(R.auditor_sha256()) == 64


# --------------------------------------------------------------------------------------------------- script


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("WEIGHTCODE_")}
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120, env=env,
                          check=False)


def test_script_writes_logs_audit_json(tmp_path: Path) -> None:
    session = tmp_path / "session"
    write_log(session / "logs" / "cells.jsonl", [record(0, LOAD), record(1, LSTSQ)])
    completed = run_script(str(session))
    assert completed.returncode == 0, completed.stderr
    data = json.loads((session / "logs" / "audit.json").read_text(encoding="utf-8"))
    assert data["verdict"] == R.VIOLATION
    assert "verdict VIOLATION" in completed.stdout and R.V_SOLVER in completed.stdout


def test_script_out_option_and_clean_verdict_exit_zero(tmp_path: Path) -> None:
    session = tmp_path / "session"
    write_log(session / "logs" / "cells.jsonl", [record(0, LOAD)])
    out = tmp_path / "elsewhere" / "audit.json"
    completed = run_script(str(session), "--out", str(out))
    assert completed.returncode == 0, completed.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == R.CLEAN
    assert not (session / "logs" / "audit.json").exists()


def test_script_rerun_options_never_overwrite(tmp_path: Path) -> None:
    """A2 (2026-11-04): a rerun beside the report of record recovers an in-flight cell in memory, prints both counts,
    and refuses to replace an existing report."""
    session = tmp_path / "session"
    log = write_log(session / "logs" / "cells.jsonl", [record(0, LOAD)])
    log.with_name(log.name + INFLIGHT_SUFFIX).write_text(json.dumps(
        {"schema": INFLIGHT_SCHEMA, "index": 1, "timestamp": "2026-11-03T00:00:00+00:00", "code": SCALES,
         "kernel_generation": 0}), encoding="utf-8")
    out = session / "audit.rerun2.json"
    assert run_script(str(session), "--out", str(out), "--no-overwrite").returncode == 2  # in flight, not recovered
    completed = run_script(str(session), "--out", str(out), "--no-overwrite", "--recover-inflight-in-memory")
    assert completed.returncode == 0, completed.stderr
    assert "(L) 3 label-scored" in completed.stdout and "(C) 3 configurations" in completed.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["candidate_count"] == 3 and data["config_count"] == 3 and data["n_cells"] == 2
    again = run_script(str(session), "--out", str(out), "--no-overwrite", "--recover-inflight-in-memory")
    assert again.returncode == 2 and "exists" in again.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == data


def test_script_refuses_a_missing_or_malformed_log(tmp_path: Path) -> None:
    session = tmp_path / "session"
    assert run_script(str(session)).returncode == 2
    write_log(session / "logs" / "cells.jsonl", ["not json"])
    completed = run_script(str(session))
    assert completed.returncode == 2 and "not valid JSON" in completed.stderr
    assert not (session / "logs" / "audit.json").exists()
