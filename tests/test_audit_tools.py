"""Predict calls in the tool log count as scored candidates (PREREG v0.3 section 4.1, HARNESS_SPEC section 7).

Synthetic logs live under pytest's tmp_path; the scripted NOWEIGHT session runs on the harness toy store, also under
tmp dirs. Nothing touches $WEIGHTCODE_*.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_audit_corpus import PRELUDE
from test_harness_fixtures import (  # noqa: F401
    BREAK_OPS,
    FIX_OPS,
    HarnessStore,
    harness_store,
    needs_sandbox,
    patch_code,
    read_jsonl,
    scripted_session,
)

from weightcode.audit import report as R
from weightcode.audit.session import (
    AuditInputError,
    PredictCall,
    audit_cells,
    audit_session,
    predict_candidates,
    read_predict_calls,
)

SHA_A, SHA_B, SHA_BASE = "a" * 64, "b" * 64, "0" * 64

# Five label-scored candidates in the cells (a loop with a literal trip count).
FIVE_SCORED = '''for s in (0.5, 0.6, 0.7, 0.8, 0.9):
    q = dict(params)
    q["fc3.weight"] = params["fc3.weight"] * s
    print(s, acc(q, xf, yf))
'''


def predict(patch_file: str | None, child: str | None, *, subset: str = "all", error: bool = False,
            attempt: str | None = "a0000") -> dict[str, Any]:
    """A predict record as weightcode.harness.mcp_server writes it."""
    if error:
        return {"time": "t", "request_id": 1, "tool": "predict", "arguments": {"patch_file": patch_file, "subset": subset},
                "is_error": True, "result": {"error": ["subset must be one of ['fail', 'rand', 'all']"]},
                "duration_s": 0.0, "meta": {}}
    return {"time": "t", "request_id": 1, "tool": "predict", "arguments": {"patch_file": patch_file, "subset": subset},
            "is_error": False, "result": {"subset": subset, "patch_file": patch_file, "n": 64, "logits": [[0.0]]},
            "duration_s": 0.0, "meta": {"attempt": attempt if patch_file is not None else None, "child_sha256": child}}


def other(tool: str) -> dict[str, Any]:
    return {"time": "t", "request_id": 1, "tool": tool, "arguments": {}, "is_error": False, "result": {},
            "duration_s": 0.0, "meta": {}}


def write_session(root: Path, cells: list[str], tools: list[dict[str, Any] | str] | None) -> Path:
    logs = root / "logs"
    logs.mkdir(parents=True)
    cells_log = logs / "cells.jsonl"
    cells_log.write_text("".join(json.dumps({"index": i, "code": c}) + "\n" for i, c in enumerate(cells)),
                         encoding="utf-8")
    if tools is not None:
        (logs / "tools.jsonl").write_text(
            "".join((t if isinstance(t, str) else json.dumps(t)) + "\n" for t in tools), encoding="utf-8")
    return cells_log


def predict_records(report: R.AuditReport) -> list[R.CandidateSetting]:
    return [s for s in report.candidate_settings if s.kind == "predict"]


def test_absent_tool_log_means_no_predict_calls(tmp_path: Path) -> None:
    report = audit_session(write_session(tmp_path, [*PRELUDE, FIVE_SCORED], None))
    assert report.verdict == R.CLEAN and report.candidate_count == 5
    assert report.tools_sha256 is None and not predict_records(report)
    assert "N-PREDICT-CALLS" not in {n.code for n in report.notes}


def test_distinct_patched_settings_count_once_each(tmp_path: Path) -> None:
    tools = [
        other("python"),
        predict(None, SHA_BASE, subset="fail"),            # line 2: the base model, not a candidate
        predict("fix.json", SHA_A),                        # line 3: candidate A
        predict("fix.json", SHA_A, subset="fail"),         # same setting, other subset
        predict("fix.json", None, subset="train", error=True),  # an error returns no logits
        predict("copy_of_fix.json", SHA_A, attempt="a0003"),    # another file, same parameters
        predict("break.json", SHA_B, attempt="a0004"),     # line 7: candidate B
        other("patch_check"),
        {"time": "t", "request_id": 9, "tool": "predict", "arguments": {"patch_file": "x.json"}, "fault": True},
    ]
    cells_log = write_session(tmp_path, [*PRELUDE], tools)
    report = audit_session(cells_log)
    assert report.verdict == R.CLEAN and report.candidate_count == 2
    found = predict_records(report)
    assert [(s.setting_id, s.cell_index, s.line, s.count, s.counted) for s in found] == [
        (f"predict:{SHA_A}", -1, 3, 1, True), (f"predict:{SHA_B}", -1, 7, 1, True)]
    assert "patch_file=fix.json" in found[0].excerpt
    assert report.tools_sha256 == hashlib.sha256((tmp_path / "logs" / "tools.jsonl").read_bytes()).hexdigest()
    note = next(n for n in report.notes if n.code == "N-PREDICT-CALLS")
    assert note.cell_index == -1 and note.line == 2
    assert note.text.startswith("7 predict calls in the tool log: 2 distinct") and "1 of the base model" in note.text
    assert "1 returned an error, 1 not answered" in note.text
    data = json.loads(json.dumps(report.to_dict()))
    assert data["tools_sha256"] == report.tools_sha256


def test_nine_distinct_predict_patches_are_a_search_violation(tmp_path: Path) -> None:
    tools = [predict(f"p{k}.json", f"{k:064x}", attempt=f"a{k:04d}") for k in range(9)]
    report = audit_session(write_session(tmp_path, [*PRELUDE], tools))
    assert report.verdict == R.VIOLATION and report.candidate_count == 9
    flag = next(f for f in report.flags if f.rule == R.V_SEARCH)
    assert flag.cell_index == -1 and flag.line == 1 and "9 distinct patches sent to the predict tool" in flag.reason


@pytest.mark.parametrize("n_predict, verdict", [(3, R.CLEAN), (4, R.VIOLATION)])
def test_predict_patches_add_to_the_candidates_scored_in_the_cells(n_predict: int, verdict: str) -> None:
    calls = [PredictCall(k + 1, f"p{k}.json", "all", True, False, f"{k:064x}", f"a{k:04d}") for k in range(n_predict)]
    report = audit_cells([*PRELUDE, FIVE_SCORED], predict_calls=calls)
    assert report.verdict == verdict and report.candidate_count == 5 + n_predict
    if verdict == R.VIOLATION:
        flag = next(f for f in report.flags if f.rule == R.V_SEARCH)
        assert flag.cell_index == len(PRELUDE)  # located at the cell sweep, the predict calls are listed
        assert f"{n_predict} distinct patches sent to the predict tool" in flag.reason


@pytest.mark.parametrize("line, message", [
    ("{not json", "is not valid JSON"),
    ('["predict"]', "is not a JSON object"),
    (json.dumps({"tool": "predict", "arguments": "fix.json", "is_error": False}), "arguments are not a JSON object"),
    (json.dumps({"tool": "predict", "arguments": {"patch_file": "fix.json"}}), "has no is_error field"),
    (json.dumps({"tool": "predict", "arguments": {"patch_file": "fix.json", "subset": "all"}, "is_error": False,
                 "meta": {"attempt": "a0000"}}), "has no meta.child_sha256"),
    (json.dumps({"tool": "predict", "arguments": {"patch_file": 3}, "is_error": False, "meta": {}}),
     "malformed patch_file"),
])
def test_malformed_tool_logs_fail(tmp_path: Path, line: str, message: str) -> None:
    cells_log = write_session(tmp_path, [*PRELUDE], [other("python"), line])
    with pytest.raises(AuditInputError, match=message):
        audit_session(cells_log)


def test_records_of_other_tools_are_not_checked(tmp_path: Path) -> None:
    path = tmp_path / "tools.jsonl"
    path.write_text(json.dumps({"tool": "python", "arguments": "anything"}) + "\n" + json.dumps(predict(None, SHA_BASE))
                    + "\n", encoding="utf-8")
    calls = read_predict_calls(path)
    assert [(c.line, c.patch_file, c.answered, c.is_error) for c in calls] == [(2, None, True, False)]
    assert predict_candidates(calls)[0] == []


def test_a_hand_built_call_without_a_parameter_hash_is_refused() -> None:
    with pytest.raises(AuditInputError, match="names no patched parameter hash"):
        predict_candidates([PredictCall(1, "fix.json", "all", True, False, None, "a0000")])


@needs_sandbox
def test_scripted_noweight_session_counts_its_predict_patches(harness_store: HarnessStore) -> None:  # noqa: F811
    """The harness's own tool log, audited by the harness at finalisation: two patches sent to predict count."""
    steps = [
        ("python", {"code": patch_code({"fix.json": FIX_OPS, "break.json": BREAK_OPS})}),
        ("predict", {"patch_file": None, "subset": "fail"}),
        ("predict", {"patch_file": "fix.json", "subset": "all"}),
        ("predict", {"patch_file": "fix.json", "subset": "fail"}),
        ("predict", {"patch_file": "break.json", "subset": "rand"}),
        ("predict", {"patch_file": "fix.json", "subset": "train"}),
        ("declare_final", {"patch_file": "fix.json", "rationale": "fixes the planted errors"}),
    ]
    result, replies = scripted_session(harness_store.dev(100), "NOWEIGHT", steps, skip_audit=False)
    assert result.status == "complete" and replies[5].is_error
    logged = [r for r in read_jsonl(result.root / "logs" / "tools.jsonl") if r["tool"] == "predict"]
    children = {r["meta"]["child_sha256"] for r in logged if not r["is_error"] and r["arguments"]["patch_file"]}
    assert len(children) == 2
    audit = result.record["audit"]
    assert audit["status"] == "done" and audit["verdict"] == R.CLEAN and audit["candidate_count"] == 2
    report = json.loads((result.root / "audit.json").read_text(encoding="utf-8"))
    counted = [s for s in report["candidate_settings"] if s["kind"] == "predict"]
    assert {s["setting_id"] for s in counted} == {f"predict:{c}" for c in children}
    assert report["tools_sha256"] == hashlib.sha256((result.root / "logs" / "tools.jsonl").read_bytes()).hexdigest()
