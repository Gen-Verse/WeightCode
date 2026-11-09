"""Tests of the information-use readout (weightcode.analysis.session_info) on scripted sessions over the toy store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_harness_fixtures import (  # noqa: F401
    BREAK_OPS,
    FIX_OPS,
    FIX_SCORE,
    HarnessStore,
    harness_store,
    needs_sandbox,
    patch_code,
    read_jsonl,
    scripted_session,
)

from weightcode.analysis.session_info import SessionInfoError, main, session_info
from weightcode.baselines.instance_io import load_instance
from weightcode.evalkit.devreadout import dev_readout
from weightcode.evalkit.instances import instances_root
from weightcode.patch import Limits, apply_patch_file

EVIDENCE_CELL = "import numpy as np\nev = np.load('/work/trial/evidence.npz')\nprint(ev['y_fail'][:3], ev['pred_fail'][:3])\n"


def _session(store: HarnessStore) -> Path:
    steps = [
        ("python", {"code": patch_code({"break.json": BREAK_OPS, "fix.json": FIX_OPS})}),
        ("python", {"code": EVIDENCE_CELL}),
        ("patch_check", {"patch_file": "break.json"}),
        ("patch_check", {"patch_file": "fix.json"}),
    ]
    result, replies = scripted_session(store.dev(100), "M2", steps)
    assert replies[3].payload["score"] == FIX_SCORE and replies[2].payload["score"] < 0
    return result.root


@needs_sandbox
def test_the_trajectory_rebuilds_each_checked_child_and_reads_it_on_v_select(harness_store: HarnessStore) -> None:  # noqa: F811
    root = _session(harness_store)
    record = session_info(root)
    rows = record["trajectory"]
    assert [row["query_index"] for row in rows] == [0, 1]
    assert [row["patch_id"] for row in rows] == ["break.json", "fix.json"]
    assert rows[0]["ops"] == [{"op": "add", "tensor": "fc2.bias"}] and rows[1]["vcheck"]["score"] == FIX_SCORE
    instance = load_instance(instances_root() / harness_store.dev(100))
    for row in rows:  # the V_select readout equals an independent rebuild of the same patch
        patch_path = root / "attempts" / row["attempt"] / row["patch_id"]
        child = apply_patch_file(instance.params, patch_path, Limits(), instance.params_sha256)
        assert child.child_sha256 == row["child_sha256"]
        assert row["vselect"] == dev_readout(instance, child.child)["readouts"]
    final = record["final"]
    assert not final["is_base_noop"] and final["query_index"] == 1 and final["patch_id"] == "fix.json"
    assert final["vselect"] == rows[1]["vselect"]
    cells = [cell for cell in read_jsonl(root / "logs" / "cells.jsonl") if cell.get("executed")]
    reads_labels = [int(cell["index"]) for cell in cells if "y_fail" in cell["code"]]
    assert record["access"]["evidence_arrays"]["y_fail"] == reads_labels and len(reads_labels) == 1
    assert record["access"]["evidence_arrays"]["y_rand"] == []
    assert record["access"]["tensors"]["fc1.weight"] and record["access"]["tensors"]["fc2.bias"]
    assert record["usage"]["queries"]["used"] == 2 and record["usage"]["status"] == "complete"


@needs_sandbox
def test_a_logged_child_hash_that_does_not_rebuild_is_refused(harness_store: HarnessStore) -> None:  # noqa: F811
    root = _session(harness_store)
    checks = root / "logs" / "checks.jsonl"
    rows = read_jsonl(checks)
    rows[0]["child_sha256"] = "0" * 64
    checks.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(SessionInfoError, match="query 0 rebuilds child"):
        session_info(root)


@needs_sandbox
def test_the_cli_writes_the_record_and_prints_the_table(harness_store: HarnessStore, tmp_path: Path,  # noqa: F811
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    root = _session(harness_store)
    out = tmp_path / "info.json"
    assert main([str(root), "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["schema"] == "weightcode.analysis.session_info.v0" and len(record["trajectory"]) == 2
    table = capsys.readouterr().out
    assert "fix.json" in table and "final: query 1" in table and "y_fail" in table


@needs_sandbox
def test_every_audit_verdict_on_record_is_reported_and_the_last_is_current(harness_store: HarnessStore) -> None:  # noqa: F811
    root = _session(harness_store)
    (root / "audit.json").write_text(json.dumps({"verdict": "VIOLATION", "candidate_count": 349, "auditor_sha256": "a"}))
    (root / "audit.rerun1.json").write_text(json.dumps({"verdict": "CLEAN", "candidate_count": 4, "auditor_sha256": "b"}))
    record = session_info(root)
    assert [(a["file"], a["verdict"]) for a in record["audits"]] == [("audit.json", "VIOLATION"),
                                                                    ("audit.rerun1.json", "CLEAN")]
    assert record["audit"]["verdict"] == "CLEAN" and record["audit"]["auditor_sha256"] == "b"
