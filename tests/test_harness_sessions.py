"""Scripted editor sessions on the toy store through the real MCP server over stdio (no Claude Code process)."""

from __future__ import annotations

import json
import re

import numpy as np
from test_harness_fixtures import (  # noqa: F401
    FIX_OPS,
    FIX_SCORE,
    OVER_FROB_OPS,
    HarnessStore,
    harness_store,
    needs_sandbox,
    patch_code,
    read_jsonl,
    run_session,
    scripted_session,
)

from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import load_params
from weightcode.evalkit.evidence import evidence_content_sha256, load_evidence
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.patch_check import load_forward, predict

pytestmark = needs_sandbox

HIDDEN = ("child_sha256", "n_changed", "executor_n_changed")


def test_noweight_rejection_and_check_results_carry_no_real_weight_values(harness_store: HarnessStore  # noqa: F811
                                                                         ) -> None:
    instance_id = harness_store.dev(100)
    steps = [
        ("python", {"code": patch_code({"big.json": OVER_FROB_OPS, "fix.json": FIX_OPS})}),
        ("patch_check", {"patch_file": "big.json"}),
        ("patch_check", {"patch_file": "fix.json"}),
        ("declare_final", {"patch_file": "fix.json", "rationale": "S = 12 on the check split"}),
    ]
    result, replies = scripted_session(instance_id, "NOWEIGHT", steps)
    assert result.status == "complete", result.record.get("infra_reasons")
    big, fix = replies[1], replies[2]

    # The editor sees which limit failed and its threshold, and no digit of the measured norm.
    assert big.is_error
    shown = " ".join(big.payload["error"])
    assert "tensor 'fc1.weight': the relative Frobenius norm of the change is above the limit 0.5" in shown
    assert not re.findall(r"\d", shown.replace("fc1", "").replace("0.5", ""))
    # The host logs keep the executor's full message with the measured value.
    root = result.root
    logged = [record for record in read_jsonl(root / "logs" / "tools.jsonl") if record["tool"] == "patch_check"][0]
    executor = " ".join(logged["meta"]["executor_messages"])
    measured = re.search(r"norm of the change is ([0-9.e+-]+), above", executor)
    assert measured is not None and measured.group(1) not in json.dumps(big.payload)
    assert result.record["rejected_attempts"][0]["messages"] == logged["meta"]["executor_messages"]
    assert result.record["rejected_attempts"][0]["redaction"] == "limits"
    attempt = json.loads((root / "attempts" / logged["meta"]["attempt"] / "attempt.json").read_text())
    assert attempt["messages"] == logged["meta"]["executor_messages"] and attempt["outcome"] == "invalid"
    assert result.record["queries"]["used"] == 1  # the rejected patch cost nothing

    # A checked patch: aggregate numbers, without the child hash or change count of the real child.
    assert not fix.is_error and fix.payload["score"] == FIX_SCORE
    assert not set(HIDDEN) & set(fix.payload)
    assert {"edit_size", "edit_frac", "queries_remaining", "fixed", "broken", "base_acc", "child_acc"} <= set(fix.payload)
    check = read_jsonl(root / "logs" / "checks.jsonl")[0]
    assert check["n_changed"] == 2 and len(check["child_sha256"]) == 64 and check["score"] == FIX_SCORE
    assert result.record["final"]["score"] == FIX_SCORE and not result.record["final"]["is_base_noop"]


def test_the_decoy_view_shows_the_decoy_models_own_evidence(harness_store: HarnessStore) -> None:  # noqa: F811
    real_id, decoy_id = harness_store.dev(100), harness_store.dev(101)
    result = run_session(real_id, "DECOY", 0, effort="low", wall_timeout_s=60.0, dry_run=True)
    assert result.status == "dry_run"
    trial = result.root / "trial"
    shown = load_evidence(trial / "evidence.npz")
    sibling = load_instance(instances_root() / decoy_id)
    real = load_instance(instances_root() / real_id)
    assert evidence_content_sha256(shown) == evidence_content_sha256(sibling.evidence)
    assert evidence_content_sha256(shown) != evidence_content_sha256(real.evidence)
    forward, _ = load_forward(trial / "forward.py")
    weights = load_params(trial / "model.npz")
    for part in ("fail", "rand"):
        np.testing.assert_array_equal(predict(forward, weights, shown[f"x_{part}"], 256), shown[f"pred_{part}"])
    session = json.loads((result.root / "session.json").read_text())
    assert session["evidence_source"] == "decoy_model_instance"
    assert session["evidence_content_sha256"] == evidence_content_sha256(shown)
    assert str(instances_root() / decoy_id) in session["forbidden_strings"]["selftest"]
