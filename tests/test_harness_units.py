"""Pure pieces of the editor harness: the final-patch rule, the run.json projection and editor-facing rejections."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_evalkit_instances_store import toy_params

from weightcode.common.identity import params_sha256, save_params
from weightcode.harness.applying import (
    GENERIC_TEXT,
    REDACTION_LIMITS,
    REDACTION_NON_FINITE,
    REDACTION_NONE,
    ArmApplier,
    EditorRejection,
)
from weightcode.harness.arms import APPLY_BY_INDEX_TO_REAL, APPLY_REAL, limits_dict
from weightcode.harness.finalize import KIND_BASE, KIND_CHECKED, KIND_DECLARED, RUN_SCHEMA, run_record, select_final
from weightcode.patch import Limits

# ---------------------------------------------------------------------------------------------------------------
# Final-patch rule (contract 5.2)


def _entry(patch_id: str, query_index: int, score: int, n_changed: int) -> dict[str, Any]:
    return {"patch_id": patch_id, "query_index": query_index, "score": score, "n_changed": n_changed}


def test_final_rule_keeps_the_base_when_no_checked_patch_has_positive_score() -> None:
    assert select_final([], budget=8, declared=None)["kind"] == KIND_BASE
    checked = [_entry("a.json", 0, 0, 3), _entry("b.json", 1, -2, 1)]
    assert select_final(checked, budget=8, declared=None)["kind"] == KIND_BASE


def test_final_rule_picks_highest_score_then_fewer_changes_then_earlier_query() -> None:
    checked = [_entry("a.json", 0, 3, 10), _entry("b.json", 1, 5, 9), _entry("c.json", 2, 5, 4),
               _entry("d.json", 3, 5, 4)]
    pick = select_final(checked, budget=8, declared=None)
    assert pick == {"kind": KIND_CHECKED, "patch_id": "c.json", "query_index": 2}


def test_final_rule_never_keeps_an_unchecked_declared_patch_while_queries_exist() -> None:
    declared = {"patch_file": "x.json", "valid": True, "attempt": "a0003", "child_sha256": "0" * 64}
    assert select_final([], budget=8, declared=declared)["kind"] == KIND_BASE
    checked = [_entry("a.json", 0, 2, 1)]
    assert select_final(checked, budget=8, declared=declared)["patch_id"] == "a.json"


def test_final_rule_with_zero_budget_keeps_a_valid_declared_patch() -> None:
    declared = {"patch_file": "x.json", "valid": True, "attempt": "a0003", "child_sha256": "1" * 64}
    pick = select_final([], budget=0, declared=declared)
    assert pick["kind"] == KIND_DECLARED and pick["patch_id"] == "x.json" and pick["attempt"] == "a0003"
    assert select_final([], budget=0, declared=dict(declared, valid=False))["kind"] == KIND_BASE
    assert select_final([], budget=0, declared=dict(declared, patch_file=None))["kind"] == KIND_BASE
    assert select_final([], budget=0, declared=None)["kind"] == KIND_BASE


# ---------------------------------------------------------------------------------------------------------------
# run.json projection

FINAL_KEYS = ("is_base_noop", "dir", "parent", "model", "child_sha256", "score", "fixed", "broken", "edit_size",
              "edit_frac", "n_changed", "patch_id", "query_index", "kind", "editor_patch", "reason")


def _result(arm: str, final: dict[str, Any], edit_limits_enforced: bool = True) -> dict[str, Any]:
    return {
        "arm": arm, "status": "complete", "session_id": "s" + "ab" * 8, "repeat": 3,
        "created_at": "2026-11-03T00:00:00+00:00", "finalized_at": "2026-11-03T01:00:00+00:00",
        "instance": {"instance_id": "i0123abcd", "split_role": "dev", "dataset": "toyset", "condition": "deploy",
                     "params_sha256": "2" * 64, "forward_sha256": "3" * 64, "evidence_rule": "evidence-v1"},
        "edit_limits_enforced": edit_limits_enforced, "infra": False,
        "audit": {"status": "done", "verdict": "CLEAN"},
        "final": final, "queries": {"used": 2, "budget": 8},
        "editor": {"model": "claude-opus-5-5", "effort": "max", "launcher": "claude", "system_prompt_mode": "replace"},
        "tokens": {"input_tokens": 10},
    }


def test_run_record_is_a_projection_of_the_session_result() -> None:
    final = {key: None for key in FINAL_KEYS} | {"is_base_noop": False, "dir": "final", "parent": "decoy_readdressed",
                                                  "model": "final/model.npz", "child_sha256": "4" * 64, "score": 5,
                                                  "fixed": 6, "broken": 1, "edit_size": 2, "edit_frac": 0.01,
                                                  "n_changed": 2, "patch_id": "p.json", "query_index": 1,
                                                  "kind": "checked", "editor_patch": "final/editor_patch.json",
                                                  "attempt": "a0002", "base_acc": 0.5}
    result = _result("DECOY", final)
    run = run_record(result)
    assert run["schema"] == RUN_SCHEMA
    assert run["method"] == "DECOY" and run["status"] == "complete" and run["repeat"] == 3
    assert run["instance"] == result["instance"]
    assert run["flags"] == {"edit_limits_enforced": True, "infra": False, "audit_verdict": "CLEAN",
                            "audit_status": "done"}
    assert run["final"] == {key: final[key] for key in FINAL_KEYS}
    assert (run["queries_used"], run["budget"]) == (2, 8)
    assert run["editor"] == {"model": "claude-opus-5-5", "effort": "max", "launcher": "claude"}
    assert run["session_result"] == "session_result.json"
    rewrite = run_record(_result("REWRITE", final | {"parent": "zeros"}, edit_limits_enforced=False))
    assert rewrite["flags"]["edit_limits_enforced"] is False and rewrite["final"]["parent"] == "zeros"


def test_run_record_of_an_audit_that_did_not_run_has_no_verdict() -> None:
    result = _result("M2", {key: None for key in FINAL_KEYS} | {"is_base_noop": True})
    result["audit"] = {"status": "not_run", "reason": "dry"}
    assert run_record(result)["flags"]["audit_verdict"] is None
    assert run_record(result)["flags"]["audit_status"] == "not_run"


# ---------------------------------------------------------------------------------------------------------------
# Editor-facing rejections (review finding A)


def _applier(tmp_path: Path, *, mode: str = APPLY_REAL, limits: Limits | None = None) -> tuple[ArmApplier, str]:
    real = toy_params(0)
    path = tmp_path / "model.npz"
    real_sha = save_params(path, real)
    session: dict[str, Any] = {"apply_mode": mode, "limits": limits_dict(limits or Limits()),
                               "real_model_sha256": real_sha, "real_model_path": str(path)}
    if mode == APPLY_BY_INDEX_TO_REAL:
        decoy_sha = params_sha256(toy_params(1))
        session.update(shown_model_sha256=decoy_sha, decoy_model_sha256=decoy_sha)
    else:
        session["shown_model_sha256"] = real_sha
    applier = ArmApplier(session)
    return applier, str(session["shown_model_sha256"])


def _patch(parent: str, ops: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": "weightcode.patch.v0", "parent_sha256": parent, "hypothesis": "unit 0 reads the class-0 pixel",
            "predicted_effect": "fixes some errors", "ops": ops}


def _rejection(applier: ArmApplier, patch: dict[str, Any], tmp_path: Path) -> EditorRejection:
    with pytest.raises(EditorRejection) as caught:
        applier.apply_for_editor(patch, tmp_path)
    return caught.value


def _digits_left(text: str, *allowed: str) -> list[str]:
    for token in allowed:
        text = text.replace(token, "")
    return re.findall(r"\d", text)


@pytest.mark.parametrize("mode", [APPLY_REAL, APPLY_BY_INDEX_TO_REAL])
def test_relative_norm_rejection_names_the_limit_but_not_the_measured_value(tmp_path: Path, mode: str) -> None:
    applier, parent = _applier(tmp_path, mode=mode)
    rejection = _rejection(applier, _patch(parent, [{"op": "add", "tensor": "fc1.weight", "index": [[0, 0]],
                                                     "values": [1000.0]}]), tmp_path)
    measured = 1000.0 / float(np.linalg.norm(toy_params(0)["fc1.weight"].astype(np.float64)))
    assert rejection.redaction == REDACTION_LIMITS
    assert rejection.messages == ["tensor 'fc1.weight': the relative Frobenius norm of the change is above the limit 0.5"]
    assert not _digits_left(" ".join(rejection.messages), "fc1", "0.5")
    logged = re.search(r"relative Frobenius norm of the change is ([0-9.e+-]+), above", rejection.executor_messages[0])
    assert logged is not None and float(logged.group(1)) == pytest.approx(measured, rel=1e-6)  # kept for the host


def test_an_exempt_tensor_is_never_named_in_a_relative_norm_rejection(tmp_path: Path) -> None:
    applier, parent = _applier(tmp_path, limits=Limits(rel_frob_exempt=["fc2.weight"]))
    grow = [{"op": "add", "tensor": name, "index": [[0, 0]], "values": [1000.0]} for name in ("fc1.weight", "fc2.weight")]
    rejection = _rejection(applier, _patch(parent, grow), tmp_path)
    assert rejection.messages == ["tensor 'fc1.weight': the relative Frobenius norm of the change is above the limit 0.5"]
    assert len(rejection.executor_messages) == 1 and "'fc1.weight'" in rejection.executor_messages[0]
    assert applier.apply_for_editor(_patch(parent, grow[1:]), tmp_path).per_tensor["fc2.weight"]["rel_frob"] > 0.5


def test_changed_fraction_rejection_hides_the_count(tmp_path: Path) -> None:
    applier, parent = _applier(tmp_path, limits=Limits(max_changed_frac=0.001))
    rejection = _rejection(applier, _patch(parent, [{"op": "scale_rows", "tensor": "fc1.weight", "rows": [0],
                                                     "factor": 1.1}]), tmp_path)
    assert rejection.redaction == REDACTION_LIMITS
    assert rejection.messages == ["size: the fraction of parameters that change is above the limit 0.001"]
    assert "16 of 172 parameters change" in rejection.executor_messages[0]


def test_non_finite_rejection_hides_the_count(tmp_path: Path) -> None:
    applier, parent = _applier(tmp_path)
    rejection = _rejection(applier, _patch(parent, [{"op": "scale_rows", "tensor": "fc1.weight", "rows": [0],
                                                     "factor": 1e38}]), tmp_path)
    assert rejection.redaction == REDACTION_NON_FINITE
    assert rejection.messages[0].startswith("tensor 'fc1.weight': some entries are not finite after the patch")
    assert not _digits_left(" ".join(rejection.messages), "fc1", "float64", "float32")
    assert "1 entries are not finite" in rejection.executor_messages[0]


@pytest.mark.parametrize("ops, fragment", [
    ([{"op": "add", "tensor": "fc1.weight", "index": [[5, k] for k in range(9)], "values": [0.01] * 9}],
     "the patch specifies 9 numbers (edit size), above the limit 0.05 x 172 parameters = 8.6"),
    ([{"op": "add", "tensor": "fc9.weight", "index": [[0, 0]], "values": [1.0]}], "unknown tensor"),
    ([{"op": "set", "tensor": "fc1.weight", "index": [[0, 99]], "values": [1.0]}], "is outside [0, 16)"),
    # the same result on any parent: float32 overflow of a set value
    ([{"op": "set", "tensor": "fc2.bias", "index": [[0]], "values": [1e39]}], "1 entries are not finite"),
])
def test_rejections_that_depend_only_on_shapes_and_the_patch_stay_verbatim(tmp_path: Path, ops: list[Any],
                                                                            fragment: str) -> None:
    applier, parent = _applier(tmp_path)
    rejection = _rejection(applier, _patch(parent, ops), tmp_path)
    assert rejection.redaction == REDACTION_NONE
    assert rejection.messages == rejection.executor_messages
    assert fragment in " ".join(rejection.messages)


def test_a_wrong_parent_hash_is_reported_verbatim_in_decoy_mode(tmp_path: Path) -> None:
    applier, _ = _applier(tmp_path, mode=APPLY_BY_INDEX_TO_REAL)
    rejection = _rejection(applier, _patch(applier.real_sha256, [{"op": "add", "tensor": "fc2.bias",
                                                                  "index": [[0]], "values": [0.1]}]), tmp_path)
    assert rejection.redaction == REDACTION_NONE and "parent_sha256" in rejection.messages[0]


def test_a_valid_patch_is_applied_unchanged_by_the_editor_path(tmp_path: Path) -> None:
    applier, parent = _applier(tmp_path)
    patch = _patch(parent, [{"op": "add", "tensor": "fc2.bias", "index": [[0]], "values": [0.1]}])
    assert applier.apply_for_editor(patch, tmp_path).child_sha256 == applier.apply(patch, tmp_path).child_sha256
    assert GENERIC_TEXT.startswith("patch:")
