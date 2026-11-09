"""The label-free M2 arm (M2NL): arm spec, trial view, label-array refusal, the declared-patch final rule, and
scripted sessions on the toy store (the real MCP server over stdio, no Claude Code process, no real data)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_evalkit_instances_store import make_split
from test_harness_fixtures import (  # noqa: F401
    FIX_OPS,
    OVER_SIZE_OPS,
    HarnessStore,
    harness_store,
    needs_sandbox,
    patch_code,
    scripted_session,
)

from weightcode.analysis.session_info import format_table, session_info
from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import load_params, params_sha256
from weightcode.evalkit.devreadout import dev_readout, final_child
from weightcode.evalkit.evidence import load_evidence
from weightcode.evalkit.instances import build_task_v1, instances_root
from weightcode.harness import runner
from weightcode.harness.arms import APPLY_REAL, ARMS, SHOWN_REAL, applied_limits, arm_spec, limits_dict, patch_check_budget
from weightcode.harness.finalize import KIND_BASE, KIND_DECLARED, RULE_REASON, rule_reason, run_record, select_final
from weightcode.harness.launch import KILL_REASON_WALL, LaunchOutcome
from weightcode.harness.leaks import ForbiddenStrings, scan_label_arrays
from weightcode.harness.prompts import Budgets, render_system, render_task
from weightcode.harness.scripted import McpError, ScriptedLauncher
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import NO_CHECK_DESCRIPTIONS, TOOL_DESCRIPTIONS, last_valid_declaration, tool_specs
from weightcode.harness.trial import DECLARED_FINAL_SELECTION, EVIDENCE_FROM_INSTANCE_RANDOM, arm_task, assemble
from weightcode.patch import Limits, apply_patch_file

CLASSES = ["c0", "c1", "c2", "c3"]
BUDGETS = Budgets(patch_checks=0, python_cells=150, cell_timeout_s=120.0, wall_timeout_s=3600.0)
NO_CHECK_STATUS_KEYS = {"cells_remaining", "cells_budget", "declared_final", "kept_patch_file", "wall_s_elapsed",
                        "wall_s_remaining"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _section(text: str, heading: str) -> str:
    """The body of one ``## heading`` section of a prompt."""
    return text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]


# ---------------------------------------------------------------------------------------------------------------
# Arm spec, task view and prompts


def test_the_spec_is_m2_without_labels_or_checks() -> None:
    m2, nl = arm_spec("M2"), arm_spec("M2NL")
    assert "M2NL" in ARMS
    assert (nl.shown, nl.evidence, nl.apply_mode, nl.size_limits) == (m2.shown, m2.evidence, m2.apply_mode,
                                                                      m2.size_limits) == (SHOWN_REAL, True, APPLY_REAL,
                                                                                          True)
    assert nl.tools == ("python", "status", "declare_final") and not nl.labels and not nl.checks
    assert m2.labels and m2.checks and all(arm_spec(arm).labels for arm in ARMS if arm != "M2NL")
    assert nl.allowed_tools == ("mcp__weightcode__python", "mcp__weightcode__status", "mcp__weightcode__declare_final")
    assert patch_check_budget(8, "M2NL") == 0 and patch_check_budget(8, "M2") == 8
    limits = build_task_v1(CLASSES)["limits"]
    assert applied_limits(limits, "M2NL") == applied_limits(limits, "M2")


def test_the_task_view_describes_only_the_random_unlabelled_sample() -> None:
    task = build_task_v1(CLASSES)
    m2, view = arm_task(task, "M2", "a" * 64), arm_task(task, "M2NL", "a" * 64)
    assert view["parent_sha256"] == "a" * 64
    assert view["evidence"] == {
        "random": {"keys": ["x_rand", "pred_rand"], "count": task["evidence"]["random"]["count"],
                   "description": "examples drawn at random from the deployment data, with the classifier's "
                                  "predictions"},
        "ids": {"keys": ["split_ids"], "description": "one opaque id per example, in x_rand order"},
    }
    described = json.dumps({"evidence": view["evidence"], "statement": view["statement"]}).lower()
    for word in ("fail", "label", "wrong", "error", "y_"):
        assert word not in described, word
    first, _, last = m2["statement"].split(". ")
    assert view["statement"].startswith(first + ". The evidence file holds random examples") and \
        view["statement"].endswith(last)
    assert "patch_check" not in view and view["final_selection"] == DECLARED_FINAL_SELECTION
    assert set(view) == (set(m2) - {"patch_check"}) | {"final_selection"}
    for key in ("schema", "classes", "patch_format", "limits", "limit_definitions", "training_free_rule"):
        assert view[key] == m2[key], key


def test_the_prompts_and_tools_state_the_declared_patch_rule() -> None:
    task = build_task_v1(CLASSES)
    m2, view = arm_task(task, "M2", "a" * 64), arm_task(task, "M2NL", "a" * 64)
    limits = limits_dict(applied_limits(task["limits"], "M2"))
    m2_text = render_task("M2", m2, limits, Budgets(8, 150, 120.0, 3600.0))
    text = render_task("M2NL", view, limits, BUDGETS)
    assert "$" not in text and "no labels and no checks" in text and "stopped after 60 minutes" in text
    assert "x_fail" not in text and "y_" not in text and "patch_check" not in text
    assert "Declare exactly one final patch" in text and DECLARED_FINAL_SELECTION in text
    for heading in ("Training-free rule", "Limits", "Patch format"):
        assert _section(text, heading) == _section(m2_text, heading), heading
    system, shared = render_system("M2NL"), render_system("M2")
    assert render_system() == shared and "Unchecked patches are never kept" in shared
    assert "Unchecked patches are never kept" not in system and "last valid `declare_final` call is kept" in system
    for heading in ("Rules", "How to work"):
        assert _section(system, heading) == _section(shared, heading), heading
    specs = {spec["name"]: spec["description"] for spec in tool_specs(arm_spec("M2NL").tools)}
    assert list(specs) == ["python", "status", "declare_final"]
    assert specs == {"python": TOOL_DESCRIPTIONS["python"], **NO_CHECK_DESCRIPTIONS}
    assert {spec["name"]: spec["description"] for spec in tool_specs(arm_spec("M2").tools)} == \
        {name: TOOL_DESCRIPTIONS[name] for name in arm_spec("M2").tools}


# ---------------------------------------------------------------------------------------------------------------
# Label arrays


def test_label_arrays_and_label_keys_are_found_in_every_file_kind(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    clean.mkdir()
    np.savez(clean / "evidence.npz", x_rand=np.zeros((2, 3)), pred_rand=np.zeros(2, np.int64),
             split_ids=np.array(["e1", "e2"]))
    (clean / "task.json").write_text(json.dumps({"evidence": {"random": {"keys": ["x_rand", "pred_rand"]}}}))
    (clean / "vocab.json").write_text(json.dumps({"y": 1, "yes": 2}))  # a bare "y" in JSON is not a label key
    assert scan_label_arrays(clean) == {}

    paths = SessionPaths(tmp_path / "session")
    paths.trial.mkdir(parents=True)
    np.savez(paths.trial / "evidence.npz", x_rand=np.zeros((2, 3)), y_rand=np.zeros(2, np.int64))
    np.savez(paths.trial / "graph.npz", features=np.zeros(2), y=np.zeros(2, np.int64))
    np.save(paths.trial / "y_fail.npy", np.zeros(2, np.int64))
    (paths.trial / "task.json").write_text(json.dumps({"evidence": {"keys": ["x_rand", "y_rand"]}, "y_fail": 1}))
    expected = {"trial/evidence.npz": ["y_rand"], "trial/graph.npz": ["y"], "trial/task.json": ["y_fail", "y_rand"],
                "trial/y_fail.npy": ["y_fail.npy"]}
    assert scan_label_arrays(paths.trial) == expected
    nothing = ForbiddenStrings(selftest=(), text=(), text_exempt=())
    unlabelled = runner.leak_scan(paths, {}, [], nothing, labels=False)
    assert unlabelled["label_arrays"] == expected and not unlabelled["passed"]
    labelled = runner.leak_scan(paths, {}, [], nothing)  # an arm that shows labels by design is not label-scanned
    assert labelled["label_arrays"] is None and labelled["passed"]


def test_the_trial_view_holds_only_the_random_part_without_labels(harness_store: HarnessStore,  # noqa: F811
                                                                  tmp_path: Path) -> None:
    instance_dir = instances_root() / harness_store.dev(100)
    session_dir = tmp_path / "session"
    (session_dir / "private").mkdir(parents=True)
    view = assemble(instance_dir, "M2NL", session_dir)
    instance = load_instance(instance_dir)
    assert {path.name for path in view.trial_dir.iterdir()} == {"model.npz", "model_spec.json", "forward.py",
                                                               "evidence.npz", "task.json", *view.input_files}
    shown, full = load_evidence(view.trial_dir / "evidence.npz"), instance.evidence
    assert sorted(shown) == ["pred_rand", "split_ids", "x_rand"]
    np.testing.assert_array_equal(shown["x_rand"], full["x_rand"])
    np.testing.assert_array_equal(shown["pred_rand"], full["pred_rand"])
    np.testing.assert_array_equal(shown["split_ids"], full["split_ids"][len(full["x_fail"]):])
    assert view.evidence_source == EVIDENCE_FROM_INSTANCE_RANDOM
    assert view.evidence_content_sha256 == params_sha256(shown)
    assert view.shown_model_sha256 == view.real_model_sha256 == instance.params_sha256
    assert params_sha256(load_params(view.trial_dir / "model.npz")) == instance.params_sha256
    assert _json(view.trial_dir / "task.json") == view.task == arm_task(instance.task, "M2NL", instance.params_sha256)
    assert view.limits == limits_dict(applied_limits(instance.task["limits"], "M2"))
    assert scan_label_arrays(view.trial_dir) == {}


def test_a_trial_with_a_label_array_is_refused_before_launch(harness_store: HarnessStore,  # noqa: F811
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    assembled = runner.assemble

    def plant_labels(instance_dir: Path, arm: str, session_dir: Path, **kwargs: Any) -> Any:
        view = assembled(instance_dir, arm, session_dir, **kwargs)
        evidence = load_evidence(view.trial_dir / "evidence.npz")
        evidence["y_rand"] = load_instance(instance_dir).evidence["y_rand"]
        np.savez(view.trial_dir / "evidence.npz", **evidence)
        return view

    monkeypatch.setattr(runner, "assemble", plant_labels)
    launcher = ScriptedLauncher(lambda client, replies: None)
    result = runner.run_session(harness_store.dev(100), "M2NL", 0, effort="low", wall_timeout_s=60.0,
                                launcher=launcher, skip_audit=True)
    assert result.status == "aborted_leak" and launcher.tool_names == []
    scan = result.record["leak_scan"]
    assert scan["label_arrays"] == {"trial/evidence.npz": ["y_rand"]} and not scan["hits"] and not scan["passed"]
    assert result.record["final"]["is_base_noop"]


# ---------------------------------------------------------------------------------------------------------------
# Final rule (contract 5.2 with a budget of zero)

VALID = {"patch_file": "fix.json", "kind": "patch", "valid": True, "attempt": "a0001", "child_sha256": "1" * 64}
NULL = {"patch_file": None, "kind": "original", "valid": True, "attempt": None, "child_sha256": "0" * 64}
OVER = {"patch_file": "big.json", "kind": "patch", "valid": False, "attempt": "a0002", "child_sha256": None,
        "messages": ["shown text"], "executor_messages": ["the patch specifies 9 numbers (edit size), above 8.6"]}


def _rule(declarations: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    pick = select_final([], budget=0, declared=last_valid_declaration(declarations))
    return pick, rule_reason(0, pick, declarations)


def test_the_last_valid_declaration_is_kept_and_an_invalid_one_changes_nothing() -> None:
    pick, reason = _rule([VALID])
    assert (pick["kind"], pick["patch_id"], pick["attempt"]) == (KIND_DECLARED, "fix.json", "a0001")
    assert "names fix.json" in reason and "failed validation" not in reason
    pick, reason = _rule([VALID, OVER])
    assert pick["kind"] == KIND_DECLARED and pick["patch_id"] == "fix.json"
    assert "big.json (attempt a0002), failed validation: the patch specifies 9 numbers (edit size)" in reason


def test_no_valid_declaration_keeps_the_base_and_says_why() -> None:
    pick, reason = _rule([])
    assert pick["kind"] == KIND_BASE and reason == f"{RULE_REASON}, zero queries: no declaration was made, so the " \
                                                   "base model is kept"
    pick, reason = _rule([OVER])
    assert pick["kind"] == KIND_BASE and "no valid declaration was made" in reason and "(edit size)" in reason
    pick, reason = _rule([VALID, NULL])
    assert pick["kind"] == KIND_BASE and "names the original model" in reason
    assert rule_reason(8, {"kind": KIND_BASE}, [OVER]) == RULE_REASON  # sessions with queries keep the plain reason


# ---------------------------------------------------------------------------------------------------------------
# Scripted sessions

TRIAL_CELL = (
    "import json, os\nimport numpy as np\n"
    "print(json.dumps({'files': sorted(os.listdir('/work/trial')), "
    "'keys': sorted(np.load('/work/trial/evidence.npz').files), 'task': json.load(open('/work/trial/task.json'))}))\n"
)


def _check_records(root: Path, record: dict[str, Any], instance_id: str) -> dict[str, Any]:
    """run.json is the projection of session_result.json, and the dev readout rebuilds and reads its final child."""
    assert _json(root / "session_result.json") == json.loads(json.dumps(record))
    run = _json(root / "run.json")
    assert run == json.loads(json.dumps(run_record(record)))
    instance = load_instance(instances_root() / instance_id)
    child = final_child(run, root, instance)
    assert params_sha256(child) == run["final"]["child_sha256"] == params_sha256(load_params(root / "final/model.npz"))
    synthetic = make_split(10, {(0, 1): 4}, seed=99, offset=0, colored=False)  # never V_select or sealed data
    assert dev_readout(instance, child, split=synthetic)["n"] == 40
    return run


@needs_sandbox
def test_a_session_keeps_its_declared_patch(harness_store: HarnessStore) -> None:  # noqa: F811
    instance_id = harness_store.dev(100)
    unknown: list[str] = []

    def script(client: Any, replies: list[Any]) -> None:
        for name, arguments in (
            ("python", {"code": TRIAL_CELL}),
            ("python", {"code": patch_code({"fix.json": FIX_OPS, "big.json": OVER_SIZE_OPS})}),
            ("status", {}),
            ("declare_final", {"patch_file": "fix.json", "rationale": "unit 0 should read the class-0 pixel"}),
            ("declare_final", {"patch_file": "big.json", "rationale": "an invalid declaration changes nothing"}),
            ("status", {}),
        ):
            replies.append(client.call(name, arguments))
        try:
            client.call("patch_check", {"patch_file": "fix.json"})
        except McpError as exc:
            unknown.append(str(exc))

    launcher = ScriptedLauncher(script)
    result = runner.run_session(instance_id, "M2NL", 0, effort="low", wall_timeout_s=300.0, launcher=launcher,
                                skip_audit=True)
    record, root = result.record, result.root
    assert result.status == "complete" and not record["infra"]
    assert launcher.tool_names == ["python", "status", "declare_final"] and unknown == ["unknown tool: patch_check"]
    trial_cell, _, before, fix, big, after = launcher.replies
    seen = json.loads(trial_cell.payload["stdout"])
    assert seen["keys"] == ["pred_rand", "split_ids", "x_rand"] and "evidence.npz" in seen["files"]
    assert "patch_check" not in seen["task"] and seen["task"]["final_selection"] == DECLARED_FINAL_SELECTION
    assert set(before.payload) == NO_CHECK_STATUS_KEYS and before.payload["kept_patch_file"] is None
    assert not fix.is_error and fix.payload["valid"] and fix.payload["kept_patch_file"] == "fix.json"
    assert "checked" not in fix.payload
    assert big.is_error and not big.payload["valid"] and "(edit size)" in big.payload["error"][0]
    assert big.payload["kept_patch_file"] == "fix.json"
    assert after.payload["declared_final"] == {"patch_file": "big.json", "valid": False}
    assert after.payload["kept_patch_file"] == "fix.json"

    final = record["final"]
    assert (final["kind"], final["patch_id"], final["parent"], final["is_base_noop"]) == (KIND_DECLARED, "fix.json",
                                                                                          "instance", False)
    assert final["score"] is None and final["query_index"] is None and "names fix.json" in final["reason"]
    assert "big.json" in final["reason"] and record["declared_final"]["patch_file"] == "big.json"
    assert record["selection"]["rule_final"]["kind"] == KIND_DECLARED
    assert record["queries"] == {"used": 0, "consumed": 0, "charged_failures": 0, "budget": 0}
    instance = load_instance(instances_root() / instance_id)
    rebuilt = apply_patch_file(instance.params, root / "final" / "patch.json", Limits(), instance.params_sha256)
    assert rebuilt.child_sha256 == final["child_sha256"] != instance.params_sha256
    run = _check_records(root, record, instance_id)
    assert run["method"] == "M2NL" and run["budget"] == 0 and run["final"]["kind"] == KIND_DECLARED

    session = _json(root / "session.json")
    assert session["budget"]["patch_checks"] == 0 and session["tools"] == ["python", "status", "declare_final"]
    assert session["leak_scan"]["label_arrays"] == {} and session["leak_scan"]["passed"]
    assert (root / "prompts" / "system.md").read_text() == render_system("M2NL")
    assert "no labels and no checks" in (root / "prompts" / "task.md").read_text()
    checks = root / "logs" / "checks.jsonl"
    assert not checks.exists() or not checks.read_text().strip()

    info = session_info(root)
    assert info["trajectory"] == [] and info["final"]["kind"] == KIND_DECLARED
    assert info["final"]["patch_id"] == "fix.json" and info["final"]["query_index"] is None
    assert info["final"]["vselect"] == dev_readout(instance, rebuilt.child)["readouts"]
    assert "final: declared fix.json (unchecked)" in format_table(info)


@needs_sandbox
@pytest.mark.parametrize("steps, why", [
    ([("python", {"code": patch_code({"big.json": OVER_SIZE_OPS})}),
      ("declare_final", {"patch_file": "big.json", "rationale": "nine small changes"})],
     "no valid declaration was made"),
    ([("python", {"code": patch_code({"fix.json": FIX_OPS})})], "no declaration was made"),
])
def test_without_a_valid_declaration_the_base_is_kept_with_the_reason(harness_store: HarnessStore,  # noqa: F811
                                                                       steps: list[Any], why: str) -> None:
    instance_id = harness_store.dev(100)
    result, replies = scripted_session(instance_id, "M2NL", steps)
    final = result.record["final"]
    assert result.status == "complete" and final["is_base_noop"] and final["kind"] == KIND_BASE
    assert why in final["reason"]
    if len(replies) == 2:  # the over-limit declaration: refused at once, and its executor messages are recorded
        assert replies[1].is_error and "(edit size)" in final["reason"] and "big.json" in final["reason"]
    run = _check_records(result.root, result.record, instance_id)
    assert run["final"]["is_base_noop"] and run["final"]["child_sha256"] == result.record["instance"]["params_sha256"]
    assert session_info(result.root)["final"]["is_base_noop"]


@dataclass
class _EndingLauncher(ScriptedLauncher):
    """A scripted editor whose process then ends as the wall limit or a launcher error would end it."""

    kill_reason: str | None = None
    error: str | None = None

    def launch(self, session: Any, paths: SessionPaths, task_prompt: str) -> LaunchOutcome:
        outcome = super().launch(session, paths, task_prompt)
        if self.kill_reason is not None:
            outcome.killed, outcome.kill_reason = True, self.kill_reason
        if self.error is not None:
            outcome.error = self.error
        return outcome


@needs_sandbox
@pytest.mark.parametrize("ending, status", [({"kill_reason": KILL_REASON_WALL}, "timed_out"),
                                            ({"error": "launcher: OSError: boom"}, "infra")])
def test_only_an_infra_session_loses_its_declared_patch(harness_store: HarnessStore,  # noqa: F811
                                                        ending: dict[str, Any], status: str) -> None:
    def script(client: Any, replies: list[Any]) -> None:
        for name, arguments in (("python", {"code": patch_code({"fix.json": FIX_OPS})}),
                                ("declare_final", {"patch_file": "fix.json", "rationale": "declared before the stop"})):
            replies.append(client.call(name, arguments))

    launcher = _EndingLauncher(script, **ending)
    result = runner.run_session(harness_store.dev(100), "M2NL", 0, effort="low", wall_timeout_s=120.0,
                                launcher=launcher, skip_audit=True)
    assert result.status == status and launcher.replies[1].payload["valid"]
    final = result.record["final"]
    assert result.record["selection"]["rule_final"]["patch_id"] == "fix.json"  # what the rule chooses
    keeps_base = status == "infra"
    if keeps_base:
        assert final["is_base_noop"] and final["kind"] == KIND_BASE and status in final["reason"]
    else:
        assert not final["is_base_noop"] and final["kind"] == KIND_DECLARED and final["patch_id"] == "fix.json"
    run = _check_records(result.root, result.record, harness_store.dev(100))
    assert run["status"] == status and run["final"]["is_base_noop"] is keeps_base
