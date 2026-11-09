"""End-to-end editor sessions on the toy store: the real MCP server over stdio, a scripted editor, finalisation,
run.json and the dev readout. No Claude Code process and no real data; the toy store lives under pytest tmp dirs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_evalkit_instances_store import make_split
from test_harness_fixtures import (  # noqa: F401
    BREAK_OPS,
    FIX_OPS,
    FIX_SCORE,
    OVER_SIZE_OPS,
    HarnessStore,
    harness_store,
    needs_sandbox,
    patch_code,
    read_jsonl,
    scripted_session,
)

from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import load_params, params_sha256
from weightcode.evalkit.devreadout import dev_readout, final_child
from weightcode.evalkit.evidence import load_evidence
from weightcode.evalkit.instances import instances_root
from weightcode.harness.finalize import run_record
from weightcode.harness.launch import KILL_REASON_WALL, LaunchOutcome
from weightcode.harness.runner import run_session
from weightcode.harness.scripted import McpError, ScriptedLauncher
from weightcode.harness.session import SessionPaths, new_session_id
from weightcode.patch import Limits, apply_patch_file

pytestmark = needs_sandbox

# Rebuilds the toy classifier from zeros without its small random weights, plus the fix: S = 12 from the zero model.
REWRITE_OPS: list[dict[str, Any]] = [
    {"op": "set", "tensor": "fc1.weight", "index": [[0, 0], [1, 1], [2, 2], [3, 3], [0, 4], [1, 4]],
     "values": [10.0, 10.0, 10.0, 10.0, 6.0, -6.0]},
    {"op": "set", "tensor": "fc2.weight", "index": [[0, 0], [1, 1], [2, 2], [3, 3]], "values": [1.0, 1.0, 1.0, 1.0]},
]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_records(result: Any, instance_id: str) -> dict[str, Any]:
    """session_result.json matches the returned record, run.json is its projection, and the dev readout parses it."""
    root = result.root
    record = _json(root / "session_result.json")
    assert record == json.loads(json.dumps(result.record))
    run = _json(root / "run.json")
    assert run == json.loads(json.dumps(run_record(record)))
    instance = load_instance(instances_root() / instance_id)
    child = final_child(run, root, instance)
    assert params_sha256(child) == run["final"]["child_sha256"] == params_sha256(load_params(root / "final/model.npz"))
    synthetic = make_split(10, {(0, 1): 4}, seed=99, offset=0, colored=False)  # never V_select or sealed data
    readout = dev_readout(instance, child, split=synthetic)
    assert readout["instance_id"] == instance_id and readout["n"] == 40
    return run


# ---------------------------------------------------------------------------------------------------------------
# Protocol


def test_mcp_protocol_over_real_stdio(harness_store: HarnessStore) -> None:  # noqa: F811
    seen: dict[str, Any] = {}

    def script(client: Any, replies: list[Any]) -> None:
        seen["init"] = client.request("initialize", {"protocolVersion": "2099-01-01", "capabilities": {},
                                                     "clientInfo": {"name": "t", "version": "0"}})
        seen["tools"] = client.list_tools()
        seen["parse"] = client.raw(b"{not json")
        seen["invalid"] = client.raw(b'{"jsonrpc": "2.0", "id": 7, "params": {}}')
        client.notify("notifications/cancelled", {"requestId": 1})  # no reply: the next reply answers the next request
        for name, method, params in (("method", "resources/list", None),
                                     ("tool", "tools/call", {"name": "nope", "arguments": {}})):
            try:
                client.request(method, params)
            except McpError as exc:
                seen[name] = exc.code
        seen["bad_args"] = client.call("status", {"extra": 1})
        seen["status"] = client.call("status", {})

    launcher = ScriptedLauncher(script)
    result = run_session(harness_store.dev(100), "M2", 0, effort="low", wall_timeout_s=120.0, launcher=launcher,
                         skip_audit=True)
    assert result.status == "complete"
    assert seen["init"]["protocolVersion"] == "2099-01-01" and seen["init"]["capabilities"] == {"tools": {}}
    assert [tool["name"] for tool in seen["tools"]] == ["python", "patch_check", "status", "declare_final"]
    assert all(tool["inputSchema"]["type"] == "object" for tool in seen["tools"])
    assert seen["parse"]["error"]["code"] == -32700 and seen["parse"]["id"] is None
    assert seen["invalid"]["error"]["code"] == -32600 and seen["invalid"]["id"] == 7
    assert seen["method"] == -32601 and seen["tool"] == -32602
    assert seen["bad_args"].is_error and "unknown arguments" in seen["bad_args"].payload["error"][0]
    assert not seen["status"].is_error and seen["status"].payload["queries_remaining"] == 8
    tools_log = read_jsonl(result.root / "logs" / "tools.jsonl")
    assert [entry["tool"] for entry in tools_log] == ["nope", "status", "status"]


# ---------------------------------------------------------------------------------------------------------------
# One session per arm


def test_m2_session_end_to_end(harness_store: HarnessStore) -> None:  # noqa: F811
    instance_id = harness_store.dev(100)
    steps = [
        ("status", {}),
        ("python", {"code": patch_code({"fix.json": FIX_OPS, "break.json": BREAK_OPS, "big.json": OVER_SIZE_OPS,
                                        "rewrite.json": REWRITE_OPS})}),
        ("patch_check", {"patch_file": "big.json"}),
        ("patch_check", {"patch_file": "rewrite.json"}),
        ("patch_check", {"patch_file": "break.json"}),
        ("patch_check", {"patch_file": "fix.json"}),
        ("patch_check", {"patch_file": "fix.json"}),
        ("declare_final", {"patch_file": "break.json", "rationale": "the declaration does not choose the kept patch"}),
        ("status", {}),
    ]
    result, replies = scripted_session(instance_id, "M2", steps, skip_audit=False)
    record = result.record
    assert result.status == "complete" and not record["infra"]
    big, rewrite, brk, fix, again = replies[2:7]
    assert big.is_error and "the patch specifies 9 numbers (edit size)" in big.payload["error"][0]
    assert rewrite.is_error and "the patch specifies 10 numbers (edit size)" in rewrite.payload["error"][0]
    assert not brk.is_error and brk.payload["score"] < 0
    assert not fix.is_error and fix.payload["score"] == FIX_SCORE and fix.payload["broken"] == 0
    assert again.is_error and "already checked" in again.payload["error"][0] and again.payload["queries_remaining"] == 6
    assert replies[-1].payload["queries_remaining"] == 6 and replies[-1].payload["declared_final"]["valid"]
    final = record["final"]
    assert (final["kind"], final["patch_id"], final["score"], final["is_base_noop"]) == ("checked", "fix.json", 12, False)
    assert (final["parent"], final["dir"], final["model"]) == ("instance", "final", "final/model.npz")
    assert record["declared_final"]["patch_file"] == "break.json"
    assert record["queries"] == {"used": 2, "consumed": 2, "charged_failures": 0, "budget": 8}
    assert [item["patch_file"] for item in record["rejected_attempts"]] == ["big.json", "rewrite.json"]
    assert record["audit"]["status"] == "done" and record["audit"]["verdict"] is not None
    instance = load_instance(instances_root() / instance_id)
    rebuilt = apply_patch_file(instance.params, result.root / "final" / "patch.json", Limits(), instance.params_sha256)
    assert rebuilt.child_sha256 == final["child_sha256"]
    run = _check_records(result, instance_id)
    assert run["method"] == "M2" and run["flags"]["edit_limits_enforced"] is True
    manifest = read_jsonl(harness_store.store.runs / "manifest.jsonl")
    assert manifest[-1]["session_id"] == result.session_id and manifest[-1]["score"] == FIX_SCORE
    session = _json(result.root / "session.json")
    assert session["provenance"]["git"]["head"] and record["provenance_after"]["git"]["head"]


def test_noweight_session_with_predict(harness_store: HarnessStore) -> None:  # noqa: F811
    instance_id = harness_store.dev(100)
    steps = [
        ("python", {"code": "import os; print(sorted(os.listdir('/work/trial')))"}),
        ("python", {"code": patch_code({"fix.json": FIX_OPS})}),
        ("predict", {"patch_file": None, "subset": "fail"}),
        ("predict", {"patch_file": "fix.json", "subset": "all"}),
        ("predict", {"patch_file": "fix.json", "subset": "train"}),
        ("patch_check", {"patch_file": "fix.json"}),
        ("declare_final", {"patch_file": "fix.json", "rationale": "S = 12"}),
    ]
    result, replies = scripted_session(instance_id, "NOWEIGHT", steps)
    assert result.status == "complete"
    assert "model.npz" not in replies[0].payload["stdout"] and "model_shapes.json" in replies[0].payload["stdout"]
    evidence = load_evidence(result.root / "trial" / "evidence.npz")
    base, patched, bad = replies[2], replies[3], replies[4]
    logits = np.asarray(base.payload["logits"])
    assert logits.shape == (len(evidence["x_fail"]), 4)
    np.testing.assert_array_equal(np.argmax(logits, axis=1), evidence["pred_fail"])
    assert patched.payload["n"] == len(evidence["x_fail"]) + len(evidence["x_rand"])
    assert bad.is_error
    assert replies[5].payload["score"] == FIX_SCORE
    run = _check_records(result, instance_id)
    assert run["method"] == "NOWEIGHT" and run["final"]["parent"] == "instance"
    assert result.record["queries"]["used"] == 1  # predict costs no query


def test_decoy_session_end_to_end(harness_store: HarnessStore) -> None:  # noqa: F811
    instance_id = harness_store.dev(100)
    steps = [
        ("python", {"code": patch_code({"fix.json": FIX_OPS})}),
        ("patch_check", {"patch_file": "fix.json"}),
        ("declare_final", {"patch_file": "fix.json", "rationale": "S = 12"}),
    ]
    result, replies = scripted_session(instance_id, "DECOY", steps)
    assert result.status == "complete"
    assert replies[1].payload["score"] == FIX_SCORE  # S is measured on the real model
    session = _json(result.root / "session.json")
    final = result.record["final"]
    assert final["parent"] == "decoy_readdressed" and final["editor_patch"] == "final/editor_patch.json"
    editor_patch = _json(result.root / "final" / "editor_patch.json")
    readdressed = _json(result.root / "final" / "patch.json")
    assert editor_patch["parent_sha256"] == session["decoy_model_sha256"] != session["real_model_sha256"]
    assert readdressed["parent_sha256"] == session["real_model_sha256"]
    assert {k: v for k, v in readdressed.items() if k != "parent_sha256"} == \
        {k: v for k, v in editor_patch.items() if k != "parent_sha256"}
    instance = load_instance(instances_root() / instance_id)
    rebuilt = apply_patch_file(instance.params, result.root / "final" / "patch.json", Limits(), instance.params_sha256)
    assert rebuilt.child_sha256 == final["child_sha256"]
    run = _check_records(result, instance_id)
    assert run["method"] == "DECOY" and run["final"]["parent"] == "decoy_readdressed"


def test_m1_session_has_no_evidence(harness_store: HarnessStore) -> None:  # noqa: F811
    instance_id = harness_store.dev(100)
    steps = [
        ("python", {"code": "import os, json; print(sorted(os.listdir('/work/trial')), "
                            "'evidence' in json.load(open('/work/trial/task.json')))"}),
        ("python", {"code": patch_code({"fix.json": FIX_OPS})}),
        ("patch_check", {"patch_file": "fix.json"}),
    ]
    result, replies = scripted_session(instance_id, "M1", steps)
    assert result.status == "complete"
    assert "evidence.npz" not in replies[0].payload["stdout"] and replies[0].payload["stdout"].strip().endswith("False")
    assert replies[2].payload["score"] == FIX_SCORE
    assert "evidence" not in (result.root / "prompts" / "task.md").read_text().split("## Tools")[0].split("## Files")[1]
    _check_records(result, instance_id)


def test_rewrite_session_applies_to_zeros_without_size_limits(harness_store: HarnessStore) -> None:  # noqa: F811
    instance_id = harness_store.dev(100)
    steps = [
        ("python", {"code": "import numpy as np; m = np.load('/work/trial/model.npz'); "
                            "print(max(float(abs(m[k]).max()) for k in m.files))"}),
        ("python", {"code": patch_code({"rewrite.json": REWRITE_OPS})}),
        ("patch_check", {"patch_file": "rewrite.json"}),
        ("declare_final", {"patch_file": "rewrite.json", "rationale": "rebuilt"}),
    ]
    result, replies = scripted_session(instance_id, "REWRITE", steps)
    assert result.status == "complete"
    assert replies[0].payload["stdout"].strip() == "0.0"
    assert replies[2].payload["score"] == FIX_SCORE  # edit size 10 is above 8.6, and allowed here
    final = result.record["final"]
    assert final["parent"] == "zeros" and not final["is_base_noop"]
    zeros = {key: np.zeros_like(value) for key, value in load_instance(instances_root() / instance_id).params.items()}
    rebuilt = apply_patch_file(zeros, result.root / "final" / "patch.json", Limits().without_size_limits(),
                               params_sha256(zeros))
    assert rebuilt.child_sha256 == final["child_sha256"]
    run = _check_records(result, instance_id)
    assert run["flags"]["edit_limits_enforced"] is False and run["final"]["parent"] == "zeros"
    task = _json(result.root / "trial" / "task.json")
    assert task["limits"]["max_edit_frac"] is None and task["limits"]["max_rel_frob"] is None


# ---------------------------------------------------------------------------------------------------------------
# Budget, invalid patches and how a session ends


def test_the_query_budget_is_exact_and_invalid_patches_are_free(harness_store: HarnessStore) -> None:  # noqa: F811
    patches = {f"p{k}.json": [{"op": "add", "tensor": "fc2.bias", "index": [[0]], "values": [0.001 * (k + 1)]}]
               for k in range(9)}
    steps: list[tuple[str, dict[str, Any] | None]] = [("python", {"code": patch_code(patches)})]
    steps += [("patch_check", {"patch_file": "missing.json"}), ("patch_check", {"patch_file": "../task.json"})]
    steps += [("patch_check", {"patch_file": f"p{k}.json"}) for k in range(9)]
    steps += [("status", {})]
    result, replies = scripted_session(harness_store.dev(100), "M2", steps)
    missing, escape = replies[1], replies[2]
    assert missing.is_error and escape.is_error
    checks = replies[3:12]
    assert [reply.payload["queries_remaining"] for reply in checks[:8]] == [7, 6, 5, 4, 3, 2, 1, 0]
    assert checks[8].is_error and "spent" in checks[8].payload["error"][0] and checks[8].payload["queries_remaining"] == 0
    assert replies[-1].payload["queries_remaining"] == 0 and len(replies[-1].payload["checked"]) == 8
    assert len(read_jsonl(result.root / "logs" / "checks.jsonl")) == 8
    assert result.record["queries"]["used"] == 8


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


@pytest.mark.parametrize("ending, status", [({"kill_reason": KILL_REASON_WALL}, "timed_out"),
                                            ({"error": "launcher: OSError: boom"}, "infra")])
def test_only_an_infra_session_loses_its_checked_patches(harness_store: HarnessStore, ending: dict[str, Any],  # noqa: F811
                                                         status: str) -> None:
    """PREREG v0.3 section 5 and contract 5.2: a timed-out session keeps the rule's choice; INFRA keeps the base."""
    def script(client: Any, replies: list[Any]) -> None:
        for name, arguments in (("python", {"code": patch_code({"fix.json": FIX_OPS})}),
                                ("patch_check", {"patch_file": "fix.json"})):
            replies.append(client.call(name, arguments))

    launcher = _EndingLauncher(script, **ending)
    result = run_session(harness_store.dev(100), "M2", 0, effort="low", wall_timeout_s=120.0, launcher=launcher,
                         skip_audit=True)
    assert result.status == status and launcher.replies[1].payload["score"] == FIX_SCORE
    final = result.record["final"]
    assert result.record["selection"]["rule_final"]["patch_id"] == "fix.json"  # what the rule chooses
    keeps_base = status == "infra"
    if keeps_base:
        assert final["is_base_noop"] and final["kind"] == "base" and status in final["reason"]
    else:
        assert not final["is_base_noop"] and final["patch_id"] == "fix.json"
    assert result.record["infra"] is keeps_base
    run = _check_records(result, harness_store.dev(100))
    assert run["status"] == status and run["final"]["is_base_noop"] is keeps_base


# ---------------------------------------------------------------------------------------------------------------
# Isolation, leaks and session ids


def test_the_selftest_aborts_a_session_whose_paths_name_an_arm(harness_store: HarnessStore,  # noqa: F811
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    runs = harness_store.store.root / "decoy_runs"
    runs.mkdir()
    monkeypatch.setenv("WEIGHTCODE_RUNS", str(runs))
    launcher = ScriptedLauncher(lambda client, replies: pytest.fail("the editor must not be launched"))
    result = run_session(harness_store.dev(100), "M2", 0, effort="low", wall_timeout_s=60.0, launcher=launcher)
    assert result.status == "aborted_selftest"
    assert "forbidden_strings_absent" in result.record["selftest"]["failed"]
    run = _json(result.root / "run.json")
    assert run["status"] == "aborted_selftest" and run["final"]["is_base_noop"] is True
    assert not (result.root / "logs" / "transcript.jsonl").exists() and launcher.tool_names == []


def test_the_leak_scan_aborts_a_prompt_with_a_forbidden_token(harness_store: HarnessStore) -> None:  # noqa: F811
    launcher = ScriptedLauncher(lambda client, replies: pytest.fail("the editor must not be launched"))
    result = run_session(harness_store.dev(100), "M2", 0, effort="low", wall_timeout_s=60.0, launcher=launcher,
                         task_prompt_override="Improve the decoy classifier.")
    assert result.status == "aborted_leak"
    assert result.record["leak_scan"]["hits"] and launcher.tool_names == []
    assert _json(result.root / "run.json")["final"]["is_base_noop"] is True


def test_session_ids_are_opaque(tmp_path: Path) -> None:
    ids = {new_session_id(["seed", "decoy", "ifee5b7da"], tmp_path) for _ in range(200)}
    assert len(ids) == 200 and all(re.fullmatch(r"s[0-9a-f]{16}", value) for value in ids)
    assert not any("seed" in value for value in ids)
    with pytest.raises(RuntimeError):
        new_session_id(["s"], tmp_path)


@pytest.mark.parametrize("abort", ["aborted_selftest", "aborted_leak"])
def test_an_aborted_dry_run_writes_no_run_record_and_no_manifest_line(harness_store: HarnessStore,  # noqa: F811
                                                                      monkeypatch: pytest.MonkeyPatch,
                                                                      abort: str) -> None:
    runs = harness_store.store.root / ("decoy_runs" if abort == "aborted_selftest" else "runs_b")
    runs.mkdir()
    monkeypatch.setenv("WEIGHTCODE_RUNS", str(runs))
    override = "Improve the decoy classifier." if abort == "aborted_leak" else None
    result = run_session(harness_store.dev(100), "M2", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                         task_prompt_override=override)
    assert result.status == abort and result.root.parent.name == "dry_runs"
    assert result.record["dry_run"] is True and (result.root / "session_result.json").is_file()
    assert not (result.root / "run.json").exists()
    assert not (runs / "manifest.jsonl").exists()
