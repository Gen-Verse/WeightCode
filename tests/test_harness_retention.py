"""The retention kept rule of a program arm (``weightcode.harness.retention``, LEARN v2, dev only): the floor, the final pick,
the wording in the system prompt and in ``declare_final``, the runner's refusals, and one scripted LEARN session over
the real MCP server under the rule. The v1 rule is checked to be unchanged wherever the rule is absent.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from test_harness_fixtures import needs_sandbox, read_jsonl  # noqa: F401
from test_harness_learn import FIX, WORSE, LearnStore, _program_cell, learn_store  # noqa: F401
from weightcode.harness import runner
from weightcode.harness.arms import ARMS, arm_spec
from weightcode.harness.finalize import FINAL_RULE_JOINT, FINAL_RULE_JOINT_RETAIN, KIND_BASE, KIND_CHECKED, select_final
from weightcode.harness.jsonio import read_json
from weightcode.harness.prompts import PROMPT_DIR, render_system, system_template_name
from weightcode.harness.scripted import ScriptedLauncher, ToolReply, call_and_record
from weightcode.harness.retention import (
    KEPT_RULE_RETAIN,
    RetentionRuleError,
    kept_rule_record,
    s_old_floor,
    session_floor,
)
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import PROGRAM_DESCRIPTIONS, tool_specs


def _section(text: str) -> str:
    start = text.index("## What is kept")
    return text[start:text.index("\n\n## ", start + 1)]


def _entry(name: str, score_new: int, score_old: int, n_changed: int, query: int) -> dict[str, Any]:
    return {"patch_id": name, "score": score_new + score_old, "score_new": score_new, "score_old": score_old,
            "n_changed": n_changed, "query_index": query}


def test_the_floor_is_minus_the_rounded_fraction_of_the_rows() -> None:
    assert s_old_floor(0.01, 4000) == -40 and s_old_floor(0.01, 2000) == -20 and s_old_floor(0.02, 150) == -3
    for frac, n_old in ((0.0, 4000), (1.0, 4000), (-0.01, 4000), (0.01, 0), (0.001, 100)):
        with pytest.raises(RetentionRuleError):
            s_old_floor(frac, n_old)
    record = kept_rule_record(0.01, 4000, "clean")
    assert record == {"name": KEPT_RULE_RETAIN, "frac": 0.01, "n_old": 4000, "check_set": "clean", "s_old_floor": -40}
    assert session_floor({"kept_rule": record}) == -40 and session_floor({}) is None
    with pytest.raises(RetentionRuleError):
        session_floor({"kept_rule": record | {"s_old_floor": -39}})
    with pytest.raises(RetentionRuleError):
        session_floor({"kept_rule": record | {"name": "joint-retain-v0"}})


def test_the_final_pick_leaves_out_programs_below_the_floor() -> None:
    # the S_joint winner loses 76 old rows; the runner-up loses 9 (the shape of a real fewshot session's first checks)
    checks = [_entry("q0.py", 1095, -9, 40, 0), _entry("q1.py", 1174, -76, 40, 1)]
    assert select_final(checks, budget=8, declared=None)["patch_id"] == "q1.py"  # v1: argmax S_joint
    assert select_final(checks, budget=8, declared=None, s_old_floor=-20)["patch_id"] == "q0.py"
    assert select_final(checks, budget=8, declared=None, s_old_floor=-5)["kind"] == KIND_BASE
    # the v1 tie breaks still apply inside the eligible pool, and S_joint <= 0 never beats the base
    tied = [_entry("a.py", 10, -1, 9, 0), _entry("b.py", 10, -1, 2, 1), _entry("c.py", 30, -50, 1, 2)]
    pick = select_final(tied, budget=8, declared=None, s_old_floor=-20)
    assert pick == {"kind": KIND_CHECKED, "patch_id": "b.py", "query_index": 1}
    assert select_final([_entry("d.py", 5, -5, 1, 0)], budget=8, declared=None, s_old_floor=-20)["kind"] == KIND_BASE
    with pytest.raises(Exception, match="needs checked programs"):
        select_final([], budget=0, declared=None, s_old_floor=-20)


def test_the_system_prompt_changes_only_what_is_kept() -> None:
    for fewshot, n_old, check_set in ((False, 4000, "clean"), (True, 2000, "old-class")):
        v1 = render_system("LEARN", fewshot=fewshot)
        assert v1 == (PROMPT_DIR / system_template_name("LEARN", fewshot)).read_text(encoding="utf-8")
        v2 = render_system("LEARN", fewshot=fewshot, kept_rule=kept_rule_record(0.01, n_old, check_set))
        floor = -n_old // 100
        assert v1.replace(_section(v1), "") == v2.replace(_section(v2), "")
        kept = " ".join(_section(v2).split())
        assert f"whose score_old is at least {floor} (at most {-floor} more {check_set} check-set images" in kept
        assert f"A program with score_old below {floor} is never kept, whatever its score." in kept
        assert f"If no checked program with score_old of at least {floor} has a score > 0" in kept
        assert "If no checked program has a score > 0" not in kept
        assert max(len(line) for line in v2.splitlines()) <= 116 and "$" not in v2
    for arm in ARMS:
        if not arm_spec(arm).program:
            with pytest.raises(ValueError, match="program arm only"):
                render_system(arm, kept_rule=kept_rule_record(0.01, 4000, "clean"))


def test_declare_final_states_the_floor_and_nothing_else_changes() -> None:
    tools = arm_spec("LEARN").tools
    v1 = tool_specs(tools, program=True)
    v2 = tool_specs(tools, program=True, s_old_floor=-40)
    assert [spec["name"] for spec in v1] == [spec["name"] for spec in v2]
    for a, b in zip(v1, v2):
        if a["name"] == "declare_final":
            assert a["description"] == PROGRAM_DESCRIPTIONS["declare_final"]
            assert "highest score among those with score_old of at least -40 (the original model if none of them" \
                in b["description"]
        else:
            assert a == b
    few = tool_specs(tools, program=True, fewshot=True, s_old_floor=-20)
    assert "score_old of at least -20" in [spec for spec in few if spec["name"] == "declare_final"][0]["description"]
    digest = hashlib.sha256(json.dumps(v1, sort_keys=True).encode()).hexdigest()
    assert digest == hashlib.sha256(json.dumps(tool_specs(tools, True, s_old_floor=None), sort_keys=True)
                                    .encode()).hexdigest()
    with pytest.raises(ValueError, match="program arm only"):
        tool_specs(arm_spec("M2").tools, program=False, s_old_floor=-40)


def test_the_runner_refuses_the_rule_outside_a_program_arm_before_any_write(learn_store: LearnStore) -> None:
    sessions = runner.runs_dir() / runner.SESSIONS_DIR
    before = sorted(sessions.iterdir()) if sessions.exists() else []
    with pytest.raises(ValueError, match="program arm only"):
        runner.run_session(learn_store.ids[100], "M2", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                           retention_floor_frac=0.01)
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\)"):
        runner.run_session(learn_store.ids[100], "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                           retention_floor_frac=1.5)
    assert (sorted(sessions.iterdir()) if sessions.exists() else []) == before


@needs_sandbox
def test_a_learn_session_under_the_retention_rule(learn_store: LearnStore) -> None:
    steps = [
        ("python", {"code": _program_cell({"fix.py": FIX, "worse.py": WORSE})}),
        ("patch_check", {"program_file": "fix.py"}),
        ("patch_check", {"program_file": "worse.py"}),
        ("declare_final", {"program_file": "fix.py", "rationale": "it scored best"}),
    ]
    listed: list[dict[str, Any]] = []

    def script(client: Any, replies: list[ToolReply]) -> None:
        listed.extend(client.list_tools())  # what the session's own MCP server shows the editor
        for name, arguments in steps:
            call_and_record(client, replies, name, arguments)

    launcher = ScriptedLauncher(script)
    result = runner.run_session(learn_store.ids[100], "LEARN", 0, effort="low", wall_timeout_s=300.0,
                                launcher=launcher, skip_audit=True, retention_floor_frac=0.02)
    assert result.status == "complete", result.record.get("infra_reasons")
    _, fix, worse, declared = launcher.replies
    assert not fix.is_error and not worse.is_error and declared.payload["valid"]
    paths = SessionPaths(result.root)
    session = read_json(paths.session_json)
    rule = session["kept_rule"]
    n_old = int(session["clean_check"]["n"])
    assert rule == kept_rule_record(0.02, n_old, "clean") and rule["s_old_floor"] == -round(0.02 * n_old)
    floor = int(rule["s_old_floor"])
    system = paths.system_prompt.read_text(encoding="utf-8")
    assert system == render_system("LEARN", kept_rule=rule) and f"at least {floor}" in system
    assert listed == tool_specs(session["tools"], True, s_old_floor=floor)
    declare = [spec for spec in listed if spec["name"] == "declare_final"][0]
    assert f"score_old of at least {floor}" in declare["description"]
    selection = read_json(paths.selection_json)
    assert selection["rule"] == FINAL_RULE_JOINT_RETAIN != FINAL_RULE_JOINT and selection["s_old_floor"] == floor
    checks = read_jsonl(paths.checks_log)
    expected = select_final(checks, budget=8, declared=None, s_old_floor=floor)
    final = result.record["final"]
    assert (final["kind"], final["patch_id"]) == (expected["kind"], expected["patch_id"])
    # on the toy store fix.py gains on both sets and worse.py loses on both, so the floor keeps fix.py; the pick that
    # the floor changes is covered with fixed scores in test_the_final_pick_leaves_out_programs_below_the_floor
    assert fix.payload["score_old"] >= floor and fix.payload["score"] > 0 and worse.payload["score_old"] < floor
    assert final["kind"] == KIND_CHECKED and final["patch_id"] == "fix.py"
    assert result.record["queries"] == {"used": 2, "consumed": 2, "charged_failures": 0, "budget": 8}
