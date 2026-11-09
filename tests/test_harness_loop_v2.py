"""Kept rule loop-retain-v2 (the E10 stream v2 and the E11 two-round stream, dev only): the stream's cumulative clean
budget instead of loop-retain-v1's per-round clean floor.

Units: the floor arithmetic (round 1 counted, a budget that grows when an earlier round fixed clean rows, a spent
budget refused), eligibility at the boundary, the texts that state the round's floor and the budget, the v1 texts and
rule unchanged byte for byte, and a v2 loop context checked against its library. Streams on the toy stores: a v2
teacher stream (stopped and resumed) whose round 2 is one scripted session, its control (a) on the same student,
round 1's count from its session's check or from the loop's own check, control (b) under v2, the E11 two-round stream's
dry run on the toy ViT store and its control (a) from a session directory through the CLI. No test touches the real
store or starts Claude Code.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from e11_toy import PLANT_CONDITION, e11_store  # noqa: F401
from test_baselines_checked_reference_stream import Grid, _break, _fix, _ids, _shift
from test_harness_e11 import LN_PLANT
from test_harness_fixtures import needs_sandbox
from test_harness_learn import WORSE, LearnStore, _program_cell, learn_store  # noqa: F401
from test_harness_loop import SESSION_KWARGS, _round_one, _round_two, _sandbox, _write, loop_store  # noqa: F401
from test_harness_program_stream import StreamStore, stream_store  # noqa: F401

from weightcode.baselines.fit import checked_reference_stream as C
from weightcode.baselines.fit.reference_stream import load_stream_instances
from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import load_params
from weightcode.evalkit.instances import instances_root
from weightcode.harness import e10, loop, loop_stream, runner, tools
from weightcode.harness.finalize import FINAL_RULE_LOOP, FINAL_RULE_LOOP_V2, KIND_CHECKED, STATUS_DRY_RUN, select_final
from weightcode.harness.instance import read_instance_record
from weightcode.harness.jsonio import file_sha256, read_json
from weightcode.harness.leaks import HINT_WORDS
from weightcode.harness.loop import (
    KEPT_HEADING,
    KEPT_RULE_LOOP,
    KEPT_RULE_LOOP_V2,
    ROUND1_SESSION_CHECK,
    LoopError,
    MultiPartOracle,
    context_clean_used,
    context_kept_rule,
    eligible,
    kept_rule_for,
    loop_final_selection,
    loop_kept_rule,
    loop_kept_rule_v2,
    loop_kept_section,
    session_loop_rule,
)
from weightcode.harness.loop_stream import MODE_SELECT, MODE_TEACHER, LoopStreamError, run_loop_stream
from weightcode.harness.prompts import render_system
from weightcode.harness.retention import s_old_floor
from weightcode.harness.scripted import ScriptedLauncher, call_and_record
from weightcode.harness.session import SessionPaths

V1_TEXT_SHA256 = {
    # sha256 of the v1 texts and rule for 4000 clean rows and 4000 rows per earlier round, one and three earlier
    # rounds, computed with origin/main c89c684's weightcode.harness.loop before loop-retain-v2 existed.
    "1:final_selection": "01f476f86680d34188af6cc57f8da9b54e8ea0f27496f8a175ca3d548a0a15e7",
    "1:kept_section": "1d861f1ee99cf89d2b30834f165da42632b574b1c90c19c5879b1119224bef5d",
    "1:rule": "faeb5659b99f3a2bd1efe3811c66459b77c978ae000d54015164073e9243d053",
    "1:tools": "fd60211e5c18916592c05846f5b023a0c6d836b5171c5ff40c20e2b4509ebc53",
    "3:final_selection": "01f476f86680d34188af6cc57f8da9b54e8ea0f27496f8a175ca3d548a0a15e7",
    "3:kept_section": "1d861f1ee99cf89d2b30834f165da42632b574b1c90c19c5879b1119224bef5d",
    "3:rule": "945649f920e6e88aa825907e0a93d6dd1a40809094ed531d4ff33e7d751e1cbc",
    "3:tools": "fd60211e5c18916592c05846f5b023a0c6d836b5171c5ff40c20e2b4509ebc53",
}
LOOP_TOOLS = ["python", "patch_check", "status", "declare_final"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _section(text: str) -> str:
    start = text.index(KEPT_HEADING) + len(KEPT_HEADING)
    return text[start:text.index("\n\n## ", start)]


def _declare(rule: dict[str, Any]) -> str:
    (spec,) = [item for item in tools.tool_specs(LOOP_TOOLS, True, loop_rule=rule) if item["name"] == "declare_final"]
    return str(spec["description"])


# ---------------------------------------------------------------------------------------------------------------
# Units


def test_the_v2_floor_is_the_remaining_budget_with_round_one_counted() -> None:
    rule = loop_kept_rule_v2(4000, {1: 4000}, {1: -19})  # round 1: the E10 gaussian B final's kept score_old
    assert rule["name"] == KEPT_RULE_LOOP_V2 and rule["check_set"] == "clean" and rule["clean_budget_floor"] == -60
    assert rule["s_old_floor"] == -41 and rule["clean_used_total"] == -19 and rule["s_prev_floor"] == {"1": -20}
    assert loop_kept_rule_v2(4000, {1: 4000}, {1: 2})["s_old_floor"] == -62  # E11 gaussian: +2 adds to the budget
    floors = [loop_kept_rule_v2(4000, {k: 4000 for k in range(1, r)},
                                dict(zip(range(1, r), (-19, -10, -1))))["s_old_floor"] for r in (2, 3, 4)]
    assert floors == [-41, -31, -30]  # the v1 E10 stream's kept score_old, read under v2
    assert loop_kept_rule_v2(4000, {1: 4000}, {1: -60})["s_old_floor"] == 0
    with pytest.raises(LoopError, match="more than the stream's budget of 60"):
        loop_kept_rule_v2(4000, {1: 4000}, {1: -61})
    with pytest.raises(LoopError, match="counts every earlier round"):
        loop_kept_rule_v2(4000, {1: 4000, 2: 4000}, {2: -10})  # round 1 left out
    first = loop_kept_rule_v2(4000, {}, {})  # a round 1 selected under the rule (control (b))
    assert first["s_old_floor"] == -60 and first["s_prev_floor"] == {} and first["clean_used"] == {}
    stored = json.loads(json.dumps(rule))
    assert session_loop_rule({"loop": {}, "kept_rule": stored}) == stored
    for tampered in ({"s_old_floor": -60}, {"clean_used": {"1": -9}}, {"clean_budget_floor": -50}):
        with pytest.raises(LoopError, match="do not follow"):
            session_loop_rule({"loop": {}, "kept_rule": {**stored, **tampered}})
    with pytest.raises(LoopError, match="has kept rule"):
        session_loop_rule({"loop": {}, "kept_rule": {**stored, "name": "loop-retain-v3"}})
    assert kept_rule_for(KEPT_RULE_LOOP_V2, 4000, {1: 4000}, {1: -19}) == rule
    assert kept_rule_for(KEPT_RULE_LOOP, 4000, {1: 4000}) == loop_kept_rule(4000, {1: 4000})
    with pytest.raises(LoopError, match="counts no earlier score_old"):
        kept_rule_for(KEPT_RULE_LOOP, 4000, {1: 4000}, {1: -19})
    with pytest.raises(LoopError, match="needs the kept score_old"):
        kept_rule_for(KEPT_RULE_LOOP_V2, 4000, {1: 4000})
    with pytest.raises(LoopError, match="is one of"):
        kept_rule_for("loop-retain-v3", 4000, {1: 4000})


def test_v2_eligibility_at_the_boundary_and_the_final_pick() -> None:
    rule = loop_kept_rule_v2(4000, {1: 4000, 2: 4000}, {1: -19, 2: -10})  # floor -31
    entry = {"score_old": -31, "score_prev_1": -20, "score_prev_2": 0}
    assert eligible(entry, rule)
    assert not eligible({**entry, "score_old": -32}, rule)
    assert not eligible({**entry, "score_prev_1": -21}, rule)
    v1 = loop_kept_rule(4000, {1: 4000, 2: 4000})
    # the v1 E10 stream's round 3: blends losing 16 to 49 clean rows, all below v1's -12
    assert [eligible({**entry, "score_old": -lost}, rule) for lost in (16, 31, 49)] == [True, True, False]
    assert not any(eligible({**entry, "score_old": -lost}, v1) for lost in (16, 31, 49))

    def checked(patch_id: str, query_index: int, score: int, old: int) -> dict[str, Any]:
        return {"patch_id": patch_id, "query_index": query_index, "score": score, "score_old": old,
                "score_prev_1": 0, "score_prev_2": 0, "score_new": score - old, "n_changed": 10}

    entries = [checked("a.py", 0, 60, -1), checked("b.py", 1, 397, -32), checked("c.py", 2, 300, -31)]
    assert select_final(entries, budget=8, declared=None, loop_rule=rule) == {
        "kind": KIND_CHECKED, "patch_id": "c.py", "query_index": 2}
    assert select_final(entries, budget=8, declared=None, loop_rule=v1)["patch_id"] == "a.py"


def test_the_v2_texts_state_this_rounds_floor_and_the_budget_and_the_v1_texts_are_unchanged() -> None:
    text = render_system("LEARN")
    rule = loop_kept_rule_v2(4000, {1: 4000}, {1: -19})
    out = loop_kept_section(text, rule)
    start = text.index(KEPT_HEADING) + len(KEPT_HEADING)
    assert out[:start] == text[:start] and out.endswith(text[text.index("\n\n## ", start):])
    section = _section(out)
    flat = " ".join(section.split())
    assert "whose score_old is at least -41 and whose score_prev_k is at least -20" in flat
    assert ("The floor -41 on score_old is what is left of this stream's clean budget: over all its rounds together, "
            "the kept updates may break at most 60 more clean check-set images than they fix, and the updates kept in "
            "the earlier rounds used 19 of it, so this round's update may break at most 41 more than it fixes.") in flat
    assert max(len(line) for line in section.splitlines()) <= 116
    plus = " ".join(_section(loop_kept_section(text, loop_kept_rule_v2(4000, {1: 4000}, {1: 2}))).split())
    assert "at least -62" in plus and "fixed 2 more than they broke, which adds 2 to it" in plus
    assert "used none of it" in " ".join(_section(loop_kept_section(
        text, loop_kept_rule_v2(4000, {1: 4000}, {1: 0}))).split())
    final = loop_final_selection(rule)
    assert final.startswith("kept at the end: among the checked programs with score_old >= -41 (the rest of this "
                            "stream's clean budget: over all its rounds the kept updates may break at most 60 more")
    declare = _declare(rule)
    assert ("score_old of at least -41 (what is left of this stream's clean budget of 60 clean check-set images, "
            "broken minus fixed, over all its rounds) and every score_prev_k of at least -20") in declare
    for shown in (flat, plus, final, declare):
        assert not [word for word in HINT_WORDS if word in shown.lower()], shown
    assert FINAL_RULE_LOOP_V2.startswith("LEARN loop round, kept rule loop-retain-v2: ") and "remaining clean" in (
        FINAL_RULE_LOOP_V2) and KEPT_RULE_LOOP in FINAL_RULE_LOOP
    for n_earlier in (1, 3):
        v1 = loop_kept_rule(4000, {k: 4000 for k in range(1, n_earlier + 1)})
        texts = {"kept_section": loop_kept_section(text, v1), "final_selection": loop_final_selection(v1),
                 "tools": json.dumps(tools.tool_specs(LOOP_TOOLS, True, loop_rule=v1), sort_keys=True),
                 "rule": json.dumps(v1, sort_keys=True)}
        for name, value in texts.items():
            assert _sha(value) == V1_TEXT_SHA256[f"{n_earlier}:{name}"], name


def test_a_v2_context_states_counts_that_agree_with_its_library() -> None:
    def scores(old: int, k: int) -> dict[str, int]:
        return {"score_new": 50, "score_old": old, "score": 50 + old, **{f"score_prev_{j}": 0 for j in range(1, k)}}

    library = [{"round": 1, "program": {"path": "p1.py"}, "scores": scores(-19, 1)},
               {"round": 2, "program": {"path": "p2.py"}, "scores": scores(-10, 2)},
               {"round": 3, "program": None, "scores": None}]
    used = [{"round": 1, "s_old": -19, "source": {"kind": ROUND1_SESSION_CHECK}},
            {"round": 2, "s_old": -10, "source": {"kind": "session_final"}},
            {"round": 3, "s_old": 0, "source": {"kind": "noop"}}]
    spec = {"name": KEPT_RULE_LOOP_V2, "clean_budget_frac": 0.015, "clean_budget_floor": -60, "s_old_floor": -31,
            "clean_used": used}
    context = {"round": 4, "library": library, "kept_rule": spec}
    n_prev = {1: 4000, 2: 4000, 3: 4000}
    assert context_clean_used(context) == {1: -19, 2: -10, 3: 0}
    assert context_kept_rule(context, 4000, n_prev) == loop_kept_rule_v2(4000, n_prev, {1: -19, 2: -10, 3: 0})
    assert context_kept_rule({"round": 4, "library": library}, 4000, n_prev) == loop_kept_rule(4000, n_prev)
    oracle_count = [{**used[0], "s_old": -21, "source": {"kind": loop_stream.ROUND1_LOOP_ORACLE}}, *used[1:]]
    assert context_clean_used({**context, "kept_rule": {**spec, "clean_used": oracle_count}})[1] == -21
    cases = [
        ([{**used[0], "s_old": -21}, *used[1:]], "not the library's -19"),  # a session check must be the library's
        ([used[0], {**used[1], "s_old": -9}, used[2]], "not the library's -10"),
        ([*used[:2], {**used[2], "s_old": -3}], "kept no update"),
        (used[:2], "rounds 1 to 3"),
        ([{**used[0], "s_old": True}, *used[1:]], "not an int"),
    ]
    for rows, message in cases:
        with pytest.raises(LoopError, match=message):
            context_clean_used({**context, "kept_rule": {**spec, "clean_used": rows}})
    with pytest.raises(LoopError, match="states s_old_floor -40"):
        context_kept_rule({**context, "kept_rule": {**spec, "s_old_floor": -40}}, 4000, n_prev)
    with pytest.raises(LoopError, match="kept_rule names"):
        context_clean_used({**context, "kept_rule": {**spec, "name": KEPT_RULE_LOOP}})


# ---------------------------------------------------------------------------------------------------------------
# Streams on the toy store


@needs_sandbox
def test_a_v2_teacher_stream_counts_round_one_and_states_the_round_two_floor(
        loop_store: tuple[LearnStore, str, str], tmp_path: Path) -> None:  # noqa: F811
    _, first, second = loop_store
    root1 = _round_one(first)
    program1 = root1 / "final" / "fix.py"
    final1 = read_json(root1 / "final" / "selection.json")["final"]
    out = tmp_path / "loops"
    calls: list[int] = []

    def launcher_for(index: int) -> ScriptedLauncher:
        calls.append(index)
        if len(calls) == 1:
            raise RuntimeError("stopped before the round-2 launch")
        return ScriptedLauncher(_round_two)

    def stream(**extra: Any) -> Path:
        return run_loop_stream(MODE_TEACHER, [first, second], program1, root1, out, _sandbox(),
                               session_kwargs=SESSION_KWARGS, launcher_for=launcher_for,
                               kept_rule=KEPT_RULE_LOOP_V2, **extra)

    with pytest.raises(RuntimeError, match="before the round-2 launch"):
        stream()
    record = read_json(stream(resume=True))  # round 1's count comes back from state.json
    assert calls == [2, 2]
    one, two = record["rounds"]
    n = int(read_instance_record(second).private["n_vcheck"])
    budget = s_old_floor(loop.LOOP_CLEAN_BUDGET_FRAC, n)
    s1 = int(final1["score_old"])
    assert one["clean_s_old"]["s_old"] == s1 and one["clean_s_old"]["source"]["kind"] == ROUND1_SESSION_CHECK
    assert one["clean_s_old"]["source"]["child_sha256"] == one["child_sha256"] == final1["child_sha256"]
    expected = loop_kept_rule_v2(n, {1: n}, {1: s1})
    assert two["kept_rule"] == expected and expected["s_old_floor"] == budget - s1
    session_root = Path(two["sessions"][0]["root"])
    paths = SessionPaths(session_root)
    session = read_json(paths.session_json)
    assert session["kept_rule"] == expected
    context = read_json(Path(two["context"]["path"]))
    assert context["kept_rule"]["s_old_floor"] == expected["s_old_floor"]
    assert context["kept_rule"]["clean_budget_floor"] == budget
    assert context["kept_rule"]["clean_used"] == [one["clean_s_old"]]
    system = " ".join(paths.system_prompt.read_text().split())
    assert f"The floor {expected['s_old_floor']} on score_old is what is left of this stream's clean budget" in system
    assert f"at most {-budget} more clean check-set images than they fix" in system
    selection = read_json(paths.selection_json)
    assert selection["rule"] == FINAL_RULE_LOOP_V2 and selection["kept_rule"] == expected
    assert two["final"]["kind"] == KIND_CHECKED and two["final"]["patch_id"] == "fix3.py"
    assert two["clean_s_old"] == {"round": 2, "s_old": two["final"]["score_old"],
                                  "source": {"kind": "session_final", "session_id": session_root.name}}
    assert record["kept_rule"] == {
        "name": KEPT_RULE_LOOP_V2, "clean_budget_frac": loop.LOOP_CLEAN_BUDGET_FRAC, "clean_budget_floor": budget,
        "clean_rows_sha256": one["clean_s_old"]["source"]["clean_rows_sha256"],
        "clean_used": [one["clean_s_old"], two["clean_s_old"]], "clean_used_total": s1 + two["final"]["score_old"],
        "s_old_floor": {"2": expected["s_old_floor"]}}
    v1_stem = loop_stream._stem(MODE_TEACHER, [first, second], {"sha256": file_sha256(program1)}, {}, SESSION_KWARGS)
    assert v1_stem not in Path(record["work_dir"]).name
    # Control (a) under v2 on the same student keeps the same child under the same rule.
    kept_program = session_root / "final" / "fix3.py"
    control = read_json(run_loop_stream(MODE_SELECT, [first, second], program1, root1, tmp_path / "control",
                                        _sandbox(), candidates={2: [_write(tmp_path, "worse.py", WORSE),
                                                                    kept_program]}, kept_rule=KEPT_RULE_LOOP_V2))
    picked = control["rounds"][1]
    assert picked["kept_rule"] == expected and picked["child_sha256"] == two["child_sha256"]
    assert [row["eligible"] for row in picked["candidates"]] == [False, True]
    assert control["rounds"][0]["clean_s_old"] == one["clean_s_old"]
    assert picked["clean_s_old"]["source"] == {"kind": "control_check", "kept": "candidate_2"}
    assert control["kept_rule"]["s_old_floor"] == {"2": expected["s_old_floor"]}
    with pytest.raises(LoopStreamError, match="kept rule is one of"):
        run_loop_stream(MODE_SELECT, [first, second], program1, root1, tmp_path / "x", _sandbox(),
                        candidates={2: [kept_program]}, kept_rule="loop-retain-v3")


@needs_sandbox
def test_round_one_counts_its_session_check_only_on_the_same_rows_base_and_child(
        loop_store: tuple[LearnStore, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    _, first, second = loop_store
    root1 = _round_one(first)
    final1 = read_json(root1 / "final" / "selection.json")["final"]
    instance = load_instance(instances_root() / first)
    base = {key: np.array(value) for key, value in instance.params.items()}
    child = load_params(root1 / "final" / "model.npz")
    same = loop_stream.round_one_clean(root1, instance, base, child, False)
    assert same["s_old"] == final1["score_old"] and same["source"]["kind"] == ROUND1_SESSION_CHECK
    assert loop_stream.round_one_clean(root1, instance, base, base, True) == {
        "round": 1, "s_old": 0, "source": {"kind": loop_stream.ROUND1_KEPT_BASE, "session_id": root1.name}}
    other = {**child, "fc2.weight": (child["fc2.weight"] * np.float32(-1.0)).astype(child["fc2.weight"].dtype)}
    counted = loop_stream.round_one_clean(root1, instance, base, other, False)
    oracle = MultiPartOracle(loop_stream.check_parts(instance, []), instance.forward, base, 1)
    assert counted["s_old"] == oracle.query(other, "q")["score_old"] < 0
    assert counted["source"]["kind"] == loop_stream.ROUND1_LOOP_ORACLE
    assert counted["source"]["why"] == ["the session's kept child is not the stream's round-1 student"]
    assert counted["source"]["session_score_old"] == final1["score_old"]
    # A session whose clean copy holds other rows is counted by the loop's check, too.
    copy = tmp_path / root1.name
    shutil.copytree(root1, copy)
    session = read_json(copy / "session.json")
    with np.load(session["clean_check"]["path"], allow_pickle=False) as archive:
        rows = {key: archive[key] for key in archive.files}
    np.savez(copy / "other_clean.npz", **{**rows, "y": rows["y"][::-1]})
    session["clean_check"] = {**session["clean_check"], "path": str(copy / "other_clean.npz"),
                              "sha256": file_sha256(copy / "other_clean.npz")}
    (copy / "session.json").write_text(json.dumps(session))
    moved = loop_stream.round_one_clean(copy, instance, base, child, False)
    assert moved["source"]["why"] == ["the session's clean V_check rows are not the loop's"]
    assert moved["s_old"] == final1["score_old"]  # the same child on the loop's rows: the same count
    instances = load_stream_instances([first, second])
    assert loop_stream.shared_clean_rows(instances) == same["source"]["clean_rows_sha256"]
    real = loop_stream.clean.load_clean_check

    def shifted_rows(target: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays, record = real(target)
        return ({**arrays, "y": arrays["y"][::-1].copy()} if target.instance_id == second else arrays), record

    monkeypatch.setattr(loop_stream.clean, "load_clean_check", shifted_rows)
    with pytest.raises(LoopStreamError, match="clean V_check rows differ"):
        loop_stream.shared_clean_rows(instances)


@needs_sandbox
def test_a_session_directory_names_its_kept_program_or_the_base(loop_store: tuple[LearnStore, str, str],  # noqa: F811
                                                                tmp_path: Path) -> None:
    _, first, second = loop_store
    root1 = _round_one(first)
    assert loop_stream.session_candidate(root1) == root1 / "final" / "fix.py"

    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "status", {})

    base = runner.run_session(second, "LEARN", 0, launcher=ScriptedLauncher(script), **SESSION_KWARGS)
    assert loop_stream.session_candidate(base.root) is None
    running = tmp_path / "s0000000000000000"
    running.mkdir()
    with pytest.raises(LoopStreamError, match="has no final yet"):
        loop_stream.session_candidate(running)
    assert loop_stream.default_wall_timeout_s([first, second]) == loop_stream.WALL_TIMEOUT_S == 5400.0


def test_control_b_under_v2_counts_its_own_kept_score_old(stream_store: StreamStore,  # noqa: F811
                                                          tmp_path: Path) -> None:
    ids = _ids(stream_store)
    grid = Grid([("fix", _fix), ("shift", _shift), ("break", _break)])
    path = C.run_checked_reference_stream("grad_full", ids, tmp_path / "v2", make_generator=grid,
                                          kept_rule=KEPT_RULE_LOOP_V2)
    record = json.loads(path.read_text())
    one, two = record["rounds"]
    n = int(one["check_parts"]["old"])
    budget = s_old_floor(loop.LOOP_CLEAN_BUDGET_FRAC, n)
    assert one["kept_rule"] == loop_kept_rule_v2(n, {}, {}) and one["kept_rule"]["s_old_floor"] == budget
    s1 = 0 if one["status"] == C.STATUS_NOOP else int(one["kept_check"]["score_old"])
    assert one["status"] == C.STATUS_OK and one["clean_s_old"] == {
        "round": 1, "s_old": s1, "source": {"kind": "control_check", "kept": one["kept"]}}
    assert two["kept_rule"] == loop_kept_rule_v2(n, {1: int(two["check_parts"]["prev_1"])}, {1: s1})
    for row in (one, two):
        for candidate in row["candidates"]:
            assert candidate["eligible"] == loop.eligible(candidate["check"], row["kept_rule"])
    assert record["kept_rule"]["s_old_floor"] == {"1": budget, "2": budget - s1}
    assert record["kept_rule"]["clean_used_total"] == sum(row["clean_s_old"]["s_old"] for row in record["rounds"])
    assert record["control"].endswith(KEPT_RULE_LOOP_V2)
    v1 = C.run_checked_reference_stream("grad_full", ids, tmp_path / "v1", make_generator=Grid([("fix", _fix)]))
    assert v1.name.split("_")[-2] != path.name.split("_")[-2]  # another identity, so another stem
    assert "kept_rule" not in json.loads(v1.read_text()) and C.round_rule({"old": 4000}, 0)["s_old_floor"] == -12
    with pytest.raises(C.CheckedReferenceError, match="one of"):
        C.round_rule({"old": 4000}, 0, KEPT_RULE_LOOP, {})


# ---------------------------------------------------------------------------------------------------------------
# E11: two rounds on the ViT-Lite student


def _e11_session(instance_id: str, programs: dict[str, str], keep: str | None) -> Path:
    """A finished scripted B session (joint-retain-v1, frac 0.005) that checks ``programs`` and keeps ``keep``, or
    checks nothing and keeps the base when ``keep`` is None."""

    def script(client: Any, replies: list[Any]) -> None:
        if keep is None:
            call_and_record(client, replies, "status", {})
            return
        call_and_record(client, replies, "python", {"code": _program_cell(programs)})
        call_and_record(client, replies, "patch_check", {"program_file": keep})
        call_and_record(client, replies, "declare_final", {"program_file": keep, "rationale": "it scored"})

    result = runner.run_session(instance_id, "LEARN", 0, effort="max", wall_timeout_s=300.0,
                                launcher=ScriptedLauncher(script), skip_audit=True, retention_floor_frac=0.005)
    assert result.status == "complete", result.record.get("infra_reasons")
    return result.root


@needs_sandbox
def test_an_e11_two_round_v2_stream_dry_runs_round_two_and_control_a_takes_a_session(
        e11_store: Any, tmp_path: Path) -> None:  # noqa: F811
    ids = [e11_store.ids[(PLANT_CONDITION, 100)], e11_store.ids[("contrast_s5", 100)]]
    root1 = _e11_session(ids[0], {"ln.py": LN_PLANT}, "ln.py")
    final1 = read_json(root1 / "final" / "selection.json")["final"]
    assert final1["kind"] == KIND_CHECKED and final1["score_old"] >= 0
    program1 = root1 / "final" / "ln.py"
    wall = loop_stream.default_wall_timeout_s(ids)
    assert wall == loop_stream.WALL_TIMEOUT_S_E11 == 7200.0
    dry = read_json(run_loop_stream(MODE_TEACHER, ids, program1, root1, tmp_path / "dry", _sandbox(),
                                    session_kwargs={"effort": "max", "wall_timeout_s": wall}, dry_run=True,
                                    kept_rule=KEPT_RULE_LOOP_V2))
    assert dry["status"] == STATUS_DRY_RUN and dry["round"] == 2 and dry["instance_id"] == ids[1]
    n = int(read_instance_record(ids[1]).private["n_vcheck"])
    expected = loop_kept_rule_v2(n, {1: n}, {1: int(final1["score_old"])})
    assert dry["kept_rule"] == expected and dry["clean_used"][0]["source"]["kind"] == ROUND1_SESSION_CHECK
    assert expected["s_old_floor"] == s_old_floor(loop.LOOP_CLEAN_BUDGET_FRAC, n) - int(final1["score_old"])
    root = Path(dry["root"])
    session = read_json(root / "session.json")
    assert session["kept_rule"] == expected and session["leak_scan"]["passed"], session["leak_scan"]["hits"]
    assert session["dataset"] == "cifar10_vl7" and session["budget"]["wall_timeout_s"] == 7200.0
    assert (session["mcp"]["tool_timeout_ms"], session["mcp"]["connect_timeout_ms"]) == (
        e10.MCP_TOOL_TIMEOUT_MS, e10.MCP_CONNECT_TIMEOUT_MS)
    assert session["loop"]["prev_instances"] == [ids[0]] and root1.name in session["forbidden_strings"]["selftest"]
    system = " ".join((root / "prompts" / "system.md").read_text().split())
    assert f"whose score_old is at least {expected['s_old_floor']}" in system
    index = json.loads((root / "trial" / "library_index.json").read_text())
    assert [(r["round"], r["file"], r["kept_update"]) for r in index["rounds"]] == [(1, "library_program_1.py", True)]
    # Control (a): round 2's B final (here a session that kept the base) given as its session directory.
    root2 = _e11_session(ids[1], {}, None)
    out = tmp_path / "control"
    assert loop_stream.main(["--mode", MODE_SELECT, "--instances", *ids, "--round1-session", str(root1),
                             "--round1-program", str(program1), "--candidates", f"2={root2}",
                             "--kept-rule", KEPT_RULE_LOOP_V2, "--threads", "1", "--out", str(out)]) == 0
    (path,) = list(out.glob("loop_checked_selection_*.json"))
    control = read_json(path)
    one, two = control["rounds"]
    assert [one["status"], two["status"]] == ["ok", "noop"] and two["candidates"] == [] and two["queries_used"] == 0
    assert two["kept_rule"] == expected and control["candidates"] == {"2": []}
    assert [item["s_old"] for item in control["kept_rule"]["clean_used"]] == [int(final1["score_old"]), 0]
    summary = control["summary"]
    assert sorted(summary["conditions"]) == sorted([PLANT_CONDITION, "contrast_s5"])
    assert summary["mean_gain_after_last_round"] == pytest.approx(np.mean(
        [row["gain_after_last_round"] for row in summary["conditions"].values()]))
    with pytest.raises(C.CheckedReferenceError, match="E11"):
        C.run_checked_reference_stream("grad_last", ids, tmp_path / "b", make_generator=Grid([]))
