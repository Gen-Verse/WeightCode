"""The loop stream (PREREG E4-loop, dev only): the one-image rule for the earlier rounds' rows, the multi-part check,
the loop-retain-v1 floors and the final pick, the prompt texts and the round library, the runner's refusals, and two
toy streams run end to end. The teacher stream runs round 2 as one scripted LEARN session on the round-1 student; it
is stopped twice and resumed, and the finished session is never launched again. The checked-selection control then
runs the same program on the same student and must reach the same child with the same check.

The store is the evalkit toy store with two conditions of one corruption (``toyblur_s2`` and ``toyblur_s3``), so the
seed-100 instances share one base model. No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from test_evalkit_instances_store import add_condition, build_condition
from test_harness_fixtures import needs_sandbox
from test_harness_learn import FIX, STRINGS, WORSE, LearnStore, _program_cell, learn_store  # noqa: F401
from weightcode.common.identity import load_params, params_sha256
from weightcode.evalkit.devreadout import DevReadoutError, readout_run
from weightcode.evalkit.patch_check import BudgetExceeded
from weightcode.harness import loop_stream, runner
from weightcode.harness.finalize import KIND_BASE, KIND_CHECKED, FinalizeError, select_final
from weightcode.harness.instance import InstanceRefused
from weightcode.harness.jsonio import file_sha256, read_json
from weightcode.harness.leaks import HINT_WORDS
from weightcode.harness.loop import (
    FILES_ANCHOR,
    KEPT_HEADING,
    KEPT_RULE_LOOP,
    LOOP_CONTEXT_SCHEMA,
    LOOP_LIBRARY_INDEX,
    LOOP_READOUT,
    LOOP_STATUS_DESCRIPTION,
    PARENT_LOOP,
    PREV_KEYS,
    LoopError,
    MultiPartOracle,
    eligible,
    kept_prev_rows,
    loop_files_section,
    loop_final_selection,
    loop_kept_rule,
    loop_kept_section,
    loop_statement,
    read_loop_context,
    session_loop_rule,
    withheld_tokens,
    write_loop_library,
)
from weightcode.harness.loop_stream import MODE_SELECT, MODE_TEACHER, LoopStreamError, run_loop_stream
from weightcode.harness.prompts import render_system
from weightcode.harness.scripted import ScriptedLauncher, ToolReply, call_and_record
from weightcode.harness.session import SessionPaths
from weightcode.sandbox.program import ProgramSandbox

SECOND = "toyblur_s3"
PROGRAM = "HYPOTHESIS = 'h'\nPREDICTED_EFFECT = 'p'\n\ndef update(student, experience):\n    return {}\n"
# Hidden unit 3 reads the true-class-3 pixel and unit 2 loses it: fixes the (3 -> 2) confusions and changes no
# prediction on an image of another class, whatever FIX changed for class 0.
FIX3 = STRINGS + """import numpy as np

def update(student, experience):
    print("prev", len(experience["y_prev"]), sorted(set(experience["round_prev"].tolist())))
    delta = np.zeros_like(student["params"]["fc1.weight"])
    delta[3, 7], delta[2, 7] = 6.0, -6.0
    return {"fc1.weight": delta}
"""
READ_CELL = ("import json\nimport numpy as np\n"
             "index = json.load(open('/work/trial/library_index.json'))\n"
             "print([(r['label'], r['file'], r['kept_update'], r['scores']) for r in index['rounds']])\n"
             "e = np.load('/work/trial/experience.npz')\n"
             "print(sorted(e.files), len(e['y_prev']), sorted(set(e['round_prev'].tolist())))\n")
SESSION_KWARGS = {"effort": "low", "wall_timeout_s": 300.0, "skip_audit": True}
LOOP_SHOWN_CHECK_KEYS = {"program_file", "query_index", "score_new", "score_old", "score_prev_1", "score",
                         "queries_remaining", "tensors", "program_stdout", "program_stderr", "duration_s",
                         "wall_s_elapsed", "wall_s_remaining"}


def _sandbox() -> ProgramSandbox:
    return ProgramSandbox(python=runner.SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)


def _write(directory: Path, name: str, source: str) -> Path:
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


# ---------------------------------------------------------------------------------------------------------------
# Units


def test_the_one_image_rule_drops_rows_whose_image_is_already_shown() -> None:
    prev = [(1, {"idx": np.array([5, 6, 7, 8])}), (2, {"idx": np.array([7, 9, 10, 5])})]
    kept, dropped = kept_prev_rows(prev, np.array([8, 11]))
    assert [(k, positions.tolist()) for k, positions in kept] == [(1, [0, 1, 2]), (2, [1, 2])]
    assert dropped == {"1": 1, "2": 2}
    kept, dropped = kept_prev_rows(prev[:1], np.array([], dtype=np.int64))
    assert kept[0][1].tolist() == [0, 1, 2, 3] and dropped == {"1": 0}


def test_the_floors_follow_the_fractions_and_bind_on_every_part() -> None:
    rule = loop_kept_rule(4000, {1: 4000, 2: 4000})
    assert rule["name"] == KEPT_RULE_LOOP and rule["check_set"] == "clean"
    assert rule["s_old_floor"] == -12 and rule["s_prev_floor"] == {"1": -20, "2": -20}
    entry = {"score_old": -12, "score_prev_1": -20, "score_prev_2": 0}
    assert eligible(entry, rule)
    assert not eligible({**entry, "score_old": -13}, rule)
    assert not eligible({**entry, "score_prev_2": -21}, rule)
    stored = json.loads(json.dumps(rule))
    assert session_loop_rule({"loop": {}, "kept_rule": stored}) == stored
    assert session_loop_rule({"kept_rule": stored}) is None
    with pytest.raises(LoopError, match="do not follow"):
        session_loop_rule({"loop": {}, "kept_rule": {**stored, "s_old_floor": -40}})
    with pytest.raises(LoopError, match="has kept rule loop-retain-v1"):
        session_loop_rule({"loop": {}, "kept_rule": {**stored, "name": "joint-retain-v1"}})
    with pytest.raises(LoopError, match="at least one earlier round"):
        loop_kept_rule(4000, {})


def test_the_final_pick_is_the_best_program_above_both_floors() -> None:
    rule = loop_kept_rule(400, {1: 400})  # floors -1 and -2

    def checked(patch_id: str, query_index: int, score: int, old: int, prev: int) -> dict[str, Any]:
        return {"patch_id": patch_id, "query_index": query_index, "score": score, "score_old": old,
                "score_prev_1": prev, "score_new": score - old - prev, "n_changed": 10}

    entries = [checked("a.py", 0, 50, -1, -2), checked("b.py", 1, 90, -2, 0), checked("c.py", 2, 70, 0, -3)]
    assert select_final(entries, budget=8, declared=None, loop_rule=rule) == {
        "kind": KIND_CHECKED, "patch_id": "a.py", "query_index": 0}
    assert select_final(entries[1:], budget=8, declared=None, loop_rule=rule)["kind"] == KIND_BASE
    assert select_final([checked("d.py", 0, 0, 0, 0)], budget=8, declared=None, loop_rule=rule)["kind"] == KIND_BASE
    with pytest.raises(FinalizeError, match="not both"):
        select_final(entries, budget=8, declared=None, s_old_floor=-5, loop_rule=rule)
    with pytest.raises(FinalizeError, match="needs checked programs"):
        select_final(entries, budget=0, declared=None, loop_rule=rule)


def _swap_forward(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return (x.reshape(len(x), -1).astype(np.float32) @ params["w"]).astype(np.float32)


def test_the_multi_part_check_scores_each_part_against_the_current_student() -> None:
    base = {"w": np.eye(2, dtype=np.float32)}
    swap = {"w": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)}
    one_hot = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    parts = {"new": (one_hot[[0, 1, 0]], np.array([1, 1, 0])),  # base wrong on row 0; swap fixes 1, breaks 2
             "old": (one_hot, np.array([0, 1])),  # base right on both; swap breaks both
             "prev_1": (one_hot[[0]], np.array([1]))}  # base wrong; swap fixes it
    with pytest.raises(ValueError, match="must be"):
        MultiPartOracle({"old": parts["old"], "new": parts["new"]}, _swap_forward, base, 2)
    oracle = MultiPartOracle(parts, _swap_forward, base, 1)
    assert oracle.n_rows() == {"new": 3, "old": 2, "prev_1": 1}
    entry = oracle.query(swap, "swap")
    assert (entry["score_new"], entry["score_old"], entry["score_prev_1"]) == (-1, -2, 1)
    assert {key: entry[key] for key in ("score", "fixed", "broken", "n", "n_base_errors")} == {
        "score": -2, "fixed": 2, "broken": 4, "n": 6, "n_base_errors": 2}
    assert entry["base_acc"] == pytest.approx(4 / 6) and entry["child_acc"] == pytest.approx(2 / 6)
    assert entry["queries_remaining"] == 0 == oracle.queries_remaining
    with pytest.raises(ValueError, match="already checked"):
        oracle.query(swap, "swap")
    with pytest.raises(BudgetExceeded):
        oracle.query(swap, "again")


def test_the_prompt_texts_state_both_floors_and_keep_every_other_byte() -> None:
    text = render_system("LEARN")
    rule = loop_kept_rule(4000, {1: 4000, 2: 4000})
    out = loop_kept_section(text, rule)
    start = text.index(KEPT_HEADING) + len(KEPT_HEADING)
    end = text.index("\n\n## ", start)
    assert out[:start] == text[:start] and out.endswith(text[end:])
    section = out[start:out.index("\n\n## ", start)]
    flat = " ".join(section.split())
    assert "whose score_old is at least -12 and whose score_prev_k is at least -20" in flat
    assert "A program below either floor is never kept" in flat
    assert max(len(line) for line in section.splitlines()) <= 116
    with pytest.raises(LoopError, match="different row counts"):
        loop_kept_section(text, loop_kept_rule(4000, {1: 4000, 2: 2000}))
    with pytest.raises(LoopError, match="word for word"):
        loop_kept_section(out, rule)
    task = ("# Task\n\nDo it.\n\n## Files in /work/trial\n\n- `task.json`: this task as data.\n\n## Tools\n\n"
            "- `python`: run code.\n")
    files = loop_files_section(task, 47, 1)
    at = task.index(FILES_ANCHOR)
    assert files.startswith(task[:at]) and files.endswith(task[at:])
    added = files[at:len(files) - len(task) + at]
    assert "47 labelled examples" in " ".join(added.split()) and LOOP_LIBRARY_INDEX in added
    assert LOOP_LIBRARY_INDEX not in loop_files_section(task, 47, 0)
    assert max(len(line) for line in added.splitlines()) <= 116
    with pytest.raises(LoopError, match="single file list"):
        loop_files_section(task + FILES_ANCHOR, 47, 1)
    for shown in (flat, added, loop_statement(1), loop_statement(3), LOOP_READOUT, loop_final_selection(rule),
                  LOOP_STATUS_DESCRIPTION):
        assert not [word for word in HINT_WORDS if word in shown.lower()], shown


def test_the_round_library_lists_every_round_and_withholds_programs_with_forbidden_strings(tmp_path: Path) -> None:
    first = _write(tmp_path, "a.py", PROGRAM)
    third = _write(tmp_path, "c.py", PROGRAM + "# ibb1f3fc6\n")
    scores_3 = {"score_new": 9, "score_old": -2, "score_prev_1": 1, "score_prev_2": 0, "score": 8}
    library = [
        {"round": 1, "program": {"path": str(first), "sha256": file_sha256(first)},
         "scores": {"score_new": 5, "score_old": -1, "score": 4}, "source": {"session_id": "sAAAA1111"}},
        {"round": 2, "program": None, "scores": None, "source": {"session_id": "sBBBB2222"}},
        {"round": 3, "program": {"path": str(third), "sha256": file_sha256(third)}, "scores": scores_3,
         "source": {"session_id": "sCCCC3333"}},
    ]
    withheld = withheld_tokens(library, ["ibb1f3fc6", "toyblur", ""])
    assert withheld == {3: ["ibb1f3fc6"]}
    trial = tmp_path / "trial"
    trial.mkdir()
    record = write_loop_library(trial, library, 4000, withheld)
    assert sorted(path.name for path in trial.iterdir()) == [LOOP_LIBRARY_INDEX, "library_program_1.py"]
    index = json.loads((trial / LOOP_LIBRARY_INDEX).read_text())
    assert [(r["label"], r["round"], r["file"], r["kept_update"]) for r in index["rounds"]] == [
        ("round 1", 1, "library_program_1.py", True), ("round 2", 2, None, False), ("round 3", 3, None, True)]
    assert index["rounds"][2]["scores"] == scores_3 and index["rounds"][1]["scores"] is None
    assert index["n_check"] == 4000 and record["withheld"] == {"3": ["ibb1f3fc6"]}
    text = (trial / LOOP_LIBRARY_INDEX).read_text()
    for secret in ("ibb1f3fc6", "sAAAA1111", "sCCCC3333", str(tmp_path)):
        assert secret not in text
    assert record["files"]["library_program_1.py"] == file_sha256(first)
    with pytest.raises(LoopError, match="already exists"):
        write_loop_library(trial, library, 4000, withheld)
    other = tmp_path / "other"
    other.mkdir()
    first.write_text(PROGRAM + "# changed\n")
    with pytest.raises(LoopError, match="the kept program changed"):
        write_loop_library(other, library, 4000, withheld)


def test_a_loop_context_is_refused_before_any_session_write(learn_store: LearnStore, tmp_path: Path) -> None:
    sessions = runner.runs_dir() / runner.SESSIONS_DIR
    before = sorted(sessions.iterdir()) if sessions.exists() else []
    target = learn_store.ids[100]
    context = tmp_path / "context.json"
    common: dict[str, Any] = {"effort": "low", "wall_timeout_s": 60.0, "dry_run": True}
    with pytest.raises(ValueError, match="program arm only"):
        runner.run_session(target, "M2", 0, loop_context=context, **common)
    with pytest.raises(ValueError, match="no prompt override, retention floor or library"):
        runner.run_session(target, "LEARN", 0, loop_context=context, retention_floor_frac=0.01, **common)
    with pytest.raises(ValueError, match="dev instances only"):
        runner.run_session(target, "LEARN", 0, loop_context=context, allow_main=True, **common)
    with pytest.raises(InstanceRefused, match="refused for a loop round: .*is missing"):
        runner.run_session(target, "LEARN", 0, loop_context=context, **common)
    context.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(InstanceRefused, match="refused for a loop round: .*is not a weightcode.harness.loop_context"):
        runner.run_session(target, "LEARN", 0, loop_context=context, **common)
    context.write_text(json.dumps({"schema": LOOP_CONTEXT_SCHEMA, "round": 1, "prev": [], "library": []}))
    with pytest.raises(InstanceRefused, match="round 2 to 8"):
        runner.run_session(target, "LEARN", 0, loop_context=context, **common)
    assert (sorted(sessions.iterdir()) if sessions.exists() else []) == before
    with pytest.raises(LoopError, match="needs the evidence of rounds 1 to 2"):
        read_loop_context(_write(tmp_path, "c3.json", json.dumps(
            {"schema": LOOP_CONTEXT_SCHEMA, "round": 3, "prev": [{"round": 1}], "library": []})))


# ---------------------------------------------------------------------------------------------------------------
# Streams end to end


@pytest.fixture
def loop_store(learn_store: LearnStore) -> tuple[LearnStore, str, str]:
    """The learn store plus a second condition; the two seed-100 dev instances share the base model."""
    add_condition(learn_store.store, name=SECOND)
    other = {o.seed: o.instance_id for o in build_condition(learn_store.store, roles=("dev",), name=SECOND)
             if o.split_role == "dev"}
    return learn_store, learn_store.ids[100], other[100]


def _round_one(first: str) -> Path:
    """A finished scripted LEARN session on round 1's instance that keeps FIX: round 1's frozen program and scores."""

    def script(client: Any, replies: list[ToolReply]) -> None:
        call_and_record(client, replies, "python", {"code": _program_cell({"fix.py": FIX})})
        call_and_record(client, replies, "patch_check", {"program_file": "fix.py"})
        call_and_record(client, replies, "declare_final", {"program_file": "fix.py", "rationale": "it scored"})

    result = runner.run_session(first, "LEARN", 0, launcher=ScriptedLauncher(script), **SESSION_KWARGS)
    assert result.status == "complete" and result.record["final"]["kind"] == "checked", result.record["infra_reasons"]
    return result.root


def _round_two(client: Any, replies: list[ToolReply]) -> None:
    call_and_record(client, replies, "python", {"code": READ_CELL})
    call_and_record(client, replies, "python", {"code": _program_cell({"fix3.py": FIX3, "worse.py": WORSE})})
    call_and_record(client, replies, "patch_check", {"program_file": "worse.py"})
    call_and_record(client, replies, "patch_check", {"program_file": "fix3.py"})
    call_and_record(client, replies, "status", {})
    call_and_record(client, replies, "declare_final", {"program_file": "fix3.py", "rationale": "above both floors"})


@needs_sandbox
def test_a_teacher_stream_runs_round_two_on_the_round_one_student(loop_store: tuple[LearnStore, str, str],
                                                                  tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    _, first, second = loop_store
    root1 = _round_one(first)
    program1 = root1 / "final" / "fix.py"
    final1 = read_json(root1 / "final" / "selection.json")["final"]
    out = tmp_path / "loops"
    launchers: list[ScriptedLauncher] = []
    calls: list[int] = []

    def launcher_for(index: int) -> ScriptedLauncher:
        calls.append(index)
        if len(calls) == 1:
            raise RuntimeError("stopped before the round-2 launch")
        launchers.append(ScriptedLauncher(_round_two))
        return launchers[-1]

    def stream(**extra: Any) -> Path:
        return run_loop_stream(MODE_TEACHER, [first, second], program1, root1, out, _sandbox(),
                               session_kwargs=SESSION_KWARGS, launcher_for=launcher_for, **extra)

    with pytest.raises(RuntimeError, match="before the round-2 launch"):
        stream()
    (work,) = [path for path in out.iterdir() if path.is_dir()]
    assert [row["round"] for row in read_json(work / "state.json")["rounds"]] == [1]
    with pytest.raises(LoopStreamError, match="pass resume=True"):
        stream()

    real_student_file = loop_stream._student_file
    stops: list[Path] = []

    def stop_once(path: Path, params: dict[str, np.ndarray]) -> dict[str, str]:
        if path.name == "student_after_round_2.npz" and not stops:
            stops.append(path)
            raise RuntimeError("stopped after the round-2 session")
        return real_student_file(path, params)

    monkeypatch.setattr(loop_stream, "_student_file", stop_once)
    with pytest.raises(RuntimeError, match="after the round-2 session"):
        stream(resume=True)
    assert calls == [2, 2] and (work / "round_2_kept.json").is_file()
    record = read_json(stream(resume=True))
    assert calls == [2, 2], "a finished session is never launched again"
    with pytest.raises(LoopStreamError, match="never overwritten"):
        stream(resume=True)

    one, two = record["rounds"]
    assert [one["status"], two["status"]] == ["ok", "ok"] and record["summary"]["n_rounds_ok"] == 2
    assert one["library_entry"]["scores"] == {key: final1[key] for key in ("score_new", "score_old", "score")}
    assert two["parent_sha256"] == one["child_sha256"] != record["base_params_sha256"]
    assert two["final"]["kind"] == KIND_CHECKED and two["final"]["patch_id"] == "fix3.py"
    assert min(two["final"][key] for key in ("score_new", "score_old", "score_prev_1")) > 0
    assert two["readouts"][second]["select"]["delta_acc"] > 0
    assert not list(work.glob("*_incomplete_*")) and len(two["sessions"]) == 1
    session_root = Path(two["sessions"][0]["root"])
    paths = SessionPaths(session_root)
    session = read_json(paths.session_json)
    n_check = int(session["loop"]["prev_checks"][0]["n"])
    assert session["kept_rule"] == loop_kept_rule(n_check, {1: n_check})
    assert session["loop"]["prev_instances"] == [first] and session["loop"]["round"] == 2
    assert session["real_model_sha256"] == one["child_sha256"]
    assert params_sha256(load_params(paths.trial / "model.npz")) == one["child_sha256"]

    experience = _npz(paths.trial / "experience.npz")
    evidence1 = _npz(work / "evidence_round_1.npz")
    new_idx = _npz(work / "evidence_round_2.npz")["idx"]
    assert set(PREV_KEYS) <= set(experience) and set(experience["round_prev"].tolist()) == {1}
    kept, dropped = kept_prev_rows([(1, evidence1)], new_idx)
    positions = kept[0][1]
    assert np.array_equal(experience["x_prev"], evidence1["x"][positions])
    assert np.array_equal(experience["split_ids_prev"], evidence1["split_ids"][positions])
    assert len(experience["y_prev"]) + dropped["1"] == len(evidence1["y"])
    assert not set(evidence1["idx"][positions].tolist()) & set(new_idx.tolist())
    assert session["experience"]["prev_dropped_shared_image"] == dropped
    assert session["experience"]["prev_per_round"] == {"1": len(experience["y_prev"])}
    assert read_json(paths.trial / "task.json")["experience"]["prev"]["per_round"] == {"1": len(experience["y_prev"])}

    index = json.loads((paths.trial / LOOP_LIBRARY_INDEX).read_text())
    assert [(r["label"], r["file"], r["scores"]) for r in index["rounds"]] == [
        ("round 1", "library_program_1.py", one["library_entry"]["scores"])]
    assert (paths.trial / "library_program_1.py").read_bytes() == program1.read_bytes()
    system = " ".join(paths.system_prompt.read_text().split())
    task = " ".join(paths.task_prompt.read_text().split())
    assert f"score_old is at least {session['kept_rule']['s_old_floor']}" in system
    assert "`x_prev`" in task and LOOP_LIBRARY_INDEX in task

    (launcher,) = launchers
    shown = " ".join(str(reply.payload) for reply in launcher.replies)
    assert "round 1" in shown and "x_prev" in shown
    checks = [reply for reply in launcher.replies if reply.name == "patch_check"]
    assert len(checks) == 2 and all(set(reply.payload) == LOOP_SHOWN_CHECK_KEYS for reply in checks)
    for secret in (first, second, "toyblur", root1.name, str(work)):
        assert secret not in shown, secret

    selection = read_json(paths.selection_json)
    assert selection["final"]["parent"] == PARENT_LOOP and selection["kept_rule"]["name"] == KEPT_RULE_LOOP
    assert [(c["patch_id"], c["eligible"]) for c in selection["candidates"]] == [("worse.py", False), ("fix3.py", True)]
    with pytest.raises(DevReadoutError, match="unknown final.parent 'loop_student'"):
        readout_run(session_root / "run.json")

    # Control (a) on the same student with the kept program reaches the same child through the same check.
    kept_program = session_root / "final" / "fix3.py"
    control = read_json(run_loop_stream(MODE_SELECT, [first, second], program1, root1, tmp_path / "control",
                                        _sandbox(), candidates={2: [_write(tmp_path, "worse.py", WORSE),
                                                                    kept_program]}))
    picked = control["rounds"][1]
    assert picked["status"] == "ok" and picked["kept"] == "candidate_2" and picked["queries_used"] == 2
    assert [row["eligible"] for row in picked["candidates"]] == [False, True]
    assert picked["child_sha256"] == two["child_sha256"] and control["rounds"][0]["child_sha256"] == one["child_sha256"]
    assert picked["kept_rule"] == session["kept_rule"]
    session_check = next(entry for entry in map(json.loads, paths.checks_log.read_text().splitlines())
                         if entry.get("patch_id") == "fix3.py")
    for key in ("score", "score_new", "score_old", "score_prev_1", "fixed_prev_1", "broken_prev_1"):
        assert picked["candidates"][1]["check"][key] == session_check[key], key


@needs_sandbox
def test_checked_selection_keeps_the_student_when_no_candidate_clears_the_floors(
        loop_store: tuple[LearnStore, str, str], tmp_path: Path) -> None:
    _, first, second = loop_store
    root1 = _round_one(first)
    program1 = root1 / "final" / "fix.py"
    worse = _write(tmp_path, "worse.py", WORSE)
    record = read_json(run_loop_stream(MODE_SELECT, [first, second], program1, root1, tmp_path / "loops", _sandbox(),
                                       candidates={2: [worse]}))
    two = record["rounds"][1]
    assert two["status"] == "noop" and two["kept"] is None
    assert two["child_sha256"] == two["parent_sha256"] == record["rounds"][0]["child_sha256"]
    assert record["summary"]["n_rounds_ok"] == 1
    n_evidence = len(_npz(Path(record["work_dir"]) / "evidence_round_1.npz")["y"])
    assert two["experience"]["n_prev"] + sum(two["experience"]["prev_dropped_shared_image"].values()) == n_evidence
    with pytest.raises(LoopStreamError, match="every round from 2"):
        run_loop_stream(MODE_SELECT, [first, second], program1, root1, tmp_path / "a", _sandbox(), candidates={})
    with pytest.raises(LoopStreamError, match="takes no candidates"):
        run_loop_stream(MODE_TEACHER, [first, second], program1, root1, tmp_path / "b", _sandbox(),
                        candidates={2: [worse]})
    with pytest.raises(LoopStreamError, match="launches no session"):
        run_loop_stream(MODE_SELECT, [first, second], program1, root1, tmp_path / "c", _sandbox(),
                        candidates={2: [worse]}, session_kwargs=SESSION_KWARGS)
    with pytest.raises(LoopStreamError, match="is not the kept program"):
        run_loop_stream(MODE_SELECT, [first, second], worse, root1, tmp_path / "d", _sandbox(), candidates={2: [worse]})
    assert not (tmp_path / "d").exists()
    with pytest.raises(LoopStreamError, match="2 to 8 rounds"):
        run_loop_stream(MODE_SELECT, [first], program1, root1, tmp_path / "e", _sandbox(), candidates={})
