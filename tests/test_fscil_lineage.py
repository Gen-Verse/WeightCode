"""The E9 lineage driver (``weightcode.harness.fscil_lineage``) on the toy store: the floors and their refusals, the summary,
and lineages T, F and C over sessions 2 to 9 with a scripted editor (an injected launcher), stops and resumes.

The scripted editor writes ``imprint_program`` (``test_harness_fscil``) with its spoiling step at session 6, so that a
program applied without a check (F) breaks the earlier sessions' classes there while a check under the floor (T's
reuse, C) keeps the student. The toy floors are R = -10, d = 0 (about -12 to -19 rows). Every directory lives under a
neutral pytest tmp dir; the sandbox self-test needs a worktree path without a forbidden string, as there.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fscil_toy import ToyFscil, build_toy_fscil_store
from test_harness_fixtures import needs_sandbox
from test_harness_fscil import imprint_program, learn_steps, needs_neutral_repo, toy_floor

from weightcode.baselines.fit.fscilrefs import STREAMS_SCHEMA
from weightcode.common.identity import load_params
from weightcode.evalkit import fscil as FS
from weightcode.harness import fscil_lineage as L
from weightcode.harness.jsonio import read_json, utc_now, write_json
from weightcode.harness.runner import SANDBOX_PYTHON
from weightcode.harness.scripted import ScriptedLauncher, call_and_record
from weightcode.sandbox.program import ProgramSandbox

R_TOY, D_TOY = -10.0, 0.0
SPOIL = 80  # the first class of session 6


def floors_document(sessions: range = range(2, 10), r: float = R_TOY, d: float = D_TOY) -> dict[str, Any]:
    rows = {}
    for t in sessions:
        n_c, n_s = FS.old_part_sizes(t)
        rows[str(t)] = {"floor_t": FS.joint_retain_floor(n_c, n_s, r, d), "R": r, "d": d, "n": 16, "n_c": n_c,
                        "n_s": n_s}
    return {"schema": L.FLOORS_SCHEMA, "sessions": rows}


def write_floors(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ToyFscil]:
    with pytest.MonkeyPatch.context() as patch:
        yield build_toy_fscil_store(tmp_path_factory.mktemp("hs"), patch)


def test_the_floors_are_recomputed_and_any_disagreement_is_refused(toy: ToyFscil, tmp_path: Path) -> None:
    floors = L.load_floors(write_floors(tmp_path / "f.json", floors_document()))
    assert sorted(floors) == list(range(2, 10)) and floors[2].floor == toy_floor(2) == -12
    assert floors[2].source["R"] == R_TOY and floors[2].source["n_c"] == 120 and floors[9].source["n_c"] == 190
    zero = L.load_floors(write_floors(tmp_path / "z.json", floors_document(range(2, 3), r=-1.0, d=0.5)))
    assert zero[2].floor == 0  # never above 0, and 0 itself is a floor (PREREG E9)
    good = floors_document(range(2, 4))
    bad = [
        ({**good, "schema": "other"}, "is not a"),
        ({**good, "sessions": {**good["sessions"], "2": {**good["sessions"]["2"], "floor_t": -11}}}, "is not"),
        ({**good, "sessions": {**good["sessions"], "2": {**good["sessions"]["2"], "n_c": 121}}}, "old-part sizes"),
        ({**good, "sessions": {"2": {k: v for k, v in good["sessions"]["2"].items() if k != "d"}}}, "lacks"),
        ({**good, "sessions": {"2": {**good["sessions"]["2"], "n": True}}}, "must be an int"),
        ({**good, "sessions": {"10": good["sessions"]["2"]}}, "is not one of"),
        ({**good, "sessions": {}}, "no sessions"),
    ]
    for document, message in bad:
        with pytest.raises(L.LineageError, match=message):
            L.load_floors(write_floors(tmp_path / "bad.json", document))
    refs = tmp_path / "refs"
    refs.mkdir()
    write_json(refs / "run.json", {"schema": STREAMS_SCHEMA, "r_points": R_TOY,
                                   "inputs": {"base": {"params_sha256": toy.base_params_sha256}}})
    n_c, n_s = FS.old_part_sizes(2)
    block = {"n_c": n_c, "n_s": n_s, "d": D_TOY, "n_checks": 16, "floor": toy_floor(2)}
    write_json(refs / "session_2.json", {"schema": STREAMS_SCHEMA, "session": 2, "floor": block})
    from_refs = L.load_reference_floors(refs, toy.base_params_sha256)
    assert sorted(from_refs) == [2] and from_refs[2].floor == floors[2].floor and from_refs[2].source["n"] == 16
    with pytest.raises(L.LineageError, match="another base"):
        L.load_reference_floors(refs, "0" * 64)
    write_json(refs / "session_2.json", {"schema": STREAMS_SCHEMA, "session": 2, "floor": {**block, "floor": 0}})
    with pytest.raises(L.LineageError, match="joint_retain_floor"):
        L.load_reference_floors(refs, toy.base_params_sha256)


def test_the_summary_reads_the_absolute_readouts() -> None:
    def accuracy(t: int, seen: float, own: float) -> dict[str, Any]:
        return {"bal_seen": seen, "per_session": {str(s): {"bal": own if s == t else 0.5} for s in range(1, t + 1)}}

    state: dict[str, Any] = {"lineage": "C", "session_1": {"accuracy": accuracy(1, 0.8, 0.8)}, "sessions": {}}
    for t in range(2, 10):
        state["sessions"][str(t)] = {"done": True, "outcome": "kept", "accuracy": accuracy(t, 0.8 - 0.05 * t, 0.3)}
    summary = L.summarize(state, teacher_bal_new_2=0.6)
    assert summary["bal_seen"]["1"] == 0.8 and summary["bal_seen"]["9"] == pytest.approx(0.35)
    assert summary["bal_seen_drop_1_to_9"] == pytest.approx(0.45)
    assert summary["bal_seen_mean_1_to_9"] == pytest.approx((0.8 + sum(0.8 - 0.05 * t for t in range(2, 10))) / 9)
    assert summary["forgetting"]["1"] == pytest.approx(0.3) and summary["forgetting"]["9"] == 0.0
    assert summary["forgetting"]["4"] == pytest.approx(0.3 - 0.5) and summary["bal_new"]["5"] == 0.3
    assert summary["c_ratio"] == pytest.approx(0.5) and summary["outcomes"]["9"] == "kept"
    del state["sessions"]["9"]
    partial = L.summarize(state)
    assert partial["bal_seen_mean_1_to_9"] is None and partial["forgetting"]["3"] is None
    assert partial["c_ratio"] is None


def _sessions(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir()) if root.is_dir() else []


@needs_sandbox
@needs_neutral_repo
def test_the_lineages_run_stop_resume_and_summarise(toy: ToyFscil, tmp_path: Path) -> None:
    program = imprint_program(SPOIL)
    steps = learn_steps(program)

    def script(client: Any, replies: list[Any]) -> None:
        for name, arguments in steps:
            call_and_record(client, replies, name, arguments)

    launcher = ScriptedLauncher(script)
    floors = L.load_floors(write_floors(tmp_path / "floors.json", floors_document()))
    sandbox = ProgramSandbox(python=SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)
    options = {"effort": "low", "wall_timeout_s": 300.0, "threads": 2, "sandbox": sandbox,
               "session_kwargs": {"launcher": launcher, "skip_audit": True}}
    sessions_dir = Path(os.environ["WEIGHTCODE_RUNS"]) / "sessions"

    def driver(lineage: str, table: Any = floors, busy: Any = lambda: []) -> L.LineageDriver:
        return L.LineageDriver(lineage, floors=table, busy=busy, **options)

    with pytest.raises(L.LineageError, match="needs the floors file"):
        L.LineageDriver("C", floors=None)
    with pytest.raises(L.LineageError, match="may not set"):
        L.LineageDriver("T", floors=floors, session_kwargs={"retention_floor_rows": 0})
    with pytest.raises(L.LineageError, match="no state yet"):
        driver("F", None).run()
    with pytest.raises(L.LineageError, match="placeholder floor"):  # refused before any instance is built
        L.dry_run(floors=None)
    dry = L.dry_run(floors=None, placeholder_floor=-12, session_kwargs={"skip_audit": True})
    assert dry.status == "dry_run" and dry.record["selftest"]["status"] == "pass", dry.record
    assert read_json(dry.root / "session.json")["kept_rule"]["floor_source"]["placeholder"] is True
    assert not (L.lineages_root() / f"T_{toy.base_params_sha256[:12]}" / "state.json").exists()  # state untouched

    # T: a busy machine stops session 2 before its launch; nothing was launched
    with pytest.raises(L.LineageError, match="editor session runs"):
        driver("T", busy=lambda: ["4242"]).run(2)
    teacher = driver("T")
    state = read_json(teacher.state_path)
    assert not state["sessions"]["2"].get("attempts") and not state["sessions"]["2"].get("done")
    assert _sessions(sessions_dir) == []
    state = teacher.run(2)
    s2 = state["sessions"]["2"]
    assert s2["done"] and s2["outcome"] == "kept" and s2["final_kind"] == "checked" and s2["status"] == "complete"
    assert s2["floor"] == toy_floor(2) and s2["library_sessions"] is None and s2["kind"] == L.KIND_LEARN
    assert s2["program"]["sha256"] and Path(s2["program"]["file"]).read_text(encoding="utf-8") == program
    assert s2["child_path"].endswith("final/model.npz")

    # T, session 3: a session killed after its check counts as INFRA and runs once more on the same context
    state = teacher.run(3)
    first = state["sessions"]["3"]
    assert first["library_sessions"] == [s2["session_id"]]
    killed = sessions_dir / first["session_id"]
    shutil.move(killed / "session_result.json", tmp_path / "result.bak")
    reset = {key: first[key] for key in ("session", "instance_id", "instance_status", "parent_params_sha256", "kind",
                                         "floor", "floor_source", "library_sessions")}
    reset["attempts"] = [{"repeat": 0, "launched_at": first["attempts"][0]["launched_at"], "status": "pending"}]
    state["sessions"]["3"] = reset
    write_json(teacher.state_path, state)
    state = driver("T").run(3)
    s3 = state["sessions"]["3"]
    assert [a["status"] for a in s3["attempts"]] == ["infra", "complete"]
    assert s3["attempts"][0]["session_id"] == killed.name and s3["attempts"][0]["resolved_from"] == (
        "killed after a check")
    assert s3["session_id"] == s3["attempts"][1]["session_id"] != killed.name and s3["outcome"] == "kept"
    assert read_json(sessions_dir / s3["session_id"] / "session.json")["library"]["entries"][0]["label"] == "session 2"

    # T, session 4: an attempt recorded with no session is a stop before launch and is launched again
    state = read_json(teacher.state_path)
    state["sessions"]["4"] = {"session": 4, "attempts": [{"repeat": 0, "launched_at": utc_now(), "status": "pending"}]}
    write_json(teacher.state_path, state)
    state = driver("T").run(4)
    s4 = state["sessions"]["4"]
    assert [(a["status"], a["repeat"]) for a in s4["attempts"]] == [("not_launched", 0), ("complete", 1)]
    library = read_json(sessions_dir / s4["session_id"] / "session.json")["library"]
    assert [entry["label"] for entry in library["entries"]] == ["session 2", "session 3"]

    # T, sessions 5 to 9: the three kept programs on the current student, at most 3 checks, the floor of the session
    state = driver("T").run()
    for t in range(5, 10):
        record = state["sessions"][str(t)]
        assert record["kind"] == L.KIND_REUSE and len(record["runs"]) == 3 and record["pick"]["budget"] == 3
        assert [run["source_session"] for run in record["runs"]] == [2, 3, 4]
        assert record["floor"] == toy_floor(t) and record["pick"]["s_old_floor"] == toy_floor(t)
        assert all(run["check"]["score_old"] < toy_floor(t) for run in record["runs"]) == (t == 6)
        assert record["outcome"] == ("no-op" if t == 6 else "kept")
    assert state["sessions"]["6"]["child_sha256"] == state["sessions"]["5"]["child_sha256"]
    launched = _sessions(sessions_dir)
    assert len(launched) == 4  # s2, s3 twice, s4
    again = driver("T").run()  # finished sessions are never launched again
    assert _sessions(sessions_dir) == launched
    assert {t: r["child_sha256"] for t, r in again["sessions"].items()} == {
        t: r["child_sha256"] for t, r in state["sessions"].items()}
    summary = again["summary"]
    assert sorted(summary["bal_seen"]) == [str(t) for t in range(1, 10)]
    assert summary["bal_seen_mean_1_to_9"] is not None and summary["bal_seen_drop_1_to_9"] is not None
    assert sorted(summary["forgetting"]) == sorted(summary["bal_new"]) == [str(t) for t in range(1, 10)]
    assert summary["outcomes"]["6"] == "no-op" and "c_ratio" not in summary

    # F: T's session-2 program, never checked; its spoiling step at session 6 is applied
    frozen = driver("F", None).run()
    assert frozen["sessions"]["2"]["kind"] == L.KIND_FROM_TEACHER
    assert frozen["sessions"]["2"]["child_sha256"] == s2["child_sha256"]
    for t in range(3, 10):
        record = frozen["sessions"][str(t)]
        assert record["kind"] == L.KIND_FROZEN and record["outcome"] == "kept" and record["pick"] is None
        assert "check" not in record["runs"][0] and record["runs"][0]["status"] == "ok"
    six = frozen["sessions"]["6"]  # applied unchecked: its own check would have refused it
    assert FS.make_fscil_oracle(six["instance_id"], 1).query(load_params(six["child_path"]), "f")["score_old"] < (
        toy_floor(6))
    assert frozen["sessions"]["3"]["instance_id"] == state["sessions"]["3"]["instance_id"]  # same student: reused
    assert "c_ratio" in frozen["summary"] and frozen["summary"]["teacher_bal_new_2"] == s2["accuracy"][
        "per_session"]["2"]["bal"]

    # C: the same program applied only when its one check meets the floor; a missing floor stops it, then it resumes
    partial = {t: floor for t, floor in floors.items() if t != 3}
    with pytest.raises(L.LineageError, match="no floor for session 3"):
        driver("C", partial).run()
    checked = driver("C").run()
    for t in range(3, 10):
        record = checked["sessions"][str(t)]
        assert record["kind"] == L.KIND_CHECKED and record["pick"]["budget"] == 1
        assert record["outcome"] == ("no-op" if t == 6 else "kept")
    assert checked["sessions"]["6"]["pick"]["kind"] == "base"
    assert checked["sessions"]["6"]["child_sha256"] == checked["sessions"]["5"]["child_sha256"]
    assert checked["sessions"]["5"]["child_sha256"] == frozen["sessions"]["5"]["child_sha256"]
    assert checked["sessions"]["7"]["child_sha256"] != frozen["sessions"]["7"]["child_sha256"]
    assert set(checked["summary"]) >= {"bal_seen", "bal_seen_mean_1_to_9", "bal_seen_drop_1_to_9", "bal_new",
                                       "forgetting", "outcomes", "c_ratio", "teacher_bal_new_2"}
    assert _sessions(sessions_dir) == launched  # F and C make no LLM call
    json.dumps(checked, allow_nan=False)
