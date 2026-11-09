"""Round 1 of a loop stream from a recorded student (``round1_from_final``, ``round1_from_record``), dev only.

On the loop toy store (``test_harness_loop``): a control (a) stream whose round 1 applies the round-1 session's saved
final runs no program in round 1 and reaches the child of the stream that runs the program, under another stem; the
default stream keeps its stem and names its source. A stream whose round 1 comes from a loop record (a finished record
or a running stream's state.json) takes exactly that record's round-1 student, also one the program would not give,
and loop-retain-v2 then counts round 1 with the loop's own check. Tampered finals and records, and wrong flag
combinations, are refused before any write. No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_harness_fixtures import needs_sandbox
from test_harness_learn import WORSE, LearnStore, learn_store  # noqa: F401
from test_harness_loop import SESSION_KWARGS, _round_one, _sandbox, _write, loop_store  # noqa: F401

from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.harness import loop_stream, runner
from weightcode.harness.jsonio import file_sha256, read_json
from weightcode.harness.loop import KEPT_RULE_LOOP_V2, ROUND1_SESSION_CHECK
from weightcode.harness.loop_stream import (
    MODE_SELECT,
    ROUND1_FROM_BASE,
    ROUND1_FROM_FINAL,
    ROUND1_FROM_PROGRAM,
    ROUND1_FROM_RECORD,
    ROUND1_LOOP_ORACLE,
    LoopStreamError,
    run_loop_stream,
)
from weightcode.harness.scripted import ScriptedLauncher, call_and_record


def _no_program_run(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("round 1 ran its program")


def _control(ids: list[str], program: Path, session: Path, out: Path, worse: Path, **kwargs: Any) -> dict[str, Any]:
    return read_json(run_loop_stream(MODE_SELECT, ids, program, session, out, _sandbox(), candidates={2: [worse]},
                                     **kwargs))


def _copy_session(root: Path, dest: Path) -> Path:
    copy = dest / root.name
    shutil.copytree(root, copy)
    return copy


@needs_sandbox
def test_round_one_from_the_session_final_runs_no_program_and_reaches_the_programs_child(
        loop_store: tuple[LearnStore, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    _, first, second = loop_store
    ids = [first, second]
    root1 = _round_one(first)
    program1 = root1 / "final" / "fix.py"
    final1 = read_json(root1 / "final" / "selection.json")["final"]
    worse = _write(tmp_path, "worse.py", WORSE)
    default = _control(ids, program1, root1, tmp_path / "default", worse)
    shown = {"2": [loop_stream._program(worse)]}
    stem = loop_stream._stem(MODE_SELECT, ids, {"sha256": file_sha256(program1)}, shown, {})
    assert Path(default["work_dir"]).name == f"loop_{MODE_SELECT}_{stem}_{default['base_params_sha256'][:12]}"
    assert default["round1"]["source"] == default["rounds"][0]["round1_source"] == {"kind": ROUND1_FROM_PROGRAM}
    assert "run" in default["rounds"][0]

    monkeypatch.setattr(loop_stream, "_round", _no_program_run)
    record = _control(ids, program1, root1, tmp_path / "final", worse, round1_from_final=True)
    one, two = record["rounds"]
    assert one["status"] == "ok" and "run" not in one
    assert one["child_sha256"] == final1["child_sha256"] == default["rounds"][0]["child_sha256"]
    assert params_sha256(load_params(Path(one["student_after"]["path"]))) == final1["child_sha256"]
    source = one["round1_source"]
    assert record["round1"]["source"] == source and source["kind"] == ROUND1_FROM_FINAL
    assert source["session_id"] == root1.name and source["child_sha256"] == final1["child_sha256"]
    assert source["deltas_sha256"] == final1["program"]["deltas_sha256"] == file_sha256(root1 / "final" / "deltas.npz")
    assert source["patch_sha256"] == final1["program"]["patch_sha256"]
    assert one["experience"]["content_sha256"] == default["rounds"][0]["experience"]["content_sha256"]
    assert one["n_changed"] == default["rounds"][0]["n_changed"] == one["n_changed_vs_base"]
    assert one["library_entry"] == default["rounds"][0]["library_entry"]
    assert two["parent_sha256"] == final1["child_sha256"] and two["child_sha256"] == default["rounds"][1]["child_sha256"]
    assert Path(record["work_dir"]).name != Path(default["work_dir"]).name  # a recorded source: another identity
    assert not (Path(record["work_dir"]) / "round_1").exists()


@needs_sandbox
def test_a_tampered_final_or_a_wrong_combination_is_refused_before_any_write(
        loop_store: tuple[LearnStore, str, str], tmp_path: Path) -> None:  # noqa: F811
    _, first, second = loop_store
    ids = [first, second]
    root1 = _round_one(first)
    worse = _write(tmp_path, "worse.py", WORSE)
    out = tmp_path / "out"

    def refused(session: Path, match: str, **kwargs: Any) -> None:
        program = kwargs.pop("program", session / "final" / "fix.py")
        with pytest.raises(LoopStreamError, match=match):
            _control(ids, program, session, out, worse, **kwargs)
        assert not out.exists()

    refused(root1, "not both", round1_from_final=True, round1_from_record=tmp_path / "r.json")
    refused(root1, "needs the round-1 program", program=None, round1_from_final=True)
    model = _copy_session(root1, tmp_path / "model")
    child = load_params(model / "final" / "model.npz")
    save_params(model / "final" / "model.npz", {**child, "fc2.bias": child["fc2.bias"] + np.float32(1.0)})
    refused(model, "model.npz does not hash to the final child", round1_from_final=True)
    patch = _copy_session(root1, tmp_path / "patch")
    text = (patch / "final" / "patch.json").read_text()
    (patch / "final" / "patch.json").write_text(text.replace('"hypothesis"', '"hypothesis" ', 1))
    refused(patch, "does not hash to the final's patch_sha256", round1_from_final=True)
    other = _copy_session(root1, tmp_path / "other")
    session = read_json(other / "session.json")
    (other / "session.json").write_text(json.dumps({**session, "instance_id": second}))
    refused(other, "is not a child of round 1's instance", round1_from_final=True)


@needs_sandbox
def test_round_one_from_a_loop_record_takes_exactly_that_records_student(
        loop_store: tuple[LearnStore, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    _, first, second = loop_store
    ids = [first, second]
    root1 = _round_one(first)
    program1 = root1 / "final" / "fix.py"
    final1 = read_json(root1 / "final" / "selection.json")["final"]
    worse = _write(tmp_path, "worse.py", WORSE)
    default_path = run_loop_stream(MODE_SELECT, ids, program1, root1, tmp_path / "default", _sandbox(),
                                   candidates={2: [worse]})
    default = read_json(default_path)
    w1 = default["rounds"][0]["child_sha256"]
    monkeypatch.setattr(loop_stream, "_round", _no_program_run)
    state = Path(default["work_dir"]) / loop_stream.STATE_FILE
    shared = _control(ids, program1, root1, tmp_path / "state", worse, round1_from_record=state)
    source = shared["rounds"][0]["round1_source"]
    assert shared["rounds"][0]["child_sha256"] == w1 and source["kind"] == ROUND1_FROM_RECORD
    assert source["stem"] == Path(default["work_dir"]).name and source["file_sha256"] == file_sha256(state)
    assert source["student_after"] == default["rounds"][0]["student_after"]

    # A record whose round-1 student is not what the program gives: the stream takes that student, not the program's.
    child = load_params(Path(default["rounds"][0]["student_after"]["path"]))
    other = {**child, "fc2.bias": (child["fc2.bias"] + np.float32(1e-3)).astype(child["fc2.bias"].dtype)}
    other_sha = save_params(tmp_path / "other_w1.npz", other)
    assert other_sha != w1
    rows = [dict(row) for row in default["rounds"]]
    rows[0].update(child_sha256=other_sha, student_after={"path": str(tmp_path / "other_w1.npz"), "sha256": other_sha})
    fabricated = tmp_path / "fabricated.json"
    fabricated.write_text(json.dumps({**default, "rounds": rows}))
    record = _control(ids, program1, root1, tmp_path / "fabricated", worse, round1_from_record=fabricated,
                      kept_rule=KEPT_RULE_LOOP_V2)
    one, two = record["rounds"]
    assert one["child_sha256"] == other_sha == two["parent_sha256"] and record["round1"]["source"]["path"] == str(
        fabricated.resolve())
    assert one["clean_s_old"]["source"]["kind"] == ROUND1_LOOP_ORACLE  # not the session's child: the loop's check
    assert one["clean_s_old"]["source"]["why"] == ["the session's kept child is not the stream's round-1 student"]
    final_v2 = _control(ids, program1, root1, tmp_path / "final_v2", worse, round1_from_final=True,
                        kept_rule=KEPT_RULE_LOOP_V2)
    assert final_v2["rounds"][0]["clean_s_old"]["source"]["kind"] == ROUND1_SESSION_CHECK
    assert final_v2["rounds"][0]["clean_s_old"]["s_old"] == final1["score_old"]

    out = tmp_path / "refused"

    def refused(record_path: Path, match: str) -> None:
        with pytest.raises(LoopStreamError, match=match):
            _control(ids, program1, root1, out, worse, round1_from_record=record_path)
        assert not out.exists()

    def variant(name: str, **changes: Any) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps({**default, "rounds": [{**default["rounds"][0], **changes}, *default["rounds"][1:]]}))
        return path

    refused(variant("tampered.json", child_sha256="0" * 64), "does not hash to round 1's child_sha256")
    refused(variant("program.json", library_entry={**default["rounds"][0]["library_entry"],
                                                   "program": {"sha256": "1" * 64}}), "did not apply the program")
    refused(variant("instance.json", instance_id=second), "is not on round 1's instance")
    refused(root1 / "final" / "selection.json", "is not a loop stream record")
    kept_base = runner.run_session(first, "LEARN", 0, launcher=ScriptedLauncher(
        lambda client, replies: call_and_record(client, replies, "status", {})), **SESSION_KWARGS).root
    noop = read_json(run_loop_stream(MODE_SELECT, ids, None, kept_base, tmp_path / "noop", _sandbox(),
                                     candidates={2: [worse]}))
    assert noop["round1"]["source"] == noop["rounds"][0]["round1_source"] == {"kind": ROUND1_FROM_BASE}


@needs_sandbox
def test_the_cli_takes_one_recorded_source_and_needs_the_round_one_program(
        loop_store: tuple[LearnStore, str, str], tmp_path: Path) -> None:  # noqa: F811
    _, first, second = loop_store
    root1 = _round_one(first)
    program1 = root1 / "final" / "fix.py"
    worse = _write(tmp_path, "worse.py", WORSE)
    common = ["--mode", MODE_SELECT, "--instances", first, second, "--round1-session", str(root1),
              "--candidates", f"2={worse}", "--threads", "1"]
    for extra in (["--round1-from-final", "--round1-from-record", str(tmp_path / "r.json")], ["--round1-from-final"]):
        with pytest.raises(SystemExit):
            loop_stream.main([*common, *extra, *(["--round1-program", str(program1)] if len(extra) > 1 else []),
                              "--out", str(tmp_path / "x")])
    assert not (tmp_path / "x").exists()
    out = tmp_path / "cli"
    assert loop_stream.main([*common, "--round1-program", str(program1), "--round1-from-final", "--out", str(out)]) == 0
    (path,) = list(out.glob("loop_checked_selection_*.json"))
    assert read_json(path)["round1"]["source"]["kind"] == ROUND1_FROM_FINAL
