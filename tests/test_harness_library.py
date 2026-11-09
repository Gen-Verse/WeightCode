"""The program library of a program-arm session (``weightcode.harness.library``, LEARN v3, dev only): which earlier sessions
it accepts, the files it writes into the trial, the one bullet it adds to the task prompt, the runner's refusals, and
one scripted LEARN session over the real MCP server that reads a library built from an earlier scripted session on
another condition of the same student.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from test_evalkit_instances_store import add_condition, build_condition
from test_harness_fixtures import needs_sandbox  # noqa: F401
from test_harness_learn import FIX, LearnStore, _program_cell, learn_store  # noqa: F401
from weightcode.analysis.session_info import format_table, session_info
from weightcode.harness import runner
from weightcode.harness.instance import InstanceRefused
from weightcode.harness.jsonio import read_json
from weightcode.harness.library import (
    LIBRARY_INDEX_FILE,
    LIBRARY_PROGRAM_TEMPLATE,
    LibraryError,
    index_document,
    library_entries,
    write_library,
)
from weightcode.harness.prompts import LIBRARY_FILES_ANCHOR, library_files_section
from weightcode.harness.scripted import ScriptedLauncher, ToolReply, call_and_record
from weightcode.harness.session import SessionPaths

TARGET = "itarget00"
MODEL = "mtoy"
CLEAN = "c" * 64
SA, SB, IA, IB = "s0001aaaa", "s0002bbbb", "iaaaa1111", "ibbbb2222"
PROGRAM = "HYPOTHESIS = 'h'\nPREDICTED_EFFECT = 'p'\n\ndef update(student, experience):\n    return {}\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_session(sessions: Path, session_id: str, instance_id: str, **override: Any) -> Path:
    """A minimal finished LEARN session on disk, as library_entries reads it; ``override`` breaks one part."""
    root = sessions / session_id
    (root / "final").mkdir(parents=True)
    (root / "logs").mkdir()
    program = override.get("program", PROGRAM).encode()
    (root / "final" / "kept.py").write_bytes(program)
    session = {"arm": override.get("arm", "LEARN"), "instance_id": instance_id, "dry_run": override.get("dry_run", False),
               "model_id": override.get("model_id", MODEL), "condition": "cond_" + instance_id,
               "clean_check": {"clean_sha256": override.get("clean_sha256", CLEAN), "n": 40}}
    if override.get("fewshot"):
        session["fewshot"] = {"n_vcheck_old": 20}
    (root / "session.json").write_text(json.dumps(session))
    (root / "session_result.json").write_text(json.dumps({"status": override.get("status", "complete")}))
    final = {"kind": override.get("kind", "checked"), "patch_id": "kept.py", "query_index": 1,
             "score_new": 12, "score_old": -3, "score": 9}
    (root / "final" / "selection.json").write_text(json.dumps({"final": final}))
    checks = [{"query_index": 0, "program_sha256": "0" * 64, "score_new": 1, "score_old": -9, "score": -8,
               "n_new": 50, "n_old": 40},
              {"query_index": 1, "program_sha256": override.get("program_sha256", _sha(program)),
               "score_new": override.get("check_score_new", 12), "score_old": -3, "score": 9, "n_new": 50,
               "n_old": 40}]
    (root / "logs" / "checks.jsonl").write_text("".join(json.dumps(check) + "\n" for check in checks))
    return root


def _entries(sessions: Path, ids: list[str]) -> list[dict[str, Any]]:
    return library_entries(ids, sessions, target_instance_id=TARGET, target_model_id=MODEL,
                           target_clean_sha256=CLEAN)


def test_the_library_takes_finished_checked_programs_of_other_conditions(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    _source_session(sessions, SA, IA)
    _source_session(sessions, SB, IB, program=PROGRAM + "# b\n")
    entries = _entries(sessions, [SA, SB])
    assert [(e["k"], e["label"], e["file"]) for e in entries] == [
        (1, "earlier condition 1", "library_program_1.py"), (2, "earlier condition 2", "library_program_2.py")]
    assert {key: entries[0][key] for key in ("score_new", "score_old", "score", "n_check_new", "n_check_clean")} == {
        "score_new": 12, "score_old": -3, "score": 9, "n_check_new": 50, "n_check_clean": 40}
    assert entries[1]["source"]["session_id"] == SB and entries[1]["_bytes"] == (PROGRAM + "# b\n").encode()
    refusals = {
        "the target instance": dict(instance_id=TARGET),
        "the base model": dict(kind="base"),
        "did not complete": dict(status="aborted_leak"),
        "is a dry run": dict(dry_run=True),
        "fewshot-v1": dict(fewshot=True),
        "writes no update program": dict(arm="M2"),
        "ran on student": dict(model_id="mother"),
        "are not the target's": dict(clean_sha256="d" * 64),
        "does not hash": dict(program_sha256="1" * 64),
        "scores differ": dict(check_score_new=13),
    }
    for k, (message, override) in enumerate(refusals.items()):
        instance = override.pop("instance_id", f"ic{k:03d}")
        _source_session(sessions, f"sbad{k}", instance, **override)
        with pytest.raises(LibraryError, match=message):
            _entries(sessions, [SA, f"sbad{k}"])
    _source_session(sessions, "s0003aaaa", IA)
    with pytest.raises(LibraryError, match=f"two library sessions ran on instance {IA}"):
        _entries(sessions, [SA, "s0003aaaa"])
    with pytest.raises(LibraryError, match="listed twice"):
        _entries(sessions, [SA, SA])
    with pytest.raises(LibraryError, match="at least one"):
        _entries(sessions, [])
    with pytest.raises(LibraryError, match="is missing"):
        _entries(sessions, ["snone"])


def test_the_trial_files_hold_the_programs_and_scores_only(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    _source_session(sessions, SA, IA)
    _source_session(sessions, SB, IB, program=PROGRAM + "# b\n")
    entries = _entries(sessions, [SA, SB])
    trial = tmp_path / "trial"
    trial.mkdir()
    record = write_library(trial, entries)
    assert sorted(path.name for path in trial.iterdir()) == [LIBRARY_INDEX_FILE, "library_program_1.py",
                                                              "library_program_2.py"]
    assert (trial / "library_program_2.py").read_bytes() == (PROGRAM + "# b\n").encode()
    assert record["files"] == {path.name: _sha(path.read_bytes()) for path in trial.iterdir()}
    index = json.loads((trial / LIBRARY_INDEX_FILE).read_text())
    assert index == index_document(entries)
    assert [set(program) for program in index["programs"]] == [
        {"label", "file", "score_new", "score_old", "score", "n_check_new", "n_check_clean"}] * 2
    text = (trial / LIBRARY_INDEX_FILE).read_text()
    for secret in (SA, SB, IA, IB, "cond_", "kept.py", str(sessions)):
        assert secret not in text
    assert record["entries"][0]["source"]["program_sha256"] == _sha(PROGRAM.encode())
    assert all("_bytes" not in entry for entry in record["entries"])
    with pytest.raises(LibraryError, match="already exists"):
        write_library(trial, entries)


def test_the_task_prompt_gains_one_bullet_at_the_end_of_its_file_list() -> None:
    text = ("# Task\n\nDo it.\n\n## Files in /work/trial\n\n- `task.json`: this task as data.\n\n## Tools\n\n"
            "- `python`: run code.\n")
    out = library_files_section(text, 4, LIBRARY_INDEX_FILE, LIBRARY_PROGRAM_TEMPLATE)
    at = text.index(LIBRARY_FILES_ANCHOR)
    assert out.startswith(text[:at]) and out.endswith(text[at:])
    added = out[at:len(out) - len(text) + at]
    assert added.startswith("\n- `library_index.json` and `library_program_1.py` to `library_program_4.py`: 4 update")
    assert "none of them for this task's condition" in " ".join(added.split())
    assert max(len(line) for line in added.splitlines()) <= 116 and "\n\n" not in added
    one = library_files_section(text, 1, LIBRARY_INDEX_FILE, LIBRARY_PROGRAM_TEMPLATE)
    assert "`library_index.json` and `library_program_1.py`: one update program written" in " ".join(one.split())
    with pytest.raises(ValueError):
        library_files_section(text, 0, LIBRARY_INDEX_FILE, LIBRARY_PROGRAM_TEMPLATE)
    with pytest.raises(ValueError, match="single file list"):
        library_files_section(text + LIBRARY_FILES_ANCHOR, 2, LIBRARY_INDEX_FILE, LIBRARY_PROGRAM_TEMPLATE)


def test_the_runner_refuses_a_library_before_any_write(learn_store: LearnStore) -> None:
    sessions = runner.runs_dir() / runner.SESSIONS_DIR
    before = sorted(sessions.iterdir()) if sessions.exists() else []
    target = learn_store.ids[100]
    with pytest.raises(ValueError, match="program arm only"):
        runner.run_session(target, "M2", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                           library_sessions=["sany"])
    with pytest.raises(ValueError, match="no prompt override"):
        runner.run_session(target, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                           library_sessions=["sany"], task_prompt_override="x")
    with pytest.raises(InstanceRefused, match="refused for a library session: .*session.json is missing"):
        runner.run_session(target, "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                           library_sessions=["snone"])
    assert (sorted(sessions.iterdir()) if sessions.exists() else []) == before


@needs_sandbox
def test_a_learn_session_reads_a_library_from_another_condition(learn_store: LearnStore) -> None:
    add_condition(learn_store.store, name="toyblur_s3")
    other = {o.seed: o.instance_id for o in build_condition(learn_store.store, roles=("dev",), name="toyblur_s3")
             if o.split_role == "dev"}

    def earlier(client: Any, replies: list[ToolReply]) -> None:
        call_and_record(client, replies, "python", {"code": _program_cell({"fix.py": FIX})})
        call_and_record(client, replies, "patch_check", {"program_file": "fix.py"})
        call_and_record(client, replies, "declare_final", {"program_file": "fix.py", "rationale": "it scored"})

    first = runner.run_session(other[100], "LEARN", 0, effort="low", wall_timeout_s=300.0,
                               launcher=ScriptedLauncher(earlier), skip_audit=True)
    assert first.status == "complete" and first.record["final"]["kind"] == "checked", first.record.get("infra_reasons")
    with pytest.raises(InstanceRefused, match="ran on the target instance"):  # its own condition is never in its library
        runner.run_session(other[100], "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                           library_sessions=[first.session_id])
    with pytest.raises(InstanceRefused, match="ran on student"):  # another seed is another student
        runner.run_session(learn_store.ids[101], "LEARN", 0, effort="low", wall_timeout_s=60.0, dry_run=True,
                           library_sessions=[first.session_id])

    read_cell = ("import json\nindex = json.load(open('/work/trial/library_index.json'))\n"
                 "print(index['programs'])\nprint(open('/work/trial/library_program_1.py').read()[:40])\n")

    def later(client: Any, replies: list[ToolReply]) -> None:
        call_and_record(client, replies, "python", {"code": read_cell})
        call_and_record(client, replies, "python", {"code": _program_cell({"fix.py": FIX})})
        call_and_record(client, replies, "patch_check", {"program_file": "fix.py"})
        call_and_record(client, replies, "declare_final", {"program_file": "fix.py", "rationale": "it scored"})

    launcher = ScriptedLauncher(later)
    second = runner.run_session(learn_store.ids[100], "LEARN", 0, effort="low", wall_timeout_s=300.0,
                                launcher=launcher, skip_audit=True, retention_floor_frac=0.02,
                                library_sessions=[first.session_id])
    assert second.status == "complete", second.record.get("infra_reasons")
    paths = SessionPaths(second.root)
    session = read_json(paths.session_json)
    library = session["library"]
    assert [entry["source"]["session_id"] for entry in library["entries"]] == [first.session_id]
    first_final = SessionPaths(first.root).root / "final" / "fix.py"
    assert (paths.trial / "library_program_1.py").read_bytes() == first_final.read_bytes()
    for name, digest in library["files"].items():
        assert session["trial_files_sha256"][name] == digest == _sha((paths.trial / name).read_bytes())
    assert session["leak_scan"]["passed"] and "kept_rule" in session
    task = paths.task_prompt.read_text(encoding="utf-8")
    assert "`library_index.json` and `library_program_1.py`: one update program written" in " ".join(task.split())
    shown = str(launcher.replies[0].payload)
    assert "earlier condition 1" in shown and "HYPOTHESIS" in shown
    assert first.session_id not in shown and other[100] not in shown
    info = session_info(second.root)
    assert [(e["label"], e["source"]["session_id"]) for e in info["library"]] == [
        ("earlier condition 1", first.session_id)]
    assert "library 1 programs" in format_table(info) and session_info(first.root)["library"] is None
