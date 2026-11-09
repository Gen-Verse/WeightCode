"""PREREG T16 follow-up M: the protected-entry limit (a PackNet-style hard mask on the entries the edit log lists).

Units: the protected entries (every entry the log lists, over all rounds), the comparison with the round's student in
the float32 sense of the log's derivation (a sub-float32 delta and -0.0 change nothing), the refusal message, and the
one system-prompt sentence after the edit-log sentence. Streams on the toy stores: arms N, L and M (L plus the limit)
on the same round-1 student with scripted round-2 sessions. In M the limit refuses a program that changes one logged
entry, as a limits refusal that uses no query, and admits programs that change only unlogged entries or a logged entry
by less than float32 resolves; M's prompts and trial equal L's except the one sentence, and with the option off N and
L are T16's arms byte for byte. A final that changes a logged entry is refused at finalisation (INFRA, base kept), as
any final whose re-application fails. The E11 two-round stream's dry run through the CLI in all three arms, and the
refusals (no edit log, checked selection, a tampered context). No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from e11_toy import PLANT_CONDITION, e11_store  # noqa: F401
from test_harness_e11 import LN_PLANT
from test_harness_fixtures import needs_sandbox
from test_harness_learn import STRINGS, LearnStore, _program_cell, learn_store  # noqa: F401
from test_harness_loop import SESSION_KWARGS, _round_one, _round_two, _sandbox, loop_store  # noqa: F401
from test_harness_loop_v2 import _e11_session
from test_harness_t16_edit_log import _assert_arms_differ_only_by_the_log, _child, _params, _trial

from weightcode.common.identity import load_params
from weightcode.harness import loop, loop_stream
from weightcode.harness.applying import REDACTION_PROTECTED, ArmApplier, ProtectedEntryChange, SessionConfigError
from weightcode.harness.finalize import KIND_BASE, STATUS_DRY_RUN, STATUS_INFRA, FinalizeError, finalize_session, write_final
from weightcode.harness.instance import read_instance_record
from weightcode.harness.jsonio import read_json, write_json
from weightcode.harness.launch import LaunchOutcome
from weightcode.harness.leaks import scan_hint_words
from weightcode.harness.ledger import CheckLedger
from weightcode.harness.loop import (
    EDIT_LOG_FILE,
    EDIT_LOG_SENTENCE,
    KEPT_RULE_LOOP_V2,
    PROTECT_SENTENCE,
    LoopError,
    edit_log_document,
    edit_log_section,
    loop_kept_rule_v2,
    loop_kept_section,
    protect_messages,
    protect_section,
    protected_changes,
    protected_counts,
    protected_entries,
)
from weightcode.harness.loop_stream import MODE_SELECT, MODE_TEACHER, LoopStreamError, run_loop_stream
from weightcode.harness.prompts import render_system
from weightcode.harness.scripted import ScriptedLauncher, ToolReply, call_and_record
from weightcode.harness.session import SessionPaths

PROPOSED_SENTENCE = ("A candidate or final that changes any entry listed in edit_log.json is refused by the "
                     "limits.")  # the coordinator's proposed wording, copied by hand; PREREG fixes it before M runs

# Round 1 (test_harness_loop.FIX) changed fc1.weight[0, 4] and [1, 4]: those two entries are logged and protected.
FIX3_LINES = "    delta[3, 7], delta[2, 7] = 6.0, -6.0\n"
PROT1 = STRINGS + ("import numpy as np\n\ndef update(student, experience):\n"
                   "    delta = np.zeros_like(student['params']['fc1.weight'])\n" + FIX3_LINES
                   + "    delta[0, 4] = 1.0\n    return {'fc1.weight': delta}\n")  # one logged entry changed
TINY = STRINGS + ("import numpy as np\n\ndef update(student, experience):\n"
                  "    delta = np.zeros(student['params']['fc1.weight'].shape)\n" + FIX3_LINES
                  + "    delta[0, 4] = 1e-12\n    return {'fc1.weight': delta}\n")  # below float32 resolution there
FIX3_ONLY = STRINGS + ("import numpy as np\n\ndef update(student, experience):\n"
                       "    delta = np.zeros_like(student['params']['fc1.weight'])\n" + FIX3_LINES
                       + "    return {'fc1.weight': delta}\n")  # unlogged entries only
ROUND_TWO_PROGRAMS = {"prot1.py": PROT1, "tiny.py": TINY, "fix3.py": FIX3_ONLY}
NOTHING = "Nothing was applied and no query was used."
REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------------------
# Units


def test_the_protected_entries_are_every_logged_entry_of_every_round() -> None:
    base = _params()
    one = _child(base)
    two = {name: value.copy() for name, value in one.items()}
    two["a.weight"][1, 2] -= 1.0  # logged again in round 2: protected once
    two["b.weight"][2, 1] += 3.0
    protected = protected_entries(edit_log_document([(1, base, one), (2, one, two)]))
    assert {name: indices.tolist() for name, indices in protected.items()} == {
        "a.weight": [8, 20, 23], "a.bias": [4], "conv.weight": [int(np.ravel_multi_index((1, 0, 1, 1), (2, 3, 2, 2)))],
        "b.weight": [7]}
    assert protected_counts(protected) == {"a.weight": 3, "a.bias": 1, "conv.weight": 1, "b.weight": 1}


def test_a_change_is_a_float32_value_change_as_in_the_logs_derivation() -> None:
    base = _params()
    w1 = _child(base)  # sets a.bias[4] to 0.0 and changes b.weight[0, 0] by nothing in float32
    protected = protected_entries(edit_log_document([(1, base, w1)]))
    assert w1["a.bias"][4] == 0.0 and "b.weight" not in protected

    def child(**edits: Any) -> dict[str, np.ndarray]:
        out = {name: value.copy() for name, value in w1.items()}
        for name, (index, value) in edits.items():
            out[name.replace("__", ".")][index] = value
        return out

    unlogged = child(a__weight=((0, 0), 5.0), b__weight=((0, 0), 9.0))
    one_logged = child(a__weight=((1, 2), w1["a.weight"][1, 2] + 1.0))
    sub_float32 = child(a__weight=((1, 2), np.float32(np.float64(w1["a.weight"][1, 2]) + 1e-12)))
    negative_zero = child(a__bias=((4,), np.float32(-0.0)))
    two_tensors = child(a__weight=((3, 5), 0.0), conv__weight=((1, 0, 1, 1), 7.0))
    assert sub_float32["a.weight"][1, 2] == w1["a.weight"][1, 2] and np.signbit(negative_zero["a.bias"][4])
    expected = [{}, {"a.weight": 1}, {}, {}, {"a.weight": 1, "conv.weight": 1}]
    for case, want in zip((unlogged, one_logged, sub_float32, negative_zero, two_tensors), expected):
        assert protected_changes(protected, w1, case) == want
        # the same sense as the log's derivation: the log of (w1 -> case) restricted to the protected entries
        later = edit_log_document([(1, base, w1), (2, w1, case)])["rounds"][1]["tensors"]
        derived = {t["tensor"]: len(set(t["flat_indices"]) & set(protected.get(t["tensor"], np.array([])).tolist()))
                   for t in later}
        assert {name: n for name, n in derived.items() if n} == want
    with pytest.raises(LoopError, match="does not have"):
        protected_changes({"absent": np.array([0])}, w1, w1)


def test_the_refusal_names_how_many_listed_entries_changed_and_in_which_tensors() -> None:
    assert protect_messages({"a.weight": 2, "conv.weight": 1}) == [
        "protected entries: the update changes 3 entries listed in edit_log.json (2 in tensor 'a.weight', 1 in tensor "
        "'conv.weight'); the limits refuse any change to a listed entry"]
    assert protect_messages({"fc1.weight": 1}) == [
        "protected entries: the update changes 1 entry listed in edit_log.json (1 in tensor 'fc1.weight'); the limits "
        "refuse any change to a listed entry"]


def test_the_limit_adds_exactly_one_sentence_after_the_edit_log_sentence() -> None:
    assert PROTECT_SENTENCE == PROPOSED_SENTENCE
    arm_n = loop_kept_section(render_system("LEARN"), loop_kept_rule_v2(4000, {1: 4000}, {1: 2}))
    arm_l = edit_log_section(arm_n)
    arm_m = protect_section(arm_l)
    assert arm_m.count(PROTECT_SENTENCE) == 1 and PROTECT_SENTENCE not in arm_l
    assert arm_m.replace(" " + PROTECT_SENTENCE, "", 1) == arm_l  # the sentence is the only difference
    assert EDIT_LOG_SENTENCE + " " + PROTECT_SENTENCE + "\n\n## What is kept\n" in arm_m  # right after, same paragraph
    assert scan_hint_words({"system": arm_m}) == {}
    with pytest.raises(LoopError, match="edit log once"):
        protect_section(arm_n)
    with pytest.raises(LoopError, match="already states"):
        protect_section(arm_m)


# ---------------------------------------------------------------------------------------------------------------
# Streams


def _round_two_limit(client: Any, replies: list[ToolReply]) -> None:
    """Round 2 of arms L and M: one program that changes one logged entry, one that changes a logged entry by less
    than float32 resolves, one that changes only unlogged entries."""
    call_and_record(client, replies, "python", {"code": _program_cell(ROUND_TWO_PROGRAMS)})
    for name in ("prot1.py", "tiny.py", "fix3.py"):
        call_and_record(client, replies, "patch_check", {"program_file": name})
    call_and_record(client, replies, "status", {})
    call_and_record(client, replies, "declare_final", {"program_file": "fix3.py", "rationale": "unlogged entries"})


def _assert_m_is_l_plus_the_limit(root_m: Path, root_l: Path) -> None:
    """Round-2 sessions of arms M and L on the same student: equal trial files (edit_log.json included) and task
    prompts, system prompts that differ by the one sentence, and the limit recorded in M only."""
    assert _trial(root_m) == _trial(root_l)
    assert (root_m / "prompts" / "task.md").read_bytes() == (root_l / "prompts" / "task.md").read_bytes()
    system_m = (root_m / "prompts" / "system.md").read_text()
    assert system_m == protect_section((root_l / "prompts" / "system.md").read_text())
    session_m, session_l = read_json(root_m / "session.json"), read_json(root_l / "session.json")
    limit = session_m["loop"]["protect_logged"]
    assert "protect_logged" not in session_l["loop"]
    assert limit["enabled"] is True and limit["sentence"] == PROPOSED_SENTENCE
    assert limit["edit_log_sha256"] == session_m["loop"]["edit_log"]["sha256"] == _trial(root_m)[EDIT_LOG_FILE]
    assert limit["student_sha256"] == session_m["real_model_sha256"] == session_l["real_model_sha256"]
    assert limit["n_protected"] == sum(limit["per_tensor"].values()) == sum(
        session_m["loop"]["edit_log"]["n_entries"].values())  # two rounds: the log's one round, every entry once
    for key in ("tools", "budget", "limits", "kept_rule"):
        assert session_m[key] == session_l[key], key
    for key in ("tool_timeout_ms", "connect_timeout_ms"):
        assert session_m["mcp"][key] == session_l["mcp"][key], key
    log_m, log_l = session_m["loop"]["edit_log"], session_l["loop"]["edit_log"]
    assert {k: v for k, v in log_m.items() if k != "source"} == {k: v for k, v in log_l.items() if k != "source"}
    assert {k: v for k, v in log_m["source"].items() if k != "path"} == {k: v for k, v in log_l["source"].items()
                                                                         if k != "path"}
    assert session_m["leak_scan"]["passed"] and session_l["leak_scan"]["passed"]


def _with_limit(session: dict[str, Any], limit: dict[str, Any]) -> dict[str, Any]:
    return {**session, "loop": {**session["loop"], "protect_logged": limit}}


@needs_sandbox
def test_arm_m_refuses_a_logged_entry_change_as_a_limit_and_keeps_arms_l_and_n_as_in_t16(
        loop_store: tuple[LearnStore, str, str], tmp_path: Path) -> None:  # noqa: F811
    _, first, second = loop_store
    root1 = _round_one(first)
    program1 = root1 / "final" / "fix.py"
    records, launchers = {}, {}
    for arm in ("n", "l", "m"):
        launchers[arm] = ScriptedLauncher(_round_two if arm == "n" else _round_two_limit)
        records[arm] = read_json(run_loop_stream(
            MODE_TEACHER, [first, second], program1, root1, tmp_path / "loops", _sandbox(),
            session_kwargs=SESSION_KWARGS, launcher_for=lambda index, arm=arm: launchers[arm],
            kept_rule=KEPT_RULE_LOOP_V2, edit_log=arm != "n", protect_logged=arm == "m", tag=f"t16_{arm}"))
    roots = {arm: Path(record["rounds"][1]["sessions"][-1]["root"]) for arm, record in records.items()}
    assert len({record["rounds"][0]["child_sha256"] for record in records.values()}) == 1  # one W1
    # Option off: arms N and L are T16's, byte for byte in what the editor gets, and state no limit anywhere.
    _assert_arms_differ_only_by_the_log(roots["l"], roots["n"])
    rule = read_json(roots["n"] / "session.json")["kept_rule"]
    system_n = loop_kept_section(render_system("LEARN"), rule)
    assert (roots["n"] / "prompts" / "system.md").read_text() == system_n
    assert (roots["l"] / "prompts" / "system.md").read_text() == edit_log_section(system_n)
    for arm in ("n", "l"):
        assert "protect_logged" not in records[arm] and "protect_logged" not in records[arm]["rounds"][1]
        assert "protect_logged" not in read_json(Path(records[arm]["rounds"][1]["context"]["path"]))
    # Option on: arm M is arm L plus the one sentence and the recorded limit.
    _assert_m_is_l_plus_the_limit(roots["m"], roots["l"])
    work = Path(records["m"]["work_dir"])
    w1 = load_params(work / "student_after_round_1.npz")
    limit = read_json(roots["m"] / "session.json")["loop"]["protect_logged"]
    assert limit["n_protected"] == 2 and limit["per_tensor"] == {"fc1.weight": 2}
    context = read_json(Path(records["m"]["rounds"][1]["context"]["path"]))
    assert context["protect_logged"] == {"schema": loop.PROTECT_SCHEMA, "n_protected": 2,
                                         "edit_log_sha256": context["edit_log"]["sha256"]}
    # The same program: checked in L, refused in M as a limits refusal that uses no query.
    by_arm = {arm: {reply.arguments.get("program_file"): reply for reply in launchers[arm].replies
                    if reply.name == "patch_check"} for arm in ("l", "m")}
    assert not by_arm["l"]["prot1.py"].is_error
    refused = by_arm["m"]["prot1.py"]
    assert refused.is_error and refused.payload["error"] == [
        "protected entries: the update changes 1 entry listed in edit_log.json (1 in tensor 'fc1.weight'); the limits "
        "refuse any change to a listed entry", NOTHING]
    assert refused.payload["queries_remaining"] == read_json(roots["m"] / "session.json")["budget"]["patch_checks"]
    for name in ("tiny.py", "fix3.py"):  # a sub-float32 delta on a logged entry changes nothing: admitted
        assert not by_arm["m"][name].is_error, by_arm["m"][name].payload
    assert by_arm["m"]["tiny.py"].payload["query_index"] == 0
    result_m, result_l = (read_json(roots[arm] / "session_result.json") for arm in ("m", "l"))
    assert (result_m["queries"]["used"], result_l["queries"]["used"]) == (2, 3)
    (rejected,) = result_m["rejected_attempts"]
    assert rejected["program_file"] == "prot1.py" and rejected["stage"] == "limits"
    assert rejected["redaction"] == REDACTION_PROTECTED and rejected["editor_messages"] == rejected["messages"]
    assert [entry["patch_id"] for entry in read_json(roots["m"] / "final" / "selection.json")["candidates"]] == [
        "tiny.py", "fix3.py"]
    # M's kept child changes no logged entry; the stream records the limit and its one refusal.
    final_m = load_params(roots["m"] / "final" / "model.npz")
    assert protected_changes(protected_entries(loop.check_edit_log(
        (roots["m"] / "trial" / EDIT_LOG_FILE).read_bytes(), 2)), w1, final_m) == {}
    stated = records["m"]["rounds"][1]["protect_logged"]
    assert stated == {**limit, "refused_attempts": 1}
    assert records["m"]["protect_logged"] == {"enabled": True, "rounds": {"2": stated}}
    assert records["m"]["edit_log"]["enabled"] is True
    stems = {arm: Path(record["work_dir"]).name for arm, record in records.items()}
    assert len(set(stems.values())) == 3

    # The final path: L kept or checked prot1.py, which changes a logged entry. With the limit in force, the kept
    # child's rebuild is refused, and finalisation keeps the base (INFRA) as for any final that fails re-application.
    session_l = read_json(roots["l"] / "session.json")
    paths_l = SessionPaths(roots["l"])
    protected_session = _with_limit(session_l, limit)
    prot1 = next(entry for entry in read_json(roots["l"] / "final" / "selection.json")["candidates"]
                 if entry["patch_id"] == "prot1.py")
    ledger = CheckLedger(session_l, paths_l)  # without the limit every logged check replays
    ledger.applier.protected = ArmApplier(protected_session).protected
    with pytest.raises(ProtectedEntryChange, match="1 entry listed in edit_log.json"):
        write_final(paths_l, ledger, {"kind": "checked", "patch_id": "prot1.py",
                                      "query_index": prot1["query_index"]}, None, "test")
    with pytest.raises(ProtectedEntryChange):
        CheckLedger(protected_session, paths_l)  # the replay of the logged check applies the limit too
    write_json(paths_l.session_json, protected_session)
    launch = LaunchOutcome(**read_json(paths_l.result_json)["launch"])
    with pytest.raises(FinalizeError, match="protected entries"):
        finalize_session(paths_l, protected_session, launch, skip_audit=True)
    again = read_json(paths_l.result_json)
    assert again["status"] == STATUS_INFRA and again["final"]["kind"] == KIND_BASE
    assert again["final"]["child_sha256"] == session_l["real_model_sha256"]
    assert again["final"]["parent"] == loop.PARENT_LOOP
    # A limit record that does not match the trial's edit log is refused when the applier is set up.
    with pytest.raises(SessionConfigError, match="protected-entry limit"):
        ArmApplier(_with_limit(session_l, {**limit, "per_tensor": {"fc1.weight": 3}}))
    with pytest.raises(SessionConfigError, match="protected-entry limit"):
        ArmApplier(_with_limit(session_l, {**limit, "student_sha256": "0" * 64}))


@needs_sandbox
def test_the_limit_is_refused_without_the_edit_log_in_checked_selection_and_on_a_tampered_context(
        loop_store: tuple[LearnStore, str, str], tmp_path: Path) -> None:  # noqa: F811
    _, first, second = loop_store
    root1 = _round_one(first)
    program1 = root1 / "final" / "fix.py"
    with pytest.raises(LoopStreamError, match="needs edit_log"):
        run_loop_stream(MODE_TEACHER, [first, second], program1, root1, tmp_path / "x", _sandbox(),
                        protect_logged=True)
    with pytest.raises(LoopStreamError, match="teacher round"):
        run_loop_stream(MODE_SELECT, [first, second], program1, root1, tmp_path / "x", _sandbox(),
                        candidates={2: [program1]}, edit_log=True, protect_logged=True)
    assert not (tmp_path / "x").exists()  # refused before any write
    args = ["--mode", MODE_TEACHER, "--instances", first, second, "--round1-session", str(root1),
            "--round1-program", str(program1), "--threads", "1", "--out", str(tmp_path / "cli")]
    for bad in (["--protect-logged"], ["--protect-logged", "--edit-log", "--mode", MODE_SELECT,
                                       "--candidates", "2=none"]):
        with pytest.raises(SystemExit):
            loop_stream.main([*args, *bad])
    assert not (tmp_path / "cli").exists()
    # A dry run with the limit: its context, then tampered copies of it, each refused before a session is written.
    dry = read_json(run_loop_stream(MODE_TEACHER, [first, second], program1, root1, tmp_path / "dry", _sandbox(),
                                    session_kwargs=SESSION_KWARGS, dry_run=True, edit_log=True, protect_logged=True,
                                    kept_rule=KEPT_RULE_LOOP_V2, tag="t16_mdry"))
    assert dry["status"] == STATUS_DRY_RUN and dry["protect_logged"]["refused_attempts"] == 0
    assert dry["protect_logged"]["n_protected"] == 2
    context = read_json(Path(dry["context"]["path"]))
    private = read_instance_record(second).private
    for tamper, match in (
            (lambda c: c.pop("edit_log"), "the context has none"),
            (lambda c: c["protect_logged"].update(edit_log_sha256="0" * 64), "another edit log"),
            (lambda c: c["protect_logged"].update(n_protected=3), "states 3 protected entries"),
            (lambda c: c["protect_logged"].update(schema="other"), "protect_logged is a"),
            (lambda c: c["edit_log"]["rounds"][-1].update(child_sha256="0" * 64), "does not end at this round"),
    ):
        changed = json.loads(json.dumps(context))
        tamper(changed)
        path = tmp_path / "tampered.json"
        write_json(path, changed)
        with pytest.raises(LoopError, match=match):
            loop.prepare_round(path, second, private)


@needs_sandbox
def test_the_e11_stream_dry_run_with_the_limit_is_arm_l_plus_one_sentence(
        e11_store: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    ids = [e11_store.ids[(PLANT_CONDITION, 100)], e11_store.ids[("contrast_s5", 100)]]
    root1 = _e11_session(ids[0], {"ln.py": LN_PLANT}, "ln.py")
    final1 = read_json(root1 / "final" / "selection.json")["final"]
    common = ["--mode", MODE_TEACHER, "--instances", *ids, "--round1-session", str(root1),
              "--round1-program", str(root1 / "final" / "ln.py"), "--round1-from-final", "--kept-rule",
              KEPT_RULE_LOOP_V2, "--dry-run", "--threads", "1", "--out", str(tmp_path / "dry")]
    roots = {}
    for arm, extra in (("m", ["--edit-log", "--protect-logged", "--tag", "t16_dm"]),
                       ("l", ["--edit-log", "--tag", "t16_dl"]), ("n", ["--tag", "t16_dn"])):
        assert loop_stream.main([*common, *extra]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["status"] == STATUS_DRY_RUN and printed["edit_log"]["enabled"] is (arm != "n")
        assert ("protect_logged" in printed) is (arm == "m")
        roots[arm] = Path(printed["root"])
        if arm == "m":
            limit = printed["protect_logged"]
            assert limit["enabled"] is True and limit["refused_attempts"] == 0
            assert limit["n_protected"] == int(final1["n_changed"]) == sum(limit["per_tensor"].values())
            assert limit["student_sha256"] == read_json(roots["m"] / "session.json")["real_model_sha256"]
    _assert_arms_differ_only_by_the_log(roots["l"], roots["n"])
    _assert_m_is_l_plus_the_limit(roots["m"], roots["l"])
    for arm in ("l", "n"):
        assert "protect_logged" not in read_json(roots[arm] / "session.json")["loop"]
    # The fairness script's M check, as run on the NSCC dry run (arm N stands in for the session of record here).
    spec = importlib.util.spec_from_file_location("t16_fairness_check", REPO / "scripts" / "t16_fairness_check.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    report_path = tmp_path / "fairness.json"
    assert script.main(["--record", str(roots["n"]), "--off", str(roots["n"]), "--on", str(roots["l"]),
                        "--protect", str(roots["m"]), "--out", str(report_path)]) == 0
    report = read_json(report_path)
    assert report["passed"] and all(report["protect_checks"].values()) and report["protect_vs_on"] == {
        "system_sha256": [report["on"]["system_sha256"], report["protect"]["system_sha256"]]}
    assert report["protect_n_protected"] == int(final1["n_changed"])
    capsys.readouterr()
    assert script.main(["--record", str(roots["n"]), "--off", str(roots["n"]), "--on", str(roots["m"]),
                        "--protect", str(roots["l"])]) == 1  # arms swapped: refused
