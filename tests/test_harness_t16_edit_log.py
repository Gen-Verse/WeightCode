"""PREREG T16: the edit log in a loop stream's later rounds (arm L) against the stream as of record (arm N).

Units: the log's derivation (every changed entry of each earlier round, flat indices, the rows and columns of a 2-D
tensor, nothing for an entry that a delta did not change), its canonical file and the checks that refuse a tampered
one; the one system-prompt sentence, word for word, and nothing else changed. Streams on the toy stores: arm L and arm
N teacher streams on the same round-1 student (scripted round 2), whose round-2 sessions differ only by edit_log.json
and the sentence; the E11 two-round stream's dry run through the CLI with --round1-from-final in both arms, the log
holding exactly round 1's changed entries; and the CLI's --api-key-file reaching the session launcher. No test touches
the real store or starts Claude Code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from e11_toy import PLANT_CONDITION, e11_store  # noqa: F401
from test_harness_e11 import LN_PLANT
from test_harness_fixtures import needs_sandbox
from test_harness_learn import LearnStore, learn_store  # noqa: F401
from test_harness_loop import SESSION_KWARGS, _round_one, _round_two, _sandbox, loop_store  # noqa: F401
from test_harness_loop_v2 import _e11_session

from weightcode.common.identity import count_changed, load_params
from weightcode.harness import loop, loop_stream, runner
from weightcode.harness.finalize import STATUS_DRY_RUN
from weightcode.harness.jsonio import file_sha256, read_json
from weightcode.harness.leaks import scan_hint_words
from weightcode.harness.loop import (
    EDIT_LOG_FILE,
    EDIT_LOG_SENTENCE,
    KEPT_RULE_LOOP_V2,
    LoopError,
    check_edit_log,
    edit_log_bytes,
    edit_log_counts,
    edit_log_document,
    edit_log_section,
    loop_kept_rule_v2,
    loop_kept_section,
)
from weightcode.harness.loop_stream import MODE_SELECT, MODE_TEACHER, LoopStreamError, run_loop_stream
from weightcode.harness.prompts import render_system
from weightcode.harness.scripted import ScriptedLauncher

PREREG_SENTENCE = ("edit_log.json lists every parameter entry that earlier rounds changed; those entries hold the "
                   "capabilities installed by earlier rounds.")  # PREREG T16 (main f55b092), copied by hand


def _params() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {"a.weight": rng.normal(size=(4, 6)).astype(np.float32),
            "a.bias": rng.normal(size=6).astype(np.float32),
            "conv.weight": rng.normal(size=(2, 3, 2, 2)).astype(np.float32),
            "b.weight": rng.normal(size=(3, 3)).astype(np.float32)}


def _child(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    child = {name: value.copy() for name, value in params.items()}
    child["a.weight"][1, 2] += 1.0
    child["a.weight"][3, 2] -= 0.5
    child["a.weight"][3, 5] += 2.0
    child["a.bias"][4] = 0.0 if child["a.bias"][4] != 0 else 1.0
    child["conv.weight"][1, 0, 1, 1] += 0.25
    tiny = np.float32(np.float64(child["b.weight"][0, 0]) + 1e-12)  # a delta that changes nothing in float32
    assert tiny == child["b.weight"][0, 0]
    child["b.weight"][0, 0] = tiny
    return child


def _sha(path: Path) -> str:
    return file_sha256(path)


# ---------------------------------------------------------------------------------------------------------------
# Units


def test_the_log_lists_every_changed_entry_with_the_rows_and_columns_of_2d_tensors() -> None:
    base = _params()
    one = _child(base)
    two = {name: value.copy() for name, value in one.items()}
    two["b.weight"][2, 1] += 3.0
    document = edit_log_document([(1, base, one), (2, one, two)])
    assert list(document) == ["rounds"] and [item["round"] for item in document["rounds"]] == [1, 2]
    first, second = document["rounds"]
    assert first["tensors"] == [
        {"tensor": "a.weight", "flat_indices": [1 * 6 + 2, 3 * 6 + 2, 3 * 6 + 5], "rows": [1, 3], "columns": [2, 5]},
        {"tensor": "a.bias", "flat_indices": [4]},
        {"tensor": "conv.weight", "flat_indices": [int(np.ravel_multi_index((1, 0, 1, 1), (2, 3, 2, 2)))]},
    ]  # b.weight's float32 no-op is not a change; the 1-D and 4-D tensors have no rows or columns
    assert second["tensors"] == [{"tensor": "b.weight", "flat_indices": [7], "rows": [2], "columns": [1]}]
    assert edit_log_counts(document) == {"1": count_changed(base, one), "2": count_changed(one, two)} == {"1": 5,
                                                                                                         "2": 1}
    data = edit_log_bytes(document)
    assert data.endswith(b"\n") and data.count(b"\n") == 1 and edit_log_bytes(json.loads(data)) == data
    shapes = {name: value.shape for name, value in two.items()}
    assert check_edit_log(data, 3, shapes) == document
    assert edit_log_document([(1, base, base)]) == {"rounds": [{"round": 1, "tensors": []}]}  # a no-op round
    with pytest.raises(LoopError, match="rounds 1 to r - 1"):
        edit_log_document([(2, base, one)])


@pytest.mark.parametrize("tamper, match", [
    (lambda d: d["rounds"][0]["tensors"][0].update(rows=[1]), "rows or columns"),
    (lambda d: d["rounds"][0]["tensors"][0].update(flat_indices=[20, 8]), "strictly increasing"),
    (lambda d: d["rounds"][0]["tensors"][1].update(flat_indices=[6]), "out of range"),
    (lambda d: d["rounds"][0]["tensors"][1].update(rows=[0]), "keys are not"),
    (lambda d: d["rounds"][0]["tensors"].append({"tensor": "absent", "flat_indices": [0]}), "not a tensor"),
    (lambda d: d["rounds"][0]["tensors"].append(dict(d["rounds"][0]["tensors"][1])), "listed twice"),
    (lambda d: d["rounds"].append({"round": 2, "tensors": []}), "lists rounds 1 to 1"),
    (lambda d: d.update(about="round 1 wrote the protected units"), "canonical form"),
])
def test_a_tampered_log_is_refused(tamper: Any, match: str) -> None:
    base = _params()
    document = json.loads(edit_log_bytes(edit_log_document([(1, base, _child(base))])))
    tamper(document)
    with pytest.raises(LoopError, match=match):
        check_edit_log(edit_log_bytes(document), 2, {name: value.shape for name, value in base.items()})
    with pytest.raises(LoopError, match="canonical form"):  # the same document, written another way
        check_edit_log(json.dumps(edit_log_document([(1, base, _child(base))]), indent=1).encode(), 2)


def test_arm_l_adds_exactly_the_prereg_sentence_and_nothing_else() -> None:
    assert EDIT_LOG_SENTENCE == PREREG_SENTENCE
    base = loop_kept_section(render_system("LEARN"), loop_kept_rule_v2(4000, {1: 4000}, {1: 2}))
    with_log = edit_log_section(base)
    assert with_log.count(PREREG_SENTENCE) == 1 and PREREG_SENTENCE not in base
    assert with_log.replace("\n\n" + PREREG_SENTENCE, "", 1) == base  # the paragraph is the only difference
    first_section = with_log[:with_log.index("\n\n## What is kept\n")]
    assert first_section.endswith("\n\n" + PREREG_SENTENCE)  # its own paragraph, at the end of the first section
    assert scan_hint_words({"system": with_log}) == {}
    with pytest.raises(LoopError, match="already states"):
        edit_log_section(with_log)


# ---------------------------------------------------------------------------------------------------------------
# Streams


def _trial(root: Path) -> dict[str, str]:
    return {path.name: _sha(path) for path in sorted((root / "trial").iterdir())}


def _assert_arms_differ_only_by_the_log(root_l: Path, root_n: Path) -> None:
    """Round-2 sessions of arm L and arm N on the same student: equal task prompts and trial files except
    edit_log.json, and system prompts that differ by the one sentence."""
    trial_l, trial_n = _trial(root_l), _trial(root_n)
    assert set(trial_l) - set(trial_n) == {EDIT_LOG_FILE} and set(trial_n) <= set(trial_l)
    assert {name: trial_l[name] for name in trial_n} == trial_n
    assert (root_l / "prompts" / "task.md").read_bytes() == (root_n / "prompts" / "task.md").read_bytes()
    system_l = (root_l / "prompts" / "system.md").read_text()
    system_n = (root_n / "prompts" / "system.md").read_text()
    assert system_l == edit_log_section(system_n)
    session_l, session_n = read_json(root_l / "session.json"), read_json(root_n / "session.json")
    assert session_n["loop"]["edit_log"] == {"enabled": False}
    assert session_l["loop"]["edit_log"]["enabled"] is True
    assert session_l["loop"]["edit_log"]["sha256"] == trial_l[EDIT_LOG_FILE] == session_l["trial_files_sha256"][
        EDIT_LOG_FILE]
    assert EDIT_LOG_FILE not in session_n["trial_files_sha256"]
    assert session_l["leak_scan"]["passed"] and session_n["leak_scan"]["passed"]
    for key in ("tools", "budget", "limits", "kept_rule"):
        assert session_l[key] == session_n[key], key
    for key in ("tool_timeout_ms", "connect_timeout_ms"):
        assert session_l["mcp"][key] == session_n["mcp"][key], key


@needs_sandbox
def test_arm_l_and_arm_n_streams_differ_only_by_the_log_in_round_two(loop_store: tuple[LearnStore, str, str],  # noqa: F811
                                                                     tmp_path: Path) -> None:
    _, first, second = loop_store
    root1 = _round_one(first)
    program1 = root1 / "final" / "fix.py"
    records = {}
    for arm in ("l", "n"):
        records[arm] = read_json(run_loop_stream(
            MODE_TEACHER, [first, second], program1, root1, tmp_path / "loops", _sandbox(),
            session_kwargs=SESSION_KWARGS, launcher_for=lambda index: ScriptedLauncher(_round_two),
            kept_rule=KEPT_RULE_LOOP_V2, edit_log=arm == "l", tag=f"t16_{arm}"))
    l_rec, n_rec = records["l"], records["n"]
    assert l_rec["work_dir"] != n_rec["work_dir"] and l_rec["tag"] == "t16_l"
    assert l_rec["rounds"][0]["child_sha256"] == n_rec["rounds"][0]["child_sha256"]  # the same W1
    assert n_rec["edit_log"] == {"enabled": False, "rounds": {"2": {"enabled": False}}}
    assert n_rec["billing"] == {"mode": "subscription"}
    root_l = Path(l_rec["rounds"][1]["sessions"][-1]["root"])
    root_n = Path(n_rec["rounds"][1]["sessions"][-1]["root"])
    _assert_arms_differ_only_by_the_log(root_l, root_n)
    # The log is round 1's changed entries, from the stream's own students.
    work = Path(l_rec["work_dir"])
    base, w1 = (load_params(work / f"student_{when}_round_1.npz") for when in ("before", "after"))
    data = (root_l / "trial" / EDIT_LOG_FILE).read_bytes()
    assert data == edit_log_bytes(edit_log_document([(1, base, w1)]))
    assert data == (work / "edit_log_round_2.json").read_bytes()
    stated = l_rec["edit_log"]["rounds"]["2"]
    assert l_rec["edit_log"]["enabled"] is True and stated == l_rec["rounds"][1]["edit_log"]
    assert stated["n_entries"] == {"1": count_changed(base, w1)} == {"1": l_rec["rounds"][0]["n_changed_vs_base"]}
    context = read_json(Path(l_rec["rounds"][1]["context"]["path"]))
    assert context["edit_log"]["sha256"] == file_sha256(work / "edit_log_round_2.json") == stated["sha256"]
    assert "edit_log" not in read_json(Path(n_rec["rounds"][1]["context"]["path"]))
    # Arm N without a tag keeps the stem a stream had before T16.
    kwargs = dict(SESSION_KWARGS)
    old = loop_stream._stem(MODE_TEACHER, [first, second], {"sha256": file_sha256(program1)}, {}, kwargs,
                            KEPT_RULE_LOOP_V2)
    assert loop_stream._stem(MODE_TEACHER, [first, second], {"sha256": file_sha256(program1)}, {}, kwargs,
                             KEPT_RULE_LOOP_V2, None, False, None) == old
    assert old not in {Path(rec["work_dir"]).name.split("_")[2] for rec in records.values()}
    with pytest.raises(LoopStreamError, match="teacher round"):
        run_loop_stream(MODE_SELECT, [first, second], program1, root1, tmp_path / "x", _sandbox(),
                        candidates={2: [program1]}, edit_log=True)
    with pytest.raises(LoopStreamError, match="stream tag"):
        run_loop_stream(MODE_TEACHER, [first, second], program1, root1, tmp_path / "x", _sandbox(), tag="T16 L")


@needs_sandbox
def test_the_e11_stream_dry_run_from_the_final_logs_exactly_round_ones_changed_entries(
        e11_store: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    ids = [e11_store.ids[(PLANT_CONDITION, 100)], e11_store.ids[("contrast_s5", 100)]]
    root1 = _e11_session(ids[0], {"ln.py": LN_PLANT}, "ln.py")
    final1 = read_json(root1 / "final" / "selection.json")["final"]
    common = ["--mode", MODE_TEACHER, "--instances", *ids, "--round1-session", str(root1),
              "--round1-program", str(root1 / "final" / "ln.py"), "--round1-from-final", "--kept-rule",
              KEPT_RULE_LOOP_V2, "--dry-run", "--threads", "1", "--out", str(tmp_path / "dry")]
    roots = {}
    for arm, extra in (("l", ["--edit-log", "--tag", "t16_dl"]), ("n", ["--tag", "t16_dn"])):
        assert loop_stream.main([*common, *extra]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["status"] == STATUS_DRY_RUN and printed["edit_log"]["enabled"] is (arm == "l")
        roots[arm] = Path(printed["root"])
    _assert_arms_differ_only_by_the_log(roots["l"], roots["n"])
    document = check_edit_log((roots["l"] / "trial" / EDIT_LOG_FILE).read_bytes(), 2)
    assert edit_log_counts(document) == {"1": int(final1["n_changed"])}
    final_child = load_params(root1 / "final" / "model.npz")
    base = load_params(Path(read_json(root1 / "session.json")["real_model_path"]))
    assert document == edit_log_document([(1, base, final_child)])
    session = read_json(roots["l"] / "session.json")
    assert session["loop"]["edit_log"]["n_entries"] == {"1": int(final1["n_changed"])}
    assert session["loop"]["edit_log"]["sentence"] == PREREG_SENTENCE


class _KeyProbe(Exception):
    pass


@needs_sandbox
def test_the_cli_passes_the_api_key_file_to_the_session_launcher(loop_store: tuple[LearnStore, str, str],  # noqa: F811
                                                                 tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    _, first, second = loop_store
    root1 = _round_one(first)
    key = tmp_path / "key"
    key.write_text("sk-not-a-real-key\n")
    key.chmod(0o600)
    seen: list[Any] = []

    class Probe:
        def __init__(self, claude_bin: str | None = None, *, api_key_file: Path | None = None,
                     oauth_token_file: Path | None = None) -> None:
            assert oauth_token_file is None  # a stream bills the key file or the login, never a token file
            seen.append(api_key_file)
            raise _KeyProbe

    monkeypatch.setattr(runner, "ClaudeLauncher", Probe)
    args = ["--mode", MODE_TEACHER, "--instances", first, second, "--round1-session", str(root1),
            "--round1-program", str(root1 / "final" / "fix.py"), "--threads", "1", "--out", str(tmp_path / "s")]
    with pytest.raises(_KeyProbe):
        loop_stream.main([*args, "--api-key-file", str(key), "--tag", "t16_key"])
    assert seen == [key.absolute()]
    (state,) = list((tmp_path / "s").glob("*/state.json"))
    assert len(read_json(state)["rounds"]) == 1  # round 1 ran; round 2 stopped at the launcher, before any session
    with pytest.raises(_KeyProbe):  # without the option the session bills to the login: no key file
        loop_stream.main([*args, "--tag", "t16_login"])
    assert seen == [key.absolute(), None]
    for bad in (["--dry-run"], ["--mode", MODE_SELECT, "--candidates", "2=none"]):
        with pytest.raises(SystemExit):
            loop_stream.main([*args, "--api-key-file", str(key), *bad])
    assert "sk-not-a-real-key" not in capsys_free_text(tmp_path)


def capsys_free_text(root: Path) -> str:
    """Every text file the stream wrote, joined (the key must appear in none)."""
    return "\n".join(path.read_text(errors="replace") for path in root.rglob("*.json"))
