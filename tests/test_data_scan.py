"""SCAN for PREREG E13: the interpretation function, the grammar, token rows, the seed-100 splits and the store.

Nothing here reads the real stores or the SCAN repository: the raw files are written from the grammar into a tmp dir
(``fake_raw``), with ``scan.RAW_SHA256`` patched to their hashes, and WEIGHTCODE_PRIVATE and WEIGHTCODE_SEALED point to tmp dirs.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pytest
import scan_toy

from weightcode.data import scan as S
from weightcode.data import scan_build as B
from weightcode.data.store import load_split, read_dataset_json, save_split, sealed_test_path, split_path

PUBLISHED = {
    "jump": ["I_JUMP"],
    "turn left": ["I_TURN_LEFT"],
    "walk left": ["I_TURN_LEFT", "I_WALK"],
    "run opposite right": ["I_TURN_RIGHT", "I_TURN_RIGHT", "I_RUN"],
    "turn opposite left": ["I_TURN_LEFT", "I_TURN_LEFT"],
    "look around right": ["I_TURN_RIGHT", "I_LOOK"] * 4,
    "turn around left": ["I_TURN_LEFT"] * 4,
    "walk twice and jump": ["I_WALK", "I_WALK", "I_JUMP"],
    "jump after walk left thrice": ["I_TURN_LEFT", "I_WALK"] * 3 + ["I_JUMP"],
    "jump around right thrice after walk": ["I_WALK"] + ["I_TURN_RIGHT", "I_JUMP"] * 12,
}


def test_interpretation_function_matches_the_published_examples() -> None:
    for command, actions in PUBLISHED.items():
        assert S.interpret(command) == actions
    for bad in ("turn", "walk walk", "twice", "walk and", "walk left left", "walk and run after look", "fly"):
        with pytest.raises(S.ScanError):
            S.interpret(bad)


def test_grammar_has_the_published_size_and_lengths() -> None:
    commands = S.grammar_commands()
    assert len(commands) == 20910 == len(set(commands))
    assert max(len(c.split()) for c in commands) == S.MAX_CMD
    assert max(len(S.interpret(c)) for c in commands) == S.OUT_SLOTS
    assert sum(S.has_primitive(c, "addprim_jump") for c in commands) == 7706 + 1  # the test file and "jump" alone
    assert S.has_primitive("walk around right", "around_right") and not S.has_primitive("walk right around left",
                                                                                         "around_right")


def test_rows_encode_decode_and_score() -> None:
    commands = ["jump", "walk around left thrice and look opposite right thrice"]
    x, y = S.encode(commands)
    assert x.shape == y.shape == (2, S.SEQ_LEN) and S.SEQ_LEN == 59
    assert S.decode(x) == commands
    assert S.actions_of(y) == [S.interpret(c) for c in commands]
    assert int((y[0] != S.IGNORE).sum()) == S.OUT_SLOTS and int((y[0] == S.END).sum()) == S.OUT_SLOTS - 1
    assert x[0, 0] == S.BOS and x[0, 2] == S.SEP and np.all(x[0, 3:3 + S.OUT_SLOTS] == S.OUT)
    pred = np.where(y == S.IGNORE, 0, y)
    assert S.row_correct(y, pred).all() and S.token_accuracy(y, pred) == 1.0
    pred[1, np.flatnonzero(y[1] != S.IGNORE)[-1]] = 0  # the last OUT slot (END) predicted as an action
    assert S.row_correct(y, pred).tolist() == [True, False]
    with pytest.raises(S.ScanError):
        S.encode(["jump"], [["I_WALK"]])


@pytest.fixture
def fake_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A SCAN-shaped checkout written from the grammar (``scan_toy.fake_raw``)."""
    return scan_toy.fake_raw(tmp_path, monkeypatch)


def test_splits_have_the_prereg_sizes_and_are_disjoint(fake_raw: Path) -> None:
    built = S.build_condition_splits(fake_raw, "addprim_jump")
    S.check_disjoint(built)
    sizes = {name: len(arrays["y"]) for name, arrays in built.private.items()}
    assert sizes == {"train": 13203 - 3000 + 1467, "old_check": 1000, "old_select": 1000, "V_check": 2569,
                     "V_select": 2569}
    assert len(built.sealed["y"]) == 2568 + 1000
    train = built.private["train"]
    assert int((train["part"] == S.PART_PRIM).sum()) == 1467
    assert not np.isin(train["part"], [S.PART_NEW]).any()
    for name in ("V_check", "V_select"):
        assert all(S.has_primitive(c, "addprim_jump") for c in S.decode(built.private[name]["x"]))
    for name in ("old_check", "old_select"):
        assert not any(S.has_primitive(c, "addprim_jump") for c in S.decode(built.private[name]["x"]))
        assert np.all(np.diff(built.private[name]["idx"]) > 0)
    again = S.build_condition_splits(fake_raw, "addprim_jump")
    assert all(np.array_equal(again.private[k]["x"], v["x"]) for k, v in built.private.items())
    ar = S.build_condition_splits(fake_raw, "around_right")
    n_new = ar.record["n_new_commands"]
    assert len(ar.private["V_check"]["y"]) + len(ar.private["V_select"]["y"]) + ar.record["sealed_sizes"]["new"] == n_new
    assert ar.record["n_primitive_rows_in_train"] == 0 and len(ar.private["V_select"]["y"]) == -(-n_new // 3)


def test_raw_copy_store_write_verify_and_refusals(fake_raw: Path) -> None:
    provenance = B.install_raw(fake_raw)
    assert set(provenance["files"]) == set(S.RAW_SHA256)
    assert stat.S_IMODE(os.stat(B.raw_dir() / "tasks.txt").st_mode) == 0o444
    with pytest.raises(FileExistsError):
        B.install_raw(fake_raw)
    B.write_base({"note": "test"})
    manifest = B.write_condition(B.build_condition("addprim_jump"), {"note": "test"})
    assert manifest["split_sizes"]["V_select"] == 2569 and manifest["test"]["n"] == 3568
    assert read_dataset_json(S.DATASET, "addprim_jump")["forward_sha256"] == read_dataset_json(S.DATASET)[
        "forward_sha256"]
    sealed = sealed_test_path(S.DATASET, "addprim_jump")
    assert stat.S_IMODE(os.stat(sealed).st_mode) == 0o600 and stat.S_IMODE(os.stat(sealed.parent).st_mode) == 0o700
    assert all(value["match"] for value in B.verify_condition("addprim_jump").values())
    with pytest.raises(FileExistsError):
        B.write_condition(B.build_condition("addprim_jump"), {})
    tampered = load_split(S.DATASET, "old_check", "addprim_jump")
    tampered["idx"] = tampered["idx"][::-1].copy()
    save_split(split_path(S.DATASET, "old_check", "addprim_jump"), tampered)
    report = B.verify_condition("addprim_jump")
    assert not report["old_check"]["match"] and report["V_select"]["match"]
    with pytest.raises(B.ScanStoreError):
        B.checked_split("addprim_jump", "old_check")
