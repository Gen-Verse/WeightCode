"""The stored seqops dataset and its conditions: build, write, verify, the exclusion check and the old part.

Nothing here reads the real stores: WEIGHTCODE_PRIVATE and WEIGHTCODE_SEALED point to tmp dirs.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pytest

from weightcode.data import seqops as S
from weightcode.data import seqops_build as B
from weightcode.data.store import load_split, read_dataset_json, save_split, sealed_test_path, split_path, split_sha256


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("private", "sealed"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(tmp_path / "private"))
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(tmp_path / "sealed"))
    return tmp_path


def _write_base() -> dict:
    return B.write_built(B.build_base(), {"forward_path": "forward.py", "forward_sha256": "0" * 64})


def test_base_build_write_verify_and_refuse_overwrite(store: Path) -> None:
    manifest = _write_base()
    assert manifest["split_sizes"] == {"V_select": 4000, "V_evidence": 2000, "V_check": 4000, "train": 4000}
    assert manifest["test"]["n"] == 4000 and manifest["held_out"]["n_keys"] == 14000
    assert all(value["match"] for value in B.verify_base().values())
    test_file = sealed_test_path(S.DATASET)
    assert stat.S_IMODE(os.stat(test_file).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(test_file.parent).st_mode) == 0o700
    with pytest.raises(FileExistsError):
        _write_base()
    tampered = load_split(S.DATASET, "V_check")
    tampered["y"] = tampered["y"].copy()
    tampered["y"][0, tampered["y"][0] >= 0] = (tampered["y"][0, tampered["y"][0] >= 0] + 1) % S.N_SYMBOLS
    save_split(split_path(S.DATASET, "V_check"), tampered)
    report = B.verify_base()
    assert not report["V_check"]["match"] and report["V_select"]["match"]


def test_checked_exclusion_matches_the_held_out_splits_and_refuses_a_moved_split(store: Path) -> None:
    manifest = _write_base()
    keys = B.checked_exclusion(manifest)
    assert keys.size == 14000 and np.all(np.diff(keys) > 0)
    train = load_split(S.DATASET, "train")
    assert not np.isin(train["idx"], keys).any()
    moved = {**manifest, "test": {**manifest["test"], "sha256": "f" * 64}}
    with pytest.raises(B.SeqOpsStoreError):
        B.checked_exclusion(moved)


def test_condition_parts_old_part_and_disjointness(store: Path) -> None:
    built = B.build_condition("new0_substitution")
    counts = {name: built.manifest["splits"][name]["part_counts"] for name in B.CONDITION_PRIVATE_SPLITS}
    assert counts == {
        "V_select": {"1": 2000, "2": 2000, "3": 2000, "4": 4000},
        "V_evidence": {"1": 2000},
        "V_check": {"1": 4000, "4": 4000},
    }
    assert built.manifest["test"]["part_counts"] == {"1": 2000, "2": 2000, "3": 2000, "4": 4000}
    base = S.build_base_splits()
    new0 = S.TOKEN_ID["OP_NEW0"]
    all_new_keys = []
    for name, arrays in [*built.splits.items(), ("test", built.test)]:
        old = arrays["part"] == S.PART_OLD
        np.testing.assert_array_equal(arrays["idx"][old], base[name]["idx"] if name != "V_evidence" else arrays["idx"][old])
        has_new = (arrays["opa"] == new0) | (arrays["opb"] == new0)
        np.testing.assert_array_equal(has_new, ~old)
        all_new_keys.append(arrays["idx"][~old])
    keys = np.concatenate(all_new_keys)
    assert np.unique(keys).size == keys.size
    assert not np.isin(keys, S.keys_of(base.values())).any()
    v_check = built.splits["V_check"]
    single = v_check["part"] == S.PART_NEW_SHORT
    assert np.all(v_check["opb"][single] == S.NONE) and v_check["n"][single].max() <= 8
    assert built.manifest["condition"]["rule_agreement"]["max_overall"] < 0.5


def test_condition_write_and_verify_need_the_base(store: Path) -> None:
    base_manifest = _write_base()
    built = B.build_condition("new0_substitution")
    manifest = B.write_built(built, B.condition_extra(base_manifest))
    assert manifest["forward_sha256"] == base_manifest["forward_sha256"]
    assert read_dataset_json(S.DATASET, "new0_substitution")["condition"]["operator"] == "OP_NEW0"
    assert all(value["match"] for value in B.verify_condition("new0_substitution").values())
    assert split_sha256(load_split(S.DATASET, "V_check", "new0_substitution")) == manifest["splits"]["V_check"]["sha256"]
    assert sealed_test_path(S.DATASET, "new0_substitution").is_file()


def test_parse_condition() -> None:
    assert B.parse_condition("new0_substitution") == (0, "substitution")
    for bad in ("new_substitution", "new0_rotation", "new00_substitution", "old0_substitution"):
        with pytest.raises(ValueError):
            B.parse_condition(bad)


def test_symbol_agreement() -> None:
    identity = S.symbol_agreement(range(S.N_SYMBOLS))
    assert identity["identity"] == 1.0 and identity["single"]["INC"] == 0.0
    comp = S.symbol_agreement(S.OLD_TABLES["COMP"])
    assert comp["single"]["COMP"] == 1.0 and comp["composed"]["INC+DEC"] == 0.0
    inc_inc = (np.arange(S.N_SYMBOLS) + 2) % S.N_SYMBOLS
    assert S.symbol_agreement(inc_inc)["composed"]["INC+INC"] == 1.0
    with pytest.raises(S.SeqOpsError):
        S.symbol_agreement([0] * S.N_SYMBOLS)
