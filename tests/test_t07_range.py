"""PREREG T07 (scripts/t07/t07_conditions.py, scripts/t07/t07_range.py): the fixed set list and seed rule, the labels,
the bootstrap and the recovered share, on known inputs. The image generation itself runs in E10's pinned environment
and is checked there by regenerating E10's dev rows (``make``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts" / "t07"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("t07_range", HERE / "t07_range.py")
R = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)
C = R.C


def test_set_list_is_the_registered_one():
    assert len(C.SETS) == 26 and len(set(C.SETS)) == 26
    assert len(C.NEW) == 21 and len(C.STORED) == 4 and len(C.UNSEEN) == 13
    assert set(C.SETS) == {C.CLEAN} | set(C.NEW) | set(C.STORED)
    assert not set(C.NEW) & set(C.STORED)
    assert C.UNSEEN == ("shot_noise", "impulse_noise", "motion_blur", "zoom_blur", "frost", "fog", "brightness",
                        "elastic_transform", "pixelate", "jpeg_compression", "speckle_noise", "spatter", "saturate")
    for name in C.NOT_RUNNABLE:
        assert not any(s.startswith(name) for s in C.SETS)
    assert [C.group_of(s) for s in C.SETS].count("unseen") == 13
    assert C.group_of("gaussian_noise_s3") == "own_gaussian_noise" and C.group_of("snow_s5") == "e10_other"
    assert C.severity_of("contrast_s2") == 2 and C.severity_of(C.CLEAN) is None


def test_seed_rule():
    assert C.seed_of("gaussian_noise", 1) == 70101 and C.seed_of("contrast", 4) == 71204
    assert C.seed_of("saturate", 5) == 71905 and C.seed_of("shot_noise", 5) == 70205
    seeds = [C.seed_of(c, s) for _, c, s in C.new_sets()]
    assert len(set(seeds)) == len(seeds)
    assert not set(seeds) & set(C.E10_SELECT_SEEDS.values())
    with pytest.raises(C.ConditionsError):
        C.seed_of("gaussian_noise", 6)
    with pytest.raises(C.ConditionsError):
        C.seed_of("not_a_corruption", 1)


def test_frost_dir_is_hash_checked(tmp_path):
    for name in C.FROST_FILES:
        (tmp_path / name).write_bytes(b"not the frost image")
    with pytest.raises(C.ConditionsError, match="frost1.png"):
        C.check_frost_dir(tmp_path)
    data = b"abc"
    (tmp_path / "x").write_bytes(data)
    assert C.git_blob_sha1(tmp_path / "x") == "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"  # git hash-object of "abc"


def test_labels():
    assert R.label(1.0, [0.1, 2.0]) == "help"
    assert R.label(0.99, [0.1, 2.0]) == "neutral"
    assert R.label(3.0, [-0.1, 6.0]) == "neutral"
    assert R.label(-1.5, [-3.0, -0.2]) == "harm"
    assert R.label(-1.5, [-3.0, 0.2]) == "neutral"


def test_bootstrap_shared_index_and_point():
    index = R.boot_index(10)
    assert index.shape == (R.BOOT_RESAMPLES, 10)
    assert np.array_equal(index, R.boot_index(10))
    values = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    out = R.boot_contrast(values, index)
    assert out["point"] == pytest.approx(20.0)
    assert out["ci95"][0] <= 20.0 <= out["ci95"][1]
    assert R.boot_contrast(np.zeros(10), index)["ci95"] == [0.0, 0.0]


def test_recovered_share_and_readout_compare():
    assert R.recovered_share(5.0, 0.90, 0.80) == (pytest.approx(10.0), pytest.approx(0.5))
    headroom, share = R.recovered_share(1.0, 0.90, 0.89)
    assert headroom == pytest.approx(1.0) and share is None
    ours = {"delta_acc": 0.1, "fixed": 3, "broken": 1, "worst_class_delta": -0.01, "base_acc": 0.5, "child_acc": 0.6,
            "n": 20}
    assert R.same_readout(ours, dict(ours)) == {}
    assert set(R.same_readout(ours, {**ours, "fixed": 4})) == {"fixed"}


def test_main_gate_rate():
    gaussian = {"patch_block2": {"patch": {"frac_above_threshold": 0.9}}, "cls_block4": {"cls": {}}}
    assert R.main_gate_rate("gaussian_noise", gaussian) == 0.9
    assert R.main_gate_rate("contrast", {"gate": {"rate_patch_tokens": 0.2}}) == 0.2


def test_networks_and_specs_are_complete():
    assert R.NETWORKS == ("base", "llm_gaussian_noise", "llm_contrast", "ft_gaussian_noise", "ft_contrast")
    for arch in R.ARCHS.values():
        assert set(arch["instances"]) == set(C.OWN) == set(arch["finals"]) == set(arch["ft"])
        for spec in list(arch["finals"].values()) + list(arch["ft"].values()):
            assert all(len(spec[k]) == 64 for k in spec if k.endswith("sha256"))
