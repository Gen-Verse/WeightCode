"""Tests for the shared identity, metrics, final-patch rule and patch-check oracle."""

from __future__ import annotations

import numpy as np
import pytest

from weightcode.common.identity import count_changed, load_params, params_sha256, save_params
from weightcode.evalkit.metrics import (
    BASE_CANDIDATE,
    Candidate,
    choose_final,
    color_following_rate,
    color_split_fixes,
    color_split_target_fixes,
    sealed_readout,
    summarize,
)
from weightcode.evalkit.patch_check import BudgetExceeded, PatchCheckOracle


def _params(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "fc1.weight": rng.normal(size=(4, 6)).astype(np.float32),
        "fc1.bias": rng.normal(size=(4,)).astype(np.float32),
        "fc2.weight": rng.normal(size=(3, 4)).astype(np.float32),
        "fc2.bias": rng.normal(size=(3,)).astype(np.float32),
    }


def _forward(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = x.reshape(len(x), -1).astype(np.float32) / 255.0
    h = np.maximum(h @ params["fc1.weight"].T + params["fc1.bias"], 0.0)
    return h @ params["fc2.weight"].T + params["fc2.bias"]


def test_hash_ignores_insertion_order_and_tracks_dtype_shape_values() -> None:
    params = _params()
    reordered = dict(reversed(list(params.items())))
    assert params_sha256(params) == params_sha256(reordered)
    changed = dict(params, **{"fc1.bias": params["fc1.bias"].copy()})
    changed["fc1.bias"][0] += 1.0
    assert params_sha256(changed) != params_sha256(params)
    as_f64 = dict(params, **{"fc1.bias": params["fc1.bias"].astype(np.float64)})
    assert params_sha256(as_f64) != params_sha256(params)
    reshaped = dict(params, **{"fc1.weight": params["fc1.weight"].reshape(6, 4)})
    assert params_sha256(reshaped) != params_sha256(params)


def test_save_reload_roundtrip(tmp_path) -> None:
    params = _params()
    digest = save_params(tmp_path / "m.npz", params)
    assert digest == params_sha256(load_params(tmp_path / "m.npz"))
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_count_changed() -> None:
    base = _params()
    child = {k: v.copy() for k, v in base.items()}
    child["fc2.weight"][0, :2] = 0.0
    assert count_changed(base, child) == 2


def test_summarize_counts_by_hand() -> None:
    y = np.array([0, 0, 0, 1, 1, 2])
    base = np.array([1, 1, 0, 1, 0, 2])
    child = np.array([0, 2, 1, 1, 0, 2])
    s = summarize(y, base, child, a=0, b=1)
    assert s.fixed_target == 1  # image 0: y=0, base 1, child 0
    assert s.broken == 1  # image 2: base right, child wrong
    assert s.fixed == 1
    assert s.score == 0  # fixed 1 - broken 1
    assert s.target_score == 0  # fixed_target 1 - broken 1
    assert s.n_base_errors == 3  # images 0, 1 and 4
    assert s.n_base_target_errors == 2  # images 0 and 1
    assert s.base_target_conf == pytest.approx(2 / 3)
    assert s.child_target_conf == pytest.approx(1 / 3)
    readout = sealed_readout(y, base, child, a=0, b=1, n_classes=3)
    assert readout["delta_acc"] == pytest.approx(0.0)
    assert readout["delta_target"] == pytest.approx(-1 / 3)
    assert readout["R"] == pytest.approx(0.0)
    assert readout["worst_class_delta"] == pytest.approx(0.0)  # class 0: 1/3 -> 1/3 (image 2 lost, image 0 won)
    fixed_both = np.array([0, 0, 0, 1, 0, 2])
    assert sealed_readout(y, base, fixed_both, a=0, b=1, n_classes=3)["R"] == pytest.approx(1.0)
    assert sealed_readout(y, base, child, a=1, b=2, n_classes=3)["R"] is None  # base makes no 1 -> 2 errors
    no_pair = sealed_readout(y, base, child, n_classes=3)
    assert no_pair["fixed_target"] is None and "R" not in no_pair
    with pytest.raises(TypeError):
        sealed_readout(y, base, child)


def test_score_counts_every_fix_not_only_the_target_pair() -> None:
    y = np.array([0, 0, 1, 2])
    base = np.array([1, 0, 2, 2])
    child = np.array([0, 0, 1, 0])  # fixes images 0 (target pair) and 2 (other), breaks image 3
    s = summarize(y, base, child)
    assert (s.fixed, s.broken, s.score) == (2, 1, 1)
    assert s.fixed_target is None and s.target_score is None
    paired = summarize(y, base, child, a=0, b=1)
    assert (paired.score, paired.fixed_target, paired.target_score) == (1, 1, 0)
    assert sealed_readout(y, base, child, n_classes=3)["delta_acc"] == pytest.approx(1 / 4)
    with pytest.raises(ValueError):
        summarize(y, base, child, a=0)


def test_color_following_rate() -> None:
    y = np.array([0, 1, 2, 3])
    color = np.array([0, 2, 1, 1])  # images 1-3 have a color that disagrees with the label
    pred = np.array([0, 2, 2, 1])  # follows color on images 1 and 3
    assert color_following_rate(pred, y, color) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        color_following_rate(pred, y, y)


def test_color_split_target_fixes() -> None:
    y = np.array([0, 0, 0, 0, 1])
    base = np.array([1, 1, 1, 0, 1])  # three base target errors (0 -> 1)
    child = np.array([0, 1, 0, 0, 1])  # fixes images 0 and 2
    color = np.array([1, 1, 5, 0, 1])  # images 0 and 1 carry the color of class 1
    split = color_split_target_fixes(y, base, child, color, a=0, b=1)
    assert split == {
        "n_base_target_errors_color_b": 2,
        "n_base_target_errors_color_other": 1,
        "fixed_target_color_b": 1,
        "fixed_target_color_other": 1,
    }


def test_color_split_fixes() -> None:
    y = np.array([0, 1, 2, 3, 4])
    base = np.array([5, 1, 7, 9, 4])  # errors on images 0, 2, 3
    color = np.array([5, 1, 7, 3, 4])  # images 0 and 2 follow their color; image 3 does not
    child = np.array([0, 1, 7, 3, 4])  # fixes image 0 (color-following) and image 3 (other)
    assert color_split_fixes(y, base, child, color) == {
        "n_base_errors_color_following": 2,
        "n_base_errors_other": 1,
        "fixed_color_following": 1,
        "fixed_other": 1,
    }


def test_choose_final_never_prefers_a_non_positive_patch() -> None:
    zero = Candidate("p0", score=0, n_changed=5, query_index=0)
    negative = Candidate("p1", score=-2, n_changed=1, query_index=1)
    assert choose_final([zero, negative]) == BASE_CANDIDATE
    best_small = Candidate("p2", score=3, n_changed=4, query_index=2)
    best_large = Candidate("p3", score=3, n_changed=9, query_index=1)
    assert choose_final([zero, best_large, best_small]).patch_id == "p2"
    early = Candidate("p4", score=3, n_changed=4, query_index=0)
    assert choose_final([best_small, early]).patch_id == "p4"


def test_oracle_budget_validation_and_aggregates_only() -> None:
    rng = np.random.default_rng(1)
    x = rng.integers(0, 256, size=(60, 1, 2, 3), dtype=np.uint8)
    base = _params()
    y = np.argmax(_forward(base, x), axis=1)
    y[:10] = (y[:10] + 1) % 3
    a, b = int(y[0]), int(np.argmax(_forward(base, x[:1]), axis=1)[0])
    oracle = PatchCheckOracle(x, y, _forward, base, target=(a, b), budget=2, batch_size=16)
    bad = dict(base, **{"fc1.bias": np.full(4, np.nan, dtype=np.float32)})
    with pytest.raises(ValueError):
        oracle.query(bad, "bad")
    assert oracle.queries_remaining == 2
    child = {k: v.copy() for k, v in base.items()}
    child["fc2.bias"][a] += 0.5
    first = oracle.query(child, "c0")
    assert first["queries_remaining"] == 1
    assert first["n_changed"] == 1
    for value in first.values():
        assert not isinstance(value, (list, np.ndarray)), "patch-check must not return per-image data"
    oracle.query(base, "noop")
    with pytest.raises(BudgetExceeded):
        oracle.query(child, "c1")
    assert [c.query_index for c in oracle.candidates()] == [0, 1]


def test_oracle_without_target_pair_scores_all_fixes() -> None:
    rng = np.random.default_rng(2)
    x = rng.integers(0, 256, size=(80, 1, 2, 3), dtype=np.uint8)
    base = _params(3)
    base_pred = np.argmax(_forward(base, x), axis=1)
    y = base_pred.copy()
    y[:20] = (y[:20] + 1) % 3
    oracle = PatchCheckOracle(x, y, _forward, base, target=None, budget=1, batch_size=32)
    child = {k: v.copy() for k, v in base.items()}
    child["fc2.bias"] += np.array([0.3, -0.2, 0.1], dtype=np.float32)
    entry = oracle.query(child, "c0")
    child_pred = np.argmax(_forward(child, x), axis=1)
    expected = summarize(y, base_pred, child_pred)
    assert entry["score"] == expected.score == expected.fixed - expected.broken
    assert entry["fixed_target"] is None and entry["target_score"] is None
