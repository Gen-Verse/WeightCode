"""Stratified split rule, Colored-MNIST coloring, normalization statistics and split storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weightcode.data import colored
from weightcode.data.registry import DATASETS, SPLIT_NAMES, SPLIT_SEED
from weightcode.data.splits import largest_remainder_quota, stratified_split
from weightcode.data.store import channel_stats, load_split, save_split, split_path, split_sha256

MNIST_CLASS_COUNTS = np.array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])


def test_largest_remainder_quota() -> None:
    assert largest_remainder_quota(np.full(10, 6000), 4000).tolist() == [400] * 10
    quota = largest_remainder_quota(MNIST_CLASS_COUNTS, 4000)
    assert quota.sum() == 4000
    exact = MNIST_CLASS_COUNTS * 4000 / MNIST_CLASS_COUNTS.sum()
    assert np.all(np.abs(quota - exact) < 1.0)
    assert largest_remainder_quota(np.array([1, 1, 1]), 2).tolist() == [1, 1, 0]


def test_stratified_split_is_a_seeded_stratified_partition() -> None:
    y = np.repeat(np.arange(10), MNIST_CLASS_COUNTS)
    np.random.default_rng(0).shuffle(y)
    sizes = dict(DATASETS["colored_mnist"].split_sizes)
    parts = stratified_split(y, sizes, SPLIT_SEED, 10)
    assert {name: part.size for name, part in parts.items()} == sizes
    joined = np.concatenate(list(parts.values()))
    assert np.array_equal(np.sort(joined), np.arange(y.size))
    for name, part in parts.items():
        assert np.all(np.diff(part) > 0)
        if name != "train":
            counts = np.bincount(y[part], minlength=10)
            assert counts.tolist() == largest_remainder_quota(MNIST_CLASS_COUNTS, sizes[name]).tolist()
    again = stratified_split(y, sizes, SPLIT_SEED, 10)
    assert all(np.array_equal(parts[name], again[name]) for name in parts)
    other = stratified_split(y, sizes, SPLIT_SEED + 1, 10)
    assert not np.array_equal(parts["V_select"], other["V_select"])


def test_balanced_datasets_split_exactly_per_class() -> None:
    for name in ("fashion_mnist", "cifar10"):
        config = DATASETS[name]
        y = np.tile(np.arange(10), config.n_official_train // 10)
        parts = stratified_split(y, dict(config.split_sizes), SPLIT_SEED, 10)
        for split in SPLIT_NAMES:
            assert np.bincount(y[parts[split]], minlength=10).tolist() == [config.split_sizes[split] // 10] * 10


def test_stratified_split_rejects_bad_sizes() -> None:
    y = np.repeat(np.arange(10), 10)
    with pytest.raises(ValueError):
        stratified_split(y, {"train": 50, "V_select": 10, "V_evidence": 10, "V_check": 10}, 0, 10)
    with pytest.raises(ValueError):
        stratified_split(y, {"train": 70, "V_select": 10, "V_check": 20}, 0, 10)


def test_palette_is_fixed_and_well_separated() -> None:
    palette = np.asarray(colored.PALETTE, dtype=float)
    assert palette.shape == (10, 3) and palette.max(axis=1).min() == 255
    distances = np.linalg.norm(palette[:, None] - palette[None], axis=-1)
    assert distances[~np.eye(10, dtype=bool)].min() >= 127


def test_colorize_matches_rounding_formula() -> None:
    gray = np.random.default_rng(0).integers(0, 256, size=(50, 28, 28), dtype=np.uint8)
    color = np.arange(50) % 10
    out = colored.colorize(gray, color)
    assert out.dtype == np.uint8 and out.shape == (50, 3, 28, 28)
    palette = np.asarray(colored.PALETTE, dtype=np.float64)
    expected = np.rint(gray[:, None].astype(np.float64) * palette[color][:, :, None, None] / 255.0)
    assert np.array_equal(out, expected.astype(np.uint8))
    assert np.array_equal(colored.colorize(gray[:, None], color), out)
    white = colored.colorize(gray[:3], np.full(3, 9))
    assert np.array_equal(white, np.repeat(gray[:3, None], 3, axis=1))


def test_biased_colors_follow_rho_and_couple_across_rho() -> None:
    y = np.random.default_rng(1).integers(0, 10, size=200_000)
    loose = colored.biased_colors(y, 0.99, colored.color_rng(SPLIT_SEED, "train"))
    strict = colored.biased_colors(y, 0.999, colored.color_rng(SPLIT_SEED, "train"))
    conflict_loose, conflict_strict = loose != y, strict != y
    assert abs(conflict_loose.mean() - 0.01) < 0.0015
    assert abs(conflict_strict.mean() - 0.001) < 0.0004
    assert not np.any(conflict_strict & ~conflict_loose)
    shifts = (loose[conflict_loose] - y[conflict_loose]) % 10
    assert shifts.min() >= 1 and np.bincount(shifts, minlength=10)[1:].min() > 0.07 * conflict_loose.sum()


def test_uniform_colors_are_independent_of_labels() -> None:
    y = np.random.default_rng(2).integers(0, 10, size=100_000)
    color = colored.uniform_colors(y.size, colored.color_rng(SPLIT_SEED, "V_select"))
    table = np.bincount(y * 10 + color, minlength=100).reshape(10, 10)
    expected = np.outer(table.sum(1), table.sum(0)) / table.sum()
    chi2 = float(((table - expected) ** 2 / expected).sum())
    assert chi2 < 150.0  # 81 degrees of freedom; the 1e-6 upper quantile is about 150
    assert colored.color_rng(SPLIT_SEED, "V_select").integers(0, 10, 5).tolist() != colored.color_rng(
        SPLIT_SEED, "V_check"
    ).integers(0, 10, 5).tolist()


def test_shortcut_decision_rule() -> None:
    assert colored.shortcut_decision([0.80] * 5, [0.97] * 5)["triggered"] is False
    assert colored.shortcut_decision([0.85] * 5, [0.97] * 5)["triggered"] is True
    decision = colored.shortcut_decision([0.5, 0.6], [0.95, 0.97])
    assert decision["gap_points"] == pytest.approx(41.0)


def test_channel_stats_are_exact() -> None:
    x = np.random.default_rng(3).integers(0, 256, size=(40, 3, 8, 8), dtype=np.uint8)
    mean, std = channel_stats(x)
    values = x.astype(np.float64) / 255.0
    np.testing.assert_allclose(mean, values.mean(axis=(0, 2, 3)), rtol=1e-12)
    np.testing.assert_allclose(std, values.std(axis=(0, 2, 3)), rtol=1e-10)


def test_split_storage_roundtrip_and_test_set_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(tmp_path))
    rng = np.random.default_rng(4)
    arrays = {
        "x": rng.integers(0, 256, size=(6, 3, 4, 4), dtype=np.uint8),
        "y": np.arange(6, dtype=np.int64),
        "idx": np.arange(10, 16, dtype=np.int64),
        "color": np.arange(6, dtype=np.int64) % 10,
    }
    digest = save_split(split_path("toy", "V_check"), arrays)
    loaded = load_split("toy", "V_check")
    assert digest == split_sha256(loaded) and all(np.array_equal(loaded[k], arrays[k]) for k in arrays)
    with pytest.raises(ValueError):
        load_split("toy", "test")
    with pytest.raises(TypeError):
        save_split(split_path("toy", "V_select"), {**arrays, "y": arrays["y"].astype(np.int32)})
