"""End-to-end build, write, verify and the pre-registered Colored-MNIST rebuild, on synthetic raw arrays.

Nothing here downloads data or touches the real stores: WEIGHTCODE_PRIVATE and WEIGHTCODE_SEALED point to temporary dirs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weightcode.data import build as B
from weightcode.data import colored
from weightcode.data.registry import DATASETS, DIGIT_CLASSES, SPLIT_NAMES
from weightcode.data.sources import RawDataset
from weightcode.data.store import load_split, read_dataset_json, split_path, split_sha256


def _synthetic_mnist(seed: int = 0) -> RawDataset:
    rng = np.random.default_rng(seed)
    y_train = np.repeat(np.arange(10), 6000)
    rng.shuffle(y_train)
    return RawDataset(
        source="MNIST",
        x_train=rng.integers(0, 256, size=(60_000, 1, 28, 28), dtype=np.uint8),
        y_train=y_train.astype(np.int64),
        x_test=rng.integers(0, 256, size=(10_000, 1, 28, 28), dtype=np.uint8),
        y_test=np.tile(np.arange(10), 1000).astype(np.int64),
        class_names=tuple(f"{d} - name" for d in DIGIT_CLASSES),
        info={},
    )


@pytest.fixture()
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("private", "sealed"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(tmp_path / "private"))
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(tmp_path / "sealed"))
    return tmp_path


def test_colored_build_layout_and_color_rules() -> None:
    raw = _synthetic_mnist()
    built = B.build("colored_mnist", raw)
    sizes = dict(DATASETS["colored_mnist"].split_sizes)
    assert {s: built.splits[s]["y"].size for s in SPLIT_NAMES} == sizes
    assert set(built.splits) == {*SPLIT_NAMES, "train_uniform"}
    for split, arrays in built.splits.items():
        assert set(arrays) == {"x", "y", "idx", "color"} and arrays["x"].shape[1:] == (3, 28, 28)
        assert np.array_equal(arrays["y"], raw.y_train[arrays["idx"]])
    train, uniform = built.splits["train"], built.splits["train_uniform"]
    assert np.array_equal(train["idx"], uniform["idx"])
    assert 0.98 < np.mean(train["color"] == train["y"]) < 0.995
    assert np.mean(uniform["color"] == uniform["y"]) < 0.12
    for split in ("V_select", "V_evidence", "V_check"):
        assert np.mean(built.splits[split]["color"] == built.splits[split]["y"]) < 0.13
    assert np.array_equal(built.test["idx"], np.arange(10_000))
    assert built.manifest["colored"]["rho"] == colored.RHO_PRIMARY
    assert built.manifest["shortcut_check"]["status"] == "pending"
    again = B.build("colored_mnist", raw)
    assert all(split_sha256(again.splits[s]) == split_sha256(built.splits[s]) for s in built.splits)


def test_write_verify_and_single_rebuild(stores: Path) -> None:
    raw = _synthetic_mnist(1)
    built = B.build("colored_mnist", raw)
    manifest = B.write(built, forward_sha256="0" * 64, provenance={}, run_info={})
    assert (stores / "sealed" / "colored_mnist" / "test.npz").exists()
    assert not (stores / "private" / "colored_mnist" / "splits" / "test.npz").exists()
    report = B.verify(B.build("colored_mnist", raw))
    assert all(entry["match"] for entry in report.values())
    with pytest.raises(FileExistsError):
        B.write(built, forward_sha256="0" * 64, provenance={}, run_info={})

    old = load_split("colored_mnist", "train")
    result = B.rebuild_colored_train(raw, colored.RHO_FALLBACK, "attempt1_rho0.99")
    new = load_split("colored_mnist", "train")
    assert result["old_sha256"] == manifest["splits"]["train"]["sha256"] == split_sha256(old)
    assert result["new_sha256"] == split_sha256(new) != result["old_sha256"]
    assert np.array_equal(old["idx"], new["idx"]) and np.array_equal(old["y"], new["y"])
    assert not np.any((new["color"] != new["y"]) & (old["color"] == old["y"]))
    assert result["new_conflict_fraction"] < 0.003
    superseded = stores / "private" / "colored_mnist" / result["old_path"]
    assert superseded.exists() and split_path("colored_mnist", "train").exists()
    assert read_dataset_json("colored_mnist")["splits"]["train"]["sha256"] == result["old_sha256"]
    with pytest.raises(FileExistsError):
        B.rebuild_colored_train(raw, colored.RHO_FALLBACK, "attempt1_rho0.99")


def test_build_rejects_mismatched_source() -> None:
    raw = _synthetic_mnist()
    with pytest.raises(ValueError):
        B.build("fashion_mnist", raw)
