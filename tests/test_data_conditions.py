"""CIFAR-10-C style deployment conditions on synthetic images and a temporary store.

Checks the transcription of the source formulas, the per-split noise streams, row preservation, and the
build, write and verify cycle. Nothing here reads the real stores: WEIGHTCODE_PRIVATE and WEIGHTCODE_SEALED point to tmp dirs.
"""

from __future__ import annotations

import stat
from pathlib import Path

import numpy as np
import pytest

from weightcode.data import conditions as C
from weightcode.data import store
from weightcode.data.registry import CIFAR10_CLASSES, SPLIT_NAMES, SPLIT_SEED
from weightcode.data.store import split_sha256


def _images(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, 3, 32, 32), dtype=np.uint8)


def _pil_like(image_chw: np.ndarray) -> np.ndarray:
    """What ``np.array(PIL image)`` gives the source: a C-contiguous (H, W, C) uint8 array."""
    return np.ascontiguousarray(image_chw.transpose(1, 2, 0))


def _source_gaussian_noise(x_hwc: np.ndarray, c: float, rng: np.random.Generator) -> np.ndarray:
    """make_cifar_c.py gaussian_noise and np.uint8 storage, with the module-level RNG replaced by ``rng``."""
    x = np.array(x_hwc) / 255.0
    return np.uint8(np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1) * 255)


def _source_contrast(x_hwc: np.ndarray, c: float) -> np.ndarray:
    """make_cifar_c.py contrast and np.uint8 storage."""
    x = np.array(x_hwc) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.uint8(np.clip((x - means) * c + means, 0, 1) * 255)


def test_constants_match_the_source_lists() -> None:
    assert C.CONDITIONS["gaussian_noise_s3"].constant == 0.08
    assert C.CONDITIONS["contrast_s3"].constant == 0.4
    assert C.CONSTANTS["gaussian_noise"] == (0.04, 0.06, 0.08, 0.09, 0.10)
    assert C.CONSTANTS["contrast"] == (0.75, 0.5, 0.4, 0.3, 0.15)


def test_gaussian_noise_equals_per_image_source_transcription() -> None:
    cond = C.CONDITIONS["gaussian_noise_s3"]
    x = _images(23, 1)
    ours = C.corrupt(x, cond, cond.rng("V_select"), chunk=5)
    rng = cond.rng("V_select")
    expected = np.stack([_source_gaussian_noise(_pil_like(img), 0.08, rng).transpose(2, 0, 1) for img in x])
    assert ours.dtype == np.uint8 and ours.shape == x.shape
    assert np.array_equal(ours, expected)
    assert np.mean(ours != x) > 0.5


def test_gaussian_noise_does_not_depend_on_chunking() -> None:
    cond = C.CONDITIONS["gaussian_noise_s3"]
    x = _images(17, 2)
    outs = [C.corrupt(x, cond, cond.rng("train"), chunk=chunk) for chunk in (1, 4, 17, 1000)]
    assert all(np.array_equal(outs[0], other) for other in outs[1:])


def test_contrast_matches_source_and_is_deterministic() -> None:
    cond = C.CONDITIONS["contrast_s3"]
    x = _images(40, 3)
    ours = C.corrupt(x, cond, None, chunk=7)
    expected = np.stack([_source_contrast(_pil_like(img), 0.4).transpose(2, 0, 1) for img in x])
    assert np.array_equal(ours, expected)
    assert np.array_equal(ours, C.corrupt(x, cond, None))
    flat = np.full((1, 3, 32, 32), 77, dtype=np.uint8)
    flat_expected = _source_contrast(_pil_like(flat[0]), 0.4).transpose(2, 0, 1)[None]
    assert np.array_equal(C.corrupt(flat, cond, None), flat_expected)
    assert set(np.unique(flat_expected)) <= {76, 77}  # a flat image keeps its level up to truncation
    spread_in = x.reshape(40, 3, -1).astype(np.float64).std(axis=2)
    spread_out = ours.reshape(40, 3, -1).astype(np.float64).std(axis=2)
    assert np.allclose(spread_out / spread_in, 0.4, atol=0.02)


def test_generator_rules() -> None:
    noise = C.CONDITIONS["gaussian_noise_s3"]
    streams = noise.streams
    assert streams == {"train": 301, "V_select": 303, "V_evidence": 304, "V_check": 305, "test": 306}
    assert len(set(streams.values())) == len(streams)
    first = noise.rng("V_check").normal(size=4)
    assert np.array_equal(first, np.random.default_rng([SPLIT_SEED, 305]).normal(size=4))
    assert C.CONDITIONS["contrast_s3"].rng("train") is None
    with pytest.raises(ValueError):
        C.corrupt(_images(2, 0), noise, None)
    with pytest.raises(ValueError):
        C.corrupt(_images(2, 0), C.CONDITIONS["contrast_s3"], noise.rng("train"))
    with pytest.raises(ValueError):
        C.Condition("bad", "cifar10", "contrast", 3, 300)


def test_corrupt_split_keeps_rows() -> None:
    cond = C.CONDITIONS["gaussian_noise_s3"]
    arrays = {"x": _images(9, 4), "y": np.arange(9, dtype=np.int64) % 10, "idx": np.arange(100, 109, dtype=np.int64)}
    out = C.corrupt_split(arrays, cond, "V_evidence")
    assert np.array_equal(out["y"], arrays["y"]) and np.array_equal(out["idx"], arrays["idx"])
    assert out["x"].shape == arrays["x"].shape and out["x"].dtype == np.uint8
    with pytest.raises(ValueError):
        C.corrupt_split({**arrays, "color": arrays["y"]}, cond, "V_evidence")


SIZES = {"train": 60, "V_select": 20, "V_evidence": 10, "V_check": 20}


@pytest.fixture()
def clean_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    """A small clean cifar10 store (splits and dataset.json) and the raw test arrays that match it."""
    for name in ("private", "sealed"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(tmp_path / "private"))
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(tmp_path / "sealed"))
    splits_meta = {}
    for offset, (split, n) in enumerate(SIZES.items()):
        arrays = {
            "x": _images(n, 10 + offset),
            "y": (np.arange(n, dtype=np.int64) * 3) % 10,
            "idx": np.arange(1000 * offset, 1000 * offset + n, dtype=np.int64),
        }
        digest = store.save_split(store.split_path("cifar10", split), arrays)
        splits_meta[split] = {"path": f"splits/{split}.npz", "n": n, "sha256": digest}
    test = {"x": _images(30, 99), "y": np.arange(30, dtype=np.int64) % 10, "idx": np.arange(30, dtype=np.int64)}
    manifest = {
        "schema": "weightcode.dataset.v0",
        "dataset": "cifar10",
        "source": "CIFAR10",
        "class_names": list(CIFAR10_CLASSES),
        "image_shape": [3, 32, 32],
        "norm": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25], "computed_on": "train"},
        "split_seed": SPLIT_SEED,
        "splits": splits_meta,
        "test": {"location": "$WEIGHTCODE_SEALED/cifar10/test.npz", "n": 30, "sha256": split_sha256(test)},
        "forward_sha256": "f" * 64,
    }
    store.write_json(store.dataset_json_path("cifar10"), manifest)
    return test


def test_build_write_verify_cycle(clean_store: dict[str, np.ndarray], tmp_path: Path) -> None:
    built = C.build_from_store("gaussian_noise_s3", clean_store)
    manifest = C.write_condition(built, {"git": {"commit": "x", "dirty": True}})
    assert store.list_conditions("cifar10") == ["gaussian_noise_s3"]
    recorded = store.read_dataset_json("cifar10", "gaussian_noise_s3")
    assert recorded == manifest
    assert recorded["condition"]["rng"]["streams"]["test"] == 306
    assert recorded["forward_sha256"] == "f" * 64 and recorded["forward_path"] == "../../forward.py"
    clean = store.read_dataset_json("cifar10")
    for split in SPLIT_NAMES:
        entry = recorded["splits"][split]
        assert entry["clean_sha256"] == clean["splits"][split]["sha256"] != entry["sha256"]
        on_disk = store.load_split("cifar10", split, "gaussian_noise_s3")
        clean_arrays = store.load_split("cifar10", split)
        assert np.array_equal(on_disk["y"], clean_arrays["y"]) and np.array_equal(on_disk["idx"], clean_arrays["idx"])
    sealed = store.sealed_test_path("cifar10", "gaussian_noise_s3")
    assert sealed == tmp_path / "sealed" / "cifar10" / "conditions" / "gaussian_noise_s3" / "test.npz"
    assert stat.S_IMODE(sealed.stat().st_mode) == 0o600
    for directory in (sealed.parent, sealed.parent.parent, sealed.parent.parent.parent):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert not (tmp_path / "private" / "cifar10" / "conditions" / "gaussian_noise_s3" / "splits" / "test.npz").exists()

    report = C.verify_condition(C.build_from_store("gaussian_noise_s3", clean_store))
    assert all(entry["match"] for entry in report.values())
    with pytest.raises(FileExistsError):
        C.write_condition(built, {})

    tampered = store.load_split("cifar10", "V_check", "gaussian_noise_s3")
    tampered["x"][0, 0, 0, 0] ^= 1
    store.save_split(store.split_path("cifar10", "V_check", "gaussian_noise_s3"), tampered)
    report = C.verify_condition(C.build_from_store("gaussian_noise_s3", clean_store))
    assert report["V_check"]["match"] is False and report["V_select"]["match"] is True


def test_build_refuses_mismatched_clean_inputs(clean_store: dict[str, np.ndarray]) -> None:
    wrong_test = {**clean_store, "y": (clean_store["y"] + 1) % 10}
    with pytest.raises(RuntimeError, match="raw official test"):
        C.build_from_store("contrast_s3", wrong_test)
    arrays = store.load_split("cifar10", "V_select")
    arrays["x"][0, 0, 0, 0] ^= 1
    store.save_split(store.split_path("cifar10", "V_select"), arrays)
    with pytest.raises(RuntimeError, match="V_select"):
        C.build_from_store("contrast_s3", clean_store)


def test_store_condition_names(clean_store: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError, match="own deployment data"):
        store.split_path("cifar10", "V_check", "deploy")
    with pytest.raises(ValueError):
        store.dataset_dir("cifar10", "Bad-Name")
    with pytest.raises(ValueError):
        store.split_path("cifar10", "test", "contrast_s3")
    assert store.list_conditions("cifar10") == []
    assert store.forward_path("cifar10") == store.dataset_dir("cifar10") / "forward.py"
