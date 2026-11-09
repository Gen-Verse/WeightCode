"""Build, write and verify the stored splits of one dataset from its official torchvision source.

``build`` is a pure function of the raw arrays, the split seed and (for Colored-MNIST) rho, so running it twice
gives byte-identical splits; ``verify`` uses that to re-derive every split and compare canonical hashes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from weightcode.data import colored
from weightcode.data.registry import N_CLASSES, SPLIT_NAMES, SPLIT_SEED, DatasetConfig, get_config
from weightcode.data.sources import RawDataset
from weightcode.data.splits import stratified_split
from weightcode.data.store import (
    channel_stats,
    dataset_dir,
    dataset_json_path,
    load_split,
    read_dataset_json,
    save_split,
    sealed_test_path,
    split_path,
    split_sha256,
    write_json,
)

DATASET_SCHEMA = "weightcode.dataset.v0"
SPLIT_RULE = (
    "per class, non-train splits get largest-remainder quotas proportional to the official train class counts; "
    "each class's indices are permuted with numpy default_rng(split_seed) (one generator, classes in increasing "
    "order) and dealt to V_select, V_evidence, V_check, then train; indices sorted within each split"
)


@dataclass
class BuiltDataset:
    """All stored arrays of one dataset plus the manifest fields that do not depend on files on disk."""

    name: str
    splits: dict[str, dict[str, np.ndarray]]
    test: dict[str, np.ndarray]
    manifest: dict[str, Any]


def _check_raw(config: DatasetConfig, raw: RawDataset) -> None:
    if raw.source != config.source:
        raise ValueError(f"{config.name} needs source {config.source}, got {raw.source}")
    if config.source == "MNIST":
        expected = tuple(name.split(" ")[0] for name in raw.class_names)
    else:
        expected = raw.class_names
    if expected != config.class_names:
        raise ValueError(f"{config.name}: torchvision classes {raw.class_names} do not match {config.class_names}")
    if raw.x_train.shape[0] != config.n_official_train or raw.x_test.shape[0] != config.n_official_test:
        raise ValueError(f"{config.name}: unexpected official set sizes {raw.x_train.shape} / {raw.x_test.shape}")
    channels = 1 if config.source == "MNIST" else config.image_shape[0]
    if raw.x_train.shape[1:] != (channels, *config.image_shape[1:]):
        raise ValueError(f"{config.name}: unexpected raw image shape {raw.x_train.shape[1:]}")


def colored_split(x_gray: np.ndarray, y: np.ndarray, idx: np.ndarray, split: str, rho: float | None) -> dict[str, np.ndarray]:
    """Colored arrays of one split: biased colors for ``train`` (needs rho), uniform colors otherwise."""
    rng = colored.color_rng(SPLIT_SEED, split)
    if split == "train":
        if rho is None:
            raise ValueError("the biased train split needs rho")
        color = colored.biased_colors(y, rho, rng)
    elif split in colored.UNIFORM_SPLITS:
        if rho is not None:
            raise ValueError(f"split {split} uses the deployment distribution and takes no rho")
        color = colored.uniform_colors(y.size, rng)
    else:
        raise KeyError(f"unknown Colored-MNIST split {split!r}")
    return {"x": colored.colorize(x_gray, color), "y": y, "idx": idx, "color": color}


def build(name: str, raw: RawDataset, rho: float = colored.RHO_PRIMARY) -> BuiltDataset:
    """Split the official train set and assemble every stored array of dataset ``name``."""
    config = get_config(name)
    _check_raw(config, raw)
    indices = stratified_split(raw.y_train, config.split_sizes, SPLIT_SEED, N_CLASSES)
    splits: dict[str, dict[str, np.ndarray]] = {}
    is_colored = name == "colored_mnist"
    for split in SPLIT_NAMES:
        idx = indices[split]
        x, y = raw.x_train[idx], raw.y_train[idx]
        if is_colored:
            splits[split] = colored_split(x, y, idx, split, rho if split == "train" else None)
        else:
            splits[split] = {"x": np.ascontiguousarray(x), "y": y, "idx": idx}
    test_idx = np.arange(raw.x_test.shape[0], dtype=np.int64)
    if is_colored:
        train_idx = indices["train"]
        splits["train_uniform"] = colored_split(raw.x_train[train_idx], raw.y_train[train_idx], train_idx, "train_uniform", None)
        test = colored_split(raw.x_test, raw.y_test, test_idx, "test", None)
    else:
        test = {"x": raw.x_test, "y": raw.y_test, "idx": test_idx}
    for split, arrays in splits.items():
        if arrays["x"].shape[1:] != config.image_shape:
            raise AssertionError(f"{split}: stored shape {arrays['x'].shape[1:]} != {config.image_shape}")
    mean, std = channel_stats(splits["train"]["x"])
    manifest: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "dataset": name,
        "source": config.source,
        "class_names": list(config.class_names),
        "image_shape": list(config.image_shape),
        "dtype": "uint8",
        "layout": "NCHW",
        "norm": {
            "mean": mean,
            "std": std,
            "computed_on": "train",
            "definition": "per-channel mean and population std (ddof=0) of x/255 over all train images and pixels",
        },
        "split_seed": SPLIT_SEED,
        "split_rule": SPLIT_RULE,
        "split_sizes": {split: int(arrays["y"].size) for split, arrays in splits.items()},
        "splits": {
            split: {
                "path": f"splits/{split}.npz",
                "n": int(arrays["y"].size),
                "sha256": split_sha256(arrays),
                "class_counts": np.bincount(arrays["y"], minlength=N_CLASSES).tolist(),
            }
            for split, arrays in splits.items()
        },
        "test": {"location": f"$WEIGHTCODE_SEALED/{name}/test.npz", "n": int(test["y"].size), "sha256": split_sha256(test)},
    }
    if is_colored:
        train_color = splits["train"]["color"]
        manifest["colored"] = {
            "palette": [list(rgb) for rgb in colored.PALETTE],
            "palette_names": list(colored.PALETTE_NAMES),
            "image_rule": "channel k = round(gray * palette[color][k] / 255), exact integer arithmetic",
            "label": "clean digit; class c owns palette color c",
            "rho": rho,
            "train_color_rule": "label color if u >= 1 - rho else (label + shift) % 10, u ~ U[0,1), shift ~ U{1..9}",
            "deployment_color_rule": "uniform over 10 colors, independent of the label",
            "deployment_splits": list(colored.UNIFORM_SPLITS),
            "color_streams": {split: [SPLIT_SEED, stream] for split, stream in colored.COLOR_STREAMS.items()},
            "train_conflict_fraction": float(np.mean(train_color != splits["train"]["y"])),
            "train_uniform_note": "same images and labels as train, recolored with the deployment distribution",
        }
        manifest["shortcut_check"] = {
            "rule": (
                "decided on V_select only: if mean base V_select accuracy (seeds 0-4) is within "
                f"{colored.SHORTCUT_GAP_POINTS:g} points of mean ceiling V_select accuracy (ceiling minus base <= "
                f"{colored.SHORTCUT_GAP_POINTS:g} percentage points), rebuild the biased train split once with "
                f"rho = {colored.RHO_FALLBACK} and retrain the base models; nothing else changes (normalization "
                "constants, forward.py and ceiling models stay)"
            ),
            "attempts": [],
            "status": "pending",
        }
    return BuiltDataset(name=name, splits=splits, test=test, manifest=manifest)


def write(built: BuiltDataset, forward_sha256: str, provenance: dict[str, Any], run_info: dict[str, Any]) -> dict[str, Any]:
    """Write private splits, the sealed test set and dataset.json; refuses to overwrite an existing dataset."""
    if dataset_json_path(built.name).exists():
        raise FileExistsError(f"{dataset_json_path(built.name)} exists; use verify instead of rebuilding")
    for split, arrays in built.splits.items():
        digest = save_split(split_path(built.name, split), arrays)
        if digest != built.manifest["splits"][split]["sha256"]:
            raise RuntimeError(f"{split}: written hash differs from the built hash")
    test_path = sealed_test_path(built.name)
    test_path.parent.mkdir(mode=0o700, exist_ok=True)
    os.chmod(test_path.parent, 0o700)
    if save_split(test_path, built.test, private_mode=True) != built.manifest["test"]["sha256"]:
        raise RuntimeError("test: written hash differs from the built hash")
    manifest = dict(built.manifest)
    manifest["forward_path"] = "forward.py"
    manifest["forward_sha256"] = forward_sha256
    manifest["source_provenance"] = provenance
    manifest["created_by"] = run_info
    write_json(dataset_json_path(built.name), manifest)
    return manifest


def verify(built: BuiltDataset) -> dict[str, Any]:
    """Compare a fresh build with dataset.json and the split files on disk (the sealed file is not read).

    For Colored-MNIST, a train split rebuilt under the shortcut rule is compared at its recorded rho.
    """
    manifest = read_dataset_json(built.name)
    report: dict[str, Any] = {}
    for split, arrays in built.splits.items():
        fresh = split_sha256(arrays)
        recorded = manifest["splits"][split]["sha256"]
        on_disk = split_sha256(load_split(built.name, split))
        report[split] = {"fresh": fresh, "recorded": recorded, "on_disk": on_disk, "match": fresh == recorded == on_disk}
    fresh_test = split_sha256(built.test)
    report["test"] = {"fresh": fresh_test, "recorded": manifest["test"]["sha256"], "match": fresh_test == manifest["test"]["sha256"]}
    norm_match = built.manifest["norm"] == {key: manifest["norm"][key] for key in built.manifest["norm"]}
    report["norm"] = {"match": norm_match}
    return report


def rebuild_colored_train(raw: RawDataset, rho: float, superseded_tag: str) -> dict[str, Any]:
    """Recolor the biased Colored-MNIST train split at ``rho`` (pre-registered single rebuild).

    Indices, labels and gray images are unchanged and re-checked against the stored split; only the colors change.
    The previous file moves to ``superseded/<superseded_tag>/splits/train.npz``. Returns the old and new hashes and
    the new conflict fraction. dataset.json is updated by the caller.
    """
    name = "colored_mnist"
    config = get_config(name)
    _check_raw(config, raw)
    idx = stratified_split(raw.y_train, config.split_sizes, SPLIT_SEED, N_CLASSES)["train"]
    old = load_split(name, "train")
    if not (np.array_equal(old["idx"], idx) and np.array_equal(old["y"], raw.y_train[idx])):
        raise RuntimeError("stored train split does not match the split rule; refusing to rebuild")
    new = colored_split(raw.x_train[idx], raw.y_train[idx], idx, "train", rho)
    old_conflict = old["color"] != old["y"]
    new_conflict = new["color"] != new["y"]
    if np.any(new_conflict & ~old_conflict):
        raise AssertionError("coupled color draws violated: a new conflict is not an old conflict")
    target = dataset_dir(name) / "superseded" / superseded_tag / "splits" / "train.npz"
    if target.exists():
        raise FileExistsError(f"{target} exists")
    target.parent.mkdir(parents=True)
    old_sha = split_sha256(old)
    split_path(name, "train").rename(target)
    new_sha = save_split(split_path(name, "train"), new)
    return {
        "old_sha256": old_sha,
        "old_path": str(target.relative_to(dataset_dir(name))),
        "new_sha256": new_sha,
        "new_conflict_fraction": float(np.mean(new_conflict)),
        "new_class_counts": np.bincount(new["y"], minlength=N_CLASSES).tolist(),
    }
