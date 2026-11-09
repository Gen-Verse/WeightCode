"""On-disk layout and I/O of split files, dataset manifests and normalization statistics.

Layout (all roots from ``weightcode.common.paths``):

* ``$WEIGHTCODE_PRIVATE/<dataset>/splits/<split>.npz``: keys ``x``, ``y`` int64, ``idx`` int64 (index into the official
  set the rows come from), plus ``color`` int64 for Colored-MNIST. ``x`` is uint8 [N, C, H, W] for image datasets,
  int64 [N] node ids for graph datasets and int64 [N, L] token rows for sequence datasets; a sequence dataset's ``y``
  is int64 [N, L] (a target per position, -1 where there is none) and ``idx`` its row key.
* ``$WEIGHTCODE_PRIVATE/<dataset>/dataset.json`` and ``forward.py`` (graph datasets add ``graph.npz``).
* ``$WEIGHTCODE_SEALED/<dataset>/test.npz``: the official test set. This module only writes it; nothing here reads it.
* A deployment condition of a dataset (for example a CIFAR-10-C corruption) keeps its splits, dataset.json and
  models under ``$WEIGHTCODE_PRIVATE/<dataset>/conditions/<condition>/`` and its sealed test under
  ``$WEIGHTCODE_SEALED/<dataset>/conditions/<condition>/test.npz``. Path functions take ``condition=None`` for the
  dataset's own data (the evalkit condition "deploy", a name refused here). A condition shares the dataset's
  forward.py, so ``forward_path`` takes no condition.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import params_sha256
from weightcode.common.paths import private_dir, sealed_dir

LOADABLE_SPLITS: tuple[str, ...] = (
    "train", "train_uniform", "V_select", "V_evidence", "V_check", "valid_unused", "old_select", "old_check", "old_replay",
    "ref_pool",
)
"""Splits that may be loaded from the private store. The official test set is not among them. The old_* splits are
rows of the trained classes held out of base training (ogbn-arxiv nodes up to 2017, ``weightcode.data.arxiv``; CIFAR-100
base-class images, ``weightcode.data.cifar100_fscil``). ``ref_pool`` holds the CIFAR-100 new-class images that only the
all-class reference of PREREG E9 trains on."""

REQUIRED_KEYS: tuple[str, ...] = ("x", "y", "idx")
RESERVED_CONDITION_NAMES: tuple[str, ...] = ("deploy",)
"""Names that never denote a stored condition: "deploy" is the dataset's own data, ``condition=None`` here."""
_NAME = re.compile(r"[a-z0-9][a-z0-9_]{0,63}")


def _relative_root(dataset: str, condition: str | None) -> Path:
    for value in (dataset, condition):
        if value is not None and not _NAME.fullmatch(value):
            raise ValueError(f"invalid dataset or condition name {value!r}")
    if condition in RESERVED_CONDITION_NAMES:
        raise ValueError(
            f"condition {condition!r} is the dataset's own deployment data; pass condition=None to the store"
        )
    return Path(dataset) if condition is None else Path(dataset) / "conditions" / condition


def dataset_dir(dataset: str, condition: str | None = None) -> Path:
    """Private directory of one dataset, or of one deployment condition of it."""
    return private_dir() / _relative_root(dataset, condition)


def split_path(dataset: str, split: str, condition: str | None = None) -> Path:
    """Path of a stored split file."""
    if split not in LOADABLE_SPLITS:
        raise ValueError(f"{split!r} is not a private split; the official test set lives only under $WEIGHTCODE_SEALED")
    return dataset_dir(dataset, condition) / "splits" / f"{split}.npz"


def sealed_test_path(dataset: str, condition: str | None = None) -> Path:
    """Path of the sealed official test set (written by prepare_data, never read by this package)."""
    return sealed_dir() / _relative_root(dataset, condition) / "test.npz"


def dataset_json_path(dataset: str, condition: str | None = None) -> Path:
    """Path of the private dataset manifest."""
    return dataset_dir(dataset, condition) / "dataset.json"


def forward_path(dataset: str) -> Path:
    """Path of the dataset's numpy reference forward, shared by all of its deployment conditions."""
    return dataset_dir(dataset) / "forward.py"


def list_conditions(dataset: str) -> list[str]:
    """Sorted names of the stored deployment conditions of ``dataset`` (those with a dataset.json)."""
    root = dataset_dir(dataset) / "conditions"
    if not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and _NAME.fullmatch(path.name)
        and path.name not in RESERVED_CONDITION_NAMES
        and (path / "dataset.json").is_file()
    )


def graph_path(dataset: str) -> Path:
    """Path of a graph dataset's ``graph.npz`` (read by its forward.py from the same directory)."""
    return dataset_dir(dataset) / "graph.npz"


def models_dir(dataset: str, condition: str | None = None) -> Path:
    """Directory holding one subdirectory per exported model plus ``index.json``."""
    return dataset_dir(dataset, condition) / "models"


def _check_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"split arrays lack keys {missing}")
    x, y, idx = arrays["x"], arrays["y"], arrays["idx"]
    is_images = x.dtype == np.uint8 and x.ndim == 4
    is_nodes = x.dtype == np.int64 and x.ndim == 1
    is_tokens = x.dtype == np.int64 and x.ndim == 2
    if not (is_images or is_nodes or is_tokens):
        raise TypeError(
            f"x must be uint8 [N, C, H, W] images, int64 [N] node ids or int64 [N, L] token rows, got {x.dtype} {x.shape}"
        )
    if is_tokens and (y.dtype != np.int64 or y.shape != x.shape):
        raise TypeError(f"token rows need int64 per-position targets y of shape {x.shape}, got {y.dtype} {y.shape}")
    for key in ("idx", *(k for k in arrays if k not in REQUIRED_KEYS), *(() if is_tokens else ("y",))):
        value = arrays[key]
        if value.dtype != np.int64 or value.shape != (x.shape[0],):
            raise TypeError(f"{key} must be int64 of shape ({x.shape[0]},), got {value.dtype} {value.shape}")
    if y.shape[0] != idx.shape[0]:
        raise ValueError("y and idx lengths differ")


def split_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Canonical hash of a split, ``params_sha256`` over all of its arrays."""
    return params_sha256(arrays)


def save_split(path: Path, arrays: Mapping[str, np.ndarray], *, private_mode: bool = False) -> str:
    """Write a split npz atomically, reload it, check the canonical hash and return it."""
    _check_arrays(arrays)
    target = Path(path)
    if target.suffix != ".npz":
        raise ValueError(f"split path must end in .npz: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = split_sha256(arrays)
    temporary = target.with_name(target.name + ".tmp.npz")
    np.savez(temporary, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
    if private_mode:
        os.chmod(temporary, 0o600)
    with np.load(temporary, allow_pickle=False) as archive:
        reloaded = split_sha256({key: archive[key] for key in archive.files})
    if reloaded != expected:
        temporary.unlink()
        raise RuntimeError(f"reload hash mismatch for {target}: {reloaded} != {expected}")
    temporary.replace(target)
    return expected


def load_split(dataset: str, split: str, condition: str | None = None) -> dict[str, np.ndarray]:
    """Load one private split. Raises for the official test set or an unknown split name."""
    path = split_path(dataset, split, condition)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    _check_arrays(arrays)
    return arrays


def channel_stats(x: np.ndarray) -> tuple[list[float], list[float]]:
    """Per-channel mean and population std (ddof = 0) of x / 255 over all images and pixels.

    Sums of x and x^2 come from exact integer histograms; only the final division and square root are floating
    point.
    """
    if x.dtype != np.uint8 or x.ndim != 4:
        raise TypeError(f"x must be uint8 [N, C, H, W], got {x.dtype} {x.shape}")
    n_values = x.shape[0] * x.shape[2] * x.shape[3]
    levels = np.arange(256, dtype=np.int64)
    means, stds = [], []
    for channel in range(x.shape[1]):
        histogram = np.bincount(x[:, channel].reshape(-1), minlength=256).astype(np.int64)
        s1 = int(histogram @ levels)
        s2 = int(histogram @ (levels * levels))
        mean = s1 / (255.0 * n_values)
        var = s2 / (255.0**2 * n_values) - mean * mean
        if var <= 0.0:
            raise ValueError(f"channel {channel} has zero variance")
        means.append(float(mean))
        stds.append(float(np.sqrt(var)))
    return means, stds


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically (temporary file, then rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text())


def read_dataset_json(dataset: str, condition: str | None = None) -> dict[str, Any]:
    """Read ``dataset.json`` and check that it describes ``dataset`` (and ``condition``, if given)."""
    manifest = read_json(dataset_json_path(dataset, condition))
    if manifest.get("dataset") != dataset:
        raise ValueError(f"dataset.json describes {manifest.get('dataset')!r}, not {dataset!r}")
    recorded = manifest.get("condition", {}).get("name") if isinstance(manifest.get("condition"), dict) else None
    if recorded != condition:
        raise ValueError(f"dataset.json describes condition {recorded!r}, not {condition!r}")
    return manifest
