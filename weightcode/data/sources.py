"""Download official datasets with torchvision and read them as numpy arrays, with file provenance.

Raw files live under ``$WEIGHTCODE_PRIVATE/raw``. torchvision checks the md5 of every archive it downloads; this module
additionally records the sha256 and size of every file of a source in ``$WEIGHTCODE_PRIVATE/raw/provenance.json``.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.paths import private_dir
from weightcode.data.store import read_json, write_json

PROVENANCE_SCHEMA = "weightcode.raw_provenance.v0"


@dataclass(frozen=True)
class RawDataset:
    """Official train and test arrays of one torchvision source, images uint8 [N, C, H, W]."""

    source: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: tuple[str, ...]
    info: dict[str, Any]


def raw_dir() -> Path:
    """Directory of the torchvision downloads."""
    path = private_dir() / "raw"
    path.mkdir(exist_ok=True)
    return path


def provenance_path() -> Path:
    """Path of the raw-file provenance record."""
    return raw_dir() / "provenance.json"


def _source_files(source: str, root: Path) -> list[Path]:
    if source in ("FashionMNIST", "MNIST"):
        folder = root / source / "raw"
        files = sorted(p for p in folder.iterdir() if p.is_file())
    elif source == "CIFAR10":
        files = [root / "cifar-10-python.tar.gz"]
        files += sorted(p for p in (root / "cifar-10-batches-py").iterdir() if p.is_file())
    else:
        raise KeyError(f"unknown source {source!r}")
    if not files:
        raise FileNotFoundError(f"no raw files found for {source} under {root}")
    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _torchvision_class(source: str) -> Any:
    import torchvision.datasets as tvd

    classes = {"FashionMNIST": tvd.FashionMNIST, "MNIST": tvd.MNIST, "CIFAR10": tvd.CIFAR10}
    if source not in classes:
        raise KeyError(f"unknown source {source!r}")
    return classes[source]


def _source_info(source: str) -> dict[str, Any]:
    import torchvision

    cls = _torchvision_class(source)
    info: dict[str, Any] = {"torchvision_class": f"torchvision.datasets.{cls.__name__}", "torchvision": torchvision.__version__}
    if source in ("FashionMNIST", "MNIST"):
        info["mirrors"] = list(cls.mirrors)
        info["resources_md5"] = [list(item) for item in cls.resources]
    else:
        info["url"] = cls.url
        info["archive_md5"] = cls.tgz_md5
        info["train_list_md5"] = [list(item) for item in cls.train_list]
        info["test_list_md5"] = [list(item) for item in cls.test_list]
    return info


def download_and_load(source: str) -> RawDataset:
    """Download (if absent) and read one torchvision source; images are returned as uint8 NCHW arrays."""
    root = raw_dir()
    cls = _torchvision_class(source)
    train = cls(root=str(root), train=True, download=True)
    test = cls(root=str(root), train=False, download=True)
    if source in ("FashionMNIST", "MNIST"):
        x_train = train.data.numpy()[:, None]
        x_test = test.data.numpy()[:, None]
        y_train = train.targets.numpy()
        y_test = test.targets.numpy()
    else:
        x_train = np.asarray(train.data).transpose(0, 3, 1, 2)
        x_test = np.asarray(test.data).transpose(0, 3, 1, 2)
        y_train = np.asarray(train.targets)
        y_test = np.asarray(test.targets)
    x_train = np.ascontiguousarray(x_train, dtype=np.uint8)
    x_test = np.ascontiguousarray(x_test, dtype=np.uint8)
    for name, array in (("x_train", x_train), ("x_test", x_test)):
        if array.dtype != np.uint8 or array.ndim != 4:
            raise TypeError(f"{source} {name}: expected uint8 NCHW, got {array.dtype} {array.shape}")
    return RawDataset(
        source=source,
        x_train=x_train,
        y_train=np.asarray(y_train, dtype=np.int64),
        x_test=x_test,
        y_test=np.asarray(y_test, dtype=np.int64),
        class_names=tuple(str(name) for name in train.classes),
        info=_source_info(source),
    )


def record_provenance(source: str, fetch_note: str | None = None) -> dict[str, Any]:
    """Hash every raw file of ``source``, make the files read-only and merge the record into provenance.json.

    ``fetch_note`` records how the archives were obtained when it was not torchvision's own downloader. If the
    source is already recorded, the files must hash to the recorded values; a mismatch raises.
    """
    root = raw_dir()
    files = {}
    for path in _source_files(source, root):
        files[str(path.relative_to(root))] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        mode = path.stat().st_mode
        os.chmod(path, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    entry = {**_source_info(source), "files": files, "fetch": fetch_note or "torchvision download (md5 checked by torchvision)"}
    record_path = provenance_path()
    record = read_json(record_path) if record_path.exists() else {"schema": PROVENANCE_SCHEMA, "sources": {}}
    if record.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError(f"unexpected provenance schema in {record_path}")
    previous = record["sources"].get(source)
    if previous is not None and previous["files"] != files:
        raise RuntimeError(f"raw files of {source} changed since they were first recorded in {record_path}")
    record["sources"][source] = entry
    write_json(record_path, record)
    return entry
