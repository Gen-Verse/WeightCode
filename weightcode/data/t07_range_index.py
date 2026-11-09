"""PREREG "T07 sealed read (list v5)": the range index, one dataset.json per (student dataset, T07 set).

The T07 sets (PREREG "T07 and M05 phase 1") are the 4,000 clean V_select rows with one corruption applied. A final or
fine-tuning pick of record is read on every set other than its own condition by the ``vision_range`` readout
(``weightcode.evalkit.sealed_range``). That readout resolves a set's rows from this index, never from the deployment
conditions: a deployment condition must hold V_evidence, V_check and V_select (``targets.check_condition_store``),
and T07 made V_select only. Layout (under $WEIGHTCODE_PRIVATE)::

    <dataset>/t07_range/<set>/dataset.json

Each file records, for its dataset and set:

- ``dev``: the set's x on the clean V_select rows: a file under $WEIGHTCODE_PRIVATE, its array, file sha256 and x sha256
  (sha256 of the C-order bytes), and the content hash of the dataset's clean V_select whose y and idx go with it. The
  21 sets T07 made point at ``t07_conditions/v1/<set>.npz`` (its t07_conditions.json must hash to
  ``CONDITIONS_SHA256``); the four stored severity-5 sets point at E10's V_select of that condition
  (``cifar10_r20/conditions/<set>/splits/V_select.npz``), whose x must hash as E10's dev_rows.json records, as T07
  read them.
- ``test``: the official CIFAR-10-C file of the corruption and the rows of the severity, in the ``cifar10_npy`` form of
  ``weightcode.evalkit.sealed_finals`` (rows "A .. B (severity S)", A = (S - 1) n), the file sha256 taken from the install
  record ``$WEIGHTCODE_SEALED/cifar10c/install.json`` (its ``files`` entries) exactly as ``weightcode.data.e10.build_condition``
  takes it, and the clean test's content hash.

The install record is the only file under $WEIGHTCODE_SEALED this module opens, and only in ``build_index``: nothing else
under the sealed root is opened, listed or statted here. Where the dataset already has a deployment condition of the
same name, or E10 has one, its test record must equal the index's. Nothing is overwritten.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.paths import private_dir
from weightcode.data.store import dataset_dir, dataset_json_path, load_split, read_dataset_json, split_sha256, write_json
from weightcode.evalkit.targets import file_sha256

INDEX_SCHEMA = "weightcode.t07_range_index.v1"
ROOT = "t07_range"
DATASETS = ("cifar10_vl7", "cifar10_r18")
CONDITIONS_DIR = "t07_conditions/v1"
CONDITIONS_JSON = "t07_conditions.json"
CONDITIONS_SHA256 = "ab232e6312051ce628768ba6926be2a9c7eebd4e95ce8b4c1a6140fab53caa12"
STORED_DATASET = "cifar10_r20"
STORED = ("gaussian_noise_s5", "defocus_blur_s5", "snow_s5", "contrast_s5")
DEV_ROWS_REL = "e10_staging/dev_rows_20261104/dev_rows.json"
DEV_ROWS_SHA256 = "b4dd5b083d47ca7cd591d55908d58306526eb1de34be2e46b5b8fba70312ad7d"
CLEAN_SPLIT_SHA256 = "75b0a7c4eea96a3ea9a27f4bac08f0de529a2d313c347c569bc8e9debde756cb"
INSTALL_REL = "cifar10c/install.json"
OFFICIAL_DIR = "CIFAR-10-C"
TEST_N = 10000
CIFAR10C_FORMAT_TEXT = "official CIFAR-10-C .npy, NHWC uint8, the official test order"
SET_RE = re.compile(r"^([a-z_]+)_s([1-5])$")
TEST_KEYS = ("location", "rows", "format", "n", "sha256", "clean_sha256")


class RangeIndexError(RuntimeError):
    """An input does not match its record, a file is missing from the install record, or a file exists."""


def x_sha256(x: np.ndarray) -> str:
    """sha256 of the C-order bytes of an image array (E10's dev_rows.json and T07's x_sha256)."""
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def parse_set(name: str) -> tuple[str, int]:
    """(corruption, severity) of a set name ``<corruption>_s<severity>``."""
    match = SET_RE.fullmatch(name)
    if match is None:
        raise RangeIndexError(f"{name!r} is not a set name <corruption>_s<1..5>")
    return match.group(1), int(match.group(2))


def index_path(dataset: str, set_name: str) -> Path:
    """Path of the range index file of one (dataset, set)."""
    parse_set(set_name)
    return dataset_dir(dataset) / ROOT / set_name / "dataset.json"


def test_rows(severity: int, n: int = TEST_N) -> str:
    """The rows of one severity block of an official CIFAR-10-C file, as the dataset.json files write them."""
    start = (int(severity) - 1) * int(n)
    return f"{start} .. {start + int(n) - 1} (severity {int(severity)})"


def test_block(corruption: str, severity: int, install: Mapping[str, Any], clean_test_sha256: str,
               n: int = TEST_N) -> dict[str, Any]:
    """The ``test`` record of one set: the official file at the install record's sha256, the severity's rows."""
    official = f"{OFFICIAL_DIR}/{corruption}.npy"
    files = install.get("files")
    entry = files.get(official) if isinstance(files, Mapping) else None
    sha = entry.get("sha256") if isinstance(entry, Mapping) else None
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise RangeIndexError(f"{official} has no sha256 in the install record; nothing written")
    return {"location": f"$WEIGHTCODE_SEALED/cifar10c/{official}", "rows": test_rows(severity, n),
            "format": CIFAR10C_FORMAT_TEXT, "n": int(n), "sha256": sha, "clean_sha256": clean_test_sha256}


def same_test(ours: Mapping[str, Any], theirs: Mapping[str, Any], what: str) -> None:
    """Refuse unless two test records agree on every key of ``TEST_KEYS``."""
    differ = [key for key in TEST_KEYS if ours.get(key) != theirs.get(key)]
    if differ:
        raise RangeIndexError(f"{what}: test record differs from the index's on {differ}")


def _load_x(path: Path, array: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if array not in archive.files:
            raise RangeIndexError(f"{path}: no array {array!r}")
        return np.ascontiguousarray(archive[array])


def _relative(path: Path) -> str:
    return path.resolve().relative_to(private_dir().resolve()).as_posix()


def dev_new(set_name: str, record: Mapping[str, Any], folder: Path) -> dict[str, Any]:
    """The dev record of a set T07 made: its file and x hashes as t07_conditions.json records them, checked."""
    row = record["sets"][set_name]
    path = folder / f"{set_name}.npz"
    if file_sha256(path) != row["file_sha256"]:
        raise RangeIndexError(f"{path}: file sha256 differs from {CONDITIONS_JSON}")
    if x_sha256(_load_x(path, "x")) != row["x_sha256"]:
        raise RangeIndexError(f"{path}: x sha256 differs from {CONDITIONS_JSON}")
    return {"source": f"T07 new set ({CONDITIONS_DIR}/{CONDITIONS_JSON} {CONDITIONS_SHA256})",
            "x_path": _relative(path), "x_array": "x", "x_file_sha256": row["file_sha256"],
            "x_sha256": row["x_sha256"], "seed": int(row["seed"])}


def dev_stored(set_name: str, corruption: str, dev_rows: Mapping[str, Any], clean: Mapping[str, np.ndarray]
               ) -> dict[str, Any]:
    """The dev record of a stored severity-5 set: E10's V_select of that condition, checked as T07 checked it."""
    meta = read_dataset_json(STORED_DATASET, set_name)
    arrays = load_split(STORED_DATASET, "V_select", set_name)
    if split_sha256(arrays) != meta["splits"]["V_select"]["sha256"]:
        raise RangeIndexError(f"{STORED_DATASET}/{set_name}: V_select content hash differs from its dataset.json")
    expected = dev_rows["rows"][f"{corruption}/V_select"]["x_sha256"]
    if x_sha256(arrays["x"]) != expected:
        raise RangeIndexError(f"{STORED_DATASET}/{set_name}: V_select x differs from E10's dev rows")
    for key in ("y", "idx"):
        if arrays[key].dtype != clean[key].dtype or not np.array_equal(arrays[key], clean[key]):
            raise RangeIndexError(f"{STORED_DATASET}/{set_name}: V_select {key} is not the clean V_select's")
    path = dataset_dir(STORED_DATASET, set_name) / "splits" / "V_select.npz"
    return {"source": f"E10 dev rows ({STORED_DATASET}/conditions/{set_name} V_select, {DEV_ROWS_REL})",
            "x_path": _relative(path), "x_array": "x", "x_file_sha256": file_sha256(path), "x_sha256": expected}


def build_index(dataset: str, install_path: Path, run_info: Mapping[str, Any], write: bool = True
                ) -> list[dict[str, Any]]:
    """Write (or, with ``write`` False, only build and check) the index files of one dataset: the 21 sets T07 made
    and the four stored severity-5 sets. Refuses before writing anything if any file exists or any check fails."""
    if dataset not in DATASETS:
        raise RangeIndexError(f"{dataset} is not a T07 student dataset {DATASETS}")
    clean_meta = read_dataset_json(dataset)
    clean_select_sha = clean_meta["splits"]["V_select"]["sha256"]
    if clean_select_sha != CLEAN_SPLIT_SHA256:
        raise RangeIndexError(f"{dataset}: clean V_select {clean_select_sha}, T07 read {CLEAN_SPLIT_SHA256}")
    clean = load_split(dataset, "V_select")
    if split_sha256(clean) != clean_select_sha:
        raise RangeIndexError(f"{dataset}: clean V_select content hash differs from its dataset.json")
    clean_test = clean_meta["test"]
    if clean_test.get("n") != TEST_N or "format" in clean_test:
        raise RangeIndexError(f"{dataset}: the clean test is not the store layout with n {TEST_N}")
    folder = private_dir() / CONDITIONS_DIR
    if file_sha256(folder / CONDITIONS_JSON) != CONDITIONS_SHA256:
        raise RangeIndexError(f"{folder / CONDITIONS_JSON}: not the T07 sets of record {CONDITIONS_SHA256}")
    record = json.loads((folder / CONDITIONS_JSON).read_text(encoding="utf-8"))
    if record["clean"]["x_sha256"] != x_sha256(clean["x"]):
        raise RangeIndexError(f"{dataset}: the T07 sets were made from other clean rows")
    if file_sha256(private_dir() / DEV_ROWS_REL) != DEV_ROWS_SHA256:
        raise RangeIndexError(f"{DEV_ROWS_REL}: not E10's dev rows of record")
    dev_rows = json.loads((private_dir() / DEV_ROWS_REL).read_text(encoding="utf-8"))
    install = json.loads(Path(install_path).read_text(encoding="utf-8"))  # the one read under $WEIGHTCODE_SEALED
    install_sha = file_sha256(Path(install_path))
    names = sorted(record["sets"]) + list(STORED)
    if len(set(names)) != len(names):
        raise RangeIndexError("a set is both made by T07 and stored")
    manifests = []
    for name in names:
        corruption, severity = parse_set(name)
        if name in record["sets"]:
            row = record["sets"][name]
            if (row["corruption"], int(row["severity"])) != (corruption, severity):
                raise RangeIndexError(f"{name}: {CONDITIONS_JSON} records {row['corruption']} s{row['severity']}")
            dev = dev_new(name, record, folder)
        else:
            dev = dev_stored(name, corruption, dev_rows, clean)
        dev.update(split="V_select", n=int(len(clean["y"])), clean_sha256=clean_select_sha)
        test = test_block(corruption, severity, install, clean_test["sha256"], TEST_N)
        if name in STORED:
            same_test(test, read_dataset_json(STORED_DATASET, name)["test"], f"{STORED_DATASET}/{name}")
        if dataset_json_path(dataset, name).is_file():
            same_test(test, read_dataset_json(dataset, name)["test"], f"{dataset}/conditions/{name}")
        manifests.append({
            "schema": INDEX_SCHEMA, "dataset": dataset, "set": name, "corruption": corruption, "severity": severity,
            "dev": dev, "test": test,
            "install_record": {"location": f"$WEIGHTCODE_SEALED/{INSTALL_REL}", "sha256": install_sha,
                               "read": "files entries only (sha256), as weightcode.data.e10.build_condition"},
            "created_by": dict(run_info),
        })
    targets = [index_path(dataset, m["set"]) for m in manifests]
    existing = [str(t) for t in targets if t.exists()]
    if existing:
        raise RangeIndexError(f"index files exist: {existing}; nothing written")
    if write:
        for target, manifest in zip(targets, manifests):
            write_json(target, manifest)
    return manifests


__all__ = ["CONDITIONS_SHA256", "DATASETS", "INDEX_SCHEMA", "RangeIndexError", "STORED", "build_index", "index_path",
           "parse_set", "same_test", "test_block", "test_rows", "x_sha256"]
