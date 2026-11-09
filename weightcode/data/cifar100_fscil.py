"""CIFAR-100 for few-shot class-incremental learning (PREREG E9, rule fscil-v1): the official archive, the TOPIC/CEC
session lists and the store dataset ``cifar100_fscil``.

Source: ``TARBALL_URL`` (the official python version), installed once as
``$WEIGHTCODE_PRIVATE/raw/cifar100/cifar-100-python.tar.gz`` together with the nine session lists of the TOPIC/CEC protocol
(``INDEX_REPO`` at ``INDEX_COMMIT``, ``data/index_list/cifar100/session_<s>.txt``) under
``$WEIGHTCODE_PRIVATE/raw/cifar100/index_list/``; every file must hash to its constant here and is recorded in
``raw/provenance.json``. The archive's pickles are read with ``RestrictedUnpickler``, which resolves only the three
numpy globals the official files name (``ALLOWED_GLOBALS``) and refuses any other.

Protocol (TOPIC/CEC, verified against the pinned lists): session 1 is the base session, classes 0-59 with every train
image (500 per class); session s = 2..9 adds the five classes 60 + 5(s - 2) .. 64 + 5(s - 2) with the five listed
shots of each, in class order. Class order is CIFAR-100 fine-label order; labels are stored unpermuted.

Splits (``STORED_SPLITS``; x uint8 [N, 3, 32, 32] from the row-major R, G, B planes, y the fine label, idx the index
into the official train array):

* base classes 0-59, per class (``BASE_PER_CLASS``): ``old_select`` 30, ``old_check`` 30, ``old_replay`` 10 and
  ``train`` the remaining 430, dealt by ``weightcode.data.splits.stratified_split`` on the base-class rows with
  ``[SPLIT_SEED, BASE_HOLDOUT_STREAM]`` in the order ``BASE_ORDER``. The base trains on ``train`` only, a recorded
  deviation from the protocol's 500 per class.
* new classes 60-99: ``V_evidence`` is the 200 listed shots of sessions 2 to 9, stored in session order and then list
  order (not sorted). The other 495 rows of each class are dealt with ``[SPLIT_SEED, NEW_CLASS_STREAM]`` in the order
  ``NEW_ORDER`` into ``V_check`` 100, ``V_select`` 200 and ``ref_pool`` 195 (``NEW_PER_CLASS``). ``ref_pool`` is read
  only by the all-class reference.
* the official test set (10,000 images, idx into the official test array) is written only, to
  ``$WEIGHTCODE_SEALED/cifar100_fscil/test.npz``; nothing here reads it back.

Every split other than V_evidence is sorted by idx. ``build`` is a pure function of the raw files, so two builds give
byte-identical splits; ``verify`` re-derives every split and compares canonical hashes.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import os
import pickle
import shutil
import stat
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.paths import sealed_dir
from weightcode.data.registry import SPLIT_SEED
from weightcode.data.sources import PROVENANCE_SCHEMA, provenance_path, raw_dir
from weightcode.data.splits import stratified_split
from weightcode.data.store import (
    dataset_dir,
    dataset_json_path,
    load_split,
    read_dataset_json,
    read_json,
    save_split,
    sealed_test_path,
    split_path,
    split_sha256,
    write_json,
)

DATASET = "cifar100_fscil"
SOURCE = "cifar-100-python"
TARBALL_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
TARBALL_NAME = "cifar-100-python.tar.gz"
TARBALL_BYTES = 169_001_437
TARBALL_SHA256 = "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
TARBALL_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
"""The md5 published with the archive (also checked by torchvision)."""
MEMBERS: Mapping[str, str] = {"train": "cifar-100-python/train", "test": "cifar-100-python/test",
                              "meta": "cifar-100-python/meta"}
INDEX_REPO = "icoz69/CEC-CVPR2021"
INDEX_COMMIT = "3a5d335f19349aa2e5fa75c88aa2c701b6e5513d"
INDEX_URL = "https://raw.githubusercontent.com/" + INDEX_REPO + "/{commit}/data/index_list/cifar100/session_{session}.txt"
INDEX_SHA256: Mapping[int, str] = {
    1: "5f402fb2c5260a05bb02ea000f9b9e32f23e305281969afa6ea47a9a22b69cf7",
    2: "654db7b72e383a1aac94a4e570ea26c9f3cb8b7bf95a7aff2a9e1626f4e0b330",
    3: "a466c0f920549096435c9b298e9f52f2b4a676db34f2c54de04f4147759a2eea",
    4: "c2d0f0a6db516358055d448b1dd8033a778ae7ba30de78adbd8ea2e1b94282b0",
    5: "2907fe7d341f9f4da0c2cdf59bcf33a50619f41e9ebd3232fe2c6c11bc30bfa4",
    6: "87f70d04e01acb3ba0bbce24f84bf78ee905721d3609e53c5f11bf92d3237197",
    7: "70131153942c3a30418843097a482d0e24dedbaa394876b101db205a5099a595",
    8: "8bcefabac5ff2d3086e3cc097b306713661f07aa670ef3935570226293e1c2f1",
    9: "242fd41330db6081d11216a8324f92aebd64545b2171bd567a162a7521a62ed7",
}
"""sha256 of session_<s>.txt (PREREG E9); FACT (825933b) and TOPIC (e76d37c) ship byte-identical lists."""

N_CLASSES = 100
N_BASE_CLASSES = 60
WAYS = 5
SHOTS = 5
N_SESSIONS = 9
SESSIONS: tuple[int, ...] = tuple(range(1, N_SESSIONS + 1))
IMAGE_SHAPE: tuple[int, int, int] = (3, 32, 32)
N_TRAIN_PER_CLASS = 500
N_TEST_PER_CLASS = 100
BASE_HOLDOUT_STREAM = 707
NEW_CLASS_STREAM = 708
BASE_ORDER: tuple[str, ...] = ("old_select", "old_check", "old_replay", "train")
BASE_PER_CLASS: Mapping[str, int] = {"old_select": 30, "old_check": 30, "old_replay": 10, "train": 430}
NEW_ORDER: tuple[str, ...] = ("V_check", "V_select", "ref_pool")
NEW_PER_CLASS: Mapping[str, int] = {"V_check": 100, "V_select": 200, "ref_pool": 195}
STORED_SPLITS: tuple[str, ...] = ("train", "old_select", "old_check", "old_replay", "V_evidence", "V_check",
                                  "V_select", "ref_pool")
MEAN: tuple[float, float, float] = (0.507, 0.487, 0.441)
STD: tuple[float, float, float] = (0.267, 0.256, 0.276)
"""CEC's normalisation (dataloader/cifar100/cifar.py, l.78 and l.84)."""
COSINE_SCALE = 16.0
BN_EPS = 1e-5
NORM_EPS = 1e-12
DATASET_SCHEMA = "weightcode.dataset.fscil.v1"
ALLOWED_GLOBALS: tuple[tuple[str, str], ...] = (
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
)
"""The only globals the archive's pickles may name: numpy's array reconstruction (checked on the official files with
pickletools: train and test name the first three, meta none)."""

SPLIT_RULE = (
    "base classes 0-59: per class old_select 30, old_check 30, old_replay 10, train 430, by "
    "weightcode.data.splits.stratified_split on the base-class train rows with seed [SPLIT_SEED, 707], dealing order "
    "old_select, old_check, old_replay, train; new classes 60-99: V_evidence = the listed shots of session_2..9.txt "
    "in session order then list order, the other 495 rows per class dealt with seed [SPLIT_SEED, 708] in the order "
    "V_check 100, V_select 200, ref_pool 195; every split but V_evidence sorted by official train index"
)


class Cifar100FscilError(RuntimeError):
    """A raw file, session list or stored split does not match the protocol; nothing is overwritten."""


def session_classes(session: int) -> list[int]:
    """Classes added in ``session``: 0-59 for session 1, else the five classes 60 + 5(s - 2) .. 64 + 5(s - 2)."""
    if session not in SESSIONS:
        raise ValueError(f"session must be in 1..{N_SESSIONS}, got {session!r}")
    if session == 1:
        return list(range(N_BASE_CLASSES))
    first = N_BASE_CLASSES + WAYS * (session - 2)
    return list(range(first, first + WAYS))


def session_of_class(label: int) -> int:
    """The session that adds class ``label``."""
    if not 0 <= int(label) < N_CLASSES:
        raise ValueError(f"label must be in 0..{N_CLASSES - 1}, got {label!r}")
    return 1 if label < N_BASE_CLASSES else 2 + (int(label) - N_BASE_CLASSES) // WAYS


# ---------------------------------------------------------------------------------------------------------------
# Raw files


def raw_root() -> Path:
    """``$WEIGHTCODE_PRIVATE/raw/cifar100``."""
    return raw_dir() / "cifar100"


def tarball_path() -> Path:
    """Installed location of the official archive."""
    return raw_root() / TARBALL_NAME


def index_list_path(session: int) -> Path:
    """Installed location of ``session_<s>.txt``."""
    if session not in SESSIONS:
        raise ValueError(f"session must be in 1..{N_SESSIONS}, got {session!r}")
    return raw_root() / "index_list" / f"session_{session}.txt"


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _install_file(source: Path, target: Path, sha256: str) -> None:
    if not target.exists():
        if _digest(source) != sha256:
            raise Cifar100FscilError(f"{source} does not hash to the recorded sha256 {sha256}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    if _digest(target) != sha256:
        raise Cifar100FscilError(f"{target} changed: sha256 {_digest(target)} != {sha256}")


def install_raw(tarball: Path, index_dir: Path, fetch_note: str) -> dict[str, Any]:
    """Copy the archive and the nine session lists into the raw store (read-only), check their hashes and record
    their provenance in ``raw/provenance.json``.

    Idempotent: installed files must hash to their constants, and an existing provenance entry must record the same
    files.
    """
    _install_file(Path(tarball), tarball_path(), TARBALL_SHA256)
    if tarball_path().stat().st_size != TARBALL_BYTES:
        raise Cifar100FscilError(f"{tarball_path()} has {tarball_path().stat().st_size} bytes, not {TARBALL_BYTES}")
    for session in SESSIONS:
        _install_file(Path(index_dir) / f"session_{session}.txt", index_list_path(session), INDEX_SHA256[session])
    root = raw_dir()
    files = {str(tarball_path().relative_to(root)): {"sha256": TARBALL_SHA256, "bytes": TARBALL_BYTES,
                                                    "md5": _digest(tarball_path(), "md5")}}
    if files[str(tarball_path().relative_to(root))]["md5"] != TARBALL_MD5:
        raise Cifar100FscilError(f"{tarball_path()} md5 differs from the published {TARBALL_MD5}")
    for session in SESSIONS:
        path = index_list_path(session)
        files[str(path.relative_to(root))] = {"sha256": INDEX_SHA256[session], "bytes": path.stat().st_size}
    with tarfile.open(tarball_path(), "r:gz") as archive:
        members = {info.name: {"bytes": info.size} for info in archive.getmembers() if info.isfile()}
    entry = {
        "url": TARBALL_URL,
        "index_lists": {"repo": INDEX_REPO, "commit": INDEX_COMMIT,
                        "url": INDEX_URL.format(commit=INDEX_COMMIT, session="<s>")},
        "files": files,
        "members": members,
        "fetch": fetch_note,
        "parser": "weightcode.data.cifar100_fscil (tarfile + RestrictedUnpickler; torchvision is not used)",
    }
    record_path = provenance_path()
    record = read_json(record_path) if record_path.exists() else {"schema": PROVENANCE_SCHEMA, "sources": {}}
    if record.get("schema") != PROVENANCE_SCHEMA:
        raise Cifar100FscilError(f"unexpected provenance schema in {record_path}")
    previous = record["sources"].get(SOURCE)
    if previous is not None and previous["files"] != entry["files"]:
        raise Cifar100FscilError(f"{SOURCE} raw files changed since they were recorded in {record_path}")
    record["sources"][SOURCE] = entry
    write_json(record_path, record)
    return entry


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that resolves only ``ALLOWED_GLOBALS`` and refuses every other global."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a CIFAR-100 pickle")
        if name == "_reconstruct":
            return importlib.import_module("numpy._core.multiarray")._reconstruct
        return getattr(np, name)


def _unpickle(data: bytes) -> dict[bytes, Any]:
    payload = RestrictedUnpickler(io.BytesIO(data), encoding="bytes").load()
    if not isinstance(payload, dict):
        raise Cifar100FscilError(f"a CIFAR-100 pickle holds {type(payload).__name__}, not a dict")
    return payload


@dataclass(frozen=True)
class RawCifar100:
    """The official arrays (train and test in archive order) and the nine session lists."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: tuple[str, ...]
    session_lists: Mapping[int, np.ndarray]


def parse_index_list(data: bytes, origin: str) -> np.ndarray:
    """One integer index per non-empty line (ASCII digits only)."""
    lines = [line.strip() for line in data.decode("ascii").splitlines()]
    values = [line for line in lines if line]
    if not values or not all(value.isdigit() for value in values):
        raise Cifar100FscilError(f"{origin}: every non-empty line must be a non-negative integer")
    return np.array([int(value) for value in values], dtype=np.int64)


def _images(batch: Mapping[bytes, Any], n: int, origin: str) -> tuple[np.ndarray, np.ndarray]:
    data, labels = np.asarray(batch[b"data"]), np.asarray(batch[b"fine_labels"], dtype=np.int64)
    if data.dtype != np.uint8 or data.shape != (n, int(np.prod(IMAGE_SHAPE))) or labels.shape != (n,):
        raise Cifar100FscilError(f"{origin}: data {data.dtype}{data.shape}, labels {labels.shape}, expected {n} rows")
    return np.ascontiguousarray(data.reshape(n, *IMAGE_SHAPE)), labels


def load_raw(tarball: Path | None = None, index_dir: Path | None = None) -> RawCifar100:
    """Read the installed (or the given) archive and session lists after checking their hashes."""
    tarball = tarball_path() if tarball is None else Path(tarball)
    if _digest(tarball) != TARBALL_SHA256:
        raise Cifar100FscilError(f"{tarball} does not hash to {TARBALL_SHA256}")
    lists = {}
    for session in SESSIONS:
        path = index_list_path(session) if index_dir is None else Path(index_dir) / f"session_{session}.txt"
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != INDEX_SHA256[session]:
            raise Cifar100FscilError(f"{path} does not hash to the PREREG E9 sha256 of session {session}")
        lists[session] = parse_index_list(data, str(path))
    with tarfile.open(tarball, "r:gz") as archive:
        payloads = {}
        for key, member in MEMBERS.items():
            handle = archive.extractfile(member)
            if handle is None:
                raise Cifar100FscilError(f"{tarball} has no file member {member}")
            payloads[key] = _unpickle(handle.read())
    x_train, y_train = _images(payloads["train"], N_CLASSES * N_TRAIN_PER_CLASS, "train")
    x_test, y_test = _images(payloads["test"], N_CLASSES * N_TEST_PER_CLASS, "test")
    names = tuple(bytes(name).decode("ascii") for name in payloads["meta"][b"fine_label_names"])
    if len(names) != N_CLASSES or len(set(names)) != N_CLASSES:
        raise Cifar100FscilError(f"meta has {len(names)} fine label names ({len(set(names))} distinct)")
    for origin, y, per_class in (("train", y_train, N_TRAIN_PER_CLASS), ("test", y_test, N_TEST_PER_CLASS)):
        if not np.array_equal(np.bincount(y, minlength=N_CLASSES), np.full(N_CLASSES, per_class)):
            raise Cifar100FscilError(f"{origin}: expected {per_class} images of each of {N_CLASSES} classes")
    return RawCifar100(x_train, y_train, x_test, y_test, names, lists)


# ---------------------------------------------------------------------------------------------------------------
# Build, write, verify


@dataclass
class BuiltFscil:
    """All stored arrays of ``cifar100_fscil`` plus the manifest fields that do not depend on files on disk."""

    splits: dict[str, dict[str, np.ndarray]]
    test: dict[str, np.ndarray]
    manifest: dict[str, Any]


def check_session_lists(y_train: np.ndarray, lists: Mapping[int, np.ndarray]) -> None:
    """The protocol facts of the lists: session 1 is every base-class train row; session s = 2..9 has SHOTS rows of
    each of its WAYS classes, in class order; all rows are distinct and in range."""
    y = np.asarray(y_train)
    base = np.flatnonzero(y < N_BASE_CLASSES)
    first = np.asarray(lists[1])
    if first.size != base.size or np.unique(first).size != first.size or not np.array_equal(np.sort(first), base):
        raise Cifar100FscilError("session_1.txt is not the set of base-class train rows")
    shots = []
    for session in SESSIONS[1:]:
        rows = np.asarray(lists[session])
        expected = np.repeat(session_classes(session), SHOTS)
        if rows.size != WAYS * SHOTS or rows.min() < 0 or rows.max() >= y.size or not np.array_equal(y[rows], expected):
            raise Cifar100FscilError(f"session_{session}.txt does not list {SHOTS} rows of each of "
                                     f"{session_classes(session)} in class order")
        shots.append(rows)
    every = np.concatenate([first, *shots])
    if np.unique(every).size != every.size:
        raise Cifar100FscilError("the session lists share a row")


def _rows(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    return {"x": np.ascontiguousarray(x[idx]), "y": y[idx].astype(np.int64), "idx": idx}


def split_indices(y_train: np.ndarray, lists: Mapping[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Official train indices of every stored split (pure; ``SPLIT_RULE``)."""
    y = np.asarray(y_train, dtype=np.int64)
    check_session_lists(y, lists)
    base_rows = np.flatnonzero(y < N_BASE_CLASSES)
    base_sizes = {name: N_BASE_CLASSES * BASE_PER_CLASS[name] for name in BASE_ORDER}
    base = stratified_split(y[base_rows], base_sizes, [SPLIT_SEED, BASE_HOLDOUT_STREAM], N_BASE_CLASSES, BASE_ORDER)
    shots = np.concatenate([np.asarray(lists[s], dtype=np.int64) for s in SESSIONS[1:]])
    new_rows = np.setdiff1d(np.flatnonzero(y >= N_BASE_CLASSES), shots)
    n_new_classes = N_CLASSES - N_BASE_CLASSES
    new_sizes = {name: n_new_classes * NEW_PER_CLASS[name] for name in NEW_ORDER}
    new = stratified_split(y[new_rows], new_sizes, [SPLIT_SEED, NEW_CLASS_STREAM], N_CLASSES, NEW_ORDER)
    result = {name: base_rows[base[name]] for name in BASE_ORDER}
    result["V_evidence"] = shots
    result.update({name: new_rows[new[name]] for name in NEW_ORDER})
    every = np.concatenate([result[name] for name in STORED_SPLITS])
    if every.size != y.size or np.unique(every).size != y.size:
        raise AssertionError("the stored splits are not a partition of the official train set")
    return {name: result[name] for name in STORED_SPLITS}


def build(raw: RawCifar100) -> BuiltFscil:
    """Every stored split, the sealed test and the manifest (pure function of the raw arrays and lists)."""
    indices = split_indices(raw.y_train, raw.session_lists)
    splits = {name: _rows(raw.x_train, raw.y_train, rows) for name, rows in indices.items()}
    test = _rows(raw.x_test, raw.y_test, np.arange(raw.y_test.size, dtype=np.int64))
    manifest = {
        "schema": DATASET_SCHEMA,
        "dataset": DATASET,
        "task": "fscil",
        "source": SOURCE,
        "class_names": list(raw.class_names),
        "n_classes": N_CLASSES,
        "image_shape": list(IMAGE_SHAPE),
        "normalization": {"mean": list(MEAN), "std": list(STD),
                          "rule": "x / 255, then per channel (x - mean) / std (CEC dataloader/cifar100/cifar.py)"},
        "head": {"op": "cosine_linear", "scale": COSINE_SCALE, "norm_eps": NORM_EPS, "bn_eps": BN_EPS},
        "sessions": {
            "n_base_classes": N_BASE_CLASSES, "ways": WAYS, "shots": SHOTS, "n_sessions": N_SESSIONS,
            "classes": {str(s): session_classes(s) for s in SESSIONS},
        },
        "index_lists": {"repo": INDEX_REPO, "commit": INDEX_COMMIT,
                        "sha256": {str(s): INDEX_SHA256[s] for s in SESSIONS}},
        "split_rule": SPLIT_RULE,
        "streams": {"base_holdouts": [SPLIT_SEED, BASE_HOLDOUT_STREAM],
                    "new_class_split": [SPLIT_SEED, NEW_CLASS_STREAM]},
        "split_seed": SPLIT_SEED,
        "per_class": {"base": dict(BASE_PER_CLASS), "new": {"V_evidence": SHOTS, **NEW_PER_CLASS}},
        "split_sizes": {name: int(arrays["y"].size) for name, arrays in splits.items()},
        "splits": {
            name: {"path": f"splits/{name}.npz", "sha256": split_sha256(arrays), "n": int(arrays["y"].size),
                   "class_counts": np.bincount(arrays["y"], minlength=N_CLASSES).astype(int).tolist()}
            for name, arrays in splits.items()
        },
        "test": {"sealed": True, "n": int(test["y"].size), "sha256": split_sha256(test),
                 "rule": "the official test set; read only by the sealed evaluation"},
    }
    return BuiltFscil(splits, test, manifest)


def write(built: BuiltFscil, forward_sha256: str, provenance: Mapping[str, Any], run_info: Mapping[str, Any]) -> dict[str, Any]:
    """Write the splits, the sealed test and dataset.json (refuses an existing dataset.json)."""
    manifest_path = dataset_json_path(DATASET)
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; use verify instead of rebuilding")
    dataset_dir(DATASET).mkdir(parents=True, exist_ok=True)
    for name, arrays in built.splits.items():
        if save_split(split_path(DATASET, name), arrays) != built.manifest["splits"][name]["sha256"]:
            raise RuntimeError(f"{name}: written hash differs from the built hash")
    sealed = sealed_dir() / DATASET
    sealed.mkdir(mode=0o700, exist_ok=True)
    sealed.chmod(0o700)
    if save_split(sealed_test_path(DATASET), built.test, private_mode=True) != built.manifest["test"]["sha256"]:
        raise RuntimeError("test: written hash differs from the built hash")
    manifest = {
        **built.manifest,
        "forward_path": "forward.py",
        "forward_sha256": forward_sha256,
        "source_provenance": dict(provenance),
        "created_by": dict(run_info),
    }
    write_json(manifest_path, manifest)
    return manifest


def verify(built: BuiltFscil) -> dict[str, Any]:
    """Compare a fresh build with dataset.json and the private split files (the sealed file is not read)."""
    recorded = read_dataset_json(DATASET)
    report: dict[str, Any] = {}
    for name, arrays in built.splits.items():
        fresh, on_disk = split_sha256(arrays), split_sha256(load_split(DATASET, name))
        report[name] = {"match": fresh == recorded["splits"][name]["sha256"] == on_disk}
    report["test"] = {"match": split_sha256(built.test) == recorded["test"]["sha256"]}
    fields = {key: value for key, value in recorded.items() if key in built.manifest}
    report["manifest"] = {"match": fields == built.manifest}
    return report


__all__ = [
    "ALLOWED_GLOBALS", "BASE_ORDER", "BASE_PER_CLASS", "BuiltFscil", "Cifar100FscilError", "DATASET", "INDEX_SHA256",
    "MEAN", "NEW_ORDER", "NEW_PER_CLASS", "N_BASE_CLASSES", "N_CLASSES", "N_SESSIONS", "RawCifar100",
    "RestrictedUnpickler", "SESSIONS", "SHOTS", "STD", "STORED_SPLITS", "WAYS", "build", "check_session_lists",
    "install_raw", "load_raw", "parse_index_list", "session_classes", "session_of_class", "split_indices",
    "verify", "write",
]
