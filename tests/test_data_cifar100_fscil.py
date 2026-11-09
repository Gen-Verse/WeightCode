"""weightcode.data.cifar100_fscil (PREREG E9): the store of CIFAR-100 under the TOPIC/CEC few-shot class-incremental
protocol.

A tiny synthetic archive with the official pickle layout and synthetic session lists stand in for the real files
(per-class sizes and file hashes shrunk or replaced with monkeypatch); the store is a tmp directory (WEIGHTCODE_PRIVATE and
WEIGHTCODE_SEALED point there), never the real one.
"""

from __future__ import annotations

import collections
import hashlib
import io
import os
import pickle
import stat
import tarfile
from pathlib import Path

import numpy as np
import pytest

from weightcode.data import cifar100_fscil as C
from weightcode.data.store import read_dataset_json, sealed_test_path, split_sha256

N_TRAIN, N_TEST = 10, 2
BASE = {"old_select": 1, "old_check": 1, "old_replay": 1, "train": 7}
NEW = {"V_check": 2, "V_select": 2, "ref_pool": 1}


# Protocol 3 writes bytes with BINBYTES. A Python 3 protocol-2 pickle routes bytes through _codecs.encode, a global
# the official Python 2 files never name (they use BINSTRING), so RestrictedUnpickler refuses it.


def _tar_member(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _fake_files(root: Path) -> tuple[Path, Path, dict]:
    rng = np.random.default_rng(5)
    y_train = rng.permutation(np.repeat(np.arange(C.N_CLASSES), N_TRAIN))
    y_test = rng.permutation(np.repeat(np.arange(C.N_CLASSES), N_TEST))
    x_train = rng.integers(0, 256, size=(y_train.size, 3072), dtype=np.uint8)
    x_test = rng.integers(0, 256, size=(y_test.size, 3072), dtype=np.uint8)
    batch = lambda x, y: {b"batch_label": b"fake", b"fine_labels": [int(v) for v in y], b"coarse_labels": [0] * y.size,
                          b"data": x, b"filenames": [b"f%d.png" % i for i in range(y.size)]}
    names = [f"class{c:02d}".encode() for c in range(C.N_CLASSES)]
    tarball = root / "cifar-100-python.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        _tar_member(archive, "cifar-100-python/train", pickle.dumps(batch(x_train, y_train), protocol=3))
        _tar_member(archive, "cifar-100-python/test", pickle.dumps(batch(x_test, y_test), protocol=3))
        _tar_member(archive, "cifar-100-python/meta", pickle.dumps({b"fine_label_names": names}, protocol=3))
    index_dir = root / "index"
    index_dir.mkdir()
    base_rows = rng.permutation(np.flatnonzero(y_train < C.N_BASE_CLASSES))
    lists = {1: base_rows}
    for session in C.SESSIONS[1:]:
        lists[session] = np.concatenate([rng.permutation(np.flatnonzero(y_train == c))[: C.SHOTS]
                                         for c in C.session_classes(session)])
    hashes = {}
    for session, rows in lists.items():
        data = ("\n".join(str(int(r)) for r in rows) + "\n").encode("ascii")
        (index_dir / f"session_{session}.txt").write_bytes(data)
        hashes[session] = hashlib.sha256(data).hexdigest()
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    md5 = hashlib.md5(tarball.read_bytes()).hexdigest()
    return tarball, index_dir, {"sha256": digest, "md5": md5, "bytes": tarball.stat().st_size, "lists": hashes,
                                "y_train": y_train, "x_train": x_train, "rows": lists}


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    for variable in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED"):
        (tmp_path / variable).mkdir()
        monkeypatch.setenv(variable, str(tmp_path / variable))
    files = tmp_path / "downloads"
    files.mkdir()
    tarball, index_dir, info = _fake_files(files)
    monkeypatch.setattr(C, "N_TRAIN_PER_CLASS", N_TRAIN)
    monkeypatch.setattr(C, "N_TEST_PER_CLASS", N_TEST)
    monkeypatch.setattr(C, "BASE_PER_CLASS", BASE)
    monkeypatch.setattr(C, "NEW_PER_CLASS", NEW)
    monkeypatch.setattr(C, "TARBALL_SHA256", info["sha256"])
    monkeypatch.setattr(C, "TARBALL_MD5", info["md5"])
    monkeypatch.setattr(C, "TARBALL_BYTES", info["bytes"])
    monkeypatch.setattr(C, "INDEX_SHA256", info["lists"])
    C.install_raw(tarball, index_dir, "synthetic test archive")
    return {"tarball": tarball, "index_dir": index_dir, "tmp": tmp_path, **info}


def test_split_sizes_disjoint_and_deterministic(fake):
    raw = C.load_raw()
    built, again = C.build(raw), C.build(C.load_raw())
    for name in C.STORED_SPLITS:
        assert split_sha256(built.splits[name]) == split_sha256(again.splits[name])
    counts = {name: np.bincount(arrays["y"], minlength=C.N_CLASSES) for name, arrays in built.splits.items()}
    for name, per_class in BASE.items():
        assert counts[name][: C.N_BASE_CLASSES].tolist() == [per_class] * C.N_BASE_CLASSES
        assert counts[name][C.N_BASE_CLASSES :].sum() == 0
    for name, per_class in {"V_evidence": C.SHOTS, **NEW}.items():
        assert counts[name][C.N_BASE_CLASSES :].tolist() == [per_class] * (C.N_CLASSES - C.N_BASE_CLASSES)
        assert counts[name][: C.N_BASE_CLASSES].sum() == 0
    every = np.concatenate([built.splits[name]["idx"] for name in C.STORED_SPLITS])
    assert every.size == C.N_CLASSES * N_TRAIN and np.unique(every).size == every.size
    shots = np.concatenate([fake["rows"][s] for s in C.SESSIONS[1:]])
    np.testing.assert_array_equal(built.splits["V_evidence"]["idx"], shots)  # session order, then list order
    for name in C.STORED_SPLITS:
        idx = built.splits[name]["idx"]
        if name != "V_evidence":
            assert np.all(np.diff(idx) > 0)
        np.testing.assert_array_equal(built.splits[name]["x"], fake["x_train"][idx].reshape(-1, 3, 32, 32))
        np.testing.assert_array_equal(built.splits[name]["y"], fake["y_train"][idx])
    assert built.manifest["sessions"]["classes"]["2"] == [60, 61, 62, 63, 64]
    assert built.manifest["sessions"]["classes"]["9"] == [95, 96, 97, 98, 99]
    assert built.manifest["class_names"][7] == "class07"


def test_write_verify_and_sealed_test(fake, monkeypatch):
    built = C.build(C.load_raw())
    manifest = C.write(built, "f" * 64, {"note": "test"}, {"git": "test"})
    with pytest.raises(FileExistsError):
        C.write(built, "f" * 64, {}, {})
    sealed = sealed_test_path(C.DATASET)
    assert stat.S_IMODE(os.stat(sealed).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(sealed.parent).st_mode) == 0o700
    assert manifest["test"]["n"] == C.N_CLASSES * N_TEST
    recorded = read_dataset_json(C.DATASET)
    for key in ("class_names", "forward_sha256", "splits", "sessions", "normalization", "index_lists"):
        assert key in recorded
    assert set(recorded["splits"]) == set(C.STORED_SPLITS)
    # verify never opens the sealed file.
    real_load = np.load

    def guarded(path, *args, **kwargs):
        if Path(path).resolve() == sealed.resolve():
            raise AssertionError("the sealed test was read")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded)
    report = C.verify(C.build(C.load_raw()))
    assert all(value["match"] for value in report.values()), report


def test_refuses_wrong_list_hash(fake, tmp_path):
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for session in C.SESSIONS:
        data = (fake["index_dir"] / f"session_{session}.txt").read_bytes()
        (tampered / f"session_{session}.txt").write_bytes(data + b"\n" if session == 4 else data)
    with pytest.raises(C.Cifar100FscilError, match="session 4"):
        C.load_raw(index_dir=tampered)
    rows = dict(fake["rows"])
    rows[3] = rows[3][::-1]  # class order broken
    with pytest.raises(C.Cifar100FscilError, match="session_3"):
        C.check_session_lists(fake["y_train"], rows)


def test_restricted_unpickler_refuses_other_globals():
    for payload in (collections.OrderedDict(a=1), {b"data": os.getcwd}, Path("x")):
        with pytest.raises(pickle.UnpicklingError, match="not allowed"):
            C.RestrictedUnpickler(io.BytesIO(pickle.dumps(payload, protocol=3)), encoding="bytes").load()
    array = np.arange(6, dtype=np.uint8).reshape(2, 3)
    loaded = C.RestrictedUnpickler(io.BytesIO(pickle.dumps({b"data": array}, protocol=3)), encoding="bytes").load()
    np.testing.assert_array_equal(loaded[b"data"], array)


def test_session_classes():
    assert C.session_classes(1) == list(range(60))
    assert C.session_classes(5) == [75, 76, 77, 78, 79]
    assert [C.session_of_class(c) for c in (0, 59, 60, 64, 65, 99)] == [1, 1, 2, 2, 3, 9]
    with pytest.raises(ValueError):
        C.session_classes(10)
