"""File safety of weightcode.patch: patch JSON loading and referenced .npy files (contract 5.0, HARNESS_SPEC section 2)."""

from __future__ import annotations

import io
import json
import os
import pickle

import numpy as np
import pytest

from weightcode.common.identity import params_sha256
from weightcode.patch import MAX_PATCH_JSON_BYTES, SCHEMA, Limits, PatchError, apply_patch, apply_patch_file, is_npy_name, load_patch

OPEN = Limits(max_changed_frac=None, max_rel_frob=None, max_edit_frac=None)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the WEIGHTCODE data locations at temporary directories so no test can touch the real store."""
    root = tmp_path_factory.mktemp("weightcode_store")
    for variable in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED", "WEIGHTCODE_RUNS"):
        (root / variable).mkdir()
        monkeypatch.setenv(variable, str(root / variable))


def _model() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {"w": rng.normal(size=(3, 4)).astype(np.float32), "b": rng.normal(size=(3,)).astype(np.float32)}


def _patch(params: dict[str, np.ndarray], *ops: dict) -> dict:
    return {
        "schema": SCHEMA,
        "parent_sha256": params_sha256(params),
        "ops": list(ops),
        "hypothesis": "test hypothesis",
        "predicted_effect": "test prediction",
    }


def _dense(name: str) -> dict:
    return {"op": "add_dense", "tensor": "w", "delta": name}


def _messages(params, patch_dir, *ops, limits=OPEN) -> list[str]:
    with pytest.raises(PatchError) as caught:
        apply_patch(params, _patch(params, *ops), patch_dir, limits, params_sha256(params))
    text = "\n".join(caught.value.messages)
    assert str(patch_dir) not in text, "messages must not reveal host paths"
    return caught.value.messages


def _write_json(path, patch: dict) -> None:
    path.write_text(json.dumps(patch))


# ----------------------------------------------------------------------------------------------- patch JSON


def test_apply_patch_file_reads_files_next_to_the_json(tmp_path, monkeypatch) -> None:
    params = _model()
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    delta = np.full((3, 4), 0.25)
    np.save(patch_dir / "d.npy", delta)
    _write_json(patch_dir / "p1.json", _patch(params, _dense("d.npy")))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    np.save(elsewhere / "d.npy", np.zeros((3, 4)))
    monkeypatch.chdir(elsewhere)
    result = apply_patch_file(params, patch_dir / "p1.json", OPEN, params_sha256(params))
    np.testing.assert_array_equal(result.child["w"], (params["w"].astype(np.float64) + delta).astype(np.float32))
    assert load_patch(patch_dir / "p1.json") == _patch(params, _dense("d.npy"))


def test_load_patch_refuses_a_symlink_even_to_a_valid_patch(tmp_path) -> None:
    params = _model()
    real = tmp_path / "real.json"
    _write_json(real, _patch(params))
    (tmp_path / "link.json").symlink_to(real)
    with pytest.raises(PatchError, match="symbolic link") as caught:
        load_patch(tmp_path / "link.json")
    assert str(tmp_path) not in str(caught.value)
    with pytest.raises(PatchError, match="symbolic link"):
        apply_patch_file(params, tmp_path / "link.json", OPEN, params_sha256(params))


def test_load_patch_size_cap_boundary(tmp_path) -> None:
    body = json.dumps(_patch(_model())).encode()
    exact = tmp_path / "exact.json"
    exact.write_bytes(body + b" " * (MAX_PATCH_JSON_BYTES - len(body)))
    assert exact.stat().st_size == MAX_PATCH_JSON_BYTES == 2 * 2**20
    assert load_patch(exact)["schema"] == SCHEMA
    over = tmp_path / "over.json"
    over.write_bytes(body + b" " * (MAX_PATCH_JSON_BYTES - len(body) + 1))
    with pytest.raises(PatchError, match=f"is {MAX_PATCH_JSON_BYTES + 1} bytes, above the {MAX_PATCH_JSON_BYTES}-byte limit"):
        load_patch(over)


@pytest.mark.parametrize(
    ("content", "needle"),
    [
        (b"\xff\xfe{}", "not valid UTF-8"),
        (b'{"schema": ', "is not valid JSON"),
        (b'{"ops": [], "ops": []}', "repeats the key 'ops'"),
        (b'{"ops": [{"op": "add", "op": "set"}]}', "repeats the key 'op'"),
        (b'{"x": NaN}', "contains NaN"),
        (b'{"x": -Infinity}', "contains -Infinity"),
        (b"[" * 200_000 + b"]" * 200_000, "nested too deeply"),
        (b'{"x": ' + b"9" * 5000 + b"}", "is not valid JSON"),
        (b"[1, 2]", "the top level must be a JSON object, got a JSON list"),
        (b'"text"', "the top level must be a JSON object, got a JSON str"),
    ],
)
def test_load_patch_rejects_malformed_json(tmp_path, content, needle) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    with pytest.raises(PatchError) as caught:
        load_patch(path)
    assert needle in str(caught.value), str(caught.value)


def test_load_patch_refuses_missing_files_directories_and_fifos(tmp_path) -> None:
    with pytest.raises(PatchError, match="'nope.json': not found"):
        load_patch(tmp_path / "nope.json")
    (tmp_path / "dir.json").mkdir()
    with pytest.raises(PatchError, match="is not a regular file"):
        load_patch(tmp_path / "dir.json")
    os.mkfifo(tmp_path / "fifo.json")
    with pytest.raises(PatchError, match="is not a regular file"):
        load_patch(tmp_path / "fifo.json")


# ----------------------------------------------------------------------------------------------- npy names


@pytest.mark.parametrize(
    "name",
    ["../d.npy", "sub/d.npy", "/tmp/d.npy", "d.npy\n", ".npy", "d.npz", "d.NPY", "d", "", "d .npy", "d\x00.npy",
     "d.npy/", "a" * 129 + ".npy", 7, None],
)
def test_referenced_names_must_be_bare_npy_names(tmp_path, name) -> None:
    params = _model()
    patch_dir = tmp_path / "patches"
    (patch_dir / "sub").mkdir(parents=True)
    np.save(tmp_path / "d.npy", np.zeros((3, 4)))
    np.save(patch_dir / "sub" / "d.npy", np.zeros((3, 4)))
    assert not is_npy_name(name)
    messages = _messages(params, patch_dir, _dense(name))
    assert len(messages) == 1 and "delta must be the bare name of a .npy file in the patch directory" in messages[0]


def test_name_pattern_accepts_the_allowed_alphabet_up_to_128_characters() -> None:
    assert is_npy_name("a" * 128 + ".npy") and is_npy_name("dir_01.v-2.npy") and is_npy_name("..npy")


def test_npy_symlink_is_refused_even_inside_the_patch_directory(tmp_path) -> None:
    params = _model()
    np.save(tmp_path / "real.npy", np.ones((3, 4)))
    (tmp_path / "link.npy").symlink_to(tmp_path / "real.npy")
    messages = _messages(params, tmp_path, _dense("link.npy"))
    assert messages == ["file 'link.npy' (ops[0].delta): is a symbolic link; only regular files are accepted"]


def test_npy_must_exist_and_be_a_regular_file(tmp_path) -> None:
    params = _model()
    (tmp_path / "dir.npy").mkdir()
    os.mkfifo(tmp_path / "fifo.npy")
    messages = _messages(params, tmp_path, _dense("missing.npy"), _dense("dir.npy"), _dense("fifo.npy"))
    assert messages == [
        "file 'missing.npy' (ops[0].delta): not found",
        "file 'dir.npy' (ops[1].delta): is not a regular file",
        "file 'fifo.npy' (ops[2].delta): is not a regular file",
    ]


def test_files_need_a_patch_directory(tmp_path) -> None:
    params = _model()
    messages = _messages(params, None, _dense("d.npy"))
    assert messages == ["file 'd.npy' (ops[0].delta): no patch directory was given, so referenced files cannot be read"]
    messages = _messages(params, tmp_path / "absent", _dense("d.npy"))
    assert messages == ["patch directory: cannot be opened (No such file or directory)"]
    (tmp_path / "plain").write_text("not a directory")
    messages = _messages(params, tmp_path / "plain", _dense("d.npy"))
    assert messages == ["patch directory: cannot be opened (Not a directory)"]


def test_patch_dir_may_be_a_string(tmp_path) -> None:
    params = _model()
    np.save(tmp_path / "d.npy", np.ones((3, 4)))
    result = apply_patch(params, _patch(params, _dense("d.npy")), str(tmp_path), OPEN, params_sha256(params))
    assert result.per_tensor["w"]["changed"] == 12


# ----------------------------------------------------------------------------------------------- npy content


def _npy_bytes(array: np.ndarray, allow_pickle: bool = False) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=allow_pickle)
    return buffer.getvalue()


def test_npy_content_rules(tmp_path) -> None:
    params = _model()
    (tmp_path / "object.npy").write_bytes(_npy_bytes(np.array([{"a": 1}], dtype=object), allow_pickle=True))
    (tmp_path / "pickle.npy").write_bytes(pickle.dumps(np.zeros((3, 4))))
    archive = io.BytesIO()
    np.savez(archive, delta=np.zeros((3, 4)))
    (tmp_path / "archive.npy").write_bytes(archive.getvalue())
    (tmp_path / "bool.npy").write_bytes(_npy_bytes(np.zeros((3, 4), dtype=bool)))
    (tmp_path / "complex.npy").write_bytes(_npy_bytes(np.zeros((3, 4), dtype=np.complex128)))
    (tmp_path / "text.npy").write_bytes(_npy_bytes(np.array(["a"] * 12).reshape(3, 4)))
    nan = np.zeros((3, 4))
    nan[1, 2] = np.nan
    nan[0, 0] = np.inf
    (tmp_path / "nan.npy").write_bytes(_npy_bytes(nan))
    whole = _npy_bytes(np.zeros((3, 4)))
    (tmp_path / "short.npy").write_bytes(whole[:-8])
    (tmp_path / "long.npy").write_bytes(whole + b"\0" * 8)
    names = ["object.npy", "pickle.npy", "archive.npy", "bool.npy", "complex.npy", "text.npy", "nan.npy", "short.npy",
             "long.npy"]
    messages = _messages(params, tmp_path, *[_dense(name) for name in names])
    expected = {
        "object.npy": "has dtype object; only integer and floating dtypes are accepted",
        "pickle.npy": "is not a .npy file (write it with numpy.save)",
        "archive.npy": "is not a .npy file (write it with numpy.save)",
        "bool.npy": "has dtype bool; only integer and floating dtypes are accepted",
        "complex.npy": "has dtype complex128; only integer and floating dtypes are accepted",
        "text.npy": "has dtype <U1; only integer and floating dtypes are accepted",
        "nan.npy": "contains 2 non-finite values",
        "short.npy": "declares shape (3, 4) of float64 (96 data bytes) but holds 88 data bytes",
        "long.npy": "declares shape (3, 4) of float64 (96 data bytes) but holds 104 data bytes",
    }
    assert messages == [f"file {name!r} (ops[{k}].delta): {expected[name]}" for k, name in enumerate(names)]


def test_npy_header_claiming_a_huge_array_is_refused_before_allocation(tmp_path) -> None:
    params = _model()
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False, "shape": (10**12,)})
    (tmp_path / "bomb.npy").write_bytes(header.getvalue() + b"\0" * 16)
    messages = _messages(params, tmp_path, _dense("bomb.npy"))
    assert messages == ["file 'bomb.npy' (ops[0].delta): declares shape (1000000000000,) of float64 (8000000000000 "
                        "data bytes) but holds 16 data bytes"]


def test_fortran_order_and_big_endian_files_load_by_value(tmp_path) -> None:
    params = _model()
    delta = np.asfortranarray((np.arange(12).reshape(3, 4) / 8).astype(">f4"))
    assert delta.dtype.byteorder == ">" and not delta.flags.c_contiguous
    np.save(tmp_path / "f.npy", delta)
    result = apply_patch(params, _patch(params, _dense("f.npy")), tmp_path, OPEN, params_sha256(params))
    expected = params["w"].astype(np.float64) + delta.astype(np.float64)
    np.testing.assert_array_equal(result.child["w"], expected.astype(np.float32))


# ----------------------------------------------------------------------------------------------- byte budget


def test_total_referenced_bytes_boundary_and_distinct_files_count_once(tmp_path) -> None:
    params = _model()
    np.save(tmp_path / "a.npy", np.full((3, 4), 0.5))
    np.save(tmp_path / "b.npy", np.full((3, 4), -0.25))
    total = (tmp_path / "a.npy").stat().st_size + (tmp_path / "b.npy").stat().st_size
    ops = (_dense("a.npy"), _dense("b.npy"), _dense("a.npy"))
    exact = Limits(max_changed_frac=None, max_rel_frob=None, max_edit_frac=None, max_file_bytes=total)
    result = apply_patch(params, _patch(params, *ops), tmp_path, exact, params_sha256(params))
    np.testing.assert_array_equal(result.child["w"], (params["w"].astype(np.float64) + 0.75).astype(np.float32))
    below = Limits(max_changed_frac=None, max_rel_frob=None, max_edit_frac=None, max_file_bytes=total - 1)
    assert _messages(params, tmp_path, *ops, limits=below) == [
        f"files: the referenced .npy files total {total} bytes, above the limit of {total - 1} bytes"
    ]


def test_default_64_mib_budget_refuses_before_reading(tmp_path) -> None:
    params = _model()
    with open(tmp_path / "sparse.npy", "wb") as handle:
        handle.truncate(64 * 2**20 + 1)
    assert _messages(params, tmp_path, _dense("sparse.npy"), limits=Limits()) == [
        f"files: the referenced .npy files total {64 * 2**20 + 1} bytes, above the limit of {64 * 2**20} bytes"
    ]


# ----------------------------------------------------------------------------------------------- races


def _swap_before_open(monkeypatch, swaps: dict) -> None:
    """Run ``swaps[name]()`` once, right before the executor's O_NOFOLLOW open of a path ending in ``name``."""
    real_open = os.open
    pending = dict(swaps)

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        if flags & os.O_NOFOLLOW:
            for name in list(pending):
                if os.fspath(path).endswith(name):
                    pending.pop(name)()
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swapping_open)


def test_npy_swapped_for_a_symlink_after_lstat_is_refused(tmp_path, monkeypatch) -> None:
    params = _model()
    np.save(tmp_path / "d.npy", np.zeros((3, 4)))
    np.save(tmp_path / "other.npy", np.ones((3, 4)))

    def swap() -> None:
        (tmp_path / "d.npy").unlink()
        (tmp_path / "d.npy").symlink_to(tmp_path / "other.npy")

    _swap_before_open(monkeypatch, {"d.npy": swap})
    assert _messages(params, tmp_path, _dense("d.npy")) == [
        "file 'd.npy' (ops[0].delta): is a symbolic link; only regular files are accepted"
    ]


def test_npy_replaced_by_another_file_or_a_fifo_after_lstat_is_refused(tmp_path, monkeypatch) -> None:
    params = _model()
    np.save(tmp_path / "d.npy", np.zeros((3, 4)))
    np.save(tmp_path / "e.npy", np.zeros((3, 4)))
    np.save(tmp_path / "other.npy", np.ones((3, 4)))

    def replace_file() -> None:
        os.replace(tmp_path / "other.npy", tmp_path / "d.npy")

    def replace_with_fifo() -> None:
        (tmp_path / "e.npy").unlink()
        os.mkfifo(tmp_path / "e.npy")

    _swap_before_open(monkeypatch, {"d.npy": replace_file, "e.npy": replace_with_fifo})
    assert _messages(params, tmp_path, _dense("d.npy"), _dense("e.npy")) == [
        "file 'd.npy' (ops[0].delta): was replaced while it was being opened; write it again and retry",
        "file 'e.npy' (ops[1].delta): was replaced while it was being opened; write it again and retry",
    ]


def test_patch_json_swapped_for_a_symlink_after_lstat_is_refused(tmp_path, monkeypatch) -> None:
    params = _model()
    _write_json(tmp_path / "p.json", _patch(params))
    _write_json(tmp_path / "other.json", _patch(params))

    def swap() -> None:
        (tmp_path / "p.json").unlink()
        (tmp_path / "p.json").symlink_to(tmp_path / "other.json")

    _swap_before_open(monkeypatch, {"p.json": swap})
    with pytest.raises(PatchError, match="'p.json': is a symbolic link"):
        load_patch(tmp_path / "p.json")
