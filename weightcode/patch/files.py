"""Safe reading of patch JSON files and the .npy files a patch references (contract section 5.0).

The executor runs outside the editor sandbox, on files the editor wrote, so every read is defensive:

- The final path component must be a regular file, checked with ``lstat`` and then opened with
  ``O_NOFOLLOW | O_NONBLOCK``; the opened descriptor must be the inode that ``lstat`` saw, so a file swapped
  for a symlink, FIFO or other file between the two calls is refused instead of followed (``O_NONBLOCK``
  keeps a swapped-in FIFO from blocking the open).
- Referenced names must fully match ``NPY_NAME_PATTERN`` (no ``/``, so no ``..`` traversal and no
  subdirectory) and are opened relative to one descriptor of the patch directory.
- Sizes are capped before and while reading: 2 MiB for the patch JSON, ``Limits.max_file_bytes`` in total
  over the distinct referenced .npy files (each distinct name is read and counted once).
- A .npy header must declare exactly as many data bytes as the file holds, which rules out allocation bombs
  before ``np.load(allow_pickle=False)`` runs. Only integer and floating dtypes are accepted.

Messages name files by their bare name only, never by a host path.
"""

from __future__ import annotations

import errno
import io
import json
import math
import os
import re
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npy_format

from weightcode.patch.errors import MessageLog, PatchError

MAX_PATCH_JSON_BYTES = 2 * 2**20
NPY_NAME_PATTERN = r"[A-Za-z0-9_.-]{1,128}\.npy"

_NPY_NAME = re.compile(NPY_NAME_PATTERN)
_NPY_MAGIC = b"\x93NUMPY"
_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_READ_CHUNK = 1 << 20

PathLike = str | os.PathLike[str]


class _FileProblem(Exception):
    """One problem with one file, phrased to follow the file's label."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text


def is_npy_name(value: object) -> bool:
    """True if ``value`` is a bare .npy file name that fully matches ``NPY_NAME_PATTERN``."""
    return isinstance(value, str) and _NPY_NAME.fullmatch(value) is not None


def _clip(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _lstat_regular(path: PathLike, dir_fd: int | None) -> os.stat_result:
    try:
        info = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        raise _FileProblem("not found") from None
    except OSError as exc:
        raise _FileProblem(f"cannot be inspected ({exc.strerror})") from None
    if stat.S_ISLNK(info.st_mode):
        raise _FileProblem("is a symbolic link; only regular files are accepted")
    if not stat.S_ISREG(info.st_mode):
        raise _FileProblem("is not a regular file")
    return info


def _read_capped(fd: int, cap: int) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = os.read(fd, min(_READ_CHUNK, cap + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > cap:
            raise _FileProblem(f"is larger than the {cap}-byte limit that applies to it")


def _read_regular(path: PathLike, dir_fd: int | None, seen: os.stat_result, cap: int) -> bytes:
    try:
        fd = os.open(path, _OPEN_FLAGS, dir_fd=dir_fd)
    except FileNotFoundError:
        raise _FileProblem("disappeared before it could be opened") from None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _FileProblem("is a symbolic link; only regular files are accepted") from None
        raise _FileProblem(f"cannot be opened ({exc.strerror})") from None
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (seen.st_dev, seen.st_ino):
            raise _FileProblem("was replaced while it was being opened; write it again and retry")
        return _read_capped(fd, cap)
    finally:
        os.close(fd)


class _DuplicateKey(ValueError):
    pass


class _NonFiniteConstant(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(name: str) -> float:
    raise _NonFiniteConstant(name)


def load_patch(path: PathLike) -> dict[str, Any]:
    """Read one patch JSON file and return its top-level object, without validating the patch schema.

    The file must be a regular file (``lstat``, then ``O_NOFOLLOW``), at most ``MAX_PATCH_JSON_BYTES``
    bytes, UTF-8, strict JSON (no NaN or Infinity constants, no duplicate keys) and a JSON object at the top
    level. Raises ``PatchError`` otherwise. ``apply_patch`` validates the content.
    """
    path = Path(path)
    label = f"patch file {_clip(path.name, 80)!r}"
    try:
        seen = _lstat_regular(path, None)
        if seen.st_size > MAX_PATCH_JSON_BYTES:
            raise _FileProblem(f"is {seen.st_size} bytes, above the {MAX_PATCH_JSON_BYTES}-byte limit")
        data = _read_regular(path, None, seen, MAX_PATCH_JSON_BYTES)
    except _FileProblem as problem:
        raise PatchError(f"{label}: {problem.text}") from None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchError(f"{label}: is not valid UTF-8 (byte {exc.start})") from None
    try:
        patch = json.loads(text, object_pairs_hook=_object_without_duplicates, parse_constant=_reject_constant)
    except _DuplicateKey as exc:
        raise PatchError(f"{label}: an object repeats the key {_clip(str(exc), 80)!r}") from None
    except _NonFiniteConstant as exc:
        raise PatchError(f"{label}: contains {exc}, which is not a finite JSON number") from None
    except json.JSONDecodeError as exc:
        raise PatchError(f"{label}: is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})") from None
    except ValueError as exc:
        raise PatchError(f"{label}: is not valid JSON ({_clip(str(exc))})") from None
    except RecursionError:
        raise PatchError(f"{label}: is nested too deeply") from None
    if not isinstance(patch, dict):
        raise PatchError(f"{label}: the top level must be a JSON object, got a JSON {type(patch).__name__}")
    return patch


def _decode_npy(data: bytes) -> np.ndarray:
    if not data.startswith(_NPY_MAGIC):
        raise _FileProblem("is not a .npy file (write it with numpy.save)")
    stream = io.BytesIO(data)
    try:
        version = npy_format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = npy_format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = npy_format.read_array_header_2_0(stream)
        else:
            raise _FileProblem(f"uses .npy format version {version[0]}.{version[1]}; versions 1.0 and 2.0 are accepted")
    except ValueError as exc:
        raise _FileProblem(f"has an invalid .npy header ({_clip(str(exc))})") from None
    if dtype.names is not None or dtype.subdtype is not None or dtype.kind not in "iuf":
        raise _FileProblem(f"has dtype {dtype}; only integer and floating dtypes are accepted")
    declared = math.prod(shape) * dtype.itemsize
    held = len(data) - stream.tell()
    if declared != held:
        raise _FileProblem(f"declares shape {tuple(shape)} of {dtype} ({declared} data bytes) but holds {held} data bytes")
    stream.seek(0)
    try:
        array = np.load(stream, allow_pickle=False)
    except (ValueError, EOFError, OSError) as exc:
        raise _FileProblem(f"could not be loaded ({_clip(str(exc))})") from None
    if not isinstance(array, np.ndarray) or array.shape != tuple(shape):
        raise _FileProblem("did not load as a single array")
    values = np.array(array, dtype=np.float64, order="C")
    non_finite = values.size - int(np.count_nonzero(np.isfinite(values)))
    if non_finite:
        raise _FileProblem(f"contains {non_finite} non-finite values")
    return values


def load_npy_files(
    patch_dir: PathLike | None,
    refs: Sequence[tuple[str, str]],
    max_total_bytes: int,
    log: MessageLog,
) -> dict[str, np.ndarray]:
    """Load the distinct files that ``refs`` name as float64 C-order arrays, logging every problem.

    ``refs`` pairs a location such as ``ops[2].u`` with a name that already passed ``is_npy_name``. Every
    file is checked (``lstat``, regular file, total size) before any is read; nothing is read if the total
    exceeds ``max_total_bytes``. A file with a problem is absent from the result and its problem is logged
    once, with every location that references it.
    """
    users: dict[str, list[str]] = {}
    for where, name in refs:
        users.setdefault(name, []).append(where)
    if not users:
        return {}

    def label(name: str) -> str:
        return f"file {name!r} ({', '.join(users[name])})"

    if patch_dir is None:
        for name in users:
            log.add(label(name), "no patch directory was given, so referenced files cannot be read")
        return {}
    try:
        dir_fd = os.open(patch_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as exc:
        log.add("patch directory", f"cannot be opened ({exc.strerror})")
        return {}
    try:
        seen: dict[str, os.stat_result] = {}
        for name in users:
            try:
                seen[name] = _lstat_regular(name, dir_fd)
            except _FileProblem as problem:
                log.add(label(name), problem.text)
        total = sum(info.st_size for info in seen.values())
        if total > max_total_bytes:
            log.add("files", f"the referenced .npy files total {total} bytes, above the limit of {max_total_bytes} bytes")
            return {}
        arrays: dict[str, np.ndarray] = {}
        budget = max_total_bytes
        for name, info in seen.items():
            try:
                data = _read_regular(name, dir_fd, info, budget)
                budget -= len(data)
                arrays[name] = _decode_npy(data)
            except _FileProblem as problem:
                log.add(label(name), problem.text)
        return arrays
    finally:
        os.close(dir_fd)
