"""Private snapshots of editor patches (HARNESS_SPEC section 4).

The editor can rewrite /work/patches at any time, also from a background thread of its kernel while a tool call
runs. Every patch-check, predict and declare_final call therefore first copies the patch JSON and every .npy file it
may reference into a fresh private attempt directory outside the sandbox, reading each file once; the executor
then validates and applies that copy. What was checked is exactly what is stored, and finalisation re-applies the
stored copy and verifies the child hash. The copy uses the same file-safety rules as ``weightcode.patch`` (bare names,
regular files only, ``lstat`` then ``O_NOFOLLOW`` and an inode check, size caps); messages name files by bare name
only and never echo file contents.
"""

from __future__ import annotations

import errno
import os
import re
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from weightcode.patch import MAX_PATCH_JSON_BYTES, PatchError, is_npy_name, load_patch

PATCH_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
PATCHES_MOUNT_PREFIX = "/work/patches/"
_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class SnapshotProblem(ValueError):
    """The patch or one of its files cannot be copied safely; ``messages`` say why."""

    def __init__(self, messages: list[str]) -> None:
        self.messages = list(messages)
        super().__init__("\n".join(self.messages))


def patch_name(value: object) -> str:
    """The bare file name of a patch in /work/patches (``/work/patches/<name>`` is accepted too)."""
    if not isinstance(value, str):
        raise SnapshotProblem(["patch_file must be a string: the bare name of a file in /work/patches"])
    name = value[len(PATCHES_MOUNT_PREFIX):] if value.startswith(PATCHES_MOUNT_PREFIX) else value
    if PATCH_NAME_RE.fullmatch(name) is None or name in (".", ".."):
        shown = value if len(value) <= 80 else value[:77] + "..."
        raise SnapshotProblem([f"patch_file {shown!r} is not a bare file name in /work/patches (letters, digits, "
                               "'_', '.', '-'; no '/')"])
    return name


def _read_regular(dir_fd: int, name: str, cap: int, label: str) -> bytes:
    try:
        seen = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        raise SnapshotProblem([f"{label}: not found in /work/patches"]) from None
    except OSError as exc:
        raise SnapshotProblem([f"{label}: cannot be inspected ({exc.strerror})"]) from None
    if stat.S_ISLNK(seen.st_mode):
        raise SnapshotProblem([f"{label}: is a symbolic link; only regular files are accepted"])
    if not stat.S_ISREG(seen.st_mode):
        raise SnapshotProblem([f"{label}: is not a regular file"])
    if seen.st_size > cap:
        raise SnapshotProblem([f"{label}: is {seen.st_size} bytes, above the {cap}-byte limit that applies to it"])
    try:
        fd = os.open(name, _OPEN_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise SnapshotProblem([f"{label}: is a symbolic link; only regular files are accepted"]) from None
        raise SnapshotProblem([f"{label}: cannot be opened ({exc.strerror})"]) from None
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (seen.st_dev, seen.st_ino):
            raise SnapshotProblem([f"{label}: was replaced while it was being read; write it again and retry"])
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, min(1 << 20, cap + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > cap:
                raise SnapshotProblem([f"{label}: is larger than the {cap}-byte limit that applies to it"])
        return b"".join(chunks)
    finally:
        os.close(fd)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def referenced_npy_names(patch: dict[str, Any]) -> list[str]:
    """Every string inside ``patch["ops"]`` that is a valid .npy file name, in first-seen order.

    The executor decides which of them are file fields; copying a superset is harmless.
    """
    names: list[str] = []
    for value in _strings(patch.get("ops", [])):
        if is_npy_name(value) and value not in names:
            names.append(value)
    return names


def write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def snapshot_patch(patches_dir: Path, name: str, dest: Path, max_file_bytes: int) -> tuple[dict[str, Any], dict[str, int]]:
    """Copy patch ``name`` and the .npy files it references from ``patches_dir`` into the new directory ``dest``.

    Returns the loaded patch (``weightcode.patch.load_patch`` of the private copy) and ``{file name: bytes}`` of what was
    copied. Raises ``SnapshotProblem`` or ``PatchError`` with editor-safe messages. A referenced name that does not
    exist is not copied; the executor then reports it as missing.
    """
    dest.mkdir(mode=0o700)
    dir_fd = os.open(patches_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    copied: dict[str, int] = {}
    try:
        data = _read_regular(dir_fd, name, MAX_PATCH_JSON_BYTES, f"patch file {name!r}")
        write_private(dest / name, data)
        copied[name] = len(data)
        patch = load_patch(dest / name)
        problems: list[str] = []
        total = 0
        for npy in referenced_npy_names(patch):
            if npy == name:
                continue
            try:
                blob = _read_regular(dir_fd, npy, max(0, max_file_bytes - total), f"file {npy!r}")
            except SnapshotProblem as problem:
                if problem.messages == [f"file {npy!r}: not found in /work/patches"]:
                    continue  # the executor reports a missing file field with its location
                problems.extend(problem.messages)
                continue
            total += len(blob)
            write_private(dest / npy, blob)
            copied[npy] = len(blob)
        if problems:
            raise SnapshotProblem(problems)
        return patch, copied
    finally:
        os.close(dir_fd)


def snapshot_file(source_dir: Path, name: str, dest: Path, cap: int, label: str) -> int:
    """Copy the one regular file ``name`` of ``source_dir`` into the new directory ``dest`` (an update program).

    The same file-safety rules as for a patch; returns the byte count. Raises ``SnapshotProblem`` with editor-safe
    messages that name the file by ``label`` only.
    """
    dest.mkdir(mode=0o700)
    dir_fd = os.open(source_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        data = _read_regular(dir_fd, name, cap, label)
        write_private(dest / name, data)
        return len(data)
    finally:
        os.close(dir_fd)


__all__ = ["PATCH_NAME_RE", "PatchError", "SnapshotProblem", "patch_name", "referenced_npy_names", "snapshot_file",
           "snapshot_patch"]
