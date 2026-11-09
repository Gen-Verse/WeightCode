"""Small file helpers shared by the harness: deterministic JSON, durable JSON lines and content hashes."""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Current UTC time, ISO 8601 with microseconds."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")


def json_text(payload: Any) -> str:
    """Deterministic JSON text: sorted keys, two-space indent, trailing newline, no NaN or infinity."""
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True) + "\n"


def write_json(path: str | Path, payload: Any) -> str:
    """Write ``payload`` as deterministic JSON through a temporary file and a rename; return the text's sha256."""
    target = Path(path)
    text = json_text(payload)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with open(temporary, "w", encoding="ascii") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return text_sha256(text)


def read_json(path: str | Path) -> Any:
    """Parse one JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: Any, *, lock: bool = False) -> None:
    """Append one JSON line with a single O_APPEND write and fsync it; ``lock`` adds an exclusive flock."""
    data = (json.dumps(record, sort_keys=True, allow_nan=False, ensure_ascii=True) + "\n").encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        if lock:
            fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(path: str | Path) -> list[Any]:
    """Every line of a JSON-lines file (a missing file has no lines). A malformed line is an error."""
    target = Path(path)
    if not target.exists():
        return []
    records = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{target}:{number} is an empty line")
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{target}:{number} is not valid JSON: {exc}") from exc
    return records


def iter_lines(path: str | Path) -> Iterator[str]:
    """Lines of a text file that may be partial (a killed writer); undecodable bytes are replaced."""
    target = Path(path)
    if not target.exists():
        return iter(())
    return iter(target.read_bytes().decode("utf-8", "replace").splitlines())


def file_sha256(path: str | Path) -> str:
    """sha256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha256(text: str) -> str:
    """sha256 of a text's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
