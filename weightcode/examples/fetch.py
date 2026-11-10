"""Downloads for the paper examples, each file checked against a pinned sha256 before use.

The Hugging Face dataset ``Boom5426/WeightCode`` is read at one pinned revision (``HF_REVISION``). Its ``SHA256SUMS``
file is itself pinned by hash (``HF_SHA256SUMS_SHA256``), and every other file is checked against the line that
``SHA256SUMS`` gives it, so a changed or truncated download is refused. ``HF_ENDPOINT`` (the variable the Hugging Face
tools use) selects a mirror. Files already present with the right hash are not downloaded again.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path

HF_REPO = "Boom5426/WeightCode"
HF_REVISION = "fef94dc474d826aa6a58586281f1bafcdc2e1d88"
HF_SHA256SUMS = "SHA256SUMS"
HF_SHA256SUMS_SHA256 = "31edb7a83866bb9950e527a8f2d7d55826089a75552f1a34efa6a8247faba0e4"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
USER_AGENT = "weightcode-examples"
TIMEOUT_S = 60.0


class FetchError(RuntimeError):
    """A download failed or a file does not have its recorded sha256."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, sha256: str) -> Path:
    """``dest`` with content of hash ``sha256``: kept if already right, otherwise downloaded from ``url`` and checked."""
    if dest.is_file() and file_sha256(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise FetchError(f"could not download {url}: {error}") from error
    got = file_sha256(partial)
    if got != sha256:
        partial.unlink()
        raise FetchError(f"{url}: sha256 {got} is not the recorded {sha256}")
    partial.replace(dest)
    return dest


def hf_url(path: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")
    return f"{endpoint}/datasets/{HF_REPO}/resolve/{HF_REVISION}/{path}"


def parse_sha256sums(text: str) -> dict[str, str]:
    """``{path: sha256}`` from ``sha256sum`` output lines (``<hash>  <path>``)."""
    sums: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, _, path = line.partition("  ")
        if len(digest) != 64 or not path:
            raise FetchError(f"{HF_SHA256SUMS}: unreadable line {line!r}")
        sums[path.lstrip("*")] = digest
    return sums


def hf_files(prefixes: Iterable[str], root: Path) -> dict[str, Path]:
    """Every dataset file whose path starts with one of ``prefixes``, downloaded under ``root`` at its dataset path.

    A prefix that names no file is an error, so a renamed path cannot silently drop a file."""
    sums_path = download(hf_url(HF_SHA256SUMS), root / HF_SHA256SUMS, HF_SHA256SUMS_SHA256)
    sums = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    chosen: dict[str, str] = {}
    for prefix in prefixes:
        hits = {path: digest for path, digest in sums.items() if path == prefix or path.startswith(prefix)}
        if not hits:
            raise FetchError(f"{HF_REPO}@{HF_REVISION[:12]} has no file under {prefix!r}")
        chosen.update(hits)
    files: dict[str, Path] = {}
    for path, digest in sorted(chosen.items()):
        files[path] = download(hf_url(path), root / path, digest)
    return files


def checked_files(sources: Mapping[str, str], sha256: Mapping[str, str], root: Path,
                  local_dir: Path | None = None) -> dict[str, Path]:
    """Files ``{relative path: url}`` under ``root``, each with its recorded hash; ``local_dir`` (an existing copy,
    such as a git checkout) is used instead of downloading when given."""
    files: dict[str, Path] = {}
    for relative, url in sources.items():
        dest = root / relative
        if local_dir is not None:
            source = local_dir / relative
            if not source.is_file():
                raise FetchError(f"{source} is missing")
            got = file_sha256(source)
            if got != sha256[relative]:
                raise FetchError(f"{source}: sha256 {got} is not the recorded {sha256[relative]}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            files[relative] = dest
        else:
            files[relative] = download(url, dest, sha256[relative])
    return files


__all__ = ["DEFAULT_HF_ENDPOINT", "FetchError", "HF_REPO", "HF_REVISION", "HF_SHA256SUMS_SHA256", "checked_files",
           "download", "file_sha256", "hf_files", "hf_url", "parse_sha256sums"]
