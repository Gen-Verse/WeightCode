"""PREREG E10 step 1: install CIFAR-10-C (Zenodo record 2535967) under $WEIGHTCODE_SEALED and record every file's sha256.

    python scripts/e10/cifar10c_install.py --tar DOWNLOADED_TAR --dest $WEIGHTCODE_SEALED/cifar10c

Checks the tar against the md5 Zenodo publishes for it (``ZENODO_MD5``), records its sha256, refuses any member that
is not a regular file or directory under ``CIFAR-10-C/``, extracts with tarfile's "data" filter, and writes
``install.json`` beside the extracted directory (each file's size and sha256, the tar's md5 and sha256, the record's
DOI and URL). It hashes bytes only: no array is loaded, so labels.npy is never read as data. Refuses to overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

ZENODO_RECORD = "2535967"
ZENODO_DOI = "10.5281/zenodo.2535967"
ZENODO_URL = "https://zenodo.org/api/records/2535967/files/CIFAR-10-C.tar/content"
ZENODO_MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
ZENODO_SIZE = 2918471680
TOP = "CIFAR-10-C"


def _digests(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)
    record_path = args.dest / "install.json"
    if (args.dest / TOP).exists() or record_path.exists():
        raise SystemExit(f"{args.dest} already holds an install; nothing written")
    size = args.tar.stat().st_size
    md5, sha = _digests(args.tar)
    if size != ZENODO_SIZE or md5 != ZENODO_MD5:
        raise SystemExit(f"{args.tar}: size {size} md5 {md5}, expected {ZENODO_SIZE} {ZENODO_MD5}")
    with tarfile.open(args.tar) as archive:
        members = archive.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if not (member.isfile() or member.isdir()) or not parts or parts[0] != TOP or ".." in parts:
                raise SystemExit(f"refused member {member.name!r} (type {member.type!r})")
        archive.extractall(args.dest, members=members, filter="data")
    files = {}
    for path in sorted((args.dest / TOP).rglob("*")):
        if path.is_file():
            files[str(path.relative_to(args.dest))] = {"size": path.stat().st_size,
                                                       "sha256": _digests(path)[1]}
    record = {
        "schema": "weightcode.e10.cifar10c_install.v1",
        "source": {"zenodo_record": ZENODO_RECORD, "doi": ZENODO_DOI, "url": ZENODO_URL,
                   "tar_size": size, "tar_md5": md5, "tar_md5_expected": ZENODO_MD5, "tar_sha256": sha},
        "extracted_to": str(args.dest / TOP),
        "n_files": len(files),
        "files": files,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "bytes hashed only; no array loaded (labels.npy never read as data)",
    }
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    with open(record_path, "x", encoding="utf-8") as handle:
        handle.write(text)
    print(f"{len(files)} files; tar sha256 {sha}; -> {record_path} {hashlib.sha256(text.encode()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
