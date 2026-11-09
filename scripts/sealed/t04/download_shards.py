"""Download the 14 ImageNet-1k validation shards and classes.py for the local sealed read of in1k_t04 (PREREG "T04",
sealed test: "Local images").

    source scripts/env.sh
    .venv/bin/python scripts/sealed/t04/download_shards.py --kit $WEIGHTCODE_RUNS/t04/sealed_kit_v6

Runs only after the user approves the sealed list. Files go to ``$WEIGHTCODE_SEALED/in1k_t04/hf_49e2ee26``
(``weightcode.evalkit.sealed_t04.SHARDS_LOCATION``) by ``huggingface_hub.snapshot_download`` at the pinned revision, with the
same file patterns as the NSCC download (scripts/t03/download_imagenet_val.pbs). huggingface_hub authenticates with its
own cached login; this script never reads, prints or passes a token. The kit's manifest.sha256 (the NSCC manifest,
sha256 pinned in ``weightcode.data.imagenet_val.MANIFEST_SHA256``) is copied next to the files and every listed file is
checked against it (``imagenet_val.verify_manifest``); a mismatch is an error. Nothing is decoded or opened as an
image. The record (file names, sizes, sha256, library version, wall time) goes to
``$WEIGHTCODE_RUNS/t04/sealed_download/download_<time>.json``; nothing else is printed.

An interrupted download can be rerun: snapshot_download keeps the files it finished, and the manifest check decides.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from weightcode.common.paths import runs_dir, sealed_dir
from weightcode.data import imagenet_val as V
from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit import sealed_t04 as ST
from weightcode.evalkit.targets import file_sha256, utc_now

ALLOW_PATTERNS = [V.CLASSES_FILE, "data/validation-*.parquet"]
"""As the NSCC download of R29 job 1 (scripts/t03/download_imagenet_val.pbs)."""
MIN_FREE_GB = 10.0
"""The shards are 6.69 GB (manifest); the rest is headroom for the read's unit files and the store."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kit", type=Path, required=True, help="the pulled kit (scripts/sealed/t04/pull_kit.sh)")
    parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    from huggingface_hub import __version__ as hub_version
    from huggingface_hub import snapshot_download

    ST.check_kit(args.kit)
    target = SF.resolve_location(ST.SHARDS_LOCATION, "the in1k_t04 shards")
    free_gb = shutil.disk_usage(sealed_dir()).free / 1e9
    if free_gb < args.min_free_gb:
        raise SystemExit(f"{free_gb:.1f} GB free under $WEIGHTCODE_SEALED; {args.min_free_gb} GB needed; nothing downloaded")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.time()
    snapshot_download(V.REPO_ID, repo_type="dataset", revision=V.REVISION, local_dir=str(target),
                      allow_patterns=ALLOW_PATTERNS, max_workers=args.max_workers)
    manifest = target / V.MANIFEST
    source_manifest = Path(args.kit) / V.MANIFEST
    if manifest.exists() and file_sha256(manifest) != file_sha256(source_manifest):
        raise SystemExit(f"{manifest} exists and is not the kit's manifest")
    shutil.copyfile(source_manifest, manifest)
    listed = V.verify_manifest(target)
    expected = {V.CLASSES_FILE} | {f"data/{name}" for name in V.shard_names()}
    if set(listed) != expected:
        raise SystemExit(f"the manifest lists {sorted(set(listed) ^ expected)} beyond or short of the 15 files")
    record = {"schema": "weightcode.t04.shard_download.v1", "time": utc_now(), "repo_id": V.REPO_ID,
              "revision": V.REVISION, "location": ST.SHARDS_LOCATION, "manifest_sha256": file_sha256(manifest),
              "files": {name: {"sha256": digest, "bytes": (target / name).stat().st_size}
                        for name, digest in sorted(listed.items())},
              "huggingface_hub": hub_version, "wall_s": time.time() - started, "manifest_check": "passed"}
    out = runs_dir() / "t04" / "sealed_download" / f"download_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    print(f"{len(listed)} files match the manifest; record {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
