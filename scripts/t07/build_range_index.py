"""PREREG "T07 sealed read (list v5)": write the range index of the two T07 student datasets.

One dataset.json per (dataset, set) under ``$WEIGHTCODE_PRIVATE/<dataset>/t07_range/<set>/`` (``weightcode.data.t07_range_index``):
the 21 sets T07 made and the four stored severity-5 sets, 25 per dataset. The only file read under $WEIGHTCODE_SEALED is the
install record ``$WEIGHTCODE_SEALED/cifar10c/install.json`` (its ``files`` entries: sha256 per official file). ``--check``
builds and checks every record and writes nothing. Nothing existing is overwritten.

    python scripts/t07/build_range_index.py --spec <PREREG commit> [--check]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from weightcode.common.paths import sealed_dir
from weightcode.data import t07_range_index as RI
from weightcode.evalkit.targets import file_sha256

REPO = Path(__file__).resolve().parents[2]


def _commit() -> str:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return head.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--spec", required=True, help="the PREREG commit the index is written under")
    parser.add_argument("--dataset", action="append", choices=RI.DATASETS, help="default: both")
    parser.add_argument("--check", action="store_true", help="build and check only; write nothing")
    args = parser.parse_args(argv)
    install = sealed_dir() / RI.INSTALL_REL
    run_info = {"script": "scripts/t07/build_range_index.py", "spec": args.spec, "commit": _commit(),
                "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    datasets = args.dataset or list(RI.DATASETS)
    built = {dataset: RI.build_index(dataset, install, run_info, write=False) for dataset in datasets}
    for dataset, manifests in built.items():  # every dataset checked before any file is written
        if not args.check:
            RI.build_index(dataset, install, run_info, write=True)
        for manifest in manifests:
            path = RI.index_path(dataset, manifest["set"])
            written = file_sha256(path)[:12] if not args.check else "not written"
            print(f"{dataset} {manifest['set']}: test {manifest['test']['rows']} sha256 "
                  f"{manifest['test']['sha256'][:12]}, dev x {manifest['dev']['x_sha256'][:12]}, file {written}",
                  flush=True)
    print(f"{'checked' if args.check else 'wrote'} {sum(map(len, built.values()))} index files", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
