"""Install, split and store CIFAR-100 for few-shot class-incremental learning (``weightcode.data.cifar100_fscil``, PREREG
E9), with the ResNet-20 students' numpy forward.py.

Steps: ``install`` copies the official archive and the nine TOPIC/CEC session lists into
$WEIGHTCODE_PRIVATE/raw/cifar100/ (read-only; every file must hash to its constant) and records their provenance.
``build`` parses the archive and writes dataset ``cifar100_fscil``: the splits (train, old_select, old_check,
old_replay, V_evidence, V_check, V_select, ref_pool), forward.py (``templates/forward_resnet20.py.tmpl``) and
dataset.json; the official test set goes to $WEIGHTCODE_SEALED/cifar100_fscil/test.npz only. ``verify`` re-derives every
split from the raw files and compares hashes (the sealed file is not read).

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/fscil_store.py install --tarball TARBALL --index-dir DIR --fetch-note "..."
    .venv/bin/python scripts/fscil_store.py build
    .venv/bin/python scripts/fscil_store.py verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from weightcode.data import cifar100_fscil as C
from weightcode.data.sources import provenance_path
from weightcode.data.store import forward_path, read_json
from weightcode.models.forward_gen import render_resnet20_forward, write_forward
from weightcode.models.runinfo import run_info

REPO_ROOT = Path(__file__).resolve().parents[1]


def forward_source() -> str:
    """forward.py of the ResNet-20 cosine-head students of ``cifar100_fscil``."""
    return render_resnet20_forward(C.MEAN, C.STD, C.N_CLASSES, C.COSINE_SCALE, C.BN_EPS, C.NORM_EPS, C.IMAGE_SHAPE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install", help="install the archive and the session lists into the raw store")
    install.add_argument("--tarball", type=Path, required=True, help="downloaded cifar-100-python.tar.gz")
    install.add_argument("--index-dir", type=Path, required=True, help="directory with session_1.txt .. session_9.txt")
    install.add_argument("--fetch-note", required=True, help="how the files were fetched")
    commands.add_parser("build", help="write the splits, forward.py and dataset.json")
    commands.add_parser("verify", help="re-derive from the raw files and compare hashes")
    args = parser.parse_args(argv)
    if args.command == "install":
        entry = C.install_raw(args.tarball, args.index_dir, args.fetch_note)
        print(json.dumps(entry["files"], indent=2))
        return 0
    built = C.build(C.load_raw())
    if args.command == "verify":
        report = C.verify(built)
        source_ok = forward_path(C.DATASET).read_text(encoding="utf-8") == forward_source()
        print(json.dumps({**{key: value["match"] for key, value in report.items()}, "forward": source_ok}, indent=2))
        return 0 if source_ok and all(value["match"] for value in report.values()) else 1
    provenance = read_json(provenance_path())["sources"][C.SOURCE]
    forward_sha = write_forward(forward_path(C.DATASET), forward_source())
    manifest = C.write(built, forward_sha, provenance, run_info(REPO_ROOT))
    print(json.dumps({"dataset": C.DATASET, "split_sizes": manifest["split_sizes"],
                      "splits": {name: block["sha256"] for name, block in manifest["splits"].items()},
                      "test_n": manifest["test"]["n"], "forward_sha256": forward_sha}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
