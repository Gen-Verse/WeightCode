"""PREREG T05: retention on ImageNetV2 matched-frequency of the base, the T04 finals and the main control (no LLM).

    source scripts/env.sh
    python scripts/t05/t05_readout.py --archive imagenetv2-matched-frequency.tar.gz --download-record RECORD \\
        --frozen FROZEN_FINALS --network base=BASE.npz:SHA256 --network final_g1=CHILD.npz:SHA256 ... --out OUT.json

Run only after the T04 finals and the main control are frozen: ``--frozen`` names the record that freezes them (its
sha256 is written into the output), and every network is passed as a params file with the params_sha256 that record
holds for it (refused unless the record's text contains that sha256). ImageNetV2 is a readout only: it chooses nothing and sets no threshold.

Data: the archive whose sha256 the download record holds (scripts/t05/download_imagenetv2.pbs). Each member
``<folder>/<file>.jpeg`` is one image whose label is the folder number (0 .. 999, the standard ImageNet class index,
the order of in1k_t04's class names); the archive must hold 10,000 images, 10 per class. Each image goes through
T04's clean pipeline (``weightcode.data.t04.center_crop_u8``: RGB, resize the shorter side to 256 bilinear with antialias,
center crop 224) and in1k_t04's forward.py (sha256 checked against its dataset.json).

Per network: top-1 accuracy, and against the base: fixed (base wrong, network right), broken (base right, network
wrong) and the break rate among the images the base gets right (broken / n_base_correct). Accuracy differences carry
a paired 95% interval over images (normal approximation of the per-image difference). The member list (names, sizes,
sha256 of the stored bytes) is written beside the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

from weightcode.common.identity import load_params, params_sha256
from weightcode.data import t04 as T
from weightcode.evalkit.targets import load_dataset_forward, read_dataset_meta

N_IMAGES = 10_000
PER_CLASS = 10
BATCH = 250


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


GZIP_MAGIC = b"\x1f\x8b"


def archive_compression(path: Path) -> str:
    """"gzip" or "none": the published imagenetv2-matched-frequency.tar.gz is a plain POSIX tar despite its name (R37
    job 50, first attempt: the gzip reader refused its first bytes before any member was read)."""
    with open(path, "rb") as handle:
        magic = handle.read(2)
    if magic == GZIP_MAGIC:
        return "gzip"
    if not tarfile.is_tarfile(path):
        raise SystemExit(f"{path}: neither gzip nor a tar archive")
    return "none"


def read_archive(path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Member records, uint8 crops [N, 3, 224, 224] and labels, in sorted member-name order."""
    mode = {"gzip": "r:gz", "none": "r:"}[archive_compression(path)]
    with tarfile.open(path, mode) as archive:
        members = sorted((m for m in archive.getmembers() if m.isfile() and m.name.lower().endswith((".jpeg", ".jpg"))),
                         key=lambda m: m.name)
        if len(members) != N_IMAGES:
            raise SystemExit(f"{path}: {len(members)} images, expected {N_IMAGES}")
        records, crops, labels = [], np.empty((N_IMAGES, *T.IMAGE_SHAPE), np.uint8), np.empty(N_IMAGES, np.int64)
        for i, member in enumerate(members):
            folder = Path(member.name).parent.name
            if not folder.isdigit() or not 0 <= int(folder) < T.N_CLASSES:
                raise SystemExit(f"{member.name}: folder is not a class index")
            data = archive.extractfile(member).read()
            records.append({"name": member.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                            "label": int(folder)})
            crops[i] = T.center_crop_u8(data)
            labels[i] = int(folder)
    counts = np.bincount(labels, minlength=T.N_CLASSES)
    if not np.all(counts == PER_CLASS):
        raise SystemExit(f"class counts range {counts.min()} .. {counts.max()}, expected {PER_CLASS} each")
    return records, crops, labels


def paired_interval(a: np.ndarray, b: np.ndarray) -> list[float]:
    """95% normal interval of mean(b - a) over images, in points."""
    d = b.astype(np.float64) - a.astype(np.float64)
    half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return [100.0 * (d.mean() - half), 100.0 * (d.mean() + half)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--download-record", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True, help="the record that froze the T04 finals")
    parser.add_argument("--network", action="append", required=True, help="name=params.npz:params_sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; nothing written")
    download = json.loads(args.download_record.read_text(encoding="utf-8"))
    archive_sha = file_sha256(args.archive)
    if archive_sha != download["sha256"]:
        raise SystemExit(f"{args.archive}: sha256 {archive_sha} differs from the download record")
    frozen_text = args.frozen.read_text(encoding="utf-8")
    networks = {}
    for item in args.network:
        name, _, rest = item.partition("=")
        path, _, expected = rest.rpartition(":")
        if len(expected) != 64 or expected not in frozen_text:
            raise SystemExit(f"{name}: params_sha256 {expected!r} is not in the freeze record {args.frozen}")
        params = load_params(Path(path))
        if params_sha256(params) != expected:
            raise SystemExit(f"{name}: params_sha256 {params_sha256(params)} != {expected}")
        networks[name] = (params, expected, str(path))
    if "base" not in networks:
        raise SystemExit("--network base=... is required (fixed and broken are counted against it)")
    meta = read_dataset_meta(T.DATASET)
    forward, forward_sha, _ = load_dataset_forward(T.DATASET, meta)
    started = time.time()
    records, crops, labels = read_archive(args.archive)
    correct = {}
    for name, (params, _, _) in networks.items():
        pred = np.concatenate([np.asarray(forward(params, crops[i:i + BATCH])).argmax(1)
                               for i in range(0, N_IMAGES, BATCH)])
        correct[name] = pred == labels
    base = correct["base"]
    table = {}
    for name, ok in correct.items():
        fixed, broken = int(np.sum(~base & ok)), int(np.sum(base & ~ok))
        table[name] = {"params_sha256": networks[name][1], "path": networks[name][2], "acc": float(ok.mean()),
                       "n_correct": int(ok.sum()), "fixed": fixed, "broken": broken,
                       "break_rate_among_base_correct": broken / max(1, int(base.sum())),
                       "delta_points": 100.0 * float(ok.mean() - base.mean()),
                       "delta_95ci_points": paired_interval(base, ok)}
    members_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    members_path = args.out.with_suffix(".members.jsonl")
    members_path.write_text(members_text, encoding="utf-8")
    record = {"schema": "weightcode.t05.readout.v1", "archive": {"path": str(args.archive), "sha256": archive_sha,
                                                          "compression": archive_compression(args.archive),
                                                          "download_record": download},
              "frozen_record": {"path": str(args.frozen), "sha256": file_sha256(args.frozen)},
              "members": {"path": str(members_path), "sha256": hashlib.sha256(members_text.encode()).hexdigest(),
                          "n": len(records)},
              "pipeline": "weightcode.data.t04.center_crop_u8, then in1k_t04 forward.py", "forward_sha256": forward_sha,
              "libraries": T.image_libraries(), "networks": table, "wall_s": time.time() - started}
    with open(args.out, "x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    print(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
