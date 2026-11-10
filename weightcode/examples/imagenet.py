"""Example ``imagenet`` (Figure 1b, right): 64 + 64 labeled images make an ImageNet ResNet-18 robust to Gaussian noise.

The network of record (instance ``i4c8c63bd``) is torchvision's ImageNet ResNet-18. Two programs that a frozen language
model wrote from 64 noisy and 64 clean labeled images (sessions ``sece3a52c03e092dd`` and ``sec0ee969740e55ce``) built
two edited networks. On all 38,000 held-out test images the paper reports top-1 under Gaussian noise (ImageNet-C,
severity 3) of 21.66% for the original and 32.38% and 39.67% for the two programs, with clean top-1 69.70%, 69.45% and
69.30%: net gains of +10.47 and +17.62 points.

This example is a small, fair look at that result. It

1. downloads the original network and the two edited networks from the Hugging Face dataset (pinned revision, every
   file checked by sha256; each edited network must hash to the child of record), the paper's test record with every
   image's correctness, ImageNet-C's official ``make_imagenet_c.py`` at the commit the paper used, and the ImageNet
   validation shards that hold the chosen images (Hugging Face ``ILSVRC/imagenet-1k`` at the paper's revision: accept
   its terms on the dataset page and log in with ``hf auth login`` first);
2. takes the first ``n`` held-out test images in row order (default 500, none chosen by result), crops them, makes
   the noisy copy exactly as the paper did (official function, per-image seed, JPEG quality 85), and reads all three
   networks on the same clean and noisy images;
3. compares every image's correctness with the paper's record and reports the agreement.

The edited networks are the published ones, checked by hash. Rerunning the two programs needs 64 clean replay images
spread over all 14 validation shards, so this example does not rerun them; the ``scan`` example shows a rerun. With
``n`` = 38,000 it reads the whole test set (all 14 shards, 6.7 GB, many CPU hours).

Needs the ``train`` and ``t03`` extras (torchvision and Pillow for the crop, pyarrow for the shards, huggingface-hub for
the login).
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import load_params, params_sha256
from weightcode.data import imagenet_val as V
from weightcode.data import t04 as T
from weightcode.examples.fetch import FetchError, checked_files, file_sha256, hf_files
from weightcode.examples.replay import ReplayError
from weightcode.harness.jsonio import utc_now

NAME = "imagenet"
RERUNS_PROGRAM = False
TITLE = "ImageNet: 64 + 64 labeled images make a ResNet-18 robust to noise (Figure 1b, right)"
INSTANCE_ID = "i4c8c63bd"
CORRUPTION, SEVERITY = "gaussian_noise", 3
PROGRAMS = {
    "program 1": {"session_id": "sece3a52c03e092dd", "entry": "T04_gaussian_LEARN_r1",
                  "child_sha256": "3a3ae203fe15d82f865ec81887d17cbd5263bb24670d3eee7c513e926a1f319e"},
    "program 2": {"session_id": "sec0ee969740e55ce", "entry": "T04_gaussian_LEARN_r2",
                  "child_sha256": "54b9a2ed9342581b27ab95cb3333e3466e6d00627f2637b3a2163289bcd1f452"},
}
RESULTS = "records/sealed/results/2fe101710970d09e4a73ae8d9edf0fa698b7776011ced485d55ed167a0845235.jsonl"
"""The test read of record (finals list v6), one line per network with per-image correctness bits."""
TEST_ROWS_SHA256 = "471d25563a658afe65dd22b770a1c4358856d9820bd3b80e00705d606d164844"
DEFAULT_IMAGES = 500
BATCH = 64
SHARDS_SHA256 = {
    "validation-00000-of-00014.parquet": "5c5ebbf7dacc851d646cf343bf12412adfb79992ae4b4f245a6ed7bd935b7e96",
    "validation-00001-of-00014.parquet": "eb6ded350ef050296dec56969998b4d2d51075439afd18a661ed5bc77fc46110",
    "validation-00002-of-00014.parquet": "6c3d792bd864876e4617b26d2918178aba5fde869f95b2d2d330583467e5683d",
    "validation-00003-of-00014.parquet": "b7f9d8ee5a265171ac245dbaa755b5621134fca5bcf510ad7203c6bcd8c11b9c",
    "validation-00004-of-00014.parquet": "507f5443ad75165b5747cd453c13c91309a0141c277301865ee882c9aa2b95cf",
    "validation-00005-of-00014.parquet": "0670d7fd838a6486da995ad4f0e09e4b4f5b56a129939dd04b18da1e88692b19",
    "validation-00006-of-00014.parquet": "1ede790d2c28fa6f24a6b94f84b8f35e56285c39dd08de48eef04ab1adc3cac0",
    "validation-00007-of-00014.parquet": "2721f55f6bff82dac74643a1dfb59b184c82c245f30548ddc86d2d14f174a920",
    "validation-00008-of-00014.parquet": "5fc969226a2a246906a8ff919e5be09df5448ba542f28208c43fb3ec87e84168",
    "validation-00009-of-00014.parquet": "3488c9aee81cef37c21e51f2264b9ceeb9f178ed5dd1d3231daad8423ebc6690",
    "validation-00010-of-00014.parquet": "1bc05111b2dbf6cfe1005c209158faff4a660a462620867cfeb8681add012719",
    "validation-00011-of-00014.parquet": "ec035e5b27ed626ed1f1988e60d76f341255b955c8b6485894669a4620112496",
    "validation-00012-of-00014.parquet": "423890b715450706dade408a2e2cf0485d769b93511658a5692285e60b6a62f7",
    "validation-00013-of-00014.parquet": "edeb352ae89966575738f5d7474c1a14b3bc0c275f4f39ed4d2b537b56c1cedc",
}
"""sha256 of each validation shard at ``imagenet_val.REVISION``, from the manifest the paper's read checked
(``imagenet_val.MANIFEST_SHA256``)."""


def decode_bits(text: str, n: int) -> np.ndarray:
    """The per-image correctness that the record stores as base64 of ``numpy.packbits(..., bitorder='big')``."""
    raw = np.frombuffer(base64.b64decode(text.encode("ascii"), validate=True), dtype=np.uint8)
    if len(raw) != (n + 7) // 8:
        raise ReplayError(f"per-image bits hold {len(raw)} bytes, {n} images need {(n + 7) // 8}")
    return np.unpackbits(raw, bitorder="big")[:n].astype(bool)


def record_bits(results_path: Path) -> dict[str, dict[str, Any]]:
    """For each program: the record's numbers and per-image correctness of original and edited network."""
    entries = {spec["entry"]: name for name, spec in PROGRAMS.items()}
    found: dict[str, dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        name = entries.get(record.get("name"))
        if name is None:
            continue
        if record["child_sha256"] != PROGRAMS[name]["child_sha256"] or record["instance_id"] != INSTANCE_ID:
            raise ReplayError(f"{record['name']}: the record is not of the published network")
        if record["rows"]["condition"]["rows_sha256"] != TEST_ROWS_SHA256:
            raise ReplayError(f"{record['name']}: the record's test rows are not the paper's")
        bits = {}
        for part, key in (("noise", "per_image"), ("clean", "per_image_clean")):
            per = record[key]
            bits[part] = {"base": decode_bits(per["base_correct"], per["n"]),
                          "child": decode_bits(per["child_correct"], per["n"])}
        found[name] = {"bits": bits, "base_params_sha256": record["base_params_sha256"],
                       "noise": record["readouts"], "clean": record["readouts_clean"],
                       "joint": record["joint"]}
    if set(found) != set(PROGRAMS):
        raise ReplayError(f"{results_path} lacks {sorted(set(PROGRAMS) - set(found))}")
    return found


def imagenet_shard(name: str, root: Path) -> Path:
    """One validation shard, downloaded with the user's Hugging Face login and checked by sha256."""
    path = root / T.SHARD_DIR / name
    if path.is_file() and file_sha256(path) == SHARDS_SHA256[name]:
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise FetchError("the imagenet example needs huggingface-hub: uv sync --locked --extra train --extra t03") from error
    try:
        got = Path(hf_hub_download(V.REPO_ID, f"{T.SHARD_DIR}/{name}", repo_type="dataset", revision=V.REVISION,
                                   local_dir=str(root)))
    except Exception as error:  # gated dataset: name the two usual causes, keep the original error
        raise FetchError(f"could not download {V.REPO_ID}/{T.SHARD_DIR}/{name}: accept the dataset's terms on its "
                         f"Hugging Face page and run `hf auth login` ({type(error).__name__}: {error})") from error
    if file_sha256(got) != SHARDS_SHA256[name]:
        raise FetchError(f"{got}: sha256 is not the one the paper's read checked")
    return got


def read_rows(shard: Path, local_rows: np.ndarray) -> tuple[list[bytes], np.ndarray]:
    """Stored image bytes and labels of the given rows of one shard."""
    import pyarrow.parquet as pq

    picked = pq.ParquetFile(shard).read(columns=["image", "label"]).take([int(r) for r in local_rows])
    images, labels = picked.column("image").to_pylist(), picked.column("label").to_pylist()
    data = []
    for image, label in zip(images, labels, strict=True):
        raw = image["bytes"] if isinstance(image, dict) else None
        if not isinstance(raw, (bytes, bytearray)) or not isinstance(label, int) or not 0 <= label < T.N_CLASSES:
            raise ReplayError(f"{shard.name}: a row without image bytes or with label {label!r}")
        data.append(bytes(raw))
    return data, np.asarray(labels, dtype=np.int64)


def run(out: Path, runner: Any = None, *, images: int = DEFAULT_IMAGES, log: Callable[[str], None] = print
        ) -> dict[str, Any]:
    """Reproduce the example under ``out`` (module docstring) and return its summary, also written to result.json.
    ``runner`` is accepted for the CLI's common call and unused: this example reruns no program."""
    del runner
    rows_all = T.test_rows()
    if T.rows_sha256(rows_all) != TEST_ROWS_SHA256:
        raise ReplayError("this code's test rows are not the paper's")
    if not 1 <= images <= len(rows_all):
        raise ValueError(f"--images must be 1 .. {len(rows_all)}")
    rows = rows_all[:images]
    downloads = out / "downloads"
    log(f"[1/4] Original and edited networks and the paper's test record from Hugging Face -> {downloads / 'hf'}")
    finals = [f"weights/session_finals/{spec['session_id']}/model.npz" for spec in PROGRAMS.values()]
    hf = hf_files([f"weights/instances/{INSTANCE_ID}/", *finals, RESULTS], downloads / "hf")
    instance = load_instance(downloads / "hf" / "weights" / "instances" / INSTANCE_ID)
    record = record_bits(hf[RESULTS])
    networks = {"original": instance.params}
    for name, spec in PROGRAMS.items():
        if record[name]["base_params_sha256"] != instance.params_sha256:
            raise ReplayError(f"{name}: the record's original network is not instance {INSTANCE_ID}")
        params = load_params(hf[f"weights/session_finals/{spec['session_id']}/model.npz"])
        if params_sha256(params) != spec["child_sha256"]:
            raise ReplayError(f"{name}: the downloaded network is not the published child {spec['child_sha256']}")
        networks[name] = params
    official_file = checked_files({"make_imagenet_c.py": T.OFFICIAL_URL}, {"make_imagenet_c.py": T.OFFICIAL_SHA256},
                                  downloads / "imagenet_c")["make_imagenet_c.py"]
    official = T.load_official(official_file, T.OFFICIAL_SHA256)

    log(f"[2/4] ImageNet validation shards that hold the first {images:,} test images "
        f"({V.REPO_ID} at {V.REVISION[:8]})")
    names = V.shard_names()
    shards, counts = [], []
    while sum(counts) <= int(rows[-1]):
        name = names[len(shards)]
        shards.append(imagenet_shard(name, downloads / "imagenet"))
        counts.append(V.shard_row_counts(shards[-1].parent, [name])[0])

    log(f"[3/4] Cropping, adding Gaussian noise (severity {SEVERITY}) and reading 3 networks on {images:,} images")
    correct = {(net, part): [] for net in networks for part in ("noise", "clean")}
    labels: list[np.ndarray] = []
    offset = 0
    for shard, count in zip(shards, counts, strict=True):
        mine = rows[(rows >= offset) & (rows < offset + count)]
        if len(mine):
            data, y = read_rows(shard, mine - offset)
            labels.append(y)
            for start in range(0, len(mine), BATCH):
                crops = np.stack([T.center_crop_u8(raw) for raw in data[start:start + BATCH]])
                sets = {"clean": crops,
                        "noise": T.corrupt_split(official, CORRUPTION, SEVERITY, crops, mine[start:start + BATCH])}
                for net, params in networks.items():
                    for part, x in sets.items():
                        pred = np.asarray(instance.forward(dict(params), x)).argmax(axis=1)
                        correct[(net, part)].append(pred == y[start:start + BATCH])
                log(f"      {min(sum(len(c) for c in correct[('original', 'clean')]), images):,} / {images:,}")
        offset += count
    got = {key: np.concatenate(parts) for key, parts in correct.items()}

    log("[4/4] Comparing every image with the paper's record")
    expected: dict[tuple[str, str], np.ndarray] = {}
    for part in ("noise", "clean"):
        bases = [record[name]["bits"][part]["base"] for name in PROGRAMS]
        if any(not np.array_equal(bases[0], other) for other in bases[1:]):
            raise ReplayError(f"the record reads the original network differently for the two programs ({part})")
        expected[("original", part)] = bases[0][:images]
        for name in PROGRAMS:
            expected[(name, part)] = record[name]["bits"][part]["child"][:images]
    same = {f"{net} / {part}": int(np.count_nonzero(got[(net, part)] == bits)) for (net, part), bits in expected.items()}
    agree, total = sum(same.values()), images * len(expected)
    accuracy = {net: {part: float(np.mean(got[(net, part)])) for part in ("noise", "clean")} for net in networks}
    paper = {name: {"noise": [record[name]["noise"]["base_acc"], record[name]["noise"]["child_acc"]],
                    "clean": [record[name]["clean"]["base_acc"], record[name]["clean"]["child_acc"]],
                    "joint_points": record[name]["joint"]} for name in PROGRAMS}
    summary = {
        "example": NAME, "created_at": utc_now(), "instance_id": INSTANCE_ID, "images": images,
        "rows": "the first images of the paper's 38,000 held-out test rows, in row order",
        "condition": f"{CORRUPTION} severity {SEVERITY}", "networks": {k: v["child_sha256"] for k, v in PROGRAMS.items()},
        "accuracy": accuracy, "agreement": {"same": agree, "total": total, "by_network": same},
        "paper_all_38000": paper,
        "matches_paper": agree == total,
    }
    run_dir = out / "runs" / utc_now().replace(":", "").replace("-", "").replace("+0000", "Z")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report(summary, log)
    log(f"Record: {run_dir / 'result.json'}")
    return summary


def report(summary: dict[str, Any], log: Callable[[str], None] = print) -> None:
    """The result table, with the paper's numbers on all 38,000 images beside it."""
    acc, n = summary["accuracy"], summary["images"]
    log("")
    log(f"  {TITLE}")
    log(f"  Top-1 accuracy on the first {n:,} held-out test images")
    log(f"  {'':42s}{'original':>10s}{'program 1':>12s}{'program 2':>12s}")
    for part, label in (("noise", f"with Gaussian noise (severity {SEVERITY})"), ("clean", "clean")):
        log(f"    {label:40s}{100 * acc['original'][part]:>9.1f}%{100 * acc['program 1'][part]:>11.1f}%"
            f"{100 * acc['program 2'][part]:>11.1f}%")
    agreement = summary["agreement"]
    log(f"  Same correctness as the paper's record, image by image (3 networks x clean and noisy): "
        f"{agreement['same']:,} of {agreement['total']:,}")
    paper = summary["paper_all_38000"]
    p1, p2 = paper["program 1"], paper["program 2"]
    log(f"  Paper, all 38,000 test images: noise {100 * p1['noise'][0]:.2f}% -> {100 * p1['noise'][1]:.2f}% / "
        f"{100 * p2['noise'][1]:.2f}%, clean {100 * p1['clean'][0]:.2f}% -> {100 * p1['clean'][1]:.2f}% / "
        f"{100 * p2['clean'][1]:.2f}%, net gain {p1['joint_points']:+.2f} / {p2['joint_points']:+.2f} pt")
    if n < 38000:
        log(f"  ({n:,} images give a rougher estimate than the paper's 38,000; the image-by-image check shows they are "
            "read as the paper read them.)")


__all__ = ["DEFAULT_IMAGES", "INSTANCE_ID", "NAME", "PROGRAMS", "TITLE", "decode_bits", "record_bits", "report", "run"]
