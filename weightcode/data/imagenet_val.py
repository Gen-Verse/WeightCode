"""The ImageNet-1k validation split as PREREG T03 reads it, and its dev subset D.

Source: the Hugging Face dataset ILSVRC/imagenet-1k at revision ``REVISION``, 14 parquet shards
``validation-00000-of-00014.parquet`` .. ``validation-00013-of-00014.parquet`` (columns ``image``, a struct of the
stored file ``bytes`` and its ``path``, and ``label``) and ``classes.py``, with a ``manifest.sha256`` written at
download (NSCC R29 job 1b). Rows are numbered 0 .. 49,999 in shard order, then row order. D is the rows whose number
is a multiple of ``D_STRIDE`` (1,000 rows); T03 decodes no other row. Each D image is identified by the sha256 of its
stored bytes.

Labels: the dataset's ``label`` is taken to be the index of torchvision's class order (the sorted WordNet ids, as in
``classes.py``). PREREG T03's pass item (4), the base's top-1 accuracy on D, checks that assumption; nothing here is
tuned to pass it. ``classes_wnid_order`` reports, as a description only, whether classes.py lists its WordNet ids in
sorted order.

pyarrow, PIL and torchvision are imported inside the functions that need them, so the index rule and the manifest
check run with numpy alone.
"""

from __future__ import annotations

import ast
import hashlib
import io
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ID = "ILSVRC/imagenet-1k"
REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
N_SHARDS = 14
N_VAL_ROWS = 50_000
D_STRIDE = 50
N_D = N_VAL_ROWS // D_STRIDE
MANIFEST = "manifest.sha256"
MANIFEST_SHA256 = "1d5fd98c4d16c799820a578fc00b696f3e4f35c3e2624fc450395cd3e39195fc"
"""sha256 of manifest.sha256 as NSCC_REQUESTS R29 job 1b records it (prefix 1d5fd98c4d16c799)."""
CLASSES_FILE = "classes.py"
CLASSES_VARIABLE = "IMAGENET2012_CLASSES"
D_INDEX_SCHEMA = "weightcode.t03.d_index.v1"


def shard_names(n_shards: int = N_SHARDS) -> list[str]:
    """The parquet file names of the validation split, in shard order."""
    return [f"validation-{index:05d}-of-{n_shards:05d}.parquet" for index in range(n_shards)]


def in_d(global_row: int, stride: int = D_STRIDE) -> bool:
    """Whether validation row ``global_row`` (0-based, shard order then row order) belongs to D."""
    return int(global_row) % int(stride) == 0


@dataclass(frozen=True)
class DRow:
    """One row of D: its global number, its shard (file name and index) and its row within the shard."""

    global_index: int
    shard: str
    shard_index: int
    row: int


def d_rows(row_counts: Sequence[int], names: Sequence[str] | None = None, stride: int = D_STRIDE,
           expected_rows: int = N_VAL_ROWS) -> list[DRow]:
    """D's rows for shards with ``row_counts`` rows each (in shard order): every global row number that is a multiple
    of ``stride``. Refuses a total other than ``expected_rows``."""
    counts = [int(count) for count in row_counts]
    names = shard_names(len(counts)) if names is None else [str(name) for name in names]
    if len(names) != len(counts) or any(count < 0 for count in counts):
        raise ValueError(f"{len(names)} shard names for {len(counts)} row counts, or a negative count")
    if sum(counts) != expected_rows:
        raise ValueError(f"the shards hold {sum(counts)} rows, expected {expected_rows}")
    rows, offset = [], 0
    for index, (name, count) in enumerate(zip(names, counts)):
        first = -offset % stride
        rows.extend(DRow(offset + local, name, index, local) for local in range(first, count, stride))
        offset += count
    if [row.global_index for row in rows] != list(range(0, expected_rows, stride)):
        raise RuntimeError("D's rows are not the multiples of the stride")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path, manifest_sha256: str | None = MANIFEST_SHA256) -> dict[str, str]:
    """Check ``root/manifest.sha256`` (its own sha256 when ``manifest_sha256`` is given) and every file it lists;
    return {relative path: sha256}. Raises on any mismatch or missing file."""
    root = Path(root)
    manifest = root / MANIFEST
    if manifest_sha256 is not None and file_sha256(manifest) != manifest_sha256:
        raise ValueError(f"{manifest}: sha256 {file_sha256(manifest)}, expected {manifest_sha256}")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition("  ")
        if len(digest) != 64 or not relative or relative in listed:
            raise ValueError(f"{manifest}: malformed or repeated line {line!r}")
        listed[relative] = digest
    for relative, digest in listed.items():
        actual = file_sha256(root / relative)
        if actual != digest:
            raise ValueError(f"{root / relative}: sha256 {actual}, manifest says {digest}")
    return listed


def parse_classes(path: Path) -> list[tuple[str, str]]:
    """(WordNet id, names) in the order classes.py lists them, read with ``ast`` (the file is never executed)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == CLASSES_VARIABLE
                                                for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call) and len(value.args) == 1 and not value.keywords:
                value = value.args[0]
            if not isinstance(value, ast.Dict):
                raise ValueError(f"{path}: {CLASSES_VARIABLE} is not a dict literal")
            pairs = [(ast.literal_eval(key), ast.literal_eval(item)) for key, item in zip(value.keys, value.values)]
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in pairs):
                raise ValueError(f"{path}: {CLASSES_VARIABLE} holds non-string entries")
            return pairs
    raise ValueError(f"{path}: no assignment to {CLASSES_VARIABLE}")


def classes_wnid_order(classes: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Description of classes.py's order: count, distinct ids, and whether the WordNet ids are sorted (torchvision's
    class order is the sorted ids)."""
    wnids = [wnid for wnid, _ in classes]
    return {"n_classes": len(wnids), "n_distinct": len(set(wnids)), "sorted": wnids == sorted(wnids),
            "first": wnids[0] if wnids else None, "last": wnids[-1] if wnids else None}


def shard_row_counts(data_dir: Path, names: Sequence[str]) -> list[int]:
    """Row counts of the shards from their parquet metadata (nothing is decoded)."""
    import pyarrow.parquet as pq

    return [int(pq.ParquetFile(Path(data_dir) / name).metadata.num_rows) for name in names]


@dataclass(frozen=True)
class DRecord:
    """A D row with its stored image: ``image_sha256`` is the sha256 of ``image_bytes``."""

    global_index: int
    shard: str
    row: int
    label: int
    image_sha256: str
    image_path: str
    image_bytes: bytes

    def index_entry(self) -> dict[str, Any]:
        """The D index row (everything but the bytes)."""
        entry = asdict(self)
        del entry["image_bytes"]
        return entry


def read_d_records(data_dir: Path, rows: Sequence[DRow]) -> Iterator[DRecord]:
    """The stored bytes and labels of exactly ``rows``, shard by shard (pyarrow). Only these rows' bytes are taken
    from the shard tables; no image is decoded here."""
    import pyarrow.parquet as pq

    by_shard: dict[str, list[DRow]] = {}
    for row in rows:
        by_shard.setdefault(row.shard, []).append(row)
    for shard, shard_rows in by_shard.items():
        table = pq.ParquetFile(Path(data_dir) / shard).read(columns=["image", "label"])
        picked = table.take([row.row for row in shard_rows])
        images = picked.column("image").to_pylist()
        labels = picked.column("label").to_pylist()
        for row, image, label in zip(shard_rows, images, labels, strict=True):
            data = image["bytes"]
            if not isinstance(data, (bytes, bytearray)) or not data:
                raise ValueError(f"{shard} row {row.row}: no stored image bytes")
            if not isinstance(label, int) or not 0 <= label < 1000:
                raise ValueError(f"{shard} row {row.row}: label {label!r} outside 0 .. 999")
            yield DRecord(row.global_index, shard, row.row, int(label), hashlib.sha256(data).hexdigest(),
                          str(image.get("path") or ""), bytes(data))
        del table, picked


def eval_transform() -> Callable[[Any], Any]:
    """torchvision's eval transform of the IMAGENET1K_V1 ResNet-18 weights."""
    from torchvision.models import ResNet18_Weights

    return ResNet18_Weights.IMAGENET1K_V1.transforms()


def preprocess(image_bytes: bytes, transform: Callable[[Any], Any] | None = None) -> np.ndarray:
    """Decode stored image bytes, convert to RGB and apply ``eval_transform``: float32 [3, 224, 224]."""
    from PIL import Image

    transform = eval_transform() if transform is None else transform
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
    tensor = transform(rgb)
    array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.shape != (3, 224, 224):
        raise ValueError(f"the eval transform gave shape {array.shape}, expected (3, 224, 224)")
    return array


def write_d_index(path: Path, entries: Iterable[dict[str, Any]], meta: dict[str, Any]) -> str:
    """Write D's index (one JSON object per line after a header line) and return the file's sha256."""
    import json

    lines = [json.dumps({"schema": D_INDEX_SCHEMA, **meta}, sort_keys=True)]
    lines += [json.dumps(entry, sort_keys=True) for entry in entries]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "CLASSES_FILE", "DRecord", "DRow", "D_INDEX_SCHEMA", "D_STRIDE", "MANIFEST", "MANIFEST_SHA256", "N_D", "N_SHARDS",
    "N_VAL_ROWS", "REPO_ID", "REVISION", "classes_wnid_order", "d_rows", "eval_transform", "file_sha256", "in_d",
    "parse_classes", "preprocess", "read_d_records", "shard_names", "shard_row_counts", "verify_manifest",
    "write_d_index",
]
