"""PREREG T04 data store: dataset ``in1k_t04``, native ImageNet installation on ImageNet-C conditions (dev side only).

Source: the ImageNet-1k validation split as PREREG T03 reads it (``weightcode.data.imagenet_val``: Hugging Face
ILSVRC/imagenet-1k at revision ``REVISION``, 14 parquet shards, rows numbered 0 .. 49,999 in shard order, then row
order). The rows are split by row number alone, fixed before any image is read:

* dev: rows with ``row % ROW_MODULUS in DEV_RESIDUES`` (12,000 rows). They hold T03's D (``row % 50 == 0``) and are
  the only rows any T04 job decodes before the single sealed read. ``split_rows`` deals them, from their labels, to
  V_evidence (2,000), V_check (4,000), V_select (4,000) and train (2,000; the clean pool of the replay rows and of the
  large-sample fine-tuning diagnostic) so that V_select and V_check hold every class (the dev readout's per-class
  deltas need each class present); each list is sorted.
* test (sealed): every other row (38,000). ``test_rows`` names them; nothing here reads them, and ``read_dev_records``
  refuses any of them before a shard is opened.

Images: the stored uint8 crop is ImageNet-C's input and the first two steps of torchvision's eval transform: RGB,
bilinear resize of the shorter side to 256 (antialias, PIL), center crop 224 (``center_crop_u8``). The network's
forward.py does x / 255 and the ImageNet normalisation (``weightcode.models.resnet18native``, arch ``resnet18_native_u8``).

Conditions: the official ImageNet-C functions ``gaussian_noise`` and ``contrast`` of ``make_imagenet_c.py``
(hendrycks/robustness at ``OFFICIAL_COMMIT``, sha256 ``OFFICIAL_SHA256``), taken from the file by ``ast`` after the
sha256 check and run as written (``load_official``; the rest of the file, which imports wand, cv2 and skimage and
loops over a dataset, is not executed). Each corrupted image follows the official pipeline after the crop: the
function on the PIL crop, ``np.uint8`` of its float output (truncation, as the official save), a JPEG save with
quality 85 and optimize=True (the official ``save``), and the decoded JPEG (``corrupt_crop``). gaussian_noise draws
from numpy's global RandomState, seeded per image with ``GAUSSIAN_SEED_BASE + row`` (the official code seeds nothing;
a per-row seed makes every image reproducible, independent of order, and paired across severities). contrast draws
nothing. A condition is ``<corruption>_s<severity>``; its severity is chosen on D by ``choose_severity`` (PREREG T04).

Class names (``class_names``): from classes.py, read with ``ast``. Each class gets its first synonym. A first synonym
shared by several classes is replaced by the class's full classes.py entry (all synonyms). A name holding a forbidden
token (the leak tokens an editor-visible file may not contain) is replaced by the class's first synonym free of them
and unused by another class, else by its WordNet id. The forbidden tokens are the caller's (``forbidden_name_tokens``).

pyarrow, PIL and torchvision are imported inside the functions that need them; the row rules and the class-name rule
run with numpy alone.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import io
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.data import imagenet_val as V

DATASET = "in1k_t04"
"""The store's dataset id. Its ``_``-separated parts ("in1k", "t04") become leak tokens
(``weightcode.evalkit.instances.leak_tokens``); no editor-visible description of the network or its data spells them."""
ROW_MODULUS = 25
DEV_RESIDUES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
N_DEV = V.N_VAL_ROWS // ROW_MODULUS * len(DEV_RESIDUES)
N_TEST = V.N_VAL_ROWS - N_DEV
SPLIT_SEED = 20261107
SPLIT_SIZES: tuple[tuple[str, int], ...] = (("V_evidence", 2000), ("V_check", 4000), ("V_select", 4000),
                                            ("train", 2000))
"""Dealing order and sizes of the dev rows (PREREG T04)."""
COVERED_SPLITS: tuple[str, ...] = ("V_select", "V_check")
"""Splits that hold every class (stage 1 of ``split_rows``), in the order a class's first rows go to them."""
SPLITS: tuple[str, ...] = ("train", "V_select", "V_evidence", "V_check")
CONDITION_SPLITS: tuple[str, ...] = ("V_evidence", "V_check", "V_select")
N_CLASSES = 1000
CHANCE = 1.0 / N_CLASSES

RESIZE_SHORTER = 256
CROP = 224
IMAGE_SHAPE: tuple[int, int, int] = (3, CROP, CROP)

OFFICIAL_REPOSITORY = "https://github.com/hendrycks/robustness"
OFFICIAL_FILE = "ImageNet-C/create_c/make_imagenet_c.py"
OFFICIAL_COMMIT = "8190fe329f5f072a06e3a2aea02bb5dda69aed9f"
"""The commit PREREG E10 pinned for the CIFAR-10-C file; the ImageNet-C file at the same commit."""
OFFICIAL_SHA256 = "3868a670617f6b931ad09d5613bf372962a47d17c2de4f15ccaec28feb9fa479"
OFFICIAL_URL = f"https://raw.githubusercontent.com/hendrycks/robustness/{OFFICIAL_COMMIT}/{OFFICIAL_FILE}"
CORRUPTIONS: tuple[str, ...] = ("gaussian_noise", "contrast")
SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)
CONSTANTS: dict[str, tuple[float, ...]] = {
    "gaussian_noise": (0.08, 0.12, 0.18, 0.26, 0.38),
    "contrast": (0.4, 0.3, 0.2, 0.1, 0.05),
}
"""The official constants by severity, restated for the records; ``check_official_constants`` holds them equal to the
file's behaviour."""
FAMILIES = {"gaussian_noise": "noise", "contrast": "digital"}
BENCHMARK_WORDS: tuple[str, ...] = ("imagenet-c", "imagenetc", "imagenet_c", "hendrycks", "dietterich", "severity",
                                    "corrupt")
"""The benchmark's names: leak words of a T04 session (``weightcode.harness.t04.leak_words``) and forbidden in class names."""
GAUSSIAN_SEED_BASE = 2_026_110_700
"""gaussian_noise seeds numpy's global RandomState with ``GAUSSIAN_SEED_BASE + row`` before each image (row = the
validation row number, 0 .. 49,999; the largest seed, 2,026,160,699, is below 2**32)."""
JPEG_QUALITY = 85
JPEG_OPTIMIZE = True

Q_MIN = 0.169
"""Severity rule (PREREG T04): the base must keep at least this share of its above-chance accuracy on D. 0.169 is E10's
lower value, gaussian_noise_s5 on CIFAR-10 with the seed-100 ResNet-20 ((23.775 - 10) / (91.275 - 10) = 0.1695,
runs/devreadouts/e10_headroom_s100_dev_20261104.json), the hardest condition on which B was met so far."""
MIN_DROP_POINTS = 15.0
"""E10's headroom rule: clean minus condition accuracy of at least 15 points."""
GAIN_SHARE = 0.15
"""PREREG T04's gain bar is GAIN_SHARE of the base's V_select headroom, rounded to the nearest 0.5 point: E10's +10
was 0.148 and 0.150 of its gaussian and contrast headroom (67.50 and 66.62 points)."""


class T04DataError(RuntimeError):
    """A T04 input does not match its record, a row outside the dev side was asked for, or a store would be
    overwritten."""


# --------------------------------------------------------------------------------------------------------------------
# Rows


def is_dev_row(row: int) -> bool:
    """Whether validation row ``row`` is on the dev side."""
    row = int(row)
    if not 0 <= row < V.N_VAL_ROWS:
        raise ValueError(f"row {row} outside 0 .. {V.N_VAL_ROWS - 1}")
    return row % ROW_MODULUS in DEV_RESIDUES


def dev_rows() -> np.ndarray:
    """The 12,000 dev rows, sorted (int64)."""
    rows = np.arange(V.N_VAL_ROWS, dtype=np.int64)
    return rows[np.isin(rows % ROW_MODULUS, DEV_RESIDUES)]


def test_rows() -> np.ndarray:
    """The 38,000 sealed test rows, sorted (int64). Only their numbers: nothing here reads them."""
    rows = np.arange(V.N_VAL_ROWS, dtype=np.int64)
    return rows[~np.isin(rows % ROW_MODULUS, DEV_RESIDUES)]


SPLIT_RULE = (f"one permutation of the sorted dev rows by numpy default_rng({SPLIT_SEED}); walking it, the first row "
              f"of each class goes to V_select and its second to V_check (stage 1, {N_CLASSES} rows each); the other "
              f"rows, in the same order, are dealt to V_evidence 2000, V_check 3000, V_select 3000, train 2000 "
              f"(stage 2); each list sorted")


def split_rows(labels: Mapping[int, int]) -> dict[str, np.ndarray]:
    """{split: sorted dev rows} by ``SPLIT_RULE``, from the label of every dev row (``labels[row]``).

    Stage 1 puts one row of every class into V_select and one into V_check, so both hold all ``N_CLASSES`` classes;
    stage 2 deals the remaining rows at random to the ``SPLIT_SIZES`` totals. ``T04DataError`` if a dev row has no
    label, a label is outside 0 .. N_CLASSES - 1, or a class has fewer than ``len(COVERED_SPLITS)`` dev rows.
    """
    rows = dev_rows()
    if sum(size for _, size in SPLIT_SIZES) != len(rows):
        raise RuntimeError("the split sizes do not add up to the dev rows")
    missing = [int(r) for r in rows if int(r) not in labels]
    if missing:
        raise T04DataError(f"{len(missing)} dev rows have no label (first {missing[:5]})")
    y = np.array([int(labels[int(r)]) for r in rows], dtype=np.int64)
    if y.min() < 0 or y.max() >= N_CLASSES:
        raise T04DataError(f"labels outside 0 .. {N_CLASSES - 1}")
    counts = np.bincount(y, minlength=N_CLASSES)
    short = np.flatnonzero(counts < len(COVERED_SPLITS))
    if short.size:
        raise T04DataError(f"classes {short[:10].tolist()} have fewer than {len(COVERED_SPLITS)} dev rows; the "
                           f"split cannot cover them")
    order = np.random.default_rng(SPLIT_SEED).permutation(len(rows))
    seen = np.zeros(N_CLASSES, dtype=np.int64)
    out: dict[str, list[int]] = {split: [] for split, _ in SPLIT_SIZES}
    rest: list[int] = []
    for position in order:
        label = int(y[position])
        if seen[label] < len(COVERED_SPLITS):
            out[COVERED_SPLITS[seen[label]]].append(int(rows[position]))
        else:
            rest.append(int(rows[position]))
        seen[label] += 1
    start = 0
    for split, size in SPLIT_SIZES:
        take = size - len(out[split])
        out[split].extend(rest[start:start + take])
        start += take
    if start != len(rest):
        raise RuntimeError("stage 2 did not deal every remaining row")
    return {split: np.sort(np.array(out[split], dtype=np.int64)) for split, _ in SPLIT_SIZES}


def rows_sha256(rows: Sequence[int] | np.ndarray) -> str:
    """sha256 of a row list as little-endian int64 bytes, in the given order."""
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def row_record(labels: Mapping[int, int] | None = None) -> dict[str, Any]:
    """The row rule with the sha256 of the dev and test row lists (written into PREREG T04 before any image is read);
    with the dev rows' labels, also the four split lists (recorded by the build)."""
    d_rows = np.arange(0, V.N_VAL_ROWS, V.D_STRIDE, dtype=np.int64)
    record: dict[str, Any] = {
        "n_val_rows": V.N_VAL_ROWS,
        "dev_rule": f"row % {ROW_MODULUS} in {list(DEV_RESIDUES)}",
        "test_rule": f"row % {ROW_MODULUS} not in {list(DEV_RESIDUES)}",
        "split_rule": SPLIT_RULE,
        "dev": {"n": int(len(dev_rows())), "sha256": rows_sha256(dev_rows())},
        "test": {"n": int(len(test_rows())), "sha256": rows_sha256(test_rows())},
        "d_in_dev": bool(np.all(np.isin(d_rows % ROW_MODULUS, DEV_RESIDUES))),
    }
    if labels is not None:
        splits = split_rows(labels)
        record["splits"] = {split: {"n": int(len(rows)), "sha256": rows_sha256(rows)} for split, rows in splits.items()}
        record["d_by_split"] = {split: int(np.isin(rows, d_rows).sum()) for split, rows in splits.items()}
    return record


def check_dev_rows(rows: Iterable[int]) -> list[int]:
    """The rows as ints; ``T04DataError`` if any is not a dev row (before anything is read)."""
    rows = [int(row) for row in rows]
    outside = [row for row in rows if not is_dev_row(row)]
    if outside:
        raise T04DataError(f"{len(outside)} rows are not dev rows (first {outside[:5]}); T04 decodes dev rows only")
    return rows


def locate(rows: Iterable[int], row_counts: Sequence[int]) -> list[V.DRow]:
    """The shard and local row of each validation row (``imagenet_val.DRow``), for shards of ``row_counts`` rows."""
    counts = [int(count) for count in row_counts]
    if sum(counts) != V.N_VAL_ROWS:
        raise ValueError(f"the shards hold {sum(counts)} rows, expected {V.N_VAL_ROWS}")
    starts = np.concatenate([[0], np.cumsum(counts)])
    names = V.shard_names(len(counts))
    located = []
    for row in rows:
        shard = int(np.searchsorted(starts, int(row), side="right") - 1)
        located.append(V.DRow(int(row), names[shard], shard, int(row) - int(starts[shard])))
    return located


SHARD_DIR = "data"
"""The shards' directory under the snapshot root (``<root>/data/validation-*.parquet``; the root holds classes.py and
manifest.sha256), as T03's scripts/t03/run_t03.py reads them."""


def read_dev_records(data_dir: Path, rows: Iterable[int]) -> Iterator[V.DRecord]:
    """Stored bytes and labels of dev ``rows`` from the shards in ``data_dir`` (the snapshot's ``SHARD_DIR``; refuses
    any other row before a shard is opened). A shard's other rows pass through memory as compressed column data when
    the shard is read; none is decoded, hashed or kept."""
    wanted = check_dev_rows(rows)
    located = locate(wanted, V.shard_row_counts(data_dir, V.shard_names()))
    yield from V.read_d_records(data_dir, located)


# --------------------------------------------------------------------------------------------------------------------
# Class names


def forbidden_name_tokens(extra: Iterable[str] = ()) -> list[str]:
    """Tokens a class name may not contain: the dataset's leak tokens, every T04 condition's leak tokens (all
    severities), ``BENCHMARK_WORDS``, the harness's arm and hint words, the registered condition names and ``extra``.
    These hold every leak word of a T04 session (``weightcode.harness.t04.leak_words``; a test holds the inclusion)."""
    from weightcode.data.conditions import CONDITIONS as REGISTERED
    from weightcode.evalkit.instances import condition_leak_tokens, leak_tokens
    from weightcode.harness.leaks import ARM_WORDS, HINT_WORDS

    tokens = set(leak_tokens(DATASET)) | set(ARM_WORDS) | set(HINT_WORDS) | set(REGISTERED) | set(BENCHMARK_WORDS)
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            tokens |= set(condition_leak_tokens(condition_name(corruption, severity),
                                                {"condition": {"corruption": corruption}}))
    tokens |= {str(token).lower() for token in extra}
    return sorted(tokens)


def class_names(classes: Sequence[tuple[str, str]], forbidden: Iterable[str]) -> tuple[list[str], list[dict[str, Any]]]:
    """(names in label order, the replacements made) by the module docstring's rule. ``classes`` is classes.py's
    (WordNet id, synonyms) list, which must be in sorted WordNet-id order (the label order)."""
    from weightcode.evalkit.instances import scan_for_tokens

    forbidden = sorted({str(token).lower() for token in forbidden})
    wnids = [wnid for wnid, _ in classes]
    if len(classes) != N_CLASSES or wnids != sorted(wnids) or len(set(wnids)) != N_CLASSES:
        raise T04DataError("classes.py must list 1000 distinct WordNet ids in sorted order (the label order)")
    synonyms = [[part.strip() for part in text.split(",") if part.strip()] for _, text in classes]
    if any(not parts for parts in synonyms):
        raise T04DataError("a class has no name in classes.py")
    first = Counter(parts[0] for parts in synonyms)
    names = [parts[0] if first[parts[0]] == 1 else ", ".join(parts) for parts in synonyms]
    changes = [{"label": i, "wnid": wnids[i], "from": synonyms[i][0], "to": names[i], "reason": "shared first synonym"}
               for i in range(N_CLASSES) if names[i] != synonyms[i][0]]
    for i in range(N_CLASSES):
        hits = scan_for_tokens(names[i], forbidden)
        if not hits:
            continue
        taken = set(names[:i] + names[i + 1:])
        clean = [s for s in synonyms[i] if not scan_for_tokens(s, forbidden) and s not in taken]
        new = clean[0] if clean else wnids[i]
        changes.append({"label": i, "wnid": wnids[i], "from": names[i], "to": new,
                        "reason": f"holds forbidden tokens {hits}"})
        names[i] = new
    if len(set(names)) != N_CLASSES:
        raise T04DataError("class names are not unique after the rule")
    left = {name: scan_for_tokens(name, forbidden) for name in names if scan_for_tokens(name, forbidden)}
    if left:
        raise T04DataError(f"class names still hold forbidden tokens: {left}")
    return names, changes


# --------------------------------------------------------------------------------------------------------------------
# Images


def center_crop_u8(image_bytes: bytes) -> np.ndarray:
    """Decode stored image bytes, convert to RGB, resize the shorter side to 256 (bilinear, antialias) and center-crop
    224: uint8 [3, 224, 224]. These are the first steps of torchvision's eval transform for the IMAGENET1K_V1 weights and
    ImageNet-C's ``trn.Resize(256), trn.CenterCrop(224)``, both on a PIL image."""
    import torchvision.transforms.functional as F
    from PIL import Image
    from torchvision.transforms import InterpolationMode

    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = image.convert("RGB")
    crop = F.center_crop(F.resize(rgb, [RESIZE_SHORTER], interpolation=InterpolationMode.BILINEAR, antialias=True),
                         [CROP, CROP])
    array = np.asarray(crop, dtype=np.uint8)
    if array.shape != (CROP, CROP, 3):
        raise ValueError(f"crop has shape {array.shape}")
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def load_official(path: Path, expected_sha256: str = OFFICIAL_SHA256) -> dict[str, Callable[..., Any]]:
    """The official ``gaussian_noise`` and ``contrast`` from ``path`` (sha256 checked), compiled from the file's own
    function definitions with ``np`` as their only global. Returns {name: function, "_sources": {name: sha256}}."""
    path = Path(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise T04DataError(f"{path}: sha256 differs from {expected_sha256}")
    text = data.decode("utf-8")
    tree = ast.parse(text)
    found = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in CORRUPTIONS}
    missing = [name for name in CORRUPTIONS if name not in found]
    if missing:
        raise T04DataError(f"{path} does not define {missing}")
    allowed = set(dir(builtins)) | {"np"}
    for name, node in found.items():
        free = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        local = {a.arg for a in node.args.args} | {n.id for n in ast.walk(node)
                                                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if free - local - allowed:
            raise T04DataError(f"{name} reads globals other than np: {sorted(free - local - allowed)}")
    module = ast.Module(body=[found[name] for name in CORRUPTIONS], type_ignores=[])
    namespace: dict[str, Any] = {"np": np, "__name__": "make_imagenet_c_functions"}
    exec(compile(module, str(path), "exec"), namespace)
    sources = {name: hashlib.sha256(ast.get_source_segment(text, found[name]).encode()).hexdigest()
               for name in CORRUPTIONS}
    return {**{name: namespace[name] for name in CORRUPTIONS}, "_sources": sources}


def gaussian_seed(row: int) -> int:
    """The global RandomState seed of row ``row``'s gaussian_noise draw."""
    return GAUSSIAN_SEED_BASE + int(row)


def corrupt_crop(official: Mapping[str, Any], corruption: str, severity: int, crop: np.ndarray, row: int) -> np.ndarray:
    """One corrupted image: the official function on the PIL crop, np.uint8 of its output, a JPEG save (quality 85,
    optimize) and its decode: uint8 [3, 224, 224]."""
    from PIL import Image

    if corruption not in CORRUPTIONS or int(severity) not in SEVERITIES:
        raise ValueError(f"unknown condition {corruption} s{severity}")
    if crop.dtype != np.uint8 or crop.shape != IMAGE_SHAPE:
        raise ValueError(f"crop must be uint8 {IMAGE_SHAPE}, got {crop.dtype}{crop.shape}")
    pil = Image.fromarray(np.ascontiguousarray(crop.transpose(1, 2, 0)), mode="RGB")
    if corruption == "gaussian_noise":
        np.random.seed(gaussian_seed(row))
    out = np.uint8(official[corruption](pil, int(severity)))
    buffer = io.BytesIO()
    Image.fromarray(out).save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=JPEG_OPTIMIZE)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        array = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def corrupt_split(official: Mapping[str, Any], corruption: str, severity: int, x: np.ndarray,
                  rows: np.ndarray) -> np.ndarray:
    """``corrupt_crop`` of every image of ``x`` (uint8 [N, 3, 224, 224]) with its validation row number."""
    if len(x) != len(rows):
        raise ValueError("one row number per image")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = corrupt_crop(official, corruption, severity, x[i], int(rows[i]))
    return out


def check_official_constants(official: Mapping[str, Any]) -> dict[str, Any]:
    """The official functions on fixed inputs against ``CONSTANTS``: contrast exactly, gaussian_noise by the scale of
    its unclipped noise. Raises on a mismatch; returns the record."""
    from PIL import Image

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(64, 192, size=(64, 64, 3), dtype=np.uint8), mode="RGB")
    x = np.asarray(image, dtype=np.float64) / 255.0
    record: dict[str, Any] = {}
    for severity in SEVERITIES:
        c = CONSTANTS["contrast"][severity - 1]
        means = x.mean(axis=(0, 1), keepdims=True)
        expected = np.clip((x - means) * c + means, 0, 1) * 255
        got = official["contrast"](image, severity)
        if not np.array_equal(np.asarray(got), expected):
            raise T04DataError(f"official contrast s{severity} differs from c = {c}")
        np.random.seed(12345)
        noisy = np.asarray(official["gaussian_noise"](image, severity)) / 255.0 - x
        np.random.seed(12345)
        expected_noise = np.random.normal(size=x.shape, scale=CONSTANTS["gaussian_noise"][severity - 1])
        if not np.allclose(noisy, np.clip(x + expected_noise, 0, 1) - x, atol=1e-12):
            raise T04DataError(f"official gaussian_noise s{severity} differs from its constant")
        record[f"s{severity}"] = {"contrast": c, "gaussian_noise": CONSTANTS["gaussian_noise"][severity - 1]}
    return record


def image_libraries() -> dict[str, Any]:
    """Versions that decide the bytes of a corrupted image: numpy, Pillow and its JPEG library, torchvision."""
    import PIL
    import torchvision
    from PIL import features

    return {"numpy": np.__version__, "Pillow": PIL.__version__, "jpeg": features.version("jpg"),
            "libjpeg_turbo": features.version("libjpeg_turbo"), "torchvision": torchvision.__version__}


# --------------------------------------------------------------------------------------------------------------------
# Conditions and the severity rule


def condition_name(corruption: str, severity: int) -> str:
    """``<corruption>_s<severity>``."""
    if corruption not in CORRUPTIONS or int(severity) not in SEVERITIES:
        raise ValueError(f"unknown T04 condition {corruption} s{severity}")
    return f"{corruption}_s{int(severity)}"


def retained_share(acc_condition: float, acc_clean: float, chance: float = CHANCE) -> float:
    """q = (a_c - chance) / (a_0 - chance): the share of the above-chance accuracy the base keeps."""
    if acc_clean <= chance:
        raise ValueError("clean accuracy at or below chance")
    return (float(acc_condition) - chance) / (float(acc_clean) - chance)


def choose_severity(acc_clean: float, acc_by_severity: Mapping[int, float], q_min: float = Q_MIN,
                    min_drop_points: float = MIN_DROP_POINTS) -> dict[str, Any]:
    """PREREG T04's severity rule on D: the highest severity with q >= ``q_min`` and a drop of at least
    ``min_drop_points``; None when no severity qualifies (the condition is then not run)."""
    rows = {}
    for severity in SEVERITIES:
        acc = float(acc_by_severity[severity])
        q = retained_share(acc, acc_clean)
        drop = 100.0 * (float(acc_clean) - acc)
        rows[severity] = {"acc": acc, "q": q, "drop_points": drop,
                          "qualifies": bool(q >= q_min and drop >= min_drop_points)}
    qualifying = [s for s in SEVERITIES if rows[s]["qualifies"]]
    return {"rule": f"highest severity with q >= {q_min} and drop >= {min_drop_points} points on D",
            "acc_clean": float(acc_clean), "by_severity": {f"s{s}": rows[s] for s in SEVERITIES},
            "chosen": max(qualifying) if qualifying else None}


def gain_bar(headroom_points: float, share: float = GAIN_SHARE) -> float:
    """PREREG T04's gain bar: ``share`` of the V_select headroom, rounded to the nearest 0.5 point (halves up)."""
    return float(np.floor(2.0 * share * float(headroom_points) + 0.5) / 2.0)


__all__ = [
    "BENCHMARK_WORDS", "CHANCE", "CONDITION_SPLITS", "CONSTANTS", "CORRUPTIONS", "COVERED_SPLITS", "CROP", "DATASET",
    "DEV_RESIDUES", "FAMILIES", "GAIN_SHARE", "GAUSSIAN_SEED_BASE", "IMAGE_SHAPE", "JPEG_OPTIMIZE", "JPEG_QUALITY", "MIN_DROP_POINTS", "N_DEV",
    "N_TEST", "OFFICIAL_COMMIT", "OFFICIAL_FILE", "OFFICIAL_REPOSITORY", "OFFICIAL_SHA256", "OFFICIAL_URL", "Q_MIN",
    "RESIZE_SHORTER", "ROW_MODULUS", "SEVERITIES", "SHARD_DIR", "SPLITS", "SPLIT_RULE", "SPLIT_SEED", "SPLIT_SIZES",
    "T04DataError",
    "center_crop_u8", "check_dev_rows", "check_official_constants", "choose_severity", "class_names", "condition_name",
    "corrupt_crop", "corrupt_split", "dev_rows", "forbidden_name_tokens", "gain_bar", "gaussian_seed",
    "image_libraries", "is_dev_row", "load_official", "locate", "read_dev_records", "retained_share", "row_record",
    "rows_sha256", "split_rows", "test_rows",
]
