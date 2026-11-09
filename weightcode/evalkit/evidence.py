"""Visible evidence of one repair instance (contract section 3, PREREG section 6).

Two rules share one draw procedure and differ only in the pool of failure rows and in the seed text.

``evidence-v1`` (user decision 2026-11-03; an instance is one dataset, deployment condition and model):

- Seed: the first 8 bytes, read big-endian as an unsigned integer, of sha256(f"{params_sha256}|{condition}|
  evidence-v1") in UTF-8, passed to ``numpy.random.default_rng``.
- Failure pool: every V_evidence row of the condition that the base model gets wrong (base prediction != y),
  whatever the confused pair. The rule needs at least k_fail = 16 such rows.

``evidence-v0`` (target-pair instances, superseded by v1 and kept for its existing instances and tests):

- Seed: as above from sha256(f"{params_sha256}|{a}|{b}|evidence-v0").
- Failure pool: the V_evidence rows with y = a and base prediction b.

Common draw:

- x_fail, y_fail, pred_fail: k_fail = 16 rows drawn uniformly without replacement from the failure pool
  (``Generator.choice`` over the ascending row ids of the pool).
- x_rand, y_rand, pred_rand: k_rand = 48 rows drawn uniformly without replacement from all other V_evidence rows
  (ascending), correct or not. Pool rows not drawn for x_fail can appear here; no row appears twice.
- split_ids: one opaque id per row, fail rows first, then random rows, drawn from the same generator after the row
  draws ("e" + 12 hex digits). The ids carry no row information; the mapping to V_evidence rows is kept only in the
  instance's private.json.

pred_fail and pred_rand are the base model's predictions. Random draws are mandatory: evidence made only of the
model's own failures is a selected sample (contract section 3). NumPy does not promise identical ``Generator``
streams across releases, so the instance records the NumPy version.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from weightcode.common.identity import params_sha256
from weightcode.evalkit.targets import K_FAIL

EVIDENCE_RULE = "evidence-v0"
EVIDENCE_RULE_V1 = "evidence-v1"
K_RAND = 48
EVIDENCE_KEYS = ("x_fail", "y_fail", "pred_fail", "x_rand", "y_rand", "pred_rand", "split_ids")
SPLIT_ID_HEX_DIGITS = 12
CONDITION_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _seed_from_text(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _check_sha(params_sha256_hex: str) -> None:
    if not _SHA_RE.fullmatch(params_sha256_hex):
        raise ValueError(f"not a sha256 hex digest: {params_sha256_hex!r}")


def evidence_seed(params_sha256_hex: str, a: int, b: int) -> int:
    """Seed of the evidence-v0 draw for one (model, target pair)."""
    _check_sha(params_sha256_hex)
    return _seed_from_text(f"{params_sha256_hex}|{int(a)}|{int(b)}|{EVIDENCE_RULE}")


def evidence_seed_v1(params_sha256_hex: str, condition: str) -> int:
    """Seed of the evidence-v1 draw for one (model, deployment condition)."""
    _check_sha(params_sha256_hex)
    if not CONDITION_RE.fullmatch(condition):
        raise ValueError(f"invalid condition name {condition!r}")
    return _seed_from_text(f"{params_sha256_hex}|{condition}|{EVIDENCE_RULE_V1}")


@dataclass(frozen=True)
class Evidence:
    """The evidence arrays plus the private mapping of each image to its V_evidence row."""

    arrays: dict[str, np.ndarray]
    row_ids: np.ndarray
    seed: int

    @property
    def split_ids(self) -> list[str]:
        """Opaque ids in file order (fail images, then random images)."""
        return [str(value) for value in self.arrays["split_ids"]]


def _check_inputs(x: np.ndarray, y: np.ndarray, base_pred: np.ndarray) -> None:
    if x.dtype != np.uint8 or x.ndim != 4:
        raise ValueError(f"x must be uint8 [N, C, H, W], got {x.dtype}{x.shape}")
    if y.ndim != 1 or base_pred.shape != y.shape or len(x) != len(y):
        raise ValueError(f"shape mismatch: x {x.shape}, y {y.shape}, base_pred {base_pred.shape}")


def _draw_from_pool(
    x: np.ndarray,
    y: np.ndarray,
    base_pred: np.ndarray,
    pool: np.ndarray,
    rng: np.random.Generator,
    seed: int,
    k_fail: int,
    k_rand: int,
) -> Evidence:
    """The common draw: k_fail rows from ``pool``, k_rand from the other rows, then the opaque ids."""
    fail_rows = rng.choice(pool, size=k_fail, replace=False)
    rest = np.setdiff1d(np.arange(len(y)), fail_rows)
    if rest.size < k_rand:
        raise ValueError(f"only {rest.size} V_evidence images left for k_rand = {k_rand}")
    rand_rows = rng.choice(rest, size=k_rand, replace=False)
    codes = rng.integers(0, 16**SPLIT_ID_HEX_DIGITS, size=k_fail + k_rand, dtype=np.int64)
    if np.unique(codes).size != codes.size:
        raise RuntimeError(f"split id collision for seed {seed}; the id width must grow (rule change)")
    split_ids = np.array([f"e{int(code):0{SPLIT_ID_HEX_DIGITS}x}" for code in codes])
    arrays = {
        "x_fail": np.ascontiguousarray(x[fail_rows]),
        "y_fail": y[fail_rows].astype(np.int64),
        "pred_fail": base_pred[fail_rows].astype(np.int64),
        "x_rand": np.ascontiguousarray(x[rand_rows]),
        "y_rand": y[rand_rows].astype(np.int64),
        "pred_rand": base_pred[rand_rows].astype(np.int64),
        "split_ids": split_ids,
    }
    row_ids = np.concatenate([fail_rows, rand_rows]).astype(np.int64)
    return Evidence(arrays=arrays, row_ids=row_ids, seed=int(seed))


def draw_evidence(
    x: np.ndarray,
    y: np.ndarray,
    base_pred: np.ndarray,
    *,
    a: int,
    b: int,
    seed: int,
    k_fail: int = K_FAIL,
    k_rand: int = K_RAND,
) -> Evidence:
    """Rule evidence-v0: failure rows from the (a, b) cell of V_evidence."""
    y = np.asarray(y)
    base_pred = np.asarray(base_pred)
    _check_inputs(x, y, base_pred)
    if a == b:
        raise ValueError("target pair must have a != b")
    rng = np.random.default_rng(seed)
    cell = np.flatnonzero((y == a) & (base_pred == b))
    if cell.size < k_fail:
        raise ValueError(f"only {cell.size} V_evidence images with y = {a} predicted {b}; k_fail is {k_fail}")
    return _draw_from_pool(x, y, base_pred, cell, rng, seed, k_fail, k_rand)


def draw_evidence_v1(
    x: np.ndarray,
    y: np.ndarray,
    base_pred: np.ndarray,
    *,
    seed: int,
    k_fail: int = K_FAIL,
    k_rand: int = K_RAND,
) -> Evidence:
    """Rule evidence-v1: failure rows from all base errors of V_evidence (base prediction != y)."""
    y = np.asarray(y)
    base_pred = np.asarray(base_pred)
    _check_inputs(x, y, base_pred)
    rng = np.random.default_rng(seed)
    errors = np.flatnonzero(base_pred != y)
    if errors.size < k_fail:
        raise ValueError(f"only {errors.size} V_evidence images are base errors; evidence-v1 needs k_fail = {k_fail}")
    return _draw_from_pool(x, y, base_pred, errors, rng, seed, k_fail, k_rand)


def evidence_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Content hash of evidence arrays (the canonical ``params_sha256`` scheme applied to them).

    The npz file hash is not stable across writes (zip timestamps), so instances compare this instead.
    """
    if set(arrays) != set(EVIDENCE_KEYS):
        raise ValueError(f"evidence keys {sorted(arrays)} != {sorted(EVIDENCE_KEYS)}")
    return params_sha256(arrays)


def load_evidence(path: str | Path) -> dict[str, np.ndarray]:
    """Load an evidence.npz without pickle."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_evidence(path: str | Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Save evidence as an uncompressed npz, reload it and return its verified content hash."""
    expected = evidence_content_sha256(arrays)
    np.savez(Path(path), **{key: np.ascontiguousarray(arrays[key]) for key in EVIDENCE_KEYS})
    reloaded = evidence_content_sha256(load_evidence(path))
    if reloaded != expected:
        raise RuntimeError(f"{path}: evidence content hash changed on reload ({reloaded} != {expected})")
    return expected
