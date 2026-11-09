"""Deployment conditions: fixed synthetic corruptions of a dataset's images (CIFAR-10-C at severity 3).

A condition keeps the rows of the clean splits (same order, ``y`` and ``idx``) and replaces ``x`` with a corrupted
copy. It covers the clean ``train`` split (for ceiling references), ``V_select``, ``V_evidence``, ``V_check`` and
the official test set, which goes only to the sealed store. Models evaluated under a condition run through the clean
dataset's forward.py, whose normalization constants the condition shares.

Formulas and constants are transcribed from the primary source (``SOURCE``: hendrycks/robustness,
ImageNet-C/create_c/make_cifar_c.py). For one image ``x`` (uint8, H x W x C in the source):

* ``gaussian_noise``, ``c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]``:
  ``clip(x / 255 + N(0, c^2), 0, 1) * 255``, one normal draw per value.
* ``contrast``, ``c = [0.75, 0.5, 0.4, 0.3, 0.15][severity - 1]``:
  ``m`` = per-channel mean of ``x / 255`` over H and W; ``clip((x / 255 - m) * c + m, 0, 1) * 255``.

The float result is stored with ``np.uint8``, which truncates toward zero, as the source does
(``np.uint8(corruption(...))``). Each image is processed as the source processes it: a contiguous ``(H, W, C)``
array (what ``np.array`` of a PIL image gives), divided by 255 in float64, the contrast mean taken with
``np.mean(axis=(0, 1))`` on that array. Truncation makes the stored value sensitive to the last bit of the float
result, so this module keeps the source's operation order rather than an algebraically equal one.

The only departure is the source of randomness: the source calls the global ``np.random.normal``; here noise comes
from one numpy Generator per stored split, ``default_rng([SPLIT_SEED, stream])`` with the streams of
``Condition.streams``, and each image in row order takes one ``(H, W, C)`` block of ``normal(0, c)`` draws (the
source's per-image shape), transposed to the stored ``(C, H, W)`` layout.

A second family, ``DrawnCondition`` (contract section 2.1, user decision 2026-11-04), holds the dev-only CIFAR-10
conditions outside CIFAR-10-C: ``colormix_d1``, ``blur5_d1`` and ``gamma_d1``. Each is a deterministic transform of
``x / 255`` whose parameters (``Q``, ``R`` or ``z``) are drawn once from ``default_rng([SPLIT_SEED, stream])`` with
one fixed stream per condition, and whose severity ``s`` is a recorded constant set by the train-split calibration
of ``weightcode.models.calibration``. Its log, ``calibration.json`` next to the condition's dataset.json, is provenance that
every build checks. The images are stored with the same rule as the CIFAR-10-C family.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from weightcode.common.paths import sealed_dir
from weightcode.data.build import DATASET_SCHEMA
from weightcode.data.registry import N_CLASSES, SPLIT_NAMES, SPLIT_SEED
from weightcode.data.store import (
    channel_stats,
    dataset_dir,
    dataset_json_path,
    load_split,
    read_dataset_json,
    read_json,
    save_split,
    sealed_test_path,
    split_path,
    split_sha256,
    write_json,
)

SOURCE: Mapping[str, str] = MappingProxyType(
    {
        "repository": "https://github.com/hendrycks/robustness",
        "file": "ImageNet-C/create_c/make_cifar_c.py",
        "commit": "ca360085d57889288c4143fddd8a707bbcae01bd",
        "url": (
            "https://raw.githubusercontent.com/hendrycks/robustness/ca360085d57889288c4143fddd8a707bbcae01bd/"
            "ImageNet-C/create_c/make_cifar_c.py"
        ),
        "file_sha256": "f865ae91dffb2b4834292ce05c876bf3d77f36417f70075a2b5259348b0c83a5",
        "functions": "gaussian_noise (line 130), contrast (line 333); images stored with np.uint8 (line 458)",
    }
)
CONSTANTS: Mapping[str, tuple[float, ...]] = MappingProxyType(
    {
        "gaussian_noise": (0.04, 0.06, 0.08, 0.09, 0.10),
        "contrast": (0.75, 0.5, 0.4, 0.3, 0.15),
    }
)
FORMULAS: Mapping[str, str] = MappingProxyType(
    {
        "gaussian_noise": "clip(x / 255 + normal(0, c), 0, 1) * 255, one draw per value",
        "contrast": "clip((x / 255 - m) * c + m, 0, 1) * 255, m = per-image per-channel mean of x / 255 over H, W",
    }
)
STORAGE_RULE = "np.uint8 of the float result in [0, 255]: truncation toward zero, as make_cifar_c.py stores images"
STREAM_OFFSETS: Mapping[str, int] = MappingProxyType(
    {"train": 1, "V_select": 3, "V_evidence": 4, "V_check": 5, "test": 6}
)
"""Per-split offsets added to a condition's ``stream_base`` (the Colored-MNIST offsets; 2 is unused here)."""
CONDITION_SPLITS: tuple[str, ...] = SPLIT_NAMES
CHUNK = 1000
"""Images corrupted per step; it bounds memory and does not change any value."""


@dataclass(frozen=True)
class Condition:
    """One corruption of one dataset at one severity."""

    name: str
    dataset: str
    corruption: str
    severity: int
    stream_base: int | None

    def __post_init__(self) -> None:
        if self.corruption not in CONSTANTS:
            raise ValueError(f"unknown corruption {self.corruption!r}")
        if not 1 <= self.severity <= len(CONSTANTS[self.corruption]):
            raise ValueError(f"severity must be in 1..{len(CONSTANTS[self.corruption])}")
        if (self.corruption == "gaussian_noise") != (self.stream_base is not None):
            raise ValueError("exactly the random corruptions need a stream base")

    @property
    def constant(self) -> float:
        """The severity's constant ``c`` from the source."""
        return CONSTANTS[self.corruption][self.severity - 1]

    @property
    def streams(self) -> dict[str, int] | None:
        """Generator stream of every stored split, or None for a deterministic corruption."""
        if self.stream_base is None:
            return None
        return {split: self.stream_base + offset for split, offset in STREAM_OFFSETS.items()}

    def rng(self, split: str) -> np.random.Generator | None:
        """The split's generator ``default_rng([SPLIT_SEED, stream])``, or None if no randomness is used."""
        streams = self.streams
        if streams is None:
            return None
        if split not in streams:
            raise KeyError(f"no stream for split {split!r}")
        return np.random.default_rng([SPLIT_SEED, streams[split]])


CORRUPTION_CONDITIONS: Mapping[str, Condition] = MappingProxyType(
    {
        "contrast_s3": Condition("contrast_s3", "cifar10", "contrast", 3, None),
        "gaussian_noise_s3": Condition("gaussian_noise_s3", "cifar10", "gaussian_noise", 3, 300),
    }
)
"""The CIFAR-10-C family."""


# ---------------------------------------------------------------------------------------------------------------
# Drawn-parameter conditions (contract section 2.1)

DRAWN_TRANSFORMS: tuple[str, ...] = ("colormix", "blur5", "gamma")
DRAWN_FORMULAS: Mapping[str, str] = MappingProxyType(
    {
        "colormix": (
            "each pixel's colour vector v of x / 255 becomes A v, A = (1 - s) I + s Q, Q a 3x3 row-stochastic matrix "
            "whose rows are drawn from Dirichlet(1, 1, 1); the rows of A are convex weights, so values stay in [0, 1] "
            "without clipping"
        ),
        "blur5": (
            "each channel of x / 255 is convolved with K = (1 - s) delta + s R, delta the 5x5 unit impulse, R a 5x5 "
            "kernel with entries drawn from Exponential(1) and normalised to sum 1; the same K for every channel, "
            "reflect padding (np.pad mode 'reflect') by 2 on each spatial side"
        ),
        "gamma": "channel c of x / 255 becomes (x / 255) ** gamma_c, gamma_c = exp(s z_c), z_c drawn from N(0, 1)",
    }
)
DRAWN_COMPUTATION: Mapping[str, str] = MappingProxyType(
    {
        "colormix": (
            "v = x / 255 in float64 (N, C, H, W); out[:, c] = A[c, 0] * v[:, 0] + A[c, 1] * v[:, 1] + A[c, 2] * v[:, 2], "
            "summed left to right; then np.uint8(out * 255)"
        ),
        "blur5": (
            "v = x / 255 in float64 (N, C, H, W), p = np.pad(v, ((0, 0), (0, 0), (2, 2), (2, 2)), mode='reflect'); "
            "out = sum over u = 0..4, w = 0..4 in row-major order of K[u, w] * p[:, :, 4 - u : 4 - u + H, "
            "4 - w : 4 - w + W] (true convolution: out[i, j] = sum K[u, w] v[i + 2 - u, j + 2 - w]), accumulated from "
            "0.0; then np.uint8(out * 255)"
        ),
        "gamma": "v = x / 255 in float64 (N, C, H, W); out[:, c] = np.power(v[:, c], gamma_c); then np.uint8(out * 255)",
    }
)
SEVERITY_GRIDS: Mapping[str, tuple[float, ...]] = MappingProxyType(
    {
        "colormix": tuple(k / 20 for k in range(1, 21)),
        "blur5": tuple(k / 20 for k in range(1, 21)),
        "gamma": tuple(k / 10 for k in range(1, 31)),
    }
)
"""Severity grids of the calibration rule: 0.05, 0.10, ..., 1.00 and 0.1, 0.2, ..., 3.0 (each value is k / 20 or
k / 10 in float64)."""
COLORMIX_MAX_COND = 10.0
"""A colormix severity is eligible only if the 2-norm condition number of A is at most this."""
BLUR_SIZE = 5
CALIBRATION_FILE = "calibration.json"
PARAMETER_DRAWS: Mapping[str, str] = MappingProxyType(
    {
        "colormix": "Q = rng.dirichlet([1.0, 1.0, 1.0], size=3) (row i of Q is row i of A's mixing weights)",
        "blur5": "E = rng.exponential(1.0, size=(5, 5)); R = E / E.sum()",
        "gamma": "z = rng.standard_normal(3) (z[c] for channel c)",
    }
)
PARAMETER_STREAMS: Mapping[str, int] = MappingProxyType({"colormix_d1": 901, "blur5_d1": 902, "gamma_d1": 903})
"""One parameter stream per drawn condition, ``default_rng([SPLIT_SEED, stream])``. Chosen to collide with no stream
in use under ``SPLIT_SEED``: Colored-MNIST colours 1..6 (``colored.COLOR_STREAMS``), gaussian_noise_s3 301 and
303..306 (base 300 plus ``STREAM_OFFSETS``), the ogbn-arxiv node permutation 700 (``arxiv.PERMUTATION_STREAM``) and its old holdouts 701
(``arxiv.OLD_HOLDOUT_STREAM``)."""


@dataclass(frozen=True)
class DrawnCondition:
    """A transform of x / 255 with parameters drawn once and a calibrated, recorded severity ``s``.

    ``severity`` is None until the calibration has run and its chosen value has been entered here. A build refuses
    an uncalibrated condition.
    """

    name: str
    dataset: str
    transform: str
    parameter_stream: int
    severity: float | None

    def __post_init__(self) -> None:
        if self.transform not in DRAWN_TRANSFORMS:
            raise ValueError(f"unknown transform {self.transform!r}")
        if self.severity is not None and self.severity not in SEVERITY_GRIDS[self.transform]:
            raise ValueError(f"{self.name}: severity {self.severity!r} is not on the {self.transform} grid")

    @property
    def grid(self) -> tuple[float, ...]:
        """The calibration grid of this condition's transform."""
        return SEVERITY_GRIDS[self.transform]

    def rng(self) -> np.random.Generator:
        """The parameter generator ``default_rng([SPLIT_SEED, parameter_stream])``."""
        return np.random.default_rng([SPLIT_SEED, self.parameter_stream])

    def parameters(self) -> dict[str, np.ndarray]:
        """The drawn parameters, from a fresh generator (the same values on every call)."""
        return draw_parameters(self.transform, self.rng())


DRAWN_CONDITIONS: Mapping[str, DrawnCondition] = MappingProxyType(
    {
        "colormix_d1": DrawnCondition("colormix_d1", "cifar10", "colormix", PARAMETER_STREAMS["colormix_d1"], 0.65),
        "blur5_d1": DrawnCondition("blur5_d1", "cifar10", "blur5", PARAMETER_STREAMS["blur5_d1"], 0.7),
        "gamma_d1": DrawnCondition("gamma_d1", "cifar10", "gamma", PARAMETER_STREAMS["gamma_d1"], 0.5),
    }
)
"""The dev-only family of contract section 2.1. Each severity is the value chosen by its calibration log."""

CONDITIONS: Mapping[str, Condition | DrawnCondition] = MappingProxyType({**CORRUPTION_CONDITIONS, **DRAWN_CONDITIONS})
"""Every registered deployment condition, both families."""


def get_condition(name: str) -> Condition | DrawnCondition:
    """Registered condition by name."""
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; known: {sorted(CONDITIONS)}")
    return CONDITIONS[name]


def draw_parameters(transform: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw a transform's parameters from ``rng`` as ``PARAMETER_DRAWS`` states."""
    if transform == "colormix":
        return {"Q": rng.dirichlet(np.ones(3), size=3)}
    if transform == "blur5":
        draws = rng.exponential(1.0, size=(BLUR_SIZE, BLUR_SIZE))
        return {"E": draws, "R": draws / draws.sum()}
    if transform == "gamma":
        return {"z": rng.standard_normal(3)}
    raise ValueError(f"unknown transform {transform!r}")


def colormix_matrix(q: np.ndarray, s: float) -> np.ndarray:
    """``A = (1 - s) I + s Q``."""
    return (1.0 - s) * np.eye(3) + s * np.asarray(q, dtype=np.float64)


def blur_kernel(r: np.ndarray, s: float) -> np.ndarray:
    """``K = (1 - s) delta + s R`` with ``delta`` the centred 5x5 unit impulse."""
    delta = np.zeros((BLUR_SIZE, BLUR_SIZE))
    delta[BLUR_SIZE // 2, BLUR_SIZE // 2] = 1.0
    return (1.0 - s) * delta + s * np.asarray(r, dtype=np.float64)


def gamma_exponents(z: np.ndarray, s: float) -> np.ndarray:
    """``gamma_c = exp(s z_c)``."""
    return np.exp(s * np.asarray(z, dtype=np.float64))


def drawn_operator(transform: str, params: Mapping[str, np.ndarray], s: float) -> np.ndarray:
    """The operator at severity ``s``: A (colormix), K (blur5) or the gammas (gamma)."""
    if transform == "colormix":
        return colormix_matrix(params["Q"], s)
    if transform == "blur5":
        return blur_kernel(params["R"], s)
    if transform == "gamma":
        return gamma_exponents(params["z"], s)
    raise ValueError(f"unknown transform {transform!r}")


def operator_diagnostics(
    transform: str, params: Mapping[str, np.ndarray], s: float, image_hw: tuple[int, int]
) -> dict[str, Any]:
    """The operator at ``s`` and its diagnostics. ``eligible`` applies the colormix ``cond_2(A) <= 10`` rule.

    blur5 reports ``min_abs_fft``: the minimum of ``|np.fft.fft2(K, s=image_hw)|``, the frequency response of K on the
    image's DFT grid (a circular-convolution proxy, since reflect padding makes the stored operator only approximately
    circular).
    """
    op = drawn_operator(transform, params, s)
    out: dict[str, Any] = {"severity": s}
    if transform == "colormix":
        cond = float(np.linalg.cond(op, 2))
        out.update(A=op.tolist(), cond_2=cond, eligible=bool(cond <= COLORMIX_MAX_COND))
    elif transform == "blur5":
        out.update(
            K_sum=float(op.sum()), min_abs_fft=float(np.abs(np.fft.fft2(op, s=image_hw)).min()), eligible=True
        )
    else:
        out.update(gammas=op.tolist(), eligible=True)
    return out


def _colormix_float(v: np.ndarray, a: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    for c in range(3):
        out[:, c] = a[c, 0] * v[:, 0] + a[c, 1] * v[:, 1] + a[c, 2] * v[:, 2]
    return out


def _blur_float(v: np.ndarray, k: np.ndarray) -> np.ndarray:
    half = BLUR_SIZE // 2
    height, width = v.shape[2], v.shape[3]
    padded = np.pad(v, ((0, 0), (0, 0), (half, half), (half, half)), mode="reflect")
    out = np.zeros_like(v)
    last = BLUR_SIZE - 1
    for u in range(BLUR_SIZE):
        for w in range(BLUR_SIZE):
            out = out + k[u, w] * padded[:, :, last - u : last - u + height, last - w : last - w + width]
    return out


def _gamma_float(v: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    for c in range(v.shape[1]):
        out[:, c] = np.power(v[:, c], gammas[c])
    return out


def transform_float(v: np.ndarray, transform: str, operator: np.ndarray) -> np.ndarray:
    """Apply a drawn transform to float64 images ``v = x / 255`` [N, C, H, W] (``DRAWN_COMPUTATION``)."""
    if v.dtype != np.float64 or v.ndim != 4:
        raise TypeError(f"v must be float64 [N, C, H, W], got {v.dtype} {v.shape}")
    if transform == "colormix":
        if v.shape[1] != 3 or operator.shape != (3, 3):
            raise ValueError("colormix needs 3 channels and a 3x3 matrix")
        return _colormix_float(v, operator)
    if transform == "blur5":
        if operator.shape != (BLUR_SIZE, BLUR_SIZE):
            raise ValueError(f"blur5 needs a {BLUR_SIZE}x{BLUR_SIZE} kernel")
        return _blur_float(v, operator)
    if transform == "gamma":
        if operator.shape != (v.shape[1],):
            raise ValueError("gamma needs one exponent per channel")
        return _gamma_float(v, operator)
    raise ValueError(f"unknown transform {transform!r}")


def apply_drawn(
    x: np.ndarray, transform: str, params: Mapping[str, np.ndarray], s: float, chunk: int = CHUNK
) -> np.ndarray:
    """Transformed copy of uint8 images [N, C, H, W] at severity ``s``, stored with ``STORAGE_RULE``.

    Raises if a float result falls outside [0, 255] after scaling (the transforms keep [0, 1] by construction, so
    this would be a defect, never something to clip). ``chunk`` bounds memory and changes no value.
    """
    if x.dtype != np.uint8 or x.ndim != 4:
        raise TypeError(f"x must be uint8 [N, C, H, W], got {x.dtype} {x.shape}")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    operator = drawn_operator(transform, params, s)
    out = np.empty_like(x)
    for start in range(0, x.shape[0], chunk):
        scaled = transform_float(x[start : start + chunk] / 255.0, transform, operator) * 255
        if not (np.all(scaled >= 0.0) and np.all(scaled < 256.0)):
            raise FloatingPointError(f"{transform} at s={s}: values outside [0, 256) before np.uint8 storage")
        out[start : start + chunk] = np.uint8(scaled)
    return out


def apply_drawn_split(arrays: Mapping[str, np.ndarray], condition: DrawnCondition) -> dict[str, np.ndarray]:
    """The drawn condition's version of one stored split at its recorded severity: same y and idx, new x."""
    if set(arrays) != {"x", "y", "idx"}:
        raise ValueError(f"expected keys x, y, idx; got {sorted(arrays)}")
    if condition.severity is None:
        raise ValueError(f"{condition.name} has no calibrated severity")
    return {
        "x": apply_drawn(arrays["x"], condition.transform, condition.parameters(), condition.severity),
        "y": np.asarray(arrays["y"]),
        "idx": np.asarray(arrays["idx"]),
    }


def calibration_path(condition: DrawnCondition) -> Path:
    """``$WEIGHTCODE_PRIVATE/<dataset>/conditions/<name>/calibration.json``, next to the condition's dataset.json."""
    return dataset_dir(condition.dataset, condition.name) / CALIBRATION_FILE


def _gaussian_noise_chunk(x: np.ndarray, c: float, rng: np.random.Generator) -> np.ndarray:
    n, channels, height, width = x.shape
    hwc = x.transpose(0, 2, 3, 1) / 255.0
    noise = rng.normal(loc=0.0, scale=c, size=(n, height, width, channels))
    out = np.clip(hwc + noise, 0, 1) * 255
    return out.astype(np.uint8).transpose(0, 3, 1, 2)


def _contrast_chunk(x: np.ndarray, c: float) -> np.ndarray:
    out = np.empty_like(x)
    for row in range(x.shape[0]):
        image = np.ascontiguousarray(x[row].transpose(1, 2, 0)) / 255.0
        means = np.mean(image, axis=(0, 1), keepdims=True)
        out[row] = np.uint8(np.clip((image - means) * c + means, 0, 1) * 255).transpose(2, 0, 1)
    return out


def corrupt(x: np.ndarray, condition: Condition, rng: np.random.Generator | None, chunk: int = CHUNK) -> np.ndarray:
    """Corrupted copy of uint8 images [N, C, H, W]; noise draws follow row order, so ``chunk`` changes nothing."""
    if x.dtype != np.uint8 or x.ndim != 4:
        raise TypeError(f"x must be uint8 [N, C, H, W], got {x.dtype} {x.shape}")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if (rng is None) != (condition.stream_base is None):
        raise ValueError(f"{condition.name}: a generator is needed exactly for random corruptions")
    out = np.empty_like(x)
    for start in range(0, x.shape[0], chunk):
        block = x[start : start + chunk]
        if condition.corruption == "gaussian_noise":
            assert rng is not None
            out[start : start + chunk] = _gaussian_noise_chunk(block, condition.constant, rng)
        else:
            out[start : start + chunk] = _contrast_chunk(block, condition.constant)
    return out


def corrupt_split(arrays: Mapping[str, np.ndarray], condition: Condition, split: str) -> dict[str, np.ndarray]:
    """The condition's version of one stored split: same y and idx, corrupted x, with the split's own stream."""
    if set(arrays) != {"x", "y", "idx"}:
        raise ValueError(f"expected keys x, y, idx; got {sorted(arrays)}")
    return {
        "x": corrupt(arrays["x"], condition, condition.rng(split)),
        "y": np.asarray(arrays["y"]),
        "idx": np.asarray(arrays["idx"]),
    }


@dataclass
class BuiltCondition:
    """Every stored array of one condition plus its manifest (without run provenance)."""

    condition: Condition | DrawnCondition
    splits: dict[str, dict[str, np.ndarray]]
    test: dict[str, np.ndarray]
    manifest: dict[str, Any]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_clean_inputs(dataset: str, raw_test: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """The clean manifest and splits, every split checked against its recorded hash.

    ``raw_test`` holds the official test arrays re-read from the raw source (x, y, idx); its hash must equal the
    clean dataset.json's test hash, so the sealed test file is never opened.
    """
    manifest = read_dataset_json(dataset)
    splits = {}
    for split in CONDITION_SPLITS:
        arrays = load_split(dataset, split)
        if split_sha256(arrays) != manifest["splits"][split]["sha256"]:
            raise RuntimeError(f"{dataset}/{split}.npz does not match its dataset.json hash")
        splits[split] = arrays
    if split_sha256(raw_test) != manifest["test"]["sha256"]:
        raise RuntimeError(f"{dataset}: raw official test arrays do not match the recorded test hash")
    return manifest, splits


def build_condition(
    name: str,
    clean_manifest: Mapping[str, Any],
    clean_splits: Mapping[str, Mapping[str, np.ndarray]],
    raw_test: Mapping[str, np.ndarray],
    clean_dataset_json_sha256: str,
) -> BuiltCondition:
    """Corrupt every clean split and the official test set and assemble the condition manifest."""
    condition = get_condition(name)
    if not isinstance(condition, Condition):
        raise TypeError(f"{name} is not a CIFAR-10-C condition; build it with build_drawn_condition")
    if clean_manifest["dataset"] != condition.dataset:
        raise ValueError(f"{name} is a condition of {condition.dataset}, not {clean_manifest['dataset']}")
    splits = {split: corrupt_split(clean_splits[split], condition, split) for split in CONDITION_SPLITS}
    test = corrupt_split(raw_test, condition, "test")
    rng_block = None
    if condition.streams is not None:
        rng_block = {
            "generator": "numpy.random.default_rng([split_seed, stream]) (PCG64), one per stored split",
            "split_seed": SPLIT_SEED,
            "streams": condition.streams,
            "draw": "normal(0, c) of shape (H, W, C) per image in row order, transposed to (C, H, W)",
            "numpy": np.__version__,
            "note": "numpy does not promise identical Generator streams across releases; verify with this version",
        }
    condition_block = {
        "name": condition.name,
        "kind": "corruption",
        "corruption": condition.corruption,
        "severity": condition.severity,
        "constant": condition.constant,
        "constants_by_severity": list(CONSTANTS[condition.corruption]),
        "formula": FORMULAS[condition.corruption],
        "storage": STORAGE_RULE,
        "computation": (
            "per image as in the source: contiguous (H, W, C) float64 array x / 255"
            + (", mean by np.mean(axis=(0, 1))" if condition.corruption == "contrast" else "")
            + ", then the formula and np.uint8; stored transposed to (C, H, W)"
        ),
        "source": dict(SOURCE),
        "rng": rng_block,
        "applied_to": (
            "rows of the clean splits train, V_select, V_evidence and V_check (same order, y and idx; only x "
            "changes) and the official test set (sealed)"
        ),
        "clean": _clean_block(clean_manifest, clean_dataset_json_sha256),
    }
    manifest = _condition_manifest(condition.dataset, condition.name, condition_block, clean_manifest, splits, test)
    return BuiltCondition(condition=condition, splits=splits, test=test, manifest=manifest)


def _clean_block(clean_manifest: Mapping[str, Any], clean_dataset_json_sha256: str) -> dict[str, Any]:
    return {
        "dataset_json_sha256": clean_dataset_json_sha256,
        "splits": {split: clean_manifest["splits"][split]["sha256"] for split in CONDITION_SPLITS},
        "test_sha256": clean_manifest["test"]["sha256"],
    }


def _condition_manifest(
    dataset: str,
    name: str,
    condition_block: Mapping[str, Any],
    clean_manifest: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, np.ndarray]],
    test: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """dataset.json of a condition (without run provenance). The fields after ``condition`` are shared by both
    families."""
    image_shape = list(clean_manifest["image_shape"])
    for split, arrays in (*splits.items(), ("test", test)):
        if list(arrays["x"].shape[1:]) != image_shape or arrays["x"].dtype != np.uint8:
            raise AssertionError(f"{split}: corrupted images have {arrays['x'].dtype} {arrays['x'].shape[1:]}")
    train_mean, train_std = channel_stats(splits["train"]["x"])
    return {
        "schema": DATASET_SCHEMA,
        "dataset": dataset,
        "condition": dict(condition_block),
        "source": clean_manifest["source"],
        "class_names": list(clean_manifest["class_names"]),
        "image_shape": image_shape,
        "dtype": "uint8",
        "layout": "NCHW",
        "norm": {
            **clean_manifest["norm"],
            "note": "copied from the clean dataset: models under this condition use its forward.py",
        },
        "corrupted_train_channel_stats": {
            "mean": train_mean,
            "std": train_std,
            "use": "information only; no model uses these constants",
        },
        "split_seed": clean_manifest["split_seed"],
        "split_sizes": {split: int(arrays["y"].size) for split, arrays in splits.items()},
        "splits": {
            split: {
                "path": f"splits/{split}.npz",
                "n": int(arrays["y"].size),
                "sha256": split_sha256(arrays),
                "class_counts": np.bincount(arrays["y"], minlength=N_CLASSES).tolist(),
                "clean_sha256": clean_manifest["splits"][split]["sha256"],
            }
            for split, arrays in splits.items()
        },
        "test": {
            "location": f"$WEIGHTCODE_SEALED/{dataset}/conditions/{name}/test.npz",
            "n": int(test["y"].size),
            "sha256": split_sha256(test),
            "clean_sha256": clean_manifest["test"]["sha256"],
        },
        "forward_path": "../../forward.py",
        "forward_sha256": clean_manifest["forward_sha256"],
    }


def check_calibration_log(
    condition: DrawnCondition, log: Mapping[str, Any], clean_manifest: Mapping[str, Any]
) -> None:
    """Refuse a calibration log that does not belong to this condition, its drawn parameters, its recorded severity
    or the clean train split."""
    params = {key: value.tolist() for key, value in condition.parameters().items()}
    checks = {
        "condition": (log.get("condition"), condition.name),
        "dataset": (log.get("dataset"), condition.dataset),
        "transform": (log.get("transform"), condition.transform),
        "split": (log.get("split"), "train"),
        "parameter_stream": (log.get("parameters", {}).get("stream"), condition.parameter_stream),
        "parameters": (log.get("parameters", {}).get("values"), params),
        "grid": (log.get("grid"), list(condition.grid)),
        "clean_train_sha256": (log.get("clean_train_sha256"), clean_manifest["splits"]["train"]["sha256"]),
        "chosen severity": (log.get("chosen", {}).get("severity"), condition.severity),
    }
    problems = [f"{key}: log {found!r} != {expected!r}" for key, (found, expected) in checks.items() if found != expected]
    if problems:
        raise ValueError(f"{condition.name}: calibration log disagrees with the registered condition: " + "; ".join(problems))


def build_drawn_condition(
    name: str,
    clean_manifest: Mapping[str, Any],
    clean_splits: Mapping[str, Mapping[str, np.ndarray]],
    raw_test: Mapping[str, np.ndarray],
    clean_dataset_json_sha256: str,
    calibration_log: Mapping[str, Any],
    calibration_sha256: str,
) -> BuiltCondition:
    """Transform every clean split and the official test set at the recorded severity and assemble the manifest.

    ``calibration_log`` is the condition's calibration.json (checked by ``check_calibration_log``) and
    ``calibration_sha256`` the sha256 of that file's bytes, recorded as provenance.
    """
    condition = get_condition(name)
    if not isinstance(condition, DrawnCondition):
        raise TypeError(f"{name} is not a drawn-parameter condition")
    if condition.severity is None:
        raise ValueError(f"{name} has no calibrated severity; run the calibration and record its chosen value")
    if clean_manifest["dataset"] != condition.dataset:
        raise ValueError(f"{name} is a condition of {condition.dataset}, not {clean_manifest['dataset']}")
    check_calibration_log(condition, calibration_log, clean_manifest)
    splits = {split: apply_drawn_split(clean_splits[split], condition) for split in CONDITION_SPLITS}
    test = apply_drawn_split(raw_test, condition)
    params = condition.parameters()
    image_hw = (int(clean_manifest["image_shape"][1]), int(clean_manifest["image_shape"][2]))
    chosen = calibration_log["chosen"]
    condition_block = {
        "name": condition.name,
        "kind": "drawn_transform",
        "transform": condition.transform,
        "severity": condition.severity,
        "formula": DRAWN_FORMULAS[condition.transform],
        "storage": STORAGE_RULE,
        "computation": DRAWN_COMPUTATION[condition.transform],
        "parameters": {
            "values": {key: value.tolist() for key, value in params.items()},
            "draw": PARAMETER_DRAWS[condition.transform],
            "generator": "numpy.random.default_rng([split_seed, stream]) (PCG64), drawn once for the condition",
            "split_seed": SPLIT_SEED,
            "stream": condition.parameter_stream,
            "numpy": np.__version__,
            "note": "numpy does not promise identical Generator streams across releases; verify with this version",
        },
        "operator_at_severity": operator_diagnostics(condition.transform, params, condition.severity, image_hw),
        "severity_rule": {
            "grid": list(condition.grid),
            "rule": calibration_log["rule"],
            "colormix_max_cond_2": COLORMIX_MAX_COND if condition.transform == "colormix" else None,
        },
        "calibration": {
            "path": CALIBRATION_FILE,
            "sha256": calibration_sha256,
            "target_drop": calibration_log["target_drop"]["value"],
            "mean_drop_at_severity": chosen["mean_drop"],
            "status": chosen["status"],
        },
        "rule_source": "docs/CONTRACT_v0_DRAFT.md section 2.1 (user decision 2026-11-04)",
        "applied_to": (
            "rows of the clean splits train, V_select, V_evidence and V_check (same order, y and idx; only x "
            "changes) and the official test set (sealed)"
        ),
        "clean": _clean_block(clean_manifest, clean_dataset_json_sha256),
    }
    manifest = _condition_manifest(condition.dataset, condition.name, condition_block, clean_manifest, splits, test)
    return BuiltCondition(condition=condition, splits=splits, test=test, manifest=manifest)


def build_from_store(name: str, raw_test: Mapping[str, np.ndarray]) -> BuiltCondition:
    """Load and check the clean inputs of a registered condition (and a drawn condition's calibration log) and
    build it."""
    condition = get_condition(name)
    clean_manifest, clean_splits = load_clean_inputs(condition.dataset, raw_test)
    clean_json_sha = _file_sha256(dataset_json_path(condition.dataset))
    if isinstance(condition, DrawnCondition):
        path = calibration_path(condition)
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; {name} must be calibrated before it is built")
        log = read_json(path)
        return build_drawn_condition(
            name, clean_manifest, clean_splits, raw_test, clean_json_sha, log, _file_sha256(path)
        )
    return build_condition(name, clean_manifest, clean_splits, raw_test, clean_json_sha)


def _make_sealed_parent(dataset: str, name: str) -> None:
    """Create ``$WEIGHTCODE_SEALED/<dataset>/conditions/<name>/`` with mode 0700 on every new level."""
    path = sealed_dir()
    for part in (dataset, "conditions", name):
        path = path / part
        path.mkdir(mode=0o700, exist_ok=True)
        path.chmod(0o700)


def write_condition(built: BuiltCondition, run_info: Mapping[str, Any]) -> dict[str, Any]:
    """Write the private splits, the sealed test and dataset.json; refuses an existing condition."""
    dataset, name = built.condition.dataset, built.condition.name
    manifest_path = dataset_json_path(dataset, name)
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; use verify instead of rebuilding")
    for split, arrays in built.splits.items():
        if save_split(split_path(dataset, split, name), arrays) != built.manifest["splits"][split]["sha256"]:
            raise RuntimeError(f"{name}/{split}: written hash differs from the built hash")
    _make_sealed_parent(dataset, name)
    if save_split(sealed_test_path(dataset, name), built.test, private_mode=True) != built.manifest["test"]["sha256"]:
        raise RuntimeError(f"{name}/test: written hash differs from the built hash")
    manifest = {**built.manifest, "created_by": dict(run_info)}
    write_json(manifest_path, manifest)
    return manifest


def verify_condition(built: BuiltCondition) -> dict[str, Any]:
    """Compare a fresh build with the condition's dataset.json and private files (the sealed file is not read)."""
    dataset, name = built.condition.dataset, built.condition.name
    recorded = read_dataset_json(dataset, name)
    report: dict[str, Any] = {}
    for split, arrays in built.splits.items():
        fresh = split_sha256(arrays)
        on_disk = split_sha256(load_split(dataset, split, name))
        expected = recorded["splits"][split]["sha256"]
        report[split] = {"fresh": fresh, "recorded": expected, "on_disk": on_disk, "match": fresh == expected == on_disk}
    fresh_test = split_sha256(built.test)
    report["test"] = {"fresh": fresh_test, "recorded": recorded["test"]["sha256"], "match": fresh_test == recorded["test"]["sha256"]}
    fields = {key: value for key, value in recorded.items() if key != "created_by"}
    report["manifest"] = {"match": fields == built.manifest}
    return report
