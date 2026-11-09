"""Colored-MNIST, the known-cause subset (contract section 2, brief item 2).

Labels are the clean digit. Class ``c`` owns palette color ``c``. An image is colored by scaling the gray digit
per channel: channel k = round(gray * palette[color][k] / 255), computed exactly in integers (an exact .5 cannot
occur because 255 is odd).

Color draws use one numpy generator per stored split, ``default_rng([split_seed, COLOR_STREAMS[split]])``:

* ``train`` (biased): for each image draw ``u ~ U[0, 1)`` and ``shift ~ U{1..9}`` (both always drawn, so the draws
  do not depend on rho). The color is the label's own color if ``u >= 1 - rho``, otherwise ``(label + shift) % 10``,
  uniform over the other 9 colors. Because the draws are shared, the conflicting images at rho = 0.999 are a
  subset of those at rho = 0.99.
* ``train_uniform``, ``V_select``, ``V_evidence``, ``V_check``, ``test`` (deployment distribution): the color is
  uniform over all 10 colors, independent of the label.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),  # 0 red
    (0, 255, 0),  # 1 green
    (0, 0, 255),  # 2 blue
    (255, 255, 0),  # 3 yellow
    (255, 0, 255),  # 4 magenta
    (0, 255, 255),  # 5 cyan
    (255, 128, 0),  # 6 orange
    (128, 0, 255),  # 7 violet
    (0, 255, 128),  # 8 spring green
    (255, 255, 255),  # 9 white
)
PALETTE_NAMES: tuple[str, ...] = (
    "red",
    "green",
    "blue",
    "yellow",
    "magenta",
    "cyan",
    "orange",
    "violet",
    "spring green",
    "white",
)
N_COLORS = len(PALETTE)

RHO_PRIMARY = 0.99
"""Probability that a biased train image carries its label's own color (attempt 1)."""
RHO_FALLBACK = 0.999
"""Pre-registered single rebuild if the shortcut check fails on V_select (attempt 2)."""
SHORTCUT_GAP_POINTS = 15.0
"""Rebuild once if mean ceiling minus mean base V_select accuracy is at most this many percentage points."""

COLOR_STREAMS: Mapping[str, int] = MappingProxyType(
    {"train": 1, "train_uniform": 2, "V_select": 3, "V_evidence": 4, "V_check": 5, "test": 6}
)
UNIFORM_SPLITS: tuple[str, ...] = ("train_uniform", "V_select", "V_evidence", "V_check", "test")

_CHUNK = 8192


def color_rng(split_seed: int, split: str) -> np.random.Generator:
    """Generator for the color draws of one stored split."""
    if split not in COLOR_STREAMS:
        raise KeyError(f"no color stream for split {split!r}; known: {sorted(COLOR_STREAMS)}")
    return np.random.default_rng([int(split_seed), COLOR_STREAMS[split]])


def biased_colors(y: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Color index per image: the label's color with probability ``rho``, else uniform over the other 9."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1], got {rho}")
    labels = np.asarray(y, dtype=np.int64)
    if labels.size and (labels.min() < 0 or labels.max() >= N_COLORS):
        raise ValueError("labels must lie in [0, 10)")
    u = rng.random(labels.size)
    shift = rng.integers(1, N_COLORS, size=labels.size)
    conflict = u < (1.0 - rho)
    return np.where(conflict, (labels + shift) % N_COLORS, labels).astype(np.int64)


def uniform_colors(n: int, rng: np.random.Generator) -> np.ndarray:
    """Color index per image, uniform over all colors and independent of the label."""
    return rng.integers(0, N_COLORS, size=int(n)).astype(np.int64)


def colorize(gray: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Return uint8 [N, 3, H, W] images with channel k = round(gray * PALETTE[color][k] / 255).

    ``gray`` is uint8 [N, H, W] or [N, 1, H, W]; ``color`` holds one palette index per image.
    """
    images = np.asarray(gray)
    if images.dtype != np.uint8:
        raise TypeError(f"gray images must be uint8, got {images.dtype}")
    if images.ndim == 4:
        if images.shape[1] != 1:
            raise ValueError(f"expected one gray channel, got shape {images.shape}")
        images = images[:, 0]
    if images.ndim != 3:
        raise ValueError(f"gray images must be [N, H, W] or [N, 1, H, W], got {images.shape}")
    colors = np.asarray(color)
    if colors.shape != (images.shape[0],) or not np.issubdtype(colors.dtype, np.integer):
        raise ValueError(f"color must be an integer array of shape ({images.shape[0]},)")
    if colors.size and (colors.min() < 0 or colors.max() >= N_COLORS):
        raise ValueError("color indices must lie in [0, 10)")
    out = np.empty((images.shape[0], 3, *images.shape[1:]), dtype=np.uint8)
    table = color_table()
    for start in range(0, images.shape[0], _CHUNK):
        stop = start + _CHUNK
        for channel in range(3):
            out[start:stop, channel] = table[colors[start:stop, None, None], channel, images[start:stop]]
    return out


def color_table() -> np.ndarray:
    """uint8 lookup table [color, channel, gray] = round(gray * PALETTE[color][channel] / 255), exact integers."""
    gray = np.arange(256, dtype=np.int64)
    palette = np.asarray(PALETTE, dtype=np.int64)
    return ((2 * gray[None, None, :] * palette[:, :, None] + 255) // 510).astype(np.uint8)


def shortcut_decision(base_acc: list[float], ceiling_acc: list[float]) -> dict[str, float | bool]:
    """Pre-registered check on V_select: rebuild if mean ceiling minus mean base accuracy <= 15 points.

    Accuracies are fractions in [0, 1]; the gap is reported in percentage points.
    """
    if not base_acc or not ceiling_acc:
        raise ValueError("need base and ceiling accuracies")
    base_mean = float(np.mean(base_acc))
    ceiling_mean = float(np.mean(ceiling_acc))
    gap = 100.0 * (ceiling_mean - base_mean)
    return {"base_mean": base_mean, "ceiling_mean": ceiling_mean, "gap_points": gap, "triggered": gap <= SHORTCUT_GAP_POINTS}
