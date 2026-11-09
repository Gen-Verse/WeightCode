"""Seeded stratified split of an official train set into train / V_select / V_evidence / V_check.

Rule (frozen with split seed 20261103):
1. Every non-train split gets per-class quotas proportional to the class counts of the full official train set,
   by the largest-remainder (Hamilton) method in exact integer arithmetic; remainder ties go to the lower class
   index. Train gets every remaining index of each class.
2. For each class in increasing order, the indices of that class are permuted once with
   ``numpy.random.default_rng(seed)`` (one generator shared across classes, consumed in class order) and dealt to
   the splits in ``ALLOCATION_ORDER``.
3. Each split's indices are returned sorted in increasing order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from weightcode.data.registry import ALLOCATION_ORDER


def largest_remainder_quota(class_counts: np.ndarray, total: int) -> np.ndarray:
    """Per-class integer counts summing to ``total`` and proportional to ``class_counts`` (Hamilton method)."""
    counts = np.asarray(class_counts, dtype=np.int64)
    if counts.ndim != 1 or np.any(counts < 0):
        raise ValueError("class_counts must be a 1-D array of non-negative integers")
    n = int(counts.sum())
    if not 0 <= total <= n:
        raise ValueError(f"total {total} outside [0, {n}]")
    scaled = counts * int(total)
    quota = scaled // n
    remainder = scaled % n
    shortfall = int(total - quota.sum())
    order = np.lexsort((np.arange(counts.size), -remainder))
    quota[order[:shortfall]] += 1
    if int(quota.sum()) != total or np.any(quota > counts):
        raise AssertionError("largest-remainder allocation is inconsistent")
    return quota


def stratified_split(
    y: np.ndarray,
    sizes: Mapping[str, int],
    seed: int | Sequence[int],
    n_classes: int,
    order: Sequence[str] = ALLOCATION_ORDER,
) -> dict[str, np.ndarray]:
    """Partition ``range(len(y))`` into the named splits (see module docstring for the rule).

    ``order`` is the dealing order and its last split takes the remainder of each class (``train`` by default;
    ogbn-arxiv deals its 2018 nodes to V_select, V_evidence, V_check and the remainder ``valid_unused``).
    ``sizes`` must name exactly the splits in ``order`` and sum to ``len(y)``. Returns sorted int64 index arrays that
    are disjoint and cover every index once.
    """
    labels = np.asarray(y)
    order = tuple(order)
    remainder = order[-1]
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("y must be a 1-D integer array")
    if len(set(order)) != len(order) or set(sizes) != set(order):
        raise ValueError(f"sizes must name exactly {order}, got {sorted(sizes)}")
    if sum(sizes.values()) != labels.size:
        raise ValueError(f"split sizes sum to {sum(sizes.values())}, expected {labels.size}")
    if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
        raise ValueError(f"labels outside [0, {n_classes})")
    counts = np.bincount(labels, minlength=n_classes).astype(np.int64)
    quotas = {name: largest_remainder_quota(counts, sizes[name]) for name in order[:-1]}
    quotas[remainder] = counts - sum(quotas.values())
    if np.any(quotas[remainder] < 0) or int(quotas[remainder].sum()) != sizes[remainder]:
        raise ValueError("held-out quotas exceed the available items of some class")

    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in order}
    for cls in range(n_classes):
        members = rng.permutation(np.flatnonzero(labels == cls))
        start = 0
        for name in order:
            stop = start + int(quotas[name][cls])
            parts[name].append(members[start:stop])
            start = stop
        if start != members.size:
            raise AssertionError(f"class {cls}: dealt {start} of {members.size} indices")
    result = {name: np.sort(np.concatenate(parts[name])).astype(np.int64) for name in order}
    covered = np.concatenate(list(result.values()))
    if covered.size != labels.size or np.unique(covered).size != labels.size:
        raise AssertionError("splits are not a partition of the index set")
    return result
