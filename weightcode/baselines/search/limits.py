"""Edit-size limits of contract section 5.1, measured and enforced for search candidates.

The limit values are read from the shared executor's ``weightcode.patch.Limits`` (every arm uses one definition), with
``getattr`` defaults so that this module works on either side of the edit-limit switch: ``max_edit_frac`` if the
executor has it, ``max_changed_frac`` if not None, and ``max_rel_frob`` on the tensors with at least
``rel_frob_min_ndim`` dimensions (default 1, every tensor). ``projection_margin`` is ours: projected deltas stay
strictly inside the norm limit, so the float32 cast cannot push them over it.

"Changed" means the float32 child entry differs from the parent entry. The search patches list exactly their
changed entries (``patches.write_sparse_patch``), so their edit size in the executor's count equals the changed
count, and both fractions cap the same number. The relative Frobenius norm of a tensor is
``||child - parent||_F / ||parent||_F`` computed in float64 from the float32 arrays; a nonzero delta on a zero tensor
has infinite relative norm. The searches keep every candidate inside the limits (localised-set caps and a radial
projection of each limited tensor's delta), ``check_limits`` re-measures the float32 child before it is written, and
the executor checks the written patch again.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from weightcode.patch import Limits

PROJECTION_MARGIN = 1e-3
"""Projected deltas land at ``(1 - margin) * max_rel_frob * ||W||_F``."""


class EditLimitError(ValueError):
    """A candidate exceeds the contract's edit-size limits."""


@dataclass(frozen=True)
class EditLimits:
    """Contract 5.1 size limits as the searches apply them; build with ``from_executor`` (None disables a limit)."""

    max_changed_frac: float | None
    max_edit_frac: float | None
    max_rel_frob: float | None
    rel_frob_min_ndim: int = 1
    projection_margin: float = PROJECTION_MARGIN

    @classmethod
    def from_executor(cls, limits: Any | None = None, projection_margin: float = PROJECTION_MARGIN) -> EditLimits:
        """The size limits of ``limits`` (default ``weightcode.patch.Limits()``), read with ``getattr`` defaults."""
        source = Limits() if limits is None else limits
        return cls(
            max_changed_frac=getattr(source, "max_changed_frac", None),
            max_edit_frac=getattr(source, "max_edit_frac", None),
            max_rel_frob=getattr(source, "max_rel_frob", None),
            rel_frob_min_ndim=int(getattr(source, "rel_frob_min_ndim", 1)),
            projection_margin=projection_margin,
        )

    @classmethod
    def from_task(cls, stated: Mapping[str, Any]) -> EditLimits:
        """The size limits a task.json ``limits`` block states, read as ``weightcode.evalkit.instances`` defines them.

        A missing or null field is not applied. Without ``rel_frob_min_ndim`` the norm limit covers every tensor
        before the edit-limit switch and the tensors of rank >= 2 after it (when ``max_edit_frac`` is stated).
        """
        min_ndim = stated.get("rel_frob_min_ndim")
        if min_ndim is None:
            min_ndim = 2 if "max_edit_frac" in stated else 1
        return cls(
            max_changed_frac=stated.get("max_changed_frac"),
            max_edit_frac=stated.get("max_edit_frac"),
            max_rel_frob=stated.get("max_rel_frob"),
            rel_frob_min_ndim=int(min_ndim),
        )

    def as_dict(self) -> dict[str, float | int | None]:
        """JSON-ready copy for the run record."""
        return {
            "max_changed_frac": self.max_changed_frac,
            "max_edit_frac": self.max_edit_frac,
            "max_rel_frob": self.max_rel_frob,
            "rel_frob_min_ndim": self.rel_frob_min_ndim,
            "projection_margin": self.projection_margin,
        }

    def changed_frac_cap(self) -> float | None:
        """The binding fraction of changed entries (edit size = changed count for the sparse writer); None if none."""
        fracs = [frac for frac in (self.max_changed_frac, self.max_edit_frac) if frac is not None]
        return min(fracs) if fracs else None

    def max_changed(self, params: Mapping[str, np.ndarray]) -> int:
        """Largest number of changed entries allowed for this parameter set (all entries if no limit applies)."""
        total = sum(int(np.asarray(value).size) for value in params.values())
        cap = self.changed_frac_cap()
        return total if cap is None else int(math.floor(cap * total))

    def frob_radius(self, tensor: np.ndarray) -> float:
        """Largest delta norm the searches allow on ``tensor`` (the limit shrunk by the margin; inf if exempt)."""
        rel = self.max_rel_frob
        array = np.asarray(tensor, dtype=np.float64)
        if rel is None or array.ndim < self.rel_frob_min_ndim:
            return math.inf
        return (1.0 - self.projection_margin) * rel * float(np.linalg.norm(array))


@dataclass(frozen=True)
class EditStats:
    """Measured size of one edit."""

    n_changed: int
    n_total: int
    changed_frac: float
    rel_frob: dict[str, float]
    max_abs_delta: dict[str, float]
    ndim: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        """JSON-ready copy for the run record."""
        return {
            "n_changed": self.n_changed,
            "n_total": self.n_total,
            "changed_frac": self.changed_frac,
            "rel_frob": dict(self.rel_frob),
            "max_abs_delta": dict(self.max_abs_delta),
            "ndim": dict(self.ndim),
        }


def edit_stats(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray]) -> EditStats:
    """Changed count and per-tensor relative Frobenius norm of ``child`` against ``base`` (changed tensors only)."""
    if set(base) != set(child):
        raise ValueError(f"key mismatch: {sorted(set(base) ^ set(child))}")
    n_changed = 0
    n_total = 0
    rel: dict[str, float] = {}
    max_abs: dict[str, float] = {}
    ndim: dict[str, int] = {}
    for key in sorted(base):
        parent = np.asarray(base[key])
        value = np.asarray(child[key])
        if parent.shape != value.shape or parent.dtype != value.dtype:
            raise ValueError(f"{key}: {parent.dtype}{parent.shape} vs {value.dtype}{value.shape}")
        n_total += parent.size
        changed = int(np.count_nonzero(parent != value))
        if changed == 0:
            continue
        n_changed += changed
        delta = value.astype(np.float64) - parent.astype(np.float64)
        parent_norm = float(np.linalg.norm(parent.astype(np.float64)))
        delta_norm = float(np.linalg.norm(delta))
        rel[key] = math.inf if parent_norm == 0.0 else delta_norm / parent_norm
        max_abs[key] = float(np.max(np.abs(delta)))
        ndim[key] = int(parent.ndim)
    return EditStats(
        n_changed=n_changed,
        n_total=n_total,
        changed_frac=n_changed / n_total,
        rel_frob=rel,
        max_abs_delta=max_abs,
        ndim=ndim,
    )


def limit_violations(stats: EditStats, limits: EditLimits) -> list[str]:
    """Every limit ``stats`` breaks; the edit size of a sparse search patch is its changed count."""
    problems = []
    for name, frac in (("max_changed_frac", limits.max_changed_frac), ("max_edit_frac", limits.max_edit_frac)):
        if frac is not None and stats.changed_frac > frac:
            problems.append(f"changed fraction {stats.changed_frac:.6f} > {name} {frac}")
    if limits.max_rel_frob is not None:
        for key, value in stats.rel_frob.items():
            if stats.ndim[key] >= limits.rel_frob_min_ndim and value > limits.max_rel_frob:
                problems.append(f"{key}: relative Frobenius norm {value:.6f} > {limits.max_rel_frob}")
    return problems


def check_limits(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], limits: EditLimits) -> EditStats:
    """Measure ``child`` and raise ``EditLimitError`` if it breaks a contract limit."""
    stats = edit_stats(base, child)
    problems = limit_violations(stats, limits)
    if problems:
        raise EditLimitError("; ".join(problems))
    return stats


def shrink_to_radius(delta: np.ndarray, radius: float) -> np.ndarray:
    """Radial projection of ``delta`` onto the ball of the given Frobenius radius (unchanged if already inside)."""
    norm = float(np.linalg.norm(delta))
    if norm <= radius:
        return delta
    if radius <= 0.0:
        return np.zeros_like(delta)
    return delta * (radius / norm)
