"""Edit-size and format limits applied by the executor (contract section 5.1, HARNESS_SPEC section 2)."""

from __future__ import annotations

import dataclasses
import io
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Limits on how large an edit may be. ``Limits.without_size_limits`` sets all of them to None (the REWRITE arm).
SIZE_LIMIT_FIELDS = ("max_changed_frac", "max_rel_frob", "max_edit_frac")
# Fields added after instances were first built, with the JSON value at which ``task_limits`` leaves them out: at
# that value they keep the behaviour from before the field existed, so a task.json without them is current.
OMITTED_AT_DEFAULT: dict[str, Any] = {"rel_frob_exempt": []}


def _is_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Limits:
    """Limits for one arm. The values are contract proposals, frozen after the pilot.

    ``max_edit_frac``: largest edit size over all parameter entries, where the edit size of a patch is the
    count of numbers its ops specify (contract 5.1: listed indices for scale_rows and scale_cols, tuples for
    the index form of set and add, 1 for the slice form with ``value``, the block size with ``values_file``,
    the direction length for project_out, ``r * (rows + cols)`` for low_rank_add, the tensor size for
    add_dense, the group-map length for tie); None disables it (the REWRITE arm). ``max_rel_frob``: largest
    ``||child - parent||_F / ||parent||_F`` for each tensor with at least ``rel_frob_min_ndim`` dimensions
    (default 2, so 1-D tensors such as biases and normalisation gains are exempt); None disables it.
    ``rel_frob_exempt``: names of tensors that ``max_rel_frob`` does not apply to, whatever their rank (a list
    from task.json is stored as a tuple); every name must be a tensor of the model the limits are applied to.
    It is for new capacity, such as the output rows of classes the model was never trained on (PREREG E2).
    ``max_changed_frac``: optional cap on the fraction of changed float32 entries over all parameters; None
    disables it. ``max_ops``: most ops in one patch.
    ``max_rank``: largest ``r`` of ``low_rank_add``. ``max_file_bytes``: largest total size of the distinct
    .npy files one patch references.
    """

    max_changed_frac: float | None = None
    max_rel_frob: float | None = 0.5
    max_ops: int = 64
    max_rank: int = 8
    max_file_bytes: int = 64 * 2**20
    max_edit_frac: float | None = 0.05
    rel_frob_min_ndim: int = 2
    rel_frob_exempt: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        problems = []
        exempt = self.rel_frob_exempt
        if isinstance(exempt, list):
            exempt = tuple(exempt)
            object.__setattr__(self, "rel_frob_exempt", exempt)
        if not (isinstance(exempt, tuple) and all(isinstance(name, str) and name for name in exempt)):
            problems.append(f"rel_frob_exempt must be a list of non-empty tensor names, got {exempt!r}")
        elif len(set(exempt)) != len(exempt):
            problems.append(f"rel_frob_exempt names a tensor more than once: {list(exempt)!r}")
        frac = self.max_changed_frac
        if frac is not None and not (_is_real(frac) and 0.0 <= frac <= 1.0):
            problems.append(f"max_changed_frac must be None or a number in [0, 1], got {frac!r}")
        edit = self.max_edit_frac
        if edit is not None and not (_is_real(edit) and 0.0 <= edit <= 1.0):
            problems.append(f"max_edit_frac must be None or a number in [0, 1], got {edit!r}")
        rel = self.max_rel_frob
        if rel is not None and not (_is_real(rel) and math.isfinite(rel) and rel >= 0.0):
            problems.append(f"max_rel_frob must be None or a finite number >= 0, got {rel!r}")
        if not (_is_count(self.rel_frob_min_ndim) and self.rel_frob_min_ndim >= 1):
            problems.append(f"rel_frob_min_ndim must be an int >= 1, got {self.rel_frob_min_ndim!r}")
        if not (_is_count(self.max_ops) and self.max_ops >= 1):
            problems.append(f"max_ops must be an int >= 1, got {self.max_ops!r}")
        if not (_is_count(self.max_rank) and self.max_rank >= 1):
            problems.append(f"max_rank must be an int >= 1, got {self.max_rank!r}")
        if not (_is_count(self.max_file_bytes) and self.max_file_bytes >= 0):
            problems.append(f"max_file_bytes must be an int >= 0, got {self.max_file_bytes!r}")
        if problems:
            raise ValueError("invalid Limits: " + "; ".join(problems))

    def rel_frob_applies(self, name: str, ndim: int) -> bool:
        """Whether ``max_rel_frob`` limits tensor ``name`` of rank ``ndim``: the limit is set, the rank is at least
        ``rel_frob_min_ndim`` and the name is not in ``rel_frob_exempt``."""
        return self.max_rel_frob is not None and ndim >= self.rel_frob_min_ndim and name not in self.rel_frob_exempt

    def without_size_limits(self) -> Limits:
        """A copy with every edit-size limit disabled and the format limits unchanged (the REWRITE arm)."""
        return dataclasses.replace(self, **{name: None for name in SIZE_LIMIT_FIELDS})

    def without_edit_limits(self, n_tensors: int) -> Limits:
        """The limits of a reference that ignores the edit limits: every edit-size limit disabled and ``max_ops`` at
        least ``n_tensors``, so a full-model delta (one op per tensor) can be written whatever the model's tensor
        count; every other format limit unchanged. On a model with at most ``max_ops`` tensors it equals
        ``without_size_limits()``."""
        if not (_is_count(n_tensors) and n_tensors >= 1):
            raise ValueError(f"n_tensors must be an int >= 1, got {n_tensors!r}")
        return dataclasses.replace(self.without_size_limits(), max_ops=max(self.max_ops, int(n_tensors)))


def dense_delta_bytes(shapes: Iterable[Sequence[int]]) -> int:
    """Total size of the float64 .npy files (``np.save``, format 1.0, header included) of a dense delta of every tensor
    with these shapes: the most file bytes a patch of one ``add_dense`` op per tensor references.

    A program arm's derived patch and a reference's delta patch are written this way (float64, so that the executor's
    float64 arithmetic reproduces the child). ``max_file_bytes`` raised to this value (``with_dense_file_cap``) never
    refuses such a patch before the stated limits do.
    """
    total = 0
    for shape in shapes:
        dims = tuple(int(d) for d in shape)
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header, {"descr": np.lib.format.dtype_to_descr(np.dtype("<f8")), "fortran_order": False, "shape": dims})
        total += header.tell() + 8 * math.prod(dims)
    return total


def with_dense_file_cap(limits: Limits, dense_bytes: int | None) -> Limits:
    """``limits`` with ``max_file_bytes = max(max_file_bytes, dense_bytes)`` (``dense_delta_bytes`` of the student);
    unchanged when ``dense_bytes`` is None or not larger. PREREG E12 (2026-11-05): a float64 dense delta of ResNet-18
    (11.19M entries) is about 89 MB, above the 64 MiB format cap, which would refuse every full-model update before
    the stated size limits apply. Like ``max_ops = max(max_ops, n_tensors)``, the format cap is raised to what the
    student's own full delta needs; on every earlier student the dense delta fits, so nothing changes there."""
    if dense_bytes is None:
        return limits
    if not (_is_count(dense_bytes) and dense_bytes >= 0):
        raise ValueError(f"dense_bytes must be an int >= 0, got {dense_bytes!r}")
    return dataclasses.replace(limits, max_file_bytes=max(limits.max_file_bytes, int(dense_bytes)))


def task_limits(limits: Limits | None = None) -> dict[str, Any]:
    """``limits`` (``Limits()`` when None) as task.json states them, with JSON values (lists, not tuples).

    The one serialization that instance builders write and that staleness checks compare a task.json against: a
    task.json is current when its ``limits`` equal ``task_limits()``. A field added after instances were built whose
    default keeps the old behaviour is left out at that default, so the existing snapshots stay current.
    """
    limits = Limits() if limits is None else limits
    if not isinstance(limits, Limits):
        raise TypeError(f"limits must be a weightcode.patch.Limits, got {type(limits).__name__}")
    stated = json.loads(json.dumps(dataclasses.asdict(limits)))
    for name, absent in OMITTED_AT_DEFAULT.items():
        if stated[name] == absent:
            del stated[name]
    return stated
