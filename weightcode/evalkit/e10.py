"""PREREG E10 (CIFAR-10-C with ResNet-20, dataset ``cifar10_r20``): the limits of an E10 update program.

An E10 update is held to the program-arm limits of the harness (``weightcode.harness.arms.applied_limits`` with a program
arm such as LEARN, a test holds the two equal): the relative Frobenius norm at most 0.5 on every tensor of rank 2 or
more, at most 20% of all parameter entries changed (BatchNorm running statistics are parameters of model.npz: 1-D,
exempt from the norm limit and counted in the 20%), every other size limit off, and max_ops = max(task.json max_ops,
tensor count). Every E10 reference run within limits is held to the same limits.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from weightcode.patch import Limits
from weightcode.patch.limits import SIZE_LIMIT_FIELDS, with_dense_file_cap

E10_DATASET = "cifar10_r20"
E10_MAX_CHANGED_FRAC = 0.20
"""At most 20% of the parameter entries changed, the changed-entry cap of every program arm."""
_PROGRAM_SIZE_FIELDS = ("max_rel_frob", "max_changed_frac")


class E10Error(ValueError):
    """An E10 limit input is malformed."""


def e10_program_limits(task_limits_json: Mapping[str, Any], n_tensors: int | None = None,
                       dense_bytes: int | None = None) -> Limits:
    """task.json's limits with every size limit but max_rel_frob off, at most ``E10_MAX_CHANGED_FRAC`` of the entries
    changed, max_ops = max(max_ops, n_tensors) when ``n_tensors`` (the student's tensor count) is given and
    max_file_bytes = max(max_file_bytes, dense_bytes) when ``dense_bytes`` (``weightcode.patch.limits.dense_delta_bytes`` of
    the student) is given (``with_dense_file_cap``; it changes nothing on the E10 and E11 students)."""
    if not isinstance(task_limits_json, Mapping):
        raise E10Error(f"task.json limits must be an object, got {type(task_limits_json).__name__}")
    if n_tensors is not None and (isinstance(n_tensors, bool) or not isinstance(n_tensors, int) or n_tensors < 1):
        raise E10Error(f"n_tensors must be an int >= 1, got {n_tensors!r}")
    limits = Limits(**dict(task_limits_json))
    disabled = {name: None for name in SIZE_LIMIT_FIELDS if name not in _PROGRAM_SIZE_FIELDS}
    max_ops = limits.max_ops if n_tensors is None else max(limits.max_ops, n_tensors)
    return with_dense_file_cap(
        dataclasses.replace(limits, **disabled, max_changed_frac=E10_MAX_CHANGED_FRAC, max_ops=max_ops), dense_bytes)


__all__ = ["E10Error", "E10_DATASET", "E10_MAX_CHANGED_FRAC", "e10_program_limits"]
