"""Edit sizes against the contract 5.1 limits, per-tensor norm-limit scaling, and patch writing and application.

Patches are applied only by the shared executor ``weightcode.patch`` (contract section 5: one executor for every arm). The
limit values are the executor's ``Limits`` defaults, so a change of the frozen limits reaches the baselines without
a second copy of the numbers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import PATCH_SCHEMA, delta_encoding
from weightcode.patch import Limits, PatchResult, apply_patch_file

_LIMITS = Limits()
MAX_EDIT_FRAC: float | None = getattr(_LIMITS, "max_edit_frac", None)
"""Contract 5.1 edit size (numbers the patch specifies) over all parameter entries; None if the executor has no
such limit."""

MAX_CHANGED_FRAC: float | None = _LIMITS.max_changed_frac
"""Contract 5.1 changed-entry fraction over all parameters (executor default); None when disabled."""

MAX_REL_FROB: float | None = _LIMITS.max_rel_frob
"""Contract 5.1 relative Frobenius norm of a tensor's delta (executor default); None when disabled."""

REL_FROB_MIN_NDIM: int = int(getattr(_LIMITS, "rel_frob_min_ndim", 1))
"""Tensors with fewer dimensions than this are exempt from the norm limit (executor rule; 1 = every tensor)."""

NORM_SCALE_MARGIN = 1e-4
"""Scaled deltas land at ``MAX_REL_FROB * (1 - margin)`` so the float32 cast cannot push them over the limit."""

PATCH_APPLIER = "weightcode.patch.apply_patch_file"

Params = dict[str, np.ndarray]


def rel_frob(parent: np.ndarray, child: np.ndarray) -> float:
    """||child - parent||_F / ||parent||_F in float64; infinite for a nonzero delta on a zero parent."""
    parent64 = np.asarray(parent, dtype=np.float64)
    delta_norm = float(np.linalg.norm(np.asarray(child, dtype=np.float64) - parent64))
    parent_norm = float(np.linalg.norm(parent64))
    if parent_norm > 0.0:
        return delta_norm / parent_norm
    return math.inf if delta_norm > 0.0 else 0.0


def edit_stats(
    base: Mapping[str, np.ndarray],
    child: Mapping[str, np.ndarray],
    *,
    edit_size: int | None = None,
    max_changed_frac: float | None = MAX_CHANGED_FRAC,
    max_rel_frob: float | None = MAX_REL_FROB,
    max_edit_frac: float | None = MAX_EDIT_FRAC,
    rel_frob_min_ndim: int = REL_FROB_MIN_NDIM,
    rel_frob_exempt: Collection[str] = (),
) -> dict[str, Any]:
    """Per-tensor and total edit size of ``child`` against ``base`` and whether it meets the size limits.

    "Changed" means the child entry differs from the parent entry. The edit size is the count of numbers a patch
    specifies (contract 5.1). Pass ``edit_size`` as the executor counted it for the patch actually written; that is the
    edit size of record (an explicit-op patch such as ``tie`` specifies far fewer numbers than it changes, and the
    delta writer can move a tensor to add_dense to stay under the JSON cap). Without it, the edit size is the
    ``delta_encoding`` prediction, labelled ``edit_size_source: "delta_encoding"``, for diagnostics only. The norm limit
    applies to tensors with at least ``rel_frob_min_ndim`` dimensions that are not named in ``rel_frob_exempt``, as in
    the executor. A limit of None is not checked.
    """
    if set(base) != set(child):
        raise ValueError(f"key mismatch: {sorted(set(base) ^ set(child))}")
    exempt = _checked_exempt(base, rel_frob_exempt)
    if edit_size is not None and edit_size < 0:
        raise ValueError(f"edit_size must be >= 0, got {edit_size}")
    per_tensor: dict[str, dict[str, Any]] = {}
    violations: list[str] = []
    n_changed = 0
    n_params = 0
    encoded_size = 0
    for key in sorted(base):
        parent = np.asarray(base[key])
        value = np.asarray(child[key])
        if parent.shape != value.shape:
            raise ValueError(f"{key}: shape {value.shape} != parent shape {parent.shape}")
        changed = int(np.count_nonzero(parent != value))
        relative = rel_frob(parent, value)
        delta = value.astype(np.float64) - parent.astype(np.float64)
        per_tensor[key] = {
            "changed": changed,
            "size": int(parent.size),
            "rel_frob": relative,
            "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        }
        if edit_size is None:
            encoding, size_of_edit = delta_encoding(delta)
            per_tensor[key].update(encoding=encoding, edit_size=size_of_edit)
            encoded_size += size_of_edit
        if key in exempt:
            per_tensor[key]["rel_frob_exempt"] = True
        elif max_rel_frob is not None and parent.ndim >= rel_frob_min_ndim and relative > max_rel_frob:
            violations.append(f"{key}: relative Frobenius norm {relative:.6g} > {max_rel_frob}")
        n_changed += changed
        n_params += int(parent.size)
    edit_size_source = "delta_encoding" if edit_size is None else "executor"
    if edit_size is None:
        edit_size = encoded_size
    changed_frac = n_changed / n_params
    edit_frac = edit_size / n_params
    if max_changed_frac is not None and changed_frac > max_changed_frac:
        violations.append(f"changed fraction {changed_frac:.6g} > {max_changed_frac}")
    if max_edit_frac is not None and edit_frac > max_edit_frac:
        violations.append(f"edit fraction {edit_frac:.6g} > {max_edit_frac}")
    return {
        "n_changed": n_changed,
        "n_params": n_params,
        "changed_frac": changed_frac,
        "edit_size": edit_size,
        "edit_frac": edit_frac,
        "edit_size_source": edit_size_source,
        "per_tensor": per_tensor,
        "within_limits": not violations,
        "violations": violations,
    }


def scale_to_norm_limit(
    base: Mapping[str, np.ndarray],
    child: Mapping[str, np.ndarray],
    tensors: Collection[str],
    *,
    limit: float | None = MAX_REL_FROB,
    margin: float = NORM_SCALE_MARGIN,
    min_ndim: int = REL_FROB_MIN_NDIM,
    exempt: Collection[str] = (),
) -> tuple[Params, dict[str, dict[str, Any]]]:
    """Scale each listed tensor's delta down to ``limit * (1 - margin)`` of its parent norm when it exceeds ``limit``.

    Returns the new child and, per listed tensor, the relative norm before, whether it was scaled, the factor and the
    realized relative norm after the cast back to the parameter dtype. A tensor whose parent norm is zero cannot carry
    any delta under a relative limit, so an over-limit delta there is removed (factor 0). Tensors with fewer than
    ``min_ndim`` dimensions are exempt from the limit, as in the executor, and so is everything when ``limit`` is None.
    The tensors named in ``exempt`` (``Limits.rel_frob_exempt``) are exempt whatever their rank and are marked so.
    """
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"margin must be in [0, 1), got {margin}")
    exempt_names = _checked_exempt(base, exempt)
    out: Params = dict(child)
    records: dict[str, dict[str, Any]] = {}
    for key in sorted(tensors):
        before = rel_frob(base[key], child[key])
        if key in exempt_names:
            records[key] = {"rel_frob_before": before, "scaled": False, "factor": 1.0, "rel_frob_after": before,
                            "exempt": True}
            continue
        if limit is None or np.ndim(base[key]) < min_ndim or before <= limit:
            records[key] = {"rel_frob_before": before, "scaled": False, "factor": 1.0, "rel_frob_after": before}
            continue
        parent64 = np.asarray(base[key], dtype=np.float64)
        delta = np.asarray(child[key], dtype=np.float64) - parent64
        parent_norm = float(np.linalg.norm(parent64))
        factor = 0.0 if parent_norm == 0.0 else limit * (1.0 - margin) * parent_norm / float(np.linalg.norm(delta))
        value = (parent64 + factor * delta).astype(base[key].dtype)
        after = rel_frob(base[key], value)
        if not after <= limit:
            raise RuntimeError(f"{key}: relative norm {after} still above {limit} after scaling by {factor}")
        out[key] = value
        records[key] = {"rel_frob_before": before, "scaled": True, "factor": factor, "rel_frob_after": after}
    return out, records


def edit_stats_under(
    base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], limits: Limits, *, edit_size: int | None = None
) -> dict[str, Any]:
    """``edit_stats`` against every size limit of ``limits``, its rank threshold and its rel-Frobenius exemptions."""
    return edit_stats(
        base,
        child,
        edit_size=edit_size,
        max_changed_frac=limits.max_changed_frac,
        max_rel_frob=limits.max_rel_frob,
        max_edit_frac=limits.max_edit_frac,
        rel_frob_min_ndim=limits.rel_frob_min_ndim,
        rel_frob_exempt=limits.rel_frob_exempt,
    )


def _checked_exempt(base: Mapping[str, np.ndarray], exempt: Collection[str]) -> frozenset[str]:
    """The exempt tensor names, refused if one is not a tensor of ``base`` (as the executor refuses it)."""
    names = frozenset(exempt)
    unknown = sorted(names - set(base))
    if unknown:
        raise ValueError(f"rel_frob_exempt names {unknown}, which are not tensors of the model")
    return names


def apply_patch_result(
    params: Mapping[str, np.ndarray],
    patch_path: str | Path,
    expected_parent_sha: str,
    *,
    enforce_limits: bool,
    limits: Limits | None = None,
) -> PatchResult:
    """Apply a patch JSON file with the shared executor and return its full result (child, counts, edit size).

    ``limits``: the run's limits, ``Limits()`` (the contract 5.1 defaults) when None. With ``enforce_limits`` the
    executor checks their size limits and rejects the patch if they fail; without, every size limit is disabled and
    the op count may reach the model's tensor count (``Limits.without_edit_limits``: a full-model delta is one op per
    tensor); every other check still runs.
    """
    base_limits = Limits() if limits is None else limits
    applied = base_limits if enforce_limits else base_limits.without_edit_limits(len(params))
    return apply_patch_file(params, patch_path, applied, expected_parent_sha)


def apply_patch_path(
    params: Mapping[str, np.ndarray],
    patch_path: str | Path,
    expected_parent_sha: str,
    *,
    enforce_limits: bool,
    limits: Limits | None = None,
) -> Params:
    """Apply a patch JSON file with the shared executor and return the child parameters (see ``apply_patch_result``)."""
    result = apply_patch_result(params, patch_path, expected_parent_sha, enforce_limits=enforce_limits, limits=limits)
    return {key: np.ascontiguousarray(value) for key, value in result.child.items()}


def write_ops_patch(
    out_dir: str | Path,
    parent_sha256: str,
    ops: Sequence[Mapping[str, Any]],
    hypothesis: str,
    predicted_effect: str,
) -> Path:
    """Write a ``weightcode.patch.v0`` patch with explicit inline ops (no .npy files) to ``out_dir/patch.json``.

    For edits whose natural op is not a delta, such as ``tie``. The executor validates the ops when the runner
    applies the patch; nothing is checked here beyond the ops being JSON objects with an ``op`` field.
    """
    if not ops or not all(isinstance(op, Mapping) and isinstance(op.get("op"), str) for op in ops):
        raise ValueError("ops must be a non-empty list of objects with an 'op' field")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    patch = {
        "schema": PATCH_SCHEMA,
        "parent_sha256": parent_sha256,
        "ops": [dict(op) for op in ops],
        "hypothesis": hypothesis,
        "predicted_effect": predicted_effect,
    }
    path = root / "patch.json"
    path.write_text(json.dumps(patch, indent=1, allow_nan=False))
    return path
