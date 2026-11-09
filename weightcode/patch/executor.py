"""``apply_patch``: all-or-nothing application of one ``weightcode.patch.v0`` patch (contract sections 5.0, 5.1).

Order of work:

1. Caller arguments are checked (``TypeError`` or ``ValueError``, never ``PatchError``): the parent must be
   a non-empty mapping of finite float32 arrays and ``expected_parent_sha`` a 64-hex-character hash.
2. Structural validation of the whole patch (schema, parent hash, every op against the tensor shapes),
   then safety checks and shape checks of every referenced .npy file. All problems are collected; any
   problem raises one ``PatchError`` before anything is computed.
3. The ops run in order on float64 copies of the float32 parent tensors they touch; each touched tensor is
   cast to float32 once at the end.
4. Finiteness and the size limits are checked on the float32 child; any violation raises ``PatchError``.
   The edit size (``Limits.max_edit_frac``) is computed from the bound ops, before anything is applied, and
   its violation is reported together with the other validation problems.

"Changed" means the float32 child entry differs in value from the parent entry. An entry whose value is
unchanged keeps the parent's exact bits (this only matters for -0.0 versus 0.0), so ``n_changed == 0`` if
and only if ``child_sha256`` equals the parent's ``params_sha256``. The relative Frobenius norm of a tensor
is ``||child - parent||_F / ||parent||_F`` in float64; a zero parent tensor that changes has an infinite
relative norm, which fails unless ``max_rel_frob`` is None. The relative-norm limit applies to tensors with at
least ``limits.rel_frob_min_ndim`` dimensions (default 2); 1-D tensors (biases, normalisation gains) are then
reported but never limited by it, and neither are the tensors named in ``limits.rel_frob_exempt``. A name there
that is not a tensor of ``params`` is a caller error (``ValueError``).

The parent hash check compares the patch's ``parent_sha256`` with ``expected_parent_sha``, the hash of the
model the editor was shown. ``params`` may be a different model with the same keys and shapes: in the
DECOY arm the patch is checked against the decoy hash and applied to the real model by index.

The input mapping and its arrays are never mutated. The child arrays are fresh, C-contiguous and
read-only, so ``child_sha256`` stays true for them; copy an array to modify it. Saving the child with
``weightcode.common.identity.save_params`` reloads it and compares the canonical hash.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import count_changed, params_sha256
from weightcode.patch.errors import MessageLog
from weightcode.patch.files import load_npy_files, load_patch
from weightcode.patch.limits import Limits
from weightcode.patch.ops import SHA256_HEX, Workspace, frobenius, parse_patch


@dataclass(frozen=True, eq=False)
class PatchResult:
    """A successfully applied patch.

    ``child``: read-only C-contiguous float32 arrays with the parent's keys (in the parent's order) and
    shapes. ``child_sha256``: ``params_sha256(child)``. ``n_changed`` and ``changed_frac``: changed entries
    and their fraction over all parameters. ``edit_size`` and ``edit_frac``: numbers the patch specifies
    (contract 5.1) and their ratio to all parameter entries. ``per_tensor``: for every tensor, ``changed`` (int), ``size``
    (int), ``rel_frob`` (float, ``inf`` for a changed zero tensor) and ``max_abs_delta`` (float).
    """

    child: dict[str, np.ndarray]
    child_sha256: str
    n_changed: int
    changed_frac: float
    per_tensor: dict[str, dict[str, Any]]
    edit_size: int = 0
    edit_frac: float = 0.0

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary without the arrays; an infinite ``rel_frob`` is written as the string "inf"."""
        per_tensor = {
            name: {key: ("inf" if isinstance(value, float) and math.isinf(value) else value) for key, value in stats.items()}
            for name, stats in self.per_tensor.items()
        }
        return {
            "child_sha256": self.child_sha256,
            "n_changed": self.n_changed,
            "n_params": sum(stats["size"] for stats in self.per_tensor.values()),
            "changed_frac": self.changed_frac,
            "edit_size": self.edit_size,
            "edit_frac": self.edit_frac,
            "per_tensor": per_tensor,
        }


def _checked_parent(params: object) -> dict[str, np.ndarray]:
    if not isinstance(params, Mapping) or not params:
        raise ValueError("params must be a non-empty mapping of tensor name to float32 array")
    parent: dict[str, np.ndarray] = {}
    for name, value in params.items():
        if not isinstance(name, str):
            raise TypeError(f"parameter names must be strings, got {name!r}")
        if not isinstance(value, np.ndarray) or value.dtype != np.float32:
            kind = value.dtype if isinstance(value, np.ndarray) else type(value).__name__
            raise TypeError(f"parameter {name!r} must be a native float32 numpy array, got {kind}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"parameter {name!r} contains non-finite values")
        parent[name] = value
    if sum(value.size for value in parent.values()) == 0:
        raise ValueError("params hold no entries")
    return parent


def _tensor_stats(parent: np.ndarray, child: np.ndarray) -> dict[str, Any]:
    changed = int(np.count_nonzero(child != parent))
    stats: dict[str, Any] = {"changed": changed, "size": int(parent.size), "rel_frob": 0.0, "max_abs_delta": 0.0}
    if changed:
        delta = child.astype(np.float64) - parent.astype(np.float64)
        numerator = frobenius(delta)
        denominator = frobenius(parent.astype(np.float64))
        stats["rel_frob"] = numerator / denominator if denominator > 0.0 else math.inf
        stats["max_abs_delta"] = float(np.max(np.abs(delta)))
    return stats


def apply_patch(
    params: Mapping[str, np.ndarray],
    patch: Mapping[str, Any],
    patch_dir: str | os.PathLike[str] | None,
    limits: Limits,
    expected_parent_sha: str,
) -> PatchResult:
    """Validate ``patch`` completely, apply it to ``params`` and return the checked child.

    ``patch_dir`` holds the .npy files the patch references (None is allowed for a patch without file
    references). Raises ``PatchError`` with every problem found; never returns a partial result and never
    mutates ``params`` or ``patch``. See the module docstring for the order of checks and the semantics.
    """
    parent = _checked_parent(params)
    if not isinstance(limits, Limits):
        raise TypeError(f"limits must be a weightcode.patch.Limits, got {type(limits).__name__}")
    unknown = [name for name in limits.rel_frob_exempt if name not in parent]
    if unknown:
        raise ValueError(f"limits.rel_frob_exempt names tensors the model does not have: {unknown!r}")
    if not isinstance(expected_parent_sha, str) or SHA256_HEX.fullmatch(expected_parent_sha) is None:
        raise ValueError("expected_parent_sha must be a params_sha256 hex digest (64 lowercase hex characters)")

    log = MessageLog()
    ops = parse_patch(patch, {name: value.shape for name, value in parent.items()}, limits, expected_parent_sha, log)
    refs = [(f"ops[{op.index}].{field}", name) for op in ops for field, name in op.file_fields().items()]
    files = load_npy_files(patch_dir, refs, limits.max_file_bytes, log)
    bound = [op.bind(files, log) for op in ops]
    n_params = sum(value.size for value in parent.values())
    edit_size = 0
    if all(bound) and not log.count:
        edit_size = sum(op.edit_size() for op in ops)
        if limits.max_edit_frac is not None and edit_size > limits.max_edit_frac * n_params:
            log.add("size", f"the patch specifies {edit_size} numbers (edit size), above the limit "
                    f"{limits.max_edit_frac!r} x {n_params} parameters = {limits.max_edit_frac * n_params:g}")
    log.raise_if_any()

    workspace = Workspace(parent)
    # Overflow and invalid arithmetic are expected for some bad patches; they yield inf or NaN, which the
    # finiteness check below reports as a PatchError, so numpy's warnings would only duplicate that report.
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        for op in ops:
            op.apply(workspace)
        cast = {name: work.astype(np.float32) for name, work in workspace.tensors.items()}

    child: dict[str, np.ndarray] = {}
    per_tensor: dict[str, dict[str, Any]] = {}
    non_finite_tensors: set[str] = set()
    for name, base in parent.items():
        if name in cast:
            value = cast[name]
            non_finite = value.size - int(np.count_nonzero(np.isfinite(value)))
            if non_finite:
                non_finite_tensors.add(name)
                log.add(f"tensor {name!r}", f"{non_finite} entries are not finite after the patch (float64 overflow or "
                        "invalid arithmetic, or values beyond the float32 range)")
            else:
                # Entries equal in value to the parent keep the parent's bits (-0.0 versus 0.0).
                np.copyto(value, base, where=value == base)
        else:
            value = np.array(base, dtype=np.float32, order="C", copy=True)
        child[name] = value
        per_tensor[name] = _tensor_stats(base, value)

    if n_params != sum(stats["size"] for stats in per_tensor.values()):
        raise RuntimeError("parameter counts disagree between the parent and the per-tensor summary")
    n_changed = count_changed(parent, child)
    if n_changed != sum(stats["changed"] for stats in per_tensor.values()):
        raise RuntimeError("changed-entry counts disagree between count_changed and the per-tensor summary")
    changed_frac = n_changed / n_params
    if limits.max_changed_frac is not None and changed_frac > limits.max_changed_frac:
        log.add("size", f"{n_changed} of {n_params} parameters change (fraction {changed_frac!r}), above the limit "
                f"{limits.max_changed_frac!r}")
    if limits.max_rel_frob is not None:
        for name, stats in per_tensor.items():
            rel = stats["rel_frob"]
            if name not in non_finite_tensors and limits.rel_frob_applies(name, parent[name].ndim) and rel > limits.max_rel_frob:
                shown = "infinite (the parent tensor is all zeros)" if math.isinf(rel) else repr(rel)
                log.add(f"tensor {name!r}", f"relative Frobenius norm of the change is {shown}, above the limit "
                        f"{limits.max_rel_frob!r}")
    log.raise_if_any()

    for value in child.values():
        value.flags.writeable = False
    return PatchResult(
        child=child,
        child_sha256=params_sha256(child),
        n_changed=n_changed,
        changed_frac=changed_frac,
        per_tensor=per_tensor,
        edit_size=edit_size,
        edit_frac=edit_size / n_params,
    )


def apply_patch_file(
    params: Mapping[str, np.ndarray],
    patch_path: str | os.PathLike[str],
    limits: Limits,
    expected_parent_sha: str,
) -> PatchResult:
    """``load_patch(patch_path)``, then ``apply_patch`` with the JSON file's directory as ``patch_dir``."""
    path = Path(patch_path)
    return apply_patch(params, load_patch(path), path.parent, limits, expected_parent_sha)
