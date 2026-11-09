"""Canonical identity and storage of a parameter set (contract section 5.2).

A model is identified by the hash of its tensors, not by the hash of the file that holds them, because
npz/zip metadata differs between writers. Schema ``weightcode.params_sha256.v0`` hashes, for each key in sorted
order: the key as UTF-8, a NUL byte, ``dtype.str`` as ASCII, a NUL byte, the shape as comma-joined decimals
(empty for a scalar), a NUL byte, then the array bytes in C order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np

SCHEMA = "weightcode.params_sha256.v0"

Params = dict[str, np.ndarray]


def params_sha256(params: Mapping[str, np.ndarray]) -> str:
    """Return the canonical sha256 hex digest of a parameter mapping."""
    if not params:
        raise ValueError("cannot hash an empty parameter set")
    digest = hashlib.sha256()
    for key in sorted(params):
        array = np.asarray(params[key])
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(dim) for dim in array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def load_params(path: str | Path) -> Params:
    """Load a parameter npz without pickle; every entry becomes a C-contiguous array."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: np.ascontiguousarray(archive[key]) for key in archive.files}


def save_params(path: str | Path, params: Mapping[str, np.ndarray]) -> str:
    """Save parameters to an uncompressed npz and return their canonical hash.

    The file is written to a temporary name and renamed, so a crash never leaves a partial checkpoint
    under the final name. The saved file is reloaded and its canonical hash compared with the input.
    """
    target = Path(path)
    if target.suffix != ".npz":
        raise ValueError(f"checkpoint path must end in .npz: {target}")
    expected = params_sha256(params)
    temporary = target.with_name(target.name + ".tmp.npz")
    np.savez(temporary, **{key: np.ascontiguousarray(value) for key, value in params.items()})
    reloaded = params_sha256(load_params(temporary))
    if reloaded != expected:
        temporary.unlink()
        raise RuntimeError(f"reload hash mismatch for {target}: {reloaded} != {expected}")
    temporary.replace(target)
    return expected


def count_changed(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray]) -> int:
    """Number of scalar entries that differ between two parameter sets with identical keys and shapes."""
    if set(base) != set(child):
        raise ValueError(f"key mismatch: {sorted(set(base) ^ set(child))}")
    changed = 0
    for key, base_value in base.items():
        child_value = child[key]
        if base_value.shape != child_value.shape:
            raise ValueError(f"shape mismatch for {key}: {base_value.shape} != {child_value.shape}")
        changed += int(np.count_nonzero(base_value != child_value))
    return changed
