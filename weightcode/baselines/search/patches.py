"""Sparse patch writer for the search baselines (contract section 5.0 ops, executor checks of section 5.1).

Both searches change few entries: Arachne the localised weights, CARE the outgoing columns of the selected neurons.
Each changed tensor becomes one index-form ``set`` op that lists the changed entries and their float32 child values
(written as exact JSON numbers), so the patch specifies exactly as many numbers as it changes and the executor
rebuilds the child bit for bit. ``weightcode.baselines.instance_io.write_delta_patch`` writes ``add``/``add_dense`` ops whose
size, in the executor's edit-size count, is the whole tensor; the executor rejects those for any tensor larger than
the edit-size limit (for example fc1 and fc2 of the MLP), although the edit itself is small.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

PATCH_SCHEMA = "weightcode.patch.v0"


def sparse_ops(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    """One ``set`` op per tensor with changed entries (value comparison, as the executor counts changes)."""
    if set(base) != set(child):
        raise ValueError(f"key mismatch: {sorted(set(base) ^ set(child))}")
    ops: list[dict[str, object]] = []
    for key in sorted(base):
        parent = np.asarray(base[key])
        value = np.asarray(child[key])
        if parent.shape != value.shape or parent.dtype != value.dtype:
            raise ValueError(f"{key}: {parent.dtype}{parent.shape} vs {value.dtype}{value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key}: child has non-finite values")
        changed = np.argwhere(parent != value)
        if len(changed) == 0:
            continue
        values = value[tuple(changed.T)]
        ops.append(
            {
                "op": "set",
                "tensor": key,
                "index": changed.tolist(),
                "values": [float(number) for number in values],
            }
        )
    return ops


def write_sparse_patch(
    out_dir: str | Path,
    parent_sha256: str,
    base: Mapping[str, np.ndarray],
    child: Mapping[str, np.ndarray],
    hypothesis: str,
    predicted_effect: str,
) -> Path:
    """Write ``child`` as index-form ``set`` ops into a new directory ``out_dir``; returns the patch.json path."""
    ops = sparse_ops(base, child)
    if not ops:
        raise ValueError("child equals the base model; there is no patch to write")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    patch = {
        "schema": PATCH_SCHEMA,
        "parent_sha256": parent_sha256,
        "ops": ops,
        "hypothesis": hypothesis,
        "predicted_effect": predicted_effect,
    }
    path = root / "patch.json"
    path.write_text(json.dumps(patch, separators=(",", ":")))
    return path
