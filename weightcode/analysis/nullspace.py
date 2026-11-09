"""PREREG D1: closed-form null-space projection of a program's edit (the AlphaEdit and gradient-projection-memory idea).

For each edited weight tensor W, the inputs to its layer on the old rows (x_old, the clean experience rows, all tokens)
on the unedited student give the uncentred second moment ``C = X^T X / N``. Its top eigenvectors holding a fraction
``e`` of the energy (the trace) span the "old" subspace ``U``, and the delta's input side is projected onto the
complement: ``dW' = dW (I - U U^T)``. Bias and other 1-D deltas are unchanged. A linear weight is read as
[out, in]; the patch projection [DIM, C, P, P] as [DIM, C * P * P], its input being the image patches in the order
the forward pass flattens them.

The layer inputs come from the instance's own forward.py (the ViT-Lite reference forward,
``weightcode.models.templates/forward_vitlite.py.tmpl``): its helpers are executed from the file's bytes, the forward pass is
replayed with the inputs captured, and the replayed logits are checked against ``forward(params, x)``. No label is
read and nothing is fitted.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

PATCH_WEIGHT = "tokenizer.conv_layers.0.0.weight"
LINEAR_RE = re.compile(r"^classifier\.blocks\.(\d+)\.(linear1|linear2)\.weight$")
FORWARD_NAMES = ("_patches", "_layernorm", "_gelu", "_linear", "_attention", "_checked", "_images", "MEAN", "STD",
                 "DEPTH", "PATCH_WEIGHT", "PATCH_BIAS", "N_PATCHES", "N_TOKENS", "forward")
PARITY_TOL = 1e-6
"""Largest allowed |replayed logit - forward.py logit| (float32 logits of the same float64 arithmetic)."""
ENERGY_FRACTIONS = (0.90, 0.99, 0.999)
"""PREREG D1's grid of energy fractions."""


class NullspaceError(RuntimeError):
    """The forward file, a tensor or a delta does not fit the projection as specified."""


@dataclass(frozen=True)
class OldBasis:
    """The top eigenvectors of one layer's input second moment holding at least ``fraction`` of its energy."""

    tensor: str
    fraction: float
    basis: np.ndarray  # [d_in, k], orthonormal columns
    captured: float  # energy fraction the k vectors hold (>= fraction)
    d_in: int
    n_rows: int

    @property
    def k(self) -> int:
        return int(self.basis.shape[1])


def forward_module(path: Path, expected_sha256: str | None = None) -> ModuleType:
    """Execute a ViT-Lite forward.py from its bytes (no second read, no __pycache__) and return the module."""
    source = Path(path).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise NullspaceError(f"{path} hashes to {digest}, expected {expected_sha256}")
    module = ModuleType(f"weightcode_d1_forward_{digest[:16]}")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    missing = [name for name in FORWARD_NAMES if not hasattr(module, name)]
    if missing:
        raise NullspaceError(f"{path} is not the ViT-Lite reference forward: missing {missing}")
    return module


def layer_of(tensor: str) -> tuple[str, int | None]:
    """("patch", None) for the patch projection, ("linear1" | "linear2", block) for an MLP weight; else refused."""
    if tensor == PATCH_WEIGHT:
        return "patch", None
    match = LINEAR_RE.match(tensor)
    if match is None:
        raise NullspaceError(f"{tensor}: D1 projects the patch projection and MLP weights only")
    return match.group(2), int(match.group(1))


def layer_inputs(module: ModuleType, params: Mapping[str, np.ndarray], x_uint8: np.ndarray,
                 tensors: Collection[str]) -> tuple[dict[str, np.ndarray], float]:
    """The inputs of each named weight tensor's layer on ``x_uint8``, and the parity of the replay.

    Returns per tensor a float64 matrix [rows, d_in]: the image patches (every image's N_PATCHES patches) for the
    patch projection; for ``blocks.i.linear1`` the norm1 output and for ``blocks.i.linear2`` the GELU hidden
    activations, every image's N_TOKENS tokens (class token included). The second value is the largest absolute
    difference between the replayed logits and ``module.forward(params, x)``; above ``PARITY_TOL`` it is refused.
    """
    wanted = {tensor: layer_of(tensor) for tensor in tensors}
    x = module._images(x_uint8)
    t = module._checked(dict(params))
    dim = t[module.PATCH_WEIGHT].shape[0]
    h = x.astype(np.float64) / 255.0
    h = (h - module.MEAN.reshape(1, -1, 1, 1)) / module.STD.reshape(1, -1, 1, 1)
    patches = module._patches(h)
    out: dict[str, np.ndarray] = {}
    if PATCH_WEIGHT in wanted:
        out[PATCH_WEIGHT] = patches.reshape(-1, patches.shape[-1])
    tokens = module._linear(patches, t[module.PATCH_WEIGHT].reshape(dim, -1), t[module.PATCH_BIAS])
    cls = np.broadcast_to(t["classifier.class_emb"], (x.shape[0], 1, dim))
    h = np.concatenate([cls, tokens], axis=1) + t["classifier.positional_emb"]
    for index in range(module.DEPTH):
        block = f"classifier.blocks.{index}"
        h = h + module._attention(module._layernorm(h, t[f"{block}.pre_norm.weight"], t[f"{block}.pre_norm.bias"]),
                                  t, block)
        h = module._layernorm(h, t[f"{block}.norm1.weight"], t[f"{block}.norm1.bias"])
        hidden = module._gelu(module._linear(h, t[f"{block}.linear1.weight"], t[f"{block}.linear1.bias"]))
        if f"{block}.linear1.weight" in wanted:
            out[f"{block}.linear1.weight"] = h.reshape(-1, h.shape[-1]).copy()
        if f"{block}.linear2.weight" in wanted:
            out[f"{block}.linear2.weight"] = hidden.reshape(-1, hidden.shape[-1]).copy()
        h = h + module._linear(hidden, t[f"{block}.linear2.weight"], t[f"{block}.linear2.bias"])
    feats = module._layernorm(h[:, 0], t["classifier.norm.weight"], t["classifier.norm.bias"])
    logits = (feats @ t["classifier.fc.weight"].T + t["classifier.fc.bias"]).astype(np.float32)
    reference = np.asarray(module.forward(dict(params), x_uint8), dtype=np.float32)
    parity = float(np.max(np.abs(logits.astype(np.float64) - reference.astype(np.float64))))
    if not parity <= PARITY_TOL:
        raise NullspaceError(f"the replayed forward differs from forward.py by {parity} (> {PARITY_TOL})")
    return out, parity


def second_moment(inputs: np.ndarray) -> np.ndarray:
    """Uncentred second moment X^T X / N of the rows of ``inputs`` (float64)."""
    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise NullspaceError(f"inputs must be a non-empty [rows, d_in] matrix, got shape {x.shape}")
    return x.T @ x / x.shape[0]


def old_basis(tensor: str, moment: np.ndarray, fraction: float, n_rows: int) -> OldBasis:
    """The fewest top eigenvectors of ``moment`` whose eigenvalues sum to at least ``fraction`` of its trace."""
    if not 0.0 < fraction < 1.0:
        raise NullspaceError(f"energy fraction {fraction} is not in (0, 1)")
    values, vectors = np.linalg.eigh((moment + moment.T) / 2.0)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    vectors = vectors[:, order]
    total = float(values.sum())
    if total <= 0.0:
        raise NullspaceError(f"{tensor}: the input second moment has no energy")
    cumulative = np.cumsum(values) / total
    k = int(np.searchsorted(cumulative, fraction, side="left")) + 1
    k = min(k, len(values))
    return OldBasis(tensor=tensor, fraction=fraction, basis=vectors[:, :k], captured=float(cumulative[k - 1]),
                    d_in=int(moment.shape[0]), n_rows=int(n_rows))


def project_input_side(delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """``dW (I - U U^T)`` with ``dW`` read as [out, prod(rest)]; returned in the delta's shape, float64."""
    d = np.asarray(delta, dtype=np.float64)
    flat = d.reshape(d.shape[0], -1)
    if flat.shape[1] != basis.shape[0]:
        raise NullspaceError(f"delta input width {flat.shape[1]} != basis dimension {basis.shape[0]}")
    return (flat - (flat @ basis) @ basis.T).reshape(d.shape)


def edited_weight_tensors(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray]) -> list[str]:
    """Names of the tensors of rank >= 2 whose entries differ between ``base`` and ``child``, sorted."""
    if set(base) != set(child):
        raise NullspaceError("base and child have different tensor names")
    return sorted(k for k in base if np.asarray(base[k]).ndim >= 2 and not np.array_equal(base[k], child[k]))


def projected_child(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray],
                    bases: Mapping[str, OldBasis]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """``child`` with the delta of every tensor in ``bases`` projected (``project_input_side``), the rest unchanged.

    Each projected tensor is base + projected delta, computed in float64 and cast once to the tensor's dtype. The
    record gives per tensor k, d_in, the energy captured, the delta's Frobenius norm before and after, the norm of the
    removed part and the changed-entry counts before and after.
    """
    out = {k: np.asarray(v) for k, v in child.items()}
    record: dict[str, Any] = {}
    for tensor, ob in sorted(bases.items()):
        b = np.asarray(base[tensor])
        c = np.asarray(child[tensor])
        if b.ndim < 2:
            raise NullspaceError(f"{tensor} is 1-D; D1 leaves 1-D deltas unchanged")
        delta = c.astype(np.float64) - b.astype(np.float64)
        projected = project_input_side(delta, ob.basis)
        new = (b.astype(np.float64) + projected).astype(c.dtype)
        out[tensor] = new
        before, after = float(np.linalg.norm(delta)), float(np.linalg.norm(projected))
        record[tensor] = {
            "k": ob.k, "d_in": ob.d_in, "n_rows": ob.n_rows, "fraction": ob.fraction, "captured": ob.captured,
            "delta_frob": before, "projected_frob": after,
            "removed_frob": float(np.linalg.norm(delta - projected)),
            "kept_frob_ratio": after / before if before > 0 else None,
            "n_changed_before": int(np.count_nonzero(c != b)), "n_changed_after": int(np.count_nonzero(new != b)),
        }
    return out, record


def moments(module: ModuleType, params: Mapping[str, np.ndarray], x_old: np.ndarray, tensors: Collection[str],
            chunk: int = 32) -> tuple[dict[str, np.ndarray], dict[str, int], float]:
    """Second moments of the named tensors' layer inputs over ``x_old``, accumulated in chunks of images.

    Returns the moments, the row counts and the worst replay parity over the chunks.
    """
    sums: dict[str, np.ndarray] = {}
    rows: dict[str, int] = {}
    parity = 0.0
    for start in range(0, len(x_old), chunk):
        inputs, p = layer_inputs(module, params, x_old[start:start + chunk], tensors)
        parity = max(parity, p)
        for tensor, x in inputs.items():
            sums[tensor] = sums.get(tensor, 0.0) + x.T @ x
            rows[tensor] = rows.get(tensor, 0) + x.shape[0]
    return {t: sums[t] / rows[t] for t in sums}, rows, parity


__all__ = ["ENERGY_FRACTIONS", "NullspaceError", "OldBasis", "PATCH_WEIGHT", "edited_weight_tensors",
           "forward_module", "layer_inputs", "layer_of", "moments", "old_basis", "project_input_side",
           "projected_child", "second_moment"]
