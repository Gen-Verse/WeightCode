"""PREREG F6b: align a ViT-Lite-7/4 student to another by activation matching, and map an edit between their coordinates.

Two independently trained networks of one architecture compute the same kind of function in permuted unit coordinates.
The residual stream (256 units) has one permutation for the whole network; each MLP has a hidden-unit permutation
(512 units). Both are exact symmetries of the network: LayerNorm is equivariant when its weight and bias are permuted,
attention's inputs and outputs live on the residual side, and the MLP hidden layer is pointwise. Attention's internal
head coordinates are not permuted, since no final edits them.

``Alignment``: ``res[i]`` is the target unit matched to source unit ``i`` (a permutation of range(256)), and
``hid[b][i]`` the same for the 512 hidden units of block ``b`` (only for the blocks that were aligned).

* ``fit_alignment``: activation matching in the manner of Git Re-Basin. For source and target the same images go through
  the eval-form forward (the instance's own ``forward.py``, replayed with the activations captured and checked against
  ``forward`` on the first chunk). The residual permutation maximises, by the Hungarian method, the sum over the eight
  residual places (the input of each of the 7 blocks over all tokens, and the final norm's input, the class token) of the
  256 x 256 correlation matrices; each hidden permutation maximises the correlation of the post-activation hidden units
  of its block over all tokens. Reported: the mean matched correlation per place and the same mean for the identity.
* ``to_source`` / ``to_target``: the exact symmetry action on a parameter dict (every tensor is classified by name; an
  unknown tensor raises, so nothing is left unpermuted silently).
* ``map_delta``: an edit expressed in source coordinates, written in the target's coordinates: entry (i, j) of a source
  tensor goes to the target units matched to the units i and j.
* ``symmetry_gate``: the target permuted into source coordinates and back equals the target exactly, and the permuted
  target gives the target's own predictions (same argmax on every row, logits within a tolerance).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np

RESIDUAL_PLACES = 8
"""The input of each of the 7 blocks and the final norm's input."""
PARITY_TOL = 1e-6
GATE_LOGIT_TOL = 1e-4
"""Largest |logit(permuted target) - logit(target)| the symmetry gate allows (float32 logits of float64 arithmetic)."""
_VAR_FLOOR = 1e-12

_BLOCK = re.compile(r"^classifier\.blocks\.(\d+)\.(.+)$")
_RES0 = {"pre_norm.weight", "pre_norm.bias", "norm1.weight", "norm1.bias", "self_attn.proj.weight",
         "self_attn.proj.bias", "linear2.bias"}


class AlignError(RuntimeError):
    """A tensor, a forward file or a permutation does not fit the alignment as specified."""


@dataclass
class Alignment:
    """The matching of one target student to a source student (module docstring)."""

    res: np.ndarray
    hid: dict[int, np.ndarray]
    stats: dict[str, Any] = field(default_factory=dict)
    hid_corr: dict[int, np.ndarray] = field(default_factory=dict)
    """Per aligned block, the correlation of each source hidden unit with its matched target unit."""

    def inverse_res(self) -> np.ndarray:
        return _inverse(self.res)

    def inverse_hid(self, block: int) -> np.ndarray:
        return _inverse(self.hid[block])


def _inverse(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


# --------------------------------------------------------------------------------------------------------------------
# The symmetry: which axis of which tensor is a residual or a hidden coordinate


def axis_specs(name: str) -> list[tuple[int, str, int | None]]:
    """The (axis, kind, block) triples of a ViT-Lite tensor: kind "res" for a residual-stream axis, "hid" for the hidden
    axis of block ``block``'s MLP. A tensor with no such axis returns []; an unknown tensor raises."""
    if name in ("tokenizer.conv_layers.0.0.weight", "tokenizer.conv_layers.0.0.bias"):
        return [(0, "res", None)]
    if name in ("classifier.class_emb", "classifier.positional_emb"):
        return [(2, "res", None)]
    if name in ("classifier.norm.weight", "classifier.norm.bias"):
        return [(0, "res", None)]
    if name == "classifier.fc.weight":
        return [(1, "res", None)]
    if name == "classifier.fc.bias":
        return []
    match = _BLOCK.match(name)
    if match is None:
        raise AlignError(f"unknown tensor {name}")
    block, part = int(match.group(1)), match.group(2)
    if part in _RES0:
        return [(0, "res", None)]
    if part == "self_attn.qkv.weight":
        return [(1, "res", None)]
    if part == "linear1.weight":
        return [(0, "hid", block), (1, "res", None)]
    if part == "linear1.bias":
        return [(0, "hid", block)]
    if part == "linear2.weight":
        return [(0, "res", None), (1, "hid", block)]
    raise AlignError(f"unknown tensor {name}")


def _take(array: np.ndarray, name: str, res: np.ndarray, hid: Mapping[int, np.ndarray], strict: bool) -> np.ndarray:
    out = np.asarray(array)
    for axis, kind, block in axis_specs(name):
        if kind == "res":
            perm = res
        elif block in hid:
            perm = hid[block]
        elif strict:
            raise AlignError(f"{name} has a hidden axis of block {block}, which has no permutation")
        else:
            continue
        if out.shape[axis] != len(perm):
            raise AlignError(f"{name}: axis {axis} has size {out.shape[axis]}, the permutation {len(perm)}")
        out = np.take(out, perm, axis=axis)
    return out


def to_source(params: Mapping[str, np.ndarray], alignment: Alignment) -> dict[str, np.ndarray]:
    """The target's parameters in source coordinates: entry i of an axis is the target unit matched to source unit i."""
    return {k: _take(v, k, alignment.res, alignment.hid, strict=False) for k, v in params.items()}


def to_target(params: Mapping[str, np.ndarray], alignment: Alignment) -> dict[str, np.ndarray]:
    """The inverse action: parameters in source coordinates written in the target's coordinates."""
    inv_hid = {b: _inverse(p) for b, p in alignment.hid.items()}
    return {k: _take(v, k, _inverse(alignment.res), inv_hid, strict=False) for k, v in params.items()}


def map_delta(delta: Mapping[str, np.ndarray], alignment: Alignment) -> dict[str, np.ndarray]:
    """A delta in source coordinates written in the target's coordinates (float64 kept). Entry (i, j) of a source tensor
    goes to the target units matched to the units i and j. A delta tensor with a hidden axis of a block that was not
    aligned raises: the edit would sit on unmatched units."""
    inv_hid = {b: _inverse(p) for b, p in alignment.hid.items()}
    inv_res = _inverse(alignment.res)
    return {k: _take(np.asarray(v, dtype=np.float64), k, inv_res, inv_hid, strict=True) for k, v in delta.items()}


# --------------------------------------------------------------------------------------------------------------------
# The Hungarian method


def hungarian_max(score: np.ndarray) -> np.ndarray:
    """The assignment maximising the sum of ``score[i, assignment[i]]`` over a square matrix (Kuhn-Munkres with
    potentials, O(n^3); no dependency). Returns ``assignment`` with assignment[i] the column of row i."""
    score = np.asarray(score, dtype=np.float64)
    n = score.shape[0]
    if score.ndim != 2 or score.shape[1] != n:
        raise AlignError(f"the score matrix must be square, got {score.shape}")
    if not np.all(np.isfinite(score)):
        raise AlignError("the score matrix has a non-finite entry")
    cost = -score
    inf = np.inf
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=np.int64)  # p[j] = the row matched to column j (1-based, 0 = none)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            candidates = np.where(free, minv[1:], inf)
            j1 = int(np.argmin(candidates)) + 1
            delta = candidates[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.empty(n, dtype=np.int64)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


# --------------------------------------------------------------------------------------------------------------------
# Activations and their cross-correlation


def capture(module: ModuleType, params: Mapping[str, np.ndarray], x_uint8: np.ndarray, hidden_blocks: Sequence[int],
            check_parity: bool = False) -> dict[str, Any]:
    """Replay the instance's forward on ``x_uint8`` and return the residual-stream activations (a list of the 8 places,
    each [rows, 256]: the 7 block inputs over all tokens, then the class token before the final norm) and the
    post-activation hidden units of ``hidden_blocks`` ({block: [rows, 512]}). With ``check_parity`` the replayed logits
    are checked against ``module.forward``."""
    x = module._images(x_uint8)
    t = module._checked(dict(params))
    dim = t[module.PATCH_WEIGHT].shape[0]
    h = x.astype(np.float64) / 255.0
    h = (h - module.MEAN.reshape(1, -1, 1, 1)) / module.STD.reshape(1, -1, 1, 1)
    tokens = module._linear(module._patches(h), t[module.PATCH_WEIGHT].reshape(dim, -1), t[module.PATCH_BIAS])
    cls = np.broadcast_to(t["classifier.class_emb"], (x.shape[0], 1, dim))
    h = np.concatenate([cls, tokens], axis=1) + t["classifier.positional_emb"]
    places: list[np.ndarray] = []
    hidden_out: dict[int, np.ndarray] = {}
    for index in range(module.DEPTH):
        block = f"classifier.blocks.{index}"
        places.append(h.reshape(-1, dim))
        h = h + module._attention(module._layernorm(h, t[f"{block}.pre_norm.weight"], t[f"{block}.pre_norm.bias"]),
                                  t, block)
        h = module._layernorm(h, t[f"{block}.norm1.weight"], t[f"{block}.norm1.bias"])
        hidden = module._gelu(module._linear(h, t[f"{block}.linear1.weight"], t[f"{block}.linear1.bias"]))
        if index in hidden_blocks:
            hidden_out[index] = hidden.reshape(-1, hidden.shape[-1])
        h = h + module._linear(hidden, t[f"{block}.linear2.weight"], t[f"{block}.linear2.bias"])
    places.append(h[:, 0])
    if check_parity:
        feats = module._layernorm(h[:, 0], t["classifier.norm.weight"], t["classifier.norm.bias"])
        logits = (feats @ t["classifier.fc.weight"].T + t["classifier.fc.bias"]).astype(np.float32)
        reference = np.asarray(module.forward(dict(params), x_uint8), dtype=np.float32)
        parity = float(np.max(np.abs(logits.astype(np.float64) - reference.astype(np.float64))))
        if not parity <= PARITY_TOL:
            raise AlignError(f"the replayed forward differs from forward.py by {parity} (> {PARITY_TOL})")
    return {"places": places, "hidden": hidden_out}


class CrossMoments:
    """Running first and second moments of two activation sets over the same rows, for their correlation matrix."""

    def __init__(self, dim_a: int, dim_b: int) -> None:
        self.n = 0
        self.sa, self.sb = np.zeros(dim_a), np.zeros(dim_b)
        self.saa, self.sbb = np.zeros(dim_a), np.zeros(dim_b)
        self.sab = np.zeros((dim_a, dim_b))

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        if a.shape[0] != b.shape[0]:
            raise AlignError("the two activation sets must have the same rows")
        self.n += a.shape[0]
        self.sa += a.sum(0)
        self.sb += b.sum(0)
        self.saa += (a * a).sum(0)
        self.sbb += (b * b).sum(0)
        self.sab += a.T @ b

    def correlation(self) -> np.ndarray:
        """corr[i, j] between source unit i and target unit j; 0 where either is constant."""
        ma, mb = self.sa / self.n, self.sb / self.n
        cov = self.sab / self.n - np.outer(ma, mb)
        va = np.maximum(self.saa / self.n - ma * ma, 0.0)
        vb = np.maximum(self.sbb / self.n - mb * mb, 0.0)
        denom = np.sqrt(np.outer(va, vb))
        corr = np.zeros_like(cov)
        ok = (va[:, None] > _VAR_FLOOR) & (vb[None, :] > _VAR_FLOOR)
        corr[ok] = cov[ok] / denom[ok]
        return np.clip(corr, -1.0, 1.0)


def _matched_mean(corr: np.ndarray, assignment: np.ndarray) -> float:
    return float(np.mean(corr[np.arange(len(assignment)), assignment]))


def fit_alignment(module: ModuleType, source: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray],
                  x_uint8: np.ndarray, hidden_blocks: Sequence[int], chunk: int = 64) -> Alignment:
    """Match ``target`` to ``source`` on the images ``x_uint8`` (module docstring)."""
    dim = int(np.asarray(source["classifier.norm.weight"]).shape[0])
    hidden = int(np.asarray(source["classifier.blocks.0.linear1.bias"]).shape[0])
    places = [CrossMoments(dim, dim) for _ in range(RESIDUAL_PLACES)]
    hids = {b: CrossMoments(hidden, hidden) for b in hidden_blocks}
    for start in range(0, len(x_uint8), chunk):
        xs = x_uint8[start:start + chunk]
        a = capture(module, source, xs, hidden_blocks, check_parity=start == 0)
        b = capture(module, target, xs, hidden_blocks, check_parity=start == 0)
        for moments, pa, pb in zip(places, a["places"], b["places"]):
            moments.add(pa, pb)
        for block, moments in hids.items():
            moments.add(a["hidden"][block], b["hidden"][block])
    corr_places = [m.correlation() for m in places]
    total = np.sum(corr_places, axis=0)
    res = hungarian_max(total)
    identity = np.arange(dim)
    stats: dict[str, Any] = {
        "rows": int(len(x_uint8)), "residual_places": {}, "hidden": {},
        "residual_sum_matched": float(total[np.arange(dim), res].sum()),
        "residual_sum_identity": float(total[identity, identity].sum()),
        "residual_fixed_points": int(np.sum(res == identity)),
    }
    for i, corr in enumerate(corr_places):
        name = "final_norm_input" if i == RESIDUAL_PLACES - 1 else f"block{i}_input"
        stats["residual_places"][name] = {"matched": _matched_mean(corr, res), "identity": _matched_mean(corr, identity),
                                          "matched_min": float(corr[np.arange(dim), res].min())}
    hid_perm: dict[int, np.ndarray] = {}
    hid_corr: dict[int, np.ndarray] = {}
    for block, moments in hids.items():
        corr = moments.correlation()
        hid_perm[block] = hungarian_max(corr)
        stats["hidden"][f"block{block}"] = {"matched": _matched_mean(corr, hid_perm[block]),
                                            "identity": _matched_mean(corr, np.arange(hidden)),
                                            "matched_min": float(corr[np.arange(hidden), hid_perm[block]].min()),
                                            "fixed_points": int(np.sum(hid_perm[block] == np.arange(hidden)))}
        hid_corr[block] = corr[np.arange(hidden), hid_perm[block]]
    return Alignment(res=res, hid=hid_perm, stats=stats, hid_corr=hid_corr)


# --------------------------------------------------------------------------------------------------------------------
# The validity gate


def compare_logits(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return {"rows": int(len(a)), "max_abs_logit_diff": float(np.max(np.abs(a - b))) if len(a) else 0.0,
            "argmax_disagreements": int(np.count_nonzero(a.argmax(1) != b.argmax(1)))}


def symmetry_gate(module: ModuleType, target: Mapping[str, np.ndarray], alignment: Alignment,
                  images: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The validity gate of PREREG F6b. ``images``: name -> uint8 rows (the condition V_select and the clean V_select).

    * the target permuted into source coordinates and back equals the target entry for entry;
    * the permuted target has the same argmax as the target on every row, and logits within ``GATE_LOGIT_TOL``.
    ``passed`` is the conjunction; a failed gate is an implementation defect, not a result."""
    moved = to_source(target, alignment)
    back = to_target(moved, alignment)
    roundtrip = all(np.array_equal(np.asarray(back[k]), np.asarray(target[k])) for k in target)
    sets = {}
    for name, x in images.items():
        sets[name] = compare_logits(module.forward(dict(target), x), module.forward(dict(moved), x))
    passed = bool(roundtrip and all(s["argmax_disagreements"] == 0 and s["max_abs_logit_diff"] <= GATE_LOGIT_TOL
                                    for s in sets.values()))
    return {"roundtrip_exact": bool(roundtrip), "sets": sets, "logit_tolerance": GATE_LOGIT_TOL, "passed": passed}


__all__ = ["Alignment", "AlignError", "CrossMoments", "GATE_LOGIT_TOL", "RESIDUAL_PLACES", "axis_specs", "capture",
           "compare_logits", "fit_alignment", "hungarian_max", "map_delta", "symmetry_gate", "to_source", "to_target"]
