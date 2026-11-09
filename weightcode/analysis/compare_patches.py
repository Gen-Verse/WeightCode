"""Compare the final child of a run with that of a reference run on one dev instance: is the run more than a channel tie?

Dev-only analysis. Both runs must name the same dev instance; main, superseded and stale instances are refused by the
dev readout's own check (``weightcode.evalkit.devreadout``). The only data read is V_select of the instance's deployment
condition, through ``devreadout.readout_split``; ``$WEIGHTCODE_SEALED`` is never opened. The counterfactual children of the
decomposition exist only in memory: they are never written as patches and never shown to an editor. The numbers are
development evidence, not confirmation.

Three views of the run (RUN) against the reference (REF), usually the label-free ``tie_inputs`` rule:

1. Function space on V_select: accuracy and the ``split_readouts`` fields of each child, the paired difference
   run minus ref, the overlap of the fixed and broken sets, and prediction agreement.
2. Weight space on the first layer with an input-channel axis. With W the base weight viewed as (out, C, a, b) and
   D = W - mean_c W, a pure channel tie at strength s gives the delta -s D exactly.

   - ``channel_difference_share = ||delta - mean_c delta||^2 / ||delta||^2``: the share of the edit that changes only
     channel differences. Any tie, however uneven, lies wholly in this part.
   - Local tie fits: one strength per output unit, per spatial position (pixel, or kernel position for a conv) and
     per (unit, position), ``s_g = -<delta_g, D_g> / ||D_g||^2``, with the share of ``||delta||^2`` each fit explains,
     the share explained with tie sign only (s_g >= 0), the share an isotropic random delta would give by chance,
     and quantiles of s_g. An uneven or partial tie reads as a tie here.
   - Uniform global tie: one strength for the whole layer, ``s_eff = -<delta, D> / ||D||^2``, the residual
     ``R = delta + s_eff D``, ``tie_fraction = 1 - ||R||^2 / ||delta||^2`` (which equals cos^2(delta, D), so it is
     sign-blind: ``signed_tie_fraction`` and the ``anti_tie`` flag carry the sign), ``cos(delta, -D)`` and
     ``channel_spread_ratio = ||W_child - mean_c W_child|| / ||D||`` (1 - s for a pure tie). A tie that acts on only
     part of the layer scores low here although it is all tie; read the local fits first.
3. Decomposition of the run, each part read out on V_select: the uniform global tie part (base with only its first
   layer moved by ``-s_eff D``) and residual (the run's child with its first-layer delta replaced by R), and the local
   tie part (base with its first layer moved by ``-max(s_g, 0) D_g`` per (unit, position)) and its residual.

Usage::

    python -m weightcode.analysis.compare_patches --run RUN --reference REF [--out FILE]

RUN and REF are run.json files or directories holding exactly one. With ``--out`` the JSON record goes to FILE and a
short table to stdout; without it the JSON record is printed as the first stdout line, followed by the table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import Params
from weightcode.evalkit.devreadout import (
    DEFAULT_BATCH_SIZE,
    READOUT_SPLIT,
    _check_child,
    _check_dev_instance,
    final_child,
    readout_split,
    run_files,
)
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.patch_check import predict
from weightcode.evalkit.sealed import split_readouts
from weightcode.evalkit.targets import file_sha256, read_json
from weightcode.models.spec import FLATTEN_ORDER
from weightcode.patch import load_patch

SCHEMA = "weightcode.analysis.compare_patches.v0"
INPUT_LAYOUT = "NCHW, flatten row-major"
CHANNEL_AXIS = 1
SCALAR_OP_FIELDS = ("axis", "strength", "factor", "value", "side")

FLAG_DELTA_ZERO = "delta_zero"
FLAG_CHANNEL_DIFFERENCE_ZERO = "channel_difference_zero"
FLAG_ANTI_TIE = "anti_tie"

SPATIAL_AXES = (2, 3)
LOCAL_FITS: dict[str, tuple[int, ...]] = {"per_unit": (0,), "per_spatial": SPATIAL_AXES, "per_unit_spatial": (0, *SPATIAL_AXES)}
DECOMPOSITION_LOCAL_FIT = "per_unit_spatial"
S_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


class CompareError(RuntimeError):
    """The comparison was refused: mismatched runs, an ambiguous first layer, or a missing record."""


@dataclass(frozen=True)
class FirstLayer:
    """The first parameterised layer, viewed with its input-channel axis at position 1."""

    name: str
    op: str
    tensor: str
    view_shape: tuple[int, int, int, int]
    input_shape: tuple[int, int, int]

    @property
    def n_channels(self) -> int:
        """Number of input channels C."""
        return self.view_shape[CHANNEL_AXIS]


# ----------------------------------------------------------------------------------------------- first layer


def first_layer(spec: Mapping[str, Any], params: Mapping[str, np.ndarray]) -> FirstLayer:
    """Resolve the first layer with an input-channel axis from a public model spec; raise if it is ambiguous.

    A ``linear`` first layer must read the flattened input (in_features = C * H * W, C, H, W flatten order) and is
    viewed as (out, C, H, W). A ``conv2d`` first layer must have C input channels and is viewed as is.
    """
    shape = spec.get("input", {}).get("shape")
    if not isinstance(shape, list) or len(shape) != 3 or not all(isinstance(d, int) and d > 0 for d in shape):
        raise CompareError(f"model spec input.shape must be three positive integers (C, H, W), got {shape!r}")
    c, h, w = shape
    layers = spec.get("layers")
    if not isinstance(layers, list):
        raise CompareError("model spec has no layer list")
    position = next((i for i, layer in enumerate(layers) if isinstance(layer, Mapping) and "weight" in layer), None)
    if position is None:
        raise CompareError("model spec has no layer with a weight tensor")
    layer = layers[position]
    tensor = layer["weight"]
    if tensor not in params:
        raise CompareError(f"first layer weight {tensor!r} is not a parameter; parameters: {sorted(params)}")
    weight_shape = tuple(int(d) for d in params[tensor].shape)
    if list(weight_shape) != list(layer.get("shape_w", [])):
        raise CompareError(f"{tensor} has shape {weight_shape}, the spec says {layer.get('shape_w')}")
    op = layer.get("op")
    if op == "linear":
        if len(weight_shape) != 2 or weight_shape[1] != c * h * w:
            raise CompareError(f"linear first layer {tensor} {weight_shape} does not read the flat input {shape}")
        if spec["input"].get("layout") != INPUT_LAYOUT:
            raise CompareError(f"input layout {spec['input'].get('layout')!r} is not {INPUT_LAYOUT!r}")
        flatten = [x for x in layers[:position] if isinstance(x, Mapping) and x.get("op") == "flatten"]
        if any(x.get("order") != FLATTEN_ORDER for x in flatten):
            raise CompareError(f"a flatten before {tensor} does not use order {FLATTEN_ORDER!r}")
        view = (weight_shape[0], c, h, w)
    elif op == "conv2d":
        if len(weight_shape) != 4 or weight_shape[1] != c:
            raise CompareError(f"conv2d first layer {tensor} {weight_shape} does not take {c} input channels")
        view = (weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3])
    else:
        raise CompareError(f"first layer op {op!r} has no input-channel axis this tool knows (linear, conv2d)")
    return FirstLayer(str(layer.get("name", tensor)), str(op), str(tensor), view, (c, h, w))


def instance_first_layer(instance: Instance) -> FirstLayer:
    """``first_layer`` of the instance's model_spec.json, checked against the hash private.json records."""
    path = instance.instance_dir / "model_spec.json"
    recorded = instance.private.get("model_spec_sha256")
    if recorded is None:
        raise CompareError(f"{instance.instance_id}: private.json records no model_spec_sha256")
    if file_sha256(path) != recorded:
        raise CompareError(f"{path}: hash differs from private.json model_spec_sha256")
    return first_layer(read_json(path), instance.params)


# ----------------------------------------------------------------------------------------------- weight space


def _view64(array: np.ndarray, layer: FirstLayer) -> np.ndarray:
    return np.asarray(array, dtype=np.float64).reshape(layer.view_shape)


def channel_difference(weight: np.ndarray, layer: FirstLayer) -> np.ndarray:
    """D = W - mean over the channel axis (broadcast), in float64, in the layer's (out, C, a, b) view."""
    view = _view64(weight, layer)
    return view - view.mean(axis=CHANNEL_AXIS, keepdims=True)


@dataclass(frozen=True)
class _Projection:
    delta: np.ndarray
    d: np.ndarray
    s_eff: float
    residual: np.ndarray
    flags: tuple[str, ...]


def _project(base_weight: np.ndarray, child_weight: np.ndarray, layer: FirstLayer) -> _Projection:
    """Project the first-layer delta onto -D (one uniform global strength). With ||D|| = 0 the tie direction is
    empty: s_eff is 0, flagged. A negative s_eff with a nonzero delta (the edit widens channel differences) is flagged
    ``anti_tie``."""
    delta = _view64(child_weight, layer) - _view64(base_weight, layer)
    d = channel_difference(base_weight, layer)
    d_sq = float(np.sum(d * d))
    flags: list[str] = []
    if d_sq == 0.0:
        flags.append(FLAG_CHANNEL_DIFFERENCE_ZERO)
        s_eff = 0.0
    else:
        s_eff = -float(np.sum(delta * d)) / d_sq
    if not np.any(delta):
        flags.append(FLAG_DELTA_ZERO)
    elif s_eff < 0.0:
        flags.append(FLAG_ANTI_TIE)
    return _Projection(delta, d, s_eff, delta + s_eff * d, tuple(flags))


def _reduce_to(array: np.ndarray, keep: tuple[int, ...]) -> np.ndarray:
    """Sum over every axis not in ``keep``, keeping dims so the result broadcasts back onto the (out, C, a, b) view."""
    return array.sum(axis=tuple(i for i in range(array.ndim) if i not in keep), keepdims=True)


def local_strengths(delta: np.ndarray, d: np.ndarray, keep: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Per-group tie strengths ``s_g = -<delta_g, D_g> / ||D_g||^2``, groups indexed by the ``keep`` axes.

    Returns (s, defined), both broadcastable onto the view; s is 0 where ``||D_g|| = 0`` (``defined`` False there).
    """
    num = _reduce_to(delta * d, keep)
    den = _reduce_to(d * d, keep)
    defined = den > 0.0
    s = np.zeros_like(num)
    np.divide(-num, den, out=s, where=defined)
    return s, defined


def local_tie_fit(delta: np.ndarray, d: np.ndarray, keep: tuple[int, ...]) -> dict[str, Any]:
    """How much of a first-layer delta a channel tie with one strength per group explains.

    ``explained_fraction`` is the least-squares share of ``||delta||^2`` (signed s_g); ``explained_fraction_tie_sign``
    keeps only groups with s_g > 0, the best fit by a non-negative, possibly uneven tie. ``isotropic_chance`` is the
    expected explained share for an isotropic random delta (defined groups / entries). Shares are None for a zero
    delta; quantiles are None when no group has ``||D_g|| > 0``.
    """
    s, defined = local_strengths(delta, d, keep)
    num = _reduce_to(delta * d, keep)
    den = _reduce_to(d * d, keep)
    gain = np.zeros_like(num)
    np.divide(num * num, den, out=gain, where=defined)
    delta_sq = float(np.sum(delta * delta))
    values = s[defined]
    n_defined = int(values.size)
    record: dict[str, Any] = {
        "n_groups": int(s.size),
        "n_groups_defined": n_defined,
        "explained_fraction": float(gain.sum()) / delta_sq if delta_sq > 0.0 else None,
        "explained_fraction_tie_sign": float(gain[s > 0.0].sum()) / delta_sq if delta_sq > 0.0 else None,
        "isotropic_chance": n_defined / int(delta.size),
        "s_quantiles": None,
        "share_groups_negative": None,
        "share_groups_above_one": None,
    }
    if n_defined:
        record["s_quantiles"] = {f"q{round(100 * q):02d}": float(v) for q, v in zip(S_QUANTILES, np.quantile(values, S_QUANTILES))}
        record["share_groups_negative"] = float(np.mean(values < 0.0))
        record["share_groups_above_one"] = float(np.mean(values > 1.0))
    return record


def tie_geometry(base_weight: np.ndarray, child_weight: np.ndarray, layer: FirstLayer) -> dict[str, Any]:
    """Uniform global tie geometry of one first-layer delta, with its channel-difference share and local tie fits.

    The top-level s_eff, tie_fraction, signed_tie_fraction, cos(delta, -D) and channel_spread_ratio describe one
    strength for the whole layer. tie_fraction equals cos^2(delta, D) and is blind to sign; signed_tie_fraction is
    sign(s_eff) * tie_fraction and flag ``anti_tie`` marks s_eff < 0. tie_fraction is 0 with flag ``delta_zero`` when
    the delta is 0 (channel_difference_share is then None). With flag ``channel_difference_zero`` (for example C = 1)
    s_eff is 0, and cos and channel_spread_ratio are None because they divide by ||D||.
    """
    p = _project(base_weight, child_weight, layer)
    delta_sq = float(np.sum(p.delta * p.delta))
    d_norm = float(np.sqrt(np.sum(p.d * p.d)))
    tie_fraction = 0.0 if delta_sq == 0.0 else 1.0 - float(np.sum(p.residual * p.residual)) / delta_sq
    delta_diff = p.delta - p.delta.mean(axis=CHANNEL_AXIS, keepdims=True)
    diff_share = float(np.sum(delta_diff * delta_diff)) / delta_sq if delta_sq > 0.0 else None
    child_view = _view64(child_weight, layer)
    spread = float(np.linalg.norm(child_view - child_view.mean(axis=CHANNEL_AXIS, keepdims=True)))
    defined = d_norm > 0.0
    cos = None
    if defined and delta_sq > 0.0:
        cos = -float(np.sum(p.delta * p.d)) / (math.sqrt(delta_sq) * d_norm)
    return {
        "channel_difference_share": diff_share,
        "local": {name: local_tie_fit(p.delta, p.d, keep) for name, keep in LOCAL_FITS.items()},
        "fit": "uniform global tie",
        "s_eff": p.s_eff,
        "tie_fraction": tie_fraction,
        "signed_tie_fraction": math.copysign(tie_fraction, p.s_eff) if p.s_eff != 0.0 else 0.0,
        "cos_delta_minus_d": cos,
        "channel_spread_ratio": spread / d_norm if defined else None,
        "delta_norm": math.sqrt(delta_sq),
        "residual_norm": float(np.linalg.norm(p.residual)),
        "d_norm": d_norm,
        "flags": list(p.flags),
    }


def _deltas(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.asarray(child[k], dtype=np.float64) - np.asarray(base[k], dtype=np.float64) for k in sorted(base)}


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return None if na == 0.0 or nb == 0.0 else float(np.sum(a * b)) / (na * nb)


def weight_space(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], layer: FirstLayer) -> dict[str, Any]:
    """Tie geometry of the first layer, the first layer's share of the squared delta, and per-tensor delta norms.

    A relative norm of a tensor whose base is all zero is None (flagged in ``relative_norm_undefined``).
    """
    deltas = _deltas(base, child)
    squared = {k: float(np.sum(v * v)) for k, v in deltas.items()}
    total = sum(squared.values())
    tensors: dict[str, dict[str, Any]] = {}
    undefined: list[str] = []
    for key, delta in deltas.items():
        base_norm = float(np.linalg.norm(np.asarray(base[key], dtype=np.float64)))
        norm = math.sqrt(squared[key])
        if base_norm == 0.0:
            relative = 0.0 if norm == 0.0 else None
            if relative is None:
                undefined.append(key)
        else:
            relative = norm / base_norm
        tensors[key] = {"delta_norm": norm, "base_norm": base_norm, "relative_norm": relative,
                        "changed_entries": int(np.count_nonzero(delta))}
    return {
        "first_layer": layer.tensor,
        "tie": tie_geometry(base[layer.tensor], child[layer.tensor], layer),
        "first_layer_share_of_squared_delta": squared[layer.tensor] / total if total > 0.0 else None,
        "total_delta_norm": math.sqrt(total),
        "tensors": tensors,
        "relative_norm_undefined": undefined,
    }


def delta_cosines(
    base: Mapping[str, np.ndarray], run: Mapping[str, np.ndarray], ref: Mapping[str, np.ndarray], layer: FirstLayer
) -> dict[str, float | None]:
    """Cosine between the run's and the ref's deltas over all parameters and over the first layer (None if a delta is 0)."""
    a, b = _deltas(base, run), _deltas(base, ref)
    flat_a = np.concatenate([v.ravel() for v in a.values()])
    flat_b = np.concatenate([v.ravel() for v in b.values()])
    return {"all_parameters": _cosine(flat_a, flat_b), "first_layer": _cosine(a[layer.tensor], b[layer.tensor])}


def _split(
    base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], layer: FirstLayer, tie_delta: np.ndarray,
    delta: np.ndarray,
) -> tuple[Params, Params]:
    """(base + tie_delta on the first layer, child with its first-layer delta replaced by delta - tie_delta)."""
    w64 = _view64(base[layer.tensor], layer)
    shape, dtype = base[layer.tensor].shape, base[layer.tensor].dtype
    tie_part = {k: np.array(v, copy=True) for k, v in base.items()}
    tie_part[layer.tensor] = (w64 + tie_delta).reshape(shape).astype(dtype)
    residual = {k: np.array(v, copy=True) for k, v in child.items()}
    residual[layer.tensor] = (w64 + (delta - tie_delta)).reshape(shape).astype(dtype)
    return tie_part, residual


def decompose(
    base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], layer: FirstLayer
) -> tuple[Params, Params, float]:
    """(tie part, residual, s_eff) of a child under the uniform global tie, in memory only.

    The tie part is the base with only its first layer moved by -s_eff D. The residual is the child with its
    first-layer delta replaced by R = delta + s_eff D; every other tensor keeps the child's delta. Arithmetic is
    float64 from the float32 tensors and each changed tensor is cast to float32 once, like the executor.
    """
    p = _project(base[layer.tensor], child[layer.tensor], layer)
    tie_part, residual = _split(base, child, layer, -p.s_eff * p.d, p.delta)
    return tie_part, residual, p.s_eff


def decompose_local(
    base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], layer: FirstLayer,
    keep: tuple[int, ...] = LOCAL_FITS[DECOMPOSITION_LOCAL_FIT],
) -> tuple[Params, Params]:
    """(local tie part, local residual) of a child, in memory only, with one non-negative strength per group.

    The tie part is the base with its first layer moved by -max(s_g, 0) D_g, the best fit by a channel tie that may
    act unevenly but never widens channel differences. The residual is the child with that tie removed from its
    first-layer delta; other tensors keep the child's delta. Same float64 arithmetic and single cast as ``decompose``.
    """
    p = _project(base[layer.tensor], child[layer.tensor], layer)
    s, _ = local_strengths(p.delta, p.d, keep)
    return _split(base, child, layer, -np.maximum(s, 0.0) * p.d, p.delta)


# ----------------------------------------------------------------------------------------------- function space


def _agreement(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    if mask is not None:
        a, b = a[mask], b[mask]
    return None if len(a) == 0 else float(np.mean(a == b))


def _sets(y: np.ndarray, base_pred: np.ndarray, child_pred: np.ndarray) -> tuple[set[int], set[int]]:
    base_ok, child_ok = base_pred == y, child_pred == y
    return set(np.flatnonzero(~base_ok & child_ok).tolist()), set(np.flatnonzero(base_ok & ~child_ok).tolist())


def _overlap(run: set[int], ref: set[int]) -> dict[str, Any]:
    both, union = len(run & ref), len(run | ref)
    return {
        "run": len(run),
        "ref": len(ref),
        "intersection": both,
        "jaccard": both / union if union else None,
        "intersection_over_run": both / len(run) if run else None,
    }


def function_space(
    data: Mapping[str, np.ndarray],
    base_pred: np.ndarray,
    run_pred: np.ndarray,
    ref_pred: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    """Readouts of run and ref against base, their paired difference, fixed and broken set overlap, and agreement.

    Jaccard and intersection_over_run are None when their denominator set is empty; agreement on base errors is
    None when the base makes none.
    """
    y = data["y"]
    n = int(len(y))
    run_ok, ref_ok = run_pred == y, ref_pred == y
    run_only, ref_only = int(np.sum(run_ok & ~ref_ok)), int(np.sum(~run_ok & ref_ok))
    run_fixed, run_broken = _sets(y, base_pred, run_pred)
    ref_fixed, ref_broken = _sets(y, base_pred, ref_pred)
    base_wrong = base_pred != y
    run_readouts = split_readouts(data, base_pred, run_pred, None, n_classes)
    ref_readouts = split_readouts(data, base_pred, ref_pred, None, n_classes)
    base_record: dict[str, Any] = {"accuracy": float(np.mean(base_pred == y))}
    if "base_cfr" in run_readouts:
        base_record["cfr"] = run_readouts["base_cfr"]
    return {
        "n": n,
        "base": base_record,
        "run": {"accuracy": float(np.mean(run_ok)), "readouts": run_readouts},
        "ref": {"accuracy": float(np.mean(ref_ok)), "readouts": ref_readouts},
        "paired_run_minus_ref": {
            "ref_wrong_run_right": run_only,
            "ref_right_run_wrong": ref_only,
            "difference": (run_only - ref_only) / n,
        },
        "fixed": _overlap(run_fixed, ref_fixed),
        "broken": _overlap(run_broken, ref_broken),
        "agreement_run_ref": {
            "all": _agreement(run_pred, ref_pred),
            "base_errors": _agreement(run_pred, ref_pred, base_wrong),
            "n_base_errors": int(np.sum(base_wrong)),
        },
    }


# ----------------------------------------------------------------------------------------------- runs


def _single_run_file(path: str | Path) -> Path:
    found = run_files([path])
    if len(found) != 1:
        raise CompareError(f"{path}: expected exactly one run.json, found {len(found)}")
    return found[0]


def op_summary(run: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Op names, tensors and counts of the run's final patch.json when ``final.parent`` is "instance"."""
    final = run["final"]
    if final.get("is_base_noop"):
        return {"applicable": False, "reason": "not applicable: base no-op final"}
    parent = final.get("parent", "instance")
    if parent != "instance":
        return {"applicable": False, "reason": f"not applicable: final.parent is {parent!r}, a saved child"}
    patch = load_patch(run_dir / str(final["dir"]) / "patch.json")
    ops = []
    for raw in patch.get("ops", []):
        entry: dict[str, Any] = {"op": raw.get("op"), "tensor": raw.get("tensor")}
        entry.update({k: raw[k] for k in SCALAR_OP_FIELDS if k in raw and not isinstance(raw[k], (list, dict))})
        ops.append(entry)
    return {
        "applicable": True,
        "n_ops": len(ops),
        "by_op": dict(sorted(Counter(str(o["op"]) for o in ops).items())),
        "by_op_tensor": dict(sorted(Counter(f"{o['op']}:{o['tensor']}" for o in ops).items())),
        "ops": ops,
    }


def compare_runs(run_path: str | Path, ref_path: str | Path, batch_size: int = DEFAULT_BATCH_SIZE) -> dict[str, Any]:
    """The full comparison record of a run against a reference run on the same dev instance."""
    run_file, ref_file = _single_run_file(run_path), _single_run_file(ref_path)
    run, ref = json.loads(run_file.read_text()), json.loads(ref_file.read_text())
    run_id, ref_id = run["instance"]["instance_id"], ref["instance"]["instance_id"]
    if run_id != ref_id:
        raise CompareError(f"run names instance {run_id}, reference names {ref_id}; both must name the same one")
    instance = load_instance(instances_root() / run_id)
    _check_dev_instance(instance)
    layer = instance_first_layer(instance)
    run_child = final_child(run, run_file.parent, instance)
    ref_child = final_child(ref, ref_file.parent, instance)
    _check_child(instance, run_child)
    _check_child(instance, ref_child)
    tie_part, residual, _ = decompose(instance.params, run_child, layer)
    tie_local, residual_local = decompose_local(instance.params, run_child, layer)
    for counterfactual in (tie_part, residual, tie_local, residual_local):
        _check_child(instance, counterfactual)
    data = readout_split(instance)
    n_classes = len(instance.task["classes"])

    def pred(params: Mapping[str, np.ndarray]) -> np.ndarray:
        return predict(instance.forward, params, data["x"], batch_size)

    base_pred, run_pred, ref_pred = pred(instance.params), pred(run_child), pred(ref_child)
    tie_pred, residual_pred = pred(tie_part), pred(residual)
    tie_local_pred, residual_local_pred = pred(tie_local), pred(residual_local)
    y = data["y"]
    decomposition: dict[str, Any] = {"fit": "uniform global tie", "local_fit": f"{DECOMPOSITION_LOCAL_FIT}, s_g >= 0"}
    for name, p in (("child_tie_part", tie_pred), ("child_residual", residual_pred),
                    ("child_tie_part_local", tie_local_pred), ("child_residual_local", residual_local_pred)):
        decomposition[name] = {"accuracy": float(np.mean(p == y)), "readouts": split_readouts(data, base_pred, p, None, n_classes)}
    decomposition["agreement_tie_part_with_run"] = _agreement(tie_pred, run_pred)
    decomposition["agreement_residual_with_base"] = _agreement(residual_pred, base_pred)
    decomposition["agreement_tie_part_local_with_run"] = _agreement(tie_local_pred, run_pred)
    decomposition["agreement_residual_local_with_base"] = _agreement(residual_local_pred, base_pred)
    return {
        "schema": SCHEMA,
        "instance_id": instance.instance_id,
        "dataset": instance.private["dataset"],
        "condition": instance.private["condition"],
        "split": READOUT_SPLIT,
        "evidence": "development",
        "run": {"path": str(run_file), "method": run.get("method"), "is_base_noop": bool(run["final"].get("is_base_noop"))},
        "ref": {"path": str(ref_file), "method": ref.get("method"), "is_base_noop": bool(ref["final"].get("is_base_noop"))},
        "first_layer": {"name": layer.name, "op": layer.op, "tensor": layer.tensor, "view_shape": list(layer.view_shape),
                        "input_shape": list(layer.input_shape)},
        "function_space": function_space(data, base_pred, run_pred, ref_pred, n_classes),
        "weight_space": {
            "run": weight_space(instance.params, run_child, layer),
            "ref": weight_space(instance.params, ref_child, layer),
            "cosine_run_ref": delta_cosines(instance.params, run_child, ref_child, layer),
        },
        "decomposition": decomposition,
        "op_summary": op_summary(run, run_file.parent),
    }


# ----------------------------------------------------------------------------------------------- output


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def format_table(record: Mapping[str, Any]) -> str:
    """Short plain-text summary of a comparison record."""
    fs, ws, dc = record["function_space"], record["weight_space"], record["decomposition"]
    rows = [("base", fs["base"]["accuracy"], None, fs["base"].get("cfr"))]
    for name, block in (("run", fs["run"]), ("ref", fs["ref"]), ("tie_local", dc["child_tie_part_local"]),
                        ("resid_local", dc["child_residual_local"]), ("tie_global", dc["child_tie_part"]),
                        ("resid_global", dc["child_residual"])):
        ro = block["readouts"]
        rows.append((name, block["accuracy"], ro["delta_acc"], ro.get("child_cfr")))
    lines = [f"{record['instance_id']} {record['split']} n {fs['n']} (development evidence)",
             f"{'child':<13}{'acc':>9}{'d_acc':>9}{'cfr':>9}"]
    lines += [f"{n:<13}{_fmt(a):>9}{_fmt(d, True):>9}{_fmt(c):>9}" for n, a, d, c in rows]
    lines.append(f"tie_local: {dc['local_fit']}; tie_global: one strength for the whole first layer")
    paired = fs["paired_run_minus_ref"]
    lines.append(f"paired run-ref {paired['difference']:+.4f} (+{paired['ref_wrong_run_right']} "
                 f"-{paired['ref_right_run_wrong']}); agreement all {_fmt(fs['agreement_run_ref']['all'])}, "
                 f"base errors {_fmt(fs['agreement_run_ref']['base_errors'])}")
    fx = fs["fixed"]
    lines.append(f"fixed run {fx['run']} ref {fx['ref']} both {fx['intersection']} jaccard {_fmt(fx['jaccard'])} "
                 f"both/run {_fmt(fx['intersection_over_run'])}")
    for name in ("run", "ref"):
        t = ws[name]["tie"]
        lines.append(f"{name} {ws[name]['first_layer']}: share of squared delta "
                     f"{_fmt(ws[name]['first_layer_share_of_squared_delta'])}, channel-difference share "
                     f"{_fmt(t['channel_difference_share'])}")
        for fit, lf in t["local"].items():
            q = lf["s_quantiles"] or {}
            lines.append(f"  local {fit:<17} explained {_fmt(lf['explained_fraction'])} tie-sign "
                         f"{_fmt(lf['explained_fraction_tie_sign'])} chance {_fmt(lf['isotropic_chance'])} "
                         f"s q05/q50/q95 {_fmt(q.get('q05'), True)}/{_fmt(q.get('q50'), True)}/{_fmt(q.get('q95'), True)}")
        lines.append(f"  uniform global tie: s_eff {_fmt(t['s_eff'], True)} tie_fraction=cos^2 "
                     f"{_fmt(t['tie_fraction'])} signed {_fmt(t['signed_tie_fraction'], True)} cos "
                     f"{_fmt(t['cos_delta_minus_d'], True)} spread {_fmt(t['channel_spread_ratio'])}"
                     + (f" flags {','.join(t['flags'])}" if t["flags"] else ""))
    cos = ws["cosine_run_ref"]
    lines.append(f"cos(run, ref) all {_fmt(cos['all_parameters'], True)} first layer {_fmt(cos['first_layer'], True)}; "
                 f"agreement with run: tie_local {_fmt(dc['agreement_tie_part_local_with_run'])} "
                 f"tie_global {_fmt(dc['agreement_tie_part_with_run'])}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m weightcode.analysis.compare_patches", description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="run.json, or a directory holding one, of the run to analyse")
    parser.add_argument("--reference", required=True, help="run.json, or a directory holding one, of the reference")
    parser.add_argument("--out", type=Path, help="write the JSON record to this file")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)
    record = compare_runs(args.run, args.reference, args.batch_size)
    text = json.dumps(record, sort_keys=True, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        print(text)
    print(format_table(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
