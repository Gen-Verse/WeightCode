"""Fixed-rule arms: one closed-form rule from editor-visible information at a few strengths, no optimization.

* ``prior_shift``: label-shift correction of the final-layer bias, the classic answer to a change of class priors
  between training and deployment (Saerens et al. 2002; Lipton et al. 2018). Only the 48 random evidence examples
  are used, because the 16 failures are a selected sample. With n_c their label counts and C classes,
  p_dep(c) = (n_c + 1) / (48 + C) (add-one smoothing), and p_pred(c) is the base model's mean softmax over the same
  48 images (logits from forward.py). The bias moves by (k / 8) (log p_dep - log p_pred), centred to zero mean (a
  shift common to all logits changes no prediction), for k = 1..8: 8 candidates, one query each.
* ``tie_inputs``: tie the first-layer input weights across input channels with one ``tie`` op,
  W2 <- W2 + s (G - W2) along axis 1 of the 2-D view ``reshape(shape[0], -1)``, G holding each column's group mean.
  MLP: fc1.weight, group of column j = j mod (H W), the pixel of column j under the C, H, W flatten order of
  model_spec.json, so each group holds the C channel entries of one pixel. CNN: conv1.weight viewed as
  [out, C kh kw], group of column j = j mod (kh kw), the kernel position (u, v). The planned strengths are 0.5 and
  1.0. The relative Frobenius norm of the change is linear in s, so the upper strength is
  s_max = min(1, max_rel_frob (1 - margin) / rel_frob(tie at s = 1)), computed from the base weights alone; if
  s_max <= 0.5 the single strength s_max is used (1 query). With one input channel every group is a single
  column and there is nothing to tie; the arm then records that and checks nothing.

* ``input_stats`` (image settings, label-free): per input channel c, the mean m_c and std s_c of the preprocessed
  input x' (after forward.py's normalization) over the 48 random evidence images only (never x_fail, a selected
  sample). For a strength t in {0.5, 1.0} the target transform is x'' = k_c (x' - t m_c), k_c = 1 + t (1/s_c - 1),
  standardizing the deployment inputs toward the statistics the first layer was trained on. It is applied in an exact
  admissible form: with g the geometric mean of the k_c and r_c = k_c / g, the first layer's input columns (MLP) or
  input channels (CNN) are scaled by r_c (``scale_cols``), its bias is set to b1 / g - W1 diag(r) (t m) (the shift
  summed over each channel's columns or kernel positions), and every later layer's bias is divided by g. ReLU, max
  pooling, flattening and linear maps are positively homogeneous, so the logits are the k-form logits divided by g and
  the prediction is unchanged; the common factor g lives in the 1-D biases, outside the matrix norm limit. For a conv
  first layer the folded mean shift also reaches zero-padded border positions (recorded in run.json "deviations").
  A ``resnet20_a`` student (PREREG E10) has no first-layer bias; conv1 is followed by bn1 in eval form, so the fold
  goes into bn1's running statistics instead: conv1's input channels are scaled by r_c, and with b the folded mean
  shift (b_o = g sum_c r_c t m_c sum_uv W1[o, c, u, v]), mu' = (mu + b) / g and var' = (var + eps) / g^2 - eps.
  Then bn1 sees exactly what the k-form transform gives (up to the same border deviation), and no later tensor
  changes. The running statistics are 1-D, outside the matrix norm limit, and counted in the changed entries.
  A ``resnet18_up`` student (PREREG E12) gets a bn1 fold too. Its normalised 32 x 32 input is upsampled
  bilinearly before conv1, and bilinear weights sum to 1 along each axis, so a per-channel affine map of the input
  commutes with the upsampling exactly. m_c and s_c are therefore taken on the 32 x 32 normalised image, where the
  transform is stated, and conv1's zero padding gives the same border deviation. Under the PREREG E12 amendment
  (WEIGHTCODE-main, 2026-11-05) the factor g goes into bn1's affine weight instead of its running variance: mu' = (mu + b)
  / g as above, gamma' = g gamma and var unchanged. Both forms compute the same function, but E10's var' is negative
  wherever var < eps (g^2 - 1), which the near-dead stem channels of ImageNet ResNet-18 (running variance about
  1e-13 to 1e-8) reach at any g > 1. The weight form has no infeasible case. resnet20_a keeps E10's form.
  A ``vit_lite_7_4`` student (PREREG E11) cannot move a common factor g into later biases (the positional embedding
  and LayerNorm are not positively homogeneous), so the k-form itself is folded into the patch projection:
  W' = W diag(k) (``scale_cols`` per input channel) and b' = b - W diag(k) (t m) summed over each channel's patch
  positions. The patches tile the image without padding, so the fold is exact. A strength whose W' exceeds the
  rel-Frobenius cap is still proposed and recorded ("admissible": false); the executor rejects it.

No arm reads V_check. All write patches that the shared executor checks under the contract limits; the final patch
follows contract 5.2 as for every arm.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance

from .edits import MAX_REL_FROB, NORM_SCALE_MARGIN, REL_FROB_MIN_NDIM, rel_frob
from .ridge import softmax
from .runner import CandidatePatch, Evidence, Generator, MethodSpec, execute_run, read_model_spec
from .torch_forward import (
    BN_FAMILIES,
    RESNET18NAT,
    RESNET18UP,
    RESNET20A,
    VITLITE,
    Architecture,
    architecture_from,
    assert_parity,
)

PRIOR_SHIFT_FRACTIONS: tuple[float, ...] = tuple(k / 8 for k in range(1, 9))
"""Fractions k / 8 of the label-shift correction, in query order."""

TIE_STRENGTHS_PLANNED: tuple[float, ...] = (0.5, 1.0)
"""Planned tie strengths; the upper one is lowered to s_max when a full tie exceeds the norm limit."""

CHW_FLATTEN_ORDERS = frozenset({"NCHW, flatten row-major", "C, H, W (row-major)"})
"""model_spec.json statements of the C, H, W flatten order that the MLP group map assumes."""

PRIOR_SHIFT_SPEC = MethodSpec(
    method="prior_shift",
    label=(
        "fixed rule, label-shift correction (Saerens et al. 2002; Lipton et al. 2018): final-layer bias moved by k/8 "
        "of log(p_dep / p_pred) from the 48 random evidence examples, k = 1..8; label counts only, no optimization"
    ),
    uses_labels_for_fit=True,
    edit_limits_enforced=True,
)

TIE_SPEC = MethodSpec(
    method="tie_inputs",
    label=(
        "fixed rule: tie the first-layer input weights across input channels (one tie op at s = 0.5 and at the "
        "largest admissible s <= 1); base weights only, no labels, no optimization"
    ),
    uses_labels_for_fit=False,
    edit_limits_enforced=True,
)


# ----------------------------------------------------------------------------------------------- prior_shift


def label_shift_direction(logits_rand: np.ndarray, y_rand: np.ndarray) -> dict[str, np.ndarray]:
    """Class counts, p_dep, p_pred and the centred bias direction log p_dep - log p_pred on the random evidence."""
    z = np.asarray(logits_rand, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError(f"logits must be [n > 0, n_classes], got {z.shape}")
    n, n_classes = z.shape
    y = np.asarray(y_rand)
    if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= n_classes:
        raise ValueError(f"labels must be {n} integers in [0, {n_classes})")
    counts = np.bincount(y.astype(np.int64), minlength=n_classes)
    p_dep = (counts + 1.0) / (n + n_classes)
    p_pred = softmax(z).mean(axis=0)
    ratio = np.log(p_dep) - np.log(p_pred)
    if not np.all(np.isfinite(ratio)):
        raise FloatingPointError(f"log(p_dep / p_pred) is not finite: p_pred = {p_pred.tolist()}")
    return {"counts": counts, "p_dep": p_dep, "p_pred": p_pred, "direction": ratio - ratio.mean()}


def prior_shift_generator(instance: Instance) -> Generator:
    """Candidate generator for ``prior_shift`` on one instance."""

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        params = instance.params
        arch = architecture_from(params, read_model_spec(instance))
        bias_name = arch.last_linear_tensors[1]
        x_rand = evidence.x[evidence.n_fail :]
        y_rand = evidence.y[evidence.n_fail :]
        logits = np.asarray(instance.forward(dict(params), x_rand), dtype=np.float64)
        if logits.shape != (len(x_rand), params[bias_name].shape[0]):
            raise ValueError(f"forward.py gives logits {logits.shape}; {bias_name} has {params[bias_name].shape[0]}")
        stats = label_shift_direction(logits, y_rand)
        direction = stats["direction"]
        base_bias = params[bias_name].astype(np.float64)
        candidates: list[CandidatePatch] = []
        for slot, fraction in enumerate(PRIOR_SHIFT_FRACTIONS):
            child = dict(params)
            child[bias_name] = (base_bias + fraction * direction).astype(params[bias_name].dtype)
            marginal = softmax(logits + fraction * direction).mean(axis=0)
            candidates.append(
                CandidatePatch(
                    slot=slot,
                    child=child,
                    config={"k": slot + 1, "fraction": fraction},
                    hypothesis=(
                        f"prior_shift: the class priors of the deployment data differ from what the model predicts; "
                        f"move {bias_name} by {fraction:g} x (log p_dep - log p_pred), p_dep from the label counts of "
                        f"the {len(y_rand)} random evidence examples (add-one), p_pred the mean softmax on them"
                    ),
                    predicted_effect=(
                        "fewer errors on held-out deployment images if the errors come from a class-prior mismatch; "
                        "no effect expected on errors with another cause"
                    ),
                    notes={
                        "marginal_after": marginal,
                        "marginal_gap_l1_after": float(np.sum(np.abs(stats["p_dep"] - marginal))),
                    },
                )
            )
        info = {
            "architecture": arch.describe(),
            "bias_tensor": bias_name,
            "n_rand": int(len(y_rand)),
            "n_classes": int(logits.shape[1]),
            "class_counts": stats["counts"],
            "p_dep": stats["p_dep"],
            "p_pred": stats["p_pred"],
            "direction": direction,
            "marginal_gap_l1_base": float(np.sum(np.abs(stats["p_dep"] - stats["p_pred"]))),
            "fractions": list(PRIOR_SHIFT_FRACTIONS),
            "uses": "x_rand, y_rand and forward.py logits on x_rand; not x_fail, not V_check",
        }
        return candidates, info

    return generate


def prior_shift_config(run_seed: int) -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    return {
        "rule": "bias += (k/8) (log p_dep - log p_pred), centred; p_dep = (n_c + 1) / (n_rand + C); "
        "p_pred = mean softmax of the base model on x_rand",
        "fractions": list(PRIOR_SHIFT_FRACTIONS),
        "randomness": "none; the run seed only names the output directory",
        "seed": run_seed,
    }


def run_prior_shift(
    instance: Instance, run_dir: Path, *, run_seed: int = 0, allow_main: bool = False, joint: bool = False,
    e10: bool = False,
) -> dict[str, Any]:
    """Run ``prior_shift`` on one instance and return its run record (also written to run.json).

    ``joint``: score with the joint oracle; the rule still reads the deployment evidence only (``execute_run``).
    """
    return execute_run(
        instance=instance,
        spec=PRIOR_SHIFT_SPEC,
        run_dir=run_dir,
        run_seed=run_seed,
        config=prior_shift_config(run_seed),
        generate=prior_shift_generator(instance),
        allow_main=allow_main,
        joint=joint,
        replay_in_fit=False,
        e10=e10,
    )


# ----------------------------------------------------------------------------------------------- tie_inputs


@dataclass(frozen=True)
class TiePlan:
    """Which tensor the tie acts on and the group of every column of its 2-D view (axis 1)."""

    tensor: str
    groups: np.ndarray
    n_channels: int
    group_rule: str

    def describe(self) -> dict[str, Any]:
        """JSON-ready summary for run records (the group map itself is in each patch)."""
        return {
            "tensor": self.tensor,
            "axis": 1,
            "n_positions": int(self.groups.size),
            "n_groups": int(np.unique(self.groups).size),
            "n_channels": self.n_channels,
            "group_rule": self.group_rule,
        }


def _check_chw_flatten(spec: Mapping[str, Any]) -> None:
    stated = [spec.get("input", {}).get("layout")]
    stated += [layer.get("order") for layer in spec.get("layers", []) if layer.get("op") == "flatten"]
    stated = [value for value in stated if value is not None]
    if not stated or any(value not in CHW_FLATTEN_ORDERS for value in stated):
        raise ValueError(
            f"model_spec.json must state the C, H, W flatten order {sorted(CHW_FLATTEN_ORDERS)}, got {stated}"
        )


def tie_plan(arch: Architecture, params: Mapping[str, np.ndarray], spec: Mapping[str, Any]) -> TiePlan:
    """The first layer's input-weight tensor and the channel-tying group map of its columns."""
    channels, height, width = arch.input_shape
    if arch.family == "mlp":
        _check_chw_flatten(spec)
        tensor = f"{arch.linear_layers[0]}.weight"
        n_positions = int(params[tensor].shape[1])
        if n_positions != channels * height * width:
            raise ValueError(f"{tensor} has {n_positions} columns, the input has {channels * height * width} entries")
        groups = np.arange(n_positions, dtype=np.int64) % (height * width)
        rule = "column j -> pixel j mod (H*W) under the C, H, W flatten order"
    else:
        tensor = f"{arch.conv_layers[0]}.weight"
        _, in_channels, kernel_h, kernel_w = params[tensor].shape
        if in_channels != channels:
            raise ValueError(f"{tensor} has {in_channels} input channels, the input has {channels}")
        groups = np.arange(in_channels * kernel_h * kernel_w, dtype=np.int64) % (kernel_h * kernel_w)
        rule = "column j of the [out, C*kh*kw] view -> kernel position j mod (kh*kw)"
    return TiePlan(tensor=tensor, groups=groups, n_channels=channels, group_rule=rule)


def tie_columns(weight: np.ndarray, groups: np.ndarray, strength: float) -> np.ndarray:
    """``W2 + s (G - W2)`` on the 2-D view along axis 1, in the executor's float64 arithmetic and order, cast once.

    Group sums add the columns in ascending position order, as ``weightcode.patch`` does, so the result is expected to
    match the executor bit for bit; entries equal in value to the parent keep the parent's bits.
    """
    parent = np.asarray(weight)
    matrix = np.array(parent, dtype=np.float64, order="C").reshape(parent.shape[0], -1)
    if groups.shape != (matrix.shape[1],):
        raise ValueError(f"groups need one id per column ({matrix.shape[1]}), got shape {groups.shape}")
    _, codes = np.unique(groups.astype(np.int64), return_inverse=True)
    codes = codes.reshape(-1)
    n_groups = int(codes.max()) + 1
    sums = np.zeros((matrix.shape[0], n_groups), dtype=np.float64)
    for position in range(matrix.shape[1]):
        sums[:, codes[position]] += matrix[:, position]
    means = sums / np.bincount(codes, minlength=n_groups).astype(np.float64)
    matrix += strength * (means[:, codes] - matrix)
    child = matrix.reshape(parent.shape).astype(parent.dtype)
    np.copyto(child, parent, where=child == parent)
    return child


def tie_strengths(full_tie_rel_frob: float, ndim: int) -> tuple[list[float], float]:
    """Strengths to check and s_max = min(1, limit (1 - margin) / rel_frob(tie at s = 1)).

    Returns [0.5, s_max] when s_max > 0.5, else [s_max]. Without a norm limit on this tensor s_max is 1.
    """
    if not full_tie_rel_frob > 0.0:
        raise ValueError("a tie that changes nothing has no strength to choose")
    if MAX_REL_FROB is None or ndim < REL_FROB_MIN_NDIM:
        upper = 1.0
    else:
        upper = min(1.0, MAX_REL_FROB * (1.0 - NORM_SCALE_MARGIN) / full_tie_rel_frob)
    lower = TIE_STRENGTHS_PLANNED[0]
    return ([lower, upper] if upper > lower else [upper]), upper


def tie_generator(instance: Instance) -> Generator:
    """Candidate generator for ``tie_inputs`` on one instance (no evidence is used)."""

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        params = instance.params
        arch = architecture_from(params, read_model_spec(instance))
        plan = tie_plan(arch, params, read_model_spec(instance))
        info: dict[str, Any] = {
            "architecture": arch.describe(),
            "plan": plan.describe(),
            "planned_strengths": list(TIE_STRENGTHS_PLANNED),
            "uses": "base weights and model_spec.json only; no evidence labels, no V_check",
        }
        weight = params[plan.tensor]
        if plan.n_channels == 1:
            info["not_applicable"] = "the input has one channel, so every group is a single column: nothing to tie"
            return [], info
        full = tie_columns(weight, plan.groups, 1.0)
        full_rel = rel_frob(weight, full)
        info["rel_frob_full_tie"] = full_rel
        if not full_rel > 0.0:
            info["not_applicable"] = "the channel weights are already tied: a tie changes nothing"
            return [], info
        strengths, s_max = tie_strengths(full_rel, weight.ndim)
        info.update(
            s_max=s_max,
            s_max_rule="min(1, max_rel_frob * (1 - margin) / rel_frob(tie at s = 1)) from the base weights",
            norm_limit=MAX_REL_FROB if weight.ndim >= REL_FROB_MIN_NDIM else None,
            margin=NORM_SCALE_MARGIN,
            strengths=strengths,
            single_strength=len(strengths) == 1,
            rel_frob_at_planned={f"{s:g}": s * full_rel for s in TIE_STRENGTHS_PLANNED},
        )
        candidates: list[CandidatePatch] = []
        for slot, strength in enumerate(strengths):
            child = dict(params)
            child[plan.tensor] = tie_columns(weight, plan.groups, strength)
            planned = [TIE_STRENGTHS_PLANNED[slot]] if len(strengths) == 2 else list(TIE_STRENGTHS_PLANNED)
            op = {"op": "tie", "tensor": plan.tensor, "axis": 1, "groups": plan.groups.tolist(), "strength": strength}
            candidates.append(
                CandidatePatch(
                    slot=slot,
                    child=child,
                    config={"strength": strength, "planned": planned},
                    hypothesis=(
                        f"tie_inputs: the model reads color through channel-specific first-layer weights; tie "
                        f"{plan.tensor} across the {plan.n_channels} input channels of each "
                        f"{'pixel' if arch.family == 'mlp' else 'kernel position'} with strength {strength:.6g}"
                    ),
                    predicted_effect=(
                        "fewer errors on held-out deployment images if the errors come from channel (color) cues; "
                        "images classified by shape should stay correct"
                    ),
                    notes={
                        "rel_frob_expected": strength * full_rel,
                        "rel_frob_realized": rel_frob(weight, child[plan.tensor]),
                    },
                    ops=[op],
                )
            )
        return candidates, info

    return generate


def tie_config(run_seed: int) -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    return {
        "rule": "one tie op on the first layer's input weights, axis 1, groups = channel-tying map",
        "planned_strengths": list(TIE_STRENGTHS_PLANNED),
        "upper_strength": (
            "s_max = min(1, max_rel_frob (1 - margin) / rel_frob(tie at s = 1)); single strength if <= 0.5"
        ),
        "norm_limit": MAX_REL_FROB,
        "margin": NORM_SCALE_MARGIN,
        "randomness": "none; the run seed only names the output directory",
        "seed": run_seed,
    }


def run_tie(
    instance: Instance, run_dir: Path, *, run_seed: int = 0, allow_main: bool = False, joint: bool = False,
    e10: bool = False,
) -> dict[str, Any]:
    """Run ``tie_inputs`` on one instance and return its run record (also written to run.json).

    ``joint``: score with the joint oracle; the rule still reads the deployment evidence only (``execute_run``).
    """
    return execute_run(
        instance=instance,
        spec=TIE_SPEC,
        run_dir=run_dir,
        run_seed=run_seed,
        config=tie_config(run_seed),
        generate=tie_generator(instance),
        allow_main=allow_main,
        joint=joint,
        replay_in_fit=False,
        e10=e10,
    )


# ----------------------------------------------------------------------------------------------- input_stats

INPUT_STATS_STRENGTHS: tuple[float, ...] = (0.5, 1.0)
"""Strengths t of the input-statistics transform, in query order."""

INPUT_STATS_SPEC = MethodSpec(
    method="input_stats",
    label=(
        "fixed rule, label-free input statistics: move the first layer's per-channel input mean and scale toward "
        "those of the 48 random evidence images (t in {0.5, 1}), folded exactly into the weights; no labels, no "
        "optimization"
    ),
    uses_labels_for_fit=False,
    edit_limits_enforced=True,
)

CONV_PADDING_DEVIATION = (
    "conv first layer: folding the mean shift t m_c into the first bias also applies it at zero-padded border "
    "positions, so the patched model equals the k-form transform with the padding value k_c (0 - t m_c) instead of 0"
)


def input_channel_stats(arch: Architecture, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and population std, over images and pixels, of the preprocessed input (x / 255 - mean) / std."""
    channels, height, width = arch.input_shape
    images = np.asarray(x)
    if images.dtype != np.uint8 or images.size != len(images) * channels * height * width or len(images) == 0:
        raise ValueError(
            f"need uint8 images of shape [n, {channels}, {height}, {width}], got {images.dtype}{images.shape}"
        )
    pre = images.reshape(len(images), channels, height, width).astype(np.float64) / 255.0
    pre = (pre - np.asarray(arch.mean)[None, :, None, None]) / np.asarray(arch.std)[None, :, None, None]
    mean = pre.mean(axis=(0, 2, 3))
    std = pre.std(axis=(0, 2, 3))
    if not (np.all(np.isfinite(std)) and np.all(std > 0.0)):
        raise ValueError(f"every input channel needs a positive finite std, got {std.tolist()}")
    return mean, std


@dataclass(frozen=True)
class InputStatsFold:
    """One strength of ``input_stats``: the patch ops, the child they give and the quantities to log."""

    strength: float
    k: np.ndarray
    g: float
    r: np.ndarray
    ops: list[dict[str, Any]]
    child: dict[str, np.ndarray]
    first_weight: str
    first_bias: str
    later_biases: list[str]
    rel_frob_k_form: float
    rel_frob_admissible: float
    bn_fold: dict[str, Any] | None = None
    patch_fold: dict[str, Any] | None = None

    def notes(self) -> dict[str, Any]:
        """JSON-ready per-candidate log."""
        if self.patch_fold is not None:
            return {"t": self.strength, "k": self.k, "first_weight": self.first_weight,
                    "first_bias": self.first_bias, "rel_frob_first_weight_k_form": self.rel_frob_k_form,
                    "patch_fold": self.patch_fold}
        if self.bn_fold is not None:
            return {"t": self.strength, "k": self.k, "g": self.g, "r": self.r, "first_weight": self.first_weight,
                    "rel_frob_first_weight_k_form": self.rel_frob_k_form,
                    "rel_frob_first_weight_admissible": self.rel_frob_admissible, "bn_fold": self.bn_fold}
        return {
            "t": self.strength,
            "k": self.k,
            "g": self.g,
            "r": self.r,
            "first_weight": self.first_weight,
            "rel_frob_first_weight_k_form": self.rel_frob_k_form,
            "rel_frob_first_weight_admissible": self.rel_frob_admissible,
            "later_biases_divided_by_g": self.later_biases,
        }


def _first_and_later(arch: Architecture) -> tuple[str, list[str]]:
    if arch.family == "mlp":
        return arch.linear_layers[0], list(arch.linear_layers[1:])
    return arch.conv_layers[0], [*arch.conv_layers[1:], *arch.linear_layers]


def _scaled_copy(parent: np.ndarray, value: np.ndarray) -> np.ndarray:
    child = value.astype(parent.dtype)
    np.copyto(child, parent, where=child == parent)
    return child


def input_stats_fold(
    arch: Architecture, params: Mapping[str, np.ndarray], mean: np.ndarray, std: np.ndarray, strength: float
) -> InputStatsFold:
    """The exact admissible form of x'' = k_c (x' - t m_c) for one strength t, as ops and as the expected child.

    The child is computed with the executor's float64 arithmetic (one multiplication per scaled entry, the set values
    cast once), so the executor is expected to reproduce it bit for bit.
    """
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    channels, height, width = arch.input_shape
    k = 1.0 + strength * (1.0 / np.asarray(std, dtype=np.float64) - 1.0)
    if k.shape != (channels,) or not np.all(k > 0.0):
        raise ValueError(f"k must hold one positive factor per channel, got {k.tolist()}")
    g = float(k[0]) if channels == 1 else float(np.exp(np.mean(np.log(k))))
    r = k / g
    shift = strength * np.asarray(mean, dtype=np.float64)
    if arch.family in BN_FAMILIES:  # resnet20_a and resnet18_up: conv1 without bias, then bn1 in eval form
        return _input_stats_fold_bn(params, strength, k, g, r, shift, BN_FOLD_G_INTO[arch.family])
    if arch.family == VITLITE:
        return _input_stats_fold_patch(arch, params, strength, k, shift)
    first, later = _first_and_later(arch)
    w_name, b_name = f"{first}.weight", f"{first}.bias"
    weight = params[w_name].astype(np.float64)
    if arch.family == "mlp":
        if weight.shape[1] != channels * height * width:
            raise ValueError(f"{w_name} has {weight.shape[1]} columns, the input has {channels * height * width}")
        column_channel = np.arange(weight.shape[1]) // (height * width)
        columns = [list(range(c * height * width, (c + 1) * height * width)) for c in range(channels)]
        scaled = weight * r[column_channel][None, :]
        k_form = weight * k[column_channel][None, :]
        correction = weight @ (r[column_channel] * shift[column_channel])
    else:
        columns = [[c] for c in range(channels)]
        scaled = weight * r[None, :, None, None]
        k_form = weight * k[None, :, None, None]
        correction = np.einsum("ocuv,c->o", weight, r * shift)
    new_bias = params[b_name].astype(np.float64) / g - correction
    inverse_g = 1.0 / g
    ops: list[dict[str, Any]] = []
    child = dict(params)
    for channel, cols in enumerate(columns):
        factor = float(r[channel])
        if factor != 1.0:
            ops.append({"op": "scale_cols", "tensor": w_name, "cols": cols, "factor": factor})
    child[w_name] = _scaled_copy(params[w_name], scaled)
    index = [[i] for i in range(new_bias.size)]
    ops.append({"op": "set", "tensor": b_name, "index": index, "values": new_bias.tolist()})
    child[b_name] = _scaled_copy(params[b_name], new_bias)
    later_biases = [f"{layer}.bias" for layer in later]
    if inverse_g != 1.0:
        for name in later_biases:
            rows = list(range(params[name].shape[0]))
            ops.append({"op": "scale_rows", "tensor": name, "rows": rows, "factor": inverse_g})
            child[name] = _scaled_copy(params[name], params[name].astype(np.float64) * inverse_g)
    return InputStatsFold(
        strength=strength,
        k=k,
        g=g,
        r=r,
        ops=ops,
        child=child,
        first_weight=w_name,
        first_bias=b_name,
        later_biases=later_biases if inverse_g != 1.0 else [],
        rel_frob_k_form=rel_frob(params[w_name], k_form),
        rel_frob_admissible=rel_frob(params[w_name], scaled),
    )


BN_FOLD_G_INTO = {RESNET20A: "running_var", RESNET18UP: "weight", RESNET18NAT: "weight"}
"""Where input_stats' bn1 fold puts the common factor g: E10's running variance, or (PREREG E12 amendment) the weight;
PREREG T04's native network takes E12's form (the same stem, so the same near-dead stem channels)."""


def _input_stats_fold_bn(params: Mapping[str, np.ndarray], strength: float, k: np.ndarray, g: float, r: np.ndarray,
                         shift: np.ndarray, g_into: str) -> InputStatsFold:
    """conv1's input channels scaled by r_c, the mean shift folded into bn1's running mean, and g into bn1.

    With z the folded conv1 output and c the folded shift, the k-form conv1 output is g (z - c), and
    bn1(g (z - c)) = gamma (z - c - mu / g) / sqrt((var + eps) / g^2) = g gamma (z - c - mu / g) / sqrt(var + eps).
    ``g_into`` "running_var" (resnet20_a, PREREG E10) sets var' = (var + eps) / g^2 - eps, which is admissible only if
    every var' > 0; "weight" (resnet18_up, PREREG E12 amendment) sets gamma' = g gamma with var unchanged, always
    admissible. Both set mu' = mu / g + c.
    """
    from weightcode.models.resnet20a import BN_EPS

    if g_into not in ("running_var", "weight"):
        raise ValueError(f"g_into must be 'running_var' or 'weight', got {g_into!r}")
    w_name, gamma_name, mean_name, var_name = "conv1.weight", "bn1.weight", "bn1.running_mean", "bn1.running_var"
    weight = params[w_name].astype(np.float64)
    if weight.ndim != 4 or weight.shape[1] != len(k):
        raise ValueError(f"{w_name} has shape {weight.shape}; expected [out, {len(k)}, kh, kw]")
    scaled = weight * r[None, :, None, None]
    correction = np.einsum("ocuv,c->o", weight, r * shift)  # = b / g
    mu = params[mean_name].astype(np.float64)
    new_mean = mu / g + correction
    if g_into == "running_var":
        new_var = (params[var_name].astype(np.float64) + BN_EPS) / g**2 - BN_EPS
        if not np.all(new_var > 0.0):
            raise ValueError(f"g = {g:.6g} leaves a non-positive running variance in bn1; the fold is not admissible")
    ops: list[dict[str, Any]] = []
    child = dict(params)
    for channel in range(len(k)):
        factor = float(r[channel])
        if factor != 1.0:
            ops.append({"op": "scale_cols", "tensor": w_name, "cols": [channel], "factor": factor})
    child[w_name] = _scaled_copy(params[w_name], scaled)
    ops.append({"op": "set", "tensor": mean_name, "index": [[i] for i in range(new_mean.size)],
                "values": new_mean.tolist()})
    child[mean_name] = _scaled_copy(params[mean_name], new_mean)
    if g_into == "running_var":
        ops.append({"op": "set", "tensor": var_name, "index": [[i] for i in range(new_var.size)],
                    "values": new_var.tolist()})
        child[var_name] = _scaled_copy(params[var_name], new_var)
        bn_fold = {"tensors": [mean_name, var_name], "eps": BN_EPS,
                   "rule": "mu' = mu / g + sum_c r_c t m_c sum_uv W1[o, c, u, v]; var' = (var + eps) / g^2 - eps"}
    else:
        if g != 1.0:
            ops.append({"op": "scale_rows", "tensor": gamma_name, "rows": list(range(params[gamma_name].shape[0])),
                        "factor": g})
            child[gamma_name] = _scaled_copy(params[gamma_name], params[gamma_name].astype(np.float64) * g)
        bn_fold = {"tensors": [mean_name, gamma_name], "eps": BN_EPS,
                   "rule": "mu' = mu / g + sum_c r_c t m_c sum_uv W1[o, c, u, v]; gamma' = g gamma; var unchanged "
                   "(PREREG E12 amendment)"}
    return InputStatsFold(
        strength=strength, k=k, g=g, r=r, ops=ops, child=child, first_weight=w_name, first_bias=mean_name,
        later_biases=[], rel_frob_k_form=rel_frob(params[w_name], weight * k[None, :, None, None]),
        rel_frob_admissible=rel_frob(params[w_name], scaled), bn_fold=bn_fold,
    )


def _input_stats_fold_patch(arch: Architecture, params: Mapping[str, np.ndarray], strength: float, k: np.ndarray,
                            shift: np.ndarray) -> InputStatsFold:
    """vit_lite_7_4: the k-form folded exactly into the patch projection, W' = W diag(k), b' = b - W diag(k) (t m)."""
    first = arch.conv_layers[0]
    w_name, b_name = f"{first}.weight", f"{first}.bias"
    weight = params[w_name].astype(np.float64)
    if weight.ndim != 4 or weight.shape[1] != len(k):
        raise ValueError(f"{w_name} has shape {weight.shape}; expected [dim, {len(k)}, patch, patch]")
    scaled = weight * k[None, :, None, None]
    new_bias = params[b_name].astype(np.float64) - np.einsum("ocuv,c->o", weight, k * shift)
    ops: list[dict[str, Any]] = []
    child = dict(params)
    for channel in range(len(k)):
        factor = float(k[channel])
        if factor != 1.0:
            ops.append({"op": "scale_cols", "tensor": w_name, "cols": [channel], "factor": factor})
    child[w_name] = _scaled_copy(params[w_name], scaled)
    ops.append({"op": "set", "tensor": b_name, "index": [[i] for i in range(new_bias.size)],
                "values": new_bias.tolist()})
    child[b_name] = _scaled_copy(params[b_name], new_bias)
    relative = rel_frob(params[w_name], scaled)
    admissible = MAX_REL_FROB is None or relative <= MAX_REL_FROB
    return InputStatsFold(
        strength=strength, k=k, g=1.0, r=k, ops=ops, child=child, first_weight=w_name, first_bias=b_name,
        later_biases=[], rel_frob_k_form=relative, rel_frob_admissible=relative,
        patch_fold={"rule": "W' = W diag(k); b'_o = b_o - sum_c k_c t m_c sum_uv W[o, c, u, v]; exact, no padding",
                    "rel_frob_cap": MAX_REL_FROB, "admissible": bool(admissible)},
    )


def input_stats_generator(instance: Instance) -> Generator:
    """Candidate generator for ``input_stats`` on one instance (x_rand only; no labels)."""

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        params = instance.params
        arch = architecture_from(params, read_model_spec(instance))
        parity = assert_parity(arch, params, instance.forward, evidence.x)
        x_rand = evidence.x[evidence.n_fail :]
        mean, std = input_channel_stats(arch, x_rand)
        candidates: list[CandidatePatch] = []
        for slot, strength in enumerate(INPUT_STATS_STRENGTHS):
            fold = input_stats_fold(arch, params, mean, std, strength)
            candidates.append(
                CandidatePatch(
                    slot=slot,
                    child=fold.child,
                    config={"t": strength},
                    hypothesis=(
                        f"input_stats: the deployment inputs' per-channel mean and scale differ from the "
                        f"normalization the first layer was trained with; transform x'' = k_c (x' - {strength:g} m_c), "
                        f"k_c = 1 + {strength:g} (1/s_c - 1), with m_c and s_c from the {len(x_rand)} random evidence "
                        + ("images, folded exactly into the patch projection: input channels scaled by k_c, its bias "
                           "reset" if arch.family == VITLITE else
                           "images, folded exactly: columns scaled by k_c / g, first bias reset, later biases divided "
                           "by g")
                    ),
                    predicted_effect=(
                        "fewer errors on held-out deployment images if the shift is a per-channel offset or scale of "
                        "the inputs; near no-op when the statistics already match"
                    ),
                    notes=fold.notes(),
                    ops=fold.ops,
                )
            )
        info: dict[str, Any] = {
            "architecture": arch.describe(),
            "parity_max_abs_logit_diff": parity,
            "n_images": int(len(x_rand)),
            "channel_mean": mean,
            "channel_std": std,
            "std": "population std (ddof 0) over images and pixels of each channel, float64",
            "strengths": list(INPUT_STATS_STRENGTHS),
            "uses": "x_rand only (no labels, not x_fail, not V_check)",
            "deviations": ([CONV_PADDING_DEVIATION] if arch.family in ("cnn", RESNET20A, RESNET18UP, RESNET18NAT)
                           else []),
        }
        return candidates, info

    return generate


def input_stats_config(run_seed: int) -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    return {
        "rule": "x'' = k_c (x' - t m_c), k_c = 1 + t (1/s_c - 1); admissible form: scale_cols by k_c / g, first bias "
        "b1 / g - W1 diag(k/g) (t m), later biases / g, g = geometric mean of k",
        "strengths": list(INPUT_STATS_STRENGTHS),
        "statistics": "per-channel mean and std of the preprocessed x_rand",
        "randomness": "none; the run seed only names the output directory",
        "seed": run_seed,
    }


def run_input_stats(
    instance: Instance, run_dir: Path, *, run_seed: int = 0, allow_main: bool = False, joint: bool = False,
    e10: bool = False,
) -> dict[str, Any]:
    """Run ``input_stats`` on one instance and return its run record (also written to run.json).

    ``joint``: score with the joint oracle; the rule still reads the deployment evidence only (``execute_run``).
    """
    return execute_run(
        instance=instance,
        spec=INPUT_STATS_SPEC,
        run_dir=run_dir,
        run_seed=run_seed,
        config=input_stats_config(run_seed),
        generate=input_stats_generator(instance),
        allow_main=allow_main,
        joint=joint,
        replay_in_fit=False,
        e10=e10,
    )
