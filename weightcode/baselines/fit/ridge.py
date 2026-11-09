"""``ridge_last``, ``ridge_last_free`` and ``ridge_margin``: closed-form refit of the final linear layer toward the base
weights.

    W = argmin_W ||H1 W^T - T||_F^2 + lam ||W - W0||_F^2,   H1 = [H 1],   W = [weight | bias],

with H the penultimate activations of the 64 evidence images (the input of the final linear layer). The solution is
W^T = (H1^T H1 + lam I)^-1 (H1^T T + lam W0^T), solved in float64.

Targets (fixed before any run): T = Z0 + c (Y - softmax(Z0)), where Z0 = H1 W0^T are the base logits on the
evidence, Y the one-hot labels and c the mean over the evidence images of (max logit - mean logit) of the base model,
a shift-invariant logit scale. T is the logit-space target of one Gauss-Newton step of cross-entropy with isotropic
curvature 1/c, so this is the closed-form counterpart of ``grad_last``. One-hot targets scaled to the logit range
would also pull the logits of every correctly classified image toward a one-hot profile and move the model where
nothing is wrong. With this T the residual T - Z0 is near zero on confidently correct images and largest where the
label disagrees with the base model's softmax.

lam = s * mean_i ||[h_i, 1]||^2 for s in 10^-3 .. 10^4 (8 values). The unit is the mean eigenvalue of the evidence
Gram matrix, so the grid spans near interpolation to near no-op on every model.

Two variants share the 8 solutions. ``ridge_last`` is bounded: each candidate's per-tensor delta is scaled into the
contract 5.1 norm limit exactly as ``grad_last`` scales its checkpoints (``scale_to_norm_limit``; tensors below the
executor's rank threshold, such as the bias, are exempt, and so are the tensors the run's limits exempt), and the
executor enforces the limits. ``ridge_last_free`` writes the solutions unscaled and ignores the limits: an
unconstrained reference, like ``grad_full``. Both are closed-form label fits, not training-free.

``ridge_margin`` (the ridge reference of PREREG E2, fixed before any E2 run) is ``ridge_last`` with margin targets:
T = Z0 except that each true-class logit below (best other logit + c) is raised to that value, with the same c. The
Gauss-Newton target fails when a class is new: a student trained without the class holds its logit about 18 to 23
below the best one on the new-class evidence rows, and one step of size c does not close that gap, so even the
interpolating solution would classify only 15 to 21 of the 50 new-class evidence rows correctly (measured on the six
fewshot-v1 instances, evidence rows only). The margin target keeps the property the Gauss-Newton target was chosen
for (no pull where the base is already right with margin c) and classifies every evidence row correctly with margin c
at interpolation.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightcode.baselines.instance_io import Instance
from weightcode.evalkit.fewshot import fewshot_program_limits

from .edits import MAX_REL_FROB, scale_to_norm_limit
from .runner import CandidatePatch, Evidence, Generator, MethodSpec, execute_run, read_model_spec
from .torch_forward import PARITY_ATOL, architecture_from, assert_parity, images_tensor, penultimate, to_tensors

RIDGE_SCALES: tuple[float, ...] = tuple(10.0**exponent for exponent in range(-3, 5))
"""lam / mean squared norm of the augmented evidence features, in query order."""

RIDGE_SPECS: dict[str, MethodSpec] = {
    "ridge_last": MethodSpec(
        method="ridge_last",
        label="closed-form label fit, scaled into the per-tensor norm limit as grad_last; not training-free",
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
    "ridge_last_free": MethodSpec(
        method="ridge_last_free",
        label="closed-form label fit, edit limits ignored (unconstrained reference); not training-free",
        uses_labels_for_fit=True,
        edit_limits_enforced=False,
    ),
    "ridge_margin": MethodSpec(
        method="ridge_margin",
        label=(
            "closed-form label fit with margin targets (true-class logit raised to the best other logit + c), scaled "
            "into the per-tensor norm limit as grad_last; not training-free"
        ),
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
}
RIDGE_METHODS: tuple[str, ...] = tuple(RIDGE_SPECS)
RIDGE_TARGETS: dict[str, str] = {"ridge_last": "gauss_newton", "ridge_last_free": "gauss_newton", "ridge_margin": "margin"}
"""Target definition of each ridge method."""
TARGET_DEFINITIONS: dict[str, str] = {
    "gauss_newton": "T = Z0 + c (onehot(y) - softmax(Z0)), c = mean over evidence of (max - mean) logit",
    "margin": (
        "T = Z0, except T[i, y_i] = max(Z0[i, y_i], max over k != y_i of Z0[i, k] + c), c = mean over evidence of "
        "(max - mean) logit"
    ),
}


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax in float64."""
    shifted = np.asarray(z, dtype=np.float64) - np.max(z, axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=1, keepdims=True)


def _logits_labels_scale(z0: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Base logits in float64, checked labels and the scale c = mean(max logit - mean logit)."""
    z0 = np.asarray(z0, dtype=np.float64)
    n, n_classes = z0.shape
    y = np.asarray(y, dtype=np.int64)
    if y.shape != (n,) or y.min() < 0 or y.max() >= n_classes:
        raise ValueError(f"labels must be {n} integers in [0, {n_classes})")
    scale = float(np.mean(np.max(z0, axis=1) - np.mean(z0, axis=1)))
    if not scale > 0.0:
        raise ValueError(f"base logit scale must be positive, got {scale}")
    return z0, y, scale


def ridge_targets(z0: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """T = Z0 + c (onehot(y) - softmax(Z0)) and the scale c = mean(max logit - mean logit) of the base model."""
    z0, y, scale = _logits_labels_scale(z0, y)
    onehot = np.eye(z0.shape[1])[y]
    return z0 + scale * (onehot - softmax(z0)), scale


def margin_targets(z0: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """T = Z0 with each true-class logit raised to (best other logit + c) where it is lower, and the scale c.

    Every other logit keeps its base value, so T - Z0 is zero on an image the base classifies with margin c and sits on
    the true class alone where it does not.
    """
    z0, y, scale = _logits_labels_scale(z0, y)
    rows = np.arange(len(y))
    others = np.where(np.eye(z0.shape[1], dtype=bool)[y], -np.inf, z0).max(axis=1)
    targets = z0.copy()
    targets[rows, y] = np.maximum(z0[rows, y], others + scale)
    return targets, scale


TARGET_FUNCTIONS = {"gauss_newton": ridge_targets, "margin": margin_targets}


def ridge_solve(h1: np.ndarray, targets: np.ndarray, w0: np.ndarray, lam: float) -> np.ndarray:
    """W (classes x features) minimizing ||h1 W^T - targets||^2 + lam ||W - w0||^2, for lam > 0."""
    if not lam > 0.0:
        raise ValueError(f"lam must be positive, got {lam}")
    n_features = h1.shape[1]
    gram = h1.T @ h1
    solution = np.linalg.solve(gram + lam * np.eye(n_features), h1.T @ targets + lam * w0.T)
    if not np.all(np.isfinite(solution)):
        raise FloatingPointError(f"ridge solution is not finite for lam = {lam}")
    return solution.T


def ridge_generator(
    instance: Instance, method: str = "ridge_last", rel_frob_exempt: Collection[str] = ()
) -> Generator:
    """Candidate generator for ``ridge_last``, ``ridge_margin`` (bounded) or ``ridge_last_free`` on one instance.

    ``rel_frob_exempt``: tensors the run's limits exempt from the norm limit; a bounded variant never scales them.
    """
    if method not in RIDGE_SPECS:
        raise ValueError(f"unknown ridge method {method!r}; expected one of {RIDGE_METHODS}")
    bounded = RIDGE_SPECS[method].edit_limits_enforced
    target_kind = RIDGE_TARGETS[method]
    exempt = tuple(rel_frob_exempt)

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        params = instance.params
        arch = architecture_from(params, read_model_spec(instance))
        parity = assert_parity(arch, params, instance.forward, evidence.x)
        weight_name, bias_name = arch.last_linear_tensors
        with torch.no_grad():
            h = penultimate(arch, to_tensors(params, dtype=torch.float64), images_tensor(evidence.x)).numpy()
        h1 = np.hstack([h, np.ones((len(h), 1))])
        w0 = np.hstack([params[weight_name].astype(np.float64), params[bias_name].astype(np.float64)[:, None]])
        z0 = h1 @ w0.T
        float64_parity = float(np.max(np.abs(z0 - np.asarray(instance.forward(dict(params), evidence.x)))))
        if not float64_parity <= PARITY_ATOL:
            raise RuntimeError(f"float64 features give logits {float64_parity:.3g} away from forward.py")
        targets, scale = TARGET_FUNCTIONS[target_kind](z0, evidence.y)
        unit = float(np.mean(np.sum(h1**2, axis=1)))
        residual_base = float(np.linalg.norm(targets - z0))
        if not residual_base > 0.0:
            raise ValueError(f"{method}: the targets equal the base logits on every evidence image; nothing to fit")
        candidates: list[CandidatePatch] = []
        for slot, relative in enumerate(RIDGE_SCALES):
            lam = relative * unit
            w = ridge_solve(h1, targets, w0, lam)
            child = dict(params)
            child[weight_name] = w[:, :-1].astype(params[weight_name].dtype)
            child[bias_name] = w[:, -1].astype(params[bias_name].dtype)
            notes: dict[str, Any] = {"target_residual_rel": float(np.linalg.norm(h1 @ w.T - targets)) / residual_base}
            if bounded:
                child, scaling = scale_to_norm_limit(params, child, [weight_name, bias_name], exempt=exempt)
                notes["norm_scaling"] = scaling
                notes["norm_scaled_tensors"] = [key for key, entry in scaling.items() if entry["scaled"]]
                scaled_w = np.hstack(
                    [child[weight_name].astype(np.float64), child[bias_name].astype(np.float64)[:, None]]
                )
                notes["target_residual_rel_after_scaling"] = (
                    float(np.linalg.norm(h1 @ scaled_w.T - targets)) / residual_base
                )
            candidates.append(
                CandidatePatch(
                    slot=slot,
                    child=child,
                    config={"lam_relative": relative, "lam": lam},
                    hypothesis=(
                        f"{method}: closed-form refit of {weight_name} and {bias_name} toward the base weights with "
                        f"lam = {relative:g} x {unit:.6g} on the {len(evidence.y)} evidence images "
                        f"({evidence.composition()}), targets {TARGET_DEFINITIONS[target_kind]}"
                        + ("; per-tensor delta scaled into the norm limit" if bounded else "")
                    ),
                    predicted_effect=(
                        "fewer errors on held-out deployment images if the final-layer fit generalizes; smaller lam "
                        "moves further from the base weights"
                    ),
                    notes=notes,
                )
            )
        info = {
            "architecture": arch.describe(),
            "parity_max_abs_logit_diff": parity,
            "float64_feature_parity": float64_parity,
            "target_definition": TARGET_DEFINITIONS[target_kind],
            "target_scale_c": scale,
            "lam_unit_mean_sq_feature_norm": unit,
            "n_features_with_bias": int(h1.shape[1]),
            "n_evidence": int(len(h1)),
            "trainable": [weight_name, bias_name],
        }
        if bounded:
            info["n_candidates_norm_scaled"] = sum(1 for c in candidates if c.notes["norm_scaled_tensors"])
        if exempt:
            info["norm_limit_exempt"] = list(exempt)
        if target_kind == "margin":
            info["n_evidence_raised"] = int(np.count_nonzero(np.any(targets != z0, axis=1)))
        return candidates, info

    return generate


def ridge_config(run_seed: int, method: str = "ridge_last") -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    bounded = RIDGE_SPECS[method].edit_limits_enforced
    if RIDGE_TARGETS[method] == "margin":
        targets = (
            "T = Z0, true-class logit raised to max(Z0 true, best other Z0 + c), c = mean(max logit - mean logit) of "
            "the base on evidence"
        )
    else:
        targets = "T = Z0 + c (onehot(y) - softmax(Z0)), c = mean(max logit - mean logit) of the base on evidence"
    return {
        "variant": "bounded: per-tensor delta scaled into the norm limit as grad_last" if bounded else "unbounded",
        "norm_limit": MAX_REL_FROB if bounded else None,
        "lam_relative_grid": list(RIDGE_SCALES),
        "lam_unit": "mean over evidence images of ||[h, 1]||^2 (h = penultimate activations)",
        "objective": "||[H 1] W^T - T||^2 + lam ||W - W0||^2 over the final layer weight and bias",
        "targets": targets,
        "solver": "numpy.linalg.solve on the normal equations, float64",
        "randomness": "none; the run seed only names the output directory",
        "seed": run_seed,
        "torch": torch.__version__,
    }


def run_ridge(
    instance: Instance,
    run_dir: Path,
    *,
    method: str = "ridge_last",
    run_seed: int = 0,
    allow_main: bool = False,
    joint: bool = False,
    fewshot: bool = False,
    e10: bool = False,
) -> dict[str, Any]:
    """Run a ridge method on one instance and return its run record.

    ``joint``: fit on the evidence plus the replay rows and score with the joint oracle (``execute_run``).
    ``fewshot``: a fewshot-v1 instance, fitted on its new-class and replay rows under ``fewshot_program_limits``.
    """
    if method not in RIDGE_SPECS:
        raise ValueError(f"unknown ridge method {method!r}; expected one of {RIDGE_METHODS}")
    exempt = fewshot_program_limits(instance.task["limits"]).rel_frob_exempt if fewshot else ()
    config = ridge_config(run_seed, method)
    if exempt and RIDGE_SPECS[method].edit_limits_enforced:
        config["norm_limit_exempt"] = list(exempt)
    return execute_run(
        instance=instance,
        spec=RIDGE_SPECS[method],
        run_dir=run_dir,
        run_seed=run_seed,
        config=config,
        generate=ridge_generator(instance, method, exempt),
        allow_main=allow_main,
        joint=joint,
        replay_in_fit=True,
        e10=e10,
        fewshot=fewshot,
    )
