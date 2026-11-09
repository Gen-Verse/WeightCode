"""``imprint``: the head rows of new classes imprinted from class-mean penultimate features (PREREG E2 reference).

Weight imprinting (Qi, Brown and Lowe, "Low-Shot Learning with Imprinted Weights", CVPR 2018) sets the classifier row
of a new class to the normalised mean of the normalised embeddings of its examples: the nearest-class-mean rule written
as classifier weights. Here the head is a plain linear layer on unnormalised ReLU features h (the input of the final
linear layer) and its train-class rows are kept, so each imprinted row gets a scale relative to them and the new
classes share one bias offset fitted on the evidence. The train-class rows and biases do not change.

Candidates (8, fixed before any E2 run), direction outer and scale inner:

- direction "mean": u_c = normalize(mean over the evidence rows of class c of normalize(h)), as in Qi et al.;
- direction "centered": the same with every h replaced by h - m before normalising, where
  m = (mean h over the new-class rows + mean h over the replay rows) / 2, and the centring folded into the bias
  (b0_c = -w_c . m). Centring before normalising is the CL2N transform of SimpleShot (Wang et al. 2019). ReLU
  features are non-negative, so uncentred class means share one orthant and their directions differ less than the
  classes do;
- scale s in IMPRINT_SCALES: w_c = s * rho * u_c, with rho the mean L2 norm of the train-class rows.

Bias: b_c = b0_c + beta for every new class c (b0_c = 0 for "mean"). beta maximises the balanced evidence accuracy,
(accuracy on the new-class rows + accuracy on the replay rows) / 2. The check scores S_new + S_old on parts of equal
size and V_select holds as many new-class as old-class rows, so the two parts weigh the same. Only beta moves, so the
objective is a step function of beta with one threshold per row:

- a new-class row whose imprinted logit is the largest among the new classes is right exactly when beta exceeds
  (best train-class logit - its imprinted logit);
- a replay row that the base ranks first among the train classes is right exactly when beta stays below (its
  true-class logit - the best imprinted logit).

The maximum is found exactly over the midpoints between consecutive thresholds and one point beyond each end (at
distance c, the base logit scale of ``ridge``). Ties go to the widest interval, then to the smaller beta.

``imprint_retain`` is supplementary to PREREG E2: it was recorded before any E2 LEARN check and competes only in the
equal-retention supplement, never in the gate. It keeps the 8 candidates and changes only beta, which becomes the most
new-class-favouring value that breaks no replay row the base classifies correctly (``retain_offset``). Such a row
stays correct exactly while beta stays below its threshold (true-class logit - best imprinted logit), so beta must stay
below beta_max, the smallest of these thresholds. The fit rows are classified alike on each interval between
consecutive thresholds, and beta is the midpoint of the highest interval below beta_max (beta_max - c when no threshold
lies below it), as ``imprint`` takes midpoints.

The method uses the labels of the evidence; it is gradient-free and closed-form, not training-free. Only the head
changes, and PREREG E2 exempts the head from max_rel_frob, so the run's limits bind it like any other E2 update.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightcode.baselines.instance_io import Instance
from weightcode.evalkit.fewshot import FEWSHOT_RULE
from weightcode.evalkit.patch_check import predict

from .runner import CandidatePatch, Evidence, Generator, MethodSpec, execute_run, read_model_spec
from .torch_forward import PARITY_ATOL, architecture_from, assert_parity, images_tensor, penultimate, to_tensors

IMPRINT_DIRECTIONS: tuple[str, ...] = ("mean", "centered")
IMPRINT_SCALES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
"""Norm of an imprinted row over the mean norm of the train-class rows, in query order within a direction."""

IMPRINT_SPEC = MethodSpec(
    method="imprint",
    label=(
        "gradient-free label fit: new-class head rows imprinted from class-mean penultimate features (mean or centred "
        "direction, 4 scales relative to the train-class rows), one shared new-class bias offset fitted for balanced "
        "evidence accuracy; train-class rows unchanged; not training-free"
    ),
    uses_labels_for_fit=True,
    edit_limits_enforced=True,
)

IMPRINT_RETAIN_SPEC = MethodSpec(
    method="imprint_retain",
    label=(
        "supplementary to PREREG E2, never the gate: imprint's 8 candidates with the shared new-class bias offset set "
        "to the most new-class-favouring value that breaks no replay row the base classifies correctly"
    ),
    uses_labels_for_fit=True,
    edit_limits_enforced=True,
)

OFFSET_RULES: dict[str, str] = {"imprint": "balanced", "imprint_retain": "retain"}
"""How each imprint method sets the shared new-class offset: ``fit_shared_offset`` or ``retain_offset``."""
IMPRINT_SPECS: dict[str, MethodSpec] = {"imprint": IMPRINT_SPEC, "imprint_retain": IMPRINT_RETAIN_SPEC}


def candidate_grid() -> list[tuple[str, float]]:
    """The 8 (direction, scale) candidates in query order: direction outer, scale inner."""
    return [(direction, scale) for direction in IMPRINT_DIRECTIONS for scale in IMPRINT_SCALES]


def class_directions(
    h: np.ndarray, y: np.ndarray, classes: Sequence[int], center: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """Unit rows normalize(mean of normalize(h - center)) per class, and the number of rows without a direction.

    A row whose (centred) feature vector is zero has no direction and does not enter any mean. A class left without
    a row that has one is an error.
    """
    feats = np.asarray(h, dtype=np.float64) if center is None else np.asarray(h, dtype=np.float64) - center
    norms = np.linalg.norm(feats, axis=1)
    usable = norms > 0.0
    unit = np.zeros_like(feats)
    unit[usable] = feats[usable] / norms[usable, None]
    rows = []
    for c in classes:
        members = (y == c) & usable
        if not members.any():
            raise ValueError(f"class {c} has no evidence row with a nonzero feature vector")
        mean = unit[members].mean(axis=0)
        length = float(np.linalg.norm(mean))
        if not length > 0.0:
            raise ValueError(f"the normalised features of class {c} average to zero")
        rows.append(mean / length)
    return np.stack(rows), int(np.count_nonzero(~usable))


def _best_other(logits: np.ndarray, true_col: np.ndarray) -> np.ndarray:
    """Per row, the largest logit outside the true column (-inf when there is no other column)."""
    masked = logits.copy()
    masked[np.arange(len(true_col)), true_col] = -np.inf
    return masked.max(axis=1) if masked.shape[1] > 1 else np.full(len(true_col), -np.inf)


def fit_shared_offset(
    z_old: np.ndarray,
    u_new: np.ndarray,
    y: np.ndarray,
    old_classes: Sequence[int],
    new_classes: Sequence[int],
    is_new_row: np.ndarray,
    step: float,
) -> dict[str, Any]:
    """The offset beta, added to every new-class logit, that maximises the balanced evidence accuracy (module doc).

    ``z_old``: base logits of the train classes (rows x train classes), ``u_new``: imprinted logits of the new classes
    before the offset, ``is_new_row``: which evidence rows carry a new class. Returns beta, its interval, the predicted
    accuracies at beta and the number of rows of each part that some beta can classify correctly.
    """
    if not step > 0.0:
        raise ValueError(f"step must be positive, got {step}")
    new_col = {int(c): j for j, c in enumerate(new_classes)}
    old_col = {int(c): j for j, c in enumerate(old_classes)}
    new_rows, old_rows = np.flatnonzero(is_new_row), np.flatnonzero(~is_new_row)
    if not len(new_rows) or not len(old_rows):
        raise ValueError("the balanced objective needs new-class rows and replay rows")
    jn = np.array([new_col[int(c)] for c in y[new_rows]])
    jo = np.array([old_col[int(c)] for c in y[old_rows]])
    u_true = u_new[new_rows, jn]
    new_ok = u_true > _best_other(u_new[new_rows], jn)
    t_new = z_old[new_rows].max(axis=1) - u_true
    z_true = z_old[old_rows, jo]
    old_ok = z_true > _best_other(z_old[old_rows], jo)
    t_old = z_true - u_new[old_rows].max(axis=1)
    thresholds = np.unique(np.concatenate([t_new[new_ok], t_old[old_ok]]))
    if thresholds.size:
        betas = np.concatenate([[thresholds[0] - step], (thresholds[:-1] + thresholds[1:]) / 2, [thresholds[-1] + step]])
        lows = np.concatenate([[-np.inf], thresholds])
        highs = np.concatenate([thresholds, [np.inf]])
    else:
        betas, lows, highs = np.zeros(1), np.full(1, -np.inf), np.full(1, np.inf)
    acc_new = (new_ok[None, :] & (t_new[None, :] < betas[:, None])).sum(axis=1) / len(new_rows)
    acc_old = (old_ok[None, :] & (t_old[None, :] > betas[:, None])).sum(axis=1) / len(old_rows)
    balanced = (acc_new + acc_old) / 2
    widths = highs - lows
    # Highest balanced accuracy, then the widest interval, then the smallest beta (betas are increasing).
    order = np.lexsort((betas, -widths, -balanced))
    best = int(order[0])
    return {
        "beta": float(betas[best]),
        "interval": [float(lows[best]), float(highs[best])],
        "balanced_acc": float(balanced[best]),
        "acc_new": float(acc_new[best]),
        "acc_old": float(acc_old[best]),
        "n_new_rows_rankable": int(np.count_nonzero(new_ok)),
        "n_old_rows_rankable": int(np.count_nonzero(old_ok)),
        "n_thresholds": int(thresholds.size),
    }


def retain_offset(
    z_old: np.ndarray,
    u_new: np.ndarray,
    y: np.ndarray,
    base_pred: np.ndarray,
    old_classes: Sequence[int],
    new_classes: Sequence[int],
    is_new_row: np.ndarray,
    step: float,
) -> dict[str, Any]:
    """The offset of ``imprint_retain``: the most new-class-favouring beta that breaks no replay row the base
    classifies correctly (module docstring).

    Arguments as for ``fit_shared_offset``, plus ``base_pred``, the base model's predictions on the evidence rows.
    Returns beta, its interval (below ``beta_max``), beta_max, the number of correct replay rows at beta_max, the
    predicted accuracies at beta and the row counts. Raises when the base classifies no replay row correctly.
    """
    if not step > 0.0:
        raise ValueError(f"step must be positive, got {step}")
    new_col = {int(c): j for j, c in enumerate(new_classes)}
    old_col = {int(c): j for j, c in enumerate(old_classes)}
    new_rows, old_rows = np.flatnonzero(is_new_row), np.flatnonzero(~is_new_row)
    if not len(new_rows) or not len(old_rows):
        raise ValueError("the retain offset needs new-class rows and replay rows")
    jn = np.array([new_col[int(c)] for c in y[new_rows]])
    jo = np.array([old_col[int(c)] for c in y[old_rows]])
    u_true = u_new[new_rows, jn]
    new_ok = u_true > _best_other(u_new[new_rows], jn)
    t_new = z_old[new_rows].max(axis=1) - u_true
    z_true = z_old[old_rows, jo]
    old_ok = z_true > _best_other(z_old[old_rows], jo)
    t_old = z_true - u_new[old_rows].max(axis=1)
    base_right = np.asarray(base_pred)[old_rows] == y[old_rows]
    if not base_right.any():
        raise ValueError("the base classifies no replay row correctly, so the retain offset has no bound")
    beta_max = float(np.min(t_old[base_right]))
    thresholds = np.unique(np.concatenate([t_new[new_ok], t_old[old_ok], [beta_max]]))
    k = int(np.searchsorted(thresholds, beta_max))
    low = -np.inf if k == 0 else float(thresholds[k - 1])
    beta = beta_max - step if k == 0 else (low + beta_max) / 2
    acc_new = float(np.count_nonzero(new_ok & (t_new < beta)) / len(new_rows))
    acc_old = float(np.count_nonzero(old_ok & (t_old > beta)) / len(old_rows))
    return {
        "beta": float(beta),
        "interval": [low, beta_max],
        "beta_max": beta_max,
        "n_replay_base_right": int(np.count_nonzero(base_right)),
        "n_replay_binding": int(np.count_nonzero(t_old[base_right] == beta_max)),
        "balanced_acc": (acc_new + acc_old) / 2,
        "acc_new": acc_new,
        "acc_old": acc_old,
        "n_new_rows_rankable": int(np.count_nonzero(new_ok)),
        "n_old_rows_rankable": int(np.count_nonzero(old_ok)),
        "n_thresholds": int(thresholds.size),
    }


def _evidence_parts(evidence: Evidence) -> np.ndarray:
    if evidence.rule != FEWSHOT_RULE or evidence.n_fail or not evidence.n_rand or not evidence.n_replay:
        raise ValueError(f"imprint needs {FEWSHOT_RULE} evidence (new-class rows, then replay rows), got {evidence.rule}")
    return np.arange(len(evidence.y)) < evidence.n_rand


def imprint_generator(instance: Instance, method: str = "imprint") -> Generator:
    """Candidate generator for ``imprint`` or ``imprint_retain`` on one fewshot-v1 instance."""
    if method not in OFFSET_RULES:
        raise ValueError(f"unknown imprint method {method!r}; expected one of {tuple(OFFSET_RULES)}")
    retain = OFFSET_RULES[method] == "retain"
    if instance.private.get("evidence_rule") != FEWSHOT_RULE:
        raise ValueError(f"{instance.instance_id}: imprint adds classes and needs a {FEWSHOT_RULE} instance")
    old_classes = [int(c) for c in instance.private["train_classes"]]
    new_classes = [int(c) for c in instance.private["new_classes"]]

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        is_new_row = _evidence_parts(evidence)
        params = instance.params
        arch = architecture_from(params, read_model_spec(instance))
        parity = assert_parity(arch, params, instance.forward, evidence.x)
        weight_name, bias_name = arch.last_linear_tensors
        with torch.no_grad():
            h = penultimate(arch, to_tensors(params, dtype=torch.float64), images_tensor(evidence.x)).numpy()
        w0 = params[weight_name].astype(np.float64)
        b0 = params[bias_name].astype(np.float64)
        z0 = h @ w0.T + b0
        float64_parity = float(np.max(np.abs(z0 - np.asarray(instance.forward(dict(params), evidence.x)))))
        if not float64_parity <= PARITY_ATOL:
            raise RuntimeError(f"float64 features give logits {float64_parity:.3g} away from forward.py")
        y = evidence.y
        rho = float(np.mean(np.linalg.norm(w0[old_classes], axis=1)))
        step = float(np.mean(np.max(z0, axis=1) - np.mean(z0, axis=1)))
        center = (h[is_new_row].mean(axis=0) + h[~is_new_row].mean(axis=0)) / 2
        directions = {}
        n_without_direction = {}
        for direction in IMPRINT_DIRECTIONS:
            directions[direction], n_without_direction[direction] = class_directions(
                h[is_new_row], y[is_new_row], new_classes, center if direction == "centered" else None
            )
        z_old = z0[:, old_classes]
        candidates: list[CandidatePatch] = []
        for direction, scale in candidate_grid():
            rows = scale * rho * directions[direction]
            offset0 = -(rows @ center) if direction == "centered" else np.zeros(len(new_classes))
            if retain:
                fit = retain_offset(z_old, h @ rows.T + offset0, y, evidence.base_pred, old_classes, new_classes,
                                    is_new_row, step)
            else:
                fit = fit_shared_offset(z_old, h @ rows.T + offset0, y, old_classes, new_classes, is_new_row, step)
            child = dict(params)
            weight = params[weight_name].copy()
            bias = params[bias_name].copy()
            weight[new_classes] = rows.astype(weight.dtype)
            bias[new_classes] = (offset0 + fit["beta"]).astype(bias.dtype)
            child[weight_name], child[bias_name] = weight, bias
            # The child as written (float32): its evidence accuracy, which the rounding can move off the prediction.
            pred = np.argmax(h @ weight.astype(np.float64).T + bias.astype(np.float64), axis=1)
            right = pred == y
            realized_new, realized_old = float(right[is_new_row].mean()), float(right[~is_new_row].mean())
            notes: dict[str, Any] = {
                **fit,
                "realized_acc_new": realized_new,
                "realized_acc_old": realized_old,
                "realized_balanced_acc": (realized_new + realized_old) / 2,
            }
            if retain:  # checked with forward.py, the predictor of the oracle and the readouts
                kept = ~is_new_row & (evidence.base_pred == y)
                child_pred = predict(instance.forward, child, evidence.x, len(y))
                broken = int(np.count_nonzero(kept & (child_pred != y)))
                if broken:
                    raise RuntimeError(f"{direction} x {scale:g}: the child as written breaks {broken} replay rows "
                                       f"the base classifies correctly (beta {fit['beta']:.6g}, beta_max "
                                       f"{fit['beta_max']:.6g})")
                notes["realized_replay_base_right_broken"] = broken
            if retain:
                bias_rule = (f"beta = {fit['beta']:.6g}, the most new-class-favouring offset that breaks none of the "
                             f"{fit['n_replay_base_right']} replay rows the base classifies correctly (midpoint of the "
                             f"highest interval below {fit['beta_max']:.6g})")
            else:
                bias_rule = f"beta = {fit['beta']:.6g}, fitted for balanced evidence accuracy"
            candidates.append(
                CandidatePatch(
                    slot=len(candidates),
                    child=child,
                    config={"direction": direction, "scale": scale},
                    hypothesis=(
                        f"imprint: rows {new_classes} of {weight_name} set to {scale:g} x {rho:.6g} (mean train-class "
                        f"row norm) x the {direction} direction of their classes' penultimate features on the "
                        f"{len(y)} evidence images ({evidence.composition()}); {bias_name} of those classes "
                        f"{'-w.m + ' if direction == 'centered' else ''}{bias_rule}; train-class rows unchanged"
                    ),
                    predicted_effect=(
                        "the new classes are recognised where their penultimate features lie near their evidence "
                        "means; train-class images keep their predictions unless an imprinted logit overtakes them"
                    ),
                    notes=notes,
                )
            )
        cosines = {
            direction: float(
                np.mean((rows @ rows.T)[~np.eye(len(new_classes), dtype=bool)]) if len(new_classes) > 1 else np.nan
            )
            for direction, rows in directions.items()
        }
        info = {
            "architecture": arch.describe(),
            "parity_max_abs_logit_diff": parity,
            "float64_feature_parity": float64_parity,
            "train_classes": old_classes,
            "new_classes": new_classes,
            "rho_mean_train_class_row_norm": rho,
            "offset_step_logit_scale": step,
            "center_norm": float(np.linalg.norm(center)),
            "rows_without_direction": n_without_direction,
            "mean_cosine_between_new_directions": cosines,
            "trainable": [weight_name, bias_name],
            "changed_rows": new_classes,
        }
        return candidates, info

    return generate


RETAIN_BIAS_RULE = (
    "b0 + beta for every new class, beta the most new-class-favouring value that breaks no replay row the base "
    "classifies correctly: the midpoint of the highest threshold interval below beta_max, the smallest (true-class "
    "logit - best imprinted logit) over those rows (beta_max - c when no threshold lies below it)"
)


def imprint_config(run_seed: int, method: str = "imprint") -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    if method not in OFFSET_RULES:
        raise ValueError(f"unknown imprint method {method!r}; expected one of {tuple(OFFSET_RULES)}")
    retain = OFFSET_RULES[method] == "retain"
    return {
        "candidate_order": [{"direction": d, "scale": s} for d, s in candidate_grid()],
        "directions": {
            "mean": "normalize(mean of normalize(h)) over the class's evidence rows (Qi et al. 2018)",
            "centered": (
                "the same on h - m, m = (mean h of the new-class rows + mean h of the replay rows) / 2, centring "
                "folded into the bias (CL2N, SimpleShot)"
            ),
        },
        "row_norm": "scale x mean L2 norm of the train-class rows of the head",
        "bias": RETAIN_BIAS_RULE if retain else (
            "b0 + beta for every new class, beta maximising (new-row accuracy + replay-row accuracy) / 2 on the "
            "evidence; exact over the threshold midpoints, ties to the widest interval, then the smaller beta"
        ),
        "offset_rule": OFFSET_RULES[method],
        "unchanged": "train-class rows and biases of the head, every other tensor",
        "features": "penultimate activations in float64 (torch rebuild, parity-checked against forward.py)",
        "randomness": "none; the run seed only names the output directory",
        "seed": run_seed,
        "torch": torch.__version__,
    }


def run_imprint(
    instance: Instance, run_dir: Path, *, run_seed: int = 0, allow_main: bool = False, method: str = "imprint"
) -> dict[str, Any]:
    """Run ``imprint`` or ``imprint_retain`` on one fewshot-v1 instance and return its run record (``execute_run``
    with ``fewshot``)."""
    if method not in IMPRINT_SPECS:
        raise ValueError(f"unknown imprint method {method!r}; expected one of {tuple(IMPRINT_SPECS)}")
    return execute_run(
        instance=instance,
        spec=IMPRINT_SPECS[method],
        run_dir=run_dir,
        run_seed=run_seed,
        config=imprint_config(run_seed, method),
        generate=imprint_generator(instance, method),
        allow_main=allow_main,
        replay_in_fit=True,
        fewshot=True,
    )
