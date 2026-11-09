"""Gradient references: Adam on mean cross-entropy over the 64 evidence images (16 failures + 48 random).

Candidates are fixed in code: lr in {1e-3, 1e-2} x steps in {10, 30, 100, 300}, full batch, run seed 0 by default,
giving 8 candidates checked once each. For each lr one trajectory runs to 300 steps and is checkpointed at 10, 30, 100
and 300 steps. Full-batch Adam from the base weights has no randomness, so this equals four separate runs.

* ``grad_last``: only the final linear layer (weight and bias) is trained.
* ``grad_full``: every parameter is trained (BatchNorm running statistics excepted, ``trainable_tensors``) and the
  edit limits are ignored: an unconstrained reference.
* ``grad_sparse``: every layer, but only the 5% of all parameters with the largest |gradient| of the same loss at the
  base model may change. The mask is fixed before training and applied to every gradient.
* ``grad_sparse19_sal`` and ``grad_sparse19_traj`` (PREREG E10, E5's masks as E7 and E9 use them): every trainable
  tensor, but only a fixed top 19% of each tensor's entries (``floor(0.19 x size)``, stable argsort of -|score|, ties
  by flat index): by |d loss / d w| at the base weights, or by |w - w_base| after unconstrained Adam at lr 1e-3 for
  300 steps (theta_full). The same grid; gradients are masked before each step; the norm limit is enforced.
* ``interp`` (PREREG E10, E7's): base + alpha (theta_full - base) on the trainable tensors, alpha in {1/8, ..., 8/8},
  edit limits ignored like grad_full: 8 candidates, theta_full fixed before any check.

The grid is a ``GradGrid``. ``GRID_E1`` (above) is the default. ``GRID_E11_SECOND`` is PREREG E11's second reference
grid (amended 2026-11-05, main 21e3ff2): Adam lr {1e-5, 1e-4} with the same step checkpoints for grad_last, grad_full,
grad_sparse19_sal and grad_sparse19_traj, and alpha k/64 (k = 1..8) for interp. theta_full (the traj mask and interp's
target) is the same unconstrained lr 1e-3, 300-step point in both grids.

On ``grad_last`` and ``grad_sparse`` the contract 5.1 per-tensor limit (relative Frobenius norm 0.5) is enforced on
each checkpoint by scaling a tensor's delta down to the limit when it is exceeded. Every scaling is recorded. The
trajectory itself is unconstrained; the scaling acts on the checkpoint only. The tensors the run's limits exempt from
the norm limit (``Limits.rel_frob_exempt``: the head of a fewshot-v1 instance, PREREG E2) are never scaled.

Fewshot runs (``fewshot=True``, PREREG E2) train on the instance's 50 new-class rows plus its 64 replay rows of the
train classes, the same rows every E2 reference and the teacher get; the grid and the optimizer are unchanged.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import dataclasses

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import params_sha256
from weightcode.evalkit.fewshot import fewshot_program_limits

from .edits import MAX_REL_FROB, scale_to_norm_limit
from .runner import CandidatePatch, Evidence, Generator, MethodSpec, execute_run, read_model_spec
from .torch_forward import (
    Architecture,
    architecture_from,
    assert_parity,
    images_tensor,
    logits,
    to_tensors,
    trainable_tensors,
)

LEARNING_RATES: tuple[float, ...] = (1e-3, 1e-2)
STEP_CHECKPOINTS: tuple[int, ...] = (10, 30, 100, 300)
SPARSE_PERCENT = 5
MASK_FRACTION = 0.19
"""Entries trained per tensor by the sparse19 references (E5's masks)."""
TRAJ_LR = 1e-3
TRAJ_STEPS = 300
"""theta_full: the unconstrained Adam point of the traj mask and of interp (grad_full's lr 1e-3, 300-step point)."""
INTERP_ALPHAS: tuple[float, ...] = tuple(k / 8 for k in range(1, 9))
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


@dataclasses.dataclass(frozen=True)
class GradGrid:
    """The candidate grid of a gradient reference: Adam learning rates (x ``STEP_CHECKPOINTS``) and interp's alphas."""

    name: str
    learning_rates: tuple[float, ...]
    interp_alphas: tuple[float, ...]
    note: str | None = None


GRID_E1 = GradGrid(name="e1", learning_rates=LEARNING_RATES, interp_alphas=INTERP_ALPHAS)
"""E1's grid: lr {1e-3, 1e-2} x steps {10, 30, 100, 300}; interp alpha k/8."""
GRID_E11_SECOND = GradGrid(
    name="e11_second",
    learning_rates=(1e-5, 1e-4),
    interp_alphas=tuple(k / 64 for k in range(1, 9)),
    note=("PREREG E11 second reference grid (amended 2026-11-05, main 21e3ff2): Adam lr {1e-5, 1e-4} x steps "
          "{10, 30, 100, 300}; interp alpha k/64, k = 1..8, toward the same theta_full as the first grid"),
)
GRID2_METHODS: tuple[str, ...] = ("grad_last", "grad_full", "grad_sparse19_sal", "grad_sparse19_traj", "interp")
"""The references PREREG E11's second grid reruns; the others keep their one run."""
GRID_T04_EDGE = GradGrid(
    name="t04_edge",
    learning_rates=tuple(float(f"{lr / 10:.12g}") for lr in GRID_E11_SECOND.learning_rates),
    interp_alphas=tuple(alpha / 8 for alpha in GRID_E11_SECOND.interp_alphas),
    note=("PREREG T04 amendment, edge extension of the reference grids (WEIGHTCODE-main, main 9f0baa9): the second grid "
          "with every learning rate divided by 10 (Adam lr {1e-6, 1e-5} x steps {10, 30, 100, 300}) and every interp "
          "alpha divided by 8 (k/512, k = 1..8, toward the same theta_full); run label t04g3"),
)
"""The T04 edge extension of ``GRID_E11_SECOND``, for the (method, condition) pairs whose primary or kept pick lies on
its conservative edge (scripts/t04/edge_extension.py)."""


def _alpha_range(alphas: tuple[float, ...]) -> str:
    """The alphas' first and last as k/den over the smallest den in (8, 64, 512) that writes every alpha exactly."""
    for den in (8, 64, 512):
        if all(abs(a * den - round(a * den)) < 1e-12 for a in alphas):
            return f"{{{round(alphas[0] * den)}/{den} .. {round(alphas[-1] * den)}/{den}}}"
    return f"{{{alphas[0]:g} .. {alphas[-1]:g}}}"


def grid_spec(method: str, grid: GradGrid = GRID_E1) -> MethodSpec:
    """The MethodSpec of ``method`` under ``grid``: E1's as is; another grid notes itself in the label."""
    spec = GRAD_SPECS[method]
    if grid == GRID_E1:
        return spec
    if method not in GRID2_METHODS:
        raise ValueError(f"{method} has no {grid.name} grid; expected one of {GRID2_METHODS}")
    label = spec.label
    if method == "interp":
        label = label.replace("alpha in {1/8 .. 8/8}", f"alpha in {_alpha_range(grid.interp_alphas)}")
    return dataclasses.replace(spec, label=f"{label}; {grid.note}")

GRAD_SPECS: dict[str, MethodSpec] = {
    "grad_last": MethodSpec(
        method="grad_last",
        label="gradient reference: Adam on the final linear layer, per-tensor norm limit enforced; not training-free",
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
    "grad_full": MethodSpec(
        method="grad_full",
        label="unconstrained gradient reference: Adam on all parameters, edit limits ignored; not training-free",
        uses_labels_for_fit=True,
        edit_limits_enforced=False,
    ),
    "grad_sparse19_sal": MethodSpec(
        method="grad_sparse19_sal",
        label=("gradient reference: Adam on a fixed top 19% of each trainable tensor's entries by base gradient "
               "saliency, per-tensor norm limit enforced; not training-free"),
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
    "grad_sparse19_traj": MethodSpec(
        method="grad_sparse19_traj",
        label=("gradient reference: Adam on a fixed top 19% of each trainable tensor's entries by |theta_full - base|, "
               "per-tensor norm limit enforced; not training-free"),
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
    "interp": MethodSpec(
        method="interp",
        label=("gradient reference: base + alpha (theta_full - base), alpha in {1/8 .. 8/8}, theta_full unconstrained "
               "Adam lr 1e-3 for 300 steps; edit limits ignored; not training-free"),
        uses_labels_for_fit=True,
        edit_limits_enforced=False,
    ),
    "grad_sparse": MethodSpec(
        method="grad_sparse",
        label=(
            "gradient reference: Adam on the top 5% |gradient| parameters (mask fixed at the base model), per-tensor "
            "norm limit enforced; not training-free"
        ),
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
}
GRAD_METHODS: tuple[str, ...] = tuple(GRAD_SPECS)


def candidate_grid(grid: GradGrid = GRID_E1) -> list[tuple[float, int]]:
    """The 8 (lr, steps) candidates in query order: lr outer, steps inner."""
    return [(lr, steps) for lr in grid.learning_rates for steps in STEP_CHECKPOINTS]


def evidence_loss(
    arch: Architecture, tensors: Mapping[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Mean cross-entropy of the rebuilt model on the evidence images."""
    return F.cross_entropy(logits(arch, tensors, x), y)


def sparse_mask(
    arch: Architecture,
    params: Mapping[str, np.ndarray],
    x: torch.Tensor,
    y: torch.Tensor,
    percent: int = SPARSE_PERCENT,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Boolean masks of the ``percent``% of all parameters with the largest |d loss / d theta| at ``params``.

    k = floor(n_params * percent / 100), so the changed fraction cannot exceed the limit. Ties are broken by the
    position in the sorted-key flattening (stable sort), so the mask is deterministic.
    """
    keys = sorted(params)
    tensors = to_tensors(params, trainable=keys)
    grads = torch.autograd.grad(evidence_loss(arch, tensors, x, y), [tensors[key] for key in keys])
    magnitude = np.concatenate([grad.detach().abs().to(torch.float64).numpy().ravel() for grad in grads])
    n_params = int(magnitude.size)
    k = n_params * percent // 100
    if k < 1:
        raise ValueError(f"{percent}% of {n_params} parameters is less than one")
    order = np.argsort(-magnitude, kind="stable")
    selected = np.zeros(n_params, dtype=bool)
    selected[order[:k]] = True
    masks: dict[str, np.ndarray] = {}
    offset = 0
    for key in keys:
        size = int(params[key].size)
        masks[key] = selected[offset : offset + size].reshape(params[key].shape)
        offset += size
    info = {
        "percent": percent,
        "k": k,
        "n_params": n_params,
        "abs_grad_threshold": float(magnitude[order[k - 1]]),
        "n_nonzero_grad": int(np.count_nonzero(magnitude)),
        "n_selected_with_zero_grad": int(np.count_nonzero(selected & (magnitude == 0.0))),
        "selected_per_tensor": {key: int(np.count_nonzero(mask)) for key, mask in masks.items()},
    }
    return masks, info


def top_fraction_masks(score: Mapping[str, np.ndarray], keys: list[str],
                       fraction: float = MASK_FRACTION) -> dict[str, np.ndarray]:
    """Per tensor, the top ``floor(fraction x size)`` entries by |score| (stable argsort of -|score|, ties by flat
    index), E5's mask rule."""
    masks = {}
    for key in keys:
        flat = np.abs(np.asarray(score[key], dtype=np.float64)).ravel()
        k = int(np.floor(fraction * flat.size))
        if k < 1:
            raise ValueError(f"{key}: {flat.size} entries give an empty {fraction:g} mask")
        order = np.argsort(-flat, kind="stable")
        mask = np.zeros(flat.size, dtype=bool)
        mask[order[:k]] = True
        masks[key] = mask.reshape(np.shape(score[key]))
    return masks


def theta_full(arch: Architecture, params: Mapping[str, np.ndarray], x: torch.Tensor, y: torch.Tensor,
               trainable: list[str], seed: int) -> dict[str, np.ndarray]:
    """Unconstrained full-batch Adam at TRAJ_LR for TRAJ_STEPS steps on the trainable tensors."""
    ((_, theta, _),) = adam_checkpoints(arch, params, x, y, trainable=trainable, lr=TRAJ_LR,
                                        checkpoints=(TRAJ_STEPS,), seed=seed)
    return theta


def adam_checkpoints(
    arch: Architecture,
    params: Mapping[str, np.ndarray],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    trainable: list[str],
    lr: float,
    checkpoints: tuple[int, ...],
    seed: int,
    masks: Mapping[str, np.ndarray] | None = None,
) -> list[tuple[int, dict[str, np.ndarray], float]]:
    """Full-batch Adam on the evidence loss; return (steps, params, loss) at each checkpoint.

    With ``masks``, every gradient is multiplied by its mask before the step, so masked-out entries keep their base
    values exactly (zero gradient and zero Adam moments give a zero update).
    """
    torch.manual_seed(seed)
    tensors = to_tensors(params, trainable=trainable)
    gradient_masks = None
    if masks is not None:
        gradient_masks = {key: torch.from_numpy(masks[key].astype(np.float32)) for key in trainable}
    optimizer = torch.optim.Adam(
        [tensors[key] for key in trainable], lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=0.0, foreach=False
    )
    wanted = set(checkpoints)
    snapshots: list[tuple[int, dict[str, np.ndarray], float]] = []
    for step in range(1, max(checkpoints) + 1):
        optimizer.zero_grad(set_to_none=True)
        evidence_loss(arch, tensors, x, y).backward()
        if gradient_masks is not None:
            for key in trainable:
                grad = tensors[key].grad
                assert grad is not None
                grad.mul_(gradient_masks[key])
        optimizer.step()
        if step in wanted:
            with torch.no_grad():
                loss = float(evidence_loss(arch, tensors, x, y))
            snapshots.append((step, {key: value.detach().numpy().copy() for key, value in tensors.items()}, loss))
    return snapshots


def _check_frozen_entries(
    base: Mapping[str, np.ndarray],
    child: Mapping[str, np.ndarray],
    trainable: list[str],
    masks: Mapping[str, np.ndarray] | None,
) -> None:
    for key, parent in base.items():
        if key not in trainable:
            if not np.array_equal(child[key], parent):
                raise RuntimeError(f"{key} is not trainable but changed during training")
        elif masks is not None and not np.array_equal(child[key][~masks[key]], parent[~masks[key]]):
            raise RuntimeError(f"{key} changed outside its sparse mask")


def _hypothesis(method: str, lr: float, steps: int, evidence: Evidence, trainable: list[str]) -> str:
    scope = {
        "grad_last": f"final layer only ({', '.join(trainable)})",
        "grad_full": f"all {len(trainable)} trainable tensors, no edit limits",
        "grad_sparse": f"all tensors under a fixed top-{SPARSE_PERCENT}% |gradient| mask",
        "grad_sparse19_sal": f"every trainable tensor under a fixed top {MASK_FRACTION:g} base-saliency mask",
        "grad_sparse19_traj": f"every trainable tensor under a fixed top {MASK_FRACTION:g} trajectory mask",
    }[method]
    return (
        f"{method}: Adam lr={lr:g} for {steps} full-batch steps on mean cross-entropy over the {len(evidence.y)} "
        f"evidence images ({evidence.composition()}); trained: {scope}"
    )


def _predicted_effect(evidence: Evidence) -> str:
    retention = (
        f"retention is fitted on the {evidence.n_replay} clean replay images"
        if evidence.n_replay
        else f"retention is not modelled beyond the {evidence.n_rand} random evidence images"
    )
    return f"fewer errors on held-out deployment images if the evidence fit generalizes; {retention}"


def grad_generator(
    instance: Instance, method: str, run_seed: int, rel_frob_exempt: Collection[str] = (), grid: GradGrid = GRID_E1
) -> Generator:
    """Candidate generator for one gradient reference on one instance.

    ``rel_frob_exempt``: tensors the run's limits exempt from the norm limit; their deltas are never scaled.
    ``grid``: the learning rates (or interp's alphas) of the 8 candidates.
    """
    if method not in GRAD_SPECS:
        raise ValueError(f"unknown gradient method {method!r}; expected one of {GRAD_METHODS}")
    spec = grid_spec(method, grid)
    exempt = tuple(rel_frob_exempt)

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        params = instance.params
        arch = architecture_from(params, read_model_spec(instance))
        parity = assert_parity(arch, params, instance.forward, evidence.x)
        x = images_tensor(evidence.x)
        y = torch.from_numpy(evidence.y.astype(np.int64))
        masks: dict[str, np.ndarray] | None = None
        info: dict[str, Any] = {"architecture": arch.describe(), "parity_max_abs_logit_diff": parity}
        if method == "grad_last":
            trainable = list(arch.last_linear_tensors)
        elif method == "grad_sparse":
            trainable = sorted(params)
        else:
            trainable = trainable_tensors(arch, params)
        if method == "grad_sparse":
            masks, info["mask"] = sparse_mask(arch, params, x, y)
        theta: dict[str, np.ndarray] | None = None
        if method in ("grad_sparse19_traj", "interp"):
            theta = theta_full(arch, params, x, y, trainable, run_seed)
            with torch.no_grad():
                theta_loss = float(evidence_loss(arch, to_tensors(theta), x, y))
            info["theta_full"] = {"lr": TRAJ_LR, "steps": TRAJ_STEPS, "sha256": params_sha256(theta),
                                  "evidence_loss": theta_loss}
        if method.startswith("grad_sparse19"):
            if method == "grad_sparse19_sal":
                tensors = to_tensors(params, trainable=trainable)
                grads = torch.autograd.grad(evidence_loss(arch, tensors, x, y), [tensors[k] for k in trainable])
                score = {k: g.detach().to(torch.float64).numpy() for k, g in zip(trainable, grads)}
            else:
                assert theta is not None
                score = {k: theta[k].astype(np.float64) - params[k].astype(np.float64) for k in trainable}
            masks = top_fraction_masks(score, trainable)
            n_mask = int(sum(int(m.sum()) for m in masks.values()))
            n_all = int(sum(v.size for v in params.values()))
            info["mask"] = {"rule": method, "fraction_per_tensor": MASK_FRACTION, "n_mask_entries": n_mask,
                            "frac_of_all_entries": n_mask / n_all}
        info["trainable"] = trainable
        if grid != GRID_E1:  # E1-grid records keep their shape
            info["grid"] = grid.name
        if exempt:
            info["norm_limit_exempt"] = list(exempt)
        with torch.no_grad():
            info["base_evidence_loss"] = float(evidence_loss(arch, to_tensors(params), x, y))
        candidates: list[CandidatePatch] = []
        if method == "interp":
            assert theta is not None
            for alpha in grid.interp_alphas:
                child = dict(params)
                for key in trainable:
                    base64 = params[key].astype(np.float64)
                    child[key] = (base64 + alpha * (theta[key].astype(np.float64) - base64)).astype(params[key].dtype)
                with torch.no_grad():
                    loss_alpha = float(evidence_loss(arch, to_tensors(child), x, y))
                candidates.append(CandidatePatch(
                    slot=len(candidates), child=child, config={"alpha": alpha},
                    hypothesis=(f"interp: base + {alpha:g} x (theta_full - base) on the trainable tensors, theta_full "
                                f"= Adam lr={TRAJ_LR:g} for {TRAJ_STEPS} full-batch steps on mean cross-entropy over "
                                f"the {len(evidence.y)} evidence images ({evidence.composition()}); no edit limits"),
                    predicted_effect=("a partial step toward the fitted point keeps more of the base's clean "
                                      "behaviour at some cost in the fitted gain"),
                    notes={"evidence_loss_at_alpha": loss_alpha}))
            if grid.interp_alphas[-1] == 1.0 and params_sha256(candidates[-1].child) != params_sha256(theta):
                raise RuntimeError("interp at alpha 1 is not theta_full")
            return candidates, info
        for lr in grid.learning_rates:
            for steps, child, loss in adam_checkpoints(
                arch, params, x, y, trainable=trainable, lr=lr, checkpoints=STEP_CHECKPOINTS, seed=run_seed, masks=masks
            ):
                _check_frozen_entries(params, child, trainable, masks)
                notes: dict[str, Any] = {"evidence_loss_at_checkpoint": loss}
                if spec.edit_limits_enforced:
                    child, scaling = scale_to_norm_limit(params, child, trainable, exempt=exempt)
                    scaled = [key for key, entry in scaling.items() if entry["scaled"]]
                    notes["norm_scaling"] = scaling
                    notes["norm_scaled_tensors"] = scaled
                    if scaled:
                        with torch.no_grad():
                            notes["evidence_loss_after_scaling"] = float(evidence_loss(arch, to_tensors(child), x, y))
                candidates.append(
                    CandidatePatch(
                        slot=len(candidates),
                        child=child,
                        config={"lr": lr, "steps": steps},
                        hypothesis=_hypothesis(method, lr, steps, evidence, trainable),
                        predicted_effect=_predicted_effect(evidence),
                        notes=notes,
                    )
                )
        info["n_candidates_norm_scaled"] = sum(1 for c in candidates if c.notes.get("norm_scaled_tensors"))
        return candidates, info

    return generate


def grad_config(method: str, run_seed: int, joint: bool = False, fewshot: bool = False,
                grid: GradGrid = GRID_E1) -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    if fewshot:
        loss = "mean cross-entropy over all evidence images (the new-class rows, then the replay rows)"
    else:
        loss = "mean cross-entropy over all evidence images (x_fail then x_rand" + (
            ", then the replay rows)" if joint else ")"
        )
    config: dict[str, Any] = {
        "learning_rates": list(grid.learning_rates),
        "step_checkpoints": list(STEP_CHECKPOINTS),
        "candidate_order": ([{"alpha": alpha} for alpha in grid.interp_alphas] if method == "interp"
                            else [{"lr": lr, "steps": steps} for lr, steps in candidate_grid(grid)]),
        "optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0},
        "loss": loss,
        "batch": "full",
        "dtype": "float32",
        "seed": run_seed,
        "norm_limit": MAX_REL_FROB if GRAD_SPECS[method].edit_limits_enforced else None,
        "sparse_percent": SPARSE_PERCENT if method == "grad_sparse" else None,
        "mask_fraction_per_tensor": MASK_FRACTION if method.startswith("grad_sparse19") else None,
        "theta_full": ({"lr": TRAJ_LR, "steps": TRAJ_STEPS} if method in ("grad_sparse19_traj", "interp") else None),
        "interp_alphas": list(grid.interp_alphas) if method == "interp" else None,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
    }
    if grid != GRID_E1:
        config["grid"] = {"name": grid.name, "note": grid.note}
    return config


def run_grad(
    instance: Instance,
    method: str,
    run_dir: Path,
    *,
    run_seed: int = 0,
    allow_main: bool = False,
    joint: bool = False,
    fewshot: bool = False,
    e10: bool = False,
    grid: GradGrid = GRID_E1,
) -> dict[str, Any]:
    """Run one gradient reference on one instance and return its run record (also written to run.json).

    ``joint``: fit on the evidence plus the replay rows and score with the joint oracle (``execute_run``).
    ``fewshot``: a fewshot-v1 instance, fitted on its new-class and replay rows under ``fewshot_program_limits``.
    ``grid``: the candidate grid (default E1's; ``GRID_E11_SECOND`` for PREREG E11's second grid).
    """
    exempt = fewshot_program_limits(instance.task["limits"]).rel_frob_exempt if fewshot else ()
    config = grad_config(method, run_seed, joint, fewshot, grid)
    spec = grid_spec(method, grid)
    if exempt and spec.edit_limits_enforced:
        config["norm_limit_exempt"] = list(exempt)
    return execute_run(
        instance=instance,
        spec=spec,
        run_dir=run_dir,
        run_seed=run_seed,
        config=config,
        generate=grad_generator(instance, method, run_seed, exempt, grid),
        allow_main=allow_main,
        joint=joint,
        replay_in_fit=True,
        e10=e10,
        fewshot=fewshot,
    )
