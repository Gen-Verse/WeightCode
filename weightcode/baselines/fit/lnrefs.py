"""Label-free LayerNorm references of PREREG E11 on a ``vit_lite_7_4`` instance: ``tent_ln`` and ``sar``.

Both read only the 64 condition inputs of the evidence (x_fail and x_rand, never a label, never the replay rows),
train only the LayerNorm affine parameters (weight and bias of the 15 LayerNorms: every layer's pre_norm and norm1 and
the final norm, 7,680 entries, all 1-D: exempt from the norm limit, counted in the changed entries) and deliver one
fixed weight set per configuration, lr in ``LN_LRS`` x steps in ``LN_STEPS`` (8 configurations), one trajectory per lr
checkpointed at each step count, checked once each by the joint oracle. A LayerNorm has no running statistics, so the
delivered model is the adapted network itself and forward.py computes it. Computation in float64, the delivered
tensors cast to float32 once; no dropout (the eval-form network, as forward.py).

* ``tent_ln`` (Wang et al. 2021, "Tent: fully test-time adaptation by entropy minimization", with LayerNorm as the
  normalization layer whose affine parameters are adapted, as its ViT follow-ups do): Adam (betas 0.9, 0.999, eps
  1e-8, no weight decay) on the mean prediction entropy of the 64 inputs, full batch.
* ``sar`` (Niu et al. 2023, "Towards stable test-time adaptation in dynamic wild world"): reliable entropy
  minimization with sharpness awareness. Each step: entropies of the 64 inputs; the inputs with entropy below
  E0 = ``SAR_MARGIN_FRACTION`` ln(C) are kept (set 1) and their mean entropy is the first loss; SAM's first step moves
  the parameters by rho g / ||g|| (rho ``SAR_RHO``, g the gradient of the first loss, ||g|| the L2 norm over all
  trained tensors); at the perturbed point the entropies of set 1 are recomputed and those still below E0 kept (set 2);
  their mean entropy's gradient is applied at the original point by the base optimizer, SGD with momentum
  ``SAR_MOMENTUM`` (the authors' choice). A step whose set 1 or set 2 is empty changes nothing (the authors' code would
  take the mean of an empty set); the counts are recorded. Model recovery (the authors' reset on a collapsing moving
  average of the loss) is not used (PREREG E11). The trained tensors are every LayerNorm affine as in ``tent_ln``;
  the authors freeze the top blocks of a 12-layer ViT-B, which a 7-layer model has no fixed counterpart for (recorded
  in run.json "deviations").
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.baselines.instance_io import Instance

from .bnrefs import mean_entropy
from .runner import CandidatePatch, Evidence, Generator, MethodSpec, execute_run, read_model_spec
from .torch_forward import VITLITE, Architecture, architecture_from, assert_parity, images_tensor, logits

LN_LRS: tuple[float, ...] = (1e-4, 1e-3)
LN_STEPS: tuple[int, ...] = (1, 3, 10, 30)
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
SAR_MARGIN_FRACTION = 0.4
"""E0 = 0.4 ln(C), the authors' entropy margin."""
SAR_RHO = 0.05
SAR_MOMENTUM = 0.9
SAM_EPS = 1e-12
"""Added to ||g|| in SAM's first step, as in the authors' SAM."""

LN_REF_SPECS: dict[str, MethodSpec] = {
    "tent_ln": MethodSpec(
        method="tent_ln",
        label=("label-free reference (Wang et al. 2021, LayerNorm form): the LayerNorm affine parameters by Adam on "
               "the mean prediction entropy of the 64 condition inputs, lr {1e-4, 1e-3} x steps {1, 3, 10, 30}; "
               "no labels"),
        uses_labels_for_fit=False,
        edit_limits_enforced=True,
    ),
    "sar": MethodSpec(
        method="sar",
        label=("label-free reference (Niu et al. 2023): the LayerNorm affine parameters by SAM (rho 0.05) over SGD "
               "momentum 0.9 on the mean entropy of the condition inputs below 0.4 ln C, lr {1e-4, 1e-3} x steps "
               "{1, 3, 10, 30}, no model recovery; no labels"),
        uses_labels_for_fit=False,
        edit_limits_enforced=True,
    ),
}
LN_REF_METHODS: tuple[str, ...] = tuple(LN_REF_SPECS)


class LnRefError(RuntimeError):
    """The instance is not a vit_lite_7_4 student, or a candidate changed a tensor it must not."""


def _tensors(params: Mapping[str, np.ndarray], trainable: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    return {key: torch.tensor(np.asarray(value), dtype=torch.float64).requires_grad_(key in trainable)
            for key, value in params.items()}


def ln_affine(arch: Architecture) -> tuple[str, ...]:
    """Weight and bias names of every LayerNorm, in forward order."""
    return tuple(f"{prefix}.{part}" for prefix in arch.ln_layers for part in ("weight", "bias"))


def _context(instance: Instance, evidence: Evidence) -> tuple[Architecture, dict[str, Any], torch.Tensor]:
    params = instance.params
    arch = architecture_from(params, read_model_spec(instance))
    if arch.family != VITLITE:
        raise LnRefError(f"tent_ln and sar need a vit_lite_7_4 student with LayerNorm, got family {arch.family!r}")
    if evidence.n_replay:
        raise LnRefError("the LayerNorm references read the condition inputs only, never the replay rows")
    parity = assert_parity(arch, params, instance.forward, evidence.x)
    info: dict[str, Any] = {"architecture": arch.describe(), "parity_max_abs_logit_diff": parity,
                            "n_inputs": int(len(evidence.x)), "inputs": f"x_fail and x_rand ({evidence.composition()})"
                            ", labels not read", "dtype": "float64, delivered tensors cast to float32"}
    return arch, info, images_tensor(evidence.x)


def _child(params: Mapping[str, np.ndarray], tensors: Mapping[str, torch.Tensor],
           names: tuple[str, ...]) -> dict[str, np.ndarray]:
    child = dict(params)
    for key in names:
        child[key] = tensors[key].detach().numpy().astype(params[key].dtype)
    for key, parent in params.items():
        if key not in names and not np.array_equal(child[key], parent):
            raise LnRefError(f"{key} changed but is not a LayerNorm affine tensor")
    return child


def _entropies(arch: Architecture, tensors: Mapping[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits(arch, tensors, x), dim=1)
    return -(log_p.exp() * log_p).sum(dim=1)


def _delivered_entropy(arch: Architecture, child: Mapping[str, np.ndarray], x: torch.Tensor) -> float:
    with torch.no_grad():
        return float(mean_entropy(logits(arch, _tensors(child), x)))


def tent_ln_generator(instance: Instance, run_seed: int) -> Generator:
    """Candidate generator of tent_ln: one Adam trajectory per lr, checkpointed at ``LN_STEPS``."""

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        arch, info, x = _context(instance, evidence)
        params = instance.params
        n = len(evidence.x)
        affine = ln_affine(arch)
        candidates: list[CandidatePatch] = []
        with torch.no_grad():
            base_entropy = float(mean_entropy(logits(arch, _tensors(params), x)))
        for lr in LN_LRS:
            torch.manual_seed(run_seed)
            tensors = _tensors(params, affine)
            optimizer = torch.optim.Adam([tensors[k] for k in affine], lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS,
                                         weight_decay=0.0, foreach=False)
            for step in range(1, max(LN_STEPS) + 1):
                optimizer.zero_grad(set_to_none=True)
                entropy = mean_entropy(logits(arch, tensors, x))
                entropy.backward()
                optimizer.step()
                if step not in LN_STEPS:
                    continue
                child = _child(params, tensors, affine)
                candidates.append(CandidatePatch(
                    slot=len(candidates), child=child, config={"lr": lr, "steps": step},
                    hypothesis=(f"tent_ln: Adam lr={lr:g} for {step} steps on the LayerNorm weights and biases, "
                                f"minimizing the mean prediction entropy of the {n} condition inputs; labels not used"),
                    predicted_effect=("more confident and, if confidence tracks correctness under the shift, more "
                                      "accurate predictions on condition images"),
                    notes={"mean_entropy_on_inputs": _delivered_entropy(arch, child, x),
                           "objective_before_last_step": float(entropy.detach())}))
        info.update({"lrs": list(LN_LRS), "steps": list(LN_STEPS), "trained": list(affine),
                     "base_mean_entropy_on_inputs": base_entropy,
                     "optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0},
                     "objective": "mean Shannon entropy of softmax over the 64 condition inputs, full batch",
                     "deviations": ["TENT adapts online per test batch; here one fixed weight set per configuration "
                                    "from the same 64 inputs (PREREG E11)"]})
        return candidates, info

    return generate


def _filtered_mean(entropies: torch.Tensor, margin: float) -> tuple[torch.Tensor | None, torch.Tensor]:
    keep = torch.nonzero(entropies < margin).flatten()
    return (entropies[keep].mean() if keep.numel() else None), keep


def sar_generator(instance: Instance, run_seed: int) -> Generator:
    """Candidate generator of sar: one SAM-over-SGD trajectory per lr, checkpointed at ``LN_STEPS``."""

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        arch, info, x = _context(instance, evidence)
        params = instance.params
        n = len(evidence.x)
        affine = ln_affine(arch)
        n_classes = int(params[arch.last_linear_tensors[0]].shape[0])
        margin = SAR_MARGIN_FRACTION * math.log(n_classes)
        candidates: list[CandidatePatch] = []
        with torch.no_grad():
            base_entropies = _entropies(arch, _tensors(params), x)
        trajectories: dict[str, list[dict[str, Any]]] = {}
        for lr in LN_LRS:
            torch.manual_seed(run_seed)
            tensors = _tensors(params, affine)
            trained = [tensors[k] for k in affine]
            optimizer = torch.optim.SGD(trained, lr=lr, momentum=SAR_MOMENTUM, weight_decay=0.0, foreach=False)
            log: list[dict[str, Any]] = []
            for step in range(1, max(LN_STEPS) + 1):
                optimizer.zero_grad(set_to_none=True)
                first_loss, keep1 = _filtered_mean(_entropies(arch, tensors, x), margin)
                record: dict[str, Any] = {"step": step, "n_set1": int(keep1.numel())}
                if first_loss is not None:
                    first_loss.backward()
                    with torch.no_grad():
                        norm = torch.sqrt(sum((t.grad ** 2).sum() for t in trained))
                        scale = SAR_RHO / (norm + SAM_EPS)
                        perturbation = [t.grad * scale for t in trained]
                        for t, e in zip(trained, perturbation):
                            t.add_(e)
                    optimizer.zero_grad(set_to_none=True)
                    second_loss, keep2 = _filtered_mean(_entropies(arch, tensors, x[keep1]), margin)
                    record.update(first_loss=float(first_loss.detach()), grad_norm=float(norm),
                                  n_set2=int(keep2.numel()))
                    if second_loss is not None:
                        second_loss.backward()
                        record["second_loss"] = float(second_loss.detach())
                    with torch.no_grad():
                        for t, e in zip(trained, perturbation):
                            t.sub_(e)
                    if second_loss is not None:
                        optimizer.step()
                record["updated"] = first_loss is not None and record.get("n_set2", 0) > 0
                log.append(record)
                if step not in LN_STEPS:
                    continue
                child = _child(params, tensors, affine)
                candidates.append(CandidatePatch(
                    slot=len(candidates), child=child, config={"lr": lr, "steps": step},
                    hypothesis=(f"sar: SAM (rho {SAR_RHO}) over SGD lr={lr:g} momentum {SAR_MOMENTUM} for {step} "
                                f"steps on the LayerNorm weights and biases, minimizing the mean entropy of the "
                                f"condition inputs (of {n}) whose entropy is below {margin:.4f}; labels not used"),
                    predicted_effect=("more confident predictions on condition images, with flat-minimum and "
                                      "reliable-sample filtering against collapse"),
                    notes={"mean_entropy_on_inputs": _delivered_entropy(arch, child, x),
                           "updates_so_far": sum(1 for r in log if r["updated"]), "last_step": log[-1]}))
            trajectories[f"{lr:g}"] = log
        info.update({"lrs": list(LN_LRS), "steps": list(LN_STEPS), "trained": list(affine),
                     "margin": margin, "margin_rule": f"{SAR_MARGIN_FRACTION} ln({n_classes})", "rho": SAR_RHO,
                     "base_optimizer": {"name": "SGD", "momentum": SAR_MOMENTUM, "weight_decay": 0.0},
                     "base_mean_entropy_on_inputs": float(base_entropies.mean()),
                     "base_n_below_margin": int((base_entropies < margin).sum()), "trajectories": trajectories,
                     "deviations": ["SAR adapts online per test batch with model recovery; here one fixed weight set "
                                    "per configuration from the same 64 inputs and no recovery (PREREG E11)",
                                    "SAR trains the norms below the top blocks of a 12-layer ViT-B and not its final "
                                    "norm; here every LayerNorm affine, as tent_ln"]})
        return candidates, info

    return generate


def ln_config(method: str, run_seed: int) -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    return {"candidate_order": [{"lr": lr, "steps": steps} for lr in LN_LRS for steps in LN_STEPS],
            "inputs": "the 64 condition inputs of the evidence (labels not read)", "seed": run_seed,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "device": "cpu"}


def run_ln_ref(instance: Instance, method: str, run_dir: Path, *, run_seed: int = 0, allow_main: bool = False,
               e10: bool = True) -> dict[str, Any]:
    """Run tent_ln or sar on one E11 instance (a joint run under the E10 program limits) and return its record."""
    if method not in LN_REF_SPECS:
        raise ValueError(f"unknown LayerNorm reference {method!r}; expected one of {LN_REF_METHODS}")
    generate = tent_ln_generator(instance, run_seed) if method == "tent_ln" else sar_generator(instance, run_seed)
    return execute_run(instance=instance, spec=LN_REF_SPECS[method], run_dir=run_dir, run_seed=run_seed,
                       config=ln_config(method, run_seed), generate=generate, allow_main=allow_main, joint=True,
                       replay_in_fit=False, e10=e10)


__all__ = ["LN_LRS", "LN_REF_METHODS", "LN_REF_SPECS", "LN_STEPS", "LnRefError", "SAR_MARGIN_FRACTION", "SAR_RHO",
           "ln_affine", "run_ln_ref", "sar_generator", "tent_ln_generator"]
