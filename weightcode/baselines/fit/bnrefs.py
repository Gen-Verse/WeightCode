"""Label-free BatchNorm references of PREREG E10 (``resnet20_a``) and E12 (``resnet18_up``): ``bn_adapt``, ``tent``.

Both read only the 64 condition inputs of the evidence (x_fail and x_rand, never a label, never the replay rows) and
deliver one fixed weight set per configuration, 8 configurations each, checked once by the joint oracle.

* ``bn_adapt`` (Schneider et al. 2020, "Improving robustness against common corruptions by covariate shift
  adaptation"): every BatchNorm layer normalizes with N/(N + n) times the student's running statistics plus
  n/(N + n) times the statistics of the 64 inputs at that layer, n = 64, prior strength N in ``BN_ADAPT_PRIORS``
  (N = 0 replaces them). The statistics are taken in one forward pass in which every earlier layer already uses its
  mixed statistics, as in the authors' implementation; the input statistics are the per-channel mean and the unbiased
  variance over images and positions (what ``F.batch_norm`` with momentum 1 writes as running statistics). The
  delivered model holds the mixed statistics as its running statistics, so forward.py computes the adapted network.
* ``tent`` (Wang et al. 2021, "Tent: fully test-time adaptation by entropy minimization"): the BatchNorm affine
  parameters (weight and bias of every BatchNorm layer) are updated by Adam (betas 0.9, 0.999, no weight decay) on the
  mean prediction entropy of the 64 inputs, with every BatchNorm layer normalizing by the statistics of those inputs
  (the train-mode normalization of TENT, gradients flowing through the statistics); lr in ``TENT_LRS`` x steps in
  ``TENT_STEPS``, one trajectory per lr checkpointed at each step count. The delivered model holds the affine
  parameters of the checkpoint and, as its running statistics, the statistics of the 64 inputs under them (bn_adapt
  at N = 0 with the new affine parameters), so forward.py computes the network TENT evaluated at that step. TENT
  normalizes with the biased batch variance; here the unbiased one is used in the objective and in the delivered
  statistics alike (a factor n/(n - 1) with n at least 64 x 8 x 8 = 4096 per channel), so the delivered model is the
  one the objective scored. TENT's own setting adapts online on each test batch; here every configuration is one
  fixed weight set from the same 64 inputs (PREREG E10).

Both change only BatchNorm tensors (1-D: exempt from the norm limit, counted in the changed entries), so every
candidate is within the E10 program limits (at most 1,376 + 1,376 of 271,098 entries on resnet20_a, 9,600 + 9,600 of
11,191,242 on resnet18_up). Computation in float64, the
delivered tensors cast to float32 once.
"""

from __future__ import annotations

import dataclasses

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.baselines.instance_io import Instance

from .runner import CandidatePatch, Evidence, Generator, MethodSpec, execute_run, read_model_spec
from .torch_forward import BN_FAMILIES, Architecture, architecture_from, assert_parity, bn_features, images_tensor

BN_ADAPT_PRIORS: tuple[int, ...] = (0, 16, 32, 64, 128, 256, 512, 1024)
"""Prior strengths N of bn_adapt, in query order (N = 0 replaces the statistics)."""
BN_ADAPT_PRIORS_T04_EDGE: tuple[int, ...] = tuple(16 * prior for prior in BN_ADAPT_PRIORS)
"""PREREG T04 amendment, edge extension of the reference grids (WEIGHTCODE-main, main 9f0baa9): bn_adapt's prior strengths
multiplied by 16, N in {0, 256, ..., 16384}; run label t04g3."""
T04_EDGE_NOTE = ("PREREG T04 amendment, edge extension of the reference grids (WEIGHTCODE-main, main 9f0baa9): prior "
                 "strengths multiplied by 16, N in {0, 256, ..., 16384}")
TENT_LRS: tuple[float, ...] = (1e-4, 1e-3)
TENT_STEPS: tuple[int, ...] = (1, 3, 10, 30)
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8

BN_REF_SPECS: dict[str, MethodSpec] = {
    "bn_adapt": MethodSpec(
        method="bn_adapt",
        label=("label-free reference (Schneider et al. 2020): BatchNorm statistics mixed with those of the 64 "
               "condition inputs, prior strength N in {0, 16, ..., 1024}; no labels, no optimization"),
        uses_labels_for_fit=False,
        edit_limits_enforced=True,
    ),
    "tent": MethodSpec(
        method="tent",
        label=("label-free reference (Wang et al. 2021): BatchNorm statistics of the 64 condition inputs and the "
               "BatchNorm affine parameters by Adam on their mean prediction entropy, lr {1e-4, 1e-3} x steps "
               "{1, 3, 10, 30}; no labels"),
        uses_labels_for_fit=False,
        edit_limits_enforced=True,
    ),
}
BN_REF_METHODS: tuple[str, ...] = tuple(BN_REF_SPECS)


class BnRefError(RuntimeError):
    """The instance is not a BatchNorm student, or a candidate changed a tensor it must not."""


def _tensors(params: Mapping[str, np.ndarray], trainable: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
    return {key: torch.tensor(np.asarray(value), dtype=torch.float64).requires_grad_(key in trainable)
            for key, value in params.items()}


def adapted_forward(arch: Architecture, tensors: Mapping[str, torch.Tensor], x: torch.Tensor, prior: float,
                    n_inputs: int) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Logits of ``x`` with every BatchNorm layer normalizing by its mixed statistics, and those statistics.

    Mixed statistics: a = prior / (prior + n_inputs); mean = a running_mean + (1 - a) batch mean, var likewise with
    the unbiased batch variance over images and positions. Gradients flow through the batch statistics.
    """
    from weightcode.models.resnet20a import BN_EPS

    weight = float(prior) / (float(prior) + float(n_inputs))
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def norm(h: torch.Tensor, prefix: str) -> torch.Tensor:
        batch_mean = h.mean(dim=(0, 2, 3))
        batch_var = h.var(dim=(0, 2, 3), unbiased=True)
        mean = weight * tensors[f"{prefix}.running_mean"] + (1.0 - weight) * batch_mean
        var = weight * tensors[f"{prefix}.running_var"] + (1.0 - weight) * batch_var
        stats[prefix] = (mean, var)
        scale = tensors[f"{prefix}.weight"] / torch.sqrt(var + BN_EPS)  # F.batch_norm's eval form, written out so that
        shift = tensors[f"{prefix}.bias"] - mean * scale  # gradients can flow through the batch statistics
        return h * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)

    features = bn_features(arch, tensors, x, bn=norm)
    logits = F.linear(features, tensors["fc.weight"], tensors["fc.bias"])
    return logits, stats


def _child_with(params: Mapping[str, np.ndarray], values: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    child = dict(params)
    for key, value in values.items():
        child[key] = value.detach().numpy().astype(params[key].dtype)
    return child


def _stats_values(stats: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for prefix, (mean, var) in stats.items():
        values[f"{prefix}.running_mean"] = mean
        values[f"{prefix}.running_var"] = var
    return values


def _check_only(params: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], allowed: set[str]) -> None:
    for key, parent in params.items():
        if key not in allowed and not np.array_equal(child[key], parent):
            raise BnRefError(f"{key} changed but is not a tensor of this method")


def mean_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean over rows of the Shannon entropy of softmax(logits)."""
    log_p = F.log_softmax(logits, dim=1)
    return -(log_p.exp() * log_p).sum(dim=1).mean()


def _context(instance: Instance, evidence: Evidence) -> tuple[Architecture, dict[str, Any], torch.Tensor]:
    params = instance.params
    arch = architecture_from(params, read_model_spec(instance))
    if arch.family not in BN_FAMILIES:
        raise BnRefError(f"bn_adapt and tent need a BatchNorm student {BN_FAMILIES}, got family {arch.family!r}")
    if evidence.n_replay:
        raise BnRefError("the BatchNorm references read the condition inputs only, never the replay rows")
    parity = assert_parity(arch, params, instance.forward, evidence.x)
    info: dict[str, Any] = {"architecture": arch.describe(), "parity_max_abs_logit_diff": parity,
                            "n_inputs": int(len(evidence.x)), "inputs": f"x_fail and x_rand ({evidence.composition()})"
                            ", labels not read", "dtype": "float64, delivered tensors cast to float32"}
    return arch, info, images_tensor(evidence.x)


def _eval_logits(arch: Architecture, params: Mapping[str, np.ndarray], x: torch.Tensor) -> torch.Tensor:
    """float64 logits of ``params`` with BatchNorm in eval form (forward.py's computation)."""
    tensors = _tensors(params)
    with torch.no_grad():
        return F.linear(bn_features(arch, tensors, x), tensors["fc.weight"], tensors["fc.bias"])


def _delivered_check(arch: Architecture, child: Mapping[str, np.ndarray], x: torch.Tensor,
                     reference: torch.Tensor) -> float:
    """Largest |logit| difference on the inputs between the delivered model in eval form and the adapted forward."""
    return float((_eval_logits(arch, child, x) - reference).abs().max())


def bn_adapt_generator(instance: Instance, priors: tuple[int, ...] = BN_ADAPT_PRIORS) -> Generator:
    """Candidate generator of bn_adapt: one candidate per prior strength in ``priors`` (``BN_ADAPT_PRIORS``, or
    ``BN_ADAPT_PRIORS_T04_EDGE`` for the T04 edge extension)."""

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        arch, info, x = _context(instance, evidence)
        params = instance.params
        n = len(evidence.x)
        stat_names = {f"{p}.{s}" for p in arch.bn_layers for s in ("running_mean", "running_var")}
        candidates: list[CandidatePatch] = []
        with torch.no_grad():
            base_entropy = float(mean_entropy(_eval_logits(arch, params, x)))
            for prior in priors:
                logits, stats = adapted_forward(arch, _tensors(params), x, float(prior), n)
                child = _child_with(params, _stats_values(stats))
                _check_only(params, child, stat_names)
                candidates.append(CandidatePatch(
                    slot=len(candidates), child=child, config={"N": prior},
                    hypothesis=(f"bn_adapt: the condition shifts the BatchNorm input statistics; every layer's "
                                f"statistics become {prior}/({prior} + {n}) x the student's plus {n}/({prior} + {n}) x "
                                f"those of the {n} condition inputs (sequential, one forward pass); labels not used"),
                    predicted_effect=("fewer errors on condition images when the shift is a change of feature "
                                      "statistics; the clean images then see statistics that are not theirs"),
                    notes={"mean_entropy_on_inputs": float(mean_entropy(logits)),
                           "delivered_vs_adapted_max_abs_logit_diff": _delivered_check(arch, child, x, logits)}))
        info.update({"priors": list(priors), "base_mean_entropy_on_inputs": base_entropy,
                     "rule": "mixed = N/(N+n) running + n/(N+n) batch statistic (mean; unbiased variance), n = 64"})
        return candidates, info

    return generate


def tent_generator(instance: Instance, run_seed: int) -> Generator:
    """Candidate generator of tent: one Adam trajectory per lr, checkpointed at ``TENT_STEPS``."""

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        arch, info, x = _context(instance, evidence)
        params = instance.params
        n = len(evidence.x)
        affine = tuple(f"{p}.{s}" for p in arch.bn_layers for s in ("weight", "bias"))
        allowed = set(affine) | {f"{p}.{s}" for p in arch.bn_layers for s in ("running_mean", "running_var")}
        candidates: list[CandidatePatch] = []
        wanted = set(TENT_STEPS)
        for lr in TENT_LRS:
            torch.manual_seed(run_seed)
            tensors = _tensors(params, affine)
            optimizer = torch.optim.Adam([tensors[k] for k in affine], lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS,
                                         weight_decay=0.0, foreach=False)
            for step in range(1, max(TENT_STEPS) + 1):
                optimizer.zero_grad(set_to_none=True)
                entropy = mean_entropy(adapted_forward(arch, tensors, x, 0.0, n)[0])
                entropy.backward()
                optimizer.step()
                if step not in wanted:
                    continue
                with torch.no_grad():
                    logits, stats = adapted_forward(arch, tensors, x, 0.0, n)
                    values = {k: tensors[k] for k in affine} | _stats_values(stats)
                    child = _child_with(params, values)
                    _check_only(params, child, allowed)
                    candidates.append(CandidatePatch(
                        slot=len(candidates), child=child, config={"lr": lr, "steps": step},
                        hypothesis=(f"tent: Adam lr={lr:g} for {step} steps on the BatchNorm weights and biases, "
                                    f"minimizing the mean prediction entropy of the {n} condition inputs with "
                                    "BatchNorm normalizing by their statistics; delivered with those statistics as "
                                    "running statistics; labels not used"),
                        predicted_effect=("more confident and, if confidence tracks correctness under the shift, "
                                          "more accurate predictions on condition images"),
                        notes={"mean_entropy_on_inputs": float(mean_entropy(logits)),
                               "objective_before_last_step": float(entropy.detach()),
                               "delivered_vs_adapted_max_abs_logit_diff": _delivered_check(arch, child, x, logits)}))
        info.update({"lrs": list(TENT_LRS), "steps": list(TENT_STEPS), "trained": list(affine),
                     "optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0},
                     "objective": "mean Shannon entropy of softmax over the 64 condition inputs, full batch",
                     "normalization": "statistics of the 64 inputs at every BatchNorm layer (unbiased variance), "
                                      "gradients through them",
                     "deviations": ["TENT adapts online per test batch with the biased batch variance; here one "
                                    "fixed weight set per configuration from the same 64 inputs, unbiased variance "
                                    "(PREREG E10)"]})
        return candidates, info

    return generate


def bn_config(method: str, run_seed: int, priors: tuple[int, ...] = BN_ADAPT_PRIORS) -> dict[str, Any]:
    """Run configuration recorded in run.json; an extension grid of bn_adapt notes itself (``grid``)."""
    if method == "bn_adapt":
        order = [{"N": prior} for prior in priors]
    else:
        order = [{"lr": lr, "steps": steps} for lr in TENT_LRS for steps in TENT_STEPS]
    config = {"candidate_order": order, "inputs": "the 64 condition inputs of the evidence (labels not read)",
              "seed": run_seed, "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "device": "cpu"}
    if priors != BN_ADAPT_PRIORS:
        config["grid"] = {"name": "t04_edge", "note": T04_EDGE_NOTE}
    return config


def run_bn_ref(instance: Instance, method: str, run_dir: Path, *, run_seed: int = 0, allow_main: bool = False,
               e10: bool = True, priors: tuple[int, ...] = BN_ADAPT_PRIORS) -> dict[str, Any]:
    """Run bn_adapt or tent on one E10 instance (a joint run under the E10 program limits) and return its record.
    ``priors``: bn_adapt's prior strengths (``BN_ADAPT_PRIORS_T04_EDGE`` for the T04 edge extension)."""
    if method not in BN_REF_SPECS:
        raise ValueError(f"unknown BatchNorm reference {method!r}; expected one of {BN_REF_METHODS}")
    if priors not in (BN_ADAPT_PRIORS, BN_ADAPT_PRIORS_T04_EDGE) or (method != "bn_adapt" and priors != BN_ADAPT_PRIORS):
        raise ValueError(f"{method}: prior strengths {priors} are not a registered bn_adapt grid")
    spec = BN_REF_SPECS[method]
    if priors != BN_ADAPT_PRIORS:
        spec = dataclasses.replace(spec, label=spec.label.replace("N in {0, 16, ..., 1024}", "N in {0, 256, ..., 16384}")
                                   + f"; {T04_EDGE_NOTE}")
    generate = bn_adapt_generator(instance, priors) if method == "bn_adapt" else tent_generator(instance, run_seed)
    return execute_run(instance=instance, spec=spec, run_dir=run_dir, run_seed=run_seed,
                       config=bn_config(method, run_seed, priors), generate=generate, allow_main=allow_main,
                       joint=True, replay_in_fit=False, e10=e10)


__all__ = ["BN_ADAPT_PRIORS", "BN_ADAPT_PRIORS_T04_EDGE", "BN_REF_METHODS", "BN_REF_SPECS", "BnRefError", "TENT_LRS", "TENT_STEPS",
           "adapted_forward", "bn_adapt_generator", "mean_entropy", "run_bn_ref", "tent_generator"]
