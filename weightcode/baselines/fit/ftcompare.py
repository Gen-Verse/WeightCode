"""PREREG F1, the fine-tuning frontier at equal information (user 2026-11-05 about 11:35, main b67036f).

Six fine-tuning methods on an E11 student (cifar10_vl7, ViT-Lite-7/4), on exactly the experience rows, the joint
check and the kept rule of the E11 references: the 64 condition rows and the 64 clean replay rows with the labels the
references use, 8 configurations each, one check per configuration, scored by the joint oracle (S_joint = S_new +
S_old). Full-batch Adam (betas 0.9/0.999, eps 1e-8, no weight decay) from the base weights, run seed 0; one
trajectory per (weight, lr), checkpointed at the listed steps. Full-batch Adam from fixed weights has no randomness,
so a checkpoint equals a separate run of that length. LoRA's A matrices are the only random draw (``lora_init``).

* ``ft_ln``: the 15 LayerNorms' affine weights and biases only (30 tensors, 7,680 entries), mean cross-entropy over
  the 128 rows; lr {1e-5, 1e-4} x steps {10, 30, 100, 300}.
* ``lora_r4`` and ``lora_r16``: W = W0 + (alpha / r) B A on the qkv, attention-projection, linear1 and linear2 weights
  of the 7 blocks (28 matrices; biases and every other tensor frozen), alpha = r so the scale is 1, B = 0 and A drawn
  as torch's nn.Linear initialisation (uniform in +-1/sqrt(fan_in)) from a generator seeded by the run seed, the
  same A for both learning rates. Mean cross-entropy over the 128 rows; lr {1e-4, 1e-3} x steps {10, 30, 100, 300}.
  The merged weights are the candidate, so the check reads the model the training forward used.
* ``ft_full_noreplay``: every tensor, mean cross-entropy over the 64 condition rows only (the replay ablation; the
  runner gives the generator the deployment rows, the joint oracle still scores); lr {1e-5, 1e-4} x steps
  {10, 30, 100, 300}.
* ``ft_full_lwf``: every tensor, mean cross-entropy over the 128 rows plus w KL(p_base || p_theta) averaged over the 64
  clean replay rows (temperature 1, p_base the base model's softmax on those rows, fixed before training); w {1, 10}
  x lr {1e-5, 1e-4} x steps {30, 300}.
* ``ft_full_l2sp``: every tensor, mean cross-entropy over the 128 rows plus (w / 2) ||theta - theta_base||^2 summed
  over every entry (L2-SP of Li et al. 2018 with every tensor in the pretrained set); w {1e-2, 1e-1} x lr
  {1e-5, 1e-4} x steps {30, 300}.

Every F1 method is a fine-tuning method as practised, so none is held to the E10 program limits (like grad_full and
interp): the executor applies each candidate without the size limits, and each checked candidate still records
whether it falls within them (``edit.within_limits``). F1's "full" and "last layer" are the E11 second-grid
``grad_full`` and ``grad_last`` runs (label ``e11g2``: Adam lr {1e-5, 1e-4} x steps {10, 30, 100, 300} on the same
128 rows), which F1 reads and does not rerun.

PREREG E12 runs F1's eight methods on its ResNet-18 student (dataset ``cifar10_r18``, family ``resnet18_up``), with
the same rows, grids, oracle, executor and run label:

* ``ft_bn`` stands for ``ft_ln``, since the student has no LayerNorm. It trains the 20 BatchNorm layers' affine
  weights and biases only (40 tensors, 9,600 entries) on ft_ln's grid, with the running statistics fixed (eval form,
  as in grad_full).
* ``lora_r4`` and ``lora_r16`` adapt every convolution weight (20) and the fc weight, as F4 defines LoRA for
  ResNet-20. A convolution [out, in, kh, kw] is adapted on its (out, in x kh x kw) reshape, W = W0 + (B A).reshape,
  with fan_in = in x kh x kw for A's draw. BatchNorm is not trained.
* ``ft_full_*`` train every tensor but the BatchNorm running statistics (``trainable_tensors``).
* F1's "full" and "last layer" are the E12 second-grid runs (label ``e12g2``).

PREREG T04 runs the same eight on the native ImageNet ResNet-18 (dataset ``in1k_t04``, family ``resnet18_native_u8``)
exactly as on E12: ``ft_bn`` for ``ft_ln``, LoRA on the 20 convolutions and fc.weight, the T04 second grid (label
``t04g2``) as F1's "full" and "last layer".
"""

from __future__ import annotations

import dataclasses

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.baselines.instance_io import Instance

from .grad import ADAM_BETAS, ADAM_EPS, STEP_CHECKPOINTS, _check_frozen_entries, adam_checkpoints, evidence_loss
from .lnrefs import ln_affine
from .runner import CandidatePatch, Evidence, Generator, MethodSpec, execute_run, read_model_spec
from .torch_forward import (
    RESNET18_FAMILIES,
    VITLITE,
    Architecture,
    architecture_from,
    assert_parity,
    images_tensor,
    logits,
    to_tensors,
    trainable_tensors,
)

PREREG_F1 = "PREREG F1 (fine-tuning comparisons, user 2026-11-05 about 11:35 Beijing, main b67036f)"
F1_DATASET = "cifar10_vl7"
F1_LABEL = "f1"
"""Run-directory label of F1 runs: ``baselines/<instance_id>/<method>/f1/<seed>/``."""

FT_LRS: tuple[float, ...] = (1e-5, 1e-4)
LORA_LRS: tuple[float, ...] = (1e-4, 1e-3)
EDGE_LR_FACTOR = 0.1
"""PREREG T04 amendment, edge extension of the reference grids (WEIGHTCODE-main, main 9f0baa9): an extended F1 method's 8
configurations are its own with every learning rate divided by 10, run label t04g3 (T04 instances only)."""
T04_EDGE_NOTE = ("PREREG T04 amendment, edge extension of the reference grids (WEIGHTCODE-main, main 9f0baa9): the F1 grid "
                 "with every learning rate divided by 10, the same steps and weights")
REG_STEPS: tuple[int, ...] = (30, 300)
LWF_WEIGHTS: tuple[float, ...] = (1.0, 10.0)
L2SP_WEIGHTS: tuple[float, ...] = (1e-2, 1e-1)
LWF_TEMPERATURE = 1.0
LORA_RANKS: dict[str, int] = {"lora_r4": 4, "lora_r16": 16}
LORA_MODULES: tuple[str, ...] = ("self_attn.qkv", "self_attn.proj", "linear1", "linear2")
LORA_ALPHA_OVER_RANK = 1.0
"""alpha = r, so the LoRA scale alpha / r is 1 at both ranks."""

F1_REUSED: dict[str, tuple[str, str]] = {"full": ("grad_full", "e11g2"), "last layer": ("grad_last", "e11g2")}
"""F1's full and last-layer methods: existing E11 second-grid runs (method, run label), read, not rerun."""

_TAIL = f"edit limits ignored; not training-free; {PREREG_F1}"
F1_SPECS: dict[str, MethodSpec] = {
    "ft_ln": MethodSpec(
        method="ft_ln",
        label=f"F1 fine-tuning: Adam on the LayerNorm affine weights and biases only, condition and replay rows; {_TAIL}",
        uses_labels_for_fit=True, edit_limits_enforced=False),
    "lora_r4": MethodSpec(
        method="lora_r4",
        label=("F1 fine-tuning: LoRA rank 4 (alpha 4) on qkv, proj, linear1 and linear2, merged, condition and replay "
               f"rows; {_TAIL}"),
        uses_labels_for_fit=True, edit_limits_enforced=False),
    "lora_r16": MethodSpec(
        method="lora_r16",
        label=("F1 fine-tuning: LoRA rank 16 (alpha 16) on qkv, proj, linear1 and linear2, merged, condition and "
               f"replay rows; {_TAIL}"),
        uses_labels_for_fit=True, edit_limits_enforced=False),
    "ft_full_noreplay": MethodSpec(
        method="ft_full_noreplay",
        label=f"F1 fine-tuning: Adam on all parameters, condition rows only (no replay); {_TAIL}",
        uses_labels_for_fit=True, edit_limits_enforced=False),
    "ft_full_lwf": MethodSpec(
        method="ft_full_lwf",
        label=("F1 fine-tuning: Adam on all parameters, condition and replay rows, plus LwF (KL to the base's outputs "
               f"on the replay rows); {_TAIL}"),
        uses_labels_for_fit=True, edit_limits_enforced=False),
    "ft_full_l2sp": MethodSpec(
        method="ft_full_l2sp",
        label=f"F1 fine-tuning: Adam on all parameters, condition and replay rows, plus L2-SP toward the base; {_TAIL}",
        uses_labels_for_fit=True, edit_limits_enforced=False),
}
F1_METHODS: tuple[str, ...] = tuple(F1_SPECS)
"""F1's six methods on an E11 (ViT-Lite-7/4) student."""
REGULARIZED = {"ft_full_lwf": LWF_WEIGHTS, "ft_full_l2sp": L2SP_WEIGHTS}

F1_E12_DATASET = "cifar10_r18"
"""PREREG E12's dataset, on which F1's methods run with ``ft_bn`` for ``ft_ln`` (module docstring)."""
F1_SPECS["ft_bn"] = MethodSpec(
    method="ft_bn",
    label=("F1 fine-tuning on a BatchNorm student (PREREG E12, for ft_ln): Adam on the BatchNorm affine weights and "
           f"biases only, running statistics fixed, condition and replay rows; {_TAIL}"),
    uses_labels_for_fit=True, edit_limits_enforced=False)
F1_E12_METHODS: tuple[str, ...] = ("ft_bn", "lora_r4", "lora_r16", "ft_full_noreplay", "ft_full_lwf", "ft_full_l2sp")
"""F1's six methods on the E12 (resnet18_up) student."""
F1_T04_DATASET = "in1k_t04"
"""PREREG T04's dataset (``weightcode.data.t04.DATASET``): the native ImageNet ResNet-18, F1's methods as on E12."""
F1_DATASET_METHODS: dict[str, tuple[str, ...]] = {F1_DATASET: F1_METHODS, F1_E12_DATASET: F1_E12_METHODS,
                                                  F1_T04_DATASET: F1_E12_METHODS}
F1_E12_REUSED: dict[str, tuple[str, str]] = {"full": ("grad_full", "e12g2"), "last layer": ("grad_last", "e12g2")}
F1_T04_REUSED: dict[str, tuple[str, str]] = {"full": ("grad_full", "t04g2"), "last layer": ("grad_last", "t04g2")}
"""F1's full and last-layer methods on T04: the T04 second-grid runs (method, run label), read, not rerun."""
F1_DATASET_REUSED: dict[str, dict[str, tuple[str, str]]] = {F1_DATASET: F1_REUSED, F1_E12_DATASET: F1_E12_REUSED,
                                                            F1_T04_DATASET: F1_T04_REUSED}
"""F1's full and last-layer methods on E12: the E12 second-grid runs (method, run label), read, not rerun."""
LORA_CONV_RULE = "every convolution weight on its (out, in x kh x kw) reshape, and fc.weight (F4's ResNet LoRA)"


class F1Error(ValueError):
    """An F1 run was refused: unknown method, or an instance that is not an E11 (cifar10_vl7) instance."""


def _edge_lrs(lrs: tuple[float, ...], edge: bool) -> tuple[float, ...]:
    """``lrs``, or with ``edge`` each divided by 10 and written with 12 significant digits (1e-06, not 1.0e-06 + ulp)."""
    return tuple(float(f"{lr * EDGE_LR_FACTOR:.12g}") for lr in lrs) if edge else lrs


def f1_configs(method: str, edge: bool = False) -> list[dict[str, Any]]:
    """The 8 configurations of ``method`` in check order: weight (if any) outer, then lr, then steps. ``edge``: the T04
    edge extension, every learning rate divided by 10."""
    ft_lrs, lora_lrs = _edge_lrs(FT_LRS, edge), _edge_lrs(LORA_LRS, edge)
    if method in ("ft_ln", "ft_bn", "ft_full_noreplay"):
        return [{"lr": lr, "steps": steps} for lr in ft_lrs for steps in STEP_CHECKPOINTS]
    if method in LORA_RANKS:
        return [{"lr": lr, "steps": steps} for lr in lora_lrs for steps in STEP_CHECKPOINTS]
    if method in REGULARIZED:
        return [{"weight": w, "lr": lr, "steps": steps}
                for w in REGULARIZED[method] for lr in ft_lrs for steps in REG_STEPS]
    raise F1Error(f"unknown F1 method {method!r}; expected one of {F1_METHODS}")


def lora_targets(arch: Architecture, params: Mapping[str, np.ndarray]) -> list[str]:
    """The weights LoRA adapts, in forward order: on vit_lite_7_4 the 28 matrices (block, then ``LORA_MODULES``); on
    resnet18_up the 20 convolution weights and fc.weight (``LORA_CONV_RULE``)."""
    if arch.family in RESNET18_FAMILIES:
        names = [f"{layer}.weight" for layer in arch.conv_layers] + [f"{arch.last_linear}.weight"]
        for name in names:
            if name not in params or np.ndim(params[name]) not in (2, 4):
                raise F1Error(f"LoRA target {name} is not a 2-D or 4-D weight of the student")
        return names
    from weightcode.models import vitlite as V

    if arch.family != VITLITE:
        raise F1Error(f"LoRA targets are defined for {VITLITE} and {RESNET18_FAMILIES} only, not {arch.family}")
    names = [f"classifier.blocks.{i}.{module}.weight" for i in range(V.N_LAYERS) for module in LORA_MODULES]
    for name in names:
        if name not in params or np.ndim(params[name]) != 2:
            raise F1Error(f"LoRA target {name} is not a 2-D tensor of the student")
    return names


def lora_matrix_shape(shape: Sequence[int]) -> tuple[int, int]:
    """(fan_out, fan_in) of a LoRA target: a matrix as it is, a convolution [out, in, kh, kw] as (out, in kh kw)."""
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 4:
        return int(shape[0]), int(shape[1] * shape[2] * shape[3])
    raise F1Error(f"a LoRA target must be 2-D or 4-D, got shape {list(shape)}")


def lora_init(params: Mapping[str, np.ndarray], targets: Sequence[str], rank: int,
              seed: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A (rank x fan_in, uniform in +-1/sqrt(fan_in), nn.Linear's default) and B (fan_out x rank, zeros) per target,
    drawn in target order from one generator seeded by ``seed``; a convolution's fan_in is in x kh x kw."""
    generator = torch.Generator().manual_seed(int(seed))
    a, b = {}, {}
    for name in targets:
        fan_out, fan_in = lora_matrix_shape(np.shape(params[name]))
        bound = 1.0 / float(np.sqrt(fan_in))
        a[name] = ((torch.rand(rank, fan_in, generator=generator, dtype=torch.float32) * 2 - 1) * bound
                   ).requires_grad_(True)
        b[name] = torch.zeros(fan_out, rank, dtype=torch.float32, requires_grad=True)
    return a, b


def _adam(leaves: list[torch.Tensor], lr: float) -> torch.optim.Adam:
    return torch.optim.Adam(leaves, lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=0.0, foreach=False)


def lora_checkpoints(arch: Architecture, params: Mapping[str, np.ndarray], x: torch.Tensor, y: torch.Tensor, *,
                     targets: Sequence[str], rank: int, lr: float, checkpoints: tuple[int, ...],
                     seed: int) -> list[tuple[int, dict[str, np.ndarray], float]]:
    """Full-batch Adam on A and B; return (steps, merged params, cross-entropy) at each checkpoint."""
    scale = LORA_ALPHA_OVER_RANK
    base = to_tensors(params)
    a, b = lora_init(params, targets, rank, seed)
    optimizer = _adam([t for name in targets for t in (a[name], b[name])], lr)

    def merged() -> dict[str, torch.Tensor]:
        tensors = dict(base)
        for name in targets:
            tensors[name] = base[name] + scale * (b[name] @ a[name]).reshape(base[name].shape)
        return tensors

    wanted, snapshots = set(checkpoints), []
    for step in range(1, max(checkpoints) + 1):
        optimizer.zero_grad(set_to_none=True)
        evidence_loss(arch, merged(), x, y).backward()
        optimizer.step()
        if step in wanted:
            with torch.no_grad():
                tensors = merged()
                loss = float(evidence_loss(arch, tensors, x, y))
                snapshots.append((step, {k: v.detach().numpy().copy() for k, v in tensors.items()}, loss))
    return snapshots


def regularized_checkpoints(arch: Architecture, params: Mapping[str, np.ndarray], x: torch.Tensor, y: torch.Tensor,
                            *, trainable: list[str], lr: float, checkpoints: tuple[int, ...], seed: int,
                            penalty: Callable[[Mapping[str, torch.Tensor], torch.Tensor], torch.Tensor],
                            ) -> list[tuple[int, dict[str, np.ndarray], dict[str, float]]]:
    """Full-batch Adam on mean cross-entropy over all rows plus ``penalty(tensors, logits)``; at each checkpoint the
    params and the objective's parts (``ce``, ``penalty``, ``total``) after the step."""
    torch.manual_seed(seed)
    tensors = to_tensors(params, trainable=trainable)
    optimizer = _adam([tensors[key] for key in trainable], lr)

    def objective() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = logits(arch, tensors, x)
        ce = F.cross_entropy(out, y)
        reg = penalty(tensors, out)
        return ce + reg, ce, reg

    wanted, snapshots = set(checkpoints), []
    for step in range(1, max(checkpoints) + 1):
        optimizer.zero_grad(set_to_none=True)
        objective()[0].backward()
        optimizer.step()
        if step in wanted:
            with torch.no_grad():
                total, ce, reg = objective()
            snapshots.append((step, {k: v.detach().numpy().copy() for k, v in tensors.items()},
                              {"ce": float(ce), "penalty": float(reg), "total": float(total)}))
    return snapshots


def lwf_penalty(base_logits_replay: torch.Tensor, k: int, weight: float
                ) -> Callable[[Mapping[str, torch.Tensor], torch.Tensor], torch.Tensor]:
    """w KL(p_base || p_theta) at temperature 1, averaged over the replay rows (the rows after the first ``k``)."""
    target = F.log_softmax(base_logits_replay / LWF_TEMPERATURE, dim=1)

    def penalty(_: Mapping[str, torch.Tensor], out: torch.Tensor) -> torch.Tensor:
        student = F.log_softmax(out[k:] / LWF_TEMPERATURE, dim=1)
        return weight * F.kl_div(student, target, log_target=True, reduction="batchmean")

    return penalty


def l2sp_penalty(params: Mapping[str, np.ndarray], trainable: list[str], weight: float
                 ) -> Callable[[Mapping[str, torch.Tensor], torch.Tensor], torch.Tensor]:
    """(w / 2) sum over every trainable entry of (theta - theta_base)^2."""
    anchor = {key: torch.from_numpy(np.array(params[key], dtype=np.float32, copy=True)) for key in trainable}

    def penalty(tensors: Mapping[str, torch.Tensor], _: torch.Tensor) -> torch.Tensor:
        return 0.5 * weight * sum(((tensors[key] - anchor[key]) ** 2).sum() for key in trainable)

    return penalty


def _composition(evidence: Evidence) -> str:
    return f"{len(evidence.y)} rows ({evidence.composition()})"


def f1_generator(instance: Instance, method: str, run_seed: int, edge: bool = False) -> Generator:
    """Candidate generator of one F1 method on one E11 instance (module docstring); ``edge``: the T04 edge extension."""
    configs = f1_configs(method, edge)

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        params = instance.params
        arch = architecture_from(params, read_model_spec(instance))
        if arch.family not in (VITLITE, *RESNET18_FAMILIES):
            raise F1Error(f"F1 runs on {VITLITE} and {RESNET18_FAMILIES} students, not {arch.family}")
        if (method == "ft_ln") != (arch.family == VITLITE) and method in ("ft_ln", "ft_bn"):
            raise F1Error(f"{method} does not apply to a {arch.family} student (ft_ln: LayerNorm, ft_bn: BatchNorm)")
        if (evidence.n_replay == 0) != (method == "ft_full_noreplay"):
            raise F1Error(f"{method} got {evidence.n_replay} replay rows; only ft_full_noreplay fits without them")
        parity = assert_parity(arch, params, instance.forward, evidence.x)
        x = images_tensor(evidence.x)
        y = torch.from_numpy(evidence.y.astype(np.int64))
        info: dict[str, Any] = {"architecture": arch.describe(), "parity_max_abs_logit_diff": parity,
                                "fit_rows": _composition(evidence), "prereg": PREREG_F1}
        with torch.no_grad():
            info["base_evidence_loss"] = float(evidence_loss(arch, to_tensors(params), x, y))
        if method == "ft_ln":
            trainable = list(ln_affine(arch))
        elif method == "ft_bn":
            trainable = [f"{prefix}.{part}" for prefix in arch.bn_layers for part in ("weight", "bias")]
        elif method in LORA_RANKS:
            trainable = lora_targets(arch, params)
            info["lora"] = {"rank": LORA_RANKS[method], "alpha": LORA_RANKS[method] * LORA_ALPHA_OVER_RANK,
                            "scale": LORA_ALPHA_OVER_RANK,
                            "modules": LORA_CONV_RULE if arch.family in RESNET18_FAMILIES else list(LORA_MODULES),
                            "targets": trainable,
                            "init": "A uniform(+-1/sqrt(fan_in)) from torch.Generator(seed), B zeros",
                            "seed": int(run_seed), "trained_entries": int(sum(
                                LORA_RANKS[method] * sum(lora_matrix_shape(np.shape(params[k]))) for k in trainable))}
        else:
            trainable = trainable_tensors(arch, params)
        info["trainable"] = trainable
        info["trainable_entries"] = int(sum(np.size(params[k]) for k in trainable))
        k = evidence.n_deployment
        base_replay = None
        if method == "ft_full_lwf":
            with torch.no_grad():
                base_replay = logits(arch, to_tensors(params), x[k:]).detach()
            info["lwf"] = {"temperature": LWF_TEMPERATURE, "rows": f"the {len(evidence.y) - k} clean replay rows",
                           "target": "the base model's softmax, fixed before training",
                           "loss": "mean CE over all rows + w KL(p_base || p_theta), batchmean over the replay rows"}
        if method == "ft_full_l2sp":
            info["l2sp"] = {"loss": "mean CE over all rows + (w/2) sum (theta - theta_base)^2 over every entry",
                            "anchor": "the base weights"}
        candidates: list[CandidatePatch] = []
        groups: dict[tuple[float | None, float], list[int]] = {}
        for config in configs:
            groups.setdefault((config.get("weight"), config["lr"]), []).append(config["steps"])
        for (weight, lr), steps in groups.items():
            if method in LORA_RANKS:
                runs = [(s, child, {"ce": loss}) for s, child, loss in lora_checkpoints(
                    arch, params, x, y, targets=trainable, rank=LORA_RANKS[method], lr=lr,
                    checkpoints=tuple(steps), seed=run_seed)]
            elif method in REGULARIZED:
                assert weight is not None
                penalty = (lwf_penalty(base_replay, k, weight) if method == "ft_full_lwf"
                           else l2sp_penalty(params, trainable, weight))
                runs = regularized_checkpoints(arch, params, x, y, trainable=trainable, lr=lr,
                                               checkpoints=tuple(steps), seed=run_seed, penalty=penalty)
            else:
                runs = [(s, child, {"ce": loss}) for s, child, loss in adam_checkpoints(
                    arch, params, x, y, trainable=trainable, lr=lr, checkpoints=tuple(steps), seed=run_seed)]
            for step, child, parts in runs:
                _check_frozen_entries(params, child, trainable, None)
                config = {"lr": lr, "steps": step} if weight is None else {"weight": weight, "lr": lr, "steps": step}
                candidates.append(CandidatePatch(
                    slot=len(candidates), child=child, config=config,
                    hypothesis=(f"{method}: Adam lr={lr:g} for {step} full-batch steps"
                                + ("" if weight is None else f", weight {weight:g}")
                                + f" on {_composition(evidence)}; {F1_SPECS[method].label}"),
                    predicted_effect=("fewer errors on held-out condition images if the fit generalizes; "
                                      + ("retention is not fitted (no replay rows)" if evidence.n_replay == 0 else
                                         f"retention is fitted on the {evidence.n_replay} clean replay rows")),
                    notes={"objective_at_checkpoint": parts}))
        if [c.config for c in candidates] != configs:
            raise RuntimeError(f"{method}: candidates {[c.config for c in candidates]} != configurations {configs}")
        return candidates, info

    return generate


def f1_config(method: str, run_seed: int, dataset: str = F1_DATASET, edge: bool = False) -> dict[str, Any]:
    """Run configuration recorded in run.json; the T04 edge extension notes itself (``grid``)."""
    resnet18 = dataset in (F1_E12_DATASET, F1_T04_DATASET)
    config = {
        "prereg": PREREG_F1,
        "candidate_order": f1_configs(method, edge),
        "optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0},
        "fit_rows": ("the 64 condition rows" if method == "ft_full_noreplay"
                     else "the 64 condition rows, then the 64 clean replay rows"),
        "batch": "full", "dtype": "float32", "seed": run_seed,
        "lora": ({"rank": LORA_RANKS[method], "alpha_over_rank": LORA_ALPHA_OVER_RANK,
                  "modules": LORA_CONV_RULE if resnet18 else list(LORA_MODULES)}
                 if method in LORA_RANKS else None),
        "lwf_temperature": LWF_TEMPERATURE if method == "ft_full_lwf" else None,
        "norm_limit": None,
        "dataset": dataset,
        "reused": {k: list(v) for k, v in F1_DATASET_REUSED.get(dataset, F1_REUSED).items()},
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(), "device": "cpu",
    }
    if edge:
        config["grid"] = {"name": "t04_edge", "note": T04_EDGE_NOTE}
        config["reused"] = {part: [method_, "t04g3"] for part, (method_, _) in F1_T04_REUSED.items()}
    return config


def run_f1(instance: Instance, method: str, run_dir: Path, *, run_seed: int = 0,
           allow_main: bool = False, edge: bool = False) -> dict[str, Any]:
    """Run one F1 method on one E11, E12 or T04 instance under the E10 limits' oracle and executor (without the size
    limits); each dataset has its own method set (``F1_DATASET_METHODS``). ``edge``: the T04 edge extension (T04
    instances only), every learning rate divided by 10."""
    if method not in F1_SPECS:
        raise F1Error(f"unknown F1 method {method!r}; expected one of {sorted(F1_SPECS)}")
    dataset = instance.private.get("dataset")
    if dataset not in F1_DATASET_METHODS:
        raise F1Error(f"F1 runs on {sorted(F1_DATASET_METHODS)} instances, got dataset {dataset!r}")
    if method not in F1_DATASET_METHODS[dataset]:
        raise F1Error(f"{method!r} is not an F1 method on {dataset}; expected one of {F1_DATASET_METHODS[dataset]}")
    if edge and dataset != F1_T04_DATASET:
        raise F1Error(f"the edge extension is PREREG T04's; {method} on {dataset} has none")
    if dataset == F1_DATASET:  # the E11 records keep their run.json shape
        config = f1_config(method, run_seed)
        config.pop("dataset")
        config.pop("reused")
    else:
        config = f1_config(method, run_seed, dataset, edge)
    spec = F1_SPECS[method]
    if edge:
        spec = dataclasses.replace(spec, label=f"{spec.label}; {T04_EDGE_NOTE}")
    return execute_run(
        instance=instance, spec=spec, run_dir=run_dir, run_seed=run_seed,
        config=config, generate=f1_generator(instance, method, run_seed, edge),
        allow_main=allow_main, joint=True, replay_in_fit=method != "ft_full_noreplay", e10=True,
    )


__all__ = ["F1Error", "F1_DATASET", "F1_DATASET_METHODS", "F1_E12_DATASET", "F1_E12_METHODS", "F1_E12_REUSED",
           "F1_LABEL", "F1_METHODS", "F1_REUSED", "F1_SPECS", "LORA_CONV_RULE", "f1_config", "f1_configs",
           "f1_generator", "l2sp_penalty", "lora_checkpoints", "lora_init", "lora_matrix_shape", "lora_targets",
           "lwf_penalty", "regularized_checkpoints", "run_f1"]
