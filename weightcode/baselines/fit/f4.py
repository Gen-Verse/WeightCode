"""PREREG F4, continual adaptation by fine-tuning over the E10 and E11 streams (WEIGHTCODE-main main 2026-11-05 15:25 text;
no LLM call). Four arms run round after round on their own current student through the control (b) stream
(``checked_reference_stream``): the rows, the multi-part check, the kept rule (loop-retain-v2), the readouts and the
summary are control (b)'s; only the candidate generators differ.

Each round has 8 configurations and one check per configuration. All arms use F1's recipe: full-batch Adam (0.9, 0.999,
1e-8), no weight decay, mean cross-entropy, eval-form forward, fp32, run seed 0, one trajectory per (weight, lr)
checkpointed at the grid's steps. None is held to the E10 program limits.

* ``f4_ft_replay``, sequential full fine-tuning with replay: the round's evidence rows then the clean replay rows.
  Every trainable tensor (BatchNorm running statistics excepted). lr {1e-5, 1e-4} x steps {10, 30, 100, 300}.
* ``f4_lora``, sequential LoRA merged at the end of each round: the same rows. A fresh adapter each round (B = 0, A
  uniform in +-1/sqrt(fan_in) from the run seed, alpha = r = 4). ViT-Lite: F1's 28 targets. ResNet-20: every
  convolution weight and the fc weight, each factored as rank 4 on its (out, in x kh x kw) reshape. lr {1e-4, 1e-3} x
  steps {10, 30, 100, 300}.
* ``f4_er``, experience replay of every earlier round's rows: the round's evidence rows, the earlier rounds' kept rows
  (one image one view), then the clean replay rows. Full fine-tuning on f4_ft_replay's grid.
* ``f4_lwf``, LwF to the previous round's student: f4_ft_replay's rows, plus w KL(p_prev || p_theta) on the clean replay
  rows at temperature 1, with p_prev the current student's softmax (the previous round's kept student, the base in round
  1). weight {1, 10} x lr {1e-5, 1e-4} x steps {30, 300}.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.baselines.instance_io import Instance

from . import checked_reference_stream as CRS
from .ftcompare import FT_LRS, LORA_ALPHA_OVER_RANK, LORA_LRS, LWF_WEIGHTS, REG_STEPS, lora_targets, lwf_penalty, regularized_checkpoints
from .grad import ADAM_BETAS, ADAM_EPS, STEP_CHECKPOINTS, _check_frozen_entries, adam_checkpoints, evidence_loss
from .runner import CandidatePatch, Evidence, Generator, MethodSpec, read_model_spec
from .torch_forward import (
    RESNET20A,
    VITLITE,
    Architecture,
    architecture_from,
    assert_parity,
    images_tensor,
    logits,
    to_tensors,
    trainable_tensors,
)

PREREG_F4 = "PREREG F4 (continual adaptation by fine-tuning, WEIGHTCODE-main main 2026-11-05 15:25 text)"
F4_REPLAY_ROWS = 64
"""The clean replay rows at the end of every round's fit rows (replay-v1)."""
LORA_RANK = 4

_TAIL = f"gradient-trained (not training-free), edit limits ignored; {PREREG_F4}"
F4_SPECS: dict[str, MethodSpec] = {
    "f4_ft_replay": MethodSpec(method="f4_ft_replay", uses_labels_for_fit=True, edit_limits_enforced=False,
                               label=f"F4 sequential full fine-tuning with replay (the round's rows + clean replay); {_TAIL}"),
    "f4_lora": MethodSpec(method="f4_lora", uses_labels_for_fit=True, edit_limits_enforced=False,
                          label=f"F4 sequential LoRA r4 merged each round (the round's rows + clean replay); {_TAIL}"),
    "f4_er": MethodSpec(method="f4_er", uses_labels_for_fit=True, edit_limits_enforced=False,
                        label=f"F4 sequential full fine-tuning with experience replay of every earlier round's rows; {_TAIL}"),
    "f4_lwf": MethodSpec(method="f4_lwf", uses_labels_for_fit=True, edit_limits_enforced=False,
                         label=f"F4 sequential full fine-tuning with LwF to the previous round's student; {_TAIL}"),
}
F4_METHODS: tuple[str, ...] = tuple(F4_SPECS)
CRS.EXTRA_SPECS.update(F4_SPECS)


class F4Error(ValueError):
    """An F4 arm cannot run as specified."""


def f4_configs(method: str) -> list[dict[str, Any]]:
    """The 8 configurations of ``method`` in check order: weight (LwF) outer, then lr, then steps."""
    if method in ("f4_ft_replay", "f4_er"):
        return [{"lr": lr, "steps": s} for lr in FT_LRS for s in STEP_CHECKPOINTS]
    if method == "f4_lora":
        return [{"lr": lr, "steps": s} for lr in LORA_LRS for s in STEP_CHECKPOINTS]
    if method == "f4_lwf":
        return [{"weight": w, "lr": lr, "steps": s} for w in LWF_WEIGHTS for lr in FT_LRS for s in REG_STEPS]
    raise F4Error(f"unknown F4 method {method!r}; expected one of {F4_METHODS}")


def lora_targets_for(arch: Architecture, params: Mapping[str, np.ndarray]) -> list[str]:
    """The tensors LoRA adapts: ViT-Lite's 28 matrices, or ResNet-20's convolution weights and the fc weight."""
    if arch.family == VITLITE:
        return lora_targets(arch, params)
    if arch.family == RESNET20A:
        names = [k for k, v in params.items() if k.endswith("weight") and np.ndim(v) == 4]
        names.append(arch.last_linear_tensors[0])
        return names
    raise F4Error(f"LoRA targets are defined for {VITLITE} and {RESNET20A}, not {arch.family}")


def lora_nd_checkpoints(arch: Architecture, params: Mapping[str, np.ndarray], x: torch.Tensor, y: torch.Tensor, *,
                        targets: Sequence[str], rank: int, lr: float, checkpoints: tuple[int, ...], seed: int
                        ) -> list[tuple[int, dict[str, np.ndarray], float]]:
    """Full-batch Adam on rank-``rank`` adapters of ``targets`` (any rank >= 2, factored on the (out, in x ...)
    reshape); returns (steps, merged params, cross-entropy). On 2-D targets it equals ``ftcompare.lora_checkpoints``
    (same draws, same updates)."""
    base = to_tensors(params)
    generator = torch.Generator().manual_seed(int(seed))
    a, b = {}, {}
    for name in targets:
        shape = tuple(np.shape(params[name]))
        fan_out, fan_in = shape[0], int(np.prod(shape[1:]))
        bound = 1.0 / float(np.sqrt(fan_in))
        a[name] = ((torch.rand(rank, fan_in, generator=generator, dtype=torch.float32) * 2 - 1) * bound
                   ).requires_grad_(True)
        b[name] = torch.zeros(fan_out, rank, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([t for n in targets for t in (a[n], b[n])], lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS,
                                 weight_decay=0.0, foreach=False)

    def merged() -> dict[str, torch.Tensor]:
        tensors = dict(base)
        for name in targets:
            tensors[name] = base[name] + LORA_ALPHA_OVER_RANK * (b[name] @ a[name]).reshape(base[name].shape)
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


def fit_rows(method: str, evidence: Evidence) -> tuple[np.ndarray, np.ndarray, int, str]:
    """The arm's fit rows from a round's control (b) evidence (the round's rows, the earlier rounds' kept rows, then the
    replay rows): returns x, y, the number of rows before the replay block, and a description."""
    n_new = evidence.n_deployment
    n_earlier = len(evidence.y) - n_new - F4_REPLAY_ROWS
    if n_earlier < 0:
        raise F4Error(f"{len(evidence.y)} rows cannot hold {n_new} round rows and {F4_REPLAY_ROWS} replay rows")
    y = evidence.y.astype(np.int64)
    if method == "f4_er":
        return evidence.x, y, n_new + n_earlier, (f"{n_new} round rows, {n_earlier} earlier rounds' rows, "
                                                   f"{F4_REPLAY_ROWS} clean replay rows")
    x = np.ascontiguousarray(np.concatenate([evidence.x[:n_new], evidence.x[-F4_REPLAY_ROWS:]]))
    return x, np.concatenate([y[:n_new], y[-F4_REPLAY_ROWS:]]), n_new, (
        f"{n_new} round rows, {F4_REPLAY_ROWS} clean replay rows ({n_earlier} earlier rows dropped)")


def f4_generator(method: str, instance: Instance, run_seed: int) -> Generator:
    """Candidate generator of one F4 arm on the current student ``instance.params`` (module docstring). Matches
    ``checked_reference_stream.GeneratorFactory``."""
    configs = f4_configs(method)

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        params = instance.params
        arch = architecture_from(params, read_model_spec(instance))
        if arch.family not in (VITLITE, RESNET20A):
            raise F4Error(f"F4 runs on {VITLITE} and {RESNET20A} students, not {arch.family}")
        x_np, y_np, k, rows = fit_rows(method, evidence)
        parity = assert_parity(arch, params, instance.forward, evidence.x)
        x, y = images_tensor(x_np), torch.from_numpy(y_np)
        info: dict[str, Any] = {"architecture": arch.describe(), "parity_max_abs_logit_diff": parity, "prereg": PREREG_F4,
                                "arm_fit_rows": rows}
        with torch.no_grad():
            info["base_evidence_loss"] = float(evidence_loss(arch, to_tensors(params), x, y))
        if method == "f4_lora":
            trainable = lora_targets_for(arch, params)
            info["lora"] = {"rank": LORA_RANK, "targets": trainable, "seed": int(run_seed)}
        else:
            trainable = trainable_tensors(arch, params)
        info["trainable"] = trainable
        base_replay = None
        if method == "f4_lwf":
            with torch.no_grad():
                base_replay = logits(arch, to_tensors(params), x[k:]).detach()
        groups: dict[tuple[float | None, float], list[int]] = {}
        for config in configs:
            groups.setdefault((config.get("weight"), config["lr"]), []).append(config["steps"])
        candidates: list[CandidatePatch] = []
        for (weight, lr), steps in groups.items():
            if method == "f4_lora":
                runs = [(s, c, {"ce": ce}) for s, c, ce in lora_nd_checkpoints(
                    arch, params, x, y, targets=trainable, rank=LORA_RANK, lr=lr, checkpoints=tuple(steps), seed=run_seed)]
            elif method == "f4_lwf":
                assert weight is not None and base_replay is not None
                runs = regularized_checkpoints(arch, params, x, y, trainable=trainable, lr=lr, checkpoints=tuple(steps),
                                               seed=run_seed, penalty=lwf_penalty(base_replay, k, weight))
            else:
                runs = [(s, c, {"ce": ce}) for s, c, ce in adam_checkpoints(
                    arch, params, x, y, trainable=trainable, lr=lr, checkpoints=tuple(steps), seed=run_seed)]
            for step, child, parts in runs:
                _check_frozen_entries(params, child, trainable, None)
                config = {"lr": lr, "steps": step} if weight is None else {"weight": weight, "lr": lr, "steps": step}
                candidates.append(CandidatePatch(
                    slot=len(candidates), child=child, config=config,
                    hypothesis=f"{method}: lr={lr:g}, {step} steps, "
                               + ("" if weight is None else f"weight {weight:g}, ") + f"on {rows}; {F4_SPECS[method].label}",
                    predicted_effect="fewer errors on the round's held-out condition images if the fit generalizes; "
                                     "earlier conditions and clean images are held by the replay rows",
                    notes={"objective_at_checkpoint": parts}))
        if [c.config for c in candidates] != configs:
            raise RuntimeError(f"{method}: candidates {[c.config for c in candidates]} != configurations {configs}")
        return candidates, info

    return generate


def make_generator(method: str, instance: Instance, run_seed: int) -> Generator:
    """``checked_reference_stream.GeneratorFactory`` for the F4 arms."""
    return f4_generator(method, instance, run_seed)


__all__ = ["F4Error", "F4_METHODS", "F4_SPECS", "f4_configs", "f4_generator", "fit_rows", "lora_nd_checkpoints",
           "lora_targets_for", "make_generator"]
