"""PREREG E14 references at equal information: behaviour cloning in E5's shapes, F1's eight fine-tuning methods, and
the label-equivalent runs, on a policyskill-v1 instance (``weightcode.evalkit.e14``).

Every method trains from the base weights with full-batch Adam (betas 0.9/0.999, eps 1e-8, no weight decay, run
seed 0) on the mean cross-entropy over the fit rows: the instance's new-demonstration steps (every step of its 8 bot
demonstrations) followed by its 64 replay steps, with the bot's action as the target (behaviour cloning, the loss the
base was trained with). One trajectory per (weight, lr), checkpointed at the listed steps; a checkpoint equals a
separate run of that length, since full-batch Adam from fixed weights has no randomness. LoRA's A matrices are the
only random draw (seeded by the run seed). The student is rebuilt in torch (``weightcode.models.e14_policy``) and checked
against forward.py on the fit rows before any step. Each method gives 8 candidates, one check each, scored by the
E14 rollout oracle (``weightcode.baselines.fit.runner.execute_run(e14=True)``).

Behaviour cloning in E5's shapes (``BC_METHODS``; lr {1e-3, 1e-2} x steps {10, 30, 100, 300}):

* ``bc_full``: every tensor, edit limits ignored (the unconstrained reference, as E5's seq_grad_full).
* ``bc_emb_head``: the instruction word embedding and the head (weight and bias).
* ``bc_new_rows``: the rows of the reserved words ("pick", "up") of the word embedding only (a gradient mask).
* ``bc_sparse19_sal`` and ``bc_sparse19_traj``: every tensor, a fixed top 19% of each tensor's entries
  (``floor(0.19 x size)``, E5's rule, ``weightcode.baselines.fit.grad.top_fraction_masks``) by |d loss / d w| at the base,
  or by |w - w_base| after unconstrained Adam at lr 1e-3 for 300 steps.

The four constrained shapes run under the E14 program limits (at most 20% of the entries changed, rel-Frob lifted),
which the executor checks on every patch; they are within them by construction, so nothing is scaled.

F1's eight (``F1_METHODS``; WEIGHTCODE-main decision (d), 2026-11-06), every one a fine-tuning method as practised and so
held to no size limit (each checked candidate still records whether it is within the E14 limits):

* ``f1_full`` and ``f1_last`` (the head): lr {1e-5, 1e-4} x steps {10, 30, 100, 300}.
* ``ft_ln``: the one LayerNorm's weight and bias, lr {1e-5, 1e-4} x steps {10, 30, 100, 300}.
* ``lora_r4``, ``lora_r16``: W = W0 + B A (alpha = r) on every conv and linear weight (instr, conv1, film1, conv2,
  film2, fc, head; a conv on its out x (in x 3 x 3) reshape, as F4), B = 0 and A as nn.Linear's initialisation from
  a generator seeded by the run seed; lr {1e-4, 1e-3} x steps {10, 30, 100, 300}; the merged weights are the
  candidate.
* ``ft_full_noreplay``: every tensor, the new steps only (the runner fits the deployment rows), lr {1e-5, 1e-4} x
  steps {10, 30, 100, 300}.
* ``ft_full_lwf``: every tensor, the fit rows plus w KL(p_base || p_theta) averaged over the 64 replay steps
  (temperature 1); w {1, 10} x lr {1e-5, 1e-4} x steps {30, 300}.
* ``ft_full_l2sp``: every tensor, the fit rows plus (w / 2) ||theta - theta_base||^2 over every entry; w {1e-2, 1e-1}
  x lr {1e-5, 1e-4} x steps {30, 300}.

Label-equivalent runs (``LABELEQ_KS``; decision (c)): ``bc_full``'s shape and grid on the first k usable new
V_evidence demonstrations (k = 16, 32, 64, 128, 256, nested; the instance's 8 are the first 8) plus the same 64
replay steps, mean loss over all rows. Only k changes. They are beside the bar, not in it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import params_sha256
from weightcode.data import e14 as D
from weightcode.models import e14_policy as P
from weightcode.models.parity import check_logit_parity

from .grad import ADAM_BETAS, ADAM_EPS, LEARNING_RATES, MASK_FRACTION, STEP_CHECKPOINTS, TRAJ_LR, TRAJ_STEPS
from .grad import top_fraction_masks
from .runner import CandidatePatch, Evidence, MethodSpec, execute_run, load_e14_evidence

PREREG_E14 = "PREREG E14 (main 9df26e9; as built and decisions main 8baaabd, d744142)"
E14_REF_LABEL = "e14"
"""Run-directory label of E14 references: ``baselines/<instance_id>/<method>/e14/<seed>/``."""
EMBEDDING = "word_emb.weight"
HEAD = ("head.weight", "head.bias")
LAYER_NORM = ("ln.weight", "ln.bias")
LORA_TARGETS = ("instr.weight", "conv1.weight", "film1.weight", "conv2.weight", "film2.weight", "fc.weight",
                "head.weight")
FT_LRS: tuple[float, ...] = (1e-5, 1e-4)
LORA_LRS: tuple[float, ...] = (1e-4, 1e-3)
REG_STEPS: tuple[int, ...] = (30, 300)
REGULARIZED: dict[str, tuple[float, ...]] = {"ft_full_lwf": (1.0, 10.0), "ft_full_l2sp": (1e-2, 1e-1)}
LORA_RANKS: dict[str, int] = {"lora_r4": 4, "lora_r16": 16}
LABELEQ_KS: tuple[int, ...] = (16, 32, 64, 128, 256)

_BC_TAIL = "behaviour cloning on the 8 demonstrations and the 64 replay steps; not training-free"
_F1_TAIL = f"edit limits ignored; not training-free; {PREREG_E14}, F1's eight"
SPECS: dict[str, MethodSpec] = {
    "bc_full": MethodSpec("bc_full", f"Adam on every tensor, edit limits ignored; {_BC_TAIL}", True, False),
    "bc_emb_head": MethodSpec("bc_emb_head", f"Adam on the word embedding and the head, E14 limits; {_BC_TAIL}", True,
                              True),
    "bc_new_rows": MethodSpec("bc_new_rows", f"Adam on the reserved word rows only, E14 limits; {_BC_TAIL}", True,
                              True),
    "bc_sparse19_sal": MethodSpec("bc_sparse19_sal", f"Adam under a fixed top-19% base-saliency mask per tensor, E14 "
                                                     f"limits; {_BC_TAIL}", True, True),
    "bc_sparse19_traj": MethodSpec("bc_sparse19_traj", f"Adam under a fixed top-19% trajectory mask per tensor, E14 "
                                                       f"limits; {_BC_TAIL}", True, True),
    "f1_full": MethodSpec("f1_full", f"F1 fine-tuning: Adam on every tensor; {_F1_TAIL}", True, False),
    "f1_last": MethodSpec("f1_last", f"F1 fine-tuning: Adam on the head only; {_F1_TAIL}", True, False),
    "ft_ln": MethodSpec("ft_ln", f"F1 fine-tuning: Adam on the LayerNorm weight and bias only; {_F1_TAIL}", True,
                        False),
    "lora_r4": MethodSpec("lora_r4", f"F1 fine-tuning: LoRA rank 4 on every conv and linear weight, merged; "
                                     f"{_F1_TAIL}", True, False),
    "lora_r16": MethodSpec("lora_r16", f"F1 fine-tuning: LoRA rank 16 on every conv and linear weight, merged; "
                                       f"{_F1_TAIL}", True, False),
    "ft_full_noreplay": MethodSpec("ft_full_noreplay", f"F1 fine-tuning: Adam on every tensor, the new steps only; "
                                                       f"{_F1_TAIL}", True, False),
    "ft_full_lwf": MethodSpec("ft_full_lwf", f"F1 fine-tuning: Adam on every tensor plus LwF on the replay steps; "
                                             f"{_F1_TAIL}", True, False),
    "ft_full_l2sp": MethodSpec("ft_full_l2sp", f"F1 fine-tuning: Adam on every tensor plus L2-SP toward the base; "
                                               f"{_F1_TAIL}", True, False),
}
for _k in LABELEQ_KS:
    SPECS[f"labeleq_k{_k}"] = MethodSpec(
        f"labeleq_k{_k}", f"label-equivalent run: bc_full's shape and grid on the first {_k} new demonstrations and the "
                          f"64 replay steps, edit limits ignored; beside the bar; not training-free", True, False)
BC_METHODS: tuple[str, ...] = ("bc_full", "bc_emb_head", "bc_new_rows", "bc_sparse19_sal", "bc_sparse19_traj")
F1_METHODS: tuple[str, ...] = ("f1_full", "f1_last", "ft_ln", "lora_r4", "lora_r16", "ft_full_noreplay",
                               "ft_full_lwf", "ft_full_l2sp")
LABELEQ_METHODS: tuple[str, ...] = tuple(f"labeleq_k{k}" for k in LABELEQ_KS)
E14_METHODS: tuple[str, ...] = BC_METHODS + F1_METHODS + LABELEQ_METHODS


class E14RefError(ValueError):
    """An E14 reference run was refused."""


def method_configs(method: str) -> list[dict[str, Any]]:
    """The 8 configurations of ``method`` in check order: weight (if any) outer, then lr, then steps."""
    if method in BC_METHODS or method in LABELEQ_METHODS:
        return [{"lr": lr, "steps": steps} for lr in LEARNING_RATES for steps in STEP_CHECKPOINTS]
    if method in ("f1_full", "f1_last", "ft_ln", "ft_full_noreplay"):
        return [{"lr": lr, "steps": steps} for lr in FT_LRS for steps in STEP_CHECKPOINTS]
    if method in LORA_RANKS:
        return [{"lr": lr, "steps": steps} for lr in LORA_LRS for steps in STEP_CHECKPOINTS]
    if method in REGULARIZED:
        return [{"weight": w, "lr": lr, "steps": steps} for w in REGULARIZED[method] for lr in FT_LRS
                for steps in REG_STEPS]
    raise E14RefError(f"unknown E14 method {method!r}; expected one of {E14_METHODS}")


def trainable_tensors(method: str, params: Mapping[str, np.ndarray]) -> list[str]:
    names = list(params)
    if method in ("bc_emb_head",):
        return [EMBEDDING, *HEAD]
    if method == "bc_new_rows":
        return [EMBEDDING]
    if method == "f1_last":
        return list(HEAD)
    if method == "ft_ln":
        return list(LAYER_NORM)
    if method in LORA_RANKS:
        return []
    return names


def rebuild(params: Mapping[str, np.ndarray]) -> P.FilmPolicy:
    return P.build_policy(P.config_from_params(params), params, torch.float32)


def parity(model: P.FilmPolicy, params: Mapping[str, np.ndarray], forward: Any, x: np.ndarray) -> dict[str, Any]:
    """forward.py against the float32 and float64 torch rebuilds on ``x`` (``check_logit_parity``, which raises)."""
    ours = np.asarray(forward(dict(params), x), dtype=np.float64)
    ref64 = P.eval_logits(P.build_policy(model.config, params, torch.float64), x)
    ref32 = P.eval_logits(model, x)
    return check_logit_parity(ours, ref64, ref32)


def lora_shape(shape: Sequence[int]) -> tuple[int, int]:
    """(fan_out, fan_in) of a LoRA target: a linear weight as it is, a conv [out, in, 3, 3] as (out, in x 9)."""
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 4:
        return int(shape[0]), int(shape[1] * shape[2] * shape[3])
    raise E14RefError(f"a LoRA target must be 2-D or 4-D, got {list(shape)}")


def _trajectory(params: Mapping[str, np.ndarray], x: torch.Tensor, y: torch.Tensor, *, method: str, lr: float,
                checkpoints: Sequence[int], seed: int, masks: Mapping[str, np.ndarray] | None = None,
                weight: float | None = None, n_fit_new: int | None = None,
                ) -> list[tuple[int, dict[str, np.ndarray], float]]:
    """Full-batch Adam on the method's loss; (steps, merged params, loss) at each checkpoint."""
    torch.manual_seed(seed)
    model = rebuild(params)
    for p in model.parameters():
        p.requires_grad_(False)
    base = {k: torch.from_numpy(np.ascontiguousarray(v)).to(torch.float32) for k, v in params.items()}
    leaves: dict[str, torch.Tensor] = {}
    lora: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    if method in LORA_RANKS:
        rank = LORA_RANKS[method]
        generator = torch.Generator().manual_seed(int(seed))
        for name in LORA_TARGETS:
            fan_out, fan_in = lora_shape(params[name].shape)
            bound = 1.0 / float(np.sqrt(fan_in))
            a = ((torch.rand(rank, fan_in, generator=generator, dtype=torch.float32) * 2 - 1) * bound)
            lora[name] = (a.requires_grad_(True), torch.zeros(fan_out, rank, dtype=torch.float32, requires_grad=True))
        optimised = [t for pair in lora.values() for t in pair]
    else:
        for name in trainable_tensors(method, params):
            leaves[name] = base[name].clone().requires_grad_(True)
        optimised = list(leaves.values())
    grad_masks = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in (masks or {}).items()}

    def merged() -> dict[str, torch.Tensor]:
        out = dict(base)
        out.update(leaves)
        for name, (a, b) in lora.items():
            out[name] = base[name] + (b @ a).reshape(base[name].shape)
        return out

    replay = slice(n_fit_new, None) if n_fit_new is not None else None
    with torch.no_grad():
        base_log_p = F.log_softmax(model(x[replay]), dim=1) if method == "ft_full_lwf" else None

    def loss_of(theta: Mapping[str, torch.Tensor]) -> torch.Tensor:
        logits = functional_call(model, dict(theta), (x,))
        loss = F.cross_entropy(logits, y)
        if method == "ft_full_lwf":
            log_q = F.log_softmax(logits[replay], dim=1)
            loss = loss + float(weight) * F.kl_div(log_q, base_log_p, log_target=True, reduction="batchmean")
        elif method == "ft_full_l2sp":
            loss = loss + 0.5 * float(weight) * sum(((theta[k] - base[k]) ** 2).sum() for k in leaves)
        return loss

    optimizer = torch.optim.Adam(optimised, lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS, weight_decay=0.0, foreach=False)
    wanted = set(checkpoints)
    out: list[tuple[int, dict[str, np.ndarray], float]] = []
    for step in range(1, max(checkpoints) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_of(merged()).backward()
        for key, mask in grad_masks.items():
            grad = leaves[key].grad
            assert grad is not None
            grad.mul_(mask)
        optimizer.step()
        if step in wanted:
            with torch.no_grad():
                theta = merged()
                value = float(loss_of(theta))
                out.append((step, {k: v.detach().numpy().astype(np.float32).copy() for k, v in theta.items()}, value))
    return out


def _saliency(params: Mapping[str, np.ndarray], x: torch.Tensor, y: torch.Tensor) -> dict[str, np.ndarray]:
    model = rebuild(params)
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    return {name: p.grad.detach().numpy().copy() for name, p in model.named_parameters()}


def _check_frozen(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], trainable: Sequence[str],
                  masks: Mapping[str, np.ndarray] | None) -> None:
    for key, parent in base.items():
        if key not in trainable and not np.array_equal(child[key], parent):
            raise RuntimeError(f"{key} is not trainable but changed")
        if masks is not None and key in masks and not np.array_equal(child[key][~masks[key]], parent[~masks[key]]):
            raise RuntimeError(f"{key} changed outside its mask")


def labeleq_evidence(instance: Instance, k: int) -> Evidence:
    """The first ``k`` usable new V_evidence demonstrations (every step) followed by the instance's 64 replay steps;
    the first 8 are the instance's own (checked)."""
    from weightcode.evalkit.e14 import K_NEW, policy_logits

    if k not in LABELEQ_KS:
        raise E14RefError(f"k must be one of {LABELEQ_KS}")
    own = load_e14_evidence(instance)
    demos = D.load_demos("V_evidence_new", str(instance.private["vevidence_demos_sha256"]))
    episodes = D.usable_episodes(demos, k)
    if list(episodes[:K_NEW]) != list(instance.private["evidence_episodes"]):
        raise E14RefError("the first 8 label-equivalent demonstrations are not the instance's")
    rows = np.flatnonzero(np.isin(demos["episode"], episodes))
    x_new = demos["x"][rows].astype(np.int64)
    y_new = demos["y"][rows].astype(np.int64)
    if not np.array_equal(x_new[: own.n_rand], own.x[: own.n_rand]):
        raise E14RefError("the label-equivalent rows do not start with the instance's new rows")
    x = np.concatenate([x_new, own.x[own.n_rand:]])
    y = np.concatenate([y_new, own.y[own.n_rand:]])
    logits = policy_logits(instance.forward, instance.params, x)
    return Evidence(x=x, y=y, n_fail=0, n_rand=int(len(x_new)),
                    sha256=params_sha256({"x_new": x_new, "y_new": y_new}), base_pred=logits.argmax(1).astype(np.int64),
                    rule=f"{own.rule}+labeleq_k{k}", n_replay=own.n_replay, replay_sha256=own.replay_sha256,
                    replay_rule=own.replay_rule)


def generator(instance: Instance, method: str, run_seed: int):  # noqa: ANN201 - runner.Generator
    if method not in SPECS:
        raise E14RefError(f"unknown E14 method {method!r}")

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        params = instance.params
        model = rebuild(params)
        info: dict[str, Any] = {"parity": parity(model, params, instance.forward, evidence.x),
                                "n_fit_rows": int(len(evidence.y)), "n_fit_new": int(evidence.n_rand),
                                "n_fit_replay": int(evidence.n_replay)}
        x = torch.from_numpy(np.ascontiguousarray(evidence.x, dtype=np.int64))
        y = torch.from_numpy(np.ascontiguousarray(evidence.y, dtype=np.int64))
        with torch.no_grad():
            info["base_fit_loss"] = float(F.cross_entropy(model(x), y))
        trainable = trainable_tensors(method, params)
        masks: dict[str, np.ndarray] | None = None
        if method == "bc_new_rows":
            mask = np.zeros(params[EMBEDDING].shape, dtype=bool)
            mask[list(D.RESERVED_IDS)] = True
            masks = {EMBEDDING: mask}
        elif method == "bc_sparse19_sal":
            masks = top_fraction_masks(_saliency(params, x, y), trainable, MASK_FRACTION)
        elif method == "bc_sparse19_traj":
            ((_, full, _),) = _trajectory(params, x, y, method="bc_full", lr=TRAJ_LR, checkpoints=(TRAJ_STEPS,),
                                          seed=run_seed)
            masks = top_fraction_masks({k: full[k] - params[k] for k in params}, trainable, MASK_FRACTION)
        if masks is not None:
            info["mask_entries"] = {k: int(v.sum()) for k, v in masks.items()}
        candidates: list[CandidatePatch] = []
        configs = method_configs(method)
        weights = sorted({c.get("weight") for c in configs}, key=lambda w: -1 if w is None else w)
        for weight in weights:
            for lr in sorted({c["lr"] for c in configs if c.get("weight") == weight}):
                steps = [c["steps"] for c in configs if c.get("weight") == weight and c["lr"] == lr]
                for step, child, loss in _trajectory(params, x, y, method=method, lr=lr, checkpoints=steps,
                                                     seed=run_seed, masks=masks, weight=weight,
                                                     n_fit_new=evidence.n_rand):
                    if method not in LORA_RANKS:
                        _check_frozen(params, child, trainable, masks)
                    config = {"lr": lr, "steps": step} if weight is None else {"weight": weight, "lr": lr,
                                                                               "steps": step}
                    candidates.append(CandidatePatch(
                        slot=len(candidates), child=child, config=config,
                        hypothesis=f"{method}: {config}, full-batch Adam on {len(evidence.y)} rows "
                                   f"({evidence.n_rand} new steps, {evidence.n_replay} replay steps)",
                        predicted_effect="the new mission done where the fit generalizes; the old mission kept where "
                                         "the replay steps hold it",
                        notes={"fit_loss_at_checkpoint": loss}))
        if [c.config for c in candidates] != configs:
            raise RuntimeError(f"{method}: candidate order {[c.config for c in candidates]} != {configs}")
        return candidates, info

    return generate


def run_config(method: str, run_seed: int) -> dict[str, Any]:
    return {
        "prereg": PREREG_E14,
        "candidate_order": method_configs(method),
        "optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0,
                      "batch": "full"},
        "loss": "mean cross-entropy of the bot's action over every fit row"
                + {"ft_full_lwf": " plus w KL(p_base || p_theta) averaged over the replay steps",
                   "ft_full_l2sp": " plus (w / 2) ||theta - theta_base||^2"}.get(method, ""),
        "lora": {"rank": LORA_RANKS[method], "alpha_over_rank": 1.0, "targets": list(LORA_TARGETS)}
        if method in LORA_RANKS else None,
        "mask_fraction_per_tensor": MASK_FRACTION if "sparse19" in method else None,
        "theta_full": {"lr": TRAJ_LR, "steps": TRAJ_STEPS} if method == "bc_sparse19_traj" else None,
        "labeleq_k": int(method.removeprefix("labeleq_k")) if method in LABELEQ_METHODS else None,
        "seed": run_seed,
        "dtype": "float32",
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "blas_threads_env": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                     "MKL_NUM_THREADS")},
        "device": "cpu",
    }


def run_e14_ref(instance: Instance, method: str, run_dir: Path, *, run_seed: int = 0) -> dict[str, Any]:
    """Run one E14 reference (``SPECS``) on one policyskill-v1 instance; returns its run record (also run.json)."""
    evidence = None
    if method in LABELEQ_METHODS:
        evidence = labeleq_evidence(instance, int(method.removeprefix("labeleq_k")))
    return execute_run(
        instance=instance,
        spec=SPECS[method],
        run_dir=run_dir,
        run_seed=run_seed,
        config=run_config(method, run_seed),
        generate=generator(instance, method, run_seed),
        require_valid_candidate=True,
        replay_in_fit=method != "ft_full_noreplay",
        e14=True,
        evidence_override=evidence,
    )


__all__ = ["BC_METHODS", "E14_METHODS", "E14_REF_LABEL", "E14RefError", "F1_METHODS", "LABELEQ_KS",
           "LABELEQ_METHODS", "LORA_TARGETS", "SPECS", "generator", "labeleq_evidence", "method_configs",
           "run_config", "run_e14_ref", "trainable_tensors"]
