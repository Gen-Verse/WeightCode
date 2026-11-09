"""Gradient references of PREREG E5 (rule seqskill-v1): Adam on the token cross-entropy of the 128 evidence rows.

The evidence is the instance's 64 new-operator rows followed by its 64 replay rows of the old operators, the rows a
LEARN session gets (``weightcode.baselines.fit.runner.load_seqskill_evidence``). The loss is the mean cross-entropy over
every labelled (OUT) position of the 128 rows, the loss the base student was trained with
(``weightcode.models.train_seq``). The student is rebuilt in torch (``weightcode.models.transformer``) and checked against
forward.py on the evidence rows before any step. The grid is that of ``weightcode.baselines.fit.grad``: lr in
{1e-3, 1e-2} x steps in {10, 30, 100, 300}, full batch, one trajectory per lr checkpointed at each step count,
8 candidates checked once each. Full-batch Adam from the base weights has no randomness.

* ``seq_grad_full``: every parameter, edit limits ignored: the unconstrained reference of the E5 bar. Its patch is one
  op per tensor, so the executor lets its op count reach the student's tensor count (70 for the d 64 tfm, above the
  format's 64; ``Limits.without_edit_limits``); run.json records these executor limits. The E5 program limits raise
  max_ops the same way (``seqskill_program_limits(..., n_tensors)``, as the harness's program arms do).
* ``seq_grad_emb_head``: the token embedding and the head (weight and bias) only.
* ``seq_grad_new_row``: the new operator's token-embedding row only (a gradient mask on ``tok_emb.weight``).

On ``seq_grad_emb_head`` and ``seq_grad_new_row`` the per-tensor norm limit of the E5 program limits (relative
Frobenius norm 0.5 on tensors of rank 2 or more) is enforced on each checkpoint by scaling an over-limit tensor's delta
down to the limit (``scale_to_norm_limit``), and every scaling is recorded. Their changed entries stay under the 20%
cap by construction; the executor checks both limits on every patch. The trajectory itself is unconstrained.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.baselines.instance_io import Instance
from weightcode.data import seqops as S
from weightcode.data.seqops_build import FORWARD_N_HEADS
from weightcode.evalkit.patch_check import ForwardFn
from weightcode.evalkit.seqskill import seqskill_program_limits
from weightcode.models import transformer as T
from weightcode.models.seq_export import PARITY_TOL, float32_tolerance

from .edits import scale_to_norm_limit
from .grad import ADAM_BETAS, ADAM_EPS, LEARNING_RATES, STEP_CHECKPOINTS, candidate_grid
from .runner import CandidatePatch, Evidence, Generator, MethodSpec, execute_run

PARITY_RTOL = PARITY_TOL
"""Largest allowed logit difference between the float32 torch rebuild and forward.py, relative to the logit scale
max(1, max |logit|) (``weightcode.models.seq_export.PARITY_RULE``, as for a student's export)."""
N_HEADS = FORWARD_N_HEADS
"""Heads of every seqops student, fixed by the dataset forward; parameter shapes do not show it."""
EMBEDDING = "tok_emb.weight"
HEAD_TENSORS: tuple[str, str] = ("head.weight", "head.bias")

SEQ_GRAD_SPECS: dict[str, MethodSpec] = {
    "seq_grad_full": MethodSpec(
        method="seq_grad_full",
        label="unconstrained gradient reference: Adam on all parameters, edit limits ignored; not training-free",
        uses_labels_for_fit=True,
        edit_limits_enforced=False,
    ),
    "seq_grad_emb_head": MethodSpec(
        method="seq_grad_emb_head",
        label="gradient reference: Adam on the token embedding and the head, E5 program limits enforced; not "
        "training-free",
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
    "seq_grad_new_row": MethodSpec(
        method="seq_grad_new_row",
        label="gradient reference: Adam on the new operator's embedding row only, E5 program limits enforced; not "
        "training-free",
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
}
SEQ_GRAD_METHODS: tuple[str, ...] = tuple(SEQ_GRAD_SPECS)
SEQSKILL_REF_LABEL = "e5"
"""Run-directory label of E5 references: ``baselines/<instance_id>/<method>/e5/<seed>/``."""


def rebuild(params: Mapping[str, np.ndarray]) -> T.SeqTransformer:
    """The student as a torch module on the CPU, float32, with ``params`` loaded (strict key match)."""
    model = T.build_seq_model(T.config_from_params(params, N_HEADS, S.PAD))
    T.load_seq_params(model, params)
    return model.eval()


def seq_parity(model: T.SeqTransformer, params: Mapping[str, np.ndarray], forward: ForwardFn,
               x: np.ndarray) -> tuple[float, float]:
    """Max absolute logit difference between the torch rebuild and forward.py on ``x``, and its bound.

    The bound is PARITY_RTOL times max(1, max |logit|) of forward.py's logits; a larger difference raises.
    """
    with torch.no_grad():
        mine = model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.int64))).numpy()
    theirs = np.asarray(forward(dict(params), x), dtype=np.float64)
    if mine.shape != theirs.shape:
        raise RuntimeError(f"torch logits {mine.shape} != forward.py logits {theirs.shape}")
    diff = float(np.max(np.abs(mine.astype(np.float64) - theirs)))
    bound = float32_tolerance(theirs)
    if not diff <= bound:
        raise RuntimeError(f"torch rebuild differs from forward.py by {diff:.3g} > {bound:.3g} "
                           f"(PARITY_RTOL {PARITY_RTOL:g} x max(1, max |logit|))")
    return diff, bound


def token_loss(model: T.SeqTransformer, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the labelled positions of the rows, as in base training."""
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=S.IGNORE)


def trainable_and_masks(method: str, params: Mapping[str, np.ndarray],
                        new_operator_id: int) -> tuple[list[str], dict[str, np.ndarray] | None]:
    """The tensors a method trains and, for ``seq_grad_new_row``, the entries it may change."""
    if method == "seq_grad_full":
        return sorted(params), None
    if method == "seq_grad_emb_head":
        return sorted([EMBEDDING, *HEAD_TENSORS]), None
    if method == "seq_grad_new_row":
        mask = np.zeros(params[EMBEDDING].shape, dtype=bool)
        mask[int(new_operator_id)] = True
        return [EMBEDDING], {EMBEDDING: mask}
    raise ValueError(f"unknown method {method!r}; expected one of {SEQ_GRAD_METHODS}")


def adam_checkpoints(
    params: Mapping[str, np.ndarray],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    trainable: Sequence[str],
    lr: float,
    checkpoints: tuple[int, ...],
    seed: int,
    masks: Mapping[str, np.ndarray] | None = None,
    device: str = "cpu",
) -> list[tuple[int, dict[str, np.ndarray], float]]:
    """Full-batch Adam on ``token_loss``; return (steps, params, loss) at each checkpoint.

    Only ``trainable`` tensors receive gradients. With ``masks``, each gradient is multiplied by its mask before the
    step, so masked-out entries keep their base values exactly (zero gradient and zero Adam moments give no update).
    ``device`` places the model, the rows and the masks (default the CPU, where every move is a no-op); the returned
    params are CPU numpy arrays either way.
    """
    torch.manual_seed(seed)
    model = rebuild(params).to(device)
    x, y = x.to(device), y.to(device)
    named = dict(model.named_parameters())
    unknown = sorted(set(trainable) - set(named))
    if unknown:
        raise ValueError(f"trainable tensors {unknown} are not parameters of the student")
    for name, tensor in named.items():
        tensor.requires_grad_(name in trainable)
    gradient_masks = {key: torch.from_numpy(value.astype(np.float32)).to(device)
                      for key, value in (masks or {}).items()}
    optimizer = torch.optim.Adam([named[key] for key in trainable], lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS,
                                 weight_decay=0.0, foreach=False)
    wanted = set(checkpoints)
    snapshots: list[tuple[int, dict[str, np.ndarray], float]] = []
    for step in range(1, max(checkpoints) + 1):
        optimizer.zero_grad(set_to_none=True)
        token_loss(model, x, y).backward()
        for key, mask in gradient_masks.items():
            grad = named[key].grad
            assert grad is not None
            grad.mul_(mask)
        optimizer.step()
        if step in wanted:
            with torch.no_grad():
                loss = float(token_loss(model, x, y))
            snapshots.append((step, T.export_seq_params(model), loss))
    return snapshots


def _check_frozen_entries(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], trainable: Sequence[str],
                          masks: Mapping[str, np.ndarray] | None) -> None:
    for key, parent in base.items():
        if key not in trainable:
            if not np.array_equal(child[key], parent):
                raise RuntimeError(f"{key} is not trainable but changed during training")
        elif masks is not None and key in masks and not np.array_equal(child[key][~masks[key]], parent[~masks[key]]):
            raise RuntimeError(f"{key} changed outside its mask")


def _hypothesis(method: str, lr: float, steps: int, evidence: Evidence, new_token: str) -> str:
    scope = {
        "seq_grad_full": "all tensors, no edit limits",
        "seq_grad_emb_head": f"the token embedding and the head ({', '.join(HEAD_TENSORS)})",
        "seq_grad_new_row": f"the {new_token} row of the token embedding only",
    }[method]
    return (
        f"{method}: Adam lr={lr:g} for {steps} full-batch steps on the mean token cross-entropy over the "
        f"{len(evidence.y)} evidence rows ({evidence.composition()}); trained: {scope}"
    )


def _predicted_effect(evidence: Evidence, new_token: str) -> str:
    return (
        f"rows of {new_token} answered right if the fit on {evidence.n_rand} new-operator rows generalizes; "
        f"retention is fitted on the {evidence.n_replay} replay rows of the old operators"
    )


def seq_grad_generator(instance: Instance, method: str, run_seed: int) -> Generator:
    """Candidate generator of one E5 gradient reference on one seqskill-v1 instance."""
    if method not in SEQ_GRAD_SPECS:
        raise ValueError(f"unknown method {method!r}; expected one of {SEQ_GRAD_METHODS}")
    spec = SEQ_GRAD_SPECS[method]
    new_id = int(instance.private["new_operator_id"])
    new_token = str(instance.private["new_operator"])
    limits = seqskill_program_limits(instance.task["limits"], len(instance.params))

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        params = instance.params
        model = rebuild(params)
        parity, parity_bound = seq_parity(model, params, instance.forward, evidence.x)
        x = torch.from_numpy(np.ascontiguousarray(evidence.x, dtype=np.int64))
        y = torch.from_numpy(np.ascontiguousarray(evidence.y, dtype=np.int64))
        trainable, masks = trainable_and_masks(method, params, new_id)
        info: dict[str, Any] = {
            "model_config": model.config.as_dict(),
            "parity_max_abs_logit_diff": parity,
            "parity_bound": parity_bound,
            "trainable": trainable,
            "n_trainable_entries": int(sum(params[k].size if masks is None or k not in masks else masks[k].sum()
                                           for k in trainable)),
            "n_params": int(sum(value.size for value in params.values())),
        }
        with torch.no_grad():
            info["base_evidence_loss"] = float(token_loss(model, x, y))
        candidates: list[CandidatePatch] = []
        for lr in LEARNING_RATES:
            for steps, child, loss in adam_checkpoints(params, x, y, trainable=trainable, lr=lr,
                                                       checkpoints=STEP_CHECKPOINTS, seed=run_seed, masks=masks):
                _check_frozen_entries(params, child, trainable, masks)
                notes: dict[str, Any] = {"evidence_loss_at_checkpoint": loss}
                if spec.edit_limits_enforced:
                    child, scaling = scale_to_norm_limit(params, child, trainable, limit=limits.max_rel_frob,
                                                         min_ndim=limits.rel_frob_min_ndim,
                                                         exempt=limits.rel_frob_exempt)
                    scaled = [key for key, entry in scaling.items() if entry["scaled"]]
                    notes["norm_scaling"] = scaling
                    notes["norm_scaled_tensors"] = scaled
                    if scaled:
                        with torch.no_grad():
                            notes["evidence_loss_after_scaling"] = float(token_loss(rebuild(child), x, y))
                candidates.append(CandidatePatch(
                    slot=len(candidates),
                    child=child,
                    config={"lr": lr, "steps": steps},
                    hypothesis=_hypothesis(method, lr, steps, evidence, new_token),
                    predicted_effect=_predicted_effect(evidence, new_token),
                    notes=notes,
                ))
        info["n_candidates_norm_scaled"] = sum(1 for c in candidates if c.notes.get("norm_scaled_tensors"))
        return candidates, info

    return generate


def seq_grad_config(method: str, run_seed: int, instance: Instance) -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    limits = seqskill_program_limits(instance.task["limits"], len(instance.params))
    return {
        "learning_rates": list(LEARNING_RATES),
        "step_checkpoints": list(STEP_CHECKPOINTS),
        "candidate_order": [{"lr": lr, "steps": steps} for lr, steps in candidate_grid()],
        "optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0},
        "loss": "mean cross-entropy over the labelled (OUT) positions of all evidence rows (the new-operator rows, "
        "then the replay rows)",
        "batch": "full",
        "dtype": "float32",
        "seed": run_seed,
        "trained": {"seq_grad_full": "all tensors", "seq_grad_emb_head": [EMBEDDING, *HEAD_TENSORS],
                    "seq_grad_new_row": f"{EMBEDDING} row {int(instance.private['new_operator_id'])}"}[method],
        "norm_limit": limits.max_rel_frob if SEQ_GRAD_SPECS[method].edit_limits_enforced else None,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
    }


def run_seq_grad(instance: Instance, method: str, run_dir: Path, *, run_seed: int = 0) -> dict[str, Any]:
    """Run one E5 gradient reference on one seqskill-v1 instance; returns its run record (also run.json).

    Dev instances only: seqskill instances are built as dev, and ``execute_run`` refuses any other role. A run whose
    candidates the executor rejects all (a student so small that the trained tensors exceed the changed-entry cap)
    fails with ``NoValidCandidate`` instead of ending as the base no-op: it has measured nothing.
    """
    return execute_run(
        instance=instance,
        spec=SEQ_GRAD_SPECS[method],
        run_dir=run_dir,
        run_seed=run_seed,
        config=seq_grad_config(method, run_seed, instance),
        generate=seq_grad_generator(instance, method, run_seed),
        require_valid_candidate=True,
        seqskill=True,
    )


__all__ = [
    "N_HEADS", "PARITY_RTOL", "SEQSKILL_REF_LABEL", "SEQ_GRAD_METHODS", "SEQ_GRAD_SPECS", "adam_checkpoints",
    "rebuild", "run_seq_grad", "seq_grad_config", "seq_grad_generator", "seq_parity", "token_loss",
    "trainable_and_masks",
]
