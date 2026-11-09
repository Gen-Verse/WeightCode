"""References of PREREG E7 (rule graphnew-v1): label fits of a graph network on the 144 evidence nodes.

The evidence is the instance's 80 new-class nodes (10 per new class) followed by its 64 replay nodes of the train
classes (``weightcode.baselines.fit.runner.load_graphnew_evidence``), the nodes a LEARN session gets. Every reference runs
through ``execute_run(..., graphnew=True)`` with run seed 0: 8 candidates, each checked once by the graphnew joint
oracle (V_check cut by class, S = S_new + S_old), and the run keeps ``oracle.final()``.

The student is the GCN of forward.py (``weightcode.models.gcn``): conv1 .. convK on the normalized adjacency A of the
instance's graph.npz, BatchNorm in eval form and ReLU after every layer but the last. It is rebuilt here as a
functional torch forward over the parameter tensors, in eval mode: no dropout, and the BatchNorm running statistics
are fixed inputs, never trained and never updated. The first layer is computed as (A X) W1^T + b1 with A X computed
once, as forward.py does. Before any fit the rebuild is checked against forward.py at every node of the graph
(``check_logit_parity``: float64 within one float32 step of each logit, float32 within PARITY_TOL of the logit scale).
Fits run in float32 with a full-graph forward per step; the loss is the mean cross-entropy over the 144 evidence
nodes.

Methods (fixed in PREREG E7 before any reference ran):

* ``grad_full``: Adam on every trainable tensor (conv weights and biases, BatchNorm weight and bias), edit limits
  ignored: the unconstrained reference. The grid is ``weightcode.baselines.fit.grad``'s, lr {1e-3, 1e-2} x steps
  {10, 30, 100, 300}, one trajectory per lr checkpointed at each step count.
* ``grad_last``: the same grid on the head (the last graph_conv layer, weight and bias) only. With the earlier layers
  fixed, the head input z = (A H_{K-1})[evidence] is a constant, and logits = z W^T + b equal forward.py's
  A (H_{K-1} W^T) + b, so the fit runs on z.
* ``grad_sparse19_sal`` and ``grad_sparse19_traj``: E5's two masks (``scripts/dev/e5_diag_sparse.py``). Every
  trainable tensor, but only a fixed top 19% of each tensor's entries (``floor(0.19 x size)``, stable argsort of
  -|score|, ties by flat index): by |d loss / d w| at the base weights, or by |w - w_base| after unconstrained Adam at
  lr 1e-3 for 300 steps. Gradients are multiplied by the mask before each step.
* ``interp``: a WiSE-FT-style mix base + alpha (theta_full - base) for alpha in {1/8, ..., 8/8}, theta_full the
  unconstrained Adam point at lr 1e-3 and 300 steps (``grad_full``'s q3), fixed in advance with no check spent on it.
  Edit limits ignored, like grad_full.
* ``ridge_margin``, ``imprint`` and ``imprint_retain``: E2's closed-form rules (``weightcode.baselines.fit.ridge`` and
  ``weightcode.baselines.fit.imprint``) on the head input z = (A H_{K-1})[ids] in float64, the exact input of the head.
  ridge_margin refits the whole head toward the base; imprint sets only the new-class rows and biases of the head
  (2 directions x 4 scales, one balanced shared offset); imprint_retain keeps imprint's candidates and sets the offset
  so that no replay node the base classifies correctly breaks (checked with forward.py). imprint_retain competes in
  the equal-retention supplement only, never in the gate.

Limits: ``graphnew_program_limits`` (head exempt from max_rel_frob, rel-Frob 0.5 on every other tensor of rank 2 or
more, at most 20% of all entries changed, BatchNorm running statistics included in the count). The methods with
enforced limits scale each candidate's over-limit tensors down to the norm limit (``scale_to_norm_limit``, the head
exempt) and record every scaling; their changed entries stay under the cap by construction (the head is 9.2% of the
d 256 student's entries, a 19% mask 18.8%). grad_full and interp are recorded as out of limits.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import Params, params_sha256
from weightcode.data.arxiv_classes import file_sha256
from weightcode.evalkit.graphnew import GRAPH_FILE, GRAPHNEW_RULE, graphnew_program_limits, head_tensors, predict_nodes
from weightcode.evalkit.patch_check import ForwardFn
from weightcode.models.gcn import BN_PARTS, _SymmetricSpMM, check_symmetric, graph_tensors
from weightcode.models.parity import FLOAT32_STEP, check_logit_parity
from weightcode.patch import Limits

from .edits import scale_to_norm_limit
from .grad import ADAM_BETAS, ADAM_EPS, LEARNING_RATES, STEP_CHECKPOINTS, candidate_grid
from .imprint import IMPRINT_DIRECTIONS, class_directions, fit_shared_offset, retain_offset
from .imprint import candidate_grid as imprint_grid
from .ridge import RIDGE_SCALES, margin_targets, ridge_solve
from .runner import CandidatePatch, Evidence, Generator, MethodSpec, execute_run, read_model_spec

GRAPHNEW_REF_LABEL = "e7"
"""Run-directory label of E7 references: ``baselines/<instance_id>/<method>/e7/<seed>/``."""
MASK_FRACTION = 0.19
"""Entries trained per tensor by the sparse references (E5's masks); under the 20% changed-entry cap."""
TRAJ_LR = 1e-3
TRAJ_STEPS = 300
"""The unconstrained Adam point theta_full: the traj mask's trajectory and interp's endpoint (grad_full's q3)."""
INTERP_ALPHAS: tuple[float, ...] = tuple(k / 8 for k in range(1, 9))
TRAINABLE_PARTS = ("weight", "bias")
"""Trainable parts of every layer; BatchNorm running_mean and running_var are never trained."""

_NOT_TRAINING_FREE = "not training-free"
GRAPH_REF_SPECS: dict[str, MethodSpec] = {
    "grad_full": MethodSpec(
        method="grad_full",
        label=("unconstrained gradient reference: Adam on every trainable tensor of the GCN (conv weights and biases, "
               f"BatchNorm weight and bias; never the running statistics), edit limits ignored; {_NOT_TRAINING_FREE}"),
        uses_labels_for_fit=True,
        edit_limits_enforced=False,
    ),
    "grad_last": MethodSpec(
        method="grad_last",
        label=f"gradient reference: Adam on the head (last graph_conv layer), E7 program limits enforced; "
              f"{_NOT_TRAINING_FREE}",
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
    "grad_sparse19_sal": MethodSpec(
        method="grad_sparse19_sal",
        label=("gradient reference: Adam on every trainable tensor, a fixed top 19% of each tensor's entries by base "
               f"gradient saliency on the evidence (E5's mask), E7 program limits enforced; {_NOT_TRAINING_FREE}"),
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
    "grad_sparse19_traj": MethodSpec(
        method="grad_sparse19_traj",
        label=("gradient reference: Adam on every trainable tensor, a fixed top 19% of each tensor's entries by |delta| "
               "after unconstrained Adam lr 1e-3 for 300 steps (E5's mask), E7 program limits enforced; "
               f"{_NOT_TRAINING_FREE}"),
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
    "interp": MethodSpec(
        method="interp",
        label=("WiSE-FT-style reference: base + alpha (theta_full - base), alpha in {1/8, ..., 8/8}, theta_full the "
               f"unconstrained Adam point at lr 1e-3 and 300 steps, edit limits ignored; {_NOT_TRAINING_FREE}"),
        uses_labels_for_fit=True,
        edit_limits_enforced=False,
    ),
    "ridge_margin": MethodSpec(
        method="ridge_margin",
        label=("closed-form label fit of the head on z = (A H2)[evidence] with margin targets (true-class logit raised "
               f"to the best other logit + c), E7 program limits enforced (head exempt); {_NOT_TRAINING_FREE}"),
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
    "imprint": MethodSpec(
        method="imprint",
        label=("gradient-free label fit: new-class head rows imprinted from class-mean aggregated penultimate features "
               "z = (A H2)[evidence] (mean or centred direction, 4 scales relative to the train-class rows), one shared "
               f"new-class bias offset fitted for balanced evidence accuracy; train-class rows unchanged; "
               f"{_NOT_TRAINING_FREE}"),
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
    "imprint_retain": MethodSpec(
        method="imprint_retain",
        label=("equal-retention supplement of PREREG E7 only, never the gate: imprint's 8 candidates with the shared "
               "new-class bias offset set to the most new-class-favouring value that breaks no replay node the base "
               "classifies correctly"),
        uses_labels_for_fit=True,
        edit_limits_enforced=True,
    ),
}
GRAPH_REF_METHODS: tuple[str, ...] = tuple(GRAPH_REF_SPECS)
GATE_METHODS: tuple[str, ...] = tuple(method for method in GRAPH_REF_METHODS if method != "imprint_retain")
"""The references whose best all-class accuracy sets the E7 bar (imprint_retain competes in the supplement only)."""


class GraphRefError(RuntimeError):
    """A graph reference could not be built as specified (parity, graph file, evidence shape)."""


# ---------------------------------------------------------------------------------------------------------------
# The functional GCN


@dataclass(frozen=True)
class GraphOps:
    """The fixed graph inputs of a GCN forward in one dtype: the CSR adjacency A and A X (the first layer's input)."""

    adj: torch.Tensor
    ax: torch.Tensor
    n_layers: int
    bn_eps: float

    @property
    def n_nodes(self) -> int:
        return int(self.ax.shape[0])


def load_instance_graph(instance: Instance) -> dict[str, np.ndarray]:
    """indptr, indices, weight and features of the instance's graph.npz, hash-checked against private.json."""
    path = Path(instance.instance_dir) / GRAPH_FILE
    if file_sha256(path) != instance.private.get("graph_file_sha256"):
        raise GraphRefError(f"{path}: hash differs from private.json graph_file_sha256")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in ("indptr", "indices", "weight", "features")}


def gcn_shape(params: Mapping[str, np.ndarray], spec: Mapping[str, Any]) -> tuple[int, float]:
    """(K, BatchNorm eps) of a GCN student: conv1 .. convK and bn1 .. bn(K-1) in ``params``, one eps in model_spec."""
    convs = sorted(int(key[4:].split(".")[0]) for key in params if key.startswith("conv") and key.endswith(".weight"))
    k = len(convs)
    expected = {f"conv{i}.{part}" for i in range(1, k + 1) for part in TRAINABLE_PARTS}
    expected |= {f"bn{i}.{part}" for i in range(1, k) for part in BN_PARTS}
    if k < 2 or convs != list(range(1, k + 1)) or set(params) != expected:
        raise GraphRefError(f"not a GCN student with conv1..convK and bn1..bn(K-1): {sorted(params)}")
    eps = {float(layer["eps"]) for layer in spec.get("layers", []) if layer.get("op") == "batchnorm_eval"}
    if len(eps) != 1:
        raise GraphRefError(f"model_spec.json must give one BatchNorm eps, found {sorted(eps)}")
    return k, eps.pop()


def trainable_tensors(params: Mapping[str, np.ndarray]) -> list[str]:
    """Every conv weight and bias and BatchNorm weight and bias; never the running statistics."""
    return sorted(key for key in params if key.rsplit(".", 1)[1] in TRAINABLE_PARTS)


def graph_ops(graph: Mapping[str, np.ndarray], n_layers: int, bn_eps: float, dtype: torch.dtype) -> GraphOps:
    """A and A X in ``dtype``; A X is computed in float64 and cast once, as forward.py computes it in float64."""
    adj64, features64 = graph_tensors(graph, torch.device("cpu"), torch.float64)
    ax = (adj64 @ features64).to(dtype)
    adj = adj64 if dtype == torch.float64 else graph_tensors(graph, torch.device("cpu"), dtype)[0]
    return GraphOps(adj=adj, ax=ax, n_layers=n_layers, bn_eps=bn_eps)


def to_tensors(params: Mapping[str, np.ndarray], dtype: torch.dtype,
               trainable: Sequence[str] = ()) -> dict[str, torch.Tensor]:
    """Fresh torch copies of ``params`` in ``dtype``; the ``trainable`` ones require gradients."""
    unknown = sorted(set(trainable) - set(params))
    if unknown:
        raise GraphRefError(f"trainable tensors {unknown} are not parameters of the student")
    tensors = {}
    for key, value in params.items():
        tensor = torch.tensor(np.asarray(value), dtype=dtype)
        tensors[key] = tensor.requires_grad_(key in trainable)
    return tensors


def _batchnorm(h: torch.Tensor, t: Mapping[str, torch.Tensor], name: str, eps: float) -> torch.Tensor:
    mean, var = t[f"{name}.running_mean"], t[f"{name}.running_var"]
    return (h - mean) / torch.sqrt(var + eps) * t[f"{name}.weight"] + t[f"{name}.bias"]


def penultimate(t: Mapping[str, torch.Tensor], ops: GraphOps) -> torch.Tensor:
    """H_{K-1} at every node: the output of the last hidden layer (after BatchNorm and ReLU), eval form."""
    h = ops.ax @ t["conv1.weight"].T + t["conv1.bias"]
    for index in range(1, ops.n_layers):
        if index > 1:
            h = _SymmetricSpMM.apply(ops.adj, h @ t[f"conv{index}.weight"].T) + t[f"conv{index}.bias"]
        h = F.relu(_batchnorm(h, t, f"bn{index}", ops.bn_eps))
    return h


def gcn_logits(t: Mapping[str, torch.Tensor], ops: GraphOps) -> torch.Tensor:
    """Logits of every node: forward.py's function, A (H_{K-1} W_K^T) + b_K."""
    k = ops.n_layers
    return _SymmetricSpMM.apply(ops.adj, penultimate(t, ops) @ t[f"conv{k}.weight"].T) + t[f"conv{k}.bias"]


def head_input(t: Mapping[str, torch.Tensor], ops: GraphOps) -> torch.Tensor:
    """z = A H_{K-1} at every node, the exact input of the head: logits = z W_K^T + b_K."""
    return _SymmetricSpMM.apply(ops.adj, penultimate(t, ops))


def rebuild_parity(params: Params, forward: ForwardFn, ops64: GraphOps, ops32: GraphOps) -> dict[str, float]:
    """``check_logit_parity`` of forward.py at every node against the float64 and float32 functional rebuilds."""
    mine = np.asarray(forward(dict(params), np.arange(ops64.n_nodes, dtype=np.int64)))
    with torch.no_grad():
        exact = gcn_logits(to_tensors(params, torch.float64), ops64).numpy()
        single = gcn_logits(to_tensors(params, torch.float32), ops32).numpy()
    return {**check_logit_parity(mine, exact, single), "n_nodes": int(ops64.n_nodes)}


def export_child(base: Params, t: Mapping[str, torch.Tensor], trainable: Sequence[str]) -> dict[str, np.ndarray]:
    """``base`` with the ``trainable`` tensors replaced by their current values in the base dtype."""
    child = dict(base)
    for key in trainable:
        child[key] = t[key].detach().numpy().astype(base[key].dtype).copy()
    return child


LossFn = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]


def adam_checkpoints(
    params: Params,
    loss_fn: LossFn,
    *,
    trainable: Sequence[str],
    lr: float,
    checkpoints: Sequence[int],
    seed: int,
    masks: Mapping[str, np.ndarray] | None = None,
) -> list[tuple[int, dict[str, np.ndarray], float]]:
    """Full-batch Adam (``grad.py``'s betas and eps, no weight decay) on ``loss_fn``; (steps, child, loss) at each
    checkpoint, the loss evaluated after that step.

    Only ``trainable`` tensors receive gradients, in float32. With ``masks``, each gradient is multiplied by its mask
    before the step, so masked-out entries keep their base values exactly (zero gradient and zero Adam moments give no
    update).
    """
    torch.manual_seed(seed)
    t = to_tensors(params, torch.float32, trainable)
    gradient_masks = {key: torch.from_numpy(value.astype(np.float32)) for key, value in (masks or {}).items()}
    optimizer = torch.optim.Adam([t[key] for key in trainable], lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS,
                                 weight_decay=0.0, foreach=False)
    wanted = set(checkpoints)
    snapshots: list[tuple[int, dict[str, np.ndarray], float]] = []
    for step in range(1, max(checkpoints) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_fn(t).backward()
        for key, mask in gradient_masks.items():
            grad = t[key].grad
            if grad is None:
                raise GraphRefError(f"{key} has no gradient")
            grad.mul_(mask)
        optimizer.step()
        if step in wanted:
            with torch.no_grad():
                loss = float(loss_fn(t))
            snapshots.append((step, export_child(params, t, trainable), loss))
    return snapshots


def top_fraction_masks(score: Mapping[str, np.ndarray], keys: Sequence[str],
                       fraction: float = MASK_FRACTION) -> dict[str, np.ndarray]:
    """Per tensor, the top ``floor(fraction x size)`` entries by |score| (stable argsort of -|score|, ties by flat
    index), E5's mask rule."""
    masks = {}
    for key in keys:
        flat = np.abs(np.asarray(score[key], dtype=np.float64)).ravel()
        k = int(np.floor(fraction * flat.size))
        if k < 1:
            raise GraphRefError(f"{key}: {flat.size} entries give an empty {fraction:g} mask")
        order = np.argsort(-flat, kind="stable")
        mask = np.zeros(flat.size, dtype=bool)
        mask[order[:k]] = True
        masks[key] = mask.reshape(np.shape(score[key]))
    return masks


def _check_frozen(base: Params, child: Mapping[str, np.ndarray], trainable: Sequence[str],
                  masks: Mapping[str, np.ndarray] | None) -> None:
    for key, parent in base.items():
        if key not in trainable:
            if not np.array_equal(child[key], parent):
                raise GraphRefError(f"{key} is not trainable but changed")
        elif masks is not None and not np.array_equal(child[key][~masks[key]], parent[~masks[key]]):
            raise GraphRefError(f"{key} changed outside its mask")


# ---------------------------------------------------------------------------------------------------------------
# Shared context of one generator call


@dataclass
class _Context:
    params: Params
    ops32: GraphOps
    ops64: GraphOps
    ids: torch.Tensor
    y: torch.Tensor
    head: tuple[str, str]
    limits: Limits
    info: dict[str, Any]


def _evidence_parts(evidence: Evidence) -> np.ndarray:
    if evidence.rule != GRAPHNEW_RULE or evidence.n_fail or not evidence.n_rand or not evidence.n_replay:
        raise GraphRefError(f"E7 references need {GRAPHNEW_RULE} evidence (new-class nodes, then replay nodes), "
                            f"got {evidence.rule}")
    if evidence.x.ndim != 1 or evidence.x.dtype != np.int64:
        raise GraphRefError(f"graphnew evidence must be int64 node ids [N], got {evidence.x.dtype}{evidence.x.shape}")
    return np.arange(len(evidence.y)) < evidence.n_rand


def _context(instance: Instance, evidence: Evidence) -> _Context:
    """Graph, rebuilds and parity at every node, before any fit."""
    started = time.perf_counter()
    _evidence_parts(evidence)
    params = instance.params
    spec = read_model_spec(instance)
    k, eps = gcn_shape(params, spec)
    head = tuple(head_tensors(spec))
    if head != (f"conv{k}.weight", f"conv{k}.bias"):
        raise GraphRefError(f"model_spec.json head {head} is not conv{k}")
    graph = load_instance_graph(instance)
    check_symmetric(graph)
    ops64, ops32 = graph_ops(graph, k, eps, torch.float64), graph_ops(graph, k, eps, torch.float32)
    del graph
    parity = rebuild_parity(params, instance.forward, ops64, ops32)
    info = {
        "rebuild": "functional torch GCN, eval form (no dropout; BatchNorm running statistics fixed), first layer "
                   "(A X) W1^T + b1 with A X precomputed in float64, as forward.py",
        "n_layers": k,
        "bn_eps": eps,
        "n_nodes": ops64.n_nodes,
        "parity_all_nodes": parity,
        "n_params": int(sum(value.size for value in params.values())),
        "setup_s": time.perf_counter() - started,
    }
    return _Context(
        params=params, ops32=ops32, ops64=ops64,
        ids=torch.from_numpy(np.ascontiguousarray(evidence.x, dtype=np.int64)),
        y=torch.from_numpy(np.ascontiguousarray(evidence.y, dtype=np.int64)),
        head=(head[0], head[1]),
        limits=graphnew_program_limits(instance.task["limits"], len(params)),
        info=info,
    )


def _full_loss(ctx: _Context) -> LossFn:
    def loss(t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(gcn_logits(t, ctx.ops32)[ctx.ids], ctx.y)
    return loss


def _evidence_loss(ctx: _Context, child: Params) -> float:
    with torch.no_grad():
        return float(_full_loss(ctx)(to_tensors(child, torch.float32)))


def _z64(ctx: _Context, forward: ForwardFn, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """z = (A H_{K-1})[x] and z0 = z W^T + b in float64, and the max relative distance of z0 from forward.py's logits
    (checked against one float32 step of max(1, |logit|), the float64 parity rule)."""
    with torch.no_grad():
        z = head_input(to_tensors(ctx.params, torch.float64), ctx.ops64)[torch.from_numpy(x)].numpy()
    w, b = (ctx.params[name].astype(np.float64) for name in ctx.head)
    z0 = z @ w.T + b
    theirs = np.asarray(forward(dict(ctx.params), x), dtype=np.float64)
    rel = float(np.max(np.abs(z0 - theirs) / np.maximum(1.0, np.abs(z0))))
    if not rel <= FLOAT32_STEP:
        raise GraphRefError(f"head input z gives logits {rel:.3g} (relative) away from forward.py > {FLOAT32_STEP:.3g}")
    return z, z0, rel


def _scale(ctx: _Context, child: Params, trainable: Sequence[str], notes: dict[str, Any]) -> Params:
    """Scale the over-limit tensors of ``child`` to the norm limit (head exempt) and record the scaling."""
    child, scaling = scale_to_norm_limit(ctx.params, child, trainable, limit=ctx.limits.max_rel_frob,
                                         min_ndim=ctx.limits.rel_frob_min_ndim, exempt=ctx.limits.rel_frob_exempt)
    notes["norm_scaling"] = scaling
    notes["norm_scaled_tensors"] = [key for key, entry in scaling.items() if entry["scaled"]]
    return child


# ---------------------------------------------------------------------------------------------------------------
# Generators


def _grad_hypothesis(method: str, scope: str, lr: float, steps: int, evidence: Evidence) -> str:
    return (f"{method}: Adam lr={lr:g} for {steps} full-batch steps on the mean cross-entropy over the "
            f"{len(evidence.y)} evidence nodes ({evidence.composition()}), full-graph forward; trained: {scope}")


_PREDICTED_GRAD = ("nodes of the new classes recognised if the fit on 10 labelled nodes per class generalizes over the "
                   "graph; retention is fitted on the replay nodes of the train classes")


def _gradient_candidates(ctx: _Context, method: str, evidence: Evidence, run_seed: int, trainable: Sequence[str],
                         loss_fn: LossFn, scope: str, enforced: bool,
                         masks: Mapping[str, np.ndarray] | None = None) -> list[CandidatePatch]:
    candidates: list[CandidatePatch] = []
    for lr in LEARNING_RATES:
        started = time.perf_counter()
        for steps, child, loss in adam_checkpoints(ctx.params, loss_fn, trainable=trainable, lr=lr,
                                                   checkpoints=STEP_CHECKPOINTS, seed=run_seed, masks=masks):
            _check_frozen(ctx.params, child, trainable, masks)
            notes: dict[str, Any] = {"evidence_loss_at_checkpoint": loss,
                                     "trajectory_s_to_checkpoint": time.perf_counter() - started}
            if enforced:
                child = _scale(ctx, child, trainable, notes)
                if notes["norm_scaled_tensors"]:
                    notes["evidence_loss_after_scaling"] = _evidence_loss(ctx, child)
            candidates.append(CandidatePatch(
                slot=len(candidates), child=child, config={"lr": lr, "steps": steps},
                hypothesis=_grad_hypothesis(method, scope, lr, steps, evidence),
                predicted_effect=_PREDICTED_GRAD, notes=notes))
    return candidates


def _theta_full(ctx: _Context, run_seed: int) -> dict[str, np.ndarray]:
    """The unconstrained Adam point at lr TRAJ_LR and TRAJ_STEPS steps on every trainable tensor."""
    ((_, theta, _),) = adam_checkpoints(ctx.params, _full_loss(ctx), trainable=trainable_tensors(ctx.params),
                                        lr=TRAJ_LR, checkpoints=(TRAJ_STEPS,), seed=run_seed)
    return theta


def graph_grad_generator(instance: Instance, method: str, run_seed: int) -> Generator:
    """Candidate generator of grad_full, grad_last, grad_sparse19_sal or grad_sparse19_traj on a graphnew instance."""
    if method not in ("grad_full", "grad_last", "grad_sparse19_sal", "grad_sparse19_traj"):
        raise ValueError(f"{method!r} is not an E7 gradient reference")
    enforced = GRAPH_REF_SPECS[method].edit_limits_enforced

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        ctx = _context(instance, evidence)
        params = ctx.params
        info = dict(ctx.info)
        info["base_evidence_loss"] = _evidence_loss(ctx, params)
        masks = None
        if method == "grad_last":
            trainable = list(ctx.head)
            z, _, rel = _z64(ctx, instance.forward, evidence.x)
            z32 = torch.from_numpy(z.astype(np.float32))
            weight_name, bias_name = ctx.head

            def loss_fn(t: Mapping[str, torch.Tensor]) -> torch.Tensor:
                return F.cross_entropy(z32 @ t[weight_name].T + t[bias_name], ctx.y)

            info["head_input_float64_parity_rel"] = rel
            info["computation"] = ("head only: z = (A H_{K-1})[evidence] in float64 (cast to float32) is constant "
                                   "while only the head trains; logits = z W^T + b")
            scope = f"the head ({weight_name}, {bias_name})"
        else:
            trainable = trainable_tensors(params)
            loss_fn = _full_loss(ctx)
            scope = "every trainable tensor (conv weights and biases, BatchNorm weight and bias), no edit limits"
            if method.startswith("grad_sparse19"):
                if method == "grad_sparse19_sal":
                    t = to_tensors(params, torch.float32, trainable)
                    loss_fn(t).backward()
                    score = {key: t[key].grad.detach().numpy().copy() for key in trainable}
                else:
                    theta = _theta_full(ctx, run_seed)
                    score = {key: theta[key].astype(np.float64) - params[key].astype(np.float64) for key in trainable}
                    info["traj_theta_sha256"] = params_sha256(theta)
                masks = top_fraction_masks(score, trainable)
                n_mask = int(sum(mask.sum() for mask in masks.values()))
                info.update({"mask_rule": method, "mask_fraction_per_tensor": MASK_FRACTION,
                             "n_mask_entries": n_mask, "mask_frac_of_all_entries": n_mask / info["n_params"]})
                scope = (f"every trainable tensor, a fixed top {MASK_FRACTION:g} of each tensor's entries by the "
                         f"{'base gradient saliency' if method.endswith('sal') else 'unconstrained trajectory'} mask")
        info["trainable"] = trainable
        info["n_trainable_entries"] = int(sum(params[k].size if masks is None else masks[k].sum() for k in trainable))
        candidates = _gradient_candidates(ctx, method, evidence, run_seed, trainable, loss_fn, scope, enforced, masks)
        if enforced:
            info["n_candidates_norm_scaled"] = sum(1 for c in candidates if c.notes["norm_scaled_tensors"])
        return candidates, info

    return generate


def interp_generator(instance: Instance, run_seed: int) -> Generator:
    """Candidate generator of interp: base + alpha (theta_full - base) for alpha in INTERP_ALPHAS."""

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        ctx = _context(instance, evidence)
        params = ctx.params
        trainable = trainable_tensors(params)
        info = dict(ctx.info)
        info["base_evidence_loss"] = _evidence_loss(ctx, params)
        started = time.perf_counter()
        theta = _theta_full(ctx, run_seed)
        info["theta_full"] = {"lr": TRAJ_LR, "steps": TRAJ_STEPS, "sha256": params_sha256(theta),
                              "trajectory_s": time.perf_counter() - started,
                              "evidence_loss": _evidence_loss(ctx, theta)}
        info["trainable"] = trainable
        candidates = []
        for alpha in INTERP_ALPHAS:
            child = dict(params)
            for key in trainable:
                base64 = params[key].astype(np.float64)
                child[key] = (base64 + alpha * (theta[key].astype(np.float64) - base64)).astype(params[key].dtype)
            candidates.append(CandidatePatch(
                slot=len(candidates), child=child, config={"alpha": alpha},
                hypothesis=(f"interp: base + {alpha:g} x (theta_full - base) on every trainable tensor, theta_full = "
                            f"Adam lr={TRAJ_LR:g} for {TRAJ_STEPS} full-batch steps on the {len(evidence.y)} evidence "
                            f"nodes ({evidence.composition()}); no edit limits"),
                predicted_effect=("a partial step toward the fitted point keeps more of the base's old-class "
                                  "behaviour at some cost in new-class recognition"),
                notes={"evidence_loss": _evidence_loss(ctx, child)}))
        if not params_sha256(candidates[-1].child) == params_sha256(theta):
            raise GraphRefError("interp at alpha 1 is not theta_full")
        return candidates, info

    return generate


def ridge_margin_generator(instance: Instance) -> Generator:
    """Candidate generator of ridge_margin on the head input z (E2's rule, ``weightcode.baselines.fit.ridge``)."""

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        ctx = _context(instance, evidence)
        params = ctx.params
        weight_name, bias_name = ctx.head
        z, z0, rel = _z64(ctx, instance.forward, evidence.x)
        h1 = np.hstack([z, np.ones((len(z), 1))])
        w0 = np.hstack([params[weight_name].astype(np.float64), params[bias_name].astype(np.float64)[:, None]])
        targets, scale = margin_targets(z0, evidence.y)
        unit = float(np.mean(np.sum(h1**2, axis=1)))
        residual_base = float(np.linalg.norm(targets - z0))
        if not residual_base > 0.0:
            raise GraphRefError("ridge_margin: the targets equal the base logits on every evidence node")
        candidates = []
        for relative in RIDGE_SCALES:
            lam = relative * unit
            w = ridge_solve(h1, targets, w0, lam)
            child = dict(params)
            child[weight_name] = w[:, :-1].astype(params[weight_name].dtype)
            child[bias_name] = w[:, -1].astype(params[bias_name].dtype)
            notes: dict[str, Any] = {"target_residual_rel": float(np.linalg.norm(h1 @ w.T - targets)) / residual_base}
            child = _scale(ctx, child, [weight_name, bias_name], notes)
            candidates.append(CandidatePatch(
                slot=len(candidates), child=child, config={"lam_relative": relative, "lam": lam},
                hypothesis=(f"ridge_margin: closed-form refit of {weight_name} and {bias_name} toward the base with "
                            f"lam = {relative:g} x {unit:.6g} on z = (A H2) at the {len(evidence.y)} evidence nodes "
                            f"({evidence.composition()}); targets: true-class logit raised to the best other + "
                            f"{scale:.4g}"),
                predicted_effect="new-class nodes recognised where their aggregated features resemble the evidence",
                notes=notes))
        info = {**ctx.info, "head_input_float64_parity_rel": rel, "target_scale_c": scale,
                "lam_unit_mean_sq_feature_norm": unit, "n_features_with_bias": int(h1.shape[1]),
                "n_evidence": int(len(h1)), "trainable": [weight_name, bias_name],
                "n_evidence_raised": int(np.count_nonzero(np.any(targets != z0, axis=1))),
                "n_candidates_norm_scaled": sum(1 for c in candidates if c.notes["norm_scaled_tensors"])}
        return candidates, info

    return generate


def graph_imprint_generator(instance: Instance, method: str = "imprint") -> Generator:
    """Candidate generator of imprint or imprint_retain on the head input z (E2's rules,
    ``weightcode.baselines.fit.imprint``): only the new-class rows of the head's weight and bias change."""
    if method not in ("imprint", "imprint_retain"):
        raise ValueError(f"{method!r} is not an E7 imprint reference")
    retain = method == "imprint_retain"
    old_classes = [int(c) for c in instance.private["train_classes"]]
    new_classes = [int(c) for c in instance.private["new_classes"]]

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        is_new_row = _evidence_parts(evidence)
        ctx = _context(instance, evidence)
        params = ctx.params
        weight_name, bias_name = ctx.head
        z, z0, rel = _z64(ctx, instance.forward, evidence.x)
        y = evidence.y
        w0 = params[weight_name].astype(np.float64)
        rho = float(np.mean(np.linalg.norm(w0[old_classes], axis=1)))
        step = float(np.mean(np.max(z0, axis=1) - np.mean(z0, axis=1)))
        center = (z[is_new_row].mean(axis=0) + z[~is_new_row].mean(axis=0)) / 2
        directions, n_without = {}, {}
        for direction in IMPRINT_DIRECTIONS:
            directions[direction], n_without[direction] = class_directions(
                z[is_new_row], y[is_new_row], new_classes, center if direction == "centered" else None)
        z_old = z0[:, old_classes]
        candidates = []
        for direction, scale in imprint_grid():
            rows = scale * rho * directions[direction]
            offset0 = -(rows @ center) if direction == "centered" else np.zeros(len(new_classes))
            u_new = z @ rows.T + offset0
            if retain:
                fit = retain_offset(z_old, u_new, y, evidence.base_pred, old_classes, new_classes, is_new_row, step)
            else:
                fit = fit_shared_offset(z_old, u_new, y, old_classes, new_classes, is_new_row, step)
            child = dict(params)
            weight, bias = params[weight_name].copy(), params[bias_name].copy()
            weight[new_classes] = rows.astype(weight.dtype)
            bias[new_classes] = (offset0 + fit["beta"]).astype(bias.dtype)
            child[weight_name], child[bias_name] = weight, bias
            pred = np.argmax(z @ weight.astype(np.float64).T + bias.astype(np.float64), axis=1)
            right = pred == y
            notes: dict[str, Any] = {**fit, "realized_acc_new": float(right[is_new_row].mean()),
                                     "realized_acc_old": float(right[~is_new_row].mean())}
            notes["realized_balanced_acc"] = (notes["realized_acc_new"] + notes["realized_acc_old"]) / 2
            if retain:  # checked with forward.py, the predictor of the oracle and the readouts
                kept = ~is_new_row & (evidence.base_pred == y)
                (child_pred,) = predict_nodes(instance.forward, child, evidence.x)
                broken = int(np.count_nonzero(kept & (child_pred != y)))
                if broken:
                    raise GraphRefError(f"{direction} x {scale:g}: the child breaks {broken} replay nodes the base "
                                        f"classifies correctly (beta {fit['beta']:.6g}, beta_max {fit['beta_max']:.6g})")
                notes["realized_replay_base_right_broken"] = broken
                bias_rule = (f"beta = {fit['beta']:.6g}, the most new-class-favouring offset that breaks none of the "
                             f"{fit['n_replay_base_right']} replay nodes the base classifies correctly")
            else:
                bias_rule = f"beta = {fit['beta']:.6g}, fitted for balanced evidence accuracy"
            child = _scale(ctx, child, [weight_name, bias_name], notes)
            candidates.append(CandidatePatch(
                slot=len(candidates), child=child, config={"direction": direction, "scale": scale},
                hypothesis=(f"{method}: rows {new_classes} of {weight_name} set to {scale:g} x {rho:.6g} (mean "
                            f"train-class row norm) x the {direction} direction of their classes' aggregated "
                            f"penultimate features z = (A H2) on the {len(y)} evidence nodes "
                            f"({evidence.composition()}); {bias_name} of those classes "
                            f"{'-w.m + ' if direction == 'centered' else ''}{bias_rule}; train-class rows unchanged"),
                predicted_effect=("the new classes are recognised where the aggregated features lie near their "
                                  "evidence means; train-class nodes keep their predictions unless an imprinted logit "
                                  "overtakes them"),
                notes=notes))
        cosines = {d: float(np.mean((r @ r.T)[~np.eye(len(new_classes), dtype=bool)])) if len(new_classes) > 1
                   else float("nan") for d, r in directions.items()}
        info = {**ctx.info, "head_input_float64_parity_rel": rel, "train_classes": old_classes,
                "new_classes": new_classes, "rho_mean_train_class_row_norm": rho, "offset_step_logit_scale": step,
                "center_norm": float(np.linalg.norm(center)), "rows_without_direction": n_without,
                "mean_cosine_between_new_directions": cosines, "trainable": [weight_name, bias_name],
                "changed_rows": new_classes}
        return candidates, info

    return generate


# ---------------------------------------------------------------------------------------------------------------
# Configs and runs


def graph_ref_config(method: str, run_seed: int, instance: Instance) -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    if method not in GRAPH_REF_SPECS:
        raise ValueError(f"unknown E7 reference {method!r}; expected one of {GRAPH_REF_METHODS}")
    limits = graphnew_program_limits(instance.task["limits"], len(instance.params))
    common: dict[str, Any] = {
        "prereg": "E7 references",
        "evidence": "the 80 new-class nodes then the 64 replay nodes of the instance (the LEARN evidence)",
        "forward": "full graph of the instance's graph.npz per evaluation; eval form (no dropout, BatchNorm frozen)",
        "parity": "check_logit_parity of forward.py at every node, float64 and float32 rebuilds, before any fit",
        "seed": run_seed,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "norm_limit": limits.max_rel_frob if GRAPH_REF_SPECS[method].edit_limits_enforced else None,
        "norm_limit_exempt": list(limits.rel_frob_exempt),
    }
    if method in ("ridge_margin", "imprint", "imprint_retain"):
        common.update({"features": "z = (A H2)[evidence] in float64, the exact head input (logits = z W^T + b)",
                       "randomness": "none; the run seed only names the output directory"})
        if method == "ridge_margin":
            common.update({"lam_relative_grid": list(RIDGE_SCALES),
                           "lam_unit": "mean over evidence nodes of ||[z, 1]||^2",
                           "objective": "||[Z 1] W^T - T||^2 + lam ||W - W0||^2 over the head weight and bias",
                           "targets": "T = Z0, true-class logit raised to max(Z0 true, best other Z0 + c), "
                                      "c = mean(max logit - mean logit) of the base on evidence"})
        else:
            common.update({"candidate_order": [{"direction": d, "scale": s} for d, s in imprint_grid()],
                           "offset_rule": "retain" if method == "imprint_retain" else "balanced",
                           "unchanged": "train-class rows and biases of the head, every other tensor"})
        return common
    common.update({"optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0},
                   "loss": "mean cross-entropy over the 144 evidence nodes", "batch": "full", "dtype": "float32"})
    if method == "interp":
        common.update({"alphas": list(INTERP_ALPHAS), "theta_full": {"lr": TRAJ_LR, "steps": TRAJ_STEPS},
                       "trained": "every trainable tensor (conv weights and biases, BatchNorm weight and bias)"})
        return common
    common.update({"learning_rates": list(LEARNING_RATES), "step_checkpoints": list(STEP_CHECKPOINTS),
                   "candidate_order": [{"lr": lr, "steps": steps} for lr, steps in candidate_grid()]})
    if method == "grad_last":
        common["trained"] = head_tensors(read_model_spec(instance))
    else:
        common["trained"] = "every trainable tensor (conv weights and biases, BatchNorm weight and bias)"
    if method.startswith("grad_sparse19"):
        common.update({"mask_fraction_per_tensor": MASK_FRACTION,
                       "mask": ("|d loss / d w| at the base weights" if method.endswith("sal") else
                                f"|w - w_base| after unconstrained Adam lr {TRAJ_LR:g}, {TRAJ_STEPS} steps"),
                       "mask_rule": "top floor(0.19 x size) per tensor, stable argsort of -|score|, ties by flat index"})
    return common


def graph_ref_generator(instance: Instance, method: str, run_seed: int) -> Generator:
    """The candidate generator of one E7 reference."""
    if method in ("grad_full", "grad_last", "grad_sparse19_sal", "grad_sparse19_traj"):
        return graph_grad_generator(instance, method, run_seed)
    if method == "interp":
        return interp_generator(instance, run_seed)
    if method == "ridge_margin":
        return ridge_margin_generator(instance)
    if method in ("imprint", "imprint_retain"):
        return graph_imprint_generator(instance, method)
    raise ValueError(f"unknown E7 reference {method!r}; expected one of {GRAPH_REF_METHODS}")


def run_graph_ref(instance: Instance, method: str, run_dir: Path, *, run_seed: int = 0) -> dict[str, Any]:
    """Run one E7 reference on one graphnew-v1 instance (dev only, ``execute_run`` with ``graphnew``); returns its run
    record (also run.json). A run whose candidates the executor rejects all fails with ``NoValidCandidate``."""
    return execute_run(
        instance=instance,
        spec=GRAPH_REF_SPECS[method],
        run_dir=run_dir,
        run_seed=run_seed,
        config=graph_ref_config(method, run_seed, instance),
        generate=graph_ref_generator(instance, method, run_seed),
        require_valid_candidate=True,
        graphnew=True,
    )


__all__ = [
    "GATE_METHODS", "GRAPHNEW_REF_LABEL", "GRAPH_REF_METHODS", "GRAPH_REF_SPECS", "INTERP_ALPHAS", "MASK_FRACTION",
    "TRAJ_LR", "TRAJ_STEPS", "GraphOps", "GraphRefError", "adam_checkpoints", "export_child", "gcn_logits",
    "gcn_shape", "graph_ops", "graph_ref_config", "graph_ref_generator", "head_input", "load_instance_graph",
    "penultimate", "rebuild_parity", "run_graph_ref", "to_tensors", "top_fraction_masks", "trainable_tensors",
]
