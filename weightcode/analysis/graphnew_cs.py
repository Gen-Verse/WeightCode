"""PREREG E7 instrument, never a reference and never gating: Correct-and-Smooth and label propagation on a graphnew
instance, with the 144 evidence nodes as the only labelled seeds.

It is not a weight edit. It shows how much the graph structure alone carries the new classes from 10 labelled nodes
per class, next to the references that edit the network.

Settings are the ogbn-arxiv settings of the official C&S code (Huang, He, Singh, Lim and Benson, "Combining Label
Propagation and Simple Models Out-performs Graph Neural Networks", ICLR 2021; github.com/CUAI/CorrectAndSmooth,
``run_experiments.py`` and ``outcome_correlation.py`` at master, read 2026-11-04), with ``train_only`` (the labelled
set is the seeds):

- ``cs_autoscale``, the arxiv "plain" dict: ``double_correlation_autoscale`` with A1 = AD, alpha1 0.87, 50
  propagations (residuals clamped to [-1, 1]), then A2 = DAD, alpha2 0.81, 50 propagations (clamped to [0, 1]);
- ``smooth_only``, the arxiv "gat" dict (the repository's setting on top of a GNN): ``only_outcome_correlation`` with
  A = DAD, alpha 0.8, 50 propagations;
- ``lp``: ``label_propagation`` from the seeds alone, without the base model: A = AD, alpha 0.9, 50 propagations.

Adjacency (``process_adj`` and ``gen_normalized_adjs``): the undirected edges of the instance's graph.npz without its
self loops, unweighted; deg the row sums; DAD = D^-1/2 A D^-1/2, DA = D^-1 A, AD = A D^-1 (a degree-0 node gets 0).
The base model output is the softmax of forward.py's logits at every node, in float64. Predictions are the argmax
over the 40 classes, as in the repository's evaluator; ``lp`` rows that received no mass are counted in the record.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

CS_SOURCE = ("github.com/CUAI/CorrectAndSmooth master, run_experiments.py (ogbn-arxiv dicts lp, plain, gat) and "
             "outcome_correlation.py, read 2026-11-04")
SETTINGS: dict[str, dict[str, Any]] = {
    "cs_autoscale": {"function": "double_correlation_autoscale", "A1": "AD", "alpha1": 0.87, "num_propagations1": 50,
                     "A2": "DAD", "alpha2": 0.81, "num_propagations2": 50, "train_only": True},
    "smooth_only": {"function": "only_outcome_correlation", "A": "DAD", "alpha": 0.8, "num_propagations": 50},
    "lp": {"function": "label_propagation", "A": "AD", "alpha": 0.9, "num_propagations": 50},
}


def normalized_adjacencies(indptr: np.ndarray, indices: np.ndarray, n_nodes: int) -> dict[str, torch.Tensor]:
    """DAD, DA and AD (float64 CSR) of the unweighted graph given by a CSR structure, self loops removed."""
    indptr, indices = np.asarray(indptr, np.int64), np.asarray(indices, np.int64)
    rows = np.repeat(np.arange(n_nodes, dtype=np.int64), np.diff(indptr))
    keep = rows != indices
    rows, cols = rows[keep], indices[keep]
    deg = np.bincount(rows, minlength=n_nodes).astype(np.float64)
    with np.errstate(divide="ignore"):
        d_isqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    new_indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n_nodes))]).astype(np.int64)
    weights = {"DAD": d_isqrt[rows] * d_isqrt[cols], "DA": d_isqrt[rows] ** 2, "AD": d_isqrt[cols] ** 2}
    return {name: torch.sparse_csr_tensor(torch.from_numpy(new_indptr), torch.from_numpy(cols),
                                          torch.from_numpy(value), size=(n_nodes, n_nodes))
            for name, value in weights.items()}


def outcome_correlation(adj: torch.Tensor, y: torch.Tensor, alpha: float, num_propagations: int,
                        post_step: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """``general_outcome_correlation`` with alpha_term True: r <- post(alpha A r + (1 - alpha) y), from r = y."""
    result = y.clone()
    for _ in range(num_propagations):
        result = alpha * (adj @ result) + (1 - alpha) * y
        result = post_step(result)
    return result


def _onehot(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.from_numpy(np.asarray(labels, np.int64)), n_classes).to(torch.float64)


def cs_autoscale(adjs: Mapping[str, torch.Tensor], model_out: torch.Tensor, seeds: np.ndarray,
                 seed_labels: np.ndarray, setting: Mapping[str, Any]) -> torch.Tensor:
    """``double_correlation_autoscale`` with train_only: correct (autoscaled residual propagation), then smooth."""
    n_classes = model_out.shape[1]
    idx = torch.from_numpy(np.asarray(seeds, np.int64))
    y = torch.zeros_like(model_out)
    y[idx] = _onehot(seed_labels, n_classes) - model_out[idx]
    resid = outcome_correlation(adjs[setting["A1"]], y, setting["alpha1"], setting["num_propagations1"],
                                lambda x: torch.clamp(x, -1.0, 1.0))
    orig_diff = y[idx].abs().sum() / idx.shape[0]
    resid_scale = orig_diff / resid.abs().sum(dim=1, keepdim=True)
    resid_scale[resid_scale.isinf()] = 1.0
    resid_scale[resid_scale > 1000] = 1.0
    res_result = model_out + resid_scale * resid
    nan = res_result.isnan()
    res_result[nan] = model_out[nan]
    y2 = res_result.clone()
    y2[idx] = _onehot(seed_labels, n_classes)
    return outcome_correlation(adjs[setting["A2"]], y2, setting["alpha2"], setting["num_propagations2"],
                               lambda x: torch.clamp(x, 0.0, 1.0))


def smooth_only(adjs: Mapping[str, torch.Tensor], model_out: torch.Tensor, seeds: np.ndarray,
                seed_labels: np.ndarray, setting: Mapping[str, Any]) -> torch.Tensor:
    """``only_outcome_correlation``: the base output with the seeds set to their labels, smoothed."""
    y = model_out.clone()
    y[torch.from_numpy(np.asarray(seeds, np.int64))] = _onehot(seed_labels, model_out.shape[1])
    return outcome_correlation(adjs[setting["A"]], y, setting["alpha"], setting["num_propagations"],
                               lambda x: torch.clamp(x, 0.0, 1.0))


def label_propagation(adjs: Mapping[str, torch.Tensor], n_nodes: int, n_classes: int, seeds: np.ndarray,
                      seed_labels: np.ndarray, setting: Mapping[str, Any]) -> torch.Tensor:
    """``label_propagation``: one-hot seeds, zeros elsewhere, propagated; no base model."""
    y = torch.zeros((n_nodes, n_classes), dtype=torch.float64)
    y[torch.from_numpy(np.asarray(seeds, np.int64))] = _onehot(seed_labels, n_classes)
    return outcome_correlation(adjs[setting["A"]], y, setting["alpha"], setting["num_propagations"],
                               lambda x: torch.clamp(x, 0.0, 1.0))


def softmax64(logits: np.ndarray) -> torch.Tensor:
    """Row softmax in float64."""
    z = torch.from_numpy(np.asarray(logits, dtype=np.float64))
    return torch.softmax(z, dim=1)


def instrument_outputs(indptr: np.ndarray, indices: np.ndarray, base_logits: np.ndarray, seeds: np.ndarray,
                       seed_labels: np.ndarray) -> dict[str, torch.Tensor]:
    """The three outputs of every node: cs_autoscale, smooth_only and lp (module docstring)."""
    n_nodes, n_classes = base_logits.shape
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError("seed nodes repeat")
    adjs = normalized_adjacencies(indptr, indices, n_nodes)
    model_out = softmax64(base_logits)
    return {
        "cs_autoscale": cs_autoscale(adjs, model_out, seeds, seed_labels, SETTINGS["cs_autoscale"]),
        "smooth_only": smooth_only(adjs, model_out, seeds, seed_labels, SETTINGS["smooth_only"]),
        "lp": label_propagation(adjs, n_nodes, n_classes, seeds, seed_labels, SETTINGS["lp"]),
    }


def zero_rows(output: torch.Tensor, nodes: Sequence[int] | np.ndarray) -> int:
    """How many of ``nodes`` received no mass (argmax would fall on class 0 by convention)."""
    rows = output[torch.from_numpy(np.asarray(nodes, np.int64))]
    return int((rows.abs().sum(dim=1) == 0).sum())


__all__ = ["CS_SOURCE", "SETTINGS", "cs_autoscale", "instrument_outputs", "label_propagation",
           "normalized_adjacencies", "outcome_correlation", "smooth_only", "softmax64", "zero_rows"]
