"""GCN students of a graph dataset (arch ``gcn``): the OGB reference GCN for ogbn-arxiv, its training and spec.

Model (``GCNConfig``; the OGB nodeproppred arxiv GCN, full batch): ``conv1`` .. ``convK`` with K = n_layers, each
h -> A @ (h @ W.T) + b on the normalized adjacency A of graph.npz (symmetric, self loops), widths n_features ->
hidden ... -> n_classes; after every layer but the last, BatchNorm1d (``bn1`` .. ``bn(K-1)``), ReLU and dropout.
Weights are Glorot-uniform and biases zero, as in PyG's GCNConv. The aggregation is a CSR sparse-dense product
whose backward multiplies by A again (A is symmetric; checked when the graph is loaded), so no scatter runs.

Training (``GCNTrainConfig``): Adam (lr 0.01, no weight decay), full batch, cross-entropy on the training nodes,
500 epochs, and the model after the last epoch is the model: no early stopping and no epoch selection. The OGB
script picks the epoch with the best valid accuracy; here the valid (2018) nodes are the deployment data, which
training never reads. Randomness: ``torch.manual_seed(seed)`` (initialization and dropout) under
``weightcode.models.train.configure_determinism``.

``forward.py`` (``templates/forward_gcn.py.tmpl``) computes the same function in numpy float64.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "gcn"
BN_PARTS: tuple[str, ...] = ("weight", "bias", "running_mean", "running_var")


@dataclass(frozen=True)
class GCNConfig:
    """Shape of a GCN student."""

    n_nodes: int
    n_features: int
    n_classes: int
    hidden: int = 256
    n_layers: int = 3
    dropout: float = 0.5
    bn_eps: float = 1e-5
    bn_momentum: float = 0.1

    def widths(self) -> list[int]:
        if self.n_layers < 1 or min(self.n_nodes, self.n_features, self.n_classes, self.hidden) < 1:
            raise ValueError(f"invalid GCN config {self}")
        return [self.n_features] + [self.hidden] * (self.n_layers - 1) + [self.n_classes]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GCNTrainConfig:
    """Fixed training recipe (module docstring); the shape fields fill a ``GCNConfig``."""

    arch: str = ARCH
    hidden: int = 256
    n_layers: int = 3
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 0.0
    epochs: int = 500
    log_every: int = 50

    def model_config(self, n_nodes: int, n_features: int, n_classes: int) -> GCNConfig:
        return GCNConfig(n_nodes, n_features, n_classes, hidden=self.hidden, n_layers=self.n_layers,
                         dropout=self.dropout)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GCNTrainResult:
    model: "GCN"
    log: list[dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0


class _SymmetricSpMM(torch.autograd.Function):
    """y = A @ h for a symmetric sparse A; the gradient for h is A @ grad (no gradient for A)."""

    @staticmethod
    def forward(ctx: Any, adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ctx.adj = adj
        return adj @ h

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, ctx.adj @ grad


class GCN(nn.Module):
    """``forward(adj, features)`` -> logits of every node [n_nodes, n_classes]."""

    def __init__(self, config: GCNConfig) -> None:
        super().__init__()
        self.config = config
        widths = config.widths()
        for index in range(1, config.n_layers + 1):
            linear = nn.Linear(widths[index - 1], widths[index])
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.add_module(f"conv{index}", linear)
            if index < config.n_layers:
                self.add_module(f"bn{index}", nn.BatchNorm1d(widths[index], eps=config.bn_eps,
                                                             momentum=config.bn_momentum))

    def forward(self, adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h = features
        k = self.config.n_layers
        for index in range(1, k + 1):
            conv = getattr(self, f"conv{index}")
            h = _SymmetricSpMM.apply(adj, h @ conv.weight.T) + conv.bias
            if index < k:
                h = F.relu(getattr(self, f"bn{index}")(h))
                h = F.dropout(h, p=self.config.dropout, training=self.training)
        return h


def graph_tensors(graph: Mapping[str, np.ndarray], device: torch.device,
                  dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """The CSR adjacency and the features of graph.npz as torch tensors on ``device``."""
    indptr, indices = np.asarray(graph["indptr"], np.int64), np.asarray(graph["indices"], np.int64)
    n = indptr.size - 1
    adj = torch.sparse_csr_tensor(torch.from_numpy(indptr), torch.from_numpy(indices),
                                  torch.from_numpy(np.asarray(graph["weight"])).to(dtype), size=(n, n),
                                  check_invariants=True)
    features = torch.from_numpy(np.ascontiguousarray(graph["features"])).to(dtype)
    return adj.to(device), features.to(device)


def check_symmetric(graph: Mapping[str, np.ndarray]) -> None:
    """Raise unless the stored adjacency equals its transpose (entries and weights), as the backward assumes."""
    indptr, indices, weight = (np.asarray(graph[key]) for key in ("indptr", "indices", "weight"))
    n = indptr.size - 1
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    forward_keys = rows * n + indices
    order = np.argsort(indices * n + rows, kind="stable")
    if not (np.array_equal(forward_keys, (indices * n + rows)[order]) and np.array_equal(weight, weight[order])):
        raise ValueError("graph adjacency is not symmetric")


def export_gcn_params(model: GCN) -> dict[str, np.ndarray]:
    """Parameters and BatchNorm running statistics as float32 numpy arrays, PyTorch names."""
    params: dict[str, np.ndarray] = {}
    for index in range(1, model.config.n_layers + 1):
        conv = getattr(model, f"conv{index}")
        params[f"conv{index}.weight"] = conv.weight.detach().cpu().numpy().astype(np.float32).copy()
        params[f"conv{index}.bias"] = conv.bias.detach().cpu().numpy().astype(np.float32).copy()
        if index < model.config.n_layers:
            norm = getattr(model, f"bn{index}")
            for part in BN_PARTS:
                params[f"bn{index}.{part}"] = getattr(norm, part).detach().cpu().numpy().astype(np.float32).copy()
    return params


def load_gcn_params(model: GCN, params: Mapping[str, np.ndarray]) -> GCN:
    """Copy ``params`` (``export_gcn_params`` names, exactly) into ``model``."""
    expected = set(export_gcn_params(model))
    if set(params) != expected:
        raise KeyError(f"parameter names differ: missing {sorted(expected - set(params))}, "
                       f"extra {sorted(set(params) - expected)}")
    with torch.no_grad():
        for key, value in params.items():
            module, part = key.rsplit(".", 1)
            target = getattr(getattr(model, module), part)
            if tuple(target.shape) != tuple(np.shape(value)):
                raise ValueError(f"{key}: shape {np.shape(value)} != {tuple(target.shape)}")
            target.copy_(torch.from_numpy(np.asarray(value, dtype=np.float32)))
    return model


def build_gcn(config: GCNConfig, params: Mapping[str, np.ndarray] | None = None,
              dtype: torch.dtype = torch.float32) -> GCN:
    """An eval-mode GCN with ``params`` loaded, in ``dtype``."""
    model = GCN(config)
    if params is not None:
        load_gcn_params(model, params)
    return model.to(dtype).eval()


def logits_all(model: GCN, adj: torch.Tensor, features: torch.Tensor) -> np.ndarray:
    """Eval logits of every node as a float64 numpy array."""
    model.eval()
    with torch.no_grad():
        return model(adj, features).detach().cpu().numpy().astype(np.float64)


def layer_norms(params: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Frobenius norm of every conv weight and bias (the per-layer norms reported for each student)."""
    return {key: float(np.linalg.norm(np.asarray(value, dtype=np.float64)))
            for key, value in params.items() if key.startswith("conv")}


def train_gcn(
    config: GCNTrainConfig,
    seed: int,
    graph: Mapping[str, np.ndarray],
    train_x: np.ndarray,
    train_y: np.ndarray,
    n_classes: int,
    device: torch.device,
    monitor: Callable[[GCN], Mapping[str, Any]] | None = None,
) -> GCNTrainResult:
    """Train one GCN on the labelled nodes ``train_x`` (module docstring); returns the last-epoch model."""
    check_symmetric(graph)
    n_nodes, n_features = np.asarray(graph["features"]).shape
    if np.asarray(train_y).min() < 0 or np.asarray(train_y).max() >= n_classes:
        raise ValueError(f"labels must lie in 0 .. {n_classes - 1}")
    torch.manual_seed(seed)
    model = GCN(config.model_config(n_nodes, n_features, n_classes)).to(device)
    adj, features = graph_tensors(graph, device)
    index = torch.from_numpy(np.asarray(train_x, np.int64)).to(device)
    labels = torch.from_numpy(np.asarray(train_y, np.int64)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    result = GCNTrainResult(model=model)
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(adj, features)[index], labels)
        loss.backward()
        optimizer.step()
        if epoch % config.log_every == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                train_acc = float((model(adj, features)[index].argmax(1) == labels).float().mean())
            entry = {"epoch": epoch, "loss": float(loss.detach()), "train_acc": train_acc}
            if monitor is not None:
                entry.update(monitor(model))
            result.log.append(entry)
    model.eval()
    result.wall_time_s = time.perf_counter() - started
    return result


def _shape(params: Mapping[str, np.ndarray], key: str) -> list[int]:
    if key not in params:
        raise KeyError(f"missing parameter {key!r}")
    return [int(size) for size in np.shape(params[key])]


def build_gcn_model_spec(model_id: str, config: GCNConfig, class_names: list[str],
                         params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Public ``model_spec.json`` of a ``gcn`` student (schema weightcode.model_spec.v0), checked for forbidden terms."""
    from weightcode.models.spec import SCHEMA, check_public

    if len(class_names) != config.n_classes:
        raise ValueError("class names do not match the config")
    layers: list[dict[str, Any]] = []
    for index in range(1, config.n_layers + 1):
        layers.append({
            "name": f"conv{index}",
            "op": "graph_conv",
            "weight": f"conv{index}.weight",
            "bias": f"conv{index}.bias",
            "shape_w": _shape(params, f"conv{index}.weight"),
            "shape_b": _shape(params, f"conv{index}.bias"),
            "computes": "h = A @ (h @ weight.T) + bias, A the normalized adjacency of graph.npz",
        })
        if index < config.n_layers:
            layers.append({
                "name": f"bn{index}",
                "op": "batchnorm_eval",
                "tensors": {part: f"bn{index}.{part}" for part in BN_PARTS},
                "shape": _shape(params, f"bn{index}.weight"),
                "eps": config.bn_eps,
                "computes": "h = (h - running_mean) / sqrt(running_var + eps) * weight + bias, per channel",
            })
            layers.append({"name": f"relu{index}", "op": "relu"})
    spec = {
        "schema": SCHEMA,
        "model_id": model_id,
        "input": {
            "kind": "node_ids",
            "shape": [],
            "dtype": "int64",
            "n_nodes": config.n_nodes,
            "n_features": config.n_features,
            "graph": "graph.npz: CSR indptr [n_nodes + 1], indices [nnz], weight [nnz] of the normalized adjacency "
                     "(symmetric, one self loop per node), features [n_nodes, n_features]; no labels",
        },
        "input_files": ["graph.npz"],
        "classes": [str(name) for name in class_names],
        "parameter_layout": "PyTorch: linear weight [out, in]; BatchNorm weight, bias, running_mean, running_var "
                            "[width]; all float32",
        "n_parameters": int(sum(np.asarray(value).size for value in params.values())),
        "layers": layers,
        "output": "logits [N, n_classes] for the requested nodes; the prediction is the argmax over classes",
        "reference_forward": "forward.py: forward(params, x_int64 node ids [N]) -> float32 logits [N, n_classes]; it "
                             "runs the whole graph from graph.npz in its own directory",
    }
    check_public(spec)
    return spec
