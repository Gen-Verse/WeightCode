"""GCN students of ogbn-arxiv: the numpy forward.py against torch, the symmetric aggregation, training and params.

A small random graph built by ``weightcode.data.arxiv.build_graph`` stands in for the real one; nothing reads the store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.data.arxiv import build_graph
from weightcode.models import gcn as G
from weightcode.models.export import ParityError, load_forward_no_cache
from weightcode.models.forward_gen import render_gcn_forward, write_forward
from weightcode.models.parity import FLOAT32_STEP, check_logit_parity

N, F_IN, K = 60, 7, 5


def _graph(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, N, size=(150, 2))
    return build_graph(edges, N, rng.standard_normal((N, F_IN)).astype(np.float32))


def _forward(tmp_path: Path, graph: dict[str, np.ndarray], chunk: int | None = None):
    write_forward(tmp_path / "forward.py", render_gcn_forward(N, F_IN, K, 1e-5))
    np.savez(tmp_path / "graph.npz", **graph)
    forward, _ = load_forward_no_cache(tmp_path / "forward.py")
    if chunk is not None:
        forward.__globals__["CHUNK_NNZ"] = chunk
    return forward


def _params(seed: int = 1) -> tuple[G.GCNConfig, dict[str, np.ndarray]]:
    config = G.GCNConfig(N, F_IN, K, hidden=16)
    torch.manual_seed(seed)
    model = G.GCN(config)
    with torch.no_grad():
        for index in (1, 2):
            norm = getattr(model, f"bn{index}")
            norm.running_mean.uniform_(-1, 1)
            norm.running_var.uniform_(0.5, 2)
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
    return config, G.export_gcn_params(model)


@pytest.mark.parametrize("chunk", [None, 7])
def test_forward_matches_torch(tmp_path: Path, chunk: int | None) -> None:
    graph = _graph()
    G.check_symmetric(graph)
    forward = _forward(tmp_path, graph, chunk)
    config, params = _params()
    mine = forward(params, np.arange(N))
    assert mine.dtype == np.float32 and mine.shape == (N, K)
    exact, float32 = (G.logits_all(G.build_gcn(config, params, dtype), *G.graph_tensors(graph, torch.device("cpu"), dtype))
                      for dtype in (torch.float64, torch.float32))
    result = check_logit_parity(mine, exact, float32)
    assert result["float64_max_rel_diff"] <= FLOAT32_STEP
    ids = np.array([5, 0, 59, 5])
    assert np.array_equal(forward(params, ids), mine[ids]) and forward(params, ids[:0]).shape == (0, K)
    with pytest.raises(ParityError, match="float64"):
        check_logit_parity(mine * np.float32(1 + 8 * FLOAT32_STEP) + np.float32(1e-3), exact, float32)


def test_forward_refuses_bad_inputs(tmp_path: Path) -> None:
    forward = _forward(tmp_path, _graph())
    _, params = _params()
    with pytest.raises(ValueError):
        forward(params, np.array([N]))
    with pytest.raises(KeyError):
        forward({**params, "extra.weight": params["conv1.weight"]}, np.arange(3))
    with pytest.raises(KeyError):
        forward({k: v for k, v in params.items() if not k.startswith("bn2")}, np.arange(3))


def test_backward_of_the_aggregation_is_the_transpose() -> None:
    graph = _graph()
    adj, _ = G.graph_tensors(graph, torch.device("cpu"), torch.float64)
    dense = adj.to_dense()
    assert torch.equal(dense, dense.T)
    h = torch.randn(N, 3, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(lambda t: G._SymmetricSpMM.apply(adj, t), (h,))
    asym = {**graph, "weight": graph["weight"].copy()}
    asym["weight"][1] += 0.5
    with pytest.raises(ValueError, match="symmetric"):
        G.check_symmetric(asym)


def test_training_learns_and_params_round_trip(tmp_path: Path) -> None:
    graph = _graph()
    rng = np.random.default_rng(3)
    y = rng.integers(0, K, size=N)
    config = G.GCNTrainConfig(hidden=16, epochs=60, log_every=20)
    first = G.train_gcn(config, 0, graph, np.arange(40), y[:40], K, torch.device("cpu"))
    again = G.train_gcn(config, 0, graph, np.arange(40), y[:40], K, torch.device("cpu"))
    params = G.export_gcn_params(first.model)
    assert all(np.array_equal(params[k], G.export_gcn_params(again.model)[k]) for k in params)
    assert first.log[-1]["train_acc"] > first.log[0]["train_acc"] or first.log[-1]["train_acc"] == 1.0
    assert set(params) == {f"conv{i}.{p}" for i in (1, 2, 3) for p in ("weight", "bias")} | {
        f"bn{i}.{p}" for i in (1, 2) for p in G.BN_PARTS}
    rebuilt = G.build_gcn(first.model.config, params)
    assert all(np.array_equal(v, G.export_gcn_params(rebuilt)[k]) for k, v in params.items())
    spec = G.build_gcn_model_spec("m0", first.model.config, [f"c{i}" for i in range(K)], params)
    assert spec["input_files"] == ["graph.npz"] and spec["n_parameters"] == sum(v.size for v in params.values())
    forward = _forward(tmp_path, graph)
    exact = G.logits_all(rebuilt.double(), *G.graph_tensors(graph, torch.device("cpu"), torch.float64))
    assert np.max(np.abs(forward(params, np.arange(N)) - exact)) <= FLOAT32_STEP * np.max(np.abs(exact)) + 1e-6
