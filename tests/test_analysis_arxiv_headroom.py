"""The E6 headroom statistics: paired differences, the mid-rank AUC, the L2 logistic fit and the GCN activations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from weightcode.analysis import arxiv_headroom as H
from weightcode.data.arxiv import build_graph
from weightcode.models import gcn as G
from weightcode.models.export import load_forward_no_cache
from weightcode.models.forward_gen import render_gcn_forward, write_forward


def test_paired_difference_counts_and_se() -> None:
    a = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
    b = np.array([1, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    out = H.paired_difference(a, b)
    d = b.astype(float) - a.astype(float)
    assert out["diff_points"] == 100 * d.mean()
    assert np.isclose(out["paired_se_points"], 100 * d.std(ddof=1) / np.sqrt(8))
    assert (out["only_b_right"], out["only_a_right"], out["discordant_fraction"]) == (2, 1, 3 / 8)


def test_auc_matches_pairwise_count_with_ties() -> None:
    rng = np.random.default_rng(0)
    pos, neg = rng.integers(0, 5, 40).astype(float), rng.integers(0, 4, 30).astype(float)
    pairwise = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    assert np.isclose(H.auc(pos, neg), pairwise)
    assert H.auc(np.array([2.0]), np.array([1.0])) == 1.0


def test_logistic_fit_is_a_stationary_point_of_the_penalized_loss() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((80, 5))
    y = (x[:, 0] + 0.5 * rng.standard_normal(80) > 0).astype(float)
    w, b = H.fit_logistic(x, y, l2=1.0)
    p = 1 / (1 + np.exp(-(x @ w + b)))
    assert np.allclose(x.T @ (p - y) + w, 0, atol=1e-8) and abs(np.sum(p - y)) < 1e-8
    assert H.auc(x[y == 1] @ w, x[y == 0] @ w) > 0.8


def test_activations_rebuild_the_forward_logits(tmp_path: Path) -> None:
    n, f, k = 50, 6, 4
    rng = np.random.default_rng(2)
    graph = build_graph(rng.integers(0, n, size=(120, 2)), n, rng.standard_normal((n, f)).astype(np.float32))
    write_forward(tmp_path / "forward.py", render_gcn_forward(n, f, k, 1e-5))
    np.savez(tmp_path / "graph.npz", **graph)
    forward, _ = load_forward_no_cache(tmp_path / "forward.py")
    torch.manual_seed(0)
    model = G.GCN(G.GCNConfig(n, f, k, hidden=8))
    for name in ("bn1", "bn2"):
        bn = getattr(model, name)
        bn.running_mean.uniform_(-0.5, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
    params = G.export_gcn_params(model)
    acts = H.activations(forward, params)
    assert acts["z"].shape == (n, 8)
    expected = forward(dict(params), np.arange(n)).astype(np.float64)
    assert np.allclose(acts["logits"], expected, atol=1e-5)
