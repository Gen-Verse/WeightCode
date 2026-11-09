"""The E7 Correct-and-Smooth instrument (``weightcode.analysis.graphnew_cs``) on a small path graph."""

from __future__ import annotations

import numpy as np
import torch

from weightcode.analysis import graphnew_cs as C


def _path(n: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [sorted([i] + [j for j in (i - 1, i + 1) if 0 <= j < n]) for i in range(n)]  # with self loops
    return np.cumsum([0] + [len(r) for r in rows]), np.concatenate(rows)


def test_normalized_adjacencies_drop_self_loops() -> None:
    indptr, indices = _path(4)
    adjs = C.normalized_adjacencies(indptr, indices, 4)
    deg = np.array([1, 2, 2, 1], dtype=float)
    a = np.zeros((4, 4))
    for i in range(3):
        a[i, i + 1] = a[i + 1, i] = 1
    assert np.allclose(adjs["DAD"].to_dense().numpy(), a / np.sqrt(np.outer(deg, deg)))
    assert np.allclose(adjs["DA"].to_dense().numpy(), a / deg[:, None])
    assert np.allclose(adjs["AD"].to_dense().numpy(), a / deg[None, :])


def test_propagations_reach_unlabelled_nodes_and_keep_seeds() -> None:
    n, c = 6, 3
    indptr, indices = _path(n)
    logits = np.zeros((n, c))
    logits[:, 0] = 2.0  # the base predicts class 0 everywhere
    seeds, labels = np.array([0, 5]), np.array([1, 2])
    outputs = C.instrument_outputs(indptr, indices, logits, seeds, labels)
    lp = outputs["lp"].numpy()
    assert lp.argmax(1)[0] == 1 and lp.argmax(1)[5] == 2 and C.zero_rows(outputs["lp"], np.arange(n)) == 0
    for name in ("cs_autoscale", "smooth_only"):
        out = outputs[name].numpy()
        assert out.shape == (n, c) and out.min() >= 0 and out.max() <= 1
        assert out[0].argmax() == 1 and out[5].argmax() == 2
    smooth = C.smooth_only(C.normalized_adjacencies(indptr, indices, n), C.softmax64(logits), seeds, labels,
                           {"A": "DAD", "alpha": 0.0, "num_propagations": 3})
    expected = torch.softmax(torch.from_numpy(logits), 1)
    expected[0] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    expected[5] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    assert torch.allclose(smooth, expected)  # alpha 0 leaves the seeded base output unchanged
