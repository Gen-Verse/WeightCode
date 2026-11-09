"""PREREG G1's family and its pure helpers (``scripts/e11/g1_search.py``); the builders run on the real instances."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "e11" / "g1_search.py"


def _g1():
    spec = importlib.util.spec_from_file_location("g1_search_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_family_is_the_amended_one() -> None:
    g1 = _g1()
    members = g1.family(7)
    names = [m.name for m in members]
    assert len(names) == len(set(names)) == 151  # 4 tokenizer + 7 blocks x 3 token kinds x (3 c1 + 4 c3)
    tok = [m for m in members if m.site == "tok"]
    assert sorted((m.corr, m.r) for m in tok) == [("c1", None), ("c2", None), ("c3", 4), ("c3", 8)]
    blocks = [m for m in members if m.site != "tok"]
    assert all(m.corr in ("c1", "c3") for m in blocks)  # c2 is tokenizer only
    assert {m.gate for m in blocks if m.corr == "c1"} == {0, 2, 6}
    assert {m.r for m in blocks if m.corr == "c3"} == {4, 8}
    assert {m.tokens for m in blocks} == {"patch", "cls", "both"}


def test_covariance_map_matches_the_variance_on_the_top_directions() -> None:
    g1 = _g1()
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2000, 6)) * np.array([3.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    b = rng.normal(size=(2000, 6)) * np.array([1.0, 1.0, 2.0, 1.0, 1.0, 0.5])
    dm = g1.cov_difference_map(np.cov(a.T, bias=True), np.cov(b.T, bias=True), 2)
    mapped = (a - a.mean(0)) @ (np.eye(6) + dm).T
    c_old, c_new = np.cov(b.T, bias=True), np.cov(a.T, bias=True)
    values, vectors = np.linalg.eigh(c_old - c_new)
    for i in np.argsort(-np.abs(values))[:2]:
        u = vectors[:, i]
        assert np.isclose(np.var(mapped @ u), u @ c_old @ u, rtol=0.05)
    assert np.linalg.matrix_rank(dm) == 2


def test_rank_key_orders_by_proxy_then_changed_entries_then_member_order() -> None:
    g1 = _g1()

    def rec(score, changed, order, strength=1.0):
        return {"proxy": {"score": score}, "n_changed": changed, "order": order, "strength": strength}

    rows = [rec(5, 10, 3), rec(7, 99, 9), rec(5, 10, 1), rec(5, 4, 8)]
    assert [(r["proxy"]["score"], r["n_changed"], r["order"]) for r in sorted(rows, key=g1.rank_key)] == [
        (7, 99, 9), (5, 4, 8), (5, 10, 1), (5, 10, 3)]


def test_select_takes_the_named_tokens() -> None:
    g1 = _g1()
    a = np.arange(2 * 65 * 3, dtype=float).reshape(2, 65, 3)
    assert g1.select(a, "patch").shape == (128, 3) and g1.select(a, "cls").shape == (2, 3)
    assert g1.select(a, "both").shape == (130, 3)
    assert np.array_equal(g1.select(a, "cls"), a[:, 0])
