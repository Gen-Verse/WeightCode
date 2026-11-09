"""PREREG F6b's alignment (``weightcode.analysis.align``) on a ViT-Lite-7/4 student: a randomly permuted copy is recovered
exactly, an edit mapped through the alignment does the same thing in both coordinate systems, and the validity gate
passes for the symmetry and fails when a tensor is left unpermuted."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from test_e11_refs import MEAN, STD, _images, _student

from weightcode.analysis import align as A
from weightcode.analysis import nullspace as N
from weightcode.models import vitlite as V
from weightcode.models.forward_gen import render_vitlite_forward

HIDDEN_BLOCKS = (2, 4)


def _module(tmp_path: Path):
    path = tmp_path / "forward.py"
    path.write_text(render_vitlite_forward(MEAN, STD, 10, V.LN_EPS))
    return N.forward_module(path)


def _random_alignment(seed: int) -> A.Alignment:
    rng = np.random.default_rng(seed)
    return A.Alignment(res=rng.permutation(256), hid={b: rng.permutation(512) for b in HIDDEN_BLOCKS})


def test_hungarian_recovers_a_permutation_and_is_optimal() -> None:
    rng = np.random.default_rng(0)
    perm = rng.permutation(40)
    score = rng.normal(size=(40, 40)) * 0.1
    score[np.arange(40), perm] += 1.0
    assert np.array_equal(A.hungarian_max(score), perm)
    for n in (2, 3, 5, 7):  # optimal against brute force
        for _ in range(5):
            m = rng.normal(size=(n, n))
            got = A.hungarian_max(m)
            assert sorted(got) == list(range(n))
            best = max(m[np.arange(n), list(q)].sum() for q in itertools.permutations(range(n)))
            assert np.isclose(m[np.arange(n), got].sum(), best)


def test_hungarian_agrees_with_scipy_where_it_is_installed() -> None:
    optimize = pytest.importorskip("scipy.optimize", reason="scipy is not installed")
    m = np.random.default_rng(1).normal(size=(64, 64))
    got = A.hungarian_max(m)
    assert sorted(got) == list(range(64))
    rows, cols = optimize.linear_sum_assignment(-m)
    assert np.isclose(m[rows, cols].sum(), m[np.arange(64), got].sum())


def test_every_tensor_of_the_architecture_is_classified() -> None:
    params = _student(0)
    for name in params:
        specs = A.axis_specs(name)
        for axis, kind, block in specs:
            assert kind in ("res", "hid") and (block is None) == (kind == "res")
            assert params[name].shape[axis] == (256 if kind == "res" else 512)
    with pytest.raises(A.AlignError):
        A.axis_specs("classifier.mystery.weight")


def test_a_permuted_copy_is_recovered_exactly_and_an_edit_maps_across(tmp_path: Path) -> None:
    module = _module(tmp_path)
    source = _student(1)
    truth = _random_alignment(5)
    target = A.to_target(source, truth)  # the same network written in other unit coordinates
    x = _images(48, 7)
    assert np.allclose(module.forward(dict(source), x), module.forward(dict(target), x), atol=1e-5)  # a symmetry
    found = A.fit_alignment(module, source, target, x, HIDDEN_BLOCKS, chunk=16)
    assert np.array_equal(found.res, truth.res)
    assert all(np.array_equal(found.hid[b], truth.hid[b]) for b in HIDDEN_BLOCKS)
    assert found.stats["residual_sum_matched"] > 8 * 256 * 0.9999
    assert found.stats["residual_places"]["block0_input"]["identity"] < 0.5  # the identity matching is poor
    assert found.stats["hidden"]["block2"]["matched"] > 0.9999
    rng = np.random.default_rng(3)
    delta = {
        "classifier.blocks.2.linear1.weight": 0.05 * rng.normal(size=(512, 256)),
        "classifier.blocks.2.linear1.bias": 0.05 * rng.normal(size=512),
        "classifier.blocks.2.linear2.weight": 0.05 * rng.normal(size=(256, 512)),
        "classifier.blocks.2.linear2.bias": 0.05 * rng.normal(size=256),
        "tokenizer.conv_layers.0.0.weight": 0.05 * rng.normal(size=(256, 3, 4, 4)),
        "tokenizer.conv_layers.0.0.bias": 0.05 * rng.normal(size=256),
    }

    def edited(params, d):
        out = dict(params)
        for k, v in d.items():
            out[k] = (np.asarray(params[k], np.float64) + v).astype(np.float32)
        return out

    mapped = A.map_delta(delta, found)
    assert {k: v.shape for k, v in mapped.items()} == {k: v.shape for k, v in delta.items()}
    source_new, target_new = edited(source, delta), edited(target, mapped)
    back = A.to_source(target_new, found)
    assert all(np.array_equal(back[k], source_new[k]) for k in source_new)  # the same edit in both coordinate systems
    assert np.allclose(module.forward(dict(source_new), x), module.forward(dict(target_new), x), atol=1e-5)
    unmapped = edited(target, delta)  # F6: the copied delta lands on the wrong units
    assert not np.allclose(module.forward(dict(source_new), x), module.forward(dict(unmapped), x), atol=1e-3)
    with pytest.raises(A.AlignError, match="no permutation"):
        A.map_delta({"classifier.blocks.3.linear1.weight": np.zeros((512, 256))}, found)


def test_the_gate_passes_for_the_symmetry_and_fails_when_a_tensor_is_not_permuted(tmp_path: Path, monkeypatch) -> None:
    module = _module(tmp_path)
    target = _student(2)  # any student: the permutation is a symmetry whatever the matching
    alignment = _random_alignment(9)
    images = {"condition": _images(40, 1), "clean": _images(40, 2)}
    gate = A.symmetry_gate(module, target, alignment, images)
    assert gate["passed"] and gate["roundtrip_exact"]
    assert all(s["argmax_disagreements"] == 0 for s in gate["sets"].values())
    real = A.axis_specs

    def leaky(name):  # linear2's residual side left out: the permuted network is no longer the same function
        specs = real(name)
        return [s for s in specs if not (name.endswith("linear2.weight") and s[1] == "res")]

    monkeypatch.setattr(A, "axis_specs", leaky)
    broken = A.symmetry_gate(module, target, alignment, images)
    assert not broken["passed"] and broken["roundtrip_exact"]  # still a clean round trip; the function changed
