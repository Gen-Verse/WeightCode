"""Unit tests of the search baselines' pure pieces: the candidate-to-query rule, edit limits, the Pareto front,
stratified positive sampling, the Arachne fitness and bounds, PSO boundary handling and the evidence split."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pytest

from weightcode.baselines.search.arachne import arachne_fitness, layer_bounds, pareto_front, sample_positives
from weightcode.baselines.search.care import periodic_bounds
from weightcode.baselines.search.evidence import CountedForward, build_evidence_sets, evidence_readout
from weightcode.baselines.search.limits import (
    EditLimitError,
    EditLimits,
    check_limits,
    edit_stats,
    limit_violations,
    shrink_to_radius,
)
from weightcode.baselines.search.netspec import cross_entropy
from weightcode.baselines.search.runner import looser_than
from weightcode.baselines.search.selection import Trajectory, select_for_query
from weightcode.patch import Limits

# Explicit limits for the arithmetic tests, independent of the executor's current values.
OLD_STYLE = EditLimits(max_changed_frac=0.05, max_edit_frac=None, max_rel_frob=0.5, rel_frob_min_ndim=1)


def test_select_for_query_known_cases() -> None:
    assert select_for_query(0, 8) == []
    assert select_for_query(5, 0) == []
    assert select_for_query(5, 8) == [0, 1, 2, 3, 4]
    assert select_for_query(8, 8) == list(range(8))
    assert select_for_query(9, 1) == [8]
    assert select_for_query(10, 4) == [0, 3, 6, 9]
    assert select_for_query(20, 8) == [0, 3, 5, 8, 11, 14, 16, 19]


def test_select_for_query_invariants() -> None:
    for n_entries in range(1, 80):
        for budget in range(1, 10):
            positions = select_for_query(n_entries, budget)
            assert len(positions) == min(n_entries, budget)
            assert positions == sorted(set(positions))
            assert positions[-1] == n_entries - 1
            if len(positions) > 1:
                assert positions[0] == 0


def test_trajectory_records_changes_only() -> None:
    trajectory = Trajectory()
    vector = np.array([1.0, 2.0])
    trajectory.record(0, vector, 0.5)
    vector[0] = 1.0  # unchanged value: no new entry
    trajectory.record(1, vector, 0.5)
    trajectory.record(2, np.array([1.0, 3.0]), 0.7)
    vector[1] = 9.0  # the stored copy must not follow the caller's array
    assert [snap.step for snap in trajectory.snapshots] == [0, 2]
    assert trajectory.snapshots[0].vector.tolist() == [1.0, 2.0]


def _params() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "a": rng.normal(size=(10, 10)).astype(np.float32),
        "b": rng.normal(size=10).astype(np.float32),
        "z": np.zeros(5, dtype=np.float32),
    }


def test_limits_are_read_from_the_executor() -> None:
    executor = Limits()
    limits = EditLimits.from_executor()
    assert limits == EditLimits.from_executor(executor)
    assert limits.max_changed_frac == getattr(executor, "max_changed_frac", None)
    assert limits.max_edit_frac == getattr(executor, "max_edit_frac", None)
    assert limits.max_rel_frob == getattr(executor, "max_rel_frob", None)
    assert limits.rel_frob_min_ndim == getattr(executor, "rel_frob_min_ndim", 1)
    assert limits.projection_margin == 1e-3
    switched = EditLimits.from_executor(Limits(max_changed_frac=None, max_edit_frac=0.02, max_rel_frob=None))
    assert (switched.changed_frac_cap(), switched.max_rel_frob) == (0.02, None)
    assert switched.frob_radius(np.ones((2, 2), np.float32)) == math.inf

    class Before:  # an executor Limits without the switched fields
        max_changed_frac = 0.05
        max_rel_frob = 0.5

    before = EditLimits.from_executor(Before())
    assert (before.max_edit_frac, before.rel_frob_min_ndim, before.changed_frac_cap()) == (None, 1, 0.05)


def test_limits_from_task_and_looser_than() -> None:
    switched = EditLimits.from_task(asdict(Limits(max_changed_frac=None, max_edit_frac=0.05, rel_frob_min_ndim=2)))
    assert (switched.changed_frac_cap(), switched.max_rel_frob, switched.rel_frob_min_ndim) == (0.05, 0.5, 2)
    implicit = EditLimits.from_task({"max_edit_frac": 0.05, "max_rel_frob": 0.5})
    assert implicit.rel_frob_min_ndim == 2  # the switch exempts 1-D tensors
    pre_switch = EditLimits.from_task({"max_changed_frac": 0.05, "max_rel_frob": 0.5})
    assert (pre_switch.changed_frac_cap(), pre_switch.rel_frob_min_ndim) == (0.05, 1)
    assert looser_than(switched, switched) == []
    assert looser_than(OLD_STYLE, switched) == []  # stricter on 1-D tensors
    assert looser_than(switched, pre_switch) == ["rel_frob_min_ndim 2 vs stated 1"]
    unlimited = EditLimits(max_changed_frac=None, max_edit_frac=None, max_rel_frob=None)
    assert len(looser_than(unlimited, switched)) == 2


def test_limits_changed_fraction() -> None:
    base = _params()  # 115 entries, cap floor(0.05 * 115) = 5
    for limits in (OLD_STYLE, EditLimits(max_changed_frac=None, max_edit_frac=0.05, max_rel_frob=0.5)):
        assert limits.max_changed(base) == 5
        child = {key: value.copy() for key, value in base.items()}
        child["a"].reshape(-1)[:5] += np.float32(0.01)
        stats = check_limits(base, child, limits)
        assert (stats.n_changed, stats.n_total) == (5, 115)
        child["a"].reshape(-1)[5] += np.float32(0.01)
        with pytest.raises(EditLimitError, match="changed fraction"):
            check_limits(base, child, limits)
    both = EditLimits(max_changed_frac=0.1, max_edit_frac=0.05, max_rel_frob=None)
    assert both.max_changed(base) == 5
    assert EditLimits(max_changed_frac=None, max_edit_frac=None, max_rel_frob=None).max_changed(base) == 115


def test_limits_relative_norm_and_zero_tensor() -> None:
    base = _params()
    child = {key: value.copy() for key, value in base.items()}
    child["b"][0] += np.float32(0.6 * np.linalg.norm(base["b"].astype(np.float64)))
    with pytest.raises(EditLimitError, match="relative Frobenius"):
        check_limits(base, child, OLD_STYLE)
    rank_two = EditLimits(max_changed_frac=None, max_edit_frac=0.05, max_rel_frob=0.5, rel_frob_min_ndim=2)
    assert check_limits(base, child, rank_two).rel_frob["b"] > 0.5  # 1-D tensors are exempt
    big = {key: value.copy() for key, value in base.items()}
    big["a"][0, 0] += np.float32(0.6 * np.linalg.norm(base["a"].astype(np.float64)))
    assert limit_violations(edit_stats(base, big), rank_two) != []
    zero = {key: value.copy() for key, value in base.items()}
    zero["z"][0] = np.float32(1e-3)
    assert math.isinf(edit_stats(base, zero).rel_frob["z"])
    with pytest.raises(EditLimitError, match="z"):
        check_limits(base, zero, OLD_STYLE)


def test_limits_radius_and_shrink() -> None:
    limits = OLD_STYLE
    tensor = np.full((2, 2), 2.0, dtype=np.float32)  # norm 4
    assert limits.frob_radius(tensor) == pytest.approx(0.999 * 0.5 * 4.0)
    rank_two = EditLimits(max_changed_frac=None, max_edit_frac=0.05, max_rel_frob=0.5, rel_frob_min_ndim=2)
    assert rank_two.frob_radius(np.ones(3, np.float32)) == math.inf
    assert rank_two.frob_radius(tensor) == pytest.approx(0.999 * 0.5 * 4.0)
    delta = np.array([3.0, 4.0])
    assert np.allclose(shrink_to_radius(delta, 1.0), [0.6, 0.8])
    assert np.array_equal(shrink_to_radius(delta, 10.0), delta)
    assert np.array_equal(shrink_to_radius(delta, 0.0), [0.0, 0.0])


def _brute_front(points: np.ndarray) -> list[int]:
    keep = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if j != i and ((np.all(q >= p) and np.any(q > p)) or (np.all(q == p) and j < i)):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def test_pareto_front_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    for _ in range(40):
        points = rng.integers(0, 6, size=(40, 2)).astype(np.float64)  # many ties and duplicates
        front = pareto_front(points)
        assert sorted(front.tolist()) == _brute_front(points)
        assert np.all(np.diff(points[front, 0]) <= 0)
    continuous = rng.normal(size=(200, 2))
    assert sorted(pareto_front(continuous).tolist()) == _brute_front(continuous)
    with pytest.raises(ValueError):
        pareto_front(np.array([[0.0, np.nan]]))


def test_sample_positives_is_stratified_and_seeded() -> None:
    y = np.array([0] * 20 + [1] * 10 + [2] * 2)
    chosen = sample_positives(y, 16, np.random.default_rng(0))
    assert np.bincount(y[chosen], minlength=3).tolist() == [10, 5, 1]
    assert len(set(chosen.tolist())) == len(chosen)
    assert np.array_equal(chosen, sample_positives(y, 16, np.random.default_rng(0)))
    assert np.array_equal(sample_positives(y, 40, np.random.default_rng(0)), np.arange(32))
    rare = np.array([0] * 30 + [1])
    assert np.bincount(rare[sample_positives(rare, 4, np.random.default_rng(0))], minlength=2).tolist() == [4, 1]


def test_arachne_fitness_matches_eq_3_and_4() -> None:
    logits = np.array([[2.0, 0.0], [0.0, 1.0], [3.0, 0.0], [0.0, 2.0]])
    y = np.zeros(4, dtype=np.int64)
    ce = cross_entropy(logits, y)
    expected = (1.0 + 1.0 / (ce[3] + 1.0)) + 10.0 * (1.0 + 1.0 / (ce[1] + 1.0))
    assert arachne_fitness(logits, y, n_neg=2, alpha=10.0) == pytest.approx(expected)
    assert ce[1] == pytest.approx(np.log1p(np.e))


def test_layer_bounds_follow_the_code() -> None:
    assert layer_bounds(np.array([-1.0, 2.0])) == (-2.0, 4.0)
    assert layer_bounds(np.array([0.5, 2.0])) == (0.25, 4.0)
    assert layer_bounds(np.array([-3.0, -1.0])) == (-6.0, -0.5)


def test_periodic_bounds_matches_pyswarms_formula() -> None:
    x = np.array([-1.25, -1.0, 0.3, 1.0, 1.5, 3.5, -3.5])
    assert np.allclose(periodic_bounds(x, -1.0, 1.0), [0.75, -1.0, 0.3, 1.0, -0.5, -0.5, 0.5])


def _pixel_forward(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Predicts class (first pixel mod 3)."""
    cls = x.reshape(len(x), -1)[:, 0].astype(np.int64) % 3
    logits = np.zeros((len(x), 3), dtype=np.float32)
    logits[np.arange(len(x)), cls] = 1.0
    return logits


def _v1_evidence() -> dict[str, np.ndarray]:
    """Failures of mixed classes (true 0, 1, 2, 0, predicted 2, 2, 0, 1) and six random images, two of them wrong."""
    x_fail = np.zeros((4, 1, 2, 2), dtype=np.uint8)
    x_fail[:, 0, 0, 0] = [2, 2, 0, 1]
    x_rand = np.zeros((6, 1, 2, 2), dtype=np.uint8)
    x_rand[:, 0, 0, 0] = [0, 1, 2, 0, 1, 2]
    return {
        "x_fail": x_fail, "y_fail": np.array([0, 1, 2, 0]), "pred_fail": np.array([2, 2, 0, 1]),
        "x_rand": x_rand, "y_rand": np.array([0, 1, 1, 0, 0, 2]), "pred_rand": np.array([0, 1, 2, 0, 1, 2]),
    }


def test_build_evidence_sets_and_readout_without_a_pair() -> None:
    evidence = _v1_evidence()
    counted = CountedForward(_pixel_forward)
    sets = build_evidence_sets(evidence, None, {}, counted)
    assert sets.target is None
    assert sets.pos_rand_indices.tolist() == [0, 1, 3, 5]
    assert (sets.n_neg, sets.n_pos, sets.n_rand_wrong) == (4, 4, 2)
    assert sets.summary()["neg_true_label_counts"] == {"0": 2, "1": 1, "2": 1}
    assert (counted.calls, counted.images) == (2, 10)
    readout = evidence_readout(counted({}, sets.x_all), sets)
    assert readout == {"neg_fixed": 0, "neg_still_wrong": 4, "pos_broken": 0, "n_neg": 4, "n_pos": 4}
    half_fixed = np.zeros((8, 3))
    half_fixed[np.arange(8), [0, 1, 0, 1, 0, 1, 1, 2]] = 1.0  # negatives 0 and 1 right; positive 2 (true 0) wrong
    assert evidence_readout(half_fixed, sets) == {
        "neg_fixed": 2, "neg_still_wrong": 2, "pos_broken": 1, "n_neg": 4, "n_pos": 4
    }


def test_build_evidence_sets_rejects_bad_failures() -> None:
    evidence = _v1_evidence()
    counted = CountedForward(_pixel_forward)
    correct = dict(evidence, y_fail=np.array([2, 1, 2, 0]), pred_fail=np.array([2, 2, 0, 1]))
    with pytest.raises(ValueError, match="1 x_fail images are classified correctly"):
        build_evidence_sets(correct, None, {}, counted)
    stale = dict(evidence, pred_rand=np.array([0, 1, 2, 0, 1, 1]))
    with pytest.raises(ValueError, match="pred_rand differs"):
        build_evidence_sets(stale, None, {}, counted)
    with pytest.raises(ValueError, match="true class 0 predicted 2"):
        build_evidence_sets(evidence, (0, 2), {}, counted)  # a pair (evidence-v0) is checked when given
    pair = dict(evidence, y_fail=np.zeros(4, np.int64), x_fail=np.full_like(evidence["x_fail"], 2), pred_fail=np.full(4, 2))
    assert build_evidence_sets(pair, (0, 2), {}, counted).target == (0, 2)
