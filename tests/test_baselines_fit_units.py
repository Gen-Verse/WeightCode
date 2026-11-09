"""Unit tests for the fitting references and random controls: closed forms, limits, draws and patch application."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from weightcode.baselines.fit.controls import (
    baseline_session_id,
    candidate_rng,
    control_run_seed,
    gaussian_delta,
    permuted_delta,
)
from weightcode.baselines.fit.edits import (
    MAX_REL_FROB,
    REL_FROB_MIN_NDIM,
    apply_patch_path,
    edit_stats,
    rel_frob,
    scale_to_norm_limit,
)
from weightcode.baselines.fit.ridge import ridge_solve, ridge_targets, softmax
from weightcode.baselines.fit.runner import jsonable, safe_name
from weightcode.baselines.fit.torch_forward import architecture_from
from weightcode.baselines.instance_io import write_delta_patch
from weightcode.common.identity import params_sha256
from weightcode.patch import PatchError


def _params(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "fc1.weight": rng.normal(size=(5, 12)).astype(np.float32),
        "fc1.bias": rng.normal(size=(5,)).astype(np.float32),
        "fc2.weight": rng.normal(size=(3, 5)).astype(np.float32),
        "fc2.bias": rng.normal(size=(3,)).astype(np.float32),
    }


SPEC = {"input": {"shape": [3, 2, 2], "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}}


def test_ridge_solve_matches_stacked_least_squares_and_limits() -> None:
    rng = np.random.default_rng(3)
    h1 = np.hstack([rng.normal(size=(20, 6)), np.ones((20, 1))])
    w0 = rng.normal(size=(4, 7))
    targets = rng.normal(size=(20, 4))
    lam = 0.7
    w = ridge_solve(h1, targets, w0, lam)
    stacked_x = np.vstack([h1, np.sqrt(lam) * np.eye(7)])
    stacked_t = np.vstack([targets, np.sqrt(lam) * w0.T])
    expected = np.linalg.lstsq(stacked_x, stacked_t, rcond=None)[0].T
    np.testing.assert_allclose(w, expected, rtol=1e-10, atol=1e-10)
    gradient = h1.T @ (h1 @ w.T - targets) + lam * (w - w0).T
    np.testing.assert_allclose(gradient, 0.0, atol=1e-9)
    np.testing.assert_allclose(ridge_solve(h1, targets, w0, 1e12), w0, atol=1e-8)
    with pytest.raises(ValueError):
        ridge_solve(h1, targets, w0, 0.0)


def test_ridge_targets_anchor_on_base_logits_and_are_shift_invariant() -> None:
    z0 = np.array([[20.0, 0.0, -1.0], [1.0, 1.2, 0.0], [0.0, 3.0, 0.5]])
    y = np.array([0, 0, 2])
    targets, scale = ridge_targets(z0, y)
    expected_scale = np.mean(z0.max(axis=1) - z0.mean(axis=1))
    assert scale == pytest.approx(expected_scale)
    np.testing.assert_allclose(targets[0], z0[0], atol=1e-6 * scale)  # confidently correct: no pull
    assert targets[1, 0] - targets[1, 1] > z0[1, 0] - z0[1, 1]  # wrong: label pushed above the predicted class
    shifted, shifted_scale = ridge_targets(z0 + 5.0, y)
    assert shifted_scale == pytest.approx(scale)
    np.testing.assert_allclose(shifted, targets + 5.0)
    np.testing.assert_allclose(softmax(z0).sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        ridge_targets(z0, np.array([0, 3, 1]))


def test_rel_frob_and_edit_stats() -> None:
    base = _params()
    child = {key: value.copy() for key, value in base.items()}
    child["fc2.bias"][1] += 1.0
    stats = edit_stats(base, child)
    assert stats["n_changed"] == 1
    assert stats["changed_frac"] == pytest.approx(1 / stats["n_params"])
    assert stats["per_tensor"]["fc2.bias"]["rel_frob"] == pytest.approx(
        1.0 / np.linalg.norm(base["fc2.bias"].astype(np.float64)), rel=1e-6
    )
    over = {key: value.copy() for key, value in base.items()}
    over["fc1.weight"] = over["fc1.weight"] * 3.0
    assert not edit_stats(base, over)["within_limits"]
    assert edit_stats(base, over, max_changed_frac=None, max_rel_frob=None, max_edit_frac=None)["within_limits"]
    assert stats["edit_size"] == 1 and stats["per_tensor"]["fc2.bias"]["encoding"] == "add"
    full = edit_stats(base, over, max_changed_frac=None, max_rel_frob=None, max_edit_frac=None)
    assert full["per_tensor"]["fc1.weight"]["encoding"] == "add_dense"
    assert full["edit_size"] == base["fc1.weight"].size
    zero = np.zeros(3, dtype=np.float32)
    assert rel_frob(zero, zero) == 0.0
    assert math.isinf(rel_frob(zero, np.ones(3, dtype=np.float32)))


def test_scale_to_norm_limit_scales_only_over_limit_tensors() -> None:
    base = _params()
    child = {key: value.copy() for key, value in base.items()}
    child["fc2.weight"] = (base["fc2.weight"] * 2.5).astype(np.float32)  # rel norm 1.5
    child["fc2.bias"] = (base["fc2.bias"] * 1.2).astype(np.float32)  # rel norm 0.2
    scaled, records = scale_to_norm_limit(base, child, ["fc2.weight", "fc2.bias"])
    assert records["fc2.weight"]["scaled"] and not records["fc2.bias"]["scaled"]
    assert records["fc2.weight"]["rel_frob_before"] == pytest.approx(1.5, rel=1e-6)
    assert records["fc2.weight"]["rel_frob_after"] <= 0.5
    assert records["fc2.weight"]["rel_frob_after"] == pytest.approx(0.5, rel=1e-3)
    assert np.array_equal(scaled["fc2.bias"], child["fc2.bias"])
    direction = scaled["fc2.weight"].astype(np.float64) - base["fc2.weight"]
    np.testing.assert_allclose(direction / np.linalg.norm(direction), 1.5 * base["fc2.weight"] / np.linalg.norm(
        1.5 * base["fc2.weight"].astype(np.float64)), atol=1e-5)
    zero_base = dict(base, **{"fc2.weight": np.zeros((3, 5), dtype=np.float32)})
    moved = dict(zero_base, **{"fc2.weight": np.ones((3, 5), dtype=np.float32)})
    cleared, info = scale_to_norm_limit(zero_base, moved, ["fc2.weight"])
    assert info["fc2.weight"]["factor"] == 0.0
    assert np.array_equal(cleared["fc2.weight"], zero_base["fc2.weight"])
    exempt, exempt_info = scale_to_norm_limit(base, dict(base, **{"fc2.bias": base["fc2.bias"] * 9}), ["fc2.bias"],
                                              min_ndim=2)
    assert not exempt_info["fc2.bias"]["scaled"]


def _sparse_delta(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    dense = rng.normal(size=(6, 7))
    sparse = np.zeros((4, 5))
    sparse.flat[[1, 7, 13]] = [0.5, -2.0, 1.25]
    return {"a.weight": dense, "b.weight": sparse, "c.bias": np.zeros(3)}


def test_permuted_delta_keeps_values_count_and_norm_exactly() -> None:
    delta = _sparse_delta(0)
    seed = control_run_seed("inst/grad_last/0", "permuted", 0)
    first = permuted_delta(delta, candidate_rng(seed, 0))
    again = permuted_delta(delta, candidate_rng(seed, 0))
    other = permuted_delta(delta, candidate_rng(seed, 1))
    for key, value in delta.items():
        np.testing.assert_array_equal(np.sort(first[key].ravel()), np.sort(value.ravel()))
        assert np.count_nonzero(first[key]) == np.count_nonzero(value)
        assert np.array_equal(first[key], again[key])
    assert not np.any(first["c.bias"])
    assert not np.array_equal(first["a.weight"], other["a.weight"])
    assert not np.array_equal(first["b.weight"], delta["b.weight"]) or not np.array_equal(
        other["b.weight"], delta["b.weight"]
    )


def test_gaussian_delta_keeps_the_reference_positions_and_norm() -> None:
    delta = _sparse_delta(1)
    seed = control_run_seed("inst/grad_last/0", "gaussian", 2)
    drawn = gaussian_delta(delta, candidate_rng(seed, 5))
    for key, value in delta.items():
        assert np.array_equal(drawn[key] != 0.0, value != 0.0), "gaussian keeps the reference's changed positions"
        assert np.linalg.norm(drawn[key]) == pytest.approx(np.linalg.norm(value), rel=1e-12, abs=0.0)
    assert not np.array_equal(drawn["a.weight"], gaussian_delta(delta, candidate_rng(seed, 6))["a.weight"])


def test_control_seeds_follow_the_prereg_rule() -> None:
    import hashlib

    session = baseline_session_id("i1234abcd", "grad_last", 0)
    assert session == "i1234abcd/grad_last/0"
    digest = hashlib.sha256(b"i1234abcd/grad_last/0|gaussian|2").digest()
    assert control_run_seed(session, "gaussian", 2) == int.from_bytes(digest[:8], "big")
    seeds = {control_run_seed(session, kind, index) for kind in ("permuted", "gaussian") for index in range(3)}
    assert len(seeds) == 6 and all(0 <= seed < 2**64 for seed in seeds)
    expected = np.random.default_rng([control_run_seed(session, "permuted", 1), 3]).standard_normal(4)
    drawn = candidate_rng(control_run_seed(session, "permuted", 1), 3).standard_normal(4)
    np.testing.assert_array_equal(drawn, expected)
    with pytest.raises(ValueError):
        control_run_seed(session, "uniform", 0)


def test_delta_patch_roundtrips_through_the_executor_with_and_without_limits(tmp_path) -> None:
    base = _params()
    child = {key: value.copy() for key, value in base.items()}
    child["fc1.weight"][0, :3] += np.float32(0.25)
    child["fc2.bias"][2] -= np.float32(1e-3)
    sha = params_sha256(base)
    path = write_delta_patch(tmp_path / "p", sha, base, child, "h", "e")
    ops = json.loads(path.read_text())["ops"]
    written = [(op["op"], op["tensor"], len(op["index"])) for op in ops]
    assert written == [("add", "fc1.weight", 3), ("add", "fc2.bias", 1)]
    assert params_sha256(apply_patch_path(base, path, sha, enforce_limits=True)) == params_sha256(child)
    with pytest.raises(PatchError):
        apply_patch_path(base, path, "0" * 64, enforce_limits=False)
    # one entry of a matrix pushed far past the norm limit: only the norm limit can reject it
    big = {key: value.copy() for key, value in base.items()}
    big["fc2.weight"][0, 0] += np.float32(100.0)
    big_path = write_delta_patch(tmp_path / "big", sha, base, big, "h", "e")
    assert edit_stats(base, big)["per_tensor"]["fc2.weight"]["rel_frob"] > MAX_REL_FROB
    with pytest.raises(PatchError):
        apply_patch_path(base, big_path, sha, enforce_limits=True)
    assert params_sha256(apply_patch_path(base, big_path, sha, enforce_limits=False)) == params_sha256(big)
    # every entry of a tensor changed: written as add_dense
    dense = dict(base, **{"fc2.weight": (base["fc2.weight"] * 1.01).astype(np.float32)})
    dense_path = write_delta_patch(tmp_path / "dense", sha, base, dense, "h", "e")
    assert [op["op"] for op in json.loads(dense_path.read_text())["ops"]] == ["add_dense"]
    assert params_sha256(apply_patch_path(base, dense_path, sha, enforce_limits=False)) == params_sha256(dense)


def test_norm_limit_on_a_vector_follows_the_executor_rank_rule(tmp_path) -> None:
    base = _params()
    sha = params_sha256(base)
    bias = {key: value.copy() for key, value in base.items()}
    bias["fc2.bias"][0] += np.float32(100.0)
    path = write_delta_patch(tmp_path / "bias", sha, base, bias, "h", "e")
    stats = edit_stats(base, bias)
    assert stats["per_tensor"]["fc2.bias"]["rel_frob"] > MAX_REL_FROB
    if REL_FROB_MIN_NDIM >= 2:  # 1-D tensors are exempt from the norm limit
        assert stats["within_limits"]
        assert params_sha256(apply_patch_path(base, path, sha, enforce_limits=True)) == params_sha256(bias)
    else:
        assert not stats["within_limits"]
        with pytest.raises(PatchError):
            apply_patch_path(base, path, sha, enforce_limits=True)


def test_architecture_inference_and_errors() -> None:
    arch = architecture_from(_params(), SPEC)
    assert arch.family == "mlp"
    assert arch.last_linear_tensors == ("fc2.weight", "fc2.bias")
    with pytest.raises(ValueError, match="outside"):
        architecture_from(dict(_params(), **{"bn1.weight": np.ones(5, np.float32)}), SPEC)
    gap = {key.replace("fc2", "fc3"): value for key, value in _params().items()}
    with pytest.raises(ValueError, match="without gaps"):
        architecture_from(gap, SPEC)
    with pytest.raises(ValueError, match="inputs"):
        architecture_from(_params(), {"input": {"shape": [1, 2, 2], "mean": [0.5], "std": [0.2]}})
    with pytest.raises(ValueError, match="input"):
        architecture_from(_params(), {"input": {"shape": [3, 2, 2]}})


def test_jsonable_and_safe_name() -> None:
    value = {"a": np.float32(1.5), "b": [np.int64(2), math.inf, -math.inf, math.nan], "c": np.bool_(True)}
    assert jsonable(value) == {"a": 1.5, "b": [2, "inf", "-inf", "nan"], "c": True}
    json.dumps(jsonable(value), allow_nan=False)
    assert safe_name("grad_last", "x") == "grad_last"
    for bad in ("..", "a/b", "", "x" * 200):
        with pytest.raises(ValueError):
            safe_name(bad, "x")


def test_label_shift_direction_matches_the_closed_form_and_ignores_logit_shifts() -> None:
    from weightcode.baselines.fit.rules import label_shift_direction

    y = np.array([0, 0, 0, 1, 2, 2], dtype=np.int64)
    uniform = label_shift_direction(np.zeros((6, 4)), y)
    p_dep = (np.array([3, 1, 2, 0]) + 1.0) / (6 + 4)
    np.testing.assert_allclose(uniform["p_dep"], p_dep)
    np.testing.assert_allclose(uniform["p_pred"], 0.25)
    expected = np.log(p_dep) - np.log(0.25)
    np.testing.assert_allclose(uniform["direction"], expected - expected.mean())
    assert abs(float(uniform["direction"].sum())) < 1e-12
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(6, 4))
    shifted = label_shift_direction(logits + 7.5, y)
    np.testing.assert_allclose(shifted["direction"], label_shift_direction(logits, y)["direction"], atol=1e-12)
    with pytest.raises(ValueError):
        label_shift_direction(logits, np.array([0, 1, 2, 3, 4, 0]))


@pytest.mark.parametrize(("shape", "groups"), [((4, 12), np.arange(12) % 4), ((3, 2, 3, 3), np.arange(18) % 9)])
@pytest.mark.parametrize("strength", [0.37, 1.0])
def test_tie_columns_matches_the_executor_bit_for_bit(shape, groups, strength) -> None:
    from weightcode.baselines.fit.rules import tie_columns
    from weightcode.patch import Limits, apply_patch

    rng = np.random.default_rng(11)
    params = {"w": rng.normal(size=shape).astype(np.float32), "b": rng.normal(size=(shape[0],)).astype(np.float32)}
    sha = params_sha256(params)
    patch = {"schema": "weightcode.patch.v0", "parent_sha256": sha, "hypothesis": "h", "predicted_effect": "p",
             "ops": [{"op": "tie", "tensor": "w", "axis": 1, "groups": groups.tolist(), "strength": strength}]}
    result = apply_patch(params, patch, None, Limits().without_size_limits(), sha)
    expected = tie_columns(params["w"], groups, strength)
    assert expected.dtype == np.float32 and expected.shape == shape
    assert np.array_equal(result.child["w"], expected)
    assert result.edit_size == groups.size
    if strength == 1.0:
        view = expected.reshape(shape[0], -1)
        for group in np.unique(groups):
            columns = view[:, groups == group]
            np.testing.assert_allclose(columns, columns[:, :1].repeat(columns.shape[1], axis=1), atol=1e-6)


def test_tie_columns_leaves_singleton_groups_unchanged() -> None:
    from weightcode.baselines.fit.rules import tie_columns

    weight = np.random.default_rng(2).normal(size=(3, 5)).astype(np.float32)
    tied = tie_columns(weight, np.array([0, 0, 2, 3, 4]), 1.0)
    assert np.array_equal(tied[:, 2:], weight[:, 2:])
    assert not np.array_equal(tied[:, :2], weight[:, :2])


def test_tie_strengths_follow_the_norm_limit() -> None:
    from weightcode.baselines.fit.edits import NORM_SCALE_MARGIN
    from weightcode.baselines.fit.rules import tie_strengths

    assert MAX_REL_FROB is not None
    cap = MAX_REL_FROB * (1.0 - NORM_SCALE_MARGIN)
    strengths, s_max = tie_strengths(0.67, 2)
    assert strengths == [0.5, cap / 0.67] and s_max == cap / 0.67
    assert tie_strengths(0.3, 2) == ([0.5, 1.0], 1.0)
    single, s_max = tie_strengths(1.2, 4)
    assert single == [cap / 1.2] and s_max < 0.5
    assert tie_strengths(cap / 0.5, 2)[0] == [0.5]
    assert tie_strengths(3.0, REL_FROB_MIN_NDIM - 1) == ([0.5, 1.0], 1.0)
    with pytest.raises(ValueError):
        tie_strengths(0.0, 2)


def test_edit_stats_uses_an_executor_edit_size_when_given() -> None:
    base = _params()
    child = {key: value.copy() for key, value in base.items()}
    child["fc1.weight"] = child["fc1.weight"] + np.float32(0.01)
    derived = edit_stats(base, child)
    assert derived["edit_size_source"] == "delta_encoding"
    assert derived["edit_size"] == base["fc1.weight"].size
    assert derived["per_tensor"]["fc1.weight"]["encoding"] == "add_dense"
    given = edit_stats(base, child, edit_size=12, max_edit_frac=12 / derived["n_params"])
    assert given["edit_size_source"] == "executor" and given["edit_size"] == 12 and given["within_limits"]
    assert "encoding" not in given["per_tensor"]["fc1.weight"]
    over = edit_stats(base, child, edit_size=13, max_edit_frac=12 / derived["n_params"])
    assert not over["within_limits"] and "edit fraction" in over["violations"][0]


def test_write_ops_patch_writes_a_patch_the_executor_applies(tmp_path) -> None:
    from weightcode.baselines.fit.edits import apply_patch_result, write_ops_patch

    base = _params()
    sha = params_sha256(base)
    op = {"op": "tie", "tensor": "fc1.weight", "axis": 1, "groups": (np.arange(12) % 4).tolist(), "strength": 0.5}
    path = write_ops_patch(tmp_path / "q0", sha, [op], "h", "p")
    assert json.loads(path.read_text())["ops"] == [op]
    result = apply_patch_result(base, path, sha, enforce_limits=False)
    assert result.edit_size == 12 and result.n_changed > 0
    with pytest.raises(ValueError):
        write_ops_patch(tmp_path / "q1", sha, [], "h", "p")
