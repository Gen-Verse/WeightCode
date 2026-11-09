"""Structural validation, parent hash, finiteness and size limits of weightcode.patch (contract 5.1)."""

from __future__ import annotations

import copy
import json
import pickle

import numpy as np
import pytest

from weightcode.common.identity import params_sha256
from weightcode.patch import SCHEMA, Limits, PatchError, apply_patch

OPEN = Limits(max_changed_frac=None, max_rel_frob=None, max_edit_frac=None)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the WEIGHTCODE data locations at temporary directories so no test can touch the real store."""
    root = tmp_path_factory.mktemp("weightcode_store")
    for variable in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED", "WEIGHTCODE_RUNS"):
        (root / variable).mkdir()
        monkeypatch.setenv(variable, str(root / variable))


def _model(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shapes = {"conv.weight": (4, 3, 2, 2), "conv.bias": (4,), "fc.weight": (3, 5), "fc.bias": (3,)}
    return {name: rng.normal(size=shape).astype(np.float32) for name, shape in shapes.items()}


def _patch(params: dict[str, np.ndarray], *ops: dict, **overrides) -> dict:
    patch = {
        "schema": SCHEMA,
        "parent_sha256": params_sha256(params),
        "ops": list(ops),
        "hypothesis": "test hypothesis",
        "predicted_effect": "test prediction",
    }
    patch.update(overrides)
    return patch


def _apply(params, *ops, limits=OPEN, patch_dir=None):
    return apply_patch(params, _patch(params, *ops), patch_dir, limits, params_sha256(params))


def _errors(params, *ops, limits=OPEN, patch_dir=None) -> list[str]:
    with pytest.raises(PatchError) as caught:
        _apply(params, *ops, limits=limits, patch_dir=patch_dir)
    return caught.value.messages


def _one_error(params, op: dict, needle: str, **kwargs) -> None:
    messages = _errors(params, op, **kwargs)
    assert any(needle in message for message in messages), messages


# ----------------------------------------------------------------------------------------------- top level


def test_parent_hash_mismatch_and_malformed_hash_are_rejected() -> None:
    params = _model()
    op = {"op": "scale_rows", "tensor": "fc.bias", "rows": [0], "factor": 2.0}
    other = params_sha256(_model(seed=1))
    with pytest.raises(PatchError) as caught:
        apply_patch(params, _patch(params, op, parent_sha256=other), None, OPEN, params_sha256(params))
    assert caught.value.messages == [
        f"parent_sha256: {other} is not the params_sha256 of the model this patch is checked against; "
        "write the hash of the model you were shown"
    ]
    for bad in (params_sha256(params).upper(), params_sha256(params)[:63], 7, None):
        with pytest.raises(PatchError, match="64 lowercase hex"):
            apply_patch(params, _patch(params, op, parent_sha256=bad), None, OPEN, params_sha256(params))


def test_patch_is_checked_against_the_shown_hash_and_applied_to_the_given_params() -> None:
    real, decoy = _model(seed=0), _model(seed=1)
    op = {"op": "scale_rows", "tensor": "fc.weight", "rows": [1], "factor": 0.0}
    patch = _patch(decoy, op)
    result = apply_patch(real, patch, None, OPEN, expected_parent_sha=params_sha256(decoy))
    expected = real["fc.weight"].copy()
    expected[1] = 0.0
    np.testing.assert_array_equal(result.child["fc.weight"], expected)
    with pytest.raises(PatchError, match="parent_sha256"):
        apply_patch(real, patch, None, OPEN, expected_parent_sha=params_sha256(real))


def test_top_level_schema_rules() -> None:
    params = _model()
    op = {"op": "scale_rows", "tensor": "fc.bias", "rows": [0], "factor": 2.0}
    sha = params_sha256(params)
    cases = [
        (_patch(params, op, schema="weightcode.patch.v1"), "schema: must be 'weightcode.patch.v0'"),
        (_patch(params, op, hypothesis="  "), "hypothesis: must be a non-empty string"),
        (_patch(params, op, predicted_effect=3), "predicted_effect: must be a non-empty string"),
        (_patch(params, op, notes="x"), "unknown field(s) 'notes'"),
        (_patch(params, op, ops={"op": "scale_rows"}), "ops: must be a list"),
        ({key: value for key, value in _patch(params, op).items() if key != "hypothesis"}, "missing field(s) hypothesis"),
        ([op], "patch: must be a JSON object"),
    ]
    for patch, needle in cases:
        with pytest.raises(PatchError) as caught:
            apply_patch(params, patch, None, OPEN, sha)
        assert any(needle in message for message in caught.value.messages), (needle, caught.value.messages)


def test_empty_ops_list_is_the_identity_patch() -> None:
    params = _model()
    result = _apply(params, limits=Limits())
    assert result.n_changed == 0 and result.changed_frac == 0.0 and result.child_sha256 == params_sha256(params)


def test_unknown_op_tensor_and_fields() -> None:
    params = _model()
    _one_error(params, {"op": "scale"}, "unknown op 'scale'")
    _one_error(params, {"tensor": "fc.bias"}, "missing field op")
    _one_error(params, {"op": "scale_rows", "tensor": "fc3.bias", "rows": [0], "factor": 1.0}, "unknown tensor 'fc3.bias'")
    _one_error(params, {"op": "scale_rows", "tensor": "fc.bias", "rows": [0], "factor": 1.0, "axis": 0},
               "unknown field(s) 'axis'")
    _one_error(params, {"op": "scale_rows", "tensor": "fc.bias", "rows": [0]}, "missing field(s) factor")
    _one_error(params, {"op": "scale_cols", "tensor": "fc.bias", "cols": [0], "factor": 1.0}, "needs rank >= 2")
    _one_error(params, {"op": "low_rank_add", "tensor": "fc.bias", "u": "u.npy", "v": "v.npy"}, "needs rank >= 2")
    _one_error(params, "scale_rows", "must be a JSON object")


def test_rank_zero_tensors_are_carried_but_cannot_be_targeted() -> None:
    params = {**_model(), "temperature": np.array(1.5, dtype=np.float32)}
    result = _apply(params, {"op": "scale_rows", "tensor": "fc.bias", "rows": [0], "factor": 2.0})
    assert result.child["temperature"].shape == () and result.child["temperature"] == np.float32(1.5)
    _one_error(params, {"op": "set", "tensor": "temperature", "slice": [], "value": 2.0}, "needs rank >= 1")
    _one_error(params, {"op": "add_dense", "tensor": "temperature", "delta": "t.npy"}, "needs rank >= 1")


@pytest.mark.parametrize(
    ("rows", "needle"),
    [
        ([], "non-empty list"),
        (3, "non-empty list"),
        ([-1], "rows[0] = -1 is outside [0, 3)"),
        ([3], "rows[0] = 3 is outside [0, 3)"),
        ([1.0], "rows[0] = 1.0 is not an integer"),
        ([True], "rows[0] = True is not an integer"),
        ([0, 2, 0], "rows[2] = 0 is repeated"),
    ],
)
def test_malformed_rows(rows, needle) -> None:
    _one_error(_model(), {"op": "scale_rows", "tensor": "fc.weight", "rows": rows, "factor": 0.5}, needle)


def test_malformed_cols_on_conv() -> None:
    _one_error(_model(), {"op": "scale_cols", "tensor": "conv.weight", "cols": [3], "factor": 0.5},
               "cols[0] = 3 is outside [0, 3)")


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), "0.5", True, None, 10**400])
def test_scale_factor_must_be_finite(factor) -> None:
    _one_error(_model(), {"op": "scale_rows", "tensor": "fc.weight", "rows": [0], "factor": factor},
               "factor must be a finite number")


@pytest.mark.parametrize(
    ("index", "values", "needle"),
    [
        ([], [], "index must be a non-empty list"),
        ([1, 2], [0.0, 0.0], "index[0] = 1 is not a list of 2 integers"),
        ([[1]], [0.0], "index[0] = [1] is not a list of 2 integers"),
        ([[1, 2, 0]], [0.0], "is not a list of 2 integers"),
        ([[3, 0]], [0.0], "index[0][0] = 3 is outside [0, 3)"),
        ([[0, 5]], [0.0], "index[0][1] = 5 is outside [0, 5)"),
        ([[0, -1]], [0.0], "index[0][1] = -1 is outside [0, 5)"),
        ([[0, 1.0]], [0.0], "index[0][1] = 1.0 is not an integer"),
        ([[False, 1]], [0.0], "index[0][0] = False is not an integer"),
        ([[0, 1], [1, 1]], [0.0], "values has 1 entries but index has 2 tuples"),
        ([[0, 1]], [float("nan")], "values[0] = nan is not a finite number"),
        ([[0, 1]], ["1"], "values[0] = '1' is not a finite number"),
        ([[0, 1]], 0.5, "values must be a list of numbers"),
    ],
)
@pytest.mark.parametrize("kind", ["set", "add"])
def test_malformed_index_form(kind, index, values, needle) -> None:
    _one_error(_model(), {"op": kind, "tensor": "fc.weight", "index": index, "values": values}, needle)


def test_index_form_needs_values_and_rejects_slice_fields() -> None:
    params = _model()
    _one_error(params, {"op": "set", "tensor": "fc.weight", "index": [[0, 0]]}, "the index form needs values")
    _one_error(params, {"op": "set", "tensor": "fc.weight", "index": [[0, 0]], "values": [1.0], "value": 1.0},
               "value belongs to the slice form")


def test_set_with_repeated_index_tuples_is_an_error() -> None:
    _one_error(_model(), {"op": "set", "tensor": "fc.weight", "index": [[1, 2], [0, 0], [1, 2]], "values": [1, 2, 3]},
               "index tuple [1, 2] appears 2 times")


def test_json_overflowing_literal_is_rejected_as_non_finite() -> None:
    op = json.loads('{"op": "add", "tensor": "fc.weight", "index": [[0, 0]], "values": [1e400]}')
    assert op["values"] == [float("inf")]
    _one_error(_model(), op, "values[0] = inf is not a finite number")


@pytest.mark.parametrize(
    ("fields", "needle"),
    [
        ({"slice": [[0, 1]], "value": 0.0}, "slice must be a list of 2 entries"),
        ({"slice": [[0, 1], None, None], "value": 0.0}, "slice must be a list of 2 entries"),
        ({"slice": [[1, 1], None], "value": 0.0}, "slice[0] = [1, 1] needs 0 <= start < stop <= 3"),
        ({"slice": [[2, 1], None], "value": 0.0}, "needs 0 <= start < stop <= 3"),
        ({"slice": [None, [0, 6]], "value": 0.0}, "slice[1] = [0, 6] needs 0 <= start < stop <= 5"),
        ({"slice": [[-1, 2], None], "value": 0.0}, "slice[0] = [-1, 2] needs 0 <= start < stop"),
        ({"slice": [[0, 2.0], None], "value": 0.0}, "is neither [start, stop] with integers nor null"),
        ({"slice": [[0], None], "value": 0.0}, "is neither [start, stop] with integers nor null"),
        ({"slice": [None, None]}, "needs exactly one of value"),
        ({"slice": [None, None], "value": 0.0, "values_file": "x.npy"}, "needs exactly one of value"),
        ({"slice": [None, None], "values": [0.0]}, "values belongs to the index form"),
        ({"slice": [None, None], "value": float("nan")}, "value must be a finite number"),
        ({"slice": [None, None], "values_file": "../x.npy"}, "values_file must be the bare name of a .npy file"),
        ({"slice": [None, None], "index": [[0, 0]], "value": 0.0}, "give exactly one of index"),
        ({"value": 0.0}, "give exactly one of index"),
    ],
)
def test_malformed_slice_form(fields, needle) -> None:
    _one_error(_model(), {"op": "add", "tensor": "fc.weight", **fields}, needle)


def test_values_file_shape_must_equal_the_block_shape(tmp_path) -> None:
    np.save(tmp_path / "block.npy", np.zeros((2, 5)))
    np.save(tmp_path / "flat.npy", np.zeros(10))
    params = _model()
    _apply(params, {"op": "set", "tensor": "fc.weight", "slice": [[1, 3], None], "values_file": "block.npy"},
           patch_dir=tmp_path)
    _one_error(params, {"op": "set", "tensor": "fc.weight", "slice": [[0, 3], None], "values_file": "block.npy"},
               "has shape [2, 5], but the slice selects a block of shape [3, 5]", patch_dir=tmp_path)
    _one_error(params, {"op": "set", "tensor": "fc.weight", "slice": [[1, 3], None], "values_file": "flat.npy"},
               "has shape [10], but the slice selects a block of shape [2, 5]", patch_dir=tmp_path)


# ----------------------------------------------------------------------------------------------- dense ops


@pytest.mark.parametrize(
    ("fields", "needle"),
    [
        ({"direction": [0.0, 0.0, 0.0, 0.0, 0.0]}, "direction is the zero vector"),
        ({"direction": [1.0, float("nan"), 0.0, 0.0, 0.0]}, "direction entries must be finite numbers"),
        ({"direction": [1.0, 0.0]}, "direction has 2 entries; side 'in' needs 5"),
        ({"direction": {"x": 1}}, "direction must be a .npy file name or a list of numbers"),
        ({"direction": "dir.txt"}, "direction must be the bare name of a .npy file"),
        ({"strength": 0.0}, "strength = 0.0 is outside (0, 1]"),
        ({"strength": 1.5}, "strength = 1.5 is outside (0, 1]"),
        ({"strength": -0.5}, "strength = -0.5 is outside (0, 1]"),
        ({"side": "both"}, "side must be 'in' or 'out'"),
        ({"bias": "fc.bias"}, "bias is allowed only with side 'out'"),
        ({"tensor": "fc.bias"}, "needs rank >= 2"),
    ],
)
def test_malformed_project_out(fields, needle) -> None:
    op = {"op": "project_out", "tensor": "fc.weight", "side": "in", "direction": [1.0, 0, 0, 0, 0], "strength": 1.0}
    _one_error(_model(), {**op, **fields}, needle)


def test_project_out_bias_and_file_direction_rules(tmp_path) -> None:
    params = _model()
    op = {"op": "project_out", "tensor": "fc.weight", "side": "out", "direction": [1.0, 0.0, 1.0], "strength": 1.0}
    _one_error(params, {**op, "bias": "conv.bias"}, "it must be 1-D of length 3")
    _one_error(params, {**op, "bias": "fc.weight"}, "bias must name a tensor other than tensor")
    _one_error(params, {**op, "bias": "nope"}, "unknown tensor 'nope'")
    np.save(tmp_path / "d5.npy", np.ones(5))
    np.save(tmp_path / "d0.npy", np.zeros(3))
    _one_error(params, {**op, "direction": "d5.npy"}, "direction 'd5.npy' has shape [5]; side 'out' needs a 1-D array of length 3",
               patch_dir=tmp_path)
    _one_error(params, {**op, "direction": "d0.npy"}, "direction 'd0.npy' is the zero vector", patch_dir=tmp_path)
    assert _apply(params, {**op, "strength": 1e-9}).n_changed >= 0


def test_low_rank_add_shapes_and_rank_limit(tmp_path) -> None:
    params = _model()
    rng = np.random.default_rng(0)
    files = {"u2.npy": (3, 2), "v2.npy": (5, 2), "u3.npy": (3, 3), "v3.npy": (5, 3), "u0.npy": (3, 0), "v0.npy": (5, 0),
             "v1d.npy": (5,), "ubad.npy": (4, 2)}
    for name, shape in files.items():
        np.save(tmp_path / name, rng.normal(size=shape))
    op = {"op": "low_rank_add", "tensor": "fc.weight"}
    two = Limits(max_changed_frac=None, max_rel_frob=None, max_edit_frac=None, max_rank=2)
    assert _apply(params, {**op, "u": "u2.npy", "v": "v2.npy"}, limits=two, patch_dir=tmp_path).n_changed == 15
    _one_error(params, {**op, "u": "u3.npy", "v": "v3.npy"}, "rank r = 3 is outside 1..2", limits=two, patch_dir=tmp_path)
    _one_error(params, {**op, "u": "u0.npy", "v": "v0.npy"}, "rank r = 0 is outside 1..8", patch_dir=tmp_path)
    _one_error(params, {**op, "u": "u2.npy", "v": "v3.npy"}, "u has r = 2 columns but v has r = 3", patch_dir=tmp_path)
    _one_error(params, {**op, "u": "u2.npy", "v": "v1d.npy"}, "expected [5, r]", patch_dir=tmp_path)
    _one_error(params, {**op, "u": "ubad.npy", "v": "v2.npy"}, "expected [3, r]", patch_dir=tmp_path)


def test_low_rank_add_rank_at_and_above_the_default_limit(tmp_path) -> None:
    params = _model()
    rng = np.random.default_rng(1)
    for r in (8, 9):
        np.save(tmp_path / f"u{r}.npy", rng.normal(size=(3, r)))
        np.save(tmp_path / f"v{r}.npy", rng.normal(size=(5, r)))
    _apply(params, {"op": "low_rank_add", "tensor": "fc.weight", "u": "u8.npy", "v": "v8.npy"}, patch_dir=tmp_path)
    _one_error(params, {"op": "low_rank_add", "tensor": "fc.weight", "u": "u9.npy", "v": "v9.npy"},
               "rank r = 9 is outside 1..8", patch_dir=tmp_path)


def test_add_dense_shape_must_match(tmp_path) -> None:
    np.save(tmp_path / "d.npy", np.zeros((5, 3)))
    _one_error(_model(), {"op": "add_dense", "tensor": "fc.weight", "delta": "d.npy"},
               "delta 'd.npy' has shape [5, 3], but 'fc.weight' has shape [3, 5]", patch_dir=tmp_path)


# ----------------------------------------------------------------------------------------------- limits


def test_max_ops_boundary() -> None:
    params = _model()
    op = {"op": "add", "tensor": "conv.weight", "index": [[0, 0, 0, 0]], "values": [1e-3]}
    no_edit_cap = Limits(max_edit_frac=None)
    assert _apply(params, *[op] * 64, limits=no_edit_cap).n_changed == 1
    messages = _errors(params, *[op] * 65, limits=no_edit_cap)
    assert messages == ["ops: has 65 ops; at most 64 are allowed"]
    small = Limits(max_ops=2, max_edit_frac=None)
    _apply(params, op, op, limits=small)
    assert _errors(params, op, op, op, limits=small) == ["ops: has 3 ops; at most 2 are allowed"]


def _thousand() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {"w": rng.uniform(1.0, 2.0, size=(10, 100)).astype(np.float32)}


@pytest.mark.parametrize(("count", "ok"), [(49, True), (50, True), (51, False)])
def test_changed_fraction_boundary(count, ok) -> None:
    params = _thousand()
    index = [[k // 100, k % 100] for k in range(count)]
    op = {"op": "set", "tensor": "w", "index": index, "values": [5.0] * count}
    limits = Limits(max_changed_frac=0.05, max_rel_frob=None, max_edit_frac=None)
    if ok:
        result = _apply(params, op, limits=limits)
        assert result.n_changed == count and result.changed_frac == count / 1000
    else:
        assert _errors(params, op, limits=limits) == [
            "size: 51 of 1000 parameters change (fraction 0.051), above the limit 0.05"
        ]


@pytest.mark.parametrize(
    ("factor", "ok"),
    [
        (float(np.nextafter(np.float32(2.0), np.float32(0.0))), True),
        (2.0, True),
        (float(np.nextafter(np.float32(2.0), np.float32(3.0))), False),
    ],
)
def test_relative_frobenius_boundary(factor, ok) -> None:
    params = {"w": np.ones((4, 4), dtype=np.float32), "b": np.ones(4, dtype=np.float32)}
    op = {"op": "scale_rows", "tensor": "w", "rows": [0], "factor": factor}
    limits = Limits(max_changed_frac=None, max_rel_frob=0.5)
    expected = abs(float(np.float32(factor)) - 1.0) / 2.0
    if ok:
        result = _apply(params, op, limits=limits)
        assert result.per_tensor["w"]["rel_frob"] == expected <= 0.5
    else:
        assert _errors(params, op, limits=limits) == [
            f"tensor 'w': relative Frobenius norm of the change is {expected!r}, above the limit 0.5"
        ]


def test_zero_parent_tensor_has_infinite_relative_norm_and_none_disables_limits() -> None:
    zeros = {"w": np.zeros((3, 5), dtype=np.float32), "b": np.zeros(3, dtype=np.float32)}
    op = {"op": "set", "tensor": "w", "slice": [None, None], "value": 1.0}
    result = _apply(zeros, op, limits=Limits(max_changed_frac=None, max_rel_frob=None, max_edit_frac=None))
    assert result.per_tensor["w"]["rel_frob"] == float("inf") and result.changed_frac == 15 / 18
    assert result.per_tensor["b"] == {"changed": 0, "size": 3, "rel_frob": 0.0, "max_abs_delta": 0.0}
    assert result.summary()["per_tensor"]["w"]["rel_frob"] == "inf"
    json.dumps(result.summary(), allow_nan=False)
    messages = _errors(zeros, op, limits=Limits(max_changed_frac=0.05, max_edit_frac=None))
    assert messages == [
        "size: 15 of 18 parameters change (fraction 0.8333333333333334), above the limit 0.05",
        "tensor 'w': relative Frobenius norm of the change is infinite (the parent tensor is all zeros), above the limit 0.5",
    ]


def test_per_tensor_summary_values() -> None:
    params = {"w": np.ones((2, 2), dtype=np.float32), "b": np.full(2, 3.0, dtype=np.float32)}
    result = _apply(params, {"op": "set", "tensor": "w", "index": [[0, 0]], "values": [3.0]},
                    {"op": "add", "tensor": "b", "index": [[1]], "values": [-0.5]})
    assert result.per_tensor["w"] == {"changed": 1, "size": 4, "rel_frob": 1.0, "max_abs_delta": 2.0}
    assert result.per_tensor["b"]["changed"] == 1 and result.per_tensor["b"]["max_abs_delta"] == 0.5
    assert result.per_tensor["b"]["rel_frob"] == pytest.approx(0.5 / np.sqrt(18.0), rel=1e-15)
    assert result.n_changed == 2 and result.changed_frac == 2 / 6
    assert result.summary()["n_params"] == 6


def test_entry_rounding_back_to_the_parent_is_not_a_change() -> None:
    params = {"w": np.full((2, 2), 0.5, dtype=np.float32)}
    result = _apply(params, {"op": "add", "tensor": "w", "index": [[0, 0]], "values": [1e-12]})
    assert result.n_changed == 0 and result.per_tensor["w"]["changed"] == 0
    assert result.child_sha256 == params_sha256(params)


def test_signed_zero_flip_is_not_a_change() -> None:
    zeros = {"w": np.zeros((2, 3), dtype=np.float32)}
    result = _apply(zeros, {"op": "scale_rows", "tensor": "w", "rows": [0, 1], "factor": -1.0},
                    {"op": "set", "tensor": "w", "index": [[0, 0]], "values": [-0.0]},
                    limits=Limits(max_edit_frac=None))
    assert not np.any(np.signbit(result.child["w"]))
    assert result.n_changed == 0 and result.child_sha256 == params_sha256(zeros)


@pytest.mark.parametrize(("count", "ok"), [(49, True), (50, True), (51, False)])
def test_edit_size_boundary(count, ok) -> None:
    params = _thousand()
    index = [[k // 100, k % 100] for k in range(count)]
    op = {"op": "add", "tensor": "w", "index": index, "values": [1e-3] * count}
    limits = Limits(max_changed_frac=None, max_rel_frob=None, max_edit_frac=0.05)
    if ok:
        result = _apply(params, op, limits=limits)
        assert result.edit_size == count and result.edit_frac == count / 1000
        assert result.summary()["edit_size"] == count
    else:
        assert _errors(params, op, limits=limits) == [
            "size: the patch specifies 51 numbers (edit size), above the limit 0.05 x 1000 parameters = 50"
        ]


def test_edit_size_counts_the_numbers_each_op_specifies(tmp_path) -> None:
    rng = np.random.default_rng(3)
    params = {"w": rng.normal(size=(6, 10)).astype(np.float32), "b": rng.normal(size=6).astype(np.float32)}
    np.save(tmp_path / "block.npy", np.full((2, 3), 1e-3))
    np.save(tmp_path / "d.npy", rng.normal(size=10))
    np.save(tmp_path / "u.npy", rng.normal(size=(6, 2)) * 1e-3)
    np.save(tmp_path / "v.npy", rng.normal(size=(10, 2)) * 1e-3)
    np.save(tmp_path / "dense.npy", np.full((6, 10), 1e-3))
    cases = [
        ({"op": "scale_rows", "tensor": "w", "rows": [0, 2, 4], "factor": 0.9}, 3),
        ({"op": "scale_cols", "tensor": "w", "cols": [1, 7], "factor": 1.1}, 2),
        ({"op": "add", "tensor": "w", "index": [[0, 0], [1, 1], [0, 0]], "values": [1e-3, 1e-3, 1e-3]}, 3),
        ({"op": "set", "tensor": "w", "index": [[2, 3]], "values": [0.5]}, 1),
        ({"op": "add", "tensor": "w", "slice": [[0, 2], [0, 3]], "value": 1e-3}, 1),
        ({"op": "add", "tensor": "w", "slice": [[0, 2], [0, 3]], "values_file": "block.npy"}, 6),
        ({"op": "project_out", "tensor": "w", "side": "in", "direction": "d.npy", "strength": 0.1}, 10),
        ({"op": "project_out", "tensor": "w", "side": "out", "direction": [1.0, 0, 0, 0, 0, 0], "strength": 0.1,
          "bias": "b"}, 6),
        ({"op": "low_rank_add", "tensor": "w", "u": "u.npy", "v": "v.npy"}, 2 * (6 + 10)),
        ({"op": "add_dense", "tensor": "w", "delta": "dense.npy"}, 60),
    ]
    open_limits = Limits(max_changed_frac=None, max_rel_frob=None, max_edit_frac=None)
    for op, size in cases:
        assert _apply(params, op, patch_dir=tmp_path, limits=open_limits).edit_size == size, op
    total = sum(size for _, size in cases)
    result = _apply(params, *[op for op, _ in cases], patch_dir=tmp_path, limits=open_limits)
    assert result.edit_size == total and result.edit_frac == total / 66


def test_rank_one_hidden_edit_fits_the_edit_size_limit_but_not_an_entry_cap(tmp_path) -> None:
    # FashionMNIST MLP shapes: a rank-1 edit of fc2 changes 16,384 entries (13.9 percent of 118,282) but
    # specifies 256 numbers, well under 5 percent.
    rng = np.random.default_rng(11)
    shapes = {"fc1.weight": (128, 784), "fc1.bias": (128,), "fc2.weight": (128, 128), "fc2.bias": (128,),
              "fc3.weight": (10, 128), "fc3.bias": (10,)}
    params = {name: (rng.normal(size=shape) * 0.05).astype(np.float32) for name, shape in shapes.items()}
    np.save(tmp_path / "u.npy", rng.normal(size=(128, 1)) * 1e-2)
    np.save(tmp_path / "v.npy", rng.normal(size=(128, 1)) * 1e-2)
    op = {"op": "low_rank_add", "tensor": "fc2.weight", "u": "u.npy", "v": "v.npy"}
    result = _apply(params, op, patch_dir=tmp_path, limits=Limits(max_changed_frac=None))
    assert result.edit_size == 256 and result.n_changed == 128 * 128 and result.changed_frac > 0.05
    entry_cap = Limits(max_changed_frac=0.05)
    assert any("parameters change" in message for message in _errors(params, op, patch_dir=tmp_path, limits=entry_cap))


def test_one_dimensional_tensors_are_exempt_from_the_relative_frobenius_limit() -> None:
    params = {"w": np.ones((4, 4), dtype=np.float32), "b": np.full(4, 0.01, dtype=np.float32)}
    shift = {"op": "add", "tensor": "b", "index": [[2]], "values": [-1.0]}
    limits = Limits(max_changed_frac=None, max_rel_frob=0.5, max_edit_frac=None)
    result = _apply(params, shift, limits=limits)
    assert result.per_tensor["b"]["rel_frob"] > 0.5 and result.child["b"][2] == np.float32(0.01 - 1.0)
    big = {"op": "scale_rows", "tensor": "w", "rows": [0, 1, 2], "factor": 0.0}
    assert any("relative Frobenius" in message for message in _errors(params, big, limits=limits))


def test_named_tensors_are_exempt_from_the_relative_frobenius_limit() -> None:
    params = {"w": np.ones((4, 4), dtype=np.float32), "head": np.full((3, 4), 0.01, dtype=np.float32)}
    limits = Limits(max_changed_frac=None, max_rel_frob=0.5, max_edit_frac=None, rel_frob_exempt=["head"])
    assert limits.rel_frob_exempt == ("head",) and limits == Limits(**json.loads(json.dumps(
        {"max_changed_frac": None, "max_rel_frob": 0.5, "max_edit_frac": None, "rel_frob_exempt": ["head"]})))
    fill = {"op": "add", "tensor": "head", "index": [[2, 0], [2, 1]], "values": [5.0, -5.0]}
    result = _apply(params, fill, limits=limits)
    assert result.per_tensor["head"]["rel_frob"] > 0.5 and result.child["head"][2, 0] == np.float32(5.01)
    assert limits.rel_frob_applies("w", 2) and not limits.rel_frob_applies("head", 2)
    big = {"op": "scale_rows", "tensor": "w", "rows": [0, 1, 2], "factor": 0.0}
    messages = _errors(params, big, fill, limits=limits)
    assert len(messages) == 1 and messages[0].startswith("tensor 'w': relative Frobenius")


def test_an_exempt_name_the_model_lacks_is_a_caller_error() -> None:
    params = _model()
    with pytest.raises(ValueError, match="rel_frob_exempt names tensors the model does not have"):
        _apply(params, {"op": "add", "tensor": "fc.bias", "index": [[0]], "values": [1.0]},
               limits=Limits(rel_frob_exempt=["fc2.weight"]))


# ----------------------------------------------------------------------------------------------- non-finite


def test_non_finite_results_are_rejected() -> None:
    params = _model()
    beyond_float32 = {"op": "scale_rows", "tensor": "fc.weight", "rows": [0], "factor": 1e300}
    messages = _errors(params, beyond_float32)
    assert messages == ["tensor 'fc.weight': 5 entries are not finite after the patch (float64 overflow or invalid "
                        "arithmetic, or values beyond the float32 range)"]
    _errors(params, beyond_float32, beyond_float32)
    huge = {"op": "add", "tensor": "conv.bias", "index": [[0], [0]], "values": [1.7e308, 1.7e308]}
    assert any("conv.bias" in message for message in _errors(params, huge))
    project = {"op": "project_out", "tensor": "fc.weight", "side": "in", "direction": [1, 1, 1, 1, 1], "strength": 1.0}
    assert any("fc.weight" in message for message in _errors(params, beyond_float32, beyond_float32, project))


def test_intermediate_overflow_overwritten_by_a_later_set_is_valid() -> None:
    params = _model()
    grow = {"op": "scale_rows", "tensor": "fc.weight", "rows": [0], "factor": 1e300}
    reset = {"op": "set", "tensor": "fc.weight", "slice": [[0, 1], None], "value": 0.25}
    result = _apply(params, grow, grow, reset)
    assert np.all(result.child["fc.weight"][0] == np.float32(0.25))


# ----------------------------------------------------------------------------------------------- all-or-nothing


def test_problems_from_every_op_are_reported_together(tmp_path) -> None:
    params = _model()
    ops = [
        {"op": "scale_rows", "tensor": "fc.weight", "rows": [9], "factor": 1.0},
        {"op": "set", "tensor": "fc.weight", "index": [[0, 0], [0, 0]], "values": [1, 2]},
        {"op": "project_out", "tensor": "fc.weight", "side": "in", "direction": [0, 0, 0, 0, 0], "strength": 2},
        {"op": "add_dense", "tensor": "fc.weight", "delta": "missing.npy"},
        {"op": "scale_rows", "tensor": "fc.bias", "rows": [0], "factor": 2.0},
    ]
    messages = _errors(params, *ops, patch_dir=tmp_path)
    assert [message.split(":")[0] for message in messages] == [
        "ops[0] (scale_rows)",
        "ops[1] (set)",
        "ops[2] (project_out)",
        "ops[2] (project_out)",
        "file 'missing.npy' (ops[3].delta)",
    ]


def test_message_list_is_capped() -> None:
    params = _model()
    messages = _errors(params, *[{"op": "nope"}] * 150, limits=Limits(max_ops=200))
    assert len(messages) == 101 and messages[-1] == "50 further problems were found and not listed"


def test_inputs_are_never_mutated() -> None:
    params = _model()
    snapshot = {name: value.copy() for name, value in params.items()}
    good = [
        {"op": "scale_rows", "tensor": "conv.weight", "rows": [0], "factor": 0.0},
        {"op": "project_out", "tensor": "fc.weight", "side": "out", "direction": [1, 2, 3], "strength": 1.0, "bias": "fc.bias"},
        {"op": "add", "tensor": "fc.weight", "index": [[0, 0], [0, 0]], "values": [0.5, 0.5]},
    ]
    patch = _patch(params, *good)
    patch_snapshot = copy.deepcopy(patch)
    result = apply_patch(params, patch, None, OPEN, params_sha256(params))
    assert result.n_changed > 0
    with pytest.raises(PatchError):
        apply_patch(params, _patch(params, *good, {"op": "scale_rows", "tensor": "fc.weight", "rows": [7], "factor": 1}),
                    None, OPEN, params_sha256(params))
    with pytest.raises(PatchError):
        apply_patch(params, patch, None, Limits(), params_sha256(params))
    assert patch == patch_snapshot
    for name, value in params.items():
        assert value.tobytes() == snapshot[name].tobytes() and value.flags.writeable
        assert not np.shares_memory(value, result.child[name])


def test_caller_errors_are_not_patch_errors() -> None:
    params = _model()
    sha = params_sha256(params)
    patch = _patch(params)
    with pytest.raises(TypeError):
        apply_patch({**params, "fc.bias": params["fc.bias"].astype(np.float64)}, patch, None, OPEN, sha)
    with pytest.raises(TypeError):
        apply_patch({**params, "fc.bias": params["fc.bias"].tolist()}, patch, None, OPEN, sha)
    bad = {**params, "fc.bias": params["fc.bias"].copy()}
    bad["fc.bias"][0] = np.nan
    with pytest.raises(ValueError) as caught:
        apply_patch(bad, patch, None, OPEN, sha)
    assert not isinstance(caught.value, PatchError)
    with pytest.raises(ValueError) as caught:
        apply_patch({}, patch, None, OPEN, sha)
    assert not isinstance(caught.value, PatchError)
    with pytest.raises(TypeError):
        apply_patch(params, patch, None, {"max_ops": 3}, sha)
    with pytest.raises(ValueError) as caught:
        apply_patch(params, patch, None, OPEN, sha.upper())
    assert not isinstance(caught.value, PatchError)


@pytest.mark.parametrize(
    "fields",
    [
        {"max_changed_frac": 1.5},
        {"max_changed_frac": -0.1},
        {"max_changed_frac": True},
        {"max_rel_frob": float("inf")},
        {"max_rel_frob": -1.0},
        {"max_ops": 0},
        {"max_rank": 0},
        {"max_rank": 2.0},
        {"max_file_bytes": -1},
        {"rel_frob_exempt": "fc.weight"},
        {"rel_frob_exempt": ["fc.weight", "fc.weight"]},
        {"rel_frob_exempt": [""]},
        {"rel_frob_exempt": [3]},
    ],
)
def test_invalid_limits_are_refused(fields) -> None:
    with pytest.raises(ValueError, match="invalid Limits"):
        Limits(**fields)


def test_task_limits_is_the_json_form_of_every_field() -> None:
    import dataclasses

    from weightcode.patch import task_limits

    every = dataclasses.asdict(Limits())
    assert every.pop("rel_frob_exempt") == () and task_limits() == every == json.loads(json.dumps(task_limits()))
    custom = Limits(max_changed_frac=0.2, max_edit_frac=None, rel_frob_exempt=["fc.weight"])
    assert task_limits(custom)["rel_frob_exempt"] == ["fc.weight"] and Limits(**task_limits(custom)) == custom
    assert Limits(**task_limits(Limits(max_changed_frac=0.2))) == Limits(max_changed_frac=0.2)
    with pytest.raises(TypeError):
        task_limits({"max_rel_frob": 0.5})


def test_patch_error_pickles_with_its_messages() -> None:
    error = PatchError(["a: one", "b: two"])
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, PatchError) and restored.messages == ["a: one", "b: two"]
