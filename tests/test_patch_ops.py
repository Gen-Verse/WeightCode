"""Op semantics of weightcode.patch against hand-computed numpy references (contract 5.0, HARNESS_SPEC section 2)."""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_max_ulp

from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.patch import OP_NAMES, SCHEMA, Limits, PatchError, apply_patch, describe_ops

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
    shapes = {
        "conv.weight": (4, 3, 2, 2),
        "conv.bias": (4,),
        "fc1.weight": (5, 12),
        "fc1.bias": (5,),
        "fc2.weight": (3, 5),
        "fc2.bias": (3,),
    }
    return {name: rng.normal(size=shape).astype(np.float32) for name, shape in shapes.items()}


def _patch(params: dict[str, np.ndarray], *ops: dict) -> dict:
    return {
        "schema": SCHEMA,
        "parent_sha256": params_sha256(params),
        "ops": list(ops),
        "hypothesis": "test hypothesis",
        "predicted_effect": "test prediction",
    }


def _apply(params, *ops, patch_dir=None, limits=OPEN):
    return apply_patch(params, _patch(params, *ops), patch_dir, limits, params_sha256(params))


def _f64(params: dict[str, np.ndarray], name: str) -> np.ndarray:
    return params[name].astype(np.float64)


def _assert_untouched(params, result, touched: set[str]) -> None:
    for name, value in params.items():
        if name not in touched:
            assert result.child[name].tobytes() == value.tobytes(), name


def test_scale_rows_and_cols_hand_computed() -> None:
    params = {"w": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), "b": np.array([1.0, -2.0], dtype=np.float32)}
    result = _apply(
        params,
        {"op": "scale_rows", "tensor": "w", "rows": [1], "factor": 0.5},
        {"op": "scale_cols", "tensor": "w", "cols": [0, 2], "factor": -1.0},
        {"op": "scale_rows", "tensor": "b", "rows": [0], "factor": 3.0},
    )
    assert_array_equal(result.child["w"], np.array([[-1.0, 2.0, -3.0], [-2.0, 2.5, -3.0]], dtype=np.float32))
    assert_array_equal(result.child["b"], np.array([3.0, -2.0], dtype=np.float32))
    assert result.n_changed == 6
    assert result.per_tensor["w"]["changed"] == 5 and result.per_tensor["b"]["changed"] == 1


def test_scale_rows_and_cols_on_conv_address_output_and_input_channels() -> None:
    params = _model()
    result = _apply(
        params,
        {"op": "scale_rows", "tensor": "conv.weight", "rows": [0, 2], "factor": 0.25},
        {"op": "scale_cols", "tensor": "conv.weight", "cols": [1], "factor": -3.0},
    )
    reference = _f64(params, "conv.weight")
    reference[[0, 2], :, :, :] *= 0.25
    reference[:, [1], :, :] *= -3.0
    assert_array_equal(result.child["conv.weight"], reference.astype(np.float32))
    _assert_untouched(params, result, {"conv.weight"})


def test_set_and_add_index_form_including_4d_tuples() -> None:
    params = _model()
    result = _apply(
        params,
        {"op": "set", "tensor": "fc2.weight", "index": [[0, 1], [2, 4]], "values": [0.5, -1.25]},
        {"op": "add", "tensor": "conv.weight", "index": [[3, 2, 1, 0], [0, 0, 0, 1]], "values": [0.75, -2]},
        {"op": "set", "tensor": "fc2.bias", "index": [[2]], "values": [7]},
    )
    fc2 = _f64(params, "fc2.weight")
    fc2[0, 1], fc2[2, 4] = 0.5, -1.25
    conv = _f64(params, "conv.weight")
    conv[3, 2, 1, 0] += 0.75
    conv[0, 0, 0, 1] += -2.0
    bias = _f64(params, "fc2.bias")
    bias[2] = 7.0
    assert_array_equal(result.child["fc2.weight"], fc2.astype(np.float32))
    assert_array_equal(result.child["conv.weight"], conv.astype(np.float32))
    assert_array_equal(result.child["fc2.bias"], bias.astype(np.float32))
    _assert_untouched(params, result, {"fc2.weight", "conv.weight", "fc2.bias"})


def test_add_with_repeated_index_tuples_accumulates_like_add_at() -> None:
    params = _model()
    result = _apply(params, {"op": "add", "tensor": "fc1.weight", "index": [[1, 2], [1, 2], [0, 0], [1, 2]],
                             "values": [0.1, 0.2, 0.3, -0.05]})
    reference = _f64(params, "fc1.weight")
    np.add.at(reference, (np.array([1, 1, 0, 1]), np.array([2, 2, 0, 2])), np.array([0.1, 0.2, 0.3, -0.05]))
    assert_array_equal(result.child["fc1.weight"], reference.astype(np.float32))
    assert result.per_tensor["fc1.weight"]["changed"] == 2


def test_set_and_add_slice_form_with_value_and_values_file(tmp_path) -> None:
    params = _model()
    block = np.arange(6, dtype=np.float64).reshape(2, 3) / 10.0
    conv_block = np.linspace(-1.0, 1.0, 2 * 3 * 1 * 2).reshape(2, 3, 1, 2).astype(np.float32)
    np.save(tmp_path / "block.npy", block)
    np.save(tmp_path / "conv_block.npy", conv_block)
    result = _apply(
        params,
        {"op": "set", "tensor": "fc1.weight", "slice": [[1, 3], [2, 5]], "values_file": "block.npy"},
        {"op": "add", "tensor": "fc1.weight", "slice": [None, [0, 1]], "value": 1.5},
        {"op": "set", "tensor": "conv.weight", "slice": [[2, 3], None, None, None], "value": 0.0},
        {"op": "add", "tensor": "conv.weight", "slice": [[0, 2], None, [1, 2], None], "values_file": "conv_block.npy"},
        {"op": "add", "tensor": "fc1.bias", "slice": [[4, 5]], "value": -0.5},
        patch_dir=tmp_path,
    )
    fc1 = _f64(params, "fc1.weight")
    fc1[1:3, 2:5] = block
    fc1[:, 0:1] += 1.5
    conv = _f64(params, "conv.weight")
    conv[2:3] = 0.0
    conv[0:2, :, 1:2, :] += conv_block.astype(np.float64)
    bias = _f64(params, "fc1.bias")
    bias[4] -= 0.5
    assert_array_equal(result.child["fc1.weight"], fc1.astype(np.float32))
    assert_array_equal(result.child["conv.weight"], conv.astype(np.float32))
    assert_array_equal(result.child["fc1.bias"], bias.astype(np.float32))


def test_project_out_hand_computed_both_sides_and_bias() -> None:
    params = {"w": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), "b": np.array([5.0, 6.0], dtype=np.float32)}
    side_in = _apply(params, {"op": "project_out", "tensor": "w", "side": "in", "direction": [2.0, 0.0], "strength": 1.0})
    assert_array_equal(side_in.child["w"], np.array([[0.0, 2.0], [0.0, 4.0]], dtype=np.float32))
    assert_array_equal(side_in.child["b"], params["b"])
    side_out = _apply(params, {"op": "project_out", "tensor": "w", "side": "out", "direction": [0.0, 3.0],
                               "strength": 0.5, "bias": "b"})
    assert_array_equal(side_out.child["w"], np.array([[1.0, 2.0], [1.5, 2.0]], dtype=np.float32))
    assert_array_equal(side_out.child["b"], np.array([5.0, 3.0], dtype=np.float32))


def _exact_matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.array([math.fsum((row * vector).tolist()) for row in matrix])


def test_project_out_on_conv_matches_reference(tmp_path) -> None:
    params = _model()
    rng = np.random.default_rng(5)
    d_in = rng.normal(size=12)
    d_out = rng.normal(size=4)
    np.save(tmp_path / "d_in.npy", d_in)
    weight = _f64(params, "conv.weight").reshape(4, 12)
    bias = _f64(params, "conv.bias")

    result = _apply(params, {"op": "project_out", "tensor": "conv.weight", "side": "in", "direction": "d_in.npy",
                             "strength": 0.7}, patch_dir=tmp_path)
    unit = d_in / np.linalg.norm(d_in)
    blas = (weight - 0.7 * np.outer(weight @ unit, unit)).reshape(4, 3, 2, 2)
    assert_array_max_ulp(result.child["conv.weight"], blas.astype(np.float32), maxulp=1)
    scaled = d_in / np.max(np.abs(d_in))
    documented = scaled / math.sqrt(math.fsum((scaled * scaled).tolist()))
    exact = (weight - 0.7 * np.outer(_exact_matvec(weight, documented), documented)).reshape(4, 3, 2, 2)
    assert_array_equal(result.child["conv.weight"], exact.astype(np.float32))

    full = _apply(params, {"op": "project_out", "tensor": "conv.weight", "side": "out", "direction": d_out.tolist(),
                           "strength": 1.0, "bias": "conv.bias"})
    unit = d_out / np.linalg.norm(d_out)
    expected_w = (weight - np.outer(unit, unit @ weight)).reshape(4, 3, 2, 2)
    expected_b = bias - unit * (unit @ bias)
    assert_array_max_ulp(full.child["conv.weight"], expected_w.astype(np.float32), maxulp=1)
    assert_array_max_ulp(full.child["conv.bias"], expected_b.astype(np.float32), maxulp=1)
    projected = unit @ full.child["conv.weight"].astype(np.float64).reshape(4, 12)
    assert np.max(np.abs(projected)) < 1e-6 * np.max(np.abs(weight))
    _assert_untouched(params, full, {"conv.weight", "conv.bias"})


def test_project_out_direction_scale_does_not_matter() -> None:
    params = _model()
    base = [0.3, -1.2, 0.0, 2.5, 0.7]
    results = [
        _apply(params, {"op": "project_out", "tensor": "fc1.weight", "side": "out",
                        "direction": [scale * x for x in base], "strength": 0.9})
        for scale in (1.0, 1e-200, 1e200)
    ]
    assert len({result.child_sha256 for result in results}) == 1


def test_low_rank_add_on_linear_and_conv(tmp_path) -> None:
    params = _model()
    rng = np.random.default_rng(9)
    u, v = rng.normal(size=(3, 2)), rng.normal(size=(5, 2))
    cu, cv = rng.normal(size=(4, 3)).astype(np.float32), rng.normal(size=(12, 3)).astype(np.float32)
    for name, array in (("u.npy", u), ("v.npy", v), ("cu.npy", cu), ("cv.npy", cv)):
        np.save(tmp_path / name, array)
    result = _apply(
        params,
        {"op": "low_rank_add", "tensor": "fc2.weight", "u": "u.npy", "v": "v.npy"},
        {"op": "low_rank_add", "tensor": "conv.weight", "u": "cu.npy", "v": "cv.npy"},
        patch_dir=tmp_path,
    )
    fc2 = _f64(params, "fc2.weight")
    conv = _f64(params, "conv.weight").reshape(4, 12)
    assert_array_max_ulp(result.child["fc2.weight"], (fc2 + u @ v.T).astype(np.float32), maxulp=1)
    assert_array_max_ulp(result.child["conv.weight"], (conv + cu.astype(np.float64) @ cv.astype(np.float64).T)
                         .reshape(4, 3, 2, 2).astype(np.float32), maxulp=1)
    ordered = np.zeros((3, 5))
    for k in range(2):
        ordered += np.outer(u[:, k], v[:, k])
    assert_array_equal(result.child["fc2.weight"], (fc2 + ordered).astype(np.float32))


def test_add_dense_on_conv_accepts_float_and_integer_files(tmp_path) -> None:
    params = _model()
    delta = np.random.default_rng(2).normal(size=(4, 3, 2, 2)).astype(np.float32)
    steps = np.arange(5, dtype=np.int16) - 2
    np.save(tmp_path / "d_conv.npy", delta)
    np.save(tmp_path / "steps.npy", steps)
    result = _apply(
        params,
        {"op": "add_dense", "tensor": "conv.weight", "delta": "d_conv.npy"},
        {"op": "add_dense", "tensor": "fc1.bias", "delta": "steps.npy"},
        patch_dir=tmp_path,
    )
    expected = _f64(params, "conv.weight") + delta.astype(np.float64)
    assert_array_equal(result.child["conv.weight"], expected.astype(np.float32))
    assert_array_equal(result.child["fc1.bias"], (_f64(params, "fc1.bias") + steps).astype(np.float32))


def test_ops_apply_in_list_order() -> None:
    params = _model()
    scale = {"op": "scale_rows", "tensor": "fc2.weight", "rows": [0], "factor": 2.0}
    add = {"op": "add", "tensor": "fc2.weight", "index": [[0, 3]], "values": [1.0]}
    first_scale = _apply(params, scale, add)
    first_add = _apply(params, add, scale)
    x = float(params["fc2.weight"][0, 3])
    assert first_scale.child["fc2.weight"][0, 3] == np.float32(x * 2.0 + 1.0)
    assert first_add.child["fc2.weight"][0, 3] == np.float32((x + 1.0) * 2.0)
    assert first_scale.child_sha256 != first_add.child_sha256
    project = {"op": "project_out", "tensor": "fc2.weight", "side": "in", "direction": [1, 0, 0, 0, 0], "strength": 1.0}
    zero_then_project = _apply(params, {"op": "set", "tensor": "fc2.weight", "slice": [None, [0, 1]], "value": 5.0}, project)
    assert np.all(zero_then_project.child["fc2.weight"][:, 0] == 0.0)


def test_float64_accumulation_is_cast_to_float32_once() -> None:
    params = {"w": np.ones((2, 2), dtype=np.float32)}
    tiny = {"op": "add", "tensor": "w", "index": [[0, 0]], "values": [1e-8]}
    assert np.float32(1.0 + 1e-8) == np.float32(1.0)
    single = _apply(params, tiny)
    assert single.n_changed == 0 and single.child_sha256 == params_sha256(params)
    repeated = _apply(params, *[tiny] * 64)
    expected = 1.0
    for _ in range(64):
        expected += 1e-8
    assert repeated.child["w"][0, 0] == np.float32(expected) != np.float32(1.0)
    assert repeated.n_changed == 1


def test_application_is_deterministic_and_independent_of_key_order(tmp_path) -> None:
    params = _model()
    np.save(tmp_path / "u.npy", np.random.default_rng(1).normal(size=(5, 1)))
    np.save(tmp_path / "v.npy", np.random.default_rng(2).normal(size=(12, 1)))
    ops = (
        {"op": "project_out", "tensor": "fc1.weight", "side": "in", "direction": list(range(1, 13)), "strength": 0.5},
        {"op": "low_rank_add", "tensor": "fc1.weight", "u": "u.npy", "v": "v.npy"},
    )
    first = _apply(params, *ops, patch_dir=tmp_path)
    second = _apply(params, *ops, patch_dir=tmp_path)
    reordered = dict(reversed(list(params.items())))
    third = apply_patch(reordered, _patch(params, *ops), tmp_path, OPEN, params_sha256(params))
    assert first.child_sha256 == second.child_sha256 == third.child_sha256
    assert list(third.child) == list(reordered)


def test_child_is_float32_read_only_and_survives_save_and_reload(tmp_path) -> None:
    params = _model()
    result = _apply(params, {"op": "scale_rows", "tensor": "conv.weight", "rows": [1], "factor": 0.5},
                    {"op": "add", "tensor": "fc2.bias", "index": [[0]], "values": [0.25]})
    assert list(result.child) == list(params)
    for name, value in result.child.items():
        assert value.dtype == np.float32 and value.shape == params[name].shape
        assert value.flags.c_contiguous and not value.flags.writeable
    assert result.child_sha256 == params_sha256(result.child)
    digest = save_params(tmp_path / "child.npz", result.child)
    reloaded = load_params(tmp_path / "child.npz")
    assert digest == result.child_sha256 == params_sha256(reloaded)
    for name, value in result.child.items():
        assert reloaded[name].tobytes() == value.tobytes()
    params["fc2.bias"][0] = 100.0
    assert result.child["fc2.bias"][0] != 100.0


def _example_blocks(text: str) -> list[dict]:
    return [json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", text, flags=re.DOTALL)]


def _example_model() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    shapes = {"fc1.weight": (128, 784), "fc1.bias": (128,), "fc2.weight": (128, 128), "fc2.bias": (128,),
              "fc3.weight": (10, 128), "fc3.bias": (10,)}
    return {name: rng.normal(size=shape).astype(np.float32) for name, shape in shapes.items()}


def _write_example_files(op: dict, shapes: dict[str, tuple[int, ...]], directory) -> None:
    rng = np.random.default_rng(4)
    shape = shapes[op["tensor"]]
    rows, cols = shape[0], math.prod(shape[1:])
    wanted = {}
    if op["op"] == "project_out" and isinstance(op["direction"], str):
        wanted[op["direction"]] = (cols,) if op["side"] == "in" else (rows,)
    if op["op"] == "low_rank_add":
        wanted[op["u"]], wanted[op["v"]] = (rows, 2), (cols, 2)
    if op["op"] == "add_dense":
        wanted[op["delta"]] = shape
    if op["op"] == "tie" and isinstance(op["groups"], str):
        length = rows if op["axis"] == 0 else cols
        np.save(directory / op["groups"], rng.integers(0, max(1, length // 4), size=length))
    assert "values_file" not in op, "extend the helper for values_file examples"
    for name, file_shape in wanted.items():
        np.save(directory / name, rng.normal(size=file_shape))


def test_describe_ops_examples_are_valid_patches(tmp_path) -> None:
    text = describe_ops()
    assert text.isascii(), "the prompt text must be plain ASCII (this also excludes the em dash)"
    blocks = _example_blocks(text)
    full, op_examples = blocks[0], blocks[1:]
    params = _example_model()
    shapes = {name: value.shape for name, value in params.items()}
    assert full["schema"] == SCHEMA and set(full) == {"schema", "parent_sha256", "ops", "hypothesis", "predicted_effect"}
    full["parent_sha256"] = params_sha256(params)
    apply_patch(params, full, None, Limits(), params_sha256(params))
    assert sorted({op["op"] for op in op_examples}) == sorted(OP_NAMES)
    for op in op_examples:
        _write_example_files(op, shapes, tmp_path)
        result = _apply(params, op, patch_dir=tmp_path)
        assert result.n_changed > 0, op


def test_describe_ops_renders_the_limits_it_is_given() -> None:
    text = describe_ops(Limits(max_changed_frac=None, max_rel_frob=None, max_edit_frac=None, max_ops=7, max_rank=3,
                               max_file_bytes=1000))
    assert "at most 7 ops" in text and "1 <= r <= 3" in text and "1000 bytes" in text
    assert "no limit on the edit size" in text and "number of changed entries is not limited" in text
    default = describe_ops()
    assert "at most 64 ops" in default and "1 <= r <= 8" in default and "64 MiB" in default
    assert "<= 0.5" in default and "1-D tensors" in default and "at most 0.05 times the" in default
    assert "number of changed entries is not limited" in default and "tie, the length of the group" in default
    assert "every tensor of rank >= 2 needs" in default


def test_describe_ops_names_the_tensors_exempt_from_the_norm_rule() -> None:
    assert "does not apply to" not in describe_ops()
    text = describe_ops(Limits(rel_frob_exempt=["fc2.weight", "fc2.bias"]))
    assert "This rule does not apply to `fc2.weight`, `fc2.bias`." in text


@pytest.mark.parametrize(("floor", "exempt"), [(1, None), (2, "1-D tensors"), (3, "tensors of rank below 3")])
def test_describe_ops_states_the_rank_floor_of_the_norm_rule(floor: int, exempt: str | None) -> None:
    text = describe_ops(Limits(rel_frob_min_ndim=floor))
    assert f"every tensor of rank >= {floor} needs" in text
    if exempt is None:
        assert "not limited by this rule" not in text
    else:
        assert exempt in text and "not limited by this rule" in text


def test_patch_error_is_a_value_error_with_messages() -> None:
    error = PatchError(["first", "second"])
    assert isinstance(error, ValueError) and error.messages == ["first", "second"] and str(error) == "first\nsecond"
    with pytest.raises(ValueError):
        PatchError([])
