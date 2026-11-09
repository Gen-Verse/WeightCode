"""The ``tie`` op of weightcode.patch against hand-computed references (contract 5.0, HARNESS_SPEC section 2)."""

from __future__ import annotations

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
    shapes = {"conv.weight": (4, 3, 2, 2), "conv.bias": (4,), "fc1.weight": (5, 12), "fc1.bias": (5,)}
    return {name: rng.normal(size=shape).astype(np.float32) for name, shape in shapes.items()}


def _apply(params, *ops, patch_dir=None, limits=OPEN):
    patch = {
        "schema": SCHEMA,
        "parent_sha256": params_sha256(params),
        "ops": list(ops),
        "hypothesis": "test hypothesis",
        "predicted_effect": "test prediction",
    }
    return apply_patch(params, patch, patch_dir, limits, params_sha256(params))


def _reference(matrix: np.ndarray, axis: int, groups: list[int], s: float) -> np.ndarray:
    """Plain-loop reference: every position moves a fraction s toward the mean of its group."""
    work = matrix.astype(np.float64)
    view = work if axis == 1 else work.T
    out = view.copy()
    labels = np.array(groups)
    for position in range(view.shape[1]):
        members = np.flatnonzero(labels == labels[position])
        mean = view[:, members].mean(axis=1)
        out[:, position] = view[:, position] + s * (mean - view[:, position])
    return (out if axis == 1 else out.T).astype(np.float32)


@pytest.mark.parametrize("s", [0.25, 1.0])
def test_tie_columns_of_a_linear_weight_matches_reference(s: float) -> None:
    params = _model()
    groups = [0, 0, 0, 3, 3, 5, 6, 6, 8, 9, 6, 11]
    result = _apply(params, {"op": "tie", "tensor": "fc1.weight", "axis": 1, "groups": groups, "strength": s})
    np.testing.assert_allclose(result.child["fc1.weight"], _reference(params["fc1.weight"], 1, groups, s), rtol=0,
                               atol=1e-6)
    for column in (5, 8, 9, 11):  # alone in their group: bitwise unchanged
        assert result.child["fc1.weight"][:, column].tobytes() == params["fc1.weight"][:, column].tobytes()
    assert result.edit_size == 12
    for name in ("conv.weight", "conv.bias", "fc1.bias"):
        assert result.child[name].tobytes() == params[name].tobytes()


def test_full_strength_makes_group_members_equal_to_their_mean() -> None:
    params = _model(1)
    groups = [1, 1, 2, 2]
    result = _apply(params, {"op": "tie", "tensor": "conv.weight", "axis": 0, "groups": groups, "strength": 1.0})
    child, parent = result.child["conv.weight"], params["conv.weight"].astype(np.float64)
    np.testing.assert_allclose(child[0], (parent[0] + parent[1]) / 2, atol=1e-6, rtol=0)
    assert child[0].tobytes() == child[1].tobytes() and child[2].tobytes() == child[3].tobytes()
    assert result.edit_size == 4


def test_tie_columns_of_a_conv_weight_use_the_flattened_in_kh_kw_order() -> None:
    params = _model(2)
    # Tie the 3 input channels at each of the 4 kernel positions: column j = c * 4 + k belongs to group k.
    groups = [j % 4 for j in range(12)]
    result = _apply(params, {"op": "tie", "tensor": "conv.weight", "axis": 1, "groups": groups, "strength": 1.0})
    child = result.child["conv.weight"]
    for c in (1, 2):
        np.testing.assert_array_equal(child[:, c], child[:, 0])
    np.testing.assert_allclose(child[:, 0], params["conv.weight"].astype(np.float64).mean(axis=1), atol=1e-6, rtol=0)


def test_tie_on_a_bias_and_group_ids_are_labels_only() -> None:
    params = _model(3)
    first = _apply(params, {"op": "tie", "tensor": "fc1.bias", "axis": 0, "groups": [0, 0, 2, 3, 0], "strength": 0.5})
    relabelled = _apply(params, {"op": "tie", "tensor": "fc1.bias", "axis": 0, "groups": [4, 4, 1, 0, 4],
                                 "strength": 0.5})
    assert first.child_sha256 == relabelled.child_sha256
    np.testing.assert_allclose(first.child["fc1.bias"], _reference(params["fc1.bias"][:, None], 0, [0, 0, 2, 3, 0],
                                                                   0.5)[:, 0], atol=1e-6, rtol=0)


def test_groups_from_a_file_match_inline_groups(tmp_path) -> None:
    params = _model(4)
    groups = [j // 3 for j in range(12)]
    np.save(tmp_path / "g_int.npy", np.array(groups, dtype=np.int32))
    np.save(tmp_path / "g_float.npy", np.array(groups, dtype=np.float64))
    inline = _apply(params, {"op": "tie", "tensor": "fc1.weight", "axis": 1, "groups": groups, "strength": 0.7})
    for name in ("g_int.npy", "g_float.npy"):
        from_file = _apply(params, {"op": "tie", "tensor": "fc1.weight", "axis": 1, "groups": name, "strength": 0.7},
                           patch_dir=tmp_path)
        assert from_file.child_sha256 == inline.child_sha256


def test_tie_is_deterministic() -> None:
    params = _model(5)
    op = {"op": "tie", "tensor": "conv.weight", "axis": 1, "groups": [j % 3 for j in range(12)], "strength": 0.3}
    assert _apply(params, op).child_sha256 == _apply(params, op).child_sha256


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"axis": 2}, "axis must be 0 or 1"),
        ({"axis": True}, "axis must be 0 or 1"),
        ({"strength": 0.0}, "outside (0, 1]"),
        ({"strength": 1.5}, "outside (0, 1]"),
        ({"groups": [0] * 11}, "shape [12]"),
        ({"groups": [0] * 11 + [12]}, "outside [0, 12)"),
        ({"groups": [0] * 11 + [-1]}, "outside [0, 12)"),
        ({"groups": [0] * 11 + [1.5]}, "must be integers"),
        ({"groups": [0] * 11 + [True]}, "must be integers"),
        ({"groups": {"a": 1}}, "file name or a list of integers"),
        ({"extra": 1}, "unknown field"),
    ],
)
def test_tie_validation_errors(change: dict, message: str) -> None:
    params = _model()
    op = {"op": "tie", "tensor": "fc1.weight", "axis": 1, "groups": [0] * 12, "strength": 0.5, **change}
    with pytest.raises(PatchError) as raised:
        _apply(params, op)
    assert any(message in text for text in raised.value.messages), raised.value.messages


def test_axis_one_needs_rank_two() -> None:
    with pytest.raises(PatchError, match="axis 1 needs rank >= 2"):
        _apply(_model(), {"op": "tie", "tensor": "fc1.bias", "axis": 1, "groups": [0], "strength": 0.5})


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([0.0] * 11 + [0.5]), "non-integer"),
        (np.array([0] * 11 + [12]), "outside [0, 12)"),
        (np.zeros((12, 1), dtype=np.int64), "shape [12]"),
    ],
)
def test_tie_group_file_errors(tmp_path, values: np.ndarray, message: str) -> None:
    np.save(tmp_path / "g.npy", values)
    with pytest.raises(PatchError) as raised:
        _apply(_model(), {"op": "tie", "tensor": "fc1.weight", "axis": 1, "groups": "g.npy", "strength": 0.5},
               patch_dir=tmp_path)
    assert any(message in text for text in raised.value.messages), raised.value.messages


def test_tie_changes_count_toward_the_changed_fraction_and_relative_norm() -> None:
    params = _model(6)
    op = {"op": "tie", "tensor": "fc1.weight", "axis": 1, "groups": [0] * 12, "strength": 1.0}
    with pytest.raises(PatchError, match="relative Frobenius norm"):
        _apply(params, op, limits=Limits(max_changed_frac=None, max_rel_frob=0.1, max_edit_frac=None))
    with pytest.raises(PatchError, match="parameters change"):
        _apply(params, op, limits=Limits(max_changed_frac=0.05, max_rel_frob=None, max_edit_frac=None))


def test_tie_edit_size_is_the_group_map_length_under_the_default_limits() -> None:
    params = _model(7)  # 117 parameter entries, so the default cap is 0.05 x 117 = 5.85 numbers
    op = {"op": "tie", "tensor": "fc1.bias", "axis": 0, "groups": [0, 0, 1, 1, 2], "strength": 1.0}
    result = _apply(params, op, limits=Limits())
    assert result.edit_size == 5 and result.per_tensor["fc1.bias"]["changed"] == 4
    wide = {"op": "tie", "tensor": "fc1.weight", "axis": 1, "groups": [0] * 12, "strength": 0.1}
    with pytest.raises(PatchError, match="specifies 12 numbers"):
        _apply(params, wide, limits=Limits())
