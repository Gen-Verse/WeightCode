"""Tests for the shared delta-patch writer and oracle factory (weightcode.baselines.instance_io)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from weightcode.baselines.instance_io import INDEX_FORM_MAX_ENTRIES, write_delta_patch
from weightcode.common.identity import params_sha256
from weightcode.patch import MAX_PATCH_JSON_BYTES, Limits, apply_patch_file


def _base(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "fc1.weight": rng.standard_normal((128, 2352)).astype(np.float32),
        "fc1.bias": rng.standard_normal(128).astype(np.float32),
        "fc2.weight": rng.standard_normal((128, 400)).astype(np.float32),
    }


def _ops(path: Path) -> dict[str, str]:
    return {op["tensor"]: op["op"] for op in json.loads(path.read_text())["ops"]}


def _applied(base: dict[str, np.ndarray], path: Path) -> tuple[dict[str, np.ndarray], int]:
    result = apply_patch_file(base, path, Limits().without_size_limits(), params_sha256(base))
    return result.child, result.edit_size


def test_a_16_row_scale_stays_index_form_and_under_the_cap(tmp_path: Path) -> None:
    base = _base(np.random.default_rng(0))
    child = {k: v.copy() for k, v in base.items()}
    child["fc1.weight"][:16] *= np.float32(1.1)
    path = write_delta_patch(tmp_path, params_sha256(base), base, child, "h", "e")
    assert path.stat().st_size <= MAX_PATCH_JSON_BYTES
    assert _ops(path) == {"fc1.weight": "add"}
    rebuilt, edit_size = _applied(base, path)
    assert edit_size == 16 * 2352
    np.testing.assert_array_equal(rebuilt["fc1.weight"], child["fc1.weight"])


def test_patches_over_the_cap_move_the_largest_tensor_to_dense(tmp_path: Path) -> None:
    base = _base(np.random.default_rng(1))
    child = {k: v.copy() for k, v in base.items()}
    child["fc1.weight"].flat[: INDEX_FORM_MAX_ENTRIES - 1] += np.float32(0.01)
    child["fc2.weight"].flat[: 39_000] += np.float32(0.01)
    child["fc2.weight"].flat[0] += np.float32(0.02)
    path = write_delta_patch(tmp_path, params_sha256(base), base, child, "h", "e")
    assert path.stat().st_size <= MAX_PATCH_JSON_BYTES
    assert _ops(path) == {"fc1.weight": "add_dense", "fc2.weight": "add"}
    assert (tmp_path / "d_fc1_weight.npy").is_file()
    rebuilt, _ = _applied(base, path)
    for key in base:
        np.testing.assert_allclose(rebuilt[key], child[key], rtol=0, atol=1e-6)


def test_full_tensor_changes_are_dense_and_untouched_tensors_get_no_op(tmp_path: Path) -> None:
    base = _base(np.random.default_rng(2))
    child = {k: v.copy() for k, v in base.items()}
    child["fc1.bias"] += np.float32(0.5)
    path = write_delta_patch(tmp_path, params_sha256(base), base, child, "h", "e")
    assert _ops(path) == {"fc1.bias": "add_dense"}
    rebuilt, edit_size = _applied(base, path)
    assert edit_size == 128
    np.testing.assert_array_equal(rebuilt["fc1.bias"], child["fc1.bias"])
    with pytest.raises(KeyError):
        _ops(path)["fc2.weight"]
