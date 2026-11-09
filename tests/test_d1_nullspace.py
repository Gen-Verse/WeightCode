"""PREREG D1's null-space projection (``weightcode.analysis.nullspace``) on a ViT-Lite-7/4 student, and the scope and grid
rules of ``scripts/e11/d1_nullspace.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from test_e11_refs import MEAN, STD, _images, _student

from weightcode.analysis import nullspace as N
from weightcode.models import vitlite as V
from weightcode.models.forward_gen import render_vitlite_forward

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "e11" / "d1_nullspace.py"


def _module(tmp_path: Path):
    path = tmp_path / "forward.py"
    path.write_text(render_vitlite_forward(MEAN, STD, 10, V.LN_EPS))
    return N.forward_module(path)


def _script():
    spec = importlib.util.spec_from_file_location("d1_nullspace_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_old_basis_takes_the_fewest_vectors_reaching_the_fraction() -> None:
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    values = np.array([5.0, 3.0, 1.0, 0.5, 0.4, 0.1])  # cumulative 0.5, 0.8, 0.9, 0.95, 0.99, 1.0
    moment = q @ np.diag(values) @ q.T
    for fraction, k in ((0.5, 1), (0.6, 2), (0.9, 3), (0.91, 4), (0.99, 5), (0.999, 6)):
        ob = N.old_basis("w", moment, fraction, 10)
        assert ob.k == k and ob.captured >= fraction - 1e-12
        assert np.allclose(np.abs(ob.basis.T @ q[:, :k]), np.eye(k), atol=1e-8)
    with pytest.raises(N.NullspaceError):
        N.old_basis("w", moment, 1.0, 10)


def test_projection_removes_exactly_the_old_directions() -> None:
    rng = np.random.default_rng(1)
    basis, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    delta = rng.normal(size=(5, 3, 2, 2))  # a conv-shaped weight, input side 3 * 2 * 2 = 12
    projected = N.project_input_side(delta, basis)
    flat, pflat = delta.reshape(5, -1), projected.reshape(5, -1)
    assert projected.shape == delta.shape
    assert np.allclose(pflat @ basis, 0.0, atol=1e-12)
    complement = np.eye(12) - basis @ basis.T
    assert np.allclose(pflat, flat @ complement, atol=1e-12)
    assert np.allclose(N.project_input_side(projected, basis), projected, atol=1e-12)  # idempotent


def test_layer_inputs_replay_the_forward_pass(tmp_path: Path) -> None:
    params = _student(3)
    module = _module(tmp_path)
    x = _images(5, 3)
    tensors = [N.PATCH_WEIGHT, "classifier.blocks.2.linear1.weight", "classifier.blocks.2.linear2.weight"]
    inputs, parity = N.layer_inputs(module, params, x, tensors)
    assert parity <= N.PARITY_TOL
    assert inputs[N.PATCH_WEIGHT].shape == (5 * module.N_PATCHES, 48)
    assert inputs["classifier.blocks.2.linear1.weight"].shape == (5 * module.N_TOKENS, 256)
    assert inputs["classifier.blocks.2.linear2.weight"].shape == (5 * module.N_TOKENS, 512)
    # The patch rows times the flattened projection plus bias are the patch tokens before the class token.
    w = params[N.PATCH_WEIGHT].astype(np.float64).reshape(256, -1)
    tokens = inputs[N.PATCH_WEIGHT] @ w.T + params["tokenizer.conv_layers.0.0.bias"]
    h = x.astype(np.float64) / 255.0
    h = (h - np.asarray(MEAN).reshape(1, -1, 1, 1)) / np.asarray(STD).reshape(1, -1, 1, 1)
    expected = module._linear(module._patches(h), w, params["tokenizer.conv_layers.0.0.bias"].astype(np.float64))
    assert np.allclose(tokens, expected.reshape(-1, 256), atol=1e-10)
    with pytest.raises(N.NullspaceError, match="patch projection and MLP weights only"):
        N.layer_inputs(module, params, x, ["classifier.blocks.0.self_attn.qkv.weight"])
    chunked, _, _ = N.moments(module, params, x, tensors, chunk=2)
    for t in tensors:
        assert np.allclose(chunked[t], N.second_moment(inputs[t]), atol=1e-10)


def test_projected_child_changes_only_the_scope_and_shrinks_the_old_response(tmp_path: Path) -> None:
    params = _student(4)
    module = _module(tmp_path)
    rng = np.random.default_rng(4)
    child = {k: v.copy() for k, v in params.items()}
    edited = [N.PATCH_WEIGHT, "classifier.blocks.2.linear1.weight", "classifier.blocks.2.linear2.weight"]
    for t in edited + ["classifier.blocks.2.linear1.bias"]:
        child[t] = (params[t] + 0.05 * rng.normal(size=params[t].shape)).astype(np.float32)
    assert N.edited_weight_tensors(params, child) == sorted(edited)
    x_old = _images(6, 4)
    moments, rows, _ = N.moments(module, params, x_old, edited)
    scope = [N.PATCH_WEIGHT, "classifier.blocks.2.linear1.weight"]
    bases = {t: N.old_basis(t, moments[t], 0.99, rows[t]) for t in scope}
    out, record = N.projected_child(params, child, bases)
    for key in params:
        if key in scope:
            assert not np.array_equal(out[key], child[key]) and out[key].dtype == child[key].dtype
        else:
            assert np.array_equal(out[key], child[key])  # out of scope and 1-D deltas unchanged
    inputs, _ = N.layer_inputs(module, params, x_old, scope)
    for t in scope:
        before = child[t].astype(np.float64) - params[t].astype(np.float64)
        after = out[t].astype(np.float64) - params[t].astype(np.float64)
        x = inputs[t]
        r_before = np.linalg.norm(x @ before.reshape(before.shape[0], -1).T)
        r_after = np.linalg.norm(x @ after.reshape(after.shape[0], -1).T)
        assert r_after < r_before
        bound = np.sqrt((1 - bases[t].captured) * np.trace(moments[t]) * len(x)) * np.linalg.norm(before.reshape(
            before.shape[0], -1), 2)
        assert r_after <= bound * (1 + 1e-4) + 1e-4  # float32 storage of the projected weight
        assert record[t]["k"] == bases[t].k and 0 < record[t]["kept_frob_ratio"] < 1


def test_gate_units_are_read_from_the_edit() -> None:
    script = _script()
    params = _student(5)
    child = {k: v.copy() for k, v in params.items()}
    for block, units in ((3, (7, 11)), (6, (2,))):
        for u in units:
            child[f"classifier.blocks.{block}.linear1.weight"][u] += 0.1
            child[f"classifier.blocks.{block}.linear2.weight"][:, u] -= 0.1
    edited = N.edited_weight_tensors(params, child)
    assert script.gate_units(params, child, edited) == {3: [7, 11], 6: [2]}
    child["classifier.blocks.6.linear2.weight"][0, 5] += 0.1  # a changed linear2 column is a changed unit
    assert script.gate_units(params, child, edited) == {3: [7, 11], 6: [2, 5]}
    wide = {k: v.copy() for k, v in params.items()}
    wide["classifier.blocks.1.linear1.weight"][:9] += 0.1
    with pytest.raises(script.D1Error, match="not a freed-unit edit"):
        script.gate_units(params, wide, N.edited_weight_tensors(params, wide))


def test_scopes_and_grids_are_the_prereg_ones() -> None:
    script = _script()
    contrast, gaussian = script.FINALS["contrast"], script.FINALS["gaussian"]
    c_edited = ["classifier.blocks.2.linear1.weight", "classifier.blocks.2.linear2.weight", N.PATCH_WEIGHT]
    scopes = script.scope_tensors(contrast, c_edited, {2: [1, 2]})
    assert scopes == {"s1": c_edited, "s2": [N.PATCH_WEIGHT]}
    assert script.wrapper_configs(contrast, scopes) == [{"scope": s, "e": e} for s in ("s1", "s2")
                                                         for e in (0.90, 0.99, 0.999)]
    g_edited = [f"classifier.blocks.{b}.{p}.weight" for b in (3, 4, 6) for p in ("linear1", "linear2")]
    g_scopes = script.scope_tensors(gaussian, g_edited, {3: [0, 1], 4: [5, 6], 6: [8, 9]})
    assert g_scopes["s2"] == [] and script.wrapper_configs(gaussian, g_scopes) == [
        {"scope": "s1", "e": e} for e in (0.90, 0.99, 0.999)]
    with pytest.raises(script.D1Error, match="PREREG D1 names"):
        script.scope_tensors(gaussian, g_edited + [N.PATCH_WEIGHT], {3: [0], 4: [5], 6: [8]})
    with pytest.raises(script.D1Error, match="not all edited"):
        script.scope_tensors(contrast, [N.PATCH_WEIGHT], {2: [1]})


def test_summarize_reads_both_record_shapes() -> None:
    script = _script()
    part = {"delta_acc": 0.1, "worst_class_delta": -0.02, "fixed": 5, "broken": 1}
    clean = {"delta_acc": -0.005, "worst_class_delta": -0.03, "fixed": 2, "broken": 4}
    joint = {"readouts": part, "readouts_clean": clean, "n": 4000, "n_clean": 4000, "clean_split_sha256": "a"}
    transfer = {"readouts": part, "readouts_clean": clean, "n_select": 4000, "n_clean": 4000,
                "clean_split_sha256": "a", "schema": "weightcode.harness.program_transfer.v0"}
    for record in (joint, transfer):
        v = script.summarize(record)
        assert v["gain"] == pytest.approx(10.0) and v["clean"] == pytest.approx(-0.5)
        assert v["joint"] == pytest.approx(9.5) and v["n"] == 4000 and v["worst_class_clean"] == pytest.approx(-3.0)
