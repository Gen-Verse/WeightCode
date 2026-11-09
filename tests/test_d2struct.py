"""PREREG D2 (``weightcode.baselines.fit.d2struct``): the program's structure trained by gradient on a ViT-Lite-7/4 student."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_e11_refs import _instance, _student
from test_ft_compare import _joint_evidence

from weightcode.baselines.fit import d2struct as D2
from weightcode.common.identity import params_sha256


def _final(params: dict[str, np.ndarray], seed: int = 3) -> dict[str, np.ndarray]:
    """A synthetic 'final': a few rows of two linear weights and one bias changed."""
    rng = np.random.default_rng(seed)
    child = {k: np.array(v, copy=True) for k, v in params.items()}
    w1, w2, b2 = "classifier.blocks.2.linear1.weight", "classifier.blocks.2.linear2.weight", "classifier.blocks.2.linear2.bias"
    child[w1][:3] += rng.normal(0, 0.1, child[w1][:3].shape).astype(np.float32)
    child[w2][:, :3] += rng.normal(0, 0.1, child[w2][:, :3].shape).astype(np.float32)
    child[b2] += np.float32(0.01)
    return child


def _short(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D2, "STEP_CHECKPOINTS", (1, 2, 3, 4))


def test_grid_is_f1_full_ft_grid() -> None:
    assert D2.d2_configs() == [{"lr": lr, "steps": s} for lr in (1e-5, 1e-4) for s in (10, 30, 100, 300)]
    assert D2.D2_METHODS == ("d2_struct_base", "d2_struct_final")
    assert not any(spec.edit_limits_enforced for spec in D2.D2_SPECS.values())


def test_structure_mask_is_the_changed_entries() -> None:
    params = _student(7)
    child = _final(params)
    trainable, masks, record = D2.structure_mask(params, child)
    assert trainable == [k for k in params if k in trainable]  # forward order
    assert set(trainable) == {"classifier.blocks.2.linear1.weight", "classifier.blocks.2.linear2.weight",
                              "classifier.blocks.2.linear2.bias"}
    expected = sum(int((child[k] != params[k]).sum()) for k in params)
    assert record["n_entries"] == expected == sum(int(m.sum()) for m in masks.values())
    with pytest.raises(D2.D2Error, match="changes no entry"):
        D2.structure_mask(params, params)


@pytest.mark.parametrize("method", ["d2_struct_base", "d2_struct_final"])
def test_candidates_move_only_masked_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    _short(monkeypatch)
    params = _student(7)
    child = _final(params)
    instance = _instance(tmp_path, params)
    monkeypatch.setattr(D2, "final_child", lambda inst, record: (child, {"child_sha256": params_sha256(child)}))
    candidates, info = D2.d2_generator(instance, method, tmp_path / "t.json", 0)(_joint_evidence(7))
    assert [c.config for c in candidates] == [{"lr": lr, "steps": s} for lr in (1e-5, 1e-4) for s in (1, 2, 3, 4)]
    _, masks, record = D2.structure_mask(params, child)
    assert info["mask"]["n_entries"] == record["n_entries"]
    start = params if method == "d2_struct_base" else child
    for c in candidates:
        moved = False
        for k in params:
            if k in masks:
                assert np.array_equal(c.child[k][~masks[k]], start[k][~masks[k]])
                moved |= not np.array_equal(c.child[k][masks[k]], start[k][masks[k]])
            else:
                assert np.array_equal(c.child[k], params[k])
        assert moved
    if method == "d2_struct_final":  # start (ii) keeps the final's values outside training: base elsewhere
        for k in masks:
            assert not np.array_equal(candidates[0].child[k], params[k])


def test_final_child_checks_the_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    params = _student(7)
    child = _final(params)
    instance = _instance(tmp_path, params)
    instance.instance_id, instance.params_sha256 = "i0", params_sha256(params)
    record_path = tmp_path / "transfer_x_i0.json"
    (tmp_path / "transfer_x_i0" / "patch").mkdir(parents=True)
    (tmp_path / "transfer_x_i0" / "patch" / "derived_patch.json").write_text("{}")
    n_changed = int(sum((child[k] != params[k]).sum() for k in params))
    record = {"schema": D2.TRANSFER_SCHEMA, "status": "ok", "child_sha256": params_sha256(child), "n_changed": n_changed,
              "target": {"instance_id": instance.instance_id, "params_sha256": instance.params_sha256},
              "program": {"sha256": "ab"}}
    record_path.write_text(json.dumps(record))
    monkeypatch.setattr(D2, "load_patch", lambda path: {})
    monkeypatch.setattr(D2, "apply_patch", lambda *a: SimpleNamespace(
        child=child, child_sha256=params_sha256(child), n_changed=n_changed))
    rebuilt, source = D2.final_child(instance, record_path)
    assert params_sha256(rebuilt) == record["child_sha256"] and source["n_changed"] == n_changed
    record["child_sha256"] = "0" * 64
    record_path.write_text(json.dumps(record))
    with pytest.raises(D2.D2Error, match="differs"):
        D2.final_child(instance, record_path)
    record["target"]["instance_id"] = "iother"
    record_path.write_text(json.dumps(record))
    with pytest.raises(D2.D2Error, match="is not"):
        D2.final_child(instance, record_path)
