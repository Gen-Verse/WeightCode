"""PREREG F4 (``weightcode.baselines.fit.f4``): the continual fine-tuning arms on the ViT-Lite and ResNet-20 students."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_e10_refs import _arch as _resnet_arch
from test_e10_refs import _forward as _resnet_forward
from test_e10_refs import _images as _resnet_images
from test_e10_refs import _student as _resnet_student
from test_e11_refs import _instance, _student
from test_ft_compare import _joint_evidence

from weightcode.baselines.fit import checked_reference_stream as CRS
from weightcode.baselines.fit import f4 as F
from weightcode.baselines.fit import ftcompare as FT
from weightcode.baselines.fit.runner import Evidence
from weightcode.baselines.fit.torch_forward import architecture_from, images_tensor


def test_grids_are_the_prereg_grids() -> None:
    assert [(c["lr"], c["steps"]) for c in F.f4_configs("f4_ft_replay")] == [(lr, s) for lr in (1e-5, 1e-4) for s in (10, 30, 100, 300)]
    assert F.f4_configs("f4_er") == F.f4_configs("f4_ft_replay")
    assert [(c["lr"], c["steps"]) for c in F.f4_configs("f4_lora")] == [(lr, s) for lr in (1e-4, 1e-3) for s in (10, 30, 100, 300)]
    assert [(c["weight"], c["lr"], c["steps"]) for c in F.f4_configs("f4_lwf")] == [
        (w, lr, s) for w in (1.0, 10.0) for lr in (1e-5, 1e-4) for s in (30, 300)]
    assert not any(spec.edit_limits_enforced for spec in F.F4_SPECS.values())
    assert set(F.F4_METHODS) <= set(CRS.EXTRA_SPECS) and CRS.method_spec("f4_lora") is F.F4_SPECS["f4_lora"]


def _stream_evidence(n_new: int = 64, n_earlier: int = 40) -> Evidence:
    """Round rows, earlier rounds' rows, then 64 replay rows (control (b)'s fit_evidence layout)."""
    ev = _joint_evidence(3, n_replay=n_earlier + F.F4_REPLAY_ROWS)
    return ev


def test_fit_rows_select_per_arm() -> None:
    ev = _stream_evidence()
    for method in ("f4_ft_replay", "f4_lora", "f4_lwf"):
        x, y, k, text = F.fit_rows(method, ev)
        assert len(y) == 64 + 64 and k == 64 and np.array_equal(x[:64], ev.x[:64]) and np.array_equal(x[64:], ev.x[-64:])
        assert "40 earlier rows dropped" in text
    x, y, k, _ = F.fit_rows("f4_er", ev)
    assert len(y) == len(ev.y) == 64 + 40 + 64 and k == 64 + 40 and np.array_equal(x, ev.x)
    with pytest.raises(F.F4Error):
        F.fit_rows("f4_er", _joint_evidence(3, n_replay=10))


def test_lora_nd_equals_f1_lora_on_vit() -> None:
    params = _student(5)
    inst_arch = architecture_from(params, __import__("test_e11_refs")._spec(params))
    ev = _joint_evidence(5)
    x, y = images_tensor(ev.x), torch.from_numpy(ev.y)
    targets = FT.lora_targets(inst_arch, params)
    a = FT.lora_checkpoints(inst_arch, params, x, y, targets=targets, rank=4, lr=1e-3, checkpoints=(1, 3), seed=0)
    b = F.lora_nd_checkpoints(inst_arch, params, x, y, targets=targets, rank=4, lr=1e-3, checkpoints=(1, 3), seed=0)
    for (sa, pa, _), (sb, pb, _) in zip(a, b, strict=True):
        assert sa == sb and all(np.array_equal(pa[k], pb[k]) for k in params)


def test_resnet_lora_moves_only_convs_and_fc() -> None:
    params = _resnet_student(2)
    arch = _resnet_arch(params)
    targets = F.lora_targets_for(arch, params)
    assert "fc.weight" in targets and all(np.ndim(params[t]) in (2, 4) for t in targets) and len(targets) == 20
    x = images_tensor(_resnet_images(64, 1).reshape(64, -1))
    y = torch.from_numpy(np.random.default_rng(1).integers(0, 10, 64).astype(np.int64))
    snaps = F.lora_nd_checkpoints(arch, params, x, y, targets=targets, rank=4, lr=1e-3, checkpoints=(3,), seed=0)
    child = snaps[0][1]
    for k in params:
        same = np.array_equal(child[k], params[k])
        assert same == (k not in targets), k


@pytest.mark.parametrize("method", list(F.F4_METHODS))
def test_generator_on_vit(tmp_path, monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    monkeypatch.setattr(F, "STEP_CHECKPOINTS", (1, 2, 3, 4))
    monkeypatch.setattr(F, "REG_STEPS", (1, 3))
    params = _student(7)
    inst = _instance(tmp_path, params)
    cands, info = F.f4_generator(method, inst, 0)(_stream_evidence())
    assert [c.config for c in cands] == F.f4_configs(method)
    assert all(np.all(np.isfinite(v)) for c in cands for v in c.child.values())
    assert any(not np.array_equal(c.child[k], params[k]) for c in cands for k in params)
    assert ("earlier rows dropped" in info["arm_fit_rows"]) == (method != "f4_er")


def test_generator_on_resnet(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from weightcode.models import resnet20a as A

    monkeypatch.setattr(F, "STEP_CHECKPOINTS", (1, 2))
    params = _resnet_student(4)
    (tmp_path / "model_spec.json").write_text(json.dumps(
        A.build_resnet20a_spec("m00000000", [f"c{i}" for i in range(10)], [0.4909, 0.4818, 0.4460],
                               [0.2471, 0.2435, 0.2616], params)))
    inst = SimpleNamespace(instance_dir=tmp_path, params=params, forward=_resnet_forward(), private={})
    n = 64 + 64
    ev = Evidence(x=_resnet_images(n, 3), y=np.random.default_rng(3).integers(0, 10, n).astype(np.int64),
                  n_fail=16, n_rand=48, sha256="0" * 64, base_pred=np.zeros(n, dtype=np.int64), rule="evidence-v1",
                  n_replay=64, replay_sha256="1" * 64)
    for method in ("f4_ft_replay", "f4_lora"):
        monkeypatch.setattr(F, "LORA_LRS", (1e-4, 1e-3))
        cands, _ = F.f4_generator(method, inst, 0)(ev)
        assert len(cands) == 4
        if method == "f4_lora":
            assert all(np.array_equal(c.child["bn1.running_mean"], params["bn1.running_mean"]) for c in cands)
