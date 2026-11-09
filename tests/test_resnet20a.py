"""E10's ResNet-20 (option A, linear head): layout, forward.py parity and the option-A shortcut."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from weightcode.models import resnet20a as A
from weightcode.models.forward_gen import render_resnet20a_forward

MEAN = [0.4909, 0.4818, 0.4460]
STD = [0.2471, 0.2435, 0.2616]


def _forward_module() -> dict:
    namespace: dict = {}
    exec(compile(render_resnet20a_forward(MEAN, STD, 10, A.BN_EPS), "forward.py", "exec"), namespace)
    return namespace


def _model_with_stats(seed: int = 0) -> A.ResNet20A:
    torch.manual_seed(seed)
    model = A.ResNet20A(10, MEAN, STD)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-0.5, 0.5)
                module.running_var.uniform_(0.5, 2.0)
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.2, 0.2)
    return model


def test_layout_matches_forward_and_paper_size() -> None:
    ns = _forward_module()
    shapes = A.param_shapes(10)
    assert {k: tuple(v) for k, v in ns["expected_shapes"]().items()} == shapes
    params = A.export_params(_model_with_stats())
    assert len(params) == 97 and sum(v.size for v in params.values()) == 271_098
    trainable = sum(p.numel() for p in A.ResNet20A(10, MEAN, STD).parameters())
    assert trainable == 269_722  # He et al.: 0.27M
    assert not any("downsample" in name or "shortcut" in name for name in params)


def test_option_a_shortcut() -> None:
    x = torch.arange(2 * 16 * 4 * 4, dtype=torch.float64).reshape(2, 16, 4, 4)
    out = A.Subsample(32)(x)
    assert out.shape == (2, 32, 2, 2)
    assert torch.equal(out[:, 8:24], x[:, :, ::2, ::2])
    assert torch.count_nonzero(out[:, :8]) == 0 and torch.count_nonzero(out[:, 24:]) == 0


def test_forward_parity_float64_and_float32() -> None:
    ns = _forward_module()
    params = A.export_params(_model_with_stats())
    x = np.random.default_rng(0).integers(0, 256, (40, 3, 32, 32), dtype=np.uint8)
    m64 = A.build_resnet20a(10, MEAN, STD, params, torch.float64)
    m32 = A.build_resnet20a(10, MEAN, STD, params, torch.float32)
    features = ns["features"](params, x)
    reference = m64.features(torch.from_numpy(x)).detach().numpy()
    assert np.max(np.abs(features - reference)) <= 1e-12 * np.max(np.abs(reference))
    logits = ns["forward"](params, x)
    assert logits.dtype == np.float32 and logits.shape == (40, 10)
    ref64 = A.eval_logits(m64, x)
    assert np.allclose(logits, ref64, rtol=1e-6, atol=1e-6)
    assert np.array_equal(logits.argmax(1), A.eval_logits(m32, x).argmax(1))
    with pytest.raises(KeyError):
        ns["forward"]({k: v for k, v in params.items() if k != "fc.bias"}, x)


def test_training_is_deterministic_on_cpu() -> None:
    x = np.random.default_rng(1).integers(0, 256, (64, 3, 32, 32), dtype=np.uint8)
    y = np.arange(64, dtype=np.int64) % 10
    config = A.HeTrainConfig(epochs=1, batch_size=32)
    runs = [A.export_params(A.train_resnet20a(config, 7, x, y, 10, MEAN, STD, torch.device("cpu")).model)
            for _ in range(2)]
    assert all(np.array_equal(runs[0][k], runs[1][k]) for k in runs[0])
