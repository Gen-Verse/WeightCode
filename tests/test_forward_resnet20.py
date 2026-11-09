"""The numpy ResNet-20 cosine-head forward (PREREG E9) against an independent torch rebuild of CEC's resnet20_cifar.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from weightcode.evalkit.patch_check import load_forward
from weightcode.models.forward_gen import render_resnet20_forward, write_forward
from weightcode.models.parity import check_logit_parity

MEAN = (0.507, 0.487, 0.441)
STD = (0.267, 0.256, 0.276)
SCALE = 16.0


class Block(nn.Module):
    """CEC BasicBlock: conv3x3-bn-relu-conv3x3-bn, plus the shortcut, then relu."""

    def __init__(self, inplanes: int, planes: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = x if self.downsample is None else self.downsample(x)
        return F.relu(out + residual)


class Net(nn.Module):
    """CEC resnet20 encoder (16/32/64, 3 blocks per stage) with the cosine head of CEC's MYNET."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        inplanes, stages = 16, []
        for planes, stride in ((16, 1), (32, 2), (64, 2)):
            stages.append(nn.Sequential(Block(inplanes, planes, stride), Block(planes, planes, 1), Block(planes, planes, 1)))
            inplanes = planes
        self.layer1, self.layer2, self.layer3 = stages
        self.fc = nn.Linear(64, n_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(MEAN, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(STD, dtype=x.dtype).view(1, 3, 1, 1)
        h = (x / 255.0 - mean) / std
        h = F.relu(self.bn1(self.conv1(h)))
        h = self.layer3(self.layer2(self.layer1(h)))
        f = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return SCALE * F.linear(F.normalize(f, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))


def _random_net(n_classes: int, seed: int) -> tuple[Net, dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    net = Net(n_classes)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                channels = module.num_features
                module.weight.copy_(0.5 + torch.rand(channels, generator=generator))
                module.bias.copy_(0.2 * torch.randn(channels, generator=generator))
                module.running_mean.copy_(0.3 * torch.randn(channels, generator=generator))
                module.running_var.copy_(0.5 + torch.rand(channels, generator=generator))
    net.eval()
    params = {
        name: value.detach().numpy().astype(np.float32)
        for name, value in net.state_dict().items()
        if not name.endswith("num_batches_tracked")
    }
    return net, params


def _forward(tmp_path, shape=(3, 32, 32), n_classes=100):
    source = render_resnet20_forward(MEAN, STD, n_classes, SCALE, 1e-5, 1e-12, input_shape=shape)
    path = tmp_path / "forward.py"
    write_forward(path, source)
    forward, _ = load_forward(path)
    return forward, path


def _torch_logits(net: Net, params: dict[str, np.ndarray], x: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    model = Net(params["fc.weight"].shape[0])
    model.load_state_dict({k: torch.from_numpy(v) for k, v in params.items()}, strict=False)
    model = model.to(dtype).eval()
    with torch.no_grad():
        return model(torch.from_numpy(x).to(dtype)).numpy()


@pytest.mark.parametrize("shape", [(3, 32, 32), (3, 8, 8)])
def test_parity_with_torch(tmp_path, shape):
    forward, _ = _forward(tmp_path, shape)
    net, params = _random_net(100, seed=3)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(7, *shape), dtype=np.uint8)
    mine = forward(params, x)
    assert mine.dtype == np.float32 and mine.shape == (7, 100)
    result = check_logit_parity(mine, _torch_logits(net, params, x, torch.float64),
                                _torch_logits(net, params, x, torch.float32))
    assert result["float64_max_rel_diff"] <= result["float64_tolerance"]


def test_parameter_inventory(tmp_path):
    forward, path = _forward(tmp_path)
    _, params = _random_net(100, seed=1)
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    expected = namespace["expected_shapes"]()
    assert {k: tuple(v.shape) for k, v in params.items()} == expected
    assert len(expected) == 106  # 21 convolutions, 21 batch norms x 4, the head
    assert sum(int(np.prod(s)) for s in expected.values()) == 279_792
    assert sum(int(np.prod(s)) for k, s in expected.items() if "running" in k) == 1_568


def test_refuses_wrong_names_shapes_and_inputs(tmp_path):
    forward, _ = _forward(tmp_path)
    _, params = _random_net(100, seed=2)
    x = np.zeros((2, 3, 32, 32), dtype=np.uint8)
    missing = {k: v for k, v in params.items() if k != "layer2.0.downsample.1.running_var"}
    with pytest.raises(KeyError):
        forward(missing, x)
    with pytest.raises(KeyError):
        forward({**params, "fc.bias": np.zeros(100, np.float32)}, x)
    with pytest.raises(ValueError):
        forward({**params, "fc.weight": np.zeros((10, 64), np.float32)}, x)
    with pytest.raises(TypeError):
        forward({**params, "conv1.weight": params["conv1.weight"].astype(np.int64)}, x)
    with pytest.raises(TypeError):
        forward(params, x.astype(np.float32))
    with pytest.raises(ValueError):
        forward(params, np.zeros((2, 3, 16, 16), dtype=np.uint8))
    empty = forward(params, np.zeros((0, 3, 32, 32), dtype=np.uint8))
    assert empty.shape == (0, 100) and empty.dtype == np.float32


def test_cosine_head(tmp_path):
    forward, path = _forward(tmp_path)
    _, params = _random_net(100, seed=4)
    x = np.random.default_rng(1).integers(0, 256, size=(5, 3, 32, 32), dtype=np.uint8)
    logits = forward(params, x)
    assert np.all(np.abs(logits) <= SCALE + 1e-5)
    # A positive rescaling of a class row leaves every logit unchanged.
    scaled = dict(params, **{"fc.weight": params["fc.weight"] * np.linspace(0.1, 7.0, 100, dtype=np.float32)[:, None]})
    np.testing.assert_allclose(forward(scaled, x), logits, rtol=0, atol=1e-5)
    # Each logit is SCALE times the cosine between the pooled features and the class row.
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    f = namespace["features"](params, x)
    w = params["fc.weight"].astype(np.float64)
    cos = (f / np.linalg.norm(f, axis=1, keepdims=True)) @ (w / np.linalg.norm(w, axis=1, keepdims=True)).T
    np.testing.assert_allclose(logits, (SCALE * cos).astype(np.float32), rtol=0, atol=1e-5)


def test_render_refuses_bad_constants():
    with pytest.raises(ValueError):
        render_resnet20_forward(MEAN, STD, 100, SCALE, 1e-5, 1e-12, input_shape=(3, 30, 30))
    with pytest.raises(ValueError):
        render_resnet20_forward(MEAN[:2], STD, 100, SCALE, 1e-5, 1e-12)
    with pytest.raises(ValueError):
        render_resnet20_forward(MEAN, STD, 100, 0.0, 1e-5, 1e-12)
    text = render_resnet20_forward(MEAN, STD, 100, SCALE, 1e-5, 1e-12)
    assert "cifar" not in text.lower() and "seed" not in text.lower()
