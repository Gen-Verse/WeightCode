"""PyTorch architectures with the exact parameter names of the contract (brief item 3).

``mlp``: fc1, fc2, fc3 (Linear), ReLU between, hidden width 128, input flattened in C, H, W order.
``cnn``: conv1 3->32, conv2 32->64, conv3 64->64 (kernel 3, stride 1, padding 1), each followed by ReLU and
2 x 2 max pooling, flatten in C, H, W order, fc1 -> 128, ReLU, fc2 -> classes. No batch norm, no dropout.
``cnn4`` (the second CIFAR architecture): conv1 3->32, conv2 32->64, conv3 64->128, conv4 128->128 (kernel 3,
stride 1, padding 1), each followed by ReLU and 2 x 2 max pooling, flatten in C, H, W order, then a single linear
head fc1 -> classes. It differs from cnn in depth, widths, the head name and the absence of a hidden linear layer,
and it stays inside the family the dataset forward.py reads from parameter names, so it needs no other forward.

Every model takes raw uint8 (or float in [0, 255]) NCHW images and applies x / 255 then per-channel
(x - mean) / std itself. Mean and std are non-persistent buffers, so ``state_dict`` holds only the weights and
biases named above.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ARCHS: tuple[str, ...] = ("mlp", "cnn", "cnn4")
MLP_HIDDEN = 128
CNN_CHANNELS: tuple[int, int, int] = (32, 64, 64)
CNN_HIDDEN = 128
CNN4_CHANNELS: tuple[int, int, int, int] = (32, 64, 128, 128)
CONV_KERNEL = 3
CONV_STRIDE = 1
CONV_PADDING = 1
POOL_KERNEL = 2
POOL_STRIDE = 2


class Preprocess(nn.Module):
    """x / 255, then per-channel (x - mean) / std, in float32."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        if len(mean) != len(std):
            raise ValueError("mean and std lengths differ")
        self.register_buffer("mean", torch.tensor(list(mean), dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(list(std), dtype=torch.float32).view(1, -1, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(torch.float32) / 255.0 - self.mean) / self.std


class MLP(nn.Module):
    """Three-layer perceptron fc1 -> ReLU -> fc2 -> ReLU -> fc3."""

    def __init__(
        self, input_shape: Sequence[int], n_classes: int, mean: Sequence[float], std: Sequence[float]
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or input_shape[0] != len(mean):
            raise ValueError(f"input_shape {tuple(input_shape)} does not match {len(mean)} channels")
        self.pre = Preprocess(mean, std)
        self.fc1 = nn.Linear(math.prod(input_shape), MLP_HIDDEN)
        self.fc2 = nn.Linear(MLP_HIDDEN, MLP_HIDDEN)
        self.fc3 = nn.Linear(MLP_HIDDEN, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x).flatten(1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class CNN(nn.Module):
    """Three conv blocks (conv, ReLU, max pool 2) then fc1 -> ReLU -> fc2."""

    def __init__(
        self, input_shape: Sequence[int], n_classes: int, mean: Sequence[float], std: Sequence[float]
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or input_shape[0] != len(mean):
            raise ValueError(f"input_shape {tuple(input_shape)} does not match {len(mean)} channels")
        c1, c2, c3 = CNN_CHANNELS
        conv = dict(kernel_size=CONV_KERNEL, stride=CONV_STRIDE, padding=CONV_PADDING)
        self.pre = Preprocess(mean, std)
        self.conv1 = nn.Conv2d(input_shape[0], c1, **conv)
        self.conv2 = nn.Conv2d(c1, c2, **conv)
        self.conv3 = nn.Conv2d(c2, c3, **conv)
        height, width = input_shape[1], input_shape[2]
        for _ in range(3):
            height = (height + 2 * CONV_PADDING - CONV_KERNEL) // CONV_STRIDE + 1
            width = (width + 2 * CONV_PADDING - CONV_KERNEL) // CONV_STRIDE + 1
            height = (height - POOL_KERNEL) // POOL_STRIDE + 1
            width = (width - POOL_KERNEL) // POOL_STRIDE + 1
        self.flat_features = c3 * height * width
        self.fc1 = nn.Linear(self.flat_features, CNN_HIDDEN)
        self.fc2 = nn.Linear(CNN_HIDDEN, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        for conv in (self.conv1, self.conv2, self.conv3):
            h = F.max_pool2d(F.relu(conv(h)), kernel_size=POOL_KERNEL, stride=POOL_STRIDE)
        h = F.relu(self.fc1(h.flatten(1)))
        return self.fc2(h)


CONV_LAYERS: dict[str, tuple[str, ...]] = {
    "mlp": (),
    "cnn": ("conv1", "conv2", "conv3"),
    "cnn4": ("conv1", "conv2", "conv3", "conv4"),
}
"""Conv layers of each architecture in forward order; each is followed by ReLU and 2 x 2 max pooling."""
FC_LAYERS: dict[str, tuple[str, ...]] = {"mlp": ("fc1", "fc2", "fc3"), "cnn": ("fc1", "fc2"), "cnn4": ("fc1",)}
"""Linear layers of each architecture in forward order; ReLU follows every one except the last (the head)."""


def head_layer(arch: str) -> str:
    """Name of the final linear layer (the classification head) of ``arch``."""
    if arch not in FC_LAYERS:
        raise KeyError(f"unknown arch {arch!r}; known: {ARCHS}")
    return FC_LAYERS[arch][-1]


class CNN4(nn.Module):
    """Four conv blocks (conv, ReLU, max pool 2), flatten, then one linear head fc1 -> classes."""

    def __init__(
        self, input_shape: Sequence[int], n_classes: int, mean: Sequence[float], std: Sequence[float]
    ) -> None:
        super().__init__()
        if len(input_shape) != 3 or input_shape[0] != len(mean):
            raise ValueError(f"input_shape {tuple(input_shape)} does not match {len(mean)} channels")
        c1, c2, c3, c4 = CNN4_CHANNELS
        conv = dict(kernel_size=CONV_KERNEL, stride=CONV_STRIDE, padding=CONV_PADDING)
        self.pre = Preprocess(mean, std)
        self.conv1 = nn.Conv2d(input_shape[0], c1, **conv)
        self.conv2 = nn.Conv2d(c1, c2, **conv)
        self.conv3 = nn.Conv2d(c2, c3, **conv)
        self.conv4 = nn.Conv2d(c3, c4, **conv)
        height, width = input_shape[1], input_shape[2]
        for _ in range(4):
            height = (height + 2 * CONV_PADDING - CONV_KERNEL) // CONV_STRIDE + 1
            width = (width + 2 * CONV_PADDING - CONV_KERNEL) // CONV_STRIDE + 1
            height = (height - POOL_KERNEL) // POOL_STRIDE + 1
            width = (width - POOL_KERNEL) // POOL_STRIDE + 1
        if height < 1 or width < 1:
            raise ValueError(f"input_shape {tuple(input_shape)} is too small for four pooling stages")
        self.flat_features = c4 * height * width
        self.fc1 = nn.Linear(self.flat_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            h = F.max_pool2d(F.relu(conv(h)), kernel_size=POOL_KERNEL, stride=POOL_STRIDE)
        return self.fc1(h.flatten(1))


def build_model(
    arch: str, input_shape: Sequence[int], n_classes: int, mean: Sequence[float], std: Sequence[float]
) -> nn.Module:
    """Construct an architecture on the CPU; parameters are initialized from torch's global generator."""
    if arch == "mlp":
        return MLP(input_shape, n_classes, mean, std)
    if arch == "cnn":
        return CNN(input_shape, n_classes, mean, std)
    if arch == "cnn4":
        return CNN4(input_shape, n_classes, mean, std)
    raise KeyError(f"unknown arch {arch!r}; known: {ARCHS}")


def export_params(model: nn.Module) -> dict[str, np.ndarray]:
    """Float32 numpy copies of the model's weights and biases, keyed by parameter name."""
    params = {}
    for name, tensor in model.state_dict().items():
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} has dtype {tensor.dtype}, expected float32")
        params[name] = tensor.detach().to("cpu").contiguous().numpy().copy()
    return params


def load_params_into(model: nn.Module, params: dict[str, np.ndarray]) -> nn.Module:
    """Load numpy parameters into ``model`` with strict key matching and return it."""
    state = {name: torch.from_numpy(value.copy()) for name, value in params.items()}
    model.load_state_dict(state, strict=True)
    return model
