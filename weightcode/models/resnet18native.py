"""PREREG T03's native ResNet-18 (arch ``resnet18_native``): torchvision's IMAGENET1K_V1 ResNet-18 with its 1000-class head.

This is E12's network (``weightcode.models.resnet18up``) without the fixed operations E12 puts in front of it: no x / 255, no
normalisation and no upsampling inside the model. The input is the preprocessed float tensor [N, 3, 224, 224] of
torchvision's eval transform for these weights (``ResNet18_Weights.IMAGENET1K_V1.transforms()``: RGB, bilinear resize
of the shorter side to 256 with antialiasing, center crop 224, scale to [0, 1], normalise with ``IMAGENET_MEAN`` and
``IMAGENET_STD``), which runs once outside the network (``weightcode.data.imagenet_val.preprocess``). Tensor names and shapes
are E12's (``resnet18up.param_shapes(1000)``), the layers of model_spec.json are E12's without the upsample layer, and
the parameters are the weights file ``resnet18-f37072fd.pth`` (sha256 checked) unchanged, the 512 x 1000 head
included. The numpy forward.py (the edit interface) is rendered by ``render_forward``.

PREREG T04 uses the same network on stored uint8 center crops (arch ``ARCH_U8``): the resize and the center crop stay
outside, and the last two steps of the eval transform (x / 255 and the normalisation) move inside, as E12's forward has
them, so that the data store keeps uint8 images (``ResNet18NativeU8``, ``render_forward_u8``,
``build_resnet18native_u8_spec``). The tensors, their names and the layer list are T03's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from weightcode.models import resnet18up as M
from weightcode.models.forward_gen import render_resnet18native_forward

ARCH = "resnet18_native"
SPEC_SCHEMA = "weightcode.model_spec.resnet18_native.v1"
N_CLASSES = 1000
INPUT_SHAPE: tuple[int, int, int] = (3, 224, 224)
RESIZE_SHORTER = 256
CROP = 224
EVAL_BATCH = 50
BN_EPS = M.BN_EPS
PREPROCESSING = (
    "outside the network, once per image: torchvision ResNet18_Weights.IMAGENET1K_V1.transforms() on the RGB image, "
    f"that is a bilinear resize of the shorter side to {RESIZE_SHORTER} (antialias True), a center crop to {CROP} x "
    f"{CROP}, a scale to [0, 1] and per channel (x - mean[c]) / std[c]; the network's input is that float32 tensor"
)


class ResNet18Native(nn.Module):
    """torchvision's ResNet-18 on the preprocessed input, with E12's module and parameter names."""

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_SHAPE[0], M.STEM_CHANNELS, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(M.STEM_CHANNELS, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stages: dict[int, list[M.BasicBlock]] = {1: [], 2: [], 3: [], 4: []}
        for name, c_in, c_out, stride in M.block_names():
            stages[int(name[len("layer")])].append(M.BasicBlock(c_in, c_out, stride))
        self.layer1 = nn.Sequential(*stages[1])
        self.layer2 = nn.Sequential(*stages[2])
        self.layer3 = nn.Sequential(*stages[3])
        self.layer4 = nn.Sequential(*stages[4])
        self.fc = nn.Linear(M.N_FEATURES, n_classes)

    def network_input(self, x: torch.Tensor) -> torch.Tensor:
        """The preprocessed input cast to the model's dtype; nothing else happens in front of conv1."""
        return x.to(self.conv1.weight.dtype)

    def upsampled(self, x: torch.Tensor) -> torch.Tensor:
        """``network_input``. The native model has no upsampling; the name lets ``resnet18up.torchvision_reference_diff``
        compare this module with ``torchvision.models.resnet18`` unchanged."""
        return self.network_input(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled [N, 512] features of preprocessed images."""
        h = self.maxpool(self.relu(self.bn1(self.conv1(self.network_input(x)))))
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        return h.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def imagenet_params(weights_path: Path) -> dict[str, np.ndarray]:
    """float32 numpy copies of every tensor of the IMAGENET1K_V1 file (sha256 checked by ``load_imagenet_state``),
    named as ``resnet18up.param_shapes(1000)``; num_batches_tracked is dropped (it does not enter eval mode)."""
    state = M.load_imagenet_state(weights_path)
    params = {}
    for name, shape in M.param_shapes(N_CLASSES).items():
        value = state[name].detach().to("cpu", torch.float32).contiguous().numpy().copy()
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
        params[name] = value
    M.check_params(params, N_CLASSES)
    return params


def build_resnet18native(params: Mapping[str, np.ndarray], dtype: torch.dtype = torch.float32) -> ResNet18Native:
    """A CPU model in ``dtype`` (eval mode) holding ``params``; num_batches_tracked keeps the module's 0 (unused in
    eval mode)."""
    n_classes = int(np.shape(params[M.HEAD_WEIGHT])[0])
    M.check_params(params, n_classes)
    model = ResNet18Native(n_classes).to(dtype)
    state = model.state_dict()
    for name in M.param_shapes(n_classes):
        state[name] = torch.from_numpy(np.ascontiguousarray(params[name])).to(dtype)
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def eval_logits(model: ResNet18Native, x: np.ndarray, batch_size: int = EVAL_BATCH) -> np.ndarray:
    """Eval-mode logits of preprocessed float images ``x`` [N, 3, 224, 224] as float64 numpy (computed in the model's
    dtype from the float32 input)."""
    images = np.asarray(x)
    if not np.issubdtype(images.dtype, np.floating) or images.ndim != 4 or tuple(images.shape[1:]) != INPUT_SHAPE:
        raise ValueError(f"inputs must be float [N, {INPUT_SHAPE}], got {images.dtype}{images.shape}")
    model.eval()
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size]))
        outputs.append(model(batch).to(torch.float64).numpy())
    return np.concatenate(outputs) if outputs else np.zeros((0, model.fc.out_features))


def render_forward(n_classes: int = N_CLASSES) -> str:
    """The forward.py source of this model (``forward_resnet18native.py.tmpl``)."""
    return render_resnet18native_forward(n_classes, BN_EPS, INPUT_SHAPE)


def network_layers(params: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """The layer list of model_spec.json: E12's layers (``resnet18up.build_resnet18up_spec``) without the upsample
    layer, built with the same helpers."""
    layers: list[dict[str, Any]] = [
        {"name": "conv1", "op": "conv2d", **M._conv("conv1.weight", params, 2, 3), "computes": M.CONV_COMPUTES},
        {"name": "bn1", "op": "batchnorm_eval", **M._bn("bn1", params), "computes": M.BN_COMPUTES},
        {"name": "relu", "op": "relu"},
        {"name": "maxpool", "op": "maxpool2d", "kernel": 3, "stride": 2, "padding": 1,
         "computes": "max over 3x3 windows with stride 2; padded positions never win (padding with -inf)"},
    ]
    for name, c_in, c_out, stride in M.block_names():
        shortcut = None
        if M.has_downsample(c_in, c_out, stride):
            shortcut = {"conv": M._conv(f"{name}.downsample.0.weight", params, stride, 0),
                        "bn": M._bn(f"{name}.downsample.1", params),
                        "computes": "s = bn(conv1x1(x)) with the block's stride"}
        layers.append({
            "name": name, "op": "basic_block", "in_channels": c_in, "out_channels": c_out, "stride": stride,
            "conv1": M._conv(f"{name}.conv1.weight", params, stride, 1), "bn1": M._bn(f"{name}.bn1", params),
            "conv2": M._conv(f"{name}.conv2.weight", params, 1, 1), "bn2": M._bn(f"{name}.bn2", params),
            "shortcut": shortcut,
            "computes": "out = relu(bn2(conv2(relu(bn1(conv1(x))))) + s), s = the shortcut of x, or s = x when the "
                        "shortcut is null",
        })
    layers.append({"name": "pool", "op": "global_avg_pool",
                   "computes": f"features = mean of h over H and W, [N, {M.N_FEATURES}]"})
    layers.append({"name": "fc", "op": "linear", "weight": M.HEAD_WEIGHT, "bias": M.HEAD_BIAS,
                   "shape": M._shape(params, M.HEAD_WEIGHT), "computes": "logits = f @ weight.T + bias"})
    return layers


def build_resnet18native_spec(model_id: str, class_names: Sequence[str], params: Mapping[str, np.ndarray],
                              mean: Sequence[float] = M.IMAGENET_MEAN,
                              std: Sequence[float] = M.IMAGENET_STD) -> dict[str, Any]:
    """Public ``model_spec.json`` of a ``resnet18_native`` model (schema ``SPEC_SCHEMA``), checked for forbidden terms
    (``weightcode.models.spec.check_public``)."""
    from weightcode.models.spec import check_public

    names = [str(name) for name in class_names]
    M.check_params(params, len(names))
    spec = {
        "schema": SPEC_SCHEMA,
        "model_id": model_id,
        "arch": ARCH,
        "input": {"kind": "preprocessed images", "shape": list(INPUT_SHAPE), "dtype": "float32", "layout": "NCHW"},
        "input_shape": list(INPUT_SHAPE),
        "preprocessing": {"mean": [float(v) for v in mean], "std": [float(v) for v in std],
                          "resize_shorter": RESIZE_SHORTER, "crop": [CROP, CROP], "interpolation": "bilinear",
                          "antialias": True, "computes": PREPROCESSING},
        "classes": names,
        "n_classes": len(names),
        "parameter_layout": "PyTorch: conv weight [out, in, kh, kw]; BatchNorm weight, bias, running_mean, "
                            f"running_var [channels]; head weight [n_classes, {M.N_FEATURES}] and bias [n_classes]; "
                            "all float32",
        "n_parameters": int(sum(np.asarray(value).size for value in params.values())),
        "tensors": {name: M._shape(params, name) for name in M.param_shapes(len(names))},
        "layers": network_layers(params),
        "output": "logits [N, n_classes]; the prediction is the argmax",
        "reference_forward": "forward.py: forward(params, x_float [N, 3, 224, 224]) -> float32 logits [N, n_classes]",
    }
    check_public(spec)
    return spec

# ---------------------------------------------------------------------------------------------------------------
# PREREG T04: the same network on stored uint8 center crops (``ARCH_U8``)

ARCH_U8 = "resnet18_native_u8"
SPEC_SCHEMA_U8 = "weightcode.model_spec.resnet18_native_u8.v1"
PREPROCESSING_U8 = (
    "outside the network, once per image: RGB, a bilinear resize of the shorter side to "
    f"{RESIZE_SHORTER} (antialias), a center crop to {CROP} x {CROP}; the stored image is that uint8 crop. Inside the "
    "network (forward.py): x / 255, then per channel (x - mean[c]) / std[c]"
)


class ResNet18NativeU8(ResNet18Native):
    """``ResNet18Native`` whose input is the uint8 center crop: x / 255 and the per-channel normalisation run in front
    of conv1, in the model's dtype (T03's preprocessing split at the crop, as E12's forward has it)."""

    def __init__(self, n_classes: int = N_CLASSES, mean: Sequence[float] = M.IMAGENET_MEAN,
                 std: Sequence[float] = M.IMAGENET_STD) -> None:
        super().__init__(n_classes)
        # float64 buffers: a float64 model then normalises exactly as forward.py does; .to(float32) rounds them once
        self.register_buffer("input_mean", torch.tensor([float(v) for v in mean], dtype=torch.float64).view(1, -1, 1, 1),
                             persistent=False)
        self.register_buffer("input_std", torch.tensor([float(v) for v in std], dtype=torch.float64).view(1, -1, 1, 1),
                             persistent=False)

    def network_input(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.conv1.weight.dtype
        return (x.to(dtype) / 255.0 - self.input_mean.to(dtype)) / self.input_std.to(dtype)


def build_resnet18native_u8(params: Mapping[str, np.ndarray], dtype: torch.dtype = torch.float32,
                            mean: Sequence[float] = M.IMAGENET_MEAN,
                            std: Sequence[float] = M.IMAGENET_STD) -> ResNet18NativeU8:
    """A CPU ``ResNet18NativeU8`` in ``dtype`` (eval mode) holding ``params``."""
    n_classes = int(np.shape(params[M.HEAD_WEIGHT])[0])
    M.check_params(params, n_classes)
    model = ResNet18NativeU8(n_classes, mean, std).to(dtype)
    state = model.state_dict()
    for name in M.param_shapes(n_classes):
        state[name] = torch.from_numpy(np.ascontiguousarray(params[name])).to(dtype)
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def eval_logits_u8(model: ResNet18NativeU8, x: np.ndarray, batch_size: int = EVAL_BATCH) -> np.ndarray:
    """Eval-mode logits of uint8 crops ``x`` [N, 3, 224, 224] as float64 numpy (computed in the model's dtype)."""
    images = np.asarray(x)
    if images.dtype != np.uint8 or images.ndim != 4 or tuple(images.shape[1:]) != INPUT_SHAPE:
        raise ValueError(f"inputs must be uint8 [N, {INPUT_SHAPE}], got {images.dtype}{images.shape}")
    model.eval()
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size]))
        outputs.append(model(batch).to(torch.float64).numpy())
    return np.concatenate(outputs) if outputs else np.zeros((0, model.fc.out_features))


def render_forward_u8(n_classes: int = N_CLASSES, mean: Sequence[float] = M.IMAGENET_MEAN,
                      std: Sequence[float] = M.IMAGENET_STD) -> str:
    """The forward.py source of the uint8 model (``forward_resnet18native_u8.py.tmpl``)."""
    from weightcode.models.forward_gen import render_resnet18native_u8_forward

    return render_resnet18native_u8_forward(mean, std, n_classes, BN_EPS, INPUT_SHAPE)


def build_resnet18native_u8_spec(model_id: str, class_names: Sequence[str], params: Mapping[str, np.ndarray],
                                 mean: Sequence[float] = M.IMAGENET_MEAN,
                                 std: Sequence[float] = M.IMAGENET_STD) -> dict[str, Any]:
    """Public ``model_spec.json`` of a ``resnet18_native_u8`` model (schema ``SPEC_SCHEMA_U8``): T03's layers, the
    uint8 input and the in-network scaling, checked for forbidden terms (``weightcode.models.spec.check_public``)."""
    from weightcode.models.spec import check_public

    names = [str(name) for name in class_names]
    M.check_params(params, len(names))
    spec = {
        "schema": SPEC_SCHEMA_U8,
        "model_id": model_id,
        "arch": ARCH_U8,
        "input": {"kind": "images", "shape": list(INPUT_SHAPE), "dtype": "uint8", "layout": "NCHW"},
        "input_shape": list(INPUT_SHAPE),
        "preprocessing": {"mean": [float(v) for v in mean], "std": [float(v) for v in std],
                          "resize_shorter": RESIZE_SHORTER, "crop": [CROP, CROP], "interpolation": "bilinear",
                          "antialias": True, "computes": PREPROCESSING_U8},
        "classes": names,
        "n_classes": len(names),
        "parameter_layout": "PyTorch: conv weight [out, in, kh, kw]; BatchNorm weight, bias, running_mean, "
                            f"running_var [channels]; head weight [n_classes, {M.N_FEATURES}] and bias [n_classes]; "
                            "all float32",
        "n_parameters": int(sum(np.asarray(value).size for value in params.values())),
        "tensors": {name: M._shape(params, name) for name in M.param_shapes(len(names))},
        "layers": network_layers(params),
        "output": "logits [N, n_classes]; the prediction is the argmax",
        "reference_forward": "forward.py: forward(params, x_uint8 [N, 3, 224, 224]) -> float32 logits [N, n_classes]",
    }
    check_public(spec)
    return spec


__all__ = [
    "ARCH", "ARCH_U8", "CROP", "EVAL_BATCH", "INPUT_SHAPE", "N_CLASSES", "PREPROCESSING", "PREPROCESSING_U8",
    "RESIZE_SHORTER", "ResNet18Native", "ResNet18NativeU8", "SPEC_SCHEMA", "SPEC_SCHEMA_U8", "build_resnet18native",
    "build_resnet18native_spec", "build_resnet18native_u8", "build_resnet18native_u8_spec", "eval_logits",
    "eval_logits_u8", "imagenet_params", "network_layers", "render_forward", "render_forward_u8",
]
