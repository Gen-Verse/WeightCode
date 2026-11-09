"""Differentiable torch rebuild of an instance's forward pass (``mlp``, ``cnn``, ``resnet20_a``, ``vit_lite_7_4``).

The editor-visible numpy ``forward.py`` defines the model. This module rebuilds the same function in torch so that
the fitting references can take gradients and read penultimate activations. ``assert_parity`` compares the rebuild
with the numpy forward before any fit, so an architecture mismatch stops a run instead of fitting a different model.

Families, inferred from parameter names (any other name raises):

* ``mlp``: linear layers ``fc1 .. fcK`` with ReLU between them and none after the last. The input is flattened
  row-major from NCHW.
* ``cnn``: ``conv1 .. convJ`` (stride 1, zero padding (k - 1) // 2, so the spatial size is kept), each followed by
  ReLU and 2x2 max pooling with stride 2 (floor), then flatten in C, H, W order and ``fc1 .. fcK`` as in ``mlp``.

* ``resnet20_a`` (PREREG E10, chosen by model_spec.json's ``arch``; the names and shapes are checked by
  ``weightcode.models.resnet20a.check_params``): He et al.'s CIFAR ResNet-20 with option-A shortcuts. conv1 (3x3, no bias),
  bn1, ReLU, nine BasicBlocks (``block_names``), global average pooling and the linear head ``fc``. BatchNorm is in
  eval form: the running statistics are inputs, never trained (``trainable_tensors`` leaves them out), unless a
  caller passes its own normalization (``resnet20a_features``'s ``bn``, used by the label-free BatchNorm references).

* ``vit_lite_7_4`` (PREREG E11, chosen by model_spec.json's ``arch``; names and shapes checked by
  ``weightcode.models.vitlite.check_params``): ViT-Lite-7/4 as forward.py states it. The patch projection
  ``tokenizer.conv_layers.0.0`` (``conv_layers``), the class token and positional embedding, 7 layers
  ``h = h + proj(attn(pre_norm(h))); h = norm1(h); h = h + linear2(gelu_tanh(linear1(h)))``, the final LayerNorm on
  the class token's row (the penultimate features) and the head ``classifier.fc`` (``linear_layers``). No dropout:
  the rebuild is the eval-form network. ``ln_layers`` lists the 15 LayerNorm prefixes in forward order.

* ``resnet18_up`` (PREREG E12, chosen by model_spec.json's ``arch``; names and shapes checked by
  ``weightcode.models.resnet18up.check_params``): torchvision's ResNet-18 behind bilinear upsampling of the normalised
  32 x 32 image to R x R (``F.interpolate``, align_corners False, no antialias; R from the spec's
  ``preprocessing.resize``). conv1 (7x7, stride 2, padding 3), bn1, ReLU, 3x3 max pooling with stride 2 and padding 1,
  eight BasicBlocks with projection shortcuts (``downsample.0`` conv 1x1, ``downsample.1`` BatchNorm) where the stride
  or width changes, global average pooling and ``fc``. BatchNorm as for resnet20_a: eval form, running statistics
  never trained, unless a caller passes its own normalization (``resnet18up_features``'s ``bn``).

* ``resnet18_native_u8`` (PREREG T04, chosen by model_spec.json's ``arch``; names and shapes checked by
  ``weightcode.models.resnet18up.check_params``, the 1000-class head included): ``resnet18_up``'s network on the stored
  224 x 224 uint8 center crop with no upsampling (``weightcode.models.resnet18native.ResNet18NativeU8``). Everything else
  is ``resnet18_up``'s, so every reference that runs on that family runs on this one unchanged (``RESNET18_FAMILIES``).

Preprocessing follows model_spec.json: ``x / 255``, then ``(x - mean[c]) / std[c]`` per channel (``input`` for
mlp/cnn, ``preprocessing`` for resnet20_a, vit_lite_7_4 and resnet18_up).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

POOL = 2
"""Max-pool kernel and stride after every conv layer of the ``cnn`` family."""

PARITY_ATOL = 1e-4
"""Largest allowed absolute logit difference between the rebuild and forward.py (contract section 3)."""

_PARAM_NAME = re.compile(r"^(conv|fc)([1-9][0-9]*)\.(weight|bias)$")
_NUMPY_DTYPES: dict[torch.dtype, type[np.floating[Any]]] = {torch.float32: np.float32, torch.float64: np.float64}

NumpyForward = Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]
BatchNormFn = Callable[[torch.Tensor, str], torch.Tensor]
"""A BatchNorm replacement: (pre-activation h, layer prefix such as "layer1.0.bn1") -> normalized h."""

RESNET20A = "resnet20_a"
VITLITE = "vit_lite_7_4"
RESNET18UP = "resnet18_up"
RESNET18NAT = "resnet18_native_u8"
RESNET18_FAMILIES: tuple[str, ...] = (RESNET18UP, RESNET18NAT)
"""torchvision's ResNet-18 layout: E12's student behind its upsampling, and T04's native network (resize 0)."""
BN_FAMILIES: tuple[str, ...] = (RESNET20A, RESNET18UP, RESNET18NAT)
"""Families with BatchNorm in eval form (running statistics as inputs, never trained by a gradient fit)."""
RUNNING_STATS = ("running_mean", "running_var")


@dataclass(frozen=True)
class Architecture:
    """Layer structure and preprocessing of one model, enough to rebuild its forward pass."""

    family: str
    input_shape: tuple[int, int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    conv_layers: tuple[str, ...]
    linear_layers: tuple[str, ...]
    bn_layers: tuple[str, ...] = ()
    """BatchNorm layer prefixes in forward order (resnet20_a only)."""
    ln_layers: tuple[str, ...] = ()
    """LayerNorm layer prefixes in forward order (vit_lite_7_4 only)."""
    resize: int = 0
    """The upsampled size R in front of the network (resnet18_up only; 0 for every other family)."""

    @property
    def last_linear(self) -> str:
        """Name of the layer that produces the logits."""
        return self.linear_layers[-1]

    @property
    def last_linear_tensors(self) -> tuple[str, str]:
        """Weight and bias names of the final linear layer."""
        return (f"{self.last_linear}.weight", f"{self.last_linear}.bias")

    def describe(self) -> dict[str, Any]:
        """JSON-ready description for run records."""
        if self.family == RESNET20A:
            return {"family": self.family, "input_shape": list(self.input_shape), "mean": list(self.mean),
                    "std": list(self.std), "conv_layers": list(self.conv_layers),
                    "bn_layers": list(self.bn_layers), "linear_layers": list(self.linear_layers),
                    "shortcut": "option A: x[:, :, ::2, ::2] with out_channels // 4 zero channels on each side",
                    "batchnorm": "eval form (running statistics)"}
        if self.family in RESNET18_FAMILIES:
            return {"family": self.family, "input_shape": list(self.input_shape), "mean": list(self.mean),
                    "std": list(self.std), "resize": self.resize,
                    "upsample": ("none" if self.family == RESNET18NAT else
                                 "bilinear, align_corners False, no antialias, after the normalisation"),
                    "conv_layers": list(self.conv_layers), "bn_layers": list(self.bn_layers),
                    "linear_layers": list(self.linear_layers),
                    "stem": "conv1 7x7 stride 2 padding 3, bn1, ReLU, max pool 3x3 stride 2 padding 1",
                    "shortcut": "downsample.0 (conv 1x1 with the stride) and downsample.1 (BatchNorm) where the stride "
                                "or width changes, else identity",
                    "batchnorm": "eval form (running statistics)"}
        if self.family == VITLITE:
            return {"family": self.family, "input_shape": list(self.input_shape), "mean": list(self.mean),
                    "std": list(self.std), "patch_projection": list(self.conv_layers),
                    "ln_layers": list(self.ln_layers), "linear_layers": list(self.linear_layers),
                    "layer": "h = h + proj(attn(pre_norm(h))); h = norm1(h); h = h + linear2(gelu_tanh(linear1(h)))",
                    "features": "final LayerNorm, class token row", "dropout": "none (eval form)"}
        return {
            "family": self.family,
            "input_shape": list(self.input_shape),
            "mean": list(self.mean),
            "std": list(self.std),
            "conv_layers": list(self.conv_layers),
            "linear_layers": list(self.linear_layers),
            "pool": POOL,
        }


def _group_layers(params: Mapping[str, np.ndarray]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, dict[int, set[str]]] = {"conv": {}, "fc": {}}
    for key in params:
        match = _PARAM_NAME.match(key)
        if match is None:
            raise ValueError(f"parameter {key!r} is outside the supported mlp/cnn families")
        groups[match.group(1)].setdefault(int(match.group(2)), set()).add(match.group(3))
    ordered: dict[str, tuple[str, ...]] = {}
    for prefix, layers in groups.items():
        indices = sorted(layers)
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError(f"{prefix} layers must be numbered 1..n without gaps, got {indices}")
        for index in indices:
            if layers[index] != {"weight", "bias"}:
                raise ValueError(f"{prefix}{index} needs a weight and a bias, got {sorted(layers[index])}")
        ordered[prefix] = tuple(f"{prefix}{index}" for index in indices)
    if not ordered["fc"]:
        raise ValueError("no fc layers: the model needs a final linear layer")
    return ordered


def _spec_input(spec: Mapping[str, Any]) -> tuple[tuple[int, int, int], tuple[float, ...], tuple[float, ...]]:
    try:
        block = spec["input"]
        shape = tuple(int(value) for value in block["shape"])
        mean = tuple(float(value) for value in block["mean"])
        std = tuple(float(value) for value in block["std"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"model_spec.json 'input' must give shape, mean and std: {error!r}") from error
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError(f"input shape must be [C, H, W], got {list(shape)}")
    if len(mean) != shape[0] or len(std) != shape[0]:
        raise ValueError(f"mean and std need one value per channel ({shape[0]}), got {len(mean)} and {len(std)}")
    if not all(value > 0.0 for value in std):
        raise ValueError(f"std must be positive, got {list(std)}")
    return (shape[0], shape[1], shape[2]), mean, std


def _resnet20a_architecture(params: Mapping[str, np.ndarray], spec: Mapping[str, Any]) -> Architecture:
    from weightcode.models import resnet20a as A

    head = params.get(A.HEAD_WEIGHT)
    if head is None or np.ndim(head) != 2:
        raise ValueError(f"a resnet20_a model needs a 2-D {A.HEAD_WEIGHT}")
    A.check_params(params, int(np.shape(head)[0]))
    try:
        shape = tuple(int(value) for value in spec["input_shape"])
        mean = tuple(float(value) for value in spec["preprocessing"]["mean"])
        std = tuple(float(value) for value in spec["preprocessing"]["std"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"resnet20_a model_spec.json needs input_shape and preprocessing mean/std: {error!r}") from error
    if len(shape) != 3 or len(mean) != shape[0] or len(std) != shape[0] or not all(value > 0.0 for value in std):
        raise ValueError(f"resnet20_a input {list(shape)} with mean {list(mean)} and std {list(std)} is inconsistent")
    convs, bns = ["conv1"], ["bn1"]
    for name, _c_in, _c_out, _stride in A.block_names():
        convs += [f"{name}.conv1", f"{name}.conv2"]
        bns += [f"{name}.bn1", f"{name}.bn2"]
    return Architecture(family=RESNET20A, input_shape=(shape[0], shape[1], shape[2]), mean=mean, std=std,
                        conv_layers=tuple(convs), linear_layers=("fc",), bn_layers=tuple(bns))


def _vitlite_architecture(params: Mapping[str, np.ndarray], spec: Mapping[str, Any]) -> Architecture:
    from weightcode.models import vitlite as V

    head = params.get(V.HEAD_WEIGHT)
    if head is None or np.ndim(head) != 2:
        raise ValueError(f"a vit_lite_7_4 model needs a 2-D {V.HEAD_WEIGHT}")
    V.check_params(params, int(np.shape(head)[0]))
    try:
        shape = tuple(int(value) for value in spec["input_shape"])
        mean = tuple(float(value) for value in spec["preprocessing"]["mean"])
        std = tuple(float(value) for value in spec["preprocessing"]["std"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"vit_lite_7_4 model_spec.json needs input_shape and preprocessing mean/std: {error!r}") from error
    if shape != V.INPUT_SHAPE or len(mean) != shape[0] or len(std) != shape[0] or not all(v > 0.0 for v in std):
        raise ValueError(f"vit_lite_7_4 input {list(shape)} with mean {list(mean)} and std {list(std)} is inconsistent")
    lns = [f"classifier.blocks.{i}.{n}" for i in range(V.N_LAYERS) for n in ("pre_norm", "norm1")]
    return Architecture(family=VITLITE, input_shape=(shape[0], shape[1], shape[2]), mean=mean, std=std,
                        conv_layers=("tokenizer.conv_layers.0.0",), linear_layers=("classifier.fc",),
                        ln_layers=(*lns, "classifier.norm"))


def _resnet18up_architecture(params: Mapping[str, np.ndarray], spec: Mapping[str, Any]) -> Architecture:
    from weightcode.models import resnet18up as M

    head = params.get(M.HEAD_WEIGHT)
    if head is None or np.ndim(head) != 2:
        raise ValueError(f"a resnet18_up model needs a 2-D {M.HEAD_WEIGHT}")
    M.check_params(params, int(np.shape(head)[0]))
    try:
        shape = tuple(int(value) for value in spec["input_shape"])
        mean = tuple(float(value) for value in spec["preprocessing"]["mean"])
        std = tuple(float(value) for value in spec["preprocessing"]["std"])
        resize = [int(value) for value in spec["preprocessing"]["resize"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"resnet18_up model_spec.json needs input_shape and preprocessing mean, std and resize: {error!r}") from error
    if (shape != M.INPUT_SHAPE or len(mean) != shape[0] or len(std) != shape[0] or not all(v > 0.0 for v in std)
            or len(resize) != 2 or resize[0] != resize[1] or resize[0] not in M.RESIZES):
        raise ValueError(f"resnet18_up input {list(shape)} with mean {list(mean)}, std {list(std)} and resize {resize} "
                         "is inconsistent")
    convs = tuple(name.removesuffix(".weight") for name in M.conv_names())
    return Architecture(family=RESNET18UP, input_shape=(shape[0], shape[1], shape[2]), mean=mean, std=std,
                        conv_layers=convs, linear_layers=("fc",), bn_layers=tuple(M.bn_prefixes()),
                        resize=int(resize[0]))


def _resnet18native_architecture(params: Mapping[str, np.ndarray], spec: Mapping[str, Any]) -> Architecture:
    from weightcode.models import resnet18native as N
    from weightcode.models import resnet18up as M

    head = params.get(M.HEAD_WEIGHT)
    if head is None or np.ndim(head) != 2:
        raise ValueError(f"a {RESNET18NAT} model needs a 2-D {M.HEAD_WEIGHT}")
    M.check_params(params, int(np.shape(head)[0]))
    try:
        shape = tuple(int(value) for value in spec["input_shape"])
        mean = tuple(float(value) for value in spec["preprocessing"]["mean"])
        std = tuple(float(value) for value in spec["preprocessing"]["std"])
        dtype = str(spec["input"]["dtype"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{RESNET18NAT} model_spec.json needs input_shape, input dtype and preprocessing mean and std: {error!r}"
        ) from error
    if (shape != N.INPUT_SHAPE or dtype != "uint8" or len(mean) != shape[0] or len(std) != shape[0]
            or not all(v > 0.0 for v in std)):
        raise ValueError(f"{RESNET18NAT} input {list(shape)} {dtype} with mean {list(mean)} and std {list(std)} is "
                         "inconsistent")
    convs = tuple(name.removesuffix(".weight") for name in M.conv_names())
    return Architecture(family=RESNET18NAT, input_shape=(shape[0], shape[1], shape[2]), mean=mean, std=std,
                        conv_layers=convs, linear_layers=("fc",), bn_layers=tuple(M.bn_prefixes()), resize=0)


def architecture_from(params: Mapping[str, np.ndarray], spec: Mapping[str, Any]) -> Architecture:
    """Infer the family and layer order from parameter names and check every shape against the spec's input.

    model_spec.json's ``arch`` "resnet20_a", "vit_lite_7_4" or "resnet18_up" selects that family; otherwise the
    parameter names decide (mlp/cnn).
    """
    if spec.get("arch") == RESNET20A:
        return _resnet20a_architecture(params, spec)
    if spec.get("arch") == VITLITE:
        return _vitlite_architecture(params, spec)
    if spec.get("arch") == RESNET18UP:
        return _resnet18up_architecture(params, spec)
    if spec.get("arch") == RESNET18NAT:
        return _resnet18native_architecture(params, spec)
    layers = _group_layers(params)
    input_shape, mean, std = _spec_input(spec)
    channels, height, width = input_shape
    for layer in layers["conv"]:
        weight, bias = params[f"{layer}.weight"], params[f"{layer}.bias"]
        if weight.ndim != 4 or weight.shape[1] != channels:
            raise ValueError(f"{layer}.weight has shape {weight.shape}; expected [out, {channels}, k, k]")
        if weight.shape[2] != weight.shape[3] or weight.shape[2] % 2 == 0:
            raise ValueError(f"{layer}.weight kernel {weight.shape[2:]} is not square and odd")
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"{layer}.bias has shape {bias.shape}; expected ({weight.shape[0]},)")
        channels, height, width = weight.shape[0], height // POOL, width // POOL
        if height < 1 or width < 1:
            raise ValueError(f"spatial size collapses to zero after {layer}")
    features = channels * height * width
    for layer in layers["fc"]:
        weight, bias = params[f"{layer}.weight"], params[f"{layer}.bias"]
        if weight.ndim != 2 or weight.shape[1] != features:
            raise ValueError(f"{layer}.weight has shape {weight.shape}; the layers before it give {features} inputs")
        if bias.shape != (weight.shape[0],):
            raise ValueError(f"{layer}.bias has shape {bias.shape}; expected ({weight.shape[0]},)")
        features = weight.shape[0]
    return Architecture(
        family="cnn" if layers["conv"] else "mlp",
        input_shape=input_shape,
        mean=mean,
        std=std,
        conv_layers=layers["conv"],
        linear_layers=layers["fc"],
    )


def to_tensors(
    params: Mapping[str, np.ndarray], *, dtype: torch.dtype = torch.float32, trainable: Collection[str] = ()
) -> dict[str, torch.Tensor]:
    """Copy parameters into torch tensors of ``dtype``; names in ``trainable`` require gradients."""
    unknown = set(trainable) - set(params)
    if unknown:
        raise KeyError(f"trainable names not in params: {sorted(unknown)}")
    np_dtype = _NUMPY_DTYPES[dtype]
    return {
        key: torch.from_numpy(np.array(value, dtype=np_dtype, copy=True)).requires_grad_(key in trainable)
        for key, value in params.items()
    }


def images_tensor(x: np.ndarray) -> torch.Tensor:
    """uint8 images as a torch tensor that owns its memory."""
    if x.dtype != np.uint8:
        raise TypeError(f"images must be uint8, got {x.dtype}")
    return torch.from_numpy(np.array(x, copy=True))


def trainable_tensors(arch: Architecture, params: Mapping[str, np.ndarray]) -> list[str]:
    """The tensors a full gradient fit trains, sorted: every tensor, except BatchNorm running statistics."""
    if arch.family in BN_FAMILIES:
        return sorted(key for key in params if not key.endswith(RUNNING_STATS))
    return sorted(params)


def _preprocessed(arch: Architecture, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    channels, height, width = arch.input_shape
    h = x.reshape(x.shape[0], channels, height, width).to(dtype) / 255.0
    mean = torch.tensor(arch.mean, dtype=dtype).view(1, channels, 1, 1)
    std = torch.tensor(arch.std, dtype=dtype).view(1, channels, 1, 1)
    return (h - mean) / std


def bn_eval(tensors: Mapping[str, torch.Tensor], h: torch.Tensor, prefix: str) -> torch.Tensor:
    """BatchNorm in eval form with the layer's running statistics (forward.py's computation)."""
    from weightcode.models.resnet20a import BN_EPS

    return F.batch_norm(h, tensors[f"{prefix}.running_mean"], tensors[f"{prefix}.running_var"],
                        tensors[f"{prefix}.weight"], tensors[f"{prefix}.bias"], training=False, eps=BN_EPS)


def resnet20a_features(arch: Architecture, tensors: Mapping[str, torch.Tensor], x: torch.Tensor,
                       bn: BatchNormFn | None = None) -> torch.Tensor:
    """Pooled [N, 64] features of a resnet20_a model; ``bn`` replaces the eval-form BatchNorm when given."""
    from weightcode.models import resnet20a as A

    norm: BatchNormFn = bn if bn is not None else (lambda h, prefix: bn_eval(tensors, h, prefix))
    h = _preprocessed(arch, x, tensors["conv1.weight"].dtype)
    h = F.relu(norm(F.conv2d(h, tensors["conv1.weight"], stride=1, padding=1), "bn1"))
    for name, c_in, c_out, stride in A.block_names():
        out = F.relu(norm(F.conv2d(h, tensors[f"{name}.conv1.weight"], stride=stride, padding=1), f"{name}.bn1"))
        out = norm(F.conv2d(out, tensors[f"{name}.conv2.weight"], stride=1, padding=1), f"{name}.bn2")
        if A.has_subsample(c_in, c_out, stride):
            pad = c_out // 4
            shortcut = F.pad(h[:, :, ::2, ::2], (0, 0, 0, 0, pad, pad), "constant", 0.0)
        else:
            shortcut = h
        h = F.relu(out + shortcut)
    return h.mean(dim=(2, 3))


def resnet18up_features(arch: Architecture, tensors: Mapping[str, torch.Tensor], x: torch.Tensor,
                        bn: BatchNormFn | None = None) -> torch.Tensor:
    """Pooled [N, 512] features of a resnet18_up or resnet18_native_u8 model (the latter without the upsampling);
    ``bn`` replaces the eval-form BatchNorm when given."""
    from weightcode.models import resnet18up as M

    if arch.family not in RESNET18_FAMILIES:
        raise ValueError(f"family {arch.family!r} is not one of {RESNET18_FAMILIES}")
    norm: BatchNormFn = bn if bn is not None else (lambda h, prefix: bn_eval(tensors, h, prefix))
    h = _preprocessed(arch, x, tensors["conv1.weight"].dtype)
    if arch.family == RESNET18UP:
        h = F.interpolate(h, size=(arch.resize, arch.resize), mode="bilinear", align_corners=False, antialias=False)
    h = F.relu(norm(F.conv2d(h, tensors["conv1.weight"], stride=2, padding=3), "bn1"))
    h = F.max_pool2d(h, kernel_size=3, stride=2, padding=1)
    for name, c_in, c_out, stride in M.block_names():
        out = F.relu(norm(F.conv2d(h, tensors[f"{name}.conv1.weight"], stride=stride, padding=1), f"{name}.bn1"))
        out = norm(F.conv2d(out, tensors[f"{name}.conv2.weight"], stride=1, padding=1), f"{name}.bn2")
        if M.has_downsample(c_in, c_out, stride):
            shortcut = norm(F.conv2d(h, tensors[f"{name}.downsample.0.weight"], stride=stride), f"{name}.downsample.1")
        else:
            shortcut = h
        h = F.relu(out + shortcut)
    return h.mean(dim=(2, 3))


def bn_features(arch: Architecture, tensors: Mapping[str, torch.Tensor], x: torch.Tensor,
                bn: BatchNormFn | None = None) -> torch.Tensor:
    """Pooled features of a BatchNorm family (``BN_FAMILIES``), with ``bn`` replacing the eval form when given."""
    if arch.family == RESNET20A:
        return resnet20a_features(arch, tensors, x, bn=bn)
    if arch.family in RESNET18_FAMILIES:
        return resnet18up_features(arch, tensors, x, bn=bn)
    raise ValueError(f"family {arch.family!r} has no BatchNorm; expected one of {BN_FAMILIES}")


def vitlite_features(arch: Architecture, tensors: Mapping[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """[N, 256] features of a vit_lite_7_4 model: the class token's row after the final LayerNorm (no dropout)."""
    from weightcode.models import vitlite as V

    dim, heads = V.EMBED_DIM, V.N_HEADS
    head_dim = dim // heads
    w = tensors["tokenizer.conv_layers.0.0.weight"]
    h = _preprocessed(arch, x, w.dtype)
    tokens = F.conv2d(h, w, tensors["tokenizer.conv_layers.0.0.bias"], stride=V.PATCH).flatten(2).transpose(1, 2)
    n = tokens.shape[0]
    h = torch.cat([tensors["classifier.class_emb"].expand(n, -1, -1), tokens], dim=1)
    h = h + tensors["classifier.positional_emb"]
    for index in range(V.N_LAYERS):
        b = f"classifier.blocks.{index}"
        a = F.layer_norm(h, (dim,), tensors[f"{b}.pre_norm.weight"], tensors[f"{b}.pre_norm.bias"], V.LN_EPS)
        qkv = F.linear(a, tensors[f"{b}.self_attn.qkv.weight"]).reshape(n, V.SEQ_LEN, 3, heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * head_dim ** -0.5).softmax(dim=-1)
        mixed = (attn @ v).transpose(1, 2).reshape(n, V.SEQ_LEN, dim)
        h = h + F.linear(mixed, tensors[f"{b}.self_attn.proj.weight"], tensors[f"{b}.self_attn.proj.bias"])
        h = F.layer_norm(h, (dim,), tensors[f"{b}.norm1.weight"], tensors[f"{b}.norm1.bias"], V.LN_EPS)
        hidden = F.gelu(F.linear(h, tensors[f"{b}.linear1.weight"], tensors[f"{b}.linear1.bias"]),
                        approximate=V.GELU_APPROXIMATE)
        h = h + F.linear(hidden, tensors[f"{b}.linear2.weight"], tensors[f"{b}.linear2.bias"])
    return F.layer_norm(h[:, 0], (dim,), tensors["classifier.norm.weight"], tensors["classifier.norm.bias"], V.LN_EPS)


def penultimate(arch: Architecture, tensors: Mapping[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Input of the final linear layer (after the preceding ReLU, or the pooled features) for uint8 images ``x``."""
    if arch.family == RESNET20A:
        return resnet20a_features(arch, tensors, x)
    if arch.family in RESNET18_FAMILIES:
        return resnet18up_features(arch, tensors, x)
    if arch.family == VITLITE:
        return vitlite_features(arch, tensors, x)
    dtype = tensors[arch.last_linear_tensors[0]].dtype
    h = _preprocessed(arch, x, dtype)
    for layer in arch.conv_layers:
        weight = tensors[f"{layer}.weight"]
        h = F.conv2d(h, weight, tensors[f"{layer}.bias"], stride=1, padding=(weight.shape[-1] - 1) // 2)
        h = F.max_pool2d(F.relu(h), kernel_size=POOL, stride=POOL)
    h = h.reshape(h.shape[0], -1)
    for layer in arch.linear_layers[:-1]:
        h = F.relu(F.linear(h, tensors[f"{layer}.weight"], tensors[f"{layer}.bias"]))
    return h


def logits(arch: Architecture, tensors: Mapping[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Logits of the rebuilt model for uint8 images ``x``."""
    weight, bias = arch.last_linear_tensors
    return F.linear(penultimate(arch, tensors, x), tensors[weight], tensors[bias])


def max_logit_diff(
    arch: Architecture,
    params: Mapping[str, np.ndarray],
    forward: NumpyForward,
    x: np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Largest absolute logit difference between the torch rebuild and the numpy forward on ``x``."""
    reference = np.asarray(forward(dict(params), x), dtype=np.float64)
    with torch.no_grad():
        rebuilt = logits(arch, to_tensors(params, dtype=dtype), images_tensor(x)).to(torch.float64).numpy()
    if reference.shape != rebuilt.shape:
        raise ValueError(f"forward.py gives logits {reference.shape}, the rebuild gives {rebuilt.shape}")
    return float(np.max(np.abs(reference - rebuilt)))


def assert_parity(
    arch: Architecture,
    params: Mapping[str, np.ndarray],
    forward: NumpyForward,
    x: np.ndarray,
    *,
    atol: float = PARITY_ATOL,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Return the max abs logit difference on ``x`` and raise if it exceeds ``atol``."""
    diff = max_logit_diff(arch, params, forward, x, dtype=dtype)
    if not diff <= atol:
        raise RuntimeError(
            f"torch rebuild ({arch.family}) differs from forward.py by {diff:.3g} > {atol:g} on {len(x)} images; "
            "the rebuilt architecture does not match this model"
        )
    return diff
