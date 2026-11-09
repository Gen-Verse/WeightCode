"""Float64 re-implementation of a ``model_spec.json`` network for localisation (contract section 3).

The search baselines score every candidate with the instance's own ``forward.py``, so fitness sees exactly the
editor's computation. Localisation needs what ``forward.py`` does not expose: the input and output of every dense
layer, interventions on hidden units and gradients. This module rebuilds the forward pass from the spec's layer
list (schema ``weightcode.model_spec.v0`` as written by ``weightcode.models.spec``), and ``check_parity`` raises unless its
logits match the instance forward within a tolerance on the given images.

Only the trailing dense stack (``linear`` layers with optional ``relu`` between them) is differentiated, because
both baselines localise over dense layers only (see ``arachne.py`` and ``care.py`` for why).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

SPEC_SCHEMA = "weightcode.model_spec.v0"
SUPPORTED_OPS: tuple[str, ...] = ("conv2d", "relu", "maxpool2d", "flatten", "linear")

ForwardFn = Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]


class SpecError(ValueError):
    """The model spec uses an op or an option this re-implementation does not support."""


@dataclass(frozen=True)
class Layer:
    """One entry of the spec's layer list, normalised."""

    op: str
    name: str | None = None
    weight: str | None = None
    bias: str | None = None
    kernel: tuple[int, int] | None = None
    stride: tuple[int, int] | None = None
    padding: tuple[int, int] | None = None


@dataclass(frozen=True)
class DenseLayer:
    """One linear layer of the trailing dense stack; ``relu`` says whether a ReLU follows it."""

    name: str
    weight: str
    bias: str
    relu: bool


@dataclass(frozen=True)
class NetSpec:
    """Parsed network: preprocessing, the full layer list and the trailing dense stack."""

    input_shape: tuple[int, int, int]
    mean: np.ndarray
    std: np.ndarray
    layers: tuple[Layer, ...]
    dense: tuple[DenseLayer, ...]
    n_prefix: int
    n_classes: int

    def dense_index(self, name: str) -> int:
        """Position of the dense layer called ``name`` in ``dense``."""
        for position, layer in enumerate(self.dense):
            if layer.name == name:
                return position
        raise KeyError(f"{name!r} is not a dense layer; dense layers: {[layer.name for layer in self.dense]}")


@dataclass(frozen=True)
class DenseTrace:
    """Activations of the dense stack for a batch.

    ``inputs[l]`` is the input of dense layer ``l``, ``pre[l]`` its pre-activation ``u W^T + b`` and ``outputs[l]``
    the value it passes on (after its ReLU if it has one). ``outputs[-1]`` are the logits.
    """

    inputs: list[np.ndarray]
    pre: list[np.ndarray]
    outputs: list[np.ndarray]

    @property
    def logits(self) -> np.ndarray:
        """Output of the last dense layer."""
        return self.outputs[-1]


def _pair(value: Any, field: str, layer_name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, Sequence) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return (int(value[0]), int(value[1]))
    raise SpecError(f"{layer_name}: {field} must be an int or a pair of ints, got {value!r}")


def _require_tensor(params: Mapping[str, np.ndarray], key: Any, ndim: int, layer_name: str) -> np.ndarray:
    if not isinstance(key, str) or key not in params:
        raise SpecError(f"{layer_name}: tensor {key!r} is not in the parameters")
    value = np.asarray(params[key])
    if value.ndim != ndim:
        raise SpecError(f"{layer_name}: {key} has {value.ndim} dims, expected {ndim}")
    return value


def parse_spec(spec: Mapping[str, Any], params: Mapping[str, np.ndarray]) -> NetSpec:
    """Parse and validate a ``weightcode.model_spec.v0`` dict against the parameter arrays.

    Raises ``SpecError`` on any op or option outside conv2d (zero padding, dilation 1, groups 1), relu, maxpool2d
    (no padding, floor mode), flatten (C, H, W row-major) and linear, so that an unsupported network fails loudly
    instead of being approximated.
    """
    if spec.get("schema") != SPEC_SCHEMA:
        raise SpecError(f"spec schema {spec.get('schema')!r} is not {SPEC_SCHEMA}")
    inp = spec["input"]
    shape = tuple(int(d) for d in inp["shape"])
    if len(shape) != 3:
        raise SpecError(f"input shape must be [C, H, W], got {list(shape)}")
    mean = np.asarray(inp["mean"], dtype=np.float64)
    std = np.asarray(inp["std"], dtype=np.float64)
    if mean.shape != (shape[0],) or std.shape != (shape[0],) or np.any(std <= 0):
        raise SpecError(f"mean {mean.tolist()} / std {std.tolist()} do not fit {shape[0]} channels")
    if inp.get("dtype", "uint8") != "uint8":
        raise SpecError(f"input dtype {inp.get('dtype')!r} is not uint8")
    layout = str(inp.get("layout", "NCHW, flatten row-major")).replace(" ", "")
    if not layout.startswith("NCHW"):
        raise SpecError(f"input layout {inp.get('layout')!r} is not NCHW")

    layers: list[Layer] = []
    for position, raw in enumerate(spec["layers"]):
        op = raw.get("op")
        label = str(raw.get("name", f"layer[{position}]"))
        if op not in SUPPORTED_OPS:
            raise SpecError(f"{label}: unsupported op {op!r}; supported: {SUPPORTED_OPS}")
        if op == "conv2d":
            weight = _require_tensor(params, raw.get("weight"), 4, label)
            bias = _require_tensor(params, raw.get("bias"), 1, label)
            kernel = _pair(raw["kernel"], "kernel", label)
            if kernel != tuple(weight.shape[2:]) or bias.shape[0] != weight.shape[0]:
                raise SpecError(f"{label}: kernel {kernel} / bias {bias.shape} do not match weight {weight.shape}")
            if raw.get("padding_mode", "zeros") != "zeros":
                raise SpecError(f"{label}: padding_mode {raw.get('padding_mode')!r} is not zeros")
            if _pair(raw.get("dilation", 1), "dilation", label) != (1, 1) or int(raw.get("groups", 1)) != 1:
                raise SpecError(f"{label}: only dilation 1 and groups 1 are supported")
            layers.append(
                Layer(
                    op=op,
                    name=label,
                    weight=raw["weight"],
                    bias=raw["bias"],
                    kernel=kernel,
                    stride=_pair(raw["stride"], "stride", label),
                    padding=_pair(raw["padding"], "padding", label),
                )
            )
        elif op == "maxpool2d":
            if _pair(raw.get("padding", 0), "padding", label) != (0, 0) or bool(raw.get("ceil_mode", False)):
                raise SpecError(f"{label}: only unpadded floor-mode max pooling is supported")
            layers.append(
                Layer(op=op, kernel=_pair(raw["kernel"], "kernel", label), stride=_pair(raw["stride"], "stride", label))
            )
        elif op == "flatten":
            order = str(raw.get("order", "C, H, W (row-major)")).replace(" ", "")
            if not order.startswith("C,H,W"):
                raise SpecError(f"{label}: flatten order {raw.get('order')!r} is not C, H, W row-major")
            layers.append(Layer(op=op))
        elif op == "linear":
            weight = _require_tensor(params, raw.get("weight"), 2, label)
            bias = _require_tensor(params, raw.get("bias"), 1, label)
            if bias.shape[0] != weight.shape[0]:
                raise SpecError(f"{label}: bias {bias.shape} does not match weight {weight.shape}")
            layers.append(Layer(op=op, name=label, weight=raw["weight"], bias=raw["bias"]))
        else:
            layers.append(Layer(op=op))

    # The dense stack is everything after the last conv / pool / flatten op.
    n_prefix = 0
    for position, layer in enumerate(layers):
        if layer.op in ("conv2d", "maxpool2d", "flatten"):
            n_prefix = position + 1
    tail = layers[n_prefix:]
    dense: list[DenseLayer] = []
    position = 0
    while position < len(tail):
        layer = tail[position]
        if layer.op != "linear" or layer.name is None or layer.weight is None or layer.bias is None:
            raise SpecError(f"dense stack must alternate linear and relu, found {layer.op!r} at {n_prefix + position}")
        relu = position + 1 < len(tail) and tail[position + 1].op == "relu"
        dense.append(DenseLayer(name=layer.name, weight=layer.weight, bias=layer.bias, relu=relu))
        position += 2 if relu else 1
    if not dense or dense[-1].relu:
        raise SpecError("the network must end in a linear layer that produces the logits")
    n_classes = int(np.asarray(params[dense[-1].weight]).shape[0])
    if "classes" in spec and len(spec["classes"]) != n_classes:
        raise SpecError(f"spec lists {len(spec['classes'])} classes but the last layer has {n_classes} outputs")
    return NetSpec(
        input_shape=(shape[0], shape[1], shape[2]),
        mean=mean,
        std=std,
        layers=tuple(layers),
        dense=tuple(dense),
        n_prefix=n_prefix,
        n_classes=n_classes,
    )


def preprocess(net: NetSpec, x_uint8: np.ndarray) -> np.ndarray:
    """x / 255 then per-channel (x - mean) / std, float64 [N, C, H, W]."""
    x = np.asarray(x_uint8)
    if x.dtype != np.uint8:
        raise TypeError(f"images must be uint8, got {x.dtype}")
    if x.ndim != 4 or tuple(x.shape[1:]) != net.input_shape:
        raise ValueError(f"images must have shape [N, {', '.join(map(str, net.input_shape))}], got {list(x.shape)}")
    h = x.astype(np.float64) / 255.0
    return (h - net.mean.reshape(1, -1, 1, 1)) / net.std.reshape(1, -1, 1, 1)


def _conv2d(
    h: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: tuple[int, int], padding: tuple[int, int]
) -> np.ndarray:
    n, channels, height, width = h.shape
    out_channels, in_channels, kh, kw = weight.shape
    if in_channels != channels:
        raise ValueError(f"conv weight {weight.shape} does not fit input {h.shape}")
    (sh, sw), (ph, pw) = stride, padding
    padded = np.pad(h, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    h_out = (height + 2 * ph - kh) // sh + 1
    w_out = (width + 2 * pw - kw) // sw + 1
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(2, 3))
    windows = windows[:, :, : (h_out - 1) * sh + 1 : sh, : (w_out - 1) * sw + 1 : sw]
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * h_out * w_out, channels * kh * kw)
    out = cols @ weight.reshape(out_channels, channels * kh * kw).T + bias
    return out.reshape(n, h_out, w_out, out_channels).transpose(0, 3, 1, 2)


def _maxpool(h: np.ndarray, kernel: tuple[int, int], stride: tuple[int, int]) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(h, kernel, axis=(2, 3))
    windows = windows[:, :, :: stride[0], :: stride[1]]
    return windows.max(axis=(4, 5))


def _tensor(params: Mapping[str, np.ndarray], key: str | None) -> np.ndarray:
    if key is None:
        raise SpecError("layer without a tensor name")
    return np.asarray(params[key], dtype=np.float64)


def dense_input(net: NetSpec, params: Mapping[str, np.ndarray], x_uint8: np.ndarray) -> np.ndarray:
    """Features entering the first dense layer, float64 [N, in_features]."""
    h = preprocess(net, x_uint8)
    for layer in net.layers[: net.n_prefix]:
        if layer.op == "conv2d":
            assert layer.stride is not None and layer.padding is not None
            h = _conv2d(h, _tensor(params, layer.weight), _tensor(params, layer.bias), layer.stride, layer.padding)
        elif layer.op == "relu":
            h = np.maximum(h, 0.0)
        elif layer.op == "maxpool2d":
            assert layer.kernel is not None and layer.stride is not None
            h = _maxpool(h, layer.kernel, layer.stride)
        elif layer.op == "flatten":
            h = h.reshape(h.shape[0], -1)
        else:
            raise SpecError(f"op {layer.op!r} cannot precede the dense stack")
    return h.reshape(h.shape[0], -1)


def dense_forward(
    net: NetSpec, params: Mapping[str, np.ndarray], u0: np.ndarray, start: int = 0
) -> DenseTrace:
    """Run dense layers ``start`` .. end on input ``u0`` (the input of dense layer ``start``)."""
    inputs, pre, outputs = [], [], []
    u = np.asarray(u0, dtype=np.float64)
    for layer in net.dense[start:]:
        weight = _tensor(params, layer.weight)
        if u.ndim != 2 or u.shape[1] != weight.shape[1]:
            raise ValueError(f"{layer.name}: input {u.shape} does not fit weight {weight.shape}")
        z = u @ weight.T + _tensor(params, layer.bias)
        h = np.maximum(z, 0.0) if layer.relu else z
        inputs.append(u)
        pre.append(z)
        outputs.append(h)
        u = h
    return DenseTrace(inputs=inputs, pre=pre, outputs=outputs)


def logits_from_output(
    net: NetSpec, params: Mapping[str, np.ndarray], layer_index: int, output: np.ndarray
) -> np.ndarray:
    """Logits when the output of dense layer ``layer_index`` (after its ReLU) is replaced by ``output``."""
    if layer_index >= len(net.dense) - 1:
        raise ValueError("the output of the last dense layer is the logits")
    return dense_forward(net, params, output, start=layer_index + 1).logits


def forward_logits(net: NetSpec, params: Mapping[str, np.ndarray], x_uint8: np.ndarray) -> np.ndarray:
    """Float64 logits of the re-implemented network."""
    return dense_forward(net, params, dense_input(net, params, x_uint8)).logits


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax in float64."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-example softmax cross-entropy ``logsumexp(z) - z_y`` in float64."""
    z = np.asarray(logits, dtype=np.float64)
    top = z.max(axis=1)
    lse = top + np.log(np.exp(z - top[:, None]).sum(axis=1))
    return lse - z[np.arange(len(z)), np.asarray(y, dtype=np.int64)]


def _act_grad(trace: DenseTrace, net: NetSpec, index: int) -> np.ndarray:
    if net.dense[index].relu:
        return (trace.pre[index] > 0.0).astype(np.float64)
    return np.ones_like(trace.pre[index])


def loss_weight_grads(
    net: NetSpec, params: Mapping[str, np.ndarray], trace: DenseTrace, y: np.ndarray
) -> list[np.ndarray]:
    """Gradient of the summed cross-entropy with respect to every dense weight, PyTorch layout [out, in]."""
    y = np.asarray(y, dtype=np.int64)
    probs = softmax(trace.logits)
    delta = probs
    delta[np.arange(len(y)), y] -= 1.0  # dCE/dz for the logits layer
    grads: list[np.ndarray] = [np.zeros(0)] * len(net.dense)
    for index in range(len(net.dense) - 1, -1, -1):
        grads[index] = delta.T @ trace.inputs[index]
        if index > 0:
            upstream = delta @ _tensor(params, net.dense[index].weight)  # dCE / d output of layer index - 1
            delta = upstream * _act_grad(trace, net, index - 1)
    return grads


def output_sum_grads(net: NetSpec, params: Mapping[str, np.ndarray], trace: DenseTrace) -> list[np.ndarray]:
    """Per-example gradient of the sum of all logits with respect to each dense layer's output.

    For the logits layer itself the gradient is all ones. For a hidden layer the output is taken after its ReLU,
    matching the output of a Keras ``Dense(..., activation='relu')`` layer that Arachne's code differentiates.
    """
    count = len(net.dense)
    grads: list[np.ndarray] = [np.zeros(0)] * count
    current = np.ones_like(trace.logits)  # d(sum logits) / d logits
    grads[count - 1] = current
    for index in range(count - 1, 0, -1):
        through = current * _act_grad(trace, net, index)  # d O / d pre-activation of layer index
        current = through @ _tensor(params, net.dense[index].weight)  # d O / d output of layer index - 1
        grads[index - 1] = current
    return grads


def check_parity(
    net: NetSpec, params: Mapping[str, np.ndarray], forward_fn: ForwardFn, x_uint8: np.ndarray, tol: float
) -> float:
    """Max absolute logit difference between this re-implementation and ``forward_fn``; raises above ``tol``."""
    ours = forward_logits(net, params, x_uint8)
    theirs = np.asarray(forward_fn(dict(params), x_uint8), dtype=np.float64)
    if theirs.shape != ours.shape:
        raise ValueError(f"forward returned logits {theirs.shape}, re-implementation {ours.shape}")
    diff = float(np.max(np.abs(ours - theirs))) if ours.size else 0.0
    if not np.isfinite(diff) or diff > tol:
        raise ValueError(f"re-implemented forward differs from the instance forward by {diff:.3e} > {tol:.1e}")
    return diff
