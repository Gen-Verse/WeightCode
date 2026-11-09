"""Public model description ``model_spec.json`` (schema ``weightcode.model_spec.v0``, contract section 3).

The spec is shown to editors. It carries the opaque model id, input format and preprocessing constants, class
names, the ordered layer list with tensor names and shapes, and output semantics. It never carries a seed, a
dataset name, a role or a parameter hash: the same spec is reused for every arm of an instance, including arms
that show different weights, so a hash in it would reveal the substitution.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from weightcode.models import arch as A

SCHEMA = "weightcode.model_spec.v0"
FORBIDDEN_TERMS: tuple[str, ...] = ("seed", "mnist", "fashion", "cifar", "colored", "ceiling", "sha256", "role")
FLATTEN_ORDER = "C, H, W (row-major)"


def _linear(name: str, params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    weight, bias = params[f"{name}.weight"], params[f"{name}.bias"]
    return {
        "name": name,
        "op": "linear",
        "weight": f"{name}.weight",
        "bias": f"{name}.bias",
        "shape_w": list(weight.shape),
        "shape_b": list(bias.shape),
        "computes": "y = x @ weight.T + bias",
    }


def _conv(name: str, params: Mapping[str, np.ndarray], in_shape: Sequence[int]) -> tuple[dict[str, Any], list[int]]:
    weight, bias = params[f"{name}.weight"], params[f"{name}.bias"]
    out_c, _, kh, kw = weight.shape
    h = (in_shape[1] + 2 * A.CONV_PADDING - kh) // A.CONV_STRIDE + 1
    w = (in_shape[2] + 2 * A.CONV_PADDING - kw) // A.CONV_STRIDE + 1
    out_shape = [int(out_c), int(h), int(w)]
    layer = {
        "name": name,
        "op": "conv2d",
        "weight": f"{name}.weight",
        "bias": f"{name}.bias",
        "shape_w": list(weight.shape),
        "shape_b": list(bias.shape),
        "kernel": [int(kh), int(kw)],
        "stride": [A.CONV_STRIDE, A.CONV_STRIDE],
        "padding": [A.CONV_PADDING, A.CONV_PADDING],
        "padding_mode": "zeros",
        "dilation": [1, 1],
        "groups": 1,
        "computes": "cross-correlation as torch.nn.functional.conv2d: out[n, o, i, j] = bias[o] + "
        "sum_{c, u, v} weight[o, c, u, v] * xpad[n, c, i * stride + u, j * stride + v]",
        "in_shape": [int(d) for d in in_shape],
        "out_shape": out_shape,
    }
    return layer, out_shape


def _pool(in_shape: Sequence[int]) -> tuple[dict[str, Any], list[int]]:
    h = (in_shape[1] - A.POOL_KERNEL) // A.POOL_STRIDE + 1
    w = (in_shape[2] - A.POOL_KERNEL) // A.POOL_STRIDE + 1
    out_shape = [int(in_shape[0]), int(h), int(w)]
    layer = {
        "op": "maxpool2d",
        "kernel": [A.POOL_KERNEL, A.POOL_KERNEL],
        "stride": [A.POOL_STRIDE, A.POOL_STRIDE],
        "padding": [0, 0],
        "ceil_mode": False,
        "in_shape": [int(d) for d in in_shape],
        "out_shape": out_shape,
    }
    return layer, out_shape


def layer_list(arch: str, input_shape: Sequence[int], params: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Ordered layer list of ``arch`` with tensor names and shapes taken from ``params``."""
    shape = [int(d) for d in input_shape]
    layers: list[dict[str, Any]] = []
    if arch in A.FC_LAYERS:
        fcs = list(A.FC_LAYERS[arch])
        for name in A.CONV_LAYERS[arch]:
            conv, shape = _conv(name, params, shape)
            layers.append(conv)
            layers.append({"op": "relu"})
            pool, shape = _pool(shape)
            layers.append(pool)
    else:
        raise KeyError(f"unknown arch {arch!r}")
    flat = int(np.prod(shape))
    layers.append({"op": "flatten", "order": FLATTEN_ORDER, "in_shape": shape, "out_features": flat})
    for position, name in enumerate(fcs):
        layer = _linear(name, params)
        if layer["shape_w"][1] != flat:
            raise ValueError(f"{name} expects {layer['shape_w'][1]} inputs, previous layer gives {flat}")
        layers.append(layer)
        flat = layer["shape_w"][0]
        if position < len(fcs) - 1:
            layers.append({"op": "relu"})
    expected = {f"{layer['name']}.{kind}" for layer in layers if "name" in layer for kind in ("weight", "bias")}
    if expected != set(params):
        raise ValueError(f"parameter names {sorted(params)} do not match the {arch} layer list {sorted(expected)}")
    return layers


def build_model_spec(
    model_id: str,
    arch: str,
    input_shape: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
    class_names: Sequence[str],
    params: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Assemble the public spec and check it contains none of the forbidden identifying terms."""
    n_params = int(sum(np.asarray(value).size for value in params.values()))
    spec = {
        "schema": SCHEMA,
        "model_id": model_id,
        "input": {
            "shape": [int(d) for d in input_shape],
            "dtype": "uint8",
            "range": [0, 255],
            "preprocess": "x/255 then (x - mean) / std, per channel",
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "layout": "NCHW, flatten row-major",
        },
        "classes": [str(name) for name in class_names],
        "parameter_layout": "PyTorch: linear weight [out, in]; conv weight [out, in, kh, kw]; all float32",
        "n_parameters": n_params,
        "layers": layer_list(arch, input_shape, params),
        "output": "logits; prediction = argmax",
        "reference_forward": "forward.py: forward(params, x_uint8) -> float32 logits [N, n_classes]",
    }
    check_public(spec)
    return spec


def check_public(spec: Mapping[str, Any]) -> None:
    """Raise if any key or string value of ``spec`` (class names excepted) contains a forbidden term."""
    payload = json.dumps({key: value for key, value in spec.items() if key != "classes"}).lower()
    hits = [term for term in FORBIDDEN_TERMS if term in payload]
    if hits:
        raise ValueError(f"model_spec contains forbidden terms {hits}")
