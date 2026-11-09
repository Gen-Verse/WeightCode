"""Render the numpy reference forward (``forward.py``) of a dataset from the template in ``templates/``.

The generated file depends only on the input shape, the normalization constants and the number of classes. It
contains no dataset name, seed or path, because it is shown to editors.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

TEMPLATE_NAME = "forward_numpy.py.tmpl"
PLACEHOLDERS: tuple[str, ...] = ("@@INPUT_SHAPE@@", "@@MEAN@@", "@@STD@@", "@@N_CLASSES@@")


def template_text() -> str:
    """Source text of the forward template."""
    return resources.files("weightcode.models").joinpath("templates", TEMPLATE_NAME).read_text(encoding="utf-8")


def _float_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(repr(float(value)) for value in values) + "]"


def render_forward(input_shape: Sequence[int], mean: Sequence[float], std: Sequence[float], n_classes: int) -> str:
    """Return forward.py source with the constants written in (floats as exact round-trip reprs)."""
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) != 3 or len(mean) != shape[0] or len(std) != shape[0]:
        raise ValueError(f"input shape {shape} needs {shape[0] if shape else '?'} mean/std values")
    if any(float(value) <= 0.0 for value in std):
        raise ValueError("std values must be positive")
    text = template_text()
    values = {
        "@@INPUT_SHAPE@@": repr(shape),
        "@@MEAN@@": _float_list(mean),
        "@@STD@@": _float_list(std),
        "@@N_CLASSES@@": str(int(n_classes)),
    }
    for placeholder in PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        text = text.replace(placeholder, values[placeholder])
    if "@@" in text:
        raise ValueError("unreplaced placeholder left in the forward template")
    compile(text, "forward.py", "exec")
    return text


def write_forward(path: Path, source: str) -> str:
    """Write ``source`` to ``path`` unless an identical file exists; refuse to overwrite a different one.

    Returns the sha256 of the file bytes, the value ``load_forward`` reports.
    """
    target = Path(path)
    data = source.encode("utf-8")
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"{target} exists with different content; models exported with it would be orphaned")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
    return hashlib.sha256(data).hexdigest()


SEQ_TEMPLATE_NAME = "forward_seq.py.tmpl"
SEQ_PLACEHOLDERS: tuple[str, ...] = ("@@SEQ_LEN@@", "@@VOCAB_SIZE@@", "@@N_OUT@@", "@@N_HEADS@@", "@@PAD_ID@@", "@@LN_EPS@@")


def render_seq_forward(seq_len: int, vocab_size: int, n_out: int, n_heads: int, pad_id: int, ln_eps: float) -> str:
    """forward.py source of a token-row dataset's Transformer students (template ``forward_seq.py.tmpl``)."""
    if min(seq_len, vocab_size, n_out, n_heads) <= 0 or not 0 <= pad_id < vocab_size or ln_eps <= 0:
        raise ValueError("invalid sequence forward constants")
    text = resources.files("weightcode.models").joinpath("templates", SEQ_TEMPLATE_NAME).read_text(encoding="utf-8")
    values = {
        "@@SEQ_LEN@@": str(int(seq_len)),
        "@@VOCAB_SIZE@@": str(int(vocab_size)),
        "@@N_OUT@@": str(int(n_out)),
        "@@N_HEADS@@": str(int(n_heads)),
        "@@PAD_ID@@": str(int(pad_id)),
        "@@LN_EPS@@": repr(float(ln_eps)),
    }
    for placeholder in SEQ_PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        text = text.replace(placeholder, values[placeholder])
    if "@@" in text:
        raise ValueError("unreplaced placeholder left in the sequence forward template")
    compile(text, "forward.py", "exec")
    return text


GCN_TEMPLATE_NAME = "forward_gcn.py.tmpl"
GCN_PLACEHOLDERS: tuple[str, ...] = ("@@N_NODES@@", "@@N_FEATURES@@", "@@N_CLASSES@@", "@@BN_EPS@@")


def render_gcn_forward(n_nodes: int, n_features: int, n_classes: int, bn_eps: float) -> str:
    """forward.py source of a graph dataset's GCN students (template ``forward_gcn.py.tmpl``)."""
    if min(n_nodes, n_features, n_classes) <= 0 or bn_eps <= 0:
        raise ValueError("invalid graph forward constants")
    text = resources.files("weightcode.models").joinpath("templates", GCN_TEMPLATE_NAME).read_text(encoding="utf-8")
    values = {
        "@@N_NODES@@": str(int(n_nodes)),
        "@@N_FEATURES@@": str(int(n_features)),
        "@@N_CLASSES@@": str(int(n_classes)),
        "@@BN_EPS@@": repr(float(bn_eps)),
    }
    for placeholder in GCN_PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        text = text.replace(placeholder, values[placeholder])
    if "@@" in text:
        raise ValueError("unreplaced placeholder left in the graph forward template")
    compile(text, "forward.py", "exec")
    return text


RESNET20_TEMPLATE_NAME = "forward_resnet20.py.tmpl"
RESNET20_PLACEHOLDERS: tuple[str, ...] = (
    "@@INPUT_SHAPE@@", "@@MEAN@@", "@@STD@@", "@@N_CLASSES@@", "@@SCALE@@", "@@BN_EPS@@", "@@NORM_EPS@@",
)


def render_resnet20_forward(
    mean: Sequence[float],
    std: Sequence[float],
    n_classes: int,
    scale: float,
    bn_eps: float,
    norm_eps: float,
    input_shape: Sequence[int] = (3, 32, 32),
) -> str:
    """forward.py source of a ResNet-20 student with a cosine head (template ``forward_resnet20.py.tmpl``, PREREG E9).

    The constants are written in as exact round-trip reprs. ``input_shape`` is [C, H, W]; H and W must be multiples of
    4 (two stride-2 stages). The file holds no dataset name, seed or path.
    """
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) != 3 or min(shape) <= 0 or shape[1] % 4 or shape[2] % 4:
        raise ValueError(f"input shape {shape} must be [C, H, W] with H and W multiples of 4")
    if len(mean) != shape[0] or len(std) != shape[0] or any(float(value) <= 0.0 for value in std):
        raise ValueError(f"input shape {shape} needs {shape[0]} mean values and {shape[0]} positive std values")
    if int(n_classes) < 2 or not float(scale) > 0.0 or not float(bn_eps) > 0.0 or not float(norm_eps) > 0.0:
        raise ValueError("invalid ResNet-20 forward constants")
    text = resources.files("weightcode.models").joinpath("templates", RESNET20_TEMPLATE_NAME).read_text(encoding="utf-8")
    values = {
        "@@INPUT_SHAPE@@": repr(shape),
        "@@MEAN@@": _float_list(mean),
        "@@STD@@": _float_list(std),
        "@@N_CLASSES@@": str(int(n_classes)),
        "@@SCALE@@": repr(float(scale)),
        "@@BN_EPS@@": repr(float(bn_eps)),
        "@@NORM_EPS@@": repr(float(norm_eps)),
    }
    for placeholder in RESNET20_PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        text = text.replace(placeholder, values[placeholder])
    if "@@" in text:
        raise ValueError("unreplaced placeholder left in the ResNet-20 forward template")
    compile(text, "forward.py", "exec")
    return text


RESNET20A_TEMPLATE_NAME = "forward_resnet20a.py.tmpl"
RESNET20A_PLACEHOLDERS: tuple[str, ...] = ("@@INPUT_SHAPE@@", "@@MEAN@@", "@@STD@@", "@@N_CLASSES@@", "@@BN_EPS@@")


def render_resnet20a_forward(
    mean: Sequence[float],
    std: Sequence[float],
    n_classes: int,
    bn_eps: float,
    input_shape: Sequence[int] = (3, 32, 32),
) -> str:
    """forward.py source of He et al.'s CIFAR ResNet-20 (option-A shortcuts, linear head with bias; PREREG E10).

    Template ``forward_resnet20a.py.tmpl``; constants written as exact round-trip reprs. The file holds no dataset
    name, seed or path.
    """
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) != 3 or min(shape) <= 0 or shape[1] % 4 or shape[2] % 4:
        raise ValueError(f"input shape {shape} must be [C, H, W] with H and W multiples of 4")
    if len(mean) != shape[0] or len(std) != shape[0] or any(float(value) <= 0.0 for value in std):
        raise ValueError(f"input shape {shape} needs {shape[0]} mean values and {shape[0]} positive std values")
    if int(n_classes) < 2 or not float(bn_eps) > 0.0:
        raise ValueError("invalid ResNet-20 forward constants")
    text = resources.files("weightcode.models").joinpath("templates", RESNET20A_TEMPLATE_NAME).read_text(encoding="utf-8")
    values = {
        "@@INPUT_SHAPE@@": repr(shape),
        "@@MEAN@@": _float_list(mean),
        "@@STD@@": _float_list(std),
        "@@N_CLASSES@@": str(int(n_classes)),
        "@@BN_EPS@@": repr(float(bn_eps)),
    }
    for placeholder in RESNET20A_PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        text = text.replace(placeholder, values[placeholder])
    if "@@" in text:
        raise ValueError("unreplaced placeholder left in the ResNet-20 (option A) forward template")
    compile(text, "forward.py", "exec")
    return text


VITLITE_TEMPLATE_NAME = "forward_vitlite.py.tmpl"
VITLITE_PLACEHOLDERS: tuple[str, ...] = ("@@INPUT_SHAPE@@", "@@MEAN@@", "@@STD@@", "@@N_CLASSES@@", "@@LN_EPS@@")


def render_vitlite_forward(
    mean: Sequence[float],
    std: Sequence[float],
    n_classes: int,
    ln_eps: float,
    input_shape: Sequence[int] = (3, 32, 32),
) -> str:
    """forward.py source of the ViT-Lite-7/4 student (PREREG E11; ``weightcode.models.vitlite``).

    Template ``forward_vitlite.py.tmpl``; constants written as exact round-trip reprs. The file holds no dataset
    name, seed or path.
    """
    shape = tuple(int(dim) for dim in input_shape)
    if shape != (3, 32, 32):
        raise ValueError(f"the ViT-Lite-7/4 forward is for [3, 32, 32] images, got {shape}")
    if len(mean) != shape[0] or len(std) != shape[0] or any(float(value) <= 0.0 for value in std):
        raise ValueError(f"input shape {shape} needs {shape[0]} mean values and {shape[0]} positive std values")
    if int(n_classes) < 2 or not float(ln_eps) > 0.0:
        raise ValueError("invalid ViT-Lite forward constants")
    text = resources.files("weightcode.models").joinpath("templates", VITLITE_TEMPLATE_NAME).read_text(encoding="utf-8")
    values = {
        "@@INPUT_SHAPE@@": repr(shape),
        "@@MEAN@@": _float_list(mean),
        "@@STD@@": _float_list(std),
        "@@N_CLASSES@@": str(int(n_classes)),
        "@@LN_EPS@@": repr(float(ln_eps)),
    }
    for placeholder in VITLITE_PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        text = text.replace(placeholder, values[placeholder])
    if "@@" in text:
        raise ValueError("unreplaced placeholder left in the ViT-Lite forward template")
    compile(text, "forward.py", "exec")
    return text


RESNET18UP_TEMPLATE_NAME = "forward_resnet18up.py.tmpl"
RESNET18UP_PLACEHOLDERS: tuple[str, ...] = ("@@INPUT_SHAPE@@", "@@MEAN@@", "@@STD@@", "@@N_CLASSES@@", "@@BN_EPS@@",
                                            "@@RESIZE@@")


def render_resnet18up_forward(
    mean: Sequence[float],
    std: Sequence[float],
    n_classes: int,
    bn_eps: float,
    resize: int,
    input_shape: Sequence[int] = (3, 32, 32),
) -> str:
    """forward.py source of the ResNet-18 student on upsampled images (PREREG E12; ``weightcode.models.resnet18up``).

    Template ``forward_resnet18up.py.tmpl``; constants written as exact round-trip reprs. The file holds no dataset
    name, seed or path.
    """
    shape = tuple(int(dim) for dim in input_shape)
    if shape != (3, 32, 32):
        raise ValueError(f"the ResNet-18 forward is for [3, 32, 32] images, got {shape}")
    if len(mean) != shape[0] or len(std) != shape[0] or any(float(value) <= 0.0 for value in std):
        raise ValueError(f"input shape {shape} needs {shape[0]} mean values and {shape[0]} positive std values")
    if int(n_classes) < 2 or not float(bn_eps) > 0.0 or int(resize) < shape[1] or int(resize) % 32:
        raise ValueError("invalid ResNet-18 forward constants")
    text = resources.files("weightcode.models").joinpath("templates", RESNET18UP_TEMPLATE_NAME).read_text(encoding="utf-8")
    values = {
        "@@INPUT_SHAPE@@": repr(shape),
        "@@MEAN@@": _float_list(mean),
        "@@STD@@": _float_list(std),
        "@@N_CLASSES@@": str(int(n_classes)),
        "@@BN_EPS@@": repr(float(bn_eps)),
        "@@RESIZE@@": str(int(resize)),
    }
    for placeholder in RESNET18UP_PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        text = text.replace(placeholder, values[placeholder])
    if "@@" in text:
        raise ValueError("unreplaced placeholder left in the ResNet-18 forward template")
    compile(text, "forward.py", "exec")
    return text


RESNET18NATIVE_TEMPLATE_NAME = "forward_resnet18native.py.tmpl"
RESNET18NATIVE_PLACEHOLDERS: tuple[str, ...] = ("@@INPUT_SHAPE@@", "@@N_CLASSES@@", "@@BN_EPS@@")


def render_resnet18native_forward(n_classes: int, bn_eps: float, input_shape: Sequence[int] = (3, 224, 224)) -> str:
    """forward.py source of the native ResNet-18 on preprocessed images (PREREG T03; ``weightcode.models.resnet18native``).

    Template ``forward_resnet18native.py.tmpl``: E12's network without its upsample layer, whose input is the
    preprocessed float tensor (the preprocessing runs once outside the network). Constants written as exact
    round-trip reprs. The file holds no dataset name, seed or path.
    """
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) != 3 or shape[0] != 3 or min(shape) <= 0 or shape[1] % 32 or shape[2] % 32:
        raise ValueError(f"input shape {shape} must be [3, H, W] with H and W multiples of 32")
    if int(n_classes) < 2 or not float(bn_eps) > 0.0:
        raise ValueError("invalid ResNet-18 forward constants")
    text = resources.files("weightcode.models").joinpath("templates", RESNET18NATIVE_TEMPLATE_NAME).read_text(
        encoding="utf-8")
    values = {
        "@@INPUT_SHAPE@@": repr(shape),
        "@@N_CLASSES@@": str(int(n_classes)),
        "@@BN_EPS@@": repr(float(bn_eps)),
    }
    for placeholder in RESNET18NATIVE_PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        text = text.replace(placeholder, values[placeholder])
    if "@@" in text:
        raise ValueError("unreplaced placeholder left in the native ResNet-18 forward template")
    compile(text, "forward.py", "exec")
    return text


RESNET18NATIVE_U8_TEMPLATE_NAME = "forward_resnet18native_u8.py.tmpl"
RESNET18NATIVE_U8_PLACEHOLDERS: tuple[str, ...] = ("@@INPUT_SHAPE@@", "@@MEAN@@", "@@STD@@", "@@N_CLASSES@@",
                                                   "@@BN_EPS@@")


def render_resnet18native_u8_forward(mean: Sequence[float], std: Sequence[float], n_classes: int, bn_eps: float,
                                     input_shape: Sequence[int] = (3, 224, 224)) -> str:
    """forward.py source of the native ResNet-18 on uint8 center crops (PREREG T04; ``weightcode.models.resnet18native``).

    Template ``forward_resnet18native_u8.py.tmpl``: T03's network (``forward_resnet18native.py.tmpl``) with the last
    two preprocessing steps of torchvision's eval transform moved inside, as E12's forward has them: x / 255, then
    (x - mean[c]) / std[c] per channel, in float64. The resize and the center crop stay outside (they make the stored
    uint8 image). Constants written as exact round-trip reprs. The file holds no dataset name, seed or path.
    """
    shape = tuple(int(dim) for dim in input_shape)
    if len(shape) != 3 or shape[0] != 3 or min(shape) <= 0 or shape[1] % 32 or shape[2] % 32:
        raise ValueError(f"input shape {shape} must be [3, H, W] with H and W multiples of 32")
    if len(mean) != shape[0] or len(std) != shape[0] or any(float(value) <= 0.0 for value in std):
        raise ValueError(f"input shape {shape} needs {shape[0]} mean values and {shape[0]} positive std values")
    if int(n_classes) < 2 or not float(bn_eps) > 0.0:
        raise ValueError("invalid ResNet-18 forward constants")
    text = resources.files("weightcode.models").joinpath("templates", RESNET18NATIVE_U8_TEMPLATE_NAME).read_text(
        encoding="utf-8")
    values = {
        "@@INPUT_SHAPE@@": repr(shape),
        "@@MEAN@@": _float_list(mean),
        "@@STD@@": _float_list(std),
        "@@N_CLASSES@@": str(int(n_classes)),
        "@@BN_EPS@@": repr(float(bn_eps)),
    }
    for placeholder in RESNET18NATIVE_U8_PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        text = text.replace(placeholder, values[placeholder])
    if "@@" in text:
        raise ValueError("unreplaced placeholder left in the native uint8 ResNet-18 forward template")
    compile(text, "forward.py", "exec")
    return text
