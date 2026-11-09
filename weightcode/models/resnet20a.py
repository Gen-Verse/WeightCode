"""The ResNet-20 student of PREREG E10 (arch ``resnet20_a``): He et al. 2016's CIFAR ResNet-20 with option-A shortcuts.

Architecture (He, Zhang, Ren and Sun, "Deep Residual Learning for Image Recognition", CVPR 2016, section 4.2; layer
names and initialisation as in the widely used reproduction akamaster/pytorch_resnet_cifar10, ``resnet.py``):

* stem: 3x3 convolution 3 -> 16 (stride 1, padding 1, no bias), BatchNorm, ReLU;
* three stages of three BasicBlocks with 16, 32 and 64 channels and strides 1, 2, 2. A block computes
  ``relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x))``; both convolutions are 3x3 with padding 1 and no bias and
  conv1 carries the block's stride. The shortcut has no parameters (option A): the identity, except in the first
  block of stages 2 and 3, where it is ``x[:, :, ::2, ::2]`` with the channels zero-padded by out_channels // 4 on
  each side;
* global average pooling to 64 features;
* a linear head ``fc`` with bias, 64 -> 10.

Initialisation: ``kaiming_normal_`` (fan-in, the default) on every convolution and on the head's weight, PyTorch's
default for the head's bias, BatchNorm weight 1 and bias 0. Parameters are named as ``param_shapes`` (97 tensors,
BatchNorm running statistics included, num_batches_tracked excluded; 271,098 entries, 269,722 trainable). The model takes raw uint8
NCHW images and applies x / 255 then per-channel (x - mean) / std itself, in its own dtype, so a float64 rebuild is an
exact reference for forward.py (``weightcode.models.forward_gen.render_resnet20a_forward``).

Training (``train_resnet20a``): the paper's recipe, SGD lr 0.1, momentum 0.9 (no nesterov), weight decay 1e-4 on
every parameter, batch 128 with shuffling, 182 epochs with the learning rate divided by 10 at epochs 91 and 136 (the
paper's 32k and 48k of 64k iterations on 45k images, per epoch here), random crop 32 after zero padding by 4 and
horizontal flip (``weightcode.models.train.augment_crop_flip``), cross-entropy. The model after the last epoch is kept.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from weightcode.models.resnet20 import (
    BLOCKS_PER_STAGE,
    BN_EPS,
    BN_PARTS,
    EVAL_BATCH,
    INPUT_SHAPE,
    N_FEATURES,
    STAGE_CHANNELS,
    STAGE_STRIDES,
    STEM_CHANNELS,
    Preprocess,
    _Indices,
)
from weightcode.models.train import AUGMENT_SEED_OFFSET, augment_crop_flip

ARCH = "resnet20_a"
SPEC_SCHEMA = "weightcode.model_spec.resnet20a.v1"
HEAD_WEIGHT = "fc.weight"
HEAD_BIAS = "fc.bias"

RECIPE = (
    "He et al. 2016 section 4.2 (CIFAR-10 ResNet-20, option A): SGD lr 0.1, momentum 0.9, weight decay 1e-4, batch "
    "128, learning rate divided by 10 at 32k and 48k iterations and stopped at 64k (45k images, about 182 epochs), "
    "random crop with padding 4 and horizontal flip. Here: 182 epochs, milestones 91 and 136 per epoch (PREREG "
    "E10); trains on E1's 40,000 train rows (a recorded deviation from 50,000); E1's per-channel normalisation "
    "(dataset.json) instead of per-pixel mean subtraction; the model after the last epoch is kept; initialisation "
    "as in akamaster/pytorch_resnet_cifar10."
)


def block_names() -> list[tuple[str, int, int, int]]:
    """(block name, in channels, out channels, stride) of the nine BasicBlocks in forward order."""
    blocks = []
    in_channels = STEM_CHANNELS
    for stage, (channels, stride) in enumerate(zip(STAGE_CHANNELS, STAGE_STRIDES), start=1):
        for index in range(BLOCKS_PER_STAGE):
            blocks.append((f"layer{stage}.{index}", in_channels, channels, stride if index == 0 else 1))
            in_channels = channels
    return blocks


def has_subsample(in_channels: int, out_channels: int, stride: int) -> bool:
    """Option A's rule: the parameter-free subsampling shortcut where the stride or the width changes."""
    return stride != 1 or in_channels != out_channels


def param_shapes(n_classes: int) -> dict[str, tuple[int, ...]]:
    """Every parameter name and shape of the student, in forward order (the exported model.npz layout)."""
    shapes: dict[str, tuple[int, ...]] = {"conv1.weight": (STEM_CHANNELS, INPUT_SHAPE[0], 3, 3)}
    shapes.update({f"bn1.{part}": (STEM_CHANNELS,) for part in BN_PARTS})
    for name, c_in, c_out, _stride in block_names():
        shapes[f"{name}.conv1.weight"] = (c_out, c_in, 3, 3)
        shapes.update({f"{name}.bn1.{part}": (c_out,) for part in BN_PARTS})
        shapes[f"{name}.conv2.weight"] = (c_out, c_out, 3, 3)
        shapes.update({f"{name}.bn2.{part}": (c_out,) for part in BN_PARTS})
    shapes[HEAD_WEIGHT] = (n_classes, N_FEATURES)
    shapes[HEAD_BIAS] = (n_classes,)
    return shapes


class Subsample(nn.Module):
    """Option A: keep every second row and column, zero-pad the channels by ``out_channels // 4`` on each side."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.pad = out_channels // 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.pad, self.pad), "constant", 0.0)


class BasicBlockA(nn.Module):
    """conv3x3 (stride) - BN - ReLU - conv3x3 - BN, plus the option-A shortcut, then ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.shortcut: nn.Module = (Subsample(out_channels) if has_subsample(in_channels, out_channels, stride)
                                    else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet20A(nn.Module):
    """He et al.'s CIFAR ResNet-20 (option A) with a linear head; raw uint8 images in, logits out."""

    def __init__(self, n_classes: int, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        self.preprocess = Preprocess(mean, std)
        self.conv1 = nn.Conv2d(INPUT_SHAPE[0], STEM_CHANNELS, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(STEM_CHANNELS, eps=BN_EPS)
        stages: dict[int, list[BasicBlockA]] = {1: [], 2: [], 3: []}
        for name, c_in, c_out, stride in block_names():
            stages[int(name[len("layer")])].append(BasicBlockA(c_in, c_out, stride))
        self.layer1 = nn.Sequential(*stages[1])
        self.layer2 = nn.Sequential(*stages[2])
        self.layer3 = nn.Sequential(*stages[3])
        self.fc = nn.Linear(N_FEATURES, n_classes)
        for module in self.modules():  # akamaster's _weights_init
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled encoder features [N, 64] of raw images."""
        h = F.relu(self.bn1(self.conv1(self.preprocess(x, self.conv1.weight.dtype))))
        h = self.layer3(self.layer2(self.layer1(h)))
        return h.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def export_params(model: ResNet20A) -> dict[str, np.ndarray]:
    """float32 numpy copies of every exported tensor, named as ``param_shapes`` (no num_batches_tracked)."""
    state = model.state_dict()
    expected = param_shapes(model.fc.out_features)
    params = {}
    for name, shape in expected.items():
        value = state[name].detach().to("cpu", torch.float32).contiguous().numpy().copy()
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
        params[name] = value
    extra = sorted(set(state) - set(expected) - {key for key in state if key.endswith("num_batches_tracked")})
    if extra:
        raise ValueError(f"model state has unexported tensors {extra}")
    return params


def check_params(params: Mapping[str, np.ndarray], n_classes: int) -> None:
    """Raise unless ``params`` has exactly the student's names and shapes, as finite float arrays."""
    expected = param_shapes(n_classes)
    if set(params) != set(expected):
        raise KeyError(f"parameter names differ: missing {sorted(set(expected) - set(params))}, "
                       f"unexpected {sorted(set(params) - set(expected))}")
    for name, shape in expected.items():
        value = np.asarray(params[name])
        if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{name}: {value.dtype}{value.shape}, expected float {shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} holds non-finite values")


def load_params_into(model: ResNet20A, params: Mapping[str, np.ndarray]) -> ResNet20A:
    """Copy exported parameters into ``model`` (in its dtype); num_batches_tracked keeps the model's own value."""
    check_params(params, model.fc.out_features)
    state = model.state_dict()
    dtype = model.fc.weight.dtype
    for name in param_shapes(model.fc.out_features):
        state[name] = torch.from_numpy(np.asarray(params[name])).to(dtype)
    model.load_state_dict(state, strict=True)
    return model


def build_resnet20a(n_classes: int, mean: Sequence[float], std: Sequence[float],
                    params: Mapping[str, np.ndarray] | None = None, dtype: torch.dtype = torch.float32) -> ResNet20A:
    """A CPU model in ``dtype`` (eval mode), with ``params`` loaded when given."""
    model = ResNet20A(n_classes, mean, std).to(dtype)
    if params is not None:
        load_params_into(model, params)
    return model.eval()


@torch.no_grad()
def eval_logits(model: ResNet20A, x: np.ndarray, batch_size: int = EVAL_BATCH,
                device: torch.device | None = None) -> np.ndarray:
    """Eval-mode logits of raw uint8 images ``x`` as a float64 numpy array (computed in the model's dtype)."""
    was_training = model.training
    model.eval()
    target = next(model.parameters()).device if device is None else device
    outputs = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size])).to(target)
        outputs.append(model(batch).to("cpu", torch.float64).numpy())
    model.train(was_training)
    return np.concatenate(outputs) if outputs else np.zeros((0, model.fc.out_features))


def _shape(params: Mapping[str, np.ndarray], name: str) -> list[int]:
    return [int(size) for size in np.shape(params[name])]


def _bn(prefix: str, params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {"tensors": {part: f"{prefix}.{part}" for part in BN_PARTS}, "shape": _shape(params, f"{prefix}.weight"),
            "eps": BN_EPS}


def _conv(name: str, params: Mapping[str, np.ndarray], stride: int, padding: int) -> dict[str, Any]:
    return {"weight": name, "shape": _shape(params, name), "stride": stride, "padding": padding, "bias": None}


BN_COMPUTES = "h = (h - running_mean) / sqrt(running_var + eps) * weight + bias, per channel"
CONV_COMPUTES = ("2-D cross-correlation as torch.nn.functional.conv2d with the given stride and zero padding, no "
                 "bias")


def build_resnet20a_spec(model_id: str, class_names: Sequence[str], mean: Sequence[float], std: Sequence[float],
                         params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Public ``model_spec.json`` of a ``resnet20_a`` student (schema ``SPEC_SCHEMA``), checked for forbidden terms
    (``weightcode.models.spec.check_public``)."""
    from weightcode.models.spec import check_public

    names = [str(name) for name in class_names]
    check_params(params, len(names))
    layers: list[dict[str, Any]] = [
        {"name": "conv1", "op": "conv2d", **_conv("conv1.weight", params, 1, 1), "computes": CONV_COMPUTES},
        {"name": "bn1", "op": "batchnorm_eval", **_bn("bn1", params), "computes": BN_COMPUTES},
        {"name": "relu", "op": "relu"},
    ]
    for name, c_in, c_out, stride in block_names():
        shortcut = None
        if has_subsample(c_in, c_out, stride):
            shortcut = {"op": "subsample_zero_pad", "stride": 2, "channel_pad_each_side": c_out // 4,
                        "computes": "s = x[:, :, ::2, ::2] with the channels zero-padded on each side; no parameters"}
        layers.append({
            "name": name, "op": "basic_block", "in_channels": c_in, "out_channels": c_out, "stride": stride,
            "conv1": _conv(f"{name}.conv1.weight", params, stride, 1), "bn1": _bn(f"{name}.bn1", params),
            "conv2": _conv(f"{name}.conv2.weight", params, 1, 1), "bn2": _bn(f"{name}.bn2", params),
            "shortcut": shortcut,
            "computes": "out = relu(bn2(conv2(relu(bn1(conv1(x))))) + s), s = the shortcut of x, or s = x when the "
                        "shortcut is null",
        })
    layers.append({"name": "pool", "op": "global_avg_pool", "computes": "features = mean of h over H and W, [N, 64]"})
    layers.append({"name": "fc", "op": "linear", "weight": HEAD_WEIGHT, "bias": HEAD_BIAS,
                   "shape": _shape(params, HEAD_WEIGHT), "computes": "logits = f @ weight.T + bias"})
    spec = {
        "schema": SPEC_SCHEMA,
        "model_id": model_id,
        "arch": ARCH,
        "input": {"kind": "images", "shape": list(INPUT_SHAPE), "dtype": "uint8", "layout": "NCHW"},
        "input_shape": list(INPUT_SHAPE),
        "preprocessing": {"mean": [float(v) for v in mean], "std": [float(v) for v in std],
                          "computes": "x / 255, then per channel (x - mean[c]) / std[c]"},
        "classes": names,
        "n_classes": len(names),
        "parameter_layout": "PyTorch: conv weight [out, in, kh, kw]; BatchNorm weight, bias, running_mean, "
                            "running_var [channels]; head weight [n_classes, 64] and bias [n_classes]; all float32",
        "n_parameters": int(sum(np.asarray(value).size for value in params.values())),
        "tensors": {name: _shape(params, name) for name in param_shapes(len(names))},
        "layers": layers,
        "output": "logits [N, n_classes]; the prediction is the argmax",
        "reference_forward": "forward.py: forward(params, x_uint8 [N, 3, 32, 32]) -> float32 logits [N, n_classes]",
    }
    check_public(spec)
    return spec


# ---------------------------------------------------------------------------------------------------------------
# Training


@dataclass(frozen=True)
class HeTrainConfig:
    """He et al.'s CIFAR recipe as PREREG E10 fixes it (module docstring); ``RECIPE`` states the deviations."""

    arch: str = ARCH
    epochs: int = 182
    batch_size: int = 128
    optimizer: str = "SGD"
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 1e-4
    schedule: str = "MultiStepLR"
    milestones: tuple[int, ...] = (91, 136)
    gamma: float = 0.1
    loss: str = "cross_entropy"
    augment: str = "random_crop_pad4_hflip"
    augment_pad: int = 4
    shuffle: bool = True
    drop_last: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HeTrainResult:
    """The model after the last epoch (on the training device) with its per-epoch log."""

    model: ResNet20A
    log: list[dict[str, float]] = field(default_factory=list)
    final: dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0


def train_resnet20a(
    config: HeTrainConfig,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    log_fn: Callable[[Mapping[str, Any]], None] | None = None,
) -> HeTrainResult:
    """Train one student from scratch on raw uint8 images ``x_train`` [N, 3, 32, 32] and labels ``y_train``.

    The caller sets determinism first (``weightcode.models.train.configure_determinism``). Initialization draws from
    ``torch.manual_seed(seed)``, shuffling from a generator seeded with ``seed`` and augmentation from one seeded with
    ``seed + AUGMENT_SEED_OFFSET``, as for E9's students. The whole training set lives on ``device``. ``final``
    holds the clean (eval-mode, unaugmented) loss and accuracy on the training rows after the last epoch.
    """
    if config.optimizer != "SGD" or config.schedule != "MultiStepLR" or config.loss != "cross_entropy":
        raise ValueError(f"unsupported recipe {config.optimizer}/{config.schedule}/{config.loss}")
    if config.augment != "random_crop_pad4_hflip":
        raise ValueError(f"unknown augmentation {config.augment!r}")
    x_np, y_np = np.asarray(x_train), np.asarray(y_train)
    if x_np.dtype != np.uint8 or x_np.ndim != 4 or tuple(x_np.shape[1:]) != INPUT_SHAPE:
        raise TypeError(f"training images must be uint8 [N, {INPUT_SHAPE}], got {x_np.dtype}{x_np.shape}")
    if y_np.dtype != np.int64 or y_np.shape != (len(x_np),) or y_np.min() < 0 or y_np.max() >= n_classes:
        raise ValueError(f"labels must be int64 [N] in 0 .. {n_classes - 1}")
    torch.manual_seed(seed)
    model = ResNet20A(n_classes, mean, std).to(device)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device)
    y = torch.from_numpy(y_np).to(device)
    loader = DataLoader(_Indices(len(x_np)), batch_size=config.batch_size, shuffle=config.shuffle,
                        drop_last=config.drop_last, generator=torch.Generator().manual_seed(seed), num_workers=0)
    augment_generator = torch.Generator().manual_seed(seed + AUGMENT_SEED_OFFSET)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, nesterov=config.nesterov,
                                weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(config.milestones),
                                                     gamma=config.gamma)
    result = HeTrainResult(model=model)
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        correct = torch.zeros((), dtype=torch.int64, device=device)
        seen = 0
        lr = float(scheduler.get_last_lr()[0])
        for index in loader:
            index = index.to(device)
            xb = augment_crop_flip(x[index], config.augment_pad, augment_generator)
            yb = y[index]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().to(torch.float64) * yb.shape[0]
            correct += (logits.detach().argmax(dim=1) == yb).sum()
            seen += int(yb.shape[0])
        scheduler.step()
        train_loss = float(loss_sum.item()) / seen
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        entry = {"epoch": epoch, "lr": lr, "train_loss": train_loss,
                 "train_acc_augmented": float(correct.item()) / seen, "epoch_time_s": time.perf_counter() - epoch_start}
        result.log.append(entry)
        if log_fn is not None:
            log_fn(entry)
    model.eval()
    with torch.no_grad():
        loss_sum_f, correct_i = 0.0, 0
        for start in range(0, len(x_np), EVAL_BATCH):
            logits = model(x[start:start + EVAL_BATCH])
            yb = y[start:start + EVAL_BATCH]
            loss_sum_f += float(F.cross_entropy(logits, yb, reduction="sum").to(torch.float64).item())
            correct_i += int((logits.argmax(dim=1) == yb).sum().item())
    result.final = {"train_loss_clean": loss_sum_f / len(x_np), "train_acc_clean": correct_i / len(x_np),
                    "n_train": int(len(x_np))}
    result.wall_time_s = time.perf_counter() - started
    return result


__all__ = [
    "ARCH", "BasicBlockA", "build_resnet20a_spec", "HEAD_BIAS", "HEAD_WEIGHT", "HeTrainConfig", "HeTrainResult", "RECIPE", "ResNet20A",
    "SPEC_SCHEMA", "Subsample", "block_names", "build_resnet20a", "check_params", "eval_logits", "export_params",
    "has_subsample", "load_params_into", "param_shapes", "train_resnet20a",
]
