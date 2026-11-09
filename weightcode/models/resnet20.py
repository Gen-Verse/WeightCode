"""The ResNet-20 student of PREREG E9 (arch ``resnet20_cosine``): CEC's CIFAR ResNet-20 with CEC's cosine head.

Architecture (icoz69/CEC-CVPR2021 at 3a5d335, ``models/resnet20_cifar.py`` and ``models/base/Network.py``):

* stem: 3x3 convolution 3 -> 16 (stride 1, padding 1, no bias), BatchNorm, ReLU;
* three stages of three BasicBlocks with 16, 32 and 64 channels and strides 1, 2, 2. A block computes
  ``relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x))``; both convolutions are 3x3 with padding 1 and no bias,
  conv1 carries the block's stride. The shortcut is the identity, except in the first block of stages 2 and 3
  (``downsample``: a 1x1 convolution with the block's stride and no bias, then BatchNorm; CEC's option B);
* global average pooling to 64 features (a mean over H and W; CEC uses ``adaptive_avg_pool2d(x, 1)``, the same value,
  whose CUDA backward has no deterministic implementation);
* the cosine head ``fc`` = ``nn.Linear(64, n_classes, bias=False)``: logits = ``scale * normalize(f) @
  normalize(fc.weight).T`` with ``normalize`` = ``F.normalize`` (row-wise L2, eps 1e-12) and scale 16 (CEC's
  ``temperature``).

Parameter names are CEC's state_dict names without the ``encoder.`` prefix and without ``num_batches_tracked``
(``PARAM_NAMES``, 98 tensors, 279,792 entries). The model takes raw uint8 NCHW images and applies x / 255 then
per-channel (x - mean) / std itself, in its own dtype, so a float64 rebuild is an exact reference for forward.py.

Training (``train_resnet20``): CEC's base stage (README pretrain line; ``base/fscil_trainer.py``
``get_optimizer_base``; ``base/helper.py`` ``base_train``): SGD lr 0.1, momentum 0.9, nesterov, weight decay 5e-4 on
every parameter, batch 128 with shuffling, 100 epochs, MultiStepLR milestones 60 and 70 with gamma 0.1 stepped once
per epoch, random crop 32 after zero padding by 4 and horizontal flip (``weightcode.models.train.augment_crop_flip``, the
same operation as torchvision RandomCrop(32, padding=4) and RandomHorizontalFlip on the raw image), cross-entropy over
all ``n_classes`` outputs (``RECIPE`` states the deviations from CEC). The model after the last epoch is kept.
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
from torch.utils.data import DataLoader, Dataset

from weightcode.models.train import AUGMENT_SEED_OFFSET, augment_crop_flip

ARCH = "resnet20_cosine"
SPEC_SCHEMA = "weightcode.model_spec.resnet20.v1"
INPUT_SHAPE: tuple[int, int, int] = (3, 32, 32)
STEM_CHANNELS = 16
STAGE_CHANNELS: tuple[int, int, int] = (16, 32, 64)
STAGE_STRIDES: tuple[int, int, int] = (1, 2, 2)
BLOCKS_PER_STAGE = 3
N_FEATURES = 64
COSINE_SCALE = 16.0
NORM_EPS = 1e-12
BN_EPS = 1e-5
BN_PARTS: tuple[str, ...] = ("weight", "bias", "running_mean", "running_var")
HEAD = "fc.weight"
EVAL_BATCH = 1024

RECIPE = (
    "CEC base stage (icoz69/CEC-CVPR2021 3a5d335: README pretrain line, base/fscil_trainer.py get_optimizer_base, "
    "base/helper.py base_train): SGD lr 0.1, momentum 0.9, nesterov, weight decay 5e-4, batch 128, 100 epochs, "
    "MultiStepLR milestones 60 and 70, gamma 0.1, random crop with padding 4 and horizontal flip, cosine head with "
    "temperature 16. Deviations (PREREG E9): cross-entropy over all 100 outputs (CEC slices the logits to the 60 base "
    "classes); the model after the last epoch is kept (CEC keeps the epoch with the best test accuracy); the base "
    "trains on 430 images per base class (old_select, old_check and old_replay held out); CEC's replacement of the "
    "head rows by training-set class means after the base stage (replace_base_fc) is not applied, so the head is the "
    "trained one; global average pooling is a mean over H and W (adaptive_avg_pool2d to 1, same value)."
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


def has_downsample(in_channels: int, out_channels: int, stride: int) -> bool:
    """CEC's rule: a downsample shortcut where the stride or the width changes."""
    return stride != 1 or in_channels != out_channels


def param_shapes(n_classes: int) -> dict[str, tuple[int, ...]]:
    """Every parameter name and shape of the student, in forward order (the exported model.npz layout)."""
    shapes: dict[str, tuple[int, ...]] = {"conv1.weight": (STEM_CHANNELS, INPUT_SHAPE[0], 3, 3)}
    shapes.update({f"bn1.{part}": (STEM_CHANNELS,) for part in BN_PARTS})
    for name, c_in, c_out, stride in block_names():
        shapes[f"{name}.conv1.weight"] = (c_out, c_in, 3, 3)
        shapes.update({f"{name}.bn1.{part}": (c_out,) for part in BN_PARTS})
        shapes[f"{name}.conv2.weight"] = (c_out, c_out, 3, 3)
        shapes.update({f"{name}.bn2.{part}": (c_out,) for part in BN_PARTS})
        if has_downsample(c_in, c_out, stride):
            shapes[f"{name}.downsample.0.weight"] = (c_out, c_in, 1, 1)
            shapes.update({f"{name}.downsample.1.{part}": (c_out,) for part in BN_PARTS})
    shapes[HEAD] = (n_classes, N_FEATURES)
    return shapes


class Preprocess(nn.Module):
    """x / 255, then per-channel (x - mean) / std, in a given dtype.

    The constants are float64 buffers (non-persistent, so not in the state dict) cast to the compute dtype at use: a
    float64 rebuild uses the exact decimal constants forward.py uses, and a float32 model their float32 roundings.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        if len(mean) != INPUT_SHAPE[0] or len(std) != INPUT_SHAPE[0]:
            raise ValueError(f"mean and std need {INPUT_SHAPE[0]} values")
        self.register_buffer("mean", torch.tensor(list(mean), dtype=torch.float64).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(list(std), dtype=torch.float64).view(1, -1, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (x.to(dtype) / 255.0 - self.mean.to(dtype)) / self.std.to(dtype)


class BasicBlock(nn.Module):
    """CEC's BasicBlock: conv3x3 (stride) - BN - ReLU - conv3x3 - BN, plus the shortcut, then ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.downsample: nn.Module | None = None
        if has_downsample(in_channels, out_channels, stride):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=BN_EPS),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = x if self.downsample is None else self.downsample(x)
        return F.relu(out + residual)


class ResNet20Cosine(nn.Module):
    """CEC's ResNet-20 encoder with the cosine head; raw uint8 images in, cosine logits out (module docstring)."""

    def __init__(self, n_classes: int, mean: Sequence[float], std: Sequence[float],
                 scale: float = COSINE_SCALE) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        self.scale = float(scale)
        self.preprocess = Preprocess(mean, std)
        self.conv1 = nn.Conv2d(INPUT_SHAPE[0], STEM_CHANNELS, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(STEM_CHANNELS, eps=BN_EPS)
        stages: dict[int, list[BasicBlock]] = {1: [], 2: [], 3: []}
        for name, c_in, c_out, stride in block_names():
            stages[int(name[len("layer")])].append(BasicBlock(c_in, c_out, stride))
        self.layer1 = nn.Sequential(*stages[1])
        self.layer2 = nn.Sequential(*stages[2])
        self.layer3 = nn.Sequential(*stages[3])
        # CEC's encoder initialisation, before the head is created (as in MYNET: the encoder, then the fc).
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        self.fc = nn.Linear(N_FEATURES, n_classes, bias=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled encoder features [N, 64] of raw images."""
        h = F.relu(self.bn1(self.conv1(self.preprocess(x, self.conv1.weight.dtype))))
        h = self.layer3(self.layer2(self.layer1(h)))
        return h.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = F.normalize(self.features(x), p=2, dim=-1, eps=NORM_EPS)
        weight = F.normalize(self.fc.weight, p=2, dim=-1, eps=NORM_EPS)
        return self.scale * F.linear(features, weight)


def export_params(model: ResNet20Cosine) -> dict[str, np.ndarray]:
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


def load_params_into(model: ResNet20Cosine, params: Mapping[str, np.ndarray]) -> ResNet20Cosine:
    """Copy exported parameters into ``model`` (in its dtype); num_batches_tracked keeps the model's own value."""
    check_params(params, model.fc.out_features)
    state = model.state_dict()
    dtype = model.fc.weight.dtype
    for name in param_shapes(model.fc.out_features):
        state[name] = torch.from_numpy(np.asarray(params[name])).to(dtype)
    model.load_state_dict(state, strict=True)
    return model


def build_resnet20(n_classes: int, mean: Sequence[float], std: Sequence[float],
                   params: Mapping[str, np.ndarray] | None = None, dtype: torch.dtype = torch.float32,
                   scale: float = COSINE_SCALE) -> ResNet20Cosine:
    """A CPU model in ``dtype`` (eval mode), with ``params`` loaded when given."""
    model = ResNet20Cosine(n_classes, mean, std, scale).to(dtype)
    if params is not None:
        load_params_into(model, params)
    return model.eval()


@torch.no_grad()
def eval_logits(model: ResNet20Cosine, x: np.ndarray, batch_size: int = EVAL_BATCH,
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


def layer_norms(params: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Frobenius norm of every convolution weight and of the head (the per-layer norms reported per student)."""
    return {name: float(np.linalg.norm(np.asarray(value, dtype=np.float64)))
            for name, value in params.items() if np.asarray(value).ndim >= 2}


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


def build_resnet20_spec(model_id: str, class_names: Sequence[str], mean: Sequence[float], std: Sequence[float],
                        params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Public ``model_spec.json`` of a ``resnet20_cosine`` student (schema ``SPEC_SCHEMA``), checked for forbidden
    terms (``weightcode.models.spec.check_public``). Its last layer is the cosine head block that evalkit.fscil reads."""
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
        if has_downsample(c_in, c_out, stride):
            shortcut = {"conv": _conv(f"{name}.downsample.0.weight", params, stride, 0),
                        "bn": _bn(f"{name}.downsample.1", params)}
        layers.append({
            "name": name,
            "op": "basic_block",
            "in_channels": c_in,
            "out_channels": c_out,
            "stride": stride,
            "conv1": _conv(f"{name}.conv1.weight", params, stride, 1),
            "bn1": _bn(f"{name}.bn1", params),
            "conv2": _conv(f"{name}.conv2.weight", params, 1, 1),
            "bn2": _bn(f"{name}.bn2", params),
            "shortcut": shortcut,
            "computes": "out = relu(bn2(conv2(relu(bn1(conv1(x))))) + s), s = bn(conv(x)) of the shortcut, or "
                        "s = x when the shortcut is null",
        })
    layers.append({"name": "pool", "op": "global_avg_pool", "computes": "features = mean of h over H and W, [N, 64]"})
    layers.append({
        "op": "cosine_linear",
        "weight": HEAD,
        "shape": _shape(params, HEAD),
        "scale": COSINE_SCALE,
        "eps": NORM_EPS,
        "computes": "logits = scale * (f / max(||f||_2, eps)) @ (W / max(||W_c||_2, eps)).T: the cosine between "
                    "the features and each row W_c, times scale; rows are L2-normalised separately",
    })
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
                            "running_var [channels]; head weight [n_classes, 64], no bias; all float32",
        "n_parameters": int(sum(np.asarray(value).size for value in params.values())),
        "tensors": {name: _shape(params, name) for name in param_shapes(len(names))},
        "layers": layers,
        "output": "logits [N, n_classes]; the prediction is the argmax over the classes in use",
        "reference_forward": "forward.py: forward(params, x_uint8 [N, 3, 32, 32]) -> float32 logits [N, n_classes]",
    }
    check_public(spec)
    return spec


# ---------------------------------------------------------------------------------------------------------------
# Training


@dataclass(frozen=True)
class FscilTrainConfig:
    """CEC's base-stage recipe (module docstring); ``RECIPE`` states the deviations."""

    arch: str = ARCH
    epochs: int = 100
    batch_size: int = 128
    optimizer: str = "SGD"
    lr: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    schedule: str = "MultiStepLR"
    milestones: tuple[int, ...] = (60, 70)
    gamma: float = 0.1
    loss: str = "cross_entropy_all_outputs"
    augment: str = "random_crop_pad4_hflip"
    augment_pad: int = 4
    temperature: float = COSINE_SCALE
    shuffle: bool = True
    drop_last: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FscilTrainResult:
    """The model after the last epoch (on the training device) with its per-epoch log."""

    model: ResNet20Cosine
    log: list[dict[str, float]] = field(default_factory=list)
    final: dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0


class _Indices(Dataset):
    """Yields 0..n-1; the DataLoader turns them into shuffled index batches."""

    def __init__(self, n: int) -> None:
        self._n = int(n)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, item: int) -> int:
        return int(item)


def train_resnet20(
    config: FscilTrainConfig,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    log_fn: Callable[[Mapping[str, Any]], None] | None = None,
) -> FscilTrainResult:
    """Train one student from scratch on raw uint8 images ``x_train`` [N, 3, 32, 32] and labels ``y_train``.

    The caller sets determinism first (``weightcode.models.train.configure_determinism``). Initialization draws from
    ``torch.manual_seed(seed)``, shuffling from a generator seeded with ``seed`` and augmentation from one seeded
    with ``seed + AUGMENT_SEED_OFFSET``. The whole training set lives on ``device``. ``final`` holds the clean
    (eval-mode, unaugmented) loss and accuracy on the training rows after the last epoch.
    """
    if config.optimizer != "SGD" or config.schedule != "MultiStepLR" or config.loss != "cross_entropy_all_outputs":
        raise ValueError(f"unsupported recipe {config.optimizer}/{config.schedule}/{config.loss}")
    if config.augment != "random_crop_pad4_hflip":
        raise ValueError(f"unknown augmentation {config.augment!r}")
    x_np, y_np = np.asarray(x_train), np.asarray(y_train)
    if x_np.dtype != np.uint8 or x_np.ndim != 4 or tuple(x_np.shape[1:]) != INPUT_SHAPE:
        raise TypeError(f"training images must be uint8 [N, {INPUT_SHAPE}], got {x_np.dtype}{x_np.shape}")
    if y_np.dtype != np.int64 or y_np.shape != (len(x_np),) or y_np.min() < 0 or y_np.max() >= n_classes:
        raise ValueError(f"labels must be int64 [N] in 0 .. {n_classes - 1}")
    torch.manual_seed(seed)
    model = ResNet20Cosine(n_classes, mean, std, config.temperature).to(device)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device)
    y = torch.from_numpy(y_np).to(device)
    loader = DataLoader(_Indices(len(x_np)), batch_size=config.batch_size, shuffle=config.shuffle,
                        drop_last=config.drop_last, generator=torch.Generator().manual_seed(seed), num_workers=0)
    augment_generator = torch.Generator().manual_seed(seed + AUGMENT_SEED_OFFSET)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, nesterov=config.nesterov,
                                weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(config.milestones),
                                                     gamma=config.gamma)
    result = FscilTrainResult(model=model)
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
        loss_sum, correct = 0.0, 0
        for start in range(0, len(x_np), EVAL_BATCH):
            logits = model(x[start:start + EVAL_BATCH])
            yb = y[start:start + EVAL_BATCH]
            loss_sum += float(F.cross_entropy(logits, yb, reduction="sum").to(torch.float64).item())
            correct += int((logits.argmax(dim=1) == yb).sum().item())
    result.final = {"train_loss_clean": loss_sum / len(x_np), "train_acc_clean": correct / len(x_np),
                    "n_train": int(len(x_np))}
    result.wall_time_s = time.perf_counter() - started
    return result


__all__ = [
    "ARCH", "BLOCKS_PER_STAGE", "BN_EPS", "BN_PARTS", "COSINE_SCALE", "FscilTrainConfig", "FscilTrainResult", "HEAD",
    "INPUT_SHAPE", "N_FEATURES", "NORM_EPS", "RECIPE", "ResNet20Cosine", "SPEC_SCHEMA", "STAGE_CHANNELS",
    "build_resnet20", "build_resnet20_spec", "check_params", "eval_logits", "export_params", "layer_norms",
    "load_params_into", "param_shapes", "train_resnet20",
]
