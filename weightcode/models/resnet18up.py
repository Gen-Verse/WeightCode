"""The ResNet-18 student of PREREG E12 (arch ``resnet18_up``): torchvision's ImageNet ResNet-18, fine-tuned on CIFAR-10.

Architecture: ``torchvision.models.resnet18`` (He et al. 2016, the ImageNet ResNet-18) with the same module and
parameter names, and two fixed operations in front of it:

* preprocessing: x / 255, then per channel (x - mean[c]) / std[c] with the ImageNet constants of torchvision's
  IMAGENET1K_V1 transforms (``IMAGENET_MEAN``, ``IMAGENET_STD``), in the model's dtype;
* bilinear upsampling of the normalised 32 x 32 image to R x R (``torch.nn.functional.interpolate``, mode bilinear,
  align_corners False, no antialias). R is a constant of the model (``ResNet18UpConfig.resize``), not a parameter.

Then torchvision's network: conv1 (7x7, stride 2, padding 3, no bias, 3 -> 64), bn1, ReLU, max pooling 3x3 with
stride 2 and padding 1; four stages ``layer1`` .. ``layer4`` of two BasicBlocks with 64, 128, 256 and 512 channels
(the first block of stages 2 to 4 has stride 2 and a projection shortcut ``downsample`` = conv 1x1 with the stride,
no bias, then BatchNorm); a block computes ``relu(bn2(conv2(relu(bn1(conv1(x))))) + shortcut(x))`` with 3x3
convolutions (padding 1, no bias); global average pooling (a mean over H and W, the value of torchvision's
AdaptiveAvgPool2d(1)) and a new linear head ``fc`` 512 -> 10 with bias. BatchNorm eps 1e-5. 102 tensors (BatchNorm
running statistics included, num_batches_tracked excluded), 11,191,242 entries, of which 9,600 are running statistics.

Start (``initial_student``): every tensor but ``fc`` from the IMAGENET1K_V1 weights file (``resnet18-f37072fd.pth``,
sha256 ``WEIGHTS_SHA256``), and ``fc`` initialised by ``torch.manual_seed(seed)`` followed by nn.Linear's default
``reset_parameters`` (kaiming_uniform with a = sqrt(5) on the weight, uniform in +-1/sqrt(512) on the bias).

Training (``train_resnet18up``), PREREG E12's recipe: SGD momentum 0.9 with Nesterov, weight decay 5e-4 on every
parameter, batch 128 shuffled without drop_last (313 steps per epoch on 40,000 rows), 15 epochs, the learning rate
set per step: a linear warmup over the first epoch, lr_t = 0.01 (t + 1) / 313 for t < 313, then a cosine to 0,
lr_t = 0.01 x 0.5 (1 + cos(pi (t - 313) / (4695 - 313))); cross-entropy; BatchNorm in train mode (momentum 0.1, as
torch's default); random crop 32 after zero padding by 4 and horizontal flip (``weightcode.models.train.augment_crop_flip``)
on the raw 32 x 32 image, before the normalisation and the upsampling; fp32 with deterministic algorithms and IEEE
fp32 matmuls and convolutions. The model after the last epoch is kept.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from weightcode.models.resnet20 import BN_EPS, BN_PARTS, INPUT_SHAPE, Preprocess, _Indices
from weightcode.models.train import AUGMENT_SEED_OFFSET, augment_crop_flip

ARCH = "resnet18_up"
SPEC_SCHEMA = "weightcode.model_spec.resnet18_up.v1"
HEAD_WEIGHT = "fc.weight"
HEAD_BIAS = "fc.bias"
N_FEATURES = 512
STEM_CHANNELS = 64
STAGE_CHANNELS: tuple[int, ...] = (64, 128, 256, 512)
STAGE_STRIDES: tuple[int, ...] = (1, 2, 2, 2)
BLOCKS_PER_STAGE = 2
RESIZES: tuple[int, ...] = (128, 96, 64)
"""PREREG E12's upsampled sizes R in order of preference; the first one the forward.py timing allows is used."""
EVAL_BATCH = 256
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
WEIGHTS_FILE = "resnet18-f37072fd.pth"
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
WEIGHTS_NAME = "torchvision ResNet18_Weights.IMAGENET1K_V1"

RECIPE = (
    "PREREG E12 (main c547e0a): torchvision resnet18 with the IMAGENET1K_V1 weights (resnet18-f37072fd.pth), fc "
    "replaced by a new 512 x 10 layer (torch.manual_seed(seed), nn.Linear's default init), fine-tuned on E1's 40,000 "
    "cifar10 train rows: SGD momentum 0.9 Nesterov, weight decay 5e-4 on every parameter, batch 128 shuffled, 15 "
    "epochs, lr 0.01 per step with a linear warmup over the first epoch (0.01 (t + 1) / 313) and a cosine to 0 over "
    "the other 14, cross-entropy, BatchNorm in train mode, random crop 32 with padding 4 and horizontal flip on the "
    "raw image, then x / 255, ImageNet normalisation and bilinear upsampling to R x R (align_corners False) inside "
    "the model; fp32, deterministic algorithms, IEEE fp32 matmuls and convolutions; the model after the last epoch is "
    "kept."
)


@dataclass(frozen=True)
class ResNet18UpConfig:
    """The fixed operations of the student: the class count and the upsampled size R."""

    n_classes: int = 10
    resize: int = 128

    def __post_init__(self) -> None:
        if self.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if self.resize not in RESIZES:
            raise ValueError(f"resize must be one of PREREG E12's sizes {RESIZES}, got {self.resize}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def block_names() -> list[tuple[str, int, int, int]]:
    """(block name, in channels, out channels, stride) of the eight BasicBlocks in forward order."""
    blocks = []
    in_channels = STEM_CHANNELS
    for stage, (channels, stride) in enumerate(zip(STAGE_CHANNELS, STAGE_STRIDES), start=1):
        for index in range(BLOCKS_PER_STAGE):
            blocks.append((f"layer{stage}.{index}", in_channels, channels, stride if index == 0 else 1))
            in_channels = channels
    return blocks


def has_downsample(in_channels: int, out_channels: int, stride: int) -> bool:
    """torchvision's rule: a projection shortcut where the stride or the width changes."""
    return stride != 1 or in_channels != out_channels


def conv_names() -> list[str]:
    """The 20 convolution weights in forward order (stem, then per block conv1, conv2 and the projection)."""
    names = ["conv1.weight"]
    for name, c_in, c_out, stride in block_names():
        names += [f"{name}.conv1.weight", f"{name}.conv2.weight"]
        if has_downsample(c_in, c_out, stride):
            names.append(f"{name}.downsample.0.weight")
    return names


def bn_prefixes() -> list[str]:
    """The 20 BatchNorm layer prefixes in forward order."""
    prefixes = ["bn1"]
    for name, c_in, c_out, stride in block_names():
        prefixes += [f"{name}.bn1", f"{name}.bn2"]
        if has_downsample(c_in, c_out, stride):
            prefixes.append(f"{name}.downsample.1")
    return prefixes


def param_shapes(n_classes: int) -> dict[str, tuple[int, ...]]:
    """Every parameter name and shape of the student, in torchvision's state-dict order (the model.npz layout)."""
    shapes: dict[str, tuple[int, ...]] = {"conv1.weight": (STEM_CHANNELS, INPUT_SHAPE[0], 7, 7)}
    shapes.update({f"bn1.{part}": (STEM_CHANNELS,) for part in BN_PARTS})
    for name, c_in, c_out, stride in block_names():
        shapes[f"{name}.conv1.weight"] = (c_out, c_in, 3, 3)
        shapes.update({f"{name}.bn1.{part}": (c_out,) for part in BN_PARTS})
        shapes[f"{name}.conv2.weight"] = (c_out, c_out, 3, 3)
        shapes.update({f"{name}.bn2.{part}": (c_out,) for part in BN_PARTS})
        if has_downsample(c_in, c_out, stride):
            shapes[f"{name}.downsample.0.weight"] = (c_out, c_in, 1, 1)
            shapes.update({f"{name}.downsample.1.{part}": (c_out,) for part in BN_PARTS})
    shapes[HEAD_WEIGHT] = (n_classes, N_FEATURES)
    shapes[HEAD_BIAS] = (n_classes,)
    return shapes


class BasicBlock(nn.Module):
    """torchvision's BasicBlock: conv3x3 (stride) - BN - ReLU - conv3x3 - BN, plus the shortcut, then ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.downsample: nn.Module | None = None
        if has_downsample(in_channels, out_channels, stride):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=BN_EPS),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(out + identity)


class ResNet18Up(nn.Module):
    """torchvision's ResNet-18 behind x / 255, ImageNet normalisation and bilinear upsampling to R x R."""

    def __init__(self, config: ResNet18UpConfig, mean: Sequence[float] = IMAGENET_MEAN,
                 std: Sequence[float] = IMAGENET_STD) -> None:
        super().__init__()
        self.config = config
        self.preprocess = Preprocess(mean, std)
        self.conv1 = nn.Conv2d(INPUT_SHAPE[0], STEM_CHANNELS, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(STEM_CHANNELS, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stages: dict[int, list[BasicBlock]] = {1: [], 2: [], 3: [], 4: []}
        for name, c_in, c_out, stride in block_names():
            stages[int(name[len("layer")])].append(BasicBlock(c_in, c_out, stride))
        self.layer1 = nn.Sequential(*stages[1])
        self.layer2 = nn.Sequential(*stages[2])
        self.layer3 = nn.Sequential(*stages[3])
        self.layer4 = nn.Sequential(*stages[4])
        self.fc = nn.Linear(N_FEATURES, config.n_classes)

    def upsampled(self, x: torch.Tensor) -> torch.Tensor:
        """The normalised images upsampled to R x R (the network's input)."""
        h = self.preprocess(x, self.conv1.weight.dtype)
        return F.interpolate(h, size=(self.config.resize, self.config.resize), mode="bilinear", align_corners=False,
                             antialias=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled [N, 512] features of raw uint8 images."""
        h = self.maxpool(self.relu(self.bn1(self.conv1(self.upsampled(x)))))
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        return h.mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_imagenet_state(weights_path: Path) -> dict[str, torch.Tensor]:
    """The IMAGENET1K_V1 state dict from ``weights_path``, refused unless its sha256 is ``WEIGHTS_SHA256``."""
    path = Path(weights_path)
    digest = file_sha256(path)
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"{path}: sha256 {digest}, expected {WEIGHTS_SHA256} ({WEIGHTS_NAME})")
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = param_shapes(1000)
    names = {key for key in state if not key.endswith("num_batches_tracked")}
    if names != set(expected):
        raise ValueError(f"{path}: tensor names differ from torchvision resnet18: {sorted(names ^ set(expected))}")
    for name, shape in expected.items():
        if tuple(state[name].shape) != shape:
            raise ValueError(f"{path}: {name} has shape {tuple(state[name].shape)}, expected {shape}")
    return state


def initial_student(config: ResNet18UpConfig, weights_path: Path, seed: int,
                    mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD) -> ResNet18Up:
    """The fine-tuning start: every tensor but fc from the ImageNet weights, fc from ``torch.manual_seed(seed)``.

    num_batches_tracked keeps the module's 0 when the file has none (it does not enter the computation: BatchNorm
    momentum is 0.1)."""
    state = load_imagenet_state(weights_path)
    model = ResNet18Up(config, mean, std)
    own = model.state_dict()
    for key in own:
        if key.startswith("fc.") or (key.endswith("num_batches_tracked") and key not in state):
            continue
        own[key] = state[key].to(own[key].dtype).clone()
    model.load_state_dict(own, strict=True)
    torch.manual_seed(int(seed))
    model.fc.reset_parameters()
    return model


def export_params(model: ResNet18Up) -> dict[str, np.ndarray]:
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


def load_params_into(model: ResNet18Up, params: Mapping[str, np.ndarray]) -> ResNet18Up:
    """Copy exported parameters into ``model`` (in its dtype); num_batches_tracked keeps the model's own value."""
    check_params(params, model.fc.out_features)
    state = model.state_dict()
    dtype = model.fc.weight.dtype
    for name in param_shapes(model.fc.out_features):
        state[name] = torch.from_numpy(np.asarray(params[name])).to(dtype)
    model.load_state_dict(state, strict=True)
    return model


def build_resnet18up(config: ResNet18UpConfig, params: Mapping[str, np.ndarray] | None = None,
                     dtype: torch.dtype = torch.float32, mean: Sequence[float] = IMAGENET_MEAN,
                     std: Sequence[float] = IMAGENET_STD) -> ResNet18Up:
    """A CPU model in ``dtype`` (eval mode), with ``params`` loaded when given."""
    model = ResNet18Up(config, mean, std).to(dtype)
    if params is not None:
        load_params_into(model, params)
    return model.eval()


@torch.no_grad()
def eval_logits(model: ResNet18Up, x: np.ndarray, batch_size: int = EVAL_BATCH,
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
UPSAMPLE_COMPUTES = (
    "bilinear interpolation as torch.nn.functional.interpolate(mode='bilinear', align_corners=False, "
    "antialias=False): along each axis, output position i reads the input at s = (i + 0.5) * 32 / R - 0.5, clamped "
    "below at 0, from rows floor(s) and min(floor(s) + 1, 31) with weights 1 - (s - floor(s)) and s - floor(s)")


def build_resnet18up_spec(model_id: str, class_names: Sequence[str], config: ResNet18UpConfig,
                          params: Mapping[str, np.ndarray], mean: Sequence[float] = IMAGENET_MEAN,
                          std: Sequence[float] = IMAGENET_STD) -> dict[str, Any]:
    """Public ``model_spec.json`` of a ``resnet18_up`` student (schema ``SPEC_SCHEMA``), checked for forbidden terms
    (``weightcode.models.spec.check_public``)."""
    from weightcode.models.spec import check_public

    names = [str(name) for name in class_names]
    if len(names) != config.n_classes:
        raise ValueError(f"{len(names)} class names for {config.n_classes} classes")
    check_params(params, len(names))
    size = int(config.resize)
    layers: list[dict[str, Any]] = [
        {"name": "upsample", "op": "resize_bilinear", "size": [size, size], "computes": UPSAMPLE_COMPUTES,
         "parameters": None},
        {"name": "conv1", "op": "conv2d", **_conv("conv1.weight", params, 2, 3), "computes": CONV_COMPUTES},
        {"name": "bn1", "op": "batchnorm_eval", **_bn("bn1", params), "computes": BN_COMPUTES},
        {"name": "relu", "op": "relu"},
        {"name": "maxpool", "op": "maxpool2d", "kernel": 3, "stride": 2, "padding": 1,
         "computes": "max over 3x3 windows with stride 2; padded positions never win (padding with -inf)"},
    ]
    for name, c_in, c_out, stride in block_names():
        shortcut = None
        if has_downsample(c_in, c_out, stride):
            shortcut = {"conv": _conv(f"{name}.downsample.0.weight", params, stride, 0),
                        "bn": _bn(f"{name}.downsample.1", params),
                        "computes": "s = bn(conv1x1(x)) with the block's stride"}
        layers.append({
            "name": name, "op": "basic_block", "in_channels": c_in, "out_channels": c_out, "stride": stride,
            "conv1": _conv(f"{name}.conv1.weight", params, stride, 1), "bn1": _bn(f"{name}.bn1", params),
            "conv2": _conv(f"{name}.conv2.weight", params, 1, 1), "bn2": _bn(f"{name}.bn2", params),
            "shortcut": shortcut,
            "computes": "out = relu(bn2(conv2(relu(bn1(conv1(x))))) + s), s = the shortcut of x, or s = x when the "
                        "shortcut is null",
        })
    layers.append({"name": "pool", "op": "global_avg_pool",
                   "computes": f"features = mean of h over H and W, [N, {N_FEATURES}]"})
    layers.append({"name": "fc", "op": "linear", "weight": HEAD_WEIGHT, "bias": HEAD_BIAS,
                   "shape": _shape(params, HEAD_WEIGHT), "computes": "logits = f @ weight.T + bias"})
    spec = {
        "schema": SPEC_SCHEMA,
        "model_id": model_id,
        "arch": ARCH,
        "input": {"kind": "images", "shape": list(INPUT_SHAPE), "dtype": "uint8", "layout": "NCHW"},
        "input_shape": list(INPUT_SHAPE),
        "preprocessing": {"mean": [float(v) for v in mean], "std": [float(v) for v in std], "resize": [size, size],
                          "computes": "x / 255, then per channel (x - mean[c]) / std[c], then bilinear upsampling "
                                      f"to {size} x {size} (the upsample layer)"},
        "classes": names,
        "n_classes": len(names),
        "parameter_layout": "PyTorch: conv weight [out, in, kh, kw]; BatchNorm weight, bias, running_mean, "
                            f"running_var [channels]; head weight [n_classes, {N_FEATURES}] and bias [n_classes]; "
                            "all float32",
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
class E12TrainConfig:
    """PREREG E12's fine-tuning recipe (module docstring)."""

    arch: str = ARCH
    epochs: int = 15
    batch_size: int = 128
    optimizer: str = "SGD"
    lr: float = 0.01
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    weight_decay_scope: str = "every parameter"
    schedule: str = "per-step linear warmup over the first epoch, then cosine to 0"
    warmup_epochs: int = 1
    loss: str = "cross_entropy"
    batchnorm: str = "train mode (momentum 0.1)"
    augment: str = "random_crop_pad4_hflip"
    augment_pad: int = 4
    augment_stage: str = "on the raw 32 x 32 image, before the normalisation and the upsampling"
    shuffle: bool = True
    drop_last: bool = False
    precision: str = "fp32, deterministic algorithms, IEEE fp32 matmuls and convolutions"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def step_lr(config: E12TrainConfig, step: int, steps_per_epoch: int) -> float:
    """The learning rate of global step ``step`` (0-based): linear warmup over ``warmup_epochs``, then cosine to 0."""
    total = config.epochs * steps_per_epoch
    warmup = config.warmup_epochs * steps_per_epoch
    if not 0 <= step < total:
        raise ValueError(f"step {step} outside [0, {total})")
    if step < warmup:
        return config.lr * (step + 1) / warmup
    return config.lr * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / (total - warmup)))


@dataclass
class E12TrainResult:
    """The model after the last epoch (on the training device) with its per-epoch log."""

    model: ResNet18Up
    log: list[dict[str, float]] = field(default_factory=list)
    final: dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0


def train_resnet18up(
    config: E12TrainConfig,
    model_config: ResNet18UpConfig,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights_path: Path,
    device: torch.device,
    log_fn: Callable[[Mapping[str, Any]], None] | None = None,
) -> E12TrainResult:
    """Fine-tune one student from the ImageNet weights on raw uint8 images ``x_train`` [N, 3, 32, 32].

    The caller sets determinism first (``weightcode.models.train.configure_determinism``). ``initial_student`` seeds the
    head with ``seed``; shuffling draws from a generator seeded with ``seed`` and augmentation from one seeded with
    ``seed + AUGMENT_SEED_OFFSET``, as for E10's students. The whole training set lives on ``device``. ``final`` holds
    the clean (eval-mode, unaugmented) loss and accuracy on the training rows after the last epoch.
    """
    if config.optimizer != "SGD" or config.loss != "cross_entropy" or config.augment != "random_crop_pad4_hflip":
        raise ValueError(f"unsupported recipe {config.optimizer}/{config.loss}/{config.augment}")
    x_np, y_np = np.asarray(x_train), np.asarray(y_train)
    if x_np.dtype != np.uint8 or x_np.ndim != 4 or tuple(x_np.shape[1:]) != INPUT_SHAPE:
        raise TypeError(f"training images must be uint8 [N, {INPUT_SHAPE}], got {x_np.dtype}{x_np.shape}")
    n_classes = model_config.n_classes
    if y_np.dtype != np.int64 or y_np.shape != (len(x_np),) or y_np.min() < 0 or y_np.max() >= n_classes:
        raise ValueError(f"labels must be int64 [N] in 0 .. {n_classes - 1}")
    model = initial_student(model_config, weights_path, seed).to(device)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device)
    y = torch.from_numpy(y_np).to(device)
    loader = DataLoader(_Indices(len(x_np)), batch_size=config.batch_size, shuffle=config.shuffle,
                        drop_last=config.drop_last, generator=torch.Generator().manual_seed(seed), num_workers=0)
    steps_per_epoch = len(loader)
    augment_generator = torch.Generator().manual_seed(seed + AUGMENT_SEED_OFFSET)
    optimizer = torch.optim.SGD(model.parameters(), lr=step_lr(config, 0, steps_per_epoch),
                                momentum=config.momentum, nesterov=config.nesterov, weight_decay=config.weight_decay)
    result = E12TrainResult(model=model)
    started = time.perf_counter()
    step = 0
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        correct = torch.zeros((), dtype=torch.int64, device=device)
        seen = 0
        lr_first = step_lr(config, step, steps_per_epoch)
        for index in loader:
            lr = step_lr(config, step, steps_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
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
            step += 1
        train_loss = float(loss_sum.item()) / seen
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        entry = {"epoch": epoch, "lr_first_step": lr_first, "lr_last_step": lr, "train_loss": train_loss,
                 "train_acc_augmented": float(correct.item()) / seen, "epoch_time_s": time.perf_counter() - epoch_start}
        result.log.append(entry)
        if log_fn is not None:
            log_fn(entry)
    if step != config.epochs * steps_per_epoch:
        raise RuntimeError(f"ran {step} steps, expected {config.epochs * steps_per_epoch}")
    model.eval()
    with torch.no_grad():
        loss_sum_f, correct_i = 0.0, 0
        for start in range(0, len(x_np), EVAL_BATCH):
            logits = model(x[start:start + EVAL_BATCH])
            yb = y[start:start + EVAL_BATCH]
            loss_sum_f += float(F.cross_entropy(logits, yb, reduction="sum").to(torch.float64).item())
            correct_i += int((logits.argmax(dim=1) == yb).sum().item())
    result.final = {"train_loss_clean": loss_sum_f / len(x_np), "train_acc_clean": correct_i / len(x_np),
                    "n_train": int(len(x_np)), "steps": step, "steps_per_epoch": steps_per_epoch}
    result.wall_time_s = time.perf_counter() - started
    return result


def torchvision_reference_diff(model: ResNet18Up, x: np.ndarray) -> dict[str, Any]:
    """Max |logit| difference between ``model`` and ``torchvision.models.resnet18`` holding the same tensors, on the
    same upsampled inputs, in eval mode (a check that the module is torchvision's network); None fields when
    torchvision is not installed."""
    try:
        import torchvision
    except ImportError:
        return {"torchvision": None, "max_abs_diff": None}
    reference = torchvision.models.resnet18(weights=None, num_classes=model.fc.out_features)
    reference.load_state_dict(model.state_dict() | {}, strict=False)
    missing = set(reference.state_dict()) - set(model.state_dict())
    if missing:
        raise ValueError(f"torchvision resnet18 has tensors the student lacks: {sorted(missing)}")
    device = next(model.parameters()).device
    dtype = model.conv1.weight.dtype
    reference = reference.to(device=device, dtype=dtype).eval()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        images = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        ours = model(images)
        theirs = reference(model.upsampled(images))
    model.train(was_training)
    return {"torchvision": torchvision.__version__, "rows": int(len(x)),
            "max_abs_diff": float((ours - theirs).abs().max().item())}


__all__ = [
    "ARCH", "BasicBlock", "E12TrainConfig", "E12TrainResult", "HEAD_BIAS", "HEAD_WEIGHT", "IMAGENET_MEAN",
    "IMAGENET_STD", "N_FEATURES", "RECIPE", "RESIZES", "ResNet18Up", "ResNet18UpConfig", "SPEC_SCHEMA",
    "WEIGHTS_FILE", "WEIGHTS_SHA256", "WEIGHTS_URL", "block_names", "bn_prefixes", "build_resnet18up",
    "build_resnet18up_spec", "check_params", "conv_names", "eval_logits", "export_params", "has_downsample",
    "initial_student", "load_imagenet_state", "load_params_into", "param_shapes", "step_lr",
    "torchvision_reference_diff", "train_resnet18up",
]
