"""Deterministic training of the base, ceiling and dev models (brief item 4).

Fixed protocol: Adam (lr 1e-3, default betas and eps, no weight decay), batch 128, cross-entropy, no early
stopping and no checkpoint selection: the model after the last epoch is the model. V_select accuracy is logged
per epoch for monitoring only and never feeds back into training.

Randomness: ``torch.manual_seed(seed)`` (initialization), a DataLoader generator seeded with ``seed`` (shuffling)
and, for augmentation, a CPU generator seeded with ``seed + AUGMENT_SEED_OFFSET`` whose draws are moved to the
device. cuDNN runs in deterministic mode, ``torch.use_deterministic_algorithms(True)`` is on and TF32 is off for
matmul and convolution, so GPU logits are true float32.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_MAIN_SEEDS: tuple[int, ...] = (0, 1)
"""Main seeds trained during development (user rule 2026-11-03); more are added with ``--seeds``."""
CONFIRMATION_SEEDS: tuple[int, ...] = (2, 3, 4)
"""Main seeds kept for later confirmation; train them only when asked (``train_base.py --group main --seeds 2 3 4``)."""
DEV_SEEDS: tuple[int, ...] = (100, 101)
"""Development-model seeds (role "dev"), never pooled with main seeds."""
AUGMENT_SEED_OFFSET = 1_000_003
CUBLAS_WORKSPACE_CONFIGS: tuple[str, ...] = (":4096:8", ":16:8")
EVAL_BATCH = 1024


@dataclass(frozen=True)
class TrainConfig:
    """Training protocol of one dataset."""

    arch: str
    epochs: int
    batch_size: int = 128
    optimizer: str = "Adam"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    loss: str = "cross_entropy"
    augment: str | None = None
    augment_pad: int = 0
    shuffle: bool = True
    drop_last: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


TRAIN_CONFIGS: Mapping[str, TrainConfig] = MappingProxyType(
    {
        "fashion_mnist": TrainConfig(arch="mlp", epochs=15),
        "colored_mnist": TrainConfig(arch="mlp", epochs=10),
        "cifar10": TrainConfig(arch="cnn", epochs=30, augment="random_crop_pad4_hflip", augment_pad=4),
    }
)


def configure_determinism(seed: int) -> dict[str, Any]:
    """Seed python, numpy and torch, and switch every determinism and precision flag on. Returns the settings."""
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in CUBLAS_WORKSPACE_CONFIGS:
        raise RuntimeError(
            "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before torch initializes CUDA (deterministic cuBLAS), "
            f"got {workspace!r}"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    return {
        "python_random_seed": seed,
        "numpy_global_seed": seed,
        "torch_manual_seed": seed,
        "loader_generator_seed": seed,
        "augment_generator_seed": seed + AUGMENT_SEED_OFFSET,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "use_deterministic_algorithms": True,
        "cuda_matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
        "cudnn_conv_fp32_precision": torch.backends.cudnn.conv.fp32_precision,
        "cublas_workspace_config": workspace,
    }


class _IndexDataset(Dataset):
    """Yields the integers 0..n-1; the DataLoader turns them into shuffled index batches."""

    def __init__(self, n: int) -> None:
        self._n = int(n)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, item: int) -> int:
        return int(item)


def augment_crop_flip(x: torch.Tensor, pad: int, generator: torch.Generator) -> torch.Tensor:
    """Random crop after zero padding by ``pad`` pixels, then horizontal flip with probability 0.5.

    Operates on raw images (zero padding is black, as torchvision RandomCrop with padding does). Offsets and flips
    are drawn on the CPU from ``generator`` and then moved to the images' device.
    """
    n, _, height, width = x.shape
    offsets = torch.randint(0, 2 * pad + 1, (n, 2), generator=generator).to(x.device)
    flips = (torch.rand(n, generator=generator) < 0.5).to(x.device)
    padded = F.pad(x, (pad, pad, pad, pad))
    rows = offsets[:, 0:1] + torch.arange(height, device=x.device)
    cols = offsets[:, 1:2] + torch.arange(width, device=x.device)
    cols = torch.where(flips[:, None], cols.flip(1), cols)
    batch = torch.arange(n, device=x.device)[:, None, None, None]
    channels = torch.arange(x.shape[1], device=x.device)[None, :, None, None]
    return padded[batch, channels, rows[:, None, :, None], cols[:, None, None, :]]


@torch.no_grad()
def predict_torch(model: nn.Module, x: torch.Tensor, batch_size: int = EVAL_BATCH) -> torch.Tensor:
    """Eval-mode logits of ``model`` on ``x`` (raw images on the model's device), batched."""
    was_training = model.training
    model.eval()
    outputs = [model(x[start : start + batch_size]) for start in range(0, x.shape[0], batch_size)]
    model.train(was_training)
    return torch.cat(outputs, dim=0)


@dataclass
class TrainResult:
    """Trained model (on the training device) with its per-epoch monitoring log."""

    model: nn.Module
    epoch_log: list[dict[str, float]] = field(default_factory=list)
    wall_time_s: float = 0.0


def train_model(
    model: nn.Module,
    config: TrainConfig,
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_monitor: torch.Tensor,
    y_monitor: torch.Tensor,
    log_fn: Any = None,
) -> TrainResult:
    """Train ``model`` (already on the device of the data tensors) for ``config.epochs`` epochs.

    ``x_train`` holds raw uint8 images and ``y_train`` int64 labels, both on the training device. The monitor
    split (V_select) is evaluated after every epoch and only logged.
    """
    if config.optimizer != "Adam" or config.loss != "cross_entropy":
        raise ValueError(f"unsupported optimizer/loss {config.optimizer}/{config.loss}")
    if x_train.dtype != torch.uint8:
        raise TypeError("training images must be uint8")
    loader = DataLoader(
        _IndexDataset(x_train.shape[0]),
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    augment_generator = torch.Generator().manual_seed(seed + AUGMENT_SEED_OFFSET)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay
    )
    result = TrainResult(model=model)
    start_all = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        model.train()
        loss_sum = torch.zeros((), dtype=torch.float64, device=x_train.device)
        seen = 0
        for index in loader:
            index = index.to(x_train.device)
            xb = x_train[index]
            yb = y_train[index]
            if config.augment == "random_crop_pad4_hflip":
                xb = augment_crop_flip(xb, config.augment_pad, augment_generator)
            elif config.augment is not None:
                raise ValueError(f"unknown augmentation {config.augment!r}")
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().to(torch.float64) * yb.shape[0]
            seen += int(yb.shape[0])
        train_loss = float(loss_sum.item()) / seen
        monitor_pred = predict_torch(model, x_monitor).argmax(dim=1)
        monitor_acc = float((monitor_pred == y_monitor).to(torch.float64).mean().item())
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "vselect_acc_monitor": monitor_acc,
            "epoch_time_s": time.perf_counter() - start,
        }
        result.epoch_log.append(entry)
        if log_fn is not None:
            log_fn(entry)
    result.wall_time_s = time.perf_counter() - start_all
    model.eval()
    return result
