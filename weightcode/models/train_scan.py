"""Training of the E13 SCAN students (arch ``tfm``, ``weightcode.models.transformer``; PREREG E13).

Fixed recipe (written in the as-built text before any training; no early stopping, no checkpoint selection: the model
after the last epoch is the model). It is E5's seqops recipe (``weightcode.models.train_seq``) on a fixed training file:

* data: the condition's stored ``train`` rows (the official training file without the 3000 held-out old commands,
  duplicates and the primitive's rows kept), every epoch in the order of ``default_rng([SPLIT_SEED, TRAIN_STREAM,
  seed, epoch]).permutation(n)``, cut into batches of ``batch_size`` (the last batch of an epoch may be smaller);
* loss: cross-entropy over the OUT positions (OUT_SLOTS per row, END included; IGNORE elsewhere), mean over the
  labelled positions of the batch;
* optimizer: AdamW, betas (0.9, 0.98), eps 1e-8; weight decay only on 2-D linear weights
  (``weightcode.models.transformer.decay_groups``);
* schedule: linear warmup to ``lr`` over ``warmup_steps``, then cosine decay to ``min_lr_ratio * lr`` at the last
  step; gradient norm clipped at ``grad_clip``;
* randomness: ``torch.manual_seed(seed)`` (initialization) through ``weightcode.models.train.configure_determinism``; the
  batch order is a function of (seed, epoch) alone.

No new-part row reaches training or its monitor: the job gets the train and old_select splits only. The monitor reads
exact match on MONITOR_N fixed training rows; nothing depends on it. The gate (PREREG E13) is exact match on the 1000
old V_select rows: at least GATE_EM for a base. The size steps are SIZE_STEPS: d 128, 4 layers, MLP 512, then, only if
it misses the gate, d 256, 6 layers, MLP 1024 (4 heads in both; forward.py is rendered with 4).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.data import scan as S
from weightcode.data.registry import SPLIT_SEED
from weightcode.models import transformer as T

TRAIN_STREAM = 1310
MONITOR_STREAM = 1311
MONITOR_N = 1024
GATE_EM = 0.95
"""PREREG E13: a student is a base only if its exact match on the 1000 old V_select rows is at least this."""
HEADROOM_EM_MAX = 0.50
"""PREREG E13: a condition runs only if the base's exact match on new V_select is at most this."""
EVAL_BATCH = 1024


@dataclass(frozen=True)
class ScanTrainConfig:
    """Recipe of one E13 student."""

    arch: str = "tfm"
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 512
    batch_size: int = 256
    epochs: int = 800
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    log_every_epochs: int = 50

    def model_config(self) -> T.TfmConfig:
        return T.TfmConfig(S.VOCAB_SIZE, S.SEQ_LEN, S.N_CLASSES, S.PAD, self.d_model, self.n_layers, self.n_heads,
                           self.d_mlp)

    def steps_per_epoch(self, n_rows: int) -> int:
        return math.ceil(int(n_rows) / self.batch_size)

    def total_steps(self, n_rows: int) -> int:
        return self.epochs * self.steps_per_epoch(n_rows)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


SIZE_STEPS: tuple[ScanTrainConfig, ...] = (ScanTrainConfig(), ScanTrainConfig(d_model=256, n_layers=6, d_mlp=1024))
"""The pre-registered sizes (PREREG E13): d 128, 4 layers, MLP 512; then d 256, 6 layers, MLP 1024."""


def lr_at(config: ScanTrainConfig, step: int, total_steps: int) -> float:
    """Learning rate of optimizer step ``step`` (0-based) of ``total_steps``."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.lr * (config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine)


def epoch_order(seed: int, epoch: int, n_rows: int) -> np.ndarray:
    """The row order of one epoch: ``default_rng([SPLIT_SEED, TRAIN_STREAM, seed, epoch]).permutation(n_rows)``."""
    return np.random.default_rng([SPLIT_SEED, TRAIN_STREAM, int(seed), int(epoch)]).permutation(int(n_rows))


def monitor_rows(n_rows: int) -> np.ndarray:
    """The fixed training rows the monitor reads (sorted)."""
    size = min(MONITOR_N, int(n_rows))
    return np.sort(np.random.default_rng([SPLIT_SEED, MONITOR_STREAM]).choice(int(n_rows), size=size, replace=False))


def torch_predict(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Per-position argmax of a torch student, in batches."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), EVAL_BATCH):
            logits = model(torch.from_numpy(np.ascontiguousarray(x[start:start + EVAL_BATCH])).to(device))
            outputs.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(outputs).astype(np.int64)


def em_summary(arrays: Mapping[str, np.ndarray], pred: np.ndarray) -> dict[str, Any]:
    """Exact match, token accuracy and exact match by number of actions and by construction."""
    right = S.row_correct(arrays["y"], pred)
    commands = S.decode(arrays["x"])
    by_construction = {}
    for name, words in S.CONSTRUCTIONS.items():
        rows = np.array([any(word in command.split() for word in words) for command in commands])
        by_construction[name] = {"n": int(rows.sum()), "em": float(right[rows].mean()) if rows.any() else None}
    lengths = np.asarray(arrays["n_act"])
    bins = {"1-8": (1, 8), "9-16": (9, 16), "17-24": (17, 24), "25-48": (25, 48)}
    by_actions = {}
    for name, (lo, hi) in bins.items():
        rows = (lengths >= lo) & (lengths <= hi)
        by_actions[name] = {"n": int(rows.sum()), "em": float(right[rows].mean()) if rows.any() else None}
    return {"n": int(len(right)), "em": float(right.mean()), "token_acc": S.token_accuracy(arrays["y"], pred),
            "by_construction": by_construction, "by_n_actions": by_actions}


@dataclass
class ScanTrainResult:
    model: T.SeqTransformer
    log: list[dict[str, Any]]
    wall_time_s: float
    steps: int
    rows_seen: int


def train_scan(
    config: ScanTrainConfig,
    seed: int,
    train: Mapping[str, np.ndarray],
    device: torch.device,
    monitor: Callable[[T.SeqTransformer], dict[str, Any]] | None = None,
) -> ScanTrainResult:
    """Train one student from scratch under the fixed recipe (module docstring) on the stored training rows."""
    started = time.perf_counter()
    x_all = torch.from_numpy(np.ascontiguousarray(train["x"], dtype=np.int64)).to(device)
    y_all = torch.from_numpy(np.ascontiguousarray(train["y"], dtype=np.int64)).to(device)
    n_rows = int(x_all.shape[0])
    if np.isin(np.asarray(train["part"]), [S.PART_NEW]).any():
        raise ValueError("a training row belongs to the new part")
    total = config.total_steps(n_rows)
    model = T.build_seq_model(config.model_config()).to(device)
    optimizer = torch.optim.AdamW(
        T.decay_groups(model, config.weight_decay), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps
    )
    log: list[dict[str, Any]] = []
    step = 0
    rows_seen = 0
    for epoch in range(config.epochs):
        order = torch.from_numpy(epoch_order(seed, epoch, n_rows)).to(device)
        running = torch.zeros((), dtype=torch.float64, device=device)
        model.train()
        for start in range(0, n_rows, config.batch_size):
            rows = order[start:start + config.batch_size]
            for group in optimizer.param_groups:
                group["lr"] = lr_at(config, step, total)
            logits = model(x_all[rows])
            loss = F.cross_entropy(logits.reshape(-1, S.N_CLASSES), y_all[rows].reshape(-1), ignore_index=S.IGNORE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running += loss.detach().double()
            step += 1
            rows_seen += int(rows.shape[0])
        if (epoch + 1) % config.log_every_epochs == 0 or epoch + 1 == config.epochs:
            entry: dict[str, Any] = {
                "epoch": epoch + 1,
                "step": step,
                "lr": lr_at(config, step - 1, total),
                "mean_loss_last_epoch": float(running) / config.steps_per_epoch(n_rows),
                "grad_norm": float(grad_norm),
                "elapsed_s": time.perf_counter() - started,
            }
            if monitor is not None:
                entry.update(monitor(model))
                model.train()
            log.append(entry)
    if step != total:
        raise AssertionError(f"ran {step} steps, expected {total}")
    return ScanTrainResult(model.eval(), log, time.perf_counter() - started, step, rows_seen)


__all__ = [
    "EVAL_BATCH", "GATE_EM", "HEADROOM_EM_MAX", "MONITOR_N", "SIZE_STEPS", "ScanTrainConfig", "ScanTrainResult",
    "TRAIN_STREAM", "em_summary", "epoch_order", "lr_at", "monitor_rows", "torch_predict", "train_scan",
]
