"""Training and export of the Transformer students of the seqops dataset (arch ``tfm``).

Fixed protocol (no early stopping, no checkpoint selection: the model after the last step is the model):

* data: base training rows from ``weightcode.data.seqops.TrainStream`` (old operators only, single and composed half and
  half, n uniform in 4..12, a fresh batch per step), refusing every key of the dataset's held-out splits;
* loss: cross-entropy over the OUT positions (IGNORE elsewhere);
* optimizer: AdamW, betas (0.9, 0.98), eps 1e-8; weight decay only on 2-D linear weights
  (``weightcode.models.transformer.decay_groups``), so rows of reserved tokens keep their initial values exactly;
* schedule: linear warmup to ``lr`` over ``warmup_steps``, then cosine decay to ``min_lr_ratio * lr`` at ``steps``;
  gradient norm clipped at ``grad_clip``;
* randomness: ``torch.manual_seed(seed)`` (initialization) through ``weightcode.models.train.configure_determinism``;
  batches are a function of (seed, step) alone.

The size check (PREREG E5): exact match on old operators over ``TRAIN_DEV_N`` rows of the training distribution drawn
from ``default_rng([SPLIT_SEED, TRAIN_DEV_STREAM])`` (held-out keys refused). It reads no stored split.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weightcode.data import seqops as S
from weightcode.data.registry import SPLIT_SEED
from weightcode.models import transformer as T

TRAIN_DEV_STREAM = 1106
TRAIN_DEV_N = 8000
SIZE_CHECK_EM = 0.95
"""Exact match on old operators the base must reach on the train-dev rows; below it, the size steps up once."""
EVAL_BATCH = 1024
PREFETCH = 4
"""Batches produced ahead by one background thread. A batch is a function of (seed, step) alone and one worker
produces them in step order, so prefetching changes timing only."""


@dataclass(frozen=True)
class SeqTrainConfig:
    """Recipe of one seqops student."""

    arch: str = "tfm"
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 256
    batch_size: int = 256
    steps: int = 40_000
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 1000

    def model_config(self) -> T.TfmConfig:
        return T.TfmConfig(S.VOCAB_SIZE, S.SEQ_LEN, S.N_SYMBOLS, S.PAD, self.d_model, self.n_layers, self.n_heads,
                           self.d_mlp)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


SIZE_STEPS: tuple[SeqTrainConfig, ...] = (SeqTrainConfig(), SeqTrainConfig(d_model=128, d_mlp=512))
"""The pre-registered sizes: the default, then one step up if the default misses SIZE_CHECK_EM."""


def lr_at(config: SeqTrainConfig, step: int) -> float:
    """Learning rate of optimizer step ``step`` (0-based)."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.lr * (config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine)


def train_dev_rows(exclude: np.ndarray) -> dict[str, np.ndarray]:
    """The size-check rows: old-operator rows from the training distribution, held-out keys refused."""
    rng = np.random.default_rng([SPLIT_SEED, TRAIN_DEV_STREAM])
    return S.make_split(S.OLD_ROWS, TRAIN_DEV_N, rng, exclude, S.old_rules(), S.PART_OLD)


def torch_predict(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Per-position argmax of a torch student, in batches."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), EVAL_BATCH):
            logits = model(torch.from_numpy(x[start : start + EVAL_BATCH]).to(device))
            outputs.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(outputs).astype(np.int64)


def em_by_operator(arrays: Mapping[str, np.ndarray], pred: np.ndarray) -> dict[str, Any]:
    """Exact match overall, on single and composed rows, and per (OPA, OPB) pair."""
    right = S.row_correct(arrays["y"], pred)
    single = arrays["opb"] == S.NONE
    pairs: dict[str, float] = {}
    for a, b in sorted({(int(a), int(b)) for a, b in zip(arrays["opa"], arrays["opb"])}):
        rows = (arrays["opa"] == a) & (arrays["opb"] == b)
        pairs[f"{S.TOKENS[a]}+{S.TOKENS[b]}"] = float(right[rows].mean())
    return {
        "em": float(right.mean()),
        "em_single": float(right[single].mean()),
        "em_composed": float(right[~single].mean()),
        "token_acc": S.token_accuracy(arrays["y"], pred),
        "min_pair_em": min(pairs.values()),
        "pair_em": pairs,
    }


@dataclass
class SeqTrainResult:
    model: T.SeqTransformer
    log: list[dict[str, Any]]
    wall_time_s: float
    n_refused: int
    reserved_rows_unchanged: bool


def train_seq(
    config: SeqTrainConfig,
    seed: int,
    exclude: np.ndarray,
    device: torch.device,
    monitor: Callable[[T.SeqTransformer], dict[str, Any]] | None = None,
) -> SeqTrainResult:
    """Train one student from scratch under the fixed protocol (module docstring)."""
    started = time.perf_counter()
    model = T.build_seq_model(config.model_config()).to(device)
    initial_reserved = model.tok_emb.weight.detach()[list(S.NEW_OP_IDS)].clone()
    optimizer = torch.optim.AdamW(
        T.decay_groups(model, config.weight_decay), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps
    )
    stream = S.TrainStream(seed, config.batch_size, exclude)
    log: list[dict[str, Any]] = []
    running = torch.zeros((), dtype=torch.float64, device=device)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pending = deque(pool.submit(stream.batch, s) for s in range(min(PREFETCH, config.steps)))
        for step in range(config.steps):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(config, step)
            x, y = pending.popleft().result()
            if step + PREFETCH < config.steps:
                pending.append(pool.submit(stream.batch, step + PREFETCH))
            model.train()
            logits = model(torch.from_numpy(x).to(device))
            loss = F.cross_entropy(logits.reshape(-1, S.N_SYMBOLS), torch.from_numpy(y).to(device).reshape(-1),
                                   ignore_index=S.IGNORE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running += loss.detach().double()  # summed on the device: no host sync per step
            if (step + 1) % config.log_every == 0 or step + 1 == config.steps:
                entry: dict[str, Any] = {
                    "step": step + 1,
                    "lr": lr_at(config, step),
                    "mean_loss": float(running) / (config.log_every if (step + 1) % config.log_every == 0 else (step + 1) % config.log_every),
                    "grad_norm": float(grad_norm),
                    "elapsed_s": time.perf_counter() - started,
                }
                running.zero_()
                if monitor is not None:
                    entry.update(monitor(model))
                log.append(entry)
    finally:
        pool.shutdown(cancel_futures=True)
    unchanged = bool(torch.equal(model.tok_emb.weight.detach()[list(S.NEW_OP_IDS)], initial_reserved))
    return SeqTrainResult(model.eval(), log, time.perf_counter() - started, stream.n_refused, unchanged)
