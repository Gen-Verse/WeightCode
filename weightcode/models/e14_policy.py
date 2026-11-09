"""PREREG E14 student: a feedforward FiLM policy over the full grid and the instruction (arch ``film_policy``).

Layers (fixed before training; ``forward_e14_policy.py.tmpl`` is the numpy forward and states the same):

* cell embedding: obj_emb [11, d_cell] + col_emb [6, d_cell] + state_emb [4, d_cell], summed per cell of the 8 x 8
  grid;
* instruction: word_emb [15, d_word] for each of the 5 words, concatenated, then instr (linear 5 * d_word -> d_instr)
  and ReLU;
* two FiLM convolution blocks: conv_k (3x3, padding 1, bias) to ``channels``, then z * (1 + g) + b with (g, b) from
  film_k (linear d_instr -> 2 * channels), then ReLU;
* fc (linear channels * 64 -> d_hidden) on the (channel, x, y) flattening, LayerNorm ``ln``, ReLU;
* head (linear d_hidden -> 7 action logits).

Sizes (``SIZES``): "base" d_cell 16, d_word 32, d_instr 128, channels 64, d_hidden 256 (1,151,799 parameters);
"next" (PREREG E14's next size if neither 50,000 nor 100,000 episodes pass the gate at "base") 32, 64, 256, 128, 512
(4,599,399 parameters).

Recipe (``E14TrainConfig``, fixed before training): imitation, cross-entropy on the bot's action at every step of
every usable training demonstration (``weightcode.data.e14``); AdamW lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay
0.01 on the convolution and linear weights only (no decay on embeddings, biases or LayerNorm, so the reserved word
rows "pick" and "up", which no training row holds, receive no gradient and no decay and keep their initial values
exactly; checked after training); batch 256; 20 epochs; linear warmup over the first 500 steps, then cosine decay to
0.01 lr at the last step; gradient norm clipped at 1.0; each epoch a permutation of the training steps from
``numpy.random.default_rng([seed, epoch])``; initialisation from ``torch.manual_seed(seed)``; CPU, float32,
deterministic algorithms. No checkpoint choice and no stopping rule: the final epoch is the student.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from weightcode.data import e14 as D

ARCH = "film_policy"
SPEC_SCHEMA = "weightcode.model_spec.film_policy.v1"
TEMPLATE = Path(__file__).with_name("templates") / "forward_e14_policy.py.tmpl"
LN_EPS = 1e-5
EVAL_BATCH = 1024
HEAD_TENSORS = ("head.weight", "head.bias")
WORD_EMBEDDING = "word_emb.weight"


@dataclass(frozen=True)
class PolicyConfig:
    """Layer sizes of a ``film_policy`` student."""

    d_cell: int = 16
    d_word: int = 32
    d_instr: int = 128
    channels: int = 64
    d_hidden: int = 256

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


SIZES = {"base": PolicyConfig(), "next": PolicyConfig(d_cell=32, d_word=64, d_instr=256, channels=128, d_hidden=512)}


def param_shapes(config: PolicyConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor of the student with its shape, in forward order (the numpy forward's ``expected_shapes``)."""
    c, g = config, D.GRID
    return {
        "obj_emb.weight": (len(D.OBJECTS), c.d_cell), "col_emb.weight": (len(D.COLORS), c.d_cell),
        "state_emb.weight": (D.N_STATES, c.d_cell), "word_emb.weight": (len(D.VOCAB), c.d_word),
        "instr.weight": (c.d_instr, D.INSTR_LEN * c.d_word), "instr.bias": (c.d_instr,),
        "conv1.weight": (c.channels, c.d_cell, 3, 3), "conv1.bias": (c.channels,),
        "film1.weight": (2 * c.channels, c.d_instr), "film1.bias": (2 * c.channels,),
        "conv2.weight": (c.channels, c.channels, 3, 3), "conv2.bias": (c.channels,),
        "film2.weight": (2 * c.channels, c.d_instr), "film2.bias": (2 * c.channels,),
        "fc.weight": (c.d_hidden, c.channels * g * g), "fc.bias": (c.d_hidden,),
        "ln.weight": (c.d_hidden,), "ln.bias": (c.d_hidden,),
        "head.weight": (len(D.ACTIONS), c.d_hidden), "head.bias": (len(D.ACTIONS),),
    }


def config_from_params(params: Mapping[str, np.ndarray]) -> PolicyConfig:
    """The sizes of a parameter set; refuses names or shapes of another layout."""
    config = PolicyConfig(d_cell=int(np.shape(params["obj_emb.weight"])[1]),
                          d_word=int(np.shape(params["word_emb.weight"])[1]),
                          d_instr=int(np.shape(params["instr.weight"])[0]),
                          channels=int(np.shape(params["conv1.weight"])[0]),
                          d_hidden=int(np.shape(params["fc.weight"])[0]))
    shapes = {str(k): tuple(int(s) for s in np.shape(v)) for k, v in params.items()}
    if shapes != param_shapes(config):
        raise ValueError(f"not a {ARCH} parameter set: {sorted(set(shapes) ^ set(param_shapes(config)))}")
    return config


class FilmPolicy(nn.Module):
    """The student (module docstring)."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        c = config
        self.config = config
        self.obj_emb = nn.Embedding(len(D.OBJECTS), c.d_cell)
        self.col_emb = nn.Embedding(len(D.COLORS), c.d_cell)
        self.state_emb = nn.Embedding(D.N_STATES, c.d_cell)
        self.word_emb = nn.Embedding(len(D.VOCAB), c.d_word)
        self.instr = nn.Linear(D.INSTR_LEN * c.d_word, c.d_instr)
        self.conv1 = nn.Conv2d(c.d_cell, c.channels, 3, padding=1)
        self.film1 = nn.Linear(c.d_instr, 2 * c.channels)
        self.conv2 = nn.Conv2d(c.channels, c.channels, 3, padding=1)
        self.film2 = nn.Linear(c.d_instr, 2 * c.channels)
        self.fc = nn.Linear(c.channels * D.GRID * D.GRID, c.d_hidden)
        self.ln = nn.LayerNorm(c.d_hidden, eps=LN_EPS)
        self.head = nn.Linear(c.d_hidden, len(D.ACTIONS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rows int64 [N, 197] -> action logits [N, 7]."""
        n = x.shape[0]
        grid = x[:, : D.OBS_LEN].reshape(n, D.GRID, D.GRID, D.N_CHANNELS)
        instr = x[:, D.OBS_LEN:]
        cells = self.obj_emb(grid[..., 0]) + self.col_emb(grid[..., 1]) + self.state_emb(grid[..., 2])
        z = cells.permute(0, 3, 1, 2)  # [n, d_cell, x, y]
        h = F.relu(self.instr(self.word_emb(instr).reshape(n, -1)))
        for conv, film in ((self.conv1, self.film1), (self.conv2, self.film2)):
            z = conv(z)
            gamma, beta = film(h).chunk(2, dim=1)
            z = F.relu(z * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None])
        u = F.relu(self.ln(self.fc(z.reshape(n, -1))))
        return self.head(u)


def build_policy(config: PolicyConfig, params: Mapping[str, np.ndarray] | None = None,
                 dtype: torch.dtype = torch.float32) -> FilmPolicy:
    """A CPU student of ``dtype`` in eval mode, with ``params`` loaded when given (strict names and shapes)."""
    model = FilmPolicy(config).to(dtype)
    if params is not None:
        load_params_into(model, params)
    return model.eval()


def export_params(model: FilmPolicy) -> dict[str, np.ndarray]:
    """float32 numpy copies of every tensor, named as in the state dict."""
    out = {name: tensor.detach().cpu().to(torch.float32).numpy().copy() for name, tensor in model.state_dict().items()}
    if {k: v.shape for k, v in out.items()} != param_shapes(model.config):
        raise RuntimeError("the state dict does not have the film_policy layout")
    return out


def load_params_into(model: FilmPolicy, params: Mapping[str, np.ndarray]) -> FilmPolicy:
    """Copy ``params`` into ``model`` (names and shapes must match exactly)."""
    if config_from_params(params) != model.config:
        raise ValueError("parameter sizes differ from the model's")
    dtype = next(model.parameters()).dtype
    state = {name: torch.from_numpy(np.ascontiguousarray(params[name])).to(dtype) for name in model.state_dict()}
    model.load_state_dict(state, strict=True)
    return model


def eval_logits(model: FilmPolicy, x: np.ndarray, batch_size: int = EVAL_BATCH) -> np.ndarray:
    """Logits float64 of rows ``x`` under ``model`` (no gradient)."""
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size], dtype=np.int64))
            out.append(model(chunk).to(torch.float64).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(D.ACTIONS)))


def forward_source() -> str:
    """The numpy forward.py of every film_policy student."""
    return TEMPLATE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------------
# model_spec.json


def build_spec(model_id: str, params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Public model_spec.json of a film_policy student: the observation and instruction encoding, the actions, every
    word's meaning (the reserved words marked reserved and untrained), the old mission's grammar and the layers.
    Checked for forbidden terms (``weightcode.models.spec.check_public``) and for E14's leak words."""
    from weightcode.models.spec import check_public

    config = config_from_params(params)
    words = []
    for index, word in enumerate(D.VOCAB):
        if word in D.RESERVED_WORDS:
            meaning = "reserved word; absent from all training rows (its embedding row is untrained)"
        elif word in ("go", "to"):
            meaning = "verb of the trained mission (\"go to\")"
        elif word in D.ARTICLES:
            meaning = ("article: \"the\" when exactly one object in the room matches the named color and type, "
                       "\"a\" when several do")
        elif word in D.COLORS:
            meaning = f"color {word} (color index {D.COLORS.index(word)} in the grid)"
        else:
            meaning = f"object type {word} (object index {D.OBJECTS.index(word)} in the grid)"
        words.append({"id": index, "word": word, "meaning": meaning})
    spec = {
        "schema": SPEC_SCHEMA,
        "model_id": model_id,
        "arch": ARCH,
        "kind": "policy",
        "input": {
            "kind": "observation rows", "dtype": "int64", "shape": [D.ROW_LEN],
            "layout": D.encoding_record()["row"],
            "grid": {
                "width": D.GRID, "height": D.GRID, "cell": "(object index, color index, state)",
                "order": "x (column, 0 = west) outer, y (row, 0 = north) middle, channel inner",
                "objects": [{"index": i, "name": name} for i, name in enumerate(D.OBJECTS)],
                "colors": [{"index": i, "name": name} for i, name in enumerate(D.COLORS)],
                "state": "door: 0 open, 1 closed, 2 locked; on the agent's cell its facing direction: 0 east (+x), "
                         "1 south (+y), 2 west (-x), 3 north (-y); 0 on every other cell",
                "agent": "exactly one cell has object index 10 (agent); its color index is 0 and its state the "
                         "direction. The cell in front of the agent is (x, y) + (1, 0), (0, 1), (-1, 0) or (0, -1) "
                         "for direction 0, 1, 2, 3",
                "walls": "the outer ring of cells is wall; the room's interior is 6 x 6",
                "carrying": "not encoded",
            },
            "instruction": {"length": D.INSTR_LEN, "layout": "[verb, verb, article, color, type] as word ids",
                            "words": words},
        },
        "actions": [{"index": i, "name": name, "meaning": D.ACTION_MEANINGS[name]} for i, name in enumerate(D.ACTIONS)],
        "trained_mission": {
            "grammar": "go to {the|a} {color} {type}, color in red green blue purple yellow grey, type in ball box key",
            "room": "one 8 x 8 grid (6 x 6 interior) with 9 objects (balls, boxes, keys) and the agent; no doors",
            "success": "the episode succeeds when the agent faces (is next to and looking at) an object of the named "
                       "color and type; it fails after 64 steps",
            "training": "imitation of a planner's actions on the trained mission only; the training actions are "
                        "left, right and forward",
        },
        "parameter_layout": "PyTorch: embedding [rows, dim]; linear weight [out, in] and bias [out]; conv weight "
                            "[out, in, 3, 3] and bias [out]; LayerNorm weight and bias [d_hidden]; all float32",
        "sizes": config.as_dict(),
        "n_parameters": int(sum(np.asarray(v).size for v in params.values())),
        "tensors": {name: list(shape) for name, shape in param_shapes(config).items()},
        "layers": [
            {"name": "cells", "computes": "e[n, x, y] = obj_emb.weight[object] + col_emb.weight[color] + "
                                          "state_emb.weight[state]", "shape": [config.d_cell, D.GRID, D.GRID]},
            {"name": "instr", "computes": "w = concat(word_emb.weight[id_0], ..., word_emb.weight[id_4]); "
                                          "h = relu(instr.weight @ w + instr.bias)", "shape": [config.d_instr]},
            {"name": "block1", "computes": "z = conv1(e); (g, b) = split(film1.weight @ h + film1.bias); "
                                           "z = relu(z * (1 + g) + b), g and b per channel",
             "shape": [config.channels, D.GRID, D.GRID]},
            {"name": "block2", "computes": "z = conv2(z); (g, b) = split(film2.weight @ h + film2.bias); "
                                           "z = relu(z * (1 + g) + b)", "shape": [config.channels, D.GRID, D.GRID]},
            {"name": "fc", "computes": "u = relu(ln(fc.weight @ flatten(z) + fc.bias)), flatten in (channel, x, y) "
                                       "C order, ln = LayerNorm with eps 1e-5", "shape": [config.d_hidden]},
            {"name": "head", "computes": "logits = head.weight @ u + head.bias", "shape": [len(D.ACTIONS)]},
        ],
        "conv": "3x3 cross-correlation as torch.nn.functional.conv2d, stride 1, zero padding 1, with bias",
        "output": "action logits [N, 7]; the action taken is the argmax (greedy, first index on ties)",
        "reference_forward": f"forward.py: forward(params, x_int64 [N, {D.ROW_LEN}]) -> float32 logits [N, 7]",
    }
    check_public(spec)
    text = str(spec).lower()
    hits = [word for word in D.LEAK_WORDS if word in text]
    if hits:
        raise ValueError(f"model_spec contains E14 leak words {hits}")
    return spec


# ---------------------------------------------------------------------------------------------------------------------
# Training


@dataclass(frozen=True)
class E14TrainConfig:
    """The recipe of the module docstring."""

    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    warmup_steps: int = 500
    min_lr_ratio: float = 0.01
    grad_clip: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def lr_at(config: E14TrainConfig, step: int, total: int) -> float:
    """Linear warmup over ``warmup_steps``, then cosine from lr to min_lr_ratio * lr at the last step."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    span = max(1, total - 1 - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / span)
    floor = config.min_lr_ratio * config.lr
    return floor + 0.5 * (config.lr - floor) * (1.0 + math.cos(math.pi * progress))


def decay_groups(model: FilmPolicy, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW groups: weight decay on conv and linear weights only."""
    decay, rest = [], []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name == "weight":
                decay.append(param)
            else:
                rest.append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": rest, "weight_decay": 0.0}]


@dataclass
class E14TrainResult:
    model: FilmPolicy
    log: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)
    reserved_rows_unchanged: bool = False
    wall_time_s: float = 0.0


def train_policy(config: E14TrainConfig, policy: PolicyConfig, x: np.ndarray, y: np.ndarray, seed: int,
                 log_every: int = 0, device: str = "cpu") -> E14TrainResult:
    """Train one student from scratch on rows ``x`` and actions ``y`` (module docstring). The caller has configured
    determinism (``weightcode.models.train.configure_determinism``) and the thread count.

    ``device`` "cuda" (PREREG E14 device amendment, main 1f1cbf7): the student is initialised on the CPU from the
    seed exactly as on "cpu", then moved to the GPU with the rows (once) and trained there with CUDA AdamW, the same
    hyperparameters and the same epoch permutations; the returned model is back on the CPU."""
    started = time.perf_counter()
    if x.dtype != np.int64 or x.ndim != 2 or x.shape[1] != D.ROW_LEN or y.shape != (len(x),):
        raise ValueError(f"training rows must be int64 [N, {D.ROW_LEN}] with one action each")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be cpu or cuda, got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device cuda asked for, but torch sees no CUDA device")
    _, instr = D.split_rows(x)
    if np.isin(instr, D.RESERVED_IDS).any() or not np.isin(y, D.OLD_ACTIONS).all():
        raise ValueError("a training row holds a reserved word or a target outside left, right, forward")
    torch.manual_seed(seed)
    model = FilmPolicy(policy)
    initial_reserved = model.word_emb.weight.detach()[list(D.RESERVED_IDS)].clone()
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(decay_groups(model, config.weight_decay), lr=config.lr,
                                  betas=(config.beta1, config.beta2), eps=config.eps)
    xt, yt = torch.from_numpy(x).to(device), torch.from_numpy(y.astype(np.int64)).to(device)
    n = len(x)
    per_epoch = math.ceil(n / config.batch_size)
    total = per_epoch * config.epochs
    step = 0
    log: list[dict[str, Any]] = []
    for epoch in range(config.epochs):
        order = torch.from_numpy(np.random.default_rng([seed, epoch]).permutation(n)).to(device)
        loss_sum = 0.0
        right = 0
        epoch_started = time.perf_counter()
        for start in range(0, n, config.batch_size):
            index = order[start:start + config.batch_size]
            for group in optimizer.param_groups:
                group["lr"] = lr_at(config, step, total)
            logits = model(xt[index])
            loss = F.cross_entropy(logits, yt[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            right += int((logits.detach().argmax(1) == yt[index]).sum())
            step += 1
            if log_every and step % log_every == 0:
                print(f"  step {step}/{total} loss {float(loss.detach()):.4f}", flush=True)
        row = {"epoch": epoch, "steps": step, "train_loss": loss_sum / n, "train_acc": right / n,
               "lr_end": lr_at(config, step - 1, total), "epoch_s": time.perf_counter() - epoch_started}
        log.append(row)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} ({row['epoch_s']:.1f} s)",
              flush=True)
    model.to("cpu")
    model.eval()
    unchanged = bool(torch.equal(model.word_emb.weight.detach()[list(D.RESERVED_IDS)], initial_reserved))
    if not unchanged:
        raise RuntimeError("a reserved word row changed during training")
    return E14TrainResult(model=model, log=log, final={**log[-1], "n_rows": n, "steps_total": total},
                          reserved_rows_unchanged=unchanged, wall_time_s=time.perf_counter() - started)


def torch_logits_fn(model: FilmPolicy) -> Any:
    """``rows -> logits`` of a torch student, for ``weightcode.data.e14.rollouts``."""
    def run(x: np.ndarray) -> np.ndarray:
        return eval_logits(model, x)
    return run


def numpy_logits_fn(forward: Any, params: Mapping[str, np.ndarray]) -> Any:
    """``rows -> logits`` of forward.py under ``params``, for ``weightcode.data.e14.rollouts``."""
    frozen = dict(params)

    def run(x: np.ndarray) -> np.ndarray:
        return np.asarray(forward(frozen, x))
    return run


def success_rate(result: D.RolloutResult) -> float:
    return float(np.mean(result.success)) if len(result.success) else 0.0


__all__ = ["ARCH", "E14TrainConfig", "E14TrainResult", "EVAL_BATCH", "FilmPolicy", "HEAD_TENSORS", "LN_EPS",
           "PolicyConfig", "SIZES", "SPEC_SCHEMA", "WORD_EMBEDDING", "build_policy", "build_spec",
           "config_from_params", "decay_groups", "eval_logits", "export_params", "forward_source", "load_params_into",
           "lr_at", "numpy_logits_fn", "param_shapes", "success_rate", "torch_logits_fn", "train_policy"]
