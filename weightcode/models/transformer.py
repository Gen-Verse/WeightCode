"""Encoder-only Transformer students for token-row datasets (arch ``tfm``), with their public layer list.

Layout (pre-LayerNorm, bidirectional attention, keys at PAD positions masked):

    h = tok_emb[x] + pos_emb[t]
    for i in 1..N_LAYERS:
        h = h + attention_i(ln1_i(h))            q, k, v, o: Linear(d, d); N_HEADS heads of size d / N_HEADS
        h = h + fc2_i(relu(fc1_i(ln2_i(h))))      fc1: Linear(d, d_mlp), fc2: Linear(d_mlp, d)
    logits = head(ln_f(h))                        head: Linear(d, n_out), at every position

Parameter names: ``tok_emb.weight`` [vocab, d], ``pos_emb.weight`` [seq_len, d]; per block ``b{i}.ln1.weight``/
``.bias``, ``b{i}.attn.{q,k,v,o}.weight`` [d, d] and ``.bias`` [d], ``b{i}.ln2.weight``/``.bias``,
``b{i}.mlp.fc1.weight`` [d_mlp, d]/``.bias``, ``b{i}.mlp.fc2.weight`` [d, d_mlp]/``.bias``; then ``ln_f.weight``/
``.bias`` and ``head.weight`` [n_out, d]/``.bias``. Linear weights use the PyTorch [out, in] layout. LayerNorm uses
the biased variance and eps LN_EPS. Initialization: every embedding and linear weight N(0, INIT_STD), biases 0,
LayerNorm gains 1 and biases 0, all from torch's global generator in module construction order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SEQ_ARCHS: tuple[str, ...] = ("tfm",)
LN_EPS = 1e-5
INIT_STD = 0.02
BLOCK_PREFIX = "b"


@dataclass(frozen=True)
class TfmConfig:
    """Shape of one Transformer student."""

    vocab_size: int
    seq_len: int
    n_out: int
    pad_id: int
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 256

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")
        if min(self.vocab_size, self.seq_len, self.n_out, self.d_model, self.n_layers, self.d_mlp) <= 0:
            raise ValueError("every size must be positive")
        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id outside the vocabulary")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class Attention(nn.Module):
    """Multi-head self-attention with separate q, k, v and output projections."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, a: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = a.shape
        head_dim = d_model // self.n_heads

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.n_heads, head_dim).transpose(1, 2)

        q, k, v = heads(self.q(a)), heads(self.k(a)), heads(self.v(a))
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        scores = scores.masked_fill(key_pad[:, None, None, :], float("-inf"))
        mixed = torch.softmax(scores, dim=-1) @ v
        return self.o(mixed.transpose(1, 2).reshape(batch, length, d_model))


class MLP(nn.Module):
    """fc1 -> ReLU -> fc2."""

    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(a)))


class Block(nn.Module):
    """Pre-LN residual block: attention then MLP."""

    def __init__(self, config: TfmConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=LN_EPS)
        self.attn = Attention(config.d_model, config.n_heads)
        self.ln2 = nn.LayerNorm(config.d_model, eps=LN_EPS)
        self.mlp = MLP(config.d_model, config.d_mlp)

    def forward(self, h: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.ln1(h), key_pad)
        return h + self.mlp(self.ln2(h))


class SeqTransformer(nn.Module):
    """Token rows int64 [B, seq_len] -> logits float32 [B, seq_len, n_out]."""

    def __init__(self, config: TfmConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        for index in range(1, config.n_layers + 1):
            setattr(self, f"{BLOCK_PREFIX}{index}", Block(config))
        self.ln_f = nn.LayerNorm(config.d_model, eps=LN_EPS)
        self.head = nn.Linear(config.d_model, config.n_out)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def blocks(self) -> list[Block]:
        return [getattr(self, f"{BLOCK_PREFIX}{index}") for index in range(1, self.config.n_layers + 1)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.config.seq_len:
            raise ValueError(f"token rows must be [B, {self.config.seq_len}], got {tuple(x.shape)}")
        positions = torch.arange(self.config.seq_len, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(positions)[None, :, :]
        key_pad = x == self.config.pad_id
        for block in self.blocks():
            h = block(h, key_pad)
        return self.head(self.ln_f(h))


def build_seq_model(config: TfmConfig) -> SeqTransformer:
    """Construct a student on the CPU; parameters come from torch's global generator."""
    return SeqTransformer(config)


def decay_groups(model: SeqTransformer, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW parameter groups: decay on 2-D linear weights only; embeddings, LayerNorm and biases undecayed.

    Embeddings are left undecayed so that rows of tokens absent from training (reserved operators) keep their
    initial values exactly: with zero gradient and no decay, Adam leaves a parameter unchanged.
    """
    decay, no_decay = [], []
    for module_name, module in model.named_modules():
        for name, param in module.named_parameters(recurse=False):
            if isinstance(module, nn.Linear) and name == "weight":
                decay.append(param)
            else:
                no_decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def export_seq_params(model: nn.Module) -> dict[str, np.ndarray]:
    """Float32 numpy copies of the student's parameters keyed by name."""
    params = {}
    for name, tensor in model.state_dict().items():
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} has dtype {tensor.dtype}, expected float32")
        params[name] = tensor.detach().to("cpu").contiguous().numpy().copy()
    return params


def load_seq_params(model: nn.Module, params: Mapping[str, np.ndarray]) -> nn.Module:
    """Load numpy parameters with strict key matching."""
    model.load_state_dict({name: torch.from_numpy(np.array(value)) for name, value in params.items()}, strict=True)
    return model


def config_from_params(params: Mapping[str, np.ndarray], n_heads: int, pad_id: int) -> TfmConfig:
    """The shape implied by a parameter set (n_heads and pad_id are not visible in shapes)."""
    vocab, d_model = params["tok_emb.weight"].shape
    seq_len = params["pos_emb.weight"].shape[0]
    n_layers = len({key.split(".")[0] for key in params if key.startswith(BLOCK_PREFIX) and key.split(".")[0][1:].isdigit()})
    d_mlp = params[f"{BLOCK_PREFIX}1.mlp.fc1.weight"].shape[0]
    n_out = params["head.weight"].shape[0]
    return TfmConfig(int(vocab), int(seq_len), int(n_out), int(pad_id), int(d_model), int(n_layers), int(n_heads), int(d_mlp))


LN_COMPUTES = "y = (a - mean) / sqrt(var + eps) * weight + bias over the last axis (biased variance)"
ATTENTION_COMPUTES = (
    "q = a @ q.weight.T + q.bias, k and v the same; split the last axis into n_heads heads of head_dim (head j holds "
    "columns j * head_dim .. (j + 1) * head_dim - 1); per head softmax(q @ k.T / sqrt(head_dim) + mask) @ v; heads "
    "concatenated in order; out = concat @ o.weight.T + o.bias; h = h + out"
)
MLP_COMPUTES = "h = h + relu(a @ fc1.weight.T + fc1.bias) @ fc2.weight.T + fc2.bias"


def _shape(params: Mapping[str, np.ndarray], key: str) -> list[int]:
    return [int(d) for d in params[key].shape]


def _layernorm(name: str, input_name: str, params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "name": name,
        "op": "layernorm",
        "input": input_name,
        "weight": f"{name}.weight",
        "bias": f"{name}.bias",
        "shape_w": _shape(params, f"{name}.weight"),
        "shape_b": _shape(params, f"{name}.bias"),
        "eps": LN_EPS,
        "computes": LN_COMPUTES,
    }


def seq_layer_list(config: TfmConfig, params: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Ordered public layer list of a ``tfm`` student, checked against the parameter names."""
    layers: list[dict[str, Any]] = [
        {
            "name": "tok_emb",
            "op": "embedding",
            "weight": "tok_emb.weight",
            "shape_w": _shape(params, "tok_emb.weight"),
            "computes": "h[n, t] = weight[x[n, t]] (row = token id)",
        },
        {
            "name": "pos_emb",
            "op": "position_embedding",
            "weight": "pos_emb.weight",
            "shape_w": _shape(params, "pos_emb.weight"),
            "computes": "h[n, t] = h[n, t] + weight[t] (row = position 0 .. seq_len - 1)",
        },
    ]
    for index in range(1, config.n_layers + 1):
        block = f"{BLOCK_PREFIX}{index}"
        layers.append(_layernorm(f"{block}.ln1", "h", params))
        tensors = {f"{part}.{kind}": f"{block}.attn.{part}.{kind}" for part in "qkvo" for kind in ("weight", "bias")}
        layers.append({
            "name": f"{block}.attn",
            "op": "attention",
            "input": f"{block}.ln1",
            "residual": True,
            "n_heads": config.n_heads,
            "head_dim": config.head_dim,
            "tensors": tensors,
            "shapes": {local: _shape(params, full) for local, full in tensors.items()},
            "mask": "bidirectional: every position attends to every position whose token is not PAD",
            "computes": ATTENTION_COMPUTES,
        })
        layers.append(_layernorm(f"{block}.ln2", "h", params))
        tensors = {f"{part}.{kind}": f"{block}.mlp.{part}.{kind}" for part in ("fc1", "fc2") for kind in ("weight", "bias")}
        layers.append({
            "name": f"{block}.mlp",
            "op": "mlp",
            "input": f"{block}.ln2",
            "residual": True,
            "activation": "relu",
            "tensors": tensors,
            "shapes": {local: _shape(params, full) for local, full in tensors.items()},
            "computes": MLP_COMPUTES,
        })
    layers.append(_layernorm("ln_f", "h", params))
    layers.append({
        "name": "head",
        "op": "linear",
        "input": "ln_f",
        "weight": "head.weight",
        "bias": "head.bias",
        "shape_w": _shape(params, "head.weight"),
        "shape_b": _shape(params, "head.bias"),
        "computes": "logits[n, t] = a[n, t] @ weight.T + bias, at every position t",
    })
    named: set[str] = set()
    for layer in layers:
        named |= {layer[key] for key in ("weight", "bias") if key in layer}
        named |= set(layer.get("tensors", {}).values())
    if named != set(params):
        raise ValueError(f"parameter names do not match the tfm layer list: {sorted(named ^ set(params))}")
    return layers


def describe_tokens(tokens: Sequence[str]) -> list[dict[str, Any]]:
    """Token table for the public spec: id and name."""
    return [{"id": index, "name": str(name)} for index, name in enumerate(tokens)]


def build_seq_model_spec(
    model_id: str,
    config: TfmConfig,
    token_table: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    layout: str,
    params: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Public ``model_spec.json`` of a ``tfm`` student (schema weightcode.model_spec.v0), checked for forbidden terms."""
    from weightcode.models.spec import SCHEMA, check_public

    if len(token_table) != config.vocab_size or len(class_names) != config.n_out:
        raise ValueError("token table or class names do not match the config")
    spec = {
        "schema": SCHEMA,
        "model_id": model_id,
        "input": {
            "kind": "token_ids",
            "shape": [config.seq_len],
            "dtype": "int64",
            "vocab_size": config.vocab_size,
            "pad_id": config.pad_id,
            "layout": layout,
            "tokens": [dict(row) for row in token_table],
        },
        "classes": [str(name) for name in class_names],
        "parameter_layout": "PyTorch: linear weight [out, in]; embedding weight [rows, d]; all float32",
        "n_parameters": int(sum(np.asarray(value).size for value in params.values())),
        "layers": seq_layer_list(config, params),
        "output": "logits [N, seq_len, n_classes] at every position; the prediction at a position is the argmax over "
        "classes; only OUT positions carry an answer",
        "reference_forward": "forward.py: forward(params, x_int64 [N, seq_len]) -> float32 logits [N, seq_len, n_classes]",
    }
    check_public(spec)
    return spec
