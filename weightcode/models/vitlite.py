"""The ViT-Lite-7/4 student of PREREG E11 (arch ``vit_lite_7_4``) and its training recipe.

Architecture: ``vit_7_4_32`` of SHI-Labs/Compact-Transformers at commit cd11f0912749d376f7465cb83534e768b6701d7d
(``src/vit.py``, ``src/utils/transformers.py``, ``src/utils/tokenizer.py``, ``src/utils/stochastic_depth.py``; Hassani
et al. 2021, "Escaping the Big Data Paradigm with Compact Transformers"), with the same module and parameter names:

* tokenizer: one 4x4 convolution, stride 4, no padding, 3 -> 256, with bias (``tokenizer.conv_layers.0.0``), flattened
  to 64 tokens of 256;
* a class token ``classifier.class_emb`` [1, 1, 256] (zeros at init) prepended, a learnable positional embedding
  ``classifier.positional_emb`` [1, 65, 256] (trunc_normal std 0.2) added, then dropout;
* 7 encoder layers (``classifier.blocks.i``), each ``src = src + drop_path(attn(pre_norm(src)))``, then
  ``src = norm1(src)`` (the LayerNorm is applied to the residual stream itself), then
  ``src = src + drop_path(dropout2(linear2(dropout1(gelu(linear1(src))))))``; attention has 4 heads, a bias-free qkv
  projection 256 -> 768, scale head_dim ** -0.5, softmax, attention dropout, an output projection 256 -> 256 with bias
  and projection dropout; the MLP is 256 -> 512 -> 256;
* a final LayerNorm (``classifier.norm``) on the class token, then ``classifier.fc`` 256 -> 10.

LayerNorm eps 1e-5. Initialisation as there: Linear weights trunc_normal std 0.02 with zero bias, LayerNorm weight 1
and bias 0, the convolution's weight kaiming_normal (its bias keeps PyTorch's default), modules created in the
source's order. 85 tensors, 3,716,874 entries, the names and shapes of the authors' published ViT-Lite-7/4 CIFAR-10
checkpoint. Dropout as PREREG E11 fixes it: 0.1 after the positional add, on the attention output projection and on
both MLP dropouts; attention dropout 0; stochastic depth linear from 0 to 0.1 over the 7 layers (what the public
code of the checkpoint's date ran). GELU in its tanh form (a recorded deviation; the source uses the exact erf
form), so forward.py needs no erf. The model takes raw uint8 NCHW images and applies x / 255 then per-channel
(x - mean) / std itself (``Preprocess``); training feeds already normalised augmented images to
``forward_normalized``.

Training (``train_vitlite``): ``examples/main.py`` of the same commit (sha256 b7984663...), which is the authors'
2021 CIFAR script: AdamW (lr 5e-4, betas 0.9 and 0.999, eps 1e-8, weight decay 3e-2 on every parameter), batch 128
shuffled without drop_last, 200 epochs, learning rate set at the start of each epoch, lr / (5 - epoch) for the 5
warmup epochs and then lr * 0.5 * (1 + cos(pi * (epoch - 5) / 195)); label-smoothing cross-entropy with smoothing
0.1 (0.9 NLL + 0.1 mean over classes of -log p); per image on PIL: the CIFAR-10 AutoAugment policy
(``weightcode.models.cct_autoaug``), random crop 32 after zero padding 4, horizontal flip 0.5, ToTensor, Normalize. fp32,
no clipping, no EMA, no mixup. The model after the last epoch is kept. Deviations (``RECIPE``): E1's 40,000 train
rows and E1's normalisation, tanh GELU, seeds, deterministic algorithms with IEEE fp32 matmuls.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from weightcode.models.resnet20 import Preprocess

ARCH = "vit_lite_7_4"
SOURCE_COMMIT = "cd11f0912749d376f7465cb83534e768b6701d7d"
INPUT_SHAPE: tuple[int, int, int] = (3, 32, 32)
PATCH = 4
EMBED_DIM = 256
N_LAYERS = 7
N_HEADS = 4
MLP_DIM = 512
N_TOKENS = (INPUT_SHAPE[1] // PATCH) * (INPUT_SHAPE[2] // PATCH)  # 64
SEQ_LEN = N_TOKENS + 1  # with the class token
LN_EPS = 1e-5
EVAL_BATCH = 1000
HEAD_WEIGHT = "classifier.fc.weight"
HEAD_BIAS = "classifier.fc.bias"
GELU_APPROXIMATE = "tanh"

RECIPE = (
    "Compact-Transformers cd11f0912749d376f7465cb83534e768b6701d7d examples/main.py (the authors' CIFAR script, "
    "unchanged since 2021-06 but for the model call): AdamW lr 5e-4 (betas 0.9/0.999, eps 1e-8), weight decay 3e-2 "
    "on all parameters, batch 128, 200 epochs, 5 warmup epochs lr / (5 - epoch), then cosine per epoch "
    "lr * 0.5 * (1 + cos(pi * (epoch - 5) / 195)), label smoothing 0.1, CIFAR-10 AutoAugment then random crop 32 "
    "with padding 4 and horizontal flip, fp32, no clipping, no EMA, no mixup. Model vit_7_4_32 of the same commit "
    "with dropout 0.1, attention dropout 0 and stochastic depth 0.1 (PREREG E11). Deviations: E1's 40,000 train "
    "rows and E1's per-channel normalisation; GELU in tanh form; the model after the last epoch is kept (as the "
    "script saves it); seeds 100 to 105; deterministic algorithms with IEEE fp32 matmuls."
)


def param_shapes(n_classes: int) -> dict[str, tuple[int, ...]]:
    """Exported parameter names and shapes, in the module order of the source (85 tensors)."""
    shapes: dict[str, tuple[int, ...]] = {
        "tokenizer.conv_layers.0.0.weight": (EMBED_DIM, INPUT_SHAPE[0], PATCH, PATCH),
        "tokenizer.conv_layers.0.0.bias": (EMBED_DIM,),
        "classifier.class_emb": (1, 1, EMBED_DIM),
        "classifier.positional_emb": (1, SEQ_LEN, EMBED_DIM),
    }
    for i in range(N_LAYERS):
        p = f"classifier.blocks.{i}."
        shapes.update({
            p + "pre_norm.weight": (EMBED_DIM,), p + "pre_norm.bias": (EMBED_DIM,),
            p + "self_attn.qkv.weight": (3 * EMBED_DIM, EMBED_DIM),
            p + "self_attn.proj.weight": (EMBED_DIM, EMBED_DIM), p + "self_attn.proj.bias": (EMBED_DIM,),
            p + "linear1.weight": (MLP_DIM, EMBED_DIM), p + "linear1.bias": (MLP_DIM,),
            p + "norm1.weight": (EMBED_DIM,), p + "norm1.bias": (EMBED_DIM,),
            p + "linear2.weight": (EMBED_DIM, MLP_DIM), p + "linear2.bias": (EMBED_DIM,),
        })
    shapes.update({
        "classifier.norm.weight": (EMBED_DIM,), "classifier.norm.bias": (EMBED_DIM,),
        HEAD_WEIGHT: (n_classes, EMBED_DIM), HEAD_BIAS: (n_classes,),
    })
    return shapes


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth per sample, as the source's ``drop_path``."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    """The source's ``Attention``: bias-free qkv, scaled dot-product softmax, output projection with bias."""

    def __init__(self, dim: int, num_heads: int, attention_dropout: float, projection_dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class TransformerEncoderLayer(nn.Module):
    """The source's ``TransformerEncoderLayer`` (pre_norm attention, then norm1 on the residual stream, then MLP)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, attention_dropout: float,
                 drop_path_rate: float, gelu_approximate: str) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model, eps=LN_EPS)
        self.self_attn = Attention(d_model, nhead, attention_dropout, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=LN_EPS)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.gelu_approximate = gelu_approximate

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(F.gelu(self.linear1(src), approximate=self.gelu_approximate)))
        return src + self.drop_path(self.dropout2(src2))


class _Tokenizer(nn.Module):
    """The source's ``Tokenizer`` with one bias-carrying 4x4 stride-4 convolution, no activation, no pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(nn.Sequential(
            nn.Conv2d(INPUT_SHAPE[0], EMBED_DIM, kernel_size=(PATCH, PATCH), stride=(PATCH, PATCH), padding=(0, 0),
                      bias=True),
            nn.Identity(), nn.Identity()))
        self.flattener = nn.Flatten(2, 3)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)


class _Classifier(nn.Module):
    """The source's ``TransformerClassifier`` with seq_pool False and a learnable positional embedding."""

    def __init__(self, n_classes: int, dropout: float, attention_dropout: float, stochastic_depth: float,
                 gelu_approximate: str) -> None:
        super().__init__()
        self.class_emb = nn.Parameter(torch.zeros(1, 1, EMBED_DIM), requires_grad=True)
        self.positional_emb = nn.Parameter(torch.zeros(1, SEQ_LEN, EMBED_DIM), requires_grad=True)
        nn.init.trunc_normal_(self.positional_emb, std=0.2)
        self.dropout = nn.Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, N_LAYERS)]
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(EMBED_DIM, N_HEADS, MLP_DIM, dropout, attention_dropout, dpr[i], gelu_approximate)
            for i in range(N_LAYERS)])
        self.norm = nn.LayerNorm(EMBED_DIM, eps=LN_EPS)
        self.fc = nn.Linear(EMBED_DIM, n_classes)
        self.apply(self._init_weight)

    @staticmethod
    def _init_weight(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = self.class_emb.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.positional_emb
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        return self.fc(self.norm(x)[:, 0])


@dataclass(frozen=True)
class ViTLiteConfig:
    """Regularisation and activation of the student (PREREG E11)."""

    n_classes: int = 10
    dropout: float = 0.1
    attention_dropout: float = 0.0
    stochastic_depth: float = 0.1
    gelu_approximate: str = GELU_APPROXIMATE

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ViTLite(nn.Module):
    """ViT-Lite-7/4: raw uint8 NCHW images in, logits out (``forward_normalized`` takes normalised images)."""

    def __init__(self, config: ViTLiteConfig, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        if config.n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if config.gelu_approximate not in ("tanh", "none"):
            raise ValueError(f"gelu_approximate must be 'tanh' or 'none', got {config.gelu_approximate!r}")
        self.config = config
        self.preprocess = Preprocess(mean, std)
        self.tokenizer = _Tokenizer()
        self.classifier = _Classifier(config.n_classes, config.dropout, config.attention_dropout,
                                      config.stochastic_depth, config.gelu_approximate)

    def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.tokenizer(z))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The class token after the final LayerNorm [N, 256] (the head's input) for raw images."""
        c = self.classifier
        h = self.tokenizer(self.preprocess(x, c.fc.weight.dtype))
        h = torch.cat((c.class_emb.expand(h.shape[0], -1, -1), h), dim=1) + c.positional_emb
        h = c.dropout(h)
        for blk in c.blocks:
            h = blk(h)
        return c.norm(h)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(self.preprocess(x, self.classifier.fc.weight.dtype))


def export_params(model: ViTLite) -> dict[str, np.ndarray]:
    """float32 numpy copies of every parameter, named as ``param_shapes``."""
    state = model.state_dict()
    expected = param_shapes(model.config.n_classes)
    if set(state) != set(expected):
        raise ValueError(f"state dict differs from param_shapes: {sorted(set(state) ^ set(expected))}")
    params = {}
    for name, shape in expected.items():
        value = state[name].detach().to("cpu", torch.float32).contiguous().numpy().copy()
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
        params[name] = value
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


def load_params_into(model: ViTLite, params: Mapping[str, np.ndarray]) -> ViTLite:
    check_params(params, model.config.n_classes)
    dtype = model.classifier.fc.weight.dtype
    model.load_state_dict({k: torch.from_numpy(np.asarray(v)).to(dtype) for k, v in params.items()}, strict=True)
    return model


def build_vitlite(config: ViTLiteConfig, mean: Sequence[float], std: Sequence[float],
                  params: Mapping[str, np.ndarray] | None = None, dtype: torch.dtype = torch.float32) -> ViTLite:
    """A CPU model in ``dtype`` (eval mode), with ``params`` loaded when given."""
    model = ViTLite(config, mean, std).to(dtype)
    if params is not None:
        load_params_into(model, params)
    return model.eval()


@torch.no_grad()
def eval_logits(model: ViTLite, x: np.ndarray, batch_size: int = EVAL_BATCH,
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
    return np.concatenate(outputs) if outputs else np.zeros((0, model.config.n_classes))


SPEC_SCHEMA = "weightcode.model_spec.vit_lite.v1"
PATCH_WEIGHT = "tokenizer.conv_layers.0.0.weight"
PATCH_BIAS = "tokenizer.conv_layers.0.0.bias"
CLASS_TOKEN = "classifier.class_emb"
POS_EMBED = "classifier.positional_emb"


def build_vitlite_spec(model_id: str, class_names: Sequence[str], mean: Sequence[float], std: Sequence[float],
                       params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Public ``model_spec.json`` of a ``vit_lite_7_4`` student (schema ``SPEC_SCHEMA``, the layout of the harness's
    synthetic stand-in), checked for forbidden terms (``weightcode.models.spec.check_public``)."""
    from weightcode.models.spec import check_public

    names = [str(name) for name in class_names]
    check_params(params, len(names))
    d, hd = EMBED_DIM, EMBED_DIM // N_HEADS
    layers: list[dict[str, Any]] = [
        {"name": "tokenizer.conv_layers.0.0", "op": "patch_projection", "weight": PATCH_WEIGHT, "bias": PATCH_BIAS,
         "patch": PATCH, "n_patches": N_TOKENS,
         "computes": "tokens[n, p] = patch_p(x[n]).flatten() @ weight.reshape(dim, -1).T + bias for the "
                     f"{N_TOKENS} non-overlapping {PATCH} x {PATCH} patches in row-major grid order, each flattened "
                     "as (channel, row, column); equal to conv2d with kernel and stride patch, no padding"},
        {"name": "embed", "op": "class_token_and_positions", "class_token": CLASS_TOKEN, "positions": POS_EMBED,
         "computes": "h = concat([class_emb, tokens], token axis) + positional_emb; token 0 is the class token"},
    ]
    for index in range(N_LAYERS):
        block = f"classifier.blocks.{index}"
        layers.append({
            "name": block, "op": "transformer_layer", "heads": N_HEADS, "head_dim": hd, "mlp_hidden": MLP_DIM,
            "pre_norm": {"weight": f"{block}.pre_norm.weight", "bias": f"{block}.pre_norm.bias"},
            "qkv": {"weight": f"{block}.self_attn.qkv.weight", "bias": None,
                    "rows": f"q = rows 0..{d - 1}, k = rows {d}..{2 * d - 1}, v = rows {2 * d}..{3 * d - 1}; head j "
                            f"uses columns j*{hd} .. (j+1)*{hd} - 1 of each"},
            "proj": {"weight": f"{block}.self_attn.proj.weight", "bias": f"{block}.self_attn.proj.bias"},
            "norm1": {"weight": f"{block}.norm1.weight", "bias": f"{block}.norm1.bias"},
            "linear1": {"weight": f"{block}.linear1.weight", "bias": f"{block}.linear1.bias"},
            "linear2": {"weight": f"{block}.linear2.weight", "bias": f"{block}.linear2.bias"},
            "computes": "h = h + proj(attention(pre_norm(h))); h = norm1(h); h = h + linear2(gelu(linear1(h))); "
                        "attention per head = softmax(q k^T / sqrt(head_dim)) v over all tokens",
        })
    layers.append({"name": "classifier.norm", "op": "layernorm", "weight": "classifier.norm.weight",
                   "bias": "classifier.norm.bias", "computes": "features f = norm(h)[:, 0], the class token's row"})
    layers.append({"name": "classifier.fc", "op": "linear", "weight": HEAD_WEIGHT, "bias": HEAD_BIAS,
                   "shape": [len(names), d], "computes": "logits = f @ weight.T + bias"})
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
        "layernorm": {"eps": LN_EPS, "computes": "(a - mean) / sqrt(var + eps) * weight + bias over the last axis, "
                                                 "biased variance"},
        "gelu": "tanh form: 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))",
        "parameter_layout": "PyTorch: linear weight [out, in]; patch projection weight [dim, channels, patch, patch]; "
                            "class_emb [1, 1, dim]; positional_emb [1, tokens, dim]; LayerNorm weight and bias [dim]; "
                            "all float32",
        "n_parameters": int(sum(np.asarray(value).size for value in params.values())),
        "tensors": {name: list(shape) for name, shape in param_shapes(len(names)).items()},
        "layers": layers,
        "output": "logits [N, n_classes]; the prediction is the argmax",
        "reference_forward": (f"forward.py: forward(params, x_uint8 [N, {', '.join(map(str, INPUT_SHAPE))}]) -> "
                              "float32 logits [N, n_classes]"),
    }
    check_public(spec)
    return spec


# ---------------------------------------------------------------------------------------------------------------
# Training


@dataclass(frozen=True)
class CCTTrainConfig:
    """``examples/main.py``'s CIFAR recipe as PREREG E11 fixes it (module docstring); ``RECIPE`` states deviations."""

    arch: str = ARCH
    source_commit: str = SOURCE_COMMIT
    epochs: int = 200
    warmup: int = 5
    batch_size: int = 128
    optimizer: str = "AdamW"
    lr: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 3e-2
    weight_decay_on: str = "all parameters"
    schedule: str = "warmup lr / (warmup - epoch), then cosine per epoch to epochs"
    loss: str = "label_smoothing_cross_entropy"
    smoothing: float = 0.1
    augment: str = "cifar10_autoaugment, random_crop32_pad4, hflip0.5"
    crop_pad: int = 4
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 4
    clip_grad_norm: float = 0.0
    precision: str = "fp32"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def epoch_lr(config: CCTTrainConfig, epoch: int) -> float:
    """``adjust_learning_rate`` of the source for 0-based ``epoch``."""
    lr = config.lr
    if epoch < config.warmup:
        return lr / (config.warmup - epoch)
    return lr * 0.5 * (1.0 + math.cos(math.pi * (epoch - config.warmup) / (config.epochs - config.warmup)))


def label_smoothing_cross_entropy(logits: torch.Tensor, target: torch.Tensor, smoothing: float) -> torch.Tensor:
    """The source's ``LabelSmoothingCrossEntropy``: (1 - s) NLL + s mean over classes of -log p, averaged."""
    log_prob = F.log_softmax(logits, dim=-1)
    nll = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    smooth = -log_prob.mean(dim=-1)
    return ((1.0 - smoothing) * nll + smoothing * smooth).mean()


def train_transform(mean: Sequence[float], std: Sequence[float], crop_pad: int) -> Callable[[Any], torch.Tensor]:
    """The source's training transform on a PIL image, with ``mean`` and ``std`` in place of its constants."""
    from torchvision import transforms

    from weightcode.models.cct_autoaug import CIFAR10Policy

    return transforms.Compose([
        CIFAR10Policy(),
        transforms.RandomCrop(INPUT_SHAPE[1], padding=crop_pad),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])


class AugmentedImages(Dataset):
    """uint8 NCHW images and labels; each item is a PIL image through ``transform`` (as torchvision's CIFAR10)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, transform: Callable[[Any], torch.Tensor]) -> None:
        self.x, self.y, self.transform = x, y, transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        from PIL import Image

        image = Image.fromarray(np.ascontiguousarray(self.x[i].transpose(1, 2, 0)))
        return self.transform(image), int(self.y[i])


@dataclass
class CCTTrainResult:
    """The model after the last epoch (on the training device) with its per-epoch log."""

    model: ViTLite
    log: list[dict[str, float]] = field(default_factory=list)
    final: dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0


def train_vitlite(
    config: CCTTrainConfig,
    model_config: ViTLiteConfig,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    log_fn: Callable[[Mapping[str, Any]], None] | None = None,
) -> CCTTrainResult:
    """Train one student from scratch on raw uint8 images ``x_train`` [N, 3, 32, 32] and labels ``y_train``.

    The caller sets determinism first (``weightcode.models.train.configure_determinism``, which seeds python, numpy and
    torch). Initialisation draws from ``torch.manual_seed(seed)``; the loader's shuffling and its workers' seeds
    (python ``random`` for AutoAugment, torch for crop and flip) come from a generator seeded with ``seed``, so a run
    is reproducible for a fixed ``num_workers``. ``final`` holds the clean (eval-mode, unaugmented) loss and accuracy
    on the training rows after the last epoch.
    """
    if config.optimizer != "AdamW" or config.loss != "label_smoothing_cross_entropy" or config.precision != "fp32":
        raise ValueError(f"unsupported recipe {config.optimizer}/{config.loss}/{config.precision}")
    if config.clip_grad_norm != 0.0:
        raise ValueError("the recipe has no gradient clipping")
    x_np, y_np = np.asarray(x_train), np.asarray(y_train)
    if x_np.dtype != np.uint8 or x_np.ndim != 4 or tuple(x_np.shape[1:]) != INPUT_SHAPE:
        raise TypeError(f"training images must be uint8 [N, {INPUT_SHAPE}], got {x_np.dtype}{x_np.shape}")
    if y_np.dtype != np.int64 or y_np.shape != (len(x_np),) or y_np.min() < 0 or y_np.max() >= model_config.n_classes:
        raise ValueError(f"labels must be int64 [N] in 0 .. {model_config.n_classes - 1}")
    torch.manual_seed(seed)
    random.seed(seed)
    model = ViTLite(model_config, mean, std).to(device)
    dataset = AugmentedImages(x_np, y_np, train_transform(mean, std, config.crop_pad))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, drop_last=config.drop_last,
                        num_workers=config.num_workers, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps,
                                  weight_decay=config.weight_decay)
    result = CCTTrainResult(model=model)
    started = time.perf_counter()
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        lr = epoch_lr(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        correct = torch.zeros((), dtype=torch.int64, device=device)
        seen = 0
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model.forward_normalized(images)
            loss = label_smoothing_cross_entropy(output, target, config.smoothing)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().to(torch.float64) * target.shape[0]
            correct += (output.detach().argmax(dim=1) == target).sum()
            seen += int(target.shape[0])
        train_loss = float(loss_sum.item()) / seen
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}")
        entry = {"epoch": epoch + 1, "lr": lr, "train_loss": train_loss,
                 "train_acc_augmented": float(correct.item()) / seen, "epoch_time_s": time.perf_counter() - epoch_start}
        result.log.append(entry)
        if log_fn is not None:
            log_fn(entry)
    model.eval()
    with torch.no_grad():
        loss_sum_f, correct_i = 0.0, 0
        for start in range(0, len(x_np), EVAL_BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(x_np[start:start + EVAL_BATCH])).to(device)
            yb = torch.from_numpy(y_np[start:start + EVAL_BATCH]).to(device)
            logits = model(xb)
            loss_sum_f += float(F.cross_entropy(logits, yb, reduction="sum").to(torch.float64).item())
            correct_i += int((logits.argmax(dim=1) == yb).sum().item())
    result.final = {"train_loss_clean": loss_sum_f / len(x_np), "train_acc_clean": correct_i / len(x_np),
                    "n_train": int(len(x_np))}
    result.wall_time_s = time.perf_counter() - started
    return result


__all__ = [
    "ARCH", "AugmentedImages", "CCTTrainConfig", "CCTTrainResult", "EMBED_DIM", "HEAD_BIAS", "HEAD_WEIGHT", "RECIPE",
    "SPEC_SCHEMA", "ViTLite", "ViTLiteConfig", "build_vitlite", "build_vitlite_spec", "check_params", "epoch_lr",
    "eval_logits", "export_params", "label_smoothing_cross_entropy", "load_params_into", "param_shapes",
    "train_transform", "train_vitlite",
]
