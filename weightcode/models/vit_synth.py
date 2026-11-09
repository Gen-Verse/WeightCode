"""Synthetic ViT-Lite students for developing the PREREG E11 harness. Never a student of record.

PREREG E11 replaces E10's ResNet-20 by ViT-Lite-7/4 (Hassani et al. 2021, "Escaping the Big Data Paradigm with
Compact Transformers", the CIFAR variant of SHI-Labs/Compact-Transformers). The real students, their forward.py and
their export come from the data session. Until they exist, this module gives the harness a stand-in of the same
architecture, random-initialised and untrained:

* ``VitConfig``, ``VIT_LITE_7_4``: 3 x 32 x 32 input, 4 x 4 patches (64 tokens) and a class token, a learnable
  positional embedding, 7 layers of width 256 with 4 heads and MLP hidden 512, a final LayerNorm and a linear head on
  the class token, tanh GELU, no BatchNorm. Tensor names and the layer follow the authors' ``ViTLite`` module:
  ``tokenizer.conv_layers.0.0`` (the patch projection, a convolution with kernel and stride 4 and a bias),
  ``classifier.class_emb`` [1, 1, 256], ``classifier.positional_emb`` [1, 65, 256],
  ``classifier.blocks.<i>.{pre_norm, self_attn.qkv, self_attn.proj, linear1, norm1, linear2}``, ``classifier.norm``
  and ``classifier.fc``: 85 tensors, 3,716,874 entries. As in the authors' TransformerEncoderLayer, ``norm1``
  normalises the residual stream after attention and the MLP adds onto it, and qkv has no bias. ``TINY`` is a small
  configuration of the same layout for the tests.
* ``render_forward``: forward.py in numpy float64 (template ``templates/forward_vit_synth.py.tmpl``: the patch
  projection as a matrix product, softmax attention, LayerNorm with torch's eps 1e-5 and biased variance, tanh GELU).
* ``torch_logits``: an independent torch rebuild (``F.conv2d`` tokenizer, ``F.layer_norm``,
  ``F.gelu(approximate="tanh")``, explicit softmax attention) for parity.
* ``init_params``: a numpy random initialisation scaled so that the untrained network's prediction depends on its
  input, as a trained one's does: kaiming normal patch projection, linear weights N(0, 1 / fan_in), zero biases,
  LayerNorm 1 and 0, class token and positional embedding N(0, 0.02). It is not the authors' scheme (truncated normal
  0.02 linear weights, positional embedding 0.2, zero class token), under which an untrained ViT-Lite gives one class
  for every image. The class token is not zero for a second reason: a changed all-zero tensor of rank >= 2 has an
  infinite relative Frobenius norm, so every edit of it would be refused (a trained class token is not zero).
* ``build_model_spec``, ``write_export``: model.npz, model_spec.json and forward.py in the format of the E10 students,
  plus ``synthetic.json``, which marks the export synthetic.
* ``build_store``: a synthetic E11 store under a scratch root (refused unless ``WEIGHTCODE_PRIVATE``, ``WEIGHTCODE_SEALED`` and
  ``WEIGHTCODE_RUNS`` all point inside it): random uint8 images, four stand-in transforms under the E10 condition names, the
  synthetic students, evidence-v1 dev instances by the project's own builder, and the replay draw. Its labels are
  planted so that ``plant_deltas`` (a LayerNorm edit) fixes rows of the planted condition and breaks no clean row.

Nothing here reads the real store, V_select of a real dataset or the sealed store.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

ARCH = "vit_lite_synthetic"
SPEC_SCHEMA = "weightcode.model_spec.vit_lite.v1"
SYNTHETIC_SCHEMA = "weightcode.e11.synthetic_export.v1"
SYNTHETIC_NOTE = ("synthetic random-init ViT-Lite stand-in for E11 harness development; untrained; never a student of "
                  "record")
TEMPLATE_NAME = "forward_vit_synth.py.tmpl"
PLACEHOLDERS: tuple[str, ...] = ("@@INPUT_SHAPE@@", "@@MEAN@@", "@@STD@@", "@@N_CLASSES@@", "@@PATCH@@", "@@DIM@@",
                                 "@@DEPTH@@", "@@HEADS@@", "@@MLP_HIDDEN@@", "@@LN_EPS@@")
PATCH_WEIGHT = "tokenizer.conv_layers.0.0.weight"
PATCH_BIAS = "tokenizer.conv_layers.0.0.bias"
CLASS_TOKEN = "classifier.class_emb"
POS_EMBED = "classifier.positional_emb"
FINAL_NORM_WEIGHT = "classifier.norm.weight"
HEAD_WEIGHT = "classifier.fc.weight"
HEAD_BIAS = "classifier.fc.bias"


@dataclass(frozen=True)
class VitConfig:
    """Shape of one ViT-Lite student."""

    image_shape: tuple[int, int, int] = (3, 32, 32)
    patch: int = 4
    dim: int = 256
    depth: int = 7
    heads: int = 4
    mlp_hidden: int = 512
    n_classes: int = 10
    ln_eps: float = 1e-5

    def __post_init__(self) -> None:
        c, h, w = self.image_shape
        if min(c, h, w, self.patch, self.dim, self.depth, self.heads, self.mlp_hidden) <= 0:
            raise ValueError("every size must be positive")
        if h % self.patch or w % self.patch:
            raise ValueError(f"image {h} x {w} is not divisible into {self.patch} x {self.patch} patches")
        if self.dim % self.heads:
            raise ValueError(f"dim {self.dim} is not divisible by heads {self.heads}")
        if self.n_classes < 2 or not self.ln_eps > 0.0:
            raise ValueError("n_classes must be at least 2 and ln_eps positive")

    @property
    def n_patches(self) -> int:
        return (self.image_shape[1] // self.patch) * (self.image_shape[2] // self.patch)

    @property
    def n_tokens(self) -> int:
        return self.n_patches + 1

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "image_shape": list(self.image_shape)}


VIT_LITE_7_4 = VitConfig()
TINY = VitConfig(image_shape=(3, 8, 8), patch=4, dim=16, depth=2, heads=2, mlp_hidden=32, n_classes=4)


def block_names(cfg: VitConfig) -> list[str]:
    """``classifier.blocks.<i>`` for every layer, in forward order."""
    return [f"classifier.blocks.{index}" for index in range(cfg.depth)]


def param_shapes(cfg: VitConfig) -> dict[str, tuple[int, ...]]:
    """Every parameter name and shape, in forward order (the model.npz layout; forward.py's ``expected_shapes``)."""
    d, hidden = cfg.dim, cfg.mlp_hidden
    shapes: dict[str, tuple[int, ...]] = {
        PATCH_WEIGHT: (d, cfg.image_shape[0], cfg.patch, cfg.patch),
        PATCH_BIAS: (d,),
        CLASS_TOKEN: (1, 1, d),
        POS_EMBED: (1, cfg.n_tokens, d),
    }
    for block in block_names(cfg):
        shapes.update({
            f"{block}.pre_norm.weight": (d,), f"{block}.pre_norm.bias": (d,),
            f"{block}.self_attn.qkv.weight": (3 * d, d),
            f"{block}.self_attn.proj.weight": (d, d), f"{block}.self_attn.proj.bias": (d,),
            f"{block}.linear1.weight": (hidden, d), f"{block}.linear1.bias": (hidden,),
            f"{block}.norm1.weight": (d,), f"{block}.norm1.bias": (d,),
            f"{block}.linear2.weight": (d, hidden), f"{block}.linear2.bias": (d,),
        })
    shapes.update({FINAL_NORM_WEIGHT: (d,), "classifier.norm.bias": (d,), HEAD_WEIGHT: (cfg.n_classes, d),
                   HEAD_BIAS: (cfg.n_classes,)})
    return shapes


def layernorm_tensors(cfg: VitConfig) -> list[str]:
    """The LayerNorm weights and biases (1-D), in forward order."""
    names = [f"{block}.{norm}.{part}" for block in block_names(cfg) for norm in ("pre_norm", "norm1")
             for part in ("weight", "bias")]
    return [*names, FINAL_NORM_WEIGHT, "classifier.norm.bias"]


TOKEN_INIT_STD = 0.02
"""Standard deviation of the stand-in's class token and positional embedding (module docstring)."""


def init_params(cfg: VitConfig, init_seed: int) -> dict[str, np.ndarray]:
    """float32 random initialisation (module docstring), named and ordered as ``param_shapes``."""
    rng = np.random.default_rng([20261104, int(init_seed)])
    norms = set(layernorm_tensors(cfg))
    params: dict[str, np.ndarray] = {}
    for name, shape in param_shapes(cfg).items():
        if name == PATCH_WEIGHT:
            fan_in = shape[1] * shape[2] * shape[3]
            value = rng.standard_normal(shape) * math.sqrt(2.0 / fan_in)  # kaiming_normal_, fan-in, ReLU gain
        elif name in (POS_EMBED, CLASS_TOKEN):
            value = rng.standard_normal(shape) * TOKEN_INIT_STD
        elif len(shape) == 2:
            value = rng.standard_normal(shape) / math.sqrt(shape[1])
        elif name in norms:
            value = np.ones(shape) if name.endswith(".weight") else np.zeros(shape)
        else:  # linear and patch biases
            value = np.zeros(shape)
        params[name] = np.ascontiguousarray(value, dtype=np.float32)
    return params


def _float_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(repr(float(value)) for value in values) + "]"


def render_forward(cfg: VitConfig, mean: Sequence[float], std: Sequence[float]) -> str:
    """forward.py source for ``cfg`` with the per-channel normalisation (constants as exact round-trip reprs)."""
    if len(mean) != cfg.image_shape[0] or len(std) != cfg.image_shape[0] or any(float(v) <= 0.0 for v in std):
        raise ValueError(f"input shape {cfg.image_shape} needs {cfg.image_shape[0]} mean and positive std values")
    text = resources.files("weightcode.models").joinpath("templates", TEMPLATE_NAME).read_text(encoding="utf-8")
    values = {
        "@@INPUT_SHAPE@@": repr(tuple(int(d) for d in cfg.image_shape)),
        "@@MEAN@@": _float_list(mean),
        "@@STD@@": _float_list(std),
        "@@N_CLASSES@@": str(int(cfg.n_classes)),
        "@@PATCH@@": str(int(cfg.patch)),
        "@@DIM@@": str(int(cfg.dim)),
        "@@DEPTH@@": str(int(cfg.depth)),
        "@@HEADS@@": str(int(cfg.heads)),
        "@@MLP_HIDDEN@@": str(int(cfg.mlp_hidden)),
        "@@LN_EPS@@": repr(float(cfg.ln_eps)),
    }
    for placeholder in PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        text = text.replace(placeholder, values[placeholder])
    if "@@" in text:
        raise ValueError("unreplaced placeholder left in the ViT forward template")
    compile(text, "forward.py", "exec")
    return text


def build_model_spec(model_id: str, cfg: VitConfig, class_names: Sequence[str], mean: Sequence[float],
                     std: Sequence[float], params: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Public ``model_spec.json`` (schema ``SPEC_SCHEMA``), checked for forbidden terms
    (``weightcode.models.spec.check_public``)."""
    from weightcode.models.spec import check_public

    expected = param_shapes(cfg)
    if {name: tuple(np.shape(value)) for name, value in params.items()} != expected:
        raise ValueError("params do not have the configuration's names and shapes")
    if len(class_names) != cfg.n_classes:
        raise ValueError(f"{len(class_names)} class names for {cfg.n_classes} outputs")
    d, hd = cfg.dim, cfg.dim // cfg.heads
    layers: list[dict[str, Any]] = [
        {"name": "tokenizer.conv_layers.0.0", "op": "patch_projection", "weight": PATCH_WEIGHT, "bias": PATCH_BIAS,
         "patch": cfg.patch, "n_patches": cfg.n_patches,
         "computes": "tokens[n, p] = patch_p(x[n]).flatten() @ weight.reshape(dim, -1).T + bias for the "
                     f"{cfg.n_patches} non-overlapping {cfg.patch} x {cfg.patch} patches in row-major grid order, "
                     "each flattened as (channel, row, column); equal to conv2d with kernel and stride patch, no "
                     "padding"},
        {"name": "embed", "op": "class_token_and_positions", "class_token": CLASS_TOKEN, "positions": POS_EMBED,
         "computes": "h = concat([class_emb, tokens], token axis) + positional_emb; token 0 is the class token"},
    ]
    for block in block_names(cfg):
        layers.append({
            "name": block, "op": "transformer_layer", "heads": cfg.heads, "head_dim": hd, "mlp_hidden": cfg.mlp_hidden,
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
    layers.append({"name": "classifier.norm", "op": "layernorm", "weight": FINAL_NORM_WEIGHT,
                   "bias": "classifier.norm.bias", "computes": "features f = norm(h)[:, 0], the class token's row"})
    layers.append({"name": "classifier.fc", "op": "linear", "weight": HEAD_WEIGHT, "bias": HEAD_BIAS,
                   "shape": [cfg.n_classes, d], "computes": "logits = f @ weight.T + bias"})
    spec = {
        "schema": SPEC_SCHEMA,
        "model_id": model_id,
        "arch": "vit_lite",
        "input": {"kind": "images", "shape": list(cfg.image_shape), "dtype": "uint8", "layout": "NCHW"},
        "input_shape": list(cfg.image_shape),
        "preprocessing": {"mean": [float(v) for v in mean], "std": [float(v) for v in std],
                          "computes": "x / 255, then per channel (x - mean[c]) / std[c]"},
        "classes": [str(name) for name in class_names],
        "n_classes": cfg.n_classes,
        "layernorm": {"eps": cfg.ln_eps, "computes": "(a - mean) / sqrt(var + eps) * weight + bias over the last "
                                                     "axis, biased variance"},
        "gelu": "tanh form: 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))",
        "parameter_layout": "PyTorch: linear weight [out, in]; patch projection weight [dim, channels, patch, patch]; "
                            "class_emb [1, 1, dim]; positional_emb [1, tokens, dim]; LayerNorm weight and bias [dim]; "
                            "all float32",
        "n_parameters": int(sum(np.asarray(value).size for value in params.values())),
        "tensors": {name: list(shape) for name, shape in expected.items()},
        "layers": layers,
        "output": "logits [N, n_classes]; the prediction is the argmax",
        "reference_forward": (f"forward.py: forward(params, x_uint8 [N, {', '.join(map(str, cfg.image_shape))}]) -> "
                              "float32 logits [N, n_classes]"),
    }
    check_public(spec)
    return spec


def torch_logits(cfg: VitConfig, mean: Sequence[float], std: Sequence[float], params: Mapping[str, np.ndarray],
                 x_uint8: np.ndarray, dtype: Any = None, batch_size: int = 256) -> np.ndarray:
    """Logits of an independent torch rebuild in ``dtype`` (float64 by default), as a float64 numpy array."""
    import torch
    import torch.nn.functional as F

    dtype = torch.float64 if dtype is None else dtype
    t = {name: torch.from_numpy(np.asarray(value)).to(dtype) for name, value in params.items()}
    mu = torch.tensor([float(v) for v in mean], dtype=dtype).reshape(1, -1, 1, 1)
    sd = torch.tensor([float(v) for v in std], dtype=dtype).reshape(1, -1, 1, 1)
    d, heads = cfg.dim, cfg.heads
    hd = d // heads
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x_uint8), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(x_uint8[start:start + batch_size])).to(dtype)
            x = (x / 255.0 - mu) / sd
            tokens = F.conv2d(x, t[PATCH_WEIGHT], t[PATCH_BIAS], stride=cfg.patch).flatten(2, 3).transpose(-2, -1)
            n = tokens.shape[0]
            h = torch.cat([t[CLASS_TOKEN].expand(n, -1, -1), tokens], dim=1) + t[POS_EMBED]
            for block in block_names(cfg):
                a = F.layer_norm(h, (d,), t[f"{block}.pre_norm.weight"], t[f"{block}.pre_norm.bias"], cfg.ln_eps)
                qkv = F.linear(a, t[f"{block}.self_attn.qkv.weight"]).reshape(n, cfg.n_tokens, 3, heads, hd)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                attn = ((q @ k.transpose(-2, -1)) * hd ** -0.5).softmax(dim=-1)
                mixed = (attn @ v).transpose(1, 2).reshape(n, cfg.n_tokens, d)
                h = h + F.linear(mixed, t[f"{block}.self_attn.proj.weight"], t[f"{block}.self_attn.proj.bias"])
                h = F.layer_norm(h, (d,), t[f"{block}.norm1.weight"], t[f"{block}.norm1.bias"], cfg.ln_eps)
                hidden = F.gelu(F.linear(h, t[f"{block}.linear1.weight"], t[f"{block}.linear1.bias"]),
                                approximate="tanh")
                h = h + F.linear(hidden, t[f"{block}.linear2.weight"], t[f"{block}.linear2.bias"])
            h = F.layer_norm(h, (d,), t[FINAL_NORM_WEIGHT], t["classifier.norm.bias"], cfg.ln_eps)
            outputs.append(F.linear(h[:, 0], t[HEAD_WEIGHT], t[HEAD_BIAS]).to(torch.float64).numpy())
    return np.concatenate(outputs) if outputs else np.zeros((0, cfg.n_classes))


def model_id_for(dataset: str, init_seed: int) -> str:
    """The synthetic student's opaque id: ``m`` + 8 hex of a hash that says synthetic."""
    return "m" + hashlib.sha256(f"{dataset}|synthetic|{int(init_seed)}".encode()).hexdigest()[:8]


SYNTH_MEAN = (0.5, 0.5, 0.5)
SYNTH_STD = (0.25, 0.25, 0.25)
CLASS_NAMES_10 = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def write_export(out_dir: Path, cfg: VitConfig, init_seed: int, class_names: Sequence[str],
                 mean: Sequence[float] = SYNTH_MEAN, std: Sequence[float] = SYNTH_STD,
                 model_id: str | None = None) -> dict[str, Any]:
    """Write model.npz, model_spec.json, forward.py and synthetic.json into a new directory ``out_dir``; return the
    synthetic.json record. Refuses an existing non-empty directory."""
    from weightcode.common.identity import params_sha256, save_params

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; a synthetic export is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    params = init_params(cfg, init_seed)
    model_id = model_id or model_id_for("synthetic", init_seed)
    digest = save_params(out / "model.npz", params)
    (out / "model_spec.json").write_text(json.dumps(build_model_spec(model_id, cfg, class_names, mean, std, params),
                                                    indent=2) + "\n", encoding="utf-8")
    source = render_forward(cfg, mean, std)
    (out / "forward.py").write_text(source, encoding="utf-8")
    record = {
        "schema": SYNTHETIC_SCHEMA, "synthetic": True, "note": SYNTHETIC_NOTE, "model_id": model_id,
        "init_seed": int(init_seed), "config": cfg.as_dict(), "mean": list(mean), "std": list(std),
        "params_sha256": digest, "params_sha256_check": params_sha256(params),
        "forward_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "n_parameters": int(sum(value.size for value in params.values())), "n_tensors": len(params),
    }
    (out / "synthetic.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------------------------------------------------------
# Synthetic store (harness dry runs and tests)

PLANT_FLIP_FRACTION = 0.25
"""``plant_deltas`` flips the sign of the final LayerNorm weight on this fraction of the dimensions."""


def plant_deltas(cfg: VitConfig, params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The planted LayerNorm edit: delta = -2 * weight on the first ``PLANT_FLIP_FRACTION`` of the dimensions of
    ``classifier.norm.weight`` (its sign flips there). 1-D, so exempt from the relative Frobenius limit and counted in
    the changed entries."""
    weight = np.asarray(params[FINAL_NORM_WEIGHT], dtype=np.float64)
    delta = np.zeros_like(weight)
    cut = int(round(PLANT_FLIP_FRACTION * cfg.dim))
    delta[:cut] = -2.0 * weight[:cut]
    return {FINAL_NORM_WEIGHT: delta}


def stand_in_transform(corruption: str, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A synthetic stand-in for one corruption name, on uint8 images (only the name is the benchmark's)."""
    a = x.astype(np.float64)
    if corruption == "gaussian_noise":
        out = a + rng.normal(0.0, 60.0, size=a.shape)
    elif corruption == "defocus_blur":
        padded = np.pad(a, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
        out = sum(padded[:, :, i:i + a.shape[2], j:j + a.shape[3]] for i in range(3) for j in range(3)) / 9.0
    elif corruption == "snow":
        out = 0.4 * a + 150.0
    elif corruption == "fog":
        out = 0.6 * a + 90.0
    elif corruption == "contrast":
        mean = a.mean(axis=(1, 2, 3), keepdims=True)
        out = (a - mean) * 0.2 + mean
    else:
        raise ValueError(f"no stand-in transform for {corruption!r}")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def planted_labels(base_clean: np.ndarray, plant_clean: np.ndarray, base_cond: np.ndarray, plant_cond: np.ndarray,
                   n_classes: int, rng: np.random.Generator) -> np.ndarray:
    """One label per row (shared by the clean row and its condition rows) such that the planted child breaks no row,
    clean or condition, and fixes the condition rows where it changes the base prediction unless that would break the
    clean row:

    * the plant changes the condition prediction and keeping it breaks no clean row: y = the planted condition pred
      (a condition fix; the clean row is right under both or fixed);
    * else, when the plant leaves the clean prediction alone: y = the clean base prediction (the condition prediction
      is then unchanged too, so nothing changes);
    * else a class none of the four predictions names (wrong everywhere), or, when the four predictions are four
      distinct classes and none is left, the planted clean prediction (a clean fix; wrong on the condition row for
      both).

    So under these labels the plant's score on any split is score_new >= 0 and score_old >= 0 (with at least 4
    classes, which the last case needs).
    """
    if n_classes < 4:
        raise ValueError("planted labels need at least 4 classes")
    y = np.empty(len(base_clean), dtype=np.int64)
    for i in range(len(y)):
        breaks_clean = base_clean[i] == plant_cond[i] != plant_clean[i]
        if plant_cond[i] != base_cond[i] and not breaks_clean:
            y[i] = plant_cond[i]
        elif base_clean[i] == plant_clean[i]:
            y[i] = base_clean[i]
        else:
            taken = (base_clean[i], plant_clean[i], plant_cond[i], base_cond[i])
            others = [c for c in range(n_classes) if c not in taken]
            y[i] = int(rng.choice(others)) if others else int(plant_clean[i])
    return y


@dataclass(frozen=True)
class SyntheticStore:
    """A synthetic E11 store: its root, dataset, model rows by seed and dev instance ids by (condition, seed)."""

    root: Path
    dataset: str
    models: dict[int, str]
    ids: dict[tuple[str, int], str]
    plant_condition: str
    record: dict[str, Any]


def _inside(path: str | None, root: Path) -> bool:
    if not path:
        return False
    resolved = Path(os.path.abspath(os.path.expanduser(path)))
    return resolved == root or root in resolved.parents


def build_store(
    root: Path,
    cfg: VitConfig,
    *,
    dataset: str,
    conditions: Mapping[str, str],
    plant_condition: str,
    class_names: Sequence[str],
    seeds: Sequence[int] = (100, 101),
    decoys: Mapping[int, int] | None = None,
    rows: Mapping[str, int] | None = None,
    instance_conditions: Sequence[str] | None = None,
    replay_n: int = 64,
    data_seed: int = 20261104,
) -> SyntheticStore:
    """Build a synthetic store under ``root`` (module docstring). ``conditions`` maps condition name to the
    corruption name its stand-in transform uses; ``instance_conditions`` (default all) get evidence-v1 dev instances
    for every seed, with ``decoys`` (default: seeds paired in order). The environment must already point
    ``WEIGHTCODE_PRIVATE``, ``WEIGHTCODE_SEALED`` and ``WEIGHTCODE_RUNS`` at ``root``/private, sealed and runs."""
    from weightcode.common.identity import params_sha256, save_params
    from weightcode.evalkit.companions import build_replay
    from weightcode.evalkit.instances import build_deployment_instances
    from weightcode.evalkit.patch_check import load_forward, predict
    from weightcode.evalkit.targets import SeedSelection

    root = Path(os.path.abspath(root))
    for variable, sub in (("WEIGHTCODE_PRIVATE", "private"), ("WEIGHTCODE_SEALED", "sealed"), ("WEIGHTCODE_RUNS", "runs")):
        if not _inside(os.environ.get(variable), root / sub):
            raise RuntimeError(f"{variable} must point inside {root / sub} for a synthetic store, got "
                               f"{os.environ.get(variable)!r}; the real store is never written")
    if plant_condition not in conditions:
        raise ValueError(f"plant condition {plant_condition!r} is not among {sorted(conditions)}")
    seeds = [int(seed) for seed in seeds]
    if decoys is None:
        decoys = {seeds[i]: seeds[i + 1 if i % 2 == 0 else i - 1] for i in range(len(seeds) - len(seeds) % 2)}
    rows = dict(rows or {"train": 128, "V_select": 256, "V_evidence": 128, "V_check": 256, "test": 64})
    names = [str(name) for name in class_names]
    private, sealed, runs = root / "private", root / "sealed", root / "runs"
    data = private / dataset
    (data / "splits").mkdir(parents=True)
    (sealed / dataset).mkdir(parents=True)
    runs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(data_seed)

    source = render_forward(cfg, SYNTH_MEAN, SYNTH_STD)
    (data / "forward.py").write_text(source, encoding="utf-8")
    forward, forward_sha = load_forward(data / "forward.py")
    params = {seed: init_params(cfg, seed) for seed in seeds}
    base = params[seeds[0]]
    planted = {name: (np.asarray(value, np.float64) + plant_deltas(cfg, base).get(name, 0.0)).astype(np.float32)
               for name, value in base.items()}

    def pred(p: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
        return predict(forward, p, x, 256)

    offset = 0
    clean: dict[str, dict[str, np.ndarray]] = {}
    cond_x: dict[str, dict[str, np.ndarray]] = {name: {} for name in conditions}
    for split, n in rows.items():
        x = rng.integers(0, 256, size=(n, *cfg.image_shape), dtype=np.uint8)
        idx = np.arange(offset, offset + n, dtype=np.int64)
        offset += n
        if split != "train":
            for name, corruption in conditions.items():
                cond_x[name][split] = stand_in_transform(corruption, x, rng)
        if split == "train":
            y = pred(base, x).astype(np.int64)
        else:
            px = cond_x[plant_condition][split]
            y = planted_labels(pred(base, x), pred(planted, x), pred(base, px), pred(planted, px), cfg.n_classes,
                               rng)
        clean[split] = {"x": x, "y": y, "idx": idx}
    for split, arrays in clean.items():
        target = sealed / dataset / "test.npz" if split == "test" else data / "splits" / f"{split}.npz"
        np.savez(target, **arrays)
    meta = {
        "dataset": dataset, "synthetic": True, "note": SYNTHETIC_NOTE, "class_names": names,
        "image_shape": list(cfg.image_shape), "norm": {"mean": list(SYNTH_MEAN), "std": list(SYNTH_STD)},
        "split_seed": data_seed, "forward_sha256": forward_sha,
        "splits": {split: {"path": f"splits/{split}.npz", "n": int(len(arrays["y"])), "sha256": params_sha256(arrays)}
                   for split, arrays in clean.items() if split != "test"},
        "test": {"location": f"$WEIGHTCODE_SEALED/{dataset}/test.npz", "n": int(len(clean["test"]["y"])),
                 "sha256": params_sha256(clean["test"])},
    }
    (data / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for name, corruption in conditions.items():
        folder = data / "conditions" / name
        (folder / "splits").mkdir(parents=True)
        (sealed / dataset / "conditions" / name).mkdir(parents=True)
        splits = {}
        for split in ("V_select", "V_evidence", "V_check", "test"):
            arrays = {**clean[split], "x": cond_x[name][split]}
            if split == "test":
                np.savez(sealed / dataset / "conditions" / name / "test.npz", **arrays)
                continue
            np.savez(folder / "splits" / f"{split}.npz", **arrays)
            splits[split] = {"path": f"splits/{split}.npz", "n": int(len(arrays["y"])),
                             "sha256": params_sha256(arrays), "clean_sha256": meta["splits"][split]["sha256"]}
        test = {**clean["test"], "x": cond_x[name]["test"]}
        condition_meta = {
            "schema": "weightcode.dataset.v0", "dataset": dataset, "synthetic": True,
            "condition": {"name": name, "kind": "corruption", "corruption": corruption, "severity": 5,
                          "constant": "synthetic stand-in", "rng": None},
            "class_names": names, "image_shape": list(cfg.image_shape), "norm": meta["norm"],
            "forward_sha256": forward_sha, "splits": splits,
            "test": {"location": f"$WEIGHTCODE_SEALED/{dataset}/conditions/{name}/test.npz", "n": int(len(test["y"])),
                     "sha256": params_sha256(test), "clean_sha256": meta["test"]["sha256"]},
        }
        (folder / "dataset.json").write_text(json.dumps(condition_meta, indent=2), encoding="utf-8")

    index = []
    models: dict[int, str] = {}
    for seed in seeds:
        model_id = model_id_for(dataset, seed)
        folder = data / "models" / model_id
        folder.mkdir(parents=True)
        digest = save_params(folder / "model.npz", params[seed])
        spec = build_model_spec(model_id, cfg, names, SYNTH_MEAN, SYNTH_STD, params[seed])
        (folder / "model_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        acc = float(np.mean(pred(params[seed], clean["V_select"]["x"]) == clean["V_select"]["y"]))
        (folder / "private.json").write_text(json.dumps({
            "schema": SYNTHETIC_SCHEMA, "synthetic": True, "note": SYNTHETIC_NOTE, "model_id": model_id,
            "seed": seed, "arch": ARCH, "role": "dev", "params_sha256": digest, "config": cfg.as_dict(),
            "n_parameters": int(sum(v.size for v in params[seed].values())), "n_tensors": len(params[seed]),
            "readouts": {"clean_vselect_acc": acc}}, indent=2), encoding="utf-8")
        index.append({"model_id": model_id, "seed": seed, "arch": ARCH, "role": "dev", "params_sha256": digest,
                      "vselect_acc": acc})
        models[seed] = model_id
    (data / "models" / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    ids: dict[tuple[str, int], str] = {}
    selection = SeedSelection(main=(), dev=tuple(seeds))
    for name in (list(conditions) if instance_conditions is None else list(instance_conditions)):
        for outcome in build_deployment_instances(dataset, name, selection=selection, decoys=dict(decoys),
                                                  split_roles=("dev",), batch_size=256):
            ids[(name, outcome.seed)] = outcome.instance_id
    build_replay(dataset, n=min(replay_n, rows["train"]))
    record = {"schema": SYNTHETIC_SCHEMA, "synthetic": True, "note": SYNTHETIC_NOTE, "dataset": dataset,
              "config": cfg.as_dict(), "rows": rows, "conditions": dict(conditions), "plant_condition": plant_condition,
              "plant": f"{FINAL_NORM_WEIGHT}: delta = -2 * weight on the first {PLANT_FLIP_FRACTION:g} of the dims",
              "models": {str(seed): model_id for seed, model_id in models.items()},
              "instances": {f"{name}|{seed}": iid for (name, seed), iid in ids.items()}}
    (root / "synthetic_store.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return SyntheticStore(root=root, dataset=dataset, models=models, ids=ids, plant_condition=plant_condition,
                          record=record)


__all__ = ["ARCH", "CLASS_NAMES_10", "CLASS_TOKEN", "FINAL_NORM_WEIGHT", "HEAD_BIAS", "HEAD_WEIGHT", "PATCH_BIAS",
           "PATCH_WEIGHT", "POS_EMBED", "SPEC_SCHEMA", "SYNTHETIC_NOTE", "SYNTHETIC_SCHEMA", "SYNTH_MEAN", "SYNTH_STD",
           "TINY", "VIT_LITE_7_4", "SyntheticStore", "VitConfig", "block_names", "build_model_spec", "build_store",
           "init_params", "layernorm_tensors", "model_id_for", "param_shapes", "plant_deltas", "planted_labels",
           "render_forward", "stand_in_transform", "torch_logits", "write_export"]
