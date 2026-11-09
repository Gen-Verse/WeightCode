"""Block-level internals of the ViT-Lite numpy forward, for the mechanism readouts M05 and T07 (no labels needed).

``trace`` runs the instance's own forward.py functions (``_checked``, ``_patches``, ``_linear``, ``_layernorm``,
``_attention``, ``_gelu`` and its constants, taken from the module globals of the loaded ``forward`` function) in
forward.py's order, so on a chunk that forward.py processes as one inner chunk (at most its ``CHUNK`` rows) the logits
equal ``forward(params, x)``; callers check this on every network (``parity``; bitwise on the E11 students and on the
synthetic template of ``weightcode.models.vit_synth``, whose final LayerNorm runs on all tokens before the class row is
taken, per row the same arithmetic).

Two gate families are read, both from a final's weights and its program's printed state:

* the gaussian final's ramp gates (sc918e11dceef637a, gate3_lower.py; ``scripts/e11/e11_mech.py`` A2): per gate two
  units j1, j2 of one block's MLP with the same linear1 row, linear1 bias gap K = 3 and opposite linear2 columns; the
  gate output is GELU(u1) w2[:, j1] + GELU(u2) w2[:, j2]; "above threshold" is u1 > 0 and "fully open" u2 > 0; a
  patch gate acts on the patch tokens, a class gate on the class token. The program prints one line per gate
  (``gate patch block 2: ... units 503,360, ...``);
* the contrast final's gated correction (s760cb52ce82fb937, k145_gated_b2_r6_orig_targets.py;
  ``scripts/mech_v1/m03_m06.py``): 8 freed units of block 2, one of which (the gate unit) has linear1 row alpha * g and
  bias -alpha * theta, and shares that row with the kappa unit (bias larger by kappa = 2.5); "open" is the gate unit's
  pre-activation > 0; the correction is the 8 units' summed output.

Offsets at a block's output (the residual stream after the MLP) compare a condition image with the clean image of the
same row (paired by idx): in the base network, within the edited network, and the edited network's condition image
against the base network's clean image (the representation a correction aims at).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

REQUIRED_NAMES = ("_checked", "_patches", "_linear", "_layernorm", "_attention", "_gelu", "MEAN", "STD", "DIM", "DEPTH",
                  "CHUNK", "forward")
PATCH_WEIGHT = "tokenizer.conv_layers.0.0.weight"
PATCH_BIAS = "tokenizer.conv_layers.0.0.bias"
"""The patch projection's tensor names (``expected_shapes`` of every ViT-Lite forward.py)."""
GAUSSIAN_K_RAMP = 3.0
"""The gaussian program's ramp height K (``scripts/e11/e11_mech.py`` K_RAMP)."""
CONTRAST_KAPPA = 2.5
"""The contrast program's bias offset of the kappa unit (``scripts/mech_v1/m03_m06.py`` KAPPA)."""
CONTRAST_BLOCK = 2
ROW_ATOL = 1e-6
BIAS_ATOL = 1e-4
PRINT_ROUND = 5e-4 + 1e-5
"""The contrast program prints alpha and theta to 3 decimals."""
GATE_LINE = re.compile(r"^gate (patch|cls) +block (\d+): .*?units (\d+),(\d+),", re.MULTILINE)


class InternalError(RuntimeError):
    """A forward module, a gate or a statistic does not match what the readout assumes."""


# ----------------------------------------------------------------------------------------------------------------
# The trace


def forward_namespace(forward: Callable[..., Any]) -> Mapping[str, Any]:
    """The module globals of a loaded ViT-Lite forward.py (``weightcode.evalkit.patch_check.load_forward``)."""
    namespace = getattr(forward, "__globals__", None)
    if not isinstance(namespace, Mapping):
        raise InternalError("forward has no module globals")
    missing = [name for name in REQUIRED_NAMES if name not in namespace]
    if missing:
        raise InternalError(f"forward.py does not define {missing}; not the ViT-Lite forward")
    return namespace


def block_name(index: int) -> str:
    return f"classifier.blocks.{index}"


@dataclass
class Trace:
    """One chunk through one network: float32 logits, the tokenizer output (patch projection before the class token
    and positions), and per requested block the MLP input (the norm1 output) and the block output."""

    logits: np.ndarray
    tokens: np.ndarray
    mlp_in: dict[int, np.ndarray] = field(default_factory=dict)
    block_out: dict[int, np.ndarray] = field(default_factory=dict)


def trace(namespace: Mapping[str, Any], params: Mapping[str, np.ndarray], x: np.ndarray, blocks: Sequence[int]
          ) -> Trace:
    """forward.py's ``_features`` and head on ``x`` (uint8 NCHW), keeping the internals of ``blocks``."""
    ns = namespace
    x = np.asarray(x)
    if x.dtype != np.uint8 or x.ndim != 4:
        raise InternalError(f"images must be uint8 NCHW, got {x.dtype}{x.shape}")
    if len(x) > int(ns["CHUNK"]):
        raise InternalError(f"trace takes at most CHUNK = {ns['CHUNK']} rows at once, got {len(x)}")
    tensors = ns["_checked"](dict(params))
    dim = int(ns["DIM"])
    h = x.astype(np.float64) / 255.0
    h = (h - ns["MEAN"].reshape(1, -1, 1, 1)) / ns["STD"].reshape(1, -1, 1, 1)
    tokens = ns["_linear"](ns["_patches"](h), tensors[PATCH_WEIGHT].reshape(dim, -1), tensors[PATCH_BIAS])
    cls = np.broadcast_to(tensors["classifier.class_emb"], (x.shape[0], 1, dim))
    h = np.concatenate([cls, tokens], axis=1) + tensors["classifier.positional_emb"]
    keep = {int(b) for b in blocks}
    out = Trace(logits=np.empty(0), tokens=tokens)
    for index in range(int(ns["DEPTH"])):
        block = block_name(index)
        h = h + ns["_attention"](ns["_layernorm"](h, tensors[f"{block}.pre_norm.weight"],
                                                  tensors[f"{block}.pre_norm.bias"]), tensors, block)
        h = ns["_layernorm"](h, tensors[f"{block}.norm1.weight"], tensors[f"{block}.norm1.bias"])
        if index in keep:
            out.mlp_in[index] = h
        hidden = ns["_gelu"](ns["_linear"](h, tensors[f"{block}.linear1.weight"], tensors[f"{block}.linear1.bias"]))
        h = h + ns["_linear"](hidden, tensors[f"{block}.linear2.weight"], tensors[f"{block}.linear2.bias"])
        if index in keep:
            out.block_out[index] = h
    f = ns["_layernorm"](h[:, 0], tensors["classifier.norm.weight"], tensors["classifier.norm.bias"])
    out.logits = (f @ tensors["classifier.fc.weight"].T + tensors["classifier.fc.bias"]).astype(np.float32)
    return out


def parity(namespace: Mapping[str, Any], params: Mapping[str, np.ndarray], x: np.ndarray, logits: np.ndarray,
           atol: float = 1e-6) -> float:
    """Largest |trace logits - forward(params, x)| on one chunk; raises above ``atol``."""
    reference = np.asarray(namespace["forward"](dict(params), x), np.float64)
    diff = float(np.max(np.abs(np.asarray(logits, np.float64) - reference))) if len(x) else 0.0
    if diff > atol:
        raise InternalError(f"trace logits differ from forward.py by {diff} (> {atol})")
    return diff


# ----------------------------------------------------------------------------------------------------------------
# Gates from weights and printed program state


@dataclass(frozen=True)
class RampGate:
    """One gaussian ramp gate: ``kind`` patch or cls, the block, units j1 (GELU(u)) and j2 (GELU(u - K))."""

    kind: str
    block: int
    j1: int
    j2: int


def parse_gaussian_gates(stdout: str) -> list[RampGate]:
    """The gates the gaussian program printed, in print order (``gate <kind> block <b>: ... units <j1>,<j2>, ...``)."""
    gates = [RampGate(kind, int(block), int(j1), int(j2)) for kind, block, j1, j2 in GATE_LINE.findall(stdout)]
    if not gates:
        raise InternalError("the program's stdout prints no gate line")
    if sum(g.kind == "patch" for g in gates) != 1:
        raise InternalError(f"expected exactly one patch gate, got {gates}")
    return gates


def check_ramp_gates(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], gates: Sequence[RampGate],
                     k_ramp: float = GAUSSIAN_K_RAMP) -> list[dict[str, Any]]:
    """Each gate's form in the child (same linear1 row, bias gap K, opposite linear2 columns), and per gate block the
    changed linear1 rows equal the union of that block's gate units. Returns per gate kappa, threshold and offset."""
    by_block: dict[int, set[int]] = {}
    for g in gates:
        by_block.setdefault(g.block, set()).update((g.j1, g.j2))
    for block, units in by_block.items():
        w1 = np.asarray(child[f"{block_name(block)}.linear1.weight"])
        w1_0 = np.asarray(base[f"{block_name(block)}.linear1.weight"])
        rows = set(int(r) for r in np.nonzero(np.any(w1 != w1_0, axis=1))[0])
        if rows != units:
            raise InternalError(f"block {block}: changed linear1 rows {sorted(rows)}, printed gate units {sorted(units)}")
    table = []
    for g in gates:
        prefix = block_name(g.block)
        w1 = np.asarray(child[f"{prefix}.linear1.weight"], np.float64)
        b1 = np.asarray(child[f"{prefix}.linear1.bias"], np.float64)
        w2 = np.asarray(child[f"{prefix}.linear2.weight"], np.float64)
        if not np.allclose(w1[g.j1], w1[g.j2], rtol=0, atol=ROW_ATOL):
            raise InternalError(f"{g}: linear1 rows of j1 and j2 differ")
        if not math.isclose(b1[g.j1] - b1[g.j2], k_ramp, abs_tol=BIAS_ATOL):
            raise InternalError(f"{g}: bias gap {b1[g.j1] - b1[g.j2]}, expected K {k_ramp}")
        if not np.allclose(w2[:, g.j1], -w2[:, g.j2], rtol=0, atol=ROW_ATOL):
            raise InternalError(f"{g}: linear2 columns of j1 and j2 are not opposite")
        kappa = float(np.linalg.norm(w1[g.j1]))
        offset = k_ramp * w2[:, g.j2]
        table.append({"kind": g.kind, "block": g.block, "j1": g.j1, "j2": g.j2, "kappa": kappa,
                      "threshold_dz": float(-b1[g.j1] / kappa), "offset_norm": float(np.linalg.norm(offset))})
    return table


def printed_units(stdout: str) -> list[int]:
    """The ``units=[...]`` list the contrast program printed, in program order (exactly one such list)."""
    found = re.findall(r"units=\[([0-9, ]*)\]", stdout)
    if len(found) != 1:
        raise InternalError(f"expected one units=[...] list in the program's stdout, found {len(found)}")
    return [int(value) for value in found[0].split(",") if value.strip()]


def printed_alpha_theta(stdout: str) -> tuple[float, float]:
    found = re.findall(r"alpha=([-0-9.]+) theta=([-0-9.]+)", stdout)
    if len(found) != 1:
        raise InternalError(f"expected one 'alpha=... theta=...' in the program's stdout, found {len(found)}")
    return float(found[0][0]), float(found[0][1])


def contrast_gate(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray], stdout: str, rank: int = 6,
                  kappa: float = CONTRAST_KAPPA, block: int = CONTRAST_BLOCK) -> dict[str, Any]:
    """The contrast program's 8 units and gate unit in the child, checked against its printed state.

    The units are the changed linear1 rows of the block and must equal the printed list. Exactly one pair of them shares
    a linear1 row; the gate is its member with the lower bias, the bias gap is kappa, the gate is units[rank + 1] of the
    printed list and the kappa unit units[rank], and alpha (the row's norm) and theta (-bias / alpha) match the printed
    values to their rounding (the definition of ``m03_m06.identify_gate`` and ``check_gate_against_stdout``)."""
    prefix = block_name(block)
    w1 = np.asarray(child[f"{prefix}.linear1.weight"], np.float64)
    b1 = np.asarray(child[f"{prefix}.linear1.bias"], np.float64)
    w1_0 = np.asarray(base[f"{prefix}.linear1.weight"])
    units = sorted(int(r) for r in np.nonzero(np.any(np.asarray(child[f"{prefix}.linear1.weight"]) != w1_0,
                                                     axis=1))[0])
    printed = printed_units(stdout)
    if sorted(printed) != units or len(units) != rank + 2:
        raise InternalError(f"changed rows {units}, printed units {printed}, expected {rank + 2}")
    pairs = [(a, b) for i, a in enumerate(units) for b in units[i + 1:]
             if np.allclose(w1[a], w1[b], rtol=0.0, atol=ROW_ATOL)]
    if len(pairs) != 1:
        raise InternalError(f"expected exactly one pair of units with the same linear1 row, found {pairs}")
    a, b = pairs[0]
    gate, kappa_unit = (a, b) if b1[a] < b1[b] else (b, a)
    gap = float(b1[kappa_unit] - b1[gate])
    if not math.isclose(gap, kappa, abs_tol=BIAS_ATOL):
        raise InternalError(f"units {gate}, {kappa_unit}: bias gap {gap}, expected kappa {kappa}")
    alpha = float(np.linalg.norm(w1[gate]))
    theta = -float(b1[gate]) / alpha
    alpha_p, theta_p = printed_alpha_theta(stdout)
    checks = {"gate_is_units[rank+1]": printed[rank + 1] == gate, "kappa_unit_is_units[rank]": printed[rank] == kappa_unit,
              "alpha_matches_print": abs(alpha - alpha_p) <= PRINT_ROUND,
              "theta_matches_print": abs(theta - theta_p) <= PRINT_ROUND}
    if not all(checks.values()):
        raise InternalError(f"gate {gate} (alpha {alpha}, theta {theta}) does not match the printed state "
                            f"{printed} alpha {alpha_p} theta {theta_p}: {checks}")
    return {"block": block, "units": units, "printed_units": printed, "gate_unit": gate, "kappa_unit": kappa_unit,
            "bias_gap": gap, "alpha": alpha, "theta": theta, "checks": checks}


# ----------------------------------------------------------------------------------------------------------------
# Accumulated statistics (sums over chunks; ``finish`` divides)


TOKEN_SLICES = {"cls": np.s_[:, 0], "patch": np.s_[:, 1:]}


class Sums:
    """Named running sums (scalars or arrays)."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def add(self, key: str, value: Any) -> None:
        value = np.asarray(value, np.float64)
        self.values[key] = self.values[key] + value if key in self.values else value.copy()

    def get(self, key: str) -> Any:
        if key not in self.values:
            raise InternalError(f"no sum {key!r}")
        return self.values[key]


def per_class_counts(mask_per_image: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Sum of a per-image quantity by class."""
    return np.bincount(np.asarray(y, np.int64), weights=np.asarray(mask_per_image, np.float64), minlength=n_classes)


def ramp_gate_sums(sums: Sums, prefix: str, mlp_in: np.ndarray, child: Mapping[str, np.ndarray], gate: RampGate,
                   gelu: Callable[[np.ndarray], np.ndarray], y: np.ndarray | None = None, n_classes: int = 0,
                   k_ramp: float = GAUSSIAN_K_RAMP) -> None:
    """A2's selectivity sums of one ramp gate on one chunk (``mlp_in`` is the edited network's norm1 output)."""
    p = block_name(gate.block)
    w1 = np.asarray(child[f"{p}.linear1.weight"], np.float64)
    b1 = np.asarray(child[f"{p}.linear1.bias"], np.float64)
    w2 = np.asarray(child[f"{p}.linear2.weight"], np.float64)
    u1 = mlp_in @ w1[gate.j1] + b1[gate.j1]
    u2 = mlp_in @ w1[gate.j2] + b1[gate.j2]
    out = gelu(u1)[..., None] * w2[:, gate.j1] + gelu(u2)[..., None] * w2[:, gate.j2]
    norm = np.linalg.norm(out, axis=-1)
    opening = (gelu(u1) - gelu(u2)) / k_ramp
    for tok, sl in TOKEN_SLICES.items():
        sums.add(f"{prefix}{tok}_n", np.size(u1[sl]))
        sums.add(f"{prefix}{tok}_above", np.sum(u1[sl] > 0.0))
        sums.add(f"{prefix}{tok}_full", np.sum(u2[sl] > 0.0))
        sums.add(f"{prefix}{tok}_opening", np.sum(opening[sl]))
        sums.add(f"{prefix}{tok}_out_norm", np.sum(norm[sl]))
    sums.add(f"{prefix}img_any_patch", np.sum(np.any(u1[:, 1:] > 0.0, axis=1)))
    sums.add(f"{prefix}images", len(u1))
    if y is not None:
        gated = (u1[:, 1:] > 0.0).mean(axis=1) if gate.kind == "patch" else (u1[:, 0] > 0.0).astype(np.float64)
        sums.add(f"{prefix}class_rate_sum", per_class_counts(gated, y, n_classes))
        sums.add(f"{prefix}class_images", per_class_counts(np.ones(len(y)), y, n_classes))


def ramp_gate_finish(sums: Sums, prefix: str, with_classes: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tok in TOKEN_SLICES:
        count = float(sums.get(f"{prefix}{tok}_n"))
        result[tok] = {"n_tokens": int(count), "frac_above_threshold": float(sums.get(f"{prefix}{tok}_above")) / count,
                       "frac_fully_open": float(sums.get(f"{prefix}{tok}_full")) / count,
                       "mean_opening": float(sums.get(f"{prefix}{tok}_opening")) / count,
                       "mean_gate_output_norm": float(sums.get(f"{prefix}{tok}_out_norm")) / count}
    images = float(sums.get(f"{prefix}images"))
    result["n_images"] = int(images)
    result["images_any_patch_above"] = float(sums.get(f"{prefix}img_any_patch")) / images
    if with_classes:
        rate, count = sums.get(f"{prefix}class_rate_sum"), sums.get(f"{prefix}class_images")
        result["per_class_gated_rate"] = [float(r / c) if c else None for r, c in zip(rate, count)]
        result["per_class_images"] = [int(c) for c in count]
    return result


def contrast_gate_sums(sums: Sums, prefix: str, mlp_in: np.ndarray, child: Mapping[str, np.ndarray],
                       units: Sequence[int], gate_unit: int, gelu: Callable[[np.ndarray], np.ndarray],
                       y: np.ndarray | None = None, n_classes: int = 0, block: int = CONTRAST_BLOCK) -> None:
    """M03's sums on one chunk: the gate unit open (pre-activation > 0) on patch and class tokens, per image any and
    majority, and the norm of the units' summed output (``m03_m06.internal_stats``)."""
    p = block_name(block)
    w1 = np.asarray(child[f"{p}.linear1.weight"], np.float64)
    b1 = np.asarray(child[f"{p}.linear1.bias"], np.float64)
    w2 = np.asarray(child[f"{p}.linear2.weight"], np.float64)
    index = np.asarray(sorted(int(u) for u in units), dtype=np.int64)
    pre = mlp_in @ w1[index].T + b1[index]
    norm = np.linalg.norm(gelu(pre) @ w2[:, index].T, axis=-1)
    opened = (mlp_in @ w1[gate_unit] + b1[gate_unit]) > 0.0
    sums.add(f"{prefix}patch_open", np.sum(opened[:, 1:]))
    sums.add(f"{prefix}cls_open", np.sum(opened[:, 0]))
    sums.add(f"{prefix}patch_n", np.size(opened[:, 1:]))
    sums.add(f"{prefix}images", len(opened))
    sums.add(f"{prefix}img_any", np.sum(np.any(opened[:, 1:], axis=1)))
    sums.add(f"{prefix}img_majority", np.sum(opened[:, 1:].mean(axis=1) > 0.5))
    sums.add(f"{prefix}norm_patch", np.sum(norm[:, 1:]))
    sums.add(f"{prefix}norm_cls", np.sum(norm[:, 0]))
    if y is not None:
        sums.add(f"{prefix}class_rate_sum", per_class_counts(opened[:, 1:].mean(axis=1), y, n_classes))
        sums.add(f"{prefix}class_images", per_class_counts(np.ones(len(y)), y, n_classes))


def contrast_gate_finish(sums: Sums, prefix: str, with_classes: bool = False) -> dict[str, Any]:
    n = float(sums.get(f"{prefix}images"))
    patch = float(sums.get(f"{prefix}patch_n"))
    result: dict[str, Any] = {
        "n_images": int(n), "n_patch_tokens": int(patch),
        "open_patch_tokens": int(sums.get(f"{prefix}patch_open")),
        "rate_patch_tokens": float(sums.get(f"{prefix}patch_open")) / patch,
        "rate_cls_token": float(sums.get(f"{prefix}cls_open")) / n,
        "share_images_any_patch_token_open": float(sums.get(f"{prefix}img_any")) / n,
        "share_images_majority_patch_tokens_open": float(sums.get(f"{prefix}img_majority")) / n,
        "units_output_norm_patch_token_mean": float(sums.get(f"{prefix}norm_patch")) / patch,
        "units_output_norm_cls_token_mean": float(sums.get(f"{prefix}norm_cls")) / n,
    }
    if with_classes:
        rate, count = sums.get(f"{prefix}class_rate_sum"), sums.get(f"{prefix}class_images")
        result["per_class_patch_open_rate"] = [float(r / c) if c else None for r, c in zip(rate, count)]
        result["per_class_images"] = [int(c) for c in count]
    return result


OFFSET_PAIRS = {
    "base": ("base_cond", "base_clean"),
    "edited_within": ("edit_cond", "edit_clean"),
    "edited_vs_base_clean": ("edit_cond", "base_clean"),
    "edit_move_cond": ("edit_cond", "base_cond"),
    "edit_move_clean": ("edit_clean", "base_clean"),
}
"""Offsets at a block's output: (minuend, subtrahend) of the four traces, paired per row."""


def offset_sums(sums: Sums, prefix: str, outs: Mapping[str, np.ndarray]) -> None:
    """Per pair of ``OFFSET_PAIRS`` and token type: the summed difference vector and the summed per-token norm."""
    for name, (a, b) in OFFSET_PAIRS.items():
        diff = np.asarray(outs[a], np.float64) - np.asarray(outs[b], np.float64)
        for tok, sl in TOKEN_SLICES.items():
            part = diff[sl]
            sums.add(f"{prefix}{name}_{tok}_vec", part.reshape(-1, part.shape[-1]).sum(axis=0))
            sums.add(f"{prefix}{name}_{tok}_norm", np.linalg.norm(part, axis=-1).sum())
            sums.add(f"{prefix}{name}_{tok}_n", part.size // part.shape[-1])


def offset_finish(sums: Sums, prefix: str) -> dict[str, Any]:
    """Per pair and token type: the norm of the mean difference (``mean_offset_norm``, A2's offset) and the mean
    per-token norm (``mean_paired_norm``)."""
    result: dict[str, Any] = {}
    for name in OFFSET_PAIRS:
        for tok in TOKEN_SLICES:
            n = float(sums.get(f"{prefix}{name}_{tok}_n"))
            result.setdefault(name, {})[tok] = {
                "mean_offset_norm": float(np.linalg.norm(sums.get(f"{prefix}{name}_{tok}_vec") / n)),
                "mean_paired_norm": float(sums.get(f"{prefix}{name}_{tok}_norm")) / n, "n_tokens": int(n)}
    for tok in TOKEN_SLICES:
        base = result["base"][tok]["mean_offset_norm"]
        result.setdefault("residual_share", {})[tok] = {
            "edited_vs_base_clean": result["edited_vs_base_clean"][tok]["mean_offset_norm"] / base if base else None,
            "edited_within": result["edited_within"][tok]["mean_offset_norm"] / base if base else None}
    return result


def token_spread(tokens: np.ndarray) -> np.ndarray:
    """Per image, sqrt(mean over channels of the variance over the image's tokens) (``m03_m06.token_spread`` (a))."""
    t = np.asarray(tokens, np.float64)
    return np.sqrt(t.var(axis=1).mean(axis=1))


def ratio_summary(num: np.ndarray, den: np.ndarray) -> dict[str, float]:
    """``num`` over ``den``, paired per image (``m03_m06.ratio_summary``)."""
    ratio = np.asarray(num, np.float64) / np.asarray(den, np.float64)
    return {"mean_of_image_ratios": float(ratio.mean()), "median_of_image_ratios": float(np.median(ratio)),
            "ratio_of_means": float(np.mean(num) / np.mean(den)), "mean_numerator": float(np.mean(num)),
            "mean_denominator": float(np.mean(den))}


# ----------------------------------------------------------------------------------------------------------------
# Descriptive association


def ranks(values: Sequence[float]) -> np.ndarray:
    """Ranks 1..n with ties given their average rank."""
    v = np.asarray(values, np.float64)
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v), np.float64)
    r[order] = np.arange(1, len(v) + 1, dtype=np.float64)
    for value in np.unique(v):
        tied = v == value
        r[tied] = r[tied].mean()
    return r


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman's rho (Pearson correlation of average ranks); None when either side is constant or n < 3."""
    if len(a) != len(b):
        raise InternalError("spearman needs paired sequences")
    if len(a) < 3:
        return None
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return None if den == 0 else float(ra @ rb) / den


__all__ = ["CONTRAST_BLOCK", "CONTRAST_KAPPA", "GAUSSIAN_K_RAMP", "InternalError", "OFFSET_PAIRS", "RampGate", "Sums",
           "Trace", "check_ramp_gates", "contrast_gate", "contrast_gate_finish", "contrast_gate_sums",
           "forward_namespace", "offset_finish", "offset_sums", "parity", "parse_gaussian_gates", "per_class_counts",
           "printed_alpha_theta", "printed_units", "ramp_gate_finish", "ramp_gate_sums", "ranks", "ratio_summary",
           "spearman", "token_spread", "trace"]
