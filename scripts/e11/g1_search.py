"""PREREG G1: a frozen generic gated-correction search (no LLM), with the realization amendment of main 5bd3817.

Spec: docs/PREREG_v0_DRAFT.md, entry "G1, a frozen generic gated-correction search" and its realization amendment.
Dev only, E11 seed-100 student, one condition per run (``--instance i00802425`` gaussian_noise_s5, ``i3eb3fc78``
contrast_s5). Information: the session's experience (64 condition rows x_new, 64 clean rows x_old, with labels) and the
student. E10 program limits, joint-retain-v1 (S_old >= -20 of 4000), at most 8 checks on V_check, the pick read on
V_select. No sealed data.

Stages (each writes ``$WEIGHTCODE_RUNS/devreadouts/g1/g1_<condition>_<stage>.jsonl`` with a .sha256, never overwritten):

* ``selftest``: realization identities and limit feasibility of the builders. Reads no label and no V_ split.
* ``singles [--shard i/n]`` then ``merge``: every member of the family at every strength: built on the student, checked against the E10 limits,
  scored by the proxy (the joint count on the 128 experience rows: condition rows fixed minus clean rows broken).
* ``search``: greedy compositions from the top 5 singles, the 8 best distinct candidates within the limits sent to the
  checks (``execute_run`` joint=True e10=True), the joint-retain-v1 pick read on V_select.
* ``ceiling [--shard i/n]``: every single member within the limits read on V_select (descriptive, selects nothing).
* ``summary``: the tables and the reading fixed in the entry.

The family (members, as built):

* Sites: the tokenizer output and the output of each of the 7 blocks.
* Corrections, from the site's tokens on x_new against x_old: c1 mean offset, c2 per-channel moment matching, c3 rank-r
  covariance alignment on the top r directions (by |eigenvalue|) of the covariance difference, r in {4, 8}.
* Tokenizer (patch tokens): c1, c2, c3 exactly, through the convolution's weight and bias.
* Blocks: c1 ungated (linear2.bias) or gated by k in {2, 6} least-used units (k/2 ramp pairs with the same whitened
  mean-difference direction and thresholds staggered by one ramp width, each pair carrying 1/(k/2) of the offset;
  the first threshold is the 99th percentile of the x_old scores, so the ramp opens for at most 1% of x_old tokens).
  c2 is not used at blocks. c3 uses 2r least-used units as r unit pairs GELU(s + l_i) - GELU(s - l_i) = 2 l_i
  (l_i: the rank-r least-squares fit, over the 128 experience rows' tokens, from LN(h) to the c3 target correction of h);
  ungated s is a large constant, gated s adds the whitened score with the same 1% start. R^2 recorded per member.
* Tokens (blocks): patch, class, both; they set the tokens whose statistics define the offset, the direction, the
  thresholds and the c3 fit. The edit acts on every token it fires on.
* Strength {0.5, 1.0}, plus the largest strength within the E10 limits when 1.0 exceeds them.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

THREADS = 4
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = str(THREADS)  # before numpy is imported, or BLAS ignores it

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("e11_mech_g1", HERE / "e11_mech.py")
mech = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mech
_spec.loader.exec_module(mech)

SMOKE = int(os.environ.get("G1_SMOKE", "0"))
"""G1_SMOKE=N: the first N members only, records under devreadouts/g1_smoke (never the record of record). A smoke run
exercises every stage on the real inputs; its records are not read for results."""
SCHEMA = "weightcode.g1.v1"
SPEC = "docs/PREREG_v0_DRAFT.md G1 entry and its realization amendment (main 5bd3817)"
TOP_SINGLES = 5
MAX_MEMBERS = 3
MAX_CHECKS = 8
FLOOR = -20
GATE_QUANTILE = 0.99
K_RAMP = 3.0
W_SD = 1.0
FROB_MAX = 0.45
SHRINK = 0.1
OPEN_MARGIN = 8.0
RANKS = (4, 8)
GATE_K = (2, 6)
TOKENS = ("patch", "cls", "both")
STRENGTHS = (0.5, 1.0)
READING_FRACTION = 0.75
READING_CLEAN_FLOOR = -1.0
BARS = {"i00802425": 23.81, "i3eb3fc78": 13.49}
CONDITION = {"i00802425": "gaussian_noise_s5", "i3eb3fc78": "contrast_s5"}


class G1Error(RuntimeError):
    """A member cannot be built as specified or an input does not match its record."""


# --------------------------------------------------------------------------------------------------------------------
# Context


@dataclasses.dataclass
class Ctx:
    inst: Any
    base: dict[str, np.ndarray]
    mod: Any
    vit: Any
    limits: Any
    x_new: np.ndarray
    y_new: np.ndarray
    x_old: np.ndarray
    y_old: np.ndarray
    base_pred_new: np.ndarray
    base_pred_old: np.ndarray
    depth: int
    dim: int
    patch_w: str
    patch_b: str


def make_ctx(instance_id: str) -> Ctx:
    mech._TARGET_ID = instance_id
    mech._INSTANCE = None
    inst = mech.instance()
    mod = mech._vit_module()
    layout = {k: v for k, v in json.loads((inst.instance_dir / "model_spec.json").read_text()).items()
              if k != "model_id"}
    base = {k: np.asarray(v) for k, v in inst.params.items()}
    vit = mod._ViT(layout, base)
    with np.load(mech.target_session_dir() / "trial" / "experience.npz", allow_pickle=False) as a:
        x_new, y_new, x_old, y_old = (np.array(a[k]) for k in ("x_new", "y_new", "x_old", "y_old"))
    ref = np.asarray(inst.forward(dict(base), x_old[:16]), np.float64)
    if np.max(np.abs(ref - vit.run(base, x_old[:16])[0])) > 1e-3:
        raise G1Error("the program's forward differs from forward.py")
    pn = vit.run(base, x_new)[0].argmax(1)
    po = vit.run(base, x_old)[0].argmax(1)
    return Ctx(inst=inst, base=base, mod=mod, vit=vit, limits=mech.program_limits(inst), x_new=x_new, y_new=y_new,
               x_old=x_old, y_old=y_old, base_pred_new=pn, base_pred_old=po, depth=len(vit.blocks),
               dim=int(np.asarray(base[vit.patch["weight"]]).shape[0]), patch_w=vit.patch["weight"],
               patch_b=vit.patch["bias"])


@dataclasses.dataclass
class Acts:
    h_new: dict[int, np.ndarray]
    h_old: dict[int, np.ndarray]
    z_new: dict[int, np.ndarray]
    z_old: dict[int, np.ndarray]


_ACTS_CACHE: dict[str, Acts] = {}


def acts(ctx: Ctx, params: Mapping[str, np.ndarray]) -> Acts:
    """Block outputs h and MLP inputs z of every block on x_new and x_old. The base student's are computed once."""
    if params is ctx.base:
        if "base" not in _ACTS_CACHE:
            _ACTS_CACHE["base"] = acts_uncached(ctx, params)
        return _ACTS_CACHE["base"]
    return acts_uncached(ctx, params)


def acts_uncached(ctx: Ctx, params: Mapping[str, np.ndarray]) -> Acts:
    every = set(range(ctx.depth))
    _, zn, hn = ctx.vit.run(params, ctx.x_new, want_mid=every, want_out=True)
    _, zo, ho = ctx.vit.run(params, ctx.x_old, want_mid=every, want_out=True)
    return Acts(hn, ho, zn, zo)


def select(a: np.ndarray, kind: str) -> np.ndarray:
    """Tokens of kind patch / cls / both of an [images, 65, D] array, as [rows, D]."""
    if kind == "patch":
        return a[:, 1:].reshape(-1, a.shape[-1])
    if kind == "cls":
        return a[:, 0]
    if kind == "both":
        return a.reshape(-1, a.shape[-1])
    raise G1Error(f"unknown token kind {kind}")


def patch_tokens(ctx: Ctx, params: Mapping[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Patch vectors g [rows, C*P*P] and the patch projection's output t = g W^T + b [rows, D] (before class token and
    positions), the same arithmetic as ``_ViT.tokens``."""
    vit = ctx.vit
    h = x.astype(np.float64) / 255.0
    h = (h - vit.mean.reshape(1, -1, 1, 1)) / vit.std.reshape(1, -1, 1, 1)
    n, C, H, W = h.shape
    P = vit.P
    g = h.reshape(n, C, H // P, P, W // P, P).transpose(0, 2, 4, 1, 3, 5).reshape(n * (H // P) * (W // P), C * P * P)
    wp = vit.T(params, ctx.patch_w).reshape(-1, C * P * P)
    return g, g @ wp.T + vit.T(params, ctx.patch_b)


def with_tensors(cur: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray], values: Mapping[str, np.ndarray]
                 ) -> dict[str, np.ndarray]:
    new = dict(cur)
    for name, value in values.items():
        new[name] = np.asarray(value).astype(np.asarray(base[name]).dtype)
    return new


# --------------------------------------------------------------------------------------------------------------------
# Tokenizer corrections (exact through the convolution's weight and bias)


def cov_difference_map(c_new: np.ndarray, c_old: np.ndarray, r: int) -> np.ndarray:
    """D = sum_i (a_i - 1) u_i u_i^T over the top-r eigendirections (by |eigenvalue|) of C_old - C_new, a_i the ratio of
    the old to the new standard deviation along u_i. t' = t + s D (t - mu_new) matches the variance along those r
    directions at s = 1 and leaves the orthogonal complement and the mean of x_new unchanged."""
    values, vectors = np.linalg.eigh(((c_old - c_new) + (c_old - c_new).T) / 2.0)
    top = np.argsort(-np.abs(values))[:r]
    u = vectors[:, top]
    var_new = np.einsum("di,de,ei->i", u, c_new, u)
    var_old = np.einsum("di,de,ei->i", u, c_old, u)
    a = np.sqrt(np.maximum(var_old, 1e-12) / np.maximum(var_new, 1e-12))
    return (u * (a - 1.0)) @ u.T


def build_tokenizer(ctx: Ctx, cur: Mapping[str, np.ndarray], corr: str, r: int | None, s: float
                    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _, tn = patch_tokens(ctx, cur, ctx.x_new)
    _, to = patch_tokens(ctx, cur, ctx.x_old)
    mu_new, mu_old = tn.mean(0), to.mean(0)
    w = ctx.vit.T(cur, ctx.patch_w)
    shape = w.shape
    wf, b = w.reshape(shape[0], -1), ctx.vit.T(cur, ctx.patch_b)
    if corr == "c1":
        nb, nw = b - s * (mu_new - mu_old), wf
    elif corr == "c2":
        a = np.where(tn.std(0) > 1e-12, to.std(0) / np.maximum(tn.std(0), 1e-12), 1.0)
        scale = 1.0 + s * (a - 1.0)
        nw, nb = wf * scale[:, None], scale * b + s * (mu_old - a * mu_new)
    elif corr == "c3":
        dm = cov_difference_map(np.cov(tn.T, bias=True), np.cov(to.T, bias=True), int(r))
        m = np.eye(shape[0]) + s * dm
        nw, nb = m @ wf, m @ b - s * dm @ mu_new
    else:
        raise G1Error(corr)
    new = with_tensors(cur, ctx.base, {ctx.patch_w: nw.reshape(shape), ctx.patch_b: nb})
    return new, {"tensors": [ctx.patch_w, ctx.patch_b]}


# --------------------------------------------------------------------------------------------------------------------
# Block corrections


def block_names(ctx: Ctx, b: int) -> tuple[str, str, str, str]:
    blk = ctx.vit.blocks[b]
    return (blk["linear1"]["weight"], blk["linear1"]["bias"], blk["linear2"]["weight"], blk["linear2"]["bias"])


def free_units(ctx: Ctx, cur: Mapping[str, np.ndarray], a: Acts, b: int, used: set[int], count: int
               ) -> tuple[list[int], np.ndarray]:
    """The ``count`` least-used units of block b (least mean |activation| over the 128 experience rows' tokens, not
    used by an earlier member), and every unit's mean activation (for the mean ablation)."""
    w1n, b1n, _, _ = block_names(ctx, b)
    z = np.concatenate([a.z_new[b].reshape(-1, ctx.dim), a.z_old[b].reshape(-1, ctx.dim)])
    act = ctx.mod._gelu(z @ ctx.vit.T(cur, w1n).T + ctx.vit.T(cur, b1n))
    order = [int(j) for j in np.argsort(np.abs(act).mean(0), kind="stable") if int(j) not in used]
    if len(order) < count:
        raise G1Error(f"block {b} has fewer than {count} free units")
    return order[:count], act.mean(0)


def gate_direction(ctx: Ctx, a: Acts, b: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Whitened mean-difference direction d (gate3_lower's ``_whitened_dir``, shrinkage 0.1) and the x_old scores used
    for the threshold. Class tokens: direction from the first half of the images, scores from the other half (as the
    program does); patch and both: all tokens."""
    zn, zo = a.z_new[b], a.z_old[b]
    if kind == "cls":
        half = len(zo) // 2
        d = ctx.mod._whitened_dir(zo[:half, 0], zn[:half, 0], SHRINK)
        return d, zo[half:, 0] @ d
    zc, zw = select(zo, kind), select(zn, kind)
    d = ctx.mod._whitened_dir(zc, zw, SHRINK)
    return d, zc @ d


def fit_budget(ctx: Ctx, w1: np.ndarray, w1n: str) -> float:
    """Relative Frobenius change of linear1 against the base student."""
    ref = ctx.vit.T(ctx.base, w1n)
    return float(np.linalg.norm(w1 - ref) / np.linalg.norm(ref))


def build_c1_block(ctx: Ctx, cur: Mapping[str, np.ndarray], b: int, kind: str, k: int, s: float,
                   used: dict[int, set[int]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    a = acts(ctx, cur)
    off = s * (select(a.h_new[b], kind).mean(0) - select(a.h_old[b], kind).mean(0))
    w1n, b1n, w2n, b2n = block_names(ctx, b)
    if k == 0:  # ungated: the offset is removed at every token through linear2.bias
        b2 = ctx.vit.T(cur, b2n) - off
        return with_tensors(cur, ctx.base, {b2n: b2}), {"tensors": [b2n], "offset_norm": float(np.linalg.norm(off))}
    pairs = k // 2
    d, scores = gate_direction(ctx, a, b, kind)
    sd = float(scores.std())
    if not (np.all(np.isfinite(d)) and np.isfinite(sd) and sd > 0):
        raise G1Error("degenerate statistics")
    t0 = float(np.quantile(scores, GATE_QUANTILE))
    width = W_SD * sd
    units, mean_act = free_units(ctx, cur, a, b, used.setdefault(b, set()), k)
    w1, b1 = ctx.vit.T(cur, w1n).copy(), ctx.vit.T(cur, b1n).copy()
    w2, b2 = ctx.vit.T(cur, w2n).copy(), ctx.vit.T(cur, b2n).copy()
    for j in units:  # mean-ablate the units' previous function
        b2 += w2[:, j] * mean_act[j]
    kappa = K_RAMP / width
    for _ in range(400):
        trial = w1.copy()
        for m in range(pairs):
            trial[units[2 * m]] = kappa * d
            trial[units[2 * m + 1]] = kappa * d
        if fit_budget(ctx, trial, w1n) <= FROB_MAX:
            break
        kappa *= 0.95
    else:
        raise G1Error("Frobenius budget cannot be met")
    w1 = trial
    for m in range(pairs):
        j1, j2 = units[2 * m], units[2 * m + 1]
        tm = t0 + m * width
        b1[j1], b1[j2] = -kappa * tm, -kappa * tm - K_RAMP
        w2[:, j1], w2[:, j2] = -off / (K_RAMP * pairs), off / (K_RAMP * pairs)
    used[b].update(units)
    new = with_tensors(cur, ctx.base, {w1n: w1, b1n: b1, w2n: w2, b2n: b2})
    open_old = float(np.mean(scores > t0))
    open_new = float(np.mean(select(a.z_new[b], kind) @ d > t0))
    return new, {"tensors": [w1n, b1n, w2n, b2n], "units": units, "kappa": kappa, "threshold": t0, "ramp_width": width,
                 "offset_norm": float(np.linalg.norm(off)), "open_old_scores": open_old, "open_new": open_new}


def build_c3_block(ctx: Ctx, cur: Mapping[str, np.ndarray], b: int, kind: str, r: int, gated: bool, s: float,
                   used: dict[int, set[int]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    a = acts(ctx, cur)
    hn, ho = select(a.h_new[b], kind), select(a.h_old[b], kind)
    zn, zo = select(a.z_new[b], kind), select(a.z_old[b], kind)
    dm = cov_difference_map(np.cov(hn.T, bias=True), np.cov(ho.T, bias=True), r)
    mu_new = hn.mean(0)
    h_all, z_all = np.concatenate([hn, ho]), np.concatenate([zn, zo])
    target = (h_all - mu_new) @ dm  # strength 1; the edit scales it by s
    za = np.concatenate([z_all, np.ones((len(z_all), 1))], axis=1)
    nn = len(hn)
    # PREREG G1 fit amendment (main bc1b219): a gated member fires only where its gate opens, so its map is fitted on
    # the selected x_new tokens only; an ungated member's map applies to every token, so it keeps the pooled fit.
    fit_rows = slice(0, nn) if gated else slice(None)
    za_fit = za[fit_rows]
    gram = za_fit.T @ za_fit
    coef = np.linalg.solve(gram + 1e-3 * np.trace(gram) / gram.shape[0] * np.eye(gram.shape[0]),
                           za_fit.T @ target[fit_rows])
    fitted = za @ coef
    _, _, vt = np.linalg.svd(fitted[fit_rows], full_matrices=False)
    v = vt[:r].T  # [D, r], orthonormal output directions
    fit_r = fitted @ v @ v.T
    var_target = float(np.sum((target - target.mean(0)) ** 2))
    r2 = 1.0 - float(np.sum((target - fit_r) ** 2)) / var_target if var_target > 0 else float("nan")
    bv = coef @ v  # [D + 1, r]: l_i(z) = z . bv[:D, i] + bv[D, i]
    l_all = za @ bv
    w1n, b1n, w2n, b2n = block_names(ctx, b)
    n1 = float(np.linalg.norm(ctx.vit.T(ctx.base, w1n)))
    n2 = float(np.linalg.norm(ctx.vit.T(ctx.base, w2n)))
    sum_b2 = float(np.sum(bv[: ctx.dim] ** 2))
    alpha = (s * s * r * n1 * n1 / (4.0 * max(sum_b2, 1e-30) * n2 * n2)) ** 0.25  # equal relative change in both layers
    w1, b1 = ctx.vit.T(cur, w1n).copy(), ctx.vit.T(cur, b1n).copy()
    w2, b2 = ctx.vit.T(cur, w2n).copy(), ctx.vit.T(cur, b2n).copy()
    units, mean_act = free_units(ctx, cur, a, b, used.setdefault(b, set()), 2 * r)
    for j in units:
        b2 += w2[:, j] * mean_act[j]
    big = alpha * float(np.max(np.abs(l_all))) + OPEN_MARGIN  # s - |alpha l| >= OPEN_MARGIN when open

    def r2_of(rows: slice) -> float:
        tv = float(np.sum((target[rows] - target[rows].mean(0)) ** 2))
        return 1.0 - float(np.sum((target[rows] - fit_r[rows]) ** 2)) / tv if tv > 0 else float("nan")

    info: dict[str, Any] = {"tensors": [w1n, b1n, w2n, b2n], "units": units, "fit_r2": r2, "alpha": alpha,
                            "fit_rows": "x_new tokens" if gated else "x_new and x_old tokens",
                            "fit_r2_x_new": r2_of(slice(0, nn)), "fit_r2_x_old": r2_of(slice(nn, None)),
                            "open_level": big}
    d = np.zeros(ctx.dim)
    kappa = 0.0
    if gated:
        d, scores = gate_direction(ctx, a, b, kind)
        sd = float(scores.std())
        if not (np.all(np.isfinite(d)) and np.isfinite(sd) and sd > 0):
            raise G1Error("degenerate statistics")
        t0 = float(np.quantile(scores, GATE_QUANTILE))
        kappa = 2.0 * big / (W_SD * sd)
    rows_pm = alpha * bv[: ctx.dim].T  # [r, D]
    for _ in range(400):
        trial = w1.copy()
        for i in range(r):
            trial[units[2 * i]] = rows_pm[i] + kappa * d
            trial[units[2 * i + 1]] = -rows_pm[i] + kappa * d
        if fit_budget(ctx, trial, w1n) <= FROB_MAX or not gated:
            break
        kappa *= 0.95
    else:
        raise G1Error("Frobenius budget cannot be met")
    w1 = trial
    for i in range(r):
        ja, jb = units[2 * i], units[2 * i + 1]
        base_bias = (-kappa * t0 - big) if gated else big
        b1[ja] = base_bias + alpha * bv[ctx.dim, i]
        b1[jb] = base_bias - alpha * bv[ctx.dim, i]
        w2[:, ja] = s * v[:, i] / (2.0 * alpha)
        w2[:, jb] = -s * v[:, i] / (2.0 * alpha)
    used[b].update(units)
    if gated:
        info.update({"kappa": kappa, "threshold": t0, "ramp_width": W_SD * sd,
                     "open_old_scores": float(np.mean(scores > t0))})
    return with_tensors(cur, ctx.base, {w1n: w1, b1n: b1, w2n: w2, b2n: b2}), info


# --------------------------------------------------------------------------------------------------------------------
# Members


@dataclasses.dataclass(frozen=True)
class Member:
    site: str  # "tok" or "b0".."b6"
    corr: str  # c1 | c2 | c3
    r: int | None = None
    gate: int = 0  # c1 blocks: k units (0 ungated); c3: 0 ungated, 1 gated
    tokens: str = "patch"

    @property
    def block(self) -> int | None:
        return None if self.site == "tok" else int(self.site[1:])

    @property
    def name(self) -> str:
        parts = [self.site, self.corr + (f"r{self.r}" if self.r else "")]
        if self.site != "tok":
            parts.append(("g" + str(self.gate) if self.corr == "c1" else ("gate" if self.gate else "open")))
            parts.append(self.tokens)
        return "_".join(parts)


def family(depth: int) -> list[Member]:
    """Every member, in the enumeration order that breaks proxy ties."""
    out = [Member("tok", "c1"), Member("tok", "c2"), Member("tok", "c3", 4), Member("tok", "c3", 8)]
    for b in range(depth):
        site = f"b{b}"
        for tokens in TOKENS:
            for k in (0, *GATE_K):
                out.append(Member(site, "c1", None, k, tokens))
            for r in RANKS:
                for gated in (0, 1):
                    out.append(Member(site, "c3", r, gated, tokens))
    return out[:SMOKE] if SMOKE else out


def apply_member(ctx: Ctx, cur: Mapping[str, np.ndarray], m: Member, s: float, used: dict[int, set[int]]
                 ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if m.site == "tok":
        return build_tokenizer(ctx, cur, m.corr, m.r, s)
    if m.corr == "c1":
        return build_c1_block(ctx, cur, m.block, m.tokens, m.gate, s, used)
    return build_c3_block(ctx, cur, m.block, m.tokens, int(m.r), bool(m.gate), s, used)


def limits_stats(ctx: Ctx, child: Mapping[str, np.ndarray]) -> dict[str, Any]:
    from weightcode.baselines.fit.edits import edit_stats_under

    return edit_stats_under(ctx.base, child, ctx.limits)


def try_build(ctx: Ctx, cur: Mapping[str, np.ndarray], m: Member, s: float, used: dict[int, set[int]] | None = None
              ) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Build member m at strength s on ``cur``. Returns (child or None, record); None when it cannot be built."""
    scratch = {k: set(v) for k, v in (used or {}).items()}
    try:
        child, info = apply_member(ctx, cur, m, s, scratch)
    except G1Error as error:
        return None, {"built": False, "error": str(error)}
    stats = limits_stats(ctx, child)
    info.update({"built": True, "within_limits": bool(stats["within_limits"]), "violations": stats["violations"],
                 "n_changed": stats["n_changed"], "changed_frac": stats["changed_frac"],
                 "per_tensor_rel_frob": {k: v["rel_frob"] for k, v in stats["per_tensor"].items() if v["changed"]}})
    if used is not None:
        used.clear()
        used.update(scratch)
    return child, info


def largest_feasible(ctx: Ctx, m: Member, hi: float, steps: int = 14) -> float | None:
    """The largest strength in (0, hi] within the E10 limits by bisection (the rel-Frobenius norm of every edit grows
    with the strength), or None when even a tiny strength is outside them."""
    child, info = try_build(ctx, ctx.base, m, 1e-3)
    if child is None or not info["within_limits"]:
        return None
    lo = 1e-3
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        child, info = try_build(ctx, ctx.base, m, mid)
        if child is not None and info["within_limits"]:
            lo = mid
        else:
            hi = mid
    return float(lo)


def params_sha(params: Mapping[str, np.ndarray]) -> str:
    from weightcode.common.identity import params_sha256

    return params_sha256(params)


def proxy(ctx: Ctx, child: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The joint count on the 128 experience rows: condition rows fixed minus clean rows broken (with the labels)."""
    pn = ctx.vit.run(child, ctx.x_new)[0].argmax(1)
    po = ctx.vit.run(child, ctx.x_old)[0].argmax(1)
    fixed = int(np.sum((ctx.base_pred_new != ctx.y_new) & (pn == ctx.y_new)))
    broken_new = int(np.sum((ctx.base_pred_new == ctx.y_new) & (pn != ctx.y_new)))
    broken = int(np.sum((ctx.base_pred_old == ctx.y_old) & (po != ctx.y_old)))
    fixed_old = int(np.sum((ctx.base_pred_old != ctx.y_old) & (po == ctx.y_old)))
    return {"fixed_new": fixed, "broken_new": broken_new, "broken_old": broken, "fixed_old": fixed_old,
            "score": fixed - broken}


# --------------------------------------------------------------------------------------------------------------------
# Records


def out_dir() -> Path:
    return mech.runs_root() / "devreadouts" / ("g1_smoke" if SMOKE else "g1")


def write_records(instance_id: str, stage: str, records: Sequence[Mapping[str, Any]]) -> Path:
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"g1_{CONDITION[instance_id]}_{stage}.jsonl"
    if path.exists():
        raise G1Error(f"{path} exists; records are never overwritten")
    from weightcode.baselines.fit.runner import jsonable

    path.write_text("".join(json.dumps(jsonable(r), sort_keys=True, allow_nan=False) + "\n" for r in records),
                    encoding="utf-8")
    (folder / f"{path.name}.sha256").write_text(f"{mech.file_sha256(path)}  {path.name}\n", encoding="utf-8")
    print(f"wrote {path} sha256 {mech.file_sha256(path)}", flush=True)
    return path


def read_records(instance_id: str, stage: str) -> list[dict[str, Any]]:
    path = out_dir() / f"g1_{CONDITION[instance_id]}_{stage}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def provenance(instance_id: str) -> dict[str, Any]:
    mech._TARGET_ID = instance_id
    return mech.provenance() | {"spec": SPEC}


# --------------------------------------------------------------------------------------------------------------------
# Stages


def cmd_singles(ctx: Ctx, instance_id: str, shard: tuple[int, int]) -> list[dict[str, Any]]:
    """Every member at every strength, scored by the proxy (the table the ceiling and the search read). Sharded: this
    process takes the members with order = index mod count; ``merge`` joins the shards into the table."""
    records: list[dict[str, Any]] = []
    index, count = shard
    for order, m in enumerate(family(ctx.depth)):
        if order % count != index:
            continue
        first_row = len(records)
        strengths = list(STRENGTHS)
        child1, info1 = try_build(ctx, ctx.base, m, 1.0)
        extra = None
        if child1 is None or not info1["within_limits"]:  # 1.0 exceeds the limits (or the kappa budget cannot be met)
            extra = largest_feasible(ctx, m, 1.0)
            if extra is not None:
                strengths.append(extra)
        for s in strengths:
            child, info = (child1, info1) if s == 1.0 else try_build(ctx, ctx.base, m, s)
            rec: dict[str, Any] = {"schema": SCHEMA, "stage": "singles", "instance_id": instance_id, "order": order,
                                   "member": dataclasses.asdict(m), "name": m.name, "strength": float(s),
                                   "strength_kind": "largest_feasible" if s == extra else "grid", **info}
            if child is not None and info["within_limits"]:
                rec["child_sha256"] = params_sha(child)
                rec["proxy"] = proxy(ctx, child)
            records.append(rec)
        feasible = any(r.get("built") and r.get("within_limits") for r in records[first_row:])
        for r in records[first_row:]:  # per member: no strength (0.5, 1.0, the largest feasible) is within the limits
            r["out_of_limits_at_every_strength"] = not feasible
        if order % 20 == 0:
            print(f"singles {order}/{len(family(ctx.depth))} {m.name}", flush=True)
    if records:
        records[0]["provenance"] = provenance(instance_id)
    write_records(instance_id, f"singles_{index}of{count}", records)
    return records


def cmd_merge(instance_id: str) -> list[dict[str, Any]]:
    """Join the singles shards into the table of record (``singles``), in member order, every shard present once."""
    shards = sorted(out_dir().glob(f"g1_{CONDITION[instance_id]}_singles_*of*.jsonl"))
    counts = {int(p.stem.rsplit("_", 1)[1].split("of")[1]) for p in shards}
    if len(counts) != 1 or len(shards) != next(iter(counts)):
        raise G1Error(f"singles shards {[p.name for p in shards]} are not one complete set")
    records = [json.loads(line) for p in shards for line in p.read_text().splitlines()]
    records.sort(key=lambda r: (r["order"], r["strength"]))
    records[0].setdefault("shards", [p.name for p in shards])
    write_records(instance_id, "singles", records)
    return records


def rank_key(rec: Mapping[str, Any]) -> tuple[int, int, int, float]:
    """Proxy descending, then fewer changed entries, then member order (then strength)."""
    return (-int(rec["proxy"]["score"]), int(rec["n_changed"]), int(rec["order"]), float(rec["strength"]))


def eligible(singles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((r for r in singles if r.get("built") and r.get("within_limits") and "proxy" in r), key=rank_key)


def cmd_search(ctx: Ctx, instance_id: str) -> list[dict[str, Any]]:
    singles = read_records(instance_id, "singles")
    ranked = eligible(singles)
    top = ranked[:TOP_SINGLES]
    members = {m.name: m for m in family(ctx.depth)}
    pool = [(members[r["name"]], r["strength"], r) for r in top]
    chains: list[dict[str, Any]] = []
    for rec in ranked:  # singles are candidates too
        m = members[rec["name"]]
        child, info = try_build(ctx, ctx.base, m, rec["strength"])
        chains.append({"parts": [(m, rec["strength"])], "child": child, "proxy": rec["proxy"], "info": info,
                       "order": rec["order"], "origin": "single", "n_changed": rec["n_changed"]})
    composed: list[dict[str, Any]] = []
    for start_m, start_s, start_rec in pool:
        parts = [(start_m, start_s)]
        used: dict[int, set[int]] = {}
        child, info = try_build(ctx, ctx.base, start_m, start_s, used)  # records the start member's units
        score = start_rec["proxy"]["score"]
        sites = {start_m.site}
        for _step in range(MAX_MEMBERS - 1):
            best = None
            for m, s, rec in pool:
                if m.site in sites:
                    continue
                trial_used = {k: set(v) for k, v in used.items()}
                trial, tinfo = try_build(ctx, child, m, s, trial_used)
                if trial is None:
                    continue
                stats = limits_stats(ctx, trial)
                if not stats["within_limits"]:
                    continue
                pr = proxy(ctx, trial)
                key = (-pr["score"], stats["n_changed"], rec["order"])
                if best is None or key < best[0]:
                    best = (key, m, s, trial, tinfo | {"n_changed": stats["n_changed"]}, pr, trial_used, rec["order"])
            if best is None or best[5]["score"] <= score:
                break
            _, m, s, child, info, pr, used, order = best
            parts = parts + [(m, s)]
            sites.add(m.site)
            score = pr["score"]
            composed.append({"parts": list(parts), "child": child, "proxy": pr, "info": info,
                             "order": order, "origin": f"greedy from {start_m.name}@{start_s:g}",
                             "n_changed": info["n_changed"]})
    everything = chains + composed
    seen: dict[str, dict[str, Any]] = {}
    for c in sorted(everything, key=lambda c: (-c["proxy"]["score"], c["n_changed"], c["order"],
                                               len(c["parts"]))):
        sha = params_sha(c["child"])
        if sha not in seen:
            seen[sha] = c | {"child_sha256": sha}
    chosen = list(seen.values())[:MAX_CHECKS]
    print(f"search: {len(everything)} candidates, {len(seen)} distinct, {len(chosen)} to the checks", flush=True)
    records = [{"schema": SCHEMA, "stage": "candidates", "instance_id": instance_id, "rank": i,
                "parts": [{"member": dataclasses.asdict(m), "name": m.name, "strength": s} for m, s in c["parts"]],
                "origin": c["origin"], "proxy": c["proxy"], "n_changed": c["n_changed"],
                "child_sha256": c["child_sha256"]} for i, c in enumerate(chosen)]
    result = run_checks(ctx, instance_id, chosen)
    records.append(result)
    records[0]["provenance"] = provenance(instance_id)
    write_records(instance_id, "search", records)
    return records


def run_checks(ctx: Ctx, instance_id: str, chosen: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """execute_run over the chosen candidates (joint oracle, E10 limits, at most 8 checks), then the joint-retain-v1
    pick read on V_select."""
    from weightcode.baselines.fit.runner import CandidatePatch, MethodSpec, execute_run
    from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE, readout_retention

    inst = ctx.inst
    mech.install_base_memo(inst)
    if len(chosen) > MAX_CHECKS:
        raise G1Error("more candidates than checks")
    stamp = int(time.time())
    method = f"g1{'_smoke' if SMOKE else ''}_{instance_id}"
    run_dir = mech.runs_root() / "e11" / "g1" / "runs" / f"{method}_{stamp}"

    def generate(evidence: Any) -> tuple[list[Any], dict[str, Any]]:
        k = evidence.n_deployment
        if not (np.array_equal(evidence.x[:k], ctx.x_new) and np.array_equal(evidence.x[k:], ctx.x_old)):
            raise G1Error("the run's evidence rows are not the session's x_new and x_old")
        out = []
        for slot, c in enumerate(chosen):
            names = [f"{m.name}@{s:g}" for m, s in c["parts"]]
            out.append(CandidatePatch(
                slot=slot, child={k_: np.asarray(v) for k_, v in c["child"].items()},
                config={"name": "+".join(names), "parts": names, "origin": c["origin"], "proxy": c["proxy"]},
                hypothesis="G1 generic gated-correction search: " + " + ".join(names),
                predicted_effect="PREREG G1; chosen by the experience-row proxy, no LLM",
                notes={"child_sha256": c["child_sha256"]}))
        return out, {"labels_read": True, "inputs": "the session's experience rows with labels (proxy only)"}

    spec = MethodSpec(method=method, label="PREREG G1: frozen generic gated-correction search (no LLM)",
                      uses_labels_for_fit=True, edit_limits_enforced=True)
    record = execute_run(instance=inst, spec=spec, run_dir=run_dir, run_seed=0,
                         config={"spec": SPEC, "kept_rule": f"joint-retain-v1, S_old >= {FLOOR} of 4000"},
                         generate=generate, joint=True, replay_in_fit=True, e10=True)
    checks = []
    for entry in record["candidates"]:
        item = {"patch_id": entry["patch_id"], "config": entry["config"], "invalid": entry.get("invalid"),
                "executor_messages": entry.get("executor_messages")}
        if "check" in entry:
            c = entry["check"]
            item.update({k: c.get(k) for k in ("query_index", "score", "score_new", "score_old", "fixed_new",
                                              "broken_new", "fixed_old", "broken_old", "n_changed")})
            item["eligible"] = int(c["score_old"]) >= FLOOR
        checks.append(item)
    pick = readout_retention(run_dir / "run.json", FLOOR, DEFAULT_BATCH_SIZE,
                             label="PREREG G1: joint-retain-v1 pick (floor -20) read on V_select")
    ro = mech.summarize_readout(pick)
    bar = BARS[instance_id]
    return {"schema": SCHEMA, "stage": "pick", "instance_id": instance_id, "checks": checks,
            "queries_used": record["queries_used"], "n_invalid": record["n_invalid"], "selection": pick["selection"],
            "pick_vcheck": pick["vcheck"], "run_dir": str(run_dir),
            "run_json_sha256": mech.file_sha256(run_dir / "run.json"), **ro,
            "reading": {"bar_joint": bar, "clean_floor": READING_CLEAN_FLOOR,
                        "reproduces_most": bool(ro["joint"] >= bar and ro["clean"] >= READING_CLEAN_FLOOR)}}


def cmd_ceiling(ctx: Ctx, instance_id: str, shard: tuple[int, int]) -> list[dict[str, Any]]:
    singles = read_records(instance_id, "singles")
    todo = [r for r in singles if r.get("built") and r.get("within_limits") and "proxy" in r]
    members = {m.name: m for m in family(ctx.depth)}
    mech.install_base_memo(ctx.inst)
    index, count = shard
    records = []
    for i, rec in enumerate(todo):
        if i % count != index:
            continue
        child, info = try_build(ctx, ctx.base, members[rec["name"]], rec["strength"])
        sha = params_sha(child)
        if sha != rec["child_sha256"]:
            raise G1Error(f"{rec['name']}@{rec['strength']}: rebuilt child {sha} != recorded {rec['child_sha256']}")
        ro = mech.readout(ctx.inst, child)
        records.append({"schema": SCHEMA, "stage": "ceiling", "instance_id": instance_id, "name": rec["name"],
                        "strength": rec["strength"], "order": rec["order"], "child_sha256": sha,
                        "proxy": rec["proxy"], "n_changed": rec["n_changed"], **ro})
        if len(records) % 10 == 0:
            print(f"ceiling shard {index}/{count}: {len(records)} done", flush=True)
    write_records(instance_id, f"ceiling_{index}of{count}", records)
    return records


def cmd_summary(instance_id: str) -> None:
    singles = read_records(instance_id, "singles")
    search = read_records(instance_id, "search")
    pick = search[-1]
    bar = BARS[instance_id]
    n_total = len(singles)
    n_ok = sum(1 for r in singles if r.get("built") and r.get("within_limits"))
    n_out = len({r["name"] for r in singles if r.get("out_of_limits_at_every_strength")})
    print(f"== {CONDITION[instance_id]} ({instance_id}): {n_total} member-strengths, {n_ok} within the limits, "
          f"{n_out} members out of limits at every strength")
    print(f"pick {pick['selection'].get('patch_id')} of {pick['selection'].get('n_eligible')}/"
          f"{pick['selection'].get('n_checks')} eligible: gain {pick['gain']:+.2f} clean {pick['clean']:+.2f} joint "
          f"{pick['joint']:+.2f} (bar {bar:+.2f} at clean >= {READING_CLEAN_FLOOR:+.1f}) reproduces most: "
          f"{pick['reading']['reproduces_most']}")
    for c in pick["checks"]:
        print(f"  {c['patch_id']} S {c.get('score')} (new {c.get('score_new')}, old {c.get('score_old')}) "
              f"{c['config']['name']}")
    shards = sorted(out_dir().glob(f"g1_{CONDITION[instance_id]}_ceiling_*of*.jsonl"))
    ceiling = [json.loads(line) for p in shards for line in p.read_text().splitlines()]
    if ceiling:
        ok = [r for r in ceiling if r["clean"] >= READING_CLEAN_FLOOR]
        best = max(ok, key=lambda r: r["joint"], default=None)
        print(f"family ceiling ({len(ceiling)} members read on V_select, descriptive): "
              + (f"best joint at clean >= {READING_CLEAN_FLOOR:+.1f}: {best['name']}@{best['strength']:g} gain "
                 f"{best['gain']:+.2f} clean {best['clean']:+.2f} joint {best['joint']:+.2f}" if best else "none"))
        clears = best is not None and best["joint"] >= bar
        print(f"ceiling clears the bar: {clears}; pick clears the bar: {pick['reading']['reproduces_most']}")


# --------------------------------------------------------------------------------------------------------------------
# Self-test (no label, no V_ split)


def cmd_selftest(ctx: Ctx, instance_id: str) -> None:
    """Realization identities on x_new and x_old activations, and limit feasibility; prints no proxy and no readout."""
    a0 = acts(ctx, ctx.base)
    b = 3
    # (1) c1 ungated: the block output moves by exactly -s * offset at every token.
    m = Member(f"b{b}", "c1", None, 0, "both")
    child, info = try_build(ctx, ctx.base, m, 1.0)
    a1 = acts(ctx, child)
    off = a0.h_new[b].reshape(-1, ctx.dim).mean(0) - a0.h_old[b].reshape(-1, ctx.dim).mean(0)
    err = np.max(np.abs((a1.h_new[b] - a0.h_new[b]) + off))
    print(f"c1 ungated b{b}: max |move + offset| = {err:.2e} within_limits {info['within_limits']}")
    # (2) tokenizer c2 at strength 1: per-channel token std of x_new equals x_old's.
    child, info = try_build(ctx, ctx.base, Member("tok", "c2"), 1.0)
    _, tn = patch_tokens(ctx, child, ctx.x_new)
    _, to = patch_tokens(ctx, ctx.base, ctx.x_old)
    print(f"tokenizer c2: max |std_new' - std_old| = {np.max(np.abs(tn.std(0) - to.std(0))):.2e} "
          f"within_limits {info['within_limits']} rel_frob {info.get('per_tensor_rel_frob')}")
    # (3) c3 ungated: the block output moves by the rank-r fit of the target, up to the stated R^2.
    m3 = Member(f"b{b}", "c3", 4, 0, "both")
    child, info = try_build(ctx, ctx.base, m3, 0.5)
    a3 = acts(ctx, child)
    moved = (a3.h_new[b] - a0.h_new[b]).reshape(-1, ctx.dim)
    hn = a0.h_new[b].reshape(-1, ctx.dim)
    ho = a0.h_old[b].reshape(-1, ctx.dim)
    dm = cov_difference_map(np.cov(hn.T, bias=True), np.cov(ho.T, bias=True), 4)
    want = 0.5 * (hn - hn.mean(0)) @ dm
    rel = np.linalg.norm(moved - want) / np.linalg.norm(want)
    print(f"c3 ungated b{b} r4 s0.5: fit R^2 {info['fit_r2']:.4f} (x_new {info['fit_r2_x_new']:.4f}, x_old "
          f"{info['fit_r2_x_old']:.4f}); the realized move differs from the target by {rel:.4f} of its norm on x_new "
          f"(the fit's error, not the pairs'; DEBUG: realized vs fit 0.011 on x_new, 0.038 on x_old), "
          f"within_limits {info['within_limits']} rel_frob {info.get('per_tensor_rel_frob')}")
    # (3b) c3 gated (fit on the selected x_new tokens only, PREREG G1 fit amendment): the three R^2 again.
    child, info = try_build(ctx, ctx.base, Member(f"b{b}", "c3", 4, 1, "both"), 0.25)
    if child is None:
        print(f"c3 gated b{b} r4 s0.25: not buildable ({info['error']})")
    else:
        print(f"c3 gated b{b} r4 s0.25 (fit rows: {info['fit_rows']}): fit R^2 {info['fit_r2']:.4f} (x_new "
              f"{info['fit_r2_x_new']:.4f}, x_old {info['fit_r2_x_old']:.4f}), open on x_old scores "
              f"{info.get('open_old_scores')}, within_limits {info['within_limits']}")
    # (4) gated c1: closed on most x_old tokens, the full offset when open.
    mg = Member(f"b{b}", "c1", None, 2, "both")
    child, info = try_build(ctx, ctx.base, mg, 1.0)
    print(f"c1 gated k2 b{b}: open on x_old scores {info.get('open_old_scores')}, on x_new {info.get('open_new')}, "
          f"kappa {info.get('kappa'):.2f}, within_limits {info['within_limits']}")
    # (5) feasibility census of the whole family at strength 1 (builds only; no labels).
    counts = {"built": 0, "within": 0, "unbuilt": 0, "outside": 0}
    for member in family(ctx.depth):
        child, info = try_build(ctx, ctx.base, member, 1.0)
        if child is None:
            counts["unbuilt"] += 1
            print(f"  unbuilt {member.name}: {info['error']}", flush=True)
        else:
            counts["built"] += 1
            counts["within" if info["within_limits"] else "outside"] += 1
            if not info["within_limits"]:
                print(f"  outside {member.name}: {info['violations']}", flush=True)
    print(f"family at strength 1: {counts}, {len(family(ctx.depth))} members")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", required=True, choices=sorted(CONDITION))
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("selftest", "search", "summary", "merge"):
        sub.add_parser(name)
    for name in ("singles", "ceiling"):
        sh = sub.add_parser(name)
        sh.add_argument("--shard", default="0/1", help="i/n: this process takes the members with index = i mod n")
    args = parser.parse_args(argv)
    if args.stage == "summary":
        cmd_summary(args.instance)
        return 0
    if args.stage == "merge":
        cmd_merge(args.instance)
        return 0
    ctx = make_ctx(args.instance)
    if args.stage == "selftest":
        cmd_selftest(ctx, args.instance)
    elif args.stage == "singles":
        i, n = (int(v) for v in args.shard.split("/"))
        cmd_singles(ctx, args.instance, (i, n))
    elif args.stage == "search":
        cmd_search(ctx, args.instance)
    else:
        i, n = (int(v) for v in args.shard.split("/"))
        cmd_ceiling(ctx, args.instance, (i, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
