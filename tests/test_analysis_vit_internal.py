"""weightcode.analysis.vit_internal: the block-level trace against the ViT forward.py template, gate parsing and checks,
the accumulated statistics and the descriptive rank correlation, on toy networks with known answers."""

from __future__ import annotations

import numpy as np
import pytest

import weightcode.analysis.vit_internal as V
from weightcode.evalkit.patch_check import load_forward
from weightcode.models import vit_synth

K = V.GAUSSIAN_K_RAMP


@pytest.fixture
def toy(tmp_path):
    """The TINY ViT forward.py (the E11 template) loaded as the instances load theirs, and random float32 params."""
    path = tmp_path / "forward.py"
    path.write_text(vit_synth.render_forward(vit_synth.TINY, [0.5, 0.4, 0.3], [0.25, 0.2, 0.3]), encoding="utf-8")
    forward, _ = load_forward(path)
    return V.forward_namespace(forward), vit_synth.init_params(vit_synth.TINY, 7)


def _images(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, 3, 8, 8), dtype=np.uint8)


def test_trace_equals_forward_bitwise(toy):
    ns, params = toy
    x = _images(9)
    tr = V.trace(ns, params, x, [0, 1])
    assert V.parity(ns, params, x, tr.logits, atol=0.0) == 0.0
    assert tr.tokens.shape == (9, 4, 16) and tr.mlp_in[1].shape == (9, 5, 16) and tr.block_out[0].shape == (9, 5, 16)
    assert set(tr.mlp_in) == {0, 1}


def test_trace_refuses_more_than_one_chunk_and_non_uint8(toy):
    ns, params = toy
    with pytest.raises(V.InternalError, match="CHUNK"):
        V.trace(ns, params, _images(int(ns["CHUNK"]) + 1), [0])
    with pytest.raises(V.InternalError, match="uint8"):
        V.trace(ns, params, _images(2).astype(np.float32), [0])


def test_forward_namespace_refuses_other_modules():
    def forward(params, x):  # noqa: ARG001
        return x

    with pytest.raises(V.InternalError, match="not the ViT-Lite forward"):
        V.forward_namespace(forward)


def _ramp_child(params, block=1, j1=3, j2=7, bias=-0.4, offset_seed=1):
    child = {k: np.array(v, copy=True) for k, v in params.items()}
    p = V.block_name(block)
    row = np.random.default_rng(offset_seed).normal(size=16).astype(np.float32)
    child[f"{p}.linear1.weight"][j1] = row
    child[f"{p}.linear1.weight"][j2] = row
    child[f"{p}.linear1.bias"][j1] = bias
    child[f"{p}.linear1.bias"][j2] = bias - K
    off = np.random.default_rng(offset_seed + 1).normal(size=16).astype(np.float32)
    child[f"{p}.linear2.weight"][:, j1] = -off / K
    child[f"{p}.linear2.weight"][:, j2] = off / K
    return child, row, off


STDOUT = ("noisy KL to clean before edit: 1.5\n"
          "gate patch block 1: |offset| 5.09, held-out/clean sd 0.433, units 3,7, kappa 6.59 (ramp 1.05 sd), thr 3.00\n"
          "  round 0 oracle class-offset removal after block 3: KL 0.521\n")


def test_parse_and_check_ramp_gates(toy):
    _, params = toy
    child, row, off = _ramp_child(params)
    gates = V.parse_gaussian_gates(STDOUT)
    assert gates == [V.RampGate("patch", 1, 3, 7)]
    (info,) = V.check_ramp_gates(params, child, gates)
    assert info["kappa"] == pytest.approx(np.linalg.norm(row.astype(np.float64)))
    assert info["offset_norm"] == pytest.approx(np.linalg.norm(off.astype(np.float64)), rel=1e-6)
    bad = {k: np.array(v, copy=True) for k, v in child.items()}
    bad["classifier.blocks.1.linear1.bias"][7] += 0.5
    with pytest.raises(V.InternalError, match="bias gap"):
        V.check_ramp_gates(params, bad, gates)
    extra = {k: np.array(v, copy=True) for k, v in child.items()}
    extra["classifier.blocks.1.linear1.weight"][11] += 1.0
    with pytest.raises(V.InternalError, match="changed linear1 rows"):
        V.check_ramp_gates(params, extra, gates)
    with pytest.raises(V.InternalError, match="no gate line"):
        V.parse_gaussian_gates("nothing here\n")


def test_ramp_gate_sums_known_rates():
    """u1 = first coordinate of each token; u2 = u1 - K. Two images of three tokens (class token first)."""
    hidden = 4
    w1 = np.zeros((hidden, 2))
    w1[0, 0] = w1[1, 0] = 1.0
    b1 = np.array([0.0, -K, 0.0, 0.0])
    w2 = np.zeros((2, hidden))
    w2[:, 0], w2[:, 1] = [-1.0, 0.0], [1.0, 0.0]
    child = {"classifier.blocks.0.linear1.weight": w1, "classifier.blocks.0.linear1.bias": b1,
             "classifier.blocks.0.linear2.weight": w2}
    mlp_in = np.zeros((2, 3, 2))
    mlp_in[0, :, 0] = [1.0, 4.0, -1.0]  # cls above, patches: fully open, shut
    mlp_in[1, :, 0] = [-2.0, -1.0, -3.0]  # all shut
    gelu = vit_synth_gelu()
    sums = V.Sums()
    gate = V.RampGate("patch", 0, 0, 1)
    V.ramp_gate_sums(sums, "x_", mlp_in, child, gate, gelu, np.array([1, 0]), 3)
    out = V.ramp_gate_finish(sums, "x_", with_classes=True)
    assert out["patch"]["n_tokens"] == 4 and out["patch"]["frac_above_threshold"] == 0.25
    assert out["patch"]["frac_fully_open"] == 0.25
    assert out["cls"]["frac_above_threshold"] == 0.5 and out["n_images"] == 2
    assert out["images_any_patch_above"] == 0.5
    assert out["per_class_gated_rate"] == [0.0, 0.5, None] and out["per_class_images"] == [1, 1, 0]
    # per token the output is gelu(u1) w2[:, 0] + gelu(u2) w2[:, 1], of norm |gelu(u1) - gelu(u1 - K)|, which is
    # also K times |opening|; the patch tokens have u1 = 4, -1, -1, -3
    diffs = [float(gelu(np.array(u)) - gelu(np.array(u - K))) for u in (4.0, -1.0, -1.0, -3.0)]
    assert out["patch"]["mean_gate_output_norm"] == pytest.approx(sum(abs(d) for d in diffs) / 4)
    assert out["patch"]["mean_opening"] == pytest.approx(sum(diffs) / 4 / K)


def vit_synth_gelu():
    return lambda a: 0.5 * a * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (a + 0.044715 * a ** 3)))


def _contrast_child(params, block=1, rank=6, kappa=V.CONTRAST_KAPPA):
    child = {k: np.array(v, copy=True) for k, v in params.items()}
    p = V.block_name(block)
    rng = np.random.default_rng(5)
    order = [9, 2, 14, 30, 21, 5, 17, 26]  # program order: rank map units, then the kappa unit, then the gate
    for u in order:
        child[f"{p}.linear1.weight"][u] = rng.normal(size=16).astype(np.float32)
    g = rng.normal(size=16)
    g /= np.linalg.norm(g)
    alpha, theta = 1.7, 0.6
    child[f"{p}.linear1.weight"][order[rank]] = (alpha * g).astype(np.float32)
    child[f"{p}.linear1.weight"][order[rank + 1]] = (alpha * g).astype(np.float32)
    child[f"{p}.linear1.bias"][order[rank + 1]] = -alpha * theta
    child[f"{p}.linear1.bias"][order[rank]] = -alpha * theta + kappa
    stdout = f"block {block}: gate d'=6.22 alpha=1.700 theta=0.600 units={order} relfrob1=0.4\n"
    return child, order, stdout


def test_contrast_gate_identification(toy):
    _, params = toy
    child, order, stdout = _contrast_child(params)
    gate = V.contrast_gate(params, child, stdout, block=1)
    assert gate["gate_unit"] == order[7] and gate["kappa_unit"] == order[6] and gate["units"] == sorted(order)
    assert gate["alpha"] == pytest.approx(1.7, abs=1e-5) and gate["theta"] == pytest.approx(0.6, abs=1e-5)
    with pytest.raises(V.InternalError, match="printed state"):
        V.contrast_gate(params, child, stdout.replace("theta=0.600", "theta=0.700"), block=1)
    with pytest.raises(V.InternalError, match="printed units"):
        V.contrast_gate(params, child, stdout.replace("units=[9,", "units=[8,"), block=1)
    wrong_kappa, _, _ = _contrast_child(params, kappa=2.0)
    with pytest.raises(V.InternalError, match="bias gap"):
        V.contrast_gate(params, wrong_kappa, stdout, block=1)


def test_contrast_gate_sums_rates(toy):
    ns, params = toy
    child, order, stdout = _contrast_child(params)
    gate = V.contrast_gate(params, child, stdout, block=1)
    x = _images(6, seed=3)
    tr = V.trace(ns, child, x, [1])
    sums = V.Sums()
    y = np.array([0, 1, 1, 2, 3, 3])
    V.contrast_gate_sums(sums, "c_", tr.mlp_in[1], child, gate["units"], gate["gate_unit"], ns["_gelu"], y, 4, block=1)
    out = V.contrast_gate_finish(sums, "c_", with_classes=True)
    pre = tr.mlp_in[1] @ np.asarray(child["classifier.blocks.1.linear1.weight"], np.float64)[gate["gate_unit"]] \
        + float(child["classifier.blocks.1.linear1.bias"][gate["gate_unit"]])
    opened = pre > 0
    assert out["open_patch_tokens"] == int(opened[:, 1:].sum())
    assert out["rate_cls_token"] == pytest.approx(opened[:, 0].mean())
    assert out["share_images_majority_patch_tokens_open"] == pytest.approx((opened[:, 1:].mean(1) > 0.5).mean())
    per_img = opened[:, 1:].mean(1)
    assert out["per_class_patch_open_rate"][1] == pytest.approx(per_img[1:3].mean())


def test_offsets_known_values():
    rng = np.random.default_rng(0)
    base_clean = rng.normal(size=(4, 3, 2))
    shift = np.array([3.0, 4.0])
    outs = {"base_clean": base_clean, "base_cond": base_clean + shift, "edit_clean": base_clean + 0.1,
            "edit_cond": base_clean + 0.1 + 0.5 * shift}
    sums = V.Sums()
    V.offset_sums(sums, "b_", {k: v[:2] for k, v in outs.items()})
    V.offset_sums(sums, "b_", {k: v[2:] for k, v in outs.items()})
    out = V.offset_finish(sums, "b_")
    assert out["base"]["patch"]["mean_offset_norm"] == pytest.approx(5.0)
    assert out["base"]["cls"]["mean_paired_norm"] == pytest.approx(5.0)
    assert out["edited_within"]["patch"]["mean_offset_norm"] == pytest.approx(2.5)
    assert out["residual_share"]["patch"]["edited_within"] == pytest.approx(0.5)
    assert out["edit_move_clean"]["patch"]["mean_paired_norm"] == pytest.approx(0.1 * np.sqrt(2))
    assert out["base"]["patch"]["n_tokens"] == 8 and out["base"]["cls"]["n_tokens"] == 4


def test_spearman_and_ranks():
    assert V.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert V.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert V.spearman([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) is None
    assert V.spearman([1, 2], [1, 2]) is None
    assert list(V.ranks([10, 20, 20, 30])) == [1.0, 2.5, 2.5, 4.0]
    # with ties: Pearson on average ranks
    a, b = [1, 2, 2, 3, 4], [3, 1, 2, 5, 4]
    ra, rb = V.ranks(a), V.ranks(b)
    assert V.spearman(a, b) == pytest.approx(np.corrcoef(ra, rb)[0, 1])


def test_token_spread_and_ratio():
    t = np.zeros((2, 4, 3))
    t[1, :, 0] = [0.0, 2.0, 0.0, 2.0]
    spread = V.token_spread(t)
    assert spread[0] == 0.0 and spread[1] == pytest.approx(np.sqrt(1.0 / 3.0))
    summary = V.ratio_summary(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
    assert summary["ratio_of_means"] == pytest.approx(0.75) and summary["mean_of_image_ratios"] == pytest.approx(0.75)
