"""PREREG M03 and M06 (scripts/mech_v1/m03_m06.py): network construction, gate identification, the fixed M06 reading
and the bootstrap, on toy parameter sets with known answers."""

from __future__ import annotations

import importlib.util
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "mech_m03_m06", Path(__file__).resolve().parents[1] / "scripts" / "mech_v1" / "m03_m06.py")
mech = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mech
SPEC.loader.exec_module(mech)

from weightcode.common.identity import params_sha256  # noqa: E402

D, HIDDEN, RANK = 6, 12, 2
UNITS_PROGRAM_ORDER = [9, 3, 7, 1]  # units[0..rank-1] carry the map, units[rank] the offset, units[rank + 1] the gate
ALPHA, THETA = 1.7, 0.6


def _toy() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """A parent and a child written the way the contrast program writes them (B, F and R), in float32."""
    rng = np.random.default_rng(0)
    parent = {
        mech.TOK: rng.normal(size=(D, 3, 2, 2)).astype(np.float32),
        mech.L1W: rng.normal(size=(HIDDEN, D)).astype(np.float32),
        mech.L1B: rng.normal(size=HIDDEN).astype(np.float32),
        mech.L2W: rng.normal(size=(D, HIDDEN)).astype(np.float32),
        mech.L2B: rng.normal(size=D).astype(np.float32),
        "classifier.fc.weight": rng.normal(size=(3, D)).astype(np.float32),
    }
    w1, b1, w2 = (np.asarray(parent[k], np.float64).copy() for k in (mech.L1W, mech.L1B, mech.L2W))
    b2 = np.asarray(parent[mech.L2B], np.float64).copy()
    g = rng.normal(size=D)
    g /= np.linalg.norm(g)
    for u in UNITS_PROGRAM_ORDER:
        b2 += 0.1 * w2[:, u]  # the mean fold
    outs = []
    for k in range(RANK):
        u = UNITS_PROGRAM_ORDER[k]
        w1[u] = ALPHA * g + rng.normal(size=D)
        b1[u] = -ALPHA * THETA + rng.normal()
        w2[:, u] = rng.normal(size=D)
        outs.append(w2[:, u].copy())
    u = UNITS_PROGRAM_ORDER[RANK]
    w1[u], b1[u], w2[:, u] = ALPHA * g, -ALPHA * THETA + mech.KAPPA, rng.normal(size=D)
    outs.append(w2[:, u].copy())
    u = UNITS_PROGRAM_ORDER[RANK + 1]
    w1[u], b1[u], w2[:, u] = ALPHA * g, -ALPHA * THETA, -np.sum(outs, axis=0)
    tok = np.asarray(parent[mech.TOK], np.float64) * 1.3
    child = dict(parent)
    child.update({mech.TOK: tok.astype(np.float32), mech.L1W: w1.astype(np.float32), mech.L1B: b1.astype(np.float32),
                  mech.L2W: w2.astype(np.float32), mech.L2B: b2.astype(np.float32)})
    return parent, child


UNITS = tuple(sorted(UNITS_PROGRAM_ORDER))


def test_units_come_from_the_linear1_delta_rows() -> None:
    parent, child = _toy()
    delta = child[mech.L1W].astype(np.float64) - parent[mech.L1W].astype(np.float64)
    assert mech.units_from_delta(delta) == UNITS
    assert mech.check_support(parent, child, UNITS) == {
        mech.TOK: parent[mech.TOK].size, mech.L1W: len(UNITS) * D, mech.L1B: len(UNITS),
        mech.L2W: len(UNITS) * D, mech.L2B: D}


def test_networks_take_exactly_their_components() -> None:
    parent, child = _toy()
    idx = list(UNITS)
    other = [u for u in range(HIDDEN) if u not in UNITS]
    nets = {name: mech.build_network(parent, child, UNITS, parts) for name, parts in mech.M03_NETWORKS}
    assert params_sha256(nets["N0"]) == params_sha256(parent)
    assert params_sha256(nets["N_FBR"]) == params_sha256(child)
    for name, net in nets.items():  # nothing outside the five component tensors moves; dtypes and shapes kept
        assert np.array_equal(net["classifier.fc.weight"], parent["classifier.fc.weight"])
        assert all(net[k].dtype == parent[k].dtype and net[k].shape == parent[k].shape for k in parent)
        assert np.array_equal(net[mech.TOK], child[mech.TOK] if "B" in name else parent[mech.TOK])
    n_b = nets["N_B"]
    assert all(np.array_equal(n_b[k], parent[k]) for k in (mech.L1W, mech.L1B, mech.L2W, mech.L2B))
    for name in ("N_F", "N_FB"):  # F: the units' linear2 columns zeroed, the child's linear2.bias, linear1 untouched
        net = nets[name]
        assert not net[mech.L2W][:, idx].any()
        assert np.array_equal(net[mech.L2W][:, other], parent[mech.L2W][:, other])
        assert np.array_equal(net[mech.L2B], child[mech.L2B])
        assert np.array_equal(net[mech.L1W], parent[mech.L1W]) and np.array_equal(net[mech.L1B], parent[mech.L1B])
    n_fr = nets["N_FR"]  # R on top of F, without B
    assert np.array_equal(n_fr[mech.TOK], parent[mech.TOK])
    for key in (mech.L1W, mech.L1B, mech.L2W, mech.L2B):
        assert np.array_equal(n_fr[key], child[key])
    assert params_sha256(n_fr) != params_sha256(child)
    assert len({params_sha256(n) for n in nets.values()}) == len(nets)
    m06 = {name: mech.build_network(parent, child, UNITS, parts) for name, parts in mech.M06_NETWORKS}
    assert params_sha256(m06["W1"]) == params_sha256(parent)
    assert params_sha256(m06["W1_B'F'R'"]) == params_sha256(child)
    assert params_sha256(m06["W1_F'R'"]) == params_sha256(nets["N_FR"])


def test_construction_refuses_r_without_f_and_unknown_parts() -> None:
    parent, child = _toy()
    with pytest.raises(ValueError):
        mech.build_network(parent, child, UNITS, "R")
    with pytest.raises(ValueError):
        mech.build_network(parent, child, UNITS, "BR")
    with pytest.raises(ValueError):
        mech.build_network(parent, child, UNITS, "FX")
    with pytest.raises(ValueError):
        mech.build_network(parent, child, UNITS, "FF")


def test_support_refuses_writes_no_component_covers() -> None:
    parent, child = _toy()
    outside = [u for u in range(HIDDEN) if u not in UNITS][0]
    for key, where in ((mech.L1W, (outside, 0)), (mech.L1B, (outside,)), (mech.L2W, (0, outside)),
                       ("classifier.fc.weight", (0, 0))):
        bad = {k: np.array(v, copy=True) for k, v in child.items()}
        bad[key][where] += 1.0
        with pytest.raises(mech.MechError):
            mech.check_support(parent, bad, UNITS)


def test_gate_unit_is_identified_from_the_weights() -> None:
    _, child = _toy()
    gate = mech.identify_gate(child[mech.L1W], child[mech.L1B], UNITS)
    assert gate["gate_unit"] == UNITS_PROGRAM_ORDER[RANK + 1]
    assert gate["kappa_unit"] == UNITS_PROGRAM_ORDER[RANK]
    assert gate["alpha"] == pytest.approx(ALPHA, abs=1e-5)
    assert gate["theta"] == pytest.approx(THETA, abs=1e-5)
    assert gate["bias_gap"] == pytest.approx(mech.KAPPA, abs=1e-5)
    stdout = (f"block 2: gate d'=6.51 alpha={ALPHA:.3f} theta={THETA:.3f} units={UNITS_PROGRAM_ORDER} "
              "relfrob1=0.452\n")
    assert mech.stdout_units(stdout) == UNITS_PROGRAM_ORDER
    checked = mech.check_gate_against_stdout(gate, UNITS_PROGRAM_ORDER, stdout, rank=RANK)
    assert all(checked["checks"].values())
    with pytest.raises(mech.MechError):  # the printed order puts another unit at units[rank + 1]
        mech.check_gate_against_stdout(gate, UNITS_PROGRAM_ORDER[::-1], stdout, rank=RANK)


def test_gate_identification_refuses_ambiguous_or_wrong_structure() -> None:
    _, child = _toy()
    l1w, l1b = np.array(child[mech.L1W], copy=True), np.array(child[mech.L1B], copy=True)
    no_pair = l1w.copy()
    no_pair[UNITS_PROGRAM_ORDER[RANK]] += 0.01
    with pytest.raises(mech.MechError):
        mech.identify_gate(no_pair, l1b, UNITS)
    two_pairs = l1w.copy()
    two_pairs[UNITS_PROGRAM_ORDER[1]] = two_pairs[UNITS_PROGRAM_ORDER[0]]
    with pytest.raises(mech.MechError):
        mech.identify_gate(two_pairs, l1b, UNITS)
    wrong_gap = l1b.copy()
    wrong_gap[UNITS_PROGRAM_ORDER[RANK]] += 0.5
    with pytest.raises(mech.MechError):
        mech.identify_gate(l1w, wrong_gap, UNITS)
    with pytest.raises(mech.MechError):
        mech.stdout_units("no list here")


@pytest.mark.parametrize(("loss_f", "loss_b", "loss_star", "reading"), [
    (20.0, 5.0, 27.0, mech.READING_COLLISION),   # F' carries most of the loss
    (5.0, 20.0, 27.0, mech.READING_BOOST),       # B' carries most of the loss
    (13.0, 12.0, 27.0, mech.READING_BOTH),       # each below half of L*
    (13.5, 13.0, 27.0, mech.READING_COLLISION),  # exactly half of L* counts
    (13.0, 13.4, 27.0, mech.READING_BOTH),       # B' larger but below half
    (13.0, 13.5, 27.0, mech.READING_BOOST),      # exactly half of L*, and the larger
    (30.0, 29.0, 27.0, mech.READING_COLLISION),  # both above L*: the larger one decides
])
def test_m06_reading_rule(loss_f: float, loss_b: float, loss_star: float, reading: str) -> None:
    result = mech.m06_reading(loss_f, loss_b, loss_star)
    assert result["reading"] == reading
    assert result["shares_of_L*"] == {"F'": loss_f / loss_star, "B'": loss_b / loss_star}
    assert not result["tie_both_conditions_hold"]


def test_m06_reading_in_rows_is_exact_at_half() -> None:
    # L* = 1139 rows (28.475 points); L(F') = 570 rows is above half (569.5), 569 rows below
    assert mech.m06_reading(570, 100, 1139)["reading"] == mech.READING_COLLISION
    assert mech.m06_reading(569, 100, 1139)["reading"] == mech.READING_BOTH
    assert mech.loss_rows_against_w1({mech.GAUSSIAN: {"fixed": 139, "broken": 1224}}) == 1085


def test_m06_reading_tie_is_flagged() -> None:
    result = mech.m06_reading(15.0, 15.0, 27.0)
    assert result["collision_condition"] and result["boost_condition"] and result["tie_both_conditions_hold"]
    assert result["reading"] == mech.READING_COLLISION


def test_loss_against_w1_is_minus_the_gaussian_change() -> None:
    assert mech.loss_against_w1({mech.GAUSSIAN: {"delta": -27.1}}) == pytest.approx(27.1)


def _correct(seed: int, n: int = 400) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return {name: {"gain": rng.random(n) < p, "clean": rng.random(n) < 0.9}
            for name, p in (("N_F", 0.5), ("N_FB", 0.6), ("N_FR", 0.65), ("N_FBR", 0.75))}


def test_bootstrap_is_deterministic_and_seeded() -> None:
    correct = _correct(1)
    first = mech.paired_bootstrap(correct, mech.M03_CONTRASTS, ("gain", "clean"))
    second = mech.paired_bootstrap(correct, mech.M03_CONTRASTS, ("gain", "clean"))
    assert first == second
    assert first["seed"] == 20261106 and first["n_resamples"] == 1000
    other = mech.paired_bootstrap(correct, mech.M03_CONTRASTS, ("gain", "clean"), seed=1)
    assert other["contrasts"] != first["contrasts"]
    for name, coefs in mech.M03_CONTRASTS.items():
        point = 100.0 * np.mean(sum(c * correct[net]["gain"].astype(int) for net, c in coefs.items()))
        assert first["contrasts"][name]["gain"]["point"] == pytest.approx(point)
        low, high = first["contrasts"][name]["gain"]["ci95"]
        assert low <= high
    # the resample index matrix is drawn once, so a contrast's numbers do not depend on which others are computed
    alone = mech.paired_bootstrap(correct, {"I": mech.M03_CONTRASTS["I = N_FBR - N_FB - N_FR + N_F"]}, ("gain",))
    assert alone["contrasts"]["I"]["gain"] == first["contrasts"]["I = N_FBR - N_FB - N_FR + N_F"]["gain"]


def test_bootstrap_matches_a_direct_computation() -> None:
    correct = _correct(2, n=50)
    result = mech.paired_bootstrap(correct, {"d": {"N_FBR": 1, "N_FB": -1}}, ("clean",), seed=7, n_resamples=200)
    index = np.random.default_rng(7).integers(0, 50, size=(200, 50))
    values = correct["N_FBR"]["clean"].astype(int) - correct["N_FB"]["clean"].astype(int)
    boot = 100.0 * values[index].mean(axis=1)
    assert result["contrasts"]["d"]["clean"]["ci95"] == pytest.approx(list(np.percentile(boot, [2.5, 97.5])))
    assert result["contrasts"]["d"]["clean"]["boot_sd"] == pytest.approx(boot.std(ddof=1))


def test_bootstrap_refuses_contrasts_that_do_not_cancel_the_parent() -> None:
    with pytest.raises(mech.MechError):
        mech.paired_bootstrap(_correct(3), {"bad": {"N_FBR": 1}}, ("gain",))


def test_bootstrap_identical_networks_give_zero() -> None:
    correct = _correct(4)
    correct["N_FB"] = correct["N_FBR"]
    result = mech.paired_bootstrap(correct, {"d": {"N_FBR": 1, "N_FB": -1}}, ("gain", "clean"))
    assert result["contrasts"]["d"]["gain"]["ci95"] == [0.0, 0.0]


def test_token_spread() -> None:
    t = np.zeros((2, 4, 3))
    t[0, :, 0] = [1.0, -1.0, 1.0, -1.0]  # image 0: channel 0 spreads with std 1, the others are constant
    t[1] += 5.0  # image 1: all tokens equal
    spread, std_all = mech.token_spread(t)
    assert spread[0] == pytest.approx(np.sqrt(1.0 / 3.0)) and spread[1] == 0.0
    assert std_all[1] == 0.0 and std_all[0] == pytest.approx(np.std(t[0]))
    summary = mech.ratio_summary(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
    assert summary["mean_of_image_ratios"] == pytest.approx(0.75)
    assert summary["ratio_of_means"] == pytest.approx(0.75)


def test_m06b_change_is_one_line_of_the_program() -> None:
    runs = os.environ.get("WEIGHTCODE_RUNS")
    if not runs:
        pytest.skip("WEIGHTCODE_RUNS is not set: the contrast final is read from the run store")
    program = Path(runs) / "sessions" / mech.SESSION_ID / "final" / mech.PROGRAM_FILE
    if not program.exists():
        pytest.skip("the contrast final is not in this store")
    text = program.read_text(encoding="utf-8")
    assert text.count(mech.M06B_OLD_LINE) == 1
    changed = text.replace(mech.M06B_OLD_LINE, mech.M06B_NEW_LINES)
    imp = np.array([5.0, 0.1, 3.0, 0.2, 9.0, 0.3, 0.4, 7.0])
    namespace = {"np": np, "imp": imp, "rank": 2}
    exec(textwrap.dedent(mech.M06B_NEW_LINES), namespace)  # noqa: S102
    assert list(namespace["units"]) == [1, 3, 5, 6]  # no exclusions apply below 360
    big = np.arange(600, dtype=float)[::-1].copy()
    big[[360, 503]] = -1.0  # the two least used, both excluded
    namespace = {"np": np, "imp": big, "rank": 6}
    exec(textwrap.dedent(mech.M06B_NEW_LINES), namespace)  # noqa: S102
    assert 360 not in namespace["units"] and 503 not in namespace["units"] and len(namespace["units"]) == 8
    assert list(namespace["units"]) == [599, 598, 597, 596, 595, 594, 593, 592]
    assert len([line for line in changed.splitlines() if line not in text.splitlines()]) == 2
