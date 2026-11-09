"""PREREG T07-U (scripts/t07/t07_u.py): the tensor-level arms and their partition check, the fixed readings on
synthetic label vectors, the additivity residual and the record digests, on toy arrays only (no store)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts" / "t07"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("t07_u", HERE / "t07_u.py")
U = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = U
SPEC.loader.exec_module(U)
C = U.C

from weightcode.common.identity import params_sha256  # noqa: E402

# ------------------------------------------------------------------------------------------------------------------
# Arms and the partition check


def toy() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    base = {"a": rng.normal(size=(2, 3)).astype(np.float32), "b": rng.normal(size=3).astype(np.float32),
            "c": rng.normal(size=(2, 2)).astype(np.float32), "d": rng.normal(size=4).astype(np.float32)}
    child = {k: v.copy() for k, v in base.items()}
    child["a"] = child["a"] * np.float32(1.45)
    child["b"][1] += np.float32(0.5)
    child["c"][:, 0] = 0
    return base, child


def test_partition_and_arms():
    base, child = toy()
    part = U.check_partition(base, child, ("a",), ("b", "c"))
    assert part["changed_entries"] == {"a": 6, "b": 1, "c": 2}
    assert part["U_entries"] == 6 and part["G_entries"] == 3
    plus_u = U.replace_tensors(base, child, ("a",))
    minus_u = U.replace_tensors(base, child, ("b", "c"))
    assert np.array_equal(plus_u["a"], child["a"]) and all(np.array_equal(plus_u[k], base[k]) for k in "bcd")
    assert np.array_equal(minus_u["a"], base["a"]) and all(np.array_equal(minus_u[k], child[k]) for k in "bcd")
    assert params_sha256(U.replace_tensors(base, child, ("a", "b", "c"))) == params_sha256(child)
    plus_u["a"][0, 0] = 99.0  # the arm is a copy: neither the base nor the child changes
    assert base["a"][0, 0] != 99.0 and child["a"][0, 0] != 99.0
    assert all(plus_u[k].dtype == base[k].dtype for k in base)


def test_partition_refusals():
    base, child = toy()
    with pytest.raises(U.UError, match=r"neither U nor G: \['c'\]"):
        U.check_partition(base, child, ("a",), ("b",))  # a changed tensor in neither set
    with pytest.raises(U.UError, match="share tensors"):
        U.check_partition(base, child, ("a", "b"), ("b", "c"))
    with pytest.raises(U.UError, match=r"does not change: \['d'\]"):
        U.check_partition(base, child, ("a",), ("b", "c", "d"))
    with pytest.raises(U.UError, match="does not have"):
        U.check_partition(base, child, ("a", "z"), ("b", "c"))
    with pytest.raises(U.UError, match="twice"):
        U.check_partition(base, child, ("a", "a"), ("b", "c"))
    with pytest.raises(U.UError, match="keys differ"):
        U.check_partition(base, {k: v for k, v in child.items() if k != "d"}, ("a",), ("b", "c"))
    with pytest.raises(U.UError, match="float64"):
        U.check_partition(base, {**child, "d": child["d"].astype(np.float64)}, ("a",), ("b", "c"))
    with pytest.raises(U.UError, match="absent"):
        U.replace_tensors(base, child, ("z",))


# ------------------------------------------------------------------------------------------------------------------
# Readings


def test_count_rules():
    assert U.at_least_75(9, 12) and not U.at_least_75(8, 12)
    assert U.at_least_75(8, 10) and not U.at_least_75(7, 10)
    assert U.at_least_75(3, 4) and not U.at_least_75(2, 4)
    assert U.at_most_25(3, 12) and not U.at_most_25(4, 12)
    assert U.at_most_25(2, 10) and not U.at_most_25(3, 10)
    assert U.at_most_25(1, 4) and not U.at_most_25(2, 4)
    assert U.at_least_25(3, 12) and not U.at_least_25(2, 12)
    assert U.at_least_25(1, 4) and not U.at_least_25(0, 4)


@pytest.mark.parametrize("n, plus, minus, reading", [
    (12, 9, 3, U.SUPPORT), (12, 12, 0, U.SUPPORT), (12, 8, 3, U.MIXED), (12, 9, 4, U.MIXED),
    (12, 3, 9, U.REFUTE), (12, 4, 9, U.MIXED), (12, 3, 8, U.MIXED), (12, 6, 6, U.MIXED),
    (10, 8, 2, U.SUPPORT), (10, 7, 2, U.MIXED), (10, 8, 3, U.MIXED), (10, 2, 8, U.REFUTE),
    (4, 3, 1, U.SUPPORT), (4, 2, 1, U.MIXED), (4, 1, 3, U.REFUTE), (4, 2, 2, U.MIXED),
])
def test_off_target_reading(n, plus, minus, reading):
    assert U.off_target_reading(n, plus, minus) == reading


@pytest.mark.parametrize("share_s5, harm, n_harm, reading", [
    (0.4999, 0, 12, U.GAIN_COSTS), (0.10, 12, 12, U.GAIN_COSTS), (0.50, 0, 12, U.GAIN_PARTLY),
    (0.74, 0, 12, U.GAIN_PARTLY), (0.75, 3, 12, U.GAIN_SEPARABLE), (0.75, 4, 12, U.GAIN_PARTLY),
    (0.90, 2, 10, U.GAIN_SEPARABLE), (0.90, 3, 10, U.GAIN_PARTLY), (1.10, 0, 10, U.GAIN_SEPARABLE),
])
def test_own_gain_reading(share_s5, harm, n_harm, reading):
    assert U.own_gain_reading(share_s5, harm, n_harm) == reading


def synthetic(arch: str, plus_harm: int, minus_harm: int, plus_help: int, minus_help: int, minus_s5: float,
              carrier_harm: int = 0, carrier_help: int = 0) -> tuple[dict, dict]:
    """Labels and changes on the 26 sets: the final as T07 recorded it on S_harm, S_help and its own sets; each arm
    harm on the first k S_harm sets and help on the first k S_help sets, neutral elsewhere."""
    s_harm, s_help = U.S_SETS[arch]["S_harm"], U.S_SETS[arch]["S_help"]
    final = {s: "harm" if s in s_harm else "help" if s in s_help or s in U.OWN[1:] else "neutral" for s in C.SETS}
    final_delta = {s: -3.0 if final[s] == "harm" else 4.0 if final[s] == "help" else 0.2 for s in C.SETS}
    final_delta[U.OWN_MAIN] = 18.0

    def arm(k_harm: int, k_help: int) -> dict[str, str]:
        return {s: "harm" if s in s_harm[:k_harm] else "help" if s in s_help[:k_help] else "neutral" for s in C.SETS}

    labels = {"final": final, "base_plus_U": arm(plus_harm, plus_help), "final_minus_U": arm(minus_harm, minus_help)}
    deltas = {"final": final_delta,
              "base_plus_U": {s: -2.0 if labels["base_plus_U"][s] == "harm" else 0.0 for s in C.SETS},
              "final_minus_U": {s: minus_s5 if s == U.OWN_MAIN else 0.0 for s in C.SETS}}
    if arch == "vit":
        labels["carrier_F"] = arm(carrier_harm, carrier_help)
        deltas["carrier_F"] = {s: 0.0 for s in C.SETS}
    return labels, deltas


def run(arch: str, *args, **kwargs) -> dict:
    labels, deltas = synthetic(arch, *args, **kwargs)
    return U.readings(arch, labels, deltas, U.S_SETS[arch]["S_harm"], U.S_SETS[arch]["S_help"])


def test_readings_support_and_separable():
    out = run("vit", 10, 2, 3, 0, 15.0)
    assert out["off_target"]["harm"]["reading"] == U.SUPPORT and out["off_target"]["help"]["reading"] == U.SUPPORT
    assert out["off_target"]["harm"]["base_plus_U_carries"] == 10 and out["off_target"]["harm"]["n_sets"] == 12
    assert out["own_gain"]["share_s5"] == pytest.approx(15.0 / 18.0)
    assert out["own_gain"]["reading"] == U.GAIN_SEPARABLE
    assert "names the first-layer boost" in out["off_target"]["harm"]["statement"]
    assert not out["carrier_F"]["F_named_as_contributor"]
    assert [row["set"] for row in out["own_gain"]["own_sets"]] == list(U.OWN)


def test_readings_refutation_and_cost():
    out = run("r18", 2, 9, 1, 4, 7.0)
    assert out["off_target"]["harm"]["reading"] == U.REFUTE and out["off_target"]["help"]["reading"] == U.REFUTE
    assert out["own_gain"]["reading"] == U.GAIN_COSTS
    assert "does not separate those two on E12" in out["off_target"]["harm"]["statement"]
    assert "carrier_F" not in out


def test_readings_mixed_and_partly():
    out = run("vit", 6, 6, 2, 2, 15.0)
    assert out["off_target"]["harm"]["reading"] == U.MIXED and out["off_target"]["help"]["reading"] == U.MIXED
    assert out["own_gain"]["reading"] == U.GAIN_PARTLY  # share 0.83 but final_minus_U harm on 6 of 12
    harm_rows = out["off_target"]["harm"]["per_set"]
    assert harm_rows[0]["base_plus_U_share"] == pytest.approx(2.0 / 3.0)
    assert out["own_gain"]["own_sets"][0]["set"] == "contrast_s1"
    assert out["own_gain"]["own_sets"][0]["final_minus_U_share"] == 0.0  # 0.0 of the final's +0.2
    out = run("vit", 9, 3, 3, 1, 9.0)
    assert out["own_gain"]["reading"] == U.GAIN_PARTLY  # share 0.5


def test_readings_carrier_and_refusal():
    assert run("vit", 10, 2, 3, 0, 15.0, carrier_harm=3)["carrier_F"]["F_named_as_contributor"]
    assert run("vit", 10, 2, 3, 0, 15.0, carrier_help=1)["carrier_F"]["F_named_as_contributor"]
    assert not run("vit", 10, 2, 3, 0, 15.0, carrier_harm=2)["carrier_F"]["F_named_as_contributor"]
    labels, deltas = synthetic("vit", 10, 2, 3, 0, 15.0)
    labels["final"]["fog_s5"] = "neutral"  # the final no longer carries help on every S_help set
    with pytest.raises(U.UError, match="S_help"):
        U.readings("vit", labels, deltas, U.S_SETS["vit"]["S_harm"], U.S_SETS["vit"]["S_help"])
    labels, deltas = synthetic("vit", 10, 2, 3, 0, 15.0)
    del labels["base_plus_U"]["snow_s5"]
    with pytest.raises(U.UError, match="no label"):
        U.readings("vit", labels, deltas, U.S_SETS["vit"]["S_harm"], U.S_SETS["vit"]["S_help"])


def test_share():
    assert U.share(2.0, -4.0) == -0.5 and U.share(1.0, 0.0) is None


# ------------------------------------------------------------------------------------------------------------------
# Additivity residual, digests, field comparison, the fixed design


def test_additivity_residual():
    rng = np.random.default_rng(1)
    c_base, c_plus, c_minus, c_final = (rng.integers(0, 2, size=500) for _ in range(4))
    values = U.additivity_values(c_final, c_plus, c_minus, c_base)
    d = {k: 100.0 * (v.mean() - c_base.mean()) for k, v in (("f", c_final), ("p", c_plus), ("m", c_minus))}
    assert 100.0 * values.mean() == pytest.approx(d["f"] - d["p"] - d["m"], abs=1e-12)
    additive = c_plus + c_minus - c_base  # a final whose effect is the sum of the two parts' effects
    assert U.additivity_values(additive, c_plus, c_minus, c_base).sum() == 0
    index = U.R.boot_index(500)
    assert U.R.boot_contrast(U.additivity_values(additive, c_plus, c_minus, c_base), index)["ci95"] == [0.0, 0.0]
    with pytest.raises(U.UError, match="shape"):
        U.additivity_values(c_final[:10], c_plus, c_minus, c_base)


def test_preds_lines_digest_is_sorted_sha256sum_text():
    lines = ["b" * 64 + "  runs/x/preds/vit/base/clean.npy", "a" * 64 + "  runs/x/preds/r18/base/clean.npy"]
    expected = hashlib.sha256(("a" * 64 + "  runs/x/preds/r18/base/clean.npy\n" + "b" * 64
                               + "  runs/x/preds/vit/base/clean.npy\n").encode()).hexdigest()
    assert U.preds_lines_digest(lines) == expected == U.preds_lines_digest(lines[::-1])


def test_same_fields():
    ours = {"fixed": 3, "broken": 1, "n": 20, "base_acc": 0.5, "child_acc": 0.6, "delta": 10.0}
    assert U.same_fields(ours, dict(ours), U.M03_FIELDS_INT, U.M03_FIELDS_FLOAT) == {}
    assert U.same_fields(ours, {**ours, "delta": 10.0 + 1e-13}, U.M03_FIELDS_INT, U.M03_FIELDS_FLOAT) == {}
    assert set(U.same_fields(ours, {**ours, "delta": 10.001, "broken": 2}, U.M03_FIELDS_INT,
                             U.M03_FIELDS_FLOAT)) == {"delta", "broken"}
    assert set(U.same_fields(ours, {k: v for k, v in ours.items() if k != "n"}, U.M03_FIELDS_INT, ())) == {"n"}


def test_design_is_fixed_and_complete():
    assert U.DESIGN["vit"]["arms"] == ("base_plus_U", "final_minus_U", "carrier_F")
    assert U.DESIGN["r18"]["arms"] == ("base_plus_U", "final_minus_U")
    for arch, spec in U.DESIGN.items():
        assert set(spec["expected"]) == set(spec["arms"])
        assert all(len(h) == 64 and int(h, 16) >= 0 for h in spec["expected"].values())
        assert not set(spec["u_tensors"]) & set(spec["g_tensors"])
        s = U.S_SETS[arch]
        assert set(s["S_harm"]) <= set(C.SETS) and set(s["S_help"]) <= set(C.SETS)
        assert not (set(s["S_harm"]) | set(s["S_help"])) & set(U.EXCLUDED_FROM_S)
        assert not set(s["S_harm"]) & set(s["S_help"])
    assert (len(U.S_SETS["vit"]["S_harm"]), len(U.S_SETS["vit"]["S_help"])) == (12, 4)
    assert (len(U.S_SETS["r18"]["S_harm"]), len(U.S_SETS["r18"]["S_help"])) == (10, 4)
    assert len(U.DESIGN["r18"]["u_tensors"]) == 3 and len(U.DESIGN["r18"]["g_tensors"]) == 8
    assert U.DESIGN["vit"]["m03"] == {"base_plus_U": ("N_B", "B"), "final_minus_U": ("N_FR", "FR"),
                                      "carrier_F": ("N_F", "F")}
    assert U.OWN == ("contrast_s1", "contrast_s2", "contrast_s3", "contrast_s4", "contrast_s5")
