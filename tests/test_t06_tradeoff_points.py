"""T06 pick rule (scripts/t06/tradeoff_points.py) on a synthetic candidate list with known answers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "t06_tradeoff_points", Path(__file__).resolve().parents[1] / "scripts" / "t06" / "tradeoff_points.py")
t06 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = t06  # dataclasses resolve annotations through sys.modules
SPEC.loader.exec_module(t06)

N_OLD = 4000
CLASSES = [f"c{k}" for k in range(3)]


def _cand(order: int, s_new: int, s_old: int, gain: float = 0.1, clean: float = 0.0, n_changed: int = 10):
    sel = t06.Readout(delta_acc=gain, fixed=50, broken=10, n=1000, n_base_errors=600, per_class=(0.2, -0.05, 0.1))
    cln = t06.Readout(delta_acc=clean, fixed=3, broken=8, n=4000, n_base_errors=400, per_class=(0.0, -0.01, -0.02))
    return t06.Candidate(order=order, cid=f"q{order}", config="", s_new=s_new, s_old=s_old, select=sel, clean=cln,
                         n_changed=n_changed)


# S_old -20 is exactly -0.5 points of 4,000 (-21 just outside); -40 is -1.0; -80 is -2.0 (-81 just outside).
CANDS = [_cand(0, 900, -21), _cand(1, 500, -20), _cand(2, 300, 5), _cand(3, 300, 0), _cand(4, 700, -40),
         _cand(5, 950, -80), _cand(6, 990, -81)]


def test_bound_filtering_is_exact_at_the_edges() -> None:
    assert t06.within_bound(-20, N_OLD, "-0.5") and not t06.within_bound(-21, N_OLD, "-0.5")
    assert t06.within_bound(-40, N_OLD, "-1.0") and not t06.within_bound(-41, N_OLD, "-1.0")
    assert t06.within_bound(-80, N_OLD, "-2.0") and not t06.within_bound(-81, N_OLD, "-2.0")
    assert t06.within_bound(0, N_OLD, "0") and not t06.within_bound(-1, N_OLD, "0")
    # largest S_new inside each bound: 0 -> q2 (tie with q3), -0.5 -> q1 (q0 at -21 is out), -1.0 -> q0,
    # -2.0 -> q5 at exactly -80 (q6 at -81 is out)
    assert [t06.pick(CANDS, N_OLD, b).order for b in t06.BOUNDS] == [2, 1, 0, 5]


def test_ties_go_to_the_earlier_check() -> None:
    # q2 and q3 both have S_new 300 inside the 0 bound; the earlier one wins whatever the list order
    assert t06.pick(list(reversed(CANDS)), N_OLD, "0").order == 2
    tied = [_cand(7, 400, -5), _cand(6, 400, -10)]
    assert t06.pick(tied, N_OLD, "-0.5").order == 6


def test_no_candidate_inside_gives_the_base() -> None:
    outside = [_cand(0, 900, -100), _cand(1, 50, -90)]
    assert all(t06.pick(outside, N_OLD, b) is None for b in t06.BOUNDS)
    op = t06.operating_point(None, CLASSES)
    assert op["base_fallback"] and op["gain"] == 0 and op["clean"] == 0 and op["broken_clean"] == 0
    assert op["n_changed"] == 0 and op["pick"] is None


def test_a_negative_s_new_inside_the_bound_is_picked_and_marked() -> None:
    # As written, the base is a fallback only, not a candidate: the largest S_new inside is taken even if <= 0.
    neg = [_cand(0, -276, 1, gain=-0.06), _cand(1, 344, -35, gain=0.087)]
    c = t06.pick(neg, N_OLD, "-0.5")
    op = t06.operating_point(c, CLASSES)
    assert c.order == 0 and op["vcheck_s_new_le_0"] and not op["base_fallback"]
    assert t06.pick(neg, N_OLD, "-1.0").order == 1


def test_operating_point_readings() -> None:
    op = t06.operating_point(_cand(0, 100, -4, gain=0.3162, clean=0.00125), CLASSES)
    assert op["gain"] == pytest.approx(31.62) and op["clean"] == pytest.approx(0.125)
    assert op["joint"] == pytest.approx(31.745)
    assert op["broken_share_clean"] == pytest.approx(8 / 3600) and op["broken_share_new"] == pytest.approx(10 / 400)
    assert op["worst_class_clean"] == {"class_index": 2, "class": "c2", "change": pytest.approx(-2.0)}
    assert op["worst_class_new"]["class"] == "c1"


def test_kept_rule_check_and_all_checks_diagnostic() -> None:
    # joint-retain-v1 (used only to check the picks of record): argmax S_joint with S_old >= -20, base included.
    assert t06.kept_joint_retain_v1(CANDS).order == 1  # q1 480 beats q2 305, q3 300; q0 is below the floor
    assert t06.kept_joint_retain_v1([_cand(0, -5, 2), _cand(1, 30, -31)]) is None  # the base wins
    eq = [_cand(0, 100, 0, n_changed=50), _cand(1, 100, 0, n_changed=5)]
    assert t06.kept_joint_retain_v1(eq).order == 1  # equal S_joint: fewer changed entries
    # the diagnostic sees checks with no V_select readout, and only reports
    checks = [(c.order, c.cid, c.s_new, c.s_old) for c in CANDS] + [(9, "q9", 990, -10)]
    assert t06.pick_over_checks(checks, N_OLD, "-0.5") == 9
    assert t06.pick_over_checks([], N_OLD, "0") is None


def test_check_method_accepts_only_rule_explained_differences() -> None:
    kept, other = _cand(0, 400, -2), _cand(1, 450, -15)
    m = t06.Method(key="x", label="x", family="F1", coverage="all", candidates=[kept, other], n_checked=2,
                   kept_of_record=kept, kept_source="synthetic")
    res = t06.check_method(m, "i0", N_OLD, CLASSES)
    # q1 has both the larger S_new (450) and the larger S_joint (435 against 398), so a record naming q0 as kept
    # cannot come from the kept rule: both checks must fail.
    assert not res["kept_rule_reproduced"] and len(res["failures"]) == 2
    m2 = t06.Method(key="x", label="x", family="F1", coverage="all", candidates=[kept, _cand(1, 410, -19)],
                    n_checked=2, kept_of_record=kept, kept_source="synthetic")
    res2 = t06.check_method(m2, "i0", N_OLD, CLASSES)
    assert res2["kept_rule_reproduced"] and not res2["same_candidate"] and res2["failures"] == []


def test_dof_follows_the_configuration_in_a_mixed_grid() -> None:
    # F1s s4 and s5 mix full and LoRA configurations; the DoF is the configuration's, never one per arm.
    per_tensor = {"a.weight": {"size": 600}, "a.bias": {"size": 20}, "b.weight": {"size": 300}}
    run = {"method": "f1s_s4_est_prior", "config": {"lora": {"rank": 4}},
           "generation": {"trainable": {"full": ["a.weight", "a.bias", "b.weight"], "lora": ["a.weight"]}},
           "candidates": [{"slot": 0, "edit": {"per_tensor": per_tensor}}]}
    lora_ref = {"generation": {"lora": {"targets": ["a.weight"], "rank": 4, "trained_entries": 160}}}
    assert t06._dof(run, lora_ref, "full")[0] == 920
    assert t06._dof(run, lora_ref, "lora") == (160, "derived: same targets and rank as F1 lora_r4's record")
    other_targets = {"generation": {"lora": {"targets": ["b.weight"], "rank": 4, "trained_entries": 99}}}
    assert t06._dof(run, other_targets, "lora") == (None, "not recorded")
    recorded = {"generation": {"lora": {"trained_entries": 86016}, "trainable_entries": 3670016}}
    assert t06._dof(recorded)[0] == 86016  # LoRA's trained factors, not the merged matrices' entries
