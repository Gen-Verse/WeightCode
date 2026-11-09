"""PREREG M05 phase 1 (scripts/mech_v1/m05.py): the fixed per-target reading, the reference tolerances and the main
gate summary, on known inputs. The cell readout itself is checked on NSCC against A2 and M03 (the diagonal cells)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "mech_m05", Path(__file__).resolve().parents[1] / "scripts" / "mech_v1" / "m05.py")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_programs_cover_all_t08_targets():
    for spec in M.PROGRAMS.values():
        assert len(spec["cells"]) == 6 and spec["own"] in spec["cells"]
        assert all(len(sha) == 64 for sha in spec["cells"].values())


def test_relation_rebuilt_rule():
    assert M.relation_rebuilt(0.95, 0.01, 0.2)["same_relation"]
    assert not M.relation_rebuilt(0.89, 0.01, 0.2)["same_relation"]
    assert not M.relation_rebuilt(0.95, 0.06, 0.2)["same_relation"]
    reading = M.relation_rebuilt(0.95, 0.01, 0.6)
    assert not reading["same_relation"] and reading["opens_on_condition"] and not reading["removes_offset"]
    assert not M.relation_rebuilt(0.95, 0.01, None)["same_relation"]
    assert M.relation_rebuilt(0.90, 0.05, 0.5)["same_relation"]


def test_reference_tolerances():
    assert M.close_rate(0.5, 0.5 + 2 / 256000, 256000)
    assert not M.close_rate(0.5, 0.5 + 3 / 256000, 256000)
    assert M.close_value(5.287878701265755, 5.287878701265755 * (1 + 5e-7))
    assert not M.close_value(5.0, 5.0 * (1 + 2e-6))


def _offsets(base, within):
    tok = {"mean_offset_norm": 0.0, "mean_paired_norm": 0.3}
    return {"base": {"patch": {**tok, "mean_offset_norm": base}}, "edited_within": {"patch": {**tok,
            "mean_offset_norm": within}}, "edited_vs_base_clean": {"patch": tok}, "edit_move_clean": {"patch": tok},
            "edit_move_cond": {"patch": tok}}


def test_main_gate_gaussian_and_contrast():
    sel = {"cond": {"patch": {"frac_above_threshold": 0.99, "mean_gate_output_norm": 4.0}},
           "clean": {"patch": {"frac_above_threshold": 0.004}}}
    gaussian = {"gates": [{"kind": "patch", "block": 2}, {"kind": "cls", "block": 4}],
                "selectivity": {"patch_block2": sel}, "offsets": {"block2": _offsets(5.0, 0.5)}}
    mg = M.main_gate("gaussian_noise", gaussian)
    assert mg["block"] == 2 and mg["cond_rate"] == 0.99 and mg["residual_share"] == pytest.approx(0.1)
    assert mg["offset_reduction"] == pytest.approx(0.9) and mg["correction_norm_cond_patch"] == 4.0
    contrast = {"gate": {"block": 2}, "gate_stats": {
        "cond": {"rate_patch_tokens": 0.98, "units_output_norm_patch_token_mean": 5.3},
        "clean": {"rate_patch_tokens": 0.0003}}, "offsets": {"block2": _offsets(2.0, 0.0)}}
    mg = M.main_gate("contrast", contrast)
    assert mg["clean_rate"] == 0.0003 and mg["residual_share"] == 0.0
    assert M.main_gate("contrast", {**contrast, "offsets": {"block2": _offsets(0.0, 0.0)}})["residual_share"] is None
