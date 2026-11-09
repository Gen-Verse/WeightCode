"""PREREG E12 C's reading (``scripts/e12/e12c_run.py``, ``condition_table``) on synthetic records."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "e12" / "e12c_run.py"


def _script():
    spec = importlib.util.spec_from_file_location("e12c_run_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rec(seed: int, gain: float, clean: float, within: bool = True) -> dict:
    return {"target": {"instance_id": f"i{seed}", "seed": seed}, "status": "ok", "within_limits": within,
            "readouts": {"delta_acc": gain / 100.0}, "readouts_clean": {"delta_acc": clean / 100.0},
            "n_changed": 10, "child_sha256": f"c{seed}"}


OWN = {"readouts": {"delta_acc": 0.30}, "readouts_clean": {"delta_acc": -0.004}, "within_limits": True,
       "child_sha256": "own"}


def test_c_supported_when_ratio_clean_and_limits_hold() -> None:
    g = _script()
    rerun = [_rec(s, 24.0, -0.5) for s in g.SEEDS]  # ratio 0.8
    frozen = [_rec(s, 0.5, -0.2) for s in g.SEEDS]
    t = g.condition_table(rerun, frozen, OWN)
    assert t["reuse_ratio"] == 0.8 and t["rerun_clean_ok"] == 5 and t["C_supported"]
    assert t["frozen_reading"] == "the program, not its values"


def test_each_clause_can_fail_c() -> None:
    g = _script()
    frozen = [_rec(s, 20.0, -0.2) for s in g.SEEDS]
    low = g.condition_table([_rec(s, 20.0, -0.5) for s in g.SEEDS], frozen, OWN)  # ratio 0.667 < 0.7
    assert not low["C_supported"] and low["frozen_reading"] == "the values transfer"
    dirty = [_rec(s, 25.0, -1.5 if s < 103 else -0.5) for s in g.SEEDS]  # clean ok on 3 of 5
    assert not g.condition_table(dirty, frozen, OWN)["C_supported"]
    refused = [_rec(s, 25.0, -0.5, within=s != 105) for s in g.SEEDS]
    t = g.condition_table(refused, frozen, OWN)
    assert not t["C_supported"] and not t["rerun_all_within_limits"]


def test_between_the_shares_both_are_reported() -> None:
    g = _script()
    rerun = [_rec(s, 25.0, -0.5) for s in g.SEEDS]
    frozen = [_rec(s, 15.0, -0.5) for s in g.SEEDS]  # share 0.6
    assert g.condition_table(rerun, frozen, OWN)["frozen_reading"] == "both reported, neither wording alone"
