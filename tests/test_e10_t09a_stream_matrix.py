"""T09-A stream matrix readings (scripts/e10/t09a_stream_matrix.py) on a toy record with known answers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "t09a", Path(__file__).resolve().parents[1] / "scripts" / "e10" / "t09a_stream_matrix.py")
t09a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t09a)

IDS = ["ia", "ib", "ic"]
CONDS = ["a_s5", "b_s5", "c_s5"]
# V_select gain (fraction) of each instance after each round; clean change after each round.
GAIN = [[0.10, -0.02, 0.01], [0.15, 0.08, 0.03], [0.14, 0.07, 0.05]]
CLEAN = [-0.002, -0.010, -0.012]


def _record() -> dict:
    rounds = []
    for t, (row, clean) in enumerate(zip(GAIN, CLEAN), start=1):
        readouts = {i: {"select": {"delta_acc": g}, "clean": {"delta_acc": clean}} for i, g in zip(IDS, row)}
        rounds.append({"round": t, "status": "ok", "condition": CONDS[t - 1], "instance_id": IDS[t - 1],
                       "readouts": readouts})
    return {"schema": "weightcode.harness.loop_stream.v0", "instances": IDS, "rounds": rounds}


def test_readings_match_hand_computation() -> None:
    res = t09a.stream_matrix(_record())
    assert res["own_marginal"] == pytest.approx([10.0, 10.0, 2.0])
    assert res["round_sum"] == pytest.approx([9.0, 17.0, 0.0])
    assert res["seen_mean"] == pytest.approx([10.0, 11.5, 26.0 / 3])
    assert res["all_mean"] == pytest.approx([3.0, 26.0 / 3, 26.0 / 3])
    a, b = res["per_condition"]["a_s5"], res["per_condition"]["b_s5"]
    assert a["forgetting_signed"] == pytest.approx(-4.0) and a["forgetting_best"] == pytest.approx(1.0)
    assert b["gain_before_own_round"] == pytest.approx(-2.0) and b["forgetting_best"] == pytest.approx(1.0)
    assert res["per_condition"]["c_s5"]["gain_before_own_round"] == pytest.approx(3.0)
    assert res["max_forgetting_signed"] == pytest.approx(1.0) and res["max_forgetting_best"] == pytest.approx(1.0)
    stop = res["stop_after_2"]
    assert stop["w2_joint"] == pytest.approx(26.0 / 3 - 1.0) and stop["rounds_after_2_mean_gain"] == pytest.approx(0.0)


def test_unequal_clean_or_missing_readout_is_refused() -> None:
    record = _record()
    record["rounds"][1]["readouts"]["ic"]["clean"]["delta_acc"] = 0.5
    with pytest.raises(t09a.StreamRecordError, match="clean"):
        t09a.stream_matrix(record)
    record = _record()
    del record["rounds"][2]["readouts"]["ia"]
    with pytest.raises(t09a.StreamRecordError, match="no readout"):
        t09a.stream_matrix(record)
