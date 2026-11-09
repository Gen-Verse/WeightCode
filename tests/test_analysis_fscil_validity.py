"""weightcode.analysis.fscil_validity (PREREG E9 V1 to V3) on the toy store."""

from __future__ import annotations

import numpy as np
import pytest
from fscil_toy import add_toy_allclass, toy_fscil  # noqa: F401

from weightcode.analysis import fscil_validity as V


def test_retention_criterion_and_balanced_difference() -> None:
    assert V.retention_criterion(-0.4) == -1.0 and V.retention_criterion(-4.93) == -4.93
    y = np.array([0, 0, 1, 1, 1, 1])
    a = np.array([1, 0, 0, 0, 0, 0], dtype=bool)
    b = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    out = V.balanced_difference(y, a, b)
    assert out["bal_a"] == pytest.approx(0.25) and out["bal_b"] == pytest.approx(0.625)
    assert out["diff_points"] == pytest.approx(37.5) and out["n_classes"] == 2


def test_validity_report_on_the_toy(toy_fscil) -> None:  # noqa: F811
    add_toy_allclass(toy_fscil)
    report = V.validity_report(100)
    assert report["n_seen"] == 65 and report["n_rows"] == 130 and report["n_rows_new"] == 10
    assert report["V1"]["base_pred_in_new_count"] <= report["n_rows"] and "pass" in report["V2"]
    assert report["V3"]["n"] == 120 and report["V3"]["R"] <= -1.0
    assert sorted(report["per_session"], key=int) == [str(t) for t in range(1, 10)]
    first = report["per_session"]["1"]["base"]
    assert first["n_new"] == 0 and first["n_old"] == 120 and first["n_seen_classes"] == 60
    assert report["per_session"]["9"]["reference"]["n_prev"] == 70
