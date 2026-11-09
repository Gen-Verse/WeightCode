"""E9 v2: check score fscil-cb-v1 and kept rule fscil-retain-v2 (weightcode.evalkit.fscil)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from weightcode.evalkit import fscil as FS


def _check_labels(session: int, per_base: int = 30, per_inc: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """y_new and y_old of a toy check in fscil_check_parts' order: session t's classes, then base, then prev."""
    y_new = np.repeat(FS.session_classes(session), per_inc)
    y_old = np.concatenate([np.repeat(np.arange(FS.N_BASE_CLASSES), per_base),
                            np.repeat(FS.prev_classes(session), per_inc).astype(np.int64)])
    return y_new, y_old


def _wrong(y: np.ndarray) -> np.ndarray:
    return (y + 1) % FS.N_CLASSES


def test_class_balance_and_row_count_disagree() -> None:
    """A child fixes 50 of 100 rows of one new class and breaks 15 of 30 rows of two base classes: +20 rows, but
    -0.5 + -0.5 + 0.5 summed class deltas, so fscil-cb-v1 scores it below the student."""
    y, part = FS.cb_check_rows(*_check_labels(2), session=2)
    base_pred = y.copy()
    new60 = np.flatnonzero(y == 60)[:50]
    base_pred[new60] = _wrong(y[new60])  # the student misses 50 rows of class 60, the child gets them
    child_pred = base_pred.copy()
    child_pred[new60] = y[new60]
    broken = np.concatenate([np.flatnonzero(y == 0)[:15], np.flatnonzero(y == 1)[:15]])
    child_pred[broken] = _wrong(y[broken])
    score = FS.cb_check_score(base_pred, child_pred, y, part, 2)
    assert score["fixed"] - score["broken"] == 20  # what a row count would score
    k = 65
    assert math.isclose(score["S"], 100 * (0.5 - 0.5 - 0.5) / k)
    assert score["S"] < 0
    assert math.isclose(score["S_new"], 100 * 0.5 / k) and score["S_prev"] == 0.0
    assert math.isclose(score["S_base"], -100 / k)
    assert math.isclose(score["S_new"] + score["S_prev"] + score["S_base"], score["S"])
    assert math.isclose(score["B_c"], -100 / 60)
    assert (score["fixed_base"], score["broken_base"], score["n_base"]) == (0, 30, 1800)
    assert math.isclose(score["d_c"], 30 / 1800)
    assert score["per_class"]["0"] == {"n": 30, "fixed": 0, "broken": 15, "delta": -0.5}
    assert score["per_class"]["60"]["delta"] == 0.5 and len(score["per_class"]) == k
    assert score["rule"] == FS.CHECK_SCORE_RULE == "fscil-cb-v1"


def test_prev_classes_and_label_checks_at_session_3() -> None:
    y_new, y_old = _check_labels(3)
    y, part = FS.cb_check_rows(y_new, y_old, 3)
    assert list(part[: len(y_new)]) == ["new"] * len(y_new)
    assert set(part[(y >= 60) & (y < 65)]) == {"prev"} and set(part[y < 60]) == {"old"}
    child = y.copy()
    prev = np.flatnonzero(y == 62)[:10]
    child[prev] = _wrong(y[prev])
    score = FS.cb_check_score(y, child, y, part, 3)
    assert math.isclose(score["S_prev"], 100 * -0.1 / 70) and score["S_new"] == 0.0 and score["B_c"] == 0.0
    assert score["d_c"] == 0.0
    with pytest.raises(FS.FscilError, match="part labels"):
        bad = part.copy()
        bad[0] = "old"
        FS.cb_check_score(y, child, y, bad, 3)
    with pytest.raises(FS.FscilError, match="not seen"):
        FS.cb_check_score(y, child, np.where(y == 0, 99, y), part, 3)
    with pytest.raises(FS.FscilError, match="neither a base class"):
        FS.cb_check_rows(y_new, np.append(y_old, 70), 3)


def test_floor_in_points() -> None:
    assert round(FS.retain_v2_floor(0.006), 3) == -0.685
    assert math.isclose(FS.retain_v2_floor(0.0), -1.11)
    assert FS.retain_v2_floor(0.5) == 0.0  # capped at 0
    assert FS.KEPT_RULE_V2 == "fscil-retain-v2"
    with pytest.raises(FS.FscilError):
        FS.retain_v2_floor(-0.1)


def _entry(s: float, b: float, d: float) -> dict[str, float]:
    return {"S": s, "B_c": b, "d_c": d}


def test_pick_tie_and_the_student() -> None:
    # Index 0 has the highest S but breaks the base beyond its floor (-1.11 at d_c 0); 1 and 3 tie within 1e-9.
    scores = [_entry(5.0, -1.2, 0.0), _entry(2.0, -0.5, 0.006), _entry(1.0, 0.0, 0.0), _entry(2.0 + 1e-10, -0.6, 0.006)]
    assert FS.retain_v2_pick(scores) == 1
    assert FS.retain_v2_pick(scores[2:]) == 1  # within the slice, S 2 + 1e-10 beats S 1
    assert FS.retain_v2_pick([_entry(-0.3, 0.0, 0.0), _entry(0.0, 0.0, 0.0)]) is None  # no S above the student's 0
    assert FS.retain_v2_pick([_entry(1e-10, 0.0, 0.0)]) is None  # a tie with the student goes to the student
    assert FS.retain_v2_pick([_entry(3.0, -1.2, 0.0)]) is None  # nothing eligible
    assert FS.retain_v2_pick([_entry(3.0, -0.69, 0.006)]) is None and FS.retain_v2_pick([_entry(3.0, -0.68, 0.006)]) == 0
    assert FS.retain_v2_pick([]) is None
