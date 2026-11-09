"""PREREG T08 (``weightcode.harness.reuse_matrix``) and the F6 copy from a source that is not on seed 100.

Unit tests on synthetic records: the cell status rules, the diagonal gate, the fixed readings (reuse share, C in E3's
form and per target, program minus copy, per-target means) and the rule that a failed gate leaves a row unread. On
the toy E11 store (needs the sandbox): a seed-101 source's frozen deltas copy onto the other five students, seed 100
included, with F6's source checks; and the whole T08 path (program arm on six targets, copy arm on five, table) for a
seed-100 and a seed-101 source of one program. No test touches the real store.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from e11_toy import PLANT_CONDITION, e11_store  # noqa: F401
from test_harness_e11 import QKV_Q, _sandbox
from test_harness_fixtures import needs_sandbox
from test_harness_frozen_edit import _fake_session

from weightcode.common.identity import params_sha256
from weightcode.harness import fewshot
from weightcode.harness import frozen_edit as F
from weightcode.harness import reuse_matrix as R
from weightcode.harness.jsonio import file_sha256, read_json, write_json
from weightcode.harness.transfer import run_transfer
from weightcode.models import vit_synth as V

COND = "toy_condition"
IDS = tuple(f"i{seed}" for seed in R.TARGET_SEEDS)


def _source(sid: str, seed: int, child: str = "child", deltas: str = "deltas") -> R.Source:
    return R.Source(session_id=sid, condition=COND, instance_id=f"i{seed}", seed=seed, has_program=True,
                    program_path=f"/x/{sid}.py", program_sha256=f"p{sid}", deltas_sha256=deltas, child_sha256=child,
                    n_changed=1)


def _prog(sid: str, target: str, gain: float, clean: float, *, status: str = "ok", within: bool = True,
          n_changed: int = 5, child: str = "c", deltas: str = "d", run_status: str = "ok") -> dict[str, Any]:
    record: dict[str, Any] = {"status": status, "target": {"instance_id": target, "params_sha256": f"w{target}"},
                              "program": {"sha256": f"p{sid}"},
                              "run": {"clock": "virtual-v1", "run": {"status": run_status, "duration_s": 1.5}}}
    if status == "ok":
        record.update(within_limits=within, n_changed=n_changed, child_sha256=child, deltas_sha256=deltas,
                      readouts={"delta_acc": gain / 100}, readouts_clean={"delta_acc": clean / 100})
    else:
        record["messages"] = ["it failed"]
    return record


def _copy(sid: str, target: str, gain: float, clean: float, deltas: str = "deltas") -> dict[str, Any]:
    return {"status": "ok", "target": {"instance_id": target, "params_sha256": f"w{target}"},
            "source": {"program": {"sha256": f"p{sid}"}}, "within_limits": True, "n_changed": 5,
            "child_sha256": f"f{target}", "deltas_sha256": deltas,
            "readouts": {"delta_acc": gain / 100}, "readouts_clean": {"delta_acc": clean / 100}}


@pytest.fixture
def one_condition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "CONDITIONS", (COND,))
    monkeypatch.setattr(R, "TARGETS", {COND: IDS})
    monkeypatch.setattr(R, "SOURCES", {COND: ("sa", "sb")})


def test_cell_status_rules() -> None:
    assert R.program_cell(_prog("s", "i101", 20.0, -0.5))["status"] == R.STATUS_OK
    cell = R.program_cell(_prog("s", "i101", 20.0, -0.5))
    assert (cell["gain"], cell["clean"], cell["joint"], cell["execution_s"]) == (20.0, -0.5, 19.5, 1.5)
    assert R.program_cell(_prog("s", "i101", 0.0, 0.0, n_changed=0))["status"] == R.STATUS_BASE
    over = R.program_cell(_prog("s", "i101", 30.0, -3.0, within=False))
    assert over["status"] == R.STATUS_LIMIT and over["gain"] is None and over["unbounded_gain"] == 30.0
    assert R.program_cell(_prog("s", "i101", 0, 0, status="program_failed", run_status="timeout"))["status"] == \
        R.STATUS_LIMIT
    assert R.program_cell(_prog("s", "i101", 0, 0, status="program_failed", run_status="error"))["status"] == \
        R.STATUS_ERROR
    patch = _prog("s", "i101", 0, 0, status="patch_failed")
    patch["n_changed"] = None
    assert R.program_cell(patch)["status"] == R.STATUS_LIMIT
    assert R.copy_cell(_copy("s", "i102", -0.25, -0.5))["joint"] == -0.75


def test_the_diagonal_gate_needs_the_sessions_child() -> None:
    source = _source("sa", 100)
    assert R.diagonal(source, _prog("sa", "i100", 30.0, 0.0, child="child", deltas="deltas"))["passed"] is True
    gate = R.diagonal(source, _prog("sa", "i100", 30.0, 0.0, child="other", deltas="deltas"))
    assert gate["passed"] is False and gate["same_child"] is False and gate["same_deltas"] is True
    last_bits = R.diagonal(source, _prog("sa", "i100", 30.0, 0.0, child="child", deltas="x"))
    assert last_bits["passed"] is True and last_bits["same_deltas"] is False  # recorded, not gated
    other_program = _prog("sa", "i100", 30.0, 0.0, child="child", deltas="deltas")
    other_program["program"]["sha256"] = "pother"
    assert R.diagonal(source, other_program)["passed"] is False
    failed = _prog("sa", "i100", 0, 0, status="program_failed", run_status="error")
    assert R.diagonal(source, failed)["passed"] is False
    with pytest.raises(R.ReuseMatrixError, match="diagonal record"):
        R.diagonal(source, _prog("sa", "i101", 30.0, 0.0))


def _records(gains: dict[str, float], cleans: dict[str, float], copy_gain: float, sid: str, own: str,
             own_child: str = "child") -> tuple[dict[str, Any], dict[str, Any]]:
    program = {t: _prog(sid, t, gains[t], cleans[t], child=own_child if t == own else "c",
                        deltas="deltas" if t == own else "d") for t in IDS}
    copy = {t: _copy(sid, t, copy_gain, -0.5) for t in IDS if t != own}
    return program, copy


@pytest.mark.usefixtures("one_condition")
def test_the_fixed_readings() -> None:
    sources = {"sa": _source("sa", 100), "sb": _source("sb", 101, child="childb")}
    # sa: own gain 20; non-own gains 16, 14, 13, 15, 12 (ratios 0.8, 0.7, 0.65, 0.75, 0.6), cleans -0.5, -1.0, -1.25,
    # 0, -2 (three >= -1.0)
    ga = {"i100": 20.0, "i101": 16.0, "i102": 14.0, "i103": 13.0, "i104": 15.0, "i105": 12.0}
    ca = {"i100": 0.25, "i101": -0.5, "i102": -1.0, "i103": -1.25, "i104": 0.0, "i105": -2.0}
    pa, cpa = _records(ga, ca, -0.25, "sa", "i100")
    # sb: own gain 10 on i101; every non-own gain 9, clean 0; one non-own cell over the limits (counted as the base)
    gb = {t: 9.0 for t in IDS} | {"i101": 10.0}
    cb = {t: 0.0 for t in IDS}
    pb, cpb = _records(gb, cb, 1.0, "sb", "i101", own_child="childb")
    pb["i105"] = _prog("sb", "i105", 25.0, -4.0, within=False)
    result = R.build(sources, {"sa": pa, "sb": pb}, {"sa": cpa, "sb": cpb})
    cond = result["conditions"][COND]
    assert cond["read_sources"] == ["sa", "sb"]
    a, b = cond["source_readings"]
    assert a["own_gain"] == 20.0 and a["program_mean_gain"] == pytest.approx(14.0)
    assert a["reuse_share"] == pytest.approx(0.7) and a["program_clean_ok"] == 3
    assert a["c_e3_form"] is False  # share 0.7 meets the ratio, clean only 3 of 5
    assert a["c_per_target_targets"] == ["i101", "i102", "i104"] and a["c_per_target"] is False
    assert a["copy_mean_gain"] == pytest.approx(-0.25) and a["program_minus_copy_gain"] == pytest.approx(14.25)
    assert a["program_mean_clean"] == pytest.approx(-0.95)
    assert a["program_minus_copy_joint"] == pytest.approx((14.0 - 0.95) - (-0.25 - 0.5))
    assert b["own_seed"] == 101 and b["counted_as_base"]["program"] == ["i105"]
    assert b["program_mean_gain"] == pytest.approx(36.0 / 5) and b["program_all_within_limits"] is False
    assert b["c_e3_form"] is False and b["c_per_target_n"] == 4 and b["c_per_target"] is True
    assert cond["rows"]["sb"]["program"]["i105"]["status"] == R.STATUS_LIMIT
    assert "i101" not in cond["rows"]["sb"]["copy"] and "i100" in cond["rows"]["sb"]["copy"]
    t100, t101 = cond["target_readings"][:2]
    assert t100["program_all_sources"] == {"n": 2, "gain": 14.5, "clean": 0.125, "joint": 14.625}
    assert t100["program_non_own"]["n"] == 1 and t100["program_non_own"]["gain"] == 9.0
    assert t100["copy_non_own"] == {"n": 1, "gain": 1.0, "clean": -0.5, "joint": 0.5}
    assert t101["program_non_own"] == {"n": 1, "gain": 16.0, "clean": -0.5, "joint": 15.5}


@pytest.mark.usefixtures("one_condition")
def test_a_failed_gate_leaves_the_row_unread_and_a_missing_cell_is_an_error() -> None:
    sources = {"sa": _source("sa", 100), "sb": _source("sb", 101, child="childb")}
    gains = {t: 10.0 for t in IDS}
    cleans = {t: 0.0 for t in IDS}
    pa, cpa = _records(gains, cleans, 0.0, "sa", "i100", own_child="drifted")
    pb, cpb = _records(gains, cleans, 0.0, "sb", "i101", own_child="childb")
    cond = R.build(sources, {"sa": pa, "sb": pb}, {"sa": cpa, "sb": cpb})["conditions"][COND]
    assert cond["read_sources"] == ["sb"] and cond["diagonal_gates"]["sa"]["passed"] is False
    assert cond["source_readings"][0]["read"] is False and "reuse_share" not in cond["source_readings"][0]
    assert cond["target_readings"][0]["program_all_sources"]["n"] == 1  # sa's cells are not read
    del cpb["i103"]
    with pytest.raises(R.ReuseMatrixError, match="no record"):
        R.build(sources, {"sa": pa, "sb": pb}, {"sa": cpa, "sb": cpb})
    cpb["i103"] = _copy("sb", "i103", 0.0, 0.0, deltas="other")
    with pytest.raises(R.ReuseMatrixError, match="not the session's"):
        R.build(sources, {"sa": pa, "sb": pb}, {"sa": cpa, "sb": cpb})


def test_the_fixed_targets_are_checked_against_the_store() -> None:
    lookup = {(c, s): R.TARGETS[c][i] for c in R.CONDITIONS for i, s in enumerate(R.TARGET_SEEDS)}
    assert R.check_targets(lambda c, s: lookup[(c, s)]) == {c: list(R.TARGETS[c]) for c in R.CONDITIONS}
    lookup[(R.CONTRAST, 103)] = "i00000000"
    with pytest.raises(R.ReuseMatrixError, match="not PREREG T08's"):
        R.check_targets(lambda c, s: lookup[(c, s)])


def _session_with_program(root: Path, e3_record: dict[str, Any], source_id: str, program: Path, seed: int) -> Path:
    """``_fake_session`` with the program file in final/ and the session's seed (what T08 reads)."""
    session = _fake_session(root, e3_record, source_id)
    shutil.copyfile(program, session / "final" / "qkv.py")
    run = read_json(session / "run.json")
    run["final"]["program"].update(file="final/qkv.py", sha256=file_sha256(program))
    run["final"]["n_changed"] = e3_record["n_changed"]
    write_json(session / "run.json", run)
    write_json(session / "session.json", dict(read_json(session / "session.json"), seed=seed))
    return session


@pytest.fixture
def toy_runs(e11_store: V.SyntheticStore, tmp_path: Path) -> dict[str, Any]:
    """E3 of the qkv program on the six toy students, and sessions made from its seed-100 and seed-101 runs."""
    ids = [e11_store.ids[(PLANT_CONDITION, seed)] for seed in R.TARGET_SEEDS]
    program = tmp_path / "qkv.py"
    program.write_text(QKV_Q, encoding="utf-8")
    written = run_transfer(program, ids, tmp_path / "transfer", _sandbox())
    e3 = {r["target"]["instance_id"]: r for r in (json.loads(p.read_text()) for p in written)}
    sessions = tmp_path / "runs_sessions"
    s100 = _session_with_program(sessions / "s1000000000000000", e3[ids[0]], ids[0], program, 100)
    s101 = _session_with_program(sessions / "s1010000000000000", e3[ids[1]], ids[1], program, 101)
    return {"ids": ids, "e3": e3, "program": program, "sessions": sessions, "s100": s100, "s101": s101}


@needs_sandbox
def test_a_seed_101_source_copies_onto_the_other_five_students(toy_runs: dict[str, Any], tmp_path: Path) -> None:
    ids, session = toy_runs["ids"], toy_runs["s101"]
    others = [ids[0], *ids[2:]]
    written = F.run_frozen(session, others, tmp_path / "copy")
    records = {r["target"]["instance_id"]: r for r in (read_json(p) for p in written)}
    assert sorted(records) == sorted(others)
    name = "classifier.blocks.0.self_attn.qkv.weight"
    delta = np.load(session / "final" / "deltas.npz")[name]
    for target, record in records.items():
        assert record["source"]["source_instance_id"] == ids[1]
        assert record["source"]["child_sha256"] == toy_runs["e3"][ids[1]]["child_sha256"]
        base = fewshot.load_any(target).params
        child = dict(base, **{name: (base[name].astype(np.float64) + delta).astype(np.float32)})
        assert record["status"] == "ok" and record["child_sha256"] == params_sha256(child)
        assert record["child_sha256"] != toy_runs["e3"][target]["child_sha256"]


@needs_sandbox
def test_the_t08_path_on_the_toy_store(toy_runs: dict[str, Any], tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    ids, program = toy_runs["ids"], toy_runs["program"]
    monkeypatch.setattr(R, "CONDITIONS", (PLANT_CONDITION,))
    monkeypatch.setattr(R, "TARGETS", {PLANT_CONDITION: tuple(ids)})
    monkeypatch.setattr(R, "SOURCES", {PLANT_CONDITION: ("s1000000000000000", "s1010000000000000")})
    sources = R.load_sources(toy_runs["sessions"])
    cells = R.cell_list(sources)
    assert [c["arm"] for c in cells].count("program") == 12 and [c["arm"] for c in cells].count("copy") == 10
    root = tmp_path / "t08"
    for sid, source in sources.items():
        run_transfer(Path(str(source.program_path)), ids, root / "program" / sid, _sandbox())
        F.run_frozen(toy_runs["sessions"] / sid, [t for t in ids if t != source.instance_id], root / "copy" / sid)
    result = R.table(root, toy_runs["sessions"], e3_root=None, f6_root=None)
    cond = result["conditions"][PLANT_CONDITION]
    assert cond["read_sources"] == ["s1000000000000000", "s1010000000000000"]
    assert all(gate["passed"] for gate in cond["diagonal_gates"].values())
    rows = cond["rows"]
    for target in ids:  # one program, so both rows rerun it identically
        a, b = rows["s1000000000000000"]["program"][target], rows["s1010000000000000"]["program"][target]
        assert a["child_sha256"] == b["child_sha256"] == toy_runs["e3"][target]["child_sha256"]
        assert a["gain"] == b["gain"] and a["status"] in R.READ_STATUSES
    reading = cond["source_readings"][1]
    others = [t for t in ids if t != ids[1]]
    gains = [rows["s1010000000000000"]["program"][t]["gain"] for t in others]
    assert reading["own_seed"] == 101 and reading["program_mean_gain"] == pytest.approx(np.mean(gains), abs=1e-4)
    copy_gains = [rows["s1010000000000000"]["copy"][t]["gain"] for t in others]
    assert reading["program_minus_copy_gain"] == pytest.approx(np.mean(gains) - np.mean(copy_gains), abs=1e-4)
