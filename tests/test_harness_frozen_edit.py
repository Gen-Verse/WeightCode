"""PREREG F6 (``weightcode.harness.frozen_edit``) on the toy E11 store: a frozen edit read as E3 reads a rerun program.

A zero delta reads as 0 / 0. The seed-100 student with its own program's deltas gives E3's child and E3's readouts
exactly (the frozen path is E3's without the program run). On the other students the frozen deltas are the source's
values, not a rerun. The source checks refuse a session whose model or deltas record does not match, and ``compare``
applies PREREG F6's fixed reading. No test touches the real store.
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

from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.harness import e11, fewshot
from weightcode.harness import frozen_edit as F
from weightcode.harness.jsonio import file_sha256, read_json, write_json
from weightcode.harness.transfer import run_transfer
from weightcode.models import vit_synth as V


def _fake_session(root: Path, e3_record: dict[str, Any], source_id: str) -> Path:
    """A session directory whose final is E3's run of the program on its own student (the layout run.json names)."""
    work = Path(e3_record["work_dir"])
    final = root / "final"
    final.mkdir(parents=True)
    shutil.copyfile(work / "run" / "out" / "deltas.npz", final / "deltas.npz")
    shutil.copyfile(work / "patch" / "derived_patch.json", final / "patch.json")
    for delta in (work / "patch").glob("delta_*.npy"):
        shutil.copyfile(delta, final / delta.name)
    instance = fewshot.load_any(source_id)
    patch = read_json(final / "patch.json")
    child = {name: value.copy() for name, value in instance.params.items()}
    for op in patch["ops"]:
        delta = np.load(final / op["delta"])
        child[op["tensor"]] = (instance.params[op["tensor"]].astype(np.float64) + delta).astype(np.float32)
    assert save_params(final / "model.npz", child) == e3_record["child_sha256"]
    digest = file_sha256(final / "deltas.npz")
    assert digest == e3_record["deltas_sha256"]
    write_json(final / "program_run.json", {"deltas_sha256": digest})
    write_json(root / "session.json", {"arm": "LEARN", "instance_id": source_id, "condition": PLANT_CONDITION,
                                       "limits": e3_record["limits"]})
    write_json(root / "run.json", {"status": "complete", "final": {
        "kind": "checked", "dir": "final", "model": "final/model.npz", "parent": "instance", "is_base_noop": False,
        "child_sha256": e3_record["child_sha256"],
        "program": {"deltas": "final/deltas.npz", "deltas_sha256": digest, "patch": "final/patch.json",
                    "run_record": "final/program_run.json", "file": "final/qkv.py", "sha256": "0" * 64}}})
    return root


@pytest.fixture
def toy_e3(e11_store: V.SyntheticStore, tmp_path: Path) -> tuple[list[str], dict[str, dict[str, Any]], Path]:
    """E3 on the toy store (a qkv program, whose deltas depend on each student's weights; seed 100 and its five
    targets) and a session made from its seed-100 run."""
    source = e11_store.ids[(PLANT_CONDITION, 100)]
    ids = [source, *e11.transfer_targets(source)]
    program = tmp_path / "qkv.py"
    program.write_text(QKV_Q, encoding="utf-8")
    written = run_transfer(program, ids, tmp_path / "transfer", _sandbox())
    records = {r["target"]["instance_id"]: r for r in (json.loads(p.read_text()) for p in written)}
    session = _fake_session(tmp_path / "sessions" / "s0123456789abcdef", records[source], source)
    return ids, records, session


def test_a_zero_delta_reads_zero_on_every_part(e11_store: V.SyntheticStore, tmp_path: Path) -> None:
    target = fewshot.load_any(e11_store.ids[(PLANT_CONDITION, 101)])
    name = "classifier.norm.weight"
    deltas_path = tmp_path / "deltas.npz"
    np.savez(deltas_path, **{name: np.zeros(target.params[name].shape)})
    edit = F.FrozenEdit(session_id="s0", source_instance_id="i0", deltas_path=deltas_path,
                        deltas_sha256=file_sha256(deltas_path), hypothesis="h", predicted_effect="p", record={})
    record = F.frozen_one(edit, target, tmp_path / "work")
    assert record["status"] == "ok" and record["within_limits"] is True and record["n_changed"] == 0
    assert record["child_sha256"] == target.params_sha256
    for part in (record["readouts"], record["readouts_clean"]):
        assert part["delta_acc"] == 0.0 and part["fixed"] == 0 and part["broken"] == 0 and part["n"] == 160
        assert all(value == 0.0 for value in part["per_class_acc_delta"])


@needs_sandbox
def test_the_own_student_reproduces_e3_and_the_others_get_the_source_values(
        toy_e3: tuple[list[str], dict[str, dict[str, Any]], Path], tmp_path: Path) -> None:
    ids, e3, session = toy_e3
    written = F.run_frozen(session, ids, tmp_path / "f6")
    records = {r["target"]["instance_id"]: r for r in (read_json(p) for p in written)}
    source = ids[0]
    own, rerun = records[source], e3[source]
    assert own["child_sha256"] == rerun["child_sha256"] and own["patch_sha256"] == rerun["patch_sha256"]
    assert own["readouts"] == rerun["readouts"] and own["readouts_clean"] == rerun["readouts_clean"]
    assert own["limits"] == rerun["limits"] and own["readout_source"] == rerun["readout_source"]
    assert own["source"]["child_sha256"] == rerun["child_sha256"]
    name = "classifier.blocks.0.self_attn.qkv.weight"
    source_delta = np.load(session / "final" / "deltas.npz")[name]
    for target in ids[1:]:
        record = records[target]
        instance = fewshot.load_any(target)
        assert record["status"] == "ok" and record["tensors"] == [name]
        assert record["child_sha256"] != e3[target]["child_sha256"]  # the rerun recomputes; F6 does not
        base = instance.params[name]
        child = dict(instance.params, **{name: (base.astype(np.float64) + source_delta).astype(np.float32)})
        assert record["child_sha256"] == params_sha256(child)
    seeds = {target: int(records[target]["target"]["seed"]) for target in ids}
    result = F.compare(list(records.values()), list(e3.values()), seeds)
    assert result["seed100_gate"]["passed"] is True and result["n_unseen"] == 5
    assert [row["seed"] for row in result["rows"]] == [100, 101, 102, 103, 104, 105]
    with pytest.raises(F.FrozenEditError, match="never overwritten"):
        F.run_frozen(session, ids, tmp_path / "f6")


@needs_sandbox
def test_the_source_checks_refuse_a_model_or_a_record_that_does_not_match(
        toy_e3: tuple[list[str], dict[str, dict[str, Any]], Path], tmp_path: Path) -> None:
    ids, e3, session = toy_e3
    edit = F.session_edit(session, tmp_path / "ok")
    assert edit.record["model_sha256"] == edit.record["child_sha256"] == e3[ids[0]]["child_sha256"]
    model = load_params(session / "final" / "model.npz")
    bad = dict(model, **{"classifier.fc.bias": model["classifier.fc.bias"] + np.float32(1e-3)})
    moved = tmp_path / "moved"
    shutil.copytree(session, moved)
    save_params(moved / "final" / "model.npz", bad)
    with pytest.raises(F.FrozenEditError, match="is not float32"):
        F.session_edit(moved, tmp_path / "bad_model")
    other = tmp_path / "other"
    shutil.copytree(session, other)
    run = read_json(other / "run.json")
    run["final"]["program"]["deltas_sha256"] = "f" * 64
    write_json(other / "run.json", run)
    with pytest.raises(F.FrozenEditError, match="differs from run.json"):
        F.session_edit(other, tmp_path / "bad_record")
    variant = tmp_path / "variant"
    shutil.copytree(session, variant)
    write_json(variant / "session.json", dict(read_json(session / "session.json"), e11_variant="rule"))
    with pytest.raises(F.FrozenEditError, match="standard LEARN session"):
        F.session_edit(variant, tmp_path / "bad_variant")


def _record(target: str, seed: int, gain: float, clean: float) -> dict[str, Any]:
    return {"status": "ok", "target": {"instance_id": target, "seed": seed}, "within_limits": True,
            "readouts": {"delta_acc": gain}, "readouts_clean": {"delta_acc": clean}, "n_changed": 1,
            "child_sha256": f"{target}{gain}"}


@pytest.mark.parametrize(("frozen", "reading"), [(0.04, F.READ_PROGRAM), (0.05, F.READ_PROGRAM),
                                                  (0.07, F.READ_BOTH), (0.09, F.READ_VALUES), (0.12, F.READ_VALUES)])
def test_the_fixed_reading_of_the_share(frozen: float, reading: str) -> None:
    """Rerun gain 0.1 on each unseen student: shares 0.4, 0.5 (boundary), 0.7, 0.9 (boundary) and 1.2."""
    seeds = {"i100": 100, **{f"i{s}": s for s in range(101, 106)}}
    rerun = [_record("i100", 100, 0.2, 0.0), *(_record(f"i{s}", s, 0.1, -0.001) for s in range(101, 106))]
    f6 = [_record("i100", 100, 0.2, 0.0), *(_record(f"i{s}", s, frozen, -0.002) for s in range(101, 106))]
    result = F.compare(f6, rerun, seeds)
    assert result["reading"] == reading and result["n_unseen"] == 5
    assert result["seed100_gate"]["passed"] is True
    assert result["rerun_mean_gain"] == pytest.approx(10.0) and result["share"] == pytest.approx(frozen / 0.1)
