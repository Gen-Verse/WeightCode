"""Tests for the sealed read of a list of finals and its dev dry run (weightcode.evalkit.sealed_finals, vision_joint).

Toy store: the evalkit toy store with the condition toyblur_s2 and its dev instances. A final is a hand-written session
(run.json, session.json with its limits, final/patch.json) or a one-round loop-stream record; a reference is a
hand-written joint baseline run with two checks whose kept pick (S_old >= -20) is not its own final. The dev
references are written with the readouts of record (``devreadout.dev_readout_joint`` for a session's info.json,
``program_stream._readout`` for a stream, ``devreadout.readout_retention`` for a reference's dev record), so a dev
record must equal them field for field. The scan_em and policy_success families have their own tests
(test_evalkit_sealed_scan.py, test_evalkit_sealed_e14.py).
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_evalkit_instances_store import (  # noqa: F401
    CONDITION,
    ToyStore,
    add_condition,
    breaking_child,
    build_condition,
    condition_dir,
    condition_test_path,
    fixing_child,
    load_npz,
    toy_store,
    write_condition_meta,
)

from weightcode.baselines.instance_io import Instance, load_instance, write_delta_patch
from weightcode.common.identity import params_sha256, save_params
from weightcode.evalkit import devreadout
from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.sealed import read_registry
from weightcode.evalkit.targets import file_sha256
from weightcode.harness.program_stream import _readout as stream_readout
from weightcode.patch import Limits, task_limits

SESSION = "s0123456789abcdef"
OTHER_SESSION = "sfedcba9876543210"
STREAM = "loop_streams/loop_teacher_toy.json"
F4_STREAM = "f4/e10/ctrlb_f4_ft_replay_toy.json"
REF_RUN = "baselines/{instance}/toy_ref/e11g2/0/run.json"
REF_RECORD = "devreadouts/toy_refs_floor20.jsonl"
OWN_RECORD = "devreadouts/toy_refs_own.jsonl"
FRONTIER_RECORD = "devreadouts/toy_refs_frontier.jsonl"
STREAM2 = "loop_streams/loop_teacher_two_rounds.json"
TRANSFER = f"devreadouts/learn/{SESSION}/transfer/transfer_0123456789ab_{{instance}}.json"
T08_TRANSFER = f"t08/program/{SESSION}/transfer_0123456789ab_{{instance}}.json"
COMMIT = "0" * 40


@dataclasses.dataclass
class Toy:
    store: ToyStore
    instance: Instance
    other: Instance
    session_child: dict[str, np.ndarray]
    stream_child: dict[str, np.ndarray]
    kept_child: dict[str, np.ndarray]
    own_child: dict[str, np.ndarray]


def _session(store: ToyStore, session_id: str, instance: Instance, child: dict[str, np.ndarray]) -> None:
    """A finished LEARN-like session: run.json, session.json with permissive limits, final/patch.json, info.json."""
    root = store.runs / "sessions" / session_id
    write_delta_patch(root / "final", instance.params_sha256, instance.params, child, "toy", "toy")
    limits = dataclasses.asdict(Limits(max_rel_frob=None, max_edit_frac=None))
    (root / "session.json").write_text(json.dumps({"limits": limits}))
    run = {"session_id": session_id, "status": "complete", "session_result": "session_result.json",
           "instance": {"instance_id": instance.instance_id, "params_sha256": instance.params_sha256},
           "final": {"is_base_noop": False, "dir": "final", "parent": "instance", "patch_id": "toy.py",
                     "query_index": 0, "n_changed": 1, "child_sha256": params_sha256(child)}}
    (root / "run.json").write_text(json.dumps(run))
    joint = devreadout.dev_readout_joint(instance, child)
    info = {"instance_id": instance.instance_id,
            "final": {"child_sha256": params_sha256(child), "vselect": joint["readouts"],
                      "vselect_clean": joint["readouts_clean"]}}
    reference = store.runs / "devreadouts" / "learn" / session_id / "info.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps(info))


def _stream(store: ToyStore, instance: Instance, child: dict[str, np.ndarray], rel: str = STREAM,
            schema: str = "weightcode.harness.loop_stream.v0") -> None:
    """A completed one-round stream record at ``rel`` whose student after round 1 is ``child``: a loop stream, or with
    ``schema`` a fine-tuning stream under the same loop (PREREG F4, the same round fields)."""
    work = store.runs / rel.removesuffix(".json")
    work.mkdir(parents=True)
    digest = save_params(work / "student_after_round_1.npz", child)
    record = {"schema": schema, "base_params_sha256": instance.params_sha256,
              "instances": [{"instance_id": instance.instance_id, "condition": CONDITION, "round": 1}],
              "rounds": [{"round": 1, "status": "ok", "condition": CONDITION, "child_sha256": digest,
                          "student_after": {"path": str(work / "student_after_round_1.npz"), "sha256": digest},
                          "readouts": {instance.instance_id: stream_readout(instance, child, 1024)}}]}
    (store.runs / rel).write_text(json.dumps(record))


def _reference(store: ToyStore, instance: Instance, kept: dict[str, np.ndarray], own: dict[str, np.ndarray]) -> None:
    """A joint baseline run with two checks: q0 (kept at S_old >= -20) and q1 (its own final, S_old -30), and the
    dev record of its kept pick, written by ``devreadout.readout_retention``."""
    run_path = store.runs / REF_RUN.format(instance=instance.instance_id)
    log, candidates = [], []
    for query, (child, score_new, score_old) in enumerate(((kept, 10, 0), (own, 50, -30))):
        patch_id = f"q{query}"
        write_delta_patch(run_path.parent / patch_id, instance.params_sha256, instance.params, child, "toy", "toy")
        n_changed = sum(int(np.count_nonzero(child[k] != instance.params[k])) for k in child)
        log.append({"patch_id": patch_id, "query_index": query, "child_sha256": params_sha256(child),
                    "n_changed": n_changed, "score": score_new + score_old, "score_new": score_new,
                    "score_old": score_old, "fixed": score_new, "broken": 0})
        candidates.append({"patch_id": patch_id, "config": {"q": query}, "edit": {"edit_size": n_changed}})
    own_entry = log[1]
    run = {"method": "toy_ref",
           "instance": {"instance_id": instance.instance_id, "params_sha256": instance.params_sha256,
                        "forward_sha256": instance.forward_sha256},
           "flags": {"edit_limits_enforced": False}, "oracle_log": log, "candidates": candidates,
           "final": {"patch_id": "q1", "is_base_noop": False, "dir": "q1", "query_index": 1,
                     "child_sha256": own_entry["child_sha256"], "edit_size": own_entry["n_changed"],
                     **{key: own_entry[key] for key in ("score", "fixed", "broken", "score_new", "score_old")}}}
    run_path.write_text(json.dumps(run))
    record = devreadout.readout_retention(run_path, s_old_floor=-20)
    (store.runs / REF_RECORD).write_text(json.dumps(devreadout.json_safe(record)) + "\n")
    own_record = devreadout.readout_run(run_path, clean=True)
    (store.runs / OWN_RECORD).write_text(json.dumps(devreadout.json_safe(own_record)) + "\n")
    frontier = devreadout.readout_candidates(run_path)
    (store.runs / FRONTIER_RECORD).write_text("".join(json.dumps(devreadout.json_safe(r)) + "\n" for r in frontier))


@pytest.fixture
def toy(toy_store: ToyStore) -> Toy:  # noqa: F811
    add_condition(toy_store)
    outcomes = [o for o in build_condition(toy_store, roles=("dev",)) if o.split_role == "dev"]
    by_seed = {o.seed: load_instance(instances_root() / o.instance_id) for o in outcomes}
    instance, other = by_seed[100], by_seed[101]
    session_child = fixing_child(dict(instance.params))
    stream_child = breaking_child(dict(instance.params))
    kept_child = fixing_child(dict(instance.params))
    kept_child["fc1.bias"][0] += 0.25  # a different child that still fixes
    own_child = breaking_child(dict(instance.params))
    own_child["fc2.bias"][1] += 0.5
    _session(toy_store, SESSION, instance, session_child)
    _stream(toy_store, instance, stream_child)
    _reference(toy_store, instance, kept_child, own_child)
    return Toy(toy_store, instance, other, session_child, stream_child, kept_child, own_child)


def _final(name: str, source: dict[str, Any], instance_id: str, child: dict[str, np.ndarray]) -> dict[str, Any]:
    return {"name": name, "readout": "vision_joint", "source": source, "instance_id": instance_id,
            "child_sha256": params_sha256(child)}


def _finals(toy: Toy) -> list[dict[str, Any]]:
    return [_final("session_final", {"kind": "session", "session_id": SESSION}, toy.instance.instance_id,
                   toy.session_child),
            _final("stream_final", {"kind": "stream_final", "record": STREAM}, toy.instance.instance_id,
                   toy.stream_child)]


def _reference_source(toy: Toy, floor: int = -20) -> dict[str, Any]:
    return {"kind": "reference", "run": REF_RUN.format(instance=toy.instance.instance_id), "pick": "kept",
            "s_old_floor": floor, "dev_record": REF_RECORD}


def _references(toy: Toy) -> list[dict[str, Any]]:
    return [{**_final("ref_kept", _reference_source(toy), toy.instance.instance_id, toy.kept_child),
             "pairs_with": "session_final"}]


def _pick_source(toy: Toy, pick: str, patch_id: str | None = None) -> dict[str, Any]:
    source = {"kind": "reference", "run": REF_RUN.format(instance=toy.instance.instance_id), "pick": pick,
              "dev_record": OWN_RECORD if pick == "own" else FRONTIER_RECORD}
    if patch_id is not None:
        source["patch_id"] = patch_id
    return source


def _two_round_stream(store: ToyStore, instance: Instance, first: dict[str, np.ndarray],
                      second: dict[str, np.ndarray]) -> None:
    """A completed two-round stream on one student whose rounds are both read on ``instance`` (round 2's condition
    instance is a stand-in id: a stream reads every round on every instance, and only ``instance`` is listed)."""
    work = store.runs / STREAM2.removesuffix(".json")
    work.mkdir(parents=True)
    rounds = []
    for number, child in ((1, first), (2, second)):
        path = work / f"student_after_round_{number}.npz"
        digest = save_params(path, child)
        rounds.append({"round": number, "status": "ok", "condition": CONDITION, "child_sha256": digest,
                       "student_after": {"path": str(path), "sha256": digest},
                       "readouts": {instance.instance_id: stream_readout(instance, child, 1024)}})
    record = {"schema": "weightcode.harness.loop_stream.v0", "base_params_sha256": instance.params_sha256,
              "instances": [{"instance_id": instance.instance_id, "condition": CONDITION, "round": 1},
                            {"instance_id": "i00000000", "condition": CONDITION, "round": 2}],
              "rounds": rounds}
    (store.runs / STREAM2).write_text(json.dumps(record))


def _transfer(store: ToyStore, target: Instance, child: dict[str, np.ndarray], layout: str = TRANSFER) -> str:
    """A transfer record (weightcode.harness.program_transfer.v0) of a session's program on ``target``: the work directory
    of the record's stem with patch/derived_patch.json, the readouts of record (``clean.dev_readout_joint``'s
    fields), and the program file under the session's final/. ``layout`` is E3's (devreadouts/learn/<id>/transfer/)
    or T08's program arm (t08/program/<id>/)."""
    rel = layout.format(instance=target.instance_id)
    path = store.runs / rel
    work = path.with_suffix("")
    patch = write_delta_patch(work / "patch", target.params_sha256, target.params, child, "toy", "toy")
    patch.rename(work / "patch" / SF.TRANSFER_PATCH)
    program = store.runs / "sessions" / SESSION / "final" / "toy_program.py"
    program.write_text("def update(student, experience):\n    return {}\n")
    limits = task_limits(Limits(max_rel_frob=None, max_edit_frac=None))
    joint = devreadout.dev_readout_joint(target, child)
    record = {"schema": SF.TRANSFER_SCHEMA, "status": "ok", "within_limits": True, "readout_child": "within_limits",
              "program": {"path": str(program), "name": program.name, "sha256": file_sha256(program)},
              "target": {"instance_id": target.instance_id, "params_sha256": target.params_sha256},
              "limits": limits, "work_dir": str(work), "patch_sha256": file_sha256(work / "patch" / SF.TRANSFER_PATCH),
              "child_sha256": params_sha256(child), "n_select": joint["n"], "readouts": joint["readouts"],
              "n_clean": joint["n_clean"], "readouts_clean": joint["readouts_clean"]}
    path.write_text(json.dumps(devreadout.json_safe(record)))
    return rel


def _write_list(path: Path, finals: list[dict[str, Any]], status: str = "draft", approval: str | None = None,
                note: str = "toy", references: list[dict[str, Any]] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": SF.LIST_SCHEMA, "status": status, "approval": approval, "note": note,
                                "finals": finals, "references": references or []}, indent=1))
    return path


def _git_state(path: Path, dirty: bool = False) -> Path:
    path.write_text(json.dumps({"commit": COMMIT, "dirty": dirty}))
    return path


# ------------------------------------------------------------------------------------------------------ dev mode


def test_dev_equals_the_readouts_of_record(toy: Toy, tmp_path: Path) -> None:
    listed = _write_list(tmp_path / "list.json", _finals(toy))
    records = SF.run_dev(listed, tmp_path / "dev.jsonl", code_state_path=_git_state(tmp_path / "git_state.json"))
    assert [r["name"] for r in records] == ["session_final", "stream_final"]
    for record, child in zip(records, (toy.session_child, toy.stream_child)):
        joint = devreadout.dev_readout_joint(toy.instance, child)
        assert record["readouts"] == devreadout.json_safe(joint["readouts"])
        assert record["readouts_clean"] == devreadout.json_safe(joint["readouts_clean"])
        assert record["reference"]["match"] and record["reference"]["differences"] == []
        assert record["gain"] == 100 * joint["readouts"]["delta_acc"]
        assert record["clean"] == 100 * joint["readouts_clean"]["delta_acc"]
        assert record["joint"] == record["gain"] + record["clean"]
        assert record["child_sha256"] == params_sha256(child) and record["mode"] == "dev"
        assert record["rows"]["condition"]["split"] == "V_select"
        assert record["code"]["commit"] == COMMIT and set(record["code"]["files"]) == set(SF.CODE_MODULES)
    assert records[0]["reference"]["fields"] == "all"
    assert records[0]["gain"] > 0, "the fixing child gains on V_select"
    assert records[1]["clean"] < 0, "the breaking child loses clean rows"
    index = records[0]["sealed_index"]
    assert index["condition_test"]["format"] == SF.FORMAT_STORE_NPZ
    assert index["condition_test"]["path"] == str(condition_test_path(toy.store))
    assert index["clean_test"]["path"] == str(toy.store.sealed / toy.store.dataset / "test.npz")
    assert index["condition_test"]["sha256_kind"] == index["clean_test"]["sha256_kind"] == SF.HASH_CONTENT
    written = [json.loads(line) for line in (tmp_path / "dev.jsonl").read_text().splitlines()]
    assert [w["name"] for w in written] == ["session_final", "stream_final"]
    with pytest.raises(SF.SealedFinalsError, match="never overwritten"):
        SF.run_dev(listed, tmp_path / "dev.jsonl", code_state_path=tmp_path / "git_state.json")


def test_a_fine_tuning_stream_is_a_reference_read_as_its_record(toy: Toy, tmp_path: Path) -> None:
    _stream(toy.store, toy.instance, toy.own_child, F4_STREAM, "weightcode.baselines.checked_reference_stream.v0")
    ref = {**_final("ref_f4", {"kind": "stream_final", "record": F4_STREAM}, toy.instance.instance_id, toy.own_child),
           "pairs_with": "stream_final"}
    listed = _write_list(tmp_path / "list.json", _finals(toy), references=[ref])
    records = SF.run_dev(listed, tmp_path / "dev.jsonl", code_state_path=_git_state(tmp_path / "git_state.json"))
    record = records[2]
    assert (record["name"], record["block"], record["child"]["kind"]) == ("ref_f4", "references", "stream_final")
    joint = devreadout.dev_readout_joint(toy.instance, toy.own_child)
    assert record["readouts"] == devreadout.json_safe(joint["readouts"])
    assert record["reference"]["match"] and record["reference"]["fields"] == SF.FIELDS_THEIRS
    assert record["pairs_with"] == "stream_final"
    record_path = toy.store.runs / F4_STREAM
    changed = json.loads(record_path.read_text())
    changed["schema"] = "weightcode.baselines.some_other_stream.v0"
    record_path.write_text(json.dumps(changed))
    with pytest.raises(SF.SealedFinalsError, match="has schema"):
        SF.run_dev(listed, tmp_path / "again.jsonl", code_state_path=tmp_path / "git_state.json")


def test_dev_reads_the_kept_pick_of_a_reference_as_its_record(toy: Toy, tmp_path: Path) -> None:
    listed = _write_list(tmp_path / "list.json", _finals(toy), references=_references(toy))
    records = SF.run_dev(listed, tmp_path / "dev.jsonl", code_state_path=_git_state(tmp_path / "git_state.json"))
    assert [(r["name"], r["block"]) for r in records] == [("session_final", "finals"), ("stream_final", "finals"),
                                                          ("ref_kept", "references")]
    ref = records[2]
    expected = devreadout.readout_retention(toy.store.runs / REF_RUN.format(instance=toy.instance.instance_id),
                                            s_old_floor=-20)
    assert ref["readouts"] == devreadout.json_safe(expected["readouts"])
    assert ref["readouts_clean"] == devreadout.json_safe(expected["readouts_clean"])
    assert ref["reference"]["match"] and ref["reference"]["fields"] == "all"
    assert ref["child"]["patch_id"] == "q0" and ref["child"]["selection"]["primary_patch_id"] == "q1"
    assert ref["pairs_with"] == "session_final" and ref["child_sha256"] == params_sha256(toy.kept_child)
    assert ref["environment"]["numpy"] == np.__version__
    # The record's other picks are not this entry's reference: another floor finds no record.
    other = _write_list(tmp_path / "other.json", _finals(toy), references=[
        {**_references(toy)[0], "source": _reference_source(toy, floor=-40),
         "child_sha256": params_sha256(toy.own_child)}])
    with pytest.raises(SF.SealedFinalsError, match="0 records of its run select q1"):
        SF.run_dev(other, tmp_path / "other.jsonl", code_state_path=tmp_path / "git_state.json")


def test_dev_reads_own_and_candidate_picks_as_their_records(toy: Toy, tmp_path: Path) -> None:
    run_path = toy.store.runs / REF_RUN.format(instance=toy.instance.instance_id)
    own = {**_final("ref_own", _pick_source(toy, "own"), toy.instance.instance_id, toy.own_child),
           "pairs_with": "session_final"}
    listed = _write_list(tmp_path / "list.json", _finals(toy), references=[*_references(toy), own])
    records = SF.run_dev(listed, tmp_path / "dev.jsonl", code_state_path=_git_state(tmp_path / "git_state.json"))
    record = records[3]
    expected = devreadout.readout_run(run_path, clean=True)
    assert record["readouts"] == devreadout.json_safe(expected["readouts"])
    assert record["reference"]["match"] and record["reference"]["fields"] == SF.FIELDS_ALL
    assert (record["child"]["pick"], record["child"]["patch_id"]) == ("own", "q1")
    assert record["child"]["selection"]["rule"] == SF.OWN_RULE
    # A candidate is one named check, whatever the run selected: q0 here (the kept pick's child, so a list of its own).
    candidate = {**_final("ref_candidate", _pick_source(toy, "candidate", "q0"), toy.instance.instance_id,
                          toy.kept_child), "pairs_with": "session_final"}
    listed = _write_list(tmp_path / "candidate.json", _finals(toy), references=[candidate])
    record = SF.run_dev(listed, tmp_path / "candidate.jsonl", code_state_path=tmp_path / "git_state.json")[2]
    (line,) = [r for r in devreadout.readout_candidates(run_path) if r["patch_id"] == "q0"]
    assert record["readouts_clean"] == devreadout.json_safe(line["readouts_clean"])
    assert record["reference"]["match"] and record["child"]["selection"]["config"] == {"q": 0}
    final = SF.load_list(listed)[1][-1]
    with pytest.raises(SF.SealedFinalsError, match="0 checks and 0 candidates with patch_id 'q9'"):
        SF.recover_child(dataclasses.replace(final, source=_pick_source(toy, "candidate", "q9")), toy.instance)
    with pytest.raises(SF.SealedFinalsError, match="candidate pick q0 records child"):
        SF.recover_child(dataclasses.replace(final, child_sha256=params_sha256(toy.own_child)), toy.instance)
    own_final = SF.load_list(tmp_path / "list.json")[1][-1]
    lines = [json.loads(text) for text in (toy.store.runs / OWN_RECORD).read_text().splitlines()]
    lines[0]["vcheck"]["score"] += 1
    (toy.store.runs / OWN_RECORD).write_text(json.dumps(lines[0]) + "\n")
    with pytest.raises(SF.SealedFinalsError, match="has V_check"):
        SF.dev_reference(own_final)
    run = json.loads(run_path.read_text())
    run["final"].update(patch_id="q0", dir="q0", child_sha256=run["oracle_log"][0]["child_sha256"])
    run_path.write_text(json.dumps(run))
    with pytest.raises(SF.SealedFinalsError, match="not a checked candidate chosen by choose_final"):
        SF.recover_child(own_final, toy.instance)


def test_a_stream_round_is_read_as_its_round_record(toy: Toy, tmp_path: Path) -> None:
    _two_round_stream(toy.store, toy.instance, toy.kept_child, toy.own_child)
    instance_id = toy.instance.instance_id
    finals = [_final("W1", {"kind": "stream_round", "record": STREAM2, "round": 1}, instance_id, toy.kept_child),
              _final("W2", {"kind": "stream_final", "record": STREAM2}, instance_id, toy.own_child)]
    ref = {**_final("ref_W1", {"kind": "stream_round", "record": STREAM, "round": 1}, instance_id, toy.stream_child),
           "pairs_with": "W1"}
    listed = _write_list(tmp_path / "list.json", finals)
    records = SF.run_dev(listed, tmp_path / "dev.jsonl", code_state_path=_git_state(tmp_path / "git_state.json"))
    for record, child, number in zip(records, (toy.kept_child, toy.own_child), (1, 2)):
        joint = devreadout.dev_readout_joint(toy.instance, child)
        assert record["readouts"] == devreadout.json_safe(joint["readouts"])
        assert record["reference"]["match"] and record["reference"]["fields"] == SF.FIELDS_THEIRS
        assert (record["child"]["round"], record["child"]["n_rounds"]) == (number, 2)
    assert records[0]["child"]["kind"] == "stream_round" and "round 1's student_after" in records[0]["child"][
        "recovered_by"]
    entries = SF.load_list(_write_list(tmp_path / "ref.json", finals, references=[ref]))[1]
    # The one-round stream's round 1 is its last: a stream_final entry, not a stream_round.
    with pytest.raises(SF.SealedFinalsError, match="reads a round before the last"):
        SF.recover_child(entries[-1], toy.instance)
    with pytest.raises(SF.SealedFinalsError, match="round 1's student is"):
        SF.recover_child(dataclasses.replace(entries[0], child_sha256=params_sha256(toy.own_child)), toy.instance)
    record = json.loads((toy.store.runs / STREAM2).read_text())
    record["rounds"][0]["round"] = 5
    (toy.store.runs / STREAM2).write_text(json.dumps(record))
    with pytest.raises(SF.SealedFinalsError, match="numbered 5"):
        SF.dev_reference(entries[0])


def test_a_transfer_final_is_rebuilt_from_its_patch_and_read_as_its_record(toy: Toy, tmp_path: Path) -> None:
    child = fixing_child(dict(toy.other.params))
    rel = _transfer(toy.store, toy.other, child)
    final = _final("C_s101", {"kind": "transfer", "record": rel}, toy.other.instance_id, child)
    listed = _write_list(tmp_path / "list.json", [final])
    (record,) = SF.run_dev(listed, tmp_path / "dev.jsonl", code_state_path=_git_state(tmp_path / "git_state.json"))
    joint = devreadout.dev_readout_joint(toy.other, child)
    assert record["readouts"] == devreadout.json_safe(joint["readouts"]) and record["gain"] > 0
    assert record["reference"]["match"] and record["reference"]["fields"] == SF.FIELDS_ALL
    assert record["child"]["kind"] == "transfer" and record["child"]["program"]["path"].endswith("toy_program.py")
    entry = SF.load_list(listed)[1][0]
    path = toy.store.runs / rel
    original = path.read_text()

    def refused(change: Any, message: str) -> None:
        data = json.loads(original)
        change(data)
        path.write_text(json.dumps(data))
        with pytest.raises(SF.SealedFinalsError, match=message):
            SF.recover_child(entry, toy.other)
        path.write_text(original)

    refused(lambda d: d.update(schema="weightcode.harness.other.v0"), "has schema")
    refused(lambda d: d["target"].update(instance_id=toy.instance.instance_id), "ran on")
    refused(lambda d: d.update(within_limits=False, readout_child="unbounded"), "only a child the limits admitted")
    refused(lambda d: d["target"].update(params_sha256="0" * 64), "records base")
    refused(lambda d: d.update(work_dir="/elsewhere"), "work_dir")
    refused(lambda d: d.update(limits={**d["limits"], "max_changed_frac": 1e-9}), "under the recorded limits")
    refused(lambda d: d.update(limits={**d["limits"], "max_ops": "many"}), "records limits|round-trip")
    with pytest.raises(SF.SealedFinalsError, match="records child"):
        SF.recover_child(dataclasses.replace(entry, child_sha256=params_sha256(toy.other.params)), toy.other)
    program = toy.store.runs / "sessions" / SESSION / "final" / "toy_program.py"
    program.write_text("# changed\n")
    with pytest.raises(SF.SealedFinalsError, match="not the file"):
        SF.recover_child(entry, toy.other)


def test_a_t08_program_arm_record_is_a_transfer_source_with_the_same_checks(toy: Toy, tmp_path: Path) -> None:
    """List v3 reads T08's program-arm children from runs/t08/program/<sid>/ (NSCC R31): the record has E3's schema
    and work directory layout, so the transfer source takes it as it stands and still refuses a changed record."""
    child = fixing_child(dict(toy.other.params))
    rel = _transfer(toy.store, toy.other, child, layout=T08_TRANSFER)
    final = _final("T08_on_s101", {"kind": "transfer", "record": rel}, toy.other.instance_id, child)
    listed = _write_list(tmp_path / "list.json", [final])
    (record,) = SF.run_dev(listed, tmp_path / "dev.jsonl", code_state_path=_git_state(tmp_path / "git_state.json"))
    assert record["reference"]["match"] and record["reference"]["fields"] == SF.FIELDS_ALL
    assert record["child"]["record"] == str(toy.store.runs / rel) and record["gain"] > 0
    entry = SF.load_list(listed)[1][0]
    path = toy.store.runs / rel
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "work_dir": str(toy.store.runs / TRANSFER.format(
        instance=toy.other.instance_id)).removesuffix(".json")}))
    with pytest.raises(SF.SealedFinalsError, match="work_dir"):
        SF.recover_child(entry, toy.other)


def test_a_family_whose_code_is_missing_is_refused(toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SF.FAMILY_MODULES, SF.READOUT_SCAN_EM, "weightcode.evalkit.no_such_family")
    scan = {"name": "scan_final", "readout": "scan_em", "source": {"kind": "session", "session_id": OTHER_SESSION},
            "instance_id": "n01234567", "child_sha256": "1" * 64}
    listed = _write_list(tmp_path / "list.json", [*_finals(toy), scan])
    assert len(SF.load_list(listed)[1]) == 3, "the list validates without importing a family"
    with pytest.raises(SF.SealedFinalsError, match="does not import in this tree"):
        SF.run_dev(listed, tmp_path / "dev.jsonl", code_state_path=_git_state(tmp_path / "git_state.json"))
    records = SF.run_dev(listed, tmp_path / "part.jsonl", code_state_path=tmp_path / "git_state.json",
                         only=["session_final"])
    assert records[0]["reference"]["match"] and set(records[0]["code"]["files"]) == set(SF.CODE_MODULES)


def test_dev_opens_nothing_under_the_sealed_store(toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sealed_root = str(toy.store.sealed.resolve())
    empty = tmp_path / "empty_sealed"
    empty.mkdir()
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(empty))
    opened: list[str] = []
    active = [True]

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if active[0] and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(args[0]))

    sys.addaudithook(hook)
    try:
        records = SF.run_dev(_write_list(tmp_path / "list.json", _finals(toy), references=_references(toy)),
                             tmp_path / "dev.jsonl", code_state_path=_git_state(tmp_path / "git_state.json"))
    finally:
        active[0] = False
    assert len(records) == 3
    assert all(r["reference"]["match"] for r in records)
    assert not [p for p in opened if p.startswith(sealed_root) or p.startswith(str(empty))]
    assert list(empty.iterdir()) == []
    assert records[0]["sealed_index"]["condition_test"]["path"].startswith(str(empty.resolve()))


def test_dev_reports_a_difference_from_the_record(toy: Toy, tmp_path: Path, capsys: pytest.CaptureFixture[str]
                                                  ) -> None:
    reference = toy.store.runs / "devreadouts" / "learn" / SESSION / "info.json"
    info = json.loads(reference.read_text())
    info["final"]["vselect_clean"]["broken"] += 1
    reference.write_text(json.dumps(info))
    listed = _write_list(tmp_path / "list.json", _finals(toy))
    state = _git_state(tmp_path / "git_state.json")
    status = SF.main(["--list", str(listed), "--dev", "--out", str(tmp_path / "dev.jsonl"), "--code-state", str(state)])
    assert status == 1
    out = capsys.readouterr().out
    assert "readouts_clean.broken" in out and "all match: False" in out
    records = [json.loads(line) for line in (tmp_path / "dev.jsonl").read_text().splitlines()]
    assert [r["reference"]["match"] for r in records] == [False, True]


def test_dev_only_reads_named_finals_under_the_whole_list(toy: Toy, tmp_path: Path) -> None:
    listed = _write_list(tmp_path / "list.json", _finals(toy))
    state = _git_state(tmp_path / "git_state.json")
    whole = SF.run_dev(listed, tmp_path / "whole.jsonl", code_state_path=state)
    part = SF.run_dev(listed, tmp_path / "part.jsonl", code_state_path=state, only=["stream_final"])
    assert [r["name"] for r in part] == ["stream_final"] and part[0]["only"] == ["stream_final"]
    assert part[0]["list_sha256"] == whole[1]["list_sha256"] and part[0]["readouts"] == whole[1]["readouts"]
    with pytest.raises(SF.SealedFinalsError, match="does not have"):
        SF.run_dev(listed, tmp_path / "x.jsonl", code_state_path=state, only=["nothing"])
    with pytest.raises(SystemExit):
        SF.main(["--list", str(listed), "--sealed", "--only", "stream_final"])


def test_cli_entry_point_runs_the_imported_module(toy: Toy, tmp_path: Path) -> None:
    listed = _write_list(tmp_path / "list.json", _finals(toy)[:1])
    state = _git_state(tmp_path / "git_state.json")
    env = {**os.environ, "WEIGHTCODE_PRIVATE": str(toy.store.private), "WEIGHTCODE_RUNS": str(toy.store.runs),
           "WEIGHTCODE_SEALED": str(toy.store.sealed)}
    done = subprocess.run([sys.executable, "-m", "weightcode.evalkit.sealed_finals", "--list", str(listed), "--dev",
                           "--out", str(tmp_path / "cli.jsonl"), "--code-state", str(state)],
                          capture_output=True, text=True, env=env, check=False)
    assert done.returncode == 0, done.stderr
    assert "all match: True" in done.stdout
    refused = subprocess.run([sys.executable, "-m", "weightcode.evalkit.sealed_finals", "--list", str(tmp_path / "no.json"),
                              "--dev", "--out", str(tmp_path / "x.jsonl")], capture_output=True, text=True, env=env,
                             check=False)
    assert refused.returncode != 0


def test_dev_code_state_from_git(toy: Toy, tmp_path: Path) -> None:
    records = SF.run_dev(_write_list(tmp_path / "list.json", _finals(toy)[:1]), tmp_path / "dev.jsonl")
    state = records[0]["code"]
    assert len(state["commit"]) == 40 and isinstance(state["dirty"], bool) and state["source"].startswith("git ")


# --------------------------------------------------------------------------------------------------------- the list


def _ref(d: dict[str, Any]) -> dict[str, Any]:
    return d["references"][0]


def _without_floor(source: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**{k: v for k, v in source.items() if k != "s_old_floor"}, **changes}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("note"), "list keys"),
        (lambda d: d.pop("references"), "list keys"),
        (lambda d: d.update(schema="other"), "schema must be one of"),
        (lambda d: d.update(status="final"), "status"),
        (lambda d: d.update(approval="yes"), "draft list has approval null"),
        (lambda d: d.update(status="approved"), "states its approval"),
        (lambda d: d.update(finals=[]), "non-empty"),
        (lambda d: d.update(references={}), "references must be a list"),
        (lambda d: d["finals"].append(dict(d["finals"][0], name="again")), "appears twice"),
        (lambda d: d["finals"].append(dict(d["finals"][1], child_sha256="0" * 64)), "duplicate name"),
        (lambda d: d["finals"][0].update(readout="other"), "not implemented"),
        (lambda d: d["finals"][0].update(readout="scan_em"), "invalid instance_id .* for readout scan_em"),
        (lambda d: d["finals"][0].update(source={"kind": "other", "session_id": SESSION}), "kind in"),
        (lambda d: d["finals"][0].update(source={"kind": "transfer", "session_id": SESSION}), "exactly the keys"),
        (lambda d: d["finals"][0].update(readout="scan_em", instance_id="n00000000",
                                         source={"kind": "transfer", "record": "x/y.json"}), "vision_joint entry"),
        (lambda d: d["finals"][1]["source"].update(kind="stream_round", round=0), "round must be"),
        (lambda d: d["finals"][1]["source"].update(kind="stream_round", round=True), "round must be"),
        (lambda d: d["finals"][1]["source"].update(kind="stream_round"), "exactly the keys"),
        (lambda d: _ref(d).update(source={"kind": "transfer", "record": "x/y.json"}), "kind in"),
        (lambda d: d["finals"][0].update(source={"kind": "session", "session_id": "s123"}), "session_id"),
        (lambda d: d["finals"][0]["source"].update(record="x"), "exactly the keys"),
        (lambda d: d["finals"][1].update(source={"kind": "stream_final", "record": "../x.json"}), "relative"),
        (lambda d: d["finals"][0].update(instance_id="x1"), "instance_id"),
        (lambda d: d["finals"][0].update(child_sha256="ABC"), "child_sha256"),
        (lambda d: d["finals"][0].update(source=_ref(d)["source"]), "kind in"),
        (lambda d: _ref(d).update(source=d["finals"][0]["source"]), "kind in"),
        (lambda d: _ref(d).pop("pairs_with"), "keys must be exactly"),
        (lambda d: _ref(d).update(pairs_with="stream_finalx"), "not a final on"),
        (lambda d: _ref(d).update(name="session_final"), "duplicate name"),
        (lambda d: _ref(d).update(child_sha256=d["finals"][0]["child_sha256"]), "appears twice"),
        (lambda d: _ref(d)["source"].update(pick="best"), "pick must be one of"),
        (lambda d: _ref(d)["source"].update(pick="own"), "with pick 'own' has exactly the keys"),
        (lambda d: _ref(d)["source"].update(pick="candidate", patch_id="q0"), "with pick 'candidate' has exactly"),
        (lambda d: _ref(d).update(source=_without_floor(_ref(d)["source"], pick="candidate", patch_id="../q0")),
         "invalid patch_id"),
        (lambda d: _ref(d).update(readout="scan_em", instance_id="n00000000",
                                  source=_without_floor(_ref(d)["source"], pick="own")), "vision_joint reference"),
        (lambda d: _ref(d)["source"].update(s_old_floor=5), "s_old_floor"),
        (lambda d: _ref(d)["source"].update(s_old_floor=True), "s_old_floor"),
        (lambda d: _ref(d)["source"].update(run="sessions/x/run.json"), "run must be a path"),
        (lambda d: _ref(d)["source"].update(run="baselines/../x/run.json"), "run must be a path"),
        (lambda d: _ref(d)["source"].update(dev_record="devreadouts/x.json"), "dev_record must be a path"),
        (lambda d: d.update(status="approved", approval="ok", references=[dict(_ref(d), pairs_with=None)]),
         "pairs with no final"),
    ],
)
def test_list_validation(toy: Toy, tmp_path: Path, change: Any, message: str) -> None:
    data = {"schema": SF.LIST_SCHEMA, "status": "draft", "approval": None, "note": "", "finals": _finals(toy),
            "references": _references(toy)}
    change(data)
    path = tmp_path / "list.json"
    path.write_text(json.dumps(data))
    with pytest.raises(SF.SealedFinalsError, match=message):
        SF.load_list(path)


def test_a_draft_reference_may_wait_for_its_final_and_v0_lists_still_load(toy: Toy, tmp_path: Path) -> None:
    waiting = _write_list(tmp_path / "waiting.json", _finals(toy),
                          references=[dict(_references(toy)[0], pairs_with=None)])
    assert SF.load_list(waiting)[1][-1].pairs_with is None
    v0 = {"schema": SF.LIST_SCHEMA_V0, "status": "draft", "approval": None, "note": "", "finals": _finals(toy)}
    path = tmp_path / "v0.json"
    path.write_text(json.dumps(v0))
    assert [f.name for f in SF.load_list(path)[1]] == ["session_final", "stream_final"]
    for change, message in ((lambda d: d.update(references=[]), "list keys"),
                            (lambda d: d["finals"][0].update(readout="scan_em"), "not implemented")):
        data = json.loads(json.dumps(v0))
        change(data)
        path.write_text(json.dumps(data))
        with pytest.raises(SF.SealedFinalsError, match=message):
            SF.load_list(path)


def test_child_recovery_refusals(toy: Toy, tmp_path: Path) -> None:
    final = SF.Final("f", "vision_joint", {"kind": "session", "session_id": SESSION}, toy.instance.instance_id,
                     "0" * 64)
    with pytest.raises(SF.SealedFinalsError, match="the list says"):
        SF.recover_child(final, toy.instance)
    wrong_instance = dataclasses.replace(final, instance_id=toy.other.instance_id,
                                         child_sha256=params_sha256(toy.session_child))
    with pytest.raises(SF.SealedFinalsError, match="ran on"):
        SF.recover_child(wrong_instance, toy.other)
    missing = dataclasses.replace(final, source={"kind": "session", "session_id": OTHER_SESSION})
    with pytest.raises(SF.SealedFinalsError, match="no .*run.json"):
        SF.recover_child(missing, toy.instance)
    listed_child = dataclasses.replace(final, child_sha256=params_sha256(toy.session_child))
    assert SF.recover_child(listed_child, toy.instance)[1]["patch_id"] == "toy.py"
    final_dir = toy.store.runs / "sessions" / SESSION / "final"
    for path in final_dir.iterdir():
        path.unlink()
    write_delta_patch(final_dir, toy.instance.params_sha256, toy.instance.params, toy.stream_child, "x", "x")
    with pytest.raises(SF.SealedFinalsError, match="rebuilt child"):
        SF.recover_child(listed_child, toy.instance)
    stream = SF.Final("s", "vision_joint", {"kind": "stream_final", "record": STREAM}, toy.other.instance_id,
                      params_sha256(toy.stream_child))
    with pytest.raises(SF.SealedFinalsError, match="not an instance of the stream"):
        SF.recover_child(stream, toy.other)
    record = json.loads((toy.store.runs / STREAM).read_text())
    record["instances"].append({"instance_id": "i00000000", "condition": CONDITION, "round": 2})
    (toy.store.runs / STREAM).write_text(json.dumps(record))
    with pytest.raises(SF.SealedFinalsError, match="only a completed stream"):
        SF.recover_child(dataclasses.replace(stream, instance_id=toy.instance.instance_id), toy.instance)


def test_reference_recovery_refusals(toy: Toy) -> None:
    ref = SF.Final("r", "vision_joint", _reference_source(toy), toy.instance.instance_id,
                   params_sha256(toy.kept_child), SF.BLOCK_REFERENCES, "session_final")
    child, provenance = SF.recover_child(ref, toy.instance)
    assert params_sha256(child) == params_sha256(toy.kept_child) and provenance["patch_id"] == "q0"
    with pytest.raises(SF.SealedFinalsError, match="kept pick q0 records child"):
        SF.recover_child(dataclasses.replace(ref, child_sha256=params_sha256(toy.own_child)), toy.instance)
    with pytest.raises(SF.SealedFinalsError, match="ran on"):
        SF.recover_child(dataclasses.replace(ref, instance_id=toy.other.instance_id), toy.other)
    # At a floor that admits the own final, the kept pick is q1; the list's child is then not the pick's.
    with pytest.raises(SF.SealedFinalsError, match="kept pick q1 records child"):
        SF.recover_child(dataclasses.replace(ref, source=_reference_source(toy, floor=-40)), toy.instance)
    run_path = toy.store.runs / REF_RUN.format(instance=toy.instance.instance_id)
    run = json.loads(run_path.read_text())
    (run_path.parent / "q0" / "patch.json").write_text((run_path.parent / "q1" / "patch.json").read_text())
    with pytest.raises(SF.SealedFinalsError, match="rebuilt child"):
        SF.recover_child(ref, toy.instance)
    run["instance"]["forward_sha256"] = "f" * 64
    run_path.write_text(json.dumps(run))
    with pytest.raises(SF.SealedFinalsError, match="forward_sha256"):
        SF.recover_child(ref, toy.instance)
    (run_path.parent / "session.json").write_text("{}")
    with pytest.raises(SF.SealedFinalsError, match="harness run"):
        SF.recover_child(ref, toy.instance)


# ------------------------------------------------------------------------------------------------------ sealed mode


def _repo_with(tmp_path: Path, list_path: Path) -> Path:
    """A git repo that tracks ``list_path``; returns the path of the committed copy."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    committed = repo / "finals.json"
    committed.write_text(list_path.read_text())
    for args in (["init", "-q"], ["add", "finals.json"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "list"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return committed


def _clean_state(monkeypatch: pytest.MonkeyPatch, repo: Path, dirty: bool = False) -> None:
    state = {"commit": COMMIT, "dirty": dirty, "source": f"git {repo}", "repo": str(repo), "files": SF.code_files()}
    monkeypatch.setattr(SF, "code_state", lambda code_state_path=None, modules=SF.CODE_MODULES: dict(state))


def _approved(toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "draft.json",
              finals: list[dict[str, Any]] | None = None, references: list[dict[str, Any]] | None = None) -> Path:
    draft = _write_list(tmp_path / name, finals or _finals(toy), status="approved", approval="toy approval",
                        references=references)
    committed = _repo_with(tmp_path / name.replace(".json", ""), draft)
    _clean_state(monkeypatch, committed.parent)
    return committed


def _expected(instance: Instance, child: dict[str, np.ndarray], condition: dict[str, np.ndarray],
              clean: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    return (devreadout.json_safe(devreadout.dev_readout(instance, child, split=condition)["readouts"]),
            devreadout.json_safe(devreadout.dev_readout(instance, child, split=clean)["readouts"]))


def test_sealed_read_of_store_npz_tests(toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    listed = _approved(toy, tmp_path, monkeypatch, references=_references(toy))
    assert SF.main(["--list", str(listed), "--sealed"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[0].startswith("results: ") and out[1] == "entries: 3"
    results = Path(out[0].removeprefix("results: "))
    records = [json.loads(line) for line in results.read_text().splitlines()]
    condition = load_npz(condition_test_path(toy.store))
    clean = load_npz(toy.store.sealed / toy.store.dataset / "test.npz")
    for record, child in zip(records, (toy.session_child, toy.stream_child, toy.kept_child)):
        readouts, readouts_clean = _expected(toy.instance, child, condition, clean)
        assert record["readouts"] == readouts and record["readouts_clean"] == readouts_clean
        assert record["mode"] == "sealed" and record["rows"]["condition"]["split"] == "sealed test"
        assert record["rows"]["condition"]["verified_sha256"] == record["sealed_index"]["condition_test"]["sha256"]
        assert record["joint"] == record["gain"] + record["clean"]
        assert "reference" not in record
    assert [r["block"] for r in records] == ["finals", "finals", "references"]
    assert [r["list_index"] for r in records] == [0, 1, 2]
    events = read_registry(toy.store.sealed / "registry.jsonl")
    assert [e["event"] for e in events] == ["evaluation_started", *["entry_started", "entry_finished"] * 3,
                                            "evaluation_finished"]
    assert events[0]["kind"] == SF.EVENT_KIND and events[0]["approval"] == "toy approval" and events[0]["workers"] == 1
    lines = results.read_text().splitlines()
    assert [e["record_sha256"] for e in events if e["event"] == "entry_finished"] == [_sha(line) for line in lines]
    assert {(e["instance_id"], e["child_sha256"]) for e in events[0]["entries"]} == {
        (toy.instance.instance_id, params_sha256(toy.session_child)),
        (toy.instance.instance_id, params_sha256(toy.stream_child)),
        (toy.instance.instance_id, params_sha256(toy.kept_child))}
    assert events[-1]["results_sha256"] == file_sha256(results)
    assert events[-1]["n_read"] == 3 and events[-1]["failed"] == {}
    assert set(events[-1]["test_sha256"]) == {r["rows"][p]["location"] for r in records for p in ("condition", "clean")}
    with pytest.raises(SF.SealedFinalsError, match="already read or attempted"):
        SF.run_sealed(listed)
    again = _approved(toy, tmp_path, monkeypatch, name="again.json", finals=_finals(toy)[1:])
    with pytest.raises(SF.SealedFinalsError, match="already read in an earlier sealed read"):
        SF.run_sealed(again)
    assert len(read_registry(toy.store.sealed / "registry.jsonl")) == 8


def _as_cifar10c(toy: Toy) -> Path:
    """Rewrite the condition's sealed test as an official-style CIFAR-10-C file (5 severities, NHWC)."""
    test = load_npz(condition_test_path(toy.store))
    n = len(test["y"])
    rng = np.random.default_rng(7)
    official = rng.integers(0, 256, size=(5 * n, *test["x"].shape[2:], test["x"].shape[1]), dtype=np.uint8)
    official[4 * n:] = test["x"].transpose(0, 2, 3, 1)
    path = toy.store.sealed / "cifar10c" / "CIFAR-10-C" / "toyblur.npy"
    path.parent.mkdir(parents=True)
    np.save(path, official)
    condition_test_path(toy.store).unlink()
    clean_json = toy.store.private / toy.store.dataset / "dataset.json"
    clean_meta = json.loads(clean_json.read_text())
    clean_meta["test"]["n"] = n  # the real stores record n on both test sets
    clean_json.write_text(json.dumps(clean_meta, indent=2))
    meta = json.loads((condition_dir(toy.store) / "dataset.json").read_text())
    meta["test"] = {"location": "$WEIGHTCODE_SEALED/cifar10c/CIFAR-10-C/toyblur.npy", "rows": f"{4 * n} .. {5 * n - 1} "
                    "(severity 5)", "format": SF.CIFAR10C_FORMAT_TEXT, "n": n, "sha256": file_sha256(path),
                    "clean_sha256": meta["test"]["clean_sha256"]}
    write_condition_meta(toy.store, meta)
    return path


def test_sealed_read_of_the_official_cifar10c_layout(toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                                     ) -> None:
    condition = load_npz(condition_test_path(toy.store))
    _as_cifar10c(toy)
    index = SF.sealed_index(toy.store.dataset, CONDITION)["condition_test"]
    n = len(condition["y"])
    assert index["format"] == SF.FORMAT_CIFAR10C and index["rows"] == {"start": 4 * n, "stop": 5 * n, "severity": 5}
    results, count = SF.run_sealed(_approved(toy, tmp_path, monkeypatch))
    assert count == 2
    clean = load_npz(toy.store.sealed / toy.store.dataset / "test.npz")
    for record, child in zip([json.loads(line) for line in results.read_text().splitlines()],
                             (toy.session_child, toy.stream_child)):
        readouts, readouts_clean = _expected(toy.instance, child, condition, clean)
        assert record["readouts"] == readouts and record["readouts_clean"] == readouts_clean
        assert record["rows"]["condition"]["rows_content_sha256"] == params_sha256(condition)


def test_a_failed_read_is_recorded_and_spends_the_list(toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
                                                       ) -> None:
    path = _as_cifar10c(toy)
    listed = _approved(toy, tmp_path, monkeypatch)
    official = np.load(path)
    official[0, 0, 0, 0] ^= 1
    np.save(path, official)
    with pytest.raises(SF.SealedEntriesFailed, match="file sha256") as raised:
        SF.run_sealed(listed)
    assert raised.value.n_read == 0 and raised.value.results_path.read_text() == ""
    events = read_registry(toy.store.sealed / "registry.jsonl")
    assert [e["event"] for e in events] == ["evaluation_started", *["entry_started", "entry_failed"] * 2,
                                            "evaluation_finished"]
    assert set(events[-1]["failed"]) == {"session_final", "stream_final"} and events[-1]["n_read"] == 0
    with pytest.raises(SF.SealedFinalsError, match="already read or attempted"):
        SF.run_sealed(listed)


# ------------------------------------------------------------------------------------------- parallel sealed mode


def _sha(line: str) -> str:
    import hashlib

    return hashlib.sha256(line.encode()).hexdigest()


def _four(toy: Toy) -> list[dict[str, Any]]:
    """Three references beside the two finals: the kept pick and a fine-tuning stream (four distinct children)."""
    _stream(toy.store, toy.instance, toy.own_child, F4_STREAM, "weightcode.baselines.checked_reference_stream.v0")
    return [*_references(toy),
            {**_final("ref_f4", {"kind": "stream_final", "record": F4_STREAM}, toy.instance.instance_id,
                      toy.own_child), "pairs_with": "stream_final"}]


def _entry_histories(events: list[dict[str, Any]]) -> dict[str, list[str]]:
    histories: dict[str, list[str]] = {}
    for e in events:
        if e["event"].startswith("entry_"):
            histories.setdefault(e["name"], []).append(e["event"])
    return histories


def test_parallel_sealed_read_reads_each_entry_once_in_one_ordered_registry(
        toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    listed = _approved(toy, tmp_path, monkeypatch, references=_four(toy))
    assert SF.main(["--list", str(listed), "--sealed", "--workers", "3"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "entries: 4"
    results = Path(out[0].removeprefix("results: "))
    lines = results.read_text().splitlines()
    records = sorted((json.loads(line) for line in lines), key=lambda r: r["list_index"])
    condition = load_npz(condition_test_path(toy.store))
    clean = load_npz(toy.store.sealed / toy.store.dataset / "test.npz")
    children = (toy.session_child, toy.stream_child, toy.kept_child, toy.own_child)
    assert [r["name"] for r in records] == ["session_final", "stream_final", "ref_kept", "ref_f4"]
    for record, child in zip(records, children):
        readouts, readouts_clean = _expected(toy.instance, child, condition, clean)
        assert record["readouts"] == readouts and record["readouts_clean"] == readouts_clean
        assert record["child_sha256"] == params_sha256(child) and record["mode"] == "sealed"
    events = read_registry(toy.store.sealed / "registry.jsonl")
    assert events[0]["event"] == "evaluation_started" and events[0]["workers"] == 3
    assert events[-1]["event"] == "evaluation_finished" and events[-1]["n_read"] == 4 and events[-1]["failed"] == {}
    assert sum(e["event"] == "evaluation_started" for e in events) == 1
    assert _entry_histories(events) == {r["name"]: ["entry_started", "entry_finished"] for r in records}
    # three entries were handed over before the first came back: the reads overlapped
    assert [e["event"] for e in events[1:4]] == ["entry_started"] * 3
    finished = {e["name"]: e["record_sha256"] for e in events if e["event"] == "entry_finished"}
    assert sorted(finished.values()) == sorted(_sha(line) for line in lines)
    assert events[-1]["results_sha256"] == file_sha256(results)
    with pytest.raises(SF.SealedFinalsError, match="already read or attempted"):
        SF.run_sealed(listed, workers=3)


def test_a_failed_worker_is_recorded_and_spent_and_the_others_are_read(
        toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listed = _approved(toy, tmp_path, monkeypatch, references=_four(toy))
    # The session's final changes on disk after the parent's checks: the worker rebuilds it and refuses the hash.
    real_recover = SF.recover_child
    good, provenance = real_recover(SF.load_list(listed)[1][0], toy.instance)
    monkeypatch.setattr(SF, "recover_child", lambda final, instance: (
        (good, provenance) if final.name == "session_final" else real_recover(final, instance)))
    other = dict(toy.session_child)
    other["fc2.bias"] = other["fc2.bias"] + 1.0
    write_delta_patch(toy.store.runs / "sessions" / SESSION / "final", toy.instance.params_sha256,
                      toy.instance.params, other, "toy", "toy")
    with pytest.raises(SF.SealedEntriesFailed, match="session_final: SealedFinalsError: .*rebuilt child") as raised:
        SF.run_sealed(listed, workers=2)
    assert raised.value.n_read == 3
    events = read_registry(toy.store.sealed / "registry.jsonl")
    histories = _entry_histories(events)
    assert histories.pop("session_final") == ["entry_started", "entry_failed"]
    assert histories == {name: ["entry_started", "entry_finished"] for name in ("stream_final", "ref_kept", "ref_f4")}
    assert events[-1]["event"] == "evaluation_finished" and list(events[-1]["failed"]) == ["session_final"]
    records = [json.loads(line) for line in raised.value.results_path.read_text().splitlines()]
    assert sorted(r["name"] for r in records) == ["ref_f4", "ref_kept", "stream_final"]
    with pytest.raises(SF.SealedFinalsError, match="already read or attempted"):
        SF.run_sealed(listed, workers=2)


def test_a_worker_opens_nothing_without_its_started_record(toy: Toy, tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    entry = SF.load_list(_write_list(tmp_path / "list.json", _finals(toy)))[1][0]
    registry = tmp_path / "registry.jsonl"
    monkeypatch.setattr(SF, "read_final", lambda *a, **k: pytest.fail("read_final ran without entry_started"))
    for history in ([], ["entry_started", "entry_finished"], ["entry_started", "entry_started"]):
        registry.write_text("".join(json.dumps({"event": e, "kind": SF.EVENT_KIND, "manifest_sha256": "m",
                                                "name": entry.name}) + "\n" for e in history))
        with pytest.raises(SF.SealedFinalsError, match="nothing of it is opened"):
            SF.read_sealed_entry(entry, 0, "m", str(registry), 8)


def _fake_read(name: str) -> dict[str, Any]:
    """A worker body for the dispatch test: "crash" ends its process the way an out-of-memory kill would."""
    if name == "crash":
        os._exit(1)
    return {"name": name}


def test_a_broken_worker_pool_fails_its_running_entries_and_the_rest_are_read() -> None:
    """The crash fails its entry; an entry whose result never arrives from the broken pool fails with it; the last
    entry is always read in a new pool. Python's executor may still deliver the result of an entry another worker of
    the broken pool completed (then it is finished, a full record), and an entry handed over just before the pool was
    seen broken fails (spent), so "a" and "b" may end either way."""
    @dataclasses.dataclass(frozen=True)
    class Named:
        name: str

    entries = [Named(n) for n in ("a", "crash", "b", "c")]
    history: list[tuple[str, str]] = []
    SF._dispatch(entries, 2, _fake_read, lambda index, entry: (entry.name,),
                 lambda index, entry: history.append(("started", entry.name)),
                 lambda index, entry, record: history.append(("finished", record["name"])),
                 lambda index, entry, error: history.append(("failed", entry.name)))
    outcomes = {name: [kind for kind, n in history if n == name] for name in ("a", "crash", "b", "c")}
    assert outcomes["crash"] == ["started", "failed"]
    assert outcomes["c"] == ["started", "finished"]
    assert outcomes["a"] in (["started", "finished"], ["started", "failed"])
    assert outcomes["b"] in (["started", "finished"], ["started", "failed"])


def test_workers_is_a_sealed_option(toy: Toy, tmp_path: Path) -> None:
    listed = _write_list(tmp_path / "list.json", _finals(toy))
    for argv in (["--dev", "--out", str(tmp_path / "x.jsonl"), "--workers", "2"], ["--sealed", "--workers", "0"]):
        with pytest.raises(SystemExit):
            SF.main(["--list", str(listed), *argv])
    with pytest.raises(SF.SealedFinalsError, match="workers must be"):
        SF.run_sealed(listed, workers=0)


def test_sealed_refusals_before_any_record(toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    draft = _write_list(tmp_path / "draft.json", _finals(toy))
    with pytest.raises(SF.SealedFinalsError, match="only an approved list"):
        SF.run_sealed(draft)
    listed = _approved(toy, tmp_path, monkeypatch)
    _clean_state(monkeypatch, listed.parent, dirty=True)
    with pytest.raises(SF.SealedFinalsError, match="not clean"):
        SF.run_sealed(listed)
    _clean_state(monkeypatch, listed.parent)
    untracked = listed.parent / "untracked.json"
    untracked.write_text(listed.read_text())
    with pytest.raises(SF.SealedFinalsError, match="not tracked"):
        SF.run_sealed(untracked)
    outside = _write_list(tmp_path / "outside.json", _finals(toy), status="approved", approval="toy")
    with pytest.raises(SF.SealedFinalsError, match="not inside"):
        SF.run_sealed(outside)
    condition_test_path(toy.store).unlink()
    with pytest.raises(SF.SealedFinalsError, match="no sealed file"):
        SF.run_sealed(listed)
    assert not (toy.store.sealed / "registry.jsonl").exists()
    with pytest.raises(SystemExit):
        SF.main(["--list", str(listed), "--sealed", "--out", str(tmp_path / "x.jsonl")])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda t: t.update(location="/abs/test.npz"), "not under"),
        (lambda t: t.update(location="$WEIGHTCODE_SEALED/../x.npy"), "leaves"),
        (lambda t: t.update(format="png"), "unknown test format"),
        (lambda t: t.update(rows="1 .. 2 (severity 5)"), "not the severity block"),
        (lambda t: t.update(rows="rows 0 to 9"), "records n and rows"),
        (lambda t: t.update(clean_sha256="f" * 64), "clean_sha256"),
        (lambda t: t.pop("sha256"), "records no test.sha256"),
    ],
)
def test_sealed_index_refusals(toy: Toy, change: Any, message: str) -> None:
    _as_cifar10c(toy)
    meta = json.loads((condition_dir(toy.store) / "dataset.json").read_text())
    change(meta["test"])
    write_condition_meta(toy.store, meta)
    with pytest.raises(SF.SealedFinalsError, match=message):
        SF.sealed_index(toy.store.dataset, CONDITION)


def test_the_draft_lists_of_record_validate() -> None:
    root = Path(__file__).resolve().parents[1] / "scripts" / "sealed"
    parsed, finals = SF.load_list(root / "finals_v0_draft.json")  # R26-sealed-dry's list
    assert parsed["status"] == "draft" and parsed["approval"] is None and len(finals) == 10
    assert len({f.child_sha256 for f in finals}) == 7
    parsed, entries = SF.load_list(root / "finals_v1_draft.json")
    assert parsed["schema"] == SF.LIST_SCHEMA and parsed["status"] == "draft" and parsed["approval"] is None
    finals = [e for e in entries if e.block == SF.BLOCK_FINALS]
    references = [e for e in entries if e.block == SF.BLOCK_REFERENCES]
    assert {f.name for f in finals} >= {f.name for f in SF.load_list(root / "finals_v0_draft.json")[1]}
    assert all(r.source["pick"] == "kept" for r in references if r.source["kind"] == "reference")
    assert {r.source["kind"] for r in references} == {"reference", "stream_final"}
    by_name = {f.name: f for f in finals}
    for ref in references:
        assert ref.pairs_with is None or by_name[ref.pairs_with].instance_id == ref.instance_id


def test_the_approved_list_is_the_draft_with_its_approval() -> None:
    root = Path(__file__).resolve().parents[1] / "scripts" / "sealed"
    approved, entries = SF.load_list(root / "finals_v1.json")
    draft = json.loads((root / "finals_v1_draft.json").read_text())
    assert approved["status"] == SF.STATUS_APPROVED and "AskUserQuestion 2026-11-06" in approved["approval"]
    assert {k: v for k, v in approved.items() if k not in ("status", "approval")} == {
        k: v for k, v in draft.items() if k not in ("status", "approval")}
    assert sum(e.block == SF.BLOCK_FINALS for e in entries) == 13
    assert sum(e.block == SF.BLOCK_REFERENCES for e in entries) == 15
    assert all(e.pairs_with is not None for e in entries if e.block == SF.BLOCK_REFERENCES)


def test_the_v2_draft_is_v1_plus_the_children_behind_the_main_tables() -> None:
    root = Path(__file__).resolve().parents[1] / "scripts" / "sealed"
    v1 = json.loads((root / "finals_v1.json").read_text())
    parsed, entries = SF.load_list(root / "finals_v2_draft.json")
    assert parsed["status"] == SF.STATUS_DRAFT and parsed["approval"] is None
    for block in (SF.BLOCK_FINALS, SF.BLOCK_REFERENCES):
        assert parsed[block][:len(v1[block])] == v1[block], f"v1's {block} come first, unchanged"
    finals = [e for e in entries if e.block == SF.BLOCK_FINALS]
    references = [e for e in entries if e.block == SF.BLOCK_REFERENCES]
    assert (len(finals), len(references)) == (35, 53)
    added = finals[13:] + references[15:]
    kinds = sorted({(e.source["kind"], e.source.get("pick")) for e in added}, key=str)
    assert kinds == sorted({("transfer", None), ("stream_round", None), ("reference", "kept"), ("reference", "own"),
                            ("reference", "candidate")}, key=str)
    by_name = {e.name: e for e in entries}
    assert all(by_name[r.pairs_with].instance_id == r.instance_id for r in references)
    assert sum(e.source["kind"] == "stream_round" for e in added) == 24, "rounds 1 to 3, four instances, two streams"
    cells = json.loads((root / "finals_v2_cells.json").read_text())

    def names(value: Any) -> list[str]:
        if isinstance(value, dict):
            return [n for v in value.values() for n in names(v)]
        return [value] if isinstance(value, str) and value in by_name else []

    mapped = set(names(cells["tables"]))
    assert {e.name for e in added} <= mapped, "every added entry fills a table cell"
    assert len(cells["base_noops"]) == 2 and any("ib1e4e794" in n for n in cells["base_noops"])
    assert any("p6ce8ad67" in n for n in cells["base_noops"]), "BabyAI's F1 kept and own picks are the base"
    assert set(cells["tables"]) == {"main", "examples", "reuse", "stack"}


def test_the_approved_v2_list_is_the_v2_draft_with_its_approval() -> None:
    root = Path(__file__).resolve().parents[1] / "scripts" / "sealed"
    approved, entries = SF.load_list(root / "finals_v2.json")
    draft = json.loads((root / "finals_v2_draft.json").read_text())
    migration = json.loads((root / "namespace_migration.json").read_text())
    migrated = next(row for row in migration["records"] if row["path"] == "scripts/sealed/finals_v2_draft.json")
    assert migrated["original_manifest_sha256"].startswith("a0c26e7d5e0bbf1e"), "the draft the user approved"
    assert SF.manifest_sha256(draft) == migrated["manifest_sha256"], "the namespace-migrated draft"
    assert approved["status"] == SF.STATUS_APPROVED and draft["status"] == SF.STATUS_DRAFT
    assert "AskUserQuestion 2026-11-06 about 17:17" in approved["approval"] and "378227b" in approved["approval"]
    assert "a0c26e7d5e0bbf1e" in approved["approval"] and "--workers 5" in approved["approval"]
    assert {k: v for k, v in approved.items() if k not in ("status", "approval")} == {
        k: v for k, v in draft.items() if k not in ("status", "approval")}
    assert sum(e.block == SF.BLOCK_FINALS for e in entries) == 35
    assert sum(e.block == SF.BLOCK_REFERENCES for e in entries) == 53
    assert all(e.pairs_with is not None for e in entries if e.block == SF.BLOCK_REFERENCES)
