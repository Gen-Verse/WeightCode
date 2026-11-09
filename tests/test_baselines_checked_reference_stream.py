"""Control (b) of E4-loop: label fits re-selected each round by the loop's check and kept rule loop-retain-v1, on their
own current student, with the loop's rows (weightcode.baselines.fit.checked_reference_stream).

The store is the program stream's toy store (two named conditions sharing one base model per seed). Its MLP has no
mean/std block, so the torch rebuild of grad_* and ridge_* does not apply to it, and the streams here run a fixed
candidate grid on the current student, as the reference-stream tests do; the E1 generators have their own tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_evalkit_instances_store import CONDITION
from test_harness_program_stream import SECOND, StreamStore, stream_store  # noqa: F401

import weightcode.harness.program_stream as program_stream
from weightcode.baselines.fit import checked_reference_stream as C
from weightcode.baselines.fit.reference_stream import load_stream_instances
from weightcode.baselines.fit.runner import CandidatePatch, Evidence, jsonable
from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import load_params, params_sha256
from weightcode.harness import loop, loop_stream

Delta = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def _fix(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    delta = np.zeros_like(params["fc1.weight"])
    delta[0, 4], delta[1, 4] = 6.0, -6.0
    return {"fc1.weight": delta}


def _shift(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    delta = np.zeros_like(params["fc2.bias"])
    delta[2] = 0.01
    return {"fc2.bias": delta}


def _break(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"fc2.weight": -2.0 * params["fc2.weight"]}


def _flood(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One bias entry, within every E1 limit, that sends most images to class 3."""
    delta = np.zeros_like(params["fc2.bias"])
    delta[3] = 50.0
    return {"fc2.bias": delta}


def _oversized(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"fc1.weight": params["fc1.weight"] * np.float32(9.0) + np.float32(1.0)}


class Grid:
    """A fixed grid of deltas on the current student, recording what each round's generator was given."""

    def __init__(self, deltas: list[tuple[str, Delta]]) -> None:
        self.deltas = deltas
        self.seen: list[dict[str, Any]] = []

    def __call__(self, method: str, instance: Instance, run_seed: int):  # noqa: ANN204
        def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
            self.seen.append({"method": method, "parent": instance.params_sha256, "x": evidence.x, "y": evidence.y,
                              "n_fail": evidence.n_fail, "n_replay": evidence.n_replay})
            candidates = []
            for slot, (name, delta) in enumerate(self.deltas):
                child = {key: np.array(value) for key, value in instance.params.items()}
                for key, value in delta(child).items():
                    child[key] = (child[key] + value).astype(np.float32)
                candidates.append(CandidatePatch(slot=slot, child=child, config={"name": name}, hypothesis="grid",
                                                 predicted_effect="none", notes={}))
            return candidates, {"parity_max_abs_logit_diff": 0.0, "trainable": ["fc1.weight", "fc2.weight"]}

        return generate


def _ids(store: StreamStore) -> list[str]:
    return [store.ids[(CONDITION, 100)], store.ids[(SECOND, 100)]]


def _expected_pick(row: dict[str, Any]) -> str | None:
    pool = [c for c in row["candidates"] if c["status"] == "checked" and c["eligible"]]
    best = min(pool, key=lambda c: (-c["check"]["score"], c["n_changed"], c["check"]["query_index"]), default=None)
    return None if best is None or best["check"]["score"] <= 0 else best["patch_id"]


def test_each_round_checks_every_fit_on_the_current_student_and_keeps_by_the_loop_rule(
        stream_store: StreamStore, tmp_path: Path) -> None:  # noqa: F811
    ids = _ids(stream_store)
    instances = load_stream_instances(ids)
    grid = Grid([("fix", _fix), ("shift", _shift), ("break", _break)])
    path = C.run_checked_reference_stream("grad_full", ids, tmp_path / "out", make_generator=grid)
    record = json.loads(path.read_text())
    one, two = record["rounds"]
    work = Path(record["work_dir"])
    base = instances[0].params
    # Round 1: the base student, the round's own V_check and clean companion, the clean floor only.
    assert one["parent_sha256"] == record["base_params_sha256"] == instances[0].params_sha256
    assert one["kept_rule"]["s_prev_floor"] == {} and one["kept_rule"]["name"] == loop.KEPT_RULE_LOOP
    assert one["kept_rule"]["s_old_floor"] == -max(1, round(loop.LOOP_CLEAN_FRAC * one["check_parts"]["old"]))
    assert list(one["check_parts"]) == ["new", "old"] and list(two["check_parts"]) == ["new", "old", "prev_1"]
    assert two["kept_rule"] == loop.loop_kept_rule(two["check_parts"]["old"], {1: two["check_parts"]["prev_1"]})
    oracle = loop.MultiPartOracle(loop_stream.check_parts(instances[0], []), instances[0].forward, base, 8)
    fixed = {key: (value + _fix(dict(base)).get(key, 0)).astype(np.float32) for key, value in base.items()}
    entry = oracle.query(fixed, "fix")
    assert one["candidates"][0]["check"]["score"] == entry["score"] > 0
    for row in (one, two):
        assert row["queries_used"] == 3 and [c["status"] for c in row["candidates"]] == ["checked"] * 3
        for candidate in row["candidates"]:
            check = candidate["check"]
            parts = [name for name in row["check_parts"]]
            assert check["score"] == sum(check[f"score_{name}"] for name in parts)
            assert candidate["eligible"] == loop.eligible(check, row["kept_rule"])
        assert row["kept"] == _expected_pick(row)
        assert row["status"] == (C.STATUS_OK if row["kept"] else C.STATUS_NOOP)
    assert not one["candidates"][2]["eligible"]  # the broken head fails the clean floor
    # Rows: the loop's evidence and experience functions on this stream's own students.
    assert two["parent_sha256"] == one["child_sha256"] == params_sha256(load_params(work / "student_after_round_1.npz"))
    student1 = load_params(work / "student_after_round_1.npz")
    evidence1 = loop_stream.evidence_for(instances[0], base)
    expected, summary = loop.loop_experience(instances[1], student1, [(1, evidence1)])
    assert two["experience"]["content_sha256"] == params_sha256(expected) == summary["content_sha256"]
    assert two["experience"]["rows_sha256"] == C.rows_sha256(expected) and set(C.rows_sha256(expected)) == {
        "new", "old", "prev"}
    seen = grid.seen[1]
    assert seen["parent"] == two["parent_sha256"]
    assert np.array_equal(seen["x"], np.concatenate([expected["x_new"], expected["x_prev"], expected["x_old"]]))
    assert np.array_equal(seen["y"], np.concatenate([expected["y_new"], expected["y_prev"], expected["y_old"]]))
    assert seen["n_replay"] == len(expected["y_prev"]) + len(expected["y_old"])
    assert two["fit_rows"]["n_earlier_evidence"] == summary["n_prev"] == two["fit_rows"]["earlier_per_round"]["1"]
    assert summary["n_prev"] + summary["prev_dropped_shared_image"]["1"] == len(evidence1["y"])
    # Readouts and summary are the program stream's.
    for row, name in ((one, "student_after_round_1.npz"), (two, "student_after_round_2.npz")):
        student = load_params(work / name)
        assert params_sha256(student) == row["child_sha256"] == row["student_after"]["sha256"]
        for other in instances:
            assert row["readouts"][other.instance_id] == json.loads(json.dumps(jsonable(
                program_stream._readout(other, student, C.DEFAULT_BATCH_SIZE))))
    assert record["summary"] == json.loads(json.dumps(jsonable(
        program_stream.stream_summary(instances, record["rounds"], program_stream.E4_MEAN_EXCLUDES))))
    with pytest.raises(C.CheckedReferenceError, match="never overwritten"):
        C.run_checked_reference_stream("grad_full", ids, tmp_path / "out", make_generator=grid)


def test_the_loop_rows_are_rebuilt_and_compared_round_by_round(stream_store: StreamStore,  # noqa: F811
                                                               tmp_path: Path) -> None:
    ids = _ids(stream_store)
    first = C.run_checked_reference_stream("grad_full", ids, tmp_path / "a",
                                           make_generator=Grid([("fix", _fix), ("shift", _shift)]))
    rebuilt = C.loop_round_rows(first)  # the record has a loop record's round fields
    record = json.loads(first.read_text())
    assert [row["content_sha256"] for row in rebuilt] == [row["experience"]["content_sha256"]
                                                         for row in record["rounds"]]
    assert record["rounds"][0]["status"] == C.STATUS_OK
    # A stream whose round 1 keeps the base shares round 1 with the loop and only the replay rows after it.
    second = json.loads(C.run_checked_reference_stream("grad_last", ids, tmp_path / "b", loop_rows=rebuilt,
                                                       loop_record=first,
                                                       make_generator=Grid([("flood", _flood),
                                                                            ("oversized", _oversized)])).read_text())
    one, two = second["rounds"]
    assert one["status"] == C.STATUS_NOOP and one["child_sha256"] == one["parent_sha256"]
    assert [c["status"] for c in one["candidates"]] == ["checked", "patch_refused"] and one["queries_used"] == 1
    assert one["loop_rows"]["same_student"] and all(one["loop_rows"]["same_rows"].values())
    assert one["loop_rows"]["same_content"]
    assert not two["loop_rows"]["same_student"] and two["loop_rows"]["same_rows"]["old"]
    assert not two["loop_rows"]["same_rows"]["new"] and not two["loop_rows"]["same_content"]
    assert all(item["select"]["delta_acc"] == 0.0 for item in one["readouts"].values())
    tampered = [dict(row) for row in rebuilt]
    tampered[1] = {**tampered[1], "rows_sha256": {**tampered[1]["rows_sha256"], "old": "0" * 64}}
    with pytest.raises(C.CheckedReferenceError, match="replay rows are not the loop's"):
        C.run_checked_reference_stream("grad_last", ids, tmp_path / "c", loop_rows=tampered,
                                       make_generator=Grid([("flood", _flood)]))
    changed = json.loads(first.read_text())
    changed["rounds"][1]["experience"]["content_sha256"] = "0" * 64
    (tmp_path / "changed.json").write_text(json.dumps(changed))
    with pytest.raises(C.CheckedReferenceError, match="the loop recorded"):
        C.loop_round_rows(tmp_path / "changed.json")


def test_refusals_happen_before_anything_is_written(stream_store: StreamStore, tmp_path: Path) -> None:  # noqa: F811
    ids = _ids(stream_store)
    with pytest.raises(C.CheckedReferenceError, match="2 to 8 rounds"):
        C.run_checked_reference_stream("grad_last", ids[:1], tmp_path / "x")
    with pytest.raises(Exception, match="unknown method"):
        C.run_checked_reference_stream("grad_sparse", ids, tmp_path / "x")
    with pytest.raises(C.CheckedReferenceError, match="do not follow"):
        C.run_checked_reference_stream("grad_last", ids, tmp_path / "x", loop_rows=[{"instance_id": i}
                                                                                    for i in ids[::-1]])
    assert not (tmp_path / "x").exists()
    assert C.round_rule({"old": 4000}, 0)["s_old_floor"] == -12
    assert C.round_rule({"old": 4000, "prev_1": 4000}, 1)["s_prev_floor"] == {"1": -20}
