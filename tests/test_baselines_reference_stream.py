"""E4 sequential label-fit references: the program stream's rounds, experience, readouts and summary, with each
round's configuration frozen from the method's E1 joint run (weightcode.baselines.fit.reference_stream).

The store is the program stream's toy store (two named conditions sharing one base model per seed). Its MLP has no
mean/std block, so the torch rebuild of grad_* and ridge_* does not apply to it, and the streams here run a fixed
candidate grid on the current student, as the E1 joint plumbing tests do; the E1 generators have their own tests.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_evalkit_instances_store import CONDITION
from test_harness_program_stream import SECOND, StreamStore, stream_store  # noqa: F401

import weightcode.harness.program_stream as program_stream
from weightcode.baselines.fit import reference_stream as R
from weightcode.baselines.fit.cli import JOINT_REF_LABEL
from weightcode.baselines.fit.runner import RUN_SCHEMA, CandidatePatch, Evidence, baseline_run_dir
from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import load_params, params_sha256

REPO = Path(__file__).resolve().parents[1]
OVERSIZED = {"lr": 1.0, "steps": 1}
GRID: dict[str, list[dict[str, Any]]] = {
    "grad_last": [{"lr": 0.01, "steps": 30}, {"lr": 0.001, "steps": 10}, OVERSIZED],
    "grad_full": [{"lr": 0.01, "steps": 30}, {"lr": 0.001, "steps": 10}, OVERSIZED],
    "ridge_last": [{"lam_relative": 1.0}, {"lam_relative": 10.0}],
}


class FixedGenerator:
    """A fixed grid on the current student: slot k shifts fc2.bias by 0.01 (k + 1); OVERSIZED rewrites fc1.weight."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def __call__(self, method: str, instance: Instance, run_seed: int):  # noqa: ANN204
        def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
            self.seen.append({"method": method, "parent": instance.params_sha256, "n_rows": len(evidence.y),
                              "n_fail": evidence.n_fail, "n_replay": evidence.n_replay, "run_seed": run_seed})
            candidates = []
            for slot, config in enumerate(GRID[method]):
                child = {name: np.array(value) for name, value in instance.params.items()}
                if config == OVERSIZED:
                    child["fc1.weight"] = child["fc1.weight"] * np.float32(10.0) + np.float32(1.0)
                else:
                    child["fc2.bias"] = child["fc2.bias"] + np.float32(0.01 * (slot + 1))
                candidates.append(CandidatePatch(slot=slot, child=child, config=dict(config), hypothesis="fixed grid",
                                                 predicted_effect="none", notes={"slot": slot}))
            return candidates, {"parity_max_abs_logit_diff": 0.0, "trainable": ["fc2.bias"]}

        return generate


def _ids(store: StreamStore, seed: int = 100) -> list[str]:
    return [store.ids[(CONDITION, seed)], store.ids[(SECOND, seed)]]


def _choices(store: StreamStore, configs: list[dict[str, Any] | None]) -> list[R.FrozenChoice]:
    instances = R.load_stream_instances(_ids(store))
    return [R.FrozenChoice(instance_id=i.instance_id, condition=str(i.private["condition"]), config=config,
                           source_instance_id=i.instance_id, source="test", source_final={})
            for i, config in zip(instances, configs)]


def _write_e1_run(instance_id: str, method: str, base_sha: str, final: dict[str, Any],
                  candidates: list[dict[str, Any]]) -> Path:
    run_dir = baseline_run_dir(instance_id, method, 0, JOINT_REF_LABEL)
    run_dir.mkdir(parents=True)
    path = run_dir / "run.json"
    path.write_text(json.dumps({"schema": RUN_SCHEMA, "status": "complete", "method": method, "joint": {},
                                "instance": {"params_sha256": base_sha}, "final": final, "candidates": candidates}))
    return path


def test_frozen_choices_take_the_config_each_e1_joint_run_kept(stream_store: StreamStore) -> None:  # noqa: F811
    instances = R.load_stream_instances(_ids(stream_store))
    base_sha = instances[0].params_sha256
    _write_e1_run(instances[0].instance_id, "grad_last", base_sha,
                  {"patch_id": "q3", "is_base_noop": False, "score": 7, "query_index": 2},
                  [{"patch_id": "q1", "config": {"lr": 0.01, "steps": 10}},
                   {"patch_id": "q3", "config": {"lr": 0.001, "steps": 300}}])
    _write_e1_run(instances[1].instance_id, "grad_last", base_sha,
                  {"patch_id": None, "is_base_noop": True, "score": 0, "query_index": None}, [])
    choices = R.frozen_choices("grad_last", instances)
    assert [c.config for c in choices] == [{"lr": 0.001, "steps": 300}, None]
    assert choices[0].source_final["score"] == 7 and choices[0].condition == CONDITION
    assert [c.source_instance_id for c in choices] == [i.instance_id for i in instances]
    _write_e1_run(instances[0].instance_id, "ridge_last", "0" * 64,
                  {"patch_id": "q0", "is_base_noop": False, "score": 1}, [{"patch_id": "q0", "config": {}}])
    with pytest.raises(R.ReferenceStreamError, match="base model differs"):
        R.frozen_choices("ridge_last", instances)
    with pytest.raises(R.ReferenceStreamError, match="no E1 joint grad_full run"):
        R.frozen_choices("grad_full", instances)


def test_another_seed_takes_the_choices_of_the_same_condition(stream_store: StreamStore) -> None:  # noqa: F811
    source = R.load_stream_instances(_ids(stream_store, 100))
    target = R.load_stream_instances(_ids(stream_store, 101))
    assert source[0].params_sha256 != target[0].params_sha256
    for instance, steps in zip(source, (300, 30)):
        _write_e1_run(instance.instance_id, "grad_last", instance.params_sha256,
                      {"patch_id": "q0", "is_base_noop": False, "score": 5, "query_index": 0},
                      [{"patch_id": "q0", "config": {"lr": 0.001, "steps": steps}}])
    choices = R.frozen_choices("grad_last", target[::-1], source_instances=source)
    assert [c.instance_id for c in choices] == [target[1].instance_id, target[0].instance_id]
    assert [c.source_instance_id for c in choices] == [source[1].instance_id, source[0].instance_id]
    assert [c.config["steps"] for c in choices if c.config is not None] == [30, 300]
    with pytest.raises(R.ReferenceStreamError, match="no frozen-from instance has its condition"):
        R.frozen_choices("grad_last", target, source_instances=source[:1])
    with pytest.raises(R.ReferenceStreamError, match="share the condition"):
        R.frozen_choices("grad_last", target, source_instances=[source[0], target[0]])


def test_a_reference_stream_follows_the_program_stream_rounds(stream_store: StreamStore,  # noqa: F811
                                                              tmp_path: Path) -> None:
    ids = _ids(stream_store)
    instances = R.load_stream_instances(ids)
    choices = _choices(stream_store, [{"lr": 0.01, "steps": 30}, {"lr": 0.01, "steps": 30}])
    generator = FixedGenerator()
    path = R.run_reference_stream("grad_last", "replay", ids, tmp_path / "out", choices=choices,
                                  make_generator=generator)
    record = json.loads(path.read_text())
    rounds = record["rounds"]
    assert [row["status"] for row in rounds] == [R.STATUS_OK, R.STATUS_OK]
    assert rounds[1]["parent_sha256"] == rounds[0]["child_sha256"] != rounds[0]["parent_sha256"]
    assert rounds[0]["parent_sha256"] == record["base_params_sha256"] == instances[0].params_sha256
    assert [seen["parent"] for seen in generator.seen] == [row["parent_sha256"] for row in rounds]
    students = [load_params(Path(record["work_dir"]) / f"round_{k}" / "student_after.npz") for k in (1, 2)]
    np.testing.assert_allclose(students[1]["fc2.bias"], instances[0].params["fc2.bias"] + np.float32(0.02), atol=1e-6)
    previous = instances[0].params
    for row, student, instance, seen in zip(rounds, students, instances, generator.seen):
        assert params_sha256(student) == row["child_sha256"]
        _, expected = program_stream.stream_experience(instance, previous)
        assert row["experience"] == json.loads(json.dumps(R.jsonable(expected)))
        assert row["fit_rows"] == {"n_round_evidence": expected["n_new"], "n_earlier_evidence": 0,
                                   "n_replay": expected["n_old"], "earlier_rounds": 0}
        assert (seen["n_rows"], seen["n_fail"], seen["n_replay"]) == (
            expected["n_new"] + expected["n_old"], expected["n_new_fail"], expected["n_old"])
        assert row["evidence"]["n_fail"] == expected["n_new_fail"]
        assert row["candidate"]["config"] == {"lr": 0.01, "steps": 30}
        assert row["edit"]["within_limits"] and set(row["rel_frob"]) == {"fc2.bias"}
        for other in instances:
            assert row["readouts"][other.instance_id] == json.loads(json.dumps(R.jsonable(
                program_stream._readout(other, student, R.DEFAULT_BATCH_SIZE))))
        previous = student
    assert record["summary"] == json.loads(json.dumps(R.jsonable(
        program_stream.stream_summary(instances, rounds, program_stream.E4_MEAN_EXCLUDES))))
    assert record["summary"]["n_rounds_ok"] == 2 and record["variant"] == "replay" and record["frozen_from"] is None
    with pytest.raises(R.ReferenceStreamError, match="never overwritten"):
        R.run_reference_stream("grad_last", "replay", ids, tmp_path / "out", choices=choices,
                               make_generator=generator)


def test_the_cumulative_variant_adds_the_earlier_rounds_rows(stream_store: StreamStore,  # noqa: F811
                                                             tmp_path: Path) -> None:
    ids = _ids(stream_store)
    choices = _choices(stream_store, [{"lam_relative": 1.0}, {"lam_relative": 10.0}])
    generator = FixedGenerator()
    record = json.loads(R.run_reference_stream("ridge_last", "cumulative", ids, tmp_path, choices=choices,
                                               make_generator=generator).read_text())
    first, second = record["rounds"]
    assert first["fit_rows"]["n_earlier_evidence"] == 0
    assert second["fit_rows"]["n_earlier_evidence"] == first["fit_rows"]["n_round_evidence"]
    assert second["evidence"]["n_replay"] == second["fit_rows"]["n_replay"] + first["fit_rows"]["n_round_evidence"]
    assert generator.seen[1]["n_rows"] == (second["fit_rows"]["n_round_evidence"] + second["fit_rows"]["n_replay"]
                                           + first["fit_rows"]["n_round_evidence"])
    assert R.EARLIER_RULE in second["evidence"]["replay_rule"] and first["evidence"]["replay_rule"] == "replay-v1"
    assert [row["candidate"]["config"]["lam_relative"] for row in (first, second)] == [1.0, 10.0]


def test_frozen_no_ops_and_refused_patches_keep_the_student(stream_store: StreamStore,  # noqa: F811
                                                            tmp_path: Path) -> None:
    ids = _ids(stream_store)
    generator = FixedGenerator()
    choices = _choices(stream_store, [None, {"lr": 0.001, "steps": 10}])
    record = json.loads(R.run_reference_stream("grad_full", "replay", ids, tmp_path / "a", choices=choices,
                                               make_generator=generator).read_text())
    first, second = record["rounds"]
    assert first["status"] == R.STATUS_NOOP and first["child_sha256"] == first["parent_sha256"]
    assert all(item["select"]["delta_acc"] == 0.0 for item in first["readouts"].values())
    assert second["status"] == R.STATUS_OK and second["parent_sha256"] == record["base_params_sha256"]
    assert record["edit_limits_enforced"] is False and record["summary"]["n_rounds_ok"] == 1
    assert len(generator.seen) == 1  # a no-op round runs no generator
    refused = json.loads(R.run_reference_stream("grad_last", "replay", ids, tmp_path / "b",
                                                choices=_choices(stream_store, [OVERSIZED, {"lr": 0.01, "steps": 30}]),
                                                make_generator=generator).read_text())
    first, second = refused["rounds"]
    assert first["status"] == R.STATUS_PATCH_REFUSED and first["child_sha256"] == first["parent_sha256"]
    assert any(message.startswith("size:") for message in first["messages"])  # E1 limits: edit size <= 5%
    assert second["status"] == R.STATUS_OK and second["parent_sha256"] == refused["base_params_sha256"]
    with pytest.raises(R.ReferenceStreamError, match="0 candidates match"):
        R.run_reference_stream("grad_last", "replay", ids, tmp_path / "c",
                               choices=_choices(stream_store, [{"lr": 0.5, "steps": 7}, None]),
                               make_generator=generator)


def test_refusals_happen_before_anything_is_written(stream_store: StreamStore, tmp_path: Path) -> None:  # noqa: F811
    ids = _ids(stream_store)
    choices = _choices(stream_store, [None, None])
    for method, variant, match in (("grad_sparse", "replay", "unknown method"), ("grad_last", "all", "variant")):
        with pytest.raises(R.ReferenceStreamError, match=match):
            R.run_reference_stream(method, variant, ids, tmp_path / "x", choices=choices)
    with pytest.raises(R.ReferenceStreamError, match="appears twice"):
        R.run_reference_stream("grad_last", "replay", [ids[0], ids[0]], tmp_path / "x", choices=choices)
    with pytest.raises(R.ReferenceStreamError, match="different base models"):
        R.run_reference_stream("grad_last", "replay", [ids[0], stream_store.ids[(SECOND, 101)]], tmp_path / "x",
                               choices=choices)
    with pytest.raises(R.ReferenceStreamError, match="do not follow"):
        R.run_reference_stream("grad_last", "replay", ids, tmp_path / "x", choices=choices[::-1])
    with pytest.raises(R.ReferenceStreamError, match="not both"):
        R.run_reference_stream("grad_last", "replay", ids, tmp_path / "x", choices=choices, frozen_from=ids)
    assert not (tmp_path / "x").exists()
    assert R.METHODS == ("grad_last", "ridge_last", "grad_full") and R.VARIANTS == ("replay", "cumulative")
    assert R.e1_generator is R.run_reference_stream.__kwdefaults__["make_generator"]


def test_the_script_refuses_unknown_methods_before_running(stream_store: StreamStore,  # noqa: F811
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("run_reference_stream", REPO / "scripts" / "run_reference_stream.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--instances", *_ids(stream_store), "--methods", "grad_sparse", "--threads", "1"]) == 2
    assert script.main(["--instances", *_ids(stream_store), "--variants", "all", "--threads", "1"]) == 2
    assert "refused" in capsys.readouterr().err
