"""The program stream (PREREG E4): experience drawn for the current student, rounds chained under the LEARN limits,
every condition read out after every round, the summary's gains and forgetting, and the refusals.

The store is the evalkit toy store with two named conditions (``toyblur_s2`` and a second name with the same
corruption), so every instance of a seed shares one base model and the companions, replay and joint readout are
evalkit's own. No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_instances_store import CONDITION, ToyStore, add_condition, build_condition, build_toy_store
from test_harness_fixtures import needs_sandbox

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import load_params, params_sha256
from weightcode.evalkit.companions import build_replay
from weightcode.evalkit.instances import instances_root
from weightcode.harness import runner
from weightcode.harness.experience import EXPERIENCE_KEYS, build_experience
from weightcode.harness.program_stream import (
    STATUS_OK,
    STATUS_PATCH_REFUSED,
    StreamError,
    run_stream,
    stream_experience,
    stream_summary,
)
from weightcode.sandbox.program import ProgramSandbox

SECOND = "toyblur2_s2"
STRINGS = 'HYPOTHESIS = "unit 0 should read the class-0 pixel"\nPREDICTED_EFFECT = "fixes errors, breaks none"\n'
FIX = STRINGS + """import numpy as np
def update(student, experience):
    w = student["params"]["fc1.weight"]
    delta = np.zeros_like(w)
    delta[0, 4], delta[1, 4] = 6.0, -6.0
    return {"fc1.weight": delta}
"""
SHIFT = STRINGS + """import numpy as np
def update(student, experience):
    delta = np.zeros_like(student["params"]["fc2.bias"])
    delta[2] = 0.5 * float(len(experience["y_new"]) > 0)
    return {"fc2.bias": delta}
"""
BIG = STRINGS + """import numpy as np
def update(student, experience):
    delta = np.zeros_like(student["params"]["fc1.weight"])
    delta[0, 0] = 1000.0
    return {"fc1.weight": delta}
"""


@dataclass
class StreamStore:
    store: ToyStore
    ids: dict[tuple[str, int], str]

    def instance(self, condition: str, seed: int = 100) -> Instance:
        return load_instance(instances_root() / self.ids[(condition, seed)])


@pytest.fixture
def stream_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> StreamStore:
    store = build_toy_store(tmp_path_factory.mktemp("hs") / "store")
    store.activate(monkeypatch)
    ids: dict[tuple[str, int], str] = {}
    for name in (CONDITION, SECOND):
        add_condition(store, name=name)
        for outcome in build_condition(store, roles=("dev",), name=name):
            if outcome.split_role == "dev":
                ids[(name, outcome.seed)] = outcome.instance_id
    build_replay(store.dataset, n=8)
    return StreamStore(store, ids)


def _sandbox() -> ProgramSandbox:
    return ProgramSandbox(python=runner.SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)


def _program(directory: Path, name: str, source: str) -> Path:
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


def test_on_the_base_model_the_stream_experience_is_the_trial_experience(stream_store: StreamStore) -> None:
    instance = stream_store.instance(CONDITION)
    drawn, summary = stream_experience(instance, instance.params)
    built, built_summary = build_experience(instance)
    assert set(drawn) == set(built) == set(EXPERIENCE_KEYS)
    for key in EXPERIENCE_KEYS:
        assert np.array_equal(drawn[key], built[key]), key
    assert summary["content_sha256"] == built_summary["content_sha256"]
    assert summary["student_params_sha256"] == instance.params_sha256 and summary["n_new_fail"] == 16


def test_the_experience_follows_the_student(stream_store: StreamStore) -> None:
    instance = stream_store.instance(CONDITION)
    other = {name: np.array(value) for name, value in instance.params.items()}
    other["fc2.bias"] = other["fc2.bias"] + np.float32(0.25)
    drawn, summary = stream_experience(instance, other)
    assert summary["student_params_sha256"] == params_sha256(other) != instance.params_sha256
    assert np.array_equal(drawn["pred_old"], _predict(instance, other, drawn["x_old"]))
    assert np.array_equal(drawn["pred_new"], _predict(instance, other, drawn["x_new"]))


def _predict(instance: Instance, params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    from weightcode.evalkit.patch_check import predict

    return predict(instance.forward, params, x, 1024)


@needs_sandbox
def test_a_stream_chains_rounds_and_reads_every_condition(stream_store: StreamStore, tmp_path: Path) -> None:
    fix, shift, big = (_program(tmp_path, name, source) for name, source in
                       (("fix.py", FIX), ("shift.py", SHIFT), ("big.py", BIG)))
    first, second = stream_store.ids[(CONDITION, 100)], stream_store.ids[(SECOND, 100)]
    out = tmp_path / "streams"
    path = run_stream([fix, shift], [first, second], out, _sandbox())
    record = json.loads(path.read_text())
    rounds = record["rounds"]
    assert [row["status"] for row in rounds] == [STATUS_OK, STATUS_OK]
    assert rounds[0]["experience"]["same_as_instance_experience"] is True
    assert rounds[1]["parent_sha256"] == rounds[0]["child_sha256"] == rounds[1]["experience"]["student_params_sha256"]
    assert "same_as_instance_experience" not in rounds[1]["experience"]
    for row in rounds:
        assert set(row["readouts"]) == {first, second}
        assert all(item["select"]["delta_acc"] is not None and item["clean"]["delta_acc"] is not None
                   for item in row["readouts"].values())
    work = Path(record["work_dir"])
    assert params_sha256(load_params(work / "round_2" / "student_after.npz")) == rounds[1]["child_sha256"]
    summary = record["summary"]
    noise = summary["conditions"][CONDITION]
    assert noise["gain_after_own_round"] == rounds[0]["readouts"][first]["select"]["delta_acc"]
    assert noise["gain_after_last_round"] == rounds[1]["readouts"][first]["select"]["delta_acc"]
    assert noise["forgetting"] == pytest.approx(noise["gain_after_own_round"] - noise["gain_after_last_round"])
    assert summary["max_forgetting"] == noise["forgetting"] and summary["n_rounds_ok"] == 2
    assert summary["mean_gain_after_last_round"] == pytest.approx(
        np.mean([summary["conditions"][name]["gain_after_last_round"] for name in (CONDITION, SECOND)]))

    refused = json.loads(run_stream([big, fix], [first, second], out, _sandbox()).read_text())
    assert refused["rounds"][0]["status"] == STATUS_PATCH_REFUSED
    assert refused["rounds"][0]["child_sha256"] == refused["rounds"][0]["parent_sha256"]
    assert any("fc1.weight" in message for message in refused["rounds"][0]["messages"])
    assert refused["rounds"][1]["parent_sha256"] == refused["base_params_sha256"]
    assert refused["rounds"][0]["readouts"][first]["select"]["delta_acc"] == 0.0
    with pytest.raises(StreamError, match="never overwritten"):
        run_stream([fix, shift], [first, second], out, _sandbox())


def test_a_stream_has_one_base_model_and_distinct_rounds(stream_store: StreamStore, tmp_path: Path) -> None:
    fix = _program(tmp_path, "fix.py", FIX)
    with pytest.raises(StreamError, match="different base models"):
        run_stream([fix, fix], [stream_store.ids[(CONDITION, 100)], stream_store.ids[(SECOND, 101)]], tmp_path / "a",
                   _sandbox())
    with pytest.raises(StreamError, match="appears twice"):
        run_stream([fix, fix], [stream_store.ids[(CONDITION, 100)]] * 2, tmp_path / "b", _sandbox())
    with pytest.raises(StreamError, match="one program per round"):
        run_stream([fix], [stream_store.ids[(CONDITION, 100)], stream_store.ids[(SECOND, 100)]], tmp_path / "c",
                   _sandbox())


def test_the_summary_leaves_out_the_excluded_conditions() -> None:
    class Fake:
        def __init__(self, instance_id: str, condition: str) -> None:
            self.instance_id, self.private = instance_id, {"condition": condition}

    def readout(gain: float, clean: float) -> dict:
        return {"select": {"delta_acc": gain, "base_acc": 0.5}, "clean": {"delta_acc": clean}, "clean_split_sha256": "c"}

    instances = [Fake("a", "noise"), Fake("b", "colormix_d1")]
    rounds = [{"status": "ok", "readouts": {"a": readout(0.10, -0.01), "b": readout(0.0, -0.01)}},
              {"status": "ok", "readouts": {"a": readout(0.07, -0.02), "b": readout(0.01, -0.02)}}]
    summary = stream_summary(instances, rounds, ("colormix_d1",))  # type: ignore[arg-type]
    assert summary["mean_gain_after_last_round"] == pytest.approx(0.07)
    assert summary["conditions"]["noise"]["forgetting"] == pytest.approx(0.03) == summary["max_forgetting"]
    assert summary["clean_split_shared"] and summary["clean_delta_acc_mean_after_last_round"] == pytest.approx(-0.02)
