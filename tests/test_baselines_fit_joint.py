"""Joint runs of the fitting references and fixed rules (PREREG E1): replay rows in the evidence, the joint oracle.

The plumbing is tested on the synthetic condition store of ``test_evalkit_instances_store`` with a fixed generator
(that store's MLP has no mean/std block, so the torch rebuild of grad_* and ridge_* does not apply to it). Routing by
method family is tested by capturing the arguments each method passes to ``execute_run``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_evalkit_instances_store import (  # noqa: F401
    ToyStore,
    add_condition,
    breaking_child,
    build_condition,
    condition_dir,
    fixing_child,
    load_npz,
    toy_store,
)

from weightcode.baselines.fit import cli, grad, ridge, rules
from weightcode.baselines.fit.runner import CandidatePatch, Evidence, MethodSpec, execute_run, load_evidence
from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.evalkit import devreadout
from weightcode.evalkit.companions import build_replay
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.patch_check import PatchCheckOracle

N_REPLAY = 8
SPEC = MethodSpec(method="joint_probe", label="test generator", uses_labels_for_fit=True, edit_limits_enforced=False)
"""Unbounded, so the executor checks both children (the breaking child exceeds the edit limits)."""
CHILDREN: tuple[Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]], ...] = (fixing_child, breaking_child)


@pytest.fixture
def joint_instance(toy_store: ToyStore) -> Instance:  # noqa: F811
    """A dev evidence-v1 instance of the toy condition, with the toy dataset's replay drawn."""
    add_condition(toy_store)
    build_replay(toy_store.dataset, n=N_REPLAY)
    outcome = next(o for o in build_condition(toy_store, roles=("dev",)) if o.split_role == "dev")
    return load_instance(instances_root() / outcome.instance_id)


def _run(instance: Instance, run_dir: Path, replay_in_fit: bool) -> tuple[dict[str, Any], list[Evidence]]:
    """One joint run whose generator records the evidence it saw and proposes the fixing and breaking children."""
    seen: list[Evidence] = []

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        seen.append(evidence)
        candidates = [
            CandidatePatch(slot=i, child=make(dict(instance.params)), config={"i": i}, hypothesis="test",
                           predicted_effect="test")
            for i, make in enumerate(CHILDREN)
        ]
        return candidates, {}

    record = execute_run(
        instance=instance, spec=SPEC, run_dir=run_dir, run_seed=0, config={}, generate=generate,
        joint=True, replay_in_fit=replay_in_fit,
    )
    return record, seen


def test_joint_run_scores_both_sets_and_fits_on_replay(
    toy_store: ToyStore, joint_instance: Instance, tmp_path: Path  # noqa: F811
) -> None:
    record, seen = _run(joint_instance, tmp_path / "run", replay_in_fit=True)
    deployment = load_evidence(joint_instance)
    (fit,) = seen
    assert fit.n_replay == N_REPLAY and len(fit.y) == deployment.n_deployment + N_REPLAY
    assert np.array_equal(fit.x[: deployment.n_deployment], deployment.x)
    assert record["evidence"]["n_replay"] == N_REPLAY and record["joint"]["replay_in_fit"] is True
    assert record["joint"]["evidence"]["replay_sha256"] == fit.replay_sha256
    assert record["queries_used"] == len(CHILDREN)
    for entry in record["oracle_log"]:
        assert entry["score"] == entry["score_new"] + entry["score_old"]
    for entry in record["candidates"]:
        assert entry["evidence_summary"]["n"] == deployment.n_deployment
        assert entry["replay_summary"]["n"] == N_REPLAY
    # Every checked score equals two single oracles: the condition V_check plus its clean companion.
    sets = [load_npz(condition_dir(toy_store) / "splits" / "V_check.npz"),
            load_npz(toy_store.private / toy_store.dataset / "splits" / "V_check.npz")]
    for entry, make in zip(record["oracle_log"], CHILDREN):
        child = make(dict(joint_instance.params))
        parts = [PatchCheckOracle(d["x"], d["y"], joint_instance.forward, joint_instance.params, None, 1)
                 .query(child, "single")["score"] for d in sets]
        assert [entry["score_new"], entry["score_old"]] == parts
    final = record["final"]
    assert final["score"] == max([0] + [entry["score"] for entry in record["oracle_log"]])
    assert final["score"] == final["score_new"] + final["score_old"]
    if not final["is_base_noop"]:
        assert final["n_vcheck"] == final["n_vcheck_new"] + final["n_vcheck_old"]
        assert final["delta_acc"] == pytest.approx(final["score"] / final["n_vcheck"])
    # The dev readout reads the joint run's final child on both V_selects.
    readout = devreadout.readout_run(tmp_path / "run" / "run.json", clean=True)
    assert readout["vcheck"]["score"] == final["score"] and "readouts_clean" in readout


def test_the_equal_retention_pick_of_a_joint_run_reads_out_on_both_v_selects(
    joint_instance: Instance, tmp_path: Path
) -> None:
    record, _ = _run(joint_instance, tmp_path / "run", replay_in_fit=True)
    run_path = tmp_path / "run" / "run.json"
    log = record["oracle_log"]
    primary = devreadout.readout_run(run_path, clean=True)
    everything = devreadout.readout_retention(run_path, s_old_floor=-10**9)
    for key in ("readouts", "readouts_clean", "is_base_noop", "final_limits"):
        assert everything[key] == primary[key], key
    assert everything["selection"]["patch_id"] == record["final"]["patch_id"] and "label" not in everything
    floors = sorted({int(entry["score_old"]) for entry in log}) + [10**9]
    for floor in floors:
        picked = devreadout.readout_retention(run_path, s_old_floor=floor, label="post hoc")
        eligible = [entry for entry in log if int(entry["score_old"]) >= floor]
        assert picked["label"] == "post hoc" and picked["selection"]["n_eligible"] == len(eligible)
        assert picked["vcheck"]["score"] == max([0, *(int(entry["score"]) for entry in eligible)])
        if picked["is_base_noop"]:
            assert picked["readouts"]["delta_acc"] == 0.0 and picked["readouts_clean"]["delta_acc"] == 0.0
            assert picked["vcheck"]["score_old"] == 0
            continue
        (entry,) = [item for item in log if item["patch_id"] == picked["selection"]["patch_id"]]
        assert picked["vcheck"]["score_old"] == entry["score_old"] >= floor
        child = devreadout.final_child({"final": {**entry, "dir": entry["patch_id"], "is_base_noop": False},
                                        "flags": record["flags"]}, tmp_path / "run", joint_instance)
        expected = devreadout.dev_readout_joint(joint_instance, child, devreadout.DEFAULT_BATCH_SIZE)
        assert picked["readouts"] == expected["readouts"] and picked["readouts_clean"] == expected["readouts_clean"]
    with pytest.raises(devreadout.DevReadoutError, match="not a fewshot-v1 run"):
        devreadout.readout_fewshot_retention(run_path)
    assert devreadout.main([str(run_path), "--retention-floor", "-40", "--label", "post hoc",
                            "--out", str(tmp_path / "picks.jsonl")]) == 0
    (line,) = (tmp_path / "picks.jsonl").read_text().splitlines()
    assert json.loads(line)["label"] == "post hoc" and json.loads(line)["selection"]["s_old_floor"] == -40


def test_rules_keep_the_deployment_evidence_in_a_joint_run(joint_instance: Instance, tmp_path: Path) -> None:
    record, seen = _run(joint_instance, tmp_path / "run", replay_in_fit=False)
    (fit,) = seen
    assert fit.n_replay == 0 and len(fit.y) == fit.n_deployment
    assert "n_replay" not in record["evidence"] and record["joint"]["evidence"]["n_replay"] == N_REPLAY
    assert all("score_new" in entry for entry in record["oracle_log"])


def test_joint_refuses_an_instance_without_a_clean_companion(joint_instance: Instance, tmp_path: Path) -> None:
    joint_instance.private["condition"] = "deploy"
    with pytest.raises(ValueError, match="named condition"):
        execute_run(instance=joint_instance, spec=SPEC, run_dir=tmp_path / "run", run_seed=0, config={},
                    generate=lambda evidence: ([], {}), joint=True)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("method", "module", "replay_in_fit"),
    [
        ("grad_last", grad, True),
        ("grad_full", grad, True),
        ("ridge_last", ridge, True),
        ("prior_shift", rules, False),
        ("tie_inputs", rules, False),
        ("input_stats", rules, False),
    ],
)
def test_run_method_routes_the_replay_by_method_family(
    monkeypatch: pytest.MonkeyPatch, method: str, module: Any, replay_in_fit: bool, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_execute_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(module, "execute_run", fake_execute_run)
    for name in ("grad_generator", "ridge_generator", "prior_shift_generator", "tie_generator", "input_stats_generator"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    instance = object.__new__(Instance)
    cli.run_method(instance, method, tmp_path / "run", run_seed=0, allow_main=False, joint=True)
    assert captured["joint"] is True and captured["replay_in_fit"] is replay_in_fit
