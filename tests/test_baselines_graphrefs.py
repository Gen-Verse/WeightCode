"""PREREG E7 references (``weightcode.baselines.fit.graphrefs``) and the runner's graphnew mode on the toy inductive store
of tests/test_evalkit_graphnew.py, with a student wide enough (hidden 160) that its head is under the 20% cap.

Covers: parity of the functional rebuild at every node, the head input identity, the trainable set (never the
BatchNorm running statistics), the 19% masks, the interp alphas and theta_full, the runner end to end on grad_last,
imprint and the sparse and unconstrained references, and the dev readouts of graphnew runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_evalkit_graphnew import build_toy_store

from weightcode.baselines.fit import graphrefs as R
from weightcode.baselines.fit.runner import MethodSpec, execute_run, load_graphnew_evidence
from weightcode.common.identity import params_sha256
from weightcode.evalkit import devreadout as D
from weightcode.evalkit import graphnew as GN
from weightcode.models.export import ParityError

HIDDEN = 160


@pytest.fixture()
def wide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    store = build_toy_store(tmp_path, monkeypatch, hidden=HIDDEN)
    (tmp_path / "WEIGHTCODE_RUNS").mkdir()
    monkeypatch.setenv("WEIGHTCODE_RUNS", str(tmp_path / "WEIGHTCODE_RUNS"))
    (outcome,) = GN.build_graphnew_instances()
    instance = GN.load_graphnew_instance(outcome["instance_id"])
    return {**store, "instance": instance, "runs": tmp_path / "WEIGHTCODE_RUNS"}


def _ctx(instance):
    return R._context(instance, load_graphnew_evidence(instance))


def test_trainable_set_and_parity_at_every_node(wide: dict) -> None:
    instance = wide["instance"]
    params = instance.params
    trainable = R.trainable_tensors(params)
    assert not any("running" in key for key in trainable)
    assert set(params) - set(trainable) == {f"bn{i}.{p}" for i in (1, 2) for p in ("running_mean", "running_var")}
    ctx = _ctx(instance)
    parity = ctx.info["parity_all_nodes"]
    assert parity["n_nodes"] == instance.private["n_nodes"]
    assert parity["float64_max_rel_diff"] <= parity["float64_tolerance"]
    # z = A H2 is the exact head input: z W^T + b equals the full logits
    t64 = R.to_tensors(params, torch.float64)
    with torch.no_grad():
        logits = R.gcn_logits(t64, ctx.ops64).numpy()
        z = R.head_input(t64, ctx.ops64).numpy()
    w, b = params["conv3.weight"].astype(np.float64), params["conv3.bias"].astype(np.float64)
    assert np.allclose(z @ w.T + b, logits, rtol=0, atol=1e-9 * max(1.0, np.abs(logits).max()))
    # a rebuild that is not this student fails the parity check
    wrong = dict(params)
    wrong["bn1.running_mean"] = wrong["bn1.running_mean"] + np.float32(0.5)
    with pytest.raises(ParityError):
        R.rebuild_parity(wrong, lambda p, x: instance.forward(dict(params), x), ctx.ops64, ctx.ops32)


def test_mask_fractions_and_ties() -> None:
    score = {"a": np.zeros((10, 10)), "b": np.arange(40, dtype=float)[::-1].copy(), "c": np.ones(7)}
    masks = R.top_fraction_masks(score, ["a", "b", "c"])
    assert masks["a"].sum() == 19 and masks["a"].ravel()[:19].all()  # all ties: the first 19 by flat index
    assert masks["b"].sum() == 7 and masks["b"][:7].all()  # floor(0.19 x 40) = 7, the largest scores
    assert masks["c"].sum() == 1 and masks["c"][0]
    with pytest.raises(R.GraphRefError, match="empty"):
        R.top_fraction_masks({"d": np.ones(5)}, ["d"])


def test_interp_alphas_and_theta_full(wide: dict) -> None:
    assert R.INTERP_ALPHAS == tuple(k / 8 for k in range(1, 9)) and R.INTERP_ALPHAS[-1] == 1.0
    instance = wide["instance"]
    evidence = load_graphnew_evidence(instance)
    candidates, info = R.interp_generator(instance, 0)(evidence)
    assert [c.config["alpha"] for c in candidates] == list(R.INTERP_ALPHAS)
    base = instance.params
    theta = R._theta_full(_ctx(instance), 0)
    assert info["theta_full"]["sha256"] == params_sha256(theta)
    for candidate in candidates:
        alpha = candidate.config["alpha"]
        for key, value in base.items():
            expected = (value.astype(np.float64) + alpha * (theta[key].astype(np.float64) - value)).astype(value.dtype)
            assert np.array_equal(candidate.child[key], expected)
        assert all(np.array_equal(candidate.child[k], base[k]) for k in base if "running" in k)
    # theta_full is grad_full's lr 1e-3, 300-step checkpoint
    grad_full, _ = R.graph_grad_generator(instance, "grad_full", 0)(evidence)
    (q3,) = [c for c in grad_full if c.config == {"lr": R.TRAJ_LR, "steps": R.TRAJ_STEPS}]
    assert params_sha256(q3.child) == params_sha256(theta)


def test_sparse_masks_keep_under_the_cap(wide: dict) -> None:
    instance = wide["instance"]
    evidence = load_graphnew_evidence(instance)
    for method in ("grad_sparse19_sal", "grad_sparse19_traj"):
        candidates, info = R.graph_grad_generator(instance, method, 0)(evidence)
        assert len(candidates) == 8 and info["mask_frac_of_all_entries"] < GN.GRAPHNEW_MAX_CHANGED_FRAC
        trainable = R.trainable_tensors(instance.params)
        for key in trainable:
            k = int(np.floor(R.MASK_FRACTION * instance.params[key].size))
            changed = max(int(np.count_nonzero(c.child[key] != instance.params[key])) for c in candidates)
            assert changed <= k
        assert all(np.array_equal(c.child[key], instance.params[key])
                   for c in candidates for key in instance.params if key not in trainable)


@pytest.mark.parametrize("method", ["grad_last", "imprint"])
def test_runner_end_to_end(wide: dict, method: str) -> None:
    instance = wide["instance"]
    run_dir = wide["runs"] / "baselines" / instance.instance_id / method / R.GRAPHNEW_REF_LABEL / "0"
    record = R.run_graph_ref(instance, method, run_dir, run_seed=0)
    on_disk = json.loads((run_dir / "run.json").read_text())
    assert on_disk["status"] == record["status"] == "complete"
    assert record["queries_used"] == 8 and record["n_invalid"] == 0
    assert record["flags"]["respects_edit_limits"] is True
    assert record["instance"]["evidence_rule"] == GN.GRAPHNEW_RULE
    assert record["evidence"]["n_new"] == 3 and record["evidence"]["n_replay"] == 5
    assert record["graphnew"]["new_classes"] == instance.private["new_classes"]
    assert record["joint"]["replay_rule"] == GN.REPLAY_RULE
    assert record["limits"]["max_changed_frac"] == GN.GRAPHNEW_MAX_CHANGED_FRAC
    assert record["generation"]["parity_all_nodes"]["n_nodes"] == instance.private["n_nodes"]
    final = record["final"]
    assert final["score"] == final["score_new"] + final["score_old"]
    checked = [entry for entry in record["candidates"] if "check" in entry]
    assert final["is_base_noop"] or final["score"] == max(entry["check"]["score"] for entry in checked)
    head = {"conv3.weight", "conv3.bias"}
    if method == "imprint":
        new = instance.private["new_classes"]
        old = instance.private["train_classes"]
        for slot in range(8):
            patch = run_dir / f"q{slot}" / "patch.json"
            assert patch.is_file()
        (candidate,) = [c for c in R.graph_imprint_generator(instance)(load_graphnew_evidence(instance))[0]
                        if c.slot == 0]
        assert np.array_equal(candidate.child["conv3.weight"][old], instance.params["conv3.weight"][old])
        assert not np.array_equal(candidate.child["conv3.weight"][new], instance.params["conv3.weight"][new])
        assert all(np.array_equal(candidate.child[k], instance.params[k]) for k in instance.params if k not in head)
    readout = D.readout_run(run_dir / "run.json")
    json.dumps(readout, allow_nan=False)  # classes without V_select rows read as null, never NaN
    assert readout["rule"] == GN.GRAPHNEW_RULE and readout["readouts_graphnew"]["n_all"] == instance.private["n_vselect"]
    kept = D.readout_retention(run_dir / "run.json", -20)
    assert kept["selection"]["s_old_floor"] == -20
    assert kept["is_base_noop"] or kept["vcheck"]["score_old"] >= -20


def test_unconstrained_reference_is_recorded_out_of_limits(wide: dict) -> None:
    instance = wide["instance"]
    run_dir = wide["runs"] / "grad_full"
    record = R.run_graph_ref(instance, "grad_full", run_dir, run_seed=0)
    assert record["status"] == "complete" and record["queries_used"] == 8
    assert record["executor_limits"]["max_changed_frac"] is None
    assert record["flags"]["respects_edit_limits"] is False
    readout = D.readout_run(run_dir / "run.json")
    assert readout["final_limits"] is None or "without" in readout["final_limits"]


def test_graphnew_mode_refuses_other_modes_and_instances(wide: dict, tmp_path: Path) -> None:
    instance = wide["instance"]
    spec = MethodSpec(method="x", label="x", uses_labels_for_fit=True, edit_limits_enforced=True)
    with pytest.raises(ValueError, match="graphnew alone"):
        execute_run(instance=instance, spec=spec, run_dir=tmp_path / "a", run_seed=0, config={},
                    generate=lambda e: ([], {}), graphnew=True, fewshot=True)
    with pytest.raises(ValueError, match="replay_in_fit"):
        execute_run(instance=instance, spec=spec, run_dir=tmp_path / "b", run_seed=0, config={},
                    generate=lambda e: ([], {}), graphnew=True, replay_in_fit=False)
    evidence = load_graphnew_evidence(instance)
    assert evidence.composition() == "3 new-class nodes, 5 replay nodes of the train classes"
    assert evidence.describe()["n_new"] == 3 and evidence.x.dtype == np.int64


def test_closed_form_references_change_the_head_only(wide: dict) -> None:
    instance = wide["instance"]
    evidence = load_graphnew_evidence(instance)
    head = {"conv3.weight", "conv3.bias"}
    ridge, info = R.ridge_margin_generator(instance)(evidence)
    assert len(ridge) == 8 and info["n_evidence_raised"] > 0
    assert all(np.array_equal(c.child[k], instance.params[k]) for c in ridge for k in instance.params if k not in head)
    if not np.any((evidence.base_pred == evidence.y)[evidence.n_rand:]):
        pytest.skip("the toy base classifies no replay node correctly, so the retain offset has no bound")
    retain, _ = R.graph_imprint_generator(instance, "imprint_retain")(evidence)
    assert [c.config for c in retain] == [c.config for c in R.graph_imprint_generator(instance)(evidence)[0]]
    assert all(c.notes["realized_replay_base_right_broken"] == 0 for c in retain)
