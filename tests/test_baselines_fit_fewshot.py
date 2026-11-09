"""The PREREG E2 references (imprint, ridge_margin, grad_last, grad_full) on fewshot-v1 instances."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_fewshot import NEW, fewshot_store  # noqa: F401

from weightcode.baselines.fit import imprint
from weightcode.baselines.fit.cli import FEWSHOT_METHODS, run_method
from weightcode.baselines.fit.edits import edit_stats, edit_stats_under, scale_to_norm_limit
from weightcode.baselines.fit.grad import run_grad
from weightcode.baselines.fit.imprint import fit_shared_offset
from weightcode.baselines.fit.ridge import margin_targets
from weightcode.baselines.fit.runner import load_fewshot_evidence
from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.evalkit import devreadout
from weightcode.evalkit import fewshot as F
from weightcode.evalkit.patch_check import predict
from weightcode.harness.arms import PROGRAM_MAX_CHANGED_FRAC, applied_limits
from weightcode.patch import Limits, task_limits

OLD = [0, 1, 2, 3, 4]
REPO = Path(__file__).resolve().parents[1]


def _instance(arch: str = "cnn", seed: int = 100) -> Instance:
    (outcome,) = F.build_fewshot_instances("cifar10", arch=arch, seeds=[seed])
    return load_instance(F.fewshot_instance_dir(outcome["instance_id"]))


def test_program_limits_are_the_learn_limits_of_the_same_task(fewshot_store: list[dict[str, object]]) -> None:  # noqa: F811
    assert F.FEWSHOT_MAX_CHANGED_FRAC == PROGRAM_MAX_CHANGED_FRAC
    for outcome in F.build_fewshot_instances("cifar10"):
        task = load_instance(F.fewshot_instance_dir(outcome["instance_id"])).task
        program = F.fewshot_program_limits(task["limits"])
        assert program == applied_limits(task["limits"], "LEARN")
        assert program.rel_frob_exempt == tuple(task["limits"]["rel_frob_exempt"])
        assert program.max_edit_frac is None and program.max_changed_frac == pytest.approx(0.20)
        assert program.max_rel_frob == task["limits"]["max_rel_frob"]


def test_the_norm_limit_helpers_skip_exempt_tensors() -> None:
    rng = np.random.default_rng(0)
    base = {
        "a.weight": rng.normal(size=(4, 3)).astype(np.float32),
        "b.weight": rng.normal(size=(2, 3)).astype(np.float32),
        "b.bias": rng.normal(size=2).astype(np.float32),
    }
    child = {key: value + np.float32(3.0) * np.abs(value).max() for key, value in base.items()}  # far beyond 0.5
    scaled, records = scale_to_norm_limit(base, child, list(base), exempt=["b.weight", "b.bias"])
    assert records["a.weight"]["scaled"] and records["a.weight"]["rel_frob_after"] <= 0.5
    assert records["b.weight"]["exempt"] and not records["b.weight"]["scaled"]
    np.testing.assert_array_equal(scaled["b.weight"], child["b.weight"])
    plain = edit_stats(base, scaled, max_edit_frac=None)
    assert [v.split(":")[0] for v in plain["violations"]] == ["b.weight"]
    exempt = edit_stats(base, scaled, max_edit_frac=None, rel_frob_exempt=["b.weight", "b.bias"])
    assert exempt["within_limits"] and exempt["per_tensor"]["b.weight"]["rel_frob_exempt"]
    limits = Limits(rel_frob_exempt=("b.weight", "b.bias"), max_edit_frac=None)
    assert edit_stats_under(base, scaled, limits) == exempt
    assert not edit_stats_under(base, scaled, Limits(max_edit_frac=None, max_changed_frac=0.2))["within_limits"]
    for call in (lambda: scale_to_norm_limit(base, child, list(base), exempt=["c.weight"]),
                 lambda: edit_stats(base, child, rel_frob_exempt=["c.weight"])):
        with pytest.raises(ValueError, match="not tensors of the model"):
            call()


def test_margin_targets_raise_only_the_true_logit_where_the_margin_is_short() -> None:
    z0 = np.array([[5.0, 0.0, -1.0], [0.0, 1.0, 9.0], [2.0, 1.9, 0.0]])
    y = np.array([0, 0, 1])
    targets, scale = margin_targets(z0, y)
    assert scale == pytest.approx(float(np.mean(z0.max(1) - z0.mean(1))))
    expected = z0.copy()
    expected[1, 0] = 9.0 + scale  # wrong: raised to the best other logit + c
    expected[2, 1] = 2.0 + scale  # wrong: likewise
    # Row 0 is right with margin 5 > c, so it is left alone.
    np.testing.assert_allclose(targets, expected)
    assert np.all(targets.argmax(1) == y)


def test_fit_shared_offset_finds_the_best_offset() -> None:
    rng = np.random.default_rng(3)
    n_new, n_old = 30, 40
    y = np.concatenate([rng.choice(NEW, n_new), rng.choice(OLD, n_old)])
    is_new = np.arange(len(y)) < n_new
    z_old = 3.0 * rng.normal(size=(len(y), 5))
    u_new = 3.0 * rng.normal(size=(len(y), 5)) - 2.0
    fit = fit_shared_offset(z_old, u_new, y, OLD, NEW, is_new, step=1.0)

    def balanced(beta: float) -> float:
        right = np.concatenate([z_old, u_new + beta], axis=1).argmax(1) == y  # columns: classes 0-4, then 5-9
        return float((right[is_new].mean() + right[~is_new].mean()) / 2)

    assert balanced(fit["beta"]) == pytest.approx(fit["balanced_acc"])
    assert fit["balanced_acc"] >= max(balanced(beta) for beta in np.linspace(-25.0, 25.0, 5001)) - 1e-12
    low, high = fit["interval"]
    assert low < fit["beta"] < high
    assert (fit["acc_new"] + fit["acc_old"]) / 2 == pytest.approx(fit["balanced_acc"])
    with pytest.raises(ValueError, match="new-class rows and replay rows"):
        fit_shared_offset(z_old, u_new, y, OLD, NEW, np.zeros(len(y), dtype=bool), step=1.0)


def test_the_four_references_run_under_the_e2_limits(fewshot_store: list[dict[str, object]],  # noqa: F811
                                                     tmp_path: Path) -> None:
    instance = _instance("cnn", 100)
    program = F.fewshot_program_limits(instance.task["limits"])
    head = ["fc2.weight", "fc2.bias"]
    for method in FEWSHOT_METHODS:
        run_dir = tmp_path / "runs" / method
        record = run_method(instance, method, run_dir, run_seed=0, allow_main=False, fewshot=True)
        assert record["status"] == "complete" and record["method"] == method
        assert record["limits"] == task_limits(program) and record["limits_source"] == "fewshot_program_limits" \
            "(task.json limits): PREREG E2, the limits of an E2 update program"
        evidence = record["evidence"]
        assert evidence == record["joint"]["evidence"]
        assert (evidence["rule"], evidence["n_new"], evidence["n_replay"]) == (F.FEWSHOT_RULE, 50, 64)
        assert evidence["replay_rule"] == F.REPLAY_RULE and record["joint"]["replay_in_fit"] is True
        assert record["fewshot"] == {"rule": F.FEWSHOT_RULE, "train_classes": OLD, "new_classes": NEW,
                                     "head_tensors": head}
        assert 1 <= record["queries_used"] <= 8
        final = record["final"]
        assert final["score"] == final["score_new"] + final["score_old"]
        assert final["n_vcheck_new"] + final["n_vcheck_old"] == instance.private["n_vcheck"]
        valid = [candidate for candidate in record["candidates"] if "edit" in candidate]
        assert valid and all(candidate["roundtrip_exact"] for candidate in valid)
        if method == "grad_full":
            assert record["flags"]["respects_edit_limits"] is False
        else:
            for candidate in valid:
                changed = {key for key, stats in candidate["edit"]["per_tensor"].items() if stats["changed"]}
                assert changed <= set(head) and candidate["edit"]["within_limits"]
            assert record["flags"]["respects_edit_limits"] is True
        if method in ("grad_last", "ridge_margin"):
            for candidate in valid:
                scaling = candidate["notes"]["norm_scaling"]
                assert set(scaling) == set(head)
                assert all(entry["exempt"] and not entry["scaled"] for entry in scaling.values())
            assert record["config"]["norm_limit_exempt"] == record["generation"]["norm_limit_exempt"] == head
        if method == "imprint":
            # Without the exemption the executor would refuse these head edits.
            assert max(c["edit"]["per_tensor"]["fc2.weight"]["rel_frob"] for c in valid) > 0.5
            for candidate in valid:
                notes = candidate["notes"]
                assert abs(notes["realized_balanced_acc"] - notes["balanced_acc"]) <= 1 / 50
            assert [c["config"] for c in record["candidates"]] == record["config"]["candidate_order"]
        readout = devreadout.readout_run(run_dir / "run.json")
        assert readout["method"] == method and readout["instance_id"] == instance.instance_id
        assert readout["readouts_fewshot"]["n_new"] + readout["readouts_fewshot"]["n_old"] == instance.private["n_vselect"]
        if not final["is_base_noop"]:
            label = devreadout.FEWSHOT_BASELINE_LIMITS
            assert readout["final_limits"] == (label if method != "grad_full" else f"{label} without size limits")
        child = devreadout.final_child(record, run_dir, instance, program)
        if method != "grad_full":  # the head refits change only the head
            assert all(np.array_equal(child[k], v) for k, v in instance.params.items() if k not in head)
        if method == "imprint":  # imprinting writes the new-class rows only
            for name in head:
                np.testing.assert_array_equal(child[name][OLD], instance.params[name][OLD])


def _predict_with_offset(z_old: np.ndarray, u_new: np.ndarray, beta: float) -> np.ndarray:
    return np.argmax(np.concatenate([z_old, u_new + beta], axis=1), axis=1)  # classes 0, 1 old, then 2, 3 new


def test_the_retain_offset_is_the_highest_interval_that_breaks_no_correct_replay_row() -> None:
    z_old = np.array([[1.0, 0.5], [2.5, 0.0], [0.0, 0.0], [3.0, 1.0], [0.0, 1.5], [0.2, 0.1], [2.0, 0.5]])
    u_new = np.array([[3.0, 1.0], [0.0, 2.0], [0.0, 1.0], [1.0, 0.5], [0.5, 0.0], [0.0, 0.0], [0.0, 0.0]])
    y = np.array([2, 3, 2, 0, 1, 0, 1])
    base_pred = np.array([0, 0, 0, 0, 1, 3, 0])  # replay rows 3 and 4 are right; 5 and 6 are wrong
    is_new_row = np.array([True, True, True, False, False, False, False])
    fit = imprint.retain_offset(z_old, u_new, y, base_pred, [0, 1], [2, 3], is_new_row, step=1.0)
    # thresholds -2, 0.2, 0.5 (rows 0, 5, 1), 1 (row 4, binding) and 2 (row 3): beta is the midpoint of [0.5, 1).
    assert fit["beta_max"] == 1.0 and fit["interval"] == [0.5, 1.0] and fit["beta"] == pytest.approx(0.75)
    assert (fit["n_replay_base_right"], fit["n_replay_binding"], fit["n_thresholds"]) == (2, 1, 5)
    pred = _predict_with_offset(z_old, u_new, fit["beta"])
    right = pred == y
    assert fit["acc_new"] == pytest.approx(2 / 3) and right[is_new_row].mean() == pytest.approx(2 / 3)
    assert fit["acc_old"] == pytest.approx(0.5) and right[~is_new_row].mean() == pytest.approx(0.5)
    kept = ~is_new_row & (base_pred == y)
    assert right[kept].all() and not (_predict_with_offset(z_old, u_new, 1.0 + 1e-9) == y)[4]
    # A base error on a replay row does not bound beta: row 5 (threshold 0.2) is lost at 0.75.
    assert not right[5]
    lowest = imprint.retain_offset(z_old - np.array([0.0, 3.0]), u_new, y, base_pred, [0, 1], [2, 3], is_new_row, 1.0)
    assert lowest["interval"][0] == -np.inf and lowest["beta"] == pytest.approx(lowest["beta_max"] - 1.0)
    with pytest.raises(ValueError, match="no replay row correctly"):
        imprint.retain_offset(z_old, u_new, y, np.full(len(y), 3), [0, 1], [2, 3], is_new_row, 1.0)


def test_imprint_retain_keeps_every_correct_replay_row_of_every_candidate(
        fewshot_store: list[dict[str, object]], tmp_path: Path) -> None:  # noqa: F811
    instance = _instance("cnn", 100)
    program = F.fewshot_program_limits(instance.task["limits"])
    run_dir = tmp_path / "imprint_retain"
    record = run_method(instance, "imprint_retain", run_dir, run_seed=0, allow_main=False, fewshot=True)
    assert record["status"] == "complete" and record["method"] == "imprint_retain"
    assert record["config"]["offset_rule"] == "retain" and record["limits"] == task_limits(program)
    assert record["config"]["candidate_order"] == imprint.imprint_config(0)["candidate_order"]
    evidence = load_fewshot_evidence(instance)
    kept = (np.arange(len(evidence.y)) >= evidence.n_rand) & (evidence.base_pred == evidence.y)
    assert kept.any()
    for candidate in record["candidates"]:
        notes = candidate["notes"]
        assert notes["realized_replay_base_right_broken"] == 0 and notes["beta"] < notes["beta_max"]
        assert notes["n_replay_base_right"] == int(kept.sum())
    for entry in record["oracle_log"]:
        child = devreadout.final_child({"final": {**entry, "dir": entry["patch_id"], "is_base_noop": False},
                                        "flags": record["flags"]}, run_dir, instance, program)
        assert np.array_equal(predict(instance.forward, child, evidence.x, 256)[kept], evidence.y[kept])
        for name in ("fc2.weight", "fc2.bias"):
            np.testing.assert_array_equal(child[name][OLD], instance.params[name][OLD])
    picked = devreadout.readout_fewshot_retention(run_dir / "run.json")
    assert picked["method"] == "imprint_retain" and picked["selection"]["n_checks"] == len(record["oracle_log"])


def _check(patch_id: str, query_index: int, score_new: int, score_old: int, n_changed: int = 10) -> dict[str, object]:
    return {"patch_id": patch_id, "query_index": query_index, "score_new": score_new, "score_old": score_old,
            "score": score_new + score_old, "n_changed": n_changed}


def test_the_equal_retention_pick_is_choose_final_on_the_checks_above_the_floor() -> None:
    checks = [_check("q0", 0, 900, -600), _check("q1", 1, 45, -20, 12), _check("q2", 2, 60, -30),
              _check("q3", 3, 25, -5, 9), _check("q4", 4, 25, -5, 9), _check("q5", 5, 26, -6, 8)]
    chosen, n_eligible = devreadout.retention_choice(checks)
    assert (chosen.patch_id, chosen.score, n_eligible) == ("q1", 25, 4)  # -20 is eligible, -30 is not
    chosen, _ = devreadout.retention_choice(checks, s_old_floor=-6)  # a tie on S_joint: fewer changed entries
    assert chosen.patch_id == "q5"
    chosen, _ = devreadout.retention_choice(checks, s_old_floor=-5)  # a tie on S_joint and size: the earlier check
    assert chosen.patch_id == "q3"
    chosen, n_eligible = devreadout.retention_choice([_check("q0", 0, 10, -10), _check("q1", 1, 900, -600)])
    assert chosen.patch_id is None and n_eligible == 1  # an eligible check with S_joint <= 0 loses to the base
    chosen, n_eligible = devreadout.retention_choice([_check("q0", 0, 900, -600)])
    assert chosen.patch_id is None and n_eligible == 0
    with pytest.raises(devreadout.DevReadoutError, match="S_new \\+ S_old"):
        devreadout.retention_choice([{**_check("q0", 0, 1, 1), "score": 3}])


def test_the_equal_retention_pick_of_a_joint_run_without_a_valid_check_is_the_base_with_its_refusals() -> None:
    base_final = {"patch_id": None, "score": 0, "is_base_noop": True}
    cap = "files: the referenced .npy files total 89089360 bytes, above the limit of 67108864 bytes"
    refused = [{"patch_id": f"q{i}", "invalid": cap, "invalid_stage": "executor"} for i in range(7)]
    refused.append({"patch_id": "q7", "invalid": "non-finite values", "invalid_stage": "generation"})
    run = {"oracle_log": [], "joint": {"oracle": "JointPatchCheckOracle"}, "final": base_final, "candidates": refused}
    picked = devreadout.retention_final(run)
    assert picked["final"]["is_base_noop"] and picked["selection"]["n_checks"] == 0
    assert picked["selection"]["n_eligible"] == 0 and picked["selection"]["patch_id"] is None
    assert picked["selection"]["no_valid_check"] == {
        "n_candidates": 8, "n_refused": 8,
        "refusals": [{"stage": "executor", "reason": cap, "count": 7},
                     {"stage": "generation", "reason": "non-finite values", "count": 1}]}
    with pytest.raises(devreadout.DevReadoutError, match="joint oracle log"):  # not a joint run
        devreadout.retention_final({**run, "joint": None})
    with pytest.raises(devreadout.DevReadoutError, match="were not refused"):  # an empty log beside a valid candidate
        devreadout.retention_final({**run, "candidates": [*refused, {"patch_id": "q8"}]})
    with pytest.raises(devreadout.DevReadoutError, match="list its refused candidates"):
        devreadout.retention_final({**run, "candidates": []})
    with pytest.raises(devreadout.DevReadoutError, match="not choose_final"):  # a final that the empty log cannot give
        devreadout.retention_final({**run, "final": {"patch_id": "q0", "score": 3}})
    checked = devreadout.retention_final({**run, "oracle_log": [_check("q0", 0, 20, -30)]})  # one check, below 0
    assert "no_valid_check" not in checked["selection"]  # only a run without any check carries the refusals


def test_the_equal_retention_readout_rebuilds_the_picked_check(fewshot_store: list[dict[str, object]],  # noqa: F811
                                                               tmp_path: Path) -> None:
    instance = _instance("cnn", 100)
    program = F.fewshot_program_limits(instance.task["limits"])
    run_dir = tmp_path / "grad_last"
    record = run_method(instance, "grad_last", run_dir, run_seed=0, allow_main=False, fewshot=True)
    run_path = run_dir / "run.json"
    log = record["oracle_log"]
    primary = devreadout.readout_run(run_path)
    everything = devreadout.readout_fewshot_retention(run_path, s_old_floor=-10**9)
    assert everything["readouts_fewshot"] == primary["readouts_fewshot"] and everything["vcheck"] == primary["vcheck"]
    assert everything["selection"]["n_eligible"] == everything["selection"]["n_checks"] == len(log)
    assert everything["selection"]["patch_id"] == record["final"]["patch_id"]
    for floor in sorted({int(entry["score_old"]) for entry in log}):
        picked = devreadout.readout_fewshot_retention(run_path, s_old_floor=floor)
        eligible = [entry for entry in log if int(entry["score_old"]) >= floor]
        assert picked["vcheck"]["score"] == max([0, *(int(entry["score"]) for entry in eligible)])
        assert picked["selection"]["s_old_floor"] == floor and picked["selection"]["n_eligible"] == len(eligible)
        if picked["is_base_noop"]:
            assert picked["selection"]["patch_id"] is None and picked["readouts_fewshot"]["all_points"] == 0.0
            continue
        (entry,) = [item for item in log if item["patch_id"] == picked["selection"]["patch_id"]]
        assert picked["vcheck"]["score_old"] == entry["score_old"] >= floor
        child = devreadout.final_child({"final": {**entry, "dir": entry["patch_id"], "is_base_noop": False},
                                        "flags": record["flags"]}, run_dir, instance, program)
        assert picked["readouts_fewshot"] == F.fewshot_readout(instance, child)
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    run = json.loads(run_path.read_text())
    (tampered / "run.json").write_text(json.dumps({**run, "final": {**run["final"], "score": run["final"]["score"] + 1}}))
    with pytest.raises(devreadout.DevReadoutError, match="not choose_final"):
        devreadout.readout_fewshot_retention(tampered / "run.json")


def test_imprint_runs_on_another_architecture(fewshot_store: list[dict[str, object]], tmp_path: Path) -> None:  # noqa: F811
    instance = _instance("cnn4", 100)
    record = run_method(instance, "imprint", tmp_path / "imprint", run_seed=0, allow_main=False, fewshot=True)
    assert record["status"] == "complete" and record["fewshot"]["head_tensors"] == ["fc1.weight", "fc1.bias"]
    assert record["generation"]["trainable"] == ["fc1.weight", "fc1.bias"]


def test_fewshot_misuse_is_refused_before_anything_is_written(fewshot_store: list[dict[str, object]],  # noqa: F811
                                                              tmp_path: Path) -> None:
    instance = _instance("cnn", 101)
    target = tmp_path / "never"
    with pytest.raises(ValueError, match="does not run on fewshot instances"):
        run_method(instance, "prior_shift", target, run_seed=0, allow_main=False, fewshot=True)
    with pytest.raises(ValueError, match="fewshot instances only"):
        run_method(instance, "imprint", target, run_seed=0, allow_main=False)
    with pytest.raises(ValueError, match="pass fewshot without joint"):
        run_grad(instance, "grad_last", target, joint=True, fewshot=True)
    task_path = Path(instance.instance_dir) / "task.json"
    task = json.loads(task_path.read_text())
    task_path.write_text(json.dumps({**task, "statement": task["statement"] + " Changed."}))
    with pytest.raises(F.StaleFewshotInstance):
        run_method(instance, "grad_last", target, run_seed=0, allow_main=False, fewshot=True)
    assert not target.exists()


def test_the_script_refuses_fewshot_with_joint_or_a_rule_method(fewshot_store: list[dict[str, object]],  # noqa: F811
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("run_fit_baselines", REPO / "scripts" / "run_fit_baselines.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.main(["--fewshot", "--joint", "--threads", "1"]) == 2
    assert script.main(["--fewshot", "--methods", "prior_shift", "--threads", "1"]) == 2
    assert script.main(["--methods", "imprint", "--threads", "1"]) == 2
    assert script.main(["--methods", "imprint_retain", "--threads", "1"]) == 2
    assert "refused" in capsys.readouterr().err
