"""weightcode.evalkit.fscil (rule fscil-v1, PREREG E9) on the toy store of ``fscil_toy``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fscil_toy import CIFAR100_FINE_NAMES, toy_fscil  # noqa: F401

from weightcode.common.identity import save_params
from weightcode.evalkit import fscil as FS
from weightcode.evalkit.instances import QUERY_BUDGET
from weightcode.evalkit.patch_check import predict


def test_session_arithmetic() -> None:
    assert FS.session_classes(1) == list(range(60))
    assert FS.session_classes(2) == [60, 61, 62, 63, 64]
    assert FS.session_classes(9) == [95, 96, 97, 98, 99]
    assert [FS.n_seen(t) for t in (1, 2, 9)] == [60, 65, 100]
    assert FS.prev_classes(2) == [] and FS.prev_classes(4) == list(range(60, 70))
    assert [FS.session_of_class(c) for c in (0, 59, 60, 64, 65, 99)] == [1, 1, 2, 2, 3, 9]
    for bad in (0, 10, 2.5):
        with pytest.raises(FS.FscilError):
            FS.session_classes(bad)


def test_masked_forward_equals_argmax_over_seen_classes() -> None:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((50, 100)).astype(np.float32)
    logits[:, 90] += 10.0  # an unseen class that would win without the mask
    masked = FS.masked_forward(lambda params, x: logits[x[:, 0]], 3)  # x carries the row index
    pred = predict(masked, {}, np.arange(50, dtype=np.uint8)[:, None], 16)
    reference = np.where(np.arange(100) < FS.n_seen(3), logits, -np.inf).argmax(axis=1)
    assert np.array_equal(pred, reference) and pred.max() < FS.n_seen(3)


def test_joint_retain_floor_e7_values_and_clamp() -> None:
    values = [FS.joint_retain_floor(3002, 3002, -4.93, d) for d in (0.0, 0.1, 0.2, 1.0)]
    assert values == [-151, -111, -94, -24]  # PREREG E7 amendment of 14:46
    assert FS.joint_retain_floor(3002, 3002, -1.0, 0.0) == -20  # the clamp stays at R = -1.0
    assert FS.joint_retain_floor(1800, 1800, -1.0, 50.0) == 0  # never above 0
    with pytest.raises(FS.FscilError):
        FS.joint_retain_floor(0, 10, -1.0, 0.1)


def test_limits_exempt_head_and_1d_tensors() -> None:
    spec = {"layers": [{"op": "resnet20_encoder"}, {"op": FS.HEAD_OP, "weight": FS.HEAD_TENSOR}]}
    limits = FS.fscil_limits(spec)
    assert not limits.rel_frob_applies("fc.weight", 2)
    assert limits.rel_frob_applies("layer1.0.conv1.weight", 4)
    assert not limits.rel_frob_applies("layer1.0.bn1.running_var", 1)
    program = FS.fscil_program_limits({"rel_frob_exempt": ["fc.weight"]})
    assert program.max_changed_frac == FS.FSCIL_MAX_CHANGED_FRAC and program.max_edit_frac is None
    with pytest.raises(FS.FscilError):
        FS.head_tensors({"layers": [{"op": "linear", "weight": "fc.weight", "bias": "fc.bias"}]})


def test_readout_summaries_pooled_and_balanced() -> None:
    y = np.array([0, 0, 0, 1, 60, 61, 65])
    part = np.array(["old", "old", "old", "old", "prev", "prev", "new"])
    base = np.array([0, 0, 1, 1, 0, 61, 0])
    child = np.array([0, 0, 0, 0, 60, 61, 65])
    summary = FS.readout_summary(base, child, y, part, 3)
    assert summary["n_old"] == 4 and summary["acc_old_base"] == 0.75 and summary["acc_old_child"] == 0.75
    assert summary["bal_old_base"] == pytest.approx((2 / 3 + 1) / 2)
    assert summary["fixed_old"] == 1 and summary["broken_old"] == 1
    assert summary["acc_new_child"] == 1.0 and summary["fixed_new"] == 1
    assert summary["per_session"]["2"]["bal_child"] == 1.0 and summary["child_pred_in_new"] == 1
    absolute = FS.accuracy_summary(child, y, part, 3)
    assert absolute["acc_seen"] == pytest.approx(6 / 7) and absolute["n_prev"] == 2
    empty = FS.readout_summary(base[:4], child[:4], y[:4], part[:4], 2)
    assert empty["n_prev"] == 0 and empty["acc_prev_base"] is None and empty["delta_bal_prev"] is None


def test_real_class_names_pass_the_leak_scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for variable in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED", "WEIGHTCODE_RUNS"):
        (tmp_path / variable).mkdir()
        monkeypatch.setenv(variable, str(tmp_path / variable))
    assert FS.fscil_scan("\n".join(CIFAR100_FINE_NAMES)) == []
    assert FS.fscil_scan("a CIFAR-100 few-shot model") == ["cifar", "cifar-100"]
    assert FS.fscil_scan(f"see {tmp_path}/WEIGHTCODE_PRIVATE/x") != []


def _child_on_new_classes(instance, scale: float = 3.0) -> dict[str, np.ndarray]:
    """A child whose head rows of the session's classes point at the mean feature of their shots (imprinting)."""
    namespace: dict = {}
    exec(compile((instance.instance_dir / "forward.py").read_text(), "forward.py", "exec"), namespace)
    features = namespace["features"](instance.params, instance.evidence["x_new"])
    child = {key: value.copy() for key, value in instance.params.items()}
    for label in FS.session_classes(instance.private["session"]):
        mean = features[instance.evidence["y_new"] == label].mean(axis=0)
        child["fc.weight"][label] = (scale * mean / np.linalg.norm(mean)).astype(np.float32)
    return child


def test_session2_instance_build_check_oracle_readout(toy_fscil) -> None:  # noqa: F811
    row = FS.build_fscil_instance(2, lineage="T")
    assert row["status"] == "created" and row["parent_kind"] == "base" and row["parent_instance"] is None
    instance = FS.load_fscil_instance(row["instance_id"])
    ev = instance.evidence
    assert tuple(sorted(ev)) == tuple(sorted(FS.EVIDENCE_KEYS))
    assert ev["x_new"].shape == (25, 3, 8, 8) and ev["x_old"].shape == (64, 3, 8, 8)
    assert ev["x_prev"].shape == (0, 3, 8, 8) and ev["x_prev"].dtype == np.uint8 and ev["y_prev"].dtype == np.int64
    assert ev["y_new"].tolist() == [c for c in range(60, 65) for _ in range(5)]
    assert ev["y_old"].max() < 60 and max(ev["pred_new"].max(), ev["pred_old"].max()) < FS.n_seen(2)
    task = FS.check_fscil_current(instance.instance_dir)
    assert task["session"] == 2 and task["n_seen"] == 65 and task["new_classes"] == list(range(60, 65))
    assert "This is session 2:" in task["statement"] and task["prediction_rule"]["n_seen"] == 65
    spec = json.loads((instance.instance_dir / "model_spec.json").read_text())
    assert spec["prediction"]["n_seen"] == 65
    store_spec = json.loads((FS.fewshot_models_dir(FS.DATASET) / toy_fscil.base_model_id / "model_spec.json")
                            .read_text())
    assert store_spec["model_id"] == toy_fscil.base_model_id  # the store's spec carries it, as the real ones do
    assert "model_id" not in spec and toy_fscil.base_model_id not in (instance.instance_dir / "model_spec.json"
                                                                     ).read_text()
    assert {k: v for k, v in spec.items() if k != "prediction"} == {k: v for k, v in store_spec.items()
                                                                   if k != "model_id"}
    assert FS.build_fscil_instance(2, lineage="F")["status"] == "reused"
    changes = [json.loads(line) for line in (FS.fscil_root() / "changes.jsonl").read_text().splitlines()]
    assert [(c["change"], c["lineage"]) for c in changes] == [("created", "T"), ("used", "F")]

    oracle = FS.make_fscil_oracle(instance)
    assert oracle.queries_remaining == QUERY_BUDGET
    child = _child_on_new_classes(instance)
    readout = FS.fscil_readout(instance, child)
    assert (readout["n_new"], readout["n_prev"], readout["n_old"], readout["n_seen"]) == (10, 0, 120, 130)
    assert readout["n_seen_classes"] == 65
    same = FS.fscil_readout(instance, instance.params)
    assert same["delta_bal_seen"] == 0.0 and same["fixed_seen"] == same["broken_seen"] == 0
    absolute = FS.fscil_accuracy(instance.params, 2)
    assert absolute["bal_seen"] == pytest.approx(same["bal_seen_child"])
    assert FS.old_part_sizes(2) == (120, 120) and FS.old_part_sizes(4) == (140, 140)


def test_session3_on_a_lineage_output(toy_fscil, tmp_path: Path) -> None:  # noqa: F811
    first = FS.build_fscil_instance(2, lineage="T")
    instance = FS.load_fscil_instance(first["instance_id"])
    child = _child_on_new_classes(instance)
    path = tmp_path / "child.npz"
    save_params(path, child)
    with pytest.raises(FS.FscilError):
        FS.build_fscil_instance(3, lineage="T", parent_params=path)  # no parent instance
    with pytest.raises(FS.FscilError):
        FS.build_fscil_instance(2, lineage="T", parent_params=path, parent_instance=first["instance_id"])
    bad = dict(child)
    bad["fc.weight"] = bad["fc.weight"][:, :32]
    save_params(tmp_path / "bad.npz", bad)
    with pytest.raises(FS.FscilError):
        FS.build_fscil_instance(3, lineage="T", parent_params=tmp_path / "bad.npz",
                                parent_instance=first["instance_id"])
    row = FS.build_fscil_instance(3, lineage="T", parent_params=path, parent_instance=first["instance_id"], run_id="r1")
    assert row["status"] == "created" and row["parent_kind"] == "lineage"
    third = FS.load_fscil_instance(row["instance_id"])
    ev = third.evidence
    assert ev["x_new"].shape[0] == 25 and ev["x_prev"].shape[0] == 25 and ev["session_prev"].tolist() == [2] * 25
    assert ev["y_prev"].tolist() == instance.evidence["y_new"].tolist()
    assert np.array_equal(ev["x_old"], instance.evidence["x_old"])  # one replay draw for every session
    assert max(ev["pred_new"].max(), ev["pred_prev"].max()) < FS.n_seen(3)
    oracle = FS.make_fscil_oracle(third)
    assert oracle.queries_remaining == QUERY_BUDGET
    assert third.private["n_vcheck_new"] == 10 and third.private["n_vcheck_old"] == 130
    assert FS.instance_chain(row["instance_id"]) == [row["instance_id"], first["instance_id"]]
    again = FS.build_fscil_instance(3, lineage="C", parent_params=path, parent_instance=first["instance_id"])
    assert again["status"] == "reused" and again["instance_id"] == row["instance_id"]
    readout = FS.fscil_readout(third, third.params)
    assert (readout["n_new"], readout["n_prev"], readout["n_old"]) == (10, 10, 120)
    assert sorted(readout["per_session"]) == ["1", "2", "3"]


def test_stale_task_is_refused(toy_fscil) -> None:  # noqa: F811
    row = FS.build_fscil_instance(2, lineage="T")
    directory = FS.fscil_instance_dir(row["instance_id"])
    task = json.loads((directory / "task.json").read_text())
    task["statement"] += " "
    (directory / "task.json").write_text(json.dumps(task))
    with pytest.raises(FS.StaleFscilInstance):
        FS.check_fscil_current(directory)


def test_task_wording_names_the_earlier_sessions() -> None:
    limits = FS.fscil_limits({"layers": [{"op": FS.HEAD_OP, "weight": FS.HEAD_TENSOR}]})
    third = FS.build_fscil_task(CIFAR100_FINE_NAMES, 3, limits)["statement"]
    ninth = FS.build_fscil_task(CIFAR100_FINE_NAMES, 9, limits)
    assert "were added in session 2;" in third and "65 (rabbit)" in third
    assert "were added in sessions 2 to 8;" in ninth["statement"] and ninth["evidence"]["prev"]["count"] == 175
    assert ninth["n_seen"] == 100 and ninth["schema"] == "weightcode.task.incremental.v1"
