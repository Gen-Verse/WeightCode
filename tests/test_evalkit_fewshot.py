"""Rule fewshot-v1 (PREREG E2): build, layout, staleness, check split, oracle and readout on a toy CIFAR-10 store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import test_models_conditions
import torch
from test_models_conditions import _make_store

from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.data import store
from weightcode.evalkit import fewshot as F
from weightcode.evalkit.instances import leak_tokens
from weightcode.evalkit.patch_check import predict
from weightcode.models.arch import build_model, export_params
from weightcode.models.export import FEWSHOT_INDEX_KEYS, fewshot_models_dir, read_fewshot_index, write_fewshot_index
from weightcode.models.spec import build_model_spec

SIZES = {"train": 200, "V_select": 120, "V_evidence": 120, "V_check": 100}
NEW = [5, 6, 7, 8, 9]


def _student(model_id: str, arch: str, seed: int) -> dict[str, object]:
    """A fewshot student as train_fewshot leaves it: random weights, new-class head biases at -50 (never predicted)."""
    meta = store.read_dataset_json("cifar10")
    torch.manual_seed(seed)
    model = build_model(arch, meta["image_shape"], 10, meta["norm"]["mean"], meta["norm"]["std"])
    params = export_params(model)
    head = {"cnn": "fc2", "cnn4": "fc1"}[arch]
    params[f"{head}.bias"][NEW] = -50.0
    folder = fewshot_models_dir("cifar10") / model_id
    folder.mkdir(parents=True)
    digest = save_params(folder / "model.npz", params)
    spec = build_model_spec(model_id, arch, meta["image_shape"], meta["norm"]["mean"], meta["norm"]["std"],
                            meta["class_names"], params)
    store.write_json(folder / "model_spec.json", spec)
    store.write_json(folder / "private.json", {"model_id": model_id, "role": "dev", "seed": seed, "arch": arch,
                                               "train_classes": [0, 1, 2, 3, 4], "family": "fewshot"})
    values = [model_id, seed, arch, "dev", digest, 0.4, [0, 1, 2, 3, 4], 0.8]
    return dict(zip(FEWSHOT_INDEX_KEYS, values))


@pytest.fixture
def fewshot_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    monkeypatch.setattr(test_models_conditions, "SIZES", SIZES)
    _make_store(tmp_path / "store", monkeypatch)
    rows = [_student("ma0000001", "cnn", 100), _student("ma0000002", "cnn", 101), _student("ma0000003", "cnn4", 100)]
    write_fewshot_index("cifar10", rows)
    return rows


def test_build_writes_the_layout_the_evidence_and_the_records(fewshot_store: list[dict[str, object]]) -> None:
    outcomes = F.build_fewshot_instances("cifar10")
    assert [o["status"] for o in outcomes] == ["created"] * 3
    assert F.active_fewshot_ids("cifar10") == [o["instance_id"] for o in outcomes]
    data = {s: store.load_split("cifar10", s) for s in ("train", "V_evidence", "V_check", "V_select")}
    replay = F.draw_replay_rows(data["train"]["y"], [0, 1, 2, 3, 4])
    tokens = leak_tokens("cifar10")
    for outcome in outcomes:
        root = F.fewshot_instance_dir(outcome["instance_id"])
        assert F.INSTANCE_ID_RE.fullmatch(root.name)
        assert sorted(p.name for p in root.iterdir()) == sorted([*F.EDITOR_VISIBLE_FILES, "private.json"])
        instance = load_instance(root)
        ev, private, task = instance.evidence, instance.private, instance.task
        assert set(ev) == set(F.EVIDENCE_KEYS)
        assert ev["x_new"].dtype == np.uint8 and ev["x_new"].shape == (50, 3, 32, 32)
        assert ev["y_new"].tolist() == [c for c in NEW for _ in range(10)]
        rows_new = np.asarray(private["evidence_row_ids_new"])
        np.testing.assert_array_equal(ev["x_new"], data["V_evidence"]["x"][rows_new])
        np.testing.assert_array_equal(ev["y_new"], data["V_evidence"]["y"][rows_new])
        assert len(set(rows_new.tolist())) == 50
        np.testing.assert_array_equal(np.asarray(private["replay_train_rows"]), replay)
        np.testing.assert_array_equal(ev["x_old"], data["train"]["x"][replay])
        assert set(ev["y_old"].tolist()) <= {0, 1, 2, 3, 4} and len(ev["y_old"]) == 64
        np.testing.assert_array_equal(ev["pred_new"], predict(instance.forward, instance.params, ev["x_new"], 64))
        np.testing.assert_array_equal(ev["pred_old"], predict(instance.forward, instance.params, ev["x_old"], 64))
        assert private["n_base_correct_evidence_new"] == 0 and private["n_base_pred_in_new_vcheck"] == 0
        ids = [*ev["split_ids_new"].tolist(), *ev["split_ids_old"].tolist()]
        assert len(set(ids)) == 114 and all(len(i) == 13 and i.startswith("e") for i in ids)
        assert private["evidence_seed"] == F.fewshot_seed(instance.params_sha256, "cifar10")
        assert private["n_vcheck"] == private["n_vcheck_new"] + private["n_vcheck_old"] == SIZES["V_check"]
        assert private["n_vcheck_new"] == int(np.isin(data["V_check"]["y"], NEW).sum())
        assert private["n_vselect"] == private["n_vselect_new"] + private["n_vselect_old"] == SIZES["V_select"]
        head = {"cnn": ["fc2.weight", "fc2.bias"], "cnn4": ["fc1.weight", "fc1.bias"]}[private["arch"]]
        assert private["head_tensors"] == head == task["limits"]["rel_frob_exempt"]
        assert task["limit_definitions"]["max_rel_frob"].endswith("exempt whatever their rank")
        assert task["schema"] == F.TASK_SCHEMA and task["new_classes"] == NEW and len(task["classes"]) == 10
        assert task["evidence"]["new"]["count"] == 50 and task["evidence"]["old"]["count"] == 64
        assert F.check_fewshot_current(root) == task
        for name in ("task.json", "model_spec.json"):
            text = (root / name).read_text().lower()
            assert not [t for t in tokens if t in text], name
    first, second = (load_instance(F.fewshot_instance_dir(o["instance_id"])) for o in outcomes[:2])
    assert first.private["evidence_row_ids_new"] != second.private["evidence_row_ids_new"]
    assert first.private["replay_train_rows"] == second.private["replay_train_rows"]


def test_a_rebuild_reuses_and_a_changed_input_is_refused(fewshot_store: list[dict[str, object]]) -> None:
    built = F.build_fewshot_instances("cifar10", arch="cnn")
    assert len(built) == 2
    again = F.build_fewshot_instances("cifar10", arch="cnn")
    assert [(o["instance_id"], o["status"]) for o in again] == [(o["instance_id"], "reused") for o in built]
    with pytest.raises(F.FewshotError, match="no fewshot students"):
        F.build_fewshot_instances("cifar10", seeds=[7])
    path = fewshot_models_dir("cifar10") / "ma0000001" / "model.npz"
    params = load_params(path)
    params["fc1.bias"] = params["fc1.bias"] + np.float32(1.0)
    save_params(path, params)
    index = read_fewshot_index("cifar10")
    index[0] = dict(index[0], params_sha256=params_sha256(params))
    write_fewshot_index("cifar10", index)
    with pytest.raises(F.FewshotError, match="differs from a rebuild"):
        F.build_fewshot_instances("cifar10", arch="cnn", seeds=[100])


def test_check_fewshot_current_refuses_changed_files(fewshot_store: list[dict[str, object]]) -> None:
    (outcome,) = F.build_fewshot_instances("cifar10", arch="cnn4")
    root = F.fewshot_instance_dir(outcome["instance_id"])
    task_text = (root / "task.json").read_text()
    (root / "task.json").write_text(task_text.replace("never saw", "rarely saw"))
    with pytest.raises(F.StaleFewshotInstance, match="task_sha256"):
        F.check_fewshot_current(root)
    (root / "task.json").write_text(task_text)
    evidence = dict(np.load(root / "evidence.npz"))
    evidence["y_old"] = evidence["y_old"].copy()
    evidence["y_old"][0] = (evidence["y_old"][0] + 1) % 5
    np.savez(root / "evidence.npz", **evidence)
    with pytest.raises(F.StaleFewshotInstance, match="evidence.npz"):
        F.check_fewshot_current(root)
    with pytest.raises(F.FewshotError, match="not a fewshot instance id"):
        F.fewshot_instance_dir("i0123abcd")


def test_check_parts_split_by_class_in_row_order() -> None:
    y = np.array([5, 0, 9, 1, 5, 4], dtype=np.int64)
    x = np.arange(6, dtype=np.uint8).reshape(6, 1, 1, 1)
    parts = F.fewshot_check_parts({"x": x, "y": y, "idx": np.arange(6)}, NEW)
    assert set(parts) == set(F.CHECK_PART_KEYS)
    assert parts["y_new"].tolist() == [5, 9, 5] and parts["x_new"].ravel().tolist() == [0, 2, 4]
    assert parts["y_old"].tolist() == [0, 1, 4] and parts["x_old"].ravel().tolist() == [1, 3, 5]
    with pytest.raises(F.FewshotError):
        F.fewshot_check_parts({"x": x, "y": y}, [])
    with pytest.raises(F.FewshotError, match="rows in the new classes"):
        F.fewshot_check_parts({"x": x[[0, 2]], "y": y[[0, 2]]}, NEW)


def test_oracle_and_readout_score_the_two_parts(fewshot_store: list[dict[str, object]]) -> None:
    (outcome,) = F.build_fewshot_instances("cifar10", arch="cnn", seeds=[100])
    instance = load_instance(F.fewshot_instance_dir(outcome["instance_id"]))
    oracle = F.make_fewshot_oracle(instance)
    assert oracle.queries_remaining == 8
    noop = oracle.query(instance.params, "q0")
    assert noop["score"] == noop["score_new"] == noop["score_old"] == 0
    child = {key: value.copy() for key, value in instance.params.items()}
    child["fc2.bias"][5] = 1e4  # every image becomes class 5
    entry = oracle.query(child, "q1")
    v_check = store.load_split("cifar10", "V_check")
    base_pred = predict(instance.forward, instance.params, v_check["x"], 64)
    old = ~np.isin(v_check["y"], NEW)
    assert entry["fixed_new"] == int(np.count_nonzero(v_check["y"] == 5))
    assert entry["broken_old"] == int(np.count_nonzero(old & (base_pred == v_check["y"])))
    assert entry["score"] == entry["score_new"] + entry["score_old"]

    same = F.fewshot_readout(instance, instance.params)
    assert same["delta_all"] == same["delta_new"] == same["delta_old"] == 0.0 and same["joint_points"] == 0.0
    assert same["acc_new_base"] == 0.0
    readout = F.fewshot_readout(instance, child)
    v_select = store.load_split("cifar10", "V_select")
    in_new = np.isin(v_select["y"], NEW)
    assert (readout["n_all"], readout["n_new"], readout["n_old"]) == (120, int(in_new.sum()), int((~in_new).sum()))
    assert readout["acc_new_child"] == pytest.approx(np.mean(v_select["y"][in_new] == 5))
    assert readout["acc_old_child"] == 0.0
    assert readout["all_points"] == pytest.approx(100 * readout["delta_all"])
    assert readout["joint_points"] == pytest.approx(100 * (readout["delta_new"] + readout["delta_old"]))
    assert readout["per_class"]["5"]["acc_child"] == 1.0
    assert readout["worst_old_class_delta"] == min(readout["per_class"][str(c)]["delta"] for c in range(5))
    assert json.loads(json.dumps(readout)) == readout


def test_devreadout_rebuilds_a_fewshot_final_under_the_head_exemption(
    fewshot_store: list[dict[str, object]], tmp_path: Path
) -> None:
    from weightcode.baselines.instance_io import write_delta_patch
    from weightcode.evalkit import devreadout
    from weightcode.patch import Limits, PatchError, apply_patch_file, task_limits

    (outcome,) = F.build_fewshot_instances("cifar10", arch="cnn", seeds=[101])
    instance = load_instance(F.fewshot_instance_dir(outcome["instance_id"]))
    child = {key: value.copy() for key, value in instance.params.items()}
    weight = child["fc2.weight"]
    weight[NEW] = 3.0 * np.abs(weight).max()  # a head change far beyond rel-Frobenius 0.5
    child["fc2.bias"][NEW] = 0.0
    run_dir = tmp_path / "runs" / instance.instance_id / "imprint" / "e2" / "0"
    patch = write_delta_patch(run_dir / "q0", instance.params_sha256, instance.params, child, "toy", "toy")
    with pytest.raises(PatchError):
        apply_patch_file(instance.params, patch, Limits(), instance.params_sha256)
    final = {"is_base_noop": False, "dir": "q0", "score": 3, "fixed": 5, "broken": 2, "score_new": 4, "score_old": -1,
             "edit_size": 645, "child_sha256": params_sha256(child)}
    run_path = run_dir / "run.json"
    program = task_limits(F.fewshot_program_limits(instance.task["limits"]))
    assert program["max_changed_frac"] == F.FEWSHOT_MAX_CHANGED_FRAC and program["max_edit_frac"] is None
    assert program["rel_frob_exempt"] == ["fc2.weight", "fc2.bias"]

    def write_run(limits: object) -> None:
        run_path.write_text(json.dumps({"instance": {"instance_id": instance.instance_id}, "method": "imprint",
                                        "flags": {"edit_limits_enforced": True}, "limits": limits, "final": final}))

    write_run(program)
    record = devreadout.readout_run(run_path)
    assert record["rule"] == F.FEWSHOT_RULE and record["final_limits"] == devreadout.FEWSHOT_BASELINE_LIMITS
    assert record["readouts_fewshot"] == F.fewshot_readout(instance, child)
    assert record["vcheck"] == {"score": 3, "fixed": 5, "broken": 2, "score_new": 4, "score_old": -1}
    for other in (instance.task["limits"], None, {"max_rel_frob": 0.5, "unknown": 1}):
        write_run(other)  # a baseline run that applied other limits, or recorded none, is refused
        with pytest.raises(devreadout.DevReadoutError, match="E2 program limits"):
            devreadout.readout_run(run_path)
    final["child_sha256"] = instance.params_sha256
    write_run(program)
    with pytest.raises(devreadout.DevReadoutError, match="rebuilt child"):
        devreadout.readout_run(run_path)


def test_module_import_does_not_need_torch() -> None:
    """CODE_AUDIT section 4: ``weightcode.models.export`` (which imports torch) is imported inside the two student-store
    functions only, so numpy-only use of the module loads without torch. A fresh interpreter blocks torch and imports
    the module from the same source tree as this test."""
    import subprocess
    import sys

    import weightcode

    src = str(Path(weightcode.__file__).resolve().parents[1])
    code = ("import sys; sys.modules['torch'] = None; import weightcode.evalkit.fewshot as F; "
            "assert 'weightcode.models.export' not in sys.modules; print(F.FEWSHOT_RULE)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": src})
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == F.FEWSHOT_RULE
