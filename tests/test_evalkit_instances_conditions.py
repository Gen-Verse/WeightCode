"""Named deployment conditions (the CIFAR-10 corruption layout) on a synthetic store: resolution, the refusals of a
condition that is not the clean data with only x changed, building next to the deploy instances, reuse,
check_task_current, the leak scan and the dev readout."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from test_evalkit_instances_store import (  # noqa: F401
    CONDITION,
    ToyStore,
    add_condition,
    build_condition,
    build_v1,
    condition_dir,
    condition_test_path,
    fixing_child,
    load_npz,
    toy_store,
    write_condition_meta,
)

import weightcode.evalkit.instances as instances
from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import params_sha256
from weightcode.evalkit import devreadout
from weightcode.evalkit.evidence import EVIDENCE_RULE_V1, K_FAIL, evidence_seed_v1
from weightcode.evalkit.instances import (
    CONDITION_LEAK_WORDS,
    StaleInstanceError,
    active_instance_ids,
    check_task_current,
    condition_leak_tokens,
    npz_text,
    scan_for_tokens,
    supersede_instances,
)
from weightcode.evalkit.metrics import sealed_readout
from weightcode.evalkit.patch_check import predict
from weightcode.evalkit.targets import ConditionMismatch, deployment_splits, store_condition


@pytest.fixture
def condition_store(toy_store: ToyStore) -> ToyStore:  # noqa: F811
    """The default toy store plus the condition toyblur_s2."""
    add_condition(toy_store)
    return toy_store


def _meta(store: ToyStore) -> dict[str, Any]:
    return json.loads((condition_dir(store) / "dataset.json").read_text())


def _edit_meta(store: ToyStore, change: Callable[[dict[str, Any]], object]) -> None:
    meta = _meta(store)
    change(meta)
    write_condition_meta(store, meta)


def _rewrite_split(
    store: ToyStore, split: str, change: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]
) -> None:
    """Change one condition split file and record its new content hash, so only the alignment check can object."""
    path = condition_dir(store) / "splits" / f"{split}.npz"
    arrays = change(load_npz(path))
    np.savez(path, **arrays)
    _edit_meta(store, lambda meta: meta["splits"][split].update(sha256=params_sha256(arrays)))


def test_a_named_condition_resolves_to_its_own_files(condition_store: ToyStore) -> None:
    meta = _meta(condition_store)
    splits = deployment_splits(condition_store.dataset, CONDITION)
    folder = condition_dir(condition_store) / "splits"
    assert (splits.v_evidence, splits.v_check, splits.v_select) == (
        folder / "V_evidence.npz", folder / "V_check.npz", folder / "V_select.npz"
    )
    assert splits.test == condition_test_path(condition_store)
    assert (splits.v_evidence_sha256, splits.v_check_sha256, splits.v_select_sha256, splits.test_sha256) == (
        meta["splits"]["V_evidence"]["sha256"], meta["splits"]["V_check"]["sha256"],
        meta["splits"]["V_select"]["sha256"], meta["test"]["sha256"],
    )
    deploy = deployment_splits(condition_store.dataset, "deploy")
    assert deploy.v_check == condition_store.private / condition_store.dataset / "splits" / "V_check.npz"
    assert deploy.test == condition_store.sealed / condition_store.dataset / "test.npz"
    assert store_condition("deploy") is None and store_condition(CONDITION) == CONDITION
    with pytest.raises(LookupError, match=r"'toyfog_s1' is not in the data store; known: \['deploy', 'toyblur_s2'\]"):
        deployment_splits(condition_store.dataset, "toyfog_s1")


MISMATCHES: dict[str, tuple[Callable[[ToyStore], None], str]] = {
    "class_names": (lambda s: _edit_meta(s, lambda m: m.update(class_names=m["class_names"][::-1])),
                    "class_names differs from the clean dataset.json"),
    "image_shape": (lambda s: _edit_meta(s, lambda m: m.update(image_shape=[1, 2, 8])),
                    "image_shape differs from the clean dataset.json"),
    "forward_sha256": (lambda s: _edit_meta(s, lambda m: m.update(forward_sha256="0" * 64)),
                       "forward_sha256 differs from the clean dataset.json"),
    "y_order": (lambda s: _rewrite_split(s, "V_check", lambda a: {**a, "y": np.roll(a["y"], 1)}),
                "V_check y differs from the clean split's"),
    "idx": (lambda s: _rewrite_split(s, "V_evidence", lambda a: {**a, "idx": a["idx"] + 1}),
            "V_evidence idx differs from the clean split's"),
    "train_y": (lambda s: _rewrite_split(s, "train", lambda a: {**a, "y": (a["y"] + 1) % 4}),
                "train y differs from the clean split's"),
    "missing_split": (lambda s: _edit_meta(s, lambda m: m["splits"].pop("V_select")), "no V_select split"),
    "split_clean_hash": (lambda s: _edit_meta(s, lambda m: m["splits"]["V_check"].update(clean_sha256="1" * 64)),
                         "V_check clean_sha256 is not the clean split's content hash"),
    "test_clean_hash": (lambda s: _edit_meta(s, lambda m: m["test"].update(clean_sha256="2" * 64)),
                        "test clean_sha256 is not the clean test's content hash"),
}


@pytest.mark.parametrize("case", sorted(MISMATCHES))
def test_a_condition_that_is_not_the_clean_data_is_refused(condition_store: ToyStore, case: str) -> None:
    mutate, message = MISMATCHES[case]
    mutate(condition_store)
    with pytest.raises(ConditionMismatch, match=message):
        deployment_splits(condition_store.dataset, CONDITION)
    with pytest.raises(ConditionMismatch, match=message):
        build_condition(condition_store, ("dev",))
    assert not (condition_store.private / "instances" / "index.json").exists()
    assert deployment_splits(condition_store.dataset, "deploy").condition == "deploy"


def test_condition_instances_use_the_condition_data_and_leave_deploy_alone(condition_store: ToyStore) -> None:
    deploy = {(o.split_role, o.seed): o.instance_id for o in build_v1(condition_store)}
    built = build_condition(condition_store)
    assert len(built) == 4 and {o.status for o in built} == {"created"}
    root = condition_store.private / "instances"
    folder = condition_dir(condition_store) / "splits"
    v_evidence, v_check = load_npz(folder / "V_evidence.npz"), load_npz(folder / "V_check.npz")
    clean_evidence = load_npz(condition_store.private / condition_store.dataset / "splits" / "V_evidence.npz")
    for outcome in built:
        instance = load_instance(root / outcome.instance_id)
        private = instance.private
        assert private["condition"] == outcome.condition == CONDITION
        assert private["vevidence_path"] == str(folder / "V_evidence.npz")
        assert private["vcheck_path"] == str(folder / "V_check.npz")
        pred_ev = predict(instance.forward, instance.params, v_evidence["x"], 64)
        pred_ch = predict(instance.forward, instance.params, v_check["x"], 64)
        clean_pred = predict(instance.forward, instance.params, clean_evidence["x"], 64)
        errors = int(np.count_nonzero(pred_ev != v_evidence["y"]))
        assert private["n_base_errors_vevidence"] == errors >= K_FAIL
        assert errors != int(np.count_nonzero(clean_pred != clean_evidence["y"]))
        assert private["n_base_errors_vcheck"] == int(np.count_nonzero(pred_ch != v_check["y"]))
        rows = np.asarray(private["evidence_row_ids"])
        np.testing.assert_array_equal(instance.evidence["x_fail"], v_evidence["x"][rows[:K_FAIL]])
        np.testing.assert_array_equal(instance.evidence["x_rand"], v_evidence["x"][rows[K_FAIL:]])
        assert np.all(instance.evidence["pred_fail"] != instance.evidence["y_fail"])
        assert private["evidence_seed"] == evidence_seed_v1(private["model_sha256"], CONDITION)
        twin = load_instance(root / deploy[(outcome.split_role, outcome.seed)])
        assert private["evidence_seed"] != twin.private["evidence_seed"]
        task_bytes = (root / outcome.instance_id / "task.json").read_bytes()
        assert task_bytes == (twin.instance_dir / "task.json").read_bytes()
        assert check_task_current(root / outcome.instance_id) == json.loads(task_bytes)
    assert {(o.split_role, o.seed): o.instance_id for o in build_v1(condition_store)} == deploy
    again = build_condition(condition_store)
    assert [(o.instance_id, o.status) for o in again] == [(o.instance_id, "reused") for o in built]
    dataset = condition_store.dataset
    in_condition = active_instance_ids(dataset=dataset, evidence_rule=EVIDENCE_RULE_V1, condition=CONDITION)
    assert set(in_condition) == {o.instance_id for o in built}
    assert set(active_instance_ids(dataset=dataset, condition="deploy")) == set(deploy.values())


def test_check_task_current_on_condition_instances(condition_store: ToyStore) -> None:
    built = build_condition(condition_store, ("dev",))
    root = condition_store.private / "instances"
    for outcome in built:
        check_task_current(root / outcome.instance_id)
    supersede_instances([built[0].instance_id], "test: condition instance retired")
    with pytest.raises(StaleInstanceError, match="superseded"):
        check_task_current(root / built[0].instance_id)
    check_task_current(root / built[1].instance_id)


def test_condition_leak_tokens_cover_every_editor_visible_file(
    condition_store: ToyStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert condition_leak_tokens(CONDITION, _meta(condition_store)) == sorted(
        {"toyblur_s2", "toyblur", "s2", *CONDITION_LEAK_WORDS}
    )
    assert npz_text({"w": np.zeros(2), "ids": np.array(["e01", "toyblur"])}) == "w\nids\ne01\ntoyblur"
    # A model_spec.json naming the corruption blocks the condition's instances; deploy does not scan for it.
    dev100 = condition_store.ids("dev")[100]
    spec_path = condition_store.private / condition_store.dataset / "models" / dev100 / "model_spec.json"
    spec = json.loads(spec_path.read_text())
    spec_path.write_text(json.dumps({**spec, "note": "robust to toyblur"}))
    with pytest.raises(ValueError, match=r"model_spec.json contains \['toyblur'\]"):
        build_condition(condition_store, ("dev",))
    assert all(not any(o.visible_file_findings.values()) for o in build_v1(condition_store, ("dev",)))
    spec_path.write_text(json.dumps(spec))
    # task.json is scanned with the condition tokens too.
    original = instances.build_task_v1

    def noisy_task(classes: list[str]) -> dict[str, Any]:
        task = original(classes)
        return {**task, "statement": task["statement"] + " Mind the noise."}

    monkeypatch.setattr(instances, "build_task_v1", noisy_task)
    with pytest.raises(ValueError, match=r"task.json for .* contains forbidden tokens \['noise'\]"):
        build_condition(condition_store, ("dev",))


def test_dev_readout_of_a_condition_instance_reads_the_condition_v_select(condition_store: ToyStore) -> None:
    built = {o.seed: o for o in build_condition(condition_store, ("dev",))}
    instance = load_instance(condition_store.private / "instances" / built[100].instance_id)
    v_select = load_npz(condition_dir(condition_store) / "splits" / "V_select.npz")
    clean = load_npz(condition_store.private / condition_store.dataset / "splits" / "V_select.npz")
    assert not np.array_equal(v_select["x"], clean["x"])
    np.testing.assert_array_equal(devreadout.readout_split(instance)["x"], v_select["x"])
    child = fixing_child(instance.params)
    record = devreadout.dev_readout(instance, child)
    base_pred = predict(instance.forward, instance.params, v_select["x"], 64)
    child_pred = predict(instance.forward, child, v_select["x"], 64)
    expected = sealed_readout(v_select["y"], base_pred, child_pred, n_classes=4)
    assert record["condition"] == CONDITION and record["split"] == "V_select" and record["n"] == len(v_select["y"])
    assert record["readouts"] == dict(expected)
    assert record["readouts"]["delta_acc"] > 0 and record["readouts"]["broken"] == 0


def test_short_tokens_match_whole_words_only() -> None:
    """``d1`` (a part of the drawn conditions' names) is hex-compatible: under substring matching most 64-id
    evidence draws contain it. Short tokens therefore match only as whole words; longer ones as substrings."""
    ids = "\n".join(f"e{code:012x}" for code in (0x3FD1A0B2C4D5, 0xD1D1D1D1D1D1, 0x000000000D10))
    assert scan_for_tokens(ids, ["d1", "s3"]) == []
    assert scan_for_tokens("severity d1, see colormix_d1", ["d1"]) == ["d1"]
    assert scan_for_tokens("D1 at the start", ["d1"]) == ["d1"]
    assert scan_for_tokens("contrast_s3", ["s3", "contrast"]) == ["contrast", "s3"]
    assert scan_for_tokens("gaussiannoise", ["noise"]) == ["noise"]


def test_drawn_conditions_add_their_transform_and_name_stems() -> None:
    tokens = condition_leak_tokens("blur5_d1", {"condition": {"transform": "blur5"}})
    assert {"blur5_d1", "blur5", "blur", "d1"} <= set(tokens)
    assert set(condition_leak_tokens("gamma_d1", {"condition": {"transform": "gamma"}})) == {
        "gamma_d1", "gamma", "d1", *CONDITION_LEAK_WORDS
    }


def test_drawn_conditions_are_dev_only(toy_store: ToyStore) -> None:  # noqa: F811
    """A main (or default, all-role) build of a drawn condition is refused before any file is read."""
    from test_evalkit_instances_store import DEV_PLAN

    from weightcode.evalkit.instances import build_deployment_instances

    for roles in (("main",), ("dev", "main")):
        with pytest.raises(ValueError, match="dev-only"):
            build_deployment_instances(toy_store.dataset, "gamma_d1", selection=DEV_PLAN, decoys={},
                                       split_roles=roles)
