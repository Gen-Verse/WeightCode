"""draw_evidence_for: rule evidence-v1 for any student on an instance's condition (PREREG E4 rounds)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from test_evalkit_instances_store import (  # noqa: F401
    CONDITION,
    ToyStore,
    add_condition,
    build_condition,
    load_npz,
    toy_store,
)

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import params_sha256
from weightcode.evalkit.evidence import EVIDENCE_KEYS, K_FAIL, K_RAND, evidence_content_sha256, evidence_seed_v1
from weightcode.evalkit.patch_check import predict
from weightcode.evalkit.student_evidence import StudentEvidenceError, draw_evidence_for


@pytest.fixture
def condition_instances(toy_store: ToyStore) -> list[Instance]:  # noqa: F811
    """The evidence-v1 instances of the toy condition, one per selected model."""
    add_condition(toy_store)
    built = build_condition(toy_store)
    assert len(built) == 4
    return [load_instance(toy_store.private / "instances" / o.instance_id) for o in built]


def _perturbed(instance: Instance) -> dict[str, np.ndarray]:
    """The instance's params with every weight scaled and shifted: a student that is in no instance."""
    rng = np.random.default_rng(7)
    return {
        key: (value * np.float32(0.9) + rng.normal(0, 0.05, value.shape).astype(np.float32)).astype(value.dtype)
        for key, value in instance.params.items()
    }


def test_base_params_reproduce_the_instance_evidence(condition_instances: list[Instance]) -> None:
    for instance in condition_instances:
        drawn = draw_evidence_for(instance, instance.params)
        assert set(drawn.arrays) == set(EVIDENCE_KEYS)
        for key in EVIDENCE_KEYS:
            np.testing.assert_array_equal(drawn.arrays[key], instance.evidence[key])
        private = instance.private
        assert drawn.content_sha256 == private["evidence_content_sha256"]
        assert drawn.seed == private["evidence_seed"]
        assert drawn.row_ids.tolist() == private["evidence_row_ids"]
        assert drawn.source_idx.tolist() == private["evidence_source_idx"]
        assert drawn.split_ids == private["evidence_split_ids"]
        assert drawn.n_errors_vevidence == private["n_base_errors_vevidence"]
        assert (drawn.n_fail, drawn.n_rand) == (K_FAIL, K_RAND)
        assert drawn.instance_id == instance.instance_id and drawn.condition == CONDITION
        assert draw_evidence_for(instance.instance_id, instance.params).content_sha256 == drawn.content_sha256


def test_another_model_gets_the_evidence_of_its_own_instance(condition_instances: list[Instance]) -> None:
    first = condition_instances[0]
    for other in condition_instances[1:]:
        drawn = draw_evidence_for(first, other.params)
        assert drawn.content_sha256 == other.private["evidence_content_sha256"] != first.private[
            "evidence_content_sha256"
        ]
        assert drawn.seed == other.private["evidence_seed"]


def test_a_new_student_gets_its_own_errors_and_its_own_stream(condition_instances: list[Instance]) -> None:
    instance = condition_instances[0]
    student = _perturbed(instance)
    drawn = draw_evidence_for(instance, student)
    assert drawn.params_sha256 == params_sha256(student) != instance.params_sha256
    assert drawn.seed == evidence_seed_v1(params_sha256(student), CONDITION) != instance.private["evidence_seed"]
    v_evidence = load_npz(instance.private["vevidence_path"])
    pred = predict(instance.forward, student, v_evidence["x"], 64)
    rows = drawn.row_ids
    assert len(set(rows.tolist())) == K_FAIL + K_RAND
    np.testing.assert_array_equal(drawn.x, v_evidence["x"][rows])
    np.testing.assert_array_equal(drawn.y, v_evidence["y"][rows])
    np.testing.assert_array_equal(drawn.pred, pred[rows])
    assert np.all(drawn.arrays["pred_fail"] != drawn.arrays["y_fail"])
    assert drawn.n_errors_vevidence == int(np.count_nonzero(pred != v_evidence["y"]))
    np.testing.assert_array_equal(drawn.source_idx, v_evidence["idx"][rows])
    assert evidence_content_sha256(drawn.arrays) == drawn.content_sha256
    again = draw_evidence_for(instance, {key: value.copy() for key, value in student.items()})
    assert again.content_sha256 == drawn.content_sha256 and again.row_ids.tolist() == rows.tolist()


def test_params_of_another_layout_are_refused(condition_instances: list[Instance]) -> None:
    instance = condition_instances[0]
    name = next(key for key in sorted(instance.params) if instance.params[key].ndim >= 2)
    missing = {key: value for key, value in instance.params.items() if key != name}
    with pytest.raises(StudentEvidenceError, match="keys"):
        draw_evidence_for(instance, missing)
    reshaped = dict(instance.params, **{name: instance.params[name].reshape(-1)})
    with pytest.raises(StudentEvidenceError, match=name):
        draw_evidence_for(instance, reshaped)
    retyped = dict(instance.params, **{name: instance.params[name].astype(np.float64)})
    with pytest.raises(StudentEvidenceError, match=name):
        draw_evidence_for(instance, retyped)


def test_a_changed_vevidence_pool_is_refused(condition_instances: list[Instance]) -> None:
    instance = condition_instances[0]
    path = instance.private["vevidence_path"]
    arrays = load_npz(path)
    arrays["x"] = arrays["x"].copy()
    arrays["x"][0] ^= 1
    np.savez(path, **arrays)
    with pytest.raises(StudentEvidenceError, match="V_evidence content hash"):
        draw_evidence_for(instance, instance.params)


def test_an_instance_of_another_rule_is_refused(condition_instances: list[Instance]) -> None:
    instance = condition_instances[0]
    private = dict(instance.private, evidence_rule="evidence-v0")
    legacy = dataclasses.replace(instance, private=private)
    with pytest.raises(StudentEvidenceError, match="evidence-v1 instances only"):
        draw_evidence_for(legacy, instance.params)
