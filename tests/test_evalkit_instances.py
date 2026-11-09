"""Tests for the evidence-v0 draw and target-pair instances (weightcode.evalkit.evidence, weightcode.evalkit.instances).

evidence-v1 instances, supersession, decoy maps and edit limits are tested in test_evalkit_instances_v1.py.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_instances_store import (  # noqa: F401
    DEV_PLAN,
    EXPECTED_CHOSEN,
    ToyStore,
    build_toy_store,
    build_v0,
    toy_store,
)

from weightcode.baselines.instance_io import load_instance, make_oracle
from weightcode.evalkit.evidence import EVIDENCE_KEYS, K_RAND, draw_evidence, evidence_seed, load_evidence
from weightcode.evalkit.instances import (
    DEFAULT_DECOYS,
    EDITOR_VISIBLE_FILES,
    QUERY_BUDGET,
    InstanceConflict,
    build_dataset_instances,
    leak_tokens,
    limit_definitions,
    read_index,
)
from weightcode.evalkit.targets import K_FAIL, run_target_selection
from weightcode.patch import OP_NAMES, task_limits


def _evidence_inputs(n: int = 200, cell: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x = rng.integers(0, 256, size=(n, 1, 2, 2), dtype=np.uint8)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    pred = y.copy()
    rows = np.flatnonzero(y == 0)[:cell]
    pred[rows] = 1
    return x, y, pred


def test_evidence_seed_is_the_preregistered_formula() -> None:
    sha = "ab" * 32
    expected = int.from_bytes(hashlib.sha256(f"{sha}|3|5|evidence-v0".encode()).digest()[:8], "big")
    assert evidence_seed(sha, 3, 5) == expected
    assert evidence_seed(sha, 5, 3) != expected
    with pytest.raises(ValueError):
        evidence_seed("not-a-hash", 3, 5)


def test_draw_evidence_contract() -> None:
    x, y, pred = _evidence_inputs()
    evidence = draw_evidence(x, y, pred, a=0, b=1, seed=7)
    arrays, rows = evidence.arrays, evidence.row_ids
    assert set(arrays) == set(EVIDENCE_KEYS)
    assert len(rows) == K_FAIL + K_RAND == len(set(rows.tolist()))
    fail_rows, rand_rows = rows[:K_FAIL], rows[K_FAIL:]
    assert np.all(y[fail_rows] == 0) and np.all(pred[fail_rows] == 1)
    assert np.array_equal(arrays["x_fail"], x[fail_rows]) and np.array_equal(arrays["x_rand"], x[rand_rows])
    assert np.array_equal(arrays["y_rand"], y[rand_rows]) and np.array_equal(arrays["pred_rand"], pred[rand_rows])
    assert np.all(arrays["y_fail"] == 0) and np.all(arrays["pred_fail"] == 1)
    assert arrays["y_fail"].dtype == np.int64 and arrays["x_fail"].dtype == np.uint8
    ids = evidence.split_ids
    assert len(set(ids)) == len(ids) == K_FAIL + K_RAND
    assert all(len(i) == 13 and i.startswith("e") for i in ids)
    again = draw_evidence(x, y, pred, a=0, b=1, seed=7)
    assert np.array_equal(again.row_ids, rows) and again.split_ids == ids
    other = draw_evidence(x, y, pred, a=0, b=1, seed=8)
    assert not np.array_equal(other.row_ids, rows)


def test_draw_evidence_refuses_a_small_cell() -> None:
    x, y, pred = _evidence_inputs(cell=K_FAIL - 1)
    with pytest.raises(ValueError, match="k_fail"):
        draw_evidence(x, y, pred, a=0, b=1, seed=7)


def _rebuild(store: ToyStore) -> list:
    return build_dataset_instances(store.dataset, selection=DEV_PLAN, decoys=DEFAULT_DECOYS)


def test_instance_layout_task_and_private(toy_store: ToyStore) -> None:  # noqa: F811
    outcomes = build_v0(toy_store)
    assert len(outcomes) == 8 and all(o.status == "created" for o in outcomes)
    assert [(o.split_role, o.seed) for o in outcomes] == [
        ("dev", 100), ("dev", 100), ("dev", 101), ("dev", 101), ("main", 0), ("main", 0), ("main", 1), ("main", 1)
    ]
    root = toy_store.private / "instances"
    with np.load(toy_store.private / toy_store.dataset / "splits" / "V_evidence.npz") as archive:
        v_evidence = {key: archive[key] for key in archive.files}
    model_ids = [str(row["model_id"]) for row in toy_store.models]
    for outcome in outcomes:
        folder = root / outcome.instance_id
        assert sorted(p.name for p in folder.iterdir()) == sorted([*EDITOR_VISIBLE_FILES, "private.json"])
        task_text = (folder / "task.json").read_text()
        lower = task_text.lower()
        assert not [t for t in [*leak_tokens(toy_store.dataset), *(m.lower() for m in model_ids)] if t in lower]
        task = json.loads(task_text)
        a, b = outcome.target
        assert task["schema"] == "weightcode.task.v0"
        assert task["target"] == {"a": a, "b": b, "a_name": task["classes"][a], "b_name": task["classes"][b]}
        assert task["selection_counts"]["n_class_a"] == 100
        assert task["selection_counts"]["n_class_a_predicted_b"] == {(0, 1): 30, (3, 2): 20}[(a, b)]
        assert task["patch_check"]["budget_queries"] == QUERY_BUDGET
        assert task["limits"] == task_limits()
        assert task["limit_definitions"] == limit_definitions(task["limits"])
        assert task["patch_format"] == {"schema": "weightcode.patch.v0", "allowed_ops": list(OP_NAMES)}
        assert len(task["training_free_rule"]["forbidden"]) == 5
        private = json.loads((folder / "private.json").read_text())
        assert private["split_role"] == outcome.split_role and private["target"] == [a, b]
        assert private["evidence_rule"] == "evidence-v0" and private["schema"] == "weightcode.instance_private.v0"
        assert private["k_fail"] == K_FAIL and private["k_rand"] == K_RAND
        assert private["reference_ceiling_models"] == []
        assert Path(private["vcheck_path"]).is_absolute() and Path(private["decoy_model_path"]).is_file()
        assert private["evidence_seed"] == evidence_seed(private["model_sha256"], a, b)
        evidence = load_evidence(folder / "evidence.npz")
        rows = np.asarray(private["evidence_row_ids"])
        assert np.array_equal(evidence["x_fail"], v_evidence["x"][rows[:K_FAIL]])
        assert np.array_equal(evidence["y_rand"], v_evidence["y"][rows[K_FAIL:]])
        assert evidence["split_ids"].tolist() == private["evidence_split_ids"]
        assert private["evidence_source_idx"] == v_evidence["idx"][rows].tolist()
        assert np.all(evidence["pred_fail"] == b)
        instance = load_instance(folder)
        oracle = make_oracle(instance)
        check = oracle.query(instance.params, "noop")
        assert check["n_base_target_errors"] == private["vcheck_count"] and check["score"] == 0
    index = read_index()
    assert [row["instance_id"] for row in index["instances"]] == [o.instance_id for o in outcomes]
    assert {tuple(row["target"]) for row in index["instances"]} == set(EXPECTED_CHOSEN)


def test_decoys_follow_the_map(toy_store: ToyStore) -> None:  # noqa: F811
    outcomes = build_v0(toy_store)
    ids = {**toy_store.ids("base"), **toy_store.ids("dev")}
    for outcome in outcomes:
        private = json.loads((toy_store.private / "instances" / outcome.instance_id / "private.json").read_text())
        expected_seed = {0: 1, 1: 0, 100: 101, 101: 100}[outcome.seed]
        assert private["decoy_seed"] == outcome.decoy_seed == expected_seed
        assert private["decoy_model_id"] == ids[expected_seed]
        assert private["decoy_model_path"].endswith(f"{ids[expected_seed]}/model.npz")


def test_rebuild_reuses_and_a_changed_instance_is_refused(toy_store: ToyStore) -> None:  # noqa: F811
    first = build_v0(toy_store)
    second = _rebuild(toy_store)
    assert [o.instance_id for o in second] == [o.instance_id for o in first]
    assert all(o.status == "reused" for o in second)
    root = toy_store.private / "instances"
    assert len([p for p in root.iterdir() if p.is_dir()]) == 8
    target = root / first[0].instance_id / "evidence.npz"
    evidence = load_evidence(target)
    evidence["y_rand"] = evidence["y_rand"][::-1].copy()
    np.savez(target, **evidence)
    with pytest.raises(InstanceConflict, match="private.json|evidence"):
        _rebuild(toy_store)


def test_dev_first_build_then_main(toy_store: ToyStore) -> None:  # noqa: F811
    dev = build_v0(toy_store, roles=("dev",))
    assert {o.split_role for o in dev} == {"dev"} and len(dev) == 4
    both = build_v0(toy_store)
    assert [o.status for o in both] == ["reused"] * 4 + ["created"] * 4


def test_index_and_directories_must_agree(toy_store: ToyStore) -> None:  # noqa: F811
    outcomes = build_v0(toy_store)
    shutil.rmtree(toy_store.private / "instances" / outcomes[-1].instance_id)
    with pytest.raises(InstanceConflict, match="disagree"):
        _rebuild(toy_store)


def test_a_selected_seed_without_a_model_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = build_toy_store(tmp_path / "store", main_seeds=(0,))
    store.activate(monkeypatch)
    run_target_selection(store.dataset, selection=DEV_PLAN, split_roles=("dev",))
    with pytest.raises(LookupError, match="main seed 1"):
        build_dataset_instances(store.dataset, selection=DEV_PLAN, decoys=DEFAULT_DECOYS, split_roles=("main",))


def test_ceiling_models_are_recorded_as_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = build_toy_store(tmp_path / "store", dataset="toycolor", colored=True, ceiling_seeds=(0, 1))
    store.activate(monkeypatch)
    outcomes = build_v0(store, roles=("dev",))
    private = json.loads((store.private / "instances" / outcomes[0].instance_id / "private.json").read_text())
    assert [ref["model_id"] for ref in private["reference_ceiling_models"]] == [
        store.ids("ceiling")[0],
        store.ids("ceiling")[1],
    ]
    assert all(o.model_id not in store.ids("ceiling").values() for o in outcomes)
