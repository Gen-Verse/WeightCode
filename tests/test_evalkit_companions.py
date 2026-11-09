"""Clean companions, the joint patch-check oracle, the clean dev readout and the replay draw (E1 learning setup)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_instances_store import (  # noqa: F401
    CONDITION,
    ToyStore,
    add_condition,
    build_condition,
    condition_dir,
    fixing_child,
    load_npz,
    toy_store,
    write_condition_meta,
)

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import params_sha256
from weightcode.evalkit import devreadout
from weightcode.evalkit.companions import (
    CompanionError,
    build_replay,
    clean_companion,
    load_clean_rows,
    load_replay,
    make_joint_oracle,
    replay_paths,
)
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.patch_check import BudgetExceeded, PatchCheckOracle, predict
from weightcode.evalkit.targets import ConditionMismatch


@pytest.fixture
def condition_store(toy_store: ToyStore) -> ToyStore:  # noqa: F811
    """The default toy store plus the condition toyblur_s2."""
    add_condition(toy_store)
    return toy_store


def _clean(store: ToyStore, split: str) -> dict[str, np.ndarray]:
    return load_npz(store.private / store.dataset / "splits" / f"{split}.npz")


def _shifted(store: ToyStore, split: str) -> dict[str, np.ndarray]:
    return load_npz(condition_dir(store) / "splits" / f"{split}.npz")


def _dev_instance(store: ToyStore) -> Instance:
    outcome = next(o for o in build_condition(store, roles=("dev",)) if o.split_role == "dev")
    return load_instance(instances_root() / outcome.instance_id)


@pytest.mark.parametrize("split", ["V_evidence", "V_check", "V_select"])
def test_companion_has_the_condition_rows_with_clean_images(condition_store: ToyStore, split: str) -> None:
    rows = clean_companion(condition_store.dataset, CONDITION, split)
    clean, shifted = _clean(condition_store, split), _shifted(condition_store, split)
    assert np.array_equal(rows.x, clean["x"])
    assert not np.array_equal(rows.x, shifted["x"])
    assert np.array_equal(rows.y, shifted["y"]) and np.array_equal(rows.idx, shifted["idx"])
    assert np.array_equal(rows.row_ids, np.arange(len(shifted["y"])))
    assert rows.clean_sha256 == params_sha256(clean) and rows.condition_sha256 == params_sha256(shifted)


@pytest.mark.parametrize(("condition", "split"), [("deploy", "V_check"), (CONDITION, "train"), (CONDITION, "test")])
def test_companion_refuses_deploy_and_other_splits(condition_store: ToyStore, condition: str, split: str) -> None:
    with pytest.raises(CompanionError):
        clean_companion(condition_store.dataset, condition, split)


def test_companion_refuses_rows_that_do_not_align(condition_store: ToyStore) -> None:
    path = condition_dir(condition_store) / "splits" / "V_check.npz"
    arrays = load_npz(path)
    arrays["y"] = np.roll(arrays["y"], 1)
    np.savez(path, **arrays)
    meta = json.loads((condition_dir(condition_store) / "dataset.json").read_text())
    meta["splits"]["V_check"]["sha256"] = params_sha256(arrays)
    write_condition_meta(condition_store, meta)
    with pytest.raises((CompanionError, ConditionMismatch)):
        clean_companion(condition_store.dataset, CONDITION, "V_check")


def test_load_clean_rows_resolves_an_instance(condition_store: ToyStore) -> None:
    instance = _dev_instance(condition_store)
    by_object = load_clean_rows(instance, "V_select")
    by_id = load_clean_rows(instance.instance_id, "V_select")
    assert np.array_equal(by_object.x, _clean(condition_store, "V_select")["x"])
    assert np.array_equal(by_object.x, by_id.x) and by_object.condition == CONDITION


def test_joint_oracle_scores_both_sets_under_one_budget(condition_store: ToyStore) -> None:
    instance = _dev_instance(condition_store)
    oracle = make_joint_oracle(instance, budget=2)
    child = fixing_child(dict(instance.params))
    parts = {}
    for name, data in (("new", _shifted(condition_store, "V_check")), ("old", _clean(condition_store, "V_check"))):
        single = PatchCheckOracle(data["x"], data["y"], instance.forward, instance.params, None, 1)
        parts[name] = single.query(child, "single")
    entry = oracle.query(child, "p1")
    assert entry["score_new"] == parts["new"]["score"] and entry["score_old"] == parts["old"]["score"]
    assert entry["score"] == parts["new"]["score"] + parts["old"]["score"]
    assert entry["fixed"] - entry["broken"] == entry["score"]
    assert entry["n"] == parts["new"]["n"] + parts["old"]["n"]
    assert entry["n_base_errors"] == parts["new"]["n_base_errors"] + parts["old"]["n_base_errors"]
    base_right = sum(p["n"] * p["base_acc"] for p in parts.values())
    child_right = sum(p["n"] * p["child_acc"] for p in parts.values())
    assert entry["base_acc"] == pytest.approx(base_right / entry["n"])
    assert entry["child_acc"] == pytest.approx(child_right / entry["n"])
    assert entry["child_acc"] - entry["base_acc"] == pytest.approx(entry["score"] / entry["n"])
    assert entry["queries_remaining"] == 1
    bad = dict(child)
    first = next(iter(bad))
    bad[first] = bad[first][..., :1]
    with pytest.raises(ValueError):
        oracle.query(bad, "bad")
    with pytest.raises(ValueError):
        oracle.query(child, "p1")
    assert oracle.queries_remaining == 1
    oracle.query(dict(instance.params), "p2")
    with pytest.raises(BudgetExceeded):
        oracle.query(child, "p3")
    assert oracle.final().score == max([0] + [c.score for c in oracle.candidates()])


def test_joint_readout_reads_both_selects(condition_store: ToyStore) -> None:
    instance = _dev_instance(condition_store)
    child = fixing_child(dict(instance.params))
    record = devreadout.dev_readout_joint(instance, child)
    clean = _clean(condition_store, "V_select")
    base_right = predict(instance.forward, instance.params, clean["x"], 1024) == clean["y"]
    child_right = predict(instance.forward, child, clean["x"], 1024) == clean["y"]
    assert record["n_clean"] == len(clean["y"])
    assert record["readouts_clean"]["delta_acc"] == pytest.approx(child_right.mean() - base_right.mean())
    assert record["readouts"] == devreadout.dev_readout(instance, child)["readouts"]


def test_clean_flag_on_a_base_noop_run(condition_store: ToyStore, tmp_path: Path) -> None:
    instance = _dev_instance(condition_store)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    final = {"is_base_noop": True, "child_sha256": instance.params_sha256, "score": 0, "fixed": 0, "broken": 0,
             "edit_size": 0}
    (run_dir / "run.json").write_text(json.dumps({"instance": {"instance_id": instance.instance_id},
                                                  "method": "TEST", "final": final}))
    record = devreadout.readout_run(run_dir / "run.json", clean=True)
    assert record["readouts"]["delta_acc"] == 0 and record["readouts_clean"]["delta_acc"] == 0
    assert record["n_clean"] == len(_clean(condition_store, "V_select")["y"])


def test_replay_is_drawn_once_from_train_and_verified(toy_store: ToyStore) -> None:  # noqa: F811
    record = build_replay(toy_store.dataset, n=8)
    arrays = load_replay(toy_store.dataset)
    train = _clean(toy_store, "train")
    rows = np.asarray(record["row_ids"])
    assert len(set(rows.tolist())) == 8 and record["pool_split"] == "train"
    assert np.array_equal(arrays["x"], train["x"][rows]) and np.array_equal(arrays["y"], train["y"][rows])
    assert len(set(arrays["split_ids"].tolist())) == 8
    assert build_replay(toy_store.dataset, n=8)["content_sha256"] == record["content_sha256"]
    with pytest.raises(CompanionError):
        build_replay(toy_store.dataset, n=9)
    npz_path, _ = replay_paths(toy_store.dataset)
    arrays["x"] = arrays["x"].copy()
    arrays["x"][0] ^= 1
    np.savez(npz_path, **arrays)
    with pytest.raises(CompanionError):
        load_replay(toy_store.dataset)
