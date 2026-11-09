"""Sealed evaluation of instances of a named deployment condition, on a synthetic store only: the test set is the
condition's own (resolved through deployment_splits), and a condition test set whose y or idx is out of line with the
clean test set is refused after the start record."""

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
    condition_test_path,
    fixing_child,
    load_npz,
    toy_store,
    write_condition_meta,
)

import weightcode.evalkit.sealed as sealed
from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import params_sha256, save_params
from weightcode.evalkit.metrics import sealed_readout
from weightcode.evalkit.patch_check import predict


def _main_seed0(store: ToyStore) -> str:
    add_condition(store)
    return next(o.instance_id for o in build_condition(store, ("main",)) if o.seed == 0)


def _manifest(store: ToyStore, tmp_path: Path, instance_id: str) -> Path:
    instance = load_instance(store.private / "instances" / instance_id)
    child_path = tmp_path / "children" / "fix.npz"
    child_path.parent.mkdir(parents=True)
    digest = save_params(child_path, fixing_child(instance.params))
    payload = {
        "schema": "weightcode.sealed_manifest.v0",
        "created_at": "2026-11-03T12:00:00+00:00",
        "note": "condition test",
        "entries": [{"instance_id": instance_id, "arm": "M2", "repeat": 0, "child_params_path": str(child_path),
                     "child_params_sha256": digest}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _events(store: ToyStore) -> list[str]:
    path = store.sealed / "registry.jsonl"
    return [json.loads(line)["event"] for line in path.read_text().splitlines()] if path.exists() else []


def test_a_condition_instance_is_read_on_the_condition_test_set(
    toy_store: ToyStore, tmp_path: Path  # noqa: F811
) -> None:
    instance_id = _main_seed0(toy_store)
    results_path, count = sealed.evaluate_manifest(_manifest(toy_store, tmp_path, instance_id))
    assert count == 1
    record = json.loads(results_path.read_text().splitlines()[0])
    meta = json.loads((condition_dir(toy_store) / "dataset.json").read_text())
    assert record["condition"] == CONDITION and record["test_sha256"] == meta["test"]["sha256"]
    assert record["test_sha256"] != meta["test"]["clean_sha256"]
    instance = load_instance(toy_store.private / "instances" / instance_id)
    test = load_npz(condition_test_path(toy_store))
    base_pred = predict(instance.forward, instance.params, test["x"], 64)
    child_pred = predict(instance.forward, fixing_child(instance.params), test["x"], 64)
    expected = sealed_readout(test["y"], base_pred, child_pred, n_classes=4)
    assert record["readouts"] == json.loads(json.dumps(expected))
    assert record["readouts"]["delta_acc"] > 0
    assert _events(toy_store) == ["evaluation_started", "evaluation_finished"]


@pytest.mark.parametrize("key", ["y", "idx"])
def test_a_condition_test_set_out_of_line_with_the_clean_one_is_refused(
    toy_store: ToyStore, tmp_path: Path, key: str  # noqa: F811
) -> None:
    instance_id = _main_seed0(toy_store)
    path = condition_test_path(toy_store)
    test = load_npz(path)
    test[key] = np.roll(test[key], 1)
    np.savez(path, **test)
    meta = json.loads((condition_dir(toy_store) / "dataset.json").read_text())
    meta["test"]["sha256"] = params_sha256(test)
    write_condition_meta(toy_store, meta)
    with pytest.raises(sealed.SealedEvalError, match=f"{key} differs from the clean test set"):
        sealed.evaluate_manifest(_manifest(toy_store, tmp_path, instance_id))
    assert _events(toy_store) == ["evaluation_started", "evaluation_failed"]
