"""Tests for the sealed evaluation CLI (weightcode.evalkit.sealed) on synthetic stores only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_instances_store import (  # noqa: F401
    PLAN,
    ToyStore,
    breaking_child,
    build_toy_store,
    build_v0,
    build_v1,
    fixing_child,
    toy_store,
)

import weightcode.evalkit.sealed as sealed
from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.evalkit.instances import supersede_instances
from weightcode.evalkit.metrics import color_following_rate, color_split_fixes, color_split_target_fixes, sealed_readout
from weightcode.evalkit.patch_check import predict


def _instances(store: ToyStore) -> dict[tuple[int, int], str]:
    """Build the evidence-v0 instances; return the main seed-0 instance id per target pair."""
    outcomes = build_v0(store)
    return {o.target: o.instance_id for o in outcomes if o.split_role == "main" and o.seed == 0 and o.target}


def _v1_instance(store: ToyStore) -> str:
    """Build the evidence-v1 instances; return the main seed-0 instance id."""
    return next(o.instance_id for o in build_v1(store) if o.split_role == "main" and o.seed == 0)


def _child(folder: Path, name: str, params: dict[str, np.ndarray]) -> tuple[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npz"
    return str(path), save_params(path, params)


def _manifest(path: Path, entries: list[dict[str, object]], note: str = "test") -> Path:
    payload = {"schema": "weightcode.sealed_manifest.v0", "created_at": "2026-11-03T12:00:00+00:00", "note": note,
               "entries": entries}
    path.write_text(json.dumps(payload, indent=2))
    return path


def _entry(instance_id: str, arm: str, repeat: int, child: tuple[str, str]) -> dict[str, object]:
    return {"instance_id": instance_id, "arm": arm, "repeat": repeat, "child_params_path": child[0],
            "child_params_sha256": child[1]}


def _registry(store: ToyStore) -> list[dict[str, object]]:
    path = store.sealed / "registry.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _test_set(store: ToyStore) -> dict[str, np.ndarray]:
    with np.load(store.sealed / store.dataset / "test.npz") as archive:
        return {key: archive[key] for key in archive.files}


@pytest.fixture
def three_entries(toy_store: ToyStore, tmp_path: Path) -> tuple[Path, dict[str, object]]:  # noqa: F811
    ids = _instances(toy_store)
    inst01 = load_instance(toy_store.private / "instances" / ids[(0, 1)])
    inst32 = load_instance(toy_store.private / "instances" / ids[(3, 2)])
    children = tmp_path / "children"
    fix = _child(children, "fix", fixing_child(inst01.params))
    noop = _child(children, "noop", inst01.params)
    broken = _child(children, "broken", breaking_child(inst32.params))
    entries = [
        _entry(ids[(0, 1)], "M2", 0, fix),
        _entry(ids[(0, 1)], "B1", 0, noop),
        _entry(ids[(3, 2)], "M2", 1, broken),
    ]
    return _manifest(tmp_path / "manifest.json", entries), {"ids": ids, "fix": fix, "noop": noop, "broken": broken}


def test_end_to_end_readouts_registry_and_output(
    toy_store: ToyStore, three_entries: tuple[Path, dict[str, object]], capsys: pytest.CaptureFixture[str]  # noqa: F811
) -> None:
    manifest_path, info = three_entries
    assert sealed.main(["--manifest", str(manifest_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    sha = sealed.manifest_sha256(json.loads(manifest_path.read_text()))
    results_path = toy_store.sealed / "results" / f"{sha}.jsonl"
    assert lines == [f"results: {results_path}", "entries: 3"]
    records = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert [(r["arm"], r["repeat"]) for r in records] == [("M2", 0), ("B1", 0), ("M2", 1)]
    test = _test_set(toy_store)
    for record in records:
        instance = load_instance(toy_store.private / "instances" / record["instance_id"])
        child_path = {info["fix"][1]: info["fix"][0], info["noop"][1]: info["noop"][0],
                      info["broken"][1]: info["broken"][0]}[record["child_params_sha256"]]
        child = load_params(child_path)
        base_pred = predict(instance.forward, instance.params, test["x"], 64)
        child_pred = predict(instance.forward, child, test["x"], 64)
        a, b = instance.target
        assert record["readouts"] == json.loads(json.dumps(sealed_readout(test["y"], base_pred, child_pred, a, b, 4)))
        assert record["split_role"] == "main" and record["target"] == [a, b]
        assert record["evidence_rule"] == "evidence-v0" and record["condition"] == "deploy"
        assert record["base_params_sha256"] == instance.params_sha256
        assert record["forward_sha256"] == instance.forward_sha256
        assert record["test_sha256"] == params_sha256(test)
        assert not [key for key in record["readouts"] if "cfr" in key or "color" in key]
    fix, noop, broken = (r["readouts"] for r in records)
    assert fix["R"] == 1.0 and fix["fixed_target"] == fix["n_base_target_errors"] == 20 and fix["broken"] == 0
    assert noop["R"] == 0.0 and noop["fixed"] == noop["broken"] == 0
    assert broken["R"] < 0 and broken["broken"] > 0
    events = _registry(toy_store)
    assert [e["event"] for e in events] == ["evaluation_started", "evaluation_finished"]
    assert all(e["manifest_sha256"] == sha for e in events)
    assert events[0]["n_entries"] == 3 and events[1]["n_entries"] == 3
    assert (toy_store.sealed / "manifests" / f"{sha}.json").read_text() == manifest_path.read_text()


def test_second_read_is_refused_even_when_reformatted(
    toy_store: ToyStore, three_entries: tuple[Path, dict[str, object]], tmp_path: Path  # noqa: F811
) -> None:
    manifest_path, _ = three_entries
    assert sealed.main(["--manifest", str(manifest_path)]) == 0
    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(json.loads(manifest_path.read_text()), separators=(",", ":")))
    assert sealed.main(["--manifest", str(compact)]) == 2
    assert len(_registry(toy_store)) == 2


def test_hash_and_layout_mismatches_are_refused_before_any_record(
    toy_store: ToyStore, three_entries: tuple[Path, dict[str, object]], tmp_path: Path  # noqa: F811
) -> None:
    _, info = three_entries
    ids = info["ids"]
    wrong_sha = (info["fix"][0], "0" * 64)
    with pytest.raises(sealed.SealedEvalError, match="params_sha256"):
        sealed.evaluate_manifest(_manifest(tmp_path / "bad_sha.json", [_entry(ids[(0, 1)], "M2", 0, wrong_sha)]))
    instance = load_instance(toy_store.private / "instances" / ids[(0, 1)])
    reshaped = dict(instance.params, **{"fc2.bias": np.zeros(5, dtype=np.float32)})
    child = _child(tmp_path / "children", "reshaped", reshaped)
    with pytest.raises(sealed.SealedEvalError, match="fc2.bias"):
        sealed.evaluate_manifest(_manifest(tmp_path / "bad_shape.json", [_entry(ids[(0, 1)], "M2", 0, child)]))
    with pytest.raises(sealed.SealedEvalError, match="unknown instance"):
        sealed.evaluate_manifest(_manifest(tmp_path / "bad_id.json", [_entry("i00000000", "M2", 0, info["fix"])]))
    assert _registry(toy_store) == []


def test_relative_child_paths_resolve_against_the_manifest(
    toy_store: ToyStore, three_entries: tuple[Path, dict[str, object]], tmp_path: Path  # noqa: F811
) -> None:
    _, info = three_entries
    relative = ("children/fix.npz", info["fix"][1])
    path = _manifest(tmp_path / "relative.json", [_entry(info["ids"][(0, 1)], "M2", 0, relative)])
    results_path, count = sealed.evaluate_manifest(path)
    assert count == 1 and results_path.exists()


def test_started_record_precedes_the_first_test_label(
    toy_store: ToyStore, three_entries: tuple[Path, dict[str, object]], monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    manifest_path, _ = three_entries
    original = sealed._load_test_set
    seen = []

    def guarded(dataset: str) -> dict[str, np.ndarray]:
        seen.append([e["event"] for e in _registry(toy_store)])
        return original(dataset)

    monkeypatch.setattr(sealed, "_load_test_set", guarded)
    sealed.evaluate_manifest(manifest_path)
    assert seen == [["evaluation_started"]]


def test_a_failure_after_start_is_recorded_and_spends_the_manifest(
    toy_store: ToyStore, three_entries: tuple[Path, dict[str, object]], monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    manifest_path, _ = three_entries

    def failing(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("readout failure")

    monkeypatch.setattr(sealed, "sealed_readout", failing)
    with pytest.raises(RuntimeError, match="readout failure"):
        sealed.evaluate_manifest(manifest_path)
    events = _registry(toy_store)
    assert [e["event"] for e in events] == ["evaluation_started", "evaluation_failed"]
    assert not list((toy_store.sealed / "results").glob("*.jsonl"))
    monkeypatch.undo()
    toy_store.activate(monkeypatch)
    with pytest.raises(sealed.SealedEvalError, match="already evaluated"):
        sealed.evaluate_manifest(manifest_path)


def test_manifest_schema_is_strict(tmp_path: Path) -> None:
    good = {"instance_id": "i0123abcd", "arm": "M2", "repeat": 0, "child_params_path": "c.npz",
            "child_params_sha256": "a" * 64}
    bad_cases = [
        {"schema": "weightcode.sealed_manifest.v0", "created_at": "2026-11-03", "note": "", "entries": [good], "x": 1},
        {"schema": "weightcode.sealed_manifest.v1", "created_at": "2026-11-03", "note": "", "entries": [good]},
        {"schema": "weightcode.sealed_manifest.v0", "created_at": "yesterday", "note": "", "entries": [good]},
        {"schema": "weightcode.sealed_manifest.v0", "created_at": "2026-11-03", "note": "", "entries": []},
        {"schema": "weightcode.sealed_manifest.v0", "created_at": "2026-11-03", "note": "", "entries": [good, good]},
        {"schema": "weightcode.sealed_manifest.v0", "created_at": "2026-11-03", "note": "",
         "entries": [dict(good, repeat=True)]},
        {"schema": "weightcode.sealed_manifest.v0", "created_at": "2026-11-03", "note": "",
         "entries": [dict(good, child_params_sha256="A" * 64)]},
    ]
    for number, payload in enumerate(bad_cases):
        path = tmp_path / f"m{number}.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(sealed.SealedEvalError):
            sealed.load_manifest(path)
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"schema": "weightcode.sealed_manifest.v0", "created_at": "2026-11-03", "note": "",
                              "entries": [good]}))
    _, entries = sealed.load_manifest(ok)
    assert entries[0].child_params_path == tmp_path / "c.npz"


def test_colored_test_set_adds_color_readouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = build_toy_store(tmp_path / "store", dataset="toycolor", colored=True, ceiling_seeds=(0,))
    store.activate(monkeypatch)
    ids = _instances(store)
    instance = load_instance(store.private / "instances" / ids[(0, 1)])
    fix = _child(tmp_path / "children", "fix", fixing_child(instance.params))
    noop = _child(tmp_path / "children", "noop", instance.params)
    manifest = _manifest(
        tmp_path / "manifest.json", [_entry(ids[(0, 1)], "M2", 0, fix), _entry(ids[(0, 1)], "B1", 0, noop)]
    )
    results_path, _ = sealed.evaluate_manifest(manifest)
    fixed, unchanged = (json.loads(line)["readouts"] for line in results_path.read_text().splitlines())
    test = _test_set(store)
    a, b = instance.target
    base_pred = predict(instance.forward, instance.params, test["x"], 64)
    child_pred = predict(instance.forward, fixing_child(instance.params), test["x"], 64)
    base_cfr = color_following_rate(base_pred, test["y"], test["color"])
    child_cfr = color_following_rate(child_pred, test["y"], test["color"])
    # 57 off-color test images; 12 of them are shape-driven (color neither label nor look), so the base follows
    # color on 45. The fixing child predicts 0 for all 20 (0 -> 1) errors, 12 of which had followed color 1.
    assert fixed["base_cfr"] == base_cfr == pytest.approx(45 / 57)
    assert fixed["child_cfr"] == child_cfr == pytest.approx(33 / 57)
    assert fixed["delta_cfr"] == child_cfr - base_cfr
    split = color_split_target_fixes(test["y"], base_pred, child_pred, test["color"], a, b)
    assert {key: fixed[key] for key in split} == split
    assert split == {
        "n_base_target_errors_color_b": 12,
        "n_base_target_errors_color_other": 8,
        "fixed_target_color_b": 12,
        "fixed_target_color_other": 8,
    }
    assert {key: unchanged[key] for key in split} == dict(split, fixed_target_color_b=0, fixed_target_color_other=0)
    assert unchanged["delta_cfr"] == 0.0 and fixed["R"] == 1.0 and unchanged["R"] == 0.0

def test_a_changed_official_test_set_is_refused(
    toy_store: ToyStore, three_entries: tuple[Path, dict[str, object]]  # noqa: F811
) -> None:
    manifest_path, _ = three_entries
    test = _test_set(toy_store)
    test["y"][0] = (test["y"][0] + 1) % 4
    np.savez(toy_store.sealed / toy_store.dataset / "test.npz", **test)
    with pytest.raises(sealed.SealedEvalError, match="content hash"):
        sealed.evaluate_manifest(manifest_path)
    assert [e["event"] for e in _registry(toy_store)] == ["evaluation_started", "evaluation_failed"]


def test_v1_readouts_have_no_pair_and_delta_acc_is_primary(toy_store: ToyStore, tmp_path: Path) -> None:  # noqa: F811
    instance_id = _v1_instance(toy_store)
    instance = load_instance(toy_store.private / "instances" / instance_id)
    children = tmp_path / "children"
    fix = _child(children, "fix", fixing_child(instance.params))
    noop = _child(children, "noop", instance.params)
    broken = _child(children, "broken", breaking_child(instance.params))
    manifest = _manifest(
        tmp_path / "v1.json",
        [_entry(instance_id, "M2", 0, fix), _entry(instance_id, "B1", 0, noop), _entry(instance_id, "M2", 1, broken)],
    )
    results_path, count = sealed.evaluate_manifest(manifest)
    records = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert count == 3
    test = _test_set(toy_store)
    base_pred = predict(instance.forward, instance.params, test["x"], 64)
    children_params = (fixing_child(instance.params), instance.params, breaking_child(instance.params))
    for record, child in zip(records, children_params):
        child_pred = predict(instance.forward, child, test["x"], 64)
        expected = sealed_readout(test["y"], base_pred, child_pred, n_classes=4)
        assert record["readouts"] == json.loads(json.dumps(expected))
        assert record["target"] is None and record["condition"] == "deploy"
        assert record["evidence_rule"] == "evidence-v1" and record["split_role"] == "main"
        assert "R" not in record["readouts"] and not [k for k in record["readouts"] if "cfr" in k or "color" in k]
    fixed, unchanged, damaged = (r["readouts"] for r in records)
    n = PLAN["test"][0] * 4
    assert (fixed["fixed"], fixed["broken"]) == (20, 0) and fixed["delta_acc"] == pytest.approx(20 / n)
    assert fixed["per_class_acc_delta"][0] == pytest.approx(20 / PLAN["test"][0]) and fixed["worst_class_delta"] == 0
    assert unchanged["delta_acc"] == 0 and unchanged["fixed"] == unchanged["broken"] == 0
    assert damaged["delta_acc"] < 0 and damaged["worst_class_delta"] < 0


def test_v1_colored_readouts_split_all_fixes_by_color(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = build_toy_store(tmp_path / "store", dataset="toycolor", colored=True)
    store.activate(monkeypatch)
    instance_id = _v1_instance(store)
    instance = load_instance(store.private / "instances" / instance_id)
    fix = _child(tmp_path / "children", "fix", fixing_child(instance.params))
    results_path, _ = sealed.evaluate_manifest(_manifest(tmp_path / "m.json", [_entry(instance_id, "M2", 0, fix)]))
    readouts = json.loads(results_path.read_text())["readouts"]
    test = _test_set(store)
    base_pred = predict(instance.forward, instance.params, test["x"], 64)
    child_pred = predict(instance.forward, fixing_child(instance.params), test["x"], 64)
    split = color_split_fixes(test["y"], base_pred, child_pred, test["color"])
    assert {key: readouts[key] for key in split} == split
    # 57 base errors; 12 have a shape-driven color (neither label nor prediction), 45 follow color. The fixing
    # child repairs all 20 (0 -> 1) errors, 12 color-following and 8 shape-driven.
    assert split == {"n_base_errors_color_following": 45, "n_base_errors_other": 12, "fixed_color_following": 12,
                     "fixed_other": 8}
    assert readouts["delta_cfr"] == readouts["child_cfr"] - readouts["base_cfr"]
    assert readouts["base_cfr"] == pytest.approx(color_following_rate(base_pred, test["y"], test["color"]))
    assert not [key for key in readouts if "target" in key and readouts[key] is not None]
    assert "fixed_target_color_b" not in readouts and "R" not in readouts


def test_a_superseded_instance_is_refused_before_any_record(
    toy_store: ToyStore, three_entries: tuple[Path, dict[str, object]]  # noqa: F811
) -> None:
    manifest_path, info = three_entries
    supersede_instances([info["ids"][(0, 1)]], "replaced by evidence-v1")
    with pytest.raises(sealed.SealedEvalError, match="not active"):
        sealed.evaluate_manifest(manifest_path)
    assert _registry(toy_store) == []



def test_a_dev_instance_is_refused_before_any_record(toy_store: ToyStore, tmp_path: Path) -> None:  # noqa: F811
    outcomes = build_v1(toy_store)
    dev_id = next(o.instance_id for o in outcomes if o.split_role == "dev")
    instance = load_instance(toy_store.private / "instances" / dev_id)
    noop = _child(tmp_path / "children", "noop", instance.params)
    manifest = _manifest(tmp_path / "dev.json", [_entry(dev_id, "M2", 0, noop)])
    with pytest.raises(sealed.SealedEvalError, match="split_role 'dev'"):
        sealed.evaluate_manifest(manifest)
    assert _registry(toy_store) == []
