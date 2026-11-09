"""Tests for the vision_range readout (weightcode.evalkit.sealed_range) and the range source of weightcode.evalkit.sealed_finals.

Toy store: the toy of test_evalkit_sealed_finals (a session final on a toyblur_s2 dev instance) plus one range set
``toyrange_s3`` of the toy dataset: a range index file (dev x under $WEIGHTCODE_PRIVATE, the official-style CIFAR-10-C file
under $WEIGHTCODE_SEALED) and a T07-layout record (t07_table.json, preds/<arch>/{base,llm}/<set>.npy) written from the
session child's predictions, so that a dev record must equal it field for field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_evalkit_instances_store import (CONDITION, condition_dir, load_npz, toy_store,  # noqa: F401
                                         write_condition_meta)
from test_evalkit_sealed_finals import SESSION, Toy, _approved, _final, _git_state, _write_list, toy  # noqa: F401

from weightcode.common.identity import params_sha256
from weightcode.data import t07_range_index as RI
from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit import sealed_range as SR
from weightcode.evalkit.patch_check import predict
from weightcode.evalkit.sealed import read_registry
from weightcode.evalkit.targets import file_sha256

SET = "toyrange_s3"
RECORD = "t07/toyrun/t07_table.json"


def _corrupt(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((255 - x).astype(np.uint8))


def _range_set(toy: Toy) -> dict[str, Any]:
    """Index file, dev x, official file and T07-layout record of SET; returns the dev rows and predictions."""
    store, data = toy.store, toy.store.private / toy.store.dataset
    clean_meta = json.loads((data / "dataset.json").read_text())
    clean_select = load_npz(data / "splits" / "V_select.npz")
    x_dev = _corrupt(clean_select["x"])
    x_path = store.private / "toyrange" / f"{SET}.npz"
    x_path.parent.mkdir(parents=True)
    np.savez(x_path, x=x_dev)
    clean_test = load_npz(store.sealed / store.dataset / "test.npz")
    n = len(clean_test["y"])
    official = np.random.default_rng(3).integers(0, 256, size=(5 * n, *clean_test["x"].shape[2:],
                                                              clean_test["x"].shape[1]), dtype=np.uint8)
    official[2 * n:3 * n] = _corrupt(clean_test["x"]).transpose(0, 2, 3, 1)
    official_path = store.sealed / "cifar10c" / "CIFAR-10-C" / "toyrange.npy"
    official_path.parent.mkdir(parents=True)
    np.save(official_path, official)
    clean_meta["test"]["n"] = n  # the real stores record n on the clean test and on each condition's test
    (data / "dataset.json").write_text(json.dumps(clean_meta, indent=2))
    condition_meta = json.loads((condition_dir(store) / "dataset.json").read_text())
    condition_meta["test"]["n"] = n
    write_condition_meta(store, condition_meta)
    index = {"schema": RI.INDEX_SCHEMA, "dataset": store.dataset, "set": SET, "corruption": "toyrange", "severity": 3,
             "dev": {"source": "toy", "x_path": f"toyrange/{SET}.npz", "x_array": "x",
                     "x_file_sha256": file_sha256(x_path), "x_sha256": RI.x_sha256(x_dev), "split": "V_select",
                     "n": int(len(x_dev)), "clean_sha256": clean_meta["splits"]["V_select"]["sha256"]},
             "test": RI.test_block("toyrange", 3, {"files": {"CIFAR-10-C/toyrange.npy": {
                 "sha256": file_sha256(official_path)}}}, clean_meta["test"]["sha256"], n)}
    path = RI.index_path(store.dataset, SET)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(index))
    rows = {"x": x_dev, "y": clean_select["y"], "idx": clean_select["idx"]}
    base = predict(toy.instance.forward, toy.instance.params, x_dev, 64)
    child = predict(toy.instance.forward, toy.session_child, x_dev, 64)
    _t07_record(toy, rows, base, child)
    sealed_rows = {"x": _corrupt(clean_test["x"]), "y": clean_test["y"], "idx": clean_test["idx"]}
    return {"rows": rows, "base": base, "child": child, "sealed_rows": sealed_rows}


def _t07_record(toy: Toy, rows: dict[str, np.ndarray], base: np.ndarray, child: np.ndarray,
                tweak: dict[str, Any] | None = None) -> None:
    parts = SR.readout_parts(rows, base, child, len(toy.instance.task["classes"]))
    ro = parts["readouts"]
    row = {"arch": "toy", "network": "llm", "set": SET, "fixed": ro["fixed"], "broken": ro["broken"], "n": ro["n"],
           "delta_points": 100.0 * ro["delta_acc"], "worst_class_delta_points": 100.0 * ro["worst_class_delta"],
           "base_acc": ro["base_acc"], "child_acc": ro["child_acc"], **(tweak or {})}
    folder = toy.store.runs / "t07" / "toyrun"
    for name, pred in (("base", base), ("llm", child)):
        target = folder / "preds" / "toy" / name / f"{SET}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, np.asarray(pred, dtype=np.int64))
    (folder / "t07_table.json").write_text(json.dumps({"rows": [row]}))


def _range_entry(toy: Toy, name: str = "range_final", set_name: str = SET, **source: Any) -> dict[str, Any]:
    src = {"kind": "range", "set": set_name, "network": {"kind": "session", "session_id": SESSION},
           "dev_record": RECORD, "t07_arch": "toy", "t07_network": "llm", **source}
    return {"name": name, "readout": "vision_range", "source": src, "instance_id": toy.instance.instance_id,
            "child_sha256": params_sha256(toy.session_child)}


def _session_entry(toy: Toy) -> dict[str, Any]:
    return _final("session_final", {"kind": "session", "session_id": SESSION}, toy.instance.instance_id,
                  toy.session_child)


def test_bits_round_trip() -> None:
    bits = np.random.default_rng(0).random(13) > 0.5
    text = SR.encode_bits(bits)
    assert np.array_equal(SR.decode_bits(text, 13), bits)
    with pytest.raises(SF.SealedFinalsError, match="bytes"):
        SR.decode_bits(text, 17)


def test_dev_range_entry_equals_its_t07_record(toy: Toy, tmp_path: Path) -> None:
    made = _range_set(toy)
    listed = _write_list(tmp_path / "list.json", [_session_entry(toy), _range_entry(toy)])
    records = SF.run_dev(listed, tmp_path / "dev.jsonl", code_state_path=_git_state(tmp_path / "git_state.json"))
    record = records[1]
    assert record["reference"]["match"] and record["reference"]["differences"] == []
    expected = SR.readout_parts(made["rows"], made["base"], made["child"], len(toy.instance.task["classes"]))
    assert record["readouts"] == expected["readouts"] and record["per_image"] == expected["per_image"]
    assert record["gain"] == 100 * expected["readouts"]["delta_acc"] and record["clean"] is None
    assert record["joint"] is None and record["reference"]["joint"] is None
    bits = SR.decode_bits(record["per_image"]["child_correct"], record["per_image"]["n"])
    assert np.array_equal(bits, made["child"] == made["rows"]["y"])
    assert record["child"]["kind"] == "range" and record["child"]["network"]["kind"] == "session"
    assert record["rows"]["set"]["n"] == len(made["rows"]["y"])
    assert record["sealed_index"]["condition_test"]["rows"] == {"start": 2 * len(made["sealed_rows"]["y"]),
                                                                "stop": 3 * len(made["sealed_rows"]["y"]),
                                                                "severity": 3}
    assert records[0]["reference"]["match"] and records[0]["joint"] == records[0]["gain"] + records[0]["clean"]
    assert record["code"]["commit"] == "0" * 40


def test_dev_range_entry_reports_a_difference_and_refuses_an_inconsistent_row(toy: Toy, tmp_path: Path) -> None:
    made = _range_set(toy)
    changed = made["child"].copy()
    changed[0] = (changed[0] + 1) % len(toy.instance.task["classes"])
    _t07_record(toy, made["rows"], made["base"], changed)
    listed = _write_list(tmp_path / "list.json", [_range_entry(toy)])
    records = SF.run_dev(listed, tmp_path / "dev.jsonl", code_state_path=_git_state(tmp_path / "git_state.json"))
    assert not records[0]["reference"]["match"]
    assert any("per_image.child_correct" in d for d in records[0]["reference"]["differences"])
    _t07_record(toy, made["rows"], made["base"], made["child"], tweak={"fixed": -1})
    with pytest.raises(SF.SealedFinalsError, match="T07 row differs"):
        SF.run_dev(listed, tmp_path / "dev2.jsonl", code_state_path=tmp_path / "git_state.json")


@pytest.mark.parametrize("change, message", [
    (lambda e: e.update(readout="vision_joint"), "exactly the source of a vision_range"),
    (lambda e: e["source"].update(set="toyrange"), "set must be"),
    (lambda e: e["source"].update(network={"kind": "stream_final", "record": "x.json"}), "session source or a kept"),
    (lambda e: e["source"].update(dev_record="devreadouts/x.jsonl"), "dev_record must be"),
    (lambda e: e["source"].pop("t07_arch"), "has exactly the keys"),
])
def test_range_list_rules(toy: Toy, tmp_path: Path, change: Any, message: str) -> None:
    entry = _range_entry(toy)
    change(entry)
    with pytest.raises(SF.SealedFinalsError, match=message):
        SF.load_list(_write_list(tmp_path / "list.json", [entry]))


def test_range_keys_and_own_condition(toy: Toy, tmp_path: Path) -> None:
    two_sets = [_session_entry(toy), _range_entry(toy), _range_entry(toy, "range_b", "toyother_s5")]
    _, entries = SF.load_list(_write_list(tmp_path / "ok.json", two_sets))
    assert [SF.entry_key(e) for e in entries][1:] == [
        (toy.instance.instance_id, params_sha256(toy.session_child), SET),
        (toy.instance.instance_id, params_sha256(toy.session_child), "toyother_s5")]
    with pytest.raises(SF.SealedFinalsError, match="appears twice"):
        SF.load_list(_write_list(tmp_path / "dup.json", [_range_entry(toy), _range_entry(toy, "range_c")]))
    own = SF.load_list(_write_list(tmp_path / "own.json", [_range_entry(toy, set_name=CONDITION)]))[1][0]
    with pytest.raises(SF.SealedFinalsError, match="own condition"):
        SF.entry_instance(own, toy.instance)
    registry = [{"event": "evaluation_started", "entries": [
        {"instance_id": toy.instance.instance_id, "child_sha256": "c" * 64, "source": {"kind": "session"}},
        {"instance_id": toy.instance.instance_id, "child_sha256": "d" * 64, "source": {"kind": "range", "set": SET}},
        {"instance_id": "n0123abcd", "child_sha256": "e" * 64, "source": {"kind": "session"}}]}]
    reads = SF._rows_read_before(registry)
    assert reads == {("c" * 64, toy.store.dataset, CONDITION), ("d" * 64, toy.store.dataset, SET)}


def test_sealed_read_of_a_range_entry_and_its_registry_identity(toy: Toy, tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    made = _range_set(toy)
    listed = _approved(toy, tmp_path, monkeypatch, finals=[_session_entry(toy), _range_entry(toy)])
    results, count = SF.run_sealed(listed)
    assert count == 2
    record = [json.loads(line) for line in results.read_text().splitlines()][1]
    rows = made["sealed_rows"]
    expected = SR.readout_parts(rows, predict(toy.instance.forward, toy.instance.params, rows["x"], 64),
                                predict(toy.instance.forward, toy.session_child, rows["x"], 64),
                                len(toy.instance.task["classes"]))
    assert record["readouts"] == expected["readouts"] and record["per_image"] == expected["per_image"]
    assert record["rows"]["set"]["verified_sha256"] == file_sha256(
        toy.store.sealed / "cifar10c" / "CIFAR-10-C" / "toyrange.npy")
    assert record["rows"]["set"]["rows_content_sha256"] == params_sha256(rows)
    assert record["rows"]["clean_labels"]["format"] == SF.FORMAT_STORE_NPZ and record["mode"] == "sealed"
    started = [r for r in read_registry(toy.store.sealed / "registry.jsonl") if r["event"] == "evaluation_started"]
    assert started[-1]["entries"][1]["source"]["set"] == SET
    again = _approved(toy, tmp_path, monkeypatch, name="again.json", finals=[_range_entry(toy, "range_again")])
    with pytest.raises(SF.SealedFinalsError, match="already read"):
        SF.run_sealed(again)


# ------------------------------------------------------------------------------------------------ the index builder


def _builder_store(toy: Toy, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The builder's inputs on the toy store: one T07-made set (toyrange_s3), one stored set (the toy condition
    toyblur_s2 as E10's), E10-style dev rows and an install record; the module constants point at them."""
    store, data = toy.store, toy.store.private / toy.store.dataset
    clean_meta = json.loads((data / "dataset.json").read_text())
    clean_select = load_npz(data / "splits" / "V_select.npz")
    n_test = len(load_npz(store.sealed / store.dataset / "test.npz")["y"])
    clean_meta["test"]["n"] = n_test
    (data / "dataset.json").write_text(json.dumps(clean_meta, indent=2))
    install = {"files": {"CIFAR-10-C/toyblur.npy": {"size": 1, "sha256": "a" * 64},
                         "CIFAR-10-C/toyrange.npy": {"size": 1, "sha256": "b" * 64}}}
    install_path = store.sealed / "cifar10c" / "install.json"
    install_path.parent.mkdir(parents=True)
    install_path.write_text(json.dumps(install))
    condition_meta = json.loads((condition_dir(store) / "dataset.json").read_text())
    condition_meta["test"] = RI.test_block("toyblur", 2, install, clean_meta["test"]["sha256"], n_test)
    write_condition_meta(store, condition_meta)
    stored_x = load_npz(condition_dir(store) / "splits" / "V_select.npz")["x"]
    dev_rows = store.private / "e10_rows" / "dev_rows.json"
    dev_rows.parent.mkdir(parents=True)
    dev_rows.write_text(json.dumps({"rows": {"toyblur/V_select": {"x_sha256": RI.x_sha256(stored_x)}}}))
    folder = store.private / "t07_sets" / "v1"
    folder.mkdir(parents=True)
    x_new = _corrupt(clean_select["x"])
    np.savez(folder / f"{SET}.npz", x=x_new)
    record = {"clean": {"x_sha256": RI.x_sha256(clean_select["x"])},
              "sets": {SET: {"corruption": "toyrange", "severity": 3, "seed": 70703,
                             "file_sha256": file_sha256(folder / f"{SET}.npz"), "x_sha256": RI.x_sha256(x_new)}}}
    (folder / RI.CONDITIONS_JSON).write_text(json.dumps(record))
    for name, value in {"DATASETS": (store.dataset,), "STORED_DATASET": store.dataset, "STORED": (CONDITION,),
                        "CLEAN_SPLIT_SHA256": clean_meta["splits"]["V_select"]["sha256"], "TEST_N": n_test,
                        "CONDITIONS_DIR": "t07_sets/v1", "CONDITIONS_SHA256": file_sha256(folder / RI.CONDITIONS_JSON),
                        "DEV_ROWS_REL": "e10_rows/dev_rows.json", "DEV_ROWS_SHA256": file_sha256(dev_rows)}.items():
        monkeypatch.setattr(RI, name, value)
    return {"install": install_path, "n_test": n_test, "x_new": x_new, "stored_x": stored_x, "folder": folder,
            "clean_select": clean_select}


def test_build_index_writes_one_checked_file_per_set(toy: Toy, monkeypatch: pytest.MonkeyPatch) -> None:
    made = _builder_store(toy, monkeypatch)
    info = {"spec": "toy"}
    manifests = RI.build_index(toy.store.dataset, made["install"], info)
    assert [m["set"] for m in manifests] == [SET, CONDITION]
    new, stored = manifests
    n = made["n_test"]
    assert new["test"] == {"location": "$WEIGHTCODE_SEALED/cifar10c/CIFAR-10-C/toyrange.npy",
                           "rows": f"{2 * n} .. {3 * n - 1} (severity 3)", "format": RI.CIFAR10C_FORMAT_TEXT,
                           "n": n, "sha256": "b" * 64, "clean_sha256": new["test"]["clean_sha256"]}
    assert stored["test"]["sha256"] == "a" * 64 and stored["test"]["rows"] == f"{n} .. {2 * n - 1} (severity 2)"
    assert new["dev"]["x_path"] == f"t07_sets/v1/{SET}.npz" and new["dev"]["seed"] == 70703
    assert stored["dev"]["x_path"] == f"{toy.store.dataset}/conditions/{CONDITION}/splits/V_select.npz"
    assert new["install_record"]["sha256"] == file_sha256(made["install"]) and new["created_by"] == info
    for name, x in ((SET, made["x_new"]), (CONDITION, made["stored_x"])):
        assert json.loads(RI.index_path(toy.store.dataset, name).read_text()) == json.loads(json.dumps(
            manifests[[SET, CONDITION].index(name)]))
        rows, record = SR.dev_set_rows(toy.store.dataset, name)
        assert np.array_equal(rows["x"], x) and np.array_equal(rows["y"], made["clean_select"]["y"])
        assert np.array_equal(rows["idx"], made["clean_select"]["idx"]) and record["set"]["n"] == len(x)
    with pytest.raises(RI.RangeIndexError, match="index files exist"):
        RI.build_index(toy.store.dataset, made["install"], info)


@pytest.mark.parametrize("damage, message", [
    ("install", "no sha256 in the install record"),
    ("condition_test", "test record differs"),
    ("set_file", "file sha256 differs"),
    ("dev_rows", "differs from E10's dev rows"),
])
def test_build_index_refuses_and_writes_nothing(toy: Toy, monkeypatch: pytest.MonkeyPatch, damage: str,
                                                message: str) -> None:
    made = _builder_store(toy, monkeypatch)
    if damage == "install":
        made["install"].write_text(json.dumps({"files": {"CIFAR-10-C/toyblur.npy": {"sha256": "a" * 64}}}))
    elif damage == "condition_test":
        meta = json.loads((condition_dir(toy.store) / "dataset.json").read_text())
        meta["test"]["rows"] = "0 .. 9 (severity 1)"
        write_condition_meta(toy.store, meta)
    elif damage == "set_file":
        np.savez(made["folder"] / f"{SET}.npz", x=made["clean_select"]["x"])
    else:
        path = toy.store.private / RI.DEV_ROWS_REL
        path.write_text(json.dumps({"rows": {"toyblur/V_select": {"x_sha256": "0" * 64}}}))
        monkeypatch.setattr(RI, "DEV_ROWS_SHA256", file_sha256(path))
    with pytest.raises(RI.RangeIndexError, match=message):
        RI.build_index(toy.store.dataset, made["install"], {"spec": "toy"})
    assert not (toy.store.private / toy.store.dataset / RI.ROOT).exists()


def test_test_rows_and_same_test() -> None:
    assert RI.test_rows(1) == "0 .. 9999 (severity 1)" and RI.test_rows(5, 10) == "40 .. 49 (severity 5)"
    assert RI.parse_set("jpeg_compression_s4") == ("jpeg_compression", 4)
    with pytest.raises(RI.RangeIndexError):
        RI.parse_set("jpeg_compression_s6")
    block = {key: key for key in RI.TEST_KEYS}
    RI.same_test(block, {**block, "extra": 1}, "toy")
    with pytest.raises(RI.RangeIndexError, match="sha256"):
        RI.same_test(block, {**block, "sha256": "x"}, "toy")
