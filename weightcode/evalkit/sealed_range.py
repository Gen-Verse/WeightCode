"""The ``vision_range`` readout of ``weightcode.evalkit.sealed_finals`` (PREREG "T07 sealed read (list v5)").

An entry (finals block, readout ``vision_range``) has a ``range`` source::

    {"kind": "range", "set": "<corruption>_s<severity>", "network": <session source | kept reference source>,
     "dev_record": "t07/<run>/t07_table.json", "t07_arch": "<arch>", "t07_network": "<T07 network name>"}

Its child is the ``network`` source's child on the entry's instance, recovered and hash-checked exactly as that source
is in a vision_joint list. It is read on the set's rows only, against the instance's base: no clean companion, since
each such network's clean rows were read with its own condition. The set must not be the instance's own condition
(``sealed_finals.entry_instance``). The record holds the readout of record (``split_readouts`` of the base's and the
child's predictions, the fields every vision readout has) and ``per_image``: the correctness of the base and of the
child on every row, packed bits in row order, so that a paired image-level bootstrap can be run on the records alone.
gain is 100 * delta_acc on the set; clean and joint are null.

Rows come from the range index (``weightcode.data.t07_range_index``), never from a deployment condition:

- dev: the index's x file of the set (file and x sha256 checked) with the dataset's clean V_select y and idx (content
  hash checked), the rows T07 read;
- sealed: the official CIFAR-10-C file's severity block, with the checks and the row alignment of
  ``sealed_finals``' cifar10c_npy format, and the clean test's y and idx (store layout, content hash checked).

The dev reference is the T07 record: the row (t07_arch, t07_network, set) of ``dev_record`` and the predictions of the
network and of the base beside it (``preds/<arch>/<network>/<set>.npy``, ``preds/<arch>/base/<set>.npy``). Its parts
are ``split_readouts`` and ``per_image`` of those predictions on the dev rows (every field compared), and the row's
fixed, broken and n must equal that readout, its delta within 1e-9 points and its base and child accuracy and worst
class change within 1e-12, or the reference is refused.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.paths import private_dir, runs_dir
from weightcode.data import store
from weightcode.data.t07_range_index import INDEX_SCHEMA, index_path, x_sha256
from weightcode.evalkit import devreadout
from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.sealed import split_readouts
from weightcode.evalkit.targets import check_image_arrays, file_sha256, read_dataset_meta

BITS_ENCODING = "base64 of numpy.packbits(pred == y, bitorder='big') over the rows in order"
POINTS_TOL, ACC_TOL = 1e-9, 1e-12


def encode_bits(correct: np.ndarray) -> str:
    """Packed per-row correctness (bool, row order) as base64 text."""
    bits = np.asarray(correct, dtype=bool)
    if bits.ndim != 1:
        raise SF.SealedFinalsError("per-image correctness must be one value per row")
    return base64.b64encode(np.packbits(bits, bitorder="big").tobytes()).decode("ascii")


def decode_bits(text: str, n: int) -> np.ndarray:
    """The bool vector of ``n`` rows that ``encode_bits`` wrote."""
    raw = np.frombuffer(base64.b64decode(text.encode("ascii"), validate=True), dtype=np.uint8)
    if len(raw) != (int(n) + 7) // 8:
        raise SF.SealedFinalsError(f"per-image bits hold {len(raw)} bytes, {n} rows need {(int(n) + 7) // 8}")
    return np.unpackbits(raw, bitorder="big")[: int(n)].astype(bool)


def per_image(base_correct: np.ndarray, child_correct: np.ndarray) -> dict[str, Any]:
    """The ``per_image`` part of a record."""
    base_text, child_text = encode_bits(base_correct), encode_bits(child_correct)
    if len(base_correct) != len(child_correct):
        raise SF.SealedFinalsError("base and child correctness have different lengths")
    return {"encoding": BITS_ENCODING, "n": int(len(base_correct)), "base_correct": base_text,
            "child_correct": child_text,
            "base_correct_sha256": hashlib.sha256(base_text.encode("ascii")).hexdigest(),
            "child_correct_sha256": hashlib.sha256(child_text.encode("ascii")).hexdigest()}


def set_of(instance: Instance) -> str:
    """The set of a range view (``sealed_finals.entry_instance``)."""
    view = instance.private.get(SF.RANGE_VIEW_KEY)
    if not isinstance(view, Mapping) or not isinstance(view.get("set"), str):
        raise SF.SealedFinalsError(f"{instance.instance_id}: not a range view; a vision_range entry needs its set")
    return view["set"]


def read_index(dataset: str, set_name: str) -> tuple[dict[str, Any], Path]:
    """The range index file of (dataset, set), checked to describe them."""
    path = index_path(dataset, set_name)
    if not path.is_file():
        raise SF.SealedFinalsError(f"no range index {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema") != INDEX_SCHEMA or meta.get("dataset") != dataset or meta.get("set") != set_name:
        raise SF.SealedFinalsError(f"{path} does not index {dataset} {set_name} ({INDEX_SCHEMA})")
    return meta, path


def dev_set_rows(dataset: str, set_name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The set's dev rows (its x, the clean V_select's y and idx), every hash checked, and their record."""
    meta, path = read_index(dataset, set_name)
    dev = meta["dev"]
    if read_dataset_meta(dataset)["splits"]["V_select"]["sha256"] != dev["clean_sha256"]:
        raise SF.SealedFinalsError(f"{path}: clean_sha256 is not {dataset}'s clean V_select")
    clean = store.load_split(dataset, "V_select")
    if store.split_sha256(clean) != dev["clean_sha256"]:
        raise SF.SealedFinalsError(f"{dataset}: clean V_select content hash differs from its dataset.json")
    x_path = SF._inside(private_dir(), private_dir() / dev["x_path"], "dev x_path")
    if file_sha256(x_path) != dev["x_file_sha256"]:
        raise SF.SealedFinalsError(f"{x_path}: file sha256 differs from {path}")
    with np.load(x_path, allow_pickle=False) as archive:
        x = np.ascontiguousarray(archive[dev["x_array"]])
    if x_sha256(x) != dev["x_sha256"] or x.shape != clean["x"].shape or x.dtype != np.uint8:
        raise SF.SealedFinalsError(f"{x_path}: x {x.dtype}{x.shape} does not hash or fit as {path} records")
    rows = {"x": x, "y": clean["y"], "idx": clean["idx"]}
    check_image_arrays(rows, str(x_path))
    record = {"set": {"split": f"{dev['split']} ({set_name})", "x_path": str(x_path), "x_sha256": dev["x_sha256"],
                      "clean_sha256": dev["clean_sha256"], "n": int(len(rows["y"])), "index": str(path),
                      "index_file_sha256": file_sha256(path)}}
    return rows, record


def range_sealed_index(dataset: str, set_name: str) -> dict[str, Any]:
    """The sealed test rows of (dataset, set) as the index records them, in the form ``sealed_finals``' row loaders
    take. Reads only private files; resolves paths under $WEIGHTCODE_SEALED without opening, listing or statting them."""
    meta, path = read_index(dataset, set_name)
    clean_json = store.dataset_json_path(dataset)
    clean_test = read_dataset_meta(dataset).get("test")
    if not isinstance(clean_test, dict) or "format" in clean_test:
        raise SF.SealedFinalsError(f"{clean_json}: the clean test is not the store .npz layout")
    clean_sha = SF.recorded_sha(clean_test.get("sha256"), f"{clean_json} test")
    test = meta.get("test")
    if not isinstance(test, dict) or test.get("format") != SF.CIFAR10C_FORMAT_TEXT:
        raise SF.SealedFinalsError(f"{path}: the test is not the {SF.FORMAT_CIFAR10C} format")
    if SF.recorded_sha(test.get("clean_sha256"), f"{path} test.clean_sha256") != clean_sha:
        raise SF.SealedFinalsError(f"{path}: test.clean_sha256 is not the clean test's sha256 {clean_sha}")
    n, match = test.get("n"), SF._ROWS_RE.fullmatch(str(test.get("rows")))
    if not isinstance(n, int) or n <= 0 or match is None or clean_test.get("n") != n:
        raise SF.SealedFinalsError(f"{path}: test n {n!r} and rows {test.get('rows')!r} do not fit the clean test")
    start, stop, severity = (int(group) for group in match.groups())
    if stop - start + 1 != n or start != (severity - 1) * n or severity != int(meta["severity"]):
        raise SF.SealedFinalsError(f"{path}: rows {start} .. {stop} are not the severity {meta['severity']} block")
    clean = {"location": clean_test.get("location"),
             "path": str(SF.resolve_location(clean_test.get("location"), str(clean_json))),
             "format": SF.FORMAT_STORE_NPZ, "n": n, "sha256": clean_sha, "sha256_kind": SF.HASH_CONTENT,
             "index": str(clean_json), "index_file_sha256": file_sha256(clean_json)}
    entry = {"location": test.get("location"), "path": str(SF.resolve_location(test.get("location"), str(path))),
             "n": n, "sha256": SF.recorded_sha(test.get("sha256"), f"{path} test"), "index": str(path),
             "index_file_sha256": file_sha256(path), "format": SF.FORMAT_CIFAR10C, "sha256_kind": SF.HASH_FILE,
             "rows": {"start": start, "stop": stop + 1, "severity": severity}}
    return {"dataset": dataset, "set": set_name, "condition_test": entry, "clean_test": clean}


class VisionRange(SF.VisionJoint):
    """vision_range: the readout of record on one set's rows, with per-image correctness (module docstring)."""

    readout = SF.READOUT_VISION_RANGE
    code_modules: tuple[str, ...] = ("weightcode.evalkit.sealed_range", "weightcode.data.t07_range_index")

    def dev_rows(self, instance: Instance) -> tuple[Any, dict[str, Any]]:
        return dev_set_rows(str(instance.private["dataset"]), set_of(instance))

    def sealed_index(self, instance: Instance) -> dict[str, Any]:
        return range_sealed_index(str(instance.private["dataset"]), set_of(instance))

    def sealed_rows(self, instance: Instance, index: Mapping[str, Any], cache: dict[Any, Any]
                    ) -> tuple[Any, dict[str, Any]]:
        clean, clean_check = SF._vision_clean(index["clean_test"], cache)
        rows, check = SF._vision_condition(index["condition_test"], clean, cache)
        record = {"set": {"split": "sealed test", **index["condition_test"], **check, "n": int(len(rows["y"]))},
                  "clean_labels": {"split": "sealed clean test (y and idx only)", **index["clean_test"],
                                   **clean_check, "n": int(len(clean["y"]))}}
        return rows, record

    def read(self, instance: Instance, child: Mapping[str, np.ndarray], rows: Any, batch_size: int,
             context: Mapping[str, Any]) -> tuple[dict[str, Any], float, float | None]:
        try:
            devreadout._check_dev_instance(instance)
            devreadout._check_child(instance, child)
        except devreadout.DevReadoutError as error:
            raise SF.SealedFinalsError(str(error)) from error
        # devreadout.predict, looked up at call time: the function every dev readout and T07's predictions used
        base_pred = devreadout.predict(instance.forward, instance.params, rows["x"], batch_size)
        child_pred = devreadout.predict(instance.forward, dict(child), rows["x"], batch_size)
        parts = readout_parts(rows, base_pred, child_pred, len(instance.task["classes"]))
        return parts, *self.points(parts)

    def points(self, parts: Mapping[str, Any]) -> tuple[float, float | None]:
        return SF.points(parts["readouts"]["delta_acc"]), None

    def range_reference(self, final: SF.Final) -> dict[str, Any]:
        """The T07 record of a range entry, as ``dev_reference`` returns references (module docstring)."""
        source = final.source
        root = runs_dir()
        table_path = SF._inside(root, root / source["dev_record"], "dev_record")
        if not table_path.is_file():
            raise SF.SealedFinalsError(f"{final.name}: no T07 record {table_path}")
        table = json.loads(table_path.read_text(encoding="utf-8"))
        key = (source["t07_arch"], source["t07_network"], source["set"])
        hits = [row for row in table.get("rows", []) if (row.get("arch"), row.get("network"), row.get("set")) == key]
        if len(hits) != 1:
            raise SF.SealedFinalsError(f"{final.name}: {len(hits)} rows {key} in {table_path}; expected one")
        folder = table_path.parent / "preds" / source["t07_arch"]
        paths = {"base": folder / "base" / f"{source['set']}.npy",
                 "child": folder / source["t07_network"] / f"{source['set']}.npy"}
        preds = {k: np.load(SF._inside(table_path.parent, p, "T07 predictions"), allow_pickle=False)
                 for k, p in paths.items()}
        instance = load_instance(instances_root() / final.instance_id)
        rows, _ = dev_set_rows(str(instance.private["dataset"]), source["set"])
        parts = readout_parts(rows, preds["base"], preds["child"], len(instance.task["classes"]))
        check_t07_row(hits[0], parts["readouts"], final.name)
        return {"path": str(table_path), "file_sha256": file_sha256(table_path), "fields": SF.FIELDS_ALL,
                "parts": parts, "gain": SF.points(parts["readouts"]["delta_acc"]), "clean": None,
                "predictions": {k: {"path": str(p), "file_sha256": file_sha256(p)} for k, p in paths.items()}}


def readout_parts(rows: Mapping[str, np.ndarray], base_pred: np.ndarray, child_pred: np.ndarray, n_classes: int
                  ) -> dict[str, Any]:
    """``readouts`` (``split_readouts``) and ``per_image`` of one pair of predictions on one set's rows."""
    base_pred, child_pred, y = np.asarray(base_pred), np.asarray(child_pred), np.asarray(rows["y"])
    if base_pred.shape != y.shape or child_pred.shape != y.shape:
        raise SF.SealedFinalsError(f"predictions {base_pred.shape} / {child_pred.shape} do not fit {y.shape} rows")
    readouts = split_readouts(rows, base_pred, child_pred, None, n_classes)
    return {"readouts": devreadout.json_safe(readouts), "per_image": per_image(base_pred == y, child_pred == y)}


def check_t07_row(row: Mapping[str, Any], readouts: Mapping[str, Any], name: str) -> None:
    """Refuse a T07 table row that does not state the readout of its own predictions."""
    problems = [key for key in ("fixed", "broken", "n") if int(row[key]) != int(readouts[key])]
    pairs = (("delta_points", 100.0 * float(readouts["delta_acc"]), POINTS_TOL),
             ("worst_class_delta_points", 100.0 * float(readouts["worst_class_delta"]), POINTS_TOL),
             ("base_acc", float(readouts["base_acc"]), ACC_TOL), ("child_acc", float(readouts["child_acc"]), ACC_TOL))
    problems += [key for key, ours, tol in pairs if not math.isclose(float(row[key]), ours, rel_tol=0.0, abs_tol=tol)]
    if problems:
        raise SF.SealedFinalsError(f"{name}: the T07 row differs from its own predictions on {problems}")


FAMILY = VisionRange()

__all__ = ["BITS_ENCODING", "FAMILY", "VisionRange", "decode_bits", "dev_set_rows", "encode_bits", "per_image",
           "range_sealed_index", "readout_parts"]
