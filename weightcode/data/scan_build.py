"""Install, build, write and verify the stored SCAN data of PREREG E13 (``weightcode.data.scan``).

Layout (paths from ``weightcode.data.store``):

* ``$WEIGHTCODE_PRIVATE/scan/raw/``: the files of the SCAN repository the build reads (``scan.RAW_SHA256``), copied once
  from a checkout at ``scan.SCAN_COMMIT``, read-only, with ``provenance.json``. Never modified, never sent to NSCC.
* ``$WEIGHTCODE_PRIVATE/scan/dataset.json`` and ``forward.py`` (shared by both conditions).
* ``$WEIGHTCODE_PRIVATE/scan/conditions/<condition>/dataset.json`` and ``splits/{train,old_check,old_select,V_check,
  V_select}.npz``; ``$WEIGHTCODE_SEALED/scan/conditions/<condition>/test.npz`` holds the sealed new part (part 1) then the
  sealed old part (part 2). The sealed file is written once by ``weightcode.data.store.save_split`` and never read here.

Every array is a pure function of the raw files, the seed-100 draw and the code, so ``verify_condition`` re-derives
each split and compares canonical hashes with dataset.json and the private files; the sealed part is compared with its
dataset.json hash only.
"""

from __future__ import annotations

import datetime
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.data import scan as S
from weightcode.data.store import (
    dataset_dir,
    dataset_json_path,
    forward_path,
    load_split,
    read_dataset_json,
    read_json,
    save_split,
    sealed_test_path,
    split_path,
    split_sha256,
    write_json,
)

DATASET_SCHEMA = "weightcode.dataset.scan.v1"
CONDITION_SCHEMA = "weightcode.dataset.scan_condition.v1"
RAW_PROVENANCE_SCHEMA = "weightcode.scan_raw.v1"
FORWARD_N_HEADS = 4
"""Attention heads of every E13 student (both pre-registered sizes); forward.py is rendered with it."""


class ScanStoreError(RuntimeError):
    """A stored SCAN file or manifest disagrees with what the code derives; nothing is overwritten."""


def raw_dir() -> Path:
    return dataset_dir(S.DATASET) / "raw"


def install_raw(checkout: Path) -> dict[str, Any]:
    """Copy the recorded files of a SCAN checkout at SCAN_COMMIT into raw/, read-only, with provenance.json.

    Refuses if raw/ exists or any file's sha256 differs from ``scan.RAW_SHA256``.
    """
    target = raw_dir()
    if target.exists():
        raise FileExistsError(f"{target} exists; the raw copy is made once")
    checkout = Path(checkout)
    files = {}
    for relative, expected in S.RAW_SHA256.items():
        digest = S.file_sha256(checkout / relative)
        if digest != expected:
            raise ScanStoreError(f"{checkout / relative}: sha256 {digest} != recorded {expected}")
        files[relative] = {"sha256": digest, "bytes": (checkout / relative).stat().st_size}
    staging = target.with_name("raw.staging")
    if staging.exists():
        raise FileExistsError(f"{staging} exists; remove it by hand after checking it")
    for relative in S.RAW_SHA256:
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkout / relative, destination)
        if S.file_sha256(destination) != S.RAW_SHA256[relative]:
            raise ScanStoreError(f"{destination}: copy hash mismatch")
        os.chmod(destination, 0o444)
    provenance = {
        "schema": RAW_PROVENANCE_SCHEMA,
        "source": S.SCAN_URL,
        "commit": S.SCAN_COMMIT,
        "copied_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "files": files,
        "rule": "copied once from a git checkout at the commit; never modified; never sent to NSCC",
    }
    write_json(staging / "provenance.json", provenance)
    os.chmod(staging / "provenance.json", 0o444)
    staging.rename(target)
    return provenance


def check_raw() -> dict[str, Any]:
    """The raw copy's provenance, refused unless every file still has its recorded sha256."""
    provenance = read_json(raw_dir() / "provenance.json")
    if provenance.get("commit") != S.SCAN_COMMIT or set(provenance.get("files", {})) != set(S.RAW_SHA256):
        raise ScanStoreError("raw/provenance.json does not record this build's commit and files")
    for relative, expected in S.RAW_SHA256.items():
        if S.file_sha256(raw_dir() / relative) != expected:
            raise ScanStoreError(f"raw/{relative} changed since the copy")
    return provenance


def forward_constants() -> dict[str, Any]:
    """The constants forward.py is rendered with (``weightcode.models.forward_gen.render_seq_forward``)."""
    from weightcode.models.transformer import LN_EPS

    return {"seq_len": S.SEQ_LEN, "vocab_size": S.VOCAB_SIZE, "n_out": S.N_CLASSES, "n_heads": FORWARD_N_HEADS,
            "pad_id": S.PAD, "ln_eps": LN_EPS}


def split_record(name: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """dataset.json entry of one split: path, size, hash and what its rows are."""
    parts, counts = np.unique(arrays["part"], return_counts=True)
    return {
        "path": f"splits/{name}.npz",
        "n": int(arrays["y"].shape[0]),
        "n_distinct_commands": int(np.unique(arrays["idx"]).size),
        "sha256": split_sha256(arrays),
        "part_counts": {str(int(p)): int(c) for p, c in zip(parts, counts)},
        "command_words_range": [int(arrays["n_cmd"].min()), int(arrays["n_cmd"].max())],
        "actions_range": [int(arrays["n_act"].min()), int(arrays["n_act"].max())],
    }


@dataclass
class BuiltCondition:
    condition: str
    splits: dict[str, dict[str, np.ndarray]]
    sealed: dict[str, np.ndarray]
    manifest: dict[str, Any]


def build_condition(condition: str) -> BuiltCondition:
    """The stored splits of one condition and its manifest (without the forward and provenance fields)."""
    check_raw()
    built = S.build_condition_splits(raw_dir(), condition)
    S.check_disjoint(built)
    sealed_parts, sealed_counts = np.unique(built.sealed["part"], return_counts=True)
    files = S.CONDITION_FILES[condition]
    manifest: dict[str, Any] = {
        "schema": CONDITION_SCHEMA,
        "dataset": S.DATASET,
        "condition": {
            "name": condition,
            "official_files": {split: {"path": path, "sha256": S.RAW_SHA256[path]} for split, path in files.items()},
            "primitive": list(S.PRIMITIVE[condition]),
            "primitive_row": S.PRIMITIVE_ROW[condition],
            "new_part": "the official test file: every command holds the primitive words as adjacent words",
            "old_part": "the distinct commands of the official training file"
            + (", without the primitive alone" if S.PRIMITIVE_ROW[condition] else ""),
            "private": "the condition name and the official file names are leak words of every E13 session",
        },
        "class_names": list(S.CLASS_NAMES),
        "input": {"shape": [S.SEQ_LEN], "dtype": "int64"},
        "parts": {str(part): name for part, name in S.PART_NAMES.items()},
        "split_seed": [S.E13_SPLIT_SEED, S.SPLIT_STREAMS[condition]],
        "split_rule": "old commands sorted by command id, one permutation: 1000 old_check, 1000 old_select, 1000 "
        "sealed old; then the official test file's commands sorted by command id, one permutation from the same "
        "generator, cut into thirds by numpy.array_split: V_check, V_select, sealed new; train = the official "
        "training rows in file order without the 3000 held-out old commands (duplicates and the primitive's rows "
        "kept); every split sorted by command id",
        "split_sizes": {name: int(arrays["y"].shape[0]) for name, arrays in built.private.items()},
        "splits": {name: split_record(name, arrays) for name, arrays in built.private.items()},
        "test": {
            "location": f"$WEIGHTCODE_SEALED/{S.DATASET}/conditions/{condition}/test.npz",
            "n": int(built.sealed["y"].shape[0]),
            "sha256": split_sha256(built.sealed),
            "part_counts": {str(int(p)): int(c) for p, c in zip(sealed_parts, sealed_counts)},
            "rule": "the sealed new third (part 1) then the sealed old commands (part 2); written once, never read in "
            "dev",
        },
        "draw": built.record,
        "numpy": np.__version__,
    }
    return BuiltCondition(condition=condition, splits=built.private, sealed=built.sealed, manifest=manifest)


def base_manifest(forward_sha: str, created_by: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": DATASET_SCHEMA,
        "dataset": S.DATASET,
        "kind": "token_rows",
        "class_names": list(S.CLASS_NAMES),
        "input": {"shape": [S.SEQ_LEN], "dtype": "int64"},
        "format": S.format_manifest(),
        "raw": {"dir": "raw", "source": S.SCAN_URL, "commit": S.SCAN_COMMIT, "files": dict(S.RAW_SHA256)},
        "conditions": list(S.CONDITIONS),
        "forward_path": "forward.py",
        "forward_sha256": forward_sha,
        "forward_constants": forward_constants(),
        "numpy": np.__version__,
        "created_by": created_by,
    }


def write_base(created_by: dict[str, Any]) -> dict[str, Any]:
    """forward.py and the dataset-level dataset.json; refuses to overwrite."""
    from weightcode.models.forward_gen import render_seq_forward, write_forward

    if dataset_json_path(S.DATASET).exists():
        raise FileExistsError(f"{dataset_json_path(S.DATASET)} exists; verify it instead of rebuilding")
    check_raw()
    forward_sha = write_forward(forward_path(S.DATASET), render_seq_forward(**forward_constants()))
    manifest = base_manifest(forward_sha, created_by)
    write_json(dataset_json_path(S.DATASET), manifest)
    return manifest


def _make_sealed_parent(condition: str) -> None:
    parent = sealed_test_path(S.DATASET, condition).parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(parent, 0o700)


def write_condition(built: BuiltCondition, created_by: dict[str, Any]) -> dict[str, Any]:
    """Write the private splits, the sealed part and dataset.json of one condition; refuses to overwrite."""
    base = read_dataset_json(S.DATASET)
    manifest_path = dataset_json_path(S.DATASET, built.condition)
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; verify it instead of rebuilding")
    if sealed_test_path(S.DATASET, built.condition).exists():
        raise FileExistsError(f"the sealed part of {built.condition} exists; it is written once")
    for name, arrays in built.splits.items():
        if save_split(split_path(S.DATASET, name, built.condition), arrays) != built.manifest["splits"][name]["sha256"]:
            raise ScanStoreError(f"{name}: written hash differs from the built hash")
    _make_sealed_parent(built.condition)
    written = save_split(sealed_test_path(S.DATASET, built.condition), built.sealed, private_mode=True)
    if written != built.manifest["test"]["sha256"]:
        raise ScanStoreError("test: written hash differs from the built hash")
    manifest = {**built.manifest, "forward_path": "../../forward.py", "forward_sha256": base["forward_sha256"],
                "base_dataset_json_sha256": S.file_sha256(dataset_json_path(S.DATASET)), "created_by": created_by}
    write_json(manifest_path, manifest)
    return manifest


def verify_condition(condition: str) -> dict[str, Any]:
    """Re-derive a condition and compare with its dataset.json and private files (the sealed file is not read)."""
    built = build_condition(condition)
    recorded = read_dataset_json(S.DATASET, condition)
    report: dict[str, Any] = {}
    for name, arrays in built.splits.items():
        fresh = split_sha256(arrays)
        on_disk = split_sha256(load_split(S.DATASET, name, condition))
        expected = recorded["splits"][name]["sha256"]
        report[name] = {"match": fresh == on_disk == expected, "fresh": fresh, "on_disk": on_disk, "recorded": expected}
    report["test"] = {"match": built.manifest["test"]["sha256"] == recorded["test"]["sha256"],
                      "fresh": built.manifest["test"]["sha256"], "recorded": recorded["test"]["sha256"],
                      "note": "the sealed file is not read"}
    for key in ("splits", "test", "condition", "draw", "split_seed"):
        if built.manifest[key] != recorded.get(key):
            report[f"manifest.{key}"] = {"match": False}
    return report


def checked_split(condition: str, name: str) -> dict[str, np.ndarray]:
    """One private split of a condition, refused unless its canonical hash is its dataset.json's."""
    manifest = read_dataset_json(S.DATASET, condition)
    arrays = load_split(S.DATASET, name, condition)
    if split_sha256(arrays) != manifest["splits"][name]["sha256"]:
        raise ScanStoreError(f"{condition}/{name} does not match its dataset.json hash")
    return arrays


__all__ = [
    "BuiltCondition", "CONDITION_SCHEMA", "DATASET_SCHEMA", "FORWARD_N_HEADS", "ScanStoreError", "base_manifest",
    "build_condition", "check_raw", "checked_split", "forward_constants", "install_raw", "raw_dir", "split_record",
    "verify_condition", "write_base", "write_condition",
]
