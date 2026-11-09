"""Build, write and verify the stored seqops dataset and its new-skill conditions (PREREG E5).

Layout (paths from ``weightcode.data.store``):

* ``$WEIGHTCODE_PRIVATE/seqops/splits/{train,V_select,V_evidence,V_check}.npz``, ``forward.py`` and ``dataset.json``;
  ``$WEIGHTCODE_SEALED/seqops/test.npz``. The held-out splits hold old-operator rows (part 4); ``train`` is the sample of
  the training distribution that evidence draws replay.
* ``$WEIGHTCODE_PRIVATE/seqops/conditions/new<k>_<kind>/splits/{V_select,V_evidence,V_check}.npz`` and ``dataset.json``;
  ``$WEIGHTCODE_SEALED/seqops/conditions/new<k>_<kind>/test.npz``. V_select, V_check and test end with the dataset's own
  split of the same name as part 4 (``weightcode.data.seqops.condition_store_splits``); every condition uses the dataset's
  forward.py.

Every array is a pure function of SPLIT_SEED, the streams and the code, so ``verify_base`` and ``verify_condition``
re-derive each split and compare canonical hashes with dataset.json and the private files. They never read a sealed
file: the sealed test is compared with its dataset.json hash only.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import params_sha256
from weightcode.data import seqops as S
from weightcode.data.registry import SPLIT_SEED
from weightcode.data.store import (
    dataset_json_path,
    load_split,
    read_dataset_json,
    save_split,
    sealed_test_path,
    split_path,
    split_sha256,
    write_json,
)

DATASET_SCHEMA = "weightcode.dataset.seqops.v1"
CONDITION_SCHEMA = "weightcode.dataset.seqops_condition.v1"
BASE_PRIVATE_SPLITS: tuple[str, ...] = ("train", "V_select", "V_evidence", "V_check")
CONDITION_PRIVATE_SPLITS: tuple[str, ...] = ("V_select", "V_evidence", "V_check")
CLASS_NAMES: tuple[str, ...] = tuple(f"s{i}" for i in range(S.N_SYMBOLS))
FORWARD_N_HEADS = 4
"""Attention heads of every seqops student (both pre-registered sizes); forward.py is rendered with it."""


class SeqOpsStoreError(RuntimeError):
    """A stored seqops split or manifest disagrees with what the code derives; nothing is overwritten."""


@dataclass
class BuiltSeqData:
    """Private splits, the sealed test arrays and the manifest of the base dataset or of one condition."""

    splits: dict[str, dict[str, np.ndarray]]
    test: dict[str, np.ndarray]
    manifest: dict[str, Any]
    condition: str | None = None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward_constants() -> dict[str, Any]:
    """The constants forward.py is rendered with (``weightcode.models.forward_gen.render_seq_forward``)."""
    from weightcode.models.transformer import LN_EPS

    return {"seq_len": S.SEQ_LEN, "vocab_size": S.VOCAB_SIZE, "n_out": S.N_SYMBOLS, "n_heads": FORWARD_N_HEADS,
            "pad_id": S.PAD, "ln_eps": LN_EPS}


def split_record(name: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """dataset.json entry of one split: path, size, hash and what its rows are."""
    parts, counts = np.unique(arrays["part"], return_counts=True)
    return {
        "path": f"splits/{name}.npz",
        "n": int(arrays["y"].shape[0]),
        "sha256": split_sha256(arrays),
        "part_counts": {str(int(p)): int(c) for p, c in zip(parts, counts)},
        "single_operator_fraction": float(np.mean(arrays["opb"] == S.NONE)),
        "length_range": [int(arrays["n"].min()), int(arrays["n"].max())],
        "symbol_coverage": S.symbol_coverage(arrays["x"]),
    }


def _test_record(location: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    parts, counts = np.unique(arrays["part"], return_counts=True)
    return {"location": location, "n": int(arrays["y"].shape[0]), "sha256": split_sha256(arrays),
            "part_counts": {str(int(p)): int(c) for p, c in zip(parts, counts)}}


def held_out_record(keys: np.ndarray) -> dict[str, Any]:
    return {
        "splits": list(S.BASE_ORDER),
        "n_keys": int(keys.size),
        "keys_sha256": params_sha256({"keys": np.asarray(keys, dtype=np.int64)}),
        "rule": "base training refuses every row key of these splits; the train sample is ordinary training data",
    }


def build_base() -> BuiltSeqData:
    """The dataset's own splits (``weightcode.data.seqops.build_base_splits``) and its manifest."""
    built = S.build_base_splits()
    test = built.pop("test")
    held = S.held_out_keys({**built, "test": test})
    manifest: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "dataset": S.DATASET,
        "kind": "token_rows",
        "class_names": list(CLASS_NAMES),
        "input": {"shape": [S.SEQ_LEN], "dtype": "int64"},
        "format": S.format_manifest(),
        "parts": {str(part): name for part, name in S.PART_NAMES.items()},
        "split_seed": SPLIT_SEED,
        "streams": {name: [SPLIT_SEED, stream] for name, stream in S.BASE_STREAMS.items()},
        "split_rule": "held-out splits drawn in the order V_select, V_evidence, V_check, test, each from its own "
        "stream, row keys distinct within and across them; then the train sample from its stream, keys distinct "
        "from every held-out split; every row is an old-operator row of the training distribution (part 4)",
        "split_sizes": {name: int(arrays["y"].shape[0]) for name, arrays in built.items()},
        "splits": {name: split_record(name, built[name]) for name in BASE_PRIVATE_SPLITS},
        "test": _test_record(f"$WEIGHTCODE_SEALED/{S.DATASET}/test.npz", test),
        "held_out": held_out_record(held),
        "numpy": np.__version__,
    }
    return BuiltSeqData(splits={name: built[name] for name in BASE_PRIVATE_SPLITS}, test=test, manifest=manifest)


def _make_sealed_parent(condition: str | None) -> None:
    parent = sealed_test_path(S.DATASET, condition).parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(parent, 0o700)


def write_built(built: BuiltSeqData, extra: dict[str, Any]) -> dict[str, Any]:
    """Write the private splits, the sealed test and dataset.json (with ``extra`` merged in); refuses to overwrite."""
    manifest_path = dataset_json_path(S.DATASET, built.condition)
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; verify it instead of rebuilding")
    for name, arrays in built.splits.items():
        if save_split(split_path(S.DATASET, name, built.condition), arrays) != built.manifest["splits"][name]["sha256"]:
            raise SeqOpsStoreError(f"{name}: written hash differs from the built hash")
    _make_sealed_parent(built.condition)
    if save_split(sealed_test_path(S.DATASET, built.condition), built.test, private_mode=True) != built.manifest["test"]["sha256"]:
        raise SeqOpsStoreError("test: written hash differs from the built hash")
    manifest = {**built.manifest, **extra}
    write_json(manifest_path, manifest)
    return manifest


def _compare(built: BuiltSeqData) -> dict[str, Any]:
    recorded = read_dataset_json(S.DATASET, built.condition)
    report: dict[str, Any] = {}
    for name, arrays in built.splits.items():
        fresh = split_sha256(arrays)
        on_disk = split_sha256(load_split(S.DATASET, name, built.condition))
        expected = recorded["splits"][name]["sha256"]
        report[name] = {"match": fresh == on_disk == expected, "fresh": fresh, "on_disk": on_disk, "recorded": expected}
    fresh_test = built.manifest["test"]["sha256"]
    report["test"] = {"match": fresh_test == recorded["test"]["sha256"], "fresh": fresh_test,
                      "recorded": recorded["test"]["sha256"], "note": "the sealed file is not read"}
    for key in ("splits", "test", "held_out", "condition", "old_part"):
        if key in built.manifest and built.manifest[key] != recorded.get(key):
            report[f"manifest.{key}"] = {"match": False}
    return report


def verify_base() -> dict[str, Any]:
    """Re-derive the base dataset and compare with dataset.json and the private files."""
    return _compare(build_base())


def checked_exclusion(manifest: dict[str, Any]) -> np.ndarray:
    """The row keys base training refuses, re-derived from the code and checked against ``manifest``.

    Every base split (the sealed test included) is regenerated in memory and its hash compared with dataset.json, so
    a training job needs no held-out file to refuse the held-out rows, and a code or numpy change that would move a
    split stops the job before its first step. Nothing is written.
    """
    built = build_base()
    for name in BASE_PRIVATE_SPLITS:
        if built.manifest["splits"][name]["sha256"] != manifest["splits"][name]["sha256"]:
            raise SeqOpsStoreError(f"{name} re-derived here does not match dataset.json")
    if built.manifest["test"]["sha256"] != manifest["test"]["sha256"]:
        raise SeqOpsStoreError("test re-derived here does not match dataset.json")
    keys = S.held_out_keys({**built.splits, "test": built.test})
    if held_out_record(keys) != manifest["held_out"]:
        raise SeqOpsStoreError("held-out keys re-derived here do not match dataset.json")
    return keys


def parse_condition(name: str) -> tuple[int, str]:
    """(index, kind) of a condition name ``new<k>_<kind>``."""
    head, _, kind = name.partition("_")
    if not (head.startswith("new") and head[3:].isdigit() and kind in S.RULE_KINDS):
        raise ValueError(f"{name!r} is not a seqops condition name new<k>_<kind> with kind in {S.RULE_KINDS}")
    index = int(head[3:])
    if S.condition_name(index, kind) != name:
        raise ValueError(f"{name!r} is not canonical")
    return index, kind


def build_condition(name: str) -> BuiltSeqData:
    """The stored splits of a new-skill condition and its manifest (the rule is drawn here, after base training)."""
    index, kind = parse_condition(name)
    rule = S.draw_new_rule(index, kind)
    base = S.build_base_splits()
    stored = S.condition_store_splits(rule, index, base)
    test = stored.pop("test")
    for split in S.CHECK_OLD_PART_SPLITS:
        arrays = stored[split] if split != "test" else test
        old = arrays["part"] == S.PART_OLD
        if split_sha256({key: value[old] for key, value in arrays.items()}) != split_sha256(base[split]):
            raise SeqOpsStoreError(f"{split}: the old part is not the dataset's own {split}")
    specs = {str(part): S.part_spec(part, rule.token).as_dict() for part in (S.PART_NEW_SHORT, S.PART_NEW_LONG,
                                                                             S.PART_NEW_COMPOSED)}
    specs[str(S.PART_OLD)] = S.OLD_ROWS.as_dict()
    manifest: dict[str, Any] = {
        "schema": CONDITION_SCHEMA,
        "dataset": S.DATASET,
        "condition": {
            "name": name,
            "index": index,
            "kind": kind,
            "operator": S.TOKENS[rule.token],
            "operator_id": int(rule.token),
            "rule_stream": [SPLIT_SEED, S.NEW_RULE_STREAM, index],
            "rule": rule.as_dict(),
            "rule_agreement": S.symbol_agreement(rule.table) if rule.table is not None else None,
            "private": "the rule is the condition's ground truth; it is never shown to an editor",
        },
        "class_names": list(CLASS_NAMES),
        "input": {"shape": [S.SEQ_LEN], "dtype": "int64"},
        "parts": {str(part): S.PART_NAMES[part] for part in S.PART_NAMES},
        "part_rows": specs,
        "split_seed": SPLIT_SEED,
        "streams": {split: [SPLIT_SEED, S.CONDITION_STREAM_BASE + 10 * index + position]
                    for position, split in enumerate(S.CONDITION_ORDER)},
        "split_rule": "new parts drawn split by split in the order V_select, V_evidence, V_check, test, parts in "
        "increasing order, each split from its own stream, row keys distinct across all of them; V_select, V_check "
        "and test then append the dataset's own split of the same name as part 4",
        "split_sizes": {split: int(arrays["y"].shape[0]) for split, arrays in stored.items()},
        "splits": {split: split_record(split, stored[split]) for split in CONDITION_PRIVATE_SPLITS},
        "test": _test_record(f"$WEIGHTCODE_SEALED/{S.DATASET}/conditions/{name}/test.npz", test),
        "old_part": {
            "splits": list(S.CHECK_OLD_PART_SPLITS),
            "source": "the dataset's own split of the same name, appended after the new parts",
            "sha256": {split: split_sha256(base[split]) for split in S.CHECK_OLD_PART_SPLITS},
        },
        "numpy": np.__version__,
    }
    return BuiltSeqData(splits={split: stored[split] for split in CONDITION_PRIVATE_SPLITS}, test=test,
                        manifest=manifest, condition=name)


def condition_extra(base_manifest: dict[str, Any]) -> dict[str, Any]:
    """Fields a condition's dataset.json takes from the base dataset: its manifest hash and the shared forward."""
    return {
        "base_dataset_json_sha256": _file_sha256(dataset_json_path(S.DATASET)),
        "forward_path": "../../forward.py",
        "forward_sha256": base_manifest["forward_sha256"],
    }


def verify_condition(name: str) -> dict[str, Any]:
    """Re-derive a condition and compare with its dataset.json and private files."""
    return _compare(build_condition(name))
