"""Rule graphnew-classes-v1 (PREREG E7): which ogbn-arxiv classes are new, drawn once from the class counts.

The draw reads only the per-class counts in ``dataset.json`` of the inductive protocol (``ogbn_arxiv_ind``), never a
model output:

* eligible: classes with at least ``MIN_EVIDENCE_ROWS`` V_evidence rows (so the 10 shots per class are at most half of
  that class's evidence pool), minus the ``N_LARGEST_EXCLUDED`` classes with the most base train rows (ties broken by
  the lower label), so that no single new class dominates the new part of V_check and V_select;
* new classes: ``N_NEW_CLASSES`` eligible classes drawn uniformly without replacement with
  ``default_rng([SPLIT_SEED, CLASS_STREAM])`` from the eligible labels in ascending order, then sorted;
* train classes: the other classes, in label order.

The record is a sidecar, ``$WEIGHTCODE_PRIVATE/ogbn_arxiv_ind/graphnew_classes.json`` (``CLASSES_FILE``); dataset.json is
not rewritten, because its hash is in the records of every model trained on this dataset. ``write_class_draw``
writes it once and refuses to change it; ``read_class_draw`` checks it against dataset.json and against a fresh draw.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.data import arxiv as A
from weightcode.data.registry import SPLIT_SEED
from weightcode.data.store import dataset_dir, read_dataset_json

CLASS_RULE = "graphnew-classes-v1"
CLASS_STREAM = 705
N_NEW_CLASSES = 8
MIN_EVIDENCE_ROWS = 20
N_LARGEST_EXCLUDED = 3
DATASET = A.INDUCTIVE_DATASET
CLASSES_FILE = "graphnew_classes.json"
SCHEMA = "weightcode.graphnew_classes.v1"
COUNT_SPLITS: tuple[str, ...] = ("train", "old_replay", "old_select", "old_check", "V_evidence", "V_check", "V_select")
ELIGIBILITY = (
    f"classes with at least {MIN_EVIDENCE_ROWS} V_evidence rows, minus the {N_LARGEST_EXCLUDED} classes with the most "
    "base train rows (ties: lower label first)"
)


class ClassDrawError(RuntimeError):
    """The class draw record is missing, differs from a fresh draw, or does not describe this dataset."""


def _counts(meta: Mapping[str, Any], split: str) -> np.ndarray:
    counts = np.asarray(meta["splits"][split]["class_counts"], dtype=np.int64)
    if counts.shape != (A.N_CLASSES,):
        raise ClassDrawError(f"dataset.json class_counts of {split} has shape {counts.shape}, expected ({A.N_CLASSES},)")
    return counts


def eligible_classes(meta: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """(eligible labels ascending, the excluded largest train classes in order of size)."""
    train, evidence = _counts(meta, "train"), _counts(meta, "V_evidence")
    largest = sorted(range(A.N_CLASSES), key=lambda c: (-int(train[c]), c))[:N_LARGEST_EXCLUDED]
    eligible = [c for c in range(A.N_CLASSES) if evidence[c] >= MIN_EVIDENCE_ROWS and c not in largest]
    if len(eligible) < N_NEW_CLASSES:
        raise ClassDrawError(f"only {len(eligible)} eligible classes; the rule draws {N_NEW_CLASSES}")
    return eligible, largest


def draw_new_classes(meta: Mapping[str, Any]) -> dict[str, Any]:
    """The draw of rule graphnew-classes-v1 on one dataset.json (pure)."""
    if meta.get("protocol") != "inductive":
        raise ClassDrawError("the E7 class draw is defined on the inductive protocol (ogbn_arxiv_ind)")
    eligible, largest = eligible_classes(meta)
    rng = np.random.default_rng([SPLIT_SEED, CLASS_STREAM])
    new = sorted(int(c) for c in rng.choice(np.asarray(eligible, dtype=np.int64), size=N_NEW_CLASSES, replace=False))
    train_classes = [c for c in range(A.N_CLASSES) if c not in new]
    counts = {split: _counts(meta, split) for split in COUNT_SPLITS}
    return {
        "schema": SCHEMA,
        "rule": CLASS_RULE,
        "dataset": DATASET,
        "split_seed": SPLIT_SEED,
        "stream": [SPLIT_SEED, CLASS_STREAM],
        "n_new_classes": N_NEW_CLASSES,
        "eligibility": ELIGIBILITY,
        "min_evidence_rows": MIN_EVIDENCE_ROWS,
        "excluded_largest": [{"label": c, "train_rows": int(counts["train"][c])} for c in largest],
        "eligible": eligible,
        "new_classes": new,
        "train_classes": train_classes,
        "new_class_rows": {split: {str(c): int(counts[split][c]) for c in new} for split in COUNT_SPLITS},
        "new_part_rows": {split: int(counts[split][new].sum()) for split in COUNT_SPLITS},
        "train_class_rows": {split: int(counts[split][train_classes].sum()) for split in COUNT_SPLITS},
    }


def class_draw_path(dataset: str = DATASET) -> Path:
    """``$WEIGHTCODE_PRIVATE/<dataset>/graphnew_classes.json``."""
    return dataset_dir(dataset) / CLASSES_FILE


def _dataset_json_sha256(dataset: str) -> str:
    return hashlib.sha256((dataset_dir(dataset) / "dataset.json").read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def write_class_draw(dataset: str = DATASET) -> dict[str, Any]:
    """Draw once and write the sidecar; an existing sidecar must equal the fresh draw and is never rewritten.

    Returns the record with ``status`` ("created" or "exists") and the file's sha256.
    """
    meta = read_dataset_json(dataset)
    record = {**draw_new_classes(meta), "dataset_json_sha256": _dataset_json_sha256(dataset)}
    path = class_draw_path(dataset)
    status = "exists"
    if path.exists():
        found = json.loads(path.read_text(encoding="utf-8"))
        if found != json.loads(_text(record)):
            raise ClassDrawError(f"{path} differs from a fresh draw; nothing overwritten")
    else:
        staging = path.with_name(f".{CLASSES_FILE}.tmp")
        staging.write_text(_text(record), encoding="utf-8")
        staging.rename(path)
        status = "created"
    return {**read_class_draw(dataset), "status": status, "sha256": file_sha256(path)}


def read_class_draw(dataset: str = DATASET) -> dict[str, Any]:
    """The sidecar record, checked against the current dataset.json and against a fresh draw."""
    path = class_draw_path(dataset)
    if not path.exists():
        raise ClassDrawError(f"{path} does not exist; run scripts/arxiv_graphnew_classes.py first")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA or record.get("dataset") != dataset:
        raise ClassDrawError(f"{path} is not a {SCHEMA} record of {dataset}")
    if record.get("dataset_json_sha256") != _dataset_json_sha256(dataset):
        raise ClassDrawError(f"{path} was drawn on another dataset.json")
    fresh = {**draw_new_classes(read_dataset_json(dataset)), "dataset_json_sha256": record["dataset_json_sha256"]}
    if json.loads(_text(fresh)) != record:
        raise ClassDrawError(f"{path} differs from a fresh draw of rule {CLASS_RULE}")
    return record


def class_draw_block(dataset: str = DATASET) -> dict[str, Any]:
    """The provenance block a student or an instance records: path, sha256, rule and the two class lists."""
    record = read_class_draw(dataset)
    return {"path": CLASSES_FILE, "sha256": file_sha256(class_draw_path(dataset)), "rule": record["rule"],
            "new_classes": record["new_classes"], "train_classes": record["train_classes"]}


__all__ = [
    "CLASSES_FILE", "CLASS_RULE", "CLASS_STREAM", "ClassDrawError", "DATASET", "MIN_EVIDENCE_ROWS", "N_LARGEST_EXCLUDED",
    "N_NEW_CLASSES", "SCHEMA", "class_draw_block", "class_draw_path", "draw_new_classes", "eligible_classes",
    "read_class_draw", "write_class_draw",
]
