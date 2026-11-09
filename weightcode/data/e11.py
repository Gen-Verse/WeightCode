"""PREREG E11 data store: dataset ``cifar10_vl7`` (E10's data, ViT-Lite-7/4 students).

Layout (under $WEIGHTCODE_PRIVATE)::

    cifar10_vl7/dataset.json          cifar10's manifest with this dataset's name, forward.py and provenance
    cifar10_vl7/forward.py            the ViT-Lite-7/4 numpy forward (render_vitlite_forward)
    cifar10_vl7/splits/*.npz          byte copies of cifar10's train, V_select, V_evidence and V_check (hash-checked)
    cifar10_vl7/replay/               byte copies of cifar10's replay files (E1's 64 clean replay rows)
    cifar10_vl7/models/               the six E11 students and index.json (weightcode.models.e11_export import)
    cifar10_vl7/conditions/<c>_s5/    byte copies of E10's condition splits, with a dataset.json naming this dataset

PREREG E11 uses E10's data unchanged: the clean rows are E1's (as in ``cifar10_r20``), and each condition's
V_evidence, V_check and V_select are byte copies of ``cifar10_r20``'s, hash-checked against its dataset.json. The
condition block (official code, seeds, fidelity record, the sealed official file it stands for) is E10's, so no
sealed file is read and no image is regenerated. The id avoids architecture words because its ``_``-separated parts
become leak tokens (``weightcode.harness.e11``). Nothing is overwritten.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from typing import Any

from weightcode.data import e10 as E10
from weightcode.data.store import (
    dataset_dir,
    dataset_json_path,
    forward_path,
    load_split,
    read_dataset_json,
    split_path,
    split_sha256,
    write_json,
)
from weightcode.evalkit.targets import file_sha256
from weightcode.models.forward_gen import render_vitlite_forward
from weightcode.models.vitlite import LN_EPS

DATASET = "cifar10_vl7"
SOURCE = E10.SOURCE
CONDITION_SOURCE = E10.DATASET
SPLITS = E10.SPLITS
CONDITION_SPLITS = E10.CONDITION_SPLITS
FORWARD_ARCH = ("vit_lite_7_4 (Hassani et al. 2021 ViT-Lite-7/4: 4x4 patches, class token, learnable positional "
                "embedding, 7 layers of width 256 with 4 heads and MLP 512, tanh GELU, linear head)")


class E11StoreError(RuntimeError):
    """The E11 store would be overwritten, or an input does not match its record."""


def build_clean(run_info: Mapping[str, Any]) -> dict[str, Any]:
    """Write cifar10_vl7's dataset.json, forward.py, splits and replay from cifar10 (refuses an existing store)."""
    root = dataset_dir(DATASET)
    if dataset_json_path(DATASET).exists() or forward_path(DATASET).exists():
        raise E11StoreError(f"{root} already holds a dataset; nothing written")
    source = read_dataset_json(SOURCE)
    source_json = dataset_json_path(SOURCE)
    mean, std = [float(v) for v in source["norm"]["mean"]], [float(v) for v in source["norm"]["std"]]
    text = render_vitlite_forward(mean, std, len(source["class_names"]), LN_EPS, source["image_shape"])
    root.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in SPLITS:
        arrays = load_split(SOURCE, split)
        if split_sha256(arrays) != source["splits"][split]["sha256"]:
            raise E11StoreError(f"{SOURCE}/{split} does not match its dataset.json hash")
        target = split_path(DATASET, split)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(split_path(SOURCE, split), target)
        if file_sha256(target) != file_sha256(split_path(SOURCE, split)):
            raise E11StoreError(f"{target}: copy differs from its source")
        splits[split] = dict(source["splits"][split])
    replay_src, replay_dst = dataset_dir(SOURCE) / "replay", root / "replay"
    replay_files = {}
    if replay_src.is_dir():
        replay_dst.mkdir()
        for path in sorted(replay_src.iterdir()):
            shutil.copyfile(path, replay_dst / path.name)
            replay_files[path.name] = file_sha256(replay_dst / path.name)
            if replay_files[path.name] != file_sha256(path):
                raise E11StoreError(f"{replay_dst / path.name}: copy differs from its source")
    forward_path(DATASET).write_text(text, encoding="utf-8")
    manifest = {
        **{key: value for key, value in source.items() if key not in ("created_by", "forward_sha256", "splits")},
        "dataset": DATASET,
        "splits": splits,
        "forward_path": "forward.py",
        "forward_sha256": file_sha256(forward_path(DATASET)),
        "forward_arch": FORWARD_ARCH,
        "derived_from": {"dataset": SOURCE, "dataset_json_sha256": file_sha256(source_json),
                         "splits": "byte copies of the source's split files", "replay_files": replay_files,
                         "rule": "PREREG E11: E10's data (E1's clean rows), E11's ViT-Lite-7/4 students"},
        "created_by": dict(run_info),
    }
    write_json(dataset_json_path(DATASET), manifest)
    return manifest


def build_condition(condition: str, run_info: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one E10 condition (``<corruption>_s5``) into cifar10_vl7, byte for byte, hash-checked."""
    if dataset_json_path(DATASET, condition).exists() or dataset_dir(DATASET, condition).exists():
        raise E11StoreError(f"{dataset_dir(DATASET, condition)} exists; nothing written")
    clean = read_dataset_json(DATASET)
    e10_clean = read_dataset_json(CONDITION_SOURCE)
    source = read_dataset_json(CONDITION_SOURCE, condition)
    if source["condition"]["name"] != condition:
        raise E11StoreError(f"{CONDITION_SOURCE}/{condition} names {source['condition']['name']!r}")
    for split in CONDITION_SPLITS:
        if clean["splits"][split]["sha256"] != e10_clean["splits"][split]["sha256"]:
            raise E11StoreError(f"clean {split} differs between {DATASET} and {CONDITION_SOURCE}")
    splits: dict[str, Any] = {}
    for split in CONDITION_SPLITS:
        arrays = load_split(CONDITION_SOURCE, split, condition)
        if split_sha256(arrays) != source["splits"][split]["sha256"]:
            raise E11StoreError(f"{CONDITION_SOURCE}/{condition}/{split} does not match its dataset.json hash")
        target = split_path(DATASET, split, condition)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(split_path(CONDITION_SOURCE, split, condition), target)
        if file_sha256(target) != file_sha256(split_path(CONDITION_SOURCE, split, condition)):
            raise E11StoreError(f"{target}: copy differs from its source")
        if split_sha256(load_split(DATASET, split, condition)) != source["splits"][split]["sha256"]:
            raise E11StoreError(f"{target}: reloaded split hashes differently")
        splits[split] = dict(source["splits"][split])
    manifest = {
        **{key: value for key, value in source.items() if key not in ("created_by", "splits", "dataset",
                                                                     "forward_sha256", "forward_path")},
        "dataset": DATASET,
        "forward_path": "../../forward.py",
        "forward_sha256": clean["forward_sha256"],
        "splits": splits,
        "derived_from": {"dataset": CONDITION_SOURCE, "condition": condition,
                         "dataset_json_sha256": file_sha256(dataset_json_path(CONDITION_SOURCE, condition)),
                         "splits": "byte copies of the source condition's split files",
                         "rule": "PREREG E11: E10's condition rows unchanged; no sealed read, no regeneration"},
        "created_by": dict(run_info),
    }
    write_json(dataset_json_path(DATASET, condition), manifest)
    return manifest


__all__ = ["CONDITION_SPLITS", "CONDITION_SOURCE", "DATASET", "E11StoreError", "SOURCE", "build_clean",
           "build_condition"]
