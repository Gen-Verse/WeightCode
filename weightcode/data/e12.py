"""PREREG E12 data store: dataset ``cifar10_r18`` (E10's data, the ImageNet-pretrained ResNet-18 student).

Layout (under $WEIGHTCODE_PRIVATE)::

    cifar10_r18/dataset.json          cifar10's manifest with this dataset's name, forward.py and provenance
    cifar10_r18/forward.py            the ResNet-18 numpy forward at the chosen R (render_resnet18up_forward)
    cifar10_r18/splits/*.npz          byte copies of cifar10's train, V_select, V_evidence and V_check (hash-checked)
    cifar10_r18/replay/               byte copies of cifar10's replay files (E1's 64 clean replay rows)
    cifar10_r18/models/               the E12 students and index.json (weightcode.models.e12_export import)
    cifar10_r18/conditions/<c>_s5/    byte copies of E10's condition splits, with a dataset.json naming this dataset

PREREG E12 uses E10's data unchanged, as E11 does: the clean rows are E1's, and each condition's V_evidence, V_check
and V_select are byte copies of ``cifar10_r20``'s, hash-checked against its dataset.json. No sealed file is read and
no image is regenerated. The student's preprocessing is not the data's: forward.py normalises with the ImageNet
constants of the pretrained weights and upsamples to R x R; dataset.json keeps cifar10's ``norm`` block (the data's
statistics) and records the student's in ``student_preprocessing``. The id avoids architecture words because its
``_``-separated parts become leak tokens (``weightcode.harness.e12``): "r18" is a whole-word token, which no editor-visible
description of the network spells. Nothing is overwritten.
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
from weightcode.models import resnet18up as M
from weightcode.models.forward_gen import render_resnet18up_forward

DATASET = "cifar10_r18"
SOURCE = E10.SOURCE
CONDITION_SOURCE = E10.DATASET
SPLITS = E10.SPLITS
CONDITION_SPLITS = E10.CONDITION_SPLITS
CONDITIONS = ("gaussian_noise_s5", "contrast_s5", "defocus_blur_s5", "snow_s5")
"""PREREG E12: gaussian_noise_s5 and contrast_s5, then the fixed substitutes defocus_blur_s5 and snow_s5 in that order
(each used only if a first condition misses headroom)."""


def forward_arch(resize: int) -> str:
    return (f"resnet18_up (torchvision resnet18 with the IMAGENET1K_V1 weights fine-tuned on E1's train rows; ImageNet "
            f"normalisation and bilinear upsampling to {resize} x {resize} in front of it, linear head with bias)")


class E12StoreError(RuntimeError):
    """The E12 store would be overwritten, or an input does not match its record."""


def build_clean(resize: int, run_info: Mapping[str, Any]) -> dict[str, Any]:
    """Write cifar10_r18's dataset.json, forward.py (at ``resize``), splits and replay from cifar10 (refuses an
    existing store)."""
    config = M.ResNet18UpConfig(resize=int(resize))
    root = dataset_dir(DATASET)
    if dataset_json_path(DATASET).exists() or forward_path(DATASET).exists():
        raise E12StoreError(f"{root} already holds a dataset; nothing written")
    source = read_dataset_json(SOURCE)
    source_json = dataset_json_path(SOURCE)
    text = render_resnet18up_forward(M.IMAGENET_MEAN, M.IMAGENET_STD, len(source["class_names"]), M.BN_EPS,
                                     config.resize, source["image_shape"])
    root.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in SPLITS:
        arrays = load_split(SOURCE, split)
        if split_sha256(arrays) != source["splits"][split]["sha256"]:
            raise E12StoreError(f"{SOURCE}/{split} does not match its dataset.json hash")
        target = split_path(DATASET, split)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(split_path(SOURCE, split), target)
        if file_sha256(target) != file_sha256(split_path(SOURCE, split)):
            raise E12StoreError(f"{target}: copy differs from its source")
        splits[split] = dict(source["splits"][split])
    replay_src, replay_dst = dataset_dir(SOURCE) / "replay", root / "replay"
    replay_files = {}
    if replay_src.is_dir():
        replay_dst.mkdir()
        for path in sorted(replay_src.iterdir()):
            shutil.copyfile(path, replay_dst / path.name)
            replay_files[path.name] = file_sha256(replay_dst / path.name)
            if replay_files[path.name] != file_sha256(path):
                raise E12StoreError(f"{replay_dst / path.name}: copy differs from its source")
    forward_path(DATASET).write_text(text, encoding="utf-8")
    manifest = {
        **{key: value for key, value in source.items() if key not in ("created_by", "forward_sha256", "splits")},
        "dataset": DATASET,
        "splits": splits,
        "forward_path": "forward.py",
        "forward_sha256": file_sha256(forward_path(DATASET)),
        "forward_arch": forward_arch(config.resize),
        "student_preprocessing": {"mean": list(M.IMAGENET_MEAN), "std": list(M.IMAGENET_STD),
                                  "resize": config.resize, "bn_eps": M.BN_EPS,
                                  "source": f"{M.WEIGHTS_NAME} transforms (torchvision); upsampling fixed by PREREG E12"},
        "derived_from": {"dataset": SOURCE, "dataset_json_sha256": file_sha256(source_json),
                         "splits": "byte copies of the source's split files", "replay_files": replay_files,
                         "rule": "PREREG E12: E10's data (E1's clean rows), E12's ImageNet-pretrained ResNet-18"},
        "created_by": dict(run_info),
    }
    write_json(dataset_json_path(DATASET), manifest)
    return manifest


def build_condition(condition: str, run_info: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one E10 condition (``<corruption>_s5``) into cifar10_r18, byte for byte, hash-checked."""
    if condition not in CONDITIONS:
        raise E12StoreError(f"{condition!r} is not an E12 condition; E12's are {list(CONDITIONS)}")
    if dataset_json_path(DATASET, condition).exists() or dataset_dir(DATASET, condition).exists():
        raise E12StoreError(f"{dataset_dir(DATASET, condition)} exists; nothing written")
    clean = read_dataset_json(DATASET)
    e10_clean = read_dataset_json(CONDITION_SOURCE)
    source = read_dataset_json(CONDITION_SOURCE, condition)
    if source["condition"]["name"] != condition:
        raise E12StoreError(f"{CONDITION_SOURCE}/{condition} names {source['condition']['name']!r}")
    for split in CONDITION_SPLITS:
        if clean["splits"][split]["sha256"] != e10_clean["splits"][split]["sha256"]:
            raise E12StoreError(f"clean {split} differs between {DATASET} and {CONDITION_SOURCE}")
    splits: dict[str, Any] = {}
    for split in CONDITION_SPLITS:
        arrays = load_split(CONDITION_SOURCE, split, condition)
        if split_sha256(arrays) != source["splits"][split]["sha256"]:
            raise E12StoreError(f"{CONDITION_SOURCE}/{condition}/{split} does not match its dataset.json hash")
        target = split_path(DATASET, split, condition)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(split_path(CONDITION_SOURCE, split, condition), target)
        if file_sha256(target) != file_sha256(split_path(CONDITION_SOURCE, split, condition)):
            raise E12StoreError(f"{target}: copy differs from its source")
        if split_sha256(load_split(DATASET, split, condition)) != source["splits"][split]["sha256"]:
            raise E12StoreError(f"{target}: reloaded split hashes differently")
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
                         "rule": "PREREG E12: E10's condition rows unchanged; no sealed read, no regeneration"},
        "created_by": dict(run_info),
    }
    write_json(dataset_json_path(DATASET, condition), manifest)
    return manifest


__all__ = ["CONDITIONS", "CONDITION_SOURCE", "CONDITION_SPLITS", "DATASET", "E12StoreError", "SOURCE", "build_clean",
           "build_condition", "forward_arch"]
