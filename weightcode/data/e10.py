"""PREREG E10 data store: dataset ``cifar10_r20`` (E1's CIFAR-10 splits, ResNet-20 students, CIFAR-10-C conditions).

Layout (under $WEIGHTCODE_PRIVATE)::

    cifar10_r20/dataset.json          cifar10's manifest with this dataset's name, forward.py and provenance
    cifar10_r20/forward.py            the ResNet-20 (option A) numpy forward (render_resnet20a_forward)
    cifar10_r20/splits/*.npz          byte copies of cifar10's train, V_select, V_evidence and V_check (hash-checked)
    cifar10_r20/replay/               byte copies of cifar10's replay files (E1's 64 clean replay rows)
    cifar10_r20/models/               the six E10 students and index.json (weightcode.models.e10_export import)
    cifar10_r20/conditions/<c>_s5/    dataset.json and splits/{V_evidence,V_check,V_select}.npz of each condition

A separate dataset id keeps E1's per-dataset machinery (forward.py, model index, conditions, instance builder) as it
is, with E10's own forward.py and students; the clean rows are E1's, byte for byte. A condition's split keeps the
clean split's y and idx and replaces x by the official code's output (``scripts/e10/cifar10c_make_dev.py``, staged
outside the store); its dataset.json records the official code, the seed of every (condition, split), the fidelity
record, and the sealed official file it stands for (``test``: the severity-5 rows of the official .npy, never read
here). Nothing is overwritten.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.evalkit.targets import file_sha256
from weightcode.data.store import (
    dataset_dir,
    dataset_json_path,
    forward_path,
    load_split,
    read_dataset_json,
    save_split,
    split_path,
    split_sha256,
    write_json,
)
from weightcode.models.forward_gen import render_resnet20a_forward
from weightcode.models.resnet20 import BN_EPS

DATASET = "cifar10_r20"
SOURCE = "cifar10"
SPLITS = ("train", "V_select", "V_evidence", "V_check")
CONDITION_SPLITS = ("V_evidence", "V_check", "V_select")
SEVERITY = 5
CORRUPTIONS = ("gaussian_noise", "defocus_blur", "snow", "fog", "contrast")
FAMILIES = {"gaussian_noise": "noise", "defocus_blur": "blur", "snow": "weather", "fog": "weather",
            "contrast": "digital"}
CONSTANTS_S5 = {"gaussian_noise": "c = 0.10", "defocus_blur": "c = (1.5, 0.1)",
                "snow": "c = (0.3, 0.3, 1.25, 0.65, 14, 12, 0.8)", "fog": "c = (1.5, 1.75)",
                "contrast": "c = 0.15"}


class E10StoreError(RuntimeError):
    """The E10 store would be overwritten, or an input does not match its record."""


def condition_name(corruption: str) -> str:
    if corruption not in CORRUPTIONS:
        raise E10StoreError(f"unknown corruption {corruption!r}")
    return f"{corruption}_s{SEVERITY}"


def build_clean(run_info: Mapping[str, Any]) -> dict[str, Any]:
    """Write cifar10_r20's dataset.json, forward.py, splits and replay from cifar10 (refuses an existing store)."""
    root = dataset_dir(DATASET)
    if dataset_json_path(DATASET).exists() or forward_path(DATASET).exists():
        raise E10StoreError(f"{root} already holds a dataset; nothing written")
    source = read_dataset_json(SOURCE)
    source_json = dataset_json_path(SOURCE)
    mean, std = [float(v) for v in source["norm"]["mean"]], [float(v) for v in source["norm"]["std"]]
    text = render_resnet20a_forward(mean, std, len(source["class_names"]), BN_EPS, source["image_shape"])
    root.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in SPLITS:
        arrays = load_split(SOURCE, split)
        digest = split_sha256(arrays)
        if digest != source["splits"][split]["sha256"]:
            raise E10StoreError(f"{SOURCE}/{split} does not match its dataset.json hash")
        target = split_path(DATASET, split)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(split_path(SOURCE, split), target)
        if file_sha256(target) != file_sha256(split_path(SOURCE, split)):
            raise E10StoreError(f"{target}: copy differs from its source")
        splits[split] = dict(source["splits"][split])
    replay_src, replay_dst = dataset_dir(SOURCE) / "replay", root / "replay"
    replay_files = {}
    if replay_src.is_dir():
        replay_dst.mkdir()
        for path in sorted(replay_src.iterdir()):
            shutil.copyfile(path, replay_dst / path.name)
            replay_files[path.name] = file_sha256(replay_dst / path.name)
            if replay_files[path.name] != file_sha256(path):
                raise E10StoreError(f"{replay_dst / path.name}: copy differs from its source")
    forward_path(DATASET).write_text(text, encoding="utf-8")
    manifest = {
        **{key: value for key, value in source.items() if key not in ("created_by", "forward_sha256", "splits")},
        "dataset": DATASET,
        "splits": splits,
        "forward_path": "forward.py",
        "forward_sha256": file_sha256(forward_path(DATASET)),
        "forward_arch": "resnet20_a (He et al. 2016 CIFAR ResNet-20, option-A shortcuts, linear head with bias)",
        "derived_from": {"dataset": SOURCE, "dataset_json_sha256": file_sha256(source_json),
                         "splits": "byte copies of the source's split files", "replay_files": replay_files,
                         "rule": "PREREG E10: E1's clean rows, E10's ResNet-20 students"},
        "created_by": dict(run_info),
    }
    write_json(dataset_json_path(DATASET), manifest)
    return manifest


def build_condition(
    corruption: str,
    staged_dir: Path,
    dev_record_path: Path,
    fidelity_path: Path,
    install_path: Path,
    run_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one condition from the staged dev rows (``<staged_dir>/<corruption>/<split>.npz``, array ``x``)."""
    name = condition_name(corruption)
    target_json = dataset_json_path(DATASET, name)
    if target_json.exists() or dataset_dir(DATASET, name).exists():
        raise E10StoreError(f"{dataset_dir(DATASET, name)} exists; nothing written")
    clean = read_dataset_json(DATASET)
    dev = json.loads(Path(dev_record_path).read_text(encoding="utf-8"))
    fidelity = json.loads(Path(fidelity_path).read_text(encoding="utf-8"))
    install = json.loads(Path(install_path).read_text(encoding="utf-8"))
    if corruption not in fidelity["e10_conditions"] or corruption not in dev["conditions"]:
        raise E10StoreError(f"{corruption} is not an E10 condition of the fidelity record")
    if dev["fidelity"]["sha256"] != file_sha256(Path(fidelity_path)):
        raise E10StoreError("the dev rows were generated from another fidelity record")
    splits: dict[str, Any] = {}
    seeds: dict[str, int] = {}
    for split in CONDITION_SPLITS:
        row = dev["rows"][f"{corruption}/{split}"]
        staged = Path(staged_dir) / corruption / f"{split}.npz"
        if file_sha256(staged) != row["out_file_sha256"]:
            raise E10StoreError(f"{staged}: sha256 differs from the dev-rows record")
        with np.load(staged, allow_pickle=False) as archive:
            if archive.files != ["x"]:
                raise E10StoreError(f"{staged} must hold x only, has {archive.files}")
            x = np.ascontiguousarray(archive["x"])
        base = load_split(DATASET, split)
        if split_sha256(base) != clean["splits"][split]["sha256"] or x.shape != base["x"].shape or x.dtype != np.uint8:
            raise E10StoreError(f"{split}: staged x {x.dtype}{x.shape} does not fit the clean split")
        if file_sha256(split_path(SOURCE, split)) != row["clean_file_sha256"]:
            raise E10StoreError(f"{split}: the dev rows were generated from another clean file")
        digest = save_split(split_path(DATASET, split, name), {"x": x, "y": base["y"], "idx": base["idx"]})
        splits[split] = {"path": f"splits/{split}.npz", "n": int(len(x)), "sha256": digest,
                         "class_counts": clean["splits"][split]["class_counts"],
                         "clean_sha256": clean["splits"][split]["sha256"]}
        seeds[split] = int(row["seed"])
    official_file = f"CIFAR-10-C/{corruption}.npy"
    manifest = {
        "schema": clean["schema"],
        "dataset": DATASET,
        "condition": {
            "name": name, "kind": "corruption", "corruption": corruption, "family": FAMILIES[corruption],
            "severity": SEVERITY, "constants": CONSTANTS_S5[corruption],
            "source": {"repository": "https://github.com/hendrycks/robustness", "file": dev["official"]["file"],
                       "commit": dev["official"]["commit"], "file_sha256": dev["official"]["sha256"],
                       "run_as": "the official distortion function, per image after convert_img, np.uint8 of the "
                                 "result (scripts/e10/cifar10c_official.py)"},
            "rng": {"generator": "numpy global RandomState, np.random.seed(seed) once per (condition, split), "
                                 "images in row order", "seeds": seeds},
            "libraries": dev["libraries"],
            "dev_rows_record_sha256": file_sha256(Path(dev_record_path)),
            "fidelity": {"record_sha256": file_sha256(Path(fidelity_path)),
                         "result": fidelity["results"][corruption]},
            "applied_to": "rows of the clean splits V_evidence, V_check and V_select (same order, y and idx; only x "
                          "changes)",
        },
        **{key: clean[key] for key in ("source", "class_names", "image_shape", "dtype", "layout", "norm",
                                       "split_seed", "forward_path", "forward_sha256")},
        "forward_path": "../../forward.py",
        "splits": splits,
        "test": {"location": f"$WEIGHTCODE_SEALED/cifar10c/{official_file}", "rows": "40000 .. 49999 (severity 5)",
                 "format": "official CIFAR-10-C .npy, NHWC uint8, the official test order",
                 "n": 10000, "sha256": install["files"][official_file]["sha256"],
                 "clean_sha256": clean["test"]["sha256"]},
        "created_by": dict(run_info),
    }
    write_json(target_json, manifest)
    return manifest


__all__ = ["CONDITION_SPLITS", "CORRUPTIONS", "DATASET", "E10StoreError", "SEVERITY", "SOURCE", "build_clean",
           "build_condition", "condition_name"]
