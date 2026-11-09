"""PREREG T04 store builders: the clean dev store, a condition, the two models, the base readouts (``weightcode.data.t04``).

Layout (under $WEIGHTCODE_PRIVATE)::

    in1k_t04/dataset.json                 manifest: class names, row rule and row-list sha256, splits, forward.py
    in1k_t04/forward.py                   the uint8 native ResNet-18 forward (render_forward_u8)
    in1k_t04/dev_index.jsonl              one line per dev row: row, shard, local row, label, stored-bytes sha256
    in1k_t04/splits/*.npz                 train, V_select, V_evidence, V_check: uint8 crops, labels, row numbers (idx)
    in1k_t04/replay/                      the 64 clean replay rows (replay-v1, from train)
    in1k_t04/models/                      seed 100 (the IMAGENET1K_V1 network) and seed 101 (the decoy) with index.json
    in1k_t04/conditions/<c>_s<k>/         dataset.json and splits/{V_evidence,V_check,V_select}.npz

Every builder refuses an existing target and runs on dev rows only. ``idx`` of a split is the validation row number.
The decoy is torchvision's resnet18 initialisation under ``torch.manual_seed(DECOY_SEED)``: the evidence-v1 layout
needs a decoy of the same architecture for every instance, and no other trained network with this layout exists. It is
never shown to an editor in a LEARN session and no reference arm uses it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.data import imagenet_val as V
from weightcode.data import t04 as T
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
from weightcode.evalkit.targets import file_sha256

SCHEMA = "weightcode.dataset.v0"
SOURCE = f"{V.REPO_ID} validation (revision {V.REVISION})"
DEV_INDEX = "dev_index.jsonl"
SOURCE_SEED = 100
DECOY_SEED = 101
PARITY_ROWS = 1000
MODEL_PRIVATE_SCHEMA = "weightcode.t04_model_private.v1"
WEIGHTS_RELPATH = "pretrained/resnet18-f37072fd.pth"


def forward_arch() -> str:
    return ("resnet18_native_u8 (torchvision resnet18 with the IMAGENET1K_V1 weights and their 1000-class head; x / 255 "
            "and the ImageNet normalisation inside the network, on the stored 224 x 224 uint8 center crop)")


def _class_counts(y: np.ndarray) -> list[int]:
    return [int(c) for c in np.bincount(y, minlength=T.N_CLASSES)]


def _coverage(y: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(y, minlength=T.N_CLASSES)
    return {"classes_present": int(np.count_nonzero(counts)), "min": int(counts.min()), "max": int(counts.max()),
            "mean": float(counts.mean())}


def check_d_index(records: Mapping[int, Mapping[str, Any]], d_index_path: Path) -> dict[str, Any]:
    """Each D row's stored-bytes sha256 and label against T03's D index (runs/t03/d_index.jsonl)."""
    lines = Path(d_index_path).read_text(encoding="utf-8").splitlines()
    header, rows = json.loads(lines[0]), [json.loads(line) for line in lines[1:] if line.strip()]
    if header.get("revision") != V.REVISION or header.get("manifest_sha256") != V.MANIFEST_SHA256:
        raise T.T04DataError(f"{d_index_path}: another source revision or manifest")
    mismatched = [r["global_index"] for r in rows
                  if records[int(r["global_index"])]["image_sha256"] != r["image_sha256"]
                  or records[int(r["global_index"])]["label"] != r["label"]]
    if len(rows) != V.N_D or mismatched:
        raise T.T04DataError(f"D index: {len(rows)} rows, {len(mismatched)} mismatched (first {mismatched[:5]})")
    return {"path": str(d_index_path), "sha256": file_sha256(Path(d_index_path)), "rows": len(rows), "match": True}


def build_clean(data_dir: Path, run_info: Mapping[str, Any], d_index_path: Path | None = None,
                verify_shards: bool = True) -> dict[str, Any]:
    """Decode the 12,000 dev rows, write the four clean splits, forward.py, dataset.json, the dev index and the replay
    draw. Refuses an existing store."""
    from weightcode.evalkit.companions import build_replay
    from weightcode.models.resnet18native import render_forward_u8

    data_dir = Path(data_dir)
    root = dataset_dir(T.DATASET)
    if dataset_json_path(T.DATASET).exists() or root.exists():
        raise T.T04DataError(f"{root} exists; nothing written")
    manifest_files = V.verify_manifest(data_dir) if verify_shards else None
    classes_path = data_dir / V.CLASSES_FILE
    classes = V.parse_classes(classes_path)
    names, changes = T.class_names(classes, T.forbidden_name_tokens())
    records: dict[int, dict[str, Any]] = {}
    crops: dict[int, np.ndarray] = {}
    for record in T.read_dev_records(data_dir / T.SHARD_DIR, T.dev_rows()):
        crops[record.global_index] = T.center_crop_u8(record.image_bytes)
        records[record.global_index] = record.index_entry()
    if sorted(records) != [int(r) for r in T.dev_rows()]:
        raise T.T04DataError("the decoded rows are not the dev rows")
    labels = {row: int(entry["label"]) for row, entry in records.items()}
    splits = T.split_rows(labels)
    row_record = T.row_record(labels)
    d_check = check_d_index(records, d_index_path) if d_index_path is not None else None
    root.mkdir(parents=True)
    split_meta = {}
    for split in T.SPLITS:
        rows = splits[split]
        arrays = {"x": np.stack([crops[int(r)] for r in rows]),
                  "y": np.array([records[int(r)]["label"] for r in rows], dtype=np.int64),
                  "idx": rows.astype(np.int64)}
        digest = save_split(split_path(T.DATASET, split), arrays)
        split_meta[split] = {"path": f"splits/{split}.npz", "n": int(len(rows)), "sha256": digest,
                             "rows_sha256": T.rows_sha256(rows), "class_counts": _class_counts(arrays["y"]),
                             "coverage": _coverage(arrays["y"])}
    index_lines = [json.dumps({"schema": "weightcode.t04.dev_index.v1", "revision": V.REVISION,
                               "manifest_sha256": V.MANIFEST_SHA256}, sort_keys=True)]
    index_lines += [json.dumps(records[int(r)], sort_keys=True) for r in T.dev_rows()]
    (root / DEV_INDEX).write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    forward_path(T.DATASET).write_text(render_forward_u8(T.N_CLASSES), encoding="utf-8")
    test_rows = T.test_rows()
    manifest = {
        "schema": SCHEMA,
        "dataset": T.DATASET,
        "source": SOURCE,
        "class_names": names,
        "class_name_rule": "weightcode.data.t04.class_names: first synonym of classes.py; a shared first synonym becomes "
                           "the full entry; a name with a forbidden token becomes the first clean synonym, else the "
                           "WordNet id",
        "class_name_changes": changes,
        "image_shape": list(T.IMAGE_SHAPE),
        "dtype": "uint8",
        "layout": "NCHW",
        "split_seed": T.SPLIT_SEED,
        "split_rule": T.SPLIT_RULE,
        "split_sizes": {split: size for split, size in T.SPLIT_SIZES},
        "rows": row_record,
        "splits": split_meta,
        "test": {"location": "not materialised: the sealed rows are decoded only at the single sealed read, locally, "
                             "from the pinned shards",
                 "rows": row_record["test_rule"], "n": int(len(test_rows)),
                 "sha256": T.rows_sha256(test_rows),
                 "sha256_definition": "sha256 of the test row numbers as little-endian int64 (no image is stored)"},
        "image_pipeline": {"crop": "PIL RGB, torchvision resize shorter side 256 (bilinear, antialias), center crop "
                                   "224; uint8 (weightcode.data.t04.center_crop_u8)",
                           "libraries": T.image_libraries()},
        "forward_path": "forward.py",
        "forward_sha256": file_sha256(forward_path(T.DATASET)),
        "forward_arch": forward_arch(),
        "student_preprocessing": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                                  "inside_forward": "x / 255, then (x - mean) / std per channel"},
        "source_provenance": {"repo_id": V.REPO_ID, "revision": V.REVISION, "manifest_sha256": V.MANIFEST_SHA256,
                              "classes_py_sha256": file_sha256(classes_path),
                              "shards_verified": manifest_files is not None},
        "dev_index": {"path": DEV_INDEX, "sha256": file_sha256(root / DEV_INDEX)},
        "d_index_check": d_check,
        "created_by": dict(run_info),
    }
    write_json(dataset_json_path(T.DATASET), manifest)
    manifest["replay"] = build_replay(T.DATASET)
    return manifest


def _checked_clean(split: str) -> dict[str, np.ndarray]:
    meta = read_dataset_json(T.DATASET)
    arrays = load_split(T.DATASET, split)
    if split_sha256(arrays) != meta["splits"][split]["sha256"]:
        raise T.T04DataError(f"clean {split} differs from its dataset.json hash")
    T.check_dev_rows(arrays["idx"])
    return arrays


def build_condition(corruption: str, severity: int, official_path: Path, run_info: Mapping[str, Any],
                    choice_record: Mapping[str, Any]) -> dict[str, Any]:
    """Write one condition from the clean dev splits (official function, np.uint8, JPEG q85, decode).
    ``choice_record`` is the severity rule's output on D for this corruption; its chosen severity must be
    ``severity``."""
    name = T.condition_name(corruption, severity)
    if choice_record.get("chosen") != int(severity):
        raise T.T04DataError(f"the severity rule chose {choice_record.get('chosen')} for {corruption}, not {severity}")
    if dataset_json_path(T.DATASET, name).exists() or dataset_dir(T.DATASET, name).exists():
        raise T.T04DataError(f"{dataset_dir(T.DATASET, name)} exists; nothing written")
    clean = read_dataset_json(T.DATASET)
    official = T.load_official(official_path)
    constants = T.check_official_constants(official)
    splits = {}
    for split in T.CONDITION_SPLITS:
        base = _checked_clean(split)
        x = T.corrupt_split(official, corruption, severity, base["x"], base["idx"])
        digest = save_split(split_path(T.DATASET, split, name), {"x": x, "y": base["y"], "idx": base["idx"]})
        splits[split] = {"path": f"splits/{split}.npz", "n": int(len(x)), "sha256": digest,
                         "class_counts": clean["splits"][split]["class_counts"],
                         "clean_sha256": clean["splits"][split]["sha256"]}
    c = T.CONSTANTS[corruption][int(severity) - 1]
    manifest = {
        "schema": SCHEMA,
        "dataset": T.DATASET,
        "condition": {
            "name": name, "kind": "corruption", "corruption": corruption, "family": T.FAMILIES[corruption],
            "severity": int(severity), "constants": f"c = {c}", "constants_check": constants,
            "source": {"repository": T.OFFICIAL_REPOSITORY, "file": T.OFFICIAL_FILE, "commit": T.OFFICIAL_COMMIT,
                       "file_sha256": T.OFFICIAL_SHA256, "function_sha256": official["_sources"],
                       "run_as": "the official function on the PIL crop, np.uint8 of its output, JPEG quality 85 "
                                 "optimize=True, decoded (weightcode.data.t04.corrupt_crop)"},
            "rng": ({"generator": "numpy global RandomState, np.random.seed(GAUSSIAN_SEED_BASE + row) before each image",
                     "GAUSSIAN_SEED_BASE": T.GAUSSIAN_SEED_BASE} if corruption == "gaussian_noise"
                    else {"generator": "none (deterministic)"}),
            "libraries": T.image_libraries(),
            "severity_choice": dict(choice_record),
            "applied_to": "rows of the clean splits V_evidence, V_check and V_select (same order, y and idx; only x "
                          "changes)",
        },
        "class_names": clean["class_names"],
        "image_shape": clean["image_shape"],
        "dtype": "uint8",
        "layout": "NCHW",
        "split_seed": clean["split_seed"],
        "forward_path": "../../forward.py",
        "forward_sha256": clean["forward_sha256"],
        "splits": splits,
        "test": {"location": clean["test"]["location"], "rows": clean["test"]["rows"], "n": clean["test"]["n"],
                 "sha256": hashlib.sha256(f"{clean['test']['sha256']}:{name}".encode()).hexdigest(),
                 "sha256_definition": "sha256 of '<clean test sha256>:<condition name>' (no image is stored)",
                 "clean_sha256": clean["test"]["sha256"]},
        "created_by": dict(run_info),
    }
    write_json(dataset_json_path(T.DATASET, name), manifest)
    return manifest


def decoy_params(seed: int = DECOY_SEED) -> dict[str, np.ndarray]:
    """torchvision resnet18 (1000 classes) initialised under ``torch.manual_seed(seed)``, as float32 numpy arrays with
    the network's tensor names (num_batches_tracked dropped)."""
    import torch
    import torchvision

    from weightcode.models import resnet18up as M

    torch.manual_seed(int(seed))
    state = torchvision.models.resnet18(weights=None, num_classes=T.N_CLASSES).state_dict()
    params = {name: state[name].detach().to(torch.float32).contiguous().numpy().copy()
              for name in M.param_shapes(T.N_CLASSES)}
    M.check_params(params, T.N_CLASSES)
    return params


def build_models(weights_path: Path, run_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Write seed 100 (the IMAGENET1K_V1 weights unchanged) and seed 101 (the decoy) into models/: model.npz,
    model_spec.json (public), private.json and index.json rows (role dev). Parity of forward.py against CPU torch on
    the first ``PARITY_ROWS`` clean V_select rows; ``vselect_acc`` is forward.py's clean V_select accuracy."""
    import torch

    from weightcode.common.identity import load_params, params_sha256, save_params
    from weightcode.data.store import models_dir
    from weightcode.evalkit.targets import load_dataset_forward, read_dataset_meta
    from weightcode.models import resnet18native as N
    from weightcode.models.export import all_model_ids, new_model_id
    from weightcode.models.parity import check_logit_parity

    root = models_dir(T.DATASET)
    if root.exists():
        raise T.T04DataError(f"{root} exists; nothing written")
    meta = read_dataset_meta(T.DATASET)
    forward, forward_sha, _ = load_dataset_forward(T.DATASET, meta)
    vselect = _checked_clean("V_select")
    weights = {"file": Path(weights_path).name, "sha256": file_sha256(Path(weights_path))}
    students = [(SOURCE_SEED, N.imagenet_params(Path(weights_path)),
                 {"origin": "torchvision ResNet18_Weights.IMAGENET1K_V1, unchanged", "weights": weights}),
                (DECOY_SEED, decoy_params(DECOY_SEED),
                 {"origin": f"torchvision resnet18 initialisation under torch.manual_seed({DECOY_SEED}); the instance "
                            "builder's decoy only"})]
    root.mkdir(parents=True)
    index, out = [], []
    for seed, params, origin in students:
        model_id = new_model_id(all_model_ids(T.DATASET))
        staging = root / f".staging-{model_id}"
        staging.mkdir()
        digest = save_params(staging / "model.npz", params)
        if params_sha256(load_params(staging / "model.npz")) != digest:
            raise RuntimeError("reloaded parameters hash differently")
        x_par = np.ascontiguousarray(vselect["x"][:PARITY_ROWS])
        ours = np.asarray(forward(params, x_par), dtype=np.float64)
        ref64 = N.eval_logits_u8(N.build_resnet18native_u8(params, torch.float64), x_par)
        ref32 = N.eval_logits_u8(N.build_resnet18native_u8(params, torch.float32), x_par)
        parity = {"rows": f"the first {PARITY_ROWS} clean V_select rows", **check_logit_parity(ours, ref64, ref32),
                  "float32_argmax_disagreements": int(np.count_nonzero(ours.argmax(1) != ref32.argmax(1)))}
        pred = np.asarray(forward(params, vselect["x"])).argmax(1)
        acc = float(np.mean(pred == vselect["y"]))
        write_json(staging / "model_spec.json",
                   N.build_resnet18native_u8_spec(model_id, meta["class_names"], params))
        write_json(staging / "private.json", {
            "schema": MODEL_PRIVATE_SCHEMA, "model_id": model_id, "dataset": T.DATASET, "condition": None,
            "seed": seed, "arch": N.ARCH_U8, "role": "dev", "store_role": "dev", "split_role": "dev",
            **origin, "params_sha256": digest, "n_parameters": int(sum(v.size for v in params.values())),
            "n_tensors": len(params), "forward_sha256": forward_sha, "parity": parity,
            "readouts": {"clean_vselect_acc": acc, "n_vselect": int(len(pred))}, **dict(run_info)})
        staging.rename(root / model_id)
        entry = {"model_id": model_id, "seed": seed, "arch": N.ARCH_U8, "role": "dev", "params_sha256": digest,
                 "vselect_acc": acc}
        index.append(entry)
        write_json(root / "index.json", index)
        out.append({**entry, "parity": parity})
    return out


def accuracy_table(params: Mapping[str, np.ndarray], forward: Any, conditions: Sequence[str],
                   splits: Sequence[str] = T.CONDITION_SPLITS) -> dict[str, Any]:
    """forward.py accuracy of ``params`` on the clean and each condition's dev splits (hash-checked)."""
    table: dict[str, Any] = {}
    for condition in (None, *conditions):
        meta = read_dataset_json(T.DATASET, condition)
        row = {}
        for split in splits:
            arrays = load_split(T.DATASET, split, condition)
            if split_sha256(arrays) != meta["splits"][split]["sha256"]:
                raise T.T04DataError(f"{condition or 'clean'} {split} differs from its dataset.json hash")
            pred = np.asarray(forward(params, arrays["x"])).argmax(1)
            row[split] = {"acc": float(np.mean(pred == arrays["y"])), "n": int(len(pred)),
                          "sha256": meta["splits"][split]["sha256"]}
        table[condition or "clean"] = row
    return table


__all__ = ["DECOY_SEED", "DEV_INDEX", "MODEL_PRIVATE_SCHEMA", "PARITY_ROWS", "SCHEMA", "SOURCE", "SOURCE_SEED",
           "WEIGHTS_RELPATH", "accuracy_table", "build_clean", "build_condition", "build_models", "check_d_index",
           "decoy_params", "forward_arch"]
