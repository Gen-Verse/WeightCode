"""A toy ``cifar100_fscil`` store for the E9 tests (rule fscil-v1), importable by other test modules::

    from fscil_toy import toy_fscil  # noqa: F401  (the pytest fixture)
    from fscil_toy import build_toy_fscil_store, random_resnet20

The store has the real class and session structure (60 base classes, sessions 2 to 9 of 5 classes) and the real class
names, 3 x 8 x 8 uint8 images, the real ResNet-20 forward (``render_resnet20_forward`` at 8 x 8) and one random
ResNet-20 ``fscil_base`` student of seed 100 in the fewshot store. Rows per class:

* base classes: train 4, old_select 2, old_check 2, old_replay 2;
* new classes: V_evidence 5 (the shots, stored in session order then class order), V_check 2, V_select 2,
  ref_pool 1 (read only by the all-class student's training rows).

So at session t the evidence holds new 25, old 64 (N_REPLAY of the 120 old_replay rows) and prev 25 (t - 2) rows; the
check has new 10 and old 120 + 10 (t - 2) rows; the readout has new 10, prev 10 (t - 2) and old 120 rows. The sealed
test is not written: nothing here reads it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from weightcode.common.identity import Params, params_sha256, save_params
from weightcode.data import store
from weightcode.evalkit import fscil as FS
from weightcode.models.export import fewshot_models_dir, write_fewshot_index
from weightcode.models.forward_gen import render_resnet20_forward, write_forward

CIFAR100_FINE_NAMES: tuple[str, ...] = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy",
    "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest",
    "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper",
    "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm",
)
"""CIFAR-100 fine label names in label order (the official meta file's fine_label_names)."""

SHAPE = (3, 8, 8)
MEAN, STD = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
PER_BASE = {"train": 4, "old_select": 2, "old_check": 2, "old_replay": 2}
PER_NEW = {"V_evidence": 5, "V_check": 2, "V_select": 2, "ref_pool": 1}
TOY_SEED = 100


@dataclass(frozen=True)
class ToyFscil:
    """What the toy store holds: the base student's id, parameters and hash."""

    dataset: str
    base_model_id: str
    base_params: Params
    base_params_sha256: str
    root: Path


def random_resnet20(rng: np.random.Generator, shapes: dict[str, tuple[int, ...]]) -> Params:
    """Random float32 ResNet-20 parameters with BatchNorm statistics away from identity (so BN eval matters)."""
    params = {}
    for name, shape in shapes.items():
        if name.endswith("running_var"):
            value = rng.uniform(0.5, 1.5, shape)
        elif name.endswith("running_mean") or name.endswith(".bias"):
            value = 0.1 * rng.standard_normal(shape)
        elif len(shape) == 1:  # BN weight
            value = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            fan_out = shape[0] * int(np.prod(shape[2:])) if len(shape) == 4 else shape[1]
            value = rng.standard_normal(shape) * np.sqrt(2.0 / fan_out)
        params[name] = np.asarray(value, dtype=np.float32)
    return params


def toy_spec(shapes: dict[str, tuple[int, ...]], model_id: str) -> dict:
    """A minimal ResNet-20 model_spec.json whose last layer is the cosine head (what evalkit.fscil reads). It carries
    its model_id, as the store's real specs do (``build_resnet20_spec``); instances must leave it out."""
    return {
        "schema": "weightcode.model_spec.resnet20.v1",
        "model_id": model_id,
        "arch": FS.ARCH,
        "input_shape": list(SHAPE),
        "classes": list(CIFAR100_FINE_NAMES),
        "n_classes": FS.N_CLASSES,
        "tensors": {name: list(shape) for name, shape in shapes.items()},
        "layers": [
            {"op": "resnet20_encoder", "description": "toy stand-in for the stem, 9 blocks and the pool"},
            {"op": FS.HEAD_OP, "weight": FS.HEAD_TENSOR, "shape": [FS.N_CLASSES, 64], "scale": 16.0, "eps": 1e-12},
        ],
    }


def _forward_module(path: Path):
    namespace: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


def build_toy_fscil_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToyFscil:
    """Write the toy store under ``tmp_path`` and point WEIGHTCODE_PRIVATE / WEIGHTCODE_SEALED / WEIGHTCODE_RUNS at it."""
    for variable in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED", "WEIGHTCODE_RUNS"):
        (tmp_path / variable).mkdir()
        monkeypatch.setenv(variable, str(tmp_path / variable))
    rng = np.random.default_rng(9)
    dataset = FS.DATASET
    parts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {split: [] for split in (*PER_BASE, *PER_NEW)}
    for label in range(FS.N_BASE_CLASSES):
        for split, count in PER_BASE.items():
            parts[split].append((np.full(count, label), rng.integers(0, 256, (count, *SHAPE))))
    for session in FS.SESSIONS:  # V_evidence: session order, then class order
        for label in FS.session_classes(session):
            for split, count in PER_NEW.items():
                parts[split].append((np.full(count, label), rng.integers(0, 256, (count, *SHAPE))))
    total = sum(len(y) for rows in parts.values() for y, _ in rows)
    official = rng.permutation(total)
    start = 0
    meta_splits = {}
    for split, rows in parts.items():
        y = np.concatenate([y for y, _ in rows]).astype(np.int64)
        x = np.concatenate([x for _, x in rows]).astype(np.uint8)
        idx = official[start : start + len(y)].astype(np.int64)
        start += len(y)
        arrays = {"x": x, "y": y, "idx": idx}
        sha = store.save_split(store.split_path(dataset, split), arrays)
        meta_splits[split] = {"path": f"splits/{split}.npz", "sha256": sha, "n": int(len(y)),
                              "class_counts": np.bincount(y, minlength=FS.N_CLASSES).astype(int).tolist()}
    forward_sha = write_forward(store.forward_path(dataset), render_resnet20_forward(
        MEAN, STD, FS.N_CLASSES, 16.0, 1e-5, 1e-12, input_shape=SHAPE))
    store.write_json(store.dataset_json_path(dataset), {
        "schema": "weightcode.dataset.fscil.v1", "dataset": dataset, "task": "fscil", "toy": True,
        "class_names": list(CIFAR100_FINE_NAMES), "n_classes": FS.N_CLASSES, "image_shape": list(SHAPE),
        "forward_sha256": forward_sha, "forward_path": "forward.py",
        "normalization": {"mean": list(MEAN), "std": list(STD)}, "splits": meta_splits,
    })
    shapes = _forward_module(store.forward_path(dataset))["expected_shapes"]()
    params = random_resnet20(rng, shapes)
    model_id = "m0f5c0a11"
    folder = fewshot_models_dir(dataset) / model_id
    folder.mkdir(parents=True)
    digest = save_params(folder / "model.npz", params)
    (folder / "model_spec.json").write_text(json.dumps(toy_spec(shapes, model_id), indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
    (folder / "private.json").write_text(json.dumps({
        "role": FS.BASE_ROLE, "train_classes": list(range(FS.N_BASE_CLASSES)), "seed": TOY_SEED, "toy": True,
    }, indent=2) + "\n", encoding="utf-8")
    write_fewshot_index(dataset, [{
        "model_id": model_id, "seed": TOY_SEED, "arch": FS.ARCH, "role": FS.BASE_ROLE, "params_sha256": digest,
        "vselect_acc": 0.0, "train_classes": list(range(FS.N_BASE_CLASSES)), "vselect_acc_train_classes": 0.0,
    }])
    assert digest == params_sha256(params)
    return ToyFscil(dataset=dataset, base_model_id=model_id, base_params=params, base_params_sha256=digest,
                    root=tmp_path)


@pytest.fixture()
def toy_fscil(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToyFscil:
    """The toy cifar100_fscil store (``build_toy_fscil_store``)."""
    return build_toy_fscil_store(tmp_path, monkeypatch)


def add_toy_allclass(toy: ToyFscil, seed: int = TOY_SEED) -> str:
    """Add a random fscil_allclass student (all 100 train classes) to the toy store; returns its model id."""
    rng = np.random.default_rng(11)
    params = {name: (value + 0.05 * rng.standard_normal(value.shape)).astype(np.float32)
              if not name.endswith("running_var") else value.copy() for name, value in toy.base_params.items()}
    model_id = "m0a11c1a5"
    folder = fewshot_models_dir(toy.dataset) / model_id
    folder.mkdir(parents=True)
    digest = save_params(folder / "model.npz", params)
    base_spec = json.loads((fewshot_models_dir(toy.dataset) / toy.base_model_id / "model_spec.json").read_text(
        encoding="utf-8"))
    (folder / "model_spec.json").write_text(json.dumps({**base_spec, "model_id": model_id}, indent=2, sort_keys=True)
                                            + "\n", encoding="utf-8")
    (folder / "private.json").write_text(json.dumps({
        "role": FS.ALLCLASS_ROLE, "train_classes": list(range(FS.N_CLASSES)), "seed": seed, "toy": True,
    }, indent=2) + "\n", encoding="utf-8")
    from weightcode.models.export import read_fewshot_index
    write_fewshot_index(toy.dataset, [*read_fewshot_index(toy.dataset), {
        "model_id": model_id, "seed": seed, "arch": FS.ARCH, "role": FS.ALLCLASS_ROLE, "params_sha256": digest,
        "vselect_acc": 0.0, "train_classes": list(range(FS.N_CLASSES)), "vselect_acc_train_classes": 0.0,
    }])
    return model_id
