"""Synthetic data store shared by the evalkit target, instance and sealed tests.

The toy classifier (a tiny ReLU MLP on 1x4x4 uint8 images) predicts an image's "look" class, encoded by one bright
pixel among pixels 0..3. The true class sits in pixel 4 + y, which the base models ignore. Making some class-a images
look like class b plants confusions with exact counts per split, so the target rule, the evidence draw and the sealed
readouts can be checked by hand. A child that reads pixel 4 fixes every (0 -> 1) error and breaks nothing.

Everything is written under pytest tmp dirs with monkeypatched WEIGHTCODE_* variables; nothing touches real locations.
Other test modules import ``build_toy_store`` and the ``toy_store`` fixture from here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from weightcode.common.identity import params_sha256, save_params
from weightcode.evalkit.instances import DEFAULT_DECOYS, BuildOutcome, build_dataset_instances, build_deployment_instances
from weightcode.evalkit.patch_check import load_forward, predict
from weightcode.evalkit.targets import DEFAULT_DEV_SEEDS, DEFAULT_MAIN_SEEDS, SeedSelection, run_target_selection

CLASSES = ["alpha", "beta", "gamma", "delta"]
N_CLASSES = len(CLASSES)
# split -> (images per class, {(a, b): number of class-a images that look like b})
PLAN: dict[str, tuple[int, dict[tuple[int, int], int]]] = {
    "train": (10, {}),
    "V_select": (100, {(0, 1): 30, (1, 0): 25, (2, 3): 25, (3, 2): 20}),
    "V_evidence": (60, {(0, 1): 20, (1, 0): 18, (2, 3): 10, (3, 2): 17}),
    "V_check": (100, {(0, 1): 12, (1, 0): 5, (2, 3): 12, (3, 2): 11}),
    "test": (150, {(0, 1): 20, (1, 0): 10, (2, 3): 15, (3, 2): 12}),
}
# Ranking: (0,1) 30, (1,0) 25, (2,3) 25, (3,2) 20. (1,0) fails V_check (5 < 10), (2,3) fails V_evidence (10 < 16).
EXPECTED_CHOSEN = [(0, 1), (3, 2)]
# Colored stores: color is the look class, except that this many of a pair's planted test-split confusions carry
# the smallest class outside {a, b} as color (shape-driven errors, the "other" branch of the color split).
TEST_COLOR_OTHER: dict[tuple[int, int], int] = {(0, 1): 8, (3, 2): 4}
# Base errors per split: the planted confusions (the base model predicts the look class).
BASE_ERRORS = {split: sum(confusions.values()) for split, (_, confusions) in PLAN.items()}

# The development seed plan and the five-seed extension the PREREG fixes (decoys 2 -> 3 -> 4 -> 2).
DEV_PLAN = SeedSelection(main=DEFAULT_MAIN_SEEDS, dev=DEFAULT_DEV_SEEDS)
FIVE_SEEDS = SeedSelection(main=(0, 1, 2, 3, 4), dev=DEFAULT_DEV_SEEDS)
FIVE_SEED_DECOYS = {**DEFAULT_DECOYS, 2: 3, 3: 4, 4: 2}

FORWARD_SOURCE = '''"""Reference forward of the synthetic test classifier: two-layer ReLU MLP on raw uint8 images."""

import numpy as np


def forward(params, x_uint8):
    h = x_uint8.reshape(len(x_uint8), -1).astype(np.float32) / np.float32(255.0)
    h = np.maximum(h @ params["fc1.weight"].T + params["fc1.bias"], np.float32(0.0))
    return (h @ params["fc2.weight"].T + params["fc2.bias"]).astype(np.float32)
'''


def make_split(
    n_per_class: int,
    confusions: dict[tuple[int, int], int],
    seed: int,
    offset: int,
    colored: bool,
    color_other: dict[tuple[int, int], int] | None = None,
) -> dict[str, np.ndarray]:
    """One split with planted confusions; ``color`` (colored stores) is the look class unless ``color_other``."""
    labels: list[int] = []
    looks: list[int] = []
    for cls in range(N_CLASSES):
        planted = 0
        for (a, b), count in sorted(confusions.items()):
            if a == cls:
                labels += [cls] * count
                looks += [b] * count
                planted += count
        labels += [cls] * (n_per_class - planted)
        looks += [cls] * (n_per_class - planted)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    y = np.asarray(labels, dtype=np.int64)[order]
    look = np.asarray(looks, dtype=np.int64)[order]
    n = len(y)
    flat = np.zeros((n, 16), dtype=np.uint8)
    flat[np.arange(n), look] = 255
    flat[np.arange(n), 4 + y] = 255
    flat[:, 8:] = rng.integers(0, 51, size=(n, 8), dtype=np.uint8)
    arrays = {"x": flat.reshape(n, 1, 4, 4), "y": y, "idx": np.arange(offset, offset + n, dtype=np.int64)}
    if colored:
        color = look.copy()
        for (a, b), count in sorted((color_other or {}).items()):
            rows = np.flatnonzero((y == a) & (look == b))[:count]
            color[rows] = min(c for c in range(N_CLASSES) if c not in (a, b))
        arrays["color"] = color
    return arrays


def toy_params(tag: int) -> dict[str, np.ndarray]:
    """Base classifier: hidden units 0..3 copy the look pixels, output c reads unit c; small random rest."""
    rng = np.random.default_rng(1000 + tag)
    w1 = rng.normal(scale=0.01, size=(8, 16))
    w1[:4, :4] += 10.0 * np.eye(4)
    w2 = rng.normal(scale=0.01, size=(4, 8))
    w2[:, :4] += np.eye(4)
    return {
        "fc1.weight": w1.astype(np.float32),
        "fc1.bias": np.zeros(8, dtype=np.float32),
        "fc2.weight": w2.astype(np.float32),
        "fc2.bias": np.zeros(4, dtype=np.float32),
    }


def fixing_child(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Hidden unit 4 reads the true-class-0 pixel and pushes logit 0: fixes every (0 -> 1) error, breaks nothing."""
    child = {key: value.copy() for key, value in params.items()}
    child["fc1.weight"][4, :] = 0.0
    child["fc1.weight"][4, 4] = 20.0
    child["fc2.weight"][:, 4] = 0.0
    child["fc2.weight"][0, 4] = 1.0
    return child


def breaking_child(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A large bias on class 2: almost everything becomes class 2."""
    child = {key: value.copy() for key, value in params.items()}
    child["fc2.bias"][2] += 100.0
    return child


def _model_spec(model_id: str) -> dict[str, object]:
    return {
        "schema": "weightcode.model_spec.v0",
        "model_id": model_id,
        "input": {"shape": [1, 4, 4], "dtype": "uint8", "preprocess": "x/255", "layout": "NCHW, flatten row-major"},
        "classes": CLASSES,
        "layers": [
            {"name": "fc1", "op": "linear", "weight": "fc1.weight", "bias": "fc1.bias", "shape_w": [8, 16]},
            {"op": "relu"},
            {"name": "fc2", "op": "linear", "weight": "fc2.weight", "bias": "fc2.bias", "shape_w": [4, 8]},
        ],
        "output": "logits; prediction = argmax",
    }


@dataclass
class ToyStore:
    """Locations and model rows of one synthetic store."""

    root: Path
    dataset: str
    models: list[dict[str, object]]

    @property
    def private(self) -> Path:
        return self.root / "private"

    @property
    def sealed(self) -> Path:
        return self.root / "sealed"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    def ids(self, role: str) -> dict[int, str]:
        """Model ids of one role by seed."""
        return {int(row["seed"]): str(row["model_id"]) for row in self.models if row["role"] == role}

    def activate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Point WEIGHTCODE_PRIVATE, WEIGHTCODE_SEALED and WEIGHTCODE_RUNS at this store."""
        monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(self.private))
        monkeypatch.setenv("WEIGHTCODE_SEALED", str(self.sealed))
        monkeypatch.setenv("WEIGHTCODE_RUNS", str(self.runs))


def build_toy_store(
    root: Path,
    *,
    dataset: str = "toyset",
    colored: bool = False,
    main_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    dev_seeds: tuple[int, ...] = (100, 101),
    ceiling_seeds: tuple[int, ...] = (),
) -> ToyStore:
    """Write splits, the sealed test set, forward.py, dataset.json and models in the data agent's layout."""
    private, sealed, runs = root / "private", root / "sealed", root / "runs"
    data = private / dataset
    (data / "splits").mkdir(parents=True)
    (sealed / dataset).mkdir(parents=True)
    runs.mkdir(parents=True)
    offset = 0
    splits = {}
    for number, (split, (n_per_class, confusions)) in enumerate(PLAN.items()):
        other = TEST_COLOR_OTHER if split == "test" else None
        arrays = make_split(
            n_per_class, confusions, seed=10 + number, offset=offset, colored=colored, color_other=other
        )
        offset += len(arrays["y"])
        splits[split] = arrays
        target = sealed / dataset / "test.npz" if split == "test" else data / "splits" / f"{split}.npz"
        np.savez(target, **arrays)
    (data / "forward.py").write_text(FORWARD_SOURCE)
    forward, forward_sha = load_forward(data / "forward.py")
    meta = {
        "dataset": dataset,
        "class_names": CLASSES,
        "image_shape": [1, 4, 4],
        "norm": {"mean": [0.1], "std": [0.3]},
        "split_seed": 20261103,
        "forward_sha256": forward_sha,
        "splits": {
            split: {"path": f"splits/{split}.npz", "n": int(len(arrays["y"])), "sha256": params_sha256(arrays)}
            for split, arrays in splits.items()
            if split != "test"
        },
        "test": {"location": f"$WEIGHTCODE_SEALED/{dataset}/test.npz", "sha256": params_sha256(splits["test"])},
    }
    (data / "dataset.json").write_text(json.dumps(meta, indent=2))
    rows: list[dict[str, object]] = []
    for role, seeds, tag_base in (("base", main_seeds, 0), ("ceiling", ceiling_seeds, 500), ("dev", dev_seeds, 0)):
        for seed in seeds:
            model_id = "m" + hashlib.sha256(f"{dataset}|{role}|{seed}".encode()).hexdigest()[:8]
            params = toy_params(tag_base + seed)
            model_dir = data / "models" / model_id
            model_dir.mkdir(parents=True)
            digest = save_params(model_dir / "model.npz", params)
            (model_dir / "model_spec.json").write_text(json.dumps(_model_spec(model_id), indent=2))
            pred = predict(forward, params, splits["V_select"]["x"], 64)
            y_select = splits["V_select"]["y"]
            acc = float(np.mean(pred == y_select))
            confusion = np.bincount(y_select * N_CLASSES + pred, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)
            private = {"seed": seed, "role": role, "params_sha256": digest, "vselect_confusion": confusion.tolist()}
            (model_dir / "private.json").write_text(json.dumps(private))
            rows.append(
                {"model_id": model_id, "seed": seed, "arch": "toy_mlp", "role": role, "params_sha256": digest,
                 "vselect_acc": acc}
            )
    (data / "models" / "index.json").write_text(json.dumps(rows, indent=2))
    return ToyStore(root=root, dataset=dataset, models=rows)


# A synthetic named deployment condition in the data agent's condition layout. Its leak tokens are toyblur_s2,
# toyblur and s2 plus the generic corruption words.
CONDITION = "toyblur_s2"
CONDITION_CORRUPTION = "toyblur"


def corrupt_split(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The toy corruption: x only. Every image's noise pixels brighten by 3, and every row whose idx is a multiple of
    5 looks one class further on (its bright look pixel moves from l to (l + 1) % 4), which changes the base errors.
    The true-class pixels, y, idx and any other array stay as they are."""
    flat = arrays["x"].reshape(len(arrays["x"]), -1).copy()
    flat[:, 8:] = np.minimum(flat[:, 8:].astype(np.int64) + 3, 255).astype(np.uint8)
    rows = np.flatnonzero(arrays["idx"] % 5 == 0)
    look = flat[rows, :4].argmax(axis=1)
    flat[rows, look] = 0
    flat[rows, (look + 1) % 4] = 255
    return {**arrays, "x": flat.reshape(arrays["x"].shape)}


def condition_dir(store: ToyStore, name: str = CONDITION) -> Path:
    """``$WEIGHTCODE_PRIVATE/<dataset>/conditions/<name>``."""
    return store.private / store.dataset / "conditions" / name


def condition_test_path(store: ToyStore, name: str = CONDITION) -> Path:
    """``$WEIGHTCODE_SEALED/<dataset>/conditions/<name>/test.npz``."""
    return store.sealed / store.dataset / "conditions" / name / "test.npz"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """All arrays of an .npz file."""
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def add_condition(store: ToyStore, name: str = CONDITION, corruption: str = CONDITION_CORRUPTION) -> dict[str, object]:
    """Write a corrupted copy of every clean split and of the test set with its dataset.json; return that manifest."""
    data = store.private / store.dataset
    clean = json.loads((data / "dataset.json").read_text())
    folder = condition_dir(store, name)
    (folder / "splits").mkdir(parents=True)
    condition_test_path(store, name).parent.mkdir(parents=True)
    splits = {}
    for split in clean["splits"]:
        corrupted = corrupt_split(load_npz(data / "splits" / f"{split}.npz"))
        np.savez(folder / "splits" / f"{split}.npz", **corrupted)
        splits[split] = {"path": f"splits/{split}.npz", "n": int(len(corrupted["y"])),
                         "sha256": params_sha256(corrupted), "clean_sha256": clean["splits"][split]["sha256"]}
    test = corrupt_split(load_npz(store.sealed / store.dataset / "test.npz"))
    np.savez(condition_test_path(store, name), **test)
    meta = {
        "schema": "weightcode.dataset.v0",
        "dataset": store.dataset,
        "condition": {"name": name, "kind": "corruption", "corruption": corruption, "severity": 2, "constant": 3,
                      "rng": None},
        "class_names": clean["class_names"],
        "image_shape": clean["image_shape"],
        "norm": clean["norm"],
        "forward_sha256": clean["forward_sha256"],
        "splits": splits,
        "test": {"location": f"$WEIGHTCODE_SEALED/{store.dataset}/conditions/{name}/test.npz",
                 "sha256": params_sha256(test), "clean_sha256": clean["test"]["sha256"]},
    }
    write_condition_meta(store, meta, name)
    return meta


def write_condition_meta(store: ToyStore, meta: dict[str, object], name: str = CONDITION) -> None:
    """Replace a condition's dataset.json."""
    (condition_dir(store, name) / "dataset.json").write_text(json.dumps(meta, indent=2))


def build_condition(
    store: ToyStore,
    roles: tuple[str, ...] = ("dev", "main"),
    name: str = CONDITION,
) -> list[BuildOutcome]:
    """evidence-v1 instances of a named condition, development seed plan and default decoys."""
    return build_deployment_instances(store.dataset, name, selection=DEV_PLAN, decoys=DEFAULT_DECOYS, split_roles=roles)


def build_v0(
    store: ToyStore,
    roles: tuple[str, ...] = ("dev", "main"),
    selection: SeedSelection = DEV_PLAN,
    decoys: dict[int, int] = DEFAULT_DECOYS,
) -> list[BuildOutcome]:
    """Target selection, then evidence-v0 (target pair) instances."""
    run_target_selection(store.dataset, selection=selection, split_roles=roles)
    return build_dataset_instances(store.dataset, selection=selection, decoys=decoys, split_roles=roles)


def build_v1(
    store: ToyStore,
    roles: tuple[str, ...] = ("dev", "main"),
    selection: SeedSelection = DEV_PLAN,
    decoys: dict[int, int] = DEFAULT_DECOYS,
) -> list[BuildOutcome]:
    """evidence-v1 instances of the condition "deploy"."""
    return build_deployment_instances(store.dataset, "deploy", selection=selection, decoys=decoys, split_roles=roles)


@pytest.fixture
def toy_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ToyStore:
    """Default store (5 main seeds, dev seeds 100 and 101, no ceiling), activated."""
    store = build_toy_store(tmp_path / "store")
    store.activate(monkeypatch)
    return store


def test_toy_store_plants_exact_confusions(tmp_path: Path) -> None:
    store = build_toy_store(tmp_path / "store", colored=True)
    forward, _ = load_forward(store.private / store.dataset / "forward.py")
    params = toy_params(0)
    for split, (n_per_class, confusions) in PLAN.items():
        path = (
            store.sealed / store.dataset / "test.npz"
            if split == "test"
            else store.private / store.dataset / "splits" / f"{split}.npz"
        )
        with np.load(path) as archive:
            x, y, color = archive["x"], archive["y"], archive["color"]
        look = x.reshape(len(x), -1)[:, :4].argmax(axis=1)
        pred = predict(forward, params, x, 64)
        assert np.array_equal(pred, look), "base prediction must equal the planted look class"
        planted_other = sum(TEST_COLOR_OTHER.values()) if split == "test" else 0
        assert int(np.count_nonzero(color != look)) == planted_other
        assert np.bincount(y, minlength=N_CLASSES).tolist() == [n_per_class] * N_CLASSES
        for a in range(N_CLASSES):
            for b in range(N_CLASSES):
                if a != b:
                    assert int(np.count_nonzero((y == a) & (pred == b))) == confusions.get((a, b), 0)
        fixed = predict(forward, fixing_child(params), x, 64)
        assert np.all(fixed[y == 0] == 0) and np.array_equal(fixed[y != 0], pred[y != 0])
