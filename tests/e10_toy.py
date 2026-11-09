"""A toy E10 store for the harness tests: the evalkit toy store's splits under the E10 dataset name, a BatchNorm
student and the four E10 conditions.

The student is the evalkit toy MLP with a BatchNorm layer in eval form between its two linear layers (fc1 without a
bias, then bn1 with weight, bias, running_mean and running_var, then ReLU and fc2), so its running statistics are
1-D parameters of model.npz as in the E10 ResNet-20. With running_mean 0, running_var 1 - eps, weight 1 and bias 0
the layer is the identity, so the toy store's planted confusions and its fixing programs carry over. Dev seeds 100
to 105 (the E10 students), one instance per (condition, seed), and the replay draw. The four conditions use the toy
store's one corruption under the E10 names (``weightcode.harness.e10.STREAM_CONDITIONS``). Everything lives under a pytest
tmp directory with WEIGHTCODE_* monkeypatched; nothing touches the real store.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from test_evalkit_instances_store import CLASSES, N_CLASSES, PLAN, ToyStore, add_condition, make_split, toy_params

from weightcode.common.identity import params_sha256, save_params
from weightcode.evalkit.companions import build_replay
from weightcode.evalkit.instances import build_deployment_instances
from weightcode.evalkit.patch_check import load_forward, predict
from weightcode.evalkit.targets import SeedSelection
from weightcode.harness import e10

BN_EPS = 1e-5
DEV_SEEDS = (100, 101, 102, 103, 104, 105)
DECOYS = {100: 101, 101: 100, 102: 103, 103: 102, 104: 105, 105: 104}
SELECTION = SeedSelection(main=(), dev=DEV_SEEDS)
FORWARD_SOURCE = '''"""Reference forward of the synthetic test classifier: linear, eval BatchNorm, ReLU, linear."""

import numpy as np

EPS = 1e-05


def forward(params, x_uint8):
    h = x_uint8.reshape(len(x_uint8), -1).astype(np.float32) / np.float32(255.0)
    h = h @ params["fc1.weight"].T
    scale = params["bn1.weight"] / np.sqrt(params["bn1.running_var"] + np.float32(EPS))
    h = (h - params["bn1.running_mean"]) * scale + params["bn1.bias"]
    h = np.maximum(h, np.float32(0.0))
    return (h @ params["fc2.weight"].T + params["fc2.bias"]).astype(np.float32)
'''


def bn_params(tag: int) -> dict[str, np.ndarray]:
    """The toy MLP's weights around an identity BatchNorm layer (module docstring)."""
    base = toy_params(tag)
    return {
        "fc1.weight": base["fc1.weight"],
        "bn1.weight": np.ones(8, dtype=np.float32),
        "bn1.bias": np.zeros(8, dtype=np.float32),
        "bn1.running_mean": np.zeros(8, dtype=np.float32),
        "bn1.running_var": np.full(8, 1.0 - BN_EPS, dtype=np.float32),
        "fc2.weight": base["fc2.weight"],
        "fc2.bias": base["fc2.bias"],
    }


def _model_spec(model_id: str) -> dict[str, object]:
    return {
        "schema": "weightcode.model_spec.v0",
        "model_id": model_id,
        "input": {"shape": [1, 4, 4], "dtype": "uint8", "preprocess": "x/255", "layout": "NCHW, flatten row-major"},
        "classes": CLASSES,
        "layers": [
            {"name": "fc1", "op": "linear", "weight": "fc1.weight", "bias": None, "shape_w": [8, 16]},
            {"name": "bn1", "op": "batchnorm_eval", "weight": "bn1.weight", "bias": "bn1.bias",
             "running_mean": "bn1.running_mean", "running_var": "bn1.running_var", "eps": BN_EPS,
             "computes": "h = (h - running_mean) / sqrt(running_var + eps) * weight + bias"},
            {"op": "relu"},
            {"name": "fc2", "op": "linear", "weight": "fc2.weight", "bias": "fc2.bias", "shape_w": [4, 8]},
        ],
        "output": "logits; prediction = argmax",
    }


def build_store(root: Path) -> ToyStore:
    """Splits, the sealed test set, forward.py, dataset.json and the six BatchNorm dev students of the E10 dataset."""
    dataset = e10.DATASET
    private, sealed, runs = root / "private", root / "sealed", root / "runs"
    data = private / dataset
    (data / "splits").mkdir(parents=True)
    (sealed / dataset).mkdir(parents=True)
    runs.mkdir(parents=True)
    offset = 0
    splits = {}
    for number, (split, (n_per_class, confusions)) in enumerate(PLAN.items()):
        arrays = make_split(n_per_class, confusions, seed=10 + number, offset=offset, colored=False)
        offset += len(arrays["y"])
        splits[split] = arrays
        target = sealed / dataset / "test.npz" if split == "test" else data / "splits" / f"{split}.npz"
        np.savez(target, **arrays)
    (data / "forward.py").write_text(FORWARD_SOURCE)
    forward, forward_sha = load_forward(data / "forward.py")
    meta = {
        "dataset": dataset, "class_names": CLASSES, "image_shape": [1, 4, 4], "norm": {"mean": [0.1], "std": [0.3]},
        "split_seed": 20261103, "forward_sha256": forward_sha,
        "splits": {split: {"path": f"splits/{split}.npz", "n": int(len(arrays["y"])), "sha256": params_sha256(arrays)}
                   for split, arrays in splits.items() if split != "test"},
        "test": {"location": f"$WEIGHTCODE_SEALED/{dataset}/test.npz", "sha256": params_sha256(splits["test"])},
    }
    (data / "dataset.json").write_text(json.dumps(meta, indent=2))
    rows: list[dict[str, object]] = []
    for seed in DEV_SEEDS:
        model_id = "m" + hashlib.sha256(f"{dataset}|dev|{seed}".encode()).hexdigest()[:8]
        params = bn_params(seed)
        model_dir = data / "models" / model_id
        model_dir.mkdir(parents=True)
        digest = save_params(model_dir / "model.npz", params)
        (model_dir / "model_spec.json").write_text(json.dumps(_model_spec(model_id), indent=2))
        y_select = splits["V_select"]["y"]
        pred = predict(forward, params, splits["V_select"]["x"], 64)
        confusion = np.bincount(y_select * N_CLASSES + pred, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)
        (model_dir / "private.json").write_text(json.dumps({"seed": seed, "role": "dev", "params_sha256": digest,
                                                            "vselect_confusion": confusion.tolist()}))
        rows.append({"model_id": model_id, "seed": seed, "arch": "toy_bn_mlp", "role": "dev", "params_sha256": digest,
                     "vselect_acc": float(np.mean(pred == y_select))})
    (data / "models" / "index.json").write_text(json.dumps(rows, indent=2))
    return ToyStore(root=root, dataset=dataset, models=rows)


@dataclass
class E10Store:
    """The toy E10 store and its dev instance ids by (condition, seed)."""

    store: ToyStore
    ids: dict[tuple[str, int], str]

    def stream(self, seed: int = 100) -> list[str]:
        return [self.ids[(condition, seed)] for condition in e10.STREAM_CONDITIONS]


def build_e10_store(root: Path, monkeypatch: pytest.MonkeyPatch) -> E10Store:
    """The store, activated, with the four E10 conditions, their dev instances and the replay draw."""
    store = build_store(root)
    store.activate(monkeypatch)
    ids: dict[tuple[str, int], str] = {}
    for corruption, condition in zip(e10.STREAM_CORRUPTIONS, e10.STREAM_CONDITIONS):
        add_condition(store, name=condition, corruption=corruption)
        for outcome in build_deployment_instances(store.dataset, condition, selection=SELECTION, decoys=DECOYS,
                                                  split_roles=("dev",)):
            ids[(condition, outcome.seed)] = outcome.instance_id
    build_replay(store.dataset, n=8)
    return E10Store(store, ids)


@pytest.fixture
def e10_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> E10Store:
    """Under a neutral directory name, as ``harness_store`` (the sandbox self-test sees the session's host paths)."""
    return build_e10_store(tmp_path_factory.mktemp("hs") / "store", monkeypatch)
