"""PREREG E11 store (weightcode.data.e11) and the student import (weightcode.models.e11_export.import_e11_models) on a fake store."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.common.identity import load_params
from weightcode.data import e11 as D
from weightcode.data.store import (
    dataset_dir,
    dataset_json_path,
    forward_path,
    models_dir,
    save_split,
    split_path,
    write_json,
)
from weightcode.evalkit.targets import file_sha256
from weightcode.models import e11_export as E
from weightcode.models import vitlite as V

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN, STD = [0.49, 0.48, 0.45], [0.25, 0.24, 0.26]
CONDITION = "snow_s5"
SIZES = {"train": 64, "V_select": 30, "V_evidence": 20, "V_check": 20}
RUN = {"git": "test", "code_sha256": "test", "versions": {}}


def _arrays(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"x": rng.integers(0, 256, size=(n, 3, 32, 32), dtype=np.uint8),
            "y": rng.integers(0, 10, size=n).astype(np.int64), "idx": np.arange(n, dtype=np.int64)}


def _manifest(dataset: str, splits: dict, **extra) -> dict:
    return {"schema": "weightcode.dataset.v0", "dataset": dataset, "source": "CIFAR10", "class_names": CLASSES,
            "image_shape": [3, 32, 32], "dtype": "uint8", "layout": "NCHW",
            "norm": {"mean": MEAN, "std": STD, "computed_on": "train"}, "split_seed": 1, "splits": splits,
            "forward_path": "forward.py", "forward_sha256": "0" * 64, "created_by": RUN, **extra}


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for var in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED", "WEIGHTCODE_RUNS"):
        (tmp_path / var).mkdir()
        monkeypatch.setenv(var, str(tmp_path / var))
    clean = {}
    for i, (split, n) in enumerate(SIZES.items()):
        arrays = _arrays(n, i)
        clean[split] = {"sha256": save_split(split_path(D.SOURCE, split), arrays), "n": n,
                        "class_counts": np.bincount(arrays["y"], minlength=10).tolist()}
        save_split(split_path(D.CONDITION_SOURCE, split), arrays)
    (dataset_dir(D.SOURCE) / "replay").mkdir()
    (dataset_dir(D.SOURCE) / "replay" / "rows.json").write_text("[1, 2]\n")
    write_json(dataset_json_path(D.SOURCE), _manifest(D.SOURCE, clean))
    write_json(dataset_json_path(D.CONDITION_SOURCE), _manifest(D.CONDITION_SOURCE, clean))
    cond = {}
    for i, split in enumerate(D.CONDITION_SPLITS):
        cond[split] = {"sha256": save_split(split_path(D.CONDITION_SOURCE, split, CONDITION),
                                            _arrays(SIZES[split], 10 + i)), "n": SIZES[split]}
    write_json(dataset_json_path(D.CONDITION_SOURCE, CONDITION), _manifest(
        D.CONDITION_SOURCE, cond, condition={"name": CONDITION, "corruption": "snow", "severity": 5},
        test={"location": "sealed", "n": 10}, forward_path="../../forward.py"))
    return tmp_path


def test_store_is_a_hash_checked_copy_with_its_own_forward(store: Path) -> None:
    manifest = D.build_clean(RUN)
    assert manifest["dataset"] == D.DATASET and manifest["forward_sha256"] == file_sha256(forward_path(D.DATASET))
    for split in SIZES:
        assert file_sha256(split_path(D.DATASET, split)) == file_sha256(split_path(D.SOURCE, split))
    text = forward_path(D.DATASET).read_text()
    for word in ("gaussian", "cifar", "vl7", "snow"):
        assert word not in text.lower()
    with pytest.raises(D.E11StoreError):
        D.build_clean(RUN)
    cond = D.build_condition(CONDITION, RUN)
    assert cond["dataset"] == D.DATASET and cond["forward_path"] == "../../forward.py"
    assert cond["forward_sha256"] == manifest["forward_sha256"] and cond["condition"]["name"] == CONDITION
    assert cond["test"] == {"location": "sealed", "n": 10}
    for split in D.CONDITION_SPLITS:
        assert file_sha256(split_path(D.DATASET, split, CONDITION)) == file_sha256(
            split_path(D.CONDITION_SOURCE, split, CONDITION))
    with pytest.raises(D.E11StoreError):
        D.build_condition(CONDITION, RUN)


def test_a_tampered_condition_split_is_refused(store: Path) -> None:
    D.build_clean(RUN)
    save_split(split_path(D.CONDITION_SOURCE, "V_check", CONDITION), _arrays(SIZES["V_check"], 99))
    with pytest.raises(D.E11StoreError, match="does not match"):
        D.build_condition(CONDITION, RUN)


def _pending(store: Path, seed: int, *, smoke: bool = False) -> Path:
    from weightcode.models.e10_export import training_rows

    run_dir = store / "run"
    root = run_dir / "private" / "cifar10" / E.PENDING_DIR
    x, _, used, manifest = training_rows()
    torch.manual_seed(seed)
    model = V.ViTLite(V.ViTLiteConfig(), MEAN, STD).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    result = V.CCTTrainResult(model=model, log=[{"epoch": 1, "train_loss": 1.0}], final={}, wall_time_s=1.0)
    entry = E.export_pending(manifest=manifest, seed=seed, config=V.CCTTrainConfig(), model_config=V.ViTLiteConfig(),
                             result=result, training_splits=used, x_train=x, determinism={}, run_info=RUN,
                             models_root=root, extra_private={"smoke": True} if smoke else None)
    index = root / "index.json"
    rows = json.loads(index.read_text()) if index.exists() else []
    write_json(index, rows + [entry])
    return run_dir


def test_import_checks_parity_and_device_and_writes_the_store(store: Path) -> None:
    D.build_clean(RUN)
    run_dir = _pending(store, 100)
    (row,) = E.import_e11_models(run_dir, "test import")
    assert row["seed"] == 100 and row["arch"] == V.ARCH and row["role"] == "dev"
    assert row["parity"]["float64_max_rel_diff"] <= E.PARITY_FLOAT64_REL_TOL
    assert row["device_check"]["max_abs_diff"] <= E.DEVICE_MAX_ABS_LOGIT_DIFF
    target = models_dir(D.DATASET) / row["model_id"]
    spec = json.loads((target / "model_spec.json").read_text())
    assert spec["schema"] == V.SPEC_SCHEMA and spec["n_parameters"] == 3_716_874
    assert sum(math.prod(s) for s in spec["tensors"].values()) == 3_716_874
    text = json.dumps(spec).lower()
    for word in ("gaussian", "cifar", "vl7", "seed"):
        assert word not in text
    assert set(load_params(target / "model.npz")) == set(V.param_shapes(10))
    private = json.loads((target / "private.json").read_text())
    assert private["dataset"] == D.DATASET and 0.0 <= private["readouts"]["clean_vselect_acc"] <= 1.0
    with pytest.raises(RuntimeError, match="already in"):
        E.import_e11_models(run_dir, "again")


def test_import_refuses_smoke_students(store: Path) -> None:
    D.build_clean(RUN)
    with pytest.raises(RuntimeError, match="smoke"):
        E.import_e11_models(_pending(store, 101, smoke=True), "smoke")
