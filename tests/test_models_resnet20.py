"""The E9 ResNet-20 student (weightcode.models.resnet20) and its export and import (weightcode.models.fscil_export).

Every test runs in a temporary store (WEIGHTCODE_PRIVATE, WEIGHTCODE_SEALED and WEIGHTCODE_RUNS point there) with synthetic images.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.common.identity import load_params
from weightcode.data import cifar100_fscil as C
from weightcode.data.store import dataset_json_path, forward_path, read_json, save_split, split_path, split_sha256, write_json
from weightcode.models import fscil_export as E
from weightcode.models import resnet20 as R
from weightcode.models.export import FEWSHOT_INDEX_KEYS, fewshot_models_dir, load_forward_no_cache, read_fewshot_index
from weightcode.models.forward_gen import render_resnet20_forward, write_forward
from weightcode.models.parity import check_logit_parity
from weightcode.models.spec import FORBIDDEN_TERMS

MEAN, STD = list(C.MEAN), list(C.STD)


def _forward_source() -> str:
    return render_resnet20_forward(MEAN, STD, C.N_CLASSES, R.COSINE_SCALE, R.BN_EPS, R.NORM_EPS)


def _forward_module() -> types.ModuleType:
    module = types.ModuleType("forward_resnet20_test")
    exec(compile(_forward_source(), "forward.py", "exec"), module.__dict__)
    return module


def _nontrivial_model(seed: int = 1) -> R.ResNet20Cosine:
    """A fresh model whose BatchNorm layers are not the identity in eval mode."""
    torch.manual_seed(seed)
    model = R.ResNet20Cosine(C.N_CLASSES, MEAN, STD)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                n = module.num_features
                module.running_mean.copy_(torch.rand(n, generator=generator) - 0.5)
                module.running_var.copy_(0.5 + 1.5 * torch.rand(n, generator=generator))
                module.weight.copy_(0.5 + torch.rand(n, generator=generator))
                module.bias.copy_(0.4 * torch.rand(n, generator=generator) - 0.2)
    return model


def _images(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, 3, 32, 32), dtype=np.uint8)


def test_parameter_names_shapes_and_counts() -> None:
    """Contract section 2 (corrected count): 106 tensors, 279,792 entries, 271,824 trainable encoder entries, and the
    names and shapes are exactly the ones forward.py expects."""
    model = R.ResNet20Cosine(C.N_CLASSES, MEAN, STD)
    params = R.export_params(model)
    assert list(params) == list(R.param_shapes(C.N_CLASSES))
    assert len(params) == 106
    assert sum(value.size for value in params.values()) == 279_792
    trainable = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("fc."))
    assert trainable == 271_824
    assert sum(value.size for name, value in params.items() if "running" in name) == 1_568
    assert all(value.dtype == np.float32 for value in params.values())
    assert not any("num_batches_tracked" in name or "preprocess" in name for name in params)
    assert _forward_module().expected_shapes() == R.param_shapes(C.N_CLASSES)
    assert {name: tuple(shape) for name, shape in R.param_shapes(C.N_CLASSES).items()
            if "downsample.0" in name} == {"layer2.0.downsample.0.weight": (32, 16, 1, 1),
                                          "layer3.0.downsample.0.weight": (64, 32, 1, 1)}
    with pytest.raises(KeyError, match="parameter names differ"):
        R.check_params({key: value for key, value in params.items() if key != "fc.weight"}, C.N_CLASSES)


def test_spec_ends_in_the_cosine_head_and_is_public() -> None:
    params = R.export_params(R.ResNet20Cosine(C.N_CLASSES, MEAN, STD))
    names = [f"class {i}" for i in range(C.N_CLASSES)]
    spec = R.build_resnet20_spec("m0000abcd", names, MEAN, STD, params)
    assert spec["layers"][-1] == {
        "op": "cosine_linear", "weight": "fc.weight", "shape": [100, 64], "scale": 16.0, "eps": 1e-12,
        "computes": spec["layers"][-1]["computes"],
    }
    assert spec["schema"] == R.SPEC_SCHEMA and spec["arch"] == R.ARCH and spec["n_parameters"] == 279_792
    assert spec["tensors"] == {name: list(shape) for name, shape in R.param_shapes(C.N_CLASSES).items()}
    blocks = [layer for layer in spec["layers"] if layer["op"] == "basic_block"]
    assert [layer["stride"] for layer in blocks] == [1, 1, 1, 2, 1, 1, 2, 1, 1]
    assert [layer["shortcut"] is not None for layer in blocks] == [False] * 3 + [True, False, False] * 2
    text = json.dumps({key: value for key, value in spec.items() if key != "classes"}).lower()
    assert not [term for term in FORBIDDEN_TERMS if term in text]


def test_numpy_forward_matches_torch_in_float64_and_float32() -> None:
    """forward.py (D's template) is an exact implementation of the torch student: float64 within one float32 step,
    float32 within the logit-scaled bound (weightcode.models.parity)."""
    model = _nontrivial_model()
    params = R.export_params(model)
    x = _images(64, 0)
    mine = _forward_module().forward(params, x)
    assert mine.dtype == np.float32 and mine.shape == (64, C.N_CLASSES)
    exact = R.eval_logits(R.build_resnet20(C.N_CLASSES, MEAN, STD, params, torch.float64), x)
    float32 = R.eval_logits(R.build_resnet20(C.N_CLASSES, MEAN, STD, params, torch.float32), x)
    parity = check_logit_parity(mine, exact, float32)
    assert parity["float64_max_rel_diff"] <= parity["float64_tolerance"]
    # The trained model in float32 on its own device gives the same logits as the float32 rebuild.
    assert np.max(np.abs(R.eval_logits(model.eval(), x) - float32)) < 1e-5
    assert np.max(np.abs(exact)) <= R.COSINE_SCALE + 1e-9


def test_training_is_seeded_and_returns_the_last_epoch() -> None:
    x, y = _images(48, 3), np.arange(48, dtype=np.int64) % 6
    config = R.FscilTrainConfig(epochs=2, batch_size=16)
    runs = [R.train_resnet20(config, 7, x, y, C.N_CLASSES, MEAN, STD, torch.device("cpu")) for _ in range(2)]
    first, second = (R.export_params(run.model) for run in runs)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    log = runs[0].log
    assert [entry["epoch"] for entry in log] == [1, 2] and all(np.isfinite(entry["train_loss"]) for entry in log)
    assert log[0]["lr"] == pytest.approx(0.1)
    assert runs[0].final["n_train"] == 48 and 0.0 <= runs[0].final["train_acc_clean"] <= 1.0
    other = R.export_params(R.train_resnet20(config, 8, x, y, C.N_CLASSES, MEAN, STD, torch.device("cpu")).model)
    assert not np.array_equal(first["conv1.weight"], other["conv1.weight"])
    with pytest.raises(ValueError, match="labels"):
        R.train_resnet20(config, 7, x, y + 100, C.N_CLASSES, MEAN, STD, torch.device("cpu"))
    # The learning-rate schedule of the recipe: 0.1, then 0.01 from epoch 61, then 0.001 from epoch 71.
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 70], gamma=0.1)
    rates = []
    for _ in range(100):
        rates.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert rates[59] == pytest.approx(0.1) and rates[60] == pytest.approx(0.01) and rates[70] == pytest.approx(0.001)


# ---------------------------------------------------------------------------------------------------------------
# Toy store and a fake NSCC run


TOY_ROWS = {"train": 2, "old_select": 1, "ref_pool": 1, "V_evidence": 1, "V_select": 1}
"""Rows per class of each toy split; train and old_select hold the base classes, the others the new ones."""


def _toy_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    for variable in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED", "WEIGHTCODE_RUNS"):
        (tmp_path / variable).mkdir()
        monkeypatch.setenv(variable, str(tmp_path / variable))
    splits = {}
    offset = 0
    for seed, (name, per_class) in enumerate(TOY_ROWS.items()):
        classes = range(C.N_BASE_CLASSES) if name in ("train", "old_select") else range(C.N_BASE_CLASSES, C.N_CLASSES)
        y = np.repeat(np.asarray(list(classes), dtype=np.int64), per_class)
        idx = np.arange(offset, offset + y.size, dtype=np.int64)
        offset += y.size
        splits[name] = {"x": _images(y.size, 100 + seed), "y": y, "idx": idx}
        save_split(split_path(C.DATASET, name), splits[name])
    sha = write_forward(forward_path(C.DATASET), _forward_source())
    manifest = {
        "schema": C.DATASET_SCHEMA,
        "dataset": C.DATASET,
        "class_names": [f"class {i}" for i in range(C.N_CLASSES)],
        "normalization": {"mean": MEAN, "std": STD},
        "split_seed": 20261103,
        "splits": {name: {"path": f"splits/{name}.npz", "sha256": split_sha256(arrays), "n": int(arrays["y"].size),
                          "class_counts": np.bincount(arrays["y"], minlength=C.N_CLASSES).tolist()}
                   for name, arrays in splits.items()},
        "forward_path": "forward.py",
        "forward_sha256": sha,
    }
    write_json(dataset_json_path(C.DATASET), manifest)
    return {"manifest": manifest, "splits": splits}


def _fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "nscc_runs" / "20261104_000000_fscil_r6_abc"
    pending = run / "private" / C.DATASET / "fewshot" / E.PENDING_DIR
    pending.mkdir(parents=True)
    write_json(run / "meta.json", {"run_name": run.name, "request": "R6-fscil-resnet20", "select": "select=1:ngpus=1",
                                   "walltime": "01:00:00", "code_digest": "abc123"})
    write_json(run / "exit.json", {"exit_code": 0, "elapsed_s": 12})
    (run / "environment.txt").write_text("0" * 64 + " /containers/image.sif\n")
    (run / "job_id.txt").write_text("123.pbs\n")
    return run


def _train_and_export(store: dict, role: str, root: Path, extra: dict | None = None) -> dict:
    manifest = store["manifest"]
    x, y, used = E.training_rows(role, manifest)
    result = R.train_resnet20(R.FscilTrainConfig(epochs=1, batch_size=32), 100, x, y, C.N_CLASSES, MEAN, STD,
                              torch.device("cpu"))
    return E.export_fscil_model(manifest=manifest, role=role, seed=100, config=R.FscilTrainConfig(epochs=1),
                                result=result, training_splits=used, x_train=x, determinism={},
                                run_info={"versions": {"torch": torch.__version__}, "device": "cpu"},
                                models_root=root, extra_private=extra)


def test_training_rows_of_each_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _toy_store(tmp_path, monkeypatch)
    x, y, used = E.training_rows("fscil_base", store["manifest"])
    assert set(used) == {"train"} and sorted(set(y.tolist())) == list(range(60)) and len(x) == 120
    x, y, used = E.training_rows("fscil_allclass", store["manifest"])
    assert list(used) == ["train", "ref_pool", "V_evidence"] and sorted(set(y.tolist())) == list(range(100))
    assert len(x) == 120 + 40 + 40
    with pytest.raises(ValueError, match="role must be"):
        E.role_train_classes("base")


def test_export_then_import_with_parity_and_readouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _toy_store(tmp_path, monkeypatch)
    run = _fake_run(tmp_path)
    pending = run / "private" / C.DATASET / "fewshot" / E.PENDING_DIR
    rows = [_train_and_export(store, role, pending) for role in ("fscil_base", "fscil_allclass")]
    write_json(pending / "index.json", rows)
    for row in rows:
        assert tuple(row) == FEWSHOT_INDEX_KEYS and row["vselect_acc"] is None
        private = read_json(pending / row["model_id"] / "private.json")
        assert private["readouts"] is None and len(private["device_check"]["logits"]) == E.DEVICE_CHECK_ROWS
        assert private["train_classes"] == E.role_train_classes(row["role"])
    assert read_fewshot_index(C.DATASET) == []  # a training job never writes the fewshot store

    checks = E.import_fscil_models(run, note="toy")
    index = read_fewshot_index(C.DATASET)
    assert {entry["role"] for entry in index} == {"fscil_base", "fscil_allclass"}
    forward, _ = load_forward_no_cache(forward_path(C.DATASET))
    splits = store["splits"]
    for entry, check in zip(sorted(index, key=lambda e: e["role"] != "fscil_base"), checks):
        assert entry["model_id"] == check["model_id"]
        params = load_params(fewshot_models_dir(C.DATASET) / entry["model_id"] / "model.npz")
        pred = {name: np.argmax(E.batched_logits(forward, params, splits[name]["x"]), axis=1)
                for name in E.READOUT_SPLITS}
        assert entry["vselect_acc"] == float(np.mean(pred["V_select"] == splits["V_select"]["y"]))
        if entry["role"] == "fscil_base":
            assert entry["vselect_acc_train_classes"] == float(np.mean(pred["old_select"] == splits["old_select"]["y"]))
        assert check["parity"]["float64_max_rel_diff"] <= check["parity"]["float64_tolerance"]
        assert check["device_check"]["max_abs_diff"] <= check["device_check"]["bound"]
        private = read_json(fewshot_models_dir(C.DATASET) / entry["model_id"] / "private.json")
        assert private["import"]["request"] == "R6-fscil-resnet20" and private["import"]["note"] == "toy"
    with pytest.raises(ValueError, match="already used"):
        E.import_fscil_models(run)


def test_import_refuses_smoke_and_foreign_students(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _toy_store(tmp_path, monkeypatch)
    run = _fake_run(tmp_path)
    pending = run / "private" / C.DATASET / "fewshot" / E.PENDING_DIR
    smoke = _train_and_export(store, "fscil_base", pending, {"smoke": True})
    write_json(pending / "index.json", [smoke])
    with pytest.raises(ValueError, match="smoke"):
        E.import_fscil_models(run)
    write_json(pending / "index.json", [{**smoke, "train_classes": list(range(100))}])
    with pytest.raises(ValueError, match="disagrees"):
        E.import_fscil_models(run)
    write_json(pending / "index.json", [{**smoke, "vselect_acc": 0.5}])
    with pytest.raises(ValueError, match="pending row"):
        E.import_fscil_models(run)
    assert read_fewshot_index(C.DATASET) == []
