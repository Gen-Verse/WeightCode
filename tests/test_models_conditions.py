"""Condition-aware training, export, evaluation and import on a tiny synthetic CIFAR-10 store (CPU, seconds).

A clean store and its gaussian_noise_s3 condition are built in tmp dirs. One clean base model and one condition
ceiling are trained with the real recipe (30 epochs of one batch each), exported with the parity check, read under
the condition (base_eval) and imported into a second store the way an NSCC run is imported.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from weightcode.data import conditions as C
from weightcode.data import store
from weightcode.data.registry import CIFAR10_CLASSES, SPLIT_SEED
from weightcode.models.export import read_index
from weightcode.models.forward_gen import render_forward, write_forward
from weightcode.models.importer import evaluate_on_condition, import_models

REPO = Path(__file__).resolve().parents[1]
SIZES = {"train": 64, "V_select": 32, "V_evidence": 16, "V_check": 16}
CONDITION = "gaussian_noise_s3"


def _train_base_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_base_under_test", REPO / "scripts" / "train_base.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_store(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    for name in ("private", "sealed"):
        (root / name).mkdir(parents=True)
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(root / "private"))
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(root / "sealed"))
    rng = np.random.default_rng(5)
    splits_meta = {}
    for offset, (split, n) in enumerate(SIZES.items()):
        y = np.arange(n, dtype=np.int64) % 10
        x = rng.integers(0, 256, size=(n, 3, 32, 32), dtype=np.uint8)
        x[:, y % 3] = np.minimum(x[:, y % 3], 60)  # a weak class signal
        arrays = {"x": x, "y": y, "idx": np.arange(100 * offset, 100 * offset + n, dtype=np.int64)}
        digest = store.save_split(store.split_path("cifar10", split), arrays)
        splits_meta[split] = {"path": f"splits/{split}.npz", "n": n, "sha256": digest}
    test = {
        "x": rng.integers(0, 256, size=(20, 3, 32, 32), dtype=np.uint8),
        "y": np.arange(20, dtype=np.int64) % 10,
        "idx": np.arange(20, dtype=np.int64),
    }
    mean, std = store.channel_stats(store.load_split("cifar10", "train")["x"])
    forward_sha = write_forward(store.forward_path("cifar10"), render_forward((3, 32, 32), mean, std, 10))
    manifest = {
        "schema": "weightcode.dataset.v0",
        "dataset": "cifar10",
        "source": "CIFAR10",
        "class_names": list(CIFAR10_CLASSES),
        "image_shape": [3, 32, 32],
        "norm": {"mean": mean, "std": std, "computed_on": "train"},
        "split_seed": SPLIT_SEED,
        "splits": splits_meta,
        "test": {"location": "$WEIGHTCODE_SEALED/cifar10/test.npz", "n": 20, "sha256": store.split_sha256(test)},
        "forward_sha256": forward_sha,
    }
    store.write_json(store.dataset_json_path("cifar10"), manifest)
    C.write_condition(C.build_from_store(CONDITION, test), {"note": "test"})
    return test


def _fake_run_dir(run_dir: Path, private_store: Path) -> Path:
    """A fetched run directory as fetch.sh leaves it, around an existing private store."""
    run_dir.mkdir(parents=True)
    shutil.copytree(private_store, run_dir / "private")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_name": "r1",
                "request": "R-test",
                "select": "select=1:ngpus=1",
                "walltime": "00:10:00",
                "code_digest": "abc",
            }
        )
    )
    (run_dir / "exit.json").write_text(json.dumps({"exit_code": 0, "elapsed_s": 5}))
    (run_dir / "job_id.txt").write_text("1.pbs\n")
    (run_dir / "environment.txt").write_text("$ cat x.sif.sha256\n" + "d" * 64 + "  image.sif\n")
    return run_dir


def _git_state(tmp_path: Path) -> Path:
    path = tmp_path / "git_state.json"
    path.write_text(json.dumps({"commit": "0" * 40, "dirty": True}))
    return path


def test_ceiling_export_base_eval_and_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tb = _train_base_module()
    _make_store(tmp_path / "remote", monkeypatch)
    git_state = _git_state(tmp_path)

    clean = tb.Trainer("cifar10", "cpu", git_state_file=git_state)
    base = clean.train_one("base", 0, "train")
    ceiling_trainer = tb.Trainer("cifar10", "cpu", CONDITION, git_state)
    ceiling = ceiling_trainer.train_one("ceiling", 0, "train", {"condition_note": "test"})

    assert [e["model_id"] for e in read_index("cifar10")] == [base["model_id"]]
    assert read_index("cifar10", CONDITION) == [ceiling]
    folder = store.models_dir("cifar10", CONDITION) / ceiling["model_id"]
    private = json.loads((folder / "private.json").read_text())
    assert private["condition"] == CONDITION and private["role"] == "ceiling"
    assert private["train_split_sha256"] == store.read_dataset_json("cifar10", CONDITION)["splits"]["train"]["sha256"]
    assert private["git"]["recorded_at_submission"] == "git_state.json"
    assert private["parity"]["max_abs_diff"] <= 1e-4

    result = evaluate_on_condition("cifar10", CONDITION, {"base": (0,)}, {"note": "test"})
    (row,) = result["models"]
    assert row["model_id"] == base["model_id"] and 0.0 <= row["vselect_acc"] <= 1.0
    assert row["clean_vselect_acc"] == base["vselect_acc"]
    again = evaluate_on_condition("cifar10", CONDITION, {"base": (0,)}, {"note": "rerun"})
    assert again["models"][0]["vselect_acc"] == row["vselect_acc"]
    with pytest.raises(ValueError, match="no clean model"):
        evaluate_on_condition("cifar10", CONDITION, {"dev": (100,)}, {})

    # Import into a second store holding the same data but no condition models, as after an NSCC run whose run
    # directory holds the remote store under private/.
    run_dir = _fake_run_dir(tmp_path / "run", tmp_path / "remote" / "private")
    local_root = tmp_path / "local"
    shutil.copytree(tmp_path / "remote", local_root)
    shutil.rmtree(local_root / "private" / "cifar10" / "conditions" / CONDITION / "models")
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(local_root / "private"))
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(local_root / "sealed"))
    checks = import_models("cifar10", CONDITION, run_dir, note="test import")
    assert [c["model_id"] for c in checks] == [ceiling["model_id"]]
    assert checks[0]["local_vselect_acc"] == ceiling["vselect_acc"]
    assert read_index("cifar10", CONDITION) == [ceiling]
    imported = store.models_dir("cifar10", CONDITION) / ceiling["model_id"]
    assert sorted(p.name for p in imported.iterdir()) == ["model.npz", "model_spec.json", "private.json"]
    block = json.loads((imported / "private.json").read_text())["import"]
    assert block["pbs_job_id"] == "1.pbs" and block["container"]["sha256"] == "d" * 64 and block["note"] == "test import"
    assert block["trained_with"]["device"] == private["device"]
    with pytest.raises(ValueError, match="already used"):
        import_models("cifar10", CONDITION, run_dir)


def test_import_refuses_tampered_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tb = _train_base_module()
    _make_store(tmp_path / "remote", monkeypatch)
    trainer = tb.Trainer("cifar10", "cpu", CONDITION, _git_state(tmp_path))
    ceiling = trainer.train_one("ceiling", 1, "train")
    run_dir = _fake_run_dir(tmp_path / "run", tmp_path / "remote" / "private")
    source = run_dir / "private" / "cifar10" / "conditions" / CONDITION / "models"
    shutil.rmtree(store.models_dir("cifar10", CONDITION))
    private_path = source / ceiling["model_id"] / "private.json"
    private = json.loads(private_path.read_text())
    private_path.write_text(json.dumps({**private, "smoke": True}))
    with pytest.raises(ValueError, match="smoke"):
        import_models("cifar10", CONDITION, run_dir)
    private_path.write_text(json.dumps({**private, "train_split_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="not the local one"):
        import_models("cifar10", CONDITION, run_dir)
    private_path.write_text(json.dumps(private))
    index = json.loads((source / "index.json").read_text())
    index[0]["vselect_acc"] = index[0]["vselect_acc"] + 0.25
    (source / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="V_select accuracy"):
        import_models("cifar10", CONDITION, run_dir)
    (run_dir / "exit.json").write_text(json.dumps({"exit_code": 1, "elapsed_s": 5}))
    with pytest.raises(ValueError, match="exited"):
        import_models("cifar10", CONDITION, run_dir)
    assert not store.models_dir("cifar10", CONDITION).exists() or read_index("cifar10", CONDITION) == []
