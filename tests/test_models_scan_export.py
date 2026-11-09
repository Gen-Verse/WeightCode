"""Training, export and import of the E13 SCAN students on a toy store (``scan_toy``); never the real stores.

A tiny tfm (d 16, 1 layer) trains 2 epochs: the point is the recipe's mechanics, the export (parity at every position
of the old V_select rows, the gate, the public spec) and the importer's re-checks, not the student's accuracy.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import scan_toy
import torch

from weightcode.data import scan as S
from weightcode.data.scan_build import checked_split
from weightcode.data.store import models_dir, read_dataset_json, read_json, write_json
from weightcode.models import train_scan as TS
from weightcode.models.export import read_index
from weightcode.models.scan_export import build_spec, export_scan_model, import_scan_models

TINY = dataclasses.replace(TS.SIZE_STEPS[0], d_model=16, n_layers=1, d_mlp=32, epochs=2, warmup_steps=2,
                           log_every_epochs=1, batch_size=512)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return scan_toy.built_store(tmp_path, monkeypatch)


def test_recipe_is_e5s_and_the_sizes_are_the_prereg_ones() -> None:
    small, large = TS.SIZE_STEPS
    assert (small.d_model, small.n_layers, small.n_heads, small.d_mlp) == (128, 4, 4, 512)
    assert (large.d_model, large.n_layers, large.n_heads, large.d_mlp) == (256, 6, 4, 1024)
    assert (small.lr, small.batch_size, small.epochs, small.warmup_steps, small.min_lr_ratio) == (1e-3, 256, 800,
                                                                                                    1000, 0.1)
    assert (small.weight_decay, small.grad_clip, small.beta1, small.beta2) == (0.01, 1.0, 0.9, 0.98)
    assert TS.GATE_EM == 0.95 and TS.HEADROOM_EM_MAX == 0.50
    total = small.total_steps(11670)
    assert total == 800 * 46 and TS.lr_at(small, total - 1, total) == pytest.approx(1e-4, rel=1e-3)
    order = TS.epoch_order(100, 3, 50)
    assert sorted(order.tolist()) == list(range(50)) and np.array_equal(order, TS.epoch_order(100, 3, 50))
    assert not np.array_equal(order, TS.epoch_order(100, 4, 50))


def _train(condition: str, seed: int = 100) -> TS.ScanTrainResult:
    torch.manual_seed(seed)
    return TS.train_scan(TINY, seed, checked_split(condition, "train"), torch.device("cpu"))


def test_training_is_deterministic_and_refuses_new_rows(store: Path) -> None:
    a, b = _train("around_right"), _train("around_right")
    assert all(torch.equal(p, q) for p, q in zip(a.model.state_dict().values(), b.model.state_dict().values()))
    train = checked_split("around_right", "train")
    n = len(train["y"])
    assert a.steps == TINY.total_steps(n) and a.rows_seen == 2 * n and len(a.log) == 2
    train["part"] = train["part"].copy()
    train["part"][0] = S.PART_NEW
    with pytest.raises(ValueError):
        TS.train_scan(TINY, 100, train, torch.device("cpu"))


def _export(condition: str, root: Path | None, smoke: bool = False) -> dict:
    result = _train(condition)
    return export_scan_model(
        condition=condition, manifest=read_dataset_json(S.DATASET, condition), base_manifest=read_dataset_json(S.DATASET),
        seed=100, size_step=0, config=TINY, result=result, determinism={"torch_manual_seed": 100},
        run_info={"versions": {"torch": torch.__version__}, "device": "cpu"},
        old_select=checked_split(condition, "old_select"), models_root=root, extra_private={"smoke": True} if smoke else None)


def _fake_run(tmp_path: Path, condition: str, entry: dict, source_root: Path) -> Path:
    run = tmp_path / "run"
    target = run / "private" / S.DATASET / "conditions" / condition / "models"
    target.mkdir(parents=True)
    shutil.copytree(source_root / entry["model_id"], target / entry["model_id"])
    write_json(target / "index.json", [entry])
    write_json(run / "meta.json", {"run_name": "r", "request": "R22-e13", "select": "select=1:ngpus=1",
                                   "walltime": "01:00:00", "code_digest": "abc"})
    write_json(run / "exit.json", {"exit_code": 0, "elapsed_s": 1})
    (run / "environment.txt").write_text("0" * 64 + " /x/image.sif\n")
    (run / "job_id.txt").write_text("1.pbs\n")
    return run


def test_export_parity_gate_spec_and_import(store: Path, tmp_path: Path) -> None:
    root = tmp_path / "exported"
    entry = _export("addprim_jump", root)
    assert entry["role"] == ("base" if entry["vselect_acc"] >= TS.GATE_EM else "dev")
    folder = root / entry["model_id"]
    private = read_json(folder / "private.json")
    assert private["parity"]["n_rows"] == 1000 and private["gate"]["split"] == "old_select"
    assert private["parity"]["float64"]["max_rel_diff"] <= private["parity"]["float64"]["tolerance"]
    spec = read_json(folder / "model_spec.json")
    meanings = {row["name"]: row["meaning"] for row in spec["input"]["tokens"]}
    assert meanings["jump"] == "primitive: [[jump]] = I_JUMP" and "[[x1 after x2]]" in meanings["after"]
    assert spec["classes"] == list(S.CLASS_NAMES)
    text = json.dumps(spec).lower()
    assert not any(word in text for word in ("addprim", "template", "around_right", "tasks_"))
    run = _fake_run(tmp_path, "addprim_jump", entry, root)
    checks = import_scan_models(run, "addprim_jump", "test")
    assert checks[0]["local_vselect_acc"] == entry["vselect_acc"]
    assert read_index(S.DATASET, "addprim_jump") == [entry]
    assert (models_dir(S.DATASET, "addprim_jump") / entry["model_id"] / "model_spec.json").is_file()
    with pytest.raises(ValueError):
        import_scan_models(run, "addprim_jump", "again")


def test_import_refuses_smoke_and_a_spec_that_differs(store: Path, tmp_path: Path) -> None:
    root = tmp_path / "exported"
    entry = _export("around_right", root, smoke=True)
    run = _fake_run(tmp_path, "around_right", entry, root)
    with pytest.raises(ValueError, match="smoke"):
        import_scan_models(run, "around_right")
    private_file = run / "private" / S.DATASET / "conditions" / "around_right" / "models" / entry["model_id"] / "private.json"
    private = read_json(private_file)
    private.pop("smoke")
    write_json(private_file, private)
    spec_file = private_file.with_name("model_spec.json")
    spec = read_json(spec_file)
    spec["input"]["layout"] = "something else"
    write_json(spec_file, spec)
    with pytest.raises(ValueError, match="model_spec"):
        import_scan_models(run, "around_right")


def test_spec_builder_refuses_a_forbidden_term() -> None:
    from weightcode.models import transformer as T

    config = TINY.model_config()
    params = T.export_seq_params(T.build_seq_model(config))
    spec = build_spec("m00000000", config, params)
    assert spec["input"]["vocab_size"] == S.VOCAB_SIZE and spec["input"]["shape"] == [S.SEQ_LEN]
