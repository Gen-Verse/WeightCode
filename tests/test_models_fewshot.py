"""PREREG E2 students: training on a class subset, export to the fewshot store and its own index (CPU, seconds)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_models_conditions import _git_state, _make_store, _train_base_module

from weightcode.data import store
from weightcode.models.export import (
    FEWSHOT_INDEX_KEYS,
    all_model_ids,
    fewshot_models_dir,
    read_fewshot_index,
    read_index,
    write_fewshot_index,
)


def test_fewshot_students_train_on_their_classes_and_stay_out_of_the_dataset_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tb = _train_base_module()
    _make_store(tmp_path / "store", monkeypatch)
    trainer = tb.Trainer("cifar10", "cpu", git_state_file=_git_state(tmp_path))
    rows = [trainer.train_fewshot(arch, 100, (0, 1, 2, 3, 4)) for arch in ("cnn", "cnn4")]
    assert read_index("cifar10") == []
    assert read_fewshot_index("cifar10") == sorted(rows, key=lambda r: r["arch"])
    train_y = store.load_split("cifar10", "train")["y"]
    for row in rows:
        assert tuple(row) == FEWSHOT_INDEX_KEYS and row["role"] == "dev" and row["train_classes"] == [0, 1, 2, 3, 4]
        folder = fewshot_models_dir("cifar10") / row["model_id"]
        private = json.loads((folder / "private.json").read_text())
        assert private["family"] == "fewshot" and private["new_classes"] == [5, 6, 7, 8, 9]
        assert private["n_train_rows"] == int(np.count_nonzero(train_y < 5))
        assert private["arch"] == row["arch"] == private["training_config"]["arch"]
        assert private["parity"]["max_abs_diff"] <= 1e-4
        confusion = np.asarray(private["vselect_confusion"])
        assert row["vselect_acc_train_classes"] == pytest.approx(np.trace(confusion[:5, :5]) / confusion[:5].sum())
        spec = json.loads((folder / "model_spec.json").read_text())
        assert len(spec["classes"]) == 10 and spec["layers"][-1]["op"] == "linear"
        assert row["model_id"] in all_model_ids("cifar10")
    with pytest.raises(RuntimeError, match="already indexed"):
        trainer.train_fewshot("cnn", 100, (0, 1, 2, 3, 4))
    for bad in [(), (4, 3), (0, 10), tuple(range(10))]:
        with pytest.raises(ValueError):
            trainer.train_fewshot("cnn", 101, bad)


def test_fewshot_index_refuses_other_keys_and_duplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_store(tmp_path / "store", monkeypatch)
    row = dict(zip(FEWSHOT_INDEX_KEYS, ["m0", 100, "cnn", "dev", "0" * 64, 0.4, [0, 1, 2, 3, 4], 0.8]))
    with pytest.raises(ValueError, match="twice"):
        write_fewshot_index("cifar10", [row, dict(row, model_id="m1")])
    with pytest.raises(ValueError, match="keys"):
        write_fewshot_index("cifar10", [{k: v for k, v in row.items() if k != "vselect_acc_train_classes"}])
    write_fewshot_index("cifar10", [dict(row, seed=101), row])
    assert [r["seed"] for r in read_fewshot_index("cifar10")] == [100, 101]


def test_parse_classes_and_the_fewshot_cli_guards() -> None:
    tb = _train_base_module()
    assert tb.parse_classes("0-4") == (0, 1, 2, 3, 4) == tb.parse_classes("4,3,2,1,0")
    assert tb.parse_classes("0-1,5") == (0, 1, 5)
    with pytest.raises(ValueError, match="repeated"):
        tb.parse_classes("0-2,2")
