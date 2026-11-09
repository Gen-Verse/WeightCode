"""CEC's replace_base_fc on the E9 students (``weightcode.models.fscil_export.replace_base_fc``), on the toy store."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fscil_toy import add_toy_allclass, toy_fscil  # noqa: F401

from weightcode.common.identity import load_params
from weightcode.data.store import read_dataset_json
from weightcode.evalkit import fscil as FS
from weightcode.models import fscil_export as X
from weightcode.models.export import fewshot_models_dir, read_fewshot_index


def test_replace_base_fc_sets_class_means_and_supersedes(toy_fscil) -> None:  # noqa: F811
    allclass_id = add_toy_allclass(toy_fscil)
    before = {row["role"]: row for row in read_fewshot_index(FS.DATASET)}
    records = X.replace_base_fc(100, note="toy test")
    assert [r["role"] for r in records] == ["fscil_base", "fscil_allclass"]
    assert {r["superseded"]["model_id"] for r in records} == {toy_fscil.base_model_id, allclass_id}

    index = {row["role"]: row for row in read_fewshot_index(FS.DATASET)}
    assert set(index) == {"fscil_base", "fscil_allclass"}
    assert index["fscil_base"]["model_id"] not in (toy_fscil.base_model_id, allclass_id)
    assert FS.base_student(100)["model_id"] == index["fscil_base"]["model_id"]

    manifest = read_dataset_json(FS.DATASET)
    features = X.checked_features(FS.DATASET, manifest)
    root = fewshot_models_dir(FS.DATASET)
    for role, row in index.items():
        old = load_params(root / before[role]["model_id"] / "model.npz")
        new = load_params(root / row["model_id"] / "model.npz")
        classes = X.role_train_classes(role)
        x, y, _ = X.training_rows(role, manifest)
        f = features(old, x)
        means = np.stack([f[y == c].mean(axis=0) for c in classes]).astype(np.float32)
        assert np.array_equal(new["fc.weight"][classes], means)
        rest = [c for c in range(FS.N_CLASSES) if c not in classes]
        assert np.array_equal(new["fc.weight"][rest], old["fc.weight"][rest])
        assert all(np.array_equal(new[k], old[k]) for k in old if k != "fc.weight")
        private = json.loads((root / row["model_id"] / "private.json").read_text())
        assert private["head_rule"]["name"] == X.HEAD_RULE and private["model_id"] == row["model_id"]
        assert private["head_rule"]["source"]["model_id"] == before[role]["model_id"]
        assert private["role"] == role and private["train_classes"] == classes
        FS.model_record(row, role)  # hash and role check of the new student
        assert (root / before[role]["model_id"] / "model.npz").exists()  # the source stays in the store

    log = (root / X.REPLACEMENTS_LOG).read_text().splitlines()
    assert len(log) == 2 and json.loads(log[0])["note"] == "toy test"
    with pytest.raises(ValueError, match="already carries head rule"):
        X.replace_base_fc(100, note="again")
    with pytest.raises(ValueError, match="note"):
        X.replace_base_fc(100, note=" ")
