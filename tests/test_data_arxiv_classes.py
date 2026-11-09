"""Rule graphnew-classes-v1 (PREREG E7) on toy class counts: eligibility, the seeded draw and the sidecar record.

The store is a tmp directory (WEIGHTCODE_PRIVATE points there), never the real one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from weightcode.data import arxiv as A
from weightcode.data import arxiv_classes as K
from weightcode.data.registry import SPLIT_SEED


def _meta() -> dict:
    rng = np.random.default_rng(0)
    train = rng.integers(100, 5000, A.N_CLASSES)
    evidence = rng.integers(0, 60, A.N_CLASSES)
    train[[3, 9, 17]] = [90000, 80000, 70000]  # the three largest
    evidence[[3, 9, 17]] = 100                 # eligible by evidence, excluded by size
    splits = {name: {"class_counts": rng.integers(0, 50, A.N_CLASSES).tolist()} for name in K.COUNT_SPLITS}
    splits["train"]["class_counts"] = train.tolist()
    splits["V_evidence"]["class_counts"] = evidence.tolist()
    return {"dataset": K.DATASET, "protocol": "inductive", "splits": splits}


def test_eligibility_and_draw_follow_the_rule() -> None:
    meta = _meta()
    eligible, largest = K.eligible_classes(meta)
    evidence = np.asarray(meta["splits"]["V_evidence"]["class_counts"])
    assert largest == [3, 9, 17]
    assert eligible == [c for c in range(A.N_CLASSES) if evidence[c] >= K.MIN_EVIDENCE_ROWS and c not in largest]
    record = K.draw_new_classes(meta)
    expected = sorted(np.random.default_rng([SPLIT_SEED, K.CLASS_STREAM]).choice(
        np.asarray(eligible), size=K.N_NEW_CLASSES, replace=False).tolist())
    assert record["new_classes"] == expected and len(expected) == K.N_NEW_CLASSES
    assert set(record["new_classes"]) <= set(eligible)
    assert record["train_classes"] == [c for c in range(A.N_CLASSES) if c not in expected]
    assert record["new_part_rows"]["V_select"] == int(np.asarray(meta["splits"]["V_select"]["class_counts"])[expected].sum())
    assert K.draw_new_classes(meta) == record


def test_too_few_eligible_classes_are_refused() -> None:
    meta = _meta()
    meta["splits"]["V_evidence"]["class_counts"] = [0] * A.N_CLASSES
    with pytest.raises(K.ClassDrawError, match="eligible"):
        K.draw_new_classes(meta)
    with pytest.raises(K.ClassDrawError, match="inductive"):
        K.draw_new_classes({**_meta(), "protocol": "transductive"})


def test_sidecar_is_written_once_and_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(tmp_path))
    folder = tmp_path / K.DATASET
    folder.mkdir()
    (folder / "dataset.json").write_text(json.dumps(_meta()), encoding="utf-8")
    first = K.write_class_draw()
    assert first["status"] == "created" and K.class_draw_path().exists()
    second = K.write_class_draw()
    assert second["status"] == "exists" and second["sha256"] == first["sha256"]
    block = K.class_draw_block()
    assert block["new_classes"] == first["new_classes"] and block["sha256"] == first["sha256"]
    tampered = json.loads(K.class_draw_path().read_text())
    tampered["new_classes"] = tampered["new_classes"][::-1]
    K.class_draw_path().write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(K.ClassDrawError, match="fresh draw"):
        K.read_class_draw()
    with pytest.raises(K.ClassDrawError, match="fresh draw"):
        K.write_class_draw()
    K.class_draw_path().unlink()
    K.write_class_draw()
    meta = _meta()
    meta["note"] = "changed"
    (folder / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(K.ClassDrawError, match="another dataset.json"):
        K.read_class_draw()
