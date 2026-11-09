"""Rule seqskill-v1: instance build and reuse, evidence format, check parts, oracle, readout and refusals.

Nothing here reads the real stores: WEIGHTCODE_PRIVATE and WEIGHTCODE_SEALED point to tmp dirs. The student is a tiny tfm
trained for a few steps; it answers nothing, which is all the rule needs from a base.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.data import seqops as S
from weightcode.data import seqops_build as B
from weightcode.data.store import forward_path, load_split, read_dataset_json, split_path
from weightcode.evalkit import seqskill as Q
from weightcode.evalkit.patch_check import load_forward
from weightcode.models import train_seq as TS
from weightcode.models.export import write_index
from weightcode.models.forward_gen import render_seq_forward, write_forward
from weightcode.models.seq_export import export_seq_model

CONDITION = "new0_substitution"
TINY = TS.SeqTrainConfig(d_model=16, n_layers=1, d_mlp=32, steps=3, warmup_steps=1, log_every=3)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("private", "sealed"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("WEIGHTCODE_PRIVATE", str(tmp_path / "private"))
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(tmp_path / "sealed"))
    sha = write_forward(forward_path(S.DATASET), render_seq_forward(**B.forward_constants()))
    base_manifest = B.write_built(B.build_base(), {"forward_path": "forward.py", "forward_sha256": sha})
    torch.manual_seed(0)
    result = TS.train_seq(TINY, 100, S.held_out_keys(S.build_base_splits()), torch.device("cpu"))
    entry = export_seq_model(manifest=base_manifest, role="base", seed=100, config=TINY, result=result,
                             size_check={"em": 0.0, "passed": False}, determinism={},
                             run_info={"versions": {}, "device": "cpu"}, vselect=load_split(S.DATASET, "V_select"))
    write_index(S.DATASET, [entry])
    B.write_built(B.build_condition(CONDITION), B.condition_extra(base_manifest))
    return tmp_path


def test_build_writes_a_current_instance_and_reuses_it(store: Path) -> None:
    (row,) = Q.build_seqskill_instances(CONDITION)
    assert row["status"] == "created" and Q.INSTANCE_ID_RE.fullmatch(row["instance_id"])
    root = Q.seqskill_instance_dir(row["instance_id"])
    assert sorted(p.name for p in root.iterdir()) == sorted([*Q.EDITOR_VISIBLE_FILES, "private.json"])
    task = Q.check_seqskill_current(root)
    assert task["new_operator"]["token"] == "OP_NEW0" and task["limits"]["max_rel_frob"] == 0.5
    text = (root / "task.json").read_text() + (root / "model_spec.json").read_text()
    assert not any(word in text.lower() for word in ("substitution", "permutation", "seqops", "seed"))
    (again,) = Q.build_seqskill_instances(CONDITION)
    assert again["status"] == "reused" and again["instance_id"] == row["instance_id"]
    private = json.loads((root / "private.json").read_text())
    assert private["evidence_rule"] == Q.SEQSKILL_RULE and private["n_vcheck_new"] == 4000
    assert private["n_vcheck_old"] == 4000 and private["evidence_symbol_coverage"] == S.N_SYMBOLS
    assert private["n_vselect_parts"] == {"i": 2000, "ii": 2000, "iii": 2000, "iv": 4000}


def test_evidence_format_and_predictions(store: Path) -> None:
    (row,) = Q.build_seqskill_instances(CONDITION)
    instance = Q.load_seqskill_instance(row["instance_id"])
    ev = instance.evidence
    assert sorted(ev) == sorted(Q.EVIDENCE_KEYS)
    for part in ("new", "old"):
        x, y, pred = ev[f"x_{part}"], ev[f"y_{part}"], ev[f"pred_{part}"]
        assert x.shape == y.shape == pred.shape == (64, S.SEQ_LEN) and pred.dtype == np.int64
        np.testing.assert_array_equal(pred == S.IGNORE, y == S.IGNORE)
        np.testing.assert_array_equal(pred, Q.seq_predictions(instance.forward, instance.params, x, y))
    new0 = S.TOKEN_ID["OP_NEW0"]
    assert np.all(ev["x_new"][:, 1] == new0) and np.all(ev["x_new"][:, 2] == S.NONE)
    assert not np.isin(ev["x_old"][:, 1:3], S.NEW_OP_IDS).any()
    ids = [*ev["split_ids_new"].tolist(), *ev["split_ids_old"].tolist()]
    assert len(set(ids)) == 128
    v_ev = load_split(S.DATASET, "V_evidence", CONDITION)
    assert np.isin(S.row_key(*S.decode_x(ev["x_new"])), v_ev["idx"]).all()


def test_oracle_and_readout(store: Path) -> None:
    (row,) = Q.build_seqskill_instances(CONDITION)
    instance = Q.load_seqskill_instance(row["instance_id"])
    oracle = Q.make_seqskill_oracle(instance, budget=1)
    same = oracle.query(instance.params, "noop")
    assert same["score"] == 0 and same["n_new"] == 4000 and same["n_old"] == 4000
    readout = Q.seqskill_readout(instance, instance.params)
    assert [readout[p]["n"] for p in ("i", "ii", "iii", "iv")] == [2000, 2000, 2000, 4000]
    assert readout["all_points"] == 0.0 and readout["new"]["n"] == 6000
    assert len(readout["iii_by_pair"]) == 16 and set(readout["ii_by_length"]) == {"9", "10", "11", "12"}
    kinds = readout["iii_by_map_kind"]
    assert set(kinds) == {"position", "symbol"} and kinds["position"]["n"] + kinds["symbol"]["n"] == 2000
    pairs = readout["iii_by_pair"]
    position_pairs = [name for name in pairs if any(f"OP_{op}" in name.split("+") for op in S.POSITION_OPS)]
    assert sum(pairs[name]["n"] for name in position_pairs) == kinds["position"]["n"]
    single, composed = readout["iv_by_kind"]["single"], readout["iv_by_kind"]["composed"]
    assert single["n"] > 0 and composed["n"] > 0 and single["n"] + composed["n"] == 4000


def test_check_parts_refuse_other_parts(store: Path) -> None:
    v_sel = load_split(S.DATASET, "V_select", CONDITION)
    with pytest.raises(Q.SeqskillError):
        Q.seqskill_check_parts(v_sel)
    parts = Q.seqskill_check_parts(load_split(S.DATASET, "V_check", CONDITION))
    assert len(parts["y_new"]) == len(parts["y_old"]) == 4000


def test_a_base_that_answers_new_rows_is_refused(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Q, "BASE_EM_MAX", 0.0)
    with pytest.raises(Q.SeqskillError, match="already answers"):
        Q.build_seqskill_instances(CONDITION)
    assert Q.read_seqskill_instance_index()["instances"] == []


def test_stale_task_is_refused(store: Path) -> None:
    (row,) = Q.build_seqskill_instances(CONDITION)
    root = Q.seqskill_instance_dir(row["instance_id"])
    task = json.loads((root / "task.json").read_text())
    task["new_operator"]["facts"] = "changed"
    (root / "task.json").write_text(json.dumps(task))
    with pytest.raises(Q.StaleSeqskillInstance):
        Q.check_seqskill_current(root)


def test_draws_and_leak_tokens() -> None:
    np.testing.assert_array_equal(Q.draw_replay_rows(4000), Q.draw_replay_rows(4000))
    rows, ids_new, ids_old = Q.draw_new_rows(2000, 7)
    assert len(set(rows.tolist())) == 64 and len(ids_new) == 64 and len(ids_old) == 64
    assert {"substitution", "new0_substitution", "seqops", "seed"} <= set(Q.seqskill_leak_tokens(CONDITION))
