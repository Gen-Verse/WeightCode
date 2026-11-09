"""Rule graphnew-v1 (PREREG E7) on a toy inductive store: class draw, a newclass student, the instance build, the
stale check, the oracle parts, the readout, hop distances and the leak scan.

A small random RawGraph stands in for the official zip (split sizes shrunk with monkeypatch); the store is a tmp
directory (WEIGHTCODE_PRIVATE and WEIGHTCODE_SEALED point there), never the real one.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.baselines.instance_io import load_instance
from weightcode.data import arxiv as A
from weightcode.data import arxiv_classes as K
from weightcode.data.store import forward_path, read_dataset_json
from weightcode.evalkit import fewshot as F
from weightcode.evalkit import graphnew as GN
from weightcode.models import gcn as G
from weightcode.models import gcn_export as E
from weightcode.models.export import fewshot_models_dir, write_fewshot_index
from weightcode.models.forward_gen import render_gcn_forward, write_forward
from weightcode.patch import Limits

N_TRAIN, N_VALID, N_TEST, N_FEAT = 240, 150, 20, 6


def build_toy_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hidden: int = 8) -> dict:
    """The toy inductive store with one newclass student of width ``hidden`` (the ``store`` fixture uses 8)."""
    for variable in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED"):
        (tmp_path / variable).mkdir()
        monkeypatch.setenv(variable, str(tmp_path / variable))
    monkeypatch.setattr(A, "OLD_SIZES", {"old_select": 30, "old_check": 30, "old_replay": 30})
    monkeypatch.setattr(A, "VALID_SIZES", {"V_select": 40, "V_evidence": 40, "V_check": 40})
    monkeypatch.setattr(K, "MIN_EVIDENCE_ROWS", 3)
    monkeypatch.setattr(K, "N_LARGEST_EXCLUDED", 0)
    monkeypatch.setattr(K, "N_NEW_CLASSES", 1)
    monkeypatch.setattr(F, "N_PER_NEW_CLASS", 3)
    monkeypatch.setattr(GN, "N_PER_NEW_CLASS", 3)
    monkeypatch.setattr(GN, "N_REPLAY", 5)
    rng = np.random.default_rng(5)
    n = N_TRAIN + N_VALID + N_TEST
    order = rng.permutation(n)
    split_idx = {"train": np.sort(order[:N_TRAIN]), "valid": np.sort(order[N_TRAIN:N_TRAIN + N_VALID]),
                 "test": np.sort(order[N_TRAIN + N_VALID:])}
    years = np.empty(n, dtype=np.int64)
    years[split_idx["train"]] = rng.integers(2010, 2018, N_TRAIN)
    years[split_idx["valid"]] = 2018
    years[split_idx["test"]] = rng.integers(2019, 2021, N_TEST)
    labels = rng.integers(0, 4, n) * 7
    features = rng.standard_normal((n, N_FEAT)).astype(np.float32) + labels[:, None].astype(np.float32) / 7
    edges = rng.integers(0, n, size=(1200, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]
    raw = A.RawGraph(features=features, labels=labels, years=years, edges=edges, split_idx=split_idx,
                     class_names=tuple(f"cs.C{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(A.N_CLASSES)))
    built = A.build_inductive(raw)
    dataset = built.dataset
    sha = write_forward(forward_path(dataset), render_gcn_forward(built.manifest["graph"]["n_nodes"], N_FEAT,
                                                                  A.N_CLASSES, 1e-5))
    A.write(built, sha, {"toy": True}, {"toy": True})
    manifest = read_dataset_json(dataset)
    draw = K.write_class_draw(dataset)
    config = G.GCNTrainConfig(hidden=hidden, epochs=20, log_every=10)
    graph, nodes, block = E.training_graph(dataset, "newclass", manifest)
    x, y, used = E.training_nodes(dataset, "newclass", manifest, nodes, draw["train_classes"])
    result = G.train_gcn(config, 0, graph, x, y, A.N_CLASSES, torch.device("cpu"))
    row = E.export_gcn_model(dataset=dataset, manifest=manifest, role="newclass", seed=100, config=config,
                             result=result, training_splits=used, train_graph=block, determinism={},
                             run_info={"toy": True}, extra_private={"class_draw": K.class_draw_block(dataset)},
                             train_classes=draw["train_classes"])
    write_fewshot_index(dataset, [row])
    return {"dataset": dataset, "draw": draw, "row": row, "built": built}


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    return build_toy_store(tmp_path, monkeypatch)


def test_build_writes_a_loadable_current_instance(store: dict) -> None:
    (outcome,) = GN.build_graphnew_instances()
    assert outcome["status"] == "created" and GN.INSTANCE_ID_RE.fullmatch(outcome["instance_id"])
    folder = GN.graphnew_instance_dir(outcome["instance_id"])
    assert sorted(p.name for p in folder.iterdir()) == sorted((*GN.EDITOR_VISIBLE_FILES, "private.json"))
    task = GN.check_graphnew_current(folder)
    assert task["new_classes"] == store["draw"]["new_classes"] and task["graph"]["file"] == GN.GRAPH_FILE
    instance = load_instance(folder)
    ev = instance.evidence
    assert sorted(ev) == sorted(GN.EVIDENCE_KEYS) and ev["x_new"].dtype == np.int64
    built = store["built"]
    v_ev, replay = built.splits["V_evidence"], built.splits["old_replay"]
    assert set(ev["x_new"].tolist()) <= set(v_ev["x"].tolist()) and set(ev["y_new"].tolist()) == set(task["new_classes"])
    assert set(ev["x_old"].tolist()) <= set(replay["x"].tolist())
    assert set(ev["y_old"].tolist()) <= set(task["train_classes"]) and ev["x_old"].size == GN.N_REPLAY
    logits = instance.forward(dict(instance.params), np.concatenate([ev["x_new"], ev["x_old"]]))
    assert np.array_equal(np.argmax(logits, 1), np.concatenate([ev["pred_new"], ev["pred_old"]]))
    assert GN.build_graphnew_instances()[0]["status"] == "reused"
    (folder / "task.json").write_text(json.dumps(task), encoding="utf-8")
    with pytest.raises(GN.StaleGraphnewInstance, match="task_sha256"):
        GN.check_graphnew_current(folder)


def test_oracle_parts_and_readout_of_the_base(store: dict) -> None:
    (outcome,) = GN.build_graphnew_instances()
    instance = GN.load_graphnew_instance(outcome["instance_id"])
    oracle = GN.make_graphnew_oracle(instance)
    assert oracle is not None
    readout = GN.graphnew_readout(instance, instance.params)
    expected_keys = set(F.class_split_readout(np.zeros(2, int), np.zeros(2, int), np.array([0, 7]), [0], [7]))
    assert expected_keys <= set(readout)
    assert readout["delta_all"] == 0.0 and readout["fixed_all"] == readout["broken_all"] == 0
    assert readout["n_new"] == instance.private["n_vselect_new"]
    for key in ("hop_new", "hop_new_same_class"):
        assert set(readout[key]) == set(GN.HOP_BINS)
        assert sum(cell["n"] for cell in readout[key].values()) == readout["n_new"]
    assert readout["old_select"]["new"]["n"] == instance.private["n_oldselect_new"]
    assert readout["acc_new_balanced_child"] == readout["acc_new_balanced_base"]


def test_hop_distances_on_a_path() -> None:
    # path 0-1-2-3-4-5 with self loops, CSR
    n = 6
    rows = [[i] + [j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)]
    indptr = np.cumsum([0] + [len(r) for r in rows])
    indices = np.concatenate([sorted(r) for r in rows])
    assert GN.hop_distances(indptr, indices, np.array([0])).tolist() == [0, 1, 2, 3, 4, 4]
    assert GN.hop_distances(indptr, indices, np.array([2, 5])).tolist() == [2, 1, 0, 1, 1, 0]
    assert GN.hop_bin(np.array([1, 2, 3, 4, 9])).tolist() == ["1", "2", "3", "gt3", "gt3"]
    with pytest.raises(GN.GraphnewError):
        GN.hop_bin(np.array([0]))


def test_leak_scan_whole_words_and_codes() -> None:
    assert GN.graphnew_scan("e0a2018bc4d5e") == []
    assert GN.graphnew_scan("papers of 2018") == ["2018"]
    assert GN.graphnew_scan('"invalid_patches": "a patch that fails validation"') == []
    assert GN.graphnew_scan("the valid split") == ["valid"]
    assert "cs.lg" in GN.graphnew_scan("class cs.LG")
    assert "arxiv" in GN.graphnew_scan("ogbn-arxiv")
    task = GN.build_graphnew_task([str(i) for i in range(40)], list(range(32)), list(range(32, 40)), Limits(), 100, 8)
    assert GN.graphnew_scan(json.dumps(task)) == []


def test_head_and_program_limits() -> None:
    spec = {"layers": [{"op": "graph_conv", "weight": "conv1.weight", "bias": "conv1.bias"},
                       {"op": "graph_conv", "weight": "conv3.weight", "bias": "conv3.bias"}]}
    assert GN.head_tensors(spec) == ["conv3.weight", "conv3.bias"]
    with pytest.raises(GN.GraphnewError):
        GN.head_tensors({"layers": [{"op": "linear", "weight": "w", "bias": "b"}]})
    stated = GN.build_graphnew_task([str(i) for i in range(40)], list(range(32)), list(range(32, 40)),
                                    GN.graphnew_limits(spec), 100, 8)["limits"]
    limits = GN.graphnew_program_limits(stated, 14)
    assert limits.max_changed_frac == GN.GRAPHNEW_MAX_CHANGED_FRAC == 0.20
    assert set(limits.rel_frob_exempt) == {"conv3.weight", "conv3.bias"} and limits.max_ops >= 14


def test_category_codes_match_the_raw_zip() -> None:
    import os

    if not os.environ.get("WEIGHTCODE_PRIVATE"):
        pytest.skip("WEIGHTCODE_PRIVATE is not set, so the raw ogbn-arxiv zip cannot be located")
    path = A.raw_zip_path()
    if not path.exists():
        pytest.skip("the raw ogbn-arxiv zip is not installed here")
    import gzip
    import io

    with zipfile.ZipFile(path) as archive, archive.open(A.MEMBERS["classes"]) as handle, \
            gzip.GzipFile(fileobj=handle) as unzipped:
        lines = io.TextIOWrapper(unzipped, encoding="ascii").read().strip().splitlines()[1:]
    codes = sorted(A.class_name(line.split(",", 1)[1]).lower() for line in lines)
    assert codes == sorted(GN.ARXIV_CATEGORY_CODES) and len(codes) == A.N_CLASSES
