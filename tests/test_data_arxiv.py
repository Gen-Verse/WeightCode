"""ogbn-arxiv on a toy graph: class-id permutation, shared split membership, the inductive snapshots, and the GCN
export of both protocols through a toy store.

A small random RawGraph stands in for the official zip (split sizes shrunk with monkeypatch); the store is a tmp
directory (WEIGHTCODE_PRIVATE and WEIGHTCODE_SEALED point there), never the real one.
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.data import arxiv as A
from weightcode.data.store import forward_path, models_dir, read_dataset_json, read_json
from weightcode.models import gcn as G
from weightcode.models import gcn_export as E
from weightcode.models.export import load_forward_no_cache
from weightcode.models.forward_gen import render_gcn_forward, write_forward

N_TRAIN, N_VALID, N_TEST, N_FEAT = 60, 30, 20, 6


@pytest.fixture()
def raw(monkeypatch: pytest.MonkeyPatch) -> A.RawGraph:
    monkeypatch.setattr(A, "OLD_SIZES", {"old_select": 6, "old_check": 6, "old_replay": 4})
    monkeypatch.setattr(A, "VALID_SIZES", {"V_select": 8, "V_evidence": 4, "V_check": 8})
    rng = np.random.default_rng(3)
    n = N_TRAIN + N_VALID + N_TEST
    order = rng.permutation(n)
    split_idx = {"train": np.sort(order[:N_TRAIN]), "valid": np.sort(order[N_TRAIN:N_TRAIN + N_VALID]),
                 "test": np.sort(order[N_TRAIN + N_VALID:])}
    years = np.empty(n, dtype=np.int64)
    years[split_idx["train"]] = rng.integers(2010, 2018, N_TRAIN)
    years[split_idx["valid"]] = 2018
    years[split_idx["test"]] = rng.integers(2019, 2021, N_TEST)
    edges = rng.integers(0, n, size=(400, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]
    return A.RawGraph(features=rng.standard_normal((n, N_FEAT)).astype(np.float32),
                      labels=rng.integers(0, 4, n) * 7, years=years, edges=edges, split_idx=split_idx,
                      class_names=tuple(f"cs.C{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(A.N_CLASSES)))


def _edge_set(graph: dict[str, np.ndarray], ids: np.ndarray) -> set[tuple[int, int]]:
    """Off-diagonal entries of a CSR graph as pairs of ``ids[row]``, ``ids[col]``."""
    rows = np.repeat(np.arange(graph["indptr"].size - 1), np.diff(graph["indptr"]))
    keep = rows != graph["indices"]
    return set(zip(ids[rows[keep]].tolist(), ids[graph["indices"][keep]].tolist()))


def _ogb_edges(raw: A.RawGraph, nodes: np.ndarray) -> set[tuple[int, int]]:
    inside = np.isin(raw.edges, nodes).all(axis=1)
    pairs = raw.edges[inside]
    return set(zip(pairs[:, 0].tolist(), pairs[:, 1].tolist())) | set(zip(pairs[:, 1].tolist(), pairs[:, 0].tolist()))


def _check_normalized(graph: dict[str, np.ndarray]) -> None:
    degree = np.diff(graph["indptr"]).astype(np.float64)
    rows = np.repeat(np.arange(degree.size), degree.astype(np.int64))
    expected = (1.0 / np.sqrt(degree[rows] * degree[graph["indices"]])).astype(np.float32)
    assert np.array_equal(graph["weight"], expected)
    diagonal = graph["indices"] == rows
    assert np.array_equal(np.bincount(rows[diagonal], minlength=degree.size), np.ones(degree.size, dtype=np.int64))


def test_labels_are_permuted_and_class_names_are_indices(raw: A.RawGraph) -> None:
    built = A.build(raw)
    permutation = A.class_permutation()
    assert sorted(permutation.tolist()) == list(range(A.N_CLASSES))
    assert not np.array_equal(permutation, np.arange(A.N_CLASSES))
    for arrays in [*built.splits.values(), built.test]:
        assert np.array_equal(arrays["y"], permutation[raw.labels[arrays["idx"]]])
    assert built.manifest["class_names"] == [str(i) for i in range(A.N_CLASSES)]
    assert np.array_equal(built.permutation["new_class_of_old"], permutation)
    assert "cs." not in json.dumps(built.manifest)


def test_both_protocols_share_split_membership(raw: A.RawGraph) -> None:
    transductive, inductive = A.build(raw), A.build_inductive(raw)
    assert transductive.manifest["protocol"] == "transductive" and inductive.manifest["protocol"] == "inductive"
    assert list(transductive.splits) == list(inductive.splits) == list(A.STORED_SPLITS)
    for name in A.STORED_SPLITS:
        one, other = transductive.splits[name], inductive.splits[name]
        assert np.array_equal(np.sort(one["idx"]), np.sort(other["idx"]))
        assert np.array_equal(transductive.permutation["new_of_old"][one["idx"]], one["x"])
        assert np.array_equal(inductive.permutation["ogb_of_new"][other["x"]], other["idx"])
        assert np.all(np.diff(other["x"]) > 0)
    union = np.concatenate([arrays["idx"] for arrays in inductive.splits.values()])
    assert np.unique(union).size == union.size == N_TRAIN + N_VALID


def test_inductive_snapshots_are_the_induced_subgraphs(raw: A.RawGraph) -> None:
    built = A.build_inductive(raw)
    ogb_of_new = built.permutation["ogb_of_new"]
    snapshot = np.concatenate([raw.split_idx["train"], raw.split_idx["valid"]])
    assert np.array_equal(np.sort(ogb_of_new), np.sort(snapshot))
    assert not np.isin(ogb_of_new, raw.split_idx["test"]).any()
    assert built.test is None and built.manifest["graph"]["n_nodes"] == N_TRAIN + N_VALID

    graph = built.graph
    assert _edge_set(graph, ogb_of_new) == _ogb_edges(raw, snapshot)
    assert np.array_equal(graph["features"], raw.features[ogb_of_new])
    _check_normalized(graph)

    train_graph = built.train_graph
    assert train_graph is not None
    nodes = train_graph["nodes"]
    assert np.all(np.diff(nodes) > 0)
    assert np.array_equal(np.sort(ogb_of_new[nodes]), raw.split_idx["train"])
    assert _edge_set(train_graph, ogb_of_new[nodes]) == _ogb_edges(raw, raw.split_idx["train"])
    assert np.array_equal(train_graph["features"], graph["features"][nodes])
    _check_normalized(train_graph)
    assert built.manifest["train_graph"]["sha256"] == A.graph_sha256(train_graph)
    old_nodes = np.concatenate([built.splits[name]["x"] for name in ("train", "old_select", "old_check", "old_replay")])
    assert np.array_equal(np.sort(old_nodes), nodes)


@pytest.mark.parametrize("protocol", ["transductive", "inductive"])
def test_export_trains_on_the_role_graph_and_checks_on_the_deployment_graph(
        raw: A.RawGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str) -> None:
    for variable in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED"):
        (tmp_path / variable).mkdir()
        monkeypatch.setenv(variable, str(tmp_path / variable))
    built = A.build(raw) if protocol == "transductive" else A.build_inductive(raw)
    dataset = A.PROTOCOLS[protocol]
    n_deploy = built.manifest["graph"]["n_nodes"]
    sha = write_forward(forward_path(dataset), render_gcn_forward(n_deploy, N_FEAT, A.N_CLASSES, 1e-5))
    A.write(built, sha, {"toy": True}, {"toy": True})
    manifest = read_dataset_json(dataset)
    forward, forward_sha = load_forward_no_cache(forward_path(dataset))
    deploy = E.checked_graph(dataset, manifest)
    config = G.GCNTrainConfig(hidden=8, epochs=3, log_every=1)
    for role in ("base", "ceiling"):
        graph, nodes, block = E.training_graph(dataset, role, manifest)
        x, y, used = E.training_nodes(dataset, role, manifest, nodes)
        inductive_base = protocol == "inductive" and role == "base"
        assert block["path"] == (A.TRAIN_GRAPH_FILE if inductive_base else "graph.npz")
        assert (nodes is not None) == inductive_base
        labelled = np.concatenate([built.splits[name]["x"] for name in E.GCN_ROLES[role]])
        assert np.array_equal(np.sort(labelled), np.sort(x if nodes is None else nodes[x]))
        assert graph["features"].shape[0] == (N_TRAIN if inductive_base else n_deploy)
        result = G.train_gcn(config, 0, graph, x, y, A.N_CLASSES, torch.device("cpu"))
        if inductive_base:
            wrong = {**block, "path": "graph.npz"}
            with pytest.raises(ValueError, match="must train on"):
                E.export_gcn_model(dataset=dataset, manifest=manifest, role=role, seed=0, config=config, result=result,
                                   training_splits=used, train_graph=wrong, determinism={}, run_info={"toy": True})
        row = E.export_gcn_model(dataset=dataset, manifest=manifest, role=role, seed=0, config=config, result=result,
                                 training_splits=used, train_graph=block, determinism={}, run_info={"toy": True})
        private = read_json(models_dir(dataset) / row["model_id"] / "private.json")
        assert private["model_config"]["n_nodes"] == n_deploy and private["protocol"] == protocol
        assert private["train_data"]["graph"] == block
        assert private["parity"]["n_nodes"] == n_deploy
        spec = read_json(models_dir(dataset) / row["model_id"] / "model_spec.json")
        assert spec["classes"] == list(A.CLASS_IDS) and spec["input"]["n_nodes"] == n_deploy
        check = E._check_one(dataset, row, models_dir(dataset), manifest, forward, forward_sha, deploy)
        assert check["local_acc"]["V_select"] == row["vselect_acc"]


def test_newclass_student_trains_on_the_train_classes_only(
        raw: A.RawGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PREREG E7: a newclass student trains on graph_train.npz with the labels of its train classes only, goes to the
    fewshot store with a FEWSHOT_INDEX_KEYS row, and records the class draw."""
    from weightcode.data import arxiv_classes as K
    from weightcode.models.export import FEWSHOT_INDEX_KEYS, fewshot_models_dir

    for variable in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED"):
        (tmp_path / variable).mkdir()
        monkeypatch.setenv(variable, str(tmp_path / variable))
    monkeypatch.setattr(K, "MIN_EVIDENCE_ROWS", 1)
    monkeypatch.setattr(K, "N_LARGEST_EXCLUDED", 0)
    monkeypatch.setattr(K, "N_NEW_CLASSES", 1)
    built = A.build_inductive(raw)
    dataset = built.dataset
    sha = write_forward(forward_path(dataset), render_gcn_forward(built.manifest["graph"]["n_nodes"], N_FEAT,
                                                                  A.N_CLASSES, 1e-5))
    A.write(built, sha, {"toy": True}, {"toy": True})
    manifest = read_dataset_json(dataset)
    draw = K.write_class_draw(dataset)
    (new_class,) = draw["new_classes"]
    with pytest.raises(ValueError, match="needs train_classes"):
        E.training_nodes(dataset, "newclass", manifest)
    with pytest.raises(ValueError, match="must be None"):
        E.training_nodes(dataset, "base", manifest, train_classes=draw["train_classes"])
    graph, nodes, block = E.training_graph(dataset, "newclass", manifest)
    assert block["path"] == A.TRAIN_GRAPH_FILE
    x, y, used = E.training_nodes(dataset, "newclass", manifest, nodes, draw["train_classes"])
    train = built.splits["train"]
    keep = np.isin(train["y"], draw["train_classes"])
    assert new_class not in set(y.tolist()) and np.array_equal(np.sort(nodes[x]), np.sort(train["x"][keep]))
    result = G.train_gcn(G.GCNTrainConfig(hidden=8, epochs=3, log_every=1), 0, graph, x, y, A.N_CLASSES,
                         torch.device("cpu"))
    row = E.export_gcn_model(dataset=dataset, manifest=manifest, role="newclass", seed=0,
                             config=G.GCNTrainConfig(hidden=8, epochs=3, log_every=1), result=result,
                             training_splits=used, train_graph=block, determinism={}, run_info={"toy": True},
                             extra_private={"class_draw": K.class_draw_block(dataset)},
                             train_classes=draw["train_classes"])
    assert tuple(row) == FEWSHOT_INDEX_KEYS and row["train_classes"] == draw["train_classes"]
    folder = fewshot_models_dir(dataset) / row["model_id"]
    private = read_json(folder / "private.json")
    assert private["train_classes"] == draw["train_classes"] and private["class_draw"]["sha256"] == draw["sha256"]
    assert private["final_vselect_acc_train_classes"] == row["vselect_acc_train_classes"]
    assert private["train_data"]["n_labelled"] == x.size
    forward, forward_sha = load_forward_no_cache(forward_path(dataset))
    deploy = E.checked_graph(dataset, manifest)
    check = E._check_one(dataset, row, fewshot_models_dir(dataset), manifest, forward, forward_sha, deploy)
    assert check["local_acc"]["V_select_train_classes"] == row["vselect_acc_train_classes"]
    with pytest.raises(ValueError, match="disagrees"):
        E._check_one(dataset, {**row, "train_classes": list(range(A.N_CLASSES - 1))}, fewshot_models_dir(dataset),
                     manifest, forward, forward_sha, deploy)
