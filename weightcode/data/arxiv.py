"""ogbn-arxiv: the official OGB files, the official time split, private node-id and class-id permutations, and two
protocols (two store datasets) that share every split's membership.

Source: ``ARXIV_URL`` (OGB nodeproppred, release v1), installed once as ``$WEIGHTCODE_PRIVATE/raw/ogbn_arxiv/arxiv.zip``
with its sha256 in ``raw/provenance.json``. The CSV files are parsed here; the ``ogb`` package is not used.

Splits (official time split; publication years from ``node_year``):

* ``train``: the official train nodes (published up to 2017) minus the old holdouts; the base trains on these.
* ``old_select`` / ``old_check`` / ``old_replay``: 4,000 / 4,000 / 2,000 of the official train nodes, dealt per class
  with largest-remainder quotas and ``default_rng([SPLIT_SEED, OLD_HOLDOUT_STREAM])``, held out of base training:
  the old-distribution dev readout, the old half of the joint check and the pool of replay evidence. A deviation
  from the OGB recipe, which trains on every official train node.
* ``V_select`` / ``V_evidence`` / ``V_check``: 4,000 / 2,000 / 4,000 of the official valid nodes (2018), dealt per
  class with largest-remainder quotas and ``SPLIT_SEED`` (``weightcode.data.splits.stratified_split``); the remaining
  valid nodes form ``valid_unused`` (read only by the optional ceiling reference). The four are disjoint.
* ``test``: the official test nodes (2019 and 2020), written only to ``$WEIGHTCODE_SEALED/ogbn_arxiv/test.npz``.

Class ids: every stored label is ``new_class_of_old[OGB label]`` with ``new_class_of_old = default_rng([SPLIT_SEED,
CLASS_PERMUTATION_STREAM]).permutation(40)``, and dataset.json names the classes "0" .. "39", so a class index ties
to no arXiv category. The splits are dealt on the official labels (so their membership does not depend on this
permutation) and stored with permuted labels.

Protocols (``PROTOCOLS``):

* ``transductive`` = dataset ``ogbn_arxiv`` (the OGB protocol): one graph of every node (2017 to 2020). Node ids are
  ``new_id = new_of_old[ogb_id]`` with ``new_of_old = default_rng([SPLIT_SEED, PERMUTATION_STREAM]).permutation(n)``.
* ``inductive`` = dataset ``ogbn_arxiv_ind``: the deployment snapshot, the subgraph induced by the nodes up to 2018
  (official train and valid nodes, 120,740), with its own compact ids from ``default_rng([SPLIT_SEED,
  INDUCTIVE_PERMUTATION_STREAM])``; its ``graph.npz`` is that snapshot. ``graph_train.npz`` is the subgraph induced
  by the nodes up to 2017 (official train nodes), the base's training graph, with ``nodes`` = the snapshot id of each
  row (ascending). Both are normalized on their own degrees. The sealed test of this protocol runs on the full graph
  of ``ogbn_arxiv`` and its test file; no test node is in this dataset.

Every stored array uses the protocol's ids, so an id carries no year, split or ordering information; the mappings
stay private (``permutation.npz``). A split file holds ``x`` = node ids (int64, ascending), ``y`` = permuted label
and ``idx`` = OGB node id.

``graph.npz`` (what the model and an editor see): the citation graph made symmetric, deduplicated, with one self loop
per node, in new-id order. ``indptr`` int64 [n + 1] and ``indices`` int64 [nnz] (ascending within each row) form the
CSR; ``weight`` float32 [nnz] = d[i]^-1/2 * d[j]^-1/2 with d the row degree including the self loop (computed in
float64, rounded once); ``features`` float32 [n, 128]. No labels, years or split membership.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import shutil
import stat
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.paths import sealed_dir
from weightcode.data.registry import SPLIT_SEED
from weightcode.data.sources import provenance_path, raw_dir
from weightcode.data.splits import stratified_split
from weightcode.data.store import (
    dataset_dir,
    dataset_json_path,
    graph_path,
    load_split,
    read_dataset_json,
    read_json,
    save_split,
    sealed_test_path,
    split_path,
    split_sha256,
    write_json,
)
from weightcode.common.identity import params_sha256

DATASET = "ogbn_arxiv"
SOURCE = "ogbn-arxiv"
ARXIV_URL = "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
ARXIV_ZIP_SHA256 = "49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276"
ARXIV_ZIP_BYTES = 83_058_288
N_NODES = 169_343
N_EDGES = 1_166_243
N_FEATURES = 128
N_CLASSES = 40
PERMUTATION_STREAM = 700
OLD_HOLDOUT_STREAM = 701
CLASS_PERMUTATION_STREAM = 702
INDUCTIVE_PERMUTATION_STREAM = 703
INDUCTIVE_DATASET = "ogbn_arxiv_ind"
PROTOCOLS: Mapping[str, str] = {"transductive": DATASET, "inductive": INDUCTIVE_DATASET}
TRAIN_ORDER: tuple[str, ...] = ("old_select", "old_check", "old_replay", "train")
OLD_SIZES: Mapping[str, int] = {"old_select": 4000, "old_check": 4000, "old_replay": 2000}
VALID_ORDER: tuple[str, ...] = ("V_select", "V_evidence", "V_check", "valid_unused")
VALID_SIZES: Mapping[str, int] = {"V_select": 4000, "V_evidence": 2000, "V_check": 4000}
STORED_SPLITS: tuple[str, ...] = ("train", "old_select", "old_check", "old_replay", "V_select", "V_evidence", "V_check",
                                  "valid_unused")
GRAPH_KEYS: tuple[str, ...] = ("indptr", "indices", "weight", "features")
TRAIN_GRAPH_FILE = "graph_train.npz"
TRAIN_GRAPH_KEYS: tuple[str, ...] = GRAPH_KEYS + ("nodes",)
PERMUTATION_FILE = "permutation.npz"
CLASS_IDS: tuple[str, ...] = tuple(str(index) for index in range(N_CLASSES))
MEMBERS = {
    "features": "arxiv/raw/node-feat.csv.gz",
    "edges": "arxiv/raw/edge.csv.gz",
    "labels": "arxiv/raw/node-label.csv.gz",
    "years": "arxiv/raw/node_year.csv.gz",
    "n_nodes": "arxiv/raw/num-node-list.csv.gz",
    "n_edges": "arxiv/raw/num-edge-list.csv.gz",
    "train": "arxiv/split/time/train.csv.gz",
    "valid": "arxiv/split/time/valid.csv.gz",
    "test": "arxiv/split/time/test.csv.gz",
    "classes": "arxiv/mapping/labelidx2arxivcategeory.csv.gz",
}


def raw_zip_path() -> Path:
    """Installed location of the official zip."""
    return raw_dir() / "ogbn_arxiv" / "arxiv.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install_raw_zip(downloaded: Path, fetch_note: str) -> dict[str, Any]:
    """Copy the downloaded zip into the raw store (read-only), check its sha256 and record its provenance.

    Idempotent: an installed zip must hash to ``ARXIV_ZIP_SHA256``; the provenance entry is rewritten only with the
    same file hash.
    """
    target = raw_zip_path()
    if not target.exists():
        if _sha256(downloaded) != ARXIV_ZIP_SHA256:
            raise RuntimeError(f"{downloaded} does not hash to the recorded ogbn-arxiv sha256")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(downloaded, temporary)
        temporary.replace(target)
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    digest = _sha256(target)
    if digest != ARXIV_ZIP_SHA256 or target.stat().st_size != ARXIV_ZIP_BYTES:
        raise RuntimeError(f"{target} changed: sha256 {digest}")
    with zipfile.ZipFile(target) as archive:
        members = {
            info.filename: {"bytes": info.file_size, "crc32": f"{info.CRC:08x}"}
            for info in archive.infolist()
            if not info.is_dir()
        }
        release = archive.read("arxiv/RELEASE_v1.txt").decode("utf-8").strip()
    entry = {
        "url": ARXIV_URL,
        "files": {"ogbn_arxiv/arxiv.zip": {"sha256": digest, "bytes": ARXIV_ZIP_BYTES}},
        "members": members,
        "release_note": release,
        "fetch": fetch_note,
        "parser": "weightcode.data.arxiv (numpy.loadtxt on the gzip CSV members; the ogb package is not used)",
    }
    record_path = provenance_path()
    record = read_json(record_path)
    previous = record["sources"].get(SOURCE)
    if previous is not None and previous["files"] != entry["files"]:
        raise RuntimeError(f"{SOURCE} raw file changed since it was recorded in {record_path}")
    record["sources"][SOURCE] = entry
    write_json(record_path, record)
    return entry


@dataclass(frozen=True)
class RawGraph:
    """The official arrays in OGB node order."""

    features: np.ndarray
    labels: np.ndarray
    years: np.ndarray
    edges: np.ndarray
    split_idx: dict[str, np.ndarray]
    class_names: tuple[str, ...]


def _read_csv(archive: zipfile.ZipFile, member: str, dtype: Any) -> np.ndarray:
    with archive.open(member) as handle, gzip.GzipFile(fileobj=handle) as unzipped:
        return np.loadtxt(io.TextIOWrapper(unzipped, encoding="ascii"), delimiter=",", dtype=dtype, ndmin=1)


def class_name(category: str) -> str:
    """'arxiv cs na' -> 'cs.NA' (the arXiv subject class code)."""
    parts = category.strip().split(" ")
    if len(parts) != 3 or parts[0] != "arxiv" or parts[1] != "cs" or not parts[2].isalpha():
        raise ValueError(f"unexpected category {category!r}")
    return f"cs.{parts[2].upper()}"


def load_raw(zip_path: Path | None = None) -> RawGraph:
    """Parse the official zip (sha256 checked first)."""
    path = raw_zip_path() if zip_path is None else Path(zip_path)
    if _sha256(path) != ARXIV_ZIP_SHA256:
        raise RuntimeError(f"{path} does not hash to the recorded ogbn-arxiv sha256")
    with zipfile.ZipFile(path) as archive:
        features = _read_csv(archive, MEMBERS["features"], np.float64).astype(np.float32)
        edges = _read_csv(archive, MEMBERS["edges"], np.int64)
        labels = _read_csv(archive, MEMBERS["labels"], np.int64)
        years = _read_csv(archive, MEMBERS["years"], np.int64)
        n_nodes = int(_read_csv(archive, MEMBERS["n_nodes"], np.int64)[0])
        n_edges = int(_read_csv(archive, MEMBERS["n_edges"], np.int64)[0])
        split_idx = {name: _read_csv(archive, MEMBERS[name], np.int64) for name in ("train", "valid", "test")}
        with archive.open(MEMBERS["classes"]) as handle, gzip.GzipFile(fileobj=handle) as unzipped:
            lines = io.TextIOWrapper(unzipped, encoding="ascii").read().strip().splitlines()
    if lines[0] != "label idx,arxiv category":
        raise ValueError(f"unexpected class mapping header {lines[0]!r}")
    pairs = [line.split(",") for line in lines[1:]]
    if [int(index) for index, _ in pairs] != list(range(len(pairs))):
        raise ValueError("class mapping rows are not 0..K-1 in order")
    classes = tuple(class_name(category) for _, category in pairs)
    if (n_nodes, n_edges) != (N_NODES, N_EDGES) or features.shape != (N_NODES, N_FEATURES):
        raise ValueError(f"unexpected sizes: nodes {n_nodes}, edges {n_edges}, features {features.shape}")
    if edges.shape != (N_EDGES, 2) or labels.shape != (N_NODES,) or years.shape != (N_NODES,):
        raise ValueError("unexpected edge, label or year array shape")
    if len(classes) != N_CLASSES or labels.min() < 0 or labels.max() >= N_CLASSES:
        raise ValueError("labels do not match the 40 classes")
    every = np.concatenate(list(split_idx.values()))
    if every.size != N_NODES or np.unique(every).size != N_NODES:
        raise ValueError("official splits are not a partition of the nodes")
    return RawGraph(features, labels, years, edges, split_idx, classes)


def node_permutation(n_nodes: int) -> np.ndarray:
    """``new_of_old``: the new id of every OGB node id."""
    return np.random.default_rng([SPLIT_SEED, PERMUTATION_STREAM]).permutation(n_nodes).astype(np.int64)


def class_permutation() -> np.ndarray:
    """``new_class_of_old``: the stored class id of every OGB label."""
    return np.random.default_rng([SPLIT_SEED, CLASS_PERMUTATION_STREAM]).permutation(N_CLASSES).astype(np.int64)


def build_graph(edges: np.ndarray, n_nodes: int, features: np.ndarray) -> dict[str, np.ndarray]:
    """CSR arrays of the symmetrized, deduplicated graph with self loops and its GCN weights (ids as given)."""
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.min() < 0 or edges.max() >= n_nodes:
        raise ValueError("edges must be [m, 2] node ids in range")
    off_diagonal = edges[edges[:, 0] != edges[:, 1]]
    nodes = np.arange(n_nodes, dtype=np.int64)
    rows = np.concatenate([off_diagonal[:, 0], off_diagonal[:, 1], nodes])
    cols = np.concatenate([off_diagonal[:, 1], off_diagonal[:, 0], nodes])
    keys = np.unique(rows * n_nodes + cols)
    rows, cols = keys // n_nodes, keys % n_nodes
    degree = np.bincount(rows, minlength=n_nodes).astype(np.int64)
    inv_sqrt = 1.0 / np.sqrt(degree.astype(np.float64))
    weight = (inv_sqrt[rows] * inv_sqrt[cols]).astype(np.float32)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(degree, out=indptr[1:])
    return {
        "indptr": indptr,
        "indices": cols.astype(np.int64),
        "weight": weight,
        "features": np.ascontiguousarray(features, dtype=np.float32),
    }


def graph_sha256(graph: Mapping[str, np.ndarray]) -> str:
    """Canonical content hash of graph.npz or graph_train.npz (``params_sha256`` over its arrays)."""
    if tuple(sorted(graph)) not in (tuple(sorted(GRAPH_KEYS)), tuple(sorted(TRAIN_GRAPH_KEYS))):
        raise ValueError(f"graph keys {sorted(graph)} are neither {sorted(GRAPH_KEYS)} nor {sorted(TRAIN_GRAPH_KEYS)}")
    return params_sha256(graph)


@dataclass
class BuiltArxiv:
    """Every stored array of one protocol's dataset plus its manifest (without run provenance).

    ``test`` is None for the inductive dataset (its sealed test is the transductive one); ``train_graph`` is None for
    the transductive dataset (base and ceiling train on graph.npz).
    """

    splits: dict[str, dict[str, np.ndarray]]
    test: dict[str, np.ndarray] | None
    graph: dict[str, np.ndarray]
    train_graph: dict[str, np.ndarray] | None
    permutation: dict[str, np.ndarray]
    manifest: dict[str, Any]

    @property
    def dataset(self) -> str:
        return str(self.manifest["dataset"])


def _split_arrays(ogb_ids: np.ndarray, id_of_ogb: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    new_ids = id_of_ogb[ogb_ids]
    if new_ids.size and new_ids.min() < 0:
        raise ValueError("a split node is not in this protocol's graph")
    order = np.argsort(new_ids, kind="stable")
    return {"x": new_ids[order], "y": labels[ogb_ids[order]].astype(np.int64), "idx": ogb_ids[order].astype(np.int64)}


def _years(raw: RawGraph, arrays: Mapping[str, np.ndarray]) -> dict[str, int]:
    values, counts = np.unique(raw.years[arrays["idx"]], return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def split_membership(raw: RawGraph) -> dict[str, np.ndarray]:
    """OGB node ids of every stored split (``SPLIT_RULE``), dealt on the official labels; shared by both protocols."""
    official_train = raw.split_idx["train"]
    train_sizes = {**OLD_SIZES, "train": official_train.size - sum(OLD_SIZES.values())}
    held = stratified_split(raw.labels[official_train], train_sizes, [SPLIT_SEED, OLD_HOLDOUT_STREAM], N_CLASSES,
                            order=TRAIN_ORDER)
    members = {name: official_train[held[name]] for name in ("train", "old_select", "old_check", "old_replay")}
    valid = raw.split_idx["valid"]
    sizes = {**VALID_SIZES, "valid_unused": valid.size - sum(VALID_SIZES.values())}
    parts = stratified_split(raw.labels[valid], sizes, SPLIT_SEED, N_CLASSES, order=VALID_ORDER)
    for name in VALID_ORDER:
        members[name] = valid[parts[name]]
    return members


SPLIT_RULE = (
    "the official train nodes (years up to 2017) are dealt per class with largest-remainder quotas, each "
    f"class permuted with numpy default_rng([split_seed, {OLD_HOLDOUT_STREAM}]) (one generator, classes in "
    "increasing order), in the order old_select, old_check, old_replay, train (the remainder, which the base "
    "trains on); the official valid nodes (2018) are dealt per class with largest-remainder quotas proportional to "
    "their class counts, each class permuted with numpy default_rng(split_seed) (one generator, classes in "
    "increasing order), in the order V_select, V_evidence, V_check, valid_unused (the remainder); test = official "
    "test nodes (2019 and 2020). Classes are dealt in official label order; the stored labels are permuted "
    "afterwards (classes). Rows of every split are sorted by node id. Both protocols store the same nodes in "
    "each split."
)
ROLES: Mapping[str, str] = {
    "train": "base training (official train minus the old holdouts)",
    "old_select": "old-distribution dev readout",
    "old_check": "old half of the joint check (V_check is the new half)",
    "old_replay": "pool of replay evidence nodes",
    "V_select": "new-distribution (2018) dev readout",
    "V_evidence": "pool of new evidence nodes",
    "V_check": "new half of the joint check",
    "valid_unused": "2018 nodes read only by the ceiling (trained on train plus valid_unused)",
}
GRAPH_FILE_KEYS: Mapping[str, str] = {
    "indptr": "int64 [n + 1]",
    "indices": "int64 [nnz], ascending within each row",
    "weight": "float32 [nnz]",
    "features": "float32 [n, 128], OGB node features",
}
NORMALIZATION = ("weight[i, j] = d_i^-1/2 d_j^-1/2, d = row degree including the self loop in this graph (float64, "
                 "stored float32)")


def _graph_block(graph: Mapping[str, np.ndarray], path: str, construction: str) -> dict[str, Any]:
    n = int(graph["indptr"].size - 1)
    keys = dict(GRAPH_FILE_KEYS)
    if "nodes" in graph:
        keys["nodes"] = "int64 [n], the deployment-graph node id of each row, ascending"
    return {
        "path": path,
        "sha256": graph_sha256(graph),
        "n_nodes": n,
        "n_features": int(graph["features"].shape[1]),
        "nnz": int(graph["indices"].size),
        "n_undirected_edges": int((graph["indices"].size - n) // 2),
        "construction": construction,
        "normalization": NORMALIZATION,
        "keys": keys,
        "excludes": "labels, years and split membership",
    }


def _manifest(raw: RawGraph, dataset: str, protocol: str, task: str, id_rule: str,
              splits: Mapping[str, Mapping[str, np.ndarray]], graph_block: dict[str, Any],
              permutation: Mapping[str, np.ndarray], permutation_note: str) -> dict[str, Any]:
    return {
        "schema": "weightcode.dataset.v0",
        "dataset": dataset,
        "protocol": protocol,
        "source": SOURCE,
        "task": task,
        "class_names": list(CLASS_IDS),
        "classes": {
            "names": "the class ids 0 .. 39 only; category names are withheld",
            "rule": "stored label = new_class_of_old[OGB label], new_class_of_old = numpy.random.default_rng("
                    f"[split_seed, {CLASS_PERMUTATION_STREAM}]).permutation(40), kept in {PERMUTATION_FILE}",
        },
        "input": {"kind": "node_ids", "dtype": "int64", "n_nodes": graph_block["n_nodes"], "id_rule": id_rule},
        "graph": graph_block,
        "permutation": {
            "path": PERMUTATION_FILE,
            "sha256": params_sha256(permutation),
            "visibility": f"private: {permutation_note}",
        },
        "split_seed": SPLIT_SEED,
        "split_rule": SPLIT_RULE,
        "split_sizes": {name: int(arrays["y"].size) for name, arrays in splits.items()},
        "official_split_sizes": {name: int(ids.size) for name, ids in raw.split_idx.items()},
        "roles": dict(ROLES),
        "splits": {
            name: {
                "path": f"splits/{name}.npz",
                "n": int(arrays["y"].size),
                "sha256": split_sha256(arrays),
                "class_counts": np.bincount(arrays["y"], minlength=N_CLASSES).tolist(),
                "years": _years(raw, arrays),
            }
            for name, arrays in splits.items()
        },
    }


def build(raw: RawGraph) -> BuiltArxiv:
    """The transductive dataset ``ogbn_arxiv``: node and class ids permuted, graph.npz of every node, every split."""
    n = raw.labels.size
    new_of_old = node_permutation(n)
    old_of_new = np.argsort(new_of_old)
    new_class_of_old = class_permutation()
    labels = new_class_of_old[raw.labels]
    splits = {name: _split_arrays(ids, new_of_old, labels) for name, ids in split_membership(raw).items()}
    test = _split_arrays(raw.split_idx["test"], new_of_old, labels)
    graph = build_graph(new_of_old[raw.edges], n, raw.features[old_of_new])
    permutation = {"new_of_old": new_of_old, "new_class_of_old": new_class_of_old}
    block = _graph_block(graph, "graph.npz", "edge list made symmetric and deduplicated, OGB self loops dropped (none "
                         "present), one self loop added per node; every node of 2017 to 2020; rows and columns in "
                         "node-id order")
    manifest = _manifest(
        raw, DATASET, "transductive",
        "transductive node classification, official OGB time split: every node is in the graph during training",
        "new id = new_of_old[OGB node id], new_of_old = numpy.random.default_rng([split_seed, "
        f"{PERMUTATION_STREAM}]).permutation(n_nodes)",
        splits, block, permutation, "maps node ids and class ids back to OGB ids")
    manifest["test"] = {
        "location": f"$WEIGHTCODE_SEALED/{DATASET}/test.npz",
        "n": int(test["y"].size),
        "sha256": split_sha256(test),
        "years": _years(raw, test),
    }
    return BuiltArxiv(splits=splits, test=test, graph=graph, train_graph=None, permutation=permutation,
                      manifest=manifest)


def build_inductive(raw: RawGraph) -> BuiltArxiv:
    """The inductive dataset ``ogbn_arxiv_ind``: the deployment snapshot (nodes up to 2018) as graph.npz, the
    training snapshot (nodes up to 2017) as graph_train.npz, compact ids of its own, the same split membership."""
    n = raw.labels.size
    snapshot = np.sort(np.concatenate([raw.split_idx["train"], raw.split_idx["valid"]]))
    ind_perm = np.random.default_rng([SPLIT_SEED, INDUCTIVE_PERMUTATION_STREAM]).permutation(snapshot.size)
    id_of_ogb = np.full(n, -1, dtype=np.int64)
    id_of_ogb[snapshot] = ind_perm
    ogb_of_new = np.empty(snapshot.size, dtype=np.int64)
    ogb_of_new[ind_perm] = snapshot
    new_class_of_old = class_permutation()
    labels = new_class_of_old[raw.labels]
    splits = {name: _split_arrays(ids, id_of_ogb, labels) for name, ids in split_membership(raw).items()}
    mapped = id_of_ogb[raw.edges]
    inside = mapped[(mapped >= 0).all(axis=1)]
    graph = build_graph(inside, snapshot.size, raw.features[ogb_of_new])
    nodes = np.sort(id_of_ogb[raw.split_idx["train"]])
    row_of_new = np.full(snapshot.size, -1, dtype=np.int64)
    row_of_new[nodes] = np.arange(nodes.size, dtype=np.int64)
    rows = row_of_new[inside]
    train_graph = {**build_graph(rows[(rows >= 0).all(axis=1)], nodes.size, graph["features"][nodes]),
                   "nodes": nodes}
    permutation = {"ogb_of_new": ogb_of_new, "new_class_of_old": new_class_of_old}
    block = _graph_block(graph, "graph.npz", "the deployment snapshot: the subgraph induced by the official train and "
                         "valid nodes (up to 2018), edges made symmetric and deduplicated, one self loop added per "
                         "node; rows and columns in node-id order")
    manifest = _manifest(
        raw, INDUCTIVE_DATASET, "inductive",
        "inductive node classification, official OGB time split: the base trains on the snapshot up to 2017 "
        "(graph_train.npz) and is deployed on the snapshot up to 2018 (graph.npz)",
        "new id = ind_perm[k] for the k-th official train or valid node in OGB id order, ind_perm = "
        f"numpy.random.default_rng([split_seed, {INDUCTIVE_PERMUTATION_STREAM}]).permutation(n_nodes)",
        splits, block, permutation, "ogb_of_new maps node ids to OGB ids; new_class_of_old maps OGB labels")
    manifest["train_graph"] = _graph_block(
        train_graph, TRAIN_GRAPH_FILE, "the training snapshot: the subgraph of graph.npz induced by the official train "
        "nodes (up to 2017), normalized on its own degrees; row r is node nodes[r] of graph.npz")
    manifest["test"] = {
        "location": f"$WEIGHTCODE_SEALED/{DATASET}/test.npz",
        "graph": f"the full graph of {DATASET} (its node ids); no test node is in this dataset",
        "n": int(raw.split_idx["test"].size),
    }
    return BuiltArxiv(splits=splits, test=None, graph=graph, train_graph=train_graph, permutation=permutation,
                      manifest=manifest)


def _save_arrays(path: Path, arrays: Mapping[str, np.ndarray], expected: str, *, private_mode: bool = False) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
    if private_mode:
        os.chmod(temporary, 0o600)
    with np.load(temporary, allow_pickle=False) as archive:
        reloaded = params_sha256({key: archive[key] for key in archive.files})
    if reloaded != expected:
        temporary.unlink()
        raise RuntimeError(f"reload hash mismatch for {path}")
    temporary.replace(path)


def write(built: BuiltArxiv, forward_sha256: str, provenance: Mapping[str, Any], run_info: Mapping[str, Any]) -> dict[str, Any]:
    """Write one protocol's graphs, permutation, splits, sealed test (transductive) and dataset.json (refuses an
    existing one)."""
    dataset = built.dataset
    manifest_path = dataset_json_path(dataset)
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; use verify instead of rebuilding")
    root = dataset_dir(dataset)
    root.mkdir(parents=True, exist_ok=True)
    _save_arrays(graph_path(dataset), built.graph, built.manifest["graph"]["sha256"])
    if built.train_graph is not None:
        _save_arrays(root / TRAIN_GRAPH_FILE, built.train_graph, built.manifest["train_graph"]["sha256"])
    _save_arrays(root / PERMUTATION_FILE, built.permutation, built.manifest["permutation"]["sha256"], private_mode=True)
    for name, arrays in built.splits.items():
        if save_split(split_path(dataset, name), arrays) != built.manifest["splits"][name]["sha256"]:
            raise RuntimeError(f"{name}: written hash differs from the built hash")
    if built.test is not None:
        sealed = sealed_dir() / dataset
        sealed.mkdir(mode=0o700, exist_ok=True)
        sealed.chmod(0o700)
        if save_split(sealed_test_path(dataset), built.test, private_mode=True) != built.manifest["test"]["sha256"]:
            raise RuntimeError("test: written hash differs from the built hash")
    manifest = {
        **built.manifest,
        "forward_path": "forward.py",
        "forward_sha256": forward_sha256,
        "source_provenance": dict(provenance),
        "created_by": dict(run_info),
    }
    write_json(manifest_path, manifest)
    return manifest


def load_graph(dataset: str = DATASET, name: str = "graph.npz") -> dict[str, np.ndarray]:
    """Load one stored graph file of ``dataset`` (graph.npz or graph_train.npz) without pickle."""
    if name not in ("graph.npz", TRAIN_GRAPH_FILE):
        raise ValueError(f"{name!r} is not a stored graph file")
    with np.load(dataset_dir(dataset) / name, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def verify(built: BuiltArxiv) -> dict[str, Any]:
    """Compare a fresh build with dataset.json and the private files of its dataset (the sealed file is not read)."""
    dataset = built.dataset
    recorded = read_dataset_json(dataset)
    report: dict[str, Any] = {}
    for name, arrays in built.splits.items():
        fresh, on_disk = split_sha256(arrays), split_sha256(load_split(dataset, name))
        expected = recorded["splits"][name]["sha256"]
        report[name] = {"match": fresh == expected == on_disk}
    if built.test is not None:
        report["test"] = {"match": split_sha256(built.test) == recorded["test"]["sha256"]}
    report["graph"] = {
        "match": graph_sha256(built.graph) == recorded["graph"]["sha256"] == graph_sha256(load_graph(dataset))
    }
    if built.train_graph is not None:
        report["train_graph"] = {"match": graph_sha256(built.train_graph) == recorded["train_graph"]["sha256"]
                                 == graph_sha256(load_graph(dataset, TRAIN_GRAPH_FILE))}
    with np.load(dataset_dir(dataset) / PERMUTATION_FILE, allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    report["permutation"] = {
        "match": params_sha256(built.permutation) == recorded["permutation"]["sha256"] == params_sha256(stored)
    }
    fields = {key: value for key, value in recorded.items() if key in built.manifest}
    report["manifest"] = {"match": fields == built.manifest}
    return report
