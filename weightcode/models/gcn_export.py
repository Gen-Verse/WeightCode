"""Export and import of the ogbn-arxiv GCN students (arch ``gcn``) in the store layout of ``weightcode.models.export``.

Every function takes the protocol's store dataset (``weightcode.data.arxiv.PROTOCOLS``: ``ogbn_arxiv`` transductive,
``ogbn_arxiv_ind`` inductive). ``export_gcn_model`` writes one student under ``models_dir(dataset)`` (or
``models_root``): model.npz, model_spec.json (``weightcode.models.gcn.build_gcn_model_spec``) and private.json, after the
numpy forward.py has passed ``weightcode.models.parity.check_logit_parity`` at every node of the deployment graph
(graph.npz; float64 exactness and the logit-scaled float32 bound). Its index row has ``INDEX_KEYS``; ``vselect_acc``
is forward.py's accuracy on the 2018 V_select nodes, and private.json adds old_select (papers up to 2017, held out of
training).

Roles: ``base`` trains on the ``train`` split; ``ceiling`` on ``train`` plus ``valid_unused`` (the 2018 nodes
outside every readout, evidence and check set). Ceilings are reference points only and are never edited.
``newclass`` (PREREG E7) trains on the ``train`` nodes of its ``train_classes`` only, with all 40 outputs kept; the
nodes of the other classes stay in the training graph unlabelled. Its students live in the fewshot store
(``weightcode.models.export.fewshot_models_dir``) with ``FEWSHOT_INDEX_KEYS`` rows. Training graph (``training_graph``):
graph.npz, except an inductive base or newclass student, which trains on graph_train.npz (the snapshot up to 2017,
BatchNorm statistics from those nodes only) and is deployed on graph.npz.

``import_gcn_models`` brings the students of a fetched NSCC run into the local store. Before anything is copied it
re-checks every model with the local files: the parameter hash, the provenance in private.json (training split
hashes and the graph hash must be the local ones), the public spec rebuilt from the parameters, parity, and both
accuracies recomputed with the local forward.py, which must equal the exported values.
"""

from __future__ import annotations

import dataclasses
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightcode.common.identity import Params, load_params, params_sha256, save_params
from weightcode.data import arxiv as A
from weightcode.data.store import forward_path, load_split, models_dir, read_dataset_json, read_json, split_sha256, write_json
from weightcode.evalkit.patch_check import ForwardFn
from weightcode.models import gcn as G
from weightcode.models.export import (
    FEWSHOT_INDEX_KEYS,
    INDEX_KEYS,
    ParityError,
    all_model_ids,
    fewshot_models_dir,
    load_forward_no_cache,
    new_model_id,
    read_fewshot_index,
    read_index,
    write_fewshot_index,
    write_index,
)
from weightcode.models.importer import MODEL_FILES, annotate_import, checked_forward, models_source, run_record
from weightcode.models.parity import PARITY_RULE, check_logit_parity

GCN_PRIVATE_SCHEMA = "weightcode.model_private.gcn.v1"
GCN_ROLES: Mapping[str, tuple[str, ...]] = {"base": ("train",), "ceiling": ("train", "valid_unused"), "newclass": ("train",)}
CLASS_SUBSET_ROLES: tuple[str, ...] = ("newclass",)
"""Roles trained on a subset of the classes: they need ``train_classes`` and live in the fewshot store."""
SNAPSHOT_ROLES: tuple[str, ...] = ("base", "newclass")
"""Roles that train on graph_train.npz in the inductive protocol."""
READOUT_SPLITS: tuple[str, ...] = ("V_select", "old_select")
OGB_DEVIATIONS = (
    "OGB reference GCN recipe (3 layers, hidden 256, BatchNorm, dropout 0.5, Adam lr 0.01, 500 epochs, full batch) "
    "with these deviations: the old holdouts (old_select, old_check, old_replay; 10,000 nodes) are removed from the "
    "official train nodes; the model after the last epoch is kept, with no epoch selection on the valid nodes; "
    "node ids and class ids are permuted (the graph and features are the official ones); in the inductive protocol "
    "the base trains on the subgraph induced by the nodes up to 2017 and is deployed on the subgraph up to 2018."
)
GRAPH_FILE = "graph.npz"


def checked_split(dataset: str, name: str, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """A stored split, refused unless it hashes as dataset.json records."""
    arrays = load_split(dataset, name)
    if split_sha256(arrays) != manifest["splits"][name]["sha256"]:
        raise RuntimeError(f"{dataset}/{name} does not match its dataset.json hash")
    return arrays


def checked_graph(dataset: str, manifest: Mapping[str, Any], name: str = GRAPH_FILE) -> dict[str, np.ndarray]:
    """graph.npz (or graph_train.npz), refused unless it hashes as dataset.json records."""
    graph = A.load_graph(dataset, name)
    block = manifest["graph"] if name == GRAPH_FILE else manifest["train_graph"]
    if block["path"] != name or A.graph_sha256(graph) != block["sha256"]:
        raise RuntimeError(f"{dataset}/{name} does not match its dataset.json hash")
    return graph


def training_graph_file(role: str, manifest: Mapping[str, Any]) -> str:
    """The graph file ``role`` trains on: graph_train.npz for the inductive base, graph.npz otherwise."""
    if role not in GCN_ROLES:
        raise ValueError(f"role must be one of {sorted(GCN_ROLES)}")
    if manifest.get("protocol") == "inductive" and role in SNAPSHOT_ROLES:
        return A.TRAIN_GRAPH_FILE
    return GRAPH_FILE


def training_graph(dataset: str, role: str, manifest: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray | None, dict[str, Any]]:
    """(graph arrays for ``weightcode.models.gcn.train_gcn``, row -> node id map or None, provenance block)."""
    name = training_graph_file(role, manifest)
    graph = checked_graph(dataset, manifest, name)
    nodes = graph.pop("nodes", None)
    block = manifest["graph"] if name == GRAPH_FILE else manifest["train_graph"]
    return graph, nodes, {"path": name, "sha256": block["sha256"], "n_nodes": block["n_nodes"]}


def check_train_classes(role: str, train_classes: Sequence[int] | None) -> list[int] | None:
    """``train_classes`` as a sorted proper subset of the 40 classes for a class-subset role, None otherwise."""
    if role not in CLASS_SUBSET_ROLES:
        if train_classes is not None:
            raise ValueError(f"role {role!r} trains on every class; train_classes must be None")
        return None
    if train_classes is None:
        raise ValueError(f"role {role!r} needs train_classes")
    classes = [int(c) for c in train_classes]
    if not classes or classes != sorted(set(classes)) or classes[0] < 0 or classes[-1] >= A.N_CLASSES \
            or len(classes) == A.N_CLASSES:
        raise ValueError(f"train_classes must be a sorted proper subset of 0..{A.N_CLASSES - 1}, got {classes}")
    return classes


def training_nodes(dataset: str, role: str, manifest: Mapping[str, Any], nodes: np.ndarray | None = None,
                   train_classes: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Labelled training nodes of ``role`` (``GCN_ROLES``) as rows of the training graph, with each split's hash.

    ``nodes`` (graph_train.npz) maps training-graph rows to node ids; every labelled node must be one of them. A
    class-subset role keeps only the nodes whose label is in ``train_classes``.
    """
    if role not in GCN_ROLES:
        raise ValueError(f"role must be one of {sorted(GCN_ROLES)}")
    classes = check_train_classes(role, train_classes)
    parts = [checked_split(dataset, name, manifest) for name in GCN_ROLES[role]]
    x = np.concatenate([part["x"] for part in parts])
    y = np.concatenate([part["y"] for part in parts])
    if classes is not None:
        keep = np.isin(y, classes)
        x, y = x[keep], y[keep]
    if np.unique(x).size != x.size:
        raise ValueError(f"{role}: training splits overlap")
    if nodes is not None:
        rows = np.searchsorted(nodes, x)
        if np.any(rows >= nodes.size) or not np.array_equal(nodes[np.minimum(rows, nodes.size - 1)], x):
            raise ValueError(f"{role}: a labelled node is not in the training graph")
        x = rows.astype(np.int64)
    return x, y, {name: manifest["splits"][name]["sha256"] for name in GCN_ROLES[role]}


def build_spec(model_id: str, config: G.GCNConfig, params: Mapping[str, np.ndarray],
               manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The public model_spec.json of an arxiv student."""
    return G.build_gcn_model_spec(model_id, config, list(manifest["class_names"]), params)


def torch_cpu_logits(config: G.GCNConfig, params: Params, graph: Mapping[str, np.ndarray],
                     dtype: torch.dtype) -> np.ndarray:
    """CPU torch eval logits of every node in ``dtype`` (float64 copy)."""
    adj, features = G.graph_tensors(graph, torch.device("cpu"), dtype)
    return G.logits_all(G.build_gcn(config, params, dtype), adj, features)


def gcn_parity(config: G.GCNConfig, params: Params, graph: Mapping[str, np.ndarray], forward: ForwardFn,
               numpy_logits: np.ndarray | None = None) -> dict[str, float]:
    """``check_logit_parity`` of forward.py at every node against CPU float64 and float32 torch rebuilds."""
    n = int(np.asarray(graph["features"]).shape[0])
    mine = forward(dict(params), np.arange(n, dtype=np.int64)) if numpy_logits is None else numpy_logits
    exact = torch_cpu_logits(config, params, graph, torch.float64)
    return check_logit_parity(mine, exact, torch_cpu_logits(config, params, graph, torch.float32))


def accuracies(logits: np.ndarray, splits: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, float]:
    """Accuracy of all-node ``logits`` on each readout split."""
    return {name: float(np.mean(np.argmax(logits[arrays["x"]], axis=1) == arrays["y"])) for name, arrays in splits.items()}


def n_labelled(manifest: Mapping[str, Any], training_splits: Mapping[str, str], classes: Sequence[int] | None) -> int:
    """Labelled training nodes: every node of the training splits, or those of ``classes`` (dataset.json counts)."""
    if classes is None:
        return int(sum(manifest["splits"][name]["n"] for name in training_splits))
    return int(sum(manifest["splits"][name]["class_counts"][c] for name in training_splits for c in classes))


def subset_accuracies(logits: np.ndarray, vselect: Mapping[str, np.ndarray], train_classes: Sequence[int]) -> dict[str, Any]:
    """A class-subset student on V_select: 40-way accuracy on the train-class rows, and how many of the other rows
    and of all rows it predicts in a class outside ``train_classes``."""
    pred = np.argmax(logits[vselect["x"]], axis=1)
    in_train = np.isin(vselect["y"], train_classes)
    return {"final_vselect_acc_train_classes": float(np.mean(pred[in_train] == vselect["y"][in_train])),
            "n_vselect_train_classes": int(in_train.sum()),
            "n_vselect_pred_outside_train_classes": int(np.count_nonzero(~np.isin(pred, train_classes)))}


def export_gcn_model(
    *,
    dataset: str,
    manifest: Mapping[str, Any],
    role: str,
    seed: int,
    config: G.GCNTrainConfig,
    result: G.GCNTrainResult,
    training_splits: Mapping[str, str],
    train_graph: Mapping[str, Any],
    determinism: Mapping[str, Any],
    run_info: Mapping[str, Any],
    models_root: Path | None = None,
    extra_private: Mapping[str, Any] | None = None,
    train_classes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Write one student's files, verify parity and return its index row (module docstring).

    ``train_graph`` is the provenance block of ``training_graph``; parity and readouts run on graph.npz. A
    class-subset role (``CLASS_SUBSET_ROLES``) needs ``train_classes``, goes to the fewshot store by default and
    returns a ``FEWSHOT_INDEX_KEYS`` row.
    """
    classes = check_train_classes(role, train_classes)
    if role not in GCN_ROLES or dict(training_splits).keys() != set(GCN_ROLES[role]):
        raise ValueError(f"role {role!r} must train on {GCN_ROLES.get(role)}, got {sorted(training_splits)}")
    if dict(train_graph)["path"] != training_graph_file(role, manifest):
        raise ValueError(f"role {role!r} must train on {training_graph_file(role, manifest)}, got {train_graph}")
    graph = checked_graph(dataset, manifest)
    model_config = dataclasses.replace(result.model.config, n_nodes=int(manifest["graph"]["n_nodes"]))
    default_root = models_dir(dataset) if classes is None else fewshot_models_dir(dataset)
    root = default_root if models_root is None else Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    model_id = new_model_id(all_model_ids(dataset) | {p.name for p in root.iterdir() if p.is_dir()})
    staging = root / f".staging-{model_id}"
    staging.mkdir()

    params = G.export_gcn_params(result.model)
    digest = save_params(staging / "model.npz", params)
    reloaded = load_params(staging / "model.npz")
    if params_sha256(reloaded) != digest:
        raise RuntimeError("reloaded parameters hash differently")
    forward, forward_sha = load_forward_no_cache(forward_path(dataset))
    if forward_sha != manifest["forward_sha256"]:
        raise RuntimeError(f"forward.py sha256 {forward_sha} != dataset.json {manifest['forward_sha256']}")

    device = next(result.model.parameters()).device
    adj, features = G.graph_tensors(graph, device)
    device_logits = G.logits_all(result.model, adj, features)
    started = time.perf_counter()
    numpy_logits = forward(dict(reloaded), np.arange(model_config.n_nodes, dtype=np.int64)).astype(np.float64)
    forward_time = time.perf_counter() - started
    try:
        parity = gcn_parity(model_config, reloaded, graph, forward, numpy_logits)
    except ParityError as error:
        raise ParityError(f"{dataset} {role} seed {seed}: {error}") from error
    readouts = {name: checked_split(dataset, name, manifest) for name in READOUT_SPLITS}
    acc = accuracies(numpy_logits, readouts)
    acc_device = accuracies(device_logits, readouts)
    subset = {} if classes is None else {"train_classes": classes,
                                         **subset_accuracies(numpy_logits, readouts["V_select"], classes)}
    disagree = {name: int(np.count_nonzero(np.argmax(numpy_logits[arrays["x"]], 1) != np.argmax(device_logits[arrays["x"]], 1)))
                for name, arrays in readouts.items()}
    spec = build_spec(model_id, model_config, reloaded, manifest)
    write_json(staging / "model_spec.json", spec)
    private = {
        "schema": GCN_PRIVATE_SCHEMA,
        "model_id": model_id,
        "dataset": dataset,
        "protocol": manifest["protocol"],
        "condition": None,
        "seed": seed,
        "arch": config.arch,
        "role": role,
        "model_config": model_config.as_dict(),
        "training_config": config.as_dict(),
        "recipe": OGB_DEVIATIONS,
        "train_data": {
            "splits": dict(training_splits),
            "n_labelled": n_labelled(manifest, training_splits, classes),
            "graph_sha256": dict(train_graph)["sha256"],
            "graph": dict(train_graph),
            "deployment_graph_sha256": manifest["graph"]["sha256"],
            "note": "full batch: every node of the training graph enters every step with its features and edges, and "
                    "BatchNorm statistics cover those nodes; labels only of these splits",
        },
        "split_seed": manifest["split_seed"],
        "determinism": dict(determinism),
        "train_log": result.log,
        "params_sha256": digest,
        "n_parameters": int(sum(value.size for value in reloaded.values())),
        "layer_norms": G.layer_norms(reloaded),
        "forward_sha256": forward_sha,
        "final_vselect_acc": acc["V_select"],
        "final_old_select_acc": acc["old_select"],
        "readout_definition": "numpy forward.py accuracy on the 4000 V_select (2018) and 4000 old_select (up to 2017) "
                              "nodes",
        "readout_acc_training_device": acc_device,
        "readout_disagreements_numpy_vs_training_device": disagree,
        "parity": {
            "nodes": "every node of the deployment graph (graph.npz)",
            "n_nodes": int(model_config.n_nodes),
            "reference": "torch eval logits, CPU float32, parameters reloaded from model.npz",
            "max_abs_diff": parity["max_abs_diff"],
            "tolerance": parity["tolerance"],
            "logit_scale": parity["logit_scale"],
            "rule": dict(PARITY_RULE),
            "float64": {"reference": "torch eval logits, CPU float64, parameters reloaded from model.npz",
                        "max_abs_diff": parity["float64_max_abs_diff"],
                        "max_rel_diff": parity["float64_max_rel_diff"],
                        "tolerance": parity["float64_tolerance"]},
            "max_abs_diff_vs_training_device": float(np.max(np.abs(numpy_logits - device_logits))),
            "numpy_forward_time_s": forward_time,
        },
        "wall_time_s": result.wall_time_s,
        **subset,
        **dict(run_info),
        **(dict(extra_private) if extra_private else {}),
    }
    write_json(staging / "private.json", private)
    staging.rename(root / model_id)
    row = {"model_id": model_id, "seed": seed, "arch": config.arch, "role": role, "params_sha256": digest,
           "vselect_acc": acc["V_select"]}
    if classes is not None:
        row.update(train_classes=classes, vselect_acc_train_classes=subset["final_vselect_acc_train_classes"])
    return row


def _check_one(dataset: str, row: Mapping[str, Any], source: Path, manifest: Mapping[str, Any], forward: ForwardFn,
               forward_sha: str, graph: Mapping[str, np.ndarray]) -> dict[str, Any]:
    subset_role = row.get("role") in CLASS_SUBSET_ROLES
    keys = FEWSHOT_INDEX_KEYS if subset_role else INDEX_KEYS
    if tuple(row) != keys:
        raise ValueError(f"index row keys {tuple(row)} != {keys}")
    classes = check_train_classes(row["role"], row["train_classes"] if subset_role else None)
    model_id = row["model_id"]
    folder = source / model_id
    files = sorted(path.name for path in folder.iterdir())
    if files != sorted(MODEL_FILES):
        raise ValueError(f"{folder}: files {files} != {sorted(MODEL_FILES)}")
    params = load_params(folder / "model.npz")
    private = read_json(folder / "private.json")
    expected = {"model_id": model_id, "dataset": dataset, "condition": None, "seed": row["seed"], "arch": row["arch"],
                "role": row["role"], "params_sha256": row["params_sha256"], "forward_sha256": forward_sha,
                "schema": GCN_PRIVATE_SCHEMA}
    found = {key: private.get(key) for key in expected}
    if params_sha256(params) != row["params_sha256"] or found != expected or private.get("train_classes") != classes:
        raise ValueError(f"{model_id}: private.json or model.npz disagrees with the index ({found} vs {expected})")
    if private.get("smoke"):
        raise ValueError(f"{model_id} is a smoke-test model and is never imported")
    role_splits = {name: manifest["splits"][name]["sha256"] for name in GCN_ROLES[row["role"]]}
    name = training_graph_file(row["role"], manifest)
    block = manifest["graph"] if name == GRAPH_FILE else manifest["train_graph"]
    expected_graph = {"path": name, "sha256": block["sha256"], "n_nodes": block["n_nodes"]}
    if private["train_data"]["splits"] != role_splits or private["train_data"]["graph"] != expected_graph \
            or private["train_data"]["deployment_graph_sha256"] != manifest["graph"]["sha256"]:
        raise ValueError(f"{model_id}: trained on splits or a graph that are not the local ones")
    config = G.GCNConfig(**private["model_config"])
    spec = read_json(folder / "model_spec.json")
    if spec != json_roundtrip(build_spec(model_id, config, params, manifest)):
        raise ValueError(f"{model_id}: model_spec.json differs from the spec rebuilt from its parameters")
    logits = forward(dict(params), np.arange(config.n_nodes, dtype=np.int64)).astype(np.float64)
    try:
        parity = gcn_parity(config, params, graph, forward, logits)
    except ParityError as error:
        raise ValueError(f"{model_id}: local parity: {error}") from error
    readouts = {name: checked_split(dataset, name, manifest) for name in READOUT_SPLITS}
    acc = accuracies(logits, readouts)
    if (acc["V_select"], acc["old_select"]) != (row["vselect_acc"], private["final_old_select_acc"]) \
            or acc["V_select"] != private["final_vselect_acc"]:
        raise ValueError(f"{model_id}: local accuracies {acc} differ from the exported ones")
    if classes is not None:
        subset = subset_accuracies(logits, readouts["V_select"], classes)
        if subset != {key: private.get(key) for key in subset} \
                or subset["final_vselect_acc_train_classes"] != row["vselect_acc_train_classes"]:
            raise ValueError(f"{model_id}: local train-class accuracies {subset} differ from the exported ones")
        acc = {**acc, "V_select_train_classes": subset["final_vselect_acc_train_classes"]}
    return {"model_id": model_id, "local_parity_max_abs_diff": parity["max_abs_diff"],
            "local_parity_tolerance": parity["tolerance"],
            "local_parity_float64_max_rel_diff": parity["float64_max_rel_diff"], "local_acc": acc}


def json_roundtrip(payload: Any) -> Any:
    import json

    return json.loads(json.dumps(payload))


def import_gcn_models(dataset: str, run_dir: Path, note: str | None = None, *,
                      fewshot: bool = False) -> list[dict[str, Any]]:
    """Check every student of a fetched NSCC run, copy them into the local store and index them.

    Each imported private.json gains an ``import`` block (run record, training environment, local re-checks and
    ``note``). Refuses a model id already used under ``dataset`` and a (role, seed) already in the local index (a
    (role, seed, train classes) in the fewshot index). Nothing is copied unless every model passes. ``fewshot``
    imports the run's class-subset students (``CLASS_SUBSET_ROLES``) into the fewshot store.
    """
    record = run_record(Path(run_dir))
    source = models_source(Path(run_dir), dataset, None)
    if fewshot:
        source = source.parent / "fewshot" / "models"
    manifest = read_dataset_json(dataset)
    forward, forward_sha = checked_forward(dataset, manifest)
    graph = checked_graph(dataset, manifest)
    rows = read_json(source / "index.json")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source}/index.json lists no models")
    local = read_fewshot_index(dataset) if fewshot else read_index(dataset)
    used_ids = all_model_ids(dataset)

    def key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
        return (entry["role"], entry["seed"], tuple(entry["train_classes"])) if fewshot else (entry["role"], entry["seed"])

    taken = {key(entry) for entry in local}
    checks = []
    for row in rows:
        if (row.get("role") in CLASS_SUBSET_ROLES) != fewshot:
            raise ValueError(f"{row['model_id']}: role {row.get('role')!r} does not belong in this import (fewshot={fewshot})")
        if row["model_id"] in used_ids:
            raise ValueError(f"model id {row['model_id']} is already used under {dataset}")
        if key(row) in taken:
            raise ValueError(f"{dataset}: {key(row)} is already in the local index")
        taken.add(key(row))
        checks.append(_check_one(dataset, row, source, manifest, forward, forward_sha, graph))
    root = fewshot_models_dir(dataset) if fewshot else models_dir(dataset)
    root.mkdir(parents=True, exist_ok=True)
    for row, check in zip(rows, checks):
        staging = root / f".staging-{row['model_id']}"
        shutil.copytree(source / row["model_id"], staging)
        private = read_json(staging / "private.json")
        annotate_import(staging, {**record, "trained_with": {"versions": private["versions"],
                                                             "device": private["device"]},
                                  "local_checks": check, "note": note})
        staging.rename(root / row["model_id"])
    if fewshot:
        write_fewshot_index(dataset, local + [dict(row) for row in rows])
    else:
        write_index(dataset, local + [dict(row) for row in rows])
    return checks
