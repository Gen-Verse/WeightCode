"""Rule graphnew-v1 (PREREG E7): instances that ask an edit to add classes a graph network never trained on.

A graphnew student is an ogbn-arxiv GCN of the inductive protocol (dataset ``ogbn_arxiv_ind``, role ``newclass`` of
``weightcode.models.gcn_export``) trained on the train nodes of the train classes of the class draw
(``weightcode.data.arxiv_classes``, sidecar graphnew_classes.json) with all 40 outputs kept, so the outputs of the new
classes never saw a labelled node of their class. There is one instance per student. The layout is fewshot-v1's
(``weightcode.evalkit.fewshot``) plus the graph, so ``weightcode.baselines.instance_io.load_instance`` loads it unchanged::

    $WEIGHTCODE_PRIVATE/graphnew_instances/
        index.json                       {"schema", "instances": [active rows], "superseded": [rows]}
        changes.jsonl                    one line per created instance
        g<8 hex>/
            model.npz, model_spec.json   byte copies of the student's files
            forward.py                   byte copy of the dataset forward; it reads graph.npz next to itself
            graph.npz                    byte copy of the deployment snapshot (nodes up to 2018), no labels
            evidence.npz                 the evidence, keys EVIDENCE_KEYS; x_* are int64 node ids into graph.npz
            task.json                    schema weightcode.task.graphnew.v1
            private.json                 schema weightcode.instance_private.graphnew.v1, never editor-visible

Evidence of rule graphnew-v1:

- new part: N_PER_NEW_CLASS V_evidence nodes of each new class, in class order, drawn without replacement with
  ``default_rng(graphnew_seed(params_sha256, dataset))`` (``fewshot.draw_new_rows``);
- old part: the replay of rule graphnew-replay-v1, N_REPLAY old_replay nodes of the train classes, uniform without
  replacement with ``default_rng([SPLIT_SEED, REPLAY_STREAM])``. It is one draw per class draw, shared by every
  student; the students never trained on old_replay (it is held out of base training, as in E6);
- pred_*: the student's forward.py argmax on those nodes;
- split ids: as in fewshot-v1.

forward.py runs the whole graph on every call and returns the rows of the requested node ids, so every forward here
is one call over all the nodes it needs (``predict_nodes``). Check (``make_graphnew_oracle``): V_check cut by class
(``graphnew_check_parts``), S_new on the rows of the new classes, S_old on the rest, S = S_new + S_old; each part is
one forward call. Readout (``graphnew_readout``): ``fewshot.class_split_readout`` on V_select (40-way), plus
class-balanced new-class accuracy, the new-class rows by hop distance to the nearest new-class evidence node (and to
the nearest evidence node of their own class), and old_select (nodes up to 2017) cut by class.

Limits: task.json states the default ``weightcode.patch.Limits`` with the head (the last graph_conv layer, conv3) exempt
from max_rel_frob (``graphnew_limits``). An update program and every E7 reference are held to
``graphnew_program_limits``: the edit-size limits off except max_rel_frob, at most 20% of the entries changed
(BatchNorm running statistics included), and max_ops at least the tensor count.

Leak scan (``graphnew_leak_tokens``, ``graphnew_scan``): the dataset tokens of ``instances.leak_tokens``, E6's words
and the 40 arXiv CS category codes. The year tokens and "valid" match as whole words only, so the opaque hex split
ids and the shared patch-check wording ("invalid_patches", "validation") do not trip the scan.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import secrets
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import Params, load_params, params_sha256
from weightcode.common.paths import private_dir
from weightcode.data import arxiv as A
from weightcode.data.arxiv_classes import CLASSES_FILE, class_draw_block, file_sha256
from weightcode.data.registry import SPLIT_SEED
from weightcode.data.store import dataset_dir
from weightcode.evalkit.evidence import _seed_from_text
from weightcode.evalkit.fewshot import (
    CHECK_PART_KEYS,
    EVIDENCE_KEYS,
    FEWSHOT_MAX_CHANGED_FRAC,
    N_PER_NEW_CLASS,
    N_REPLAY,
    _exclusive_lock,
    class_split_readout,
    draw_new_rows,
    fewshot_check_parts,
)
from weightcode.evalkit.fewshot import EDITOR_VISIBLE_FILES as FEWSHOT_VISIBLE_FILES
from weightcode.evalkit.instances import (
    _FINAL_SELECTION,
    QUERY_BUDGET,
    SHORT_TOKEN_MAX_LEN,
    TRAINING_FREE_RULE,
    leak_tokens,
    limit_definitions,
    npz_text,
    patch_limits,
)
from weightcode.evalkit.patch_check import ForwardFn, JointPatchCheckOracle
from weightcode.evalkit.targets import (
    class_names,
    json_text,
    jsonable,
    load_dataset_forward,
    no_bytecode_cache,
    read_dataset_meta,
    read_json,
    split_path,
    utc_now,
    write_json_atomic,
)
from weightcode.models.export import fewshot_models_dir, read_fewshot_index
from weightcode.patch import OP_NAMES, Limits
from weightcode.patch import SCHEMA as PATCH_SCHEMA
from weightcode.patch.limits import SIZE_LIMIT_FIELDS

DATASET = A.INDUCTIVE_DATASET
GRAPHNEW_RULE = "graphnew-v1"
REPLAY_RULE = "graphnew-replay-v1"
REPLAY_STREAM = 706
"""Generator stream of the graphnew replay; the class draw uses 705 (``weightcode.data.arxiv_classes.CLASS_STREAM``)."""
STUDENT_ROLE = "newclass"
TASK_SCHEMA = "weightcode.task.graphnew.v1"
PRIVATE_SCHEMA = "weightcode.instance_private.graphnew.v1"
INDEX_SCHEMA = "weightcode.graphnew_instances_index.v1"
CHANGES_LOG_NAME = "changes.jsonl"
ROOT_NAME = "graphnew_instances"
INSTANCE_ID_RE = re.compile(r"^g[0-9a-f]{8}$")
GRAPH_FILE = "graph.npz"
EDITOR_VISIBLE_FILES: tuple[str, ...] = (*FEWSHOT_VISIBLE_FILES, GRAPH_FILE)
SPLITS: tuple[str, ...] = ("old_replay", "old_select", "V_evidence", "V_check", "V_select")
HEAD_OP = "graph_conv"
GRAPHNEW_MAX_CHANGED_FRAC = FEWSHOT_MAX_CHANGED_FRAC
MAX_HOPS = 3
HOP_BINS: tuple[str, ...] = ("1", "2", "3", "gt3")
ARXIV_CATEGORY_CODES: tuple[str, ...] = (
    "cs.ai", "cs.ar", "cs.cc", "cs.ce", "cs.cg", "cs.cl", "cs.cr", "cs.cv", "cs.cy", "cs.db", "cs.dc", "cs.dl",
    "cs.dm", "cs.ds", "cs.et", "cs.fl", "cs.gl", "cs.gr", "cs.gt", "cs.hc", "cs.ir", "cs.it", "cs.lg", "cs.lo",
    "cs.ma", "cs.mm", "cs.ms", "cs.na", "cs.ne", "cs.ni", "cs.oh", "cs.os", "cs.pf", "cs.pl", "cs.ro", "cs.sc",
    "cs.sd", "cs.se", "cs.si", "cs.sy",
)
"""The 40 ogbn-arxiv classes as arXiv subject codes, lower case (a test checks them against the raw zip)."""
GRAPHNEW_LEAK_WORDS: tuple[str, ...] = ("arxiv", "ogb", "valid", "cs.", "temporal", "year", "2017", "2018", "2019", "2020")
"""PREREG E6 and E7: words that would tell an editor the dataset, the time split or the class names."""
WHOLE_WORD_TOKENS: frozenset[str] = frozenset({"valid", "2017", "2018", "2019", "2020"})
"""Leak tokens matched as whole words only: a substring match would hit the hex split ids and "invalid"."""
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class GraphnewError(RuntimeError):
    """A graphnew instance, its store or its inputs were refused; nothing is overwritten."""


class StaleGraphnewInstance(GraphnewError):
    """An instance that is not active or whose files the current code would not write; it is not run."""


# ---------------------------------------------------------------------------------------------------------------
# Rules (pure)


def graphnew_root() -> Path:
    """``$WEIGHTCODE_PRIVATE/graphnew_instances`` (created if missing)."""
    root = private_dir() / ROOT_NAME
    root.mkdir(exist_ok=True)
    return root


def graphnew_seed(params_sha256_hex: str, dataset: str) -> int:
    """Seed of the graphnew-v1 evidence draw of one student."""
    if not _SHA_RE.fullmatch(params_sha256_hex):
        raise ValueError(f"not a sha256 hex digest: {params_sha256_hex!r}")
    return _seed_from_text(f"{params_sha256_hex}|{dataset}|{GRAPHNEW_RULE}")


def head_tensors(spec: Mapping[str, Any]) -> list[str]:
    """Weight and bias names of the head: the last layer of model_spec.json, which must be a graph_conv layer."""
    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers or layers[-1].get("op") != HEAD_OP:
        raise GraphnewError(f"model_spec.json must end in a {HEAD_OP} layer (the head)")
    return [str(layers[-1]["weight"]), str(layers[-1]["bias"])]


def graphnew_limits(spec: Mapping[str, Any]) -> Limits:
    """The default limits with the head exempt from max_rel_frob (PREREG E7)."""
    return Limits(rel_frob_exempt=tuple(head_tensors(spec)))


_PROGRAM_SIZE_FIELDS = ("max_rel_frob", "max_changed_frac")


def graphnew_program_limits(task_limits_json: Mapping[str, Any], n_tensors: int | None = None) -> Limits:
    """The limits that bind an E7 update and every E7 reference run within limits: task.json's limits with every
    size limit but max_rel_frob off, at most ``GRAPHNEW_MAX_CHANGED_FRAC`` of the entries changed and the head still
    exempt from max_rel_frob. ``n_tensors``: the student's tensor count; max_ops becomes max(max_ops, n_tensors), so
    a patch derived from a whole update fits (PREREG E7: max_ops is the tensor count)."""
    if not isinstance(task_limits_json, Mapping):
        raise GraphnewError(f"task.json limits must be an object, got {type(task_limits_json).__name__}")
    if n_tensors is not None and (isinstance(n_tensors, bool) or not isinstance(n_tensors, int) or n_tensors < 1):
        raise GraphnewError(f"n_tensors must be an int >= 1, got {n_tensors!r}")
    limits = Limits(**dict(task_limits_json))
    disabled = {name: None for name in SIZE_LIMIT_FIELDS if name not in _PROGRAM_SIZE_FIELDS}
    max_ops = limits.max_ops if n_tensors is None else max(limits.max_ops, n_tensors)
    return dataclasses.replace(limits, **disabled, max_changed_frac=GRAPHNEW_MAX_CHANGED_FRAC, max_ops=max_ops)


def graphnew_leak_tokens() -> list[str]:
    """Lower-case tokens no editor-visible file may contain (match them with ``graphnew_scan``)."""
    return sorted(set(leak_tokens(DATASET)) | set(GRAPHNEW_LEAK_WORDS) | set(ARXIV_CATEGORY_CODES))


def graphnew_scan(text: str, tokens: Sequence[str] | None = None) -> list[str]:
    """Tokens found in ``text``, case-insensitive: whole words for ``WHOLE_WORD_TOKENS`` and for tokens of at most
    ``SHORT_TOKEN_MAX_LEN`` characters, substrings otherwise. ``tokens`` defaults to ``graphnew_leak_tokens()``."""
    lower = text.lower()
    found = set()
    for token in {token.lower() for token in (graphnew_leak_tokens() if tokens is None else tokens)}:
        if token in WHOLE_WORD_TOKENS or len(token) <= SHORT_TOKEN_MAX_LEN:
            if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lower):
                found.add(token)
        elif token in lower:
            found.add(token)
    return sorted(found)


def graphnew_check_parts(arrays: Mapping[str, np.ndarray], new_classes: Sequence[int]) -> dict[str, np.ndarray]:
    """V_check cut by class (``fewshot.fewshot_check_parts``): x_new, y_new, x_old, y_old, x as int64 node ids."""
    parts = fewshot_check_parts(arrays, new_classes)
    for key in ("x_new", "x_old"):
        if parts[key].ndim != 1 or not np.issubdtype(parts[key].dtype, np.integer):
            raise GraphnewError(f"check part {key} must be integer node ids [N], got {parts[key].dtype} {parts[key].shape}")
        parts[key] = parts[key].astype(np.int64)
    return parts


def draw_replay_rows(y_old_replay: np.ndarray, train_classes: Sequence[int]) -> np.ndarray:
    """Rule graphnew-replay-v1: N_REPLAY old_replay rows of the train classes, uniform, without replacement."""
    pool = np.flatnonzero(np.isin(np.asarray(y_old_replay), [int(c) for c in train_classes]))
    if pool.size < N_REPLAY:
        raise GraphnewError(f"only {pool.size} old_replay rows in the train classes; the replay needs {N_REPLAY}")
    rng = np.random.default_rng([SPLIT_SEED, REPLAY_STREAM])
    return rng.choice(pool, size=N_REPLAY, replace=False).astype(np.int64)


def hop_distances(indptr: np.ndarray, indices: np.ndarray, sources: np.ndarray, max_hops: int = MAX_HOPS) -> np.ndarray:
    """Hops from every node to the nearest of ``sources`` over the CSR graph (0 at a source), ``max_hops + 1`` for
    nodes further than ``max_hops`` or unreachable. The graph is symmetric, so out-edges are the neighbours."""
    indptr, indices = np.asarray(indptr, dtype=np.int64), np.asarray(indices, dtype=np.int64)
    n = indptr.size - 1
    unreached = max_hops + 1
    dist = np.full(n, unreached, dtype=np.int64)
    frontier = np.unique(np.asarray(sources, dtype=np.int64))
    if frontier.size and (frontier[0] < 0 or frontier[-1] >= n):
        raise GraphnewError("hop sources must be node ids of the graph")
    dist[frontier] = 0
    for hop in range(1, max_hops + 1):
        starts = indptr[frontier]
        lengths = indptr[frontier + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            break
        ends = np.cumsum(lengths)
        offsets = np.arange(total, dtype=np.int64) - np.repeat(ends - lengths, lengths) + np.repeat(starts, lengths)
        neighbours = np.unique(indices[offsets])
        frontier = neighbours[dist[neighbours] == unreached]
        dist[frontier] = hop
        if frontier.size == 0:
            break
    return dist


def hop_bin(distance: np.ndarray) -> np.ndarray:
    """``HOP_BINS`` label of each distance >= 1 (``hop_distances``)."""
    distance = np.asarray(distance)
    if np.any(distance < 1):
        raise GraphnewError("hop bins are defined for distances >= 1 (readout nodes are never evidence nodes)")
    return np.where(distance > MAX_HOPS, HOP_BINS[-1], distance.astype(str))


def build_graphnew_task(
    names: Sequence[str], train_classes: Sequence[int], new_classes: Sequence[int], limits: Limits,
    n_nodes: int, n_features: int,
) -> dict[str, Any]:
    """Editor-visible task record of rule graphnew-v1 (schema weightcode.task.graphnew.v1)."""
    trained = ", ".join(str(names[c]) for c in train_classes)
    added = ", ".join(str(names[c]) for c in new_classes)
    stated = patch_limits(limits)
    return {
        "schema": TASK_SCHEMA,
        "statement": (
            f"This graph neural network classifies the nodes of a citation graph, one node per paper, into "
            f"{len(names)} classes, with one output per class. {GRAPH_FILE} holds the graph: {n_nodes} nodes with "
            f"{n_features} features each and the normalized adjacency. The network was trained only on labelled "
            f"nodes of the classes {trained}. The outputs of the classes {added} never saw a labelled node of their "
            f"class, so the network does not predict them. The evidence file holds {N_PER_NEW_CLASS} labelled nodes "
            f"of each of these new classes and {N_REPLAY} labelled nodes of the classes it was trained on, given as "
            f"node indices into {GRAPH_FILE}, with the network's predictions. Change the network's weights directly, "
            "without training it, so that it also recognises the new classes on the other nodes of the graph, while "
            "the nodes of the trained classes that it classifies correctly stay correct."
        ),
        "classes": list(names),
        "train_classes": [int(c) for c in train_classes],
        "new_classes": [int(c) for c in new_classes],
        "graph": {
            "file": GRAPH_FILE,
            "n_nodes": int(n_nodes),
            "n_features": int(n_features),
            "description": "CSR arrays indptr [n_nodes + 1] and indices [nnz] with weight [nnz], the normalized "
            "adjacency (symmetric, one self loop per node), and features [n_nodes, n_features]; no labels. forward.py "
            "reads it from its own directory",
        },
        "evidence": {
            "new": {
                "keys": ["x_new", "y_new", "pred_new"],
                "count": N_PER_NEW_CLASS * len(new_classes),
                "per_class": N_PER_NEW_CLASS,
                "description": "labelled nodes of the new classes, in class order, drawn at random from held-out "
                f"nodes; x_new are int64 node indices into {GRAPH_FILE}; the network has never seen these classes",
            },
            "old": {
                "keys": ["x_old", "y_old", "pred_old"],
                "count": N_REPLAY,
                "description": "labelled nodes of the classes the network was trained on, a fixed random draw from "
                f"held-out nodes of those classes; x_old are int64 node indices into {GRAPH_FILE}",
            },
            "ids": {"keys": ["split_ids_new", "split_ids_old"], "description": "one opaque id per example"},
        },
        "patch_check": {
            "budget_queries": QUERY_BUDGET,
            "readout": "each query applies one patch and returns aggregate numbers on a held-out check set of nodes, "
            "each against the unedited network, for two parts, the nodes of the new classes and the nodes of the "
            "trained classes: accuracy, fixed (nodes the unedited network got wrong that the patched one gets right), "
            "broken (nodes the unedited network got right that the patched one gets wrong), the changed-parameter "
            "count, S_new and S_old (fixed - broken on each part) and S = S_new + S_old; no per-node information",
            "invalid_patches": "a patch that fails validation consumes no query",
            "final_selection": _FINAL_SELECTION,
        },
        "patch_format": {"schema": PATCH_SCHEMA, "allowed_ops": list(OP_NAMES)},
        "limits": stated,
        "limit_definitions": limit_definitions(stated),
        "training_free_rule": TRAINING_FREE_RULE,
    }


def predict_nodes(forward: ForwardFn, params: Mapping[str, np.ndarray], *node_sets: np.ndarray) -> list[np.ndarray]:
    """Argmax predictions for each node set from one forward call over all of them (forward.py runs the whole graph
    on every call), with a finiteness check."""
    sets = [np.asarray(nodes, dtype=np.int64) for nodes in node_sets]
    if any(nodes.ndim != 1 for nodes in sets):
        raise GraphnewError("node sets must be 1-D arrays of node ids")
    joined = np.concatenate(sets) if sets else np.zeros(0, dtype=np.int64)
    logits = np.asarray(forward(dict(params), joined))
    if logits.ndim != 2 or logits.shape[0] != joined.size:
        raise GraphnewError(f"forward returned logits of shape {logits.shape} for {joined.size} nodes")
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("forward produced non-finite logits")
    pred = np.argmax(logits, axis=1).astype(np.int64)
    bounds = np.cumsum([0] + [nodes.size for nodes in sets])
    return [pred[bounds[i]:bounds[i + 1]] for i in range(len(sets))]


# ---------------------------------------------------------------------------------------------------------------
# Store


def _text_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    if tuple(sorted(arrays)) != tuple(sorted(EVIDENCE_KEYS)):
        raise GraphnewError(f"evidence keys {sorted(arrays)} != {sorted(EVIDENCE_KEYS)}")
    return params_sha256(arrays)


def _save_evidence(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    expected = _evidence_content_sha256(arrays)
    np.savez(path, **{key: np.ascontiguousarray(arrays[key]) for key in EVIDENCE_KEYS})
    with np.load(path, allow_pickle=False) as archive:
        reloaded = {key: archive[key] for key in archive.files}
    if _evidence_content_sha256(reloaded) != expected:
        raise GraphnewError(f"{path}: evidence content hash changed on reload")
    return expected


def read_graphnew_instance_index() -> dict[str, Any]:
    """The graphnew instance index; an empty one before the first build."""
    path = graphnew_root() / "index.json"
    if not path.exists():
        return {"schema": INDEX_SCHEMA, "instances": [], "superseded": []}
    index = read_json(path)
    if not isinstance(index, dict) or index.get("schema") != INDEX_SCHEMA:
        raise GraphnewError(f"{path} is not a {INDEX_SCHEMA} index")
    return index


def active_graphnew_ids() -> list[str]:
    """Ids of the active graphnew instances, in index order."""
    return [str(row["instance_id"]) for row in read_graphnew_instance_index()["instances"]]


def graphnew_instance_dir(instance_id: str) -> Path:
    """Directory of one graphnew instance; refuses an id of another format."""
    if not INSTANCE_ID_RE.fullmatch(str(instance_id)):
        raise GraphnewError(f"{instance_id!r} is not a graphnew instance id (g + 8 hex)")
    return graphnew_root() / str(instance_id)


def load_graphnew_instance(instance: Instance | str) -> Instance:
    """An Instance, or the graphnew instance with this id loaded and hash-checked (``load_instance``)."""
    return instance if isinstance(instance, Instance) else load_instance(graphnew_instance_dir(instance))


def _index_row(private: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("instance_id", "dataset", "arch", "seed", "model_id", "split_role", "train_classes", "new_classes",
            "graph_sha256", "evidence_rule", "created_at")
    return {key: private[key] for key in keys}


def _new_instance_id(root: Path) -> str:
    while True:
        instance_id = "g" + secrets.token_hex(4)
        if not (root / instance_id).exists() and not (root / f".tmp-{instance_id}").exists():
            return instance_id


def _check_node_split(arrays: Mapping[str, np.ndarray], origin: str, n_nodes: int) -> None:
    for key in ("x", "y", "idx"):
        if key not in arrays or arrays[key].ndim != 1 or arrays[key].dtype != np.int64:
            raise GraphnewError(f"{origin}: {key} must be int64 [N]")
    x = arrays["x"]
    if not (len(x) == len(arrays["y"]) == len(arrays["idx"])) or (x.size and (x.min() < 0 or x.max() >= n_nodes)):
        raise GraphnewError(f"{origin}: arrays do not align or node ids leave 0..{n_nodes - 1}")


def _load_node_split(path: Path, expected: str, n_nodes: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    _check_node_split(arrays, str(path), n_nodes)
    if params_sha256(arrays) != expected:
        raise GraphnewError(f"{path}: content hash differs from the recorded {expected}")
    return arrays


_RECORDED_SPLITS = {"V_check": "vcheck", "V_select": "vselect", "old_select": "oldselect"}


def _load_recorded(private: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    key = _RECORDED_SPLITS[split]
    return _load_node_split(Path(str(private[f"{key}_path"])), str(private[f"{key}_content_sha256"]),
                            int(private["n_nodes"]))


# ---------------------------------------------------------------------------------------------------------------
# Build


def _scan_visible(tokens: Sequence[str], **texts: str) -> None:
    for where, text in texts.items():
        found = graphnew_scan(text, tokens)
        if found:
            raise GraphnewError(f"{where} contains {found}; no editor-visible file may")


def _plan(
    row: Mapping[str, Any],
    names: Sequence[str],
    draw: Mapping[str, Any],
    data: Mapping[str, Mapping[str, np.ndarray]],
    split_record: Mapping[str, Mapping[str, Any]],
    replay_rows: np.ndarray,
    forward: ForwardFn,
    forward_sha: str,
    graph_record: Mapping[str, Any],
    tokens: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Evidence arrays, task and private record (without id and time) of one student's instance."""
    train_classes = [int(c) for c in draw["train_classes"]]
    new_classes = [int(c) for c in draw["new_classes"]]
    source = fewshot_models_dir(DATASET) / str(row["model_id"])
    params = load_params(source / "model.npz")
    digest = params_sha256(params)
    if digest != row["params_sha256"]:
        raise GraphnewError(f"{source}/model.npz hash {digest} != fewshot index {row['params_sha256']}")
    model_private = read_json(source / "private.json")
    if model_private.get("train_classes") != train_classes or model_private.get("role") != STUDENT_ROLE \
            or model_private.get("class_draw", {}).get("sha256") != draw["sha256"]:
        raise GraphnewError(f"{source}/private.json does not describe a {STUDENT_ROLE} student of this class draw")
    if model_private.get("train_data", {}).get("deployment_graph_sha256") != graph_record["sha256"]:
        raise GraphnewError(f"{source}/private.json was not deployed on this graph.npz")
    spec_text = (source / "model_spec.json").read_text(encoding="utf-8")
    spec = json.loads(spec_text)
    if spec.get("classes") != list(names):
        raise GraphnewError(f"{source}/model_spec.json class list differs from dataset.json")
    limits = graphnew_limits(spec)
    missing = sorted(set(limits.rel_frob_exempt) - set(params))
    if missing:
        raise GraphnewError(f"head tensors {missing} are not in {source}/model.npz")

    v_ev, replay = data["V_evidence"], data["old_replay"]
    seed = graphnew_seed(digest, DATASET)
    new_rows, ids_new, ids_old = draw_new_rows(v_ev["y"], new_classes, seed, len(replay_rows))
    x_new, y_new = v_ev["x"][new_rows].astype(np.int64), v_ev["y"][new_rows].astype(np.int64)
    x_old, y_old = replay["x"][replay_rows].astype(np.int64), replay["y"][replay_rows].astype(np.int64)
    parts = graphnew_check_parts(data["V_check"], new_classes)
    v_sel, o_sel = data["V_select"], data["old_select"]
    pred_new, pred_old, pred_new_ch, pred_old_ch, pred_sel = predict_nodes(
        forward, params, x_new, x_old, parts["x_new"], parts["x_old"], v_sel["x"])
    arrays = {
        "x_new": x_new,
        "y_new": y_new,
        "pred_new": pred_new,
        "split_ids_new": np.array(ids_new),
        "x_old": x_old,
        "y_old": y_old,
        "pred_old": pred_old,
        "split_ids_old": np.array(ids_old),
    }
    task = build_graphnew_task(names, train_classes, new_classes, limits, graph_record["n_nodes"],
                               graph_record["n_features"])
    task_text = json_text(task)
    _scan_visible(
        sorted(set(tokens) | {str(row["model_id"]).lower()}),
        **{"task.json": task_text, "evidence.npz": npz_text(arrays)},
    )
    _scan_visible(tokens, **{"model_spec.json": spec_text, "model.npz names": npz_text(params)})
    in_new_sel = np.isin(v_sel["y"], new_classes)
    in_new_osel = np.isin(o_sel["y"], new_classes)
    private = {
        "schema": PRIVATE_SCHEMA,
        "evidence_rule": GRAPHNEW_RULE,
        "dataset": DATASET,
        "arch": row["arch"],
        "seed": int(row["seed"]),
        "model_id": row["model_id"],
        "role": "dev",
        "split_role": "dev",
        "train_classes": train_classes,
        "new_classes": new_classes,
        "class_draw": dict(draw),
        "model_sha256": digest,
        "model_source_path": str(source / "model.npz"),
        "model_spec_sha256": file_sha256(source / "model_spec.json"),
        "forward_sha256": forward_sha,
        "graph_path": graph_record["path"],
        "graph_sha256": graph_record["sha256"],
        "graph_file_sha256": graph_record["file_sha256"],
        "n_nodes": int(graph_record["n_nodes"]),
        "head_tensors": list(limits.rel_frob_exempt),
        "vevidence_path": split_record["V_evidence"]["path"],
        "vevidence_content_sha256": split_record["V_evidence"]["content_sha256"],
        "n_vevidence_new_pool": int(np.count_nonzero(np.isin(v_ev["y"], new_classes))),
        "replay_rule": REPLAY_RULE,
        "replay_stream": [SPLIT_SEED, REPLAY_STREAM],
        "replay_path": split_record["old_replay"]["path"],
        "replay_content_sha256": split_record["old_replay"]["content_sha256"],
        "replay_rows": [int(r) for r in replay_rows],
        "replay_source_idx": [int(i) for i in replay["idx"][replay_rows]],
        "evidence_seed": seed,
        "evidence_row_ids_new": [int(r) for r in new_rows],
        "evidence_source_idx_new": [int(i) for i in v_ev["idx"][new_rows]],
        "evidence_split_ids_new": ids_new,
        "evidence_split_ids_old": ids_old,
        "evidence_content_sha256": _evidence_content_sha256(arrays),
        "n_base_correct_evidence_new": int(np.count_nonzero(pred_new == y_new)),
        "n_base_errors_evidence_old": int(np.count_nonzero(pred_old != y_old)),
        "vcheck_path": split_record["V_check"]["path"],
        "vcheck_content_sha256": split_record["V_check"]["content_sha256"],
        "n_vcheck": int(len(data["V_check"]["y"])),
        "n_vcheck_new": int(len(parts["y_new"])),
        "n_vcheck_old": int(len(parts["y_old"])),
        "n_base_errors_vcheck_new": int(np.count_nonzero(pred_new_ch != parts["y_new"])),
        "n_base_errors_vcheck_old": int(np.count_nonzero(pred_old_ch != parts["y_old"])),
        "n_base_pred_in_new_vcheck": int(np.count_nonzero(np.isin(np.concatenate([pred_new_ch, pred_old_ch]), new_classes))),
        "vselect_path": split_record["V_select"]["path"],
        "vselect_content_sha256": split_record["V_select"]["content_sha256"],
        "n_vselect": int(len(v_sel["y"])),
        "n_vselect_new": int(np.count_nonzero(in_new_sel)),
        "n_vselect_old": int(np.count_nonzero(~in_new_sel)),
        "n_base_pred_in_new_vselect": int(np.count_nonzero(np.isin(pred_sel, new_classes))),
        "oldselect_path": split_record["old_select"]["path"],
        "oldselect_content_sha256": split_record["old_select"]["content_sha256"],
        "n_oldselect_new": int(np.count_nonzero(in_new_osel)),
        "n_oldselect_old": int(np.count_nonzero(~in_new_osel)),
        "n_per_new_class": N_PER_NEW_CLASS,
        "n_replay": int(len(replay_rows)),
        "task_sha256": _text_sha256(task_text),
    }
    return arrays, task, private


def _copy_checked(source: Path, target: Path, expected: str, what: str) -> None:
    shutil.copyfile(source, target)
    if file_sha256(target) != expected:
        raise GraphnewError(f"{target}: {what} copy hash mismatch")


def _create(root: Path, arrays: Mapping[str, np.ndarray], task: Mapping[str, Any], core: Mapping[str, Any],
            forward_path: Path) -> dict[str, Any]:
    instance_id = _new_instance_id(root)
    staging = root / f".tmp-{instance_id}"
    staging.mkdir()
    source = Path(core["model_source_path"])
    shutil.copyfile(source, staging / "model.npz")
    if params_sha256(load_params(staging / "model.npz")) != core["model_sha256"]:
        raise GraphnewError(f"{staging}/model.npz: copy hash mismatch")
    _copy_checked(source.with_name("model_spec.json"), staging / "model_spec.json", core["model_spec_sha256"],
                  "model_spec.json")
    _copy_checked(forward_path, staging / "forward.py", core["forward_sha256"], "forward.py")
    _copy_checked(Path(core["graph_path"]), staging / GRAPH_FILE, core["graph_file_sha256"], GRAPH_FILE)
    if _save_evidence(staging / "evidence.npz", arrays) != core["evidence_content_sha256"]:
        raise GraphnewError(f"{staging}/evidence.npz: content hash mismatch")
    (staging / "task.json").write_text(json_text(task), encoding="utf-8")
    private = dict(core, instance_id=instance_id, created_at=utc_now(), numpy_version=np.__version__)
    (staging / "private.json").write_text(json_text(private), encoding="utf-8")
    with no_bytecode_cache():
        load_instance(staging)
    os.rename(staging, root / instance_id)
    return jsonable(private)


def _same_build(instance_dir: Path, task: Mapping[str, Any], core: Mapping[str, Any]) -> None:
    private = read_json(instance_dir / "private.json")
    stable = ("evidence_content_sha256", "task_sha256", "model_sha256", "forward_sha256", "model_spec_sha256",
              "graph_file_sha256", "vcheck_content_sha256", "vselect_content_sha256", "oldselect_content_sha256",
              "replay_rows", "class_draw")
    changed = [key for key in stable if private.get(key) != json.loads(json.dumps(jsonable(core.get(key))))]
    if changed or json.loads((instance_dir / "task.json").read_text(encoding="utf-8")) != json.loads(json_text(task)):
        raise GraphnewError(f"{instance_dir}: differs from a rebuild in {changed or ['task.json']}; nothing overwritten")


def _graph_record(meta: Mapping[str, Any]) -> dict[str, Any]:
    path = dataset_dir(DATASET) / GRAPH_FILE
    graph = A.load_graph(DATASET)
    block = meta["graph"]
    if block.get("path") != GRAPH_FILE or A.graph_sha256(graph) != block["sha256"]:
        raise GraphnewError(f"{path} does not match its dataset.json hash")
    return {"path": str(path), "sha256": block["sha256"], "file_sha256": file_sha256(path),
            "n_nodes": int(graph["features"].shape[0]), "n_features": int(graph["features"].shape[1])}


def build_graphnew_instances(*, seeds: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """Build (or reuse) one graphnew-v1 instance per newclass student of the class draw; one row per student.

    Students come from the fewshot index of ``ogbn_arxiv_ind`` (arch gcn, role newclass, the draw's train classes),
    optionally some ``seeds``. An active instance of the same model id is reused when a rebuild gives the same
    evidence, task and inputs, and refused otherwise.
    """
    draw = class_draw_block(DATASET)
    students = [
        row for row in read_fewshot_index(DATASET)
        if row["arch"] == "gcn" and row["role"] == STUDENT_ROLE and row["train_classes"] == draw["train_classes"]
        and (seeds is None or int(row["seed"]) in {int(s) for s in seeds})
    ]
    if not students:
        raise GraphnewError(f"no {STUDENT_ROLE} students of {DATASET} for the class draw, seeds {seeds}")
    meta = read_dataset_meta(DATASET)
    names = class_names(meta)
    forward, forward_sha, forward_path = load_dataset_forward(DATASET, meta)
    tokens = graphnew_leak_tokens()
    _scan_visible(tokens, **{"forward.py": forward_path.read_text(encoding="utf-8")})
    graph_record = _graph_record(meta)
    data, split_record = {}, {}
    for split in SPLITS:
        path = split_path(DATASET, split)
        expected = meta["splits"][split]["sha256"]
        data[split] = _load_node_split(path, expected, graph_record["n_nodes"])
        split_record[split] = {"path": str(path), "content_sha256": expected}
    replay_rows = draw_replay_rows(data["old_replay"]["y"], draw["train_classes"])
    root = graphnew_root()
    outcomes = []
    with _exclusive_lock(root):
        index = read_graphnew_instance_index()
        for row in sorted(students, key=lambda r: int(r["seed"])):
            arrays, task, core = _plan(row, names, draw, data, split_record, replay_rows, forward, forward_sha,
                                       graph_record, tokens)
            existing = [r for r in index["instances"] if r["model_id"] == row["model_id"]]
            if existing:
                (match,) = existing
                _same_build(root / match["instance_id"], task, core)
                outcomes.append({**match, "status": "reused"})
                continue
            private = _create(root, arrays, task, core, forward_path)
            index["instances"].append(_index_row(private))
            write_json_atomic(root / "index.json", index)
            with open(root / CHANGES_LOG_NAME, "a", encoding="utf-8") as log:
                log.write(json.dumps({"change": "created", **_index_row(private)}, sort_keys=True) + "\n")
            outcomes.append({**_index_row(private), "status": "created"})
    return outcomes


# ---------------------------------------------------------------------------------------------------------------
# Running an instance: stale check, oracle, readout


def check_graphnew_current(instance_dir: str | Path) -> dict[str, Any]:
    """Refuse an instance that is not active or whose files the current code would not write; else return its task.

    Checks the directory name, the index, the rule, the task.json bytes against private.json, the task against a
    fresh ``build_graphnew_task``, and the hashes of model.npz, model_spec.json, forward.py, graph.npz and
    evidence.npz. The runner calls it before a session or a reference.
    """
    root = Path(instance_dir)
    private = read_json(root / "private.json")
    instance_id = str(private.get("instance_id"))
    if root.name != instance_id:
        raise StaleGraphnewInstance(f"{root}: directory name differs from private.json instance_id {instance_id!r}")
    index = read_graphnew_instance_index()
    if instance_id not in {row["instance_id"] for row in index["instances"]}:
        retired = {row["instance_id"] for row in index["superseded"]}
        status = "superseded" if instance_id in retired else "not in the index"
        raise StaleGraphnewInstance(f"graphnew instance {instance_id} is {status}; only active instances are run")
    if private.get("evidence_rule") != GRAPHNEW_RULE:
        raise StaleGraphnewInstance(f"{instance_id} follows {private.get('evidence_rule')!r}, not {GRAPHNEW_RULE}")
    task_text = (root / "task.json").read_text(encoding="utf-8")
    if _text_sha256(task_text) != private.get("task_sha256"):
        raise StaleGraphnewInstance(f"{root / 'task.json'}: bytes do not match private.json task_sha256")
    task = json.loads(task_text)
    spec = read_json(root / "model_spec.json")
    graph = task.get("graph", {})
    expected = json.loads(json_text(build_graphnew_task(
        task.get("classes", []), private["train_classes"], private["new_classes"], graphnew_limits(spec),
        graph.get("n_nodes", -1), graph.get("n_features", -1))))
    if task != expected or graph.get("n_nodes") != private.get("n_nodes"):
        changed = sorted(key for key in set(task) | set(expected) if task.get(key) != expected.get(key))
        raise StaleGraphnewInstance(f"{instance_id}: task.json differs from what the current code writes in {changed}")
    if file_sha256(root / "model_spec.json") != private.get("model_spec_sha256"):
        raise StaleGraphnewInstance(f"{root / 'model_spec.json'}: hash differs from private.json")
    if file_sha256(root / GRAPH_FILE) != private.get("graph_file_sha256"):
        raise StaleGraphnewInstance(f"{root / GRAPH_FILE}: hash differs from private.json")
    instance = load_instance(root)  # model.npz and forward.py hashes against private.json
    if _evidence_content_sha256(instance.evidence) != private.get("evidence_content_sha256"):
        raise StaleGraphnewInstance(f"{root / 'evidence.npz'}: content hash differs from private.json")
    return task


def make_graphnew_oracle(instance: Instance | str, budget: int = QUERY_BUDGET) -> JointPatchCheckOracle:
    """Joint counted oracle of a graphnew instance: its V_check cut by class, each part in one forward call."""
    resolved = load_graphnew_instance(instance)
    private = resolved.private
    parts = graphnew_check_parts(_load_recorded(private, "V_check"), private["new_classes"])
    if (len(parts["y_new"]), len(parts["y_old"])) != (private["n_vcheck_new"], private["n_vcheck_old"]):
        raise GraphnewError(f"{resolved.instance_id}: V_check parts differ in size from the instance record")
    batch_size = max(len(parts["y_new"]), len(parts["y_old"]))
    return JointPatchCheckOracle(parts["x_new"], parts["y_new"], parts["x_old"], parts["y_old"], resolved.forward,
                                 resolved.params, budget, batch_size)


def _accuracy(correct: np.ndarray) -> float:
    return float(np.mean(correct)) if correct.size else float("nan")


def _cell(base: np.ndarray, pred: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    acc_base, acc_child = _accuracy(base == y), _accuracy(pred == y)
    return {"n": int(y.size), "acc_base": acc_base, "acc_child": acc_child, "delta": acc_child - acc_base,
            "fixed": int(np.count_nonzero((base != y) & (pred == y))),
            "broken": int(np.count_nonzero((base == y) & (pred != y)))}


def instance_graph(instance: Instance) -> dict[str, np.ndarray]:
    """indptr and indices of the instance's graph.npz (hash-checked against private.json)."""
    path = Path(instance.instance_dir) / GRAPH_FILE
    if file_sha256(path) != instance.private.get("graph_file_sha256"):
        raise GraphnewError(f"{path}: hash differs from private.json")
    with np.load(path, allow_pickle=False) as archive:
        return {"indptr": archive["indptr"], "indices": archive["indices"]}


def graphnew_readout(instance: Instance | str, child: Params) -> dict[str, Any]:
    """40-way readout of ``child`` against the instance's base (PREREG E7 readouts).

    Every key of ``fewshot.class_split_readout`` on V_select (``all_points`` is what the E7 gate reads), plus:

    - acc_new_balanced_{base,child} and delta_new_balanced: the mean over the new classes of their accuracy;
    - hop_new[bin]: the V_select rows of the new classes by hop distance (``HOP_BINS``) to the nearest new-class
      evidence node; hop_new_same_class[bin]: to the nearest evidence node of the row's own class;
    - old_select: the nodes up to 2017 cut by class, "old" (retention on that distribution) and "new" (recognition of
      the new classes on earlier papers), reported, not gated.

    Each cell holds n, acc_base, acc_child, delta, fixed and broken. Base and child each take one forward call.
    """
    resolved = load_graphnew_instance(instance)
    private = resolved.private
    v_sel, o_sel = _load_recorded(private, "V_select"), _load_recorded(private, "old_select")
    base_sel, base_osel = predict_nodes(resolved.forward, resolved.params, v_sel["x"], o_sel["x"])
    pred_sel, pred_osel = predict_nodes(resolved.forward, child, v_sel["x"], o_sel["x"])
    train_classes = [int(c) for c in private["train_classes"]]
    new_classes = [int(c) for c in private["new_classes"]]
    y = v_sel["y"].astype(np.int64)
    result = class_split_readout(base_sel, pred_sel, y, train_classes, new_classes)
    balanced = {side: float(np.mean([result["per_class"][str(c)][f"acc_{side}"] for c in new_classes]))
                for side in ("base", "child")}
    result["acc_new_balanced_base"], result["acc_new_balanced_child"] = balanced["base"], balanced["child"]
    result["delta_new_balanced"] = balanced["child"] - balanced["base"]

    graph = instance_graph(resolved)
    evidence = resolved.evidence
    in_new = np.isin(y, new_classes)
    nearest_any = hop_distances(graph["indptr"], graph["indices"], evidence["x_new"])[v_sel["x"]]
    nearest_own = np.full(y.size, MAX_HOPS + 1, dtype=np.int64)
    for label in new_classes:
        rows = y == label
        sources = evidence["x_new"][evidence["y_new"] == label]
        nearest_own[rows] = hop_distances(graph["indptr"], graph["indices"], sources)[v_sel["x"][rows]]
    for key, distance in (("hop_new", nearest_any), ("hop_new_same_class", nearest_own)):
        bins = hop_bin(distance[in_new])
        result[key] = {label: _cell(base_sel[in_new][bins == label], pred_sel[in_new][bins == label],
                                    y[in_new][bins == label]) for label in HOP_BINS}

    y_o = o_sel["y"].astype(np.int64)
    new_o = np.isin(y_o, new_classes)
    result["old_select"] = {"old": _cell(base_osel[~new_o], pred_osel[~new_o], y_o[~new_o]),
                            "new": _cell(base_osel[new_o], pred_osel[new_o], y_o[new_o])}
    return result


__all__ = [
    "ARXIV_CATEGORY_CODES", "CHECK_PART_KEYS", "DATASET", "EDITOR_VISIBLE_FILES", "EVIDENCE_KEYS",
    "GRAPHNEW_LEAK_WORDS", "GRAPHNEW_MAX_CHANGED_FRAC", "GRAPHNEW_RULE", "GRAPH_FILE", "HOP_BINS", "INSTANCE_ID_RE",
    "MAX_HOPS", "N_PER_NEW_CLASS", "N_REPLAY", "PRIVATE_SCHEMA", "REPLAY_RULE", "REPLAY_STREAM", "ROOT_NAME",
    "STUDENT_ROLE", "TASK_SCHEMA", "WHOLE_WORD_TOKENS", "GraphnewError", "StaleGraphnewInstance",
    "active_graphnew_ids", "build_graphnew_instances", "build_graphnew_task", "check_graphnew_current",
    "draw_replay_rows", "graphnew_check_parts", "graphnew_instance_dir", "graphnew_leak_tokens", "graphnew_limits",
    "graphnew_program_limits", "graphnew_readout", "graphnew_root", "graphnew_scan", "graphnew_seed", "head_tensors",
    "hop_bin", "hop_distances", "instance_graph", "load_graphnew_instance", "make_graphnew_oracle", "predict_nodes",
    "read_graphnew_instance_index",
]
