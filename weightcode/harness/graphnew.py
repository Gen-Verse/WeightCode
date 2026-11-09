"""The harness side of rule graphnew-v1 (PREREG E7): the one access point of the editor harness to
``weightcode.evalkit.graphnew``, which owns the rule (the instances, their evidence, the class cut of V_check, the limits,
the V_select readout and the leak scan).

A graphnew instance asks an update to add classes a graph network never trained on. It runs in the LEARN arm only and
follows the fewshot-v1 path (``weightcode.harness.fewshot``) step by step, with the differences PREREG E7 states:

- the instance id is ``g`` + 8 hex and the instance lives under ``$WEIGHTCODE_PRIVATE/graphnew_instances``; the runner
  refuses it unless ``check_graphnew_current`` accepts it (active in its index, task.json and files as the current
  code writes them);
- the inputs ``x_*`` are int64 node ids into the instance's ``graph.npz`` (the deployment snapshot, no labels), which
  model_spec.json lists as its one input file. The trial and the session's private directory get byte copies of it
  beside forward.py, each hashed against private.json ``graph_file_sha256``; forward.py loads it once and caches it;
- every forward runs the whole graph, so the harness predicts all the node sets of one step in one call
  (``weightcode.evalkit.graphnew.predict_nodes``): the experience check is one forward, and each check set of a patch check
  is one forward (``check_oracle``: the oracle's batch is the part size, as in ``make_graphnew_oracle``), never
  batches of ``PREDICT_BATCH``;
- the experience is the instance's evidence.npz itself (the fewshot LEARN keys: 10 labelled nodes of each new class
  and the 64 replay nodes of the train classes), with every ``pred_*`` re-checked;
- the limits are ``graphnew_program_limits`` (the head, conv3, exempt from max_rel_frob; at most 20% of the entries
  changed, BatchNorm running statistics counted; max_ops at least the tensor count), which must equal the LEARN arm's
  ``applied_limits`` of the same task (``program_limits``);
- private.json records the content hash of V_check (``vcheck_content_sha256``), not its file bytes.
  ``vcheck_file_sha256`` checks the content and returns the bytes' hash, which session.json records as
  ``vcheck_sha256``; the ledger then checks the bytes as in every session and the content again (``check_parts``);
- the check is V_check cut by class (``graphnew_check_parts``): S_new over the rows of the new classes, S_old over
  the rest. The session ledger cuts the session's V_check itself, from the ``graphnew`` section of session.json
  (``session_record``), so the MCP server needs no store variable;
- the readout is ``graphnew_readout`` on V_select (40-way, child against base): fewshot's keys plus class-balanced
  new-class accuracy, the new-class rows by hop distance to the evidence and old_select cut by class;
- the leak scan adds ``graphnew_leak_tokens`` to the text scan and matches every token with ``graphnew_scan`` (whole
  words for "valid" and the years, which occur in "invalid" and in hex ids); the sandbox self-test keeps the base
  strings, since mount ids and sizes in /proc/self/mountinfo are numbers that can spell a year;
- the kept rule is joint-retain-v1 with a floor given as a row count (``weightcode.harness.retention.kept_rule_record_rows``):
  PREREG E7 computes it from R and the references' d, outside the harness, and a graphnew session needs it.

There is no decoy, no corruption condition and no clean companion. ``weightcode.evalkit.graphnew`` imports torch (through
``weightcode.evalkit.fewshot``), so it is imported inside the functions that need it: an evidence-v1 session never loads it.
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import params_sha256
from weightcode.common.paths import private_dir

GRAPHNEW_RULE = "graphnew-v1"  # weightcode.evalkit.graphnew.GRAPHNEW_RULE; test_harness_graphnew checks that the two agree
INSTANCE_ID_RE = re.compile(r"^g[0-9a-f]{8}$")  # weightcode.evalkit.graphnew.INSTANCE_ID_RE, checked the same way
GRAPHNEW_ROOT_NAME = "graphnew_instances"  # weightcode.evalkit.graphnew.ROOT_NAME
GRAPH_FILE = "graph.npz"  # weightcode.evalkit.graphnew.GRAPH_FILE
EXPERIENCE_SOURCE = "instance_graphnew_v1"
EXPERIENCE_KEYS = ("x_new", "y_new", "pred_new", "split_ids_new", "x_old", "y_old", "pred_old", "split_ids_old")
READOUT_PARTS = ("new", "old", "all")
CHECK_SET = "old-class"  # the rows behind score_old, as the retention kept rule names them
CHECK_UNIT = "nodes"
VCHECK_KEYS = ("x", "y", "idx")
# private.json fields the harness reads from a graphnew instance; an instance without one of them is refused.
REQUIRED_PRIVATE_FIELDS = (
    "instance_id",
    "split_role",
    "dataset",
    "arch",
    "seed",
    "model_id",
    "evidence_rule",
    "train_classes",
    "new_classes",
    "head_tensors",
    "n_nodes",
    "graph_file_sha256",
    "vcheck_path",
    "vcheck_content_sha256",
    "n_vcheck",
    "n_vcheck_new",
    "n_vcheck_old",
    "n_base_errors_vcheck_new",
    "n_base_errors_vcheck_old",
    "model_sha256",
    "forward_sha256",
    "task_sha256",
    "evidence_content_sha256",
    "model_spec_sha256",
)
# private.json fields that hold host paths; every one present is a forbidden string of the session.
PRIVATE_PATH_FIELDS = ("vcheck_path", "vselect_path", "vevidence_path", "replay_path", "oldselect_path",
                       "model_source_path", "graph_path")


class GraphnewHarnessError(RuntimeError):
    """A graphnew instance or session record is inconsistent, so no LEARN view, check or readout is built from it."""


def is_graphnew_id(instance_id: object) -> bool:
    """True for an id of the graphnew store's form (``g`` and 8 hex digits)."""
    return isinstance(instance_id, str) and INSTANCE_ID_RE.fullmatch(instance_id) is not None


def is_graphnew(private: Mapping[str, Any]) -> bool:
    """True when private.json says the instance follows rule graphnew-v1."""
    return private.get("evidence_rule") == GRAPHNEW_RULE


def instance_dir(instance_id: str) -> Path:
    """``$WEIGHTCODE_PRIVATE/graphnew_instances/<instance_id>``, the path of ``weightcode.evalkit.graphnew.graphnew_instance_dir``
    computed without creating the store root, so that resolving an id writes nothing; refuses an id of another form."""
    if not is_graphnew_id(instance_id):
        raise GraphnewHarnessError(f"{instance_id!r} is not a graphnew instance id ('g' and 8 hex digits)")
    return private_dir() / GRAPHNEW_ROOT_NAME / instance_id


def check_current(path: str | Path) -> dict[str, Any]:
    """``weightcode.evalkit.graphnew.check_graphnew_current``: the instance's task, or ``StaleGraphnewInstance``."""
    from weightcode.evalkit.graphnew import check_graphnew_current

    return check_graphnew_current(path)


def stale_error() -> type[Exception]:
    """The evalkit exception for an instance that is not active or not current (and every other graphnew refusal)."""
    from weightcode.evalkit.graphnew import GraphnewError

    return GraphnewError


def active_index_row(instance_id: str) -> dict[str, Any] | None:
    """The instance's row among the active rows of the graphnew index, or None."""
    from weightcode.evalkit.graphnew import read_graphnew_instance_index

    rows = read_graphnew_instance_index()["instances"]
    found = [dict(row) for row in rows if isinstance(row, dict) and row.get("instance_id") == instance_id]
    return found[0] if found else None


def leak_words() -> list[str]:
    """``weightcode.evalkit.graphnew.graphnew_leak_tokens``: the dataset tokens, E6's words and the 40 category codes (the
    instance build scans its files for the same)."""
    from weightcode.evalkit.graphnew import graphnew_leak_tokens

    return graphnew_leak_tokens()


def scan(text: str, tokens: Sequence[str]) -> list[str]:
    """``weightcode.evalkit.graphnew.graphnew_scan``: the tokens found in ``text`` (whole words for "valid", the years and
    short tokens, substrings otherwise)."""
    from weightcode.evalkit.graphnew import graphnew_scan

    return graphnew_scan(text, list(tokens))


def program_limits(task_limits_json: Mapping[str, Any], n_tensors: int) -> Any:
    """``weightcode.evalkit.graphnew.graphnew_program_limits`` of the task's limits and the student's tensor count, which
    must equal the LEARN arm's ``applied_limits`` of the same task (the limits a session applies); a difference is a
    harness fault, so the view is refused."""
    from weightcode.evalkit.graphnew import graphnew_program_limits
    from weightcode.harness.arms import applied_limits

    limits = graphnew_program_limits(task_limits_json, n_tensors)
    if limits != applied_limits(task_limits_json, "LEARN", n_tensors):
        raise GraphnewHarnessError("graphnew_program_limits differs from the LEARN arm's applied limits")
    return limits


def head_tensors(spec: Mapping[str, Any]) -> list[str]:
    """``weightcode.evalkit.graphnew.head_tensors``: the weight and bias of the last graph_conv layer (the head)."""
    from weightcode.evalkit.graphnew import head_tensors as rule_head

    return rule_head(spec)


def _class_list(private: Mapping[str, Any], key: str) -> list[int]:
    values = private.get(key)
    if not isinstance(values, list) or not values or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise GraphnewHarnessError(f"private.json {key} must be a non-empty list of class indices, got {values!r}")
    return [int(v) for v in values]


def _node_ids(x: np.ndarray, n_nodes: int, what: str) -> None:
    if x.ndim != 1 or x.dtype != np.int64 or x.size == 0:
        raise GraphnewHarnessError(f"{what}: expected int64 node ids [N], got {x.dtype}{x.shape}")
    if x.min() < 0 or x.max() >= n_nodes:
        raise GraphnewHarnessError(f"{what}: node ids leave 0 .. {n_nodes - 1}")


def experience(instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The LEARN experience of a graphnew instance and its host-side record.

    The arrays are the instance's evidence under ``EXPERIENCE_KEYS``: ``x_*`` int64 node ids of the graph, the new
    and old parts sharing no node. ``pred_new`` and ``pred_old`` must be the student's predictions, from one forward
    over both node sets; ``y_new`` must hold every new class equally often and ``y_old`` only train classes.
    """
    from weightcode.evalkit.graphnew import predict_nodes

    private = instance.private
    if not is_graphnew(private):
        raise GraphnewHarnessError(f"{instance.instance_id} follows {private.get('evidence_rule')!r}, not "
                                   f"{GRAPHNEW_RULE}")
    evidence = instance.evidence
    if sorted(evidence) != sorted(EXPERIENCE_KEYS):
        raise GraphnewHarnessError(f"{instance.instance_id}: evidence keys {sorted(evidence)} are not the LEARN keys "
                                   f"{sorted(EXPERIENCE_KEYS)}")
    arrays = {key: np.ascontiguousarray(evidence[key]) for key in EXPERIENCE_KEYS}
    new_classes, train_classes = _class_list(private, "new_classes"), _class_list(private, "train_classes")
    if set(new_classes) & set(train_classes):
        raise GraphnewHarnessError(f"new classes {new_classes} and train classes {train_classes} overlap")
    n_nodes = int(private["n_nodes"])
    for part in ("new", "old"):
        x, y, shown = arrays[f"x_{part}"], arrays[f"y_{part}"], arrays[f"pred_{part}"]
        if not (len(x) == len(y) == len(shown) == len(arrays[f"split_ids_{part}"])):
            raise GraphnewHarnessError(f"{instance.instance_id}: the {part} part has unequal lengths")
        _node_ids(x, n_nodes, f"{instance.instance_id} x_{part}")
    if np.intersect1d(arrays["x_new"], arrays["x_old"]).size:
        raise GraphnewHarnessError(f"{instance.instance_id}: a node is in both the new and the old part")
    pred_new, pred_old = predict_nodes(instance.forward, instance.params, arrays["x_new"], arrays["x_old"])
    n_wrong = {}
    for part, pred in (("new", pred_new), ("old", pred_old)):
        shown, y = arrays[f"pred_{part}"], arrays[f"y_{part}"]
        if not np.array_equal(pred, shown):
            raise GraphnewHarnessError(f"{instance.instance_id}: pred_{part} disagrees with the student on "
                                       f"{int(np.count_nonzero(pred != shown))} of {len(y)} nodes")
        n_wrong[part] = int(np.count_nonzero(pred != y))
    labels, counts = np.unique(arrays["y_new"], return_counts=True)
    if sorted(int(label) for label in labels) != sorted(new_classes) or len(set(counts.tolist())) != 1:
        raise GraphnewHarnessError(f"{instance.instance_id}: y_new holds {dict(zip(labels.tolist(), counts.tolist()))}; "
                                   f"the new classes are {new_classes}, each equally often")
    if not np.isin(arrays["y_old"], train_classes).all():
        raise GraphnewHarnessError(f"{instance.instance_id}: y_old holds a class outside the train classes")
    ids = [*arrays["split_ids_new"].tolist(), *arrays["split_ids_old"].tolist()]
    if len(set(ids)) != len(ids):
        raise GraphnewHarnessError(f"{instance.instance_id}: the split ids are not unique")
    record = {
        "file": "experience.npz",
        "source": EXPERIENCE_SOURCE,
        "rule": GRAPHNEW_RULE,
        "content_sha256": params_sha256(arrays),
        "evidence_content_sha256": private["evidence_content_sha256"],
        "student_params_sha256": instance.params_sha256,
        "n_new": int(len(arrays["y_new"])),
        "n_new_fail": n_wrong["new"],
        "n_old": int(len(arrays["y_old"])),
        "n_old_fail": n_wrong["old"],
        "per_class": int(counts[0]),
        "new_classes": new_classes,
        "train_classes": train_classes,
        "inputs": "int64 node ids into graph.npz",
    }
    if record["content_sha256"] != private["evidence_content_sha256"]:
        raise GraphnewHarnessError(f"{instance.instance_id}: the experience differs from the recorded evidence")
    return arrays, record


def _read_vcheck(path: Path) -> tuple[bytes, dict[str, np.ndarray]]:
    data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return data, {key: archive[key] for key in archive.files}


def vcheck_file_sha256(private: Mapping[str, Any]) -> str:
    """The sha256 of the V_check file's bytes, once its content (``params_sha256`` of its arrays) equals private.json
    ``vcheck_content_sha256`` and its row count ``n_vcheck``; session.json records it as ``vcheck_sha256``."""
    import hashlib

    data, arrays = _read_vcheck(Path(str(private["vcheck_path"])))
    if params_sha256(arrays) != private["vcheck_content_sha256"]:
        raise GraphnewHarnessError(f"{private['vcheck_path']}: content hash differs from private.json "
                                   "vcheck_content_sha256")
    if sorted(arrays) != sorted(VCHECK_KEYS) or len(arrays["y"]) != int(private["n_vcheck"]):
        raise GraphnewHarnessError(f"{private['vcheck_path']}: expected arrays {list(VCHECK_KEYS)} with "
                                   f"{private['n_vcheck']} rows")
    return hashlib.sha256(data).hexdigest()


def session_record(private: Mapping[str, Any]) -> dict[str, Any]:
    """The ``graphnew`` section of session.json: what the ledger needs to cut V_check and to check the graph copy
    its forward reads, and the base error counts."""
    missing = [name for name in REQUIRED_PRIVATE_FIELDS if name not in private]
    if missing:
        raise GraphnewHarnessError(f"private.json lacks {missing}")
    return {
        "rule": GRAPHNEW_RULE,
        "train_classes": _class_list(private, "train_classes"),
        "new_classes": _class_list(private, "new_classes"),
        "n_vcheck_new": int(private["n_vcheck_new"]),
        "n_vcheck_old": int(private["n_vcheck_old"]),
        "n_base_errors_vcheck_new": int(private["n_base_errors_vcheck_new"]),
        "n_base_errors_vcheck_old": int(private["n_base_errors_vcheck_old"]),
        "vcheck_content_sha256": str(private["vcheck_content_sha256"]),
        "graph_file": GRAPH_FILE,
        "graph_file_sha256": str(private["graph_file_sha256"]),
        "n_nodes": int(private["n_nodes"]),
        "head_tensors": [str(name) for name in private["head_tensors"]],
    }


def check_parts(arrays: Mapping[str, np.ndarray], record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """V_check cut by class (``weightcode.evalkit.graphnew.graphnew_check_parts``), after its content hash is checked
    against ``record``; the part sizes must be the recorded ones and the node ids those of the graph."""
    from weightcode.evalkit.graphnew import graphnew_check_parts

    if record.get("rule") != GRAPHNEW_RULE:
        raise GraphnewHarnessError(f"the session's graphnew record names rule {record.get('rule')!r}")
    if params_sha256(dict(arrays)) != record.get("vcheck_content_sha256"):
        raise GraphnewHarnessError("V_check content differs from the session's vcheck_content_sha256")
    parts = graphnew_check_parts(arrays, [int(c) for c in record["new_classes"]])
    sizes = (len(parts["y_new"]), len(parts["y_old"]))
    if sizes != (int(record["n_vcheck_new"]), int(record["n_vcheck_old"])):
        raise GraphnewHarnessError(f"V_check cuts into {sizes} new and old rows; the instance records "
                                   f"{(record['n_vcheck_new'], record['n_vcheck_old'])}")
    if np.isin(parts["y_old"], [int(c) for c in record["train_classes"]], invert=True).any():
        raise GraphnewHarnessError("an old-part V_check row has a label outside the train classes")
    for key in ("x_new", "x_old"):
        _node_ids(parts[key], int(record["n_nodes"]), f"V_check {key}")
    return parts


def check_oracle(parts: Mapping[str, np.ndarray], forward: Any, base_params: Mapping[str, np.ndarray],
                 budget: int) -> Any:
    """evalkit's ``JointPatchCheckOracle`` on the two check parts, with a batch as large as the larger part, so that
    each part is one forward call (one full-graph forward per check set), as ``make_graphnew_oracle`` builds it."""
    from weightcode.evalkit.patch_check import JointPatchCheckOracle

    batch_size = max(len(parts["y_new"]), len(parts["y_old"]))
    return JointPatchCheckOracle(parts["x_new"], parts["y_new"], parts["x_old"], parts["y_old"], forward, base_params,
                                 budget, batch_size)


def _no_nan(value: Any) -> Any:
    """``value`` with every NaN float replaced by None (an accuracy over zero rows), so the record is strict JSON."""
    if isinstance(value, dict):
        return {key: _no_nan(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_no_nan(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def readout(instance: Instance, child: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """``graphnew_readout`` of ``child`` on V_select (``readouts_graphnew``; an accuracy over zero rows, NaN there,
    is None here) and, per part (new classes, old classes, all), the change in the fields of a dev readout: n,
    base_acc, child_acc, delta_acc, fixed and broken. Base and child take one forward each."""
    from weightcode.evalkit.graphnew import graphnew_readout

    full = _no_nan(graphnew_readout(instance, dict(child)))
    parts = {part: {"n": full[f"n_{part}"], "base_acc": full[f"acc_{part}_base"],
                    "child_acc": full[f"acc_{part}_child"], "delta_acc": full[f"delta_{part}"],
                    "fixed": full[f"fixed_{part}"], "broken": full[f"broken_{part}"]} for part in READOUT_PARTS}
    return {"readouts_graphnew": full, "parts": parts}


def student_norms(params: Mapping[str, np.ndarray], spec: Mapping[str, Any]) -> dict[str, float]:
    """The Frobenius norm of every graph_conv weight of the layout (PREREG E7 transfer: reported next to the gain)."""
    layers = spec.get("layers")
    if not isinstance(layers, list):
        raise GraphnewHarnessError("model_spec.json has no layers list")
    names = [str(layer["weight"]) for layer in layers if isinstance(layer, dict) and layer.get("op") == "graph_conv"]
    if not names:
        raise GraphnewHarnessError("model_spec.json names no graph_conv layer")
    return {name: float(np.linalg.norm(np.asarray(params[name], dtype=np.float64))) for name in names}


def private_paths(private: Mapping[str, Any]) -> list[str]:
    """The host paths recorded in a graphnew private.json (``PRIVATE_PATH_FIELDS``)."""
    return [str(private[name]) for name in PRIVATE_PATH_FIELDS if private.get(name)]


__all__ = [
    "CHECK_SET", "CHECK_UNIT", "EXPERIENCE_KEYS", "EXPERIENCE_SOURCE", "GRAPHNEW_ROOT_NAME", "GRAPHNEW_RULE",
    "GRAPH_FILE", "GraphnewHarnessError", "INSTANCE_ID_RE", "PRIVATE_PATH_FIELDS", "READOUT_PARTS",
    "REQUIRED_PRIVATE_FIELDS", "VCHECK_KEYS", "active_index_row", "check_current", "check_oracle", "check_parts",
    "experience", "head_tensors", "instance_dir", "is_graphnew", "is_graphnew_id", "leak_words", "private_paths",
    "program_limits", "readout", "scan", "session_record", "stale_error", "student_norms", "vcheck_file_sha256",
]
