"""Repair-instance directories (contract sections 3 and 5.4, HARNESS_SPEC section 1, PREREG section 2).

Layout::

    $WEIGHTCODE_PRIVATE/instances/<instance_id>/
        model.npz  model_spec.json  forward.py  evidence.npz  task.json      editor-visible originals
        private.json                                                         never shown to an editor
    $WEIGHTCODE_PRIVATE/instances/index.json      "instances": active rows; "superseded": rows no runner may select
    $WEIGHTCODE_PRIVATE/instances/changes.jsonl   append-only log of supersessions

Two instance rules share this layout:

- ``evidence-v1`` (user decision 2026-11-03, the current rule): one instance per (dataset, deployment condition,
  model). The evidence holds base errors drawn from all errors on the condition's V_evidence plus random rows
  (``weightcode.evalkit.evidence``); task.json (schema ``weightcode.task.v1``) names no target pair; private.json (schema
  ``weightcode.instance_private.v1``) records the condition, the V_check location and hashes and the base error counts.
- ``evidence-v0``: one instance per (dataset, model, target pair) from the model's targets.json, task.json schema
  ``weightcode.task.v0`` with the pair. Superseded by v1; kept for its existing instances and tests.

model.npz and model_spec.json are byte copies of the model's files and forward.py is a byte copy of the dataset
forward; every copy is hash-checked. task.json holds a neutral statement, the class names, the patch-check budget,
the patch format, the edit size limits and the training-free rule, and carries no dataset name, seed or model id,
which a token scan enforces.

Instances are built for the selected seeds (``weightcode.evalkit.targets.SeedSelection``) of main models (index role
"base", split_role "main") and dev models (role "dev", split_role "dev"); ceiling models of the same dataset are
recorded in private.json as references. Decoys (contract 5.4, PREREG section 2) follow an explicit seed map passed in
by the caller. The scripts default to main 0 <-> 1 and dev 100 <-> 101 (user rule 2026-11-03); the PREREG's
five-seed extension 2 -> 3 -> 4 -> 2 only adds entries, so instances of seeds 0 and 1 keep their decoys. A decoy is
always a selected model of the same role and architecture: no instance points at the weights of an unselected seed,
and no dev instance points at main-seed weights.

Instance ids are random ("i" + 8 hex digits); everything else is a deterministic function of the model, its decoy,
the dataset files and the rules. A rebuild finds the active instance with the same key, checks that its
deterministic content is unchanged and reuses it. Instances are never modified or deleted. Any difference, and any
active instance of an unselected model, is refused; the way out is to supersede the old instances (their rows move
to "superseded" in index.json with a logged reason, their directories stay in place) and build new ones.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import load_instance
from weightcode.common.identity import Params, load_params, params_sha256
from weightcode.common.paths import private_dir
from weightcode.data.conditions import DRAWN_CONDITIONS
from weightcode.evalkit.evidence import (
    EVIDENCE_RULE,
    EVIDENCE_RULE_V1,
    K_RAND,
    Evidence,
    draw_evidence,
    draw_evidence_v1,
    evidence_content_sha256,
    evidence_seed,
    evidence_seed_v1,
    load_evidence,
    save_evidence,
)
from weightcode.evalkit.patch_check import ForwardFn, predict
from weightcode.evalkit.targets import (
    DEPLOY_CONDITION,
    K_FAIL,
    ROLE_CEILING,
    ROLE_DEV,
    ROLE_MAIN,
    RULE_ID,
    SPLIT_ROLES,
    DeploymentSplits,
    ModelEntry,
    SeedSelection,
    class_names,
    deployment_splits,
    file_sha256,
    json_text,
    jsonable,
    load_checked_split,
    load_dataset_forward,
    load_model_params,
    load_split,
    model_dir,
    no_bytecode_cache,
    read_condition_meta,
    read_dataset_meta,
    read_json,
    read_model_index,
    read_targets,
    select_entries,
    split_path,
    targets_path,
    utc_now,
    write_json_atomic,
)
from weightcode.patch import OP_NAMES, Limits, task_limits
from weightcode.patch import SCHEMA as PATCH_SCHEMA

TASK_SCHEMA = "weightcode.task.v0"
TASK_SCHEMA_V1 = "weightcode.task.v1"
PRIVATE_SCHEMA = "weightcode.instance_private.v0"
PRIVATE_SCHEMA_V1 = "weightcode.instance_private.v1"
INDEX_SCHEMA = "weightcode.instances_index.v0"
CHANGE_SCHEMA = "weightcode.instance_change.v0"
CHANGES_LOG_NAME = "changes.jsonl"
EVIDENCE_RULES = (EVIDENCE_RULE, EVIDENCE_RULE_V1)
QUERY_BUDGET = 8
# Decoy map of the development phase (PREREG section 2, user rule 2026-11-03): seed -> decoy seed. The scripts use it
# as their CLI default; library functions take the map as an argument. Five main seeds would add 2: 3, 3: 4, 4: 2.
DEFAULT_DECOYS: dict[int, int] = {0: 1, 1: 0, 100: 101, 101: 100}
INSTANCE_ID_RE = re.compile(r"^i[0-9a-f]{8}$")
EDITOR_VISIBLE_FILES = ("model.npz", "model_spec.json", "forward.py", "evidence.npz", "task.json")
# private.json fields that legitimately differ between two builds of the same instance.
NONDETERMINISTIC_PRIVATE_KEYS = frozenset({"instance_id", "created_at", "numpy_version"})
DATASET_NAME_TOKENS = ("fashion", "mnist", "cifar", "colored", "coloured")
# Words that would tell an editor the corruption of a named condition (coordinator rule 2026-11-03). Every named
# condition adds these, its own name and the parts of its name and of its corruption; none may appear in any
# editor-visible file. The images themselves show the corruption, which is allowed.
CONDITION_LEAK_WORDS = ("contrast", "corrupt", "gaussian", "noise", "severity")
_DECOY_PAIR_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")

# Edit-size definitions of contract 5.1, one per weightcode.patch.Limits field, before and after the edit-limit switch
# (max_edit_frac; max_rel_frob restricted to tensors of rank >= 2). task.json states the values Limits() holds at
# build time with the definitions of exactly those fields, so the same code is right on either side of the switch.
LIMIT_DEFINITIONS = {
    "max_changed_frac": "changed entries over all parameter entries; an entry is changed if its float32 value "
    "differs from the parent's",
    # Complete per-op list, matching Op.edit_size and describe_ops in weightcode.patch.
    "max_edit_frac": "numbers the patch specifies, counted per op (add_dense: tensor size; index-form set/add: "
    "number of values; slice-form set/add: 1 with one value, the block size with a values file; scale_rows/cols: "
    "positions; project_out: direction length; low_rank_add: r*(rows+cols); tie: group-map length), over all "
    "parameter entries",
    "max_rel_frob": "per tensor, ||child - parent||_F / ||parent||_F",
    "rel_frob_min_ndim": "smallest tensor rank that max_rel_frob applies to; tensors of lower rank are exempt",
    "max_ops": "operations per patch",
    "max_rank": "rank r of u and v in low_rank_add",
    "max_file_bytes": "total bytes of the distinct .npy files one patch references",
    "rel_frob_exempt": "names of tensors that max_rel_frob does not apply to, whatever their rank",
}
_REL_FROB_RANK_2 = "per tensor of rank >= 2, ||child - parent||_F / ||parent||_F; 1-D tensors are exempt"


def patch_limits(limits: Limits | None = None) -> dict[str, Any]:
    """``limits`` as task.json states them (``weightcode.patch.task_limits``); an undefined field is refused.

    None means the values ``weightcode.patch.Limits()`` holds now, which every evidence-v1 task.json states.
    """
    limits = task_limits() if limits is None else task_limits(limits)
    undefined = sorted(set(limits) - set(LIMIT_DEFINITIONS))
    if undefined:
        raise RuntimeError(f"weightcode.patch.Limits has fields {undefined} that task.json has no definition for")
    return limits


def limit_definitions(limits: Mapping[str, Any]) -> dict[str, str]:
    """Definitions of exactly the fields in ``limits``; a field whose value is None says that it is not applied.

    max_rel_frob covers the tensors of rank >= rel_frob_min_ndim when Limits has that field. Without it, the
    switched limits (those with max_edit_frac) exempt 1-D tensors, as the switch specifies, and the earlier limits
    cover every tensor.
    """
    definitions = {key: LIMIT_DEFINITIONS[key] for key in limits}
    if "max_rel_frob" in limits:
        if "rel_frob_min_ndim" in limits:
            min_ndim = limits["rel_frob_min_ndim"]
            if min_ndim == 2:
                definitions["max_rel_frob"] = _REL_FROB_RANK_2
            elif min_ndim is not None:
                definitions["max_rel_frob"] = (
                    f"per tensor of rank >= {min_ndim}, ||child - parent||_F / ||parent||_F; tensors of lower rank "
                    "are exempt"
                )
        elif "max_edit_frac" in limits:
            definitions["max_rel_frob"] = _REL_FROB_RANK_2
    if "max_rel_frob" in limits and limits.get("rel_frob_exempt"):
        definitions["max_rel_frob"] += "; the tensors named in rel_frob_exempt are exempt whatever their rank"
    for key, value in limits.items():
        if value is None:
            definitions[key] += "; null: this limit is not applied"
    return definitions


# Contract 5.5, stated as instructions.
TRAINING_FREE_RULE: dict[str, Any] = {
    "forbidden": [
        "gradients of any loss with respect to parameters (autograd, hand-written backpropagation, finite-difference "
        "gradients)",
        "solvers whose targets derive from evidence labels (lstsq, solve, pinv, inv, closed-form regression or a "
        "linear probe)",
        "any .fit( call",
        "iterative parameter updates driven by a label-dependent objective (gradient descent, coordinate descent, "
        "evolutionary or random search that keeps the best)",
        "scoring more than 8 distinct full candidate patches against evidence labels in one session",
    ],
    "allowed": [
        "forward passes, activations and weight inspection",
        "label-conditioned descriptive statistics (class means and their differences, SVD/PCA, correlations)",
        "single-element ablation sweeps for attribution",
        "comparing up to 8 explicit candidate patches on evidence",
    ],
    "audit": "every executed cell is logged and audited against these rules",
}

_FINAL_SELECTION = (
    "kept at the end: among the checked patches and the unedited classifier (S = 0), the one with the highest S on "
    "the check split; ties go to fewer changed parameters, then to the earlier query; a patch with S <= 0 and an "
    "unchecked patch are never kept"
)


class InstanceConflict(RuntimeError):
    """An existing instance or index disagrees with what the rules produce now; nothing is overwritten."""


def instances_root() -> Path:
    """``$WEIGHTCODE_PRIVATE/instances`` (created if missing)."""
    root = private_dir() / "instances"
    root.mkdir(exist_ok=True)
    return root


# ---------------------------------------------------------------------------------------------------------------
# task.json


def build_task(
    classes: Sequence[str], a: int, b: int, n_class_a: int, n_class_a_predicted_b: int
) -> dict[str, Any]:
    """Editor-visible task record of rule evidence-v0 (schema weightcode.task.v0) for target pair (a, b)."""
    a_name, b_name = classes[a], classes[b]
    limits = patch_limits()
    return {
        "schema": TASK_SCHEMA,
        "statement": (
            f"Repair a trained image classifier on one systematic confusion: images of class {a} ({a_name}) are "
            f"often predicted as class {b} ({b_name}). Change the classifier's weights directly, without training "
            f"it, so that fewer images of class {a} are predicted as class {b} on new images, while its other "
            "predictions stay as they are."
        ),
        "classes": list(classes),
        "target": {"a": int(a), "b": int(b), "a_name": a_name, "b_name": b_name},
        "selection_counts": {
            "n_class_a": int(n_class_a),
            "n_class_a_predicted_b": int(n_class_a_predicted_b),
            "note": "counts of the unedited classifier on a held-out selection split, measured before any edit",
        },
        "patch_check": {
            "budget_queries": QUERY_BUDGET,
            "readout": "each query applies one patch and returns aggregate numbers on a held-out check split, each "
            "against the unedited classifier: the target confusion rate P(pred = b | y = a), accuracy, fixed_target "
            "(class-a images the unedited classifier called b that the patched one calls a), fixed and broken "
            "counts (broken: images the unedited classifier got right that the patched one gets wrong), the "
            "changed-parameter count and S = fixed_target - broken; no per-image information",
            "invalid_patches": "a patch that fails validation consumes no query",
            "final_selection": _FINAL_SELECTION,
        },
        "patch_format": {"schema": PATCH_SCHEMA, "allowed_ops": list(OP_NAMES)},
        "limits": limits,
        "limit_definitions": limit_definitions(limits),
        "training_free_rule": TRAINING_FREE_RULE,
    }


def build_task_v1(classes: Sequence[str]) -> dict[str, Any]:
    """Editor-visible task record of rule evidence-v1 (schema weightcode.task.v1); it names no target pair."""
    limits = patch_limits()
    return {
        "schema": TASK_SCHEMA_V1,
        "statement": (
            "Improve a trained image classifier on the data it is deployed on. The evidence file holds examples "
            "from the deployment data that the classifier gets wrong, plus random examples from the same data, with "
            "their labels and the classifier's predictions. Change the classifier's weights directly, without "
            "training it, so that it classifies more new deployment images correctly, while the images it already "
            "classifies correctly stay correct."
        ),
        "classes": list(classes),
        "evidence": {
            "fail": {
                "keys": ["x_fail", "y_fail", "pred_fail"],
                "count": K_FAIL,
                "description": "examples from the deployment data that the classifier gets wrong, drawn at random "
                "from all of its errors there",
            },
            "random": {
                "keys": ["x_rand", "y_rand", "pred_rand"],
                "count": K_RAND,
                "description": "random examples from the same deployment data, whether the classifier gets them "
                "right or not",
            },
            "ids": {"keys": ["split_ids"], "description": "one opaque id per example, fail examples first"},
        },
        "patch_check": {
            "budget_queries": QUERY_BUDGET,
            "readout": "each query applies one patch and returns aggregate numbers on a held-out check split of the "
            "deployment data, each against the unedited classifier: accuracy, fixed (images the unedited classifier "
            "got wrong that the patched one gets right), broken (images the unedited classifier got right that the "
            "patched one gets wrong), the changed-parameter count and S = fixed - broken; no per-image information",
            "invalid_patches": "a patch that fails validation consumes no query",
            "final_selection": _FINAL_SELECTION,
        },
        "patch_format": {"schema": PATCH_SCHEMA, "allowed_ops": list(OP_NAMES)},
        "limits": limits,
        "limit_definitions": limit_definitions(limits),
        "training_free_rule": TRAINING_FREE_RULE,
    }


def leak_tokens(dataset: str) -> list[str]:
    """Lower-case tokens that must not appear in editor-visible text: dataset names and the word seed."""
    parts = {part for part in dataset.lower().split("_") if len(part) >= 3}
    return sorted(set(DATASET_NAME_TOKENS) | parts | {dataset.lower(), "seed"})


def condition_leak_tokens(condition: str, meta: Mapping[str, Any]) -> list[str]:
    """Lower-case tokens naming a named condition's corruption: ``CONDITION_LEAK_WORDS``, the condition name, the
    recorded corruption or transform name (``meta["condition"]["corruption"]`` for the CIFAR-10-C family,
    ``["transform"]`` for the drawn family), every ``_``-separated part of these, and each part's stem without
    trailing digits when that stem has at least 3 letters (so ``contrast_s3`` adds ``contrast`` and ``s3``, and
    ``blur5_d1`` adds ``blur5``, ``blur`` and ``d1``)."""
    names = {condition.lower()}
    record = meta.get("condition")
    for field in ("corruption", "transform"):
        value = record.get(field) if isinstance(record, dict) else None
        if isinstance(value, str) and value:
            names.add(value.lower())
    parts = {part for name in names for part in name.split("_") if part}
    stems = {stem for part in parts if len(stem := part.rstrip("0123456789")) >= 3}
    return sorted(names | parts | stems | set(CONDITION_LEAK_WORDS))


SHORT_TOKEN_MAX_LEN = 3
"""Tokens of at most this many characters (name parts such as ``s3`` or ``d1``) match only as whole words: not
preceded or followed by a letter or digit. Opaque hex codes (``e`` plus 12 hex digits, the evidence split_ids) then
cannot match one by chance: under substring matching ``d1`` occurs in most 64-id evidence draws."""


def scan_for_tokens(text: str, tokens: Iterable[str]) -> list[str]:
    """Tokens found in ``text``, case-insensitive: a substring match, or a whole-word match for short tokens
    (``SHORT_TOKEN_MAX_LEN``)."""
    lower = text.lower()
    found = set()
    for token in {token.lower() for token in tokens}:
        if len(token) <= SHORT_TOKEN_MAX_LEN:
            if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lower):
                found.add(token)
        elif token in lower:
            found.add(token)
    return sorted(found)


def npz_text(arrays: Mapping[str, np.ndarray]) -> str:
    """The text an editor can read in an .npz file: its member names and the values of its string arrays."""
    words = list(arrays)
    for value in arrays.values():
        if value.dtype.kind in ("U", "S"):
            words.extend(str(item) for item in value.ravel().tolist())
    return "\n".join(words)


# ---------------------------------------------------------------------------------------------------------------
# Decoys


def parse_decoys(text: str) -> dict[int, int]:
    """Parse a decoy map such as ``"0:1,1:0,100:101,101:100"`` (comma-separated seed:decoy_seed pairs)."""
    decoys: dict[int, int] = {}
    for part in text.split(","):
        match = _DECOY_PAIR_RE.fullmatch(part)
        if not match:
            raise ValueError(f"decoy map {text!r}: {part!r} is not seed:decoy_seed")
        seed, decoy = int(match.group(1)), int(match.group(2))
        if seed in decoys:
            raise ValueError(f"decoy map {text!r} names seed {seed} twice")
        decoys[seed] = decoy
    return decoys


def format_decoys(decoys: Mapping[int, int]) -> str:
    """Inverse of ``parse_decoys``, entries in seed order."""
    return ",".join(f"{seed}:{decoys[seed]}" for seed in sorted(decoys))


def check_decoys(selection: SeedSelection, decoys: Mapping[int, int]) -> dict[int, int]:
    """Validate a decoy map against the seed selection; returns it as a plain dict in seed order.

    Every selected seed needs a decoy seed of the same role (main to main, dev to dev) that is itself selected and
    differs from it. Entries for unselected seeds are refused, so the map and the selection always change together
    and no instance can point at an unselected model.
    """
    clean: dict[int, int] = {}
    for seed, decoy in decoys.items():
        if type(seed) is not int or type(decoy) is not int:
            raise TypeError(f"decoy map entries must be ints, got {seed!r}: {decoy!r}")
        clean[seed] = decoy
    selected = set(selection.main) | set(selection.dev)
    missing, extra = sorted(selected - set(clean)), sorted(set(clean) - selected)
    if missing or extra:
        raise ValueError(
            f"the decoy map must cover exactly the selected seeds: no entry for {missing}, entries for unselected "
            f"seeds {extra}"
        )
    for seed, decoy in sorted(clean.items()):
        group, name = (selection.main, "main") if seed in selection.main else (selection.dev, "dev")
        if decoy == seed:
            raise ValueError(f"{name} seed {seed} cannot be its own decoy")
        if decoy not in group:
            raise ValueError(f"the decoy of {name} seed {seed} is seed {decoy}, which is not a selected {name} seed")
    return dict(sorted(clean.items()))


def resolve_decoy(entries: Sequence[ModelEntry], entry: ModelEntry, decoys: Mapping[int, int]) -> ModelEntry:
    """Decoy model of an instance: the model of the same role and architecture with seed ``decoys[entry.seed]``."""
    if entry.role not in (ROLE_MAIN, ROLE_DEV):
        raise ValueError(f"model {entry.model_id} has role {entry.role!r}; only base and dev models get decoys")
    if entry.seed not in decoys:
        raise LookupError(f"the decoy map has no entry for seed {entry.seed} ({entry.model_id})")
    decoy_seed = decoys[entry.seed]
    matches = [e for e in entries if e.role == entry.role and e.seed == decoy_seed and e.arch == entry.arch]
    if len(matches) != 1:
        raise LookupError(
            f"decoy for {entry.model_id} (role {entry.role}, seed {entry.seed}) needs exactly one {entry.role} model "
            f"with seed {decoy_seed} and arch {entry.arch!r}; found {len(matches)}"
        )
    decoy = matches[0]
    if decoy.model_id == entry.model_id or decoy.params_sha256 == entry.params_sha256:
        raise ValueError(f"decoy of {entry.model_id} is the model itself")
    return decoy


def _check_same_layout(base: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray], what: str) -> None:
    if set(base) != set(other):
        raise ValueError(f"{what}: keys differ from the base model: {sorted(set(base) ^ set(other))}")
    for key, value in base.items():
        if other[key].shape != value.shape or other[key].dtype != value.dtype:
            raise ValueError(f"{what}: {key} is {other[key].dtype}{other[key].shape}, base {value.dtype}{value.shape}")


# ---------------------------------------------------------------------------------------------------------------
# Store state: instance directories, index.json, changes.jsonl


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _rule_of(private: Mapping[str, Any]) -> str:
    rule = private.get("evidence_rule")
    if rule not in EVIDENCE_RULES:
        raise InstanceConflict(f"instance {private.get('instance_id')!r} has unknown evidence_rule {rule!r}")
    return str(rule)


def _scan_instances(root: Path) -> dict[str, dict[str, Any]]:
    """private.json of every instance directory, by instance id; refuses leftovers of a crashed build."""
    stale = sorted(p.name for p in root.glob(".tmp-*"))
    if stale:
        raise InstanceConflict(
            f"unfinished instance builds in {root}: {stale}. They were never registered; remove them by hand "
            "(rm -r) after inspection and rerun."
        )
    privates = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        if not INSTANCE_ID_RE.fullmatch(path.name):
            raise InstanceConflict(f"unexpected directory {path} in the instance root")
        private = read_json(path / "private.json")
        if private.get("instance_id") != path.name:
            raise InstanceConflict(f"{path}/private.json names instance {private.get('instance_id')!r}")
        _rule_of(private)
        privates[path.name] = private
    return privates


def _instance_key(private: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identity of an instance: (rule, dataset, condition, model) in v1, (rule, dataset, model, a, b) in v0."""
    if _rule_of(private) == EVIDENCE_RULE_V1:
        return (EVIDENCE_RULE_V1, str(private["dataset"]), str(private["condition"]), str(private["model_id"]))
    a, b = private["target"]
    return (EVIDENCE_RULE, str(private["dataset"]), str(private["model_id"]), int(a), int(b))


INDEX_FIELDS = (
    "instance_id",
    "dataset",
    "split_role",
    "model_id",
    "seed",
    "arch",
    "model_sha256",
    "forward_sha256",
    "target",
    "target_names",
    "target_choice",
    "target_rank",
    "vselect_counts",
    "decoy_model_id",
    "evidence_content_sha256",
    "created_at",
)
INDEX_FIELDS_V1 = (
    "instance_id",
    "evidence_rule",
    "dataset",
    "condition",
    "split_role",
    "model_id",
    "seed",
    "arch",
    "model_sha256",
    "forward_sha256",
    "decoy_model_id",
    "n_base_errors_vevidence",
    "n_base_errors_vcheck",
    "evidence_content_sha256",
    "created_at",
)
SUPERSEDED_FIELDS = ("superseded_at", "superseded_reason")


def _index_row(private: Mapping[str, Any]) -> dict[str, Any]:
    fields = INDEX_FIELDS_V1 if _rule_of(private) == EVIDENCE_RULE_V1 else INDEX_FIELDS
    return {field: private[field] for field in fields}


def _order(private: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        private["dataset"],
        _rule_of(private),
        private.get("condition", ""),
        SPLIT_ROLES.index(private["split_role"]),
        private["seed"],
        private["model_id"],
        private.get("target_choice", 0),
    )


def read_index() -> dict[str, Any]:
    """``$WEIGHTCODE_PRIVATE/instances/index.json``; ``instances`` lists the active rows, ``superseded`` the retired ones."""
    path = instances_root() / "index.json"
    if not path.exists():
        return {"schema": INDEX_SCHEMA, "instances": [], "superseded": []}
    index = read_json(path)
    if index.get("schema") != INDEX_SCHEMA or not isinstance(index.get("instances"), list):
        raise InstanceConflict(f"{path} is not a {INDEX_SCHEMA} file")
    superseded = index.setdefault("superseded", [])
    if not isinstance(superseded, list):
        raise InstanceConflict(f"{path}: 'superseded' must be a list")
    return index


@dataclass
class _StoreState:
    """All instance directories and which of them are superseded (with when and why)."""

    root: Path
    privates: dict[str, dict[str, Any]]
    superseded: dict[str, dict[str, str]]

    def active(self) -> dict[str, dict[str, Any]]:
        return {i: p for i, p in self.privates.items() if i not in self.superseded}


def _rows_by_id(rows: list[Any], what: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str):
            raise InstanceConflict(f"index.json {what}: malformed row {row!r}")
        if row["instance_id"] in by_id:
            raise InstanceConflict(f"index.json {what}: {row['instance_id']} is listed twice")
        by_id[row["instance_id"]] = row
    return by_id


def _load_state(root: Path) -> _StoreState:
    """Scan the directories and check index.json against them, row by row."""
    privates = _scan_instances(root)
    index = read_index()
    active_rows = _rows_by_id(index["instances"], "instances")
    retired_rows = _rows_by_id(index["superseded"], "superseded")
    both = sorted(set(active_rows) & set(retired_rows))
    if both:
        raise InstanceConflict(f"index.json lists {both} as both active and superseded")
    listed = set(active_rows) | set(retired_rows)
    if listed != set(privates):
        raise InstanceConflict(
            "instances/index.json and the instance directories disagree: only in index "
            f"{sorted(listed - set(privates))}, only on disk {sorted(set(privates) - listed)}"
        )
    for instance_id, row in active_rows.items():
        if jsonable(row) != jsonable(_index_row(privates[instance_id])):
            raise InstanceConflict(f"index row of {instance_id} differs from its private.json")
    superseded = {}
    for instance_id, row in retired_rows.items():
        meta = {field: row.get(field) for field in SUPERSEDED_FIELDS}
        if not all(isinstance(value, str) and value for value in meta.values()):
            raise InstanceConflict(f"superseded row of {instance_id} lacks {list(SUPERSEDED_FIELDS)}")
        body = {key: value for key, value in row.items() if key not in SUPERSEDED_FIELDS}
        if jsonable(body) != jsonable(_index_row(privates[instance_id])):
            raise InstanceConflict(f"superseded index row of {instance_id} differs from its private.json")
        superseded[instance_id] = meta
    return _StoreState(root=root, privates=privates, superseded=superseded)


def _write_index(state: _StoreState) -> None:
    active = sorted(state.active().values(), key=_order)
    retired = sorted((state.privates[i] for i in state.superseded), key=_order)
    write_json_atomic(
        state.root / "index.json",
        {
            "schema": INDEX_SCHEMA,
            "updated_at": utc_now(),
            "instances": [_index_row(private) for private in active],
            "superseded": [
                dict(_index_row(private), **state.superseded[private["instance_id"]]) for private in retired
            ],
        },
    )


def _log_change(root: Path, record: Mapping[str, Any]) -> None:
    line = json.dumps(dict(record, schema=CHANGE_SCHEMA, time=utc_now()), sort_keys=True, allow_nan=False)
    with open(root / CHANGES_LOG_NAME, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def active_instance_ids(
    *, dataset: str | None = None, evidence_rule: str | None = None, condition: str | None = None
) -> list[str]:
    """Ids of the active (not superseded) instances, optionally of one dataset, evidence rule and condition.

    ``condition`` selects evidence-v1 instances of that deployment condition (v0 instances record none).
    """
    if evidence_rule is not None and evidence_rule not in EVIDENCE_RULES:
        raise ValueError(f"unknown evidence rule {evidence_rule!r}; use {EVIDENCE_RULES}")
    state = _load_state(instances_root())
    return [
        private["instance_id"]
        for private in sorted(state.active().values(), key=_order)
        if (dataset is None or private["dataset"] == dataset)
        and (evidence_rule is None or _rule_of(private) == evidence_rule)
        and (condition is None or private.get("condition") == condition)
    ]


def supersede_instances(instance_ids: Sequence[str], reason: str) -> list[dict[str, Any]]:
    """Retire active instances: their rows move to index.json "superseded", so no runner selects them.

    The directories stay in place and unchanged, so earlier runs on them remain auditable. Every supersession is
    logged with its reason to ``changes.jsonl``. Returns the superseded rows.
    """
    if not reason.strip():
        raise ValueError("superseding instances needs a reason")
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError(f"instance ids repeat: {list(instance_ids)}")
    root = instances_root()
    with _exclusive_lock(root / ".build.lock"):
        state = _load_state(root)
        unknown = [i for i in instance_ids if i not in state.privates]
        retired = [i for i in instance_ids if i in state.superseded]
        if unknown or retired:
            raise InstanceConflict(f"cannot supersede unknown instances {unknown} or superseded ones {retired}")
        when = utc_now()
        for instance_id in instance_ids:
            row = _index_row(state.privates[instance_id])
            _log_change(root, {"event": "superseded", "instance_id": instance_id, "reason": reason, "row": row})
            state.superseded[instance_id] = {"superseded_at": when, "superseded_reason": reason}
        _write_index(state)
        return [dict(_index_row(state.privates[i]), **state.superseded[i]) for i in instance_ids]



class StaleInstanceError(InstanceConflict):
    """An instance that must not be run: not active in the index, or its task.json is no longer what the code writes."""


def check_task_current(instance_dir: str | Path) -> dict[str, Any]:
    """Refuse to run an instance that is not active or whose task.json the current code would not write; else return it.

    task.json is a build-time snapshot of the task text, ``OP_NAMES`` and ``Limits()``. After any of them changes, an
    existing instance states rules that the executor no longer enforces. The store is immutable, so such an instance
    is superseded and rebuilt under a new id (contract section 3). Every runner calls this before an arm or a
    baseline runs on an instance.
    """
    root = Path(instance_dir)
    private = read_json(root / "private.json")
    instance_id = str(private.get("instance_id"))
    if root.name != instance_id:
        raise StaleInstanceError(f"{root}: directory name differs from private.json instance_id {instance_id!r}")
    index = read_index()
    if instance_id not in {row.get("instance_id") for row in index["instances"]}:
        retired = {row.get("instance_id") for row in index["superseded"]}
        status = "superseded" if instance_id in retired else "not in the index"
        raise StaleInstanceError(f"instance {instance_id} is {status}; only active instances are run")
    rule = _rule_of(private)
    if rule != EVIDENCE_RULE_V1:
        raise StaleInstanceError(f"instance {instance_id} follows {rule}; only {EVIDENCE_RULE_V1} instances are run")
    task_text = (root / "task.json").read_text(encoding="utf-8")
    if _text_sha256(task_text) != private.get("task_sha256"):
        raise StaleInstanceError(f"{root / 'task.json'}: bytes do not match private.json task_sha256")
    task = json.loads(task_text)
    expected = json.loads(json_text(build_task_v1(task.get("classes", []))))
    if task != expected:
        changed = sorted(key for key in set(task) | set(expected) if task.get(key) != expected.get(key))
        raise StaleInstanceError(
            f"instance {instance_id}: task.json differs from what the current code writes in {changed}; "
            "supersede the instance and rebuild it"
        )
    return task


def decoy_evidence(instance_dir: str | Path) -> tuple[Path, str]:
    """Evidence for the DECOY view of an evidence-v1 instance: the evidence of its decoy model's own instance.

    The evidence-v1 draw depends only on the model's parameter hash, the deployment condition, the condition's
    V_evidence and the model's predictions on it. The evidence drawn from the decoy model's own errors is therefore
    exactly the evidence of the active sibling instance whose base model is the decoy (same dataset, condition and
    rule), and it is internally consistent with the decoy weights: its pred_* are the decoy's predictions and its
    fail rows are the decoy's errors. Returns that evidence.npz and its canonical content hash, after checking that
    exactly one such sibling is active and that the file still has the hash the sibling recorded. Raises when there is
    no such sibling, so a DECOY view never falls back to the real model's evidence.
    """
    root = Path(instance_dir)
    private = read_json(root / "private.json")
    if _rule_of(private) != EVIDENCE_RULE_V1:
        raise StaleInstanceError(f"{root}: decoy evidence is defined for {EVIDENCE_RULE_V1} instances only")
    decoy_sha = private.get("decoy_model_sha256")
    if not isinstance(decoy_sha, str) or not decoy_sha:
        raise InstanceConflict(f"{root}: private.json records no decoy_model_sha256")
    state = _load_state(instances_root())
    siblings = [
        other
        for other in state.active().values()
        if _rule_of(other) == EVIDENCE_RULE_V1
        and other["dataset"] == private["dataset"]
        and other["condition"] == private["condition"]
        and other["model_sha256"] == decoy_sha
    ]
    if len(siblings) != 1:
        raise InstanceConflict(
            f"{root}: expected exactly one active instance with the decoy model {private.get('decoy_model_id')} on "
            f"({private['dataset']}, {private['condition']}), found {[s['instance_id'] for s in siblings]}"
        )
    sibling = siblings[0]
    path = instances_root() / str(sibling["instance_id"]) / "evidence.npz"
    digest = evidence_content_sha256(load_evidence(path))
    if digest != sibling.get("evidence_content_sha256"):
        raise InstanceConflict(f"{path}: content hash {digest} != recorded {sibling.get('evidence_content_sha256')}")
    return path, digest

def _new_instance_id(root: Path) -> str:
    for _ in range(64):
        candidate = "i" + secrets.token_hex(4)
        if not (root / candidate).exists() and not (root / f".tmp-{candidate}").exists():
            return candidate
    raise RuntimeError("could not draw an unused instance id")


# ---------------------------------------------------------------------------------------------------------------
# Plans


@dataclass(frozen=True)
class _Plan:
    """Deterministic content of one instance before an id is assigned."""

    entry: ModelEntry
    task: dict[str, Any]
    evidence: Evidence
    private_core: dict[str, Any]

    @property
    def key(self) -> tuple[Any, ...]:
        return _instance_key(self.private_core)


@dataclass(frozen=True)
class BuildOutcome:
    """One planned instance after a build: newly created or reused unchanged.

    ``condition`` is set for evidence-v1 instances, ``target`` and ``target_names`` for evidence-v0 instances.
    ``counts`` holds the base error counts (v1) or the V_select counts of the pair (v0).
    """

    instance_id: str
    status: str
    dataset: str
    evidence_rule: str
    split_role: str
    model_id: str
    seed: int
    decoy_model_id: str
    decoy_seed: int
    condition: str | None
    target: tuple[int, int] | None
    target_names: tuple[str, str] | None
    counts: dict[str, int]
    visible_file_findings: dict[str, list[str]]


def _outcome(private: Mapping[str, Any], status: str) -> BuildOutcome:
    v1 = _rule_of(private) == EVIDENCE_RULE_V1
    if v1:
        counts = {
            key: int(private[key])
            for key in ("n_base_errors_vevidence", "n_vevidence", "n_base_errors_vcheck", "n_vcheck")
        }
    else:
        counts = {key: int(value) for key, value in private["vselect_counts"].items()}
    return BuildOutcome(
        instance_id=str(private["instance_id"]),
        status=status,
        dataset=str(private["dataset"]),
        evidence_rule=_rule_of(private),
        split_role=str(private["split_role"]),
        model_id=str(private["model_id"]),
        seed=int(private["seed"]),
        decoy_model_id=str(private["decoy_model_id"]),
        decoy_seed=int(private["decoy_seed"]),
        condition=str(private["condition"]) if v1 else None,
        target=None if v1 else (int(private["target"][0]), int(private["target"][1])),
        target_names=None if v1 else (str(private["target_names"][0]), str(private["target_names"][1])),
        counts=counts,
        visible_file_findings={k: list(v) for k, v in private["visible_file_scan"].items()},
    )


@dataclass(frozen=True)
class _DatasetContext:
    dataset: str
    names: list[str]
    forward: ForwardFn
    forward_sha256: str
    forward_path: Path
    entries: list[ModelEntry]
    ceiling_refs: list[dict[str, Any]]
    visible_scan_forward: list[str]
    task_tokens: list[str]
    condition: str
    # Tokens scanned in every editor-visible file; dataset tokens are recorded, condition tokens refused.
    visible_tokens: list[str]
    condition_tokens: list[str]


def _refuse_condition_tokens(findings: Iterable[str], condition_tokens: Sequence[str], where: str) -> None:
    named = sorted(set(findings) & set(condition_tokens))
    if named:
        raise ValueError(f"{where} contains {named}, which name the condition's corruption; no editor-visible file may")


def _dataset_context(dataset: str, condition: str = DEPLOY_CONDITION) -> _DatasetContext:
    """Clean dataset metadata, forward and models, plus the leak tokens of ``condition``.

    A named condition uses the clean dataset's models, forward.py and class names (``check_condition_store`` has
    checked that they apply) and adds ``condition_leak_tokens`` to the scan; ``deploy`` scans the dataset tokens only,
    as before named conditions existed.
    """
    meta = read_dataset_meta(dataset)
    names = class_names(meta)
    forward, forward_sha, forward_path = load_dataset_forward(dataset, meta)
    entries = read_model_index(dataset)
    ceilings = sorted((e for e in entries if e.role == ROLE_CEILING), key=lambda e: (e.seed, e.model_id))
    condition_tokens: list[str] = []
    if condition != DEPLOY_CONDITION:
        condition_tokens = condition_leak_tokens(condition, read_condition_meta(dataset, condition))
    tokens = sorted(set(leak_tokens(dataset)) | set(condition_tokens))
    forward_findings = scan_for_tokens(forward_path.read_text(encoding="utf-8"), tokens)
    _refuse_condition_tokens(forward_findings, condition_tokens, str(forward_path))
    return _DatasetContext(
        dataset=dataset,
        names=names,
        forward=forward,
        forward_sha256=forward_sha,
        forward_path=forward_path,
        entries=entries,
        ceiling_refs=[
            {"model_id": e.model_id, "seed": e.seed, "arch": e.arch, "params_sha256": e.params_sha256}
            for e in ceilings
        ],
        visible_scan_forward=forward_findings,
        task_tokens=sorted(set(tokens) | {e.model_id.lower() for e in entries}),
        condition=condition,
        visible_tokens=tokens,
        condition_tokens=condition_tokens,
    )


def _model_part(
    ctx: _DatasetContext, entry: ModelEntry, decoys: Mapping[int, int]
) -> tuple[Params, dict[str, Any]]:
    """Load and check a model and its decoy; return the parameters and the private fields both rules share."""
    dataset = ctx.dataset
    params = load_model_params(dataset, entry)
    decoy = resolve_decoy(ctx.entries, entry, decoys)
    decoy_dir = model_dir(dataset, decoy.model_id)
    decoy_params = load_params(decoy_dir / "model.npz")
    if params_sha256(decoy_params) != decoy.params_sha256:
        raise ValueError(f"decoy {decoy.model_id}: model.npz hash differs from the index")
    _check_same_layout(params, decoy_params, f"decoy {decoy.model_id}")
    source_dir = model_dir(dataset, entry.model_id)
    spec_text = (source_dir / "model_spec.json").read_text(encoding="utf-8")
    spec_findings = scan_for_tokens(spec_text, ctx.visible_tokens)
    _refuse_condition_tokens(spec_findings, ctx.condition_tokens, f"{source_dir}/model_spec.json")
    npz_findings = scan_for_tokens(npz_text(params), ctx.visible_tokens)
    if npz_findings:
        raise ValueError(f"{source_dir}/model.npz contains {npz_findings} in its array names")
    spec = read_json(source_dir / "model_spec.json")
    if not isinstance(spec, dict):
        raise ValueError(f"{source_dir}/model_spec.json is not a JSON object")
    if "classes" in spec and spec["classes"] != ctx.names:
        raise ValueError(f"{source_dir}/model_spec.json class list differs from dataset.json")
    shared = {
        "dataset": dataset,
        "split_role": entry.split_role,
        "model_id": entry.model_id,
        "seed": entry.seed,
        "arch": entry.arch,
        "role": entry.role,
        "model_sha256": entry.params_sha256,
        "model_source_path": str(source_dir / "model.npz"),
        "model_spec_sha256": file_sha256(source_dir / "model_spec.json"),
        "forward_sha256": ctx.forward_sha256,
        "decoy_model_id": decoy.model_id,
        "decoy_seed": decoy.seed,
        "decoy_model_path": str(decoy_dir / "model.npz"),
        "decoy_model_sha256": decoy.params_sha256,
        "decoy_model_spec_path": str(decoy_dir / "model_spec.json"),
        "reference_ceiling_models": [dict(ref) for ref in ctx.ceiling_refs],
        "k_fail": K_FAIL,
        "k_rand": K_RAND,
        "visible_file_scan": {
            "model_spec.json": spec_findings,
            "forward.py": list(ctx.visible_scan_forward),
        },
    }
    return params, shared


def _checked_task_text(ctx: _DatasetContext, task: Mapping[str, Any], what: str) -> str:
    text = json_text(task)
    findings = scan_for_tokens(text, ctx.task_tokens)
    if findings:
        raise ValueError(f"task.json for {what} contains forbidden tokens {findings}")
    return text


def _evidence_fields(evidence: Evidence, source_idx: np.ndarray) -> dict[str, Any]:
    return {
        "evidence_seed": evidence.seed,
        "evidence_row_ids": [int(row) for row in evidence.row_ids],
        "evidence_split_ids": evidence.split_ids,
        "evidence_source_idx": [int(source_idx[row]) for row in evidence.row_ids],
        "evidence_content_sha256": evidence_content_sha256(evidence.arrays),
    }


def _plan_model(
    ctx: _DatasetContext,
    entry: ModelEntry,
    v_ev: Mapping[str, np.ndarray],
    split_file_sha256: Mapping[str, str],
    decoys: Mapping[int, int],
    batch_size: int,
) -> list[_Plan]:
    """Rule evidence-v0: one plan per chosen target pair of the model's targets.json."""
    dataset = ctx.dataset
    targets = read_targets(dataset, entry, ctx.forward_sha256)
    if targets["classes"] != ctx.names:
        raise ValueError(f"{targets_path(dataset, entry.model_id)}: class list differs from dataset.json")
    params, shared = _model_part(ctx, entry, decoys)
    pred = predict(ctx.forward, params, v_ev["x"], batch_size)
    plans = []
    for choice, chosen in enumerate(targets["chosen"], start=1):
        a, b = int(chosen["a"]), int(chosen["b"])
        cell = int(np.count_nonzero((v_ev["y"] == a) & (pred == b)))
        if cell != int(chosen["vevidence_count"]):
            raise ValueError(
                f"{entry.model_id} ({a}, {b}): V_evidence cell has {cell} images now, targets.json says "
                f"{chosen['vevidence_count']}"
            )
        seed = evidence_seed(entry.params_sha256, a, b)
        evidence = draw_evidence(v_ev["x"], v_ev["y"], pred, a=a, b=b, seed=seed, k_fail=K_FAIL, k_rand=K_RAND)
        task = build_task(ctx.names, a, b, chosen["vselect_n_a"], chosen["vselect_count"])
        task_text = _checked_task_text(ctx, task, f"{entry.model_id} ({a}, {b})")
        private_core = {
            "schema": PRIVATE_SCHEMA,
            **shared,
            "target": [a, b],
            "target_names": [ctx.names[a], ctx.names[b]],
            "target_choice": choice,
            "target_rank": int(chosen["rank"]),
            "targets_rule": RULE_ID,
            "targets_path": str(targets_path(dataset, entry.model_id)),
            "vselect_counts": {
                "n_class_a": int(chosen["vselect_n_a"]),
                "n_class_a_predicted_b": int(chosen["vselect_count"]),
            },
            "vevidence_count": int(chosen["vevidence_count"]),
            "vcheck_count": int(chosen["vcheck_count"]),
            "vevidence_path": str(split_path(dataset, "V_evidence")),
            "vevidence_sha256": split_file_sha256["V_evidence"],
            "vcheck_path": str(split_path(dataset, "V_check")),
            "vcheck_sha256": split_file_sha256["V_check"],
            "evidence_rule": EVIDENCE_RULE,
            **_evidence_fields(evidence, v_ev["idx"]),
            "task_sha256": _text_sha256(task_text),
        }
        plans.append(_Plan(entry=entry, task=task, evidence=evidence, private_core=private_core))
    return plans


def _plan_model_v1(
    ctx: _DatasetContext,
    entry: ModelEntry,
    splits: DeploymentSplits,
    data: Mapping[str, Mapping[str, np.ndarray]],
    split_file_sha256: Mapping[str, str],
    decoys: Mapping[int, int],
    batch_size: int,
) -> _Plan:
    """Rule evidence-v1: the plan of one (model, deployment condition) instance."""
    params, shared = _model_part(ctx, entry, decoys)
    v_ev, v_ch = data["V_evidence"], data["V_check"]
    pred_ev = predict(ctx.forward, params, v_ev["x"], batch_size)
    pred_ch = predict(ctx.forward, params, v_ch["x"], batch_size)
    seed = evidence_seed_v1(entry.params_sha256, splits.condition)
    evidence = draw_evidence_v1(v_ev["x"], v_ev["y"], pred_ev, seed=seed, k_fail=K_FAIL, k_rand=K_RAND)
    evidence_findings = scan_for_tokens(npz_text(evidence.arrays), ctx.visible_tokens)
    if evidence_findings:
        raise ValueError(f"evidence of {entry.model_id} ({splits.condition}) contains {evidence_findings}")
    task = build_task_v1(ctx.names)
    task_text = _checked_task_text(ctx, task, f"{entry.model_id} ({splits.condition})")
    private_core = {
        "schema": PRIVATE_SCHEMA_V1,
        **shared,
        "condition": splits.condition,
        "vevidence_path": str(splits.v_evidence),
        "vevidence_sha256": split_file_sha256["V_evidence"],
        "vevidence_content_sha256": splits.v_evidence_sha256,
        "vcheck_path": str(splits.v_check),
        "vcheck_sha256": split_file_sha256["V_check"],
        "vcheck_content_sha256": splits.v_check_sha256,
        "n_vevidence": int(len(v_ev["y"])),
        "n_base_errors_vevidence": int(np.count_nonzero(pred_ev != v_ev["y"])),
        "n_vcheck": int(len(v_ch["y"])),
        "n_base_errors_vcheck": int(np.count_nonzero(pred_ch != v_ch["y"])),
        "evidence_rule": EVIDENCE_RULE_V1,
        **_evidence_fields(evidence, v_ev["idx"]),
        "task_sha256": _text_sha256(task_text),
    }
    return _Plan(entry=entry, task=task, evidence=evidence, private_core=private_core)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Applying plans


def _compare_existing(instance_dir: Path, plan: _Plan) -> None:
    """Check that an existing instance has exactly the planned deterministic content."""
    hint = "instances are never modified; supersede it and rebuild to apply a changed rule, model or decoy"
    with no_bytecode_cache():
        load_instance(instance_dir)
    private = read_json(instance_dir / "private.json")
    core = {key: value for key, value in private.items() if key not in NONDETERMINISTIC_PRIVATE_KEYS}
    expected = jsonable(plan.private_core)
    if core != expected:
        differing = sorted(k for k in set(core) | set(expected) if core.get(k) != expected.get(k))
        raise InstanceConflict(f"{instance_dir}: private.json differs from a rebuild in {differing}; {hint}")
    if read_json(instance_dir / "task.json") != jsonable(plan.task):
        raise InstanceConflict(f"{instance_dir}: task.json differs from a rebuild; {hint}")
    if file_sha256(instance_dir / "task.json") != plan.private_core["task_sha256"]:
        raise InstanceConflict(f"{instance_dir}: task.json bytes differ from a rebuild; {hint}")
    if evidence_content_sha256(load_evidence(instance_dir / "evidence.npz")) != plan.private_core[
        "evidence_content_sha256"
    ]:
        raise InstanceConflict(f"{instance_dir}: evidence.npz differs from a rebuild; {hint}")
    if file_sha256(instance_dir / "model_spec.json") != plan.private_core["model_spec_sha256"]:
        raise InstanceConflict(f"{instance_dir}: model_spec.json differs from the model's file; {hint}")


def _create(root: Path, plan: _Plan, forward_path: Path) -> dict[str, Any]:
    """Write one instance under a temporary name, verify it, then rename it into place."""
    instance_id = _new_instance_id(root)
    staging = root / f".tmp-{instance_id}"
    staging.mkdir()
    core = plan.private_core
    shutil.copyfile(core["model_source_path"], staging / "model.npz")
    if params_sha256(load_params(staging / "model.npz")) != core["model_sha256"]:
        raise RuntimeError(f"{staging}/model.npz: copy hash mismatch")
    shutil.copyfile(Path(core["model_source_path"]).with_name("model_spec.json"), staging / "model_spec.json")
    if file_sha256(staging / "model_spec.json") != core["model_spec_sha256"]:
        raise RuntimeError(f"{staging}/model_spec.json: copy hash mismatch")
    shutil.copyfile(forward_path, staging / "forward.py")
    if file_sha256(staging / "forward.py") != core["forward_sha256"]:
        raise RuntimeError(f"{staging}/forward.py: copy hash mismatch")
    if save_evidence(staging / "evidence.npz", plan.evidence.arrays) != core["evidence_content_sha256"]:
        raise RuntimeError(f"{staging}/evidence.npz: content hash mismatch")
    (staging / "task.json").write_text(json_text(plan.task), encoding="utf-8")
    private = dict(core, instance_id=instance_id, created_at=utc_now(), numpy_version=np.__version__)
    (staging / "private.json").write_text(json_text(private), encoding="utf-8")
    with no_bytecode_cache():
        load_instance(staging)
    os.rename(staging, root / instance_id)
    return jsonable(private)


def _apply_plans(
    ctx: _DatasetContext, rule: str, selection: SeedSelection, plans: Sequence[_Plan]
) -> list[BuildOutcome]:
    """Under the build lock: refuse stray instances, compare existing ones with their plans, then create the rest."""
    root = instances_root()
    with _exclusive_lock(root / ".build.lock"):
        state = _load_state(root)
        by_key: dict[tuple[Any, ...], str] = {}
        for instance_id, private in state.active().items():
            key = _instance_key(private)
            if key in by_key:
                raise InstanceConflict(f"two active instances for {key}: {by_key[key]} and {instance_id}")
            by_key[key] = instance_id
        selected = {entry.model_id for entry in ctx.entries if selection.includes(entry)}
        stray = [
            f"{i} ({p['split_role']} seed {p['seed']}, {p['model_id']})"
            for i, p in sorted(state.active().items(), key=lambda item: _order(item[1]))
            if p["dataset"] == ctx.dataset and _rule_of(p) == rule and p["model_id"] not in selected
        ]
        if stray:
            raise InstanceConflict(
                f"{ctx.dataset}: active {rule} instances of models outside the seed selection: {stray}. Supersede "
                "them first (supersede_instances, or build_instances.py --supersede-rule) or select their seeds."
            )
        existing = [(plan, by_key.get(plan.key)) for plan in plans]
        for plan, instance_id in existing:
            if instance_id is not None:
                _compare_existing(root / instance_id, plan)
        outcomes = []
        for plan, found in existing:
            if found is None:
                private = _create(root, plan, ctx.forward_path)
                instance_id = str(private["instance_id"])
                state.privates[instance_id] = private
                by_key[plan.key] = instance_id
                _write_index(state)
                status = "created"
            else:
                instance_id = found
                status = "reused"
            outcomes.append(_outcome(state.privates[instance_id], status))
        if not (root / "index.json").exists():
            _write_index(state)
    return outcomes


def build_deployment_instances(
    dataset: str,
    condition: str,
    *,
    selection: SeedSelection,
    decoys: Mapping[int, int],
    split_roles: Sequence[str] = SPLIT_ROLES,
    batch_size: int = 1024,
) -> list[BuildOutcome]:
    """Rule evidence-v1: build (or reuse) one instance per selected model for one deployment condition.

    Dev models come before main models. The condition's V_evidence and V_check are checked against the content
    hashes dataset.json records; a model with fewer than k_fail base errors on V_evidence is refused. The drawn
    conditions of contract section 2.1 are dev-only: a request for any other split role is refused before anything
    is read.
    """
    if condition in DRAWN_CONDITIONS and set(split_roles) - {"dev"}:
        raise ValueError(
            f"{condition} is a dev-only condition (contract section 2.1); split roles {sorted(split_roles)} were asked"
        )
    decoy_map = check_decoys(selection, decoys)
    splits = deployment_splits(dataset, condition)
    ctx = _dataset_context(dataset, condition)
    entries = select_entries(ctx.entries, selection, split_roles)
    data = {
        "V_evidence": load_checked_split(splits.v_evidence, splits.v_evidence_sha256),
        "V_check": load_checked_split(splits.v_check, splits.v_check_sha256),
    }
    file_sha = {"V_evidence": file_sha256(splits.v_evidence), "V_check": file_sha256(splits.v_check)}
    plans = [_plan_model_v1(ctx, entry, splits, data, file_sha, decoy_map, batch_size) for entry in entries]
    return _apply_plans(ctx, EVIDENCE_RULE_V1, selection, plans)


def build_dataset_instances(
    dataset: str,
    *,
    selection: SeedSelection,
    decoys: Mapping[int, int],
    split_roles: Sequence[str] = SPLIT_ROLES,
    batch_size: int = 1024,
) -> list[BuildOutcome]:
    """Rule evidence-v0: build (or reuse) the instance of every chosen target pair of the selected models.

    Dev models come before main models. Every selected model must already have its targets.json.
    """
    decoy_map = check_decoys(selection, decoys)
    ctx = _dataset_context(dataset)
    entries = select_entries(ctx.entries, selection, split_roles)
    v_evidence = load_split(dataset, "V_evidence")
    file_sha = {split: file_sha256(split_path(dataset, split)) for split in ("V_evidence", "V_check")}
    plans = [
        plan
        for entry in entries
        for plan in _plan_model(ctx, entry, v_evidence, file_sha, decoy_map, batch_size)
    ]
    return _apply_plans(ctx, EVIDENCE_RULE, selection, plans)
