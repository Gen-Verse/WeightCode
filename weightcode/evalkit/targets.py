"""Pre-registered target selection, rule ``targets-v0`` (contract section 2, PREREG sections 2 and 6).

The rule is fixed before any edit and is written verbatim into every ``targets.json``:

1. Predict every V_select image with the model, using the dataset's ``forward.py``.
2. Count every off-diagonal pair (a, b), a != b: images of true class a predicted as b. (b, a) is a separate pair.
3. Rank all off-diagonal pairs by that count, descending; ties go to (a, b) ascending.
4. Walk the ranking. A pair is accepted only if V_evidence holds at least k_fail = 16 images with y = a and model
   prediction b (enough to fill the evidence file) and V_check holds at least 10 such images (enough base errors
   for the patch-check score to move).
5. The first 2 accepted pairs are the targets. A shortfall is recorded, never filled by relaxing the rule.

Targets are selected for main models (index role "base") and dev models (role "dev") of the selected seeds
(``SeedSelection``, passed in by the caller; the scripts default to main seeds 0 and 1 and dev seeds 100 and 101 by
the user rule of 2026-11-03). Base models of other seeds get nothing, and ceiling models are references and get
none. Which models are selected is not part of the rule, so a targets.json does not depend on the selection.
The ranking uses V_select alone. V_evidence and V_check enter only through the two feasibility counts
of step 4, and the official test set is never read. Pairs with a zero V_select count stay at the end of the ranking;
they could only be reached if fewer than two pairs with a positive count passed step 4.

The module also holds the read-only accessors for the data store (splits, dataset metadata, forward, model index)
that the evidence, instance and sealed stages share.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import Params, load_params, params_sha256
from weightcode.data import store
from weightcode.evalkit.patch_check import ForwardFn, load_forward, predict

TARGETS_SCHEMA = "weightcode.targets.v0"
RULE_ID = "targets-v0"
K_FAIL = 16
MIN_VCHECK = 10
N_TARGETS = 2
RULE_TEXT = (
    "targets-v0. (1) Predict every V_select image with the model, using the dataset forward.py. "
    "(2) Count every off-diagonal pair (a, b), a != b: images of true class a predicted as b; (b, a) is a separate "
    "pair. (3) Rank all off-diagonal pairs by that count, descending; ties go to (a, b) ascending. "
    "(4) Walk the ranking; a pair is accepted only if V_evidence holds at least k_fail = 16 images with y = a and "
    "model prediction b and V_check holds at least 10 such images. "
    "(5) The first 2 accepted pairs are the targets; a shortfall is recorded, never filled by relaxing the rule. "
    "Main (role base) and dev models get targets; ceiling models get none. The official test set is never read."
)

ROLE_MAIN = "base"
ROLE_DEV = "dev"
ROLE_CEILING = "ceiling"
MODEL_ROLES = (ROLE_MAIN, ROLE_DEV, ROLE_CEILING)
SPLIT_ROLE_OF = {ROLE_MAIN: "main", ROLE_DEV: "dev"}
SPLIT_ROLES = ("dev", "main")
SELECTION_SPLITS = ("V_select", "V_evidence", "V_check")

# Seed selection during development (user rule 2026-11-03, PREREG section 2): main seeds 0 and 1 only, seeds 2 to 4
# only when the user says so; dev seeds 100 and 101. These are the scripts' CLI defaults. Library functions take the
# selection as an argument and have no default of their own.
DEFAULT_MAIN_SEEDS: tuple[int, ...] = (0, 1)
DEFAULT_DEV_SEEDS: tuple[int, ...] = (100, 101)
_SEED_RE = re.compile(r"^\d+$")

# Deployment condition of the dataset's own deployment data (user decision 2026-11-03); the data store calls it
# condition None. Any other condition is a named split set of the store ($WEIGHTCODE_PRIVATE/<dataset>/conditions/<name>,
# for example the CIFAR-10 corruptions contrast_s3 and gaussian_noise_s3).
DEPLOY_CONDITION = "deploy"
# Splits every named condition must hold: the dev readout split and the two splits an instance is built from.
CONDITION_SPLITS = ("V_select", "V_evidence", "V_check")
# dataset.json fields a named condition must share with the clean dataset, because its instances use the clean
# models and the clean forward.py.
CONDITION_SHARED_FIELDS = ("class_names", "image_shape", "forward_sha256")

# The index's vselect_acc is the export's own numpy forward.py accuracy on V_select, so the two should agree exactly;
# a gap of more than this many images means the checkpoint, the forward or the split changed after export.
VSELECT_ACC_TOLERANCE_IMAGES = 2

_DATASET_RE = re.compile(r"^[a-z0-9_]+$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class TargetsConflict(RuntimeError):
    """An existing targets.json differs from what the pre-registered rule produces now."""


class ConditionMismatch(ValueError):
    """A named deployment condition is not the clean dataset with only the images changed."""


# ---------------------------------------------------------------------------------------------------------------
# Small shared helpers


def utc_now() -> str:
    """Current UTC time as an ISO 8601 string with seconds."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: str | Path) -> str:
    """sha256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    """Parse a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_text(payload: Any) -> str:
    """Deterministic JSON text (sorted keys, two-space indent, trailing newline, no NaN)."""
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json_atomic(path: str | Path, payload: Any) -> None:
    """Write deterministic JSON to a temporary name in the same directory and rename it into place."""
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(json_text(payload), encoding="utf-8")
    os.replace(temporary, target)


@contextmanager
def no_bytecode_cache() -> Iterator[None]:
    """Run forward.py imports with bytecode caching off, so no ``__pycache__`` appears beside them.

    ``load_forward`` now executes the hashed bytes directly and writes no cache itself; this guard keeps an
    instance directory holding exactly its own files even if a forward.py imports further modules.
    """
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def jsonable(payload: Any) -> Any:
    """Round-trip through JSON so tuples become lists and the result compares equal to a parsed file."""
    return json.loads(json.dumps(payload, allow_nan=False))


# ---------------------------------------------------------------------------------------------------------------
# Data store accessors ($WEIGHTCODE_PRIVATE/<dataset>/...), thin checks around weightcode.data.store


def dataset_dir(dataset: str) -> Path:
    """``$WEIGHTCODE_PRIVATE/<dataset>``; the name must be lower-case letters, digits and underscores and must exist."""
    if not _DATASET_RE.fullmatch(dataset):
        raise ValueError(f"invalid dataset name {dataset!r}")
    root = store.dataset_dir(dataset)
    if not root.is_dir():
        raise FileNotFoundError(f"no dataset directory {root}")
    return root


def split_path(dataset: str, split: str) -> Path:
    """Location of one private split file (the official test set is not reachable here)."""
    dataset_dir(dataset)
    return store.split_path(dataset, split)


def check_image_arrays(arrays: Mapping[str, np.ndarray], origin: str) -> None:
    """Check the split format: x uint8 [N, C, H, W], y and idx int64 [N], optional integer color [N]."""
    for key in ("x", "y", "idx"):
        if key not in arrays:
            raise ValueError(f"{origin}: missing key {key!r} (has {sorted(arrays)})")
    x, y, idx = arrays["x"], arrays["y"], arrays["idx"]
    if x.dtype != np.uint8 or x.ndim != 4:
        raise ValueError(f"{origin}: x must be uint8 [N, C, H, W], got {x.dtype}{x.shape}")
    for name, array in (("y", y), ("idx", idx)):
        if array.dtype != np.int64 or array.ndim != 1:
            raise ValueError(f"{origin}: {name} must be int64 [N], got {array.dtype}{array.shape}")
    if not len(x) == len(y) == len(idx):
        raise ValueError(f"{origin}: length mismatch x {len(x)}, y {len(y)}, idx {len(idx)}")
    if "color" in arrays:
        color = arrays["color"]
        if color.ndim != 1 or len(color) != len(y) or not np.issubdtype(color.dtype, np.integer):
            raise ValueError(f"{origin}: color must be an integer array [N], got {color.dtype}{color.shape}")


def load_split(dataset: str, split: str) -> dict[str, np.ndarray]:
    """Load one private split with its format checked (``weightcode.data.store.load_split``)."""
    dataset_dir(dataset)
    arrays = store.load_split(dataset, split)
    check_image_arrays(arrays, str(store.split_path(dataset, split)))
    return arrays


def read_dataset_meta(dataset: str) -> dict[str, Any]:
    """``dataset.json`` of one dataset, checked to describe ``dataset``."""
    dataset_dir(dataset)
    meta = store.read_dataset_json(dataset)
    if not isinstance(meta, dict):
        raise ValueError(f"{dataset}/dataset.json is not a JSON object")
    return meta


@dataclass(frozen=True)
class DeploymentSplits:
    """Files of one deployment condition and the canonical content hashes dataset.json records for them.

    ``v_select`` is the held-out split of the dev readout (``weightcode.evalkit.devreadout``); no arm and no evidence draw
    reads it.
    """

    dataset: str
    condition: str
    v_evidence: Path
    v_evidence_sha256: str
    v_check: Path
    v_check_sha256: str
    test: Path
    test_sha256: str
    v_select: Path
    v_select_sha256: str


def _recorded_sha(meta: Mapping[str, Any], label: str, *keys: str) -> str:
    node: Any = meta
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    if not isinstance(node, str) or not _SHA_RE.fullmatch(node):
        raise ValueError(f"{label}/dataset.json records no {'.'.join(keys)} content hash")
    return node


def store_condition(condition: str) -> str | None:
    """The data store's name of a deployment condition: None for ``deploy``, the dataset's own data."""
    return None if condition == DEPLOY_CONDITION else condition


def read_condition_meta(dataset: str, condition: str) -> dict[str, Any]:
    """dataset.json of a named condition; an unknown name is refused with the list of known conditions."""
    dataset_dir(dataset)
    known = store.list_conditions(dataset)
    if condition not in known:
        raise LookupError(
            f"{dataset}: deployment condition {condition!r} is not in the data store; known: "
            f"{[DEPLOY_CONDITION, *known]}"
        )
    meta = store.read_dataset_json(dataset, condition)
    if not isinstance(meta, dict):
        raise ValueError(f"{dataset}/conditions/{condition}/dataset.json is not a JSON object")
    return meta


def _split_member(path: Path, key: str) -> np.ndarray:
    """One array of a split file, read without loading the others (x stays on disk)."""
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise ConditionMismatch(f"{path}: no {key!r} array")
        return archive[key]


def check_condition_store(
    dataset: str, condition: str, meta: Mapping[str, Any], clean_meta: Mapping[str, Any]
) -> None:
    """Refuse a named condition unless it is the clean dataset with only ``x`` changed.

    Checked against the clean dataset.json and split files: ``CONDITION_SHARED_FIELDS`` are equal; the condition holds
    ``CONDITION_SPLITS``; every split it records names a clean split, records that split's content hash as
    ``clean_sha256``, and has the clean split's ``y`` and ``idx`` (same order); its test records the clean test hash
    and size. The sealed test files are not read here (the sealed evaluation compares their ``y`` and ``idx``).
    """
    label = f"{dataset} condition {condition}"
    problems = [f"{key} differs from the clean dataset.json" for key in CONDITION_SHARED_FIELDS
                if meta.get(key) != clean_meta.get(key)]
    splits, clean_splits = meta.get("splits"), clean_meta.get("splits")
    if not isinstance(splits, dict) or not isinstance(clean_splits, dict):
        raise ConditionMismatch(f"{label}: dataset.json of the condition or the clean dataset has no 'splits' object")
    problems += [f"no {split} split" for split in CONDITION_SPLITS if split not in splits]
    c = store_condition(condition)
    for split in sorted(splits):
        if split not in clean_splits:
            problems.append(f"split {split} has no clean counterpart")
            continue
        clean_sha = _recorded_sha(clean_meta, dataset, "splits", split, "sha256")
        if splits[split].get("clean_sha256") != clean_sha:
            problems.append(f"{split} clean_sha256 is not the clean split's content hash {clean_sha}")
        for key in ("y", "idx"):
            ours = _split_member(store.split_path(dataset, split, c), key)
            clean = _split_member(store.split_path(dataset, split), key)
            if ours.dtype != clean.dtype or not np.array_equal(ours, clean):
                problems.append(f"{split} {key} differs from the clean split's")
    test, clean_test = meta.get("test"), clean_meta.get("test")
    if not isinstance(test, dict) or not isinstance(clean_test, dict):
        problems.append("dataset.json of the condition or the clean dataset has no 'test' object")
    else:
        if test.get("clean_sha256") != _recorded_sha(clean_meta, dataset, "test", "sha256"):
            problems.append("test clean_sha256 is not the clean test's content hash")
        if test.get("n") != clean_test.get("n"):
            problems.append(f"test n {test.get('n')} != clean test n {clean_test.get('n')}")
    if problems:
        raise ConditionMismatch(f"{label} is refused: " + "; ".join(problems))


def deployment_splits(dataset: str, condition: str) -> DeploymentSplits:
    """Resolve a deployment condition of a dataset to its evidence, check, dev readout and sealed test files.

    ``deploy`` is the dataset's own deployment data: V_evidence, V_check and V_select of the private store and the
    official test set of the sealed store. A named condition (``store.list_conditions``) is the same layout under
    ``conditions/<name>``, refused unless ``check_condition_store`` passes. The hashes are the ones the condition's
    dataset.json records; the files are read and checked against them by whoever loads them.
    """
    clean_meta = read_dataset_meta(dataset)
    if condition == DEPLOY_CONDITION:
        meta, label = clean_meta, dataset
    else:
        meta, label = read_condition_meta(dataset, condition), f"{dataset}/conditions/{condition}"
        check_condition_store(dataset, condition, meta, clean_meta)
    c = store_condition(condition)
    return DeploymentSplits(
        dataset=dataset,
        condition=condition,
        v_evidence=store.split_path(dataset, "V_evidence", c),
        v_evidence_sha256=_recorded_sha(meta, label, "splits", "V_evidence", "sha256"),
        v_check=store.split_path(dataset, "V_check", c),
        v_check_sha256=_recorded_sha(meta, label, "splits", "V_check", "sha256"),
        test=store.sealed_test_path(dataset, c),
        test_sha256=_recorded_sha(meta, label, "test", "sha256"),
        v_select=store.split_path(dataset, "V_select", c),
        v_select_sha256=_recorded_sha(meta, label, "splits", "V_select", "sha256"),
    )


def load_checked_split(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    """Load a split file by path, check its format and its canonical content hash."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    check_image_arrays(arrays, str(path))
    digest = params_sha256(arrays)
    if digest != expected_sha256:
        raise ValueError(f"{path}: content hash {digest} != recorded {expected_sha256}")
    return arrays


def class_names(meta: Mapping[str, Any]) -> list[str]:
    """Class names from dataset.json (``class_names``), index = class id."""
    names = meta.get("class_names")
    if not isinstance(names, list) or len(names) < 2 or not all(isinstance(n, str) and n for n in names):
        raise ValueError(f"dataset.json needs a 'class_names' list of non-empty strings, got {names!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate class names: {names}")
    return list(names)


def load_dataset_forward(dataset: str, meta: Mapping[str, Any]) -> tuple[ForwardFn, str, Path]:
    """Import the dataset's forward.py and check its file hash against dataset.json."""
    dataset_dir(dataset)
    path = store.forward_path(dataset)
    with no_bytecode_cache():
        forward, digest = load_forward(path)
    expected = meta.get("forward_sha256")
    if digest != expected:
        raise ValueError(f"{path}: sha256 {digest} != dataset.json forward_sha256 {expected}")
    return forward, digest, path


@dataclass(frozen=True)
class ModelEntry:
    """One row of ``models/index.json``."""

    model_id: str
    seed: int
    arch: str
    role: str
    params_sha256: str
    vselect_acc: float

    @property
    def split_role(self) -> str:
        """Split role of a model that gets instances: main for role base, dev for role dev."""
        if self.role not in SPLIT_ROLE_OF:
            raise ValueError(f"model {self.model_id} has role {self.role!r}, which has no split role")
        return SPLIT_ROLE_OF[self.role]


def _model_entry(row: Any, origin: str) -> ModelEntry:
    if not isinstance(row, dict):
        raise ValueError(f"{origin}: model rows must be objects, got {row!r}")
    for key in ("model_id", "seed", "arch", "role", "params_sha256", "vselect_acc"):
        if key not in row:
            raise ValueError(f"{origin}: model row lacks {key!r}: {row}")
    model_id, seed, arch, role, digest = row["model_id"], row["seed"], row["arch"], row["role"], row["params_sha256"]
    if not isinstance(model_id, str) or not _MODEL_ID_RE.fullmatch(model_id):
        raise ValueError(f"{origin}: invalid model_id {model_id!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{origin}: seed of {model_id} must be an int, got {seed!r}")
    if role not in MODEL_ROLES:
        raise ValueError(f"{origin}: role of {model_id} must be one of {MODEL_ROLES}, got {role!r}")
    if not isinstance(digest, str) or not _SHA_RE.fullmatch(digest):
        raise ValueError(f"{origin}: params_sha256 of {model_id} is not a sha256 hex digest")
    acc = row["vselect_acc"]
    if not isinstance(acc, (int, float)) or isinstance(acc, bool):
        raise ValueError(f"{origin}: vselect_acc of {model_id} must be a number")
    return ModelEntry(model_id, seed, str(arch), role, digest, float(acc))


def read_model_index(dataset: str) -> list[ModelEntry]:
    """All models of one dataset from ``models/index.json`` (a JSON list), with unique ids and (role, seed, arch)."""
    dataset_dir(dataset)
    path = store.models_dir(dataset) / "index.json"
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list of model rows")
    entries = [_model_entry(row, str(path)) for row in rows]
    ids = [entry.model_id for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate model ids")
    slots = [(entry.role, entry.seed, entry.arch) for entry in entries]
    if len(set(slots)) != len(slots):
        raise ValueError(f"{path}: two models share (role, seed, arch)")
    return entries


def model_dir(dataset: str, model_id: str) -> Path:
    """``$WEIGHTCODE_PRIVATE/<dataset>/models/<model_id>``."""
    if not _MODEL_ID_RE.fullmatch(model_id):
        raise ValueError(f"invalid model_id {model_id!r}")
    dataset_dir(dataset)
    return store.models_dir(dataset) / model_id


def load_model_params(dataset: str, entry: ModelEntry) -> Params:
    """Load a model's parameters and check their canonical hash against the index."""
    path = model_dir(dataset, entry.model_id) / "model.npz"
    params = load_params(path)
    digest = params_sha256(params)
    if digest != entry.params_sha256:
        raise ValueError(f"{path}: params_sha256 {digest} != index {entry.params_sha256}")
    return params


@dataclass(frozen=True)
class SeedSelection:
    """Training seeds whose models get targets and instances: ``main`` (index role base) and ``dev`` (role dev).

    The two sets are disjoint, so a seed names one model per architecture and a decoy map can be keyed by seed.
    """

    main: tuple[int, ...]
    dev: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, seeds in (("main", self.main), ("dev", self.dev)):
            if not isinstance(seeds, tuple) or not all(type(seed) is int for seed in seeds):
                raise TypeError(f"{name} seeds must be a tuple of ints, got {seeds!r}")
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{name} seeds {seeds} repeat a seed")
        shared = sorted(set(self.main) & set(self.dev))
        if shared:
            raise ValueError(f"seeds {shared} are both main and dev seeds; the two sets must be disjoint")

    def seeds_for(self, role: str) -> tuple[int, ...]:
        """Selected seeds of index role ``role`` (base or dev)."""
        if role == ROLE_MAIN:
            return self.main
        if role == ROLE_DEV:
            return self.dev
        raise ValueError(f"role {role!r} has no selected seeds; only base and dev models get targets and instances")

    def includes(self, entry: ModelEntry) -> bool:
        """Whether the model is a selected main or dev model."""
        return entry.role in SPLIT_ROLE_OF and entry.seed in self.seeds_for(entry.role)


def parse_seeds(text: str) -> tuple[int, ...]:
    """Parse a comma-separated seed list such as ``"0,1"``; an empty string selects no seed."""
    if not text.strip():
        return ()
    parts = [part.strip() for part in text.split(",")]
    bad = [part for part in parts if not _SEED_RE.fullmatch(part)]
    if bad:
        raise ValueError(f"seed list {text!r}: {bad} are not non-negative integers")
    return tuple(int(part) for part in parts)


def select_entries(
    entries: Iterable[ModelEntry], selection: SeedSelection, split_roles: Sequence[str]
) -> list[ModelEntry]:
    """Selected models of the requested split roles, dev before main, then by seed.

    Every selected seed of a requested split role must have a model of that role in the index; a missing one is an
    error, never a silent skip.
    """
    unknown = set(split_roles) - set(SPLIT_ROLES)
    if unknown:
        raise ValueError(f"unknown split roles {sorted(unknown)}; use {SPLIT_ROLES}")
    entries = list(entries)
    chosen = []
    for role, split_role in SPLIT_ROLE_OF.items():
        if split_role not in split_roles:
            continue
        for seed in selection.seeds_for(role):
            matches = [e for e in entries if e.role == role and e.seed == seed]
            if not matches:
                raise LookupError(
                    f"{split_role} seed {seed} is selected, but models/index.json has no {role} model with that seed"
                )
            chosen.extend(matches)
    return sorted(chosen, key=lambda e: (SPLIT_ROLES.index(e.split_role), e.seed, e.model_id))


# ---------------------------------------------------------------------------------------------------------------
# The rule


def confusion_counts(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Confusion matrix C[a, b] = #{y = a, pred = b}, int64 [n_classes, n_classes]."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    if y.ndim != 1 or pred.shape != y.shape:
        raise ValueError(f"shape mismatch: y {y.shape}, pred {pred.shape}")
    if not (np.issubdtype(y.dtype, np.integer) and np.issubdtype(pred.dtype, np.integer)):
        raise TypeError("labels and predictions must be integer arrays")
    for name, array in (("y", y), ("pred", pred)):
        if array.size and (array.min() < 0 or array.max() >= n_classes):
            raise ValueError(f"{name} has values outside [0, {n_classes})")
    counts = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(counts, (y, pred), 1)
    return counts


def rank_pairs(confusion: np.ndarray) -> list[tuple[int, int, int]]:
    """All off-diagonal (a, b, count), sorted by count descending, then (a, b) ascending."""
    confusion = np.asarray(confusion)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(f"confusion matrix must be square, got {confusion.shape}")
    n = confusion.shape[0]
    pairs = [(a, b, int(confusion[a, b])) for a in range(n) for b in range(n) if a != b]
    return sorted(pairs, key=lambda item: (-item[2], item[0], item[1]))


@dataclass(frozen=True)
class PairDecision:
    """One ranked pair with its feasibility counts and the rule's decision."""

    rank: int
    a: int
    b: int
    vselect_count: int
    vselect_n_a: int
    vevidence_count: int
    vcheck_count: int
    accepted: bool
    chosen: bool
    reasons: tuple[str, ...]


def select_pairs(
    vselect: np.ndarray,
    vevidence: np.ndarray,
    vcheck: np.ndarray,
    *,
    k_fail: int = K_FAIL,
    min_vcheck: int = MIN_VCHECK,
    n_targets: int = N_TARGETS,
) -> list[PairDecision]:
    """Apply steps 3 to 5 of the rule to three confusion matrices; returns the full ranked list."""
    shapes = {np.shape(vselect), np.shape(vevidence), np.shape(vcheck)}
    if len(shapes) != 1:
        raise ValueError(f"confusion matrices differ in shape: {shapes}")
    decisions = []
    n_chosen = 0
    for rank, (a, b, count) in enumerate(rank_pairs(vselect), start=1):
        n_evidence = int(vevidence[a, b])
        n_check = int(vcheck[a, b])
        reasons = []
        if n_evidence < k_fail:
            reasons.append(f"V_evidence count {n_evidence} < k_fail {k_fail}")
        if n_check < min_vcheck:
            reasons.append(f"V_check count {n_check} < {min_vcheck}")
        accepted = not reasons
        chosen = accepted and n_chosen < n_targets
        if chosen:
            n_chosen += 1
            reasons.append(f"accepted; chosen as target {n_chosen}")
        elif accepted:
            reasons.append(f"accepted; not chosen, {n_targets} targets already chosen")
        decisions.append(
            PairDecision(
                rank=rank,
                a=a,
                b=b,
                vselect_count=count,
                vselect_n_a=int(np.asarray(vselect)[a].sum()),
                vevidence_count=n_evidence,
                vcheck_count=n_check,
                accepted=accepted,
                chosen=chosen,
                reasons=tuple(reasons),
            )
        )
    return decisions


# ---------------------------------------------------------------------------------------------------------------
# targets.json


def targets_path(dataset: str, model_id: str) -> Path:
    """``$WEIGHTCODE_PRIVATE/<dataset>/models/<model_id>/targets.json``."""
    return model_dir(dataset, model_id) / "targets.json"


def _exported_vselect_confusion(dataset: str, entry: ModelEntry) -> list[list[int]] | None:
    """The V_select confusion the export recorded in the model's private.json, if it recorded one."""
    path = model_dir(dataset, entry.model_id) / "private.json"
    if not path.exists():
        return None
    record = read_json(path)
    if not isinstance(record, dict):
        raise ValueError(f"{path} is not a JSON object")
    return record.get("vselect_confusion")


def compute_targets(
    dataset: str,
    entry: ModelEntry,
    *,
    names: Sequence[str],
    forward: ForwardFn,
    forward_sha256: str,
    splits: Mapping[str, Mapping[str, np.ndarray]],
    split_sha256: Mapping[str, str],
    batch_size: int,
) -> dict[str, Any]:
    """The targets.json record of one model (everything except ``created_at`` is deterministic)."""
    if entry.role not in SPLIT_ROLE_OF:
        raise ValueError(f"model {entry.model_id} has role {entry.role!r}; only base and dev models get targets")
    params = load_model_params(dataset, entry)
    n_classes = len(names)
    confusion = {}
    for split in SELECTION_SPLITS:
        pred = predict(forward, params, splits[split]["x"], batch_size)
        confusion[split] = confusion_counts(splits[split]["y"], pred, n_classes)
    vselect = confusion["V_select"]
    n_vselect = int(vselect.sum())
    acc = float(np.trace(vselect)) / n_vselect
    gap_images = abs(acc - entry.vselect_acc) * n_vselect
    if gap_images > VSELECT_ACC_TOLERANCE_IMAGES + 1e-6:
        raise ValueError(
            f"model {entry.model_id}: forward.py gives V_select accuracy {acc:.6f}, index says "
            f"{entry.vselect_acc:.6f} ({gap_images:.1f} images apart); forward and checkpoint disagree"
        )
    exported = _exported_vselect_confusion(dataset, entry)
    if exported is not None and exported != vselect.tolist():
        raise ValueError(
            f"model {entry.model_id}: V_select confusion from forward.py differs from the one recorded at export"
        )
    decisions = select_pairs(vselect, confusion["V_evidence"], confusion["V_check"])
    chosen = [
        {
            "a": d.a,
            "b": d.b,
            "a_name": names[d.a],
            "b_name": names[d.b],
            "rank": d.rank,
            "vselect_count": d.vselect_count,
            "vselect_n_a": d.vselect_n_a,
            "vevidence_count": d.vevidence_count,
            "vcheck_count": d.vcheck_count,
        }
        for d in decisions
        if d.chosen
    ]
    return {
        "schema": TARGETS_SCHEMA,
        "rule_id": RULE_ID,
        "rule": RULE_TEXT,
        "rule_params": {"k_fail": K_FAIL, "min_vcheck": MIN_VCHECK, "n_targets": N_TARGETS},
        "dataset": dataset,
        "model_id": entry.model_id,
        "seed": entry.seed,
        "arch": entry.arch,
        "role": entry.role,
        "split_role": entry.split_role,
        "params_sha256": entry.params_sha256,
        "forward_sha256": forward_sha256,
        "split_files": {
            split: {"sha256": split_sha256[split], "n": int(len(splits[split]["y"]))} for split in SELECTION_SPLITS
        },
        "classes": list(names),
        "vselect_acc_index": entry.vselect_acc,
        "vselect_acc_forward": acc,
        "vselect_confusion_check": "equal to model private.json" if exported is not None else "not recorded at export",
        "vselect_confusion": vselect.tolist(),
        "ranked_pairs": [
            {
                "rank": d.rank,
                "a": d.a,
                "b": d.b,
                "vselect_count": d.vselect_count,
                "vselect_n_a": d.vselect_n_a,
                "vevidence_count": d.vevidence_count,
                "vcheck_count": d.vcheck_count,
                "accepted": d.accepted,
                "chosen": d.chosen,
                "reasons": list(d.reasons),
            }
            for d in decisions
        ],
        "chosen": chosen,
        "shortfall": N_TARGETS - len(chosen),
    }


def _without_created_at(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "created_at"}


def write_targets(path: str | Path, record: Mapping[str, Any]) -> bool:
    """Write targets.json; an identical existing file is kept (returns False), a different one is refused."""
    target = Path(path)
    fresh = jsonable(_without_created_at(record))
    if target.exists():
        existing = read_json(target)
        if _without_created_at(existing) == fresh:
            return False
        raise TargetsConflict(
            f"{target} exists with different content; pre-registered targets are never overwritten. "
            "Inspect both versions and remove the old file by hand if the model really changed."
        )
    write_json_atomic(target, dict(fresh, created_at=utc_now()))
    return True


def read_targets(dataset: str, entry: ModelEntry, forward_sha256: str) -> dict[str, Any]:
    """Read a model's targets.json and check that it belongs to this checkpoint, forward and rule."""
    path = targets_path(dataset, entry.model_id)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/select_targets.py --dataset {dataset} first")
    record = read_json(path)
    checks = {
        "schema": TARGETS_SCHEMA,
        "rule_id": RULE_ID,
        "dataset": dataset,
        "model_id": entry.model_id,
        "params_sha256": entry.params_sha256,
        "forward_sha256": forward_sha256,
    }
    for key, expected in checks.items():
        if record.get(key) != expected:
            raise ValueError(f"{path}: {key} is {record.get(key)!r}, expected {expected!r}")
    return record


@dataclass(frozen=True)
class TargetsOutcome:
    """What target selection did for one model."""

    model_id: str
    seed: int
    split_role: str
    path: Path
    written: bool
    chosen: tuple[dict[str, Any], ...]
    shortfall: int


def run_target_selection(
    dataset: str,
    *,
    selection: SeedSelection,
    split_roles: Sequence[str] = SPLIT_ROLES,
    batch_size: int = 1024,
) -> list[TargetsOutcome]:
    """Apply the rule to the selected main and/or dev models of a dataset and write each model's targets.json."""
    meta = read_dataset_meta(dataset)
    names = class_names(meta)
    forward, forward_sha, _ = load_dataset_forward(dataset, meta)
    entries = select_entries(read_model_index(dataset), selection, split_roles)
    splits = {split: load_split(dataset, split) for split in SELECTION_SPLITS}
    split_sha = {split: file_sha256(split_path(dataset, split)) for split in SELECTION_SPLITS}
    outcomes = []
    for entry in entries:
        record = compute_targets(
            dataset,
            entry,
            names=names,
            forward=forward,
            forward_sha256=forward_sha,
            splits=splits,
            split_sha256=split_sha,
            batch_size=batch_size,
        )
        path = targets_path(dataset, entry.model_id)
        written = write_targets(path, record)
        outcomes.append(
            TargetsOutcome(
                model_id=entry.model_id,
                seed=entry.seed,
                split_role=entry.split_role,
                path=path,
                written=written,
                chosen=tuple(record["chosen"]),
                shortfall=int(record["shortfall"]),
            )
        )
    return outcomes
