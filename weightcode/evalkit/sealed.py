"""Sealed evaluation on the official test set (contract sections 1 and 6, PREREG sections 3 and 8).

Usage::

    python -m weightcode.evalkit.sealed --manifest PATH

The manifest (schema ``weightcode.sealed_manifest.v0``) has exactly the keys schema, created_at (ISO 8601), note and
entries; each entry has exactly instance_id, arm, repeat (int >= 0), child_params_path (absolute, or relative to the
manifest's directory) and child_params_sha256 (canonical ``params_sha256``). (instance_id, arm, repeat) is unique.

Procedure, in this order:

1. Validate the manifest.
2. Load every child, verify its canonical hash, and verify that it has the same keys, shapes and dtypes as its
   instance's base model and only finite values. Children are streamed, so memory stays bounded.
3. Hash the manifest: sha256 of its canonical JSON (sorted keys, no whitespace), so reformatting the file does not
   make a new manifest.
4. Refuse if that hash appears anywhere in ``$WEIGHTCODE_SEALED/registry.jsonl``: one test read per manifest, a failed
   evaluation included.
5. Store a copy of the manifest as ``manifests/<sha>.json`` and append an ``evaluation_started`` record (manifest
   hash, time, entry count, entries) before any test label is loaded.
6. Per instance, load the sealed test set of its (dataset, deployment condition) (``weightcode.evalkit.targets.
   deployment_splits``; an evidence-v0 instance uses the dataset's own condition "deploy") and check its canonical
   content hash against the hash dataset.json records, predict base and child with the instance's own forward.py
   (``weightcode.evalkit.patch_check.load_forward`` through ``load_instance``) and compute the readouts with
   ``weightcode.evalkit.metrics.sealed_readout``, passed through unchanged. Its primary readout is delta_acc =
   (fixed - broken) / n over the whole test set, paired per image; an evidence-v0 instance passes its target pair,
   which adds R and the other pair readouts. If the test set has a ``color`` key, the readouts also get base_cfr,
   child_cfr and delta_cfr from ``weightcode.evalkit.metrics.color_following_rate``, every field of
   ``weightcode.evalkit.metrics.color_split_fixes`` (all base errors and fixes split by color-following or other) and,
   with a target pair, every field of ``weightcode.evalkit.metrics.color_split_target_fixes``.
7. Write ``results/<sha>.jsonl``, one record per entry in manifest order, and append ``evaluation_finished``. Any
   failure after step 5 appends ``evaluation_failed`` and re-raises; the manifest stays spent.

Every instance in the manifest must be active in ``$WEIGHTCODE_PRIVATE/instances/index.json`` and have split_role "main";
a superseded or dev instance is refused before anything is recorded (dev instances use ``weightcode.evalkit.devreadout``). Only the results path and the entry count are printed. Readouts never go to
stdout.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import load_params, params_sha256
from weightcode.common.paths import private_dir, sealed_dir
from weightcode.data import store
from weightcode.evalkit.evidence import CONDITION_RE, EVIDENCE_RULE, EVIDENCE_RULE_V1
from weightcode.evalkit.instances import read_index
from weightcode.evalkit.metrics import color_following_rate, color_split_fixes, color_split_target_fixes, sealed_readout
from weightcode.evalkit.patch_check import predict
from weightcode.evalkit.targets import (
    DEPLOY_CONDITION,
    DeploymentSplits,
    check_image_arrays,
    deployment_splits,
    file_sha256,
    no_bytecode_cache,
    utc_now,
)

MANIFEST_SCHEMA = "weightcode.sealed_manifest.v0"
RESULT_SCHEMA = "weightcode.sealed_result.v0"
REGISTRY_NAME = "registry.jsonl"
DEFAULT_BATCH_SIZE = 1024

_MANIFEST_KEYS = frozenset({"schema", "created_at", "note", "entries"})
_ENTRY_KEYS = frozenset({"instance_id", "arm", "repeat", "child_params_path", "child_params_sha256"})
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_INSTANCE_RE = re.compile(r"^i[0-9a-f]{8}$")
_ARM_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_DATASET_RE = re.compile(r"^[a-z0-9_]+$")


class SealedEvalError(RuntimeError):
    """A manifest, checkpoint or registry state that the sealed evaluator refuses."""


@dataclass(frozen=True)
class ManifestEntry:
    """One checkpoint to evaluate."""

    instance_id: str
    arm: str
    repeat: int
    child_params_path: Path
    child_params_sha256: str


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of a parsed manifest (sorted keys, compact separators, ASCII)."""
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def load_manifest(path: str | Path) -> tuple[dict[str, Any], list[ManifestEntry]]:
    """Parse and validate a manifest; returns the parsed object and its entries with resolved paths."""
    path = Path(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SealedEvalError(f"{path}: not valid JSON ({error})") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        keys = sorted(manifest) if isinstance(manifest, dict) else type(manifest).__name__
        raise SealedEvalError(f"{path}: manifest keys must be exactly {sorted(_MANIFEST_KEYS)}, got {keys}")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise SealedEvalError(f"{path}: schema must be {MANIFEST_SCHEMA!r}, got {manifest['schema']!r}")
    if not isinstance(manifest["created_at"], str):
        raise SealedEvalError(f"{path}: created_at must be an ISO 8601 string")
    try:
        datetime.fromisoformat(manifest["created_at"])
    except ValueError as error:
        raise SealedEvalError(f"{path}: created_at is not ISO 8601 ({error})") from error
    if not isinstance(manifest["note"], str):
        raise SealedEvalError(f"{path}: note must be a string")
    rows = manifest["entries"]
    if not isinstance(rows, list) or not rows:
        raise SealedEvalError(f"{path}: entries must be a non-empty list")
    entries = []
    slots = set()
    for number, row in enumerate(rows):
        where = f"{path}: entry {number}"
        if not isinstance(row, dict) or set(row) != _ENTRY_KEYS:
            raise SealedEvalError(f"{where}: keys must be exactly {sorted(_ENTRY_KEYS)}")
        instance_id, arm, repeat = row["instance_id"], row["arm"], row["repeat"]
        child_path, child_sha = row["child_params_path"], row["child_params_sha256"]
        if not isinstance(instance_id, str) or not _INSTANCE_RE.fullmatch(instance_id):
            raise SealedEvalError(f"{where}: invalid instance_id {instance_id!r}")
        if not isinstance(arm, str) or not _ARM_RE.fullmatch(arm):
            raise SealedEvalError(f"{where}: invalid arm {arm!r}")
        if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 0:
            raise SealedEvalError(f"{where}: repeat must be an int >= 0, got {repeat!r}")
        if not isinstance(child_path, str) or not child_path:
            raise SealedEvalError(f"{where}: child_params_path must be a non-empty string")
        if not isinstance(child_sha, str) or not _SHA_RE.fullmatch(child_sha):
            raise SealedEvalError(f"{where}: child_params_sha256 must be 64 lower-case hex digits")
        slot = (instance_id, arm, repeat)
        if slot in slots:
            raise SealedEvalError(f"{where}: duplicate (instance_id, arm, repeat) {slot}")
        slots.add(slot)
        resolved = Path(child_path) if Path(child_path).is_absolute() else path.parent / child_path
        entries.append(ManifestEntry(instance_id, arm, repeat, resolved, child_sha))
    return manifest, entries


def read_registry(path: Path) -> list[dict[str, Any]]:
    """All registry records; a malformed line is an error, never skipped."""
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SealedEvalError(f"{path}:{number}: malformed registry line ({error})") from error
        if not isinstance(record, dict) or "event" not in record or "manifest_sha256" not in record:
            raise SealedEvalError(f"{path}:{number}: registry record lacks event or manifest_sha256")
        records.append(record)
    return records


def _append_registry(path: Path, record: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_child(entry: ManifestEntry) -> dict[str, np.ndarray]:
    if not entry.child_params_path.is_file():
        raise SealedEvalError(f"child checkpoint {entry.child_params_path} is not a regular file")
    child = load_params(entry.child_params_path)
    digest = params_sha256(child)
    if digest != entry.child_params_sha256:
        raise SealedEvalError(
            f"{entry.child_params_path}: params_sha256 {digest} != manifest {entry.child_params_sha256}"
        )
    return child


def _check_child_layout(child: Mapping[str, np.ndarray], instance: Instance, entry: ManifestEntry) -> None:
    base = instance.params
    if set(child) != set(base):
        raise SealedEvalError(
            f"{entry.child_params_path}: keys differ from base of {instance.instance_id}: "
            f"{sorted(set(child) ^ set(base))}"
        )
    for key, value in base.items():
        if child[key].shape != value.shape or child[key].dtype != value.dtype:
            raise SealedEvalError(
                f"{entry.child_params_path}: {key} is {child[key].dtype}{child[key].shape}, "
                f"base {value.dtype}{value.shape}"
            )
        if not np.all(np.isfinite(child[key])):
            raise SealedEvalError(f"{entry.child_params_path}: {key} has non-finite values")


@dataclass(frozen=True)
class _Setting:
    """What an instance is evaluated on: its dataset, deployment condition and, for evidence-v0, target pair."""

    dataset: str
    condition: str
    evidence_rule: str
    target: tuple[int, int] | None


def _instance_setting(instance: Instance) -> _Setting:
    private = instance.private
    dataset = private.get("dataset")
    if not isinstance(dataset, str) or not _DATASET_RE.fullmatch(dataset):
        raise SealedEvalError(f"instance {instance.instance_id}: invalid dataset {dataset!r} in private.json")
    rule = private.get("evidence_rule")
    if rule == EVIDENCE_RULE_V1:
        condition = private.get("condition")
        if not isinstance(condition, str) or not CONDITION_RE.fullmatch(condition):
            raise SealedEvalError(f"instance {instance.instance_id}: invalid condition {condition!r} in private.json")
        return _Setting(dataset, condition, rule, None)
    if rule == EVIDENCE_RULE:
        # evidence-v0 instances were built on the dataset's own splits and are read on its own test set.
        a, b = private["target"]
        return _Setting(dataset, DEPLOY_CONDITION, rule, (int(a), int(b)))
    raise SealedEvalError(f"instance {instance.instance_id}: unknown evidence_rule {rule!r} in private.json")


def _load_test_set(splits: DeploymentSplits) -> dict[str, np.ndarray]:
    """Load one sealed test set. Called only after the evaluation_started record is on disk.

    The test set of a named condition must hold the clean test set's ``y`` and ``idx`` in the same order (only ``x``
    is corrupted). ``deployment_splits`` checks this for the private splits but never opens a sealed file, so the
    test sets are compared here.
    """
    with np.load(splits.test, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    check_image_arrays(arrays, str(splits.test))
    if splits.condition != DEPLOY_CONDITION:
        clean_path = store.sealed_test_path(splits.dataset)
        with np.load(clean_path, allow_pickle=False) as archive:
            for key in ("y", "idx"):
                clean = archive[key]
                if clean.dtype != arrays[key].dtype or not np.array_equal(clean, arrays[key]):
                    raise SealedEvalError(f"{splits.test}: {key} differs from the clean test set {clean_path}")
    return arrays


def split_readouts(
    test: Mapping[str, np.ndarray],
    base_pred: np.ndarray,
    child_pred: np.ndarray,
    target: tuple[int, int] | None,
    n_classes: int,
) -> dict[str, Any]:
    """Readout fields of one split, paired per image: ``sealed_readout`` plus, if the split has ``color``, the
    Colored-MNIST color readouts. Shared with the dev readout (``weightcode.evalkit.devreadout``) so both mean the same."""
    a, b = target if target is not None else (None, None)
    readouts = dict(sealed_readout(test["y"], base_pred, child_pred, a, b, n_classes=n_classes))
    if "color" in test:
        y, color = test["y"], test["color"]
        base_cfr = color_following_rate(base_pred, y, color)
        child_cfr = color_following_rate(child_pred, y, color)
        cfr = {"base_cfr": base_cfr, "child_cfr": child_cfr, "delta_cfr": child_cfr - base_cfr}
        groups = [
            ("color-following rate", cfr),
            ("color split of all fixes", color_split_fixes(y, base_pred, child_pred, color)),
        ]
        if a is not None and b is not None:
            target_split = color_split_target_fixes(y, base_pred, child_pred, color, a, b)
            groups.append(("color split of target fixes", target_split))
        for name, fields in groups:
            clash = sorted(set(fields) & set(readouts))
            if clash:
                raise SealedEvalError(f"{name} fields {clash} collide with existing readouts; refusing to overwrite")
            readouts.update(fields)
    return readouts


def _write_results(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def evaluate_manifest(manifest_path: str | Path, batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[Path, int]:
    """Run the sealed procedure for one manifest; returns the results path and the number of records."""
    manifest_path = Path(manifest_path)
    manifest, entries = load_manifest(manifest_path)

    active = {row["instance_id"] for row in read_index()["instances"]}
    instances: dict[str, Instance] = {}
    settings: dict[str, _Setting] = {}
    for entry in entries:
        if entry.instance_id not in instances:
            instance_dir = private_dir() / "instances" / entry.instance_id
            if not instance_dir.is_dir():
                raise SealedEvalError(f"unknown instance {entry.instance_id}")
            if entry.instance_id not in active:
                raise SealedEvalError(
                    f"instance {entry.instance_id} is not active in instances/index.json (superseded or unregistered)"
                )
            with no_bytecode_cache():
                instances[entry.instance_id] = load_instance(instance_dir)
            role = instances[entry.instance_id].private.get("split_role")
            if role != "main":
                raise SealedEvalError(
                    f"instance {entry.instance_id} has split_role {role!r}; dev and main instances share the official "
                    "test set, so only main instances are evaluated (contract section 6)"
                )
            settings[entry.instance_id] = _instance_setting(instances[entry.instance_id])
        child = _load_child(entry)
        _check_child_layout(child, instances[entry.instance_id], entry)
        del child

    sha = manifest_sha256(manifest)
    test_splits: dict[tuple[str, str], DeploymentSplits] = {}
    for setting in settings.values():
        key = (setting.dataset, setting.condition)
        if key not in test_splits:
            try:
                test_splits[key] = deployment_splits(setting.dataset, setting.condition)
            except (LookupError, ValueError) as error:
                raise SealedEvalError(f"{setting.dataset} / {setting.condition}: {error}") from error
            if not test_splits[key].test.is_file():
                raise SealedEvalError(f"no sealed test set at {test_splits[key].test}")

    root = sealed_dir()
    registry = root / REGISTRY_NAME
    results_dir = root / "results"
    manifests_dir = root / "manifests"
    for directory in (results_dir, manifests_dir):
        directory.mkdir(mode=0o700, exist_ok=True)
    results_path = results_dir / f"{sha}.jsonl"

    with _exclusive_lock(root / ".sealed.lock"):
        if any(record["manifest_sha256"] == sha for record in read_registry(registry)):
            raise SealedEvalError(f"manifest {sha} was already evaluated or attempted (see {registry})")
        if results_path.exists():
            raise SealedEvalError(f"{results_path} exists although the registry has no record of {sha}")
        manifest_copy = manifests_dir / f"{sha}.json"
        shutil.copyfile(manifest_path, manifest_copy)
        _append_registry(
            registry,
            {
                "event": "evaluation_started",
                "manifest_sha256": sha,
                "time": utc_now(),
                "n_entries": len(entries),
                "manifest_copy": str(manifest_copy),
                "entries": [
                    {
                        "instance_id": e.instance_id,
                        "arm": e.arm,
                        "repeat": e.repeat,
                        "child_params_sha256": e.child_params_sha256,
                        "split_role": instances[e.instance_id].private.get("split_role"),
                    }
                    for e in entries
                ],
            },
        )
        try:
            tests: dict[tuple[str, str], dict[str, np.ndarray]] = {}
            test_sha: dict[str, str] = {}
            base_preds: dict[str, np.ndarray] = {}
            records = []
            for entry in entries:
                instance = instances[entry.instance_id]
                setting = settings[entry.instance_id]
                key = (setting.dataset, setting.condition)
                label = f"{setting.dataset}/{setting.condition}"
                if key not in tests:
                    splits = test_splits[key]
                    tests[key] = _load_test_set(splits)
                    test_sha[label] = params_sha256(tests[key])
                    if test_sha[label] != splits.test_sha256:
                        raise SealedEvalError(
                            f"{splits.test} content hash {test_sha[label]} != recorded {splits.test_sha256}"
                        )
                test = tests[key]
                n_classes = len(instance.task["classes"])
                if entry.instance_id not in base_preds:
                    base_preds[entry.instance_id] = predict(instance.forward, instance.params, test["x"], batch_size)
                child = _load_child(entry)
                child_pred = predict(instance.forward, child, test["x"], batch_size)
                del child
                records.append(
                    {
                        "schema": RESULT_SCHEMA,
                        "manifest_sha256": sha,
                        "instance_id": entry.instance_id,
                        "arm": entry.arm,
                        "repeat": entry.repeat,
                        "child_params_sha256": entry.child_params_sha256,
                        "base_params_sha256": instance.params_sha256,
                        "forward_sha256": instance.forward_sha256,
                        "dataset": setting.dataset,
                        "condition": setting.condition,
                        "evidence_rule": setting.evidence_rule,
                        "split_role": instance.private.get("split_role"),
                        "target": None if setting.target is None else list(setting.target),
                        "test_sha256": test_sha[label],
                        "readouts": split_readouts(
                            test, base_preds[entry.instance_id], child_pred, setting.target, n_classes
                        ),
                    }
                )
            _write_results(results_path, records)
        except BaseException as error:
            _append_registry(
                registry,
                {
                    "event": "evaluation_failed",
                    "manifest_sha256": sha,
                    "time": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        _append_registry(
            registry,
            {
                "event": "evaluation_finished",
                "manifest_sha256": sha,
                "time": utc_now(),
                "n_entries": len(records),
                "results_path": str(results_path),
                "results_sha256": file_sha256(results_path),
                "test_sha256": test_sha,
            },
        )
    return results_path, len(records)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prints only the results path and the entry count."""
    parser = argparse.ArgumentParser(
        prog="python -m weightcode.evalkit.sealed", description="Sealed evaluation of a frozen manifest on the test set."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="manifest JSON (weightcode.sealed_manifest.v0)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="forward batch size")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        results_path, count = evaluate_manifest(args.manifest, batch_size=args.batch_size)
    except SealedEvalError as error:
        print(f"sealed evaluation refused: {error}", file=sys.stderr)
        return 2
    print(f"results: {results_path}")
    print(f"entries: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
