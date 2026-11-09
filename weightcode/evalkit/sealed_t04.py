"""The ``vision_t04`` readout of ``weightcode.evalkit.sealed_finals`` and the one sealed read of in1k_t04 (PREREG "T04",
sealed test section; list v6).

An entry (readout ``vision_t04``) is a network on a dev instance of the native ImageNet dataset ``in1k_t04``
(``weightcode.data.t04``): a LEARN session's final or a reference run's kept pick, recovered and hash-checked exactly as a
``vision_joint`` entry is. Its readout of record is the vision readout of record: ``split_readouts`` of base and child
predictions on the condition's rows and on their clean companion, gain and clean 100 * delta_acc. On the dev rows
(V_select of the condition and the clean V_select) a record must equal the dev readout of record field for field, so
the dev dry run (``python -m weightcode.evalkit.sealed_finals --list ... --dev``) runs through this family unchanged.

in1k_t04's 38,000 test rows are not materialised anywhere. They are the validation rows with ``row % 25`` not in 0..5
of the 14 Hugging Face parquet shards at the pinned revision, decoded once, at the single sealed read, on the
workstation from shards downloaded under ``$WEIGHTCODE_SEALED/in1k_t04/hf_49e2ee26`` (``scripts/sealed/t04/
download_shards.py``). The generic sealed path of ``sealed_finals`` (one entry at a time, rows cached in memory) does
not fit 38,000 images at 224 x 224 and refuses this family before anything is registered. This module's read is one
streaming pass instead::

    python -m weightcode.evalkit.sealed_t04 plan --list LIST --out PLAN
    python -m weightcode.evalkit.sealed_t04 fidelity --kit KIT --shards SHARDS --out RECORD [--conditions C ...]
    python -m weightcode.evalkit.sealed_t04 sealed --list LIST --secondary PLAN --kit KIT [--workers N] [--batch-size B]

The kit (``scripts/sealed/t04/pull_kit.sh``, from NSCC; nothing sealed in it) holds the in1k_t04 dataset.json files,
the official make_imagenet_c.py, the shards' manifest.sha256 and the frozen networks (``frozen/<name>.npz`` exported
by ``scripts/t04/export_frozen.py`` on NSCC, where each child was rebuilt from its source and hash-checked); its
``kit.json`` lists every file's sha256. The instances are installed under ``$WEIGHTCODE_PRIVATE/instances`` by the same
script (base params and forward.py come from them).

``fidelity`` (dev rows only; also the first step of ``sealed``): the shards and classes.py against the manifest
(``imagenet_val.verify_manifest``); the dev rows' labels; the four split row lists (``t04.split_rows``) against the
row-list sha256 dataset.json records; the clean V_check decoded and cropped (``t04.center_crop_u8``) and each
condition's V_check corrupted (``t04.corrupt_split``) from the local shards, each against the content sha256 of the
NSCC store. Any mismatch stops before any test row (PREREG T04, "fidelity first").

``sealed``: refused unless the list is approved, every entry is ``vision_t04``, the code is a clean git tree in which
the list and the secondary plan are tracked, the kit verifies, every frozen file has the entry's params_sha256, and
the fidelity check passes. Under ``$WEIGHTCODE_SEALED/.sealed.lock`` it refuses a list or a (instance, child) any earlier
started record of ``$WEIGHTCODE_SEALED/registry.jsonl`` holds, copies the list to ``manifests/<sha>.json`` and appends
``evaluation_started`` (every entry, the plan's hash, the fidelity record) and ``entry_started`` for every entry
before any test row is opened. The test rows are then read in 14 units, one per shard, ``--workers`` at a time (each
unit in a spawned process with the environment's BLAS threads; a unit opens its shard only after finding the read's
``evaluation_started`` in the registry). A unit decodes and crops each of its test rows once, applies each needed
corruption (the official function, np.uint8, JPEG q85, decode; gaussian_noise seeded per row as in the store build),
forwards every network that reads that set, and writes only the predictions (no image) to ``reads/<sha>/
unit_<k>.npz``; ``unit_finished`` records the file's sha256. When every unit is done the predictions are assembled in
row order (checked against the test row list and its sha256) and every entry gets one record in
``results/<sha>.jsonl`` and ``entry_finished``; then ``evaluation_finished``. An error or an interrupt (a unit that raises, a worker killed
for memory, Ctrl-C) appends ``evaluation_interrupted`` with the units finished so far and re-raises; the entries stay
spent for every other list, and nothing is printed or summarised from the unit files.

``--resume`` continues a read that has ``evaluation_started`` and neither ``evaluation_finished`` nor
``evaluation_failed`` (an interrupt, or a process killed without any record): same list, secondary plan, kit, code
commit and batch size (the hash of the read plan); units with a ``unit_finished`` record and an intact file are kept,
the others are read; ``evaluation_resumed`` is appended. It is the same read, never a second one, and nothing is
chosen by it. A resume is the coordinator's decision: it is never started automatically.

Per entry the record holds, against the instance's base on the 38,000 test rows: ``readouts`` (the condition),
``readouts_clean`` (clean), their ``per_image`` correctness bits (``sealed_range.per_image``), gain, clean and joint;
and, for the networks the secondary plan names (the base, the LEARN finals and the main control), ``secondary``: per
set (the target corruption at severities 1 to 5 and the other corruption at its chosen severity) the readout on the
20,000 test rows with ``row % 25`` in 6..15. Only the results path and counts are printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import load_params, params_sha256
from weightcode.common.paths import runs_dir, sealed_dir
from weightcode.data import imagenet_val as V
from weightcode.data import store
from weightcode.data import t04 as T
from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit.patch_check import load_forward
from weightcode.evalkit.sealed import REGISTRY_NAME, _append_registry, _exclusive_lock, manifest_sha256, read_registry
from weightcode.evalkit.sealed_range import readout_parts
from weightcode.evalkit.targets import file_sha256, read_condition_meta, read_dataset_meta, utc_now

READOUT = "vision_t04"
SHARDS_LOCATION = "$WEIGHTCODE_SEALED/in1k_t04/hf_49e2ee26"
"""Where the local sealed read finds the 14 validation shards, classes.py and manifest.sha256 (PREREG T04)."""
FORMAT = "in1k_t04 validation shards: test rows decoded once at the sealed read (weightcode.data.t04)"
NOT_MATERIALISED = "not materialised"
SECONDARY_RESIDUES: tuple[int, ...] = tuple(range(6, 16))
"""PREREG T04: the secondary readout reads the test rows with row % 25 in 6..15 (20,000 rows)."""
EVENT_KIND = "sealed_t04"
UNIT_FINISHED, EVALUATION_RESUMED, EVALUATION_INTERRUPTED = "unit_finished", "evaluation_resumed", \
    "evaluation_interrupted"
KIT_SCHEMA, KIT_FILE = "weightcode.t04.sealed_kit.v1", "kit.json"
PLAN_SCHEMA = "weightcode.t04.secondary_plan.v1"
SECONDARY_CONDITIONS = {"gaussian": "gaussian_noise_s3", "contrast": "contrast_s4"}
"""List tag -> the condition of record (PREREG T04: the severities the dev severity rule chose)."""
FIDELITY_SCHEMA = "weightcode.t04.fidelity.v1"
DEFAULT_BATCH = 64
DEFAULT_WORKERS = 1
CLEAN = "clean"
PRED_DTYPE = np.int16


class VisionT04(SF.VisionJoint):
    """``vision_joint`` on in1k_t04 instances, with the shard-based sealed index; no generic sealed read."""

    readout = READOUT
    code_modules: tuple[str, ...] = ("weightcode.evalkit.sealed_t04", "weightcode.data.t04", "weightcode.data.imagenet_val")
    generic_sealed = False

    def load_instance(self, instance_id: str) -> Instance:
        instance = super().load_instance(instance_id)
        if instance.private.get("dataset") != T.DATASET:
            raise SF.SealedFinalsError(f"{instance_id} is on {instance.private.get('dataset')!r}; a {READOUT} entry is "
                                       f"on {T.DATASET}")
        return instance

    def sealed_index(self, instance: Instance) -> dict[str, Any]:
        return t04_sealed_index(str(instance.private["condition"]))

    def sealed_paths(self, index: Mapping[str, Any]) -> list[Path]:
        root = Path(index["shards"]["path"])
        return [root / V.MANIFEST, root / V.CLASSES_FILE] + [root / T.SHARD_DIR / name for name in V.shard_names()]

    def sealed_rows(self, instance: Instance, index: Mapping[str, Any], cache: dict[Any, Any]
                    ) -> tuple[Any, dict[str, Any]]:
        raise SF.SealedFinalsError(f"{READOUT}: the test rows are read only by the streaming pass of "
                                   "weightcode.evalkit.sealed_t04, never one entry at a time")

    def environment(self) -> dict[str, Any]:
        return {"image_libraries": T.image_libraries()}


# ------------------------------------------------------------------------------------------------- metadata and index


def _meta(condition: str | None, meta_root: Path | None) -> tuple[dict[str, Any], Path]:
    """dataset.json of in1k_t04 (``condition`` None) or of one condition: from the store, or from a kit's copy."""
    if meta_root is None:
        try:
            meta = read_dataset_meta(T.DATASET) if condition is None else read_condition_meta(T.DATASET, condition)
        except (LookupError, ValueError) as error:
            raise SF.SealedFinalsError(f"{T.DATASET}/{condition}: {error}") from error
        return meta, store.dataset_json_path(T.DATASET, condition)
    path = meta_root / "dataset.json" if condition is None else meta_root / "conditions" / condition / "dataset.json"
    if not path.is_file():
        raise SF.SealedFinalsError(f"no {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def t04_sealed_index(condition: str, meta_root: Path | None = None) -> dict[str, Any]:
    """The sealed test of one in1k_t04 condition as the dataset.json files record it (from the store, or from a kit's
    copies under ``meta_root``). Opens only those files; the shard directory is resolved under ``$WEIGHTCODE_SEALED`` and
    nothing below it is opened, listed or statted."""
    clean_meta, clean_json = _meta(None, meta_root)
    condition_meta, condition_json = _meta(condition, meta_root)
    clean_test, condition_test = clean_meta.get("test") or {}, condition_meta.get("test") or {}
    rows_sha = T.rows_sha256(T.test_rows())
    if not str(clean_test.get("location", "")).startswith(NOT_MATERIALISED) or clean_test.get("sha256") != rows_sha \
            or clean_test.get("n") != T.N_TEST:
        raise SF.SealedFinalsError(f"{clean_json}: the test record is not in1k_t04's unmaterialised row rule "
                                   f"(n {T.N_TEST}, row-list sha256 {rows_sha})")
    expected = hashlib.sha256(f"{rows_sha}:{condition}".encode()).hexdigest()
    if condition_test.get("clean_sha256") != rows_sha or condition_test.get("sha256") != expected:
        raise SF.SealedFinalsError(f"{condition_json}: the test record does not name the clean test rows and "
                                   f"{condition}")
    recipe = condition_meta.get("condition") or {}
    if recipe.get("name") != condition or recipe.get("corruption") not in T.CORRUPTIONS:
        raise SF.SealedFinalsError(f"{condition_json}: no corruption recipe for {condition}")
    source = clean_meta.get("source_provenance") or {}
    if source.get("revision") != V.REVISION or source.get("manifest_sha256") != V.MANIFEST_SHA256:
        raise SF.SealedFinalsError(f"{clean_json}: source revision or manifest is not the pinned one")
    return {
        "dataset": T.DATASET, "condition": condition, "format": FORMAT,
        "shards": {"location": SHARDS_LOCATION, "path": str(SF.resolve_location(SHARDS_LOCATION, str(clean_json))),
                   "repo_id": V.REPO_ID, "revision": V.REVISION, "manifest_sha256": V.MANIFEST_SHA256,
                   "classes_py_sha256": source.get("classes_py_sha256")},
        "clean_test": {"rows": clean_test.get("rows"), "n": clean_test.get("n"), "rows_sha256": rows_sha,
                       "index": str(clean_json), "index_file_sha256": file_sha256(clean_json)},
        "condition_test": {"sha256": expected, "n": condition_test.get("n"), "corruption": recipe["corruption"],
                           "severity": recipe.get("severity"), "source": recipe.get("source"), "rng": recipe.get("rng"),
                           "index": str(condition_json), "index_file_sha256": file_sha256(condition_json)},
    }


def condition_recipe(condition: str, meta_root: Path | None = None) -> tuple[str, int]:
    """(corruption, severity) of an in1k_t04 condition name, checked against its dataset.json when one is given."""
    corruption, _, severity = condition.rpartition("_s")
    if corruption not in T.CORRUPTIONS or not severity.isdigit() or int(severity) not in T.SEVERITIES:
        raise SF.SealedFinalsError(f"{condition!r} is not <corruption>_s<1..5> of {list(T.CORRUPTIONS)}")
    if meta_root is not None:
        recipe = _meta(condition, meta_root)[0].get("condition") or {}
        if (recipe.get("corruption"), recipe.get("severity")) != (corruption, int(severity)):
            raise SF.SealedFinalsError(f"{condition}: dataset.json records {recipe.get('corruption')} "
                                       f"s{recipe.get('severity')}")
    return corruption, int(severity)


# ----------------------------------------------------------------------------------------------------------- the kit


def check_kit(kit: Path) -> dict[str, Any]:
    """Every file ``kit.json`` lists, by sha256; the official file and the shard manifest by their pinned hashes."""
    kit = Path(kit)
    record_path = kit / KIT_FILE
    if not record_path.is_file():
        raise SF.SealedFinalsError(f"no {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema") != KIT_SCHEMA or not isinstance(record.get("files"), dict) or not record["files"]:
        raise SF.SealedFinalsError(f"{record_path}: not a {KIT_SCHEMA} record with files")
    bad = []
    for relative, digest in sorted(record["files"].items()):
        path = SF._inside(kit, kit / relative, "kit file")  # noqa: SLF001, the pipeline's own containment check
        if not path.is_file() or file_sha256(path) != digest:
            bad.append(relative)
    if bad:
        raise SF.SealedFinalsError(f"{kit}: files missing or not as kit.json records: {bad}")
    for relative, digest in (("make_imagenet_c.py", T.OFFICIAL_SHA256), (V.MANIFEST, V.MANIFEST_SHA256)):
        if record["files"].get(relative) != digest:
            raise SF.SealedFinalsError(f"{kit}: {relative} is not the pinned file {digest[:16]}...")
    return {"path": str(kit), "kit_json_sha256": file_sha256(record_path), "n_files": len(record["files"]),
            "source": record.get("source")}


def kit_meta_root(kit: Path) -> Path:
    return Path(kit) / T.DATASET


KIT_REQUIRED = ("make_imagenet_c.py", V.MANIFEST, f"{T.DATASET}/dataset.json")


def write_kit(kit: Path, source: str) -> dict[str, Any]:
    """Write ``kit.json`` (every file under the kit with its sha256; never overwritten) for a kit pulled from NSCC by
    ``scripts/sealed/t04/pull_kit.sh``, then check it (``check_kit``)."""
    kit = Path(kit)
    record_path = kit / KIT_FILE
    files = {str(path.relative_to(kit)): file_sha256(path) for path in sorted(kit.rglob("*"))
             if path.is_file() and path != record_path}
    missing = [name for name in KIT_REQUIRED if name not in files]
    if missing or not any(name.startswith("frozen/") for name in files):
        raise SF.SealedFinalsError(f"{kit}: no {missing or 'frozen networks'}; nothing written")
    with open(record_path, "x", encoding="utf-8") as handle:
        json.dump({"schema": KIT_SCHEMA, "time": utc_now(), "source": source, "files": files}, handle, indent=1,
                  sort_keys=True)
    return check_kit(kit)


def install_instances(kit: Path) -> list[dict[str, Any]]:
    """Copy the kit's instances (``instances/<id>``) to ``$WEIGHTCODE_PRIVATE/instances/<id>``. An instance already there is
    left as it is when every file is identical and refused otherwise; the installed copy loads and its base hash is the
    one its private.json records."""
    from weightcode.baselines.instance_io import load_instance
    from weightcode.evalkit.instances import instances_root

    check_kit(kit)
    done = []
    for source in sorted((Path(kit) / "instances").iterdir()):
        if not SF.INSTANCE_PATTERNS[READOUT].fullmatch(source.name):
            raise SF.SealedFinalsError(f"{source}: not an instance directory")
        target = instances_root() / source.name
        ours = {p.name: file_sha256(p) for p in sorted(source.iterdir()) if p.is_file()}
        if target.exists():
            theirs = {p.name: file_sha256(p) for p in sorted(target.iterdir()) if p.is_file()}
            if theirs != ours:
                raise SF.SealedFinalsError(f"{target} exists and differs from the kit's copy; nothing installed")
            state = "present, identical"
        else:
            temporary = target.with_name(f".{target.name}.installing")
            shutil.copytree(source, temporary)
            temporary.rename(target)
            state = "installed"
        instance = load_instance(target)
        if instance.private.get("dataset") != T.DATASET:
            raise SF.SealedFinalsError(f"{target}: not an {T.DATASET} instance")
        done.append({"instance_id": source.name, "state": state, "params_sha256": instance.params_sha256,
                     "condition": instance.private.get("condition"), "files": ours})
    return done


# ------------------------------------------------------------------------------------------------------ the fidelity


def _labels_of_dev_rows(data_dir: Path) -> dict[int, int]:
    return {record.global_index: record.label for record in T.read_dev_records(data_dir, T.dev_rows())}


def fidelity(kit: Path, shards: Path, conditions: Sequence[str], out: Path | None = None) -> dict[str, Any]:
    """The fidelity check (module docstring); dev rows only. Writes its record to ``out`` (never overwritten) and
    raises ``SealedFinalsError`` on any mismatch, after writing the record."""
    started = time.time()
    kit, shards = Path(kit), Path(shards)
    kit_record = check_kit(kit)
    meta_root = kit_meta_root(kit)
    clean_meta = _meta(None, meta_root)[0]
    problems: list[str] = []
    try:
        listed = V.verify_manifest(shards, V.MANIFEST_SHA256)
        manifest = {"files": len(listed), "match": True}
    except (OSError, ValueError) as error:
        raise SF.SealedFinalsError(f"{shards}: the shards do not match the manifest ({error})") from error
    data_dir = shards / T.SHARD_DIR
    labels = _labels_of_dev_rows(data_dir)
    splits = T.split_rows(labels)
    rows_check = {}
    for split, rows in splits.items():
        ours, theirs = T.rows_sha256(rows), clean_meta["splits"][split]["rows_sha256"]
        rows_check[split] = {"ours": ours, "recorded": theirs, "match": ours == theirs}
        if ours != theirs:
            problems.append(f"{split} rows")
    rows = splits["V_check"]
    records = {record.global_index: record for record in T.read_dev_records(data_dir, rows)}
    x = np.stack([T.center_crop_u8(records[int(row)].image_bytes) for row in rows])
    clean = {"x": x, "y": np.array([records[int(row)].label for row in rows], dtype=np.int64),
             "idx": rows.astype(np.int64)}
    del records
    content = {}
    ours, theirs = store.split_sha256(clean), clean_meta["splits"]["V_check"]["sha256"]
    content[CLEAN] = {"ours": ours, "recorded": theirs, "match": ours == theirs}
    if ours != theirs:
        problems.append("clean V_check content")
    official = T.load_official(kit / "make_imagenet_c.py", T.OFFICIAL_SHA256)
    constants = T.check_official_constants(official)
    for condition in conditions:
        corruption, severity = condition_recipe(condition, meta_root)
        arrays = {"x": T.corrupt_split(official, corruption, severity, clean["x"], rows), "y": clean["y"],
                  "idx": clean["idx"]}
        ours = store.split_sha256(arrays)
        theirs = _meta(condition, meta_root)[0]["splits"]["V_check"]["sha256"]
        content[condition] = {"ours": ours, "recorded": theirs, "match": ours == theirs}
        if ours != theirs:
            problems.append(f"{condition} V_check content")
    record = {"schema": FIDELITY_SCHEMA, "time": utc_now(), "kit": kit_record, "shards": str(shards),
              "manifest": manifest, "dev_rows": len(labels), "split_rows": rows_check, "v_check_content": content,
              "conditions": list(conditions), "official_constants": constants,
              "libraries": T.image_libraries(), "passed": not problems,
              "problems": problems, "wall_s": time.time() - started}
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=1, sort_keys=True)
    if problems:
        raise SF.SealedFinalsError(f"fidelity check failed: {problems}; no test row was read")
    return record


# ---------------------------------------------------------------------------------------------------------- the plan


def secondary_rows(test_rows: np.ndarray, modulus: int = T.ROW_MODULUS,
                   residues: Sequence[int] = SECONDARY_RESIDUES) -> np.ndarray:
    return test_rows[np.isin(test_rows % modulus, residues)]


def entries_sha256(parsed: Mapping[str, Any]) -> str:
    """sha256 of a list's entries alone (canonical JSON of its finals and references): the approval, which changes
    the list's own sha256, leaves it unchanged, so a plan written for the draft holds for the approved list."""
    return manifest_sha256({"finals": parsed["finals"], "references": parsed["references"]})


def secondary_sets(condition: str) -> list[str]:
    """PREREG T04: the target corruption at severities 1 to 5 and the other corruption at its chosen severity."""
    corruption, _ = condition_recipe(condition)
    (other,) = [c for c in T.CORRUPTIONS if c != corruption]
    chosen = {condition_recipe(c)[0]: c for c in SECONDARY_CONDITIONS.values()}
    return [f"{corruption}_s{severity}" for severity in T.SEVERITIES] + [chosen[other]]


def make_secondary_plan(list_path: Path) -> dict[str, Any]:
    """The secondary plan of a T04 list: per condition, the base, both LEARN finals and the main control of record
    (PREREG T04, "for the base, the LEARN finals and the main control only"), found by their list names
    ``T04_<tag>_base``, ``T04_<tag>_LEARN_r1``/``_r2`` and ``T04_<tag>_main_<method>``; refused unless each is
    there exactly once on its condition's instance."""
    parsed, entries = SF.load_list(list_path)
    groups = []
    for tag, condition in SECONDARY_CONDITIONS.items():
        mine = [e for e in entries if e.name.startswith(f"T04_{tag}_")]
        instances = {e.instance_id for e in mine}
        if len(instances) != 1:
            raise SF.SealedFinalsError(f"{list_path}: the {tag} entries are on {sorted(instances)}, not one instance")
        names = []
        for role, test in (("base", lambda n, t=tag: n == f"T04_{t}_base"),
                           ("LEARN_r1", lambda n, t=tag: n == f"T04_{t}_LEARN_r1"),
                           ("LEARN_r2", lambda n, t=tag: n == f"T04_{t}_LEARN_r2"),
                           ("main", lambda n, t=tag: n.startswith(f"T04_{t}_main_"))):
            found = [e.name for e in mine if test(e.name)]
            if len(found) != 1:
                raise SF.SealedFinalsError(f"{list_path}: {len(found)} entries hold the {tag} {role} role")
            names.append(found[0])
        groups.append({"tag": tag, "condition": condition, "instance_id": instances.pop(), "entries": names,
                       "sets": secondary_sets(condition)})
    return {"schema": PLAN_SCHEMA, "list": str(list_path), "entries_sha256": entries_sha256(parsed),
            "rows": f"test rows with row % {T.ROW_MODULUS} in {list(SECONDARY_RESIDUES)}",
            "rule": "PREREG T04 sealed test, secondary (descriptive): the target corruption at severities 1 to 5 and "
                    "the other corruption at its chosen severity, for the base, the LEARN finals and the main control",
            "groups": groups}


def load_secondary_plan(path: Path, parsed: Mapping[str, Any], entries: Sequence[SF.Final],
                        instances: Mapping[str, Instance]) -> dict[str, Any]:
    """The committed secondary plan of a list, checked against the list's entries and their instances."""
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA or plan.get("entries_sha256") != entries_sha256(parsed):
        raise SF.SealedFinalsError(f"{path}: not a {PLAN_SCHEMA} plan of these entries")
    by_name = {entry.name: entry for entry in entries}
    for group in plan["groups"]:
        instance = instances.get(group["instance_id"])
        if instance is None or str(instance.private.get("condition")) != group["condition"]:
            raise SF.SealedFinalsError(f"{path}: {group['instance_id']} is not an instance of the list on "
                                       f"{group['condition']}")
        for name in group["entries"]:
            entry = by_name.get(name)
            if entry is None or entry.instance_id != group["instance_id"]:
                raise SF.SealedFinalsError(f"{path}: {name} is not an entry of the list on {group['instance_id']}")
        if group["sets"] != secondary_sets(group["condition"]):
            raise SF.SealedFinalsError(f"{path}: the sets of {group['condition']} are not PREREG T04's")
    return plan


def build_plan(entries: Sequence[SF.Final], instances: Mapping[str, Instance], children: Mapping[str, Path],
               secondary: Mapping[str, Any] | None, kit: Path, shards: Path, batch_size: int,
               sha: str, registry: Path, out_dir: Path) -> dict[str, Any]:
    """What every unit reads and forwards (JSON-safe, so a spawned worker needs nothing else)."""
    forwards = {(i.forward_sha256, i.params_sha256) for i in instances.values()}
    if len(forwards) != 1:
        raise SF.SealedFinalsError(f"the list's instances do not share one base and forward.py: {sorted(forwards)}")
    first = next(iter(instances.values()))
    base_key = first.params_sha256
    networks = {base_key: {"path": str(first.instance_dir / "model.npz"), "params_sha256": base_key}}
    primary: dict[str, list[str]] = {}
    clean: list[str] = [base_key]
    for entry in entries:
        condition = str(instances[entry.instance_id].private["condition"])
        keys = primary.setdefault(condition, [base_key])
        if entry.child_sha256 != base_key:
            networks.setdefault(entry.child_sha256, {"path": str(children[entry.name]),
                                                     "params_sha256": entry.child_sha256})
            for group in (keys, clean):
                if entry.child_sha256 not in group:
                    group.append(entry.child_sha256)
    second: dict[str, list[str]] = {}
    for group in (secondary or {}).get("groups", []):
        keys = [base_key] + [e.child_sha256 for e in entries if e.name in group["entries"]]
        for set_name in group["sets"]:
            bucket = second.setdefault(set_name, [])
            bucket.extend(key for key in dict.fromkeys(keys) if key not in bucket)
    return {
        "sha": sha, "registry": str(registry), "out_dir": str(out_dir), "shards": str(shards),
        "official": str(Path(kit) / "make_imagenet_c.py"), "official_sha256": T.OFFICIAL_SHA256,
        "forward": str(first.instance_dir / "forward.py"), "forward_sha256": first.forward_sha256,
        "batch_size": int(batch_size), "networks": networks, "clean": clean, "primary": primary,
        "secondary": second, "n_val_rows": V.N_VAL_ROWS, "row_modulus": T.ROW_MODULUS,
        "dev_residues": list(T.DEV_RESIDUES), "secondary_residues": list(SECONDARY_RESIDUES),
        "shard_names": V.shard_names(),
    }


# --------------------------------------------------------------------------------------------------------- the units


def _plan_test_rows(plan: Mapping[str, Any]) -> np.ndarray:
    rows = np.arange(int(plan["n_val_rows"]), dtype=np.int64)
    return rows[~np.isin(rows % int(plan["row_modulus"]), plan["dev_residues"])]


def unit_rows(plan: Mapping[str, Any], unit: int, row_counts: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """(global rows, local rows) of the test rows in shard ``unit``, in row order."""
    counts = [int(c) for c in row_counts]
    if sum(counts) != int(plan["n_val_rows"]) or len(counts) != len(plan["shard_names"]):
        raise SF.SealedFinalsError(f"the shards hold {sum(counts)} rows in {len(counts)} files, expected "
                                   f"{plan['n_val_rows']} in {len(plan['shard_names'])}")
    start = int(sum(counts[:unit]))
    rows = _plan_test_rows(plan)
    mine = rows[(rows >= start) & (rows < start + counts[unit])]
    return mine, mine - start


def _registry_open(plan: Mapping[str, Any]) -> None:
    """Refuse unless the registry holds this read's evaluation_started and nothing that closes it."""
    events = [r["event"] for r in read_registry(Path(plan["registry"])) if r.get("manifest_sha256") == plan["sha"]]
    if "evaluation_started" not in events or "evaluation_finished" in events or "evaluation_failed" in events:
        raise SF.SealedFinalsError(f"the registry records {events[:3]}... for {plan['sha']}; no shard is opened")


def read_test_records(data_dir: Path, shard: str, local_rows: np.ndarray, global_rows: np.ndarray,
                      plan: Mapping[str, Any]) -> tuple[list[bytes], np.ndarray]:
    """Stored bytes and labels of the given test rows of one shard; refuses a dev row and a shard outside
    ``$WEIGHTCODE_SEALED`` (the shards live there only on the workstation; nothing sealed exists on NSCC)."""
    import pyarrow.parquet as pq

    root = sealed_dir().resolve()
    path = (Path(data_dir) / shard).resolve()
    if root not in path.parents:
        raise SF.SealedFinalsError(f"{path} is not under $WEIGHTCODE_SEALED; test rows are read only from there")
    modulus, dev = int(plan["row_modulus"]), set(plan["dev_residues"])
    if any(int(r) % modulus in dev for r in global_rows):
        raise SF.SealedFinalsError("a dev row was asked for as a test row")
    if len(local_rows) != len(global_rows):
        raise SF.SealedFinalsError("one shard row per global row")
    if len(local_rows) == 0:
        return [], np.empty(0, dtype=np.int64)
    table = pq.ParquetFile(path).read(columns=["image", "label"])
    picked = table.take([int(r) for r in local_rows])
    images, labels = picked.column("image").to_pylist(), picked.column("label").to_pylist()
    del table, picked
    data = []
    for row, image, label in zip(global_rows, images, labels, strict=True):
        raw = image["bytes"] if isinstance(image, dict) else None
        if not isinstance(raw, (bytes, bytearray)) or not raw or not isinstance(label, int) \
                or not 0 <= label < T.N_CLASSES:
            raise SF.SealedFinalsError(f"{shard} row {row}: no stored image bytes, or label {label!r} outside "
                                       f"0 .. {T.N_CLASSES - 1}")
        data.append(bytes(raw))
    return data, np.asarray(labels, dtype=np.int64)


def _predict(forward: Callable[..., Any], params: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    logits = np.asarray(forward(dict(params), x))
    if logits.ndim != 2 or logits.shape[0] != len(x) or not np.all(np.isfinite(logits)):
        raise SF.SealedFinalsError(f"forward returned logits {logits.shape} for {len(x)} rows, or non-finite values")
    return logits.argmax(axis=1).astype(PRED_DTYPE)


def read_unit(plan: Mapping[str, Any], unit: int) -> dict[str, Any]:
    """One shard's test rows: crop, corrupt, forward, and write the predictions (module docstring)."""
    _registry_open(plan)
    started = time.time()
    forward, forward_sha = load_forward(Path(plan["forward"]))
    if forward_sha != plan["forward_sha256"]:
        raise SF.SealedFinalsError(f"{plan['forward']}: sha256 {forward_sha}, expected {plan['forward_sha256']}")
    params = {}
    for key, net in plan["networks"].items():
        params[key] = load_params(Path(net["path"]))
        if params_sha256(params[key]) != key:
            raise SF.SealedFinalsError(f"{net['path']}: params_sha256 is not {key}")
    official = T.load_official(Path(plan["official"]), plan["official_sha256"])
    data_dir = Path(plan["shards"]) / T.SHARD_DIR
    names = list(plan["shard_names"])
    counts = V.shard_row_counts(data_dir, names)
    rows, local = unit_rows(plan, unit, counts)
    images, y = read_test_records(data_dir, names[unit], local, rows, plan)
    in_secondary = np.isin(rows % int(plan["row_modulus"]), plan["secondary_residues"])
    full_sets = [CLEAN, *plan["primary"]]
    full_nets = {CLEAN: plan["clean"], **plan["primary"]}
    preds: dict[str, list[np.ndarray]] = {f"p__{s}__{k}": [] for s in full_sets for k in full_nets[s]}
    preds.update({f"s__{s}__{k}": [] for s, keys in plan["secondary"].items() for k in keys})
    batch = int(plan["batch_size"])
    for start in range(0, len(rows), batch):
        part = slice(start, start + batch)
        crops = np.stack([T.center_crop_u8(raw) for raw in images[part]])
        rows_b, sec_b = rows[part], in_secondary[part]
        sets = {CLEAN: crops}
        for condition in plan["primary"]:
            corruption, severity = condition_recipe(condition)
            sets[condition] = T.corrupt_split(official, corruption, severity, crops, rows_b)
        for s in full_sets:
            for key in full_nets[s]:
                preds[f"p__{s}__{key}"].append(_predict(forward, params[key], sets[s]))
        if sec_b.any():
            for s, keys in plan["secondary"].items():
                done = full_nets.get(s, [])
                if s in sets:
                    x_sec = sets[s][sec_b]
                else:
                    corruption, severity = condition_recipe(s)
                    x_sec = T.corrupt_split(official, corruption, severity, crops[sec_b], rows_b[sec_b])
                for key in keys:
                    if key in done:
                        preds[f"s__{s}__{key}"].append(preds[f"p__{s}__{key}"][-1][sec_b])
                    else:
                        preds[f"s__{s}__{key}"].append(_predict(forward, params[key], x_sec))
        del crops, sets
    arrays = {key: (np.concatenate(parts) if parts else np.empty(0, PRED_DTYPE)) for key, parts in preds.items()}
    arrays.update(idx=rows, y=y, idx_secondary=rows[in_secondary], y_secondary=y[in_secondary])
    out = Path(plan["out_dir"]) / f"unit_{unit:02d}.npz"
    temporary = out.with_name(out.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(out)
    return {"unit": unit, "path": str(out), "sha256": file_sha256(out), "n": int(len(rows)),
            "n_secondary": int(in_secondary.sum()), "wall_s": time.time() - started}


# ---------------------------------------------------------------------------------------------------- the sealed read


def _dispatch_units(plan: Mapping[str, Any], units: Sequence[int], workers: int,
                    on_finished: Callable[[dict[str, Any]], None]) -> None:
    """Read the units, ``workers`` at a time in spawned processes (in this process when there is one worker); the first
    failure stops the read (the error is re-raised after the running units end)."""
    if workers == 1:
        for unit in units:
            on_finished(read_unit(plan, unit))
        return
    context = multiprocessing.get_context("spawn")
    pending = list(units)
    failure: BaseException | None = None
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        running: dict[Future[dict[str, Any]], int] = {}
        while (pending and failure is None) or running:
            while pending and failure is None and len(running) < workers:
                unit = pending.pop(0)
                running[pool.submit(read_unit, dict(plan), unit)] = unit
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                unit = running.pop(future)
                try:
                    on_finished(future.result())
                except BaseException as error:  # noqa: BLE001, the read stops and the error is re-raised below
                    failure = failure or SF.SealedFinalsError(f"unit {unit}: {type(error).__name__}: {error}")
    if failure is not None:
        raise failure


def _load_unit(path: Path, digest: str) -> dict[str, np.ndarray]:
    if file_sha256(path) != digest:
        raise SF.SealedFinalsError(f"{path}: sha256 is not the one its unit_finished recorded")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def assemble(plan: Mapping[str, Any], unit_records: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """All units' predictions in row order, checked against the test rows and the secondary rows."""
    units = [_load_unit(Path(unit_records[k]["path"]), unit_records[k]["sha256"])
             for k in range(len(plan["shard_names"]))]
    joined = {key: np.concatenate([u[key] for u in units]) for key in units[0]}
    rows = _plan_test_rows(plan)
    if not np.array_equal(joined["idx"], rows):
        raise SF.SealedFinalsError("the units' rows are not the test rows in order")
    expected_secondary = rows[np.isin(rows % int(plan["row_modulus"]), plan["secondary_residues"])]
    if not np.array_equal(joined["idx_secondary"], expected_secondary):
        raise SF.SealedFinalsError("the units' secondary rows are not the secondary test rows in order")
    return joined


def entry_record(entry: SF.Final, index: int, instance: Instance, joined: Mapping[str, np.ndarray],
                 plan: Mapping[str, Any], secondary_sets: Sequence[str], provenance: Mapping[str, Any],
                 sealed_index: Mapping[str, Any], n_classes: int) -> dict[str, Any]:
    """One entry's record from the assembled predictions (module docstring)."""
    condition = str(instance.private["condition"])
    base, child = instance.params_sha256, entry.child_sha256
    y, y_sec = joined["y"], joined["y_secondary"]
    cond = readout_parts({"y": y}, joined[f"p__{condition}__{base}"], joined[f"p__{condition}__{child}"], n_classes)
    clean = readout_parts({"y": y}, joined[f"p__{CLEAN}__{base}"], joined[f"p__{CLEAN}__{child}"], n_classes)
    secondary = {s: readout_parts({"y": y_sec}, joined[f"s__{s}__{base}"], joined[f"s__{s}__{child}"], n_classes)
                 for s in secondary_sets}
    gain, clean_points = SF.points(cond["readouts"]["delta_acc"]), SF.points(clean["readouts"]["delta_acc"])
    rows_sha = T.rows_sha256(joined["idx"])
    return {
        "schema": SF.RESULT_SCHEMA, "mode": SF.MODE_SEALED, "block": entry.block, **entry.as_dict(),
        "list_index": index, "dataset": instance.private.get("dataset"), "condition": condition,
        "split_role": instance.private.get("split_role"), "seed": instance.private.get("seed"),
        "base_params_sha256": base, "forward_sha256": instance.forward_sha256, "child": dict(provenance),
        "rows": {"condition": {"split": "sealed test", "n": int(len(y)), "rows_sha256": rows_sha,
                               "corruption": dict(zip(("name", "severity"), condition_recipe(condition)))},
                 "clean": {"split": "sealed clean test", "n": int(len(y)), "rows_sha256": rows_sha},
                 "secondary": {"rows": f"test rows with row % {plan['row_modulus']} in {plan['secondary_residues']}",
                               "n": int(len(y_sec)), "rows_sha256": T.rows_sha256(joined["idx_secondary"])}},
        "sealed_index": dict(sealed_index), "batch_size": plan["batch_size"],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, **FAMILY.environment()},
        "readouts": cond["readouts"], "readouts_clean": clean["readouts"],
        "per_image": cond["per_image"], "per_image_clean": clean["per_image"],
        "secondary": {s: {"readouts": part["readouts"], "per_image": part["per_image"],
                          "gain": SF.points(part["readouts"]["delta_acc"])} for s, part in secondary.items()},
        "gain": gain, "clean": clean_points, "joint": gain + clean_points,
    }


def _frozen_children(kit: Path, entries: Sequence[SF.Final]) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    """Each entry's exported params file in the kit and the export record line that made it, hash-checked."""
    lines: dict[str, dict[str, Any]] = {}
    for path in sorted((Path(kit) / "frozen").glob("export_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for row in record["entries"]:
            if row["name"] in lines:
                raise SF.SealedFinalsError(f"{row['name']} is exported twice in the kit")
            lines[row["name"]] = {**row, "export_record": path.name, "export_record_sha256": file_sha256(path),
                                  "list_sha256": record.get("list_sha256"), "code_commit": record["code"]["commit"]}
    paths, provenance = {}, {}
    for entry in entries:
        row = lines.get(entry.name)
        path = Path(kit) / "frozen" / f"{entry.name}.npz"
        if row is None or row["params_sha256"] != entry.child_sha256 or row["instance_id"] != entry.instance_id:
            raise SF.SealedFinalsError(f"{entry.name}: no export of this child on {entry.instance_id} in the kit")
        if not path.is_file() or file_sha256(path) != row["file_sha256"]:
            raise SF.SealedFinalsError(f"{path}: missing or not the exported file")
        if params_sha256(load_params(path)) != entry.child_sha256:
            raise SF.SealedFinalsError(f"{path}: params_sha256 is not {entry.child_sha256}")
        paths[entry.name] = path
        provenance[entry.name] = {"kind": "frozen_export", "path": str(path), "file_sha256": row["file_sha256"],
                                  "export_record": row["export_record"],
                                  "export_record_sha256": row["export_record_sha256"],
                                  "export_code_commit": row["code_commit"], "recovered_on_nscc": row["child"]}
    return paths, provenance


def _closing(events: Iterable[Mapping[str, Any]]) -> bool:
    return any(e["event"] in ("evaluation_finished", "evaluation_failed") for e in events)


def run_sealed(list_path: Path, plan_path: Path, kit: Path, batch_size: int = DEFAULT_BATCH,
               workers: int = DEFAULT_WORKERS, resume: bool = False) -> tuple[Path, int]:
    """The one sealed read of an approved list of ``vision_t04`` entries (module docstring)."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1 or batch_size < 1:
        raise SF.SealedFinalsError("workers and batch size must be integers >= 1")
    parsed, entries = SF.load_list(list_path)
    if parsed["status"] != SF.STATUS_APPROVED:
        raise SF.SealedFinalsError(f"{list_path}: status {parsed['status']!r}; only an approved list is read")
    if any(entry.readout != READOUT for entry in entries):
        raise SF.SealedFinalsError(f"{list_path}: every entry of an in1k_t04 read is {READOUT}")
    sha = manifest_sha256(parsed)
    state = SF.code_state(modules=SF.list_modules(entries))
    if state["dirty"]:
        raise SF.SealedFinalsError(f"{state['repo']}: the tree is not clean; the code is frozen by commit before the read")
    list_relative = SF._check_list_committed(list_path, state)  # noqa: SLF001
    plan_relative = SF._check_list_committed(plan_path, state)  # noqa: SLF001
    kit_record = check_kit(kit)
    instances = {e.instance_id: FAMILY.load_instance(e.instance_id) for e in entries}
    for instance in instances.values():
        kit_copy = Path(kit) / "instances" / instance.instance_id
        ours = {p.name: file_sha256(p) for p in sorted(instance.instance_dir.iterdir()) if p.is_file()}
        theirs = {p.name: file_sha256(p) for p in sorted(kit_copy.iterdir()) if p.is_file()} \
            if kit_copy.is_dir() else None
        if ours != theirs:
            raise SF.SealedFinalsError(f"{instance.instance_dir} is not the kit's copy of {instance.instance_id}")
    secondary = load_secondary_plan(plan_path, parsed, entries, instances)
    for instance in instances.values():
        if instance.private.get("split_role") != "dev":
            raise SF.SealedFinalsError(f"{instance.instance_id}: the entries of record are on dev instances")
    children, provenance = _frozen_children(kit, entries)
    meta_root = kit_meta_root(kit)
    indexes = {c: t04_sealed_index(c, meta_root) for c in {str(i.private["condition"]) for i in instances.values()}}
    shards = Path(next(iter(indexes.values()))["shards"]["path"])
    root = sealed_dir()
    registry = root / REGISTRY_NAME
    out_dir = root / T.DATASET / "reads" / sha
    plan = build_plan(entries, instances, children, secondary, kit, shards, batch_size, sha, registry, out_dir)
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    fidelity_out = runs_dir() / "t04" / "sealed_fidelity" / f"fidelity_{sha[:16]}_{time.time_ns()}.json"
    fidelity_passed = fidelity(kit, shards, sorted(indexes), fidelity_out)["passed"]
    secondary_sets = {name: group["sets"] for group in secondary["groups"] for name in group["entries"]}
    unit_records: dict[int, dict[str, Any]] = {}
    with _exclusive_lock(root / ".sealed.lock"):
        history = read_registry(registry)
        mine = [r for r in history if r.get("manifest_sha256") == sha]
        if resume:
            started = [r for r in mine if r["event"] == "evaluation_started"]
            if len(started) != 1 or _closing(mine):
                raise SF.SealedFinalsError(f"--resume needs one open evaluation_started of {sha}; the registry has "
                                           f"{[r['event'] for r in mine]}")
            same = {"code_commit": state["commit"], "plan_sha256": plan_sha,
                    "kit_json_sha256": kit_record["kit_json_sha256"]}
            differ = [k for k, v in same.items() if started[0].get(k) != v]
            if differ:
                raise SF.SealedFinalsError(f"--resume: {differ} differ from the started read; nothing is read")
            for record in mine:
                if record["event"] == UNIT_FINISHED and Path(record["path"]).is_file() \
                        and file_sha256(Path(record["path"])) == record["sha256"]:
                    unit_records[int(record["unit"])] = record
            _append_registry(registry, {"event": EVALUATION_RESUMED, "kind": EVENT_KIND, "manifest_sha256": sha,
                                        "time": utc_now(), "units_kept": sorted(unit_records)})
        else:
            if mine:
                raise SF.SealedFinalsError(f"list {sha} was already read or attempted (see {registry})")
            again = sorted({(e.instance_id, e.child_sha256) for e in entries} & SF._read_before(history))  # noqa: SLF001
            if again:
                raise SF.SealedFinalsError(f"entries already read in an earlier sealed read: {again}")
            if out_dir.exists():
                raise SF.SealedFinalsError(f"{out_dir} exists although the registry has no record of {sha}")
            (root / "manifests").mkdir(mode=0o700, exist_ok=True)
            shutil.copyfile(list_path, root / "manifests" / f"{sha}.json")
            _append_registry(registry, {
                "event": "evaluation_started", "kind": EVENT_KIND, "manifest_sha256": sha, "time": utc_now(),
                "n_entries": len(entries), "manifest_copy": str(root / "manifests" / f"{sha}.json"),
                "list_path": list_relative, "secondary_plan": plan_relative, "approval": parsed["approval"],
                "code_commit": state["commit"], "plan_sha256": plan_sha, "kit_json_sha256": kit_record["kit_json_sha256"],
                "fidelity_record": str(fidelity_out), "fidelity_sha256": file_sha256(fidelity_out),
                "fidelity_passed": fidelity_passed,
                "workers": workers, "batch_size": batch_size,
                "threads": {name: os.environ.get(name) for name in SF.THREAD_VARIABLES},
                "entries": [{"name": e.name, "block": e.block, "instance_id": e.instance_id,
                             "child_sha256": e.child_sha256, "source": dict(e.source), "readout": e.readout}
                            for e in entries],
            })
            for index, entry in enumerate(entries):
                _append_registry(registry, {"event": SF.EVENT_ENTRY_STARTED, "kind": EVENT_KIND,
                                            "manifest_sha256": sha, "time": utc_now(), "name": entry.name,
                                            "list_index": index, "instance_id": entry.instance_id,
                                            "child_sha256": entry.child_sha256})
        out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        results_dir = root / "results"
        results_dir.mkdir(mode=0o700, exist_ok=True)
        results_path = results_dir / f"{sha}.jsonl"

        def finished(record: dict[str, Any]) -> None:
            unit_records[int(record["unit"])] = record
            _append_registry(registry, {"event": UNIT_FINISHED, "kind": EVENT_KIND, "manifest_sha256": sha,
                                        "time": utc_now(), **record})

        try:
            todo = [k for k in range(len(plan["shard_names"])) if k not in unit_records]
            _dispatch_units(plan, todo, min(workers, max(1, len(todo))), finished)
            joined = assemble(plan, unit_records)
            n_classes = len(next(iter(instances.values())).task["classes"])
            if results_path.exists():
                raise SF.SealedFinalsError(f"{results_path} exists")
            lines = []
            for index, entry in enumerate(entries):
                instance = instances[entry.instance_id]
                record = entry_record(entry, index, instance, joined, plan, secondary_sets.get(entry.name, []),
                                      provenance[entry.name], indexes[str(instance.private["condition"])], n_classes)
                record.update(list_sha256=sha, list_path=list_relative, list_status=parsed["status"], code=state,
                              kit=kit_record, fidelity_record=str(fidelity_out), time=utc_now())
                lines.append(json.dumps(record, sort_keys=True, allow_nan=False))
            with open(results_path, "x", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            for index, (entry, line) in enumerate(zip(entries, lines)):
                _append_registry(registry, {"event": SF.EVENT_ENTRY_FINISHED, "kind": EVENT_KIND,
                                            "manifest_sha256": sha, "time": utc_now(), "name": entry.name,
                                            "list_index": index, "instance_id": entry.instance_id,
                                            "child_sha256": entry.child_sha256,
                                            "record_sha256": hashlib.sha256(line.encode()).hexdigest()})
        except BaseException as error:
            _append_registry(registry, {"event": EVALUATION_INTERRUPTED, "kind": EVENT_KIND, "manifest_sha256": sha,
                                        "time": utc_now(), "error": f"{type(error).__name__}: {error}",
                                        "units_finished": sorted(unit_records)})
            raise
        _append_registry(registry, {"event": "evaluation_finished", "kind": EVENT_KIND, "manifest_sha256": sha,
                                    "time": utc_now(), "n_entries": len(entries), "n_read": len(entries),
                                    "failed": {}, "results_path": str(results_path),
                                    "results_sha256": file_sha256(results_path),
                                    "units": {str(k): v["sha256"] for k, v in sorted(unit_records.items())}})
    return results_path, len(entries)


# ----------------------------------------------------------------------------------------------------------- the CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m weightcode.evalkit.sealed_t04", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fid = sub.add_parser("fidelity", help="the fidelity check on dev rows (no test row)")
    fid.add_argument("--kit", type=Path, required=True)
    fid.add_argument("--shards", type=Path, required=True)
    fid.add_argument("--conditions", nargs="+", default=["gaussian_noise_s3", "contrast_s4"])
    fid.add_argument("--out", type=Path, required=True)
    plan_parser = sub.add_parser("plan", help="write the secondary plan of a list (no data is opened)")
    plan_parser.add_argument("--list", type=Path, required=True)
    plan_parser.add_argument("--out", type=Path, required=True)
    kit_parser = sub.add_parser("kit", help="write kit.json of a pulled kit and install its instances")
    kit_parser.add_argument("--kit", type=Path, required=True)
    kit_parser.add_argument("--source", required=True, help="where the kit was pulled from (recorded)")
    sealed = sub.add_parser("sealed", help="the one sealed read of an approved list")
    sealed.add_argument("--list", type=Path, required=True)
    sealed.add_argument("--secondary", type=Path, required=True, help="the list's committed secondary plan")
    sealed.add_argument("--kit", type=Path, required=True)
    sealed.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    sealed.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    sealed.add_argument("--resume", action="store_true", help="continue a read whose process died (module docstring)")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            plan = make_secondary_plan(args.list)
            with open(args.out, "x", encoding="utf-8") as handle:
                handle.write(json.dumps(plan, indent=1) + "\n")
            print(f"{args.out}: {sum(len(g['entries']) for g in plan['groups'])} entries in {len(plan['groups'])} "
                  f"groups, entries_sha256 {plan['entries_sha256']}")
            return 0
        if args.command == "kit":
            record = write_kit(args.kit, args.source)
            for row in install_instances(args.kit):
                print(f"{row['instance_id']}: {row['state']} ({row['condition']}, base {row['params_sha256'][:16]})")
            print(f"kit {args.kit}: {record['n_files']} files, kit.json sha256 {record['kit_json_sha256']}")
            return 0
        if args.command == "fidelity":
            record = fidelity(args.kit, args.shards, args.conditions, args.out)
            print(f"fidelity passed: {args.out} ({record['wall_s']:.0f} s)")
            return 0
        results, count = run_sealed(args.list, args.secondary, args.kit, args.batch_size, args.workers, args.resume)
    except SF.SealedFinalsError as error:
        print(f"refused or failed: {error}", file=sys.stderr)
        return 2
    print(f"results: {results}")
    print(f"entries: {count}")
    return 0


FAMILY = VisionT04()

__all__ = ["FAMILY", "FORMAT", "READOUT", "SHARDS_LOCATION", "VisionT04", "assemble", "build_plan", "check_kit",
           "condition_recipe", "entries_sha256", "entry_record", "fidelity", "load_secondary_plan", "main",
           "install_instances", "make_secondary_plan", "read_unit", "run_sealed", "secondary_sets", "write_kit",
           "t04_sealed_index", "unit_rows"]


if __name__ == "__main__":
    from weightcode.evalkit import sealed_t04 as _imported

    sys.exit(_imported.main())
