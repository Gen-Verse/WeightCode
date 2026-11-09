"""Sealed read of a frozen list of finals (PREREG "Sealed read", scenario plan M6), and its dev dry run.

Usage::

    python -m weightcode.evalkit.sealed_finals --list LIST.json --dev --out FILE [--code-state GIT_STATE.json] \
        [--only NAME ...]
    python -m weightcode.evalkit.sealed_finals --list LIST.json --sealed

A list (schema ``weightcode.sealed_finals_list.v1``) has exactly the keys schema, status ("draft" or "approved"), approval
(null for a draft, a non-empty statement of who approved it and when for an approved list), note, finals and
references. A v0 list (``weightcode.sealed_finals_list.v0``, R26-sealed-dry) has no references block and only vision_joint
finals. Every entry is one (child, instance) pair, and the pipeline writes one record per entry. Names and
(instance_id, child_sha256) are unique over both blocks.

- ``finals``: the LLM finals and what their programs produce. Each has exactly name, readout, source, instance_id and
  child_sha256; its source is one of:

  - ``{"kind": "session", "session_id": "s..."}``: the session's final, rebuilt from ``$WEIGHTCODE_RUNS/sessions/<id>/
    final/patch.json`` on the instance's base under the session's recorded limits by ``devreadout.final_child``, which
    must give the recorded child hash.
  - ``{"kind": "stream_final", "record": "<dir>/<name>.json"}`` (vision_joint only): the student after the last round
    of a completed stream, a loop stream ``weightcode.harness.loop_stream.v0`` or a fine-tuning stream under the same loop
    ``weightcode.baselines.checked_reference_stream.v0`` (PREREG F4), loaded from the record's ``student_after`` file and
    hash-checked; the instance must be one of the stream's, with its base.
  - ``{"kind": "stream_round", "record": ..., "round": k}`` (vision_joint only): the same for round k (numbered from 1
    as the record numbers them) of a completed stream, before its last round, so a stream's score after each round is
    read on every instance of the stream.
  - ``{"kind": "transfer", "record": "devreadouts/learn/<id>/transfer/<name>.json"}`` (vision_joint only): a frozen
    final program run on another student (``weightcode.harness.program_transfer.v0``, PREREG C), an ok run whose child the
    limits admitted. The child is the record's ``derived_patch.json`` (beside it, in the work directory of the same
    stem) applied to the target's base under the limits the transfer recorded, checked against its child hash; the
    program file must still be the one it ran.
  - ``{"kind": "range", "set": S, "network": N, "dev_record": "t07/<run>/t07_table.json", "t07_arch": A,
    "t07_network": T}`` (readout ``vision_range`` only, and a vision_range entry has no other source): the child of
    the source N (a session source, or a kept reference source as in the references block) on the entry's instance,
    recovered and checked as N is, and read on the set S of the instance's dataset instead of its own condition
    (``weightcode.evalkit.sealed_range``; PREREG "T07 sealed read (list v5)"). Its read is identified by (child, dataset,
    set): within a list (instance_id, child_sha256, set) is unique, and a sealed read refuses it when an earlier
    started record of any kind read the same child on the same dataset and set (a vision_joint entry's set is its
    instance's condition). A range entry's set must not be its instance's own condition.
- ``references`` (optional block, possibly empty): reference picks read beside the finals for a paired comparison.
  Each has the keys of a final plus ``pairs_with`` (the name of the final on the same instance, or null in a draft
  whose final has not run yet). Its source is a stream's student (``stream_final`` or ``stream_round``, as for a
  final) or ``{"kind": "reference", "run": "baselines/.../run.json", "pick": P, ..., "dev_record":
  "devreadouts/<file>.jsonl"}``, a pick of a baseline run rebuilt by ``devreadout.final_child`` under the limits the
  run applied (the family's program limits for a constrained run) and checked against the child hash its check
  recorded. ``dev_record`` holds the pick's dev readout of record; only the dev mode reads it. The picks:

  - ``"kept"`` with ``"s_old_floor": F``: the kept pick under joint-retain-v1 at S_old >= F
    (``devreadout.retention_final``); its dev record line carries that selection.
  - ``"own"`` (vision_joint only): the run's own final, refused unless it is ``choose_final`` over the run's whole
    oracle log and not the base no-op (the pick without a retention floor); its dev record is the run's own-final line
    (``devreadout.readout_run``), whose V_check numbers must be the final's.
  - ``"candidate"`` with ``"patch_id": "q..."`` (vision_joint only): one named checked candidate, for a configuration
    fixed before the run (a recipe carried unchanged to another student); its dev record is the frontier line
    (``devreadout.readout_candidates``) of that patch with the entry's child hash.

Readout families (``readout``), each the readout of record of its experiments, on the condition's rows and the
retention rows, child against the instance's base:

- ``vision_joint`` (this module; E10, E11, E12, the E10 stream, the E11 ladder): ``devreadout.dev_readout`` on the
  condition rows and on their clean companion (``split_readouts``); gain and clean are 100 * delta_acc.
- ``scan_em`` (``weightcode.evalkit.sealed_scan``; E13): ``scanskill.scan_readout_arrays`` on the new and the old commands;
  gain and clean are its gain_points and clean_points.
- ``policy_success`` (``weightcode.evalkit.sealed_e14``; E14): greedy rollouts of base and child on the new and the old
  mission episodes, the fields of ``e14.e14_readout``; gain and clean are its gain and clean.
- ``vision_range`` (``weightcode.evalkit.sealed_range``; T07 at test level): the vision readout of record on one set's
  rows of the range index, plus per-image correctness of base and child; gain is 100 * delta_acc on the set, clean
  and joint are null (no clean companion is read).
- ``vision_t04`` (``weightcode.evalkit.sealed_t04``; T04, native ImageNet): ``vision_joint`` on in1k_t04 instances. Its
  dev mode runs here; its test rows are decoded only by that module's streaming read, and ``--sealed`` here refuses
  a list holding such an entry before anything is registered (a family with ``generic_sealed`` False).

A family's module imports its experiment's code, so a list that names a family whose code is not in the tree is
refused when the family is first needed, never read with something else. joint = gain + clean (points), null when
the family reads no clean rows.

Both modes run one path. For each entry: load the instance (the family's loader and current-instance checks),
recover the child, take the rows from the mode's row source, and call the family's readout. The record holds the
readout, gain, clean and joint, the child hash and how the child was recovered, the readout code commit and the
hashes of the code files (the common modules and the families of the list), the environment versions, and the
family's sealed index entries for the instance (read from the private dataset.json files only).

- ``--dev``: the rows are the dev rows of record (V_select of the condition and its retention rows), so a record must
  equal the dev readout of record field for field. Each entry is compared with its reference: a session's
  ``$WEIGHTCODE_RUNS/devreadouts/learn/<id>/info.json`` (the family's fields of ``final``, every field), a stream's own
  record (the round's ``readouts`` of the instance, the fields the stream keeps), a transfer's own record (its
  ``readouts`` and ``readouts_clean``, every field), a reference pick's line of its ``dev_record`` (every field).
  Nothing under ``$WEIGHTCODE_SEALED`` is opened, listed or read; only the root directory's existence is checked, as every
  dev readout does. Exit status 1 if any entry differs from its reference.
- ``--sealed``: the rows are the instance's sealed test rows, at the locations and with the hashes the dataset.json
  files record. Refused unless the list is approved (each reference then pairs with a final), the code is a clean git
  tree, and the list file is tracked in it, so the list and the readout code are frozen by commit before the read. In
  this order, as ``weightcode.evalkit.sealed``: validate the list; load every instance and child and check their hashes;
  resolve the sealed index and check that its files exist; under ``$WEIGHTCODE_SEALED/.sealed.lock`` refuse if the list's
  canonical hash, or any (instance_id, child_sha256) of the list, appears in a started record of
  ``$WEIGHTCODE_SEALED/registry.jsonl`` (the registry ``weightcode.evalkit.sealed`` keeps; each entry is read once); copy the list
  to ``manifests/<sha>.json`` and append ``evaluation_started`` (every entry of the list) before any sealed file is
  opened; then read the entries, ``--workers`` at a time (default 1), each in a worker process when there are more
  than one. Per entry the parent appends ``entry_started`` before the entry is handed to a worker, and the worker opens
  no file of the entry until it finds that record in the registry; a finished entry's record is appended to
  ``results/<sha>.jsonl`` (flushed and synced, in completion order, with its ``list_index``) and then
  ``entry_finished``; a failed entry gets ``entry_failed`` and stays spent, and the other entries are still read (a
  worker pool that breaks fails the entries whose results never arrive, and the rest go to a new pool). Only the parent writes the
  registry and the results file, under the lock, so the registry is one ordered history. At the end
  ``evaluation_finished`` (entries read and failed, the results file's sha256, the tests' hashes); exit status 1 when
  an entry failed. An error of the read itself, outside an entry, appends ``evaluation_failed`` and re-raises. The list
  and every entry of it stay spent in each case. A worker uses the BLAS threads of its environment
  (``OMP_NUM_THREADS``), so three reads at 4 threads each is ``OMP_NUM_THREADS=4 ... --sealed --workers 3``. Only the
  results path and the counts are printed; readouts never go to stdout.

Sealed test formats of vision_joint (``sealed_index``):

- ``store_npz``: a test record without ``format`` (the store layout, ``weightcode.data.store.sealed_test_path``): an .npz with
  x, y and idx whose canonical content hash is the recorded sha256. A condition's must have the clean test's y and idx.
- ``cifar10c_npy``: format "official CIFAR-10-C .npy, NHWC uint8, the official test order" (``weightcode.data.e10``): the
  official file, whose file sha256 is the recorded sha256 (the install record's), and rows "A .. B (severity S)" with
  B - A + 1 = n and A = (S - 1) n. The condition rows are x = file[A .. B] in NCHW with the clean test's y and idx,
  row i of the block being clean test row i. That alignment is positional: the official files keep the official test
  order, and the E10 fidelity record ($WEIGHTCODE_RUNS/e10/fidelity/e10_fidelity_20261104.json) found defocus_blur and
  contrast bit-identical, row for row, to the official code run on the clean test.npz x. Labels come from the clean
  test; the official labels.npy is not read (its only hash is in the sealed install record).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.identity import load_params, params_sha256
from weightcode.common.paths import runs_dir, sealed_dir
from weightcode.data import store
from weightcode.evalkit import devreadout
from weightcode.evalkit.companions import load_clean_rows
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.metrics import Candidate, choose_final
from weightcode.evalkit.sealed import REGISTRY_NAME, _append_registry, _exclusive_lock, manifest_sha256, read_registry
from weightcode.evalkit.targets import (
    DEPLOY_CONDITION,
    check_image_arrays,
    deployment_splits,
    file_sha256,
    read_condition_meta,
    read_dataset_meta,
    utc_now,
)
from weightcode.patch import Limits, PatchError, apply_patch, task_limits

LIST_SCHEMA_V0 = "weightcode.sealed_finals_list.v0"
LIST_SCHEMA = "weightcode.sealed_finals_list.v1"
RESULT_SCHEMA = "weightcode.sealed_finals_result.v1"
EVENT_KIND = "sealed_finals"
EVENT_ENTRY_STARTED, EVENT_ENTRY_FINISHED, EVENT_ENTRY_FAILED = "entry_started", "entry_finished", "entry_failed"
DEFAULT_WORKERS = 1
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
STATUS_DRAFT, STATUS_APPROVED = "draft", "approved"
BLOCK_FINALS, BLOCK_REFERENCES = "finals", "references"
READOUT_VISION_JOINT = "vision_joint"
READOUT_SCAN_EM = "scan_em"
READOUT_POLICY_SUCCESS = "policy_success"
READOUT_VISION_RANGE = "vision_range"
READOUT_VISION_T04 = "vision_t04"
FAMILY_MODULES = {
    READOUT_VISION_JOINT: "weightcode.evalkit.sealed_finals",
    READOUT_SCAN_EM: "weightcode.evalkit.sealed_scan",
    READOUT_POLICY_SUCCESS: "weightcode.evalkit.sealed_e14",
    READOUT_VISION_RANGE: "weightcode.evalkit.sealed_range",
    READOUT_VISION_T04: "weightcode.evalkit.sealed_t04",
}
"""Readout family -> the module whose ``FAMILY`` implements it (imported when first needed)."""
INSTANCE_PATTERNS = {
    READOUT_VISION_JOINT: re.compile(r"^i[0-9a-f]{8}$"),
    READOUT_SCAN_EM: re.compile(r"^n[0-9a-f]{8}$"),
    READOUT_POLICY_SUCCESS: re.compile(r"^p[0-9a-f]{8}$"),
    READOUT_VISION_RANGE: re.compile(r"^i[0-9a-f]{8}$"),
    READOUT_VISION_T04: re.compile(r"^i[0-9a-f]{8}$"),
}
READOUTS = tuple(FAMILY_MODULES)
SOURCE_SESSION, SOURCE_STREAM_FINAL, SOURCE_REFERENCE = "session", "stream_final", "reference"
SOURCE_STREAM_ROUND, SOURCE_TRANSFER = "stream_round", "transfer"
SOURCE_RANGE = "range"
RANGE_VIEW_KEY = "range"
"""Key of the private record of a range entry's instance view: the set it reads and the instance's own condition."""
PICK_KEPT, PICK_OWN, PICK_CANDIDATE = "kept", "own", "candidate"
OWN_RULE = "the run's own final: choose_final over its whole oracle log, no S_old floor"
CANDIDATE_RULE = "one named checked candidate of the run, selected by nothing in the run"
TRANSFER_SCHEMA = "weightcode.harness.program_transfer.v0"
TRANSFER_PATCH = "derived_patch.json"
STREAM_SCHEMAS = ("weightcode.harness.loop_stream.v0", "weightcode.baselines.checked_reference_stream.v0")
MODE_DEV, MODE_SEALED = "dev", "sealed"
FORMAT_STORE_NPZ = "store_npz"
FORMAT_CIFAR10C = "cifar10c_npy"
CIFAR10C_FORMAT_TEXT = "official CIFAR-10-C .npy, NHWC uint8, the official test order"
HASH_CONTENT = "canonical content hash (params_sha256 over every array)"
HASH_FILE = "sha256 of the file bytes"
SEALED_PREFIX = "$WEIGHTCODE_SEALED/"
DEFAULT_BATCH_SIZE = devreadout.DEFAULT_BATCH_SIZE
FIELDS_ALL, FIELDS_THEIRS = "all", "the stream's"

_LIST_KEYS = {LIST_SCHEMA_V0: frozenset({"schema", "status", "approval", "note", "finals"}),
              LIST_SCHEMA: frozenset({"schema", "status", "approval", "note", "finals", "references"})}
_ENTRY_KEYS = {BLOCK_FINALS: frozenset({"name", "readout", "source", "instance_id", "child_sha256"}),
               BLOCK_REFERENCES: frozenset({"name", "readout", "source", "instance_id", "child_sha256", "pairs_with"})}
_BLOCK_SOURCES = {BLOCK_FINALS: (SOURCE_SESSION, SOURCE_STREAM_FINAL, SOURCE_STREAM_ROUND, SOURCE_TRANSFER,
                                 SOURCE_RANGE),
                  BLOCK_REFERENCES: (SOURCE_REFERENCE, SOURCE_STREAM_FINAL, SOURCE_STREAM_ROUND)}
_SOURCE_KEYS = {SOURCE_SESSION: frozenset({"kind", "session_id"}), SOURCE_STREAM_FINAL: frozenset({"kind", "record"}),
                SOURCE_STREAM_ROUND: frozenset({"kind", "record", "round"}),
                SOURCE_TRANSFER: frozenset({"kind", "record"}),
                SOURCE_RANGE: frozenset({"kind", "set", "network", "dev_record", "t07_arch", "t07_network"})}
_REFERENCE_KEYS = {PICK_KEPT: frozenset({"kind", "run", "pick", "s_old_floor", "dev_record"}),
                   PICK_OWN: frozenset({"kind", "run", "pick", "dev_record"}),
                   PICK_CANDIDATE: frozenset({"kind", "run", "pick", "patch_id", "dev_record"})}
_STREAM_KINDS = (SOURCE_STREAM_FINAL, SOURCE_STREAM_ROUND)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_SESSION_RE = re.compile(r"^s[0-9a-f]{16}$")
_PATCH_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_RECORD_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*\.json$")
_RUN_RE = re.compile(r"^baselines/[A-Za-z0-9_./-]+/run\.json$")
_DEV_RECORD_RE = re.compile(r"^devreadouts/[A-Za-z0-9_./-]+\.jsonl$")
_ROWS_RE = re.compile(r"^(\d+) \.\. (\d+) \(severity (\d+)\)$")
_SET_RE = re.compile(r"^[a-z_]+_s[1-5]$")
_T07_RECORD_RE = re.compile(r"^t07/[A-Za-z0-9_.-]+/t07_table\.json$")
_RANGE_NETWORK_KINDS = {SOURCE_SESSION: BLOCK_FINALS, SOURCE_REFERENCE: BLOCK_REFERENCES}
"""The sources a range entry's network may have, and the block whose rules check each."""

CODE_MODULES = (
    "weightcode.evalkit.sealed_finals",
    "weightcode.evalkit.devreadout",
    "weightcode.evalkit.sealed",
    "weightcode.evalkit.metrics",
    "weightcode.evalkit.patch_check",
    "weightcode.evalkit.companions",
    "weightcode.evalkit.targets",
    "weightcode.baselines.instance_io",
    "weightcode.patch.executor",
    "weightcode.patch.files",
    "weightcode.patch.ops",
    "weightcode.patch.limits",
    "weightcode.evalkit.e10",
)
"""Modules whose source files are hashed into every record: this pipeline, the vision readout of record and what it
calls, the instance loader (it loads each instance's forward.py), the patch applier and its limits (they rebuild
finals and picks; ``e10_program_limits`` are the limits of the vision reference runs). A family adds its own modules
(``Family.code_modules``)."""


class SealedFinalsError(RuntimeError):
    """A list, child, reference, index entry, code state or registry state that this pipeline refuses."""


class SealedEntriesFailed(SealedFinalsError):
    """A sealed read that finished with failed entries: they are recorded and spent, the others were read."""

    def __init__(self, message: str, results_path: Path, n_read: int) -> None:
        super().__init__(message)
        self.results_path = results_path
        self.n_read = n_read

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (str(self), self.results_path, self.n_read))


@dataclass(frozen=True)
class Final:
    """One entry of a list: a child read on one instance (a final, or a reference pick beside one)."""

    name: str
    readout: str
    source: Mapping[str, Any]
    instance_id: str
    child_sha256: str
    block: str = BLOCK_FINALS
    pairs_with: str | None = None

    def as_dict(self) -> dict[str, Any]:
        entry = {"name": self.name, "readout": self.readout, "source": dict(self.source),
                 "instance_id": self.instance_id, "child_sha256": self.child_sha256}
        if self.block == BLOCK_REFERENCES:
            entry["pairs_with"] = self.pairs_with
        return entry


# ------------------------------------------------------------------------------------------------- readout families


class Family(Protocol):
    """A readout family: how its instances load, which rows it reads in each mode, and its readout of record."""

    readout: str
    code_modules: tuple[str, ...]

    def load_instance(self, instance_id: str) -> Instance:
        """The instance, refused unless it is a current instance of the family's rule."""
        ...

    def reference_limits(self, run: Mapping[str, Any], run_dir: Path, instance: Instance) -> Limits | None:
        """The limits a reference run applied when it enforced them (None: ``Limits()``), checked against the run."""
        ...

    def read_context(self, final: Final, run: Mapping[str, Any] | None) -> dict[str, Any]:
        """What the readout of record takes from a reference run beyond the child (for example a fit set)."""
        ...

    def dev_rows(self, instance: Instance) -> tuple[Any, dict[str, Any]]:
        """The dev rows of record and a record of what they are."""
        ...

    def sealed_index(self, instance: Instance) -> dict[str, Any]:
        """The instance's sealed test entries as the private dataset.json files record them; opens nothing sealed."""
        ...

    def sealed_paths(self, index: Mapping[str, Any]) -> list[Path]:
        """The sealed files ``sealed_index`` names."""
        ...

    def sealed_rows(self, instance: Instance, index: Mapping[str, Any], cache: dict[Any, Any]
                    ) -> tuple[Any, dict[str, Any]]:
        """The sealed test rows, hash-checked when first loaded, and a record of what they are."""
        ...

    def read(self, instance: Instance, child: Mapping[str, np.ndarray], rows: Any, batch_size: int,
             context: Mapping[str, Any]) -> tuple[dict[str, Any], float, float]:
        """The readout parts of the record (JSON-safe), gain and clean in points."""
        ...

    def session_parts(self, info_final: Mapping[str, Any]) -> dict[str, Any]:
        """The readout parts of a session's dev readout of record (info.json ``final``)."""
        ...

    def reference_parts(self, lines: Sequence[Mapping[str, Any]], final: Final, patch_id: str | None
                        ) -> dict[str, Any]:
        """The readout parts of a reference pick in the dev record lines of its run."""
        ...

    def points(self, parts: Mapping[str, Any]) -> tuple[float, float]:
        """Gain and clean of a readout's parts."""
        ...

    def environment(self) -> dict[str, Any]:
        """Versions of the packages the family's readout runs on, beyond Python and numpy."""
        ...


def family(readout: str) -> Family:
    """The family implementing ``readout``; refused when its module does not import in this tree."""
    name = FAMILY_MODULES.get(readout)
    if name is None:
        raise SealedFinalsError(f"readout {readout!r} is not implemented; known: {list(READOUTS)}")
    if readout == READOUT_VISION_JOINT:
        return VISION
    try:
        module = importlib.import_module(name)
    except ImportError as error:
        raise SealedFinalsError(f"readout {readout!r} needs {name}, which does not import in this tree ({error}); "
                                "its experiment's code must be merged before its entries are read") from error
    return module.FAMILY


def retention_line(lines: Sequence[Mapping[str, Any]], final: Final, patch_id: str | None) -> Mapping[str, Any]:
    """The one ``devreadout.readout_retention`` record of a run whose selection is this pick at this floor."""
    floor = final.source["s_old_floor"]
    hits = [line for line in lines if isinstance(line.get("selection"), Mapping)
            and line["selection"].get("patch_id") == patch_id and line["selection"].get("s_old_floor") == floor
            and line.get("instance_id") == final.instance_id]
    if len(hits) != 1:
        raise SealedFinalsError(f"{final.name}: {len(hits)} records of its run select {patch_id} at S_old >= {floor} "
                                f"in {final.source['dev_record']}; expected one")
    return hits[0]


class VisionJoint:
    """vision_joint: ``devreadout.dev_readout`` on the condition rows and on their clean companion."""

    readout = READOUT_VISION_JOINT
    code_modules: tuple[str, ...] = ()

    def load_instance(self, instance_id: str) -> Instance:
        return load_instance(instances_root() / instance_id)

    def reference_limits(self, run: Mapping[str, Any], run_dir: Path, instance: Instance) -> Limits | None:
        """As ``devreadout._readout_repair_run``: the E10 program limits for a run that records them, else None."""
        if not str(run.get("limits_source", "")).startswith("e10_program_limits"):
            return None
        expected = devreadout.e10_program_limits(instance.task["limits"], len(instance.params),
                                                 devreadout.dense_delta_bytes(v.shape for v in instance.params.values()))
        try:
            return devreadout._program_baseline_limits(run, run_dir, expected, "e10", "E10")
        except devreadout.DevReadoutError as error:
            raise SealedFinalsError(str(error)) from error

    def read_context(self, final: Final, run: Mapping[str, Any] | None) -> dict[str, Any]:
        return {}

    def dev_rows(self, instance: Instance) -> tuple[Any, dict[str, Any]]:
        splits = deployment_splits(str(instance.private["dataset"]), str(instance.private["condition"]))
        condition = devreadout.readout_split(instance)
        clean = load_clean_rows(instance, devreadout.READOUT_SPLIT)
        record = {
            "condition": {"split": devreadout.READOUT_SPLIT, "path": str(splits.v_select),
                          "sha256": splits.v_select_sha256, "n": int(len(condition["y"]))},
            "clean": {"split": f"{devreadout.READOUT_SPLIT} clean companion", "sha256": clean.clean_sha256,
                      "n": int(len(clean.y))},
        }
        return (condition, clean.arrays()), record

    def sealed_index(self, instance: Instance) -> dict[str, Any]:
        return sealed_index(str(instance.private["dataset"]), str(instance.private["condition"]))

    def sealed_paths(self, index: Mapping[str, Any]) -> list[Path]:
        return [Path(index["condition_test"]["path"]), Path(index["clean_test"]["path"])]

    def sealed_rows(self, instance: Instance, index: Mapping[str, Any], cache: dict[Any, Any]
                    ) -> tuple[Any, dict[str, Any]]:
        clean, clean_check = _vision_clean(index["clean_test"], cache)
        condition, condition_check = _vision_condition(index["condition_test"], clean, cache)
        record = {
            "condition": {"split": "sealed test", **index["condition_test"], **condition_check,
                          "n": int(len(condition["y"]))},
            "clean": {"split": "sealed clean test", **index["clean_test"], **clean_check, "n": int(len(clean["y"]))},
        }
        return (condition, clean), record

    def read(self, instance: Instance, child: Mapping[str, np.ndarray], rows: Any, batch_size: int,
             context: Mapping[str, Any]) -> tuple[dict[str, Any], float, float]:
        condition, clean = rows
        try:
            readouts = devreadout.dev_readout(instance, dict(child), batch_size, split=condition)["readouts"]
            readouts_clean = devreadout.dev_readout(instance, dict(child), batch_size, split=clean)["readouts"]
        except devreadout.DevReadoutError as error:
            raise SealedFinalsError(str(error)) from error
        parts = {"readouts": devreadout.json_safe(readouts), "readouts_clean": devreadout.json_safe(readouts_clean)}
        return parts, *self.points(parts)

    def session_parts(self, info_final: Mapping[str, Any]) -> dict[str, Any]:
        return {"readouts": info_final["vselect"], "readouts_clean": info_final["vselect_clean"]}

    def reference_parts(self, lines: Sequence[Mapping[str, Any]], final: Final, patch_id: str | None
                        ) -> dict[str, Any]:
        line = retention_line(lines, final, patch_id)
        return {"readouts": line["readouts"], "readouts_clean": line["readouts_clean"]}

    def points(self, parts: Mapping[str, Any]) -> tuple[float, float]:
        return points(parts["readouts"]["delta_acc"]), points(parts["readouts_clean"]["delta_acc"])

    def environment(self) -> dict[str, Any]:
        return {}


VISION = VisionJoint()


# ---------------------------------------------------------------------------------------------------------- the list


def load_list(path: str | Path) -> tuple[dict[str, Any], list[Final]]:
    """Parse and validate a list; returns the parsed object and its entries (finals, then references). Touches no
    store file and imports no family module."""
    path = Path(path)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SealedFinalsError(f"{path}: not valid JSON ({error})") from error
    if not isinstance(parsed, dict) or parsed.get("schema") not in _LIST_KEYS:
        got = parsed.get("schema") if isinstance(parsed, dict) else type(parsed).__name__
        raise SealedFinalsError(f"{path}: schema must be one of {sorted(_LIST_KEYS)}, got {got!r}")
    schema = parsed["schema"]
    if set(parsed) != _LIST_KEYS[schema]:
        raise SealedFinalsError(f"{path}: {schema} list keys must be exactly {sorted(_LIST_KEYS[schema])}, got "
                                f"{sorted(parsed)}")
    status, approval = parsed["status"], parsed["approval"]
    if status == STATUS_DRAFT:
        if approval is not None:
            raise SealedFinalsError(f"{path}: a draft list has approval null")
    elif status == STATUS_APPROVED:
        if not isinstance(approval, str) or not approval.strip():
            raise SealedFinalsError(f"{path}: an approved list states its approval (who, when, how)")
    else:
        raise SealedFinalsError(f"{path}: status must be {STATUS_DRAFT!r} or {STATUS_APPROVED!r}, got {status!r}")
    if not isinstance(parsed["note"], str):
        raise SealedFinalsError(f"{path}: note must be a string")
    if not isinstance(parsed["finals"], list) or not parsed["finals"]:
        raise SealedFinalsError(f"{path}: finals must be a non-empty list")
    references = parsed.get("references", [])
    if not isinstance(references, list):
        raise SealedFinalsError(f"{path}: references must be a list (possibly empty)")
    readouts = (READOUT_VISION_JOINT,) if schema == LIST_SCHEMA_V0 else READOUTS
    entries: list[Final] = []
    names: set[str] = set()
    pairs: set[tuple[str, ...]] = set()
    for block, rows in ((BLOCK_FINALS, parsed["finals"]), (BLOCK_REFERENCES, references)):
        for number, row in enumerate(rows):
            entry = _entry(row, block, readouts, f"{path}: {block} {number}")
            if entry.name in names:
                raise SealedFinalsError(f"{path}: {block} {number}: duplicate name {entry.name!r}")
            if entry_key(entry) in pairs:
                raise SealedFinalsError(f"{path}: {block} {number}: {entry_key(entry)} appears twice; each entry is "
                                        "read once")
            names.add(entry.name)
            pairs.add(entry_key(entry))
            entries.append(entry)
    finals = {entry.name: entry for entry in entries if entry.block == BLOCK_FINALS}
    for entry in entries:
        if entry.block != BLOCK_REFERENCES:
            continue
        if entry.pairs_with is None:
            if status == STATUS_APPROVED:
                raise SealedFinalsError(f"{path}: reference {entry.name} pairs with no final; in an approved list "
                                        "every reference is read beside its final")
            continue
        paired = finals.get(entry.pairs_with)
        if paired is None or paired.instance_id != entry.instance_id or paired.readout != entry.readout:
            raise SealedFinalsError(f"{path}: reference {entry.name} pairs_with {entry.pairs_with!r}, which is not a "
                                    f"final on {entry.instance_id} with readout {entry.readout}")
    return parsed, entries


def entry_key(entry: Final) -> tuple[str, ...]:
    """What is read once within a list: (instance_id, child_sha256), and the set for a range entry."""
    if entry.source.get("kind") == SOURCE_RANGE:
        return (entry.instance_id, entry.child_sha256, str(entry.source["set"]))
    return (entry.instance_id, entry.child_sha256)


def _entry(row: Any, block: str, readouts: Sequence[str], where: str) -> Final:
    if not isinstance(row, dict) or set(row) != _ENTRY_KEYS[block]:
        raise SealedFinalsError(f"{where}: keys must be exactly {sorted(_ENTRY_KEYS[block])}")
    name, readout, source = row["name"], row["readout"], row["source"]
    instance_id, child_sha = row["instance_id"], row["child_sha256"]
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise SealedFinalsError(f"{where}: invalid name {name!r}")
    if readout not in readouts:
        raise SealedFinalsError(f"{where}: readout {readout!r} is not implemented; known: {list(readouts)}")
    if not isinstance(instance_id, str) or not INSTANCE_PATTERNS[readout].fullmatch(instance_id):
        raise SealedFinalsError(f"{where}: invalid instance_id {instance_id!r} for readout {readout}")
    if not isinstance(child_sha, str) or not _SHA_RE.fullmatch(child_sha):
        raise SealedFinalsError(f"{where}: child_sha256 must be 64 lower-case hex digits")
    _check_source(source, block, readout, where)
    pairs_with = row.get("pairs_with")
    if block == BLOCK_REFERENCES and pairs_with is not None and not isinstance(pairs_with, str):
        raise SealedFinalsError(f"{where}: pairs_with must be a final's name or null")
    return Final(name, readout, dict(source), instance_id, child_sha, block, pairs_with)


def _relative(value: Any, pattern: re.Pattern[str], what: str, where: str) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value) or ".." in Path(value).parts:
        raise SealedFinalsError(f"{where}: {what} must be a path relative to $WEIGHTCODE_RUNS of the form "
                                f"{pattern.pattern}, got {value!r}")


def _check_source(source: Any, block: str, readout: str, where: str) -> None:
    kinds = _BLOCK_SOURCES[block]
    if not isinstance(source, dict) or source.get("kind") not in kinds:
        raise SealedFinalsError(f"{where}: source must be an object with kind in {sorted(kinds)}")
    kind = source["kind"]
    if (kind == SOURCE_RANGE) != (readout == READOUT_VISION_RANGE):
        raise SealedFinalsError(f"{where}: a {SOURCE_RANGE} source is exactly the source of a {READOUT_VISION_RANGE} "
                                f"entry (kind {kind!r}, readout {readout!r})")
    if kind == SOURCE_REFERENCE:
        pick = source.get("pick")
        if not isinstance(pick, str) or pick not in _REFERENCE_KEYS:
            raise SealedFinalsError(f"{where}: pick must be one of {sorted(_REFERENCE_KEYS)} ({PICK_KEPT!r}: the kept "
                                    f"pick under joint-retain-v1; {PICK_OWN!r}: {OWN_RULE}; {PICK_CANDIDATE!r}: "
                                    f"{CANDIDATE_RULE}), got {pick!r}")
        keys = _REFERENCE_KEYS[pick]
        if set(source) != keys:
            raise SealedFinalsError(f"{where}: a {kind} source with pick {pick!r} has exactly the keys {sorted(keys)}")
    elif set(source) != _SOURCE_KEYS[kind]:
        raise SealedFinalsError(f"{where}: a {kind} source has exactly the keys {sorted(_SOURCE_KEYS[kind])}")
    if kind == SOURCE_RANGE:
        _check_range(source, where)
    if kind == SOURCE_SESSION and not (isinstance(source["session_id"], str)
                                       and _SESSION_RE.fullmatch(source["session_id"])):
        raise SealedFinalsError(f"{where}: invalid session_id {source['session_id']!r}")
    if kind in (*_STREAM_KINDS, SOURCE_TRANSFER):
        if readout != READOUT_VISION_JOINT:
            raise SealedFinalsError(f"{where}: a {kind} source is a vision_joint entry")
        _relative(source["record"], _RECORD_RE, "record", where)
    if kind == SOURCE_STREAM_ROUND:
        number = source["round"]
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise SealedFinalsError(f"{where}: round must be an integer >= 1 (the stream's round number), got "
                                    f"{number!r}")
    if kind == SOURCE_REFERENCE:
        _relative(source["run"], _RUN_RE, "run", where)
        _relative(source["dev_record"], _DEV_RECORD_RE, "dev_record", where)
        pick = source["pick"]
        if pick != PICK_KEPT and readout != READOUT_VISION_JOINT:
            raise SealedFinalsError(f"{where}: a {pick!r} pick is a vision_joint reference")
        if pick == PICK_KEPT:
            floor = source["s_old_floor"]
            if not isinstance(floor, int) or isinstance(floor, bool) or floor > 0:
                raise SealedFinalsError(f"{where}: s_old_floor must be an integer <= 0, got {floor!r}")
        if pick == PICK_CANDIDATE and not (isinstance(source["patch_id"], str)
                                           and _PATCH_ID_RE.fullmatch(source["patch_id"])):
            raise SealedFinalsError(f"{where}: invalid patch_id {source['patch_id']!r}")


def _check_range(source: Mapping[str, Any], where: str) -> None:
    if not isinstance(source["set"], str) or not _SET_RE.fullmatch(source["set"]):
        raise SealedFinalsError(f"{where}: set must be <corruption>_s<1..5>, got {source['set']!r}")
    _relative(source["dev_record"], _T07_RECORD_RE, "dev_record", where)
    for key in ("t07_arch", "t07_network"):
        if not isinstance(source[key], str) or not _NAME_RE.fullmatch(source[key]):
            raise SealedFinalsError(f"{where}: invalid {key} {source[key]!r}")
    network = source["network"]
    kind = network.get("kind") if isinstance(network, dict) else None
    if kind not in _RANGE_NETWORK_KINDS or (kind == SOURCE_REFERENCE and network.get("pick") != PICK_KEPT):
        raise SealedFinalsError(f"{where}: a range network is a session source or a kept reference source")
    _check_source(network, _RANGE_NETWORK_KINDS[kind], READOUT_VISION_JOINT, f"{where} network")


def range_network(final: Final) -> Final:
    """The entry whose child a range entry reads: its network source on the same instance (vision_joint rules)."""
    network = final.source["network"]
    return Final(final.name, READOUT_VISION_JOINT, dict(network), final.instance_id, final.child_sha256,
                 _RANGE_NETWORK_KINDS[network["kind"]])


def entry_instance(final: Final, instance: Instance) -> Instance:
    """The instance a family reads an entry with: the instance itself, or for a range entry a view of it whose private
    record names the set (``RANGE_VIEW_KEY``); its own condition is refused there."""
    if final.source["kind"] != SOURCE_RANGE:
        return instance
    own, wanted = instance.private.get("condition"), final.source["set"]
    if wanted == own:
        raise SealedFinalsError(f"{final.name}: {wanted} is the own condition of {instance.instance_id}; it is read "
                                "by the network's own entry, not by a range entry")
    return dataclasses.replace(instance, private={**instance.private,
                                                  RANGE_VIEW_KEY: {"set": wanted, "own_condition": own}})


# ------------------------------------------------------------------------------------------------- child recovery


def _inside(root: Path, path: Path, what: str) -> Path:
    resolved = path.resolve()
    root = root.resolve()
    if resolved != root and root not in resolved.parents:
        raise SealedFinalsError(f"{what} {path} is not inside {root}")
    return resolved


def session_dir(session_id: str) -> Path:
    return runs_dir() / "sessions" / session_id


def recover_child(final: Final, instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The child of ``final`` and how it was recovered; its hash must be the list's ``child_sha256``."""
    kind = final.source["kind"]
    if kind == SOURCE_RANGE:
        child, network = recover_child(range_network(final), instance)
        provenance = {"kind": SOURCE_RANGE, "set": final.source["set"], "network": network}
    elif kind == SOURCE_SESSION:
        child, provenance = _session_child(final, instance)
    elif kind in _STREAM_KINDS:
        child, provenance = _stream_child(final, instance)
    elif kind == SOURCE_TRANSFER:
        child, provenance = _transfer_child(final, instance)
    else:
        child, provenance = _reference_child(final, instance)
    digest = params_sha256(child)
    if digest != final.child_sha256:
        raise SealedFinalsError(f"{final.name}: recovered child {digest} != the list's child_sha256 "
                                f"{final.child_sha256}")
    return child, provenance


def _session_child(final: Final, instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    session_id = final.source["session_id"]
    root = session_dir(session_id)
    run_path = root / "run.json"
    if not run_path.is_file():
        raise SealedFinalsError(f"{final.name}: no {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("session_id", session_id) != session_id:
        raise SealedFinalsError(f"{final.name}: {run_path} names session {run.get('session_id')!r}")
    recorded = run.get("instance", {})
    if recorded.get("instance_id") != final.instance_id:
        raise SealedFinalsError(f"{final.name}: session {session_id} ran on {recorded.get('instance_id')!r}, the list "
                                f"says {final.instance_id}")
    if recorded.get("params_sha256") != instance.params_sha256:
        raise SealedFinalsError(f"{final.name}: session {session_id} records base {recorded.get('params_sha256')!r}, "
                                f"the instance's is {instance.params_sha256}")
    if run["final"].get("child_sha256") != final.child_sha256 and not run["final"].get("is_base_noop"):
        raise SealedFinalsError(f"{final.name}: session {session_id} records final child "
                                f"{run['final'].get('child_sha256')!r}, the list says {final.child_sha256}")
    try:
        child = devreadout.final_child(run, root, instance)
    except devreadout.DevReadoutError as error:
        raise SealedFinalsError(f"{final.name}: {error}") from error
    provenance = {
        "kind": SOURCE_SESSION, "session_id": session_id, "run_json": str(run_path),
        "run_json_sha256": file_sha256(run_path), "session_status": run.get("status"),
        "patch_id": run["final"].get("patch_id"), "query_index": run["final"].get("query_index"),
        "is_base_noop": bool(run["final"].get("is_base_noop")), "n_changed": run["final"].get("n_changed"),
        "recovered_by": "devreadout.final_child: final/patch.json applied to the instance base under the limits in "
                        "session.json, checked against run.json final.child_sha256",
    }
    return child, provenance


def _stream_record(final: Final) -> tuple[Path, dict[str, Any]]:
    root = runs_dir()
    path = _inside(root, root / final.source["record"], "stream record")
    if not path.is_file():
        raise SealedFinalsError(f"{final.name}: no stream record {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") not in STREAM_SCHEMAS:
        raise SealedFinalsError(f"{final.name}: {path} has schema {record.get('schema')!r}; known "
                                f"{list(STREAM_SCHEMAS)}")
    return path, record


def stream_round(final: Final, record: Mapping[str, Any]) -> Mapping[str, Any]:
    """The round of a completed stream an entry reads: the last (``stream_final``) or round k before it
    (``stream_round``, numbered from 1 as the record numbers them)."""
    stream_ids = [row.get("instance_id") for row in record.get("instances", [])]
    if final.instance_id not in stream_ids:
        raise SealedFinalsError(f"{final.name}: {final.instance_id} is not an instance of the stream {stream_ids}")
    rounds = record.get("rounds", [])
    if not rounds or len(rounds) != len(stream_ids):
        raise SealedFinalsError(f"{final.name}: the stream has {len(rounds)} rounds for {len(stream_ids)} instances; "
                                "only a completed stream has a final or round students of record")
    if final.source["kind"] == SOURCE_STREAM_FINAL:
        return rounds[-1]
    number = int(final.source["round"])
    if number >= len(rounds):
        raise SealedFinalsError(f"{final.name}: round {number} of a {len(rounds)}-round stream; a stream_round entry "
                                "reads a round before the last (the last round's student is a stream_final entry)")
    row = rounds[number - 1]
    if row.get("round") != number:
        raise SealedFinalsError(f"{final.name}: the stream's round {number} is numbered {row.get('round')!r}")
    return row


def _stream_child(final: Final, instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path, record = _stream_record(final)
    row = stream_round(final, record)
    if record.get("base_params_sha256") != instance.params_sha256:
        raise SealedFinalsError(f"{final.name}: the stream's base {record.get('base_params_sha256')!r} is not the "
                                f"instance's base {instance.params_sha256}")
    after = row.get("student_after") or {}
    if after.get("sha256") != final.child_sha256 or row.get("child_sha256") != final.child_sha256:
        raise SealedFinalsError(f"{final.name}: round {row.get('round')!r}'s student is {after.get('sha256')!r}, the "
                                f"list says {final.child_sha256}")
    student = _inside(runs_dir(), Path(str(after.get("path"))), "student_after")
    child = load_params(student)
    which = "the last round's" if final.source["kind"] == SOURCE_STREAM_FINAL else f"round {row.get('round')}'s"
    provenance = {
        "kind": final.source["kind"], "record": str(path), "record_sha256": file_sha256(path),
        "round": row.get("round"), "n_rounds": len(record["rounds"]), "round_status": row.get("status"),
        "condition": row.get("condition"), "student_after": str(student),
        "recovered_by": f"{which} student_after file, checked against its recorded sha256",
    }
    return child, provenance


def transfer_record(final: Final) -> tuple[Path, dict[str, Any]]:
    """A transfer entry's record (``weightcode.harness.program_transfer.v0``), refused unless it is an ok run of the program
    on the entry's instance whose child the limits admitted and whose child hash is the list's."""
    root = runs_dir()
    path = _inside(root, root / final.source["record"], "transfer record")
    if not path.is_file():
        raise SealedFinalsError(f"{final.name}: no transfer record {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != TRANSFER_SCHEMA:
        raise SealedFinalsError(f"{final.name}: {path} has schema {record.get('schema')!r}, not {TRANSFER_SCHEMA}")
    target = record.get("target") or {}
    if target.get("instance_id") != final.instance_id:
        raise SealedFinalsError(f"{final.name}: {path} ran on {target.get('instance_id')!r}, the list says "
                                f"{final.instance_id}")
    if (record.get("status") != "ok" or record.get("within_limits") is not True
            or record.get("readout_child") != "within_limits"):
        raise SealedFinalsError(f"{final.name}: {path} is a transfer with status {record.get('status')!r}, "
                                f"within_limits {record.get('within_limits')!r}; only a child the limits admitted is "
                                "the program's result")
    if record.get("child_sha256") != final.child_sha256:
        raise SealedFinalsError(f"{final.name}: {path} records child {record.get('child_sha256')!r}, the list says "
                                f"{final.child_sha256}")
    return path, record


def _transfer_child(final: Final, instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path, record = transfer_record(final)
    recorded_base = record["target"].get("params_sha256")
    if recorded_base != instance.params_sha256:
        raise SealedFinalsError(f"{final.name}: {path} records base {recorded_base!r}, the instance's is "
                                f"{instance.params_sha256}")
    program = record.get("program") or {}
    program_path = _inside(runs_dir() / "sessions", Path(str(program.get("path"))), "transfer program")
    if not program_path.is_file() or file_sha256(program_path) != program.get("sha256"):
        raise SealedFinalsError(f"{final.name}: the program {program_path} is missing or not the file {path} ran "
                                f"(sha256 {program.get('sha256')!r})")
    work = path.with_suffix("")
    if work.resolve() != Path(str(record.get("work_dir"))).resolve():
        raise SealedFinalsError(f"{final.name}: {path} records work_dir {record.get('work_dir')!r}, not {work}")
    patch_dir = work / "patch"
    patch_file = patch_dir / TRANSFER_PATCH
    if not patch_file.is_file() or file_sha256(patch_file) != record.get("patch_sha256"):
        raise SealedFinalsError(f"{final.name}: {patch_file} is missing or not the patch {path} applied")
    try:
        limits = Limits(**dict(record["limits"]))
    except (TypeError, ValueError) as error:
        raise SealedFinalsError(f"{final.name}: {path} records limits {record.get('limits')!r} ({error})") from error
    if task_limits(limits) != record["limits"]:
        raise SealedFinalsError(f"{final.name}: {path} limits {record['limits']} do not round-trip as Limits")
    try:
        result = apply_patch(instance.params, json.loads(patch_file.read_text(encoding="utf-8")), patch_dir, limits,
                             instance.params_sha256)
    except PatchError as error:
        raise SealedFinalsError(f"{final.name}: {patch_file} under the recorded limits: {error}") from error
    if result.child_sha256 != record["child_sha256"]:
        raise SealedFinalsError(f"{final.name}: rebuilt child {result.child_sha256} != the transfer's child "
                                f"{record['child_sha256']}")
    provenance = {
        "kind": SOURCE_TRANSFER, "record": str(path), "record_sha256": file_sha256(path),
        "program": {"path": str(program_path), "sha256": program.get("sha256")},
        "patch": str(patch_file), "patch_sha256": record.get("patch_sha256"), "limits": record["limits"],
        "n_changed": result.n_changed,
        "recovered_by": "the transfer's derived_patch.json applied to the target's base under the limits the transfer "
                        "applied (weightcode.patch.apply_patch), checked against the child hash the transfer recorded",
    }
    return dict(result.child), provenance


def reference_run(final: Final) -> tuple[Path, dict[str, Any]]:
    """A reference entry's baseline run.json, refused unless it is a baseline run on the entry's instance."""
    root = runs_dir()
    path = _inside(root, root / final.source["run"], "reference run")
    if not path.is_file():
        raise SealedFinalsError(f"{final.name}: no reference run {path}")
    run = json.loads(path.read_text(encoding="utf-8"))
    if (path.parent / devreadout.SESSION_JSON).is_file() or run.get("session_result") is not None:
        raise SealedFinalsError(f"{final.name}: {path} is a harness run; a reference is a baseline run")
    if run.get("instance", {}).get("instance_id") != final.instance_id:
        raise SealedFinalsError(f"{final.name}: {path} ran on {run.get('instance', {}).get('instance_id')!r}, the "
                                f"list says {final.instance_id}")
    return path, run


def kept_pick(final: Final, run: Mapping[str, Any]) -> dict[str, Any]:
    """``devreadout.retention_final`` of a reference run at the entry's floor: the kept pick's final and selection."""
    try:
        return devreadout.retention_final(run, int(final.source["s_old_floor"]))
    except devreadout.DevReadoutError as error:
        raise SealedFinalsError(f"{final.name}: {error}") from error


def _log_candidates(final: Final, run: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    log = run.get("oracle_log")
    if not isinstance(log, list) or not log or not all("score_old" in entry for entry in log):
        raise SealedFinalsError(f"{final.name}: a {final.source['pick']!r} pick needs the run's joint oracle log")
    return log


def own_pick(final: Final, run: Mapping[str, Any]) -> dict[str, Any]:
    """The run's own final, refused unless it is ``choose_final`` over the run's whole oracle log (as
    ``devreadout.retention_final`` checks) and a checked candidate rather than the base no-op."""
    log = _log_candidates(final, run)
    primary = choose_final([Candidate(patch_id=str(e["patch_id"]), score=int(e["score"]),
                                      n_changed=int(e["n_changed"]), query_index=int(e["query_index"])) for e in log])
    own = run["final"]
    if own.get("is_base_noop") or primary.patch_id != own.get("patch_id") or primary.score != own.get("score"):
        raise SealedFinalsError(f"{final.name}: the run's final {own.get('patch_id')!r} is not a checked candidate "
                                "chosen by choose_final over its oracle log")
    selection = {"rule": OWN_RULE, "n_checks": len(log), "patch_id": own["patch_id"],
                 "query_index": own.get("query_index")}
    return {"final": dict(own), "selection": selection}


def candidate_pick(final: Final, run: Mapping[str, Any]) -> dict[str, Any]:
    """The checked candidate ``patch_id`` of the run, as ``devreadout.readout_candidates`` rebuilds it."""
    log = _log_candidates(final, run)
    patch_id = final.source["patch_id"]
    entries = [entry for entry in log if str(entry["patch_id"]) == patch_id]
    candidates = [c for c in run.get("candidates", []) if c.get("patch_id") == patch_id]
    if len(entries) != 1 or len(candidates) != 1:
        raise SealedFinalsError(f"{final.name}: the run has {len(entries)} checks and {len(candidates)} candidates "
                                f"with patch_id {patch_id!r}; expected one of each")
    (entry,), (candidate,) = entries, candidates
    pick = {"patch_id": patch_id, "is_base_noop": False, "dir": patch_id, "query_index": int(entry["query_index"]),
            "child_sha256": entry["child_sha256"], "edit_size": int(candidate["edit"]["edit_size"]),
            **{key: int(entry[key]) for key in ("score", "fixed", "broken", "score_new", "score_old")}}
    selection = {"rule": CANDIDATE_RULE, "n_checks": len(log), "patch_id": patch_id,
                 "query_index": int(entry["query_index"]), "config": candidate.get("config")}
    return {"final": pick, "selection": selection}


def reference_pick(final: Final, run: Mapping[str, Any]) -> dict[str, Any]:
    """The pick a reference entry names: kept, own or a named candidate; its ``final`` and ``selection``."""
    pick = final.source["pick"]
    if pick == PICK_KEPT:
        return kept_pick(final, run)
    if pick == PICK_OWN:
        return own_pick(final, run)
    return candidate_pick(final, run)


_PICK_RECOVERY = {
    PICK_KEPT: "devreadout.retention_final over the run's oracle log",
    PICK_OWN: "the run's own final, checked to be choose_final over its oracle log",
    PICK_CANDIDATE: "the named checked candidate of the run's oracle log",
}


def _reference_child(final: Final, instance: Instance) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path, run = reference_run(final)
    recorded = run["instance"]
    for key, ours in (("params_sha256", instance.params_sha256), ("forward_sha256", instance.forward_sha256)):
        if recorded.get(key) != ours:
            raise SealedFinalsError(f"{final.name}: {path} records {key} {recorded.get(key)!r}, the instance's is "
                                    f"{ours}")
    which = final.source["pick"]
    picked = reference_pick(final, run)
    pick = picked["final"]
    recorded_child = instance.params_sha256 if pick["is_base_noop"] else pick["child_sha256"]
    if recorded_child != final.child_sha256:
        raise SealedFinalsError(f"{final.name}: the {which} pick {pick['patch_id']} records child {recorded_child}, "
                                f"the list says {final.child_sha256}")
    limits = family(final.readout).reference_limits(run, path.parent, instance)
    try:
        child = devreadout.final_child({**run, "final": pick}, path.parent, instance, limits)
    except devreadout.DevReadoutError as error:
        raise SealedFinalsError(f"{final.name}: {error}") from error
    provenance = {
        "kind": SOURCE_REFERENCE, "run_json": str(path), "run_json_sha256": file_sha256(path),
        "method": run.get("method"), "pick": which, "selection": picked["selection"],
        "patch_id": pick["patch_id"], "query_index": pick["query_index"], "is_base_noop": bool(pick["is_base_noop"]),
        "recovered_by": f"{_PICK_RECOVERY[which]}, then devreadout.final_child: the pick's patch.json applied to the "
                        "instance base under the limits the run applied, checked against the child hash its check "
                        "recorded",
    }
    return child, provenance


def read_context(final: Final) -> dict[str, Any]:
    """The family's read context of an entry (from its reference run, if it is a reference)."""
    run = reference_run(final)[1] if final.source["kind"] == SOURCE_REFERENCE else None
    return family(final.readout).read_context(final, run)


# ------------------------------------------------------------------------------------------------ the sealed index


def resolve_location(location: Any, what: str) -> Path:
    """``$WEIGHTCODE_SEALED/<relative>`` as a path under the sealed root; nothing below it is opened or statted."""
    if not isinstance(location, str) or not location.startswith(SEALED_PREFIX):
        raise SealedFinalsError(f"{what}: location {location!r} is not under {SEALED_PREFIX}")
    relative = Path(location[len(SEALED_PREFIX):])
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SealedFinalsError(f"{what}: location {location!r} leaves {SEALED_PREFIX}")
    return sealed_dir() / relative


def recorded_sha(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise SealedFinalsError(f"{what}: no recorded sha256")
    return value


def sealed_index(dataset: str, condition: str) -> dict[str, Any]:
    """The vision_joint sealed test sets of one (dataset, condition) as its dataset.json files record them.

    Reads only the private dataset.json files (and, through ``deployment_splits``, the condition's store check of its
    private splits). Resolves paths under ``$WEIGHTCODE_SEALED`` without opening, listing or statting anything below it.
    """
    if condition == DEPLOY_CONDITION:
        raise SealedFinalsError(f"{dataset}: the deploy condition has no clean companion; vision_joint needs a named "
                                "condition")
    try:
        deployment_splits(dataset, condition)
        clean_meta = read_dataset_meta(dataset)
        condition_meta = read_condition_meta(dataset, condition)
    except (LookupError, ValueError) as error:  # ConditionMismatch is a ValueError
        raise SealedFinalsError(f"{dataset}/{condition}: {error}") from error
    clean_json = store.dataset_json_path(dataset)
    condition_json = store.dataset_json_path(dataset, condition)
    clean_test, condition_test = clean_meta.get("test"), condition_meta.get("test")
    if not isinstance(clean_test, dict) or not isinstance(condition_test, dict):
        raise SealedFinalsError(f"{dataset}/{condition}: dataset.json records no test object")
    clean_sha = recorded_sha(clean_test.get("sha256"), f"{clean_json} test")
    if "format" in clean_test:
        raise SealedFinalsError(f"{clean_json}: the clean test has format {clean_test['format']!r}; only the store "
                                ".npz layout is known for a clean test set")
    if recorded_sha(condition_test.get("clean_sha256"), f"{condition_json} test.clean_sha256") != clean_sha:
        raise SealedFinalsError(f"{condition_json}: test.clean_sha256 is not the clean test's sha256 {clean_sha}")
    clean = {
        "location": clean_test.get("location"),
        "path": str(resolve_location(clean_test.get("location"), str(clean_json))), "format": FORMAT_STORE_NPZ,
        "n": clean_test.get("n"), "sha256": clean_sha, "sha256_kind": HASH_CONTENT,
        "index": str(clean_json), "index_file_sha256": file_sha256(clean_json),
    }
    entry: dict[str, Any] = {
        "location": condition_test.get("location"),
        "path": str(resolve_location(condition_test.get("location"), str(condition_json))),
        "n": condition_test.get("n"), "sha256": recorded_sha(condition_test.get("sha256"), f"{condition_json} test"),
        "index": str(condition_json), "index_file_sha256": file_sha256(condition_json),
    }
    fmt = condition_test.get("format")
    if fmt is None:
        entry.update(format=FORMAT_STORE_NPZ, sha256_kind=HASH_CONTENT)
    elif fmt == CIFAR10C_FORMAT_TEXT:
        n = condition_test.get("n")
        match = _ROWS_RE.fullmatch(str(condition_test.get("rows")))
        if not isinstance(n, int) or n <= 0 or match is None:
            raise SealedFinalsError(f"{condition_json}: a {FORMAT_CIFAR10C} test records n and rows 'A .. B (severity "
                                    f"S)', got n {n!r}, rows {condition_test.get('rows')!r}")
        start, stop, severity = (int(group) for group in match.groups())
        if stop - start + 1 != n or start != (severity - 1) * n:
            raise SealedFinalsError(f"{condition_json}: rows {start} .. {stop} at severity {severity} are not the "
                                    f"severity block of n {n}")
        if clean_test.get("n") not in (None, n):
            raise SealedFinalsError(f"{condition_json}: test n {n} != the clean test's n {clean_test.get('n')}")
        entry.update(format=FORMAT_CIFAR10C, sha256_kind=HASH_FILE, rows={"start": start, "stop": stop + 1,
                                                                          "severity": severity})
    else:
        raise SealedFinalsError(f"{condition_json}: unknown test format {fmt!r}")
    return {"dataset": dataset, "condition": condition, "condition_test": entry, "clean_test": clean}


# -------------------------------------------------------------------------------------------------- row sources


def load_store_npz(path: Path, sha256: str, n: Any = None) -> dict[str, np.ndarray]:
    """A store-layout .npz, refused unless its canonical content hash is ``sha256`` (and it has ``n`` rows)."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    digest = params_sha256(arrays)
    if digest != sha256:
        raise SealedFinalsError(f"{path}: content hash {digest} != recorded {sha256}")
    if n is not None and any(len(value) != n for value in arrays.values()):
        raise SealedFinalsError(f"{path}: arrays of {sorted({len(v) for v in arrays.values()})} rows, recorded n {n}")
    return arrays


def _vision_clean(entry: Mapping[str, Any], cache: dict[Any, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    key = ("vision clean", entry["path"], entry["sha256"])
    if key not in cache:
        arrays = load_store_npz(Path(entry["path"]), entry["sha256"], entry.get("n"))
        check_image_arrays(arrays, str(entry["path"]))
        cache[key] = arrays, {"verified_sha256": entry["sha256"]}
    return cache[key]


def _vision_condition(entry: Mapping[str, Any], clean: Mapping[str, np.ndarray], cache: dict[Any, Any]
                      ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    key = ("vision condition", entry["path"], entry["sha256"], json.dumps(entry.get("rows"), sort_keys=True))
    if key not in cache:
        path = Path(entry["path"])
        if entry["format"] == FORMAT_STORE_NPZ:
            arrays = load_store_npz(path, entry["sha256"])
            for name in ("y", "idx"):
                if arrays[name].dtype != clean[name].dtype or not np.array_equal(arrays[name], clean[name]):
                    raise SealedFinalsError(f"{path}: {name} differs from the clean test set")
            verified = {"verified_sha256": entry["sha256"]}
        elif entry["format"] == FORMAT_CIFAR10C:
            digest = file_sha256(path)
            if digest != entry["sha256"]:
                raise SealedFinalsError(f"{path}: file sha256 {digest} != recorded {entry['sha256']}")
            start, stop = int(entry["rows"]["start"]), int(entry["rows"]["stop"])
            official = np.load(path, mmap_mode="r", allow_pickle=False)
            if official.dtype != np.uint8 or official.ndim != 4 or official.shape[0] < stop:
                raise SealedFinalsError(f"{path}: {official.dtype}{official.shape} is not an NHWC uint8 file with "
                                        f"rows {start} .. {stop - 1}")
            x = np.ascontiguousarray(np.asarray(official[start:stop]).transpose(0, 3, 1, 2))
            del official
            if x.shape != clean["x"].shape:
                raise SealedFinalsError(f"{path}: rows {start} .. {stop - 1} give x {x.shape}, the clean test "
                                        f"{clean['x'].shape}")
            arrays = {"x": x, "y": clean["y"], "idx": clean["idx"]}
            verified = {"verified_sha256": digest, "rows_content_sha256": params_sha256(arrays)}
        else:
            raise SealedFinalsError(f"{path}: unknown format {entry['format']!r}")
        check_image_arrays(arrays, str(path))
        cache[key] = arrays, verified
    return cache[key]


class RowReader:
    """The rows of one mode: the families' dev rows, or their sealed rows (cached per file within one read)."""

    def __init__(self, mode: str) -> None:
        if mode not in (MODE_DEV, MODE_SEALED):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self._cache: dict[Any, Any] = {}

    def rows(self, fam: Family, instance: Instance) -> tuple[Any, dict[str, Any]]:
        if self.mode == MODE_DEV:
            return fam.dev_rows(instance)
        return fam.sealed_rows(instance, fam.sealed_index(instance), self._cache)


# ---------------------------------------------------------------------------------------------------- code state


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SealedFinalsError(f"git {' '.join(args)} in {repo}: {result.stderr.strip() or result.returncode}")
    return result.stdout


def list_modules(entries: Sequence[Final]) -> tuple[str, ...]:
    """``CODE_MODULES`` and the modules of every family the entries name (each family imported once)."""
    modules = list(CODE_MODULES)
    for readout in dict.fromkeys(entry.readout for entry in entries):
        modules.extend(name for name in family(readout).code_modules if name not in modules)
    return tuple(modules)


def code_files(modules: Sequence[str] = CODE_MODULES) -> dict[str, str]:
    """sha256 of the source file of every module in ``modules``."""
    hashes = {}
    for name in modules:
        module = importlib.import_module(name)
        hashes[name] = file_sha256(Path(str(module.__file__)))
    return hashes


def code_state(code_state_path: Path | None = None, modules: Sequence[str] = CODE_MODULES) -> dict[str, Any]:
    """The readout code's commit and whether its tree is clean, from git or from a store job's git_state.json."""
    if code_state_path is not None:
        state = json.loads(Path(code_state_path).read_text(encoding="utf-8"))
        commit, dirty = state.get("commit"), state.get("dirty")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit) or not isinstance(dirty, bool):
            raise SealedFinalsError(f"{code_state_path}: needs commit (40 hex digits) and dirty (bool)")
        return {"commit": commit, "dirty": dirty, "source": str(code_state_path),
                "source_sha256": file_sha256(Path(code_state_path)), "files": code_files(modules)}
    repo = Path(__file__).resolve().parent
    top = Path(_git(repo, "rev-parse", "--show-toplevel").strip())
    commit = _git(top, "rev-parse", "HEAD").strip()
    status = _git(top, "status", "--porcelain")
    return {"commit": commit, "dirty": bool(status.strip()), "source": f"git {top}", "repo": str(top),
            "files": code_files(modules)}


def _check_list_committed(list_path: Path, state: Mapping[str, Any]) -> str:
    """The list's path in the clean repo of ``state``; refused unless it is tracked there."""
    top = Path(str(state["repo"]))
    resolved = _inside(top, Path(list_path), "the list")
    relative = str(resolved.relative_to(top.resolve()))
    if not _git(top, "ls-files", "--", relative).strip():
        raise SealedFinalsError(f"{list_path}: not tracked in {top}; commit the approved list before the read")
    return relative


def environment(fam: Family) -> dict[str, Any]:
    return {"python": platform.python_version(), "numpy": np.__version__, **fam.environment()}


# ---------------------------------------------------------------------------------------------------- readouts


def points(delta_acc: float) -> float:
    return 100.0 * float(delta_acc)


def load_entry(final: Final) -> tuple[Instance, dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Instance, recovered child, its provenance and the read context of one entry."""
    instance = family(final.readout).load_instance(final.instance_id)
    child, provenance = recover_child(final, instance)
    return entry_instance(final, instance), child, provenance, read_context(final)


def read_final(final: Final, rows: RowReader, batch_size: int = DEFAULT_BATCH_SIZE,
               loaded: tuple[Instance, dict[str, np.ndarray], dict[str, Any], dict[str, Any]] | None = None
               ) -> dict[str, Any]:
    """One record: the entry's child read by its family on the row source's rows (module docstring)."""
    fam = family(final.readout)
    instance, child, provenance, context = load_entry(final) if loaded is None else loaded
    data, rows_record = rows.rows(fam, instance)
    parts, gain, clean = fam.read(instance, child, data, batch_size, context)
    return {
        "schema": RESULT_SCHEMA,
        "mode": rows.mode,
        "block": final.block,
        **final.as_dict(),
        "dataset": instance.private.get("dataset"),
        "condition": instance.private.get("condition"),
        "split_role": instance.private.get("split_role"),
        "seed": instance.private.get("seed"),
        "base_params_sha256": instance.params_sha256,
        "forward_sha256": instance.forward_sha256,
        "child": provenance,
        "rows": rows_record,
        "sealed_index": fam.sealed_index(instance),
        "batch_size": batch_size,
        "environment": environment(fam),
        **parts,
        "gain": gain,
        "clean": clean,
        "joint": None if clean is None else gain + clean,
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dev_reference(final: Final) -> dict[str, Any]:
    """The dev readout of record of ``final``: where it is, which fields it keeps, and its readout parts."""
    fam = family(final.readout)
    kind = final.source["kind"]
    extra: dict[str, Any] = {}
    if kind == SOURCE_RANGE:
        return fam.range_reference(final)  # type: ignore[attr-defined]
    if kind == SOURCE_SESSION:
        path = runs_dir() / "devreadouts" / "learn" / final.source["session_id"] / "info.json"
        if not path.is_file():
            raise SealedFinalsError(f"{final.name}: no dev readout of record {path}")
        info = json.loads(path.read_text(encoding="utf-8"))
        record = info.get("final", {})
        if info.get("instance_id") != final.instance_id or record.get("child_sha256") != final.child_sha256:
            raise SealedFinalsError(f"{final.name}: {path} reads {info.get('instance_id')!r} child "
                                    f"{record.get('child_sha256')!r}, not this final")
        parts, fields = fam.session_parts(record), FIELDS_ALL
    elif kind in _STREAM_KINDS:
        path, stream = _stream_record(final)
        row = stream_round(final, stream)
        entry = row.get("readouts", {}).get(final.instance_id)
        if entry is None or row.get("child_sha256") != final.child_sha256:
            raise SealedFinalsError(f"{final.name}: {path} has no round {row.get('round')!r} readout of "
                                    f"{final.instance_id} with child {final.child_sha256}")
        parts, fields = {"readouts": entry["select"], "readouts_clean": entry["clean"]}, FIELDS_THEIRS
        extra = {"n_checks": {"condition": entry.get("n_select"), "clean": entry.get("n_clean")}}
    elif kind == SOURCE_TRANSFER:
        path, record = transfer_record(final)
        parts, fields = {"readouts": record["readouts"], "readouts_clean": record["readouts_clean"]}, FIELDS_ALL
        extra = {"n_checks": {"condition": record.get("n_select"), "clean": record.get("n_clean")}}
    else:
        root = runs_dir()
        path = _inside(root, root / final.source["dev_record"], "dev_record")
        if not path.is_file():
            raise SealedFinalsError(f"{final.name}: no dev record {path}")
        run_path, run = reference_run(final)
        lines = [line for line in _jsonl(path) if Path(str(line.get("run", ""))).resolve() == run_path]
        if not lines:
            raise SealedFinalsError(f"{final.name}: {path} holds no record of {run_path}")
        if final.source["pick"] == PICK_KEPT:
            parts = fam.reference_parts(lines, final, kept_pick(final, run)["final"]["patch_id"])
        else:
            parts = pick_parts(lines, final, reference_pick(final, run)["final"])
        fields = FIELDS_ALL
    gain, clean = fam.points(parts)
    return {"path": str(path), "file_sha256": file_sha256(path), "fields": fields, "parts": parts, "gain": gain,
            "clean": clean, **extra}


def pick_parts(lines: Sequence[Mapping[str, Any]], final: Final, pick: Mapping[str, Any]) -> dict[str, Any]:
    """The dev record line of an own or candidate pick (vision_joint) among the lines of its run: for own, the run's
    one own-final line (``devreadout.readout_run``: no selection, no patch_id, its V_check numbers the final's); for a
    candidate, the frontier line (``devreadout.readout_candidates``) of the patch with the pick's child hash."""
    on_instance = [line for line in lines if line.get("instance_id") == final.instance_id]
    if final.source["pick"] == PICK_OWN:
        hits = [line for line in on_instance if "selection" not in line and "patch_id" not in line]
        for line in hits:
            vcheck = line.get("vcheck") or {}
            if line.get("is_base_noop") or any(vcheck.get(k) != pick.get(k) for k in ("score", "fixed", "broken")):
                raise SealedFinalsError(f"{final.name}: the own-final line of {final.source['dev_record']} has V_check "
                                        f"{vcheck}, the run's final {pick.get('patch_id')} has another")
    else:
        hits = [line for line in on_instance
                if line.get("patch_id") == pick["patch_id"] and line.get("child_sha256") == final.child_sha256]
    if len(hits) != 1:
        raise SealedFinalsError(f"{final.name}: {len(hits)} {final.source['pick']} lines of its run in "
                                f"{final.source['dev_record']}; expected one")
    return {"readouts": hits[0]["readouts"], "readouts_clean": hits[0]["readouts_clean"]}


def _flatten(value: Any, prefix: str) -> dict[str, Any]:
    if isinstance(value, Mapping) and value:
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}"))
        return flat
    return {prefix: value}


def compare_with_reference(record: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Field-by-field equality of a dev record with its reference (every field, or the fields a stream keeps)."""
    differences: list[str] = []
    for part, theirs_raw in reference["parts"].items():
        ours = _flatten(record.get(part), part)
        theirs = _flatten(devreadout.json_safe(theirs_raw), part)
        keys = sorted(set(ours) | set(theirs)) if reference["fields"] == FIELDS_ALL else sorted(theirs)
        for key in keys:
            if key not in ours or key not in theirs or ours[key] != theirs[key]:
                differences.append(f"{key}: ours {ours.get(key)!r}, record {theirs.get(key)!r}")
    for part, n in (reference.get("n_checks") or {}).items():
        if n is not None and n != record["rows"][part]["n"]:
            differences.append(f"n of {part}: ours {record['rows'][part]['n']}, record {n}")
    return {"path": reference["path"], "file_sha256": reference["file_sha256"], "fields": reference["fields"],
            "gain": reference["gain"], "clean": reference["clean"],
            "joint": None if reference["clean"] is None else reference["gain"] + reference["clean"],
            "match": not differences, "differences": differences}


# ---------------------------------------------------------------------------------------------------- the two modes


def run_dev(list_path: Path, out: Path, batch_size: int = DEFAULT_BATCH_SIZE,
            code_state_path: Path | None = None, only: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The dev dry run: every entry on its dev rows, compared with its readout of record.

    ``only`` names a subset of the list's entries (dev only, so one list can be read by parallel processes, and the
    entries of a family whose code is not in this tree can be left out); each record carries the whole list's hash
    and the code files of the families it reads. The sealed read always reads the whole list.
    """
    parsed, entries = load_list(list_path)
    if only is not None:
        unknown = sorted(set(only) - {entry.name for entry in entries})
        if unknown or not only:
            raise SealedFinalsError(f"--only names entries the list does not have: {unknown or 'none given'}")
        entries = [entry for entry in entries if entry.name in set(only)]
    state = code_state(code_state_path, list_modules(entries))
    list_sha = manifest_sha256(parsed)
    if out.exists():
        raise SealedFinalsError(f"{out} exists; a dev record is never overwritten")
    rows = RowReader(MODE_DEV)
    records = []
    for entry in entries:
        record = read_final(entry, rows, batch_size)
        record.update(list_sha256=list_sha, list_path=str(list_path), list_status=parsed["status"], code=state,
                      reference=compare_with_reference(record, dev_reference(entry)), time=utc_now(),
                      only=None if only is None else sorted(only))
        records.append(record)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return records


def _read_before(registry: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """(instance_id, child hash) of every entry of every started record, of this pipeline or of evalkit.sealed."""
    seen = set()
    for record in registry:
        if record.get("event") != "evaluation_started":
            continue
        for entry in record.get("entries", []):
            child = entry.get("child_sha256") or entry.get("child_params_sha256")
            seen.add((str(entry.get("instance_id")), str(child)))
    return seen


def _vision_rows(instance_id: str) -> tuple[str, str] | None:
    """(dataset, condition) of a vision instance from its private.json; None for another family's instance."""
    if not INSTANCE_PATTERNS[READOUT_VISION_JOINT].fullmatch(instance_id):
        return None
    private = json.loads((instances_root() / instance_id / "private.json").read_text(encoding="utf-8"))
    return str(private["dataset"]), str(private["condition"])


def range_read(entry: Final) -> tuple[str, str, str]:
    """(child, dataset, set) of a range entry: the rows its sealed read opens."""
    rows = _vision_rows(entry.instance_id)
    if rows is None:
        raise SealedFinalsError(f"{entry.name}: {entry.instance_id} is not a vision instance")
    return (entry.child_sha256, rows[0], str(entry.source["set"]))


def _rows_read_before(registry: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    """(child, dataset, condition or set) of every vision entry of every started record: the condition rows each
    vision_joint entry read, the set each range entry read."""
    seen = set()
    for record in registry:
        if record.get("event") != "evaluation_started":
            continue
        for entry in record.get("entries", []):
            rows = _vision_rows(str(entry.get("instance_id")))
            if rows is None:
                continue
            child = str(entry.get("child_sha256") or entry.get("child_params_sha256"))
            source = entry.get("source") or {}
            seen.add((child, rows[0], str(source["set"]) if source.get("kind") == SOURCE_RANGE else rows[1]))
    return seen


def _entry_events(registry: Path, sha: str, name: str) -> list[str]:
    return [r["event"] for r in read_registry(registry)
            if r.get("manifest_sha256") == sha and r.get("name") == name and r.get("kind") == EVENT_KIND]


def read_sealed_entry(final: Final, index: int, sha: str, registry: str, batch_size: int,
                      rows: RowReader | None = None,
                      loaded: tuple[Instance, dict[str, np.ndarray], dict[str, Any], dict[str, Any]] | None = None
                      ) -> dict[str, Any]:
    """One entry of a started sealed read (in a worker process, or in the parent with one worker). It opens no file of
    the entry unless the registry holds this entry's ``entry_started`` and nothing after it."""
    events = _entry_events(Path(registry), sha, final.name)
    if events != [EVENT_ENTRY_STARTED]:
        raise SealedFinalsError(f"{final.name}: the registry records {events} for this entry of {sha}, not one "
                                f"{EVENT_ENTRY_STARTED}; nothing of it is opened")
    record = read_final(final, rows if rows is not None else _worker_rows(), batch_size, loaded=loaded)
    record["list_index"] = index
    return record


_WORKER_ROWS: RowReader | None = None


def _worker_rows() -> RowReader:
    """The sealed row source of a worker process, its cache kept over the entries it reads."""
    global _WORKER_ROWS
    if _WORKER_ROWS is None:
        _WORKER_ROWS = RowReader(MODE_SEALED)
    return _WORKER_ROWS


def _dispatch(entries: Sequence[Final], workers: int, read_one: Callable[..., dict[str, Any]],
              submit_args: Callable[[int, Final], tuple[Any, ...]], on_started: Callable[[int, Final], None],
              on_finished: Callable[[int, Final, dict[str, Any]], None],
              on_failed: Callable[[int, Final, BaseException], None]) -> None:
    """Read every entry once, ``workers`` at a time in spawned processes; ``on_started`` runs before an entry is
    handed over, then ``on_finished`` or ``on_failed`` once."""
    pending = list(enumerate(entries))
    context = multiprocessing.get_context("spawn")
    while pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            running: dict[Future[dict[str, Any]], tuple[int, Final]] = {}
            broken = False
            while (pending or running) and not broken:
                while pending and len(running) < workers:
                    index, entry = pending.pop(0)
                    on_started(index, entry)
                    running[pool.submit(read_one, *submit_args(index, entry))] = (index, entry)
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    index, entry = running.pop(future)
                    try:
                        record = future.result()
                    except BrokenProcessPool as error:
                        broken = True
                        on_failed(index, entry, error)
                    except Exception as error:  # an entry's failure: recorded and spent, the others go on
                        on_failed(index, entry, error)
                    else:
                        on_finished(index, entry, record)
            for index, entry in running.values():  # a broken pool fails every entry whose result did not arrive
                on_failed(index, entry, BrokenProcessPool("the worker pool broke while this entry was being read"))


def run_sealed(list_path: Path, batch_size: int = DEFAULT_BATCH_SIZE, workers: int = DEFAULT_WORKERS
               ) -> tuple[Path, int]:
    """The sealed read of an approved, committed list (module docstring, in that order); returns the results path and
    the number of entries read. Raises after ``evaluation_finished`` when an entry failed."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise SealedFinalsError(f"workers must be an integer >= 1, got {workers!r}")
    parsed, entries = load_list(list_path)
    if parsed["status"] != STATUS_APPROVED:
        raise SealedFinalsError(f"{list_path}: status {parsed['status']!r}; only an approved list is read")
    own_read = sorted({entry.readout for entry in entries
                       if not getattr(family(entry.readout), "generic_sealed", True)})
    if own_read:
        raise SealedFinalsError(f"{list_path}: readouts {own_read} are read only by their own module's sealed read; "
                                "nothing is registered")
    state = code_state(modules=list_modules(entries))
    if state["dirty"]:
        raise SealedFinalsError(f"{state['repo']}: the tree is not clean; the readout code is frozen by commit before "
                                "the read")
    list_relative = _check_list_committed(list_path, state)
    loaded = {}
    for entry in entries:
        fam = family(entry.readout)
        instance = fam.load_instance(entry.instance_id)
        if instance.private.get("split_role") != "dev":
            raise SealedFinalsError(f"{entry.name}: {entry.instance_id} has split_role "
                                    f"{instance.private.get('split_role')!r}; the entries of record are dev instances")
        child, provenance = recover_child(entry, instance)
        instance = entry_instance(entry, instance)
        for path in fam.sealed_paths(fam.sealed_index(instance)):
            if not path.is_file():
                raise SealedFinalsError(f"{entry.name}: no sealed file at {path}")
        loaded[entry.name] = (instance, child, provenance, read_context(entry))
    workers = min(workers, len(entries))
    sha = manifest_sha256(parsed)
    root = sealed_dir()
    registry = root / REGISTRY_NAME
    results_dir, manifests_dir = root / "results", root / "manifests"
    for directory in (results_dir, manifests_dir):
        directory.mkdir(mode=0o700, exist_ok=True)
    results_path = results_dir / f"{sha}.jsonl"
    failed: dict[str, str] = {}
    tests: dict[str, str] = {}
    n_read = 0
    with _exclusive_lock(root / ".sealed.lock"):
        records_so_far = read_registry(registry)
        if any(record["manifest_sha256"] == sha for record in records_so_far):
            raise SealedFinalsError(f"list {sha} was already read or attempted (see {registry})")
        again = sorted({(e.instance_id, e.child_sha256) for e in entries if e.source["kind"] != SOURCE_RANGE}
                       & _read_before(records_so_far))
        ranges = [e for e in entries if e.source["kind"] == SOURCE_RANGE]
        if ranges:
            reads = _rows_read_before(records_so_far)
            again += sorted(key for key in (range_read(e) for e in ranges) if key in reads)
        if again:
            raise SealedFinalsError(f"entries already read in an earlier sealed read: {again}")
        if results_path.exists():
            raise SealedFinalsError(f"{results_path} exists although the registry has no record of {sha}")
        list_copy = manifests_dir / f"{sha}.json"
        shutil.copyfile(list_path, list_copy)
        _append_registry(registry, {
            "event": "evaluation_started", "kind": EVENT_KIND, "manifest_sha256": sha, "time": utc_now(),
            "n_entries": len(entries), "manifest_copy": str(list_copy), "list_path": list_relative,
            "approval": parsed["approval"], "code_commit": state["commit"], "workers": workers,
            "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
            "entries": [{"name": e.name, "block": e.block, "instance_id": e.instance_id,
                         "child_sha256": e.child_sha256, "source": dict(e.source), "readout": e.readout}
                        for e in entries],
        })

        def event(kind: str, index: int, entry: Final, **fields: Any) -> None:
            _append_registry(registry, {"event": kind, "kind": EVENT_KIND, "manifest_sha256": sha, "time": utc_now(),
                                        "name": entry.name, "list_index": index, "instance_id": entry.instance_id,
                                        "child_sha256": entry.child_sha256, **fields})

        try:
            with open(results_path, "x", encoding="utf-8") as handle:
                def finished(index: int, entry: Final, record: dict[str, Any]) -> None:
                    nonlocal n_read
                    record.update(list_sha256=sha, list_path=list_relative, list_status=parsed["status"], code=state,
                                  time=utc_now())
                    line = json.dumps(record, sort_keys=True, allow_nan=False)
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    parts = {part["location"]: part["verified_sha256"] for part in record["rows"].values()}
                    tests.update(parts)
                    n_read += 1
                    event(EVENT_ENTRY_FINISHED, index, entry, record_sha256=hashlib.sha256(line.encode()).hexdigest(),
                          test_sha256=parts)

                def failure(index: int, entry: Final, error: BaseException) -> None:
                    failed[entry.name] = f"{type(error).__name__}: {error}"
                    event(EVENT_ENTRY_FAILED, index, entry, error=failed[entry.name])

                if workers == 1:
                    rows = RowReader(MODE_SEALED)
                    for index, entry in enumerate(entries):
                        event(EVENT_ENTRY_STARTED, index, entry)
                        try:
                            record = read_sealed_entry(entry, index, sha, str(registry), batch_size, rows=rows,
                                                       loaded=loaded[entry.name])
                        except Exception as error:  # an entry's failure: recorded and spent, the others go on
                            failure(index, entry, error)
                        else:
                            finished(index, entry, record)
                else:
                    _dispatch(entries, workers, read_sealed_entry,
                              lambda index, entry: (entry, index, sha, str(registry), batch_size),
                              lambda index, entry: event(EVENT_ENTRY_STARTED, index, entry), finished, failure)
        except BaseException as error:
            _append_registry(registry, {"event": "evaluation_failed", "kind": EVENT_KIND, "manifest_sha256": sha,
                                        "time": utc_now(), "error": f"{type(error).__name__}: {error}",
                                        "n_read": n_read, "failed": dict(failed)})
            raise
        _append_registry(registry, {"event": "evaluation_finished", "kind": EVENT_KIND, "manifest_sha256": sha,
                                    "time": utc_now(), "n_entries": len(entries), "n_read": n_read,
                                    "failed": dict(failed), "results_path": str(results_path),
                                    "results_sha256": file_sha256(results_path), "test_sha256": tests})
    if failed:
        raise SealedEntriesFailed(f"read {n_read} of {len(entries)} entries; failed, recorded and spent: "
                                  + "; ".join(f"{name}: {error}" for name, error in failed.items()),
                                  results_path, n_read)
    return results_path, n_read


def _num(value: float | None, width: int) -> str:
    return f"{'n/a':>{width}}" if value is None else f"{value:>+{width}.3f}"


def _table(records: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"{'entry':<34} {'instance':<10} {'child':<12} {'gain':>8} {'clean':>7} {'joint':>8}   "
             f"{'rec gain':>8} {'rec clean':>9} {'rec joint':>9}  match"]
    for r in records:
        ref = r["reference"]
        lines.append(f"{r['name']:<34} {r['instance_id']:<10} {r['child_sha256'][:12]:<12} {r['gain']:>+8.3f} "
                     f"{_num(r['clean'], 7)} {_num(r['joint'], 8)}   {ref['gain']:>+8.3f} {_num(ref['clean'], 9)} "
                     f"{_num(ref['joint'], 9)}  {ref['match']}")
        lines.extend(f"    {difference}" for difference in ref["differences"])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m weightcode.evalkit.sealed_finals", description=__doc__.split("\n")[0])
    parser.add_argument("--list", required=True, type=Path, help=f"the list of finals ({LIST_SCHEMA})")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", action="store_true",
                      help="dry run on the dev rows of record, compared with the dev readouts of record")
    mode.add_argument("--sealed", action="store_true", help="the one sealed read of an approved, committed list")
    parser.add_argument("--out", type=Path, help="--dev: the JSONL record file (never overwritten)")
    parser.add_argument("--code-state", type=Path,
                        help="--dev in a store job without .git: the job's git_state.json (commit, dirty)")
    parser.add_argument("--only", nargs="+", metavar="NAME", help="--dev: read only these entries of the list")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, help="--sealed: entries read at a time, each in a worker process with "
                                                    f"the environment's BLAS threads (default {DEFAULT_WORKERS})")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.workers is not None and (args.dev or args.workers < 1):
        parser.error("--workers is a --sealed option and must be >= 1")
    if args.dev:
        if args.out is None:
            parser.error("--dev needs --out")
        try:
            records = run_dev(args.list, args.out, args.batch_size, args.code_state, args.only)
        except SealedFinalsError as error:
            print(f"dev dry run refused: {error}", file=sys.stderr)
            return 2
        print(_table(records))
        print(f"records: {args.out} ({len(records)}); all match: {all(r['reference']['match'] for r in records)}")
        return 0 if all(r["reference"]["match"] for r in records) else 1
    if args.out is not None or args.code_state is not None or args.only is not None:
        parser.error("--sealed reads the whole list, writes under $WEIGHTCODE_SEALED and takes the code state from git; "
                     "drop --out, --code-state and --only")
    try:
        results_path, count = run_sealed(args.list, args.batch_size,
                                         DEFAULT_WORKERS if args.workers is None else args.workers)
    except SealedEntriesFailed as error:
        print(f"results: {error.results_path}")
        print(f"entries: {error.n_read}")
        print(f"sealed read finished with failed entries (recorded and spent): {error}", file=sys.stderr)
        return 1
    except SealedFinalsError as error:
        print(f"sealed read refused: {error}", file=sys.stderr)
        return 2
    print(f"results: {results_path}")
    print(f"entries: {count}")
    return 0


__all__ = ["FAMILY_MODULES", "Family", "Final", "RowReader", "SealedEntriesFailed", "SealedFinalsError", "VisionJoint",
           "code_state",
           "compare_with_reference", "dev_reference", "entry_instance", "entry_key", "family", "load_entry",
           "load_list", "main", "range_read", "read_final",
           "read_sealed_entry", "recover_child", "run_dev", "run_sealed", "sealed_index"]


if __name__ == "__main__":
    # Run the imported module's main, so that the family modules and this entry point share one SealedFinalsError.
    from weightcode.evalkit import sealed_finals as _imported

    sys.exit(_imported.main())
