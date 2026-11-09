"""B1 matched random controls for a reference patch: does it matter where an edit lands, given its size and norm?

Definition (PREREG, agreed with the harness session). The reference delta is child - base per tensor in float64, where
the child is the reference's final model (see "References" below).

* ``permuted``: within each tensor the delta values are permuted over all positions of that tensor. The per-tensor
  changed count and the multiset of delta values, hence the Frobenius norm, are kept exactly.
* ``gaussian``: within each tensor, the same changed positions as the reference delta, with i.i.d. standard normal
  values rescaled to the reference tensor's delta Frobenius norm.

A control is 3 runs per kind (run_index 0, 1, 2). Each run draws 8 candidates, each costing one patch-check query from
that run's own budget of 8, and its final is ``PatchCheckOracle.final()`` (argmax S with the contract 5.2
tie-breaks). The reported B1 value of a kind is the mean over its 3 runs. Seeds: the run seed is the first 8 bytes,
read big-endian as an unsigned integer, of sha256(f"{session_id}|{kind}|{run_index}") in UTF-8; candidate j of a run
uses ``numpy.random.default_rng([run_seed, j])`` and consumes the tensors in sorted-name order. If the reference is the
base no-op, each run records that and writes no candidate: the control of a no-op is the no-op. The child is the
float32 cast of base + drawn delta, so a realized delta can differ from the drawn one by rounding (a tiny value can
vanish on a large weight); per-tensor realized counts and norms are recorded next to the reference's.

References. A reference is a run record in the shared run-record format (a run directory holding ``run.json``, or the
run.json itself), or a bare patch (a patch JSON file with its .npy files beside it, or a directory holding
``patch.json``).

* Run record: any run.json the dev readout reads, i.e. ``instance.instance_id``, ``final.is_base_noop``, ``final.dir``,
  ``final.child_sha256`` and optionally ``final.parent`` and ``final.model``; a fitting run of this package
  (``weightcode.baseline_run.v0``) and an editor session of the harness (``weightcode.editor_run.v0``: M2, NOWEIGHT, DECOY) both
  qualify. The final child is rebuilt by ``weightcode.evalkit.devreadout.final_child``, the dev readout's own function,
  which checks it against ``final.child_sha256``. The arm is the record's ``method``. The session id is the record's
  ``session_id`` (harness sessions) or, for a fitting run, f"{instance_id}/{arm}/{run_seed}".
* Bare patch: applied to the base model with the shared executor ``weightcode.patch`` (size limits disabled for this step
  only, every other check kept); the arm and the session id must be given.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import PATCH_SCHEMA, Instance
from weightcode.common.identity import Params, params_sha256
from weightcode.evalkit.devreadout import final_child

from .edits import PATCH_APPLIER, apply_patch_result, edit_stats
from .runner import (
    CandidatePatch,
    Evidence,
    Generator,
    MethodSpec,
    execute_run,
    invalid_reason,
    safe_name,
)

FINAL_CHILD_REBUILDER = "weightcode.evalkit.devreadout.final_child"
"""How the child of a run-record reference is rebuilt (recorded in each control's run.json)."""

N_DRAWS = 8
"""Candidates per control run, one patch-check query each."""

N_RUNS = 3
"""Control runs per kind; the reported B1 value is the mean of their final scores."""

CONTROL_KINDS: tuple[str, ...] = ("permuted", "gaussian")
CONTROL_METHODS = CONTROL_KINDS

CONTROL_SPECS: dict[str, MethodSpec] = {
    "permuted": MethodSpec(
        method="permuted",
        label="matched random control B1: reference delta permuted over positions within each tensor",
        uses_labels_for_fit=False,
        edit_limits_enforced=False,
    ),
    "gaussian": MethodSpec(
        method="gaussian",
        label=(
            "matched random control B1: gaussian values at the reference's changed positions, rescaled to each "
            "tensor's reference delta norm"
        ),
        uses_labels_for_fit=False,
        edit_limits_enforced=False,
    ),
}


@dataclass(frozen=True)
class ResolvedReference:
    """Where a reference argument points: a run record (with its directory) or a bare patch JSON."""

    kind: str
    patch_path: Path | None
    run_record: dict[str, Any] | None
    run_dir: Path | None = None

    @property
    def arm(self) -> str | None:
        """The ``method`` of a run-record reference (a baseline method or a harness arm), None for a bare patch."""
        if self.run_record is None:
            return None
        method = self.run_record.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError(f"reference run record in {self.run_dir} has no method")
        return method

    def session_id(self, instance_id: str) -> str | None:
        """B1 session id of a run-record reference: its ``session_id``, or f"{instance_id}/{arm}/{run_seed}"."""
        if self.run_record is None:
            return None
        recorded = self.run_record.get("session_id")
        if recorded is not None:
            return str(recorded)
        if "run_seed" not in self.run_record:
            raise ValueError(f"reference run record in {self.run_dir} has neither session_id nor run_seed")
        return baseline_session_id(instance_id, str(self.arm), int(self.run_record["run_seed"]))

    def output_label(self) -> str | None:
        """Directory label of the controls of a run-record reference: the arm, plus the session id for a session.

        A fitting run has one run per arm and instance, so its label is the arm. An editor session can repeat on an
        instance, so its label is f"{arm}_{session_id}".
        """
        if self.run_record is None:
            return None
        recorded = self.run_record.get("session_id")
        return str(self.arm) if recorded is None else f"{self.arm}_{recorded}"


@dataclass(frozen=True)
class Reference:
    """A reference patch reduced to its realized per-tensor delta on the instance's base model."""

    source: str
    kind: str
    patch_path: Path | None
    is_noop: bool
    delta: dict[str, np.ndarray]
    child_sha256: str
    applier: str | None
    edit: dict[str, Any]
    run_record_final: dict[str, Any] | None
    run_record_meta: dict[str, Any] | None = None

    def describe(self) -> dict[str, Any]:
        """JSON-ready summary for run records."""
        return {
            "source": self.source,
            "kind": self.kind,
            "patch_path": str(self.patch_path) if self.patch_path is not None else None,
            "is_noop": self.is_noop,
            "child_sha256": self.child_sha256,
            "applier": self.applier,
            "edit": self.edit,
            "run_record_final": self.run_record_final,
            "run_record": self.run_record_meta,
        }


def _run_reference(record: Any, run_dir: Path, instance: Instance) -> ResolvedReference:
    """Check that a run record names this instance and has a final pick; the final child is rebuilt later."""
    if not isinstance(record, dict):
        raise ValueError(f"{run_dir}/run.json is not a JSON object")
    owner = record.get("instance", {}).get("instance_id") if isinstance(record.get("instance"), dict) else None
    if owner != instance.instance_id:
        raise ValueError(f"reference run in {run_dir} belongs to instance {owner!r}, not {instance.instance_id!r}")
    final = record.get("final")
    if not isinstance(final, dict) or not isinstance(final.get("is_base_noop"), bool):
        raise ValueError(
            f"reference run in {run_dir} has no final pick with is_base_noop (status {record.get('status')!r})"
        )
    if not final["is_base_noop"]:
        safe_name(str(final.get("dir")), "final dir")
        if not isinstance(final.get("child_sha256"), str):
            raise ValueError(f"reference run in {run_dir}: final.child_sha256 is missing")
    return ResolvedReference("run_record", None, record, run_dir)


def resolve_reference(path: str | Path, instance: Instance) -> ResolvedReference:
    """Map a reference argument to a run-record reference or a bare patch JSON (see the module docstring)."""
    target = Path(path)
    if target.is_dir():
        if (target / "patch.json").is_file():
            return ResolvedReference("patch_dir", target / "patch.json", None)
        if (target / "run.json").is_file():
            return _run_reference(json.loads((target / "run.json").read_text()), target, instance)
        raise FileNotFoundError(f"{target} holds neither patch.json nor run.json")
    if target.is_file() and target.suffix == ".json":
        payload = json.loads(target.read_text())
        if isinstance(payload, dict) and payload.get("schema") == PATCH_SCHEMA:
            return ResolvedReference("patch_file", target, None)
        if isinstance(payload, dict) and "final" in payload and "instance" in payload:
            return _run_reference(payload, target.parent, instance)
        raise ValueError(f"{target}: neither a patch ({PATCH_SCHEMA}) nor a run record (instance and final)")
    raise FileNotFoundError(f"reference {target} is not a patch JSON file, a patch directory or a run directory")


def load_reference(instance: Instance, path: str | Path) -> Reference:
    """Rebuild the reference's final child (or apply a bare patch without size limits) and keep child - base."""
    resolved = resolve_reference(path, instance)
    base = instance.params
    if resolved.run_record is not None:
        assert resolved.run_dir is not None
        rebuilt = final_child(resolved.run_record, resolved.run_dir, instance)
        child: Params = {key: np.ascontiguousarray(value) for key, value in rebuilt.items()}
        problem = invalid_reason(base, child)
        if problem is not None:
            raise ValueError(f"reference final child of {resolved.run_dir} does not fit the base model: {problem}")
        applier_name = FINAL_CHILD_REBUILDER
        recorded = resolved.run_record["final"].get("edit_size")
        is_count = isinstance(recorded, int) and not isinstance(recorded, bool)
        # The reference run's executor counted its final patch; without that count the delta_encoding prediction is
        # recorded and labelled as such. The controls match changed counts and norms, never the edit size.
        edit = edit_stats(base, child, edit_size=recorded if is_count else None)
        edit["edit_size_from"] = "final.edit_size of the reference run" if is_count else "delta_encoding prediction"
    else:
        assert resolved.patch_path is not None
        result = apply_patch_result(base, resolved.patch_path, instance.params_sha256, enforce_limits=False)
        child = {key: np.ascontiguousarray(value) for key, value in result.child.items()}
        applier_name = PATCH_APPLIER
        edit = edit_stats(base, child, edit_size=result.edit_size)
    delta = {key: child[key].astype(np.float64) - base[key].astype(np.float64) for key in sorted(base)}
    is_noop = not any(np.any(value != 0.0) for value in delta.values())
    record = resolved.run_record
    return Reference(
        source=str(path),
        kind=resolved.kind,
        patch_path=resolved.patch_path,
        is_noop=is_noop,
        delta=delta,
        child_sha256=params_sha256(child),
        applier=applier_name,
        edit=edit,
        run_record_final=None if record is None else record["final"],
        run_record_meta=None if record is None else {
            "schema": record.get("schema"),
            "method": record.get("method"),
            "status": record.get("status"),
            "session_id": record.get("session_id"),
            "run_seed": record.get("run_seed"),
            "edit_limits_enforced": record.get("flags", {}).get("edit_limits_enforced"),
        },
    )


def baseline_session_id(instance_id: str, arm: str, arm_run_seed: int) -> str:
    """Session id of a baseline run for B1 seeding: f"{instance_id}/{arm}/{arm_run_seed}"."""
    return f"{instance_id}/{arm}/{int(arm_run_seed)}"


def control_run_seed(session_id: str, kind: str, run_index: int) -> int:
    """First 8 bytes, big-endian unsigned, of sha256(f"{session_id}|{kind}|{run_index}") in UTF-8."""
    if kind not in CONTROL_KINDS:
        raise ValueError(f"unknown control kind {kind!r}; expected one of {CONTROL_KINDS}")
    digest = hashlib.sha256(f"{session_id}|{kind}|{int(run_index)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def candidate_rng(run_seed: int, candidate: int) -> np.random.Generator:
    """Generator of candidate j of a control run: ``numpy.random.default_rng([run_seed, j])``."""
    return np.random.default_rng([int(run_seed), int(candidate)])


def permuted_delta(delta: Mapping[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Permute each tensor's delta over all of its positions; all-zero tensors stay zero and draw nothing."""
    out: dict[str, np.ndarray] = {}
    for key in sorted(delta):
        value = np.asarray(delta[key], dtype=np.float64)
        if not np.any(value != 0.0):
            out[key] = np.zeros_like(value)
            continue
        flat = value.ravel()
        out[key] = flat[rng.permutation(flat.size)].reshape(value.shape)
    return out


def gaussian_delta(delta: Mapping[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per tensor: i.i.d. standard normal values at the reference's changed positions, rescaled to its delta norm."""
    out: dict[str, np.ndarray] = {}
    for key in sorted(delta):
        value = np.asarray(delta[key], dtype=np.float64)
        positions = np.flatnonzero(value)
        drawn = np.zeros(value.size, dtype=np.float64)
        if positions.size:
            values = rng.standard_normal(positions.size)
            values *= float(np.linalg.norm(value)) / float(np.linalg.norm(values))
            drawn[positions] = values
        out[key] = drawn.reshape(value.shape)
    return out


_DRAWERS = {"permuted": permuted_delta, "gaussian": gaussian_delta}
_HOW = {"permuted": "values permuted over positions", "gaussian": "gaussian values at the reference's positions"}


def _match(reference: Reference, child_edit: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    match: dict[str, dict[str, Any]] = {}
    for key, ref in reference.edit["per_tensor"].items():
        got = child_edit["per_tensor"][key]
        if ref["changed"] or got["changed"]:
            match[key] = {
                "changed_reference": ref["changed"],
                "changed_control": got["changed"],
                "rel_frob_reference": ref["rel_frob"],
                "rel_frob_control": got["rel_frob"],
            }
    return match


def control_generator(instance: Instance, kind: str, reference_path: str | Path, run_seed: int) -> Generator:
    """Candidate generator for one B1 control run of one reference on one instance."""
    if kind not in _DRAWERS:
        raise ValueError(f"unknown control kind {kind!r}; expected one of {CONTROL_KINDS}")

    def generate(evidence: Evidence) -> tuple[list[CandidatePatch], dict[str, Any]]:
        reference = load_reference(instance, reference_path)
        info: dict[str, Any] = {"reference": reference.describe(), "reference_is_noop": reference.is_noop}
        if reference.is_noop:
            return [], info
        base = instance.params
        candidates: list[CandidatePatch] = []
        for draw in range(N_DRAWS):
            drawn = _DRAWERS[kind](reference.delta, candidate_rng(run_seed, draw))
            child = {key: (base[key].astype(np.float64) + drawn[key]).astype(base[key].dtype) for key in base}
            candidates.append(
                CandidatePatch(
                    slot=draw,
                    child=child,
                    config={"candidate": draw, "rng": [run_seed, draw]},
                    hypothesis=(
                        f"{kind} control of reference {reference.child_sha256[:12]}: same per-tensor changed count "
                        f"and Frobenius norm, {_HOW[kind]}"
                    ),
                    predicted_effect="null control: no targeted effect expected beyond what the edit size alone gives",
                    notes={"match": _match(reference, edit_stats(base, child))},
                )
            )
        return candidates, info

    return generate


def control_config(
    kind: str, arm: str, session_id: str, run_index: int, run_seed: int, reference_path: str | Path
) -> dict[str, Any]:
    """Run configuration recorded in run.json."""
    return {
        "b1": {
            "kind": kind,
            "arm": arm,
            "session_id": session_id,
            "run_index": int(run_index),
            "n_runs": N_RUNS,
            "run_seed": int(run_seed),
            "seed_rule": "run_seed = sha256(f'{session_id}|{kind}|{run_index}')[:8] big-endian; candidate j: "
            "numpy.random.default_rng([run_seed, j])",
            "reported_value": "mean of the final S over the run indices of this kind",
        },
        "reference_argument": str(reference_path),
        "n_draws": N_DRAWS,
    }


def run_control(
    instance: Instance,
    kind: str,
    reference_path: str | Path,
    *,
    session_id: str,
    arm: str,
    run_index: int,
    run_dir: Path,
    allow_main: bool = False,
) -> dict[str, Any]:
    """Run one B1 control run (one kind, one run index) against one reference and return its run record.

    A run whose draws are all invalid (for example every patch rejected by the executor) is recorded with status
    "failed" and the executor's message, and raises ``NoValidCandidate``: it is never reported as a no-op.
    """
    if kind not in CONTROL_SPECS:
        raise ValueError(f"unknown control kind {kind!r}; expected one of {CONTROL_KINDS}")
    run_seed = control_run_seed(session_id, kind, run_index)
    return execute_run(
        instance=instance,
        spec=CONTROL_SPECS[kind],
        run_dir=run_dir,
        run_seed=run_seed,
        config=control_config(kind, arm, session_id, run_index, run_seed, reference_path),
        generate=control_generator(instance, kind, reference_path, run_seed),
        allow_main=allow_main,
        require_valid_candidate=True,
    )


def b1_value(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Reported B1 value of one kind: mean final S (and mean accuracy change) over its complete control runs.

    Refuses when a run is not complete (a failed run, such as one whose draws were all rejected, has no B1 value).
    ``n_invalid`` lists the invalid draws per run, which a complete run can still have.
    """
    if not records:
        raise ValueError("no control runs")
    incomplete = [(record.get("config", {}).get("b1", {}).get("run_index"), record.get("status"), record.get("error"))
                  for record in records if record.get("status") != "complete"]
    if incomplete:
        raise ValueError(f"B1 is not reported over incomplete control runs: {incomplete}")
    scores = [int(record["final"]["score"]) for record in records]
    deltas = [record["final"].get("delta_acc") for record in records]
    known = [float(value) for value in deltas if value is not None]
    return {
        "n_runs": len(records),
        "n_invalid": [int(record.get("n_invalid", 0)) for record in records],
        "final_scores": scores,
        "mean_score": float(np.mean(scores)),
        "mean_delta_acc": float(np.mean(known)) if len(known) == len(records) else None,
    }
