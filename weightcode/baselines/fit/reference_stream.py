"""Sequential label-fit references for the E4 stream (PREREG E4, "References": sequential label fits on each round's
64 + 64 examples, once with clean replay only and once with the earlier rounds' examples added;
``scripts/run_reference_stream.py``).

A reference stream runs one E1 label fit (``grad_last``, ``ridge_last`` or ``grad_full``) round after round on the
student of a program stream (``weightcode.harness.program_stream``), with the same rounds, experience, readouts and summary:

1. Experience: ``stream_experience(instance, student)``, the rows a program gets in that round (rule evidence-v1
   drawn for the current student on the round's condition, plus the dataset's replay rows with its predictions).
2. Fit rows. Variant ``replay``: the round's 64 evidence rows and the 64 replay rows. Variant ``cumulative``: the
   same plus the evidence rows of every earlier round, as drawn in that round.
3. Configuration: frozen per condition at the one the method's E1 joint run selected with its 8 checks on a base
   student (``frozen_choices``). By default that is the run on the round's own instance. With ``frozen_from`` it is
   the run on the instance of the same condition in another set of instances: seeds 101 to 105 take seed 100's
   choices, as seed 100's frozen programs run there unchanged. That is the feedback a frozen program got in its LEARN
   session, and nothing is checked in the stream, for a reference as for a program. An E1 run whose final was the
   base no-op makes its round a no-op (``noop_frozen``).
4. The method's E1 candidate generator runs on the current student with the fit rows, and the candidate with the
   frozen configuration is kept. Bounded methods scale it into the norm limit as in E1. It is written as a delta
   patch on the current student and applied by the shared executor under the method's E1 limits (``Limits()``; no
   size limits for ``grad_full``). A refused patch or evidence draw leaves the student unchanged and is recorded.
5. Every instance of the stream is read out on the new student by the program stream's own readout
   (``program_stream._readout``, through ``weightcode.harness.clean.dev_readout_joint``), and the summary is the program
   stream's ``stream_summary``.

Nothing reads V_check, the sealed store or an editor session; main instances are refused. One JSON record per
reference stream, never overwritten; each round's patch and student are kept in its work directory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import weightcode.harness.program_stream as program_stream
from weightcode.baselines.instance_io import Instance, load_instance, write_delta_patch
from weightcode.common.identity import Params, count_changed, params_sha256, save_params
from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE
from weightcode.evalkit.evidence import EVIDENCE_RULE_V1
from weightcode.evalkit.instances import instances_root
from weightcode.evalkit.patch_check import predict
from weightcode.harness import clean as harness_clean
from weightcode.harness.program_stream import E4_MEAN_EXCLUDES, stream_experience, stream_summary
from weightcode.patch import PatchError

from .cli import JOINT_REF_LABEL
from .edits import apply_patch_result, edit_stats
from .grad import GRAD_SPECS, grad_generator
from .ridge import RIDGE_SPECS, ridge_generator
from .runner import (
    CandidatePatch,
    Evidence,
    Generator,
    MethodSpec,
    baseline_run_dir,
    jsonable,
    provenance,
    read_run_record,
    write_json_atomic,
)

SCHEMA = "weightcode.baselines.reference_stream.v0"
METHODS: tuple[str, ...] = ("grad_last", "ridge_last", "grad_full")
VARIANTS: tuple[str, ...] = ("replay", "cumulative")
FROZEN_KEYS: dict[str, tuple[str, ...]] = {"grad_last": ("lr", "steps"), "grad_full": ("lr", "steps"),
                                           "ridge_last": ("lam_relative",)}
"""The configuration fields a frozen choice fixes; ridge's absolute lam follows from the round's features."""
STATUS_OK = program_stream.STATUS_OK
STATUS_NOOP = "noop_frozen"
STATUS_EVIDENCE_REFUSED = program_stream.STATUS_EVIDENCE_REFUSED
STATUS_PATCH_REFUSED = program_stream.STATUS_PATCH_REFUSED
EARLIER_RULE = "earlier rounds' evidence rows as drawn in their round"
FIT_PREDICT_BATCH = 256


class ReferenceStreamError(RuntimeError):
    """A reference stream cannot run as specified."""


@dataclass(frozen=True)
class FrozenChoice:
    """The configuration of one round, frozen from the method's E1 joint run on the round's source instance."""

    instance_id: str
    condition: str
    config: dict[str, Any] | None
    source_instance_id: str
    source: str
    source_final: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _spec(method: str) -> MethodSpec:
    if method not in METHODS:
        raise ReferenceStreamError(f"unknown method {method!r}; expected one of {METHODS}")
    return {**GRAD_SPECS, **RIDGE_SPECS}[method]


def load_dev_instances(instance_ids: Sequence[str]) -> list[Instance]:
    """Distinct dev instances, hash-checked on load; a main instance is refused."""
    if len(set(instance_ids)) != len(instance_ids):
        raise ReferenceStreamError(f"an instance appears twice: {list(instance_ids)}")
    instances = [load_instance(instances_root() / instance_id) for instance_id in instance_ids]
    for instance in instances:
        if instance.private.get("split_role") != "dev":
            raise ReferenceStreamError(f"{instance.instance_id}: reference streams use dev instances only")
    return instances


def load_stream_instances(instance_ids: Sequence[str]) -> list[Instance]:
    """The stream's dev instances, distinct and sharing one base model (as ``program_stream`` requires)."""
    if not instance_ids:
        raise ReferenceStreamError("a stream needs at least one round")
    instances = load_dev_instances(instance_ids)
    if len({instance.params_sha256 for instance in instances}) != 1:
        raise ReferenceStreamError("the stream's instances have different base models; a stream has one student")
    return instances


def sources_by_condition(instances: Sequence[Instance], source_instances: Sequence[Instance]) -> list[Instance]:
    """For each round, the source instance of the same condition (one per condition, same dataset and architecture)."""
    by_condition: dict[str, Instance] = {}
    for source in source_instances:
        condition = str(source.private.get("condition"))
        if condition in by_condition:
            raise ReferenceStreamError(f"{by_condition[condition].instance_id} and {source.instance_id} share the "
                                       f"condition {condition}; frozen choices need one source per condition")
        by_condition[condition] = source
    sources = []
    for instance in instances:
        condition = str(instance.private.get("condition"))
        if condition not in by_condition:
            raise ReferenceStreamError(f"{instance.instance_id}: no frozen-from instance has its condition {condition}")
        source = by_condition[condition]
        for key in ("dataset", "arch"):
            if source.private.get(key) != instance.private.get(key):
                raise ReferenceStreamError(f"{instance.instance_id} and its source {source.instance_id} differ in "
                                           f"{key}: {instance.private.get(key)!r} and {source.private.get(key)!r}")
        sources.append(source)
    return sources


def frozen_choices(method: str, instances: Sequence[Instance], run_seed: int = 0,
                   source_instances: Sequence[Instance] | None = None) -> list[FrozenChoice]:
    """Each round's configuration: the one the method's E1 joint run on the round's source instance kept.

    The source is the round's own instance, or with ``source_instances`` the one of the same condition there
    (``sources_by_condition``). The run must be complete, joint, of this method, and on the source instance's own
    base model; a base no-op final gives ``config`` None.
    """
    _spec(method)
    sources = list(instances) if source_instances is None else sources_by_condition(instances, source_instances)
    choices = []
    for instance, source in zip(instances, sources):
        path = baseline_run_dir(source.instance_id, method, run_seed, JOINT_REF_LABEL) / "run.json"
        if not path.is_file():
            raise ReferenceStreamError(f"{source.instance_id}: no E1 joint {method} run at {path}")
        run = read_run_record(path)
        problems = []
        if run.get("status") != "complete" or "joint" not in run or run.get("method") != method:
            problems.append(f"status {run.get('status')!r}, method {run.get('method')!r}, joint {'joint' in run}")
        if run["instance"]["params_sha256"] != source.params_sha256:
            problems.append("its base model differs from the source instance's")
        if problems:
            raise ReferenceStreamError(f"{path}: not a usable E1 joint run ({'; '.join(problems)})")
        final = run["final"]
        config = None
        if not final["is_base_noop"]:
            (kept,) = [entry for entry in run["candidates"] if entry.get("patch_id") == final["patch_id"]]
            config = {key: kept["config"][key] for key in FROZEN_KEYS[method]}
        choices.append(FrozenChoice(
            instance_id=instance.instance_id, condition=str(instance.private.get("condition")), config=config,
            source_instance_id=source.instance_id, source=str(path),
            source_final={key: final.get(key) for key in ("patch_id", "is_base_noop", "score", "score_new",
                                                          "score_old", "query_index")},
        ))
    return choices


def fit_evidence(instance: Instance, params: Params, experience: Mapping[str, np.ndarray], n_fail: int,
                 earlier: Sequence[Mapping[str, np.ndarray]]) -> Evidence:
    """The fit rows of a round: its evidence rows, then the earlier rounds' evidence rows, then the replay rows.

    ``n_fail``: the failure rows of the round's evidence-v1 draw (its first rows; the random rows can be errors too).
    The earlier rows and the replay rows are the ``n_replay`` part, which the E1 joint fits train on like the
    evidence rows. Predictions are the current student's.
    """
    n_new = int(len(experience["y_new"]))
    if not 0 < n_fail <= n_new or not np.all(experience["pred_new"][:n_fail] != experience["y_new"][:n_fail]):
        raise ReferenceStreamError(f"the round's evidence does not start with {n_fail} failure rows")
    x_parts = [experience["x_new"], *(rows["x_new"] for rows in earlier), experience["x_old"]]
    y_parts = [experience["y_new"], *(rows["y_new"] for rows in earlier), experience["y_old"]]
    x = np.ascontiguousarray(np.concatenate(x_parts))
    y = np.concatenate(y_parts).astype(np.int64)
    replay = {"x_old": experience["x_old"], "y_old": experience["y_old"]}
    for index, rows in enumerate(earlier):
        replay[f"x_earlier_{index}"], replay[f"y_earlier_{index}"] = rows["x_new"], rows["y_new"]
    return Evidence(
        x=x, y=y, n_fail=n_fail, n_rand=n_new - n_fail,
        sha256=params_sha256({"x_new": experience["x_new"], "y_new": experience["y_new"]}),
        base_pred=predict(instance.forward, params, x, FIT_PREDICT_BATCH), rule=EVIDENCE_RULE_V1,
        n_replay=int(len(y) - n_new), replay_sha256=params_sha256(replay),
        replay_rule="replay-v1" if not earlier else f"replay-v1 and {EARLIER_RULE}",
    )


GeneratorFactory = Callable[[str, Instance, int], Generator]
"""(method, instance on the current student, run seed) -> the candidate generator of that round."""


def e1_generator(method: str, instance: Instance, run_seed: int) -> Generator:
    """The method's E1 candidate generator on ``instance`` (whose params are the current student), as in E1 joint."""
    return grad_generator(instance, method, run_seed) if method in GRAD_SPECS else ridge_generator(instance, method)


def _kept(candidates: Sequence[CandidatePatch], method: str, config: Mapping[str, Any]) -> CandidatePatch:
    keys = FROZEN_KEYS[method]
    matches = [c for c in candidates if all(c.config.get(key) == config[key] for key in keys)]
    if len(matches) != 1:
        raise ReferenceStreamError(f"{method}: {len(matches)} candidates match the frozen configuration {dict(config)}")
    return matches[0]


def _round(index: int, method: str, variant: str, choice: FrozenChoice, instance: Instance, params: Params,
           earlier: list[dict[str, np.ndarray]], work: Path, run_seed: int,
           make_generator: GeneratorFactory) -> tuple[Params, dict[str, Any]]:
    """One round on the student ``params`` (module docstring, steps 1 to 4); appends the round's rows to ``earlier``."""
    work.mkdir(mode=0o700)
    spec = _spec(method)
    parent_sha = params_sha256(params)
    record: dict[str, Any] = {"round": index, "instance_id": instance.instance_id,
                              "condition": instance.private.get("condition"), "parent_sha256": parent_sha,
                              "choice": choice.as_dict()}
    try:
        experience, summary = stream_experience(instance, params)
    except ValueError as refusal:  # evidence-v1 has no draw for this student
        record.update(status=STATUS_EVIDENCE_REFUSED, messages=[str(refusal)], child_sha256=parent_sha)
        return params, record
    record["experience"] = summary
    prior = list(earlier) if variant == "cumulative" else []
    earlier.append({"x_new": experience["x_new"], "y_new": experience["y_new"]})
    record["fit_rows"] = {"n_round_evidence": int(len(experience["y_new"])),
                          "n_earlier_evidence": int(sum(len(rows["y_new"]) for rows in prior)),
                          "n_replay": int(len(experience["y_old"])), "earlier_rounds": len(prior)}
    if choice.config is None:
        record.update(status=STATUS_NOOP, child_sha256=parent_sha)
        return params, record
    evidence = fit_evidence(instance, params, experience, int(summary["n_new_fail"]), prior)
    record["evidence"] = evidence.describe()
    current = dataclasses.replace(instance, params=dict(params), params_sha256=parent_sha)
    candidates, generation = make_generator(method, current, run_seed)(evidence)
    kept = _kept(candidates, method, choice.config)
    record.update(candidate={"slot": kept.slot, "config": kept.config, "notes": kept.notes},
                  generation={key: generation.get(key) for key in ("parity_max_abs_logit_diff", "trainable",
                                                                    "n_candidates_norm_scaled")})
    patch_dir = work / "patch"
    write_delta_patch(patch_dir, parent_sha, params, kept.child, kept.hypothesis, kept.predicted_effect)
    try:
        result = apply_patch_result(params, patch_dir / "patch.json", parent_sha,
                                    enforce_limits=spec.edit_limits_enforced)
    except PatchError as refusal:
        record.update(status=STATUS_PATCH_REFUSED, messages=list(refusal.messages), child_sha256=parent_sha)
        return params, record
    child = {name: np.ascontiguousarray(value) for name, value in result.child.items()}
    if result.child_sha256 != params_sha256(kept.child):
        raise ReferenceStreamError(f"round {index}: the executor's child differs from the kept candidate")
    if save_params(work / "student_after.npz", child) != result.child_sha256:
        raise ReferenceStreamError(f"round {index}: the saved student does not hash to the child")
    edit = edit_stats(params, child, edit_size=result.edit_size)
    record.update(status=STATUS_OK, child_sha256=result.child_sha256,
                  edit={key: edit[key] for key in ("n_changed", "n_params", "changed_frac", "edit_size", "edit_frac",
                                                   "within_limits", "violations")},
                  rel_frob={key: stats["rel_frob"] for key, stats in edit["per_tensor"].items() if stats["changed"]})
    return child, record


def run_reference_stream(method: str, variant: str, instance_ids: Sequence[str], out_dir: Path, *,
                         choices: Sequence[FrozenChoice] | None = None, frozen_from: Sequence[str] | None = None,
                         run_seed: int = 0, batch_size: int = DEFAULT_BATCH_SIZE,
                         mean_excludes: Sequence[str] = E4_MEAN_EXCLUDES,
                         make_generator: GeneratorFactory = e1_generator) -> Path:
    """Run one reference stream (module docstring) and write its record into ``out_dir``; never overwrites one.

    ``choices``: the rounds' frozen configurations, by default ``frozen_choices`` on the rounds' own instances, or
    with ``frozen_from`` (instance ids) on the instances of the same conditions there. Pass one of the two at most.
    ``make_generator`` is the E1 generator; tests on stores without a torch-rebuildable model pass a fixed one.
    """
    spec = _spec(method)
    if variant not in VARIANTS:
        raise ReferenceStreamError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if choices is not None and frozen_from is not None:
        raise ReferenceStreamError("pass the frozen choices or the instances they come from, not both")
    instances = load_stream_instances(instance_ids)
    if choices is None:
        sources = None if frozen_from is None else load_dev_instances(frozen_from)
        choices = frozen_choices(method, instances, run_seed, sources)
    choices = list(choices)
    if [choice.instance_id for choice in choices] != list(instance_ids):
        raise ReferenceStreamError("the frozen choices do not follow the stream's rounds")
    base = {name: np.array(value) for name, value in instances[0].params.items()}
    base_sha = instances[0].params_sha256
    identity = json.dumps([method, variant, list(instance_ids), [c.config for c in choices],
                           [c.source_instance_id for c in choices], run_seed])
    stem = f"refstream_{method}_{variant}_{hashlib.sha256(identity.encode()).hexdigest()[:12]}_{base_sha[:12]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path, work = out_dir / f"{stem}.json", out_dir / stem
    if path.exists() or work.exists():
        raise ReferenceStreamError(f"{path} or its work directory exists already; records are never overwritten")
    # Read before the first round, as a fit run reads it at its start: the code and tree the rounds ran with.
    started = {**provenance(), "program_stream_sha256": hashlib.sha256(
        Path(program_stream.__file__).read_bytes()).hexdigest()}
    work.mkdir(mode=0o700)
    n_params = sum(value.size for value in base.values())
    params, rounds, earlier = base, [], []
    for index, (choice, instance) in enumerate(zip(choices, instances), start=1):
        params, record = _round(index, method, variant, choice, instance, params, earlier,
                                work / f"round_{index}", run_seed, make_generator)
        changed = count_changed(base, params)
        record.update(n_changed_vs_base=changed, changed_frac_vs_base=changed / n_params,
                      readouts={other.instance_id: program_stream._readout(other, params, batch_size)
                                for other in instances})
        rounds.append(record)
    stream = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": method, "label": spec.label, "variant": variant,
        "edit_limits_enforced": spec.edit_limits_enforced,
        "instances": [{"round": index, "instance_id": instance.instance_id,
                       "dataset": instance.private.get("dataset"), "condition": instance.private.get("condition"),
                       "model_id": instance.private.get("model_id"), "split_role": instance.private.get("split_role")}
                      for index, instance in enumerate(instances, start=1)],
        "frozen_from": None if frozen_from is None else list(frozen_from),
        "choices": [choice.as_dict() for choice in choices],
        "base_params_sha256": base_sha, "run_seed": run_seed, "batch_size": batch_size, "work_dir": str(work),
        "readout_source": harness_clean.source_name(),
        "rounds": rounds,
        "summary": stream_summary(instances, rounds, mean_excludes),
        "provenance": started,
    }
    write_json_atomic(path, jsonable(stream))
    return path
