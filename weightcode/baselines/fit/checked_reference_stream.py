"""Equal-feedback label-fit references of the loop stream (PREREG "E4 with the teacher in the loop", control (b);
dev only; ``scripts/dev/e4loop_control_b.py``).

A checked reference stream runs one E1 label fit (``grad_last``, ``ridge_last`` or ``grad_full``) round after round on
its own current student. It is a sequential reference stream (``reference_stream``) of variant (b), except that each
round re-selects the fit's configuration by the loop's check and kept rule instead of keeping the one E1 chose on the
base. Every part is composed from the code the loop and the reference streams ran:

1. Rows. Before round r, the evidence that round r draws for its condition on the current student is saved
   (``loop_stream.evidence_for``), as the loop stream saves it. The round's experience is built by the function the
   loop used: round 1 by ``program_stream.stream_experience`` (the loop's round 1 is a program-stream round), round
   r >= 2 by ``loop.loop_experience`` with the evidence the earlier rounds saved, so the earlier rows follow the
   one-image rule of the loop's amendment and carry the current student's predictions.
2. Fit rows, variant (b): the round's evidence rows, then the kept earlier rows round by round, then the replay rows
   (``reference_stream.fit_evidence``).
3. Candidates: the method's E1 generator on the current student (``reference_stream.e1_generator``), its 8 E1
   configurations. Each is written as a delta patch on the current student and applied by the shared executor under
   the method's E1 limits (``Limits()``; no size limits for ``grad_full``), as in E1 and the reference streams. A
   malformed or refused candidate is not checked.
4. Check: every applied candidate is checked once by ``loop.MultiPartOracle`` over ``loop_stream.check_parts``
   (V_check of the round's condition, its clean companion, the V_check of each earlier round's condition), against the
   current student, under the loop's budget ``loop_stream.SELECT_BUDGET``.
5. Kept rule loop-retain-v1 (``loop.loop_kept_rule``, ``loop.eligible``) with the final pick of the loop's control
   (a) (``choose_final`` over the candidates that clear the floors: the highest score, ties to fewer changed
   entries, then to the earlier check; the current student when none has score > 0). Round 1 has no earlier
   condition, so only its clean floor applies (``round_rule``). Under loop-retain-v2 (``kept_rule``,
   ``loop.loop_kept_rule_v2``) the clean floor of round r is the stream's remaining clean budget, -60 of 4000 less
   the kept score_old of this stream's own rounds 1 to r - 1 (a no-op round counts 0); round 1 is selected under
   the rule too, so its floor is the whole budget. Each round records the count (``clean_s_old``).
6. Readouts and summary: every stream instance on the new student (``program_stream._readout``, through
   ``dev_readout_joint``, against the base) and ``program_stream.stream_summary``, as in E4 and the loop.

``loop_round_rows`` rebuilds a loop stream's experience round by round on that loop's own students from its saved
evidence and checks it against the content hashes the loop recorded; the stream compares its own rows with them
(``compare_rows``). Only round 1 shares the student with the loop, so only there must the new and earlier rows agree;
the replay rows agree in every round. Nothing reads the sealed store or an editor session; main instances are refused.
One JSON record per stream, never overwritten; each round's evidence, patches and student stay in its work directory.

PREREG E10's control (b) (a stream of E10 instances, ``weightcode.harness.e10``) adds the label-free BatchNorm references
``bn_adapt`` and ``tent`` (``weightcode.baselines.fit.bnrefs``), each re-selected every round over its 8 configurations in
the same way. Their fit rows are the round's 64 condition inputs alone (``Evidence.deployment``: no earlier rows, no
replay rows, no label read), as in the E10 B references, on the current student. On E10 instances every candidate is
applied under the E10 program limits (``weightcode.evalkit.e10.e10_program_limits``: the teacher's LEARN limits), enforced
for a method whose spec enforces edit limits and lifted for ``grad_full``, as the E10 B references are; E1 streams keep
E1's ``Limits()``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import weightcode.harness.program_stream as program_stream
from weightcode.baselines.instance_io import Instance, write_delta_patch
from weightcode.common.identity import count_changed, load_params, params_sha256, save_params
from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE
from weightcode.evalkit.e10 import e10_program_limits
from weightcode.patch.limits import dense_delta_bytes
from weightcode.evalkit.metrics import Candidate, choose_final
from weightcode.harness import clean, e10, e11, loop, loop_stream
from weightcode.harness.jsonio import read_json
from weightcode.harness.ledger import load_checked_npz
from weightcode.harness.program_stream import E4_MEAN_EXCLUDES, stream_experience, stream_summary
from weightcode.harness.retention import s_old_floor
from weightcode.patch import Limits, PatchError, task_limits

from .bnrefs import BN_REF_SPECS, bn_adapt_generator, tent_generator
from .edits import apply_patch_result, edit_stats
from .reference_stream import (
    GeneratorFactory,
    ReferenceStreamError,
    _spec,
    e1_generator,
    fit_evidence,
    load_stream_instances,
)
from .runner import MethodSpec, invalid_reason, jsonable, provenance, write_json_atomic

SCHEMA = "weightcode.baselines.checked_reference_stream.v0"
E1_METHODS: tuple[str, ...] = ("grad_last", "ridge_last", "grad_full")
LABEL_FREE_METHODS: tuple[str, ...] = tuple(BN_REF_SPECS)
"""PREREG E10's label-free BatchNorm references (bn_adapt, tent): E10 streams only."""
METHODS: tuple[str, ...] = (*E1_METHODS, *LABEL_FREE_METHODS)
VARIANT = "cumulative"  # variant (b) of the E4 sequential label fits, with the loop's one-image rule
LABEL_FREE_VARIANT = "condition inputs only"  # the round's 64 condition inputs, as the E10 B references
STATUS_OK = program_stream.STATUS_OK
STATUS_NOOP = loop_stream.STATUS_NOOP
ROW_GROUPS: dict[str, tuple[str, ...]] = {"new": ("x_new", "split_ids_new"), "old": ("x_old", "split_ids_old"),
                                          "prev": ("x_prev", "split_ids_prev")}
"""The rows of an experience, without labels or predictions: each group's inputs and opaque ids."""
EARLIER_RULE = ("earlier rounds' evidence rows as drawn in their round, one image one view "
                "(weightcode.harness.loop.loop_experience)")


class CheckedReferenceError(ReferenceStreamError):
    """A checked reference stream cannot run as specified, or its rows disagree with the loop's."""


EXTRA_SPECS: dict[str, MethodSpec] = {}
"""Methods registered by other modules (``weightcode.baselines.fit.f4``: PREREG F4's fine-tuning arms). They run through the
same rounds, rows, check and readouts, with their own generators (``make_generator``); they are not control (b)."""


def method_spec(method: str) -> MethodSpec:
    """The method's spec: E1's for the label fits, ``bnrefs.BN_REF_SPECS`` for bn_adapt and tent, ``EXTRA_SPECS``."""
    if method in EXTRA_SPECS:
        return EXTRA_SPECS[method]
    if method in BN_REF_SPECS:
        return BN_REF_SPECS[method]
    return _spec(method)


def stream_generator(method: str, instance: Instance, run_seed: int) -> Any:
    """The candidate generator of ``method`` on ``instance`` (whose params are the current student): E1's for the
    label fits (``e1_generator``), the E10 B references' for bn_adapt and tent."""
    if method == "bn_adapt":
        return bn_adapt_generator(instance)
    if method == "tent":
        return tent_generator(instance, run_seed)
    return e1_generator(method, instance, run_seed)


def round_limits(instance: Instance, n_tensors: int) -> Limits | None:
    """The limits a round's candidates are applied under before ``enforce_limits``: the E10 program limits on an
    E10 instance, else None (E1's ``Limits()``)."""
    if e10.is_e10(instance.private):
        return e10_program_limits(instance.task["limits"], n_tensors,
                                  dense_delta_bytes(v.shape for v in instance.params.values()))
    return None


# ---------------------------------------------------------------------------------------------------------------
# Rows


def round_experience(instance: Instance, params: Mapping[str, np.ndarray],
                     prev: Sequence[tuple[int, Mapping[str, np.ndarray]]]) -> tuple[dict[str, np.ndarray],
                                                                                     dict[str, Any]]:
    """The experience round ``len(prev) + 1`` gets in the loop: ``stream_experience`` in round 1, else
    ``loop.loop_experience`` with the earlier rounds' saved evidence."""
    if not prev:
        return stream_experience(instance, params)
    return loop.loop_experience(instance, params, prev)


def rows_sha256(experience: Mapping[str, np.ndarray]) -> dict[str, str]:
    """For each row group present (``ROW_GROUPS``), the hash of its inputs and ids in order."""
    return {group: params_sha256({key: experience[key] for key in keys})
            for group, keys in ROW_GROUPS.items() if all(key in experience for key in keys)}


def earlier_rows(experience: Mapping[str, np.ndarray], n_earlier: int) -> list[dict[str, np.ndarray]]:
    """The kept earlier rows of a loop experience, one mapping per earlier round in order, in ``fit_evidence``'s form."""
    if n_earlier == 0:
        return []
    rounds = np.asarray(experience["round_prev"])
    return [{"x_new": np.ascontiguousarray(experience["x_prev"][rounds == k]),
             "y_new": np.ascontiguousarray(experience["y_prev"][rounds == k])} for k in range(1, n_earlier + 1)]


def loop_round_rows(record_path: Path) -> list[dict[str, Any]]:
    """Rebuild every round's experience of the loop stream at ``record_path`` on that loop's own student before the
    round, from the evidence its earlier rounds saved, and check it against the loop's records.

    Checked per round: the student file hashes as recorded; ``evidence_for`` on it gives the saved evidence file; the
    rebuilt experience has the content hash the loop recorded (a round-1 program round and a checked-selection round
    carry it in the stream record, a teacher round in its session's session.json). Returns per round the content hash
    and ``rows_sha256``. Raises ``CheckedReferenceError`` on the first disagreement.
    """
    record = read_json(Path(record_path))
    ids = [str(item["instance_id"]) for item in record["instances"]]
    instances = load_stream_instances(ids)
    out: list[dict[str, Any]] = []
    prev: list[tuple[int, dict[str, np.ndarray]]] = []
    for row, instance in zip(record["rounds"], instances):
        k = int(row["round"])
        student = load_params(Path(row["student_before"]["path"]))
        if params_sha256(student) != row["student_before"]["sha256"]:
            raise CheckedReferenceError(f"{record_path} round {k}: the saved student does not hash as recorded")
        saved_file = row["prev_item"]["evidence"]
        saved = load_checked_npz(Path(saved_file["path"]), saved_file["sha256"], f"{record_path} round {k} evidence")
        drawn = loop_stream.evidence_for(instance, student)
        if sorted(saved) != sorted(drawn) or not all(np.array_equal(saved[key], drawn[key]) for key in drawn):
            raise CheckedReferenceError(f"{record_path} round {k}: evidence_for does not give the saved evidence")
        experience, summary = round_experience(instance, student, prev)
        if "experience" in row:
            expected = row["experience"]["content_sha256"]
            source = "stream record"
        else:
            session = Path(row["sessions"][-1]["root"]) / "session.json"
            expected = read_json(session)["experience"]["content_sha256"]
            source = str(session)
        content = params_sha256(experience)
        if content != expected or summary["content_sha256"] != expected:
            raise CheckedReferenceError(f"{record_path} round {k}: the rebuilt experience hashes to {content}, the "
                                        f"loop recorded {expected} ({source})")
        out.append({"round": k, "instance_id": instance.instance_id, "student_sha256": row["student_before"]["sha256"],
                    "content_sha256": content, "content_source": source, "rows_sha256": rows_sha256(experience)})
        prev.append((k, saved))
    return out


def compare_rows(index: int, ours: Mapping[str, Any], theirs: Mapping[str, Any], same_student: bool) -> dict[str, Any]:
    """This round's rows against the loop's round ``index``. The replay rows must agree in every round; on the same
    student (round 1) every row group and the whole content must agree. Raises ``CheckedReferenceError`` otherwise."""
    groups = sorted(set(ours["rows_sha256"]) | set(theirs["rows_sha256"]))
    same = {group: ours["rows_sha256"].get(group) == theirs["rows_sha256"].get(group) for group in groups}
    result = {"loop_round": int(theirs["round"]), "loop_student_sha256": theirs["student_sha256"],
              "same_student": same_student, "loop_rows_sha256": dict(theirs["rows_sha256"]), "same_rows": same,
              "same_content": ours["content_sha256"] == theirs["content_sha256"]}
    if int(theirs["round"]) != index or not same.get("old", False):
        raise CheckedReferenceError(f"round {index}: the replay rows are not the loop's")
    if same_student and not (all(same.values()) and result["same_content"]):
        raise CheckedReferenceError(f"round {index}: on the loop's own student the experience is not the loop's")
    return result


# ---------------------------------------------------------------------------------------------------------------
# One round


def round_rule(n_rows: Mapping[str, int], n_earlier: int, kept_rule: str = loop.KEPT_RULE_LOOP,
               clean_used: Mapping[int, int] | None = None) -> dict[str, Any]:
    """The kept rule of a round: ``loop.loop_kept_rule`` when there are earlier rounds; in round 1, the same rule with
    its clean floor only (no earlier condition exists). Under loop-retain-v2, ``loop.loop_kept_rule_v2`` with the
    kept score_old of the earlier rounds ``clean_used`` (none in round 1)."""
    if kept_rule == loop.KEPT_RULE_LOOP_V2:
        return loop.loop_kept_rule_v2(int(n_rows["old"]), {k: int(n_rows[loop.prev_part(k)])
                                                           for k in range(1, n_earlier + 1)}, dict(clean_used or {}))
    if kept_rule != loop.KEPT_RULE_LOOP or clean_used is not None:
        raise CheckedReferenceError(f"the kept rule is one of {loop.KEPT_RULES} (v1 counts no earlier score_old), "
                                    f"not {kept_rule!r}")
    if n_earlier:
        return loop.loop_kept_rule(int(n_rows["old"]), {k: int(n_rows[loop.prev_part(k)])
                                                        for k in range(1, n_earlier + 1)})
    return {"name": loop.KEPT_RULE_LOOP, "clean_frac": float(loop.LOOP_CLEAN_FRAC),
            "prev_frac": float(loop.LOOP_PREV_FRAC), "n_old": int(n_rows["old"]), "n_prev": {}, "check_set": "clean",
            "s_old_floor": s_old_floor(loop.LOOP_CLEAN_FRAC, int(n_rows["old"])), "s_prev_floor": {}}


def _check_fields(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items()
            if key.startswith(("score", "fixed", "broken")) or key in ("query_index", "n_changed", "child_sha256")}


def checked_round(index: int, method: str, instance: Instance, params: dict[str, np.ndarray],
                  earlier: Sequence[Instance], prev: Sequence[tuple[int, Mapping[str, np.ndarray]]], work: Path,
                  run_seed: int, make_generator: GeneratorFactory, kept_rule: str = loop.KEPT_RULE_LOOP,
                  clean_used: Mapping[int, int] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any],
                                                                        dict[str, np.ndarray]]:
    """Round ``index`` on the current student ``params`` (module docstring, steps 2 to 5); returns the next student,
    the round record and the round's experience (for the row comparison). ``clean_used``: loop-retain-v2's kept
    score_old of rounds 1 to ``index - 1``."""
    if len(earlier) != index - 1 or [k for k, _ in prev] != list(range(1, index)):
        raise CheckedReferenceError(f"round {index} needs the instances and evidence of rounds 1 to {index - 1}")
    spec = method_spec(method)
    limits = round_limits(instance, len(params))
    work.mkdir(mode=0o700)
    parent_sha = params_sha256(params)
    experience, summary = round_experience(instance, params, prev)
    record: dict[str, Any] = {
        "round": index, "instance_id": instance.instance_id, "condition": instance.private.get("condition"),
        "parent_sha256": parent_sha,
        "experience": {key: value for key, value in summary.items() if key not in ("prev_idx", "new_source_idx")}
        | {"rows_sha256": rows_sha256(experience)},
    }
    prior = earlier_rows(experience, index - 1)
    evidence = fit_evidence(instance, params, experience, int(summary["n_new_fail"]), prior)
    record["fit_rows"] = {"variant": VARIANT, "earlier_rule": EARLIER_RULE,
                          "n_round_evidence": int(len(experience["y_new"])),
                          "n_earlier_evidence": int(sum(len(rows["y_new"]) for rows in prior)),
                          "earlier_per_round": {str(k): int(len(rows["y_new"])) for k, rows in enumerate(prior, 1)},
                          "n_replay": int(len(experience["y_old"])), "earlier_rounds": len(prior)}
    if method in LABEL_FREE_METHODS:  # the round's condition inputs alone, as the E10 B references
        evidence = evidence.deployment()
        record["fit_rows"] = {"variant": LABEL_FREE_VARIANT, "n_round_evidence": int(len(experience["y_new"])),
                              "n_earlier_evidence": 0, "n_replay": 0, "labels_read": False}
    record["evidence"] = evidence.describe()
    record["limits"] = task_limits(Limits() if limits is None else limits)
    record["edit_limits_enforced"] = spec.edit_limits_enforced
    current = dataclasses.replace(instance, params=dict(params), params_sha256=parent_sha)
    candidates, generation = make_generator(method, current, run_seed)(evidence)
    if not 0 < len(candidates) <= loop_stream.SELECT_BUDGET:
        raise CheckedReferenceError(f"round {index}: {len(candidates)} candidates for a budget of "
                                    f"{loop_stream.SELECT_BUDGET} checks")
    record["generation"] = {key: generation.get(key) for key in ("parity_max_abs_logit_diff", "trainable",
                                                                  "n_candidates_norm_scaled", "base_evidence_loss",
                                                                  "target_scale_c", "lam_unit_mean_sq_feature_norm",
                                                                  "arm_fit_rows")
                            if key in generation}
    oracle = loop.MultiPartOracle(loop_stream.check_parts(instance, earlier), instance.forward, params,
                                  loop_stream.SELECT_BUDGET)
    rule = round_rule(oracle.n_rows(), index - 1, kept_rule, clean_used)
    record["kept_rule"] = rule
    record["check_parts"] = oracle.n_rows()
    rows: list[dict[str, Any]] = []
    children: dict[str, dict[str, np.ndarray]] = {}
    for candidate in candidates:
        name = f"q{candidate.slot}"
        row: dict[str, Any] = {"patch_id": name, "slot": candidate.slot, "config": candidate.config,
                               "notes": candidate.notes}
        rows.append(row)
        problem = invalid_reason(params, candidate.child)
        if problem is not None:
            row.update(status="invalid", messages=[problem])
            continue
        patch_dir = work / name
        write_delta_patch(patch_dir, parent_sha, params, candidate.child, candidate.hypothesis,
                          candidate.predicted_effect)
        try:
            result = apply_patch_result(params, patch_dir / "patch.json", parent_sha,
                                        enforce_limits=spec.edit_limits_enforced, limits=limits)
        except PatchError as refusal:  # as in E1: a refused patch consumes no check
            row.update(status="patch_refused", messages=list(refusal.messages))
            continue
        child = {key: np.ascontiguousarray(value) for key, value in result.child.items()}
        edit = edit_stats(params, child, edit_size=result.edit_size)
        entry = oracle.query(child, name)
        children[name] = child
        row.update(status="checked", child_sha256=result.child_sha256, n_changed=int(result.n_changed),
                   roundtrip_exact=result.child_sha256 == params_sha256(candidate.child),
                   edit={key: edit[key] for key in ("n_changed", "changed_frac", "edit_size", "edit_frac",
                                                    "within_limits", "violations")},
                   rel_frob={key: stats["rel_frob"] for key, stats in edit["per_tensor"].items() if stats["changed"]},
                   eligible=loop.eligible(entry, rule), check=_check_fields(entry))
    record["candidates"] = rows
    record["queries_used"] = loop_stream.SELECT_BUDGET - oracle.queries_remaining
    pool = [Candidate(patch_id=row["patch_id"], score=int(row["check"]["score"]), n_changed=int(row["n_changed"]),
                      query_index=int(row["check"]["query_index"]))
            for row in rows if row["status"] == "checked" and row["eligible"]]
    chosen = choose_final(pool)
    record["n_eligible"] = len(pool)
    if chosen.patch_id is None:
        record.update(status=STATUS_NOOP, kept=None, kept_config=None, child_sha256=parent_sha)
        return params, record, experience
    child = children[str(chosen.patch_id)]
    kept = next(row for row in rows if row["patch_id"] == chosen.patch_id)
    if save_params(work / "student_after.npz", child) != kept["child_sha256"]:
        raise CheckedReferenceError(f"round {index}: the saved student does not hash to the kept child")
    record.update(status=STATUS_OK, kept=str(chosen.patch_id), kept_config=kept["config"],
                  child_sha256=kept["child_sha256"], kept_check=kept["check"], rel_frob=kept["rel_frob"])
    return child, record, experience


# ---------------------------------------------------------------------------------------------------------------
# The stream


def _torch_threads() -> dict[str, Any]:
    import torch

    return {"version": torch.__version__, "threads": torch.get_num_threads()}


def run_checked_reference_stream(method: str, instance_ids: Sequence[str], out_dir: Path, *,
                                 loop_rows: Sequence[Mapping[str, Any]] | None = None,
                                 loop_record: Path | None = None, run_seed: int = 0,
                                 batch_size: int = DEFAULT_BATCH_SIZE,
                                 mean_excludes: Sequence[str] = E4_MEAN_EXCLUDES,
                                 make_generator: GeneratorFactory = stream_generator,
                                 kept_rule: str = loop.KEPT_RULE_LOOP, allow_e11: bool = False) -> Path:
    """Run one checked reference stream (module docstring) and write its record into ``out_dir``; never overwrites.

    ``loop_rows``: ``loop_round_rows`` of the loop stream to compare the rows with (``compare_rows``), round by round;
    ``loop_record`` names that record in the stream record. ``make_generator`` is ``stream_generator``; tests on
    stores without a torch-rebuildable model pass a fixed one. ``kept_rule``: loop-retain-v1 or loop-retain-v2.
    ``allow_e11``: lets E11 (ViT-Lite) instances through, for a registered ``EXTRA_SPECS`` method whose generator
    rebuilds the student (PREREG F4); control (b)'s own E1 generators still refuse them.
    """
    spec = method_spec(method)
    if method not in METHODS and method not in EXTRA_SPECS:
        raise CheckedReferenceError(f"control (b) re-selects {METHODS}, not {method!r}")
    if kept_rule not in loop.KEPT_RULES:
        raise CheckedReferenceError(f"the kept rule is one of {loop.KEPT_RULES}, not {kept_rule!r}")
    v2 = kept_rule == loop.KEPT_RULE_LOOP_V2
    if not 2 <= len(instance_ids) <= loop.MAX_ROUNDS:
        raise CheckedReferenceError(f"a loop stream has 2 to {loop.MAX_ROUNDS} rounds, not {len(instance_ids)}")
    instances = load_stream_instances(instance_ids)
    if any(e11.is_e11(instance.private) for instance in instances) and not (allow_e11 and method in EXTRA_SPECS):
        raise CheckedReferenceError("an E11 (ViT-Lite) control (b) needs the E11 reference generators (the ViT torch "
                                    "rebuild, the E11 grids, tent_ln) and a grid choice; this module has neither")
    kinds = {e10.is_e10(instance.private) for instance in instances}
    if len(kinds) != 1:
        raise CheckedReferenceError("a stream's instances are all E10 instances or none")
    if method in LABEL_FREE_METHODS and kinds != {True}:
        raise CheckedReferenceError(f"{method} is a PREREG E10 BatchNorm reference; it runs on E10 instances only")
    clean_rows_sha = loop_stream.shared_clean_rows(instances) if v2 else None
    if loop_rows is not None:
        loop_rows = list(loop_rows)
        if [row["instance_id"] for row in loop_rows] != list(instance_ids):
            raise CheckedReferenceError("the loop rows do not follow the stream's rounds")
    base = {name: np.array(value) for name, value in instances[0].params.items()}
    base_sha = instances[0].params_sha256
    identity = json.dumps([SCHEMA, method, VARIANT, list(instance_ids), run_seed, loop.KEPT_RULE_LOOP,
                           loop_stream.SELECT_BUDGET] if not v2 else
                          [SCHEMA, method, VARIANT, list(instance_ids), run_seed, kept_rule,
                           loop.LOOP_CLEAN_BUDGET_FRAC, loop_stream.SELECT_BUDGET])
    stem = f"ctrlb_{method}_{hashlib.sha256(identity.encode()).hexdigest()[:12]}_{base_sha[:12]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path, work = out_dir / f"{stem}.json", out_dir / stem
    if path.exists() or work.exists():
        raise CheckedReferenceError(f"{path} or its work directory exists already; records are never overwritten")
    started = {**provenance(), "program_stream_sha256": hashlib.sha256(
        Path(program_stream.__file__).read_bytes()).hexdigest(), "loop_sha256": hashlib.sha256(
        Path(loop.__file__).read_bytes()).hexdigest(), "loop_stream_sha256": hashlib.sha256(
        Path(loop_stream.__file__).read_bytes()).hexdigest(), "torch": _torch_threads()}
    work.mkdir(mode=0o700)
    n_params = sum(value.size for value in base.values())
    params: dict[str, np.ndarray] = base
    rounds: list[dict[str, Any]] = []
    prev: list[tuple[int, dict[str, np.ndarray]]] = []
    used: dict[int, int] = {}  # loop-retain-v2: this stream's kept score_old, round by round
    for index, instance in enumerate(instances, start=1):
        before = loop_stream._student_file(work / f"student_before_round_{index}.npz", params)
        evidence_file = loop_stream._evidence_file(work / f"evidence_round_{index}.npz",
                                                   loop_stream.evidence_for(instance, params))
        child, record, experience = checked_round(index, method, instance, params, instances[:index - 1], prev,
                                                  work / f"round_{index}", run_seed, make_generator, kept_rule,
                                                  dict(used) if v2 else None)
        if v2:
            s_old = 0 if record["status"] == STATUS_NOOP else int(record["kept_check"]["score_old"])
            record["clean_s_old"] = {"round": index, "s_old": s_old,
                                     "source": {"kind": "noop" if record["status"] == STATUS_NOOP
                                                else "control_check", "kept": record["kept"]}}
            used[index] = s_old
        if loop_rows is not None:
            ours = {"rows_sha256": record["experience"]["rows_sha256"],
                    "content_sha256": params_sha256(experience)}
            theirs = loop_rows[index - 1]
            record["loop_rows"] = compare_rows(index, ours, theirs, theirs["student_sha256"] == before["sha256"])
        changed = count_changed(base, child)
        record.update(student_before=before, prev_item={"round": index, "instance_id": instance.instance_id,
                                                        "evidence": evidence_file},
                      n_changed_vs_base=changed, changed_frac_vs_base=changed / n_params,
                      readouts={other.instance_id: program_stream._readout(other, child, batch_size)
                                for other in instances})
        record["student_after"] = loop_stream._student_file(work / f"student_after_round_{index}.npz", child)
        rounds.append(record)
        prev.append((index, load_checked_npz(Path(evidence_file["path"]), evidence_file["sha256"],
                                             f"round {index} evidence")))
        params = child
    stream = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "control": ("E10 control (b): equal-feedback references, re-selected each round by " + kept_rule
                    if kinds == {True} else
                    "E4-loop control (b): equal-feedback label fits, re-selected each round by " + kept_rule),
        "method": method, "label": spec.label,
        "variant": LABEL_FREE_VARIANT if method in LABEL_FREE_METHODS else VARIANT,
        "edit_limits_enforced": spec.edit_limits_enforced,
        "limits_source": ("e10_program_limits(task.json limits, n_tensors)" if kinds == {True} else "Limits()"),
        "check_budget_per_round": loop_stream.SELECT_BUDGET,
        "instances": [{"round": index, "instance_id": instance.instance_id,
                       "dataset": instance.private.get("dataset"), "condition": instance.private.get("condition"),
                       "model_id": instance.private.get("model_id"), "split_role": instance.private.get("split_role")}
                      for index, instance in enumerate(instances, start=1)],
        "base_params_sha256": base_sha, "run_seed": run_seed, "batch_size": batch_size, "work_dir": str(work),
        "readout_source": clean.source_name(),
        "loop_record": None if loop_record is None else str(loop_record),
        "rounds": rounds,
        "summary": stream_summary(instances, rounds, mean_excludes),
        "provenance": started,
    }
    if v2:
        stream["kept_rule"] = {
            "name": kept_rule, "clean_budget_frac": loop.LOOP_CLEAN_BUDGET_FRAC,
            "clean_budget_floor": rounds[0]["kept_rule"]["clean_budget_floor"], "clean_rows_sha256": clean_rows_sha,
            "clean_used": [row["clean_s_old"] for row in rounds], "clean_used_total": int(sum(used.values())),
            "s_old_floor": {str(row["round"]): row["kept_rule"]["s_old_floor"] for row in rounds},
        }
    write_json_atomic(path, jsonable(stream))
    return path


__all__ = ["CheckedReferenceError", "E1_METHODS", "LABEL_FREE_METHODS", "METHODS", "SCHEMA", "checked_round",
           "compare_rows", "earlier_rows", "loop_round_rows", "method_spec", "round_experience", "round_limits",
           "round_rule", "rows_sha256", "run_checked_reference_stream", "stream_generator"]
