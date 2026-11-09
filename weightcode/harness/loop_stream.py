"""Run a loop stream (PREREG E4-loop, dev only) or its checked-selection control, round after round, on one student.

Both modes share round 1 and the bookkeeping:

1. Round 1 runs a frozen program on the base student, exactly as a program-stream round
   (``weightcode.harness.program_stream._round``). Its library entry is that program with the scores of its own session's
   kept check (``round_one_entry``), which was a check on this same base student. When that session's final is the
   base (PREREG E10: round 1 is the first condition's B final, whatever it is), round 1 is given no program, keeps
   the base and enters the library as a round that kept no update (``round_one_noop_entry``). Instead of running the
   program, round 1 can take a recorded child of it (``round1_from_final``: the session's saved final, re-applied to
   the base from ``final/patch.json``, ``round_one_from_final``; ``round1_from_record``: round 1's student of an
   earlier loop stream, ``round_one_from_record``), so that a stream and its control share one exact round-1 student
   whatever the machine did when the program ran. Round 1's record and the stream record name the source
   (``round1_source``); a recorded source enters the stream's identity, so the default stream keeps its stem.
2. Before round r, the stream saves the current student and the evidence round r draws for its condition
   (``evidence_for``: evidence-v1 on the current student, with the dataset idx of every row). Later rounds get these
   rows as the earlier rounds' evidence.
3. Round r >= 2:
   - mode ``teacher``: one LEARN session (``weightcode.harness.runner.run_session`` with a loop context file,
     ``weightcode.harness.loop``) on round r's instance and the current student; the session's ``final/model.npz``, a
     child of that student, is the next student. A session that ends INFRA is run once more on the same context;
     a second INFRA keeps the student unchanged. A session aborted before launch stops the stream, which resumes
     from that round once the fault is fixed. The kept program and its part scores become round r's library entry.
   - mode ``checked_selection`` (control (a), no LLM call): every candidate frozen program of round r's condition is
     run on the current student and its loop experience, its update is applied under the LEARN limits and checked by
     the same ``MultiPartOracle`` (budget 8), and the kept child is the argmax score over the candidates that clear
     both floors of loop-retain-v1, else the current student (``select_round``). A round given no candidate
     (``ROUND=none``: PREREG E10's control (a) when round r's B final is the base) keeps the current student and
     spends no check.
4. After every round, each stream instance is read out on the current student against the base
   (``program_stream._readout``), and the summary is ``program_stream.stream_summary``, as in E4.

Kept rule (``kept_rule``): loop-retain-v1 by default. Under loop-retain-v2 (``weightcode.harness.loop.loop_kept_rule_v2``)
round r's clean floor is the stream's remaining clean budget, -60 of the 4000 clean V_check rows less the kept
score_old of rounds 1 to r - 1, counted by the stream round by round (``clean_s_old`` in each round record):

- Round 1 counts its session's kept score_old when that check scored the same clean rows on the same base and its
  child is the stream's round-1 student (``ROUND1_SESSION_CHECK``); otherwise the stream counts it with the loop's
  check on its own clean part (``ROUND1_LOOP_ORACLE``). A round 1 that kept the base counts 0.
- A teacher round counts its session final's score_old, a control (a) round its kept candidate's, a no-op round 0.
- Every round's clean part must be the same rows (``shared_clean_rows``), so the sum is the stream's net clean change
  on them against the base. A teacher round's context states the counts, the budget and the floor; the session
  computes the same rule from them (``loop.context_kept_rule``), and the stream checks that it did.

The v2 rule enters the stream's identity, so a v1 stream keeps its stem, its context files and its records.

Every round's student, evidence, context and record stay in the stream's work directory, and ``state.json`` lists
the finished rounds, so a stream that stopped resumes with ``resume=True`` from its next round. On resume every kept
file is hash-checked, files of the unfinished round are reused only when they equal what the round would write, a
partial round directory is moved aside, and a teacher round whose session already finished is not launched again
(``round_<r>_kept.json``). The stream record is written once, never overwritten. Nothing reads V_select except the
readouts, and nothing reads the sealed store.

A teacher dry run (``dry_run=True``) runs the rounds before the first teacher round as above, then prepares that
round's session as a dry run (trial, prompts, leak scan and sandbox self-test under ``$WEIGHTCODE_RUNS/dry_runs``, no
Claude Code) and stops; it writes ``dry_run_round_<r>.json`` (the session id and status) instead of a stream record.

Edit log (PREREG T16; ``edit_log=True``, teacher mode only): before teacher round r the stream writes
``edit_log_round_<r>.json`` in its work directory, derived from its saved students of rounds 1 to r - 1 (for each
round, the entries where student_after differs from student_before, ``loop.edit_log_document``; each round's count must
equal that round's changed-entry count), and names it in round r's context (``edit_log``: path, sha256, per-round
counts). The session copies it into its trial as edit_log.json and adds the one PREREG sentence to its system prompt
(``weightcode.harness.loop``). After the session the stream checks that session.json ``loop.edit_log`` names the same file
(arm L) or says ``enabled`` false (arm N), and records it in the round record; the stream record states the arm
(``edit_log``). The option enters the stream's identity, so arm N keeps the stem it would have had before T16.
``tag`` (a short [a-z0-9_] label, e.g. one T16 session) also enters the identity, so repeated streams with the same
arguments get their own stems; it reaches no session. ``session_kwargs["api_key_file"]`` (teacher mode) bills every
session to that key file (``weightcode.harness.launch``); the stream record states the billing mode and the file's path,
never the key.

Protected-entry limit (T16 follow-up M; ``protect_logged=True``, with ``edit_log`` only): each teacher round's context
also names the limit (``protect_context``: the edit log's sha256 and the number of entries it lists over all earlier
rounds), so the session refuses every update that changes one of those entries and states the limit in one sentence
(``weightcode.harness.loop``). After the session the stream checks session.json ``loop.protect_logged`` against its record
(``check_session_protect``) and records it in the round record with the number of attempts the limit refused; the
dry-run record and the stream record carry it too. The option enters the stream's identity; a stream without it writes
no ``protect_logged`` key anywhere, and its sessions must state none.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance
from weightcode.common.identity import count_changed, load_params, params_sha256, save_params
from weightcode.evalkit.devreadout import DEFAULT_BATCH_SIZE
from weightcode.evalkit.metrics import Candidate, choose_final
from weightcode.harness import clean, e11, loop
from weightcode.harness.arms import applied_limits, limits_dict
from weightcode.harness.experience import EXPERIENCE_FILE
from weightcode.harness.finalize import (
    PARENT_INSTANCE,
    STATUS_ABORTED_LEAK,
    STATUS_ABORTED_SELFTEST,
    STATUS_DRY_RUN,
    STATUS_INFRA,
)
from weightcode.harness.jsonio import file_sha256, read_json, utc_now, write_json
from weightcode.harness.launch import billing_record
from weightcode.harness.ledger import load_checked_npz
from weightcode.harness.program import ProgramProblem, run_update, write_derived_patch
from weightcode.harness.program_stream import (
    E4_MEAN_EXCLUDES,
    STATUS_OK,
    _load_stream,
    _program,
    _readout,
    _round,
    stream_experience,
    stream_summary,
)
from weightcode.harness.retention import s_old_floor
from weightcode.harness.transfer import LEARN_ARM, stage_student
from weightcode.patch import PatchError, apply_patch, apply_patch_file
from weightcode.patch.limits import dense_delta_bytes
from weightcode.sandbox.program import PROGRAM_WALL_S, ProgramSandbox

LOOP_STREAM_SCHEMA = "weightcode.harness.loop_stream.v0"
MODE_TEACHER = "teacher"
MODE_SELECT = "checked_selection"
MODES = (MODE_TEACHER, MODE_SELECT)
STATUS_NOOP = "noop"
STATE_FILE = "state.json"
KEPT_TEMPLATE = "round_{k}_kept.json"
SELECT_BUDGET = 8  # PREREG E4-loop: the loop rounds' check budget, which control (a) shares
STOPS_STREAM = frozenset({STATUS_ABORTED_LEAK, STATUS_ABORTED_SELFTEST, STATUS_DRY_RUN})
ROUND1_LOOP_ORACLE = "loop_oracle"  # round 1's clean count by the loop's own check (loop-retain-v2)
ROUND1_KEPT_BASE = "round1_kept_base"
ROUND1_FROM_PROGRAM = "program_run"  # round 1's source (``round1_source``): its program run on the base (default)
ROUND1_FROM_BASE = "session_kept_base"  # its session kept the base, so round 1 keeps it
ROUND1_FROM_FINAL = "session_final"  # the session's saved final, re-applied to the base
ROUND1_FROM_RECORD = "loop_record"  # round 1's student of an earlier loop stream
WALL_TIMEOUT_S = 5400.0  # PREREG E4-loop and E10: 90 min per loop session
WALL_TIMEOUT_S_E11 = 7200.0  # PREREG E11: --wall-timeout-min 120
EDIT_LOG_TEMPLATE = "edit_log_round_{k}.json"  # PREREG T16: round k's edit log in the stream's work directory
TAG_RE = re.compile(r"[a-z0-9_]{1,32}")


class LoopStreamError(RuntimeError):
    """A loop stream cannot run or resume as specified."""


def evidence_for(instance: Instance, params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The evidence rows a round draws for its own condition on ``params``, with their dataset idx
    (``loop.EVIDENCE_KEYS``)."""
    from weightcode.evalkit.student_evidence import draw_evidence_for

    experience, _ = stream_experience(instance, params)
    drawn = draw_evidence_for(instance, dict(params))
    if not np.array_equal(np.asarray(drawn.arrays["split_ids"]), experience["split_ids_new"]):
        raise LoopStreamError("two evidence draws for the same student give other rows")
    return {"x": experience["x_new"], "y": experience["y_new"], "split_ids": experience["split_ids_new"],
            "idx": np.asarray(drawn.source_idx, dtype=np.int64)}


def round_one_entry(session_root: Path, program: Path) -> dict[str, Any]:
    """Round 1's library entry: its frozen program with the scores of its own session's kept check."""
    final = read_json(Path(session_root) / "final" / "selection.json")["final"]
    if final.get("kind") != "checked":
        raise LoopStreamError(f"{session_root} kept no checked program")
    recorded = (final.get("program") or {}).get("sha256")
    if recorded != file_sha256(program):
        raise LoopStreamError(f"{program} is not the kept program of {session_root}")
    scores = {key: int(final[key]) for key in ("score_new", "score_old", "score")}
    return {"round": 1, "program": {"path": str(Path(program).resolve()), "sha256": recorded}, "scores": scores,
            "source": {"session_id": Path(session_root).name}}


def round_one_noop_entry(session_root: Path) -> dict[str, Any]:
    """Round 1's library entry when its session kept the base: no program and no scores."""
    final = read_json(Path(session_root) / "final" / "selection.json")["final"]
    if not final.get("is_base_noop") or final.get("kind") == "checked":
        raise LoopStreamError(f"{session_root} kept a checked program; give it as the round-1 program")
    return {"round": 1, "program": None, "scores": None, "source": {"session_id": Path(session_root).name}}


def round_one_from_final(session_root: Path, instance: Instance, program: Mapping[str, Any]
                         ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Round 1's student from its session's saved final, without running the program, and its ``round1_source``.

    The session must be a LEARN session on round 1's instance and base whose final is the checked ``program``. The
    final's ``patch.json`` and ``deltas.npz`` must hash to the final's record and ``model.npz`` to its child;
    ``patch.json`` is applied to the base under round 1's LEARN limits and must give that child."""
    root = Path(session_root)
    session = read_json(root / "session.json")
    final = read_json(root / "final" / "selection.json")["final"]
    recorded = final.get("program") or {}
    base_sha = instance.params_sha256
    if final.get("kind") != "checked" or final.get("is_base_noop") or recorded.get("sha256") != program["sha256"]:
        raise LoopStreamError(f"{root}: its final is not the checked program {program['path']}")
    if final.get("parent") != PARENT_INSTANCE or session.get("real_model_sha256") != base_sha \
            or session.get("instance_id") != instance.instance_id:
        raise LoopStreamError(f"{root}: its final is not a child of round 1's instance {instance.instance_id} and the "
                              "stream's base")
    files = {key: root / str(recorded[key]) for key in ("patch", "deltas")}
    for key, path in files.items():
        if file_sha256(path) != recorded[f"{key}_sha256"]:
            raise LoopStreamError(f"{path} does not hash to the final's {key}_sha256")
    model = root / str(final["model"])
    if params_sha256(load_params(model)) != final["child_sha256"]:
        raise LoopStreamError(f"{model} does not hash to the final child")
    base = {name: np.array(value) for name, value in instance.params.items()}
    limits = applied_limits(instance.task["limits"], LEARN_ARM, len(base),
                            dense_delta_bytes(v.shape for v in base.values()))
    try:
        result = apply_patch_file(base, files["patch"], limits, base_sha)
    except PatchError as refusal:
        raise LoopStreamError(f"{files['patch']} is refused on the base: {list(refusal.messages)}") from None
    if result.child_sha256 != final["child_sha256"]:
        raise LoopStreamError(f"{files['patch']} applied to the base does not give the final child")
    source = {"kind": ROUND1_FROM_FINAL, "session_id": root.name, "model": str(model),
              "child_sha256": result.child_sha256, "patch": str(files["patch"]),
              "patch_sha256": recorded["patch_sha256"], "deltas": str(files["deltas"]),
              "deltas_sha256": recorded["deltas_sha256"], "run_record": str(root / str(recorded.get("run_record")))}
    return {name: np.array(value) for name, value in result.child.items()}, source


def round_one_from_record(record_path: Path, instance: Instance, program: Mapping[str, Any], session_root: Path
                          ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Round 1's student of an earlier loop stream, without running the program, and its ``round1_source``.

    ``record_path`` is a loop stream record or the ``state.json`` of a stream still running. Its round 1 must have
    applied ``program`` of ``session_root`` to the same base on round 1's instance, and its ``student_after`` file
    must hash to that round's ``child_sha256``."""
    path = Path(record_path)
    record = read_json(path)
    rounds = record.get("rounds") if isinstance(record, dict) else None
    if not rounds or record.get("schema") != LOOP_STREAM_SCHEMA or rounds[0].get("round") != 1:
        raise LoopStreamError(f"{path} is not a loop stream record (or state.json) with a finished round 1")
    row = rounds[0]
    entry = row.get("library_entry") or {}
    if row.get("instance_id") != instance.instance_id or row.get("parent_sha256") != instance.params_sha256:
        raise LoopStreamError(f"{path}: its round 1 is not on round 1's instance {instance.instance_id} and the "
                              "stream's base")
    if row.get("status") != STATUS_OK or (entry.get("program") or {}).get("sha256") != program["sha256"] \
            or (entry.get("source") or {}).get("session_id") != Path(session_root).name:
        raise LoopStreamError(f"{path}: its round 1 did not apply the program {program['path']} of "
                              f"{Path(session_root).name}")
    after = row.get("student_after") or {}
    child = load_params(Path(str(after.get("path"))))
    if after.get("sha256") != row["child_sha256"] or params_sha256(child) != row["child_sha256"]:
        raise LoopStreamError(f"{after.get('path')} does not hash to round 1's child_sha256 in {path}")
    source = {"kind": ROUND1_FROM_RECORD, "path": str(path.resolve()), "file_sha256": file_sha256(path),
              "stem": str(record.get("stem") or path.stem), "student_after": dict(after),
              "child_sha256": row["child_sha256"],
              **{key: row.get(key) for key in ("deltas_sha256", "patch_sha256", "n_changed")}}
    return child, source


def _vcheck(instance: Instance) -> dict[str, np.ndarray]:
    return load_checked_npz(Path(str(instance.private["vcheck_path"])), str(instance.private["vcheck_sha256"]),
                            f"{instance.instance_id} V_check")


def check_parts(instance: Instance, earlier: Sequence[Instance]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The loop check of round r as a loop session builds it (``weightcode.harness.ledger``): its V_check, the clean
    companion (the same rows, y and idx in order), then each earlier round's V_check."""
    new = _vcheck(instance)
    rows, _ = clean.load_clean_check(instance)
    if not (np.array_equal(rows["y"], new["y"]) and "idx" in new and np.array_equal(rows["idx"], new["idx"])):
        raise LoopStreamError(f"{instance.instance_id}: the clean V_check rows are not V_check's rows")
    parts = {"new": (new["x"], new["y"]), "old": (rows["x"], rows["y"])}
    for k, other in enumerate(earlier, start=1):
        arrays = _vcheck(other)
        parts[loop.prev_part(k)] = (arrays["x"], arrays["y"])
    return parts


# ---------------------------------------------------------------------------------------------------------------
# loop-retain-v2: the stream's clean budget


def shared_clean_rows(instances: Sequence[Instance]) -> str:
    """The content hash of the clean V_check rows every round's clean part scores; ``LoopStreamError`` unless all the
    stream's instances have the same rows (x, y and idx, in order), without which the kept score_old do not add up."""
    hashes = {}
    for instance in instances:
        rows, _ = clean.load_clean_check(instance)
        hashes[instance.instance_id] = params_sha256(rows)
    if len(set(hashes.values())) != 1:
        raise LoopStreamError(f"{loop.KEPT_RULE_LOOP_V2} counts one clean budget over the stream, but the rounds' clean "
                              f"V_check rows differ: {hashes}")
    return next(iter(hashes.values()))


def round_one_clean(round1_session: Path, instance: Instance, base: Mapping[str, np.ndarray],
                    child: Mapping[str, np.ndarray], kept_base: bool) -> dict[str, Any]:
    """Round 1's kept score_old under loop-retain-v2 (module docstring), with its source.

    The round-1 session's kept check counts when its private clean V_check copy (hash-checked) holds the loop's clean
    rows, its base is the stream's base and its kept child is the stream's round-1 student. Otherwise the loop's check
    (``loop.MultiPartOracle`` on ``check_parts(instance, [])``, one query) scores the round-1 student against the
    base, and the session's number is recorded beside it."""
    root = Path(round1_session)
    if kept_base:
        return {"round": 1, "s_old": 0, "source": {"kind": ROUND1_KEPT_BASE, "session_id": root.name}}
    session = read_json(root / "session.json")
    final = read_json(root / "final" / "selection.json")["final"]
    child_sha, base_sha = params_sha256(child), params_sha256(base)
    rows, _ = clean.load_clean_check(instance)
    record = session.get("clean_check") or {}
    why = []
    copy = Path(str(record.get("path")))
    if not copy.is_file() or file_sha256(copy) != record.get("sha256"):
        why.append("the session's clean V_check copy is missing or does not hash to its record")
    else:
        with np.load(copy, allow_pickle=False) as archive:
            if sorted(archive.files) != sorted(rows) or not all(np.array_equal(archive[key], rows[key]) for key in rows):
                why.append("the session's clean V_check rows are not the loop's")
    if session.get("real_model_sha256") != base_sha:
        why.append("the session's base is not the stream's base")
    if final.get("child_sha256") != child_sha:
        why.append("the session's kept child is not the stream's round-1 student")
    common = {"session_id": root.name, "clean_rows_sha256": params_sha256(rows), "child_sha256": child_sha}
    if not why:
        return {"round": 1, "s_old": int(final["score_old"]), "source": {"kind": loop.ROUND1_SESSION_CHECK, **common}}
    oracle = loop.MultiPartOracle(check_parts(instance, []), instance.forward, base, 1)
    entry = oracle.query(child, "round_1")
    return {"round": 1, "s_old": int(entry["score_old"]),
            "source": {"kind": ROUND1_LOOP_ORACLE, **common, "why": why, "session_score_old": final.get("score_old"),
                       "check": {key: entry[key] for key in ("score_old", "fixed_old", "broken_old", "n_old")}}}


def kept_s_old(mode: str, index: int, record: Mapping[str, Any]) -> dict[str, Any]:
    """Round ``index``'s (>= 2) kept score_old under loop-retain-v2: 0 for a no-op round, else the session final's
    (teacher) or the kept candidate's check (control (a))."""
    if record["status"] == STATUS_NOOP:
        return {"round": index, "s_old": 0, "source": {"kind": "noop"}}
    if mode == MODE_TEACHER:
        return {"round": index, "s_old": int(record["final"]["score_old"]),
                "source": {"kind": "session_final", "session_id": Path(record["sessions"][-1]["root"]).name}}
    kept = next(row for row in record["candidates"] if row["name"] == record["kept"])
    return {"round": index, "s_old": int(kept["check"]["score_old"]),
            "source": {"kind": "control_check", "kept": record["kept"]}}


def session_candidate(root: Path) -> Path | None:
    """A ``--candidates`` value naming a session directory: that session's kept program, hash-checked against its
    final, or None when its final is the base (the round then has no candidate). A session without a final yet is
    refused."""
    selection = Path(root) / "final" / "selection.json"
    if not selection.is_file():
        raise LoopStreamError(f"{root} has no final yet ({selection} is missing)")
    final = read_json(selection)["final"]
    if final.get("kind") == "checked":
        program = Path(root) / str(final["program"]["file"])
        if file_sha256(program) != final["program"]["sha256"]:
            raise LoopStreamError(f"{program} does not hash to the kept program of {root}")
        return program
    if final.get("is_base_noop"):
        return None
    raise LoopStreamError(f"{root}: its final is neither a checked program nor the base")


def select_round(index: int, instance: Instance, params: dict[str, np.ndarray], earlier: Sequence[Instance],
                 prev: Sequence[tuple[int, Mapping[str, np.ndarray]]], candidates: Sequence[Path], work: Path,
                 sandbox: ProgramSandbox, wall_s: float, kept_rule: str = loop.KEPT_RULE_LOOP,
                 clean_used: Mapping[int, int] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Control (a) round ``index``: check every candidate on the current student and keep by the stream's kept rule
    (loop-retain-v1, or v2 with the earlier rounds' kept score_old ``clean_used``); with no candidate, keep the
    current student and spend no check."""
    if not 0 <= len(candidates) <= SELECT_BUDGET:
        raise LoopStreamError(f"round {index}: {len(candidates)} candidates for a budget of {SELECT_BUDGET} checks")
    work.mkdir(mode=0o700)
    parent_sha = params_sha256(params)
    limits = applied_limits(instance.task["limits"], LEARN_ARM, len(params),
                            dense_delta_bytes(v.shape for v in params.values()))
    experience = loop.loop_experience(instance, params, prev)
    if not candidates:  # the oracle's base predictions are not needed: the rule is stated from the row counts
        rule = loop.kept_rule_for(kept_rule, int(instance.private["n_vcheck"]),
                                  {k: int(other.private["n_vcheck"]) for k, other in enumerate(earlier, start=1)},
                                  clean_used)
        return params, {"round": index, "instance_id": instance.instance_id,
                        "condition": instance.private.get("condition"), "parent_sha256": parent_sha,
                        "limits": limits_dict(limits), "kept_rule": rule,
                        "experience": {key: value for key, value in experience[1].items()
                                       if key not in ("prev_idx", "new_source_idx")},
                        "candidates": [], "queries_used": 0, "status": STATUS_NOOP, "kept": None,
                        "child_sha256": parent_sha, "reason": "no candidate program for this round"}
    oracle = loop.MultiPartOracle(check_parts(instance, earlier), instance.forward, params, SELECT_BUDGET)
    rows = oracle.n_rows()
    rule = loop.kept_rule_for(kept_rule, rows["old"], {k: rows[loop.prev_part(k)] for k in range(1, len(earlier) + 1)},
                              clean_used)
    record: dict[str, Any] = {"round": index, "instance_id": instance.instance_id,
                              "condition": instance.private.get("condition"), "parent_sha256": parent_sha,
                              "limits": limits_dict(limits), "kept_rule": rule,
                              "experience": {key: value for key, value in experience[1].items()
                                             if key not in ("prev_idx", "new_source_idx")},
                              "candidates": []}
    children: dict[str, dict[str, np.ndarray]] = {}
    for number, program in enumerate(candidates, start=1):
        name = f"candidate_{number}"
        row: dict[str, Any] = {"name": name, "program": _program(Path(program))}
        cwork = work / name
        cwork.mkdir(mode=0o700)
        student, experience_file, _ = stage_student(instance, cwork / "student", params=params, experience=experience)
        try:
            update = run_update(student, experience_file, Path(program), cwork / "run", sandbox, wall_s)
        except ProgramProblem as problem:
            record["candidates"].append(row | {"status": "program_failed", "messages": problem.messages})
            continue
        patch_dir = cwork / "patch"
        patch_dir.mkdir()
        patch = write_derived_patch(update, params, parent_sha, patch_dir)
        try:
            result = apply_patch(params, patch, patch_dir, limits, parent_sha)
        except PatchError as refusal:
            record["candidates"].append(row | {"status": "patch_refused", "messages": list(refusal.messages)})
            continue
        entry = oracle.query(result.child, name)
        children[name] = {key: np.array(value) for key, value in result.child.items()}
        row.update(status="checked", child_sha256=result.child_sha256, n_changed=result.n_changed,
                   eligible=loop.eligible(entry, rule),
                   check={key: value for key, value in entry.items()
                          if key.startswith(("score", "fixed", "broken")) or key in ("query_index", "n_changed")})
        record["candidates"].append(row)
    pool = [Candidate(patch_id=row["name"], score=int(row["check"]["score"]), n_changed=int(row["n_changed"]),
                      query_index=int(row["check"]["query_index"]))
            for row in record["candidates"] if row["status"] == "checked" and row["eligible"]]
    chosen = choose_final(pool)
    record["queries_used"] = SELECT_BUDGET - oracle.queries_remaining
    if chosen.patch_id is None:
        record.update(status=STATUS_NOOP, kept=None, child_sha256=parent_sha)
        return params, record
    child = children[str(chosen.patch_id)]
    record.update(status=STATUS_OK, kept=str(chosen.patch_id), child_sha256=params_sha256(child))
    return child, record


def teacher_round(index: int, instance: Instance, params: Mapping[str, np.ndarray], evidence: Mapping[str, np.ndarray],
                  context_path: Path, session_kwargs: Mapping[str, Any], launcher: Any | None
                  ) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Mode ``teacher`` round ``index``: one LEARN session (once more after INFRA); the next student, the round
    record and the round's library entry."""
    from weightcode.harness.runner import run_session

    parent_sha = params_sha256(params)
    sessions = []
    for _ in (1, 2):
        result = run_session(instance.instance_id, LEARN_ARM, 0, loop_context=context_path, launcher=launcher,
                             **dict(session_kwargs))
        sessions.append({"session_id": result.session_id, "status": result.status, "root": str(result.root)})
        if result.status in STOPS_STREAM:  # nothing was launched or kept: a fault to fix, then resume
            raise LoopStreamError(f"round {index}: session {result.session_id} ended {result.status}; the stream "
                                  "stops here and resumes from this round")
        if result.status != STATUS_INFRA:
            break
    root = Path(sessions[-1]["root"])
    if read_json(root / "session.json")["real_model_sha256"] != parent_sha:
        raise LoopStreamError(f"{root}: the session's student is not the stream's current student")
    with np.load(root / "trial" / EXPERIENCE_FILE, allow_pickle=False) as archive:
        if not (np.array_equal(archive["x_new"], evidence["x"])
                and np.array_equal(archive["split_ids_new"], evidence["split_ids"])):
            raise LoopStreamError(f"round {index}: the session's evidence rows are not the rows the stream drew")
    final = read_json(root / "final" / "selection.json")["final"]
    if final.get("parent") != loop.PARENT_LOOP:
        raise LoopStreamError(f"{root}: the final's parent is {final.get('parent')!r}, not the loop student")
    child = load_params(root / "final" / "model.npz")
    if params_sha256(child) != final["child_sha256"]:
        raise LoopStreamError(f"{root}: final/model.npz does not hash to the final child")
    session_record = read_json(root / "session_result.json")
    noop = bool(final.get("is_base_noop"))
    if noop != (final["child_sha256"] == parent_sha):
        raise LoopStreamError(f"{root}: a no-op final must keep the student, and only a no-op may")
    record: dict[str, Any] = {
        "round": index, "instance_id": instance.instance_id, "condition": instance.private.get("condition"),
        "sessions": sessions, "status": STATUS_NOOP if noop else STATUS_OK,
        "session_status": sessions[-1]["status"], "child_sha256": final["child_sha256"],
        "final": {key: final.get(key) for key in ("kind", "patch_id", "query_index", "score", "n_changed", "reason")}
        | {key: value for key, value in final.items() if key.startswith(("score_new", "score_old", "score_prev_"))},
        "queries_used": session_record["queries"]["used"],
        "audit_verdict": (session_record.get("audit") or {}).get("verdict"),
        "code_changed_during_session": session_record.get("code_changed_during_session"),
        "rejected_attempts": len(session_record.get("rejected_attempts") or []),
    }
    if noop:
        entry = {"round": index, "program": None, "scores": None, "source": {"session_id": root.name}}
    else:
        scores = {key: int(value) for key, value in final.items()
                  if key in ("score", "score_new", "score_old") or key.startswith("score_prev_")}
        entry = {"round": index, "program": {"path": str(root / str(final["program"]["file"])),
                                             "sha256": final["program"]["sha256"]},
                 "scores": scores, "source": {"session_id": root.name}}
    return child, record, entry


def _stem(mode: str, instance_ids: Sequence[str], first: Mapping[str, Any],
          candidates: Mapping[int, Sequence[Mapping[str, Any]]], session_kwargs: Mapping[str, Any],
          kept_rule: str = loop.KEPT_RULE_LOOP, round1_source: Mapping[str, Any] | None = None,
          edit_log: bool = False, tag: str | None = None, protect_logged: bool = False) -> str:
    identity: list[Any] = [mode, list(instance_ids), first["sha256"],
                           {str(k): [item["sha256"] for item in v] for k, v in sorted(candidates.items())},
                           {key: repr(value) for key, value in sorted(session_kwargs.items())}]
    if kept_rule != loop.KEPT_RULE_LOOP:  # a v1 stream keeps the stem it had before v2 existed
        identity.append({"kept_rule": kept_rule, "clean_budget_frac": loop.LOOP_CLEAN_BUDGET_FRAC})
    if round1_source is not None:  # a recorded round-1 student; a stream that runs its program keeps its stem
        identity.append({"round1_source": round1_source["kind"], "child_sha256": round1_source["child_sha256"]})
    if edit_log:  # PREREG T16 arm L; arm N keeps the stem it had before T16
        identity.append({"edit_log": loop.EDIT_LOG_SCHEMA})
    if protect_logged:  # T16 follow-up M; arms L and N keep their T16 stems
        identity.append({"protect_logged": loop.PROTECT_SCHEMA})
    if tag is not None:
        identity.append({"tag": tag})
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]


def edit_log_file(path: Path, rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Round r's edit log (PREREG T16) from the finished rounds 1 to r - 1 of a stream (their ``student_before`` and
    ``student_after`` files, hash-checked): written at ``path``, or checked equal to the file already there. Each
    round's count must be its changed-entry count. Returns the loop context's ``edit_log`` record."""
    triples = []
    sources = []
    for row in rounds:
        pair = []
        for key in ("student_before", "student_after"):
            params = load_params(Path(str(row[key]["path"])))
            if params_sha256(params) != row[key]["sha256"]:
                raise LoopStreamError(f"round {row['round']}: {row[key]['path']} does not hash to its record")
            pair.append(params)
        triples.append((int(row["round"]), pair[0], pair[1]))
        sources.append({"round": int(row["round"]), "parent_sha256": row["student_before"]["sha256"],
                        "child_sha256": row["student_after"]["sha256"], "n_changed": count_changed(*pair)})
    try:
        document = loop.edit_log_document(triples)
    except loop.LoopError as exc:
        raise LoopStreamError(str(exc)) from None
    counts = loop.edit_log_counts(document)
    if counts != {str(item["round"]): item["n_changed"] for item in sources}:
        raise LoopStreamError(f"the edit log counts {counts} are not the rounds' changed entries {sources}")
    data = loop.edit_log_bytes(document)
    if path.exists():
        if path.read_bytes() != data:
            raise LoopStreamError(f"{path} holds another edit log")
    else:
        path.write_bytes(data)
    return {"schema": loop.EDIT_LOG_SCHEMA, "path": str(path), "sha256": file_sha256(path), "n_entries": counts,
            "rounds": sources}


def check_session_edit_log(index: int, root: Path, expected: Mapping[str, Any] | None) -> dict[str, Any]:
    """session.json ``loop.edit_log`` of round ``index``'s session: arm L must name the stream's file (same sha256 in
    the context record and in the trial), arm N must say ``enabled`` false (or nothing, a session made before T16).
    Returns the arm as stated."""
    stated = (read_json(Path(root) / "session.json").get("loop") or {}).get("edit_log")
    if expected is None:
        if stated not in (None, {"enabled": False}):
            raise LoopStreamError(f"round {index}: the stream has no edit log, the session states {stated}")
        return {"enabled": False}
    if not isinstance(stated, Mapping) or not stated.get("enabled") \
            or not stated.get("sha256") == stated.get("trial_sha256") == expected["sha256"] \
            or stated.get("source") != json.loads(json.dumps(expected)):
        raise LoopStreamError(f"round {index}: the session's edit log {stated} is not the stream's {expected}")
    return dict(stated)


def protect_context(index: int, log_record: Mapping[str, Any]) -> dict[str, Any]:
    """Round ``index``'s loop-context ``protect_logged`` record (T16 follow-up M): the edit log it protects, by
    sha256, and the number of entries that log lists over all earlier rounds, counted from the stream's file."""
    data = Path(str(log_record["path"])).read_bytes()
    if hashlib.sha256(data).hexdigest() != log_record["sha256"]:
        raise LoopStreamError(f"{log_record['path']} changed after it was written")
    try:
        protected = loop.protected_entries(loop.check_edit_log(data, index))
    except loop.LoopError as exc:
        raise LoopStreamError(str(exc)) from None
    return {"schema": loop.PROTECT_SCHEMA, "edit_log_sha256": log_record["sha256"],
            "n_protected": int(sum(loop.protected_counts(protected).values()))}


def check_session_protect(index: int, root: Path, expected: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """session.json ``loop.protect_logged`` of round ``index``'s session: with the limit it must be enabled with the
    stream's record (the same edit log and count), without it absent. With the limit, returns the session's record
    plus ``refused_attempts``, the attempts the limit refused (session_result.json, redaction "protected"; none for a
    dry run)."""
    from weightcode.harness.applying import REDACTION_PROTECTED

    stated = (read_json(Path(root) / "session.json").get("loop") or {}).get("protect_logged")
    if expected is None:
        if stated is not None:
            raise LoopStreamError(f"round {index}: the stream has no protected-entry limit, the session states {stated}")
        return None
    if not isinstance(stated, Mapping) or stated.get("enabled") is not True \
            or stated.get("source") != json.loads(json.dumps(expected)) \
            or stated.get("edit_log_sha256") != expected["edit_log_sha256"] \
            or stated.get("n_protected") != expected["n_protected"]:
        raise LoopStreamError(f"round {index}: the session's protected-entry limit {stated} is not the stream's "
                              f"{expected}")
    result_path = Path(root) / "session_result.json"
    rejected = (read_json(result_path).get("rejected_attempts") or []) if result_path.is_file() else []
    return dict(stated) | {"refused_attempts": sum(1 for item in rejected
                                                   if item.get("redaction") == REDACTION_PROTECTED)}


def _student_file(path: Path, params: Mapping[str, np.ndarray]) -> dict[str, str]:
    """Save ``params`` at ``path``, or check that the file already there holds them."""
    sha = params_sha256(params)
    if path.exists():
        if params_sha256(load_params(path)) != sha:
            raise LoopStreamError(f"{path} holds another student")
    elif save_params(path, params) != sha:
        raise LoopStreamError(f"{path} does not hash to the student")
    return {"path": str(path), "sha256": sha}


def _evidence_file(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    """Save the evidence ``arrays`` at ``path``, or check that the file already there holds them."""
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            same = sorted(archive.files) == sorted(arrays) and all(np.array_equal(archive[key], arrays[key])
                                                                     for key in arrays)
        if not same:
            raise LoopStreamError(f"{path} holds other evidence rows")
    else:
        np.savez(path, **arrays)
    return {"path": str(path), "sha256": file_sha256(path)}


def _context_file(path: Path, context: Mapping[str, Any]) -> dict[str, str]:
    if path.exists() and read_json(path) != json.loads(json.dumps(context)):
        raise LoopStreamError(f"{path} holds another context")
    if not path.exists():
        write_json(path, context)
    return {"path": str(path), "sha256": file_sha256(path)}


def _fresh_dir(path: Path) -> list[str]:
    """Move a partial round directory aside (never delete it); returns where it went."""
    if not path.exists():
        return []
    number = 1
    while path.with_name(f"{path.name}_incomplete_{number}").exists():
        number += 1
    moved = path.with_name(f"{path.name}_incomplete_{number}")
    path.rename(moved)
    return [str(moved)]


def run_loop_stream(mode: str, instance_ids: Sequence[str], round1_program: Path | None, round1_session: Path,
                    out_dir: Path, sandbox: ProgramSandbox, *, candidates: Mapping[int, Sequence[Path]] | None = None,
                    session_kwargs: Mapping[str, Any] | None = None,
                    launcher_for: Callable[[int], Any] | None = None, resume: bool = False,
                    batch_size: int = DEFAULT_BATCH_SIZE, wall_s: float = PROGRAM_WALL_S,
                    dry_run: bool = False, kept_rule: str = loop.KEPT_RULE_LOOP, round1_from_final: bool = False,
                    round1_from_record: Path | None = None, edit_log: bool = False, tag: str | None = None,
                    protect_logged: bool = False) -> Path:
    """Run (or resume) one loop stream and write its record into ``out_dir`` (module docstring).

    ``round1_program`` None: the round-1 session kept the base, and round 1 keeps it. ``dry_run`` (teacher mode):
    stop at the first teacher round after preparing its session as a dry run; returns ``dry_run_round_<r>.json``.
    ``kept_rule``: ``loop.KEPT_RULE_LOOP`` (v1) or ``loop.KEPT_RULE_LOOP_V2`` (the stream's clean budget).
    ``round1_from_final`` / ``round1_from_record`` (at most one, each with ``round1_program``): round 1 does not run
    the program and takes the session's saved final (``round_one_from_final``) or round 1's student of that loop
    stream record (``round_one_from_record``). ``edit_log`` (teacher mode, PREREG T16 arm L) gives every teacher round
    the edit log of the rounds before it; ``tag`` labels the stream in its identity (module docstring).
    ``protect_logged`` (with ``edit_log``, T16 follow-up M) adds the protected-entry limit on the entries that log lists.
    """
    if protect_logged and not edit_log:
        raise LoopStreamError("the protected-entry limit protects the entries of the edit log; it needs edit_log")
    if edit_log and mode != MODE_TEACHER:
        raise LoopStreamError("the edit log is given to a teacher round; checked selection launches no session")
    if tag is not None and not TAG_RE.fullmatch(tag):
        raise LoopStreamError(f"a stream tag is 1 to 32 of [a-z0-9_], not {tag!r}")
    if mode not in MODES:
        raise LoopStreamError(f"mode must be one of {MODES}, not {mode!r}")
    if kept_rule not in loop.KEPT_RULES:
        raise LoopStreamError(f"the kept rule is one of {loop.KEPT_RULES}, not {kept_rule!r}")
    v2 = kept_rule == loop.KEPT_RULE_LOOP_V2
    if dry_run and (mode != MODE_TEACHER or launcher_for is not None or "dry_run" in (session_kwargs or {})):
        raise LoopStreamError("a dry run is a teacher stream's, with no launcher and no dry_run session argument")
    if not 2 <= len(instance_ids) <= loop.MAX_ROUNDS:
        raise LoopStreamError(f"a loop stream has 2 to {loop.MAX_ROUNDS} rounds, not {len(instance_ids)}")
    session_kwargs = dict(session_kwargs or {})
    chosen = {int(k): [Path(p) for p in v] for k, v in (candidates or {}).items()}
    if mode == MODE_SELECT and sorted(chosen) != list(range(2, len(instance_ids) + 1)):
        raise LoopStreamError("checked selection needs candidates for every round from 2 on")
    if mode == MODE_TEACHER and chosen:
        raise LoopStreamError("a teacher stream takes no candidates")
    if mode == MODE_SELECT and (session_kwargs or launcher_for is not None):
        raise LoopStreamError("checked selection launches no session")
    if round1_from_final and round1_from_record is not None:
        raise LoopStreamError("round 1 comes from the session's final or from a loop record, not both")
    if (round1_from_final or round1_from_record is not None) and round1_program is None:
        raise LoopStreamError("a recorded round-1 student needs the round-1 program (its session's kept program)")
    instances = _load_stream(instance_ids)
    clean_rows_sha = shared_clean_rows(instances) if v2 else None  # refused before any write
    first = None if round1_program is None else _program(Path(round1_program))
    round1_entry = (round_one_noop_entry(Path(round1_session)) if first is None  # refused before any write
                    else round_one_entry(Path(round1_session), Path(first["path"])))
    round1_child: dict[str, np.ndarray] | None = None  # a recorded round-1 student, checked before any write
    if first is None:
        round1_source: dict[str, Any] = {"kind": ROUND1_FROM_BASE}
    elif round1_from_final:
        round1_child, round1_source = round_one_from_final(Path(round1_session), instances[0], first)
    elif round1_from_record is not None:
        round1_child, round1_source = round_one_from_record(Path(round1_from_record), instances[0], first,
                                                            Path(round1_session))
    else:
        round1_source = {"kind": ROUND1_FROM_PROGRAM}
    if dry_run:
        session_kwargs["dry_run"] = True
    shown_candidates = {k: [_program(p) for p in v] for k, v in chosen.items()}
    base = {name: np.array(value) for name, value in instances[0].params.items()}
    base_sha = instances[0].params_sha256
    first_identity = {"sha256": f"base:{Path(round1_session).name}"} if first is None else first
    stem_id = _stem(mode, instance_ids, first_identity, shown_candidates, session_kwargs, kept_rule,
                    None if round1_child is None else round1_source, edit_log, tag, protect_logged)
    stem = f"loop_{mode}_{stem_id}_{base_sha[:12]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path, work = out_dir / f"{stem}.json", out_dir / stem
    if path.exists():
        raise LoopStreamError(f"{path} exists already; records are never overwritten")
    state_path = work / STATE_FILE
    if work.exists() and not resume:
        raise LoopStreamError(f"{work} exists; pass resume=True to continue it")
    if not work.exists():
        work.mkdir(mode=0o700)
        write_json(state_path, {"schema": LOOP_STREAM_SCHEMA, "stem": stem, "rounds": []})
    rounds = list(read_json(state_path)["rounds"])
    n_params = sum(value.size for value in base.values())
    params: dict[str, np.ndarray] = base
    library: list[dict[str, Any]] = []
    prev_items: list[dict[str, Any]] = []
    clean_used: list[dict[str, Any]] = []  # loop-retain-v2: each finished round's kept score_old, with its source
    for done in rounds:  # resume: every kept file must still hash as recorded
        params = load_params(Path(done["student_after"]["path"]))
        if params_sha256(params) != done["student_after"]["sha256"]:
            raise LoopStreamError(f"round {done['round']}: the saved student changed")
        library.append(done["library_entry"])
        prev_items.append(done["prev_item"])
        if v2:
            clean_used.append(done["clean_s_old"])
    for index in range(len(rounds) + 1, len(instances) + 1):
        instance = instances[index - 1]
        used = {int(item["round"]): int(item["s_old"]) for item in clean_used} if v2 else None
        expected_rule = None if index == 1 or not (v2 or dry_run) else loop.kept_rule_for(  # the session's rule
            kept_rule, int(instance.private["n_vcheck"]),
            {k: int(other.private["n_vcheck"]) for k, other in enumerate(instances[:index - 1], start=1)}, used)
        context_rule = None if not v2 or expected_rule is None else {
            "name": kept_rule, "clean_budget_frac": expected_rule["clean_budget_frac"],
            "clean_budget_floor": expected_rule["clean_budget_floor"], "clean_used": list(clean_used),
            "s_old_floor": expected_rule["s_old_floor"]}
        before = _student_file(work / f"student_before_round_{index}.npz", params)
        evidence_arrays = evidence_for(instance, params)
        evidence = _evidence_file(work / f"evidence_round_{index}.npz", evidence_arrays)
        moved = _fresh_dir(work / f"round_{index}")
        if index == 1 and first is None:  # the experience a program would have got, recorded for control (b)
            child = params
            record = {"round": 1, "instance_id": instance.instance_id, "condition": instance.private.get("condition"),
                      "parent_sha256": params_sha256(params), "experience": stream_experience(instance, params)[1],
                      "status": STATUS_NOOP, "child_sha256": params_sha256(params),
                      "reason": f"the round-1 session {Path(round1_session).name} kept the base",
                      "round1_source": round1_source}
            entry = round1_entry
        elif index == 1 and round1_child is not None:  # a recorded child of the round-1 program: no program run
            child = round1_child
            limits = applied_limits(instance.task["limits"], LEARN_ARM, len(params),
                                    dense_delta_bytes(v.shape for v in params.values()))
            n_changed = count_changed(params, child)
            record = {"round": 1, "instance_id": instance.instance_id, "condition": instance.private.get("condition"),
                      "parent_sha256": params_sha256(params),
                      "program": {"name": first["name"], "sha256": first["sha256"]}, "limits": limits_dict(limits),
                      "experience": stream_experience(instance, params)[1], "status": STATUS_OK,
                      "child_sha256": params_sha256(child), "n_changed": n_changed,
                      "changed_frac": n_changed / n_params, "round1_source": round1_source}
            entry = round1_entry
        elif index == 1:
            child, record = _round(1, Path(first["path"]), instance, params, work / "round_1", sandbox, wall_s)
            if record["status"] != STATUS_OK:
                raise LoopStreamError(f"round 1 did not apply its program: {record['status']}")
            record["round1_source"] = round1_source
            entry = round1_entry
        elif dry_run:
            log_record = None if not edit_log else edit_log_file(work / EDIT_LOG_TEMPLATE.format(k=index), rounds)
            protect = None if not protect_logged or log_record is None else protect_context(index, log_record)
            context = {"schema": loop.LOOP_CONTEXT_SCHEMA, "stream_id": stem, "round": index,
                       "instance_id": instance.instance_id, "base_params_sha256": base_sha, "student": before,
                       "prev": prev_items, "library": library,
                       **({} if context_rule is None else {"kept_rule": context_rule}),
                       **({} if log_record is None else {"edit_log": log_record}),
                       **({} if protect is None else {"protect_logged": protect})}
            context_file = _context_file(work / f"context_round_{index}.json", context)
            from weightcode.harness.runner import run_session

            result = run_session(instance.instance_id, LEARN_ARM, 0, loop_context=Path(context_file["path"]),
                                 **session_kwargs)
            session_rule = read_json(Path(result.root) / "session.json").get("kept_rule")
            if session_rule != expected_rule:
                raise LoopStreamError(f"round {index}: the dry-run session's kept rule {session_rule} is not the "
                                      f"stream's {expected_rule}")
            session_log = check_session_edit_log(index, Path(result.root), log_record)
            session_protect = check_session_protect(index, Path(result.root), protect)
            dry = work / f"dry_run_round_{index}.json"
            write_json(dry, {"schema": LOOP_STREAM_SCHEMA, "round": index, "instance_id": instance.instance_id,
                             "session_id": result.session_id, "status": result.status, "root": str(result.root),
                             "context": context_file, "created_at": utc_now(), "edit_log": session_log,
                             **({"kept_rule": expected_rule, "clean_used": list(clean_used)} if v2 else {}),
                             **({} if session_protect is None else {"protect_logged": session_protect})})
            return dry
        elif mode == MODE_TEACHER:
            log_record = None if not edit_log else edit_log_file(work / EDIT_LOG_TEMPLATE.format(k=index), rounds)
            protect = None if not protect_logged or log_record is None else protect_context(index, log_record)
            context = {"schema": loop.LOOP_CONTEXT_SCHEMA, "stream_id": stem, "round": index,
                       "instance_id": instance.instance_id, "base_params_sha256": base_sha, "student": before,
                       "prev": prev_items, "library": library,
                       **({} if context_rule is None else {"kept_rule": context_rule}),
                       **({} if log_record is None else {"edit_log": log_record}),
                       **({} if protect is None else {"protect_logged": protect})}
            context_file = _context_file(work / f"context_round_{index}.json", context)
            kept_path = work / KEPT_TEMPLATE.format(k=index)
            if kept_path.exists():  # this round's session finished before the stream stopped: no second launch
                kept = read_json(kept_path)
                child = load_params(Path(kept["child"]["path"]))
                if params_sha256(child) != kept["child"]["sha256"] or kept["context"] != context_file:
                    raise LoopStreamError(f"{kept_path} does not match its child or this round's context")
                record, entry = kept["record"], kept["entry"]
            else:
                launcher = None if launcher_for is None else launcher_for(index)
                child, record, entry = teacher_round(index, instance, params, evidence_arrays,
                                                     Path(context_file["path"]), session_kwargs, launcher)
                record["context"] = context_file
                root = Path(record["sessions"][-1]["root"])
                write_json(kept_path, {"child": {"path": str(root / "final" / "model.npz"),
                                                 "sha256": params_sha256(child)},
                                       "context": context_file, "record": record, "entry": entry})
            if v2:  # after round_<r>_kept.json, so that a mismatch never launches the round again
                session_rule = read_json(Path(record["sessions"][-1]["root"]) / "session.json").get("kept_rule")
                if session_rule != expected_rule:
                    raise LoopStreamError(f"round {index}: the session's kept rule {session_rule} is not the "
                                          f"stream's {expected_rule}")
                record["kept_rule"] = expected_rule
            # PREREG T16: the arm the session ran (a session made before T16 states none: arm N)
            record["edit_log"] = check_session_edit_log(index, Path(record["sessions"][-1]["root"]), log_record)
            if protect is not None:  # T16 follow-up M: the limit as the session states it, and what it refused
                record["protect_logged"] = check_session_protect(index, Path(record["sessions"][-1]["root"]), protect)
            else:  # a stream without the limit: its sessions state none
                check_session_protect(index, Path(record["sessions"][-1]["root"]), None)
        else:
            prev = [(int(item["round"]), load_checked_npz(Path(item["evidence"]["path"]), item["evidence"]["sha256"],
                                                          f"round {item['round']} evidence"))
                    for item in prev_items]
            child, record = select_round(index, instance, params, instances[:index - 1], prev, chosen[index],
                                         work / f"round_{index}", sandbox, wall_s, kept_rule, used)
            entry = {"round": index, "program": None, "scores": None, "source": {"kept": record.get("kept")}}
        if v2:
            record["clean_s_old"] = (round_one_clean(Path(round1_session), instance, base, child, first is None)
                                     if index == 1 else kept_s_old(mode, index, record))
        changed = count_changed(base, child)
        record.update(parent_sha256=params_sha256(params), student_before=before, incomplete_dirs=moved,
                      n_changed_vs_base=changed, changed_frac_vs_base=changed / n_params,
                      readouts={other.instance_id: _readout(other, child, batch_size) for other in instances})
        after = _student_file(work / f"student_after_round_{index}.npz", child)
        prev_item = {"round": index, "instance_id": instance.instance_id, "evidence": evidence}
        rounds.append(record | {"student_after": after, "library_entry": entry, "prev_item": prev_item})
        write_json(state_path, {"schema": LOOP_STREAM_SCHEMA, "stem": stem, "rounds": rounds})
        params = child
        library.append(entry)
        prev_items.append(prev_item)
        if v2:
            clean_used.append(record["clean_s_old"])
    for item in [*([] if first is None else [first]),
                 *(program for programs in shown_candidates.values() for program in programs)]:
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise LoopStreamError(f"{item['path']} changed while the stream ran")
    stream = {
        "schema": LOOP_STREAM_SCHEMA, "created_at": utc_now(), "mode": mode,
        "round1": {"program": first, "session": str(round1_session), "kept_base": first is None,
                   "source": round1_source},
        "candidates": {str(k): v for k, v in shown_candidates.items()},
        "session_kwargs": {key: repr(value) for key, value in session_kwargs.items()},
        "instances": [{"round": i, "instance_id": inst.instance_id, "condition": inst.private.get("condition")}
                      for i, inst in enumerate(instances, start=1)],
        "base_params_sha256": base_sha, "sandbox": sandbox.as_dict(), "wall_s": wall_s, "batch_size": batch_size,
        "readout_source": clean.source_name(), "work_dir": str(work), "rounds": rounds,
        "summary": stream_summary(instances, rounds, E4_MEAN_EXCLUDES),
    }
    if mode == MODE_TEACHER:  # PREREG T16: the arm, each teacher round's edit log as its session states it; billing
        stream["edit_log"] = {"enabled": bool(edit_log),
                              "rounds": {str(row["round"]): row["edit_log"] for row in rounds if "edit_log" in row}}
        stream["billing"] = billing_record(session_kwargs.get("api_key_file"))
    if protect_logged:  # T16 follow-up M; a stream without the limit has no such key
        stream["protect_logged"] = {"enabled": True, "rounds": {str(row["round"]): row["protect_logged"]
                                                                for row in rounds if "protect_logged" in row}}
    if tag is not None:
        stream["tag"] = tag
    if v2:
        budget = s_old_floor(loop.LOOP_CLEAN_BUDGET_FRAC, int(instances[0].private["n_vcheck"]))
        stream["kept_rule"] = {
            "name": kept_rule, "clean_budget_frac": loop.LOOP_CLEAN_BUDGET_FRAC, "clean_budget_floor": budget,
            "clean_rows_sha256": clean_rows_sha, "clean_used": [row["clean_s_old"] for row in rounds],
            "clean_used_total": int(sum(int(row["clean_s_old"]["s_old"]) for row in rounds)),
            "s_old_floor": {str(row["round"]): row["kept_rule"]["s_old_floor"] for row in rounds if "kept_rule" in row},
        }
    write_json(path, stream)
    return path


def default_wall_timeout_s(instance_ids: Sequence[str]) -> float:
    """A loop session's wall limit when none is given: PREREG E11's 120 min on E11 instances, else 90 min."""
    from weightcode.harness.instance import read_instance_record

    kinds = {e11.is_e11(read_instance_record(instance_id).private) for instance_id in instance_ids}
    if len(kinds) != 1:
        raise LoopStreamError("a stream's instances are all E11 instances or none")
    return WALL_TIMEOUT_S_E11 if kinds == {True} else WALL_TIMEOUT_S


def main(argv: Sequence[str] | None = None) -> int:
    from weightcode.harness.runner import DEFAULT_MEMORY_BYTES, DEFAULT_THREADS, SANDBOX_PYTHON

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--instances", nargs="+", required=True, help="one dev instance per round, in round order")
    parser.add_argument("--round1-program", type=Path, default=None,
                        help="the frozen program of round 1; omit it when the round-1 session kept the base")
    parser.add_argument("--round1-session", type=Path, required=True,
                        help="the session directory that kept the round-1 program (its scores enter the library)")
    recorded = parser.add_mutually_exclusive_group()
    recorded.add_argument("--round1-from-final", action="store_true",
                          help="round 1 does not run its program: it applies the round-1 session's saved final "
                               "(final/patch.json, checked against the final's hashes and final/model.npz) to the base")
    recorded.add_argument("--round1-from-record", type=Path, default=None, metavar="LOOP_RECORD",
                          help="round 1 does not run its program: its student is round 1's student_after of this loop "
                               "stream record (or the state.json of a running stream), checked against that round's "
                               "child_sha256, so the stream shares that stream's exact round-1 student")
    parser.add_argument("--candidates", nargs="*", default=[], metavar="ROUND=PROGRAM",
                        help="checked selection: one ROUND=PROGRAM per candidate, rounds from 2 on, in check order; "
                             "ROUND=none gives that round no candidate (it keeps the student); ROUND=<session "
                             "directory> is that session's kept program, or no candidate if its final is the base")
    parser.add_argument("--kept-rule", choices=loop.KEPT_RULES, default=loop.KEPT_RULE_LOOP,
                        help="loop-retain-v1 (per-round clean floor) or loop-retain-v2 (the stream's clean budget)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="teacher: run the rounds before the first session, dry-run that session and stop")
    parser.add_argument("--effort", default="max")
    parser.add_argument("--wall-timeout-s", type=float, default=None,
                        help=f"teacher: each session's wall limit (default {WALL_TIMEOUT_S:g} s, "
                             f"{WALL_TIMEOUT_S_E11:g} s on E11 instances)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--memory-bytes", type=int, default=DEFAULT_MEMORY_BYTES)
    parser.add_argument("--sandbox-python", type=Path, default=SANDBOX_PYTHON)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--edit-log", action="store_true",
                        help="teacher (PREREG T16 arm L): every teacher round gets edit_log.json, the parameter entries "
                             "the earlier rounds changed, and the one PREREG sentence in its system prompt")
    parser.add_argument("--protect-logged", action="store_true",
                        help="teacher, with --edit-log (T16 follow-up M): the limits also refuse every update that "
                             "changes an entry edit_log.json lists, and the system prompt states it in one sentence")
    parser.add_argument("--tag", default=None,
                        help="a [a-z0-9_] label that enters the stream's identity (one T16 session, e.g. t16_l1)")
    parser.add_argument("--api-key-file", type=Path, default=None,
                        help="teacher: bill every session to the Anthropic API key in this file (mode 600; the key is "
                             "never printed or recorded), instead of the Claude Code login")
    args = parser.parse_args(argv)
    if (args.round1_from_final or args.round1_from_record is not None) and args.round1_program is None:
        parser.error("--round1-from-final and --round1-from-record need --round1-program")
    if args.mode != MODE_TEACHER and (args.edit_log or args.protect_logged or args.api_key_file is not None):
        parser.error("--edit-log, --protect-logged and --api-key-file are for a teacher stream")
    if args.protect_logged and not args.edit_log:
        parser.error("--protect-logged protects the entries edit_log.json lists; it needs --edit-log")
    if args.api_key_file is not None and args.dry_run:
        parser.error("a dry run launches no session; it takes no --api-key-file")
    candidates: dict[int, list[Path]] = {}
    no_candidate: set[int] = set()
    for item in args.candidates:
        number, sep, program = item.partition("=")
        if not sep or not number.isdigit():
            parser.error(f"--candidates takes ROUND=PROGRAM, not {item!r}")
        resolved: Path | None = None if program == "none" else Path(program)
        if resolved is not None and resolved.is_dir():  # a session directory: its kept program, or the base
            resolved = session_candidate(resolved)
            print(f"round {number}: session {program} -> {resolved if resolved is not None else 'none (kept the base)'}",
                  flush=True)
        if (resolved is None and int(number) in candidates) or (resolved is not None and int(number) in no_candidate):
            parser.error(f"round {number} is given both none and programs")
        if resolved is None:
            no_candidate.add(int(number))
            continue
        candidates.setdefault(int(number), []).append(resolved)
    for number in no_candidate:
        candidates[number] = []
    sandbox = ProgramSandbox(python=args.sandbox_python, memory_bytes=args.memory_bytes, threads=args.threads)
    session_kwargs: dict[str, Any] = {}
    if args.mode == MODE_TEACHER:
        wall = args.wall_timeout_s if args.wall_timeout_s is not None else default_wall_timeout_s(args.instances)
        session_kwargs = {"effort": args.effort, "wall_timeout_s": float(wall), "threads": args.threads,
                          "memory_bytes": args.memory_bytes, "sandbox_python": args.sandbox_python}
        if args.api_key_file is not None:  # only then: a stream billed to the login keeps its stem
            session_kwargs["api_key_file"] = args.api_key_file.absolute()
    path = run_loop_stream(args.mode, args.instances, args.round1_program, args.round1_session, args.out, sandbox,
                           candidates=candidates, session_kwargs=session_kwargs, resume=args.resume,
                           batch_size=args.batch_size, dry_run=args.dry_run, kept_rule=args.kept_rule,
                           round1_from_final=args.round1_from_final, round1_from_record=args.round1_from_record,
                           edit_log=args.edit_log, tag=args.tag, protect_logged=args.protect_logged)
    record = json.loads(path.read_text(encoding="utf-8"))
    protect = {"protect_logged": record["protect_logged"]} if "protect_logged" in record else {}
    if args.dry_run:
        print(json.dumps({"dry_run": str(path), **{key: record[key] for key in ("round", "session_id", "status",
                                                                                 "root", "edit_log")},
                          **({"kept_rule": record["kept_rule"], "clean_used": record["clean_used"]}
                             if "kept_rule" in record else {}), **protect}, indent=1))
        return 0
    print(json.dumps({"record": str(path), "rounds": [row["status"] for row in record["rounds"]],
                      "edit_log": record.get("edit_log"), "tag": record.get("tag"), "summary": record["summary"],
                      **({"kept_rule": record["kept_rule"]} if "kept_rule" in record else {}), **protect}, indent=1))
    return 0


__all__ = ["EDIT_LOG_TEMPLATE", "LOOP_STREAM_SCHEMA", "MODES", "MODE_SELECT", "MODE_TEACHER", "ROUND1_FROM_BASE",
           "ROUND1_FROM_FINAL", "ROUND1_FROM_PROGRAM", "ROUND1_FROM_RECORD", "ROUND1_KEPT_BASE", "ROUND1_LOOP_ORACLE", "WALL_TIMEOUT_S",
           "WALL_TIMEOUT_S_E11", "LoopStreamError", "check_parts", "check_session_edit_log", "check_session_protect",
           "default_wall_timeout_s", "edit_log_file", "evidence_for", "protect_context",
           "kept_s_old", "main", "round_one_clean", "round_one_entry", "round_one_from_final", "round_one_from_record",
           "round_one_noop_entry", "run_loop_stream", "select_round", "session_candidate", "shared_clean_rows",
           "teacher_round"]
