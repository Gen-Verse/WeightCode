"""After the editor exits: status, final patch, audit and the session records (HARNESS_SPEC sections 5 and 7).

Order: wait until the MCP server has released its lock (terminating a leftover server), log any cell the server left
in flight (``recover_interrupted_cell``, always), read the transcript, classify the session, rebuild the oracle state
from ``logs/checks.jsonl`` (every logged query is applied and queried again and must match), apply the final-patch
rule, write ``final/``, run the audit, then write ``session_result.json`` (the full record), ``run.json`` (the shared
run-record projection that ``weightcode.evalkit.devreadout`` reads) and one line of ``$WEIGHTCODE_RUNS/manifest.jsonl``. A session
of an E11 variant view (the rule view of PREREG K1 and the information ladder's U, K0b and K0a; session.json
``e11_variant``) carries ``e11_variant`` in all three; no other session has the key.

Status (PREREG v0.3 section 5): ``complete`` when the editor ended by itself without error; ``timed_out`` at the
wall limit; ``crashed`` when Claude Code ended with an error result or a non-zero exit; ``infra`` on a rate-limit
rejection, an API error status or a harness crash (rerun once). A complete, timed-out or crashed session keeps the
final-patch rule's choice over the patches it checked (contract 5.2). An INFRA session keeps the base model, and the
patch the rule would have chosen is still recorded (``selection.rule_final``). An audit VIOLATION does not change the
record; the primary analysis applies it.

Zero budget (M2NL, an arm without ``patch_check``): the rule keeps the patch of the last valid declaration, applied and
validated by the executor within the arm's limits when it was declared and applied again here (``kind: declared``,
``score`` None); a declaration that failed validation changes nothing. No valid declaration, or a last valid one that
names null, keeps the base, and ``final.reason`` says which, with the executor's messages of a last declaration that
failed validation.

LEARN (a program arm): the candidates are the checked programs, scored by S_joint (``FINAL_RULE_JOINT``); a checked
pick is rebuilt from its derived patch like a checked patch, ``final/`` also keeps the program file, its
``deltas.npz`` and its run record, and ``final.program`` names them with their hashes (None for a base final). The
declaration is recorded only. A session with ``kept_rule`` ``joint-retain-v1`` (``weightcode.harness.retention``, dev
only) applies the same rule to the checked programs with score_old >= its ``s_old_floor`` (``FINAL_RULE_JOINT_RETAIN``).
A session with ``kept_rule`` ``fscil-retain-v2`` (PREREG E9 v2, ``weightcode.harness.fscil.session_kept_v2``) keeps by
``weightcode.evalkit.fscil.retain_v2_pick`` on the checks' fscil-cb-v1 scores (``FINAL_RULE_FSCIL_V2``, ``select_final_v2``),
and its candidates and final carry those scores and floor_c.

Hidden check (K0a of the PREREG information ladder: session.json ``e11_variant`` "named_nocheck", ``e11_view``
``hidden_check`` ``weightcode.harness.e11.HIDDEN_CHECK``). The editor had no ``patch_check``; its ``declare_final`` ran and
validated each declared program within the limits and scored nothing. Finalisation takes the last valid declaration
(contract 5.2 with a budget of zero: the one declared final), re-applies that attempt's derived patch, checks the
child once on V_check with the session's one query and appends the check to ``logs/checks.jsonl`` marked
``hidden: true``, then applies the session's kept rule (joint-retain-v1) to that one check with the positive-budget
rule: eligible (score_old >= s_old_floor and score > 0) keeps it, otherwise the base. No declaration, a last valid one
that names null, a check the oracle charges (non-finite logits) and a status that keeps the base all keep the base,
and ``hidden_check`` in selection.json and session_result.json says which, with the check's numbers when it ran.
The rule never falls back to the zero-budget pick of an unchecked declaration, and a K0a session with a check logged
before finalisation, another budget than one query or no kept rule is a ``FinalizeError`` (recorded as INFRA). The
declared program's V_select readout comes from ``weightcode.analysis.session_info``, which reads every logged check.

Audit (coordinator decision 2026-11-03): while ``weightcode.audit`` does not export ``audit_session``, the record says
``audit: {"status": "not_run", "reason": ...}`` and carries no verdict; nothing is guessed. An audit that exists but
fails is recorded as ``status: "error"`` and raised as ``AuditUnavailable`` after every record is written. Only
``skip_audit=True`` (tests) records ``audit: {"status": "skipped"}``.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import params_sha256, save_params
from weightcode.evalkit.metrics import Candidate, choose_final
from weightcode.harness.arms import APPLY_BY_INDEX_TO_REAL, APPLY_ZERO, arm_spec
from weightcode.harness.jsonio import append_jsonl, file_sha256, iter_lines, read_json, read_jsonl, utc_now, write_json
from weightcode.harness.launch import KILL_REASON_WALL, LaunchOutcome
from weightcode.harness import fscil, loop
from weightcode.harness.ledger import CHECK_RECORD_SCHEMA, JOINT_REPLAY_FIELDS, CheckLedger
from weightcode.harness.program import DELTAS_FILE, DERIVED_PATCH, RUN_RECORD
from weightcode.harness.retention import KEPT_RULE_RETAIN, session_floor
from weightcode.harness.session import MANIFEST_SCHEMA, SESSION_RESULT_SCHEMA, SessionPaths, manifest_path
from weightcode.harness.stream import StreamSummary, infra_reasons, parse_stream
from weightcode.harness.tools import last_valid_declaration
from weightcode.patch import Limits, apply_patch_file, load_patch
from weightcode.sandbox import recover_interrupted_cell

STATUS_COMPLETE = "complete"
STATUS_TIMED_OUT = "timed_out"
STATUS_CRASHED = "crashed"
STATUS_INFRA = "infra"
STATUS_ABORTED_SELFTEST = "aborted_selftest"
STATUS_ABORTED_LEAK = "aborted_leak"
STATUS_DRY_RUN = "dry_run"
KEEPS_FINAL_PATCH = frozenset({STATUS_COMPLETE, STATUS_TIMED_OUT, STATUS_CRASHED})

RUN_SCHEMA = "weightcode.editor_run.v0"
SELECTION_SCHEMA = "weightcode.harness.selection.v0"
PARENT_INSTANCE = "instance"
PARENT_DECOY = "decoy_readdressed"
PARENT_ZEROS = "zeros"
FINAL_DIR = "final"
FINAL_MODEL = "final/model.npz"
FINAL_PATCH = "patch.json"
EDITOR_PATCH = "editor_patch.json"
KIND_BASE = "base"
KIND_CHECKED = "checked"
KIND_DECLARED = "declared"
FINAL_RULE = (
    "contract 5.2: argmax S = fixed - broken over the checked patches and the unedited model (S = 0); ties go to "
    "fewer changed parameters, then to the earlier query; with a budget of zero queries, the patch of the last valid "
    "declaration (the unedited model if it names null or there is none)"
)
FINAL_RULE_JOINT = (
    "LEARN: argmax S_joint = score_new + score_old over the checked programs and the unedited model (S_joint = 0); "
    "ties go to fewer changed parameters, then to the earlier query; the kept child is the checked program's derived "
    "patch applied to the real model"
)
FINAL_RULE_JOINT_RETAIN = (
    "LEARN, kept rule " + KEPT_RULE_RETAIN + ": argmax S_joint = score_new + score_old over the checked programs with "
    "score_old >= the session's s_old_floor and the unedited model (S_joint = 0); ties go to fewer changed parameters, "
    "then to the earlier query; the kept child is the checked program's derived patch applied to the real model"
)
FINAL_RULE_LOOP = (
    "LEARN loop round, kept rule " + loop.KEPT_RULE_LOOP + ": argmax score = score_new + score_old + every "
    "score_prev_k over the checked programs with score_old >= s_old_floor and every score_prev_k >= its s_prev_floor, "
    "and the unedited loop student (score = 0); ties go to fewer changed parameters, then to the earlier query; the "
    "kept child is the checked program's derived patch applied to the loop student"
)
FINAL_RULE_LOOP_V2 = FINAL_RULE_LOOP.replace(
    loop.KEPT_RULE_LOOP + ":", loop.KEPT_RULE_LOOP_V2 + ":").replace(
    "score_old >= s_old_floor", "score_old >= s_old_floor (the stream's remaining clean budget)")
FINAL_RULE_FSCIL_V2 = (
    "LEARN, kept rule " + fscil.KEPT_RULE_V2 + " on check score " + fscil.CHECK_SCORE_V2 + " (PREREG E9 v2): argmax S "
    "over the checked programs with B_c >= floor_c = min(0, R - 0.11 + 164.5 sqrt(d_c / n_c + d_c / n_s)) and the "
    "unedited model (S = 0, B_c = 0); a program is kept only with S > 0 (|S| <= 1e-9 points is a tie, and a tie at 0 "
    "goes to the unedited model); ties among programs go to the earlier query; the kept child is the checked "
    "program's derived patch applied to the real model"
)
RULE_REASON = "final-patch rule (contract 5.2)"
HIDDEN_CHECK_SCHEMA = "weightcode.harness.hidden_check.v1"
HIDDEN_NON_FINITE = "the declared program's child produces non-finite logits on the check set"
JOINT_FINAL_FIELDS = ("n", "n_base_errors", *JOINT_REPLAY_FIELDS)
SERVER_WAIT_S = 30.0


class AuditUnavailable(RuntimeError):
    """The audit could not run; every session record was written first."""


class FinalizeError(RuntimeError):
    """The final patch could not be rebuilt; the session was recorded with the base model and status infra."""


# ---------------------------------------------------------------------------------------------------------------
# Server, cells and status


def wait_for_server(paths: SessionPaths, timeout_s: float = SERVER_WAIT_S) -> dict[str, Any]:
    """Wait until no MCP server holds the session lock; SIGTERM, then SIGKILL, a server that outlives ``timeout_s``."""
    info: dict[str, Any] = {"waited_s": 0.0, "terminated_pid": None, "signal": None}
    if not paths.server_lock.exists():
        info["note"] = "no server ever started"
        return info
    fd = os.open(paths.server_lock, os.O_RDWR | os.O_CLOEXEC)
    began = time.monotonic()
    try:
        for sig in (None, signal.SIGTERM, signal.SIGKILL):
            if sig is not None:
                pid = int(paths.server_pid.read_text(encoding="ascii").strip())
                info.update(terminated_pid=pid, signal=signal.Signals(sig).name)
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(0.05)
                    continue
                fcntl.flock(fd, fcntl.LOCK_UN)
                info["waited_s"] = round(time.monotonic() - began, 3)
                return info
        raise RuntimeError(f"the MCP server of {paths.root.name} still holds its lock after SIGKILL")
    finally:
        os.close(fd)


def server_faults(paths: SessionPaths) -> list[str]:
    """Faults the MCP server logged (``logs/server.jsonl`` events ``fault`` and ``refused_second_server``)."""
    faults = []
    for line in iter_lines(paths.server_log):
        try:
            event = json.loads(line)
        except ValueError:
            faults.append("server.jsonl has an unreadable line")
            continue
        if event.get("event") == "fault":
            faults.append(f"mcp server: {event.get('error')}")
        elif event.get("event") == "refused_second_server":
            faults.append("mcp server: a second server for the session was refused")
    return faults


def cell_counts(paths: SessionPaths, max_cells: int) -> dict[str, Any]:
    """Cells logged, executed, refused, timed out and interrupted, from the cell log."""
    records = read_jsonl(paths.cells_log)
    statuses: dict[str, int] = {}
    for record in records:
        statuses[str(record.get("status"))] = statuses.get(str(record.get("status")), 0) + 1
    return {
        "logged": len(records),
        "used": sum(1 for record in records if record.get("executed")),
        "budget": max_cells,
        "by_status": dict(sorted(statuses.items())),
        "restarts": sum(1 for record in records if record.get("restarted")),
    }


def classify(launch: LaunchOutcome, stream: StreamSummary, faults: Sequence[str]) -> tuple[str, list[str]]:
    """The session status and, for INFRA, the reasons (PREREG section 5)."""
    errors = ([launch.error] if launch.error else []) + list(faults)
    reasons = infra_reasons(stream, harness_errors=errors, stream_expected=launch.stream_expected,
                            killed_by_harness=launch.killed, exit_code=launch.exit_code)
    if reasons:
        return STATUS_INFRA, reasons
    if launch.kill_reason == KILL_REASON_WALL:
        return STATUS_TIMED_OUT, []
    if launch.stream_expected:
        result = stream.result
        if launch.exit_code != 0 or result is None or result.get("is_error") is True:
            return STATUS_CRASHED, []
    elif launch.exit_code != 0:
        return STATUS_INFRA, [f"harness_exception: the MCP server exited with status {launch.exit_code}"]
    return STATUS_COMPLETE, []


def rejected_attempts(paths: SessionPaths) -> list[dict[str, Any]]:
    """Every rejected patch: tool, file, the executor's full messages and the text the editor was shown."""
    rejected = []
    for path in sorted(paths.attempts.glob("a*/attempt.json")):
        record = read_json(path)
        if record.get("outcome") == "invalid":
            rejected.append({key: record.get(key) for key in ("attempt", "tool", "patch_file", "time", "messages",
                                                              "editor_messages", "redaction")}
                            | {key: record[key] for key in ("program_file", "stage") if key in record})
    return rejected


# ---------------------------------------------------------------------------------------------------------------
# Final patch


def select_final(checked: Sequence[Mapping[str, Any]], *, budget: int, declared: Mapping[str, Any] | None,
                 s_old_floor: int | None = None, loop_rule: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The final pick under contract 5.2, as a record with ``kind`` base, checked or declared.

    ``declared`` is the declaration the zero-budget rule uses (``finalize_session`` passes the last valid one); with a
    positive budget it is ignored. ``s_old_floor`` (kept rule ``joint-retain-v1``, ``weightcode.harness.retention``) leaves
    out every checked program with score_old below it before the rule picks; None is the v1 rule. ``loop_rule`` (kept
    rule ``loop-retain-v1`` or ``loop-retain-v2``, ``weightcode.harness.loop``) leaves out every checked program below either
    of its floors.
    """
    if s_old_floor is not None and loop_rule is not None:
        raise FinalizeError("a session has one kept rule, not both the retention and the loop rule")
    if budget == 0:
        if s_old_floor is not None or loop_rule is not None:
            raise FinalizeError("the retention kept rule needs checked programs; this session has no queries")
        if declared is not None and declared.get("valid") and declared.get("patch_file") is not None:
            return {"kind": KIND_DECLARED, "patch_id": declared["patch_file"], "query_index": None,
                    "attempt": declared["attempt"], "child_sha256": declared["child_sha256"]}
        return {"kind": KIND_BASE, "patch_id": None, "query_index": None}
    if s_old_floor is not None:
        checked = [entry for entry in checked if int(entry["score_old"]) >= s_old_floor]
    if loop_rule is not None:
        checked = [entry for entry in checked if loop.eligible(entry, loop_rule)]
    candidates = [Candidate(patch_id=str(entry["patch_id"]), score=int(entry["score"]),
                            n_changed=int(entry["n_changed"]), query_index=int(entry["query_index"]))
                  for entry in checked]
    chosen = choose_final(candidates)
    if chosen.patch_id is None:
        return {"kind": KIND_BASE, "patch_id": None, "query_index": None}
    return {"kind": KIND_CHECKED, "patch_id": chosen.patch_id, "query_index": chosen.query_index}


def select_final_v2(checked: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    """The final pick under fscil-retain-v2 (PREREG E9 v2): ``weightcode.evalkit.fscil.retain_v2_pick`` over the checked
    programs in query order, with ``rule``'s R and base-row counts; the pick must be a check whose recorded
    ``eligible`` is true. Returns a ``select_final`` record (kind base or checked)."""
    from weightcode.evalkit.fscil import retain_v2_pick

    entries = list(checked)
    if [int(entry["query_index"]) for entry in entries] != sorted(int(entry["query_index"]) for entry in entries):
        raise FinalizeError("the checked programs are not in query order")
    index = retain_v2_pick(entries, float(rule["r_points"]), int(rule["n_c"]), int(rule["n_s"]))
    if index is None:
        return {"kind": KIND_BASE, "patch_id": None, "query_index": None}
    entry = entries[index]
    if entry.get("eligible") is not True:
        raise FinalizeError(f"the v2 pick {entry['patch_id']!r} is not recorded as eligible")
    return {"kind": KIND_CHECKED, "patch_id": str(entry["patch_id"]), "query_index": int(entry["query_index"])}


def rule_reason(budget: int, pick: Mapping[str, Any], declarations: Sequence[Mapping[str, Any]]) -> str:
    """Why the rule chose ``pick``: the rule name, and with zero queries which declaration counted."""
    if budget != 0:
        return RULE_REASON
    if pick["kind"] == KIND_DECLARED:
        why = f"the last valid declaration names {pick['patch_id']} (attempt {pick['attempt']}), kept unchecked"
    elif not declarations:
        why = "no declaration was made, so the base model is kept"
    elif last_valid_declaration(declarations) is None:
        why = "no valid declaration was made, so the base model is kept"
    else:
        why = "the last valid declaration names the original model, so the base model is kept"
    last = declarations[-1] if declarations else None
    if last is not None and not last.get("valid"):
        messages = last.get("executor_messages") or last.get("messages") or []
        why += (f"; the last declaration, {last.get('patch_file')} (attempt {last.get('attempt')}), failed "
                f"validation: {'; '.join(str(message) for message in messages)}")
    return f"{RULE_REASON}, zero queries: {why}"


def is_hidden_check(session: Mapping[str, Any]) -> bool:
    """True for a K0a session (module docstring): ``e11_variant`` "named_nocheck" with a hidden check in
    ``e11_view``."""
    from weightcode.harness.e11 import HIDDEN_CHECK, NAMED_NOCHECK

    hidden = session.get("e11_variant") == NAMED_NOCHECK
    if hidden != ((session.get("e11_view") or {}).get("hidden_check") == HIDDEN_CHECK):
        raise FinalizeError("session.json e11_variant and e11_view disagree on the hidden check")
    return hidden


def run_hidden_check(paths: SessionPaths, session: Mapping[str, Any], ledger: CheckLedger,
                     declared: Mapping[str, Any] | None) -> dict[str, Any]:
    """K0a's one post-session check of ``declared``, the last valid declaration (module docstring); returns the
    ``hidden_check`` record. The check is appended to checks.jsonl and to ``ledger`` like a tool check."""
    from weightcode.harness.e11 import HIDDEN_CHECK

    if ledger.entries:
        raise FinalizeError("a K0a session has checks logged before its hidden check")
    if ledger.budget != 1 or session.get("kept_rule", {}).get("name") != KEPT_RULE_RETAIN:
        raise FinalizeError("a K0a session has one query and keeps under joint-retain-v1")
    record: dict[str, Any] = {"schema": HIDDEN_CHECK_SCHEMA, "rule": HIDDEN_CHECK, "time": utc_now(), "ran": False,
                              "declaration": None if declared is None else {
                                  key: declared.get(key) for key in ("time", "program_file", "attempt",
                                                                     "program_sha256", "child_sha256")}}
    if declared is None:
        return {**record, "reason": "no valid declaration was made"}
    if declared.get("program_file") is None:
        return {**record, "reason": "the last valid declaration names the original model"}
    name, attempt = str(declared["program_file"]), str(declared["attempt"])
    attempt_dir = paths.attempts / attempt
    attempt_record = read_json(attempt_dir / "attempt.json")
    if (attempt_record.get("tool") != "declare_final" or attempt_record.get("outcome") != "valid"
            or file_sha256(attempt_dir / name) != declared["program_sha256"]
            or attempt_record.get("child_sha256") != declared["child_sha256"]):
        raise FinalizeError(f"the declared program's attempt {attempt} does not match its declaration")
    result = ledger.applier.apply(load_patch(attempt_dir / DERIVED_PATCH), attempt_dir)
    if result.child_sha256 != declared["child_sha256"]:
        raise FinalizeError("re-applying the declared program's derived patch gives another child")
    base_record = {"schema": CHECK_RECORD_SCHEMA, "time": utc_now(), "patch_file": DERIVED_PATCH, "program_file": name,
                   "program_sha256": declared["program_sha256"], "deltas_sha256": declared["deltas_sha256"],
                   "attempt": attempt, "apply_mode": ledger.applier.apply_mode,
                   "files_sha256": dict(attempt_record["files_sha256"]), "hidden": True}
    try:
        entry = ledger.oracle.query(result.child, name)
    except FloatingPointError:
        charge = dict(base_record, charged_failure=True, patch_id=name, reason=HIDDEN_NON_FINITE,
                      child_sha256=result.child_sha256, executor=result.summary())
        append_jsonl(paths.checks_log, charge)
        ledger.record(charge)
        return {**record, "ran": True, "charged_failure": True, "reason": HIDDEN_NON_FINITE}
    if entry["child_sha256"] != result.child_sha256:
        raise FinalizeError("the oracle hashed the declared child differently from the executor")
    logged = dict(base_record, **entry, executor=result.summary())
    append_jsonl(paths.checks_log, logged)
    ledger.record(logged)
    return {**record, "ran": True, "query_index": entry["query_index"],
            **{key: entry[key] for key in ("score", "score_new", "score_old", "fixed", "broken", "n_changed")}}


def hidden_check_reason(hidden: Mapping[str, Any], pick: Mapping[str, Any], floor: int) -> str:
    """Why K0a's final is what it is: the hidden check's outcome and the kept rule's verdict."""
    if not hidden["ran"] or hidden.get("charged_failure"):
        why = f"{hidden['reason']}, so the base model is kept"
    elif pick["kind"] == KIND_CHECKED:
        why = (f"the last valid declaration, {hidden['declaration']['program_file']} (attempt "
               f"{hidden['declaration']['attempt']}), was checked once after the session: score {hidden['score']}, "
               f"score_old {hidden['score_old']} >= {floor}, so it is kept")
    else:
        why = (f"the last valid declaration, {hidden['declaration']['program_file']} (attempt "
               f"{hidden['declaration']['attempt']}), was checked once after the session: score {hidden['score']}, "
               f"score_old {hidden['score_old']} (floor {floor}); it is not eligible, so the base model is kept")
    return f"{RULE_REASON}, hidden check ({hidden['rule']}): {why}"


def final_parent(apply_mode: str) -> str:
    return {APPLY_BY_INDEX_TO_REAL: PARENT_DECOY, APPLY_ZERO: PARENT_ZEROS}.get(apply_mode, PARENT_INSTANCE)


def _base_final(real_sha: str, reason: str, parent: str = PARENT_INSTANCE) -> dict[str, Any]:
    return {
        "kind": KIND_BASE, "is_base_noop": True, "patch_id": None, "query_index": None, "attempt": None,
        "dir": None, "parent": parent, "model": FINAL_MODEL, "child_sha256": real_sha, "score": 0,
        "fixed": 0, "broken": 0, "base_acc": None, "child_acc": None, "n_changed": 0, "edit_size": 0,
        "edit_frac": 0.0, "executor_n_changed": 0, "editor_patch": None, "reason": reason,
    }


def write_final(paths: SessionPaths, ledger: CheckLedger, pick: Mapping[str, Any], declared: Mapping[str, Any] | None,
                reason: str) -> dict[str, Any]:
    """Write ``final/model.npz`` and, for a patch, ``final/patch.json`` re-applicable to ``final.parent``."""
    applier = ledger.applier
    final_dir = paths.final
    final_dir.mkdir(exist_ok=True)
    if pick["kind"] == KIND_BASE:
        record = _base_final(applier.real_sha256, reason, loop.PARENT_LOOP if ledger.loop else PARENT_INSTANCE)
        saved = save_params(paths.final_model, applier.real)
        if saved != applier.real_sha256:
            raise FinalizeError("final/model.npz of the base model does not hash to the real model")
        return record
    program: dict[str, Any] | None = None
    if pick["kind"] == KIND_CHECKED:
        entry = ledger.entry_for(str(pick["patch_id"]))
        result = ledger.reapply(entry)
        attempt, patch_file = str(entry["attempt"]), str(entry["patch_file"])
        numbers = {key: entry[key] for key in ("score", "fixed", "broken", "base_acc", "child_acc", "n_changed")}
        if ledger.joint:
            numbers.update({key: entry[key] for key in JOINT_FINAL_FIELDS + loop.part_fields(ledger.n_prev)})
            if ledger.fscil_v2 is not None:  # PREREG E9 v2: the kept check's fscil-cb-v1 scores and floor
                numbers.update({key: entry[key] for key in (*fscil.V2_SHOWN_FIELDS, *fscil.V2_COUNT_FIELDS)})
            program = {"file": f"{FINAL_DIR}/{entry['program_file']}", "sha256": entry["program_sha256"],
                       "deltas": f"{FINAL_DIR}/{DELTAS_FILE}", "deltas_sha256": entry["deltas_sha256"],
                       "patch": f"{FINAL_DIR}/{FINAL_PATCH}", "run_record": f"{FINAL_DIR}/{RUN_RECORD}"}
    else:
        if declared is None:
            raise FinalizeError("a declared final without a declaration")
        attempt, patch_file = str(declared["attempt"]), str(declared["patch_file"])
        attempt_dir = paths.attempts / attempt
        result = applier.apply(load_patch(attempt_dir / patch_file), attempt_dir)
        if result.child_sha256 != declared["child_sha256"]:
            raise FinalizeError("re-applying the declared patch gives another child")
        numbers = {"score": None, "fixed": None, "broken": None, "base_acc": None, "child_acc": None,
                   "n_changed": int(np.sum([np.count_nonzero(applier.real[k] != result.child[k]) for k in applier.real]))}
    attempt_dir = paths.attempts / attempt
    saved = save_params(paths.final_model, result.child)
    if saved != result.child_sha256:
        raise FinalizeError("final/model.npz does not hash to the final child")
    for item in sorted(attempt_dir.iterdir()):
        if item.name not in (patch_file, "attempt.json") and item.is_file():
            shutil.copyfile(item, final_dir / item.name)
    parent = loop.PARENT_LOOP if ledger.loop else final_parent(applier.apply_mode)
    editor_patch = None
    if parent == PARENT_DECOY:
        shutil.copyfile(attempt_dir / patch_file, final_dir / EDITOR_PATCH)
        editor_patch = f"{FINAL_DIR}/{EDITOR_PATCH}"
        readdressed = json.loads((attempt_dir / patch_file).read_text(encoding="utf-8"))
        readdressed["parent_sha256"] = applier.real_sha256
        (final_dir / FINAL_PATCH).write_text(json.dumps(readdressed, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    else:
        shutil.copyfile(attempt_dir / patch_file, final_dir / FINAL_PATCH)
    verify_final_patch(paths, ledger, parent, result.child_sha256)
    if program is not None:
        for key, path in (("sha256", program["file"]), ("deltas_sha256", program["deltas"])):
            if file_sha256(paths.root / path) != program[key]:
                raise FinalizeError(f"{path} does not hash to the checked program's record")
        program["patch_sha256"] = file_sha256(paths.final / FINAL_PATCH)
    record = {
        "kind": pick["kind"], "is_base_noop": False, "patch_id": pick["patch_id"],
        "query_index": pick.get("query_index"), "attempt": attempt, "dir": FINAL_DIR, "parent": parent,
        "model": FINAL_MODEL, "child_sha256": result.child_sha256, **numbers, "edit_size": result.edit_size,
        "edit_frac": result.edit_frac, "executor_n_changed": result.n_changed, "editor_patch": editor_patch,
        "reason": reason,
    }
    if ledger.joint:
        record["program"] = program
    return record


def verify_final_patch(paths: SessionPaths, ledger: CheckLedger, parent: str, child_sha256: str) -> None:
    """``final/patch.json`` applied to the parent named in run.json must give the final child."""
    applier = ledger.applier
    if parent == PARENT_ZEROS:
        params = {key: np.zeros_like(value) for key, value in applier.real.items()}
    else:
        params = applier.real
    result = apply_patch_file(params, paths.final / FINAL_PATCH, applier.limits, params_sha256(params))
    if result.child_sha256 != child_sha256:
        raise FinalizeError(f"final/{FINAL_PATCH} applied to the {parent} parent does not give the final child")


# ---------------------------------------------------------------------------------------------------------------
# Audit and records


def run_audit(paths: SessionPaths, skip_audit: bool) -> dict[str, Any]:
    """Audit the cell log with ``weightcode.audit.audit_session``; the result's ``status`` says what happened."""
    if skip_audit:
        return {"status": "skipped"}
    import weightcode.audit as audit_package

    audit_session = getattr(audit_package, "audit_session", None)
    if audit_session is None:
        return {"status": "not_run", "reason": "weightcode.audit does not export audit_session yet; no verdict"}
    try:
        report = audit_session(paths.cells_log)
        data = report.to_dict()
    except Exception as exc:  # recorded, then raised by finalize_session after the records are written
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    write_json(paths.audit_json, data)
    return {
        "status": "done",
        "verdict": data.get("verdict"),
        "path": paths.audit_json.name,
        "sha256": file_sha256(paths.audit_json),
        "n_flags": len(data.get("flags", [])),
        "rule_counts": data.get("rule_counts"),
        "candidate_count": data.get("candidate_count"),
        "candidate_count_is_lower_bound": data.get("candidate_count_is_lower_bound"),
        "auditor_sha256": data.get("auditor_sha256"),
    }


def run_record(result: Mapping[str, Any]) -> dict[str, Any]:
    """The run.json projection of session_result.json (the shared run-record format of the dev readout)."""
    final = result["final"]
    return {
        "schema": RUN_SCHEMA,
        "method": result["arm"],
        "status": result["status"],
        "session_id": result["session_id"],
        "repeat": result["repeat"],
        "created_at": result["created_at"],
        "finalized_at": result["finalized_at"],
        "instance": dict(result["instance"]),
        "flags": {
            "edit_limits_enforced": result["edit_limits_enforced"],
            "infra": result["infra"],
            "audit_verdict": result["audit"].get("verdict"),
            "audit_status": result["audit"]["status"],
        },
        "final": {key: final[key] for key in ("is_base_noop", "dir", "parent", "model", "child_sha256", "score",
                                              "fixed", "broken", "edit_size", "edit_frac", "n_changed", "patch_id",
                                              "query_index", "kind", "editor_patch", "reason")}
        | {key: final[key] for key in ("score_new", "score_old", "program") if key in final}
        | {key: value for key, value in final.items() if key.startswith("score_prev_")},
        **({"loop": dict(result["loop"])} if "loop" in result else {}),
        **({"e11_variant": result["e11_variant"]} if "e11_variant" in result else {}),
        "queries_used": result["queries"]["used"],
        "budget": result["queries"]["budget"],
        "editor": {key: result["editor"][key] for key in ("model", "effort", "launcher")},
        "session_result": "session_result.json",
    }


def write_records(paths: SessionPaths, result: dict[str, Any], *, manifest: bool = True) -> None:
    """session_result.json, run.json and (unless a dry run) one manifest line."""
    write_json(paths.result_json, result)
    write_json(paths.root / "run.json", run_record(result))
    if manifest:
        append_jsonl(manifest_path(), {
            "schema": MANIFEST_SCHEMA,
            "time": utc_now(),
            "session_id": result["session_id"],
            "arm": result["arm"],
            "repeat": result["repeat"],
            "instance_id": result["instance"]["instance_id"],
            **({"e11_variant": result["e11_variant"]} if "e11_variant" in result else {}),
            "status": result["status"],
            "infra": result["infra"],
            "root": str(paths.root),
            "final_child_sha256": result["final"]["child_sha256"],
            "is_base_noop": result["final"]["is_base_noop"],
            "score": result["final"]["score"],
            "audit_verdict": result["audit"].get("verdict"),
        }, lock=True)


def base_result(session: Mapping[str, Any], status: str, **fields: Any) -> dict[str, Any]:
    """The fields every session_result.json has, for the session in ``session``."""
    return {
        "schema": SESSION_RESULT_SCHEMA,
        "session_id": session["session_id"],
        "arm": session["arm"],
        "repeat": session["repeat"],
        "status": status,
        "created_at": session["created_at"],
        "finalized_at": utc_now(),
        "instance": {
            "instance_id": session["instance_id"],
            "split_role": session["split_role"],
            "dataset": session["dataset"],
            "condition": session["condition"],
            "params_sha256": session["real_model_sha256"],
            "forward_sha256": session["forward_sha256"],
            "evidence_rule": session["evidence_rule"],
        },
        "edit_limits_enforced": bool(session["edit_limits_enforced"]),
        "editor": dict(session["editor"]),
        **({"loop": {key: session["loop"][key] for key in ("stream_id", "round", "student_sha256",
                                                           "base_params_sha256")}} if "loop" in session else {}),
        **({"e11_variant": session["e11_variant"]} if "e11_variant" in session else {}),
        **fields,
    }


def finalize_session(
    paths: SessionPaths,
    session: Mapping[str, Any],
    launch: LaunchOutcome,
    *,
    skip_audit: bool = False,
    provenance_after: Mapping[str, Any] | None = None,
    code_changes: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalise one launched session and return its session_result record (see the module docstring).

    ``extra`` holds pre-launch records (the self-test summary and the leak scan) that session_result.json repeats.
    """
    server = wait_for_server(paths)
    recovered = recover_interrupted_cell(paths.cells_log)
    stream = parse_stream(paths.transcript)
    faults = server_faults(paths)
    status, reasons = classify(launch, stream, faults)
    declared_all = read_jsonl(paths.declared_log)
    declared = declared_all[-1] if declared_all else None
    rule_declared = last_valid_declaration(declared_all)  # what the zero-budget rule uses
    budget = int(session["budget"]["patch_checks"])
    joint = arm_spec(str(session["arm"])).program
    finalize_error: str | None = None
    loop_rule = loop.session_loop_rule(session)
    fscil_v2 = fscil.session_kept_v2(session)  # PREREG E9 v2, or None
    floor = None if loop_rule is not None or fscil_v2 is not None else session_floor(session)
    loop_text = (None if loop_rule is None
                 else FINAL_RULE_LOOP_V2 if loop_rule["name"] == loop.KEPT_RULE_LOOP_V2 else FINAL_RULE_LOOP)
    rule = (FINAL_RULE if not joint else loop_text if loop_text is not None else FINAL_RULE_FSCIL_V2
            if fscil_v2 is not None else FINAL_RULE_JOINT if floor is None else FINAL_RULE_JOINT_RETAIN)
    selection: dict[str, Any] = {"schema": SELECTION_SCHEMA, "rule": rule, "budget": budget,
                                 **({"s_old_floor": floor} if floor is not None else {}),
                                 **({"kept_rule": loop_rule} if loop_rule is not None else {}),
                                 **({"kept_rule": fscil_v2} if fscil_v2 is not None else {})}
    parent_base = loop.PARENT_LOOP if loop_rule is not None else PARENT_INSTANCE
    hidden: dict[str, Any] | None = None
    try:
        hidden_session = is_hidden_check(session)
        ledger = CheckLedger(session, paths)
        if hidden_session:  # K0a: the one post-session check of the last valid declaration (module docstring)
            if floor is None or not joint:
                raise FinalizeError("a K0a session is a program session under joint-retain-v1")
            hidden = run_hidden_check(paths, session, ledger, rule_declared)
        checked = ledger.checked()
        if (floor is not None or loop_rule is not None or fscil_v2 is not None) and not joint:
            raise FinalizeError("the retention kept rule applies to a program arm only")
        if fscil_v2 is not None:
            if budget == 0:
                raise FinalizeError("the E9 v2 kept rule needs checked programs; this session has no queries")
            rule_pick = select_final_v2(checked, fscil_v2)
        else:
            rule_pick = select_final(checked, budget=budget, declared=rule_declared, s_old_floor=floor,
                                     loop_rule=loop_rule)
        if status in KEEPS_FINAL_PATCH:
            pick = rule_pick
            reason = (rule_reason(budget, rule_pick, declared_all) if hidden is None
                      else hidden_check_reason(hidden, rule_pick, int(floor or 0)))
        else:
            pick = {"kind": KIND_BASE, "patch_id": None, "query_index": None}
            reason = f"status {status}: the session keeps the base model (PREREG section 5)"
        final = write_final(paths, ledger, pick, rule_declared, reason)
        selection.update(
            candidates=[{key: entry.get(key) for key in ("patch_id", "query_index", "score", "fixed", "broken",
                                                         "n_changed", "child_sha256", "attempt")}
                        | {"edit_size": entry["executor"]["edit_size"], "edit_frac": entry["executor"]["edit_frac"]}
                        | ({key: entry[key] for key in ("score_new", "score_old", "program_file", "program_sha256",
                                                        "deltas_sha256")} if joint else {})
                        | {key: entry[key] for key in loop.part_fields(ledger.n_prev)}
                        | ({"eligible": loop.eligible(entry, loop_rule)} if loop_rule is not None else {})
                        | ({key: entry[key] for key in (*fscil.V2_SHOWN_FIELDS, *fscil.V2_COUNT_FIELDS)}
                           if fscil_v2 is not None else {})
                        for entry in checked],
            charged_failures=[{"patch_id": entry["patch_id"], "reason": entry["reason"]} for entry in ledger.charged],
            rule_final=rule_pick,
            queries_used=ledger.queries_used,
            **({"hidden_check": hidden} if hidden is not None else {}),
        )
        queries = {"used": ledger.queries_used, "consumed": len(checked), "charged_failures": len(ledger.charged),
                   "budget": budget}
    except Exception as exc:  # recorded as INFRA with the base model, then raised after the records are written
        finalize_error = f"{type(exc).__name__}: {exc}"
        status = STATUS_INFRA
        reasons = [*reasons, f"harness_exception: finalisation: {finalize_error}"]
        final = _base_final(str(session["real_model_sha256"]), "finalisation failed; base model recorded", parent_base)
        selection.update(error=finalize_error, rule_final=None,
                         **({"hidden_check": hidden} if hidden is not None else {}))
        queries = {"used": None, "consumed": None, "charged_failures": None, "budget": budget}
    if joint:
        final.setdefault("program", None)
    selection.update(final=final, declared_final=declared,
                     declared_matches_final=None if declared is None else declared.get("child_sha256")
                     == final["child_sha256"])
    write_json(paths.selection_json, selection)
    audit = run_audit(paths, skip_audit)
    result = base_result(
        session,
        status,
        infra=status == STATUS_INFRA,
        infra_reasons=reasons,
        launch=launch.as_dict(),
        stream=stream.as_dict(),
        wall_s=launch.wall_s,
        tokens=None if stream.result is None else stream.as_dict()["result"]["usage"],
        cells=cell_counts(paths, int(session["budget"]["python_cells"])),
        recovered_cell=recovered,
        queries=queries,
        final=final,
        selection={"path": f"{FINAL_DIR}/{paths.selection_json.name}", "rule_final": selection.get("rule_final"),
                   "n_candidates": len(selection.get("candidates", []))},
        declared_final=declared,
        rejected_attempts=rejected_attempts(paths),
        audit=audit,
        server=server | {"faults": faults},
        finalize_error=finalize_error,
        provenance_after=None if provenance_after is None else dict(provenance_after),
        code_changed_during_session=list(code_changes),
        **({"hidden_check": hidden} if hidden is not None else {}),
        **dict(extra or {}),
    )
    write_records(paths, result)
    if finalize_error is not None:
        raise FinalizeError(f"session {paths.root.name}: {finalize_error} (recorded as INFRA with the base model)")
    if audit["status"] == "error":
        raise AuditUnavailable(f"session {paths.root.name}: the audit failed: {audit['error']} "
                               "(every session record was written)")
    return result


def aborted_result(paths: SessionPaths, session: Mapping[str, Any], status: str, real_params: Mapping[str, Any],
                   *, dry_run: bool = False, **fields: Any) -> dict[str, Any]:
    """Record a session that never launched (self-test or leak-scan refusal): base model, no audit.

    A dry run writes session_result.json only: no run.json and no manifest line, like a dry run that passed.
    """
    paths.final.mkdir(exist_ok=True)
    real_sha = str(session["real_model_sha256"])
    if save_params(paths.final_model, real_params) != real_sha:
        raise FinalizeError("final/model.npz of the base model does not hash to the real model")
    final = _base_final(real_sha, f"status {status}: the editor was never launched",
                        loop.PARENT_LOOP if "loop" in session else PARENT_INSTANCE)
    if arm_spec(str(session["arm"])).program:
        final["program"] = None
    result = base_result(session, status, infra=False, infra_reasons=[], launch=None, stream=None, wall_s=None,
                         tokens=None, cells=cell_counts(paths, int(session["budget"]["python_cells"])),
                         recovered_cell=None,
                         queries={"used": 0, "consumed": 0, "charged_failures": 0,
                                  "budget": int(session["budget"]["patch_checks"])},
                         final=final, selection=None, declared_final=None,
                         audit={"status": "not_run", "reason": f"status {status}"}, server=None, finalize_error=None,
                         dry_run=dry_run, **fields)
    if dry_run:
        write_json(paths.result_json, result)
    else:
        write_records(paths, result)
    return result


__all__ = [
    "AuditUnavailable", "FINAL_RULE", "FINAL_RULE_FSCIL_V2", "FINAL_RULE_JOINT", "FINAL_RULE_JOINT_RETAIN",
    "FinalizeError", "PARENT_DECOY", "PARENT_INSTANCE", "PARENT_ZEROS",
    "RUN_SCHEMA", "STATUS_ABORTED_LEAK", "STATUS_ABORTED_SELFTEST", "STATUS_COMPLETE", "STATUS_CRASHED",
    "STATUS_DRY_RUN", "STATUS_INFRA", "STATUS_TIMED_OUT", "aborted_result", "base_result", "cell_counts", "classify",
    "finalize_session", "hidden_check_reason", "is_hidden_check", "rejected_attempts", "rule_reason",
    "run_hidden_check", "run_audit", "run_record", "select_final",
    "select_final_v2", "server_faults",
    "verify_final_patch",
    "wait_for_server", "write_final", "write_records",
]
