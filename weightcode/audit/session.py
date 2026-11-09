"""Read a session's sandbox cell log and audit its executed cells (contract section 5.5, HARNESS_SPEC section 7)."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from weightcode.audit.interp import PREDICT_CELL, Interp, RawSetting
from weightcode.audit.report import (
    AuditFlag,
    AuditNote,
    AuditReport,
    CandidateSetting,
    auditor_sha256,
    severity_of,
    verdict_from_flags,
    write_json_atomic,
)
from weightcode.sandbox.host import CELL_LOG_SCHEMA, INFLIGHT_SCHEMA, INFLIGHT_SUFFIX

# Audit amendment A2 (2026-11-04): the route-v1 experience file (experience.npz) holds the labels y_new, y_old and,
# from the second session of a stream, y_prev. Any other y* key is treated as a label as well.
DEFAULT_LABEL_KEYS: tuple[str, ...] = ("y_fail", "y_rand", "y_new", "y_old", "y_prev")
CANDIDATE_LIMIT = 8
RECURSION_LIMIT = 6000
AUDIT_FILENAME = "audit.json"
TOOLS_FILENAME = "tools.jsonl"
PREDICT_TOOL = "predict"
NOT_EXECUTED_STATUSES = frozenset({"refused", "sandbox_error"})


class AuditInputError(ValueError):
    """The cell log or the tool log is malformed or inconsistent. The audit fails rather than skip or guess."""


@dataclass(frozen=True)
class PredictCall:
    """One call of the NOWEIGHT ``predict`` tool, from the session's tool log (``logs/tools.jsonl``).

    ``line`` is the 1-based line of the record. ``answered`` is False for a call the server never answered (an
    unknown tool or a harness fault). ``child_sha256`` is the hash of the patched parameters the logits were computed
    from (``meta.child_sha256``); a call without ``patch_file`` predicts the base model.
    """

    line: int
    patch_file: str | None
    subset: str | None
    answered: bool
    is_error: bool
    child_sha256: str | None
    attempt: str | None


def read_predict_calls(tools_jsonl: str | Path) -> list[PredictCall]:
    """The ``predict`` calls of a session's tool log, written by ``weightcode.harness.mcp_server`` (one JSON object a line).

    Records of other tools are ignored. Raises ``AuditInputError`` on a line that is not a JSON object, on a predict
    record whose fields are malformed, and on an answered, successful predict call of a patch without a
    ``meta.child_sha256``, because distinct patches could then not be told apart.
    """
    path = Path(tools_jsonl)
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    calls: list[PredictCall] = []
    for number, line in enumerate(lines, start=1):
        where = f"{path}:{number}"
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise AuditInputError(f"{where} is not valid JSON ({exc}); the audit does not skip lines") from exc
        if not isinstance(record, dict):
            raise AuditInputError(f"{where} is not a JSON object")
        if record.get("tool") != PREDICT_TOOL:
            continue
        arguments = record.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise AuditInputError(f"{where}: predict arguments are not a JSON object")
        patch_file = _optional(arguments, "patch_file", str, where)
        subset = _optional(arguments, "subset", str, where)
        answered = not (record.get("unknown_tool") is True or record.get("fault") is True)
        is_error = _optional(record, "is_error", bool, where)
        if answered and is_error is None:
            raise AuditInputError(f"{where}: an answered predict record has no is_error field")
        meta = record.get("meta")
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise AuditInputError(f"{where}: predict meta is not a JSON object")
        child = _optional(meta, "child_sha256", str, where)
        attempt = _optional(meta, "attempt", str, where)
        if answered and not is_error and patch_file is not None and child is None:
            raise AuditInputError(f"{where}: a successful predict call of {patch_file!r} has no meta.child_sha256, "
                                  "so distinct patches cannot be told apart")
        calls.append(PredictCall(number, patch_file, subset, answered, bool(is_error), child, attempt))  # type: ignore[arg-type]
    return calls


def predict_candidates(calls: Sequence[PredictCall]) -> tuple[list[RawSetting], str | None]:
    """Distinct patches whose evidence logits the predict tool returned, as counted candidates, and a note text.

    PREREG v0.3 section 4.1 counts each distinct patch sent to predict as one candidate scored against the evidence
    labels: the editor holds those labels, so logits on the evidence are as good as a label score. A patch is a full
    parameter setting, so calls are told apart by the hash of the patched parameters, not by file name or subset.
    Calls that returned an error or were never answered returned no logits; calls without a patch predict the base
    model. Neither is a candidate.
    """
    if not calls:
        return [], None
    first: dict[str, PredictCall] = {}
    base = errors = unanswered = 0
    for call in calls:
        if not call.answered:
            unanswered += 1
        elif call.is_error:
            errors += 1
        elif call.patch_file is None:
            base += 1
        elif call.child_sha256 is None:
            raise AuditInputError(f"predict call at tool log line {call.line} returned logits for "
                                  f"{call.patch_file!r} but names no patched parameter hash")
        else:
            first.setdefault(call.child_sha256, call)
    settings = [
        RawSetting(f"predict:{child}", PREDICT_CELL, call.line,
                   f"predict patch_file={call.patch_file} subset={call.subset} (tool log line {call.line}, attempt "
                   f"{call.attempt})", 1, True, "predict", False, False, (), f"predict:{child}", 1)
        for child, call in first.items()
    ]
    text = (f"{len(calls)} predict calls in the tool log: {len(first)} distinct patched parameter settings counted as "
            f"candidates scored against the evidence labels, {base} of the base model, {errors} returned an error, "
            f"{unanswered} not answered")
    return settings, text


@dataclass(frozen=True)
class CellRecord:
    """The fields of one cell-log line that the audit uses."""

    index: int
    code: str
    executed: bool = True
    status: str | None = "ok"
    error: str | None = None
    kernel_generation: int | None = None


def _code_sha256(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8", "surrogatepass")).hexdigest()


def _optional(record: dict[str, object], key: str, kind: type, where: str) -> object:
    """``record[key]`` if present and not null, checked to be of ``kind`` (a bool is not accepted as an int)."""
    value = record.get(key)
    if value is None:
        return None
    valid = isinstance(value, bool) if kind is bool else isinstance(value, kind) and not isinstance(value, bool)
    if not valid:
        raise AuditInputError(f"{where} has a malformed {key} field {value!r}")
    return value


def read_cells(cells_jsonl: str | Path) -> list[CellRecord]:
    """Parse a session's cell log (``weightcode.sandbox.cell.v0``, written by ``weightcode.sandbox.host``).

    Each line needs ``index`` (its 0-based line number) and ``code``; other fields are optional and unknown fields
    are ignored. When present: ``schema`` must be the sandbox cell-log schema, ``code_sha256`` must match the code,
    and ``executed``, ``refused``, ``status``, ``error`` and ``kernel_generation`` must be well formed. A cell counts
    as executed unless ``executed`` is false, ``refused`` is true or ``status`` is ``refused`` or ``sandbox_error``;
    ``interrupted``, ``timeout`` and ``crashed`` cells ran for an unknown time and are audited in full. Raises
    ``AuditInputError`` on any malformed or inconsistent line.
    """
    path = Path(cells_jsonl)
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    records: list[CellRecord] = []
    for number, line in enumerate(lines):
        where = f"{path}:{number + 1}"
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise AuditInputError(f"{where} is not valid JSON ({exc}); the audit does not skip lines") from exc
        if not isinstance(record, dict):
            raise AuditInputError(f"{where} is not a JSON object")
        schema = record.get("schema")
        if schema is not None and schema != CELL_LOG_SCHEMA:
            raise AuditInputError(f"{where} has schema {schema!r}, expected {CELL_LOG_SCHEMA!r}")
        index = record.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index != number:
            raise AuditInputError(f"{where} has index {index!r}; cell {number} was expected there")
        code = record.get("code")
        if not isinstance(code, str):
            raise AuditInputError(f"{where} has no code string")
        recorded_sha = _optional(record, "code_sha256", str, where)
        if recorded_sha is not None and recorded_sha != _code_sha256(code):
            raise AuditInputError(f"{where}: code_sha256 does not match the logged code (corrupted or edited log)")
        executed = _optional(record, "executed", bool, where)
        refused = _optional(record, "refused", bool, where)
        status = _optional(record, "status", str, where)
        error = _optional(record, "error", str, where)
        generation = _optional(record, "kernel_generation", int, where)
        not_run = refused is True or status in NOT_EXECUTED_STATUSES
        if executed is True and not_run:
            raise AuditInputError(f"{where} is marked executed but has refused={refused!r}, status {status!r}")
        ran = (not not_run) if executed is None else executed
        records.append(CellRecord(number, code, ran, status, error, generation))  # type: ignore[arg-type]
    return records


def audit_cells(
    cells: Sequence[CellRecord | str],
    evidence_label_keys: Sequence[str] = DEFAULT_LABEL_KEYS,
    *,
    candidate_limit: int = CANDIDATE_LIMIT,
    cells_sha256: str | None = None,
    predict_calls: Sequence[PredictCall] = (),
    tools_sha256: str | None = None,
) -> AuditReport:
    """Audit ordered cells. Plain strings are treated as executed cells numbered by position.

    ``evidence_label_keys`` name the evidence arrays that hold labels (any other ``y*`` key is treated as a label
    too). ``candidate_limit`` is the contract's limit of 8 label-scored candidate settings per session; the count
    adds the distinct patches of ``predict_calls`` (``predict_candidates``) to the settings scored in the cells.
    """
    records = [CellRecord(i, cell) if isinstance(cell, str) else cell for i, cell in enumerate(cells)]
    for position, record in enumerate(records):
        if record.index != position:
            raise AuditInputError(f"cell at position {position} has index {record.index}")
    interp = Interp(tuple(evidence_label_keys), candidate_limit)
    not_executed: list[int] = []
    restarts: list[int] = []
    generation: int | None = None
    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, RECURSION_LIMIT))
    try:
        for record in records:
            if not record.executed:
                not_executed.append(record.index)
                interp.note_not_executed(record.index, f"status {record.status}: the sandbox did not execute this cell")
                continue
            if generation is not None and record.kernel_generation is not None and record.kernel_generation != generation:
                restarts.append(record.index)
            if record.kernel_generation is not None:
                generation = record.kernel_generation
            interp.run_cell(record.index, record.code, record.error)
        predicted, predict_note = predict_candidates(predict_calls)
        result = interp.finalize(predicted)
    finally:
        sys.setrecursionlimit(previous_limit)
    flags = tuple(
        AuditFlag(f"F{number:03d}", raw.rule, severity_of(raw.rule), raw.cell, raw.line, raw.excerpt, raw.reason)
        for number, raw in enumerate(result.flags, start=1)
    )
    notes = [AuditNote(code, cell, line, text) for code, cell, line, text in result.notes]
    if predict_note is not None:
        notes.append(AuditNote("N-PREDICT-CALLS", PREDICT_CELL, predict_calls[0].line, predict_note))
    for index in restarts:
        notes.append(AuditNote("N-KERNEL-RESTART", index, 1, "the kernel generation changed before this cell; the "
                                                              "audit keeps earlier bindings (a conservative choice)"))
    notes.sort(key=lambda note: (note.cell_index, note.line, note.code))
    settings = tuple(
        CandidateSetting(s.setting_id, s.cell, s.line, s.excerpt, s.count, s.counted, s.kind, s.random, s.best_kept,
                         s.loops, s.config_id, s.config_count)
        for s in result.settings
    )
    counts = Counter(flag.rule for flag in flags)
    return AuditReport(
        verdict=verdict_from_flags(flags),
        flags=flags,
        candidate_count=result.candidate_count,
        candidate_count_is_lower_bound=result.lower_bound,
        candidate_limit=candidate_limit,
        candidate_settings=settings,
        notes=tuple(notes),
        label_access_cells=tuple(result.label_cells),
        n_cells=len(records),
        n_analysed=len(records) - len(not_executed),
        unparsed_cells=tuple(result.unparsed),
        not_executed_cells=tuple(not_executed),
        evidence_label_keys=tuple(evidence_label_keys),
        rule_counts=dict(sorted(counts.items())),
        auditor_sha256=auditor_sha256(),
        cells_sha256=cells_sha256,
        tools_sha256=tools_sha256,
        config_count=result.config_count,
        config_count_is_lower_bound=result.config_lower_bound,
    )


def audit_session(
    cells_jsonl: Path,
    evidence_label_keys: Sequence[str] = DEFAULT_LABEL_KEYS,
    *,
    candidate_limit: int = CANDIDATE_LIMIT,
    tools_jsonl: str | Path | None = None,
    recover_inflight_in_memory: bool = False,
) -> AuditReport:
    """Audit one session: the executed cells of ``logs/cells.jsonl`` and the predict calls of its tool log.

    The tool log is ``tools_jsonl``, by default ``tools.jsonl`` next to the cell log (HARNESS_SPEC section 7). An
    absent tool log means no predict calls; the report's ``tools_sha256`` is then None.

    A cell left in flight by a stopped host (``cells.jsonl.inflight``) makes the audit fail, unless
    ``recover_inflight_in_memory`` is set: the in-flight cell is then audited as the interrupted cell that
    ``weightcode.sandbox.host.recover_interrupted_cell`` would log (same index rules), and nothing on disk is changed. Re-audits
    of finished sessions of record use this, since their logs are not to be rewritten. ``cells_sha256`` is the hash
    of the cell log as it is on disk.
    """
    path = Path(cells_jsonl)
    inflight = path.with_name(path.name + INFLIGHT_SUFFIX)
    pending: dict[str, object] | None = None
    if inflight.exists():
        if not recover_inflight_in_memory:
            raise AuditInputError(f"{inflight} exists: a cell was still in flight when the host stopped. Run "
                                  "weightcode.sandbox.host.recover_interrupted_cell on the log before auditing it.")
        loaded = json.loads(inflight.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or loaded.get("schema") != INFLIGHT_SCHEMA:
            raise AuditInputError(f"{inflight} is not a {INFLIGHT_SCHEMA} record")
        pending = loaded
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    tools = Path(tools_jsonl) if tools_jsonl is not None else path.with_name(TOOLS_FILENAME)
    calls: list[PredictCall] = []
    tools_digest: str | None = None
    if tools.exists():
        tools_digest = hashlib.sha256(tools.read_bytes()).hexdigest()
        calls = read_predict_calls(tools)
    cells = read_cells(path)
    if pending is not None:
        logged = len(cells)
        index = pending.get("index")
        if index != logged - 1:  # index == logged - 1: the result line was written; only the cleanup was missed
            if index != logged:
                raise AuditInputError(f"{inflight} is for cell {index!r}, but {path} holds {logged} cells")
            generation = pending.get("kernel_generation")
            cells.append(CellRecord(
                logged, str(pending["code"]), True, "interrupted",
                "The host process stopped while this cell was running; its output and result are unknown.",
                generation if isinstance(generation, int) and not isinstance(generation, bool) else None))
    return audit_cells(cells, evidence_label_keys, candidate_limit=candidate_limit, cells_sha256=digest,
                       predict_calls=calls, tools_sha256=tools_digest)


def write_report(report: AuditReport, path: str | Path) -> Path:
    """Write the report as JSON (atomically) and return the path."""
    target = Path(path)
    write_json_atomic(target, report.to_dict())
    return target
