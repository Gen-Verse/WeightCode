"""LLM judge for SOFT flags: builds the frozen prompt, runs ``claude -p`` without tools and parses its decisions.

The judge only resolves SOFT flags of a static audit report (``logs/audit.json``). It never changes static VIOLATION
flags. Running it is a separate step from the static audit (HARNESS_SPEC section 7) and is never done in automated
tests, which use a stand-in executable instead of the real CLI.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from weightcode.audit.report import RULES, SOFT, VIOLATION, auditor_sha256, write_json_atomic
from weightcode.audit.session import AUDIT_FILENAME, CellRecord, read_cells

JUDGE_SCHEMA = "weightcode.audit.judge.v0"
JUDGE_FILENAME = "audit_judge.json"
PROMPT_FILENAME = "audit_judge_prompt.md"
TEMPLATE_PATH = Path(__file__).with_name("judge_prompt.md")
PLACEHOLDERS = ("@@SUMMARY@@", "@@FLAGS@@", "@@CELLS@@")
# Linux limits one command-line argument to 128 KiB (MAX_ARG_STRLEN) and the prompt is passed as an argument.
PROMPT_BYTE_LIMIT = 120_000
CONTEXT_LINES = 25
DECISIONS = ("ALLOWED", "VIOLATION")
CONFIDENCES = ("high", "low")
DEFAULT_TIMEOUT_S = 900.0


class JudgeError(RuntimeError):
    """The judge could not be run, or its output does not follow the required format."""


class JudgeFailed(JudgeError):
    """A judge attempt failed after its record was started; the record with status "failed" is at ``record_path``."""

    def __init__(self, message: str, record_path: Path) -> None:
        super().__init__(message)
        self.record_path = record_path


def template_text() -> str:
    """The frozen judge prompt template."""
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    for placeholder in PLACEHOLDERS:
        if text.count(placeholder) != 1:
            raise JudgeError(f"{TEMPLATE_PATH} must contain {placeholder} exactly once")
    return text


def template_sha256() -> str:
    return hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()


def _numbered(code: str, keep: set[int] | None = None) -> str:
    lines = code.split("\n")
    width = len(str(len(lines)))
    out: list[str] = []
    skipped = False
    for number, line in enumerate(lines, start=1):
        if keep is not None and number not in keep:
            if not skipped:
                out.append(f"{'.' * width} | [lines omitted]")
                skipped = True
            continue
        skipped = False
        out.append(f"{number:>{width}} | {line}")
    return "\n".join(out)


def _summary(report: Mapping[str, object]) -> str:
    settings = [s for s in report.get("candidate_settings", []) if isinstance(s, dict)]
    counted = [s for s in settings if s.get("counted")]
    lines = [
        f"- Static verdict: {report.get('verdict')}.",
        f"- Executed cells analysed: {report.get('n_analysed')} of {report.get('n_cells')} logged.",
        f"- Distinct candidate settings scored against label-dependent values: {report.get('candidate_count')}"
        f"{' or more' if report.get('candidate_count_is_lower_bound') else ''} (limit {report.get('candidate_limit')}).",
        f"- Cells that read evidence labels: {list(report.get('label_access_cells', []))}.",
    ]
    for setting in counted[:20]:
        if setting.get("kind") == "predict":
            lines.append(f"  - counted: predict tool call, tool log line {setting.get('line')}: "
                         f"`{setting.get('excerpt')}`")
            continue
        lines.append(
            f"  - counted: cell {setting.get('cell_index')} line {setting.get('line')}, count {setting.get('count')}, "
            f"kind {setting.get('kind')}, random {setting.get('random')}, best kept {setting.get('best_kept')}: "
            f"`{setting.get('excerpt')}`")
    for note in report.get("notes", []):
        if isinstance(note, dict):
            where = (f"tool log line {note.get('line')}" if note.get("cell_index") == -1
                     else f"cell {note.get('cell_index')} line {note.get('line')}")
            lines.append(f"- note {note.get('code')} ({where}): {note.get('text')}")
    violations = [f for f in report.get("flags", []) if isinstance(f, dict) and f.get("severity") == VIOLATION]
    if violations:
        lines.append(f"- Static VIOLATION flags (already final, not for you to decide): "
                     f"{sorted({str(f.get('rule')) for f in violations})}.")
    return "\n".join(lines)


def soft_flags(report: Mapping[str, object]) -> list[dict[str, object]]:
    """The SOFT flags of a report as dictionaries."""
    return [f for f in report.get("flags", []) if isinstance(f, dict) and f.get("severity") == SOFT]


def _flag_listing(flags: Sequence[Mapping[str, object]]) -> str:
    listing = []
    for flag in flags:
        rule = str(flag.get("rule"))
        listing.append({
            "flag_id": flag.get("flag_id"),
            "rule": rule,
            "rule_text": RULES[rule][1] if rule in RULES else None,
            "cell_index": flag.get("cell_index"),
            "line": flag.get("line"),
            "excerpt": flag.get("excerpt"),
            "reason": flag.get("reason"),
        })
    return json.dumps(listing, indent=1, ensure_ascii=False)


def build_prompt(report: Mapping[str, object], cells: Sequence[CellRecord],
                 byte_limit: int = PROMPT_BYTE_LIMIT) -> tuple[str, dict[str, list[int]]]:
    """Fill the template. Returns the prompt and which cells were shown whole, clipped or omitted.

    Cells are added in priority order (flagged cells, cells named in flag reasons, cells that read labels, the rest)
    while the prompt stays within ``byte_limit`` bytes. A flagged cell too large to show whole is shown as windows of
    ``CONTEXT_LINES`` lines around its flagged lines.
    """
    template = template_text()
    flags = soft_flags(report)
    base = (template.replace("@@SUMMARY@@", _summary(report)).replace("@@FLAGS@@", _flag_listing(flags)))
    budget = byte_limit - len(base.encode("utf-8")) - 2000
    if budget <= 0:
        raise JudgeError("the summary and the flag list alone exceed the prompt size limit")
    executed = {cell.index: cell for cell in cells if cell.executed}
    flagged_lines: dict[int, set[int]] = {}
    for flag in flags:
        flagged_lines.setdefault(int(flag["cell_index"]), set()).add(int(flag["line"]))  # type: ignore[call-overload]
    referenced: list[int] = []
    for flag in report.get("flags", []):
        if isinstance(flag, dict):
            referenced += [int(m) for m in re.findall(r"cell (\d+)", str(flag.get("reason", "")))]
    label_cells = [int(i) for i in report.get("label_access_cells", [])]  # type: ignore[union-attr]
    order: list[int] = []
    for index in [*sorted(flagged_lines), *referenced, *label_cells, *sorted(executed)]:
        if index in executed and index not in order:
            order.append(index)
    shown: dict[int, str] = {}
    status: dict[str, list[int]] = {"whole": [], "clipped": [], "omitted": []}
    used = 0
    for index in order:
        cell = executed[index]
        block = f"### cell {index} (status {cell.status})\n```python\n{_numbered(cell.code)}\n```\n"
        size = len(block.encode("utf-8"))
        if used + size <= budget:
            shown[index] = block
            used += size
            status["whole"].append(index)
            continue
        if index in flagged_lines:
            keep = {n for line in flagged_lines[index] for n in range(line - CONTEXT_LINES, line + CONTEXT_LINES + 1)}
            block = (f"### cell {index} (status {cell.status}, clipped to the flagged lines)\n```python\n"
                     f"{_numbered(cell.code, keep)}\n```\n")
            size = len(block.encode("utf-8"))
            if used + size <= budget:
                shown[index] = block
                used += size
                status["clipped"].append(index)
                continue
        status["omitted"].append(index)
    listing = "\n".join(shown[index] for index in sorted(shown))
    if status["omitted"]:
        listing += f"\nCells omitted for length: {sorted(status['omitted'])}.\n"
    prompt = base.replace("@@CELLS@@", listing)
    if len(prompt.encode("utf-8")) > byte_limit:
        raise JudgeError("the prompt exceeds the size limit after filling")
    for key in status:
        status[key].sort()
    return prompt, status


def _json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        value = json.loads(stripped)
    except ValueError as exc:
        raise JudgeError(f"the judge reply is not one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise JudgeError("the judge reply is not a JSON object")
    return value


def parse_decisions(text: str, flag_ids: Sequence[str]) -> list[dict[str, str]]:
    """Validate the judge reply: exactly one well-formed decision per SOFT flag id."""
    value = _json_object(text)
    decisions = value.get("decisions")
    if not isinstance(decisions, list):
        raise JudgeError("the judge reply has no decisions list")
    expected = list(flag_ids)
    seen: dict[str, dict[str, str]] = {}
    for item in decisions:
        if not isinstance(item, dict):
            raise JudgeError("a decision is not a JSON object")
        flag_id = item.get("flag_id")
        if flag_id not in expected:
            raise JudgeError(f"decision for unknown flag id {flag_id!r}")
        if flag_id in seen:
            raise JudgeError(f"two decisions for flag id {flag_id}")
        decision = item.get("decision")
        confidence = item.get("confidence")
        if decision not in DECISIONS:
            raise JudgeError(f"flag {flag_id}: decision {decision!r} is not one of {DECISIONS}")
        if confidence not in CONFIDENCES:
            raise JudgeError(f"flag {flag_id}: confidence {confidence!r} is not one of {CONFIDENCES}")
        clause, reason = item.get("clause", ""), item.get("reason", "")
        if not isinstance(clause, str) or not isinstance(reason, str):
            raise JudgeError(f"flag {flag_id}: clause and reason must be strings")
        seen[str(flag_id)] = {"flag_id": str(flag_id), "decision": str(decision), "confidence": str(confidence),
                              "clause": clause, "reason": reason}
    missing = [flag_id for flag_id in expected if flag_id not in seen]
    if missing:
        raise JudgeError(f"no decision for flag ids {missing}")
    return [seen[flag_id] for flag_id in expected]


def judge_command(claude_bin: str, prompt: str, model: str | None = None, effort: str | None = None) -> list[str]:
    """The judge's command line: no tools, no settings, no MCP servers, no session persistence, JSON output."""
    command = [claude_bin, "-p", prompt, "--tools", "", "--setting-sources", "", "--strict-mcp-config",
               "--disable-slash-commands", "--no-session-persistence", "--output-format", "json"]
    if model:
        command += ["--model", model]
    if effort:
        command += ["--effort", effort]
    return command


def _run(command: list[str], cwd: Path, timeout_s: float) -> tuple[int | None, str, str, bool]:
    with open(os.devnull, "rb") as devnull:
        try:
            process = subprocess.Popen(command, cwd=cwd, stdin=devnull, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True, text=True, encoding="utf-8",
                                       errors="replace")
        except OSError as exc:
            raise JudgeError(f"cannot start {command[0]}: {exc}") from exc
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
            return process.returncode, stdout, stderr, False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return None, stdout, stderr, True


def _claude_version(claude_bin: str, cwd: Path) -> str:
    with open(os.devnull, "rb") as devnull:
        try:
            completed = subprocess.run([claude_bin, "--version"], cwd=cwd, stdin=devnull, capture_output=True,
                                       text=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise JudgeError(f"{claude_bin} --version failed: {exc}") from exc
    if completed.returncode != 0:
        raise JudgeError(f"{claude_bin} --version failed with exit status {completed.returncode}: "
                         f"{completed.stderr.strip()[:500]}")
    return completed.stdout.strip()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def run_judge(session_dir: str | Path, *, claude_bin: str = "claude", model: str | None = None,
              effort: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S, force: bool = False,
              report_path: str | Path | None = None) -> dict[str, object]:
    """Resolve the SOFT flags of ``<session_dir>/logs/audit.json`` and write ``logs/audit_judge.json``.

    ``report_path`` names a static report written elsewhere; the judge record and prompt are then written next to it.
    The static report must be current: its auditor hash must match this package and its cells hash must match
    ``logs/cells.jsonl``. The judge runs only for a SOFT_FLAGS verdict unless ``force`` is set (a static VIOLATION
    stays a VIOLATION whatever the judge says). Raises ``JudgeError`` when the report is missing or stale (nothing is
    written) and ``JudgeFailed`` when the prompt cannot be built, the CLI fails or the reply is malformed (the record
    of the failed attempt is written first).
    """
    logs = Path(session_dir) / "logs"
    report_path = Path(report_path) if report_path is not None else logs / AUDIT_FILENAME
    cells_path = logs / "cells.jsonl"
    if not report_path.is_file():
        raise JudgeError(f"{report_path} does not exist; run scripts/audit_session.py first")
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("auditor_sha256") != auditor_sha256():
        raise JudgeError(f"{report_path} was written by a different audit implementation; rerun audit_session.py")
    cells_sha = hashlib.sha256(cells_path.read_bytes()).hexdigest() if cells_path.exists() else None
    if report.get("cells_sha256") != cells_sha:
        raise JudgeError(f"{report_path} does not match the current {cells_path}; rerun audit_session.py")
    flags = soft_flags(report)
    record: dict[str, object] = {
        "schema": JUDGE_SCHEMA,
        "created_at": _now(),
        "static_verdict": report.get("verdict"),
        "audit_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "template_sha256": template_sha256(),
        "auditor_sha256": report.get("auditor_sha256"),
        "soft_flag_ids": [f.get("flag_id") for f in flags],
        "model_requested": model,
        "effort_requested": effort,
    }
    target = report_path.with_name(JUDGE_FILENAME)
    if not flags or (report.get("verdict") != "SOFT_FLAGS" and not force):
        record.update(status="not_needed", decisions=[], judge_verdict=None, final_verdict=report.get("verdict"))
        write_json_atomic(target, record)
        return record
    cells = read_cells(cells_path)
    try:
        prompt, shown = build_prompt(report, cells)
        report_path.with_name(PROMPT_FILENAME).write_text(prompt, encoding="utf-8")
        command = judge_command(claude_bin, prompt, model, effort)
        record.update(prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(), prompt_file=PROMPT_FILENAME,
                      cells_shown=shown, command=[part if part != prompt else "<prompt>" for part in command])
        with tempfile.TemporaryDirectory(prefix="weightcode_judge_") as workdir:
            record["claude_version"] = _claude_version(claude_bin, Path(workdir))
            began = time.monotonic()
            returncode, stdout, stderr, timed_out = _run(command, Path(workdir), timeout_s)
        record.update(duration_s=round(time.monotonic() - began, 3), exit_code=returncode, timed_out=timed_out,
                      stderr=stderr[-4000:])
        if timed_out:
            raise JudgeError(f"the judge did not finish within {timeout_s:g} s")
        if returncode != 0:
            raise JudgeError(f"the judge exited with status {returncode}")
        try:
            envelope = json.loads(stdout)
        except ValueError as exc:
            raise JudgeError(f"the CLI output is not JSON: {exc}") from exc
        if not isinstance(envelope, dict):
            raise JudgeError("the CLI output is not a JSON object")
        record["cli_result"] = {key: envelope.get(key) for key in ("type", "subtype", "is_error", "num_turns",
                                                                    "duration_ms", "total_cost_usd", "usage",
                                                                    "session_id")}
        usage = envelope.get("modelUsage")
        record["models_used"] = sorted(usage) if isinstance(usage, dict) else []
        record["reply"] = envelope.get("result")
        if envelope.get("is_error") or envelope.get("subtype") != "success":
            raise JudgeError(f"the CLI reported an error (subtype {envelope.get('subtype')!r})")
        if not isinstance(envelope.get("result"), str):
            raise JudgeError("the CLI output has no result text")
        decisions = parse_decisions(str(envelope["result"]), [str(f.get("flag_id")) for f in flags])
    except JudgeError as exc:
        record.update(status="failed", error=str(exc), decisions=[], judge_verdict=None, final_verdict=None)
        write_json_atomic(target, record)
        raise JudgeFailed(str(exc), target) from exc
    judge_verdict = VIOLATION if any(d["decision"] == VIOLATION for d in decisions) else "ALLOWED"
    final = VIOLATION if (report.get("verdict") == VIOLATION or judge_verdict == VIOLATION) else "ALLOWED"
    record.update(status="ok", decisions=decisions, judge_verdict=judge_verdict, final_verdict=final,
                  low_confidence=[d["flag_id"] for d in decisions if d["confidence"] == "low"])
    write_json_atomic(target, record)
    return record
