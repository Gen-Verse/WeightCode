"""PREREG T16 fairness check: what the round-2 editor gets in arm N and arm L against the round-2 session of record.

Compares session directories (a real session or a dry run) on everything the editor sees: the rendered system and
task prompts, every trial file by sha256 (as on disk), and the session.json fields that set the round (student,
kept rule, budget, limits, MCP timeouts, editor model and effort, experience and clean-check records, library entries,
earlier-round checks). Arm N must equal the record; arm L must equal arm N except edit_log.json and the one PREREG
sentence. Writes a JSON report and exits 1 if a requirement fails. Reads no API key and nothing under sealed/::

    .venv/bin/python scripts/t16_fairness_check.py --record $WEIGHTCODE_RUNS/sessions/sa651245fee587200 \\
        --off <arm N round-2 session or dry run> [--on <arm L round-2 session or dry run>] --out report.json

T16 follow-up M (the protected-entry limit): ``--protect <arm M round-2 session or dry run>`` (with ``--on``) also
checks that arm M equals arm L in every trial file (edit_log.json included), the task prompt and every round-setting
field, that its system prompt is arm L's plus the one limit sentence right after the edit-log sentence, and that its
session.json records the limit on arm L's edit log (enabled, the trial edit_log.json's sha256, the round's student,
every logged entry protected); arms N and L must state no limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weightcode.harness.loop import (
    EDIT_LOG_FILE,
    EDIT_LOG_SENTENCE,
    PROTECT_SENTENCE,
    LoopError,
    edit_log_section,
    protect_section,
)

SESSION_FIELDS = ("real_model_sha256", "kept_rule", "budget", "limits", "tools", "allowed_tools", "experience",
                  "clean_check", "instance_id", "n_vcheck", "vcheck_sha256", "forward_sha256", "edit_limits_enforced")
EDITOR_FIELDS = ("model", "effort", "system_prompt_mode")
MCP_FIELDS = ("tool_timeout_ms", "connect_timeout_ms")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def view(root: Path) -> dict[str, Any]:
    """What the editor of the session at ``root`` got, by hash, and the round-setting fields of its session.json."""
    session = json.loads((root / "session.json").read_text(encoding="utf-8"))
    loop = session.get("loop") or {}
    library = session.get("library") or {}
    return {
        "session_id": root.name,
        "dry_run": bool(session.get("dry_run")),
        "system_sha256": _sha((root / "prompts" / "system.md").read_bytes()),
        "task_sha256": _sha((root / "prompts" / "task.md").read_bytes()),
        "trial": {path.name: _sha(path.read_bytes()) for path in sorted((root / "trial").iterdir())},
        # the clean check's private copy lives in each session's own directory: its path differs, its hashes may not
        "session": {key: ({k: v for k, v in session[key].items() if k != "path"} if key == "clean_check"
                          and isinstance(session.get(key), dict) else session.get(key)) for key in SESSION_FIELDS},
        "editor": {key: (session.get("editor") or {}).get(key) for key in EDITOR_FIELDS},
        "mcp": {key: (session.get("mcp") or {}).get(key) for key in MCP_FIELDS},
        "library": {"entries": library.get("entries"), "withheld": library.get("withheld")},
        "loop": {"round": loop.get("round"), "student_sha256": loop.get("student_sha256"),
                 "base_params_sha256": loop.get("base_params_sha256"), "prev_instances": loop.get("prev_instances"),
                 "prev_checks": [{key: item.get(key) for key in ("round", "instance_id", "sha256", "n")}
                                 for item in loop.get("prev_checks") or []]},
        "edit_log": loop.get("edit_log"),
        "protect_logged": loop.get("protect_logged"),
        "claude_version": (session.get("claude") or {}).get("version"),
    }


def _without_source_path(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """An edit-log record without its source's path (each stream writes its log in its own work directory)."""
    if not isinstance(record, dict):
        return record
    return {**record, "source": {k: v for k, v in (record.get("source") or {}).items() if k != "path"}}


def differences(a: dict[str, Any], b: dict[str, Any], skip: Sequence[str] = ()) -> dict[str, Any]:
    """Every top-level part of two views that differs (a, b), except ``skip``."""
    out = {}
    for key in a:
        if key in ("session_id", "dry_run", "claude_version", "edit_log", "protect_logged", *skip):
            continue
        if a[key] != b.get(key):
            if isinstance(a[key], dict):
                out[key] = {k: [a[key].get(k), b[key].get(k)] for k in sorted(set(a[key]) | set(b[key]))
                            if a[key].get(k) != b[key].get(k)}
            else:
                out[key] = [a[key], b.get(key)]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, default=None)
    parser.add_argument("--protect", type=Path, default=None, help="arm M (follow-up M), compared with --on")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.protect is not None and args.on is None:
        parser.error("--protect is compared with arm L: give --on")
    record, off = view(args.record), view(args.off)
    report: dict[str, Any] = {"record": record, "off": off, "off_vs_record": differences(record, off)}
    report["off_equals_record"] = not report["off_vs_record"] and off["edit_log"] in (None, {"enabled": False})
    ok = report["off_equals_record"]
    if args.on is not None:
        on = view(args.on)
        system_off = (args.off / "prompts" / "system.md").read_text(encoding="utf-8")
        system_on = (args.on / "prompts" / "system.md").read_text(encoding="utf-8")
        trial_extra = sorted(set(on["trial"]) - set(off["trial"]))
        checks = {
            "system_is_off_plus_sentence": system_on == edit_log_section(system_off),
            "sentence_once": system_on.count(EDIT_LOG_SENTENCE) == 1,
            "task_equal": on["task_sha256"] == off["task_sha256"],
            "trial_extra_is_edit_log": trial_extra == [EDIT_LOG_FILE],
            "trial_others_equal": {k: v for k, v in on["trial"].items() if k != EDIT_LOG_FILE} == off["trial"],
            "edit_log_recorded": bool((on["edit_log"] or {}).get("enabled"))
                                 and on["edit_log"].get("sha256") == on["trial"].get(EDIT_LOG_FILE),
            "rest_equal": not differences(off, on, skip=("system_sha256", "trial")),
        }
        report.update(on=on, on_vs_off=differences(off, on), on_checks=checks,
                      on_n_entries=(on["edit_log"] or {}).get("n_entries"))
        ok = ok and all(checks.values())
        if args.protect is not None:  # follow-up M: arm L plus the limit and its one sentence
            m = view(args.protect)
            system_m = (args.protect / "prompts" / "system.md").read_text(encoding="utf-8")
            limit = m["protect_logged"] or {}
            # the E11 two-round stream: one logged round, so every logged entry is protected exactly once
            n_logged = sum(((on["edit_log"] or {}).get("n_entries") or {}).values())
            try:
                expected_m: str | None = protect_section(system_on)
            except LoopError:  # arm L's prompt lacks the edit-log sentence or already states the limit
                expected_m = None
            m_checks = {
                "system_is_on_plus_sentence": system_m == expected_m,
                "sentence_once": system_m.count(PROTECT_SENTENCE) == 1,
                "sentence_after_edit_log_sentence": EDIT_LOG_SENTENCE + " " + PROTECT_SENTENCE in system_m,
                "task_equal": m["task_sha256"] == on["task_sha256"],
                "trial_equal": m["trial"] == on["trial"],
                "edit_log_equal": _without_source_path(m["edit_log"]) == _without_source_path(on["edit_log"]),
                "limit_recorded": limit.get("enabled") is True and limit.get("sentence") == PROTECT_SENTENCE
                                  and limit.get("edit_log_sha256") == m["trial"].get(EDIT_LOG_FILE),
                "limit_on_the_rounds_student": limit.get("student_sha256") == m["session"]["real_model_sha256"]
                                               == on["session"]["real_model_sha256"],
                "every_logged_entry_protected": n_logged > 0 and limit.get("n_protected") == n_logged
                                                == sum((limit.get("per_tensor") or {}).values()),
                "off_and_on_state_no_limit": off["protect_logged"] is None and on["protect_logged"] is None,
                "rest_equal": not differences(on, m, skip=("system_sha256",)),
            }
            report.update(protect=m, protect_vs_on=differences(on, m), protect_checks=m_checks,
                          protect_n_protected=limit.get("n_protected"))
            ok = ok and all(m_checks.values())
    report["passed"] = ok
    text = json.dumps(report, indent=1)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("off_vs_record", "off_equals_record", "on_checks", "on_vs_off",
                                                   "on_n_entries", "protect_checks", "protect_vs_on",
                                                   "protect_n_protected", "passed") if key in report}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
