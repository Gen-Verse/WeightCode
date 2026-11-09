"""Resolve the SOFT flags of a session's static audit with the frozen LLM judge (``weightcode/audit/judge_prompt.md``).

Reads ``<session_dir>/logs/audit.json`` (written by ``scripts/audit_session.py``, or ``--report``) and the cell log,
and writes ``logs/audit_judge.json`` next to the report. The judge is ``claude -p`` with no tools, no settings, no MCP
servers and stdin from /dev/null; it runs only when the static verdict is SOFT_FLAGS (or with ``--force``) and never
overrides a static VIOLATION. The final verdict is CLEAN or VIOLATION when the judge was not needed, and ALLOWED or
VIOLATION after it ran.

Exit status: 0 when a judge record with status ok or not_needed was written, 1 when the judge failed or replied
badly (a record with status failed is written), 2 when the static report is missing, stale or unreadable.

Example::

    .venv/bin/python scripts/audit_judge.py "$WEIGHTCODE_RUNS/sessions/<session_id>" --model claude-opus-5-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXIT_JUDGE = 1
EXIT_INPUT = 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    from weightcode.audit.judge import DEFAULT_TIMEOUT_S

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir", type=Path, help="session directory holding logs/cells.jsonl and logs/audit.json")
    parser.add_argument("--report", type=Path, help="static report path (default <session_dir>/logs/audit.json)")
    parser.add_argument("--claude-bin", default="claude", help="claude CLI executable (default: claude on PATH)")
    parser.add_argument("--model", help="model passed to claude --model (default: the CLI default)")
    parser.add_argument("--effort", help="effort level passed to claude --effort")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds before the judge is killed")
    parser.add_argument("--force", action="store_true",
                        help="also run when the static verdict is VIOLATION (it stays VIOLATION)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from weightcode.audit.judge import JudgeError, JudgeFailed, run_judge
    from weightcode.audit.session import AuditInputError

    report = args.report if args.report is not None else args.session_dir / "logs" / "audit.json"
    if not report.is_file():
        print(f"error: {report} does not exist; run scripts/audit_session.py first", file=sys.stderr)
        return EXIT_INPUT
    try:
        record = run_judge(args.session_dir, claude_bin=args.claude_bin, model=args.model, effort=args.effort,
                           timeout_s=args.timeout, force=args.force, report_path=report)
    except JudgeFailed as exc:
        print(f"judge failed: {exc} (record in {exc.record_path})", file=sys.stderr)
        return EXIT_JUDGE
    except JudgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except (AuditInputError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    print(f"judge {record['status']}: static verdict {record['static_verdict']}, final verdict "
          f"{record['final_verdict']}")
    for decision in record.get("decisions", []):  # type: ignore[union-attr]
        print(f"  {decision['flag_id']} {decision['decision']} ({decision['confidence']}): {decision['reason']}")
    print(f"wrote {report.with_name('audit_judge.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
