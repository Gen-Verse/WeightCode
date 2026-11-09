"""Run the static training-free audit (contract section 5.5, HARNESS_SPEC section 7) on one session directory.

Reads ``<session_dir>/logs/cells.jsonl`` and the ``predict`` calls of ``<session_dir>/logs/tools.jsonl`` (absent:
no predict calls), and writes the report (schema ``weightcode.audit.v0``) to ``<session_dir>/logs/audit.json``, or to
``--out``. Prints the verdict, the candidate count and the flags. The verdict is a reported field: the exit status is 0
whenever a report was written, whatever the verdict. Exit status 2 means nothing was written because the cell log is
missing, a log is malformed, or the cell log still has a cell in flight (run
``weightcode.sandbox.host.recover_interrupted_cell`` on it first).

The harness writes its own report to ``<session_dir>/audit.json`` when a session finalises. A rerun with a changed
auditor goes beside it, ``--out <session_dir>/audit.rerun<N>.json --no-overwrite``, so the report of record is never
replaced. ``--recover-inflight-in-memory`` audits a cell left in flight by a stopped host as the interrupted cell it
would be logged as, without changing the logs.

Audit amendment A2 (2026-11-04): the printed summary gives both counts, (L) the literal count of label-scored
settings that the verdict uses and (C) the configuration count, reported only.

Example::

    .venv/bin/python scripts/audit_session.py "$WEIGHTCODE_RUNS/sessions/<session_id>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXIT_INPUT = 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir", type=Path, help="session directory holding logs/cells.jsonl")
    parser.add_argument("--out", type=Path, help="report path (default <session_dir>/logs/audit.json)")
    parser.add_argument("--label-key", action="append", dest="label_keys", metavar="KEY",
                        help="evidence key holding labels (repeatable; default y_fail, y_rand, y_new, y_old, y_prev)")
    parser.add_argument("--recover-inflight-in-memory", action="store_true",
                        help="audit a cell left in flight as the interrupted cell, without changing the logs")
    parser.add_argument("--no-overwrite", action="store_true", help="exit 2 if the report path already exists")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from weightcode.audit.session import AUDIT_FILENAME, DEFAULT_LABEL_KEYS, AuditInputError, audit_session, write_report

    cells = args.session_dir / "logs" / "cells.jsonl"
    if not cells.is_file():
        print(f"error: {cells} does not exist", file=sys.stderr)
        return EXIT_INPUT
    keys = tuple(args.label_keys) if args.label_keys else DEFAULT_LABEL_KEYS
    out = args.out if args.out is not None else args.session_dir / "logs" / AUDIT_FILENAME
    if args.no_overwrite and out.exists():
        print(f"error: {out} exists and --no-overwrite was given", file=sys.stderr)
        return EXIT_INPUT
    try:
        report = audit_session(cells, keys, recover_inflight_in_memory=args.recover_inflight_in_memory)
    except AuditInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    if args.no_overwrite and out.exists():
        print(f"error: {out} appeared while auditing and --no-overwrite was given", file=sys.stderr)
        return EXIT_INPUT
    write_report(report, out)
    bound = " or more" if report.candidate_count_is_lower_bound else ""
    config_bound = " or more" if report.config_count_is_lower_bound else ""
    print(f"verdict {report.verdict}: {len(report.violation_flags())} violation flags, {len(report.soft_flags())} "
          f"soft flags, (L) {report.candidate_count}{bound} label-scored candidate settings (limit "
          f"{report.candidate_limit}), (C) {report.config_count}{config_bound} configurations (reported only), "
          f"{report.n_analysed} of {report.n_cells} cells analysed")
    for flag in report.flags:
        print(f"  {flag.flag_id} {flag.rule} cell {flag.cell_index} line {flag.line}: {flag.excerpt}")
        print(f"      {flag.reason}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
