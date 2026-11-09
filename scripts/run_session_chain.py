"""Run editor sessions one after another under a quota guard (sequencing around scripts/run_editor_session.py).

Each SPEC is INSTANCE:ARM or INSTANCE:ARM:REPEAT (repeat 0 by default). Sessions run one at a time with the same effort
and wall limit. Before each launch after the first, the chain stops when the previous session

- was INFRA (rate-limit rejection, API error or harness crash; PREREG section 5 reruns it once, by hand),
- left no session_result.json, or not exactly one new session directory whose session.json names the spec (the
  runner failed before finalisation; matching on session.json lets several chains run in parallel), or
- ended with a rate-limit event whose five-hour or seven-day utilization is at or above ``--stop-at``.

Each outcome is appended to ``--log`` as one JSON line: the spec, the runner's exit code, the session root, status,
final S, wall time, tokens, the last rate-limit event and the stop reason, if any. The runner's output goes to
``<log stem>.<instance>_<arm>_r<repeat>.out`` next to the log. The chain never reruns a session. A runner exit code
that is not zero with a finalised, non-INFRA session (for example a failed end-of-session audit, which can be rerun
offline) does not stop the chain.

Usage::

    source scripts/env.sh
    .venv/bin/python scripts/run_session_chain.py --log RUNLOG.jsonl --effort max --wall-timeout-min 90 \\
        ib338573e:M2 i7346ae21:M2 ibb1f3fc6:M2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNNER = Path(__file__).resolve().with_name("run_editor_session.py")


def parse_spec(spec: str) -> tuple[str, str, int]:
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not all(parts):
        raise argparse.ArgumentTypeError(f"spec {spec!r} is not INSTANCE:ARM[:REPEAT]")
    return parts[0], parts[1], int(parts[2]) if len(parts) == 3 else 0


def utilization(result: dict[str, Any]) -> float | None:
    """The larger five-hour or seven-day utilization of the session's last rate-limit event, if it reported one."""
    event = (result.get("stream") or {}).get("rate_limit_last") or {}
    values = [float(window["utilization"]) for window in (event.get("five_hour"), event.get("seven_day"))
              if isinstance(window, dict) and window.get("utilization") is not None]
    return max(values) if values else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dirs(path: Path) -> set[str]:
    return {entry.name for entry in path.iterdir() if entry.is_dir()} if path.exists() else set()


def _matching(sessions: Path, names: Sequence[str], instance: str, arm: str, repeat: int) -> list[str]:
    """New session directories whose session.json names this instance, arm and repeat (chains may run in parallel)."""
    hits = []
    for name in names:
        try:
            meta = json.loads((sessions / name / "session.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (meta.get("instance_id"), meta.get("arm"), meta.get("repeat")) == (instance, arm, repeat):
            hits.append(name)
    return hits


def run_one(spec: tuple[str, str, int], args: argparse.Namespace, sessions: Path) -> dict[str, Any]:
    """Launch one session and return its log record (``stop_reason`` is None when the chain may continue)."""
    instance, arm, repeat = spec
    command = [sys.executable, str(args.runner), "--instance", instance, "--arm", arm, "--repeat", str(repeat),
               "--effort", args.effort, "--wall-timeout-min", str(args.wall_timeout_min)]
    if args.dry_run:
        command.append("--dry-run")
    before = _dirs(sessions)
    out_path = args.log.with_name(f"{args.log.stem}.{instance}_{arm}_r{repeat}.out")
    record: dict[str, Any] = {"spec": f"{instance}:{arm}:{repeat}", "started": _now(), "command": command[1:],
                              "output": str(out_path)}
    with out_path.open("w", encoding="utf-8") as out:
        record["exit_code"] = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=False).returncode
    record["ended"] = _now()
    new = _matching(sessions, sorted(_dirs(sessions) - before), instance, arm, repeat)
    if len(new) != 1:
        record["stop_reason"] = f"expected one new session directory for this spec under {sessions}, found {len(new)}: {new}"
        return record
    root = sessions / new[0]
    record["root"] = str(root)
    result_path = root / "session_result.json"
    if not result_path.exists():
        record["stop_reason"] = f"no session_result.json in {root}"
        return record
    result = json.loads(result_path.read_text(encoding="utf-8"))
    final = result.get("final") or {}
    stream = result.get("stream") or {}
    record.update(status=result.get("status"), infra=result.get("infra"), score=final.get("score"),
                  is_base_noop=final.get("is_base_noop"), wall_s=result.get("wall_s"), tokens=result.get("tokens"),
                  rate_limit_last=stream.get("rate_limit_last"), utilization=utilization(result))
    if result.get("infra"):
        record["stop_reason"] = f"INFRA: {result.get('infra_reasons')}"
    elif record["utilization"] is not None and record["utilization"] >= args.stop_at:
        record["stop_reason"] = f"utilization {record['utilization']} >= --stop-at {args.stop_at}"
    else:
        record["stop_reason"] = None
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("specs", nargs="+", type=parse_spec, metavar="INSTANCE:ARM[:REPEAT]")
    parser.add_argument("--log", type=Path, required=True, help="JSON-lines log, appended to")
    parser.add_argument("--effort", required=True)
    parser.add_argument("--wall-timeout-min", type=float, required=True)
    parser.add_argument("--stop-at", type=float, default=0.95, help="stop at this five-hour or seven-day utilization")
    parser.add_argument("--dry-run", action="store_true", help="pass --dry-run to the runner (no Claude Code)")
    parser.add_argument("--runner", type=Path, default=RUNNER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    sessions = Path(os.environ["WEIGHTCODE_RUNS"]) / ("dry_runs" if args.dry_run else "sessions")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(args.specs):
        record = run_one(spec, args, sessions)
        record["index"] = index
        with args.log.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"{record['ended']} {record['spec']} exit {record['exit_code']} status {record.get('status')} "
              f"S {record.get('score')} utilization {record.get('utilization')} stop {record['stop_reason']}", flush=True)
        if record["stop_reason"] is not None:
            remaining = [f"{i}:{a}:{r}" for i, a, r in args.specs[index + 1:]]
            if remaining:
                print(f"chain stopped; not launched: {remaining}", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
