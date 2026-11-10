"""One run of an update program in the editor sandbox (LEARN arm, docs/HARNESS_SPEC.md sections 3 and 5).

``run_program`` starts ``program_driver.py`` under bubblewrap with the kernel's isolation (``isolation_argv``: every
namespace unshared, so no network; no nested user namespace; the same read-only system mounts, environment and capped
tmpfs mounts) and the kernel's memory limit, and with two work mounts only: ``/work/in``, read-only, holding exactly the
files the caller staged there, and ``/work/out``, writable and empty. Nothing persists between runs. The driver imports
the program, calls ``update(student, experience)`` once and writes ``deltas.npz`` and ``result.json`` to /work/out. A
run that takes longer than ``wall_s`` (``PROGRAM_WALL_S``, the one place that sets the 300 s limit) is killed with its
whole sandbox. stdout and stderr go to host files outside the mounts, and ``RLIMIT_FSIZE`` caps every file the run
writes, so a run cannot fill the disk. Inside the run the program sees the driver's virtual clock ``PROGRAM_CLOCK``
(``virtual-v1``, see ``program_driver``), so a time budget of its own cannot make its output depend on machine load;
the wall limit is real time.

Everything the run writes is untrusted: the program runs in the driver's process and can change any of its behaviour.
Callers validate ``deltas.npz`` themselves and treat ``result.json`` as text (``weightcode.harness.program``).

``UnisolatedRunner`` runs the same driver without bubblewrap (``unisolated_launcher.py`` redirects its two directories),
with the same wall limit, memory and file-size caps, clock and thread environment. It exists to replay a published
program whose sha256 the caller has checked against a record (``weightcode.examples``) on machines without a
suitable bubblewrap. The program can then read the host's files and use the network, so it must never run a program
a language model has just written.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import resource
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weightcode.sandbox.host import SEAL_ARGV, SandboxConfig, isolated_environment, isolation_argv

PROGRAM_WALL_S = 300.0
PROGRAM_CLOCK = "virtual-v1"  # the driver's clock (program_driver.CLOCK_VIRTUAL_V1), recorded with every run
DRIVER_SCRIPT = Path(__file__).resolve().with_name("program_driver.py")
LAUNCHER_SCRIPT = Path(__file__).resolve().with_name("unisolated_launcher.py")
ISOLATION_NONE = "none"
DRIVER_MOUNT = "/opt/weightcode/program_driver.py"
IN_MOUNT = "/work/in"
OUT_MOUNT = "/work/out"
DRIVER_HOME = "/tmp"
FILE_SIZE_LIMIT_BYTES = 256 * 2**20  # RLIMIT_FSIZE of the run: deltas.npz, result.json, stdout and stderr each
RESULT_MAX_BYTES = 1 << 20
REAP_TIMEOUT_S = 10.0
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"
STATUS_DIED = "died"


class ProgramSandboxError(RuntimeError):
    """The program sandbox cannot be configured or started (a harness fault, not a problem of the program)."""


@dataclass(frozen=True)
class ProgramSandbox:
    """Interpreter, bubblewrap and resource limits of program runs: those of the session's kernel sandbox."""

    python: Path
    memory_bytes: int
    threads: int
    tmpfs_bytes: int = 2**30
    bwrap: Path = Path("/usr/bin/bwrap")
    max_output_chars: int = 30_000

    @classmethod
    def from_config(cls, config: SandboxConfig) -> ProgramSandbox:
        """The program sandbox of a session: its kernel's interpreter, bwrap, memory, threads and tmpfs caps."""
        return cls(python=config.python, memory_bytes=config.memory_bytes, threads=config.threads,
                   tmpfs_bytes=config.tmpfs_bytes, bwrap=config.bwrap, max_output_chars=config.max_output_chars)

    def as_dict(self) -> dict[str, Any]:
        return {"python": str(self.python), "memory_bytes": self.memory_bytes, "threads": self.threads,
                "tmpfs_bytes": self.tmpfs_bytes, "bwrap": str(self.bwrap), "max_output_chars": self.max_output_chars,
                "driver_sha256": driver_sha256(), "file_size_limit_bytes": FILE_SIZE_LIMIT_BYTES}


@dataclass(frozen=True)
class UnisolatedRunner:
    """Runs the driver without bubblewrap (module docstring): only for a published program checked by its sha256.

    ``python`` must have numpy importable; it is normally the engine's own interpreter (``sys.executable``).
    """

    python: Path
    memory_bytes: int
    threads: int
    max_output_chars: int = 30_000

    def as_dict(self) -> dict[str, Any]:
        return {"isolation": ISOLATION_NONE, "python": str(self.python), "memory_bytes": self.memory_bytes,
                "threads": self.threads, "max_output_chars": self.max_output_chars, "driver_sha256": driver_sha256(),
                "launcher_sha256": hashlib.sha256(LAUNCHER_SCRIPT.read_bytes()).hexdigest(),
                "file_size_limit_bytes": FILE_SIZE_LIMIT_BYTES}


ProgramRunner = ProgramSandbox | UnisolatedRunner
"""Where a program runs: in the bubblewrap sandbox, or (published programs only) without it."""


@dataclass(frozen=True)
class ProgramRun:
    """What one run produced. ``result`` is the driver's result.json (None if absent or unreadable)."""

    status: str
    duration_s: float
    exit_code: int | None
    stdout: str
    stderr: str
    result: dict[str, Any] | None
    out_dir: Path

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready record (the out dir by name only)."""
        return {"status": self.status, "duration_s": round(self.duration_s, 3), "exit_code": self.exit_code,
                "stdout": self.stdout, "stderr": self.stderr, "result": self.result}


def driver_sha256() -> str:
    """sha256 of the driver script bound into the sandbox."""
    return hashlib.sha256(DRIVER_SCRIPT.read_bytes()).hexdigest()


def build_program_argv(sandbox: ProgramSandbox, in_dir: Path, out_dir: Path, program_name: str) -> list[str]:
    """The bwrap command line of one run: the kernel's isolation prefix, /work/in read-only, /work/out, the driver."""
    interpreter = Path(os.path.abspath(sandbox.python))
    argv = isolation_argv(sandbox.bwrap, sandbox.python, isolated_environment(sandbox.threads, DRIVER_HOME),
                          sandbox.tmpfs_bytes)
    argv += [
        "--ro-bind", str(in_dir.resolve()), IN_MOUNT,
        "--bind", str(out_dir.resolve()), OUT_MOUNT,
        "--ro-bind", str(DRIVER_SCRIPT), DRIVER_MOUNT,
        *SEAL_ARGV,
        str(interpreter), "-I", "-u", DRIVER_MOUNT, "--program", program_name, "--clock", PROGRAM_CLOCK,
    ]  # fmt: skip
    return argv


def build_unisolated_argv(runner: UnisolatedRunner, in_dir: Path, out_dir: Path, program_name: str) -> list[str]:
    """The command line of one run without bubblewrap: the launcher, the driver and the two work directories."""
    return [str(runner.python), "-I", "-u", str(LAUNCHER_SCRIPT), "--driver", str(DRIVER_SCRIPT),
            "--in-dir", str(in_dir.resolve()), "--out-dir", str(out_dir.resolve()), "--",
            "--program", program_name, "--clock", PROGRAM_CLOCK]


def _limit_run(memory_bytes: int) -> None:
    """preexec_fn: the kernel's address-space cap, no core dumps, and the file-size cap of the run."""
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_SIZE_LIMIT_BYTES, FILE_SIZE_LIMIT_BYTES))


def _head_tail(path: Path, limit: int) -> str:
    """At most ``limit`` characters of a log file: its head and tail when it is longer."""
    size = path.stat().st_size
    with open(path, "rb") as handle:
        if size <= 4 * limit:
            text = handle.read().decode("utf-8", errors="replace")
        else:
            head = handle.read(2 * limit).decode("utf-8", errors="replace")
            handle.seek(size - 2 * limit)
            text = head + handle.read().decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    half = max(0, (limit - 40) // 2)
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters omitted ...]\n{text[-half:] if half else ''}"


def _read_result(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "result.json"
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > RESULT_MAX_BYTES:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def run_program(sandbox: ProgramRunner, run_dir: Path, program_name: str, wall_s: float = PROGRAM_WALL_S
                ) -> ProgramRun:
    """Run the program ``run_dir/in/<program_name>`` once (see the module docstring), in the bubblewrap sandbox or,
    for an ``UnisolatedRunner``, without it.

    ``run_dir/in`` must hold the staged inputs; ``run_dir/out`` is created empty; stdout and stderr are written to
    ``run_dir/stdout.txt`` and ``run_dir/stderr.txt``. Status: ``ok`` (the driver reported success), ``error`` (the
    driver reported the program's exception), ``timeout`` (killed at ``wall_s``) or ``died`` (the run ended without a
    readable result, for example killed by a resource limit).
    """
    if not wall_s > 0:
        raise ValueError("wall_s must be positive")
    in_dir, out_dir = run_dir / "in", run_dir / "out"
    if not (in_dir / program_name).is_file():
        raise ProgramSandboxError(f"the staged program {program_name!r} is missing from {in_dir}")
    if isinstance(sandbox, UnisolatedRunner):
        out_dir.mkdir(mode=0o700)
        argv = build_unisolated_argv(sandbox, in_dir, out_dir, program_name)
        env, cwd = isolated_environment(sandbox.threads, str(run_dir.resolve())), str(out_dir.resolve())
    else:
        if not os.access(sandbox.bwrap, os.X_OK):
            raise ProgramSandboxError(f"bubblewrap is missing or not executable at {sandbox.bwrap}")
        out_dir.mkdir(mode=0o700)
        argv = build_program_argv(sandbox, in_dir, out_dir, program_name)
        env, cwd = {}, "/"
    stdout_path, stderr_path = run_dir / "stdout.txt", run_dir / "stderr.txt"
    with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
        began = time.monotonic()
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, close_fds=True,
                                    start_new_session=True, env=env, cwd=cwd,
                                    preexec_fn=functools.partial(_limit_run, sandbox.memory_bytes))
        except (OSError, subprocess.SubprocessError) as exc:
            what = "the program" if isinstance(sandbox, UnisolatedRunner) else "bubblewrap"
            raise ProgramSandboxError(f"could not start {what}: {exc}") from exc
        timed_out = False
        try:
            code: int | None = proc.wait(timeout=wall_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                code = proc.wait(timeout=REAP_TIMEOUT_S)
            except subprocess.TimeoutExpired as exc:
                raise ProgramSandboxError(f"the killed program sandbox was not reaped within {REAP_TIMEOUT_S:g} s"
                                          ) from exc
        duration = time.monotonic() - began
    result = None if timed_out else _read_result(out_dir)
    if timed_out:
        status = STATUS_TIMEOUT
    elif result is not None and result.get("status") == STATUS_OK and code == 0:
        status = STATUS_OK
    elif result is not None and result.get("status") == STATUS_ERROR:
        status = STATUS_ERROR
    else:
        status = STATUS_DIED
    limit = sandbox.max_output_chars // 2
    return ProgramRun(status=status, duration_s=duration, exit_code=code, stdout=_head_tail(stdout_path, limit),
                      stderr=_head_tail(stderr_path, limit), result=result, out_dir=out_dir)


__all__ = ["DRIVER_MOUNT", "DRIVER_SCRIPT", "FILE_SIZE_LIMIT_BYTES", "IN_MOUNT", "ISOLATION_NONE", "LAUNCHER_SCRIPT",
           "OUT_MOUNT", "PROGRAM_CLOCK", "PROGRAM_WALL_S",
           "STATUS_DIED", "STATUS_ERROR", "STATUS_OK", "STATUS_TIMEOUT", "ProgramRun", "ProgramRunner", "ProgramSandbox",
           "ProgramSandboxError", "UnisolatedRunner", "build_program_argv", "build_unisolated_argv", "driver_sha256",
           "run_program"]
