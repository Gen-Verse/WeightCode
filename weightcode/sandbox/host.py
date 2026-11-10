"""Host side of the sandboxed Python kernel (docs/HARNESS_SPEC.md section 3).

``SandboxKernel`` starts ``kernel.py`` inside bubblewrap and runs editor cells in it one at a time.

Channels. Requests go to the kernel's stdin. Responses come back on a dedicated pipe passed into the sandbox
with ``pass_fds`` (verified to survive bwrap 0.11.1 on 2026-11-03). The kernel's fd 1 and fd 2 are two more
pipes that this process reads as the cell's stdout and stderr, so ``print``, ``os.write(1, ...)``, C-level
output and child processes are all captured, and none of them can corrupt a response. A response that fails
validation (bad JSON, wrong request id, extra bytes) is treated as a protocol error: the kernel is killed and
restarted, never trusted.

Isolation beyond the spec's argv (all verified on this host): ``--unshare-user --disable-userns`` blocks
nested user namespaces, which would otherwise let user code mount new, unbounded tmpfs; ``--remount-ro /``
and ``--remount-ro /dev`` make the sandbox's in-memory root and /dev read-only, because bwrap leaves them
writable and sized at half the host RAM; /tmp and /dev/shm are separate tmpfs mounts capped at
``tmpfs_bytes``. Nothing else from the host filesystem is bound.

Interpreter. ``python`` is ``<venv>/bin/python`` of a NumPy-only venv. The venv is bound read-only at its own path. A
base interpreter under /usr needs nothing more. One elsewhere, such as a uv-managed CPython under
``~/.local/share/uv/python``, is bound file by file (``interpreter_mounts``): its executable, its standard library
directory and any ``libpython`` files, all read-only, and the symbolic links on the way to it are recreated. The rest of
that installation and of the directories around it stays invisible.

Bubblewrap. ``--disable-userns`` needs bubblewrap 0.8.0 or newer (``MIN_BWRAP_VERSION``); ``check_bwrap`` refuses an
older one by name before anything starts. ``$WEIGHTCODE_BWRAP`` overrides the default /usr/bin/bwrap.

Log. Every cell, executed or refused, is appended to ``config.log_path`` as one JSON line written by this
process, which lies outside every directory bound into the sandbox. Before a cell is sent to the kernel,
its code is written to ``<log_path>.inflight``; if this process dies mid-cell, the next ``SandboxKernel`` on
the same log (or ``recover_interrupted_cell``) logs that cell as ``interrupted``. Cell counters resume from
an existing log, so restarting the host process does not reset the cell budget.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import json
import os
import re
import resource
import secrets
import selectors
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

KERNEL_SCRIPT = Path(__file__).resolve().with_name("kernel.py")
KERNEL_MOUNT = "/opt/weightcode/kernel.py"
WORK_MOUNT = "/work"
TRIAL_MOUNT = "/work/trial"
PATCHES_MOUNT = "/work/patches"
SCRATCH_MOUNT = "/work/scratch"
PROTOCOL = "weightcode.kernel.v0"
CELL_LOG_SCHEMA = "weightcode.sandbox.cell.v0"
INFLIGHT_SCHEMA = "weightcode.sandbox.inflight.v0"
INFLIGHT_SUFFIX = ".inflight"
DEFAULT_BWRAP = Path(os.environ.get("WEIGHTCODE_BWRAP") or "/usr/bin/bwrap")
"""The bubblewrap binary when none is given: ``$WEIGHTCODE_BWRAP`` if set, else the distribution's /usr/bin/bwrap."""
MIN_BWRAP_VERSION = (0, 8, 0)
"""The oldest bubblewrap with every flag of ``isolation_argv`` (``--disable-userns`` arrived in 0.8.0)."""
SYSTEM_TREE = "/usr"
USR_LINKS = {"/bin": "usr/bin", "/lib": "usr/lib", "/lib64": "usr/lib64"}  # made by isolation_argv
_MAX_SYMLINKS = 40

STATE_LOST_NOTE = (
    "The Python kernel was restarted (state lost): all variables, imports and definitions from earlier cells "
    "are gone. Files written to /work/patches and /work/scratch are kept."
)

_READ_CHUNK = 1 << 16  # bytes per read from an output pipe and per write to the request pipe
_RESPONSE_READ_CHUNK = 1 << 20  # bytes per read from the response pipe
_DRAIN_S = 0.25  # after a response or a kill: time allowed to collect output already in the pipes
_EXIT_GRACE_S = 2.0  # a kernel that closed its response pipe gets this long to exit before it is killed
_REAP_TIMEOUT_S = 10.0  # a killed sandbox must be reaped within this time, or SandboxError is raised
_JSON_ESCAPE_FACTOR = 12  # worst-case bytes per character in ensure_ascii JSON (a surrogate pair)


class SandboxError(RuntimeError):
    """The sandbox cannot be configured, started or restarted, or its log is inconsistent."""


@dataclass(frozen=True)
class SandboxConfig:
    """One sandbox session. Directory fields are host paths, seen inside as /work/{trial,patches,scratch}.

    The fields after ``max_output_chars`` are additions to the spec; their defaults keep spec-style
    construction valid. ``max_log_bytes`` bounds what one output stream of one cell keeps in the log (head and
    tail; the true byte count is always logged). ``tmpfs_bytes`` caps each of the two writable in-memory
    mounts, /tmp and /dev/shm. ``bwrap`` is the bubblewrap binary; ``startup_timeout_s`` bounds kernel startup.
    """

    trial_dir: Path
    patches_dir: Path
    scratch_dir: Path
    log_path: Path
    python: Path  # <venv>/bin/python of the sandbox venv (numpy only; see "Interpreter" in the module docstring)
    cell_timeout_s: float = 120.0
    max_cells: int = 150
    memory_bytes: int = 8 * 2**30
    threads: int = 4
    max_output_chars: int = 30_000
    max_log_bytes: int = 4 * 2**20
    tmpfs_bytes: int = 2**30
    bwrap: Path = DEFAULT_BWRAP
    startup_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        for name in ("trial_dir", "patches_dir", "scratch_dir", "log_path", "python", "bwrap"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        checks = (
            ("cell_timeout_s", self.cell_timeout_s > 0),
            ("max_cells", self.max_cells >= 0),
            ("memory_bytes", self.memory_bytes > 0),
            ("threads", self.threads >= 1),
            ("max_output_chars", self.max_output_chars >= 0),
            ("max_log_bytes", self.max_log_bytes >= 1024),
            ("tmpfs_bytes", self.tmpfs_bytes >= 2**20),
            ("startup_timeout_s", self.startup_timeout_s > 0),
        )
        for name, valid in checks:
            if not valid:
                raise ValueError(f"invalid SandboxConfig.{name}: {getattr(self, name)!r}")


@dataclass(frozen=True)
class CellResult:
    """What the editor gets back for one cell; text fields are cut to ``max_output_chars`` in total.

    ``refused`` (an addition to the spec, default False) marks a cell that was not executed because the
    cell budget was spent.
    """

    index: int
    stdout: str
    stderr: str
    error: str | None
    duration_s: float
    timed_out: bool
    truncated: bool
    restarted: bool
    refused: bool = False


@dataclass(frozen=True)
class _Outcome:
    kind: str  # "line" | "timeout" | "eof" | "protocol"
    line: bytes = b""
    detail: str = ""


class _StreamCapture:
    """Bytes one output stream received during one cell, bounded: a head, a rolling tail and a total count."""

    def __init__(self, limit: int) -> None:
        self._tail_cap = min(limit // 4, 256 * 1024)
        self._head_cap = limit - self._tail_cap
        self._head = bytearray()
        self._tail = bytearray()
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self._head_cap - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self._tail += chunk
            excess = len(self._tail) - self._tail_cap
            if excess > 0:
                del self._tail[:excess]

    def text(self) -> tuple[str, bool]:
        """Decoded text and whether bytes were dropped from the middle."""
        omitted = self.total - len(self._head) - len(self._tail)
        if omitted == 0:
            return bytes(self._head + self._tail).decode("utf-8", "replace"), False
        head = bytes(self._head).decode("utf-8", "replace")
        tail = bytes(self._tail).decode("utf-8", "replace")
        return f"{head}\n[... {omitted} bytes omitted from the log ...]\n{tail}", True


def venv_root(python: str | Path) -> Path:
    """Root of the virtual environment of ``<venv>/bin/python``, without resolving the interpreter symlink."""
    interpreter = Path(os.path.abspath(python))
    root = interpreter.parent.parent
    if not (root / "pyvenv.cfg").is_file():
        raise SandboxError(f"{interpreter} is not <venv>/bin/python of a virtual environment ({root}/pyvenv.cfg is missing)")
    return root


@dataclass(frozen=True)
class InterpreterMounts:
    """What the sandbox needs besides /usr to start ``<venv>/bin/python`` (see ``interpreter_mounts``)."""

    venv: Path
    """The venv root as given, bound read-only at the same path."""
    binds: tuple[Path, ...]
    """Read-only binds of a base interpreter outside /usr: its executable, standard library and libpython files."""
    symlinks: tuple[tuple[str, str], ...]
    """``(target, link)`` for each symbolic link outside /usr and the venv on the way to the interpreter."""

    def argv(self) -> list[str]:
        """The bwrap arguments: the venv, the binds, then the links (with their parent directories)."""
        argv = ["--ro-bind", str(self.venv), str(self.venv)]
        for path in self.binds:
            argv += ["--ro-bind", str(path), str(path)]
        for target, link in self.symlinks:
            parent = os.path.dirname(link)
            if parent != "/":
                argv += ["--dir", parent]
            argv += ["--symlink", target, link]
        return argv

    def sources(self) -> tuple[Path, ...]:
        """The host paths whose contents the sandbox can read through these mounts (resolved)."""
        return (self.venv.resolve(), *(path.resolve() for path in self.binds))

    def visible(self) -> tuple[str, ...]:
        """The paths these mounts create inside the sandbox: the venv, each bind and each recreated link."""
        return (str(self.venv), *(str(path) for path in self.binds), *(link for _, link in self.symlinks))


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _walk(start: str, rest: list[str], trees: tuple[str, ...]) -> tuple[list[tuple[str, str]], str]:
    """Resolve ``start/rest`` component by component the way the sandbox will see it. Paths inside ``trees`` (bound
    whole) exist there with their own links; every other symbolic link met on the way is returned as ``(target,
    link)`` to be recreated. Returns those links and the final path."""
    links: list[tuple[str, str]] = []
    current, pending, followed = start, list(rest), 0
    while pending:
        name = pending.pop(0)
        if name in ("", "."):
            continue
        if name == "..":
            current = os.path.dirname(current) or "/"
            continue
        candidate = os.path.join(current, name)
        if not os.path.islink(candidate):
            current = candidate
            continue
        followed += 1
        if followed > _MAX_SYMLINKS:
            raise SandboxError(f"more than {_MAX_SYMLINKS} symbolic links on the way to {os.path.join(start, *rest)}")
        target = os.readlink(candidate)
        if not any(_within(candidate, tree) for tree in trees) and USR_LINKS.get(candidate) != target:
            links.append((target, candidate))
        if target.startswith("/"):
            current = "/"
        pending = [part for part in target.split("/") if part] + pending
    return links, current


def interpreter_mounts(python: str | Path) -> InterpreterMounts:
    """The mounts that let ``<venv>/bin/python`` start inside the sandbox (module docstring, "Interpreter").

    The venv is bound whole. When the interpreter it links to lies under /usr, that is all. Otherwise the base
    installation must look like one: ``<prefix>/bin/pythonX.Y`` with ``<prefix>/lib/pythonX.Y/os.py``. Its executable,
    standard library directory and ``<prefix>/lib/libpython*`` files are bound read-only, and the links on the way are
    recreated. Raises ``SandboxError`` when the interpreter cannot be made visible that way.
    """
    venv = venv_root(python)
    interpreter = Path(os.path.abspath(python))
    trees = (SYSTEM_TREE, str(venv))
    links, final = _walk(str(venv), [*interpreter.relative_to(venv).parts], trees)
    if not os.path.isfile(final):
        raise SandboxError(f"{interpreter} does not lead to an interpreter file (it ends at {final})")
    binds: list[Path] = []
    if not any(_within(final, tree) for tree in trees):
        real = Path(final)
        match = re.fullmatch(r"python(\d+\.\d+)", real.name)
        if match is None:
            raise SandboxError(f"{interpreter} resolves to {real}; outside /usr the base interpreter must be named "
                               "pythonX.Y (create the venv from that file)")
        prefix = real.parent.parent
        stdlib = prefix / "lib" / f"python{match.group(1)}"
        if real.parent.name != "bin" or not (stdlib / "os.py").is_file():
            raise SandboxError(f"{real} is not <prefix>/bin/python{match.group(1)} of an installation with "
                               f"{stdlib}/os.py, so its standard library cannot be bound into the sandbox")
        libraries = sorted(p for p in (prefix / "lib").glob(f"libpython{match.group(1)}*.so*") if p.is_file())
        libraries += [p for p in [prefix / "lib" / "libpython3.so"] if p.is_file()]
        binds = [real, stdlib, *libraries]
    visible = [str(venv), *(str(path) for path in binds)]
    for _, link in links:
        for path in visible:
            if _within(path, link) or _within(link, path):
                raise SandboxError(f"the symbolic link {link} on the way to the interpreter overlaps {path}; create "
                                   "the sandbox venv from the interpreter's real path (realpath of the base python)")
    return InterpreterMounts(venv=venv, binds=tuple(binds), symlinks=tuple(links))


def isolated_environment(threads: int, home: str) -> dict[str, str]:
    """The complete environment set inside a sandbox (after --clearenv), with ``HOME`` at ``home``."""
    value = str(threads)
    return {
        "PATH": "/usr/bin",
        "HOME": home,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
    }


def sandbox_environment(config: SandboxConfig) -> dict[str, str]:
    """The complete environment set inside the kernel's sandbox (after --clearenv)."""
    return isolated_environment(config.threads, SCRATCH_MOUNT)


# Closes every isolated argv after its work mounts: the in-memory root and /dev become read-only, cwd is /work.
SEAL_ARGV = ("--remount-ro", "/dev", "--remount-ro", "/", "--chdir", WORK_MOUNT)


def isolation_argv(bwrap: Path, python: Path, environment: dict[str, str], tmpfs_bytes: int) -> list[str]:
    """The bwrap flags, environment and system mounts shared by the kernel and the update-program driver
    (``weightcode.sandbox.program``): every namespace unshared, no nested user namespace, /usr, /proc, /dev, the venv of
    ``python`` (and its base interpreter when that lies outside /usr, ``interpreter_mounts``) and the two capped tmpfs
    mounts. The caller adds its work mounts, then ``SEAL_ARGV`` and its command."""
    mounts = interpreter_mounts(python)
    tmpfs = str(tmpfs_bytes)
    argv = [
        str(bwrap),
        "--unshare-all",
        "--unshare-user",
        "--disable-userns",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
    ]
    for key, value in environment.items():
        argv += ["--setenv", key, value]
    argv += [
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--proc", "/proc",
        "--dev", "/dev",
        "--size", tmpfs, "--tmpfs", "/dev/shm",
        "--size", tmpfs, "--tmpfs", "/tmp",
        *mounts.argv(),
    ]  # fmt: skip
    return argv


def build_bwrap_argv(config: SandboxConfig, response_fd: int | str) -> list[str]:
    """The bwrap command line for one kernel: /usr, the venv, the three session dirs and the kernel script."""
    interpreter = Path(os.path.abspath(config.python))
    argv = isolation_argv(config.bwrap, config.python, sandbox_environment(config), config.tmpfs_bytes)
    argv += [
        "--ro-bind", str(config.trial_dir.resolve()), TRIAL_MOUNT,
        "--bind", str(config.patches_dir.resolve()), PATCHES_MOUNT,
        "--bind", str(config.scratch_dir.resolve()), SCRATCH_MOUNT,
        "--ro-bind", str(KERNEL_SCRIPT), KERNEL_MOUNT,
        *SEAL_ARGV,
        str(interpreter), "-I", "-u", KERNEL_MOUNT, "--response-fd", str(response_fd),
    ]  # fmt: skip
    return argv


def kernel_sha256() -> str:
    """sha256 of the kernel script that is bound into the sandbox."""
    return hashlib.sha256(KERNEL_SCRIPT.read_bytes()).hexdigest()


def bwrap_version(bwrap: str | Path = DEFAULT_BWRAP) -> str:
    """``bwrap --version`` output, for provenance records."""
    completed = subprocess.run([str(bwrap), "--version"], capture_output=True, text=True, timeout=10, env={}, check=True)
    return completed.stdout.strip()


def check_bwrap(bwrap: str | Path = DEFAULT_BWRAP) -> str:
    """The ``bwrap --version`` of a bubblewrap that can run the sandbox; ``SandboxError`` naming the remedy otherwise
    (missing, unreadable version, or older than ``MIN_BWRAP_VERSION``)."""
    needed = ".".join(map(str, MIN_BWRAP_VERSION))
    if not os.access(bwrap, os.X_OK):
        raise SandboxError(f"bubblewrap is missing or not executable at {bwrap}: install bubblewrap {needed} or newer "
                           "(Ubuntu 24.04+, Debian 12+: sudo apt install bubblewrap) or pass its path with --bwrap")
    try:
        text = bwrap_version(bwrap)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxError(f"could not run {bwrap} --version: {exc}") from exc
    match = re.fullmatch(r"bubblewrap (\d+)\.(\d+)\.(\d+)\S*", text)
    if match is None:
        raise SandboxError(f"cannot read the version of {bwrap} from {text!r}")
    if tuple(int(part) for part in match.groups()) < MIN_BWRAP_VERSION:
        raise SandboxError(f"{bwrap} is {text}; the sandbox needs bubblewrap {needed} or newer (for --disable-userns). "
                           "Install a newer one, or pass its path with --bwrap")
    return text


def recover_interrupted_cell(log_path: str | Path) -> dict[str, object] | None:
    """Log the cell that a dead host process left in flight, if any, and return its log record.

    The runner should call this after a session whose host process was killed; ``SandboxKernel`` calls it
    on construction.
    """
    log_path = Path(os.path.abspath(log_path))
    inflight = _inflight_path(log_path)
    if not inflight.exists():
        return None
    pending = json.loads(inflight.read_text(encoding="utf-8"))
    if not isinstance(pending, dict) or pending.get("schema") != INFLIGHT_SCHEMA:
        raise SandboxError(f"{inflight} is not a {INFLIGHT_SCHEMA} record")
    logged, _ = _log_counts(log_path)
    if pending.get("index") == logged - 1:  # its result line was written; only the cleanup was missed
        inflight.unlink()
        return None
    if pending.get("index") != logged:
        raise SandboxError(f"{inflight} is for cell {pending.get('index')}, but {log_path} holds {logged} cells")
    record = _cell_record(
        index=logged,
        timestamp=str(pending["timestamp"]),
        status="interrupted",
        code=str(pending["code"]),
        executed=True,
        error="The host process stopped while this cell was running; its output and result are unknown.",
        kernel_generation=int(pending["kernel_generation"]),
    )
    _append_json_line(log_path, record)
    inflight.unlink()
    return record


class SandboxKernel:
    """A persistent Python kernel in bubblewrap. Not thread-safe: one caller runs one cell at a time.

    Start and run it from a thread that lives as long as the kernel: bwrap's --die-with-parent fires when the
    spawning thread exits. The rlimits are applied in ``preexec_fn``, which is unsafe if other threads of the
    host process fork at the same moment.
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._venv = venv_root(config.python)
        self._log_path = Path(os.path.abspath(config.log_path))
        self._inflight = _inflight_path(self._log_path)
        self._check_paths()
        self._kernel_sha256 = kernel_sha256()
        # A response line holds at most two kernel-clipped strings of max_log_bytes characters.
        self._response_limit = 2 * _JSON_ESCAPE_FACTOR * config.max_log_bytes + _RESPONSE_READ_CHUNK
        self._proc: subprocess.Popen[bytes] | None = None
        self._req_w: int | None = None
        self._out_r: int | None = None
        self._err_r: int | None = None
        self._resp_r: int | None = None
        self._resp_buffer = bytearray()
        self._carry: tuple[_StreamCapture, _StreamCapture] | None = None
        self._ready: dict[str, object] = {}
        self._generation = 0
        self._started = False
        self._closed = False
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        recover_interrupted_cell(self._log_path)
        self._next_index, self._executed = _log_counts(self._log_path)

    # ------------------------------------------------------------------ public API

    @property
    def config(self) -> SandboxConfig:
        return self._config

    @property
    def cells_used(self) -> int:
        """Cells sent to a kernel so far in this session (refused cells excluded)."""
        return self._executed

    @property
    def cells_remaining(self) -> int:
        return max(0, self._config.max_cells - self._executed)

    @property
    def cells_logged(self) -> int:
        """Lines in the cell log, refused cells included."""
        return self._next_index

    @property
    def generation(self) -> int:
        """0 for the first kernel, incremented by every restart."""
        return self._generation

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def ready_info(self) -> dict[str, object]:
        """The kernel's startup message (protocol, interpreter version, pid inside the sandbox)."""
        return dict(self._ready)

    def start(self) -> None:
        """Start the first kernel and wait for its ready message."""
        if self._closed:
            raise SandboxError("the kernel is closed")
        if self._started:
            raise SandboxError("the kernel is already started")
        self._spawn()
        self._started = True

    def run(self, code: str) -> CellResult:
        """Execute one cell, or refuse it when the cell budget is spent. Every call is logged."""
        if not isinstance(code, str):
            raise TypeError(f"code must be a str, got {type(code).__name__}")
        if self._closed:
            raise SandboxError("the kernel is closed")
        if not self._started:
            raise SandboxError("call start() before run()")
        config = self._config
        index = self._next_index
        timestamp = _utc_now()
        if self._executed >= config.max_cells:
            error = f"Cell refused: the budget of {config.max_cells} Python cells is used up. The code was not executed."
            self._log(_cell_record(
                index=index, timestamp=timestamp, status="refused", code=code, executed=False, refused=True,
                error=error, kernel_generation=self._generation,
            ))  # fmt: skip
            return CellResult(index, "", "", error, 0.0, False, False, False, refused=True)

        out, err = self._take_captures()
        notes: list[str] = []
        restarted = False
        if not self.running:
            # The kernel exited between cells (or an earlier restart failed): start a fresh one first.
            status = self._shutdown(out, err)
            try:
                self._spawn(out, err)
            except SandboxError as exc:
                self._log(_cell_record(
                    index=index, timestamp=timestamp, status="sandbox_error", code=code, executed=False,
                    error=f"The sandbox kernel could not be restarted: {exc}", kernel_generation=self._generation,
                ))  # fmt: skip
                raise
            self._generation += 1
            restarted = True
            reason = "an earlier restart had failed" if status is None else f"exit status {status}"
            notes += [f"The Python kernel had stopped before this cell ({reason}).", STATE_LOST_NOTE]
        generation = self._generation
        self._drain(out, err, wait=False)  # output that arrived between cells belongs to this cell

        token = secrets.token_hex(16)
        request = {"type": "run", "id": token, "index": index, "code": code, "max_chars": config.max_log_bytes}
        payload = (json.dumps(request, ensure_ascii=True) + "\n").encode("ascii")
        inflight = {"schema": INFLIGHT_SCHEMA, "index": index, "timestamp": timestamp, "code": code,
                    "kernel_generation": generation}  # fmt: skip
        _write_json_atomic(self._inflight, inflight)
        began = time.monotonic()
        outcome = self._exchange(payload, began + config.cell_timeout_s, out, err)
        duration = time.monotonic() - began

        shown: str | None = None
        cell_error: str | None = None
        if outcome.kind == "line":
            values, problem = _parse_result(outcome.line, token, index)
            if values is not None:
                self._drain(out, err, wait=False)
                shown, cell_error = values
            else:
                outcome = _Outcome("protocol", detail=problem)

        failure: str | None = None
        restart_error: SandboxError | None = None
        if outcome.kind == "line":
            status_name = "error" if cell_error else "ok"
        else:
            exit_status = self._shutdown(out, err, grace_s=_EXIT_GRACE_S if outcome.kind == "eof" else 0.0)
            if outcome.kind == "timeout":
                status_name = "timeout"
                failure = f"TimeoutError: the cell did not finish within {config.cell_timeout_s:g} s and was stopped."
            elif outcome.kind == "eof":
                status_name = "crashed"
                failure = f"KernelDied: the Python kernel exited while running this cell (exit status {exit_status})."
            else:
                status_name = "protocol_error"
                failure = f"SandboxProtocolError: {outcome.detail}. The kernel was stopped."
            try:
                self._spawn()
            except SandboxError as exc:
                restart_error = exc
            else:
                self._generation += 1
                restarted = True

        parts = list(notes)
        if cell_error:
            parts.append(cell_error.rstrip("\n"))
        if failure:
            parts.append(failure)
            parts.append(STATE_LOST_NOTE if restart_error is None else f"The kernel could not be restarted: {restart_error}")
        error_text = "\n".join(parts) if parts else None

        stdout_log, stdout_cut = out.text()
        stderr_log, stderr_cut = err.text()
        if shown is not None:
            stdout_log = _append_repr(stdout_log, shown)
        (ret_stdout, ret_stderr, ret_error), clipped = _fit_output(
            [stdout_log, stderr_log, error_text or ""], config.max_output_chars
        )
        truncated = clipped or stdout_cut or stderr_cut
        self._executed += 1
        self._log(_cell_record(
            index=index, timestamp=timestamp, status=status_name, code=code, executed=True, stdout=stdout_log,
            stderr=stderr_log, error=error_text, duration_s=duration, timed_out=outcome.kind == "timeout",
            restarted=restarted, truncated=truncated, log_truncated=stdout_cut or stderr_cut,
            stdout_bytes=out.total, stderr_bytes=err.total, kernel_generation=generation,
        ))  # fmt: skip
        if restart_error is not None:
            raise SandboxError(f"cell {index} ended the kernel and no new kernel could be started") from restart_error
        return CellResult(
            index=index,
            stdout=ret_stdout,
            stderr=ret_stderr,
            error=None if error_text is None else ret_error,
            duration_s=duration,
            timed_out=outcome.kind == "timeout",
            truncated=truncated,
            restarted=restarted,
        )

    def close(self) -> None:
        """Kill the sandbox and everything in it. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._shutdown()

    def provenance(self) -> dict[str, object]:
        """What ran: protocol, log schema, kernel hash, bwrap version and argv, interpreter paths."""
        config = self._config
        return {
            "protocol": PROTOCOL,
            "cell_log_schema": CELL_LOG_SCHEMA,
            "kernel_sha256": self._kernel_sha256,
            "bwrap": str(config.bwrap),
            "bwrap_version": bwrap_version(config.bwrap),
            "argv": build_bwrap_argv(config, "<response-fd>"),
            "python": str(config.python),
            "python_realpath": str(Path(config.python).resolve()),
            "kernel_python": self._ready.get("python"),
        }

    def __enter__(self) -> SandboxKernel:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ process management

    def _check_paths(self) -> None:
        config = self._config
        check_bwrap(config.bwrap)
        interpreter = Path(os.path.abspath(config.python))
        if not interpreter.exists():
            raise SandboxError(f"sandbox interpreter {interpreter} does not exist")
        sources = interpreter_mounts(config.python).sources()  # raises when the interpreter cannot be made visible
        dirs: dict[str, Path] = {}
        for name in ("trial_dir", "patches_dir", "scratch_dir"):
            path = getattr(config, name)
            if not path.is_dir():
                raise SandboxError(f"{name} {path} is not an existing directory")
            dirs[name] = path.resolve()
        names = list(dirs)
        for position, first in enumerate(names):
            for second in names[position + 1 :]:
                if _is_within(dirs[first], dirs[second]) or _is_within(dirs[second], dirs[first]):
                    raise SandboxError(f"{first} {dirs[first]} and {second} {dirs[second]} overlap")
            for source in sources:
                if _is_within(dirs[first], source) or _is_within(source, dirs[first]):
                    raise SandboxError(f"{first} {dirs[first]} overlaps the interpreter mount {source}")
        log_real = self._log_path.resolve()
        for name, path in [*dirs.items(), *(("interpreter mount", source) for source in sources)]:
            if _is_within(log_real, path):
                raise SandboxError(f"log_path {self._log_path} lies inside {name} {path}, which the sandbox can see")
        if not KERNEL_SCRIPT.is_file():
            raise SandboxError(f"kernel script {KERNEL_SCRIPT} is missing")

    def _spawn(self, out: _StreamCapture | None = None, err: _StreamCapture | None = None) -> None:
        """Start bwrap and the kernel; startup output goes to ``out``/``err`` or is carried to the next cell."""
        config = self._config
        if kernel_sha256() != self._kernel_sha256:
            raise SandboxError(f"{KERNEL_SCRIPT} changed during the session")
        pipes = [os.pipe() for _ in range(4)]
        descriptors = [fd for pair in pipes for fd in pair]
        if min(descriptors) < 3:
            for fd in descriptors:
                os.close(fd)
            raise SandboxError("file descriptors 0, 1 and 2 must be open in the host process")
        (req_r, req_w), (out_r, out_w), (err_r, err_w), (resp_r, resp_w) = pipes
        try:
            proc = subprocess.Popen(
                build_bwrap_argv(config, resp_w),
                stdin=req_r,
                stdout=out_w,
                stderr=err_w,
                pass_fds=(resp_w,),
                close_fds=True,
                start_new_session=True,
                env={},
                cwd="/",
                preexec_fn=functools.partial(_limit_child, config.memory_bytes),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            for fd in descriptors:
                os.close(fd)
            raise SandboxError(f"could not start bubblewrap: {exc}") from exc
        for fd in (req_r, out_w, err_w, resp_w):
            os.close(fd)
        for fd in (req_w, out_r, err_r, resp_r):
            os.set_blocking(fd, False)
        self._proc = proc
        self._req_w, self._out_r, self._err_r, self._resp_r = req_w, out_r, err_r, resp_r
        self._resp_buffer = bytearray()

        carry = out is None or err is None
        if out is None or err is None:
            out, err = self._fresh_captures()
        outcome = self._exchange(b"", time.monotonic() + config.startup_timeout_s, out, err)
        ready: dict[str, object] | None = None
        if outcome.kind == "line":
            try:
                message = json.loads(outcome.line)
            except ValueError:
                message = None
            if isinstance(message, dict) and message.get("type") == "ready" and message.get("protocol") == PROTOCOL:
                ready = message
            else:
                outcome = _Outcome("protocol", detail=f"unexpected first message {outcome.line[:200]!r}")
        if ready is None:
            status = self._shutdown(out, err)
            stderr_text, _ = err.text()
            reason = {"timeout": f"no ready message within {config.startup_timeout_s:g} s",
                      "eof": "the kernel exited during startup"}.get(outcome.kind, outcome.detail)
            raise SandboxError(f"the sandbox kernel did not start: {reason} (exit status {status}); stderr: {stderr_text[-2000:]!r}")
        self._ready = ready
        if carry:
            self._carry = (out, err)

    def _shutdown(self, out: _StreamCapture | None = None, err: _StreamCapture | None = None,
                  grace_s: float = 0.0) -> int | None:
        """Kill the sandbox's process group, reap bwrap, collect remaining output, close the pipes.

        Killing bwrap's group kills its in-namespace init; the kernel then kills every other process in the
        sandbox's pid namespace, including ones that called setsid().
        """
        proc = self._proc
        if proc is None:
            self._close_fds()
            return None
        if grace_s > 0:
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                pass  # still running after the grace period: it is killed below
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # the group emptied between poll() and killpg()
        try:
            status = proc.wait(timeout=_REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"bubblewrap (pid {proc.pid}) did not exit after SIGKILL") from exc
        if out is not None and err is not None:
            self._drain(out, err, wait=True)
        self._close_fds()
        self._proc = None
        return status

    def _close_fds(self) -> None:
        for name in ("_req_w", "_out_r", "_err_r", "_resp_r"):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def _output_closed(self, fd: int) -> None:
        """The kernel closed fd 1 or fd 2 (user code did): stop reading that pipe."""
        for name in ("_out_r", "_err_r"):
            if getattr(self, name) == fd:
                os.close(fd)
                setattr(self, name, None)

    def _fresh_captures(self) -> tuple[_StreamCapture, _StreamCapture]:
        limit = self._config.max_log_bytes
        return _StreamCapture(limit), _StreamCapture(limit)

    def _take_captures(self) -> tuple[_StreamCapture, _StreamCapture]:
        """Captures for the next cell, seeded with any output a kernel produced while starting."""
        if self._carry is not None:
            captures, self._carry = self._carry, None
            return captures
        return self._fresh_captures()

    def _exchange(self, payload: bytes, deadline: float, out: _StreamCapture, err: _StreamCapture) -> _Outcome:
        """Write ``payload`` to the kernel and read output until one response line arrives or time runs out."""
        assert self._resp_r is not None and self._req_w is not None
        selector = selectors.DefaultSelector()
        try:
            selector.register(self._resp_r, selectors.EVENT_READ, "response")
            for fd, capture in ((self._out_r, out), (self._err_r, err)):
                if fd is not None:
                    selector.register(fd, selectors.EVENT_READ, capture)
            pending = memoryview(payload)
            if pending:
                selector.register(self._req_w, selectors.EVENT_WRITE, "request")
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return _Outcome("timeout")
                for key, _ in selector.select(remaining):
                    if key.data == "request":
                        try:
                            written = os.write(key.fd, pending[:_READ_CHUNK])
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            return _Outcome("eof", detail="the kernel stopped reading requests")
                        pending = pending[written:]
                        if not pending:
                            selector.unregister(key.fd)
                    elif key.data == "response":
                        try:
                            chunk = os.read(key.fd, _RESPONSE_READ_CHUNK)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            return _Outcome("eof", detail="the response channel closed")
                        self._resp_buffer += chunk
                        end = self._resp_buffer.find(b"\n")
                        if end >= 0:
                            line = bytes(self._resp_buffer[:end])
                            extra = len(self._resp_buffer) - end - 1
                            self._resp_buffer = bytearray()
                            if extra or pending:
                                return _Outcome("protocol", detail="the response channel carried data beyond one response")
                            return _Outcome("line", line=line)
                        if len(self._resp_buffer) > self._response_limit:
                            return _Outcome("protocol", detail=f"a response exceeded {self._response_limit} bytes")
                    else:
                        try:
                            chunk = os.read(key.fd, _READ_CHUNK)
                        except BlockingIOError:
                            continue
                        if chunk:
                            key.data.feed(chunk)
                        else:
                            selector.unregister(key.fd)
                            self._output_closed(key.fd)
        finally:
            selector.close()

    def _drain(self, out: _StreamCapture, err: _StreamCapture, *, wait: bool) -> None:
        """Collect output already in the pipes (``wait=False``) or up to end of file (``wait=True``), within _DRAIN_S."""
        selector = selectors.DefaultSelector()
        try:
            for fd, capture in ((self._out_r, out), (self._err_r, err)):
                if fd is not None:
                    selector.register(fd, selectors.EVENT_READ, capture)
            deadline = time.monotonic() + _DRAIN_S
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                events = selector.select(remaining if wait else 0)
                if not events and not wait:
                    break
                for key, _ in events:
                    try:
                        chunk = os.read(key.fd, _READ_CHUNK)
                    except BlockingIOError:
                        continue
                    if chunk:
                        key.data.feed(chunk)
                    else:
                        selector.unregister(key.fd)
                        if not wait:
                            self._output_closed(key.fd)
        finally:
            selector.close()

    def _log(self, record: dict[str, object]) -> None:
        _append_json_line(self._log_path, record)
        self._next_index += 1
        self._inflight.unlink(missing_ok=True)


# ---------------------------------------------------------------------- helpers


def _limit_child(memory_bytes: int) -> None:
    """preexec_fn: address-space cap and no core dumps for bwrap and everything it starts."""
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _is_within(path: Path, root: Path) -> bool:
    """True when ``path`` is ``root`` or lies below it (both already absolute and resolved)."""
    return path == root or root in path.parents


def _inflight_path(log_path: Path) -> Path:
    return log_path.with_name(log_path.name + INFLIGHT_SUFFIX)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _cell_record(
    *,
    index: int,
    timestamp: str,
    status: str,
    code: str,
    executed: bool,
    kernel_generation: int,
    stdout: str = "",
    stderr: str = "",
    error: str | None = None,
    duration_s: float = 0.0,
    timed_out: bool = False,
    restarted: bool = False,
    refused: bool = False,
    truncated: bool = False,
    log_truncated: bool = False,
    stdout_bytes: int = 0,
    stderr_bytes: int = 0,
) -> dict[str, object]:
    """One cell-log line; every line has the same keys."""
    return {
        "schema": CELL_LOG_SCHEMA,
        "timestamp": timestamp,
        "index": index,
        "status": status,
        "executed": executed,
        "refused": refused,
        "code": code,
        "code_sha256": _sha256_text(code),
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "duration_s": round(duration_s, 6),
        "timed_out": timed_out,
        "restarted": restarted,
        "truncated": truncated,
        "log_truncated": log_truncated,
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
        "kernel_generation": kernel_generation,
    }


def _append_json_line(path: Path, record: dict[str, object]) -> None:
    """Append one JSON line with a single O_APPEND descriptor and fsync it."""
    data = (json.dumps(record, ensure_ascii=True) + "\n").encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    data = json.dumps(payload, ensure_ascii=True).encode("ascii")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _log_counts(log_path: Path) -> tuple[int, int]:
    """(lines, executed cells) of an existing cell log. A malformed log is an error, never skipped."""
    if not log_path.exists():
        return 0, 0
    logged = executed = 0
    with log_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise SandboxError(f"{log_path}:{number + 1} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict) or record.get("schema") != CELL_LOG_SCHEMA or record.get("index") != number:
                raise SandboxError(f"{log_path}:{number + 1} is not cell {number} of a {CELL_LOG_SCHEMA} log")
            logged += 1
            executed += bool(record.get("executed"))
    return logged, executed


def _parse_result(line: bytes, token: str, index: int) -> tuple[tuple[str | None, str | None] | None, str]:
    """Validate a kernel response; return ((repr, error), "") or (None, problem)."""
    try:
        message = json.loads(line)
    except ValueError:
        return None, f"the kernel sent a line that is not valid JSON ({len(line)} bytes)"
    if not isinstance(message, dict):
        return None, "the kernel sent a response that is not a JSON object"
    if message.get("type") == "fatal":
        return None, f"the kernel rejected the request: {message.get('error')}"
    if message.get("type") != "result" or message.get("id") != token or message.get("index") != index:
        return None, "the response does not match the request"
    shown, error = message.get("repr"), message.get("error")
    if not (shown is None or isinstance(shown, str)) or not (error is None or isinstance(error, str)):
        return None, "the response fields 'repr' and 'error' must be strings or null"
    return (shown, error), ""


def _append_repr(stdout: str, shown: str) -> str:
    if stdout and not stdout.endswith("\n"):
        stdout += "\n"
    return stdout + shown + "\n"


def _clip(text: str, limit: int) -> str:
    """Keep the head and tail of ``text`` within exactly ``limit`` characters, marker included."""
    if len(text) <= limit:
        return text
    template = "\n[... {} characters omitted ...]\n"
    bound = len(template.format(len(text)))
    if limit < bound + 2:
        return text[:limit]
    keep = limit - bound
    head = keep // 2
    tail = keep - head
    return text[:head] + template.format(len(text) - keep) + text[len(text) - tail :]


def _fit_output(texts: Sequence[str], limit: int) -> tuple[list[str], bool]:
    """Share ``limit`` characters max-min fairly among ``texts`` and clip each to its share."""
    lengths = [len(text) for text in texts]
    if sum(lengths) <= limit:
        return list(texts), False
    shares = [0] * len(texts)
    remaining = limit
    for rank, position in enumerate(sorted(range(len(texts)), key=lengths.__getitem__)):
        share = remaining // (len(texts) - rank)
        shares[position] = min(lengths[position], share)
        remaining -= shares[position]
    return [_clip(text, share) for text, share in zip(texts, shares)], True
