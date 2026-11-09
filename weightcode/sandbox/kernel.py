"""WeightCode sandbox kernel: one persistent Python namespace driven by JSON-line requests.

This script runs inside bubblewrap under the sandbox interpreter, started by ``weightcode.sandbox.host`` as
``python -I -u /opt/weightcode/kernel.py --response-fd N``. It uses the standard library only; numpy and every
other module are imported by user code, never by the kernel.

Protocol ``weightcode.kernel.v0``:

- Startup. The request stream (fd 0) and the response pipe (fd N, inherited through bwrap) are moved to
  close-on-exec descriptors numbered 100 or higher, and fd 0 is pointed at /dev/null. User code that reads
  stdin gets end of file, and processes it starts inherit neither channel. The kernel then writes
  ``{"type": "ready", ...}`` on the response pipe.
- Each request is one JSON line ``{"type": "run", "id", "index", "code", "max_chars"}``. The kernel runs
  ``code`` in the persistent namespace and writes one JSON line
  ``{"type": "result", "id", "index", "repr", "error"}`` on the response pipe. ``repr`` is the repr of the
  last statement when it is an expression whose value is not None.
- User output is not part of the protocol. fd 1 and fd 2 are pipes that the host reads directly, so
  ``print``, ``os.write(1, ...)``, C-level output and child processes all land in the cell's captured
  stdout or stderr and cannot corrupt a response.
- End of file on the request stream ends the kernel.
"""

import ast
import builtins
import fcntl
import json
import linecache
import os
import sys
import traceback
import types

PROTOCOL = "weightcode.kernel.v0"
PRIVATE_FD_FLOOR = 100  # protocol descriptors live at or above this number, away from files user code opens

# Frames whose code comes from this file are kernel machinery and are cut from user tracebacks.
_KERNEL_CODE_FILE = (lambda: None).__code__.co_filename
# The kernel module stays referenced after sys.modules["__main__"] is replaced by the user namespace.
_KERNEL_MODULE = sys.modules[__name__]


def _clip(text, limit):
    """Keep the head and the tail of ``text`` when it is longer than ``limit`` characters."""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[... {omitted} characters omitted by the kernel ...]\n{text[len(text) - tail:]}"


def _format_compile_error(exc):
    """A cell that does not compile: the exception alone, without kernel frames."""
    return "".join(traceback.format_exception_only(type(exc), exc))


def _format_runtime_error(exc):
    """A cell that raised: the traceback from the first user frame on."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename == _KERNEL_CODE_FILE:
        tb = tb.tb_next
    exc = exc.with_traceback(tb)
    sys.last_type, sys.last_value, sys.last_traceback = type(exc), exc, tb
    sys.last_exc = exc
    text = "".join(traceback.format_exception(type(exc), exc, tb))
    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
        text += f"[kernel] {type(exc).__name__} was caught: the kernel keeps running and its state is intact.\n"
    return text


def _describe(exc, formatter):
    """Format an exception; if formatting itself fails, say so instead of losing the error."""
    try:
        return formatter(exc)
    except BaseException as failure:
        return f"{type(exc).__name__} (its traceback could not be formatted: {type(failure).__name__})\n"


def run_cell(namespace, code, filename, max_chars):
    """Run one cell in ``namespace`` and return ``(repr_of_last_expression, error_text)``."""
    linecache.cache[filename] = (len(code), None, code.splitlines(keepends=True), filename)
    try:
        tree = ast.parse(code, filename=filename, mode="exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(body=tree.body.pop().value)
        body = compile(tree, filename, "exec", dont_inherit=True)
        tail = None if last is None else compile(last, filename, "eval", dont_inherit=True)
    except BaseException as exc:
        return None, _clip(_describe(exc, _format_compile_error), max_chars)
    try:
        exec(body, namespace)
        value = None if tail is None else eval(tail, namespace)
        shown = None if value is None else repr(value)
    except BaseException as exc:
        return None, _clip(_describe(exc, _format_runtime_error), max_chars)
    return (None if shown is None else _clip(shown, max_chars)), None


def _flush_user_streams():
    """Flush whatever sys.stdout and sys.stderr are now, so their output precedes the response."""
    problems = []
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.flush()
        except BaseException as exc:
            problems.append(f"[kernel] sys.{name}.flush() failed: {type(exc).__name__}: {exc}")
    return problems


def _user_namespace():
    """A fresh ``__main__`` module for user code, installed in sys.modules like a script's namespace."""
    module = types.ModuleType("__main__")
    module.__dict__["__builtins__"] = builtins
    sys.modules["__main__"] = module
    return module.__dict__


def _send(fd, message):
    data = (json.dumps(message, ensure_ascii=True) + "\n").encode("ascii")
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _read_line(fd, pending):
    """Next request line, or None at end of file."""
    while True:
        end = pending.find(b"\n")
        if end >= 0:
            line = bytes(pending[:end])
            del pending[: end + 1]
            return line
        chunk = os.read(fd, 1 << 16)
        if not chunk:
            return None
        pending += chunk


def _parse_request(line):
    request = json.loads(line)
    if not isinstance(request, dict) or request.get("type") != "run":
        raise ValueError(f"unknown request type {request.get('type') if isinstance(request, dict) else request!r}")
    token, index, code, max_chars = request["id"], request["index"], request["code"], request["max_chars"]
    if not isinstance(token, str) or not isinstance(code, str):
        raise TypeError("id and code must be strings")
    if not isinstance(index, int) or not isinstance(max_chars, int) or max_chars < 1:
        raise TypeError("index and max_chars must be integers, max_chars positive")
    return token, index, code, max_chars


def main(argv):
    """Serve requests until end of file; return the process exit status."""
    if len(argv) != 2 or argv[0] != "--response-fd":
        os.write(2, b"usage: kernel.py --response-fd N\n")
        return 2
    inherited = int(argv[1])
    response = fcntl.fcntl(inherited, fcntl.F_DUPFD_CLOEXEC, PRIVATE_FD_FLOOR)
    os.close(inherited)
    requests = fcntl.fcntl(0, fcntl.F_DUPFD_CLOEXEC, PRIVATE_FD_FLOOR)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    # -I makes the interpreter ignore PYTHONDONTWRITEBYTECODE, so the kernel sets the flag itself.
    sys.dont_write_bytecode = True
    sys.argv = [""]
    namespace = _user_namespace()
    _send(response, {"type": "ready", "protocol": PROTOCOL, "python": sys.version, "pid": os.getpid()})
    pending = bytearray()
    while True:
        line = _read_line(requests, pending)
        if line is None:
            return 0
        try:
            token, index, code, max_chars = _parse_request(line)
        except (ValueError, KeyError, TypeError) as exc:
            _send(response, {"type": "fatal", "error": f"malformed request: {type(exc).__name__}: {exc}"})
            return 3
        shown, error = run_cell(namespace, code, f"<cell-{index}>", max_chars)
        problems = _flush_user_streams()
        if problems:
            error = "\n".join(([error.rstrip("\n")] if error else []) + problems) + "\n"
        _send(response, {"type": "result", "id": token, "index": index, "repr": shown, "error": error})


if __name__ == "__main__":
    try:
        status = main(sys.argv[1:])
    except BaseException:
        # A kernel bug: report it on fd 2, which the host captures, and stop.
        os.write(2, traceback.format_exc().encode("utf-8", "replace"))
        status = 4
    # os._exit: user threads and atexit hooks registered by user code must not keep the kernel alive.
    os._exit(status)
