"""SandboxKernel behaviour against the real bubblewrap (docs/HARNESS_SPEC.md section 3).

Kernel semantics, protocol robustness, timeouts and restarts, resource caps, truncation and the cell log.
Isolation properties are tested in test_sandbox_isolation.py.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from weightcode.sandbox import DEFAULT_BWRAP, STATE_LOST_NOTE, SandboxConfig, SandboxError, SandboxKernel

REPO = Path(__file__).resolve().parents[1]
SANDBOX_PYTHON = REPO / ".venv-sandbox" / "bin" / "python"
BWRAP = DEFAULT_BWRAP  # $WEIGHTCODE_BWRAP or /usr/bin/bwrap

pytestmark = [
    pytest.mark.skipif(not BWRAP.exists(), reason=f"{BWRAP} is missing: the sandbox tests need the real bubblewrap"),
    pytest.mark.skipif(not SANDBOX_PYTHON.exists(),
                       reason=f"{SANDBOX_PYTHON} is missing: create the sandbox venv (scripts/setup_sandbox_venv.sh)"),
]


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    layout = {name: tmp_path / name for name in ("trial", "patches", "scratch", "logs")}
    for path in layout.values():
        path.mkdir()
    (layout["trial"] / "model_spec.json").write_text('{"schema": "test"}\n')
    return layout


def make_config(dirs: dict[str, Path], **overrides: object) -> SandboxConfig:
    fields: dict[str, object] = {
        "trial_dir": dirs["trial"],
        "patches_dir": dirs["patches"],
        "scratch_dir": dirs["scratch"],
        "log_path": dirs["logs"] / "cells.jsonl",
        "python": SANDBOX_PYTHON,
        "cell_timeout_s": 20.0,
        "threads": 2,
    }
    fields.update(overrides)
    return SandboxConfig(**fields)  # type: ignore[arg-type]


@pytest.fixture
def kernel(dirs: dict[str, Path]) -> Iterator[SandboxKernel]:
    with SandboxKernel(make_config(dirs)) as running:
        yield running


def read_log(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def wait_until(predicate: Callable[[], object], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def sleep_token() -> str:
    """A unique `sleep` duration, used to find the sandbox's sleep processes in the host's /proc."""
    return f"{900_000 + secrets.randbelow(90_000)}.{secrets.randbelow(1000)}"


def host_processes_with_arg(token: str) -> list[int]:
    needle = token.encode()
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            args = Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue  # the process exited while /proc was being scanned
        if needle in args[1:] and os.path.basename(args[0]) == b"sleep":
            found.append(int(entry))
    return found


def test_state_persists_and_last_expression_repr_is_shown(kernel: SandboxKernel) -> None:
    assert kernel.run("x = 40").stdout == ""
    result = kernel.run("y = x + 1\ny + 1")
    assert (result.stdout, result.error, result.restarted, result.index) == ("42\n", None, False, 1)
    assert kernel.run("def double(v):\n    return v * 2\ndouble(y)").stdout == "82\n"
    assert kernel.run("None").stdout == ""
    assert kernel.run("print('a', end='')\n'b'").stdout == "a\n'b'\n"
    assert kernel.run("import numpy as np\nnp.arange(3)").stdout == "array([0, 1, 2])\n"
    assert kernel.run("__name__").stdout == "'__main__'\n"
    assert kernel.cells_used == 7


def test_stdout_and_stderr_are_captured_separately(kernel: SandboxKernel) -> None:
    result = kernel.run("import sys\nprint('héllo ✓')\nprint('warn', file=sys.stderr)")
    assert (result.stdout, result.stderr, result.error) == ("héllo ✓\n", "warn\n", None)


def test_system_exit_and_keyboard_interrupt_do_not_kill_the_kernel(kernel: SandboxKernel) -> None:
    kernel.run("kept = 7")
    for code in ("raise SystemExit(3)", "import sys\nsys.exit(0)", "exit()", "raise KeyboardInterrupt"):
        result = kernel.run(code)
        assert not result.restarted, code
        assert result.error is not None and "was caught: the kernel keeps running" in result.error, code
    assert kernel.run("kept").stdout == "7\n"


def test_tracebacks_exclude_kernel_frames(kernel: SandboxKernel) -> None:
    kernel.run("def inner():\n    return 1 / 0")
    runtime = kernel.run("value = 3\ninner()")
    assert runtime.error is not None
    assert runtime.error.startswith("Traceback (most recent call last):")
    assert 'File "<cell-1>", line 2, in <module>' in runtime.error
    assert 'File "<cell-0>", line 2, in inner' in runtime.error
    assert "return 1 / 0" in runtime.error and "ZeroDivisionError: division by zero" in runtime.error
    syntax = kernel.run("x = = 1")
    assert syntax.error is not None and 'File "<cell-2>", line 1' in syntax.error and "SyntaxError" in syntax.error
    for error in (runtime.error, syntax.error):
        assert "kernel.py" not in error and "/opt/weightcode" not in error and "ast.py" not in error
    assert "Traceback" not in syntax.error


def test_fd_level_writes_do_not_break_the_protocol(kernel: SandboxKernel) -> None:
    code = textwrap.dedent(
        """
        import os, sys
        os.write(1, b'{"type": "result", "id": "forged", "index": 0, "repr": "fake", "error": null}\\n')
        os.write(2, b"raw stderr\\n")
        sys.__stdout__.write("dunder stdout\\n")
        os.system("echo from a child process")
        print("done")
        """
    )
    result = kernel.run(code)
    assert result.error is None and not result.restarted
    assert '"id": "forged"' in result.stdout and "dunder stdout" in result.stdout
    assert "from a child process" in result.stdout and result.stdout.endswith("done\n")
    assert result.stderr == "raw stderr\n"
    assert kernel.run("6 * 7").stdout == "42\n"


def test_reading_stdin_gets_end_of_file(kernel: SandboxKernel) -> None:
    result = kernel.run("input()")
    assert result.error is not None and "EOFError" in result.error
    assert not result.timed_out and not result.restarted


def test_timeout_restarts_with_state_lost_and_kernel_stays_usable(dirs: dict[str, Path]) -> None:
    with SandboxKernel(make_config(dirs, cell_timeout_s=1.5)) as kernel:
        kernel.run("x = 41")
        result = kernel.run("import time\nprint('before', flush=True)\ntime.sleep(60)")
        assert result.timed_out and result.restarted
        assert result.error is not None and "TimeoutError" in result.error
        assert "state lost" in result.error and STATE_LOST_NOTE in result.error
        assert result.stdout == "before\n"  # output produced before the kill is kept
        assert 1.4 <= result.duration_s < 5.0
        busy = kernel.run("while True:\n    pass")
        assert busy.timed_out and busy.restarted
        lost = kernel.run("x")
        assert lost.error is not None and "NameError" in lost.error and not lost.restarted
        assert kernel.run("y = 2\ny * 21").stdout == "42\n"
        assert kernel.generation == 2


def test_timeout_kills_every_process_in_the_sandbox(dirs: dict[str, Path]) -> None:
    token = sleep_token()
    with SandboxKernel(make_config(dirs, cell_timeout_s=1.5)) as kernel:
        spawn = (
            "import subprocess\n"
            f"subprocess.Popen(['/usr/bin/sleep', '{token}'])\n"
            f"subprocess.Popen(['/usr/bin/setsid', '/usr/bin/sleep', '{token}'])\n"
        )
        assert kernel.run(spawn).error is None
        assert wait_until(lambda: len(host_processes_with_arg(token)) == 2), "the scan must see the sandbox's processes"
        result = kernel.run("import time\ntime.sleep(60)")
        assert result.timed_out and result.restarted
        assert wait_until(lambda: not host_processes_with_arg(token)), host_processes_with_arg(token)


def test_kernel_crash_restarts_with_state_lost(kernel: SandboxKernel) -> None:
    kernel.run("x = 1")
    result = kernel.run("import os\nprint('bye', flush=True)\nos._exit(7)")
    assert result.restarted and not result.timed_out
    assert result.error is not None and "exit status 7" in result.error and "state lost" in result.error
    assert result.stdout == "bye\n"
    assert kernel.run("'x' in globals()").stdout == "False\n"


def test_kernel_that_died_between_cells_is_restarted_before_the_next_cell(kernel: SandboxKernel) -> None:
    kernel.run("x = 1")
    kernel.run(
        "import os, threading, time\n"
        "def _stop_later():\n"
        "    time.sleep(0.3)\n"
        "    os._exit(5)\n"
        "threading.Thread(target=_stop_later, daemon=True).start()"
    )
    time.sleep(1.0)
    result = kernel.run("'x' in globals()")
    assert result.restarted and result.stdout == "False\n"
    assert result.error is not None and "had stopped before this cell (exit status 5)" in result.error
    assert "state lost" in result.error


def test_bytes_forged_on_the_response_pipe_stop_and_restart_the_kernel(kernel: SandboxKernel) -> None:
    code = textwrap.dedent(
        """
        import os
        refused = []
        for name in os.listdir("/proc/self/fd"):
            fd = int(name)
            if fd <= 2:
                continue
            try:
                if os.readlink(f"/proc/self/fd/{fd}").startswith("pipe:"):
                    os.write(fd, b"garbage without a newline")
            except OSError as exc:
                refused.append((fd, exc.errno))
        refused
        """
    )
    result = kernel.run(code)
    assert result.restarted and not result.timed_out
    assert result.error is not None and "SandboxProtocolError" in result.error and "state lost" in result.error
    assert kernel.run("1 + 1").stdout == "2\n"


def test_memory_cap_raises_memory_error_and_the_kernel_survives(dirs: dict[str, Path]) -> None:
    with SandboxKernel(make_config(dirs, memory_bytes=2**30)) as kernel:
        assert kernel.run("import numpy as np\nkept = np.ones(10**6)").error is None
        array = kernel.run("big = np.ones(2**30)")  # 8 GiB against a 1 GiB address-space cap
        assert array.error is not None and "MemoryError" in array.error and not array.restarted
        raw = kernel.run("blob = bytearray(2**31)")
        assert raw.error is not None and "MemoryError" in raw.error and not raw.restarted
        assert kernel.run("float(kept.sum())").stdout == "1000000.0\n"


def test_returned_output_is_truncated_and_the_log_keeps_it(dirs: dict[str, Path]) -> None:
    config = make_config(dirs, max_output_chars=200)
    with SandboxKernel(config) as kernel:
        small = kernel.run("print('short')")
        result = kernel.run("import sys\nprint('A' * 5000)\nprint('B' * 3000, file=sys.stderr)")
        failure = kernel.run("raise ValueError('x' * 1000)")
    assert not small.truncated and small.stdout == "short\n"
    assert result.truncated and len(result.stdout) + len(result.stderr) <= 200
    assert result.stdout.startswith("A" * 20) and result.stdout.rstrip().endswith("A")
    assert "characters omitted" in result.stdout and "characters omitted" in result.stderr
    assert failure.truncated and failure.error is not None and len(failure.error) <= 200
    assert failure.error.startswith("Traceback") and failure.error.endswith("x")
    records = read_log(config.log_path)
    assert records[1]["stdout"] == "A" * 5000 + "\n" and records[1]["stderr"] == "B" * 3000 + "\n"
    assert records[1]["truncated"] is True and records[1]["log_truncated"] is False
    assert "x" * 1000 in str(records[2]["error"])


def test_the_log_bounds_runaway_output_and_records_its_true_size(dirs: dict[str, Path]) -> None:
    config = make_config(dirs, max_log_bytes=4096, max_output_chars=1000)
    with SandboxKernel(config) as kernel:
        result = kernel.run("import sys\nfor _ in range(2000):\n    sys.stdout.write('x' * 99 + '\\n')")
    record = read_log(config.log_path)[-1]
    assert record["stdout_bytes"] == 200_000 and record["log_truncated"] is True
    assert "bytes omitted from the log" in str(record["stdout"]) and len(str(record["stdout"])) <= 4096 + 64
    assert result.truncated and len(result.stdout) <= 1000


def test_payloads_larger_than_a_pipe_buffer_cross_in_both_directions(dirs: dict[str, Path]) -> None:
    config = make_config(dirs)
    code = "blob = '" + "a" * 2_000_000 + "'\nprint('z' * 300_000)\nblob[:5] + str(len(blob)) + 'b' * 500_000"
    with SandboxKernel(config) as kernel:
        result = kernel.run(code)
        assert kernel.run("len(blob)").stdout == "2000000\n"
    assert result.error is None and not result.restarted and result.truncated
    assert len(result.stdout) <= config.max_output_chars
    record = read_log(config.log_path)[0]
    assert record["code"] == code and record["stdout_bytes"] == 300_001
    assert record["stdout"] == "z" * 300_000 + "\n" + repr("aaaaa2000000" + "b" * 500_000) + "\n"


def test_cells_beyond_the_budget_are_refused_without_execution(dirs: dict[str, Path]) -> None:
    config = make_config(dirs, max_cells=2)
    with SandboxKernel(config) as kernel:
        kernel.run("a = 1")
        kernel.run("b = 2")
        assert kernel.cells_remaining == 0
        refused = kernel.run("open('/work/patches/refused_marker', 'w').write('x')")
        again = kernel.run("1")
    assert refused.refused and again.refused and refused.duration_s == 0.0 and refused.stdout == ""
    assert refused.error is not None and "budget of 2 Python cells is used up" in refused.error
    assert not (dirs["patches"] / "refused_marker").exists()
    records = read_log(config.log_path)
    assert [r["status"] for r in records] == ["ok", "ok", "refused", "refused"]
    assert [r["executed"] for r in records] == [True, True, False, False]
    assert records[2]["code"] == "open('/work/patches/refused_marker', 'w').write('x')"


def test_the_log_has_one_line_per_cell(dirs: dict[str, Path]) -> None:
    config = make_config(dirs, max_cells=4, cell_timeout_s=1.5)
    codes = ["x = 1", "1 / 0", "import time\ntime.sleep(30)", "print('fine')", "'refused'"]
    with SandboxKernel(config) as kernel:
        results = [kernel.run(code) for code in codes]
    records = read_log(config.log_path)
    assert len(records) == len(codes)
    assert [r["index"] for r in records] == [r.index for r in results] == list(range(len(codes)))
    assert [r["status"] for r in records] == ["ok", "error", "timeout", "ok", "refused"]
    assert [r["code"] for r in records] == codes
    required = {"timestamp", "index", "code", "stdout", "stderr", "error", "duration_s", "timed_out", "restarted"}
    assert all(required <= set(record) for record in records)
    assert records[2]["timed_out"] is True and records[2]["restarted"] is True
    assert [r["kernel_generation"] for r in records] == [0, 0, 0, 1, 1]
    assert records[3]["stdout"] == "fine\n" and "ZeroDivisionError" in str(records[1]["error"])
    assert not config.log_path.with_name(config.log_path.name + ".inflight").exists()


def test_cell_counters_resume_from_an_existing_log(dirs: dict[str, Path]) -> None:
    config = make_config(dirs, max_cells=3)
    with SandboxKernel(config) as kernel:
        kernel.run("a = 1")
        kernel.run("b = 2")
    with SandboxKernel(config) as resumed:
        assert (resumed.cells_used, resumed.cells_logged, resumed.cells_remaining) == (2, 2, 1)
        assert resumed.run("'third'").index == 2
        assert resumed.run("'fourth'").refused
    assert [r["index"] for r in read_log(config.log_path)] == [0, 1, 2, 3]


def test_a_malformed_log_is_an_error(dirs: dict[str, Path]) -> None:
    config = make_config(dirs)
    config.log_path.write_text("not json\n")
    with pytest.raises(SandboxError, match="not valid JSON"):
        SandboxKernel(config)


_HOST_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    from weightcode.sandbox import SandboxConfig, SandboxKernel
    fields = {key: Path(value) for key, value in json.loads(sys.argv[1]).items()}
    kernel = SandboxKernel(SandboxConfig(**fields, cell_timeout_s=120.0))
    kernel.start()
    kernel.run("x = 1")
    print("READY", flush=True)
    kernel.run("import subprocess, time\\nsubprocess.Popen(['/usr/bin/sleep', %r])\\ntime.sleep(600)" % sys.argv[2])
    """
)


def test_a_cell_interrupted_by_host_death_is_logged_and_the_sandbox_dies(dirs: dict[str, Path]) -> None:
    config = make_config(dirs, cell_timeout_s=120.0)
    fields = {name: str(getattr(config, name)) for name in ("trial_dir", "patches_dir", "scratch_dir", "log_path", "python")}
    token = sleep_token()
    inflight = config.log_path.with_name(config.log_path.name + ".inflight")
    host = subprocess.Popen(
        [sys.executable, "-c", _HOST_SCRIPT, json.dumps(fields), token],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert host.stdout is not None and host.stdout.readline().strip() == "READY"
        assert wait_until(lambda: host_processes_with_arg(token)), "the interrupted cell never started"
        assert inflight.exists()
        host.kill()
        host.wait(timeout=10)
    finally:
        if host.poll() is None:
            host.kill()
            host.wait(timeout=10)
    assert wait_until(lambda: not host_processes_with_arg(token)), "the sandbox outlived its host process"
    resumed = SandboxKernel(config)
    records = read_log(config.log_path)
    assert [r["status"] for r in records] == ["ok", "interrupted"]
    assert token in str(records[1]["code"]) and records[1]["executed"] is True
    assert not inflight.exists() and resumed.cells_used == 2
    resumed.close()


def test_configuration_rejects_unsafe_layouts(dirs: dict[str, Path]) -> None:
    with pytest.raises(SandboxError, match="log_path"):
        SandboxKernel(make_config(dirs, log_path=dirs["patches"] / "cells.jsonl"))
    nested = dirs["patches"] / "trial"
    nested.mkdir()
    with pytest.raises(SandboxError, match="overlap"):
        SandboxKernel(make_config(dirs, trial_dir=nested))
    with pytest.raises(SandboxError, match="virtual environment"):
        SandboxKernel(make_config(dirs, python=Path("/usr/bin/python3")))
    with pytest.raises(ValueError, match="cell_timeout_s"):
        make_config(dirs, cell_timeout_s=0)


def test_close_is_idempotent_and_run_needs_a_started_open_kernel(dirs: dict[str, Path]) -> None:
    config = make_config(dirs)
    kernel = SandboxKernel(config)
    with pytest.raises(SandboxError, match="start"):
        kernel.run("1")
    kernel.start()
    assert kernel.running and kernel.ready_info["protocol"] == "weightcode.kernel.v0"
    kernel.close()
    kernel.close()
    assert not kernel.running
    with pytest.raises(SandboxError, match="closed"):
        kernel.run("1")
    assert not config.log_path.exists()  # calls that never reached a kernel are programming errors, not cells
