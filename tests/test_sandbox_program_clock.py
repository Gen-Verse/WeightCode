"""The virtual clock of update-program runs (``virtual-v1``, ``weightcode.sandbox.program_driver``).

Units: every replaced clock advances one shared counter by one step per read, ``time.sleep`` is left alone, an unknown
clock is refused, and the host passes and records the clock. Runs on the learn toy store: a program that stops a loop
on its own elapsed time gives one iteration count and one deltas file on repeated runs and under CPU contention; a
program that polls its virtual budget forever is still stopped by the host's real wall limit; programs that never read
the clock give the deltas.npz bytes that origin/main's driver gave before the virtual clock existed.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
from test_harness_fixtures import needs_sandbox
from test_harness_learn import (  # noqa: F401
    BIG,
    FIX,
    STRINGS,
    WIDE,
    WORSE,
    LearnStore,
    _assembled,
    _sandbox,
    _student,
    _write_program,
    learn_store,
)

from weightcode.harness.program import ProgramProblem, Update, run_update
from weightcode.sandbox import program_driver as driver
from weightcode.sandbox.program import PROGRAM_CLOCK, ProgramSandbox, build_program_argv

COMPUTE = '''HYPOTHESIS = "a data-dependent update"
PREDICTED_EFFECT = "deterministic deltas"
import numpy as np

def update(student, experience):
    params = student["params"]
    w = params["fc1.weight"].astype(np.float64)
    x = experience["x_new"].reshape(len(experience["x_new"]), -1).astype(np.float64)[:, : w.shape[1]] / 255.0
    logits = np.asarray(student["forward"](params, experience["x_new"]), dtype=np.float64)
    g = np.tanh(x @ w.T) * logits.std()
    print("rows", x.shape, logits.shape)
    return {"fc1.weight": 1e-3 * (g.T @ x) / len(x), "fc2.bias": 1e-4 * logits.mean(axis=0)}
'''
# sha256 of deltas.npz of these programs on the learn store's seed-100 trial, computed with origin/main 25d5983's
# program driver (real clocks) before the virtual clock existed.
BEFORE_DELTAS_SHA256 = {
    "fix.py": ("8bca91a6d6f304cebb324b4f170a85614b8759967a2259c4ac6a1d9b4bcfb15c", FIX),
    "wide.py": ("b01bf0763741eee377bec7d9809726d0fc8a186dd5f2becd50e0e6bf88dedcbc", WIDE),
    "big.py": ("eb4de50295d8c7a18af9451ff11dd88a687503bfaa7018069954e394f5dc9ec1", BIG),
    "worse.py": ("efa828235c5314e223e3d532ab1f5139b17bea99b0b0cd4b9b6be30aabdb44a0", WORSE),
    "compute.py": ("74dd3f439702963fc187ea3089dc6c8b2c399d2cc03c9b6e5ab5341f3e0eeb1c", COMPUTE),
}
BUDGET = STRINGS + '''import time
from time import perf_counter
import numpy as np

def update(student, experience):
    t0, n = time.time(), 0
    while time.time() - t0 <= 1.0:  # a 1 s budget on the wall clock, as the E10 final's 220 s one
        n += 1
    p0, m = perf_counter(), 0
    while perf_counter() - p0 <= 0.01:  # a clock bound at import
        m += 1
    print("iterations", n, m)
    delta = np.zeros_like(student["params"]["fc2.bias"])
    delta[0], delta[1] = n * 1e-9, m * 1e-9
    return {"fc2.bias": delta}
'''
POLL_FOREVER = STRINGS + '''import time

def update(student, experience):
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1e6:
        pass
    return {}
'''
CLOCK_NAMES = ("time", "time_ns", "perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns", "process_time",
               "process_time_ns", "thread_time", "thread_time_ns", "clock_gettime", "clock_gettime_ns")


# ---------------------------------------------------------------------------------------------------------------
# Units


def _fake_time() -> types.SimpleNamespace:
    def real() -> float:
        raise AssertionError("a real clock was read")

    fake = types.SimpleNamespace(**{name: real for name in CLOCK_NAMES})
    fake.CLOCK_REALTIME, fake.CLOCK_MONOTONIC, fake.sleep = time.CLOCK_REALTIME, time.CLOCK_MONOTONIC, time.sleep
    return fake


def test_every_clock_reads_one_counter_that_advances_one_step_per_read() -> None:
    fake = _fake_time()
    clock = driver.install_clock(driver.CLOCK_VIRTUAL_V1, fake)
    step = driver.CLOCK_STEP_NS
    assert step == 1_000 and clock.reads == 0 and fake.sleep is time.sleep
    assert fake.time() == (driver.WALL_START_NS + step) / 1e9
    assert fake.time_ns() == driver.WALL_START_NS + 2 * step
    assert fake.perf_counter() == (driver.MONOTONIC_START_NS + 3 * step) / 1e9
    assert fake.perf_counter_ns() == driver.MONOTONIC_START_NS + 4 * step
    assert fake.monotonic() == (driver.MONOTONIC_START_NS + 5 * step) / 1e9
    assert fake.monotonic_ns() == driver.MONOTONIC_START_NS + 6 * step
    assert fake.process_time() == (driver.CPU_START_NS + 7 * step) / 1e9
    assert fake.process_time_ns() == driver.CPU_START_NS + 8 * step
    assert fake.thread_time() == (driver.CPU_START_NS + 9 * step) / 1e9
    assert fake.thread_time_ns() == driver.CPU_START_NS + 10 * step
    assert fake.clock_gettime(time.CLOCK_REALTIME) == (driver.WALL_START_NS + 11 * step) / 1e9
    assert fake.clock_gettime_ns(time.CLOCK_MONOTONIC) == driver.MONOTONIC_START_NS + 12 * step
    assert clock.reads == 12
    t0 = fake.time()
    assert [round((fake.time() - t0) * 1e6) for _ in range(3)] == [1, 2, 3]  # 1 us per read
    again = driver.install_clock(driver.CLOCK_VIRTUAL_V1, _fake_time())
    assert again.reads == 0  # each run starts from the same value
    with pytest.raises(ValueError, match="unknown clock"):
        driver.install_clock("real", _fake_time())
    with pytest.raises(ValueError, match="positive"):
        driver.VirtualClock(0)


def test_the_host_passes_and_records_the_clock(tmp_path: Path) -> None:
    assert PROGRAM_CLOCK == driver.CLOCK_VIRTUAL_V1 and driver.CLOCKS == (PROGRAM_CLOCK,)
    sandbox = ProgramSandbox(python=Path(sys.executable), memory_bytes=2**30, threads=1)
    argv = build_program_argv(sandbox, tmp_path, tmp_path, "p.py")
    assert argv[-4:] == ["--program", "p.py", "--clock", PROGRAM_CLOCK]
    with pytest.raises(SystemExit):  # refused by argparse, before any clock is replaced
        driver.main(["--program", "p.py", "--clock", "real"])
    assert isinstance(time.time, types.BuiltinFunctionType)  # this process keeps its real clocks


# ---------------------------------------------------------------------------------------------------------------
# Runs


def _run(store: LearnStore, tmp_path: Path, name: str, source: str, label: str, **kwargs: float) -> Update:
    trial = tmp_path / "session100" / "trial"
    if not trial.exists():
        _assembled(store, tmp_path)
    program = tmp_path / name
    if not program.exists():
        _write_program(tmp_path, name, source)
    return run_update(_student(trial), trial / "experience.npz", program, tmp_path / f"run_{label}", _sandbox(),
                      **kwargs)


def _counts(update: Update) -> tuple[int, int]:
    found = re.search(r"iterations (\d+) (\d+)", update.record["shown"]["program_stdout"])
    assert found is not None, update.record["shown"]
    return int(found.group(1)), int(found.group(2))


def _contended(store: LearnStore, tmp_path: Path, label: str) -> tuple[Update, bool]:
    """A BUDGET run pinned to one CPU that three busy processes share with it (the run inherits the affinity)."""
    before = os.sched_getaffinity(0)
    cpu = max(before)
    busy: list[subprocess.Popen[bytes]] = []
    try:
        os.sched_setaffinity(0, {cpu})
        busy = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(3)]
        update = _run(store, tmp_path, "budget.py", BUDGET, label)
        loaded = all(proc.poll() is None for proc in busy)
    finally:
        for proc in busy:
            proc.send_signal(signal.SIGKILL)
            proc.wait()
        os.sched_setaffinity(0, before)
    return update, loaded


@needs_sandbox
def test_a_program_that_stops_on_its_own_elapsed_time_gives_one_count_under_any_load(
        learn_store: LearnStore, tmp_path: Path) -> None:  # noqa: F811
    first = _run(learn_store, tmp_path, "budget.py", BUDGET, "1")
    second = _run(learn_store, tmp_path, "budget.py", BUDGET, "2")
    loaded, was_loaded = _contended(learn_store, tmp_path, "loaded")
    assert was_loaded
    n, m = _counts(first)
    assert _counts(second) == _counts(loaded) == (n, m)
    assert 999_000 <= n <= 1_001_000 and 9_000 <= m <= 11_000  # the budget runs out after 1e6 reads of 1 us each
    assert first.record["deltas_sha256"] == second.record["deltas_sha256"] == loaded.record["deltas_sha256"]
    for update in (first, second, loaded):
        result = update.record["run"]["result"]
        assert update.record["clock"] == PROGRAM_CLOCK == result["clock"]
        assert result["clock_reads"] == first.record["run"]["result"]["clock_reads"] >= n + m
    real = [update.record["run"]["result"]["update_s"] for update in (first, second, loaded)]
    assert real[2] > 1.5 * min(real[:2]), real  # the contention slowed the loaded run in real time
    print("update_s (real):", real, "reads:", first.record["run"]["result"]["clock_reads"])


@needs_sandbox
def test_a_program_polling_its_virtual_budget_is_still_stopped_by_the_wall_limit(
        learn_store: LearnStore, tmp_path: Path) -> None:  # noqa: F811
    began = time.monotonic()
    with pytest.raises(ProgramProblem) as caught:
        _run(learn_store, tmp_path, "poll.py", POLL_FOREVER, "poll", wall_s=2.0)
    assert caught.value.messages == ["the program did not return within 2 seconds and was stopped"]
    assert caught.value.run is not None and caught.value.run["run"]["status"] == "timeout"
    assert caught.value.run["clock"] == PROGRAM_CLOCK and time.monotonic() - began < 30.0


@needs_sandbox
def test_programs_that_do_not_read_the_clock_give_the_deltas_they_gave_before(
        learn_store: LearnStore, tmp_path: Path) -> None:  # noqa: F811
    for name, (before, source) in BEFORE_DELTAS_SHA256.items():
        update = _run(learn_store, tmp_path, name, source, name)
        assert update.record["deltas_sha256"] == before, name
        assert update.record["clock"] == PROGRAM_CLOCK and update.record["run"]["result"]["clock_reads"] == 0, name
