"""``UnisolatedRunner``: the update-program driver without bubblewrap (``weightcode.sandbox.program``).

It exists to replay published, hash-checked programs on machines without a suitable bubblewrap
(``weightcode reproduce``). It must behave like the sandbox in everything but isolation: on the learn toy store the
programs of test_sandbox_program_clock give the deltas.npz bytes that the sandboxed driver gave, the virtual clock is
installed, the wall limit still stops a program, and the launcher leaves the driver file unchanged. None of these tests
needs bubblewrap.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from test_harness_learn import LearnStore, _assembled, _student, _write_program, learn_store  # noqa: F401
from test_sandbox_program_clock import BEFORE_DELTAS_SHA256, BUDGET, POLL_FOREVER

from weightcode.harness.program import ProgramProblem, Update, run_update
from weightcode.sandbox.program import (
    DRIVER_SCRIPT,
    ISOLATION_NONE,
    LAUNCHER_SCRIPT,
    PROGRAM_CLOCK,
    UnisolatedRunner,
    build_unisolated_argv,
)


def _runner() -> UnisolatedRunner:
    return UnisolatedRunner(python=Path(sys.executable), memory_bytes=4 * 2**30, threads=1)


def _run(store: LearnStore, tmp_path: Path, name: str, source: str, label: str, **kwargs: float) -> Update:
    trial = tmp_path / "session100" / "trial"
    if not trial.exists():
        _assembled(store, tmp_path)
    program = tmp_path / name
    if not program.exists():
        _write_program(tmp_path, name, source)
    return run_update(_student(trial), trial / "experience.npz", program, tmp_path / f"run_{label}", _runner(),
                      **kwargs)


def test_programs_give_the_deltas_of_the_sandboxed_driver(learn_store: LearnStore, tmp_path: Path) -> None:  # noqa: F811
    for name, (before, source) in BEFORE_DELTAS_SHA256.items():
        update = _run(learn_store, tmp_path, name, source, name)
        assert update.record["deltas_sha256"] == before, name
        assert update.record["sandbox"]["isolation"] == ISOLATION_NONE
        assert update.record["run"]["result"]["clock_reads"] == 0, name


def test_the_virtual_clock_is_installed(learn_store: LearnStore, tmp_path: Path) -> None:  # noqa: F811
    first = _run(learn_store, tmp_path, "budget.py", BUDGET, "1")
    second = _run(learn_store, tmp_path, "budget.py", BUDGET, "2")
    assert first.record["deltas_sha256"] == second.record["deltas_sha256"]
    for update in (first, second):
        result = update.record["run"]["result"]
        assert update.record["clock"] == PROGRAM_CLOCK == result["clock"]
        assert result["clock_reads"] > 1_000_000  # the 1 s budget ran out on virtual reads, not on real time


def test_the_wall_limit_stops_a_program(learn_store: LearnStore, tmp_path: Path) -> None:  # noqa: F811
    began = time.monotonic()
    with pytest.raises(ProgramProblem) as caught:
        _run(learn_store, tmp_path, "poll.py", POLL_FOREVER, "poll", wall_s=2.0)
    assert caught.value.messages == ["the program did not return within 2 seconds and was stopped"]
    assert caught.value.run is not None and caught.value.run["run"]["status"] == "timeout"
    assert time.monotonic() - began < 30.0


def test_the_command_line_runs_the_unchanged_driver_without_bubblewrap(tmp_path: Path) -> None:
    argv = build_unisolated_argv(_runner(), tmp_path / "in", tmp_path / "out", "p.py")
    assert "bwrap" not in " ".join(argv)
    assert argv[:4] == [sys.executable, "-I", "-u", str(LAUNCHER_SCRIPT)]
    assert argv[argv.index("--driver") + 1] == str(DRIVER_SCRIPT)
    assert argv[-4:] == ["--program", "p.py", "--clock", PROGRAM_CLOCK]
