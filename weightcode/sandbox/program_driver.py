"""In-sandbox driver of one update program (``weightcode.sandbox.program``), bound read-only at /opt/weightcode/program_driver.py.

Run as ``python -I -u /opt/weightcode/program_driver.py --program <name> --clock virtual-v1`` with /work/in read-only and
/work/out writable. It reads /work/in/params.npz, experience.npz, layout.json, forward.py and the program ``<name>``;
it calls ``update({"params": ..., "forward": ..., "layout": ...}, experience)`` once and writes the returned arrays to
/work/out/deltas.npz (one stored .npy member per key, ``allow_pickle=False``) and a JSON record to
/work/out/result.json. It uses numpy and the standard library only.

Clock ``virtual-v1``: before forward.py and the program are loaded, the driver replaces the clocks of the ``time``
module (``time``, ``perf_counter``, ``monotonic``, ``process_time``, ``thread_time``, ``clock_gettime``, each with its
``_ns`` twin) by one virtual clock: a counter that starts at a fixed value and advances by ``CLOCK_STEP_NS`` (1 us)
on every read, whatever the machine does. A program's output then depends on its inputs only, also when it stops a
loop on its own elapsed time (the E10 gaussian B final ``v8_alloc_branch_n48_x155.py`` broke its unit sweep at 220 s
of wall time, so its deltas depended on machine load).

The step makes such a budget effectively never fire: a budget of B seconds needs B * 10**6 reads (220 s: 2.2e8 reads;
that program reads the clock once per unit, 688 units). Even a loop that does nothing but poll the clock (about
0.16 us of real time per read in the sandbox) reaches 220 s after about 35 s of real time, at the same read under any
load, well inside the host's 300 s wall limit; a 1 ns step would push such a loop past it. The host's wall limit
(``weightcode.sandbox.program.PROGRAM_WALL_S``, real time, outside the sandbox) is unchanged, so a program that never
returns is still stopped. Not virtual: ``time.sleep`` (it blocks in real time and does not advance the clock),
``time.localtime``, ``gmtime``, ``ctime`` and ``strftime`` without a time argument, ``datetime`` and ``os.times``;
``threading``, loaded before the clock is replaced, keeps real time for its timeouts, while a module the program
imports later binds the virtual clock. Threads of the program share one counter, so the values they read depend on
the order of their reads. The driver times ``update`` with the real monotonic clock (``update_s``); ``result.json``
reports the clock and the number of virtual reads (``clock_reads``).

The program runs in this process and can change anything here, so the host treats both outputs as untrusted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import traceback
import types
import zipfile
from typing import Any

import numpy as np

IN_DIR = "/work/in"
OUT_DIR = "/work/out"
MAX_TEXT_CHARS = 20_000
MODULE_STRINGS = ("HYPOTHESIS", "PREDICTED_EFFECT")
CLOCK_VIRTUAL_V1 = "virtual-v1"
CLOCKS = (CLOCK_VIRTUAL_V1,)
CLOCK_STEP_NS = 1_000  # 1 us per read: see the module docstring for the choice
WALL_START_NS = 1_700_000_000 * 10**9  # time.time() and CLOCK_REALTIME start at 2023-11-14 22:13:20 UTC
MONOTONIC_START_NS = 1_000 * 10**9  # perf_counter, monotonic and the other clock_gettime clocks start at 1000 s
CPU_START_NS = 10**9  # process_time and thread_time start at 1 s
_REAL_MONOTONIC = time.monotonic  # the driver's own timing of update(), bound before the clock is replaced


class VirtualClock:
    """The ``virtual-v1`` clock: every read of any replaced clock advances one shared counter by ``step_ns``."""

    def __init__(self, step_ns: int = CLOCK_STEP_NS) -> None:
        if step_ns <= 0:
            raise ValueError("the clock step must be positive")
        self.step_ns = step_ns
        self._reads = itertools.count(1)
        self.reads = 0

    def _tick(self, start_ns: int) -> int:
        self.reads = next(self._reads)
        return start_ns + self.reads * self.step_ns

    def functions(self, realtime_id: int | None) -> dict[str, Any]:
        """The replacements, by ``time`` attribute name. ``realtime_id``: ``time.CLOCK_REALTIME`` (None if absent)."""

        def wall_ns() -> int:
            return self._tick(WALL_START_NS)

        def monotonic_ns() -> int:
            return self._tick(MONOTONIC_START_NS)

        def cpu_ns() -> int:
            return self._tick(CPU_START_NS)

        def clock_gettime_ns(clk_id: int) -> int:
            return wall_ns() if realtime_id is not None and clk_id == realtime_id else monotonic_ns()

        def seconds(read: Any) -> Any:
            def clock(*args: Any) -> float:
                return read(*args) / 1_000_000_000
            return clock

        return {"time": seconds(wall_ns), "time_ns": wall_ns, "perf_counter": seconds(monotonic_ns),
                "perf_counter_ns": monotonic_ns, "monotonic": seconds(monotonic_ns), "monotonic_ns": monotonic_ns,
                "process_time": seconds(cpu_ns), "process_time_ns": cpu_ns, "thread_time": seconds(cpu_ns),
                "thread_time_ns": cpu_ns, "clock_gettime": seconds(clock_gettime_ns),
                "clock_gettime_ns": clock_gettime_ns}


def install_clock(name: str, module: Any = time) -> VirtualClock:
    """Replace the clocks of ``module`` (the ``time`` module) by the clock ``name``; returns it."""
    if name not in CLOCKS:
        raise ValueError(f"unknown clock {name!r}; this driver has {CLOCKS}")
    clock = VirtualClock()
    for attribute, function in clock.functions(getattr(module, "CLOCK_REALTIME", None)).items():
        if hasattr(module, attribute):  # thread_time and clock_gettime exist on every Linux build
            setattr(module, attribute, function)
    return clock


def _write_result(record: dict[str, Any]) -> None:
    with open(f"{OUT_DIR}/result.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle)


def _load_npz(name: str) -> dict[str, np.ndarray]:
    with np.load(f"{IN_DIR}/{name}", allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def _module(path: str, name: str) -> types.ModuleType:
    with open(path, "rb") as handle:
        source = handle.read()
    module = types.ModuleType(name)
    module.__file__ = path
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def _save_deltas(arrays: dict[str, np.ndarray]) -> None:
    """``numpy.savez`` without keyword arguments, so no key can collide with its parameters."""
    with zipfile.ZipFile(f"{OUT_DIR}/deltas.npz", "w", zipfile.ZIP_STORED) as archive:
        for key, value in arrays.items():
            with archive.open(key + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, value, allow_pickle=False)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_TEXT_CHARS else text[: MAX_TEXT_CHARS // 2] + "\n[...]\n" + text[-MAX_TEXT_CHARS // 2 :]


def _output(deltas: object) -> dict[str, np.ndarray]:
    if not isinstance(deltas, dict):
        raise TypeError(f"update returned {type(deltas).__name__}; it must return a dict of tensor name -> array")
    arrays: dict[str, np.ndarray] = {}
    for key, value in deltas.items():
        if not isinstance(key, str):
            raise TypeError(f"update returned a key of type {type(key).__name__}; keys must be tensor names (str)")
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise TypeError(f"update returned {key!r} as an array of dtype {array.dtype}; deltas must hold real numbers")
        arrays[key] = array
    return arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--clock", required=True, choices=CLOCKS)
    args = parser.parse_args(argv)
    clock = install_clock(args.clock)  # before forward.py and the program can bind any clock
    stage = "load"
    try:
        params = _load_npz("params.npz")
        experience = _load_npz("experience.npz")
        with open(f"{IN_DIR}/layout.json", encoding="utf-8") as handle:
            layout = json.load(handle)
        forward = getattr(_module(f"{IN_DIR}/forward.py", "weightcode_student_forward"), "forward")
        stage = "import"
        program = _module(f"{IN_DIR}/{args.program}", "weightcode_update_program")
        update = getattr(program, "update", None)
        if not callable(update):
            raise TypeError("the program defines no callable update(student, experience)")
        strings = {name.lower(): getattr(program, name, None) for name in MODULE_STRINGS}
        stage = "update"
        began = _REAL_MONOTONIC()
        deltas = update({"params": params, "forward": forward, "layout": layout}, experience)
        update_s = _REAL_MONOTONIC() - began
        stage = "output"
        arrays = _output(deltas)
        _save_deltas(arrays)
        _write_result({"status": "ok", "update_s": update_s, "keys": list(arrays), "clock": args.clock,
                       "clock_reads": clock.reads,
                       **{name: (_clip(value) if isinstance(value, str) else None) for name, value in strings.items()},
                       **{f"{name}_type": type(value).__name__ for name, value in strings.items()}})
    except BaseException:  # reported to the host as the program's error; the host validates every output anyway
        _write_result({"status": "error", "stage": stage, "error": _clip(traceback.format_exc()), "clock": args.clock,
                       "clock_reads": clock.reads})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
