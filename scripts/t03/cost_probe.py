"""PREREG T03 cost probe: one forward.py readout in a fresh process that imports numpy only, as a program does.

    python scripts/t03/cost_probe.py --forward F --params P --inputs X.npy --labels Y.npy --n N --out OUT.json

Loads forward.py (executes exactly its bytes), the params npz and the first N preprocessed images, runs forward once
over them and the argmax check against the labels, and writes the timings and the process's memory peaks: VmHWM
(peak resident set of this process image) and VmPeak (peak address space, what RLIMIT_AS, the sandbox's memory limit,
bounds) from /proc/self/status. ru_maxrss is written too but is not this process's own peak: Linux carries the
parent's peak RSS across fork and exec into it. The thread variables are inherited from the caller and recorded.
Nothing here imports torch or weightcode.
"""

from __future__ import annotations

import time

STARTED = time.perf_counter()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import resource  # noqa: E402
from pathlib import Path  # noqa: E402
from types import ModuleType  # noqa: E402

import numpy as np  # noqa: E402

IMPORTED = time.perf_counter()
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def proc_status() -> dict[str, int]:
    """VmPeak, VmHWM and VmRSS of this process in bytes (Linux)."""
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("VmPeak", "VmHWM", "VmRSS"):
            amount, unit = rest.split()
            if unit != "kB":
                raise ValueError(f"unexpected unit {unit!r} in /proc/self/status")
            values[key] = int(amount) * 1024
    return values


def read_first_rows(path: Path, n: int) -> np.ndarray:
    """The first ``n`` rows of a C-order .npy file, read into a fresh array (no memory map, so the peak RSS holds the
    n rows only)."""
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"{path}: .npy format version {version} is not read here")
        if fortran or len(shape) < 1 or not 1 <= n <= shape[0]:
            raise ValueError(f"{path}: shape {shape} (fortran {fortran}); cannot read {n} rows")
        per_row = int(np.prod(shape[1:], dtype=np.int64))
        array = np.fromfile(handle, dtype=dtype, count=n * per_row)
    if array.size != n * per_row:
        raise ValueError(f"{path}: read {array.size} values, expected {n * per_row}")
    return array.reshape((n, *shape[1:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    t0 = time.perf_counter()
    source = args.forward.read_bytes()
    module = ModuleType("t03_probe_forward")
    exec(compile(source, str(args.forward), "exec"), module.__dict__)
    with np.load(args.params, allow_pickle=False) as archive:
        params = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
    t1 = time.perf_counter()
    x = read_first_rows(args.inputs, args.n)
    y = read_first_rows(args.labels, args.n)
    t2 = time.perf_counter()
    logits = module.forward(params, x)
    t3 = time.perf_counter()
    correct = int(np.count_nonzero(np.asarray(logits).argmax(1) == y))
    t4 = time.perf_counter()
    status = proc_status()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    record = {
        "n_images": int(args.n),
        "forward_sha256": hashlib.sha256(source).hexdigest(),
        "threads": {name: os.environ.get(name) for name in THREAD_VARS},
        "numpy": np.__version__,
        "import_s": IMPORTED - STARTED,
        "load_forward_and_params_s": t1 - t0,
        "load_inputs_s": t2 - t1,
        "forward_s": t3 - t2,
        "check_s": t4 - t3,
        "forward_s_per_image": (t3 - t2) / args.n,
        "process_s": t4 - STARTED,
        "n_correct": correct,
        "logits_sha256": hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest(),
        "ru_maxrss_bytes": int(usage.ru_maxrss) * 1024,
        "vm_peak_bytes": status["VmPeak"],
        "vm_hwm_bytes": status["VmHWM"],
        "user_cpu_s": usage.ru_utime,
        "system_cpu_s": usage.ru_stime,
    }
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
