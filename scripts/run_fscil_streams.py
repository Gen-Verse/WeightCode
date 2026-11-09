"""Run the PREREG E9 reference streams in lockstep (``weightcode.baselines.fit.fscilrefs.run_streams``) over a process pool.

    .venv/bin/python scripts/run_fscil_streams.py --out DIR --validity FILE --sessions 2 [3 ...] \\
        [--workers 12] [--fit-threads 2]

R comes from the validity record (V3.R), whose sha256 is recorded. The fits of the streams of a session run in
parallel, one task per stream with ``--fit-threads`` torch threads; every check and readout maps forward.py over the
pool in the oracle's own 256-row batches. Resuming (sessions starting after the last written one) needs the same
inputs, R and fit threads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_DATA: Any = None


def _init(fit_threads: int) -> None:
    global _DATA
    import torch

    torch.set_num_threads(fit_threads)
    from weightcode.baselines.fit import fscilrefs as FR

    _DATA = FR.load_stream_data(need_allclass=False)


def _fit(method: str, params: dict, session: int) -> dict:
    from weightcode.baselines.fit import fscilrefs as FR

    return FR.fit_stream(method, params, FR.session_rows(_DATA, session), _DATA)


def _logits(params: dict, x: np.ndarray) -> np.ndarray:
    return np.asarray(_DATA.forward(params, x), dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validity", type=Path, required=True)
    parser.add_argument("--sessions", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--fit-threads", type=int, default=2)
    parser.add_argument("--code-commit", default=None, help="the commit the code copy was checked out at")
    args = parser.parse_args(argv)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"  # forward.py chunks run one per worker; torch threads are set per worker
    from weightcode.baselines.fit import fscilrefs as FR

    validity_bytes = args.validity.read_bytes()
    validity = json.loads(validity_bytes)
    r_points = float(validity["V3"]["R"])
    data = FR.load_stream_data(need_allclass=False)
    if validity["base"]["params_sha256"] != data.base_row["params_sha256"]:
        raise SystemExit("the validity record is of another base student")
    provenance = {"validity": {"path": str(args.validity), "sha256": hashlib.sha256(validity_bytes).hexdigest(),
                               "R": r_points},
                  "fit_threads": args.fit_threads, "code_commit": args.code_commit,
                  "host": socket.gethostname(), "argv": sys.argv}

    def log(message: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)

    log(f"R {r_points}; base {data.base_row['model_id']}; sessions {args.sessions}; workers {args.workers}")
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.fit_threads,)) as pool:
        forward = FR.ChunkedForward(data.forward, pool_map=pool.starmap, chunk_fn=_logits)
        floors = FR.run_streams(data, r_points=r_points, out_dir=args.out, sessions=args.sessions,
                                fit_many=lambda tasks: pool.starmap(_fit, tasks), forward=forward,
                                provenance=provenance, log=log)
    print(json.dumps({str(t): v for t, v in floors.items()}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
