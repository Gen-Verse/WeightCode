"""PREREG E7 memory peaks of a graphnew instance, one measurement per fresh process (Order step 6).

``--what forward``: one forward.py call over every node of the graph (load graph.npz, A X, three layers).
``--what oracle``: build the graphnew joint oracle (V_check cut by class) and spend one check on a child (the base
with its head bias shifted, so the check is a real evaluation, never a recorded query of any run).

Prints one JSON line with the peak RSS (``getrusage`` ru_maxrss of this process) and the wall time of each phase.
Run each mode in its own process, inside a capped scope and under ``/usr/bin/time -v``::

    systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 /usr/bin/time -v \\
        .venv/bin/python scripts/dev/e7_memory_peaks.py --instance g06de3fa0 --what forward

The sandboxed program run of a LEARN session is the harness's to measure (weightcode.harness), not this script's.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", required=True)
    parser.add_argument("--what", choices=("forward", "oracle"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(args.threads)
    sys.dont_write_bytecode = True
    import numpy as np

    from weightcode.evalkit import graphnew as GN

    phases: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    instance = GN.load_graphnew_instance(args.instance)
    phases["load_instance"] = {"s": time.perf_counter() - started, "peak_rss_mib": _rss_mib()}
    if args.what == "forward":
        t0 = time.perf_counter()
        logits = instance.forward(dict(instance.params), np.arange(int(instance.private["n_nodes"]), dtype=np.int64))
        phases["forward_all_nodes"] = {"s": time.perf_counter() - t0, "peak_rss_mib": _rss_mib(),
                                       "logits_shape": list(np.shape(logits))}
    else:
        t0 = time.perf_counter()
        oracle = GN.make_graphnew_oracle(instance)
        phases["build_oracle"] = {"s": time.perf_counter() - t0, "peak_rss_mib": _rss_mib()}
        child = dict(instance.params)
        bias = instance.params["conv3.bias"].copy()
        bias[[int(c) for c in instance.private["new_classes"]]] += np.float32(1.0)
        child["conv3.bias"] = bias
        t0 = time.perf_counter()
        oracle.query(child, "memory_probe")  # its score is not read: a probe informs no decision
        phases["one_check"] = {"s": time.perf_counter() - t0, "peak_rss_mib": _rss_mib()}
    print(json.dumps({"instance_id": instance.instance_id, "what": args.what, "threads": args.threads,
                      "wall_s": time.perf_counter() - started, "peak_rss_mib": _rss_mib(), "phases": phases}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
