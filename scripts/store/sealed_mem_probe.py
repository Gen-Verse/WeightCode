"""Memory and time of one vision_joint entry read at the sealed test's row count, on synthetic rows (no sealed file).

PREREG Sealed read, list v2: the sealed read runs ``--workers`` entries at a time on one machine, so each worker's peak
memory and time per entry at the sealed row count decide how many run at once. This probe takes an entry of a list,
recovers its child as the read does (``sealed_finals.load_entry``), takes the entry's dev rows of record
(``Family.dev_rows``) and tiles every array of each split along its rows to ``--rows`` rows (the sealed test has
10,000 condition rows and 10,000 clean rows for cifar10_vl7, cifar10_r18 and cifar10_r20), then calls the family's
readout on them as the read does (``Family.read``). It writes one JSON record: the arrays' shapes, dtypes and bytes,
the process's peak resident set (``getrusage`` ru_maxrss) after imports, after loading and after the readout, and the
readout's wall time. The readout itself is not written: tiled rows are not a test set, and the numbers would mean
nothing. Nothing under $WEIGHTCODE_SEALED is opened; the sealed index is not consulted.

    OMP_NUM_THREADS=4 python -B scripts/store/sealed_mem_probe.py --list LIST --name NAME --rows 10000 --out FILE
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.evalkit import sealed_finals as SF

LABEL = ("synthetic rows: the entry's dev rows of record tiled to the sealed row count; memory and time only, the "
         "readout is not written")


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def tile(arrays: Mapping[str, np.ndarray], rows: int) -> dict[str, np.ndarray]:
    """Every array repeated along its first axis and cut to ``rows`` rows."""
    out = {}
    for key, value in arrays.items():
        reps = -(-rows // len(value))
        out[key] = np.ascontiguousarray(np.concatenate([value] * reps)[:rows])
    return out


def _describe(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": int(value.nbytes)}
            for key, value in arrays.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=SF.DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.rows < 1:
        raise SystemExit("--rows must be >= 1")
    rss = {"imports": _rss_mb()}
    _, entries = SF.load_list(args.list)
    (entry,) = [e for e in entries if e.name == args.name]
    if entry.readout != SF.READOUT_VISION_JOINT:
        raise SystemExit(f"{entry.name}: a {entry.readout} entry; the probe tiles vision_joint rows")
    fam = SF.family(entry.readout)
    instance, child, provenance, context = SF.load_entry(entry)
    (condition, clean), _ = fam.dev_rows(instance)
    condition, clean = tile(condition, args.rows), tile(clean, args.rows)
    rss["loaded"] = _rss_mb()
    start = time.monotonic()
    fam.read(instance, child, (condition, clean), args.batch_size, context)
    seconds = time.monotonic() - start
    rss["peak"] = _rss_mb()
    record = {
        "label": LABEL, "name": entry.name, "instance_id": entry.instance_id,
        "dataset": instance.private.get("dataset"), "condition": instance.private.get("condition"),
        "child_kind": provenance["kind"], "rows": args.rows, "batch_size": args.batch_size,
        "arrays": {"condition": _describe(condition), "clean": _describe(clean)},
        "param_bytes": int(sum(v.nbytes for v in instance.params.values())),
        "max_rss_mb": rss, "read_seconds": seconds,
        "threads": {name: os.environ.get(name) for name in SF.THREAD_VARIABLES}, "nproc": os.cpu_count(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps({k: record[k] for k in ("name", "dataset", "rows", "max_rss_mb", "read_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
