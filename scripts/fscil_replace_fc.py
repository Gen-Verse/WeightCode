"""Apply CEC's replace_base_fc to the E9 students of one seed (``weightcode.models.fscil_export.replace_base_fc``).

    .venv/bin/python scripts/fscil_replace_fc.py --seed 100 --note "..." [--workers 4]

The features of the training rows come from the dataset's hash-checked forward.py, in chunks of 256 rows mapped over
``--workers`` processes (1: serial); the chunking does not change a feature. Prints the replacement records.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from typing import Any

import numpy as np

CHUNK = 256
_FEATURES: Any = None


def _init() -> None:
    global _FEATURES
    from weightcode.data.store import read_dataset_json
    from weightcode.models.fscil_export import DATASET, checked_features

    _FEATURES = checked_features(DATASET, read_dataset_json(DATASET))


def _chunk(args: tuple[dict, np.ndarray]) -> np.ndarray:
    params, x = args
    return np.asarray(_FEATURES(params, x), dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    from weightcode.models.fscil_export import replace_base_fc

    if args.workers <= 1:
        records = replace_base_fc(args.seed, note=args.note)
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:

            def features(params: dict, x: np.ndarray) -> np.ndarray:
                chunks = [(params, x[i : i + CHUNK]) for i in range(0, len(x), CHUNK)]
                return np.concatenate(pool.map(_chunk, chunks))

            records = replace_base_fc(args.seed, note=args.note, features=features)
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
