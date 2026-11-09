"""PREREG E11 store steps in the project environment (``weightcode.data.e11``, ``weightcode.models.e11_export``).

    source scripts/env.sh
    python scripts/e11/e11_store.py clean
    python scripts/e11/e11_store.py condition gaussian_noise_s5 defocus_blur_s5 snow_s5 contrast_s5
    python scripts/e11/e11_store.py import --run-dir $WEIGHTCODE_PRIVATE/nscc_runs/RUN --note "..."
    python scripts/e11/e11_store.py headroom --seed 100 --out FILE
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/e11/e11_store.py timing --seed 100

``condition`` copies E10's condition rows from ``cifar10_r20`` byte for byte (no sealed read, no regeneration).
``headroom`` (the entry's check before any instance, E10's rule): the student's clean V_select accuracy and its
accuracy on each stored condition's V_select; a condition runs only if clean minus condition is at least
``HEADROOM_POINTS``. ``timing``: forward.py's wall time on ``TIMING_ROWS`` train rows (no readout split) with the
thread count the environment sets, best of ``TIMING_REPEATS``; the number PREREG E11 records for the MCP timeouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from weightcode.data import e11 as D
from weightcode.data.store import list_conditions, load_split, read_dataset_json, split_sha256
from weightcode.models import e11_export as E
from weightcode.models.runinfo import run_info

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADROOM_POINTS = 15.0
TIMING_ROWS = 8000
TIMING_REPEATS = 3
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _student(seed: int) -> tuple[dict, object, str, dict]:
    from weightcode.common.identity import load_params, params_sha256
    from weightcode.data.store import models_dir
    from weightcode.evalkit.targets import load_dataset_forward, read_dataset_meta, read_model_index

    meta = read_dataset_meta(D.DATASET)
    forward, forward_sha, _ = load_dataset_forward(D.DATASET, meta)
    (entry,) = [e for e in read_model_index(D.DATASET) if e.seed == seed and e.role == "dev"]
    params = load_params(models_dir(D.DATASET) / entry.model_id / "model.npz")
    if params_sha256(params) != entry.params_sha256:
        raise SystemExit("model.npz hash differs from the index")
    return params, forward, forward_sha, {"model_id": entry.model_id, "params_sha256": entry.params_sha256}


def _write_once(out: Path, record: dict) -> None:
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    with open(out, "x", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    print(f"-> {out} {hashlib.sha256(text.encode()).hexdigest()}")


def _headroom(seed: int, out: Path) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists; nothing written")
    params, forward, forward_sha, student = _student(seed)

    def accuracy(condition: str | None) -> tuple[float, str]:
        arrays = load_split(D.DATASET, "V_select", condition)
        recorded = read_dataset_json(D.DATASET, condition)["splits"]["V_select"]["sha256"]
        if split_sha256(arrays) != recorded:
            raise SystemExit(f"V_select of {condition or 'clean'} differs from its dataset.json")
        pred = np.asarray(forward(params, arrays["x"])).argmax(1)
        return float(np.mean(pred == arrays["y"])), recorded

    clean, clean_sha = accuracy(None)
    rows = {}
    for condition in list_conditions(D.DATASET):
        acc, sha = accuracy(condition)
        gap = 100.0 * (clean - acc)
        rows[condition] = {"vselect_acc": acc, "vselect_sha256": sha, "gap_points": gap,
                           "runs": gap >= HEADROOM_POINTS}
    record = {"schema": "weightcode.e11.headroom.v1", "rule": f"clean minus condition >= {HEADROOM_POINTS} points on "
              "V_select (PREREG E11 on E10's rule, before any instance)", **student, "seed": seed,
              "forward_sha256": forward_sha, "clean_vselect_acc": clean, "clean_vselect_sha256": clean_sha,
              "conditions": rows}
    _write_once(out, record)
    return record


def _timing(seed: int, out: Path | None) -> dict:
    threads = {name: os.environ.get(name) for name in THREAD_VARS}
    if len(set(threads.values())) != 1 or None in threads.values():
        raise SystemExit(f"set {', '.join(THREAD_VARS)} to one thread count before the run, got {threads}")
    params, forward, forward_sha, student = _student(seed)
    train = load_split(D.DATASET, "train")
    x = np.ascontiguousarray(train["x"][:TIMING_ROWS])
    forward(params, x[:8])  # warm-up
    times = []
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter()
        forward(params, x)
        times.append(time.perf_counter() - started)
    record = {"schema": "weightcode.e11.forward_timing.v1", **student, "seed": seed, "forward_sha256": forward_sha,
              "rows": f"the first {TIMING_ROWS} rows of {D.DATASET}/train", "threads": threads,
              "repeats_s": times, "best_s": min(times), "numpy": np.__version__, "cpu_count": os.cpu_count(),
              "host": os.uname().nodename}
    if out is not None:
        if out.exists():
            raise SystemExit(f"{out} exists; nothing written")
        _write_once(out, record)
    else:
        print(json.dumps(record, indent=1))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="group", required=True)
    sub.add_parser("clean")
    cond = sub.add_parser("condition")
    cond.add_argument("conditions", nargs="+")
    imp = sub.add_parser("import")
    imp.add_argument("--run-dir", type=Path, required=True)
    imp.add_argument("--note", required=True)
    head = sub.add_parser("headroom")
    head.add_argument("--seed", type=int, default=100)
    head.add_argument("--out", type=Path, required=True)
    tim = sub.add_parser("timing")
    tim.add_argument("--seed", type=int, default=100)
    tim.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.group == "clean":
        manifest = D.build_clean(run_info(REPO_ROOT))
        print(json.dumps({"dataset": manifest["dataset"], "forward_sha256": manifest["forward_sha256"],
                          "splits": {k: v["sha256"] for k, v in manifest["splits"].items()}}, indent=1))
    elif args.group == "condition":
        info = run_info(REPO_ROOT)
        for condition in args.conditions:
            manifest = D.build_condition(condition, info)
            print(json.dumps({"condition": manifest["condition"]["name"],
                              "splits": {k: v["sha256"] for k, v in manifest["splits"].items()}}, indent=1))
    elif args.group == "import":
        print(json.dumps(E.import_e11_models(args.run_dir, args.note), indent=1))
    elif args.group == "headroom":
        _headroom(args.seed, args.out)
    else:
        _timing(args.seed, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
