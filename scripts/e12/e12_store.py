"""PREREG E12 store steps in the project environment (``weightcode.data.e12``, ``weightcode.models.e12_export``).

    source scripts/env.sh
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/e12/e12_store.py timing --resize 128 --out FILE
    python scripts/e12/e12_store.py clean --resize R
    python scripts/e12/e12_store.py condition gaussian_noise_s5 contrast_s5 defocus_blur_s5 snow_s5
    python scripts/e12/e12_store.py import --run-dir $WEIGHTCODE_PRIVATE/nscc_runs/RUN --note "..."
    python scripts/e12/e12_store.py headroom --seed 100 --out FILE

``timing`` (PREREG E12's R rule, before training and before any instance): forward.py rendered at ``--resize``, run
on the fine-tuning start of the seed (every tensor but fc from the ImageNet weights, fc from the seed), on the first
``TIMING_ROWS`` rows of cifar10's train split (no readout split) with the thread count the environment sets, best of
``TIMING_REPEATS``; also on the first ``PROGRAM_ROWS`` rows (a program's experience rows). The time does not depend on
the parameter values, only on R. The host's load average is recorded around every repeat. ``clean`` writes the store
with forward.py at the chosen R. ``condition`` copies E10's condition rows from ``cifar10_r20`` byte for byte (no
sealed read, no regeneration). ``headroom`` (before any instance, E10's rule): the student's clean V_select accuracy
and its accuracy on each stored condition's V_select; a condition runs only if clean minus condition is at least
``HEADROOM_POINTS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

from weightcode.data import e12 as D
from weightcode.data.store import list_conditions, load_split, read_dataset_json, split_sha256
from weightcode.models import e12_export as E
from weightcode.models import resnet18up as M
from weightcode.models.runinfo import run_info

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADROOM_POINTS = 15.0
TIMING_ROWS = 8000
PROGRAM_ROWS = 128
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
    record = {"schema": "weightcode.e12.headroom.v1", "rule": f"clean minus condition >= {HEADROOM_POINTS} points on "
              "V_select (PREREG E12 on E10's rule, before any instance); gaussian_noise_s5 and contrast_s5 first, "
              "then defocus_blur_s5, snow_s5 in that order for a condition short of it", **student, "seed": seed,
              "forward_sha256": forward_sha, "clean_vselect_acc": clean, "clean_vselect_sha256": clean_sha,
              "conditions": rows}
    _write_once(out, record)
    return record


def _timed(forward, params, x, repeats: int) -> tuple[list[float], list[list[float]]]:
    times, loads = [], []
    for _ in range(repeats):
        before = os.getloadavg()
        started = time.perf_counter()
        forward(params, x)
        times.append(time.perf_counter() - started)
        loads.append([round(v, 2) for v in (*before, *os.getloadavg())])
    return times, loads


def _timing(resize: int, seed: int, repeats: int, out: Path | None) -> dict:
    from weightcode.evalkit.targets import file_sha256
    from weightcode.models.forward_gen import render_resnet18up_forward

    threads = {name: os.environ.get(name) for name in THREAD_VARS}
    if len(set(threads.values())) != 1 or None in threads.values():
        raise SystemExit(f"set {', '.join(THREAD_VARS)} to one thread count before the run, got {threads}")
    if out is not None and out.exists():
        raise SystemExit(f"{out} exists; nothing written")
    info = run_info(REPO_ROOT)
    config = M.ResNet18UpConfig(resize=int(resize))
    params = M.export_params(M.initial_student(config, E.weights_path(), seed))
    text = render_resnet18up_forward(M.IMAGENET_MEAN, M.IMAGENET_STD, config.n_classes, M.BN_EPS, config.resize)
    module = types.ModuleType("forward")
    exec(compile(text, "forward.py", "exec"), module.__dict__)
    x_all, _, splits, _ = E.training_rows()
    x = np.ascontiguousarray(x_all[:TIMING_ROWS])
    module.forward(params, x[:8])  # warm-up
    times, loads = _timed(module.forward, params, x, repeats)
    small, small_loads = _timed(module.forward, params, np.ascontiguousarray(x[:PROGRAM_ROWS]), repeats)
    record = {"schema": "weightcode.e12.forward_timing.v1", "resize": config.resize, "seed": seed,
              "student": "the fine-tuning start (ImageNet weights, fc from the seed); the time depends on R only",
              "weights_sha256": file_sha256(E.weights_path()),
              "forward_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "rows": f"the first {TIMING_ROWS} rows of cifar10/train", "train_sha256": splits["train"],
              "threads": threads, "repeats_s": times, "best_s": min(times),
              "loadavg_before_after": loads, "program_rows": PROGRAM_ROWS, "program_rows_repeats_s": small,
              "program_rows_best_s": min(small), "program_rows_loadavg": small_loads,
              "numpy": np.__version__, "cpu_count": os.cpu_count(), "host": os.uname().nodename,
              "run_info": info}
    if out is not None:
        _write_once(out, record)
    else:
        print(json.dumps(record, indent=1))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="group", required=True)
    clean = sub.add_parser("clean")
    clean.add_argument("--resize", type=int, required=True, choices=M.RESIZES)
    cond = sub.add_parser("condition")
    cond.add_argument("conditions", nargs="+")
    imp = sub.add_parser("import")
    imp.add_argument("--run-dir", type=Path, required=True)
    imp.add_argument("--note", required=True)
    head = sub.add_parser("headroom")
    head.add_argument("--seed", type=int, default=E.SOURCE_SEED)
    head.add_argument("--out", type=Path, required=True)
    tim = sub.add_parser("timing")
    tim.add_argument("--resize", type=int, required=True, choices=M.RESIZES)
    tim.add_argument("--seed", type=int, default=E.SOURCE_SEED)
    tim.add_argument("--repeats", type=int, default=TIMING_REPEATS)
    tim.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.group == "clean":
        manifest = D.build_clean(args.resize, run_info(REPO_ROOT))
        print(json.dumps({"dataset": manifest["dataset"], "forward_sha256": manifest["forward_sha256"],
                          "resize": manifest["student_preprocessing"]["resize"],
                          "splits": {k: v["sha256"] for k, v in manifest["splits"].items()}}, indent=1))
    elif args.group == "condition":
        info = run_info(REPO_ROOT)
        for condition in args.conditions:
            manifest = D.build_condition(condition, info)
            print(json.dumps({"condition": manifest["condition"]["name"],
                              "splits": {k: v["sha256"] for k, v in manifest["splits"].items()}}, indent=1))
    elif args.group == "import":
        print(json.dumps(E.import_e12_models(args.run_dir, args.note), indent=1))
    elif args.group == "headroom":
        _headroom(args.seed, args.out)
    else:
        _timing(args.resize, args.seed, args.repeats, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
