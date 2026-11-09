"""PREREG T04 store steps (``weightcode.data.t04``, ``weightcode.data.t04_build``). Dev rows only; run on NSCC inside
scripts/nscc_session/outer.sh, so that the records hold the workstation's store paths.

    python scripts/t04/t04_store.py rows                                    # the row rule and its sha256 (no data)
    python scripts/t04/t04_store.py clean --data-dir SHARDS [--d-index runs/t03/d_index.jsonl]
    python scripts/t04/t04_store.py models --weights $WEIGHTCODE_PRIVATE/pretrained/resnet18-f37072fd.pth
    python scripts/t04/t04_store.py diag --official make_imagenet_c.py --out $WEIGHTCODE_RUNS/t04/severity_diag.json
    python scripts/t04/t04_store.py condition --corruption gaussian_noise --official make_imagenet_c.py \\
        --diag $WEIGHTCODE_RUNS/t04/severity_diag.json
    python scripts/t04/t04_store.py base --out $WEIGHTCODE_RUNS/t04/base_readouts.json
    python scripts/t04/t04_store.py instances --base $WEIGHTCODE_RUNS/t04/base_readouts.json
    OMP_NUM_THREADS=4 python scripts/t04/t04_store.py timing --out $WEIGHTCODE_RUNS/t04/timing.json

``diag`` (PREREG T04's severity rule, before any condition is stored): the seed-100 network's forward.py accuracy on
D (the clean dev rows whose number is a multiple of 50, read from the clean store) and on D under each corruption at
severities 1 to 5, and the rule's choice per corruption (``weightcode.data.t04.choose_severity``). ``condition`` stores the
chosen severity of one corruption, and refuses a corruption the rule chose none for. ``base`` (before any instance):
the base's accuracy on the clean and every stored condition's V_evidence, V_check and V_select, the V_select headroom
H, E10's headroom rule (H >= 15 points) and the gain bar (``weightcode.data.t04.gain_bar``). ``instances`` builds the
evidence-v1 instances (seeds 100 and 101, the decoy map 100 <-> 101) of every stored condition that passes the
headroom rule. ``timing`` measures forward.py per check (8,000 rows) and per program pass (128 rows) at the thread
count the environment sets (the sessions use 4) and derives the session wall limit; it also times the large-sample
diagnostic's Adam step (``scripts/t04/large_ft.py``: 64 condition V_evidence rows and 64 clean train rows, forward and
backward at 224 x 224, every trainable tensor) at each of ``--train-threads`` torch threads, which sizes that job (R37).
The step's weights are discarded. Every record is written once (refused if it exists).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from weightcode.data import t04 as T
from weightcode.data import t04_build as B
from weightcode.models.runinfo import run_info

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_once(out: Path, record: dict) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    with open(out, "x", encoding="utf-8") as handle:
        handle.write(text)
    print(f"-> {out} sha256 {hashlib.sha256(text.encode()).hexdigest()}")


def _student(seed: int) -> tuple[dict, object, dict]:
    from weightcode.common.identity import load_params, params_sha256
    from weightcode.data.store import models_dir
    from weightcode.evalkit.targets import load_dataset_forward, read_dataset_meta, read_model_index

    meta = read_dataset_meta(T.DATASET)
    forward, forward_sha, _ = load_dataset_forward(T.DATASET, meta)
    (entry,) = [e for e in read_model_index(T.DATASET) if e.seed == seed and e.role == "dev"]
    params = load_params(models_dir(T.DATASET) / entry.model_id / "model.npz")
    if params_sha256(params) != entry.params_sha256:
        raise SystemExit("model.npz hash differs from the index")
    return params, forward, {"model_id": entry.model_id, "seed": seed, "params_sha256": entry.params_sha256,
                             "forward_sha256": forward_sha}


def _d_rows_from_store() -> dict[str, np.ndarray]:
    """D's crops, labels and rows, gathered from the clean splits (hash-checked) in row order."""
    from weightcode.data.store import load_split, read_dataset_json, split_sha256

    meta = read_dataset_json(T.DATASET)
    xs, ys, rows = [], [], []
    for split in T.SPLITS:
        arrays = load_split(T.DATASET, split)
        if split_sha256(arrays) != meta["splits"][split]["sha256"]:
            raise SystemExit(f"clean {split} differs from its dataset.json hash")
        keep = arrays["idx"] % 50 == 0
        xs.append(arrays["x"][keep]); ys.append(arrays["y"][keep]); rows.append(arrays["idx"][keep])
    x, y, idx = np.concatenate(xs), np.concatenate(ys), np.concatenate(rows)
    order = np.argsort(idx, kind="stable")
    if not np.array_equal(idx[order], np.arange(0, 50_000, 50)):
        raise SystemExit("the clean store does not hold exactly D's 1,000 rows at multiples of 50")
    return {"x": x[order], "y": y[order], "idx": idx[order]}


def cmd_diag(args: argparse.Namespace) -> int:
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists; nothing written")
    params, forward, student = _student(B.SOURCE_SEED)
    official = T.load_official(Path(args.official))
    constants = T.check_official_constants(official)
    d = _d_rows_from_store()
    started = time.time()
    clean_acc = float(np.mean(np.asarray(forward(params, d["x"])).argmax(1) == d["y"]))
    by_corruption = {}
    for corruption in T.CORRUPTIONS:
        accs = {}
        for severity in T.SEVERITIES:
            x = T.corrupt_split(official, corruption, severity, d["x"], d["idx"])
            accs[severity] = float(np.mean(np.asarray(forward(params, x)).argmax(1) == d["y"]))
            print(f"{corruption} s{severity}: {accs[severity]:.4f}", flush=True)
        by_corruption[corruption] = T.choose_severity(clean_acc, accs)
    record = {"schema": "weightcode.t04.severity_diag.v1", "rows": "D: clean dev rows with row % 50 == 0 (1,000)",
              "d_rows_sha256": T.rows_sha256(d["idx"]), "student": student, "acc_clean_d": clean_acc,
              "q_min": T.Q_MIN, "min_drop_points": T.MIN_DROP_POINTS, "corruptions": by_corruption,
              "official": {"file": T.OFFICIAL_FILE, "commit": T.OFFICIAL_COMMIT, "sha256": T.OFFICIAL_SHA256,
                           "function_sha256": official["_sources"], "constants_check": constants},
              "libraries": T.image_libraries(), "wall_s": time.time() - started, "run_info": run_info(REPO_ROOT)}
    _write_once(Path(args.out), record)
    print(json.dumps({c: r["chosen"] for c, r in by_corruption.items()}))
    return 0


def cmd_condition(args: argparse.Namespace) -> int:
    diag = json.loads(Path(args.diag).read_text(encoding="utf-8"))
    choice = diag["corruptions"][args.corruption]
    if choice["chosen"] is None:
        print(f"the severity rule chose no severity for {args.corruption}; the condition is not run")
        return 0
    record = dict(choice, diag_file=str(args.diag),
                  diag_sha256=hashlib.sha256(Path(args.diag).read_bytes()).hexdigest())
    manifest = B.build_condition(args.corruption, int(choice["chosen"]), Path(args.official), run_info(REPO_ROOT),
                                 record)
    print(json.dumps({"condition": manifest["condition"]["name"],
                      "splits": {s: v["sha256"] for s, v in manifest["splits"].items()}}, indent=1))
    return 0


def cmd_base(args: argparse.Namespace) -> int:
    from weightcode.data.store import list_conditions

    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists; nothing written")
    params, forward, student = _student(B.SOURCE_SEED)
    conditions = list_conditions(T.DATASET)
    table = B.accuracy_table(params, forward, conditions)
    clean = table["clean"]["V_select"]["acc"]
    verdicts = {}
    for condition in conditions:
        headroom = 100.0 * (clean - table[condition]["V_select"]["acc"])
        verdicts[condition] = {"headroom_points": headroom, "passes": bool(headroom >= T.MIN_DROP_POINTS),
                               "gain_bar_points": T.gain_bar(headroom)}
    record = {"schema": "weightcode.t04.base_readouts.v1", "student": student, "accuracy": table,
              "headroom_rule": f"clean minus condition V_select accuracy >= {T.MIN_DROP_POINTS} points (E10's)",
              "gain_bar_rule": f"{T.GAIN_SHARE} x headroom, rounded to the nearest 0.5 point", "conditions": verdicts,
              "run_info": run_info(REPO_ROOT)}
    _write_once(Path(args.out), record)
    print(json.dumps(verdicts, indent=1))
    return 0


def cmd_timing(args: argparse.Namespace) -> int:
    """forward.py time per check (a stored condition's V_check and the clean V_check, 8,000 rows) and per program pass
    (the 128 experience rows), worst of ``--repeats``, at the thread count the environment sets; peak RSS; the session
    settings that follow (E11's session-length rule and E10's MCP floors)."""
    import math
    import os
    import resource

    from weightcode.data.store import list_conditions, load_split

    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists; nothing written")
    params, forward, student = _student(B.SOURCE_SEED)
    condition = sorted(list_conditions(T.DATASET))[0]
    x = np.concatenate([load_split(T.DATASET, "V_check", condition)["x"], load_split(T.DATASET, "V_check")["x"]])
    check, program = [], []
    for _ in range(args.repeats):
        load = os.getloadavg()[0]
        started = time.perf_counter()
        for i in range(0, len(x), 1024):
            np.asarray(forward(params, x[i:i + 1024]))
        check.append({"s": time.perf_counter() - started, "load_1min_before": load})
        started = time.perf_counter()
        np.asarray(forward(params, x[:128]))
        program.append(time.perf_counter() - started)
    worst = max(c["s"] for c in check)
    wall_min = 120 + max(0.0, 8 * worst / 60.0 - 30.0)
    train_step = _train_step_timing(condition, args.train_threads, args.train_steps)
    record = {"schema": "weightcode.t04.timing.v1", "student": student, "condition": condition, "rows_per_check": len(x),
              "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
              "check_s": check, "worst_check_s": worst, "program_128_rows_s": program,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
              "session_wall_limit_min": int(5 * math.ceil(wall_min / 5)),
              "wall_rule": "120 min raised by the 8-check prediction time above 30 min, rounded up to 5 min (E11's)",
              "tool_timeout_fits": bool(300 + worst <= 1200), "connect_timeout_fits": bool(worst <= 900),
              "large_ft_step_s": train_step, "run_info": run_info(REPO_ROOT)}
    _write_once(Path(args.out), record)
    print(json.dumps({k: record[k] for k in ("worst_check_s", "session_wall_limit_min", "peak_rss_gib",
                                             "tool_timeout_fits", "connect_timeout_fits", "large_ft_step_s")},
                     indent=1))
    return 0


def _train_step_timing(condition: str, thread_counts: list[int], steps: int) -> dict[str, dict]:
    """Seconds per large-sample Adam step (one warm-up, then ``steps`` timed) at each torch thread count."""
    import importlib.util

    import torch

    from weightcode.baselines.instance_io import load_instance
    from weightcode.evalkit.instances import instances_root
    from weightcode.harness import t04 as H

    spec = importlib.util.spec_from_file_location("t04_large_ft", REPO_ROOT / "scripts" / "t04" / "large_ft.py")
    large = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(large)
    ids = H.condition_ids()
    condition = condition if condition in ids else sorted(ids)[0]  # a condition that has a seed-100 instance
    instance = load_instance(instances_root() / ids[condition])
    params = instance.params
    arch = large.architecture_from(params, large.read_model_spec(instance))
    trainable = large.trainable_tensors(arch, params)
    from weightcode.data.store import load_split
    from weightcode.evalkit.targets import deployment_splits, load_checked_split

    splits = deployment_splits(T.DATASET, condition)
    cond = load_checked_split(splits.v_evidence, splits.v_evidence_sha256)
    clean = load_split(T.DATASET, "train")
    half = large.HALF_BATCH
    xb = large.images_tensor(np.concatenate([cond["x"][:half], clean["x"][:half]]))
    yb = torch.from_numpy(np.concatenate([cond["y"][:half], clean["y"][:half]]).astype(np.int64))
    out = {}
    for threads in thread_counts:
        torch.set_num_threads(int(threads))
        tensors = large.to_tensors(params, trainable=trainable)
        optimizer = torch.optim.Adam([tensors[key] for key in trainable], lr=1e-5, foreach=False)
        times = []
        for i in range(steps + 1):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = large.F.cross_entropy(large.logits(arch, tensors, xb), yb)
            loss.backward()
            optimizer.step()
            if i:  # the first step is a warm-up
                times.append(time.perf_counter() - started)
        out[str(threads)] = {"steps_s": times, "max_s": max(times), "batch": int(len(yb)), "condition": condition,
                             "torch_threads": torch.get_num_threads()}
    return out


def cmd_instances(args: argparse.Namespace) -> int:
    from weightcode.evalkit.instances import build_deployment_instances
    from weightcode.evalkit.targets import SeedSelection

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    out = {}
    for condition, verdict in sorted(base["conditions"].items()):
        if not verdict["passes"]:
            out[condition] = "headroom rule failed; no instance"
            continue
        outcomes = build_deployment_instances(
            T.DATASET, condition, selection=SeedSelection(main=(), dev=(B.SOURCE_SEED, B.DECOY_SEED)),
            decoys={B.SOURCE_SEED: B.DECOY_SEED, B.DECOY_SEED: B.SOURCE_SEED}, split_roles=("dev",),
            batch_size=args.batch_size)
        out[condition] = [dataclasses.asdict(outcome) for outcome in outcomes]
    print(json.dumps(out, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("rows")
    clean = sub.add_parser("clean")
    clean.add_argument("--data-dir", type=Path, required=True)
    clean.add_argument("--d-index", type=Path)
    models = sub.add_parser("models")
    models.add_argument("--weights", type=Path, required=True)
    diag = sub.add_parser("diag")
    diag.add_argument("--official", type=Path, required=True)
    diag.add_argument("--out", type=Path, required=True)
    condition = sub.add_parser("condition")
    condition.add_argument("--corruption", choices=T.CORRUPTIONS, required=True)
    condition.add_argument("--official", type=Path, required=True)
    condition.add_argument("--diag", type=Path, required=True)
    base = sub.add_parser("base")
    base.add_argument("--out", type=Path, required=True)
    instances = sub.add_parser("instances")
    instances.add_argument("--base", type=Path, required=True)
    instances.add_argument("--batch-size", type=int, default=256)
    timing = sub.add_parser("timing")
    timing.add_argument("--out", type=Path, required=True)
    timing.add_argument("--repeats", type=int, default=2)
    timing.add_argument("--train-threads", type=int, nargs="+", default=[4, 8])
    timing.add_argument("--train-steps", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "rows":
        print(json.dumps(T.row_record(), indent=1))
        return 0
    if args.command == "clean":
        manifest = B.build_clean(args.data_dir, run_info(REPO_ROOT), d_index_path=args.d_index)
        print(json.dumps({"splits": {s: {k: v[k] for k in ("n", "sha256", "coverage")}
                                     for s, v in manifest["splits"].items()},
                          "forward_sha256": manifest["forward_sha256"],
                          "class_name_changes": manifest["class_name_changes"],
                          "d_index_check": manifest["d_index_check"]}, indent=1))
        return 0
    if args.command == "models":
        print(json.dumps(B.build_models(args.weights, run_info(REPO_ROOT)), indent=1))
        return 0
    return {"diag": cmd_diag, "condition": cmd_condition, "base": cmd_base, "instances": cmd_instances,
            "timing": cmd_timing}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
