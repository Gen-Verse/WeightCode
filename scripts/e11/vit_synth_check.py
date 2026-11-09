"""E11 harness checks on a SYNTHETIC random-init ViT-Lite-7/4 (``weightcode.models.vit_synth``); never a student of record.

Subcommands, each writing one JSON record next to its inputs and printing it:

* ``export --out DIR``: model.npz, model_spec.json, forward.py and synthetic.json of a random-init ViT-Lite-7/4.
* ``parity --export DIR``: the export's numpy forward.py (float64 inside, float32 logits) against an independent torch
  float64 rebuild (``vit_synth.torch_logits``) on random uint8 images.
* ``timing --export DIR``: the export's forward through ``weightcode.evalkit.patch_check.predict`` at the check's batch
  size on N random 32x32 images, under the MCP server's BLAS thread variables (the caller sets them; refused when they
  differ from ``--threads``). The verdict compares one check (program run up to PROGRAM_WALL_S plus V_check new and
  clean predictions) with the MCP tool timeout, and the server's base predictions with the connect timeout.
* ``store --root ROOT``: a synthetic E11 store of full-size ViT-Lite-7/4 students (``vit_synth.build_store``) for
  dry runs; WEIGHTCODE_PRIVATE, WEIGHTCODE_SEALED and WEIGHTCODE_RUNS must already point at ROOT/private, sealed and runs.
* ``scripted --instance ID``: a scripted LEARN session (no LLM: ``weightcode.harness.scripted.ScriptedLauncher``) on a
  synthetic instance that checks a LayerNorm program, a qkv program and a one-entry-per-tensor program (85 ops, the
  max_ops of this layout), is refused on rel-Frob (qkv, positional embedding, class token) and on the 20% cap, declares
  the LayerNorm program and reads the final out on V_select and its clean companion.

Example (the synthetic store keeps the real store untouched)::

    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    S=$WEIGHTCODE_RUNS/e11/synthetic
    .venv/bin/python scripts/e11/vit_synth_check.py export --out $S/export_s100
    .venv/bin/python scripts/e11/vit_synth_check.py parity --export $S/export_s100 --n 1000
    .venv/bin/python scripts/e11/vit_synth_check.py timing --export $S/export_s100 --n 8000 --threads 4
    R=$S/store_v1; export WEIGHTCODE_PRIVATE=$R/private WEIGHTCODE_SEALED=$R/sealed WEIGHTCODE_RUNS=$R/runs
    .venv/bin/python scripts/e11/vit_synth_check.py store --root $R
    .venv/bin/python scripts/run_editor_session.py --instance $(.venv/bin/python -m weightcode.harness.e11 ids \\
        --condition gaussian_noise_s5) --arm LEARN --repeat 0 --effort max --wall-timeout-min 90 \\
        --retention-floor-frac 0.005 --dry-run
    .venv/bin/python scripts/e11/vit_synth_check.py scripted --instance $(.venv/bin/python -m weightcode.harness.e11 ids \\
        --condition gaussian_noise_s5)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.harness import e10, e11
from weightcode.models import vit_synth as V

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
CHECK_ROWS = 8000
"""PREREG E10 and E11: a check predicts 4,000 V_check condition rows and their 4,000 clean companions."""
STORE_ROWS = {"train": 128, "V_select": 256, "V_evidence": 128, "V_check": 256, "test": 64}
STORE_SEEDS = (100, 101)

STRINGS = 'HYPOTHESIS = "synthetic harness check"\nPREDICTED_EFFECT = "exercises the limits"\n'
PROGRAMS = {
    "ln.py": STRINGS + """import numpy as np

def update(student, experience):
    w = np.asarray(student["params"]["classifier.norm.weight"], dtype=np.float64)
    delta = np.zeros_like(w)
    cut = int(round(0.25 * w.size))
    delta[:cut] = -2.0 * w[:cut]
    return {"classifier.norm.weight": delta}
""",
    "qkv.py": STRINGS + """import numpy as np

def update(student, experience):
    w = np.asarray(student["params"]["classifier.blocks.3.self_attn.qkv.weight"], dtype=np.float64)
    delta = np.zeros_like(w)
    d = w.shape[1]
    delta[:d] = 0.3 * w[:d]
    return {"classifier.blocks.3.self_attn.qkv.weight": delta}
""",
    "every.py": STRINGS + """import numpy as np

def update(student, experience):
    out = {}
    for name, value in student["params"].items():
        delta = np.zeros(value.shape)
        delta.flat[0] = 1e-3
        out[name] = delta
    return out
""",
    "wide.py": STRINGS + """import numpy as np

def update(student, experience):
    p = student["params"]
    out = {name: np.full(p[name].shape, 1e-4) for name in p if p[name].ndim == 1}
    for i in range(7):
        name = f"classifier.blocks.{i}.linear1.weight"
        out[name] = np.full(p[name].shape, 1e-4)
    return out
""",
}
for _name, _tensor in (("qkv_big.py", "classifier.blocks.3.self_attn.qkv.weight"),
                       ("pos.py", V.POS_EMBED), ("cls.py", V.CLASS_TOKEN)):
    PROGRAMS[_name] = STRINGS + f"""import numpy as np

def update(student, experience):
    return {{{_tensor!r}: 0.6 * np.asarray(student["params"][{_tensor!r}], dtype=np.float64)}}
"""
ORDER = ["ln.py", "qkv.py", "qkv_big.py", "pos.py", "cls.py", "wide.py", "every.py"]
EXPECT_ERROR = {"ln.py": False, "qkv.py": False, "qkv_big.py": True, "pos.py": True, "cls.py": True, "wide.py": True,
                "every.py": False}


def _write(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


def _export(directory: Path) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    from weightcode.common.identity import load_params
    from weightcode.evalkit.patch_check import load_forward

    record = json.loads((directory / "synthetic.json").read_text())
    if not record.get("synthetic"):
        raise SystemExit(f"{directory} is not a synthetic export")
    forward, _ = load_forward(directory / "forward.py")
    return forward, load_params(directory / "model.npz"), record


def _images(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(n, *V.VIT_LITE_7_4.image_shape), dtype=np.uint8)


def cmd_export(args: argparse.Namespace) -> None:
    record = V.write_export(args.out, V.VIT_LITE_7_4, args.seed, V.CLASS_NAMES_10)
    record["summary"] = e11.check_student(V.init_params(V.VIT_LITE_7_4, args.seed))
    _write(Path(args.out) / "export.json", record)


def cmd_parity(args: argparse.Namespace) -> None:
    import importlib.util

    directory = Path(args.export)
    forward, params, record = _export(directory)
    spec = importlib.util.spec_from_file_location("synthetic_forward", directory / "forward.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg, x = V.VIT_LITE_7_4, _images(args.n, args.data_seed)
    ref, ours64, ours32 = [], [], []
    for start in range(0, args.n, 250):
        batch = x[start:start + 250]
        ref.append(V.torch_logits(cfg, record["mean"], record["std"], params, batch))
        ours64.append(module.features(params, batch) @ params[V.HEAD_WEIGHT].astype(np.float64).T
                      + params[V.HEAD_BIAS].astype(np.float64))
        ours32.append(np.asarray(forward(params, batch)))
    r, o64, o32 = np.concatenate(ref), np.concatenate(ours64), np.concatenate(ours32)
    scale = float(np.max(np.abs(r)))
    out = {"synthetic": True, "export": str(directory), "n": args.n, "data_seed": args.data_seed,
           "torch_dtype": "float64", "max_abs_logit": scale,
           "f64_max_abs_diff": float(np.max(np.abs(o64 - r))), "f64_rel": float(np.max(np.abs(o64 - r)) / scale),
           "f32_out_dtype": str(o32.dtype), "f32_max_abs_diff": float(np.max(np.abs(o32.astype(np.float64) - r))),
           "f32_rel": float(np.max(np.abs(o32.astype(np.float64) - r)) / scale),
           "argmax_disagree_f64": int(np.sum(o64.argmax(1) != r.argmax(1))),
           "argmax_disagree_f32": int(np.sum(o32.argmax(1) != r.argmax(1))),
           "n_distinct_predictions": int(len(np.unique(r.argmax(1))))}
    out["passed"] = out["f64_rel"] <= 1e-10 and out["f32_rel"] <= 1e-6 and out["argmax_disagree_f32"] == 0
    _write(directory / f"parity_n{args.n}.json", out)
    if not out["passed"]:
        raise SystemExit("parity failed")


def cmd_timing(args: argparse.Namespace) -> None:
    from weightcode.evalkit.patch_check import predict
    from weightcode.harness.runner import DEFAULT_THREADS
    from weightcode.sandbox.program import PROGRAM_WALL_S

    threads = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    if any(value != str(args.threads) for value in threads.values()):
        raise SystemExit(f"set {', '.join(THREAD_VARIABLES)} to {args.threads} before starting Python; got {threads}")
    directory = Path(args.export)
    forward, params, _ = _export(directory)
    x = _images(args.n, args.data_seed)
    predict(forward, params, x[:256], args.batch_size)  # warm-up: imports, BLAS pool, page faults
    load_before = os.getloadavg()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    predict(forward, params, x, args.batch_size)
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (after.ru_utime - usage.ru_utime) + (after.ru_stime - usage.ru_stime)
    per_8000 = wall * CHECK_ROWS / args.n
    tool_s, connect_s = e11.MCP_TOOL_TIMEOUT_MS / 1000.0, e11.MCP_CONNECT_TIMEOUT_MS / 1000.0
    check_worst = PROGRAM_WALL_S + per_8000
    out = {
        "synthetic": True, "export": str(directory), "n": args.n, "batch_size": args.batch_size,
        "threads": threads, "harness_default_threads": DEFAULT_THREADS, "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2), "cpu_over_wall": round(cpu / wall, 2), "ms_per_row": round(1000 * wall / args.n, 3),
        "wall_s_per_8000_rows": round(per_8000, 1), "maxrss_mib": round(after.ru_maxrss / 1024, 1),
        "loadavg_before": [round(v, 2) for v in load_before], "loadavg_after": [round(v, 2) for v in os.getloadavg()],
        "cpu_count": os.cpu_count(), "machine": platform.processor() or platform.machine(),
        "numpy": np.__version__, "python": sys.version.split()[0],
        "verdict": {
            "tool_timeout_s": tool_s, "program_wall_s": PROGRAM_WALL_S,
            "check_worst_s": round(check_worst, 1), "check_fits_tool_timeout": check_worst < tool_s,
            "check_margin_s": round(tool_s - check_worst, 1),
            "prediction_budget_s": tool_s - PROGRAM_WALL_S,
            "speedup_needed_for_tool_timeout": round(max(1.0, per_8000 / (tool_s - PROGRAM_WALL_S)), 2),
            "connect_timeout_s": connect_s, "base_predictions_s": round(per_8000, 1),
            "base_predictions_fit_connect_timeout": per_8000 < connect_s,
            "session_s": 5400.0, "checks_per_session_at_prediction_only": int(5400.0 // per_8000),
        },
    }
    _write(directory / f"timing_n{args.n}_t{args.threads}.json", out)


def cmd_store(args: argparse.Namespace) -> None:
    root = Path(args.root)
    conditions = dict(zip(e10.STREAM_CONDITIONS, e10.STREAM_CORRUPTIONS))
    start = time.perf_counter()
    store = V.build_store(root, V.VIT_LITE_7_4, dataset=e11.DATASET, conditions=conditions,
                          plant_condition=args.plant_condition, class_names=V.CLASS_NAMES_10, seeds=STORE_SEEDS,
                          decoys={100: 101, 101: 100}, rows=STORE_ROWS, instance_conditions=(args.plant_condition,))
    record = {"synthetic": True, "root": str(root), "dataset": store.dataset, "build_s": round(time.perf_counter() -
              start, 1), "models": store.models, "ids": {f"{c}|{s}": i for (c, s), i in store.ids.items()},
              "rows": STORE_ROWS, "condition_ids_seed100": e11.condition_ids(100)}
    _write(root / "store_check.json", record)


def cmd_scripted(args: argparse.Namespace) -> None:
    from weightcode.common.identity import load_params
    from weightcode.evalkit.devreadout import readout_run
    from weightcode.harness import runner
    from weightcode.harness.instance import read_instance_record
    from weightcode.harness.jsonio import read_json
    from weightcode.harness.scripted import ScriptedLauncher, call_and_record
    from weightcode.harness.session import SessionPaths

    record = read_instance_record(args.instance)
    if not (e11.is_e11(record.private) and Path(os.environ["WEIGHTCODE_PRIVATE"]).parent.joinpath(
            "store_check.json").is_file()):
        raise SystemExit(f"{args.instance} is not an instance of a synthetic E11 store; this command runs only there")
    payload = json.dumps(json.dumps(PROGRAMS))
    cell = ("import json\n" f"for name, source in json.loads({payload}).items():\n"
            "    open('/work/patches/' + name, 'w').write(source)\n" f"print(sorted(json.loads({payload})))\n")

    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "python", {"code": cell})
        for name in ORDER:
            call_and_record(client, replies, "patch_check", {"program_file": name})
        call_and_record(client, replies, "status", {})
        call_and_record(client, replies, "declare_final", {"program_file": "ln.py", "rationale": "synthetic"})

    launcher = ScriptedLauncher(script)
    start = time.perf_counter()
    result = runner.run_session(args.instance, "LEARN", 0, effort="max", wall_timeout_s=args.wall_timeout_min * 60.0,
                                launcher=launcher, skip_audit=True, retention_floor_frac=0.005)
    session_s = time.perf_counter() - start
    checks = dict(zip(ORDER, launcher.replies[1:1 + len(ORDER)]))
    paths = SessionPaths(result.root)
    logged = [json.loads(line) for line in paths.checks_log.read_text().splitlines()]
    session = read_json(paths.session_json)
    final = read_json(paths.selection_json)["final"]
    start = time.perf_counter()
    readout = readout_run(result.root / "run.json", clean=True)
    out = {
        "synthetic": True, "instance": args.instance, "status": result.status, "root": str(result.root),
        "session_s": round(session_s, 1), "readout_s": round(time.perf_counter() - start, 1),
        "student": e11.check_student(load_params(record.instance_dir / "model.npz")),
        "limits": session["limits"], "mcp": session["mcp"], "kept_rule": session["kept_rule"],
        "checks": {name: {"is_error": reply.is_error, "expected_error": EXPECT_ERROR[name],
                          **({"error": reply.payload.get("error")} if reply.is_error else
                             {key: reply.payload.get(key) for key in ("tensors", "score_new", "score_old", "score",
                                                                       "edit_size", "edit_frac")})}
                   for name, reply in checks.items()},
        "logged": [{key: entry.get(key) for key in ("patch_id", "score_new", "score_old", "score", "n_changed")}
                   for entry in logged],
        "final": {key: final.get(key) for key in ("kind", "patch_id", "score", "score_new", "score_old")},
        "readout": {"n": readout["n"], "n_clean": readout["n_clean"], "delta_acc": readout["readouts"]["delta_acc"],
                    "delta_acc_clean": readout["readouts_clean"]["delta_acc"],
                    "final_limits": None if readout["final_limits"] is None else str(readout["final_limits"])},
    }
    out["passed"] = (result.status == "complete" and all(c["is_error"] == c["expected_error"]
                                                         for c in out["checks"].values())
                     and out["final"]["patch_id"] == "ln.py" and out["readout"]["delta_acc"] > 0
                     and out["readout"]["delta_acc_clean"] >= 0)
    _write(result.root / "scripted_check.json", out)
    if not out["passed"]:
        raise SystemExit("the scripted synthetic session did not behave as expected")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=100)
    p = sub.add_parser("parity")
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--data-seed", type=int, default=7)
    p = sub.add_parser("timing")
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--n", type=int, default=CHECK_ROWS)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1024, help="the check oracle's predict batch size")
    p.add_argument("--data-seed", type=int, default=11)
    p = sub.add_parser("store")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plant-condition", default="gaussian_noise_s5", choices=e10.STREAM_CONDITIONS)
    p = sub.add_parser("scripted")
    p.add_argument("--instance", required=True)
    p.add_argument("--wall-timeout-min", type=float, default=30.0)
    args = parser.parse_args(argv)
    {"export": cmd_export, "parity": cmd_parity, "timing": cmd_timing, "store": cmd_store,
     "scripted": cmd_scripted}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
