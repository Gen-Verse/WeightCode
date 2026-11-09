"""PREREG T03: the native ImageNet interface and cost precheck (no LLM), one run on one node.

    python scripts/t03/run_t03.py --data-root DIR --weights resnet18-f37072fd.pth --out OUT \
        --code-commit SHA [--threads 4] [--job-id ID] [--ncpus 8 --mem-gb 29 --walltime-s 7200] [--smoke ...]

DIR holds manifest.sha256, classes.py and data/validation-*.parquet (weightcode.data.imagenet_val). The run checks the
manifest, decodes only D (1,000 rows, global row number % 50 == 0), preprocesses with torchvision's IMAGENET1K_V1 eval
transform, and checks PREREG T03's pass items (1) to (4) (weightcode.evalkit.t03_native) with forward.py
(weightcode.models.resnet18native.render_forward) against float64 and float32 torch rebuilds. The cost items run in fresh
numpy-only processes (scripts/t03/cost_probe.py) at the inherited thread settings. Records go to OUT:
t03_summary.json, t03_summary.md, d_index.jsonl (D's rows, hashes and labels), forward.py, model_spec.json,
logits.npz, the fixed edit's patch.json, cost probe outputs and run.log. OUT/work holds the preprocessed D tensor, the
labels and the params files (large; not records).

--smoke is for a synthetic dataset only: it accepts another manifest hash and row count and marks every record smoke.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from weightcode.data import imagenet_val as I
from weightcode.evalkit import t03_native as T
from weightcode.evalkit.patch_check import load_forward
from weightcode.models import resnet18native as N
from weightcode.models import resnet18up as M
from weightcode.models.forward_gen import write_forward
from weightcode.sandbox.host import SandboxConfig
from weightcode.sandbox.program import PROGRAM_WALL_S

SUMMARY_SCHEMA = "weightcode.t03.summary.v1"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
PROBE_SIZES = (1000, 64)
"""Cost probes: a 1,000-image check (all of D) and a 64-image readout (four forward.py chunks)."""
SMOKE_SMALL_PROBE = 2
TORCHVISION_CHECK_ROWS = 8
SANDBOX_MEMORY_BYTES = SandboxConfig.__dataclass_fields__["memory_bytes"].default
SANDBOX_CELL_TIMEOUT_S = SandboxConfig.__dataclass_fields__["cell_timeout_s"].default
READOUTS_PER_NODE = 2
"""Readouts that share one 8-core node at the session settings (4 threads each): a readout fits when its peak RSS is
at most the node's memory divided by this."""


def log(handle, message: str) -> None:
    line = f"{datetime.datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{','.join(map(str, array.shape))}|".encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    """VmHWM of this process (/proc/self/status), in bytes."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            amount, unit = line.split()[1:3]
            if unit != "kB":
                raise ValueError(f"unexpected unit {unit!r} in /proc/self/status")
            return int(amount) * 1024
    raise RuntimeError("no VmHWM line in /proc/self/status")


def timed(fn, *args):
    start = time.perf_counter()
    value = fn(*args)
    return value, time.perf_counter() - start


def versions() -> dict[str, Any]:
    import numpy
    import PIL
    import pyarrow
    import torchvision

    blas = numpy.show_config(mode="dicts").get("Build Dependencies", {}).get("blas")
    return {"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__,
            "numpy_blas": blas, "torch": torch.__version__, "torchvision": torchvision.__version__,
            "pillow": PIL.__version__, "pyarrow": pyarrow.__version__,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_parallel_info": torch.__config__.parallel_info().splitlines()[:12]}


def run_probe(python: Path, script: Path, forward: Path, params: Path, inputs: Path, labels: Path, n: int,
              out: Path) -> dict[str, Any]:
    """One cost probe in a fresh process; the parent's wall time from launch to exit is added."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    start = time.perf_counter()
    completed = subprocess.run([str(python), "-I", str(script), "--forward", str(forward), "--params", str(params),
                                "--inputs", str(inputs), "--labels", str(labels), "--n", str(n), "--out", str(out)],
                               env=env, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"cost probe n={n} failed ({completed.returncode}): {completed.stderr[-4000:]}")
    record = json.loads(out.read_text())
    record["process_wall_s"] = wall
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--job-id", default=os.environ.get("PBS_JOBID", ""))
    parser.add_argument("--ncpus", type=int, default=8)
    parser.add_argument("--mem-gb", type=float, default=29.0)
    parser.add_argument("--walltime-s", type=int, default=7200)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-manifest-sha256", default=None)
    parser.add_argument("--smoke-expected-rows", type=int, default=None)
    args = parser.parse_args()

    started_wall = time.time()
    started = time.perf_counter()
    if args.smoke != (args.smoke_expected_rows is not None):
        raise SystemExit("--smoke and --smoke-expected-rows go together")
    for name in THREAD_VARS:
        if os.environ.get(name) != str(args.threads):
            raise SystemExit(f"{name} must be {args.threads} in the environment (got {os.environ.get(name)!r})")
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    work = out / "work"
    work.mkdir()
    handle = open(out / "run.log", "w", encoding="utf-8")
    steps: dict[str, float] = {}
    expected_rows = args.smoke_expected_rows if args.smoke else I.N_VAL_ROWS
    manifest_expected = args.smoke_manifest_sha256 if args.smoke else I.MANIFEST_SHA256

    # Data: the manifest, classes.py and D.
    root = Path(args.data_root)
    log(handle, f"verify manifest under {root}")
    listed, steps["verify_manifest_s"] = timed(I.verify_manifest, root, manifest_expected)
    shards = I.shard_names()
    missing = [name for name in [f"data/{s}" for s in shards] + [I.CLASSES_FILE] if name not in listed]
    if missing:
        raise RuntimeError(f"manifest lacks {missing}")
    classes = I.parse_classes(root / I.CLASSES_FILE)
    class_order = I.classes_wnid_order(classes)
    from torchvision.models import ResNet18_Weights

    categories = list(ResNet18_Weights.IMAGENET1K_V1.meta["categories"])
    first_names = [names.split(",")[0].strip() for _, names in classes]
    class_order["torchvision_categories_equal_first_name"] = int(sum(a == b for a, b in zip(first_names, categories)))
    counts = I.shard_row_counts(root / "data", shards)
    rows = I.d_rows(counts, shards, expected_rows=expected_rows)
    log(handle, f"{sum(counts)} rows in {len(shards)} shards; D has {len(rows)} rows")

    transform = I.eval_transform()
    x = np.empty((len(rows), *N.INPUT_SHAPE), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    entries = []
    start = time.perf_counter()
    for position, record in enumerate(I.read_d_records(root / "data", rows)):
        if (record.global_index, record.shard, record.row) != (rows[position].global_index, rows[position].shard,
                                                              rows[position].row):
            raise RuntimeError(f"D record {position} is out of order")
        x[position] = I.preprocess(record.image_bytes, transform)
        labels[position] = record.label
        entries.append(record.index_entry())
    if len(entries) != len(rows):
        raise RuntimeError(f"read {len(entries)} D records, expected {len(rows)}")
    steps["read_and_preprocess_s"] = time.perf_counter() - start
    d_meta = {"repo_id": I.REPO_ID, "revision": I.REVISION, "rule": f"global row % {I.D_STRIDE} == 0",
              "n_rows": len(entries), "n_val_rows": int(sum(counts)), "manifest_sha256": I.file_sha256(
                  root / I.MANIFEST), "smoke": bool(args.smoke)}
    d_index_sha = I.write_d_index(out / "d_index.jsonl", entries, d_meta)
    np.save(work / "d_inputs.npy", x)
    np.save(work / "d_labels.npy", labels)
    log(handle, f"D read and preprocessed in {steps['read_and_preprocess_s']:.1f} s; d_index sha256 {d_index_sha}")

    # The model, its params file and forward.py.
    params = N.imagenet_params(args.weights)
    weights_sha = I.file_sha256(args.weights)
    reloaded, save_record = T.save_reload(params, work / "model.npz")
    source = N.render_forward()
    forward_sha = write_forward(out / "forward.py", source)
    forward, loaded_sha = load_forward(out / "forward.py")
    if loaded_sha != forward_sha:
        raise RuntimeError("forward.py hash changed between writing and loading")
    spec = N.build_resnet18native_spec("t03native", first_names, params)
    (out / "model_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    # forward.py: two runs on the base, one on the reloaded params.
    run1, steps["forward_base_run1_s"] = timed(forward, params, x)
    log(handle, f"forward.py run 1: {steps['forward_base_run1_s']:.1f} s")
    run2, steps["forward_base_run2_s"] = timed(forward, params, x)
    run_reloaded, steps["forward_reloaded_s"] = timed(forward, reloaded, x)
    log(handle, f"forward.py run 2 and reloaded: {steps['forward_base_run2_s']:.1f} s, {steps['forward_reloaded_s']:.1f} s")

    # Torch rebuilds of the base.
    model64 = N.build_resnet18native(params, torch.float64)
    ref64, steps["torch_float64_s"] = timed(N.eval_logits, model64, x)
    del model64
    model32 = N.build_resnet18native(params, torch.float32)
    ref32, steps["torch_float32_s"] = timed(N.eval_logits, model32, x)
    torchvision_check = M.torchvision_reference_diff(model32, x[:TORCHVISION_CHECK_ROWS])
    del model32
    log(handle, f"torch float64 {steps['torch_float64_s']:.1f} s, float32 {steps['torch_float32_s']:.1f} s")

    # The fixed test edit through the patch interface, saved and reloaded.
    edited, edit_record = T.apply_fixed_edit(params, out / "fixed_edit")
    (out / "fixed_edit" / "edited.npz").replace(work / "edited.npz")
    edit_record["save_reload"]["file"] = "work/edited.npz"
    edited_fwd, steps["forward_edited_s"] = timed(forward, edited, x)
    model64 = N.build_resnet18native(edited, torch.float64)
    edited64 = N.eval_logits(model64, x)
    del model64
    model32 = N.build_resnet18native(edited, torch.float32)
    edited32 = N.eval_logits(model32, x)
    del model32
    log(handle, "edited model done")

    # Pass items.
    item1 = T.parity_record(run1, ref64, ref32)
    edited_parity = T.parity_record(edited_fwd, edited64, edited32)
    reloaded_equal = T.bitwise_equal(run_reloaded, run1)
    delta = np.abs(edited_fwd.astype(np.float64) - run1.astype(np.float64))
    item2 = {
        "save_reload": save_record,
        "reloaded_logits_bitwise_equal": bool(reloaded_equal),
        "fixed_edit": edit_record,
        "fixed_edit_ops": [dict(op) for op in T.FIXED_EDIT_OPS],
        "edited_parity": edited_parity,
        "edit_effect": {"max_abs_logit_change": float(delta.max()),
                        "rows_with_argmax_change": int(np.count_nonzero(edited_fwd.argmax(1) != run1.argmax(1)))},
        "verdict": "pass" if (save_record["same_sha256"] and save_record["arrays_bitwise_equal"] and reloaded_equal
                              and edited_parity["verdict"] == "pass") else "fail",
    }
    same_runs = T.bitwise_equal(run1, run2)
    item3 = {"bitwise_equal": bool(same_runs), "run1_logits_sha256": array_sha256(run1),
             "run2_logits_sha256": array_sha256(run2), "verdict": "pass" if same_runs else "fail"}
    item4 = T.accuracy_record(run1, labels)
    item4["torch_float32_top1_percent"] = T.accuracy_record(ref32, labels)["top1_percent"]
    np.savez(out / "logits.npz", forward_base=run1, torch64_base=ref64, torch32_base=ref32, forward_edited=edited_fwd,
             torch64_edited=edited64, torch32_edited=edited32, labels=labels,
             global_index=np.array([e["global_index"] for e in entries], dtype=np.int64))
    log(handle, f"items: (1) {item1['verdict']} (2) {item2['verdict']} (3) {item3['verdict']} (4) {item4['verdict']} "
                f"top1 {item4['top1_percent']:.2f}")
    partial = {"note": "pass items before the cost probes; t03_summary.json supersedes this file when present",
               "items": {"1_parity": item1, "2_save_reload_and_edit": item2, "3_determinism": item3,
                         "4_accuracy": item4}, "steps_s": steps, "smoke": bool(args.smoke)}
    (out / "t03_items_partial.json").write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")

    # Cost: fresh numpy-only processes at the inherited thread settings.
    probe_script = Path(__file__).resolve().with_name("cost_probe.py")
    sizes = PROBE_SIZES if not args.smoke else (len(rows), min(SMOKE_SMALL_PROBE, len(rows)))
    probes = {}
    for n in sizes:
        probe = run_probe(Path(sys.executable), probe_script, out / "forward.py", work / "model.npz",
                          work / "d_inputs.npy", work / "d_labels.npy", n, out / f"cost_probe_{n}.json")
        probe["logits_equal_main_run"] = probe["logits_sha256"] == hashlib.sha256(
            np.ascontiguousarray(run1[:n]).tobytes()).hexdigest()
        probes[str(n)] = probe
        log(handle, f"probe n={n}: forward {probe['forward_s']:.1f} s, peak RSS {probe['vm_hwm_bytes'] / 2**30:.2f} "
                    f"GiB, logits equal the main run {probe['logits_equal_main_run']}")
    main_probe = probes[str(sizes[0])]
    per_image = main_probe["forward_s_per_image"]
    per_image_with_inputs = (main_probe["forward_s"] + main_probe["load_inputs_s"]) / main_probe["n_images"]
    fixed_overhead = main_probe["process_wall_s"] - main_probe["forward_s"] - main_probe["load_inputs_s"] - main_probe[
        "check_s"]
    n_in_wall = int(max(0.0, PROGRAM_WALL_S - fixed_overhead) // per_image_with_inputs)
    node_bytes = args.mem_gb * 1e9
    rss_limit = node_bytes / READOUTS_PER_NODE
    elapsed = time.perf_counter() - started
    cost = {
        "settings": {"ncpus": args.ncpus, "mem_gb": args.mem_gb, "walltime_s": args.walltime_s,
                     "threads": {name: os.environ.get(name) for name in THREAD_VARS},
                     "torch_threads": torch.get_num_threads()},
        "forward_s_per_image": per_image,
        "forward_s_per_image_with_input_load": per_image_with_inputs,
        "probe_peak_rss_bytes": {n: p["vm_hwm_bytes"] for n, p in probes.items()},
        "probe_peak_rss_source": "VmHWM of the probe process (/proc/self/status); its ru_maxrss is not used because "
                                 "Linux carries the parent's peak across exec into it",
        "probe_vm_peak_bytes": {n: p["vm_peak_bytes"] for n, p in probes.items()},
        "check_images": main_probe["n_images"],
        "check_forward_s": main_probe["forward_s"],
        "check_process_wall_s": main_probe["process_wall_s"],
        "probe_logits_equal_main_run": {n: p["logits_equal_main_run"] for n, p in probes.items()},
        "program_wall_s": PROGRAM_WALL_S,
        "program_fixed_overhead_s": fixed_overhead,
        "evidence_images_in_program_wall": n_in_wall,
        "evidence_rule": "floor((PROGRAM_WALL_S - fixed overhead) / per-image time with input load), from the "
                         f"{main_probe['n_images']}-image probe; fixed overhead = interpreter start, numpy import, "
                         "forward.py and params load",
        "params_file_bytes": save_record["file_bytes"],
        "main_process_peak_rss_bytes": peak_rss_bytes(),
        "main_forward_runs_s": {k: v for k, v in steps.items() if k.startswith("forward_")},
        "torch_s": {"float64": steps["torch_float64_s"], "float32": steps["torch_float32_s"]},
        "sandbox_memory_limit_bytes": SANDBOX_MEMORY_BYTES,
        "sandbox_cell_timeout_s": SANDBOX_CELL_TIMEOUT_S,
        "readout_rss_limit_bytes": rss_limit,
        "fit_rule": f"a forward.py readout at {args.threads} threads has peak RSS <= the job's memory / "
                    f"{READOUTS_PER_NODE} (two readouts per {args.ncpus}-core node), and the run finished inside its "
                    "walltime; the 300 s and sandbox numbers inform T04 and do not decide T03",
        "elapsed_s_at_summary": elapsed,
    }
    rss_fits = all(p["vm_hwm_bytes"] <= rss_limit for p in probes.values())
    time_fits = elapsed < args.walltime_s
    cost["readout_rss_fits"] = bool(rss_fits)
    cost["walltime_fits"] = bool(time_fits)
    cost["verdict"] = "fits" if (rss_fits and time_fits) else "does not fit"
    cost["t04_flags"] = {
        "check_forward_within_program_wall": bool(main_probe["forward_s"] <= PROGRAM_WALL_S),
        "probe_vm_peak_within_sandbox_limit": bool(all(p["vm_peak_bytes"] <= SANDBOX_MEMORY_BYTES
                                                       for p in probes.values())),
        "probe_rss_within_sandbox_limit": bool(all(p["vm_hwm_bytes"] <= SANDBOX_MEMORY_BYTES
                                                   for p in probes.values())),
    }

    items = {"1_parity": item1, "2_save_reload_and_edit": item2, "3_determinism": item3, "4_accuracy": item4}
    failed = [name for name, item in items.items() if item["verdict"] != "pass"]
    bottlenecks = failed + ([] if cost["verdict"] == "fits" else ["cost"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "task": "PREREG T03, the native ImageNet interface and cost precheck",
        "smoke": bool(args.smoke),
        "outcome": "pass" if not bottlenecks else "pause",
        "bottlenecks": bottlenecks,
        "pass_items": items,
        "cost": cost,
        "data": {**d_meta, "d_index_file": "d_index.jsonl", "d_index_sha256": d_index_sha,
                 "d_inputs_sha256": array_sha256(x), "d_labels_sha256": array_sha256(labels),
                 "shard_row_counts": dict(zip(shards, counts)), "classes": class_order,
                 "preprocessing": repr(transform), "label_assumption": "HF label index = torchvision class index"},
        "model": {"arch": N.ARCH, "weights_file": args.weights.name, "weights_sha256": weights_sha,
                  "weights_sha256_expected": M.WEIGHTS_SHA256, "params_sha256": save_record["params_sha256"],
                  "n_parameters": int(sum(v.size for v in params.values())), "n_tensors": len(params),
                  "forward_sha256": forward_sha, "forward_chunk": 16, "spec_schema": N.SPEC_SCHEMA,
                  "torchvision_check": torchvision_check},
        "steps_s": steps,
        "run": {"code_commit": args.code_commit, "job_id": args.job_id, "host": socket.gethostname(),
                "started": datetime.datetime.fromtimestamp(started_wall).astimezone().isoformat(timespec="seconds"),
                "elapsed_s": time.perf_counter() - started, "argv": sys.argv},
        "versions": versions(),
    }
    (out / "t03_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "t03_summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    log(handle, f"outcome {summary['outcome']} bottlenecks {bottlenecks}")
    handle.close()


def summary_markdown(s: dict[str, Any]) -> str:
    """A short human-readable view of the summary (the JSON is the record)."""
    i1, i2, i3, i4 = (s["pass_items"][k] for k in ("1_parity", "2_save_reload_and_edit", "3_determinism",
                                                      "4_accuracy"))
    c = s["cost"]
    gib = 2**30
    lines = [
        "# T03 native ImageNet precheck" + (" (SMOKE, synthetic data)" if s["smoke"] else ""),
        "",
        f"Outcome: **{s['outcome']}**" + (f" (bottlenecks: {', '.join(s['bottlenecks'])})" if s["bottlenecks"] else ""),
        f"Job {s['run']['job_id'] or 'none'} on {s['run']['host']}, code {s['run']['code_commit']}, "
        f"elapsed {s['run']['elapsed_s']:.0f} s.",
        f"Weights sha256 {s['model']['weights_sha256']}; params_sha256 {s['model']['params_sha256']}; "
        f"forward.py sha256 {s['model']['forward_sha256']}.",
        f"D: {s['data']['n_rows']} rows ({s['data']['rule']}), d_index.jsonl sha256 {s['data']['d_index_sha256']}.",
        "",
        "## Pass items",
        "",
        f"1. Parity on D, {i1['verdict']}: float64 max rel diff {i1['float64_max_rel_diff']:.3g} (bound "
        f"{i1['float64_tolerance']:.3g}); float32 max abs diff {i1['max_abs_diff']:.3g} (bound {i1['tolerance']:.3g}); "
        f"float32 argmax disagreements {i1['float32_argmax_disagreements']}.",
        f"2. Save/reload and fixed edit, {i2['verdict']}: same params_sha256 {i2['save_reload']['same_sha256']}, "
        f"reloaded logits bitwise equal {i2['reloaded_logits_bitwise_equal']}; edit n_changed "
        f"{i2['fixed_edit']['n_changed']}, edited parity {i2['edited_parity']['verdict']} (float64 max rel diff "
        f"{i2['edited_parity']['float64_max_rel_diff']:.3g}, float32 max abs diff "
        f"{i2['edited_parity']['max_abs_diff']:.3g}, bound {i2['edited_parity']['tolerance']:.3g}); max logit change "
        f"from the edit {i2['edit_effect']['max_abs_logit_change']:.3g}.",
        f"3. Determinism, {i3['verdict']}: two runs bitwise equal {i3['bitwise_equal']}.",
        f"4. Top-1 on D, {i4['verdict']}: {i4['top1_percent']:.1f}% ({i4['n_correct']}/{i4['rows']}; band "
        f"{i4['band'][0]:.3f} to {i4['band'][1]:.3f}; SE {i4['binomial_se_percent']:.2f}); top-5 "
        f"{i4['top5_percent']:.1f}%.",
        "",
        "## Cost (" + f"{c['settings']['ncpus']} cores, {c['settings']['threads']['OMP_NUM_THREADS']} threads)",
        "",
        f"- forward.py per image: {c['forward_s_per_image']:.3f} s; a {c['check_images']}-image check: "
        f"{c['check_forward_s']:.1f} s forward, {c['check_process_wall_s']:.1f} s process wall (fresh numpy-only "
        "process).",
        "- peak RSS per probe (GiB): " + ", ".join(f"{n} images {v / gib:.2f}" for n, v in
                                                     c["probe_peak_rss_bytes"].items()) + "; VmPeak (GiB): " +
        ", ".join(f"{n} images {v / gib:.2f}" for n, v in c["probe_vm_peak_bytes"].items()) + ".",
        f"- evidence images a program can forward in {c['program_wall_s']:.0f} s: {c['evidence_images_in_program_wall']} "
        f"(fixed overhead {c['program_fixed_overhead_s']:.1f} s).",
        f"- params file: {c['params_file_bytes']} bytes.",
        f"- cost verdict: {c['verdict']} ({c['fit_rule']}).",
        f"- T04 flags: {json.dumps(c['t04_flags'])}.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
