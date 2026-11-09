"""PREREG E14: the store record, the bot demonstrations, the student chain and its import (``weightcode.data.e14``,
``weightcode.models.e14_policy``, ``weightcode.models.e14_export``).

Subcommands:

* ``init`` (workstation, once): the store's forward.py, the sealed seed record (written once to
  ``$WEIGHTCODE_SEALED/e14_gotolocal/test.npz``; only its hash is printed) and dataset.json.
* ``demos`` (NSCC): the full-map bot's demonstrations of the 100,000 train seeds (old mission) and of the 512 new
  V_evidence seeds, in ``--workers`` processes, to ``$WEIGHTCODE_PRIVATE/e14_gotolocal/demos/`` with demos.json (counts,
  canonical sha256, the installed versions and the initial-state fingerprints of every dev episode set, so the
  workstation can confirm the same episodes).
* ``chain`` (NSCC): PREREG E14's order (``weightcode.models.e14_export.CHAIN``): train seed ``--seed`` at "base" on 50,000
  episodes, read the gate with forward.py, and stop at the first student that passes; otherwise 100,000 episodes, then
  the "next" size. Each student goes to the pending directory with its gate and headroom readings. ``--device cuda``
  trains on the GPU (PREREG E14 device amendment, main 1f1cbf7) and records the device, CUDA and cuDNN versions in
  each student's determinism block; data and the forward.py rollouts stay on the CPU.
* ``import-demos`` (workstation, once): the fetched run's demonstrations into ``$WEIGHTCODE_PRIVATE/e14_gotolocal/demos/``
  after the hash, bot-regeneration and fingerprint checks (``cmd_import_demos``).
* ``import`` (workstation, after import-demos): ``weightcode.models.e14_export.import_e14_models`` on a fetched run
  directory.

``--smoke`` (demos and chain): 2,000 train episodes and 64 V_evidence episodes, one epoch, rollouts timed on the smoke
seed ranges only (no V_select episode is run); every output is marked smoke and the import refuses it.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK = 2_000
SMOKE_TRAIN = 2_000
SMOKE_EVIDENCE = 64


def set_threads(threads: int) -> None:
    for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
        os.environ[variable] = str(threads)


def cmd_init(args: argparse.Namespace) -> int:
    from weightcode.data import e14 as D
    from weightcode.data.store import write_json
    from weightcode.models import e14_export as X

    record_path = D.store_dir() / "dataset.json"
    if record_path.exists():
        raise SystemExit(f"{record_path} exists; init runs once")
    forward_sha = X.write_store_forward()
    sealed = D.write_sealed_record()
    record = X.dataset_record(D.store_dir() / "wheels", sealed)
    if record["forward_sha256"] != forward_sha:
        raise RuntimeError("forward.py hash differs from the record")
    write_json(record_path, record)
    print(json.dumps({"dataset_json": str(record_path), "forward_sha256": forward_sha,
                      "sealed_record_sha256": sealed["sha256"]}, indent=1))
    return 0


def _chunks(mission: str, start: int, stop: int) -> list[tuple[str, int, int]]:
    return [(mission, a, min(a + CHUNK, stop)) for a in range(start, stop, CHUNK)]


def cmd_demos(args: argparse.Namespace) -> int:
    import multiprocessing

    import numpy as np

    from weightcode.data import e14 as D
    from weightcode.data.store import write_json

    started = time.perf_counter()
    n_train = SMOKE_TRAIN if args.smoke else D.SEED_RANGES["train"].n
    n_evidence = SMOKE_EVIDENCE if args.smoke else D.SEED_RANGES["V_evidence_new"].n
    sets = {"train": ("old", D.SEED_RANGES["train"].start, n_train),
            "V_evidence_new": ("new", D.SEED_RANGES["V_evidence_new"].start, n_evidence)}
    manifest: dict[str, Any] = {"smoke": bool(args.smoke), "versions": D.environment_versions(), "sets": {},
                                "workers": args.workers}
    context = multiprocessing.get_context("spawn")
    with context.Pool(args.workers) as pool:
        for name, (mission, start, n) in sets.items():
            t0 = time.perf_counter()
            chunks = pool.map(D.demo_chunk, _chunks(mission, start, start + n))
            arrays = D.merge_demo_chunks(chunks)
            if not np.array_equal(arrays["ep_seed"], np.arange(start, start + n)):
                raise RuntimeError(f"{name}: the episodes are not the seeds {start}..{start + n - 1} in order")
            digest = D.save_demos(name, arrays)
            manifest["sets"][name] = {"file": D.DEMO_FILES[name], "sha256": digest, "mission": mission,
                                      "seed_start": start, "n_seeds": n, **D.demo_summary(arrays),
                                      "wall_s": time.perf_counter() - t0}
            print(name, json.dumps(manifest["sets"][name]), flush=True)
    names = ("smoke_old", "smoke_new") if args.smoke else (
        "V_check_old", "V_select_old", "V_check_new", "V_select_new")
    manifest["fingerprints"] = {name: D.initial_fingerprint(D.SEED_RANGES[name].mission, D.SEED_RANGES[name].seeds)
                                for name in names}
    manifest["fingerprints"]["train_first_1000"] = D.initial_fingerprint("old", D.SEED_RANGES["train"].seeds[:1000])
    manifest["wall_s"] = time.perf_counter() - started
    write_json(D.demos_dir() / D.DEMOS_MANIFEST, manifest)
    print(json.dumps({"fingerprints": manifest["fingerprints"], "wall_s": manifest["wall_s"]}), flush=True)
    return 0


def _job_forward() -> Any:
    from weightcode.evalkit.patch_check import load_forward
    from weightcode.models import e14_policy as P

    sys.dont_write_bytecode = True
    path = Path(tempfile.mkdtemp()) / "forward.py"
    path.write_text(P.forward_source(), encoding="utf-8")
    return load_forward(path)[0]


def _smoke_readout(forward: Any, params: Any) -> dict[str, Any]:
    from weightcode.data import e14 as D
    from weightcode.models import e14_policy as P

    out: dict[str, Any] = {}
    for mission in D.MISSIONS:
        t0 = time.perf_counter()
        result = D.rollouts(P.numpy_logits_fn(forward, params), mission, D.SEED_RANGES[f"smoke_{mission}"].seeds)
        out[f"smoke_{mission}_wall_s"] = time.perf_counter() - t0
        out[f"smoke_{mission}_mean_length"] = float(result.length.mean())
    out["gate"] = {"passed": False, "note": "smoke: no V_select episode is run"}
    return out


def _device_record(torch: Any, device: str) -> dict[str, Any]:
    """The training device for a student's determinism block (PREREG E14 device amendment, main 1f1cbf7)."""
    if device == "cpu":
        return {"device": "cpu"}
    if not torch.cuda.is_available():
        raise SystemExit("--device cuda: torch sees no CUDA device")
    return {"device": "cuda", "gpu_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "n_visible_gpus": torch.cuda.device_count()}


def cmd_chain(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    torch.backends.mkldnn.enabled = args.mkldnn == "on"
    from weightcode.data import e14 as D
    from weightcode.data.store import write_json
    from weightcode.models import e14_export as X
    from weightcode.models import e14_policy as P
    from weightcode.models.runinfo import run_info
    from weightcode.models.train import configure_determinism

    manifest = json.loads((D.demos_dir() / D.DEMOS_MANIFEST).read_text(encoding="utf-8"))
    if bool(manifest["smoke"]) != bool(args.smoke):
        raise SystemExit("the demonstrations' smoke flag differs from --smoke")
    demos = D.load_demos("train", manifest["sets"]["train"]["sha256"])
    forward = _job_forward()
    info = run_info(REPO_ROOT, args.git_state_file)
    root = X.pending_models_dir() if args.models_root is None else Path(args.models_root)
    rows: list[dict[str, Any]] = []
    chain = ((("base", SMOKE_TRAIN),) if args.smoke else X.CHAIN)
    config = P.E14TrainConfig(epochs=1) if args.smoke else P.E14TrainConfig()
    for size, n_episodes in chain:
        if args.smoke:
            keep = demos["episode"] < n_episodes
            x, y = demos["x"][keep].astype(np.int64), demos["y"][keep].astype(np.int64)
        else:
            x, y = D.train_rows(demos, n_episodes)
        determinism = {**configure_determinism(args.seed), "torch_threads": torch.get_num_threads(),
                       "mkldnn_enabled": torch.backends.mkldnn.enabled, **_device_record(torch, args.device)}
        print(f"== {size} at {n_episodes} episodes: {len(x)} rows, {args.threads} threads, device {args.device}",
              flush=True)
        result = P.train_policy(config, P.SIZES[size], x, y, seed=args.seed, device=args.device)
        params = P.export_params(result.model)
        t0 = time.perf_counter()
        gate = _smoke_readout(forward, params) if args.smoke else X.readout(forward, params)
        gate["wall_s"] = time.perf_counter() - t0
        demo_record = {"train_demos_sha256": manifest["sets"]["train"]["sha256"],
                       "demos_manifest_versions": manifest["versions"]}
        row = X.export_pending(seed=args.seed, size=size, n_episodes=n_episodes, config=config, result=result,
                               demos=demo_record, x_train=x, gate=gate, determinism=determinism, run_info=info,
                               root=root)
        if args.smoke:
            private_path = root / row["model_id"] / "private.json"
            private = json.loads(private_path.read_text(encoding="utf-8"))
            write_json(private_path, {**private, "smoke": True})
        rows.append(row)
        write_json(root / "index.json", rows)
        print(json.dumps({**row, "gate": gate.get("gate"), "headroom": gate.get("headroom"),
                          "train_wall_s": result.wall_time_s}), flush=True)
        if gate["gate"]["passed"]:
            break
    else:
        if not args.smoke:
            print("no student passed the gate: PREREG E14 is reported and not run", flush=True)
    return 0


LOCAL_TRAIN_EPISODES = 2_000


def cmd_import_demos(args: argparse.Namespace) -> int:
    """Install a fetched run's demonstrations in the workstation store after three checks: each file hashes to the
    job's demos.json; the workstation's bot gives the same arrays on every new V_evidence episode and on the first
    ``LOCAL_TRAIN_EPISODES`` train episodes; and every recorded initial-state fingerprint is the workstation's."""
    import shutil

    import numpy as np

    from weightcode.data import e14 as D
    from weightcode.data.store import write_json

    source = Path(args.run_dir) / "private" / D.DATASET / "demos"
    manifest = json.loads((source / D.DEMOS_MANIFEST).read_text(encoding="utf-8"))
    if manifest["smoke"]:
        raise SystemExit("a smoke run's demonstrations never enter the store")
    if (D.demos_dir() / D.DEMOS_MANIFEST).exists():
        raise SystemExit(f"{D.demos_dir()} already holds demonstrations; nothing overwritten")
    local_versions = D.environment_versions()
    if manifest["versions"] != local_versions:
        raise SystemExit(f"job versions {manifest['versions']} differ from the workstation's {local_versions}")
    checks: dict[str, Any] = {}
    for name, entry in manifest["sets"].items():
        arrays = D.load_demo_file(source / entry["file"])
        if D.demos_sha256(arrays) != entry["sha256"]:
            raise SystemExit(f"{name}: {entry['file']} does not hash to the job's demos.json")
        start = int(entry["seed_start"])
        n_local = int(entry["n_seeds"]) if name == "V_evidence_new" else LOCAL_TRAIN_EPISODES
        local = D.demo_chunk((entry["mission"], start, start + n_local))
        keep_rows = arrays["episode"] < n_local
        for key in D.DEMO_KEYS:
            job = arrays[key][keep_rows] if len(arrays[key]) == len(arrays["episode"]) else arrays[key][:n_local]
            if not np.array_equal(job, local[key]):
                raise SystemExit(f"{name}: the workstation bot differs from the job's on key {key}")
        checks[name] = {"episodes_regenerated": n_local, "equal": True}
    for name, digest in manifest["fingerprints"].items():
        if name == "train_first_1000":
            ours = D.initial_fingerprint("old", D.SEED_RANGES["train"].seeds[:1000])
        else:
            ours = D.initial_fingerprint(D.SEED_RANGES[name].mission, D.SEED_RANGES[name].seeds)
        if ours != digest:
            raise SystemExit(f"fingerprint {name}: workstation {ours} != job {digest}")
    checks["fingerprints_equal"] = sorted(manifest["fingerprints"])
    D.demos_dir().mkdir(parents=True, exist_ok=True)
    for entry in manifest["sets"].values():
        shutil.copyfile(source / entry["file"], D.demos_dir() / entry["file"])
    write_json(D.demos_dir() / D.DEMOS_MANIFEST,
               {**manifest, "import": {"run_dir": str(args.run_dir), "note": args.note, "checks": checks}})
    for name, entry in manifest["sets"].items():
        D.load_demos(name, entry["sha256"])
    print(json.dumps({"installed": str(D.demos_dir()), "checks": checks}, indent=1))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Diagnostic (not a result): train the base size from scratch for ``--steps`` steps on the smoke train
    demonstrations at ``--threads``, with oneDNN (mkldnn) and the deterministic-algorithm switch on or off, and print
    the torch build and parallel settings and the time per step. Run after ``demos --smoke``; the student is dropped."""
    import math

    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    torch.backends.mkldnn.enabled = args.mkldnn == "on"
    from weightcode.data import e14 as D
    from weightcode.models import e14_policy as P
    from weightcode.models.train import configure_determinism

    if args.deterministic == "on":
        configure_determinism(args.seed)
    else:
        torch.manual_seed(args.seed)
    print(torch.__config__.parallel_info(), flush=True)
    print({"threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
           "mkldnn_enabled": torch.backends.mkldnn.enabled, "mkldnn_available": torch.backends.mkldnn.is_available(),
           "deterministic": torch.are_deterministic_algorithms_enabled(), "cpu_capability":
               torch.backends.cpu.get_cpu_capability()}, flush=True)
    manifest = json.loads((D.demos_dir() / D.DEMOS_MANIFEST).read_text(encoding="utf-8"))
    if not manifest["smoke"]:
        raise SystemExit("the probe runs on the smoke demonstrations only")
    demos = D.load_demos("train", manifest["sets"]["train"]["sha256"])
    x, y = demos["x"].astype(np.int64), demos["y"].astype(np.int64)
    epochs = math.ceil(args.steps / math.ceil(len(x) / P.E14TrainConfig().batch_size))
    result = P.train_policy(P.E14TrainConfig(epochs=epochs), P.SIZES["base"], x, y, seed=args.seed)
    steps = int(result.final["steps_total"])
    print(json.dumps({"probe": vars(args), "steps": steps, "wall_s": result.wall_time_s,
                      "ms_per_step": 1000 * result.wall_time_s / steps, "final_loss": result.final["train_loss"]},
                     default=str), flush=True)
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from weightcode.models import e14_export as X

    for row in X.import_e14_models(Path(args.run_dir), args.note):
        print(json.dumps(row, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    demos = sub.add_parser("demos")
    demos.add_argument("--workers", type=int, default=8)
    demos.add_argument("--smoke", action="store_true")
    chain = sub.add_parser("chain")
    chain.add_argument("--seed", type=int, default=100)
    chain.add_argument("--threads", type=int, default=16)
    chain.add_argument("--mkldnn", choices=("on", "off"), default="on")
    chain.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    chain.add_argument("--smoke", action="store_true")
    chain.add_argument("--models-root", type=Path)
    chain.add_argument("--git-state-file", type=Path)
    probe = sub.add_parser("probe")
    probe.add_argument("--threads", type=int, required=True)
    probe.add_argument("--steps", type=int, default=2000)
    probe.add_argument("--seed", type=int, default=100)
    probe.add_argument("--mkldnn", choices=("on", "off"), default="on")
    probe.add_argument("--deterministic", choices=("on", "off"), default="on")
    for name in ("import-demos", "import"):
        imp = sub.add_parser(name)
        imp.add_argument("--run-dir", required=True)
        imp.add_argument("--note", required=True)
    args = parser.parse_args(argv)
    faulthandler.enable()  # a native crash prints every thread's Python stack
    set_threads(getattr(args, "threads", 1) if args.command in ("chain", "probe") else 1)
    return {"init": cmd_init, "demos": cmd_demos, "chain": cmd_chain, "probe": cmd_probe,
            "import-demos": cmd_import_demos, "import": cmd_import}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
