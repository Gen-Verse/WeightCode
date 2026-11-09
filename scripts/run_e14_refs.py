"""PREREG E14 references, F1 and the label-equivalent runs on the E14 dev instance, their V_select readouts and the bar.

Subcommands (run seed 0). ``--threads`` sets the BLAS threads of the numpy forward.py rollouts before numpy loads.
Torch runs at ``--torch-threads`` (default 1: PREREG E14 "Reference fits", main 83f3e77, so the CPU Adam update never
enters the OpenMP parallel region where every R23 crash sat) with torch.backends.mkldnn.enabled False (main b4d7654).

    fit --instance ID --methods M [M ...]
        one run per method (``weightcode.baselines.fit.e14refs.run_e14_ref``): 8 candidates, one rollout check each on the
        500 new and 500 old V_check episodes, written to $WEIGHTCODE_RUNS/baselines/<ID>/<M>/e14/0/ (never overwritten).
    read --instance ID --methods M [M ...] --out FILE
        ``weightcode.evalkit.e14read.read_run`` of each method's run.json: every candidate on V_select, the run's own final
        and the kept pick at S_old >= -3; one JSON line per method.
    bar --reads FILE [FILE ...] --out FILE [--wrapper FILE]
        ``e14_bar`` over the read records: best kept joint of the 13 references at equal information + 1, the
        supplementary test (>= 90% new V_select success), the label-equivalent curve and, given the wrapper record,
        the wrapper beside the bar; every pick with new success split by demonstrated (color, type).
    wrapper --instance ID --out FILE
        PREREG E14's hand-written wrapper (base plus pickup on a matching front cell) on V_select; beside the bar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _threads(n: int) -> None:
    if n < 1:
        raise SystemExit("--threads must be at least 1")
    for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
        os.environ[variable] = str(n)
    sys.dont_write_bytecode = True  # importing an instance's forward.py must not write __pycache__ into $WEIGHTCODE_PRIVATE


def _write_lines(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records))
    os.replace(tmp, path)


def _torch(n: int) -> None:
    import torch

    if n < 1:
        raise SystemExit("--torch-threads must be at least 1")
    torch.set_num_threads(n)
    torch.backends.mkldnn.enabled = False


def cmd_fit(args: argparse.Namespace) -> int:
    from weightcode.baselines.fit.cli import summary_line
    from weightcode.baselines.fit.e14refs import E14_METHODS, E14_REF_LABEL, run_e14_ref
    from weightcode.baselines.fit.runner import baseline_run_dir, check_split_role
    from weightcode.evalkit.e14 import load_e14_instance

    unknown = [m for m in args.methods if m not in E14_METHODS]
    if unknown:
        raise SystemExit(f"unknown E14 methods {unknown}; known: {E14_METHODS}")
    instance = load_e14_instance(args.instance)
    check_split_role(instance, allow_main=False)
    for method in args.methods:
        run_dir = baseline_run_dir(instance.instance_id, method, args.seed, E14_REF_LABEL)
        started = time.perf_counter()
        record = run_e14_ref(instance, method, run_dir, run_seed=args.seed)
        print(summary_line(record), f"({time.perf_counter() - started:.1f} s)", flush=True)
        print(f"  -> {run_dir}", flush=True)
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    from weightcode.baselines.fit.e14refs import E14_REF_LABEL
    from weightcode.baselines.fit.runner import baseline_run_dir
    from weightcode.evalkit.e14read import read_run

    records = []
    for method in args.methods:
        path = baseline_run_dir(args.instance, method, args.seed, E14_REF_LABEL) / "run.json"
        started = time.perf_counter()
        record = read_run(path, args.floor)
        records.append(record)
        own, kept = record["own"], record["kept"]
        print(f"{method}: own {own['patch_id'] or 'base'} gain {own['gain']:+.1f} clean {own['clean']:+.1f} joint "
              f"{own['joint']:+.1f}; kept {kept['patch_id'] or 'base'} gain {kept['gain']:+.1f} clean "
              f"{kept['clean']:+.1f} joint {kept['joint']:+.1f} (new success {kept['success_new']:.3f}); "
              f"{time.perf_counter() - started:.1f} s", flush=True)
    _write_lines(args.out, records)
    return 0


def cmd_bar(args: argparse.Namespace) -> int:
    from weightcode.baselines.fit.e14refs import BC_METHODS, F1_METHODS, LABELEQ_METHODS
    from weightcode.evalkit.e14read import e14_bar

    records = {}
    for path in args.reads:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            if record["method"] in records:
                raise SystemExit(f"method {record['method']} read twice")
            records[record["method"]] = record
    wrapper = None
    if args.wrapper is not None:
        lines = args.wrapper.read_text().splitlines()
        if len(lines) != 1:
            raise SystemExit(f"{args.wrapper}: expected one wrapper record, found {len(lines)} lines")
        wrapper = json.loads(lines[0])
    bar = e14_bar(records, BC_METHODS, F1_METHODS, LABELEQ_METHODS, wrapper=wrapper)
    _write_lines(args.out, [bar])
    print(json.dumps(bar, indent=1, sort_keys=True))
    return 0


def cmd_wrapper(args: argparse.Namespace) -> int:
    from weightcode.evalkit.e14 import e14_wrapper_readout

    record = {"instance_id": args.instance, "label": "PREREG E14 hand-written wrapper, beside the bar (main d744142)",
              "readout": e14_wrapper_readout(args.instance)}
    _write_lines(args.out, [record])
    ro = record["readout"]
    print(f"wrapper: new success {ro['new']['success_child']:.3f} (base {ro['new']['success_base']:.3f}), old "
          f"{ro['old']['success_child']:.3f} (base {ro['old']['success_base']:.3f}), joint {ro['joint']:+.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "read", "wrapper"):
        p = sub.add_parser(name)
        p.add_argument("--instance", required=True)
        p.add_argument("--threads", type=int, default=4, help="BLAS threads of the numpy rollouts")
        p.add_argument("--torch-threads", type=int, default=1, help="torch threads (fits)")
        if name != "wrapper":
            p.add_argument("--methods", nargs="+", required=True)
            p.add_argument("--seed", type=int, default=0)
        if name != "fit":
            p.add_argument("--out", type=Path, required=True)
    sub.choices["read"].add_argument("--floor", type=int, default=-3)
    bar = sub.add_parser("bar")
    bar.add_argument("--reads", nargs="+", required=True)
    bar.add_argument("--out", type=Path, required=True)
    bar.add_argument("--wrapper", type=Path, default=None,
                     help="the wrapper record (wrapper --out) to put beside the bar")
    args = parser.parse_args(argv)
    if args.command != "bar":
        _threads(args.threads)
        _torch(args.torch_threads)
    return {"fit": cmd_fit, "read": cmd_read, "bar": cmd_bar, "wrapper": cmd_wrapper}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
