"""Run the PREREG E13 references, the rule-label check and the walk diagnostic on scanskill-v1 instances.

References (``weightcode.baselines.fit.scanrefs``): E5's seqgrad set and F1's eight configurations on the experience rows,
and the rule-generated references (fit sets ``full`` and ``mixed``), seed 0, 8 checks each by the scan joint oracle.
Each writes ``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/<label>/0/`` (label ``f1`` for F1's eight, ``e13`` for the
rest); an existing run directory is never overwritten.

Three modes::

    source scripts/env.sh
    # local only (reads $WEIGHTCODE_PRIVATE/scan/raw/tasks.txt, which never goes to NSCC): every rule-generated row's
    # labels against the official file; writes the record the reference runs check their fit sets against
    .venv/bin/python scripts/run_scan_refs.py check-labels --out scripts/store/e13_rulegen_labels.json
    # references (NSCC GPU main of R22-e13, or a smoke with --methods); --device cuda fits on the GPU with
    # deterministic algorithms and IEEE float32 matmuls, the checks stay on the CPU at --threads
    .venv/bin/python scripts/run_scan_refs.py refs --instance nXXXXXXXX [--methods M ...] \
        --label-record scripts/store/e13_rulegen_labels.json --threads 1 --device cuda
    # the fixed walk diagnostic (addprim_jump only)
    .venv/bin/python scripts/run_scan_refs.py walk --instance nXXXXXXXX --out FILE

Dev readouts are a separate step: the run's final, the kept pick (S_old >= -5 of 1000) and, for F1, every checked
configuration::

    .venv/bin/python -m weightcode.evalkit.devreadout RUN.json [...] --out FILE
    .venv/bin/python -m weightcode.evalkit.devreadout RUN.json [...] --retention-floor -5 --out FILE
    .venv/bin/python -m weightcode.evalkit.devreadout RUN.json [...] --all-candidates --out FILE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
LABEL_RECORD_SCHEMA = "weightcode.e13_rulegen_labels.v1"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("check-labels", "refs", "walk"))
    parser.add_argument("--instance", help="scanskill instance id n<8 hex> (refs, walk)")
    parser.add_argument("--methods", nargs="+", help="refs: these references only (default: every one of the condition)")
    parser.add_argument("--label-record", type=Path, help="refs: the check-labels record of the rule-generated rows")
    parser.add_argument("--seed", type=int, default=0, help="run seed (torch seed; directory name)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch and BLAS")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="refs: the fit device")
    parser.add_argument("--out", type=Path, help="check-labels, walk: write the JSON record here")
    return parser.parse_args(argv)


def _write(path: Path | None, record: dict) -> None:
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(text, end="")
        return
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
        os.environ[variable] = str(args.threads)
    if args.device == "cuda":  # before torch initialises CUDA (deterministic cuBLAS)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.dont_write_bytecode = True  # importing an instance's forward.py must not write __pycache__ into $WEIGHTCODE_PRIVATE
    import torch

    torch.set_num_threads(args.threads)
    from weightcode.baselines.fit import scanrefs as R
    from weightcode.baselines.fit.cli import EXIT_REFUSED, summary_line
    from weightcode.baselines.fit.runner import SplitRoleError, check_split_role
    from weightcode.data import scan as S
    from weightcode.evalkit.scanskill import load_scan_instance, scan_instance_dir

    if args.mode == "check-labels":
        from weightcode.data.scan_build import raw_dir

        tasks = raw_dir() / S.ALL_FILE
        record = {"schema": LABEL_RECORD_SCHEMA, "fits": {}}
        for condition in S.CONDITIONS:
            for fit in R.RULEGEN_FITS.values():
                generated = R.rulegen_fit_set(condition, fit)
                checked = R.check_rows_against_official(tasks, generated["commands"], generated["y"])
                record["fits"][f"{condition}/{fit}"] = {**generated["record"], "official_check": checked}
        _write(args.out, record)
        return 0

    if not args.instance:
        raise SystemExit(f"{args.mode} needs --instance")
    instance = load_scan_instance(args.instance)
    try:
        check_split_role(instance, allow_main=False)
    except SplitRoleError as error:
        print(f"refused: {error}\nNothing was run.", file=sys.stderr)
        return EXIT_REFUSED
    condition = str(instance.private["condition"])
    if args.mode == "walk":
        _write(args.out, R.walk_diagnostic(instance))
        return 0

    methods = list(args.methods) if args.methods else list(R.condition_methods(condition))
    unknown = [m for m in methods if m not in R.condition_methods(condition)]
    if unknown:
        print(f"refused: {unknown} are not E13 references of {condition}\nNothing was run.", file=sys.stderr)
        return EXIT_REFUSED
    fits = {}
    if any(m in R.RULEGEN_METHODS for m in methods):
        if args.label_record is None:
            raise SystemExit("rule-generated references need --label-record (the local check-labels record)")
        labels = json.loads(args.label_record.read_text())
        if labels.get("schema") != LABEL_RECORD_SCHEMA:
            raise SystemExit(f"{args.label_record}: schema {labels.get('schema')!r} != {LABEL_RECORD_SCHEMA}")
        for fit in sorted({R.RULEGEN_FITS[R.rulegen_parts(m)[0]] for m in methods if m in R.RULEGEN_METHODS}):
            generated = R.rulegen_fit_set(condition, fit)
            checked = labels["fits"][f"{condition}/{fit}"]
            if (checked["commands_sha256"] != generated["record"]["commands_sha256"]
                    or checked["official_check"]["mismatches"] != 0):
                raise SystemExit(f"{condition}/{fit}: the generated fit set is not the one checked against tasks.txt")
            generated["record"]["official_check"] = checked["official_check"]
            fits[fit] = generated
    settings = R.configure_device(args.device)
    print(f"instance {instance.instance_id} ({condition}, {scan_instance_dir(instance.instance_id)}); methods {methods}; "
          f"device {json.dumps(settings, sort_keys=True)}", flush=True)
    for method in methods:
        fit_set = fits[R.RULEGEN_FITS[R.rulegen_parts(method)[0]]] if method in R.RULEGEN_METHODS else None
        record = R.run_reference(instance, method, run_seed=args.seed, fit_set=fit_set, device=args.device)
        print(summary_line(record), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
