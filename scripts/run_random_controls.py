"""Run B1 matched random controls (permuted, gaussian) against a reference patch per instance.

Instances: the given ``--instance`` ids, or by default every instance in ``$WEIGHTCODE_PRIVATE/instances/index.json`` with
split_role "dev" (optionally one ``--dataset``). Every selected instance's private.json split_role is checked before
anything runs: a main instance is refused (exit status 2, nothing written) unless ``--allow-main`` is given, which is
reserved for runs after the pre-registration freeze (PREREG section 2).

The reference is a run in the shared run-record format (a directory holding run.json, or the run.json): a fitting run
written by run_fit_baselines.py or a harness editor session (M2, NOWEIGHT, DECOY). Its final child is rebuilt by
``weightcode.evalkit.devreadout.final_child``; a base no-op final is recorded and no control is written. A bare patch (a
patch JSON file or a directory holding patch.json) also works. With more than one instance, ``--reference-patch``
must contain the literal ``{instance_id}``, which is replaced per instance.

For a run the arm is its ``method``, the B1 session id is its ``session_id`` (a harness session) or
"<instance_id>/<arm>/<run seed>" (a fitting run), and the output label is the arm (a fitting run) or
"<arm>_<session_id>" (a harness session, which can repeat on one instance); ``--ref-label`` and ``--session-id``, if
given, must agree with them. For a bare patch both are required and the label is ``--ref-label``.

Each kind runs ``--runs`` times (default 3, run_index 0, 1, 2), each run with its own counted oracle of 8 queries
(8 seeded candidates; see ``weightcode.baselines.fit.controls``), written to
``$WEIGHTCODE_RUNS/baselines/<instance_id>/<kind>/<label>/r<run_index>/``. The B1 value of a kind is the mean final S over
its runs, printed after them. An existing run directory is never overwritten.

Example::

    source scripts/env.sh
    .venv/bin/python scripts/run_random_controls.py --dataset fashion_mnist \
        --reference-patch "$WEIGHTCODE_RUNS/baselines/{instance_id}/grad_last/0"
"""

from __future__ import annotations

import argparse
import os
import re
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
KINDS = ("permuted", "gaussian")
DEFAULT_RUNS = 3
INSTANCE_FIELD = "{instance_id}"
EXIT_FAILED_RUNS = 1
"""Exit status when a control run failed (for example every draw rejected by the executor); B1 is not reported."""
LABEL = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def reference_label(value: str) -> str:
    """argparse type: a plain path component, as weightcode.baselines.fit.runner.safe_name requires."""
    if not LABEL.fullmatch(value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError(f"{value!r} must match {LABEL.pattern} and not be '.' or '..'")
    return value


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", action="append", help="instance id (repeatable); default: select from index")
    parser.add_argument("--split-role", choices=("dev", "main"), default="dev", help="index selection (default dev)")
    parser.add_argument("--dataset", help="restrict the index selection to one dataset")
    parser.add_argument("--reference-patch", required=True, help="patch JSON, patch dir or baseline run dir")
    parser.add_argument(
        "--ref-label", type=reference_label, help="output label (default for a run: its arm, or <arm>_<session id>)"
    )
    parser.add_argument("--session-id", help="B1 session id (default for a baseline run: <instance>/<arm>/<seed>)")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="control runs per kind (run_index 0..runs-1)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for BLAS")
    parser.add_argument("--allow-main", action="store_true", help="permit main instances (only after the freeze)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
        os.environ[variable] = str(args.threads)
    sys.dont_write_bytecode = True  # importing an instance's forward.py must not write __pycache__ into $WEIGHTCODE_PRIVATE
    from weightcode.baselines.fit.cli import (
        EXIT_REFUSED,
        SelectionError,
        load_checked_instances,
        refusal_message,
        select_instance_ids,
        summary_line,
    )
    from weightcode.baselines.fit.controls import b1_value, load_reference, resolve_reference, run_control
    from weightcode.baselines.fit.runner import NoValidCandidate, SplitRoleError, control_run_dir

    try:
        ids = select_instance_ids(args.instance, split_role=args.split_role, dataset=args.dataset)
        if len(ids) > 1 and INSTANCE_FIELD not in args.reference_patch:
            raise SelectionError(f"{len(ids)} instances selected but --reference-patch lacks {INSTANCE_FIELD}")
        instances = load_checked_instances(ids, allow_main=args.allow_main)
        plan = []
        for instance in instances:
            reference = args.reference_patch.replace(INSTANCE_FIELD, instance.instance_id)
            resolved = resolve_reference(reference, instance)
            if resolved.run_record is not None:
                arm = str(resolved.arm)
                session_id = str(resolved.session_id(instance.instance_id))
                label = str(resolved.output_label())
                if not LABEL.fullmatch(label):
                    raise SelectionError(f"the reference run's output label {label!r} is not a plain path component")
                if args.ref_label not in (None, label):
                    raise SelectionError(f"--ref-label {args.ref_label!r} differs from the reference run's {label!r}")
                if args.session_id not in (None, session_id):
                    raise SelectionError(f"--session-id {args.session_id!r} differs from the run's {session_id!r}")
            elif args.ref_label is None or args.session_id is None:
                raise SelectionError("a bare patch reference needs --ref-label and --session-id")
            else:
                arm, session_id, label = args.ref_label, args.session_id, args.ref_label
            load_reference(instance, reference)  # rebuild and hash-check the reference child before any run starts
            plan.append((instance, reference, arm, session_id, label))
        existing = [
            str(control_run_dir(instance.instance_id, kind, label, run_index))
            for instance, _, _, _, label in plan
            for kind in args.kinds
            for run_index in range(args.runs)
            if control_run_dir(instance.instance_id, kind, label, run_index).exists()
        ]
        if existing:
            raise SelectionError(f"control run directories already exist and are never overwritten: {existing}")
    except (SplitRoleError, SelectionError) as error:
        print(refusal_message(error), file=sys.stderr)
        return EXIT_REFUSED
    print(f"instances: {[instance.instance_id for instance in instances]}", flush=True)
    failures = []
    for instance, reference, arm, session_id, label in plan:
        for kind in args.kinds:
            records, failed = [], []
            for run_index in range(args.runs):
                run_dir = control_run_dir(instance.instance_id, kind, label, run_index)
                try:
                    record = run_control(
                        instance,
                        kind,
                        reference,
                        session_id=session_id,
                        arm=arm,
                        run_index=run_index,
                        run_dir=run_dir,
                        allow_main=args.allow_main,
                    )
                except NoValidCandidate as error:
                    failed.append(f"r{run_index}: {error}")
                    print(f"FAILED {instance.instance_id} {kind}[{label} r{run_index}]: {error}", flush=True)
                    print(f"  -> {run_dir} (status failed)", flush=True)
                    continue
                print(summary_line(record), flush=True)
                print(f"  -> {run_dir}", flush=True)
                records.append(record)
            if failed:
                failures.append(f"{instance.instance_id} {kind}[{label}]")
                print(f"B1 {instance.instance_id} {kind}[{label}] session {session_id}: NOT REPORTED, "
                      f"{len(failed)} of {args.runs} runs failed: {failed}", flush=True)
                continue
            value = b1_value(records)
            print(
                f"B1 {instance.instance_id} {kind}[{label}] session {session_id}: final S per run "
                f"{value['final_scores']}, mean S {value['mean_score']:.3f}, mean delta_acc {value['mean_delta_acc']}, "
                f"invalid draws per run {value['n_invalid']}",
                flush=True,
            )
    if failures:
        print(f"B1 not reported for {failures}", file=sys.stderr)
        return EXIT_FAILED_RUNS
    return 0


if __name__ == "__main__":
    sys.exit(main())
