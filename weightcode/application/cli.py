"""Public CLI. Optional model dependencies load only for model operations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


BWRAP_HELP = ("bubblewrap binary, 0.8.0 or newer (default: $WEIGHTCODE_BWRAP or /usr/bin/bwrap)")


def parser() -> argparse.ArgumentParser:
    from weightcode.sandbox.host import DEFAULT_BWRAP

    p = argparse.ArgumentParser(prog="weightcode", description="Install and reuse executable neural weight programs.")
    p.add_argument("--threads", type=int, default=4, help="CPU/BLAS threads (default: 4)")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Test the actual Linux/NumPy sandbox")
    doctor.add_argument("--sandbox-python", type=Path, required=True)
    doctor.add_argument("--bwrap", type=Path, default=DEFAULT_BWRAP, help=BWRAP_HELP)
    model = sub.add_parser("import-model", help="Import a plain FP32 torchvision ResNet-18 state_dict")
    model.add_argument("--checkpoint", type=Path, required=True)
    model.add_argument("--labels", type=Path, required=True, help="JSON ordered list of class names")
    model.add_argument("--out", type=Path, required=True)
    data = sub.add_parser("prepare-data", help="Convert chosen class-folder images to uint8 crops; never split or sample")
    for name in ("model", "new", "old", "out"):
        data.add_argument(f"--{name}", type=Path, required=True)
    task = sub.add_parser("prepare-task", help="Bind independent experience, selection and final datasets")
    for name in ("model", "experience", "selection", "final", "out"):
        task.add_argument(f"--{name}", type=Path, required=True)
    task.add_argument("--goal", required=True)
    task.add_argument("--retention-drop", type=float, required=True, help="Maximum old-accuracy loss, fraction [0,1]")
    task.add_argument("--min-new-gain", type=float, required=True, help="Minimum new-accuracy gain, fraction [0,1]")
    pack = sub.add_parser("pack-program", help="Package source with an explicit model compatibility contract")
    for name in ("program", "model", "out"):
        pack.add_argument(f"--{name}", type=Path, required=True)
    for name in ("apply", "synthesize"):
        run = sub.add_parser(name, help="Reuse a program" if name == "apply" else "Synthesize using controlled Claude Code")
        for field in ("task", "out"):
            run.add_argument(f"--{field}", type=Path, required=True)
        run.add_argument("--sandbox-python", type=Path, required=name == "synthesize",
                         help="Sandbox venv interpreter: programs run inside bubblewrap" + (
                             "" if name == "synthesize" else
                             ". Without it, --program-sha256 is required and the program runs unsandboxed"))
        run.add_argument("--bwrap", type=Path, default=DEFAULT_BWRAP, help=BWRAP_HELP)
        run.add_argument("--wall-s", type=float, required=True, help="Editor/apply wall budget; independent final validation follows")
        run.add_argument("--program-wall-s", type=float, default=300)
        if name == "apply":
            run.add_argument("--program", type=Path, required=True, help="Program bundle directory")
            run.add_argument("--program-sha256", default=None,
                             help="sha256 of the bundle's program.py that you trust; checked before it runs. Required "
                                  "without --sandbox-python")
        else:
            run.add_argument("--editor-model", required=True)
            run.add_argument("--checks", type=int, required=True)
            run.add_argument("--cells", type=int, required=True)
    from weightcode.examples import EXAMPLES
    rep = sub.add_parser("reproduce", help="Reproduce a paper example from public data: no API key, no language model")
    rep.add_argument("example", choices=sorted(EXAMPLES))
    rep.add_argument("--out", type=Path, default=None, help="Output directory (default: out/<example>)")
    rep.add_argument("--sandbox-python", type=Path, default=None,
                     help="Run the published program inside the bubblewrap sandbox with this interpreter. Default: "
                          "run it without the sandbox, after checking its published sha256")
    rep.add_argument("--bwrap", type=Path, default=DEFAULT_BWRAP, help=BWRAP_HELP + "; with --sandbox-python")
    rep.add_argument("--scan-dir", type=Path, default=None,
                     help="scan: an existing checkout of github.com/brendenlake/SCAN instead of downloading its files")
    rep.add_argument("--images", type=int, default=None,
                     help="imagenet: how many held-out test images to read, the first in row order (default 500)")
    return p


def reproduce(a: argparse.Namespace) -> int:
    """``weightcode reproduce``: run one example of ``weightcode.examples``; exit 0 only if it matches the paper."""
    import importlib
    import sys

    from weightcode.examples import EXAMPLES
    from weightcode.sandbox.program import ProgramSandbox, UnisolatedRunner

    module = importlib.import_module(EXAMPLES[a.example])
    memory_bytes, threads = 8 * 2**30, 4  # the program sandbox of the sessions of record
    if a.sandbox_python is None:
        runner = UnisolatedRunner(python=Path(sys.executable), memory_bytes=memory_bytes, threads=threads)
        if getattr(module, "RERUNS_PROGRAM", False):
            print("Running the published program without the sandbox (its sha256 is checked first). "
                  "Pass --sandbox-python to run it inside bubblewrap.")
    else:
        runner = ProgramSandbox(python=a.sandbox_python, memory_bytes=memory_bytes, threads=threads, bwrap=a.bwrap)
    if a.scan_dir is not None and a.example != "scan":
        raise ValueError("--scan-dir applies to the scan example only")
    if a.images is not None and a.example != "imagenet":
        raise ValueError("--images applies to the imagenet example only")
    options: dict[str, object] = {}
    if a.scan_dir is not None:
        options["scan_dir"] = a.scan_dir
    if a.images is not None:
        options["images"] = a.images
    summary = module.run(a.out or Path("out") / a.example, runner, **options)
    return 0 if summary.get("matches_paper") else 1


def main(argv=None) -> int:
    p = parser()
    a = p.parse_args(argv)
    if a.threads < 1:
        p.error("--threads must be positive")
    if a.command == "apply" and a.sandbox_python is None and a.program_sha256 is None:
        p.error("apply without --sandbox-python needs --program-sha256: the sha256 of the program.py you trust")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(a.threads)
    try:
        if a.command == "reproduce":
            return reproduce(a)
        if a.command == "doctor":
            from weightcode.application.runtime import doctor
            result = doctor(a.sandbox_python, threads=a.threads, bwrap=a.bwrap)
        else:
            import torch
            torch.set_num_threads(a.threads)
            from weightcode.application.bundles import pack_program, prepare_task
            if a.command == "import-model":
                from weightcode.application.resnet import import_checkpoint
                result = import_checkpoint(a.checkpoint, json.loads(a.labels.read_text()), a.out)
            elif a.command == "prepare-data":
                from weightcode.application.data import prepare_data
                result = prepare_data(a.model, a.new, a.old, a.out)
            elif a.command == "prepare-task":
                result = prepare_task(a.model, a.experience, a.selection, a.final, a.out, goal=a.goal,
                                      retention_drop=a.retention_drop, min_new_gain=a.min_new_gain)
            elif a.command == "pack-program":
                result = pack_program(a.program, a.model, a.out)
            else:
                from weightcode.application.runtime import apply_program
                from weightcode.application.editor import synthesize
                common = dict(wall_s=a.wall_s, program_wall_s=a.program_wall_s, threads=a.threads, bwrap=a.bwrap)
                if a.command == "apply":
                    result = apply_program(a.task, a.program, a.out, a.sandbox_python,
                                           program_sha256=a.program_sha256, **common)
                else:
                    result = synthesize(a.task, a.out, a.sandbox_python, editor_model=a.editor_model,
                                        checks=a.checks, cells=a.cells, **common)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        p.exit(2, f"weightcode: {exc}\n")
    return 0
