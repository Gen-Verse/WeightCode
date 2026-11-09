"""Public CLI. Optional model dependencies load only for model operations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="weightcode", description="Install and reuse executable neural weight programs.")
    p.add_argument("--threads", type=int, default=4, help="CPU/BLAS threads (default: 4)")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Test the actual Linux/NumPy sandbox")
    doctor.add_argument("--sandbox-python", type=Path, required=True)
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
        for field in ("task", "out", "sandbox-python"):
            run.add_argument(f"--{field}", type=Path, required=True)
        run.add_argument("--wall-s", type=float, required=True, help="Editor/apply wall budget; independent final validation follows")
        run.add_argument("--program-wall-s", type=float, default=300)
        if name == "apply":
            run.add_argument("--program", type=Path, required=True, help="Program bundle directory")
        else:
            run.add_argument("--editor-model", required=True)
            run.add_argument("--checks", type=int, required=True)
            run.add_argument("--cells", type=int, required=True)
    return p


def main(argv=None) -> int:
    p = parser()
    a = p.parse_args(argv)
    if a.threads < 1:
        p.error("--threads must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(a.threads)
    try:
        if a.command == "doctor":
            from weightcode.application.runtime import doctor
            result = doctor(a.sandbox_python, threads=a.threads)
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
                common = dict(wall_s=a.wall_s, program_wall_s=a.program_wall_s, threads=a.threads)
                if a.command == "apply":
                    result = apply_program(a.task, a.program, a.out, a.sandbox_python, **common)
                else:
                    result = synthesize(a.task, a.out, a.sandbox_python, editor_model=a.editor_model,
                                        checks=a.checks, cells=a.cells, **common)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        p.exit(2, f"weightcode: {exc}\n")
    return 0
