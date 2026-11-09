"""Bounded application checks and finalization, using the research execution kernel."""

from __future__ import annotations

import platform
import shutil
import sys
import time
from importlib.metadata import version
from dataclasses import asdict
from pathlib import Path

import numpy as np

from weightcode.application.bundles import (bind, copy_task, dataset, digest, pack_program, program_bundle,
                                           program_source, read, sha, task_bundle, write)
from weightcode.application.resnet import export_checkpoint, forward_module, load_buffers, reference, reference_logits
from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.harness.program import ProgramProblem, Student, run_update, write_derived_patch
from weightcode.harness.session import SessionPaths
from weightcode.patch import Limits, PatchError, apply_patch
from weightcode.patch.limits import dense_delta_bytes, with_dense_file_cap
from weightcode.sandbox.host import SandboxConfig, SandboxKernel
from weightcode.sandbox.program import ProgramSandbox


def score(params: dict, data: dict) -> dict:
    forward = forward_module(len(params["fc.bias"])).forward
    result = {}
    for role in ("new", "old"):
        x, y = data[f"x_{role}"], data[f"y_{role}"]
        correct = 0
        # Small chunks keep host memory bounded independently of dataset length.
        for start in range(0, len(x), 4):
            pred = forward(params, x[start:start + 4]).argmax(axis=1)
            correct += int(np.count_nonzero(pred == y[start:start + 4]))
        result[role] = {"correct": correct, "n": len(y), "accuracy": correct / len(y)}
    return result


def decision(base: dict, child: dict, task: dict) -> dict:
    gain = child["new"]["accuracy"] - base["new"]["accuracy"]
    old = child["old"]["accuracy"] - base["old"]["accuracy"]
    return {"new_gain": gain, "old_change": old, "net_gain": gain + old,
            "eligible": gain > 0 and gain + 1e-12 >= task["min_new_gain"]
            and old + 1e-12 >= -task["retention_drop"]}


def deployment_score(params: dict, buffers: dict, data: dict) -> dict:
    model = reference(params, buffers)
    result = {}
    for role in ("new", "old"):
        logits = reference_logits(model, data[f"x_{role}"])
        y = data[f"y_{role}"]
        correct = int(np.count_nonzero(logits.argmax(1) == y))
        result[role] = {"correct": correct, "n": len(y), "accuracy": correct / len(y)}
    return result


def prepare_run(task: Path, out: Path, sandbox_python: Path, *, checks: int, cells: int,
                wall_s: float, program_wall_s: float = 300, threads: int = 4) -> dict:
    if type(checks) is not int or checks < 1 or type(cells) is not int or cells < 0 \
            or not 0 < wall_s < float("inf") or not 0 < program_wall_s < float("inf") or threads < 1:
        raise ValueError("checks>=1, cells>=0, finite positive timeouts and threads>=1 required")
    # Includes real sandbox startup, imports and filesystem/network isolation checks.
    probe = doctor(sandbox_python, threads=threads)
    task_bundle(task)
    paths = SessionPaths(out.absolute())
    paths.root.parent.mkdir(parents=True, exist_ok=True)
    paths.create()
    copy_task(task, paths.private / "task")
    t, m, params = task_bundle(paths.private / "task")
    paths.trial.mkdir()
    for name in ("params.npz", "forward.py"):
        shutil.copyfile(paths.private / "task" / "model" / name, paths.trial / name)
    shutil.copyfile(paths.private / "task" / "experience.npz", paths.trial / "experience.npz")
    write(paths.trial / "layout.json", m)
    limits = with_dense_file_cap(Limits(max_ops=max(64, len(params))), dense_delta_bytes(v.shape for v in params.values()))
    public = {"goal": t["goal"], "retention_drop": t["retention_drop"], "min_new_gain": t["min_new_gain"],
              "limits": asdict(limits), "selection_rule": t["selection_rule"], "checks": checks}
    write(paths.trial / "task.json", public)
    base = score(params, dataset(paths.private / "task" / "selection.npz", len(m["labels"])))
    write(paths.private / "baseline.json", base)
    config = {"schema": "weightcode.application_run.v1", "task_manifest_sha256": sha(paths.private / "task" / "manifest.json"),
              "sandbox_python": str(sandbox_python.absolute()), "checks": checks, "cells": cells,
              "wall_s": wall_s, "program_wall_s": program_wall_s, "threads": threads,
              "limits": asdict(limits), "started_unix": time.time(), "python": platform.python_version(),
              "numpy": np.__version__, "platform": platform.platform(), "sandbox_probe": probe,
              "torch": version("torch"), "torchvision": version("torchvision"),
              "invocation": [sys.executable, *sys.argv],
              "application_source_sha256": digest({p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}),
              "trial_files": bind(paths.trial,
                  ["params.npz", "forward.py", "experience.npz", "layout.json", "task.json"])}
    write(paths.root / "run.json", config)
    return config


def doctor(python: Path, *, threads: int = 4) -> dict:
    import tempfile
    from weightcode.sandbox.host import bwrap_version, venv_root

    venv_root(python)
    with tempfile.TemporaryDirectory(prefix="weightcode-doctor-") as tmp:
        root = Path(tmp)
        for name in ("trial", "patches", "scratch"):
            (root / name).mkdir()
        (root / "hidden.txt").write_text("not mounted")
        config = SandboxConfig(root / "trial", root / "patches", root / "scratch", root / "cells.jsonl",
                               python.absolute(), threads=threads, max_cells=1)
        with SandboxKernel(config) as kernel:
            code = ("import os, pathlib, socket, numpy, sys\n"
                    f"assert not pathlib.Path({str(root / 'hidden.txt')!r}).exists()\n"
                    "assert pathlib.Path('/work/trial').is_dir()\n"
                    "assert socket.if_nameindex() == [(1, 'lo')]\n"
                    "assert 'torch' not in sys.modules\nprint(numpy.__version__)")
            result = kernel.run(code)
            if result.error or result.timed_out:
                raise RuntimeError(f"sandbox self-check failed: {result.error} {result.stderr}")
    return {"status": "ok", "bubblewrap": bwrap_version(), "numpy": result.stdout.strip(),
            "sandbox_python": str(python.absolute()), "threads": threads}


class Application:
    """One run. All checks reserve a durable slot before running any candidate."""

    def __init__(self, root: Path):
        self.paths = SessionPaths(root.absolute())
        self.config = read(self.paths.root / "run.json")
        if self.config.get("schema") != "weightcode.application_run.v1":
            raise ValueError("unsupported application run")
        self.task_root = self.paths.private / "task"
        if sha(self.task_root / "manifest.json") != self.config["task_manifest_sha256"]:
            raise ValueError("run task changed")
        self.task, self.model, self.params = task_bundle(self.task_root)
        self.selection = dataset(self.task_root / "selection.npz", len(self.model["labels"]))
        self.base = read(self.paths.private / "baseline.json")
        self.limits = Limits(**self.config["limits"])
        self.sandbox = ProgramSandbox(Path(self.config["sandbox_python"]), 8 * 2**30, self.config["threads"])
        self.kernel: SandboxKernel | None = None

    def attempts(self) -> list[Path]:
        return sorted(self.paths.attempts.glob("[0-9][0-9][0-9][0-9]"))

    def status(self) -> dict:
        elapsed = time.time() - self.config["started_unix"]
        return {"checks_used": len(self.attempts()), "checks_remaining": max(0, self.config["checks"] - len(self.attempts())),
                "wall_s_remaining": max(0.0, self.config["wall_s"] - elapsed), "selection_base": self.base,
                "frozen": (self.paths.root / "freeze.json").exists()}

    def require_open(self) -> None:
        s = self.status()
        if s["frozen"] or s["wall_s_remaining"] <= 0:
            raise ValueError("run is frozen or editor wall budget exhausted")

    def python(self, code: str) -> dict:
        self.require_open()
        if self.kernel is None:
            p = self.paths
            self.kernel = SandboxKernel(SandboxConfig(p.trial, p.patches, p.scratch, p.cells_log,
                Path(self.config["sandbox_python"]), max_cells=self.config["cells"], threads=self.config["threads"],
                cell_timeout_s=min(120, self.status()["wall_s_remaining"])))
            self.kernel.start()
        return asdict(self.kernel.run(code))

    def check(self, name: str) -> dict:
        self.require_open()
        if not self.status()["checks_remaining"]:
            raise ValueError("check budget exhausted")
        if not name or Path(name).name != name or not name.endswith(".py"):
            raise ValueError("program_file must be a .py basename in /work/patches")
        source = self.paths.patches / name
        if source.is_symlink() or not source.is_file():
            raise ValueError("program must be a regular file in /work/patches")
        # Reserved before syntax validation/execution; invalid checks also consume a slot.
        attempt = self.paths.attempts / f"{len(self.attempts()) + 1:04d}"
        attempt.mkdir()
        record = {"index": len(self.attempts()), "status": "rejected"}
        try:
            (attempt / "program.py").write_bytes(program_source(source))
            record["program_sha256"] = sha(attempt / "program.py")
            student = Student(self.params, self.task_root / "model" / "params.npz",
                              self.task_root / "model" / "forward.py", {}, self.model)
            update = run_update(student, self.task_root / "experience.npz", attempt / "program.py",
                                attempt / "run", self.sandbox,
                                wall_s=min(self.config["program_wall_s"], self.status()["wall_s_remaining"]))
            write(attempt / "execution.json", update.record)
            patch = write_derived_patch(update, self.params, self.model["params_sha256"], attempt)
            result = apply_patch(self.params, patch, attempt, self.limits, self.model["params_sha256"])
            if any(np.any(v < 0) for k, v in result.child.items() if k.endswith("running_var")):
                raise ValueError("candidate has negative BatchNorm variance")
            measured = score(result.child, self.selection)
            record.update(status="checked", score=measured, decision=decision(self.base, measured, self.task),
                          patch=result.summary())
            save_params(attempt / "child.npz", result.child)
        except (ProgramProblem, PatchError, SyntaxError, ValueError) as exc:
            record["error"] = str(exc)
            if isinstance(exc, ProgramProblem) and exc.run is not None:
                write(attempt / "execution.json", dict(exc.run))
        write(attempt / "check.json", record)
        return record

    def freeze(self) -> dict:
        frozen = self.paths.root / "freeze.json"
        if frozen.exists():
            return read(frozen)
        eligible = []
        for path in self.attempts():
            if (path / "check.json").is_file():
                r = read(path / "check.json")
                if r["status"] == "checked" and r["decision"]["eligible"]:
                    eligible.append((r["decision"]["net_gain"], -r["index"], path, r))
        best = max(eligible, key=lambda item: item[:2]) if eligible else None
        record = {"schema": "weightcode.application_freeze.v1", "selected": best[2].name if best else None,
                  "program_sha256": best[3]["program_sha256"] if best else None,
                  "child_sha256": best[3]["patch"]["child_sha256"] if best else self.model["params_sha256"],
                  "checks_used": len(self.attempts())}
        write(frozen, record)
        return record

    def finalize(self) -> dict:
        self.close()
        frozen = self.freeze()  # No final-set scoring until after this write-once decision.
        if self.paths.result_json.exists():
            raise FileExistsError("run already finalized")
        selected = frozen["selected"]
        candidate = self.params
        if selected:
            pick = self.paths.attempts / selected
            candidate = load_params(pick / "child.npz")
            if params_sha256(candidate) != frozen["child_sha256"] or sha(pick / "program.py") != frozen["program_sha256"]:
                raise ValueError("frozen candidate changed")
        final = dataset(self.task_root / "final.npz", len(self.model["labels"]))
        base = score(self.params, final)
        measured = score(candidate, final) if selected else base
        verdict = decision(base, measured, self.task)
        buffers = load_buffers(self.task_root / "model")
        deployed_base = deployment_score(self.params, buffers, final)
        deployed_candidate = deployment_score(candidate, buffers, final) if selected else deployed_base
        deployed_verdict = decision(deployed_base, deployed_candidate, self.task)
        accepted = bool(selected and verdict["eligible"] and deployed_verdict["eligible"])
        released = candidate if accepted else self.params
        save_params(self.paths.final / "params.npz", released)
        export = export_checkpoint(released, buffers,
                                   self.paths.final / "model.pt", final["x_new"][:1])
        if selected:
            pack_program(pick / "program.py", self.task_root / "model", self.paths.final / "program",
                provenance={"kind": "application-run", "task_manifest_sha256": self.config["task_manifest_sha256"],
                            "source_model_sha256": self.model["params_sha256"], "final_accepted": accepted,
                            "final_decision": verdict, "frozen_program_sha256": frozen["program_sha256"]})
        report = {"schema": "weightcode.application_report.v1", "status": "accepted" if accepted else
                  "final-rejected" if selected else "base-kept", "freeze": frozen, "final_base": base,
                  "final_candidate": measured, "final_decision": verdict, "export": export,
                  "torchvision_final_base": deployed_base, "torchvision_final_candidate": deployed_candidate,
                  "torchvision_final_decision": deployed_verdict,
                  "output_is_updated_model": accepted, "run_manifest_sha256": sha(self.paths.root / "run.json"),
                  "note": "Application validation, not a reproduction of frozen paper experiments."}
        write(self.paths.result_json, report)
        write(self.paths.final / "report.json", report)
        return report

    def close(self) -> None:
        if self.kernel is not None:
            self.kernel.close()
            self.kernel = None


def apply_program(task: Path, program: Path, out: Path, sandbox_python: Path, **kwargs) -> dict:
    _, model, _ = task_bundle(task)
    manifest = program_bundle(program, model)
    prepare_run(task, out, sandbox_python, checks=1, cells=0, **kwargs)
    app = Application(out)
    shutil.copyfile(program / "program.py", app.paths.patches / "program.py")
    if sha(app.paths.patches / "program.py") != manifest["files"]["program.py"]:
        raise ValueError("program changed during snapshotting")
    app.check("program.py")
    return app.finalize()
