"""LEARN update programs on the host side (HARNESS_SPEC section 5): one run, delta validation and the derived patch.

A program is one ``.py`` file that defines ``update(student, experience) -> {tensor name: delta}`` and the non-empty
module strings ``HYPOTHESIS`` and ``PREDICTED_EFFECT``. ``run_update`` stages a fresh copy of exactly the student's
weights (``params.npz``), its ``forward.py`` and the input files that forward reads, ``layout.json`` (the visible
``model_spec.json``), ``experience.npz`` and the program into ``<work>/in`` and runs it once in the sandbox
(``weightcode.sandbox.program``, 300 s, under the driver's virtual clock, recorded as ``clock``). Everything the run wrote
is untrusted and validated here: ``deltas.npz`` must hold stored ``.npy`` members only, every key a tensor of the
student, every delta of that tensor's exact shape, dtype float16, float32 or float64, and finite; the module strings
must be non-empty ``str``. Every problem found is reported, in the words the editor sees, as ``ProgramProblem``; such
a run consumes no query.

``write_derived_patch`` turns valid deltas into an ordinary ``weightcode.patch.v0`` patch: one ``add_dense`` op per returned
tensor, in the model's tensor order, with the delta as a float64 ``.npy`` file, and the program's two strings as
``hypothesis`` and ``predicted_effect``. The arm applier then applies it within the limits exactly as a patch (so the
limit wording and its redaction are those of the patch path), and the ledger, finalisation, the dev readout and
``session_info`` reuse the patch path unchanged. The child is ``float32(float64(parent) + float64(delta))``.

Blind runs (``run_update(..., blind=True)``, the LEARN_BLIND arm of PREREG A4a): the editor of that arm never sees a
parameter value, and a failed run costs no query, so nothing a program controls may reach the editor's text. The
editor is shown only the run's exit status (``ok``, ``error``, ``timeout`` or ``died``; never the numeric exit code,
which a program sets with ``os._exit``) and, for a failure, the stage at which it failed in fixed wording: no stdout,
stderr, exception text or traceback, no key, shape, dtype or count of what update returned, and no file the run
wrote. The delta problems are shown as their kinds (``BLIND_DELTA_TEXTS``), once each and in a fixed order, naming
no tensor: which tensors fail, and in what order, is the program's choice and could carry the values it reads.
``ProgramProblem`` then carries the redacted ``messages`` for the editor and the full ``host_messages`` for the host
logs, and the run record's ``shown`` is ``{"exit_status": ...}``. Without ``blind`` every message, record and result
is as before.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.harness.experience import MODULE_STRINGS
from weightcode.harness.jsonio import file_sha256, json_text
from weightcode.harness.snapshot import SnapshotProblem, patch_name
from weightcode.patch import SCHEMA
from weightcode.sandbox.program import (
    PROGRAM_CLOCK,
    PROGRAM_WALL_S,
    STATUS_DIED,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_TIMEOUT,
    ProgramSandbox,
    run_program,
)

PROGRAM_SUFFIX = ".py"
MAX_PROGRAM_BYTES = 1 << 20
DERIVED_PATCH = "derived_patch.json"
DELTAS_FILE = "deltas.npz"
RUN_RECORD = "program_run.json"
RUN_DIR = "run"
DELTA_DTYPES = ("float16", "float32", "float64")
DELTA_FILE_SLACK_BYTES = 1 << 20
MAX_STRING_CHARS = 4000
SHOWN_OUTPUT_CHARS = 6000


class ProgramProblem(ValueError):
    """A run of the program failed or produced invalid output; ``messages`` are shown to the editor verbatim and
    ``host_messages`` (by default the same) go to the host logs; they differ only for a blind run (module docstring)."""

    def __init__(self, messages: list[str], run: Mapping[str, Any] | None = None,
                 host_messages: list[str] | None = None) -> None:
        self.messages = list(messages)
        self.host_messages = list(self.messages if host_messages is None else host_messages)
        self.run = None if run is None else dict(run)
        super().__init__("\n".join(self.messages))


@dataclass(frozen=True)
class Student:
    """One student as a program sees it, with the host files a run stages."""

    params: Mapping[str, np.ndarray]
    params_file: Path
    forward_file: Path
    input_files: Mapping[str, Path]
    layout: Mapping[str, Any]


@dataclass(frozen=True)
class Update:
    """A valid run: the deltas (float64), the program's two strings and the record of the run."""

    deltas: dict[str, np.ndarray]
    hypothesis: str
    predicted_effect: str
    record: dict[str, Any]
    deltas_path: Path

    @property
    def tensors(self) -> list[str]:
        return list(self.deltas)


def program_name(raw: object) -> str:
    """The bare name of a program file in /work/patches: a patch-style name ending in ``.py``."""
    name = patch_name(raw)
    if not name.endswith(PROGRAM_SUFFIX) or name == PROGRAM_SUFFIX:
        raise SnapshotProblem([f"program_file {name!r} must be the name of a .py file in /work/patches"])
    return name


def _clip(text: str, limit: int = SHOWN_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = (limit - 40) // 2
    return f"{text[:half]}\n[... {len(text) - 2 * half} characters omitted ...]\n{text[-half:]}"


def stage_inputs(in_dir: Path, student: Student, experience_file: Path, program_file: Path) -> dict[str, str]:
    """Copy exactly the run's inputs into the new directory ``in_dir``; returns ``{name: sha256}``."""
    in_dir.mkdir(mode=0o700)
    copies = {"params.npz": student.params_file, "experience.npz": experience_file,
              "forward.py": student.forward_file, **dict(student.input_files)}
    if program_file.name in copies or program_file.name == "layout.json":
        raise ProgramProblem([f"program_file {program_file.name!r} is the name of an input of the run; choose another"])
    copies[program_file.name] = program_file
    for name, source in copies.items():
        shutil.copyfile(source, in_dir / name)
    (in_dir / "layout.json").write_text(json_text(dict(student.layout)), encoding="utf-8")
    return {path.name: file_sha256(path) for path in sorted(in_dir.iterdir())}


RUN_STAGES = {"load": "while its inputs were loaded", "import": "while the program was imported",
              "update": "in update", "output": "in what update returned"}
BLIND_UNSHOWN = "the program's output and error text are not returned"
# The delta problems of a blind run, one per kind, in the order they are shown (module docstring).
BLIND_BAD_KEY = "deltas: a key is not the name of a tensor of the student (model_shapes.json lists the names)"
BLIND_BAD_SHAPE = "deltas: a delta does not have its tensor's shape (model_shapes.json lists the shapes)"
BLIND_BAD_DTYPE = "deltas: a delta is not float16, float32 or float64"
BLIND_NON_FINITE = "deltas: a delta has entries that are not finite"
BLIND_DELTA_TEXTS = (BLIND_BAD_KEY, BLIND_BAD_SHAPE, BLIND_BAD_DTYPE, BLIND_NON_FINITE)


def _blind_run_messages(run: Mapping[str, Any], wall_s: float) -> list[str]:
    """``_run_messages`` without anything the program controls (module docstring)."""
    status, result = run["status"], run.get("result") or {}
    if status == STATUS_TIMEOUT:
        return [f"the program did not return within {wall_s:g} seconds and was stopped (exit status timeout)"]
    if status == STATUS_ERROR:
        where = RUN_STAGES.get(str(result.get("stage")), "")
        return [f"the program raised an exception {where} (exit status error); {BLIND_UNSHOWN}"]
    return ["the program run ended without a result (exit status died; a memory or file-size limit, or an exit of "
            f"the program's own, ends a run this way); {BLIND_UNSHOWN}"]


def _run_messages(run: Mapping[str, Any], wall_s: float) -> list[str]:
    status, result = run["status"], run.get("result") or {}
    if status == STATUS_TIMEOUT:
        return [f"the program did not return within {wall_s:g} seconds and was stopped"]
    if status == STATUS_ERROR:
        stage = result.get("stage")
        where = RUN_STAGES.get(str(stage), "")
        return [f"the program failed {where} (it ran as /work/in/<program>):\n{_clip(str(result.get('error')))}"]
    stderr = str(run.get("stderr") or "")
    return [f"the program run ended without a result (exit status {run.get('exit_code')}; a memory or file-size limit "
            f"ends a run this way); stderr ends with: {stderr[-2000:]!r}"]


def _problem(problems: list[tuple[str, str]], blind: bool) -> ProgramProblem:
    """``ProgramProblem`` of ``(full, blind)`` message pairs: the full texts kept for the host; a blind run shows each
    blind text once, ``BLIND_DELTA_TEXTS`` in their fixed order (module docstring)."""
    full = [text for text, _ in problems]
    if not blind:
        return ProgramProblem(full, host_messages=full)
    found = {shown for _, shown in problems}
    shown = [text for text in BLIND_DELTA_TEXTS if text in found]
    shown += sorted(found - set(BLIND_DELTA_TEXTS))
    return ProgramProblem(shown, host_messages=full)


def load_deltas(path: Path, params: Mapping[str, np.ndarray], blind: bool = False) -> dict[str, np.ndarray]:
    """The validated deltas of ``deltas.npz`` as float64 arrays (module docstring); ``ProgramProblem`` otherwise.
    ``blind``: the problems are shown without anything the program controls (module docstring)."""
    n_params = int(sum(value.size for value in params.values()))
    cap = 8 * n_params + DELTA_FILE_SLACK_BYTES
    if not path.is_file() or path.is_symlink():
        raise _problem([("the run wrote no deltas.npz", "the run wrote no deltas.npz")], blind)
    size = path.stat().st_size
    if size > cap:
        raise _problem([(f"deltas.npz is {size} bytes, above the cap of {cap} bytes (8 bytes per parameter entry plus "
                         "1 MiB)", f"deltas.npz is above the cap of {cap} bytes (8 bytes per parameter entry plus "
                         "1 MiB)")], blind)
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
        names = [member.filename for member in members]
        if any(member.compress_type != zipfile.ZIP_STORED for member in members) or len(set(names)) != len(names) \
                or not all(name.endswith(".npy") for name in names) or sum(m.file_size for m in members) > cap:
            raise ValueError("members must be distinct stored .npy files within the cap")
        with np.load(path, allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile, EOFError) as exc:
        raise _problem([(f"deltas.npz is not a readable .npz of stored .npy members: {exc}",
                         "deltas.npz is not a readable .npz of stored .npy members")], blind) from None
    problems: list[tuple[str, str]] = []
    deltas: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if key not in params:
            problems.append((f"deltas: {key!r} is not a tensor of the student (its tensors: {', '.join(params)})",
                             BLIND_BAD_KEY))
            continue
        if value.shape != params[key].shape:
            problems.append((f"deltas[{key!r}]: shape {list(value.shape)}, but the tensor has shape "
                             f"{list(params[key].shape)}", BLIND_BAD_SHAPE))
            continue
        if value.dtype.name not in DELTA_DTYPES or value.dtype.byteorder not in ("=", "<", "|"):
            problems.append((f"deltas[{key!r}]: dtype {value.dtype}; a delta must be float16, float32 or float64",
                             BLIND_BAD_DTYPE))
            continue
        bad = int(value.size - np.count_nonzero(np.isfinite(value)))
        if bad:
            problems.append((f"deltas[{key!r}]: {bad} entries are not finite", BLIND_NON_FINITE))
            continue
        deltas[key] = np.ascontiguousarray(value, dtype=np.float64)
    if problems:
        raise _problem(problems, blind)
    return {key: deltas[key] for key in params if key in deltas}


def _module_strings(result: Mapping[str, Any]) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Problems with the program's two module strings as ``(full, blind)`` pairs, and the strings."""
    problems, strings = [], {}
    for name in MODULE_STRINGS:
        value, kind = result.get(name.lower()), result.get(f"{name.lower()}_type")
        if not isinstance(value, str) or not value.strip():
            found = "missing" if kind in (None, "NoneType") else "empty" if kind == "str" else f"of type {kind}"
            problems.append((f"the program must define {name} as a non-empty str at module level; it is {found}",
                             f"the program must define {name} as a non-empty str at module level"))
        else:
            strings[name] = value[:MAX_STRING_CHARS]
    return problems, strings


def run_update(student: Student, experience_file: Path, program_file: Path, work_dir: Path,
               sandbox: ProgramSandbox, wall_s: float = PROGRAM_WALL_S, blind: bool = False) -> Update:
    """Run ``program_file`` once on ``student`` and its experience in ``work_dir`` (created) and validate the output.

    The staged input copies are removed after the run; ``work_dir`` keeps ``out/`` and the stdout and stderr logs.
    Raises ``ProgramProblem`` (with the run's record) for anything the editor can fix. ``blind`` (the LEARN_BLIND arm):
    the record's ``shown`` is the exit status alone and a problem's ``messages`` hold nothing the program controls
    (module docstring); the run, the validation and the record are otherwise the same.
    """
    work_dir.mkdir(mode=0o700)
    inputs = stage_inputs(work_dir / "in", student, experience_file, program_file)
    try:
        run = run_program(sandbox, work_dir, program_file.name, wall_s).as_dict()
    finally:
        shutil.rmtree(work_dir / "in")
    record: dict[str, Any] = {"program_file": program_file.name, "program_sha256": inputs[program_file.name],
                              "inputs_sha256": inputs, "wall_s": wall_s, "clock": PROGRAM_CLOCK,
                              "sandbox": sandbox.as_dict(), "run": run}
    if blind:  # the exit status alone (module docstring)
        shown: dict[str, Any] = {"exit_status": str(run["status"])}
    else:
        shown = {"program_stdout": _clip(str(run["stdout"])), "program_stderr": _clip(str(run["stderr"])),
                 "duration_s": run["duration_s"]}
    record["shown"] = shown
    if run["status"] != STATUS_OK:
        if run["status"] not in (STATUS_ERROR, STATUS_TIMEOUT, STATUS_DIED):  # pragma: no cover - fixed statuses
            raise RuntimeError(f"unknown program run status {run['status']!r}")
        full = _run_messages(run, wall_s)
        raise ProgramProblem(_blind_run_messages(run, wall_s) if blind else full, record, full)
    problems, strings = _module_strings(run["result"] or {})
    shown_problems = [blind_text if blind else text for text, blind_text in problems]
    host_problems = [text for text, _ in problems]
    deltas_path = work_dir / "out" / DELTAS_FILE
    try:
        deltas = load_deltas(deltas_path, student.params, blind)
    except ProgramProblem as problem:
        raise ProgramProblem(shown_problems + problem.messages, record, host_problems + problem.host_messages) from None
    if problems:
        raise ProgramProblem(shown_problems, record, host_problems)
    record["deltas_sha256"] = file_sha256(deltas_path)
    record["tensors"] = list(deltas)
    return Update(deltas=deltas, hypothesis=strings["HYPOTHESIS"], predicted_effect=strings["PREDICTED_EFFECT"],
                  record=record, deltas_path=deltas_path)


def delta_file_name(index: int) -> str:
    return f"delta_{index:03d}.npy"


def write_derived_patch(update: Update, params: Mapping[str, np.ndarray], parent_sha256: str, dest: Path
                        ) -> dict[str, Any]:
    """Write the derived patch (``DERIVED_PATCH``) and its delta files into ``dest``; returns the patch."""
    ops = []
    for index, name in enumerate(params):
        if name in update.deltas:
            file_name = delta_file_name(index)
            np.save(dest / file_name, update.deltas[name], allow_pickle=False)
            ops.append({"op": "add_dense", "tensor": name, "delta": file_name})
    patch = {"schema": SCHEMA, "parent_sha256": parent_sha256, "ops": ops, "hypothesis": update.hypothesis,
             "predicted_effect": update.predicted_effect}
    (dest / DERIVED_PATCH).write_text(json.dumps(patch, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return patch


__all__ = ["BLIND_BAD_DTYPE", "BLIND_BAD_KEY", "BLIND_BAD_SHAPE", "BLIND_DELTA_TEXTS", "BLIND_NON_FINITE",
           "BLIND_UNSHOWN", "DELTAS_FILE", "DELTA_DTYPES", "DERIVED_PATCH", "MAX_PROGRAM_BYTES", "PROGRAM_SUFFIX",
           "RUN_DIR", "RUN_RECORD", "RUN_STAGES", "ProgramProblem", "Student", "Update", "load_deltas", "program_name",
           "run_update", "stage_inputs", "write_derived_patch"]
