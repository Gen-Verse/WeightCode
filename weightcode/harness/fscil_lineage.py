"""The lineages of PREREG E9 (rule fscil-v1): T, the teacher; F, the frozen program; C, checked selection; and those of
PREREG E9 v2: T2, F2 and C2, the same three roles under kept rule fscil-retain-v2 on check score fscil-cb-v1.

A lineage is a sequence of students over sessions 2 to 9 of one base student (seed 100). Session t runs on the
lineage's instance of session t, which ``weightcode.evalkit.fscil.build_fscil_instance`` builds (or reuses) on the lineage's
output at t - 1 and the instance it was made on. Every session ends in an output (the kept child, or the session's own
student for a no-op) and is read with ``fscil_readout`` (child against the session's student) and ``fscil_accuracy``
(the child alone at session t).

- T. Sessions 2, 3 and 4 are one LEARN session each (``weightcode.harness.runner.run_session``; effort max, 90 min wall
  limit, 4 threads), under joint-retain-v1 with the session's floor in rows and a library of T's earlier kept programs
  (``weightcode.harness.fscil.library_entries``). Sessions 5 to 9 make no LLM call: each of T's kept programs of sessions 2 to
  4 runs on the current student and its experience, is checked through the instance's oracle (``make_fscil_oracle``,
  one query per program that runs and passes the limits, at most 3) and the kept rule with the session's floor picks
  among them and the student (score 0); with no kept program the session is a no-op.
- F. T's session-2 kept program runs unchanged at sessions 3 to 9 on F's own student, applied with no check. A program
  error or a limit violation is a recorded no-op. Session 2 is T's.
- C. The same program at sessions 3 to 9, applied only if its one check meets joint-retain-v1 with the session's floor
  (score > 0 and score_old >= floor), else a no-op. Session 2 is T's.

Floors: a JSON file (``load_floors``) ``{"schema": FLOORS_SCHEMA, "sessions": {"<t>": {"floor_t", "R", "d", "n",
"n_c", "n_s"}}}``, where n_c and n_s must be ``old_part_sizes(t)`` and floor_t ``joint_retain_floor(n_c, n_s, R, d)``;
n is the count of reference candidates behind d. Or the reference streams' run directory (``load_reference_floors``),
whose ``session_<t>.json`` records hold the same numbers. Any floor in -n_c .. 0 is accepted, 0 included (PREREG E9:
"never above 0"). T and C refuse a session without a floor; F needs none.

State: ``$WEIGHTCODE_RUNS/fscil_lineages/<lineage>_<base sha12>/state.json``, rewritten atomically after every step. The
driver resumes where it stopped, with E4-loop's rules:

- a session whose LEARN run finished is never launched again;
- a session that ends INFRA is run once more on the same context (same instance, floor and library); a second INFRA is
  a no-op;
- a session that ends aborted_leak, aborted_selftest or dry_run stops the driver (a fault to fix by hand);
- an attempt recorded as launched with no result is resolved from ``$WEIGHTCODE_RUNS/sessions``: its session (same instance,
  created at or after the launch, not claimed by another attempt) with a session_result.json is the finished run; one
  without a result but with a check is counted as INFRA; one without a check, or no session at all, is a stop before
  launch and is launched again.

E9 v2 (T2, F2, C2): the lineages run as T, F and C on their own state directories and instance chains (lineage names
T2, F2 and C2; F2 and C2 start from T2's session 2), and every kept rule is fscil-retain-v2
(``weightcode.harness.fscil.v2_rule_for``): T2's LEARN sessions run with ``fscil_version="v2"``, and the checks of T2's
sessions 5 to 9 and of C2 are ``weightcode.harness.fscil.make_cb_oracle`` checks picked by
``weightcode.harness.finalize.select_final_v2``. They read no floors: every floor is per candidate. T, F and C are unchanged.

Only one LEARN session runs at a time: the driver holds a lock per lineage for its whole run and the shared LEARN lock
around each launch, and refuses to launch while any editor session's MCP server runs on the machine (``busy``). In
production the driver itself runs as a systemd unit (``scripts/run_fscil_lineage.py``), so the LEARN session runs in
that unit.
"""

from __future__ import annotations

import fcntl
import math
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.common.paths import runs_dir
from weightcode.harness import fscil
from weightcode.harness.finalize import (
    KEEPS_FINAL_PATCH,
    STATUS_ABORTED_LEAK,
    STATUS_ABORTED_SELFTEST,
    STATUS_DRY_RUN,
    STATUS_INFRA,
)
from weightcode.harness.jsonio import file_sha256, read_json, utc_now, write_json

LINEAGES = ("T", "F", "C")  # PREREG E9
TEACHER = "T"
FROZEN = "F"
CHECKED = "C"
LINEAGES_V2 = ("T2", "F2", "C2")  # PREREG E9 v2
ALL_LINEAGES = (*LINEAGES, *LINEAGES_V2)
ROLES = {"T": TEACHER, "F": FROZEN, "C": CHECKED, "T2": TEACHER, "F2": FROZEN, "C2": CHECKED}
FIRST_SESSION = 2
LAST_SESSION = 9
TEACHER_SESSIONS = (2, 3, 4)
STATE_SCHEMA = "weightcode.fscil_lineage.v1"
FLOORS_SCHEMA = "weightcode.fscil_floors.v1"
LINEAGES_DIR = "fscil_lineages"
LEARN_LOCK = ".learn.lock"
SESSIONS_DIR = "sessions"
DEFAULT_EFFORT = "max"
DEFAULT_WALL_S = 90 * 60.0
DEFAULT_THREADS = 4
MAX_REUSE_CHECKS = 3
KIND_LEARN = "learn"  # T, sessions 2 to 4
KIND_REUSE = "reuse"  # T, sessions 5 to 9
KIND_FROZEN = "frozen"  # F, sessions 3 to 9
KIND_CHECKED = "checked"  # C, sessions 3 to 9
KIND_FROM_TEACHER = "from_teacher"  # F and C, session 2
OUTCOME_KEPT = "kept"
OUTCOME_NOOP = "no-op"
STATUS_PENDING = "pending"  # an attempt recorded before its launch, with no result yet
STATUS_NOT_LAUNCHED = "not_launched"  # a pending attempt resolved as a stop before launch
STOPS_LINEAGE = frozenset({STATUS_ABORTED_LEAK, STATUS_ABORTED_SELFTEST, STATUS_DRY_RUN})  # loop_stream.STOPS_STREAM
MCP_SERVER_MODULE = "weightcode.harness.mcp_server"

Learn = Callable[..., Any]  # run_session's signature -> a SessionResult (session_id, status, root, record)
Busy = Callable[[], list[str]]


class LineageError(RuntimeError):
    """The lineage cannot go on: a missing floor or input, a busy machine, or a session that stops the lineage."""


@dataclass(frozen=True)
class FloorRecord:
    """The kept-rule floor of one session in rows, with the inputs it was computed from (PREREG E9)."""

    session: int
    floor: int
    source: dict[str, Any]


# ---------------------------------------------------------------------------------------------------------------
# Floors


def load_floors(path: str | Path) -> dict[int, FloorRecord]:
    """The floors file (module docstring), each floor recomputed with ``joint_retain_floor`` from its own inputs and
    its old-part sizes checked against ``old_part_sizes``; any disagreement refuses the whole file."""
    from weightcode.evalkit.fscil import joint_retain_floor, old_part_sizes

    path = Path(path)
    document = read_json(path)
    if not isinstance(document, dict) or document.get("schema") != FLOORS_SCHEMA:
        raise LineageError(f"{path} is not a {FLOORS_SCHEMA} file")
    sessions = document.get("sessions")
    if not isinstance(sessions, dict) or not sessions:
        raise LineageError(f"{path} has no sessions")
    digest = file_sha256(path)
    floors: dict[int, FloorRecord] = {}
    for key, record in sessions.items():
        t = int(key)
        if str(t) != key or not FIRST_SESSION <= t <= LAST_SESSION or not isinstance(record, dict):
            raise LineageError(f"{path}: session {key!r} is not one of {FIRST_SESSION} to {LAST_SESSION}")
        missing = [name for name in ("floor_t", "R", "d", "n", "n_c", "n_s") if name not in record]
        if missing:
            raise LineageError(f"{path}: session {t} lacks {missing}")
        for name in ("floor_t", "n", "n_c", "n_s"):
            if isinstance(record[name], bool) or not isinstance(record[name], int):
                raise LineageError(f"{path}: session {t} {name} must be an int, got {record[name]!r}")
        for name in ("R", "d"):
            if isinstance(record[name], bool) or not isinstance(record[name], (int, float)) or not math.isfinite(
                    record[name]):
                raise LineageError(f"{path}: session {t} {name} must be a finite number, got {record[name]!r}")
        if record["n"] < 1:
            raise LineageError(f"{path}: session {t} n must count at least one reference candidate")
        sizes = old_part_sizes(t)
        if (record["n_c"], record["n_s"]) != sizes:
            raise LineageError(f"{path}: session {t} old-part sizes {(record['n_c'], record['n_s'])} are not the "
                               f"store's {sizes}")
        expected = joint_retain_floor(record["n_c"], record["n_s"], float(record["R"]), float(record["d"]))
        if record["floor_t"] != expected:
            raise LineageError(f"{path}: session {t} floor_t {record['floor_t']} is not joint_retain_floor's "
                               f"{expected} for its inputs")
        source = {"rule": "PREREG E9 joint_retain_floor", "path": str(path), "sha256": digest, "session": t,
                  **{name: record[name] for name in ("floor_t", "R", "d", "n", "n_c", "n_s")}}
        floors[t] = FloorRecord(session=t, floor=int(record["floor_t"]), source=source)
    return floors


def load_reference_floors(refs_dir: str | Path, base_params_sha256: str) -> dict[int, FloorRecord]:
    """The floors of the reference streams' run (``weightcode.baselines.fit.fscilrefs.run_streams``, PREREG E9): R from
    ``run.json``, d, its check count n, n_c, n_s and floor_t from the ``floor`` block of every written
    ``session_<t>.json``. The run must be on the lineage's base student, and every floor is recomputed with
    ``joint_retain_floor`` and its old-part sizes checked against ``old_part_sizes``, as ``load_floors`` does."""
    from weightcode.baselines.fit.fscilrefs import STREAMS_SCHEMA
    from weightcode.evalkit.fscil import joint_retain_floor, old_part_sizes

    refs_dir = Path(refs_dir)
    run = read_json(refs_dir / "run.json")
    if run.get("schema") != STREAMS_SCHEMA:
        raise LineageError(f"{refs_dir}/run.json is not a {STREAMS_SCHEMA} run")
    if (run.get("inputs") or {}).get("base", {}).get("params_sha256") != base_params_sha256:
        raise LineageError(f"{refs_dir} ran on another base student than the lineage's")
    r_points = float(run["r_points"])
    run_sha = file_sha256(refs_dir / "run.json")
    floors: dict[int, FloorRecord] = {}
    for t in range(FIRST_SESSION, LAST_SESSION + 1):
        path = refs_dir / f"session_{t}.json"
        if not path.is_file():
            continue
        record = read_json(path)
        block = record.get("floor") or {}
        if record.get("schema") != STREAMS_SCHEMA or int(record.get("session", -1)) != t:
            raise LineageError(f"{path} is not the {STREAMS_SCHEMA} record of session {t}")
        sizes = old_part_sizes(t)
        if (block.get("n_c"), block.get("n_s")) != sizes:
            raise LineageError(f"{path}: old-part sizes {(block.get('n_c'), block.get('n_s'))} are not the store's "
                               f"{sizes}")
        expected = joint_retain_floor(int(block["n_c"]), int(block["n_s"]), r_points, float(block["d"]))
        if block.get("floor") != expected:
            raise LineageError(f"{path}: floor {block.get('floor')} is not joint_retain_floor's {expected}")
        source = {"rule": "PREREG E9 joint_retain_floor", "path": str(path), "sha256": file_sha256(path),
                  "run_json_sha256": run_sha, "session": t, "floor_t": expected, "R": r_points,
                  "d": float(block["d"]), "n": int(block["n_checks"]), "n_c": int(block["n_c"]),
                  "n_s": int(block["n_s"])}
        floors[t] = FloorRecord(session=t, floor=expected, source=source)
    if not floors:
        raise LineageError(f"{refs_dir} has no session record yet")
    return floors


def check_repo_path() -> None:
    """Refuse a repo whose path holds a dataset token: every session mounts the repo, and its sandbox self-test
    would abort the session (aborted_selftest) on that path."""
    from weightcode.evalkit.fscil import DATASET
    from weightcode.evalkit.instances import leak_tokens, scan_for_tokens
    from weightcode.harness.provenance import REPO_ROOT

    found = scan_for_tokens(str(REPO_ROOT), leak_tokens(DATASET))
    if found:
        raise LineageError(f"the repo path {REPO_ROOT} holds {found}; a LEARN session's self-test refuses it, so "
                           "launch from a worktree whose path holds none")


def _floor(floors: Mapping[int, FloorRecord] | None, t: int, lineage: str, instance: Any) -> FloorRecord:
    """The session's floor, whose n_c must be the size of the instance's old check part."""
    if floors is None or t not in floors:
        raise LineageError(f"lineage {lineage}: no floor for session {t}; the floors file must state it before the "
                           "session runs")
    floor = floors[t]
    if int(floor.source["n_c"]) != int(instance.private["n_vcheck_old"]):
        raise LineageError(f"session {t}: the floor's n_c {floor.source['n_c']} is not the old check part of "
                           f"{instance.instance_id} ({instance.private['n_vcheck_old']} rows)")
    return floor


def lineage_version(lineage: str) -> str:
    """``weightcode.harness.fscil.FSCIL_V2`` for T2, F2 and C2, ``FSCIL_V1`` for T, F and C."""
    if lineage not in ALL_LINEAGES:
        raise LineageError(f"lineage {lineage!r} is not one of {ALL_LINEAGES}")
    return fscil.FSCIL_V2 if lineage in LINEAGES_V2 else fscil.FSCIL_V1


def teacher_of(lineage: str) -> str:
    """The teacher lineage whose session 2 a lineage starts from (T for T, F and C; T2 for T2, F2 and C2)."""
    return "T2" if lineage_version(lineage) == fscil.FSCIL_V2 else TEACHER


# ---------------------------------------------------------------------------------------------------------------
# State, locks and the machine check


def lineages_root() -> Path:
    """``$WEIGHTCODE_RUNS/fscil_lineages`` (created if missing)."""
    root = runs_dir() / LINEAGES_DIR
    root.mkdir(exist_ok=True)
    return root


def state_dir(lineage: str, base_params_sha256: str) -> Path:
    """``$WEIGHTCODE_RUNS/fscil_lineages/<lineage>_<base sha12>`` (created if missing)."""
    if lineage not in ALL_LINEAGES:
        raise LineageError(f"lineage {lineage!r} is not one of {ALL_LINEAGES}")
    path = lineages_root() / f"{lineage}_{base_params_sha256[:12]}"
    path.mkdir(exist_ok=True)
    return path


@contextmanager
def exclusive(path: Path, what: str) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` or refuse at once when another process holds it."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LineageError(f"{what} is held by another process ({path})") from None
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def running_editor_sessions() -> list[str]:
    """Pids of the processes on this machine that run an editor session's MCP server (any session, any tool)."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if MCP_SERVER_MODULE.encode() in command:
            found.append(entry.name)
    return found


def _read_state(path: Path, lineage: str, base: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"schema": STATE_SCHEMA, "lineage": lineage, "base_model_id": base["model_id"],
                "base_params_sha256": base["params_sha256"], "seed": int(base["seed"]), "created_at": utc_now(),
                "sessions": {}}
    state = read_json(path)
    if (state.get("schema") != STATE_SCHEMA or state.get("lineage") != lineage
            or state.get("base_params_sha256") != base["params_sha256"]):
        raise LineageError(f"{path} is not the {STATE_SCHEMA} state of lineage {lineage} on base {base['model_id']}")
    return state


def read_lineage_state(lineage: str, base_seed: int = 100) -> dict[str, Any] | None:
    """The recorded state of a lineage on the base student of ``base_seed``, or None before its first step."""
    from weightcode.evalkit.fscil import base_student

    base = base_student(base_seed)
    path = state_dir(lineage, str(base["params_sha256"])) / "state.json"
    return _read_state(path, lineage, base) if path.exists() else None


# ---------------------------------------------------------------------------------------------------------------
# Running a program on an instance


def run_program(program: Path, instance: Any, work: Path, sandbox: Any, wall_s: float | None = None
                ) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Run ``program`` once on the instance's student and experience (``stage_student``), turn its deltas into the
    derived patch and apply it under the fscil limits; returns (child or None, record). A program that fails or whose
    update a limit refuses gives None and a record of why."""
    from weightcode.harness.program import DERIVED_PATCH, ProgramProblem, run_update, write_derived_patch
    from weightcode.harness.transfer import stage_student
    from weightcode.patch import PatchError, apply_patch
    from weightcode.sandbox.program import PROGRAM_WALL_S

    work.mkdir(parents=True, mode=0o700)
    limits = fscil.program_limits(instance.task["limits"], len(instance.params))
    student, experience_file, experience = stage_student(instance, work / "student")
    record: dict[str, Any] = {"program": {"path": str(program), "name": program.name, "sha256": file_sha256(program)},
                              "instance_id": instance.instance_id, "parent_sha256": instance.params_sha256,
                              "experience": experience, "work_dir": str(work)}
    try:
        update = run_update(student, experience_file, program, work / "run", sandbox,
                            PROGRAM_WALL_S if wall_s is None else wall_s)
    except ProgramProblem as problem:
        record.update(status="program_failed", messages=problem.messages)
        return None, record
    record.update(deltas_sha256=update.record["deltas_sha256"], tensors=update.tensors,
                  hypothesis=update.hypothesis, predicted_effect=update.predicted_effect)
    patch_dir = work / "patch"
    patch_dir.mkdir()
    patch = write_derived_patch(update, instance.params, instance.params_sha256, patch_dir)
    record["patch_sha256"] = file_sha256(patch_dir / DERIVED_PATCH)
    try:
        result = apply_patch(instance.params, patch, patch_dir, limits, instance.params_sha256)
    except PatchError as refusal:
        record.update(status="limits_refused", messages=list(refusal.messages))
        return None, record
    record.update(status="ok", child_sha256=result.child_sha256, n_changed=result.n_changed,
                  changed_frac=result.changed_frac)
    return {name: np.array(value) for name, value in result.child.items()}, record


# ---------------------------------------------------------------------------------------------------------------
# The driver


class LineageDriver:
    """One lineage's driver (module docstring). ``learn`` replaces ``run_session`` (tests); ``busy`` lists running
    editor sessions (default ``running_editor_sessions``); ``sandbox`` runs programs (default: the session sandbox's
    interpreter and memory, ``threads``); ``session_kwargs`` are further run_session keywords (a launcher, the audit),
    never those the lineage fixes."""

    def __init__(self, lineage: str, *, floors: Mapping[int, FloorRecord] | None, base_seed: int = 100,
                 learn: Learn | None = None, busy: Busy | None = None, sandbox: Any | None = None,
                 effort: str = DEFAULT_EFFORT, wall_timeout_s: float = DEFAULT_WALL_S,
                 threads: int = DEFAULT_THREADS, session_kwargs: Mapping[str, Any] | None = None) -> None:
        from weightcode.evalkit.fscil import base_student

        self.version = lineage_version(lineage)
        self.v2 = self.version == fscil.FSCIL_V2
        self.role = ROLES[lineage]
        if self.v2 and floors:
            raise LineageError(f"lineage {lineage} keeps by {fscil.KEPT_RULE_V2}, whose floor is per candidate; it "
                               "reads no floors")
        if not self.v2 and self.role in (TEACHER, CHECKED) and not floors:
            raise LineageError(f"lineage {lineage} keeps by joint-retain-v1 and needs the floors file")
        if self.role == TEACHER:
            check_repo_path()
        self.lineage = lineage
        self.floors = dict(floors or {})
        self.base_seed = int(base_seed)
        self.base = base_student(base_seed)
        self.dir = state_dir(lineage, str(self.base["params_sha256"]))
        self.state_path = self.dir / "state.json"
        self.learn = learn
        self.busy = running_editor_sessions if busy is None else busy
        self._sandbox = sandbox
        self.effort, self.wall_timeout_s, self.threads = effort, float(wall_timeout_s), int(threads)
        self.session_kwargs = dict(session_kwargs or {})
        fixed = {"effort", "wall_timeout_s", "threads", "retention_floor_rows", "retention_floor_source",
                 "retention_floor_frac", "library_sessions", "loop_context", "dry_run",
                 "fscil_version"} & set(self.session_kwargs)
        if fixed:
            raise LineageError(f"session_kwargs may not set {sorted(fixed)}: the lineage fixes them")
        self.state: dict[str, Any] = {}

    # -- bookkeeping ----------------------------------------------------------------------------------------------

    @property
    def sandbox(self) -> Any:
        if self._sandbox is None:
            from weightcode.harness.runner import DEFAULT_MEMORY_BYTES, SANDBOX_PYTHON
            from weightcode.sandbox.program import ProgramSandbox

            self._sandbox = ProgramSandbox(python=SANDBOX_PYTHON, memory_bytes=DEFAULT_MEMORY_BYTES,
                                           threads=self.threads)
        return self._sandbox

    def _save(self) -> None:
        self.state["updated_at"] = utc_now()
        write_json(self.state_path, self.state)

    def _session(self, t: int) -> dict[str, Any]:
        return self.state["sessions"].setdefault(str(t), {"session": t})

    def _done(self, t: int) -> bool:
        return bool(self.state["sessions"].get(str(t), {}).get("done"))

    def _teacher_state(self) -> dict[str, Any]:
        teacher = teacher_of(self.lineage)
        state = read_lineage_state(teacher, self.base_seed)
        if state is None:
            others = [name for name in (LINEAGES_V2 if self.v2 else LINEAGES) if name != teacher]
            raise LineageError(f"lineage {teacher} has no state yet; {others[0]} and {others[1]} start from "
                               f"{teacher}'s session 2")
        return state

    def _kept_v2(self, instance: Any) -> dict[str, Any]:
        """The fscil-retain-v2 record of the session (E9 v2), read from the store and the instance."""
        try:
            return fscil.v2_rule_for(instance.private)
        except (fscil.FscilHarnessError, OSError, KeyError) as exc:
            raise LineageError(f"lineage {self.lineage}: the {fscil.KEPT_RULE_V2} rule of {instance.instance_id} "
                               f"cannot be read: {exc}") from None

    # -- instances --------------------------------------------------------------------------------------------------

    def _instance(self, t: int) -> Any:
        """The lineage's session-t instance, built (or reused) on its output at t - 1 and recorded."""
        from weightcode.evalkit.fscil import build_fscil_instance, load_fscil_instance

        record = self._session(t)
        if "instance_id" not in record:
            if t == FIRST_SESSION:
                row = build_fscil_instance(t, lineage=self.lineage, run_id=f"{self.lineage}-start",
                                           base_seed=self.base_seed)
            else:
                before = self.state["sessions"][str(t - 1)]
                if not before.get("done"):
                    raise LineageError(f"lineage {self.lineage}: session {t - 1} has no output yet")
                row = build_fscil_instance(t, lineage=self.lineage, parent_params=before["child_path"],
                                           parent_instance=before["instance_id"], run_id=before["run_id"],
                                           base_seed=self.base_seed)
            record.update(instance_id=row["instance_id"], instance_status=row["status"],
                          parent_params_sha256=row["parent_params_sha256"])
            self._save()
        instance = load_fscil_instance(str(record["instance_id"]))
        fscil.check_current(instance.instance_dir)
        if t > FIRST_SESSION and instance.params_sha256 != self.state["sessions"][str(t - 1)]["child_sha256"]:
            raise LineageError(f"session {t} instance {instance.instance_id} is not on the session {t - 1} output")
        return instance

    # -- outputs and readouts ---------------------------------------------------------------------------------------

    def _finish(self, t: int, instance: Any, child: Mapping[str, np.ndarray], child_path: Path, outcome: str,
                run_id: str, **fields: Any) -> None:
        digest = params_sha256(child)
        if params_sha256(load_params(child_path)) != digest:
            raise LineageError(f"session {t}: {child_path} does not hash to the session's output")
        record = self._session(t)
        record.update(fields)
        record.update(outcome=outcome, child_sha256=digest, child_path=str(child_path), run_id=run_id,
                      readout=fscil.readout(instance, child)["readouts_fscil"], accuracy=fscil.accuracy(child, t),
                      done=True, finished_at=utc_now())
        self._save()

    def _save_child(self, t: int, child: Mapping[str, np.ndarray]) -> Path:
        path = self.dir / f"session_{t}" / "student_after.npz"
        path.parent.mkdir(exist_ok=True)
        if save_params(path, child) != params_sha256(child):
            raise LineageError(f"session {t}: the saved output does not hash to the child")
        return path

    def _work(self, t: int, label: str) -> Path:
        base = self.dir / f"session_{t}" / "runs"
        base.mkdir(parents=True, exist_ok=True)
        index = len(list(base.iterdir()))
        return base / f"{index:02d}_{label}"

    # -- T, sessions 2 to 4 -------------------------------------------------------------------------------------------

    def _library(self, t: int) -> list[str] | None:
        ids = [self.state["sessions"][str(s)]["session_id"] for s in range(FIRST_SESSION, t)
               if self.state["sessions"].get(str(s), {}).get("final_kind") == "checked"]
        return ids or None

    def _claimed(self) -> set[str]:
        return {attempt["session_id"] for record in self.state["sessions"].values()
                for attempt in record.get("attempts", []) if attempt.get("session_id")}

    def _resolve_pending(self, attempt: dict[str, Any], instance_id: str) -> None:
        """An attempt with no recorded result: find its session (module docstring) and record what it reached."""
        claimed = self._claimed()
        found = []
        for session_json in sorted((runs_dir() / SESSIONS_DIR).glob("s*/session.json")):
            try:
                session = read_json(session_json)
            except (OSError, ValueError):
                continue
            if (session.get("instance_id") == instance_id and session.get("arm") == "LEARN"
                    and not session.get("dry_run") and str(session.get("created_at")) >= attempt["launched_at"]
                    and session_json.parent.name not in claimed):
                found.append(session_json.parent)
        if len(found) > 1:
            raise LineageError(f"{len(found)} unclaimed sessions on {instance_id} since {attempt['launched_at']}: "
                               f"{[path.name for path in found]}; resolve by hand")
        if not found:
            attempt.update(status=STATUS_NOT_LAUNCHED, resolved_at=utc_now(), resolved_from="no session")
            return
        root = found[0]
        attempt["session_id"] = root.name
        if (root / "session_result.json").is_file():
            result = read_json(root / "session_result.json")
            attempt.update(status=str(result["status"]), resolved_at=utc_now(), resolved_from="session_result.json")
            return
        checks = root / "logs" / "checks.jsonl"
        has_check = checks.is_file() and any(line.strip() for line in checks.read_text(encoding="utf-8").splitlines())
        attempt.update(status=STATUS_INFRA if has_check else STATUS_NOT_LAUNCHED, resolved_at=utc_now(),
                       resolved_from="killed after a check" if has_check else "killed before any check")

    def _teacher_session(self, t: int) -> None:
        from weightcode.harness.runner import run_session

        instance = self._instance(t)
        floor = None if self.v2 else _floor(self.floors, t, self.lineage, instance)
        record = self._session(t)
        record.setdefault("kind", KIND_LEARN)
        library = self._library(t)
        if floor is None:  # E9 v2: no floor, the session keeps by fscil-retain-v2
            record.update(kept_rule=self._kept_v2(instance), library_sessions=library)
        else:
            record.update(floor=floor.floor, floor_source=floor.source, library_sessions=library)
        attempts = record.setdefault("attempts", [])
        if attempts and attempts[-1].get("status") == STATUS_PENDING:
            self._resolve_pending(attempts[-1], instance.instance_id)
            self._save()
        while True:
            statuses = [attempt["status"] for attempt in attempts]
            if any(status in STOPS_LINEAGE for status in statuses):
                stopped = next(a for a in attempts if a["status"] in STOPS_LINEAGE)
                raise LineageError(f"lineage {self.lineage} session {t}: session {stopped.get('session_id')} ended "
                                   f"{stopped['status']}; the lineage stops here and resumes from this session")
            finished = [attempt for attempt in attempts if attempt["status"] in KEEPS_FINAL_PATCH]
            if finished:
                return self._teacher_output(t, instance, finished[-1])
            if statuses.count(STATUS_INFRA) >= 2:  # a second INFRA: the session is a no-op on the current student
                path = self._save_child(t, instance.params)
                return self._finish(t, instance, instance.params, path, OUTCOME_NOOP,
                                    run_id=str(attempts[-1].get("session_id") or f"{self.lineage}-s{t}"),
                                    final_kind="base", reason="two INFRA attempts")
            running = self.busy()
            if running:
                raise LineageError(f"an editor session runs on this machine (pids {running}); one LEARN session at "
                                   f"a time, so session {t} is not launched")
            attempt = {"repeat": len(attempts), "launched_at": utc_now(), "status": STATUS_PENDING}
            attempts.append(attempt)
            self._save()
            with exclusive(lineages_root() / LEARN_LOCK, "the LEARN lock"):
                rule: dict[str, Any] = ({"fscil_version": fscil.FSCIL_V2} if floor is None else
                                        {"retention_floor_rows": floor.floor, "retention_floor_source": floor.source})
                kwargs = dict(self.session_kwargs, effort=self.effort, wall_timeout_s=self.wall_timeout_s,
                              threads=self.threads, **rule, library_sessions=library)
                launch = run_session if self.learn is None else self.learn
                result = launch(instance.instance_id, "LEARN", attempt["repeat"], **kwargs)
            attempt.update(session_id=result.session_id, status=str(result.status), ended_at=utc_now())
            self._save()

    def _teacher_output(self, t: int, instance: Any, attempt: Mapping[str, Any]) -> None:
        root = runs_dir() / SESSIONS_DIR / str(attempt["session_id"])
        session = read_json(root / "session.json")
        if session.get("instance_id") != instance.instance_id or session.get("real_model_sha256") != (
                instance.params_sha256):
            raise LineageError(f"{root}: the session is not on lineage {self.lineage}'s session-{t} instance")
        result = read_json(root / "session_result.json")
        final = result["final"]
        child_path = root / "final" / "model.npz"
        child = load_params(child_path)
        if params_sha256(child) != final["child_sha256"]:
            raise LineageError(f"{root}: final/model.npz does not hash to the final child")
        kept = final["kind"] == "checked"
        if kept == (final["child_sha256"] == instance.params_sha256):
            raise LineageError(f"{root}: a checked final must change the student and any other final keep it")
        program = None
        if (session.get("kept_rule") or {}).get("name") != (fscil.KEPT_RULE_V2 if self.v2 else "joint-retain-v1"):
            raise LineageError(f"{root}: the session's kept rule is not lineage {self.lineage}'s")
        if kept:
            program = {"file": str(root / str(final["program"]["file"])), "patch_id": final["patch_id"],
                       "sha256": final["program"]["sha256"], "query_index": final["query_index"],
                       "score": final["score"], "score_new": final["score_new"], "score_old": final["score_old"],
                       **({key: final[key] for key in fscil.V2_LIBRARY_FIELDS} if self.v2 else {})}
            if file_sha256(Path(program["file"])) != program["sha256"]:
                raise LineageError(f"{root}: the kept program does not hash to its final record")
        self._finish(t, instance, child, child_path, OUTCOME_KEPT if kept else OUTCOME_NOOP,
                     run_id=str(attempt["session_id"]), session_id=str(attempt["session_id"]),
                     status=str(result["status"]), final_kind=final["kind"], program=program)

    # -- offline sessions --------------------------------------------------------------------------------------------

    def _teacher_programs(self, sessions: Sequence[int]) -> list[dict[str, Any]]:
        teacher = self.state if self.role == TEACHER else self._teacher_state()
        programs = []
        for s in sessions:
            record = teacher["sessions"].get(str(s), {})
            name = teacher_of(self.lineage)
            if not record.get("done"):
                raise LineageError(f"{name}'s session {s} has not finished; lineage {self.lineage} needs its output")
            if record.get("program") is not None:
                program = dict(record["program"], session=s)
                if file_sha256(Path(program["file"])) != program["sha256"]:
                    raise LineageError(f"{name}'s session-{s} program {program['file']} changed since it was kept")
                programs.append(program)
        return programs

    def _check_and_keep(self, t: int, instance: Any, programs: Sequence[Mapping[str, Any]],
                        floor: FloorRecord | None, budget: int
                        ) -> tuple[dict[str, np.ndarray] | None, list[dict[str, Any]], dict[str, Any]]:
        """Run each program, check each child that passes the limits (one query each) and pick by joint-retain-v1
        with ``floor``, or with ``floor`` None (E9 v2) by fscil-retain-v2 on ``make_cb_oracle`` checks."""
        from weightcode.evalkit.fscil import make_fscil_oracle
        from weightcode.harness.finalize import select_final, select_final_v2

        rule = self._kept_v2(instance) if floor is None else None
        oracle = make_fscil_oracle(instance, budget) if rule is None else fscil.make_cb_oracle(instance, budget, rule)
        fields = ("query_index", "score", "score_new", "score_old", "fixed_new", "broken_new", "fixed_old",
                  "broken_old", "n_changed", "child_sha256",
                  *((*fscil.V2_SHOWN_FIELDS, *fscil.V2_COUNT_FIELDS) if rule is not None else ()))
        runs, children, checked = [], {}, []
        for program in programs:
            label = f"s{program['session']}"
            child, record = run_program(Path(program["file"]), instance, self._work(t, label), self.sandbox)
            record["source_session"] = program["session"]
            if child is not None:
                try:
                    reply = oracle.query(child, label)
                except ValueError as exc:  # the oracle refused the child before consuming a query
                    record.update(status="check_refused", messages=[str(exc)])
                else:
                    record["check"] = {key: reply[key] for key in fields}
                    checked.append({**record["check"], "patch_id": label})
                    children[label] = child
            runs.append(record)
        if rule is not None:
            pick = select_final_v2(checked, rule)
            chosen = children.get(str(pick["patch_id"])) if pick["kind"] == "checked" else None
            return chosen, runs, {**pick, "kept_rule": rule, "n_checks": len(checked), "budget": budget}
        assert floor is not None
        pick = select_final(checked, budget=budget, declared=None, s_old_floor=floor.floor)
        chosen = children.get(str(pick["patch_id"])) if pick["kind"] == "checked" else None
        return chosen, runs, {**pick, "s_old_floor": floor.floor, "n_checks": len(checked), "budget": budget}

    def _reuse_session(self, t: int) -> None:
        instance = self._instance(t)
        floor = None if self.v2 else _floor(self.floors, t, self.lineage, instance)
        programs = self._teacher_programs(TEACHER_SESSIONS)
        record = self._session(t)
        if floor is None:  # E9 v2
            record.update(kind=KIND_REUSE, kept_rule=self._kept_v2(instance),
                          programs=[{k: p[k] for k in ("session", "file", "sha256")} for p in programs])
        else:
            record.update(kind=KIND_REUSE, floor=floor.floor, floor_source=floor.source,
                          programs=[{k: p[k] for k in ("session", "file", "sha256")} for p in programs])
        if not programs:
            path = self._save_child(t, instance.params)
            return self._finish(t, instance, instance.params, path, OUTCOME_NOOP, run_id=f"{self.lineage}-s{t}",
                                runs=[], pick=None, reason="T kept no program in sessions 2 to 4")
        child, runs, pick = self._check_and_keep(t, instance, programs, floor, min(MAX_REUSE_CHECKS, len(programs)))
        output = instance.params if child is None else child
        path = self._save_child(t, output)
        self._finish(t, instance, output, path, OUTCOME_NOOP if child is None else OUTCOME_KEPT,
                     run_id=f"{self.lineage}-s{t}", runs=runs, pick=pick)

    def _session_two_program(self) -> dict[str, Any]:
        (program,) = self._teacher_programs((FIRST_SESSION,)) or (None,)
        if program is None:
            raise LineageError(f"{teacher_of(self.lineage)}'s session 2 kept no program; lineage {self.lineage} has "
                               "none to run")
        return program

    def _from_teacher(self) -> None:
        """F and C, session 2: T's session 2 (its instance, output and readouts); F2 and C2 likewise from T2's."""
        teacher = self._teacher_state()["sessions"].get(str(FIRST_SESSION), {})
        if not teacher.get("done"):
            raise LineageError(f"{teacher_of(self.lineage)}'s session 2 has not finished; lineage {self.lineage} "
                               "starts from its output")
        self._session_two_program()
        record = self._session(FIRST_SESSION)
        record.update({key: teacher[key] for key in ("instance_id", "instance_status", "parent_params_sha256",
                                                     "outcome", "child_sha256", "child_path", "run_id", "readout",
                                                     "accuracy", "session_id", "program")},
                      kind=KIND_FROM_TEACHER,
                      teacher_state=str(state_dir(teacher_of(self.lineage), str(self.base["params_sha256"]))
                                        / "state.json"),
                      done=True, finished_at=utc_now())
        self._save()

    def _frozen_or_checked_session(self, t: int) -> None:
        instance = self._instance(t)
        program = self._session_two_program()
        record = self._session(t)
        record.update(kind=KIND_FROZEN if self.role == FROZEN else KIND_CHECKED,
                      program={k: program[k] for k in ("session", "file", "sha256")})
        if self.role == FROZEN:
            child, run = run_program(Path(program["file"]), instance, self._work(t, "s2"), self.sandbox)
            output = instance.params if child is None else child
            path = self._save_child(t, output)
            return self._finish(t, instance, output, path, OUTCOME_NOOP if child is None else OUTCOME_KEPT,
                                run_id=f"{self.lineage}-s{t}", runs=[run], pick=None)
        if self.v2:  # E9 v2: C2 checks by fscil-retain-v2
            floor = None
            record.update(kept_rule=self._kept_v2(instance))
        else:
            floor = _floor(self.floors, t, self.lineage, instance)
            record.update(floor=floor.floor, floor_source=floor.source)
        child, runs, pick = self._check_and_keep(t, instance, [program], floor, 1)
        output = instance.params if child is None else child
        path = self._save_child(t, output)
        self._finish(t, instance, output, path, OUTCOME_NOOP if child is None else OUTCOME_KEPT,
                     run_id=f"{self.lineage}-s{t}", runs=runs, pick=pick)

    # -- the loop -----------------------------------------------------------------------------------------------------

    def run(self, until: int = LAST_SESSION) -> dict[str, Any]:
        """Run (or resume) the lineage through session ``until``; returns the state with its summary."""
        if not FIRST_SESSION <= int(until) <= LAST_SESSION:
            raise LineageError(f"until must be a session of {FIRST_SESSION} to {LAST_SESSION}")
        with exclusive(self.dir / ".lock", f"lineage {self.lineage}"):
            self.state = _read_state(self.state_path, self.lineage, self.base)
            if "session_1" not in self.state:
                self.state["session_1"] = {"accuracy": fscil.accuracy(self._base_params(), 1)}
                self._save()
            for t in range(FIRST_SESSION, int(until) + 1):
                if self._done(t):
                    continue
                if self.role == TEACHER:
                    if t in TEACHER_SESSIONS:
                        self._teacher_session(t)
                    else:
                        self._reuse_session(t)
                elif t == FIRST_SESSION:
                    self._from_teacher()
                else:
                    self._frozen_or_checked_session(t)
            self.state["summary"] = summarize(self.state, self._teacher_bal_new_2())
            self._save()
            return self.state

    def _base_params(self) -> dict[str, np.ndarray]:
        from weightcode.evalkit.fscil import model_record

        _, params, _, _ = model_record(self.base, "fscil_base")
        return params

    def _teacher_bal_new_2(self) -> float | None:
        teacher = self.state if self.role == TEACHER else self._teacher_state()
        record = teacher["sessions"].get(str(FIRST_SESSION), {})
        return None if not record.get("done") else record["accuracy"]["per_session"][str(FIRST_SESSION)]["bal"]


def summarize(state: Mapping[str, Any], teacher_bal_new_2: float | None = None) -> dict[str, Any]:
    """The FSCIL summaries of one lineage (PREREG E9 readouts), from its recorded absolute readouts.

    ``bal_seen`` per session 1 to 9 (all-seen class-balanced accuracy, the main endpoint), its mean over sessions 1 to 9
    and the drop from session 1 to 9 (None until every session has a readout); ``bal_new`` per session (the session's
    own classes, class-balanced); ``forgetting``: for the classes of each session s, their class-balanced accuracy at
    session s minus at session 9; and for F and C ``c_ratio``, the mean of ``bal_new`` over sessions 3 to 9 divided by
    T's at session 2 (E3's ratio, PREREG E9 C)."""
    accuracy = {1: state.get("session_1", {}).get("accuracy")}
    for key, record in state.get("sessions", {}).items():
        if record.get("done"):
            accuracy[int(key)] = record["accuracy"]
    sessions = range(1, LAST_SESSION + 1)
    bal_seen = {str(t): (accuracy[t]["bal_seen"] if accuracy.get(t) else None) for t in sessions}
    bal_new = {str(t): (accuracy[t]["per_session"][str(t)]["bal"] if accuracy.get(t) else None) for t in sessions}
    complete = all(value is not None for value in bal_seen.values())
    last = accuracy.get(LAST_SESSION)
    forgetting = {str(s): (None if not accuracy.get(s) or last is None
                           or accuracy[s]["per_session"][str(s)]["bal"] is None
                           or last["per_session"][str(s)]["bal"] is None
                           else accuracy[s]["per_session"][str(s)]["bal"] - last["per_session"][str(s)]["bal"])
                  for s in sessions}
    summary: dict[str, Any] = {
        "bal_seen": bal_seen,
        "bal_seen_mean_1_to_9": float(np.mean([bal_seen[str(t)] for t in sessions])) if complete else None,
        "bal_seen_drop_1_to_9": (bal_seen["1"] - bal_seen[str(LAST_SESSION)]) if complete else None,
        "bal_new": bal_new,
        "forgetting": forgetting,
        "outcomes": {str(t): state["sessions"][str(t)].get("outcome") for t in range(FIRST_SESSION, LAST_SESSION + 1)
                     if str(t) in state.get("sessions", {})},
    }
    if ROLES.get(str(state.get("lineage"))) in (FROZEN, CHECKED):
        later = [bal_new[str(t)] for t in range(3, LAST_SESSION + 1)]
        summary["c_ratio"] = (float(np.mean(later)) / teacher_bal_new_2
                              if all(v is not None for v in later) and teacher_bal_new_2 else None)
        summary["teacher_bal_new_2"] = teacher_bal_new_2
    return summary


def dry_run(*, floors: Mapping[int, FloorRecord] | None, placeholder_floor: int | None = None, base_seed: int = 100,
            effort: str = DEFAULT_EFFORT, wall_timeout_s: float = DEFAULT_WALL_S, threads: int = DEFAULT_THREADS,
            session_kwargs: Mapping[str, Any] | None = None, lineage: str = TEACHER) -> Any:
    """T's session 2 as a dry run (``run_session(dry_run=True)``): the session-2 instance built or reused, the floor of
    the floors file, or ``placeholder_floor`` marked as a placeholder when no floor is stated yet. The lineage state is
    not touched. ``lineage`` "T2" (E9 v2) dry-runs T2's session 2 under fscil-retain-v2, with no floor."""
    from weightcode.evalkit.fscil import build_fscil_instance
    from weightcode.harness.runner import run_session

    if lineage not in (TEACHER, "T2"):
        raise LineageError(f"only a teacher's session 2 has a dry run, not lineage {lineage!r}")
    check_repo_path()
    if lineage == "T2":
        if floors or placeholder_floor is not None:
            raise LineageError(f"lineage T2 keeps by {fscil.KEPT_RULE_V2}; its dry run takes no floor")
        row = build_fscil_instance(FIRST_SESSION, lineage=lineage, run_id="dry-run", base_seed=base_seed)
        return run_session(str(row["instance_id"]), "LEARN", 0, **dict(session_kwargs or {}), effort=effort,
                           wall_timeout_s=wall_timeout_s, threads=threads, dry_run=True,
                           fscil_version=fscil.FSCIL_V2)
    if floors and FIRST_SESSION in floors:
        floor, source = floors[FIRST_SESSION].floor, floors[FIRST_SESSION].source
    elif placeholder_floor is not None:
        floor, source = int(placeholder_floor), {"placeholder": True, "note": "a stand-in floor for a dry run, not the "
                                                 "PREREG E9 floor of session 2"}
    else:
        raise LineageError("a dry run needs the session-2 floor or a placeholder floor")
    row = build_fscil_instance(FIRST_SESSION, lineage=TEACHER, run_id="dry-run", base_seed=base_seed)
    return run_session(str(row["instance_id"]), "LEARN", 0, **dict(session_kwargs or {}), effort=effort,
                       wall_timeout_s=wall_timeout_s, threads=threads, dry_run=True, retention_floor_rows=floor,
                       retention_floor_source=source)


__all__ = ["ALL_LINEAGES", "CHECKED", "FLOORS_SCHEMA", "FROZEN", "FloorRecord", "LINEAGES", "LINEAGES_V2",
           "LineageDriver", "LineageError", "ROLES", "STATE_SCHEMA", "TEACHER", "check_repo_path", "dry_run",
           "exclusive", "lineage_version", "lineages_root", "load_floors", "load_reference_floors",
           "read_lineage_state", "run_program", "running_editor_sessions", "state_dir", "summarize", "teacher_of"]
