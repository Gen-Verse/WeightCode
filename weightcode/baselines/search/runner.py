"""Run one search baseline on one instance and record it (contract sections 4, 5.0, 5.1 and 5.2).

Order of operations, so that the search cannot see V_check: load the editor-visible files, check the
re-implemented forward against ``forward.py`` on the evidence, localise and search on evidence only, pick up to
``query_budget`` candidates by ``selection.QUERY_RULE``, and only then build the counted patch-check oracle and spend
the queries. Every selected candidate is measured against the edit limits, written as a sparse patch to ``q<k>/``
(``patches.write_sparse_patch``) and applied with the shared executor (``weightcode.patch.apply_patch_file``) under the
executor's default limits. A patch the executor rejects is recorded with the executor's messages and consumes no
query (task.json: "a patch that fails validation consumes no query"); for an accepted patch the executor's child is
what the oracle checks, so the V_check numbers belong to the patch file on disk. The final patch is
``oracle.final()`` (contract 5.2). Outputs go to ``$WEIGHTCODE_RUNS/baselines/<instance_id>/<method>/<run_seed>/``.

The edit limits come from ``weightcode.patch.Limits()`` (``EditLimits.from_executor``); a task.json whose limits differ
from them belongs to an instance built before a limits change and is refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import DEFAULT_QUERY_BUDGET, Instance, load_instance, make_oracle
from weightcode.baselines.search import arachne, care
from weightcode.baselines.search.evidence import CountedForward, EvidenceSets, build_evidence_sets, evidence_readout
from weightcode.baselines.search.limits import EditLimits, edit_stats, limit_violations
from weightcode.baselines.search.netspec import NetSpec, check_parity, parse_spec
from weightcode.baselines.search.patches import write_sparse_patch
from weightcode.baselines.search.selection import QUERY_RULE, SearchResult, Snapshot, select_for_query
from weightcode.common.identity import count_changed
from weightcode.common.paths import runs_dir
from weightcode.evalkit.metrics import BASE_CANDIDATE, Candidate
from weightcode.patch import Limits, PatchError, apply_patch_file, task_limits

RUN_SCHEMA = "weightcode.search_baseline_run.v1"
METHODS: tuple[str, ...] = ("arachne", "care")
METHOD_CODES: dict[str, int] = {"arachne": 1, "care": 2}
PATCH_WRITER = "weightcode.baselines.search.patches.write_sparse_patch (index-form set ops)"
PATCH_APPLIER = "weightcode.patch.apply_patch_file with the executor's default Limits()"
PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunOptions:
    """Protocol parameters of a run; defaults are the contract values and the papers' configurations."""

    query_budget: int = DEFAULT_QUERY_BUDGET
    parity_tol: float = 1e-5
    limits: EditLimits = field(default_factory=EditLimits.from_executor)
    arachne: arachne.ArachneConfig = field(default_factory=arachne.ArachneConfig)
    care: care.CareConfig = field(default_factory=care.CareConfig)


class SplitRoleError(PermissionError):
    """The instance's split role does not allow this run."""


def check_split_role(instance: Instance, allow_main: bool) -> str:
    """Return the split role; refuse main instances unless ``allow_main`` and refuse missing or unknown roles.

    PREREG section 2: non-LLM arms run on dev instances until their code and parameters are frozen.
    """
    role = instance.private.get("split_role")
    if role == "dev":
        return role
    if role == "main":
        if allow_main:
            return role
        raise SplitRoleError(
            f"{instance.instance_id} is a main instance; search baselines run on main instances only after the "
            "protocol freeze (PREREG section 2). Pass allow_main explicitly once frozen."
        )
    raise SplitRoleError(f"{instance.instance_id}: private.json split_role is {role!r}, expected 'dev' or 'main'")


def baseline_run_dir(instance_id: str, method: str, run_seed: int, out_root: str | Path | None = None) -> Path:
    """``<out_root or $WEIGHTCODE_RUNS/baselines>/<instance_id>/<method>/<run_seed>``."""
    root = Path(out_root) if out_root is not None else runs_dir() / "baselines"
    return root / instance_id / method / str(int(run_seed))


def looser_than(limits: EditLimits, stated: EditLimits) -> list[str]:
    """Ways in which ``limits`` would allow a sparse search patch that ``stated`` forbids."""
    problems = []
    cap, stated_cap = limits.changed_frac_cap(), stated.changed_frac_cap()
    if stated_cap is not None and (cap is None or cap > stated_cap):
        problems.append(f"changed-entry fraction cap {cap} vs stated {stated_cap}")
    if stated.max_rel_frob is not None:
        if limits.max_rel_frob is None or limits.max_rel_frob > stated.max_rel_frob:
            problems.append(f"max_rel_frob {limits.max_rel_frob} vs stated {stated.max_rel_frob}")
        if limits.rel_frob_min_ndim > stated.rel_frob_min_ndim:
            problems.append(f"rel_frob_min_ndim {limits.rel_frob_min_ndim} vs stated {stated.rel_frob_min_ndim}")
    return problems


def check_task(instance: Instance, limits: EditLimits) -> dict[str, Any]:
    """The task file must state the executor's current limits, and the search limits must not be looser.

    Returns the stated limits for the run record. A task.json whose limits differ from ``weightcode.patch.Limits()``
    was written before a limits change: its editor and this baseline would be held to different limits.
    """
    stated = instance.task.get("limits")
    if not isinstance(stated, Mapping):
        raise ValueError(f"{instance.instance_id}: task.json states no limits")
    executor = task_limits()
    if dict(stated) != executor:
        raise ValueError(
            f"{instance.instance_id}: task.json limits {dict(stated)} differ from weightcode.patch.Limits() {executor}; "
            "the instance predates a limits change"
        )
    problems = looser_than(limits, EditLimits.from_task(stated))
    if problems:
        raise ValueError(f"search limits are looser than the task's: {'; '.join(problems)}")
    return dict(stated)


def _git_state() -> dict[str, Any]:
    root = PACKAGE_DIR.parents[2]
    state: dict[str, Any] = {}
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "weightcode", "scripts"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        state["error"] = f"git state unavailable: {error}"
        return state
    state["head"] = head.stdout.strip()
    state["uncommitted"] = [line for line in dirty.stdout.splitlines() if line.strip()]
    return state


def code_fingerprint() -> dict[str, Any]:
    """sha256 of every module of this package plus the git state, so a run can be tied to exact code."""
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(PACKAGE_DIR.glob("*.py"))}
    return {"package_sha256": files, "git": _git_state()}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def write_json_atomic(path: Path, record: Mapping[str, Any]) -> None:
    """Write ``record`` as JSON through a temporary file and an atomic rename."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, default=_json_default))
    temporary.replace(path)


def _hypothesis(method: str, problem: Any, snap: Snapshot, position: int, total: int) -> str:
    tensors = ", ".join(problem.tensors)
    if method == "arachne":
        what = f"Arachne-style DE over {problem.dim} weights chosen by bidirectional localisation in {tensors}"
        fitness = f"evidence fitness (Eq. 4, higher is better) {snap.fitness:.4f}"
    else:
        what = f"CARE-style PSO rescaling the outputs of {problem.dim} causally localised neurons via {tensors}"
        fitness = f"evidence cost (Eq. 10, lower is better) {snap.fitness:.4f}"
    return f"{what}; best-so-far solution {position + 1} of {total} (step {snap.step}); {fitness}."


def _predicted(readout: Mapping[str, int]) -> str:
    return (
        f"On evidence: {readout['neg_fixed']}/{readout['n_neg']} failures classified correctly, "
        f"{readout['pos_broken']}/{readout['n_pos']} correctly classified random images broken. No claim beyond "
        "the evidence."
    )


def _search(
    method: str,
    net: NetSpec,
    instance: Instance,
    sets: EvidenceSets,
    counted: CountedForward,
    options: RunOptions,
    seeds: Mapping[str, list[int]],
) -> tuple[Any, Any, SearchResult, dict[str, float]]:
    started = time.perf_counter()
    if method == "arachne":
        localisation: Any = arachne.localise(
            net, instance.params, sets, options.arachne, np.random.default_rng(seeds["localisation"]), options.limits
        )
        problem: Any = arachne.ArachneProblem(
            instance.params, localisation.refs, sets, counted, options.arachne.alpha, options.limits
        )
        localised = time.perf_counter()
        result = arachne.run_de(problem, options.arachne, np.random.default_rng(seeds["search"]))
    elif method == "care":
        localisation = care.localise(net, instance.params, sets, options.care, options.limits)
        problem = care.CareProblem(
            instance.params, localisation.selected, sets, counted, options.care.alpha, options.limits
        )
        localised = time.perf_counter()
        result = care.run_pso(problem, options.care, np.random.default_rng(seeds["search"]))
    else:
        raise KeyError(f"unknown method {method!r}; known: {METHODS}")
    return localisation, problem, result, {"localisation_s": localised - started, "search_s": time.perf_counter() - localised}


def _final_record(
    instance: Instance, final: Candidate, log: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    entry = next((item for item in log if item["patch_id"] == final.patch_id), None)
    if final.patch_id is not None and entry is None:
        raise RuntimeError(f"final patch {final.patch_id} is missing from the oracle log")
    candidate = next((item for item in candidates if item["patch_id"] == final.patch_id), None)
    fixed_target = None
    if entry is not None and entry.get("fixed_target") is not None:
        fixed_target = int(entry["fixed_target"])
    elif entry is None and instance.target is not None:
        fixed_target = 0
    return {
        "patch_id": final.patch_id,
        "is_base_noop": final.patch_id is None,
        "dir": final.patch_id,
        "query_index": final.query_index,
        "score": final.score,
        "fixed": int(entry["fixed"]) if entry else 0,
        "broken": int(entry["broken"]) if entry else 0,
        "base_acc": float(entry["base_acc"]) if entry else None,
        "child_acc": float(entry["child_acc"]) if entry else None,
        "fixed_target": fixed_target,
        "n_changed": final.n_changed,
        "edit_size": int(candidate["executor"]["edit_size"]) if candidate else 0,
        "edit_frac": float(candidate["executor"]["edit_frac"]) if candidate else 0.0,
        "child_sha256": str(entry["child_sha256"]) if entry else instance.params_sha256,
    }


def _largest_rel_frob(summary: Mapping[str, Any]) -> tuple[str | None, float]:
    """The changed tensor with the largest relative Frobenius norm in an executor summary, and that norm."""
    largest: tuple[str | None, float] = (None, 0.0)
    for name, stats in summary["per_tensor"].items():
        if stats["changed"] > 0:
            value = float(stats["rel_frob"])
            if largest[0] is None or value > largest[1]:
                largest = (name, value)
    return largest


def _counts(counted: CountedForward) -> dict[str, int]:
    return {"calls": counted.calls, "images": counted.images}


def _minus(after: Mapping[str, int], before: Mapping[str, int]) -> dict[str, int]:
    return {key: int(after[key] - before[key]) for key in after}


def run_search_baseline(
    instance_dir: str | Path,
    method: str,
    run_seed: int,
    allow_main: bool = False,
    out_root: str | Path | None = None,
    options: RunOptions | None = None,
) -> dict[str, Any]:
    """Run ``method`` on one instance with one seed; return the run record, also written to ``run.json``.

    The run directory must not exist yet (a rerun would spend new patch-check queries on the same instance). A
    failed run still writes run.json with status "failed", the error and the oracle log so far, then re-raises.
    """
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; known: {METHODS}")
    options = options or RunOptions()
    started = time.perf_counter()
    instance = load_instance(instance_dir)
    role = check_split_role(instance, allow_main)
    stated_limits = check_task(instance, options.limits)
    run_dir = baseline_run_dir(instance.instance_id, method, run_seed, out_root)
    run_dir.mkdir(parents=True, exist_ok=False)
    module = arachne if method == "arachne" else care
    method_config = options.arachne if method == "arachne" else options.care
    code = METHOD_CODES[method]
    seeds = {"localisation": [int(run_seed), code, 0], "search": [int(run_seed), code, 1]}
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "method": method,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "instance": {
            "instance_id": instance.instance_id,
            "split_role": role,
            "allow_main": allow_main,
            "params_sha256": instance.params_sha256,
            "forward_sha256": instance.forward_sha256,
            "evidence_rule": instance.private.get("evidence_rule"),
            "target": None if instance.target is None else list(instance.target),
        },
        "run_seed": int(run_seed),
        "seeds": seeds,
        "budget": options.query_budget,
        "uses_gradient_for_localization": module.USES_GRADIENT_FOR_LOCALIZATION,
        "flags": {
            "uses_gradient_for_localization": module.USES_GRADIENT_FOR_LOCALIZATION,
            "uses_labels_for_search": True,
            "edit_limits_enforced": True,
            "respects_edit_limits": None,
        },
        "config": {
            "method": method_config.as_dict(),
            "query_budget": options.query_budget,
            "parity_tol": options.parity_tol,
            "query_rule": QUERY_RULE,
        },
        "limits": {
            **options.limits.as_dict(),
            "from_executor": options.limits == EditLimits.from_executor(projection_margin=options.limits.projection_margin),
        },
        "task_limits": stated_limits,
        "executor_limits": task_limits(),
        "deviations": list(module.DEVIATIONS),
        "patch_writer": PATCH_WRITER,
        "patch_applier": PATCH_APPLIER,
        "provenance": {
            "code": code_fingerprint(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "host": platform.node(),
            "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        },
    }
    write_json_atomic(run_dir / "run.json", record)
    timing: dict[str, float] = {}
    oracle = None
    executor_limits = Limits()
    try:
        counted = CountedForward(instance.forward)
        sets = build_evidence_sets(instance.evidence, instance.target, instance.params, counted)
        net = parse_spec(json.loads((instance.instance_dir / "model_spec.json").read_text()), instance.params)
        record["evidence"] = sets.summary()
        record["forward_parity_max_abs"] = check_parity(net, instance.params, counted, sets.x_all, options.parity_tol)
        setup = _counts(counted)
        timing["setup_s"] = time.perf_counter() - started

        localisation, problem, result, search_timing = _search(method, net, instance, sets, counted, options, seeds)
        timing.update(search_timing)
        after_search = _counts(counted)
        record["localisation"] = localisation.as_dict()
        snapshots = result.trajectory.snapshots
        eligible = [i for i, snap in enumerate(snapshots) if count_changed(instance.params, problem.child(snap.vector)) > 0]
        positions = select_for_query(len(eligible), options.query_budget)
        record["search"] = {
            "steps_run": result.steps_run,
            "stop_reason": result.stop_reason,
            "n_fitness_evaluations": result.n_evaluations,
            "fitness_direction": "maximise" if result.maximize else "minimise",
            "best_fitness": result.best_fitness,
            "best_fitness_history": result.best_fitness_history,
            "trajectory": [{"step": s.step, "fitness": s.fitness} for s in snapshots],
            "trajectory_non_noop_indices": eligible,
            "selected_trajectory_indices": [eligible[p] for p in positions],
            "method_final_best_is_noop": bool(snapshots) and (len(snapshots) - 1) not in eligible,
        }
        if method == "care":
            cost, ub, accuracy = problem.evaluate(result.best_vector)
            record["search"]["care_final_checks"] = {
                "cost": cost,
                "ub": ub,
                "accuracy_on_positives": accuracy,
                "property_satisfied": ub < options.care.property_threshold,
                "accuracy_drop": 1.0 - accuracy,
                "accuracy_drop_exceeds_threshold": (1.0 - accuracy) > options.care.accuracy_drop_threshold,
            }

        oracle_started = time.perf_counter()
        checked: list[dict[str, Any]] = []
        record["candidates"] = checked
        if positions:
            oracle = make_oracle(instance, budget=options.query_budget)
        for k, position in enumerate(positions):
            snap = snapshots[eligible[position]]
            child = problem.child(snap.vector)
            patch_id = f"q{k}"
            patch_dir = run_dir / patch_id
            measured = edit_stats(instance.params, child)
            own_violations = limit_violations(measured, options.limits)
            entry: dict[str, Any] = {
                "patch_id": patch_id,
                "dir": patch_id,
                "trajectory_index": eligible[position],
                "step": snap.step,
                "evidence_fitness": snap.fitness,
                "is_method_final_best": eligible[position] == len(snapshots) - 1,
                "search_limit_violations": own_violations,
            }
            checked.append(entry)
            readout_preview = evidence_readout(counted(child, sets.x_all), sets)
            write_sparse_patch(
                patch_dir,
                instance.params_sha256,
                instance.params,
                child,
                _hypothesis(method, problem, snap, position, len(eligible)),
                _predicted(readout_preview),
            )
            try:
                applied = apply_patch_file(instance.params, patch_dir / "patch.json", executor_limits, instance.params_sha256)
            except PatchError as error:
                entry["valid"] = False
                entry["executor_error"] = list(error.messages)
                entry["edit"] = measured.as_dict()
                write_json_atomic(run_dir / "run.json", {**record, "oracle_log": oracle.log if oracle else []})
                continue
            if own_violations:
                raise RuntimeError(f"{patch_id}: the executor accepted a patch that breaks the search limits: {own_violations}")
            entry["valid"] = True
            differing = count_changed(child, applied.child)
            entry["roundtrip_exact"] = differing == 0
            entry["roundtrip_entries_differing"] = differing
            if differing:
                entry["roundtrip_max_abs_diff"] = max(
                    float(np.max(np.abs(applied.child[key].astype(np.float64) - np.asarray(child[key], np.float64))))
                    for key in child
                )
            entry["executor"] = applied.summary()
            largest_name, largest_value = _largest_rel_frob(entry["executor"])
            entry["largest_rel_frob"] = {"tensor": largest_name, "value": largest_value}
            stats = edit_stats(instance.params, applied.child)
            entry["edit"] = {**stats.as_dict(), "within_limits": not limit_violations(stats, options.limits)}
            entry["evidence_readout"] = evidence_readout(counted(applied.child, sets.x_all), sets)
            assert oracle is not None
            result_entry = oracle.query(applied.child, patch_id)
            entry["query_index"] = result_entry["query_index"]
            entry["check"] = result_entry
            write_json_atomic(run_dir / "run.json", {**record, "oracle_log": oracle.log})
        final = oracle.final() if oracle is not None else BASE_CANDIDATE
        log = oracle.log if oracle is not None else []
        record["final"] = _final_record(instance, final, log, checked)
        record["queries_used"] = len(log)
        record["invalid_patches"] = [
            {"patch_id": entry["patch_id"], "executor_error": entry["executor_error"]}
            for entry in checked
            if not entry["valid"]
        ]
        record["flags"]["respects_edit_limits"] = all(
            entry["valid"] and entry["edit"]["within_limits"] for entry in checked
        )
        timing["oracle_s"] = time.perf_counter() - oracle_started
        end = _counts(counted)
        record["forward_counts"] = {
            "setup": setup,
            "search": _minus(after_search, setup),
            "readout": _minus(end, after_search),
            "total": end,
            "reimplementation_analysis_images": localisation.analysis_images,
            "note": "instance forward.py calls and images on evidence; oracle V_check forwards are not included",
        }
        record["status"] = "complete"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if oracle is not None:
            record["oracle_log"] = oracle.log
        timing["wall_s"] = time.perf_counter() - started
        record["timing"] = timing
        write_json_atomic(run_dir / "run.json", record)
    return record


def summary_line(record: Mapping[str, Any]) -> str:
    """One-line human summary of a run record."""
    head = f"{record['instance']['instance_id']} {record['method']} seed {record['run_seed']}: {record['status']}"
    if record["status"] != "complete":
        return f"{head} ({record.get('error')})"
    final = record["final"]
    chosen = "base no-op" if final["is_base_noop"] else final["patch_id"]
    per_query = [f"{e['patch_id']}:{e['score']}({e['fixed']}/{e['broken']})" for e in record.get("oracle_log", [])]
    target = "" if final.get("fixed_target") is None else f" fixed_target={final['fixed_target']}"
    search = record["search"]
    return (
        f"{head}; final {chosen} S={final['score']} fixed={final['fixed']} broken={final['broken']}{target} "
        f"n_changed={final['n_changed']}; queries {record['queries_used']}/{record['budget']}, invalid "
        f"{len(record['invalid_patches'])}; S(fixed/broken) per query {per_query}; search {search['stop_reason']} "
        f"after {search['steps_run']} steps, {search['n_fitness_evaluations']} evaluations; "
        f"wall {record['timing']['wall_s']:.1f}s"
    )
