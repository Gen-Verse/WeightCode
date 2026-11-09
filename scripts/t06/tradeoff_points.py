"""T06 (Execution Plan v1.1 section 10): acquisition-retention operating points from saved candidates.

The rule is docs/PREREG_v0_DRAFT.md, the bullet "T06, the acquisition-retention trade-off from saved candidates"
(main 036958f), fixed before any of its numbers. It is applied as written:

- Bounds on the V_check clean change: at least 0, -0.5, -1.0 and -2.0 points, that is
  S_old * 100 >= bound * n_old (S_old >= 0, -20, -40, -80 of the 4,000 clean V_check rows on E11 and E12).
- Per method and bound: among the method's saved candidates that have both a V_check and a V_select readout, the
  one with the largest V_check S_new inside the bound, ties to the earlier check. It is read on V_select: gain
  (condition rows) and clean change (the clean companion), fixed and broken on both, the share of the base's
  correct rows broken, and the worst class.
- A method with no candidate inside a bound gives the base (gain 0, clean 0) and is marked so. The base is not a
  candidate otherwise, so a pick inside a bound may have S_new <= 0; such picks are marked, not replaced.
- A record that holds V_select readouts only for the kept pick and the run's own pick (the reference methods of
  the bar, except the two second-grid runs that are F1's full and last-layer arms) contributes those candidates
  only, and is marked so.
- Output: a table of operating points (no curve) and a cost table. Development evidence.

Sources (read only; nothing in the store is changed): the LLM session's checked programs
(runs/devreadouts/learn/<sid>/info.json "trajectory", with n_changed from runs/sessions/<sid>/final/selection.json);
every configuration of each F1 and F1s arm (the *_frontier_* readouts, joined to their run.json oracle_log); the G1
search's 8 checks (run.json oracle_log, joined by child sha256 to the family-ceiling readouts); the bar's reference
methods (the *_floor20_* kept picks and *_own_* finals, joined to run.json). Every pick that is the same candidate
as a pick of record is checked against that record (and, for the LLM finals and the G1 picks, against the values
in docs/DEV_RESULTS.md); a mismatch stops the script before anything is written.

Usage:
    python scripts/t06/tradeoff_points.py [--runs-root DIR] [--instances-root DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA = "weightcode.t06.tradeoff_points.v0"
POINTS = 100.0
BOUNDS: tuple[str, ...] = ("0", "-0.5", "-1.0", "-2.0")  # V_check clean change, points (PREREG T06)
KEPT_FLOOR = -20  # joint-retain-v1, the frozen kept rule, used only to check the picks of record
RULE_TEXT = (
    "PREREG T06 (main 036958f): bounds on the V_check clean change >= 0, -0.5, -1.0, -2.0 points "
    "(S_old * 100 >= bound * n_old); per method and bound, among its saved candidates with both a V_check and a "
    "V_select readout, the largest V_check S_new inside the bound, ties to the earlier check, read on V_select; "
    "no candidate inside a bound gives the base (gain 0, clean 0), marked."
)


class RecordError(ValueError):
    """A record that cannot be read as this analysis needs, or a check against a pick of record that fails."""


# ----------------------------------------------------------------------------------------------------------------
# Core rule (pure functions; tested in tests/test_t06_tradeoff_points.py)


@dataclass(frozen=True)
class Readout:
    """One V_select readout (condition rows or their clean companion); delta and per-class values are fractions."""

    delta_acc: float
    fixed: int
    broken: int
    n: int
    n_base_errors: int
    per_class: tuple[float, ...]


@dataclass(frozen=True)
class Candidate:
    """A saved candidate with its V_check scores and its V_select readouts."""

    order: int  # check order (query index); ties in the pick go to the smaller
    cid: str
    config: str
    s_new: int
    s_old: int
    select: Readout
    clean: Readout
    n_changed: int | None = None
    child_sha256: str | None = None
    exec_s: float | None = None  # program execution time, when the record keeps one
    dof: int | None = None  # trainable degrees of freedom of this configuration, when recorded or derivable


def within_bound(s_old: int, n_old: int, bound_points: str) -> bool:
    """True if the V_check clean change S_old / n_old, in points, is at least the bound (exact arithmetic)."""
    return Fraction(s_old) * 100 >= Fraction(bound_points) * n_old


def pick(candidates: Iterable[Candidate], n_old: int, bound_points: str) -> Candidate | None:
    """The candidate with the largest V_check S_new inside the bound, ties to the earlier check; None if none."""
    inside = [c for c in candidates if within_bound(c.s_old, n_old, bound_points)]
    if not inside:
        return None
    return min(inside, key=lambda c: (-c.s_new, c.order))


def pick_over_checks(checks: Iterable[tuple[int, str, int, int]], n_old: int, bound_points: str) -> int | None:
    """Diagnostic: the order of the largest-S_new check inside the bound over every V_check record, whether or not it
    has a V_select readout. It is never used as a pick; it shows where missing readouts could change one."""
    inside = [c for c in checks if within_bound(c[3], n_old, bound_points)]
    if not inside:
        return None
    return min(inside, key=lambda c: (-c[2], c[0]))[0]


def kept_joint_retain_v1(candidates: Iterable[Candidate], floor: int = KEPT_FLOOR) -> Candidate | None:
    """The frozen kept rule, for the checks only: argmax S_joint over candidates with S_old >= floor plus the base
    (S_joint 0, 0 changed entries); ties to fewer changed entries, then the earlier check. None means the base."""
    best: Candidate | None = None
    best_key: tuple[int, int, int] = (0, 0, -1)  # (S_joint, -n_changed, -order) of the base
    for c in candidates:
        if c.s_old < floor:
            continue
        if c.n_changed is None:
            raise RecordError(f"kept-rule check needs n_changed for {c.cid}")
        key = (c.s_new + c.s_old, -c.n_changed, -c.order)
        if key > best_key:
            best, best_key = c, key
    return best


def _share(broken: int, readout: Readout) -> float:
    correct = readout.n - readout.n_base_errors
    if correct <= 0:
        raise RecordError("a readout with no correct base rows")
    return broken / correct


def _worst(per_class: Sequence[float], classes: Sequence[str]) -> dict[str, Any]:
    if not per_class:
        raise RecordError("a readout without per-class changes")
    k = min(range(len(per_class)), key=lambda i: (per_class[i], i))
    return {"class_index": k, "class": classes[k] if k < len(classes) else str(k), "change": per_class[k] * POINTS}


def operating_point(c: Candidate | None, classes: Sequence[str]) -> dict[str, Any]:
    """The V_select reading of a pick; the base (gain 0, clean 0, nothing fixed or broken) when c is None."""
    if c is None:
        zero = {"class_index": None, "class": None, "change": 0.0}
        return {"base_fallback": True, "pick": None, "config": None, "order": None, "vcheck_s_new": 0,
                "vcheck_s_old": 0, "gain": 0.0, "clean": 0.0, "joint": 0.0, "fixed_new": 0, "broken_new": 0,
                "fixed_clean": 0, "broken_clean": 0, "broken_share_new": 0.0, "broken_share_clean": 0.0,
                "worst_class_new": zero, "worst_class_clean": zero, "n_changed": 0, "child_sha256": None,
                "vcheck_s_new_le_0": False, "exec_s": None, "trainable_dof": None}
    gain = c.select.delta_acc * POINTS
    clean = c.clean.delta_acc * POINTS
    return {"base_fallback": False, "pick": c.cid, "config": c.config, "order": c.order, "vcheck_s_new": c.s_new,
            "vcheck_s_old": c.s_old, "gain": gain, "clean": clean, "joint": gain + clean,
            "fixed_new": c.select.fixed, "broken_new": c.select.broken, "fixed_clean": c.clean.fixed,
            "broken_clean": c.clean.broken, "broken_share_new": _share(c.select.broken, c.select),
            "broken_share_clean": _share(c.clean.broken, c.clean),
            "worst_class_new": _worst(c.select.per_class, classes),
            "worst_class_clean": _worst(c.clean.per_class, classes), "n_changed": c.n_changed,
            "child_sha256": c.child_sha256, "vcheck_s_new_le_0": c.s_new <= 0, "exec_s": c.exec_s,
            "trainable_dof": c.dof}


# ----------------------------------------------------------------------------------------------------------------
# Record readers


@dataclass
class Method:
    """One method's saved candidates on one instance, with its pick of record and its cost fields."""

    key: str
    label: str
    family: str
    coverage: str  # "all" (every checked candidate has a V_select readout) or "kept_own" (kept and own picks only)
    candidates: list[Candidate]
    n_checked: int
    kept_of_record: Candidate | None = None  # None with kept_is_base False means no pick of record
    kept_is_base: bool = False
    kept_source: str | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    all_checks: list[tuple[int, str, int, int]] = field(default_factory=list)  # (order, cid, S_new, S_old), V_check

    def checks_on_vcheck(self) -> list[tuple[int, str, int, int]]:
        """Every check on V_check, with or without a V_select readout (diagnostic only)."""
        return self.all_checks or [(c.order, c.cid, c.s_new, c.s_old) for c in self.candidates]


class Store:
    """Read-only access to the store, with the sha256 of every file read."""

    def __init__(self, runs: Path, instances: Path) -> None:
        self.runs = runs
        self.instances = instances
        self.read: dict[str, str] = {}

    def _path(self, rel: str, root: Path) -> Path:
        p = (root / rel).resolve()
        if "sealed" in p.parts:
            raise RecordError(f"refusing to read under sealed/: {p}")
        return p

    def _note(self, p: Path, root_name: str, root: Path) -> None:
        key = f"{root_name}/{p.relative_to(root.resolve())}"
        if key not in self.read:
            self.read[key] = hashlib.sha256(p.read_bytes()).hexdigest()

    def json(self, rel: str) -> Any:
        p = self._path(rel, self.runs)
        self._note(p, "runs", self.runs)
        return json.loads(p.read_text())

    def jsonl(self, rel: str) -> list[dict[str, Any]]:
        p = self._path(rel, self.runs)
        self._note(p, "runs", self.runs)
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def sha(self, rel: str) -> str:
        p = self._path(rel, self.runs)
        self._note(p, "runs", self.runs)
        return self.read[f"runs/{p.relative_to(self.runs.resolve())}"]

    def classes(self, instance_id: str) -> list[str]:
        p = self._path(f"{instance_id}/task.json", self.instances)
        self._note(p, "private/instances", self.instances)
        return [str(c) for c in json.loads(p.read_text())["classes"]]

    def run_rel(self, abs_run: str) -> str:
        """A run.json path as a record names it (absolute, under some store root) made relative to runs/."""
        marker = "/runs/"
        if marker not in abs_run:
            raise RecordError(f"run path without /runs/: {abs_run}")
        return abs_run.split(marker, 1)[1]


def _readout(r: dict[str, Any]) -> Readout:
    return Readout(delta_acc=float(r["delta_acc"]), fixed=int(r["fixed"]), broken=int(r["broken"]), n=int(r["n"]),
                   n_base_errors=int(r["n_base_errors"]),
                   per_class=tuple(float(x) for x in r["per_class_acc_delta"]))


def _config(cfg: Any) -> str:
    if not isinstance(cfg, dict):
        return str(cfg)
    return ", ".join(f"{k} {v}" for k, v in cfg.items())


def _oracle_by_index(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(o["query_index"]): o for o in run["oracle_log"]}


def _all_checks(run: dict[str, Any]) -> list[tuple[int, str, int, int]]:
    return [(int(o["query_index"]), f"q{int(o['query_index'])}", int(o["score_new"]), int(o["score_old"]))
            for o in run["oracle_log"]]


def _candidate_configs(run: dict[str, Any]) -> dict[int, str]:
    return {int(c["slot"]): _config(c.get("config")) for c in run.get("candidates", []) if "slot" in c}


def _fit_cost(run: dict[str, Any]) -> dict[str, Any]:
    cfg = run.get("config", {})
    timing = run.get("timing", {})
    return {"generate_s": timing.get("generate_s"), "check_s": timing.get("check_s"), "wall_s": timing.get("wall_s"),
            "torch": cfg.get("torch"), "threads": cfg.get("torch_threads"), "device": cfg.get("device")}


def _dof(run: dict[str, Any], lora_reference: dict[str, Any] | None = None,
         mode: str | None = None) -> tuple[int | None, str]:
    """Trainable degrees of freedom as recorded (or derived, and then said so). F1s records both a full and a
    LoRA trainable list; which one a configuration trains is its config's mode (s4, s5 mix both), or the arm (s3
    is LoRA only, s1 and s2 full only)."""
    gen = run.get("generation", {})
    lora = gen.get("lora")
    if isinstance(lora, dict) and lora.get("trained_entries") is not None:
        return int(lora["trained_entries"]), "recorded (generation.lora.trained_entries)"
    mask = gen.get("mask")
    if isinstance(mask, dict) and mask.get("n_mask_entries") is not None:
        return int(mask["n_mask_entries"]), "recorded (generation.mask.n_mask_entries)"
    if gen.get("trainable_entries") is not None:
        return int(gen["trainable_entries"]), "recorded (generation.trainable_entries)"
    trainable = gen.get("trainable")
    per_tensor = ((run.get("candidates") or [{}])[0].get("edit") or {}).get("per_tensor") or {}
    if isinstance(trainable, dict):  # F1s records both lists; only the s3 arm trains the LoRA factors (PREREG F1s)
        uses_lora = mode == "lora" if mode is not None else run.get("method") == F1S_LORA_ARM
        names = trainable.get("lora" if uses_lora else "full") or []
        if uses_lora:
            ref = (lora_reference or {}).get("generation", {}).get("lora", {})
            rank = (run.get("config", {}).get("lora") or {}).get("rank")
            if ref.get("targets") == names and ref.get("rank") == rank and ref.get("trained_entries") is not None:
                return int(ref["trained_entries"]), "derived: same targets and rank as F1 lora_r4's record"
            return None, "not recorded"
        trainable = names
    if isinstance(trainable, list) and trainable and per_tensor and all(t in per_tensor for t in trainable):
        return sum(int(per_tensor[t]["size"]) for t in trainable), "derived: sum of the trainable tensors' sizes"
    return None, "not recorded"


def load_session(store: Store, sid: str, label: str) -> Method:
    """The LLM session's checked programs, with n_changed and the program's execution time."""
    info = store.json(f"devreadouts/learn/{sid}/info.json")
    selection = store.json(f"sessions/{sid}/final/selection.json")
    sel_by_index = {int(c["query_index"]): c for c in selection["candidates"]}
    cands = []
    for row in info["trajectory"]:
        qi = int(row["query_index"])
        if not (row.get("vcheck") and row.get("vselect") and row.get("vselect_clean")):
            continue
        sel = sel_by_index.get(qi)
        if sel is None or sel["child_sha256"] != row["child_sha256"]:
            raise RecordError(f"{sid} q{qi}: selection.json does not hold this child")
        run = store.json(f"sessions/{sid}/attempts/{sel['attempt']}/program_run.json")
        cands.append(Candidate(order=qi, cid=f"q{qi} {row['patch_id']}", config=row["patch_id"],
                               s_new=int(row["vcheck"]["score_new"]), s_old=int(row["vcheck"]["score_old"]),
                               select=_readout(row["vselect"]), clean=_readout(row["vselect_clean"]),
                               n_changed=int(sel["n_changed"]), child_sha256=row["child_sha256"],
                               exec_s=float(run["run"]["duration_s"])))
    final = info["final"]
    kept = next((c for c in cands if c.order == int(final["query_index"])), None) if not final["is_base_noop"] else None
    usage = info["usage"]
    final_run = store.json(f"sessions/{sid}/final/program_run.json")
    cost = {"session_wall_s": usage["wall_s"], "output_tokens": usage["tokens"]["output_tokens"],
            "thinking_tokens": usage["tokens"].get("thinking_tokens"), "checks_used": usage["queries"]["used"],
            "model": usage["editor"]["model"], "effort": usage["editor"]["effort"],
            "sandbox_threads": final_run["sandbox"]["threads"], "final_program_run_s": final_run["run"]["duration_s"],
            "trainable_dof": None, "trainable_dof_source": "not applicable (a written program)"}
    return Method(key="llm_b_final", label=label, family="LLM session (B final)", coverage="all", candidates=cands,
                  n_checked=int(usage["queries"]["used"]), kept_of_record=kept,
                  kept_is_base=bool(final["is_base_noop"]), kept_source=f"devreadouts/learn/{sid}/info.json final",
                  cost=cost)


def _kept_from_floor20(row: dict[str, Any] | None, cands: Sequence[Candidate], source: str) -> tuple[Candidate | None,
                                                                                                    bool, str | None]:
    if row is None:
        return None, False, None
    if row["is_base_noop"]:
        return None, True, source
    qi = int(row["selection"]["query_index"])
    hit = [c for c in cands if c.order == qi]
    if len(hit) != 1:
        raise RecordError(f"{source}: kept q{qi} is not among the candidates")
    return hit[0], False, source


def _by_method(rows: Sequence[dict[str, Any]], method: str, source: str) -> dict[str, Any]:
    hit = [r for r in rows if r["method"] == method]
    if len(hit) != 1:
        raise RecordError(f"{source}: {len(hit)} rows for {method}")
    return hit[0]


def load_frontier(store: Store, frontier: str, method: str, key: str, label: str, family: str,
                  floor20: str | None, own: str | None, lora_reference: str | None = None) -> Method:
    """Every checked configuration of one arm, read on V_select (a *_frontier_* readout), joined to its run.json."""
    rows = [r for r in store.jsonl(frontier) if r["method"] == method]
    if not rows:
        raise RecordError(f"{frontier}: no rows for {method}")
    runs = {r["run"] for r in rows}
    if len(runs) != 1:
        raise RecordError(f"{frontier}: {method} rows from {len(runs)} runs")
    run_rel = store.run_rel(runs.pop())
    run = store.json(run_rel)
    oracle = _oracle_by_index(run)
    cands = []
    for r in rows:
        qi = int(r["query_index"])
        o = oracle.get(qi)
        if o is None or o["child_sha256"] != r["child_sha256"]:
            raise RecordError(f"{frontier} {method} q{qi}: child differs from run.json oracle_log")
        vc = r["vcheck"]
        if (int(vc["score_new"]), int(vc["score_old"])) != (int(o["score_new"]), int(o["score_old"])):
            raise RecordError(f"{frontier} {method} q{qi}: V_check differs from run.json oracle_log")
        cands.append(Candidate(order=qi, cid=f"q{qi}", config=_config(r.get("config")), s_new=int(vc["score_new"]),
                               s_old=int(vc["score_old"]), select=_readout(r["readouts"]),
                               clean=_readout(r["readouts_clean"]), n_changed=int(o["n_changed"]),
                               child_sha256=r["child_sha256"]))
    if len(cands) != len(run["oracle_log"]):
        raise RecordError(f"{frontier} {method}: {len(cands)} readouts for {len(run['oracle_log'])} checks")
    m = Method(key=key, label=label, family=family, coverage="all", candidates=sorted(cands, key=lambda c: c.order),
               n_checked=len(run["oracle_log"]), cost=_fit_cost(run))
    ref_run = store.json(lora_reference) if lora_reference else None
    modes = {int(c["slot"]): (c.get("config") or {}).get("mode") for c in run.get("candidates", [])}
    per_cand = {qi: _dof(run, ref_run, mode) for qi, mode in modes.items()}
    m.candidates = [replace(c, dof=per_cand[c.order][0]) for c in m.candidates]
    if len({v for v in per_cand.values()}) == 1:
        dof, dof_src = next(iter(per_cand.values()))
    else:
        dof, dof_src = None, "per configuration: " + "; ".join(
            sorted({f"{mo} {_dof(run, ref_run, mo)[0]:,} ({_dof(run, ref_run, mo)[1]})" for mo in modes.values()
                    if mo is not None}))
    m.cost.update({"trainable_dof": dof, "trainable_dof_source": dof_src, "run": run_rel})
    if floor20 is not None:
        row = _by_method(store.jsonl(floor20), method, floor20)
        m.kept_of_record, m.kept_is_base, m.kept_source = _kept_from_floor20(row, m.candidates, floor20)
        _check_same_readouts(row, m.kept_of_record, floor20, method)
    if own is not None:
        orow = _by_method(store.jsonl(own), method, own)
        fin = run["final"]
        if bool(orow["is_base_noop"]) != bool(fin.get("is_base_noop")):
            raise RecordError(f"{own} {method}: own row and run.json final disagree on the base")
        if not orow["is_base_noop"]:
            c = next(c for c in m.candidates if c.order == int(fin["query_index"]))
            _check_same_readouts(orow, c, own, method)
    return m


def _check_same_readouts(row: dict[str, Any], c: Candidate | None, source: str, method: str) -> None:
    """A kept or own record row and the candidate it names must hold the same V_select readouts."""
    if c is None or row.get("is_base_noop"):
        return
    a, b = _readout(row["readouts"]), _readout(row["readouts_clean"])
    if (a, b) != (c.select, c.clean):
        raise RecordError(f"{source} {method}: the record's readouts differ from the candidate {c.cid}")


def load_kept_own(store: Store, floor20: str, own: str, method: str, key: str, label: str, family: str) -> Method:
    """A record holding only the kept pick (floor20) and the run's own final: those candidates only."""
    krow = _by_method(store.jsonl(floor20), method, floor20)
    orow = _by_method(store.jsonl(own), method, own)
    run_rel = store.run_rel(krow["run"])
    if store.run_rel(orow["run"]) != run_rel:
        raise RecordError(f"{method}: kept and own rows name different runs")
    run = store.json(run_rel)
    oracle = _oracle_by_index(run)
    configs = _candidate_configs(run)
    cands: dict[int, Candidate] = {}

    def add(row: dict[str, Any], qi: int, source: str) -> None:
        o = oracle[qi]
        vc = row["vcheck"]
        if (int(vc["score"]), int(vc["fixed"]), int(vc["broken"])) != (int(o["score"]), int(o["fixed"]),
                                                                       int(o["broken"])):
            raise RecordError(f"{source} {method}: V_check differs from run.json oracle_log q{qi}")
        c = Candidate(order=qi, cid=f"q{qi}", config=configs.get(qi, ""), s_new=int(o["score_new"]),
                      s_old=int(o["score_old"]), select=_readout(row["readouts"]),
                      clean=_readout(row["readouts_clean"]), n_changed=int(o["n_changed"]),
                      child_sha256=o["child_sha256"])
        if qi in cands and (cands[qi].select, cands[qi].clean) != (c.select, c.clean):
            raise RecordError(f"{method} q{qi}: kept and own readouts of one candidate differ")
        cands[qi] = c

    if not krow["is_base_noop"]:
        add(krow, int(krow["selection"]["query_index"]), floor20)
    if not orow["is_base_noop"]:
        fin = run["final"]
        if fin.get("is_base_noop"):
            raise RecordError(f"{own} {method}: own row is not the base but run.json final is")
        add(orow, int(fin["query_index"]), own)
    m = Method(key=key, label=label, family=family, coverage="kept_own",
               candidates=sorted(cands.values(), key=lambda c: c.order), n_checked=len(run["oracle_log"]),
               cost=_fit_cost(run), all_checks=_all_checks(run))
    dof, dof_src = _dof(run)
    m.candidates = [replace(c, dof=dof) for c in m.candidates]  # before the kept pick is bound to a candidate
    m.kept_of_record, m.kept_is_base, m.kept_source = _kept_from_floor20(krow, m.candidates, floor20)
    m.cost.update({"trainable_dof": dof, "trainable_dof_source": dof_src, "run": run_rel})
    return m


def load_g1(store: Store, search: str, ceiling: Sequence[str], run_dir: str, label: str) -> Method:
    """G1's 8 checks (run.json oracle_log), read on V_select through the family ceiling (joined by child sha256)."""
    run = store.json(f"{run_dir}/run.json")
    pick_rows = [r for r in store.jsonl(search) if r.get("stage") == "pick"]
    if len(pick_rows) != 1:
        raise RecordError(f"{search}: {len(pick_rows)} pick rows")
    prow = pick_rows[0]
    if not str(prow["run_dir"]).endswith(run_dir) or prow["run_json_sha256"] != store.sha(f"{run_dir}/run.json"):
        raise RecordError(f"{search}: the pick row does not name {run_dir}/run.json as read")
    by_sha: dict[str, dict[str, Any]] = {}
    for shard in ceiling:
        side = Path(store.runs / f"{shard}.sha256").read_text().split()[0]
        if side != store.sha(shard):
            raise RecordError(f"{shard}: sha256 differs from its .sha256 file")
        for r in store.jsonl(shard):
            if r.get("stage") == "ceiling":
                by_sha.setdefault(r["child_sha256"], r)
    checks = {int(c["query_index"]): c for c in prow["checks"]}
    cands = []
    missing = []
    for o in run["oracle_log"]:
        qi = int(o["query_index"])
        chk = checks[qi]
        if (int(chk["score_new"]), int(chk["score_old"])) != (int(o["score_new"]), int(o["score_old"])):
            raise RecordError(f"G1 q{qi}: search checks differ from run.json oracle_log")
        r = by_sha.get(o["child_sha256"])
        if r is None:
            missing.append(qi)
            continue
        n, nc = int(r["n"]), int(r["n_clean"])
        sel = Readout(delta_acc=float(r["gain"]) / POINTS, fixed=int(r["fixed"]), broken=int(r["broken"]), n=n,
                      n_base_errors=round(n * (1 - float(r["base_acc"]))),
                      per_class=tuple(float(x) / POINTS for x in r["per_class_gain"]))
        cln = Readout(delta_acc=float(r["clean"]) / POINTS, fixed=int(r["fixed_clean"]),
                      broken=int(r["broken_clean"]), n=nc, n_base_errors=round(nc * (1 - float(r["base_acc_clean"]))),
                      per_class=tuple(float(x) / POINTS for x in r["per_class_clean"]))
        cands.append(Candidate(order=qi, cid=f"q{qi}", config=str(chk["config"]["name"]), s_new=int(o["score_new"]),
                               s_old=int(o["score_old"]), select=sel, clean=cln, n_changed=int(o["n_changed"]),
                               child_sha256=o["child_sha256"]))
    m = Method(key="g1_search", label=label, family="G1 search (no LLM)", coverage="all", candidates=cands,
               n_checked=len(run["oracle_log"]), cost=_fit_cost(run), all_checks=_all_checks(run))
    if missing:
        m.coverage = "partial"
        names = ", ".join(f"q{q} {checks[q]['config']['name']}" for q in missing)
        m.notes.append(f"{len(missing)} checks have no V_select readout (the family ceiling reads single members "
                       f"only): {names}")
    # The kept pick of record is the search row's joint-retain-v1 selection; run.json "final" is the run's own
    # (no-floor) pick, as for the references.
    sel_qi = prow["selection"]["query_index"]
    m.kept_is_base = sel_qi is None
    m.kept_of_record = None if m.kept_is_base else next(c for c in cands if c.order == int(sel_qi))
    m.kept_source = f"{search} pick row selection"
    pv = prow["pick_vcheck"]
    want = (0, 0) if m.kept_is_base else (m.kept_of_record.s_new, m.kept_of_record.s_old)
    if (int(pv["score_new"]), int(pv["score_old"])) != want:
        raise RecordError("G1: the pick row's V_check differs from the kept check's")
    if not m.kept_is_base:
        k = m.kept_of_record
        if (round(k.select.delta_acc * POINTS, 9), round(k.clean.delta_acc * POINTS, 9)) != (
                round(float(prow["gain"]), 9), round(float(prow["clean"]), 9)):
            raise RecordError("G1: the ceiling readout of the kept check differs from the search pick row")
    m.cost.update({"trainable_dof": None, "trainable_dof_source": "not applicable (a fixed family member)",
                   "run": f"{run_dir}/run.json",
                   "note": "generate_s covers building the 8 checked members; the singles/proxy stage over the "
                           "family is not timed in the run record"})
    return m


# ----------------------------------------------------------------------------------------------------------------
# Instances (fixed in PREREG T06 and its brief before any number; E10, E13 and E14 are not covered, see NOT_COVERED)

F1_ARMS_E11 = (("ft_ln", "F1 LayerNorm only"), ("lora_r4", "F1 LoRA r4"), ("lora_r16", "F1 LoRA r16"),
               ("ft_full_noreplay", "F1 full, no replay"), ("ft_full_lwf", "F1 full + LwF"),
               ("ft_full_l2sp", "F1 full + L2-SP"))
F1_ARMS_E12 = (("ft_bn", "F1 BatchNorm only"),) + F1_ARMS_E11[1:]
F1S_LORA_ARM = "f1s_s3_lora_aug"
F1S_ARMS = (("f1s_s1_gentle", "F1s s1 gentle"), ("f1s_s2_aug", "F1s s2 crop, flip, minibatches"),
            ("f1s_s3_lora_aug", "F1s s3 LoRA r4 + s2"), ("f1s_s4_est_prior", "F1s s4 estimated corruption"),
            ("f1s_s5_oracle_prior", "F1s s5 oracle corruption (more information)"))
REFS_FIRST_E11 = ("grad_last", "grad_full", "ridge_last", "prior_shift", "tie_inputs", "input_stats",
                  "grad_sparse19_sal", "grad_sparse19_traj", "interp", "tent_ln", "sar")
REFS_FIRST_E12 = ("grad_last", "grad_full", "ridge_last", "prior_shift", "tie_inputs", "input_stats",
                  "grad_sparse19_sal", "grad_sparse19_traj", "interp", "bn_adapt", "tent")
REFS_SECOND = ("grad_sparse19_sal", "grad_sparse19_traj", "interp")  # grad_full, grad_last: F1's full, last layer


@dataclass(frozen=True)
class InstanceSpec:
    """Where one instance's records are (paths relative to the runs root)."""

    experiment: str
    instance_id: str
    condition: str
    session: str
    f1_prefix: str
    grid2: str
    f1_arms: tuple[tuple[str, str], ...]
    refs_first: tuple[str, ...]
    ref_suffix: str  # e.g. "_dev_20261104" (first grid), see refs_files
    ref2_suffix: str
    f1s: bool
    g1_run_dir: str | None
    g1_cond: str | None
    lora_ref: str  # F1 lora_r4 run.json (for F1s s3's derived DoF)


INSTANCES: tuple[InstanceSpec, ...] = (
    InstanceSpec("E11", "i00802425", "gaussian_noise_s5", "sc918e11dceef637a", "f1", "e11g2", F1_ARMS_E11,
                 REFS_FIRST_E11, "_dev_20261104", "_dev_20261105", True, "e11/g1/runs/g1_i00802425_1793874179",
                 "gaussian_noise_s5", "baselines/i00802425/lora_r4/f1/0/run.json"),
    InstanceSpec("E11", "i3eb3fc78", "contrast_s5", "s760cb52ce82fb937", "f1", "e11g2", F1_ARMS_E11,
                 REFS_FIRST_E11, "_dev_20261104", "_dev_20261105", True, "e11/g1/runs/g1_i3eb3fc78_1793874274",
                 "contrast_s5", "baselines/i3eb3fc78/lora_r4/f1/0/run.json"),
    InstanceSpec("E12", "i26d23d3e", "gaussian_noise_s5", "s2abaefab30d20b42", "f1e12", "e12g2", F1_ARMS_E12,
                 REFS_FIRST_E12, "_dev_nscc_20261105", "_dev_nscc_20261105", False, None, None,
                 "baselines/i26d23d3e/lora_r4/f1/0/run.json"),
    InstanceSpec("E12", "ie3840b28", "contrast_s5", "sccd12677dcdd125b", "f1e12", "e12g2", F1_ARMS_E12,
                 REFS_FIRST_E12, "_dev_nscc_20261105", "_dev_nscc_20261105", False, None, None,
                 "baselines/ie3840b28/lora_r4/f1/0/run.json"),
)

NOT_COVERED = {
    "E10 (ResNet-20; id14b4f0b, id64b2abc, if90033fc, i1bf0c953)": (
        "the B sessions kept per-candidate V_select readouts, but no comparison method did: the references hold "
        "V_select readouts for the kept and own picks only and no F1 or G1 frontier was run on E10, so no "
        "non-LLM method there has per-candidate readouts (criterion fixed before any T06 number)"),
    "E13 SCAN (ncc5c260f, n8826c49e) and E14 BabyAI (p6ce8ad67)": (
        "the sessions' trajectories carry no per-candidate V_select readout (info.json trajectory rows without "
        "vselect), so the LLM side of the table cannot be built"),
    "other E11 arms on these students (K1, ladder U/K0b/K0a, A4a/A4b, D1, D2, K1 ft_rule, A3 T0/T1)": (
        "not in the PREREG T06 method list (the LLM session's checked programs, every F1 arm, the G1 search, the "
        "bar's reference methods); they answer other information settings"),
    "E11 seeds 101 and 102 (i905213a3, iab18c0aa, i4a486d70, i5d76f25c)": (
        "no B session has run on these students (the replication sessions wait for the freeze)"),
}

# Picks of record quoted in docs/DEV_RESULTS.md (gain, clean in points), checked when T06's -0.5 pick is that pick.
DEV_RESULTS_PICKS: dict[tuple[str, str], tuple[float, float, str]] = {
    ("i00802425", "llm_b_final"): (31.625, 0.125, "E11 B gaussian and mechanism A1 entries (+31.625, +0.125)"),
    ("i3eb3fc78", "llm_b_final"): (17.925, 0.05, "E11 B contrast entry (gain +17.93, clean +0.05, joint +17.975)"),
    ("i26d23d3e", "llm_b_final"): (34.975, -0.375, "E12 B gaussian entry (gain +34.975, clean -0.375)"),
    ("ie3840b28", "llm_b_final"): (21.475, -0.325, "E12 B contrast entry (gain +21.475, clean -0.325)"),
    ("i00802425", "g1_search"): (24.2, -0.075, "G1 entry (gain +24.20, clean -0.07, joint +24.12)"),
}

# Documented sha256 of record files (docs/PREREG_v0_DRAFT.md; G1 and E12 info.json prefixes in docs/DEV_RESULTS.md).
EXPECTED_SHA256: dict[str, str] = {
    "devreadouts/e11_refs_i00802425_gaussian_noise_s5_floor20_dev_20261104.jsonl":
        "73c6be89eb0974ca84243a2ab721942c8b43737b5a248f5493aee1a40f42a2b3",
    "devreadouts/e11_refs_i00802425_gaussian_noise_s5_own_dev_20261104.jsonl":
        "e6b592802db189f1e566f8d8924e0e7aff7b8ab62de350570bb2655b122b32aa",
    "devreadouts/e11_refs_i3eb3fc78_contrast_s5_floor20_dev_20261104.jsonl":
        "a7adc9dd52c5dd59d7d923547f1154426769b537916c93d5371850af9841ce3b",
    "devreadouts/e11_refs_i3eb3fc78_contrast_s5_own_dev_20261104.jsonl":
        "9ce47687ef5bbeb7d0b93d71886558154ea10d4de68f5516101abf1d5a56bb9a",
    "devreadouts/e11g2_refs_i00802425_gaussian_noise_s5_floor20_dev_20261105.jsonl":
        "dcf72c2ba393f7851e693a2f0daed805f25a5ab26c6d85ef78eb576cc1c40f3e",
    "devreadouts/e11g2_refs_i00802425_gaussian_noise_s5_own_dev_20261105.jsonl":
        "816ff4b486090841a804a2d3dfa2be96123d907ac926b089102d32b093a4fa6d",
    "devreadouts/e11g2_refs_i3eb3fc78_contrast_s5_floor20_dev_20261105.jsonl":
        "e1a530af758d35db881c49ab72cdb4069ef3917ce517c88efbbad610c07599a3",
    "devreadouts/e11g2_refs_i3eb3fc78_contrast_s5_own_dev_20261105.jsonl":
        "e5cfad7c9428b2d276514052e929c87d0108f18fb48d53268fc4383f44f61f07",
    "devreadouts/g1/g1_gaussian_noise_s5_search.jsonl": "fdcfb8f3711df90d",
    "devreadouts/g1/g1_contrast_s5_search.jsonl": "ede8edcdc0ea38bd",
    "e11/g1/runs/g1_i00802425_1793874179/run.json": "9fcd9558af6bf860",
    "e11/g1/runs/g1_i3eb3fc78_1793874274/run.json": "7101c31f82029054",
    "devreadouts/learn/s2abaefab30d20b42/info.json": "80ff4a11a1ba2dc5",
    "devreadouts/learn/sccd12677dcdd125b/info.json": "1c6855920266485d",
}


def load_instance(store: Store, spec: InstanceSpec) -> list[Method]:
    """Every covered method's candidates on one instance, in the table's order."""
    i, cond, dv, nscc = spec.instance_id, spec.condition, "devreadouts", "_dev_nscc_20261105.jsonl"
    methods = [load_session(store, spec.session, f"LLM B final ({spec.session})")]
    f1 = f"{dv}/{spec.f1_prefix}_{i}_{cond}"
    g2_front = f"{dv}/{spec.f1_prefix}_{spec.grid2}_{i}_{cond}_frontier{nscc}"
    g2 = f"{dv}/{spec.grid2}_refs_{i}_{cond}"
    g2_kept, g2_own = f"{g2}_floor20{spec.ref2_suffix}.jsonl", f"{g2}_own{spec.ref2_suffix}.jsonl"
    for method, label in (("grad_full", "F1 full (grad_full, second grid)"),
                          ("grad_last", "F1 last layer (grad_last, second grid)")):
        methods.append(load_frontier(store, g2_front, method, f"f1_{method}", label, "F1", g2_kept, g2_own))
    for method, label in spec.f1_arms:
        methods.append(load_frontier(store, f"{f1}_frontier{nscc}", method, f"f1_{method}", label, "F1",
                                     f"{f1}_floor20{nscc}", f"{f1}_own{nscc}"))
    if spec.f1s:
        for arm, label in F1S_ARMS:
            base = f"{dv}/f1s_{i}_{cond}_{arm}"
            methods.append(load_frontier(store, f"{base}_frontier{nscc}", arm, arm, label, "F1s",
                                         f"{base}_floor20{nscc}", f"{base}_own{nscc}", lora_reference=spec.lora_ref))
    if spec.g1_run_dir is not None:
        c = spec.g1_cond
        shards = [f"{dv}/g1/g1_{c}_ceiling_{k}of6.jsonl" for k in range(6)]
        methods.append(load_g1(store, f"{dv}/g1/g1_{c}_search.jsonl", shards, spec.g1_run_dir, "G1 search"))
    r1 = f"{dv}/{spec.experiment.lower()}_refs_{i}_{cond}"
    r1_kept, r1_own = f"{r1}_floor20{spec.ref_suffix}.jsonl", f"{r1}_own{spec.ref_suffix}.jsonl"
    for method in spec.refs_first:
        methods.append(load_kept_own(store, r1_kept, r1_own, method, f"ref1_{method}", f"ref {method} (first grid)",
                                     "reference, first grid"))
    for method in REFS_SECOND:
        methods.append(load_kept_own(store, g2_kept, g2_own, method, f"ref2_{method}", f"ref {method} (second grid)",
                                     "reference, second grid"))
    keys = [m.key for m in methods]
    if len(set(keys)) != len(keys):
        raise RecordError(f"{i}: duplicate method keys")
    return methods


# ----------------------------------------------------------------------------------------------------------------
# Checks against the picks of record


def check_method(m: Method, instance_id: str, n_old: int, classes: Sequence[str]) -> dict[str, Any]:
    """Compare T06's -0.5 pick (the kept floor) with the pick of record; failures are returned, not hidden."""
    t06 = pick(m.candidates, n_old, "-0.5")
    out: dict[str, Any] = {"method": m.key, "kept_source": m.kept_source,
                           "kept_of_record": "base" if m.kept_is_base else (m.kept_of_record.cid if m.kept_of_record
                                                                            else None),
                           "t06_pick_at_-0.5": t06.cid if t06 else "base", "failures": []}
    if m.kept_source is None:
        out["status"] = "no pick of record"
        return out
    if m.coverage == "all":
        rep = kept_joint_retain_v1(m.candidates)
        same_rep = (rep is None and m.kept_is_base) or (rep is not None and rep is m.kept_of_record)
        out["kept_rule_reproduced"] = same_rep
        if not same_rep:
            out["failures"].append(f"joint-retain-v1 on the candidates gives {rep.cid if rep else 'base'}, "
                                   f"the record {out['kept_of_record']}")
    same = (t06 is None and m.kept_is_base) or (t06 is not None and t06 is m.kept_of_record)
    out["same_candidate"] = same
    if not same:
        # The two rules may differ only as their definitions allow; anything else is a reader defect.
        k = m.kept_of_record
        if t06 is None:
            out["status"] = "differs: T06 gives the base"
            out["failures"].append("T06 gives the base inside -0.5 although the kept pick of record is inside it")
        elif m.kept_is_base:
            explained = t06.s_new + t06.s_old <= 0
            out["status"] = ("differs: the kept rule keeps the base (no eligible S_joint above 0); T06 has no base "
                             "candidate and takes the largest S_new inside the bound")
            if not explained:
                out["failures"].append(f"kept is the base but T06's {t06.cid} has S_joint above 0 inside -0.5")
        else:
            explained = (t06.s_new, -t06.order) > (k.s_new, -k.order) and k.s_new + k.s_old >= t06.s_new + t06.s_old
            out["status"] = "differs: argmax S_new (T06) against argmax S_joint (kept rule)"
            if not explained:
                out["failures"].append(f"T06's {t06.cid} and the kept {k.cid} differ in a way the rules do not allow")
        return out
    out["status"] = "same candidate"
    exp = DEV_RESULTS_PICKS.get((instance_id, m.key))
    if exp is not None:
        op = operating_point(t06, classes)
        gain, clean, src = exp
        ok = abs(op["gain"] - gain) < 1e-9 and abs(op["clean"] - clean) < 1e-9
        out["dev_results"] = {"gain": gain, "clean": clean, "source": src, "equal": ok}
        if not ok:
            out["failures"].append(f"DEV_RESULTS {src}: {gain} / {clean}, T06 {op['gain']} / {op['clean']}")
    return out


# ----------------------------------------------------------------------------------------------------------------
# Output


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _f(x: float | None, nd: int = 3, sign: bool = True) -> str:
    if x is None:
        return "n/r"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _int(x: int | None) -> str:
    return "n/r" if x is None else f"{x:,}"


def _cov(m: dict[str, Any]) -> str:
    n = m["n_candidates"]
    if m["coverage"] == "kept_own":
        return f"{n} of {m['n_checked']} (kept and own only)"
    if m["coverage"] == "partial":
        return f"{n} of {m['n_checked']} (the others have no V_select readout)"
    return f"{n} of {m['n_checked']}"


def render_markdown(result: dict[str, Any]) -> str:
    """The operating-point and cost tables as Markdown (numbers only, no figure)."""
    L: list[str] = []
    L.append("# T06 operating points: acquisition against retention, from saved candidates")
    L.append("")
    L.append(f"Rule: {result['rule']}")
    L.append("")
    L.append("Values are V_select points: gain on the 4,000 condition rows, clean change on the 4,000 clean rows, "
             "joint = gain + clean. The bound is on the V_check clean change. Marks: B = no candidate inside the "
             "bound, so the base; K = the record holds V_select readouts only for the kept and own picks; "
             "S- = the pick's V_check S_new is <= 0 (the rule as written has no base candidate inside a bound; "
             "under the T06 amendment of PREREG, written after this point was seen, such a pick reads as the base, "
             "gain 0 and clean 0, and both readings are reported); "
             "+ = a check without a V_select readout has a larger V_check S_new inside the bound, so a fuller "
             "record could move this point. Development evidence (seed 100, one run each); no curve is drawn "
             "between points.")
    L.append("")
    L.append(f"Generated by scripts/t06/tradeoff_points.py at commit {result['provenance']['git_head']}.")
    for inst in result["instances"]:
        L.append("")
        L.append(f"## {inst['experiment']} {inst['condition']} ({inst['instance_id']})")
        L.append("")
        L.append("Gain / clean change (points) at each bound on the V_check clean change:")
        L.append("")
        L.append("| method | candidates | " + " | ".join(f">= {b}" for b in BOUNDS) + " |")
        L.append("|---|---|" + "---|" * len(BOUNDS))
        for m in inst["methods"]:
            cells = []
            for b in BOUNDS:
                p = m["points"][b]
                mark = "B" if p["base_fallback"] else ""
                if m["coverage"] == "kept_own":
                    mark += "K"
                if p["vcheck_s_new_le_0"]:
                    mark += " S- (base under the amendment)"
                if p["missing_readout_would_change_pick"]:
                    mark += " +"
                cells.append(f"{_f(p['gain'])} / {_f(p['clean'])}" + (f" {mark.strip()}" if mark.strip() else ""))
            L.append(f"| {m['label']} | {_cov(m)} | " + " | ".join(cells) + " |")
        notes = [f"- {m['label']}: {n}" for m in inst["methods"] for n in m["notes"]]
        if notes:
            L.append("")
            L += notes
        L.append("")
        L.append("Fixed / broken at each bound, condition rows; clean rows:")
        L.append("")
        L.append("| method | " + " | ".join(f">= {b}" for b in BOUNDS) + " |")
        L.append("|---|" + "---|" * len(BOUNDS))
        for m in inst["methods"]:
            cells = [f"{p['fixed_new']}/{p['broken_new']}; {p['fixed_clean']}/{p['broken_clean']}"
                     for p in (m["points"][b] for b in BOUNDS)]
            L.append(f"| {m['label']} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("Detail per pick (share = broken / the base's correct rows; worst class = lowest per-class change):")
        L.append("")
        L.append("| method | bound | pick | config | V_check S_new / S_old | gain | clean | joint | "
                 "broken share cond / clean (%) | worst class clean | worst class cond |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for m in inst["methods"]:
            for b in BOUNDS:
                p = m["points"][b]
                if p["base_fallback"]:
                    L.append(f"| {m['label']} | {b} | base (B) | | | 0 | 0 | 0 | 0 / 0 | | |")
                    continue
                wc, wn = p["worst_class_clean"], p["worst_class_new"]
                L.append(f"| {m['label']} | {b} | {p['pick']} | {p['config']} | {p['vcheck_s_new']} / "
                         f"{p['vcheck_s_old']} | {_f(p['gain'])} | {_f(p['clean'])} | {_f(p['joint'])} | "
                         f"{_f(100 * p['broken_share_new'], 2, False)} / "
                         f"{_f(100 * p['broken_share_clean'], 2, False)} | "
                         f"{wc['class']} {_f(wc['change'], 2)} | {wn['class']} {_f(wn['change'], 2)} |")
        L.append("")
        L.append("Checks against the picks of record (T06's -0.5 bound is the kept rule's floor; the kept rule "
                 "maximizes S_joint with the base as a candidate, T06 maximizes S_new):")
        L.append("")
        L.append("| method | kept of record | T06 at -0.5 | status | kept rule reproduced | DEV_RESULTS |")
        L.append("|---|---|---|---|---|---|")
        for c in inst["checks"]:
            dr = c.get("dev_results")
            drs = "" if dr is None else (f"{_f(dr['gain'])} / {_f(dr['clean'])} equal" if dr["equal"] else "MISMATCH")
            rep = c.get("kept_rule_reproduced")
            L.append(f"| {c['method']} | {c['kept_of_record']} | {c['t06_pick_at_-0.5']} | {c['status']} | "
                     f"{'' if rep is None else ('yes' if rep else 'NO')} | {drs} |")
        L.append("")
        L.append("Cost (n_changed = entries the pick changes; times from the run record. Machines and threads "
                 "differ, so times are not comparable across rows: torch 2.14.0+cu130 is the 4090 workstation's "
                 "build, 2.14.0a0+...nv26.09 the NSCC NGC image, per the as-built notes in PREREG):")
        L.append("")
        L.append("| method | trainable DoF | n_changed at " + " / ".join(BOUNDS) + " | kept of record n_changed | "
                 "fit or generate s | check s | wall s | environment |")
        L.append("|---|---|---|---|---|---|---|---|")
        for m in inst["methods"]:
            c = m["cost"]
            nch = " / ".join(_int(m["points"][b]["n_changed"]) for b in BOUNDS)
            if m["key"] == "llm_b_final":
                ex = m["points"]["-0.5"]["exec_s"]
                L.append(f"| {m['label']} | program | {nch} | {_int(m['kept_n_changed'])} | session "
                         f"{c['session_wall_s']:.0f} (pick's program run {_f(ex, 1, False)}) | in session | "
                         f"{c['session_wall_s']:.0f} | {c['model']} effort {c['effort']}, "
                         f"{c['output_tokens']:,} output tokens, sandbox {c['sandbox_threads']} threads |")
                continue
            src = str(c.get("trainable_dof_source", ""))
            dof = "n/a" if src.startswith("not applicable") else _int(c.get("trainable_dof"))
            if c.get("trainable_dof") is not None and not src.startswith("recorded"):
                dof += " (derived)"
            if src.startswith("per configuration"):
                per = ["base" if m["points"][b]["base_fallback"] else _int(m["points"][b]["trainable_dof"])
                       for b in BOUNDS]
                dof = "per pick " + " / ".join(per) + " (derived)"
            env = (f"torch {c.get('torch')}" + (f", {c['threads']} threads" if c.get("threads") else "")
                   if c.get("torch") else "n/r")
            L.append(f"| {m['label']} | {dof} | {nch} | {_int(m['kept_n_changed'])} | "
                     f"{_f(c.get('generate_s'), 0, False)} | {_f(c.get('check_s'), 0, False)} | "
                     f"{_f(c.get('wall_s'), 0, False)} | {env} |")
    L.append("")
    L.append("## Not covered")
    L.append("")
    for k, v in result["not_covered"].items():
        L.append(f"- {k}: {v}.")
    L.append("")
    return "\n".join(L)


def build(store: Store, repo: Path) -> dict[str, Any]:
    """Read every record, apply the rule, run the checks; raise RecordError on any failed check."""
    instances = []
    failures: list[str] = []
    for spec in INSTANCES:
        classes = store.classes(spec.instance_id)
        methods = load_instance(store, spec)
        n_old = 4000
        info = store.json(f"devreadouts/learn/{spec.session}/info.json")
        if int(info["kept_rule"]["n_old"]) != n_old:
            raise RecordError(f"{spec.instance_id}: n_old {info['kept_rule']['n_old']}")
        rows = []
        checks = []
        for m in methods:
            for c in m.candidates:
                if c.clean.n != n_old:
                    raise RecordError(f"{m.key} {c.cid}: clean V_select has {c.clean.n} rows")
            pts = {b: operating_point(pick(m.candidates, n_old, b), classes) for b in BOUNDS}
            for b in BOUNDS:
                full = pick_over_checks(m.checks_on_vcheck(), n_old, b)
                pts[b]["pick_over_all_checks"] = None if full is None else f"q{full}"
                pts[b]["missing_readout_would_change_pick"] = full != pts[b]["order"]
            chk = check_method(m, spec.instance_id, n_old, classes)
            failures += [f"{spec.instance_id} {m.key}: {f}" for f in chk["failures"]]
            checks.append(chk)
            kept_n = 0 if m.kept_is_base else (m.kept_of_record.n_changed if m.kept_of_record else None)
            rows.append({"key": m.key, "label": m.label, "family": m.family, "coverage": m.coverage,
                         "n_candidates": len(m.candidates), "n_checked": m.n_checked, "points": pts,
                         "candidates": [{"order": c.order, "cid": c.cid, "config": c.config, "s_new": c.s_new,
                                         "s_old": c.s_old, "gain": c.select.delta_acc * POINTS,
                                         "clean": c.clean.delta_acc * POINTS, "n_changed": c.n_changed,
                                         "child_sha256": c.child_sha256} for c in m.candidates],
                         "kept_of_record": "base" if m.kept_is_base else (m.kept_of_record.cid if m.kept_of_record
                                                                          else None),
                         "kept_n_changed": kept_n, "cost": m.cost, "notes": m.notes})
        instances.append({"experiment": spec.experiment, "instance_id": spec.instance_id,
                          "condition": spec.condition, "session": spec.session, "n_old": n_old, "classes": classes,
                          "methods": rows, "checks": checks})
    for rel, exp in EXPECTED_SHA256.items():
        got = store.read.get(f"runs/{rel}")
        if got is None:
            failures.append(f"documented record {rel} was not read")
        elif not got.startswith(exp):
            failures.append(f"{rel}: sha256 {got} does not match the documented {exp}")
    if failures:
        raise RecordError("checks failed:\n  " + "\n  ".join(failures))
    script = Path(__file__).resolve()
    return {"schema": SCHEMA, "rule": RULE_TEXT, "bounds_points": list(BOUNDS), "kept_floor_for_checks": KEPT_FLOOR,
            "instances": instances, "not_covered": NOT_COVERED,
            "inputs_sha256": dict(sorted(store.read.items())),
            "provenance": {"script": str(script.relative_to(repo)) if script.is_relative_to(repo) else str(script),
                           "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                           "git_head": _git_head(repo), "python": sys.version.split()[0],
                           "label": "development evidence; numbers from records, no model run"}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    runs_default = os.environ.get("WEIGHTCODE_RUNS")
    ap.add_argument("--runs-root", type=Path, default=Path(runs_default) if runs_default else None,
                    required=not runs_default, help="the store's runs/ (read only); defaults to WEIGHTCODE_RUNS")
    ap.add_argument("--instances-root", type=Path, default=None,
                    help="private/instances (class names only); default: <runs-root>/../private/instances")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <runs-root>/t06")
    args = ap.parse_args(argv)
    runs = args.runs_root
    store = Store(runs, args.instances_root or runs.parent / "private" / "instances")
    out_dir = args.out_dir or runs / "t06"
    repo = Path(__file__).resolve().parents[2]
    result = build(store, repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tradeoff_points.json").write_text(json.dumps(result, indent=1, sort_keys=False) + "\n")
    (out_dir / "tradeoff_points.md").write_text(render_markdown(result))
    for name in ("tradeoff_points.json", "tradeoff_points.md"):
        print(name, hashlib.sha256((out_dir / name).read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
