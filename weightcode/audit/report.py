"""Audit report: rule catalogue, flags, candidate records, verdict and serialisation (schema weightcode.audit.v0)."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = "weightcode.audit.v0"
RULES_VERSION = "weightcode.audit.rules.v0+a2.20261104"  # Audit amendment A2 (2026-11-04): counting rules changed

CLEAN = "CLEAN"
SOFT_FLAGS = "SOFT_FLAGS"
VIOLATION = "VIOLATION"
SOFT = "SOFT"

# Rule ids. A "V-" rule is a static VIOLATION, an "S-" rule a SOFT flag that the frozen LLM judge resolves.
V_AUTOGRAD = "V-GRAD-AUTOGRAD"
V_BACKPROP = "V-GRAD-BACKPROP"
V_FINITE_DIFF = "V-GRAD-FINITE-DIFF"
V_SOLVER = "V-SOLVER-LABEL-TARGET"
V_FIT = "V-FIT-CALL"
V_OPTIMIZER = "V-OPTIMIZER-API"
V_ITERATIVE = "V-ITERATIVE-UPDATE"
V_SEARCH = "V-SEARCH-COUNT"
S_IMPORT = "S-IMPORT"
S_GRAD_NAME = "S-GRAD-NAME"
S_GRAD_UNUSED = "S-GRAD-NO-UPDATE"
S_SOLVER_INDIRECT = "S-SOLVER-INDIRECT"
S_SOLVER_SOFT = "S-SOLVER-SOFT-TAINT"
S_RANDOM_SMALL = "S-RANDOM-SEARCH-SMALL"
S_DYNAMIC = "S-DYNAMIC-CODE"
S_UNPARSED = "S-UNPARSED"
S_LIMIT = "S-ANALYSIS-LIMIT"

RULES: dict[str, tuple[str, str]] = {
    V_AUTOGRAD: (VIOLATION, "autograd use: .backward(), requires_grad, torch.autograd, jax.grad, GradientTape and similar"),
    V_BACKPROP: (VIOLATION, "hand-written gradient: an error signal (model output minus a label encoding), possibly "
                            "contracted with activations, used to update parameters or written into a patch"),
    V_FINITE_DIFF: (VIOLATION, "finite-difference quotient of label-dependent scores used to update parameters or "
                               "written into a patch"),
    V_SOLVER: (VIOLATION, "lstsq, solve, tensorsolve or polyfit with a label-derived target, or an inv/pinv result "
                          "multiplied with a label-derived operand"),
    V_FIT: (VIOLATION, "a .fit( call (also partial_fit, fit_transform, fit_predict)"),
    V_OPTIMIZER: (VIOLATION, "call to an optimiser or model-fitting API (torch.optim, optax, scipy.optimize, "
                             "sklearn model selection)"),
    V_ITERATIVE: (VIOLATION, "a loop updates a parameter or patch quantity whose new value or gating condition depends "
                             "on a label-dependent evaluation of that same quantity"),
    V_SEARCH: (VIOLATION, "more than the limit of distinct candidate parameter settings scored against label-dependent "
                          "values in the session, counting distinct patches sent to the predict tool (unknown loop "
                          "bounds count as above the limit)"),
    S_IMPORT: (SOFT, "attempted import of torch, jax, autograd, tensorflow, sklearn, scipy or a similar library"),
    S_GRAD_NAME: (SOFT, "gradient-like name bound to a label-dependent value, or .grad/.gradient( access, without a "
                        "detected gradient computation"),
    S_GRAD_UNUSED: (SOFT, "hand-written gradient or finite-difference quotient computed, but no parameter update or "
                          "patch written from it was observed"),
    S_SOLVER_INDIRECT: (SOFT, "a solver or inverse whose inputs are label-free at the call, whose result later meets "
                              "label-dependent values"),
    S_SOLVER_SOFT: (SOFT, "a solver whose design matrix is label-dependent while its target is not, or whose target "
                          "depends on labels only through a candidate setting built from label-dependent statistics, "
                          "or a solver target or inverse operand built from the x_fail split (membership in x_fail "
                          "is label-derived: a possible probe of failure membership)"),
    S_RANDOM_SMALL: (SOFT, "random candidate settings, at most the limit, scored against label-dependent values with "
                           "the best one kept"),
    S_DYNAMIC: (SOFT, "code the audit cannot see: exec/eval/compile of non-constant code, subprocesses, editor-written "
                      "Python files or modules, dynamic imports or dynamic attribute lookup on modules"),
    S_UNPARSED: (SOFT, "a cell that may have run could not be parsed by the audit; regex fallback only"),
    S_LIMIT: (SOFT, "the analysis budget of a cell was exhausted; the rest of that cell was not analysed"),
}


def severity_of(rule: str) -> str:
    """Severity (VIOLATION or SOFT) of a rule id."""
    if rule not in RULES:
        raise KeyError(f"unknown audit rule id {rule!r}")
    return RULES[rule][0]


@dataclass(frozen=True)
class AuditFlag:
    """One finding: rule id, location (cell index and 1-based line within that cell's code), excerpt and reason."""

    flag_id: str
    rule: str
    severity: str
    cell_index: int
    line: int
    excerpt: str
    reason: str


@dataclass(frozen=True)
class CandidateSetting:
    """One distinct candidate parameter setting (or loop family of settings) scored against label-dependent values.

    ``count`` is the number of settings the record stands for (None when a loop bound is unknown).
    ``counted`` is False for ablation sweeps, which the contract allows and the count excludes. ``kind`` is
    ``explicit``, ``loop``, ``ablation`` or ``predict``; a ``predict`` record is a distinct patched parameter setting
    whose evidence logits the predict tool returned, with ``cell_index`` -1 and ``line`` the 1-based line of its first
    call in ``logs/tools.jsonl``.

    Audit amendment A2 (2026-11-04): ``config_id`` names the configuration the setting belongs to (its modification
    sites without the calling statements' locations) and ``config_count`` is the number of configurations it stands
    for: loops that vary only which evidence rows the setting is built from (folds, views, classes) count once, as
    do single-element index sweeps. Several records can share one ``config_id``; ``AuditReport.config_count`` counts
    each configuration once. Reported only; the verdict uses ``count``.
    """

    setting_id: str
    cell_index: int
    line: int
    excerpt: str
    count: int | None
    counted: bool
    kind: str
    random: bool
    best_kept: bool
    loops: tuple[dict[str, object], ...] = ()
    config_id: str = ""
    config_count: int | None = None


@dataclass(frozen=True)
class AuditNote:
    """Information that is not a flag: excluded ablation sweeps, cells that never ran, parse fallbacks."""

    code: str
    cell_index: int
    line: int
    text: str


@dataclass(frozen=True)
class AuditReport:
    """Result of auditing one session's executed cells against contract section 5.5.

    ``verdict`` is VIOLATION if any VIOLATION flag, SOFT_FLAGS if any SOFT flag, else CLEAN. ``candidate_count`` is the
    number of distinct candidate parameter settings scored against label-dependent values (ablation sweeps excluded),
    a lower bound when ``candidate_count_is_lower_bound``; it includes the distinct patches sent to the predict tool.
    ``label_access_cells`` are the cells that read a label array from the evidence file. ``n_analysed`` counts the
    cells that ran (refused and sandbox-error cells did not). ``auditor_sha256`` names the audit implementation and
    judge prompt, ``cells_sha256`` the cell log audited and ``tools_sha256`` the tool log (None when there was none).

    Audit amendment A2 (2026-11-04): ``candidate_count`` is the literal count (L): every distinct setting compared with
    labels, each leave-one-out fold and each view-specific setting counted on its own. ``config_count`` is the
    configuration count (C): one rule with fixed free parameters counts once however many folds or views it was
    evaluated on, each distinct free-parameter value (a strength, a threshold, a program) counts once, and a setting
    rebuilt with the same arguments at another statement is not counted again. (C) is reported, not gated: the verdict
    and V-SEARCH-COUNT use (L).
    """

    verdict: str
    flags: tuple[AuditFlag, ...]
    candidate_count: int
    candidate_count_is_lower_bound: bool
    candidate_limit: int
    candidate_settings: tuple[CandidateSetting, ...]
    notes: tuple[AuditNote, ...]
    label_access_cells: tuple[int, ...]
    n_cells: int
    n_analysed: int
    unparsed_cells: tuple[int, ...]
    not_executed_cells: tuple[int, ...]
    evidence_label_keys: tuple[str, ...]
    rule_counts: Mapping[str, int]
    auditor_sha256: str
    cells_sha256: str | None
    tools_sha256: str | None = None
    config_count: int = 0
    config_count_is_lower_bound: bool = False
    schema: str = SCHEMA
    rules_version: str = RULES_VERSION
    rule_catalogue: Mapping[str, tuple[str, str]] = field(default_factory=lambda: dict(RULES))

    def soft_flags(self) -> tuple[AuditFlag, ...]:
        return tuple(flag for flag in self.flags if flag.severity == SOFT)

    def violation_flags(self) -> tuple[AuditFlag, ...]:
        return tuple(flag for flag in self.flags if flag.severity == VIOLATION)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["rule_counts"] = dict(self.rule_counts)
        data["rule_catalogue"] = {rule: {"severity": sev, "text": text} for rule, (sev, text) in self.rule_catalogue.items()}
        return data


def verdict_from_flags(flags: Sequence[AuditFlag]) -> str:
    """VIOLATION if any VIOLATION flag, SOFT_FLAGS if any SOFT flag, else CLEAN."""
    if any(flag.severity == VIOLATION for flag in flags):
        return VIOLATION
    if any(flag.severity == SOFT for flag in flags):
        return SOFT_FLAGS
    return CLEAN


def auditor_sha256() -> str:
    """Hash of the audit package sources and the judge prompt, so a report names the rule implementation it used."""
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted([*package.glob("*.py"), package / "judge_prompt.md"])
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"audit package file missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_json_atomic(path: str | Path, data: Mapping[str, object]) -> None:
    """Write JSON to a temporary file in the same directory and rename it over ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temporary.replace(target)
