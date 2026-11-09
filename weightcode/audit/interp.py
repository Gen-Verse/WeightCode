"""Abstract interpreter over the ordered cells of one session: the data-flow half of the training-free audit.

The interpreter executes every cell's AST abstractly in one persistent namespace, as the sandbox kernel does, with
``AbsVal`` values (see ``weightcode.audit.absval``). It tracks label taint, parameter provenance, candidate parameter
settings and their forward evaluations, loop-carried state and control dependence. Loops run to a fixpoint (at most
``MAX_PASSES`` passes), user functions are analysed at each call site with the caller's argument values, and branches
are joined. Rules that need data flow are checked here; purely syntactic rules live in ``weightcode.audit.syntactic``.
The method and its blind spots are documented in the ``weightcode.audit`` package docstring.
"""

from __future__ import annotations

import ast
import builtins
import fnmatch
import hashlib
import itertools
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace

from weightcode.audit.absval import (
    CONST,
    DICT,
    EVID,
    EVID_C,
    FAILSET,
    FILE,
    FUNC,
    GLOBALS,
    HARD,
    INSTANCE,
    LABEL,
    LABEL_BUILT,
    LSIZE,
    METHOD,
    MODEL_C,
    MODULE,
    NOCONST,
    OTHER,
    PARAM,
    PATH,
    PDICT,
    RANDOM,
    RNG,
    SAMPLE,
    SEQ,
    STAT,
    WEIGHT,
    AbsVal,
    alternatives,
    candref_tags,
    combine_keyvar,
    is_tainted,
    join,
    join_all,
    signature,
    strip_candref,
)
from weightcode.audit.report import (
    S_DYNAMIC,
    S_GRAD_NAME,
    S_GRAD_UNUSED,
    S_LIMIT,
    S_RANDOM_SMALL,
    S_SOLVER_INDIRECT,
    S_SOLVER_SOFT,
    S_UNPARSED,
    V_BACKPROP,
    V_FINITE_DIFF,
    V_ITERATIVE,
    V_SEARCH,
    V_SOLVER,
)
from weightcode.audit.syntactic import (
    line_excerpt,
    regex_bound_names,
    regex_mentioned_names,
    regex_scan,
    syntactic_flags,
    update_aliases,
)

MAX_PASSES = 3
LOOP_MARK_DEPTH = 5
# Tags of values that carry derivative information: error signals, hand-written gradients, finite-difference quotients
# and differences of label-dependent scores of two parameter settings. They raise a gradient rule when they reach a
# parameter value or a patch; an index, mask or comparison computed from them does not carry them on.
GRAD_PREFIXES = ("err:", "grad:", "fd:", "sd:")
SIDE_EFFECT_NODES = (ast.Call, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)
MAX_CALL_DEPTH = 8
MAX_ALTERNATIVES = 32
MAX_EVAL_TAGS = 64
# Abstract interpretation steps per cell before S-ANALYSIS-LIMIT. Real sessions (dev wave 1, 2026-11-04): the largest
# cell took 418,672 steps (11.5 s, session sb77dd2c07fb68078 cell 48, nested helper calls); most cells take under 30,000.
# Five times that largest cell bounds one cell at about a minute.
STEP_LIMIT_PER_CELL = 2_000_000
SMALL_STEP = 0.1
UNBOUNDED_LABEL = "unbounded"
PREDICT_CELL = -1  # cell index of candidates scored through the predict tool (the tool log, not a cell)

GRAD_NAME = re.compile(
    r"^(?:grads?|gradients?|grad_\w+|\w+_grads?|\w+_gradients?|dl_?d\w+|dloss\w*|d_loss\w*|dlogits?\w*|d_logits?\w*|"
    r"jac|jacobians?|jac_\w+|\w+_jac|backprop\w*|backward\w*)$",
    re.IGNORECASE,
)
# Index-valued numpy functions and array methods: their results select elements, they are not model outputs in value.
INDEX_VALUED = frozenset({
    "argmax", "argmin", "nanargmax", "nanargmin", "argsort", "argpartition", "nonzero", "flatnonzero", "argwhere",
    "searchsorted", "digitize", "lexsort",
})
SPEC_PREFIX = "spec:"  # qual of an importlib spec of a /work/trial module file
KNOWN_MODULES = frozenset({
    "numpy", "json", "math", "os", "sys", "re", "itertools", "functools", "collections", "pathlib", "copy", "time",
    "random", "statistics", "typing", "dataclasses", "pickle", "hashlib", "io", "string", "textwrap", "pprint",
    "operator", "heapq", "bisect", "warnings", "contextlib", "enum", "abc", "numbers", "decimal", "fractions",
    "struct", "base64", "zlib", "gzip", "csv", "glob", "shutil", "tempfile", "datetime", "importlib", "traceback",
    "inspect", "types", "weakref", "array", "queue", "gc", "forward", "zipfile", "uuid", "secrets", "cmath",
})
DYNAMIC_EXEC = frozenset({
    "os.system", "os.popen", "os.execv", "os.execve", "os.execl", "os.execlp", "os.execvp", "os.spawnl", "os.spawnv",
    "os.posix_spawn", "os.fork", "runpy.run_path", "runpy.run_module", "pty.spawn",
})
DYNAMIC_ROOTS = frozenset({"subprocess", "multiprocessing", "ctypes", "concurrent"})
SHAPE_ATTRS = frozenset({"shape", "ndim", "size", "dtype", "nbytes", "itemsize", "strides", "flags"})
MODULE_CONSTS = frozenset({
    "numpy.pi", "numpy.e", "numpy.inf", "numpy.nan", "numpy.newaxis", "numpy.euler_gamma", "math.pi", "math.e",
    "math.inf", "math.nan", "math.tau", "numpy.float32", "numpy.float64", "numpy.int64", "numpy.int32",
    "numpy.uint8", "numpy.bool_", "numpy.float16", "numpy.intp",
})
# Audit amendment A2 (2026-11-04): numpy scalar types are constants when passed as a dtype, but calling one is a cast
# (``np.float32(P[k] + d[k])``) that keeps its argument's value and modification history. Before, the call went to an
# unknown constant and returned a value without history, so every program patched in that way looked like the base.
NUMPY_DTYPE_CTORS = frozenset({
    "numpy.float32", "numpy.float64", "numpy.int64", "numpy.int32", "numpy.uint8", "numpy.bool_", "numpy.float16",
    "numpy.intp",
})
# Audit amendment A2 (2026-11-04): the trial file that holds the evidence. Evidence-v1 named it evidence.npz; route v1
# (LEARN sessions) names it experience.npz. Before A2 only "evidence" was recognised, so /work/trial/experience.npz was
# read as a model file and no label of any route-v1 session reached the analysis.
EVIDENCE_FILE_NAMES = ("evidence", "experience")
# Evidence keys a copy of the container holds explicitly besides the label keys: evidence-v1 and route v1 (x_new,
# x_old, x_prev and their predictions and ids). Any other y* key is a label, x* and pred* keys are evidence data.
EVIDENCE_DATA_KEYS = ("x_fail", "x_rand", "pred_fail", "pred_rand", "split_ids", "x_new", "x_old", "x_prev",
                      "pred_new", "pred_old", "pred_prev", "split_ids_new", "split_ids_old", "split_ids_prev")
# Audit amendment A2 (2026-11-04): reductions over the sample axis that turn label-selected per-sample model features
# into a feature-space readout (a class-mean prototype, an imprinted head row). See Interp._readout_tag.
READOUT_REDUCE = frozenset({"mean", "sum", "average", "nanmean", "nansum", "median", "nanmedian"})
# Comparison operators that apply a threshold to a model output (a decision rule): A2 (2026-11-04) decision variance.
ORDER_COMPARE = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
DECISION_ARITH = (ast.Add, ast.Sub, ast.Mult)
NP_ORDER_COMPARE = frozenset({"greater", "greater_equal", "less", "less_equal"})
NP_DECISION_ELEMENTWISE = frozenset({"maximum", "minimum", "fmax", "fmin", "clip"})


def is_evidence_path(path: str) -> bool:
    """True when a file path names the trial evidence container (``evidence.npz``, ``experience.npz``)."""
    lowered = path.lower()
    return any(name in lowered for name in EVIDENCE_FILE_NAMES)

NP_REDUCE = frozenset({
    "sum", "mean", "average", "std", "var", "max", "min", "amax", "amin", "argmax", "argmin", "median", "percentile",
    "quantile", "nanmean", "nansum", "nanmax", "nanmin", "nanstd", "nanvar", "ptp", "prod", "count_nonzero", "any",
    "all", "nanargmax", "nanargmin", "nanmedian", "nanpercentile", "nanquantile", "cumsum", "cumprod", "trace",
})
NP_VIEW = frozenset({
    "reshape", "ravel", "transpose", "squeeze", "expand_dims", "atleast_1d", "atleast_2d", "atleast_3d",
    "broadcast_to", "swapaxes", "moveaxis", "asarray", "asanyarray", "ascontiguousarray", "asfortranarray",
})
NP_COPY = frozenset({"array", "copy", "require"})
NP_UNARY = frozenset({
    "abs", "absolute", "exp", "expm1", "exp2", "log", "log1p", "log2", "log10", "sqrt", "square", "sign", "tanh",
    "sinh", "cosh", "sin", "cos", "tan", "arctan", "arcsin", "arccos", "floor", "ceil", "round", "around", "rint",
    "fix", "negative", "positive", "reciprocal", "nan_to_num", "real", "imag", "conj", "conjugate", "flip", "fliplr",
    "flipud", "rot90", "roll", "sort", "diff", "ediff1d", "tril", "triu", "diag", "diagflat", "diagonal", "cbrt",
    "isfinite", "isnan", "isinf", "signbit", "logical_not", "invert", "tile", "repeat", "pad", "resize",
})
# Index functions whose output length depends on the input values (a label-dependent count when they see labels).
VARIABLE_SIZE = frozenset({"nonzero", "flatnonzero", "argwhere", "unique", "extract", "compress"})
NP_INDEX = frozenset({
    "argsort", "nonzero", "flatnonzero", "argwhere", "unique", "searchsorted", "digitize", "bincount", "lexsort",
    "argpartition", "partition", "isin",
})
NP_ARITH = {
    "add": ast.Add, "subtract": ast.Sub, "multiply": ast.Mult, "divide": ast.Div, "true_divide": ast.Div,
    "floor_divide": ast.FloorDiv, "power": ast.Pow, "float_power": ast.Pow, "mod": ast.Mod, "remainder": ast.Mod,
    "fmod": ast.Mod,
}
NP_ELEMENTWISE = frozenset({
    "maximum", "minimum", "fmax", "fmin", "clip", "where", "hypot", "arctan2", "copysign", "heaviside", "logaddexp",
    "logaddexp2", "select", "choose", "interp", "nextafter", "ldexp",
})
NP_COMPARE = frozenset({
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal", "isclose", "allclose", "array_equal",
    "logical_and", "logical_or", "logical_xor", "array_equiv",
})
NP_MATMUL = frozenset({"dot", "matmul", "inner", "vdot", "tensordot", "outer", "kron", "einsum", "cross"})
NP_JOIN = frozenset({
    "concatenate", "stack", "vstack", "hstack", "column_stack", "row_stack", "dstack", "block", "append", "insert",
    "delete", "take", "take_along_axis", "compress", "extract", "split", "array_split", "hsplit", "vsplit",
})
NP_STATS = frozenset({"cov", "corrcoef", "histogram", "histogram2d", "histogramdd", "correlate", "convolve", "polyval"})
NP_MUTATE = frozenset({"copyto", "put", "place", "putmask", "fill_diagonal", "put_along_axis"})
NP_SHAPE_NEW = frozenset({"zeros", "ones", "empty"})
NP_LIKE = frozenset({"zeros_like", "ones_like", "empty_like"})
LINALG_SOLVERS = frozenset({"lstsq", "solve", "tensorsolve", "solve_triangular", "cho_solve", "lu_solve", "solveh_banded",
                            "solve_banded"})
LINALG_INVERSES = frozenset({"pinv", "inv", "tensorinv", "pinvh"})
LINALG_TUPLES = {"svd": 3, "eig": 2, "eigh": 2, "qr": 2, "slogdet": 2, "lu": 3}
RANDOM_CTORS = frozenset({"default_rng", "RandomState", "Generator", "PCG64", "SeedSequence", "MT19937", "Philox",
                          "SFC64", "BitGenerator", "Random", "SystemRandom"})


class _BudgetExceeded(Exception):
    """The per-cell analysis step budget was used up."""


@dataclass
class Scope:
    """A namespace: the session globals, a function call's locals, a class body or a comprehension."""

    parent: "Scope | None"
    is_module: bool
    vars: dict[str, AbsVal] = field(default_factory=dict)
    global_names: set[str] = field(default_factory=set)
    nonlocal_names: set[str] = field(default_factory=set)


@dataclass(eq=False)
class FuncInfo:
    node: ast.AST
    name: str
    closure: Scope | None
    cell: int
    lines: list[str]
    defaults: list[AbsVal]
    kw_defaults: dict[str, AbsVal]
    owner: "ClassInfo | None" = None


@dataclass(eq=False)
class ClassInfo:
    name: str
    methods: dict[str, FuncInfo]
    attrs: dict[str, AbsVal]
    bases: list["ClassInfo"]

    def lookup(self, name: str) -> "FuncInfo | AbsVal | None":
        if name in self.methods:
            return self.methods[name]
        if name in self.attrs:
            return self.attrs[name]
        for base in self.bases:
            found = base.lookup(name)
            if found is not None:
                return found
        return None


@dataclass(eq=False)
class PartialInfo:
    func: AbsVal
    args: list[AbsVal]
    kwargs: dict[str, AbsVal]


@dataclass
class Frame:
    scope: Scope
    func: FuncInfo | None
    cell: int
    lines: list[str]
    returns: list[AbsVal] = field(default_factory=list)
    yields: list[AbsVal] = field(default_factory=list)
    stmt: tuple[int, int] = (0, 0)


@dataclass
class Event:
    """An assignment inside a loop: which names it updated, the assigned value's tags and the gating tags."""

    names: frozenset[str]
    value_tags: frozenset[str]
    gate_tags: frozenset[str]
    cell: int
    line: int
    excerpt: str


@dataclass
class LoopCtx:
    """One active loop. ``rebound`` and ``kept`` decide which values hold one iteration's result after it.

    ``rebound``: names the loop's own frame binds by plain assignment. ``kept``: names mutated in place (anywhere,
    helpers included), augmented, or bound to a value computed from their own previous value; these may hold the
    results of every iteration (a list appended to, an array stored into by index, a running total).
    """

    lid: str
    trip: int | None
    kind: str
    cell: int
    line: int
    events: dict[tuple[int, int, frozenset[str]], Event] = field(default_factory=dict)
    exit_gates: set[str] = field(default_factory=set)
    var_kinds: dict[str, str] = field(default_factory=dict)
    frame: Frame | None = None
    rebound: set[tuple[int, str]] = field(default_factory=set)  # (id of the binding scope, name)
    kept: set[tuple[int, str]] = field(default_factory=set)


@dataclass
class PartInfo:
    """One modification site (or readout site) of a setting history.

    ``config`` names the site for the configuration count (A2, 2026-11-04): the same site with the call context reduced
    to callee and argument tokens, without the calling statements' locations, so one rule called with the same
    arguments in two cells is one configuration while ``make(0.5)`` and ``make(0.8)`` stay two.
    """

    cell: int
    line: int
    excerpt: str
    best_kept: bool = False
    config: str = ""


@dataclass
class SettingInfo:
    key: str
    sigs: set[str]
    cell: int
    line: int
    excerpt: str


@dataclass
class SiteInfo:
    kind: str
    cell: int
    line: int
    excerpt: str
    tainted_input: bool = False
    used: bool = False
    flagged: str | None = None


@dataclass(frozen=True)
class RawResultFlag:
    rule: str
    cell: int
    line: int
    excerpt: str
    reason: str


@dataclass(frozen=True)
class RawSetting:
    setting_id: str
    cell: int
    line: int
    excerpt: str
    count: int | None
    counted: bool
    kind: str
    random: bool
    best_kept: bool
    loops: tuple[dict[str, object], ...]
    config_id: str = ""
    config_count: int | None = None


@dataclass(frozen=True)
class InterpResult:
    flags: list[RawResultFlag]
    notes: list[tuple[str, int, int, str]]
    settings: list[RawSetting]
    candidate_count: int
    lower_bound: bool
    label_cells: list[int]
    unparsed: list[int]
    config_count: int = 0
    config_lower_bound: bool = False


# Kinds of loop variance in an evaluation signature: "i" which single element an index hits (a knockout), "m" which
# elements, "f" the value, through which evidence rows were selected (a fold, a view or a class), "d" a free
# parameter of the decision rule applied to the evaluation (a threshold or an offset), "v" the value otherwise (a
# strength, a program). The literal count (L) multiplies by every kind but "i"; the configuration count (C) by "m",
# "d" and "v". "f", "d" and the count (C) are Audit amendment A2 (2026-11-04).
_STRENGTH = {None: 0, "i": 1, "m": 2, "f": 3, "d": 4, "v": 5}
COUNTED_KINDS = frozenset({"m", "f", "d", "v"})
CONFIG_KINDS = frozenset({"m", "d", "v"})


# Tags that make a value vary with a loop: in value (lv), by a single or multi-element index (li, lim), through
# the evidence rows it was selected from (lf: a fold or view; lc: the rows of one class) or by a key read of one
# setting (lk). lf, lc and lk are Audit amendment A2 (2026-11-04).
LOOP_VARIANCE_PREFIXES = ("lv:", "li:", "lim:", "lf:", "lk:", "lc:")


def _readout_tags(tags: frozenset[str]) -> frozenset[str]:
    """The feature-space readout tags (``rd:<site>``, A2 2026-11-04) among ``tags``."""
    return frozenset(t for t in tags if t.startswith("rd:"))


def _is_data_arg(value: AbsVal) -> bool:
    """An argument that is evidence-derived data (inputs, features, outputs), not a setting or a free parameter."""
    return (EVID in value.tags and not alternatives(value)
            and value.role not in (WEIGHT, PDICT, MODEL_C, FUNC, MODULE, METHOD, CONST))


def _known_truth(value: AbsVal) -> bool | None:
    """The truth value of an untainted, exactly known constant (None, a bool, a number, a string, a tuple), else None.

    A2 (2026-11-04): used to run only the taken branch of an ``if`` and to short-circuit ``and``/``or``.
    """
    if value.role != CONST or value.tags or not value.has_const():
        return None
    const = value.const
    if const is None or isinstance(const, (bool, int, float, str, bytes, tuple)):
        return bool(const)
    return None


def _drop_decision(tags: frozenset[str]) -> frozenset[str]:
    """Evaluation tags without their decision-rule loops (``=d``, A2 2026-11-04).

    A thresholded mask used as an index selects the rows a fixed setting is scored on (accuracy per confidence bin):
    the threshold then picks a view, not a decision rule, and does not multiply the count.
    """
    if not any(t.startswith("eval:") and "=d" in t for t in tags):
        return tags
    out: set[str] = set()
    for tag in tags:
        if tag.startswith("eval:"):
            key, sig = _split_eval(tag)
            if sig != UNBOUNDED_LABEL:
                tag = f"eval:{key}#{';'.join(p for p in sig.split(';') if p and not p.endswith('=d'))}"
        out.add(tag)
    return frozenset(out)


def _strongest(a: str | None, b: str | None) -> str | None:
    return a if _STRENGTH[a] >= _STRENGTH[b] else b


def _merge_keys(a: str, b: str) -> str:
    parts = {p for p in a.split("+") if p} | {p for p in b.split("+") if p}
    return "+".join(sorted(parts))


def _merge_sigs(a: str, b: str) -> str:
    """Merge two loop signatures; an overflowed (``UNBOUNDED_LABEL``) signature absorbs the other, conservatively."""
    if UNBOUNDED_LABEL in (a, b):
        return UNBOUNDED_LABEL
    kinds: dict[str, str] = {}
    rnd = False
    for part in [*a.split(";"), *b.split(";")]:
        if not part:
            continue
        if part == "rng":
            rnd = True
            continue
        loop, kind = part.split("=")
        kinds[loop] = _strongest(kinds.get(loop), kind) or kind
    out = [f"{loop}={kind}" for loop, kind in sorted(kinds.items())]
    if rnd:
        out.append("rng")
    return ";".join(out)


def _split_eval(tag: str) -> tuple[str, str]:
    body = tag[len("eval:"):]
    key, _, sig = body.partition("#")
    return key, sig


def _sig_loops(sig: str) -> set[str]:
    return {part.split("=", 1)[0] for part in sig.split(";") if "=" in part}


def _loop_evals(tags: frozenset[str], lid: str) -> bool:
    """True when an evaluation tag among ``tags`` counts loop ``lid`` in its multiplicity."""
    return any(t.startswith("eval:") and lid in _sig_loops(_split_eval(t)[1]) for t in tags)


def _without_loop(tag: str, lid: str) -> str:
    """``tag`` with loop ``lid`` removed from the signature of an evaluation tag (other tags unchanged)."""
    if not tag.startswith("eval:"):
        return tag
    key, sig = _split_eval(tag)
    if sig == UNBOUNDED_LABEL or lid not in _sig_loops(sig):
        return tag
    kept = [part for part in sig.split(";") if part and part.split("=", 1)[0] != lid]
    return f"eval:{key}#{';'.join(kept)}"


def _evaluated_settings(tags: frozenset[str]) -> frozenset[str]:
    """Candidate settings (and loop-carried quantities) whose forward evaluations a value is computed from."""
    return frozenset(_split_eval(t)[0] if t.startswith("eval:") else t for t in tags if t.startswith(("eval:", "evalc:")))


def _parameter_settings(tags: frozenset[str]) -> frozenset[str]:
    """Evaluated settings other than feature-space readouts (A2, 2026-11-04).

    A readout's evaluation (features minus a prototype, a leave-one-out mean ``(mu * n - h_i) / (n - 1)``) is a
    feature, not a score: a difference of two such values is a distance or a prototype update, not a finite
    difference of scores, so readout-only settings never make a score difference.
    """
    return frozenset(s for s in _evaluated_settings(tags)
                     if not all(part.startswith("r") for part in s.split("+")))


def _no_grad(tags: frozenset[str]) -> frozenset[str]:
    """Tags of a selection made by, or a text formatted from, a value: derivative information does not pass."""
    if not any(t.startswith(GRAD_PREFIXES) for t in tags):
        return tags
    return frozenset(t for t in tags if not t.startswith(GRAD_PREFIXES))


def _grad_tags(tags: frozenset[str]) -> frozenset[str]:
    return frozenset(t for t in tags if t.startswith(GRAD_PREFIXES))


def _index_tags(tags: frozenset[str]) -> frozenset[str]:
    """Tags of an index or boolean mask computed from ``tags``: gradient values do not survive, ``hard`` is set."""
    if any(t.startswith(GRAD_PREFIXES) for t in tags):
        tags = frozenset(t for t in tags if not t.startswith(GRAD_PREFIXES))
    return tags | {HARD} if tags else tags


def _mask_tags(tags: frozenset[str]) -> frozenset[str]:
    """Tags of a comparison result: a boolean mask carries neither gradient values nor hard-prediction status."""
    if HARD in tags or any(t.startswith(GRAD_PREFIXES) for t in tags):
        return frozenset(t for t in tags if t != HARD and not t.startswith(GRAD_PREFIXES))
    return tags


def _with_lsize(tags: frozenset[str]) -> frozenset[str]:
    """Tags of a selection or an index set whose length depends on labels."""
    return tags | {LSIZE} if LABEL in tags else tags


def _size_value(value: AbsVal, const: object = NOCONST) -> AbsVal:
    """``len``, ``shape`` or ``size`` of ``value``: label-dependent when the value's length is (a class count).

    The size of anything holding ``x_fail`` keeps the ``failset`` tag, so an indicator target built from it
    (``np.r_[np.ones(len(xf)), np.zeros(len(xr))]``) is recognised at a solver.
    """
    if LSIZE not in value.tags:
        if FAILSET in value.tags:
            return AbsVal(role=CONST, const=const, tags=frozenset({FAILSET}))
        return AbsVal(role=CONST, const=const)
    tags = frozenset(t for t in strip_candref(value.tags) if not t.startswith(GRAD_PREFIXES))
    return AbsVal(role=STAT, tags=(tags - {HARD, LSIZE}) | {LABEL})


def _is_zero(value: AbsVal) -> bool:
    """A known numeric constant equal to zero (not a bool)."""
    return value.has_const() and isinstance(value.const, (int, float)) and not isinstance(value.const, bool) \
        and value.const == 0


def _union_key(alts: Iterable[str]) -> str:
    """One setting name for alternative histories that a single evaluation may have: the union of their sites."""
    parts: set[str] = set()
    for alt in alts:
        parts.update(p for p in alt.split("+") if p)
    return "+".join(sorted(parts))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fold(op: ast.operator, a: object, b: object) -> object:
    """Fold arithmetic on small numeric or string constants; NOCONST when the result is not a plain constant."""
    if _is_number(a) and _is_number(b):
        if isinstance(op, ast.Add):
            return a + b  # type: ignore[operator]
        if isinstance(op, ast.Sub):
            return a - b  # type: ignore[operator]
        if isinstance(op, ast.Mult):
            return a * b  # type: ignore[operator]
        if isinstance(op, ast.Div) and b != 0:
            return a / b  # type: ignore[operator]
        if isinstance(op, ast.FloorDiv) and b != 0:
            return a // b  # type: ignore[operator]
        if isinstance(op, ast.Mod) and b != 0:
            return a % b  # type: ignore[operator]
        if isinstance(op, ast.Pow) and abs(b) <= 64 and abs(a) <= 1e6:  # type: ignore[arg-type]
            return a ** b  # type: ignore[operator]
        return NOCONST
    if isinstance(a, str) and isinstance(b, str) and isinstance(op, ast.Add):
        return a + b
    return NOCONST


def _syntax_error_recorded(error: str | None) -> bool:
    """True when the sandbox recorded a compile-time error for the cell (the kernel then runs none of it)."""
    if not error or "Traceback (most recent call last)" in error:  # a runtime traceback means the cell started
        return False
    return any(re.search(rf"^{name}\b", error, re.MULTILINE) for name in ("SyntaxError", "IndentationError", "TabError"))


def _const_int(value: AbsVal | None) -> int | None:
    if value is not None and value.has_const() and _is_number(value.const) and float(value.const).is_integer():
        return int(value.const)
    return None


def _const_number(value: AbsVal | None) -> float | None:
    if value is not None and value.has_const() and _is_number(value.const):
        return float(value.const)
    return None


def _shape_length(value: AbsVal | None) -> int | None:
    if value is None or not value.has_const():
        return None
    if _is_number(value.const):
        return _const_int(value)
    if isinstance(value.const, tuple) and value.const and _is_number(value.const[0]):
        first = value.const[0]
        return int(first) if float(first).is_integer() else None
    return None


class Interp:
    """Abstract interpreter for one session. Call ``run_cell`` for each executed cell in order, then ``finalize``."""

    def __init__(self, label_keys: Sequence[str], candidate_limit: int) -> None:
        self.label_keys = tuple(label_keys)
        self.limit = int(candidate_limit)
        self.module = Scope(parent=None, is_module=True)
        self.frame = Frame(scope=self.module, func=None, cell=-1, lines=[])
        self.frames: list[Frame] = []
        self.loops: list[LoopCtx] = []
        self.active_loop_ids: set[str] = set()
        self.loop_trips: dict[str, int | None] = {}
        self.loop_meta: dict[str, tuple[int, int, str, str]] = {}
        self.gates: list[frozenset[str]] = []
        self.flags: dict[tuple[str, int, int], RawResultFlag] = {}
        self.notes: list[tuple[str, int, int, str]] = []
        self.parts: dict[str, PartInfo] = {}
        # Setting names produced for one evaluation from several alternative histories, with those alternatives, and
        # setting names produced from a single history (see _evaluation and _count_candidates).
        self.joined: dict[str, set[str]] = {}
        self.individual: set[str] = set()
        self.settings: dict[str, SettingInfo] = {}
        self.registered: set[str] = set()
        self.sites: dict[str, SiteInfo] = {}
        self.grad_names: dict[tuple[int, int], tuple[str, str]] = {}
        self.files: dict[str, AbsVal] = {}
        self.file_tags: frozenset[str] = frozenset()
        self.dynamic_tags: frozenset[str] = frozenset()
        self.editor_paths: list[str] = []
        self.aliases: dict[str, str] = {}
        self.label_cells: set[int] = set()
        self.unparsed: list[int] = []
        self.active_funcs: list[FuncInfo] = []
        self.call_sites: list[tuple[int, int, int]] = []
        self.call_keys: list[str] = []
        self.config_keys: list[str] = []  # call keys without the calling statement's location (A2, 2026-11-04)
        self.steps = 0
        self._obj_ids = itertools.count(1)
        self._stable_objs: dict[tuple[object, ...], int] = {}
        self._last_self: AbsVal | None = None
        self._expr = self._expr_table()
        self.bin_depth = 0  # > 0 while the two comparisons of a bin (lo <= s) & (s < hi) are evaluated (A2)

    # ------------------------------------------------------------------ cells

    def run_cell(self, index: int, code: str, error: str | None) -> None:
        """Analyse one executed cell in the persistent namespace."""
        lines = code.splitlines()
        self.frame = Frame(scope=self.module, func=None, cell=index, lines=lines)
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError) as exc:
            self._unparsed_cell(index, code, error, exc)
            return
        try:
            compile(code, f"<cell {index}>", "exec", dont_inherit=True)
        except SyntaxError as exc:
            # The kernel compiles a cell before running any of it, so a compile error means nothing ran.
            if _syntax_error_recorded(error):
                self.notes.append(("N-NOT-COMPILED", index, getattr(exc, "lineno", None) or 1,
                                   f"the cell does not compile ({exc.msg}) and its recorded error is a syntax error, "
                                   "so it never ran"))
                return
            self.notes.append(("N-COMPILE-MISMATCH", index, getattr(exc, "lineno", None) or 1,
                               f"the cell does not compile for the audit ({exc.msg}) but no syntax error was recorded; "
                               "it is analysed as if it ran"))
        update_aliases(tree, self.aliases)
        for raw in syntactic_flags(tree, lines, self.aliases):
            self._flag_at(raw.rule, index, raw.line, raw.excerpt, raw.reason)
        self.steps = 0
        try:
            self.exec_block(tree.body)
        except _BudgetExceeded:
            self._flag_at(S_LIMIT, index, 1, line_excerpt(lines, 1),
                          f"analysis stopped after {STEP_LIMIT_PER_CELL} steps; later statements of this cell were not analysed")
        except RecursionError:
            self._flag_at(S_LIMIT, index, 1, line_excerpt(lines, 1),
                          "an expression or call chain is nested too deeply for the audit; the cell was not fully analysed")
        finally:
            self.frames.clear()
            self.frame = Frame(scope=self.module, func=None, cell=index, lines=lines)
            self.loops.clear()
            self.active_loop_ids.clear()
            self.gates.clear()
            self.active_funcs.clear()
            self.call_sites.clear()
            self.call_keys.clear()
            self.config_keys.clear()

    def note_not_executed(self, index: int, reason: str) -> None:
        self.notes.append(("N-NOT-EXECUTED", index, 1, reason))

    def _unparsed_cell(self, index: int, code: str, error: str | None, exc: Exception) -> None:
        self.unparsed.append(index)
        lines = code.splitlines()
        line = getattr(exc, "lineno", None) or 1
        message = getattr(exc, "msg", str(exc))
        if _syntax_error_recorded(error):
            self.notes.append(("N-UNPARSED-NOT-RUN", index, line,
                               f"the cell does not parse ({message}) and its recorded error is a syntax error, so it never ran"))
            return
        hard, suspicious = regex_scan(code)
        for raw in hard:
            self._flag_at(raw.rule, index, raw.line, raw.excerpt, raw.reason)
        reason = f"the cell could not be parsed by the audit ({message}, line {line}) and may have run; regex fallback only"
        if suspicious:
            reason += "; it contains: " + ", ".join(suspicious)
        self._flag_at(S_UNPARSED, index, line, line_excerpt(lines, line), reason)
        tags: set[str] = set()
        for name in regex_mentioned_names(code):
            value = self.module.vars.get(name)
            if value is not None:
                tags |= self._deep_tags(value)
        if any(name in code for name in EVIDENCE_FILE_NAMES) or any(key in code for key in self.label_keys):
            tags |= {LABEL, EVID}
        frozen = frozenset(strip_candref(frozenset(tags)))
        for name in regex_bound_names(code):
            previous = self.module.vars.get(name)
            self.module.vars[name] = previous.plus(frozen) if previous is not None else AbsVal(tags=frozen)

    # --------------------------------------------------------------- flags

    def _flag_at(self, rule: str, cell: int, line: int, excerpt: str, reason: str) -> None:
        key = (rule, cell, line)
        if key not in self.flags:
            self.flags[key] = RawResultFlag(rule, cell, line, excerpt, reason)

    def _flag(self, rule: str, node: ast.AST, reason: str) -> None:
        line = getattr(node, "lineno", 1) or 1
        if self.call_sites:
            cell, call_line, _ = self.call_sites[0]
            if cell != self.frame.cell or self.frame.func is not None:
                reason = f"{reason} (reached from the call at cell {cell} line {call_line})"
        self._flag_at(rule, self.frame.cell, line, line_excerpt(self.frame.lines, line), reason)

    def _excerpt(self, node: ast.AST) -> str:
        return line_excerpt(self.frame.lines, getattr(node, "lineno", 1) or 1)

    # ------------------------------------------------------------- helpers

    def _tick(self) -> None:
        self.steps += 1
        if self.steps > STEP_LIMIT_PER_CELL:
            raise _BudgetExceeded()

    def _new_obj(self) -> int:
        return next(self._obj_ids)

    def _stable_obj(self, *key: object) -> int:
        if key not in self._stable_objs:
            self._stable_objs[key] = self._new_obj()
        return self._stable_objs[key]

    def _callsig(self) -> str:
        """Call context of the code being analysed, part of every site and loop id inside user functions."""
        if not self.call_keys:
            return ""
        return "@" + hashlib.sha1("/".join(self.call_keys).encode()).hexdigest()[:8]

    @staticmethod
    def _call_key(info: FuncInfo, caller: Frame, node: ast.AST, args: Sequence[AbsVal], kwargs: dict[str, AbsVal],
                  bound_self: AbsVal | None, config: bool = False) -> str:
        """Identity of one call: the calling statement, the callee and a token per argument.

        A token is the argument's constant value when known, else the argument expression when it has no side effects
        (no call, walrus, await or yield inside it), else the call's column. Two calls of one function in one
        statement with the same constant or side-effect-free arguments (``acc(apply(p, patch), xr, yr),
        acc(apply(p, patch), xf, yf)``) therefore build one setting, while ``make(0.5)`` and ``make(0.8)``, or
        ``make(s1)`` and ``make(s2)``, build two. Calls the audit makes on the caller's behalf (``map``, key functions)
        have no one-to-one argument expressions and fall back to the column.

        With ``config`` (the configuration count, A2 2026-11-04) an argument expression with calls inside it is still
        named by its text, so ``apply(P, {"b": 0.1 * np.ones(10)})`` and ``apply(P, {"b": 0.2 * np.ones(10)})`` are
        two configurations although the column is the same; the caller's location is left out by the caller.
        """
        col = getattr(node, "col_offset", 0) or 0
        arg_nodes: list[ast.expr | None] = [None] * len(args)
        kw_nodes: dict[str, ast.expr] = {}
        self_node: ast.expr | None = None
        if isinstance(node, ast.Call) and not any(isinstance(a, ast.Starred) for a in node.args) \
                and all(k.arg is not None for k in node.keywords) and len(node.args) == len(args):
            arg_nodes = list(node.args)
            kw_nodes = {k.arg: k.value for k in node.keywords if k.arg is not None}
            if bound_self is not None and isinstance(node.func, ast.Attribute):
                self_node = node.func.value

        def token(value: AbsVal | None, expr: ast.expr | None) -> str:
            if value is not None and value.has_const() and (
                    value.const is None or isinstance(value.const, (int, float, str, bytes, tuple))):
                return "=" + repr(value.const)[:80]
            if expr is not None and (config or not any(isinstance(sub, SIDE_EFFECT_NODES) for sub in ast.walk(expr))):
                return "~" + hashlib.sha1(ast.dump(expr).encode()).hexdigest()[:10]
            return f"!{col}"

        fn = info.node
        parts = [f"{caller.cell}.{caller.stmt[0]}.{caller.stmt[1]}",
                 f"{info.cell}.{getattr(fn, 'lineno', 0)}.{getattr(fn, 'col_offset', 0)}"]
        if bound_self is not None:
            parts.append("self" + token(None, self_node))
        parts.extend(token(value, expr) for value, expr in zip(args, arg_nodes))
        parts.extend(f"{key}{token(value, kw_nodes.get(key))}" for key, value in sorted(kwargs.items()))
        return ":".join(parts)

    def _site(self, node: ast.AST, kind: str) -> str:
        return f"{kind}{self.frame.cell}.{getattr(node, 'lineno', 0)}.{getattr(node, 'col_offset', 0)}{self._callsig()}"

    def _config_site(self, node: ast.AST, kind: str) -> str:
        """``_site`` with the call context reduced to callees and argument tokens (A2, 2026-11-04, configuration count)."""
        context = ""
        if self.config_keys:
            context = "@" + hashlib.sha1("/".join(self.config_keys).encode()).hexdigest()[:8]
        return f"{kind}{self.frame.cell}.{getattr(node, 'lineno', 0)}.{getattr(node, 'col_offset', 0)}{context}"

    def _loop_id(self, node: ast.AST, suffix: str = "") -> str:
        sig = self._callsig().replace("@", "_")
        return f"L{self.frame.cell}_{getattr(node, 'lineno', 0)}_{getattr(node, 'col_offset', 0)}{suffix}{sig}"

    def _gate_tags(self) -> frozenset[str]:
        if not self.gates:
            return frozenset()
        out: set[str] = set()
        for gate in self.gates:
            out |= gate
        return frozenset(out)

    def _deep_tags(self, value: AbsVal, depth: int = 0) -> frozenset[str]:
        tags = set(value.tags)
        if depth < 3:
            for item in (value.items or {}).values():
                tags |= self._deep_tags(item, depth + 1)
            for item in value.elems or ():
                tags |= self._deep_tags(item, depth + 1)
            if value.default is not None:
                tags |= self._deep_tags(value.default, depth + 1)
        return frozenset(tags)

    def _shape_tags(self, shape: AbsVal | None) -> frozenset[str]:
        """Tags a new array takes from its shape argument: a label-dependent count, or a size of the x_fail split."""
        if shape is None:
            return frozenset()
        return self._deep_tags(shape) & {FAILSET, LABEL}

    def _register_site(self, site: str, kind: str, node: ast.AST, tainted_input: bool = False) -> None:
        if site not in self.sites:
            self.sites[site] = SiteInfo(kind, self.frame.cell, getattr(node, "lineno", 1) or 1, self._excerpt(node),
                                        tainted_input=tainted_input)

    # --------------------------------------------------------------- scopes

    def _live_scopes(self) -> list[Scope]:
        seen: list[Scope] = []
        for frame in [*self.frames, self.frame]:
            scope: Scope | None = frame.scope
            while scope is not None:
                if all(scope is not s for s in seen):
                    seen.append(scope)
                scope = scope.parent
        if all(self.module is not s for s in seen):
            seen.append(self.module)
        return seen

    def lookup(self, name: str) -> AbsVal | None:
        scope = self.frame.scope
        if not scope.is_module and name in scope.global_names:
            return self.module.vars.get(name)
        current: Scope | None = scope
        while current is not None:
            if name in current.vars:
                return current.vars[name]
            current = current.parent
        return self.module.vars.get(name)

    def _scope_of(self, name: str) -> Scope | None:
        scope = self.frame.scope
        if not scope.is_module and name in scope.global_names:
            return self.module if name in self.module.vars else None
        current: Scope | None = scope
        while current is not None:
            if name in current.vars:
                return current
            current = current.parent
        return self.module if name in self.module.vars else None

    def _bind(self, name: str, value: AbsVal) -> None:
        scope = self.frame.scope
        if scope.is_module or name in scope.global_names:
            self.module.vars[name] = value
            return
        if name in scope.nonlocal_names:
            current = scope.parent
            while current is not None and not current.is_module:
                if name in current.vars:
                    current.vars[name] = value
                    return
                current = current.parent
        scope.vars[name] = value

    def _rebind_obj(self, obj: int, value: AbsVal) -> set[str]:
        """Give every name (and directly contained item) that refers to object ``obj`` the new value.

        A container whose summary entry (``default``, used when keys or positions are unknown) is that object gets a
        weak update: the summary becomes the join of the old and the new value, because only one of the entries it
        stands for was changed.
        """
        names: set[str] = set()
        changed_pairs: list[tuple[int, str]] = []
        new = replace(value, obj=obj)
        for scope in self._live_scopes():
            for name, current in list(scope.vars.items()):
                if current.obj == obj:
                    scope.vars[name] = new
                    names.add(name)
                    changed_pairs.append((id(scope), name))
                    continue
                changed = False
                items, elems, default = current.items, current.elems, current.default
                if items and any(item.obj == obj for item in items.values()):
                    items = {k: (new if item.obj == obj else item) for k, item in items.items()}
                    changed = True
                if elems and any(item.obj == obj for item in elems):
                    elems = tuple(new if item.obj == obj else item for item in elems)
                    changed = True
                if default is not None and default.obj == obj:
                    default = replace(join(default, new), obj=obj)
                    changed = True
                if not changed:
                    continue
                updated = replace(current, items=items, elems=elems, default=default)
                if current.role in (DICT, PDICT, INSTANCE):
                    scope.vars[name] = self._refresh_container(updated)
                else:
                    extra = strip_candref(new.tags) | candref_tags(frozenset(alternatives(new)))
                    scope.vars[name] = replace(updated, tags=current.tags | extra)
                names.add(name)
                changed_pairs.append((id(scope), name))
        if self.loops and changed_pairs:
            for ctx in self.loops:
                ctx.kept.update(changed_pairs)
        return names

    def _binding_key(self, name: str) -> tuple[int, str] | None:
        scope = self._scope_of(name)
        return (id(scope), name) if scope is not None else None

    def _note_kept(self, name: str) -> None:
        """A name changed in place or accumulated while loops are active: it may hold every iteration's result."""
        key = self._binding_key(name) if self.loops else None
        if key is not None:
            for ctx in self.loops:
                ctx.kept.add(key)

    def _note_rebound(self, name: str, value: AbsVal) -> None:
        """A plain binding of ``name`` inside the loops of the current frame (see ``LoopCtx``)."""
        key = self._binding_key(name)
        if key is None:
            return
        for ctx in self.loops:
            if ctx.frame is not self.frame:
                continue
            ctx.rebound.add(key)
            if f"carry:{ctx.lid}:{name}" in value.tags:
                ctx.kept.add(key)

    def _snapshot(self) -> list[tuple[Scope, dict[str, AbsVal]]]:
        scopes = [self.frame.scope] if self.frame.scope is self.module else [self.frame.scope, self.module]
        return [(scope, dict(scope.vars)) for scope in scopes]

    def _restore(self, snap: list[tuple[Scope, dict[str, AbsVal]]]) -> None:
        for scope, values in snap:
            scope.vars = dict(values)

    @staticmethod
    def _join_snap(a: list[tuple[Scope, dict[str, AbsVal]]], b: list[tuple[Scope, dict[str, AbsVal]]]
                   ) -> list[tuple[Scope, dict[str, AbsVal]]]:
        out = []
        for (scope, va), (_, vb) in zip(a, b):
            merged = dict(va)
            for name, value in vb.items():
                merged[name] = join(merged[name], value) if name in merged else value
            out.append((scope, merged))
        return out

    @staticmethod
    def _snap_signature(snap: list[tuple[Scope, dict[str, AbsVal]]]) -> tuple[object, ...]:
        return tuple(tuple(sorted((name, signature(value)) for name, value in values.items())) for _, values in snap)

    @staticmethod
    def _changed_names(a: list[tuple[Scope, dict[str, AbsVal]]], b: list[tuple[Scope, dict[str, AbsVal]]]) -> set[str]:
        changed: set[str] = set()
        for (_, va), (_, vb) in zip(a, b):
            for name, value in vb.items():
                if value.role in (FUNC, MODULE):
                    continue
                if name not in va or signature(va[name]) != signature(value):
                    changed.add(name)
        return changed

    # ----------------------------------------------------------- statements

    def exec_block(self, stmts: Iterable[ast.stmt]) -> None:
        for stmt in stmts:
            self.exec_stmt(stmt)

    def exec_stmt(self, node: ast.stmt) -> None:
        self._tick()
        self.frame.stmt = (getattr(node, "lineno", 0) or 0, getattr(node, "col_offset", 0) or 0)
        handler = getattr(self, "st_" + type(node).__name__, None)
        if handler is not None:
            handler(node)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.ev(child)
            elif isinstance(child, ast.stmt):
                self.exec_stmt(child)

    def st_Expr(self, node: ast.Expr) -> None:
        self.ev(node.value)

    def st_Assign(self, node: ast.Assign) -> None:
        value = self.ev(node.value)
        for target in node.targets:
            self.assign(target, value, node)

    def st_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.assign(node.target, self.ev(node.value), node)

    def st_AugAssign(self, node: ast.AugAssign) -> None:
        value = self.ev(node.value)
        target = node.target
        if isinstance(target, ast.Name):
            current = self.lookup(target.id) or AbsVal()
            new = self.combine(node.op, current, value, node)
            if current.obj is not None and current.role in (WEIGHT, SAMPLE, STAT, OTHER, SEQ, DICT, PDICT):
                new = replace(new, obj=current.obj)
                names = self._rebind_obj(current.obj, new)
                if target.id not in names:
                    self._bind(target.id, new)
                    names.add(target.id)
            else:
                self._bind(target.id, new)
                names = {target.id}
            self._note_kept(target.id)
            self._after_assign(target.id, new, node)
            self._record_event(names, new.tags, node, aug=True)
        else:
            self._mutate(target, value, node, node.op)

    def st_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._define_function(node)

    def st_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._define_function(node)

    def st_ClassDef(self, node: ast.ClassDef) -> None:
        bases = []
        for base in node.bases:
            value = self.ev(base)
            if isinstance(value.fn, ClassInfo):
                bases.append(value.fn)
        for keyword in node.keywords:
            self.ev(keyword.value)
        info = ClassInfo(node.name, {}, {}, bases)
        scope = Scope(parent=None if self.frame.scope.is_module else self.frame.scope, is_module=False)
        saved = self.frame.scope
        self.frame.scope = scope
        try:
            self.exec_block(node.body)
        finally:
            self.frame.scope = saved
        for name, value in scope.vars.items():
            if value.role == FUNC and isinstance(value.fn, FuncInfo):
                value.fn.owner = info
                value.fn.closure = None if saved.is_module else saved
                info.methods[name] = value.fn
            else:
                info.attrs[name] = value
        self._bind(node.name, AbsVal(role=FUNC, fn=info))

    def st_Return(self, node: ast.Return) -> None:
        value = self.ev(node.value) if node.value is not None else AbsVal(role=CONST, const=None)
        self.frame.returns.append(value)
        if self.loops:
            self.loops[-1].exit_gates |= self._gate_tags()

    def st_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                scope = self._scope_of(target.id)
                if scope is not None:
                    scope.vars.pop(target.id, None)
            else:
                self.ev(target)

    def st_If(self, node: ast.If) -> None:
        test = self.ev(node.test)
        truth = _known_truth(test)
        if truth is not None:
            # A2 (2026-11-04): a test whose value is an exactly known constant (``if bias_scale and ...`` with
            # bias_scale=None) runs one branch only. Before, both ran, and an unscaled forward call built a candidate.
            self.exec_block(node.body if truth else node.orelse)
            return
        entry = self._snapshot()
        self.gates.append(test.tags)
        try:
            self.exec_block(node.body)
            after_body = self._snapshot()
            self._restore(entry)
            self.exec_block(node.orelse)
            after_else = self._snapshot()
        finally:
            self.gates.pop()
        self._restore(self._join_snap(after_body, after_else))

    def st_With(self, node: ast.With) -> None:
        for item in node.items:
            value = self.ev(item.context_expr)
            if item.optional_vars is not None:
                self.assign(item.optional_vars, value, node, record=False)
        self.exec_block(node.body)

    st_AsyncWith = st_With

    def st_Raise(self, node: ast.Raise) -> None:
        if node.exc is not None:
            self.ev(node.exc)
        if self.loops:
            self.loops[-1].exit_gates |= self._gate_tags()

    def st_Try(self, node: ast.Try) -> None:
        entry = self._snapshot()
        self.exec_block(node.body)
        after_body = self._snapshot()
        outs = []
        for handler in node.handlers:
            self._restore(self._join_snap(entry, after_body))
            if handler.type is not None:
                self.ev(handler.type)
            if handler.name:
                self._bind(handler.name, AbsVal())
            self.exec_block(handler.body)
            outs.append(self._snapshot())
        self._restore(after_body)
        self.exec_block(node.orelse)
        merged = self._snapshot()
        for out in outs:
            merged = self._join_snap(merged, out)
        self._restore(merged)
        self.exec_block(node.finalbody)

    st_TryStar = st_Try

    def st_Assert(self, node: ast.Assert) -> None:
        self.ev(node.test)
        if node.msg is not None:
            self.ev(node.msg)

    def st_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self._bind(alias.asname, AbsVal(role=MODULE, qual=alias.name))
            else:
                root = alias.name.split(".")[0]
                self._bind(root, AbsVal(role=MODULE, qual=root))
            self._check_editor_module(alias.name, node)

    def st_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level == 0:
            self._check_editor_module(module, node)
        for alias in node.names:
            if alias.name == "*":
                continue
            qual = f"{module}.{alias.name}" if module and node.level == 0 else None
            self._bind(alias.asname or alias.name, AbsVal(role=MODULE, qual=qual) if qual else AbsVal())

    def _check_editor_module(self, module: str, node: ast.AST) -> None:
        root = module.split(".")[0]
        if self.editor_paths and root and root not in KNOWN_MODULES:
            self._flag(S_DYNAMIC, node, f"imports module {module} while editor-writable paths are on sys.path "
                                        f"({', '.join(self.editor_paths)}); its code is not visible to the audit")

    def st_Global(self, node: ast.Global) -> None:
        if not self.frame.scope.is_module:
            self.frame.scope.global_names.update(node.names)

    def st_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.frame.scope.nonlocal_names.update(node.names)

    def st_Pass(self, node: ast.Pass) -> None:
        return

    def st_Break(self, node: ast.Break) -> None:
        if self.loops:
            self.loops[-1].exit_gates |= self._gate_tags()

    st_Continue = st_Break

    def st_Match(self, node: ast.Match) -> None:
        subject = self.ev(node.subject)
        entry = self._snapshot()
        outs = []
        for case in node.cases:
            self._restore(entry)
            for sub in ast.walk(case.pattern):
                name = getattr(sub, "name", None)
                if isinstance(name, str):
                    self._bind(name, AbsVal(tags=subject.tags, role=subject.role))
                for rest in getattr(sub, "names", []) or []:
                    if isinstance(rest, str):
                        self._bind(rest, AbsVal(tags=subject.tags))
            gate = subject.tags
            if case.guard is not None:
                gate = gate | self.ev(case.guard).tags
            self.gates.append(gate)
            try:
                self.exec_block(case.body)
            finally:
                self.gates.pop()
            outs.append(self._snapshot())
        merged = entry
        for out in outs:
            merged = self._join_snap(merged, out)
        self._restore(merged)

    def _define_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        defaults = [self.ev(default) for default in node.args.defaults]
        kw_defaults = {
            arg.arg: self.ev(default)
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
            if default is not None
        }
        for decorator in node.decorator_list:
            self.ev(decorator)
        info = FuncInfo(node, node.name, None if self.frame.scope.is_module else self.frame.scope, self.frame.cell,
                        self.frame.lines, defaults, kw_defaults)
        self._bind(node.name, AbsVal(role=FUNC, fn=info))

    # ---------------------------------------------------------- assignment

    def assign(self, target: ast.expr, value: AbsVal, node: ast.AST, record: bool = True) -> None:
        if isinstance(target, ast.Name):
            old = self.lookup(target.id)
            if old is not None and old.role == WEIGHT and PARAM in old.tags and value is not old:
                fresh = _grad_tags(value.tags) - _grad_tags(old.tags)
                if fresh:
                    self._update_sink(fresh, node, f"parameter array {target.id} is reassigned")
            self._after_assign(target.id, value, node)
            self._bind(target.id, value)
            if self.loops:
                self._note_rebound(target.id, value)
            if record:
                self._record_event({target.id}, value.tags, node)
        elif isinstance(target, (ast.Tuple, ast.List)):
            self._unpack(target.elts, value, node, record)
        elif isinstance(target, ast.Starred):
            self.assign(target.value, AbsVal(role=SEQ, tags=value.tags, default=self._element_of(value)), node, record)
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            self._mutate(target, value, node, None, record)

    def _after_assign(self, name: str, value: AbsVal, node: ast.AST) -> None:
        gate = self._gate_tags()
        if gate:
            self._best_keeping(value, gate)
        if GRAD_NAME.match(name) and (is_tainted(value.tags) or LABEL_BUILT in value.tags) and not _grad_tags(value.tags):
            key = (self.frame.cell, getattr(node, "lineno", 1) or 1)
            self.grad_names.setdefault(key, (name, self._excerpt(node)))

    def _unpack(self, elts: list[ast.expr], value: AbsVal, node: ast.AST, record: bool) -> None:
        starred = [i for i, elt in enumerate(elts) if isinstance(elt, ast.Starred)]
        if value.elems is not None and not starred and len(value.elems) == len(elts):
            for elt, item in zip(elts, value.elems):
                self.assign(elt, item, node, record)
            return
        if value.elems is not None and len(starred) == 1:
            star = starred[0]
            after = len(elts) - star - 1
            if len(value.elems) >= len(elts) - 1:
                for elt, item in zip(elts[:star], value.elems[:star]):
                    self.assign(elt, item, node, record)
                middle = value.elems[star: len(value.elems) - after]
                self.assign(elts[star], AbsVal(role=SEQ, elems=tuple(middle), length=len(middle),
                                               default=join_all(list(middle)),
                                               tags=frozenset().union(*(m.tags for m in middle))), node, record)
                for elt, item in zip(elts[star + 1:], value.elems[len(value.elems) - after:]):
                    self.assign(elt, item, node, record)
                return
        element = self._element_of(value)
        for elt in elts:
            self.assign(elt, element, node, record)

    def _record_event(self, names: set[str], value_tags: frozenset[str], node: ast.AST, aug: bool = False,
                      index_tags: frozenset[str] | None = None) -> None:
        if not self.loops or not names:
            return
        gate = self._gate_tags()
        line = getattr(node, "lineno", 1) or 1
        frozen_names = frozenset(names)
        for ctx in self.loops:
            key = (self.frame.cell, line, frozen_names)
            existing = ctx.events.get(key)
            if existing is None:
                ctx.events[key] = Event(frozen_names, value_tags, gate, self.frame.cell, line, self._excerpt(node))
            else:
                existing.value_tags = existing.value_tags | value_tags
                existing.gate_tags = existing.gate_tags | gate
            loop = ctx.lid
            if aug:
                kind: str | None = "v"
            elif f"lv:{loop}" in value_tags:
                kind = "v"
            elif index_tags is not None and f"lim:{loop}" in index_tags:
                kind = "m"
            elif index_tags is not None and f"li:{loop}" in index_tags:
                kind = "i"
            elif f"lim:{loop}" in value_tags:
                kind = "m"
            elif f"li:{loop}" in value_tags:
                kind = "i"
            else:
                kind = None
            for name in names:
                ctx.var_kinds[name] = _strongest(ctx.var_kinds.get(name), kind) or ctx.var_kinds.get(name, "") or ""
                if not ctx.var_kinds[name]:
                    del ctx.var_kinds[name]

    def _mutate(self, target: ast.expr, value: AbsVal, node: ast.AST, aug_op: ast.operator | None,
                record: bool = True) -> None:
        chain: list[ast.expr] = []
        current: ast.expr = target
        while isinstance(current, (ast.Subscript, ast.Attribute)):
            chain.append(current)
            current = current.value
        chain.reverse()
        root = current
        root_value = self.ev(root)
        index_values = [self.ev(level.slice) if isinstance(level, ast.Subscript) else None for level in chain]
        holders = [root_value]
        for level, index in zip(chain[:-1], index_values[:-1]):
            holders.append(self._read_level(holders[-1], level, index, node))
        last, last_index = chain[-1], index_values[-1]
        holder = holders[-1]
        if holder.role == GLOBALS and isinstance(last, ast.Subscript) and last_index is not None:
            self._write_globals(last_index, value)
            return
        if aug_op is not None and isinstance(aug_op, ast.Mult) and _is_zero(value):
            element = value  # w[j] *= 0 knocks the element out, like w[j] = 0
        elif aug_op is not None:
            old = self._read_level(holder, last, last_index, node)
            element = self.combine(aug_op, old, value, node)
        else:
            element = value
        single = isinstance(last, ast.Subscript) and self._is_single_index(last.slice)
        converted_index = None
        if last_index is not None:
            converted_index = self._index_variance(last_index.tags, single)
        updated = self._write_level(holder, last, last_index, element, node, single, aug_op=aug_op)
        for i in range(len(chain) - 2, -1, -1):
            updated = self._write_level(holders[i], chain[i], index_values[i], updated, node, True, structural=True)
        names: set[str] = set()
        if isinstance(root, ast.Name):
            if root_value.obj is not None:
                names = self._rebind_obj(root_value.obj, updated)
            if root.id not in names:
                self._bind(root.id, updated)
                names.add(root.id)
        elif root_value.obj is not None:
            names = self._rebind_obj(root_value.obj, updated)
        if isinstance(root, ast.Name):
            self._note_kept(root.id)
        gate = self._gate_tags()
        if gate:
            self._best_keeping(element, gate)
        if record:
            self._record_event(names, element.tags, node, aug=aug_op is not None, index_tags=converted_index)

    def _write_globals(self, key: AbsVal, value: AbsVal) -> None:
        if isinstance(key.const, str) and key.const.isidentifier():
            self.module.vars[key.const] = value
        else:
            self.dynamic_tags = self.dynamic_tags | strip_candref(self._deep_tags(value))

    def _read_level(self, holder: AbsVal, level: ast.expr, index: AbsVal | None, node: ast.AST) -> AbsVal:
        if isinstance(level, ast.Attribute):
            return self._attribute(holder, level.attr, level)
        assert isinstance(level, ast.Subscript) and index is not None
        return self._subscript(holder, index, level, level.slice)

    def _write_level(self, holder: AbsVal, level: ast.expr, index: AbsVal | None, element: AbsVal, node: ast.AST,
                     single: bool, structural: bool = False, aug_op: ast.operator | None = None) -> AbsVal:
        if isinstance(level, ast.Attribute):
            items = dict(holder.items or {})
            items[level.attr] = element
            if holder.role in (INSTANCE, DICT, PDICT) or holder.items is not None:
                return self._refresh_container(replace(holder, items=items))
            return holder.plus(strip_candref(element.tags))
        assert index is not None
        role = holder.role
        key = index.const if isinstance(index.const, (str, int)) and not isinstance(index.const, bool) else None
        if role in (PDICT, DICT, MODEL_C, EVID_C, INSTANCE):
            if role in (PDICT, MODEL_C):
                fresh = _grad_tags(element.tags) - _grad_tags(holder.tags)
                if fresh:
                    self._update_sink(fresh, node, "a parameter dictionary entry is written")
            items = dict(holder.items or {})
            default = holder.default
            if role == MODEL_C and default is None:
                default = self._model_item(holder, NOCONST)
            if isinstance(key, str):
                items[key] = element
            else:
                default = join(default, element)
                items = {k: join(v, element) for k, v in items.items()}
            new_role = PDICT if role == MODEL_C else role
            if role == DICT and element.role == WEIGHT and PARAM in element.tags:
                new_role = PDICT
            updated = replace(holder, role=new_role, items=items, default=default,
                              tags=holder.tags | strip_candref(index.tags))
            return self._refresh_container(updated)
        if role == SEQ:
            elems = holder.elems
            if elems is not None and isinstance(key, int) and -len(elems) <= key < len(elems):
                elems = tuple(element if i == key % len(elems) else item for i, item in enumerate(elems))
            else:
                elems = None
            tags = holder.tags | strip_candref(element.tags) | candref_tags(frozenset(alternatives(element)))
            return replace(holder, elems=elems, default=join(holder.default, element), tags=tags | index.tags)
        index_tags = self._index_variance(index.tags, single)
        tags = holder.tags | element.tags | index_tags
        if structural:
            return replace(holder, tags=strip_candref(tags), cands=holder.cands | element.cands,
                           pristine=holder.pristine and element.pristine)
        if role == WEIGHT and PARAM in holder.tags:
            fresh = _grad_tags(element.tags) - _grad_tags(holder.tags)
            if fresh:
                self._update_sink(fresh, node, "a parameter array is written in place")
            if single and not structural:
                # A knockout (w[j] = 0, or a value fixed before the loop) varies only in which element it hits. A
                # value that changes with the iteration (b[k] -= 0.3, w[j] = w[j] * s) is a perturbation sweep: every
                # iteration is a different candidate setting, not an ablation.
                for ctx in self.loops:
                    lid = ctx.lid
                    if f"li:{lid}" in index_tags and any(t in element.tags for t in (
                            f"lv:{lid}", f"li:{lid}", f"lim:{lid}", f"lk:{lid}")):
                        tags = tags | {f"lv:{lid}"}
            cands = self._new_cands(node, holder.cands)
            return replace(holder, tags=self._as_candidate(strip_candref(tags)), cands=cands, pristine=False,
                           evaldiff=False)
        if (isinstance(aug_op, (ast.Sub, ast.Add)) and role == SAMPLE and PARAM in holder.tags and EVID in holder.tags
                and HARD not in holder.tags and LABEL in index.tags):
            # p[np.arange(n), y] -= 1: model outputs minus a one-hot label encoding, written in place. A plain
            # assignment at the label positions (masking the true class) is not an error signal.
            site = self._site(node, "err")
            self._register_site(site, "err", node)
            tags = tags | {f"err:{site}"}
        # A2 (2026-11-04): per-sample values written into a buffer (pad[:, :, 4:36, 4:36] = x, out[i] = f(x[i])) make
        # the buffer per-sample; before, it kept the role of np.zeros, so features of augmented evidence were not
        # per-sample data and their readouts and evaluations were missed.
        new_role = SAMPLE if element.role == SAMPLE and role in (STAT, OTHER) else role
        return replace(holder, role=new_role, tags=strip_candref(tags), pristine=False, evaldiff=False)

    def _index_variance(self, tags: frozenset[str], single: bool) -> frozenset[str]:
        if not any(t.startswith("lv:") for t in tags):
            return tags
        out = set()
        for tag in tags:
            if tag.startswith("lv:"):
                out.add(("li:" if single else "lim:") + tag[3:])
            else:
                out.add(tag)
        return frozenset(out)

    def _loops_of(self, tags: frozenset[str]) -> set[str]:
        """Active loops that ``tags`` vary with in any way (value, index, row selection or key read)."""
        loops = {t.split(":", 1)[1] for t in tags if t.startswith(LOOP_VARIANCE_PREFIXES)}
        return loops & self.active_loop_ids

    def _key_read_variance(self, base: AbsVal, index_tags: frozenset[str], slice_node: ast.expr) -> frozenset[str]:
        """Audit amendment A2 (2026-11-04): a loop-varying index into a candidate setting or a readout reads another
        part (a row, a class prototype) of the same setting each iteration.

        Such loops become ``lk:`` (key read): the setting does not vary with them, so a per-row read of a candidate in
        a loop over classes is one evaluation of that candidate, not a single-element sweep (before A2 it was taken
        for an ablation and excluded). Loops the base itself varies with keep the index-variance reading.
        """
        own = {t.split(":", 1)[1] for t in base.tags if t.startswith(LOOP_VARIANCE_PREFIXES)}
        keyed = {lid for lid in self._loops_of(index_tags) if lid not in own}
        tags = frozenset(t for t in index_tags if not (t.startswith(("lv:", "li:", "lim:"))
                                                       and t.split(":", 1)[1] in keyed))
        tags = tags | {f"lk:{lid}" for lid in keyed}
        if base.role == WEIGHT:
            tags = self._index_variance(tags, self._is_single_index(slice_node))
        return tags

    def _row_variance(self, index_tags: frozenset[str]) -> frozenset[str]:
        """Audit amendment A2 (2026-11-04): a loop-varying selection of evidence rows (a leave-one-out mask, a fold,
        the rows of one class) makes the selected data vary with the loop in value, through the rows (``lf:``).

        Before A2 a mask written with one loop-varying element (``m[i] = False``) kept the single-index reading, so a
        setting fitted on each leave-one-out fold was taken for a single-element ablation sweep and excluded. A
        setting built from such data is a separate setting per fold for the literal count and one configuration for
        the configuration count.
        """
        loops = {t.split(":", 1)[1] for t in index_tags if t.startswith(("lv:", "li:", "lim:"))} & self.active_loop_ids
        if not loops:
            return index_tags
        tags = frozenset(t for t in index_tags if not (t.startswith(("li:", "lim:")) and t.split(":", 1)[1] in loops))
        # A loop whose variance comes from a label equality (``y == c``) selects the rows of one class: lc, not lf.
        folds = {lid for lid in loops if f"lc:{lid}" not in index_tags}
        return tags | {f"lv:{lid}" for lid in loops} | {f"lf:{lid}" for lid in folds}

    def _class_view(self, values: Sequence[AbsVal]) -> frozenset[str]:
        """``lc:`` tags of a label equality with a loop-varying value (``y == c`` in a loop over classes, A2
        2026-11-04): the selection it makes varies with the loop by class. A readout built per class inside the loop
        is one row of one prototype head (see ``_sig``)."""
        loops: set[str] = set()
        for lid in self.active_loop_ids:
            varying = [v for v in values if f"lv:{lid}" in v.tags and v.role not in (WEIGHT, PDICT, MODEL_C)
                       and PARAM not in v.tags and not any(t.startswith("eval:") for t in v.tags)]
            labels = [v for v in values if LABEL in v.tags and f"lv:{lid}" not in v.tags]
            if varying and labels:
                loops.add(lid)
        return frozenset(f"lc:{lid}" for lid in loops)

    def _hyperparameter_loops(self, values: Iterable[AbsVal]) -> set[str]:
        """Active loops along which one of ``values`` is a free parameter of a decision rule: a value taken from the
        loop (a threshold, an offset, a strength) that carries no evidence, label, model or evaluation information."""
        loops: set[str] = set()
        for value in values:
            if value.role in (SAMPLE, WEIGHT, PDICT, MODEL_C, EVID_C, FUNC, MODULE, METHOD):
                continue
            if value.tags & {LABEL, PARAM, EVID} or any(t.startswith(("eval:", "evalc:", "carry:")) for t in value.tags):
                continue
            loops |= {t[3:] for t in value.tags if t.startswith("lv:")} & self.active_loop_ids
        return loops

    def _decision_variance(self, tags: frozenset[str], params: Iterable[AbsVal]) -> frozenset[str]:
        """Audit amendment A2 (2026-11-04): an evaluated setting combined with a loop-varying free parameter of the
        decision rule (``scores > kappa``, ``logits + delta`` in a loop over kappa or delta) is another setting per
        value of that parameter. Before A2 a threshold applied after the evaluation did not multiply the count."""
        evals = sorted(t for t in tags if t.startswith("eval:"))
        if not evals:
            return tags
        loops = self._hyperparameter_loops(params)
        if not loops:
            return tags
        # The rule is one setting however many evaluations it combines (a new-class prototype score over the best
        # old-class score, thresholded): one composite setting per value of the free parameter, not one per part.
        key = ""
        sig = ";".join(f"{lid}=d" for lid in sorted(loops))
        for tag in evals:
            part_key, part_sig = _split_eval(tag)
            key = _merge_keys(key, part_key)
            sig = _merge_sigs(sig, part_sig)
        return (tags - frozenset(evals)) | {f"eval:{key}#{sig}"}

    def _readout_evaluation(self, readout: AbsVal, data: AbsVal, tags: frozenset[str]) -> frozenset[str]:
        """Tags of a feature-space readout applied to per-sample data (A2, 2026-11-04): the readout is evaluated.

        As for a parameter setting built from labels, the evaluation output is not itself a label score: the
        readout's label dependence (it was built from label-selected rows) becomes ``label_built`` unless the data
        side is label-dependent. The readout tags do not pass into the output; its evaluation tag names it.
        """
        add, remove = self._evaluation(readout, data)
        tags = (tags - remove - _readout_tags(readout.tags)) | add
        if LABEL in readout.tags and LABEL not in data.tags:
            tags = (tags - {LABEL}) | {LABEL_BUILT}
        return tags

    def _readout_tag(self, node: ast.AST | None, array: AbsVal, axis: AbsVal | None, role: str,
                     tags: frozenset[str], name: str) -> frozenset[str]:
        """Audit amendment A2 (2026-11-04): a reduction over the sample axis of label-selected model features (a
        class-mean prototype ``F[y == c].mean(0)``, a mean residual) is a feature-space readout, tagged ``rd:<site>``.

        Before A2 such a statistic was descriptive however it was used, so a head whose rows are class prototypes,
        scored in feature space (``F @ protos.T``), was never a candidate setting. A readout applied to evidence
        features is now evaluated like a parameter setting (``_matmul``, ``combine``).
        """
        if node is None or name not in READOUT_REDUCE or array.role != SAMPLE or role != STAT:
            return tags
        if axis is None or not axis.has_const():
            return tags
        ax = axis.const
        if isinstance(ax, bool) or not (ax == 0 or (isinstance(ax, tuple) and 0 in ax)):
            return tags
        if PARAM not in tags or EVID not in tags:
            return tags
        site = self._site(node, "r")
        if site not in self.parts:
            self.parts[site] = PartInfo(self.frame.cell, getattr(node, "lineno", 1) or 1, self._excerpt(node),
                                        config=self._config_site(node, "r"))
        return tags | {f"rd:{site}"}

    def _varies_now(self, value: AbsVal) -> bool:
        return any(f"lv:{loop}" in value.tags for loop in self.active_loop_ids)

    def _is_single_index(self, slice_node: ast.expr) -> bool:
        """True when every loop-varying component of an index selects one element (not a slice or a list)."""
        components = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]
        for component in components:
            if isinstance(component, ast.Slice):
                parts = [p for p in (component.lower, component.upper, component.step) if p is not None]
                if any(self._varies_now(self.ev(p)) for p in parts):
                    return False
                continue
            value = self.ev(component)
            if not self._varies_now(value):
                continue
            if isinstance(component, (ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.Call)):
                return False
            if isinstance(component, ast.Subscript) and isinstance(component.slice, ast.Slice):
                return False
            if value.multi or value.role == SEQ:
                return False
        return True

    # --------------------------------------------------------- candidates

    def _as_candidate(self, tags: frozenset[str]) -> frozenset[str]:
        """A parameter setting built from label-dependent statistics is label-built, not itself a label score."""
        if LABEL in tags:
            return (tags - {LABEL}) | {LABEL_BUILT}
        return tags

    @staticmethod
    def _cap_alternatives(alts: set[str]) -> frozenset[str]:
        """At most ``MAX_ALTERNATIVES`` alternatives; the overflow is merged into one, so no site is lost."""
        if len(alts) <= MAX_ALTERNATIVES:
            return frozenset(alts)
        ordered = sorted(alts)
        return frozenset([*ordered[: MAX_ALTERNATIVES - 1], _union_key(ordered[MAX_ALTERNATIVES - 1:])])

    def _new_cands(self, node: ast.AST, parents: frozenset[str]) -> frozenset[str]:
        """Alternative histories after a modification at ``node``: each parent history plus this site."""
        site = self._site(node, "c")
        if site not in self.parts:
            self.parts[site] = PartInfo(self.frame.cell, getattr(node, "lineno", 1) or 1, self._excerpt(node),
                                        config=self._config_site(node, "c"))
        if not parents:
            return frozenset({site})
        return self._cap_alternatives({_merge_keys(parent, site) for parent in parents})

    def _combine_alternatives(self, groups: Iterable[frozenset[str]]) -> frozenset[str]:
        """Histories of a container: one alternative of every entry, merged."""
        combos = {""}
        for group in groups:
            if not group:
                continue
            combos = {_merge_keys(c, a) for c in combos for a in group}
            if len(combos) > MAX_ALTERNATIVES:
                combos = set(self._cap_alternatives(combos))
        return frozenset(c for c in combos if c)

    def _sig(self, weight: AbsVal) -> str:
        parts = []
        for ctx in self.loops:
            loop = ctx.lid
            if f"enum:{loop}" in weight.tags:
                continue
            if f"lv:{loop}" in weight.tags and loop not in weight.keyvar:
                # A2 (2026-11-04): a setting that varies with the loop only through which evidence rows it was built
                # from (a leave-one-out fold, a view: lf; the rows of one class: lc) is kind "f", a separate setting
                # and one configuration. A readout built per class inside a class loop is one row of one prototype
                # head, scored row by row: the head does not vary with that loop.
                if f"lc:{loop}" in weight.tags and not weight.cands and _readout_tags(weight.tags):
                    continue
                rows = f"lf:{loop}" in weight.tags or f"lc:{loop}" in weight.tags
                parts.append(f"{loop}=f" if rows else f"{loop}=v")
            elif f"lim:{loop}" in weight.tags:
                parts.append(f"{loop}=m")
            elif f"li:{loop}" in weight.tags:
                parts.append(f"{loop}=i")
        if RANDOM in weight.tags:
            parts.append("rng")
        return ";".join(parts)

    def _evaluation(self, weight: AbsVal, data: AbsVal) -> tuple[frozenset[str], frozenset[str]]:
        """Tags produced by evaluating parameter setting ``weight`` on evidence-derived ``data``.

        Inside a loop that enumerates known candidate settings (``enum:L``), each alternative history is a separate
        setting. Otherwise the alternatives are the possible histories of the one setting this evaluation sees (they
        come from control-flow merges) and are named by their union; ``_count_candidates`` does not count such a
        union again when each of its alternatives was scored on its own.

        Evaluations the setting itself was built from (a direction projected out with ``M @ d``, a bias recentred on
        the model's own outputs) do not pass into the output: the output evaluates the setting, whose history names
        those steps already, so scoring it is one scoring event of one setting.
        """
        add: set[str] = set()
        remove: set[str] = set()
        cands = weight.cands
        if not cands:
            # A2 (2026-11-04): a feature-space readout (class-mean prototypes of label-selected model features, see
            # _readout_tag) is a setting named by its readout sites. A parameter setting built from a readout keeps
            # its own modification history, so its tensors keep one name.
            readouts = sorted(t[len("rd:"):] for t in weight.tags if t.startswith("rd:"))
            if readouts:
                cands = frozenset({"+".join(readouts)})
        if cands:
            remove.update(t for t in weight.tags if t.startswith("eval:"))
            sig = self._sig(weight)
            enumerated = any(f"enum:{ctx.lid}" in weight.tags for ctx in self.loops)
            if enumerated or len(cands) == 1:
                keys = sorted(cands)
                self.individual.update(k for k in keys if k not in self.joined)
            else:
                union = _union_key(cands)
                self.joined.setdefault(union, set()).update(cands)
                keys = [union]
            data_evals = [t for t in data.tags if t.startswith("eval:")]
            if data_evals:
                for tag in data_evals:
                    data_key, data_sig = _split_eval(tag)
                    for key in keys:
                        merged = _merge_keys(data_key, key)
                        if merged != key:
                            self.individual.add(merged)
                        add.add(f"eval:{merged}#{_merge_sigs(data_sig, sig)}")
                remove.update(data_evals)
            else:
                for key in keys:
                    add.add(f"eval:{key}#{sig}")
        for tag in weight.tags:
            if tag.startswith("carry:"):
                add.add("evalc:" + tag[len("carry:"):])
        if len(add) > MAX_EVAL_TAGS:
            kept = sorted(t for t in add if not t.startswith("eval:"))
            add = set(kept) | {f"eval:MANY#{UNBOUNDED_LABEL}"}
        return frozenset(add), frozenset(remove)

    def _register(self, tag: str, node: ast.AST) -> None:
        """Record that a candidate setting was scored against labels, at the top-level statement that did it."""
        self.registered.add(tag)
        key, sig = _split_eval(tag)
        info = self.settings.get(key)
        if info is None:
            if self.frames and self.call_sites:
                top, line = self.frames[0], self.call_sites[0][1] or 1
                cell, excerpt = top.cell, line_excerpt(top.lines, line)
            else:
                line = getattr(node, "lineno", 1) or 1
                cell, excerpt = self.frame.cell, self._excerpt(node)
            info = self.settings[key] = SettingInfo(key, set(), cell, line, excerpt)
        info.sigs.add(sig)

    def _best_keeping(self, value: AbsVal, gate: frozenset[str]) -> None:
        if not is_tainted(gate):
            return
        evals = [t for t in gate if t.startswith("eval:")]
        if not evals:
            return
        alts = alternatives(value)
        if not alts:
            return
        eval_parts = [set(_split_eval(t)[0].split("+")) for t in evals]
        for alt in alts:
            alt_parts = set(alt.split("+"))
            if any(alt_parts <= parts for parts in eval_parts):
                for part in alt_parts:
                    if part in self.parts:
                        self.parts[part].best_kept = True

    def _update_sink(self, grad_tags: frozenset[str], node: ast.AST, what: str) -> None:
        names = {"err": "an error signal (model output minus a label encoding)",
                 "grad": "a hand-written gradient (an error signal contracted with activations)",
                 "fd": "a finite-difference quotient of label-dependent scores",
                 "sd": "a difference of label-dependent scores of two parameter settings (a finite difference)"}
        for tag in sorted(grad_tags):
            kind, site = tag.split(":", 1)
            info = self.sites.get(site)
            where = f" at cell {info.cell} line {info.line}" if info is not None else ""
            if info is not None:
                info.used = True
            rule = V_FINITE_DIFF if kind in ("fd", "sd") else V_BACKPROP
            self._flag(rule, node, f"{what} from {names[kind]} computed{where}")

    # ------------------------------------------------------------ expressions

    def _expr_table(self) -> dict[type, Callable[[ast.AST], AbsVal]]:
        table: dict[type, Callable[[ast.AST], AbsVal]] = {
            ast.Constant: self.ex_Constant, ast.Name: self.ex_Name, ast.Attribute: self.ex_Attribute,
            ast.Subscript: self.ex_Subscript, ast.BinOp: self.ex_BinOp, ast.UnaryOp: self.ex_UnaryOp,
            ast.BoolOp: self.ex_BoolOp, ast.Compare: self.ex_Compare, ast.Call: self.ex_Call, ast.IfExp: self.ex_IfExp,
            ast.Lambda: self.ex_Lambda, ast.List: self.ex_Display, ast.Tuple: self.ex_Display, ast.Set: self.ex_Display,
            ast.Dict: self.ex_Dict, ast.ListComp: self.ex_Comp, ast.SetComp: self.ex_Comp,
            ast.GeneratorExp: self.ex_Comp, ast.DictComp: self.ex_Comp, ast.JoinedStr: self.ex_JoinedStr,
            ast.FormattedValue: self.ex_FormattedValue, ast.NamedExpr: self.ex_NamedExpr, ast.Starred: self.ex_Starred,
            ast.Slice: self.ex_Slice, ast.Await: self.ex_Yield, ast.Yield: self.ex_Yield, ast.YieldFrom: self.ex_Yield,
        }
        return table

    def ev(self, node: ast.AST) -> AbsVal:
        self._tick()
        handler = self._expr.get(type(node))
        value = handler(node) if handler is not None else self._ev_generic(node)
        return self._finish(value, node)

    def _finish(self, value: AbsVal, node: ast.AST) -> AbsVal:
        tags = value.tags
        if not tags or LABEL not in tags:
            return value
        extra = []
        for tag in tags:
            if tag.startswith("evalc:"):
                rest = tag[len("evalc:"):]
                loop = rest.split(":", 1)[0]
                if loop in self.active_loop_ids:
                    obj = "obj:" + rest
                    if obj not in tags:
                        extra.append(obj)
            elif tag.startswith("eval:") and tag not in self.registered:
                self._register(tag, node)
        return value.plus(frozenset(extra)) if extra else value

    def _ev_generic(self, node: ast.AST) -> AbsVal:
        tags: set[str] = set()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                tags |= self.ev(child).tags
        return AbsVal(tags=strip_candref(frozenset(tags)))

    def ex_Constant(self, node: ast.Constant) -> AbsVal:
        value = node.value
        length = len(value) if isinstance(value, (str, bytes)) else None
        return AbsVal(role=CONST, const=value, length=length)

    def ex_Name(self, node: ast.Name) -> AbsVal:
        value = self.lookup(node.id)
        if value is not None:
            return value
        if hasattr(builtins, node.id):
            return AbsVal(role=FUNC, qual=f"builtins.{node.id}")
        return AbsVal(tags=self.dynamic_tags, qual=f"?{node.id}")

    def ex_Attribute(self, node: ast.Attribute) -> AbsVal:
        return self._attribute(self.ev(node.value), node.attr, node)

    def _attribute(self, base: AbsVal, attr: str, node: ast.AST) -> AbsVal:
        if base.role == MODULE and base.qual:
            qual = f"{base.qual}.{attr}"
            if qual in NUMPY_DTYPE_CTORS:
                return AbsVal(role=CONST, qual=qual)  # a dtype; calling it is a cast (A2, 2026-11-04)
            if qual in MODULE_CONSTS:
                return AbsVal(role=CONST)
            return AbsVal(role=MODULE, qual=qual)
        if base.role == FUNC and isinstance(base.fn, ClassInfo):
            found = base.fn.lookup(attr)
            if isinstance(found, FuncInfo):
                return AbsVal(role=FUNC, fn=found)
            return found if isinstance(found, AbsVal) else AbsVal()
        if base.role == INSTANCE:
            if base.items and attr in base.items:
                return base.items[attr]
            if isinstance(base.fn, ClassInfo):
                found = base.fn.lookup(attr)
                if isinstance(found, FuncInfo):
                    return AbsVal(role=FUNC, fn=found, recv=base)
                if isinstance(found, AbsVal):
                    return found
            return AbsVal(tags=strip_candref(base.tags))
        if attr in ("T", "mT"):
            return replace(base, evaldiff=False)
        if attr in ("shape", "size"):
            return _size_value(base)
        if attr in SHAPE_ATTRS:
            return AbsVal(role=CONST)
        if attr == "files" and base.role in (MODEL_C, EVID_C):
            return AbsVal(role=SEQ, default=AbsVal(role=CONST), recv=base if base.role == EVID_C else None,
                          mode="keys" if base.role == EVID_C else None)
        if attr in ("real", "imag", "flat"):
            return replace(base, evaldiff=False)
        if base.role == PATH and attr in ("name", "stem", "suffix", "parent", "parts"):
            return AbsVal(role=CONST if attr != "parent" else PATH, path=base.path)
        return AbsVal(role=METHOD, qual=attr, recv=base, tags=base.tags)

    def ex_Subscript(self, node: ast.Subscript) -> AbsVal:
        base = self.ev(node.value)
        index = self.ev(node.slice)
        return self._subscript(base, index, node, node.slice)

    def _model_item(self, base: AbsVal, key: object) -> AbsVal:
        obj = self._stable_obj("model", base.obj, key if isinstance(key, str) else "*")
        return AbsVal(tags=base.tags | {PARAM}, role=WEIGHT, pristine=True, obj=obj,
                      qual=key if isinstance(key, str) else None)

    def _evidence_item(self, base: AbsVal, key: object) -> AbsVal:
        if not isinstance(key, str) or key in self.label_keys or key.startswith("y"):
            # An evidence key the audit cannot resolve may be a label array: it is treated as one.
            tags = {LABEL, EVID}
            self.label_cells.add(self.frame.cell)
        elif key.startswith(("x", "pred")):
            tags = {EVID}
        else:
            tags = set()
        if isinstance(key, str) and key.endswith("_fail"):
            tags = tags | {FAILSET}
        obj = self._stable_obj("evidence", base.obj, key if isinstance(key, str) else "*")
        return AbsVal(tags=frozenset(tags) | strip_candref(base.tags), role=SAMPLE, obj=obj)

    def _subscript(self, base: AbsVal, index: AbsVal, node: ast.AST, slice_node: ast.expr) -> AbsVal:
        key = index.const if index.has_const() else NOCONST
        role = base.role
        if role == MODEL_C:
            if base.items and isinstance(key, str) and key in base.items:
                return base.items[key]
            return self._model_item(base, key)
        if role == EVID_C:
            if base.items and isinstance(key, str) and key in base.items:
                item = base.items[key]
                if is_tainted(item.tags):
                    self.label_cells.add(self.frame.cell)
                return item
            return self._evidence_item(base, key)
        if role == GLOBALS:
            if isinstance(key, str):
                found = self.module.vars.get(key)
                if found is not None:
                    return found
            return AbsVal(tags=self.dynamic_tags)
        if role in (PDICT, DICT, INSTANCE):
            if isinstance(key, str) and base.items is not None and key in base.items:
                found = base.items[key]
            else:
                pool = list((base.items or {}).values())
                if base.default is not None:
                    pool.append(base.default)
                found = join_all(pool) if pool else AbsVal(tags=strip_candref(base.tags))
            if role == PDICT and found.role == WEIGHT and base.cands and found.cands != base.cands:
                # Evaluating any tensor of a candidate parameter dict evaluates that whole candidate setting.
                found = replace(found, cands=base.cands)
            self._select_by_index(base, index)
            if not index.tags:
                return found
            selected = found.plus(strip_candref(index.tags))
            hyper = (role == DICT and not found.cands and not alternatives(found)
                     and found.role in (CONST, STAT, OTHER) and not any(t.startswith("eval:") for t in found.tags))
            if (role == PDICT and found.role == WEIGHT) or hyper:
                # A loop-varying key selects another tensor of the same setting each iteration (a forward pass written
                # as a loop over layers): the setting does not vary with that loop unless the dict itself does. A2
                # (2026-11-04): the same holds for a dict of per-layer constants (bias_scale[name] in that loop).
                keyed = {t[3:] for t in index.tags if t.startswith("lv:")} - {
                    t[3:] for t in self._deep_tags(base) if t.startswith("lv:")}
                if keyed:
                    selected = replace(selected, keyvar=selected.keyvar | frozenset(keyed))
            return selected
        if role == SEQ:
            if base.elems is not None and isinstance(key, int) and not isinstance(key, bool) and -len(base.elems) <= key < len(base.elems):
                found = base.elems[key]
            elif isinstance(slice_node, ast.Slice):
                elems = None
                if base.elems is not None and all(
                    p is None or _const_int(self.ev(p)) is not None
                    for p in (slice_node.lower, slice_node.upper, slice_node.step)
                ):
                    lower = _const_int(self.ev(slice_node.lower)) if slice_node.lower is not None else None
                    upper = _const_int(self.ev(slice_node.upper)) if slice_node.upper is not None else None
                    step = _const_int(self.ev(slice_node.step)) if slice_node.step is not None else None
                    if step != 0:
                        elems = base.elems[lower:upper:step]
                return replace(base, elems=elems, length=len(elems) if elems is not None else None, multi=True,
                               obj=self._new_obj(), tags=base.tags | strip_candref(index.tags))
            else:
                found = self._element_of(base)
            self._select_by_index(base, index)
            return found.plus(strip_candref(index.tags)) if index.tags else found
        if role in (FUNC, MODULE, METHOD, FILE, PATH, RNG):
            return AbsVal(tags=strip_candref(base.tags | index.tags))
        index_tags = index.tags
        if (role == WEIGHT and base.cands) or (role != SAMPLE and _readout_tags(base.tags)):
            index_tags = self._key_read_variance(base, index_tags, slice_node)
        elif role == WEIGHT or base.basis:
            index_tags = self._index_variance(index_tags, self._is_single_index(slice_node))
        elif role == SAMPLE and EVID in base.tags:
            index_tags = self._row_variance(index_tags)
        index_tags = _drop_decision(index_tags)
        if HARD in index_tags and not base.basis:
            # Selecting entries by an index keeps the selected values; only np.eye(k)[pred] builds hard predictions.
            index_tags = index_tags - {HARD}
        if is_tainted(index_tags) and any(t.startswith(("solver:", "inv:")) for t in base.tags):
            self._meet((base, index), node)
        tags = base.tags | strip_candref(index_tags)
        if LABEL in index_tags or LSIZE in index_tags:
            tags = tags | {LSIZE}  # x[y == c], x[np.flatnonzero(...)]: how many entries are selected depends on labels
        new_role = role
        if index.role == SAMPLE and role != SAMPLE:
            new_role = SAMPLE
        if role == CONST:
            new_role = CONST if not tags else STAT
        multi = isinstance(slice_node, ast.Slice) or base.multi or index.multi
        return AbsVal(tags=tags, role=new_role, pristine=base.pristine if role == WEIGHT else False,
                      cands=base.cands, obj=base.obj if role == WEIGHT else self._new_obj(), multi=multi)

    def _select_by_index(self, container: AbsVal, index: AbsVal) -> None:
        if is_tainted(index.tags) and any(t.startswith("eval:") for t in index.tags):
            self._best_keeping(container, index.tags)

    def ex_BinOp(self, node: ast.BinOp) -> AbsVal:
        if isinstance(node.op, (ast.BitAnd, ast.BitOr)) and all(
                isinstance(side, ast.Compare) and any(isinstance(op, ORDER_COMPARE) for op in side.ops)
                for side in (node.left, node.right)):
            # A2 (2026-11-04): (s >= lo) & (s < hi) over the bins of a loop is a stratification, not a decision rule.
            self.bin_depth += 1
            try:
                left, right = self.ev(node.left), self.ev(node.right)
            finally:
                self.bin_depth -= 1
            return self.combine(node.op, left, right, node)
        return self.combine(node.op, self.ev(node.left), self.ev(node.right), node)

    def combine(self, op: ast.operator, left: AbsVal, right: AbsVal, node: ast.AST) -> AbsVal:
        """Abstract semantics of a binary arithmetic operation."""
        if isinstance(op, ast.MatMult):
            return self._matmul(left, right, node)
        self._meet((left, right), node)
        if left.role in (MODULE, FUNC) and left.qual in MODULE_CONSTS:
            left = AbsVal(role=CONST)
        if left.role == CONST and right.role == CONST:
            const = _fold(op, left.const, right.const) if left.has_const() and right.has_const() else NOCONST
            length = len(const) if isinstance(const, str) else None
            return AbsVal(tags=left.tags | right.tags, role=CONST, const=const, length=length,
                          path=const if isinstance(const, str) else None)
        if (left.role == PATH or right.role == PATH) and isinstance(op, ast.Div):
            return self._path_join([left, right])
        if left.role == SEQ or right.role == SEQ:
            return self._seq_op(op, left, right)
        if left.role in (DICT, PDICT) and right.role in (DICT, PDICT) and isinstance(op, ast.BitOr):
            items = dict(left.items or {})
            items.update(right.items or {})
            return self._refresh_container(AbsVal(role=PDICT if PDICT in (left.role, right.role) else DICT,
                                                  items=items, default=join(left.default, right.default),
                                                  obj=self._new_obj()))
        tags = left.tags | right.tags
        extra: set[str] = set()
        cands: frozenset[str] = frozenset()
        pristine = False
        lw, rw = left.role == WEIGHT, right.role == WEIGHT
        ls, rs = left.role == SAMPLE, right.role == SAMPLE
        if (lw and rs) or (rw and ls):
            weight, data = (left, right) if lw else (right, left)
            add, remove = self._evaluation(weight, data)
            tags = (strip_candref(tags) - remove) | add
            role = SAMPLE
        elif lw or rw:
            if lw and rw and left.pristine and right.pristine:
                role = STAT
                tags = strip_candref(tags)
            else:
                weight, other = (left, right) if lw else (right, left)
                if lw and rw and PARAM not in left.tags and PARAM in right.tags:
                    weight, other = right, left
                if PARAM in weight.tags:
                    fresh = _grad_tags(other.tags) - _grad_tags(weight.tags)
                    if fresh:
                        self._update_sink(fresh, node, "a parameter array is computed")
                    parents = weight.cands | (other.cands if other.role == WEIGHT else frozenset())
                    cands = self._new_cands(node, parents)
                    tags = self._as_candidate(strip_candref(tags))
                else:
                    cands = left.cands | right.cands
                    tags = strip_candref(tags)
                role = WEIGHT
        elif ls or rs:
            role = SAMPLE
            tags = strip_candref(tags)
            data, other = (left, right) if ls else (right, left)
            if (isinstance(op, (ast.Sub, ast.Mult, ast.Add, ast.Div)) and other.role != SAMPLE
                    and EVID in data.tags and _readout_tags(other.tags)):
                # A2 (2026-11-04): F * proto (summed: a dot product), F - proto (a distance), F + shift and F / scale
                # (an activation-space correction fitted on evidence statistics) evaluate the readout.
                tags = self._readout_evaluation(other, data, tags)
            if isinstance(op, DECISION_ARITH) and other.role != SAMPLE and PARAM in data.tags:
                tags = self._decision_variance(tags, [other])
            if isinstance(op, (ast.Add, ast.Sub)) and LABEL not in left.tags and LABEL not in right.tags:
                # A carried quantity added to model outputs before any label comparison (a bias or an activation
                # offset) is evaluated through the model. Adding a carried total to label-dependent scores is an
                # accumulator, not an evaluation of the total.
                model, other = (left, right) if (ls and self._model_like(left)) else (right, left)
                if self._model_like(model):
                    extra |= {"evalc:" + t[len("carry:"):] for t in other.tags if t.startswith("carry:")}
        else:
            role = STAT
            tags = strip_candref(tags)
        if isinstance(op, ast.Sub) and self._is_error_signal(left, right):
            site = self._site(node, "err")
            self._register_site(site, "err", node)
            extra.add(f"err:{site}")
        evaldiff = (
            isinstance(op, ast.Sub)
            and is_tainted(left.tags)
            and is_tainted(right.tags)
            and bool(_parameter_settings(left.tags | right.tags))
        )
        # A2 (2026-11-04): a side evaluated only through feature-space readouts (a prototype, a distance, the shape
        # of a weight built from readouts) is a feature, not a score, so it makes no score difference either way.
        readout_only = any(_evaluated_settings(v.tags) and not _parameter_settings(v.tags) for v in (left, right))
        if evaldiff and not readout_only and _parameter_settings(left.tags) != _parameter_settings(right.tags):
            # Scores of two different settings subtracted: a finite difference. Comparing candidates through it is
            # allowed; it becomes V-GRAD-FINITE-DIFF only when it reaches a parameter value or a patch, and an unused
            # one raises nothing (score gains are routinely printed).
            site = self._site(node, "sd")
            self._register_site(site, "sd", node)
            extra.add(f"sd:{site}")
        if isinstance(op, ast.Div) and left.evaldiff:
            step = _const_number(right)
            if step is not None and 0 < abs(step) <= SMALL_STEP:
                site = self._site(node, "fd")
                self._register_site(site, "fd", node)
                extra.add(f"fd:{site}")
        length = left.length if left.length is not None else right.length
        # A2 (2026-11-04): key-only loop variance survives arithmetic (b * bias_scale[name] in a loop over layers is
        # one setting, as p[name] is); before, it was dropped and the layer loop multiplied the setting.
        return AbsVal(tags=frozenset(tags) | extra, role=role, cands=cands, pristine=pristine, evaldiff=evaldiff,
                      obj=self._new_obj(), length=length if role != CONST else None,
                      keyvar=combine_keyvar((left, right)))

    @staticmethod
    def _model_like(value: AbsVal) -> bool:
        """Per-sample values computed through the model or through a loop-carried quantity applied to evidence."""
        return value.role == SAMPLE and (PARAM in value.tags or any(t.startswith("evalc:") for t in value.tags))

    @staticmethod
    def _is_error_signal(left: AbsVal, right: AbsVal) -> bool:
        """Per-sample model outputs (logits or probabilities) minus a label encoding: d loss / d outputs.

        Reduced statistics are excluded (a difference of class means is descriptive), and so are hard predictions:
        one-hot(argmax) minus one-hot(labels), contracted with activations, is a difference of class sums.
        """
        def model_output(v: AbsVal) -> bool:
            return v.role == SAMPLE and PARAM in v.tags and EVID in v.tags and HARD not in v.tags

        def label_encoding(v: AbsVal) -> bool:
            return LABEL in v.tags and PARAM not in v.tags

        return (model_output(left) and label_encoding(right)) or (model_output(right) and label_encoding(left))

    def _matmul(self, left: AbsVal, right: AbsVal, node: ast.AST) -> AbsVal:
        self._meet((left, right), node)
        self._inverse_check(left, right, node)
        tags = left.tags | right.tags
        extra: set[str] = set()
        remove: frozenset[str] = frozenset()
        lw, rw = left.role == WEIGHT, right.role == WEIGHT
        if lw and rw:
            role = STAT
        elif lw or rw:
            weight, data = (left, right) if lw else (right, left)
            if EVID in data.tags or data.role == SAMPLE:
                add, remove = self._evaluation(weight, data)
                extra |= add
            role = SAMPLE if data.role == SAMPLE else STAT
        elif left.role == SAMPLE and right.role == SAMPLE:
            role = STAT
        elif SAMPLE in (left.role, right.role):
            role = SAMPLE
            data, other = (left, right) if left.role == SAMPLE else (right, left)
            if EVID in data.tags and _readout_tags(other.tags):
                # A2 (2026-11-04): per-sample features against a feature-space readout (F @ protos.T) evaluate it.
                tags = self._readout_evaluation(other, data, frozenset(tags))
            if EVID in data.tags and LABEL not in data.tags and LABEL not in other.tags:
                # A loop-carried array applied to per-sample evidence data (the weights of a probe or of a layer the
                # editor rebuilt) is evaluated on the evidence, like a parameter array.
                extra |= {"evalc:" + t[len("carry:"):] for t in other.tags if t.startswith("carry:")}
        else:
            role = STAT
        if (self._has_err(left) and right.role == SAMPLE) or (self._has_err(right) and left.role == SAMPLE):
            site = self._site(node, "grad")
            self._register_site(site, "grad", node)
            extra.add(f"grad:{site}")
        return AbsVal(tags=(strip_candref(tags) - remove) | extra, role=role, obj=self._new_obj())

    @staticmethod
    def _has_err(value: AbsVal) -> bool:
        return any(t.startswith("err:") for t in value.tags)

    def _inverse_check(self, left: AbsVal, right: AbsVal, node: ast.AST) -> None:
        for inv_side, other in ((left, right), (right, left)):
            for tag in inv_side.tags:
                if not tag.startswith("inv:") or tag in other.tags:
                    continue
                site = tag[len("inv:"):]
                info = self.sites.get(site)
                if info is None or info.flagged == "V":
                    continue
                where = f"cell {info.cell} line {info.line}"
                if LABEL in other.tags:
                    info.flagged = "V"
                    self._flag(V_SOLVER, node, f"the inverse or pseudo-inverse computed at {where} is multiplied with a "
                                               "label-derived operand (a closed-form fit)")
                elif info.flagged is None and LABEL_BUILT in other.tags:
                    info.flagged = "S"
                    self._flag(S_SOLVER_SOFT, node, f"the inverse computed at {where} is multiplied with an operand that "
                                                    "depends on labels only through a label-built candidate setting")
                elif info.flagged is None and info.tainted_input and not is_tainted(other.tags):
                    info.flagged = "S"
                    self._flag(S_SOLVER_SOFT, node, f"the inverse of a label-dependent matrix computed at {where} is "
                                                    "applied to a label-free operand (possibly a fit onto constructed targets)")
                elif info.flagged is None and FAILSET in other.tags and other.role != SAMPLE:
                    info.flagged = "S"
                    self._flag(S_SOLVER_SOFT, node, f"the inverse computed at {where} is multiplied with a statistic or "
                                                    "target built from the x_fail split (membership is label-derived): "
                                                    "possibly a closed-form probe of failure membership")

    def _meet(self, operands: Sequence[AbsVal], node: ast.AST) -> None:
        """Soft flag when the result of a label-free solver or inverse meets label-dependent values."""
        for i, a in enumerate(operands):
            sites = [t for t in a.tags if t.startswith(("solver:", "inv:"))]
            if not sites:
                continue
            for j, b in enumerate(operands):
                if i == j or not is_tainted(b.tags):
                    continue
                for tag in sites:
                    if tag in b.tags:
                        continue
                    site = tag.split(":", 1)[1]
                    info = self.sites.get(site)
                    if info is None or info.flagged is not None or info.tainted_input:
                        continue
                    info.flagged = "S"
                    self._flag(S_SOLVER_INDIRECT, node, f"the result of the {info.kind} at cell {info.cell} line "
                                                        f"{info.line} (label-free inputs) meets label-dependent values here")

    def ex_UnaryOp(self, node: ast.UnaryOp) -> AbsVal:
        value = self.ev(node.operand)
        if isinstance(node.op, ast.Not):
            return AbsVal(tags=strip_candref(value.tags), role=CONST if value.role == CONST else STAT)
        if value.role == CONST and value.has_const() and _is_number(value.const):
            if isinstance(node.op, ast.USub):
                return replace(value, const=-value.const)
            if isinstance(node.op, ast.UAdd):
                return value
            return replace(value, const=NOCONST)
        if value.role == WEIGHT and PARAM in value.tags and isinstance(node.op, ast.USub):
            return replace(value, cands=self._new_cands(node, value.cands), pristine=False, obj=self._new_obj(),
                           tags=self._as_candidate(strip_candref(value.tags)), evaldiff=False)
        return replace(value, obj=self._new_obj() if value.obj is not None else None, evaldiff=False,
                       pristine=value.pristine and value.role != WEIGHT)

    def ex_BoolOp(self, node: ast.BoolOp) -> AbsVal:
        values: list[AbsVal] = []
        for operand in node.values:
            value = self.ev(operand)
            values.append(value)
            truth = _known_truth(value)
            if truth is not None and truth == isinstance(node.op, ast.Or):
                # A2 (2026-11-04): short-circuit on an exactly known constant (``bias_scale and ...`` with
                # bias_scale=None is None); the operands after it are not evaluated, as in Python.
                return join_all(values)
        return join_all(values)

    def ex_Compare(self, node: ast.Compare) -> AbsVal:
        values = [self.ev(node.left)] + [self.ev(c) for c in node.comparators]
        self._meet(values, node)
        tags = _mask_tags(frozenset().union(*(strip_candref(v.tags) for v in values)))
        if sum(isinstance(op, ORDER_COMPARE) for op in node.ops) == 1 and not self.bin_depth:
            # A2 (2026-11-04): a threshold swept by a loop; a chained lo <= s < hi is a bin (a stratification).
            tags = self._decision_variance(tags, values)
        if any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
            tags = tags | self._class_view(values)
        if any(v.role == SAMPLE for v in values):
            role = SAMPLE
        elif all(v.role == CONST for v in values):
            role = CONST
        else:
            role = STAT
        return AbsVal(tags=tags, role=role, obj=self._new_obj())

    def ex_IfExp(self, node: ast.IfExp) -> AbsVal:
        test = self.ev(node.test)
        self.gates.append(test.tags)
        try:
            body = self.ev(node.body)
            orelse = self.ev(node.orelse)
        finally:
            self.gates.pop()
        result = join(body, orelse)
        self._best_keeping(result, test.tags)
        # A2 (2026-11-04): a thresholded test chooses which value is computed (``acc[m].mean() if m.sum() else None``
        # per confidence bin); the threshold does not make the chosen value another setting.
        extra = _drop_decision(strip_candref(test.tags))
        if result.role == WEIGHT:
            return replace(result, tags=self._as_candidate(result.tags | extra))
        return result.plus(extra)

    def ex_Lambda(self, node: ast.Lambda) -> AbsVal:
        defaults = [self.ev(default) for default in node.args.defaults]
        info = FuncInfo(node, "<lambda>", None if self.frame.scope.is_module else self.frame.scope, self.frame.cell,
                        self.frame.lines, defaults, {})
        return AbsVal(role=FUNC, fn=info)

    def ex_Display(self, node: ast.List | ast.Tuple | ast.Set) -> AbsVal:
        elems: list[AbsVal] = []
        known = True
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                starred = self.ev(elt.value)
                if starred.elems is not None:
                    elems.extend(starred.elems)
                else:
                    known = False
                    elems.append(self._element_of(starred))
            else:
                elems.append(self.ev(elt))
        tags: set[str] = set()
        for elem in elems:
            tags |= strip_candref(elem.tags) | candref_tags(frozenset(alternatives(elem)))
        const = NOCONST
        if isinstance(node, ast.Tuple) and known and all(e.role == CONST and e.has_const() for e in elems):
            const = tuple(e.const for e in elems)
        return AbsVal(role=SEQ, tags=frozenset(tags), elems=tuple(elems) if known else None,
                      length=len(elems) if known else None, default=join_all(elems) if elems else None,
                      obj=self._new_obj(), const=const, multi=len(elems) != 1)

    def ex_Dict(self, node: ast.Dict) -> AbsVal:
        items: dict[str, AbsVal] = {}
        default: AbsVal | None = None
        parameter_base = False
        evidence_base = False
        exact = True
        extra: set[str] = set()
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                spread = self.ev(value_node)
                if spread.role == EVID_C:
                    spread = self._evidence_copy(spread)
                    evidence_base = True
                exact = False
                if spread.role in (MODEL_C, PDICT):
                    parameter_base = True
                if spread.items:
                    items.update(spread.items)
                if spread.role == MODEL_C:
                    default = join(default, self._model_item(spread, NOCONST))
                elif spread.default is not None:
                    default = join(default, spread.default)
                extra |= strip_candref(spread.tags)
                continue
            key = self.ev(key_node)
            value = self.ev(value_node)
            if isinstance(key.const, str):
                items[key.const] = value
            else:
                default = join(default, value)
                exact = False
                extra |= strip_candref(key.tags)
        if evidence_base and default is None:
            return AbsVal(role=EVID_C, items=items, tags=frozenset(extra), obj=self._new_obj())
        is_param = parameter_base or any(v.role == WEIGHT and PARAM in v.tags for v in items.values())
        container = AbsVal(role=PDICT if is_param else DICT, items=items, default=default, tags=frozenset(extra),
                           length=len(items) if exact else None, obj=self._new_obj())
        return self._refresh_container(container)

    def _refresh_container(self, value: AbsVal) -> AbsVal:
        """Recompute a dict-like container's tags, pristine flag and candidate alternatives from its entries."""
        entries = list((value.items or {}).values())
        if value.default is not None:
            entries.append(value.default)
        tags = set(strip_candref(value.tags))
        for entry in entries:
            tags |= strip_candref(entry.tags) | candref_tags(frozenset(alternatives(entry)))
        if value.role == PDICT:
            pristine = all(entry.pristine or entry.role != WEIGHT for entry in entries)
            cands = self._combine_alternatives(entry.cands for entry in entries if entry.role == WEIGHT)
            return replace(value, tags=frozenset(tags), pristine=pristine, cands=cands)
        return replace(value, tags=frozenset(tags))

    def ex_JoinedStr(self, node: ast.JoinedStr) -> AbsVal:
        parts: list[str] = []
        constant = True
        inners: list[tuple[frozenset[str], str]] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
                continue
            inner = self.ev(value)
            inners.append((inner.tags, inner.role))
            if inner.has_const() and isinstance(inner.const, (str, int, float)) and not isinstance(inner.const, bool):
                parts.append(str(inner.const))
            elif inner.path is not None:
                parts.append(inner.path)
            else:
                constant = False
                parts.append("*")
        text = "".join(parts)
        tags = self._text_tags(inners)
        return AbsVal(role=CONST if not tags else STAT, tags=tags, const=text if constant else NOCONST, path=text)

    @staticmethod
    def _text_tags(parts: Sequence[tuple[frozenset[str], str]]) -> frozenset[str]:
        """Tags of a text formatted from values with the given (tags, role).

        A text presents values computed elsewhere; each formatted value was already analysed on its own, and a label
        comparison inside one of them is registered there. A label score formatted next to a label-free evaluation
        (a candidate's name quoting its agreement with a teacher) therefore does not score the second setting: the
        text keeps the evaluations only of label-dependent parts, and of per-sample model outputs when per-sample
        labels are formatted beside them (the reader can compare those).
        """
        labelled_samples = any(LABEL in tags and role == SAMPLE for tags, role in parts)
        out: set[str] = set()
        for tags, role in parts:
            tags = _no_grad(strip_candref(tags))
            if LABEL not in tags and not (labelled_samples and role == SAMPLE):
                tags = frozenset(t for t in tags if not t.startswith("eval:"))
            out |= tags
        return frozenset(out)

    def ex_FormattedValue(self, node: ast.FormattedValue) -> AbsVal:
        value = self.ev(node.value)
        if node.format_spec is not None:
            spec = self.ev(node.format_spec)
            return AbsVal(tags=value.tags | spec.tags, role=value.role)
        return value

    def ex_NamedExpr(self, node: ast.NamedExpr) -> AbsVal:
        value = self.ev(node.value)
        self.assign(node.target, value, node)
        return value

    def ex_Starred(self, node: ast.Starred) -> AbsVal:
        return self.ev(node.value)

    def ex_Slice(self, node: ast.Slice) -> AbsVal:
        tags: set[str] = set()
        for part in (node.lower, node.upper, node.step):
            if part is not None:
                tags |= self.ev(part).tags
        return AbsVal(role=CONST, tags=strip_candref(frozenset(tags)), multi=True)

    def ex_Yield(self, node: ast.AST) -> AbsVal:
        inner = getattr(node, "value", None)
        value = self.ev(inner) if inner is not None else AbsVal(role=CONST, const=None)
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            self.frame.yields.append(value if isinstance(node, ast.Yield) else self._element_of(value))
        return AbsVal(tags=value.tags)

    # ------------------------------------------------------------- iteration

    def _element_of(self, value: AbsVal) -> AbsVal:
        role = value.role
        if role == SEQ:
            if value.elems:
                return join_all(list(value.elems))
            if value.default is not None:
                return value.default
            return AbsVal(tags=strip_candref(value.tags))
        if role in (DICT, PDICT, MODEL_C, EVID_C, GLOBALS):
            return AbsVal(role=CONST)
        if role in (WEIGHT, SAMPLE, STAT, OTHER):
            return replace(value, obj=self._new_obj() if role != WEIGHT else value.obj, length=None, multi=False)
        if role == CONST:
            return AbsVal(role=CONST, tags=value.tags)
        return AbsVal(tags=strip_candref(value.tags))

    def _push_loop(self, lid: str, trip: int | None, kind: str, node: ast.AST) -> LoopCtx:
        line = getattr(node, "lineno", 1) or 1
        self.loop_trips[lid] = trip
        self.loop_meta[lid] = (self.frame.cell, line, self._excerpt(node), kind)
        ctx = LoopCtx(lid, trip, kind, self.frame.cell, line, frame=self.frame)
        self.loops.append(ctx)
        self.active_loop_ids.add(lid)
        return ctx

    def _pop_loop(self, ctx: LoopCtx) -> None:
        popped = self.loops.pop()
        assert popped is ctx
        if all(other.lid != ctx.lid for other in self.loops):
            self.active_loop_ids.discard(ctx.lid)

    def _loop_extra(self, lid: str, iterable: AbsVal) -> frozenset[str]:
        extra = {f"lv:{lid}"}
        if iterable.elems is not None and any(alternatives(elem) for elem in iterable.elems):
            extra.add(f"enum:{lid}")
        return frozenset(extra)

    def _loop_element(self, iterable: AbsVal, lid: str) -> AbsVal:
        """The value a loop target takes: an element of ``iterable`` marked as varying with loop ``lid``.

        The mark reaches the components of the element as well, so unpacking ``for key, value in d.items()`` keeps
        the variance on ``value``.
        """
        extra = self._loop_extra(lid, iterable)
        if f"enum:{lid}" in extra and iterable.elems is not None:
            iterable = replace(iterable, elems=tuple(self._collapse_alternatives(e, 0) for e in iterable.elems))

        def mark(value: AbsVal, depth: int) -> AbsVal:
            value = value.plus(extra)
            if depth >= LOOP_MARK_DEPTH:
                return value
            changes: dict[str, object] = {}
            if value.elems is not None:
                changes["elems"] = tuple(mark(item, depth + 1) for item in value.elems)
            if value.items is not None:
                changes["items"] = {key: mark(item, depth + 1) for key, item in value.items.items()}
            if value.default is not None:
                changes["default"] = mark(value.default, depth + 1)
            return replace(value, **changes) if changes else value  # type: ignore[arg-type]

        return mark(self._element_of(iterable), 0)

    def _collapse_alternatives(self, value: AbsVal, depth: int) -> AbsVal:
        """An element of an enumerated loop is one setting: its alternative histories are named by their union.

        Alternatives of one setting come from control-flow merges where it was built (the branches of an op-applying
        helper); the loop enumerates settings, not those histories. The union is recorded as joined, so it is not
        counted again where each of its alternatives was scored on its own.
        """
        changes: dict[str, object] = {}
        if value.role in (PDICT, WEIGHT) and len(value.cands) > 1:
            union = _union_key(value.cands)
            self.joined.setdefault(union, set()).update(value.cands)
            changes["cands"] = frozenset({union})
        if value.elems is not None and depth < LOOP_MARK_DEPTH:
            elems = tuple(self._collapse_alternatives(e, depth + 1) for e in value.elems)
            refs: set[str] = set()
            for element in elems:
                refs |= candref_tags(frozenset(alternatives(element)))
            changes["elems"] = elems
            changes["tags"] = strip_candref(value.tags) | refs
        return replace(value, **changes) if changes else value  # type: ignore[arg-type]

    def st_For(self, node: ast.For) -> None:
        iterable = self.ev(node.iter)
        lid = self._loop_id(node)
        element = self._loop_element(iterable, lid)
        carried = self._assigned_names(node.body) - self._target_names(node.target)

        def bind() -> None:
            self.assign(node.target, element, node, record=False)

        entry = self._run_loop(node, lid, iterable.length, "for", carried, bind, None, node.body)
        self.exec_block(node.orelse)
        self._post_loop_lengths(node, iterable.length, entry)

    st_AsyncFor = st_For

    def st_While(self, node: ast.While) -> None:
        lid = self._loop_id(node)
        carried = self._assigned_names(node.body)
        self._run_loop(node, lid, None, "while", carried, lambda: None, node.test, node.body)
        self.exec_block(node.orelse)

    def _run_loop(self, node: ast.AST, lid: str, trip: int | None, kind: str, carried: set[str],
                  bind: Callable[[], None], test: ast.expr | None, body: list[ast.stmt]
                  ) -> list[tuple[Scope, dict[str, AbsVal]]]:
        ctx = self._push_loop(lid, trip, kind, node)
        entry = self._snapshot()
        last = entry
        previous_signature: tuple[object, ...] | None = None
        current_carried = set(carried)
        try:
            for pass_no in range(MAX_PASSES):
                if pass_no:
                    self._restore(self._join_snap(entry, last))
                self._tag_carried(current_carried, ctx, variant=pass_no > 0)
                bind()
                gate: frozenset[str] = frozenset()
                if test is not None:
                    tested = self.ev(test)
                    gate = tested.tags
                    ctx.exit_gates |= tested.tags
                self.gates.append(gate)
                try:
                    self.exec_block(body)
                finally:
                    self.gates.pop()
                last = self._snapshot()
                current_signature = self._snap_signature(last)
                grown = current_carried | self._changed_names(entry, last)
                if current_signature == previous_signature and grown == current_carried:
                    break
                previous_signature = current_signature
                current_carried = grown
        finally:
            self._pop_loop(ctx)
        self._restore(self._join_snap(entry, last))
        self._settle_iteration_evals(ctx)
        self._strip_loop_tags(lid)
        self._check_iterative(ctx)
        return entry

    def _settle_iteration_evals(self, ctx: LoopCtx) -> None:
        """After loop ``ctx``: a value holding one iteration's result names one setting, not one per iteration.

        A name the loop only rebinds (never mutated in place, never bound to a value computed from its own previous
        value) holds one iteration's result after the loop (the last, or one picked by a condition), so the
        evaluations it carries lose this loop's multiplicity. Per-iteration results that escape the loop separately
        (appended, stored by key or index, accumulated) stay in such a collection and keep it, as does anything
        computed from a loop-carried collection of model-derived values (outputs, scores or candidate settings). A
        loop-carried parameter setting (a patch composed op by op) is one setting, not a collection; a carried counter
        or a log of names holds nothing model-derived.
        """
        lid = ctx.lid
        carry = f"carry:{lid}:"
        snapshot = self._snapshot()
        collections: set[str] = set()
        held_tags: set[str] = set()  # evaluations a collection holds
        held_sites: set[str] = set()  # modification sites of the settings a collection holds
        for scope, values in snapshot:
            for name, value in values.items():
                if (id(scope), name) not in ctx.kept or value.role in (WEIGHT, PDICT, MODEL_C, FUNC, MODULE):
                    continue
                deep = self._deep_tags(value)
                if PARAM not in deep and not _loop_evals(deep, lid):
                    continue  # a counter or a log of names
                collections.add(name)
                held_tags.update(t for t in deep if t.startswith("eval:"))
                for alt in self._deep_alternatives(value, 0):
                    held_sites.update(p for p in alt.split("+") if p)

        def free(tag: str) -> bool:
            """An evaluation tag of this loop that no collection holds, by tag or by the setting it names."""
            if not tag.startswith("eval:") or tag in held_tags or lid not in _sig_loops(_split_eval(tag)[1]):
                return False
            return not any(p in held_sites for p in _split_eval(tag)[0].split("+") if p)

        for scope, values in snapshot:
            for name, value in values.items():
                key = (id(scope), name)
                if key not in ctx.rebound or key in ctx.kept:
                    continue
                deep = self._deep_tags(value)
                if not any(free(t) for t in deep):
                    continue
                if any(t.startswith(carry) and t[len(carry):] in collections for t in deep):
                    continue
                scope.vars[name] = self._drop_loop_component(value, lid, free, 0)

    def _deep_alternatives(self, value: AbsVal, depth: int) -> set[str]:
        found = alternatives(value)
        if depth < 3:
            for item in [*(value.items or {}).values(), *(value.elems or ()),
                         *([value.default] if value.default is not None else [])]:
                found |= self._deep_alternatives(item, depth + 1)
        return found

    def _drop_loop_component(self, value: AbsVal, lid: str, free: Callable[[str], bool], depth: int) -> AbsVal:
        changes: dict[str, object] = {}
        if any(free(t) for t in value.tags):
            changes["tags"] = frozenset(_without_loop(t, lid) if free(t) else t for t in value.tags)
        if depth < 3:
            if value.items:
                items = {k: self._drop_loop_component(v, lid, free, depth + 1) for k, v in value.items.items()}
                if any(items[k] is not v for k, v in value.items.items()):
                    changes["items"] = items
            if value.elems:
                elems = tuple(self._drop_loop_component(v, lid, free, depth + 1) for v in value.elems)
                if any(a is not b for a, b in zip(elems, value.elems)):
                    changes["elems"] = elems
            if value.default is not None:
                default = self._drop_loop_component(value.default, lid, free, depth + 1)
                if default is not value.default:
                    changes["default"] = default
        return replace(value, **changes) if changes else value  # type: ignore[arg-type]

    def _tag_carried(self, names: set[str], ctx: LoopCtx, variant: bool) -> None:
        for name in names:
            scope = self._scope_of(name)
            if scope is None:
                continue
            value = scope.vars[name]
            if value.role in (FUNC, MODULE):
                continue
            extra = {f"carry:{ctx.lid}:{name}"}
            if variant:
                kind = ctx.var_kinds.get(name, "v")
                extra.add({"v": f"lv:{ctx.lid}", "m": f"lim:{ctx.lid}", "i": f"li:{ctx.lid}"}[kind])
            scope.vars[name] = value.plus(frozenset(extra))

    def _strip_loop_tags(self, lid: str) -> None:
        """Remove the tags of loop ``lid`` from every live binding, including values held inside containers."""
        exact = {f"lv:{lid}", f"li:{lid}", f"lim:{lid}", f"enum:{lid}", f"lf:{lid}", f"lk:{lid}", f"lc:{lid}"}
        prefixes = (f"carry:{lid}:", f"evalc:{lid}:", f"obj:{lid}:")

        def has(value: AbsVal) -> bool:
            return any(t in exact or t.startswith(prefixes) for t in value.tags)

        def clean(value: AbsVal, depth: int) -> AbsVal:
            changes: dict[str, object] = {}
            if has(value):
                changes["tags"] = frozenset(t for t in value.tags if t not in exact and not t.startswith(prefixes))
            if depth < 3:
                if value.items:
                    items = {k: clean(v, depth + 1) for k, v in value.items.items()}
                    if any(items[k] is not v for k, v in value.items.items()):
                        changes["items"] = items
                if value.elems:
                    elems = tuple(clean(v, depth + 1) for v in value.elems)
                    if any(a is not b for a, b in zip(elems, value.elems)):
                        changes["elems"] = elems
                if value.default is not None:
                    default = clean(value.default, depth + 1)
                    if default is not value.default:
                        changes["default"] = default
            return replace(value, **changes) if changes else value  # type: ignore[arg-type]

        for scope, values in self._snapshot():
            for name, value in values.items():
                cleaned = clean(value, 0)
                if cleaned is not value:
                    scope.vars[name] = cleaned

    def _check_iterative(self, ctx: LoopCtx) -> None:
        loop_gates = frozenset(ctx.exit_gates)
        for event in ctx.events.values():
            via_value = [n for n in event.names if f"obj:{ctx.lid}:{n}" in event.value_tags]
            via_gate = [n for n in event.names if f"obj:{ctx.lid}:{n}" in (event.gate_tags | loop_gates)]
            hits = sorted(set(via_value) | set(via_gate))
            if not hits:
                continue
            how = "its new value" if via_value else "the condition that gates or ends the update"
            reason = (f"the {ctx.kind} loop at cell {ctx.cell} line {ctx.line} updates {', '.join(hits)}; {how} depends "
                      f"on a label-dependent evaluation of {', '.join(hits)} itself (iterative optimisation)")
            self._flag_at(V_ITERATIVE, event.cell, event.line, event.excerpt, reason)

    def _post_loop_lengths(self, node: ast.For, trip: int | None, entry: list[tuple[Scope, dict[str, AbsVal]]]) -> None:
        if trip is None:
            return
        counts = self._append_counts(node.body, 1)
        if counts is None:
            return
        before = {}
        for _, values in entry:
            before.update(values)
        for name, count in counts.items():
            current = self.lookup(name)
            start = before.get(name)
            if current is None or start is None or start.length is None or count is None:
                continue
            if current.role not in (SEQ, DICT, PDICT):
                continue
            scope = self._scope_of(name)
            if scope is not None:
                scope.vars[name] = replace(current, length=start.length + trip * count)

    def _append_counts(self, body: list[ast.stmt], multiplier: int) -> dict[str, int | None] | None:
        """Containers grown by appends at the top of a loop body (possibly in nested loops with known bounds)."""
        counts: dict[str, int | None] = {}
        for stmt in body:
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.Break, ast.Continue)):
                    return None
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                    name = call.func.value.id
                    if call.func.attr in ("append", "add"):
                        counts[name] = None if counts.get(name, 0) is None else (counts.get(name) or 0) + multiplier
                    elif call.func.attr == "extend":
                        arg = call.args[0] if call.args else None
                        size = len(arg.elts) if isinstance(arg, (ast.List, ast.Tuple)) else None
                        if size is None:
                            counts[name] = None
                        else:
                            counts[name] = None if counts.get(name, 0) is None else (counts.get(name) or 0) + size * multiplier
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Subscript):
                target = stmt.targets[0]
                if isinstance(target.value, ast.Name) and not isinstance(target.slice, ast.Constant):
                    name = target.value.id
                    counts[name] = None if counts.get(name, 0) is None else (counts.get(name) or 0) + multiplier
            elif isinstance(stmt, ast.For):
                inner_iter = self._static_length(stmt.iter)
                inner = self._append_counts(stmt.body, multiplier * inner_iter) if inner_iter is not None else None
                names = self._mutated_containers(stmt.body)
                for name in names:
                    if inner is None or name not in inner:
                        counts[name] = None
                    else:
                        add = inner[name]
                        counts[name] = None if add is None or counts.get(name, 0) is None else (counts.get(name) or 0) + add
            else:
                for name in self._mutated_containers([stmt]):
                    counts[name] = None
        return counts

    def _static_length(self, node: ast.expr) -> int | None:
        if isinstance(node, (ast.List, ast.Tuple)) and not any(isinstance(e, ast.Starred) for e in node.elts):
            return len(node.elts)
        if isinstance(node, ast.Name):
            value = self.lookup(node.id)
            return value.length if value is not None else None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            args = [_const_int(self._peek_const(a)) for a in node.args]
            if args and all(a is not None for a in args):
                return len(range(*args))  # type: ignore[arg-type]
        return None

    def _peek_const(self, node: ast.expr) -> AbsVal | None:
        if isinstance(node, ast.Constant):
            return AbsVal(role=CONST, const=node.value)
        if isinstance(node, ast.Name):
            return self.lookup(node.id)
        return None

    @staticmethod
    def _mutated_containers(body: list[ast.stmt]) -> set[str]:
        names: set[str] = set()
        for stmt in body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and isinstance(sub.func.value, ast.Name):
                    if sub.func.attr in ("append", "add", "extend", "insert", "update", "setdefault"):
                        names.add(sub.func.value.id)
                elif isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                            names.add(target.value.id)
        return names

    @staticmethod
    def _target_names(target: ast.expr) -> set[str]:
        return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}

    @staticmethod
    def _assigned_names(body: list[ast.stmt]) -> set[str]:
        """Names a loop body binds or mutates (syntactically), excluding nested function and class bodies."""
        names: set[str] = set()

        def root(node: ast.expr) -> str | None:
            while isinstance(node, (ast.Subscript, ast.Attribute)):
                node = node.value
            return node.id if isinstance(node, ast.Name) else None

        def visit(node: ast.AST) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                if not isinstance(node, ast.Lambda):
                    names.add(node.name)
                return
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for sub in ast.walk(target):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
                    name = root(target)
                    if name:
                        names.add(name)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                names.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
            elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        names.update(n.id for n in ast.walk(item.optional_vars) if isinstance(n, ast.Name))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                name = root(node.func.value)
                if name and node.func.attr in ("append", "extend", "insert", "update", "fill", "sort", "pop", "remove",
                                               "clear", "setdefault", "put", "itemset", "resize", "shuffle"):
                    names.add(name)
            for child in ast.iter_child_nodes(node):
                visit(child)

        for stmt in body:
            visit(stmt)
        return names

    def ex_Comp(self, node: ast.AST) -> AbsVal:
        assert isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp))
        saved = self.frame.scope
        comp_scope = Scope(parent=None if saved.is_module else saved, is_module=False)
        pushed: list[LoopCtx] = []
        gates = 0
        trips: list[int | None] = []
        try:
            for i, generator in enumerate(node.generators):
                iterable = self.ev(generator.iter)
                if i == 0 and isinstance(node, ast.DictComp) and len(node.generators) == 1 and not generator.is_async:
                    copied = self._evidence_comprehension(node, iterable, comp_scope)
                    if copied is not None:
                        return copied
                if i == 0:
                    self.frame.scope = comp_scope
                lid = self._loop_id(generator.iter, "_c")
                ctx = self._push_loop(lid, iterable.length, "comprehension", generator.iter)
                pushed.append(ctx)
                trips.append(iterable.length if not generator.ifs else None)
                element = self._loop_element(iterable, lid)
                self.assign(generator.target, element, generator.iter, record=False)
                for condition in generator.ifs:
                    self.gates.append(self.ev(condition).tags)
                    gates += 1
            if isinstance(node, ast.DictComp):
                key = self.ev(node.key)
                value = self.ev(node.value)
                result_value = value
                key_tags = strip_candref(key.tags)
            else:
                result_value = self.ev(node.elt)
                key_tags = frozenset()
        finally:
            for _ in range(gates):
                self.gates.pop()
            for ctx in reversed(pushed):
                self._pop_loop(ctx)
            self.frame.scope = saved
        length: int | None = 1
        for trip in trips:
            length = None if (trip is None or length is None) else length * trip
        loop_ids = {ctx.lid for ctx in pushed}
        cleaned = replace(result_value, tags=frozenset(
            t for t in result_value.tags if not any(t.endswith(f":{lid}") or f":{lid}:" in t for lid in loop_ids)))
        if isinstance(node, ast.DictComp):
            is_param = cleaned.role == WEIGHT and PARAM in cleaned.tags
            container = AbsVal(role=PDICT if is_param else DICT, items={}, default=cleaned, tags=key_tags,
                               length=length, obj=self._new_obj())
            return self._refresh_container(container)
        tags = strip_candref(cleaned.tags) | candref_tags(frozenset(alternatives(cleaned)))
        return AbsVal(role=SEQ, default=cleaned, tags=tags, length=length, obj=self._new_obj(), multi=True)

    def _implicit_loop(self, node: ast.AST, iterable: AbsVal, function: AbsVal, tag: str) -> tuple[AbsVal, AbsVal]:
        """Apply ``function`` to an element of ``iterable`` as an implicit loop (map, filter, key functions)."""
        lid = self._loop_id(node, tag)
        ctx = self._push_loop(lid, iterable.length, "implicit", node)
        element = self._loop_element(iterable, lid)
        try:
            result = self._dispatch(function, [element], {}, node)
        finally:
            self._pop_loop(ctx)
        return element, result

    # ------------------------------------------------------------------ calls

    def ex_Call(self, node: ast.Call) -> AbsVal:
        function = self.ev(node.func)
        args: list[AbsVal] = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                starred = self.ev(arg.value)
                if starred.elems is not None:
                    args.extend(starred.elems)
                else:
                    args.append(self._element_of(starred))
            else:
                args.append(self.ev(arg))
        kwargs: dict[str, AbsVal] = {}
        for keyword in node.keywords:
            value = self.ev(keyword.value)
            if keyword.arg is None:
                for key, item in (value.items or {}).items():
                    kwargs[key] = item
                if value.default is not None:
                    kwargs[f"**{len(kwargs)}"] = value.default
            else:
                kwargs[keyword.arg] = value
        result = self._dispatch(function, args, kwargs, node)
        if function.role == FUNC and isinstance(function.fn, FuncInfo):
            # A2 (2026-11-04): a user function taken from a loop (``for name, build in [...]: build(u)``) is another
            # rule each iteration, so its result varies with that loop in value. Only one of the functions is
            # analysed (a join keeps one), but the count sees one setting per iteration.
            varying = {t for t in function.tags if t.startswith("lv:") and t[3:] in self.active_loop_ids}
            if varying and result.role in (WEIGHT, PDICT, SAMPLE, STAT):
                result = result.plus(frozenset(varying))
        return result

    def _dispatch(self, function: AbsVal, args: list[AbsVal], kwargs: dict[str, AbsVal], node: ast.AST) -> AbsVal:
        if function.role == FUNC and isinstance(function.fn, FuncInfo):
            result = self.call_user(function.fn, args, kwargs, node, bound_self=function.recv)
            if function.recv is not None and function.recv.obj is not None and self._last_self is not None:
                self._rebind_obj(function.recv.obj, self._last_self)
            return result
        if function.role == FUNC and isinstance(function.fn, ClassInfo):
            return self._instantiate(function.fn, args, kwargs, node)
        if function.role == FUNC and isinstance(function.fn, PartialInfo):
            partial = function.fn
            return self._dispatch(partial.func, [*partial.args, *args], {**partial.kwargs, **kwargs}, node)
        if function.role == CONST and function.qual in NUMPY_DTYPE_CTORS and args:
            # A2 (2026-11-04): np.float32(a) is a cast; the value keeps its role, tags and modification history.
            first = args[0]
            if first.role == SEQ:
                return self._array_from_seq(first)
            return replace(first, obj=self._new_obj() if first.obj is not None else None, evaldiff=False)
        if function.qual and function.role in (FUNC, MODULE) and not function.qual.startswith("?"):
            return self.call_qual(function.qual, args, kwargs, node)
        if function.role == METHOD and function.recv is not None and function.qual:
            return self.call_method(function.recv, function.qual, args, kwargs, node)
        return self.call_unknown(function, args, kwargs, node)

    def _instantiate(self, info: ClassInfo, args: list[AbsVal], kwargs: dict[str, AbsVal], node: ast.AST) -> AbsVal:
        instance = AbsVal(role=INSTANCE, items={}, fn=info, obj=self._new_obj())
        init = info.lookup("__init__")
        if isinstance(init, FuncInfo):
            self.call_user(init, args, kwargs, node, bound_self=instance)
            if self._last_self is not None:
                instance = replace(self._last_self, obj=instance.obj)
        return instance

    def call_user(self, info: FuncInfo, args: list[AbsVal], kwargs: dict[str, AbsVal], node: ast.AST,
                  bound_self: AbsVal | None = None) -> AbsVal:
        """Analyse a user-defined function or lambda body with the caller's argument values."""
        self._last_self = None
        if len(self.call_sites) >= MAX_CALL_DEPTH or any(info is active for active in self.active_funcs):
            return self.call_unknown(None, [*([bound_self] if bound_self is not None else []), *args], kwargs, node)
        fn_node = info.node
        spec = fn_node.args  # type: ignore[attr-defined]
        scope = Scope(parent=info.closure, is_module=False)
        params = [*spec.posonlyargs, *spec.args]
        positional = ([bound_self] if bound_self is not None else []) + list(args)
        remaining = dict(kwargs)
        n_defaults = len(info.defaults)
        for i, param in enumerate(params):
            if i < len(positional):
                value = positional[i]
            elif param.arg in remaining:
                value = remaining.pop(param.arg)
            elif i >= len(params) - n_defaults:
                value = info.defaults[i - (len(params) - n_defaults)]
            else:
                value = AbsVal()
            scope.vars[param.arg] = value
        extra = positional[len(params):]
        if spec.vararg is not None:
            scope.vars[spec.vararg.arg] = AbsVal(role=SEQ, elems=tuple(extra), length=len(extra),
                                                 default=join_all(extra) if extra else None,
                                                 tags=frozenset().union(*(e.tags for e in extra)) if extra else frozenset())
        for kwarg in spec.kwonlyargs:
            scope.vars[kwarg.arg] = remaining.pop(kwarg.arg, info.kw_defaults.get(kwarg.arg, AbsVal()))
        if spec.kwarg is not None:
            scope.vars[spec.kwarg.arg] = AbsVal(role=DICT, items=dict(remaining), obj=self._new_obj())
        if not isinstance(fn_node, ast.Lambda):
            for sub in self._own_nodes(fn_node.body):  # type: ignore[attr-defined]
                if isinstance(sub, ast.Global):
                    scope.global_names.update(sub.names)
                elif isinstance(sub, ast.Nonlocal):
                    scope.nonlocal_names.update(sub.names)
        frame = Frame(scope=scope, func=info, cell=info.cell, lines=info.lines)
        caller = self.frame
        self.frames.append(caller)
        self.frame = frame
        self.active_funcs.append(info)
        call_line = getattr(node, "lineno", 0) or 0
        self.call_sites.append((caller.cell, call_line, getattr(node, "col_offset", 0) or 0))
        call_key = self._call_key(info, caller, node, list(args), kwargs, bound_self)
        self.call_keys.append(call_key)
        # Configuration count (A2, 2026-11-04): the calling statement's location and the evidence-data arguments (the
        # view a rule is applied to or fitted on) do not name another configuration.
        data_token = AbsVal(role=CONST, const="<data>")
        config_key = self._call_key(info, caller, node, [data_token if _is_data_arg(a) else a for a in args],
                                    {k: data_token if _is_data_arg(v) else v for k, v in kwargs.items()}, bound_self,
                                    config=True)
        self.config_keys.append(config_key.split(":", 1)[1])
        try:
            if isinstance(fn_node, ast.Lambda):
                result = self.ev(fn_node.body)
            else:
                self.exec_block(fn_node.body)  # type: ignore[attr-defined]
                result = join_all(frame.returns) if frame.returns else AbsVal(role=CONST, const=None)
                if frame.yields:
                    produced = join_all(frame.yields)
                    result = AbsVal(role=SEQ, default=produced, tags=strip_candref(produced.tags) | candref_tags(
                        frozenset(alternatives(produced))), obj=self._new_obj())
        finally:
            self.frame = self.frames.pop()
            self.active_funcs.pop()
            self.call_sites.pop()
            self.call_keys.pop()
            self.config_keys.pop()
        self._last_self = scope.vars.get(params[0].arg) if (bound_self is not None and params) else None
        return result

    @staticmethod
    def _own_nodes(body: list[ast.stmt]) -> Iterable[ast.AST]:
        stack: list[ast.AST] = list(body)
        while stack:
            node = stack.pop()
            yield node
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    stack.append(child)

    def call_unknown(self, function: AbsVal | None, args: list[AbsVal], kwargs: dict[str, AbsVal],
                     node: ast.AST) -> AbsVal:
        """Conservative semantics of a call whose body the audit cannot see: the result depends on every argument.

        A call that receives both a parameter setting and evidence data is treated as a forward evaluation of that
        setting on that data.
        """
        values = list(args) + list(kwargs.values())
        if function is not None and function.role == METHOD and function.recv is not None:
            values = [function.recv, *values]
        self._meet(values, node)
        tags: set[str] = set()
        for value in values:
            tags |= strip_candref(value.tags)
        if function is not None and function.role not in (FUNC, MODULE, METHOD) and not (function.qual or "").startswith("?"):
            tags |= strip_candref(function.tags)
        weights = [v for v in values if v.role in (WEIGHT, PDICT, MODEL_C)]
        datas = [v for v in values if v.role not in (WEIGHT, PDICT, MODEL_C) and (EVID in v.tags or v.role == SAMPLE)]
        if weights and datas:
            combined = AbsVal(role=WEIGHT, tags=frozenset().union(*(w.tags for w in weights)),
                              cands=self._combine_alternatives(w.cands for w in weights),
                              keyvar=combine_keyvar(weights))
            data = join_all(datas)
            add, remove = self._evaluation(combined, data)
            role = SAMPLE if any(d.role == SAMPLE for d in datas) else STAT
            return AbsVal(tags=(frozenset(tags) - remove) | add, role=role, obj=self._new_obj())
        role = SAMPLE if any(v.role == SAMPLE for v in values) else OTHER
        return AbsVal(tags=frozenset(tags), role=role, obj=self._new_obj())

    # ---------------------------------------------------------- known calls

    def call_qual(self, qual: str, args: list[AbsVal], kwargs: dict[str, AbsVal], node: ast.AST) -> AbsVal:
        parts = qual.split(".")
        root, name = parts[0], parts[-1]
        if root == "builtins":
            return self._builtin(name, args, kwargs, node)
        if root == "numpy":
            return self._numpy(tuple(parts[1:-1]), name, args, kwargs, node)
        if root == "scipy" and len(parts) >= 3 and parts[1] == "linalg":
            return self._numpy(("linalg",), name, args, kwargs, node)
        if root in ("json", "pickle"):
            return self._serial(name, args, kwargs, node)
        if root == "copy" and name in ("copy", "deepcopy") and args:
            value = args[0]
            return replace(value, obj=self._new_obj()) if value.obj is not None else value
        if qual in ("os.path.join", "posixpath.join"):
            return self._path_join(args)
        if root == "pathlib" and name in ("Path", "PurePath", "PosixPath", "PurePosixPath"):
            return self._path_join(args) if args else AbsVal(role=PATH, path=".")
        if qual == "runpy.run_path" and ((self._path_of(args[0]) if args else None) or "").startswith("/work/trial/"):
            return self.call_unknown(None, args, kwargs, node)  # the trial's own code, as visible as an import of it
        if root in DYNAMIC_ROOTS or qual in DYNAMIC_EXEC or (root == "os" and name.startswith(("exec", "spawn"))):
            self._flag(S_DYNAMIC, node, f"call to {qual} runs code outside the audited cells")
            return self.call_unknown(None, args, kwargs, node)
        if qual == "importlib.import_module":
            target = args[0] if args else AbsVal()
            if isinstance(target.const, str):
                self._check_editor_module(target.const, node)
                return AbsVal(role=MODULE, qual=target.const)
            self._flag(S_DYNAMIC, node, "importlib.import_module with a non-constant name")
            return AbsVal()
        if qual == "importlib.util.spec_from_file_location":
            location = self._path_of(args[1]) if len(args) > 1 else self._path_of(kwargs.get("location"))
            if location is None or not location.startswith("/work/trial/"):
                self._flag(S_DYNAMIC, node, f"loads a module from {location or 'a non-constant path'}, outside /work/trial; "
                                            "its code is not visible to the audit")
                return AbsVal(role=OTHER)
            stem = location.rsplit("/", 1)[-1].removesuffix(".py")
            return AbsVal(role=OTHER, qual=f"{SPEC_PREFIX}{stem}")
        if qual == "importlib.util.module_from_spec":
            # A trial module (forward.py) loaded by file location is the module ``import forward`` binds; any other
            # module is unknown, and its functions are calls whose bodies the audit cannot see (call_unknown).
            spec_value = args[0] if args else kwargs.get("spec", AbsVal())
            if (spec_value.qual or "").startswith(SPEC_PREFIX):
                return AbsVal(role=MODULE, qual=spec_value.qual[len(SPEC_PREFIX):])
            return AbsVal(role=MODULE, qual="?module")
        if qual in ("sys.path.append", "sys.path.insert"):
            location = self._path_of(args[-1]) if args else None
            if location is None or location.startswith(("/work/scratch", "/work/patches")) or not location.startswith("/work/trial"):
                self.editor_paths.append(location or "<non-constant>")
            return AbsVal(role=CONST)
        if root == "functools" and name == "partial" and args:
            return AbsVal(role=FUNC, fn=PartialInfo(args[0], list(args[1:]), dict(kwargs)))
        if root == "functools" and name == "reduce" and len(args) >= 2:
            _, result = self._implicit_loop(node, args[1], args[0], "_r")
            return AbsVal(tags=strip_candref(result.tags | args[1].tags), role=STAT)
        if root == "itertools":
            return self._itertools(name, args, kwargs)
        if root == "random":
            return self._random_value(args, kwargs)
        if root == "math" or root == "statistics" or root == "operator":
            tags: set[str] = set()
            for value in [*args, *kwargs.values()]:
                tags |= strip_candref(value.tags)
            return AbsVal(role=STAT if tags else CONST, tags=frozenset(tags))
        return self.call_unknown(None, args, kwargs, node)

    def _builtin(self, name: str, args: list[AbsVal], kwargs: dict[str, AbsVal], node: ast.AST) -> AbsVal:
        first = args[0] if args else None
        if name == "len":
            if first is None:
                return AbsVal(role=CONST)
            return _size_value(first, first.length if first.length is not None else NOCONST)
        if name == "range":
            bounds = [_const_int(a) for a in args]
            length = len(range(*bounds)) if bounds and all(b is not None for b in bounds) and bounds[-1] != 0 or (
                len(bounds) < 3 and bounds and all(b is not None for b in bounds)) else None  # type: ignore[arg-type]
            tags = frozenset().union(*(strip_candref(a.tags) for a in args)) if args else frozenset()
            return AbsVal(role=SEQ, default=AbsVal(role=CONST, tags=tags), length=length, tags=tags, multi=True)
        if name == "enumerate" and first is not None:
            elems = None
            if first.elems is not None:
                elems = tuple(AbsVal(role=SEQ, elems=(AbsVal(role=CONST, const=i), e), length=2,
                                     default=e, tags=strip_candref(e.tags) | candref_tags(frozenset(alternatives(e))))
                              for i, e in enumerate(first.elems))
            element = self._element_of(first)
            pair = AbsVal(role=SEQ, elems=(AbsVal(role=CONST), element), length=2, default=element,
                          tags=strip_candref(element.tags) | candref_tags(frozenset(alternatives(element))))
            return AbsVal(role=SEQ, elems=elems, default=pair, length=first.length, tags=first.tags, obj=self._new_obj())
        if name == "zip" and args:
            lengths = [a.length for a in args]
            length = min(lengths) if all(l is not None for l in lengths) else None  # type: ignore[type-var]
            elems = None
            if all(a.elems is not None for a in args):
                n = min(len(a.elems) for a in args)  # type: ignore[arg-type]
                elems = tuple(self._tuple([a.elems[i] for a in args]) for i in range(n))  # type: ignore[index]
            pair = self._tuple([self._element_of(a) for a in args])
            tags = frozenset().union(*(a.tags for a in args))
            return AbsVal(role=SEQ, elems=elems, default=pair, length=length, tags=tags, obj=self._new_obj())
        if name in ("list", "tuple", "set", "frozenset", "sorted", "reversed", "iter"):
            if first is None:
                return AbsVal(role=SEQ, elems=(), length=0, obj=self._new_obj())
            result = self._as_sequence(first)
            key = kwargs.get("key")
            if name == "sorted" and key is not None:
                _, scored = self._implicit_loop(node, first, key, "_s")
                self._best_keeping(first, scored.tags)
                result = result.plus(_no_grad(strip_candref(scored.tags)))
            return result
        if name == "dict":
            return self._dict_call(first, kwargs)
        if name in ("max", "min"):
            key = kwargs.get("key")
            if len(args) == 1:
                iterable = args[0]
                element = self._element_of(iterable)
                selection = strip_candref(iterable.tags)
                if key is not None:
                    element, scored = self._implicit_loop(node, iterable, key, "_m")
                    selection = selection | strip_candref(scored.tags)
                self._best_keeping(iterable, selection | iterable.tags)
                selection = _no_grad(selection)
                return element.plus(selection) if element.role != WEIGHT else replace(
                    element, tags=self._as_candidate(element.tags | selection))
            result = join_all(args)
            if key is not None:
                for arg in args:
                    scored = self._dispatch(key, [arg], {}, node)
                    result = result.plus(_no_grad(strip_candref(scored.tags)))
            return result
        if name == "sum" and first is not None:
            element = self._element_of(first)
            return AbsVal(tags=strip_candref(element.tags | first.tags), role=STAT if element.role != WEIGHT else WEIGHT,
                          cands=element.cands if element.role == WEIGHT else frozenset(), obj=self._new_obj())
        if name in ("abs", "float", "int", "round", "bool", "complex", "str", "repr", "format", "ascii", "ord", "chr",
                    "hex", "bin", "oct", "hash"):
            if first is None:
                return AbsVal(role=CONST)
            tags = frozenset().union(*(strip_candref(a.tags) for a in args))
            const = NOCONST
            if first.has_const() and name in ("float", "int", "abs", "round", "bool", "str") and (
                    _is_number(first.const) or isinstance(first.const, str)):
                try:
                    const = {"float": float, "int": int, "abs": abs, "round": round, "bool": bool, "str": str}[name](first.const)
                except (TypeError, ValueError, OverflowError):
                    const = NOCONST
            if first.role == WEIGHT and name == "round":
                return replace(first, cands=self._new_cands(node, first.cands), pristine=False, obj=self._new_obj())
            if name in ("str", "repr", "format", "ascii", "hex", "bin", "oct", "hash", "bool"):
                tags = _no_grad(tags)
            return AbsVal(role=CONST if not tags else STAT, tags=tags, const=const)
        if name == "print":
            return AbsVal(role=CONST, const=None)
        if name in ("isinstance", "issubclass", "callable", "hasattr", "id", "type", "dir", "help", "input",
                    "breakpoint", "super", "object", "property", "staticmethod", "classmethod", "NotImplemented"):
            return AbsVal(role=CONST)
        if name == "getattr" and first is not None:
            attr = args[1] if len(args) > 1 else AbsVal()
            if isinstance(attr.const, str):
                return self._attribute(first, attr.const, node)
            if first.role == MODULE:
                self._flag(S_DYNAMIC, node, f"getattr on module {first.qual} with a non-constant name")
            return AbsVal(tags=strip_candref(first.tags | attr.tags))
        if name == "setattr" and len(args) >= 3:
            target, attr, value = args[0], args[1], args[2]
            if target.obj is not None:
                items = dict(target.items or {})
                if isinstance(attr.const, str):
                    items[attr.const] = value
                self._rebind_obj(target.obj, self._refresh_container(replace(target, items=items)) if target.items is not None
                                 else target.plus(strip_candref(value.tags)))
            return AbsVal(role=CONST)
        if name in ("globals", "locals", "vars") and not args:
            return AbsVal(role=GLOBALS)
        if name in ("exec", "eval", "compile"):
            return self._dynamic_code(name, args, node)
        if name == "open":
            path = self._path_of(first) if first is not None else None
            mode = args[1] if len(args) > 1 else kwargs.get("mode")
            mode_text = mode.const if mode is not None and isinstance(mode.const, str) else "r"
            return AbsVal(role=FILE, path=path, mode=mode_text, obj=self._new_obj(), tags=self._path_variance(first))
        if name == "map" and len(args) >= 2:
            _, result = self._implicit_loop(node, args[1], args[0], "_map")
            tags = strip_candref(result.tags) | candref_tags(frozenset(alternatives(result)))
            return AbsVal(role=SEQ, default=result, length=args[1].length, tags=tags, obj=self._new_obj())
        if name == "filter" and len(args) >= 2:
            element, result = self._implicit_loop(node, args[1], args[0], "_filter")
            self._best_keeping(args[1], result.tags)
            return replace(self._as_sequence(args[1]), length=None).plus(strip_candref(result.tags))
        if name == "next" and first is not None:
            return self._element_of(first)
        if name in ("any", "all"):
            tags = frozenset().union(*(strip_candref(self._deep_tags(a)) for a in args)) if args else frozenset()
            return AbsVal(role=CONST if not tags else STAT, tags=tags)
        if name in ("divmod", "pow") and len(args) >= 2:
            return self.combine(ast.Pow() if name == "pow" else ast.FloorDiv(), args[0], args[1], node)
        if name == "__import__":
            target = first.const if first is not None and isinstance(first.const, str) else None
            if target is None:
                self._flag(S_DYNAMIC, node, "__import__ with a non-constant name")
                return AbsVal()
            return AbsVal(role=MODULE, qual=target.split(".")[0])
        return self.call_unknown(None, args, kwargs, node)

    def _tuple(self, elems: list[AbsVal]) -> AbsVal:
        tags: set[str] = set()
        for e in elems:
            tags |= strip_candref(e.tags) | candref_tags(frozenset(alternatives(e)))
        return AbsVal(role=SEQ, elems=tuple(elems), length=len(elems), default=join_all(elems), tags=frozenset(tags))

    def _as_sequence(self, value: AbsVal) -> AbsVal:
        if value.role == SEQ:
            return replace(value, obj=self._new_obj())
        if value.role in (DICT, PDICT):
            keys = None
            if value.items is not None and value.default is None:
                keys = tuple(AbsVal(role=CONST, const=k) for k in value.items)
            return AbsVal(role=SEQ, elems=keys, length=len(keys) if keys is not None else value.length,
                          default=AbsVal(role=CONST), obj=self._new_obj())
        if value.role in (MODEL_C, EVID_C):
            return AbsVal(role=SEQ, default=AbsVal(role=CONST), obj=self._new_obj())
        element = self._element_of(value)
        tags = strip_candref(value.tags) | candref_tags(frozenset(alternatives(element)))
        return AbsVal(role=SEQ, default=element, length=value.length, tags=tags, obj=self._new_obj())

    def _evidence_keys(self, container: AbsVal) -> list[str]:
        """Keys of an evidence container: those it holds, then the label keys and the standard evidence keys."""
        held = list(container.items or {})
        standard = (*self.label_keys, *EVIDENCE_DATA_KEYS)
        return held + [key for key in standard if key not in held]

    def _evidence_value(self, container: AbsVal, key: str) -> AbsVal:
        if container.items and key in container.items:
            return container.items[key]
        return self._evidence_item(container, key)

    def _evidence_copy(self, container: AbsVal) -> AbsVal:
        """A dict copy of the evidence container (``dict(ev)``, ``{**ev}``): one value per key, as in the container."""
        return AbsVal(role=EVID_C, items={key: self._evidence_value(container, key)
                                          for key in self._evidence_keys(container)}, obj=self._new_obj())

    def _evidence_comprehension(self, node: ast.DictComp, iterable: AbsVal, scope: Scope) -> AbsVal | None:
        """A dict comprehension over the keys or items of an evidence container, evaluated once per key.

        ``{k: v for k, v in ev.items()}`` or ``{k: ev[k] for k in ev.files}`` copies the container; evaluated with an
        unknown key, every value would join the label arrays. Each key is bound in turn (conditions that are constant
        and false drop it), so the copy keeps labels on the label keys. Returns None when a key of the result is not
        a constant string, and the comprehension is analysed as any other.
        """
        generator = node.generators[0]
        if iterable.role == EVID_C:
            container, mode = iterable, "keys"
        elif iterable.role == SEQ and iterable.recv is not None and iterable.recv.role == EVID_C:
            container, mode = iterable.recv, iterable.mode
        else:
            return None
        saved = self.frame.scope
        self.frame.scope = scope
        items: dict[str, AbsVal] = {}
        try:
            for key in self._evidence_keys(container):
                key_value = AbsVal(role=CONST, const=key)
                element = (self._tuple([key_value, self._evidence_value(container, key)]) if mode == "items"
                           else key_value)
                self.assign(generator.target, element, generator.iter, record=False)
                if any(test.has_const() and not test.const for test in (self.ev(c) for c in generator.ifs)):
                    continue
                result_key = self.ev(node.key)
                if not isinstance(result_key.const, str):
                    return None
                items[result_key.const] = self.ev(node.value)
        finally:
            self.frame.scope = saved
        return AbsVal(role=EVID_C, items=items, obj=self._new_obj())

    def _dict_call(self, first: AbsVal | None, kwargs: dict[str, AbsVal]) -> AbsVal:
        if first is not None and first.role in (MODEL_C, PDICT):
            items = dict(first.items or {})
            default = first.default if first.role == PDICT else self._model_item(first, NOCONST)
            items.update(kwargs)
            return self._refresh_container(AbsVal(role=PDICT, items=items, default=default, obj=self._new_obj(),
                                                  pristine=first.pristine or first.role == MODEL_C))
        if first is not None and first.role == EVID_C:
            return self._evidence_copy(first)
        if first is not None and first.role == SEQ and first.recv is not None and first.recv.role == EVID_C \
                and first.mode == "items":
            return self._evidence_copy(first.recv)  # dict(ev.items())
        if first is not None and first.role == DICT:
            items = dict(first.items or {})
            items.update(kwargs)
            return self._refresh_container(replace(first, items=items, obj=self._new_obj()))
        items = dict(kwargs)
        default = None
        if first is not None:
            pair = self._element_of(first)
            default = pair.elems[1] if pair.elems is not None and len(pair.elems) == 2 else self._element_of(pair)
        is_param = any(v.role == WEIGHT and PARAM in v.tags for v in items.values()) or (
            default is not None and default.role == WEIGHT and PARAM in default.tags)
        return self._refresh_container(AbsVal(role=PDICT if is_param else DICT, items=items, default=default,
                                              obj=self._new_obj()))

    def _itertools(self, name: str, args: list[AbsVal], kwargs: dict[str, AbsVal]) -> AbsVal:
        lengths = [a.length for a in args]
        length: int | None = None
        if name == "product":
            repeat = _const_int(kwargs.get("repeat")) or 1
            if all(l is not None for l in lengths):
                length = math.prod(lengths) ** repeat  # type: ignore[arg-type]
        elif name in ("combinations", "permutations", "combinations_with_replacement") and args:
            n = args[0].length
            r = _const_int(args[1]) if len(args) > 1 else n
            if n is not None and r is not None:
                if name == "combinations":
                    length = math.comb(n, r)
                elif name == "permutations":
                    length = math.perm(n, r)
                else:
                    length = math.comb(n + r - 1, r)
        elif name == "chain" and all(l is not None for l in lengths):
            length = sum(lengths)  # type: ignore[arg-type]
        elif name == "repeat" and len(args) > 1:
            length = _const_int(args[1])
        element = self._tuple([self._element_of(a) for a in args]) if name in ("product", "zip_longest") else (
            join_all([self._element_of(a) for a in args]) if args else AbsVal())
        tags = frozenset().union(*(a.tags for a in args)) if args else frozenset()
        return AbsVal(role=SEQ, default=element, length=length, tags=tags, obj=self._new_obj())

    def _random_value(self, args: list[AbsVal], kwargs: dict[str, AbsVal], base: AbsVal | None = None) -> AbsVal:
        tags: set[str] = {RANDOM}
        for value in [*args, *kwargs.values()]:
            tags |= strip_candref(value.tags)
        tags |= {f"lv:{loop}" for loop in self.active_loop_ids}
        role = STAT
        if base is not None and base.role in (WEIGHT, SAMPLE):
            role = base.role
        return AbsVal(tags=frozenset(tags), role=role, obj=self._new_obj())

    def _dynamic_code(self, name: str, args: list[AbsVal], node: ast.AST) -> AbsVal:
        source = args[0] if args else AbsVal()
        if isinstance(source.const, str) and name in ("exec", "eval"):
            try:
                tree = ast.parse(source.const, mode="exec" if name == "exec" else "eval")
            except SyntaxError:
                self._flag(S_DYNAMIC, node, f"{name} of a constant string that does not parse")
                return AbsVal()
            for sub in ast.walk(tree):
                if hasattr(sub, "lineno"):
                    sub.lineno = getattr(node, "lineno", 1)  # type: ignore[attr-defined]
                    sub.end_lineno = getattr(node, "lineno", 1)  # type: ignore[attr-defined]
            if isinstance(tree, ast.Module):
                self.exec_block(tree.body)
                return AbsVal(role=CONST, const=None)
            return self.ev(tree.body)  # type: ignore[attr-defined]
        if source.path is not None and source.path.startswith("/work/trial/") and "*" not in source.path:
            return AbsVal(role=CONST, const=None)
        self._flag(S_DYNAMIC, node, f"{name} of non-constant code; what it ran is not visible to the audit")
        self.dynamic_tags = self.dynamic_tags | strip_candref(source.tags)
        return AbsVal(tags=strip_candref(source.tags))

    # ----------------------------------------------------------- files

    def _path_of(self, value: AbsVal | None) -> str | None:
        if value is None:
            return None
        if isinstance(value.const, str):
            return value.const
        if value.path is not None:
            return value.path
        return None

    def _path_join(self, values: list[AbsVal]) -> AbsVal:
        parts = []
        for value in values:
            text = self._path_of(value)
            parts.append(text if text is not None else "*")
        joined = ""
        for part in parts:
            if not joined:
                joined = part
            elif part.startswith("/"):
                joined = part
            else:
                joined = joined.rstrip("/") + "/" + part
        tags = frozenset().union(*(strip_candref(v.tags) for v in values)) if values else frozenset()
        return AbsVal(role=PATH, path=joined, tags=tags)

    @staticmethod
    def _is_patch_path(path: str | None) -> bool:
        return path is None or "*" in path and "/patches" not in path and "/scratch" not in path or "/patches" in path or (
            path.startswith("patches/"))

    def _file_write(self, path: str | None, payload: AbsVal, node: ast.AST) -> None:
        tags = self._deep_tags(payload)
        grads = _grad_tags(tags)
        if grads and self._is_patch_path(path):
            self._update_sink(grads, node, f"data written to {path or 'a file with a non-constant path'}")
        if path is not None and path.endswith(".py"):
            self._flag(S_DYNAMIC, node, f"writes the Python source file {path}; code run from files is not analysed")
        if path is not None and "*" not in path:
            self.files[path] = payload
        self.file_tags = self.file_tags | strip_candref(tags)

    def _file_read(self, path: str | None, role: str = OTHER) -> AbsVal:
        if path is not None and path.startswith("/work/trial/") and "*" not in path:
            if is_evidence_path(path):  # A2 (2026-11-04): evidence.npz and experience.npz
                return AbsVal(role=EVID_C, obj=self._new_obj(), path=path)
            return AbsVal(role=CONST, path=path)
        if path is not None and path in self.files:
            return replace(self.files[path], obj=self._new_obj())
        if path is not None and "*" in path:
            matches = [v for p, v in self.files.items() if fnmatch.fnmatch(p, path)]
            if matches:
                return replace(join_all(matches), obj=self._new_obj())
        return AbsVal(role=role, tags=self.file_tags, obj=self._new_obj())

    def _path_variance(self, value: AbsVal | None) -> frozenset[str]:
        """Loops a path or file handle varies with: what is read through it is another file each iteration."""
        if value is None:
            return frozenset()
        loops = {t.split(":", 1)[1] for t in value.tags if t.startswith(("lv:", "li:", "lim:"))}
        return frozenset(f"lv:{lid}" for lid in loops if lid in self.active_loop_ids)

    def _mark_deep(self, value: AbsVal, extra: frozenset[str], depth: int) -> AbsVal:
        """``value`` with ``extra`` tags on it and on its components (as a loop element is marked)."""
        if not extra:
            return value
        value = value.plus(extra)
        if depth >= LOOP_MARK_DEPTH:
            return value
        changes: dict[str, object] = {}
        if value.elems is not None:
            changes["elems"] = tuple(self._mark_deep(item, extra, depth + 1) for item in value.elems)
        if value.items is not None:
            changes["items"] = {key: self._mark_deep(item, extra, depth + 1) for key, item in value.items.items()}
        if value.default is not None:
            changes["default"] = self._mark_deep(value.default, extra, depth + 1)
        return replace(value, **changes) if changes else value  # type: ignore[arg-type]

    def _np_load(self, args: list[AbsVal], kwargs: dict[str, AbsVal]) -> AbsVal:
        source = args[0] if args else kwargs.get("file")
        return self._mark_deep(self._np_load_path(source), self._path_variance(source), 0)

    def _np_load_path(self, source: AbsVal | None) -> AbsVal:
        path = source.path if source is not None and source.role == FILE else self._path_of(source)
        if path is None:
            return AbsVal(role=EVID_C, tags=self.file_tags, obj=self._new_obj())
        lowered = path.lower()
        if is_evidence_path(path):  # A2 (2026-11-04): evidence.npz and experience.npz
            return AbsVal(role=EVID_C, obj=self._new_obj(), path=path)
        if path in self.files:
            stored = self.files[path]
            return replace(stored, obj=self._new_obj())
        if lowered.endswith(".npz") and ("model" in lowered or lowered.startswith("/work/trial/")):
            return AbsVal(role=MODEL_C, obj=self._new_obj(), path=path, pristine=True)
        return self._file_read(path, role=STAT)

    def _np_save(self, name: str, args: list[AbsVal], kwargs: dict[str, AbsVal], node: ast.AST) -> AbsVal:
        target = args[0] if args else kwargs.get("file", kwargs.get("fname"))
        path = target.path if target is not None and target.role == FILE else self._path_of(target)
        if name in ("save", "savetxt"):
            payload = args[1] if len(args) > 1 else kwargs.get("arr", kwargs.get("X", AbsVal()))
        else:
            items = {k: v for k, v in kwargs.items() if k not in ("file", "allow_pickle")}
            for i, value in enumerate(args[1:]):
                items[f"arr_{i}"] = value
            is_param = any(v.role == WEIGHT and PARAM in v.tags for v in items.values())
            payload = self._refresh_container(AbsVal(role=PDICT if is_param else DICT, items=items, obj=self._new_obj()))
            if not is_param:
                payload = replace(payload, role=EVID_C if any(EVID in v.tags for v in items.values()) else DICT)
        self._file_write(path, payload, node)
        return AbsVal(role=CONST, const=None)

    def _serial(self, name: str, args: list[AbsVal], kwargs: dict[str, AbsVal], node: ast.AST) -> AbsVal:
        if name in ("dump",) and len(args) >= 2:
            handle = args[1]
            self._file_write(handle.path if handle.role == FILE else self._path_of(handle), args[0], node)
            return AbsVal(role=CONST, const=None)
        if name == "dumps" and args:
            return AbsVal(role=CONST if not self._deep_tags(args[0]) else STAT, tags=strip_candref(self._deep_tags(args[0])),
                          const=NOCONST)
        if name == "load" and args:
            handle = args[0]
            stored = self._file_read(handle.path if handle.role == FILE else self._path_of(handle))
            return self._mark_deep(stored, self._path_variance(handle), 0)
        if name == "loads" and args:
            return AbsVal(tags=strip_candref(self._deep_tags(args[0])), role=OTHER, obj=self._new_obj())
        return self.call_unknown(None, args, kwargs, node)

    # ----------------------------------------------------------- numpy

    def _numpy(self, sub: tuple[str, ...], name: str, args: list[AbsVal], kwargs: dict[str, AbsVal],
               node: ast.AST) -> AbsVal:
        if sub and sub[0] == "random":
            if name in RANDOM_CTORS:
                return AbsVal(role=RNG, obj=self._new_obj())
            if name == "seed":
                return AbsVal(role=CONST, const=None)
            if name == "shuffle" and args:
                self._mutate_value(args[0], self._random_value([args[0]], {}, args[0]), node)
                return AbsVal(role=CONST, const=None)
            return self._random_value(args, kwargs)
        if sub and sub[0] == "linalg":
            return self._linalg(name, args, kwargs, node)
        if sub and sub[0] in ("fft", "char", "ma", "lib", "polynomial", "testing", "emath"):
            return self.call_unknown(None, args, kwargs, node)
        first = args[0] if args else None
        if name == "load":
            return self._np_load(args, kwargs)
        if name in ("save", "savez", "savez_compressed", "savetxt"):
            return self._np_save(name, args, kwargs, node)
        if name in ("loadtxt", "genfromtxt", "fromfile"):
            return self._file_read(self._path_of(first), role=STAT)
        if name in NP_SHAPE_NEW:
            return AbsVal(role=STAT, length=_shape_length(first), obj=self._new_obj(), tags=self._shape_tags(first))
        if name == "full":
            fill = args[1] if len(args) > 1 else kwargs.get("fill_value", AbsVal())
            return AbsVal(role=STAT, length=_shape_length(first), obj=self._new_obj(),
                          tags=strip_candref(fill.tags) | self._shape_tags(first))
        if name in ("eye", "identity"):
            return AbsVal(role=STAT, basis=True, length=_const_int(first), obj=self._new_obj())
        if name in NP_LIKE or name == "full_like":
            base = first or AbsVal()
            fill_tags = strip_candref(args[1].tags) if name == "full_like" and len(args) > 1 else frozenset()
            size_tags = _size_value(base).tags & {FAILSET, LABEL}
            return AbsVal(role=WEIGHT if base.role == WEIGHT else STAT, tags=fill_tags | size_tags, length=base.length,
                          obj=self._new_obj())
        if name in ("arange", "linspace", "logspace", "geomspace"):
            return self._np_range(name, args, kwargs)
        if name in NP_COPY or (name in NP_VIEW and first is not None and first.role != SEQ):
            if first is None:
                return AbsVal(role=STAT, obj=self._new_obj())
            if first.role == SEQ:
                return self._array_from_seq(first)
            view = name in NP_VIEW
            return replace(first, obj=first.obj if view else self._new_obj(), evaldiff=False)
        if name in NP_VIEW and first is not None:
            return self._array_from_seq(first)
        if name in NP_REDUCE:
            axis = kwargs.get("axis", args[1] if len(args) > 1 else None)
            extra = [v for k, v in kwargs.items() if k in ("weights",)]
            return self._reduce(first or AbsVal(), axis, extra, name, node)
        if name in NP_ARITH and len(args) >= 2:
            result = self.combine(NP_ARITH[name](), args[0], args[1], node)
            return self._ufunc_out(result, kwargs, node)
        if name in NP_COMPARE:
            values = list(args)
            self._meet(values, node)
            tags = _mask_tags(frozenset().union(*(strip_candref(v.tags) for v in values))) if values else frozenset()
            if name in NP_ORDER_COMPARE:
                tags = self._decision_variance(tags, values)  # A2 (2026-11-04)
            if name in ("equal", "not_equal"):
                tags = tags | self._class_view(values)
            role = SAMPLE if any(v.role == SAMPLE for v in values) else STAT
            return AbsVal(tags=tags, role=role, obj=self._new_obj())
        if name in NP_ELEMENTWISE:
            if name == "where" and len(args) == 1:
                return AbsVal(tags=_with_lsize(_index_tags(strip_candref(args[0].tags))), role=STAT, multi=True,
                              obj=self._new_obj())
            values = list(args)
            condition: AbsVal | None = None
            if name == "where" and len(values) == 3:
                condition, values = values[0], values[1:]
            result = values[0] if values else AbsVal()
            for value in values[1:]:
                result = self.combine(ast.Add(), result, value, node)
            if len(values) == 1 and result.role == WEIGHT and PARAM in result.tags:
                result = replace(result, cands=self._new_cands(node, result.cands), pristine=False, obj=self._new_obj())
            if condition is not None:
                extra = strip_candref(condition.tags)
                result = replace(result, tags=self._as_candidate(result.tags | extra)) if result.role == WEIGHT else result.plus(extra)
            return self._ufunc_out(result, kwargs, node)
        if name in NP_UNARY:
            if first is None:
                return AbsVal(role=STAT)
            tags = frozenset().union(*(strip_candref(a.tags) for a in args))
            if first.role == WEIGHT and PARAM in first.tags:
                return replace(first, tags=self._as_candidate(tags | first.tags), cands=self._new_cands(node, first.cands),
                               pristine=False, obj=self._new_obj(), evaldiff=False)
            role = first.role if first.role in (SAMPLE, STAT, WEIGHT) else STAT
            return self._ufunc_out(AbsVal(tags=tags, role=role, obj=self._new_obj(), length=first.length), kwargs, node)
        if name in NP_INDEX:
            if first is None:
                return AbsVal(role=STAT)
            tags = frozenset().union(*(strip_candref(a.tags) for a in args))
            if name in INDEX_VALUED:
                tags = _index_tags(tags)
            elif name == "isin":
                tags = _mask_tags(tags)
            if name in VARIABLE_SIZE:
                tags = _with_lsize(tags)
            axis = kwargs.get("axis", args[1] if len(args) > 1 and name == "argsort" else None)
            role = SAMPLE if first.role == SAMPLE and axis is not None and _const_int(axis) not in (None, 0) else STAT
            return AbsVal(tags=tags, role=role, multi=True, obj=self._new_obj())
        if name in NP_MATMUL:
            operands = list(args)
            if name == "einsum" and operands and operands[0].role == CONST:
                operands = operands[1:]
            if not operands:
                return AbsVal(role=STAT)
            result = operands[0]
            for operand in operands[1:]:
                result = self._matmul(result, operand, node)
            return result
        if name in NP_JOIN:
            return self._np_join(name, args, node)
        if name in NP_STATS:
            values = list(args)
            tags = frozenset().union(*(strip_candref(v.tags) for v in values)) if values else frozenset()
            return AbsVal(tags=tags, role=STAT, obj=self._new_obj())
        if name == "gradient" and first is not None:
            tags = frozenset().union(*(strip_candref(v.tags) for v in args))
            if is_tainted(first.tags) and any(t.startswith(("eval:", "evalc:")) for t in first.tags):
                site = self._site(node, "fd")
                self._register_site(site, "fd", node)
                tags = tags | {f"fd:{site}"}
            return AbsVal(tags=tags, role=STAT, obj=self._new_obj())
        if name in NP_MUTATE and args:
            values = args[1:]
            tags = frozenset().union(*(strip_candref(v.tags) for v in values)) if values else frozenset()
            self._mutate_value(args[0], AbsVal(tags=tags, role=args[0].role), node)
            return AbsVal(role=CONST, const=None)
        if name == "polyfit" and len(args) >= 2:
            return self._solver("polyfit", args[0], args[1], node)
        if name == "meshgrid":
            elems = tuple(AbsVal(tags=strip_candref(a.tags), role=STAT, obj=self._new_obj()) for a in args)
            return AbsVal(role=SEQ, elems=elems, length=len(elems), default=join_all(list(elems)))
        if name in ("isscalar", "ndim", "shape", "size", "result_type", "promote_types", "dtype", "finfo", "iinfo",
                    "set_printoptions", "seterr", "printoptions", "errstate", "array_str", "array_repr", "array2string"):
            if name in ("array_str", "array_repr", "array2string") and first is not None:
                return AbsVal(role=STAT, tags=strip_candref(first.tags))
            if name in ("shape", "size") and first is not None:
                return _size_value(first)
            return AbsVal(role=CONST)
        if first is not None and name in ("float32", "float64", "int64", "int32", "uint8", "bool_", "float16"):
            return replace(first, obj=self._new_obj()) if first.role != CONST else first
        return self.call_unknown(None, args, kwargs, node)

    def _ufunc_out(self, result: AbsVal, kwargs: dict[str, AbsVal], node: ast.AST) -> AbsVal:
        out = kwargs.get("out")
        if out is not None and out.obj is not None:
            self._mutate_value(out, result, node)
        return result

    def _mutate_value(self, target: AbsVal, value: AbsVal, node: ast.AST) -> None:
        """In-place update of the object behind ``target`` (np.copyto, out=, rng.shuffle and similar)."""
        if target.obj is None:
            return
        if target.role == WEIGHT and PARAM in target.tags:
            fresh = _grad_tags(value.tags) - _grad_tags(target.tags)
            if fresh:
                self._update_sink(fresh, node, "a parameter array is overwritten in place")
            new = replace(target, tags=self._as_candidate(strip_candref(target.tags | value.tags)),
                          cands=self._new_cands(node, target.cands), pristine=False)
        else:
            new = replace(target, tags=strip_candref(target.tags | value.tags), pristine=False)
        names = self._rebind_obj(target.obj, new)
        self._record_event(names, value.tags, node, aug=True)

    def _np_range(self, name: str, args: list[AbsVal], kwargs: dict[str, AbsVal]) -> AbsVal:
        tags = frozenset().union(*(strip_candref(a.tags) for a in [*args, *kwargs.values()])) if (args or kwargs) else frozenset()
        length: int | None = None
        if name == "arange":
            numbers = [_const_number(a) for a in args]
            if numbers and all(n is not None for n in numbers):
                if len(numbers) == 1:
                    start, stop, step = 0.0, numbers[0], 1.0
                elif len(numbers) == 2:
                    start, stop, step = numbers[0], numbers[1], 1.0
                else:
                    start, stop, step = numbers[0], numbers[1], numbers[2]
                if step:
                    length = max(0, math.ceil((stop - start) / step))  # type: ignore[operator]
        else:
            num = kwargs.get("num", args[2] if len(args) > 2 else None)
            length = _const_int(num) if num is not None else 50
        return AbsVal(role=STAT, tags=tags, length=length, obj=self._new_obj(), multi=True)

    def _array_from_seq(self, value: AbsVal) -> AbsVal:
        element = self._element_of(value)
        role = element.role if element.role in (WEIGHT, SAMPLE) else STAT
        tags = strip_candref(self._deep_tags(value))
        cands = element.cands if role == WEIGHT else frozenset()
        return AbsVal(tags=tags, role=role, length=value.length, cands=cands, obj=self._new_obj(),
                      pristine=element.pristine if role == WEIGHT else False)

    def _np_join(self, name: str, args: list[AbsVal], node: ast.AST) -> AbsVal:
        first = args[0] if args else AbsVal()
        pieces = list(first.elems) if first.role == SEQ and first.elems is not None else (
            [self._element_of(first)] if first.role == SEQ else [first])
        pieces += list(args[1:]) if name in ("append", "insert", "take", "compress", "extract") else []
        tags: set[str] = set()
        for piece in pieces:
            tags |= strip_candref(self._deep_tags(piece))
        if name in ("delete", "take", "compress") and len(args) > 1 and first.role == SAMPLE and EVID in first.tags:
            # A2 (2026-11-04): np.delete(x, i, 0) in a loop over i is a leave-one-out selection of evidence rows.
            tags |= self._row_variance(strip_candref(args[1].tags))
        if any(p.role == SAMPLE for p in pieces):
            return AbsVal(tags=frozenset(tags), role=SAMPLE, obj=self._new_obj())
        weights = [p for p in pieces if p.role == WEIGHT]
        if weights and any(PARAM in p.tags for p in weights) and name not in ("split", "array_split", "hsplit", "vsplit"):
            others = [p for p in pieces if p.role != WEIGHT]
            fresh = frozenset()
            for other in others:
                fresh |= _grad_tags(other.tags)
            if fresh:
                self._update_sink(fresh, node, "a parameter array is assembled")
            parents = self._combine_alternatives(p.cands for p in weights)
            return AbsVal(tags=self._as_candidate(frozenset(tags)), role=WEIGHT, cands=self._new_cands(node, parents),
                          obj=self._new_obj())
        return AbsVal(tags=frozenset(tags), role=STAT, obj=self._new_obj())

    def _reduce(self, array: AbsVal, axis: AbsVal | None, extra: list[AbsVal], name: str,
                node: ast.AST | None = None) -> AbsVal:
        tags = set(strip_candref(array.tags))
        for value in extra:
            tags |= strip_candref(value.tags)
        if array.role == SAMPLE:
            if axis is None:
                role = STAT
            elif axis.has_const():
                ax = axis.const
                role = STAT if (ax is None or ax == 0 or (isinstance(ax, tuple) and 0 in ax)) else SAMPLE
            else:
                role = SAMPLE
        elif array.role == SEQ:
            element = self._element_of(array)
            tags |= strip_candref(element.tags)
            role = STAT
        else:
            role = STAT if array.role != CONST else CONST
        multi = name in ("argsort",)
        final = _index_tags(frozenset(tags)) if name in INDEX_VALUED else frozenset(tags)
        final = self._readout_tag(node, array, axis, role, final, name)
        return AbsVal(tags=final, role=role if tags or role != CONST else CONST, obj=self._new_obj(), multi=multi)

    def _linalg(self, name: str, args: list[AbsVal], kwargs: dict[str, AbsVal], node: ast.AST) -> AbsVal:
        first = args[0] if args else AbsVal()
        if name in LINALG_SOLVERS:
            target = args[1] if len(args) > 1 else kwargs.get("b", AbsVal())
            solved = self._solver(name, first, target, node)
            if name == "lstsq":
                return AbsVal(role=SEQ, elems=(solved, AbsVal(role=STAT, tags=solved.tags), AbsVal(role=CONST),
                                               AbsVal(role=STAT, tags=strip_candref(first.tags))),
                              length=4, default=solved, tags=solved.tags)
            return solved
        if name in LINALG_INVERSES:
            site = self._site(node, "inv")
            self._register_site(site, "inverse", node, tainted_input=is_tainted(first.tags))
            return AbsVal(tags=strip_candref(first.tags) | {f"inv:{site}"}, role=STAT, obj=self._new_obj())
        if name == "multi_dot" and first.elems:
            result = first.elems[0]
            for operand in first.elems[1:]:
                result = self._matmul(result, operand, node)
            return result
        tags = strip_candref(first.tags)
        if name in LINALG_TUPLES:
            count = LINALG_TUPLES[name]
            compute_uv = kwargs.get("compute_uv")
            if name == "svd" and compute_uv is not None and compute_uv.const is False:
                return AbsVal(tags=tags, role=STAT, obj=self._new_obj())
            elems = []
            for i in range(count):
                role = SAMPLE if (name == "svd" and i == 0 and first.role == SAMPLE) else STAT
                elems.append(AbsVal(tags=tags, role=role, obj=self._new_obj()))
            return AbsVal(role=SEQ, elems=tuple(elems), length=count, default=join_all(elems), tags=tags)
        return AbsVal(tags=tags, role=STAT, obj=self._new_obj())

    def _solver(self, kind: str, design: AbsVal, target: AbsVal, node: ast.AST) -> AbsVal:
        site = self._site(node, "solver")
        if LABEL in target.tags:
            self._flag(V_SOLVER, node, f"{kind} target is derived from evidence labels (a closed-form fit)")
        elif LABEL_BUILT in target.tags:
            self._flag(S_SOLVER_SOFT, node, f"{kind} target depends on labels only through a label-built candidate "
                                            "setting")
        elif is_tainted(design.tags):
            self._flag(S_SOLVER_SOFT, node, f"{kind} design matrix is label-dependent while its target is not "
                                            "(possibly a fit onto constructed targets)")
        elif FAILSET in target.tags and target.role != SAMPLE:
            self._flag(S_SOLVER_SOFT, node, f"{kind} target is built from the x_fail split (every x_fail example is "
                                            "one the base model gets wrong, so membership is label-derived): "
                                            "possibly a linear probe of failure membership")
        tags = strip_candref(design.tags | target.tags)
        if not (is_tainted(design.tags) or is_tainted(target.tags) or LABEL_BUILT in tags):
            self._register_site(site, "solver", node)
            tags = tags | {f"solver:{site}"}
        return AbsVal(tags=tags, role=STAT, obj=self._new_obj())

    # ----------------------------------------------------------- methods

    def call_method(self, recv: AbsVal, name: str, args: list[AbsVal], kwargs: dict[str, AbsVal],
                    node: ast.AST) -> AbsVal:
        role = recv.role
        if role == RNG:
            if name == "shuffle" and args:
                self._mutate_value(args[0], self._random_value([args[0]], {}, args[0]), node)
                return AbsVal(role=CONST, const=None)
            if name in ("choice", "permutation") and args and args[0].role in (SEQ, WEIGHT, SAMPLE, STAT):
                base = self._element_of(args[0]) if name == "choice" else args[0]
                return self._random_value(args, kwargs, base).plus(strip_candref(base.tags))
            return self._random_value(args, kwargs)
        if role == FILE:
            return self._file_method(recv, name, args, node)
        if role == PATH:
            return self._path_method(recv, name, args, kwargs, node)
        if role == GLOBALS:
            if name == "update" and args:
                source = args[0]
                if source.role == EVID_C:
                    for key in (*self.label_keys, *EVIDENCE_DATA_KEYS):
                        self.module.vars[key] = self._evidence_item(source, key)
                elif source.items:
                    for key, value in source.items.items():
                        if key.isidentifier():
                            self.module.vars[key] = value
                if source.role not in (EVID_C,) and (source.default is not None or not source.items):
                    self.dynamic_tags = self.dynamic_tags | strip_candref(self._deep_tags(source))
                for key, value in kwargs.items():
                    self.module.vars[key] = value
                return AbsVal(role=CONST, const=None)
            if name == "get" and args and isinstance(args[0].const, str):
                return self.module.vars.get(args[0].const, AbsVal(tags=self.dynamic_tags))
            return AbsVal(tags=self.dynamic_tags)
        if role in (DICT, PDICT, MODEL_C, EVID_C):
            return self._dict_method(recv, name, args, kwargs, node)
        if role == SEQ:
            return self._seq_method(recv, name, args, kwargs, node)
        if role in (WEIGHT, SAMPLE, STAT, OTHER, CONST):
            return self._array_method(recv, name, args, kwargs, node)
        return self.call_unknown(AbsVal(role=METHOD, qual=name, recv=recv), args, kwargs, node)

    def _file_method(self, recv: AbsVal, name: str, args: list[AbsVal], node: ast.AST) -> AbsVal:
        if name in ("read", "readline", "readlines"):
            stored = self._file_read(recv.path)
            tags = stored.tags | self._path_variance(recv)
            return AbsVal(role=CONST if not tags else STAT, tags=tags, path=recv.path)
        if name in ("write", "writelines") and args:
            self._file_write(recv.path, args[0], node)
            return AbsVal(role=CONST)
        if name in ("__enter__",):
            return recv
        return AbsVal(role=CONST)

    def _path_method(self, recv: AbsVal, name: str, args: list[AbsVal], kwargs: dict[str, AbsVal],
                     node: ast.AST) -> AbsVal:
        if name in ("read_text", "read_bytes"):
            stored = self._file_read(recv.path)
            return AbsVal(role=CONST if not stored.tags else STAT, tags=stored.tags, path=recv.path)
        if name in ("write_text", "write_bytes") and args:
            self._file_write(recv.path, args[0], node)
            return AbsVal(role=CONST)
        if name == "open":
            mode = args[0] if args else kwargs.get("mode")
            return AbsVal(role=FILE, path=recv.path, mode=mode.const if mode is not None and isinstance(mode.const, str) else "r",
                          obj=self._new_obj())
        if name in ("joinpath", "with_name", "with_suffix", "resolve", "absolute", "expanduser"):
            if name == "joinpath":
                return self._path_join([recv, *args])
            return AbsVal(role=PATH, path=recv.path)
        if name in ("glob", "iterdir", "rglob"):
            return AbsVal(role=SEQ, default=AbsVal(role=PATH, path=(recv.path or "") + "/*"))
        return AbsVal(role=CONST)

    def _dict_method(self, recv: AbsVal, name: str, args: list[AbsVal], kwargs: dict[str, AbsVal],
                     node: ast.AST) -> AbsVal:
        exact = recv.items is not None and recv.default is None and recv.role in (DICT, PDICT)
        if recv.role in (MODEL_C, EVID_C) and name in ("keys", "values", "items", "get"):
            source, mode = (recv, name) if recv.role == EVID_C else (None, None)
            if name == "keys":
                return AbsVal(role=SEQ, default=AbsVal(role=CONST), obj=self._new_obj(), recv=source, mode=mode)
            value = self._model_item(recv, NOCONST) if recv.role == MODEL_C else self._evidence_item(recv, None)
            if name == "get":
                key = args[0] if args else AbsVal()
                return self._subscript(recv, key, node, ast.Constant(value=key.const if key.has_const() else None))
            if name == "values":
                return AbsVal(role=SEQ, default=value, tags=strip_candref(value.tags), obj=self._new_obj())
            pair = self._tuple([AbsVal(role=CONST), value])
            return AbsVal(role=SEQ, default=pair, tags=pair.tags, obj=self._new_obj(), recv=source, mode=mode)
        if name == "keys":
            elems = tuple(AbsVal(role=CONST, const=k) for k in recv.items) if exact else None  # type: ignore[union-attr]
            return AbsVal(role=SEQ, elems=elems, length=len(elems) if elems is not None else recv.length,
                          default=AbsVal(role=CONST), obj=self._new_obj())
        if name == "values":
            values = list((recv.items or {}).values()) + ([recv.default] if recv.default is not None else [])
            elems = tuple(recv.items.values()) if exact else None  # type: ignore[union-attr]
            tags = strip_candref(recv.tags) | frozenset().union(*(candref_tags(frozenset(alternatives(v))) for v in values)) if values else frozenset()
            return AbsVal(role=SEQ, elems=elems, length=len(elems) if elems is not None else recv.length,
                          default=join_all(values) if values else None, tags=tags, obj=self._new_obj())
        if name == "items":
            values = list((recv.items or {}).values()) + ([recv.default] if recv.default is not None else [])
            elems = tuple(self._tuple([AbsVal(role=CONST, const=k), v]) for k, v in recv.items.items()) if exact else None  # type: ignore[union-attr]
            pair = self._tuple([AbsVal(role=CONST), join_all(values) if values else AbsVal()])
            return AbsVal(role=SEQ, elems=elems, length=len(elems) if elems is not None else recv.length,
                          default=pair, tags=pair.tags, obj=self._new_obj())
        if name in ("get", "pop", "setdefault"):
            key = args[0] if args else AbsVal()
            fallback = args[1] if len(args) > 1 else None
            found = self._subscript(recv, key, node, ast.Constant(value=key.const if key.has_const() else None))
            if fallback is not None and not (isinstance(key.const, str) and recv.items and key.const in recv.items):
                found = join(found, fallback)
            if name in ("pop", "setdefault") and recv.obj is not None:
                items = dict(recv.items or {})
                if name == "pop" and isinstance(key.const, str):
                    items.pop(key.const, None)
                elif name == "setdefault" and isinstance(key.const, str) and fallback is not None:
                    items.setdefault(key.const, fallback)
                default = recv.default
                if name == "setdefault" and not isinstance(key.const, str) and fallback is not None:
                    # A2 (2026-11-04): d.setdefault(k, []).append(v) with a key the audit cannot resolve. The returned
                    # entry is the container's summary entry, so a mutation of it reaches the container (a weak
                    # update). Before, the entry was a fresh object and what was appended to it was lost.
                    obj = default.obj if default is not None and default.obj is not None else self._new_obj()
                    default = replace(join(default, fallback), obj=obj)
                    found = replace(join(found, fallback), obj=obj)
                self._rebind_obj(recv.obj, self._refresh_container(replace(recv, items=items, default=default,
                                                                           length=None)))
            return found
        if name == "copy":
            return replace(recv, obj=self._new_obj())
        if name == "update" and recv.obj is not None:
            items = dict(recv.items or {})
            default = recv.default
            for source in args:
                items.update(source.items or {})
                if source.default is not None:
                    default = join(default, source.default)
                if source.role == MODEL_C:
                    default = join(default, self._model_item(source, NOCONST))
            items.update(kwargs)
            role = recv.role
            if any(v.role == WEIGHT and PARAM in v.tags for v in items.values()):
                role = PDICT
            new = self._refresh_container(replace(recv, role=role, items=items, default=default, length=None))
            fresh = _grad_tags(new.tags) - _grad_tags(recv.tags)
            if fresh and role == PDICT:
                self._update_sink(fresh, node, "a parameter dictionary is updated")
            names = self._rebind_obj(recv.obj, new)
            self._record_event(names, new.tags, node)
            return AbsVal(role=CONST, const=None)
        if name == "clear" and recv.obj is not None:
            self._rebind_obj(recv.obj, replace(recv, items={}, default=None, length=0))
            return AbsVal(role=CONST, const=None)
        return self.call_unknown(AbsVal(role=METHOD, qual=name, recv=recv), args, kwargs, node)

    def _seq_method(self, recv: AbsVal, name: str, args: list[AbsVal], kwargs: dict[str, AbsVal],
                    node: ast.AST) -> AbsVal:
        if name in ("append", "add", "insert") and args:
            value = args[-1]
            elems = recv.elems + (value,) if (recv.elems is not None and name == "append" and not self.loops) else None
            length = recv.length + 1 if (recv.length is not None and not self.loops) else None
            tags = recv.tags | strip_candref(value.tags) | candref_tags(frozenset(alternatives(value)))
            new = replace(recv, elems=elems, length=length, default=join(recv.default, value), tags=tags)
            if recv.obj is not None:
                names = self._rebind_obj(recv.obj, new)
                self._record_event(names, value.tags, node)
            return AbsVal(role=CONST, const=None)
        if name == "extend" and args:
            source = args[0]
            element = self._element_of(source)
            elems = recv.elems + source.elems if (recv.elems is not None and source.elems is not None and not self.loops) else None
            length = recv.length + source.length if (recv.length is not None and source.length is not None and not self.loops) else None
            tags = recv.tags | strip_candref(self._deep_tags(source)) | candref_tags(frozenset(alternatives(element)))
            new = replace(recv, elems=elems, length=length, default=join(recv.default, element), tags=tags)
            if recv.obj is not None:
                names = self._rebind_obj(recv.obj, new)
                self._record_event(names, element.tags, node)
            return AbsVal(role=CONST, const=None)
        if name == "pop":
            element = self._element_of(recv)
            if recv.obj is not None:
                self._rebind_obj(recv.obj, replace(recv, elems=None, length=None))
            return element
        if name == "sort":
            key = kwargs.get("key")
            if key is not None:
                _, scored = self._implicit_loop(node, recv, key, "_s")
                self._best_keeping(recv, scored.tags)
                if recv.obj is not None:
                    self._rebind_obj(recv.obj, recv.plus(strip_candref(scored.tags)))
            return AbsVal(role=CONST, const=None)
        if name in ("copy",):
            return replace(recv, obj=self._new_obj())
        if name in ("index", "count"):
            return AbsVal(role=CONST, tags=strip_candref(recv.tags))
        if name in ("reverse", "remove", "clear"):
            if recv.obj is not None:
                self._rebind_obj(recv.obj, replace(recv, elems=None, length=None if name != "clear" else 0))
            return AbsVal(role=CONST, const=None)
        if name == "join" and recv.role == SEQ:
            return AbsVal(role=CONST, tags=strip_candref(self._deep_tags(recv)))
        return self.call_unknown(AbsVal(role=METHOD, qual=name, recv=recv), args, kwargs, node)

    def _array_method(self, recv: AbsVal, name: str, args: list[AbsVal], kwargs: dict[str, AbsVal],
                      node: ast.AST) -> AbsVal:
        if recv.role == CONST and isinstance(recv.const, str):
            if name == "join" and args:
                source = args[0]
                if source.elems is not None and all(isinstance(e.const, str) for e in source.elems):
                    text = recv.const.join(e.const for e in source.elems)
                    return AbsVal(role=CONST, const=text, path=text)
                return AbsVal(role=STAT if self._deep_tags(source) else CONST, tags=strip_candref(self._deep_tags(source)))
            if name == "format":
                tags = self._text_tags([(self._deep_tags(a), a.role) for a in [*args, *kwargs.values()]])
            else:
                tags = frozenset().union(*(strip_candref(self._deep_tags(a)) for a in args)) if args else frozenset()
            return AbsVal(role=CONST if not tags else STAT, tags=tags)
        if name == "copy":
            return replace(recv, obj=self._new_obj(), evaldiff=False)
        if name in ("astype", "view", "reshape", "ravel", "flatten", "squeeze", "transpose", "swapaxes", "conj",
                    "byteswap", "__array__"):
            alias = name in ("view", "reshape", "ravel", "squeeze", "transpose", "swapaxes")
            return replace(recv, obj=recv.obj if alias else self._new_obj(), evaldiff=False)
        if name in NP_REDUCE:
            axis = kwargs.get("axis", args[0] if args else None)
            return self._reduce(recv, axis, [], name, node)
        if name == "dot" and args:
            return self._matmul(recv, args[0], node)
        if name == "item":
            return AbsVal(role=STAT if recv.tags else CONST, tags=strip_candref(recv.tags))
        if name == "tolist":
            element = AbsVal(role=STAT, tags=strip_candref(recv.tags))
            return AbsVal(role=SEQ, tags=strip_candref(recv.tags), default=element, length=recv.length, obj=self._new_obj())
        if name in ("argsort", "nonzero", "argpartition", "searchsorted"):
            index_tags = _index_tags(strip_candref(
                recv.tags | frozenset().union(*(a.tags for a in args)) if args else recv.tags))
            return AbsVal(tags=_with_lsize(index_tags) if name == "nonzero" else index_tags,
                          role=SAMPLE if recv.role == SAMPLE and args and _const_int(args[0]) not in (None, 0) else STAT,
                          multi=True, obj=self._new_obj())
        if name in ("round", "clip", "repeat", "take", "compress", "diagonal", "cumsum", "cumprod", "__abs__"):
            tags = recv.tags | frozenset().union(*(a.tags for a in args)) if args else recv.tags
            if recv.role == WEIGHT and PARAM in recv.tags:
                return replace(recv, tags=self._as_candidate(strip_candref(tags)), cands=self._new_cands(node, recv.cands),
                               pristine=False, obj=self._new_obj())
            return replace(recv, tags=strip_candref(tags), obj=self._new_obj(), pristine=False)
        if name in ("fill", "sort", "put", "itemset", "resize", "partition", "setfield"):
            values = args[1:] if name == "put" else args
            tags = frozenset().union(*(strip_candref(a.tags) for a in values)) if values else frozenset()
            self._mutate_value(recv, AbsVal(tags=tags, role=recv.role), node)
            return AbsVal(role=CONST, const=None)
        if recv.role == OTHER and self._setting_and_data([*args, *kwargs.values()]):
            # ``model.forward(p, x)`` on an object the audit cannot see into: the call_unknown semantics apply.
            return self.call_unknown(AbsVal(role=METHOD, qual=name, recv=recv), args, kwargs, node)
        tags = recv.tags | frozenset().union(*(a.tags for a in args)) if args else recv.tags
        return AbsVal(tags=strip_candref(tags), role=recv.role if recv.role != WEIGHT else STAT, obj=self._new_obj())

    @staticmethod
    def _setting_and_data(values: list[AbsVal]) -> bool:
        """True when the values include a parameter setting and evidence-derived data (a forward evaluation's inputs)."""
        weights = any(v.role in (WEIGHT, PDICT, MODEL_C) for v in values)
        return weights and any(v.role not in (WEIGHT, PDICT, MODEL_C) and (EVID in v.tags or v.role == SAMPLE)
                               for v in values)

    def _seq_op(self, op: ast.operator, left: AbsVal, right: AbsVal) -> AbsVal:
        tags = left.tags | right.tags
        if isinstance(op, ast.Add) and left.role == SEQ and right.role == SEQ:
            elems = left.elems + right.elems if left.elems is not None and right.elems is not None else None
            length = left.length + right.length if left.length is not None and right.length is not None else None
            return AbsVal(role=SEQ, elems=elems, length=length, default=join(left.default, right.default), tags=tags,
                          obj=self._new_obj())
        if isinstance(op, ast.Mult):
            seq, count = (left, right) if left.role == SEQ else (right, left)
            times = _const_int(count)
            length = seq.length * times if seq.length is not None and times is not None else None
            elems = seq.elems * times if seq.elems is not None and times is not None and times <= 64 else None
            return AbsVal(role=SEQ, elems=elems, length=length, default=seq.default, tags=tags, obj=self._new_obj())
        return AbsVal(tags=strip_candref(tags), role=STAT, obj=self._new_obj())

    # ---------------------------------------------------------- finalize

    def finalize(self, external: Sequence[RawSetting] = ()) -> InterpResult:
        """Session-level rules (unused gradients, gradient-like names, the candidate count) and the result.

        ``external`` are counted candidates scored outside the cells (distinct patches sent to the predict tool); they
        join the count and the V-SEARCH-COUNT decision.
        """
        violation_lines = {(f.cell, f.line) for f in self.flags.values() if f.rule in (V_BACKPROP, V_FINITE_DIFF)}
        for site, info in self.sites.items():
            if info.kind in ("grad", "fd") and not info.used:
                what = ("a hand-written gradient (an error signal contracted with activations)" if info.kind == "grad"
                        else "a finite-difference quotient of label-dependent scores")
                self._flag_at(S_GRAD_UNUSED, info.cell, info.line, info.excerpt,
                              f"{what} is computed here, but no parameter update or patch written from it was observed")
        for (cell, line), (name, excerpt) in self.grad_names.items():
            if (cell, line) in violation_lines:
                continue
            self._flag_at(S_GRAD_NAME, cell, line, excerpt,
                          f"the gradient-like name {name} is bound to a label-dependent value")
        settings, total, lower = self._count_candidates(external)
        config_total, config_lower = self._count_configurations(settings)
        return InterpResult(
            flags=sorted(self.flags.values(), key=lambda f: (f.cell, f.line, f.rule)),
            notes=list(self.notes),
            settings=settings,
            candidate_count=total,
            lower_bound=lower,
            label_cells=sorted(self.label_cells),
            unparsed=list(self.unparsed),
            config_count=config_total,
            config_lower_bound=config_lower,
        )

    def _multiplier(self, sig: str) -> tuple[int | None, list[dict[str, object]], bool]:
        multiplier: int | None = 1
        loops: list[dict[str, object]] = []
        random = False
        if sig == UNBOUNDED_LABEL:
            return None, loops, False
        for part in sig.split(";"):
            if not part:
                continue
            if part == "rng":
                random = True
                continue
            loop, kind = part.split("=")
            trip = self.loop_trips.get(loop)
            cell, line, excerpt, loop_kind = self.loop_meta.get(loop, (-1, 0, "", "loop"))
            loops.append({"cell": cell, "line": line, "excerpt": excerpt, "loop_kind": loop_kind, "trip": trip,
                          "variance": {"v": "value", "i": "single-element index", "m": "multi-element index",
                                       "f": "evidence rows (fold or view)", "d": "decision-rule parameter"}[kind]})
            if multiplier is not None:
                multiplier = None if trip is None else multiplier * trip
        return multiplier, loops, random

    def _partial_histories(self) -> set[str]:
        """Audit amendment A2 (2026-11-04): setting names that are a strict part of another name scored at the same
        statement, where all sites of the larger name were built in one call of a helper.

        A forward helper that returns every layer's activations in one dict (``run(params, x)['logits']``) hands the
        scoring statement the outputs together with earlier layers' activations, whose histories stop at those layers.
        They are the same setting seen part way through the forward pass, not separate settings.
        """
        by_statement: dict[tuple[int, int], list[str]] = {}
        for key, info in self.settings.items():
            by_statement.setdefault((info.cell, info.line), []).append(key)
        partial: set[str] = set()
        for keys in by_statement.values():
            if len(keys) < 2:
                continue
            for key in keys:
                parts = {p for p in key.split("+") if p}
                for other in keys:
                    other_parts = {p for p in other.split("+") if p}
                    if not parts < other_parts or not self.settings[key].sigs <= self.settings[other].sigs:
                        continue
                    contexts = {p[p.index("@"):] if "@" in p else "" for p in other_parts}
                    if len(contexts) == 1 and "" not in contexts:
                        partial.add(key)
                        break
        return partial

    def _config_multiplier(self, sig: str) -> int | None:
        """Configurations one evaluation signature stands for (A2, 2026-11-04): the product of the trips of its
        value and multi-element loops. Fold, view and class loops (``f``) and single-element sweeps (``i``) count
        once; None when such a loop bound is unknown."""
        if sig == UNBOUNDED_LABEL:
            return None
        multiplier: int | None = 1
        for part in sig.split(";"):
            if "=" not in part:
                continue
            loop, kind = part.split("=")
            if kind not in CONFIG_KINDS:
                continue
            trip = self.loop_trips.get(loop)
            multiplier = None if (trip is None or multiplier is None) else multiplier * trip
        return multiplier

    def _config_id(self, key: str) -> str:
        """The configuration a setting name belongs to (A2, 2026-11-04): its sites without call locations."""
        parts = {self.parts[p].config if p in self.parts and self.parts[p].config else p for p in key.split("+") if p}
        return "+".join(sorted(parts))

    def _count_configurations(self, records: Sequence[RawSetting]) -> tuple[int, bool]:
        """Audit amendment A2 (2026-11-04), configuration count (C): distinct configurations among counted records.

        Records that share a configuration (the same rule rebuilt with the same arguments at another statement)
        count once, with the largest multiplicity among them. Reported only; the verdict uses the literal count.
        """
        best: dict[str, int | None] = {}
        for record in records:
            if not record.counted:
                continue
            config = record.config_id or record.setting_id
            mult = record.config_count
            if config in best:
                previous = best[config]
                best[config] = None if (previous is None or mult is None) else max(previous, mult)
            else:
                best[config] = mult
        total = 0
        lower = False
        for mult in best.values():
            if mult is None:
                total += self.limit + 1
                lower = True
            else:
                total += mult
        return total, lower

    def _count_candidates(self, external: Sequence[RawSetting] = ()) -> tuple[list[RawSetting], int, bool]:
        """Distinct candidate settings scored against labels, with loop multiplicities, plus ``external`` ones.

        Every distinct setting name counts. A name produced only by joining alternative histories at one evaluation
        is not counted when each of those alternatives was scored on its own. A single-element sweep (index variance
        only, no value variance, nothing kept by score, no random draw) is an ablation for attribution and is
        excluded, as contract 5.5 allows. An unknown loop bound counts as more than the limit.

        Audit amendment A2 (2026-11-04): a setting built from loop-selected evidence rows (``f``, a leave-one-out fold
        or a view) is a separate setting per iteration, so such a loop is never an ablation. Each record also carries
        its configuration (``config_id``, ``config_count``) for the configuration count (C).
        """
        scored_alone = set(self.settings) & self.individual
        records: list[RawSetting] = []
        total = 0
        lower = False
        partial = self._partial_histories()
        for key, info in self.settings.items():
            if key in partial:
                self.notes.append(("N-CANDIDATE-PARTIAL", info.cell, info.line,
                                   "a partial modification history of a setting scored at the same statement, built "
                                   "within one call of a helper (the activations of earlier layers held with the "
                                   "outputs); it is not counted apart from the full setting"))
                continue
            covered = self.joined.get(key)
            if key not in self.individual and covered and covered <= scored_alone:
                self.notes.append(("N-CANDIDATE-COVERED", info.cell, info.line,
                                   "a setting whose possible modification histories were each scored on their own "
                                   "is scored again here; it is not counted twice"))
                continue
            best_mult: int | None = 0
            best_loops: list[dict[str, object]] = []
            best_sig = ""
            random = False
            config_mult: int | None = 0
            for sig in sorted(info.sigs):
                mult, loops, rnd = self._multiplier(sig)
                random = random or rnd
                if best_mult is not None and (mult is None or mult > best_mult):
                    best_mult, best_loops, best_sig = mult, loops, sig
                sig_config = self._config_multiplier(sig)
                if config_mult is not None and (sig_config is None or sig_config > config_mult):
                    config_mult = sig_config
            kinds = {part.split("=")[1] for part in best_sig.split(";") if "=" in part}
            best_kept = any(self.parts[p].best_kept for p in key.split("+") if p in self.parts)
            ablation = "i" in kinds and not kinds & COUNTED_KINDS and not best_kept and not random
            kind = "ablation" if ablation else ("loop" if best_loops else "explicit")
            counted = not ablation
            if counted:
                if best_mult is None:
                    total += self.limit + 1
                    lower = True
                else:
                    total += best_mult
            else:
                trips = [loop["trip"] for loop in best_loops]
                self.notes.append(("N-ABLATION-SWEEP", info.cell, info.line,
                                   "single-element ablation sweep scored against label-dependent values "
                                   f"(loop bounds {trips}); excluded from the candidate count as contract 5.5 allows"))
            records.append(RawSetting(key, info.cell, info.line, info.excerpt, best_mult, counted, kind, random,
                                      best_kept, tuple(best_loops), self._config_id(key),
                                      config_mult if counted else 0))
        for record in external:
            records.append(record)
            if record.counted:
                total += record.count if record.count is not None else self.limit + 1
                lower = lower or record.count is None
        records.sort(key=lambda r: (r.cell, r.line, r.setting_id))
        if total > self.limit:
            counted_records = [r for r in records if r.counted]
            top = sorted(counted_records, key=lambda r: (r.cell == PREDICT_CELL,
                                                         -(r.count if r.count is not None else 10**9)))[:5]
            first = top[0]
            predicted = [r for r in counted_records if r.cell == PREDICT_CELL]

            def where(r: RawSetting) -> str:
                return f"tool log line {r.line} (predict)" if r.cell == PREDICT_CELL else f"cell {r.cell} line {r.line}"

            listing = "; ".join(
                f"{where(r)}: {r.count if r.count is not None else 'unbounded'} ({r.kind}"
                f"{', random' if r.random else ''}{', best kept' if r.best_kept else ''})" for r in top)
            if predicted:
                listing += f"; {len(predicted)} distinct patches sent to the predict tool in all"
            self._flag_at(V_SEARCH, first.cell, first.line, first.excerpt,
                          f"{total}{' or more' if lower else ''} distinct candidate settings were scored against "
                          f"label-dependent values (limit {self.limit}): {listing}")
        else:
            for record in records:
                if record.counted and record.random and record.best_kept:
                    self._flag_at(S_RANDOM_SMALL, record.cell, record.line, record.excerpt,
                                  f"{record.count} random candidate settings scored against label-dependent values with "
                                  "the best kept (within the limit, but a random search)")
        return records, total, lower
