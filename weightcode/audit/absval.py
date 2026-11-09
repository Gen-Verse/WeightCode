"""Abstract values of the audit interpreter: provenance tags, roles and the join used at control-flow merges.

An ``AbsVal`` over-approximates every concrete value an expression can take. Its ``tags`` record provenance:

- ``label``: derived from an evidence label array (``y_fail``, ``y_rand``, any other ``y*`` key, or an evidence key
  the audit cannot resolve); ``label_built``: a candidate parameter setting built from label-dependent statistics.
- ``param``: depends on model weights; ``evid``: depends on evidence arrays; ``rng``: depends on a random draw;
  ``hard``: an index or hard prediction (argmax, argsort, nonzero), not a model output in value.
- ``lsize``: the value's length depends on labels (a selection by a label-dependent mask or index), so its ``len``,
  ``shape`` and ``size`` are label-dependent (a class count); ``failset``: derived from the ``x_fail`` split. Every
  ``x_fail`` example is one the base model gets wrong, so membership in it is label-derived information; it only
  raises SOFT flags at solvers and inverses.
- ``err:<site>`` (per-sample model output minus a label encoding), ``grad:<site>`` (an error signal contracted with
  per-sample activations), ``fd:<site>`` (a finite-difference quotient of label-dependent scores) and ``sd:<site>``
  (a difference of label-dependent scores of two different parameter settings). A parameter value or patch that
  carries one of them is a gradient-rule violation; selections, masks and text made from them do not carry them.
- ``solver:<site>`` and ``inv:<site>``: results of solver or inverse calls whose inputs were label-free.
- loop-scoped tags, all parameterised by a loop id ``L``: ``lv:L`` (varies in value from iteration to iteration),
  ``li:L`` / ``lim:L`` (varies only in which single / multiple elements an index selects), ``enum:L`` (the loop
  enumerates known candidate settings), ``carry:L:n`` (derived from the value variable ``n`` held when the iteration
  began), ``evalc:L:n`` (an evaluation of the carried quantity ``n`` through the model or on evidence data) and
  ``obj:L:n`` (a label-dependent evaluation of ``n``, i.e. an objective that feeds back into ``n``).
- ``eval:<setting>#<sig>``: the value is computed from a forward evaluation of candidate setting ``<setting>``;
  ``<sig>`` lists the active loops in which that setting varied. ``candref:<setting>``: a container holds it.

A candidate setting is named by its modification history: the ``+``-joined sorted set of modification sites
(``c<cell>.<line>.<col>`` plus a call-context hash). ``cands`` holds the alternative histories a value may have;
alternatives come from control-flow merges or from containers of several candidates.

``role`` says what kind of object the value is (weight array, parameter dict, per-sample data, statistic, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

# Roles.
WEIGHT = "weight"
PDICT = "pdict"
SAMPLE = "sample"
STAT = "stat"
CONST = "const"
MODEL_C = "model_c"
EVID_C = "evid_c"
FUNC = "func"
MODULE = "module"
METHOD = "method"
RNG = "rng"
SEQ = "seq"
DICT = "dict"
FILE = "file"
PATH = "path"
GLOBALS = "globals"
INSTANCE = "instance"
OTHER = "other"

ROLE_RANK = {
    WEIGHT: 12, PDICT: 11, MODEL_C: 10, EVID_C: 10, SAMPLE: 9, STAT: 8, SEQ: 7, DICT: 7, INSTANCE: 6, RNG: 5,
    FILE: 5, PATH: 5, FUNC: 4, METHOD: 4, MODULE: 4, GLOBALS: 4, OTHER: 2, CONST: 0,
}

ARRAY_ROLES = frozenset({WEIGHT, SAMPLE, STAT, OTHER})
CONTAINER_ROLES = frozenset({SEQ, DICT, PDICT, INSTANCE})

# Information tags.
LABEL = "label"
LABEL_BUILT = "label_built"
PARAM = "param"
EVID = "evid"
RANDOM = "rng"
HARD = "hard"
LSIZE = "lsize"
FAILSET = "failset"


class _NoConst:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<noconst>"


NOCONST: Any = _NoConst()


@dataclass(frozen=True, eq=False)
class AbsVal:
    """Abstract value. Instances are immutable; use ``dataclasses.replace`` or the helpers below."""

    tags: frozenset[str] = frozenset()
    role: str = OTHER
    pristine: bool = False
    cands: frozenset[str] = frozenset()
    const: Any = NOCONST
    length: int | None = None
    elems: tuple["AbsVal", ...] | None = None
    items: dict[str, "AbsVal"] | None = None
    default: "AbsVal | None" = None
    qual: str | None = None
    fn: Any = None
    recv: "AbsVal | None" = None
    obj: int | None = None
    basis: bool = False
    multi: bool = False
    evaldiff: bool = False
    path: str | None = None
    mode: str | None = None
    # Loops over which this value varies only because a loop-varying key selects one tensor of a parameter setting
    # that itself does not vary with them (``p[name]`` in a loop over layer names): the setting is one candidate.
    keyvar: frozenset[str] = frozenset()

    def has_const(self) -> bool:
        return self.const is not NOCONST

    def plus(self, extra: frozenset[str] | set[str]) -> "AbsVal":
        if not extra or extra <= self.tags:
            return self
        return replace(self, tags=self.tags | frozenset(extra))


def is_tainted(tags: frozenset[str]) -> bool:
    """Label-dependent in the sense of contract 5.5: derived from the evidence label arrays."""
    return LABEL in tags


def tags_with_prefix(tags: frozenset[str], prefix: str) -> list[str]:
    return [tag for tag in tags if tag.startswith(prefix)]


def strip_candref(tags: frozenset[str]) -> frozenset[str]:
    if not any(tag.startswith("candref:") for tag in tags):
        return tags
    return frozenset(tag for tag in tags if not tag.startswith("candref:"))


def candref_tags(cands: frozenset[str]) -> frozenset[str]:
    return frozenset(f"candref:{cand}" for cand in cands)


def alternatives(value: AbsVal) -> set[str]:
    """Candidate settings a value is or holds: its own alternatives plus those of contained candidates."""
    found = set(value.cands)
    found.update(tag[len("candref:"):] for tag in value.tags if tag.startswith("candref:"))
    return found


def _same_const(a: Any, b: Any) -> bool:
    # Constants are Python literals (numbers, strings, bytes, bool, None, tuples of these), so == never raises.
    return type(a) is type(b) and bool(a == b)


def join(a: AbsVal | None, b: AbsVal | None) -> AbsVal:
    """Least upper bound of two abstract values (control-flow merge)."""
    if a is None and b is None:
        return AbsVal()
    if a is None:
        return b  # type: ignore[return-value]
    if b is None or a is b:
        return a
    role = a.role if a.role == b.role else max((a.role, b.role), key=lambda r: ROLE_RANK.get(r, 1))
    const = a.const if a.has_const() and b.has_const() and _same_const(a.const, b.const) else NOCONST
    elems = None
    if a.elems is not None and b.elems is not None and len(a.elems) == len(b.elems):
        elems = tuple(join(x, y) for x, y in zip(a.elems, b.elems))
    items = None
    if a.items is not None or b.items is not None:
        items = dict(a.items or {})
        for key, value in (b.items or {}).items():
            items[key] = join(items[key], value) if key in items else value
    default = join(a.default, b.default) if (a.default is not None or b.default is not None) else None
    return AbsVal(
        tags=a.tags | b.tags,
        role=role,
        pristine=a.pristine and b.pristine,
        cands=a.cands | b.cands,
        const=const,
        length=a.length if a.length == b.length else None,
        elems=elems,
        items=items,
        default=default,
        qual=a.qual if a.qual == b.qual else None,
        fn=a.fn if a.fn is b.fn else (a.fn if a.fn is not None else b.fn),
        recv=a.recv if a.recv is b.recv else None,
        obj=a.obj if a.obj == b.obj else None,
        basis=a.basis and b.basis,
        multi=a.multi or b.multi,
        evaldiff=False,
        path=a.path if a.path == b.path else None,
        mode=a.mode if a.mode == b.mode else None,
        keyvar=combine_keyvar((a, b)),
    )


def combine_keyvar(values: Iterable[AbsVal]) -> frozenset[str]:
    """Loops over which a combination of values varies only by key selection (``AbsVal.keyvar``).

    A loop stays key-only when no value varies with it otherwise (an ``lv:`` tag without the loop in ``keyvar``).
    """
    keyed: set[str] = set()
    strict: set[str] = set()
    for value in values:
        keyed |= value.keyvar
        strict |= {tag[3:] for tag in value.tags if tag.startswith("lv:")} - value.keyvar
    return frozenset(keyed - strict)


def join_all(values: list[AbsVal]) -> AbsVal:
    result: AbsVal | None = None
    for value in values:
        result = value if result is None else join(result, value)
    return result if result is not None else AbsVal()


def signature(value: AbsVal | None, depth: int = 0) -> tuple[object, ...]:
    """Comparable summary used to detect a loop fixpoint."""
    if value is None:
        return ()
    items: tuple[object, ...] = ()
    if value.items is not None and depth < 2:
        items = tuple(sorted((key, signature(item, depth + 1)) for key, item in value.items.items()))
    default = signature(value.default, depth + 1) if value.default is not None and depth < 2 else ()
    elems = tuple(signature(item, depth + 1) for item in value.elems) if value.elems is not None and depth < 2 else ()
    return (value.role, value.tags, value.cands, value.pristine, value.length, value.obj, items, default, elems,
            value.keyvar)
