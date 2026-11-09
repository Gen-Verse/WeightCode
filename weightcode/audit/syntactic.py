"""Syntactic rules applied to each parsed cell, and the regex fallback for cells the audit cannot parse.

These rules need no data flow: imports of autograd or fitting libraries, autograd and optimiser API calls, ``.fit(``.
Names are resolved through the import aliases seen so far in the session, so ``import torch as T`` followed by
``T.autograd.grad(...)`` is recognised.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass

from weightcode.audit.report import S_GRAD_NAME, S_IMPORT, S_UNPARSED, V_AUTOGRAD, V_FIT, V_OPTIMIZER

FORBIDDEN_MODULES = frozenset({
    "torch", "torchvision", "functorch", "jax", "jaxlib", "flax", "optax", "haiku", "autograd", "tensorflow", "keras",
    "sklearn", "scipy", "statsmodels", "theano", "mxnet", "paddle", "chainer", "xgboost", "lightgbm",
})

AUTOGRAD_APIS = (
    "torch.autograd", "torch.func.grad", "torch.func.grad_and_value", "torch.func.vjp", "torch.func.jvp",
    "torch.func.jacrev", "torch.func.jacfwd", "torch.func.hessian", "functorch", "jax.grad", "jax.value_and_grad",
    "jax.jacfwd", "jax.jacrev", "jax.jacobian", "jax.hessian", "jax.vjp", "jax.jvp", "jax.linearize", "autograd.grad",
    "autograd.elementwise_grad", "autograd.value_and_grad", "autograd.jacobian", "autograd.hessian",
    "autograd.holomorphic_grad", "autograd.grad_named", "autograd.multigrad_dict", "tensorflow.GradientTape",
    "tensorflow.gradients", "mxnet.autograd", "paddle.grad", "paddle.autograd", "chainer.grad", "theano.grad",
    "theano.tensor.grad",
)
OPTIMIZER_APIS = (
    "torch.optim", "optax", "scipy.optimize", "tensorflow.keras.optimizers", "keras.optimizers", "flax.training",
    "sklearn.model_selection.cross_val_score", "sklearn.model_selection.cross_validate",
    "sklearn.model_selection.cross_val_predict", "sklearn.model_selection.GridSearchCV",
    "sklearn.model_selection.RandomizedSearchCV", "sklearn.model_selection.learning_curve",
    "sklearn.model_selection.validation_curve", "sklearn.model_selection.permutation_test_score",
)
NOT_OPTIMIZERS = frozenset({"scipy.optimize.linear_sum_assignment"})
FIT_ATTRS = frozenset({"fit", "partial_fit", "fit_transform", "fit_predict"})
AUTOGRAD_ATTR_CALLS = frozenset({"backward", "requires_grad_", "zero_grad"})


@dataclass(frozen=True)
class RawFlag:
    """A flag before it is numbered and attached to a cell."""

    rule: str
    line: int
    excerpt: str
    reason: str


def line_excerpt(lines: list[str], line: int, limit: int = 200) -> str:
    """Stripped source line ``line`` (1-based), clipped to ``limit`` characters."""
    if 1 <= line <= len(lines):
        text = lines[line - 1].strip()
    else:
        text = ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _matches(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def update_aliases(tree: ast.AST, aliases: dict[str, str]) -> None:
    """Record the import aliases bound by a cell: local name -> fully qualified module or object name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    aliases[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                if alias.name != "*":
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"


def dotted_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Fully qualified dotted name of ``a.b.c`` if its root is an imported alias, else None."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name) or current.id not in aliases:
        return None
    parts.append(aliases[current.id])
    return ".".join(reversed(parts))


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def syntactic_flags(tree: ast.AST, lines: list[str], aliases: dict[str, str]) -> list[RawFlag]:
    """Flags that follow from the syntax of one cell alone. ``aliases`` must already include this cell's imports."""
    flags: list[RawFlag] = []

    def add(rule: str, node: ast.AST, reason: str) -> None:
        line = getattr(node, "lineno", 1)
        flags.append(RawFlag(rule, line, line_excerpt(lines, line), reason))

    api_attribute_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    add(S_IMPORT, node, f"attempted import of {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level == 0 and root in FORBIDDEN_MODULES:
                add(S_IMPORT, node, f"attempted import from {node.module}")
        elif isinstance(node, ast.Call):
            func = node.func
            name = dotted_name(func, aliases)
            if name is not None:
                api_attribute_nodes.add(id(func))
                if _matches(name, AUTOGRAD_APIS):
                    add(V_AUTOGRAD, node, f"call to the autograd API {name}")
                elif _matches(name, OPTIMIZER_APIS) and name not in NOT_OPTIMIZERS:
                    add(V_OPTIMIZER, node, f"call to the optimiser or fitting API {name}")
                elif name in ("importlib.import_module", "builtins.__import__"):
                    target = _const_str(node.args[0] if node.args else None)
                    if target is not None and target.split(".")[0] in FORBIDDEN_MODULES:
                        add(S_IMPORT, node, f"attempted dynamic import of {target}")
            if isinstance(func, ast.Name) and func.id == "__import__":
                target = _const_str(node.args[0] if node.args else None)
                if target is not None and target.split(".")[0] in FORBIDDEN_MODULES:
                    add(S_IMPORT, node, f"attempted dynamic import of {target}")
            if isinstance(func, ast.Attribute):
                if func.attr in FIT_ATTRS:
                    add(V_FIT, node, f"a .{func.attr}( call fits parameters to data")
                elif func.attr in AUTOGRAD_ATTR_CALLS:
                    add(V_AUTOGRAD, node, f"a .{func.attr}() call is autograd or optimiser API use")
                elif func.attr == "gradient" and name is None:
                    add(S_GRAD_NAME, node, "a .gradient( call looks like an autograd tape")
            for keyword in node.keywords:
                if keyword.arg == "requires_grad":
                    add(V_AUTOGRAD, node, "requires_grad= asks an autograd library to track gradients")
        elif isinstance(node, ast.Attribute):
            if node.attr == "requires_grad":
                add(V_AUTOGRAD, node, ".requires_grad is autograd API use")
            elif node.attr == "grad" and id(node) not in api_attribute_nodes and dotted_name(node, aliases) is None:
                add(S_GRAD_NAME, node, ".grad attribute access looks like reading an autograd gradient")
    return flags


_REGEX_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (V_AUTOGRAD, re.compile(r"\.backward\s*\(|requires_grad|torch\.autograd|GradientTape|"
                            r"\b(?:jax|autograd)\s*\.\s*(?:grad|value_and_grad|jacfwd|jacrev|jacobian|hessian|vjp|jvp)\b"),
     "autograd API use (regex fallback)"),
    (V_FIT, re.compile(r"\.(?:fit|partial_fit|fit_transform|fit_predict)\s*\("), "a .fit( call (regex fallback)"),
    (V_OPTIMIZER, re.compile(r"torch\.optim|\boptax\.|scipy\.optimize|\boptimize\.minimize\b"),
     "optimiser API use (regex fallback)"),
    (S_IMPORT, re.compile(r"^\s*(?:import|from)\s+(?:" + "|".join(sorted(FORBIDDEN_MODULES)) + r")\b", re.MULTILINE),
     "attempted import of an autograd or fitting library (regex fallback)"),
)
_REGEX_SUSPICIOUS = (
    (re.compile(r"\b(?:lstsq|pinv|solve|inv|tensorsolve|polyfit)\s*\("), "solver or inverse call"),
    (re.compile(r"\b(?:for|while)\b"), "loop"),
    (re.compile(r"\b(?:exec|eval|compile|subprocess|os\.system)\b"), "dynamic code"),
)
_ASSIGNED = re.compile(r"^\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*(?:[-+*/@%&|^]|//|\*\*)?=(?!=)", re.MULTILINE)
_BOUND = re.compile(r"\b(?:for|as|def|class|import)\s+([A-Za-z_]\w*)")
_IDENT = re.compile(r"[A-Za-z_]\w*")


def _line_of(code: str, offset: int) -> int:
    return code.count("\n", 0, offset) + 1


def regex_scan(code: str) -> tuple[list[RawFlag], list[str]]:
    """Regex rules for an unparsable cell. Returns hard-rule flags and descriptions of suspicious patterns."""
    lines = code.splitlines()
    flags: list[RawFlag] = []
    for rule, pattern, reason in _REGEX_RULES:
        for match in pattern.finditer(code):
            line = _line_of(code, match.start())
            flags.append(RawFlag(rule, line, line_excerpt(lines, line), reason))
    suspicious = [label for pattern, label in _REGEX_SUSPICIOUS if pattern.search(code)]
    return flags, suspicious


def regex_bound_names(code: str) -> set[str]:
    """Names an unparsable cell probably binds (assignment targets, loop variables, defs, imports)."""
    names: set[str] = set()
    for match in _ASSIGNED.finditer(code):
        names.update(part.strip() for part in match.group(1).split(","))
    names.update(match.group(1) for match in _BOUND.finditer(code))
    return {name for name in names if name.isidentifier()}


def regex_mentioned_names(code: str) -> set[str]:
    """Identifiers mentioned anywhere in the cell text."""
    return set(_IDENT.findall(code))


UNPARSED_RULE = S_UNPARSED
