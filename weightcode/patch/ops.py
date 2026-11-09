"""Structural validation and float64 application of the ``weightcode.patch.v0`` ops (contract section 5.0).

``parse_patch`` checks the top-level object and turns every raw op into a validated op, logging each
problem it finds instead of stopping at the first. Ops that reference .npy files check the loaded arrays in
``bind``. ``apply`` then runs on float64 working copies of the tensors (see ``Workspace``).

Rules enforced here, beyond the contract table:

- Indices are non-negative integers (JSON integers); negative indices, floats and booleans are errors.
- ``rows`` and ``cols`` are non-empty lists of distinct indices, because a repeated index is ambiguous.
- ``index`` holds full index tuples, one integer per tensor axis. ``set`` with a repeated tuple is an
  error; ``add`` with repeated tuples accumulates them (``numpy.add.at`` semantics).
- ``slice`` has one entry per axis, each ``null`` (the whole axis) or ``[start, stop]`` with integers
  ``0 <= start < stop <= axis length``. The slice form takes exactly one of a finite scalar ``value`` or a
  ``values_file`` whose shape equals the block shape (no broadcasting).
- ``project_out`` normalises its direction to unit length and rejects zero, non-finite and wrongly sized
  directions; ``strength`` is in (0, 1]; ``bias`` is allowed with ``side: "out"`` only and must name a
  1-D tensor of length ``shape[0]`` other than ``tensor``.
- ``low_rank_add`` takes 2-D ``u`` [shape[0], r] and ``v`` [prod(shape[1:]), r], 1 <= r <= max_rank.
- ``tie`` takes ``axis`` 0 or 1 of the 2-D view (axis 1 needs rank >= 2), ``groups`` with one integer group id
  in [0, L) per position of that axis (L = its length), inline or as a file of integral values, and
  ``strength`` in (0, 1]. A position alone in its group is left unchanged.
- ``factor``, ``value``, ``values`` and inline directions must be finite numbers.
- Unknown fields are errors, at the top level and in every op. Ops never target rank-0 tensors.
- Products are computed without BLAS: matrix-vector products are exactly rounded sums (``math.fsum``)
  and ``u v^T`` adds the r outer products in a fixed order, so the child is bitwise reproducible across
  machines and BLAS thread settings. ``tie`` accumulates group sums with elementwise additions in ascending
  position order, which is reproducible for the same reason.
"""

from __future__ import annotations

import itertools
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from weightcode.patch.errors import MessageLog
from weightcode.patch.files import NPY_NAME_PATTERN, is_npy_name
from weightcode.patch.limits import Limits

SCHEMA = "weightcode.patch.v0"
OP_NAMES = ("scale_rows", "scale_cols", "set", "add", "project_out", "low_rank_add", "add_dense", "tie")
TOP_LEVEL_KEYS = ("schema", "parent_sha256", "ops", "hypothesis", "predicted_effect")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")

_QUOTED = 5
_SUM_CHUNK = 1 << 16

Shape = tuple[int, ...]


# ----------------------------------------------------------------------------------------------- helpers


def _short(value: object, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _summary(problems: Sequence[str]) -> str:
    if len(problems) <= _QUOTED:
        return "; ".join(problems)
    return "; ".join(problems[:_QUOTED]) + f"; and {len(problems) - _QUOTED} more"


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _finite_float(value: object) -> float | None:
    """The value as a float if it is a finite real number (not a bool), else None."""
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _is_list(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _exact_sum(values: np.ndarray) -> float:
    """Exactly rounded sum of a float64 array; NaN if the sum is not representable (the op then fails)."""
    flat = values.ravel()
    chunks = (flat[start : start + _SUM_CHUNK].tolist() for start in range(0, flat.size, _SUM_CHUNK))
    try:
        return math.fsum(itertools.chain.from_iterable(chunks))
    except (OverflowError, ValueError):
        # fsum raises on an overflowing exact sum or on inf - inf. The entry becomes NaN, which the
        # post-compute finiteness check reports as a patch error unless a later op overwrites it.
        return math.nan


def _matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """``matrix @ vector`` with one exactly rounded sum per row."""
    return np.array([_exact_sum(row * vector) for row in matrix], dtype=np.float64)


def _vecmat(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """``vector @ matrix`` with one exactly rounded sum per column."""
    products = np.ascontiguousarray((vector[:, None] * matrix).T)
    return np.array([_exact_sum(column) for column in products], dtype=np.float64)


def frobenius(values: np.ndarray) -> float:
    """Frobenius norm of a float64 array via an exactly rounded sum of squares."""
    return math.sqrt(_exact_sum(np.square(values)))


def unit_vector(values: np.ndarray) -> np.ndarray | str:
    """``values / ||values||`` computed without overflow or underflow, or the reason it is impossible."""
    if values.size and not np.all(np.isfinite(values)):
        return "has non-finite entries"
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale == 0.0:
        return "is the zero vector, which has no direction"
    scaled = values / scale
    return scaled / math.sqrt(_exact_sum(scaled * scaled))


class Workspace:
    """Float64 C-order working copies of the tensors that ops touch, made from the float32 parent on first use."""

    def __init__(self, parent: Mapping[str, np.ndarray]) -> None:
        self._parent = parent
        self.tensors: dict[str, np.ndarray] = {}

    def tensor(self, name: str) -> np.ndarray:
        """The working copy of ``name``; in-place changes to it are the op's effect."""
        if name not in self.tensors:
            self.tensors[name] = np.array(self._parent[name], dtype=np.float64, order="C")
        return self.tensors[name]

    def matrix(self, name: str) -> np.ndarray:
        """The 2-D view ``reshape(shape[0], -1)`` of the working copy; it shares memory with it."""
        array = self.tensor(name)
        return array.reshape(array.shape[0], -1)


# ----------------------------------------------------------------------------------------------- op types


class Op(ABC):
    """A structurally valid op. ``bind`` checks referenced arrays; ``apply`` changes the workspace."""

    def __init__(self, index: int, kind: str, tensor: str) -> None:
        self.index = index
        self.kind = kind
        self.tensor = tensor

    @property
    def where(self) -> str:
        return f"ops[{self.index}] ({self.kind})"

    def file_fields(self) -> dict[str, str]:
        """Field name to referenced .npy file name, for the fields that reference files."""
        return {}

    def bind(self, files: Mapping[str, np.ndarray], log: MessageLog) -> bool:
        """Check the loaded arrays this op references. A file that failed to load was already logged."""
        return True

    @abstractmethod
    def edit_size(self) -> int:
        """Numbers this op specifies (contract 5.1 edit size); valid after a successful ``bind``."""

    @abstractmethod
    def apply(self, workspace: Workspace) -> None:
        """Apply the op in float64 to the workspace."""


class _ScaleAxis(Op):
    def __init__(self, index: int, kind: str, tensor: str, axis: int, positions: np.ndarray, factor: float) -> None:
        super().__init__(index, kind, tensor)
        self._selector = (slice(None),) * axis + (positions,)
        self._n_positions = int(positions.size)
        self._factor = factor

    def edit_size(self) -> int:
        return self._n_positions

    def apply(self, workspace: Workspace) -> None:
        workspace.tensor(self.tensor)[self._selector] *= self._factor


class _IndexWrite(Op):
    def __init__(self, index: int, kind: str, tensor: str, coords: np.ndarray, values: np.ndarray) -> None:
        super().__init__(index, kind, tensor)
        self._coords = tuple(coords[:, axis] for axis in range(coords.shape[1]))
        self._values = values

    def edit_size(self) -> int:
        return int(self._values.size)

    def apply(self, workspace: Workspace) -> None:
        array = workspace.tensor(self.tensor)
        if self.kind == "add":
            np.add.at(array, self._coords, self._values)
        else:
            array[self._coords] = self._values


class _SliceWrite(Op):
    def __init__(
        self, index: int, kind: str, tensor: str, block: tuple[slice, ...], block_shape: Shape,
        value: float | None, values_file: str | None,
    ) -> None:
        super().__init__(index, kind, tensor)
        self._block = block
        self._block_shape = block_shape
        self._data: float | np.ndarray | None = value
        self._values_file = values_file

    def file_fields(self) -> dict[str, str]:
        return {} if self._values_file is None else {"values_file": self._values_file}

    def edit_size(self) -> int:
        return 1 if self._values_file is None else math.prod(self._block_shape)

    def bind(self, files: Mapping[str, np.ndarray], log: MessageLog) -> bool:
        if self._values_file is None:
            return True
        array = files.get(self._values_file)
        if array is None:
            return False
        if array.shape != self._block_shape:
            log.add(self.where, f"values_file {self._values_file!r} has shape {list(array.shape)}, but the slice "
                    f"selects a block of shape {list(self._block_shape)}")
            return False
        self._data = array
        return True

    def apply(self, workspace: Workspace) -> None:
        if self._data is None:
            raise RuntimeError(f"{self.where}: applied before its values_file was bound")
        array = workspace.tensor(self.tensor)
        if self.kind == "add":
            array[self._block] += self._data
        else:
            array[self._block] = self._data


class _ProjectOut(Op):
    def __init__(
        self, index: int, tensor: str, side: str, length: int, direction: np.ndarray | None,
        direction_file: str | None, strength: float, bias: str | None,
    ) -> None:
        super().__init__(index, "project_out", tensor)
        self._side = side
        self._length = length
        self._direction = direction
        self._direction_file = direction_file
        self._strength = strength
        self._bias = bias

    def file_fields(self) -> dict[str, str]:
        return {} if self._direction_file is None else {"direction": self._direction_file}

    def edit_size(self) -> int:
        return self._length

    def bind(self, files: Mapping[str, np.ndarray], log: MessageLog) -> bool:
        if self._direction_file is None:
            return True
        array = files.get(self._direction_file)
        if array is None:
            return False
        if array.shape != (self._length,):
            log.add(self.where, f"direction {self._direction_file!r} has shape {list(array.shape)}; side "
                    f"{self._side!r} needs a 1-D array of length {self._length}")
            return False
        unit = unit_vector(array)
        if isinstance(unit, str):
            log.add(self.where, f"direction {self._direction_file!r} {unit}")
            return False
        self._direction = unit
        return True

    def apply(self, workspace: Workspace) -> None:
        d = self._direction
        if d is None:
            raise RuntimeError(f"{self.where}: applied before its direction was bound")
        s = self._strength
        matrix = workspace.matrix(self.tensor)
        if self._side == "in":
            matrix -= s * np.outer(_matvec(matrix, d), d)
        else:
            matrix -= s * np.outer(d, _vecmat(d, matrix))
            if self._bias is not None:
                bias = workspace.tensor(self._bias)
                bias -= s * d * _exact_sum(d * bias)


class _LowRankAdd(Op):
    def __init__(self, index: int, tensor: str, shape: Shape, u_file: str, v_file: str, max_rank: int) -> None:
        super().__init__(index, "low_rank_add", tensor)
        self._rows = shape[0]
        self._cols = math.prod(shape[1:])
        self._u_file = u_file
        self._v_file = v_file
        self._max_rank = max_rank
        self._u: np.ndarray | None = None
        self._v: np.ndarray | None = None

    def file_fields(self) -> dict[str, str]:
        return {"u": self._u_file, "v": self._v_file}

    def edit_size(self) -> int:
        if self._u is None:
            raise RuntimeError(f"{self.where}: edit size requested before u and v were bound")
        return int(self._u.shape[1]) * (self._rows + self._cols)

    def bind(self, files: Mapping[str, np.ndarray], log: MessageLog) -> bool:
        u = files.get(self._u_file)
        v = files.get(self._v_file)
        if u is None or v is None:
            return False
        problems = []
        if u.ndim != 2 or u.shape[0] != self._rows:
            problems.append(f"u {self._u_file!r} has shape {list(u.shape)}, expected [{self._rows}, r]")
        if v.ndim != 2 or v.shape[0] != self._cols:
            problems.append(f"v {self._v_file!r} has shape {list(v.shape)}, expected [{self._cols}, r]")
        if not problems:
            if u.shape[1] != v.shape[1]:
                problems.append(f"u has r = {u.shape[1]} columns but v has r = {v.shape[1]}")
            elif not 1 <= u.shape[1] <= self._max_rank:
                problems.append(f"rank r = {u.shape[1]} is outside 1..{self._max_rank}")
        for problem in problems:
            log.add(self.where, problem)
        if problems:
            return False
        self._u, self._v = u, v
        return True

    def apply(self, workspace: Workspace) -> None:
        if self._u is None or self._v is None:
            raise RuntimeError(f"{self.where}: applied before u and v were bound")
        matrix = workspace.matrix(self.tensor)
        update = np.zeros_like(matrix)
        for k in range(self._u.shape[1]):
            update += np.outer(self._u[:, k], self._v[:, k])
        matrix += update


class _AddDense(Op):
    def __init__(self, index: int, tensor: str, shape: Shape, delta_file: str) -> None:
        super().__init__(index, "add_dense", tensor)
        self._shape = shape
        self._delta_file = delta_file
        self._delta: np.ndarray | None = None

    def file_fields(self) -> dict[str, str]:
        return {"delta": self._delta_file}

    def edit_size(self) -> int:
        return math.prod(self._shape)

    def bind(self, files: Mapping[str, np.ndarray], log: MessageLog) -> bool:
        delta = files.get(self._delta_file)
        if delta is None:
            return False
        if delta.shape != self._shape:
            log.add(self.where, f"delta {self._delta_file!r} has shape {list(delta.shape)}, but {self.tensor!r} "
                    f"has shape {list(self._shape)}")
            return False
        self._delta = delta
        return True

    def apply(self, workspace: Workspace) -> None:
        if self._delta is None:
            raise RuntimeError(f"{self.where}: applied before its delta was bound")
        workspace.tensor(self.tensor)[...] += self._delta


def group_problem(labels: np.ndarray, length: int) -> str | None:
    """Why ``labels`` is not a valid group map of an axis of ``length`` positions, or None if it is."""
    if labels.shape != (length,):
        return f"has shape {list(labels.shape)}; it needs one group id per position, shape [{length}]"
    if not np.all(np.isfinite(labels)):
        return "has non-finite entries"
    if not np.array_equal(labels, np.floor(labels)):
        return "has non-integer entries; group ids are integers"
    if labels.size and (labels.min() < 0 or labels.max() >= length):
        return f"has group ids outside [0, {length})"
    return None


def group_codes(labels: np.ndarray) -> np.ndarray:
    """Dense group codes 0..G-1 (in order of the group ids) for a valid group map."""
    _, codes = np.unique(labels.astype(np.int64), return_inverse=True)
    return codes.reshape(-1).astype(np.int64)


class _Tie(Op):
    """``W2 <- W2 + s (G - W2)`` along one axis of the 2-D view, where G holds each position's group mean."""

    def __init__(
        self, index: int, tensor: str, axis: int, length: int, codes: np.ndarray | None,
        groups_file: str | None, strength: float,
    ) -> None:
        super().__init__(index, "tie", tensor)
        self._axis = axis
        self._length = length
        self._codes = codes
        self._groups_file = groups_file
        self._strength = strength

    def file_fields(self) -> dict[str, str]:
        return {} if self._groups_file is None else {"groups": self._groups_file}

    def edit_size(self) -> int:
        return self._length

    def bind(self, files: Mapping[str, np.ndarray], log: MessageLog) -> bool:
        if self._groups_file is None:
            return True
        labels = files.get(self._groups_file)
        if labels is None:
            return False
        problem = group_problem(labels, self._length)
        if problem is not None:
            log.add(self.where, f"groups {self._groups_file!r} {problem}")
            return False
        self._codes = group_codes(labels)
        return True

    def apply(self, workspace: Workspace) -> None:
        codes = self._codes
        if codes is None:
            raise RuntimeError(f"{self.where}: applied before its groups were bound")
        matrix = workspace.matrix(self.tensor)
        view = matrix if self._axis == 1 else matrix.T  # positions run along axis 1 of ``view``
        n_groups = int(codes.max()) + 1
        sums = np.zeros((view.shape[0], n_groups), dtype=np.float64)
        for position in range(self._length):
            sums[:, codes[position]] += view[:, position]
        means = sums / np.bincount(codes, minlength=n_groups).astype(np.float64)
        view += self._strength * (means[:, codes] - view)


# ----------------------------------------------------------------------------------------------- parsing


class _Context:
    """What parsers need: tensor shapes, limits and the message log."""

    def __init__(self, shapes: Mapping[str, Shape], limits: Limits, log: MessageLog) -> None:
        self.shapes = shapes
        self.limits = limits
        self.log = log


def _check_fields(where: str, raw: Mapping[Any, Any], required: Sequence[str], optional: Sequence[str], log: MessageLog) -> None:
    missing = [name for name in required if name not in raw]
    unknown = sorted(str(key) for key in raw if key not in required and key not in optional)
    if missing:
        log.add(where, f"missing field(s) {', '.join(missing)}")
    if unknown:
        allowed = ", ".join([*required, *optional])
        log.add(where, f"unknown field(s) {', '.join(_short(name, 40) for name in unknown)}; allowed: {allowed}")


def _tensor(where: str, raw: Mapping[str, Any], field: str, min_rank: int, ctx: _Context) -> str | None:
    if field not in raw:
        return None
    name = raw[field]
    if not isinstance(name, str):
        ctx.log.add(where, f"{field} must be a tensor name (string), got {_short(name)}")
        return None
    if name not in ctx.shapes:
        ctx.log.add(where, f"unknown tensor {_short(name)}; tensors: {', '.join(sorted(ctx.shapes))}")
        return None
    rank = len(ctx.shapes[name])
    if rank < min_rank:
        ctx.log.add(where, f"tensor {name!r} has rank {rank}; this op needs rank >= {min_rank}")
        return None
    return name


def _number(
    where: str, raw: Mapping[str, Any], field: str, ctx: _Context,
    check: Callable[[float], bool] | None = None, rule: str = "",
) -> float | None:
    if field not in raw:
        return None
    number = _finite_float(raw[field])
    if number is None:
        ctx.log.add(where, f"{field} must be a finite number, got {_short(raw[field])}")
        return None
    if check is not None and not check(number):
        ctx.log.add(where, f"{field} = {number!r} is outside {rule}")
        return None
    return number


def _file_name(where: str, raw: Mapping[str, Any], field: str, ctx: _Context) -> str | None:
    if field not in raw:
        return None
    name = raw[field]
    if not is_npy_name(name):
        ctx.log.add(where, f"{field} must be the bare name of a .npy file in the patch directory, matching "
                    f"{NPY_NAME_PATTERN}; got {_short(name)}")
        return None
    return name


def _positions(where: str, raw: Mapping[str, Any], field: str, length: int, ctx: _Context) -> np.ndarray | None:
    if field not in raw:
        return None
    values = raw[field]
    if not _is_list(values) or not values:
        ctx.log.add(where, f"{field} must be a non-empty list of indices, got {_short(values)}")
        return None
    problems = []
    seen: set[int] = set()
    for k, value in enumerate(values):
        if not _is_int(value):
            problems.append(f"{field}[{k}] = {_short(value)} is not an integer")
        elif not 0 <= value < length:
            problems.append(f"{field}[{k}] = {value} is outside [0, {length})")
        elif int(value) in seen:
            problems.append(f"{field}[{k}] = {value} is repeated")
        else:
            seen.add(int(value))
    if problems:
        ctx.log.add(where, _summary(problems))
        return None
    return np.array(values, dtype=np.int64)


def _scale(index: int, kind: str, raw: Mapping[str, Any], ctx: _Context) -> Op | None:
    axis, field = (0, "rows") if kind == "scale_rows" else (1, "cols")
    where = f"ops[{index}] ({kind})"
    _check_fields(where, raw, ("op", "tensor", field, "factor"), (), ctx.log)
    tensor = _tensor(where, raw, "tensor", axis + 1, ctx)
    positions = None
    if tensor is not None:
        positions = _positions(where, raw, field, ctx.shapes[tensor][axis], ctx)
    factor = _number(where, raw, "factor", ctx)
    if tensor is None or positions is None or factor is None:
        return None
    return _ScaleAxis(index, kind, tensor, axis, positions, factor)


def _index_tuples(where: str, values: object, shape: Shape, ctx: _Context) -> np.ndarray | None:
    rank = len(shape)
    if not _is_list(values) or not values:
        ctx.log.add(where, f"index must be a non-empty list of index tuples, got {_short(values)}")
        return None
    problems = []
    for k, item in enumerate(values):
        if not _is_list(item) or len(item) != rank:
            problems.append(f"index[{k}] = {_short(item)} is not a list of {rank} integers (one per axis)")
            continue
        for axis, value in enumerate(item):
            if not _is_int(value):
                problems.append(f"index[{k}][{axis}] = {_short(value)} is not an integer")
                break
            if not 0 <= value < shape[axis]:
                problems.append(f"index[{k}][{axis}] = {value} is outside [0, {shape[axis]})")
                break
    if problems:
        ctx.log.add(where, _summary(problems))
        return None
    return np.array(values, dtype=np.int64).reshape(len(values), rank)


def _value_list(where: str, values: object, count: int | None, ctx: _Context) -> np.ndarray | None:
    if not _is_list(values):
        ctx.log.add(where, f"values must be a list of numbers, got {_short(values)}")
        return None
    numbers = [_finite_float(value) for value in values]
    problems = [f"values[{k}] = {_short(values[k])} is not a finite number" for k, n in enumerate(numbers) if n is None]
    if problems:
        ctx.log.add(where, _summary(problems))
    if count is not None and len(values) != count:
        ctx.log.add(where, f"values has {len(values)} entries but index has {count} tuples")
        return None
    if problems:
        return None
    return np.array(numbers, dtype=np.float64)


def _slice_block(where: str, entries: object, shape: Shape, ctx: _Context) -> tuple[tuple[slice, ...], Shape] | None:
    rank = len(shape)
    if not _is_list(entries) or len(entries) != rank:
        ctx.log.add(where, f"slice must be a list of {rank} entries (one per axis), each [start, stop] or null; "
                    f"got {_short(entries)}")
        return None
    block: list[slice] = []
    dims: list[int] = []
    problems = []
    for axis, entry in enumerate(entries):
        length = shape[axis]
        if entry is None:
            block.append(slice(0, length))
            dims.append(length)
        elif not (_is_list(entry) and len(entry) == 2 and _is_int(entry[0]) and _is_int(entry[1])):
            problems.append(f"slice[{axis}] = {_short(entry)} is neither [start, stop] with integers nor null")
        elif not 0 <= entry[0] < entry[1] <= length:
            problems.append(f"slice[{axis}] = [{entry[0]}, {entry[1]}] needs 0 <= start < stop <= {length}")
        else:
            block.append(slice(int(entry[0]), int(entry[1])))
            dims.append(int(entry[1]) - int(entry[0]))
    if problems:
        ctx.log.add(where, _summary(problems))
        return None
    return tuple(block), tuple(dims)


def _write(index: int, kind: str, raw: Mapping[str, Any], ctx: _Context) -> Op | None:
    where = f"ops[{index}] ({kind})"
    before = ctx.log.count
    _check_fields(where, raw, ("op", "tensor"), ("index", "values", "slice", "value", "values_file"), ctx.log)
    tensor = _tensor(where, raw, "tensor", 1, ctx)
    has_index, has_slice = "index" in raw, "slice" in raw
    if has_index == has_slice:
        ctx.log.add(where, "give exactly one of index (with values) or slice (with value or values_file)")
        return None
    if has_index:
        for field in ("value", "values_file"):
            if field in raw:
                ctx.log.add(where, f"{field} belongs to the slice form; the index form takes values")
        coords = None if tensor is None else _index_tuples(where, raw["index"], ctx.shapes[tensor], ctx)
        values = None
        if "values" not in raw:
            ctx.log.add(where, "the index form needs values, one number per index tuple")
        else:
            values = _value_list(where, raw["values"], None if coords is None else len(coords), ctx)
        if tensor is None or coords is None or values is None or ctx.log.count > before:
            return None
        if kind == "set":
            shape = ctx.shapes[tensor]
            unique, counts = np.unique(np.ravel_multi_index(tuple(coords.T), shape), return_counts=True)
            repeated = [
                f"index tuple {[int(i) for i in np.unravel_index(int(entry), shape)]} appears {int(times)} times"
                for entry, times in zip(unique[counts > 1], counts[counts > 1])
            ]
            if repeated:
                ctx.log.add(where, _summary(repeated) + "; set needs distinct tuples (add accumulates repeats)")
                return None
        return _IndexWrite(index, kind, tensor, coords, values)
    if "values" in raw:
        ctx.log.add(where, "values belongs to the index form; the slice form takes value or values_file")
    if ("value" in raw) == ("values_file" in raw):
        ctx.log.add(where, "the slice form needs exactly one of value (a number) or values_file (a .npy file)")
    value = _number(where, raw, "value", ctx)
    values_file = _file_name(where, raw, "values_file", ctx)
    if tensor is None:
        return None
    sliced = _slice_block(where, raw["slice"], ctx.shapes[tensor], ctx)
    if sliced is None or ctx.log.count > before:
        return None
    block, block_shape = sliced
    return _SliceWrite(index, kind, tensor, block, block_shape, value, values_file)


def _project_out(index: int, raw: Mapping[str, Any], ctx: _Context) -> Op | None:
    where = f"ops[{index}] (project_out)"
    before = ctx.log.count
    _check_fields(where, raw, ("op", "tensor", "side", "direction", "strength"), ("bias",), ctx.log)
    tensor = _tensor(where, raw, "tensor", 2, ctx)
    side = raw.get("side")
    if "side" in raw and side not in ("in", "out"):
        ctx.log.add(where, f"side must be 'in' or 'out', got {_short(side)}")
    strength = _number(where, raw, "strength", ctx, lambda s: 0.0 < s <= 1.0, "(0, 1]")
    bias = None
    if "bias" in raw:
        if side == "in":
            ctx.log.add(where, "bias is allowed only with side 'out'")
        bias = _tensor(where, raw, "bias", 1, ctx)
        if bias is not None and tensor is not None:
            if bias == tensor:
                ctx.log.add(where, "bias must name a tensor other than tensor")
            elif ctx.shapes[bias] != (ctx.shapes[tensor][0],):
                ctx.log.add(where, f"bias {bias!r} has shape {list(ctx.shapes[bias])}; it must be 1-D of length "
                            f"{ctx.shapes[tensor][0]} (shape[0] of {tensor!r})")
    length: int | None = None
    if tensor is not None and side in ("in", "out"):
        shape = ctx.shapes[tensor]
        length = shape[0] if side == "out" else math.prod(shape[1:])
    direction: np.ndarray | None = None
    direction_file: str | None = None
    raw_direction = raw.get("direction")
    if isinstance(raw_direction, str):
        direction_file = _file_name(where, raw, "direction", ctx)
    elif _is_list(raw_direction):
        numbers = [_finite_float(value) for value in raw_direction]
        if any(number is None for number in numbers):
            ctx.log.add(where, "direction entries must be finite numbers")
        elif length is not None and len(numbers) != length:
            ctx.log.add(where, f"direction has {len(numbers)} entries; side {side!r} needs {length}")
        elif length is not None:
            unit = unit_vector(np.array(numbers, dtype=np.float64))
            if isinstance(unit, str):
                ctx.log.add(where, f"direction {unit}")
            else:
                direction = unit
    elif "direction" in raw:
        ctx.log.add(where, f"direction must be a .npy file name or a list of numbers, got {_short(raw_direction)}")
    if tensor is None or strength is None or length is None or ctx.log.count > before:
        return None
    if direction is None and direction_file is None:
        return None
    return _ProjectOut(index, tensor, str(side), length, direction, direction_file, strength, bias)


def _low_rank_add(index: int, raw: Mapping[str, Any], ctx: _Context) -> Op | None:
    where = f"ops[{index}] (low_rank_add)"
    before = ctx.log.count
    _check_fields(where, raw, ("op", "tensor", "u", "v"), (), ctx.log)
    tensor = _tensor(where, raw, "tensor", 2, ctx)
    u_file = _file_name(where, raw, "u", ctx)
    v_file = _file_name(where, raw, "v", ctx)
    if tensor is None or u_file is None or v_file is None or ctx.log.count > before:
        return None
    return _LowRankAdd(index, tensor, ctx.shapes[tensor], u_file, v_file, ctx.limits.max_rank)


def _add_dense(index: int, raw: Mapping[str, Any], ctx: _Context) -> Op | None:
    where = f"ops[{index}] (add_dense)"
    before = ctx.log.count
    _check_fields(where, raw, ("op", "tensor", "delta"), (), ctx.log)
    tensor = _tensor(where, raw, "tensor", 1, ctx)
    delta_file = _file_name(where, raw, "delta", ctx)
    if tensor is None or delta_file is None or ctx.log.count > before:
        return None
    return _AddDense(index, tensor, ctx.shapes[tensor], delta_file)


def _tie(index: int, raw: Mapping[str, Any], ctx: _Context) -> Op | None:
    where = f"ops[{index}] (tie)"
    before = ctx.log.count
    _check_fields(where, raw, ("op", "tensor", "axis", "groups", "strength"), (), ctx.log)
    tensor = _tensor(where, raw, "tensor", 1, ctx)
    axis = raw.get("axis")
    if "axis" in raw and not (_is_int(axis) and axis in (0, 1)):
        ctx.log.add(where, f"axis must be 0 or 1 (an axis of the 2-D view), got {_short(axis)}")
        axis = None
    strength = _number(where, raw, "strength", ctx, lambda s: 0.0 < s <= 1.0, "(0, 1]")
    length: int | None = None
    if tensor is not None and axis is not None:
        shape = ctx.shapes[tensor]
        if axis == 1 and len(shape) < 2:
            ctx.log.add(where, f"tensor {tensor!r} has rank 1; axis 1 needs rank >= 2")
        else:
            length = shape[0] if axis == 0 else math.prod(shape[1:])
    codes: np.ndarray | None = None
    groups_file: str | None = None
    raw_groups = raw.get("groups")
    if isinstance(raw_groups, str):
        groups_file = _file_name(where, raw, "groups", ctx)
    elif _is_list(raw_groups):
        if not all(_is_int(value) for value in raw_groups):
            ctx.log.add(where, "groups entries must be integers (group ids)")
        elif length is not None:
            problem = group_problem(np.array(raw_groups, dtype=np.float64), length)
            if problem is not None:
                ctx.log.add(where, f"groups {problem}")
            else:
                codes = group_codes(np.array(raw_groups, dtype=np.int64))
    elif "groups" in raw:
        ctx.log.add(where, f"groups must be a .npy file name or a list of integers, got {_short(raw_groups)}")
    if tensor is None or axis is None or strength is None or length is None or ctx.log.count > before:
        return None
    if codes is None and groups_file is None:
        return None
    return _Tie(index, tensor, int(axis), length, codes, groups_file, strength)


def _parse_op(index: int, raw: object, ctx: _Context) -> Op | None:
    if not isinstance(raw, Mapping):
        ctx.log.add(f"ops[{index}]", f"must be a JSON object with an 'op' field, got {_short(raw)}")
        return None
    if "op" not in raw:
        ctx.log.add(f"ops[{index}]", f"missing field op; ops: {', '.join(OP_NAMES)}")
        return None
    kind = raw["op"]
    if kind not in OP_NAMES:
        ctx.log.add(f"ops[{index}]", f"unknown op {_short(kind)}; ops: {', '.join(OP_NAMES)}")
        return None
    before = ctx.log.count
    if kind in ("scale_rows", "scale_cols"):
        op = _scale(index, kind, raw, ctx)
    elif kind in ("set", "add"):
        op = _write(index, kind, raw, ctx)
    elif kind == "project_out":
        op = _project_out(index, raw, ctx)
    elif kind == "low_rank_add":
        op = _low_rank_add(index, raw, ctx)
    elif kind == "add_dense":
        op = _add_dense(index, raw, ctx)
    else:
        op = _tie(index, raw, ctx)
    return op if ctx.log.count == before else None


def parse_patch(
    patch: object, shapes: Mapping[str, Shape], limits: Limits, expected_parent_sha: str, log: MessageLog
) -> list[Op]:
    """Validate the top level and every op of ``patch``; return the ops that are structurally valid.

    Every problem is logged. The caller must check ``log`` before applying anything, because the returned
    list omits invalid ops.
    """
    if not isinstance(patch, Mapping):
        log.add("patch", f"must be a JSON object, got {type(patch).__name__}")
        return []
    _check_fields("patch", patch, TOP_LEVEL_KEYS, (), log)
    if "schema" in patch and patch["schema"] != SCHEMA:
        log.add("schema", f"must be {SCHEMA!r}, got {_short(patch['schema'])}")
    if "parent_sha256" in patch:
        declared = patch["parent_sha256"]
        if not isinstance(declared, str) or SHA256_HEX.fullmatch(declared) is None:
            log.add("parent_sha256", f"must be 64 lowercase hex characters (params_sha256), got {_short(declared)}")
        elif declared != expected_parent_sha:
            log.add("parent_sha256", f"{declared} is not the params_sha256 of the model this patch is checked "
                    "against; write the hash of the model you were shown")
    for field in ("hypothesis", "predicted_effect"):
        if field in patch and not (isinstance(patch[field], str) and patch[field].strip()):
            log.add(field, f"must be a non-empty string, got {_short(patch[field])}")
    if "ops" not in patch:
        return []
    raw_ops = patch["ops"]
    if not isinstance(raw_ops, list):
        log.add("ops", f"must be a list of op objects, got {type(raw_ops).__name__}")
        return []
    if len(raw_ops) > limits.max_ops:
        log.add("ops", f"has {len(raw_ops)} ops; at most {limits.max_ops} are allowed")
    ctx = _Context(shapes, limits, log)
    parsed = [_parse_op(index, raw, ctx) for index, raw in enumerate(raw_ops)]
    return [op for op in parsed if op is not None]
