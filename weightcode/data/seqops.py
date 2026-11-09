"""seqops: a synthetic sequence-operator family for the new-skill experiment.

A row applies one or two operators to a random symbol string. The student reads the token row

    [BOS, OPA, OPB, x_1 .. x_n, SEP, OUT x n]   right-padded with PAD to SEQ_LEN

and predicts one symbol at every OUT position. The answer is y = OPB(OPA(x)); OPB = NONE applies OPA alone. ``y`` is
stored per position: the symbol index (0 .. N_SYMBOLS - 1) at the k-th OUT position holds output k, and every other
position holds IGNORE. A row is right when every labelled position is right (exact match), so one rule scores
images (a label per row) and sequences (labels at some positions) alike.

Operators. An operator is either a position map (out[k] = in[index[k]], defined for every length) or a symbol map
(out[k] = table[in[k]]):

* old operators, the base's trained skills: position maps REV (reverse), ROTL1 and ROTR1 (rotate by one), SWAP2
  (swap positions 2j and 2j + 1; with odd n the last symbol stays); symbol maps INC (s + 1 mod N_SYMBOLS), DEC
  (s - 1 mod N_SYMBOLS), COMP (N_SYMBOLS - 1 - s) and SUBA, a fixed random permutation of the symbols drawn from
  ``default_rng([SPLIT_SEED, SUBA_STREAM])``.
* reserved operator tokens NEW0 .. NEW3. No base training row contains one (``sample_old_rows`` can only draw old
  operators), so their embedding rows keep their initial values. A new skill binds one of them to a rule drawn by
  ``draw_new_rule`` after the base is trained. The only rule kind implemented is ``substitution``: a random
  permutation of the symbols, which is defined at every length and identifiable from examples that cover the symbols.

Row keys. ``row_key`` packs (OPA, OPB, n, x) into one int64 (token ids below 32, n below 16, 4 bits per symbol), so
two rows with the same key are the same question. Held-out splits are drawn disjoint from each other by key, and
base training refuses every key of a held-out split (``TrainStream``).

Generators: every draw uses ``numpy.random.default_rng`` (PCG64) with a seed list that starts with SPLIT_SEED. The
streams in use under SPLIT_SEED elsewhere are 1..6, 301..306, 700, 901..903, 9001 and 9002; seqops uses 1100..1199.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from weightcode.data.registry import SPLIT_SEED

DATASET = "seqops"
N_SYMBOLS = 16
MIN_LEN = 4
MAX_LEN = 12
SEQ_LEN = 4 + 2 * MAX_LEN
"""BOS, OPA, OPB, up to MAX_LEN symbols, SEP, up to MAX_LEN OUT slots."""
IGNORE = -1

SPECIAL_TOKENS: tuple[str, ...] = ("PAD", "BOS", "SEP", "OUT", "NONE")
POSITION_OPS: tuple[str, ...] = ("REV", "ROTL1", "ROTR1", "SWAP2")
SYMBOL_OPS: tuple[str, ...] = ("INC", "DEC", "COMP", "SUBA")
OLD_OPS: tuple[str, ...] = POSITION_OPS + SYMBOL_OPS
NEW_OPS: tuple[str, ...] = ("NEW0", "NEW1", "NEW2", "NEW3")
TOKENS: tuple[str, ...] = (
    SPECIAL_TOKENS + tuple(f"OP_{name}" for name in OLD_OPS + NEW_OPS) + tuple(f"s{k}" for k in range(N_SYMBOLS))
)
TOKEN_ID: Mapping[str, int] = MappingProxyType({name: index for index, name in enumerate(TOKENS)})
VOCAB_SIZE = len(TOKENS)
PAD, BOS, SEP, OUT, NONE = (TOKEN_ID[name] for name in SPECIAL_TOKENS)
SYMBOL_BASE = TOKEN_ID["s0"]
OLD_OP_IDS: tuple[int, ...] = tuple(TOKEN_ID[f"OP_{name}"] for name in OLD_OPS)
NEW_OP_IDS: tuple[int, ...] = tuple(TOKEN_ID[f"OP_{name}"] for name in NEW_OPS)

TRAIN_STREAM = 1100
"""Base training batches: ``default_rng([SPLIT_SEED, TRAIN_STREAM, model_seed, step])``."""
SUBA_STREAM = 1110
NEW_RULE_STREAM = 1120
"""Rule of reserved operator k: ``default_rng([SPLIT_SEED, NEW_RULE_STREAM, k])``."""
PAIR_FRACTION = 0.5
"""Share of two-operator rows among old-operator rows (training and the old held-out splits)."""
RULE_KINDS: tuple[str, ...] = ("substitution",)

_KEY_SYMBOL_BITS = 4
_KEY_PAYLOAD_BITS = _KEY_SYMBOL_BITS * MAX_LEN
if N_SYMBOLS > 1 << _KEY_SYMBOL_BITS or max((NONE, *OLD_OP_IDS, *NEW_OP_IDS)) >= 32 or MAX_LEN >= 16:
    raise AssertionError("row_key packing assumes <= 16 symbols, operator token ids below 32 and lengths below 16")


class SeqOpsError(ValueError):
    """A row, rule or draw that seqops refuses."""


def _symbol_table(name: str) -> np.ndarray:
    s = np.arange(N_SYMBOLS, dtype=np.int64)
    if name == "INC":
        return (s + 1) % N_SYMBOLS
    if name == "DEC":
        return (s - 1) % N_SYMBOLS
    if name == "COMP":
        return N_SYMBOLS - 1 - s
    if name == "SUBA":
        table = np.random.default_rng([SPLIT_SEED, SUBA_STREAM]).permutation(N_SYMBOLS).astype(np.int64)
        if np.array_equal(table, s):
            raise AssertionError("SUBA drew the identity")
        return table
    raise KeyError(f"{name!r} is not a symbol operator")


OLD_TABLES: Mapping[str, np.ndarray] = MappingProxyType({name: _symbol_table(name) for name in SYMBOL_OPS})


def position_index(name: str, n: int) -> np.ndarray:
    """Index map of a position operator at length n: out[k] = in[index[k]]."""
    if not MIN_LEN <= n <= MAX_LEN:
        raise SeqOpsError(f"length {n} outside {MIN_LEN}..{MAX_LEN}")
    k = np.arange(n, dtype=np.int64)
    if name == "REV":
        return n - 1 - k
    if name == "ROTL1":
        return (k + 1) % n
    if name == "ROTR1":
        return (k - 1) % n
    if name == "SWAP2":
        partner = k ^ 1
        return np.where(partner < n, partner, k)
    raise KeyError(f"{name!r} is not a position operator")


@dataclass(frozen=True)
class Rule:
    """What one operator token does: a position map (by name) or a symbol table."""

    token: int
    kind: str
    """"position" or "symbol"."""
    table: tuple[int, ...] | None = None
    name: str | None = None

    def apply(self, sym: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Apply to rows sym [B, MAX_LEN] (symbols in the first n[b] slots); slots beyond n stay as given."""
        out = sym.copy()
        if self.kind == "symbol":
            table = np.asarray(self.table, dtype=np.int64)
            valid = np.arange(MAX_LEN)[None, :] < n[:, None]
            out[valid] = table[sym[valid]]
            return out
        for length in np.unique(n):
            rows = np.flatnonzero(n == length)
            index = position_index(str(self.name), int(length))
            out[np.ix_(rows, np.arange(length))] = sym[np.ix_(rows, index)]
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"token": TOKENS[self.token], "kind": self.kind, "table": None if self.table is None else list(self.table),
                "name": self.name}


def old_rules() -> dict[int, Rule]:
    """The trained operators keyed by token id."""
    rules: dict[int, Rule] = {}
    for name in POSITION_OPS:
        rules[TOKEN_ID[f"OP_{name}"]] = Rule(TOKEN_ID[f"OP_{name}"], "position", name=name)
    for name in SYMBOL_OPS:
        rules[TOKEN_ID[f"OP_{name}"]] = Rule(TOKEN_ID[f"OP_{name}"], "symbol", table=tuple(int(v) for v in OLD_TABLES[name]))
    return rules


def draw_new_rule(index: int, kind: str = "substitution") -> Rule:
    """The rule bound to reserved operator NEW<index>, from ``default_rng([SPLIT_SEED, NEW_RULE_STREAM, index])``.

    ``substitution``: a uniform random permutation of the symbols that is neither the identity nor any old table.
    Call only after the base models are trained: no training code path reads it.
    """
    if not 0 <= index < len(NEW_OPS):
        raise SeqOpsError(f"reserved operator index {index} outside 0..{len(NEW_OPS) - 1}")
    if kind not in RULE_KINDS:
        raise SeqOpsError(f"rule kind {kind!r} is not implemented; implemented: {RULE_KINDS}")
    table = np.random.default_rng([SPLIT_SEED, NEW_RULE_STREAM, index]).permutation(N_SYMBOLS).astype(np.int64)
    taken = [np.arange(N_SYMBOLS, dtype=np.int64), *OLD_TABLES.values()]
    if any(np.array_equal(table, other) for other in taken):
        raise SeqOpsError(f"NEW{index} drew a table equal to the identity or an old operator")
    return Rule(NEW_OP_IDS[index], "symbol", table=tuple(int(v) for v in table))


def apply_ops(sym: np.ndarray, n: np.ndarray, opa: np.ndarray, opb: np.ndarray, rules: Mapping[int, Rule]) -> np.ndarray:
    """Outputs OPB(OPA(x)) [B, MAX_LEN] (zeros beyond each row's length); NONE as OPB is the identity."""
    sym = np.asarray(sym, dtype=np.int64)
    n, opa, opb = (np.asarray(v, dtype=np.int64) for v in (n, opa, opb))
    if sym.shape != (n.size, MAX_LEN) or opa.shape != n.shape or opb.shape != n.shape:
        raise SeqOpsError("sym must be [B, MAX_LEN] with n, opa, opb of length B")
    mid = np.zeros_like(sym)
    for token in np.unique(opa):
        if int(token) not in rules:
            raise SeqOpsError(f"no rule for OPA token {TOKENS[int(token)]}")
        rows = opa == token
        mid[rows] = rules[int(token)].apply(sym[rows], n[rows])
    out = mid.copy()
    for token in np.unique(opb):
        if int(token) == NONE:
            continue
        if int(token) not in rules:
            raise SeqOpsError(f"no rule for OPB token {TOKENS[int(token)]}")
        rows = opb == token
        out[rows] = rules[int(token)].apply(mid[rows], n[rows])
    beyond = np.arange(MAX_LEN)[None, :] >= n[:, None]
    out[beyond] = 0
    return out


def encode_rows(sym: np.ndarray, n: np.ndarray, opa: np.ndarray, opb: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Token rows x [B, SEQ_LEN] and per-position targets y [B, SEQ_LEN] (IGNORE off the OUT positions)."""
    sym, out = np.asarray(sym, dtype=np.int64), np.asarray(out, dtype=np.int64)
    n, opa, opb = (np.asarray(v, dtype=np.int64) for v in (n, opa, opb))
    if np.any((n < MIN_LEN) | (n > MAX_LEN)):
        raise SeqOpsError(f"lengths must lie in {MIN_LEN}..{MAX_LEN}")
    if np.any((sym < 0) | (sym >= N_SYMBOLS)) or np.any((out < 0) | (out >= N_SYMBOLS)):
        raise SeqOpsError("symbols must lie in 0..N_SYMBOLS-1")
    batch = n.size
    pos = np.arange(SEQ_LEN)[None, :]
    lengths = n[:, None]
    x = np.full((batch, SEQ_LEN), PAD, dtype=np.int64)
    x[:, 0], x[:, 1], x[:, 2] = BOS, opa, opb
    is_sym = (pos >= 3) & (pos < 3 + lengths)
    sym_at = np.take_along_axis(sym, np.clip(pos - 3, 0, MAX_LEN - 1).repeat(batch, axis=0), axis=1)
    x = np.where(is_sym, SYMBOL_BASE + sym_at, x)
    x = np.where(pos == 3 + lengths, SEP, x)
    is_out = (pos >= 4 + lengths) & (pos < 4 + 2 * lengths)
    x = np.where(is_out, OUT, x)
    out_at = np.take_along_axis(out, np.clip(pos - 4 - lengths, 0, MAX_LEN - 1), axis=1)
    y = np.where(is_out, out_at, IGNORE).astype(np.int64)
    return x, y


def row_key(sym: np.ndarray, n: np.ndarray, opa: np.ndarray, opb: np.ndarray) -> np.ndarray:
    """int64 key per row packing (OPA, OPB, n, x); equal keys mean the same question."""
    sym = np.asarray(sym, dtype=np.int64)
    n, opa, opb = (np.asarray(v, dtype=np.int64) for v in (n, opa, opb))
    valid = np.arange(MAX_LEN)[None, :] < n[:, None]
    shifts = (_KEY_SYMBOL_BITS * np.arange(MAX_LEN, dtype=np.int64))[None, :]
    payload = np.sum(np.where(valid, sym, 0) << shifts, axis=1)
    header = (opa * 32 + opb) * 16 + n
    return (header << _KEY_PAYLOAD_BITS) | payload


def decode_x(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(sym [B, MAX_LEN], n, opa, opb) of token rows; raises on a row that is not well formed."""
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != SEQ_LEN:
        raise SeqOpsError(f"token rows must be [B, {SEQ_LEN}], got {x.shape}")
    sep = x == SEP
    if np.any(sep.sum(axis=1) != 1) or np.any(x[:, 0] != BOS):
        raise SeqOpsError("every row needs BOS first and exactly one SEP")
    n = np.argmax(sep, axis=1) - 3
    sym = np.zeros((x.shape[0], MAX_LEN), dtype=np.int64)
    valid = np.arange(MAX_LEN)[None, :] < n[:, None]
    sym[valid] = x[:, 3 : 3 + MAX_LEN][valid] - SYMBOL_BASE
    rebuilt, _ = encode_rows(sym, n, x[:, 1], x[:, 2], np.zeros_like(sym))
    if not np.array_equal(rebuilt, x):
        raise SeqOpsError("token rows are not well formed")
    return sym, n, x[:, 1].copy(), x[:, 2].copy()


@dataclass(frozen=True)
class RowSpec:
    """A distribution over rows: operator pairs, a length range and the share of two-operator rows."""

    opa: tuple[int, ...]
    opb: tuple[int, ...]
    lengths: tuple[int, int]
    pair_fraction: float
    """Probability that OPB is drawn from ``opb``; otherwise OPB = NONE."""
    pairs: tuple[tuple[int, int], ...] = field(default=())
    """If given, (OPA, OPB) is drawn uniformly from these pairs instead (``opa``/``opb``/``pair_fraction`` unused)."""

    def sample(self, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lo, hi = self.lengths
        if not MIN_LEN <= lo <= hi <= MAX_LEN:
            raise SeqOpsError(f"length range {self.lengths} outside {MIN_LEN}..{MAX_LEN}")
        n = rng.integers(lo, hi + 1, size=size).astype(np.int64)
        if self.pairs:
            chosen = rng.integers(0, len(self.pairs), size=size)
            table = np.asarray(self.pairs, dtype=np.int64)
            opa, opb = table[chosen, 0], table[chosen, 1]
        else:
            opa = np.asarray(self.opa, dtype=np.int64)[rng.integers(0, len(self.opa), size=size)]
            second = np.asarray(self.opb, dtype=np.int64)[rng.integers(0, len(self.opb), size=size)]
            opb = np.where(rng.random(size) < self.pair_fraction, second, NONE).astype(np.int64)
        sym = rng.integers(0, N_SYMBOLS, size=(size, MAX_LEN)).astype(np.int64)
        sym[np.arange(MAX_LEN)[None, :] >= n[:, None]] = 0
        return sym, n, opa, opb

    def as_dict(self) -> dict[str, Any]:
        return {
            "opa": [TOKENS[t] for t in self.opa],
            "opb": [TOKENS[t] for t in self.opb],
            "lengths": list(self.lengths),
            "pair_fraction": self.pair_fraction,
            "pairs": [[TOKENS[a], TOKENS[b]] for a, b in self.pairs],
        }


OLD_ROWS = RowSpec(opa=OLD_OP_IDS, opb=OLD_OP_IDS, lengths=(MIN_LEN, MAX_LEN), pair_fraction=PAIR_FRACTION)
"""Old-operator rows: the base training distribution and the old held-out splits."""


def new_single_rows(new_token: int, lengths: tuple[int, int]) -> RowSpec:
    """Single-operator rows of a reserved operator."""
    return RowSpec(opa=(new_token,), opb=(NONE,), lengths=lengths, pair_fraction=0.0)


def new_composed_rows(new_token: int, lengths: tuple[int, int]) -> RowSpec:
    """A reserved operator composed with every old operator, both orders, uniform over the pairs."""
    pairs = tuple((new_token, old) for old in OLD_OP_IDS) + tuple((old, new_token) for old in OLD_OP_IDS)
    return RowSpec(opa=(), opb=(), lengths=lengths, pair_fraction=0.0, pairs=pairs)


def draw_unique(spec: RowSpec, size: int, rng: np.random.Generator, exclude: np.ndarray) -> tuple[np.ndarray, ...]:
    """``size`` rows of ``spec`` with keys distinct from each other and from ``exclude`` (sorted int64).

    Rows are drawn in rounds of ``size`` and kept in draw order while new; the result is a function of the generator
    state alone. Returns (sym, n, opa, opb, key).
    """
    if size <= 0:
        raise SeqOpsError("size must be positive")
    kept: list[tuple[np.ndarray, ...]] = []
    seen = np.asarray(exclude, dtype=np.int64)
    if seen.size and np.any(np.diff(seen) <= 0):
        raise SeqOpsError("exclude must be sorted and unique")
    have = 0
    for _ in range(64):
        sym, n, opa, opb = spec.sample(rng, size)
        key = row_key(sym, n, opa, opb)
        _, first = np.unique(key, return_index=True)
        fresh = np.zeros(size, dtype=bool)
        fresh[first] = True
        fresh &= ~np.isin(key, seen)
        order = np.flatnonzero(fresh)[: size - have]
        kept.append((sym[order], n[order], opa[order], opb[order], key[order]))
        seen = np.union1d(seen, key[order])
        have += order.size
        if have == size:
            return tuple(np.concatenate([part[i] for part in kept]) for i in range(5))
    raise SeqOpsError(f"could not draw {size} distinct rows from {spec.as_dict()}")


def make_split(
    spec: RowSpec, size: int, rng: np.random.Generator, exclude: np.ndarray, rules: Mapping[int, Rule], part: int
) -> dict[str, np.ndarray]:
    """One split's arrays: x, y [N, SEQ_LEN]; idx (the row key), n, opa, opb, part [N]."""
    sym, n, opa, opb, key = draw_unique(spec, size, rng, exclude)
    out = apply_ops(sym, n, opa, opb, rules)
    x, y = encode_rows(sym, n, opa, opb, out)
    return {"x": x, "y": y, "idx": key, "n": n, "opa": opa, "opb": opb, "part": np.full(size, part, dtype=np.int64)}


def concat_splits(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate split arrays key by key (all parts must have the same keys)."""
    keys = list(parts[0])
    if any(list(p) != keys for p in parts):
        raise SeqOpsError("parts differ in keys")
    return {key: np.concatenate([np.asarray(p[key]) for p in parts]) for key in keys}


class TrainStream:
    """Base training batches of old-operator rows, refusing every key in ``exclude``.

    Batch ``step`` comes from ``default_rng([SPLIT_SEED, TRAIN_STREAM, model_seed, step])``: rows are drawn in rounds
    of ``batch_size`` and kept in draw order while their key is not excluded, so a batch is a function of (model_seed,
    step) and the exclusion set alone. Duplicate keys inside a batch are allowed (training rows are i.i.d.).
    """

    def __init__(self, model_seed: int, batch_size: int, exclude: np.ndarray) -> None:
        if batch_size <= 0:
            raise SeqOpsError("batch_size must be positive")
        self.model_seed = int(model_seed)
        self.batch_size = int(batch_size)
        self.exclude = np.unique(np.asarray(exclude, dtype=np.int64))
        self.rules = old_rules()
        self.n_refused = 0

    def batch(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng([SPLIT_SEED, TRAIN_STREAM, self.model_seed, int(step)])
        chunks: list[tuple[np.ndarray, ...]] = []
        have = 0
        for _ in range(64):
            sym, n, opa, opb = OLD_ROWS.sample(rng, self.batch_size)
            keep = ~np.isin(row_key(sym, n, opa, opb), self.exclude)
            self.n_refused += int(np.count_nonzero(~keep))
            order = np.flatnonzero(keep)[: self.batch_size - have]
            chunks.append((sym[order], n[order], opa[order], opb[order]))
            have += order.size
            if have == self.batch_size:
                sym, n, opa, opb = (np.concatenate([c[i] for c in chunks]) for i in range(4))
                return encode_rows(sym, n, opa, opb, apply_ops(sym, n, opa, opb, self.rules))
        raise SeqOpsError("training batch could not avoid the excluded keys")


def keys_of(splits: Iterable[Mapping[str, np.ndarray]]) -> np.ndarray:
    """Sorted unique row keys of the given splits."""
    arrays = [np.asarray(s["idx"], dtype=np.int64) for s in splits]
    return np.unique(np.concatenate(arrays)) if arrays else np.zeros(0, dtype=np.int64)


def row_correct(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Exact match per row: every labelled position (y != IGNORE) predicted right. Needs a label in every row."""
    y, pred = np.asarray(y), np.asarray(pred)
    if y.shape != pred.shape or y.ndim != 2:
        raise SeqOpsError(f"y {y.shape} and pred {pred.shape} must be equal [N, L]")
    labelled = y != IGNORE
    if np.any(~labelled.any(axis=1)):
        raise SeqOpsError("a row has no labelled position")
    return np.all((pred == y) | ~labelled, axis=1)


def token_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Share of labelled positions predicted right."""
    y, pred = np.asarray(y), np.asarray(pred)
    labelled = y != IGNORE
    return float(np.mean(pred[labelled] == y[labelled]))


# ---------------------------------------------------------------------------------------------------------------
# Stored splits
# ---------------------------------------------------------------------------------------------------------------

PART_NEW_SHORT, PART_NEW_LONG, PART_NEW_COMPOSED, PART_OLD = 1, 2, 3, 4
PART_NAMES: Mapping[int, str] = MappingProxyType({
    PART_NEW_SHORT: "new operator alone, lengths of the examples",
    PART_NEW_LONG: "new operator alone, longer lengths",
    PART_NEW_COMPOSED: "new operator composed with an old operator, both orders, lengths of the examples",
    PART_OLD: "old operators (retention), single and composed, all lengths",
})
NEW_SHORT_LENGTHS = (MIN_LEN, 8)
NEW_LONG_LENGTHS = (9, MAX_LEN)

BASE_ORDER: tuple[str, ...] = ("V_select", "V_evidence", "V_check", "test")
"""Held-out old-operator splits of the dataset, drawn in this order, disjoint by key; base training refuses them."""
BASE_SIZES: Mapping[str, int] = MappingProxyType({"V_select": 4000, "V_evidence": 2000, "V_check": 4000, "test": 4000})
BASE_STREAMS: Mapping[str, int] = MappingProxyType(
    {"V_select": 1101, "V_evidence": 1102, "V_check": 1103, "test": 1104, "train": 1105}
)
TRAIN_SAMPLE_SIZE = 4000
"""Stored ``train`` split: a sample of the training distribution (the replay pool), disjoint from the held-out
splits. Base training does not refuse its keys: like a replay row of an image dataset, it is ordinary training data."""


def build_base_splits() -> dict[str, dict[str, np.ndarray]]:
    """The dataset's own splits: held-out old-operator rows (part 4) and the train sample, all from fixed streams."""
    rules = old_rules()
    splits: dict[str, dict[str, np.ndarray]] = {}
    taken = np.zeros(0, dtype=np.int64)
    for name in BASE_ORDER:
        rng = np.random.default_rng([SPLIT_SEED, BASE_STREAMS[name]])
        splits[name] = make_split(OLD_ROWS, BASE_SIZES[name], rng, taken, rules, PART_OLD)
        taken = np.union1d(taken, splits[name]["idx"])
    rng = np.random.default_rng([SPLIT_SEED, BASE_STREAMS["train"]])
    splits["train"] = make_split(OLD_ROWS, TRAIN_SAMPLE_SIZE, rng, taken, rules, PART_OLD)
    return splits


def held_out_keys(splits: Mapping[str, Mapping[str, np.ndarray]]) -> np.ndarray:
    """Keys base training refuses: every held-out split of the dataset (not the train sample)."""
    return keys_of(splits[name] for name in BASE_ORDER)


CONDITION_ORDER: tuple[str, ...] = ("V_select", "V_evidence", "V_check", "test")
CONDITION_STREAM_BASE = 1130
"""Streams of reserved operator k's condition splits: CONDITION_STREAM_BASE + 10 * k + position in CONDITION_ORDER."""
CONDITION_SIZES: Mapping[str, Mapping[int, int]] = MappingProxyType({
    "V_select": MappingProxyType({PART_NEW_SHORT: 2000, PART_NEW_LONG: 2000, PART_NEW_COMPOSED: 2000}),
    "V_evidence": MappingProxyType({PART_NEW_SHORT: 2000}),
    "V_check": MappingProxyType({PART_NEW_SHORT: 4000}),
    "test": MappingProxyType({PART_NEW_SHORT: 2000, PART_NEW_LONG: 2000, PART_NEW_COMPOSED: 2000}),
})
"""Default sizes of a new-skill condition. The old part (4) of a check is the dataset's own split of the same name.
V_check holds part 1 only by default, so no feedback reaches the longer-length and composed parts."""


def condition_name(index: int, kind: str) -> str:
    return f"new{index}_{kind}"


def part_spec(part: int, new_token: int) -> RowSpec:
    if part == PART_NEW_SHORT:
        return new_single_rows(new_token, NEW_SHORT_LENGTHS)
    if part == PART_NEW_LONG:
        return new_single_rows(new_token, NEW_LONG_LENGTHS)
    if part == PART_NEW_COMPOSED:
        return new_composed_rows(new_token, NEW_SHORT_LENGTHS)
    raise SeqOpsError(f"part {part} is not a new-operator part")


def build_condition_splits(
    rule: Rule, index: int, sizes: Mapping[str, Mapping[int, int]] = CONDITION_SIZES
) -> dict[str, dict[str, np.ndarray]]:
    """Splits of the new-skill condition of reserved operator ``index`` bound to ``rule`` (parts 1 to 3 only).

    Splits are drawn in CONDITION_ORDER, parts in increasing order within a split, each split from its own stream;
    keys are disjoint across all of them. They never coincide with a base row (those hold no reserved operator).
    """
    if rule.token != NEW_OP_IDS[index]:
        raise SeqOpsError(f"rule is bound to {TOKENS[rule.token]}, not OP_NEW{index}")
    rules = {**old_rules(), rule.token: rule}
    splits: dict[str, dict[str, np.ndarray]] = {}
    taken = np.zeros(0, dtype=np.int64)
    for position, name in enumerate(CONDITION_ORDER):
        rng = np.random.default_rng([SPLIT_SEED, CONDITION_STREAM_BASE + 10 * index + position])
        parts = []
        for part in sorted(sizes[name]):
            arrays = make_split(part_spec(part, rule.token), sizes[name][part], rng, taken, rules, part)
            taken = np.union1d(taken, arrays["idx"])
            parts.append(arrays)
        splits[name] = concat_splits(parts)
    return splits


CHECK_OLD_PART_SPLITS: tuple[str, ...] = ("V_select", "V_check", "test")
"""Stored condition splits that also hold the old part (4): the dataset's own split of the same name is appended after
the new parts, so one file holds every part its check or readout reads. V_evidence holds the new part only; the old
rows of an evidence draw come from the train sample."""


def condition_store_splits(
    rule: Rule, index: int, base: Mapping[str, Mapping[str, np.ndarray]] | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """The stored splits of new-skill condition ``index``: ``build_condition_splits``, with the base split of the same
    name (``build_base_splits``, or ``base``) appended as part 4 to each split in CHECK_OLD_PART_SPLITS."""
    base = build_base_splits() if base is None else base
    new = build_condition_splits(rule, index)
    return {
        name: concat_splits([new[name], base[name]]) if name in CHECK_OLD_PART_SPLITS else dict(new[name])
        for name in CONDITION_ORDER
    }


def symbol_agreement(table: Sequence[int]) -> dict[str, Any]:
    """How close a substitution table is to what the base already computes on symbols (recorded, never a refusal).

    Agreement with a map f is the share of the N_SYMBOLS symbols s with table[s] == f(s). The maps are the identity
    (what every position map does to a symbol), each old symbol map, and every composition b(a(s)) of two old symbol
    maps, a first as in a two-operator row.
    """
    t = np.asarray(table, dtype=np.int64)
    if sorted(t.tolist()) != list(range(N_SYMBOLS)):
        raise SeqOpsError("table must be a permutation of the symbols")
    maps = {name: np.asarray(OLD_TABLES[name], dtype=np.int64) for name in SYMBOL_OPS}
    single = {name: float(np.mean(t == m)) for name, m in maps.items()}
    composed = {f"{a}+{b}": float(np.mean(t == maps[b][maps[a]])) for a in SYMBOL_OPS for b in SYMBOL_OPS}
    identity = float(np.mean(t == np.arange(N_SYMBOLS)))
    return {
        "definition": "share of symbols s with table[s] == f(s), for f the identity, an old symbol map, or b(a(s)) "
        "with old symbol maps a then b",
        "identity": identity,
        "single": single,
        "composed": composed,
        "max_single": max(single.values()),
        "max_composed": max(composed.values()),
        "max_overall": max(identity, *single.values(), *composed.values()),
    }


def symbol_coverage(x: np.ndarray) -> int:
    """Number of distinct symbols in the input strings of token rows."""
    sym, n, _, _ = decode_x(x)
    valid = np.arange(MAX_LEN)[None, :] < n[:, None]
    return int(np.unique(sym[valid]).size)


def format_manifest() -> dict[str, Any]:
    """The row format and operator definitions recorded in dataset.json."""
    return {
        "tokens": list(TOKENS),
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "n_symbols": N_SYMBOLS,
        "lengths": [MIN_LEN, MAX_LEN],
        "row": "[BOS, OPA, OPB, x_1..x_n, SEP, OUT x n], right-padded with PAD; y = OPB(OPA(x)), OPB = NONE applies "
        "OPA alone; target y[k-th OUT position] = output symbol k, IGNORE (-1) elsewhere; a row is right when every "
        "labelled position is right",
        "old_operators": [rule.as_dict() for rule in old_rules().values()],
        "position_maps": {
            "REV": "out[k] = in[n - 1 - k]",
            "ROTL1": "out[k] = in[(k + 1) mod n]",
            "ROTR1": "out[k] = in[(k - 1) mod n]",
            "SWAP2": "out[k] = in[k xor 1] if (k xor 1) < n else in[k]",
        },
        "reserved_operators": [f"OP_{name}" for name in NEW_OPS],
        "old_rows": OLD_ROWS.as_dict(),
        "row_key": "((OPA * 32 + OPB) * 16 + n) << 48 | sum_k x_k << (4 k), int64",
        "generator": "numpy.random.default_rng([split_seed, stream]) (PCG64)",
        "train_batches": "default_rng([split_seed, 1100, model_seed, step]); rows whose key is in a held-out split are "
        "refused and redrawn",
    }


def token_meanings() -> list[dict[str, Any]]:
    """Public meaning of every token for a student's model_spec: the trained format and operators.

    Reserved operators are described only as reserved and untrained; which one a task binds, and to what, is the
    task's business, never the model's.
    """
    meanings = {
        "PAD": "padding after the last OUT slot; attention ignores it as a key",
        "BOS": "first token of every row",
        "SEP": "ends the input string",
        "OUT": "one output slot per input symbol; the prediction at the k-th OUT slot is output symbol k",
        "NONE": "second operator slot left empty: the row applies its first operator alone",
        "OP_REV": "operator, position map: out[k] = in[n - 1 - k]",
        "OP_ROTL1": "operator, position map: out[k] = in[(k + 1) mod n]",
        "OP_ROTR1": "operator, position map: out[k] = in[(k - 1) mod n]",
        "OP_SWAP2": "operator, position map: out[k] = in[k xor 1] if (k xor 1) < n else in[k]",
        "OP_INC": f"operator, symbol map: s -> (s + 1) mod {N_SYMBOLS}",
        "OP_DEC": f"operator, symbol map: s -> (s - 1) mod {N_SYMBOLS}",
        "OP_COMP": f"operator, symbol map: s -> {N_SYMBOLS - 1} - s",
        "OP_SUBA": "operator, symbol map: s -> table[s], table = " + str([int(v) for v in OLD_TABLES["SUBA"]]),
    }
    rows = []
    for index, name in enumerate(TOKENS):
        if name.startswith("s") and name[1:].isdigit():
            meaning = f"symbol {name[1:]}"
        elif name in meanings:
            meaning = meanings[name]
        else:
            meaning = "reserved operator token; absent from all training rows (its embedding row is untrained)"
        rows.append({"id": index, "name": name, "meaning": meaning})
    return rows


INPUT_LAYOUT = (
    "[BOS, OPA, OPB, x_1 .. x_n, SEP, OUT x n], right-padded with PAD to seq_len; n from 4 to 12; the answer is "
    "OPB(OPA(x)) (OPA applied first), one symbol per OUT slot"
)
