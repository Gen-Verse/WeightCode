"""PREREG E13 "T11 SCAN controls and M04" (Execution Plan v1.1 section 16, mechanism plan section 9): the pieces that
read the E13 finals and references beside the bar. Pure functions on arrays and parameter dicts; the script
``scripts/e13/t11_scan.py`` binds them to the instances, the finals and the records. No LLM, no sealed read.

* Command-structure groups (``command_groups``): every V_select command gets one value per dimension, computed from
  the command string alone and fixed before any readout. Generic dimensions (both parts): ``conjunction`` (none, and,
  after), ``repetition`` (none, twice, thrice, twice+thrice), ``turns`` (the highest direction modifier in the command:
  none < plain < opposite < around), ``turn_verb`` (the verb turn present: no, yes), ``words`` (1 to 9) and ``actions``
  (the length of the meaning, bins 1-8, 9-16, 17-24, 25-48). New-part dimensions per condition: addprim_jump
  ``jump_unit`` (the modifier of the phrase that holds jump: bare, plain, opposite, around; two_phrases when both
  phrases hold it), ``jump_rep`` (that phrase's repetition) and ``other_phrase`` (none, jump, primitive, turn);
  around_right ``ar_phrases`` (1 or 2 phrases hold "around right"), ``ar_rep`` and ``other_phrase`` (none, around
  right, around left, opposite, plain, bare).
* External renaming wrappers on the base (``rename_jump_predictions``, ``swap_lr_predictions``): a consistent,
  meaning-preserving renaming of the new symbol into an old one, applied outside the network, with the output renamed
  back. addprim_jump: jump becomes a primitive p the command does not hold, and I_p becomes I_JUMP in the output
  (consistent on exactly the rows without p). around_right: left and right swap in the command and I_TURN_LEFT and
  I_TURN_RIGHT swap in the output, on the rows that hold "around right" (consistent on every row; in the base's
  training distribution only when the command holds no "around left").
* An isomorphic relabelling of a student (``Relabel``): token ids of the 13 words and class ids of the 6 actions
  permuted, the embedding rows and the head rows and biases moved with them, the layout's token table and class list
  rewritten; the network computes the same function under the renaming.
* The component networks of the addprim_jump final (``component_children``): E only (the final's jump embedding row),
  H only (the final's I_JUMP head row and bias) and E+H, which must equal the final bit for bit.
* A traced forward (``traced_forward``) that runs a dataset forward.py's own functions, in its own chunks, and returns
  the residual stream after each block and the head input at chosen positions; its logits must equal forward.py's
  bit for bit.
* Slot-level readouts (``slot_readout``) and hidden-state comparisons (``state_comparison``).
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from weightcode.data import scan as S
from weightcode.evalkit.metrics import IGNORE, rows_right, token_accuracy

EMBEDDING = "tok_emb.weight"
HEAD_WEIGHT = "head.weight"
HEAD_BIAS = "head.bias"
PRIMITIVES_TRAINED: tuple[str, ...] = ("walk", "look", "run")
"""addprim_jump: the primitives the base was trained on in every construction, in the fixed order the wrappers try."""
MODIFIERS: tuple[str, ...] = ("bare", "plain", "opposite", "around")
TURN_LEVELS: tuple[str, ...] = ("none", "plain", "opposite", "around")
REPETITIONS: tuple[str, ...] = ("none", "twice", "thrice")
ACTION_BINS: tuple[tuple[int, int], ...] = ((1, 8), (9, 16), (17, 24), (25, S.OUT_SLOTS))
SLOT_BINS: tuple[tuple[int, int], ...] = ((0, 7), (8, 15), (16, 23), (24, S.OUT_SLOTS - 1))
GENERIC_DIMENSIONS: tuple[str, ...] = ("conjunction", "repetition", "turns", "turn_verb", "words", "actions")
NEW_DIMENSIONS: Mapping[str, tuple[str, ...]] = {
    "addprim_jump": ("jump_unit", "jump_rep", "other_phrase"),
    "around_right": ("ar_phrases", "ar_rep", "other_phrase"),
}
DIMENSION_VALUES: Mapping[str, tuple[str, ...]] = {
    "conjunction": ("none", "and", "after"),
    "repetition": ("none", "twice", "thrice", "twice+thrice"),
    "turns": TURN_LEVELS,
    "turn_verb": ("no", "yes"),
    "words": tuple(str(n) for n in range(1, S.MAX_CMD + 1)),
    "actions": tuple(f"{lo}-{hi}" for lo, hi in ACTION_BINS),
    "jump_unit": (*MODIFIERS, "two_phrases"),
    "jump_rep": (*REPETITIONS, "two_phrases"),
    "ar_phrases": ("1", "2"),
    "ar_rep": (*REPETITIONS, "two_phrases"),
    "other_phrase": ("none", "jump", "primitive", "turn", "around right", "around left", "opposite", "plain", "bare"),
}
"""The value order of every dimension (a readout lists the values present, in this order)."""


class ScanT11Error(ValueError):
    """A T11 SCAN control was refused: a command outside the grammar, an inconsistent relabelling, a final that changes
    more than its components, or a traced forward that differs from forward.py."""


# ---------------------------------------------------------------------------------------------------------------
# Command structure


@dataclasses.dataclass(frozen=True)
class Phrase:
    """One phrase S of a SCAN command: verb (walk, look, run, jump or turn), modifier (bare, plain, opposite, around),
    direction (left, right or None) and repetition (none, twice, thrice)."""

    verb: str
    modifier: str
    direction: str | None
    rep: str

    def holds(self, word: str) -> bool:
        return word in (self.verb, self.direction) or (word == self.modifier and word in ("opposite", "around"))


def parse_phrase(words: Sequence[str]) -> Phrase:
    """The phrase of ``words`` (S -> V | V twice | V thrice); raises on anything the grammar does not produce."""
    words = list(words)
    rep = words[-1] if words and words[-1] in ("twice", "thrice") else "none"
    unit = words[:-1] if rep != "none" else words
    if not unit or unit[0] not in (*S.VERBS, "turn"):
        raise ScanT11Error(f"{' '.join(words)!r} is not a SCAN phrase")
    verb = unit[0]
    if len(unit) == 1 and verb != "turn":
        phrase = Phrase(verb, "bare", None, rep)
    elif len(unit) == 2 and unit[1] in ("left", "right"):
        phrase = Phrase(verb, "plain", unit[1], rep)
    elif len(unit) == 3 and unit[1] in ("opposite", "around") and unit[2] in ("left", "right"):
        phrase = Phrase(verb, unit[1], unit[2], rep)
    else:
        raise ScanT11Error(f"{' '.join(words)!r} is not a SCAN phrase")
    S.interpret(words)  # the interpretation function accepts exactly the grammar's phrases
    return phrase


def parse_command(command: str) -> tuple[str, tuple[Phrase, ...]]:
    """(conjunction, phrases) of a SCAN command: conjunction none, and or after; one or two phrases in word order."""
    words = command.split()
    joins = [k for k, word in enumerate(words) if word in ("and", "after")]
    if len(joins) > 1:
        raise ScanT11Error(f"{command!r} has more than one conjunction")
    if not joins:
        return "none", (parse_phrase(words),)
    k = joins[0]
    return words[k], (parse_phrase(words[:k]), parse_phrase(words[k + 1:]))


def _bin(value: int, bins: Sequence[tuple[int, int]]) -> str:
    for lo, hi in bins:
        if lo <= value <= hi:
            return f"{lo}-{hi}"
    raise ScanT11Error(f"{value} lies in no bin of {bins}")


def _is_around_right(phrase: Phrase) -> bool:
    return phrase.modifier == "around" and phrase.direction == "right"


def _other_kind(phrase: Phrase) -> str:
    if phrase.modifier == "around":
        return f"around {phrase.direction}"
    return phrase.modifier


def command_groups(command: str, condition: str, part: str) -> dict[str, str]:
    """One value per dimension for one command (module docstring); ``part`` is "new" or "old", and only a new command
    gets its condition's new-part dimensions."""
    if condition not in S.CONDITIONS or part not in ("new", "old"):
        raise ScanT11Error(f"unknown condition {condition!r} or part {part!r}")
    conjunction, phrases = parse_command(command)
    reps = [rep for rep in REPETITIONS[1:] if any(phrase.rep == rep for phrase in phrases)]
    turn_level = max((TURN_LEVELS.index("none" if p.modifier == "bare" else p.modifier) for p in phrases))
    groups = {
        "conjunction": conjunction,
        "repetition": "+".join(reps) if reps else "none",
        "turns": TURN_LEVELS[turn_level],
        "turn_verb": "yes" if any(p.verb == "turn" for p in phrases) else "no",
        "words": str(len(command.split())),
        "actions": _bin(len(S.interpret(command)), ACTION_BINS),
    }
    if part == "old":
        if S.has_primitive(command, condition):
            raise ScanT11Error(f"{condition}: old command {command!r} holds the new part's words")
        return groups
    if not S.has_primitive(command, condition):
        raise ScanT11Error(f"{condition}: new command {command!r} lacks {S.PRIMITIVE[condition]}")
    if condition == "addprim_jump":
        holders = [p for p in phrases if p.verb == "jump"]
        others = [p for p in phrases if p.verb != "jump"]
        both = len(holders) == 2
        groups["jump_unit"] = "two_phrases" if both else holders[0].modifier
        groups["jump_rep"] = "two_phrases" if both else holders[0].rep
        groups["other_phrase"] = ("none" if len(phrases) == 1 else "jump" if both
                                  else "turn" if others[0].verb == "turn" else "primitive")
    else:
        holders = [p for p in phrases if _is_around_right(p)]
        others = [p for p in phrases if not _is_around_right(p)]
        both = len(holders) == 2
        groups["ar_phrases"] = str(len(holders))
        groups["ar_rep"] = "two_phrases" if both else holders[0].rep
        groups["other_phrase"] = ("none" if len(phrases) == 1 else "around right" if both else _other_kind(others[0]))
    return groups


def dimensions(condition: str, part: str) -> tuple[str, ...]:
    """The dimensions a part is grouped by."""
    return GENERIC_DIMENSIONS + (NEW_DIMENSIONS[condition] if part == "new" else ())


def part_readout(y: np.ndarray, base: np.ndarray, child: np.ndarray, rows: np.ndarray | None = None
                 ) -> dict[str, Any]:
    """n, exact match of base and child, delta, fixed, broken and token accuracies on ``rows`` (all rows when None);
    ``{"n": 0}`` on an empty selection."""
    y, base, child = np.asarray(y), np.asarray(base), np.asarray(child)
    if rows is not None:
        rows = np.asarray(rows, dtype=bool)
        if not rows.any():
            return {"n": 0}
        y, base, child = y[rows], base[rows], child[rows]
    rb, rc = rows_right(y, base), rows_right(y, child)
    em_base, em_child = float(np.mean(rb)), float(np.mean(rc))
    return {"n": int(len(y)), "em_base": em_base, "em_child": em_child, "delta": em_child - em_base,
            "fixed": int(np.count_nonzero(~rb & rc)), "broken": int(np.count_nonzero(rb & ~rc)),
            "token_acc_base": token_accuracy(y, base), "token_acc_child": token_accuracy(y, child)}


def grouped_readout(commands: Sequence[str], condition: str, part: str, y: np.ndarray, base: np.ndarray,
                    child: np.ndarray) -> dict[str, dict[str, dict[str, Any]]]:
    """``part_readout`` per value of every dimension of the part, the values present listed in ``DIMENSION_VALUES``
    order; every row counts once per dimension."""
    groups = [command_groups(command, condition, part) for command in commands]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dim in dimensions(condition, part):
        values = [g[dim] for g in groups]
        unknown = sorted(set(values) - set(DIMENSION_VALUES[dim]))
        if unknown:
            raise ScanT11Error(f"dimension {dim} has values outside its list: {unknown}")
        column = np.array(values)
        out[dim] = {value: part_readout(y, base, child, column == value)
                    for value in DIMENSION_VALUES[dim] if (column == value).any()}
    return out


# ---------------------------------------------------------------------------------------------------------------
# External renaming wrappers


PredictFn = Callable[[np.ndarray], np.ndarray]
"""Argmax predictions [N, L] of a fixed network for token rows [N, L]."""


def replace_tokens(x: np.ndarray, mapping: Mapping[int, int]) -> np.ndarray:
    """``x`` with every token id in ``mapping`` replaced at once (a swap maps both ways)."""
    x = np.asarray(x)
    out = x.copy()
    for source, target in mapping.items():
        out[x == source] = target
    return out


def replace_classes(pred: np.ndarray, mapping: Mapping[int, int]) -> np.ndarray:
    """``pred`` with every class id in ``mapping`` replaced at once; IGNORE and unmapped classes unchanged."""
    pred = np.asarray(pred)
    out = pred.copy()
    for source, target in mapping.items():
        out[pred == source] = target
    return out


def rename_choice(command: str, choice: str) -> str | None:
    """The primitive jump is renamed to in ``command``: ``choice`` itself when the command does not hold it (None when
    it does: the renaming would not be consistent), or for ``auto`` the first of ``PRIMITIVES_TRAINED`` it does not
    hold."""
    words = set(command.split())
    if "jump" not in words:
        raise ScanT11Error(f"{command!r} holds no jump")
    if choice == "auto":
        return next(p for p in PRIMITIVES_TRAINED if p not in words)
    if choice not in PRIMITIVES_TRAINED:
        raise ScanT11Error(f"rename choice {choice!r} is not auto or one of {PRIMITIVES_TRAINED}")
    return None if choice in words else choice


def rename_jump_predictions(predict: PredictFn, x: np.ndarray, choice: str) -> tuple[np.ndarray, np.ndarray]:
    """addprim_jump renaming wrapper on new rows (each holds jump): jump becomes primitive p in the command and I_p
    becomes I_JUMP in the base's output. Returns (predictions, consistent): a row whose command holds p is renamed
    with the same p and flagged inconsistent (an I_p of the command's own p is read as I_JUMP too)."""
    x = np.asarray(x)
    commands = S.decode(x)
    picks = [rename_choice(c, choice) for c in commands]
    consistent = np.array([p is not None for p in picks], dtype=bool)
    if choice != "auto":
        picks = [choice] * len(commands)
    pred = np.full(x.shape, IGNORE, dtype=np.int64)
    for p in PRIMITIVES_TRAINED:
        rows = np.array([q == p for q in picks], dtype=bool)
        if not rows.any():
            continue
        renamed = replace_tokens(x[rows], {S.TOKEN_ID["jump"]: S.TOKEN_ID[p]})
        out = np.asarray(predict(renamed), dtype=np.int64)
        pred[rows] = replace_classes(out, {S.CLASS_ID[S.VERB_ACTION[p]]: S.CLASS_ID["I_JUMP"]})
    return pred, consistent


def swap_lr_predictions(predict: PredictFn, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """around_right renaming wrapper: on the rows that hold "around right", left and right swap in the command and
    I_TURN_LEFT and I_TURN_RIGHT swap in the base's output; every other row is the base's own prediction. Returns
    (predictions, applied)."""
    x = np.asarray(x)
    commands = S.decode(x)
    applied = np.array([S.has_primitive(c, "around_right") for c in commands], dtype=bool)
    pred = np.asarray(predict(x), dtype=np.int64).copy()
    if applied.any():
        left, right = S.TOKEN_ID["left"], S.TOKEN_ID["right"]
        tl, tr = S.CLASS_ID["I_TURN_LEFT"], S.CLASS_ID["I_TURN_RIGHT"]
        swapped = np.asarray(predict(replace_tokens(x[applied], {left: right, right: left})), dtype=np.int64)
        pred[applied] = replace_classes(swapped, {tl: tr, tr: tl})
    return pred, applied


def holds_around_left(commands: Sequence[str]) -> np.ndarray:
    """Rows whose command holds the words around left next to each other (the swap renames them to around right)."""
    return np.array([any(tuple(c.split()[k:k + 2]) == ("around", "left") for k in range(len(c.split()) - 1))
                     for c in commands], dtype=bool)


# ---------------------------------------------------------------------------------------------------------------
# An isomorphic relabelling of a student


@dataclasses.dataclass(frozen=True)
class Relabel:
    """A permutation of token ids and class ids: ``token_map[old] = new``, ``class_map[old] = new``."""

    token_map: tuple[int, ...]
    class_map: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, values, size in (("token_map", self.token_map, S.VOCAB_SIZE),
                                   ("class_map", self.class_map, S.N_CLASSES)):
            if sorted(values) != list(range(size)):
                raise ScanT11Error(f"{name} {values} is not a permutation of 0 .. {size - 1}")
        if self.token_map[S.PAD] != S.PAD:
            raise ScanT11Error("PAD keeps its id (forward.py masks PAD keys by id)")

    @property
    def token_inverse(self) -> np.ndarray:
        return np.argsort(np.asarray(self.token_map))

    @property
    def class_inverse(self) -> np.ndarray:
        return np.argsort(np.asarray(self.class_map))

    def as_dict(self) -> dict[str, Any]:
        return {"token_map": {S.TOKENS[i]: int(j) for i, j in enumerate(self.token_map)},
                "class_map": {S.CLASS_NAMES[i]: int(j) for i, j in enumerate(self.class_map)}}


WORD_SHIFT = 5
ACTION_SHIFT = 3


def default_relabel() -> Relabel:
    """The fixed relabelling of PREREG E13 T11: the 13 word ids (4 to 16) shifted cyclically by WORD_SHIFT within
    themselves, the 6 action classes (0 to 5) by ACTION_SHIFT; PAD, BOS, SEP, OUT and END keep their ids. No id is a
    fixed point of either shift."""
    n_special = len(S.SPECIAL_TOKENS)
    n_words = len(S.WORDS)
    tokens = list(range(n_special)) + [n_special + (k + WORD_SHIFT) % n_words for k in range(n_words)]
    n_actions = len(S.ACTIONS)
    classes = [(k + ACTION_SHIFT) % n_actions for k in range(n_actions)] + [S.END]
    return Relabel(tuple(tokens), tuple(classes))


def relabel_params(params: Mapping[str, np.ndarray], relabel: Relabel) -> dict[str, np.ndarray]:
    """The student under ``relabel``: embedding rows and head rows and biases moved to their new ids; every other
    tensor is the same array value."""
    out = {name: np.array(value, copy=True) for name, value in params.items()}
    out[EMBEDDING] = np.asarray(params[EMBEDDING])[relabel.token_inverse].copy()
    out[HEAD_WEIGHT] = np.asarray(params[HEAD_WEIGHT])[relabel.class_inverse].copy()
    out[HEAD_BIAS] = np.asarray(params[HEAD_BIAS])[relabel.class_inverse].copy()
    return out


def relabel_tokens(x: np.ndarray, relabel: Relabel) -> np.ndarray:
    return np.asarray(relabel.token_map, dtype=np.int64)[np.asarray(x, dtype=np.int64)]


def relabel_classes(y: np.ndarray, relabel: Relabel) -> np.ndarray:
    """Targets or predictions under ``relabel``; IGNORE stays IGNORE."""
    y = np.asarray(y, dtype=np.int64)
    out = y.copy()
    labelled = y != IGNORE
    out[labelled] = np.asarray(relabel.class_map, dtype=np.int64)[y[labelled]]
    return out


def relabel_layout(layout: Mapping[str, Any], relabel: Relabel) -> dict[str, Any]:
    """The layout (model_spec without model_id) under ``relabel``: each token keeps its name and meaning with its new
    id, listed in id order (a program may read an id from the list position), and the class list reordered. Refuses a
    layout whose token table or class list is not the SCAN one."""
    out = json_copy(layout)
    tokens = out["input"]["tokens"]
    if [t["name"] for t in tokens] != list(S.TOKENS) or [t["id"] for t in tokens] != list(range(S.VOCAB_SIZE)):
        raise ScanT11Error("the layout's token table is not the SCAN table in id order")
    if list(out["classes"]) != list(S.CLASS_NAMES):
        raise ScanT11Error("the layout's class list is not the SCAN class list")
    moved = [{**token, "id": int(relabel.token_map[token["id"]])} for token in tokens]
    out["input"]["tokens"] = sorted(moved, key=lambda token: token["id"])
    classes = [None] * S.N_CLASSES
    for old, name in enumerate(S.CLASS_NAMES):
        classes[relabel.class_map[old]] = name
    out["classes"] = classes
    return out


def unrelabel_classes(pred: np.ndarray, relabel: Relabel) -> np.ndarray:
    """Class ids (targets or predictions) of the relabelled student moved back to the original ids; IGNORE stays."""
    pred = np.asarray(pred, dtype=np.int64)
    out = pred.copy()
    labelled = pred != IGNORE
    out[labelled] = relabel.class_inverse[pred[labelled]]
    return out


def unrelabel_deltas(deltas: Mapping[str, np.ndarray], relabel: Relabel) -> dict[str, np.ndarray]:
    """Deltas computed on the relabelled student, moved back to the original ids."""
    out = {name: np.array(value, copy=True) for name, value in deltas.items()}
    if EMBEDDING in out:
        out[EMBEDDING] = np.asarray(deltas[EMBEDDING])[np.asarray(relabel.token_map)].copy()
    for name in (HEAD_WEIGHT, HEAD_BIAS):
        if name in out:
            out[name] = np.asarray(deltas[name])[np.asarray(relabel.class_map)].copy()
    return out


def json_copy(value: Any) -> Any:
    import json

    return json.loads(json.dumps(value))


# ---------------------------------------------------------------------------------------------------------------
# Component networks of the addprim_jump final


def changed_entries(base: Mapping[str, np.ndarray], child: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per tensor, the boolean mask of entries whose float32 value differs (tensors without a change left out)."""
    if sorted(base) != sorted(child):
        raise ScanT11Error("base and child have different tensors")
    out = {}
    for name in base:
        a, b = np.asarray(base[name]), np.asarray(child[name])
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ScanT11Error(f"{name}: {a.dtype}{a.shape} against {b.dtype}{b.shape}")
        mask = a != b
        if mask.any():
            out[name] = mask
    return out


def component_children(base: Mapping[str, np.ndarray], final: Mapping[str, np.ndarray], token: str = "jump",
                       action: str = "I_JUMP") -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """E only, H only and E+H (M04): the final's ``token`` embedding row, its ``action`` head row and bias, and both,
    each put on the base; everything else is the base's. Refuses a final that changes any other entry. Returns the
    three children and a record of the changed entries."""
    changed = changed_entries(base, final)
    t, c = S.TOKEN_ID[token], S.CLASS_ID[action]
    allowed = {EMBEDDING: ("row", t), HEAD_WEIGHT: ("row", c), HEAD_BIAS: ("entry", c)}
    for name, mask in changed.items():
        if name not in allowed:
            raise ScanT11Error(f"the final changes {name}, which is neither component")
        outside = mask.copy()
        outside[allowed[name][1]] = False
        if outside.any():
            raise ScanT11Error(f"the final changes {name} outside {allowed[name][0]} {allowed[name][1]}")
    record = {"changed": {name: int(mask.sum()) for name, mask in changed.items()},
              "n_changed": int(sum(int(mask.sum()) for mask in changed.values())),
              "E": f"{EMBEDDING} row {t} ({token})", "H": f"{HEAD_WEIGHT} row {c} and {HEAD_BIAS}[{c}] ({action})"}

    def put(names: Sequence[str]) -> dict[str, np.ndarray]:
        child = {name: np.array(value, copy=True) for name, value in base.items()}
        for name in names:
            child[name] = np.array(final[name], copy=True)
        return child

    children = {"E_only": put([EMBEDDING]), "H_only": put([HEAD_WEIGHT, HEAD_BIAS]),
                "EH": put([EMBEDDING, HEAD_WEIGHT, HEAD_BIAS])}
    for name, value in final.items():
        if np.asarray(value).tobytes() != children["EH"][name].tobytes():
            raise ScanT11Error(f"E+H differs from the final in {name}")
    return children, record


# ---------------------------------------------------------------------------------------------------------------
# Traced forward


def traced_forward(forward: Callable[..., np.ndarray], params: Mapping[str, np.ndarray], x: np.ndarray,
                   keep: np.ndarray, with_blocks: bool = True) -> dict[str, Any]:
    """Run a SCAN forward.py's own functions (taken from ``forward.__globals__``) in its own chunks and return
    ``logits`` (float32 [N, L, C], as forward.py casts them), ``blocks`` (the residual stream after each block, float64
    [n_keep, d] per block, at the positions ``keep`` marks, row-major) and ``head_in`` (the head's input, ln_f of the
    last residual, float64 [n_keep, d]); ``with_blocks`` False leaves ``blocks`` empty. The caller checks the logits
    against forward.py."""
    g = forward.__globals__
    tensor, layernorm, attention, linear = g["_tensor"], g["_layernorm"], g["_attention"], g["_linear"]
    chunk, pad = int(g["CHUNK"]), int(g["PAD_ID"])
    x = np.asarray(x, dtype=np.int64)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != x.shape:
        raise ScanT11Error(f"keep {keep.shape} does not match x {x.shape}")
    blocks = g["_blocks"](params)
    logits, per_block, head_in = [], [[] for _ in blocks], []
    tok, pos = tensor(params, "tok_emb.weight"), tensor(params, "pos_emb.weight")
    for start in range(0, len(x), chunk):
        xc, kc = x[start:start + chunk], keep[start:start + chunk]
        h = tok[xc] + pos[None, :, :]
        key_pad = xc == pad
        for index, block in enumerate(blocks):
            a = layernorm(h, tensor(params, block + ".ln1.weight"), tensor(params, block + ".ln1.bias"))
            h = h + attention(a, params, block + ".attn", key_pad)
            a = layernorm(h, tensor(params, block + ".ln2.weight"), tensor(params, block + ".ln2.bias"))
            h = h + linear(np.maximum(linear(a, params, block + ".mlp.fc1"), 0.0), params, block + ".mlp.fc2")
            if with_blocks:
                per_block[index].append(h[kc])
        a = layernorm(h, tensor(params, "ln_f.weight"), tensor(params, "ln_f.bias"))
        head_in.append(a[kc])
        logits.append(linear(a, params, "head").astype(np.float32))
    width = tok.shape[1]
    return {"logits": np.concatenate(logits) if logits else np.zeros((0,) + x.shape[1:] + (0,), np.float32),
            "blocks": [np.concatenate(b) if b else np.zeros((0, width)) for b in per_block],
            "head_in": np.concatenate(head_in) if head_in else np.zeros((0, width)),
            "block_names": list(blocks)}


def max_abs_difference(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ScanT11Error(f"shapes {a.shape} and {b.shape} differ")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())


# ---------------------------------------------------------------------------------------------------------------
# Slot-level readouts


def summarize(values: np.ndarray) -> dict[str, Any]:
    """n, mean, min, p1, p5, median, p95, p99, max and the share above zero of a set of numbers."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return {"n": 0}
    q = np.percentile(v, [1, 5, 50, 95, 99])
    return {"n": int(v.size), "mean": float(v.mean()), "min": float(v.min()), "p1": float(q[0]), "p5": float(q[1]),
            "median": float(q[2]), "p95": float(q[3]), "p99": float(q[4]), "max": float(v.max()),
            "share_positive": float(np.mean(v > 0))}


def _margin(logits: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """logit[cls] - max over the other classes, per slot."""
    picked = np.take_along_axis(logits, cls[:, None], axis=1)[:, 0]
    others = logits.copy()
    np.put_along_axis(others, cls[:, None], -np.inf, axis=1)
    return picked - others.max(axis=1)


def slot_readout(y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    """Per labelled (OUT) slot: accuracy and predicted-class counts per target class; the target margin
    (logit[target] - max other) per target class; at I_JUMP target slots the W/L/R spread (max - mean of the I_WALK,
    I_LOOK and I_RUN logits) and the JUMP gap (logit[I_JUMP] - max of those three); at every other slot the JUMP
    intrusion (logit[I_JUMP] - max of the other classes; above zero means I_JUMP is predicted); accuracy per slot-index
    bin."""
    y = np.asarray(y)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape[:2] != y.shape or logits.shape[2] != S.N_CLASSES:
        raise ScanT11Error(f"logits {logits.shape} do not fit targets {y.shape}")
    labelled = y != IGNORE
    target = y[labelled].astype(np.int64)
    lg = logits[labelled]
    pred = lg.argmax(axis=1)
    slot_index = (np.cumsum(labelled, axis=1) - 1)[labelled]
    margin = _margin(lg, target)
    jump = S.CLASS_ID["I_JUMP"]
    prims = [S.CLASS_ID[S.VERB_ACTION[p]] for p in PRIMITIVES_TRAINED]
    by_target = {}
    for cls, name in enumerate(S.CLASS_NAMES):
        rows = target == cls
        if not rows.any():
            continue
        by_target[name] = {"n": int(rows.sum()), "accuracy": float(np.mean(pred[rows] == cls)),
                           "predicted": {S.CLASS_NAMES[k]: int(np.count_nonzero(pred[rows] == k))
                                         for k in range(S.N_CLASSES) if np.any(pred[rows] == k)},
                           "target_margin": summarize(margin[rows])}
    at_jump = target == jump
    wlr = lg[:, prims]
    record: dict[str, Any] = {
        "n_slots": int(labelled.sum()), "accuracy": float(np.mean(pred == target)), "by_target": by_target,
        "jump_slots": {"wlr_spread": summarize((wlr.max(axis=1) - wlr.mean(axis=1))[at_jump]),
                       "jump_gap": summarize((lg[:, jump] - wlr.max(axis=1))[at_jump])},
        "jump_intrusion": summarize(_margin(lg, np.full(len(target), jump))[~at_jump]),
        "by_slot_index": {f"{lo}-{hi}": {"n": int(rows.sum()), "accuracy": float(np.mean(pred[rows] == target[rows]))}
                          for lo, hi in SLOT_BINS if (rows := (slot_index >= lo) & (slot_index <= hi)).any()},
    }
    return record


# ---------------------------------------------------------------------------------------------------------------
# Hidden states at the decision positions


def rename_rows(x: np.ndarray, primitive: str) -> np.ndarray:
    """Every jump token of ``x`` replaced by ``primitive`` (the same structure with a trained primitive)."""
    return replace_tokens(x, {S.TOKEN_ID["jump"]: S.TOKEN_ID[primitive]})


def state_comparison(states: np.ndarray, references: Sequence[np.ndarray]) -> dict[str, Any]:
    """Per slot, a network's state against the base's states at the same slot of the renamed commands (one array per
    primitive, same row order): cosine and relative distance to the references' mean, the references' own spread
    around their mean (relative), and which reference is nearest."""
    states = np.asarray(states, dtype=np.float64)
    refs = np.stack([np.asarray(r, dtype=np.float64) for r in references])
    if refs.shape[1:] != states.shape:
        raise ScanT11Error(f"references {refs.shape} do not fit states {states.shape}")
    if states.shape[0] == 0:
        return {"n": 0}
    mean = refs.mean(axis=0)
    norm = np.linalg.norm(mean, axis=1)
    cos = np.sum(states * mean, axis=1) / (np.linalg.norm(states, axis=1) * norm)
    rel = np.linalg.norm(states - mean, axis=1) / norm
    spread = np.linalg.norm(refs - mean[None], axis=2).mean(axis=0) / norm
    nearest = np.linalg.norm(refs - states[None], axis=2).argmin(axis=0)
    return {"n": int(states.shape[0]), "cos_to_mean": summarize(cos), "rel_dist_to_mean": summarize(rel),
            "ref_spread": summarize(spread),
            "nearest": {PRIMITIVES_TRAINED[k]: int(np.count_nonzero(nearest == k)) for k in range(len(refs))}}


def predictions_sha256(pred: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(pred, dtype=np.int64)).tobytes()).hexdigest()


__all__ = [
    "ACTION_BINS", "DIMENSION_VALUES", "EMBEDDING", "GENERIC_DIMENSIONS", "HEAD_BIAS", "HEAD_WEIGHT", "MODIFIERS",
    "NEW_DIMENSIONS", "PRIMITIVES_TRAINED", "Phrase", "Relabel", "SLOT_BINS", "ScanT11Error", "bytes_equal",
    "changed_entries", "command_groups", "component_children", "default_relabel", "dimensions", "grouped_readout",
    "holds_around_left", "max_abs_difference", "parse_command", "parse_phrase", "part_readout", "predictions_sha256",
    "relabel_classes", "relabel_layout", "relabel_params", "relabel_tokens", "rename_choice",
    "unrelabel_classes",
    "rename_jump_predictions", "rename_rows", "replace_classes", "replace_tokens", "slot_readout", "state_comparison",
    "summarize", "swap_lr_predictions", "traced_forward", "unrelabel_deltas",
]
