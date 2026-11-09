"""SCAN (Lake and Baroni 2018) for PREREG E13: a known primitive inside an old composition.

A command of SCAN's grammar is a string of at most MAX_CMD words; its meaning is a sequence of at most OUT_SLOTS
actions, given by the interpretation function of the paper (``interpret``, written here from the published rules and
checked against every row of the official files). The student (arch ``tfm``) reads the token row

    [BOS, w_1 .. w_n, SEP, OUT x OUT_SLOTS]   right-padded with PAD to SEQ_LEN

and predicts one class at every OUT position: the k-th action of the command's meaning, or END at every position after
the last action. ``y`` is stored per position (the class index at the k-th OUT position, IGNORE elsewhere), and a row
is right when every OUT position is right (exact match).

Conditions (PREREG E13), each with its own student:

* ``addprim_jump``: add_prim_split/tasks_{train,test}_addprim_jump.txt. The training file holds every command without
  "jump" and the command "jump" alone (repeated); the test file holds every command that composes "jump".
* ``around_right``: template_split/tasks_{train,test}_template_around_right.txt (Loula et al. 2018). The test file
  holds the commands that contain "around right"; the training file none of them.

Splits, per condition, by one seed-100 draw (``default_rng([E13_SPLIT_SEED, SPLIT_STREAMS[condition]])``): the old
commands are the distinct commands of the official training file (for addprim_jump without "jump" alone), sorted by
command id; one permutation deals the first 1000 to old V_check, the next 1000 to old V_select and the next 1000 to
the sealed old part. Then one permutation of the official test file's commands (sorted by command id) is cut into
thirds (``numpy.array_split``): new V_check, new V_select, sealed new. The student trains on the official training
file's rows in file order without the 3000 held-out commands, with its duplicates and its "jump" rows as they are.
Every stored split is sorted by command id.

A command id is the index of the command in the sorted list of the 20,910 distinct commands of tasks.txt (the whole
grammar), so it names one command across conditions.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

DATASET = "scan"
CONDITIONS: tuple[str, ...] = ("addprim_jump", "around_right")
SCAN_URL = "https://github.com/brendenlake/SCAN"
SCAN_COMMIT = "c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f"
ALL_FILE = "tasks.txt"
CONDITION_FILES: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "addprim_jump": MappingProxyType({"train": "add_prim_split/tasks_train_addprim_jump.txt",
                                      "test": "add_prim_split/tasks_test_addprim_jump.txt"}),
    "around_right": MappingProxyType({"train": "template_split/tasks_train_template_around_right.txt",
                                      "test": "template_split/tasks_test_template_around_right.txt"}),
})
RAW_SHA256: Mapping[str, str] = MappingProxyType({
    "README.md": "a383edc2fcc023202b12cafcf937754a681637afbb4ac930ac577d2fa2036c53",
    "LICENSE": "5caff9408d400e1d9b271bf48b967154cdf6a37c16c965f91775da3af9291f67",
    "tasks.txt": "d824c7d9e9149ba497e25f701981dac79954e04bf670d22e5c209f076ac6c89f",
    "add_prim_split/tasks_train_addprim_jump.txt": "e3c78485692e028eed4205e1b8b3fd0e785ef2c12d45b5b89204dadf5b239e23",
    "add_prim_split/tasks_test_addprim_jump.txt": "8147ae2651422c701162e174d9d66158daa57858a26b30317ac61797118ac7a5",
    "template_split/tasks_train_template_around_right.txt":
        "d5d35d174d557645fc61f289424afa0d95fde21422c59a9819e2d0fcfa9d419f",
    "template_split/tasks_test_template_around_right.txt":
        "fc78a5d1077dc2c206d6c42ed89fb9022867618c3f331179bafd3ec1ca699474",
})
"""sha256 of every file copied from the SCAN repository at SCAN_COMMIT; the build refuses a file that differs."""

PRIMITIVE: Mapping[str, tuple[str, ...]] = MappingProxyType({"addprim_jump": ("jump",),
                                                             "around_right": ("around", "right")})
"""The words of each condition's new part: a command is in the new part when it contains them as adjacent words."""
PRIMITIVE_ROW: Mapping[str, str | None] = MappingProxyType({"addprim_jump": "jump", "around_right": None})
"""The command of the primitive alone in the training file (addprim_jump only), given to the teacher once."""

SPECIAL_TOKENS: tuple[str, ...] = ("PAD", "BOS", "SEP", "OUT")
VERBS: tuple[str, ...] = ("walk", "look", "run", "jump")
WORDS: tuple[str, ...] = (*VERBS, "turn", "left", "right", "opposite", "around", "twice", "thrice", "and", "after")
TOKENS: tuple[str, ...] = SPECIAL_TOKENS + WORDS
TOKEN_ID: Mapping[str, int] = MappingProxyType({name: index for index, name in enumerate(TOKENS)})
VOCAB_SIZE = len(TOKENS)
PAD, BOS, SEP, OUT = (TOKEN_ID[name] for name in SPECIAL_TOKENS)
ACTIONS: tuple[str, ...] = ("I_WALK", "I_LOOK", "I_RUN", "I_JUMP", "I_TURN_LEFT", "I_TURN_RIGHT")
END_CLASS = "END"
CLASS_NAMES: tuple[str, ...] = (*ACTIONS, END_CLASS)
CLASS_ID: Mapping[str, int] = MappingProxyType({name: index for index, name in enumerate(CLASS_NAMES)})
N_CLASSES = len(CLASS_NAMES)
END = CLASS_ID[END_CLASS]
VERB_ACTION: Mapping[str, str] = MappingProxyType({verb: f"I_{verb.upper()}" for verb in VERBS})
TURN_ACTION: Mapping[str, str] = MappingProxyType({"left": "I_TURN_LEFT", "right": "I_TURN_RIGHT"})
MAX_CMD = 9
"""Longest command of the grammar (and of every official file), in words."""
OUT_SLOTS = 48
"""L of PREREG E13: the longest action sequence in either condition's files (and in the whole grammar)."""
SEQ_LEN = 2 + MAX_CMD + OUT_SLOTS
IGNORE = -1

PART_NEW, PART_OLD, PART_PRIM = 1, 2, 3
PART_NAMES: Mapping[int, str] = MappingProxyType({
    PART_NEW: "new part: a command of the official test file (it composes the condition's primitive)",
    PART_OLD: "old part: a command of the official training file (for addprim_jump, not 'jump' alone)",
    PART_PRIM: "the primitive alone as it appears in training ('jump', addprim_jump only)",
})

E13_SPLIT_SEED = 100
"""PREREG E13: the splits are one seed-100 draw."""
SPLIT_STREAMS: Mapping[str, int] = MappingProxyType({"addprim_jump": 1301, "around_right": 1302})
N_OLD_HELD = 1000
OLD_HELD_ORDER: tuple[str, ...] = ("old_check", "old_select", "sealed_old")
NEW_ORDER: tuple[str, ...] = ("V_check", "V_select", "sealed_new")
PRIVATE_SPLITS: tuple[str, ...] = ("train", "old_check", "old_select", "V_check", "V_select")
"""Stored private splits of a condition: the student's training rows, old V_check and V_select (old commands held out
of training) and new V_check and V_select (thirds of the official test file). The sealed old and new parts go to
$WEIGHTCODE_SEALED only."""
CONSTRUCTIONS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "twice_thrice": ("twice", "thrice"), "around": ("around",), "opposite": ("opposite",), "turn": ("turn",),
    "and": ("and",), "after": ("after",),
})
"""The readout's breakdown by construction (PREREG E13): the rows whose command holds any of the words."""


class ScanError(ValueError):
    """A file, command, row or draw that the SCAN build refuses."""


# ---------------------------------------------------------------------------------------------------------------
# The interpretation function, as published
# ---------------------------------------------------------------------------------------------------------------

def _unit(words: Sequence[str]) -> list[str]:
    """[[V]] for V a primitive with its direction and modifier: U | U left | U opposite left | U around left | the
    same with turn; ``turn`` alone is not a command."""
    if not words:
        raise ScanError("an empty phrase is not a SCAN unit")
    if len(words) == 1:
        if words[0] not in VERBS:
            raise ScanError(f"{' '.join(words)!r} is not a primitive")
        return [VERB_ACTION[words[0]]]
    verb, direction = words[0], words[-1]
    if direction not in TURN_ACTION or (verb not in VERBS and verb != "turn"):
        raise ScanError(f"{' '.join(words)!r} is not a primitive with a direction")
    turn = TURN_ACTION[direction]
    act = [] if verb == "turn" else [VERB_ACTION[verb]]
    if len(words) == 2:
        return [turn, *act]
    if len(words) == 3 and words[1] == "opposite":
        return [turn, turn, *act]
    if len(words) == 3 and words[1] == "around":
        return [turn, *act] * 4
    raise ScanError(f"{' '.join(words)!r} is not a SCAN unit")


def _phrase(words: Sequence[str]) -> list[str]:
    """[[S]]: a unit, or a unit followed by twice or thrice."""
    if words and words[-1] in ("twice", "thrice"):
        return _unit(words[:-1]) * (2 if words[-1] == "twice" else 3)
    return _unit(words)


def interpret(command: str | Sequence[str]) -> list[str]:
    """The action sequence of a SCAN command (Lake and Baroni 2018, interpretation function).

    [[walk]] = I_WALK, and so for look, run and jump; [[turn left]] = I_TURN_LEFT; [[u left]] = I_TURN_LEFT [[u]];
    [[u opposite left]] = I_TURN_LEFT I_TURN_LEFT [[u]]; [[u around left]] = (I_TURN_LEFT [[u]]) x 4; the same with
    right; turn has no action of its own; [[x twice]] = [[x]] [[x]]; [[x thrice]] = [[x]] [[x]] [[x]];
    [[x1 and x2]] = [[x1]] [[x2]]; [[x1 after x2]] = [[x2]] [[x1]].
    """
    words = command.split() if isinstance(command, str) else list(command)
    unknown = sorted(set(words) - set(WORDS))
    if not words or unknown or len(words) > MAX_CMD:
        raise ScanError(f"{' '.join(words)!r} is not a SCAN command")
    joins = [k for k, word in enumerate(words) if word in ("and", "after")]
    if len(joins) > 1:
        raise ScanError(f"{' '.join(words)!r} has more than one conjunction")
    if not joins:
        return _phrase(words)
    k = joins[0]
    first, second = _phrase(words[:k]), _phrase(words[k + 1:])
    return first + second if words[k] == "and" else second + first


def grammar_commands() -> list[str]:
    """Every command of SCAN's phrase-structure grammar (20,910), sorted: C -> S | S and S | S after S;
    S -> V | V twice | V thrice; V -> D opposite D' | D around D' | D | U (D' the direction); D -> U left | U right |
    turn left | turn right; U -> walk | look | run | jump."""
    units = list(VERBS)
    for verb in (*VERBS, "turn"):
        for direction in ("left", "right"):
            units.append(f"{verb} {direction}")
            units.append(f"{verb} opposite {direction}")
            units.append(f"{verb} around {direction}")
    phrases = [f"{unit}{suffix}" for unit in units for suffix in ("", " twice", " thrice")]
    commands = list(phrases)
    for join in ("and", "after"):
        commands += [f"{a} {join} {b}" for a in phrases for b in phrases]
    return sorted(commands)


def has_primitive(command: str | Sequence[str], condition: str) -> bool:
    """True when the command holds the condition's primitive words as adjacent words."""
    words = command.split() if isinstance(command, str) else list(command)
    target = PRIMITIVE[condition]
    width = len(target)
    return any(tuple(words[k:k + width]) == target for k in range(len(words) - width + 1))


# ---------------------------------------------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_tasks(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """(command, actions) per line of a SCAN file ("IN: <command> OUT: <actions>"), in file order."""
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("IN: ") or " OUT: " not in line:
            raise ScanError(f"line {number} is not 'IN: ... OUT: ...': {line[:80]!r}")
        command, actions = line[len("IN: "):].split(" OUT: ")
        rows.append((" ".join(command.split()), tuple(actions.split())))
    return rows


def read_raw(raw_dir: Path, relative: str) -> list[tuple[str, tuple[str, ...]]]:
    """Rows of one raw file, refused unless its sha256 is RAW_SHA256's and every row's actions are ``interpret``'s."""
    path = Path(raw_dir) / relative
    if relative not in RAW_SHA256:
        raise ScanError(f"{relative} is not a recorded SCAN file")
    digest = file_sha256(path)
    if digest != RAW_SHA256[relative]:
        raise ScanError(f"{path}: sha256 {digest} != recorded {RAW_SHA256[relative]}")
    rows = parse_tasks(path.read_text(encoding="utf-8"))
    for command, actions in rows:
        if tuple(interpret(command)) != actions:
            raise ScanError(f"{relative}: {command!r} has actions that differ from the interpretation function")
    return rows


def command_ids(raw_dir: Path) -> dict[str, int]:
    """Command -> its id: its index in the sorted distinct commands of tasks.txt, which must be the grammar's."""
    commands = sorted({command for command, _ in read_raw(raw_dir, ALL_FILE)})
    if commands != grammar_commands():
        raise ScanError("tasks.txt does not hold exactly the commands of the grammar")
    return {command: index for index, command in enumerate(commands)}


# ---------------------------------------------------------------------------------------------------------------
# Token rows
# ---------------------------------------------------------------------------------------------------------------

def encode(commands: Sequence[str], actions: Sequence[Sequence[str]] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Token rows x [N, SEQ_LEN] and per-position targets y [N, SEQ_LEN] (IGNORE off the OUT positions).

    ``actions`` defaults to ``interpret`` of each command; given, each must equal it.
    """
    n = len(commands)
    x = np.full((n, SEQ_LEN), PAD, dtype=np.int64)
    y = np.full((n, SEQ_LEN), IGNORE, dtype=np.int64)
    for row, command in enumerate(commands):
        words = command.split()
        meaning = interpret(words)
        if actions is not None and tuple(actions[row]) != tuple(meaning):
            raise ScanError(f"{command!r}: given actions differ from the interpretation function")
        if len(meaning) > OUT_SLOTS:
            raise ScanError(f"{command!r} has {len(meaning)} actions, more than {OUT_SLOTS}")
        m = len(words)
        x[row, 0] = BOS
        x[row, 1:1 + m] = [TOKEN_ID[word] for word in words]
        x[row, 1 + m] = SEP
        x[row, 2 + m:2 + m + OUT_SLOTS] = OUT
        classes = [CLASS_ID[action] for action in meaning] + [END] * (OUT_SLOTS - len(meaning))
        y[row, 2 + m:2 + m + OUT_SLOTS] = classes
    return x, y


def decode(x: np.ndarray) -> list[str]:
    """The commands of well-formed token rows; raises on any row ``encode`` would not write."""
    x = np.asarray(x, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != SEQ_LEN:
        raise ScanError(f"token rows must be [N, {SEQ_LEN}], got {x.shape}")
    commands = []
    for row in x:
        sep = np.flatnonzero(row == SEP)
        if row[0] != BOS or sep.size != 1 or not 2 <= sep[0] <= 1 + MAX_CMD:
            raise ScanError("every row needs BOS first and exactly one SEP after 1 to 9 words")
        words = [TOKENS[int(t)] for t in row[1:sep[0]]]
        if any(word not in WORDS for word in words):
            raise ScanError(f"row holds a non-word token before SEP: {words}")
        command = " ".join(words)
        if not np.array_equal(encode([command])[0][0], row):
            raise ScanError(f"row of {command!r} is not well formed")
        commands.append(command)
    return commands


def actions_of(y: np.ndarray) -> list[list[str]]:
    """The action sequences encoded in targets y [N, SEQ_LEN] (the classes before the first END)."""
    out = []
    for row in np.asarray(y):
        labels = row[row != IGNORE]
        if labels.size != OUT_SLOTS:
            raise ScanError(f"a row has {labels.size} labelled positions, not {OUT_SLOTS}")
        stop = np.flatnonzero(labels == END)
        length = int(stop[0]) if stop.size else OUT_SLOTS
        if np.any(labels[length:] != END):
            raise ScanError("an action follows END")
        out.append([CLASS_NAMES[int(c)] for c in labels[:length]])
    return out


def row_correct(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Exact match per row: every labelled (OUT) position predicted right."""
    y, pred = np.asarray(y), np.asarray(pred)
    if y.shape != pred.shape or y.ndim != 2:
        raise ScanError(f"y {y.shape} and pred {pred.shape} must be equal [N, L]")
    labelled = y != IGNORE
    if np.any(~labelled.any(axis=1)):
        raise ScanError("a row has no labelled position")
    return np.all((pred == y) | ~labelled, axis=1)


def token_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Share of labelled (OUT) positions predicted right."""
    y, pred = np.asarray(y), np.asarray(pred)
    labelled = y != IGNORE
    return float(np.mean(pred[labelled] == y[labelled]))


def split_arrays(commands: Sequence[str], ids: Mapping[str, int], part: int | Sequence[int]) -> dict[str, np.ndarray]:
    """A split's arrays: x, y [N, SEQ_LEN]; idx (command id), part, n_cmd (words), n_act (actions) [N]."""
    x, y = encode(commands)
    parts = np.full(len(commands), part, dtype=np.int64) if np.isscalar(part) else np.asarray(part, dtype=np.int64)
    return {
        "x": x,
        "y": y,
        "idx": np.array([ids[c] for c in commands], dtype=np.int64),
        "part": parts,
        "n_cmd": np.array([len(c.split()) for c in commands], dtype=np.int64),
        "n_act": np.array([len(interpret(c)) for c in commands], dtype=np.int64),
    }


def _sorted_by_id(commands: Iterable[str], ids: Mapping[str, int]) -> list[str]:
    return sorted(commands, key=lambda command: ids[command])


@dataclass
class ConditionSplits:
    """Every array of one condition: the private splits and the two sealed parts, plus the draw's record."""

    private: dict[str, dict[str, np.ndarray]]
    sealed: dict[str, np.ndarray]
    record: dict[str, Any]


def build_condition_splits(raw_dir: Path, condition: str) -> ConditionSplits:
    """The splits of one condition (module docstring), a pure function of the raw files and the code."""
    if condition not in CONDITIONS:
        raise ScanError(f"{condition!r} is not an E13 condition; one of {CONDITIONS}")
    ids = command_ids(raw_dir)
    train_rows = read_raw(raw_dir, CONDITION_FILES[condition]["train"])
    test_rows = read_raw(raw_dir, CONDITION_FILES[condition]["test"])
    primitive_row = PRIMITIVE_ROW[condition]
    train_commands = {command for command, _ in train_rows}
    test_commands = {command for command, _ in test_rows}
    if train_commands & test_commands:
        raise ScanError(f"{condition}: the official train and test files share commands")
    if not all(has_primitive(c, condition) for c in test_commands):
        raise ScanError(f"{condition}: a test command lacks the primitive {PRIMITIVE[condition]}")
    with_primitive = sorted(c for c in train_commands if has_primitive(c, condition))
    if with_primitive != ([primitive_row] if primitive_row else []):
        raise ScanError(f"{condition}: training commands with the primitive are {with_primitive[:5]}")
    old = _sorted_by_id(train_commands - ({primitive_row} if primitive_row else set()), ids)
    new = _sorted_by_id(test_commands, ids)
    if len(test_rows) != len(new):
        raise ScanError(f"{condition}: the test file repeats a command")

    rng = np.random.default_rng([E13_SPLIT_SEED, SPLIT_STREAMS[condition]])
    old_perm = rng.permutation(len(old))
    held = {name: _sorted_by_id([old[i] for i in old_perm[k * N_OLD_HELD:(k + 1) * N_OLD_HELD]], ids)
            for k, name in enumerate(OLD_HELD_ORDER)}
    new_perm = rng.permutation(len(new))
    thirds = {name: _sorted_by_id([new[i] for i in chunk], ids)
              for name, chunk in zip(NEW_ORDER, np.array_split(new_perm, len(NEW_ORDER)))}
    held_out = set().union(*held.values())
    kept = [(command, actions) for command, actions in train_rows if command not in held_out]
    train_parts = [PART_PRIM if command == primitive_row else PART_OLD for command, _ in kept]
    train = split_arrays([command for command, _ in kept], ids, train_parts)
    private = {
        "train": train,
        "old_check": split_arrays(held["old_check"], ids, PART_OLD),
        "old_select": split_arrays(held["old_select"], ids, PART_OLD),
        "V_check": split_arrays(thirds["V_check"], ids, PART_NEW),
        "V_select": split_arrays(thirds["V_select"], ids, PART_NEW),
    }
    sealed_new = split_arrays(thirds["sealed_new"], ids, PART_NEW)
    sealed_old = split_arrays(held["sealed_old"], ids, PART_OLD)
    sealed = {key: np.concatenate([sealed_new[key], sealed_old[key]]) for key in sealed_new}
    record = {
        "n_train_rows_official": len(train_rows),
        "n_train_rows": int(len(kept)),
        "n_train_distinct_commands": len({command for command, _ in kept}),
        "n_primitive_rows_in_train": int(sum(1 for part in train_parts if part == PART_PRIM)),
        "n_old_commands": len(old),
        "n_new_commands": len(new),
        "held_out_ids_sha256": hashlib.sha256(
            np.array(sorted(ids[c] for c in held_out), dtype=np.int64).tobytes()).hexdigest(),
        "sealed_sizes": {"new": len(thirds["sealed_new"]), "old": len(held["sealed_old"])},
    }
    return ConditionSplits(private=private, sealed=sealed, record=record)


def check_disjoint(splits: ConditionSplits) -> None:
    """No command is in two of the held-out parts, and no held-out command is in the training rows."""
    sets = {name: set(arrays["idx"].tolist()) for name, arrays in splits.private.items()}
    sealed_ids = set(splits.sealed["idx"].tolist())
    held = [sets["old_check"], sets["old_select"], sets["V_check"], sets["V_select"], sealed_ids]
    total = sum(len(s) for s in held)
    if len(set().union(*held)) != total:
        raise ScanError("held-out parts share a command")
    if sets["train"] & set().union(*held):
        raise ScanError("a held-out command is in the training rows")


# ---------------------------------------------------------------------------------------------------------------
# The public statement of the grammar (model_spec.json token meanings)
# ---------------------------------------------------------------------------------------------------------------

TOKEN_MEANINGS: Mapping[str, str] = MappingProxyType({
    "PAD": "padding after the last OUT slot; attention ignores it as a key",
    "BOS": "first token of every row",
    "SEP": "ends the command",
    "OUT": f"one of the {OUT_SLOTS} output slots after SEP; the prediction at the k-th OUT slot is the k-th action of "
    "the command's meaning, or END after the last action",
    "walk": "primitive: [[walk]] = I_WALK",
    "look": "primitive: [[look]] = I_LOOK",
    "run": "primitive: [[run]] = I_RUN",
    "jump": "primitive: [[jump]] = I_JUMP",
    "turn": "turn with a direction and no other action: [[turn left]] = I_TURN_LEFT, [[turn right]] = I_TURN_RIGHT; "
    "[[turn opposite left]] = I_TURN_LEFT I_TURN_LEFT; [[turn around left]] = I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT "
    "I_TURN_LEFT; the same with right",
    "left": "direction: [[u left]] = I_TURN_LEFT [[u]] for a primitive u",
    "right": "direction: [[u right]] = I_TURN_RIGHT [[u]] for a primitive u",
    "opposite": "[[u opposite left]] = I_TURN_LEFT I_TURN_LEFT [[u]]; [[u opposite right]] = I_TURN_RIGHT I_TURN_RIGHT "
    "[[u]]",
    "around": "[[u around left]] = I_TURN_LEFT [[u]] I_TURN_LEFT [[u]] I_TURN_LEFT [[u]] I_TURN_LEFT [[u]]; "
    "[[u around right]] = I_TURN_RIGHT [[u]] I_TURN_RIGHT [[u]] I_TURN_RIGHT [[u]] I_TURN_RIGHT [[u]]",
    "twice": "[[x twice]] = [[x]] [[x]]",
    "thrice": "[[x thrice]] = [[x]] [[x]] [[x]]",
    "and": "[[x1 and x2]] = [[x1]] [[x2]]",
    "after": "[[x1 after x2]] = [[x2]] [[x1]]",
})
"""Every token's meaning: SCAN's interpretation function as published, one rule per word."""
GRAMMAR = (
    "commands: C -> S | S and S | S after S; S -> V | V twice | V thrice; V -> D opposite left | D opposite right | "
    "D around left | D around right | D left | D right | U; D -> U | turn; U -> walk | look | run | jump (turn "
    "appears only with a direction). [[.]] is a command's meaning, a sequence of actions; meanings concatenate"
)
INPUT_LAYOUT = (
    f"[BOS, command words, SEP, OUT x {OUT_SLOTS}], right-padded with PAD to seq_len; a command has 1 to {MAX_CMD} "
    f"words of the grammar below; the answer at the k-th OUT slot is the k-th action of the command's meaning, END "
    f"at every slot after the last action. Grammar: {GRAMMAR}"
)


def token_table() -> list[dict[str, Any]]:
    """id, name and meaning of every token, for a student's public model_spec.json."""
    return [{"id": index, "name": name, "meaning": TOKEN_MEANINGS[name]} for index, name in enumerate(TOKENS)]


def format_manifest() -> dict[str, Any]:
    """The row format recorded in dataset.json."""
    return {
        "tokens": list(TOKENS),
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "max_command_words": MAX_CMD,
        "out_slots": OUT_SLOTS,
        "classes": list(CLASS_NAMES),
        "row": INPUT_LAYOUT,
        "targets": "y[k-th OUT position] = class index of the k-th action, END after the last action, IGNORE (-1) "
        "elsewhere; a row is right when every OUT position is right",
        "command_id": "index in the sorted distinct commands of tasks.txt (the 20,910 commands of the grammar)",
        "interpretation": dict(TOKEN_MEANINGS),
    }


__all__ = [
    "ACTIONS", "ALL_FILE", "BOS", "CLASS_ID", "CLASS_NAMES", "CONDITIONS", "CONDITION_FILES", "CONSTRUCTIONS",
    "ConditionSplits", "DATASET", "END", "E13_SPLIT_SEED", "GRAMMAR", "IGNORE", "INPUT_LAYOUT", "MAX_CMD",
    "N_CLASSES", "N_OLD_HELD", "OUT", "OUT_SLOTS", "PAD", "PART_NAMES", "PART_NEW", "PART_OLD", "PART_PRIM",
    "PRIMITIVE", "PRIMITIVE_ROW", "PRIVATE_SPLITS", "RAW_SHA256", "SCAN_COMMIT", "SCAN_URL", "SEP", "SEQ_LEN",
    "SPLIT_STREAMS", "ScanError", "TOKENS", "TOKEN_ID", "TOKEN_MEANINGS", "VOCAB_SIZE", "WORDS", "actions_of",
    "build_condition_splits", "check_disjoint", "command_ids", "decode", "encode", "file_sha256", "format_manifest",
    "grammar_commands", "has_primitive", "interpret", "parse_tasks", "read_raw", "row_correct", "split_arrays",
    "token_accuracy", "token_table",
]
