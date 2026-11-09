"""Forbidden strings and the pre-launch leak scan (HARNESS_SPEC sections 5, 6 and 10).

The editor may learn about its instance only from its files. Names that identify the arm, the instance, the
models, the dataset or the condition must therefore appear neither in editor-visible text (trial files, prompts,
tool descriptions) nor in what the sandbox can read about its own mounts (/proc/self/mountinfo shows the host path
of every bind source), its environment or its trial listing.

One documented exemption: the generic condition ``deploy`` (``weightcode.evalkit.targets.DEPLOY_CONDITION``, "the
dataset's own deployment data") is the stem of the words "deployment" and "deployed" that the owner-approved task
wording and task.json must use, and names no kind of shift. It is still a forbidden string for the sandbox
self-test, but not for the text scan. Every other condition name is scanned in full.

Label arrays (M2NL, an arm without labels): ``scan_label_arrays`` finds every label array or label key in a trial
directory. A label array is named ``y`` or ``y_<name>`` (V_check and the splits use ``y``, evidence ``y_fail`` and
``y_rand``): such a member of an .npz, such an .npy or any file of that name is a hit, and so is a JSON object key
or string value ``y_<name>`` (a bare ``"y"`` in JSON is not, because vocabularies and format text may hold it).

A graphnew-v1 session (PREREG E7) scans its texts with ``weightcode.evalkit.graphnew.graphnew_scan`` instead of
``scan_for_tokens`` (``scanner``): the same matching, except that "valid" and the years match as whole words only.

Library programs of a loop round (PREREG E10). ``ForbiddenStrings.library_text``, when set, is the scan of the round
library's program files (``library_program_<k>.py``, written by ``weightcode.harness.loop.write_loop_library``) in place of
``text``: an E10 session scans them with E4-loop's standard (the base list: condition names with an underscore,
instance, model and session ids, host paths, the dataset's tokens) and every other trial file, the prompts and the
tool descriptions with the full E10 list. A library program is the teacher's own earlier program; its comments may
describe an earlier round's images in ordinary words ("contrast", "noise"), which name no condition of this round.

Exempt sentences (K0a and K0b of the PREREG information ladder, ``weightcode.harness.e11``). ``ForbiddenStrings.
sentence_exempt`` holds whole sentences that name the condition by design (the capability sentence, which is the
information of that rung). ``strip_exempt`` removes every exact occurrence of them from a text before the token scan
(``scan_texts``, ``scan_trial``) and the hint-word scan (``scan_hint_words``), so their words are exempt there and
nowhere else: the same word in any other place is still a hit. ``exempt_counts`` says how often each occurs per text.
Without exempt sentences every scan and record is as before.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.evalkit.instances import leak_tokens, scan_for_tokens
from weightcode.evalkit.targets import DEPLOY_CONDITION

ARM_WORDS = ("noweight", "decoy", "rewrite")
# The prompts never mention colour, noise, contrast or a shift (HARNESS_SPEC section 6); scanned in prompts and tool
# descriptions only, because format text such as model_spec.json may name input channels legitimately.
HINT_WORDS = ("color", "colour", "noise", "contrast", "corrupt", "shift")
TEXT_SUFFIXES = (".json", ".py", ".md", ".txt", ".csv")
LABEL_ARRAY_RE = re.compile(r"y(_\w+)?")
LABEL_KEY_RE = re.compile(r"y_\w+")
Scanner = Callable[[str, Sequence[str]], list[str]]  # (text, tokens) -> the tokens found, as ``scan_for_tokens``
LIBRARY_PROGRAM_PATTERN = r"library_program_[1-9][0-9]*\.py"
LIBRARY_PROGRAM_RE = re.compile(LIBRARY_PROGRAM_PATTERN)
"""The names of a round library's program files (``weightcode.harness.loop.LOOP_PROGRAM_TEMPLATE``)."""
SENTENCE_EXEMPT_REASON = ("the capability sentence names the condition, which is the information of this rung "
                          "(PREREG information ladder, K0a and K0b); every other text is scanned in full")


def strip_exempt(text: str, sentences: Sequence[str] = ()) -> str:
    """``text`` with every exact occurrence of each of ``sentences`` replaced by a space (module docstring)."""
    for sentence in sentences:
        if sentence:
            text = text.replace(sentence, " ")
    return text


def exempt_counts(texts: Mapping[str, str], sentences: Sequence[str]) -> dict[str, dict[str, int]]:
    """``{text name: {sentence: occurrences}}`` for every text and exempt sentence."""
    return {name: {sentence: text.count(sentence) for sentence in sentences} for name, text in texts.items()}


@dataclass(frozen=True)
class ForbiddenStrings:
    """The forbidden strings of one instance: for the sandbox self-test and for the text scan."""

    selftest: tuple[str, ...]
    text: tuple[str, ...]
    text_exempt: tuple[str, ...]
    library_text: tuple[str, ...] | None = None
    """The scan of the round library's program files where it differs from ``text`` (module docstring), else None."""
    sentence_exempt: tuple[str, ...] = ()
    """Whole sentences removed from every text before the text and hint-word scans (module docstring)."""

    @property
    def library_scan(self) -> tuple[str, ...]:
        """The tokens a round library's program files are scanned for (and withheld on)."""
        return self.text if self.library_text is None else self.library_text

    def as_dict(self) -> dict[str, Any]:
        record = {"selftest": list(self.selftest), "text": list(self.text), "text_exempt": list(self.text_exempt),
                  "text_exempt_reason": None if not self.text_exempt else (
                      f"the generic condition {DEPLOY_CONDITION!r} is the stem of the required wording 'deployment'")}
        if self.library_text is not None:
            record["library_text"] = list(self.library_text)
            record["library_text_files"] = LIBRARY_PROGRAM_PATTERN
        if self.sentence_exempt:
            record["sentence_exempt"] = list(self.sentence_exempt)
            record["sentence_exempt_reason"] = SENTENCE_EXEMPT_REASON
        return record


def base_tokens(private: Mapping[str, Any], instance_id: str) -> list[str]:
    """Arm words, the instance id, the real and decoy model ids, the dataset name and its leak tokens, the condition.

    A fewshot-v1 instance (PREREG E2) has no decoy and no condition; their tokens are then left out.
    """
    dataset = str(private["dataset"])
    optional = [str(private[name]) for name in ("decoy_model_id", "condition") if private.get(name)]
    tokens = {*ARM_WORDS, instance_id, str(private["model_id"]), dataset, *optional, *leak_tokens(dataset)}
    return sorted(token for token in tokens if token)


def case_variants(tokens: Iterable[str]) -> list[str]:
    """Each token as given, lower case, upper case and capitalised (the self-test matches case-sensitively)."""
    variants = set()
    for token in tokens:
        variants.update({token, token.lower(), token.upper(), token.capitalize()})
    return sorted(variants)


def forbidden_strings(
    private: Mapping[str, Any],
    instance_id: str,
    *,
    paths: Iterable[str] = (),
    conditions: Iterable[str] = (),
) -> ForbiddenStrings:
    """Forbidden strings for one instance (see the module docstring for the one text-scan exemption).

    ``paths`` (host paths such as the V_check file, private.json and the sealed root) are added as given, without
    case variants; ``conditions`` (every known condition name) are added like the instance's own condition.
    """
    condition = str(private.get("condition") or "")
    tokens = sorted({*base_tokens(private, instance_id), *(name for name in conditions if name)})
    exempt = (condition,) if condition == DEPLOY_CONDITION else ()
    path_tokens = sorted({path for path in paths if path})
    text = tuple(sorted({*(token for token in tokens if token not in exempt), *path_tokens}))
    selftest = tuple(sorted({*case_variants(tokens), *path_tokens}))
    return ForbiddenStrings(selftest=selftest, text=text, text_exempt=exempt)


def scan_hint_words(texts: Mapping[str, str], sentence_exempt: Sequence[str] = ()) -> dict[str, list[str]]:
    """Words that would hint at the kind of shift or at a fix (HARNESS_SPEC section 6), found in editor text outside
    the exempt sentences."""
    return scan_texts(texts, HINT_WORDS, sentence_exempt=sentence_exempt)


def npz_text(data: bytes) -> str:
    """The scannable text of an .npz file: member names and the contents of string arrays."""
    parts = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        parts.extend(archive.namelist())
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        for key in arrays.files:
            value = arrays[key]
            if value.dtype.kind in "US":
                parts.extend(str(item) for item in value.ravel().tolist())
    return "\n".join(parts)


def file_text(path: Path) -> str:
    """Text of one editor-visible file for scanning; a binary file that is neither .npz nor text is an error."""
    data = path.read_bytes()
    if path.suffix == ".npz":
        return npz_text(data)
    if path.suffix == ".npy":
        array = np.load(io.BytesIO(data), allow_pickle=False)
        return "\n".join(str(item) for item in array.ravel().tolist()) if array.dtype.kind in "US" else ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name}: not UTF-8 text and not .npz/.npy, so it cannot be leak-scanned") from exc


def scan_texts(texts: Mapping[str, str], tokens: Iterable[str],
               scanner: Scanner = scan_for_tokens, *, sentence_exempt: Sequence[str] = ()) -> dict[str, list[str]]:
    """``{name: tokens found}`` for every text with at least one hit (``scanner``, by default ``scan_for_tokens``,
    case-insensitive), outside the exempt sentences (module docstring)."""
    tokens = list(tokens)
    hits = {}
    for name, text in texts.items():
        found = scanner(strip_exempt(text, sentence_exempt), tokens)
        if found:
            hits[name] = found
    return hits


def _json_label_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        keys = [str(key) for key in value if LABEL_KEY_RE.fullmatch(str(key))]
        return keys + [hit for item in value.values() for hit in _json_label_keys(item)]
    if isinstance(value, list):
        return [hit for item in value for hit in _json_label_keys(item)]
    return [value] if isinstance(value, str) and LABEL_KEY_RE.fullmatch(value) else []


def scan_label_arrays(trial_dir: Path) -> dict[str, list[str]]:
    """``{trial/<file>: label arrays or keys}`` for every file of ``trial_dir`` that holds one (module docstring)."""
    hits: dict[str, list[str]] = {}
    for path in sorted(trial_dir.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"trial entry {path.name} is not a regular file")
        found = [path.name] if LABEL_ARRAY_RE.fullmatch(path.name.split(".", 1)[0]) else []
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                found += [key for key in archive.files if LABEL_ARRAY_RE.fullmatch(key)]
        elif path.suffix == ".json":
            found += _json_label_keys(json.loads(path.read_text(encoding="utf-8")))
        if found:
            hits[f"trial/{path.name}"] = sorted(set(found))
    return hits


def trial_texts(trial_dir: Path) -> dict[str, str]:
    """``{trial/<file>: file name and scannable text}`` for every file of the trial directory (``file_text``)."""
    texts: dict[str, str] = {}
    for path in sorted(trial_dir.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"trial entry {path.name} is not a regular file")
        texts[f"trial/{path.name}"] = path.name + "\n" + file_text(path)
    return texts


def scan_trial(trial_dir: Path, tokens: Iterable[str], scanner: Scanner = scan_for_tokens, *,
               library_tokens: Iterable[str] | None = None,
               sentence_exempt: Sequence[str] = ()) -> dict[str, list[str]]:
    """Leak scan of every file in the trial directory, file names included, outside the exempt sentences.
    ``library_tokens``, when given, replace ``tokens`` for the round library's program files (``LIBRARY_PROGRAM_RE``;
    module docstring)."""
    texts: dict[str, str] = {}
    library: dict[str, str] = {}
    for path in sorted(trial_dir.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"trial entry {path.name} is not a regular file")
        own = library if library_tokens is not None and LIBRARY_PROGRAM_RE.fullmatch(path.name) else texts
        own[f"trial/{path.name}"] = path.name + "\n" + file_text(path)
    hits = scan_texts(texts, tokens, scanner, sentence_exempt=sentence_exempt)
    if library:
        hits.update(scan_texts(library, library_tokens or (), scanner, sentence_exempt=sentence_exempt))
    return hits
