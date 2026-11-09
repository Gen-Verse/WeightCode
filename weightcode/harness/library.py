"""The program library of a program-arm session (LEARN v3, dev only; PREREG section 0, "Dev iteration, LEARN v3").

A library is the kept program of each of a few earlier LEARN sessions on other conditions of the same student, with
the V_check scores that its own session's kept check returned. A v3 session's trial holds it as flat files (the trial
leak scan reads every top-level file): ``library_index.json`` and ``library_program_<k>.py``, k = 1, 2, ..., in the
order the sessions are given. The index labels each program "earlier condition k" and gives its score_new, score_old
and score with the row counts of the two check sets; it holds no condition name, session id, instance id or V_select
number. The program files are the earlier finals byte for byte.

``library_entries`` reads and checks the sources before anything of the new session is written: each is a complete,
non-dry-run session of a program arm on an evidence-v1 instance of the same student (model id) with the same clean
V_check rows (clean_sha256) as the target, on another instance than the target and than every other entry, whose final
is a checked program whose file hashes to the program_sha256 of its check. ``write_library`` writes the files into a
trial directory and returns the session.json ``library`` record.

An fscil-v1 session (PREREG E9) has its own library, the lineage's kept programs of earlier sessions labelled by
session (``weightcode.harness.fscil.library_entries`` and ``library_index``); ``write_library`` writes it with that index, and
an entry whose program is withheld (``file`` None) gets no program file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LIBRARY_SCHEMA = "weightcode.harness.library.v1"
LIBRARY_INDEX_FILE = "library_index.json"
LIBRARY_PROGRAM_TEMPLATE = "library_program_{k}.py"
LIBRARY_LABEL_TEMPLATE = "earlier condition {k}"
MAX_LIBRARY_PROGRAMS = 8
PROGRAM_ARMS = ("LEARN",)


class LibraryError(ValueError):
    """A library source that is not a usable earlier session for this target."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LibraryError(f"{path} is missing") from None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _kept_check(root: Path, query_index: int) -> dict[str, Any]:
    checks = [json.loads(line) for line in (root / "logs" / "checks.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    matches = [check for check in checks if int(check["query_index"]) == query_index]
    if len(matches) != 1:
        raise LibraryError(f"{root.name}: {len(matches)} checks with query index {query_index}")
    return matches[0]


def library_entries(session_ids: Sequence[str], sessions_dir: Path, *, target_instance_id: str,
                    target_model_id: str, target_clean_sha256: str) -> list[dict[str, Any]]:
    """The checked library entries for a target session, in the given order (see the module docstring)."""
    if not session_ids:
        raise LibraryError("a library needs at least one earlier session")
    if len(session_ids) > MAX_LIBRARY_PROGRAMS:
        raise LibraryError(f"a library holds at most {MAX_LIBRARY_PROGRAMS} programs, not {len(session_ids)}")
    if len(set(session_ids)) != len(session_ids):
        raise LibraryError(f"a library session is listed twice: {list(session_ids)}")
    entries: list[dict[str, Any]] = []
    seen_instances: set[str] = set()
    for k, session_id in enumerate(session_ids, start=1):
        root = Path(sessions_dir) / session_id
        session = _read_json(root / "session.json")
        result = _read_json(root / "session_result.json")
        selection = _read_json(root / "final" / "selection.json")
        instance_id = str(session["instance_id"])
        if session.get("arm") not in PROGRAM_ARMS:
            raise LibraryError(f"{session_id}: arm {session.get('arm')!r} writes no update program")
        if session.get("dry_run"):
            raise LibraryError(f"{session_id} is a dry run")
        if result.get("status") != "complete":
            raise LibraryError(f"{session_id} did not complete (status {result.get('status')!r})")
        own = [name for name in ("fewshot", "seqskill", "graphnew", "fscil", "policyskill") if name in session]
        if own:
            rule = f"{own[0]}-v1"
            raise LibraryError(f"{session_id} ran on a {rule} instance; a library holds evidence-v1 programs")
        if instance_id == target_instance_id:
            raise LibraryError(f"{session_id} ran on the target instance {target_instance_id}")
        if instance_id in seen_instances:
            raise LibraryError(f"two library sessions ran on instance {instance_id}")
        if str(session.get("model_id")) != target_model_id:
            raise LibraryError(f"{session_id} ran on student {session.get('model_id')}, not {target_model_id}")
        clean_sha = (session.get("clean_check") or {}).get("clean_sha256")
        if clean_sha != target_clean_sha256:
            raise LibraryError(f"{session_id}: its clean V_check rows ({clean_sha}) are not the target's")
        final = selection["final"]
        if final.get("kind") != "checked":
            raise LibraryError(f"{session_id} kept the base model; it has no program to add")
        query_index = int(final["query_index"])
        check = _kept_check(root, query_index)
        program_path = root / "final" / str(final["patch_id"])
        if not program_path.is_file():
            raise LibraryError(f"{session_id}: the kept program {program_path} is missing")
        source = program_path.read_bytes()
        if _sha256(source) != check.get("program_sha256"):
            raise LibraryError(f"{session_id}: {program_path.name} does not hash to its check's program_sha256")
        scores = {key: int(final[key]) for key in ("score_new", "score_old", "score")}
        if any(int(check[key]) != scores[key] for key in scores):
            raise LibraryError(f"{session_id}: the selection's scores differ from its check's")
        seen_instances.add(instance_id)
        entries.append({
            "k": k,
            "label": LIBRARY_LABEL_TEMPLATE.format(k=k),
            "file": LIBRARY_PROGRAM_TEMPLATE.format(k=k),
            **scores,
            "n_check_new": int(check["n_new"]),
            "n_check_clean": int(check["n_old"]),
            "source": {"session_id": session_id, "instance_id": instance_id, "condition": session.get("condition"),
                       "query_index": query_index, "program_file": program_path.name,
                       "program_sha256": check["program_sha256"], "program_path": str(program_path)},
            "_bytes": source,
        })
    return entries


def index_document(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The trial's ``library_index.json``: labels, files and check scores only."""
    return {
        "schema": LIBRARY_SCHEMA,
        "about": ("Update programs written in earlier sessions for other input conditions on this same classifier, "
                  "none of them for this task's condition. For each: the scores its own kept patch_check returned "
                  "there, score_new on that condition's held-out check set of n_check_new images and score_old on "
                  "the held-out clean check set of n_check_clean images, which is the same clean check set as this "
                  "task's."),
        "programs": [{key: entry[key] for key in ("label", "file", "score_new", "score_old", "score", "n_check_new",
                                                  "n_check_clean")} for entry in entries],
    }


def write_library(trial_dir: Path, entries: Sequence[Mapping[str, Any]],
                  index: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write the library files into ``trial_dir``; return the session.json ``library`` record. ``index`` replaces
    ``index_document(entries)`` (an fscil-v1 library); an entry whose ``file`` is None writes no program file."""
    trial_dir = Path(trial_dir)
    files: dict[str, str] = {}
    for entry in entries:
        if entry["file"] is None:
            continue
        path = trial_dir / str(entry["file"])
        if path.exists():
            raise LibraryError(f"{path} already exists")
        path.write_bytes(entry["_bytes"])
        files[path.name] = _sha256(path.read_bytes())
    index = index_document(entries) if index is None else index
    index_path = trial_dir / LIBRARY_INDEX_FILE
    if index_path.exists():
        raise LibraryError(f"{index_path} already exists")
    index_path.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    files[LIBRARY_INDEX_FILE] = _sha256(index_path.read_bytes())
    return {
        "schema": LIBRARY_SCHEMA,
        "files": files,
        "entries": [{key: value for key, value in entry.items() if key != "_bytes"} for entry in entries],
    }


__all__ = ["LIBRARY_INDEX_FILE", "LIBRARY_LABEL_TEMPLATE", "LIBRARY_PROGRAM_TEMPLATE", "LIBRARY_SCHEMA",
           "LibraryError", "MAX_LIBRARY_PROGRAMS", "index_document", "library_entries", "write_library"]
