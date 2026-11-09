"""T00: one regenerable table of every LLM editor session, failures included, with a cost table.

Execution Plan v1.1, sections 4 and 21. Parsing only: no model run, no LLM call.

Reads (read-only):
- <runs>/sessions/<sid>/: session.json, session_result.json, run.json, audit*.json, final/selection.json,
  final/patch.json, final/program_run.json, final/model.npz (array headers only), trial/model.npz (headers only)
  or trial/model_shapes.json, logs/transcript.jsonl, logs/cells.jsonl(.inflight), logs/checks.jsonl;
- <runs>/manifest.jsonl, <runs>/devreadouts/ (learn/<sid>/info.json and top-level *_<sid>*.jsonl),
  <runs>/loop_streams/loop_teacher_*.json, <runs>/fscil_lineages/*/state.json;
- <private>/instances/index.json;
- docs/DEV_RESULTS.md and the one hand-made mapping scripts/t00/session_readings.json.

Writes <out>/session_table.jsonl (one row per session) and <out>/session_table.md (session table, cost table,
per-experiment summary, problems found, undeterminable fields). Default <out> is <runs>/t00.

A field that is not recorded is null, and the row's "missing" dict gives the reason. A field that does not apply to
the session (a patch arm has no program, a single-round session has no round) is null with its reason in the row's
"not_applicable" dict. No value is invented.
Dollar amounts are API-equivalent at list price, not spend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import zipfile
from datetime import datetime
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from numpy.lib import format as npy_format

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "weightcode.t00.session_table.v1"

# List prices in US$ per million tokens. Source: the claude-api skill bundled with Claude Code 2.1.318,
# shared/model-migration.md ("$4 / $20 per MTok; 5-minute cache writes $5 and 1-hour cache writes $8; cache reads
# $0.20 per MTok") and shared/prompt-caching.md (writes 1.25x input for 5 min, 2x for 1 h).
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write_5m": 5.0,
                        "cache_write_1h": 8.0},
}
PRICE_SOURCE = ("list price, claude-api skill (Claude Code 2.1.318): input $4, output $20, cache read $0.20, "
                "cache write 5 min $5 (1.25x), 1 h $8 (2x) per MTok")
CLI_MATCH_TOL_USD = 1e-6
AUDIT_FILES = ("audit.json", "audit.rerun1.json", "audit.rerun2.json")
NOT_A_RECORD_AUDIT = "audit.wip_auditor_not_a_record.json"
MEMORY_KEY_HINTS = ("peak", "rss", "max_memory", "maxrss")
# Top-level devreadout files that hold a session's own final readout. b1_wave1_<sid>.jsonl files are baseline runs
# keyed to a session (their run paths are under runs/baselines/), not the session's readout, so they are not read.
SESSION_READOUT_PATTERNS = ("wave1_{sid}.jsonl", "m2_pilot_{sid}.jsonl", "e8_session_{sid}_*.jsonl")


class T00Error(RuntimeError):
    """A record that the table cannot be built from without guessing."""


# ----------------------------------------------------------------------------------------------- small helpers

def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _maybe_json(path: Path) -> Any | None:
    return _load_json(path) if path.is_file() else None


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _count_lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


class Row:
    """One session row: values plus, for every null field, a reason in one of two dicts.

    "missing": the quantity exists for this session but is not recorded. "not_applicable": the session has no
    such quantity (a patch arm has no program run, a single-round session has no round number, a completed
    session has no failure reason).
    """

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.missing: dict[str, str] = {}
        self.not_applicable: dict[str, str] = {}

    def set(self, key: str, value: Any, reason_if_none: str | None = None, na: bool = False) -> None:
        self.values[key] = value
        self.missing.pop(key, None)
        self.not_applicable.pop(key, None)
        if value is None:
            if reason_if_none is None:
                raise T00Error(f"field {key} is null and no reason was given")
            (self.not_applicable if na else self.missing)[key] = reason_if_none

    def as_dict(self) -> dict:
        out = dict(self.values)
        out["missing"] = dict(sorted(self.missing.items()))
        out["not_applicable"] = dict(sorted(self.not_applicable.items()))
        return out


# ----------------------------------------------------------------------------------------------- tokens and cost

TOKEN_KEYS = ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h")


def tokens_from_model_usage(model_usage: dict) -> dict[str, dict[str, int]]:
    """Per-model token counts from the CLI result event's modelUsage block."""
    out: dict[str, dict[str, int]] = {}
    for model, u in model_usage.items():
        w5 = u.get("cacheCreationEphemeral5mInputTokens")
        w1 = u.get("cacheCreationEphemeral1hInputTokens")
        total_w = u.get("cacheCreationInputTokens")
        if w5 is None or w1 is None:
            raise T00Error(f"modelUsage of {model} lacks the 5 min / 1 h cache-write split")
        if total_w is not None and w5 + w1 != total_w:
            raise T00Error(f"modelUsage of {model}: 5 min + 1 h cache writes {w5 + w1} != total {total_w}")
        out[model] = {"input": int(u["inputTokens"]), "output": int(u["outputTokens"]),
                      "cache_read": int(u["cacheReadInputTokens"]), "cache_write_5m": int(w5),
                      "cache_write_1h": int(w1), "thinking": int(u.get("thinkingTokens") or 0)}
    return out


def transcript_usage(path: Path) -> dict[str, Any]:
    """Aggregate the stream-json transcript over distinct assistant message ids.

    The CLI writes one line per content block and repeats the message's usage on each, so usage is counted once
    per message id. Input and cache counts per message are final, so their sums are exact. A message's
    output_tokens on these lines can be partial, so the output sum is a lower bound. The last result event, if
    any, is returned as well.
    """
    by_id: dict[str, dict[str, Any]] = {}
    result: dict | None = None
    n_lines = n_unparsed = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                n_unparsed += 1
                continue
            if ev.get("type") == "assistant":
                msg = ev.get("message") or {}
                mid, usage = msg.get("id"), msg.get("usage")
                if not mid or not usage:
                    continue
                cc = usage.get("cache_creation") or {}
                rec = {"model": msg.get("model"), "input": int(usage.get("input_tokens") or 0),
                       "cache_read": int(usage.get("cache_read_input_tokens") or 0),
                       "cache_write_5m": int(cc.get("ephemeral_5m_input_tokens") or 0),
                       "cache_write_1h": int(cc.get("ephemeral_1h_input_tokens") or 0),
                       "cache_write_total": int(usage.get("cache_creation_input_tokens") or 0),
                       "output": int(usage.get("output_tokens") or 0)}
                prev = by_id.get(mid)
                if prev is None:
                    by_id[mid] = rec
                else:
                    prev["output"] = max(prev["output"], rec["output"])
            elif ev.get("type") == "result":
                result = ev
    models = sorted({r["model"] for r in by_id.values() if r["model"]})
    agg = {k: sum(r[k] for r in by_id.values()) for k in
           ("input", "cache_read", "cache_write_5m", "cache_write_1h", "cache_write_total", "output")}
    return {"n_lines": n_lines, "n_unparsed": n_unparsed, "n_messages": len(by_id), "models": models,
            "sums": agg, "result": result}


def list_price_usd(tokens: dict[str, int], model: str) -> float:
    """API-equivalent cost at list price for one model's token counts."""
    if model not in PRICES_PER_MTOK:
        raise T00Error(f"no list price recorded for model {model}")
    p = PRICES_PER_MTOK[model]
    return (tokens["input"] * p["input"] + tokens["output"] * p["output"] + tokens["cache_read"] * p["cache_read"]
            + tokens["cache_write_5m"] * p["cache_write_5m"] + tokens["cache_write_1h"] * p["cache_write_1h"]) / 1e6


# ----------------------------------------------------------------------------------------------- tensor layouts

def npz_layout(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Tensor name -> (shape, dtype) from an .npz, reading only the array headers."""
    out: dict[str, tuple[tuple[int, ...], str]] = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                version = npy_format.read_magic(f)
                if version == (1, 0):
                    shape, _, dtype = npy_format.read_array_header_1_0(f)
                elif version in ((2, 0), (3, 0)):
                    shape, _, dtype = npy_format.read_array_header_2_0(f)
                else:
                    raise T00Error(f"{path}: unsupported npy version {version} in {name}")
            key = name[:-4] if name.endswith(".npy") else name
            out[key] = (tuple(int(s) for s in shape), str(dtype))
    return out


def shapes_layout(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Tensor name -> (shape, dtype) from a weightcode.model_shapes.v0 file."""
    o = _load_json(path)
    return {t["name"]: (tuple(int(s) for s in t["shape"]), str(t["dtype"])) for t in o["tensors"]}


def inference_check(sdir: Path) -> tuple[dict | None, str | None]:
    """Compare the final model's tensor layout with the base's. Inference cost equals the base's when every
    tensor keeps its name, shape and dtype, because the forward pass is the harness's fixed forward.py."""
    final = sdir / "final" / "model.npz"
    if not final.is_file():
        return None, "no final/model.npz (the session wrote no final model)"
    if (sdir / "trial" / "model.npz").is_file():
        base, basis = npz_layout(sdir / "trial" / "model.npz"), "trial/model.npz"
    elif (sdir / "trial" / "model_shapes.json").is_file():
        base, basis = shapes_layout(sdir / "trial" / "model_shapes.json"), "trial/model_shapes.json"
    else:
        return None, "no base layout in trial/ (neither model.npz nor model_shapes.json)"
    fin = npz_layout(final)
    n_params = sum(_prod(s) for s, _ in fin.values())
    diff = sorted(k for k in set(base) | set(fin) if base.get(k) != fin.get(k))
    return {"equal_to_base": not diff, "basis": f"final/model.npz headers against {basis}",
            "n_tensors": len(fin), "n_parameters": n_params, "differing_tensors": diff}, None


def _prod(shape: Iterable[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def find_memory_keys(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Paths of keys whose name suggests a peak-memory measurement."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if any(h in k.lower() for h in MEMORY_KEY_HINTS) and isinstance(v, (int, float)):
                found.append((path, v))
            found.extend(find_memory_keys(v, path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(find_memory_keys(v, f"{prefix}[{i}]"))
    return found


# ----------------------------------------------------------------------------------------------- readout sources

def index_streams(runs: Path) -> dict[str, dict]:
    """session_id -> {stream, round, prev_readouts, readouts, instance_id} from the loop-teacher stream records."""
    out: dict[str, dict] = {}
    for p in sorted((runs / "loop_streams").glob("loop_teacher_*.json")):
        rec = _load_json(p)
        rounds = rec.get("rounds") or []
        for i, rnd in enumerate(rounds):
            for s in rnd.get("sessions") or []:
                sid = s.get("session_id")
                if not sid:
                    continue
                if sid in out:
                    raise T00Error(f"session {sid} appears in two stream rounds")
                prev = rounds[i - 1].get("readouts") if i > 0 else None
                out[sid] = {"stream": p.name, "round": rnd.get("round"), "instance_id": rnd.get("instance_id"),
                            "condition": rnd.get("condition"), "session_status": s.get("status"),
                            "readouts": rnd.get("readouts"), "prev_readouts": prev,
                            "seen_conditions": [r.get("condition") for r in rounds[:i]]}
    return out


def index_fscil(runs: Path) -> dict[str, dict]:
    """session_id -> {lineage, session, readout} from the FSCIL lineage states."""
    out: dict[str, dict] = {}
    root = runs / "fscil_lineages"
    if not root.is_dir():
        return out
    for p in sorted(root.glob("*/state.json")):
        st = _load_json(p)
        sessions = st.get("sessions") or {}
        items = sessions.items() if isinstance(sessions, dict) else enumerate(sessions)
        for k, s in items:
            sid = s.get("session_id")
            if sid:
                out[sid] = {"lineage": p.parent.name, "session": k, "readout": s.get("readout") or {}}
    return out


def _pair(d: Any, base_key: str = "base_acc", child_key: str = "child_acc") -> tuple[float | None, float | None]:
    if not isinstance(d, dict):
        return None, None
    return d.get(base_key), d.get(child_key)


def readout(sid: str, runs: Path, streams: dict, fscil: dict) -> dict:
    """Target and retention readouts, before and after the session, with their source."""
    info = runs / "devreadouts" / "learn" / sid / "info.json"
    if info.is_file():
        fin = _load_json(info).get("final") or {}
        pairs = {k: list(_pair(v)) for k, v in fin.items() if isinstance(v, dict) and "child_acc" in v}
        tkey = "vselect_new" if "vselect_new" in pairs else ("vselect" if "vselect" in pairs else None)
        okey = "vselect_clean" if "vselect_clean" in pairs else ("vselect_old" if "vselect_old" in pairs else None)
        return {"source": str(info.relative_to(runs)), "target_metric": tkey and f"final.{tkey} acc",
                "old_metric": okey and f"final.{okey} acc", "target": pairs.get(tkey) if tkey else None,
                "old": pairs.get(okey) if okey else None, "all_pairs": pairs,
                "why_no_old": None if okey else "info.json final holds no clean or old-class readout"}
    if sid in streams:
        s = streams[sid]
        iid = s["instance_id"]
        cur, prev = s["readouts"] or {}, s["prev_readouts"] or {}
        if iid not in cur or iid not in prev:
            return {"source": None, "why": f"stream {s['stream']} round {s['round']} lacks readouts for {iid}"}
        return {"source": f"loop_streams/{s['stream']} round {s['round']}",
                "target_metric": f"V_select acc on {iid}, previous round's student -> this round's student",
                "old_metric": "V_select clean acc, previous round's student -> this round's student",
                "target": [_get(prev, iid, "select", "child_acc"), _get(cur, iid, "select", "child_acc")],
                "old": [_get(prev, iid, "clean", "child_acc"), _get(cur, iid, "clean", "child_acc")],
                "all_pairs": None, "why_no_old": None}
    if sid in fscil:
        f = fscil[sid]
        r = f["readout"]
        return {"source": f"fscil_lineages/{f['lineage']}/state.json session {f['session']}",
                "target_metric": "class-balanced V_select acc on this session's new classes (bal_new)",
                "old_metric": "class-balanced V_select acc on the base classes (bal_old)",
                "target": [r.get("bal_new_base"), r.get("bal_new_child")],
                "old": [r.get("bal_old_base"), r.get("bal_old_child")],
                "all_pairs": {k: [r.get(f"bal_{k}_base"), r.get(f"bal_{k}_child")] for k in ("new", "prev", "old", "seen")},
                "why_no_old": None}
    files = sorted(p for pat in SESSION_READOUT_PATTERNS for p in (runs / "devreadouts").glob(pat.format(sid=sid)))
    if len(files) > 1:
        raise T00Error(f"more than one top-level devreadout names {sid}: {[p.name for p in files]}")
    if files:
        rows = _jsonl(files[0])
        if len(rows) != 1:
            raise T00Error(f"{files[0].name}: expected one row, found {len(rows)}")
        o = rows[0]
        src = str(files[0].relative_to(runs))
        if "readouts_seqskill" in o:
            rs = o["readouts_seqskill"]
            return {"source": src, "target_metric": "V_select exact match, part (i) new operator n 4 to 8",
                    "old_metric": "V_select exact match, part (iv) old operators",
                    "target": [_get(rs, "i", "em_base"), _get(rs, "i", "em_child")],
                    "old": [_get(rs, "iv", "em_base"), _get(rs, "iv", "em_child")],
                    "all_pairs": {k: [_get(rs, k, "em_base"), _get(rs, k, "em_child")] for k in ("i", "ii", "iii", "iv")},
                    "why_no_old": None}
        if "readouts" in o:
            return {"source": src, "target_metric": f"V_select acc ({o.get('condition')})", "old_metric": None,
                    "target": list(_pair(o["readouts"])), "old": None, "all_pairs": None,
                    "why_no_old": f"{files[0].name} holds the target split's V_select readout only; no retention readout recorded"}
        raise T00Error(f"{files[0].name}: no readouts block this table knows")
    return {"source": None, "why": "no readout file names this session (devreadouts/learn, top-level devreadouts, "
                                   "loop_streams and fscil_lineages searched)"}


# ----------------------------------------------------------------------------------------------- DEV_RESULTS

def dev_results_index(path: Path) -> tuple[dict[str, int], list[str]]:
    """Heading text -> line number, and the file's lines."""
    lines = path.read_text().splitlines()
    heads = {ln: i + 1 for i, ln in enumerate(lines) if ln.startswith("## ")}
    return heads, lines


def _section(lines: list[str], start: int) -> str:
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start - 1:end])


# ----------------------------------------------------------------------------------------------- one session

def session_row(sid: str, runs: Path, ctx: dict) -> dict:
    sdir = runs / "sessions" / sid
    sj = _load_json(sdir / "session.json")
    sr = _maybe_json(sdir / "session_result.json")
    run = _maybe_json(sdir / "run.json")
    sel = _maybe_json(sdir / "final" / "selection.json")
    patch_path = sdir / "final" / "patch.json"
    patch = _maybe_json(patch_path)
    prun = _maybe_json(sdir / "final" / "program_run.json")
    reading = ctx["readings"].get(sid)
    if reading is None:
        raise T00Error(f"session_readings.json has no entry for {sid}")
    row = Row()
    has_result = sr is not None
    no_result = "no session_result.json (the session ended before finalize)"
    tr_path = sdir / "logs" / "transcript.jsonl"
    tr = transcript_usage(tr_path) if tr_path.is_file() else None

    # identity and provenance
    loop = sj.get("loop") or {}
    fs = sj.get("fscil") or {}
    if loop:
        run_id = f"{loop.get('stream_id')}#round{loop.get('round')}"
    elif sid in ctx["fscil"]:
        f = ctx["fscil"][sid]
        run_id = f"fscil_lineages/{f['lineage']}#session{f['session']}"
    else:
        run_id = sid
    row.set("run_id", run_id)
    row.set("session_id", sid)
    row.set("experiment", reading["experiment"])
    row.set("round", reading.get("round", loop.get("round") or fs.get("session")),
            "single-round session (no loop or lineage round)", na=True)
    row.set("created_at", sj.get("created_at"), "session.json has no created_at")
    row.set("task_id", sj.get("instance_id"), "session.json has no instance_id")
    row.set("condition", sj.get("condition"), "session.json has no condition (see condition_label)")
    row.set("condition_label", reading.get("condition"), "session_readings.json gives no condition label")
    row.set("dataset", sj.get("dataset"), "session.json has no dataset")
    row.set("arm", sj.get("arm"), "session.json has no arm")
    idx_row = _get(sj, "instance", "index_row") or {}
    in_index = sj.get("instance_id") in ctx["index_ids"]
    superseded = sj.get("instance_id") in ctx["superseded_ids"]
    row.set("instance_index", {"in_index": in_index, "superseded": superseded, "arch": idx_row.get("arch"),
                               "split_role": idx_row.get("split_role") or sj.get("split_role")})
    row.set("model_id", sj.get("model_id"), "session.json has no model_id")
    row.set("model_seed", sj.get("seed"), "session.json has no seed")
    row.set("checkpoint_hash", sj.get("real_model_sha256"), "session.json has no real_model_sha256")
    row.set("shown_model_sha256", sj.get("shown_model_sha256"), "session.json has no shown_model_sha256")
    for parent_src, parent in (("final/patch.json parent_sha256", (patch or {}).get("parent_sha256")),
                               ("session.json loop.student_sha256", loop.get("student_sha256")),
                               ("session.json fscil.parent_params_sha256", fs.get("parent_params_sha256")),
                               ("session.json real_model_sha256 (the model the trial was built from; no final patch)",
                                sj.get("real_model_sha256"))):
        if parent is not None:
            break
    row.set("parent_checkpoint", parent, "no final/patch.json, loop student, lineage parent or real_model_sha256")
    row.set("parent_checkpoint_source", parent_src if parent is not None else None, "no parent recorded")
    row.set("seen_tasks", loop.get("prev_instances") if loop else (fs.get("prev_classes") if fs else None),
            "single-task session: no earlier round or lineage session", na=True)
    git = _get(sj, "provenance", "git") or {}
    row.set("code_commit", git.get("head"), "session.json provenance.git has no head")
    row.set("code_dirty", git.get("dirty"), "session.json provenance.git has no dirty flag")
    editor = sj.get("editor") or {}
    row.set("editor_model", editor.get("model"), "session.json editor has no model")
    row.set("effort", editor.get("effort"), "session.json editor has no effort")
    row.set("claude_version", editor.get("claude_version") or _get(sj, "claude", "version"),
            "session.json records no Claude Code version")
    kept = sj.get("kept_rule")
    row.set("protocol_version", {"evidence_rule": sj.get("evidence_rule"),
                                 "kept_rule": kept.get("name") if isinstance(kept, dict) else None,
                                 "system_prompt_sha256": _get(sj, "prompts", "system_sha256"),
                                 "task_prompt_sha256": _get(sj, "prompts", "task_sha256"),
                                 "program_driver_sha256": _get(sj, "program", "driver_sha256")})

    # information regime and evidence
    exp = sj.get("experience") if isinstance(sj.get("experience"), dict) else {}
    trial_arrays = {}
    for name in ("experience.npz", "evidence.npz"):
        p = sdir / "trial" / name
        if p.is_file():
            with zipfile.ZipFile(p) as z:
                trial_arrays[name] = sorted(n[:-4] if n.endswith(".npy") else n for n in z.namelist())
    row.set("information_regime", {
        "arm": sj.get("arm"), "evidence_source": sj.get("evidence_source"), "e11_variant": sj.get("e11_variant"),
        "weights_shown": not bool(sj.get("blind")), "blind_predictions_shown": _get(sj, "blind", "predictions_shown"),
        "labels_new_shown_flag": exp.get("labels_new_shown"), "n_new_rows": exp.get("n_new"),
        "n_old_rows": exp.get("n_old"), "trial_arrays": trial_arrays,
        "trial_label_arrays": sorted({a for names in trial_arrays.values() for a in names if a.startswith("y_")}),
        "library": bool(sj.get("library")), "lift_rel_frob": sj.get("lift_rel_frob")})
    row.set("evidence_ids_or_manifest", {"evidence_content_sha256": sj.get("evidence_content_sha256"),
                                         "experience_content_sha256": exp.get("content_sha256"),
                                         "evidence_path": sj.get("evidence_path")})

    # budgets and use
    budget = sj.get("budget") or {}
    checks_used, checks_src = None, None
    q = (sr or {}).get("queries") or {}
    if q.get("used") is not None:
        checks_used, checks_src = q["used"], "session_result.queries.used"
    elif sel is not None and sel.get("queries_used") is not None:
        checks_used, checks_src = sel["queries_used"], "final/selection.json queries_used (session_result.queries.used is null)"
    elif not has_result:
        cpath = sdir / "logs" / "checks.jsonl"
        checks_used = _count_lines(cpath) if cpath.is_file() else 0
        checks_src = "logs/checks.jsonl lines" if cpath.is_file() else "no logs/checks.jsonl (no check logged)"
    cpath = sdir / "logs" / "checks.jsonl"
    checks_log = _count_lines(cpath) if cpath.is_file() else 0
    cells_used = _get(sr, "cells", "used")
    cells_src = "session_result.cells.used"
    if cells_used is None and (sdir / "logs" / "cells.jsonl").is_file():
        cells_used = _count_lines(sdir / "logs" / "cells.jsonl")
        cells_src = "logs/cells.jsonl lines (completed cells; any in-flight cell not counted)"
    row.set("feedback_budget_and_used", {
        "checks_budget": budget.get("patch_checks"), "checks_used": checks_used, "checks_used_source": checks_src,
        "checks_log_lines": checks_log, "check_cap": sj.get("check_cap"),
        "cells_budget": budget.get("python_cells"), "cells_used": cells_used, "cells_used_source": cells_src,
        "cells_by_status": _get(sr, "cells", "by_status"), "kernel_restarts": _get(sr, "cells", "restarts"),
        "wall_budget_s": budget.get("wall_timeout_s")})
    if checks_used is None:
        row.missing["feedback_budget_and_used.checks_used"] = "no check count in session_result, selection.json or logs"
    row.set("checks_used", checks_used, "no check count recorded")
    row.set("checks_budget", budget.get("patch_checks"), "session.json budget has no patch_checks")
    row.set("cells_used", cells_used, "no cell count recorded")
    row.set("cells_budget", budget.get("python_cells"), "session.json budget has no python_cells")
    sel_rule = kept.get("name") if isinstance(kept, dict) else None
    sel_src = "session.json kept_rule.name"
    if sel_rule is None and sel is not None and sel.get("rule"):
        sel_rule, sel_src = sel["rule"], "final/selection.json rule (session.json kept_rule is null)"
    row.set("selection_rule", sel_rule, "no kept rule in session.json and no final/selection.json")
    row.set("selection_rule_source", sel_src if sel_rule is not None else None, "no selection rule recorded")
    row.set("n_candidates_checked", len(sel.get("candidates") or []) if sel else None, "no final/selection.json")

    # status
    if has_result:
        status = sr.get("status")
        launch = sr.get("launch") or {}
        res = _get(sr, "stream", "result") or {}
        problems = []
        if sr.get("infra"):
            problems.append(f"infra: {sr.get('infra_reasons')}")
        if launch.get("exit_code") not in (0, None) or launch.get("killed"):
            problems.append(f"launch exit {launch.get('exit_code')} killed {launch.get('killed')} {launch.get('kill_reason')}")
        if res.get("is_error"):
            problems.append(f"CLI result is_error ({res.get('subtype')}, {res.get('api_error_status')})")
        if sr.get("finalize_error"):
            problems.append(f"finalize_error: {sr.get('finalize_error')}")
        row.set("status", status, "session_result.json has no status")
        row.set("failure_reason", "; ".join(problems) or None,
                "no failure: complete, no infra flag, exit 0, CLI result not an error", na=True)
        row.set("infra", sr.get("infra"), "session_result.json has no infra flag")
    else:
        inflight = sdir / "logs" / "cells.jsonl.inflight"
        obs = ["no session_result.json"]
        if tr is None:
            obs.append("no logs/transcript.jsonl")
        elif tr["result"] is None:
            obs.append("no result event in logs/transcript.jsonl")
        else:
            obs.append("logs/transcript.jsonl has a result event")
        if inflight.is_file():
            infl = _load_json(inflight)
            obs.append(f"logs/cells.jsonl.inflight holds cell {infl.get('index')} at {infl.get('timestamp')}")
        reason = reading.get("reason") or reading.get("reading")
        row.set("status", "aborted")
        row.set("failure_reason", "; ".join(obs) + (f". DEV_RESULTS/PREREG: {reason}" if reason else ""))
        row.set("infra", None, no_result)
    row.set("rejected_attempts", len((sr or {}).get("rejected_attempts") or []) if has_result else None, no_result)

    # final
    final = (sr or {}).get("final") or {}
    prog_sha = _get(final, "program", "sha256") or _get(sr, "declared_final", "program_sha256")
    if prog_sha is not None:
        row.set("program_hash", prog_sha)
    elif not has_result:
        row.set("program_hash", None, no_result)
    elif final.get("is_base_noop"):
        row.set("program_hash", None, "the final is the base (no-op): no program", na=True)
    elif sj.get("program") is None:
        row.set("program_hash", None, f"arm {sj.get('arm')} submits patches, not programs", na=True)
    else:
        row.set("program_hash", None, "session_result.final has no program sha256")
    if patch_path.is_file():
        row.set("final_patch_sha256", sha256_file(patch_path))
    elif has_result and final.get("is_base_noop"):
        row.set("final_patch_sha256", None, "the final is the base (no-op): no patch", na=True)
    else:
        row.set("final_patch_sha256", None, no_result if not has_result else "no final/patch.json")
    row.set("final_child_sha256", final.get("child_sha256"), no_result if not has_result else "session_result.final has no child_sha256")
    row.set("is_base_noop", final.get("is_base_noop"), no_result if not has_result else "session_result.final has no is_base_noop")
    row.set("n_changed", final.get("n_changed"), no_result if not has_result else "session_result.final has no n_changed")
    row.set("final_kind", final.get("kind"), no_result if not has_result else "session_result.final has no kind")

    # audit
    audits: dict[str, Any] = {}
    for name in AUDIT_FILES:
        a = _maybe_json(sdir / name)
        if a is not None:
            audits[name] = {"verdict": a.get("verdict"), "candidate_count": a.get("candidate_count"),
                            "candidate_count_is_lower_bound": a.get("candidate_count_is_lower_bound"),
                            "config_count": a.get("config_count"),
                            "label_access_cells": len(a.get("label_access_cells") or []),
                            "rule_counts": a.get("rule_counts"), "rules_version": a.get("rules_version")}
    latest = next((n for n in reversed(AUDIT_FILES) if n in audits), None)
    row.set("audit", audits or None, "no audit*.json record")
    row.set("audit_verdict", _get(audits, "audit.json", "verdict"),
            "no audit.json" if "audit.json" not in audits else "audit.json has no verdict")
    row.set("audit_latest", {"file": latest, "verdict": audits[latest]["verdict"]} if latest else None,
            "no audit record")
    row.set("audit_in_session_result", _get(sr, "audit", "verdict") if has_result else None,
            no_result if not has_result else f"session_result.audit has no verdict (status {_get(sr, 'audit', 'status')})")
    row.set("audit_judge_present", (sdir / "audit_judge.json").is_file())
    row.set("code_changed_during_session", (sr or {}).get("code_changed_during_session") if has_result else None,
            no_result)

    # readouts
    ro = readout(sid, runs, ctx["streams"], ctx["fscil"])
    if ro.get("source") is None:
        why = ro.get("why") or "no readout"
        if not has_result:
            why = f"{no_result}; {why}"
        for k in ("target_before", "target_after", "old_before", "old_after"):
            row.set(k, None, why)
        row.set("readout", None, why)
    else:
        tb, ta = (ro["target"] or [None, None])
        ob, oa = (ro["old"] or [None, None])
        no_old = ro.get("why_no_old") or "readout source holds no value"
        row.set("target_before", tb, "readout source holds no base value")
        row.set("target_after", ta, "readout source holds no child value")
        row.set("old_before", ob, no_old)
        row.set("old_after", oa, no_old)
        row.set("readout", {"source": ro["source"], "target_metric": ro["target_metric"],
                            "old_metric": ro["old_metric"], "all_pairs": ro.get("all_pairs")})

    # frozen-rule reading: met null is either "no_reading" (DEV_RESULTS has none: missing) or "not_applicable"
    # (no criterion applies to this session: a stop before launch, a gate read as not applicable, a lineage
    # session that carries no criterion)
    met = reading.get("met")
    na_reading = reading.get("null_kind") == "not_applicable"
    why = reading.get("reason") or "no DEV_RESULTS reading"
    row.set("success_by_frozen_rule", met, why, na=na_reading)
    row.set("success_scope", reading.get("met_scope"), why, na=na_reading)
    row.set("bar", reading.get("bar"), why, na=na_reading)
    row.set("frozen_rule_reading", reading.get("reading"), why, na=na_reading)
    row.set("dev_results_heading", reading.get("dev_results_heading"), why)
    row.set("secondary_readings", reading.get("secondary") or [])
    row.set("notes", reading.get("notes") or [])

    # tokens and cost
    row.set("transcript", {"present": tr is not None, "n_lines": tr and tr["n_lines"],
                           "n_unparsed": tr and tr["n_unparsed"], "n_messages": tr and tr["n_messages"],
                           "has_result_event": bool(tr and tr["result"])})
    res = _get(sr, "stream", "result") or {}
    mu = res.get("model_usage") or (tr and tr["result"] and tr["result"].get("modelUsage"))
    cost: dict[str, Any] = {"price_basis": PRICE_SOURCE}
    if mu:
        per_model = tokens_from_model_usage(mu)
        tok = {k: sum(v[k] for v in per_model.values()) for k in TOKEN_KEYS + ("thinking",)}
        usd = sum(list_price_usd(v, m) for m, v in per_model.items())
        cost.update({"tokens_source": "CLI result modelUsage (session_result.stream.result.model_usage)",
                     "models": sorted(per_model), "output_is_lower_bound": False})
        cli = res.get("total_cost_usd")
        if cli is None and tr and tr["result"]:
            cli = tr["result"].get("total_cost_usd")
        row.set("cli_total_cost_usd", cli, "the CLI result records no total_cost_usd")
        row.set("api_equiv_cost_usd", usd)
        row.set("api_equiv_cost_usd_lower_bound", None, "not needed: the exact cost is known", na=True)
        cost["cli_matches_list_price"] = None if cli is None else abs(cli - usd) <= CLI_MATCH_TOL_USD
        if tr is not None:
            s = tr["sums"]
            cost["transcript_input_cache_sums_match"] = all(
                s[k] == tok[k] for k in ("input", "cache_read", "cache_write_5m", "cache_write_1h"))
    elif tr is not None:
        s = tr["sums"]
        tok = {"input": s["input"], "output": s["output"], "cache_read": s["cache_read"],
               "cache_write_5m": s["cache_write_5m"], "cache_write_1h": s["cache_write_1h"], "thinking": None}
        models = tr["models"]
        cost.update({"tokens_source": "logs/transcript.jsonl summed over distinct assistant message ids",
                     "models": models, "output_is_lower_bound": True})
        if s["cache_write_5m"] + s["cache_write_1h"] != s["cache_write_total"]:
            raise T00Error(f"{sid}: transcript cache-write split does not add up")
        if len(models) != 1:
            raise T00Error(f"{sid}: transcript names models {models}; cost per model is not separable here")
        lb = list_price_usd({k: tok[k] for k in TOKEN_KEYS}, models[0])
        row.set("cli_total_cost_usd", None, "no CLI result event with modelUsage (session_result.json and transcript)")
        row.set("api_equiv_cost_usd", None,
                "output tokens are a lower bound (no CLI result event); input and cache tokens are exact")
        row.set("api_equiv_cost_usd_lower_bound", lb)
        row.missing["tokens.thinking"] = "thinking tokens are reported only in the CLI result event"
    else:
        tok = None
        for k in ("cli_total_cost_usd", "api_equiv_cost_usd", "api_equiv_cost_usd_lower_bound"):
            row.set(k, None, "no transcript and no CLI result")
    row.set("llm_tokens", tok, "no transcript and no CLI result")
    row.set("cost_detail", cost)

    # time
    gen = sr.get("wall_s") if has_result else None
    if has_result:
        row.set("generation_time_s", gen if gen is not None else _get(sr, "launch", "wall_s"),
                "session_result has no wall_s")
        row.set("generation_time_s_lower_bound", None, "not needed: the session wall time is recorded", na=True)
        row.set("api_time_s", (res.get("duration_api_ms") / 1000.0) if res.get("duration_api_ms") is not None else None,
                "CLI result has no duration_api_ms")
        row.set("num_turns", res.get("num_turns"), "CLI result has no num_turns")
    else:
        stamps = []
        cl = sdir / "logs" / "cells.jsonl"
        if cl.is_file():
            stamps += [r.get("timestamp") for r in _jsonl(cl) if r.get("timestamp")]
        infl = sdir / "logs" / "cells.jsonl.inflight"
        if infl.is_file():
            stamps.append(_load_json(infl).get("timestamp"))
        row.set("generation_time_s", None, "no session_result.json: the session end was not recorded")
        lb = None
        if stamps and sj.get("created_at"):
            t0 = datetime.fromisoformat(sj["created_at"])
            lb = max((datetime.fromisoformat(t) - t0).total_seconds() for t in stamps if t)
        row.set("generation_time_s_lower_bound", lb, "no timestamped cell after session creation")
        row.set("api_time_s", None, no_result)
        row.set("num_turns", None, no_result)
    if prun is not None:
        run_rec = prun.get("run") or {}
        result = run_rec.get("result") or {}
        row.set("execution_time_s", run_rec.get("duration_s"), "final/program_run.json has no run.duration_s")
        row.set("program_update_s", result.get("update_s"), "final/program_run.json run.result has no update_s")
        row.set("program_wall_cap_s", prun.get("wall_s"), "final/program_run.json has no wall_s")
    else:
        na_run = False
        if not has_result:
            why = no_result
        elif final.get("is_base_noop"):
            why, na_run = "the final is the base (no-op): no program ran", True
        elif sj.get("program") is None:
            why, na_run = f"arm {sj.get('arm')} submits patches: no program run", True
        else:
            why = "no final/program_run.json"
        for k in ("execution_time_s", "program_update_s", "program_wall_cap_s"):
            row.set(k, None, why, na=na_run)

    # program clock: None means the driver before virtual-v1 (a program that calls the clock gets the real one)
    prog_cfg = sj.get("program")
    clock = (prog_cfg or {}).get("clock") or (prun or {}).get("clock") or _get(prun, "run", "result", "clock")
    if prog_cfg is None:
        row.set("program_clock", None, f"arm {sj.get('arm')} has no program driver", na=True)
    else:
        row.set("program_clock", clock, "None: no clock key in session.json program or program_run.json, i.e. the "
                                       "driver before virtual-v1, which gives a program the real clock", na=True)
    reads = _get(prun, "run", "result", "clock_reads")
    if not has_result:
        row.set("program_clock_reads", None, no_result)
    elif prun is None:
        row.set("program_clock_reads", None, "no final program run", na=True)
    else:
        row.set("program_clock_reads", reads, "final/program_run.json result has no clock_reads key (no run from "
                                              "the driver before virtual-v1 records one)")

    # inference and memory
    inf, why = inference_check(sdir)
    row.set("inference_cost", inf, why)
    mem_hits = find_memory_keys(sr) + find_memory_keys(prun)
    sandbox_cap = _get(sj, "sandbox", "memory_bytes")
    if mem_hits:
        row.set("peak_memory", {path: v for path, v in mem_hits})
    else:
        cap_s = f"the sandbox memory cap was {sandbox_cap} bytes" if sandbox_cap is not None else \
            "session.json records no sandbox memory cap"
        row.set("peak_memory", None, "not recorded by the harness (no peak or RSS field in session_result.json or "
                                     f"program_run.json); {cap_s}")
    row.set("sandbox_memory_cap_bytes", sandbox_cap, "session.json sandbox has no memory_bytes")

    # manifest
    man = ctx["manifest"].get(sid)
    row.set("manifest", {"present": man is not None, "status": (man or {}).get("status"),
                         "audit_verdict": (man or {}).get("audit_verdict"), "score": (man or {}).get("score")})
    row.set("result_path", str(sdir))
    return row.as_dict()


# ----------------------------------------------------------------------------------------------- problems

def _why(r: dict, key: str) -> str:
    """The recorded reason a field of row r is null ("" when the field is not null)."""
    return r["missing"].get(key) or r["not_applicable"].get(key) or ""


def problems(rows: list[dict], ctx: dict) -> list[str]:
    """Observed problems only, one line each."""
    out: list[str] = []

    def ids(pred) -> list[str]:
        return [r["session_id"] for r in rows if pred(r)]

    cc = [(r["session_id"], r["code_changed_during_session"]) for r in rows if r["code_changed_during_session"]]
    if cc:
        out.append(f"code_changed_during_session non-empty in {len(cc)} sessions (provenance markers that changed "
                   "between launch and finalize): " + ", ".join(f"{s} ({'+'.join(map(str, c))})" for s, c in cc))
    v = ids(lambda r: r["audit_verdict"] == "VIOLATION")
    if v:
        out.append(f"audit.json verdict VIOLATION in {len(v)} sessions: {', '.join(v)}")
    vl = [(r["session_id"], r["audit_latest"]["file"]) for r in rows
          if r["audit_latest"] and r["audit_latest"]["verdict"] == "VIOLATION" and r["audit_latest"]["file"] != "audit.json"]
    if vl:
        out.append(f"latest audit rerun verdict VIOLATION in {len(vl)} sessions: "
                   + ", ".join(f"{s} ({f})" for s, f in vl))
    changed = [(r["session_id"], r["audit_verdict"], r["audit_latest"]["verdict"], r["audit_latest"]["file"])
               for r in rows if r["audit_latest"] and r["audit_verdict"] is not None
               and r["audit_latest"]["verdict"] != r["audit_verdict"]]
    if changed:
        out.append("audit verdict differs between audit.json and the latest rerun in "
                   f"{len(changed)} sessions: " + ", ".join(f"{s} {a} -> {b} ({f})" for s, a, b, f in changed))
    sr_mis = [(r["session_id"], r["audit_in_session_result"], r["audit_verdict"]) for r in rows
              if r["status"] != "aborted" and r["audit_in_session_result"] != r["audit_verdict"]]
    if sr_mis:
        out.append("session_result.audit verdict differs from audit.json in "
                   + ", ".join(f"{s} (session_result {a}, audit.json {b})" for s, a, b in sr_mis))
    ab = [r for r in rows if r["status"] == "aborted"]
    if ab:
        out.append(f"aborted sessions (no session_result.json) {len(ab)}: "
                   + "; ".join(f"{r['session_id']} ({r['experiment']}): {r['failure_reason']}" for r in ab))
    inf = ids(lambda r: r["infra"] is True or (r["status"] != "aborted" and r["failure_reason"]))
    out.append(f"infra flag or launch/CLI error in completed sessions: {', '.join(inf) if inf else 'none'}")
    to = [(r["session_id"], (r["feedback_budget_and_used"]["cells_by_status"] or {}).get("timeout", 0),
           r["feedback_budget_and_used"]["kernel_restarts"]) for r in rows if r["status"] != "aborted"]
    to = [t for t in to if t[1] or t[2]]
    if to:
        out.append(f"cell timeouts or kernel restarts in {len(to)} completed sessions: "
                   + ", ".join(f"{s} ({a} timeout, {b} restart)" for s, a, b in to))
    rej = [(r["session_id"], r["rejected_attempts"]) for r in rows if r["rejected_attempts"]]
    if rej:
        out.append("rejected attempts recorded: " + ", ".join(f"{s} ({n})" for s, n in rej))
    nt = ids(lambda r: not r["transcript"]["present"])
    out.append(f"missing transcripts: {', '.join(nt) if nt else 'none'}")
    nre = ids(lambda r: r["status"] != "aborted" and not r["transcript"]["has_result_event"])
    if nre:
        out.append(f"completed sessions whose transcript has no result event: {', '.join(nre)}")
    nr = [(r["session_id"], _why(r, "target_after")) for r in rows if r["target_after"] is None]
    if nr:
        out.append(f"no target readout in {len(nr)} sessions: " + "; ".join(f"{s} ({w})" for s, w in nr))
    no = [r["session_id"] for r in rows if r["target_after"] is not None and r["old_after"] is None]
    if no:
        out.append(f"target readout but no retention readout in {len(no)} sessions: {', '.join(no)}")
    qn = [r["session_id"] for r in rows if (r["feedback_budget_and_used"]["checks_used_source"] or "").startswith("final/selection.json")]
    if qn:
        out.append(f"session_result.queries.used is null in {len(qn)} sessions (checks taken from final/selection.json): "
                   + ", ".join(qn))
    cm = [(r["session_id"], r["checks_used"], r["feedback_budget_and_used"]["checks_log_lines"]) for r in rows
          if r["checks_used"] is not None and r["checks_used"] != r["feedback_budget_and_used"]["checks_log_lines"]]
    if cm:
        out.append("checks used differ from logs/checks.jsonl lines in: "
                   + ", ".join(f"{s} ({a} vs {b} lines)" for s, a, b in cm))
    real = ids(lambda r: _why(r, "program_clock").startswith("None: no clock key"))
    virt = ids(lambda r: r["program_clock"] == "virtual-v1")
    out.append(f"program clock: virtual-v1 in {len(virt)} sessions; no clock key (driver before virtual-v1, real "
               f"clock) in {len(real)} program sessions: {', '.join(real)}; "
               f"{len(ids(lambda r: _why(r, 'program_clock').startswith('arm ')))} patch-arm sessions "
               "have no program driver")
    cr = [(r["session_id"], r["program_clock_reads"], r["program_clock"]) for r in rows if r["program_clock_reads"]]
    if cr:
        out.append("final programs that read the clock (program_run clock_reads > 0): "
                   + ", ".join(f"{s} ({n} reads, clock {c})" for s, n, c in cr))
    kd = [r["session_id"] for r in rows if any("wall clock" in n for n in r["notes"])]
    if kd:
        out.append(f"known wall-clock defect recorded in DEV_RESULTS for: {', '.join(kd)}")
    cli = [r["session_id"] for r in rows if r["cost_detail"].get("cli_matches_list_price") is False]
    out.append(f"CLI total_cost_usd differs from the list-price arithmetic in: {', '.join(cli) if cli else 'none'}")
    trm = [r["session_id"] for r in rows if r["cost_detail"].get("transcript_input_cache_sums_match") is False]
    out.append(f"transcript input/cache sums differ from the CLI result in: {', '.join(trm) if trm else 'none'}")
    infx = [r["session_id"] for r in rows if r["inference_cost"] and not r["inference_cost"]["equal_to_base"]]
    out.append(f"final tensor layout differs from the base in: {', '.join(infx) if infx else 'none'}")
    nm = ids(lambda r: not r["manifest"]["present"])
    out.append(f"sessions missing from runs/manifest.jsonl: {', '.join(nm) if nm else 'none'}")
    mm = [r["session_id"] for r in rows if r["manifest"]["present"] and r["status"] != "aborted"
          and (r["manifest"]["status"] != r["status"] or r["manifest"]["audit_verdict"] != r["audit_in_session_result"])]
    if mm:
        out.append(f"manifest status or audit verdict differs from session_result in: {', '.join(mm)}")
    ni = ids(lambda r: not r["instance_index"]["in_index"])
    if ni:
        out.append(f"instance not in private/instances/index.json (other private dirs, not read): {', '.join(ni)}")
    su = ids(lambda r: r["instance_index"]["superseded"])
    if su:
        out.append(f"instance listed as superseded in the index: {', '.join(su)}")
    dirty = ids(lambda r: r["code_dirty"] is True)
    if dirty:
        out.append(f"git tree dirty at launch: {', '.join(dirty)}")
    out.extend(ctx["reading_problems"])
    if ctx["not_a_record_audits"]:
        out.append(f"{NOT_A_RECORD_AUDIT} present (excluded from the audit fields) in: "
                   + ", ".join(ctx["not_a_record_audits"]))
    return out


# ----------------------------------------------------------------------------------------------- markdown

def _f(x: Any, nd: int = 2) -> str:
    if x is None:
        return "null"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _pp(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "null"
    return f"{100 * before:.2f} -> {100 * after:.2f} ({100 * (after - before):+.2f})"


def _met(m: Any) -> str:
    return {True: "met", False: "missed", None: "null"}[m]


def _summ(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    return f"{statistics.median(xs):,.2f} [{min(xs):,.2f}, {max(xs):,.2f}]"


def _summ_int(xs: list[int]) -> str:
    if not xs:
        return "n/a"
    return f"{statistics.median(xs):,.0f} [{min(xs):,}, {max(xs):,}]"


def experiment_summary(rows: list[dict]) -> list[dict]:
    order: list[str] = []
    for r in sorted(rows, key=lambda r: r["created_at"] or ""):
        if r["experiment"] not in order:
            order.append(r["experiment"])
    out = []
    for e in order:
        rs = [r for r in rows if r["experiment"] == e]
        done = [r for r in rs if r["status"] != "aborted"]
        costs = [r["api_equiv_cost_usd"] for r in done if r["api_equiv_cost_usd"] is not None]
        outs = [r["llm_tokens"]["output"] for r in done if r["llm_tokens"]]
        tots = [sum(r["llm_tokens"][k] for k in TOKEN_KEYS) for r in done if r["llm_tokens"]]
        lbs = [r["api_equiv_cost_usd_lower_bound"] for r in rs if r["api_equiv_cost_usd_lower_bound"] is not None]
        out.append({"experiment": e, "n_sessions": len(rs), "n_complete": len(done), "n_aborted": len(rs) - len(done),
                    "n_met": sum(1 for r in done if r["success_by_frozen_rule"] is True),
                    "n_missed": sum(1 for r in done if r["success_by_frozen_rule"] is False),
                    "n_null_no_reading": sum(1 for r in done if r["success_by_frozen_rule"] is None
                                             and "success_by_frozen_rule" in r["missing"]),
                    "n_null_not_applicable": sum(1 for r in done if r["success_by_frozen_rule"] is None
                                                 and "success_by_frozen_rule" in r["not_applicable"]),
                    "met_scope": sorted({r["success_scope"] for r in rs if r["success_scope"]}),
                    "cost_usd_sum": sum(costs), "cost_usd": costs, "output_tokens": outs, "total_tokens": tots,
                    "aborted_cost_usd_lower_bound_sum": sum(lbs)})
    return out


_SID_RE = re.compile(r"(?<![0-9a-f])s[0-9a-f]{16}(?![0-9a-f])")


def _norm_reason(why: str) -> str:
    """First sentence of a reason with session ids replaced by <sid>, for grouping."""
    first = re.split(r"(?<=\w)\. (?=[A-Z])", why, maxsplit=1)[0]
    return _SID_RE.sub("<sid>", first)


def _reason_table(L: list[str], rows: list[dict], which: str) -> None:
    groups: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        for k, why in r[which].items():
            groups.setdefault((k, _norm_reason(why)), []).append(r["session_id"])
    L.append("| field | sessions | reason |")
    L.append("|---|---|---|")
    for (k, why), sids in sorted(groups.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        who = f"{len(sids)}" if len(sids) > 6 else f"{len(sids)}: {', '.join(sids)}"
        L.append(f"| {k} | {who} | {why} |")
    L.append("")


def write_markdown(path: Path, rows: list[dict], summary: list[dict], probs: list[str], meta: dict) -> None:
    L: list[str] = []
    L.append("# T00 session table (generated)")
    L.append("")
    L.append(f"Generated by scripts/t00/session_table.py from the records; do not edit by hand. Schema {SCHEMA}.")
    L.append(f"Inputs: {meta['n_session_dirs']} session dirs under {meta['runs']}/sessions; session_readings.json "
             f"sha256 {meta['readings_sha256']}; DEV_RESULTS.md sha256 {meta['dev_results_sha256']}; manifest.jsonl "
             f"sha256 {meta['manifest_sha256']} ({meta['manifest_rows']} rows).")
    L.append("Dollar amounts are API-equivalent at list price, not spend. " + PRICE_SOURCE + ".")
    L.append("Readouts are accuracies on V_select (dev) in percent, before -> after (change in points); the "
             "jsonl row names each metric. n_changed is quoted, never edit_size. A condition marked * is the "
             "label from session_readings.json (session.json records no condition).")
    L.append("")
    st: dict[str, int] = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    L.append(f"Sessions: {len(rows)} ({', '.join(f'{k} {v}' for k, v in sorted(st.items()))}).")
    tot = sum(r["api_equiv_cost_usd"] for r in rows if r["api_equiv_cost_usd"] is not None)
    lb = sum(r["api_equiv_cost_usd_lower_bound"] for r in rows if r["api_equiv_cost_usd_lower_bound"] is not None)
    L.append(f"Total API-equivalent cost at list price: ${tot:,.2f} over the completed sessions, plus at least "
             f"${lb:,.2f} for the aborted ones (output tokens a lower bound).")
    L.append("")
    L.append("## Sessions")
    L.append("")
    L.append("| session | experiment | rd | arm | condition | model / seed | status | frozen rule | checks | cells | "
             "n_changed | target | old | audit (latest) | clock |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        aud = _f(r["audit_verdict"])
        if r["audit_latest"] and r["audit_latest"]["file"] != "audit.json":
            aud += f" ({r['audit_latest']['verdict']})"
        cond = r["condition"] if r["condition"] is not None else (
            f"{r['condition_label']} *" if r["condition_label"] is not None else "null")
        L.append(f"| {r['session_id']} | {r['experiment']} | {_f(r['round'])} | {r['arm']} | {cond} | "
                 f"{r['model_id']} / {r['model_seed']} | {r['status']} | {_met(r['success_by_frozen_rule'])}"
                 f"{' (stream)' if r['success_scope'] == 'stream' else ''} | {_f(r['checks_used'])}/{_f(r['checks_budget'])} | "
                 f"{_f(r['cells_used'])}/{_f(r['cells_budget'])} | {_f(r['n_changed'])} | "
                 f"{_pp(r['target_before'], r['target_after'])} | {_pp(r['old_before'], r['old_after'])} | {aud} | "
                 f"{_f(r['program_clock'])} |")
    L.append("")
    L.append("## Cost")
    L.append("")
    L.append("Tokens per session. Output includes thinking. Cache writes are split by lifetime as the CLI records "
             "them. Generation is the session's wall time; execution is the final program's run; inference cost "
             "is compared by tensor layout (a final with the base's tensor names, shapes and dtypes runs the "
             "harness's fixed forward at the base's cost).")
    L.append("")
    L.append("| session | input | output (thinking) | cache read | cache write 1 h | cache write 5 min | "
             "API-equiv $ | CLI $ | generation s | API s | program run s | inference |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        t = r["llm_tokens"]
        if not t:
            L.append(f"| {r['session_id']} | null | null | null | null | null | null | null | null | null | null | null |")
            continue
        lbflag = r["cost_detail"].get("output_is_lower_bound")
        out_s = f"{t.get('output', 0):,}{' (lower bound)' if lbflag else ''}"
        if t.get("thinking") is not None:
            out_s += f" ({t['thinking']:,})"
        usd = _f(r["api_equiv_cost_usd"]) if r["api_equiv_cost_usd"] is not None else \
            f">= {_f(r['api_equiv_cost_usd_lower_bound'])}"
        gen = _f(r["generation_time_s"], 0) if r["generation_time_s"] is not None else \
            f">= {_f(r['generation_time_s_lower_bound'], 0)}"
        inf = r["inference_cost"]
        inf_s = "= base" if inf and inf["equal_to_base"] else ("differs" if inf else "null")
        L.append(f"| {r['session_id']} | {t.get('input', 0):,} | {out_s} | {t.get('cache_read', 0):,} | "
                 f"{t.get('cache_write_1h', 0):,} | {t.get('cache_write_5m', 0):,} | {usd} | "
                 f"{_f(r['cli_total_cost_usd'])} | {gen} | {_f(r['api_time_s'], 0)} | "
                 f"{_f(r['execution_time_s'], 1)} | {inf_s} |")
    L.append("")
    L.append("## Per experiment")
    L.append("")
    L.append("Met / launched counts completed sessions; aborted sessions are stops before launch (PREREG and "
             "DEV_RESULTS) and are listed apart. \"No reading\": a criterion applies but DEV_RESULTS gives no "
             "reading. \"No criterion\": none applies to the session (a gate read as not applicable, a lineage "
             "session that carries no criterion). A stream-scope reading counts once per round session. Medians "
             "and ranges are over completed sessions.")
    L.append("")
    L.append("| experiment | sessions | met / launched | missed | no reading | no criterion | aborted | "
             "output tokens median [min, max] | total tokens median [min, max] | API-equiv $ median [min, max] | "
             "API-equiv $ sum |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary:
        ab = f"{s['n_aborted']}" + (f" (>= ${s['aborted_cost_usd_lower_bound_sum']:.2f})" if s["n_aborted"] else "")
        scope = " (stream)" if s["met_scope"] == ["stream"] else ""
        L.append(f"| {s['experiment']} | {s['n_sessions']} | {s['n_met']} / {s['n_complete']}{scope} | {s['n_missed']} | "
                 f"{s['n_null_no_reading']} | {s['n_null_not_applicable']} | {ab} | {_summ_int(s['output_tokens'])} | "
                 f"{_summ_int(s['total_tokens'])} | "
                 f"{_summ(s['cost_usd'])} | {s['cost_usd_sum']:,.2f} |")
    L.append("")
    L.append("## Problems found (observed in the records)")
    L.append("")
    for p in probs:
        L.append(f"- {p}")
    L.append("")
    L.append("## Fields that could not be determined")
    L.append("")
    L.append("The quantity exists for the session but the records do not give it. Reasons are grouped by field "
             "and by the first sentence of the reason, with session ids written as <sid>; the jsonl row holds "
             "each full reason in its \"missing\" dict.")
    L.append("")
    _reason_table(L, rows, "missing")
    L.append("## Fields that do not apply")
    L.append("")
    L.append("Null because the session has no such quantity (a patch arm has no program, a single-round session "
             "has no round, a no-op final has no patch). The jsonl row holds each reason in \"not_applicable\".")
    L.append("")
    _reason_table(L, rows, "not_applicable")
    path.write_text("\n".join(L) + "\n")


# ----------------------------------------------------------------------------------------------- build

def _env_dir(value: Path | None, var: str) -> Path:
    if value is not None:
        p = value
    else:
        v = os.environ.get(var)
        if not v:
            raise T00Error(f"pass the directory or set {var} (scripts/env.sh)")
        p = Path(v)
    p = p.expanduser().resolve()
    if not p.is_dir():
        raise T00Error(f"{p} is not a directory")
    if "sealed" in p.parts:
        raise T00Error(f"{p} lies under a sealed directory; this script never reads sealed material")
    return p


def build(runs: Path, private: Path, readings_path: Path, dev_results: Path) -> tuple[list[dict], list[dict], list[str], dict]:
    sessions_root = runs / "sessions"
    sids = sorted(p.name for p in sessions_root.iterdir() if p.is_dir())
    readings_doc = _load_json(readings_path)
    readings = readings_doc["sessions"]
    extra = sorted(set(readings) - set(sids))
    lacking = sorted(set(sids) - set(readings))
    if extra or lacking:
        raise T00Error(f"session_readings.json does not match the session dirs: extra {extra}, lacking {lacking}")
    for sid, r in readings.items():
        if r.get("met") is None and not r.get("reason"):
            raise T00Error(f"{sid}: met is null without a reason")
        if r.get("met") is None and r.get("null_kind") not in ("no_reading", "not_applicable"):
            raise T00Error(f"{sid}: met is null without null_kind no_reading or not_applicable")
        if r.get("met") is not None and "null_kind" in r:
            raise T00Error(f"{sid}: null_kind is given but met is not null")
    heads, lines = dev_results_index(dev_results)
    reading_problems = []
    for sid, r in readings.items():
        h = r.get("dev_results_heading")
        if h is None:
            continue
        if h not in heads:
            raise T00Error(f"{sid}: heading not found in {dev_results.name}: {h}")
        r["dev_results_line"] = heads[h]
        named = r.get("session_named_under") or h
        if named not in heads:
            raise T00Error(f"{sid}: heading not found in {dev_results.name}: {named}")
        if sid not in _section(lines, heads[named]):
            reading_problems.append(f"{sid}: its DEV_RESULTS heading section does not name the session id "
                                    f"({h[:60]}...)")
    manifest_path = runs / "manifest.jsonl"
    man_rows = _jsonl(manifest_path)
    manifest: dict[str, dict] = {}
    for m in man_rows:
        if m["session_id"] in manifest:
            raise T00Error(f"manifest.jsonl lists {m['session_id']} twice")
        manifest[m["session_id"]] = m
    index = _load_json(private / "instances" / "index.json")
    index_ids = {r["instance_id"] for r in index["instances"]}
    superseded_ids = {(r.get("instance_id") if isinstance(r, dict) else r) for r in index.get("superseded") or []}
    ctx = {"readings": readings, "streams": index_streams(runs), "fscil": index_fscil(runs), "manifest": manifest,
           "index_ids": index_ids, "superseded_ids": superseded_ids, "reading_problems": reading_problems,
           "not_a_record_audits": [s for s in sids if (sessions_root / s / NOT_A_RECORD_AUDIT).is_file()]}
    rows = [session_row(sid, runs, ctx) for sid in sids]
    for r in rows:
        line = readings[r["session_id"]].get("dev_results_line")
        r["dev_results_line"] = line
        if line is None:
            r["missing"]["dev_results_line"] = "no DEV_RESULTS entry"
    rows.sort(key=lambda r: (r["created_at"] or "", r["session_id"]))
    summary = experiment_summary(rows)
    probs = problems(rows, ctx)
    meta = {"runs": str(runs), "n_session_dirs": len(sids), "readings_sha256": sha256_file(readings_path),
            "dev_results_sha256": sha256_file(dev_results), "manifest_sha256": sha256_file(manifest_path),
            "manifest_rows": len(man_rows)}
    return rows, summary, probs, meta


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, help="runs root (default $WEIGHTCODE_RUNS)")
    parser.add_argument("--private", type=Path, help="private root (default $WEIGHTCODE_PRIVATE)")
    parser.add_argument("--readings", type=Path, default=REPO / "scripts" / "t00" / "session_readings.json")
    parser.add_argument("--dev-results", type=Path, default=REPO / "docs" / "DEV_RESULTS.md")
    parser.add_argument("--out-dir", type=Path, help="output directory (default <runs>/t00)")
    args = parser.parse_args(argv)
    runs = _env_dir(args.runs, "WEIGHTCODE_RUNS")
    private = _env_dir(args.private, "WEIGHTCODE_PRIVATE")
    out_dir = (args.out_dir or runs / "t00").expanduser().resolve()
    rows, summary, probs, meta = build(runs, private, args.readings, args.dev_results)
    out_dir.mkdir(parents=True, exist_ok=True)
    jl = out_dir / "session_table.jsonl"
    with jl.open("w") as f:
        for r in rows:
            f.write(json.dumps({"schema": SCHEMA, **r}, sort_keys=True) + "\n")
    md = out_dir / "session_table.md"
    write_markdown(md, rows, summary, probs, meta)
    print(f"wrote {jl} ({len(rows)} rows, sha256 {sha256_file(jl)})")
    print(f"wrote {md} (sha256 {sha256_file(md)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
