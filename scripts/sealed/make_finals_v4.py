"""Write the sealed-read list v4 from the T16 stream records (PREREG Sealed read; T16 "List v4", main b5b3083).

The composition (user, 2026-11-06 about 23:15 Beijing, AskUserQuestion "批准 v4", fixed before any T16 result):

- the six T16 streams' W2, the five new sessions (arms L and N) and the session of record sa651245fee587200 (arm N's
  first), met or missed alike, each read on both of the stream's instances (gaussian i00802425, contrast i3eb3fc78):
  12 entries of kind ``stream_final``;
- W1 on i3eb3fc78 (kind ``stream_round``, round 1 of the stream of record), so that the marginal gain from W1 to W2
  has a test number. W1 on i00802425 is list v2's E11_B_gaussian_s100 (the same child on the same instance).

Every stream must be the E11 two-round stream of record: base fd7f6d4b..., instances (i00802425, i3eb3fc78), two
completed rounds, round 1 the gaussian B final's child (e40c2078...), round 2 a contrast_s5 round with status ok. A W2
that equals W1 (a base no-op round 2), or a W2 that another stream also produced, is listed once and the note says
which (a sealed read refuses a pair it has seen). No entry may repeat an (instance, child) of lists v1 to v3.

Usage (reads $WEIGHTCODE_RUNS only; opens nothing under $WEIGHTCODE_SEALED)::

    .venv/bin/python scripts/sealed/make_finals_v4.py --out scripts/sealed/finals_v4.json \
        --stream N1=loop_streams/loop_teacher_92b6e2bd54be_fd7f6d4b8014.json --stream L1=<record> ... \
        --earlier scripts/sealed/finals_v1.json scripts/sealed/finals_v2.json scripts/sealed/finals_v3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from weightcode.common.paths import runs_dir
from weightcode.evalkit import sealed_finals as SF

READOUT = SF.READOUT_VISION_JOINT
BASE = "fd7f6d4b80145c98bc7c6a9cbc2c8e440e3f45e328168df2a8b55bec3f7c5b6a"
W1_CHILD = "e40c20786ae91c904cfd7a195addb7e03b6601b955d731518da1ff54bc01c251"
INSTANCES = {"gaussian": "i00802425", "contrast": "i3eb3fc78"}
RECORD_SESSION = "sa651245fee587200"
LABELS = ("N1", "L1", "N2", "L2", "N3", "L3")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_stream(label: str, rel: str) -> dict[str, Any]:
    """The stream's record, checked to be the E11 two-round stream; returns its round-2 facts."""
    path = runs_dir() / rel
    record = _json(path)
    where = f"{label} ({rel})"
    if record.get("schema") not in SF.STREAM_SCHEMAS or record.get("base_params_sha256") != BASE:
        raise SystemExit(f"{where}: schema {record.get('schema')!r} or base {record.get('base_params_sha256')!r}")
    if [row.get("instance_id") for row in record.get("instances", [])] != list(INSTANCES.values()):
        raise SystemExit(f"{where}: instances {[row.get('instance_id') for row in record.get('instances', [])]}")
    rounds = record.get("rounds", [])
    if len(rounds) != 2 or [r.get("round") for r in rounds] != [1, 2]:
        raise SystemExit(f"{where}: {len(rounds)} rounds; only a completed two-round stream is listed")
    one, two = rounds
    if one.get("child_sha256") != W1_CHILD or (one.get("student_after") or {}).get("sha256") != W1_CHILD:
        raise SystemExit(f"{where}: round 1's student is not W1 ({W1_CHILD[:12]}...)")
    if two.get("condition") != "contrast_s5" or two.get("instance_id") != INSTANCES["contrast"]:
        raise SystemExit(f"{where}: round 2 is {two.get('condition')} on {two.get('instance_id')}")
    if two.get("status") != "ok" or two.get("parent_sha256") != W1_CHILD:
        raise SystemExit(f"{where}: round 2 status {two.get('status')!r}, parent {two.get('parent_sha256')!r}")
    child = two.get("child_sha256")
    if not child or (two.get("student_after") or {}).get("sha256") != child:
        raise SystemExit(f"{where}: round 2's student_after does not match its child")
    sessions = [str(s.get("session_id") if isinstance(s, dict) else s) for s in two.get("sessions", [])]
    return {"label": label, "record": rel, "w2": child, "sessions": sessions}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stream", action="append", required=True, help="LABEL=record path relative to $WEIGHTCODE_RUNS")
    parser.add_argument("--earlier", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    streams = dict(s.split("=", 1) for s in args.stream)
    if sorted(streams) != sorted(LABELS):
        raise SystemExit(f"streams {sorted(streams)}; the composition is {list(LABELS)}")
    facts = [check_stream(label, streams[label]) for label in LABELS]
    if RECORD_SESSION not in facts[0]["sessions"]:
        raise SystemExit(f"N1 is not the stream of record: sessions {facts[0]['sessions']}")
    earlier: dict[tuple[str, str], str] = {}
    for path in args.earlier:
        listed = _json(path)
        for row in listed["finals"] + listed.get("references", []):
            earlier[(row["instance_id"], row["child_sha256"])] = f"{path.name} {row['name']}"
    finals: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], str] = {}
    notes: list[str] = []

    def add(name: str, source: dict[str, Any], inst: str, child: str) -> None:
        key = (inst, child)
        if key in earlier:
            notes.append(f"{name} is not listed: its child is {earlier[key]}")
            return
        if key in seen:
            notes.append(f"{name} is not listed: its child equals {seen[key]}")
            return
        seen[key] = name
        finals.append({"name": name, "readout": READOUT, "source": source, "instance_id": inst, "child_sha256": child})

    add("E11_stream_W1_contrast", {"kind": SF.SOURCE_STREAM_ROUND, "record": facts[0]["record"], "round": 1},
        INSTANCES["contrast"], W1_CHILD)
    for f in facts:
        for cond, inst in INSTANCES.items():
            add(f"E11_stream_T16_{f['label']}_W2_{cond}", {"kind": SF.SOURCE_STREAM_FINAL, "record": f["record"]},
                inst, f["w2"])
    streams_text = "; ".join(f"{f['label']} {', '.join(f['sessions'])} ({f['record']}, W2 {f['w2'][:12]}...)"
                             for f in facts)
    note = (
        "List v4 of the sealed read (PREREG T16 \"List v4\", main b5b3083: user 2026-11-06 about 23:15 Beijing, "
        "AskUserQuestion \"批准 v4\", composition fixed before any T16 result), built by WEIGHTCODE-main from the stream "
        "records (scripts/sealed/make_finals_v4.py). The six T16 streams of the E11 two-round stream (base "
        "fd7f6d4b..., W1 = the gaussian B final's child e40c2078...; round 2 one contrast_s5 LEARN session), met or "
        f"missed alike: {streams_text}. Arm N is the protocol of record (N1 is the session of record), arm L adds the "
        "edit log. Each stream's W2 is read on both instances (gaussian i00802425, contrast i3eb3fc78), and W1 on "
        "i3eb3fc78 (W1 on i00802425 is list v2's E11_B_gaussian_s100). "
        + (" ".join(n + "." for n in notes) + " " if notes else "")
        + f"{len(finals)} entries. Each entry is dev-checked equal to its record on NSCC before the read. Read once, "
        "locally, after the v3 read ends; the follow-up's sessions are not on this list."
    )
    out = {"schema": SF.LIST_SCHEMA, "status": SF.STATUS_DRAFT, "approval": None, "note": note, "finals": finals,
           "references": []}
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    SF.load_list(args.out)
    print(f"{args.out}: {len(finals)} entries; {len(notes)} not listed")
    for n in notes:
        print(" ", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
