"""Write the sealed-read list v3 from the store's run records (PREREG Sealed read, "List v3" and its "Composition").

The composition (user, 2026-11-06 about 22:20 Beijing, AskUserQuestion "全部，约 56 条"; PREREG main 0119883):

1. the eight T01 and T02 finals, met or missed alike (source kind ``session``);
2. the T08 program-arm children of these eight sources on their five non-own students, 40 entries, every cell whatever
   its dev reading (kind ``transfer``; records $WEIGHTCODE_RUNS/t08/program/<sid>/transfer_<program sha12>_<iid>.json, each
   with its work directory of the same stem). The copy arm is not read; the two B finals of record (sc918e11dceef637a,
   s760cb52ce82fb937) are not sources here, since their children are list v2's E11 C entries;
3. for seeds 101 and 102 and each condition, the reference that sets the B bar and the best kept F1 pick (DEV_RESULTS
   "E11 replication references and F1 on seeds 101 and 102"; bars of record PREREG 92a96af), 8 references, each paired
   with the T01 final on its instance (kind ``reference``, pick ``kept`` at S_old >= -20, the floor of those bars).

Every child hash comes from its record: a session's ``run.json`` final, a transfer's ``child_sha256``, the kept pick of
``devreadout.retention_final`` at S_old >= -20. Each reference's dev record line is checked here against the gain /
clean / joint of the DEV_RESULTS table, so a pick that is not the table's stops the build. The one tie in the table,
i905213a3's interp, is in both grids (e11 q0, alpha 1/8; e11g2 q7, alpha 8/64) with one child sha256 and the same
readouts; the table's bar names the first grid, as e11rep_tables.py ``refs`` keeps the first of equal joints (e11 is
read before e11g2, strict >), and the build checks that both grids give that child and those readouts. No entry may
repeat an (instance, child) of list v1 or v2 (a sealed read refuses a pair the registry has seen).

Usage (reads $WEIGHTCODE_RUNS only; opens nothing under $WEIGHTCODE_SEALED)::

    .venv/bin/python scripts/sealed/make_finals_v3.py --out scripts/sealed/finals_v3.json \
        --earlier scripts/sealed/finals_v1.json scripts/sealed/finals_v2.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weightcode.common.paths import runs_dir
from weightcode.evalkit import devreadout
from weightcode.evalkit import sealed_finals as SF

READOUT = SF.READOUT_VISION_JOINT
FLOOR = -20
TARGETS = {"gaussian": {100: "i00802425", 101: "i905213a3", 102: "iab18c0aa", 103: "ib1b195f6", 104: "ie1be7cd9",
                        105: "i36df1f8f"},
           "contrast": {100: "i3eb3fc78", 101: "i4a486d70", 102: "i5d76f25c", 103: "id83bf155", 104: "ib1e4e794",
                        105: "iecf07b07"}}
CONDITIONS = {"gaussian": "gaussian_noise_s5", "contrast": "contrast_s5"}
FINALS = (
    # name, session, condition, seed, repeat (session.json)
    ("T02_B_gaussian_s100_r1", "s4253319a365b126a", "gaussian", 100, 1),
    ("T02_B_gaussian_s100_r2", "scb92c3eb5d5226cc", "gaussian", 100, 2),
    ("T01_B_gaussian_s101", "scf16f7fb35855c03", "gaussian", 101, 0),
    ("T01_B_gaussian_s102", "sf45990dee41ac233", "gaussian", 102, 0),
    ("T02_B_contrast_s100_r1", "s7ae7d105d7655ed5", "contrast", 100, 1),
    ("T02_B_contrast_s100_r2", "sfeb1cbc77d6a6042", "contrast", 100, 2),
    ("T01_B_contrast_s101", "s3db14093bc6c3538", "contrast", 101, 0),
    ("T01_B_contrast_s102", "s037ead942a4cc8aa", "contrast", 102, 0),
)
T08_PROGRAM = "t08/program"
REFERENCES = (
    # name, condition, seed, method, grid tag, dev record stem, DEV_RESULTS gain / clean / joint (points)
    ("ref_E11_gaussian_s101_bar_interp", "gaussian", 101, "interp", "e11",
     "e11_refs_i905213a3_gaussian_noise_s5_floor20_dev_20261105", (28.325, -1.150, 27.175)),
    ("ref_E11_gaussian_s101_F1_lora_r16", "gaussian", 101, "lora_r16", "f1",
     "f1_i905213a3_gaussian_noise_s5_floor20_dev_nscc_20261106", (16.675, -0.500, 16.175)),
    ("ref_E11_gaussian_s102_bar_interp_e11g2", "gaussian", 102, "interp", "e11g2",
     "e11g2_refs_iab18c0aa_gaussian_noise_s5_floor20_dev_nscc_20261106", (21.675, -0.675, 21.000)),
    ("ref_E11_gaussian_s102_F1_grad_last_e11g2", "gaussian", 102, "grad_last", "e11g2",
     "e11g2_refs_iab18c0aa_gaussian_noise_s5_floor20_dev_nscc_20261106", (11.025, -0.775, 10.250)),
    ("ref_E11_contrast_s101_bar_grad_sparse19_sal_e11g2", "contrast", 101, "grad_sparse19_sal", "e11g2",
     "e11g2_refs_i4a486d70_contrast_s5_floor20_dev_nscc_20261106", (7.225, -0.300, 6.925)),
    ("ref_E11_contrast_s101_F1_ft_full_lwf", "contrast", 101, "ft_full_lwf", "f1",
     "f1_i4a486d70_contrast_s5_floor20_dev_nscc_20261106", (7.375, -0.025, 7.350)),
    ("ref_E11_contrast_s102_bar_grad_full_e11g2", "contrast", 102, "grad_full", "e11g2",
     "e11g2_refs_i5d76f25c_contrast_s5_floor20_dev_nscc_20261106", (10.050, -0.550, 9.500)),
    ("ref_E11_contrast_s102_F1_lora_r16", "contrast", 102, "lora_r16", "f1",
     "f1_i5d76f25c_contrast_s5_floor20_dev_nscc_20261106", (10.350, -0.400, 9.950)),
)
"""The table of DEV_RESULTS "E11 replication references and F1 on seeds 101 and 102": per instance the best kept
reference of both grids (the B bar, PREREG 92a96af) and the best kept F1 pick. F1's full and last-layer methods are
the e11g2 grid's grad_full and grad_last runs (scripts/store/e11rep_tables.py ``f1``), so iab18c0aa's grad_last
is the e11g2 run. Bars of record: i905213a3 +28.175, iab18c0aa +22.000, i4a486d70 +7.925, i5d76f25c +10.500."""
TIE_TWIN = {"ref_E11_gaussian_s101_bar_interp": ("e11g2", "e11g2_refs_i905213a3_gaussian_noise_s5_floor20_dev_20261105")}
"""The table's one tie: the same interp child in the second grid (alpha 8/64 = 1/8), checked to be the same child with
the same readouts and left off the list (one child is read once)."""


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def session_final(name: str, session: str, cond: str, seed: int, repeat: int) -> dict[str, Any]:
    """A T01 or T02 final: the session's recorded final child on its instance, with its dev readout of record."""
    root = runs_dir()
    run = _json(root / "sessions" / session / "run.json")
    spec = _json(root / "sessions" / session / "session.json")
    inst = TARGETS[cond][seed]
    if run["instance"]["instance_id"] != inst or spec.get("repeat") != repeat or spec.get("seed") != seed:
        raise SystemExit(f"{name}: session {session} is on {run['instance']['instance_id']} seed {spec.get('seed')} "
                         f"repeat {spec.get('repeat')}, expected {inst} seed {seed} repeat {repeat}")
    final = run["final"]
    if final.get("is_base_noop") or not final.get("child_sha256"):
        raise SystemExit(f"{name}: session {session}'s final is the base no-op; it has no child to read")
    info = _json(root / "devreadouts" / "learn" / session / "info.json")
    if info["instance_id"] != inst or info["final"]["child_sha256"] != final["child_sha256"]:
        raise SystemExit(f"{name}: the dev readout of {session} is not of its final")
    return {"name": name, "readout": READOUT, "source": {"kind": SF.SOURCE_SESSION, "session_id": session},
            "instance_id": inst, "child_sha256": final["child_sha256"]}


def transfer_children(final_name: str, session: str, cond: str, seed: int) -> list[dict[str, Any]]:
    """The T08 program-arm children of a source on its five non-own students (an ok run the limits admitted)."""
    root = runs_dir()
    directory = root / T08_PROGRAM / session
    program = _json(root / "sessions" / session / "run.json")["final"]
    entries = []
    for target_seed, inst in TARGETS[cond].items():
        if target_seed == seed:
            continue
        hits = sorted(directory.glob(f"transfer_*_{inst}.json"))
        if len(hits) != 1:
            raise SystemExit(f"{final_name}: {len(hits)} T08 program records on {inst} in {directory}")
        path = hits[0]
        record = _json(path)
        sha12 = path.stem.split("_")[1]
        if (record.get("schema") != SF.TRANSFER_SCHEMA or record["target"]["instance_id"] != inst
                or record["target"]["condition"] != CONDITIONS[cond]
                or not record["program"]["sha256"].startswith(sha12)
                or Path(record["program"]["path"]).parent != root / "sessions" / session / "final"
                or Path(record["program"]["path"]).name != program.get("patch_id")):
            raise SystemExit(f"{final_name}: {path} is not the final program of {session} on {inst}")
        if record["status"] != "ok" or record["within_limits"] is not True or record["readout_child"] != "within_limits":
            raise SystemExit(f"{final_name}: {path} has status {record['status']!r}; only an admitted child is read")
        short = final_name.replace("_B_", "_")
        entries.append({"name": f"T08_{short}_on_s{target_seed}", "readout": READOUT,
                        "source": {"kind": SF.SOURCE_TRANSFER, "record": str(path.relative_to(root))},
                        "instance_id": inst, "child_sha256": record["child_sha256"]})
    return entries


def _points(line: Mapping[str, Any]) -> tuple[float, float, float]:
    gain = round(100 * line["readouts"]["delta_acc"], 3)
    clean = round(100 * line["readouts_clean"]["delta_acc"], 3)
    return gain, clean, round(gain + clean, 3)


def kept_reference(name: str, cond: str, seed: int, method: str, tag: str, stem: str,
                   expected: Sequence[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """A kept pick at S_old >= -20 and the line of its dev record, checked against the DEV_RESULTS numbers."""
    root = runs_dir()
    inst = TARGETS[cond][seed]
    rel = f"baselines/{inst}/{method}/{tag}/0/run.json"
    run = _json(root / rel)
    picked = devreadout.retention_final(run, FLOOR)
    pick = picked["final"]
    if pick["is_base_noop"]:
        raise SystemExit(f"{name}: {rel} keeps the base no-op at S_old >= {FLOOR}")
    dev_record = f"devreadouts/{stem}.jsonl"
    lines = [line for line in _jsonl(root / dev_record)
             if Path(str(line.get("run"))).resolve() == (root / rel).resolve()
             and (line.get("selection") or {}).get("patch_id") == pick["patch_id"]
             and (line.get("selection") or {}).get("s_old_floor") == FLOOR and line.get("instance_id") == inst]
    if len(lines) != 1:
        raise SystemExit(f"{name}: {len(lines)} lines of {dev_record} select {pick['patch_id']} of {rel} at {FLOOR}")
    if _points(lines[0]) != tuple(expected):
        raise SystemExit(f"{name}: {dev_record} reads {_points(lines[0])}, DEV_RESULTS says {tuple(expected)}")
    entry = {"name": name, "readout": READOUT,
             "source": {"kind": SF.SOURCE_REFERENCE, "run": rel, "pick": SF.PICK_KEPT, "s_old_floor": FLOOR,
                        "dev_record": dev_record},
             "instance_id": inst, "child_sha256": pick["child_sha256"]}
    return entry, {"patch_id": pick["patch_id"], "line": lines[0]}


def check_twin(name: str, cond: str, seed: int, method: str, entry: Mapping[str, Any], listed: Mapping[str, Any],
               expected: Sequence[float]) -> str:
    """The tie's second-grid pick has the listed child and the same readouts; returns a sentence for the note."""
    tag, stem = TIE_TWIN[name]
    twin, found = kept_reference(f"{name} ({tag})", cond, seed, method, tag, stem, expected)
    if twin["child_sha256"] != entry["child_sha256"]:
        raise SystemExit(f"{name}: the {tag} pick has child {twin['child_sha256']}, not {entry['child_sha256']}; the "
                         "tie is between two children and is not resolved here")
    for part in ("readouts", "readouts_clean"):
        if found["line"][part] != listed["line"][part]:
            raise SystemExit(f"{name}: the {tag} pick's {part} differ from the listed pick's")
    return (f"The table's one tie: {entry['source']['run']} ({listed['patch_id']}) and {twin['source']['run']} "
            f"({found['patch_id']}) keep the same child ({entry['child_sha256'][:16]}...) with the same readouts; it "
            "is listed once, from the first grid, as the table's bar names it")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--earlier", type=Path, nargs="+", required=True,
                        help="earlier lists (v1, v2) whose (instance, child) pairs v3 may not repeat")
    args = parser.parse_args()
    finals: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    for name, session, cond, seed, repeat in FINALS:
        finals.append(session_final(name, session, cond, seed, repeat))
        children.extend(transfer_children(name, session, cond, seed))
    pairs_with = {(cond, seed): name for name, _, cond, seed, repeat in FINALS if name.startswith("T01_")}
    references: list[dict[str, Any]] = []
    tie_notes: list[str] = []
    for name, cond, seed, method, tag, stem, expected in REFERENCES:
        entry, found = kept_reference(name, cond, seed, method, tag, stem, expected)
        if name in TIE_TWIN:
            tie_notes.append(check_twin(name, cond, seed, method, entry, found, expected))
        references.append({**entry, "pairs_with": pairs_with[(cond, seed)]})
    earlier: dict[tuple[str, str], str] = {}
    for path in args.earlier:
        listed = _json(path)
        for row in listed["finals"] + listed.get("references", []):
            earlier[(row["instance_id"], row["child_sha256"])] = f"{path.name} {row['name']}"
    repeats = [f"{row['name']} = {earlier[(row['instance_id'], row['child_sha256'])]}"
               for row in finals + children + references if (row["instance_id"], row["child_sha256"]) in earlier]
    if repeats:
        raise SystemExit(f"entries already on an earlier list: {repeats}")
    all_finals = finals + children
    n = len(all_finals) + len(references)
    note = (
        "List v3 of the sealed read (PREREG Sealed read, \"List v3\", main 0119883: user 2026-11-06 about 17:45 Beijing, "
        "\"单独预注册 v3，读一次\"; composition about 22:20 Beijing, \"全部，约 56 条\", written after T01, T02 and T08 "
        "ended and before this list was built), drafted by the list-v3 subagent of WEIGHTCODE-main from the store's run "
        "records (scripts/sealed/make_finals_v3.py). (1) The eight T01 and T02 finals, met or missed alike: T02 repeats "
        "1 and 2 on the seed-100 students (gaussian s4253319a365b126a, scb92c3eb5d5226cc; contrast s7ae7d105d7655ed5, "
        "sfeb1cbc77d6a6042) and T01 on seeds 101 and 102 (gaussian scf16f7fb35855c03, sf45990dee41ac233; contrast "
        "s3db14093bc6c3538, s037ead942a4cc8aa). (2) The T08 program-arm children of these eight sources on their five "
        "non-own students (NSCC R31, runs/t08/program/), 40 entries, every cell whatever its dev reading; the copy arm "
        "is not read, and the two B finals of record are not sources here (their children are list v2's E11 C "
        "entries). (3) For seeds 101 and 102 and each condition, the kept reference that sets the B bar (PREREG "
        "92a96af) and the best kept F1 pick (DEV_RESULTS \"E11 replication references and F1 on seeds 101 and 102\"), "
        f"joint-retain-v1 at S_old >= {FLOOR}, each paired with the T01 final on its instance: i905213a3 interp and "
        "lora_r16; iab18c0aa interp e11g2 and grad_last e11g2 (F1's last-layer method is the e11g2 grad_last run); "
        "i4a486d70 grad_sparse19_sal e11g2 and ft_full_lwf; i5d76f25c grad_full e11g2 and lora_r16. "
        f"{'; '.join(tie_notes)}. No entry repeats an (instance, child) of list v1 or v2. {n} entries "
        f"({len(finals)} session finals, {len(children)} transfer children, {len(references)} references). Each "
        "entry is dev-checked equal to its readout of record (NSCC R32) before the read. Read once, locally, after the "
        "v2 read ends, only once the user has approved the list and it is committed with status approved."
    )
    out = {"schema": SF.LIST_SCHEMA, "status": SF.STATUS_DRAFT, "approval": None, "note": note,
           "finals": all_finals, "references": references}
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    SF.load_list(args.out)
    print(f"{args.out}: {len(all_finals)} finals ({len(finals)} sessions, {len(children)} transfers), "
          f"{len(references)} references ({n} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
