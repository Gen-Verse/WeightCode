"""Write the sealed-read list v6 (PREREG "T04", sealed test section): the T04 networks of record on in1k_t04.

Per condition (gaussian_noise_s3 on i4c8c63bd, contrast_s4 on i62cbf956; seed 100) the roles are:

- PREREG-required (the sealed test's primary readout names them): the base; both LEARN finals; the main control of
  record (the arm-3 kept pick with the highest V_select joint over all grids); bn_adapt and the large-sample diagnostic
  (arm 6) at their kept picks of record.
- Proposed (the user decides): the bar's source (the arm-4 or arm-5 kept pick with the highest V_select joint, over
  all grids) and every other arm-3 kept pick of record, so that the claim "every LEARN final is above every
  fine-tuning pick at matched information" can be read at test level.

A reference's kept pick of record is its best V_select joint over its grids' kept picks (PREREG T04 edge-extension
amendment); on a tie the run before the extension is named, as scripts/t04/ref_table.py names it (a repeated
configuration is the same network). The kept rows, arms, the bar and the main control are computed by ref_table.py's
own functions from the kept readouts of record, so this list and DEV_RESULTS "T04 references" read the same records.

Each distinct (instance, child) is one entry; a role whose child is already an entry maps to it (a kept pick that is
the base maps to the condition's base entry, named T04_<condition>_base). The LEARN finals are the ``finals`` (source
kind session); every other entry is a ``references`` entry (source kind reference, pick kept at S_old >= -20) paired
with repeat 1's final on its instance (pairing is nominal: every entry of a condition is read on the same rows).

Outputs: the list (status draft, approval null) and the role map (every role, required or proposed, its source, its
record numbers and the entry that reads it). Reads JSON records only (the T04 store's kept readouts and run.json files,
and the sessions' run.json and info.json); opens nothing under $WEIGHTCODE_SEALED.

    .venv/bin/python scripts/sealed/make_finals_v6.py --store T04_STORE \
        --sessions R37I1/store/runs R37I2/store/runs R37C1/store/runs R37C2/store/runs \
        --out scripts/sealed/finals_v6.json --roles-out scripts/sealed/finals_v6_roles.json [--required-only]

T04_STORE is the T04 store root (its runs/baselines and runs/devreadouts/t04); each --sessions directory holds
sessions/<SID>/run.json and devreadouts/learn/<SID>/info.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weightcode.evalkit import devreadout
from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit.sealed import manifest_sha256
from weightcode.common.paths import relocate_run_path

REPO = Path(__file__).resolve().parents[2]
FLOOR = -20
CONDITIONS = {  # condition -> (instance, LEARN repeat 1, LEARN repeat 2) of record (DEV_RESULTS "T04 LEARN")
    "gaussian_noise_s3": ("i4c8c63bd", "sece3a52c03e092dd", "sec0ee969740e55ce"),
    "contrast_s4": ("i62cbf956", "sf37fe1409ceb446a", "sd5cdd5dd4e51c1b3"),
}
SHORT = {"gaussian_noise_s3": "gaussian", "contrast_s4": "contrast"}
REQUIRED, PROPOSED = "PREREG-required", "proposed"


class ListError(RuntimeError):
    pass


def _ref_table() -> Any:
    spec = importlib.util.spec_from_file_location("t04_ref_table", REPO / "scripts" / "t04" / "ref_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(row: Mapping[str, Any], store: Path, condition: str) -> dict[str, Any]:
    """The role fields of one kept row of ref_table.load: run, pick and its child, dev record, record numbers."""
    run = row["run"]
    rel_run = f"baselines/{run['instance']['instance_id']}/{row['method']}/{row['label']}/0/run.json"
    run_path = store / "runs" / rel_run
    if _json(run_path) != run:
        raise ListError(f"{rel_run}: not the run ref_table read")
    picked = devreadout.retention_final(run, FLOOR)
    if picked["selection"]["patch_id"] != row["kept"]:
        raise ListError(f"{rel_run}: retention_final picks {picked['selection']['patch_id']}, the kept readout "
                        f"{row['kept']}")
    final = picked["final"]
    base = run["instance"]["params_sha256"]
    child = base if final["is_base_noop"] else final["child_sha256"]
    dev_record = f"devreadouts/t04/{condition}/{row['file']}"
    lines = [json.loads(x) for x in (store / "runs" / dev_record).read_text().splitlines() if x.strip()]
    mine = [x for x in lines if relocate_run_path(x["run"], store) == run_path.resolve()]
    if len(mine) != 1:
        raise ListError(f"{dev_record}: {len(mine)} lines of {rel_run}")
    gain, clean, joint = row["points"]
    return {"method": row["method"], "label": row["label"], "arm": row["arm"], "pick": row["kept"] or "base",
            "config": row["kept_config"], "run": rel_run, "dev_record": dev_record,
            "dev_record_sha256": _sha(store / "runs" / dev_record), "child_sha256": child,
            "is_base": child == base, "record": {"gain": gain, "clean": clean, "joint": joint,
                                                 "vcheck_s_new": row["s_new"], "vcheck_s_old": row["s_old"]}}


def _session(sid: str, instance: str, session_roots: Sequence[Path]) -> dict[str, Any]:
    hits = [root for root in session_roots if (root / "sessions" / sid / "run.json").is_file()]
    if len(hits) != 1:
        raise ListError(f"session {sid}: found in {len(hits)} of the --sessions roots")
    root = hits[0]
    run = _json(root / "sessions" / sid / "run.json")
    info_path = root / "devreadouts" / "learn" / sid / "info.json"
    info = _json(info_path)
    child = run["final"]["child_sha256"]
    if run["instance"]["instance_id"] != instance or info["instance_id"] != instance:
        raise ListError(f"session {sid} is not on {instance}")
    if info["final"]["child_sha256"] != child or run["final"].get("is_base_noop"):
        raise ListError(f"session {sid}: run.json and info.json disagree on the final, or it is the base")
    gain = SF.points(info["final"]["vselect"]["delta_acc"])
    clean = SF.points(info["final"]["vselect_clean"]["delta_acc"])
    return {"session_id": sid, "child_sha256": child, "info": f"devreadouts/learn/{sid}/info.json",
            "info_sha256": _sha(info_path), "patch_id": run["final"].get("patch_id"),
            "record": {"gain": gain, "clean": clean, "joint": gain + clean}}


def _best_by_method(rows: Sequence[Mapping[str, Any]], arm: str, ref_table: Any) -> dict[str, dict[str, Any]]:
    methods = sorted({x["method"] for x in rows if x["arm"] == arm})
    return {m: ref_table.best([x for x in rows if x["method"] == m and x["arm"] == arm], (arm,)) for m in methods}


def condition_roles(condition: str, store: Path, session_roots: Sequence[Path], ref_table: Any
                    ) -> list[dict[str, Any]]:
    """Every role of one condition, in list order: LEARN r1, r2; the required references; the proposed ones."""
    instance, sid1, sid2 = CONDITIONS[condition]
    rows = ref_table.load(store, condition)
    roles: list[dict[str, Any]] = []
    for repeat, sid in ((1, sid1), (2, sid2)):
        roles.append({"role": f"LEARN final, repeat {repeat}", "key": f"LEARN_r{repeat}", "required": REQUIRED,
                      "block": SF.BLOCK_FINALS, **_session(sid, instance, session_roots)})
    main = ref_table.best(rows, ("3",))
    bar = ref_table.best(rows, ("4", "5"))
    (large,) = [x for x in rows if x["method"] == "ft_large"]
    bn = ref_table.best([x for x in rows if x["method"] == "bn_adapt"], ("4",))
    required = [("main control of record (arm 3)", f"main_{main['method']}", main),
                ("large-sample diagnostic at its kept pick (arm 6)", "large_sample", large),
                ("bn_adapt at its kept pick of record (arm 4)", "bn_adapt", bn)]
    for text, key, row in required:
        roles.append({"role": text, "key": key, "required": REQUIRED, "block": SF.BLOCK_REFERENCES,
                      **_record(row, store, condition)})
    roles.append({"role": "bar source: best kept pick of arms 4 and 5 over all grids", "key": f"bar_{bar['method']}",
                  "required": PROPOSED, "block": SF.BLOCK_REFERENCES, **_record(bar, store, condition)})
    for method, row in sorted(_best_by_method(rows, "3", ref_table).items()):
        if row is main:
            continue
        roles.append({"role": f"arm-3 kept pick of record ({method})", "key": f"arm3_{method}", "required": PROPOSED,
                      "block": SF.BLOCK_REFERENCES, **_record(row, store, condition)})
    base = {r["child_sha256"] for r in roles if r.get("is_base")}
    if len(base) > 1:
        raise ListError(f"{condition}: kept picks name {len(base)} different base hashes")
    return roles


def build(store: Path, session_roots: Sequence[Path], required_only: bool
          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ref_table = _ref_table()
    finals: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    role_map: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        instance = CONDITIONS[condition][0]
        tag = SHORT[condition]
        roles = condition_roles(condition, store, session_roots, ref_table)
        by_child: dict[str, str] = {}
        pair = f"T04_{tag}_LEARN_r1"
        for role in roles:
            entry_name = by_child.get(role["child_sha256"])
            created = False
            in_list = not (required_only and role["required"] == PROPOSED)
            if entry_name is None and in_list:
                created = True
                if role["block"] == SF.BLOCK_FINALS:
                    entry_name = f"T04_{tag}_{role['key']}"
                    finals.append({"name": entry_name, "readout": SF.READOUT_VISION_T04,
                                   "source": {"kind": SF.SOURCE_SESSION, "session_id": role["session_id"]},
                                   "instance_id": instance, "child_sha256": role["child_sha256"]})
                else:
                    entry_name = f"T04_{tag}_base" if role["is_base"] else f"T04_{tag}_{role['key']}"
                    references.append({"name": entry_name, "readout": SF.READOUT_VISION_T04,
                                       "source": {"kind": SF.SOURCE_REFERENCE, "run": role["run"],
                                                  "pick": SF.PICK_KEPT, "s_old_floor": FLOOR,
                                                  "dev_record": role["dev_record"]},
                                       "instance_id": instance, "child_sha256": role["child_sha256"],
                                       "pairs_with": pair})
                by_child[role["child_sha256"]] = entry_name
            role_map.append({"condition": condition, "instance_id": instance,
                             **{k: v for k, v in role.items() if k not in ("block",)},
                             "entry": entry_name, "entry_source_is_this_role": created})
    return finals, references, role_map


def _fmt(x: float) -> str:
    return f"{x:+.3f}"


def note_text(finals: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]],
              role_map: Sequence[Mapping[str, Any]], required_only: bool) -> str:
    n = len(finals) + len(references)
    shared = [r for r in role_map if r["entry"] is not None and not r["entry_source_is_this_role"]]
    shared_text = "; ".join(f"{r['condition']} {r['role']} ({r['method']}/{r['label']} {r['pick']}) -> {r['entry']}"
                            for r in shared)
    proposed = sorted({r["entry"] for r in role_map if r["required"] == PROPOSED and r["entry"]}
                      - {r["entry"] for r in role_map if r["required"] == REQUIRED})
    return (
        "List v6 of the sealed read (PREREG \"T04\", sealed test section, main 475a899; the edge-extension amendment, "
        "main 9f0baa9), drafted by the t04-sealed subagent of WEIGHTCODE-main from the T04 records (scripts/sealed/"
        "make_finals_v6.py; role map scripts/sealed/finals_v6_roles.json). Native ImageNet ResNet-18 (in1k_t04), seed "
        "100: gaussian_noise_s3 on i4c8c63bd and contrast_s4 on i62cbf956. PREREG-required entries: the base, both "
        "LEARN finals, the main control of record (gaussian ft_bn f1 q5; contrast grad_full t04g3 q0), bn_adapt and "
        "the large-sample diagnostic at their kept picks of record. "
        + ("Proposed entries, for the user to keep or drop (the paper states that the finals are above every "
           f"fine-tuning pick at matched information): {', '.join(proposed)}. " if not required_only else
           "Proposed entries (the bar's source and the other arm-3 picks) are left out of this variant. ")
        + "A kept pick that is the base is read once, as the condition's base entry: " + shared_text + ". "
        "References pair with repeat 1's final on their instance; the pairing is nominal, every entry of a condition is "
        "read on the same rows. "
        f"{n} entries ({len(finals)} finals, {len(references)} references). Each entry is dev-checked on NSCC (R37 jobs "
        "51 to 54: the child rebuilt from its session patch or reference run, its V_select and clean V_select readouts "
        "equal to the records field for field) before the read. Read once, locally, after the user approves the list: "
        "clean and the condition on the 38,000 test rows (row % 25 not in 0..5) from the 14 validation shards at "
        "revision 49e2ee26, after the fidelity check; the secondary readout (severities 1 to 5 of the target "
        "corruption and the other corruption at its chosen severity on the 20,000 rows with row % 25 in 6..15) for the "
        "base, the LEARN finals and the main control only."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True, help="the T04 store root")
    parser.add_argument("--sessions", type=Path, nargs="+", required=True,
                        help="runs/ directories holding sessions/<SID> and devreadouts/learn/<SID>")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roles-out", type=Path, required=True)
    parser.add_argument("--required-only", action="store_true", help="leave the proposed entries out")
    args = parser.parse_args(argv)
    finals, references, role_map = build(args.store, args.sessions, args.required_only)
    out = {"schema": SF.LIST_SCHEMA, "status": "draft", "approval": None,
           "note": note_text(finals, references, role_map, args.required_only), "finals": finals,
           "references": references}
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    parsed, entries = SF.load_list(args.out)  # the list validates as the pipeline reads it
    sha = manifest_sha256(parsed)
    roles = {"schema": "weightcode.t04_v6_roles.v1", "list": args.out.name, "list_sha256": sha,
             "required_only": args.required_only, "s_old_floor": FLOOR, "roles": role_map}
    args.roles_out.write_text(json.dumps(roles, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(entries)} entries ({len(finals)} finals, {len(references)} references); canonical sha256 "
          f"{sha}")
    for r in role_map:
        rec = r["record"]
        print(f"  {r['condition']:<17} {r['required']:<15} {r['role'][:48]:<48} -> {str(r['entry']):<36} "
              f"{r['child_sha256'][:12]} {_fmt(rec['gain'])} {_fmt(rec['clean'])} {_fmt(rec['joint'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
