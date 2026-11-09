"""Write the freeze record of the networks T05 reads (PREREG "T05"): scripts/t05/frozen_t04.json.

    .venv/bin/python scripts/t05/make_frozen_t04.py --list scripts/sealed/finals_v6.json \
        --roles scripts/sealed/finals_v6_roles.json --exports EXPORT.json ... --out scripts/t05/frozen_t04.json

PREREG T05 reads, per T04 condition, the base, each LEARN final, the main control of record, bn_adapt's kept pick and
the large-sample diagnostic's kept pick. Each role is taken from the v6 role map (scripts/sealed/make_finals_v6.py)
and resolved to the list entry that reads it; each entry to the params file R37 jobs 51 to 54 exported for it
(scripts/t04/export_frozen.py, after the entry's dev check passed: the child rebuilt from its session patch or
reference run and hash-checked). A kept pick that is the base is the base network (the two conditions share one
base). Each distinct network is listed once under ``networks`` (name, path inside the NSCC job, params_sha256, file
sha256), the form scripts/t04/submit_t04.sh's t05 preset passes to scripts/t05/t05_readout.py; ``roles`` maps every
T05 role to its network. The record is written once (never overwritten) and committed before job 50.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit.sealed import manifest_sha256

SCHEMA = "weightcode.t05.frozen_t04.v1"
T05_KEYS = ("LEARN_r1", "LEARN_r2", "main_", "bn_adapt", "large_sample")
"""PREREG T05's networks per condition besides the base: role keys of the v6 role map (main_ is a prefix)."""


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--exports", type=Path, nargs="+", required=True, help="export records of jobs 51 to 54")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; nothing written")
    parsed, entries = SF.load_list(args.list)
    list_sha = manifest_sha256(parsed)
    roles = json.loads(args.roles.read_text(encoding="utf-8"))
    if roles["list_sha256"] != list_sha:
        raise SystemExit(f"{args.roles} is the role map of list {roles['list_sha256']}, not {list_sha}")
    by_name = {entry.name: entry for entry in entries}
    exported: dict[str, dict] = {}
    export_records = []
    for path in args.exports:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != "weightcode.t04.frozen_export.v1" or record.get("list_sha256") != list_sha:
            raise SystemExit(f"{path}: not an export record of list {list_sha}")
        export_records.append({"file": path.name, "sha256": _sha(path), "code_commit": record["code"]["commit"],
                               "dirty": record["code"]["dirty"], "n": len(record["entries"])})
        for row in record["entries"]:
            if row["name"] in exported:
                raise SystemExit(f"{row['name']} is exported twice")
            entry = by_name[row["name"]]
            if row["params_sha256"] != entry.child_sha256 or row["instance_id"] != entry.instance_id:
                raise SystemExit(f"{row['name']}: export {row['params_sha256']} differs from the list")
            exported[row["name"]] = row
    networks: dict[str, dict] = {}
    t05_roles = []
    for role in roles["roles"]:
        if not role["key"].startswith(T05_KEYS):
            continue
        entry_name = role["entry"]
        row = exported.get(entry_name)
        if row is None:
            raise SystemExit(f"{role['condition']} {role['role']}: entry {entry_name} has no export")
        name = "base" if row["is_base"] else entry_name.removeprefix("T04_")
        network = {"name": name, "path": row["path"], "params_sha256": row["params_sha256"],
                   "file_sha256": row["file_sha256"], "instance_id": row["instance_id"], "entry": entry_name}
        if name in networks and networks[name]["params_sha256"] != network["params_sha256"]:
            raise SystemExit(f"{name}: two different networks")
        networks.setdefault(name, network)
        t05_roles.append({"condition": role["condition"], "role": role["role"], "required": role["required"],
                          "network": name, "child_sha256": role["child_sha256"]})
    base_rows = [row for row in exported.values() if row["is_base"]]
    if not base_rows or len({row["params_sha256"] for row in base_rows}) != 1:
        raise SystemExit("the exports do not name one base")
    base = base_rows[0]
    networks.setdefault("base", {"name": "base", "path": base["path"], "params_sha256": base["params_sha256"],
                                 "file_sha256": base["file_sha256"], "instance_id": base["instance_id"],
                                 "entry": base["name"]})
    ordered = [networks["base"]] + [n for k, n in networks.items() if k != "base"]
    record = {"schema": SCHEMA, "prereg": "T05 (retention on ImageNetV2)",
              "list": {"path": str(args.list), "sha256": list_sha, "status": parsed["status"]},
              "roles_file": {"path": str(args.roles), "sha256": _sha(args.roles)},
              "export_records": export_records, "networks": ordered, "roles": t05_roles,
              "note": "Frozen before ImageNetV2 is decoded. Each network is the params file exported for its list "
                      "entry after the entry's dev check (R37 jobs 51 to 54); a kept pick that is the base reads "
                      "as the base. ImageNetV2 chooses nothing."}
    with open(args.out, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=1) + "\n")
    print(f"{args.out}: {len(ordered)} networks, {len(t05_roles)} roles")
    for network in ordered:
        print(f"  {network['name']:<26} {network['params_sha256'][:16]} {network['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
