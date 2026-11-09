"""Export the frozen networks of a sealed list to params files (R37 jobs 51 to 54; PREREG T04 and T05).

    python scripts/t04/export_frozen.py --list scripts/sealed/finals_v6.json --only NAME ... \
        --out-dir $WEIGHTCODE_RUNS/frozen/v6 --record $WEIGHTCODE_RUNS/frozen/v6/export_<tag>.json

For each named entry the child is recovered exactly as the sealed pipeline recovers it (``weightcode.evalkit.sealed_finals.
recover_child``: a session's final/patch.json, or a reference run's kept pick, applied to the instance's base under
the limits the run applied, and refused unless its params_sha256 is the list's child_sha256), saved as
``<out-dir>/<name>.npz`` (``weightcode.common.identity.save_params``, which reloads the file and checks its hash), and
recorded with its params_sha256 and file sha256. These files are what T05's readout (``scripts/t05/t05_readout.py``)
and the local sealed read of in1k_t04 (``weightcode.evalkit.sealed_t04``) load; both refuse a file whose params_sha256 is
not the frozen one. Nothing under $WEIGHTCODE_SEALED is opened. An existing file or record is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit.sealed import manifest_sha256
from weightcode.evalkit.targets import file_sha256, utc_now

SCHEMA = "weightcode.t04.frozen_export.v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--only", nargs="+", required=True, metavar="NAME")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.record.exists():
        raise SystemExit(f"{args.record} exists; nothing written")
    parsed, entries = SF.load_list(args.list)
    by_name = {entry.name: entry for entry in entries}
    unknown = sorted(set(args.only) - set(by_name))
    if unknown:
        raise SystemExit(f"--only names entries the list does not have: {unknown}")
    chosen = [by_name[name] for name in args.only]
    targets = [args.out_dir / f"{entry.name}.npz" for entry in chosen]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise SystemExit(f"params files exist: {existing}; nothing written")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state = SF.code_state(modules=SF.list_modules(chosen))
    rows = []
    for entry, path in zip(chosen, targets):
        fam = SF.family(entry.readout)
        instance = fam.load_instance(entry.instance_id)
        child, provenance = SF.recover_child(entry, instance)
        digest = save_params(path, child)
        reloaded = params_sha256(load_params(path))
        if digest != entry.child_sha256 or reloaded != entry.child_sha256:
            raise SystemExit(f"{entry.name}: saved {digest}, reloaded {reloaded}, list {entry.child_sha256}")
        rows.append({"name": entry.name, "block": entry.block, "instance_id": entry.instance_id,
                     "base_params_sha256": instance.params_sha256, "is_base": entry.child_sha256 == instance.params_sha256,
                     "child_sha256": entry.child_sha256, "params_sha256": reloaded, "path": str(path),
                     "file_sha256": file_sha256(path), "bytes": path.stat().st_size, "child": provenance})
        print(f"{entry.name}: {path} params_sha256 {reloaded}", flush=True)
    record = {"schema": SCHEMA, "list": str(args.list), "list_sha256": manifest_sha256(parsed),
              "list_status": parsed["status"], "code": state, "time": utc_now(), "entries": rows}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with open(args.record, "x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, sort_keys=True)
    print(f"record {args.record} ({len(rows)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
