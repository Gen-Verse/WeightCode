"""Write the sealed-read list v5 (PREREG "T07 sealed read (list v5)"): T07's table at test level.

Composition: T07's eight networks (per architecture the B finals of record and F1's kept picks at floor -20, as T07
names them ``llm_<c>`` and ``ft_<c>``) on T07's 26 sets (``t07_table.json`` ``sets``: the clean rows and 25
corruption sets; glass_blur and gaussian_blur stay out as in T07). The test cell of (network, set) is the network's
child read against its base on the official CIFAR-10-C test rows of that set (10,000 per set) or on the clean test.

A cell is NOT read again when an earlier sealed list (v2 to v4) already read the same child on the same dataset's
rows of that set: the network's own condition and the clean test (its own vision_joint entry of list v2, fields
``readouts`` and ``readouts_clean``), and any other vision_joint entry of v2 to v4 whose instance's condition is the
set (for example list v4's E11_stream_W1_contrast, the ViT gaussian final's child on i3eb3fc78). Every other cell is
one ``vision_range`` entry (source kind ``range``) whose network is the v2 entry's source copied verbatim.

Outputs: the list (status draft, approval null) and the reuse map, one row per cell of T07's table saying where its
test cell comes from. Reads $WEIGHTCODE_RUNS (the T07 record) and the instances' private.json; opens nothing under
$WEIGHTCODE_SEALED.

    .venv/bin/python scripts/sealed/make_finals_v5.py --out scripts/sealed/finals_v5.json \
        --reuse-out scripts/sealed/finals_v5_reused.json --network-list scripts/sealed/finals_v2.json \
        --earlier scripts/sealed/finals_v2.json scripts/sealed/finals_v3.json scripts/sealed/finals_v4.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weightcode.common.paths import runs_dir
from weightcode.evalkit import sealed_finals as SF

T07_RECORD = "t07/r38_20261107_124729_t07_r38_main_e4e638836a15/t07_table.json"
T07_RECORD_SHA256 = "55982f0da04b35140b494022768e64e4f0aa2115f7aff4084fbd2453265fa556"
CLEAN = "clean"
NETWORKS = {  # (T07 arch, T07 network) -> (block, list v2 entry) of record
    ("vit", "llm_gaussian_noise"): ("finals", "E11_B_gaussian_s100"),
    ("vit", "llm_contrast"): ("finals", "E11_B_contrast_s100"),
    ("vit", "ft_gaussian_noise"): ("references", "ref_E11_gaussian_F1_lora_r4"),
    ("vit", "ft_contrast"): ("references", "ref_E11_contrast_F1_ft_full_l2sp"),
    ("r18", "llm_gaussian_noise"): ("finals", "E12_B_gaussian_s100"),
    ("r18", "llm_contrast"): ("finals", "E12_B_contrast_s100"),
    ("r18", "ft_gaussian_noise"): ("references", "ref_E12_gaussian_F1_lora_r16"),
    ("r18", "ft_contrast"): ("references", "ref_E12_contrast_F1_lora_r16"),
}
DATASETS = {"vit": "cifar10_vl7", "r18": "cifar10_r18"}


class ListError(RuntimeError):
    pass


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def t07_cells(record: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """(arch, network, set) of every row of T07's table, in its order; every network has every set once."""
    cells = [(r["arch"], r["network"], r["set"]) for r in record["rows"]]
    if len(set(cells)) != len(cells) or sorted({(a, n) for a, n, _ in cells}) != sorted(NETWORKS):
        raise ListError("T07's rows are not one row per (network, set) of the eight networks")
    for arch, network in NETWORKS:
        if [s for a, n, s in cells if (a, n) == (arch, network)] != list(record["sets"]):
            raise ListError(f"{arch} {network}: T07's rows do not cover its sets in order")
    return cells


def earlier_reads(lists: Sequence[Path]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """(child, dataset, condition) -> the first vision_joint entry of the earlier lists that read it."""
    reads: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in lists:
        for row in (lambda p: p["finals"] + p.get("references", []))(_json(path)):
            if row["readout"] != SF.READOUT_VISION_JOINT:
                continue
            rows = SF._vision_rows(row["instance_id"])  # noqa: SLF001, the registry's own (dataset, condition)
            if rows is None:
                raise ListError(f"{path.name} {row['name']}: {row['instance_id']} is not a vision instance")
            reads.setdefault((row["child_sha256"], *rows), {"list": path.name, "entry": row["name"],
                                                             "instance_id": row["instance_id"]})
    return reads


def build(record: Mapping[str, Any], network_list: Path, earlier: Sequence[Path]
          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The v5 entries and the reuse map of all of T07's cells."""
    listed = _json(network_list)
    by_name = {(block, row["name"]): row for block in ("finals", "references") for row in listed[block]}
    reads = earlier_reads(earlier)
    entries: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for arch, network, set_name in t07_cells(record):
        block, name = NETWORKS[(arch, network)]
        own = by_name[(block, name)]
        source = own["source"]
        if block == "references" and (source.get("kind") != SF.SOURCE_REFERENCE or source.get("pick") != SF.PICK_KEPT
                                      or source.get("s_old_floor") != -20):
            raise ListError(f"{name}: not F1's kept pick at floor -20")
        if block == "finals" and source.get("kind") != SF.SOURCE_SESSION:
            raise ListError(f"{name}: not a session final")
        dataset, condition = SF._vision_rows(own["instance_id"])  # noqa: SLF001
        if dataset != DATASETS[arch]:
            raise ListError(f"{name}: {own['instance_id']} is on {dataset}, T07's {arch} reads {DATASETS[arch]}")
        cell = {"arch": arch, "network": network, "set": set_name, "dataset": dataset,
                "instance_id": own["instance_id"], "child_sha256": own["child_sha256"]}
        if set_name == CLEAN:
            cells.append({**cell, "status": "reused", "list": network_list.name, "entry": name,
                          "field": "readouts_clean"})
            continue
        earlier_read = reads.get((own["child_sha256"], dataset, set_name))
        if earlier_read is not None:
            cells.append({**cell, "status": "reused", "list": earlier_read["list"], "entry": earlier_read["entry"],
                          "earlier_instance_id": earlier_read["instance_id"], "field": "readouts"})
            continue
        if set_name == condition:
            raise ListError(f"{name}: its own condition {condition} has no earlier read")
        entry = {"name": f"T07_{arch}_{network}__{set_name}", "readout": SF.READOUT_VISION_RANGE,
                 "source": {"kind": SF.SOURCE_RANGE, "set": set_name, "network": dict(source),
                            "dev_record": T07_RECORD, "t07_arch": arch, "t07_network": network},
                 "instance_id": own["instance_id"], "child_sha256": own["child_sha256"]}
        entries.append(entry)
        cells.append({**cell, "status": "new", "list": "finals_v5.json", "entry": entry["name"], "field": "readouts"})
    return entries, cells


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-out", type=Path, required=True)
    parser.add_argument("--network-list", type=Path, required=True, help="list v2 (the networks' entries of record)")
    parser.add_argument("--earlier", type=Path, nargs="+", required=True, help="lists v2 to v4")
    args = parser.parse_args(argv)
    t07_path = runs_dir() / T07_RECORD
    if hashlib.sha256(t07_path.read_bytes()).hexdigest() != T07_RECORD_SHA256:
        raise ListError(f"{t07_path}: not T07's table of record {T07_RECORD_SHA256}")
    entries, cells = build(_json(t07_path), args.network_list, args.earlier)
    reused = [c for c in cells if c["status"] == "reused"]
    counts = {f"{arch}": sum(1 for e in entries if e["source"]["t07_arch"] == arch) for arch in DATASETS}
    reuse_text = "; ".join(f"{c['arch']} {c['network']} {c['set']} from {c['list']} {c['entry']} ({c['field']})"
                           for c in reused)
    note = (
        "List v5 of the sealed read (PREREG \"T07 sealed read (list v5)\"; user 2026-11-07 about 13:40 Beijing via "
        "WEIGHTCODE-main, \"2、要\"), built by scripts/sealed/make_finals_v5.py from T07's table of record "
        f"({T07_RECORD}, sha256 {T07_RECORD_SHA256[:16]}...). T07's eight networks (B finals of record and F1's kept "
        "picks at floor -20, sources copied from list v2) on T07's 25 corruption sets, each read on the official "
        "CIFAR-10-C test rows of its corruption and severity (10,000 per set) through the range index "
        "($WEIGHTCODE_PRIVATE/<dataset>/t07_range/<set>/dataset.json). "
        f"{len(entries)} vision_range entries ({', '.join(f'{k} {v}' for k, v in counts.items())}). "
        f"{len(reused)} of T07's {len(cells)} cells are not read again; their test cells are the earlier reads: "
        f"{reuse_text}. Each entry is dev-checked equal to its T07 row and predictions on NSCC before the read. Read "
        "once, after the user approves the list."
    )
    out = {"schema": SF.LIST_SCHEMA, "status": "draft", "approval": None, "note": note, "finals": entries,
           "references": []}
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    SF.load_list(args.out)  # the list validates as the pipeline reads it
    reuse = {"schema": "weightcode.t07_v5_cells.v1", "t07_record": T07_RECORD, "t07_record_sha256": T07_RECORD_SHA256,
             "list": args.out.name, "n_cells": len(cells), "n_new": len(entries), "n_reused": len(reused),
             "cells": cells}
    args.reuse_out.write_text(json.dumps(reuse, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(entries)} entries {counts}; {args.reuse_out}: {len(cells)} cells, {len(reused)} reused")
    for c in reused:
        print(f"  reused {c['arch']:<3} {c['network']:<19} {c['set']:<18} <- {c['list']} {c['entry']} ({c['field']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
