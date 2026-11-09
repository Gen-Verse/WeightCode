"""scripts/sealed/make_finals_v5.py's reuse rule and scripts/sealed/make_test_table_v5.py on synthetic records (no store
or sealed file): T07's cell rule on per-image records, the canonical interval of counts-only records, the paired and
point-only final-minus-fine-tuning rows, the base consistency checks and the dev-test agreement readings."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sealed"))
SPEC = importlib.util.spec_from_file_location("make_test_table_v5", ROOT / "scripts/sealed/make_test_table_v5.py")
MT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MT)  # type: ignore[union-attr]
T = MT.T

from weightcode.evalkit.sealed import manifest_sha256  # noqa: E402
from weightcode.evalkit.sealed_range import per_image  # noqa: E402

SETS = ["clean", "gaussian_noise_s5", "contrast_s5", "fog_s5"]
N = 60


def _readouts(base: np.ndarray, child: np.ndarray) -> dict[str, Any]:
    return {"n": len(base), "fixed": int(np.sum(child & ~base)), "broken": int(np.sum(base & ~child)),
            "base_acc": float(base.mean()), "child_acc": float(child.mean()),
            "delta_acc": float(child.mean() - base.mean()), "worst_class_delta": -0.01}


def _world(seed: int = 0) -> dict[str, Any]:
    """Per-image correctness of a base and four networks per architecture and set; a cell is reused (counts only)
    on clean and on the network's own condition at severity 5, new elsewhere."""
    rng = np.random.default_rng(seed)
    cells, records_new, records_old, truth = [], {}, {}, {}
    for arch in MT.ARCHS:
        base = {s: rng.random(N) < 0.6 for s in SETS}
        for nw in MT.NETWORKS:
            own = nw.split("_", 1)[1] + "_s5"
            old_name = f"old_{arch}_{nw}"
            for s in SETS:
                child = base[s].copy()
                flip = rng.random(N) < 0.3
                child[flip] = ~child[flip]
                truth[(arch, nw, s)] = (base[s], child)
                if s in ("clean", own):
                    field = "readouts_clean" if s == "clean" else "readouts"
                    records_old.setdefault(old_name, {"name": old_name, "mode": "dev", "instance_id": "i0000000a",
                                                      "child_sha256": f"{arch}{nw}"})[field] = _readouts(base[s], child)
                    cells.append({"arch": arch, "network": nw, "set": s, "status": "reused", "list": "old.json",
                                  "entry": old_name, "field": field})
                else:
                    name = f"T07_{arch}_{nw}__{s}"
                    records_new[name] = {"name": name, "mode": "dev", "instance_id": "i0000000a",
                                         "child_sha256": f"{arch}{nw}", "readout": "vision_range",
                                         "source": {"set": s}, "readouts": _readouts(base[s], child),
                                         "per_image": per_image(base[s], child), "reference": {"match": True}}
                    cells.append({"arch": arch, "network": nw, "set": s, "status": "new", "list": "finals_v5.json",
                                  "entry": name, "field": "readouts"})
    return {"cells": cells, "records": {**records_new, **records_old}, "truth": truth, "new": records_new,
            "old": records_old}


def _t07_like(world: dict[str, Any]) -> dict[str, Any]:
    """T07's dev table of the same per-image data, computed directly from T07's functions."""
    index = T.boot_index(N)
    rows, ft = [], []
    for (arch, nw, s), (base, child) in world["truth"].items():
        boot = T.boot_contrast(child.astype(np.int64) - base.astype(np.int64), index)
        clean_base = world["truth"][(arch, nw, "clean")][0]
        headroom, share = T.recovered_share(boot["point"], clean_base.mean(), base.mean())
        ro = _readouts(base, child)
        rows.append({"arch": arch, "network": nw, "set": s, "delta_points": boot["point"], "ci95": boot["ci95"],
                     "boot_sd": boot["boot_sd"], "label": T.label(boot["point"], boot["ci95"]), "fixed": ro["fixed"],
                     "broken": ro["broken"], "worst_class_delta_points": -1.0, "base_acc": ro["base_acc"],
                     "child_acc": ro["child_acc"], "n": N, "base_loss_from_clean_points": headroom,
                     "recovered_share": share})
    for arch in MT.ARCHS:
        for c in MT.CONDITIONS:
            for s in SETS:
                a, b = world["truth"][(arch, f"llm_{c}", s)][1], world["truth"][(arch, f"ft_{c}", s)][1]
                boot = T.boot_contrast(a.astype(np.int64) - b.astype(np.int64), index)
                ft.append({"arch": arch, "condition": c, "set": s, "llm_minus_ft_points": boot["point"],
                           "ci95": boot["ci95"]})
    return {"sets": SETS, "rows": rows, "llm_minus_ft": ft}


def test_new_cells_follow_t07s_rule_and_reused_cells_use_the_canonical_vector() -> None:
    world = _world()
    t07 = _t07_like(world)
    table = MT.build({"cells": world["cells"]}, world["records"], t07, "dev")
    index = T.boot_index(N)
    for row in table["rows"]:
        base, child = world["truth"][(row["arch"], row["network"], row["set"])]
        assert row["delta_points"] == pytest.approx(100 * (child.mean() - base.mean()), abs=1e-9)
        if row["source"]["status"] == "new":
            expected = T.boot_contrast(child.astype(np.int64) - base.astype(np.int64), index)
        else:
            canonical = np.zeros(N, np.int64)
            canonical[:row["fixed"]] = 1
            canonical[row["fixed"]:row["fixed"] + row["broken"]] = -1
            expected = T.boot_contrast(canonical, index)
        assert row["ci95"] == expected["ci95"] and row["label"] == T.label(expected["point"], expected["ci95"])
    paired = {(r["arch"], r["condition"], r["set"]): r["paired"] for r in table["llm_minus_ft"]}
    assert sum(paired.values()) == 2 * 2 * 2 and not paired[("vit", "contrast", "contrast_s5")]
    assert not paired[("r18", "gaussian_noise", "clean")] and paired[("vit", "contrast", "fog_s5")]
    problems = MT.reproduction(table, t07)
    assert all("not an error" in p for p in problems)
    read = MT.readings(table, t07)
    assert all(r["n_sets"] == len(SETS) for r in read.values())


def test_readings_count_label_changes_against_the_dev_table() -> None:
    world = _world()
    t07 = _t07_like(world)
    table = MT.build({"cells": world["cells"]}, world["records"], t07, "dev")
    target = next(r for r in t07["rows"] if (r["arch"], r["network"], r["set"]) == ("vit", "llm_contrast", "fog_s5"))
    ours = next(r for r in table["rows"] if (r["arch"], r["network"], r["set"]) == ("vit", "llm_contrast", "fog_s5"))
    before = MT.readings(table, t07)["vit llm_contrast"]  # a reused cell's canonical interval may move its label
    target["label"] = "harm" if ours["label"] == "help" else "help"
    read = MT.readings(table, t07)["vit llm_contrast"]
    assert read["n_same_label_as_dev"] == before["n_same_label_as_dev"] - 1
    assert read["changed"][-1] == {"set": "fog_s5", "dev": target["label"], "test": ours["label"]}
    assert all(c["set"] in ("clean", "contrast_s5") for c in before["changed"])
    assert read["n_reversed"] == (1 if ours["label"] in ("help", "harm") else 0)
    assert any("vit llm_contrast fog_s5 label" in p for p in MT.reproduction(table, t07))


@pytest.mark.parametrize("damage, message", [
    ("base", "bases are not correct on the same images"),
    ("bits", "do not give the readout's fixed and broken"),
    ("reference", "did not match its T07 record"),
    ("size", "cells of different sizes"),
])
def test_build_refuses_inconsistent_records(damage: str, message: str) -> None:
    world = _world()
    records = world["records"]
    name = "T07_vit_llm_contrast__fog_s5"
    if damage == "base":
        base, child = world["truth"][("vit", "llm_contrast", "fog_s5")]
        base2, child2 = base.copy(), child.copy()
        base2[0], child2[0] = ~base2[0], ~child2[0]
        records[name].update(readouts=_readouts(base2, child2), per_image=per_image(base2, child2))
        records[name]["readouts"]["base_acc"] = float(base.mean())
    elif damage == "bits":
        records[name]["readouts"]["fixed"] += 1
    elif damage == "reference":
        records[name]["reference"]["match"] = False
    else:
        base, child = world["truth"][("vit", "llm_contrast", "fog_s5")]
        records[name].update(readouts=_readouts(base[:-1], child[:-1]), per_image=per_image(base[:-1], child[:-1]))
    with pytest.raises(MT.TableError, match=message):
        MT.build({"cells": world["cells"]}, records, _t07_like(world), "dev")


def test_load_records_checks_mode_list_and_entry(tmp_path: Path) -> None:
    listed = {"schema": "x", "finals": [{"name": "a", "instance_id": "i0000000a", "child_sha256": "c"}],
              "references": []}
    list_path = tmp_path / "finals_vx.json"
    list_path.write_text(json.dumps(listed))
    good = {"name": "a", "mode": "sealed", "list_sha256": manifest_sha256(listed), "instance_id": "i0000000a",
            "child_sha256": "c"}
    results = tmp_path / "r.jsonl"
    for change, message in ((None, None), ({"mode": "dev"}, "mode"), ({"list_sha256": "0"}, "list_sha256"),
                            ({"child_sha256": "d"}, "instance or child"), ({"name": "b"}, "no record")):
        results.write_text(json.dumps({**good, **(change or {})}) + "\n")
        if message is None:
            assert MT.load_records([results], "sealed", {list_path.name: list_path}, {"a": list_path.name})["a"] == good
        else:
            with pytest.raises(MT.TableError, match=message):
                MT.load_records([results], "sealed", {list_path.name: list_path}, {"a": list_path.name})


def test_the_list_and_reuse_map_of_record() -> None:
    """finals_v5.json and finals_v5_reused.json: 191 range entries, 17 reused cells, every T07 cell once."""
    from weightcode.evalkit import sealed_finals as SF

    parsed, entries = SF.load_list(ROOT / "scripts/sealed/finals_v5.json")
    reuse = json.loads((ROOT / "scripts/sealed/finals_v5_reused.json").read_text(encoding="utf-8"))
    # Approved by the user 2026-11-07 about 16:12 Beijing (main 5fbb743); the entries are those of the draft.
    assert parsed["status"] == "approved" and "2026-11-07" in parsed["approval"] and parsed["references"] == []
    assert len(entries) == 191 and {e.readout for e in entries} == {"vision_range"}
    assert reuse["n_cells"] == 208 and reuse["n_new"] == 191 and reuse["n_reused"] == 17
    new = {c["entry"] for c in reuse["cells"] if c["status"] == "new"}
    assert new == {e.name for e in entries}
    reused = [c for c in reuse["cells"] if c["status"] == "reused"]
    assert sum(c["set"] == "clean" for c in reused) == 8
    assert {(c["list"], c["entry"]) for c in reused if c["list"] == "finals_v4.json"} == {
        ("finals_v4.json", "E11_stream_W1_contrast")}
    v2 = {e["name"]: e for block in ("finals", "references")
          for e in json.loads((ROOT / "scripts/sealed/finals_v2.json").read_text(encoding="utf-8"))[block]}
    for entry in entries:
        network = v2[MT_NETWORKS[(entry.source["t07_arch"], entry.source["t07_network"])]]
        assert entry.source["network"] == network["source"] and entry.child_sha256 == network["child_sha256"]
        assert entry.instance_id == network["instance_id"] and entry.source["set"] in SETS_T07


SETS_T07 = set(T.C.SETS) - {"clean"}
MT_NETWORKS = {(a, n): name for (a, n), (_, name) in
               __import__("make_finals_v5").NETWORKS.items()}
