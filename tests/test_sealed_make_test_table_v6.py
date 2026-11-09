"""make_test_table_v6 on synthetic sealed records of list v6 (no sealed data) and on its interval rule."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit.sealed import manifest_sha256
from weightcode.evalkit.sealed_range import per_image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("make_test_table_v6", ROOT / "scripts/sealed/make_test_table_v6.py")
TT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TT)
N = 38_000


def approved_list(tmp: Path) -> Path:
    parsed = json.loads((ROOT / "scripts/sealed/finals_v6.json").read_text())
    parsed["status"], parsed["approval"] = "approved", "synthetic test"
    out = tmp / "finals_v6.json"
    out.write_text(json.dumps(parsed))
    shutil.copy(ROOT / "scripts/sealed/finals_v6_roles.json", tmp / "finals_v6_roles.json")
    return out


def readouts(base: np.ndarray, child: np.ndarray) -> dict:
    return {"n": len(base), "fixed": int((~base & child).sum()), "broken": int((base & ~child).sum()),
            "base_acc": float(base.mean()), "child_acc": float(child.mean()),
            "delta_acc": float(child.mean() - base.mean())}


def records(list_path: Path, rng: np.random.Generator) -> list[dict]:
    parsed, entries = SF.load_list(list_path)
    sha = manifest_sha256(parsed)
    base = {}
    out = []
    for entry in entries:
        if entry.instance_id not in base:
            base[entry.instance_id] = (rng.random(N) < 0.2, rng.random(N) < 0.7)
        b_cond, b_clean = base[entry.instance_id]
        flip = 0.0 if entry.name.endswith("_base") else 0.05
        c_cond = np.where(rng.random(N) < flip, ~b_cond, b_cond)
        c_clean = np.where(rng.random(N) < flip / 5, ~b_clean, b_clean)
        gain = 100.0 * float(np.mean(c_cond.astype(np.int8) - b_cond.astype(np.int8)))
        clean = 100.0 * float(np.mean(c_clean.astype(np.int8) - b_clean.astype(np.int8)))
        out.append({"name": entry.name, "instance_id": entry.instance_id, "child_sha256": entry.child_sha256,
                    "readout": entry.readout, "mode": "sealed", "list_sha256": sha,
                    "condition": "gaussian_noise_s3" if "gaussian" in entry.name else "contrast_s4",
                    "readouts": readouts(b_cond, c_cond), "readouts_clean": readouts(b_clean, c_clean),
                    "per_image": per_image(b_cond, c_cond), "per_image_clean": per_image(b_clean, c_clean),
                    "gain": gain, "clean": clean, "joint": gain + clean})
    return out


def write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_interval_matches_hand_computation() -> None:
    d = np.array([1, 0, 0, -1, 1, 1, 0, 0], dtype=np.int8)
    got = TT.interval(d)
    half = 100 * 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    assert got["point"] == pytest.approx(25.0)
    assert got["lo"] == pytest.approx(25.0 - half) and got["hi"] == pytest.approx(25.0 + half)


def test_sealed_table_readings_follow_the_records(tmp_path: Path) -> None:
    list_path = approved_list(tmp_path)
    rows = records(list_path, np.random.default_rng(0))
    table = TT.build(list_path, "sealed", write(tmp_path / "r.jsonl", rows), [])
    by_name = {r["name"]: r for r in rows}
    for row in table["entries"]:
        assert row["intervals"]["gain"]["point"] == pytest.approx(by_name[row["name"]]["gain"], abs=1e-9)
        assert row["intervals"]["joint"]["point"] == pytest.approx(by_name[row["name"]]["joint"], abs=1e-9)
        joint = row["intervals"]["joint"]
        if row["name"].endswith("_base"):
            assert joint["lo"] == joint["point"] == joint["hi"] == 0.0
        else:
            assert joint["lo"] < joint["point"] < joint["hi"]
    g = table["readings"]["gaussian_noise_s3"]
    assert g["bar"] == pytest.approx(by_name["T04_gaussian_bar_grad_sparse19_sal"]["joint"] + 1.0)
    assert set(g["ft_joints"]) == {"main_ft_bn", "arm3_ft_full_l2sp", "arm3_ft_full_lwf", "arm3_ft_full_noreplay",
                                   "arm3_grad_full", "arm3_grad_last", "arm3_lora_r16", "arm3_lora_r4"}
    c = table["readings"]["contrast_s4"]
    assert c["ft_joints"]["arm3_ft_full_l2sp"] == 0.0 and c["ft_joints"]["arm3_ft_full_lwf"] == 0.0
    for condition in (g, c):
        for final in condition["finals"].values():
            r = by_name[final["entry"]]
            expect = (r["gain"] >= condition["G"] - 1e-9 and r["clean"] >= -1.0 - 1e-9
                      and r["joint"] >= condition["bar"] - 1e-9)
            assert final["B_met"] == expect
            assert final["joint_minus_main_interval"]["point"] == pytest.approx(final["joint_minus_main"], abs=1e-6)


def test_sealed_table_refuses_a_record_off_the_list(tmp_path: Path) -> None:
    list_path = approved_list(tmp_path)
    rows = records(list_path, np.random.default_rng(1))
    rows[3]["child_sha256"] = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(TT.TableError, match="child_sha256"):
        TT.build(list_path, "sealed", write(tmp_path / "r.jsonl", rows), [])


def test_sealed_table_refuses_bits_that_disagree_with_the_gain(tmp_path: Path) -> None:
    list_path = approved_list(tmp_path)
    rows = records(list_path, np.random.default_rng(2))
    rows[0]["gain"] += 0.5
    with pytest.raises(TT.TableError, match="per-image"):
        TT.build(list_path, "sealed", write(tmp_path / "r.jsonl", rows), [])


def test_sealed_table_refuses_the_draft_list(tmp_path: Path) -> None:
    # The list of record is approved since 2026-11-08 (main 3ecd682), so the refusal is tested on a draft copy.
    parsed = json.loads((ROOT / "scripts/sealed/finals_v6.json").read_text())
    parsed["status"], parsed["approval"] = "draft", None
    draft = tmp_path / "finals_v6.json"
    draft.write_text(json.dumps(parsed))
    with pytest.raises(TT.TableError, match="approved"):
        TT.build(draft, "sealed", tmp_path / "none.jsonl", [])
