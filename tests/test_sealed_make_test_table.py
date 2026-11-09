"""scripts/sealed/make_test_table.py on synthetic records of the approved list v2 (no store or sealed file)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("make_test_table", ROOT / "scripts/sealed/make_test_table.py")
MT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MT)  # type: ignore[union-attr]

LIST = ROOT / "scripts/sealed/finals_v2.json"
CELLS = json.loads((ROOT / "scripts/sealed/finals_v2_cells.json").read_text(encoding="utf-8"))


def _records(gain: float = 12.0, clean: float = -0.5) -> dict[str, dict]:
    records = {}
    for entry in MT.list_entries(LIST):
        record = {"name": entry["name"], "gain": gain, "clean": clean, "joint": gain + clean}
        if entry.get("readout") == "scan_em":
            record["readouts_scan"] = {"new": {"em_child": 0.6, "em_base": 0.1}}
        records[entry["name"]] = record
    return records


def test_every_entry_is_used_and_cells_follow_their_definitions() -> None:
    records = _records()
    records["E11_C_gaussian_s103"].update(gain=6.0, joint=5.5)  # ratio 0.5 on one student
    result = MT.build(records, MT.list_entries(LIST), CELLS)
    assert result["missing"] == [] and result["not_used_by_any_cell"] == []
    main = result["cells"]["main"]
    assert main["vit_gauss"] == {"gain": 12.0, "clean": -0.5, "net": 11.5, "ft_kept": 11.5, "ft_free": 11.5,
                                 "ft_free_clean": -0.5, "ref_kept": 11.5}
    assert main["babyai_pickup"]["ft_kept"] == 0.0  # base no-op by construction
    gauss = result["cells"]["reuse"]["gaussian"]["program"]
    assert gauss["gain"] == pytest.approx((4 * 12.0 + 6.0) / 5)
    assert gauss["share"] == pytest.approx(gauss["gain"] / 12.0)
    assert gauss["n_ratio_and_clean_ok"] == 4
    assert result["cells"]["reuse"]["contrast"]["reselect"]["gain"] == pytest.approx(4 * 12.0 / 5)  # s104 no-op
    assert result["cells"]["stack"]["program"] == [11.5] * 4
    assert result["cells"]["stack"]["max_forgetting"] == 0.0
    assert result["bars"]["B E11_B_gaussian_s100"]["reading"] == "missed"  # joint not >= reference joint + 1
    assert result["bars"]["E13 addprim_jump graded"]["reading"] == "met"


def test_missing_entry_leaves_cells_null_and_reading_unreadable() -> None:
    records = _records()
    del records["E10_stream_v2_W4_snow"]
    result = MT.build(records, MT.list_entries(LIST), CELLS)
    assert result["missing"] == ["E10_stream_v2_W4_snow"]
    assert result["cells"]["stack"]["program"][3] is None and result["cells"]["stack"]["max_forgetting"] is None
    assert result["bars"]["E10 stream v2 (absolute criteria and the F4 comparison)"]["reading"].startswith("not readable")


def test_stream_forgetting_is_signed_and_clean_must_agree() -> None:
    records = _records()
    records["E10_stream_v2_W3_snow"].update(gain=4.5, joint=4.0)
    records["E10_stream_v2_W4_snow"].update(gain=4.6, joint=4.1)
    for cond in MT.STREAM_CONDITIONS:  # gaussian and defocus rise after their own rounds
        for w, g in (("W1", 1.0), ("W2", 2.0)):
            name = f"E10_stream_v2_{w}_{cond}"
            if cond in ("gaussian", "defocus"):
                records[name].update(gain=g, joint=g - 0.5)
    result = MT.build(records, MT.list_entries(LIST), CELLS)
    assert result["stream"]["program"]["max_forgetting"] == pytest.approx(-0.1)
    records["E10_stream_v2_W2_snow"].update(clean=-0.6, joint=12.0 - 0.6)
    with pytest.raises(MT.TableError):
        MT.build(records, MT.list_entries(LIST), CELLS)
