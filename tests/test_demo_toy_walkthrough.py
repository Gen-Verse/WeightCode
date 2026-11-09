"""scripts/demo/toy_walkthrough.py (README step 4) runs end to end on a toy store in a pytest tmp directory: the
scripted session picks the planted program, its readout gains on the condition without a clean loss, all six students
get a transfer record, and a second run into the same directory is refused."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from test_harness_fixtures import needs_sandbox

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("toy_walkthrough", ROOT / "scripts/demo/toy_walkthrough.py")
TW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TW)  # type: ignore[union-attr]


@needs_sandbox
def test_the_walkthrough_runs_end_to_end(tmp_path_factory: pytest.TempPathFactory,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    # main() sets the three store variables itself; setting them here first lets monkeypatch restore them afterwards.
    for name in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED", "WEIGHTCODE_RUNS"):
        monkeypatch.setenv(name, "unset-by-the-test")
    root = tmp_path_factory.mktemp("hs") / "store"  # a neutral name: the sandbox self-test sees the session's paths
    assert TW.main(["--root", str(root)]) == 0

    (session,) = (root / "runs" / "sessions").iterdir()
    final = json.loads((session / "final" / "selection.json").read_text())["final"]
    assert final["patch_id"] == TW.FINAL_PROGRAM
    run = json.loads((session / "run.json").read_text())
    assert run["status"] == "complete"
    records = [json.loads(p.read_text()) for p in sorted((root / "transfer").glob("transfer_*.json"))]
    assert len(records) == 6 and all(r["status"] == "ok" for r in records)
    own = next(r for r in records if r["target"]["instance_id"] == run["instance"]["instance_id"])
    assert own["readouts"]["delta_acc"] > 0 and own["readouts_clean"]["delta_acc"] >= 0
    assert (root / "env.sh").read_text().count("export WEIGHTCODE_") == 3

    with pytest.raises(SystemExit, match="already exists and is not empty"):
        TW.main(["--root", str(root)])
