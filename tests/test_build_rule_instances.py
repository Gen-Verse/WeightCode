"""scripts/build_rule_instances.py: a thin entry over the scanskill-v1 and policyskill-v1 builders. It passes its
arguments through unchanged, refuses options of the other rule, and on the toy SCAN store builds the instance that a
direct call of the builder then reuses."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from weightcode.evalkit import e14 as E
from weightcode.evalkit import scanskill as K

SPEC = importlib.util.spec_from_file_location(
    "build_rule_instances", Path(__file__).resolve().parents[1] / "scripts" / "build_rule_instances.py")
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def test_constants_match_the_builders() -> None:
    assert script.SCAN_RULE == K.SCAN_RULE
    assert script.POLICY_RULE == E.POLICYSKILL_RULE
    assert script.SCAN_DEFAULT_BATCH == K.EVALUATION_BATCH


def test_arguments_reach_the_builders_unchanged(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(K, "build_scan_instances",
                        lambda condition, *, seeds=None, batch_size=K.EVALUATION_BATCH:
                        calls.append(("scan", (condition, seeds, batch_size))) or [{"instance_id": "n00000000"}])
    monkeypatch.setattr(E, "build_e14_instance",
                        lambda model_id: calls.append(("e14", model_id)) or {"instance_id": "p00000000"})
    assert script.main(["--rule", "scanskill-v1", "--condition", "addprim_jump"]) == 0
    assert script.main(["--rule", "scanskill-v1", "--condition", "around_right", "--seeds", "100,101",
                        "--batch-size", "256"]) == 0
    assert script.main(["--rule", "policyskill-v1", "--model-id", "mdb8464d0"]) == 0
    assert calls == [("scan", ("addprim_jump", None, 1024)), ("scan", ("around_right", [100, 101], 256)),
                     ("e14", "mdb8464d0")]
    assert '"instance_id": "p00000000"' in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["--rule", "scanskill-v1"],
    ["--rule", "scanskill-v1", "--condition", "addprim_jump", "--model-id", "m1"],
    ["--rule", "scanskill-v1", "--condition", "addprim_jump", "--batch-size", "0"],
    ["--rule", "policyskill-v1"],
    ["--rule", "policyskill-v1", "--model-id", "m1", "--condition", "addprim_jump"],
    ["--rule", "policyskill-v1", "--model-id", "m1", "--seeds", "100"],
    ["--rule", "evidence-v1", "--condition", "addprim_jump"],
])
def test_options_of_the_other_rule_are_refused(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(K, "build_scan_instances", lambda *a, **k: pytest.fail("built despite a refusal"))
    monkeypatch.setattr(E, "build_e14_instance", lambda *a, **k: pytest.fail("built despite a refusal"))
    with pytest.raises(SystemExit) as raised:
        script.main(argv)
    assert raised.value.code == 2


def test_toy_scan_store_build_then_reuse(tmp_path: Path, capsys) -> None:
    pytest.importorskip("torch")
    import json

    import scan_toy

    patcher = pytest.MonkeyPatch()
    try:
        scan_toy.store_with_students(tmp_path, patcher)
        assert script.main(["--rule", "scanskill-v1", "--condition", "addprim_jump"]) == 0
        (line,) = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
        (again,) = K.build_scan_instances("addprim_jump")
        assert again["instance_id"] == line["instance_id"]
    finally:
        patcher.undo()
