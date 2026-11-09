"""The `weightcode` skill's documents agree with the engine: every documented `weightcode` command line parses with
the real CLI, the skill's front matter is valid, and the reuse walkthrough (scripts/demo/skill_cli_walkthrough.py)
runs the documented commands end to end on a synthetic project."""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
from pathlib import Path

import pytest
from test_harness_fixtures import needs_sandbox

from weightcode.application.cli import parser

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "weightcode"
DOCS = [ROOT / "README.md", SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))]
COMMAND = re.compile(r"^(?:uv run(?: --project \S+)? |\.venv/bin/)?weightcode (\S.*)$")


def documented_commands() -> list[tuple[str, str]]:
    """Every line inside a fenced block of the skill documents and README that runs the `weightcode` CLI."""
    found = []
    for path in DOCS:
        fenced = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("```"):
                fenced = not fenced
                continue
            match = COMMAND.match(line.strip()) if fenced else None
            if match:
                found.append((f"{path.relative_to(ROOT)}: {line.strip()}", match.group(1)))
    return found


def test_the_documents_name_every_subcommand() -> None:
    used = {shlex.split(args)[0] for _, args in documented_commands()}
    assert used >= {"doctor", "import-model", "prepare-data", "prepare-task", "pack-program", "apply", "synthesize"}


@pytest.mark.parametrize("where, args", documented_commands(), ids=lambda value: value[:60])
def test_every_documented_command_parses(where: str, args: str) -> None:
    try:
        parser().parse_args(shlex.split(args))
    except SystemExit as exc:  # argparse exits on an unknown or missing argument
        raise AssertionError(f"{where} does not parse") from exc


def test_the_skill_front_matter() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    head = text.split("---")[1]
    fields = dict(line.split(": ", 1) for line in head.strip().splitlines())
    assert fields["name"] == "weightcode" == SKILL.name
    assert 0 < len(fields["description"]) <= 1024
    for link in re.findall(r"\]\(([^)#]+)\)", text):
        assert (SKILL / link).is_file(), link


@needs_sandbox
def test_the_reuse_walkthrough_runs_the_documented_commands(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("skill_cli_walkthrough",
                                                  ROOT / "scripts/demo/skill_cli_walkthrough.py")
    walkthrough = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(walkthrough)  # type: ignore[union-attr]
    root = tmp_path / "project"
    assert walkthrough.main(["--root", str(root)]) == 0
    final = root / "work" / "reuse" / "final"
    report = json.loads((final / "report.json").read_text())
    assert report["status"] in {"accepted", "base-kept", "final-rejected"}
    assert (final / "model.pt").is_file() and report["output_is_updated_model"] == (report["status"] == "accepted")
    assert report["final_base"]["old"]["n"] == report["final_base"]["new"]["n"] == 18
