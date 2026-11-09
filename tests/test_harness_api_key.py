"""The editor's API-key billing option (weightcode.harness.launch: read_api_key, billing_record, ClaudeLauncher)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from weightcode.harness import launch

SESSION = {"mcp": {"tool_timeout_ms": 1000, "connect_timeout_ms": 1000}}
KEY = "sk-test-not-a-real-key-0123456789"


def _key_file(tmp_path: Path, content: str = KEY + "\n", mode: int = 0o600) -> Path:
    path = tmp_path / "anthropic_api_key"
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)
    return path


def _fake_claude(tmp_path: Path) -> str:
    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    return str(exe)


def test_key_is_read_from_a_private_one_line_file(tmp_path: Path) -> None:
    assert launch.read_api_key(_key_file(tmp_path)) == KEY


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o660])
def test_group_or_world_access_is_refused(tmp_path: Path, mode: int) -> None:
    with pytest.raises(launch.ApiKeyFileError, match="mode"):
        launch.read_api_key(_key_file(tmp_path, mode=mode))


@pytest.mark.parametrize("content", ["", "\n\n", f"{KEY}\n{KEY}\n", "two words\n"])
def test_anything_but_one_key_line_is_refused_without_echoing_it(tmp_path: Path, content: str) -> None:
    with pytest.raises(launch.ApiKeyFileError) as raised:
        launch.read_api_key(_key_file(tmp_path, content=content))
    assert KEY not in str(raised.value) and "two words" not in str(raised.value)


def test_missing_file_and_directory_are_refused(tmp_path: Path) -> None:
    with pytest.raises(launch.ApiKeyFileError, match="does not exist"):
        launch.read_api_key(tmp_path / "absent")
    with pytest.raises(launch.ApiKeyFileError, match="regular file"):
        launch.read_api_key(tmp_path)


def test_billing_record_never_holds_the_key(tmp_path: Path) -> None:
    path = _key_file(tmp_path)
    assert launch.billing_record(None) == {"mode": launch.BILLING_SUBSCRIPTION}
    record = launch.billing_record(path)
    assert record == {"mode": launch.BILLING_API_KEY_FILE, "api_key_file": str(path.absolute())}
    assert KEY not in repr(record)


def test_launcher_sets_the_key_only_with_a_file_and_strips_inherited_keys(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "inherited-from-parent")
    plain = launch.ClaudeLauncher(_fake_claude(tmp_path))
    env, removed = plain.environment(SESSION)
    assert launch.API_KEY_ENV not in env and launch.API_KEY_ENV in removed
    assert plain.billing == {"mode": launch.BILLING_SUBSCRIPTION}
    keyed = launch.ClaudeLauncher(_fake_claude(tmp_path), api_key_file=_key_file(tmp_path))
    env, removed = keyed.environment(SESSION)
    assert env[launch.API_KEY_ENV] == KEY and launch.API_KEY_ENV in removed
    assert keyed.billing["mode"] == launch.BILLING_API_KEY_FILE and KEY not in repr(keyed.billing)


def test_launcher_refuses_a_bad_file_at_construction(tmp_path: Path) -> None:
    with pytest.raises(launch.ApiKeyFileError):
        launch.ClaudeLauncher(_fake_claude(tmp_path), api_key_file=_key_file(tmp_path, mode=0o644))
    assert os.environ.get("ANTHROPIC_API_KEY") != KEY
