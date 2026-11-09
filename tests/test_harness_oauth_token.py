"""The editor's OAuth-token billing option (weightcode.harness.launch: read_oauth_token, billing_record, ClaudeLauncher).

A long-lived subscription token from ``claude setup-token`` bills batch sessions on NSCC compute nodes (user
2026-11-07). The file rules are the API key file's; the token reaches only the editor's environment and never a record.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from weightcode.harness import launch

SESSION = {"mcp": {"tool_timeout_ms": 1000, "connect_timeout_ms": 1000}}
TOKEN = "sk-ant-oat01-test-not-a-real-token-0123456789"
KEY = "sk-test-not-a-real-key-0123456789"


def _secret_file(tmp_path: Path, name: str, content: str, mode: int = 0o600) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)
    return path


def _fake_claude(tmp_path: Path) -> str:
    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    return str(exe)


def test_token_is_read_from_a_private_one_line_file(tmp_path: Path) -> None:
    assert launch.read_oauth_token(_secret_file(tmp_path, "tok", TOKEN + "\n")) == TOKEN


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644])
def test_group_or_world_access_is_refused(tmp_path: Path, mode: int) -> None:
    with pytest.raises(launch.OAuthTokenFileError, match="mode"):
        launch.read_oauth_token(_secret_file(tmp_path, "tok", TOKEN + "\n", mode=mode))


@pytest.mark.parametrize("content", ["", f"{TOKEN}\n{TOKEN}\n", "two words\n"])
def test_anything_but_one_token_line_is_refused_without_echoing_it(tmp_path: Path, content: str) -> None:
    with pytest.raises(launch.OAuthTokenFileError) as raised:
        launch.read_oauth_token(_secret_file(tmp_path, "tok", content))
    assert TOKEN not in str(raised.value) and "two words" not in str(raised.value)
    assert isinstance(raised.value, launch.CredentialFileError)


def test_billing_record_names_the_file_never_the_token(tmp_path: Path) -> None:
    path = _secret_file(tmp_path, "tok", TOKEN + "\n")
    record = launch.billing_record(None, path)
    assert record == {"mode": launch.BILLING_OAUTH_TOKEN_FILE, "oauth_token_file": str(path.absolute())}
    assert TOKEN not in repr(record)
    with pytest.raises(ValueError, match="not both"):
        launch.billing_record(_secret_file(tmp_path, "key", KEY + "\n"), path)


def test_launcher_sets_the_token_only_with_a_file_and_strips_inherited_credentials(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(launch.OAUTH_TOKEN_ENV, "inherited-from-parent")
    monkeypatch.setenv(launch.API_KEY_ENV, "inherited-from-parent")
    plain = launch.ClaudeLauncher(_fake_claude(tmp_path))
    env, removed = plain.environment(SESSION)
    assert launch.OAUTH_TOKEN_ENV not in env and launch.OAUTH_TOKEN_ENV in removed
    tokened = launch.ClaudeLauncher(_fake_claude(tmp_path), oauth_token_file=_secret_file(tmp_path, "tok", TOKEN))
    env, removed = tokened.environment(SESSION)
    assert env[launch.OAUTH_TOKEN_ENV] == TOKEN and launch.API_KEY_ENV not in env
    assert tokened.billing["mode"] == launch.BILLING_OAUTH_TOKEN_FILE and TOKEN not in repr(tokened.billing)


def test_launcher_refuses_a_bad_file_or_both_files_at_construction(tmp_path: Path) -> None:
    with pytest.raises(launch.OAuthTokenFileError):
        launch.ClaudeLauncher(_fake_claude(tmp_path),
                              oauth_token_file=_secret_file(tmp_path, "tok", TOKEN, mode=0o644))
    with pytest.raises(ValueError, match="not both"):
        launch.ClaudeLauncher(_fake_claude(tmp_path), api_key_file=_secret_file(tmp_path, "key", KEY),
                              oauth_token_file=_secret_file(tmp_path, "tok2", TOKEN))


def test_the_cli_takes_one_billing_file_only() -> None:
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_editor_session.py"
    proc = subprocess.run([sys.executable, str(script), "--api-key-file", "/nonexistent/a",
                           "--oauth-token-file", "/nonexistent/b", "--help"], capture_output=True, text=True)
    assert proc.returncode == 2 and "not allowed with argument" in proc.stderr
