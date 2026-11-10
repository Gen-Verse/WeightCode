"""Which bubblewrap and which interpreters the sandbox accepts (``weightcode.sandbox.host``).

``check_bwrap`` refuses a bubblewrap older than 0.8.0 by name. ``interpreter_mounts`` mounts a venv whose base
interpreter lies outside /usr (a uv-managed CPython, say) file by file and recreates the links on the way, and refuses
layouts it cannot mount. These tests use fake layouts; the real sandbox runs in test_sandbox_kernel and
test_sandbox_isolation with whatever interpreter ``.venv-sandbox`` links to.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from weightcode.sandbox.host import SandboxError, check_bwrap, interpreter_mounts


def _fake_bwrap(tmp_path: Path, output: str) -> Path:
    path = tmp_path / "bwrap"
    path.write_text(f"#!/bin/sh\necho '{output}'\n")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("output,ok", [("bubblewrap 0.8.0", True), ("bubblewrap 0.11.1", True),
                                       ("bubblewrap 0.10.0", True), ("bubblewrap 0.6.1", False),
                                       ("bubblewrap 0.7.9", False), ("something else", False)])
def test_check_bwrap_needs_0_8(tmp_path: Path, output: str, ok: bool) -> None:
    bwrap = _fake_bwrap(tmp_path, output)
    if ok:
        assert check_bwrap(bwrap) == output
    else:
        with pytest.raises(SandboxError, match="0.8.0|cannot read the version"):
            check_bwrap(bwrap)


def test_check_bwrap_names_a_missing_binary(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="missing or not executable.*--bwrap"):
        check_bwrap(tmp_path / "nope")


def _installation(root: Path, version: str = "3.14", *, stdlib: bool = True) -> Path:
    """A fake standalone CPython: <prefix>/bin/pythonX.Y, <prefix>/lib/pythonX.Y/os.py, libpython, other files."""
    prefix = root / f"cpython-{version}.9-linux"
    (prefix / "bin").mkdir(parents=True)
    exe = prefix / "bin" / f"python{version}"
    exe.write_text("")
    if stdlib:
        (prefix / "lib" / f"python{version}").mkdir(parents=True)
        (prefix / "lib" / f"python{version}" / "os.py").write_text("")
    (prefix / "lib").mkdir(exist_ok=True)
    (prefix / "lib" / f"libpython{version}.so.1.0").write_text("")
    (prefix / "share" / "secret").mkdir(parents=True)
    (root / f"cpython-{version}-linux").symlink_to(prefix.name)  # a uv-style minor-version link, relative
    return root / f"cpython-{version}-linux" / "bin" / f"python{version}"


def _venv(root: Path, base: Path) -> Path:
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text(f"home = {base.parent}\n")
    (root / "bin" / "python").symlink_to(base)
    return root / "bin" / "python"


def test_a_base_outside_usr_is_mounted_file_by_file(tmp_path: Path) -> None:
    base = _installation(tmp_path / "uv")
    python = _venv(tmp_path / "venv", base)
    mounts = interpreter_mounts(python)
    prefix = tmp_path / "uv" / "cpython-3.14.9-linux"
    assert mounts.venv == tmp_path / "venv"
    assert mounts.binds == (prefix / "bin" / "python3.14", prefix / "lib" / "python3.14",
                            prefix / "lib" / "libpython3.14.so.1.0")
    assert mounts.symlinks == (("cpython-3.14.9-linux", str(tmp_path / "uv" / "cpython-3.14-linux")),)
    argv = mounts.argv()
    assert argv[argv.index("--symlink") - 1] == str(tmp_path / "uv")  # its parent directory is created first
    assert not any("share" in part for part in argv)  # the rest of the installation stays invisible


def test_a_base_under_usr_needs_only_the_venv(tmp_path: Path) -> None:
    candidates = sorted(Path("/usr/bin").glob("python3.[0-9]*"))
    base = next((p for p in candidates if p.name.count(".") == 1 and os.path.realpath(p).startswith("/usr/")), None)
    if base is None:
        pytest.skip("no /usr/bin/python3.X on this host")
    mounts = interpreter_mounts(_venv(tmp_path / "venv", base))
    assert mounts.binds == () and mounts.symlinks == ()


def test_an_installation_without_its_standard_library_is_refused(tmp_path: Path) -> None:
    python = _venv(tmp_path / "venv", _installation(tmp_path / "uv", stdlib=False))
    with pytest.raises(SandboxError, match="standard library"):
        interpreter_mounts(python)


def test_an_interpreter_not_named_pythonx_y_is_refused(tmp_path: Path) -> None:
    exe = tmp_path / "opt" / "bin" / "py"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    with pytest.raises(SandboxError, match="pythonX.Y"):
        interpreter_mounts(_venv(tmp_path / "venv", exe))


def test_a_link_on_the_venv_path_is_refused(tmp_path: Path) -> None:
    """A link that the venv's own bind would shadow cannot be recreated: ask for real paths instead."""
    real = tmp_path / "real"
    base = _installation(real / "uv")
    _venv(real / "venv", base)
    (tmp_path / "alias").symlink_to(real)
    python = _venv(tmp_path / "venv2", tmp_path / "alias" / "uv" / "cpython-3.14-linux" / "bin" / "python3.14")
    mounts = interpreter_mounts(python)  # the alias link lies beside the venv: recreated
    assert (str(real), str(tmp_path / "alias")) in mounts.symlinks
    nested = _venv(real / "uv" / "venv3", tmp_path / "alias" / "uv" / "cpython-3.14-linux" / "bin" / "python3.14")
    assert interpreter_mounts(nested).binds  # a venv inside the real tree, reached without the alias: fine
    # The venv is reached through the alias and so is its base: the alias link would sit on the venv's own path.
    _venv(real / "venv4", tmp_path / "alias" / "uv" / "cpython-3.14-linux" / "bin" / "python3.14")
    with pytest.raises(SandboxError, match="overlaps"):
        interpreter_mounts(tmp_path / "alias" / "venv4" / "bin" / "python")


def test_the_cli_requires_a_sandbox_or_a_trusted_hash(capsys: pytest.CaptureFixture[str]) -> None:
    from weightcode.application.cli import main, parser

    common = ["--task", "t", "--out", "o", "--wall-s", "1"]
    with pytest.raises(SystemExit) as stop:
        main(["apply", *common, "--program", "p"])
    assert stop.value.code == 2 and "--program-sha256" in capsys.readouterr().err
    args = parser().parse_args(["apply", *common, "--program", "p", "--program-sha256", "a" * 64])
    assert args.sandbox_python is None and args.program_sha256 == "a" * 64
    with pytest.raises(SystemExit):  # synthesize always needs the sandbox
        parser().parse_args(["synthesize", *common, "--editor-model", "m", "--checks", "1", "--cells", "1"])
    args = parser().parse_args(["doctor", "--sandbox-python", "x", "--bwrap", "/opt/bwrap/bin/bwrap"])
    assert args.bwrap == Path("/opt/bwrap/bin/bwrap")
