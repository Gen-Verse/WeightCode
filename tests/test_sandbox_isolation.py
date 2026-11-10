"""Sandbox isolation against the real bubblewrap (docs/HARNESS_SPEC.md section 3) and the isolation self-test.

The fake $WEIGHTCODE_* roots live under ~/.cache, not /tmp: the sandbox mounts its own /tmp, so host paths under
/tmp would be invisible for a trivial reason and the invisibility checks would prove nothing.
"""

from __future__ import annotations

import ast
import os
import secrets
import shutil
import socket
import tempfile
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

import weightcode.sandbox.host as host_module
from weightcode.sandbox import (DEFAULT_BWRAP, KERNEL_SCRIPT, SandboxConfig, SandboxKernel, isolation_selftest,
                                sandbox_environment)
from weightcode.sandbox.selftest import INSIDE_CHECKS

REPO = Path(__file__).resolve().parents[1]
SANDBOX_PYTHON = REPO / ".venv-sandbox" / "bin" / "python"
SANDBOX_VENV = SANDBOX_PYTHON.parent.parent
BWRAP = DEFAULT_BWRAP  # $WEIGHTCODE_BWRAP or /usr/bin/bwrap

pytestmark = [
    pytest.mark.skipif(not BWRAP.exists(), reason=f"{BWRAP} is missing: the sandbox tests need the real bubblewrap"),
    pytest.mark.skipif(not SANDBOX_PYTHON.exists(),
                       reason=f"{SANDBOX_PYTHON} is missing: create the sandbox venv (scripts/setup_sandbox_venv.sh)"),
]

HOST_SIDE_CHECKS = {
    "bind_sources_safe",
    "kernel_start",
    "patches_bind_is_session_dir",
    "scratch_bind_is_session_dir",
    "protocol_survives_fd_writes",
}


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Path]]:
    """Fake $WEIGHTCODE_PRIVATE, $WEIGHTCODE_SEALED and $WEIGHTCODE_RUNS with a marker file each, removed afterwards."""
    root = Path(tempfile.mkdtemp(prefix="weightcode-sandbox-test-", dir=Path.home() / ".cache"))
    roots = {}
    for name, variable in (("private", "WEIGHTCODE_PRIVATE"), ("sealed", "WEIGHTCODE_SEALED"), ("runs", "WEIGHTCODE_RUNS")):
        path = root / name
        path.mkdir()
        (path / "SECRET.txt").write_text(f"{name} secret\n")
        monkeypatch.setenv(variable, str(path))
        roots[name] = path
    yield roots
    shutil.rmtree(root)


def make_dirs(base: Path) -> dict[str, Path]:
    layout = {name: base / name for name in ("trial", "patches", "scratch", "logs")}
    for path in layout.values():
        path.mkdir(parents=True)
    (layout["trial"] / "model_spec.json").write_text('{"schema": "test"}\n')
    return layout


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    return make_dirs(tmp_path / "session")


def make_config(dirs: dict[str, Path], **overrides: object) -> SandboxConfig:
    fields: dict[str, object] = {
        "trial_dir": dirs["trial"],
        "patches_dir": dirs["patches"],
        "scratch_dir": dirs["scratch"],
        "log_path": dirs["logs"] / "cells.jsonl",
        "python": SANDBOX_PYTHON,
        "cell_timeout_s": 20.0,
        "threads": 2,
    }
    fields.update(overrides)
    return SandboxConfig(**fields)  # type: ignore[arg-type]


@pytest.fixture
def kernel(dirs: dict[str, Path]) -> Iterator[SandboxKernel]:
    with SandboxKernel(make_config(dirs)) as running:
        yield running


def value_of(kernel: SandboxKernel, code: str) -> object:
    """Run a cell whose last expression is a literal-friendly value and return that value."""
    result = kernel.run(code)
    assert result.error is None, result.error
    return ast.literal_eval(result.stdout)


def test_numpy_present_and_other_numeric_stacks_absent(kernel: SandboxKernel) -> None:
    version = value_of(kernel, "import numpy\nnumpy.__version__")
    assert isinstance(version, str) and version.startswith("2.")
    for module in ("scipy", "torch", "sklearn", "weightcode"):
        result = kernel.run(f"import {module}")
        assert result.error is not None and "ModuleNotFoundError" in result.error, module


def test_network_dns_and_host_sockets_are_unreachable(kernel: SandboxKernel) -> None:
    abstract_name = f"\0weightcode-sandbox-test-{secrets.token_hex(8)}"
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as abstract,
    ):
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        port = listener.getsockname()[1]
        abstract.bind(abstract_name)
        abstract.listen(4)
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass  # the host itself reaches the listener, so the sandbox's failure below is meaningful
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(abstract_name)
        code = textwrap.dedent(
            f"""
            import socket
            outcomes = {{}}
            def attempt(label, action):
                try:
                    action()
                except OSError as exc:
                    outcomes[label] = type(exc).__name__
                else:
                    outcomes[label] = "CONNECTED"
            def unix():
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.connect({abstract_name!r})
            attempt("tcp_internet", lambda: socket.create_connection(("1.1.1.1", 443), timeout=3).close())
            attempt("dns", lambda: socket.getaddrinfo("example.com", 443))
            attempt("host_loopback_listener", lambda: socket.create_connection(("127.0.0.1", {port}), timeout=3).close())
            attempt("host_abstract_unix_socket", unix)
            outcomes
            """
        )
        outcomes = value_of(kernel, code)
    assert isinstance(outcomes, dict) and len(outcomes) == 4
    assert "CONNECTED" not in outcomes.values(), outcomes
    assert outcomes["dns"] == "gaierror"


def test_host_files_and_protected_roots_are_invisible(store: dict[str, Path], dirs: dict[str, Path]) -> None:
    home = Path.home()
    hidden = [
        home / ".claude",
        store["private"],
        store["sealed"],
        store["runs"],
        store["private"] / "SECRET.txt",
        Path("/etc/passwd"),
        Path("/etc"),
        REPO / "weightcode",
        REPO / ".venv",
        REPO / "docs",
        KERNEL_SCRIPT,
        dirs["trial"],
        dirs["logs"],
    ]
    present_on_host = {path for path in hidden if os.path.lexists(path)}
    assert present_on_host >= {store["private"] / "SECRET.txt", REPO / "weightcode", KERNEL_SCRIPT, dirs["trial"]}
    with SandboxKernel(make_config(dirs)) as kernel:
        visible = value_of(kernel, f"import os\n[p for p in {[str(p) for p in hidden]!r} if os.path.lexists(p)]")
        repo_listing = value_of(kernel, f"import os\nsorted(os.listdir({str(REPO)!r}))")
        home_listing = value_of(kernel, f"import os\nsorted(os.listdir({str(home)!r})) if os.path.exists({str(home)!r}) else None")
    assert visible == []
    assert repo_listing == [SANDBOX_VENV.name]  # the repo root shows nothing but the path to the sandbox venv
    if SANDBOX_VENV.is_relative_to(home):
        assert home_listing == [SANDBOX_VENV.relative_to(home).parts[0]]
    else:
        assert home_listing is None


def test_parent_environment_is_not_visible(
    store: dict[str, Path], dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    canary = secrets.token_hex(16)
    monkeypatch.setenv("WEIGHTCODE_CANARY", canary)
    config = make_config(dirs)
    with SandboxKernel(config) as kernel:
        environment = value_of(kernel, "import os\ndict(os.environ)")
        initial = value_of(kernel, "open('/proc/self/environ', 'rb').read()")
    expected = sandbox_environment(config)
    assert isinstance(environment, dict) and isinstance(initial, bytes)
    assert set(environment) <= set(expected) | {"PWD", "LC_CTYPE"}
    assert {key: environment.get(key) for key in expected} == expected
    assert not any("WEIGHTCODE" in key for key in environment)
    for secret in (canary, str(store["private"]), str(store["sealed"]), "WEIGHTCODE"):
        assert secret.encode() not in initial and secret not in repr(environment)


def test_trial_is_read_only_while_patches_and_scratch_are_writable(kernel: SandboxKernel, dirs: dict[str, Path]) -> None:
    code = textwrap.dedent(
        """
        import errno, os
        outcome = {}
        for label, action in (
            ("create", lambda: open("/work/trial/new.txt", "x").close()),
            ("modify", lambda: open("/work/trial/model_spec.json", "a").close()),
            ("delete", lambda: os.unlink("/work/trial/model_spec.json")),
            ("chmod", lambda: os.chmod("/work/trial/model_spec.json", 0o777)),
        ):
            try:
                action()
            except OSError as exc:
                outcome[label] = errno.errorcode[exc.errno]
            else:
                outcome[label] = "ALLOWED"
        outcome["read"] = open("/work/trial/model_spec.json").read()
        with open("/work/patches/p01.json", "w") as handle:
            handle.write('{"schema": "weightcode.patch.v0"}')
        with open(os.path.expanduser("~/note.txt"), "w") as handle:
            handle.write("scratch")
        outcome["cwd"] = os.getcwd()
        outcome["home"] = os.path.expanduser("~")
        outcome
        """
    )
    outcome = value_of(kernel, code)
    assert isinstance(outcome, dict)
    assert {outcome[label] for label in ("create", "modify", "delete", "chmod")} == {"EROFS"}
    assert outcome["read"] == '{"schema": "test"}\n'
    assert (outcome["cwd"], outcome["home"]) == ("/work", "/work/scratch")
    assert (dirs["patches"] / "p01.json").read_text() == '{"schema": "weightcode.patch.v0"}'
    assert (dirs["scratch"] / "note.txt").read_text() == "scratch"
    assert sorted(os.listdir(dirs["trial"])) == ["model_spec.json"]
    assert (dirs["trial"] / "model_spec.json").read_text() == '{"schema": "test"}\n'


def test_sandbox_root_and_system_paths_are_read_only_and_tmp_is_bounded(dirs: dict[str, Path]) -> None:
    with SandboxKernel(make_config(dirs, tmpfs_bytes=16 * 2**20)) as kernel:
        code = textwrap.dedent(
            """
            import errno, os
            outcome = {}
            for target in ("/probe", "/work/probe", "/usr/probe", "/dev/probe", "/home/probe", "/opt/weightcode/probe",
                           "/tmp/probe", "/dev/shm/probe"):
                try:
                    with open(target, "x") as handle:
                        handle.write("x")
                except OSError as exc:
                    outcome[target] = errno.errorcode[exc.errno]
                else:
                    os.unlink(target)
                    outcome[target] = "WRITABLE"
            try:
                open("/opt/weightcode/kernel.py", "a").close()
            except OSError as exc:
                outcome["kernel"] = errno.errorcode[exc.errno]
            try:
                with open("/tmp/big", "wb") as handle:
                    for _ in range(32):
                        handle.write(b"\\0" * 2**20)
            except OSError as exc:
                outcome["tmp_32MiB"] = errno.errorcode[exc.errno]
            else:
                outcome["tmp_32MiB"] = "WRITTEN"
            os.unlink("/tmp/big")
            outcome
            """
        )
        outcome = value_of(kernel, code)
    assert isinstance(outcome, dict)
    for target in ("/probe", "/work/probe", "/usr/probe", "/dev/probe", "/home/probe", "/opt/weightcode/probe"):
        assert outcome[target] == "EROFS", (target, outcome)
    assert outcome["kernel"] == "EROFS"
    assert outcome["/tmp/probe"] == outcome["/dev/shm/probe"] == "WRITABLE"
    assert outcome["tmp_32MiB"] == "ENOSPC"


def test_isolation_selftest_passes_and_leaves_no_trace(store: dict[str, Path], dirs: dict[str, Path]) -> None:
    config = make_config(dirs)
    report = isolation_selftest(config)
    failures = {name: report["checks"][name] for name in report["failed"]}  # type: ignore[index]
    assert report["status"] == "pass", failures
    assert set(report["checks"]) == set(INSIDE_CHECKS) | HOST_SIDE_CHECKS  # type: ignore[arg-type]
    assert not config.log_path.exists()  # probe cells never enter the session's cell log
    selftest_log = Path(report["environment"]["log_path"])  # type: ignore[index]
    assert selftest_log.parent == config.log_path.parent and len(selftest_log.read_text().splitlines()) == 3
    assert list(dirs["patches"].iterdir()) == [] and list(dirs["scratch"].iterdir()) == []
    assert sorted(os.listdir(dirs["trial"])) == ["model_spec.json"]


def test_isolation_selftest_detects_injected_leaks(
    store: dict[str, Path], dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = host_module.build_bwrap_argv

    def leaky(config: SandboxConfig, response_fd: int | str) -> list[str]:
        argv = original(config, response_fd)
        position = argv.index("--remount-ro")
        extra = [
            "--setenv", "WEIGHTCODE_LEAK", "1",
            "--ro-bind", str(store["private"]), str(store["private"]),
            "--bind", str(config.trial_dir.resolve()), "/work/trial",
        ]  # fmt: skip
        return argv[:position] + extra + argv[position:]

    monkeypatch.setattr(host_module, "build_bwrap_argv", leaky)
    report = isolation_selftest(make_config(dirs))
    assert report["status"] == "fail"
    assert set(report["failed"]) == {"env_clean", "private_invisible", "trial_readonly", "host_home_and_repo_hidden"}  # type: ignore[arg-type]
    assert sorted(os.listdir(dirs["trial"])) == ["model_spec.json"]  # the probe file was removed host-side


def test_isolation_selftest_flags_a_bind_source_inside_private(store: dict[str, Path], dirs: dict[str, Path]) -> None:
    trial = store["private"] / "instances" / "i0" / "trial"
    trial.mkdir(parents=True)
    (trial / "model_spec.json").write_text("{}\n")
    report = isolation_selftest(make_config(dirs, trial_dir=trial))
    assert report["failed"] == ["bind_sources_safe"]


def test_isolation_selftest_finds_forbidden_strings_in_bind_source_paths(store: dict[str, Path], tmp_path: Path) -> None:
    named = make_dirs(tmp_path / "sessions" / "DECOY-r1")  # a session directory named after its arm
    report = isolation_selftest(make_config(named), forbidden_strings=["DECOY"])
    assert report["failed"] == ["forbidden_strings_absent"]
    opaque = make_dirs(tmp_path / "sessions" / secrets.token_hex(8))
    assert isolation_selftest(make_config(opaque), forbidden_strings=["DECOY"])["status"] == "pass"


def test_isolation_selftest_fails_closed_without_protected_roots(dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("WEIGHTCODE_PRIVATE", "WEIGHTCODE_SEALED", "WEIGHTCODE_RUNS"):
        monkeypatch.delenv(variable, raising=False)
    report = isolation_selftest(make_config(dirs))
    assert report["status"] == "fail"
    assert set(report["failed"]) == {"private_invisible", "sealed_invisible", "runs_invisible"}  # type: ignore[arg-type]
