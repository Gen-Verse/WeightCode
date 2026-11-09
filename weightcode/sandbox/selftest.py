"""Isolation self-test for the editor sandbox. The runner calls it before every session and aborts on failure.

``isolation_selftest(config)`` starts a fresh kernel with the session's own mounts, runs a fixed set of probe
cells inside it, adds host-side confirmations, and returns one pass or fail entry per check. Every check fails
closed: a probe that raises, does not report, or cannot be evaluated (for example because ``$WEIGHTCODE_PRIVATE`` is
unset) is a failure. The probe cells are logged to their own file next to ``config.log_path``, never to the
session's cell log, so they use none of the editor's cell budget.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import secrets
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from weightcode.common.paths import PRIVATE_VAR, RUNS_VAR, SEALED_VAR, private_dir, runs_dir, sealed_dir
from weightcode.sandbox.host import (
    KERNEL_MOUNT,
    KERNEL_SCRIPT,
    PATCHES_MOUNT,
    SCRATCH_MOUNT,
    TRIAL_MOUNT,
    SandboxConfig,
    SandboxError,
    SandboxKernel,
    _is_within,
    build_bwrap_argv,
    bwrap_version,
    kernel_sha256,
    sandbox_environment,
    venv_root,
)

SELFTEST_SCHEMA = "weightcode.sandbox.selftest.v0"
SELFTEST_CELL_TIMEOUT_S = 60.0  # probe cells need a few seconds; the session's own timeout may be shorter
SELFTEST_OUTPUT_CHARS = 2**20  # the probe report must come back whole, whatever the session's output limit
FORBIDDEN_MODULES = ("scipy", "torch", "sklearn", "jax", "tensorflow", "autograd", "weightcode")
TCP_PROBES = (("1.1.1.1", 443), ("2606:4700:4700::1111", 443))  # public anycast resolvers: reachable with network
DNS_PROBE = "example.com"
OPTIONAL_SANDBOX_ENV = ("PWD", "LC_CTYPE")  # set by bwrap --chdir and by Python's C-locale coercion
REPORT_PREFIX = "WEIGHTCODE_SELFTEST "

# Checks the probe cell must report; a missing one fails.
INSIDE_CHECKS = (
    "nested_userns_blocked",
    "tcp_blocked",
    "dns_blocked",
    "network_namespace_isolated",
    "claude_config_invisible",
    "host_home_and_repo_hidden",
    "private_invisible",
    "sealed_invisible",
    "runs_invisible",
    "host_session_paths_invisible",
    "etc_passwd_absent",
    "env_clean",
    "trial_readonly",
    "patches_writable",
    "scratch_writable",
    "system_paths_readonly",
    "tmpfs_bounded",
    "rlimits_applied",
    "no_capabilities",
    "pid_namespace",
    "interpreter_isolated",
    *(f"{module}_absent" for module in FORBIDDEN_MODULES),
    "numpy_present",
)

_PROBE_CELL = r'''
def _weightcode_selftest_probe(params):
    import errno, importlib.util, json, os, resource, shutil, socket, sys

    results = {}

    def check(name, function):
        try:
            passed, detail = function()
        except BaseException as exc:
            passed, detail = False, f"probe raised {type(exc).__name__}: {exc}"
        results[name] = {"status": "pass" if passed else "fail",
                         "detail": detail if isinstance(detail, str) else json.dumps(detail, sort_keys=True)}

    def nested_userns():
        pid = os.fork()
        if pid == 0:
            try:
                os.unshare(os.CLONE_NEWUSER)
            except OSError:
                os._exit(17)
            os._exit(0)
        _, status = os.waitpid(pid, 0)
        code = os.waitstatus_to_exitcode(status)
        return code == 17, {17: "unshare(CLONE_NEWUSER) refused", 0: "unshare(CLONE_NEWUSER) succeeded"}.get(code, f"child exited {code}")

    def tcp():
        outcomes = {}
        for host, port in params["tcp_probes"]:
            try:
                connection = socket.create_connection((host, port), timeout=3)
            except OSError as exc:
                outcomes[f"{host}:{port}"] = f"failed: {type(exc).__name__}: {exc}"
            else:
                connection.close()
                outcomes[f"{host}:{port}"] = "CONNECTED"
        return "CONNECTED" not in outcomes.values(), outcomes

    def dns():
        try:
            infos = socket.getaddrinfo(params["dns_probe"], 443)
        except OSError as exc:
            return True, f"lookup failed: {type(exc).__name__}: {exc}"
        return False, f"resolved to {sorted({info[4][0] for info in infos})}"

    def interfaces():
        with open("/proc/net/dev") as handle:
            names = sorted(line.split(":")[0].strip() for line in handle.read().splitlines()[2:])
        return names == ["lo"], names

    def absent(paths):
        def run():
            visible = [path for path in paths if os.path.lexists(path)]
            return not visible, f"visible: {visible}" if visible else f"not visible: {paths}"
        return run

    def protected(name):
        def run():
            entry = params["protected"][name]
            if entry["error"]:
                return False, entry["error"]
            return absent(entry["paths"])()
        return run

    def home_and_repo():
        problems = []
        parts = params["venv"].strip("/").split("/")
        for depth in range(1, len(parts)):
            ancestor = "/" + "/".join(parts[:depth])
            if ancestor == "/usr" or ancestor.startswith("/usr/"):
                continue
            entries = sorted(os.listdir(ancestor))
            if entries != [parts[depth]]:
                problems.append(f"{ancestor} lists {entries}")
        home = params["home"].rstrip("/")
        if os.path.lexists(home) and not params["venv"].startswith(home + "/"):
            problems.append(f"{home} is visible")
        return not problems, problems or "only the directories leading to the venv are visible, each with one entry"

    def environment():
        expected, optional = params["env"], set(params["env_optional"])
        extra = sorted(set(os.environ) - set(expected) - optional)
        wrong = sorted(key for key, value in expected.items() if os.environ.get(key) != value)
        weightcode = sorted(key for key, value in os.environ.items() if "WEIGHTCODE" in key.upper() or "WEIGHTCODE" in value.upper())
        return not (extra or wrong or weightcode), {"unexpected": extra, "wrong_value": wrong, "mentions_weightcode": weightcode}

    def trial():
        target = params["trial_mount"] + "/" + params["probe_name"]
        try:
            with open(target, "x") as handle:
                handle.write("probe")
        except OSError as exc:
            same = sorted(os.listdir(params["trial_mount"])) == params["trial_listing"]
            refused = exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM)
            return refused and same, f"write refused with {errno.errorcode.get(exc.errno, exc.errno)}; listing matches the host: {same}"
        return False, f"created {target}"

    def writable(mount):
        def run():
            target = mount + "/" + params["probe_name"]
            with open(target, "x") as handle:
                handle.write(params["nonce"])
            return True, f"created {target}"
        return run

    def system_paths():
        outcome = {}
        for directory in ("/", "/work", "/usr", "/dev", "/opt/weightcode", "/home"):
            target = directory.rstrip("/") + "/" + params["probe_name"]
            try:
                open(target, "x").close()
            except OSError as exc:
                outcome[directory] = errno.errorcode.get(exc.errno, str(exc.errno))
            else:
                os.unlink(target)
                outcome[directory] = "WRITABLE"
        try:
            open(params["kernel_mount"], "a").close()
        except OSError as exc:
            outcome[params["kernel_mount"]] = errno.errorcode.get(exc.errno, str(exc.errno))
        else:
            outcome[params["kernel_mount"]] = "WRITABLE"
        return "WRITABLE" not in outcome.values(), outcome

    def tmpfs():
        sizes = {path: shutil.disk_usage(path).total for path in ("/tmp", "/dev/shm")}
        limit = params["tmpfs_bytes"] + resource.getpagesize()
        return all(0 < size <= limit for size in sizes.values()), sizes

    def rlimits():
        address = resource.getrlimit(resource.RLIMIT_AS)
        core = resource.getrlimit(resource.RLIMIT_CORE)
        cap = params["memory_bytes"]
        return address == (cap, cap) and core == (0, 0), {"RLIMIT_AS": list(address), "RLIMIT_CORE": list(core)}

    def capabilities():
        with open("/proc/self/status") as handle:
            caps = {line.split(":")[0]: int(line.split()[1], 16) for line in handle if line.startswith("Cap")}
        return bool(caps) and not any(caps.values()), {key: hex(value) for key, value in caps.items()}

    def pid_namespace():
        with open("/proc/1/cmdline", "rb") as handle:
            init = handle.read().split(b"\0")[0].decode()
        return os.path.basename(init) == "bwrap", f"pid 1 is {init}; this process is pid {os.getpid()}"

    def interpreter():
        flags = {"isolated": sys.flags.isolated, "no_user_site": sys.flags.no_user_site,
                 "dont_write_bytecode": sys.dont_write_bytecode, "version": sys.version.split()[0]}
        return bool(sys.flags.isolated and sys.flags.no_user_site and sys.dont_write_bytecode), flags

    def module_absent(name):
        def run():
            spec = importlib.util.find_spec(name)
            return spec is None, "not importable" if spec is None else f"importable from {spec.origin}"
        return run

    def numpy_present():
        import numpy
        return True, numpy.__version__

    check("nested_userns_blocked", nested_userns)  # first: fork before any thread exists
    check("tcp_blocked", tcp)
    check("dns_blocked", dns)
    check("network_namespace_isolated", interfaces)
    check("claude_config_invisible", absent([params["claude_dir"]]))
    check("host_home_and_repo_hidden", home_and_repo)
    for name in ("private", "sealed", "runs"):
        check(f"{name}_invisible", protected(name))
    check("host_session_paths_invisible", absent(params["host_paths"]))
    check("etc_passwd_absent", absent(["/etc/passwd", "/etc"]))
    check("env_clean", environment)
    check("trial_readonly", trial)
    check("patches_writable", writable(params["patches_mount"]))
    check("scratch_writable", writable(params["scratch_mount"]))
    check("system_paths_readonly", system_paths)
    check("tmpfs_bounded", tmpfs)
    check("rlimits_applied", rlimits)
    check("no_capabilities", capabilities)
    check("pid_namespace", pid_namespace)
    check("interpreter_isolated", interpreter)
    for name in params["forbidden_modules"]:
        check(f"{name}_absent", module_absent(name))
    check("numpy_present", numpy_present)  # last: importing numpy starts BLAS threads

    if params["forbidden_strings"]:
        def forbidden():
            with open("/proc/self/mountinfo") as handle:
                haystacks = {"mountinfo": handle.read()}
            haystacks["environ"] = json.dumps(dict(os.environ))
            haystacks["trial_listing"] = " ".join(os.listdir(params["trial_mount"]))
            hits = sorted(f"{needle!r} in {where}" for needle in params["forbidden_strings"]
                          for where, text in haystacks.items() if needle in text)
            return not hits, hits or "none found"
        check("forbidden_strings_absent", forbidden)

    sources = {}
    with open("/proc/self/mountinfo") as handle:
        for line in handle:
            fields = line.split()
            if fields[4].startswith(("/work", "/opt")) or fields[4] == params["venv"]:
                sources[fields[4]] = fields[3]
    return {"checks": results, "info": {"mount_sources": sources, "hostname": os.uname().nodename}}

print("WEIGHTCODE_SELFTEST " + __import__("json").dumps(_weightcode_selftest_probe(__import__("json").loads(__WEIGHTCODE_PARAMS__))))
'''

_PROTOCOL_CELL = r"""
import os
os.write(1, b'{"type": "result", "id": "WEIGHTCODE_FORGED", "index": -1, "repr": null, "error": null}\n')
os.write(2, b'\x00\xff stray-fd2 bytes\n')
print("WEIGHTCODE_PROTOCOL_OK")
"""


def isolation_selftest(
    config: SandboxConfig,
    *,
    log_path: Path | None = None,
    forbidden_strings: Sequence[str] = (),
) -> dict[str, object]:
    """Run the fixed isolation probes in a fresh kernel with ``config``'s mounts; one pass or fail per check.

    ``forbidden_strings`` (for example the arm name) must not appear in what the sandbox can read about its
    own mounts and environment: /proc/self/mountinfo shows the host path of every bind source, so a session
    directory named after its arm would tell the editor its arm. The returned dict has ``status`` ("pass" only
    if every check passed), ``failed``, ``checks`` ({name: {"status", "detail"}}) and ``environment``.
    A configuration that ``SandboxKernel`` rejects outright (missing bwrap or venv, overlapping directories, a
    log inside a bound directory) raises ``SandboxError`` instead of returning a report.
    """
    began = time.monotonic()
    nonce = secrets.token_hex(8)
    probe_name = f".weightcode_selftest_{nonce}"
    checks: dict[str, dict[str, str]] = {}

    def record(name: str, passed: bool, detail: object) -> None:
        text = detail if isinstance(detail, str) else json.dumps(detail, sort_keys=True, default=str)
        checks[name] = {"status": "pass" if passed else "fail", "detail": text}

    protected = _protected_roots()
    record("bind_sources_safe", *_bind_sources_safe(config, protected))
    if log_path is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = Path(config.log_path).with_name(f"selftest_{stamp}_{nonce}.jsonl")
    probe_config = dataclasses.replace(
        config,
        log_path=log_path,
        max_cells=8,
        cell_timeout_s=SELFTEST_CELL_TIMEOUT_S,
        max_output_chars=max(config.max_output_chars, SELFTEST_OUTPUT_CHARS),
    )
    environment: dict[str, object] = {
        "bwrap_version": bwrap_version(config.bwrap),
        "kernel_sha256": kernel_sha256(),
        "argv": build_bwrap_argv(config, "<response-fd>"),
        "log_path": str(log_path),
    }
    params = _probe_params(config, protected, probe_name, nonce, forbidden_strings)
    kernel = SandboxKernel(probe_config)
    try:
        try:
            kernel.start()
        except SandboxError as exc:
            record("kernel_start", False, str(exc))
        else:
            record("kernel_start", True, f"ready: python {kernel.ready_info.get('python')}")
            _run_probes(kernel, params, config, probe_name, nonce, record, checks, environment)
    finally:
        kernel.close()
        environment["probe_files_removed"] = _remove_probe_files(config, probe_name)
    failed = sorted(name for name, entry in checks.items() if entry["status"] != "pass")
    return {
        "schema": SELFTEST_SCHEMA,
        "status": "fail" if failed else "pass",
        "failed": failed,
        "checks": checks,
        "environment": environment,
        "duration_s": round(time.monotonic() - began, 3),
    }


def _run_probes(
    kernel: SandboxKernel,
    params: dict[str, object],
    config: SandboxConfig,
    probe_name: str,
    nonce: str,
    record: Callable[[str, bool, object], None],
    checks: dict[str, dict[str, str]],
    environment: dict[str, object],
) -> None:
    result = kernel.run(_PROBE_CELL.replace("__WEIGHTCODE_PARAMS__", repr(json.dumps(params))))
    report = _parse_report(result.stdout)
    expected = list(INSIDE_CHECKS) + (["forbidden_strings_absent"] if params["forbidden_strings"] else [])
    if report is None:
        for name in expected:
            record(name, False, f"the probe cell did not report (error: {result.error!r}, stderr: {result.stderr[-500:]!r})")
    else:
        reported = report.get("checks", {})
        for name in expected:
            entry = reported.get(name)
            if isinstance(entry, dict) and entry.get("status") in ("pass", "fail"):
                checks[name] = {"status": str(entry["status"]), "detail": str(entry.get("detail", ""))}
            else:
                record(name, False, "the probe cell did not report this check")
        environment["inside"] = report.get("info", {})
    for name, directory in (("patches", config.patches_dir), ("scratch", config.scratch_dir)):
        host_file = Path(directory) / probe_name
        seen = host_file.is_file() and not host_file.is_symlink() and host_file.read_text() == nonce
        record(f"{name}_bind_is_session_dir", seen,
               f"the file written inside {'appeared' if seen else 'did not appear'} at {host_file}")
    forged = kernel.run(_PROTOCOL_CELL)
    after = kernel.run("6 * 7")
    intact = (
        forged.error is None
        and not forged.restarted
        and "WEIGHTCODE_FORGED" in forged.stdout
        and "WEIGHTCODE_PROTOCOL_OK" in forged.stdout
        and "stray-fd2" in forged.stderr
        and after.error is None
        and not after.restarted
        and after.stdout == "42\n"
    )
    record("protocol_survives_fd_writes", intact,
           {"forged_cell": {"stdout": forged.stdout[-300:], "stderr": forged.stderr[-300:], "error": forged.error,
                            "restarted": forged.restarted},
            "next_cell": {"stdout": after.stdout, "error": after.error, "restarted": after.restarted}})


def _parse_report(stdout: str) -> dict[str, object] | None:
    for line in stdout.splitlines():
        if line.startswith(REPORT_PREFIX):
            try:
                report = json.loads(line[len(REPORT_PREFIX):])
            except ValueError:
                return None
            return report if isinstance(report, dict) else None
    return None


def _protected_roots() -> dict[str, dict[str, object]]:
    """$WEIGHTCODE_PRIVATE, $WEIGHTCODE_SEALED and $WEIGHTCODE_RUNS as given and as resolved; an unset variable is recorded as an error."""
    roots: dict[str, dict[str, object]] = {}
    for name, variable, resolve in (
        ("private", PRIVATE_VAR, private_dir),
        ("sealed", SEALED_VAR, sealed_dir),
        ("runs", RUNS_VAR, runs_dir),
    ):
        try:
            resolved = resolve()
        except RuntimeError as exc:
            roots[name] = {"variable": variable, "paths": [], "error": str(exc)}
            continue
        given = Path(os.path.abspath(os.path.expanduser(os.environ[variable])))
        roots[name] = {"variable": variable, "paths": sorted({str(given), str(resolved)}), "error": None}
    return roots


def _bind_sources_safe(config: SandboxConfig, protected: dict[str, dict[str, object]]) -> tuple[bool, str]:
    """No bind source may contain a protected root, or lie inside $WEIGHTCODE_PRIVATE or $WEIGHTCODE_SEALED."""
    sources = {
        "venv": venv_root(config.python).resolve(),
        "trial": Path(config.trial_dir).resolve(),
        "patches": Path(config.patches_dir).resolve(),
        "scratch": Path(config.scratch_dir).resolve(),
        "kernel": KERNEL_SCRIPT,
    }
    problems = []
    for name, entry in protected.items():
        for raw in entry["paths"]:  # type: ignore[union-attr]
            root = Path(str(raw)).resolve()
            for source_name, source in sources.items():
                if _is_within(root, source):
                    problems.append(f"the {source_name} bind {source} contains ${entry['variable']} ({root})")
                if name != "runs" and _is_within(source, root):
                    problems.append(f"the {source_name} bind {source} lies inside ${entry['variable']} ({root})")
    detail = "; ".join(sorted(set(problems))) or (
        "no bind source contains $WEIGHTCODE_PRIVATE, $WEIGHTCODE_SEALED or $WEIGHTCODE_RUNS or lies inside the first two"
    )
    return not problems, detail


def _probe_params(
    config: SandboxConfig,
    protected: dict[str, dict[str, object]],
    probe_name: str,
    nonce: str,
    forbidden_strings: Sequence[str],
) -> dict[str, object]:
    home = Path.home()
    log_dir = Path(os.path.abspath(config.log_path)).parent
    host_paths = set()
    for path in (config.trial_dir, config.patches_dir, config.scratch_dir, log_dir, KERNEL_SCRIPT.parent):
        host_paths.add(os.path.abspath(path))
        host_paths.add(str(Path(path).resolve()))
    return {
        "nonce": nonce,
        "probe_name": probe_name,
        "venv": str(venv_root(config.python)),
        "home": str(home),
        "claude_dir": str(home / ".claude"),
        "protected": {name: {"paths": entry["paths"], "error": entry["error"]} for name, entry in protected.items()},
        "host_paths": sorted(host_paths),
        "env": sandbox_environment(config),
        "env_optional": list(OPTIONAL_SANDBOX_ENV),
        "trial_mount": TRIAL_MOUNT,
        "patches_mount": PATCHES_MOUNT,
        "scratch_mount": SCRATCH_MOUNT,
        "kernel_mount": KERNEL_MOUNT,
        "trial_listing": sorted(os.listdir(config.trial_dir)),
        "tmpfs_bytes": config.tmpfs_bytes,
        "memory_bytes": config.memory_bytes,
        "tcp_probes": [list(probe) for probe in TCP_PROBES],
        "dns_probe": DNS_PROBE,
        "forbidden_modules": list(FORBIDDEN_MODULES),
        "forbidden_strings": list(forbidden_strings),
    }


def _remove_probe_files(config: SandboxConfig, probe_name: str) -> list[str]:
    """Delete probe files from the host side so the editor never sees them; return what was removed."""
    removed = []
    for directory in (config.trial_dir, config.patches_dir, config.scratch_dir):
        target = Path(directory) / probe_name
        if os.path.lexists(target):
            target.unlink()
            removed.append(str(target))
    return removed
