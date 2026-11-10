"""Editor sandbox: a persistent Python kernel inside bubblewrap (docs/HARNESS_SPEC.md section 3).

``SandboxKernel`` runs cells; ``isolation_selftest`` must pass before every session.
"""

from weightcode.sandbox.host import (
    CELL_LOG_SCHEMA,
    DEFAULT_BWRAP,
    KERNEL_MOUNT,
    KERNEL_SCRIPT,
    MIN_BWRAP_VERSION,
    PATCHES_MOUNT,
    PROTOCOL,
    SCRATCH_MOUNT,
    STATE_LOST_NOTE,
    TRIAL_MOUNT,
    CellResult,
    InterpreterMounts,
    SandboxConfig,
    SandboxError,
    SandboxKernel,
    build_bwrap_argv,
    bwrap_version,
    check_bwrap,
    interpreter_mounts,
    kernel_sha256,
    recover_interrupted_cell,
    sandbox_environment,
    venv_root,
)
from weightcode.sandbox.selftest import SELFTEST_SCHEMA, isolation_selftest

__all__ = [
    "CELL_LOG_SCHEMA",
    "DEFAULT_BWRAP",
    "KERNEL_MOUNT",
    "KERNEL_SCRIPT",
    "MIN_BWRAP_VERSION",
    "PATCHES_MOUNT",
    "PROTOCOL",
    "SCRATCH_MOUNT",
    "SELFTEST_SCHEMA",
    "STATE_LOST_NOTE",
    "TRIAL_MOUNT",
    "CellResult",
    "InterpreterMounts",
    "SandboxConfig",
    "SandboxError",
    "SandboxKernel",
    "build_bwrap_argv",
    "bwrap_version",
    "check_bwrap",
    "interpreter_mounts",
    "isolation_selftest",
    "kernel_sha256",
    "recover_interrupted_cell",
    "sandbox_environment",
    "venv_root",
]
