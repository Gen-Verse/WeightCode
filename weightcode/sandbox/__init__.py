"""Editor sandbox: a persistent Python kernel inside bubblewrap (docs/HARNESS_SPEC.md section 3).

``SandboxKernel`` runs cells; ``isolation_selftest`` must pass before every session.
"""

from weightcode.sandbox.host import (
    CELL_LOG_SCHEMA,
    KERNEL_MOUNT,
    KERNEL_SCRIPT,
    PATCHES_MOUNT,
    PROTOCOL,
    SCRATCH_MOUNT,
    STATE_LOST_NOTE,
    TRIAL_MOUNT,
    CellResult,
    SandboxConfig,
    SandboxError,
    SandboxKernel,
    build_bwrap_argv,
    bwrap_version,
    kernel_sha256,
    recover_interrupted_cell,
    sandbox_environment,
    venv_root,
)
from weightcode.sandbox.selftest import SELFTEST_SCHEMA, isolation_selftest

__all__ = [
    "CELL_LOG_SCHEMA",
    "KERNEL_MOUNT",
    "KERNEL_SCRIPT",
    "PATCHES_MOUNT",
    "PROTOCOL",
    "SCRATCH_MOUNT",
    "SELFTEST_SCHEMA",
    "STATE_LOST_NOTE",
    "TRIAL_MOUNT",
    "CellResult",
    "SandboxConfig",
    "SandboxError",
    "SandboxKernel",
    "build_bwrap_argv",
    "bwrap_version",
    "isolation_selftest",
    "kernel_sha256",
    "recover_interrupted_cell",
    "sandbox_environment",
    "venv_root",
]
