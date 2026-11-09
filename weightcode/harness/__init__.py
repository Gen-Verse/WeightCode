"""Editor harness: trial assembly, the MCP tool server, the runner and the session records (docs/HARNESS_SPEC.md).

``run_session`` runs one editor session on one evidence-v1 instance; ``scripts/run_editor_session.py`` is its CLI.
"""

from weightcode.harness.arms import ARMS, arm_spec
from weightcode.harness.finalize import run_record, select_final
from weightcode.harness.instance import InstanceRefused, find_instance, list_instances, read_instance_record
from weightcode.harness.runner import SessionResult, run_session

__all__ = [
    "ARMS",
    "InstanceRefused",
    "SessionResult",
    "arm_spec",
    "find_instance",
    "list_instances",
    "read_instance_record",
    "run_record",
    "run_session",
    "select_final",
]
