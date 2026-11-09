"""Patch executor for the ``weightcode.patch.v0`` format (contract sections 5.0 and 5.1, HARNESS_SPEC section 2).

One executor applies every patch, whether an LLM editor or a baseline wrote it. ``apply_patch`` validates the
whole patch first and reports every problem it finds in one ``PatchError``; it then computes the child in
float64 from the float32 parent, casts once to float32, and checks finiteness and the size limits. It never
mutates its inputs and never returns a partial result. ``load_patch`` reads a patch JSON file defensively
and ``describe_ops`` renders the editor-facing description of the format.
"""

from weightcode.patch.describe import describe_ops
from weightcode.patch.errors import PatchError
from weightcode.patch.executor import PatchResult, apply_patch, apply_patch_file
from weightcode.patch.files import MAX_PATCH_JSON_BYTES, NPY_NAME_PATTERN, is_npy_name, load_patch
from weightcode.patch.limits import Limits, task_limits
from weightcode.patch.ops import OP_NAMES, SCHEMA

__all__ = [
    "MAX_PATCH_JSON_BYTES",
    "NPY_NAME_PATTERN",
    "OP_NAMES",
    "SCHEMA",
    "Limits",
    "PatchError",
    "PatchResult",
    "apply_patch",
    "apply_patch_file",
    "describe_ops",
    "is_npy_name",
    "load_patch",
    "task_limits",
]
