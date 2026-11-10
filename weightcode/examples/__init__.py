"""Paper examples that anyone can reproduce from public data, with no language model and no API key.

Each example downloads a published network and the program a language model wrote for it, reruns the program, checks
that the edited network is the published one bit for bit, and reads both networks on the paper's held-out test set.
Run them with ``weightcode reproduce <example>``.
"""

from __future__ import annotations

EXAMPLES: dict[str, str] = {
    "scan": "weightcode.examples.scan",
}
"""Example name -> module with ``TITLE`` and ``run(out, runner, ...)``."""

__all__ = ["EXAMPLES"]
