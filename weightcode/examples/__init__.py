"""Paper examples that anyone can reproduce from public data, with no language model and no API key.

``scan`` downloads a published network and the program a language model wrote for it, reruns the program, checks
that the edited network is the published one bit for bit, and reads both networks on the paper's held-out test set.
``imagenet`` reads the published original and edited networks (checked by hash) on the first held-out test images and
compares every image with the paper's record. Run them with ``weightcode reproduce <example>``.
"""

from __future__ import annotations

EXAMPLES: dict[str, str] = {
    "scan": "weightcode.examples.scan",
    "imagenet": "weightcode.examples.imagenet",
}
"""Example name -> module with ``TITLE`` and ``run(out, runner, ...)``."""

__all__ = ["EXAMPLES"]
