"""Draw the E7 new classes once (rule graphnew-classes-v1, ``weightcode.data.arxiv_classes``) and print the record.

Reads only the class counts of ``$WEIGHTCODE_PRIVATE/ogbn_arxiv_ind/dataset.json``; writes the sidecar
``graphnew_classes.json`` next to it, or checks an existing one against a fresh draw.

Usage:
    source scripts/env.sh
    .venv/bin/python scripts/arxiv_graphnew_classes.py
"""

from __future__ import annotations

import json
import sys

from weightcode.data.arxiv_classes import write_class_draw


def main() -> int:
    print(json.dumps(write_class_draw(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
