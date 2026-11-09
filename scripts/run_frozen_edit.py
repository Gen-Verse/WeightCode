"""PREREG F6: add a LEARN session's final deltas unchanged to other students and read each as E3 reads a rerun.

Example::

    source scripts/env.sh
    .venv/bin/python scripts/run_frozen_edit.py run --session $WEIGHTCODE_RUNS/sessions/<id> --instances <dev id> ... \\
        --out $WEIGHTCODE_RUNS/f6/<id>
    .venv/bin/python scripts/run_frozen_edit.py compare --f6 $WEIGHTCODE_RUNS/f6/<id> \\
        --e3 $WEIGHTCODE_RUNS/devreadouts/learn/<id>/transfer

The session's final/deltas.npz is checked against its records and against model.npz minus the base, then added to
each target under the target's LEARN limits and read on V_select of the condition and its clean companion
(``weightcode.harness.frozen_edit``). One JSON record per target; records are never overwritten.
"""

from __future__ import annotations

from weightcode.harness.frozen_edit import main

if __name__ == "__main__":
    raise SystemExit(main())
