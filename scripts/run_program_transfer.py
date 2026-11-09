"""Run a frozen LEARN update program on other students and read out each update (docs/HARNESS_SPEC.md section 5).

Example::

    source scripts/env.sh
    .venv/bin/python scripts/run_program_transfer.py --program $WEIGHTCODE_RUNS/sessions/<id>/final/update.py \\
        --instances <dev id> <dev id> --out $WEIGHTCODE_RUNS/transfer/<name>

For each target dev instance the experience is assembled exactly as LEARN trial assembly does, the program runs once in
the LEARN program sandbox, and one JSON record is written per run: the within-limits flag (with the unbounded child's
readout when the limits refuse the update) and the readouts on V_select of the condition and on its clean companion.
V_check and the sealed store are never read; main instances are refused (``weightcode.harness.transfer``).
"""

from __future__ import annotations

from weightcode.harness.transfer import main

if __name__ == "__main__":
    raise SystemExit(main())
