"""Run a stream of frozen LEARN update programs on one student, round after round (PREREG E4, HARNESS_SPEC section 5).

Example (the E4 stream on seed 100, one program per round in the order noise, contrast, blur, gamma, colormix)::

    source scripts/env.sh
    .venv/bin/python scripts/run_program_stream.py \\
        --programs <noise.py> <contrast.py> <blur.py> <gamma.py> <colormix.py> \\
        --instances ibb1f3fc6 i7346ae21 if9258f2a i4f9f5974 i984a5043 --out $WEIGHTCODE_RUNS/streams/<name>

Each round draws the current student's own evidence-v1 sample on its condition, adds the replay rows with the current
student's predictions, runs its program once in the LEARN program sandbox and applies the update under the LEARN limits.
After every round all the stream's conditions and their clean companions are read out on V_select. V_check and the
sealed store are never read; main instances are refused (``weightcode.harness.program_stream``).
"""

from __future__ import annotations

from weightcode.harness.program_stream import main

if __name__ == "__main__":
    raise SystemExit(main())
