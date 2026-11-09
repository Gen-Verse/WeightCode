"""Run a loop stream (PREREG E4-loop, dev only): the teacher in the loop, or control (a), checked selection.

Example (the E4-loop stream on seed 100: round 1 the v1 noise final, rounds 2 to 5 one LEARN session each)::

    source scripts/env.sh
    .venv/bin/python scripts/run_loop_stream.py --mode teacher \\
        --instances ibb1f3fc6 i7346ae21 if9258f2a i4f9f5974 i984a5043 \\
        --round1-program $WEIGHTCODE_RUNS/sessions/s8965de35e9ae7407/final/gate_v2.py \\
        --round1-session $WEIGHTCODE_RUNS/sessions/s8965de35e9ae7407 --out $WEIGHTCODE_RUNS/loop_streams

Control (a) takes ``--mode checked_selection`` and one ``--candidates ROUND=PROGRAM`` per frozen final, rounds 2 to 5,
in check order. A stream that stopped resumes with ``--resume`` from its next round; a round whose session finished
is not launched again (``weightcode.harness.loop_stream``).

PREREG E10's stream (seed 100, the four conditions in the E10 order; round 1 the gaussian_noise B session).
``--round1-program`` is omitted when that session kept the base, and control (a) gives ``ROUND=none`` for a round
whose B session kept the base::

    .venv/bin/python scripts/run_loop_stream.py --mode teacher \\
        --instances $(.venv/bin/python -m weightcode.harness.e10 ids) --round1-session $WEIGHTCODE_RUNS/sessions/<gaussian B> \\
        [--round1-program <its final program>] --out $WEIGHTCODE_RUNS/loop_streams
    .venv/bin/python scripts/run_loop_stream.py --mode checked_selection \\
        --instances $(.venv/bin/python -m weightcode.harness.e10 ids) --round1-session $WEIGHTCODE_RUNS/sessions/<gaussian B> \\
        [--round1-program <its final program>] --candidates 2=<defocus B final|none> 3=<snow B final|none> \\
        4=<contrast B final|none> --out $WEIGHTCODE_RUNS/loop_streams

``--kept-rule loop-retain-v2`` replaces loop-retain-v1's per-round clean floor (-12) by the stream's clean budget: -60
of the 4000 clean V_check rows over the whole stream, round 1 counted (``weightcode.harness.loop.loop_kept_rule_v2``). A
candidate given as a session directory (``ROUND=<session>``) is that session's kept program, or no candidate when its
final is the base. On E11 instances the session wall limit defaults to 120 min (PREREG E11). The E11 two-round stream
(seed 100: gaussian_noise_s5 then contrast_s5, round 1 the gaussian B final)::

    .venv/bin/python scripts/run_loop_stream.py --mode teacher --kept-rule loop-retain-v2 \\
        --instances i00802425 i3eb3fc78 --round1-session $WEIGHTCODE_RUNS/sessions/sc918e11dceef637a \\
        --round1-program $WEIGHTCODE_RUNS/sessions/sc918e11dceef637a/final/gate3_lower.py --out $WEIGHTCODE_RUNS/loop_streams

Round 1 reruns its program on the base by default. ``--round1-from-final`` instead applies the round-1 session's saved
final (``final/patch.json``, checked against the final's hashes), and ``--round1-from-record <loop record>`` takes
round 1's student of that stream (a finished record, or the ``state.json`` of a running one), so a control shares a
teacher stream's exact round-1 student. Both need ``--round1-program``, and the record names the source
(``round1.source``). Control (a) on the E10 v2 teacher stream's round-1 student::

    .venv/bin/python scripts/run_loop_stream.py --mode checked_selection --kept-rule loop-retain-v2 \\
        --instances $(.venv/bin/python -m weightcode.harness.e10 ids) --round1-session $WEIGHTCODE_RUNS/sessions/<gaussian B> \\
        --round1-program <its final program> --round1-from-record $WEIGHTCODE_RUNS/loop_streams/<teacher stream>/state.json \\
        --candidates 2=<defocus B final|none> 3=<snow B final|none> 4=<contrast B final|none> \\
        --out $WEIGHTCODE_RUNS/loop_streams

PREREG T16 (the E11 two-round stream with and without the edit log): ``--edit-log`` gives the round-2 session
edit_log.json and the one PREREG sentence (arm L); without it the session is the stream of record (arm N). ``--tag``
gives each session's stream its own stem, and ``--api-key-file`` bills the session to that key file (NSCC, R33;
scripts/nscc_session/submit_stream.sh)::

    .venv/bin/python scripts/run_loop_stream.py --mode teacher --kept-rule loop-retain-v2 \\
        --instances i00802425 i3eb3fc78 --round1-session $WEIGHTCODE_RUNS/sessions/sc918e11dceef637a \\
        --round1-program $WEIGHTCODE_RUNS/sessions/sc918e11dceef637a/final/gate3_lower.py --round1-from-final \\
        --effort max --tag t16_l1 --edit-log --out $WEIGHTCODE_RUNS/loop_streams

T16 follow-up M (the protected-entry limit): ``--protect-logged``, with ``--edit-log`` only, adds a limit that refuses
every update changing an entry edit_log.json lists, applied with the session's other limits, and one sentence after
the edit-log sentence (``weightcode.harness.loop``); the same command as arm L with ``--protect-logged`` and its own
``--tag`` (e.g. t16_m1).
"""

from __future__ import annotations

from weightcode.harness.loop_stream import main

if __name__ == "__main__":
    raise SystemExit(main())
