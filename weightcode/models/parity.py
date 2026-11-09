"""Logit parity between a numpy forward.py and torch rebuilds of the same student (seqops and graph students).

forward.py computes in float64 and casts the logits once to float32, so against a float64 torch rebuild it agrees
within one float32 rounding step of each logit: that is the exactness test of the implementation. Against the
float32 torch model it agrees within PARITY_TOL times the logit scale max(1, max |logit|), because float32 rounding
grows with the magnitude of the values summed (an absolute bound fails a correct forward once logits are large).
"""

from __future__ import annotations

import numpy as np

from weightcode.models.export import PARITY_TOL, ParityError

FLOAT32_STEP = float(np.finfo(np.float32).eps)
"""2**-23: one float32 rounding step relative to the value. The cast of a float64 logit to float32 moves it by at
most half a step, so forward.py and a float64 rebuild agree within one step of max(1, |logit|) at every logit."""
PARITY_RULE = {
    "float32": "max |forward.py - torch cpu float32| <= PARITY_TOL * max(1, max |logit|) over the parity rows",
    "float64": "|forward.py - torch cpu float64| <= 2**-23 * max(1, |logit|) at every logit",
}


def logit_scale(logits: np.ndarray) -> float:
    """max(1, max |logit|): the magnitude float32 rounding error is relative to (absolute below 1)."""
    return max(1.0, float(np.max(np.abs(logits)))) if logits.size else 1.0


def float32_tolerance(logits: np.ndarray) -> float:
    """PARITY_TOL scaled to the logit magnitude of ``logits``: the float32 agreement bound of ``PARITY_RULE``."""
    return PARITY_TOL * logit_scale(logits)


def check_logit_parity(mine: np.ndarray, exact: np.ndarray, float32: np.ndarray) -> dict[str, float]:
    """``PARITY_RULE`` for forward.py logits ``mine`` against float64 (``exact``) and float32 torch logits.

    Returns the measured differences and bounds; raises ParityError if either bound fails.
    """
    mine, exact, float32 = (np.asarray(a, dtype=np.float64) for a in (mine, exact, float32))
    if not mine.shape == exact.shape == float32.shape:
        raise ParityError(f"forward.py logits {mine.shape}, torch {exact.shape} and {float32.shape} differ in shape")
    rel64 = float(np.max(np.abs(mine - exact) / np.maximum(1.0, np.abs(exact)))) if mine.size else 0.0
    result = {
        "max_abs_diff": float(np.max(np.abs(mine - float32))) if mine.size else 0.0,
        "tolerance": float32_tolerance(exact),
        "logit_scale": logit_scale(exact),
        "float64_max_abs_diff": float(np.max(np.abs(mine - exact))) if mine.size else 0.0,
        "float64_max_rel_diff": rel64,
        "float64_tolerance": FLOAT32_STEP,
    }
    if not rel64 <= FLOAT32_STEP:
        raise ParityError(f"forward.py differs from the float64 torch rebuild by {rel64:.3g} of max(1, |logit|) > "
                          f"{FLOAT32_STEP:.3g} (one float32 step): the forward does not implement this student")
    if not result["max_abs_diff"] <= result["tolerance"]:
        raise ParityError(f"forward.py vs torch cpu float32 max abs diff {result['max_abs_diff']:.3g} > "
                          f"{result['tolerance']:.3g} (PARITY_TOL {PARITY_TOL:g} x logit scale {result['logit_scale']:.4g})")
    return result
