"""The retention-constrained kept rule of a program arm (LEARN v2, dev only; docs/DEV_RESULTS.md, PREREG section 0).

v1, a session without ``kept_rule`` in session.json: the kept program is the argmax of S_joint = score_new + score_old
over the checked programs and the base model (``weightcode.harness.finalize.FINAL_RULE_JOINT``).

``joint-retain-v1``: the same rule over the checked programs whose score_old is at least ``s_old_floor``, with
``s_old_floor = -round(frac * n_old)``, where ``n_old`` is the number of V_check rows behind score_old: the clean
companion's rows on an evidence-v1 instance, the old-class rows on a fewshot-v1 instance. With frac 0.01 that is
-40 of 4000 clean rows and -20 of 2000 old-class rows, the -1.0 point floor of the E1 and E2 gates. The rule is
stated to the teacher in the system prompt (``weightcode.harness.prompts.render_system``) and in the program arm's
``declare_final`` description (``weightcode.harness.tools.tool_specs``); ties and the base are as in v1.

The floor can also be given as a row count (``kept_rule_record_rows``, PREREG E7): the E7 floor is the nearest
integer of a formula in R and the references' d, computed outside the harness, so the session takes the number itself,
an int <= 0, and records ``floor_kind: "rows"`` and the optional ``floor_source`` (for example the formula's inputs)
instead of ``frac``. The rule and the way the system prompt states it are the same; only how the floor is given
differs. A frac record has no ``floor_kind`` and is unchanged. A ``floor_source`` with ``"placeholder": true`` marks a
stand-in floor (a dry run before the floor is known), which ``session_floor`` refuses outside a dry run.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

KEPT_RULE_RETAIN = "joint-retain-v1"


class RetentionRuleError(ValueError):
    """A kept-rule record or argument that is not the retention rule as defined here."""


def s_old_floor(frac: float, n_old: int) -> int:
    """The score_old floor for a fraction ``frac`` of ``n_old`` rows: ``-round(frac * n_old)``, at most -1."""
    if not 0.0 < float(frac) < 1.0:
        raise RetentionRuleError(f"the retention floor fraction must lie in (0, 1), not {frac!r}")
    if int(n_old) <= 0:
        raise RetentionRuleError(f"the retention floor needs a positive row count, not {n_old!r}")
    floor = -int(round(float(frac) * int(n_old)))
    if floor > -1:
        raise RetentionRuleError(f"a floor of {floor} on {n_old} rows keeps no loss at all; use a larger fraction")
    return floor


def kept_rule_record(frac: float, n_old: int, check_set: str) -> dict[str, Any]:
    """The session.json ``kept_rule`` entry of a retention-constrained session."""
    return {"name": KEPT_RULE_RETAIN, "frac": float(frac), "n_old": int(n_old), "check_set": check_set,
            "s_old_floor": s_old_floor(frac, n_old)}


FLOOR_KIND_ROWS = "rows"


def check_floor_rows(floor: object, n_old: int | None = None) -> int:
    """A floor given as a row count: an int (not a bool), at most 0, and with ``n_old`` at least ``-n_old``."""
    if isinstance(floor, bool) or not isinstance(floor, int):
        raise RetentionRuleError(f"a floor in rows is an int, not {floor!r}")
    if floor > 0:
        raise RetentionRuleError(f"a floor in rows is at most 0 (never above 0), not {floor}")
    if n_old is not None:
        if int(n_old) <= 0:
            raise RetentionRuleError(f"the retention floor needs a positive row count, not {n_old!r}")
        if floor < -int(n_old):
            raise RetentionRuleError(f"a floor in rows lies in -{int(n_old)} .. 0 on {n_old} rows, not {floor}")
    return int(floor)


def is_placeholder(source: object) -> bool:
    """True for a ``floor_source`` that marks its floor as a dry-run stand-in."""
    return isinstance(source, Mapping) and source.get("placeholder") is True


def kept_rule_record_rows(floor_rows: int, n_old: int, check_set: str,
                          source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The session.json ``kept_rule`` entry of a retention session whose floor is given in rows (module docstring);
    ``source`` (a JSON object, or None) says where the number comes from and is copied as ``floor_source``."""
    if source is not None and not isinstance(source, Mapping):
        raise RetentionRuleError(f"a floor source is a JSON object, not {type(source).__name__}")
    return {"name": KEPT_RULE_RETAIN, "s_old_floor": check_floor_rows(floor_rows, n_old), "n_old": int(n_old),
            "check_set": check_set, "floor_kind": FLOOR_KIND_ROWS,
            "floor_source": None if source is None else dict(source)}


def session_floor(session: Mapping[str, Any]) -> int | None:
    """The score_old floor of a session record, or None for a v1 session (no ``kept_rule``)."""
    rule = session.get("kept_rule")
    if rule is None:
        return None
    if rule.get("name") != KEPT_RULE_RETAIN:
        raise RetentionRuleError(f"unknown kept rule {rule.get('name')!r}")
    if "floor_kind" in rule:  # a floor given in rows (PREREG E7)
        if rule["floor_kind"] != FLOOR_KIND_ROWS or "frac" in rule:
            raise RetentionRuleError(f"a kept_rule with floor_kind {rule['floor_kind']!r} is not a floor in rows")
        floor = check_floor_rows(rule["s_old_floor"], rule["n_old"])
        if is_placeholder(rule.get("floor_source")) and not session.get("dry_run"):
            raise RetentionRuleError("a placeholder floor is for a dry run only")
        return floor
    floor = int(rule["s_old_floor"])
    if floor != s_old_floor(rule["frac"], rule["n_old"]):
        raise RetentionRuleError(f"kept_rule s_old_floor {floor} does not follow from frac and n_old")
    return floor


__all__ = ["FLOOR_KIND_ROWS", "KEPT_RULE_RETAIN", "RetentionRuleError", "check_floor_rows", "is_placeholder",
           "kept_rule_record", "kept_rule_record_rows", "s_old_floor", "session_floor"]
