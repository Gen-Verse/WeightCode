"""Search trajectories and the fixed rule that picks which candidates spend patch-check queries.

The rule is fixed in code before any run and uses evidence only; V_check enters only through the queries it
issues, and the final patch is then chosen by the shared rule of contract 5.2 (``PatchCheckOracle.final``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QUERY_RULE = (
    "best_so_far_evenly_spaced.v0: record the best-so-far solution (by the method's own evidence fitness) after "
    "the initial population and after every generation or iteration, keeping one entry each time it changes; drop "
    "entries whose float32 child equals the base model; if at most `budget` entries remain, check all of them, "
    "otherwise check entries at positions floor(k * (m - 1) / (budget - 1) + 0.5), k = 0 .. budget - 1, which "
    "always includes the first and the final best (with budget 1, only the final best); queries are issued in "
    "search order."
)


@dataclass(frozen=True)
class Snapshot:
    """A best-so-far solution: its search vector, the step it was reached and its evidence fitness."""

    step: int
    vector: np.ndarray
    fitness: float


class Trajectory:
    """Distinct best-so-far solutions in search order."""

    def __init__(self) -> None:
        self.snapshots: list[Snapshot] = []

    def record(self, step: int, vector: np.ndarray, fitness: float) -> None:
        """Append ``vector`` unless it equals the latest entry."""
        if self.snapshots and np.array_equal(self.snapshots[-1].vector, vector):
            return
        self.snapshots.append(Snapshot(step=int(step), vector=np.array(vector, dtype=np.float64), fitness=float(fitness)))


@dataclass(frozen=True)
class SearchResult:
    """Outcome of one search run."""

    trajectory: Trajectory
    best_vector: np.ndarray
    best_fitness: float
    steps_run: int
    stop_reason: str
    best_fitness_history: list[float]
    n_evaluations: int
    maximize: bool


def select_for_query(n_entries: int, budget: int) -> list[int]:
    """Positions (into the de-duplicated, no-op-free trajectory) to patch-check, per ``QUERY_RULE``."""
    if budget <= 0 or n_entries <= 0:
        return []
    if n_entries <= budget:
        return list(range(n_entries))
    if budget == 1:
        return [n_entries - 1]
    positions = [int(np.floor(k * (n_entries - 1) / (budget - 1) + 0.5)) for k in range(budget)]
    if len(set(positions)) != budget or positions[-1] != n_entries - 1:
        raise AssertionError(f"query positions {positions} are not distinct or miss the final entry")
    return positions
