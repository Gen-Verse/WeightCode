"""Search-based repair baselines: Arachne-style (localise, then differential evolution) and CARE-style
(causal neuron localisation, then particle swarm optimisation).

They answer whether a repair gain needs reasoning, or whether localise-then-search on the same evidence gets
there. Both see exactly the editor-visible files plus the counted patch-check oracle (contract sections 3 to 5).

Modules: ``netspec`` (float64 re-implementation of a model_spec network for activations, interventions and
gradients), ``evidence`` (negative and positive evidence sets, counted instance forward), ``limits`` (contract 5.1
edit limits), ``selection`` (trajectories and the fixed candidate-to-query rule), ``arachne``, ``care`` and
``runner`` (one recorded run per instance, method and seed).
"""

from weightcode.baselines.search.arachne import ArachneConfig
from weightcode.baselines.search.care import CareConfig
from weightcode.baselines.search.limits import EditLimits
from weightcode.baselines.search.runner import METHODS, RunOptions, SplitRoleError, run_search_baseline, summary_line
from weightcode.baselines.search.selection import QUERY_RULE

__all__ = [
    "METHODS",
    "QUERY_RULE",
    "ArachneConfig",
    "CareConfig",
    "EditLimits",
    "RunOptions",
    "SplitRoleError",
    "run_search_baseline",
    "summary_line",
]
