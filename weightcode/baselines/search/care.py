"""CARE-style search repair: causality-based neuron localisation and particle swarm optimisation.

Source: B. Sun, J. Sun, L. H. Pham, J. Shi, "Causality-Based Neural Network Repair", ICSE 2022
(arXiv 2204.09274v2). Section numbers below refer to that version.

Property (evidence-v1 instances). The repair property is "the model predicts the true label" on the negative
inputs, the evidence failures (random base-model errors of any class).

Localisation (Sec. 3.3, Definition 3.3, Algorithm 1 lines 5 to 8). The causal attribution of a hidden neuron x to
y is ACE = E[y | do(x = beta)], estimated by fixing x at a value while the inputs vary (Eq. 6 and 7). For safety
repair the paper sets y to the prediction value of the desirable class t, y = N_t(x); here the desirable class of
each negative input is its true label, so y = softmax probability of the true label, averaged over the negatives.
Following the authors' released implementation (Socrates, solver/causal_impl.py: get_ie_do_h, stepsize 16), beta
takes 16 evenly spaced values between the minimum and the maximum of the neuron's value over the negatives, and a
neuron's score is the mean of the 16 interventional expectations. Neurons are ranked by score in descending order,
as every ranking in the released code does (backdoor: ``ie_ave_matrix.sort(reverse=True)``), and the top 10% are
kept (Alg. 1 line 8, Sec. 4.2). For CNNs the paper applies CARE "on dense layers only" (Sec. 4.2, backdoor
removal); for feed-forward networks it analyses all hidden neurons. Both rules give the hidden units of the dense
layers here (MLP fc1 and fc2 outputs, CNN fc1 outputs).

Repair (Sec. 3.4, Eq. 8 to 10). PSO over one parameter per selected neuron. As in the released code
(``Model.apply_repair_fixed``), parameter r in [-1, 1] rescales the neuron's output, x <- (1 + r) x, which is
written into the weights as a scaling of the neuron's outgoing column in the next linear layer (identical
function). Fitness (Eq. 10), minimised: (1 - alpha) * UB + alpha * (1 - accuracy), with UB = fraction of negative
inputs still misclassified (the property violation rate, as the safety violation rate of Definition 2.4) and accuracy
on the positive inputs.
Configuration (Sec. 4.1, Table 6): omega = 0.8, c1 = c2 = 0.41, 20 particles, at most 100 iterations, stop as soon
as the property holds or after 10 consecutive iterations without a better location; alpha = 0.2, the value of
Table 6 for the three backdoor-removal image models, one of which is the Fashion-MNIST CNN. Swarm mechanics follow
the released code (pyswarms GlobalBestPSO): every particle starts at r = 0, initial velocities are U[0, 1), positions
outside the bounds wrap around periodically, and "no better location" means the global best cost improved by less
than ftol * (1 + |best|) with ftol = 1e-3.

Deviations from the paper and code, also written to every run record (``DEVIATIONS``).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from weightcode.baselines.search.evidence import CountedForward, EvidenceSets
from weightcode.baselines.search.limits import EditLimits
from weightcode.baselines.search.netspec import NetSpec, dense_forward, dense_input, logits_from_output, softmax
from weightcode.baselines.search.selection import SearchResult, Trajectory

USES_GRADIENT_FOR_LOCALIZATION = False

DEVIATIONS: tuple[str, ...] = (
    "Property and y (evidence-v1): the paper's properties are fairness, backdoor success rate and safety violation; "
    "here the property is 'the model predicts the true label' on the 16 evidence failures (random base errors of "
    "any class), with y = softmax probability of each failure's true label averaged over the failures (as "
    "y_safe = N_t(x) with t the desirable class) and UB = fraction of evidence failures still misclassified.",
    "Ranking direction: neurons are ranked by the mean interventional expectation of y in descending order, the "
    "order of every ranking in the released code (backdoor and fairness); the paper does not state the order for "
    "its safety case, whose y is a desirable-class value as here, and the released code has no safety solver. "
    "Ascending order (neurons whose interventions lower p_true) is the untested alternative.",
    "Accuracy in the fitness is measured on the correctly classified random evidence images (the positives); the "
    "paper samples a separate clean set.",
    "Initial velocity: the paper says zero, which with all particles starting at r = 0 would never move; the "
    "released code's pyswarms default, U[0, 1) per dimension, is used.",
    "A neuron whose value is constant on the analysis inputs gets the exact interventional expectation (the "
    "unchanged mean of y); the released code returns the constant activation itself in that case.",
    "Property threshold: the search stops when fewer than 1% of the evidence failures are still misclassified "
    "(with 16 failures, none); 1% is the paper's safety-repair target.",
    "The post-search accuracy check of Alg. 1 lines 18 to 20 (reject if the accuracy drop exceeds 3%) is recorded "
    "as a diagnostic, not used to discard candidates; the shared final-patch rule of contract 5.2 decides.",
    "Edit limits (contract 5.1, absent from the paper; values from weightcode.patch.Limits, recorded in run.json): the "
    "selected neurons are cut back, lowest score first, until scaling all their outgoing columns can change at "
    "most floor(min(max_edit_frac, max_changed_frac) * n_params) entries, and each particle's per-tensor delta "
    "is radially projected (all r of that tensor shrunk together) into (1 - 1e-3) * max_rel_frob * ||W||_F for "
    "tensors the norm limit covers.",
    "Fitness is computed with the instance's numpy forward.py; interventions for localisation use a float64 "
    "re-implementation checked against it.",
    "Random numbers come from numpy PCG64 seeded from the run seed, not pyswarms' global numpy state.",
    "Output: instead of returning only the final global best, up to 8 global-best solutions are patch-checked on "
    "V_check under the fixed candidate rule (selection.QUERY_RULE) and the final patch is chosen by contract 5.2; "
    "the final global best is always one of the checked candidates.",
)


@dataclass(frozen=True)
class CareConfig:
    """Hyperparameters, fixed from Sec. 4.1, Table 6 and the released code before any run."""

    alpha: float = 0.2
    particles: int = 20
    inertia: float = 0.8
    c1: float = 0.41
    c2: float = 0.41
    max_iterations: int = 100
    patience: int = 10
    ftol: float = 1e-3
    bound_low: float = -1.0
    bound_high: float = 1.0
    steps: int = 16
    top_fraction: float = 0.1
    property_threshold: float = 0.01
    accuracy_drop_threshold: float = 0.03
    layers: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, object]:
        """JSON-ready copy of the configuration."""
        record = asdict(self)
        record["layers"] = list(self.layers) if self.layers is not None else None
        return record


@dataclass(frozen=True)
class NeuronRef:
    """A hidden dense unit and the next-layer weight whose column carries its output."""

    layer: str
    unit: int
    next_weight: str


@dataclass(frozen=True)
class CareLocalisation:
    """Ranked neurons, the selected subset and the scores."""

    selected: list[NeuronRef]
    selected_scores: np.ndarray
    n_candidates: int
    n_top: int
    cap: int
    layers: list[str]
    base_mean_y: float
    analysis_images: int

    def as_dict(self) -> dict[str, object]:
        """JSON-ready summary for the run record, with every selected neuron and its ACE score."""
        return {
            "layers": list(self.layers),
            "n_candidates": self.n_candidates,
            "n_top_fraction": self.n_top,
            "n_selected": len(self.selected),
            "cap_max_changed": self.cap,
            "y": "softmax probability of the true label, mean over the negatives",
            "ranking": "descending mean of E[y | do(unit = beta)] over 16 betas",
            "base_mean_y_on_negatives": self.base_mean_y,
            "selected": [
                {"layer": ref.layer, "unit": ref.unit, "next_weight": ref.next_weight, "ace_score": float(score)}
                for ref, score in zip(self.selected, self.selected_scores, strict=True)
            ],
        }


def hidden_layers(net: NetSpec, requested: Sequence[str] | None) -> list[str]:
    """Hidden dense layers whose units are candidates (all of them when ``requested`` is None)."""
    names = [layer.name for layer in net.dense[:-1]]
    if requested is None:
        return names
    unknown = sorted(set(requested) - set(names))
    if unknown:
        raise KeyError(f"layers {unknown} are not hidden dense layers of this network ({names})")
    return [name for name in names if name in set(requested)]


def causal_scores(
    net: NetSpec,
    params: Mapping[str, np.ndarray],
    x_neg: np.ndarray,
    y_neg: np.ndarray,
    layers: Sequence[str],
    steps: int,
) -> tuple[list[NeuronRef], np.ndarray, float, int]:
    """ACE score of every hidden unit of ``layers``: mean over ``steps`` interventions of E[p_true | do(unit = beta)].

    p_true of a negative input is the softmax probability of its label in ``y_neg``. Returns the neuron references,
    their scores, the base mean of p_true on ``x_neg`` and the number of image forwards spent in the
    re-implementation.
    """
    labels = np.asarray(y_neg, dtype=np.int64)
    n = len(x_neg)
    if labels.shape != (n,):
        raise ValueError(f"y_neg has shape {labels.shape}, expected ({n},)")
    rows = np.arange(n)
    trace = dense_forward(net, params, dense_input(net, params, x_neg))
    base_mean = float(softmax(trace.logits)[rows, labels].mean())
    refs: list[NeuronRef] = []
    scores: list[float] = []
    images = n
    for name in layers:
        index = net.dense_index(name)
        output = trace.outputs[index]
        next_weight = net.dense[index + 1].weight
        for unit in range(output.shape[1]):
            low, high = float(output[:, unit].min()), float(output[:, unit].max())
            refs.append(NeuronRef(layer=name, unit=unit, next_weight=next_weight))
            if low == high:
                scores.append(base_mean)
                continue
            betas = np.linspace(low, high, steps)
            batch = np.repeat(output[None, :, :], steps, axis=0)
            batch[:, :, unit] = betas[:, None]
            logits = logits_from_output(net, params, index, batch.reshape(steps * n, -1))
            expectation = softmax(logits).reshape(steps, n, -1)[:, rows, labels].mean(axis=1)
            scores.append(float(expectation.mean()))
            images += steps * n
    return refs, np.asarray(scores, dtype=np.float64), base_mean, images


def localise(
    net: NetSpec,
    params: Mapping[str, np.ndarray],
    sets: EvidenceSets,
    config: CareConfig,
    limits: EditLimits,
) -> CareLocalisation:
    """Rank hidden units by causal attribution and keep the top fraction, cut back to the edit-size cap."""
    layers = hidden_layers(net, config.layers)
    if not layers:
        raise ValueError("the network has no hidden dense layer to localise")
    refs, scores, base_mean, images = causal_scores(net, params, sets.x_neg, sets.y_neg, layers, config.steps)
    order = sorted(range(len(refs)), key=lambda i: -scores[i])  # stable: ties keep layer, then unit order
    n_top = max(1, int(np.floor(config.top_fraction * len(refs))))
    cap = limits.max_changed(params)
    selected: list[int] = []
    budget = 0
    for i in order[:n_top]:
        ref = refs[i]
        cost = int(np.count_nonzero(np.asarray(params[ref.next_weight])[:, ref.unit]))
        if budget + cost > cap:
            break
        selected.append(i)
        budget += cost
    return CareLocalisation(
        selected=[refs[i] for i in selected],
        selected_scores=scores[selected],
        n_candidates=len(refs),
        n_top=n_top,
        cap=cap,
        layers=layers,
        base_mean_y=base_mean,
        analysis_images=images,
    )


class CareProblem:
    """Maps a PSO position (one r per selected neuron) to a child model and its Eq. (10) cost on evidence."""

    def __init__(
        self,
        base: Mapping[str, np.ndarray],
        neurons: Sequence[NeuronRef],
        sets: EvidenceSets,
        counted: CountedForward,
        alpha: float,
        limits: EditLimits,
    ) -> None:
        if not neurons:
            raise ValueError("no selected neurons")
        self.base = {key: np.asarray(value) for key, value in base.items()}
        self.neurons = list(neurons)
        self.tensors = sorted({ref.next_weight for ref in neurons})
        self.group = np.asarray([self.tensors.index(ref.next_weight) for ref in neurons], dtype=np.int64)
        self.units = np.asarray([ref.unit for ref in neurons], dtype=np.int64)
        self.column_norm = np.asarray(
            [float(np.linalg.norm(self.base[ref.next_weight][:, ref.unit].astype(np.float64))) for ref in neurons]
        )
        self.radius = np.asarray([limits.frob_radius(self.base[name]) for name in self.tensors])
        self.sets = sets
        self.x_all = sets.x_all
        self.counted = counted
        self.alpha = float(alpha)

    @property
    def dim(self) -> int:
        """Number of searched neurons (one r each)."""
        return len(self.neurons)

    def project(self, position: np.ndarray) -> np.ndarray:
        """Shrink each tensor's r-vector so that ||sum_j r_j W[:, j] e_j^T||_F stays within the radius."""
        out = np.array(position, dtype=np.float64)
        for group, radius in enumerate(self.radius):
            mask = self.group == group
            norm = float(np.sqrt(np.sum((out[mask] * self.column_norm[mask]) ** 2)))
            if norm > radius:
                out[mask] *= radius / norm if radius > 0 else 0.0
        return out

    def child(self, position: np.ndarray) -> dict[str, np.ndarray]:
        """Child parameters: column ``unit`` of each next-layer weight scaled by (1 + r), computed in float64."""
        child = dict(self.base)
        for group, name in enumerate(self.tensors):
            mask = self.group == group
            tensor = self.base[name].astype(np.float64)
            tensor[:, self.units[mask]] *= 1.0 + np.asarray(position, dtype=np.float64)[mask]
            child[name] = tensor.astype(self.base[name].dtype)
        return child

    def logits(self, position: np.ndarray) -> np.ndarray:
        """Instance-forward logits of the child on all evidence inputs (negatives first); one counted call."""
        return self.counted(self.child(position), self.x_all)

    def cost_parts(self, logits: np.ndarray) -> tuple[float, float]:
        """(UB, accuracy): fraction of negatives still misclassified, accuracy on positives."""
        pred = np.argmax(logits, axis=1)
        n_neg = self.sets.n_neg
        ub = float(np.mean(pred[:n_neg] != self.sets.y_neg))
        accuracy = float(np.mean(pred[n_neg:] == self.sets.y_pos))
        return ub, accuracy

    def evaluate(self, position: np.ndarray) -> tuple[float, float, float]:
        """Eq. (10) cost with its UB and accuracy parts."""
        ub, accuracy = self.cost_parts(self.logits(position))
        return (1.0 - self.alpha) * ub + self.alpha * (1.0 - accuracy), ub, accuracy


def periodic_bounds(position: np.ndarray, low: float, high: float) -> np.ndarray:
    """pyswarms ``BoundaryHandler(strategy='periodic')``: coordinates outside [low, high] wrap around."""
    out = np.array(position, dtype=np.float64)
    width = high - low
    below = out < low
    above = out > high
    out[below] = high - np.mod(low - out[below], width)
    out[above] = low + np.mod(out[above] - high, width)
    return out


def run_pso(problem: CareProblem, config: CareConfig, rng: np.random.Generator) -> SearchResult:
    """Global-best PSO (Eq. 8, 9) with the released code's swarm mechanics; minimises the Eq. (10) cost."""
    count, dim = config.particles, problem.dim
    position = np.zeros((count, dim), dtype=np.float64)
    velocity = rng.random((count, dim))
    pbest_position = position.copy()
    pbest_cost = np.full(count, np.inf)
    pbest_ub = np.full(count, np.inf)
    gbest_position = position[0].copy()
    gbest_cost = np.inf
    gbest_ub = np.inf
    trajectory = Trajectory()
    history: list[float] = []
    no_gain: deque[bool] = deque(maxlen=config.patience)
    evaluations = 0
    stop_reason = "max_iterations"
    iteration = -1
    for iteration in range(config.max_iterations):
        costs = np.empty(count)
        ubs = np.empty(count)
        for particle in range(count):
            costs[particle], ubs[particle], _ = problem.evaluate(position[particle])
        evaluations += count
        better = costs < pbest_cost
        pbest_position[better] = position[better]
        pbest_cost[better] = costs[better]
        pbest_ub[better] = ubs[better]
        previous = gbest_cost
        leader = int(np.argmin(pbest_cost))
        if pbest_cost[leader] < gbest_cost:
            gbest_position = pbest_position[leader].copy()
            gbest_cost = float(pbest_cost[leader])
            gbest_ub = float(pbest_ub[leader])
        trajectory.record(iteration, gbest_position, gbest_cost)
        history.append(gbest_cost)
        if gbest_ub < config.property_threshold:
            stop_reason = "property_satisfied"
            break
        no_gain.append(bool(abs(gbest_cost - previous) < config.ftol * (1.0 + abs(previous))))
        if iteration >= config.patience and all(no_gain):
            stop_reason = "no_improvement"
            break
        cognitive = config.c1 * rng.random((count, dim)) * (pbest_position - position)
        social = config.c2 * rng.random((count, dim)) * (gbest_position[None, :] - position)
        velocity = config.inertia * velocity + cognitive + social
        position = periodic_bounds(position + velocity, config.bound_low, config.bound_high)
        position = np.stack([problem.project(row) for row in position])
    return SearchResult(
        trajectory=trajectory,
        best_vector=gbest_position.copy(),
        best_fitness=float(gbest_cost),
        steps_run=iteration + 1,
        stop_reason=stop_reason,
        best_fitness_history=history,
        n_evaluations=evaluations,
        maximize=False,
    )
