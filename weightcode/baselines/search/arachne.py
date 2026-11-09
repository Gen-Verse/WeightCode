"""Arachne-style search repair: bidirectional localisation and differential evolution.

Source: J. Sohn, S. Kang, S. Yoo, "Arachne: Search Based Repair of Deep Neural Networks", ACM TOSEM 2023
(arXiv 1912.12463v2). Section numbers below refer to that version.

Localisation (Sec. 2.3, Algorithm 1, "bidirectional localisation"). For every candidate weight w_ij (input unit i
of a dense layer, output unit j) two quantities are computed on the negative inputs and on a sample of positive
inputs of the same size (Alg. 1 line 2): the gradient loss GL = |dL/dw_ij| (L = softmax cross-entropy) and the
forward impact FI = (|o_i w_ij| / sum_l |o_l w_lj|) * |dO/do'_j|. The objectives are the ratios
GL_neg / (1 + GL_pos) and FI_neg / (1 + FI_pos) (Alg. 1 lines 6 and 9, footnote 2), and the localised set is
their Pareto front (Alg. 1 line 12).

Patch generation (Sec. 2.4, Algorithm 2). Differential evolution over the localised weights: trial elements
v_i = x1_i + F (x2_i - x3_i) for a forced random index and with probability CR otherwise (Eq. 1, Alg. 2 lines
4 to 11), replacement when fitness(V) >= fitness(X) (Eq. 2). Initial values are drawn from a Gaussian with the
mean and standard deviation of the weight's own layer (Sec. 2.4.2). Fitness (Sec. 2.4.3, Eq. 3 and 4): an input
scores 1 if classified correctly and 1 / (Loss + 1) otherwise; fitness = sum over all positives + alpha * sum
over negatives. Configuration (Sec. 4.2, 4.3): CR = 0.7, F ~ U(0.5, 1) once per generation, population 100,
at most 100 generations, stop after 10 generations without a change of the best, alpha = 10.

Where the paper's prose is silent or differs from its code, the authors' implementation (github.com/coinse/arachne:
run_localise.py, search/de_vk.py, auto_patch_vk.py) is followed: weights only, not biases, are localised; FI
takes absolute values and normalises per input before averaging; dO/do'_j differentiates the sum of all logits
(``tf.gradients(model.output, layer.output)``) and is L1-normalised per input before averaging; GL is the absolute
value of the gradient summed over inputs; positives for localisation are a stratified sample by label; trial
elements are clipped to [2 min, 2 max] of the weight's layer (halved instead of doubled when min > 0 or max < 0);
replacement is in place within a generation.

Deviations from the paper and code, also written to every run record (``DEVIATIONS``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from weightcode.baselines.search.evidence import CountedForward, EvidenceSets
from weightcode.baselines.search.limits import EditLimits, shrink_to_radius
from weightcode.baselines.search.netspec import (
    DenseTrace,
    NetSpec,
    cross_entropy,
    dense_forward,
    dense_input,
    loss_weight_grads,
    output_sum_grads,
)
from weightcode.baselines.search.selection import SearchResult, Trajectory

USES_GRADIENT_FOR_LOCALIZATION = True

DEVIATIONS: tuple[str, ...] = (
    "Layers: the candidate weights are the weight matrices of every dense (linear) layer (MLP fc1, fc2, fc3; CNN "
    "fc1, fc2); biases are excluded as in the paper and code. The authors' code (main_rq3_7.py, target_all=True) "
    "also localises over Conv2D kernels and arXiv v1 used only the last layer; conv kernels are excluded here to "
    "localise over the final (dense) layers, as CARE does for CNNs, and because one shared conv weight acts at "
    "every spatial position.",
    "Edit limits (contract 5.1, absent from the paper; values from weightcode.patch.Limits, recorded in run.json): the "
    "Pareto front is capped at floor(min(max_edit_frac, max_changed_frac) * n_params) weights, keeping the highest "
    "gradient-loss ratios, and each tensor's delta of every DE individual is radially projected into "
    "(1 - 1e-3) * max_rel_frob * ||W||_F around the base values for tensors the norm limit covers.",
    "Forward impact and gradient loss follow the authors' code where its arithmetic differs from the prose "
    "(absolute values, per-input L1 normalisation, O = sum of logits, hidden outputs taken after their ReLU).",
    "Evidence (evidence-v1): negatives are the 16 evidence failures, random base-model errors of any class, and "
    "positives the correctly classified random evidence images; localisation (GL with the cross-entropy of the "
    "true label) and the fitness use the true labels, as in the paper. This is the paper's RQ2 setting (random "
    "misclassifications, mixed types) rather than the single-type repair of RQ3 onwards; the paper samples 10% "
    "of the test-set errors as negatives (RQ2) or a held-out half of the test set (other RQs), here the 16 "
    "failures and the 48 random images of the evidence file are all the inputs.",
    "Fitness is computed with the instance's numpy forward.py; the paper uses the Keras model.",
    "Early stop after 10 consecutive generations without improvement of the best fitness (paper text); the code "
    "stops after 11.",
    "Random numbers come from numpy PCG64 seeded from the run seed; the DEAP and Python random streams of the "
    "authors' code are not reproduced.",
    "Output: instead of returning only the final best, up to 8 best-so-far solutions are patch-checked on V_check "
    "under the fixed candidate rule (selection.QUERY_RULE) and the final patch is chosen by contract 5.2; the final "
    "best is always one of the checked candidates.",
)


@dataclass(frozen=True)
class ArachneConfig:
    """Hyperparameters, fixed from Sec. 4.2 and 4.3 of the paper before any run."""

    alpha: float = 10.0
    population: int = 100
    max_generations: int = 100
    crossover_rate: float = 0.7
    mutation_low: float = 0.5
    mutation_high: float = 1.0
    patience: int = 10
    layers: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, object]:
        """JSON-ready copy of the configuration."""
        record = asdict(self)
        record["layers"] = list(self.layers) if self.layers is not None else None
        return record


@dataclass(frozen=True)
class WeightRef:
    """One localised weight: tensor name and (out, in) index in PyTorch layout."""

    tensor: str
    index: tuple[int, int]


@dataclass(frozen=True)
class Localisation:
    """Result of bidirectional localisation."""

    refs: list[WeightRef]
    grad_ratio: np.ndarray
    fwd_ratio: np.ndarray
    layers: list[str]
    n_candidates: int
    front_size: int
    cap: int
    pos_sample: np.ndarray
    analysis_images: int

    def as_dict(self) -> dict[str, object]:
        """JSON-ready summary for the run record, with every localised weight and its two ratios."""
        return {
            "layers": list(self.layers),
            "n_candidates": self.n_candidates,
            "pareto_front_size": self.front_size,
            "n_localised": len(self.refs),
            "cap_max_changed": self.cap,
            "truncated": len(self.refs) < self.front_size,
            "pos_sample_size": int(len(self.pos_sample)),
            "pos_sample_indices": [int(i) for i in self.pos_sample],
            "localised": [
                {
                    "tensor": ref.tensor,
                    "index": [int(v) for v in ref.index],
                    "grad_ratio": float(g),
                    "fwd_ratio": float(f),
                }
                for ref, g, f in zip(self.refs, self.grad_ratio, self.fwd_ratio, strict=True)
            ],
        }


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated rows of ``points`` [M, 2] when both columns are maximised.

    Exact duplicates collapse to the lowest index, as in the authors' ``localise_by_chgd_unchgd``. The result is
    ordered by the first objective, descending.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be [M, 2], got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("objectives must be finite")
    order = np.lexsort((np.arange(len(points)), -points[:, 1], -points[:, 0]))
    front = []
    best_second = -np.inf
    for index in order:
        if points[index, 1] > best_second:
            front.append(int(index))
            best_second = points[index, 1]
    return np.asarray(front, dtype=np.int64)


def sample_positives(y_pos: np.ndarray, n_neg: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified sample of positives of about ``n_neg`` items (Alg. 1 line 2; code: sample_input_for_loc_sophis).

    Each label gets round(n_neg * share) items, at least one and at most all of that label.
    """
    y_pos = np.asarray(y_pos)
    if n_neg >= len(y_pos):
        return np.arange(len(y_pos))
    chosen: list[int] = []
    for label in np.unique(y_pos):
        group = np.flatnonzero(y_pos == label)
        count = int(np.round(n_neg * len(group) / len(y_pos)))
        count = min(max(count, 1), len(group))
        chosen.extend(int(i) for i in rng.choice(group, size=count, replace=False))
    return np.asarray(sorted(chosen), dtype=np.int64)


def forward_impact(net: NetSpec, params: Mapping[str, np.ndarray], trace: DenseTrace) -> list[np.ndarray]:
    """FI per dense weight, [out, in] per layer, averaged over the traced inputs."""
    behind_grads = output_sum_grads(net, params, trace)
    result = []
    for index, layer in enumerate(net.dense):
        weight = np.asarray(params[layer.weight], dtype=np.float64)
        contrib = np.abs(trace.inputs[index][:, None, :] * weight[None, :, :])
        denom = contrib.sum(axis=2, keepdims=True)
        front = np.divide(contrib, denom, out=np.zeros_like(contrib), where=denom > 0).mean(axis=0)
        grad = np.abs(behind_grads[index])
        gden = grad.sum(axis=1, keepdims=True)
        behind = np.divide(grad, gden, out=np.zeros_like(grad), where=gden > 0).mean(axis=0)
        result.append(front * behind[:, None])
    return result


def gradient_loss(
    net: NetSpec, params: Mapping[str, np.ndarray], trace: DenseTrace, y: np.ndarray
) -> list[np.ndarray]:
    """GL per dense weight: absolute value of the cross-entropy gradient summed over the traced inputs."""
    return [np.abs(g) for g in loss_weight_grads(net, params, trace, y)]


def allowed_layers(net: NetSpec, requested: Sequence[str] | None) -> list[str]:
    """Dense layers whose weights are candidates (all dense layers when ``requested`` is None)."""
    names = [layer.name for layer in net.dense]
    if requested is None:
        return names
    unknown = sorted(set(requested) - set(names))
    if unknown:
        raise KeyError(f"layers {unknown} are not dense layers of this network ({names})")
    return [name for name in names if name in set(requested)]


def localise(
    net: NetSpec,
    params: Mapping[str, np.ndarray],
    sets: EvidenceSets,
    config: ArachneConfig,
    rng: np.random.Generator,
    limits: EditLimits,
) -> Localisation:
    """Bidirectional localisation (Alg. 1) over the allowed dense layers."""
    layers = allowed_layers(net, config.layers)
    pos_sample = sample_positives(sets.y_pos, sets.n_neg, rng)
    trace_neg = dense_forward(net, params, dense_input(net, params, sets.x_neg))
    trace_pos = dense_forward(net, params, dense_input(net, params, sets.x_pos[pos_sample]))
    gl_neg = gradient_loss(net, params, trace_neg, sets.y_neg)
    gl_pos = gradient_loss(net, params, trace_pos, sets.y_pos[pos_sample])
    fi_neg = forward_impact(net, params, trace_neg)
    fi_pos = forward_impact(net, params, trace_pos)

    refs: list[WeightRef] = []
    points = []
    for name in layers:
        index = net.dense_index(name)
        tensor = net.dense[index].weight
        grad_ratio = gl_neg[index] / (1.0 + gl_pos[index])
        fwd_ratio = fi_neg[index] / (1.0 + fi_pos[index])
        rows, cols = np.indices(grad_ratio.shape)
        refs.extend(WeightRef(tensor, (int(r), int(c))) for r, c in zip(rows.ravel(), cols.ravel(), strict=True))
        points.append(np.stack([grad_ratio.ravel(), fwd_ratio.ravel()], axis=1))
    stacked = np.concatenate(points, axis=0)
    front = pareto_front(stacked)
    cap = limits.max_changed(params)
    kept = front[:cap]
    return Localisation(
        refs=[refs[i] for i in kept],
        grad_ratio=stacked[kept, 0],
        fwd_ratio=stacked[kept, 1],
        layers=layers,
        n_candidates=int(len(stacked)),
        front_size=int(len(front)),
        cap=cap,
        pos_sample=pos_sample,
        analysis_images=int(sets.n_neg + len(pos_sample)),
    )


def arachne_fitness(logits: np.ndarray, y: np.ndarray, n_neg: int, alpha: float) -> float:
    """Eq. (3) and (4): sum of positive scores + alpha * sum of negative scores; negatives come first."""
    y = np.asarray(y, dtype=np.int64)
    correct = np.argmax(logits, axis=1) == y
    score = np.where(correct, 1.0, 1.0 / (cross_entropy(logits, y) + 1.0))
    return float(score[n_neg:].sum() + alpha * score[:n_neg].sum())


def layer_bounds(tensor: np.ndarray) -> tuple[float, float]:
    """Clipping range of DE values for weights of this layer (code: DE_searcher.set_bounds)."""
    low = float(np.min(tensor))
    high = float(np.max(tensor))
    low = low * 2.0 if low < 0 else low / 2.0
    high = high * 2.0 if high > 0 else high / 2.0
    return low, high


class ArachneProblem:
    """Maps a DE vector over the localised weights to a child model and scores it on evidence."""

    def __init__(
        self,
        base: Mapping[str, np.ndarray],
        refs: Sequence[WeightRef],
        sets: EvidenceSets,
        counted: CountedForward,
        alpha: float,
        limits: EditLimits,
    ) -> None:
        if not refs:
            raise ValueError("no localised weights")
        self.base = {key: np.asarray(value) for key, value in base.items()}
        self.refs = list(refs)
        self.tensors = sorted({ref.tensor for ref in refs})
        self.group = np.asarray([self.tensors.index(ref.tensor) for ref in refs], dtype=np.int64)
        self.flat = np.asarray(
            [np.ravel_multi_index(ref.index, self.base[ref.tensor].shape) for ref in refs], dtype=np.int64
        )
        self.x0 = np.asarray([float(self.base[ref.tensor][ref.index]) for ref in refs], dtype=np.float64)
        means, stds, lows, highs, radii = {}, {}, {}, {}, {}
        for name in self.tensors:
            tensor = self.base[name].astype(np.float64)
            means[name], stds[name] = float(tensor.mean()), float(tensor.std())
            lows[name], highs[name] = layer_bounds(tensor)
            radii[name] = limits.frob_radius(tensor)
        self.init_mean = np.asarray([means[ref.tensor] for ref in refs])
        self.init_std = np.asarray([stds[ref.tensor] for ref in refs])
        self.low = np.asarray([lows[ref.tensor] for ref in refs])
        self.high = np.asarray([highs[ref.tensor] for ref in refs])
        self.radius = np.asarray([radii[name] for name in self.tensors])
        self.sets = sets
        self.x_all = sets.x_all
        self.y_all = sets.y_all
        self.counted = counted
        self.alpha = float(alpha)

    @property
    def dim(self) -> int:
        """Number of searched weights."""
        return len(self.refs)

    def project(self, vector: np.ndarray) -> np.ndarray:
        """Shrink each tensor's delta from the base values into its Frobenius radius."""
        out = np.array(vector, dtype=np.float64)
        for group, radius in enumerate(self.radius):
            mask = self.group == group
            out[mask] = self.x0[mask] + shrink_to_radius(out[mask] - self.x0[mask], float(radius))
        return out

    def child(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        """Child parameters with the localised weights set to ``vector`` (cast to float32)."""
        child = dict(self.base)
        for group, name in enumerate(self.tensors):
            mask = self.group == group
            tensor = self.base[name].copy()
            tensor.reshape(-1)[self.flat[mask]] = np.asarray(vector, dtype=np.float64)[mask].astype(tensor.dtype)
            child[name] = tensor
        return child

    def logits(self, vector: np.ndarray) -> np.ndarray:
        """Instance-forward logits of the child on all evidence inputs (negatives first); one counted call."""
        return self.counted(self.child(vector), self.x_all)

    def fitness(self, vector: np.ndarray) -> float:
        """Eq. (4) fitness of the child on the evidence."""
        return arachne_fitness(self.logits(vector), self.y_all, self.sets.n_neg, self.alpha)


def run_de(problem: ArachneProblem, config: ArachneConfig, rng: np.random.Generator) -> SearchResult:
    """Differential evolution of Alg. 2 with the paper's configuration; maximises the Arachne fitness."""
    size, dim = config.population, problem.dim
    if size < 4:
        raise ValueError("DE needs a population of at least 4")
    population = np.empty((size, dim), dtype=np.float64)
    for member in range(size):
        draw = rng.normal(problem.init_mean, problem.init_std).astype(np.float32).astype(np.float64)
        population[member] = problem.project(draw)
    fitness = np.asarray([problem.fitness(vector) for vector in population])
    evaluations = size
    best_index = int(np.argmax(fitness))
    best_vector, best_fitness = population[best_index].copy(), float(fitness[best_index])
    trajectory = Trajectory()
    trajectory.record(0, best_vector, best_fitness)
    history = [best_fitness]
    unchanged = 0
    stop_reason = "max_generations"
    generation = 0
    for generation in range(1, config.max_generations + 1):
        previous_best = best_fitness
        scale = float(rng.uniform(config.mutation_low, config.mutation_high))
        for target in range(size):
            others = np.delete(np.arange(size), target)
            first, second, third = rng.choice(others, size=3, replace=False)
            mask = rng.random(dim) < config.crossover_rate
            mask[int(rng.integers(dim))] = True
            trial = population[target].copy()
            mutant = population[first] + scale * (population[second] - population[third])
            trial[mask] = np.clip(mutant[mask], problem.low[mask], problem.high[mask])
            trial = problem.project(trial)
            trial_fitness = problem.fitness(trial)
            evaluations += 1
            if trial_fitness >= fitness[target]:
                population[target] = trial
                fitness[target] = trial_fitness
                if trial_fitness > best_fitness:
                    best_vector, best_fitness = trial.copy(), float(trial_fitness)
        trajectory.record(generation, best_vector, best_fitness)
        history.append(best_fitness)
        unchanged = unchanged + 1 if np.float32(best_fitness) == np.float32(previous_best) else 0
        if unchanged >= config.patience:
            stop_reason = "no_improvement"
            break
    return SearchResult(
        trajectory=trajectory,
        best_vector=best_vector,
        best_fitness=best_fitness,
        steps_run=generation,
        stop_reason=stop_reason,
        best_fitness_history=history,
        n_evaluations=evaluations,
        maximize=True,
    )
