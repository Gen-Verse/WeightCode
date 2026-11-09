"""Search baselines on tiny synthetic instances: the re-implemented forward, gradients, localisation, the searches,
end-to-end runs and the CLI guard. Everything is written to temporary directories.

Each toy instance has the real evidence-v1 layout (model.npz, model_spec.json in the weightcode.model_spec.v0 format,
forward.py, evidence.npz, task.json with the executor's limits and no pair, private.json with correct hashes and
split_role and no target, a fake V_check outside the instance directory). Labels come from a random teacher network
whose output biases balance the classes; the base model is the teacher with the output bias of one class raised, so
images of every other class near that boundary become base errors. x_fail holds 16 random base errors of any class,
x_rand 48 random images. The toy forward.py uses naive loops, independent of the im2col code in
``weightcode.baselines.search.netspec``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from weightcode.baselines.instance_io import load_instance
from weightcode.baselines.search import arachne, care, runner
from weightcode.baselines.search.evidence import CountedForward, build_evidence_sets
from weightcode.baselines.search.limits import EditLimits, check_limits, edit_stats, limit_violations
from weightcode.baselines.search.netspec import (
    SpecError,
    check_parity,
    cross_entropy,
    dense_forward,
    dense_input,
    logits_from_output,
    loss_weight_grads,
    output_sum_grads,
    parse_spec,
    softmax,
)
from weightcode.baselines.search.patches import write_sparse_patch
from weightcode.baselines.search.runner import RunOptions, SplitRoleError, run_search_baseline
from weightcode.baselines.search.selection import select_for_query
from weightcode.common.identity import load_params, params_sha256, save_params
from weightcode.evalkit.metrics import Candidate, choose_final
from weightcode.evalkit.patch_check import load_forward
from weightcode.models.forward_gen import render_forward
from weightcode.patch import Limits, PatchError, apply_patch_file, task_limits

REPO = Path(__file__).resolve().parents[1]
N_CLASSES = 4
SHAPES = {"mlp": (1, 6, 6), "cnn": (3, 16, 16)}
LAST = {"mlp": "fc3", "cnn": "fc2"}
SMALL = RunOptions(
    arachne=arachne.ArachneConfig(population=12, max_generations=8, patience=4),
    care=care.CareConfig(particles=8, max_iterations=10, patience=4),
)

TOY_FORWARD = '''"""Toy reference forward for tests: naive loops, same signature as an instance forward.py."""
import numpy as np

INPUT_SHAPE = __SHAPE__
MEAN = np.array(__MEAN__, dtype=np.float64)
STD = np.array(__STD__, dtype=np.float64)


def _conv3x3_same(h, w, b):
    n, c, height, width = h.shape
    padded = np.zeros((n, c, height + 2, width + 2))
    padded[:, :, 1:-1, 1:-1] = h
    out = np.empty((n, w.shape[0], height, width))
    for i in range(height):
        for j in range(width):
            out[:, :, i, j] = np.tensordot(padded[:, :, i:i + 3, j:j + 3], w, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


def _pool2(h):
    n, c, height, width = h.shape
    out = np.empty((n, c, height // 2, width // 2))
    for i in range(height // 2):
        for j in range(width // 2):
            out[:, :, i, j] = h[:, :, 2 * i:2 * i + 2, 2 * j:2 * j + 2].max(axis=(2, 3))
    return out


def forward(params, x_uint8):
    x = np.asarray(x_uint8)
    if x.dtype != np.uint8 or tuple(x.shape[1:]) != INPUT_SHAPE:
        raise ValueError("bad input")
    h = (x.astype(np.float64) / 255.0 - MEAN[None, :, None, None]) / STD[None, :, None, None]
    k = 1
    while "conv%d.weight" % k in params:
        w = np.asarray(params["conv%d.weight" % k], dtype=np.float64)
        b = np.asarray(params["conv%d.bias" % k], dtype=np.float64)
        h = _pool2(np.maximum(_conv3x3_same(h, w, b), 0.0))
        k += 1
    h = h.reshape(len(h), -1)
    m = 1
    while "fc%d.weight" % m in params:
        h = h @ np.asarray(params["fc%d.weight" % m], dtype=np.float64).T + np.asarray(params["fc%d.bias" % m], dtype=np.float64)
        if "fc%d.weight" % (m + 1) in params:
            h = np.maximum(h, 0.0)
        m += 1
    return h.astype(np.float32)
'''


def toy_params(arch: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """He-initialised toy network with the real parameter names (conv1..3, fc1..)."""
    params: dict[str, np.ndarray] = {}
    if arch == "cnn":
        channels = [3, 4, 4, 4]
        for k in range(3):
            fan_in = channels[k] * 9
            params[f"conv{k + 1}.weight"] = rng.normal(0, np.sqrt(2 / fan_in), (channels[k + 1], channels[k], 3, 3))
            params[f"conv{k + 1}.bias"] = rng.normal(0, 0.1, channels[k + 1])
        sizes = [("fc1", 8, 16), ("fc2", N_CLASSES, 8)]
    else:
        sizes = [("fc1", 12, 36), ("fc2", 10, 12), ("fc3", N_CLASSES, 10)]
    for name, out, inp in sizes:
        params[f"{name}.weight"] = rng.normal(0, np.sqrt(2 / inp), (out, inp))
        params[f"{name}.bias"] = rng.normal(0, 0.1, out)
    return {key: value.astype(np.float32) for key, value in params.items()}


def toy_spec(arch: str, params: dict[str, np.ndarray], mean: list[float], std: list[float]) -> dict:
    """A model_spec.json in the layout written by ``weightcode.models.spec``."""
    shape = list(SHAPES[arch])
    layers: list[dict] = []
    if arch == "cnn":
        for name in ("conv1", "conv2", "conv3"):
            weight = params[f"{name}.weight"]
            conv_out = [int(weight.shape[0]), shape[1], shape[2]]
            layers += [
                {
                    "name": name, "op": "conv2d", "weight": f"{name}.weight", "bias": f"{name}.bias",
                    "shape_w": list(weight.shape), "shape_b": [int(weight.shape[0])], "kernel": [3, 3],
                    "stride": [1, 1], "padding": [1, 1], "padding_mode": "zeros", "dilation": [1, 1], "groups": 1,
                    "in_shape": shape, "out_shape": conv_out,
                },
                {"op": "relu"},
            ]
            pooled = [conv_out[0], conv_out[1] // 2, conv_out[2] // 2]
            layers.append(
                {"op": "maxpool2d", "kernel": [2, 2], "stride": [2, 2], "padding": [0, 0], "ceil_mode": False,
                 "in_shape": conv_out, "out_shape": pooled}
            )
            shape = pooled
    layers.append({"op": "flatten", "order": "C, H, W (row-major)", "in_shape": shape, "out_features": int(np.prod(shape))})
    names = ["fc1", "fc2"] if arch == "cnn" else ["fc1", "fc2", "fc3"]
    for position, name in enumerate(names):
        weight = params[f"{name}.weight"]
        layers.append(
            {"name": name, "op": "linear", "weight": f"{name}.weight", "bias": f"{name}.bias",
             "shape_w": list(weight.shape), "shape_b": [int(weight.shape[0])]}
        )
        if position < len(names) - 1:
            layers.append({"op": "relu"})
    return {
        "schema": "weightcode.model_spec.v0",
        "model_id": f"mtoy{arch}",
        "input": {"shape": list(SHAPES[arch]), "dtype": "uint8", "range": [0, 255], "mean": mean, "std": std,
                  "layout": "NCHW, flatten row-major"},
        "classes": [f"c{i}" for i in range(N_CLASSES)],
        "layers": layers,
    }


@dataclass(frozen=True)
class Toy:
    """A written toy instance and what the tests need to know about it."""

    instance_dir: Path
    arch: str
    params: dict[str, np.ndarray]
    forward: Callable[[dict[str, np.ndarray], np.ndarray], np.ndarray]
    spec: dict
    mean: list[float]
    std: list[float]


def write_toy_instance(root: Path, arch: str, seed: int, instance_id: str, split_role: str | None = "dev") -> Toy:
    """Write a toy instance to ``root/instance_id`` and its fake V_check to ``root/vcheck``."""
    rng = np.random.default_rng(seed)
    shape = SHAPES[arch]
    mean, std = [0.5] * shape[0], [0.29] * shape[0]
    instance_dir = root / instance_id
    instance_dir.mkdir(parents=True)
    source = TOY_FORWARD.replace("__SHAPE__", repr(tuple(shape))).replace("__MEAN__", repr(mean))
    (instance_dir / "forward.py").write_text(source.replace("__STD__", repr(std)))
    forward, forward_sha = load_forward(instance_dir / "forward.py")

    teacher = toy_params(arch, rng)
    last = LAST[arch]
    pool = rng.integers(0, 256, size=(3000, *shape), dtype=np.uint8)
    no_bias = dict(teacher, **{f"{last}.bias": np.zeros(N_CLASSES, np.float32)})
    raw = forward(no_bias, pool).astype(np.float64)
    bias = np.zeros(N_CLASSES)
    for step in range(400):  # balance the teacher's classes on the pool (decaying steps avoid oscillation)
        freq = np.bincount(np.argmax(raw + bias, axis=1), minlength=N_CLASSES) / len(pool)
        bias -= 0.3 * 0.985**step * (np.log(freq + 1e-3) - np.log(1.0 / N_CLASSES))
    teacher[f"{last}.bias"] = (bias - bias.mean()).astype(np.float32)
    logits = forward(teacher, pool).astype(np.float64)
    y = np.argmax(logits, axis=1)
    raised = 1
    margins = (logits.max(axis=1) - logits[:, raised])[y != raised]
    base = {key: value.copy() for key, value in teacher.items()}
    base[f"{last}.bias"][raised] += np.float32(np.quantile(margins, 0.2))
    pred = np.argmax(forward(base, pool), axis=1)
    evidence_pool, vcheck = np.arange(1500), np.arange(1500, 2300)
    errors = evidence_pool[pred[evidence_pool] != y[evidence_pool]]
    assert len(errors) >= 40 and len(np.unique(y[errors])) >= 2, f"toy defect too weak: {len(errors)} errors"
    fail = rng.permutation(errors)[:16]
    rand = rng.permutation(np.setdiff1d(evidence_pool, fail))[:48]

    save_params(instance_dir / "model.npz", base)
    np.savez(
        instance_dir / "evidence.npz",
        x_fail=pool[fail], y_fail=y[fail], pred_fail=pred[fail],
        x_rand=pool[rand], y_rand=y[rand], pred_rand=pred[rand],
        split_ids=np.array([f"toy:{i}" for i in np.concatenate([fail, rand])]),
    )
    spec = toy_spec(arch, base, mean, std)
    (instance_dir / "model_spec.json").write_text(json.dumps(spec, indent=1))
    task = {"schema": "weightcode.task.v1", "limits": task_limits()}
    (instance_dir / "task.json").write_text(json.dumps(task))
    vcheck_path = root / "vcheck" / f"{instance_id}.npz"
    vcheck_path.parent.mkdir(exist_ok=True)
    np.savez(vcheck_path, x=pool[vcheck], y=y[vcheck])
    private = {
        "schema": "weightcode.instance_private.v0",
        "instance_id": instance_id,
        "model_sha256": params_sha256(base),
        "forward_sha256": forward_sha,
        "evidence_rule": "evidence-v1",
        "vcheck_path": str(vcheck_path),
        "dataset": "toy",
        "arch": arch,
    }
    if split_role is not None:
        private["split_role"] = split_role
    (instance_dir / "private.json").write_text(json.dumps(private))
    return Toy(instance_dir, arch, base, forward, spec, mean, std)


def set_split_role(instance_dir: Path, role: str | None) -> None:
    private = json.loads((instance_dir / "private.json").read_text())
    private.pop("split_role", None)
    if role is not None:
        private["split_role"] = role
    (instance_dir / "private.json").write_text(json.dumps(private))


@pytest.fixture(scope="module", params=["mlp", "cnn"])
def toy(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Toy:
    arch = request.param
    return write_toy_instance(tmp_path_factory.mktemp(f"toy_{arch}"), arch, seed=7, instance_id=f"itoy{arch}")


@pytest.fixture(scope="module")
def mlp_toy(tmp_path_factory: pytest.TempPathFactory) -> Toy:
    return write_toy_instance(tmp_path_factory.mktemp("toy_mlp_only"), "mlp", seed=11, instance_id="itoymlp")


def _float64(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.astype(np.float64) for key, value in params.items()}


def _evidence_sets(toy: Toy) -> tuple:
    instance = load_instance(toy.instance_dir)
    counted = CountedForward(instance.forward)
    sets = build_evidence_sets(instance.evidence, instance.target, instance.params, counted)
    return instance, counted, sets


# ---------------------------------------------------------------- re-implemented forward and its derivatives


def test_toy_instance_loads_and_has_the_failure_definition(toy: Toy) -> None:
    instance, _, sets = _evidence_sets(toy)
    assert instance.private["split_role"] == "dev"
    assert instance.target is None and sets.target is None
    assert sets.n_neg == 16 and sets.n_pos > 25
    assert len(np.unique(sets.y_neg)) >= 2  # failures of more than one true class
    assert np.all(np.argmax(toy.forward(toy.params, sets.x_neg), axis=1) != sets.y_neg)


def test_netspec_matches_toy_forward(toy: Toy) -> None:
    instance, counted, sets = _evidence_sets(toy)
    net = parse_spec(toy.spec, toy.params)
    assert [layer.name for layer in net.dense] == (["fc1", "fc2"] if toy.arch == "cnn" else ["fc1", "fc2", "fc3"])
    assert check_parity(net, toy.params, toy.forward, sets.x_all, 1e-5) < 1e-5


def test_netspec_matches_production_forward_template(toy: Toy, tmp_path: Path) -> None:
    path = tmp_path / "forward.py"
    path.write_text(render_forward(SHAPES[toy.arch], toy.mean, toy.std, N_CLASSES))
    production, _ = load_forward(path)
    _, _, sets = _evidence_sets(toy)
    net = parse_spec(toy.spec, toy.params)
    assert check_parity(net, toy.params, production, sets.x_all, 1e-5) < 1e-5


def test_parse_spec_rejects_unsupported_options(toy: Toy) -> None:
    for mutate, message in (
        (lambda s: s.update(schema="other"), "schema"),
        (lambda s: s["layers"].insert(0, {"op": "batchnorm"}), "unsupported op"),
        (lambda s: s["layers"][-1].update(weight="missing.weight"), "not in the parameters"),
    ):
        spec = json.loads(json.dumps(toy.spec))
        mutate(spec)
        with pytest.raises(SpecError, match=message):
            parse_spec(spec, toy.params)
    if toy.arch == "cnn":
        spec = json.loads(json.dumps(toy.spec))
        spec["layers"][0]["dilation"] = [2, 2]
        with pytest.raises(SpecError, match="dilation"):
            parse_spec(spec, toy.params)


def test_loss_weight_grads_match_finite_differences(toy: Toy) -> None:
    _, _, sets = _evidence_sets(toy)
    params = _float64(toy.params)
    net = parse_spec(toy.spec, params)
    x, y = sets.x_all[:20], sets.y_all[:20]
    grads = loss_weight_grads(net, params, dense_forward(net, params, dense_input(net, params, x)), y)
    rng = np.random.default_rng(0)
    eps = 1e-6
    for index, layer in enumerate(net.dense):
        for _ in range(6):
            row, col = (int(rng.integers(n)) for n in params[layer.weight].shape)
            values = []
            for sign in (1.0, -1.0):
                bumped = dict(params)
                bumped[layer.weight] = params[layer.weight].copy()
                bumped[layer.weight][row, col] += sign * eps
                logits = dense_forward(net, bumped, dense_input(net, bumped, x)).logits
                values.append(cross_entropy(logits, y).sum())
            numeric = (values[0] - values[1]) / (2 * eps)
            assert grads[index][row, col] == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_output_sum_grads_match_finite_differences(toy: Toy) -> None:
    _, _, sets = _evidence_sets(toy)
    params = _float64(toy.params)
    net = parse_spec(toy.spec, params)
    trace = dense_forward(net, params, dense_input(net, params, sets.x_all[:5]))
    grads = output_sum_grads(net, params, trace)
    assert np.array_equal(grads[-1], np.ones_like(trace.logits))
    eps = 1e-6
    for index in range(len(net.dense) - 1):
        output = trace.outputs[index]
        for unit in range(output.shape[1]):
            up, down = output.copy(), output.copy()
            up[:, unit] += eps
            down[:, unit] -= eps
            numeric = (
                logits_from_output(net, params, index, up).sum(axis=1)
                - logits_from_output(net, params, index, down).sum(axis=1)
            ) / (2 * eps)
            assert np.allclose(grads[index][:, unit], numeric, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------- Arachne localisation and DE


def _brute_ratios(toy: Toy, sets, pos_sample: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """GL and FI ratios per dense layer by explicit loops and finite differences."""
    params = _float64(toy.params)
    net = parse_spec(toy.spec, params)

    def gl_fi(x: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        trace = dense_forward(net, params, dense_input(net, params, x))
        gl, fi = [], []
        eps = 1e-6
        for index, layer in enumerate(net.dense):
            weight = params[layer.weight]
            grad = np.zeros_like(weight)
            for row in range(weight.shape[0]):
                for col in range(weight.shape[1]):
                    values = []
                    for sign in (1.0, -1.0):
                        bumped = dict(params)
                        bumped[layer.weight] = weight.copy()
                        bumped[layer.weight][row, col] += sign * eps
                        logits = dense_forward(net, bumped, trace.inputs[index], start=index).logits
                        values.append(cross_entropy(logits, y).sum())
                    grad[row, col] = (values[0] - values[1]) / (2 * eps)
            gl.append(np.abs(grad))
            u = trace.inputs[index]
            front = np.zeros_like(weight)
            for row in range(weight.shape[0]):
                for n in range(len(u)):
                    contrib = np.abs(u[n] * weight[row])
                    if contrib.sum() > 0:
                        front[row] += contrib / contrib.sum()
            front /= len(u)
            if index == len(net.dense) - 1:
                g = np.ones_like(trace.logits)
            else:
                g = np.zeros_like(trace.outputs[index])
                for unit in range(g.shape[1]):
                    up, down = trace.outputs[index].copy(), trace.outputs[index].copy()
                    up[:, unit] += eps
                    down[:, unit] -= eps
                    g[:, unit] = (
                        logits_from_output(net, params, index, up).sum(axis=1)
                        - logits_from_output(net, params, index, down).sum(axis=1)
                    ) / (2 * eps)
            g = np.abs(g)
            sums = g.sum(axis=1, keepdims=True)
            behind = np.divide(g, sums, out=np.zeros_like(g), where=sums > 0).mean(axis=0)
            fi.append(front * behind[:, None])
        return gl, fi

    gl_neg, fi_neg = gl_fi(sets.x_neg, sets.y_neg)
    gl_pos, fi_pos = gl_fi(sets.x_pos[pos_sample], sets.y_pos[pos_sample])
    grad_ratio = [n / (1.0 + p) for n, p in zip(gl_neg, gl_pos, strict=True)]
    fwd_ratio = [n / (1.0 + p) for n, p in zip(fi_neg, fi_pos, strict=True)]
    return grad_ratio, fwd_ratio


def test_arachne_localisation_matches_brute_force(mlp_toy: Toy) -> None:
    toy = mlp_toy
    _, _, sets = _evidence_sets(toy)
    net = parse_spec(toy.spec, toy.params)
    config = arachne.ArachneConfig()
    no_cap = EditLimits(max_changed_frac=None, max_edit_frac=None, max_rel_frob=None)
    loc = arachne.localise(net, toy.params, sets, config, np.random.default_rng(5), no_cap)
    pos_sample = arachne.sample_positives(sets.y_pos, sets.n_neg, np.random.default_rng(5))
    assert np.array_equal(loc.pos_sample, pos_sample)
    grad_ratio, fwd_ratio = _brute_ratios(toy, sets, pos_sample)
    points, refs = [], []
    for index, layer in enumerate(net.dense):
        points.append(np.stack([grad_ratio[index].ravel(), fwd_ratio[index].ravel()], axis=1))
        refs += [(layer.weight, (r, c)) for r in range(grad_ratio[index].shape[0]) for c in range(grad_ratio[index].shape[1])]
    stacked = np.concatenate(points)
    ours = np.array([[g, f] for g, f in zip(loc.grad_ratio, loc.fwd_ratio, strict=True)])
    expected_refs = {refs[i] for i in arachne.pareto_front(stacked)}
    assert {(ref.tensor, ref.index) for ref in loc.refs} == expected_refs
    by_ref = {refs[i]: stacked[i] for i in range(len(refs))}
    expected = np.array([by_ref[(ref.tensor, ref.index)] for ref in loc.refs])
    assert np.allclose(ours, expected, rtol=1e-4, atol=1e-9)
    capped = arachne.localise(net, toy.params, sets, config, np.random.default_rng(5), EditLimits.from_executor())
    assert capped.cap == EditLimits.from_executor().max_changed(toy.params)
    assert [(r.tensor, r.index) for r in capped.refs] == [(r.tensor, r.index) for r in loc.refs][: capped.cap]


def _arachne_problem(toy: Toy):
    instance, counted, sets = _evidence_sets(toy)
    net = parse_spec(toy.spec, instance.params)
    loc = arachne.localise(net, instance.params, sets, arachne.ArachneConfig(), np.random.default_rng(0), EditLimits.from_executor())
    problem = arachne.ArachneProblem(instance.params, loc.refs, sets, counted, 10.0, EditLimits.from_executor())
    return instance, counted, problem


def test_arachne_problem_projection_and_child(toy: Toy) -> None:
    instance, _, problem = _arachne_problem(toy)
    huge = problem.x0 + np.random.default_rng(1).normal(0, 50.0, problem.dim)
    child = problem.child(problem.project(huge))
    stats = check_limits(instance.params, child, EditLimits.from_executor())
    assert stats.n_changed <= problem.dim
    for key in instance.params:
        if key not in problem.tensors:
            assert np.array_equal(child[key], instance.params[key])
    assert np.array_equal(problem.child(problem.x0)[problem.tensors[0]], instance.params[problem.tensors[0]])


def test_de_is_deterministic_monotone_and_within_limits(toy: Toy) -> None:
    config = arachne.ArachneConfig(population=10, max_generations=6, patience=3)
    instance, _, problem = _arachne_problem(toy)
    first = arachne.run_de(problem, config, np.random.default_rng(3))
    second = arachne.run_de(problem, config, np.random.default_rng(3))
    assert np.array_equal(first.best_vector, second.best_vector)
    assert first.best_fitness_history == second.best_fitness_history
    fits = [snap.fitness for snap in first.trajectory.snapshots]
    assert all(later > earlier for earlier, later in zip(fits, fits[1:]))
    assert first.best_fitness == pytest.approx(problem.fitness(first.best_vector))
    assert first.n_evaluations == config.population * (first.steps_run + 1)
    assert first.stop_reason in {"no_improvement", "max_generations"}
    for snap in first.trajectory.snapshots:
        check_limits(instance.params, problem.child(snap.vector), EditLimits.from_executor())


# ---------------------------------------------------------------- CARE localisation and PSO


def test_care_scores_match_interventions_through_the_instance_forward(toy: Toy) -> None:
    instance, _, sets = _evidence_sets(toy)
    net = parse_spec(toy.spec, instance.params)
    layers = care.hidden_layers(net, None)
    refs, scores, base_mean, _ = care.causal_scores(net, instance.params, sets.x_neg, sets.y_neg, layers, 16)
    params = _float64(instance.params)
    trace = dense_forward(net, params, dense_input(net, params, sets.x_neg))
    rng = np.random.default_rng(2)
    for position in rng.choice(len(refs), size=min(6, len(refs)), replace=False):
        ref = refs[position]
        index = net.dense_index(ref.layer)
        values = trace.outputs[index][:, ref.unit]
        nxt = net.dense[index + 1]
        expectations = []
        for beta in np.linspace(values.min(), values.max(), 16):
            # do(unit = beta): drop the unit's column and add beta times it to the next bias
            intervened = dict(params)
            column = params[nxt.weight][:, ref.unit].copy()
            intervened[nxt.weight] = params[nxt.weight].copy()
            intervened[nxt.weight][:, ref.unit] = 0.0
            intervened[nxt.bias] = params[nxt.bias] + beta * column
            logits = toy.forward(intervened, sets.x_neg).astype(np.float64)
            expectations.append(softmax(logits)[np.arange(sets.n_neg), sets.y_neg].mean())  # p_true
        if values.min() == values.max():
            assert scores[position] == pytest.approx(base_mean)
        else:
            assert scores[position] == pytest.approx(float(np.mean(expectations)), abs=1e-5)
    p_true = softmax(toy.forward(instance.params, sets.x_neg).astype(np.float64))[np.arange(sets.n_neg), sets.y_neg]
    assert base_mean == pytest.approx(float(p_true.mean()), abs=1e-6)


def test_care_cost_counts_negatives_still_misclassified(toy: Toy) -> None:
    _, _, sets, _, problem = _care_problem(toy)
    logits = np.zeros((sets.n_neg + sets.n_pos, N_CLASSES))
    logits[np.arange(sets.n_neg), sets.y_neg] = 1.0  # every negative fixed
    logits[sets.n_neg + np.arange(sets.n_pos), sets.y_pos] = 1.0
    assert problem.cost_parts(logits) == (0.0, 1.0)
    logits[0] = 0.0
    logits[0, (sets.y_neg[0] + 1) % N_CLASSES] = 1.0  # one negative wrong again, to any class
    logits[sets.n_neg] = 0.0
    logits[sets.n_neg, (sets.y_pos[0] + 1) % N_CLASSES] = 1.0  # one positive broken
    ub, accuracy = problem.cost_parts(logits)
    assert ub == pytest.approx(1 / sets.n_neg) and accuracy == pytest.approx(1 - 1 / sets.n_pos)


def _care_problem(toy: Toy):
    instance, counted, sets = _evidence_sets(toy)
    net = parse_spec(toy.spec, instance.params)
    loc = care.localise(net, instance.params, sets, care.CareConfig(), EditLimits.from_executor())
    problem = care.CareProblem(instance.params, loc.selected, sets, counted, 0.2, EditLimits.from_executor())
    return instance, net, sets, loc, problem


def test_care_localisation_keeps_the_top_tenth(toy: Toy) -> None:
    instance, net, sets, loc, _ = _care_problem(toy)
    n_hidden = sum(np.asarray(instance.params[layer.weight]).shape[0] for layer in net.dense[:-1])
    assert loc.n_candidates == n_hidden
    assert loc.n_top == max(1, int(np.floor(0.1 * n_hidden)))
    assert len(loc.selected) == loc.n_top  # the toy caps are not binding
    assert np.all(np.diff(loc.selected_scores) <= 0)


def test_care_column_scaling_equals_output_scaling(toy: Toy) -> None:
    instance, net, sets, loc, problem = _care_problem(toy)
    r = np.random.default_rng(4).uniform(-0.3, 0.3, problem.dim)
    child = problem.child(problem.project(r))
    r = problem.project(r)
    params = _float64(instance.params)
    trace = dense_forward(net, params, dense_input(net, params, sets.x_all))
    by_layer: dict[str, list[tuple[int, float]]] = {}
    for ref, value in zip(problem.neurons, r, strict=True):
        by_layer.setdefault(ref.layer, []).append((ref.unit, value))
    # Apply the scalings layer by layer through the re-implementation, starting from the earliest layer.
    first = min(net.dense_index(name) for name in by_layer)
    u = trace.outputs[first].copy()
    for index in range(first, len(net.dense) - 1):
        if index > first:
            u = dense_forward(net, params, u, start=index).outputs[0]
        for unit, value in by_layer.get(net.dense[index].name, []):
            u[:, unit] *= 1.0 + value
    expected = dense_forward(net, params, u, start=len(net.dense) - 1).logits
    assert np.allclose(toy.forward(child, sets.x_all), expected, atol=1e-4)


def test_care_projection_respects_the_norm_limit(toy: Toy) -> None:
    instance, _, _, _, problem = _care_problem(toy)
    child = problem.child(problem.project(np.full(problem.dim, 50.0)))
    stats = check_limits(instance.params, child, EditLimits.from_executor())
    assert all(value <= 0.5 for value in stats.rel_frob.values())


def test_pso_is_deterministic_and_monotone(toy: Toy) -> None:
    config = care.CareConfig(particles=6, max_iterations=8, patience=3)
    instance, _, _, _, problem = _care_problem(toy)
    first = care.run_pso(problem, config, np.random.default_rng(9))
    second = care.run_pso(problem, config, np.random.default_rng(9))
    assert np.array_equal(first.best_vector, second.best_vector)
    assert first.best_fitness_history == second.best_fitness_history
    snaps = first.trajectory.snapshots
    assert np.array_equal(snaps[0].vector, np.zeros(problem.dim))  # every particle starts at r = 0
    assert all(later.fitness < earlier.fitness for earlier, later in zip(snaps, snaps[1:]))
    assert first.stop_reason in {"property_satisfied", "no_improvement", "max_iterations"}
    assert first.n_evaluations == config.particles * first.steps_run
    for snap in snaps:
        check_limits(instance.params, problem.child(snap.vector), EditLimits.from_executor())
        assert np.all(np.abs(snap.vector) <= 1.0)


# ---------------------------------------------------------------- executor round trip


def test_sparse_patch_round_trips_through_the_executor(mlp_toy: Toy, tmp_path: Path) -> None:
    base = load_params(mlp_toy.instance_dir / "model.npz")
    rng = np.random.default_rng(3)
    child = {key: value.copy() for key, value in base.items()}
    flat = child["fc2.weight"].reshape(-1)
    picks = rng.choice(flat.size, 5, replace=False)
    flat[picks] = (flat[picks] + rng.normal(0, 0.05, 5)).astype(np.float32)
    child["fc3.weight"][:, 2] = (child["fc3.weight"][:, 2].astype(np.float64) * 1.3).astype(np.float32)
    n_changed = 5 + int(np.count_nonzero(base["fc3.weight"][:, 2] != child["fc3.weight"][:, 2]))
    write_sparse_patch(tmp_path / "p", params_sha256(base), base, child, "test", "test")
    result = apply_patch_file(base, tmp_path / "p" / "patch.json", Limits(), params_sha256(base))
    for key in base:  # within float32 rounding; the set ops are in fact exact
        diff = np.abs(result.child[key].astype(np.float64) - child[key].astype(np.float64))
        assert np.all(diff <= np.spacing(np.abs(child[key])).astype(np.float64))
    assert result.child_sha256 == params_sha256(child)
    assert result.n_changed == n_changed
    assert result.summary()["edit_size"] == n_changed
    with pytest.raises(ValueError, match="no patch"):
        write_sparse_patch(tmp_path / "q", params_sha256(base), base, base, "test", "test")


# ---------------------------------------------------------------- end-to-end runs


def check_run(record: dict, run_dir: Path, toy: Toy, method: str, budget: int) -> dict:
    """Invariants every complete run must satisfy; returns the record read back from run.json."""
    on_disk = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == on_disk["status"] == "complete"
    assert on_disk["final"] == record["final"]
    assert on_disk["uses_gradient_for_localization"] is (method == "arachne")
    assert on_disk["instance"]["split_role"] == "dev"
    assert on_disk["instance"]["target"] is None and on_disk["instance"]["evidence_rule"] == "evidence-v1"
    assert on_disk["limits"]["from_executor"] and on_disk["executor_limits"] == task_limits()
    assert on_disk["task_limits"] == task_limits()
    limits = EditLimits.from_executor()
    log = on_disk["oracle_log"]
    valid = [candidate for candidate in on_disk["candidates"] if candidate["valid"]]
    assert on_disk["invalid_patches"] == [
        {"patch_id": c["patch_id"], "executor_error": c["executor_error"]} for c in on_disk["candidates"] if not c["valid"]
    ]
    assert on_disk["queries_used"] == len(log) == len(valid) <= budget
    assert [entry["patch_id"] for entry in log] == [candidate["patch_id"] for candidate in valid]
    base = load_params(toy.instance_dir / "model.npz")
    for entry, candidate in zip(log, valid, strict=True):
        applied = apply_patch_file(base, run_dir / entry["patch_id"] / "patch.json", Limits(), params_sha256(base))
        assert applied.child_sha256 == entry["child_sha256"]
        assert candidate["roundtrip_exact"] and candidate["search_limit_violations"] == []
        assert candidate["executor"]["edit_size"] == entry["n_changed"]  # sparse ops specify only what changes
        assert candidate["executor"]["edit_frac"] <= limits.changed_frac_cap()
        assert candidate["edit"]["within_limits"]
        assert limit_violations(edit_stats(base, applied.child), limits) == []
        largest = max(candidate["edit"]["rel_frob"].values())  # the executor's float64 sums differ in the last bit
        assert candidate["largest_rel_frob"]["value"] == pytest.approx(largest, rel=1e-12)
        assert all(candidate["edit"]["ndim"][name] == 2 for name in candidate["edit"]["rel_frob"])  # weights only
        assert entry["n_changed"] == candidate["edit"]["n_changed"] > 0
        assert entry["fixed_target"] is None and entry["score"] == entry["fixed"] - entry["broken"]
        assert candidate["check"] == entry
    final = choose_final([Candidate(e["patch_id"], e["score"], e["n_changed"], e["query_index"]) for e in log])
    assert on_disk["final"]["patch_id"] == final.patch_id
    assert on_disk["final"]["score"] == final.score
    assert on_disk["final"]["fixed_target"] is None
    if final.patch_id is not None:
        chosen = next(entry for entry in log if entry["patch_id"] == final.patch_id)
        assert (on_disk["final"]["fixed"], on_disk["final"]["broken"]) == (chosen["fixed"], chosen["broken"])
    search = on_disk["search"]
    eligible = search["trajectory_non_noop_indices"]
    assert search["selected_trajectory_indices"] == [eligible[p] for p in select_for_query(len(eligible), budget)]
    fits = [snap["fitness"] for snap in search["trajectory"]]
    increasing = method == "arachne"
    assert all((later > earlier) if increasing else (later < earlier) for earlier, later in zip(fits, fits[1:]))
    if eligible and eligible[-1] == len(fits) - 1:
        assert on_disk["candidates"][-1]["is_method_final_best"]
    counts = on_disk["forward_counts"]
    assert counts["search"]["calls"] == search["n_fitness_evaluations"]
    assert counts["total"]["calls"] == sum(counts[phase]["calls"] for phase in ("setup", "search", "readout"))
    assert on_disk["forward_parity_max_abs"] < 1e-5
    return on_disk


@pytest.mark.parametrize("method", ["arachne", "care"])
def test_end_to_end_run(toy: Toy, method: str, tmp_path: Path) -> None:
    record = run_search_baseline(toy.instance_dir, method, run_seed=0, out_root=tmp_path, options=SMALL)
    run_dir = tmp_path / toy.instance_dir.name / method / "0"
    check_run(record, run_dir, toy, method, budget=8)


def test_runs_are_reproducible(mlp_toy: Toy, tmp_path: Path) -> None:
    logs = []
    for out in ("one", "two"):
        record = run_search_baseline(mlp_toy.instance_dir, "arachne", 1, out_root=tmp_path / out, options=SMALL)
        logs.append([(entry["child_sha256"], entry["score"]) for entry in record["oracle_log"]])
    assert logs[0] == logs[1] and logs[0]


def test_small_budget_subsamples_the_trajectory(mlp_toy: Toy, tmp_path: Path) -> None:
    options = replace(SMALL, query_budget=2)
    record = run_search_baseline(mlp_toy.instance_dir, "arachne", 2, out_root=tmp_path, options=options)
    on_disk = check_run(record, tmp_path / mlp_toy.instance_dir.name / "arachne" / "2", mlp_toy, "arachne", budget=2)
    assert on_disk["queries_used"] == min(2, len(on_disk["search"]["trajectory_non_noop_indices"]))


def test_oracle_is_built_only_after_the_search(mlp_toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    real_de, real_oracle = arachne.run_de, runner.make_oracle

    def traced_de(*args, **kwargs):
        result = real_de(*args, **kwargs)
        events.append("search_done")
        return result

    def traced_oracle(*args, **kwargs):
        events.append("oracle")
        return real_oracle(*args, **kwargs)

    monkeypatch.setattr(arachne, "run_de", traced_de)
    monkeypatch.setattr(runner, "make_oracle", traced_oracle)
    run_search_baseline(mlp_toy.instance_dir, "arachne", 3, out_root=tmp_path, options=SMALL)
    assert events == ["search_done", "oracle"]


@pytest.mark.parametrize("role", ["main", None, "test"])
def test_runner_refuses_non_dev_instances(mlp_toy: Toy, tmp_path: Path, role: str | None) -> None:
    copy = tmp_path / "instances" / f"itoy_{role}"
    shutil.copytree(mlp_toy.instance_dir, copy)
    set_split_role(copy, role)
    with pytest.raises(SplitRoleError):
        run_search_baseline(copy, "arachne", 0, out_root=tmp_path / "runs", options=SMALL)
    assert not (tmp_path / "runs").exists()
    if role == "main":
        assert runner.check_split_role(load_instance(copy), allow_main=True) == "main"


def test_runner_refuses_a_task_with_stale_limits(mlp_toy: Toy, tmp_path: Path) -> None:
    copy = tmp_path / "instances" / "itoy_stale"
    shutil.copytree(mlp_toy.instance_dir, copy)
    task = json.loads((copy / "task.json").read_text())
    task["limits"] = {"max_changed_frac": 0.05, "max_rel_frob": 0.5}  # as written before the edit-limit switch
    (copy / "task.json").write_text(json.dumps(task))
    with pytest.raises(ValueError, match="predates a limits change"):
        run_search_baseline(copy, "care", 0, out_root=tmp_path / "runs", options=SMALL)
    assert not (tmp_path / "runs").exists()
    looser = replace(SMALL, limits=EditLimits(max_changed_frac=None, max_edit_frac=None, max_rel_frob=0.5))
    with pytest.raises(ValueError, match="looser"):
        run_search_baseline(mlp_toy.instance_dir, "care", 0, out_root=tmp_path / "runs", options=looser)
    assert not (tmp_path / "runs").exists()


def test_an_invalid_patch_is_recorded_and_consumes_no_query(
    mlp_toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_apply = runner.apply_patch_file
    calls: list[str] = []

    def reject_first(params, path, limits, parent_sha):
        calls.append(Path(path).parent.name)
        if len(calls) == 1:
            raise PatchError(["op 0 (set fc2.weight): rejected for the test"])
        return real_apply(params, path, limits, parent_sha)

    monkeypatch.setattr(runner, "apply_patch_file", reject_first)
    record = run_search_baseline(mlp_toy.instance_dir, "arachne", 4, out_root=tmp_path, options=SMALL)
    on_disk = check_run(record, tmp_path / mlp_toy.instance_dir.name / "arachne" / "4", mlp_toy, "arachne", budget=8)
    assert on_disk["invalid_patches"] == [{"patch_id": "q0", "executor_error": ["op 0 (set fc2.weight): rejected for the test"]}]
    assert [entry["query_index"] for entry in on_disk["oracle_log"]] == list(range(len(calls) - 1))
    assert on_disk["oracle_log"][0]["patch_id"] == "q1"
    assert not on_disk["flags"]["respects_edit_limits"]


def test_runner_refuses_an_existing_run_dir(mlp_toy: Toy, tmp_path: Path) -> None:
    run_dir = tmp_path / mlp_toy.instance_dir.name / "care" / "0"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        run_search_baseline(mlp_toy.instance_dir, "care", 0, out_root=tmp_path, options=SMALL)
    assert list(run_dir.iterdir()) == []


# ---------------------------------------------------------------- CLI selection and the main-instance guard


def _cli_store(tmp_path: Path, source: Toy) -> tuple[dict[str, str], Path]:
    """A private store with dev, main and index-mismatch copies of ``source``, plus an empty runs root."""
    private, runs, sealed = tmp_path / "private", tmp_path / "runs", tmp_path / "sealed"
    for path in (runs, sealed):
        path.mkdir()
    entries = []
    for instance_id, private_role, index_role in (
        ("idev1", "dev", "dev"),
        ("imain1", "main", "main"),
        ("imismatch", "main", "dev"),
    ):
        shutil.copytree(source.instance_dir, private / "instances" / instance_id)
        set_split_role(private / "instances" / instance_id, private_role)
        entries.append({"instance_id": instance_id, "dataset": "toy", "arch": "mlp", "split_role": index_role,
                        "seed": 100, "evidence_rule": "evidence-v1"})
    (private / "instances" / "index.json").write_text(json.dumps({"instances": entries}))
    env = dict(os.environ, WEIGHTCODE_PRIVATE=str(private), WEIGHTCODE_RUNS=str(runs), WEIGHTCODE_SEALED=str(sealed))
    return env, runs


def _cli(env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    script = REPO / "scripts" / "run_search_baselines.py"
    return subprocess.run([sys.executable, str(script), *args], env=env, capture_output=True, text=True, timeout=300)


def test_cli_refuses_main_instances(mlp_toy: Toy, tmp_path: Path) -> None:
    env, runs = _cli_store(tmp_path, mlp_toy)
    for args in (
        ("--instance", "imain1"),
        ("--instance", "idev1", "--instance", "imain1"),
        ("--split-role", "main"),
        ("--instance", "imismatch"),
        (),  # the default dev selection includes imismatch, whose private.json says main
    ):
        result = _cli(env, *args, "--methods", "care")
        assert result.returncode == 2, (args, result.stdout, result.stderr)
        assert "refused, nothing was run" in result.stderr
        assert "main" in result.stderr
        assert not (runs / "baselines").exists()


def test_cli_selects_dev_instances_by_default(mlp_toy: Toy, tmp_path: Path) -> None:
    env, runs = _cli_store(tmp_path, mlp_toy)
    index_path = Path(env["WEIGHTCODE_PRIVATE"]) / "instances" / "index.json"
    index = json.loads(index_path.read_text())
    index["instances"] = [entry for entry in index["instances"] if entry["instance_id"] != "imismatch"]
    index_path.write_text(json.dumps(index))
    result = _cli(env, "--dry-run")
    assert result.returncode == 0, result.stderr
    listed = [line.split()[0] for line in result.stdout.splitlines()]
    assert listed == ["idev1"]
    assert not (runs / "baselines").exists()


def test_cli_rejects_mixed_selection_options(mlp_toy: Toy, tmp_path: Path) -> None:
    env, runs = _cli_store(tmp_path, mlp_toy)
    for args in (("--instance", "idev1", "--dataset", "toy"), ("--instance", "idev1", "--split-role", "dev")):
        result = _cli(env, *args)
        assert result.returncode == 2 and "cannot be combined" in result.stderr
    assert _cli(env, "--instance", "inotthere").returncode == 2
    assert not (runs / "baselines").exists()


def test_forward_hash_in_private_json_is_the_file_hash(mlp_toy: Toy) -> None:
    private = json.loads((mlp_toy.instance_dir / "private.json").read_text())
    assert private["forward_sha256"] == hashlib.sha256((mlp_toy.instance_dir / "forward.py").read_bytes()).hexdigest()
