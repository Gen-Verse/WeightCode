"""E10 references on the ResNet-20 option-A student: the torch rebuild, input_stats' BatchNorm fold, bn_adapt, tent
and the E10 program limits."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from weightcode.baselines.fit import bnrefs as B
from weightcode.baselines.fit import rules as RU
from weightcode.baselines.fit import torch_forward as TF
from weightcode.evalkit.e10 import e10_program_limits
from weightcode.harness.arms import applied_limits
from weightcode.models import resnet20a as A
from weightcode.models.forward_gen import render_resnet20a_forward
from weightcode.patch import Limits, task_limits

MEAN = [0.4909, 0.4818, 0.4460]
STD = [0.2471, 0.2435, 0.2616]
CLASSES = [f"c{i}" for i in range(10)]


def _student(seed: int = 0) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    model = A.ResNet20A(10, MEAN, STD)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-0.5, 0.5)
                module.running_var.uniform_(0.5, 2.0)
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.2, 0.2)
    return A.export_params(model)


def _forward():
    namespace: dict = {}
    exec(compile(render_resnet20a_forward(MEAN, STD, 10, A.BN_EPS), "forward.py", "exec"), namespace)
    return namespace["forward"]


def _images(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, 3, 32, 32), dtype=np.uint8)


def _arch(params: dict[str, np.ndarray]) -> TF.Architecture:
    return TF.architecture_from(params, A.build_resnet20a_spec("m00000000", CLASSES, MEAN, STD, params))


def test_rebuild_matches_forward_and_trainable_set() -> None:
    params = _student()
    arch = _arch(params)
    assert arch.family == TF.RESNET20A and arch.last_linear_tensors == ("fc.weight", "fc.bias")
    assert len(arch.bn_layers) == 19 and arch.conv_layers[0] == "conv1"
    x = _images(16)
    assert TF.assert_parity(arch, params, _forward(), x, dtype=torch.float64) < 1e-5
    assert TF.assert_parity(arch, params, _forward(), x) < 1e-4
    trainable = TF.trainable_tensors(arch, params)
    assert not any(k.endswith(("running_mean", "running_var")) for k in trainable)
    assert len(trainable) == 97 - 2 * 19


def test_input_stats_fold_is_exact_on_interior_positions() -> None:
    params = _student(1)
    arch = _arch(params)
    x = _images(48, 1)
    mean, std = RU.input_channel_stats(arch, x)
    fold = RU.input_stats_fold(arch, params, mean, std, 1.0)
    assert set(k for k in params if not np.array_equal(fold.child[k], params[k])) == {
        "conv1.weight", "bn1.running_mean", "bn1.running_var"}
    k = fold.k
    t64 = TF.to_tensors(params, dtype=torch.float64)
    c64 = TF.to_tensors(fold.child, dtype=torch.float64)
    pre = TF._preprocessed(arch, torch.from_numpy(x), torch.float64)
    target_in = torch.tensor(k).view(1, 3, 1, 1) * (pre - torch.tensor(mean).view(1, 3, 1, 1))
    target = TF.bn_eval(t64, F.conv2d(target_in, t64["conv1.weight"], padding=1), "bn1")
    ours = TF.bn_eval(c64, F.conv2d(pre, c64["conv1.weight"], padding=1), "bn1")
    interior = (slice(None), slice(None), slice(1, -1), slice(1, -1))
    assert torch.max(torch.abs(ours[interior] - target[interior])) < 1e-4
    assert fold.rel_frob_admissible <= fold.rel_frob_k_form + 1e-12


def test_bn_adapt_limits_and_delivered_model() -> None:
    params = _student(2)
    arch = _arch(params)
    x = torch.from_numpy(_images(64, 2))
    t = B._tensors(params)
    with torch.no_grad():
        far, _ = B.adapted_forward(arch, t, x, 1e12, 64)
        base = B._eval_logits(arch, params, x)
        assert torch.max(torch.abs(far - base)) < 1e-6  # a huge prior keeps the student's statistics
        logits, stats = B.adapted_forward(arch, t, x, 0.0, 64)
    child = B._child_with(params, B._stats_values(stats))
    assert B._delivered_check(arch, child, x, logits) < 1e-4
    changed = {k for k in params if not np.array_equal(child[k], params[k])}
    assert changed <= {f"{p}.{s}" for p in arch.bn_layers for s in ("running_mean", "running_var")}
    stem = TF._preprocessed(arch, x, torch.float64)
    h = F.conv2d(stem, t["conv1.weight"], padding=1)
    assert np.allclose(child["bn1.running_mean"], h.mean(dim=(0, 2, 3)).numpy(), atol=1e-6)
    assert np.allclose(child["bn1.running_var"], h.var(dim=(0, 2, 3), unbiased=True).numpy(), rtol=1e-5)


def test_tent_lowers_entropy_and_changes_batchnorm_only() -> None:
    params = _student(3)
    arch = _arch(params)
    x = torch.from_numpy(_images(64, 3))
    affine = tuple(f"{p}.{s}" for p in arch.bn_layers for s in ("weight", "bias"))
    tensors = B._tensors(params, affine)
    optimizer = torch.optim.Adam([tensors[k] for k in affine], lr=1e-3)
    first = None
    for _ in range(10):
        optimizer.zero_grad()
        entropy = B.mean_entropy(B.adapted_forward(arch, tensors, x, 0.0, 64)[0])
        first = float(entropy) if first is None else first
        entropy.backward()
        optimizer.step()
    with torch.no_grad():
        last = float(B.mean_entropy(B.adapted_forward(arch, tensors, x, 0.0, 64)[0]))
    assert last < first
    assert all(tensors[k].grad is None for k in params if k not in affine)


def test_e10_limits_equal_the_harness_program_arm() -> None:
    task = task_limits(Limits())
    ours = e10_program_limits(task, 97)
    assert ours == applied_limits(task, "LEARN", 97)
    assert ours.max_changed_frac == 0.20 and ours.max_rel_frob == 0.5 and ours.max_ops == 97
    assert ours.max_edit_frac is None
    with pytest.raises(ValueError):
        e10_program_limits(task, 0)


def test_every_runner_the_cli_dispatches_with_e10_accepts_it() -> None:
    """cli.run_method passes ``e10`` to each runner; a runner without the keyword fails only when its method runs."""
    import inspect

    from weightcode.baselines.fit import grad, ridge

    runners = (grad.run_grad, ridge.run_ridge, RU.run_prior_shift, RU.run_tie, RU.run_input_stats, B.run_bn_ref)
    assert all("e10" in inspect.signature(runner).parameters for runner in runners)
