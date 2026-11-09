"""PREREG E12 (``resnet18_up``, dataset ``cifar10_r18``): the student, its numpy forward, the training schedule, the torch
rebuild and the references adapted to it (input_stats' bn1 fold, bn_adapt, F1's ft_bn and convolution LoRA), the CLI
dispatch, and the harness module (student check, MCP floors, leak words in the editor-visible text).

The students here are random-weight resnet18_up networks at R = 64 (the ImageNet weights file is not read), with the
BatchNorm statistics moved away from 0 and 1 so that the eval-form BatchNorm matters. No test touches the store.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from weightcode.baselines.fit import bnrefs as B
from weightcode.baselines.fit import cli as C
from weightcode.baselines.fit import ftcompare as FT
from weightcode.baselines.fit import rules as RU
from weightcode.baselines.fit import torch_forward as TF
from weightcode.baselines.fit.runner import Evidence
from weightcode.common.identity import params_sha256
from weightcode.evalkit.instances import condition_leak_tokens, leak_tokens, scan_for_tokens
from weightcode.harness import e12 as H
from weightcode.models import resnet18up as M
from weightcode.models.forward_gen import render_resnet18up_forward
from weightcode.baselines.instance_io import write_delta_patch
from weightcode.evalkit.e10 import e10_program_limits
from weightcode.harness.arms import applied_limits
from weightcode.patch import Limits, apply_patch, load_patch, task_limits
from weightcode.patch.errors import PatchError
from weightcode.patch.limits import dense_delta_bytes

R = 64
CLASSES = [f"c{i}" for i in range(10)]
CONFIG = M.ResNet18UpConfig(resize=R)


def _student(seed: int = 0) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    model = M.ResNet18Up(CONFIG)
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith("running_mean"):
                buffer.copy_(0.1 * torch.randn_like(buffer))
            elif name.endswith("running_var"):
                buffer.copy_(0.5 + torch.rand_like(buffer))
        for name, p in model.named_parameters():
            if name.endswith("bn1.weight") or name.endswith("bn2.weight"):
                p.copy_(0.8 + 0.4 * torch.rand_like(p))
    return M.export_params(model)


def _forward_text(mean=M.IMAGENET_MEAN, std=M.IMAGENET_STD) -> str:
    return render_resnet18up_forward(mean, std, 10, M.BN_EPS, R)


def _forward(mean=M.IMAGENET_MEAN, std=M.IMAGENET_STD):
    namespace: dict = {}
    exec(compile(_forward_text(mean, std), "forward.py", "exec"), namespace)
    return namespace["forward"]


def _images(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, 3, 32, 32), dtype=np.uint8)


def _spec(params: dict[str, np.ndarray]) -> dict:
    return M.build_resnet18up_spec("m00000000", CLASSES, CONFIG, params)


def _arch(params: dict[str, np.ndarray]) -> TF.Architecture:
    return TF.architecture_from(params, _spec(params))


def test_layout_counts_and_names() -> None:
    shapes = M.param_shapes(10)
    assert len(shapes) == 102 and sum(int(np.prod(s)) for s in shapes.values()) == 11_191_242
    assert len(M.conv_names()) == 20 and len(M.bn_prefixes()) == 20
    stats = [k for k in shapes if k.endswith(("running_mean", "running_var"))]
    assert len(stats) == 40 and sum(int(np.prod(shapes[k])) for k in stats) == 9_600
    torchvision = pytest.importorskip("torchvision")
    reference = torchvision.models.resnet18(weights=None, num_classes=10).state_dict()
    assert {k for k in reference if not k.endswith("num_batches_tracked")} == set(shapes)


def test_step_lr_is_warmup_then_cosine() -> None:
    config = M.E12TrainConfig()
    steps = 313
    assert M.step_lr(config, 0, steps) == pytest.approx(0.01 / 313)
    assert M.step_lr(config, 312, steps) == pytest.approx(0.01)
    assert M.step_lr(config, 313, steps) == pytest.approx(0.01)
    middle = 313 + (15 * 313 - 313) // 2
    assert M.step_lr(config, middle, steps) == pytest.approx(0.005, rel=1e-3)
    assert 0.0 < M.step_lr(config, 15 * 313 - 1, steps) < 1e-8
    with pytest.raises(ValueError):
        M.step_lr(config, 15 * 313, steps)


def test_weights_file_is_hash_checked(tmp_path: Path) -> None:
    fake = tmp_path / M.WEIGHTS_FILE
    fake.write_bytes(b"not the weights")
    with pytest.raises(ValueError, match="sha256"):
        M.load_imagenet_state(fake)


def test_forward_matches_torch_and_torchvision() -> None:
    params = _student()
    x = _images(6)
    ours = np.asarray(_forward()(params, x), dtype=np.float64)
    ref64 = M.eval_logits(M.build_resnet18up(CONFIG, params, torch.float64), x)
    scale = max(1.0, float(np.max(np.abs(ref64))))
    assert np.max(np.abs(ours - ref64)) / scale < 2.0**-23
    model = M.build_resnet18up(CONFIG, params, torch.float32)
    check = M.torchvision_reference_diff(model, x)
    if check["torchvision"] is not None:
        assert check["max_abs_diff"] < 1e-4


def test_rebuild_matches_forward_and_trainable_set() -> None:
    params = _student(1)
    arch = _arch(params)
    assert arch.family == TF.RESNET18UP and arch.resize == R and arch.last_linear_tensors == ("fc.weight", "fc.bias")
    assert len(arch.conv_layers) == 20 and arch.conv_layers[0] == "conv1" and len(arch.bn_layers) == 20
    x = _images(5, 1)
    assert TF.assert_parity(arch, params, _forward(), x, dtype=torch.float64) < 1e-6
    assert TF.assert_parity(arch, params, _forward(), x) < 1e-4
    trainable = TF.trainable_tensors(arch, params)
    assert len(trainable) == 62 and not any(k.endswith(("running_mean", "running_var")) for k in trainable)
    feats = TF.penultimate(arch, TF.to_tensors(params, dtype=torch.float64), torch.from_numpy(x))
    model = M.build_resnet18up(CONFIG, params, torch.float64)
    with torch.no_grad():
        assert torch.allclose(feats, model.features(torch.from_numpy(x)), rtol=0, atol=1e-10)


def _kform_logits(params: dict[str, np.ndarray], x: np.ndarray, k: np.ndarray) -> torch.Tensor:
    """The base network on x'' = k_c x' (no mean shift, so no padding deviation)."""
    model = M.build_resnet18up(CONFIG, params, torch.float64)
    with torch.no_grad():
        h = model.upsampled(torch.from_numpy(x)) * torch.tensor(k).view(1, 3, 1, 1)
        h = model.maxpool(model.relu(model.bn1(model.conv1(h))))
        h = model.layer4(model.layer3(model.layer2(model.layer1(h))))
        return model.fc(h.mean(dim=(2, 3)))


def test_input_stats_fold_goes_into_bn1_and_is_exact_without_a_shift() -> None:
    params = _student(2)
    arch = _arch(params)
    x = _images(4, 2)
    std = np.array([0.7, 1.3, 1.1])
    fold = RU.input_stats_fold(arch, params, np.zeros(3), std, 1.0)
    changed = {k for k in params if not np.array_equal(fold.child[k], params[k])}
    assert changed == {"conv1.weight", "bn1.running_mean", "bn1.weight"}  # PREREG E12 amendment: g into the weight
    target = _kform_logits(params, x, fold.k)
    ours = M.eval_logits(M.build_resnet18up(CONFIG, fold.child, torch.float64), x)
    assert np.max(np.abs(ours - target.numpy())) < 1e-4  # float32 storage of the folded tensors
    mean, sd = RU.input_channel_stats(arch, _images(48, 3))
    assert mean.shape == sd.shape == (3,)


def _with_near_dead_stem_channels(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Two bn1 channels like the near-dead stem channels of ImageNet ResNet-18 (running variance 1e-13 and 3e-9)."""
    out = dict(params)
    var, gamma = params["bn1.running_var"].copy(), params["bn1.weight"].copy()
    var[[1, 5]] = np.array([1e-13, 3e-9], dtype=var.dtype)
    gamma[[1, 5]] = np.array([2e-8, -5e-8], dtype=gamma.dtype)
    out["bn1.running_var"], out["bn1.weight"] = var, gamma
    return out


def test_input_stats_fold_scales_bn1_weight_and_matches_forward_py_on_the_transformed_input() -> None:
    """PREREG E12 amendment: on resnet18_up, g goes into bn1.weight; E10's variance form refuses this student."""
    params = _with_near_dead_stem_channels(_student(4))
    arch = _arch(params)
    sha = params_sha256(params)
    std = np.array([0.6, 0.8, 0.9])  # s_c < 1 as under contrast: every k_c > 1, so g > 1
    x = _images(8, 4)
    mean_in, std_in = np.asarray(M.IMAGENET_MEAN), np.asarray(M.IMAGENET_STD)
    for t in RU.INPUT_STATS_STRENGTHS:
        fold = RU.input_stats_fold(arch, params, np.zeros(3), std, t)
        assert fold.g > 1.0 and fold.bn_fold["tensors"] == ["bn1.running_mean", "bn1.weight"]
        changed = {k for k in params if not np.array_equal(fold.child[k], params[k])}
        assert changed == {"conv1.weight", "bn1.running_mean", "bn1.weight"}
        np.testing.assert_array_equal(fold.child["bn1.running_var"], params["bn1.running_var"])
        patch = {"schema": "weightcode.patch.v0", "parent_sha256": sha, "hypothesis": "h", "predicted_effect": "p",
                 "ops": fold.ops}
        result = apply_patch(params, patch, None, Limits().without_size_limits(), sha)
        assert result.child_sha256 == params_sha256(fold.child), "the expected child must match the executor"
        # forward.py with x'' = k_c (x' - 0) in its own normalisation: (x / 255 - MEAN) / (STD / k_c).
        target = np.asarray(_forward(mean_in, std_in / fold.k)(params, x), dtype=np.float64)
        ours = np.asarray(_forward()(dict(result.child), x), dtype=np.float64)
        assert np.max(np.abs(ours - target)) < TF.PARITY_ATOL
        with pytest.raises(ValueError, match="non-positive running variance"):
            RU._input_stats_fold_bn(params, t, fold.k, fold.g, fold.r, np.zeros(3), "running_var")


def test_input_stats_fold_with_a_mean_shift_is_exact_on_interior_bn1_positions() -> None:
    params = _with_near_dead_stem_channels(_student(5))
    arch = _arch(params)
    x = _images(6, 5)
    mean, std = np.array([0.4, -0.3, 0.2]), np.array([0.6, 0.8, 0.9])
    for t in RU.INPUT_STATS_STRENGTHS:
        fold = RU.input_stats_fold(arch, params, mean, std, t)
        base = M.build_resnet18up(CONFIG, params, torch.float64)
        child = M.build_resnet18up(CONFIG, fold.child, torch.float64)
        with torch.no_grad():
            up = base.upsampled(torch.from_numpy(x))
            shifted = torch.tensor(fold.k).view(1, 3, 1, 1) * (up - torch.tensor(t * mean).view(1, 3, 1, 1))
            target = base.bn1(base.conv1(shifted))
            ours = child.bn1(child.conv1(up))
        interior = (slice(None), slice(None), slice(2, -2), slice(2, -2))  # conv1 is 7 x 7, stride 2, padding 3
        assert torch.max(torch.abs(ours[interior] - target[interior])) < 1e-4  # float32 storage of the folded tensors
        assert torch.max(torch.abs(ours - target)) > 1e-3, "the zero-padding border deviation must be visible"


def test_dense_delta_bytes_is_the_size_of_the_saved_float64_files(tmp_path: Path) -> None:
    shapes = [(64, 3, 7, 7), (64,), (10, 512), (512, 256, 3, 3)]
    for i, shape in enumerate(shapes):
        np.save(tmp_path / f"t{i}.npy", np.zeros(shape, dtype=np.float64))
    assert dense_delta_bytes(shapes) == sum((tmp_path / f"t{i}.npy").stat().st_size for i in range(len(shapes)))


def test_the_file_cap_lets_a_full_float64_delta_of_resnet18_through(tmp_path: Path) -> None:
    """PREREG E12 (2026-11-05): the 64 MiB format cap refused every dense candidate on ResNet-18 in R17 236542."""
    params = _student(6)
    dense = dense_delta_bytes(v.shape for v in params.values())
    assert dense > 64 * 2**20
    task = task_limits(Limits())
    ours = e10_program_limits(task, len(params), dense)
    assert ours == applied_limits(task, "LEARN", len(params), dense)  # references and LEARN sessions: one cap
    assert ours.max_file_bytes == dense and ours.max_ops == len(params)
    assert e10_program_limits(task, len(params)).max_file_bytes == 64 * 2**20
    assert applied_limits(task, "M2", len(params), dense).max_file_bytes == 64 * 2**20  # patch arms unchanged
    small = {"w": np.zeros((16, 3, 3, 3), np.float32), "b": np.zeros(16, np.float32)}  # a student whose delta fits
    assert e10_program_limits(task, 2, dense_delta_bytes(v.shape for v in small.values())) == e10_program_limits(task, 2)
    rng = np.random.default_rng(6)
    child = {k: (v + rng.normal(0.0, 1e-3, v.shape)).astype(v.dtype) for k, v in params.items()}
    sha = params_sha256(params)
    path = write_delta_patch(tmp_path / "full", sha, params, child, "h", "p")
    executor = ours.without_size_limits()
    result = apply_patch(params, load_patch(path), path.parent, executor, sha)
    assert result.child_sha256 == params_sha256(child)
    with pytest.raises(PatchError, match="above the limit of 67108864 bytes"):
        apply_patch(params, load_patch(path), path.parent, e10_program_limits(task, len(params)).without_size_limits(),
                    sha)


def _instance(tmp_path: Path, params: dict[str, np.ndarray]) -> SimpleNamespace:
    (tmp_path / "model_spec.json").write_text(json.dumps(_spec(params)))
    return SimpleNamespace(instance_dir=tmp_path, params=params, forward=_forward(),
                           private={"dataset": C.E12_DATASET})


def _evidence(seed: int, n_replay: int = 0) -> Evidence:
    n = 64 + n_replay
    x = _images(n, seed)
    return Evidence(x=x, y=np.arange(n, dtype=np.int64) % 10, n_fail=16, n_rand=48, sha256="0" * 64,
                    base_pred=np.zeros(n, dtype=np.int64), rule="evidence-v1", n_replay=n_replay)


def test_bn_adapt_changes_running_statistics_only(tmp_path: Path) -> None:
    params = _student(3)
    instance = _instance(tmp_path, params)
    candidates, info = B.bn_adapt_generator(instance)(_evidence(3))
    assert [c.config for c in candidates] == [{"N": n} for n in B.BN_ADAPT_PRIORS]
    stats = {k for k in params if k.endswith(("running_mean", "running_var"))}
    for candidate in candidates:
        assert {k for k in params if not np.array_equal(candidate.child[k], params[k])} <= stats
        assert candidate.notes["delivered_vs_adapted_max_abs_logit_diff"] < 1e-4
    assert info["architecture"]["family"] == TF.RESNET18UP


def test_f1_lora_on_convolutions_and_ft_bn(tmp_path: Path) -> None:
    params = _student(4)
    arch = _arch(params)
    targets = FT.lora_targets(arch, params)
    assert len(targets) == 21 and targets[0] == "conv1.weight" and targets[-1] == "fc.weight"
    for rank, expected in ((4, 145_780), (16, 583_120)):
        assert sum(rank * sum(FT.lora_matrix_shape(params[k].shape)) for k in targets) == expected
    a, b = FT.lora_init(params, targets, 4, 0)
    assert a["conv1.weight"].shape == (4, 3 * 7 * 7) and b["conv1.weight"].shape == (64, 4)
    assert float(a["conv1.weight"].detach().abs().max()) <= 1.0 / np.sqrt(147)
    x = torch.from_numpy(_images(4, 4))
    y = torch.arange(4) % 10
    ((steps, child, _),) = FT.lora_checkpoints(arch, params, x, y, targets=targets, rank=4, lr=1e-3,
                                               checkpoints=(1,), seed=0)
    assert steps == 1 and all(child[k].shape == params[k].shape for k in params)
    changed = {k for k in params if not np.array_equal(child[k], params[k])}
    assert changed and changed <= set(targets)
    assert FT.f1_configs("ft_bn") == FT.f1_configs("ft_ln")
    assert FT.F1_DATASET_METHODS[C.E12_DATASET] == FT.F1_E12_METHODS
    instance = _instance(tmp_path, params)
    with pytest.raises(FT.F1Error, match="does not apply"):
        FT.f1_generator(instance, "ft_ln", 0)(_evidence(5, 64))


def test_cli_dispatch_for_e12(tmp_path: Path) -> None:
    instance = SimpleNamespace(private={"dataset": C.E12_DATASET})
    assert C.E12_METHODS == C.E10_METHODS and "bn_adapt" in C.E12_METHODS
    with pytest.raises(ValueError, match="runs with e12"):
        C.run_method(instance, "grad_last", tmp_path, run_seed=0, allow_main=False, joint=True, e10=True)
    with pytest.raises(ValueError, match="LayerNorm"):
        C.run_method(instance, "tent_ln", tmp_path, run_seed=0, allow_main=False, joint=True, e10=True, e12=True)
    other = SimpleNamespace(private={"dataset": "cifar10_r20"})
    with pytest.raises(ValueError, match="need a cifar10_r18"):
        C.run_method(other, "grad_last", tmp_path, run_seed=0, allow_main=False, joint=True, e10=True, e12=True)


def test_harness_student_check_and_timeouts() -> None:
    params = _student(5)
    summary = H.check_student(params)
    assert summary["n_tensors"] == 102 and summary["n_rel_frob_tensors"] == 21
    assert summary["n_exempt_1d_tensors"] == 81 and summary["n_batchnorm_stat_entries"] == 9_600
    broken = dict(params)
    broken.pop("bn1.running_var")
    with pytest.raises(H.E12HarnessError):
        H.check_student(broken)
    private = {"dataset": H.DATASET, "evidence_rule": "evidence-v1"}
    assert H.is_e12(private) and not H.is_e12({**private, "dataset": "cifar10_r20"})
    assert H.mcp_timeouts(private, None, None, (1, 1)) == (H.MCP_TOOL_TIMEOUT_MS, H.MCP_CONNECT_TIMEOUT_MS)
    assert H.mcp_timeouts(private, 2_400_000, 1_800_000, (1, 1)) == (2_400_000, 1_800_000)
    with pytest.raises(ValueError):
        H.mcp_timeouts(private, 600_000, None, (1, 1))
    assert H.leak_words() == sorted(H.leak_words()) and "imagenet-c" in H.leak_words()


def test_editor_visible_text_has_no_leak_token() -> None:
    tokens = set(leak_tokens(H.DATASET)) | set(H.leak_words())
    for condition, corruption in (("gaussian_noise_s5", "gaussian_noise"), ("contrast_s5", "contrast"),
                                  ("defocus_blur_s5", "defocus_blur"), ("snow_s5", "snow")):
        tokens |= set(condition_leak_tokens(condition, {"condition": {"corruption": corruption}}))
    assert scan_for_tokens(_forward_text(), tokens) == []
    assert scan_for_tokens(json.dumps(_spec(_student(6))), tokens) == []
    assert scan_for_tokens(" ".join(M.param_shapes(10)), tokens) == []


def test_render_refuses_a_size_off_the_grid() -> None:
    with pytest.raises(ValueError):
        render_resnet18up_forward(M.IMAGENET_MEAN, M.IMAGENET_STD, 10, M.BN_EPS, 40)
    with pytest.raises(ValueError):
        M.ResNet18UpConfig(resize=112)
