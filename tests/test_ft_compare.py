"""PREREG F1's fine-tuning methods (``weightcode.baselines.fit.ftcompare``) on a ViT-Lite-7/4 student, and the dev readout's
descriptive frontier (``devreadout.readout_candidates``) on the toy joint store."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from test_baselines_fit_joint import CHILDREN, _run, joint_instance  # noqa: F401
from test_e11_refs import _arch, _images, _instance, _student
from test_evalkit_instances_store import toy_store  # noqa: F401

from weightcode.baselines.fit import ftcompare as FT
from weightcode.baselines.fit import lnrefs as L
from weightcode.baselines.fit.runner import Evidence
from weightcode.baselines.fit.torch_forward import logits, to_tensors
from weightcode.baselines.instance_io import Instance
from weightcode.evalkit import devreadout


def _joint_evidence(seed: int, n_replay: int = 64) -> Evidence:
    x = _images(64 + n_replay, seed)
    y = np.random.default_rng(seed).integers(0, 10, 64 + n_replay).astype(np.int64)
    return Evidence(x=x, y=y, n_fail=16, n_rand=48, sha256="0" * 64, base_pred=np.zeros(64 + n_replay, dtype=np.int64),
                    rule="evidence-v1", n_replay=n_replay, replay_sha256="1" * 64 if n_replay else None)


def _short(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short trajectories: the grid's shape and the plumbing are tested, not the lengths."""
    monkeypatch.setattr(FT, "STEP_CHECKPOINTS", (1, 2, 3, 4))
    monkeypatch.setattr(FT, "REG_STEPS", (1, 3))


def _changed(params: dict[str, np.ndarray], child: dict[str, np.ndarray]) -> set[str]:
    return {k for k in params if not np.array_equal(child[k], params[k])}


def test_grids_are_the_prereg_grids() -> None:
    steps = [10, 30, 100, 300]
    for method in ("ft_ln", "ft_full_noreplay"):
        assert FT.f1_configs(method) == [{"lr": lr, "steps": s} for lr in (1e-5, 1e-4) for s in steps]
    for method in ("lora_r4", "lora_r16"):
        assert FT.f1_configs(method) == [{"lr": lr, "steps": s} for lr in (1e-4, 1e-3) for s in steps]
    assert FT.f1_configs("ft_full_lwf") == [{"weight": w, "lr": lr, "steps": s}
                                            for w in (1.0, 10.0) for lr in (1e-5, 1e-4) for s in (30, 300)]
    assert FT.f1_configs("ft_full_l2sp") == [{"weight": w, "lr": lr, "steps": s}
                                             for w in (1e-2, 1e-1) for lr in (1e-5, 1e-4) for s in (30, 300)]
    assert all(len(FT.f1_configs(m)) == 8 for m in FT.F1_METHODS)
    assert not any(spec.edit_limits_enforced for spec in FT.F1_SPECS.values())
    assert FT.F1_REUSED == {"full": ("grad_full", "e11g2"), "last layer": ("grad_last", "e11g2")}
    with pytest.raises(FT.F1Error, match="unknown F1 method"):
        FT.f1_configs("grad_full")


@pytest.mark.parametrize("method", ["ft_ln", "lora_r4", "ft_full_l2sp"])
def test_candidates_change_only_their_tensors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    _short(monkeypatch)
    params = _student(7)
    instance = _instance(tmp_path, params)
    candidates, info = FT.f1_generator(instance, method, 0)(_joint_evidence(7))
    assert [c.config for c in candidates] == FT.f1_configs(method) and len(candidates) == 8
    arch = _arch(params)
    allowed = {"ft_ln": set(L.ln_affine(arch)), "lora_r4": set(FT.lora_targets(arch, params)),
               "ft_full_l2sp": set(params)}[method]
    assert set(info["trainable"]) == allowed
    for c in candidates:
        changed = _changed(params, c.child)
        assert changed and changed <= allowed
        assert all(c.child[k].dtype == np.float32 for k in c.child)
    if method == "lora_r4":
        assert len(allowed) == 28 and info["lora"]["rank"] == 4 and info["lora"]["scale"] == 1.0
        for c in candidates:
            for k in allowed:
                delta = c.child[k].astype(np.float64) - params[k].astype(np.float64)
                s = np.linalg.svd(delta, compute_uv=False)  # float32 rounding of W0 + BA adds a full-rank floor
                assert s[0] > 0 and int((s > 1e-3 * s[0]).sum()) <= 4
    if method == "ft_full_l2sp":
        for c in candidates:
            parts = c.notes["objective_at_checkpoint"]
            assert parts["penalty"] > 0 and parts["total"] == pytest.approx(parts["ce"] + parts["penalty"], rel=1e-5)


def test_lora_init_is_seeded_and_merging_is_the_training_forward(tmp_path: Path) -> None:
    params = _student(8)
    arch = _arch(params)
    targets = FT.lora_targets(arch, params)
    a1, b1 = FT.lora_init(params, targets, 16, 0)
    a2, _ = FT.lora_init(params, targets, 16, 0)
    a3, _ = FT.lora_init(params, targets, 16, 1)
    assert all(torch.equal(a1[k], a2[k]) for k in targets) and not torch.equal(a1[targets[0]], a3[targets[0]])
    assert all(float(b1[k].abs().max()) == 0.0 for k in targets)
    assert all(float(a1[k].abs().max()) <= 1 / np.sqrt(params[k].shape[1]) for k in targets)
    x, y = torch.from_numpy(_images(16, 8)), torch.from_numpy(np.arange(16) % 10)
    ((_, child, loss),) = FT.lora_checkpoints(arch, params, x, y, targets=targets, rank=16, lr=1e-3,
                                              checkpoints=(2,), seed=0)
    with torch.no_grad():
        assert float(F.cross_entropy(logits(arch, to_tensors(child), x), y)) == pytest.approx(loss, rel=1e-6)


def test_noreplay_fits_the_condition_rows_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _short(monkeypatch)
    instance = _instance(tmp_path, _student(9))
    with pytest.raises(FT.F1Error, match="only ft_full_noreplay fits without them"):
        FT.f1_generator(instance, "ft_full_noreplay", 0)(_joint_evidence(9))
    with pytest.raises(FT.F1Error, match="only ft_full_noreplay fits without them"):
        FT.f1_generator(instance, "ft_ln", 0)(_joint_evidence(9, n_replay=0))
    candidates, info = FT.f1_generator(instance, "ft_full_noreplay", 0)(_joint_evidence(9, n_replay=0))
    assert len(candidates) == 8 and info["fit_rows"].startswith("64 rows")
    assert FT.f1_config("ft_full_noreplay", 0)["fit_rows"] == "the 64 condition rows"


def test_penalties_vanish_at_the_base_and_have_their_stated_form() -> None:
    params = _student(10)
    arch = _arch(params)
    trainable = sorted(params)
    tensors = to_tensors(params)
    x = torch.from_numpy(_images(24, 10))
    with torch.no_grad():
        out = logits(arch, tensors, x)
    assert float(FT.l2sp_penalty(params, trainable, 0.1)(tensors, out)) == 0.0
    assert float(FT.lwf_penalty(out[16:], 16, 10.0)(tensors, out)) == pytest.approx(0.0, abs=1e-6)
    moved = dict(tensors)
    key = "classifier.fc.bias"
    moved[key] = tensors[key] + torch.tensor(np.eye(10, dtype=np.float32)[0] * 0.5)
    assert float(FT.l2sp_penalty(params, trainable, 0.1)(moved, out)) == pytest.approx(0.5 * 0.1 * 0.25, rel=1e-5)
    shifted = out.clone()
    shifted[16:, 0] += 3.0
    p = F.softmax(out[16:], 1)
    expected = 10.0 * float((p * (F.log_softmax(out[16:], 1) - F.log_softmax(shifted[16:], 1))).sum(1).mean())
    assert float(FT.lwf_penalty(out[16:], 16, 10.0)(tensors, shifted)) == pytest.approx(expected, rel=1e-5)


def test_lwf_candidates_record_both_parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _short(monkeypatch)
    params = _student(11)
    candidates, info = FT.f1_generator(_instance(tmp_path, params), "ft_full_lwf", 0)(_joint_evidence(11))
    assert [c.config for c in candidates] == FT.f1_configs("ft_full_lwf")
    assert info["lwf"]["temperature"] == 1.0
    for c in candidates:
        parts = c.notes["objective_at_checkpoint"]
        assert parts["penalty"] >= 0 and parts["total"] == pytest.approx(parts["ce"] + parts["penalty"], rel=1e-5)


def test_run_f1_refuses_other_datasets_and_methods() -> None:
    other = SimpleNamespace(private={"dataset": "cifar10_r20"})
    with pytest.raises(FT.F1Error, match="cifar10_vl7"):
        FT.run_f1(other, "ft_ln", Path("/nonexistent"))
    with pytest.raises(FT.F1Error, match="unknown F1 method"):
        FT.run_f1(SimpleNamespace(private={"dataset": FT.F1_DATASET}), "grad_full", Path("/nonexistent"))


def test_frontier_reads_every_checked_candidate(joint_instance: Instance, tmp_path: Path) -> None:  # noqa: F811
    record, _ = _run(joint_instance, tmp_path / "run", replay_in_fit=True)
    run_path = tmp_path / "run" / "run.json"
    frontier = devreadout.readout_candidates(run_path)
    log = record["oracle_log"]
    assert [r["patch_id"] for r in frontier] == [e["patch_id"] for e in log] and len(frontier) == len(CHILDREN)
    for r, entry, make in zip(frontier, log, CHILDREN):
        expected = devreadout.dev_readout_joint(joint_instance, make(dict(joint_instance.params)))
        assert r["readouts"] == expected["readouts"] and r["readouts_clean"] == expected["readouts_clean"]
        assert r["vcheck"]["score"] == entry["score"] and r["child_sha256"] == entry["child_sha256"]
        assert r["label"] == devreadout.FRONTIER_LABEL
    if not record["final"]["is_base_noop"]:
        own = devreadout.readout_run(run_path, clean=True)
        (picked,) = [r for r in frontier if r["patch_id"] == record["final"]["patch_id"]]
        assert own["readouts"] == picked["readouts"] and own["readouts_clean"] == picked["readouts_clean"]
    out = tmp_path / "frontier.jsonl"
    assert devreadout.main([str(run_path), "--all-candidates", "--out", str(out)]) == 0
    assert [json.loads(line)["patch_id"] for line in out.read_text().splitlines()] == [e["patch_id"] for e in log]
    with pytest.raises(SystemExit):
        devreadout.main([str(run_path), "--all-candidates", "--clean"])
