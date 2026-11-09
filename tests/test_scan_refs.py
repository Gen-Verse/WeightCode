"""E13 scanskill-v1 instances, the scan joint oracle, the readout and the references on a toy store (``scan_toy``).

The store is written from the grammar and its students are tiny (the gate is set to 0 there), so these tests check
the instance contract, the references' mechanics and the readout arithmetic, never a number of record.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import scan_toy

from weightcode.baselines.fit import scanrefs as R
from weightcode.baselines.instance_io import load_instance
from weightcode.data import scan as S
from weightcode.evalkit import devreadout as D
from weightcode.evalkit import scanskill as K

FAST_STEPS = (1, 2, 3, 4)


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    """The toy store with one tiny base per condition and its scanskill-v1 instance (condition -> instance id)."""
    patcher = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("scan_store")
    try:
        scan_toy.store_with_students(tmp, patcher)
        ids = {}
        for condition in S.CONDITIONS:
            (row,) = K.build_scan_instances(condition)
            ids[condition] = row["instance_id"]
        patcher.setattr(R, "STEP_CHECKPOINTS", FAST_STEPS)
        patcher.setattr(R, "REG_STEPS", (1, 2))
        patcher.setattr(R, "TRAJ_STEPS", 3)
        yield ids
    finally:
        patcher.undo()


def _instance(store: dict[str, str], condition: str):
    return load_instance(K.scan_instance_dir(store[condition]))


def test_instances_follow_the_contract(store: dict[str, str]) -> None:
    for condition, instance_id in store.items():
        assert K.INSTANCE_ID_RE.fullmatch(instance_id)
        instance = _instance(store, condition)
        K.check_scan_current(instance.instance_dir)
        private = instance.private
        assert private["evidence_rule"] == K.SCAN_RULE and private["split_role"] == "dev"
        assert private["n_replay"] == K.N_REPLAY == 64
        evidence = instance.evidence
        assert sorted(evidence) == sorted(K.evidence_keys(condition))
        if condition == "addprim_jump":
            assert S.decode(evidence["x_prim"]) == ["jump"] and private["n_prim"] == 1
        else:
            assert "x_prim" not in evidence and private["n_prim"] == 0
        old = S.decode(evidence["x_old"])
        assert not any(S.has_primitive(c, condition) for c in old) and "jump" not in old
        visible = "\n".join((instance.instance_dir / name).read_text() for name in ("task.json", "model_spec.json"))
        assert not any(word.lower() in visible.lower() for word in K.LEAK_WORDS)
        # A rebuild gives the same evidence and inputs, so the active instance is reused.
        (again,) = K.build_scan_instances(condition)
        assert again["instance_id"] == instance_id


def test_oracle_scores_the_base_as_zero_and_counts_checks(store: dict[str, str]) -> None:
    instance = _instance(store, "around_right")
    oracle = K.make_scan_oracle(instance, budget=2)
    entry = oracle.query(dict(instance.params), "p0")
    assert (entry["score"], entry["score_new"], entry["score_old"]) == (0, 0, 0)
    assert entry["n"] == instance.private["n_vcheck_new"] + instance.private["n_vcheck_old"]
    assert oracle.queries_remaining == 1
    readout = K.scan_readout(instance, dict(instance.params))
    assert readout["joint_points"] == 0.0 and readout["new"]["n"] == instance.private["n_vselect_new"]


def test_methods_and_grids() -> None:
    assert "seq_grad_new_row" not in R.condition_methods("around_right")
    assert "ft_full_noreplay" not in R.condition_methods("around_right")
    assert set(R.condition_methods("addprim_jump")) == set(R.METHODS)
    assert R.run_label("lora_r4") == "f1" and R.run_label("rulegen_grad_full") == "e13"
    assert all(len(R.configs(m)) == 8 for m in R.METHODS)
    a = R.rulegen_fit_set("addprim_jump", "mixed")
    b = R.rulegen_fit_set("addprim_jump", "mixed")
    assert a["record"]["commands_sha256"] == b["record"]["commands_sha256"] and len(a["commands"]) == 4000
    assert all(S.has_primitive(c, "addprim_jump") for c in a["commands"][:2000])
    assert not any(S.has_primitive(c, "addprim_jump") for c in a["commands"][2000:])
    assert len(set(a["commands"])) == 4000


def test_rule_rows_match_the_official_labels(store: dict[str, str]) -> None:
    from weightcode.data.scan_build import raw_dir

    fit = R.rulegen_fit_set("around_right", "full")
    record = R.check_rows_against_official(raw_dir() / S.ALL_FILE, fit["commands"], fit["y"])
    assert record["mismatches"] == 0 and record["n_rows"] == 2000
    with pytest.raises(R.ScanRefError):
        R.check_rows_against_official(raw_dir() / S.ALL_FILE, fit["commands"][::-1], fit["y"])


def _run(store: dict[str, str], condition: str, method: str, tmp_path: Path, fit=None) -> tuple[dict, Path]:
    instance = _instance(store, condition)
    run_dir = tmp_path / method
    record = R.run_reference(instance, method, fit_set=fit, run_dir=run_dir)
    return record, run_dir / "run.json"


def test_seqgrad_reference_through_the_scan_path(store: dict[str, str], tmp_path: Path) -> None:
    record, path = _run(store, "addprim_jump", "seq_grad_new_row", tmp_path)
    assert record["scan"]["condition"] == "addprim_jump" and record["limits_source"] == K.SCAN_LIMITS_SOURCE
    assert len(record["oracle_log"]) == 8 and record["flags"]["edit_limits_enforced"]
    instance = _instance(store, "addprim_jump")
    readout = D.readout_run(path)
    assert set(readout["readouts_scan"]) >= {"new", "old", "new_by_construction", "joint_points"}
    kept = D.readout_retention(path, s_old_floor=-5)
    assert kept["selection"]["s_old_floor"] == -5
    # Only the jump row of tok_emb may change.
    if not readout["is_base_noop"]:
        child = D.final_child(json.loads(path.read_text()), path.parent, instance,
                              K.scan_program_limits(instance.task["limits"], len(instance.params)))
        changed = {k for k in child if not np.array_equal(child[k], instance.params[k])}
        assert changed <= {R.EMBEDDING}
        rows = np.flatnonzero(np.any(child[R.EMBEDDING] != instance.params[R.EMBEDDING], axis=1))
        assert set(rows.tolist()) <= {S.TOKEN_ID["jump"]}


def test_f1_lora_and_lwf_on_around_right(store: dict[str, str], tmp_path: Path) -> None:
    instance = _instance(store, "around_right")
    targets = R.lora_targets(instance.params)
    assert len(targets) == 6 and all(t.startswith("b1.") for t in targets)
    record, path = _run(store, "around_right", "lora_r4", tmp_path)
    assert not record["flags"]["edit_limits_enforced"] and len(record["oracle_log"]) == 8
    frontier = D.readout_candidates(path)
    assert [r["config"] for r in frontier] == R.configs("lora_r4")
    assert [r["vcheck"]["score"] for r in frontier] == [e["score"] for e in record["oracle_log"]]
    run = json.loads(path.read_text())
    if not run["final"]["is_base_noop"]:
        limits = K.scan_program_limits(instance.task["limits"], len(instance.params))
        child = D.final_child(run, path.parent, instance, limits)
        assert {k for k in child if not np.array_equal(child[k], instance.params[k])} <= set(targets)
    assert K.main(["ids", "--condition", "around_right"]) == 0
    record, path = _run(store, "around_right", "ft_full_lwf", tmp_path)
    assert [c["config"] for c in record["candidates"]] == R.configs("ft_full_lwf")
    assert D.readout_run(path)["method"] == "ft_full_lwf"
    with pytest.raises(R.ScanRefError):
        R.scan_generator(instance, "ft_full_noreplay", 0)


def test_rulegen_reference_reports_fit_coverage(store: dict[str, str], tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R, "N_RULEGEN", 200)
    monkeypatch.setattr(R, "N_RULEGEN_OLD", 100)
    fit = R.rulegen_fit_set("addprim_jump", "mixed")
    record, path = _run(store, "addprim_jump", "rulegen_mixed_sparse19_sal", tmp_path, fit)
    assert len(record["config"]["fit_set"]["command_ids"]) == 300
    coverage = D.readout_run(path)["readouts_scan"]["fit_coverage"]
    assert coverage["n_in_fit_set"] + coverage["n_not_in_fit_set"] == coverage["n_new"]
    assert coverage["n_in_fit_set"] > 0
    with pytest.raises(R.ScanRefError):
        R.run_reference(_instance(store, "addprim_jump"), "rulegen_grad_full", fit_set=fit, run_dir=tmp_path / "x")


def test_walk_diagnostic(store: dict[str, str]) -> None:
    result = R.walk_diagnostic(_instance(store, "addprim_jump"))
    assert result["new_relabelled"]["n"] == _instance(store, "addprim_jump").private["n_vselect_new"]
    assert result["old"]["n"] == 1000
    with pytest.raises(R.ScanRefError):
        R.walk_diagnostic(_instance(store, "around_right"))


def test_cli_check_labels_and_refs(store: dict[str, str], tmp_path: Path) -> None:
    import importlib.util

    import torch

    threads = torch.get_num_threads()
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_scan_refs.py"
    spec = importlib.util.spec_from_file_location("run_scan_refs", script)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    out = tmp_path / "labels.json"
    try:
        assert cli.main(["check-labels", "--out", str(out), "--threads", "2"]) == 0
        labels = json.loads(out.read_text())
        assert set(labels["fits"]) == {f"{c}/{f}" for c in S.CONDITIONS for f in ("full", "mixed")}
        assert all(v["official_check"]["mismatches"] == 0 for v in labels["fits"].values())
        assert cli.main(["refs", "--instance", store["around_right"], "--methods", "f1_last", "--threads", "2"]) == 0
        assert cli.main(["refs", "--instance", store["around_right"], "--methods", "seq_grad_new_row"]) != 0
        assert cli.main(["walk", "--instance", store["addprim_jump"], "--out", str(tmp_path / "walk.json")]) == 0
        assert cli.main(["refs", "--instance", store["addprim_jump"], "--methods", "rulegen_grad_full", "--threads",
                         "2", "--label-record", str(out)]) == 0
    finally:
        torch.set_num_threads(threads)


CUDA = pytest.param("cuda", marks=pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA"))


@pytest.fixture
def device_flags(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore torch's global determinism and precision flags and the thread and cuBLAS variables after a test that
    configures a device."""
    import os

    import torch

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"):
        if variable in os.environ:
            monkeypatch.setenv(variable, os.environ[variable])
        else:
            monkeypatch.delenv(variable, raising=False)
    saved = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
             torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.fp32_precision,
             torch.backends.cudnn.conv.fp32_precision, torch.get_num_threads())
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = saved[1], saved[2]
        torch.backends.cuda.matmul.fp32_precision, torch.backends.cudnn.conv.fp32_precision = saved[3], saved[4]
        torch.set_num_threads(saved[5])


def test_device_settings_refuse_an_unprepared_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    assert R.configure_device("cpu") == {"device": "cpu"} == R.device_record("cpu")
    with pytest.raises(R.ScanRefError):
        R.configure_device("tpu")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(R.ScanRefError):
        R.configure_device("cuda")


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA")
def test_gpu_fits_match_the_cpu_and_run_every_kind(store: dict[str, str], tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch, device_flags: None) -> None:
    from weightcode.baselines.fit import seqgrad as SG
    from weightcode.baselines.fit.runner import load_scan_evidence

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", R.CUBLAS_WORKSPACE)
    settings = R.configure_device("cuda")
    assert settings["use_deterministic_algorithms"] and settings["cuda_matmul_fp32_precision"] == "ieee"
    instance = _instance(store, "addprim_jump")
    evidence = load_scan_evidence(instance)
    x, y = R._rows(evidence.x, evidence.y)
    runs = {device: SG.adam_checkpoints(instance.params, x, y, trainable=sorted(instance.params), lr=1e-2,
                                        checkpoints=(5, 20), seed=0, device=device) for device in ("cpu", "cuda")}
    # Both are IEEE float32; the GPU's summation order differs, and Adam turns the noise on near-zero gradients into
    # steps of up to lr, so entries drift apart over many steps while the loss agrees (toy student: within 1e-4 at
    # 5 steps entrywise, losses within 1e-4 at 20 steps of lr 1e-2).
    for (s_cpu, p_cpu, l_cpu), (s_gpu, p_gpu, l_gpu) in zip(runs["cpu"], runs["cuda"], strict=True):
        assert s_cpu == s_gpu and abs(l_cpu - l_gpu) < 1e-4
        assert all(p_gpu[k].dtype == np.float32 and p_gpu[k].shape == p_cpu[k].shape for k in p_cpu)
    (_, p_cpu, _), (_, p_gpu, _) = runs["cpu"][0], runs["cuda"][0]
    assert all(np.allclose(p_gpu[k], p_cpu[k], atol=1e-4) for k in p_cpu)
    fit = R.rulegen_fit_set("addprim_jump", "mixed")
    for method in ("seq_grad_sparse19_traj", "seq_grad_sparse19_sal", "lora_r4", "ft_full_lwf", "ft_full_l2sp",
                   "rulegen_mixed_grad_full"):
        record = R.run_reference(instance, method, fit_set=fit if method in R.RULEGEN_METHODS else None,
                                 run_dir=tmp_path / method, device="cuda")
        assert record["status"] == "complete" and record["config"]["device"] == "cuda"
        assert record["config"]["device_settings"]["use_deterministic_algorithms"]
        generation = record["generation"]
        assert generation["device"] == "cuda"
        assert generation["device_parity_max_abs_logit_diff"] <= generation["parity_bound"]
        assert len(record["oracle_log"]) == 8


@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_smoke_script_time_and_paths(store: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                     device_flags: None, device: str) -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "store" / "e13_smoke.py"
    spec = importlib.util.spec_from_file_location("e13_smoke", script)
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    assert smoke.main(["time", "--condition", "around_right", "--threads", "2", "--device", device,
                       "--steps", "1"]) == 0
    assert smoke.main(["paths", "--condition", "addprim_jump", "--method", "ft_full_l2sp", "--threads", "2",
                       "--device", device, "--out-root", str(tmp_path)]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    kinds = [line["kind"] for line in lines]
    assert kinds.count("adam_full") == 3 and "load" in kinds and "opened" in kinds
    (paths,) = [line for line in lines if line["kind"] == "paths"]
    assert paths["status"] == "complete"
    run = tmp_path / store["addprim_jump"] / "ft_full_l2sp" / "run.json"
    assert json.loads(run.read_text())["config"]["device"] == device
    assert D.readout_retention(run, s_old_floor=-5)["selection"]["s_old_floor"] == -5
    assert len(D.readout_candidates(run)) == 8
