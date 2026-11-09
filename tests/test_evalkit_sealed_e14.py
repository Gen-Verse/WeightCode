"""Tests for the policy_success family of the sealed read (weightcode.evalkit.sealed_e14, PREREG E14) on a scratch store.

The store and instance are test_e14_refs's (an untrained film_policy recorded as a gated base, V_check and V_select
cut to 6 episodes each). This module adds a sealed record of 3 old and 3 new seeds (``write_sealed_record`` with the
counts patched) and its dataset.json ``sealed_record``, a hand-written session whose info.json holds
``e14.e14_readout``, and a hand-written joint reference run under the E14 program limits (two checks, the kept pick at
S_old >= -3 not its own final) with its ``e14read.read_run`` record. These tests need the E14 code
and minigrid (branch e14-babyai) and are skipped in a tree without them.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("minigrid")
pytest.importorskip("weightcode.evalkit.e14")

from test_e14_refs import instance_id, store  # noqa: E402, F401

from weightcode.baselines.fit.e14refs import E14_REF_LABEL  # noqa: E402
from weightcode.baselines.fit.runner import baseline_run_dir  # noqa: E402
from weightcode.baselines.instance_io import write_delta_patch  # noqa: E402
from weightcode.common.identity import params_sha256  # noqa: E402
from weightcode.common.paths import runs_dir, sealed_dir  # noqa: E402
from weightcode.data import e14 as D  # noqa: E402
from weightcode.evalkit import devreadout  # noqa: E402
from weightcode.evalkit import e14 as E  # noqa: E402
from weightcode.evalkit import sealed_e14 as SE  # noqa: E402
from weightcode.evalkit import sealed_finals as SF  # noqa: E402
from weightcode.evalkit import e14read as ER  # noqa: E402
from weightcode.evalkit.e14read import read_run  # noqa: E402
from weightcode.evalkit.sealed import read_registry  # noqa: E402
from weightcode.patch import Limits, task_limits  # noqa: E402

SESSION = "s0e14000000000001"
METHOD = "bc_emb_head"
COMMIT = "0" * 40
SEALED_COUNTS = {"old": 3, "new": 3}


@dataclasses.dataclass
class PolicyToy:
    instance_id: str
    session_child: dict[str, np.ndarray]
    ref_run: str
    ref_record: str
    ref_child_sha256: str


def _changed(params: dict[str, np.ndarray], name: str, entries: slice, step: float) -> dict[str, np.ndarray]:
    child = {key: value.copy() for key, value in params.items()}
    flat = child[name].reshape(-1)
    flat[entries] += step
    return child


def _reference_run(instance: Any) -> tuple[Path, str]:
    """A hand-written joint E14 reference run under the E14 program limits with two checks: q0 (S_old 0, kept at
    S_old >= -3) and q1 (S_joint 5 at S_old -5, its own final); returns run.json and the kept child's hash."""
    run_dir = baseline_run_dir(instance.instance_id, METHOD, 0, E14_REF_LABEL)
    names = sorted(instance.params)
    children = (_changed(instance.params, names[0], slice(0, 4), 0.05), _changed(instance.params, names[1],
                                                                                    slice(0, 3), -0.05))
    log, candidates = [], []
    for query, (child, score_new, score_old) in enumerate(zip(children, (2, 10), (0, -5))):
        patch_id = f"q{query}"
        write_delta_patch(run_dir / patch_id, instance.params_sha256, instance.params, child, "toy", "toy")
        n_changed = sum(int(np.count_nonzero(child[k] != instance.params[k])) for k in child)
        log.append({"patch_id": patch_id, "query_index": query, "child_sha256": params_sha256(child),
                    "n_changed": n_changed, "score": score_new + score_old, "score_new": score_new,
                    "score_old": score_old, "fixed": score_new, "broken": -score_old, "fixed_new": score_new,
                    "broken_new": 0, "fixed_old": 0, "broken_old": -score_old})
        candidates.append({"patch_id": patch_id, "config": {"q": query},
                           "edit": {"edit_size": n_changed, "within_limits": True}})
    own = log[1]
    run = {"method": METHOD, "e14": {"rule": E.POLICYSKILL_RULE},
           "instance": {"instance_id": instance.instance_id, "params_sha256": instance.params_sha256,
                        "forward_sha256": instance.forward_sha256},
           "limits": task_limits(ER.program_limits(instance)), "flags": {"edit_limits_enforced": True},
           "oracle_log": log, "candidates": candidates,
           "final": {"patch_id": "q1", "is_base_noop": False, "dir": "q1", "query_index": 1,
                     "child_sha256": own["child_sha256"], "edit_size": own["n_changed"],
                     **{key: own[key] for key in ("score", "fixed", "broken", "score_new", "score_old")}}}
    (run_dir / "run.json").write_text(json.dumps(run))
    return run_dir / "run.json", log[0]["child_sha256"]


@pytest.fixture(scope="module")
def toy(store, instance_id: str) -> PolicyToy:  # noqa: F811
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    patcher = pytest.MonkeyPatch()
    try:
        patcher.setattr(D, "SEALED_COUNTS", SEALED_COUNTS)
        sealed = D.write_sealed_record()
        (D.store_dir() / "dataset.json").write_text(json.dumps({"dataset": D.DATASET, "sealed_record": sealed}))
    finally:
        patcher.undo()
    instance = E.load_e14_instance(instance_id)
    run_path, ref_child = _reference_run(instance)
    record = read_run(run_path)
    assert record["kept"]["patch_id"] == "q0" and record["own"]["patch_id"] == "q1"
    ref_record = "devreadouts/e14_read_toy_bc_emb_head.jsonl"
    (runs_dir() / ref_record).parent.mkdir(parents=True, exist_ok=True)
    (runs_dir() / ref_record).write_text(json.dumps(devreadout.json_safe(record)) + "\n")
    child = {key: value.copy() for key, value in instance.params.items()}
    name = sorted(child)[0]
    child[name] = child[name] + 0.01
    root = runs_dir() / "sessions" / SESSION
    write_delta_patch(root / "final", instance.params_sha256, instance.params, child, "toy", "toy")
    (root / "session.json").write_text(json.dumps({"limits": dataclasses.asdict(Limits(max_rel_frob=None,
                                                                                         max_edit_frac=None))}))
    (root / "run.json").write_text(json.dumps({
        "session_id": SESSION, "status": "complete", "session_result": "session_result.json",
        "instance": {"instance_id": instance_id, "params_sha256": instance.params_sha256},
        "final": {"is_base_noop": False, "dir": "final", "parent": "instance", "patch_id": "toy.py",
                  "query_index": 0, "n_changed": int(child[name].size), "child_sha256": params_sha256(child)}}))
    info = runs_dir() / "devreadouts" / "learn" / SESSION / "info.json"
    info.parent.mkdir(parents=True, exist_ok=True)
    info.write_text(json.dumps({"instance_id": instance_id, "final": {
        "child_sha256": params_sha256(child),
        "readouts_policyskill": devreadout.json_safe(E.e14_readout(instance, child))}}))
    return PolicyToy(instance_id, child, str(run_path.relative_to(runs_dir())), ref_record, ref_child)


def _entries(toy: PolicyToy) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finals = [{"name": "policy_final", "readout": "policy_success",
               "source": {"kind": "session", "session_id": SESSION}, "instance_id": toy.instance_id,
               "child_sha256": params_sha256(toy.session_child)}]
    references = [{"name": "policy_ref", "readout": "policy_success",
                   "source": {"kind": "reference", "run": toy.ref_run, "pick": "kept", "s_old_floor": -3,
                              "dev_record": toy.ref_record},
                   "instance_id": toy.instance_id, "child_sha256": toy.ref_child_sha256,
                   "pairs_with": "policy_final"}]
    return finals, references


def _write_list(path: Path, toy: PolicyToy, status: str = "draft", approval: str | None = None) -> Path:
    finals, references = _entries(toy)
    path.write_text(json.dumps({"schema": SF.LIST_SCHEMA, "status": status, "approval": approval, "note": "toy",
                                "finals": finals, "references": references}))
    return path


def _git_state(path: Path) -> Path:
    path.write_text(json.dumps({"commit": COMMIT, "dirty": False}))
    return path


def test_dev_equals_the_e14_readouts_of_record(toy: PolicyToy, tmp_path: Path) -> None:
    records = SF.run_dev(_write_list(tmp_path / "list.json", toy), tmp_path / "dev.jsonl",
                         code_state_path=_git_state(tmp_path / "git_state.json"))
    assert [r["name"] for r in records] == ["policy_final", "policy_ref"]
    for record in records:
        assert record["reference"]["match"], record["reference"]["differences"]
        ro = record["readouts_policyskill"]
        assert record["gain"] == ro["gain"] and record["clean"] == ro["clean"]
        assert ro["new"]["n"] == ro["old"]["n"] == 6
        assert record["rows"]["new"]["split"] == "V_select_new"
        assert record["environment"]["minigrid"] == D.environment_versions()["minigrid"]
        assert set(record["code"]["files"]) == set(SF.CODE_MODULES) | set(SE.PolicySuccess.code_modules)
        index = record["sealed_index"]["test"]
        assert index["path"] == str(sealed_dir() / D.DATASET / "test.npz") and index["n"] == 6


def test_the_environment_must_be_the_instances(toy: PolicyToy, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(D, "environment_versions", lambda: {"gymnasium": "0.0.0", "minigrid": "0.0.0"})
    with pytest.raises(SF.SealedFinalsError, match="installed environment"):
        SE.FAMILY.load_instance(toy.instance_id)


def _approved(toy: PolicyToy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    committed = _write_list(repo / "finals.json", toy, status="approved", approval="toy approval")
    for args in (["init", "-q"], ["add", "finals.json"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "list"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    state = {"commit": COMMIT, "dirty": False, "source": f"git {repo}", "repo": str(repo), "files": {}}
    monkeypatch.setattr(SF, "code_state", lambda code_state_path=None, modules=SF.CODE_MODULES: dict(state))
    return committed


def _sealed_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copy = tmp_path / "sealed_copy"
    shutil.copytree(sealed_dir(), copy)
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(copy))
    return copy


def test_sealed_read_of_the_e14_seeds(toy: PolicyToy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    copy = _sealed_copy(tmp_path, monkeypatch)
    with np.load(copy / D.DATASET / "test.npz", allow_pickle=False) as archive:
        seeds = archive["x"].copy()
    results, count = SF.run_sealed(_approved(toy, tmp_path, monkeypatch))
    assert count == 2
    instance = SE.FAMILY.load_instance(toy.instance_id)
    final = [json.loads(line) for line in results.read_text().splitlines()][0]
    ro = final["readouts_policyskill"]
    child_fn = E.logits_fn(instance.forward, toy.session_child)
    for part, rows in (("old", seeds[:3]), ("new", seeds[3:])):
        assert ro[part]["n"] == 3 and final["rows"][part]["n"] == 3
        assert ro[part]["success_sha256_child"] == E.success_sha256(D.rollouts(child_fn, part, rows).success)
    assert ro["joint"] == ro["gain"] + ro["clean"]
    events = read_registry(copy / "registry.jsonl")
    assert [e["event"] for e in events] == ["evaluation_started", *["entry_started", "entry_finished"] * 2,
                                            "evaluation_finished"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda a: a.update(y=a["y"][::-1].copy()), "old then new"),
        (lambda a: a.update(x=np.full_like(a["x"], D.SEALED_FLOOR)), "distinct"),
        (lambda a: a.update(extra=np.zeros(len(a["x"]), dtype=np.int64)), "keys"),
    ],
)
def test_e14_sealed_record_refusals(toy: PolicyToy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: Any,
                                    message: str) -> None:
    copy = _sealed_copy(tmp_path, monkeypatch)
    path = copy / D.DATASET / "test.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    change(arrays)
    np.savez(path, **arrays)
    dataset = D.store_dir() / "dataset.json"
    original = dataset.read_text()
    meta = json.loads(original)
    meta["sealed_record"]["sha256"] = params_sha256(arrays)
    dataset.write_text(json.dumps(meta))
    try:
        instance = SE.FAMILY.load_instance(toy.instance_id)
        with pytest.raises(SF.SealedFinalsError, match=message):
            SE.FAMILY.sealed_rows(instance, SE.FAMILY.sealed_index(instance), {})
    finally:
        dataset.write_text(original)


def test_e14_reference_needs_the_e14_program_limits(toy: PolicyToy) -> None:
    entry = SF.Final("policy_ref", "policy_success", _entries(toy)[1][0]["source"], toy.instance_id,
                     toy.ref_child_sha256, SF.BLOCK_REFERENCES, "policy_final")
    instance = SE.FAMILY.load_instance(toy.instance_id)
    assert params_sha256(SF.recover_child(entry, instance)[0]) == toy.ref_child_sha256
    path = runs_dir() / toy.ref_run
    original = path.read_text()
    run = json.loads(original)
    run["limits"] = dict(run["limits"], max_changed_frac=0.9)
    path.write_text(json.dumps(run))
    try:
        with pytest.raises(SF.SealedFinalsError, match="not the E14 program limits"):
            SF.recover_child(entry, instance)
    finally:
        path.write_text(original)
