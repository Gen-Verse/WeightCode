"""Tests for the scan_em family of the sealed read (weightcode.evalkit.sealed_scan, PREREG E13) on the SCAN toy store.

The store is ``scan_toy.store_with_students`` (written from the grammar, tiny students, gate 0) with its addprim_jump
scanskill-v1 instance and its sealed test.npz in the test's tmp sealed directory. A final is a hand-written session
(run.json, session.json, final/patch.json, info.json with ``scanskill.scan_readout``); a reference is a real
rule-generated E13 reference run (``scanrefs.run_reference``, its fit set cut to 300 commands) with the dev record
``devreadout.readout_retention`` writes for it at S_old >= -5. These tests need the E13 code (branch e13-scan) and
are skipped in a tree without it.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("weightcode.evalkit.scanskill")
pytest.importorskip("torch")

import scan_toy  # noqa: E402

from weightcode.baselines.fit import scanrefs as R  # noqa: E402
from weightcode.baselines.fit.runner import baseline_run_dir  # noqa: E402
from weightcode.baselines.instance_io import load_instance, write_delta_patch  # noqa: E402
from weightcode.common.identity import params_sha256  # noqa: E402
from weightcode.common.paths import runs_dir, sealed_dir  # noqa: E402
from weightcode.data import scan as S  # noqa: E402
from weightcode.data.store import dataset_json_path  # noqa: E402
from weightcode.evalkit import devreadout as D  # noqa: E402
from weightcode.evalkit import scanskill as K  # noqa: E402
from weightcode.evalkit import sealed_finals as SF  # noqa: E402
from weightcode.evalkit import sealed_scan as SS  # noqa: E402
from weightcode.evalkit.sealed import read_registry  # noqa: E402
from weightcode.patch import Limits  # noqa: E402

CONDITION = "addprim_jump"
SESSION = "s5ca0000000000001"
METHOD = "rulegen_mixed_sparse19_sal"
COMMIT = "0" * 40


@dataclasses.dataclass
class ScanToy:
    instance_id: str
    session_child: dict[str, np.ndarray]
    ref_run: str
    ref_record: str
    ref_child_sha256: str


def _session(instance: Any, child: dict[str, np.ndarray]) -> None:
    root = runs_dir() / "sessions" / SESSION
    write_delta_patch(root / "final", instance.params_sha256, instance.params, child, "toy", "toy")
    limits = dataclasses.asdict(Limits(max_rel_frob=None, max_edit_frac=None))
    (root / "session.json").write_text(json.dumps({"limits": limits}))
    run = {"session_id": SESSION, "status": "complete", "session_result": "session_result.json",
           "instance": {"instance_id": instance.instance_id, "params_sha256": instance.params_sha256},
           "final": {"is_base_noop": False, "dir": "final", "parent": "instance", "patch_id": "jump.py",
                     "query_index": 0, "n_changed": 128, "child_sha256": params_sha256(child)}}
    (root / "run.json").write_text(json.dumps(run))
    info = {"instance_id": instance.instance_id,
            "final": {"child_sha256": params_sha256(child), "readouts_scan": K.scan_readout(instance, child)}}
    reference = runs_dir() / "devreadouts" / "learn" / SESSION / "info.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps(D.json_safe(info)))


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ScanToy]:
    patcher = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("sealed_scan_store")
    try:
        scan_toy.store_with_students(tmp, patcher)
        (row,) = K.build_scan_instances(CONDITION)
        instance = load_instance(K.scan_instance_dir(row["instance_id"]))
        child = {key: value.copy() for key, value in instance.params.items()}
        child[R.EMBEDDING][S.TOKEN_ID["jump"]] = child[R.EMBEDDING][S.TOKEN_ID["walk"]]
        _session(instance, child)
        for name, value in (("STEP_CHECKPOINTS", (1, 2, 3, 4)), ("REG_STEPS", (1, 2)), ("TRAJ_STEPS", 3),
                            ("N_RULEGEN", 200), ("N_RULEGEN_OLD", 100)):
            patcher.setattr(R, name, value)
        run_dir = baseline_run_dir(instance.instance_id, METHOD, 0, R.run_label(METHOD))
        R.run_reference(instance, METHOD, fit_set=R.rulegen_fit_set(CONDITION, "mixed"), run_dir=run_dir)
        record = D.readout_retention(run_dir / "run.json", s_old_floor=-5)
        ref_record = "devreadouts/e13_refs_toy_floor5.jsonl"
        (runs_dir() / ref_record).write_text(json.dumps(D.json_safe(record)) + "\n")
        pick = SF.kept_pick(SF.Final("x", SF.READOUT_SCAN_EM, {"s_old_floor": -5}, instance.instance_id, "0" * 64),
                            json.loads((run_dir / "run.json").read_text()))["final"]
        ref_child = instance.params_sha256 if pick["is_base_noop"] else pick["child_sha256"]
        yield ScanToy(instance.instance_id, child, str((run_dir / "run.json").relative_to(runs_dir())), ref_record,
                      ref_child)
    finally:
        patcher.undo()


def _entries(toy: ScanToy) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finals = [{"name": "scan_final", "readout": "scan_em", "source": {"kind": "session", "session_id": SESSION},
               "instance_id": toy.instance_id, "child_sha256": params_sha256(toy.session_child)}]
    references = [{"name": "scan_ref", "readout": "scan_em",
                   "source": {"kind": "reference", "run": toy.ref_run, "pick": "kept", "s_old_floor": -5,
                              "dev_record": toy.ref_record},
                   "instance_id": toy.instance_id, "child_sha256": toy.ref_child_sha256, "pairs_with": "scan_final"}]
    return finals, references


def _write_list(path: Path, toy: ScanToy, status: str = "draft", approval: str | None = None) -> Path:
    finals, references = _entries(toy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": SF.LIST_SCHEMA, "status": status, "approval": approval, "note": "toy",
                                "finals": finals, "references": references}))
    return path


def _git_state(path: Path) -> Path:
    path.write_text(json.dumps({"commit": COMMIT, "dirty": False}))
    return path


def test_dev_equals_the_scan_readouts_of_record(toy: ScanToy, tmp_path: Path) -> None:
    records = SF.run_dev(_write_list(tmp_path / "list.json", toy), tmp_path / "dev.jsonl",
                         code_state_path=_git_state(tmp_path / "git_state.json"))
    assert [r["name"] for r in records] == ["scan_final", "scan_ref"]
    instance = SS.FAMILY.load_instance(toy.instance_id)
    final, ref = records
    assert final["readouts_scan"] == D.json_safe(K.scan_readout(instance, toy.session_child))
    for record in records:
        assert record["reference"]["match"], record["reference"]["differences"]
        assert record["gain"] == record["readouts_scan"]["gain_points"]
        assert record["clean"] == record["readouts_scan"]["clean_points"]
        assert record["rows"]["new"]["split"] == "V_select" and record["rows"]["old"]["split"] == "old_select"
        assert set(record["code"]["files"]) == set(SF.CODE_MODULES) | set(SS.ScanEM.code_modules)
        index = record["sealed_index"]["test"]
        assert index["path"] == str(sealed_dir() / "scan" / "conditions" / CONDITION / "test.npz")
        assert index["n_new"] + index["n_old"] == index["n"]
    assert "fit_coverage" in ref["readouts_scan"] and "fit_coverage" not in final["readouts_scan"]
    assert ref["child"]["kind"] == "reference" and ref["child"]["selection"]["s_old_floor"] == -5


def test_dev_reports_a_changed_scan_record(toy: ScanToy, tmp_path: Path) -> None:
    path = runs_dir() / "devreadouts" / "learn" / SESSION / "info.json"
    original = path.read_text()
    info = json.loads(original)
    info["final"]["readouts_scan"]["new_by_construction"]["twice_thrice"]["fixed"] += 1
    path.write_text(json.dumps(info))
    try:
        records = SF.run_dev(_write_list(tmp_path / "list.json", toy), tmp_path / "dev.jsonl",
                             code_state_path=_git_state(tmp_path / "git_state.json"), only=["scan_final"])
    finally:
        path.write_text(original)
    assert not records[0]["reference"]["match"]
    assert any("readouts_scan.new_by_construction.twice_thrice.fixed" in d
               for d in records[0]["reference"]["differences"])


def _approved(toy: ScanToy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    draft = _write_list(tmp_path / "draft.json", toy, status="approved", approval="toy approval")
    repo = tmp_path / "repo"
    repo.mkdir()
    committed = repo / "finals.json"
    committed.write_text(draft.read_text())
    for args in (["init", "-q"], ["add", "finals.json"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "list"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    state = {"commit": COMMIT, "dirty": False, "source": f"git {repo}", "repo": str(repo), "files": {}}
    monkeypatch.setattr(SF, "code_state", lambda code_state_path=None, modules=SF.CODE_MODULES: dict(state))
    return committed


def _sealed_parts() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(sealed_dir() / "scan" / "conditions" / CONDITION / "test.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    new = arrays["part"] == S.PART_NEW
    return ({k: v[new] for k, v in arrays.items()}, {k: v[~new] for k, v in arrays.items()})


def test_sealed_read_of_the_scan_test(toy: ScanToy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(_copy_sealed(tmp_path)))
    results, count = SF.run_sealed(_approved(toy, tmp_path, monkeypatch))
    assert count == 2
    records = [json.loads(line) for line in results.read_text().splitlines()]
    instance = SS.FAMILY.load_instance(toy.instance_id)
    new, old = _sealed_parts()
    entry = _reference_entry(toy)
    fit_ids = SS.FAMILY.read_context(entry, json.loads((runs_dir() / toy.ref_run).read_text()))["fit_ids"]
    assert fit_ids and len(fit_ids) == 300
    ref_child = SF.recover_child(entry, instance)[0]
    for record, child, ids in zip(records, (toy.session_child, ref_child), (None, fit_ids)):
        expected = K.scan_readout_arrays(new, old, instance.forward, instance.params, child, fit_ids=ids)
        assert record["readouts_scan"] == D.json_safe(expected)
        assert record["rows"]["new"]["n"] == len(new["y"]) and record["rows"]["old"]["n"] == len(old["y"])
        assert record["rows"]["new"]["verified_sha256"] == record["sealed_index"]["test"]["sha256"]
    events = read_registry(sealed_dir() / "registry.jsonl")
    assert [e["event"] for e in events] == ["evaluation_started", *["entry_started", "entry_finished"] * 2,
                                            "evaluation_finished"]
    assert list(events[-1]["test_sha256"]) == [records[0]["sealed_index"]["test"]["location"]]


def _copy_sealed(tmp_path: Path) -> Path:
    """A private copy of the toy's sealed directory, so a test can spend or change it."""
    import shutil

    copy = tmp_path / "sealed_copy"
    shutil.copytree(sealed_dir(), copy)
    return copy


def test_a_sealed_file_out_of_order_is_refused_and_spent(toy: ScanToy, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    copy = _copy_sealed(tmp_path)
    monkeypatch.setenv("WEIGHTCODE_SEALED", str(copy))
    path = copy / "scan" / "conditions" / CONDITION / "test.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key][::-1].copy() for key in archive.files}
    np.savez(path, **arrays)
    original = dataset_json_path(S.DATASET, CONDITION).read_text()
    meta = json.loads(original)
    meta["test"]["sha256"] = params_sha256(arrays)
    dataset_json_path(S.DATASET, CONDITION).write_text(json.dumps(meta))
    try:
        with pytest.raises(SF.SealedEntriesFailed, match="parts are not"):
            SF.run_sealed(_approved(toy, tmp_path, monkeypatch))
    finally:
        dataset_json_path(S.DATASET, CONDITION).write_text(original)
    events = read_registry(copy / "registry.jsonl")
    assert [e["event"] for e in events] == ["evaluation_started", *["entry_started", "entry_failed"] * 2,
                                            "evaluation_finished"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: m["test"].pop("part_counts"), "part_counts"),
        (lambda m: m["test"].update(n=1), "summing to n"),
        (lambda m: m["test"].update(location="/abs/test.npz"), "not under"),
        (lambda m: m["test"].pop("sha256"), "no recorded sha256"),
        (lambda m: m["splits"]["V_select"].update(sha256="0" * 64), "not the instance's"),
    ],
)
def test_scan_index_refusals(toy: ScanToy, change: Any, message: str) -> None:
    path = dataset_json_path(S.DATASET, CONDITION)
    original = path.read_text()
    meta = json.loads(original)
    change(meta)
    path.write_text(json.dumps(meta))
    try:
        with pytest.raises(SF.SealedFinalsError, match=message):
            SS.FAMILY.sealed_index(load_instance(K.scan_instance_dir(toy.instance_id)))
    finally:
        path.write_text(original)


def _reference_entry(toy: ScanToy) -> SF.Final:
    source = _entries(toy)[1][0]["source"]
    return SF.Final("scan_ref", "scan_em", source, toy.instance_id, toy.ref_child_sha256, SF.BLOCK_REFERENCES,
                    "scan_final")


def test_scan_reference_refusals(toy: ScanToy) -> None:
    entry = _reference_entry(toy)
    instance = SS.FAMILY.load_instance(toy.instance_id)
    assert params_sha256(SF.recover_child(entry, instance)[0]) == toy.ref_child_sha256
    run_path = runs_dir() / toy.ref_run
    original = run_path.read_text()
    run = json.loads(original)
    run["limits"] = dict(run["limits"], max_changed_frac=0.5)
    run_path.write_text(json.dumps(run))
    try:
        with pytest.raises(SF.SealedFinalsError, match="E13 program limits"):
            SF.recover_child(entry, instance)
    finally:
        run_path.write_text(original)
