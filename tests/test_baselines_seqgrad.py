"""PREREG E5 gradient references: the seqskill branch of execute_run, the three references, limits and dev readouts.

Nothing here reads the real stores: WEIGHTCODE_PRIVATE, WEIGHTCODE_SEALED and WEIGHTCODE_RUNS point to tmp dirs. The student is a
tiny tfm trained for a few steps; it answers nothing, which is all the rule needs from a base.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from weightcode.baselines.fit import seqgrad as G
from weightcode.baselines.fit.runner import baseline_run_dir, load_seqskill_evidence
from weightcode.data import seqops as S
from weightcode.data import seqops_build as B
from weightcode.data.store import forward_path, load_split
from weightcode.evalkit import devreadout as D
from weightcode.evalkit import seqskill as Q
from weightcode.models import train_seq as TS
from weightcode.models.export import write_index
from weightcode.models.forward_gen import render_seq_forward, write_forward
from weightcode.models.seq_export import export_seq_model
from weightcode.patch import Limits, task_limits

CONDITION = "new0_substitution"
TINY = TS.SeqTrainConfig(d_model=16, n_layers=2, d_mlp=64, steps=3, warmup_steps=1, log_every=3)
"""Two blocks, so the token embedding and the head stay under the 20% changed-entry cap (800 of 7840 entries)."""


@pytest.fixture(scope="module")
def instance(tmp_path_factory: pytest.TempPathFactory):
    """One store and instance for the module (each test writes its runs under its own run seed)."""
    root = tmp_path_factory.mktemp("seqgrad")
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("private", "sealed", "runs"):
            (root / name).mkdir()
            monkeypatch.setenv(f"WEIGHTCODE_{name.upper()}", str(root / name))
        sha = write_forward(forward_path(S.DATASET), render_seq_forward(**B.forward_constants()))
        base_manifest = B.write_built(B.build_base(), {"forward_path": "forward.py", "forward_sha256": sha})
        torch.manual_seed(0)
        result = TS.train_seq(TINY, 100, S.held_out_keys(S.build_base_splits()), torch.device("cpu"))
        entry = export_seq_model(manifest=base_manifest, role="base", seed=100, config=TINY, result=result,
                                 size_check={"em": 0.0, "passed": False}, determinism={},
                                 run_info={"versions": {}, "device": "cpu"}, vselect=load_split(S.DATASET, "V_select"))
        write_index(S.DATASET, [entry])
        B.write_built(B.build_condition(CONDITION), B.condition_extra(base_manifest))
        (row,) = Q.build_seqskill_instances(CONDITION)
        yield Q.load_seqskill_instance(row["instance_id"])


def test_unconstrained_limits_make_room_for_one_op_per_tensor(tmp_path: Path) -> None:
    from weightcode.baselines.fit.edits import apply_patch_result
    from weightcode.baselines.instance_io import write_delta_patch
    from weightcode.common.identity import params_sha256
    from weightcode.patch import PatchError

    limits = Q.seqskill_program_limits(task_limits(Limits()))
    assert limits.without_edit_limits(30) == limits.without_size_limits()
    assert limits.without_edit_limits(70).max_ops == 70 and limits.without_edit_limits(70).max_changed_frac is None
    with pytest.raises(ValueError):
        limits.without_edit_limits(0)
    base = {f"t{k:02d}.weight": np.full((2, 2), 1.0, dtype=np.float32) for k in range(70)}
    child = {key: value + np.float32(0.25) for key, value in base.items()}
    write_delta_patch(tmp_path / "p", params_sha256(base), base, child, "h", "e")
    applied = apply_patch_result(base, tmp_path / "p" / "patch.json", params_sha256(base), enforce_limits=False,
                                 limits=limits)
    assert applied.child_sha256 == params_sha256(child)
    with pytest.raises(PatchError, match="at most 64"):
        apply_patch_result(base, tmp_path / "p" / "patch.json", params_sha256(base), enforce_limits=True,
                           limits=limits)
    run = {"flags": {"edit_limits_enforced": False}}
    rebuilt, label = D.run_limits(run, tmp_path, limits, "E5 program limits", n_tensors=70)
    assert rebuilt.max_ops == 70 and label.endswith("max_ops 70")
    assert D.run_limits(run, tmp_path, limits, "E5", n_tensors=30)[1] == "E5 without size limits"


def test_program_limits_equal_the_learn_limits() -> None:
    import inspect

    from weightcode.harness.arms import PROGRAM_MAX_CHANGED_FRAC, applied_limits

    stated = task_limits(Limits())
    assert Q.seqskill_program_limits(stated) == applied_limits(stated, "LEARN")
    assert Q.SEQSKILL_MAX_CHANGED_FRAC == PROGRAM_MAX_CHANGED_FRAC
    limits = Q.seqskill_program_limits(stated)
    assert limits.max_rel_frob == 0.5 and limits.max_edit_frac is None and limits.rel_frob_exempt == ()
    assert Q.seqskill_program_limits(stated, 70).max_ops == 70 and Q.seqskill_program_limits(stated, 30).max_ops == 64
    with pytest.raises(Q.SeqskillError):
        Q.seqskill_program_limits(stated, 0)
    if "n_tensors" not in inspect.signature(applied_limits).parameters:
        pytest.skip("this tree's harness predates applied_limits(..., n_tensors) (e5-seqskill 47d9255)")
    for n in (30, 70):
        assert Q.seqskill_program_limits(stated, n) == applied_limits(stated, "LEARN", n_tensors=n)


def test_evidence_and_rebuild_parity(instance) -> None:
    evidence = load_seqskill_evidence(instance)
    assert evidence.x.shape == evidence.y.shape == (128, S.SEQ_LEN) and evidence.n_rand == evidence.n_replay == 64
    assert evidence.describe()["n_new"] == 64 and "new-operator rows" in evidence.composition()
    diff, bound = G.seq_parity(G.rebuild(instance.params), instance.params, instance.forward, evidence.x)
    assert 0.0 <= diff <= bound and bound >= G.PARITY_RTOL


def test_new_row_training_changes_that_row_only(instance) -> None:
    evidence = load_seqskill_evidence(instance)
    new_id = int(instance.private["new_operator_id"])
    trainable, masks = G.trainable_and_masks("seq_grad_new_row", instance.params, new_id)
    x, y = torch.from_numpy(evidence.x), torch.from_numpy(evidence.y)
    ((steps, child, _),) = G.adam_checkpoints(instance.params, x, y, trainable=trainable, lr=1e-2, checkpoints=(10,),
                                              seed=0, masks=masks)
    changed = {key for key in child if not np.array_equal(child[key], instance.params[key])}
    assert steps == 10 and changed == {G.EMBEDDING}
    rows = np.flatnonzero(np.any(child[G.EMBEDDING] != instance.params[G.EMBEDDING], axis=1))
    assert rows.tolist() == [new_id]


def test_the_three_references_run_and_read_out(instance) -> None:
    records = {}
    for method in G.SEQ_GRAD_METHODS:
        run_dir = baseline_run_dir(instance.instance_id, method, 0, G.SEQSKILL_REF_LABEL)
        records[method] = G.run_seq_grad(instance, method, run_dir)
    for method, record in records.items():
        assert record["status"] == "complete" and record["queries_used"] == 8 and record["n_invalid"] == 0
        assert record["limits"] == task_limits(Q.seqskill_program_limits(instance.task["limits"], len(instance.params)))
        assert record["executor_limits"]["max_ops"] == max(64, len(instance.params))
        assert record["joint"]["evidence"]["n_new"] == 64 and record["seqskill"]["new_operator"] == "OP_NEW0"
        assert all("score_new" in entry for entry in record["oracle_log"])
    assert records["seq_grad_emb_head"]["flags"]["respects_edit_limits"] is True
    assert records["seq_grad_new_row"]["flags"]["respects_edit_limits"] is True
    assert records["seq_grad_full"]["flags"]["respects_edit_limits"] is False

    for method in G.SEQ_GRAD_METHODS:
        path = baseline_run_dir(instance.instance_id, method, 0, G.SEQSKILL_REF_LABEL) / "run.json"
        primary = D.readout_run(path)
        assert primary["rule"] == Q.SEQSKILL_RULE and primary["condition"] == CONDITION
        assert [primary["readouts_seqskill"][p]["n"] for p in ("i", "ii", "iii", "iv")] == [2000, 2000, 2000, 4000]
        kept = D.readout_retention(path, s_old_floor=-20)
        assert kept["selection"]["s_old_floor"] == -20 and "readouts_seqskill" in kept
        if kept["is_base_noop"]:
            assert kept["readouts_seqskill"]["all"]["delta"] == 0.0


def test_devreadout_cli_writes_seqskill_records(instance, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    run_dir = baseline_run_dir(instance.instance_id, "seq_grad_emb_head", 2, G.SEQSKILL_REF_LABEL)
    G.run_seq_grad(instance, "seq_grad_emb_head", run_dir, run_seed=2)
    for extra, name in (([], "final.jsonl"), (["--retention-floor", "-20"], "kept.jsonl")):
        assert D.main([str(run_dir / "run.json"), "--out", str(tmp_path / name), "--label", "test", *extra]) == 0
        (record,) = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
        assert record["label"] == "test" and record["method"] == "seq_grad_emb_head"
        assert "iv_by_kind" in record["readouts_seqskill"]
    assert "V_select points i" in capsys.readouterr().out


def test_a_run_recording_other_limits_is_refused_at_readout(instance) -> None:
    run_dir = baseline_run_dir(instance.instance_id, "seq_grad_new_row", 1, G.SEQSKILL_REF_LABEL)
    G.run_seq_grad(instance, "seq_grad_new_row", run_dir, run_seed=1)
    run = json.loads((run_dir / "run.json").read_text())
    run["limits"] = task_limits(Limits())
    (run_dir / "run.json").write_text(json.dumps(run))
    with pytest.raises(D.DevReadoutError, match="E5 program limits"):
        D.readout_run(run_dir / "run.json")


def test_supersede_and_retire(instance) -> None:
    """Last in the module: it supersedes the shared instance and retires its student."""
    from weightcode.models.export import read_index
    from weightcode.models.seq_export import RETIRED_LOG, retire_seq_model

    root = Q.seqskill_instance_dir(instance.instance_id).parents[1]
    with pytest.raises(Q.SeqskillError, match="reason"):
        Q.supersede_seqskill_instance(instance.instance_id, " ")
    moved = Q.supersede_seqskill_instance(instance.instance_id, "test")
    assert moved["reason"] == "test" and Q.active_seqskill_ids() == []
    with pytest.raises(Q.StaleSeqskillInstance, match="superseded"):
        Q.check_seqskill_current(Q.seqskill_instance_dir(instance.instance_id))
    with pytest.raises(Q.SeqskillError, match="not an active"):
        Q.supersede_seqskill_instance(instance.instance_id, "again")
    (row,) = read_index(S.DATASET)
    record = retire_seq_model(row["model_id"], "test")
    assert read_index(S.DATASET) == [] and record["row"] == row
    assert (root / S.DATASET / "models" / RETIRED_LOG).read_text().count("\n") == 1
    with pytest.raises(ValueError, match="not in the seqops model index"):
        retire_seq_model(row["model_id"], "again")
