"""Instance refusals of the editor harness: every one happens before the runner writes anything."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_evalkit_instances_store import build_v0
from test_harness_fixtures import HarnessStore, harness_store, needs_sandbox  # noqa: F401

import weightcode.evalkit.instances as evalkit_instances
import weightcode.harness.instance as harness_instance
from weightcode.evalkit.instances import supersede_instances
from weightcode.harness.instance import InstanceRefused, read_instance_record
from weightcode.harness.runner import run_session
from weightcode.patch import Limits, task_limits


def _nothing_written(store: HarnessStore) -> bool:
    return not (store.store.runs / "sessions").exists() and not (store.store.runs / "dry_runs").exists()


def _dry_run(instance_id: str, arm: str = "M2") -> None:
    run_session(instance_id, arm, 0, effort="low", wall_timeout_s=60.0, dry_run=True)


def test_a_superseded_instance_is_refused_with_its_reason(harness_store: HarnessStore) -> None:  # noqa: F811
    instance_id = harness_store.dev(100)
    supersede_instances([instance_id], "test: retired for the harness")
    with pytest.raises(InstanceRefused) as caught:
        _dry_run(instance_id)
    message = str(caught.value)
    assert instance_id in message and "superseded" in message and "test: retired for the harness" in message
    assert _nothing_written(harness_store)


def test_an_instance_built_under_other_limits_is_refused_naming_the_field(harness_store: HarnessStore,  # noqa: F811
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    def newer_limits(**kwargs: object) -> Limits:
        return Limits(**{"max_edit_frac": 0.1, **kwargs})  # type: ignore[arg-type]

    monkeypatch.setattr(evalkit_instances, "Limits", newer_limits)
    # Once evalkit writes task_limits() (the shared form), the patch reaches it through that name.
    monkeypatch.setattr(evalkit_instances, "task_limits", lambda: task_limits(newer_limits()), raising=False)
    monkeypatch.setattr(harness_instance, "task_limits", lambda: task_limits(newer_limits()))
    instance_id = harness_store.dev(100)
    with pytest.raises(InstanceRefused) as caught:
        _dry_run(instance_id)
    message = str(caught.value)
    assert instance_id in message and "limits.max_edit_frac: task.json has 0.05; Limits() has 0.1" in message
    assert _nothing_written(harness_store)


def test_an_evidence_v0_instance_is_refused(harness_store: HarnessStore) -> None:  # noqa: F811
    v0 = build_v0(harness_store.store, roles=("dev",))
    with pytest.raises(InstanceRefused, match="v0|evidence-v0"):
        _dry_run(v0[0].instance_id)
    assert _nothing_written(harness_store)


def test_a_main_instance_is_refused_unless_allowed(harness_store: HarnessStore) -> None:  # noqa: F811
    with pytest.raises(InstanceRefused, match="main instance"):
        _dry_run(harness_store.main(0))
    assert _nothing_written(harness_store)
    assert read_instance_record(harness_store.main(0), allow_main=True).split_role == "main"


def test_unknown_and_malformed_ids_are_refused(harness_store: HarnessStore) -> None:  # noqa: F811
    for bad in ("i00000000", "../instances", "ifee5b7da/x"):
        with pytest.raises(InstanceRefused):
            _dry_run(bad)
    assert _nothing_written(harness_store)


def test_decoy_is_refused_without_the_decoy_models_own_evidence(harness_store: HarnessStore) -> None:  # noqa: F811
    supersede_instances([harness_store.dev(101)], "test: decoy sibling retired")
    with pytest.raises(InstanceRefused) as caught:
        _dry_run(harness_store.dev(100), "DECOY")
    assert "no DECOY view" in str(caught.value) and "exactly one active instance" in str(caught.value)
    assert _nothing_written(harness_store)


@needs_sandbox
def test_the_other_arms_still_run_when_only_the_decoy_evidence_is_missing(harness_store: HarnessStore  # noqa: F811
                                                                         ) -> None:
    supersede_instances([harness_store.dev(101)], "test: decoy sibling retired")
    result = run_session(harness_store.dev(100), "M2", 0, effort="low", wall_timeout_s=60.0, dry_run=True)
    assert result.status == "dry_run"
    assert Path(result.root).parent.name == "dry_runs"
