"""PREREG T04's edge-extension amendment (main 9f0baa9): the pair rule of scripts/t04/edge_extension.py, and the
extension grids the runners build (run label t04g3) equal the rule's scaling of each source grid."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from weightcode.baselines.fit import bnrefs as BN
from weightcode.baselines.fit import cli as C
from weightcode.baselines.fit import ftcompare as FT
from weightcode.baselines.fit import grad as G
from weightcode.common.paths import relocate_run_path

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("t04_edge_extension", ROOT / "scripts/t04/edge_extension.py")
assert _spec is not None and _spec.loader is not None
EE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EE)


def rule_scaled(configs: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """The amendment's extension of a source grid, as edge_extension.py writes it."""
    factor = EE.EDGES[key][2]
    return [dict(c) | {key: EE.scaled(c[key], factor, key)} for c in configs]


def test_grad_and_interp_extension_grids_are_the_rule_applied_to_the_second_grid() -> None:
    assert G.GRID_T04_EDGE.learning_rates == (1e-6, 1e-5)
    assert G.GRID_T04_EDGE.interp_alphas == tuple(k / 512 for k in range(1, 9))
    source = [{"lr": lr, "steps": s} for lr, s in G.candidate_grid(G.GRID_E11_SECOND)]
    assert rule_scaled(source, "lr") == [{"lr": lr, "steps": s} for lr, s in G.candidate_grid(G.GRID_T04_EDGE)]
    alphas = [{"alpha": a} for a in G.GRID_E11_SECOND.interp_alphas]
    assert rule_scaled(alphas, "alpha") == [{"alpha": a} for a in G.GRID_T04_EDGE.interp_alphas]
    assert "alpha in {1/512 .. 8/512}" in G.grid_spec("interp", G.GRID_T04_EDGE).label
    for method in G.GRID2_METHODS:
        config = G.grad_config(method, 0, joint=True, grid=G.GRID_T04_EDGE)
        assert config["grid"]["name"] == "t04_edge" and "9f0baa9" in config["grid"]["note"]


def test_bn_adapt_extension_grid_is_the_rule_applied_to_its_priors() -> None:
    source = [{"N": n} for n in BN.BN_ADAPT_PRIORS]
    assert rule_scaled(source, "N") == [{"N": n} for n in BN.BN_ADAPT_PRIORS_T04_EDGE]
    assert BN.BN_ADAPT_PRIORS_T04_EDGE == (0, 256, 512, 1024, 2048, 4096, 8192, 16384)
    config = BN.bn_config("bn_adapt", 0, BN.BN_ADAPT_PRIORS_T04_EDGE)
    assert config["candidate_order"] == [{"N": n} for n in BN.BN_ADAPT_PRIORS_T04_EDGE] and "grid" in config
    assert "grid" not in BN.bn_config("bn_adapt", 0)


@pytest.mark.parametrize("method", FT.F1_DATASET_METHODS[FT.F1_T04_DATASET])
def test_f1_extension_grids_are_the_rule_applied_to_each_f1_grid(method: str) -> None:
    edge = FT.f1_configs(method, edge=True)
    assert edge == rule_scaled(FT.f1_configs(method), "lr")
    assert len(edge) == 8
    assert min(c["lr"] for c in edge) == pytest.approx(min(c["lr"] for c in FT.f1_configs(method)) / 10, rel=1e-12)
    config = FT.f1_config(method, 0, FT.F1_T04_DATASET, edge=True)
    assert config["candidate_order"] == edge and config["grid"]["name"] == "t04_edge"
    assert config["reused"] == {"full": ["grad_full", "t04g3"], "last layer": ["grad_last", "t04g3"]}


def test_edge_runs_are_refused_outside_t04_and_for_methods_without_an_extension(tmp_path: Path) -> None:
    t04 = SimpleNamespace(private={"dataset": C.T04_DATASET})
    kwargs = dict(run_seed=0, allow_main=False, joint=True, e10=True)
    with pytest.raises(ValueError, match="edge extension is PREREG T04's"):
        C.run_method(t04, "grad_last", tmp_path, edge=True, e12=True, **kwargs)
    with pytest.raises(ValueError, match="edge extension is PREREG T04's"):
        C.run_method(t04, "grad_last", tmp_path, edge=True, t04=True, grid2=True, **kwargs)
    for method in ("tent", "prior_shift", "ridge_last", "tie_inputs", "input_stats"):
        with pytest.raises(ValueError, match="no T04 edge-extension grid"):
            C.run_method(t04, method, tmp_path, edge=True, t04=True, **kwargs)
    assert C.T04_EDGE_LABEL == EE.EXTENSION_LABEL == "t04g3"
    with pytest.raises(ValueError, match="not a registered bn_adapt grid"):
        BN.run_bn_ref(t04, "tent", tmp_path, priors=BN.BN_ADAPT_PRIORS_T04_EDGE)
    e12 = SimpleNamespace(private={"dataset": FT.F1_E12_DATASET}, instance_id="x")
    with pytest.raises(FT.F1Error, match="edge extension is PREREG T04's"):
        FT.run_f1(e12, "ft_bn", tmp_path, edge=True)


def _record(method: str, primary: str | None, kept: str | None) -> dict[str, Any]:
    return {"method": method, "condition": "c", "instance_id": "i0",
            "selection": {"primary_patch_id": primary, "patch_id": kept}}


def _run(configs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"candidates": [{"patch_id": f"q{i}", "config": c} for i, c in enumerate(configs)]}


def test_pair_rule_cases() -> None:
    lr_grid = [{"lr": lr, "steps": s} for lr in (1e-5, 1e-4) for s in (10, 30, 100, 300)]
    on_primary = EE.decide(_record("grad_full", "q0", None), _run(lr_grid), "t04g2")
    assert on_primary["extended"] and on_primary["trigger"] == ['primary q0 {"lr": 1e-05, "steps": 10}']
    assert on_primary["configs"][:4] == [{"lr": 1e-6, "steps": s} for s in (10, 30, 100, 300)]
    assert on_primary["repeats_of_source"] == [4, 5, 6, 7]
    on_kept = EE.decide(_record("ft_bn", "q6", "q1"), _run(lr_grid), "f1")
    assert on_kept["extended"] and on_kept["trigger"] == ['kept q1 {"lr": 1e-05, "steps": 30}']
    off = EE.decide(_record("ft_bn", "q5", "q5"), _run(lr_grid), "f1")
    assert not off["extended"] and off["reason"] == "primary and kept off the edge"
    base = EE.decide(_record("grad_last", None, None), _run(lr_grid), "t04")
    assert not base["extended"] and base["reason"] == "primary and kept are the base"
    priors = [{"N": n} for n in BN.BN_ADAPT_PRIORS]
    assert EE.decide(_record("bn_adapt", "q7", None), _run(priors), "t04")["extended"]
    assert not EE.decide(_record("bn_adapt", "q5", None), _run(priors), "t04")["extended"]
    for method in ("prior_shift", "ridge_last", "tie_inputs", "input_stats", "ft_large"):
        assert not EE.decide(_record(method, "q0", "q0"), _run(lr_grid), "t04")["extended"]
    with pytest.raises(EE.EdgeError, match="not a candidate"):
        EE.decide(_record("grad_full", "q9", None), _run(lr_grid), "t04g2")


def test_pair_script_reads_kept_files_and_maps_the_store(tmp_path: Path) -> None:
    store = tmp_path / "store"
    run_dir = store / "runs" / "baselines" / "i0" / "grad_full" / "t04g2" / "0"
    run_dir.mkdir(parents=True)
    lr_grid = [{"lr": lr, "steps": s} for lr in (1e-5, 1e-4) for s in (10, 30, 100, 300)]
    (run_dir / "run.json").write_text(json.dumps(_run(lr_grid)))
    kept = store / "runs" / "devreadouts" / "t04" / "c"
    kept.mkdir(parents=True)
    record = _record("grad_full", "q0", None) | {"run": "/previous/store/runs/baselines/i0/grad_full/t04g2/0/run.json"}
    (kept / "kept_x.jsonl").write_text(json.dumps(record) + "\n")
    out = tmp_path / "pairs.json"
    assert EE.main(["--devreadouts", str(store / "runs" / "devreadouts" / "t04"), "--store", str(store),
                    "--conditions", "c", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["n_extended"] == 1 and result["decisions"][0]["label"] == "t04g2"


@pytest.mark.parametrize("recorded", ["/previous/store/runs/baselines/run.json", "runs/baselines/run.json"])
def test_recorded_runs_paths_relocate_to_the_explicit_store(recorded: str, tmp_path: Path) -> None:
    assert relocate_run_path(recorded, tmp_path) == tmp_path / "runs/baselines/run.json"


@pytest.mark.parametrize("recorded", ["private/run.json", "runs/../private/run.json", "runs/nested/runs/run.json"])
def test_recorded_runs_paths_refuse_ambiguous_or_escaping_roots(recorded: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a store runs path"):
        relocate_run_path(recorded, tmp_path)
