"""E9 reference streams (``weightcode.baselines.fit.fscilrefs``) on the toy store of ``fscil_toy``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fscil_toy import toy_fscil  # noqa: F401

from weightcode.baselines.fit import fscilrefs as FR
from weightcode.common.identity import params_sha256, save_params
from weightcode.data.store import forward_path
from weightcode.evalkit import fscil as FS
from weightcode.evalkit.patch_check import predict


def _features(params, x):
    namespace: dict = {}
    exec(compile(forward_path(FS.DATASET).read_text(), "forward.py", "exec"), namespace)
    return namespace["features"](params, x)


def test_session_rows_are_the_instance_rows(toy_fscil, tmp_path: Path) -> None:  # noqa: F811
    data = FR.load_stream_data(need_allclass=False)
    rows2 = FR.session_rows(data, 2)
    instance = FS.load_fscil_instance(FS.build_fscil_instance(2, lineage="T")["instance_id"])
    ev = instance.evidence
    assert np.array_equal(rows2.x_fit, np.concatenate([ev["x_new"], ev["x_old"], ev["x_prev"]]))
    assert np.array_equal(rows2.y_fit, np.concatenate([ev["y_new"], ev["y_old"], ev["y_prev"]]))
    assert (rows2.record["n_new"], rows2.record["n_old"], rows2.record["n_prev"]) == (25, 64, 0)
    save_params(tmp_path / "p.npz", data.base_params)
    third = FS.load_fscil_instance(FS.build_fscil_instance(3, lineage="T", parent_params=tmp_path / "p.npz",
                                                           parent_instance=instance.instance_id)["instance_id"])
    rows3 = FR.session_rows(data, 3)
    ev3 = third.evidence
    assert np.array_equal(rows3.x_fit, np.concatenate([ev3["x_new"], ev3["x_old"], ev3["x_prev"]]))
    assert rows3.record["n_check_old"] == third.private["n_vcheck_old"]
    assert rows3.limits.max_changed_frac == FS.FSCIL_MAX_CHANGED_FRAC and rows3.limits.max_edit_frac is None


def test_rebuild_parity_and_imprint_rows(toy_fscil) -> None:  # noqa: F811
    data = FR.load_stream_data(need_allclass=False)
    rows = FR.session_rows(data, 2)
    ctx = FR.FitContext("imprint", rows, dict(data.base_params), data.mean, data.std).bind(data.forward)
    assert ctx.parity["float64_max_rel_diff"] <= ctx.parity["float64_tolerance"]
    assert np.allclose(ctx.f64, _features(data.base_params, rows.x_fit), rtol=1e-7, atol=1e-9)
    candidates = FR.generate(ctx)
    assert [c.config["direction"] for c in candidates] == ["mean", "centered"]
    new = rows.part_fit == "new"
    f = _features(data.base_params, rows.x_fit)
    for c, label in zip(range(5), FS.session_classes(2)):
        expected = f[new & (rows.y_fit == label)].mean(axis=0).astype(np.float32)
        assert np.allclose(candidates[0].child["fc.weight"][label], expected, rtol=1e-6, atol=1e-7)
    for candidate in candidates:
        changed = np.flatnonzero(np.any(candidate.child["fc.weight"] != data.base_params["fc.weight"], axis=1))
        assert set(changed.tolist()) <= set(FS.session_classes(2))
        assert all(np.array_equal(candidate.child[k], data.base_params[k]) for k in data.base_params if k != "fc.weight")
        assert candidate.notes["edit"]["within_limits"]


def test_retain_beta_breaks_no_protected_row_when_it_can(toy_fscil) -> None:  # noqa: F811
    data = FR.load_stream_data(need_allclass=False)
    rows = FR.session_rows(data, 2)
    ctx = FR.FitContext("imprint_retain", rows, dict(data.base_params), data.mean, data.std).bind(data.forward)
    for candidate in FR.generate(ctx):
        retain = candidate.notes["retain"]
        assert retain["beta"] in FR.RETAIN_BETAS and len(retain["ladder"]) == len(FR.RETAIN_BETAS)
        pred = predict(FS.masked_forward(data.forward, 2), candidate.child, rows.x_fit, 256)
        protected = (rows.part_fit != "new") & (ctx.pred == rows.y_fit)
        assert int(np.count_nonzero(protected & (pred != rows.y_fit))) == retain["broken_protected"]
        smaller = [e for e in retain["ladder"] if e["beta"] < retain["beta"]]
        assert all(e["broken"] > retain["broken_protected"] for e in smaller)


def test_every_method_respects_its_scope_and_limits(toy_fscil) -> None:  # noqa: F811
    data = FR.load_stream_data(need_allclass=False)
    rows = FR.session_rows(data, 2)
    base = dict(data.base_params)
    trainable = FR.trainable_names(base)
    k = FS.n_seen(2)
    expected_count = {"imprint": 2, "imprint_plain": 1, "imprint_retain": 2}
    for method in FR.STREAMS:
        ctx = FR.FitContext(method, rows, base, data.mean, data.std).bind(data.forward)
        candidates = FR.generate(ctx)
        assert len(candidates) == expected_count.get(method, 8), method
        assert len({c.patch_id for c in candidates}) == len(candidates)
        for c in candidates:
            for key in base:
                if key.endswith(("running_mean", "running_var")):
                    assert np.array_equal(c.child[key], base[key]), (method, key)
            if method in ("ridge_margin", "grad_last"):
                assert all(np.array_equal(c.child[x], base[x]) for x in base if x != "fc.weight")
                assert np.array_equal(c.child["fc.weight"][k:], base["fc.weight"][k:]), method
            if method.startswith("grad_sparse19"):
                for key in trainable:
                    frac = np.count_nonzero(c.child[key] != base[key]) / base[key].size
                    assert frac <= FR.MASK_FRACTION + 1e-12, (method, key)
            assert c.notes["edit"]["within_limits"] or method in FR.LIMITS_IGNORED, (method, c.notes["edit"])
    interp = FR.generate(FR.FitContext("interp", rows, base, data.mean, data.std).bind(data.forward))
    full = FR.theta_full(FR.FitContext("interp", rows, base, data.mean, data.std).bind(data.forward))
    assert all(np.array_equal(interp[-1].child[x], full[x].astype(base[x].dtype)) for x in trainable)


def _sphere_objective(h1, t, w0, lam, w):
    return float(np.sum((h1 @ w - t) ** 2) + lam * np.sum((w - w0) ** 2))


def test_sphere_ridge_rows_is_the_unit_minimizer() -> None:
    rng = np.random.default_rng(0)
    h1 = rng.normal(size=(40, 3))
    w0 = rng.normal(size=(4, 3))
    w0 /= np.linalg.norm(w0, axis=1, keepdims=True)
    targets = 3.0 * rng.normal(size=(40, 4))
    sphere = rng.normal(size=(200_000, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    for lam in (1e-3, 1.0, 50.0):
        rows = FR.sphere_ridge_rows(h1, targets, w0, lam)
        assert np.allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1e-12)
        free = np.linalg.solve(h1.T @ h1 + lam * np.eye(3), h1.T @ targets + lam * w0.T).T
        for j in range(4):
            mine = _sphere_objective(h1, targets[:, j], w0[j], lam, rows[j])
            grid = (np.sum((sphere @ h1.T - targets[:, j]) ** 2, axis=1)
                    + lam * np.sum((sphere - w0[j]) ** 2, axis=1)).min()
            assert mine <= grid + 1e-9 * max(1.0, abs(grid))
            renorm = free[j] / np.linalg.norm(free[j])
            assert mine <= _sphere_objective(h1, targets[:, j], w0[j], lam, renorm) + 1e-9
    same = FR.sphere_ridge_rows(h1, h1 @ w0.T, w0, 1.0)  # targets the current logits: the rows stay
    assert np.allclose(same, w0, atol=1e-9)
    with pytest.raises(FR.FscilRefError):
        FR.sphere_ridge_rows(h1, targets, w0, 0.0)


def test_sphere_ridge_rows_hard_case() -> None:
    h1 = np.diag(np.sqrt([0.0, 1.0, 2.0, 3.0]))  # G = diag(0.5, 1.5, 2.5, 3.5); b = (0, 0.5, 0, 0) has no e_0 part
    w0 = np.array([[0.0, 1.0, 0.0, 0.0]])
    rows = FR.sphere_ridge_rows(h1, np.zeros((4, 1)), w0, 0.5)
    assert np.allclose(np.abs(rows[0]), [np.sqrt(0.75), 0.5, 0.0, 0.0], atol=1e-12)
    assert _sphere_objective(h1, np.zeros(4), w0[0], 0.5, rows[0]) < _sphere_objective(h1, np.zeros(4), w0[0], 0.5,
                                                                                       w0[0])


def test_keep_rule_floor_ties_and_base() -> None:
    log = [{"patch_id": "a", "score": 5, "score_old": -30, "n_changed": 10, "query_index": 0},
           {"patch_id": "b", "score": 3, "score_old": -5, "n_changed": 10, "query_index": 1},
           {"patch_id": "c", "score": 3, "score_old": 0, "n_changed": 4, "query_index": 2}]
    assert FR.keep(log, -20).patch_id == "c"  # a is below the floor; b and c tie on S, c changes fewer entries
    assert FR.keep(log, -40).patch_id == "a"
    assert FR.keep([{**log[1], "score": -1}], -20).patch_id is None  # the current student (S = 0) wins
    flips = FR.flip_rate({"x": [{"n_old": 10, "fixed_old": 1, "broken_old": 2}],
                          "y": [{"n_old": 10, "fixed_old": 0, "broken_old": 0},
                                {"n_old": 10, "fixed_old": 5, "broken_old": 5}]}, 10)
    assert flips["d"] == pytest.approx(0.3) and flips["n_checks"] == 3
    own = [{"patch_id": "p", "score": 9, "score_old": -12, "n_changed": 3, "query_index": 0, "fixed_old": 30,
            "broken_old": 42},
           {"patch_id": "q", "score": 4, "score_old": -6, "n_changed": 3, "query_index": 1, "fixed_old": 0,
            "broken_old": 6}]
    floors = FR.candidate_floors(own, 1800, 1800, -1.0)
    assert floors == [FS.joint_retain_floor(1800, 1800, -1.0, 72 / 1800), FS.joint_retain_floor(1800, 1800, -1.0,
                                                                                                6 / 1800)]
    assert floors == [0, -14]  # p flips 4% of the old rows (floor 0, S_old -12 fails); q flips 0.3% (floor -14)
    assert FR.keep_per_candidate(own, 1800, 1800, -1.0).patch_id == "q"
    assert FR.keep(own, -11).patch_id == "q" and FR.keep(own, -14).patch_id == "p"  # one shared floor


def test_chunked_forward_equals_the_oracle_batches(toy_fscil) -> None:  # noqa: F811
    data = FR.load_stream_data(need_allclass=False)
    x = data.splits["V_select"]["x"]
    chunked = FR.ChunkedForward(data.forward, pool_map=lambda fn, args: [fn(*a) for a in args],
                                chunk_fn=lambda params, chunk: np.asarray(data.forward(params, chunk)))
    big = predict(FS.masked_forward(chunked, 4), data.base_params, x, 10**9)
    small = predict(FS.masked_forward(data.forward, 4), data.base_params, x, FS.EVALUATION_BATCH)
    assert np.array_equal(big, small)


_VOLATILE = ("fit_s", "trajectory_s_to_checkpoint", "model")


def _stable(value):
    """A session record without timings and the model path (they differ between two output directories)."""
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def test_lockstep_run_and_resume(toy_fscil, tmp_path: Path) -> None:  # noqa: F811
    data = FR.load_stream_data(need_allclass=False)
    methods = ("imprint", "imprint_retain", "ridge_margin", "grad_last", "imprint_plain")
    one = FR.run_streams(data, r_points=-1.0, out_dir=tmp_path / "one", sessions=[2, 3], methods=methods,
                         log=lambda _: None)
    FR.run_streams(data, r_points=-1.0, out_dir=tmp_path / "two", sessions=[2], methods=methods, log=lambda _: None)
    two = FR.run_streams(data, r_points=-1.0, out_dir=tmp_path / "two", sessions=[3], methods=methods,
                         log=lambda _: None)
    assert one[3]["floor"] == two[3]["floor"] and one[3]["d"] == two[3]["d"]
    for t in (2, 3):
        a = json.loads((tmp_path / "one" / f"session_{t}.json").read_text())
        b = json.loads((tmp_path / "two" / f"session_{t}.json").read_text())
        assert _stable(a) == _stable(b)
        n_c, n_s = FS.old_part_sizes(t)
        assert a["floor"]["floor"] == FS.joint_retain_floor(n_c, n_s, -1.0, a["floor"]["d"])
        for method, record in a["streams"].items():
            kept = record["kept"]
            if method == "imprint_plain":
                assert record["checks"] is None and kept["patch_id"] == f"imprint_plain_s{t}_q0"
            elif kept["patch_id"] is not None:
                assert kept["check"]["score_old"] >= a["floor"]["floor"]
            supplement = record["supplement_per_candidate_floor"]
            if method == "imprint_plain":
                assert supplement is None
            else:
                assert supplement["floors"] == FR.candidate_floors(record["checks"], n_c, n_s, -1.0)
                assert supplement["readout"]["absolute"]["session"] == t
                if supplement["pick"] is not None:
                    i = [c["patch_id"] for c in record["checks"]].index(supplement["pick"])
                    assert record["checks"][i]["score_old"] >= supplement["floors"][i]
            assert record["readout"]["absolute"]["session"] == t
    s3 = json.loads((tmp_path / "one" / "session_3.json").read_text())
    for method, record in s3["streams"].items():
        s2 = json.loads((tmp_path / "one" / "session_2.json").read_text())["streams"][method]
        assert record["fit"]["parent_params_sha256"] == s2["kept"]["params_sha256"]
    with pytest.raises(FR.FscilRefError):
        FR.run_streams(data, r_points=-1.0, out_dir=tmp_path / "three", sessions=[3], methods=methods,
                       log=lambda _: None)
    assert params_sha256(data.base_params) == toy_fscil.base_params_sha256
