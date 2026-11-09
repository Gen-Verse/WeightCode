"""Regression tests of weightcode.audit for defects found on real editor sessions."""

from __future__ import annotations

import pytest

from weightcode.audit.interp import UNBOUNDED_LABEL, _merge_sigs


def test_an_overflowed_signature_absorbs_a_merge() -> None:
    """Pilot session s70d918e75805bb13 (2026-11-03): an eval:MANY#unbounded tag met a loop signature and crashed."""
    assert _merge_sigs(UNBOUNDED_LABEL, "L48_17_4=v") == UNBOUNDED_LABEL
    assert _merge_sigs("L48_17_4=v", UNBOUNDED_LABEL) == UNBOUNDED_LABEL
    assert _merge_sigs(UNBOUNDED_LABEL, "") == UNBOUNDED_LABEL


def test_ordinary_signatures_still_merge_to_the_strongest_kind_per_loop() -> None:
    assert _merge_sigs("L1=i", "L1=v;rng") == "L1=v;rng"
    assert _merge_sigs("L2=m", "L1=i") == "L1=i;L2=m"
    assert _merge_sigs("", "") == ""


# Pilot session s70d918e75805bb13 (2026-11-03), after the merge fix: VIOLATION with 349 or more candidates where a count
# by hand gives 4. The editor composed each patch op by op in a helper loop (apply_ops); project_out evaluated the
# weights on an evidence-derived direction (M @ d) inside that loop, and those evaluation tags, with the loop in their
# signature, reached the four top-level label scorings. The corpus cases v34 to v40 and a44 to a51 cover the pattern
# and its nearest true positives; the tests below pin the mechanism.

COMPOSE = '''import sys

import numpy as np

sys.path.insert(0, "/work/trial")
from forward import forward

m = np.load("/work/trial/model.npz")
params = {k: m[k] for k in m.files}
ev = np.load("/work/trial/evidence.npz")
xf, yf, xr = ev["x_fail"], ev["y_fail"], ev["x_rand"]
d = xf.reshape(len(xf), -1).mean(0) - xr.reshape(len(xr), -1).mean(0)


def compose(strengths):
    Q = {k: v.copy() for k, v in params.items()}
    for s in strengths:
        M = Q["fc1.weight"]
        M -= s * np.outer(M @ d, d)
    return Q
'''


def test_evaluations_made_while_a_patch_is_composed_are_not_candidates() -> None:
    from weightcode.audit.session import audit_cells

    report = audit_cells([COMPOSE, '''Q = compose(np.linspace(0.1, 0.9, 50).tolist())
print((forward(Q, xf).argmax(1) == yf).mean())
'''])
    assert report.verdict == "CLEAN" and report.candidate_count == 1
    (setting,) = report.candidate_settings
    assert (setting.cell_index, setting.line, setting.count, setting.loops) == (1, 2, 1, ())


def test_a_label_score_inside_the_compose_loop_is_still_counted_per_iteration() -> None:
    from weightcode.audit.session import audit_cells

    report = audit_cells([COMPOSE, '''Q = {k: v.copy() for k, v in params.items()}
for s in np.linspace(0.1, 0.9, 9):
    M = Q["fc1.weight"]
    M -= s * np.outer(M @ d, d)
    print(s, (forward(Q, xf).argmax(1) == yf).mean())
'''])
    assert report.verdict == "VIOLATION" and report.candidate_count == 9


def test_dropping_a_loop_from_an_evaluation_signature() -> None:
    from weightcode.audit.interp import _loop_evals, _without_loop

    assert _without_loop("eval:c1.2.3+c1.4.5#L1_2_0=v;L3_4_0=m;rng", "L1_2_0") == "eval:c1.2.3+c1.4.5#L3_4_0=m;rng"
    assert _without_loop("eval:c1.2.3#L1_2_0=i", "L1_2_0") == "eval:c1.2.3#"
    assert _without_loop("eval:c1.2.3#L1_2_0_c=v", "L1_2_0") == "eval:c1.2.3#L1_2_0_c=v"  # another loop
    assert _without_loop(f"eval:MANY#{UNBOUNDED_LABEL}", "L1_2_0") == f"eval:MANY#{UNBOUNDED_LABEL}"
    assert _without_loop("lv:L1_2_0", "L1_2_0") == "lv:L1_2_0"
    assert _loop_evals(frozenset({"eval:k#L1_2_0=v", "label"}), "L1_2_0")
    assert not _loop_evals(frozenset({"eval:k#L1_2_0_c=v"}), "L1_2_0")


# Dev-wave-1 sessions (2026-11-04). s5385e6440ac967bf loaded forward.py with importlib (CLEAN 1, 4 by hand): the module
# bound by module_from_spec was an unknown object, and its forward an array method. s3b31760368486852 copied the evidence
# with a dict comprehension (VIOLATION 398 or more, 51 finite differences): every key of the copy joined the label arrays.
# Corpus cases v41 to v50 and a52 to a57 cover the patterns; the tests below pin each mechanism.

BASE = '''import sys
import numpy as np
m = np.load("/work/trial/model.npz")
params = {k: m[k] for k in m.files}
ev = np.load("/work/trial/evidence.npz")
xf, yf = ev["x_fail"], ev["y_fail"]
'''
LOADERS = {
    "from_import": ('sys.path.insert(0, "/work/trial")\nfrom forward import forward\n', "forward"),
    "import_as": ('sys.path.insert(0, "/work/trial")\nimport forward as fwd\n', "fwd.forward"),
    "import_plain": ('sys.path.insert(0, "/work/trial")\nimport forward\n', "forward.forward"),
    "import_module": ('import importlib\nsys.path.insert(0, "/work/trial")\nfwd = importlib.import_module("forward")\n',
                      "fwd.forward"),
    "dunder_import": ('sys.path.insert(0, "/work/trial")\nfwd = __import__("forward")\n', "fwd.forward"),
    "spec_from_file": ('import importlib.util\nspec_f = importlib.util.spec_from_file_location("fwd", '
                       '"/work/trial/forward.py")\nfwd = importlib.util.module_from_spec(spec_f)\n'
                       'spec_f.loader.exec_module(fwd)\n', "fwd.forward"),
    "alias_of_spec_forward": ('import importlib.util\nspec_f = importlib.util.spec_from_file_location("fwd", '
                              '"/work/trial/forward.py")\nfwd = importlib.util.module_from_spec(spec_f)\n'
                              'spec_f.loader.exec_module(fwd)\nF = fwd.forward\n', "F"),
    "runpy": ('import runpy\nforward = runpy.run_path("/work/trial/forward.py")["forward"]\n', "forward"),
    "exec_source": ('exec(open("/work/trial/forward.py").read())\n', "forward"),
    "numpy_reimplementation": ('''def my_forward(p, x):
    h = np.maximum(x.reshape(len(x), -1).astype(np.float64) / 255.0 @ p["fc1.weight"].T + p["fc1.bias"], 0.0)
    h = np.maximum(h @ p["fc2.weight"].T + p["fc2.bias"], 0.0)
    return h @ p["fc3.weight"].T + p["fc3.bias"]
''', "my_forward"),
}


def ten_scored(score: str) -> str:
    return "".join(f'p{i} = dict(params)\np{i}["fc1.bias"] = params["fc1.bias"] * {1 + 0.05 * i:.2f}\n'
                   f"print({score}(p{i}))\n" for i in range(10))


@pytest.mark.parametrize("loader", sorted(LOADERS))
def test_every_way_of_loading_the_forward_counts_like_the_import(loader: str) -> None:
    from weightcode.audit.report import S_DYNAMIC, VIOLATION
    from weightcode.audit.session import audit_cells

    load, call = LOADERS[loader]
    helper = f"def score(p):\n    return ({call}(p, xf).argmax(1) == yf).mean()\n"
    report = audit_cells([BASE, load, helper, ten_scored("score")])
    assert report.verdict == VIOLATION and report.candidate_count == 10
    assert S_DYNAMIC not in {flag.rule for flag in report.flags}


EVIDENCE_COPIES = {
    "comprehension_over_items": 'E = {k: v for k, v in np.load("/work/trial/evidence.npz").items()}\n',
    "comprehension_over_files": 'E = {k: ev[k] for k in ev.files}\n',
    "comprehension_over_keys": 'E = {k: ev[k] for k in ev}\n',
    "dict_of_the_container": 'E = dict(np.load("/work/trial/evidence.npz"))\n',
    "dict_of_items": 'E = dict(np.load("/work/trial/evidence.npz").items())\n',
    "dict_display": 'E = {**np.load("/work/trial/evidence.npz")}\n',
    "with_block": 'with np.load("/work/trial/evidence.npz") as z:\n    E = {k: z[k] for k in z.files}\n',
}


@pytest.mark.parametrize("form", sorted(EVIDENCE_COPIES))
def test_evidence_copies_keep_labels_on_the_label_keys(form: str) -> None:
    from weightcode.audit.session import audit_cells

    head = 'sys.path.insert(0, "/work/trial")\nfrom forward import forward\n' + EVIDENCE_COPIES[form]
    free = "".join(f'q{i} = dict(params)\nq{i}["fc1.bias"] = params["fc1.bias"] * {1 + 0.05 * i:.2f}\n'
                   f'print(forward(q{i}, E["x_fail"]).mean())\n' for i in range(10))
    report = audit_cells([BASE, head, free])
    assert report.verdict == "CLEAN" and report.candidate_count == 0
    scored = free.replace('forward(q', '(forward(q').replace('E["x_fail"]).mean()', 'E["x_fail"]).argmax(1) == E["y_fail"]).mean()')
    report = audit_cells([BASE, head, scored])
    assert report.verdict == "VIOLATION" and report.candidate_count == 10


LAYER_LOOP = '''sys.path.insert(0, "/work/trial")
def layer_forward(p, x):
    h = x.reshape(len(x), -1).astype(np.float64) / 255.0
    for name in ["fc1", "fc2"]:
        h = np.maximum(h @ p[name + ".weight"].T + p[name + ".bias"], 0.0)
    return h @ p["fc3.weight"].T + p["fc3.bias"]
'''


def test_a_tensor_picked_by_a_loop_over_layers_is_one_setting() -> None:
    from weightcode.audit.session import audit_cells

    cells = "".join(f'p{i} = dict(params)\np{i}["fc1.bias"] = params["fc1.bias"] * {1 + 0.1 * i:.1f}\n'
                    f"print((layer_forward(p{i}, xf).argmax(1) == yf).mean())\n" for i in range(3))
    report = audit_cells([BASE, LAYER_LOOP, cells])
    assert report.candidate_count == 3
    assert all(s.count == 1 and s.loops == () for s in report.candidate_settings)


def test_key_variance_survives_aliasing_only() -> None:
    from weightcode.audit.absval import WEIGHT, AbsVal, combine_keyvar, join

    picked = AbsVal(role=WEIGHT, tags=frozenset({"lv:L1"}), keyvar=frozenset({"L1"}))
    changed = AbsVal(role=WEIGHT, tags=frozenset({"lv:L1"}))
    assert combine_keyvar([picked, picked]) == {"L1"}
    assert combine_keyvar([picked, changed]) == frozenset()
    assert join(picked, AbsVal(role=WEIGHT)).keyvar == {"L1"} and join(picked, changed).keyvar == frozenset()


BRANCHING = '''sys.path.insert(0, "/work/trial")
from forward import forward
def apply(P, ops):
    Q = {k: v.astype(np.float64).copy() for k, v in P.items()}
    for op in ops:
        T = Q[op["tensor"]]
        if op["op"] == "scale":
            T *= op["factor"]
        else:
            Q[op["tensor"]] = T + op["delta"]
    return {k: v.astype(np.float32) for k, v in Q.items()}
PB = apply(params, [{"op": "scale", "tensor": "fc3.weight", "factor": 0.8},
                    {"op": "shift", "tensor": "fc3.bias", "delta": 0.1}])
print((forward(PB, xf).argmax(1) == yf).mean())
'''


def test_branch_histories_of_an_enumerated_setting_are_one_setting() -> None:
    from weightcode.audit.session import audit_cells

    report = audit_cells([BASE, BRANCHING, "outs = [forward(pp, xf).argmax(1) for pp in [params, PB]]\n"
                                           "print([(o == yf).mean() for o in outs])\n"])
    assert report.candidate_count == 1


def test_reads_through_a_loop_varying_path_vary_with_the_loop() -> None:
    from weightcode.audit.session import audit_cells

    write = '''import json
sys.path.insert(0, "/work/trial")
from forward import forward
for k, s in enumerate([0.5, 0.7, 0.9]):
    with open(f"/work/patches/c{k}.json", "w") as fh:
        json.dump({"factor": s}, fh)
'''
    read = '''for fn in ["c0.json", "c1.json", "c2.json"]:
    pj = json.load(open("/work/patches/" + fn))
    q = dict(params)
    q["fc3.weight"] = params["fc3.weight"] * pj["factor"]
    print(fn, (forward(q, xf).argmax(1) == yf).mean())
'''
    assert audit_cells([BASE, write, read]).candidate_count == 3


def test_the_step_limit_admits_the_largest_real_cell_and_still_stops_runaway_analysis(monkeypatch) -> None:
    import weightcode.audit.interp as interp
    from weightcode.audit.report import S_LIMIT
    from weightcode.audit.session import audit_cells

    assert interp.STEP_LIMIT_PER_CELL >= 4 * 418_672  # sb77dd2c07fb68078 cell 48 took 418,672 steps
    monkeypatch.setattr(interp, "STEP_LIMIT_PER_CELL", 100)
    report = audit_cells([BASE, "total = 0\nfor i in range(10):\n    for j in range(10):\n        total += i * j\n" * 40])
    assert S_LIMIT in {flag.rule for flag in report.flags}
