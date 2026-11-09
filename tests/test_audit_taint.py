"""Label taint across cells of one persistent namespace (contract section 5.5, HARNESS_SPEC section 7).

Each test is a short session whose label use is spread over several cells: through names, functions, containers,
files and instances. Controls check that the same paths carry no taint when no label is involved.
"""

from __future__ import annotations

import pytest

from weightcode.audit import report as R
from weightcode.audit.session import audit_cells

LOAD = '''import json
import numpy as np
m = np.load("/work/trial/model.npz")
params = {k: m[k] for k in m.files}
ev = np.load("/work/trial/evidence.npz")
xr, yr, pr = ev["x_rand"], ev["y_rand"], ev["pred_rand"]
xf, yf, pf = ev["x_fail"], ev["y_fail"], ev["pred_fail"]
H = np.maximum(xr.reshape(len(xr), -1) / 255.0 @ params["fc1.weight"].T + params["fc1.bias"], 0)
'''
FIT = 'W, *_ = np.linalg.lstsq(H, T, rcond=None)\nparams["fc2.weight"] = W.T\n'


def audit(*cells: str) -> R.AuditReport:
    return audit_cells([LOAD, *cells])


def rules(report: R.AuditReport) -> set[str]:
    return {flag.rule for flag in report.flags}


def assert_solver_violation_at(report: R.AuditReport, cell: int) -> None:
    assert report.verdict == R.VIOLATION, [(f.rule, f.cell_index, f.reason) for f in report.flags]
    assert any(f.rule == R.V_SOLVER and f.cell_index == cell for f in report.flags), report.flags


def test_label_alias_defined_in_one_cell_and_used_two_cells_later() -> None:
    report = audit("labels = yr\n", "T = np.eye(10)[labels]\n", FIT)
    assert_solver_violation_at(report, 3)


def test_function_defined_in_one_cell_and_called_with_labels_in_another() -> None:
    helper = '''def ridge(features, classes, lam=1e-2):
    targets = np.eye(10)[classes]
    gram = features.T @ features + lam * np.eye(features.shape[1])
    return np.linalg.solve(gram, features.T @ targets)
'''
    report = audit(helper, "W = ridge(H, yr)\nparams['fc2.weight'] = W.T\n")
    (flag,) = [f for f in report.flags if f.rule == R.V_SOLVER]
    assert (flag.cell_index, flag.line) == (1, 4), "the flag points into the function body"
    assert "reached from the call at cell 2 line 1" in flag.reason


def test_same_function_called_without_labels_is_clean() -> None:
    helper = '''def ridge(features, targets, lam=1e-2):
    gram = features.T @ features + lam * np.eye(features.shape[1])
    return np.linalg.solve(gram, features.T @ targets)
'''
    report = audit(helper, "W = ridge(H, np.eye(10)[pr])\n")
    assert report.verdict == R.CLEAN, "base predictions are not labels"


def test_rebinding_a_name_clears_its_taint() -> None:
    report = audit("T = np.eye(10)[yr]\n", "T = np.eye(10)[pr]\n", FIT)
    assert report.verdict == R.CLEAN


def test_rebinding_in_one_branch_keeps_the_taint() -> None:
    report = audit("T = np.eye(10)[yr]\n", "if H.shape[0] > 100:\n    T = np.eye(10)[pr]\n", FIT)
    assert_solver_violation_at(report, 3)


def test_taint_through_a_dict_across_cells() -> None:
    report = audit('store = {"targets": yr, "feats": H}\n', 'T = np.eye(10)[store["targets"]]\n', FIT)
    assert_solver_violation_at(report, 3)


def test_untainted_dict_entry_stays_clean() -> None:
    report = audit('store = {"targets": yr, "preds": pr}\n', 'T = np.eye(10)[store["preds"]]\n', FIT)
    assert report.verdict == R.CLEAN


def test_taint_through_a_list_mutated_in_a_later_cell() -> None:
    report = audit("bag = []\n", "bag.append(yr)\n", "T = np.eye(10)[bag[0]]\n", FIT)
    assert_solver_violation_at(report, 4)


def test_taint_through_a_saved_scratch_file() -> None:
    report = audit('np.save("/work/scratch/t.npy", np.eye(10)[yr])\n', 'T = np.load("/work/scratch/t.npy")\n', FIT)
    assert_solver_violation_at(report, 3)


def test_taint_through_a_json_file() -> None:
    write = 'json.dump({"y": yr.tolist()}, open("/work/scratch/y.json", "w"))\n'
    read = 'T = np.eye(10)[np.array(json.load(open("/work/scratch/y.json"))["y"])]\n'
    report = audit(write, read, FIT)
    assert_solver_violation_at(report, 3)


def test_taint_through_an_instance_attribute_set_in_an_earlier_cell() -> None:
    cls = '''class Box:
    def __init__(self, labels):
        self.labels = labels
'''
    report = audit(cls, "box = Box(yr)\n", "T = np.eye(10)[box.labels]\n", FIT)
    assert_solver_violation_at(report, 4)


def test_taint_through_a_closure_and_a_default_argument() -> None:
    report = audit("make_t = lambda: np.eye(10)[yr]\n", "T = make_t()\n", FIT)
    assert_solver_violation_at(report, 3)
    report = audit("def targets(y=yr):\n    return np.eye(10)[y]\n", "T = targets()\n", FIT)
    assert_solver_violation_at(report, 3)


def test_taint_through_globals() -> None:
    report = audit('globals()["T"] = np.eye(10)[yr]\n', FIT)
    assert_solver_violation_at(report, 2)


def test_error_signal_built_over_three_cells() -> None:
    report = audit(
        "z = forward_z = H @ params['fc2.weight'].T\np = np.exp(z - z.max(1, keepdims=True))\np /= p.sum(1, keepdims=True)\n",
        "e = p - np.eye(p.shape[1])[yr]\n",
        "G = e.T @ H / len(yr)\n",
        "params['fc2.weight'] = params['fc2.weight'] - 0.1 * G\n",
    )
    assert report.verdict == R.VIOLATION
    assert R.V_BACKPROP in rules(report)


def test_gradient_without_update_is_soft() -> None:
    report = audit(
        "z = H @ params['fc2.weight'].T\np = np.exp(z - z.max(1, keepdims=True))\np /= p.sum(1, keepdims=True)\n",
        "G = (p - np.eye(p.shape[1])[yr]).T @ H\nprint(np.abs(G).max())\n",
    )
    assert report.verdict == R.SOFT_FLAGS and R.S_GRAD_UNUSED in rules(report)


def test_label_derived_mask_taints_the_selection() -> None:
    report = audit("sel = yr == 3\n", "T = np.eye(10)[sel.astype(int)]\n", FIT)
    assert_solver_violation_at(report, 3)


def test_accuracy_is_label_derived() -> None:
    report = audit("acc_r = float((pr == yr).mean())\n", "T = np.full((len(H), 10), acc_r)\n", FIT)
    assert_solver_violation_at(report, 3)


# ------------------------------------------------------------------------------------- candidates across cells

SCALE = '''for s in {scales}:
    w = params["fc2.weight"] * s
    h = np.maximum(xr.reshape(len(xr), -1) / 255.0 @ params["fc1.weight"].T + params["fc1.bias"], 0)
    print(s, ((h @ w.T).argmax(1) == yr).mean())
'''


@pytest.mark.parametrize("first, second, verdict, count", [
    ((0.5, 0.8, 1.2, 1.5), (0.6, 0.9, 1.1, 1.4), R.CLEAN, 8),
    ((0.5, 0.8, 1.2, 1.5, 1.7), (0.6, 0.9, 1.1, 1.4), R.VIOLATION, 9),
])
def test_candidates_add_up_over_cells(first: tuple[float, ...], second: tuple[float, ...], verdict: str,
                                      count: int) -> None:
    report = audit(SCALE.format(scales=first), SCALE.format(scales=second))
    assert report.verdict == verdict and report.candidate_count == count


def test_one_setting_scored_in_two_cells_counts_once() -> None:
    build = 'w = params["fc2.weight"] * 0.9\nh = np.maximum(xr.reshape(len(xr), -1) / 255.0 @ params["fc1.weight"].T, 0)\n'
    score = 'print(((h @ w.T).argmax(1) == yr).mean())\n'
    report = audit(build, score, score)
    assert report.candidate_count == 1


def test_setting_edited_again_in_a_later_cell_is_a_new_candidate() -> None:
    first = 'w = params["fc2.weight"] * 0.9\nh = np.maximum(xr.reshape(len(xr), -1) / 255.0 @ params["fc1.weight"].T, 0)\n'
    score = 'print(((h @ w.T).argmax(1) == yr).mean())\n'
    report = audit(first, score, "w[3] += 0.1\n", score)
    assert report.candidate_count == 2


def test_scoring_against_base_predictions_is_not_counted() -> None:
    report = audit(SCALE.format(scales=tuple(0.1 * k for k in range(1, 13))).replace("== yr", "== pr"))
    assert report.verdict == R.CLEAN and report.candidate_count == 0
