"""Helpers shared by ``scripts/run_fit_baselines.py`` and ``scripts/run_random_controls.py``.

Instance selection: explicit ids, or every row of ``$WEIGHTCODE_PRIVATE/instances/index.json`` with the requested
split_role (default "dev") and, optionally, dataset. Every selected instance's private.json split_role is checked
before any run starts; a main instance is refused unless --allow-main is given (PREREG section 2). Fewshot selection
(PREREG E2) reads ``$WEIGHTCODE_PRIVATE/fewshot_instances/index.json`` instead; those instances are dev only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.common.paths import private_dir
from weightcode.evalkit.fewshot import active_fewshot_ids, load_fewshot_instance
from weightcode.evalkit.instances import read_index

from .bnrefs import BN_ADAPT_PRIORS, BN_ADAPT_PRIORS_T04_EDGE, BN_REF_METHODS, run_bn_ref
from .grad import GRAD_METHODS, GRID2_METHODS, GRID_E1, GRID_E11_SECOND, GRID_T04_EDGE, run_grad
from .imprint import IMPRINT_SPECS, run_imprint
from .lnrefs import LN_REF_METHODS, run_ln_ref
from .ridge import RIDGE_METHODS, run_ridge
from .rules import run_input_stats, run_prior_shift, run_tie
from .runner import SplitRoleError, check_split_role, safe_name

IMPRINT_METHODS: tuple[str, ...] = tuple(IMPRINT_SPECS)
"""``imprint`` and ``imprint_retain``: they add classes, so they run on fewshot-v1 instances only."""
RULE_METHODS: tuple[str, ...] = ("prior_shift", "tie_inputs", "input_stats")
FIT_METHODS: tuple[str, ...] = (*GRAD_METHODS, *RIDGE_METHODS, *IMPRINT_METHODS, *RULE_METHODS, *BN_REF_METHODS,
                                *LN_REF_METHODS)
"""Methods of ``scripts/run_fit_baselines.py``: the fitting references, then the fixed-rule arms."""

FEWSHOT_METHODS: tuple[str, ...] = ("imprint", "ridge_margin", "grad_last", "grad_full")
"""The four E2 references in PREREG order: imprinting/NCM with one bias offset, ridge on the penultimate features,
last-layer and full fine-tuning with replay. The E2 gate is the best of them, within the limits or not."""

FEWSHOT_SUPPLEMENTARY: tuple[str, ...] = ("imprint_retain",)
"""References of the PREREG E2 equal-retention supplement only, never of the gate; not run by default."""

FEWSHOT_CAPABLE: tuple[str, ...] = (*GRAD_METHODS, *RIDGE_METHODS, *IMPRINT_METHODS)
"""Methods that run on a fewshot-v1 instance; the fixed rules are defined on evidence-v1 rows only."""

E10_METHODS: tuple[str, ...] = ("grad_last", "grad_full", "ridge_last", "prior_shift", "tie_inputs", "input_stats",
                                 "grad_sparse19_sal", "grad_sparse19_traj", "interp", "bn_adapt", "tent")
"""The E10 references in PREREG order: E1's six, E5's two masks and E7's interp, the label-free BatchNorm baselines."""

E10_ONLY: tuple[str, ...] = BN_REF_METHODS
"""bn_adapt and tent need a BatchNorm student; they run with e10 only."""

E11_DATASET = "cifar10_vl7"
"""PREREG E11's dataset (``weightcode.data.e11.DATASET``): E10's conditions with a ViT-Lite-7/4 student."""

E11_METHODS: tuple[str, ...] = ("grad_last", "grad_full", "ridge_last", "prior_shift", "tie_inputs", "input_stats",
                                 "grad_sparse19_sal", "grad_sparse19_traj", "interp", "tent_ln", "sar")
"""The E11 references in PREREG order: E1's six adapted to ViT-Lite-7/4, E5's two masks and E7's interp, the
label-free LayerNorm baselines; bn_adapt and tent do not apply (no BatchNorm)."""

E11_ONLY: tuple[str, ...] = LN_REF_METHODS
"""tent_ln and sar need a LayerNorm student; they run with e11 only."""

E11_REF_LABEL = "e11"
"""Run-directory label of E11 runs: ``baselines/<instance_id>/<method>/e11/<seed>/``."""

E11_GRID2_METHODS: tuple[str, ...] = GRID2_METHODS
"""The references of PREREG E11's second grid (amended 2026-11-05): grad_last, grad_full, the two sparse19 masks and
interp, each rerun once over ``weightcode.baselines.fit.grad.GRID_E11_SECOND``."""

E11_GRID2_LABEL = "e11g2"
"""Run-directory label of the E11 second-grid runs: ``baselines/<instance_id>/<method>/e11g2/<seed>/``, apart from
the first grid's ``e11``."""

E12_DATASET = "cifar10_r18"
"""PREREG E12's dataset (``weightcode.data.e12.DATASET``): E10's conditions with the ImageNet-pretrained ResNet-18."""

E12_METHODS: tuple[str, ...] = E10_METHODS
"""The E12 references: E10's set adapted to ResNet-18 as E10 adapted E1's (PREREG E12), in E10's order. bn_adapt and
tent apply (the student has BatchNorm); tent_ln and sar do not."""

E12_REF_LABEL = "e12"
"""Run-directory label of E12 runs: ``baselines/<instance_id>/<method>/e12/<seed>/``."""

E12_GRID2_METHODS: tuple[str, ...] = GRID2_METHODS
"""The E12 second-grid runs (``GRID_E11_SECOND``): grad_last and grad_full are F1's "last layer" and "full" (PREREG
F1 read over E12); the two sparse19 masks and interp complete the grid as on E11."""

E12_GRID2_LABEL = "e12g2"
"""Run-directory label of the E12 second-grid runs: ``baselines/<instance_id>/<method>/e12g2/<seed>/``."""

T04_DATASET = "in1k_t04"
"""PREREG T04's dataset (``weightcode.data.t04.DATASET``): the native ImageNet ResNet-18 on its own ImageNet-C conditions."""

T04_METHODS: tuple[str, ...] = E12_METHODS
"""The T04 references: E12's set (E10's eleven) on the native network, in E10's order (PREREG T04)."""

T04_REF_LABEL = "t04"
"""Run-directory label of T04 runs: ``baselines/<instance_id>/<method>/t04/<seed>/``."""

T04_GRID2_METHODS: tuple[str, ...] = GRID2_METHODS
"""The T04 second-grid runs (``GRID_E11_SECOND``), as on E12; grad_full and grad_last are F1's "full" and "last
layer" on T04."""

T04_GRID2_LABEL = "t04g2"
"""Run-directory label of the T04 second-grid runs: ``baselines/<instance_id>/<method>/t04g2/<seed>/``."""

T04_EDGE_METHODS: tuple[str, ...] = (*GRID2_METHODS, "bn_adapt")
"""The fitting references the T04 edge extension can rerun (PREREG T04 amendment, main 9f0baa9): the second-grid
methods on ``GRID_T04_EDGE`` and bn_adapt on ``BN_ADAPT_PRIORS_T04_EDGE``. Which (method, condition) pairs run is
decided by scripts/t04/edge_extension.py and recorded in R37."""

T04_EDGE_LABEL = "t04g3"
"""Run-directory label of the T04 edge-extension runs: ``baselines/<instance_id>/<method>/t04g3/<seed>/``."""

E10_REF_LABEL = "e10"
"""Run-directory label of E10 runs: ``baselines/<instance_id>/<method>/e10/<seed>/``."""

JOINT_REF_LABEL = "e1joint"
"""Run-directory label of joint runs (PREREG E1): ``baselines/<instance_id>/<method>/e1joint/<seed>/``."""

FEWSHOT_REF_LABEL = "e2"
"""Run-directory label of fewshot runs (PREREG E2): ``baselines/<instance_id>/<method>/e2/<seed>/``."""

EXIT_REFUSED = 2
"""Exit status of a script that refuses its instance selection; nothing has been run or written."""


class SelectionError(ValueError):
    """The instance selection is empty or inconsistent with the other arguments."""


def load_instance_by_id(instance_id: str) -> Instance:
    """Load ``$WEIGHTCODE_PRIVATE/instances/<instance_id>`` with hash checks."""
    return load_instance(private_dir() / "instances" / safe_name(instance_id, "instance id"))


def select_instance_ids(explicit: Sequence[str] | None, *, split_role: str, dataset: str | None) -> list[str]:
    """Explicit ids if given, else the index rows with ``split_role`` (and ``dataset`` if given), in index order."""
    if explicit:
        return list(dict.fromkeys(explicit))
    rows = read_index()["instances"]
    chosen = [
        row["instance_id"]
        for row in rows
        if row["split_role"] == split_role and (dataset is None or row["dataset"] == dataset)
    ]
    if not chosen:
        raise SelectionError(f"index.json has no instance with split_role {split_role!r} and dataset {dataset!r}")
    return chosen


def load_checked_instances(instance_ids: Sequence[str], *, allow_main: bool) -> list[Instance]:
    """Load every instance and check its split_role before anything runs; raises ``SplitRoleError`` on refusal."""
    instances = [load_instance_by_id(instance_id) for instance_id in instance_ids]
    for instance in instances:
        check_split_role(instance, allow_main)
    return instances


def select_fewshot_ids(explicit: Sequence[str] | None, *, dataset: str | None) -> list[str]:
    """Explicit fewshot ids if given, else the active fewshot instances (of ``dataset`` if given), in index order."""
    if explicit:
        return list(dict.fromkeys(explicit))
    chosen = active_fewshot_ids(dataset)
    if not chosen:
        raise SelectionError(f"the fewshot index has no active instance of dataset {dataset!r}")
    return chosen


def load_checked_fewshot_instances(instance_ids: Sequence[str]) -> list[Instance]:
    """Load every fewshot instance (hash-checked) and refuse any that is not dev before anything runs."""
    instances = [load_fewshot_instance(instance_id) for instance_id in instance_ids]
    for instance in instances:
        check_split_role(instance, allow_main=False)
    return instances


def run_method(
    instance: Instance,
    method: str,
    run_dir: Path,
    *,
    run_seed: int,
    allow_main: bool,
    joint: bool = False,
    fewshot: bool = False,
    e10: bool = False,
    e11: bool = False,
    grid2: bool = False,
    e12: bool = False,
    t04: bool = False,
    edge: bool = False,
) -> dict[str, Any]:
    """Run one fitting reference or fixed-rule arm on one instance and return its run record.

    ``joint`` (PREREG E1): the label fits train on the evidence plus the replay rows, the fixed rules keep the
    deployment evidence, and all are scored by the joint oracle. ``fewshot`` (PREREG E2): a fewshot-v1 instance, one
    of ``FEWSHOT_CAPABLE``; ``imprint`` and ``imprint_retain`` run only there. ``e10`` (PREREG E10, with ``joint``):
    the run is held to the E10 program limits; ``bn_adapt`` and ``tent`` run only there. ``e11`` (PREREG E11, with
    ``e10``: E10's limits and oracle on a cifar10_vl7 instance): ``tent_ln`` and ``sar`` run only there, and
    ``bn_adapt`` and ``tent`` do not. ``e12`` (PREREG E12, with ``e10``: E10's limits and oracle on a cifar10_r18
    instance, a BatchNorm student): ``bn_adapt`` and ``tent`` run, ``tent_ln`` and ``sar`` do not. ``grid2`` (with
    ``e11``, ``e12`` or ``t04``): the second grid ``GRID_E11_SECOND``, for ``E11_GRID2_METHODS`` only. ``t04``
    (PREREG T04, with ``e10``: E10's limits and oracle on an in1k_t04 instance, the native ImageNet ResNet-18) runs
    exactly as ``e12``. ``edge`` (with ``t04``; PREREG T04's edge-extension amendment): ``GRID_T04_EDGE`` for the
    second-grid methods and ``BN_ADAPT_PRIORS_T04_EDGE`` for bn_adapt, ``T04_EDGE_METHODS`` only.
    """
    if e10 and (not joint or fewshot):
        raise ValueError("e10 runs are joint runs on evidence-v1 instances (joint=True, fewshot=False)")
    if (e11 or e12 or t04) and not e10:
        raise ValueError("e11, e12 and t04 runs are E10-limit joint runs (pass e10=True with e11, e12 or t04)")
    if int(e11) + int(e12) + int(t04) > 1:
        raise ValueError("a run is e11, e12 or t04, not two of them")
    if grid2 and not (e11 or e12 or t04):
        raise ValueError("the second grid is PREREG E11's and E12's: pass e11=True with grid2 (or e12=True with grid2)")
    if edge and (not t04 or grid2):
        raise ValueError("the edge extension is PREREG T04's: pass t04=True and edge=True without grid2")
    if edge and method not in T04_EDGE_METHODS:
        raise ValueError(f"{method} has no T04 edge-extension grid; expected one of {T04_EDGE_METHODS}")
    if grid2 and method not in E11_GRID2_METHODS:
        raise ValueError(f"{method} has no E11 second-grid run (the E12 grid is the same); expected one of "
                         f"{E11_GRID2_METHODS}")
    dataset = instance.private.get("dataset") if e10 else None
    if e11 and dataset != E11_DATASET:
        raise ValueError(f"e11 runs need a {E11_DATASET} instance, got dataset {dataset!r}")
    if e12 and dataset != E12_DATASET:
        raise ValueError(f"e12 runs need a {E12_DATASET} instance, got dataset {dataset!r}")
    if e10 and not e11 and dataset == E11_DATASET:
        raise ValueError(f"a {E11_DATASET} instance runs with e11")
    if e10 and not e12 and dataset == E12_DATASET:
        raise ValueError(f"a {E12_DATASET} instance runs with e12")
    if t04 and dataset != T04_DATASET:
        raise ValueError(f"t04 runs need a {T04_DATASET} instance, got dataset {dataset!r}")
    if e10 and not t04 and dataset == T04_DATASET:
        raise ValueError(f"a {T04_DATASET} instance runs with t04")
    if method in E11_ONLY:
        if not e11:
            raise ValueError(f"{method} is a LayerNorm reference of PREREG E11 and runs with e11 only")
        return run_ln_ref(instance, method, run_dir, run_seed=run_seed, allow_main=allow_main, e10=True)
    if method in E10_ONLY and e11:
        raise ValueError(f"{method} is a BatchNorm reference; the E11 student has no BatchNorm")
    if method in E10_ONLY:
        if not e10:
            raise ValueError(f"{method} is a BatchNorm reference of PREREG E10 and runs with e10 only")
        return run_bn_ref(instance, method, run_dir, run_seed=run_seed, allow_main=allow_main, e10=True,
                          priors=BN_ADAPT_PRIORS_T04_EDGE if edge else BN_ADAPT_PRIORS)
    if fewshot and method not in FEWSHOT_CAPABLE:
        raise ValueError(f"{method!r} does not run on fewshot instances; expected one of {FEWSHOT_CAPABLE}")
    if method in IMPRINT_METHODS:
        if not fewshot:
            raise ValueError(f"{method} adds classes and runs on fewshot instances only (fewshot=True)")
        return run_imprint(instance, run_dir, run_seed=run_seed, allow_main=allow_main, method=method)
    if method in GRAD_METHODS:
        return run_grad(
            instance, method, run_dir, run_seed=run_seed, allow_main=allow_main, joint=joint, fewshot=fewshot, e10=e10,
            grid=GRID_T04_EDGE if edge else GRID_E11_SECOND if grid2 else GRID_E1,
        )
    if method in RIDGE_METHODS:
        return run_ridge(
            instance, run_dir, method=method, run_seed=run_seed, allow_main=allow_main, joint=joint, fewshot=fewshot,
            e10=e10,
        )
    if method == "prior_shift":
        return run_prior_shift(instance, run_dir, run_seed=run_seed, allow_main=allow_main, joint=joint, e10=e10)
    if method == "tie_inputs":
        return run_tie(instance, run_dir, run_seed=run_seed, allow_main=allow_main, joint=joint, e10=e10)
    if method == "input_stats":
        return run_input_stats(instance, run_dir, run_seed=run_seed, allow_main=allow_main, joint=joint, e10=e10)
    raise ValueError(f"unknown method {method!r}; expected one of {FIT_METHODS}")


def refusal_message(error: Exception) -> str:
    """Clear stderr text for a refused selection."""
    if isinstance(error, SplitRoleError):
        return (
            f"refused: {error}\nNothing was run. Main instances are run only after the pre-registration freeze, "
            "and then only with --allow-main."
        )
    return f"refused: {error}\nNothing was run."


def summary_line(record: Mapping[str, Any]) -> str:
    """One-line human summary of a run record."""
    instance_id = record["instance"]["instance_id"]
    b1 = record.get("config", {}).get("b1")
    if b1 is not None:
        head = f"{instance_id} {record['method']}[{b1['arm']} r{b1['run_index']}]: {record['status']}"
    else:
        head = f"{instance_id} {record['method']} seed {record['run_seed']}: {record['status']}"
    if record["status"] != "complete":
        return f"{head} ({record.get('error')})"
    final = record["final"]
    generation = record.get("generation", {})
    if generation.get("reference_is_noop"):
        return f"{head}; reference is the base no-op, no control written"
    if generation.get("not_applicable"):
        return f"{head}; not applicable ({generation['not_applicable']}), nothing checked"
    per_query = [
        f"{e['score']}({e['fixed']}/{e['broken']})" + (f"[{e['score_new']}|{e['score_old']}]" if "score_new" in e else "")
        for e in record.get("oracle_log", [])
    ]
    chosen = "base no-op" if final["is_base_noop"] else final["patch_id"]
    target = "" if final.get("fixed_target") is None else f" fixed_target={final['fixed_target']}"
    delta = final.get("delta_acc")
    shown_delta = "n/a" if delta is None else f"{delta:+.4f}"
    if "score_new" in final:
        target += f" S_new={final['score_new']} S_old={final['score_old']}"
    return (
        f"{head}; final {chosen} S={final['score']} fixed={final['fixed']} broken={final['broken']}{target} "
        f"delta_acc={shown_delta} edit_size={final['edit_size']} n_changed={final['n_changed']}; "
        f"queries {record['queries_used']}/{record['budget']}, invalid {record.get('n_invalid', 0)}; "
        f"S(fixed/broken) per query {per_query}; respects_edit_limits={record['flags']['respects_edit_limits']}"
    )
