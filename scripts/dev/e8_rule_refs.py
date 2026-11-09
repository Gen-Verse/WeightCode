"""PREREG E8 references, rule-generated rows: a gradient fit on rows written from OP_NEW0's rule, no example of it.

The knowledge-equivalent fits of E8 (PREREG entry "E8, a rule given as knowledge, no example of it", section
References and its amendment's "Added reference"). The information is that of an E8 session: the rule, the old
operators' rules that model_spec.json states, and the 64 old-operator replay rows of the instance, no V_evidence row.
Two fit sets:

* full rows (``rulegen_*``): 2064 rows,
  * 1000 rows of the new operator alone, n 4 to 12 (``seqops.new_single_rows(13, (4, 12))``);
  * 1000 rows of the new operator composed with each old operator in either slot, n 4 to 8
    (``seqops.new_composed_rows(13, (4, 8))``);
  * the instance's 64 replay rows (the second half of the seqskill evidence, after its 64 new-operator rows).
* mixed rows (``rulegen_mixed_*``): 4064 rows, the 2000 generated new-operator rows above, 2000 old-operator rows from
  ``seqops.OLD_ROWS`` (the distribution of the base's old-operator training rows: every old operator, one or two, n 4
  to 12, two-operator share 0.5) on their own generator ``default_rng(OLD_ROW_STREAM)``, and the 64 replay rows.

The new-operator rows come from one generator, ``default_rng(ROW_STREAM)``, single rows first; their hash is asserted
equal to that of the full-row runs (358c642). Every draw goes through ``seqops.draw_unique`` with every key of the
condition's V_select and V_check (all parts) excluded, so a generated row equal to a held-out check or readout row is
dropped and redrawn. The sealed test is never read. Answers are ``apply_ops`` with the old rules and the condition
rule, whose table is read from the condition's dataset.json and checked against ``draw_new_rule(0)``. Every fit row is
then checked against an independent evaluation of the rules as model_spec.json states them (its token meanings and
row layout, parsed here, never ``seqops``), with OP_NEW0 stated in the E8 form; any mismatch stops the script. The
generator ignores the 64 V_evidence new-operator rows that ``execute_run`` hands it; only the replay part of the
evidence enters the fit.

Recipes, each the E5 recipe of ``weightcode.baselines.fit.seqgrad`` (mean token cross-entropy, full-batch Adam, lr
{1e-3, 1e-2} x steps {10, 30, 100, 300}, run seed 0, 8 candidates, each checked once by the seqskill joint oracle):

* ``*_grad_full``: every tensor, edit limits ignored (seq_grad_full on the fit rows).
* ``*_sparse19_sal``: every tensor, a fixed top 19% of each tensor's entries by |d loss / d w| at the base weights on
  the fit rows, each checkpoint scaled to the norm limit (seq_grad_sparse19_sal of ``scripts/dev/e5_diag_sparse.py``,
  masks on the fit rows).
* ``*_sparse19_traj``: as above, entries ranked by |w - w_base| after unconstrained Adam lr 1e-3, 300 steps on the fit
  rows.

Runs go to ``$WEIGHTCODE_RUNS/baselines/<instance>/<method>/e8ref/<seed>/``; read them with ``weightcode.evalkit.devreadout``
(own final, and ``--retention-floor -20`` for the kept rule).

    source scripts/env.sh
    python -B scripts/dev/e8_rule_refs.py --instance qa000f6cd --rows-only
    python -B scripts/dev/e8_rule_refs.py --instance qa000f6cd --methods rulegen_mixed_grad_full
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
ROW_STREAM = (20261104, 8001)
"""Seed list of the generator of the rule-generated new-operator rows (fixed in PREREG E8 by this script)."""
NEW_ROWS_SHA256 = "d97b208a6d81feba5909e2dfef8d2838d04426726859b1775c993e18d9161210"
"""params_sha256 of the 2000 generated new-operator rows of the full-row runs (358c642, PREREG E8 amendment)."""
OLD_ROW_STREAM = (20261104, 8002)
"""Seed list of the generator of the 2000 old-operator rows of the mixed fit set (fixed by this script)."""
NEW_RULE_INDEX = 0
N_SINGLE = 1000
SINGLE_LENGTHS = (4, 12)
N_COMPOSED = 1000
COMPOSED_LENGTHS = (4, 8)
N_OLD = 2000
EXCLUDED_SPLITS = ("V_select", "V_check")
MASK_FRACTION = 0.19
"""Entries trained per tensor by the sparse methods, as in e5_diag_sparse.py."""
TRAJ_LR = 1e-3
TRAJ_STEPS = 300
REF_LABEL = "e8ref"
RECIPES = ("grad_full", "sparse19_sal", "sparse19_traj")
FIT_PREFIX = {"full": "rulegen", "mixed": "rulegen_mixed"}
METHODS = tuple(f"{FIT_PREFIX[fit]}_{recipe}" for fit in FIT_PREFIX for recipe in RECIPES)
FIT_OF = {f"{FIT_PREFIX[fit]}_{recipe}": fit for fit in FIT_PREFIX for recipe in RECIPES}
RECIPE_OF = {f"{FIT_PREFIX[fit]}_{recipe}": recipe for fit in FIT_PREFIX for recipe in RECIPES}
RECIPE_WORDS = {
    "grad_full": "Adam on all parameters, edit limits ignored",
    "sparse19_sal": "Adam on a fixed 19% of every tensor's entries (base saliency on the fit rows), E5 program limits "
    "enforced",
    "sparse19_traj": "Adam on a fixed 19% of every tensor's entries (lr 1e-3 300-step trajectory on the fit rows), E5 "
    "program limits enforced",
}
FIT_WORDS = {
    "full": "2000 rows generated from the rule plus the 64 replay rows",
    "mixed": "2000 rows generated from the rule, 2000 old-operator rows generated from the old rules and the 64 "
    "replay rows",
}
LABELS = {
    method: (f"PREREG E8 rule-generated {'mixed-row' if FIT_OF[method] == 'mixed' else 'full-row'} reference: "
             f"{RECIPE_WORDS[RECIPE_OF[method]]}, fit on {FIT_WORDS[FIT_OF[method]]}; not training-free")
    for method in METHODS
}
PREREG_REFS = {
    "full": "E8, a rule given as knowledge, no example of it: References, rule-generated full rows",
    "mixed": "E8, a rule given as knowledge, no example of it: Amendment, Added reference, rule-generated mixed rows",
}

# ---------------------------------------------------------------------------------------------------------------
# The rules as model_spec.json states them, evaluated without seqops
# ---------------------------------------------------------------------------------------------------------------

LAYOUT_ANSWER = "the answer is OPB(OPA(x)) (OPA applied first)"
NONE_MEANING = "the row applies its first operator alone"
POSITION_PREFIX = "operator, position map: out[k] = "
SYMBOL_PREFIX = "operator, symbol map: s -> "
TABLE_SEPARATOR = ", table = "
IGNORE_TARGET = -1
_FORMULA_NODES = (ast.Expression, ast.BinOp, ast.Add, ast.Sub, ast.Mod, ast.BitXor, ast.Compare, ast.Lt, ast.IfExp,
                  ast.Name, ast.Load, ast.Constant, ast.Subscript)

Operator = Callable[[list[int]], list[int]]


def _compile_formula(text: str, names: frozenset[str]) -> Any:
    """A stated formula ("in[n - 1 - k]", "(s + 1) mod 16", ...) as code over ``names``; anything else is refused."""
    python = re.sub(r"\bin\[", "x[", text.replace(" mod ", " % ").replace(" xor ", " ^ "))
    tree = ast.parse(python, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _FORMULA_NODES):
            raise ValueError(f"formula {text!r}: {type(node).__name__} is not an allowed construct")
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError(f"formula {text!r}: unknown name {node.id!r}")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, int)):
            raise ValueError(f"formula {text!r}: constant {node.value!r} is not an integer")
    return compile(tree, "<model_spec meaning>", "eval")


def stated_operator(meaning: str) -> Operator:
    """The operator a token meaning states: a position map "out[k] = <formula of in, k, n>" or a symbol map
    "s -> <formula of s>" with an optional ", table = [..]"."""
    if meaning.startswith(POSITION_PREFIX):
        code = _compile_formula(meaning[len(POSITION_PREFIX):], frozenset({"x", "k", "n"}))

        def position_map(x: list[int]) -> list[int]:
            return [int(eval(code, {"__builtins__": {}}, {"x": x, "k": k, "n": len(x)})) for k in range(len(x))]

        return position_map
    if meaning.startswith(SYMBOL_PREFIX):
        formula, _, table_text = meaning[len(SYMBOL_PREFIX):].partition(TABLE_SEPARATOR)
        table = [int(v) for v in json.loads(table_text)] if table_text else None
        code = _compile_formula(formula, frozenset({"s", "table"} if table is not None else {"s"}))

        def symbol_map(x: list[int]) -> list[int]:
            return [int(eval(code, {"__builtins__": {}}, {"s": s, "table": table})) for s in x]

        return symbol_map
    raise ValueError(f"meaning {meaning!r} states no operator")


def check_rows_as_stated(model_spec: Mapping[str, Any], x: Sequence[Sequence[int]], y: Sequence[Sequence[int]],
                         extra_meanings: Mapping[str, str]) -> dict[str, Any]:
    """Recompute every row's targets from model_spec.json's stated layout and token meanings; raise on any mismatch.

    ``extra_meanings`` states operators by token name in the same form (OP_NEW0's rule in the E8 form). Rows are
    decoded by the stated token names (BOS, operator, operator or NONE, symbols s<k>, SEP, one OUT per symbol, PAD),
    the answer is OPB(OPA(x)) as the layout states, and the targets must hold it at the OUT positions and -1 elsewhere.
    """
    layout = str(model_spec["input"]["layout"])
    if LAYOUT_ANSWER not in layout:
        raise RuntimeError(f"model_spec.json layout {layout!r} does not state {LAYOUT_ANSWER!r}")
    tokens = {str(t["name"]): (int(t["id"]), str(t["meaning"])) for t in model_spec["input"]["tokens"]}
    if NONE_MEANING not in tokens["NONE"][1]:
        raise RuntimeError(f"model_spec.json NONE meaning {tokens['NONE'][1]!r} does not state {NONE_MEANING!r}")
    special = {name: tokens[name][0] for name in ("PAD", "BOS", "SEP", "OUT", "NONE")}
    symbol_of = {tid: int(m.group(1)) for name, (tid, _) in tokens.items() if (m := re.fullmatch(r"s(\d+)", name))}
    operators: dict[int, Operator] = {}
    stated: list[str] = []
    for name, (tid, meaning) in tokens.items():
        text = extra_meanings.get(name, meaning)
        if text.startswith((POSITION_PREFIX, SYMBOL_PREFIX)):
            operators[tid] = stated_operator(text)
            stated.append(name)
    unknown_extra = sorted(set(extra_meanings) - set(tokens))
    if unknown_extra:
        raise RuntimeError(f"extra meanings for tokens model_spec.json does not have: {unknown_extra}")
    rows_by_pair: dict[str, int] = {}
    for index, (row, target) in enumerate(zip(x, y, strict=True)):
        row, target = [int(v) for v in row], [int(v) for v in target]
        if row[0] != special["BOS"] or row.count(special["SEP"]) != 1:
            raise RuntimeError(f"row {index}: not BOS-first with one SEP")
        sep = row.index(special["SEP"])
        symbols = [symbol_of[t] for t in row[3:sep]]
        n = len(symbols)
        if row[sep + 1:sep + 1 + n] != [special["OUT"]] * n or any(t != special["PAD"] for t in row[sep + 1 + n:]):
            raise RuntimeError(f"row {index}: not {n} OUT slots then PAD")
        opa, opb = row[1], row[2]
        if opa not in operators or (opb != special["NONE"] and opb not in operators):
            raise RuntimeError(f"row {index}: operator {opa} or {opb} has no stated rule")
        answer = operators[opa](symbols)
        if opb != special["NONE"]:
            answer = operators[opb](answer)
        expected = [IGNORE_TARGET] * len(row)
        expected[sep + 1:sep + 1 + n] = answer
        if target != expected:
            raise RuntimeError(f"row {index}: targets {target} differ from the stated rules' {expected}")
        pair = f"{opa}+{opb}"
        rows_by_pair[pair] = rows_by_pair.get(pair, 0) + 1
    return {"n_rows": len(x), "operators_stated": sorted(stated), "n_operator_pairs": len(rows_by_pair),
            "mismatches": 0}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", required=True, help="seqskill instance id q<8 hex>")
    parser.add_argument("--methods", nargs="+", choices=METHODS, help="methods to run (required unless --rows-only)")
    parser.add_argument("--seed", type=int, default=0, help="run seed (torch seed; directory name)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for torch and BLAS")
    parser.add_argument("--rows-only", action="store_true",
                        help="build and check the fit rows, print their counts and sha256, run nothing")
    args = parser.parse_args(argv)
    if not args.rows_only and not args.methods:
        parser.error("--methods is required unless --rows-only")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for variable in THREAD_VARIABLES:  # before numpy is imported, or BLAS ignores it
        os.environ[variable] = str(args.threads)
    sys.dont_write_bytecode = True
    import numpy as np

    from weightcode.common.identity import params_sha256
    from weightcode.data import seqops as S
    from weightcode.data.store import dataset_json_path, read_dataset_json
    from weightcode.evalkit.seqskill import load_seqskill_instance, seqskill_program_limits
    from weightcode.evalkit.targets import file_sha256

    instance = load_seqskill_instance(args.instance)
    private = instance.private
    params = instance.params
    new_id = int(private["new_operator_id"])
    condition = str(private["condition"])

    # The condition rule, from the condition record, checked against its generator and the instance.
    rule_path = dataset_json_path(S.DATASET, condition)
    rule_file_sha = file_sha256(rule_path)
    if rule_file_sha != private["condition_dataset_json_sha256"]:
        raise RuntimeError(f"{rule_path}: sha256 {rule_file_sha} != private.json condition_dataset_json_sha256")
    recorded = read_dataset_json(S.DATASET, condition)["condition"]
    table = tuple(int(v) for v in recorded["rule"]["table"])
    drawn = S.draw_new_rule(NEW_RULE_INDEX)
    if (int(recorded["operator_id"]) != new_id or drawn.token != new_id or recorded["rule"]["kind"] != "symbol"
            or recorded["rule"]["token"] != S.TOKENS[new_id] or table != drawn.table):
        raise RuntimeError(f"condition rule {recorded['rule']} differs from draw_new_rule({NEW_RULE_INDEX}) or from "
                           f"the instance's operator {S.TOKENS[new_id]}")
    rules = {**S.old_rules(), new_id: S.Rule(new_id, "symbol", table=table)}

    # Excluded keys: every row of the condition's V_select and V_check, file hashes and stored keys checked.
    excluded_record: dict[str, Any] = {}
    excluded_parts = []
    for split in EXCLUDED_SPLITS:
        tag = {"V_select": "vselect", "V_check": "vcheck"}[split]
        path = private[f"{tag}_path"]
        digest = file_sha256(path)
        if digest != private[f"{tag}_sha256"]:
            raise RuntimeError(f"{path}: sha256 {digest} != private.json {tag}_sha256")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
        if params_sha256(arrays) != private[f"{tag}_content_sha256"]:
            raise RuntimeError(f"{path}: content hash differs from private.json {tag}_content_sha256")
        if not np.array_equal(S.row_key(*S.decode_x(arrays["x"])), arrays["idx"]):
            raise RuntimeError(f"{path}: stored row keys differ from the keys of its token rows")
        excluded_parts.append({"idx": arrays["idx"]})
        excluded_record[split] = {"path": str(path), "sha256": digest, "n_rows": int(len(arrays["idx"]))}
    exclude = S.keys_of(excluded_parts)
    excluded = {"splits": excluded_record, "n_keys": int(exclude.size),
                "rule": "every generated key differs from every key of these splits (all parts); the sealed test is "
                "not read"}

    def draw_counted(spec: S.RowSpec, size: int, rng: np.random.Generator,
                     seen: np.ndarray) -> tuple[tuple[np.ndarray, ...], dict[str, int]]:
        """``S.draw_unique`` and, from a copy of the generator, how many sampled rows it refused and why.

        ``refused_excluded``: rows whose key is in ``seen`` (a V_select or V_check row); ``refused_repeat``: rows
        repeating a key drawn before them in this draw. Rows sampled after the last kept one are not counted.
        """
        mirror = np.random.Generator(type(rng.bit_generator)())
        mirror.bit_generator.state = rng.bit_generator.state
        drawn_rows = S.draw_unique(spec, size, rng, seen)
        counts = {"rounds": 0, "refused_excluded": 0, "refused_repeat": 0}
        held = np.asarray(seen, dtype=np.int64)
        have, kept = 0, []
        while have < size:
            counts["rounds"] += 1
            sym, n, opa, opb = spec.sample(mirror, size)
            key = S.row_key(sym, n, opa, opb)
            _, first = np.unique(key, return_index=True)
            fresh = np.zeros(size, dtype=bool)
            fresh[first] = True
            fresh &= ~np.isin(key, held)
            order = np.flatnonzero(fresh)[: size - have]
            cut = int(order[-1]) + 1 if have + order.size == size else size
            refused = ~fresh[:cut]
            in_seen = np.isin(key[:cut], seen)
            counts["refused_excluded"] += int(np.count_nonzero(refused & in_seen))
            counts["refused_repeat"] += int(np.count_nonzero(refused & ~in_seen))
            kept.append(key[order])
            held = np.union1d(held, key[order])
            have += order.size
        same_state = mirror.bit_generator.state == rng.bit_generator.state
        if not (np.array_equal(np.concatenate(kept), drawn_rows[4]) and same_state):
            raise RuntimeError("the refusal count did not reproduce draw_unique")
        return drawn_rows, counts

    def encoded(parts: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
        sym, n, opa, opb, key = parts
        if np.unique(key).size != key.size or np.isin(key, exclude).any():
            raise RuntimeError("generated rows repeat a key or hold an excluded key")
        x, y = S.encode_rows(sym, n, opa, opb, S.apply_ops(sym, n, opa, opb, rules))
        if not np.array_equal(S.row_key(*S.decode_x(x)), key):
            raise RuntimeError("generated token rows do not decode to their keys")
        return x, y

    def length_counts(lengths: np.ndarray) -> dict[str, int]:
        values, counts = np.unique(lengths, return_counts=True)
        return {str(int(v)): int(c) for v, c in zip(values, counts, strict=True)}

    def pair_counts(opa: np.ndarray, opb: np.ndarray) -> dict[str, int]:
        pairs = [f"{S.TOKENS[int(a)]}+{S.TOKENS[int(b)]}" for a, b in zip(opa, opb, strict=True)]
        names, counts = np.unique(np.asarray(pairs), return_counts=True)
        return {str(p): int(c) for p, c in zip(names, counts, strict=True)}

    # New-operator rows: one generator, single rows first; the hash of the full-row runs is asserted.
    rng = np.random.default_rng(list(ROW_STREAM))
    single, single_counts = draw_counted(S.new_single_rows(new_id, SINGLE_LENGTHS), N_SINGLE, rng, exclude)
    composed, composed_counts = draw_counted(S.new_composed_rows(new_id, COMPOSED_LENGTHS), N_COMPOSED, rng,
                                             np.union1d(exclude, single[4]))
    new_parts = tuple(np.concatenate([a, b]) for a, b in zip(single, composed, strict=True))
    x_gen, y_gen = encoded(new_parts)
    new_sha = params_sha256({"x": x_gen, "y": y_gen})
    if new_sha != NEW_ROWS_SHA256:
        raise RuntimeError(f"generated new-operator rows hash {new_sha}, the full-row runs' rows {NEW_ROWS_SHA256}")

    # Old-operator rows of the mixed set: the base's old-operator training distribution, own generator.
    old_parts, old_counts = draw_counted(S.OLD_ROWS, N_OLD, np.random.default_rng(list(OLD_ROW_STREAM)), exclude)
    x_old, y_old = encoded(old_parts)
    if np.isin(old_parts[2:4], S.NEW_OP_IDS).any():
        raise RuntimeError("a generated old-operator row holds a reserved operator")

    # The replay rows: the old-operator half of the instance's evidence, labels checked against the old rules.
    x_rep = np.ascontiguousarray(instance.evidence["x_old"], dtype=np.int64)
    y_rep = np.ascontiguousarray(instance.evidence["y_old"], dtype=np.int64)
    if len(x_rep) != int(private["n_replay"]):
        raise RuntimeError(f"{len(x_rep)} replay rows, private.json n_replay says {private['n_replay']}")
    rep_parts = S.decode_x(x_rep)
    if np.isin(rep_parts[2:], S.NEW_OP_IDS).any():
        raise RuntimeError("a replay row holds a reserved operator")
    x_chk, y_chk = S.encode_rows(*rep_parts, S.apply_ops(*rep_parts, S.old_rules()))
    if not (np.array_equal(x_chk, x_rep) and np.array_equal(y_chk, y_rep)):
        raise RuntimeError("replay rows differ from the old rules' answers")
    old_in_replay = int(np.count_nonzero(np.isin(old_parts[4], S.row_key(*rep_parts))))

    # Every fit row against the rules as model_spec.json states them, OP_NEW0 in the E8 form.
    spec_path = Path(instance.instance_dir) / "model_spec.json"
    spec_sha = file_sha256(spec_path)
    if spec_sha != private["model_spec_sha256"]:
        raise RuntimeError(f"{spec_path}: sha256 {spec_sha} != private.json model_spec_sha256")
    model_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    new_meaning = f"{SYMBOL_PREFIX}table[s]{TABLE_SEPARATOR}{json.dumps(list(table))}"
    x_all = np.concatenate([x_gen, x_old, x_rep])
    y_all = np.concatenate([y_gen, y_old, y_rep])
    stated_check = check_rows_as_stated(model_spec, x_all.tolist(), y_all.tolist(), {S.TOKENS[new_id]: new_meaning})
    stated_check.update({"rows": f"the {len(x_gen)} new-operator, {len(x_old)} old-operator and {len(x_rep)} replay "
                         "rows", "model_spec": str(spec_path), "model_spec_sha256": spec_sha,
                         "new_operator_meaning": new_meaning,
                         "evaluator": "check_rows_as_stated in this script: stated layout and token meanings parsed "
                         "from model_spec.json, no seqops function"})

    new_block = {
        "rule": {"token": S.TOKENS[new_id], "kind": "symbol", "table": list(table), "source": str(rule_path),
                 "source_sha256": rule_file_sha, "equals_draw_new_rule": NEW_RULE_INDEX},
        "row_stream": list(ROW_STREAM),
        "draw": "seqops.draw_unique from one generator, single rows first, then composed rows",
        "single": {"spec": S.new_single_rows(new_id, SINGLE_LENGTHS).as_dict(), "n": N_SINGLE,
                   "by_length": length_counts(single[1]), "redraw": single_counts,
                   "sha256": params_sha256({"x": x_gen[:N_SINGLE], "y": y_gen[:N_SINGLE]})},
        "composed": {"spec": S.new_composed_rows(new_id, COMPOSED_LENGTHS).as_dict(), "n": N_COMPOSED,
                     "by_length": length_counts(composed[1]), "by_pair": pair_counts(composed[2], composed[3]),
                     "redraw": composed_counts,
                     "sha256": params_sha256({"x": x_gen[N_SINGLE:], "y": y_gen[N_SINGLE:]})},
        "generated_sha256": new_sha,
        "generated_sha256_rule": "weightcode.common.identity.params_sha256 of {'x': x [2000, SEQ_LEN], 'y': y [2000, "
        "SEQ_LEN]} int64, single rows then composed rows",
    }
    replay_block = {"n": int(len(x_rep)), "sha256": params_sha256({"x_old": instance.evidence["x_old"],
                                                                   "y_old": instance.evidence["y_old"]})}
    fit_sets: dict[str, dict[str, Any]] = {
        "full": {
            "x": np.concatenate([x_gen, x_rep]), "y": np.concatenate([y_gen, y_rep]),
            "words": (f"{len(x_gen) + len(x_rep)} fit rows ({N_SINGLE} generated from the rule alone at n "
                      f"{SINGLE_LENGTHS[0]} to {SINGLE_LENGTHS[1]}, {N_COMPOSED} generated composed at n "
                      f"{COMPOSED_LENGTHS[0]} to {COMPOSED_LENGTHS[1]}, {len(x_rep)} replay)"),
            "retention": f"retention is fitted on the {len(x_rep)} replay rows",
            "record": {
                "source": "rows generated from the condition rule plus the instance's replay rows; the 64 V_evidence "
                "new-operator rows of the evidence are not in the fit",
                **new_block,
                "order": "generated single rows, generated composed rows, replay rows",
            },
        },
        "mixed": {
            "x": np.concatenate([x_gen, x_old, x_rep]), "y": np.concatenate([y_gen, y_old, y_rep]),
            "words": (f"{len(x_gen) + len(x_old) + len(x_rep)} fit rows ({N_SINGLE} generated from the rule alone at n "
                      f"{SINGLE_LENGTHS[0]} to {SINGLE_LENGTHS[1]}, {N_COMPOSED} generated composed at n "
                      f"{COMPOSED_LENGTHS[0]} to {COMPOSED_LENGTHS[1]}, {N_OLD} generated old-operator rows, "
                      f"{len(x_rep)} replay)"),
            "retention": (f"retention is fitted on the {N_OLD} old-operator rows generated from the stated old rules "
                          f"and the {len(x_rep)} replay rows"),
            "record": {
                "source": "rows generated from the condition rule, old-operator rows generated from the old rules "
                "model_spec.json states, and the instance's replay rows; the 64 V_evidence new-operator rows of the "
                "evidence are not in the fit",
                **new_block,
                "old": {"spec": S.OLD_ROWS.as_dict(), "generator": "seqops.OLD_ROWS through seqops.draw_unique, the "
                        "distribution of the base's old-operator training rows", "row_stream": list(OLD_ROW_STREAM),
                        "n": N_OLD, "by_length": length_counts(old_parts[1]),
                        "n_two_operator": int(np.count_nonzero(old_parts[3] != S.NONE)),
                        "by_pair": pair_counts(old_parts[2], old_parts[3]), "redraw": old_counts,
                        "n_equal_to_a_replay_row": old_in_replay,
                        "sha256": params_sha256({"x": x_old, "y": y_old})},
                "order": "generated single rows, generated composed rows, generated old-operator rows, replay rows",
            },
        },
    }
    for name, fit in fit_sets.items():
        fit["record"].update({
            "evidence_new_rows_in_fit": False, "excluded": excluded, "replay": replay_block,
            "stated_rule_check": stated_check, "n_fit": int(len(fit["x"])),
            "fit_sha256": params_sha256({"x": fit["x"], "y": fit["y"]}),
            "fit_sha256_rule": f"params_sha256 of {{'x', 'y'}} [{len(fit['x'])}, SEQ_LEN] int64 in the order above",
        })
    print(f"new rows: {N_SINGLE} single (by length {new_block['single']['by_length']}, redraw {single_counts}), "
          f"{N_COMPOSED} composed (by length {new_block['composed']['by_length']}, redraw {composed_counts}); "
          f"sha256 {new_sha}", flush=True)
    old_record = fit_sets["mixed"]["record"]["old"]
    print(f"old rows: {N_OLD} (by length {old_record['by_length']}, two-operator {old_record['n_two_operator']}, "
          f"redraw {old_counts}, equal to a replay row {old_in_replay}); sha256 {old_record['sha256']}", flush=True)
    print(f"stated-rule check: {stated_check['n_rows']} rows, operators {stated_check['operators_stated']}, "
          f"{stated_check['n_operator_pairs']} operator pairs, 0 mismatches", flush=True)
    for name, fit in fit_sets.items():
        print(f"fit set {name}: {fit['record']['n_fit']} rows, fit sha256 {fit['record']['fit_sha256']}", flush=True)
    if args.rows_only:
        return 0

    import torch

    torch.set_num_threads(args.threads)
    from weightcode.baselines.fit.edits import scale_to_norm_limit
    from weightcode.baselines.fit.grad import LEARNING_RATES, STEP_CHECKPOINTS
    from weightcode.baselines.fit.runner import CandidatePatch, Evidence, MethodSpec, baseline_run_dir, execute_run
    from weightcode.baselines.fit.seqgrad import adam_checkpoints, rebuild, seq_grad_config, seq_parity, token_loss

    limits = seqskill_program_limits(instance.task["limits"], len(params))
    trainable = sorted(params)

    def top_fraction(score: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        masks = {}
        for name in trainable:
            flat = np.abs(score[name]).ravel()
            k = int(np.floor(MASK_FRACTION * flat.size))
            if k < 1:
                raise RuntimeError(f"{name}: {flat.size} entries give an empty {MASK_FRACTION:g} mask")
            order = np.argsort(-flat, kind="stable")
            mask = np.zeros(flat.size, dtype=bool)
            mask[order[:k]] = True
            masks[name] = mask.reshape(score[name].shape)
        return masks

    def make_generator(method: str):
        recipe = RECIPE_OF[method]
        sparse = recipe != "grad_full"
        fit = fit_sets[FIT_OF[method]]
        x_fit = fit["x"]
        x_t = torch.from_numpy(np.ascontiguousarray(x_fit, dtype=np.int64))
        y_t = torch.from_numpy(np.ascontiguousarray(fit["y"], dtype=np.int64))

        def generate(evidence: Evidence):
            k = evidence.n_deployment
            if (evidence.n_replay != len(x_rep) or not np.array_equal(evidence.x[k:], x_rep)
                    or not np.array_equal(evidence.y[k:], y_rep)):
                raise RuntimeError("the evidence's replay part is not the instance's replay rows")
            model = rebuild(params)
            parity, parity_bound = seq_parity(model, params, instance.forward, x_fit)
            masks = None
            info: dict[str, Any] = {"model_config": model.config.as_dict(), "parity_max_abs_logit_diff": parity,
                                    "parity_bound": parity_bound, "parity_rows": "the fit rows",
                                    "evidence_new_rows_in_fit": False, "n_fit": int(len(x_fit)),
                                    "fit_set": FIT_OF[method], "fit_sha256": fit["record"]["fit_sha256"],
                                    "fit_generated_sha256": new_sha}
            if sparse:
                if recipe == "sparse19_sal":
                    for tensor in model.parameters():
                        tensor.requires_grad_(True)
                    token_loss(model, x_t, y_t).backward()
                    score = {name: p.grad.detach().numpy().copy() for name, p in model.named_parameters()}
                else:
                    (_, traj, _), = adam_checkpoints(params, x_t, y_t, trainable=trainable, lr=TRAJ_LR,
                                                     checkpoints=(TRAJ_STEPS,), seed=args.seed)
                    score = {name: traj[name].astype(np.float64) - params[name].astype(np.float64)
                             for name in trainable}
                if set(score) != set(trainable):
                    raise RuntimeError(f"score keys differ from the student's: {sorted(set(score) ^ set(trainable))}")
                masks = top_fraction(score)
                n_trainable = int(sum(m.sum() for m in masks.values()))
                info.update({"mask_rule": method, "mask_fraction_per_tensor": MASK_FRACTION,
                             "mask_rows": "the fit rows"})
            else:
                n_trainable = int(sum(params[name].size for name in trainable))
                info["trainable"] = trainable
            n_params = int(sum(v.size for v in params.values()))
            info.update({"n_trainable_entries": n_trainable, "n_params": n_params,
                         "trainable_frac": n_trainable / n_params})
            with torch.no_grad():
                info["base_fit_loss"] = float(token_loss(rebuild(params), x_t, y_t))
            scope = (f"every tensor, top {MASK_FRACTION:g} of each tensor's entries by the {method} mask" if sparse
                     else "all tensors, no edit limits")
            candidates = []
            for lr in LEARNING_RATES:
                for steps, child, loss in adam_checkpoints(params, x_t, y_t, trainable=trainable, lr=lr,
                                                           checkpoints=STEP_CHECKPOINTS, seed=args.seed, masks=masks):
                    notes: dict[str, Any] = {"fit_loss_at_checkpoint": loss}
                    if sparse:
                        for name in trainable:
                            if not np.array_equal(child[name][~masks[name]], params[name][~masks[name]]):
                                raise RuntimeError(f"{name} changed outside its mask")
                        child, scaling = scale_to_norm_limit(params, child, trainable, limit=limits.max_rel_frob,
                                                             min_ndim=limits.rel_frob_min_ndim,
                                                             exempt=limits.rel_frob_exempt)
                        notes["norm_scaled_tensors"] = [name for name, entry in scaling.items() if entry["scaled"]]
                        if notes["norm_scaled_tensors"]:
                            with torch.no_grad():
                                notes["fit_loss_after_scaling"] = float(token_loss(rebuild(child), x_t, y_t))
                    candidates.append(CandidatePatch(
                        slot=len(candidates), child=child, config={"lr": lr, "steps": steps},
                        hypothesis=(f"{method}: Adam lr={lr:g} for {steps} full-batch steps on the mean token "
                                    f"cross-entropy over the {fit['words']}; trained: {scope}"),
                        predicted_effect=(f"rows of {S.TOKENS[new_id]} answered right, alone and composed, if a fit "
                                          f"on rows written from the rule generalizes; {fit['retention']}"),
                        notes=notes))
            if sparse:
                info["n_candidates_norm_scaled"] = sum(1 for c in candidates if c.notes["norm_scaled_tensors"])
            return candidates, info

        return generate

    for method in args.methods:
        sparse = RECIPE_OF[method] != "grad_full"
        fit = fit_sets[FIT_OF[method]]
        spec = MethodSpec(method=method, label=LABELS[method], uses_labels_for_fit=True, edit_limits_enforced=sparse)
        config = seq_grad_config("seq_grad_full", args.seed, instance)
        config.update({
            "loss": f"mean cross-entropy over the labelled (OUT) positions of the {fit['words']}",
            "trained": (f"every tensor, top {MASK_FRACTION:g} of its entries by the {method} mask" if sparse
                        else "all tensors"),
            "norm_limit": limits.max_rel_frob if sparse else None,
            "mask_fraction_per_tensor": MASK_FRACTION if sparse else None,
            "traj_mask": {"lr": TRAJ_LR, "steps": TRAJ_STEPS} if method.endswith("traj") else None,
            "fit_rows": fit["record"],
            "prereg": PREREG_REFS[FIT_OF[method]],
        })
        run_dir = baseline_run_dir(instance.instance_id, method, args.seed, REF_LABEL)
        record = execute_run(instance=instance, spec=spec, run_dir=run_dir, run_seed=args.seed, config=config,
                             generate=make_generator(method), require_valid_candidate=True, seqskill=True)
        print(method, run_dir / "run.json", record.get("final", {}).get("patch_id"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
