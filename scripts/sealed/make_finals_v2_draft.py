"""Write the sealed-read list draft v2 and its cell map from the approved list v1 and the store's run records.

PREREG Sealed read, list v2 (main 9c80cfe / eddd941; user decision about 16:15 Beijing on 2026-11-06): v1's 28
entries, plus every child behind a cell of the paper's main tables (Paper/manuscript/tables/table_data.json,
paper-writing 242cb7c and f0768ff): tab:main (per setting the program, the best F1 kept pick, the best F1 own pick
without the clean floor and the best kept fixed reference; new here are the E12 own picks), tab:examples (in the
supplement since f0768ff, cells kept: F1s kept picks, own picks without the clean floor), tab:reuse (the E11 C program
transfers, the F3 recipe carried unchanged, the F3 per-student re-selection) and tab:stack (the E10 teacher v2 stream
and the F4 ft_replay stream after rounds 1 to 3, each round on all four stream instances, since the stream score is the
mean over all four conditions plus the clean change).

Every child hash comes from its run record: a transfer's ``child_sha256``, a stream round's ``child_sha256``, the kept
pick of ``devreadout.retention_final`` at S_old >= -20, a run's own final, or the oracle-log entry of the recipe's
configuration. An entry whose (instance_id, child_sha256) is already on the list is not added again; its cell maps to
the entry already there. A base no-op pick gets no entry (child = base, so gain = clean = 0 on any rows).

Usage (reads $WEIGHTCODE_RUNS and $WEIGHTCODE_PRIVATE only; opens nothing under $WEIGHTCODE_SEALED)::

    .venv/bin/python scripts/sealed/make_finals_v2_draft.py \
        --v1 scripts/sealed/finals_v1.json --out scripts/sealed/finals_v2_draft.json \
        --cells scripts/sealed/finals_v2_cells.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from weightcode.common.paths import runs_dir
from weightcode.evalkit import devreadout
from weightcode.evalkit import sealed_finals as SF

READOUT = SF.READOUT_VISION_JOINT
FLOOR = -20
E11_FINALS = {"gaussian": ("E11_B_gaussian_s100", "i00802425", "gaussian_noise_s5"),
              "contrast": ("E11_B_contrast_s100", "i3eb3fc78", "contrast_s5")}
E11_SESSIONS = {"gaussian": "sc918e11dceef637a", "contrast": "s760cb52ce82fb937"}
E11_TARGETS = {"gaussian": {101: "i905213a3", 102: "iab18c0aa", 103: "ib1b195f6", 104: "ie1be7cd9", 105: "i36df1f8f"},
               "contrast": {101: "i4a486d70", 102: "i5d76f25c", 103: "id83bf155", 104: "ib1e4e794", 105: "iecf07b07"}}
F3_METHOD = {"gaussian": "lora_r4", "contrast": "ft_full_l2sp"}
F3_RECIPE = {"gaussian": {"lr": 0.001, "steps": 10}, "contrast": {"weight": 0.1, "lr": 1e-05, "steps": 300}}
"""The F3 recipes as fixed by F1 (DEV_RESULTS F1: gaussian LoRA r4 {lr 1e-3, 10 steps}, contrast L2-SP {weight 0.1,
lr 1e-5, 300 steps})."""
E12_FINALS = {"gaussian": ("E12_B_gaussian_s100", "i26d23d3e", "gaussian_noise_s5"),
              "contrast": ("E12_B_contrast_s100", "ie3840b28", "contrast_s5")}
E12_OWN = {"gaussian": "ft_full_l2sp", "contrast": "lora_r16"}
"""tab:main's best F1 own pick on ResNet-18 (WEIGHTCODE-main 2026-11-06, from the f1e12 own records: gaussian L2-SP
+31.625 / -2.375, contrast LoRA r16 +28.925 / -2.80)."""
E14 = ("E14_B_pickup_s100", "p6ce8ad67", -3)
E14_F1 = ("f1_full", "f1_last", "ft_ln", "lora_r4", "lora_r16", "ft_full_noreplay", "ft_full_lwf", "ft_full_l2sp")
STREAMS = {"program": ("E10_stream_v2", "loop_streams/loop_teacher_88d4ae0edeaf_4dacc6c42bfa.json"),
           "ft": ("ref_E10_stream_F4_ft_replay", "f4/e10/ctrlb_f4_ft_replay_b325008afc61_4dacc6c42bfa.json")}
STREAM_CONDITIONS = {"id14b4f0b": "gaussian", "id64b2abc": "defocus", "if90033fc": "snow", "i1bf0c953": "contrast"}
F1_ARM = {"F1_lora_r16": ("lora_r16", "f1", "f1_{inst}_{cond}_{which}_dev_nscc_20261105.jsonl"),
          "F1_ft_full_l2sp": ("ft_full_l2sp", "f1", "f1_{inst}_{cond}_{which}_dev_nscc_20261105.jsonl")}
for _arm in ("s1_gentle", "s3_lora_aug", "s4_est_prior", "s5_oracle_prior"):
    F1_ARM[f"F1s_{_arm}"] = (f"f1s_{_arm}", "f1s", "f1s_{inst}_{cond}_f1s_" + _arm + "_{which}_dev_nscc_20261105.jsonl")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class Draft:
    """The entries of the draft in order, with the (instance, child) pairs already listed."""

    def __init__(self, v1: Mapping[str, Any]) -> None:
        self.finals = [dict(entry) for entry in v1["finals"]]
        self.references = [dict(entry) for entry in v1["references"]]
        self.by_pair = {(e["instance_id"], e["child_sha256"]): e["name"] for e in self.finals + self.references}
        self.base_noops: list[str] = []

    def add(self, block: str, name: str, instance_id: str, child: str, source: dict[str, Any],
            pairs_with: str | None = None) -> str:
        """Add an entry, or return the name of the entry already reading this child on this instance."""
        known = self.by_pair.get((instance_id, child))
        if known is not None:
            return known
        entry = {"name": name, "readout": READOUT, "source": source, "instance_id": instance_id, "child_sha256": child}
        if block == SF.BLOCK_REFERENCES:
            entry["pairs_with"] = pairs_with
            self.references.append(entry)
        else:
            self.finals.append(entry)
        self.by_pair[(instance_id, child)] = name
        return name


def _run(instance: str, method: str, tag: str) -> tuple[str, dict[str, Any]]:
    rel = f"baselines/{instance}/{method}/{tag}/0/run.json"
    return rel, _json(runs_dir() / rel)


def kept(draft: Draft, name: str, instance: str, method: str, tag: str, dev_record: str, pairs_with: str,
         cell: str) -> str:
    rel, run = _run(instance, method, tag)
    pick = devreadout.retention_final(run, FLOOR)["final"]
    if pick["is_base_noop"]:
        draft.base_noops.append(f"{cell}: {method} on {instance} keeps the base no-op at S_old >= {FLOOR}")
        return "base no-op (gain 0, clean 0 by construction; no entry)"
    source = {"kind": SF.SOURCE_REFERENCE, "run": rel, "pick": SF.PICK_KEPT, "s_old_floor": FLOOR,
              "dev_record": dev_record}
    return draft.add(SF.BLOCK_REFERENCES, name, instance, pick["child_sha256"], source, pairs_with)


def own(draft: Draft, name: str, instance: str, method: str, tag: str, dev_record: str, pairs_with: str) -> str:
    rel, run = _run(instance, method, tag)
    final = run["final"]
    if final.get("is_base_noop"):
        raise SystemExit(f"{name}: the run's own final is the base no-op")
    source = {"kind": SF.SOURCE_REFERENCE, "run": rel, "pick": SF.PICK_OWN, "dev_record": dev_record}
    return draft.add(SF.BLOCK_REFERENCES, name, instance, final["child_sha256"], source, pairs_with)


def candidate(draft: Draft, name: str, instance: str, method: str, tag: str, config: Mapping[str, Any],
              dev_record: str, pairs_with: str) -> str:
    rel, run = _run(instance, method, tag)
    hits = [c for c in run["candidates"] if c.get("config") == dict(config)]
    if len(hits) != 1:
        raise SystemExit(f"{name}: {len(hits)} candidates of {rel} with configuration {config}")
    patch_id = hits[0]["patch_id"]
    (entry,) = [e for e in run["oracle_log"] if str(e["patch_id"]) == patch_id]
    source = {"kind": SF.SOURCE_REFERENCE, "run": rel, "pick": SF.PICK_CANDIDATE, "patch_id": patch_id,
              "dev_record": dev_record}
    return draft.add(SF.BLOCK_REFERENCES, name, instance, entry["child_sha256"], source, pairs_with)


def main_table(draft: Draft, examples_cells: Mapping[str, Any]) -> dict[str, Any]:
    """tab:main: per setting the program, the best F1 kept pick, the best F1 own pick and the best kept fixed
    reference. ViT rows are tab:examples' best-of-8 cells; ResNet-18 adds its own picks; on BabyAI every F1 method
    keeps the base no-op, as its own final and at the floor, which is checked here."""
    cells: dict[str, Any] = {}
    for cond in ("gaussian", "contrast"):
        ex = examples_cells[cond]
        cells[f"vit_{cond}"] = {"program": ex["program"], "f1_kept": ex["kept_best8"], "f1_own": ex["nobound_best8"],
                                "fixed": ex["fixed"]}
    for cond, (final_name, inst, condition) in E12_FINALS.items():
        method = E12_OWN[cond]
        cells[f"r18_{cond}"] = {
            "program": final_name,
            "f1_kept": f"ref_E12_{cond}_F1_lora_r16",
            "f1_own": own(draft, f"ref_E12_{cond}_F1_{method}_own", inst, method, "f1",
                          f"devreadouts/f1e12_{inst}_{condition}_own_dev_nscc_20261105.jsonl", final_name),
            "fixed": ("ref_E12_gaussian_bar_grad_sparse19_traj" if cond == "gaussian"
                      else "ref_E12_contrast_bar_bn_adapt"),
        }
    final_name, inst, floor = E14
    for method in E14_F1:
        _, run = _run(inst, method, "e14")
        kept_pick = devreadout.retention_final(run, floor)["final"]
        if not (run["final"].get("is_base_noop") and kept_pick["is_base_noop"]):
            raise SystemExit(f"tab:main BabyAI: {method} on {inst} does not keep the base no-op; it needs an entry")
    draft.base_noops.append(f"tab:main BabyAI F1 kept and own: all {len(E14_F1)} F1 methods on {inst} keep the base "
                            f"no-op, as their own final and at S_old >= {floor}")
    noop = "base no-op (gain 0, clean 0 by construction; no entry)"
    cells["babyai"] = {"program": final_name, "f1_kept": noop, "f1_own": noop,
                       "fixed": "ref_E14_pickup_bar_bc_emb_head"}
    return cells


def examples(draft: Draft) -> dict[str, Any]:
    """tab:examples: kept_tuned (F1s s3 gaussian, s1 contrast), kept_est (s4), kept_true (s5) and the no-floor rows."""
    cells: dict[str, Any] = {}
    for cond, (final_name, inst, condition) in E11_FINALS.items():
        def record(arm: str, which: str) -> str:
            return "devreadouts/" + F1_ARM[arm][2].format(inst=inst, cond=condition, which=which)

        def kept_arm(arm: str) -> str:
            method, tag, _ = F1_ARM[arm]
            return kept(draft, f"ref_E11_{cond}_{arm}_kept", inst, method, tag, record(arm, "floor20"), final_name,
                        f"tab:examples {arm} {cond}")

        def own_arm(arm: str) -> str:
            method, tag, _ = F1_ARM[arm]
            return own(draft, f"ref_E11_{cond}_{arm}_own", inst, method, tag, record(arm, "own"), final_name)

        tuned = "F1s_s3_lora_aug" if cond == "gaussian" else "F1s_s1_gentle"
        best8_own = "F1_lora_r16" if cond == "gaussian" else "F1_ft_full_l2sp"
        cells[cond] = {
            "program": final_name,
            "kept_best8": "ref_E11_gaussian_F1_lora_r4" if cond == "gaussian" else "ref_E11_contrast_F1_ft_full_l2sp",
            "kept_tuned": kept_arm(tuned),
            "nobound_best8": own_arm(best8_own),
            "nobound_tuned": own_arm(tuned),
            "nobound_true": own_arm("F1s_s5_oracle_prior"),
            "fixed": ("ref_E11_gaussian_bar_grad_sparse19_sal" if cond == "gaussian"
                      else "ref_E11_contrast_bar_grad_sparse19_traj"),
        }
        if cond == "contrast":
            cells[cond]["kept_est"] = kept_arm("F1s_s4_est_prior")
            cells[cond]["kept_true"] = kept_arm("F1s_s5_oracle_prior")
        else:
            cells[cond]["kept_est"] = cells[cond]["kept_true"] = "unchanged network (g_unchanged; no entry)"
    return cells


def reuse(draft: Draft) -> dict[str, Any]:
    """tab:reuse: per condition and unseen student, the program's transfer, the recipe and the re-selection."""
    cells: dict[str, Any] = {}
    for cond, targets in E11_TARGETS.items():
        session = E11_SESSIONS[cond]
        transfer_dir = runs_dir() / "devreadouts" / "learn" / session / "transfer"
        rows: dict[str, Any] = {}
        for seed, inst in targets.items():
            (path,) = sorted(transfer_dir.glob(f"transfer_*_{inst}.json"))
            record = _json(path)
            condition = record["target"]["condition"]
            final_name = draft.add(SF.BLOCK_FINALS, f"E11_C_{cond}_s{seed}", inst, record["child_sha256"],
                                   {"kind": SF.SOURCE_TRANSFER, "record": str(path.relative_to(runs_dir()))})
            method = F3_METHOD[cond]
            stem = f"devreadouts/f3_{inst}_{condition}_{method}"
            rows[str(seed)] = {
                "instance_id": inst,
                "program": final_name,
                "recipe": candidate(draft, f"ref_E11_C_{cond}_s{seed}_F3_recipe_{method}", inst, method, "f1",
                                    F3_RECIPE[cond], f"{stem}_frontier_dev_nscc_20261105.jsonl", final_name),
                "reselect": kept(draft, f"ref_E11_C_{cond}_s{seed}_F3_reselect_{method}", inst, method, "f1",
                                 f"{stem}_floor20_dev_nscc_20261105.jsonl", final_name,
                                 f"tab:reuse reselect {cond} s{seed}"),
            }
        cells[cond] = rows
    return cells


def stack(draft: Draft) -> dict[str, Any]:
    """tab:stack: both streams after rounds 1 to 3 on all four instances (round 4 is in v1)."""
    cells: dict[str, Any] = {}
    finals_by_round: dict[tuple[int, str], str] = {}
    for row, (prefix, rel) in STREAMS.items():
        record = _json(runs_dir() / rel)
        rounds = record["rounds"]
        per_round: dict[str, Any] = {}
        for number, round_row in enumerate(rounds, start=1):
            names = {}
            for inst, short in STREAM_CONDITIONS.items():
                name = f"{prefix}_W{number}_{short}"
                if number == len(rounds):
                    names[short] = name  # v1's stream_final entry
                    if draft.by_pair.get((inst, round_row["child_sha256"])) != name:
                        raise SystemExit(f"{name}: not the v1 entry of round {number} on {inst}")
                    continue
                source = {"kind": SF.SOURCE_STREAM_ROUND, "record": rel, "round": number}
                if row == "program":
                    names[short] = draft.add(SF.BLOCK_FINALS, name, inst, round_row["child_sha256"], source)
                    finals_by_round[(number, inst)] = names[short]
                else:
                    names[short] = draft.add(SF.BLOCK_REFERENCES, name, inst, round_row["child_sha256"], source,
                                             finals_by_round[(number, inst)])
            per_round[f"W{number}"] = names
        cells[row] = per_round
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    args = parser.parse_args()
    v1 = _json(args.v1)
    if v1["status"] != SF.STATUS_APPROVED:
        raise SystemExit(f"{args.v1}: v2 is built on the approved v1")
    draft = Draft(v1)
    examples_cells = examples(draft)
    cells = {"main": main_table(draft, examples_cells), "examples": examples_cells, "reuse": reuse(draft),
             "stack": stack(draft)}
    n_v1 = len(v1["finals"]) + len(v1["references"])
    n = len(draft.finals) + len(draft.references)
    note = (
        "Draft v2 of the sealed-read list (PREREG Sealed read, list v2: main 9c80cfe / eddd941; user decision about "
        "16:15 Beijing on 2026-11-06, before any sealed read), drafted by session weightcode-0e for WEIGHTCODE-main from the "
        f"store's run records. It keeps the {n_v1} entries of the approved list v1 unchanged (scripts/sealed/"
        "finals_v1.json) and adds every child behind a cell of the paper's main tables (Paper/manuscript/tables/"
        "table_data.json, paper-writing 242cb7c and f0768ff): tab:main, the E12 F1 own picks without the clean floor "
        "(gaussian L2-SP, contrast LoRA r16); tab:examples (in the supplement, cells kept), the F1s kept picks "
        "(gaussian s3; contrast s1, s4, s5) and the own picks without the clean floor (F1 LoRA r16 gaussian; F1s s3 "
        "gaussian; F1s s5 gaussian and contrast); tab:reuse, the E11 C transfers of the two B finals to the seed 101 "
        "to 105 students, the F3 recipe carried unchanged (gaussian LoRA r4 {lr 1e-3, 10 steps}, contrast L2-SP "
        "{0.1, 1e-5, 300 steps}) and the F3 per-student re-selection (kept at S_old >= -20); tab:stack, the E10 teacher v2 stream and the F4 "
        "ft_replay stream after rounds 1 to 3, each on all four stream instances, since the stream score after a "
        "round is the mean gain over all four conditions plus the clean change. A child already on the list is read "
        "once and maps to every cell it fills (scripts/sealed/finals_v2_cells.json): the F1 L2-SP contrast own pick "
        "is its kept pick, the F1s s1 contrast own pick is its kept pick, and the F3 re-selection equals the recipe "
        "on ib1b195f6, i36df1f8f and id83bf155. No entry for a base no-op pick, whose gain and clean are 0 by "
        f"construction on any rows: {'; '.join(draft.base_noops) or 'none'}. {n} entries. Read once, only once the "
        "user has approved the list and it is committed with status approved."
    )
    out = {"schema": SF.LIST_SCHEMA, "status": SF.STATUS_DRAFT, "approval": None, "note": note,
           "finals": draft.finals, "references": draft.references}
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    SF.load_list(args.out)
    cell_map = {"about": ("Table cell -> sealed-list entry name for list v2 (scripts/sealed/finals_v2_draft.json); "
                          "cells of Paper/manuscript/tables/table_data.json. A shared child maps to one entry."),
                "list": str(args.out), "tables": cells, "base_noops": draft.base_noops}
    args.cells.write_text(json.dumps(cell_map, indent=2) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(draft.finals)} finals, {len(draft.references)} references ({n} entries; v1 {n_v1}); "
          f"base no-ops {len(draft.base_noops)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
