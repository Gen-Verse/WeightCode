"""Readout family ``scan_em`` of the sealed read (PREREG E13, SCAN at the rule level; ``weightcode.evalkit.sealed_finals``).

The readout of record is ``scanskill.scan_readout_arrays``: exact match of child and base on the new commands and on
the old commands, the new part by construction, and gain = 100 x delta EM new, clean = 100 x delta EM old (a rule-
generated reference's readout adds its fit-set coverage, from the run's ``config.fit_set.command_ids``, as
``devreadout`` reads it). The rows:

- dev: the instance's recorded new V_select and old V_select (``scanskill.load_recorded``, hash-checked), the rows of
  ``scanskill.scan_readout`` and of every E13 dev readout of record;
- sealed: ``$WEIGHTCODE_SEALED/scan/conditions/<condition>/test.npz`` as the condition's dataset.json ``test`` entry records
  it (location, n, canonical content sha256, part counts): the sealed new third (part 1) then the sealed old commands
  (part 2). Both parts come from one file; the read checks the hash, the row count and that the parts are in that
  order before splitting them.

The instance must be a current dev scanskill-v1 instance (``scanskill.check_scan_current``), and the condition's
dataset.json must record the instance's V_select and old V_select hashes, so the sealed entry is the one of the
condition the instance was built from. This module imports ``weightcode.evalkit.scanskill`` (branch e13-scan); in a tree
without it the family is refused by ``sealed_finals.family``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance, load_instance
from weightcode.data import scan as S
from weightcode.data.store import dataset_json_path, read_dataset_json
from weightcode.evalkit import devreadout
from weightcode.evalkit import scanskill as K
from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit.targets import file_sha256
from weightcode.patch import Limits

PART_ORDER = "part 1 (the sealed new third) then part 2 (the sealed old commands)"


class ScanEM:
    """scan_em: ``scanskill.scan_readout_arrays`` on the new and the old commands, child against base."""

    readout = SF.READOUT_SCAN_EM
    code_modules: tuple[str, ...] = ("weightcode.evalkit.sealed_scan", "weightcode.evalkit.scanskill", "weightcode.data.scan")

    def load_instance(self, instance_id: str) -> Instance:
        instance = load_instance(K.scan_instance_dir(instance_id))
        if instance.private.get("evidence_rule") != K.SCAN_RULE:
            raise SF.SealedFinalsError(f"{instance_id}: follows {instance.private.get('evidence_rule')!r}, not "
                                       f"{K.SCAN_RULE}")
        try:
            K.check_scan_current(instance.instance_dir)
        except K.ScanskillError as error:
            raise SF.SealedFinalsError(str(error)) from error
        return instance

    def reference_limits(self, run: Mapping[str, Any], run_dir: Path, instance: Instance) -> Limits | None:
        """The E13 program limits, which a scanskill reference run must record (as ``devreadout`` rebuilds it)."""
        expected = K.scan_program_limits(instance.task["limits"], len(instance.params))
        try:
            return devreadout._program_baseline_limits(run, run_dir, expected, "scan", "E13")
        except devreadout.DevReadoutError as error:
            raise SF.SealedFinalsError(str(error)) from error

    def read_context(self, final: SF.Final, run: Mapping[str, Any] | None) -> dict[str, Any]:
        fit_set = ((run or {}).get("config") or {}).get("fit_set")
        fit_ids = fit_set.get("command_ids") if isinstance(fit_set, Mapping) else None
        return {"fit_ids": None if fit_ids is None else [int(i) for i in fit_ids]}

    def dev_rows(self, instance: Instance) -> tuple[Any, dict[str, Any]]:
        private = instance.private
        try:
            new, old = K.load_recorded(private, "V_select"), K.load_recorded(private, "old_select")
        except K.ScanskillError as error:
            raise SF.SealedFinalsError(f"{instance.instance_id}: {error}") from error
        record = {part: {"split": split, "path": str(private[f"{K.SPLIT_KEYS[split]}_path"]),
                         "sha256": private[f"{K.SPLIT_KEYS[split]}_content_sha256"], "n": int(len(arrays["y"]))}
                  for part, split, arrays in (("new", "V_select", new), ("old", "old_select", old))}
        return (new, old), record

    def sealed_index(self, instance: Instance) -> dict[str, Any]:
        private = instance.private
        condition = str(private["condition"])
        if condition not in S.CONDITIONS:
            raise SF.SealedFinalsError(f"{instance.instance_id}: condition {condition!r} is not an E13 condition")
        path = dataset_json_path(S.DATASET, condition)
        meta = read_dataset_json(S.DATASET, condition)
        for split in ("V_select", "old_select"):
            recorded = (meta.get("splits", {}).get(split) or {}).get("sha256")
            if recorded != private.get(f"{K.SPLIT_KEYS[split]}_content_sha256"):
                raise SF.SealedFinalsError(f"{path}: {split} sha256 {recorded!r} is not the instance's; the instance "
                                           "was not built from this condition's splits")
        test = meta.get("test")
        if not isinstance(test, dict):
            raise SF.SealedFinalsError(f"{path}: records no test object")
        n, counts = test.get("n"), test.get("part_counts")
        if (not isinstance(n, int) or not isinstance(counts, dict) or set(counts) != {str(S.PART_NEW), str(S.PART_OLD)}
                or sum(int(c) for c in counts.values()) != n):
            raise SF.SealedFinalsError(f"{path}: the test entry needs n and part_counts of parts {S.PART_NEW} and "
                                       f"{S.PART_OLD} summing to n, got n {n!r}, part_counts {counts!r}")
        entry = {
            "location": test.get("location"),
            "path": str(SF.resolve_location(test.get("location"), str(path))),
            "format": SF.FORMAT_STORE_NPZ, "n": n,
            "sha256": SF.recorded_sha(test.get("sha256"), f"{path} test"), "sha256_kind": SF.HASH_CONTENT,
            "n_new": int(counts[str(S.PART_NEW)]), "n_old": int(counts[str(S.PART_OLD)]), "order": PART_ORDER,
            "rule": test.get("rule"), "index": str(path), "index_file_sha256": file_sha256(path),
        }
        return {"dataset": S.DATASET, "condition": condition, "test": entry}

    def sealed_paths(self, index: Mapping[str, Any]) -> list[Path]:
        return [Path(index["test"]["path"])]

    def sealed_rows(self, instance: Instance, index: Mapping[str, Any], cache: dict[Any, Any]
                    ) -> tuple[Any, dict[str, Any]]:
        entry = index["test"]
        key = ("scan", entry["path"], entry["sha256"])
        if key not in cache:
            path = Path(entry["path"])
            arrays = SF.load_store_npz(path, entry["sha256"], entry["n"])
            n_new = entry["n_new"]
            part = np.asarray(arrays["part"])
            if not (np.all(part[:n_new] == S.PART_NEW) and np.all(part[n_new:] == S.PART_OLD)):
                raise SF.SealedFinalsError(f"{path}: the parts are not {PART_ORDER} with {n_new} new rows")
            cache[key] = ({name: value[:n_new] for name, value in arrays.items()},
                          {name: value[n_new:] for name, value in arrays.items()})
        new, old = cache[key]
        record = {part: {"split": f"sealed {part}", "location": entry["location"], "path": entry["path"],
                         "verified_sha256": entry["sha256"], "n": int(len(arrays["y"]))}
                  for part, arrays in (("new", new), ("old", old))}
        return (new, old), record

    def read(self, instance: Instance, child: Mapping[str, np.ndarray], rows: Any, batch_size: int,
             context: Mapping[str, Any]) -> tuple[dict[str, Any], float, float]:
        new, old = rows
        result = K.scan_readout_arrays(new, old, instance.forward, instance.params, dict(child), batch_size,
                                       fit_ids=context.get("fit_ids"))
        parts = {"readouts_scan": devreadout.json_safe(result)}
        return parts, *self.points(parts)

    def session_parts(self, info_final: Mapping[str, Any]) -> dict[str, Any]:
        return {"readouts_scan": info_final["readouts_scan"]}

    def reference_parts(self, lines: Sequence[Mapping[str, Any]], final: SF.Final, patch_id: str | None
                        ) -> dict[str, Any]:
        return {"readouts_scan": SF.retention_line(lines, final, patch_id)["readouts_scan"]}

    def points(self, parts: Mapping[str, Any]) -> tuple[float, float]:
        return float(parts["readouts_scan"]["gain_points"]), float(parts["readouts_scan"]["clean_points"])

    def environment(self) -> dict[str, Any]:
        return {}


FAMILY = ScanEM()

__all__ = ["FAMILY", "PART_ORDER", "ScanEM"]
