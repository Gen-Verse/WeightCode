"""Readout family ``policy_success`` of the sealed read (PREREG E14, BabyAI pickup; ``weightcode.evalkit.sealed_finals``).

The readout of record is ``e14.e14_readout``: greedy forward.py rollouts of base and child on the new-mission and the
old-mission episodes, per part ``e14._summary`` (success, fixed, broken, lengths, pickups, by article and by object
type) with the split by demonstrated (color, type) (``e14._seen_split``), and gain = 100 x delta success new, clean =
100 x delta success old. ``e14_readout`` fixes its episodes to the V_select seed ranges, so this module runs the same
steps with the episode seeds as a parameter; the dev mode shows the two equal field for field. The rows:

- dev: the V_select seed ranges of the code (``weightcode.data.e14.SEED_RANGES``), checked against the instance's recorded
  seed hashes, and the base's episodes checked against the instance's recorded base success hashes, as
  ``e14_readout`` checks them;
- sealed: ``$WEIGHTCODE_SEALED/e14_gotolocal/test.npz`` as dataset.json ``sealed_record`` records it (canonical content
  sha256, n_old, n_new): x the seeds, old then new, y the mission code, idx the position. The read checks the hash,
  the keys, the order and codes, and that the seeds are distinct and in [SEALED_FLOOR, 2**31), above every dev range.

The installed minigrid and gymnasium must be the versions the instance was built with (its private.json
``environment_versions``); a different environment version is refused, since it can change the episodes. This module
imports ``weightcode.evalkit.e14`` and minigrid (branch e14-babyai); in a tree without them the family is refused by
``sealed_finals.family``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.baselines.instance_io import Instance
from weightcode.data import e14 as D
from weightcode.evalkit import devreadout
from weightcode.evalkit import e14 as E
from weightcode.evalkit import e14read as ER
from weightcode.evalkit import sealed_finals as SF
from weightcode.evalkit.targets import file_sha256
from weightcode.patch import Limits, task_limits

SEALED_KEYS = frozenset({"x", "y", "idx"})
PARTS = tuple(E.SELECT_RANGES)
"""The parts in ``e14_readout``'s order: new, then old."""


def _check_environment(instance: Instance) -> dict[str, str]:
    installed = D.environment_versions()
    recorded = instance.private.get("environment_versions")
    if installed != recorded:
        raise SF.SealedFinalsError(f"{instance.instance_id}: installed environment {installed} is not the instance's "
                                   f"{recorded}")
    return installed


class PolicySuccess:
    """policy_success: greedy rollouts of base and child on the new and the old mission episodes."""

    readout = SF.READOUT_POLICY_SUCCESS
    code_modules: tuple[str, ...] = ("weightcode.evalkit.sealed_e14", "weightcode.evalkit.e14", "weightcode.evalkit.e14read",
                                     "weightcode.data.e14")

    def load_instance(self, instance_id: str) -> Instance:
        instance = E.load_e14_instance(instance_id)
        if instance.private.get("evidence_rule") != E.POLICYSKILL_RULE:
            raise SF.SealedFinalsError(f"{instance_id}: follows {instance.private.get('evidence_rule')!r}, not "
                                       f"{E.POLICYSKILL_RULE}")
        try:
            E.check_e14_current(instance.instance_dir)
        except E.E14InstanceError as error:
            raise SF.SealedFinalsError(str(error)) from error
        _check_environment(instance)
        return instance

    def reference_limits(self, run: Mapping[str, Any], run_dir: Path, instance: Instance) -> Limits | None:
        """The E14 program limits, which an E14 reference run must record (as ``e14read.read_run`` checks)."""
        if (run.get("e14") or {}).get("rule") != E.POLICYSKILL_RULE:
            raise SF.SealedFinalsError(f"{run_dir}: not an E14 run (record e14.rule)")
        expected = ER.program_limits(instance)
        if run.get("limits") != task_limits(expected):
            raise SF.SealedFinalsError(f"{run_dir}: records limits {run.get('limits')!r}, not the E14 program limits "
                                       f"{task_limits(expected)}")
        return expected

    def read_context(self, final: SF.Final, run: Mapping[str, Any] | None) -> dict[str, Any]:
        return {}

    def dev_rows(self, instance: Instance) -> tuple[Any, dict[str, Any]]:
        private = instance.private
        seeds, record = {}, {}
        for part, name in E.SELECT_RANGES.items():
            values = np.asarray(D.SEED_RANGES[name].seeds, dtype=np.int64)
            digest = E.seeds_sha256(values)
            if digest != private["vselect_seeds_sha256"][part]:
                raise SF.SealedFinalsError(f"{instance.instance_id}: the code's {name} seeds differ from the instance "
                                           "record")
            seeds[part] = values
            record[part] = {"split": name, "seeds_sha256": digest, "n": int(len(values))}
        return {"seeds": seeds, "base_success_sha256": dict(private["base_vselect_success_sha256"])}, record

    def sealed_index(self, instance: Instance) -> dict[str, Any]:
        path = D.store_dir() / "dataset.json"
        if not path.is_file():
            raise SF.SealedFinalsError(f"no {path}")
        meta = json.loads(path.read_text(encoding="utf-8"))
        sealed = meta.get("sealed_record")
        if not isinstance(sealed, dict):
            raise SF.SealedFinalsError(f"{path}: records no sealed_record")
        n_old, n_new = sealed.get("n_old"), sealed.get("n_new")
        if not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in (n_old, n_new)):
            raise SF.SealedFinalsError(f"{path}: sealed_record needs positive n_old and n_new, got {n_old!r}, "
                                       f"{n_new!r}")
        entry = {
            "location": sealed.get("path"),
            "path": str(SF.resolve_location(sealed.get("path"), str(path))),
            "format": SF.FORMAT_STORE_NPZ, "n": n_old + n_new, "n_old": n_old, "n_new": n_new,
            "sha256": SF.recorded_sha(sealed.get("sha256"), f"{path} sealed_record"), "sha256_kind": SF.HASH_CONTENT,
            "order": "old then new; x the seeds, y the mission code, idx the position", "rule": sealed.get("rule"),
            "index": str(path), "index_file_sha256": file_sha256(path),
        }
        return {"dataset": D.DATASET, "condition": None, "test": entry}

    def sealed_paths(self, index: Mapping[str, Any]) -> list[Path]:
        return [Path(index["test"]["path"])]

    def sealed_rows(self, instance: Instance, index: Mapping[str, Any], cache: dict[Any, Any]
                    ) -> tuple[Any, dict[str, Any]]:
        entry = index["test"]
        key = ("e14", entry["path"], entry["sha256"])
        if key not in cache:
            path = Path(entry["path"])
            arrays = SF.load_store_npz(path, entry["sha256"], entry["n"])
            if set(arrays) != SEALED_KEYS:
                raise SF.SealedFinalsError(f"{path}: keys {sorted(arrays)}, expected {sorted(SEALED_KEYS)}")
            x, y, idx = (np.asarray(arrays[name]) for name in ("x", "y", "idx"))
            n_old = entry["n_old"]
            codes = np.concatenate([np.full(n_old, D.MISSION_CODES["old"]),
                                    np.full(entry["n_new"], D.MISSION_CODES["new"])])
            if not np.array_equal(y, codes) or not np.array_equal(idx, np.arange(len(x))):
                raise SF.SealedFinalsError(f"{path}: y and idx are not old then new in position order")
            if (x.dtype != np.int64 or len(np.unique(x)) != len(x) or x.min() < D.SEALED_FLOOR
                    or x.max() >= D.SEALED_CEILING):
                raise SF.SealedFinalsError(f"{path}: the seeds are not distinct int64 values in [{D.SEALED_FLOOR}, "
                                           f"{D.SEALED_CEILING})")
            cache[key] = {"old": x[:n_old], "new": x[n_old:]}
        seeds = cache[key]
        record = {part: {"split": f"sealed {part}", "location": entry["location"], "path": entry["path"],
                         "verified_sha256": entry["sha256"], "n": int(len(seeds[part]))} for part in PARTS}
        return {"seeds": dict(seeds), "base_success_sha256": None}, record

    def read(self, instance: Instance, child: Mapping[str, np.ndarray], rows: Any, batch_size: int,
             context: Mapping[str, Any]) -> tuple[dict[str, Any], float, float]:
        """``e14_readout``'s steps on ``rows['seeds']`` (``batch_size`` does not apply: one forward per step)."""
        _check_environment(instance)
        seen = E.demonstrated_descriptors(instance.evidence)
        base_fn = E.logits_fn(instance.forward, instance.params)
        child_fn = E.logits_fn(instance.forward, dict(child))
        expected = rows["base_success_sha256"]
        result: dict[str, Any] = {"demonstrated_descriptors": sorted([D.COLORS[c], D.OLD_TYPES[t]] for c, t in seen)}
        for part in PARTS:
            seeds = rows["seeds"][part]
            base = D.rollouts(base_fn, part, seeds)
            if expected is not None and E.success_sha256(base.success) != expected[part]:
                raise SF.SealedFinalsError(f"{instance.instance_id}: the base's {part} episodes differ from its import "
                                           "readout")
            child_result = D.rollouts(child_fn, part, seeds)
            result[part] = E._summary(base, child_result)
            result[part]["by_demonstrated_descriptor"] = E._seen_split(base, child_result, seen)
        result["gain"] = result["new"]["points"]
        result["clean"] = result["old"]["points"]
        result["joint"] = result["gain"] + result["clean"]
        parts = {"readouts_policyskill": devreadout.json_safe(result)}
        return parts, *self.points(parts)

    def session_parts(self, info_final: Mapping[str, Any]) -> dict[str, Any]:
        return {"readouts_policyskill": info_final["readouts_policyskill"]}

    def reference_parts(self, lines: Sequence[Mapping[str, Any]], final: SF.Final, patch_id: str | None
                        ) -> dict[str, Any]:
        """The kept pick's candidate readout in the run's ``e14read.read_run`` record."""
        floor = final.source["s_old_floor"]
        if patch_id is None:
            raise SF.SealedFinalsError(f"{final.name}: the kept pick at S_old >= {floor} is the base no-op, whose "
                                       "read_run record holds no candidate readout to check against")
        hits = [line for line in lines if (line.get("kept") or {}).get("patch_id") == patch_id
                and (line["kept"].get("selection") or {}).get("s_old_floor") == floor
                and line.get("instance_id") == final.instance_id]
        if len(hits) != 1:
            raise SF.SealedFinalsError(f"{final.name}: {len(hits)} read_run records keep {patch_id} at S_old >= "
                                       f"{floor} in {final.source['dev_record']}; expected one")
        candidates = [c for c in hits[0]["candidates"] if c["patch_id"] == patch_id]
        if len(candidates) != 1 or candidates[0].get("child_sha256") != final.child_sha256:
            raise SF.SealedFinalsError(f"{final.name}: the record has no single candidate {patch_id} with the listed "
                                       "child")
        return {"readouts_policyskill": candidates[0]["readout"]}

    def points(self, parts: Mapping[str, Any]) -> tuple[float, float]:
        return float(parts["readouts_policyskill"]["gain"]), float(parts["readouts_policyskill"]["clean"])

    def environment(self) -> dict[str, Any]:
        return dict(D.environment_versions())


FAMILY = PolicySuccess()

__all__ = ["FAMILY", "PARTS", "PolicySuccess", "SEALED_KEYS"]
