"""PREREG E11 in the harness: E10 with a ViT-Lite-7/4 student (dataset ``cifar10_vl7``).

An E11 instance is an ordinary evidence-v1 instance of the dataset ``cifar10_vl7`` on E10's data: the same four
CIFAR-10-C conditions at severity 5 under the same names, the same rows, trial, experience, limits, check and readouts
(``weightcode.harness.trial``, ``experience``, ``arms.applied_limits``, ``clean``). Only the student differs: ViT-Lite-7/4
(Hassani et al. 2021) with LayerNorm and no BatchNorm. The limits are rank-based and need no change: every
LayerNorm weight and bias, and every linear and patch-projection bias, is 1-D, so exempt from the relative Frobenius
limit and counted in the 20% of changed entries; the class token and the positional embedding are rank 2 or more and
fall under it, as do the patch projection, qkv, proj, the MLP weights and the head weight; max_ops is
max(64, tensor count) (85 for the authors' layout). What this module adds:

* ``DATASET``: the store's dataset id, which the data session builds the E11 store under. Its ``_``-separated parts
  become leak tokens (``weightcode.evalkit.instances.leak_tokens``), so the name avoids architecture words: with
  ``cifar10_vit`` the word "vit" would be forbidden in forward.py and model_spec.json, which must describe the
  network. "vl7" stands for ViT-Lite-7/4.
* Leak words: E10's (``weightcode.harness.e10.leak_words``: every E10 corruption's condition leak tokens, the four family
  names and the benchmark's names), because the conditions are E10's. Architecture words ("transformer",
  "attention", "vit", "layernorm", "patch", "token") are not leak words: the editor reads forward.py and
  model_spec.json, which name the architecture, and nothing in the leak rules names one. One condition leak word,
  "gaussian" (``weightcode.evalkit.instances.CONDITION_LEAK_WORDS``), is a hazard for a ViT forward.py: a docstring that
  spells out GELU as "Gaussian error linear unit" fails the instance builder's scan and the pre-launch scan.
* Self-test tokens (``self_test_tokens``): E10's condition and corruption names with an underscore, every active E11
  and E10 instance id (the conditions are shared, so an id of either names one) and the given session ids.
* MCP timeouts (``mcp_timeouts``): E10's floors, 1,200 s per tool call and 900 s to connect, refused below. A
  ViT-Lite-7/4 check predicts 4,000 condition rows and 4,000 clean rows after the program run (up to 300 s). The
  synthetic stand-in's numpy forward took 231 s per 8,000 rows on 4 threads (2026-11-04, i7-13700 at load average 23
  to 34, ``scripts/e11/vit_synth_check.py timing``): a worst-case check is about 530 s, inside 1,200 s, and the
  server's base predictions of the same 8,000 rows fit the 900 s connect timeout. The real student's forward is timed
  again by the data session (PREREG E11).
* ``check_student``: before any session write, an E11 student must hold no BatchNorm running statistics and exactly
  one class token and one positional embedding, both of rank >= 2, found by name (``CLASS_TOKEN_NAMES``,
  ``POSITIONAL_NAMES``: the authors' ``class_emb`` and ``positional_emb``, timm's ``cls_token`` and ``pos_embed``,
  torchvision's ``class_token`` and ``pos_embedding``). A squeezed 1-D class token would be exempt from rel-Frob,
  against PREREG E11, so it is refused rather than run.
* Lookups for the launch commands (``find``, ``condition_ids``, ``transfer_targets``; ``python -m weightcode.harness.e11``):
  instances are found by dataset, condition and seed in ``$WEIGHTCODE_PRIVATE/instances/index.json``, never by a remembered
  id. A condition that misses headroom has no instance, and ``ids`` reports it as null.

E11's two-round loop stream (gaussian_noise_s5, then contrast_s5, under loop-retain-v2) runs through
``weightcode.harness.loop_stream`` unchanged: the runner applies this module's MCP floors, leak words, self-test tokens and
student check to each loop session, and ``loop_stream.default_wall_timeout_s`` gives E11 sessions their 120 min.

The rule view (PREREG K1, the knowledge route on the E11 ViT; ``VARIANTS``, ``check_variant``, ``condition_rule``).
The same instance, student, checks, kept rule, limits, effort and wall as an E11 LEARN session; only the editor's
information differs: the experience holds the instance's 64 clean labelled replay rows alone (no condition row,
``weightcode.harness.experience.rule_experience``), and the task statement states the condition's rule as a formula
(``RULE_SENTENCES``, the PREREG K1 wording; ``weightcode.harness.experience.learn_task_rule``). A rule is stated only for a
condition listed in ``CONDITION_RULES``, and only when the condition record
(``$WEIGHTCODE_PRIVATE/<dataset>/conditions/<condition>/dataset.json``) says the condition was made by the pinned official
code (``OFFICIAL_FILE_SHA256``) with that corruption, severity and constant: ``condition_rule`` reads and checks it and
returns the sentence with the record's hash. Any other instance, and any loop round, is refused. The condition's name
and every E10 leak word stay forbidden: the rule names no condition, so no leak word is exempted.

The information ladder (PREREG "Information ladder on one student: K0a, K0b, U"; ``LADDER_VARIANTS``,
``ladder_view``). Three more views of the same E11 LEARN session on the one PREREG student (the seed-100 student of
``LADDER_CONDITION``, gaussian_noise_s5, i00802425 in the store), each refused on any other instance, arm or round:

* ``UNLABELLED`` (U): E11's experience with ``y_new`` withheld (``weightcode.harness.experience.unlabelled_experience``):
  x_new, pred_new and split_ids_new and the labelled clean part stay, rows and order E11's; the texts that call the
  condition rows labelled are changed and the text that says which of them are the student's errors is removed, each
  found exactly once. The selection (the first 16 of the 64 drawn from the errors) is inherited from E11 and not
  stated to the editor. Checks, kept rule and leak words are E11's.
* ``NAMED`` (K0b): no data. The sandbox holds no image array of any kind: no experience file, only the network
  (model.npz, model_spec.json, forward.py) and task.json, and a program gets an empty experience. The statement names
  the capability in the PREREG sentence (``CAPABILITY_SENTENCES``), which is the one text exempt from the condition's
  leak words and hint words (``weightcode.harness.leaks.ForbiddenStrings.sentence_exempt``); E11's 8 checks and kept rule.
* ``NAMED_NOCHECK`` (K0a): K0b without ``patch_check``. ``declare_final`` runs and validates the program within the
  limits without scoring it; after the session the harness checks the last valid declaration once on V_check, which
  the editor never sees, and keeps it only if joint-retain-v1 makes it eligible, else the base
  (``weightcode.harness.finalize``, ``HIDDEN_CHECK``). No prompt or tool text of this view says "check".

Nothing here reads V_select, the sealed store or a condition's images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from weightcode.evalkit.evidence import EVIDENCE_RULE_V1
from weightcode.harness import e10

DATASET = "cifar10_vl7"
CONDITIONS = e10.STREAM_CONDITIONS
"""PREREG E11: E10's four conditions, in E10's B order; each runs only if it passes headroom on the E11 student."""
SOURCE_SEED = e10.SOURCE_SEED
TRANSFER_SEEDS = e10.TRANSFER_SEEDS
MCP_CONNECT_TIMEOUT_MS = e10.MCP_CONNECT_TIMEOUT_MS
MCP_TOOL_TIMEOUT_MS = e10.MCP_TOOL_TIMEOUT_MS
CLASS_TOKEN_NAMES = ("class_emb", "cls_token", "class_token")
POSITIONAL_NAMES = ("positional_emb", "pos_embed", "pos_embedding")
BATCHNORM_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")

# The rule view (PREREG K1; module docstring): the standard view is every E11 session before K1.
STANDARD = "standard"
RULE = "rule"
UNLABELLED = "unlabelled"
NAMED = "named"
NAMED_NOCHECK = "named_nocheck"
VARIANTS = (STANDARD, RULE, UNLABELLED, NAMED, NAMED_NOCHECK)
LADDER_VARIANTS = (UNLABELLED, NAMED, NAMED_NOCHECK)
"""The information-ladder views (module docstring): U, K0b and K0a."""
NAMED_VARIANTS = (NAMED, NAMED_NOCHECK)
"""The no-data views (K0b, K0a): no experience file, and the capability sentence in the statement."""
LADDER_CONDITION = "gaussian_noise_s5"
"""The ladder's one condition (PREREG): its seed-``SOURCE_SEED`` E11 student is the ladder's one student."""
CAPABILITY_SENTENCES = {
    "gaussian_noise_s5": ("Make the classifier robust to images corrupted by additive Gaussian pixel noise, while "
                          "keeping "
                          "its accuracy on clean images."),
}
"""The sentence that names the capability in K0a and K0b, worded as the PREREG ladder entry words it (user, 2026-11-05
at 20:25 and 20:40)."""
HIDDEN_CHECK = "hidden-check-v1"
"""K0a's post-session rule: one V_check check of the last valid declaration, unseen by the editor, then the session's
kept rule (``weightcode.harness.finalize``)."""
RULE_SENTENCES = {
    "gaussian_noise_s5": (
        "Every image of the condition is made from a clean uint8 image x as np.uint8(clip(x / 255 + n, 0, 1) * 255), "
        "with n drawn independently per value from a normal distribution of mean 0 and standard deviation 0.10."
    ),
}
"""The rule of each condition the rule view states, worded as PREREG K1 words it (user, 2026-11-05 19:03)."""
CONDITION_RULES = {"gaussian_noise_s5": {"corruption": "gaussian_noise", "severity": 5, "constants": "c = 0.10"}}
"""What the condition record must say for its rule to be stated: the corruption, severity and constant of the pinned
official code (make_cifar_c.py gaussian_noise, c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1])."""
OFFICIAL_FILE_SHA256 = "f865ae91dffb2b4834292ce05c876bf3d77f36417f70075a2b5259348b0c83a5"
"""sha256 of the official make_cifar_c.py (``weightcode.data.conditions.SOURCE["file_sha256"]``, restated so that the
runner does not import the data module; a test holds the two equal)."""


class E11HarnessError(LookupError):
    """The E11 instances asked for do not exist as one active instance each, or a student is not an E11 student."""


def is_e11(private: Mapping[str, Any]) -> bool:
    """True for an evidence-v1 instance of the E11 dataset (its private.json)."""
    return private.get("dataset") == DATASET and private.get("evidence_rule") == EVIDENCE_RULE_V1


def check_variant(variant: str | None) -> str:
    """``variant`` as one of ``VARIANTS``; None is the standard view (``ValueError`` for another name)."""
    name = STANDARD if variant is None else variant
    if name not in VARIANTS:
        raise ValueError(f"unknown E11 variant {variant!r}; one of {VARIANTS}")
    return name


def ladder_view(private: Mapping[str, Any], variant: str) -> dict[str, Any]:
    """The record of ladder view ``variant`` (``LADDER_VARIANTS``) on the instance of ``private``.

    Refused (``E11HarnessError``) unless the instance is an E11 evidence-v1 instance of ``LADDER_CONDITION`` on the
    seed-``SOURCE_SEED`` student, the PREREG ladder's one student. A no-data view (``NAMED_VARIANTS``) records its
    capability sentence and, for K0a, the hidden check (``HIDDEN_CHECK``)."""
    if variant not in LADDER_VARIANTS:
        raise E11HarnessError(f"{variant!r} is not a ladder view; the ladder's are {list(LADDER_VARIANTS)}")
    if not is_e11(private):
        raise E11HarnessError(f"the ladder views are for an E11 evidence-v1 instance (dataset {DATASET}), not one of "
                              f"dataset {private.get('dataset')!r} under {private.get('evidence_rule')!r}")
    condition, seed = str(private.get("condition")), private.get("seed")
    if condition != LADDER_CONDITION or seed != SOURCE_SEED:
        raise E11HarnessError(f"the ladder views run on the {LADDER_CONDITION} instance of the seed-{SOURCE_SEED} "
                              f"student only, not {condition!r} seed {seed!r}")
    record: dict[str, Any] = {"variant": variant, "condition": condition, "seed": int(seed)}
    if variant in NAMED_VARIANTS:
        record["capability_sentence"] = CAPABILITY_SENTENCES[condition]
    if variant == NAMED_NOCHECK:
        record["hidden_check"] = HIDDEN_CHECK
    return record


def condition_rule(private: Mapping[str, Any]) -> dict[str, Any]:
    """The rule the rule view (PREREG K1) states for the instance of ``private``, and its provenance.

    Refused (``E11HarnessError``) unless the instance is an E11 evidence-v1 instance of a condition in
    ``CONDITION_RULES`` and its condition record (``weightcode.data.store.dataset_json_path``) names that condition and
    states its corruption, severity and constant, made by the official code file of ``OFFICIAL_FILE_SHA256``. Returns
    the sentence (``RULE_SENTENCES``), the stated fields and the record's sha256."""
    from weightcode.data.store import dataset_json_path

    if not is_e11(private):
        raise E11HarnessError(f"the rule view (PREREG K1) is for an E11 evidence-v1 instance (dataset {DATASET}), not "
                              f"one of dataset {private.get('dataset')!r} under {private.get('evidence_rule')!r}")
    condition = str(private.get("condition"))
    if condition not in CONDITION_RULES:
        raise E11HarnessError(f"the rule view states a rule for {sorted(CONDITION_RULES)} only, not {condition!r}")
    path = dataset_json_path(DATASET, condition)
    data = path.read_bytes()
    record = json.loads(data.decode("utf-8")).get("condition")
    if not isinstance(record, Mapping) or record.get("name") != condition or record.get("kind") != "corruption":
        raise E11HarnessError(f"the condition record of {condition} does not describe that corruption condition")
    expected = CONDITION_RULES[condition]
    stated = {key: record.get(key) for key in expected}
    if stated != expected:
        raise E11HarnessError(f"the condition record of {condition} states {stated}, not {expected}, so its rule "
                              "is not the one the rule view states")
    source = record.get("source")
    if not isinstance(source, Mapping) or source.get("file_sha256") != OFFICIAL_FILE_SHA256:
        raise E11HarnessError(f"the condition record of {condition} does not name the official code file "
                              f"(sha256 {OFFICIAL_FILE_SHA256}) as its source")
    return {"variant": RULE, "condition": condition, "sentence": RULE_SENTENCES[condition], **stated,
            "source": dict(source), "condition_dataset_json_sha256": hashlib.sha256(data).hexdigest()}


def leak_words() -> list[str]:
    """Lower-case words an E11 session's text scan forbids beyond its base list: E10's (module docstring)."""
    return e10.leak_words()


def instance_ids() -> list[str]:
    """Every active E11 instance id, any split role."""
    from weightcode.harness.instance import list_instances

    return sorted(str(row["instance_id"]) for row in list_instances(split_role=None, dataset=DATASET))


def self_test_tokens(session_ids: Iterable[str] = ()) -> list[str]:
    """Tokens an E11 session forbids like a condition name (self-test and text scan): E10's condition and corruption
    names with an underscore, every active E11 and E10 instance id and ``session_ids``."""
    return sorted({*e10.self_test_tokens(session_ids), *instance_ids()})


def mcp_timeouts(private: Mapping[str, Any], tool_ms: int | None, connect_ms: int | None,
                 defaults: tuple[int, int]) -> tuple[int, int]:
    """The (tool, connect) MCP timeouts of a session: the given values, else E10's floors on an E11 instance and
    ``defaults`` otherwise. An E11 instance refuses values below the floors (``ValueError``)."""
    return e10.floored_mcp_timeouts(is_e11(private), "E11", "the ViT-Lite check predictions", tool_ms, connect_ms,
                                    defaults)


def _named(params: Mapping[str, Any], stems: Sequence[str]) -> list[str]:
    return sorted(name for name in params if any(stem in str(name).lower() for stem in stems))


def check_student(params: Mapping[str, Any]) -> dict[str, Any]:
    """The student's tensor kinds under the E11 limits (module docstring); ``E11HarnessError`` if it holds BatchNorm
    statistics or not exactly one class token and one positional embedding of rank >= 2."""
    shapes = {str(name): tuple(np.shape(value)) for name, value in params.items()}
    problems = []
    batchnorm = sorted(name for name in shapes if name.endswith(BATCHNORM_SUFFIXES))
    if batchnorm:
        problems.append(f"BatchNorm statistics {batchnorm[:4]}{' ...' if len(batchnorm) > 4 else ''}; an E11 student "
                        "has none")
    found = {}
    for kind, stems in (("class token", CLASS_TOKEN_NAMES), ("positional embedding", POSITIONAL_NAMES)):
        names = _named(shapes, stems)
        if len(names) != 1:
            problems.append(f"{len(names)} {kind} tensors {names} (names containing one of {list(stems)}); expected 1")
        elif len(shapes[names[0]]) < 2:
            problems.append(f"the {kind} {names[0]} has shape {list(shapes[names[0]])}: rank < 2 would exempt it from "
                            "the relative Frobenius limit, which PREREG E11 applies to it")
        else:
            found[kind] = names[0]
    if problems:
        raise E11HarnessError("not an E11 student: " + "; ".join(problems))
    one_d = [name for name, shape in shapes.items() if len(shape) < 2]
    return {"n_tensors": len(shapes), "n_entries": int(sum(int(np.prod(shape)) for shape in shapes.values())),
            "class_token": found["class token"], "positional_embedding": found["positional embedding"],
            "n_rel_frob_tensors": len(shapes) - len(one_d), "n_exempt_1d_tensors": len(one_d),
            "n_exempt_1d_entries": int(sum(int(np.prod(shapes[name])) for name in one_d))}


def find(condition: str, seed: int) -> str:
    """The one active E11 dev instance of ``condition`` and student ``seed``."""
    from weightcode.harness.instance import find_instance

    if condition not in CONDITIONS:
        raise E11HarnessError(f"{condition!r} is not an E11 condition; E11's are {list(CONDITIONS)}")
    try:
        return find_instance(dataset=DATASET, seed=int(seed), condition=condition)
    except LookupError as exc:
        raise E11HarnessError(str(exc)) from None


def condition_ids(seed: int = SOURCE_SEED) -> dict[str, str | None]:
    """Condition -> the one active E11 dev instance of student ``seed``, or None where there is none (a condition that
    missed headroom). More than one instance of a (condition, seed) is an error."""
    from weightcode.harness.instance import list_instances

    found: dict[str, str | None] = {}
    for condition in CONDITIONS:
        rows = list_instances(dataset=DATASET, seed=int(seed), condition=condition)
        if len(rows) > 1:
            raise E11HarnessError(f"{len(rows)} active E11 instances of {condition} seed {seed}: "
                                  f"{[row.get('instance_id') for row in rows]}")
        found[condition] = str(rows[0]["instance_id"]) if rows else None
    return found


def transfer_targets(instance_id: str) -> list[str]:
    """PREREG E11 C: the instances of the same condition on the students of ``TRANSFER_SEEDS``."""
    from weightcode.harness.instance import read_instance_record

    private = read_instance_record(instance_id).private
    if not is_e11(private):
        raise E11HarnessError(f"{instance_id} is not an E11 instance")
    if int(private["seed"]) != SOURCE_SEED:
        raise E11HarnessError(f"{instance_id} is a seed-{private['seed']} instance; C transfers seed {SOURCE_SEED}")
    return [find(str(private["condition"]), seed) for seed in TRANSFER_SEEDS]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E11 instance lookups for the launch commands")
    sub = parser.add_subparsers(dest="command", required=True)
    ids = sub.add_parser("ids", help="one condition's instance id, or condition -> id (null: none) as JSON")
    ids.add_argument("--seed", type=int, default=SOURCE_SEED)
    ids.add_argument("--condition", help="print only this condition's id (an error unless exactly one exists)")
    targets = sub.add_parser("targets", help="C: the same condition's instances of seeds 101 to 105")
    targets.add_argument("instance")
    args = parser.parse_args(argv)
    if args.command == "targets":
        print(" ".join(transfer_targets(args.instance)))
        return 0
    if args.condition is not None:
        print(find(args.condition, args.seed))
        return 0
    print(json.dumps(condition_ids(args.seed), indent=1))
    return 0


__all__ = ["BATCHNORM_SUFFIXES", "CAPABILITY_SENTENCES", "CLASS_TOKEN_NAMES", "CONDITIONS", "CONDITION_RULES",
           "DATASET", "HIDDEN_CHECK", "LADDER_CONDITION", "LADDER_VARIANTS", "MCP_CONNECT_TIMEOUT_MS",
           "MCP_TOOL_TIMEOUT_MS", "NAMED", "NAMED_NOCHECK", "NAMED_VARIANTS", "OFFICIAL_FILE_SHA256",
           "POSITIONAL_NAMES", "RULE", "RULE_SENTENCES", "SOURCE_SEED", "STANDARD", "TRANSFER_SEEDS", "UNLABELLED",
           "VARIANTS", "E11HarnessError", "check_student", "check_variant", "condition_ids", "condition_rule", "find",
           "instance_ids", "is_e11", "ladder_view", "leak_words", "main", "mcp_timeouts", "self_test_tokens",
           "transfer_targets"]

if __name__ == "__main__":
    raise SystemExit(main())
