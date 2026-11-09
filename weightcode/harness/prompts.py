"""Editor prompts (HARNESS_SPEC section 6): ``prompts/system.md`` for every arm and ``prompts/task_<arm>.md``.

The system prompt is identical across arms and carries the owner-approved wording: the role, what is kept, the
rules and how to work. The one exception is M2NL, which has no ``patch_check``: its ``prompts/system_M2NL.md`` is
``system.md`` with the two passages that state the checked-patch rule (what is kept, and the closing lines on
``patch_check`` and ``declare_final``) replaced by the declared-patch rule; the role, the rules and how to work are
word for word the same. LEARN's ``prompts/system_LEARN.md`` is ``system.md`` with the sentences that are false for an
update program replaced (what is kept, the program's two strings, where programs are written, which file counts, and
the closing lines); every other paragraph is byte-identical. A fewshot-v1 LEARN session (PREREG E2) uses
``prompts/system_LEARN_fewshot.md``: ``system_LEARN.md`` with the two passages that are false for that task replaced
(the role, which says the model errs on deployment data, and the names of the two check sets); every other paragraph
is byte-identical. A seqskill-v1 LEARN session (PREREG E5) uses ``prompts/system_LEARN_seqskill.md``: ``system_LEARN.md``
with the title, the role and the names of the two check sets replaced for a sequence model, and
``prompts/task_LEARN_seqskill.md``: ``task_LEARN.md`` with "classifier" read as "model", the class line naming the
output symbols and one sentence on the targets of the experience; every other line is the same. Its variant
rule_only (PREREG E8: the new operator's rule given in model_spec.json, no row of it in the experience) uses
``prompts/system_LEARN_seqskill_rule.md``: ``system_LEARN_seqskill.md`` with one sentence of Rules extended. "Do not
score more than 8 distinct candidate patches against the evidence labels in total." gains ", and scoring a patch
against answers computed from the new operator's rule counts toward the same limit of 8", because rows the teacher
builds from a given rule are labelled data it could otherwise score without bound (the PREREG E8 training-free rule);
the one source line that held the end of that sentence becomes two and every other byte is the same. It also uses
``prompts/task_LEARN_seqskill_rule.md``: ``task_LEARN_seqskill.md`` with the ``experience.npz`` bullet describing the
one part the file has (the old-operator rows, and no row of the new operator), since the two-part bullet would be
false; every other line is the same. A scanskill-v1 instance (PREREG E13, the seqskill variant scan_rule: commands of
words mapped to action sequences, the interpretation rules in model_spec.json) uses ``prompts/system_LEARN_scan.md``,
``system_LEARN_seqskill_rule.md`` with the role and the names of the check sets written for commands and the Rules
sentence naming the interpretation rules, and ``prompts/task_LEARN_scan.md``, ``task_LEARN_seqskill_rule.md`` with the
action line and an ``experience.npz`` bullet for the primitive row (when the condition has one) and the old rows. A graphnew-v1 LEARN session (PREREG E7: new classes for a graph network whose
inputs are node ids of one citation graph) uses ``prompts/system_LEARN_graphnew.md``: ``system_LEARN_fewshot.md`` with
its two task passages reworded for the graph (the role: a classifier of the nodes of one citation graph, one node per
paper, trained only on labelled nodes of some classes, each input a node id of the graph in ``graph.npz`` and one
forward pass running the whole graph; and the "What is kept" paragraph, which names nodes where fewshot names images
and says the check sets' nodes are never named, since every node is in the graph), each paragraph re-wrapped to
``PROMPT_WIDTH``; the two kept-rule sentences ``retention_kept_section`` replaces, the title and every other paragraph
are byte-identical. It also uses ``prompts/task_LEARN_graphnew.md``: ``task_LEARN.md`` (the task template of a fewshot
session) with two bullets of the file list reworded, the ``forward.py`` bullet (``x`` holds int64 node ids of the graph
in ``graph.npz``, which forward.py reads from its own directory, and every call runs the whole graph) and the end of
the ``experience.npz`` bullet (the inputs ``x_*`` are int64 node ids into ``graph.npz``); every other line is the
same. A session under the retention kept rule (``weightcode.harness.retention``, dev only) has the two
sentences of "What is kept" that state the v1 rule replaced by the retention rule with its floor
(``retention_kept_section``; its unit is nodes on a graphnew-v1 instance); that section is re-wrapped and every other
byte is the template's. An fscil-v1 LEARN session (PREREG E9: one few-shot class-incremental session t of CIFAR-100 on
a lineage's current student) uses ``prompts/system_LEARN_fscil.md``: ``system_LEARN_fewshot.md`` with its two task
passages reworded for the incremental setting, each re-wrapped to ``PROMPT_WIDTH`` (the role: the classifier learns new
classes in sessions, predicts the highest-scoring output among the classes seen so far, was first trained on the base
classes only and carries the earlier sessions' updates; and the "What is kept" paragraph, whose two check sets are the
images of the classes added in this session and the images of the classes seen before, the base classes and those of
the earlier sessions, both scored under the session's prediction rule); the two kept-rule sentences
``retention_kept_section`` replaces, the title and every other paragraph are byte-identical. It also uses
``prompts/task_LEARN_fscil.md``: ``task_LEARN.md`` with two bullets of the file list reworded, the ``forward.py`` bullet
(a logit for every class, and the prediction is the argmax over the outputs of the classes seen so far, as
``model_spec.json`` states under ``prediction``) and the ``experience.npz`` bullet (three parts: the session's shots, the
base replay rows and the shots of the earlier sessions with ``session_prev``, filled from the placeholders ``n_prev``
and ``prev_description``, with the predictions under the session's prediction rule); every other line is the same. The
session number, the classes added now and before, the fact that the earlier sessions' updates are in the classifier and
the prediction rule are in the statement (``weightcode.harness.fscil.fscil_statement``), the class names in the class line,
both filled from the LEARN task.json. The lineage's library (``weightcode.harness.fscil.library_entries``) adds one bullet at
the end of the file list (``fscil_library_files_section``), worded for the earlier sessions of this classifier.
A policyskill-v1 LEARN session (PREREG E14: a new mission for a trained grid-room policy) uses
``prompts/system_LEARN_policyskill.md``: ``system_LEARN.md`` with the title, the role (a policy acting in a grid
room, trained on one mission, whose new mission has untrained instruction words and an action it was never trained
to take) and the "What is kept" paragraph (two check sets of episodes, each run greedily in its own room and right
when its mission is completed) replaced, each re-wrapped to ``PROMPT_WIDTH``; the two kept-rule sentences
``retention_kept_section`` replaces (its unit is episodes) and every other paragraph are byte-identical. It also uses
``prompts/task_LEARN_policyskill.md``: ``task_LEARN.md`` with "classifier" read as "policy", the class line naming
the actions (filled from task.json ``actions``) and the ``experience.npz`` bullet describing the demonstrated steps
(the planner's action, the policy's logits and, for the new part, the demonstration and step of each row); every
other line is the same. No policyskill prompt names an object attribute (``weightcode.harness.leaks.HINT_WORDS``): the
statement refers to the missions' grammars in task.json.

LEARN_BLIND (PREREG A4a) uses ``system_LEARN.md`` unchanged and ``prompts/task_LEARN_BLIND.md``: ``task_LEARN.md``
with three passages replaced. The ``model.npz`` bullet is NOWEIGHT's ``model_shapes.json`` bullet worded for a
program (the update program receives the weights when ``patch_check`` runs it); the ``experience.npz`` bullet is the
placeholder ``experience_bullet``, ``BLIND_EXPERIENCE_BULLET`` (no ``pred_*``) or, when the session keeps the
predictions, ``LEARN_EXPERIENCE_BULLET`` (the bullet of ``task_LEARN.md``), chosen by the task.json experience keys;
and the sentence that says the program's prints are returned is "A check returns" and task.json
``program.check_returns``. Every other line is the same.

The E11 rule view (PREREG K1: the condition's rule given as a formula in the statement, no image of the condition)
uses ``system_LEARN.md`` unchanged and ``prompts/task_LEARN_rule.md``: ``task_LEARN.md`` with the ``experience.npz``
bullet describing the one part the file has (labelled clean examples with the classifier's predictions, and no image
of the new condition), since the two-part bullet would be false; every other line is the same. The rule and the
sentence that no image of the condition is given are in the statement (``weightcode.harness.experience.learn_task_rule``).

The information ladder (PREREG; ``weightcode.harness.e11`` U, K0b and K0a) adds no template file: each view is derived from
``task_LEARN.md`` and ``system_LEARN.md`` when rendered, by exact replacements that must each find their text exactly
once (``LADDER_TASK_EDITS``, ``ladder_task_template``), or the view is refused. U (``unlabelled``) replaces the
``experience.npz`` bullet's opening, which calls every row labelled, and drops ``y_new`` from it
(``UNLABELLED_EXPERIENCE_BULLET``); its ``$new_description`` is task.json's, which in U no longer says how the
condition rows were drawn; its system prompt is E11's. K0b (``named``) removes the ``experience.npz`` bullet
and the experience from the ``task.json`` bullet, and says that ``patch_check`` and the program run on this classifier
(not "and its experience"); its system prompt is E11's. K0a (``named_nocheck``) is K0b without ``patch_check``: the
tool list, the budget lines, the readout section and the sentences on what a check runs are replaced by what
``declare_final`` does (it runs the program and validates its update within the limits, and scores nothing), and its
system prompt (``nocheck_kept_section``) replaces "What is kept" by the post-session rule (the last valid
declaration, scored once on held-out images, kept under the session's floor), the sentence on which file counts and
the closing lines (M2NL's form). No K0a prompt or tool text says "check" (``render_task`` and ``render_system``
refuse one that does).

Each task template
adds only facts, filled from the arm's task.json and the session budgets: the statement, the class names, the file list, the tools, the budgets, the readout and final-selection
text of task.json, its training-free rule, the limits and ``describe_ops()`` for the limits that apply. Outside
the file list and the tool list the task templates are word-for-word identical (DECOY's equals M2's), and every
op list comes from task.json and ``weightcode.patch.describe_ops``, never from this module. ``task_LEARN.md`` has no
patch format: it is filled from the LEARN task.json (``weightcode.harness.experience.learn_task``: the experience, the
program interface, the joint readout, the program limits and the training-free rule with its program part).
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from weightcode.harness import seqskill as seqskill_variants
from weightcode.harness.arms import ARMS, arm_spec
from weightcode.patch import Limits, describe_ops

PROMPT_DIR = Path(__file__).resolve().with_name("prompts")
SYSTEM_TEMPLATE = "system.md"
# arms whose final-patch rule differs from the shared wording
ARM_SYSTEM_TEMPLATES = {"M2NL": "system_M2NL.md", "LEARN": "system_LEARN.md", "LEARN_BLIND": "system_LEARN.md"}
FEWSHOT_SYSTEM_TEMPLATE = "system_LEARN_fewshot.md"  # a program arm on a fewshot-v1 instance (PREREG E2)
SEQSKILL_SYSTEM_TEMPLATE = "system_LEARN_seqskill.md"  # a program arm on a seqskill-v1 instance (PREREG E5)
SEQSKILL_TASK_TEMPLATE = "task_LEARN_seqskill.md"
# the seqskill-v1 rule_only variant (PREREG E8): the rule given as knowledge, no example of it
SEQSKILL_RULE_SYSTEM_TEMPLATE = "system_LEARN_seqskill_rule.md"
SEQSKILL_RULE_TASK_TEMPLATE = "task_LEARN_seqskill_rule.md"
# the seqskill variant scan_rule (PREREG E13): a scanskill-v1 instance, the interpretation rules given, no target row
SCAN_SYSTEM_TEMPLATE = "system_LEARN_scan.md"
SCAN_TASK_TEMPLATE = "task_LEARN_scan.md"
GRAPHNEW_SYSTEM_TEMPLATE = "system_LEARN_graphnew.md"  # a program arm on a graphnew-v1 instance (PREREG E7)
GRAPHNEW_TASK_TEMPLATE = "task_LEARN_graphnew.md"
FSCIL_SYSTEM_TEMPLATE = "system_LEARN_fscil.md"  # a program arm on an fscil-v1 instance (PREREG E9)
FSCIL_TASK_TEMPLATE = "task_LEARN_fscil.md"
POLICYSKILL_SYSTEM_TEMPLATE = "system_LEARN_policyskill.md"  # a program arm on a policyskill-v1 instance (PREREG E14)
POLICYSKILL_TASK_TEMPLATE = "task_LEARN_policyskill.md"
# the E11 rule view (PREREG K1): the condition's rule given as a formula, no image of it; system_LEARN.md unchanged
E11_RULE_TASK_TEMPLATE = "task_LEARN_rule.md"
# The v1 kept rule as the LEARN system templates state it, sentence by sentence (``retention_kept_section``).
KEPT_HEADING = "## What is kept\n\n"
KEPT_RULE_SENTENCE = ("After the session, among the programs you checked with `patch_check`, the update of the one "
                      "with the highest score = score_new + score_old is kept.")
KEPT_BASE_SENTENCE = "If no checked program has a score > 0, the original model is kept."
PROMPT_WIDTH = 116  # the wrap width of the LEARN system templates
# The experience.npz bullet of task_LEARN_BLIND.md (PREREG A4a): without the predictions (the default), or with them,
# as in task_LEARN.md; the placeholders are those of the LEARN task template.
BLIND_EXPERIENCE_BULLET = (
    "- `experience.npz`: labelled examples, in two parts. `x_new`, `y_new`: $n_new $new_description. `x_old`,\n"
    "  `y_old`: $n_old $old_description. `split_ids_new` and `split_ids_old` are opaque ids, and the inputs `x_*` are\n"
    "  in the format that `model_spec.json` describes. The classifier's predictions on these examples are not in the\n"
    "  file."
)
LEARN_EXPERIENCE_BULLET = (
    "- `experience.npz`: labelled examples with the classifier's predictions, in two parts. `x_new`, `y_new`, "
    "`pred_new`:\n"
    "  $n_new $new_description. `x_old`, `y_old`, `pred_old`: $n_old $old_description. `pred_*` are the classifier's\n"
    "  predictions, `split_ids_new` and `split_ids_old` are opaque ids, and the inputs `x_*` are in the format that\n"
    "  `model_spec.json` describes."
)
# The information ladder (module docstring): the exact replacements that derive each view's task template from
# task_LEARN.md, applied in order, each of whose old text must occur exactly once.
UNLABELLED_EXPERIENCE_BULLET = (
    "- `experience.npz`: examples with the classifier's predictions, in two parts; only the clean part is labelled.\n"
    "  `x_new`, `pred_new`: $n_new $new_description. `x_old`, `y_old`, `pred_old`: $n_old $old_description. "
    "`pred_*` are\n"
    "  the classifier's predictions, `split_ids_new` and `split_ids_old` are opaque ids, and the inputs `x_*` are in "
    "the\n"
    "  format that `model_spec.json` describes."
)
LEARN_TASK_JSON_BULLET = (
    "- `task.json`: this task as data, with `parent_sha256`, the experience, the program interface, the limits and "
    "their\n"
    "  definitions, the patch-check readout, the final selection and the training-free rule."
)
LEARN_CHECK_TOOL = (
    "- `patch_check`: run one update program file from /work/patches on this classifier and its experience, apply its\n"
    "  update and score the updated classifier on the two held-out check sets."
)
LEARN_PROGRAM_TOOLS = (
    LEARN_CHECK_TOOL + "\n"
    "- `status`: queries and cells remaining, and the scores of every program checked so far.\n"
    "- `declare_final`: record the program you want kept, and why."
)
LEARN_CHECK_RUNS = (
    "Each `patch_check` call runs the program once, in a fresh sandbox, on this classifier and its experience: $runs.\n"
    "The returned deltas are validated (names, shapes, dtypes, finite values), checked against the limits below and\n"
    "applied; then the updated classifier is scored. What the program prints is returned with the result."
)
LADDER_TASK_EDITS: dict[str, tuple[tuple[str, str], ...]] = {
    "unlabelled": ((LEARN_EXPERIENCE_BULLET, UNLABELLED_EXPERIENCE_BULLET),),
    "named": (
        (LEARN_EXPERIENCE_BULLET + "\n", ""),
        (LEARN_TASK_JSON_BULLET, (
            "- `task.json`: this task as data, with `parent_sha256`, the program interface, the limits and their "
            "definitions,\n  the patch-check readout, the final selection and the training-free rule.")),
        (LEARN_CHECK_TOOL, (
            "- `patch_check`: run one update program file from /work/patches on this classifier, apply its update and "
            "score\n  the updated classifier on the two held-out check sets.")),
        ("in a fresh sandbox, on this classifier and its experience: $runs.", "in a fresh sandbox, on this classifier: "
                                                                               "$runs."),
    ),
    "named_nocheck": (
        (LEARN_EXPERIENCE_BULLET + "\n", ""),
        (LEARN_TASK_JSON_BULLET, (
            "- `task.json`: this task as data, with `parent_sha256`, the program interface, the limits and their "
            "definitions,\n  the final selection and the training-free rule.")),
        (LEARN_PROGRAM_TOOLS, (
            "- `status`: cells remaining, and your last declaration.\n"
            "- `declare_final`: declare the program you want kept, and why; it is run once on this classifier and its "
            "update is\n  validated within the limits, but not scored.")),
        ("- `patch_check`: $budget_queries queries in total ($invalid_programs).\n", ""),
        ("- A program run that takes longer than $program_wall_s seconds is stopped; it uses no query.",
         "- A program run that takes longer than $program_wall_s seconds is stopped, and its declaration is not "
         "valid."),
        ("A stopped session keeps the final selection below over the\n  programs checked before the stop,",
         "A stopped session keeps the final selection below over the\n  declarations made before the stop,"),
        ("## Patch-check readout\n\n$readout\n\n", ""),
        (LEARN_CHECK_RUNS, (
            "Each `declare_final` call with a program runs it once, in a fresh sandbox, on this classifier: $runs.\n"
            "The returned deltas are validated (names, shapes, dtypes, finite values), held to the limits below and\n"
            "applied; the updated classifier is not scored. What the program prints is returned with the result.")),
    ),
}
# K0a's system prompt (``nocheck_kept_section``): "What is kept" of system_LEARN.md as it stands, and what replaces it.
LEARN_KEPT_BODY = (
    "After the session, among the programs you checked with `patch_check`, the update of the one with the highest "
    "score = score_new + score_old is kept. Each part is fixed - broken on its own held-out check set, one of "
    "new-condition images (score_new) and one of clean images (score_old). fixed counts check-set images the original "
    "model got wrong and the updated model gets right; broken counts check-set images the original model got right and "
    "the updated model gets wrong. The check sets are held out and you never see their images. If no checked program "
    "has a score > 0, the original model is kept. Unchecked programs are never kept."
)
NOCHECK_KEPT_BODY = (
    "After the session, the update of the program named in your last valid `declare_final` call (the update computed "
    "when you declared it) is scored once against the original model on two held-out sets, one of new-condition images "
    "(score_new) and one of clean images (score_old). Each part is fixed - broken: fixed counts held-out images the "
    "original model got wrong and the updated model gets right; broken counts held-out images the original model got "
    "right and the updated model gets wrong. The update is kept if its score_old is at least {floor} (at most "
    "{allowance} more clean held-out images broken than fixed) and its score = score_new + score_old is > 0; "
    "otherwise the original model is kept. You never see these images, and no score is shown to you. If you make no "
    "valid declaration, or declare null, the original model is kept."
)
NOCHECK_SYSTEM_EDITS = (
    ("A program counts only once you pass its file name to `patch_check`.",
     "A program counts only once you pass its file name to `declare_final`."),
    ("When you have finished, call `declare_final` with the program you want kept (or null for the original model) "
     "and a\nshort rationale, then stop. The declaration is recorded as your own choice; it does not change which "
     "update is kept\nunder the rule above.",
     "When you have finished, call `declare_final` with the one program you want kept (or null for the original model) "
     "and\na short rationale, then stop."),
)


def _rule_only(seqskill: bool, seqskill_variant: str | None) -> bool:
    """True for the seqskill-v1 rule_only variant; a variant other than the standard one needs ``seqskill``."""
    try:
        variant = seqskill_variants.check_variant(seqskill_variant)
    except seqskill_variants.SeqskillHarnessError as exc:
        raise ValueError(str(exc)) from None
    if variant != seqskill_variants.STANDARD and not seqskill:
        raise ValueError(f"the seqskill variant {variant!r} applies to a seqskill-v1 session only")
    return variant == seqskill_variants.RULE_ONLY


def _scan(seqskill: bool, seqskill_variant: str | None) -> bool:
    """True for the scan_rule variant (a scanskill-v1 instance, PREREG E13); it needs ``seqskill``."""
    _rule_only(seqskill, seqskill_variant)
    return seqskill_variants.check_variant(seqskill_variant) == seqskill_variants.SCAN


def _one_rule(fewshot: bool, seqskill: bool, graphnew: bool, fscil: bool = False, policyskill: bool = False) -> None:
    if fewshot + seqskill + graphnew + fscil + policyskill > 1:
        raise ValueError("an instance follows fewshot-v1, seqskill-v1, graphnew-v1, fscil-v1 or policyskill-v1, not "
                         "both")


def system_template_name(arm: str | None = None, fewshot: bool = False, seqskill: bool = False,
                         seqskill_variant: str | None = None, graphnew: bool = False, fscil: bool = False,
                         policyskill: bool = False) -> str:
    """The system template of ``arm``: ``system.md`` unless the arm has its own (``ARM_SYSTEM_TEMPLATES``);
    ``fewshot`` (a program arm on a fewshot-v1 instance) gives ``FEWSHOT_SYSTEM_TEMPLATE``, ``seqskill`` (a program
    arm on a seqskill-v1 instance) ``SEQSKILL_SYSTEM_TEMPLATE``, or ``SEQSKILL_RULE_SYSTEM_TEMPLATE`` for the
    rule_only ``seqskill_variant``, and ``graphnew`` (a program arm on a graphnew-v1 instance)
    ``GRAPHNEW_SYSTEM_TEMPLATE``, and ``fscil`` (a program arm on an fscil-v1 instance) ``FSCIL_SYSTEM_TEMPLATE``,
    and ``policyskill`` (a program arm on a policyskill-v1 instance) ``POLICYSKILL_SYSTEM_TEMPLATE``; the scan_rule
    ``seqskill_variant`` gives ``SCAN_SYSTEM_TEMPLATE``."""
    rule_only = _rule_only(seqskill, seqskill_variant)
    scan = _scan(seqskill, seqskill_variant)
    _one_rule(fewshot, seqskill, graphnew, fscil, policyskill)
    if policyskill:
        if arm is None or not arm_spec(arm).program:
            raise ValueError(f"a policyskill-v1 session runs in a program arm only, not {arm!r}")
        return POLICYSKILL_SYSTEM_TEMPLATE
    if fscil:
        if arm is None or not arm_spec(arm).program:
            raise ValueError(f"an fscil-v1 session runs in a program arm only, not {arm!r}")
        return FSCIL_SYSTEM_TEMPLATE
    if graphnew:
        if arm is None or not arm_spec(arm).program:
            raise ValueError(f"a graphnew-v1 session runs in a program arm only, not {arm!r}")
        return GRAPHNEW_SYSTEM_TEMPLATE
    if seqskill:
        if arm is None or not arm_spec(arm).program:
            raise ValueError(f"a seqskill-v1 session runs in a program arm only, not {arm!r}")
        if scan:
            return SCAN_SYSTEM_TEMPLATE
        return SEQSKILL_RULE_SYSTEM_TEMPLATE if rule_only else SEQSKILL_SYSTEM_TEMPLATE
    if fewshot:
        if arm is None or not arm_spec(arm).program:
            raise ValueError(f"a fewshot-v1 session runs in a program arm only, not {arm!r}")
        return FEWSHOT_SYSTEM_TEMPLATE
    if arm is None:
        return SYSTEM_TEMPLATE
    arm_spec(arm)
    return ARM_SYSTEM_TEMPLATES.get(arm, SYSTEM_TEMPLATE)


def _e11_ladder(e11_variant: str | None, arm: str, *others: bool) -> str | None:
    """The ladder variant (U, K0b, K0a) of ``e11_variant``, a LEARN view of an evidence-v1 instance only, or None."""
    from weightcode.harness.e11 import LADDER_VARIANTS, check_variant

    variant = check_variant(e11_variant)
    if variant not in LADDER_VARIANTS:
        return None
    if arm != "LEARN" or any(others):
        raise ValueError(f"the E11 ladder view {variant!r} is a LEARN view of an evidence-v1 instance, not {arm!r} on "
                         "another rule")
    return variant


def _replace_exactly_once(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{what} does not hold {old[:60]!r} exactly once, so the view cannot be derived from it")
    return text.replace(old, new)


def ladder_task_template(variant: str, text: str) -> str:
    """The task template of ladder view ``variant``: ``text`` (task_LEARN.md) with ``LADDER_TASK_EDITS[variant]``
    applied in order, each found exactly once."""
    for old, new in LADDER_TASK_EDITS[variant]:
        text = _replace_exactly_once(text, old, new, f"the LEARN task template ({variant} view)")
    return text


def nocheck_kept_section(text: str, s_old_floor: int) -> str:
    """K0a's system prompt: ``text`` (system_LEARN.md) with its "What is kept" section, which must state the v1 rule
    word for word, replaced by ``NOCHECK_KEPT_BODY`` with floor ``s_old_floor`` (re-wrapped to ``PROMPT_WIDTH``), and
    ``NOCHECK_SYSTEM_EDITS`` applied, each found exactly once. The result must not say "check"."""
    if s_old_floor > 0:
        raise ValueError(f"a retention floor is at most 0, not {s_old_floor}")
    start = text.index(KEPT_HEADING) + len(KEPT_HEADING)
    end = text.index("\n\n## ", start)
    if " ".join(text[start:end].split()) != LEARN_KEPT_BODY:
        raise ValueError("the What is kept section does not state the v1 kept rule word for word")
    body = NOCHECK_KEPT_BODY.format(floor=s_old_floor, allowance=-s_old_floor)
    text = text[:start] + textwrap.fill(body, width=PROMPT_WIDTH, break_long_words=False,
                                        break_on_hyphens=False) + text[end:]
    for old, new in NOCHECK_SYSTEM_EDITS:
        text = _replace_exactly_once(text, old, new, "the LEARN system template (named_nocheck view)")
    if "check" in text.lower():
        raise ValueError("the K0a system prompt still says 'check'")
    return text


def _e11_rule(e11_variant: str | None, arm: str, *others: bool) -> bool:
    """True for the E11 rule view (PREREG K1), which is a LEARN view of an evidence-v1 instance only."""
    from weightcode.harness.e11 import RULE, check_variant

    if check_variant(e11_variant) != RULE:
        return False
    if arm != "LEARN" or any(others):
        raise ValueError(f"the E11 rule view is a LEARN view of an evidence-v1 instance, not {arm!r} on another rule")
    return True


def task_template_name(arm: str, seqskill: bool = False, seqskill_variant: str | None = None,
                       graphnew: bool = False, fscil: bool = False, e11_variant: str | None = None,
                       policyskill: bool = False) -> str:
    """``task_<arm>.md``; ``seqskill`` (a program arm on a seqskill-v1 instance) gives ``SEQSKILL_TASK_TEMPLATE``, or
    ``SEQSKILL_RULE_TASK_TEMPLATE`` for the rule_only ``seqskill_variant``, and ``graphnew`` (a program arm on a
    graphnew-v1 instance) ``GRAPHNEW_TASK_TEMPLATE`` and ``fscil`` (a program arm on an fscil-v1 instance)
    ``FSCIL_TASK_TEMPLATE``; ``e11_variant`` "rule" (the LEARN arm on an E11 instance, PREREG K1) gives
    ``E11_RULE_TASK_TEMPLATE``; the scan_rule ``seqskill_variant`` gives ``SCAN_TASK_TEMPLATE``; ``policyskill`` (a
    program arm on a policyskill-v1 instance) gives ``POLICYSKILL_TASK_TEMPLATE``."""
    rule_only = _rule_only(seqskill, seqskill_variant)
    scan = _scan(seqskill, seqskill_variant)
    _one_rule(False, seqskill, graphnew, fscil, policyskill)
    if _e11_rule(e11_variant, arm, seqskill, graphnew, fscil, policyskill):
        return E11_RULE_TASK_TEMPLATE
    if policyskill:
        if not arm_spec(arm).program:
            raise ValueError(f"a policyskill-v1 session runs in a program arm only, not {arm!r}")
        return POLICYSKILL_TASK_TEMPLATE
    if fscil:
        if not arm_spec(arm).program:
            raise ValueError(f"an fscil-v1 session runs in a program arm only, not {arm!r}")
        return FSCIL_TASK_TEMPLATE
    if graphnew:
        if not arm_spec(arm).program:
            raise ValueError(f"a graphnew-v1 session runs in a program arm only, not {arm!r}")
        return GRAPHNEW_TASK_TEMPLATE
    if seqskill:
        if not arm_spec(arm).program:
            raise ValueError(f"a seqskill-v1 session runs in a program arm only, not {arm!r}")
        if scan:
            return SCAN_TASK_TEMPLATE
        return SEQSKILL_RULE_TASK_TEMPLATE if rule_only else SEQSKILL_TASK_TEMPLATE
    arm_spec(arm)
    return f"task_{arm}.md"


def template_names() -> list[str]:
    return [SYSTEM_TEMPLATE, *ARM_SYSTEM_TEMPLATES.values(), FEWSHOT_SYSTEM_TEMPLATE, SEQSKILL_SYSTEM_TEMPLATE,
            SEQSKILL_RULE_SYSTEM_TEMPLATE, GRAPHNEW_SYSTEM_TEMPLATE, *(task_template_name(arm) for arm in ARMS),
            SEQSKILL_TASK_TEMPLATE, SEQSKILL_RULE_TASK_TEMPLATE, GRAPHNEW_TASK_TEMPLATE, FSCIL_SYSTEM_TEMPLATE,
            FSCIL_TASK_TEMPLATE, E11_RULE_TASK_TEMPLATE, SCAN_SYSTEM_TEMPLATE, SCAN_TASK_TEMPLATE,
            POLICYSKILL_SYSTEM_TEMPLATE, POLICYSKILL_TASK_TEMPLATE]


def template_sha256s() -> dict[str, str]:
    """sha256 of every prompt template file."""
    return {name: hashlib.sha256((PROMPT_DIR / name).read_bytes()).hexdigest() for name in template_names()}


@dataclass(frozen=True)
class Budgets:
    """Session budgets shown to the editor."""

    patch_checks: int
    python_cells: int
    cell_timeout_s: float
    wall_timeout_s: float


def retention_kept_section(text: str, s_old_floor: int, check_set: str, unit: str = "images") -> str:
    """``text`` with the two sentences of its "What is kept" section that state the v1 kept rule replaced by the
    retention rule with floor ``s_old_floor`` on the ``check_set`` ``unit`` (images; rows for a sequence model;
    ``weightcode.harness.retention``); the section is re-wrapped to ``PROMPT_WIDTH`` and every byte outside it is kept."""
    if s_old_floor > 0:
        raise ValueError(f"a retention floor is at most 0, not {s_old_floor}")
    start = text.index(KEPT_HEADING) + len(KEPT_HEADING)
    end = text.index("\n\n## ", start)
    body = " ".join(text[start:end].split())
    if body.count(KEPT_RULE_SENTENCE) != 1 or body.count(KEPT_BASE_SENTENCE) != 1:
        raise ValueError("the What is kept section does not state the v1 kept rule sentence by sentence")
    body = body.replace(KEPT_RULE_SENTENCE, (
        f"After the session, among the programs you checked with `patch_check` whose score_old is at least "
        f"{s_old_floor} (at most {-s_old_floor} more {check_set} check-set {unit} broken than fixed), the update of the "
        f"one with the highest score = score_new + score_old is kept."))
    body = body.replace(KEPT_BASE_SENTENCE, (
        f"A program with score_old below {s_old_floor} is never kept, whatever its score. If no checked program with "
        f"score_old of at least {s_old_floor} has a score > 0, the original model is kept."))
    wrapped = textwrap.fill(body, width=PROMPT_WIDTH, break_long_words=False, break_on_hyphens=False)
    return text[:start] + wrapped + text[end:]


def fscil_v2_kept_section(text: str, kept_rule: Mapping[str, Any]) -> str:
    """``text``, the fscil-v1 system template, with the whole body of its "What is kept" section replaced by the
    fscil-retain-v2 statement (``weightcode.harness.fscil.kept_section_v2``, PREREG E9 v2), re-wrapped to ``PROMPT_WIDTH``;
    every byte outside that body is kept. The wrap keeps floor_c's formula on one line and never starts a line with
    an operator (``-``, ``+``, ``>``, ``*``), which Markdown would read as a list or a quote."""
    from weightcode.harness.fscil import KEPT_RULE_V2, kept_section_v2, v2_floor_formula

    if kept_rule.get("name") != KEPT_RULE_V2:
        raise ValueError(f"the v2 kept section states kept rule {KEPT_RULE_V2}, not {kept_rule.get('name')!r}")
    start = text.index(KEPT_HEADING) + len(KEPT_HEADING)
    end = text.index("\n\n## ", start)
    body = " ".join(text[start:end].split())
    if body.count(KEPT_RULE_SENTENCE) != 1 or body.count(KEPT_BASE_SENTENCE) != 1:
        raise ValueError("the What is kept section does not state the v1 kept rule sentence by sentence")
    glue = "\x00"  # not whitespace to textwrap: the words it joins stay on one line
    statement = kept_section_v2(kept_rule)
    formula = f"floor_c = {v2_floor_formula(kept_rule)}"
    if statement.count(formula) != 1 or glue in statement:
        raise ValueError("the v2 kept section does not state floor_c's formula once")
    statement = statement.replace(formula, formula.replace(" ", glue))
    for operator in "-+>*":
        statement = statement.replace(f" {operator}", f"{glue}{operator}")
    wrapped = textwrap.fill(statement, width=PROMPT_WIDTH, break_long_words=False,
                            break_on_hyphens=False).replace(glue, " ")
    return text[:start] + wrapped + text[end:]


LIBRARY_FILES_ANCHOR = "\n\n## Tools\n"  # the end of the file list in every task template


def library_files_section(text: str, n_programs: int, index_file: str, program_template: str) -> str:
    """``text``, a rendered task prompt of a program arm, with one bullet at the end of its file list for a program
    library of ``n_programs`` programs (``weightcode.harness.library``, LEARN v3); every other byte is kept."""
    if n_programs < 1:
        raise ValueError(f"a library holds at least one program, not {n_programs}")
    if text.count(LIBRARY_FILES_ANCHOR) != 1:
        raise ValueError("the task prompt has no single file list ending before its Tools section")
    first, last = program_template.format(k=1), program_template.format(k=n_programs)
    files = f"`{first}`" if n_programs == 1 else f"`{first}` to `{last}`"
    what = "one update program" if n_programs == 1 else f"{n_programs} update programs"
    bullet = (f"- `{index_file}` and {files}: {what} written in earlier sessions for other input conditions on this "
              f"same classifier, none of them for this task's condition, each as it was written. `{index_file}` gives "
              f"for each the scores that its own kept `patch_check` returned there: score_new on that condition's "
              f"held-out check set and score_old on the held-out clean check set, which is the same clean check set "
              f"as this task's.")
    wrapped = textwrap.fill(bullet, width=PROMPT_WIDTH, subsequent_indent="  ", break_long_words=False,
                            break_on_hyphens=False)
    at = text.index(LIBRARY_FILES_ANCHOR)
    return text[:at] + "\n" + wrapped + text[at:]


def fscil_library_files_section(text: str, bullet: str) -> str:
    """``text``, a rendered task prompt of an fscil-v1 session, with ``bullet`` (``weightcode.harness.fscil.library_bullet``)
    wrapped as the last bullet of its file list, where ``library_files_section`` puts the LEARN v3 bullet."""
    if text.count(LIBRARY_FILES_ANCHOR) != 1:
        raise ValueError("the task prompt has no single file list ending before its Tools section")
    if not bullet.startswith("- "):
        raise ValueError("a file-list bullet starts with '- '")
    wrapped = textwrap.fill(bullet, width=PROMPT_WIDTH, subsequent_indent="  ", break_long_words=False,
                            break_on_hyphens=False)
    at = text.index(LIBRARY_FILES_ANCHOR)
    return text[:at] + "\n" + wrapped + text[at:]


def render_system(arm: str | None = None, fewshot: bool = False,
                  kept_rule: Mapping[str, Any] | None = None, seqskill: bool = False,
                  seqskill_variant: str | None = None, graphnew: bool = False, fscil: bool = False,
                  e11_variant: str | None = None, policyskill: bool = False) -> str:
    """The system prompt of ``arm`` (no placeholders; ``system.md`` for every arm but those with their own, the
    fewshot-v1 LEARN variant when ``fewshot``, the seqskill-v1 one when ``seqskill``, of ``seqskill_variant``, the
    graphnew-v1 one when ``graphnew`` and the fscil-v1 one when ``fscil``). ``kept_rule``, a session.json
    ``kept_rule`` record of a program arm, states the retention rule in "What is kept" (``retention_kept_section``;
    its unit is rows on a seqskill-v1 instance, nodes on a graphnew-v1 instance and episodes on a policyskill-v1
    instance, the policyskill-v1 template when ``policyskill``), or on an fscil-v1 session the
    E9 v2 rule fscil-retain-v2 (``fscil_v2_kept_section``). ``e11_variant`` "named_nocheck" (K0a, the LEARN arm on an
    E11 instance) states its post-session rule with the floor of ``kept_rule``, which it requires
    (``nocheck_kept_section``); every other ladder variant has E11's system prompt."""
    name = system_template_name(arm, fewshot, seqskill, seqskill_variant, graphnew, fscil, policyskill)
    text = (PROMPT_DIR / name).read_text(encoding="utf-8")
    if "$" in text:
        raise ValueError(f"{name} must not contain placeholders")
    if _e11_ladder(e11_variant, str(arm), fewshot, seqskill, graphnew, fscil, policyskill) == "named_nocheck":
        if kept_rule is None or kept_rule.get("name") != "joint-retain-v1":
            raise ValueError("the K0a view keeps by joint-retain-v1 after its one post-session check; it needs that "
                             "kept rule")
        return nocheck_kept_section(text, int(kept_rule["s_old_floor"]))
    if kept_rule is not None:
        if arm is None or not arm_spec(arm).program:
            raise ValueError(f"the retention kept rule applies to a program arm only, not {arm!r}")
        from weightcode.harness.fscil import KEPT_RULE_V2

        if kept_rule.get("name") == KEPT_RULE_V2:  # PREREG E9 v2
            if not fscil:
                raise ValueError("kept rule fscil-retain-v2 is for an fscil-v1 session")
            return fscil_v2_kept_section(text, kept_rule)
        text = retention_kept_section(text, int(kept_rule["s_old_floor"]), str(kept_rule["check_set"]),
                                      "rows" if seqskill else "nodes" if graphnew else
                                      "episodes" if policyskill else "images")
    return text


def _number(value: float) -> str:
    return f"{value:g}"


def _bullets(items: Sequence[Any]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _limits_text(limits: Mapping[str, Any], definitions: Mapping[str, Any]) -> str:
    lines = []
    for name in sorted(limits):
        value = limits[name]
        shown = "not applied" if value is None else json.dumps(value)
        definition = definitions.get(name)
        lines.append(f"- `{name}` = {shown}" + (f": {definition}" if definition else ""))
    return "\n".join(lines)


def _input_files_text(names: Sequence[str]) -> str:
    return "".join(
        f"\n- `{name}`: an input file the model needs; `model_spec.json` lists it under `input_files`." for name in names
    )


def _program_values(task: Mapping[str, Any], budgets: Budgets, new_part: bool = True,
                    experience_part: bool = True) -> dict[str, str]:
    """The LEARN placeholders, from the LEARN task.json (``weightcode.harness.experience.learn_task``); ``new_part`` False
    (the seqskill-v1 rule_only view) for an experience without a new part, whose task.json must then have none;
    ``experience_part`` False (the ladder's no-data views) for a task.json without an experience section. A task.json
    without ``patch_check`` (K0a) gives its top-level ``final_selection`` and no readout or query budget."""
    program, check = task["program"], task.get("patch_check")
    student = program["student"]
    if not experience_part:
        if "experience" in task:
            raise ValueError("the no-data task.json has an experience section")
        new: dict[str, str] = {}
    elif new_part:
        experience = task["experience"]
        new = {"n_new": str(int(experience["new"]["count"])),
               "new_description": str(experience["new"]["description"]),
               "n_old": str(int(experience["old"]["count"])),
               "old_description": str(experience["old"]["description"])}
    elif "new" in task["experience"]:
        raise ValueError("the rule_only task.json has a new experience part")
    else:
        experience = task["experience"]
        new = {"n_old": str(int(experience["old"]["count"])),
               "old_description": str(experience["old"]["description"])}
    checks = ({"final_selection": str(task["final_selection"])} if check is None else
              {"budget_queries": str(int(budgets.patch_checks)), "invalid_programs": str(check["invalid_programs"]),
               "readout": str(check["readout"]), "final_selection": str(check["final_selection"])})
    return {
        **new,
        **checks,
        "program_wall_s": _number(float(program["wall_limit_s"])),
        "program_file": str(program["file"]),
        "program_defines": _bullets([f"`{item}`" if item.startswith("update(") else item
                                     for item in program["defines"]]),
        "student_params": str(student["params"]),
        "student_forward": str(student["forward"]),
        "student_layout": str(student["layout"]),
        "experience_arrays": str(program["experience"]),
        "returns": str(program["returns"]),
        "runs": str(program["runs"]),
        "transfer": str(program["transfer"]),
        "program_rule": _bullets(task["training_free_rule"]["program"]),
    }


def _scan_prim_bullet(task: Mapping[str, Any]) -> str:
    """The scan_rule task prompt's sentence on the primitive part of the experience, from task.json ``experience.prim``
    (keys, count and description), or nothing when the condition has no primitive row."""
    prim = task["experience"].get("prim")
    if prim is None:
        return ""
    keys = ", ".join(f"`{key}`" for key in prim["keys"])
    count = int(prim["count"])
    return f" {keys}: {count} row{'' if count == 1 else 's'}: {prim['description']}."


def _blind_values(task: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, str]:
    """The LEARN_BLIND placeholders: the experience bullet, chosen by whether the task.json experience keys hold the
    predictions and filled from ``values``, and ``check_returns`` from task.json ``program.check_returns``."""
    keys = [*task["experience"]["new"]["keys"], *task["experience"]["old"]["keys"]]
    shown = {"pred_new", "pred_old"} & set(keys)
    if shown not in (set(), {"pred_new", "pred_old"}):
        raise ValueError(f"a blind task.json shows both predictions or neither, not {sorted(shown)}")
    bullet = LEARN_EXPERIENCE_BULLET if shown else BLIND_EXPERIENCE_BULLET
    return {"experience_bullet": Template(bullet).substitute(values),
            "check_returns": str(task["program"]["check_returns"])}


def _check_ladder_text(variant: str, text: str, what: str) -> None:
    """A ladder view's text never names what the view withholds: y_new and which condition rows are errors (U,
    ``weightcode.harness.experience.UNLABELLED_ERROR_WORDS``); an experience file, part or label (K0b,
    K0a, ``weightcode.harness.experience.NAMED_EXPERIENCE_WORDS``); a check (K0a)."""
    from weightcode.harness.experience import NAMED_EXPERIENCE_WORDS, UNLABELLED_ERROR_WORDS

    words = ("y_new", *UNLABELLED_ERROR_WORDS) if variant == "unlabelled" else NAMED_EXPERIENCE_WORDS
    found = [word for word in words if word in text]
    if variant == "named_nocheck" and "check" in text.lower():
        found.append("check")
    if found:
        raise ValueError(f"the {variant} view's {what} names {found}")


def render_task(arm: str, task: Mapping[str, Any], limits: Mapping[str, Any], budgets: Budgets,
                input_files: Sequence[str] = (), seqskill: bool = False, seqskill_variant: str | None = None,
                graphnew: bool = False, fscil: bool = False, e11_variant: str | None = None,
                policyskill: bool = False) -> str:
    """The task prompt of ``arm``, filled from the arm's task.json, the applied limits and the budgets.

    A task.json without ``patch_check`` (the view of an arm without that tool) gives its top-level
    ``final_selection`` and no readout or query budget. A program arm (LEARN) shows the limits of its task.json,
    the ones a program's deltas are held to, and no patch format. ``seqskill_variant`` "rule_only" (a seqskill-v1
    session) renders ``SEQSKILL_RULE_TASK_TEMPLATE`` from a task.json whose experience has the old part alone;
    ``graphnew`` (a graphnew-v1 session) renders ``GRAPHNEW_TASK_TEMPLATE``; ``fscil`` (an fscil-v1 session) renders
    ``FSCIL_TASK_TEMPLATE`` and fills its third experience part from the task.json ``experience.prev``. LEARN_BLIND
    (PREREG A4a) fills its experience bullet and what a check returns (``_blind_values``). ``e11_variant`` "rule"
    (PREREG K1, the LEARN arm on an E11 instance) renders ``E11_RULE_TASK_TEMPLATE`` from a task.json whose experience
    has the old part alone (``weightcode.harness.experience.learn_task_rule``). A ladder ``e11_variant`` (U, K0b, K0a)
    renders ``ladder_task_template`` of ``task_LEARN.md`` from that view's task.json (module docstring); a no-data
    prompt must name no experience file or array, and K0a's must not say "check". ``policyskill`` (a policyskill-v1
    session) renders ``POLICYSKILL_TASK_TEMPLATE``, whose class line names the actions of task.json ``actions``.
    """
    rule_only = _rule_only(seqskill, seqskill_variant)
    scan = _scan(seqskill, seqskill_variant)
    e11_rule = _e11_rule(e11_variant, arm, seqskill, graphnew, fscil, policyskill)
    ladder = _e11_ladder(e11_variant, arm, seqskill, graphnew, fscil, policyskill)
    text = (PROMPT_DIR / task_template_name(arm, seqskill, seqskill_variant, graphnew, fscil,
                                            e11_variant, policyskill)).read_text(encoding="utf-8")
    template = Template(text if ladder is None else ladder_task_template(ladder, text))
    named = ladder in ("named", "named_nocheck")
    if arm_spec(arm).program:
        rule = task["training_free_rule"]
        shown = task["limits"]
        if any(limits.get(name) != value for name, value in shown.items()):
            raise ValueError("the LEARN task.json limits differ from the applied limits")
        values = {
            "statement": str(task["statement"]),
            "classes": json.dumps(list(task["actions" if policyskill else "classes"])),
            "input_files": _input_files_text(input_files),
            "max_cells": str(int(budgets.python_cells)),
            "cell_timeout_s": _number(budgets.cell_timeout_s),
            "wall_minutes": _number(round(budgets.wall_timeout_s / 60.0, 1)),
            "forbidden": _bullets(rule["forbidden"]),
            "allowed": _bullets(rule["allowed"]),
            "audit": str(rule["audit"]),
            "limits": _limits_text(shown, task.get("limit_definitions", {})),
            **_program_values(task, budgets, new_part=not (rule_only or e11_rule or scan), experience_part=not named),
            **({"n_prev": str(int(task["experience"]["prev"]["count"])),
                "prev_description": str(task["experience"]["prev"]["description"])} if fscil else {}),
            **({"prim_bullet": _scan_prim_bullet(task)} if scan else {}),
        }
        if arm_spec(arm).blind:
            values.update(_blind_values(task, values))
        rendered = template.substitute(values)
        if ladder is not None:
            _check_ladder_text(ladder, rendered, "task prompt")
        return rendered
    rule = task["training_free_rule"]
    applied = Limits(**dict(limits))
    check = task.get("patch_check")
    if check is None:
        checks: dict[str, str] = {"final_selection": str(task["final_selection"])}
    else:
        checks = {"budget_queries": str(int(budgets.patch_checks)),
                  "invalid_patches": str(check["invalid_patches"]),
                  "readout": str(check["readout"]), "final_selection": str(check["final_selection"])}
    values = {
        "statement": str(task["statement"]),
        "classes": json.dumps(list(task["classes"])),
        "input_files": _input_files_text(input_files),
        **checks,
        "max_cells": str(int(budgets.python_cells)),
        "cell_timeout_s": _number(budgets.cell_timeout_s),
        "wall_minutes": _number(round(budgets.wall_timeout_s / 60.0, 1)),
        "forbidden": _bullets(rule["forbidden"]),
        "allowed": _bullets(rule["allowed"]),
        "audit": str(rule["audit"]),
        "limits": _limits_text(limits, task.get("limit_definitions", {})),
        "allowed_ops": ", ".join(f"`{op}`" for op in task["patch_format"]["allowed_ops"]),
        "patch_format": describe_ops(applied).strip(),
    }
    return template.substitute(values)
