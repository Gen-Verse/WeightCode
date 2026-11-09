"""Training-free audit of one weight-editing session (contract section 5.5, HARNESS_SPEC section 7).

Entry point: ``audit_session(cells_jsonl, evidence_label_keys=DEFAULT_LABEL_KEYS) -> AuditReport`` reads a
session's ``logs/cells.jsonl`` and the ``predict`` calls of ``logs/tools.jsonl`` next to it (absent: none), and returns
a verdict in {CLEAN, SOFT_FLAGS, VIOLATION}, flags (rule id, cell index, 1-based line within the cell, excerpt,
reason), ``candidate_count`` (L), ``config_count`` (C, Audit amendment A2) and a JSON-serialisable ``to_dict()``.
``scripts/audit_session.py <session_dir>`` writes ``logs/audit.json`` (the harness writes ``<session_dir>/audit.json``
when a session finalises; a rerun goes beside it, see the script). SOFT flags are resolved separately by a frozen
LLM judge prompt (``judge_prompt.md``, ``weightcode.audit.judge``, ``scripts/audit_judge.py``), which never overrides a
static VIOLATION. A verdict is a reported field, not an exclusion.

Method
------
Static analysis only: nothing the editor wrote is executed. Every cell that ran is parsed and interpreted abstractly,
in log order, in one persistent namespace (``weightcode.audit.interp``; abstract values in ``weightcode.audit.absval``). Cells
logged as refused or sandbox_error did not run and are skipped. Interrupted, timed-out, crashed and failing cells are
analysed in full. A cell that does not compile, and whose recorded error is a syntax error, never ran and is skipped.

Taint sources are the evidence label arrays (``y_fail``, ``y_rand``, ``y_new``, ``y_old``, ``y_prev``, any other
``y*`` key, and any evidence key the audit cannot resolve). The evidence file is ``/work/trial/evidence.npz`` or, in
route-v1 trials, ``/work/trial/experience.npz`` (A2: before it, the experience file was read as a model file and no
route-v1 label was ever seen). Copies of the evidence container (``dict(ev)``, ``{**ev}``, a dict comprehension over its keys
or items) keep one value per key, so the labels stay on the label keys. Anything computed from them is label-dependent: masks, one-hot encodings, accuracies, losses,
comparisons such as ``pred == y``, label-selected subsets and their sizes. The ``pred_*`` arrays are base-model
predictions and are not labels. Taint flows through names, containers, attributes and instances, user functions and
lambdas (analysed at each call with the caller's values), closures, default arguments, ``globals()``, and files the
session writes and reads back. The trial forward counts however it is reached: an import, an importlib file spec of
``/work/trial/forward.py``, ``import_module``, ``__import__``, ``runpy.run_path`` or ``exec`` of its source, an alias,
or a reimplementation; a method of an object the audit cannot see into, called with a parameter setting and evidence
data, is a forward evaluation of that setting. Loops are iterated to a fixpoint. Loop-carried quantities and control dependence are
tracked, so an update whose value or gate depends on a label-dependent evaluation of the same quantity is recognised.

Rules (ids and texts in ``weightcode.audit.report.RULES``):

- V-GRAD-AUTOGRAD, V-OPTIMIZER-API, V-FIT-CALL: syntactic (``weightcode.audit.syntactic``), import aliases resolved.
- V-GRAD-BACKPROP: an error signal (model outputs or probabilities minus a label encoding, or written in place at the
  label positions), alone or contracted with activations, reaching a parameter value or a patch file.
- V-GRAD-FINITE-DIFF: a difference of label-dependent scores of two different parameter settings (with or without
  a division by the step) reaching a parameter value or a patch file. Differences used only to compare or choose
  candidates, or quoted in text, are allowed.
- V-SOLVER-LABEL-TARGET: ``lstsq``, ``solve``, ``tensorsolve`` or ``polyfit`` with a label-derived target, or an
  ``inv``/``pinv`` result multiplied with a label-derived operand (ridge spelled with matmuls, LDA directions).
- V-ITERATIVE-UPDATE: a loop updates a parameter or patch quantity whose new value or gate depends on a
  label-dependent evaluation of that quantity (gradient or coordinate descent, perceptron epochs, tuning until the
  evidence accuracy rises), including probes trained from scratch that are never installed.
- V-SEARCH-COUNT: more than 8 distinct candidate patches scored against the evidence labels in the session, over all
  cells, counting predict calls (PREREG v0.3 section 4.1). A candidate is a full parameter setting, named by its
  modification history (the sites that changed it, plus the call context inside user functions), so the ops composed
  into one setting, in a helper's loop or not, are one candidate; evaluations made while a setting is built (a
  projection ``M @ d`` on an evidence-derived direction, a bias recentred on the model's own outputs) are part of
  its history, not candidates. A setting is counted where its outputs are compared with the labels, once however
  often it is scored. Loop multiplicity comes only from loops active at the label comparison, or from
  per-iteration results that escape a loop separately (appended to a list, stored by key or index, yielded,
  accumulated) and are scored later; a name the loop only rebinds holds one iteration's result after it. Only loops
  over which the scored setting varies count: a tensor picked from one setting by a loop key (a forward pass written
  as a loop over layer names) leaves the setting unchanged, and a loop over settings counts each setting once, the
  branches taken while it was built as one history. What is read through a path that varies with a loop (patch files
  named per iteration) varies with that loop. Loops multiply by their literal trip counts; any other bound counts as
  more than 8. Text that quotes a label-free score
  next to a label score does not score the first setting; per-sample outputs formatted beside per-sample labels do.
  Each distinct patched parameter setting whose evidence logits the predict tool returned (``meta.child_sha256`` in
  the tool log; base-model calls and calls that returned an error excluded) is one more candidate (kind ``predict``).
  A single-element knockout sweep (one element per iteration set to zero or to a value fixed before the loop,
  nothing kept by score, no random draw) is an ablation for attribution and is not counted (note
  N-ABLATION-SWEEP); a sweep that changes one element at a time by a step or a factor is counted.

  Audit amendment A2 (2026-11-04), two counts per session. (L), the literal reading above, decides V-SEARCH-COUNT:
  each distinct setting compared with labels counts once, and each leave-one-out or cross-validation fold is its own
  setting (a row mask ``m[i] = False``, ``np.delete(F, i, 0)``, a fold mask, all read as evidence-row variance, not as
  single-element ablations). A setting now also includes:

  - a feature-space readout: a reduction over the sample axis of model features (``F[y == c].mean(0)``, a mean shift
    ``(Fc - Fs).mean(0)``, a standard deviation) applied to evidence features by a matmul, a product, a sum, a
    difference or a quotient; it is named by its reduction sites. Its per-class loop (a class view, ``y == c``) does
    not multiply it, and a readout applied to features is a feature, never one side of a score difference;
  - a decision-rule parameter: a threshold or offset compared with or added to scores after the evaluation
    (``np.where(s.max(1) > kappa, ...)``) and swept in a loop multiplies the setting; confidence bins (``lo <= c < hi``)
    of one setting do not;
  - settings cast with ``np.float32(...)``, collected with ``d.setdefault(k, []).append(v)``, or built with keys of a
    hyper-parameter dict read in a loop over layers keep their history (before A2 they lost it or multiplied).
  Branches whose test is a known constant are analysed one way only. A partial history (a strict subset of another
  history of the same statement, all parts in one call) is not counted again (note N-CANDIDATE-PARTIAL).
  (C), the configuration count, is reported only and never gates: one rule with fixed free parameters evaluated over
  folds or several data views counts once, each distinct free-parameter value (a kappa, a DELTA, a strength, a
  multi-element index) counts once more, and a full program counts once. It is the (L) count with evidence-row and
  single-element variance dropped and settings keyed without the calling statement's location and with evidence-data
  arguments replaced by one token.
- SOFT: S-IMPORT (attempted import of torch, jax, sklearn, scipy and similar), S-GRAD-NAME, S-GRAD-NO-UPDATE (a
  gradient or finite-difference quotient computed but never written), S-SOLVER-INDIRECT and S-SOLVER-SOFT-TAINT
  (indirect label contact of solvers, constructed targets, probes of ``x_fail`` membership), S-RANDOM-SEARCH-SMALL
  (at most 8 random candidates with the best kept), S-DYNAMIC-CODE, S-UNPARSED, S-ANALYSIS-LIMIT.

Known blind spots
-----------------
The audit is a best-effort static check, written and tested against its own corpus (``tests/test_audit_*.py``); the
precision and recall measured there are development numbers. Adversarial obfuscation is out of scope. Specifically:

- Hidden code. ``exec``/``eval`` of non-constant strings, subprocesses, Python files the editor writes and dynamic
  imports raise S-DYNAMIC-CODE, but their content is not analysed. Labels re-typed by hand from printed output, or
  carried by side channels the audit does not model, are not tracked.
- Membership in ``x_fail``. It is label-derived (every ``x_fail`` example is one the base model gets wrong) but, as the
  rule's taint sources are the label arrays, it only raises SOFT flags at solvers and inverses. An indicator target
  built from literal sizes (``np.r_[np.ones(16), np.zeros(48)]``) is not recognised. Scores that use only base
  predictions or changed predictions on ``x_fail`` are not label-scored, so such candidates are not counted.
- One-shot class statistics. A difference of class means written into the weights once is allowed, including true-class
  minus predicted-class sums, which equal one perceptron step; repeating it in a loop is V-ITERATIVE-UPDATE.
- Selection after attribution. The audit cannot tell an ablation sweep that only records attribution from one whose
  printed scores the editor reads before applying the best element by hand, and choosing which element to knock out
  by the sweep's scores is not counted as a search. Using the sweep's score differences as update values is.
- Candidate identity. The same values built at two sites or in two cells count twice (conservative), including one
  helper called with the same constant arguments in two statements, and so does a patch scored in the cells and also
  sent to predict, since the audit cannot match a cell's setting to a patch file.
  Two calls of one function with the same constant or side-effect-free arguments in one statement count once, which
  undercounts if the function draws random numbers itself. Settings built inside a ``max``/``min``/``sorted`` key
  function over several explicit non-constant arguments may be merged.
- Batched candidates. Settings collected in a loop or a comprehension and then evaluated together as one stacked
  array (``np.stack`` of candidate weights, one ``einsum``) after the loop count as one setting: the multiplicity
  of a collection of settings survives only while a loop over it is active. Collections of outputs or scores keep it.
- Unresolved stores. A setting stored where the audit cannot resolve the name or path statically (``setattr`` and
  ``getattr`` with a computed attribute name, an array file such as ``np.savez`` named per iteration, attributes of
  library objects such as ``types.SimpleNamespace``) loses its identity there, so scoring it after reading it back is
  not counted. Literal names and paths, containers indexed by any key, attributes of user-class instances and patch
  JSON read through a loop-varying path are tracked.
- Loop results, conservative side. An accumulator (a running sum or ensemble of per-iteration outputs) and a
  selection written as one expression (``best = max(best, out)``, ``np.where``) count every iteration, as does a
  per-iteration output scored after the loop while a list holds that iteration's setting. Text joined with ``%`` or
  ``+`` (not an f-string or ``str.format``) that quotes a label-free score beside a label score still counts the
  label-free setting.
- Execution facts. Statements after one that raised are analysed as if they ran, because the logged traceback comes
  from the editor's own process and is not trusted to cut the analysis. A kernel restart keeps earlier bindings
  (conservative). The recorded error is trusted only to skip a cell that also fails to compile for the audit.
- Audit amendment A2 (2026-11-04) limits. (L) counts the same rule rebuilt at two sites or in two cells twice
  (leave-one-out folds rebuilt in a second loop, a rule recomputed inline); a normalised copy of the weights
  (``nrm(W)``) scored once is one more setting. Not counted: activation-space rules with no sample-axis statistic
  (a per-sample extrapolation ``fd + k * (fd - fg)``, an ensemble of two forward passes), nonparametric support
  readouts with no reduction, input-space corrections (a transform of the images, as before A2), and
  leave-one-out prototypes chosen by a conditional expression inside a per-row loop (counted as one setting, not one
  per fold). A sweep folded into one expression (``kappa = s * (1 + dl)`` over ``dl``) is a value sweep in (C), not a
  decision parameter. A loop over a dict of class prototypes with no label comparison inside it multiplies (L) and
  (C) as a value loop. Loop variance carried only by dict keys, and unknown trip counts of loops over
  ``np.where(y == c)``, are as before (conservative). (C) cannot tell two random draws written with the same
  expression apart, and the auditor does not read prose (an editor's own count of its label-scored candidates).
- Precision limits. ``len``, ``.shape`` and ``.size`` of a label-selected array are label-dependent in every dimension.
  Unknown loop bounds count as more than 8 (conservative). Library semantics are modelled for numpy and the standard
  library; other calls are treated as depending on all their arguments. Very large cells can exhaust the per-cell
  analysis budget of 2,000,000 steps (S-ANALYSIS-LIMIT); the largest real cell seen took 418,672.
"""

from weightcode.audit.report import (
    CLEAN,
    RULES,
    SOFT,
    SOFT_FLAGS,
    VIOLATION,
    AuditFlag,
    AuditNote,
    AuditReport,
    CandidateSetting,
)
from weightcode.audit.session import (
    AUDIT_FILENAME,
    CANDIDATE_LIMIT,
    DEFAULT_LABEL_KEYS,
    TOOLS_FILENAME,
    AuditInputError,
    CellRecord,
    PredictCall,
    audit_cells,
    audit_session,
    predict_candidates,
    read_cells,
    read_predict_calls,
    write_report,
)

__all__ = [
    "AUDIT_FILENAME",
    "CANDIDATE_LIMIT",
    "CLEAN",
    "DEFAULT_LABEL_KEYS",
    "RULES",
    "SOFT",
    "SOFT_FLAGS",
    "TOOLS_FILENAME",
    "VIOLATION",
    "AuditFlag",
    "AuditInputError",
    "AuditNote",
    "AuditReport",
    "CandidateSetting",
    "CellRecord",
    "PredictCall",
    "audit_cells",
    "audit_session",
    "predict_candidates",
    "read_cells",
    "read_predict_calls",
    "write_report",
]
