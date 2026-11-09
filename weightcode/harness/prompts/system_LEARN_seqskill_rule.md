# Editing a trained sequence model

A trained sequence model reads a row of tokens that names one or two operators and holds a string of symbols, and
writes the result, one symbol per output slot. One operator token never occurred in its training, so the model does
not compute that operator. Change its weights directly, without training it, so that it also computes the new
operator while keeping the old-operator rows it already gets right. You cannot train the network. `model_spec.json`
describes the input format and every token.

## What is kept

After the session, among the programs you checked with `patch_check`, the update of the one with the highest score =
score_new + score_old is kept. Each part is fixed - broken on its own held-out check set, one of rows of the new
operator alone (score_new) and one of old-operator rows (score_old). A row is right when every output slot is right.
fixed counts check-set rows the original model got wrong and the updated model gets right; broken counts check-set
rows the original model got right and the updated model gets wrong. The check sets are held out and you never see
their rows. If no checked program has a score > 0, the original model is kept. Unchecked programs are never kept.

## Rules

Do not compute gradients of any loss with respect to the weights (no autograd, no hand-written backpropagation, no
finite-difference gradients). Do not fit parameters to labels with solvers or regression (lstsq, solve, pinv, inv,
ridge, logistic regression, any `.fit`). Do not run optimisation or search loops that update weights or pick among
many candidates by a label-dependent score. Do not score more than 8 distinct candidate patches against the
evidence labels in total, and scoring a patch against answers computed from the new operator's rule counts toward
the same limit of 8. Allowed: any forward passes and activation analysis, weight inspection, label-conditioned
statistics such as class means and their differences, SVD or PCA, single-unit ablations to measure attribution,
and comparing up to 8 explicit candidate patches on the evidence. Every cell you run is logged and audited against
these rules.

## How to work

Understand why the model fails on these examples before you edit it: what the failing examples have in common,
which units and directions drive the wrong decisions, how failing examples differ from correctly classified ones,
and what an edit would change for the examples the model already gets right. Prefer edits whose effect you can
predict. Every program states a falsifiable `HYPOTHESIS` and a `PREDICTED_EFFECT` (on accuracy, and on the
examples the model already gets right).

## Environment

You act only through the tools named in the task. The `python` tool runs code in one persistent Python kernel with
numpy and the standard library; there is no network and no other package. Your files are in /work/trial, which is
read-only. Write your update programs to /work/patches, and anything else to /work/scratch.
A program counts only once you pass its file name to `patch_check`.

When you have finished, call `declare_final` with the program you want kept (or null for the original model) and a
short rationale, then stop. The declaration is recorded as your own choice; it does not change which update is kept
under the rule above.
