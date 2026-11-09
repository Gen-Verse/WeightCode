# Editing a trained sequence model

A trained sequence model reads a command, a row of word tokens, and writes the action sequence the command denotes,
one action per output slot and then END. It fails on a family of commands that the task names, commands built from
words and constructions it was trained on but combined in a way it never saw. Change its weights directly, without
training it, so that it also answers those commands while keeping the commands it already gets right. You cannot
train the network. `model_spec.json` describes the input format and gives the meaning of every word as
interpretation rules.

## What is kept

After the session, among the programs you checked with `patch_check`, the update of the one with the highest score =
score_new + score_old is kept. Each part is fixed - broken on its own held-out check set, one of target commands
(score_new) and one of commands of the training distribution that were never in the model's training rows
(score_old). A row is right when every output slot is right, END included. fixed counts check-set rows the original
model got wrong and the updated model gets right; broken counts check-set rows the original model got right and the
updated model gets wrong. The check sets are held out and you never see their rows. If no checked program has a
score > 0, the original model is kept. Unchecked programs are never kept.

## Rules

Do not compute gradients of any loss with respect to the weights (no autograd, no hand-written backpropagation, no
finite-difference gradients). Do not fit parameters to labels with solvers or regression (lstsq, solve, pinv, inv,
ridge, logistic regression, any `.fit`). Do not run optimisation or search loops that update weights or pick among
many candidates by a label-dependent score. Do not score more than 8 distinct candidate patches against the
evidence labels in total, and scoring a patch against answers computed from the interpretation rules counts toward
the same limit of 8. Allowed: any forward passes and activation analysis, weight inspection, label-conditioned
statistics such as class means and their differences, SVD or PCA, single-unit ablations to measure attribution,
and comparing up to 8 explicit candidate patches on the evidence. Every cell you run is logged and audited against
these rules.

## How to work

Understand why the model fails on the target commands before you edit it: how it computes the commands it gets
right, where the target commands take a different path, which units and directions carry the words involved, and
what an edit would change for the commands the model already gets right. Prefer edits whose effect you can predict.
Every program states a falsifiable `HYPOTHESIS` and a `PREDICTED_EFFECT` (on the target commands, and on the
commands the model already gets right).

## Environment

You act only through the tools named in the task. The `python` tool runs code in one persistent Python kernel with
numpy and the standard library; there is no network and no other package. Your files are in /work/trial, which is
read-only. Write your update programs to /work/patches, and anything else to /work/scratch.
A program counts only once you pass its file name to `patch_check`.

When you have finished, call `declare_final` with the program you want kept (or null for the original model) and a
short rationale, then stop. The declaration is recorded as your own choice; it does not change which update is kept
under the rule above.
