# Editing a trained classifier

A trained classifier is used on deployment data, and it gets some of those examples wrong. Change its weights
directly, without training it, so that it gets more deployment examples right while keeping the ones it already
gets right. You cannot train the network. `model_spec.json` describes the input format.

## What is kept

After the session, the patch named in your last valid `declare_final` call is kept: a patch that passes validation
within the limits, or null for the original model. If you make no valid declaration, the original model is kept. No
patch is checked or scored during the session. The kept model is measured afterwards on deployment examples that are
held out and that you never see.

## Rules

Do not compute gradients of any loss with respect to the weights (no autograd, no hand-written backpropagation, no
finite-difference gradients). Do not fit parameters to labels with solvers or regression (lstsq, solve, pinv, inv,
ridge, logistic regression, any `.fit`). Do not run optimisation or search loops that update weights or pick among
many candidates by a label-dependent score. Do not score more than 8 distinct candidate patches against the
evidence labels in total. Allowed: any forward passes and activation analysis, weight inspection, label-conditioned
statistics such as class means and their differences, SVD or PCA, single-unit ablations to measure attribution,
and comparing up to 8 explicit candidate patches on the evidence. Every cell you run is logged and audited against
these rules.

## How to work

Understand why the model fails on these examples before you edit it: what the failing examples have in common,
which units and directions drive the wrong decisions, how failing examples differ from correctly classified ones,
and what an edit would change for the examples the model already gets right. Prefer edits whose effect you can
predict. Every patch states a falsifiable `hypothesis` and a `predicted_effect` (on accuracy, and on the examples
the model already gets right).

## Environment

You act only through the tools named in the task. The `python` tool runs code in one persistent Python kernel with
numpy and the standard library; there is no network and no other package. Your files are in /work/trial, which is
read-only. Write patches, and the .npy files they reference, to /work/patches, and anything else to /work/scratch.
A patch counts only once you pass its file name to `declare_final`.

When you have finished, call `declare_final` with the one patch you want kept (or null for the original model) and
a short rationale, then stop.
