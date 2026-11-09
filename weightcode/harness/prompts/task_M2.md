# Task

$statement

Class names by index: $classes

## Files in /work/trial

- `model.npz`: the classifier's weights, one float32 array per tensor (load it with `numpy.load`).
- `model_spec.json`: the architecture, the tensor names and layouts, and the input format.
- `forward.py`: `forward(params, x) -> logits`, the reference forward pass; `model_spec.json` describes the input
  format.$input_files
- `evidence.npz`: labelled deployment examples. `x_fail`, `y_fail`, `pred_fail`: examples the model gets wrong,
  drawn at random from all of its errors on the deployment data. `x_rand`, `y_rand`, `pred_rand`: random deployment
  examples, whether the model gets them right or not. `pred_*` are the model's predictions, `split_ids` are opaque
  ids, and the inputs `x_*` are in the format that `model_spec.json` describes.
- `task.json`: this task as data, with `parent_sha256`, the limits and their definitions, the patch-check readout,
  the final selection and the training-free rule.

## Tools

- `python`: run code in the persistent Python kernel.
- `patch_check`: check one patch file from /work/patches on the held-out check set.
- `status`: queries and cells remaining, and the S of every patch checked so far.
- `declare_final`: record the patch you want kept, and why.

## Budgets

- `patch_check`: $budget_queries queries in total ($invalid_patches).
- `python`: $max_cells cells in total. A cell that runs longer than $cell_timeout_s seconds is stopped, which
  restarts the kernel and loses all variables; files in /work/patches and /work/scratch are kept.
- The session is stopped after $wall_minutes minutes. A stopped session keeps the final selection below over the
  patches checked before the stop, so call `declare_final` and stop before then.
- Analysis and forward passes on your own files are not counted.

## Patch-check readout

$readout

## Final selection

$final_selection

## Training-free rule

Forbidden:
$forbidden

Allowed:
$allowed

Audit: $audit.

## Limits

$limits

## Patch format

Set `parent_sha256` to the value of `parent_sha256` in /work/trial/task.json. Allowed ops: $allowed_ops.

$patch_format
