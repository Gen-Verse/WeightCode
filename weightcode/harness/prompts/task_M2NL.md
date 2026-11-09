# Task

$statement

Class names by index: $classes

## Files in /work/trial

- `model.npz`: the classifier's weights, one float32 array per tensor (load it with `numpy.load`).
- `model_spec.json`: the architecture, the tensor names and layouts, and the input format.
- `forward.py`: `forward(params, x) -> logits`, the reference forward pass; `model_spec.json` describes the input
  format.$input_files
- `evidence.npz`: unlabelled deployment examples. `x_rand`: examples drawn at random from the deployment data.
  `pred_rand` are the model's predictions on them, `split_ids` are opaque ids, one per example, and the inputs
  `x_rand` are in the format that `model_spec.json` describes.
- `task.json`: this task as data, with `parent_sha256`, the limits and their definitions, the final selection and
  the training-free rule.

You have no labels and no checks: no file holds a label, and no tool scores a patch or reports its accuracy.

## Tools

- `python`: run code in the persistent Python kernel.
- `status`: cells remaining, your last declaration, and the patch kept if the session ended now.
- `declare_final`: declare the one patch you want kept, and why. It is validated within the limits, not scored.

## Budgets

- `python`: $max_cells cells in total. A cell that runs longer than $cell_timeout_s seconds is stopped, which
  restarts the kernel and loses all variables; files in /work/patches and /work/scratch are kept.
- The session is stopped after $wall_minutes minutes. A stopped session keeps the last valid declaration made
  before the stop, so call `declare_final` before then.
- Analysis and forward passes on your own files are not counted.

## Final selection

$final_selection

Declare exactly one final patch. A declaration that fails validation returns its problems and changes nothing; fix
the patch and declare it again.

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
