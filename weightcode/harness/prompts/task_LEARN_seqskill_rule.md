# Task

$statement

Output symbols by index (the answer at each output slot): $classes

## Files in /work/trial

- `model.npz`: the model's weights, one float32 array per tensor (load it with `numpy.load`).
- `model_spec.json`: the architecture, the tensor names and layouts, and the input format.
- `forward.py`: `forward(params, x) -> logits`, the reference forward pass; `model_spec.json` describes the input
  format.$input_files
- `experience.npz`: labelled examples of the old operators with the model's predictions, and no row of the new
  operator. `x_old`, `y_old`, `pred_old`: $n_old $old_description. `y_old` holds the answer symbol at each output
  slot and -1 elsewhere, `pred_old` holds the model's predictions at those slots and -1 elsewhere, `split_ids_old`
  are opaque ids, and the inputs `x_old` are token rows in the format that `model_spec.json` describes.
- `task.json`: this task as data, with `parent_sha256`, the experience, the program interface, the limits and their
  definitions, the patch-check readout, the final selection and the training-free rule.

## Tools

- `python`: run code in the persistent Python kernel.
- `patch_check`: run one update program file from /work/patches on this model and its experience, apply its
  update and score the updated model on the two held-out check sets.
- `status`: queries and cells remaining, and the scores of every program checked so far.
- `declare_final`: record the program you want kept, and why.

## Budgets

- `patch_check`: $budget_queries queries in total ($invalid_programs).
- `python`: $max_cells cells in total. A cell that runs longer than $cell_timeout_s seconds is stopped, which
  restarts the kernel and loses all variables; files in /work/patches and /work/scratch are kept.
- A program run that takes longer than $program_wall_s seconds is stopped; it uses no query.
- The session is stopped after $wall_minutes minutes. A stopped session keeps the final selection below over the
  programs checked before the stop, so call `declare_final` and stop before then.
- Analysis and forward passes on your own files are not counted, and neither are runs of your program in your own
  kernel.

## Patch-check readout

$readout

## Final selection

$final_selection

## Update program

The program is $program_file. It defines:
$program_defines

`update` gets two dicts. `student` holds `params`, $student_params; `forward`, $student_forward; and `layout`,
$student_layout. `experience` holds $experience_arrays.

It returns $returns.

Each `patch_check` call runs the program once, in a fresh sandbox, on this model and its experience: $runs.
The returned deltas are validated (names, shapes, dtypes, finite values), checked against the limits below and
applied; then the updated model is scored. What the program prints is returned with the result.

Transfer: $transfer.

## Training-free rule

Forbidden:
$forbidden

Allowed:
$allowed

Audit: $audit.

The update program:
$program_rule

## Limits

$limits
