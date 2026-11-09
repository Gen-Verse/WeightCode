# Task

$statement

Class names by index: $classes

## Files in /work/trial

- `model_shapes.json`: the name, shape and dtype of every weight tensor. The weights are not in /work/trial; the
  update program receives them when `patch_check` runs it.
- `model_spec.json`: the architecture, the tensor names and layouts, and the input format.
- `forward.py`: `forward(params, x) -> logits`, the reference forward pass; `model_spec.json` describes the input
  format.$input_files
$experience_bullet
- `task.json`: this task as data, with `parent_sha256`, the experience, the program interface, the limits and their
  definitions, the patch-check readout, the final selection and the training-free rule.

## Tools

- `python`: run code in the persistent Python kernel.
- `patch_check`: run one update program file from /work/patches on this classifier and its experience, apply its
  update and score the updated classifier on the two held-out check sets.
- `status`: queries and cells remaining, and the scores of every program checked so far.
- `declare_final`: record the program you want kept, and why.

## Budgets

- `patch_check`: $budget_queries queries in total ($invalid_programs).
- `python`: $max_cells cells in total. A cell that runs longer than $cell_timeout_s seconds is stopped, which
  restarts the kernel and loses all variables; files in /work/patches and /work/scratch are kept.
- A program run that takes longer than $program_wall_s seconds is stopped; it uses no query.
- The session is stopped after $wall_minutes minutes. A stopped session keeps the final selection below over the
  programs checked before the stop, so call `declare_final` and stop before then.
- Every tool result also gives `wall_s_elapsed` and `wall_s_remaining`: the seconds since this session started and
  the seconds left before it is stopped.
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

Each `patch_check` call runs the program once, in a fresh sandbox, on this classifier and its experience: $runs.
The returned deltas are validated (names, shapes, dtypes, finite values), checked against the limits below and
applied; then the updated classifier is scored. A check returns $check_returns.

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
