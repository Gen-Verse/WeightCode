---
name: weightcode
description: Install a capability in a trained neural network by synthesizing or reusing an executable weight-update program with WeightCode, without gradient fine-tuning. Use when a user wants a trained model (first adapter: torchvision ResNet-18) to handle a new condition or class decision while keeping its old behaviour, wants to rerun a saved weight program on another checkpoint, or wants such an edit validated and exported. Ordinary source-code edits do not need this skill.
---

# WeightCode

Turn the user's capability request into a checked weight program and an independently runnable model. The
WeightCode engine executes programs, enforces edit/check budgets, selects candidates, and exports artifacts.
This skill translates the task and operates that engine.

## Choose a workflow

- **Reuse:** the user has a compatible program bundle, or a program file to pack with `pack-program` (the engine
  checkout ships one example, `programs/resnet18-ridge-head/program.py`, which is weak and often ends `base-kept`).
  Apply it to their model and their own experience; no LLM search is needed. A frozen weight delta is tied to one
  model; a program recomputes an edit from each model.
- **Synthesize:** the user needs a new program. Use the controlled Claude Code editor with an explicit model,
  check count, Python-cell count, and wall budget. Keep user-authorized API/subscription spending within that run.
- **Inspect/package:** explain or package an existing program without promising a capability gain.

Read [references/workflow.md](references/workflow.md) for installation, commands, input layout and output meanings.
Run `weightcode --help` and `weightcode doctor --sandbox-python <checkout>/.venv-sandbox/bin/python` before model
editing; from outside the engine checkout, prefix each command with `uv run --project <checkout>`. If the engine is
missing, use an isolated checkout/environment as described there. If the engine has not run on this machine before,
or `doctor` fails, run the two no-API checks in its "Check the engine" section first. They write about 1.5 GB, so
ask before writing them into the user's project.

## Establish the task

Resolve these from the request and supplied files; ask only for missing decisions that affect execution:

- Supported model checkpoint and an ordered JSON list of class names. First adapter: **FP32 torchvision
  ResNet-18**, plain state_dict, standard ImageNet resize/crop/normalization. Do not load an arbitrary pickled module
  or guess a different architecture's compatibility.
- Capability goal, labeled new-condition examples, and labeled old-condition examples to preserve.
- Three user-chosen, disjoint roles: experience visible to the editor/program, selection used for bounded aggregate
  feedback, final validation used only after the program is frozen. Do not invent labels, substitute datasets,
  silently repartition data, or use paper test sets as application experience.
- Allowed old-accuracy loss and required new-accuracy gain, both as fractions. The engine also requires a strictly
  positive new gain. A value of `0.01` means one percentage point. Accuracy moves in steps of 1/n for n images of a
  condition in a split: with 18 old images, `--retention-drop 0.01` allows no lost image. Tell the user when the
  split sizes make a threshold coarser than they asked for.
- A one-sentence `--goal` describing the capability. The editor sees it during synthesis; reuse only records it.
- Explicit fresh output paths; for synthesis, the editor model and compute/check budget.

For existing model/task/program bundles, validate the recorded contracts through the engine. Keep class order,
preprocessing and tensor layout binding intact; changing a manifest to bypass incompatibility does not adapt a model.

## Execute and interpret

1. Import the supported checkpoint and prepare explicitly chosen image roles. The importer verifies NumPy versus
   torchvision forward parity. These synthetic parity probes establish implementation agreement, not task accuracy.
2. Bind a task with `prepare-task`. Reuse with `apply`, or synthesize with `synthesize`. Each candidate is computed
   from the original weights. Invalid checks consume a slot; do not restart a run to obtain extra checks.
3. Let the engine freeze its selection before final scoring. Do not inspect final labels while designing a
   candidate. Do not retry against the same final split after seeing its results; a revised scientific evaluation
   needs a new, independently justified protocol. When nothing was selected (`freeze.selected` is null), the final
   split has scored only the original model: say so, and let the user decide whether a further attempt may reuse
   that split. Report every attempt run against one final split.
4. Inspect `final/report.json`, not just editor text. `accepted` means both selection and final validation met the
   stated thresholds on these supplied data. `base-kept` and `final-rejected` export the original model. Report those
   outcomes directly. The report holds final-split scores only, and its `final_candidate` is the frozen selection,
   which is the original model when nothing was selected. Selection-split scores are in `private/baseline.json`
   (original model) and `attempts/<n>/check.json` (each candidate's `score`, `decision` and `patch`). Small user
   validation sets do not establish general performance or reproduce the paper.
5. Deliver the exported model, selected program bundle if present, `final/report.json`, the run manifest
   `run.json` and the exact commands. State data roles, thresholds, resource budget and any unverified live-editor
   behavior. Changed weights are in the selected attempt's `check.json` (`patch.n_changed`, `patch.per_tensor`);
   for `base-kept`, say that the released model is unchanged and report the rejected candidates separately.

The program/editor sandbox exposes model, forward code and experience only; selection scoring stays in the host.
The skill and its outer agent are not an access-control boundary: an outer agent with filesystem access could read
task files, and each run keeps copies of the selection and final data under `<run>/private/task/`. For genuinely
confidential evaluation, keep final data with a separate evaluator/account.

Do not modify frozen research contracts, historical result records, original checkpoints or user files to make an
application run pass. All commands require new output paths. Committing or publishing artifacts remains a separate
user action.
