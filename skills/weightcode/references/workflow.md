# Portable weight-program workflow

## Engine and sandbox

Use Linux with bubblewrap 0.11+, unprivileged user namespaces, a distribution Python under `/usr`, and `uv`.
The program interpreter is a separate NumPy-only venv. There is no unsandboxed fallback.

```bash
git clone https://github.com/Gen-Verse/WeightCode.git
cd WeightCode
uv sync --locked --extra train --extra dev
bash scripts/setup_sandbox_venv.sh
uv run weightcode doctor --sandbox-python .venv-sandbox/bin/python
```

`doctor` prints `"status": "ok"` with the bubblewrap and NumPy versions once the sandbox works.

The `weightcode` executable is installed in the engine environment. When operating from elsewhere, use
`uv run --project <engine-checkout> weightcode ...` and give paths inside the checkout as absolute paths: the sandbox
interpreter and any program file, such as `<engine-checkout>/programs/resnet18-ridge-head/program.py`. Paths of the
user's own files stay relative to the user's project. uv may warn that `VIRTUAL_ENV` does not match the project
environment; the warning is harmless.
Do not bind a venv containing credentials or user data as the sandbox interpreter; its entire directory is mounted
read-only. The supplied setup script creates a NumPy-only environment.

## Check the engine (no API key, about one minute)

```bash
uv run python scripts/demo/skill_cli_walkthrough.py --root out/skill_cli
uv run python scripts/demo/skill_roundtrip.py --out out/skill_demo --sandbox-python .venv-sandbox/bin/python
```

The first writes a small synthetic project (`model.pt`, `labels.json`, three image folders) and runs every reuse
command of this page on it, printing each command; its example program is too weak for that project, so it ends
with `status: base-kept`. The second drives the editor's MCP server with a scripted editor on two ResNet-18
backbones and ends with `"fixture_statuses": ["accepted", "accepted", "accepted"]` and `"exact_replay": true`.
Both use synthetic images, so neither measures recognition quality. They write about 0.4 GB and 1.1 GB; run them from
the checkout as shown, or pass `--root` and `--out` paths elsewhere when the checkout should stay clean.

## Import and prepare data

The checkpoint is a **plain** `torchvision.models.resnet18(...).state_dict()` saved with `torch.save`.
FP32 weights, BatchNorm statistics and existing int64 counters are preserved. Legacy torchvision files without
training counters get zero counters, as in PyTorch loading; this is recorded and does not affect eval inference.
Wrapped checkpoints, half precision, custom
layers and custom normalization are unsupported. `labels.json` is an ordered list matching the head rows.

```bash
uv run weightcode import-model --checkpoint model.pt --labels labels.json --out work/model
```

Each chosen data role has `new/<class-name>/<image>` and `old/<class-name>/<image>` directories. Class names must
match the ordered labels; a role may contain only the relevant classes. Images are RGB-converted, resized with
bilinear antialiasing, then center-cropped. Normalization happens inside the fixed forward adapter.

```bash
uv run weightcode prepare-data --model work/model --new images/experience/new --old images/experience/old --out work/experience.npz
uv run weightcode prepare-data --model work/model --new images/selection/new --old images/selection/old --out work/selection.npz
uv run weightcode prepare-data --model work/model --new images/final/new --old images/final/old --out work/final.npz
uv run weightcode prepare-task --model work/model --experience work/experience.npz --selection work/selection.npz --final work/final.npz --goal 'Recognize objects under the supplied corruption while retaining clean recognition' --retention-drop 0.01 --min-new-gain 0.01 --out work/task
```

`--goal` is one sentence describing the capability; the editor reads it during synthesis, and reuse only records
it. The two thresholds are fractions of accuracy, and `old_change` is negative for a loss. Accuracy moves in steps of
1/n for n images of a condition: with 18 images, `--retention-drop 0.01` allows no lost old image, and any accepted
gain is at least one image (5.6 points).

Alternatively supply `.npz` files with exactly `x_new`, `y_new`, `x_old`, `y_old`: nonempty uint8 RGB crops
`[N,3,224,224]` and int64 label indices `[N]`. Experience, selection and final images must be disjoint.
Exact duplicate crops across roles are rejected; semantic/subject leakage still needs a suitable user split.
The engine copies the chosen inputs and binds their content hashes. It does not split, corrupt, augment or sample.

## Reuse

A program bundle contains `program.py` and a manifest binding its source hash, adapter, input contract, tensor
layout and class order. Compatibility does not imply a gain on a different checkpoint; rerun and validate it.

```bash
uv run weightcode pack-program --program programs/resnet18-ridge-head/program.py --model work/model --out work/program
uv run weightcode apply --task work/task --program work/program --sandbox-python .venv-sandbox/bin/python --wall-s 600 --out work/reuse
```

The bundled ridge-head example is an analytic, gradient-free head correction with old-logit anchors. It is a
reusable example, **not one of the paper's reported programs**. Its data-dependent edits may fail validation or
retention. Packing arbitrary source records compatibility, not trust or measured performance.

For a second checkpoint, import that checkpoint, prepare a task with its own permitted experience and validation,
then pass the same program bundle to `apply`. Matching tensor shapes with a reordered class list is rejected.

## Synthesize

Requires a logged-in `claude` CLI. The current backend uses Claude Code, with native tools disabled and only the
application MCP tools allowed. An outer Codex agent can operate the skill; this is not a Codex synthesis backend.
The following is an example budget, not permission to incur spending without the user's requested run:

```bash
uv run weightcode synthesize --task work/task --sandbox-python .venv-sandbox/bin/python --editor-model sonnet --checks 8 --cells 100 --wall-s 1800 --out work/synthesis
```

Tools: `python(code)`, `check(program_file)`, `status()`, `finish()`. The editor sees `/work/trial` read-only and
can write `/work/patches` and `/work/scratch`. It cannot read selection or final data through its tools.
Every submitted program consumes a check, including invalid ones. Candidates start from the original model.
Eligible means strictly positive new gain meeting `min_new_gain` and old loss no greater than `retention_drop`.
Among eligible candidates, highest new+old accuracy sum wins; earliest wins a tie. Otherwise keep the base.
Selection and final use unrounded argmax of the adapter's float32 logits (float64 internal arithmetic).
Release also requires the FP32 torchvision model to meet the same final thresholds; its results are reported separately.

Default edit limits: at most 5% of entries specified by patch operations; relative Frobenius change at most 0.5
for tensors of rank at least two. Shape/finiteness checks and the existing patch executor enforce these limits.
Program runs have 8 GiB address space, a wall cap per run set by `--program-wall-s` (default 300 s), no network and
a deterministic virtual clock. `--wall-s` is the outer editor/apply budget; setup/baseline and independent final
validation occur outside it. For `apply`, which runs one program once, `--wall-s 600` leaves room for that run.
These counts and timeouts bound workflow resources; they do not establish an exact monetary charge.

## Output and replay

```text
run/
  run.json                    # the run manifest: input bindings, budgets, edit limits, runtime versions
  attempts/0001/               # snapshotted source, execution record, derived patch, candidate weights
    check.json                # selection-split score, eligibility decision, patch summary (n_changed, per_tensor)
  freeze.json                 # write-once selection before final validation; selected is null if none qualified
  private/
    baseline.json             # selection-split score of the original model
    task/                     # copies of the task bundle, including selection.npz and final.npz
  trial/                      # the sandbox's read-only inputs: params, forward code, experience, layout, task.json
  patches/                    # submitted program files (the applied program, for apply)
  scratch/ cwd/ prompts/      # editor working directories, used during synthesis
  logs/                       # editor transcript and MCP/kernel logs (synthesis)
  session_result.json         # the command's JSON result, as printed
  final/
    model.pt                  # plain torchvision state_dict, reload and forward parity checked
    params.npz                # canonical released parameters
    report.json               # accepted / base-kept / final-rejected and final-split results
    program/                  # chosen executable bundle, if an eligible program was selected
```

`report.json` scores the final split only. `final_base` is the original model and `final_candidate` the frozen
selection; when `freeze.selected` is null they are the same model. The selection-split numbers behind a decision are
in `private/baseline.json` and each `attempts/<n>/check.json`.

`model.pt` contains the updated model only for `accepted`; otherwise it contains the original. A selected program
can still be exported after final rejection, with `final_accepted: false` in its provenance. The rejected candidate
stays under `attempts/`; do not describe it as a released capability. A completed negative result exits normally;
execution/configuration failures exit nonzero and retain partial records without publishing a final model.

Replay the chosen program with `apply` into a fresh output directory. Compare the canonical `params_sha256` and
program hash, not ZIP/pickle file byte hashes. The runtime, NumPy-only venv, input files and program must match for
exact replay. The supplied application smoke demo creates two independently initialized real ResNet-18 backbones
and synthetic images for integration checks; its outcomes are not task-performance evidence.
