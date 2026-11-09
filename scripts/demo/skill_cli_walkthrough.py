#!/usr/bin/env python3
"""The `weightcode` skill's reuse workflow, command by command, on a small synthetic project (no API, about 30 s).

It first writes what a user would bring:
- `model.pt`: a plain torchvision ResNet-18 state_dict. The backbone is random (seed 0); the head is fitted to the old
  condition, standing in for a network trained on the old task.
- `labels.json`: the ordered class list.
- `images/experience`, `images/selection`, `images/final`: three disjoint folders, each with `new/<class>/` and
  `old/<class>/` PNG files. A class is a colour under heavy noise in the new condition and a stripe orientation in
  the old one.

Then it runs the exact commands of `skills/weightcode/references/workflow.md` through the installed `weightcode`
executable (doctor, import-model, prepare-data three times, prepare-task, pack-program, apply) and prints the outcome.
With this project the bundled example program caps its head change at 10% of the head's norm, which is too small to
flip a new-condition decision on the selection split, so nothing is selected and the engine keeps the original model
(`base-kept`). That is the release rule at work: an update is exported only when it meets the thresholds on both the
selection and the final split. The summary prints the selection-split scores from `private/baseline.json` and
`attempts/*/check.json`, because `final/report.json` holds only final-split scores. The outcome says nothing about
recognition quality. `scripts/demo/skill_roundtrip.py` shows an accepted update and its replay.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABELS = ["red", "green", "blue"]
COUNTS = {"experience": 6, "selection": 6, "final": 6}  # images per class and per condition (new, old)
SEEDS = {"experience": 11, "selection": 12, "final": 13}


def _image(condition: str, k: int, rng: np.random.Generator) -> np.ndarray:
    """A 64 x 64 RGB image of class ``k``: new = class colour plus strong noise, old = class stripe orientation."""
    if condition == "new":
        base = np.zeros((64, 64, 3))
        base[..., k] = 200.0
        return np.clip(base + rng.normal(0.0, 70.0, base.shape), 0, 255).astype(np.uint8)
    grid = np.indices((64, 64)).astype(np.float64)
    direction = (grid[0], grid[1], grid[0] + grid[1])[k]
    stripes = 127.5 + 100.0 * np.sin(direction * 0.6 + rng.uniform(0, 2 * np.pi))
    gray = stripes[..., None].repeat(3, axis=2) + rng.normal(0.0, 10.0, (64, 64, 3))
    return np.clip(gray, 0, 255).astype(np.uint8)


def fit_head_on_old_task(model) -> None:
    """Stand in for a network trained on the old task: a ridge fit of the head on 30 old-condition images per class
    (seed 10, never part of the three folders). The new condition stays unlearned."""
    import torch

    rng = np.random.default_rng(10)
    images = np.stack([_image("old", k, rng) for k in range(len(LABELS)) for _ in range(30)])
    y = np.repeat(np.arange(len(LABELS)), 30)
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", antialias=True)
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    head, model.fc = model.fc, torch.nn.Identity()
    with torch.no_grad():
        f = model((x - mean) / std).double().numpy()
    model.fc = head
    f1 = np.concatenate([f, np.ones((len(f), 1))], axis=1)
    target = 4.0 * (np.eye(len(LABELS))[y] - 1.0 / len(LABELS))
    w = np.linalg.solve(f1.T @ f1 + 1e-2 * np.eye(f1.shape[1]), f1.T @ target).T
    with torch.no_grad():
        model.fc.weight.copy_(torch.from_numpy(w[:, :-1]).float())
        model.fc.bias.copy_(torch.from_numpy(w[:, -1]).float())


def write_project(root: Path) -> None:
    """The files a user brings: model.pt, labels.json and the three image folders."""
    import torch
    from PIL import Image
    from torchvision.models import resnet18

    torch.manual_seed(0)
    model = resnet18(weights=None, num_classes=len(LABELS)).eval()
    fit_head_on_old_task(model)
    torch.save(model.state_dict(), root / "model.pt")
    (root / "labels.json").write_text(json.dumps(LABELS))
    for role, count in COUNTS.items():
        rng = np.random.default_rng(SEEDS[role])
        for condition in ("new", "old"):
            for k, label in enumerate(LABELS):
                folder = root / "images" / role / condition / label
                folder.mkdir(parents=True)
                for i in range(count):
                    Image.fromarray(_image(condition, k, rng)).save(folder / f"{i:03d}.png")


def run(command: list[str], cwd: Path) -> dict:
    """Run one `weightcode` command as a user would, echo it, and return its JSON output."""
    print("$ " + shlex.join(command), flush=True)
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"skill_cli_walkthrough: command failed ({done.returncode}):\n{done.stderr.strip()}")
    return json.loads(done.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=REPO / "out" / "skill_cli",
                        help="a new directory for the synthetic project and its outputs (default: out/skill_cli)")
    parser.add_argument("--sandbox-python", type=Path, default=REPO / ".venv-sandbox" / "bin" / "python")
    args = parser.parse_args(argv)
    root = args.root.absolute()
    if root.exists():
        sys.exit(f"skill_cli_walkthrough: {root} already exists; remove it or pass another --root")
    sandbox = args.sandbox_python.absolute()
    weightcode = Path(sys.executable).parent / "weightcode"  # the executable `uv sync` installs next to python
    if not weightcode.exists():
        sys.exit(f"skill_cli_walkthrough: {weightcode} not found; run `uv sync --extra train` in the checkout")
    root.mkdir(parents=True)
    write_project(root)
    print(f"Synthetic project in {root}: model.pt, labels.json, images/{{experience,selection,final}}")

    wc = [str(weightcode)]
    run(wc + ["doctor", "--sandbox-python", str(sandbox)], root)
    run(wc + ["import-model", "--checkpoint", "model.pt", "--labels", "labels.json", "--out", "work/model"], root)
    for role in COUNTS:
        run(wc + ["prepare-data", "--model", "work/model", "--new", f"images/{role}/new",
                  "--old", f"images/{role}/old", "--out", f"work/{role}.npz"], root)
    run(wc + ["prepare-task", "--model", "work/model", "--experience", "work/experience.npz",
              "--selection", "work/selection.npz", "--final", "work/final.npz",
              "--goal", "Recognize the colour classes under heavy noise while keeping the stripe classes",
              "--retention-drop", "0.01", "--min-new-gain", "0.01", "--out", "work/task"], root)
    program = REPO / "programs" / "resnet18-ridge-head" / "program.py"
    run(wc + ["pack-program", "--program", str(program), "--model", "work/model", "--out", "work/program"], root)
    report = run(wc + ["apply", "--task", "work/task", "--program", "work/program",
                       "--sandbox-python", str(sandbox), "--wall-s", "600", "--out", "work/reuse"], root)

    reuse = root / "work" / "reuse"
    final = reuse / "final"

    def counts(scores: dict) -> str:
        return (f"new {scores['new']['correct']}/{scores['new']['n']}, "
                f"old {scores['old']['correct']}/{scores['old']['n']}")

    print(f"\nstatus: {report['status']} (updated model exported: {report['output_is_updated_model']})")
    print(f"  selection split, original model: {counts(json.loads((reuse / 'private' / 'baseline.json').read_text()))}")
    for check_file in sorted((reuse / "attempts").glob("*/check.json")):
        check = json.loads(check_file.read_text())
        print(f"  selection split, candidate {check['index']}: {counts(check['score'])}, "
              f"{check['patch']['n_changed']} weights changed, eligible: {check['decision']['eligible']}")
    print(f"  final split, original model: {counts(report['final_base'])}")
    if report["freeze"]["selected"] is not None:
        print(f"  final split, selected program: {counts(report['final_candidate'])}")
    print(f"outputs: {final / 'model.pt'}, {final / 'report.json'}"
          + (f", {final / 'program'}" if (final / "program").is_dir() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
