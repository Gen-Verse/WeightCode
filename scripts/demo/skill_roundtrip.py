#!/usr/bin/env python3
"""No-API integration demo: real ResNet-18 backbones, synthetic task, controlled MCP editor and reuse.

This checks engineering behavior, not recognition quality or the paper's scientific claims.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18

from weightcode.application.bundles import new_dir, pack_program, prepare_task, read, sha, write
from weightcode.application.editor import synthesize
from weightcode.application.resnet import import_checkpoint
from weightcode.application.runtime import apply_program
from weightcode.harness.jsonio import utc_now
from weightcode.harness.launch import LaunchOutcome, record_launch_start

# Deliberately simple engineering fixture: calibrate a class-1 bias using the current model's own logits.
PROGRAM = '''import numpy as np
HYPOTHESIS = "Integration fixture: recompute a class margin from this model's experience."
PREDICTED_EFFECT = "Exercise an accepted update and exact replay on synthetic class-1 examples."
def update(student, experience):
    logits = student["forward"](student["params"], experience["x_new"])
    delta = np.zeros_like(student["params"]["fc.bias"], dtype=np.float64)
    delta[1] = float(np.max(logits[:, 0] - logits[:, 1])) + 2.0
    return {"fc.bias": delta}
'''


class McpFixtureLauncher:
    """Scripted editor using exactly the public MCP process; no substitute scoring/patch path."""

    name = "application-mcp-fixture"
    stream_expected = False

    def launch(self, session, paths, task_prompt):
        record_launch_start(paths, session)
        config = read(paths.mcp_json)["mcpServers"]["weightcode"]
        with paths.claude_stderr.open("wb") as err:
            proc = subprocess.Popen([config["command"], *config["args"]], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=err, cwd=paths.cwd, env={**os.environ, **config["env"]}, text=True)
            try:
                def call(method, params):
                    request_id = call.counter
                    call.counter += 1
                    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method,
                                                 "params": params}) + "\n")
                    proc.stdin.flush()
                    line = proc.stdout.readline()
                    if not line:
                        detail = paths.server_stderr.read_text() if paths.server_stderr.exists() else ""
                        raise RuntimeError(f"MCP server exited unexpectedly: {detail}; see {paths.server_log}")
                    response = json.loads(line)
                    if "error" in response or response.get("result", {}).get("isError"):
                        raise RuntimeError(response)
                    transcript.append({"request": method, "params": params, "response": response})
                    return response
                call.counter = 1
                transcript = []
                call("initialize", {"protocolVersion": "2024-11-05", "clientInfo": {"name": self.name, "version": "1"}})
                call("tools/list", {})
                code = ("from pathlib import Path\nimport numpy as np\n"
                        "assert sorted(p.name for p in Path('/work/trial').iterdir()) == "
                        "['experience.npz','forward.py','layout.json','params.npz','task.json']\n"
                        f"Path('/work/patches/update.py').write_text({PROGRAM!r})")
                response = call("tools/call", {"name": "python", "arguments": {"code": code}})
                cell = json.loads(response["result"]["content"][0]["text"])
                if cell["error"]:
                    raise RuntimeError(cell)
                response = call("tools/call", {"name": "check", "arguments": {"program_file": "update.py"}})
                checked = json.loads(response["result"]["content"][0]["text"])
                if checked["status"] != "checked":
                    raise RuntimeError(checked)
                call("tools/call", {"name": "finish", "arguments": {}})
                proc.stdin.close()
                proc.wait(timeout=30)
                write(paths.logs / "fixture_transcript.json", {"messages": transcript})
                return LaunchOutcome(self.name, False, utc_now(), ended_at=utc_now(), exit_code=proc.returncode)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()


def run_demo(out: Path, sandbox_python: Path) -> dict:
    out = new_dir(out)
    torch.set_num_threads(4)
    labels = ["synthetic-0", "synthetic-1"]
    datasets = []
    for role, seed in (("experience", 601), ("selection", 602), ("final", 603)):
        rng = np.random.default_rng(seed)
        data = {f"x_{r}": rng.integers(0, 256, (1, 3, 224, 224), dtype=np.uint8) for r in ("new", "old")}
        data.update({f"y_{r}": np.ones(1, dtype=np.int64) for r in ("new", "old")})
        path = out / f"{role}.npz"
        np.savez(path, **data)
        datasets.append(path)
    models, tasks = [], []
    for seed in (101, 102):
        torch.manual_seed(seed)
        model = resnet18(weights=None, num_classes=2)
        with torch.no_grad():
            model.fc.weight[0].abs_().mul_(0.01)
            model.fc.weight[1].abs_().mul_(-0.01)
            model.fc.bias.copy_(torch.tensor([0.25, 0.0]))
        checkpoint = out / f"source-{seed}.pt"
        torch.save(model.state_dict(), checkpoint)
        bundle = out / f"model-{seed}"
        import_checkpoint(checkpoint, labels, bundle)
        task = out / f"task-{seed}"
        prepare_task(bundle, *datasets, task, goal="Synthetic engineering integration fixture", retention_drop=0,
                     min_new_gain=1)
        models.append(bundle)
        tasks.append(task)
    first = synthesize(tasks[0], out / "synthesis", sandbox_python, editor_model="scripted-fixture", checks=1,
                       cells=2, wall_s=300, launcher=McpFixtureLauncher())
    program = out / "synthesis" / "final" / "program"
    replay = apply_program(tasks[0], program, out / "replay", sandbox_python, wall_s=300)
    transfer = apply_program(tasks[1], program, out / "transfer", sandbox_python, wall_s=300)
    assert first["status"] == replay["status"] == transfer["status"] == "accepted"
    assert first["export"]["params_sha256"] == replay["export"]["params_sha256"]
    source_delta = sha(out / "synthesis" / "attempts" / "0001" / "run" / "out" / "deltas.npz")
    target_delta = sha(out / "transfer" / "attempts" / "0001" / "run" / "out" / "deltas.npz")
    assert source_delta != target_delta, "fixture must recompute a different edit for the second backbone"
    # Also execute the distributable example against the real adapter, without requiring a gain.
    example = Path(__file__).resolve().parents[2] / "programs" / "resnet18-ridge-head" / "program.py"
    pack_program(example, models[0], out / "example-program")
    example_report = apply_program(tasks[0], out / "example-program", out / "example", sandbox_python, wall_s=300)
    check = read(out / "example" / "attempts" / "0001" / "check.json")
    assert check["status"] == "checked", check
    result = {"status": "ok", "kind": "synthetic-engineering-only", "backbone_seeds": [101, 102],
              "image_seeds": [601, 602, 603], "exact_replay": True, "distinct_recomputed_deltas": True,
              "source_model_sha256": read(models[0] / "manifest.json")["params_sha256"],
              "target_model_sha256": read(models[1] / "manifest.json")["params_sha256"],
              "fixture_statuses": [first["status"], replay["status"], transfer["status"]],
              "example_status": example_report["status"], "live_paid_editor_tested": False}
    write(out / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sandbox-python", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.out, args.sandbox_python.absolute()), indent=2))


if __name__ == "__main__":
    main()
