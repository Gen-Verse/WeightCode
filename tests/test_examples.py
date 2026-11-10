"""The paper examples of ``weightcode reproduce`` (``weightcode.examples``).

Offline: every download is checked against its recorded sha256 (a wrong file is refused and removed, a right one is
kept and not fetched again), ``SHA256SUMS`` lines parse, a dataset path that names no file is an error, and the CLI
parses the command. Online (``WEIGHTCODE_ONLINE_TESTS=1``): the SCAN example downloads its files, reruns the program
without the sandbox and reproduces the paper's test reading exactly.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from weightcode.application.cli import parser
from weightcode.examples import EXAMPLES
from weightcode.examples import fetch as F


def _file(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def test_download_checks_the_sha256(tmp_path: Path) -> None:
    digest = _file(tmp_path / "src" / "a.txt", "hello\n")
    url = (tmp_path / "src" / "a.txt").as_uri()
    dest = tmp_path / "dl" / "a.txt"
    assert F.download(url, dest, digest) == dest and dest.read_text() == "hello\n"
    (tmp_path / "src" / "a.txt").unlink()
    assert F.download(url, dest, digest) == dest  # already present with the right hash: not fetched again
    _file(tmp_path / "src" / "b.txt", "changed\n")
    with pytest.raises(F.FetchError, match="is not the recorded"):
        F.download((tmp_path / "src" / "b.txt").as_uri(), tmp_path / "dl" / "b.txt", digest)
    assert not (tmp_path / "dl" / "b.txt").exists() and not (tmp_path / "dl" / "b.txt.part").exists()


def test_hf_files_follow_sha256sums(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "endpoint" / "datasets" / F.HF_REPO / "resolve" / F.HF_REVISION
    lines = [f"{_file(root / path, path)}  {path}" for path in ("w/i1/model.npz", "w/i1/task.json", "r/s1/run.json")]
    sums = "\n".join(lines) + "\n"
    _file(root / "SHA256SUMS", sums)
    monkeypatch.setenv("HF_ENDPOINT", (tmp_path / "endpoint").as_uri())
    monkeypatch.setattr(F, "HF_SHA256SUMS_SHA256", hashlib.sha256(sums.encode()).hexdigest())
    got = F.hf_files(["w/i1/", "r/s1/run.json"], tmp_path / "out")
    assert sorted(got) == ["r/s1/run.json", "w/i1/model.npz", "w/i1/task.json"]
    assert (tmp_path / "out" / "w" / "i1" / "task.json").read_text() == "w/i1/task.json"
    with pytest.raises(F.FetchError, match="has no file under"):
        F.hf_files(["w/i2/"], tmp_path / "out")


def test_parse_sha256sums_refuses_a_bad_line() -> None:
    assert F.parse_sha256sums("a" * 64 + "  x/y\n\n") == {"x/y": "a" * 64}
    with pytest.raises(F.FetchError):
        F.parse_sha256sums("abc  x\n")


def test_the_examples_run_without_pytorch() -> None:
    """``uv sync --locked`` installs numpy only; the reproduce path must not import torch or torchvision."""
    import subprocess
    import sys

    code = ("import sys\nsys.modules['torch'] = sys.modules['torchvision'] = None\n"
            "import weightcode.application.cli, weightcode.examples.replay\n"
            "from weightcode.examples import EXAMPLES\nimport importlib\n"
            "[importlib.import_module(m) for m in EXAMPLES.values()]\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


def test_the_cli_parses_every_example() -> None:
    for name in EXAMPLES:
        args = parser().parse_args(["reproduce", name, "--out", "out/x"])
        assert args.command == "reproduce" and args.example == name and args.sandbox_python is None


@pytest.mark.skipif(os.environ.get("WEIGHTCODE_ONLINE_TESTS") != "1", reason="downloads ~10 MB; set WEIGHTCODE_ONLINE_TESTS=1")
def test_scan_reproduces_the_paper(tmp_path: Path) -> None:
    import sys

    from weightcode.examples import scan
    from weightcode.sandbox.program import UnisolatedRunner

    runner = UnisolatedRunner(python=Path(sys.executable), memory_bytes=8 * 2**30, threads=4)
    summary = scan.run(tmp_path, runner, log=lambda _: None)
    assert summary["replay"]["exact"] and summary["matches_paper"]
    assert summary["new"]["n"] == 2568 and summary["new"]["em_base"] == 0.0 and summary["replay"]["n_changed"] == 257


def test_imagenet_reads_the_records_per_image_bits() -> None:
    """The record stores correctness as base64 of ``numpy.packbits(..., bitorder='big')``; a wrong length is refused."""
    import base64

    import numpy as np

    from weightcode.examples import imagenet
    from weightcode.examples.replay import ReplayError

    bits = np.random.default_rng(0).random(13) < 0.5
    text = base64.b64encode(np.packbits(bits, bitorder="big").tobytes()).decode("ascii")
    assert np.array_equal(imagenet.decode_bits(text, 13), bits)
    with pytest.raises(ReplayError, match="need"):
        imagenet.decode_bits(text, 30)
