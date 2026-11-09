"""PREREG T05: the ImageNetV2 readout script on a synthetic archive (4 classes, 2 images each, a stub forward).

Checks the archive and freeze-record refusals, the folder-number labels, the clean T04 crop, and the fixed / broken /
break-rate / paired-interval table. No ImageNetV2 file, no store, no network weights.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from weightcode.common.identity import params_sha256

SCRIPT = Path(__file__).parents[1] / "scripts" / "t05" / "t05_readout.py"


def _module():
    spec = importlib.util.spec_from_file_location("t05_readout", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jpeg(value: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((260, 300, 3), value, np.uint8)).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@pytest.fixture()
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    module = _module()
    monkeypatch.setattr(module, "N_IMAGES", 8)
    monkeypatch.setattr(module, "PER_CLASS", 2)
    monkeypatch.setattr(module.T, "N_CLASSES", 4)
    archive = tmp_path / "v2.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for label in range(4):
            for k in range(2):
                data = _jpeg(30 * label + 10 * k + 5)
                info = tarfile.TarInfo(f"imagenetv2-matched-frequency-format-val/{label}/img{k}.jpeg")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    record = tmp_path / "download_record.json"
    record.write_text(json.dumps({"sha256": sha}))

    def forward(params, x):  # class = the bin of the mean pixel; params["shift"] moves every prediction
        bins = np.clip((x.reshape(len(x), -1).mean(1) // 30).astype(int) + int(params["shift"][0]), 0, 3)
        return np.eye(4)[bins]

    monkeypatch.setattr(module, "read_dataset_meta", lambda dataset: {})
    monkeypatch.setattr(module, "load_dataset_forward", lambda dataset, meta: (forward, "f" * 64, None))
    nets = {}
    for name, shift in (("base", 0), ("final", 1)):
        params = {"shift": np.array([shift], np.int64)}
        path = tmp_path / f"{name}.npz"
        np.savez(path, **params)
        nets[name] = (path, params_sha256(params))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({name: sha for name, (_, sha) in nets.items()}))
    return {"module": module, "archive": archive, "record": record, "frozen": frozen, "nets": nets, "tmp": tmp_path}


def _args(s: dict, out: Path, nets: dict | None = None) -> list[str]:
    args = ["--archive", str(s["archive"]), "--download-record", str(s["record"]), "--frozen", str(s["frozen"]),
            "--out", str(out)]
    for name, (path, sha) in (nets or s["nets"]).items():
        args += ["--network", f"{name}={path}:{sha}"]
    return args


def test_readout_table(setup: dict) -> None:
    out = setup["tmp"] / "out.json"
    assert setup["module"].main(_args(setup, out)) == 0
    record = json.loads(out.read_text())
    base, final = record["networks"]["base"], record["networks"]["final"]
    assert base["acc"] == 1.0 and base["fixed"] == 0 and base["broken"] == 0
    # shift 1 moves labels 0..2 to the wrong class and leaves label 3 (clipped) right
    assert final["n_correct"] == 2 and final["broken"] == 6 and final["break_rate_among_base_correct"] == 0.75
    assert final["delta_points"] == pytest.approx(-75.0)
    low, high = final["delta_95ci_points"]
    assert low < -75.0 < high
    assert record["members"]["n"] == 8 and record["frozen_record"]["sha256"]


def test_refusals(setup: dict) -> None:
    module = setup["module"]
    path, sha = setup["nets"]["final"]
    with pytest.raises(SystemExit, match="not in the freeze record"):
        module.main(_args(setup, setup["tmp"] / "a.json", {**setup["nets"], "other": (path, "0" * 64)}))
    with pytest.raises(SystemExit, match="base=... is required"):
        module.main(_args(setup, setup["tmp"] / "b.json", {"final": (path, sha)}))
    setup["record"].write_text(json.dumps({"sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="differs from the download record"):
        module.main(_args(setup, setup["tmp"] / "c.json"))


def test_plain_tar_named_tar_gz_reads_the_same(setup: dict) -> None:
    """The published imagenetv2-matched-frequency.tar.gz is a plain POSIX tar: the same members, read the same way."""
    module = setup["module"]
    plain = setup["tmp"] / "plain.tar.gz"
    with tarfile.open(setup["archive"], "r:gz") as source, tarfile.open(plain, "w", format=tarfile.PAX_FORMAT) as tar:
        for member in source.getmembers():
            tar.addfile(member, source.extractfile(member))
    assert module.archive_compression(plain) == "none" and module.archive_compression(setup["archive"]) == "gzip"
    setup["record"].write_text(json.dumps({"sha256": hashlib.sha256(plain.read_bytes()).hexdigest()}))
    out = setup["tmp"] / "plain.json"
    assert module.main(_args({**setup, "archive": plain}, out)) == 0
    record = json.loads(out.read_text())
    assert record["archive"]["compression"] == "none" and record["networks"]["final"]["broken"] == 6
    garbage = setup["tmp"] / "garbage.tar.gz"
    garbage.write_bytes(b"Pa" + b"\0" * 10)
    with pytest.raises(SystemExit, match="neither gzip nor a tar"):
        module.archive_compression(garbage)
