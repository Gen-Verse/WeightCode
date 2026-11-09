"""The host CPU field of session provenance (``weightcode.harness.provenance.cpu_info``, CODE_AUDIT section 9): parsed from
``/proc/cpuinfo`` text, ``None`` where the file is absent, and added to ``collect`` without changing any other field."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from weightcode.harness import provenance as P

X86 = """processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Toy Xeon(R) CPU @ 2.00GHz
flags\t\t: fpu sse sse2 ssse3 sse4_1 sse4_2 avx avx2 fma f16c avx512f avx512_vnni amx_tile smep sme misalignsse

processor\t: 1
vendor_id\t: GenuineIntel
model name\t: Toy Xeon(R) CPU @ 2.00GHz
flags\t\t: fpu sse sse2 ssse3 sse4_1 sse4_2 avx avx2 fma f16c avx512f avx512_vnni amx_tile smep sme misalignsse
"""

ARM = """processor\t: 0
BogoMIPS\t: 50.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 asimdhp sve sve2
CPU implementer\t: 0x41

processor\t: 1
BogoMIPS\t: 50.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 asimdhp sve sve2
CPU implementer\t: 0x41
"""


def test_x86_model_and_simd_flags(tmp_path: Path) -> None:
    path = tmp_path / "cpuinfo"
    path.write_text(X86)
    info = P.cpu_info(path)
    assert info == {
        "source": str(path),
        "model": "Toy Xeon(R) CPU @ 2.00GHz",
        "n_logical": 2,
        "simd_flags": ["amx_tile", "avx", "avx2", "avx512_vnni", "avx512f", "f16c", "fma", "sse", "sse2", "sse4_1",
                       "sse4_2", "ssse3"],
        "flags_sha256": hashlib.sha256(" ".join(sorted(X86.splitlines()[3].split(":")[1].split())).encode()).hexdigest(),
    }
    # smep (supervisor mode), sme (AMD memory encryption) and misalignsse are not SIMD instruction sets
    assert not {"smep", "sme", "misalignsse", "fpu"} & set(info["simd_flags"])


def test_arm_features_and_missing_model(tmp_path: Path) -> None:
    path = tmp_path / "cpuinfo"
    path.write_text(ARM)
    info = P.cpu_info(path)
    assert info is not None
    assert info["model"] is None  # the file names none: recorded as None, not guessed
    assert info["n_logical"] == 2
    assert info["simd_flags"] == ["asimd", "asimdhp", "sve", "sve2"]


def test_no_cpuinfo_file_gives_none(tmp_path: Path) -> None:
    assert P.cpu_info(tmp_path / "absent") is None


def test_no_flags_line_gives_none_fields(tmp_path: Path) -> None:
    path = tmp_path / "cpuinfo"
    path.write_text("processor\t: 0\nmodel name\t: Toy\n")
    assert P.cpu_info(path) == {"source": str(path), "model": "Toy", "n_logical": 1, "simd_flags": None,
                                "flags_sha256": None}


@pytest.mark.skipif(not P.CPUINFO_PATH.exists(), reason="no /proc/cpuinfo on this host")
def test_this_host_is_read() -> None:
    info = P.cpu_info()
    assert info is not None and info["n_logical"] >= 1 and info["flags_sha256"] is not None


def test_collect_adds_cpu_and_keeps_every_other_field(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``collect`` gains one key, ``cpu``; the other keys, and ``code_changed``'s keys, are as before."""
    path = tmp_path / "cpuinfo"
    path.write_text(X86)
    monkeypatch.setattr(P, "CPUINFO_PATH", path)
    monkeypatch.setattr(P, "git_state", lambda: {"head": "abc", "status_diff_sha256": "0", "dirty": False,
                                                 "n_status_lines": 0})
    monkeypatch.setattr(P, "bwrap_version", lambda: "bubblewrap 0.11.1")
    monkeypatch.setattr(P, "sandbox_versions", lambda python: {"python": "3.14.4", "numpy": "2.5.4"})
    record = P.collect(sandbox_python=Path("/nonexistent/python"), prompts={"system": "x"}, claude=None)
    assert list(record) == ["git", "source_sha256", "source_tree_sha256", "prompt_templates_sha256",
                            "rendered_prompts_sha256", "kernel_sha256", "bwrap_version", "sandbox", "host", "claude",
                            "cpu"]
    assert record["cpu"] == P.cpu_info(path)
    other = {**record, "cpu": None}
    assert P.code_changed(record, other) == []  # a CPU difference is not a code change
