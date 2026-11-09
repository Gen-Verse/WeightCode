"""What one scripted LEARN-family session shows the editor, for the PREREG A4 tests (``test_harness_a4``).

``probe_session`` runs one scripted session on the toy E11 store (``e11_toy``) through the real MCP server: it writes
the probe programs below, checks each one, asks for ``status`` and declares a final. The probe programs are built to
push parameter values out of the program run by every route a program has: ``talk.py`` prints qkv weights to stdout
and stderr and writes them to a file, then returns a valid update; ``boom.py`` raises an exception whose message holds
a weight; ``badkey.py`` returns a delta under a key spelled from a weight and deltas of a weight-dependent wrong
shape; ``dtype.py`` returns a structured dtype whose field name is spelled from a weight; ``bits.py`` returns a delta
with a weight-dependent number of non-finite entries; ``exit7.py`` ends the run with a weight-dependent exit code; and
``ln.py`` and ``qkv_big.py`` are an ordinary update and one over the rel-Frob limit. The toy LayerNorm weights are all
1.0, so the value probes read the random qkv weight.

``shown_digest`` reduces what the editor saw to digests: every trial file (text by its bytes, an .npz by the
content hash of its arrays, since ``numpy.savez`` stamps the time into the archive), both prompts, the ``tools/list``
reply and every tool reply with its timing fields removed (``wall_s_elapsed``, ``wall_s_remaining`` and
``duration_s`` depend on the machine). The B-arm reference is the snapshot from main 9a65b20, before any A4 code, with the package and schema names
migrated to WeightCode. ``B_SHOWN_WEIGHTCODE`` retains a byte-level check of that view.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from test_harness_learn import STRINGS, _program_cell

from weightcode.common.identity import params_sha256
from weightcode.harness import runner
from weightcode.harness.scripted import ScriptedLauncher, call_and_record

WALL_S = 300.0
TIMING_FIELDS = ("wall_s_elapsed", "wall_s_remaining", "duration_s")
LN = "classifier.norm.weight"
QKV = "classifier.blocks.0.self_attn.qkv.weight"

TALK = STRINGS + f"""import numpy as np

def update(student, experience):
    q = np.asarray(student["params"][{QKV!r}], dtype=np.float64).ravel()
    print("TALK_VALUE", repr(float(q[0])), repr(q[:4].tolist()))
    import sys
    sys.stderr.write("TALK_ERR " + repr(float(q[1])) + "\\n")
    open("/work/out/notes.txt", "w").write("TALK_FILE " + repr(q[:64].tolist()))
    w = np.asarray(student["params"][{LN!r}], dtype=np.float64)
    delta = np.zeros_like(w)
    cut = int(round(0.25 * w.size))
    delta[:cut] = -2.0 * w[:cut]
    return {{{LN!r}: delta}}
"""
BOOM = STRINGS + f"""import numpy as np

def update(student, experience):
    q = np.asarray(student["params"][{QKV!r}], dtype=np.float64).ravel()
    raise ValueError("BOOM_VALUE " + repr(float(q[0])))
"""
BADKEY = STRINGS + f"""import numpy as np

def update(student, experience):
    q = np.asarray(student["params"][{QKV!r}], dtype=np.float64).ravel()
    odd = np.zeros(int(abs(q[0]) * 1e3) % 97 + 3)
    return {{"KEY_VALUE_" + str(int(abs(q[0]) * 1e6)): odd, {QKV!r}: odd}}
"""
DTYPE = STRINGS + f"""import numpy as np

def update(student, experience):
    q = np.asarray(student["params"][{QKV!r}], dtype=np.float64).ravel()
    w = student["params"][{LN!r}]
    return {{{LN!r}: np.zeros(w.shape, dtype=[("FIELD_" + str(int(abs(q[1]) * 1e6)), "f8")])}}
"""
BITS = STRINGS + f"""import numpy as np

def update(student, experience):
    q = np.asarray(student["params"][{QKV!r}], dtype=np.float64).ravel()
    w = student["params"][{LN!r}]
    delta = np.zeros(w.shape)
    delta[: int(abs(q[0]) * 1e3) % w.size + 1] = np.nan
    return {{{LN!r}: delta}}
"""
EXIT7 = STRINGS + f"""import os
import numpy as np

def update(student, experience):
    q = np.asarray(student["params"][{QKV!r}], dtype=np.float64).ravel()
    os._exit(int(abs(q[0]) * 1e3) % 100 + 7)
"""
LN_PLANT = STRINGS + f"""import numpy as np

def update(student, experience):
    w = np.asarray(student["params"][{LN!r}], dtype=np.float64)
    delta = np.zeros_like(w)
    cut = int(round(0.25 * w.size))
    delta[:cut] = -2.0 * w[:cut]
    return {{{LN!r}: delta}}
"""
QKV_BIG = STRINGS + f"""import numpy as np

def update(student, experience):
    return {{{QKV!r}: 0.6 * np.asarray(student["params"][{QKV!r}], dtype=np.float64)}}
"""
PROBES = {"talk.py": TALK, "boom.py": BOOM, "badkey.py": BADKEY, "dtype.py": DTYPE, "bits.py": BITS,
          "exit7.py": EXIT7, "ln.py": LN_PLANT, "qkv_big.py": QKV_BIG}
PROBE_ORDER = ("talk.py", "boom.py", "badkey.py", "dtype.py", "bits.py", "exit7.py", "ln.py", "qkv_big.py")
VALUE_MARKERS = ("TALK_VALUE", "TALK_ERR", "TALK_FILE", "BOOM_VALUE", "KEY_VALUE", "FIELD_")


def probe_session(instance_id: str, arm: str, kernel_cells: Sequence[str] = (),
                  **kwargs: Any) -> tuple[runner.SessionResult, list[Any], list[Any]]:
    """One scripted session of ``arm`` running the probe programs (module docstring), after the ``kernel_cells``
    (none for the recorded B digest); returns the result, the tool replies and the ``tools/list`` reply."""
    listed: list[Any] = []

    def script(client: Any, replies: list[Any]) -> None:
        listed.extend(client.list_tools())
        for code in kernel_cells:
            call_and_record(client, replies, "python", {"code": code})
        call_and_record(client, replies, "python", {"code": _program_cell(PROBES)})
        for name in PROBE_ORDER:
            call_and_record(client, replies, "patch_check", {"program_file": name})
        call_and_record(client, replies, "status", {})
        call_and_record(client, replies, "declare_final", {"program_file": "ln.py", "rationale": "it scored"})

    launcher = ScriptedLauncher(script)
    result = runner.run_session(instance_id, arm, 0, effort="max", wall_timeout_s=WALL_S, launcher=launcher,
                                skip_audit=True, retention_floor_frac=0.005, **kwargs)
    return result, launcher.replies, listed


def without_timing(value: Any) -> Any:
    """``value`` with every ``TIMING_FIELDS`` key removed, recursively."""
    if isinstance(value, Mapping):
        return {key: without_timing(item) for key, item in value.items() if key not in TIMING_FIELDS}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_sha(value: Any) -> str:
    return _text_sha(json.dumps(value, sort_keys=True))


def trial_digest(trial: Path) -> dict[str, str]:
    """``{file: digest}`` of a trial directory: an .npz by the content hash of its arrays, every other file by its
    bytes."""
    found = {}
    for path in sorted(trial.iterdir()):
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                found[path.name] = "arrays:" + params_sha256({key: archive[key] for key in archive.files})
        else:
            found[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def shown_digest(root: Path, replies: list[Any], listed: list[Any]) -> dict[str, Any]:
    """Digests of everything the session showed the editor (module docstring)."""
    return {
        "trial": trial_digest(root / "trial"),
        "system_prompt": _text_sha((root / "prompts" / "system.md").read_text(encoding="utf-8")),
        "task_prompt": _text_sha((root / "prompts" / "task.md").read_text(encoding="utf-8")),
        "tools": _json_sha(listed),
        "replies": [f"{reply.name}:{reply.is_error}:{_json_sha(without_timing(reply.payload))}" for reply in replies],
    }


B_SHOWN_WEIGHTCODE: dict[str, Any] = {'trial': {'experience.npz': 'arrays:df5eb9b6c167c882373d6c7d71feca35292fe288e0c6fc9e889d12cb703ebd2b',
           'forward.py': 'bd0bd988862c996a69bbcb39b979c725a48ce6fb109d2799d2be402a2abf7dc6',
           'model.npz': 'arrays:b680d1cca43a2ba685e4cc2eb7ad91921561168c72dd5cffd4867f91f1766613',
           'model_spec.json': '5191cfd3f285c6b6e51e0afcca85296f786bd1f75273a644113552b59dd15bf2',
           'task.json': '812535932a8d09be994678c8fbe8891d078fd39cb73336f7acce0866f1db7445'},
 'system_prompt': '7471546d860b8ea98809ca0564968cda671a85943780c389f75afe1d6abfc9d3',
 'task_prompt': 'e65ca2dd5edc8a3e27bc8dd77bb1f8608b4541be39cae413ce1382f72c6e9288',
 'tools': '39cd5e0e7cbd06e904be8111307c5708299f3087da050054c7b2d6aebfceaa3b',
 'replies': ['python:False:8491ade115162f5446e5b2dd860b547fdc8af66b4568470b2a730f4b7d52beaa',
             'patch_check:False:af652b72235cd7f8a99e004a3a0c77f64c79a9ee22ddf904a3ae8bd56326581a',
             'patch_check:True:2d69b7c1f969c1a5d7802dc20127130efe0bbbeb9bb55523660de862a16f24fe',
             'patch_check:True:523a75d8a2aed8eea8317f36cd72b142eebd5ed4f80425b07306a0b7ad4e9a9f',
             'patch_check:True:dc1e16f22521fb84803b5f3bd56fcf2f74fce95f93956ce0c4b5f5a0eb50accd',
             'patch_check:True:aae5b80380a9b0b6315741419d0617e0fd9bc345bb017401a11b42692edad82b',
             'patch_check:True:b5c6f41562087d816aaec7917ccb0d547db519fc2a6fea8e64fdded1e01c59f8',
             'patch_check:False:5bd11a17d04d29bb59336cfe72bff562586934c2d85e055534ba3e9a7d6ad6f6',
             'patch_check:True:744bd87291b977c31f8232f306fdb81829b28dde4941e6ec3e2935a3976370ef',
             'status:False:8cd9b48c4b351006d5500a4fac5a37a3c54e13289bc3baf9f0db8e9b03a91f3a',
             'declare_final:False:a5043a99e7ccfc87d8a68d36d02c70a4b3a206d09c17f00f6a949c514bb36432']}
"""Pre-A4 reference (main 9a65b20, numpy 2.5.4), with only the public namespace migrated."""
