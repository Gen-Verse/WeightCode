"""PREREG E10 in the harness (``weightcode.harness.e10``): the leak words, ids and MCP timeouts of an E10 session, the LEARN
limits on a BatchNorm student, a dry run and a scripted session on toy E10 instances, the session clock in every tool
result, the C transfer, a loop stream whose round 1 kept the base, control (a) rounds without a candidate, a teacher
dry run, the round library's programs scanned on E4-loop's standard, and control (b) with the label-free references
under the E10 program limits.

The store is ``e10_toy``: the evalkit toy splits under the E10 dataset name with a BatchNorm MLP student (running
statistics are 1-D parameters, as in the E10 ResNet-20), dev seeds 100 to 105 and the four E10 conditions. The limit
test uses the ResNet-20 tensor shapes themselves. No test touches the real store or starts Claude Code.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from e10_toy import E10Store, e10_store  # noqa: F401
from test_baselines_checked_reference_stream import Grid
from test_harness_fixtures import needs_sandbox
from test_harness_learn import FIX, STRINGS, _program_cell

from weightcode.baselines.fit import checked_reference_stream as C
from weightcode.common.identity import params_sha256
from weightcode.evalkit.e10 import e10_program_limits
from weightcode.evalkit.evidence import EVIDENCE_RULE_V1
from weightcode.evalkit.instances import scan_for_tokens
from weightcode.harness import e10, loop, loop_stream, runner
from weightcode.harness.arms import applied_limits
from weightcode.harness.finalize import KIND_BASE, KIND_CHECKED, STATUS_DRY_RUN
from weightcode.harness.instance import InstanceRefused, read_instance_record
from weightcode.harness.jsonio import read_json
from weightcode.harness.launch import LAUNCH_START_SCHEMA, LaunchStartError, read_launch_start, record_launch_start
from weightcode.harness.leaks import LIBRARY_PROGRAM_RE, scan_trial
from weightcode.harness.loop_stream import MODE_SELECT, MODE_TEACHER, LoopStreamError, run_loop_stream
from weightcode.harness.program import write_derived_patch
from weightcode.harness.scripted import ScriptedLauncher, call_and_record
from weightcode.harness.session import SessionPaths
from weightcode.harness.tools import ToolServer
from weightcode.harness.transfer import run_transfer
from weightcode.patch import Limits, PatchError, apply_patch, task_limits
from weightcode.sandbox.program import ProgramSandbox

BN_SHIFT = STRINGS + """import numpy as np

def update(student, experience):
    var = student["params"]["bn1.running_var"]
    delta = np.zeros_like(var)
    delta[5] = 50.0 * var[5]
    return {"bn1.running_var": delta}
"""
WALL_S = 300.0
CLOCK = ("wall_s_elapsed", "wall_s_remaining")


def _sandbox() -> ProgramSandbox:
    return ProgramSandbox(python=runner.SANDBOX_PYTHON, memory_bytes=4 * 2**30, threads=1)


def _write(directory: Path, name: str, source: str) -> Path:
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------------------------
# Units


def test_the_constants_restate_the_data_module() -> None:
    from weightcode.data import e10 as data

    assert (e10.DATASET, e10.SEVERITY, e10.CORRUPTIONS, e10.FAMILIES) == (data.DATASET, data.SEVERITY,
                                                                         data.CORRUPTIONS, data.FAMILIES)
    assert [e10.condition_name(c) for c in e10.CORRUPTIONS] == [data.condition_name(c) for c in data.CORRUPTIONS]
    assert e10.STREAM_CONDITIONS == ("gaussian_noise_s5", "defocus_blur_s5", "snow_s5", "contrast_s5")
    assert e10.TRANSFER_SEEDS == (101, 102, 103, 104, 105) and e10.SOURCE_SEED == 100


def test_the_leak_words_name_every_corruption_and_the_benchmark() -> None:
    words = e10.leak_words()
    assert words == sorted(set(words)) and all(word == word.lower() for word in words)
    for word in ("gaussian_noise_s5", "gaussian", "noise", "defocus_blur_s5", "defocus", "blur", "snow_s5", "snow",
                 "fog_s5", "fog", "contrast_s5", "contrast", "s5", "severity", "corrupt", "cifar-10-c", "hendrycks",
                 "weather", "digital"):
        assert word in words, word
    assert scan_for_tokens("Trained on CIFAR-10-C at Severity 5", words) == ["cifar-10-c", "severity"]
    assert scan_for_tokens("frog fogged s50 deer", words) == []  # short tokens match whole words only
    assert scan_for_tokens("a fog layer, s5", words) == ["fog", "s5"]
    assert e10.library_session_ids([{"source": {"session_id": "sb"}}, {"source": {"kept": None}},
                                    {"source": {"session_id": "sa"}}, {"program": None}]) == ["sa", "sb"]


def test_an_e10_session_gets_longer_mcp_timeouts_and_refuses_shorter_ones() -> None:
    defaults = (runner.DEFAULT_MCP_TOOL_TIMEOUT_MS, runner.DEFAULT_MCP_CONNECT_TIMEOUT_MS)
    ours = {"dataset": e10.DATASET, "evidence_rule": EVIDENCE_RULE_V1}
    assert e10.mcp_timeouts(ours, None, None, defaults) == (e10.MCP_TOOL_TIMEOUT_MS, e10.MCP_CONNECT_TIMEOUT_MS)
    assert e10.mcp_timeouts(ours, 1_500_000, 1_000_000, defaults) == (1_500_000, 1_000_000)
    for tool, connect in ((600_000, None), (None, 120_000)):
        with pytest.raises(ValueError, match="at least 1200000 ms per tool call and 900000 ms to connect"):
            e10.mcp_timeouts(ours, tool, connect, defaults)
    other = {"dataset": "cifar10", "evidence_rule": EVIDENCE_RULE_V1}
    assert e10.mcp_timeouts(other, None, None, defaults) == defaults
    assert e10.mcp_timeouts(other, 700_000, 130_000, defaults) == (700_000, 130_000)
    assert not e10.is_e10({"dataset": e10.DATASET, "evidence_rule": "fewshot-v1"})


def test_the_learn_limits_on_the_resnet20_tensors_exempt_and_count_the_batchnorm_statistics(tmp_path: Path) -> None:
    from weightcode.models.resnet20a import param_shapes

    params = {name: np.ones(shape, np.float32) for name, shape in param_shapes(10).items()}
    sha = params_sha256(params)
    n_entries = sum(value.size for value in params.values())
    stats = [name for name in params if name.endswith(("running_mean", "running_var"))]
    assert len(params) == 97 and n_entries == 271_098 and len(stats) == 38
    limits = applied_limits(task_limits(Limits()), "LEARN", len(params))
    assert limits == e10_program_limits(task_limits(Limits()), len(params))
    assert (limits.max_ops, limits.max_rel_frob, limits.rel_frob_min_ndim, limits.max_changed_frac) == (97, 0.5, 2, 0.2)

    def apply(deltas: dict[str, np.ndarray], name: str) -> Any:
        directory = tmp_path / name
        directory.mkdir()
        update = SimpleNamespace(deltas=deltas, hypothesis="h", predicted_effect="p")
        patch = write_derived_patch(update, params, sha, directory)  # type: ignore[arg-type]
        return apply_patch(params, patch, directory, limits, sha)

    big_stats = {name: np.full(params[name].shape, 9.0) for name in stats}  # 1-D: rel-Frob 9, exempt
    result = apply(big_stats, "stats")
    assert result.n_changed == sum(params[name].size for name in stats) == 1376
    with pytest.raises(PatchError, match="tensor 'conv1.weight': relative Frobenius norm of the change is 0.6"):
        apply({"conv1.weight": np.full(params["conv1.weight"].shape, 0.6)}, "frob")
    cap = int(0.2 * n_entries)  # 54,219 entries
    conv_a, conv_b = "layer3.1.conv1.weight", "layer3.1.conv2.weight"
    rest = cap - params[conv_a].size - 1376
    partial = np.zeros(params[conv_b].size)
    partial[:rest] = 0.01
    within = {**big_stats, conv_a: np.full(params[conv_a].shape, 0.01), conv_b: partial.reshape(params[conv_b].shape)}
    assert apply(within, "at_cap").n_changed == cap
    over = dict(within, **{conv_b: np.where(np.arange(partial.size) <= rest, 0.01, 0.0).reshape(params[conv_b].shape)})
    with pytest.raises(PatchError, match="size: 54220 of 271098 parameters change .*above the limit 0.2"):
        apply(over, "over")
    without_stats = {key: value for key, value in over.items() if key not in stats}
    assert apply(without_stats, "no_stats").n_changed == cap + 1 - 1376  # the statistics were what broke the cap
    one_each = {name: np.where(np.arange(params[name].size) == 0, 1e-3, 0.0).reshape(params[name].shape)
                for name in params}
    assert apply(one_each, "every_tensor").n_changed == 97  # one op per tensor, max_ops = 97


def test_the_clock_counts_from_the_launch_record(tmp_path: Path) -> None:
    paths = SessionPaths(tmp_path / "session")
    paths.create()
    with pytest.raises(LaunchStartError, match="missing"):
        read_launch_start(paths)
    record = record_launch_start(paths, {"budget": {"wall_timeout_s": 60.0}})
    assert read_launch_start(paths) == record and record["schema"] == LAUNCH_START_SCHEMA
    with pytest.raises(LaunchStartError, match="launched once"):
        record_launch_start(paths, {"budget": {"wall_timeout_s": 60.0}})
    server = SimpleNamespace(launch_unix=1000.0, wall_timeout_s=60.0)
    assert ToolServer.clock(server, now=1012.34) == {"wall_s_elapsed": 12.3, "wall_s_remaining": 47.7}  # type: ignore
    assert ToolServer.clock(server, now=1075.0) == {"wall_s_elapsed": 75.0, "wall_s_remaining": 0.0}  # type: ignore
    assert ToolServer.clock(server, now=990.0) == {"wall_s_elapsed": 0.0, "wall_s_remaining": 60.0}  # type: ignore
    paths.launch_start.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(LaunchStartError, match="is not a"):
        read_launch_start(paths)


# ---------------------------------------------------------------------------------------------------------------
# Sessions on toy E10 instances


def test_the_forbidden_strings_of_an_e10_session(e10_store: E10Store) -> None:
    instance_id = e10_store.ids[("snow_s5", 100)]
    record = read_instance_record(instance_id)
    assert e10.is_e10(record.private) and record.private["condition"] == "snow_s5"
    tokens = e10.self_test_tokens(["s0123456789abcdef"])
    assert sorted(e10_store.ids.values()) == e10.instance_ids() and set(e10.instance_ids()) <= set(tokens)
    forbidden = runner.session_forbidden_strings(record, extra_tokens=tokens)
    assert set(e10.leak_words()) <= set(forbidden.text)
    for token in (*e10_store.ids.values(), "s0123456789abcdef", "gaussian_noise_s5", "defocus_blur", "contrast_s5",
                  "fog_s5", "snow_s5"):
        assert token in forbidden.text and token in forbidden.selftest, token
    for word in ("fog", "s5", "noise", "snow", "blur", "severity"):  # text scan only: they occur in host paths
        assert word not in forbidden.selftest, word
    other = dataclasses.replace(record, private={**record.private, "dataset": "toyset"})
    with pytest.raises(InstanceRefused, match="extra forbidden ids apply to an fscil-v1 or an E10 instance"):
        runner.session_forbidden_strings(other, extra_tokens=tokens)
    plain = runner.session_forbidden_strings(other).text
    assert "fog" not in plain and "hendrycks" not in plain and "defocus_blur" not in plain


def test_the_stream_and_transfer_lookups(e10_store: E10Store) -> None:
    assert e10.stream_ids() == e10_store.stream(100)
    assert e10.stream_ids(103) == e10_store.stream(103)
    source = e10_store.ids[("defocus_blur_s5", 100)]
    assert e10.transfer_targets(source) == [e10_store.ids[("defocus_blur_s5", seed)] for seed in e10.TRANSFER_SEEDS]
    with pytest.raises(e10.E10HarnessError, match="C transfers seed 100"):
        e10.transfer_targets(e10_store.ids[("defocus_blur_s5", 101)])
    assert e10.main(["targets", source]) == 0


@needs_sandbox
def test_an_e10_dry_run_states_the_floor_the_limits_and_the_clock(e10_store: E10Store) -> None:
    instance_id = e10_store.ids[("gaussian_noise_s5", 100)]
    result = runner.run_session(instance_id, "LEARN", 0, effort="max", wall_timeout_s=5400.0, dry_run=True,
                                retention_floor_frac=0.005)
    assert result.status == STATUS_DRY_RUN and result.root.parent.name == runner.DRY_RUNS_DIR
    session = read_json(result.root / "session.json")
    assert session["mcp"]["tool_timeout_ms"] == e10.MCP_TOOL_TIMEOUT_MS
    assert session["mcp"]["connect_timeout_ms"] == e10.MCP_CONNECT_TIMEOUT_MS
    n_clean = int(session["clean_check"]["n"])
    assert session["kept_rule"]["s_old_floor"] == -round(0.005 * n_clean) and session["leak_scan"]["passed"]
    limits = session["limits"]
    assert (limits["max_ops"], limits["max_rel_frob"], limits["max_changed_frac"]) == (64, 0.5, 0.2)
    assert set(e10.leak_words()) <= set(session["forbidden_strings"]["text"])
    assert set(e10_store.ids.values()) <= set(session["forbidden_strings"]["selftest"])
    task = " ".join((result.root / "prompts" / "task.md").read_text().split())
    assert "`wall_s_elapsed` and `wall_s_remaining`: the seconds since this session started" in task
    with pytest.raises(InstanceRefused, match="MCP timeouts of at least"):
        runner.run_session(instance_id, "LEARN", 0, effort="max", wall_timeout_s=5400.0, dry_run=True,
                           retention_floor_frac=0.005, mcp_tool_timeout_ms=600_000)


@needs_sandbox
def test_a_scripted_e10_session_checks_a_batchnorm_program_and_every_result_has_the_clock(
        e10_store: E10Store) -> None:
    instance_id = e10_store.ids[("contrast_s5", 100)]

    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "python", {"code": _program_cell({"bn.py": BN_SHIFT, "fix.py": FIX})})
        call_and_record(client, replies, "patch_check", {"program_file": "bn.py"})
        call_and_record(client, replies, "patch_check", {"program_file": "fix.py"})
        call_and_record(client, replies, "patch_check", {"program_file": "missing.py"})
        call_and_record(client, replies, "status", {"unknown": 1})
        call_and_record(client, replies, "status", {})
        call_and_record(client, replies, "declare_final", {"program_file": "fix.py", "rationale": "it scored"})

    launcher = ScriptedLauncher(script)
    result = runner.run_session(instance_id, "LEARN", 0, effort="low", wall_timeout_s=WALL_S, launcher=launcher,
                                skip_audit=True, retention_floor_frac=0.005)
    assert result.status == "complete", result.record.get("infra_reasons")
    replies = launcher.replies
    assert [reply.is_error for reply in replies] == [False, False, False, True, True, False, False]
    elapsed = [reply.payload["wall_s_elapsed"] for reply in replies]
    assert elapsed == sorted(elapsed) and 0.0 <= elapsed[0] and elapsed[-1] < WALL_S
    for reply in replies:
        assert set(CLOCK) <= set(reply.payload), reply.name
        assert abs(reply.payload["wall_s_remaining"] - (WALL_S - reply.payload["wall_s_elapsed"])) <= 0.11
    bn, fix = replies[1].payload, replies[2].payload
    assert bn["tensors"] == ["bn1.running_var"] and fix["tensors"] == ["fc1.weight"] and fix["score"] > 0
    paths = SessionPaths(result.root)
    launch = read_launch_start(paths)
    assert launch["wall_timeout_s"] == WALL_S
    final = read_json(paths.selection_json)["final"]
    assert final["kind"] == KIND_CHECKED and final["patch_id"] == "fix.py"
    checks = [json.loads(line) for line in paths.checks_log.read_text().splitlines()]
    assert [entry["patch_id"] for entry in checks] == ["bn.py", "fix.py"]
    assert checks[0]["n_changed"] == 1  # a 51-fold running variance: 1-D, exempt from rel-Frob, one entry changed


@needs_sandbox
def test_the_c_transfer_runs_the_frozen_program_on_the_other_students(e10_store: E10Store, tmp_path: Path) -> None:
    source = e10_store.ids[("gaussian_noise_s5", 100)]
    targets = e10.transfer_targets(source)
    fix = _write(tmp_path, "fix.py", FIX)
    written = run_transfer(fix, [source, *targets], tmp_path / "transfer", _sandbox())
    records = [json.loads(path.read_text()) for path in written]
    assert [record["target"]["instance_id"] for record in records] == [source, *targets]
    for record in records:
        assert record["status"] == "ok" and record["within_limits"] is True and record["tensors"] == ["fc1.weight"]
        assert record["readouts"]["delta_acc"] is not None and record["readouts_clean"]["delta_acc"] is not None


# ---------------------------------------------------------------------------------------------------------------
# The stream and its controls


def _base_session(instance_id: str) -> Path:
    """A finished scripted LEARN session that checks nothing, so its final is the base: a B final that kept the base."""

    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "status", {})

    result = runner.run_session(instance_id, "LEARN", 0, effort="low", wall_timeout_s=WALL_S,
                                launcher=ScriptedLauncher(script), skip_audit=True, retention_floor_frac=0.005)
    final = read_json(result.root / "final" / "selection.json")["final"]
    assert result.status == "complete" and final["kind"] == KIND_BASE and final["is_base_noop"]
    return result.root


@needs_sandbox
def test_a_stream_whose_round_one_kept_the_base_and_control_a_rounds_without_a_candidate(
        e10_store: E10Store, tmp_path: Path) -> None:
    ids = e10_store.stream(100)
    root1 = _base_session(ids[0])
    fix = _write(tmp_path, "fix.py", FIX)
    with pytest.raises(LoopStreamError, match="kept no checked program"):
        run_loop_stream(MODE_SELECT, ids, fix, root1, tmp_path / "refused", _sandbox(),
                        candidates={2: [], 3: [fix], 4: []})
    record = read_json(run_loop_stream(MODE_SELECT, ids, None, root1, tmp_path / "loops", _sandbox(),
                                       candidates={2: [], 3: [fix], 4: []}))
    assert [row["status"] for row in record["rounds"]] == ["noop", "noop", "ok", "noop"]
    one, two, three, four = record["rounds"]
    assert record["round1"]["kept_base"] is True and record["round1"]["program"] is None
    assert one["library_entry"] == {"round": 1, "program": None, "scores": None,
                                    "source": {"session_id": root1.name}}
    assert one["child_sha256"] == two["child_sha256"] == record["base_params_sha256"]
    assert one["experience"]["content_sha256"] and two["queries_used"] == 0 and two["candidates"] == []
    assert list(two["kept_rule"]["n_prev"].values()) == [int(read_instance_record(ids[0]).private["n_vcheck"])]
    assert three["kept"] == "candidate_1" and three["queries_used"] == 1
    assert four["child_sha256"] == three["child_sha256"] and four["queries_used"] == 0
    assert record["summary"]["n_rounds_ok"] == 1
    with pytest.raises(SystemExit):
        loop_stream.main(["--mode", MODE_SELECT, "--instances", *ids, "--round1-session", str(root1), "--out",
                          str(tmp_path / "cli"), "--candidates", "2=none", f"2={fix}"])
    rows = C.loop_round_rows(Path(record["work_dir"]).with_suffix(".json"))
    assert [row["round"] for row in rows] == [1, 2, 3, 4]


@needs_sandbox
def test_a_teacher_dry_run_prepares_round_two_on_the_round_one_student(e10_store: E10Store, tmp_path: Path) -> None:
    ids = e10_store.stream(100)[:2]
    root1 = _base_session(ids[0])
    kwargs = {"effort": "max", "wall_timeout_s": 5400.0}
    with pytest.raises(LoopStreamError, match="a dry run is a teacher stream's"):
        run_loop_stream(MODE_SELECT, ids, None, root1, tmp_path / "x", _sandbox(), candidates={2: []}, dry_run=True)
    path = run_loop_stream(MODE_TEACHER, ids, None, root1, tmp_path / "dry", _sandbox(), session_kwargs=kwargs,
                           dry_run=True)
    dry = read_json(path)
    assert dry["round"] == 2 and dry["status"] == STATUS_DRY_RUN and dry["instance_id"] == ids[1]
    session = read_json(Path(dry["root"]) / "session.json")
    assert Path(dry["root"]).parent.name == runner.DRY_RUNS_DIR and session["loop"]["round"] == 2
    assert session["loop"]["prev_instances"] == [ids[0]] and session["leak_scan"]["passed"]
    assert root1.name in session["forbidden_strings"]["selftest"]  # the library's source session
    assert session["mcp"]["connect_timeout_ms"] == e10.MCP_CONNECT_TIMEOUT_MS
    index = json.loads((Path(dry["root"]) / "trial" / "library_index.json").read_text())
    assert [(r["round"], r["file"], r["kept_update"]) for r in index["rounds"]] == [(1, None, False)]


def test_library_programs_are_scanned_on_e4_loop_standard_and_the_rest_on_the_e10_list(
        e10_store: E10Store, tmp_path: Path) -> None:
    session_id = "s0123456789abcdef"
    record = read_instance_record(e10_store.ids[("defocus_blur_s5", 100)])
    forbidden = runner.session_forbidden_strings(record, (), extra_tokens=e10.self_test_tokens((session_id,)))
    assert forbidden.library_scan == forbidden.library_text and set(forbidden.library_text) < set(forbidden.text)
    assert {"contrast", "noise", "blur", "snow", "s5", "severity"} <= set(forbidden.text) - set(forbidden.library_text)
    assert forbidden.as_dict()["library_text"] == list(forbidden.library_text)
    other = e10_store.ids[("snow_s5", 103)]
    heads = {1: "# the new images have lower contrast, more noise, some blur and snow\n",
             2: "# as on gaussian_noise_s5\n", 3: f"# fitted on {other}\n", 4: f"# from session {session_id}\n"}
    expected = {2: "gaussian_noise_s5", 3: other, 4: session_id}
    library = []
    for k, head in heads.items():
        library.append({"round": k, "program": {"path": str(_write(tmp_path, f"p{k}.py", head + FIX))}})
    withheld = loop.withheld_tokens(library, forbidden.library_scan)
    assert sorted(withheld) == [2, 3, 4] and all(expected[k] in withheld[k] for k in expected)
    assert loop.withheld_tokens(library[:1], forbidden.text) == {1: ["blur", "contrast", "noise", "snow"]}
    trial = tmp_path / "trial"
    trial.mkdir()
    _write(trial, "library_program_1.py", heads[1] + FIX)
    _write(trial, "library_program_2.py", heads[2] + FIX)
    _write(trial, "notes.json", json.dumps({"note": "lower contrast"}))
    hits = scan_trial(trial, forbidden.text, library_tokens=forbidden.library_text)
    assert sorted(hits) == ["trial/library_program_2.py", "trial/notes.json"]
    assert hits["trial/notes.json"] == ["contrast"]
    assert sorted(scan_trial(trial, forbidden.text)) == sorted(f"trial/{p.name}" for p in trial.iterdir())
    assert [bool(LIBRARY_PROGRAM_RE.fullmatch(name)) for name in (loop.LOOP_PROGRAM_TEMPLATE.format(k=1),
            loop.LOOP_PROGRAM_TEMPLATE.format(k=12), "library_program_0.py", loop.LOOP_LIBRARY_INDEX)] == [
        True, True, False, False]
    plain = runner.session_forbidden_strings(read_instance_record(e10_store.ids[("snow_s5", 100)]), ())
    assert plain.library_text is not None  # every E10 session; a non-E10 session keeps None (test_harness_loop)


def _kept_session(instance_id: str, source: str, repeat: int) -> tuple[Path, Path]:
    """A finished scripted LEARN session that checks ``source`` and keeps it: the root and the kept program file."""

    def script(client: Any, replies: list[Any]) -> None:
        call_and_record(client, replies, "python", {"code": _program_cell({"prog.py": source})})
        call_and_record(client, replies, "patch_check", {"program_file": "prog.py"})
        call_and_record(client, replies, "declare_final", {"program_file": "prog.py", "rationale": "it scored"})

    result = runner.run_session(instance_id, "LEARN", repeat, effort="low", wall_timeout_s=WALL_S,
                                launcher=ScriptedLauncher(script), skip_audit=True, retention_floor_frac=0.005)
    final = read_json(result.root / "final" / "selection.json")["final"]
    assert result.status == "complete" and final["kind"] == KIND_CHECKED and final["score"] > 0
    program = result.root / read_json(result.root / "run.json")["final"]["program"]["file"]
    assert program.read_text(encoding="utf-8") == source
    return result.root, program


@needs_sandbox
def test_a_library_program_naming_conditions_in_words_is_shown_and_one_naming_ids_is_withheld(
        e10_store: E10Store, tmp_path: Path) -> None:
    ids = e10_store.stream(100)[:2]
    kwargs = {"effort": "max", "wall_timeout_s": 5400.0}
    other = e10_store.ids[("snow_s5", 103)]
    cases = [("shown", "# the new images have lower contrast and more noise\n" + FIX),
             ("withheld", f"# fitted for defocus_blur_s5 on {other}\n" + FIX)]
    for repeat, (name, source) in enumerate(cases):
        root1, program = _kept_session(ids[0], source, repeat)
        dry = read_json(run_loop_stream(MODE_TEACHER, ids, program, root1, tmp_path / name, _sandbox(),
                                        session_kwargs=kwargs, dry_run=True))
        root = Path(dry["root"])
        session = read_json(root / "session.json")
        scan = session["leak_scan"]
        assert dry["status"] == STATUS_DRY_RUN and scan["passed"], scan["hits"]
        assert {"contrast", "noise"} <= set(scan["tokens"]) and not {"contrast", "noise"} & set(scan["library_tokens"])
        index = json.loads((root / "trial" / "library_index.json").read_text())
        shown = root / "trial" / "library_program_1.py"
        if name == "shown":  # in full, as written
            assert [(r["file"], r["kept_update"]) for r in index["rounds"]] == [("library_program_1.py", True)]
            assert shown.read_text(encoding="utf-8") == source and session["library"]["withheld"] == {}
        else:  # an underscored condition name and an instance id: scores only
            assert [(r["file"], r["kept_update"]) for r in index["rounds"]] == [(None, True)]
            assert not shown.exists() and index["rounds"][0]["scores"]["score"] > 0
            assert {"defocus_blur_s5", other} <= set(session["library"]["withheld"]["1"])


def _stats(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"bn1.running_var": np.full_like(params["bn1.running_var"], 0.5),
            "bn1.running_mean": np.full_like(params["bn1.running_mean"], 0.01)}


def _wide(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"fc1.weight": np.full_like(params["fc1.weight"], 1e-3)}  # 128 of 196 entries: above the 20% cap


def test_control_b_fits_the_condition_inputs_alone_under_the_e10_limits(e10_store: E10Store, tmp_path: Path) -> None:
    ids = e10_store.stream(100)[:2]
    grid = Grid([("stats", _stats), ("wide", _wide)])
    record = json.loads(C.run_checked_reference_stream("bn_adapt", ids, tmp_path / "b", make_generator=grid)
                        .read_text())
    instance = read_instance_record(ids[0])
    expected = task_limits(e10_program_limits(instance.task["limits"], 7))
    assert record["variant"] == C.LABEL_FREE_VARIANT and record["edit_limits_enforced"] is True
    for row, seen in zip(record["rounds"], grid.seen):
        assert row["limits"] == expected
        assert row["fit_rows"]["n_replay"] == 0 and row["fit_rows"]["labels_read"] is False
        assert seen["n_replay"] == 0 and len(seen["x"]) == row["fit_rows"]["n_round_evidence"] == 64
        statuses = {candidate["config"]["name"]: candidate["status"] for candidate in row["candidates"]}
        assert statuses == {"stats": "checked", "wide": "patch_refused"}
    full = Grid([("stats", _stats), ("wide", _wide)])
    record = json.loads(C.run_checked_reference_stream("grad_full", ids, tmp_path / "f", make_generator=full)
                        .read_text())
    assert record["edit_limits_enforced"] is False and full.seen[0]["n_replay"] == 8  # E1's rows with the replay
    assert {c["config"]["name"]: c["status"] for c in record["rounds"][0]["candidates"]} == {"stats": "checked",
                                                                                              "wide": "checked"}
    assert C.method_spec("tent").uses_labels_for_fit is False and set(C.METHODS) == {
        "grad_last", "ridge_last", "grad_full", "bn_adapt", "tent"}
