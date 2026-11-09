"""PREREG E14 T11 controls: the external rules (trigger, dead-end, verb copy) on hand-built observation rows, the
rollout trace and the per-episode features."""

from __future__ import annotations

import numpy as np
import pytest

from weightcode.data import e14 as D
from weightcode.evalkit import e14 as E

EMPTY, WALL, AGENT = (D.OBJECTS.index(name) for name in ("empty", "wall", "agent"))
BALL, BOX, KEY = (D.OBJECTS.index(name) for name in ("ball", "box", "key"))
RED, BLUE, GREY = (D.COLORS.index(name) for name in ("red", "blue", "grey"))
FORWARD, LEFT, RIGHT = (D.ACTIONS.index(name) for name in ("forward", "left", "right"))
EAST, WEST = 0, 2
BASE_LOGITS = np.array([0.0, 1.0, 5.0, -20.0, -30.0, -30.0, -30.0], dtype=np.float32)
"""The stub policy's logits: forward first, right second, pickup far below."""


def room() -> np.ndarray:
    grid = np.zeros((D.GRID, D.GRID, D.N_CHANNELS), dtype=np.int64)
    grid[..., 0] = EMPTY
    grid[[0, -1], :, :] = (WALL, GREY, 0)
    grid[:, [0, -1], :] = (WALL, GREY, 0)
    return grid


def row(front: tuple[int, int] | None, agent_at: tuple[int, int] = (3, 3), direction: int = EAST,
        words: tuple[str, ...] = ("pick", "up", "the", "red", "ball")) -> np.ndarray:
    """One observation row: the agent at ``agent_at`` facing ``direction``; ``front`` (object, color) placed in the
    faced cell when given (else the faced cell keeps what the room has there)."""
    grid = room()
    ax, ay = agent_at
    grid[ax, ay] = (AGENT, RED, direction)
    if front is not None:
        dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[direction]
        grid[ax + dx, ay + dy] = (front[0], front[1], 0)
    return np.concatenate([grid.reshape(-1), D.encode_mission(" ".join(words))])


def hand_rows() -> dict[str, np.ndarray]:
    return {
        "trigger": row((BALL, RED)),
        "object": row((BOX, BLUE)),
        "wall": row(None, agent_at=(1, 3), direction=WEST),
        "wrong": row((KEY, RED)),
        "free": row(None),
    }


def stub_forward(params: dict, x: np.ndarray) -> np.ndarray:
    """Fixed logits; left gets +10 when the first verb is go, so a test can see which verbs the policy read."""
    out = np.tile(BASE_LOGITS, (len(x), 1))
    out[:, LEFT] += 10.0 * (x[:, D.OBS_LEN] == D.WORD_ID["go"])
    return out.astype(np.float32)


PARAMS = {"w": np.zeros(1, dtype=np.float32)}


def stacked(names: list[str]) -> np.ndarray:
    rows = hand_rows()
    return np.stack([rows[name] for name in names])


def test_faced_cells_on_hand_rows() -> None:
    names = ["trigger", "object", "wall", "wrong", "free"]
    cells = E.faced_cells(stacked(names))
    assert cells.obj.tolist() == [BALL, BOX, WALL, KEY, EMPTY]
    assert cells.trigger.tolist() == [True, False, False, False, False]
    assert cells.blocked.tolist() == [True, True, True, True, False]
    assert cells.nontarget_object.tolist() == [False, True, False, True, False]
    assert cells.wrong_target.tolist() == [False, False, False, True, False]


def test_faced_cells_refuses_bad_rows() -> None:
    bad = hand_rows()["free"].copy()
    bad[: D.OBS_LEN].reshape(D.GRID, D.GRID, D.N_CHANNELS)[5, 5] = (AGENT, RED, 0)  # two agents
    with pytest.raises(E.E14InstanceError):
        E.faced_cells(bad[None])
    door = row((D.OBJECTS.index("door"), RED))
    with pytest.raises(E.E14InstanceError):
        E.faced_cells(door[None])


def test_trigger_only_rule_equals_the_wrapper_of_record() -> None:
    x = stacked(["trigger", "object", "wall", "wrong", "free"])
    rng = np.random.default_rng(0)
    extra = []
    for _ in range(40):  # random positions, directions, faced objects and instructions
        ax, ay = int(rng.integers(1, D.GRID - 1)), int(rng.integers(1, D.GRID - 1))
        direction = int(rng.integers(4))
        obj = int(rng.choice([BALL, BOX, KEY]))
        color = int(rng.integers(len(D.COLORS)))
        words = ("pick", "up", "a", D.COLORS[int(rng.integers(len(D.COLORS)))], D.OLD_TYPES[int(rng.integers(3))])
        dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[direction]
        inside = 0 < ax + dx < D.GRID - 1 and 0 < ay + dy < D.GRID - 1
        extra.append(row((obj, color) if inside else None, (ax, ay), direction, words))
    x = np.concatenate([x, np.stack(extra)])
    ours = E.control_logits_fn(stub_forward, PARAMS, trigger=True, deadend=False, goto=False)(x)
    record = E.wrapper_logits_fn(stub_forward, PARAMS)(x)
    assert ours.dtype == record.dtype == np.float32
    assert np.array_equal(ours, record)


def test_wrapper_v2_on_trigger_blocked_and_free_rows() -> None:
    names = ["trigger", "object", "wall", "wrong", "free"]
    x = stacked(names)
    out = E.wrapper_v2_logits_fn(stub_forward, PARAMS)(x)
    base = stub_forward(PARAMS, x)
    assert out.argmax(axis=1).tolist() == [D.PICKUP, RIGHT, RIGHT, RIGHT, FORWARD]
    for i, name in enumerate(names[:4]):  # blocked: forward below the row's minimum, nothing else moved but pickup
        assert out[i, FORWARD] == np.float32(base[i].min() - 1.0)
        keep = [a for a in range(len(D.ACTIONS)) if a not in (FORWARD, D.PICKUP)]
        assert np.array_equal(out[i, keep], base[i, keep]), name
    # the trigger is applied after the dead-end rule: pickup is the lowered row's maximum (right, 1.0) plus 1
    assert out[0, D.PICKUP] == np.float32(2.0)
    assert np.array_equal(out[1:, D.PICKUP], base[1:, D.PICKUP])
    assert np.array_equal(out[4], base[4])  # a free row is untouched


def test_deadend_only_has_no_trigger() -> None:
    x = stacked(["trigger", "object", "wall", "free"])
    out = E.deadend_only_logits_fn(stub_forward, PARAMS)(x)
    base = stub_forward(PARAMS, x)
    assert out.argmax(axis=1).tolist() == [RIGHT, RIGHT, RIGHT, FORWARD]
    assert np.array_equal(out[:, D.PICKUP], base[:, D.PICKUP])
    assert np.array_equal(out[3], base[3])


def test_goto_variants_read_go_and_to() -> None:
    x = stacked(["trigger", "object", "free"])
    old = row(None, words=("go", "to", "the", "red", "ball"))
    mapped = E.goto_rows(np.concatenate([x, old[None]]))
    assert mapped[:, D.OBS_LEN].tolist() == [D.WORD_ID["go"]] * 4
    assert mapped[:, D.OBS_LEN + 1].tolist() == [D.WORD_ID["to"]] * 4
    assert np.array_equal(mapped[:, : D.OBS_LEN], np.concatenate([x, old[None]])[:, : D.OBS_LEN])
    assert np.array_equal(mapped[3], old)  # an old-mission row is unchanged
    v1 = E.control_logits_fn(stub_forward, PARAMS, **E.T11_RULES["wrapper_v1_goto"])(x)
    v2 = E.control_logits_fn(stub_forward, PARAMS, **E.T11_RULES["wrapper_v2_goto"])(x)
    assert v1.argmax(axis=1).tolist() == [D.PICKUP, LEFT, LEFT]  # left +10 under go: the base read the copied verbs
    assert v2.argmax(axis=1).tolist() == [D.PICKUP, LEFT, LEFT]
    assert np.array_equal(v1[2], stub_forward(PARAMS, E.goto_rows(x))[2])
    assert v2[1, FORWARD] < v2[1].max() and v1[1, FORWARD] == stub_forward(PARAMS, E.goto_rows(x))[1, FORWARD]


def _fake_rollout(trace: E.RolloutTrace, episodes: list[list[np.ndarray]]) -> np.ndarray:
    """Call ``trace`` as ``rollouts`` does: step t on the episodes with more than t rows, in order."""
    lengths = np.array([len(e) for e in episodes])
    for t in range(int(lengths.max())):
        running = np.flatnonzero(lengths > t)
        trace(np.stack([episodes[i][t] for i in running]))
    return lengths


def test_trace_assigns_rows_to_episodes() -> None:
    rows = hand_rows()
    episodes = [[rows["free"], rows["object"], rows["trigger"]], [rows["wall"]], [rows["free"], rows["free"]]]
    trace = E.RolloutTrace(lambda x: stub_forward(PARAMS, x))
    lengths = _fake_rollout(trace, episodes)
    (steps,) = trace.steps([lengths])
    assert steps["obj"][0, :3].tolist() == [EMPTY, BOX, BALL] and steps["obj"][0, 3] == -1
    assert steps["obj"][1, :2].tolist() == [WALL, -1]
    assert steps["obj"][2, :3].tolist() == [EMPTY, EMPTY, -1]
    assert steps["action"][0, :3].tolist() == [FORWARD] * 3
    assert steps["want_type"].tolist() == [BALL] * 3 and steps["want_color"].tolist() == [RED] * 3
    with pytest.raises(E.E14InstanceError):
        trace.steps([np.array([3, 2, 2])])  # lengths that do not fit the calls


def test_trajectory_features_and_categories() -> None:
    rows = hand_rows()
    episodes = [
        [rows["free"], rows["object"], rows["trigger"]],  # blocked by a non-target object, then faces the target
        [rows["wall"], rows["wall"], rows["wall"]],  # forward into a wall, never the target
        [rows["free"], rows["wrong"], rows["free"]],  # faces a partial match
        [rows["trigger"], rows["free"]],  # faces the target first: nothing before
    ]
    trace = E.RolloutTrace(lambda x: stub_forward(PARAMS, x))
    lengths = _fake_rollout(trace, episodes)
    (steps,) = trace.steps([lengths])
    feats = E.trajectory_features(steps, np.array([False, False, False, False]))
    assert feats["faced_target"].tolist() == [True, False, False, True]
    assert feats["before_blocked"].tolist() == [True, False, True, False]
    assert feats["before_wrong"].tolist() == [False, False, True, False]
    assert feats["before_stuck"].tolist() == [True, True, True, False]  # the stub presses forward everywhere
    assert feats["no_target"].tolist() == [False, True, True, False]
    assert feats["forward_blocked_steps"].tolist() == [2, 3, 1, 1]
    assert feats["carried_nontarget"].tolist() == [False] * 4 and feats["drops"].tolist() == [0] * 4
    cats = E.episode_categories(feats)
    assert sorted(cats) == sorted(E.T11_CATEGORIES)
    assert cats["plain"].tolist() == [False, False, False, True]

    def picker(x: np.ndarray) -> np.ndarray:  # pickup on every row: a wrong pickup in episode 2
        out = stub_forward(PARAMS, x)
        out[:, D.PICKUP] = 100.0
        return out

    trace = E.RolloutTrace(picker)
    lengths = _fake_rollout(trace, [[rows["free"], rows["wrong"], rows["trigger"]], [rows["trigger"]]])
    feats = E.trajectory_features(trace.steps([lengths])[0], np.array([False, True]))
    assert feats["carried_nontarget"].tolist() == [True, False]
    assert feats["no_target"].tolist() == [False, False]


def test_trace_on_real_rollouts() -> None:
    pytest.importorskip("minigrid")
    seeds = D.SEED_RANGES["smoke_new"].seeds[:6]
    trace = E.RolloutTrace(lambda x: stub_forward(PARAMS, x))  # always forward: every episode presses into a blocker
    result = D.rollouts(trace, "new", seeds)
    (steps,) = trace.steps([result.length])
    assert np.all(result.length == D.MAX_STEPS) and not result.success.any()
    feats = E.trajectory_features(steps, result.success)
    assert np.all(feats["forward_blocked_steps"] > 0)
    assert np.all(feats["before_stuck"] | feats["faced_target"])
    assert np.array_equal(steps["want_type"], np.array([D.OBJECTS.index(D.OLD_TYPES[t]) for t in result.obj_type]))
    assert np.array_equal(steps["want_color"], result.obj_color)
