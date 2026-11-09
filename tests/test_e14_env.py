"""PREREG E14 environment: pinned minigrid, the new level, the encoding, the full-map bot and greedy rollouts."""

from __future__ import annotations

import numpy as np
import pytest

minigrid = pytest.importorskip("minigrid")

from weightcode.data import e14 as E  # noqa: E402


def test_versions_and_registered_old_level() -> None:
    assert E.environment_versions() == {"minigrid": E.PINNED["minigrid"], "gymnasium": E.PINNED["gymnasium"]}
    import gymnasium as gym

    with E.quiet():
        spec = gym.spec(E.OLD_LEVEL)
    assert spec.entry_point == "minigrid.envs.babyai:GoToLocal"
    assert spec.kwargs == {} and spec.max_episode_steps is None
    env = E.make_env("old")
    assert type(env.unwrapped).__name__ == "GoToLocal"
    assert (env.unwrapped.room_size, env.unwrapped.num_dists) == (8, 8)
    assert type(E.make_env("new").unwrapped).__name__ == E.NEW_LEVEL


def test_same_seed_same_room_under_both_missions() -> None:
    old, new = E.make_env("old"), E.make_env("new")
    for seed in (0, 7, 2_100_003):
        a, b = E.reset(old, seed), E.reset(new, seed)
        assert np.array_equal(a["image"], b["image"])
        assert a["mission"].split()[2:] == b["mission"].split()[2:]
        assert a["mission"].startswith("go to ") and b["mission"].startswith("pick up ")
        assert old.unwrapped.max_steps == new.unwrapped.max_steps == E.MAX_STEPS


def test_rows_and_encoding() -> None:
    env = E.make_env("old")
    row = E.obs_row(E.reset(env, 3))
    assert row.dtype == np.int64 and row.shape == (E.ROW_LEN,)
    E.check_rows(row[None], E.mission_verbs("old"))
    grid, instr = E.split_rows(row[None])
    agent = np.argwhere(grid[0, :, :, 0] == E.OBJECTS.index("agent"))
    assert agent.tolist() == [list(env.unwrapped.agent_pos)]
    assert grid[0, agent[0, 0], agent[0, 1], 2] == env.unwrapped.agent_dir
    assert E.decode_mission(instr[0]) == E.reset(env, 3)["mission"]
    with pytest.raises(E.E14Error):
        E.encode_mission("go to the red ball please")


def test_full_map_bot_uses_only_mission_actions() -> None:
    old = E.demo_arrays(E.bot_episode("old", s) for s in range(60))
    new = E.demo_arrays(E.bot_episode("new", s) for s in range(2_000_000, 2_000_060))
    assert old["ep_usable"].all() and new["ep_usable"].all()
    assert set(np.unique(old["y"]).tolist()) <= set(E.OLD_ACTIONS)
    assert (new["y"] == E.PICKUP).sum() == 60
    E.check_rows(old["x"].astype(np.int64), ("go", "to"))
    E.check_rows(new["x"].astype(np.int64), ("pick", "up"))
    merged = E.merge_demo_chunks([E.demo_chunk(("old", 0, 30)), E.demo_chunk(("old", 30, 60))])
    for key in E.DEMO_KEYS:
        assert np.array_equal(merged[key], old[key]), key


def test_bot_replay_succeeds_in_rollouts() -> None:
    """Replaying the bot's own actions as one-hot logits reproduces its successes (the rollout loop is right)."""
    seeds = list(range(2_100_000, 2_100_012))
    plans = {s: E.bot_episode("new", s).actions for s in seeds}
    calls: list[int] = []

    def logits_fn(x: np.ndarray) -> np.ndarray:
        t = len(calls)  # the step; the batch holds the episodes still running, in seed order
        calls.append(len(x))
        alive = [s for s in seeds if len(plans[s]) > t]
        assert len(alive) == len(x)
        out = np.zeros((len(x), len(E.ACTIONS)))
        for row, s in enumerate(alive):
            out[row, plans[s][t]] = 1.0
        return out

    result = E.rollouts(logits_fn, "new", seeds)
    assert result.success.all()
    assert result.length.tolist() == [len(plans[s]) for s in seeds]
    assert result.n_pickups.tolist() == [1] * len(seeds)


def test_constant_policy_fails_and_is_deterministic() -> None:
    seeds = np.arange(1_000_000, 1_000_020)

    def turn(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), len(E.ACTIONS)))
        out[:, 0] = 1.0
        return out

    a = E.rollouts(turn, "old", seeds)
    b = E.rollouts(turn, "old", seeds)
    assert not a.success.any() and (a.length == E.MAX_STEPS).all()
    assert np.array_equal(a.success, b.success)
    assert E.initial_fingerprint("old", seeds) == E.initial_fingerprint("old", seeds)


def test_sealed_record_form() -> None:
    seeds = E.draw_sealed_seeds()
    arrays = E.sealed_record_arrays(seeds)
    assert len(arrays["x"]) == 2000 and len(set(arrays["x"].tolist())) == 2000
    assert arrays["x"].min() >= E.SEALED_FLOOR and arrays["x"].max() < 2**31
    assert arrays["y"].tolist() == [0] * 1000 + [1] * 1000
    top = max(r.start + r.n for r in E.SEED_RANGES.values())
    assert top < E.SEALED_FLOOR
