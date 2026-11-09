"""PREREG E14 (BabyAI, a new skill in a trained policy): the environment, its encoding, the episode sets and rollouts.

Environment. minigrid (Farama Foundation) 3.2.0 with gymnasium 1.4.0 (``PINNED``; wheel sha256 in ``WHEELS``). The
old mission is BabyAI-GoToLocal-v0 (``minigrid.envs.babyai.goto.GoToLocal`` with its registered defaults, room 8,
8 distractors): "go to the {color} {type}" in one 8 x 8 room. minigrid registers no level of exactly the new form
(PickupDist is a 7 x 7 room with 5 distractors and type-only or color-only descriptors), so the new mission is
``GoToLocalPickup``, a subclass of GoToLocal whose ``gen_mission`` is GoToLocal's line for line with
``PickupInstr(ObjDesc(obj.type, obj.color))`` (strict False, minigrid's default) in place of ``GoToInstr``. The
random draws are the same, so one seed gives the same room, objects, agent and target under both missions; only the
instruction differs. Success is the verifier's: the agent faces the target (old) or picks it up (new). Both levels
end at max_steps = 64 (``RoomGridLevel``: room_size ** 2 for one navigation).

Observation (``obs_row``): the fully observed symbolic grid of ``minigrid.wrappers.FullyObsWrapper``, uint8
[8, 8, 3] indexed [x, y, channel] with channel 0 the object index, 1 the color index and 2 the state (door state, or
the agent's direction on the agent's cell), flattened in that C order, followed by the instruction as 5 word ids
(``VOCAB``). One row is int64 [197]. The agent's position and direction are its cell in the grid. The carried
object is not encoded (FullyObsWrapper does not encode it; neither mission carries before success).

Demonstrations (``bot_episode``): minigrid's BabyAI bot (``minigrid.utils.baby_ai_bot.BabyAIBot``) with its
visibility mask set to the whole grid at the start of the episode, so it plans on the full map, as the student sees
it. With the default partial-view mask the bot explores and in 1,000 GoToLocal episodes took 2 pickup and 2 drop
actions (moving a blocker it had not seen around) and gave 13 conflicting labels for repeated (observation,
instruction) pairs; with the full map it took neither and gave none (probe of 2026-11-06, before any E14 data).
An old demonstration whose bot actions leave {left, right, forward}, or that does not succeed, is not used and is
counted (``DemoSet.n_excluded``); a new one whose actions leave {left, right, forward, pickup} likewise.

Episode sets (``SEED_RANGES``): disjoint environment seed ranges, fixed before any training. The sealed episodes
(1,000 old and 1,000 new) are seeds drawn once by the operating system's generator from [``SEALED_FLOOR``, 2**31),
above every dev range, and written once to ``$WEIGHTCODE_SEALED/<DATASET>/test.npz`` (``write_sealed_record``); only its
hash is recorded elsewhere, and nothing in dev runs them.

Rollouts (``rollouts``): greedy (argmax of the 7 logits, first index on ties), so an episode is a deterministic
function of (weights, mission, seed). Every episode of a call steps in lockstep and the forward runs once per step
on the rows of the episodes still running. minigrid prints "Sampling rejected" to stdout during some resets and
pygame prints a banner on import; both are captured here, so a caller's stdout (the MCP server's protocol stream)
never receives them.
"""

from __future__ import annotations

import contextlib
import functools
import io
import os
import secrets
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DATASET = "e14_gotolocal"
"""Store dataset id: ``$WEIGHTCODE_PRIVATE/e14_gotolocal`` (demonstrations, dataset.json, models) and the sealed record."""

PINNED = {"minigrid": "3.2.0", "gymnasium": "1.4.0", "pygame-ce": "2.5.8", "cloudpickle": "3.1.2",
          "farama-notifications": "0.0.6", "typing-extensions": "4.16.0"}
WHEELS = {
    "minigrid-3.2.0-py3-none-any.whl": "9de2d46320357543e5f665ba0e8590f4d6ca3f4aa8c56562e7f1e506973a73f3",
    "gymnasium-1.4.0-py3-none-any.whl": "1cb947c59e7c72d8eabb2c2274c00cd20948e69f8452a9ef4092223be24fdf0e",
    "cloudpickle-3.1.2-py3-none-any.whl": "9acb47f6afd73f60dc1df93bb801b472f05ff42fa6c84167d25cb206be1fbf4a",
    "farama_notifications-0.0.6-py3-none-any.whl": "f84839188efa1ce5bb361c2a84881b2dc2c0d0d7fb661ff00421820170930935",
    "typing_extensions-4.16.0-py3-none-any.whl": "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8",
    "pygame_ce-2.5.8-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl":
        "bf9c8b4376077efec6ae31443172ae98545f278d0e4426852e81c28ab1f5d7c2",
    "pygame_ce-2.5.8-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.whl":
        "0027bd2255cbb39789d9c7ff95a5f040f56d1cfeb09504497447c0ee06169a24",
}
"""PyPI sha256 of each wheel (pygame-ce for the NSCC container's Python 3.12 and the workstation's 3.14)."""

OLD_LEVEL = "BabyAI-GoToLocal-v0"
NEW_LEVEL = "GoToLocalPickup"
MISSIONS = ("old", "new")
MISSION_CODES = {"old": 0, "new": 1}
LEAK_WORDS = ("babyai-gotolocal-v0", "gotolocal", "gotolocalpickup", "pickupinstr", "babyai", "minigrid")
"""PREREG E14 leak words: the level names (whole, and the GoToLocal stem) and "PickupInstr"; with them the platform
and library names, on E10's standard of adding the benchmark's names. Scanned in every editor-visible text."""

GRID = 8
N_CHANNELS = 3
INSTR_LEN = 5
OBS_LEN = GRID * GRID * N_CHANNELS
ROW_LEN = OBS_LEN + INSTR_LEN
MAX_STEPS = 64

OBJECTS = ("unseen", "empty", "wall", "floor", "door", "key", "ball", "box", "goal", "lava", "agent")
COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
N_STATES = 4
"""Channel 2: door 0 open, 1 closed, 2 locked; on the agent's cell its direction 0..3; 0 on every other cell."""
DIRECTIONS = ("east (+x)", "south (+y)", "west (-x)", "north (-y)")
ACTIONS = ("left", "right", "forward", "pickup", "drop", "toggle", "done")
ACTION_MEANINGS = {
    "left": "turn left (direction - 1 mod 4)",
    "right": "turn right (direction + 1 mod 4)",
    "forward": "move one cell in the facing direction if that cell is empty",
    "pickup": "pick up the object in the facing cell if not carrying one",
    "drop": "drop the carried object in the facing cell if it is empty",
    "toggle": "toggle (open or close) the object in the facing cell",
    "done": "no effect in these missions",
}
OLD_ACTIONS = (0, 1, 2)
NEW_ACTIONS = (0, 1, 2, 3)
PICKUP = 3

VOCAB = ("go", "to", "pick", "up", "the", "a", "red", "green", "blue", "purple", "yellow", "grey", "ball", "box", "key")
WORD_ID = {word: index for index, word in enumerate(VOCAB)}
RESERVED_WORDS = ("pick", "up")
RESERVED_IDS = tuple(WORD_ID[word] for word in RESERVED_WORDS)
OLD_TYPES = ("ball", "box", "key")
ARTICLES = ("the", "a")
"""Instruction: [verb, verb, article, color, type]. "the" when one object in the room matches color and type, "a"
when several do (``minigrid ObjDesc.surface``). GoToLocal always names both color and type."""


@dataclass(frozen=True)
class SeedRange:
    """One episode set: ``n`` consecutive environment seeds from ``start`` under one mission."""

    name: str
    mission: str
    start: int
    n: int

    @property
    def seeds(self) -> np.ndarray:
        return np.arange(self.start, self.start + self.n, dtype=np.int64)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "mission": self.mission, "start": self.start, "stop": self.start + self.n,
                "n": self.n}


SEED_RANGES: dict[str, SeedRange] = {
    "train": SeedRange("train", "old", 0, 100_000),
    "V_check_old": SeedRange("V_check_old", "old", 1_000_000, 500),
    "V_select_old": SeedRange("V_select_old", "old", 1_100_000, 1_000),
    "V_evidence_new": SeedRange("V_evidence_new", "new", 2_000_000, 512),
    "V_check_new": SeedRange("V_check_new", "new", 2_100_000, 500),
    "V_select_new": SeedRange("V_select_new", "new", 2_200_000, 1_000),
    "smoke_old": SeedRange("smoke_old", "old", 9_000_000, 500),
    "smoke_new": SeedRange("smoke_new", "new", 9_100_000, 500),
}
"""Dev episode sets. The base trains on the first 50,000 train seeds (100,000 in the fallback). The teacher's 8 new
demonstrations are the first 8 usable bot episodes of V_evidence_new in seed order, and the label-equivalent amounts
(16 to 256) the first k, so the sets are nested. smoke_old and smoke_new serve timing and determinism smokes and tests
only; no result reads them."""
TRAIN_SIZES = (50_000, 100_000)
SEALED_FLOOR = 1_000_000_000
SEALED_CEILING = 2**31
SEALED_COUNTS = {"old": 1_000, "new": 1_000}


def check_ranges() -> None:
    """Every dev range is disjoint from the others and below ``SEALED_FLOOR``."""
    spans = sorted((r.start, r.start + r.n, r.name) for r in SEED_RANGES.values())
    for (a0, a1, a), (b0, _, b) in zip(spans, spans[1:]):
        if b0 < a1:
            raise ValueError(f"seed ranges {a} and {b} overlap")
    if spans[-1][1] > SEALED_FLOOR:
        raise ValueError("a dev range reaches the sealed floor")


check_ranges()


class E14Error(RuntimeError):
    """A row, mission, demonstration or rollout does not follow the E14 encoding."""


# ---------------------------------------------------------------------------------------------------------------------
# minigrid, imported quietly


@contextlib.contextmanager
def quiet() -> Any:
    """Capture stdout (minigrid's rejection-sampling prints, pygame's banner); stderr is untouched."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@functools.cache
def _minigrid() -> dict[str, Any]:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    with quiet():
        import gymnasium
        import minigrid
        from minigrid.core import constants
        from minigrid.envs.babyai.core.verifier import ObjDesc, PickupInstr
        from minigrid.envs.babyai.goto import GoToLocal
        from minigrid.utils.baby_ai_bot import BabyAIBot
        from minigrid.wrappers import FullyObsWrapper

    class GoToLocalPickup(GoToLocal):
        """GoToLocal's room generator with "pick up the {color} {type}" (PREREG E14's new mission)."""

        def gen_mission(self) -> None:
            self.place_agent()
            objs = self.add_distractors(num_distractors=self.num_dists, all_unique=False)
            self.check_objs_reachable()
            obj = self._rand_elem(objs)
            self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))

    versions = {"minigrid": minigrid.__version__, "gymnasium": gymnasium.__version__}
    for name, version in versions.items():
        if version != PINNED[name]:
            raise E14Error(f"{name} {version} is installed; E14 pins {PINNED[name]}")
    if tuple(constants.OBJECT_TO_IDX) != OBJECTS or tuple(constants.COLOR_TO_IDX) != COLORS:
        raise E14Error("minigrid's object or color indices differ from the E14 encoding")
    if list(constants.OBJECT_TO_IDX.values()) != list(range(len(OBJECTS))):
        raise E14Error("minigrid's object indices are not 0..10 in order")
    return {"GoToLocal": GoToLocal, "GoToLocalPickup": GoToLocalPickup, "BabyAIBot": BabyAIBot,
            "FullyObsWrapper": FullyObsWrapper, "versions": versions}


def environment_versions() -> dict[str, str]:
    """The installed minigrid and gymnasium versions (checked against ``PINNED``)."""
    return dict(_minigrid()["versions"])


def make_env(mission: str) -> Any:
    """A FullyObsWrapper around a fresh level of ``mission`` ("old": GoToLocal, "new": GoToLocalPickup)."""
    if mission not in MISSIONS:
        raise E14Error(f"unknown mission {mission!r}")
    mg = _minigrid()
    level = mg["GoToLocal"] if mission == "old" else mg["GoToLocalPickup"]
    with quiet():
        return mg["FullyObsWrapper"](level())


def reset(env: Any, seed: int) -> dict[str, Any]:
    """``env.reset(seed=seed)`` with stdout captured; returns the observation."""
    with quiet():
        obs, _ = env.reset(seed=int(seed))
    return obs


# ---------------------------------------------------------------------------------------------------------------------
# Encoding


def encode_mission(text: str) -> np.ndarray:
    """The 5 word ids of an instruction; refuses any other length or word."""
    words = str(text).split()
    if len(words) != INSTR_LEN or any(word not in WORD_ID for word in words):
        raise E14Error(f"instruction {text!r} is not 5 words of the E14 vocabulary")
    return np.asarray([WORD_ID[word] for word in words], dtype=np.int64)


def decode_mission(ids: Sequence[int]) -> str:
    return " ".join(VOCAB[int(i)] for i in ids)


def obs_row(obs: Mapping[str, Any]) -> np.ndarray:
    """One int64 [ROW_LEN] row: the full grid [x, y, channel] in C order, then the instruction's word ids."""
    image = np.asarray(obs["image"])
    if image.shape != (GRID, GRID, N_CHANNELS):
        raise E14Error(f"expected a {GRID} x {GRID} x {N_CHANNELS} grid, got {image.shape}")
    return np.concatenate([image.astype(np.int64).reshape(-1), encode_mission(obs["mission"])])


def split_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(grid int64 [N, 8, 8, 3], instruction int64 [N, 5]) of rows [N, ROW_LEN]."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != ROW_LEN:
        raise E14Error(f"expected rows [N, {ROW_LEN}], got {x.shape}")
    return x[:, :OBS_LEN].reshape(-1, GRID, GRID, N_CHANNELS), x[:, OBS_LEN:]


def check_rows(x: np.ndarray, verbs: tuple[str, str] | None = None) -> None:
    """Rows of the encoding: index ranges per channel, one agent cell, instruction words in place; ``verbs`` set:
    every row's first two words are those."""
    grid, instr = split_rows(x)
    if x.dtype != np.int64:
        raise E14Error(f"rows must be int64, got {x.dtype}")
    if grid[..., 0].min() < 0 or grid[..., 0].max() >= len(OBJECTS) or grid[..., 1].min() < 0 or grid[
            ..., 1].max() >= len(COLORS) or grid[..., 2].min() < 0 or grid[..., 2].max() >= N_STATES:
        raise E14Error("a grid value is outside the object, color or state indices")
    if not np.all((grid[..., 0] == OBJECTS.index("agent")).sum(axis=(1, 2)) == 1):
        raise E14Error("a row does not have exactly one agent cell")
    if instr.min() < 0 or instr.max() >= len(VOCAB):
        raise E14Error("an instruction id is outside the vocabulary")
    if not (np.isin(instr[:, 2], [WORD_ID[a] for a in ARTICLES]).all()
            and np.isin(instr[:, 3], [WORD_ID[c] for c in COLORS]).all()
            and np.isin(instr[:, 4], [WORD_ID[t] for t in OLD_TYPES]).all()):
        raise E14Error("an instruction is not [verb, verb, article, color, type]")
    if verbs is not None and not (np.all(instr[:, 0] == WORD_ID[verbs[0]]) and np.all(instr[:, 1] == WORD_ID[verbs[1]])):
        raise E14Error(f"an instruction does not start with {' '.join(verbs)!r}")


def mission_verbs(mission: str) -> tuple[str, str]:
    return ("go", "to") if mission == "old" else ("pick", "up")


# ---------------------------------------------------------------------------------------------------------------------
# Demonstrations


@dataclass
class Episode:
    """One bot episode: the row seen before each action, the action, and the outcome."""

    seed: int
    mission: str
    rows: np.ndarray
    actions: np.ndarray
    success: bool

    @property
    def n_steps(self) -> int:
        return int(len(self.actions))


def bot_episode(mission: str, seed: int, env: Any | None = None) -> Episode:
    """One full-map BabyAI bot episode of ``mission`` from environment seed ``seed`` (module docstring)."""
    env = make_env(mission) if env is None else env
    obs = reset(env, seed)
    bot = _minigrid()["BabyAIBot"](env)
    bot.vis_mask[:] = True
    rows, actions = [], []
    success = False
    for _ in range(MAX_STEPS):
        with quiet():
            action = int(bot.replan())
        rows.append(obs_row(obs))
        actions.append(action)
        with quiet():
            obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            success = bool(terminated and reward > 0)
            break
    return Episode(int(seed), mission, np.stack(rows).astype(np.int64), np.asarray(actions, dtype=np.int64), success)


def usable(episode: Episode) -> bool:
    """A demonstration that succeeds and uses only its mission's actions (old: left, right, forward; new: and
    pickup)."""
    allowed = OLD_ACTIONS if episode.mission == "old" else NEW_ACTIONS
    return episode.success and bool(np.isin(episode.actions, allowed).all())


DEMO_KEYS = ("x", "y", "episode", "step", "ep_seed", "ep_start", "ep_len", "ep_success", "ep_usable")


def demo_arrays(episodes: Iterable[Episode]) -> dict[str, np.ndarray]:
    """Demonstration arrays: every step of every usable episode (x uint8 rows, y actions, episode index into the
    ep_* arrays, step), and every episode's seed, first row, length, success and usability (unusable episodes have
    no rows: ep_start -1)."""
    xs, ys, eps, steps = [], [], [], []
    ep_seed, ep_start, ep_len, ep_success, ep_usable = [], [], [], [], []
    n = 0
    for index, episode in enumerate(episodes):
        ok = usable(episode)
        ep_seed.append(episode.seed)
        ep_len.append(episode.n_steps)
        ep_success.append(int(episode.success))
        ep_usable.append(int(ok))
        ep_start.append(n if ok else -1)
        if ok:
            xs.append(episode.rows)
            ys.append(episode.actions)
            eps.append(np.full(episode.n_steps, index, dtype=np.int64))
            steps.append(np.arange(episode.n_steps, dtype=np.int64))
            n += episode.n_steps
    x = np.concatenate(xs) if xs else np.zeros((0, ROW_LEN), dtype=np.int64)
    if x.size and (x.min() < 0 or x.max() > 255):
        raise E14Error("a row value does not fit uint8")
    return {
        "x": x.astype(np.uint8), "y": np.concatenate(ys) if ys else np.zeros(0, np.int64),
        "episode": np.concatenate(eps) if eps else np.zeros(0, np.int64),
        "step": np.concatenate(steps) if steps else np.zeros(0, np.int64),
        "ep_seed": np.asarray(ep_seed, dtype=np.int64), "ep_start": np.asarray(ep_start, dtype=np.int64),
        "ep_len": np.asarray(ep_len, dtype=np.int64), "ep_success": np.asarray(ep_success, dtype=np.int64),
        "ep_usable": np.asarray(ep_usable, dtype=np.int64),
    }


def demo_chunk(args: tuple[str, int, int]) -> dict[str, np.ndarray]:
    """``demo_arrays`` of seeds [start, stop) of one mission (a worker's share; picklable)."""
    mission, start, stop = args
    env = make_env(mission)
    return demo_arrays(bot_episode(mission, seed, env) for seed in range(start, stop))


def merge_demo_chunks(chunks: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate ``demo_chunk`` outputs in order, re-basing episode indices and first-row offsets."""
    out: dict[str, list[np.ndarray]] = {key: [] for key in DEMO_KEYS}
    n_ep = n_rows = 0
    for chunk in chunks:
        out["x"].append(chunk["x"])
        out["y"].append(chunk["y"])
        out["episode"].append(chunk["episode"] + n_ep)
        out["step"].append(chunk["step"])
        out["ep_seed"].append(chunk["ep_seed"])
        out["ep_start"].append(np.where(chunk["ep_start"] >= 0, chunk["ep_start"] + n_rows, -1))
        for key in ("ep_len", "ep_success", "ep_usable"):
            out[key].append(chunk[key])
        n_ep += len(chunk["ep_seed"])
        n_rows += len(chunk["y"])
    return {key: np.concatenate(values) for key, values in out.items()}


def usable_episodes(demos: Mapping[str, np.ndarray], k: int) -> np.ndarray:
    """Episode indices of the first ``k`` usable episodes (in seed order) of a demonstration set."""
    usable_eps = np.flatnonzero(demos["ep_usable"] == 1)
    if len(usable_eps) < k:
        raise E14Error(f"only {len(usable_eps)} usable episodes, {k} asked")
    return usable_eps[:k]


def first_usable(demos: Mapping[str, np.ndarray], k: int) -> np.ndarray:
    """Row indices of the first ``k`` usable episodes (in seed order) of a demonstration set."""
    return np.flatnonzero(np.isin(demos["episode"], usable_episodes(demos, k)))


def demo_summary(demos: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Counts of a demonstration set: episodes, usable, excluded, steps, action counts, mean length."""
    n_ep = int(len(demos["ep_seed"]))
    usable_n = int(demos["ep_usable"].sum())
    actions = {ACTIONS[a]: int(np.count_nonzero(demos["y"] == a)) for a in range(len(ACTIONS))}
    return {"n_episodes": n_ep, "n_usable": usable_n, "n_excluded": n_ep - usable_n,
            "n_failed": int(n_ep - demos["ep_success"].sum()), "n_steps": int(len(demos["y"])),
            "actions": {k: v for k, v in actions.items() if v},
            "mean_len_usable": float(demos["ep_len"][demos["ep_usable"] == 1].mean()) if usable_n else None,
            "seed_first": int(demos["ep_seed"][0]) if n_ep else None,
            "seed_last": int(demos["ep_seed"][-1]) if n_ep else None}


# ---------------------------------------------------------------------------------------------------------------------
# Rollouts

LogitsFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class RolloutResult:
    """Per-episode outcomes of one greedy rollout call, in the order of ``seeds``."""

    mission: str
    seeds: np.ndarray
    success: np.ndarray
    length: np.ndarray
    article: np.ndarray
    obj_type: np.ndarray
    obj_color: np.ndarray
    n_pickups: np.ndarray

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {"seeds": self.seeds, "success": self.success, "length": self.length, "article": self.article,
                "obj_type": self.obj_type, "obj_color": self.obj_color,
                "n_pickups": self.n_pickups}


def rollouts(logits_fn: LogitsFn, mission: str, seeds: Sequence[int] | np.ndarray) -> RolloutResult:
    """Greedy episodes of ``mission`` from each seed: ``logits_fn(rows [B, ROW_LEN] int64) -> [B, 7]`` once per step
    on the running episodes; argmax action; an episode ends on the verifier's success or at max_steps."""
    seeds = np.asarray(seeds, dtype=np.int64)
    n = len(seeds)
    envs = [make_env(mission) for _ in range(n)]
    obs = [reset(env, int(seed)) for env, seed in zip(envs, seeds)]
    verbs = mission_verbs(mission)
    article = np.zeros(n, dtype=np.int64)
    obj_type = np.zeros(n, dtype=np.int64)
    obj_color = np.zeros(n, dtype=np.int64)
    for i, ob in enumerate(obs):
        ids = encode_mission(ob["mission"])
        if decode_mission(ids[:2]) != " ".join(verbs):
            raise E14Error(f"seed {seeds[i]}: mission {ob['mission']!r} is not a {mission} mission")
        article[i] = ARTICLES.index(VOCAB[ids[2]])
        obj_type[i] = OLD_TYPES.index(VOCAB[ids[4]])
        obj_color[i] = COLORS.index(VOCAB[ids[3]])
    success = np.zeros(n, dtype=bool)
    length = np.zeros(n, dtype=np.int64)
    n_pickups = np.zeros(n, dtype=np.int64)
    running = list(range(n))
    for _ in range(MAX_STEPS):
        if not running:
            break
        x = np.stack([obs_row(obs[i]) for i in running])
        logits = np.asarray(logits_fn(x))
        if logits.shape != (len(running), len(ACTIONS)):
            raise E14Error(f"the policy returned logits of shape {logits.shape}")
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("the policy produced non-finite logits")
        actions = np.argmax(logits, axis=1)
        still = []
        for i, action in zip(running, actions.tolist()):
            with quiet():
                obs[i], reward, terminated, truncated, _ = envs[i].step(int(action))
            length[i] += 1
            n_pickups[i] += int(action == PICKUP)
            if terminated or truncated:
                success[i] = bool(terminated and reward > 0)
            else:
                still.append(i)
        running = still
    if running:
        raise E14Error("an episode outlived max_steps")
    return RolloutResult(mission, seeds, success, length, article, obj_type, obj_color, n_pickups)


def initial_fingerprint(mission: str, seeds: Sequence[int] | np.ndarray) -> str:
    """sha256 of the first rows of the episodes of ``seeds`` (the room, objects, agent and instruction each seed
    makes): equal on two hosts means the same episodes."""
    import hashlib

    env = make_env(mission)
    digest = hashlib.sha256()
    for seed in np.asarray(seeds, dtype=np.int64).tolist():
        digest.update(obs_row(reset(env, seed)).astype("<i8").tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# The sealed record


def draw_sealed_seeds() -> dict[str, np.ndarray]:
    """1,000 old and 1,000 new distinct seeds from the operating system's generator in [SEALED_FLOOR, 2**31)."""
    drawn: set[int] = set()
    out = {}
    for mission in MISSIONS:
        seeds = []
        while len(seeds) < SEALED_COUNTS[mission]:
            value = SEALED_FLOOR + secrets.randbelow(SEALED_CEILING - SEALED_FLOOR)
            if value not in drawn:
                drawn.add(value)
                seeds.append(value)
        out[mission] = np.asarray(seeds, dtype=np.int64)
    return out


def sealed_record_arrays(seeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The store form of the sealed record: x the seeds (old then new), y the mission code, idx the position."""
    x = np.concatenate([np.asarray(seeds[m], dtype=np.int64) for m in MISSIONS])
    y = np.concatenate([np.full(len(seeds[m]), MISSION_CODES[m], dtype=np.int64) for m in MISSIONS])
    return {"x": x, "y": y, "idx": np.arange(len(x), dtype=np.int64)}


def write_sealed_record() -> dict[str, Any]:
    """Draw the sealed seeds and write them once to ``$WEIGHTCODE_SEALED/<DATASET>/test.npz`` (refuses an existing file);
    returns the record's canonical sha256 and counts, never the seeds."""
    from weightcode.data.store import save_split, sealed_test_path

    path = sealed_test_path(DATASET)
    if path.exists():
        raise FileExistsError(f"{path} exists; the sealed record is written once")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    arrays = sealed_record_arrays(draw_sealed_seeds())
    digest = save_split(path, arrays, private_mode=True)
    return {"path": f"$WEIGHTCODE_SEALED/{DATASET}/test.npz", "sha256": digest,
            "n_old": SEALED_COUNTS["old"], "n_new": SEALED_COUNTS["new"],
            "rule": f"distinct seeds from secrets.randbelow in [{SEALED_FLOOR}, 2**31), old then new"}


# ---------------------------------------------------------------------------------------------------------------------
# Store files

DEMO_FILES = {"train": "train_old.npz", "V_evidence_new": "vevidence_new.npz"}
DEMOS_MANIFEST = "demos.json"


def store_dir() -> Any:
    """``$WEIGHTCODE_PRIVATE/e14_gotolocal``."""
    from weightcode.data.store import dataset_dir

    return dataset_dir(DATASET)


def demos_dir() -> Any:
    return store_dir() / "demos"


def demos_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    from weightcode.common.identity import params_sha256

    return params_sha256(arrays)


def save_demos(name: str, arrays: Mapping[str, np.ndarray]) -> str:
    """Write one demonstration set to ``demos/<file>`` (refuses an existing file); returns its canonical sha256."""
    path = demos_dir() / DEMO_FILES[name]
    if path.exists():
        raise FileExistsError(f"{path} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    if sorted(arrays) != sorted(DEMO_KEYS):
        raise E14Error(f"demonstration keys {sorted(arrays)} are not {sorted(DEMO_KEYS)}")
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
    digest = demos_sha256(arrays)
    if demos_sha256(load_demo_file(temporary)) != digest:
        temporary.unlink()
        raise RuntimeError(f"reload hash mismatch for {path}")
    temporary.replace(path)
    return digest


def load_demo_file(path: Any) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.ascontiguousarray(archive[key]) for key in archive.files}


def load_demos(name: str, expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    """One demonstration set from the store, hash-checked against ``expected_sha256`` or the demos manifest."""
    import json

    arrays = load_demo_file(demos_dir() / DEMO_FILES[name])
    if expected_sha256 is None:
        manifest = json.loads((demos_dir() / DEMOS_MANIFEST).read_text(encoding="utf-8"))
        expected_sha256 = manifest["sets"][name]["sha256"]
    if demos_sha256(arrays) != expected_sha256:
        raise E14Error(f"demonstration set {name} does not hash to {expected_sha256}")
    return arrays


def train_rows(demos: Mapping[str, np.ndarray], n_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    """(rows int64, actions) of every step of the usable episodes among the first ``n_episodes`` train seeds."""
    seeds = demos["ep_seed"]
    if n_episodes not in TRAIN_SIZES or len(seeds) < n_episodes or not np.array_equal(
            seeds[:n_episodes], SEED_RANGES["train"].seeds[:n_episodes]):
        raise E14Error(f"the train demonstrations do not hold the first {n_episodes} train seeds")
    keep = demos["episode"] < n_episodes
    return demos["x"][keep].astype(np.int64), demos["y"][keep].astype(np.int64)


def ranges_record() -> dict[str, Any]:
    """The dev seed ranges and the sealed rule, as dataset.json records them."""
    return {"dev": {name: r.as_dict() for name, r in SEED_RANGES.items()}, "train_sizes": list(TRAIN_SIZES),
            "sealed": {"n_old": SEALED_COUNTS["old"], "n_new": SEALED_COUNTS["new"], "floor": SEALED_FLOOR,
                       "ceiling": SEALED_CEILING}}


def encoding_record() -> dict[str, Any]:
    """The observation, instruction and action encoding, as model_spec.json and dataset.json state it."""
    return {
        "row": f"int64 [{ROW_LEN}]: the grid [x, y, channel] in C order ({OBS_LEN} values), then {INSTR_LEN} word ids",
        "grid": {"width": GRID, "height": GRID, "channels": ["object", "color", "state"],
                 "objects": list(OBJECTS), "colors": list(COLORS),
                 "state": "door: 0 open, 1 closed, 2 locked; agent cell: direction 0 east (+x), 1 south (+y), "
                          "2 west (-x), 3 north (-y); otherwise 0"},
        "instruction": {"length": INSTR_LEN, "layout": "[verb, verb, article, color, type]", "vocab": list(VOCAB),
                        "reserved": list(RESERVED_WORDS)},
        "actions": list(ACTIONS),
        "max_steps": MAX_STEPS,
    }


__all__ = [
    "ACTIONS", "ACTION_MEANINGS", "ARTICLES", "COLORS", "DATASET", "DEMO_KEYS", "DIRECTIONS", "E14Error", "Episode",
    "GRID", "INSTR_LEN", "LEAK_WORDS", "MAX_STEPS", "MISSIONS", "MISSION_CODES", "NEW_ACTIONS", "NEW_LEVEL",
    "N_CHANNELS", "N_STATES", "OBJECTS", "OBS_LEN", "OLD_ACTIONS", "OLD_LEVEL", "OLD_TYPES", "PICKUP", "PINNED",
    "RESERVED_IDS", "RESERVED_WORDS", "ROW_LEN", "RolloutResult", "SEALED_COUNTS", "SEALED_FLOOR", "SEED_RANGES",
    "SeedRange", "TRAIN_SIZES", "VOCAB", "WHEELS", "WORD_ID", "bot_episode", "check_rows", "decode_mission",
    "demo_arrays", "demo_chunk", "demo_summary", "encode_mission", "encoding_record", "environment_versions",
    "first_usable", "initial_fingerprint", "make_env", "merge_demo_chunks", "mission_verbs", "obs_row", "quiet",
    "usable_episodes",
    "ranges_record", "reset", "rollouts", "sealed_record_arrays", "split_rows", "usable", "write_sealed_record",
]
