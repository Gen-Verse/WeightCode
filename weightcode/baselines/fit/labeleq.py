"""PREREG F2, the label-equivalent amount (user 2026-11-05 about 11:35 Beijing, main b67036f).

Full fine-tuning of the seed-100 E11 student (ViT-Lite-7/4 mae049764, dataset cifar10_vl7) on N labelled corrupted
CIFAR-10 train images mixed 1:1 with N clean train images, for N in ``F2_NS``. The question: how many labelled
condition images does gradient fine-tuning need to reach the teacher's V_select joint at clean >= -1.0 (N*)?

Data (``build_f2_data``). The pool is E1's 40,000 cifar10 train rows (``cifar10_vl7/splits/train.npz``), the rows the
student trained on; they are disjoint from V_evidence, V_check and V_select (checked by row id). PREREG's top grid
point is 50,000, the size of the official train set, but E1 holds out 10,000 train rows as the dev splits, so the
pool has 40,000 rows and the top grid point here is 40,000 (``F2_N_SPEC_TOP`` records the PREREG value). The
corrupted rows come from the pinned official CIFAR-10-C code at severity 5 in E10's fidelity-checked environment
(``scripts/f2/f2_corrupt_train.py``, one seed per condition, all 40,000 rows in row order). Subsets are nested: two
permutations of the pool from ``numpy.random.default_rng(F2_SUBSET_SEED)``; for each N the corrupted rows are the
first N of ``perm_corrupt`` and the clean replay rows the first N of ``perm_replay`` (an independent draw, so the
replay is not paired with the corrupted images, as the experience's replay rows are not). The same subsets serve
both conditions. The permutations are stored (``subsets.npz``) and shipped, not regenerated on the cluster.

Training (``train_trajectory``; NSCC GPU, ``scripts/train_f2.py``). The student's own recipe (``weightcode.models.vitlite``,
the examples/main.py recipe of PREREG E11) with these parts kept for fine-tuning, identical for every N and lr:
AdamW (betas 0.9/0.999, eps 1e-8, weight decay 3e-2 on every parameter), label-smoothing cross-entropy 0.1, the
model in train mode (dropout 0.1, attention dropout 0, stochastic depth 0.1), tanh GELU, fp32 with IEEE matmuls and
deterministic algorithms, no clipping, no EMA, no mixup, and the training transform on every row (CIFAR-10
AutoAugment, random crop 32 with padding 4, horizontal flip, normalisation), applied after the offline corruption.
Not kept: the from-scratch initialisation (training starts at the student's weights) and the 200-epoch warmup and
cosine schedule; the learning rate is the grid's, constant, with no warmup, as in the references' and F1's
fine-tunes. Batch 128, each batch 64 corrupted rows and 64 clean rows (``BalancedBatchSampler``: each half walks
its own set in passes, a fresh permutation per pass; every N is a multiple of 64). One trajectory per (condition, N,
lr), checkpointed at 100 and 1000 steps; at the constant learning rate the 100-step checkpoint is the 100-step run
(the timing smoke checks this bit for bit). The N = 40,000 trajectories continue to 5000 steps: the ceiling, one
candidate per lr. Run seed 0 (``configure_determinism``; the loader's shuffling and its workers' augmentation draws
come from generators seeded with it).

Checks and readouts (the 4090 CPU; ``run_f2``). For each (condition, N) the 8 configurations (``f2_configs``: lr
outer, steps inner) are one baseline run of method ``ft_labeleq`` (run label ``f2_n<N>``; the ceiling ``f2_ceiling``,
4 candidates), checked once each by the joint oracle exactly as the references (S_joint = S_new + S_old on the
condition's V_check and its clean companion), held to no edit limit (every weight changes: "edit limits ignored",
as grad_full and interp), and read with ``weightcode.evalkit.devreadout`` (``--retention-floor -20``: joint-retain-v1's
pick; ``--all-candidates``: the descriptive frontier). The instance's experience rows are not used for fitting: the
fit rows of F2 are the N + N train images, and every record says so.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weightcode.common.identity import load_params, params_sha256, save_params

PREREG_F2 = "PREREG F2 (fine-tuning comparisons, user 2026-11-05 about 11:35 Beijing, main b67036f)"
F2_DATASET = "cifar10_vl7"
F2_STUDENT = "mae049764"
F2_STUDENT_SHA256 = "fd7f6d4b80145c98bc7c6a9cbc2c8e440e3f45e328168df2a8b55bec3f7c5b6a"
F2_CONDITIONS: tuple[str, ...] = ("gaussian_noise", "contrast")
F2_POOL_ROWS = 40000
F2_NS: tuple[int, ...] = (64, 256, 1024, 4096, 16384, 40000)
F2_N_SPEC_TOP = 50000
"""PREREG's top grid point; the train pool has 40,000 rows (module docstring), so the grid stops at 40,000."""
F2_LRS: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4)
F2_STEPS: tuple[int, ...] = (100, 1000)
F2_CEILING_STEPS = 5000
F2_BATCH = 128
F2_HALF = F2_BATCH // 2
F2_SUBSET_SEED = 20261105
F2_RUN_SEED = 0
F2_CROP_PAD = 4
F2_WEIGHT_DECAY = 3e-2
F2_BETAS: tuple[float, float] = (0.9, 0.999)
F2_EPS = 1e-8
F2_SMOOTHING = 0.1
F2_CHECK_ROWS = 64
"""Device check: eval-mode logits on the first 64 clean train rows, compared with forward.py at import."""
F2_DEVICE_MAX_ABS = 1e-3
F2_METHOD = "ft_labeleq"
F2_DIR = "f2_train"
F2_CORRUPT_RUN = "corrupt_20261105"
F2_DATA_JSON = "f2_data.json"
F2_SUBSETS = "subsets.npz"
F2_IMPORT_JSON = "f2_import.json"
CEILING = "ceiling"
F2_TEACHER_JOINT: dict[str, float | None] = {"gaussian_noise_s5": 31.75, "contrast_s5": 17.975}
"""The teacher's V_select joint per condition (PREREG F2), from the E11 B sessions' final V_select readouts
(runs/devreadouts/learn/<session>/info.json): gaussian_noise_s5 +31.75 = 31.625 + 0.125 (session sc918e11dceef637a,
instance i00802425); contrast_s5 +17.975 = 17.925 + 0.05 (session s760cb52ce82fb937, instance i3eb3fc78; the
coordinator's +17.97 and +17.98 are its two roundings)."""
F2_CLEAN_FLOOR_POINTS = -1.0

RECIPE_KEPT = (
    "from the student's recipe (weightcode.models.vitlite RECIPE, PREREG E11): AdamW betas (0.9, 0.999), eps 1e-8, weight "
    "decay 3e-2 on every parameter; label-smoothing cross-entropy 0.1; train mode with dropout 0.1, attention dropout "
    "0, stochastic depth 0.1; tanh GELU; fp32 with IEEE matmuls and deterministic algorithms; no clipping, EMA or "
    "mixup; the training transform (CIFAR-10 AutoAugment, random crop 32 with padding 4, horizontal flip, "
    "normalisation) on every row, after the offline corruption. Not kept: the from-scratch initialisation and the "
    "200-epoch warmup and cosine schedule (constant grid learning rate, no warmup). Batch 128 = 64 corrupted + 64 "
    "clean rows."
)
LIMITS_NOTE = ("edit limits ignored (lim-ignored): full fine-tuning changes every weight, like grad_full and interp; "
               "each check records whether it falls within the E10 program limits")


class F2Error(ValueError):
    """An F2 input, configuration or record does not match what PREREG F2 fixes."""


# --------------------------------------------------------------------------------------------- grid


def f2_configs() -> list[dict[str, Any]]:
    """The 8 configurations of one N in check order: lr outer, steps inner."""
    return [{"lr": lr, "steps": steps} for lr in F2_LRS for steps in F2_STEPS]


def ceiling_configs() -> list[dict[str, Any]]:
    """The ceiling's candidates in check order: N = 40,000 at 5000 steps, one per lr."""
    return [{"lr": lr, "steps": F2_CEILING_STEPS} for lr in F2_LRS]


def groups() -> list[str]:
    """Check groups of one condition: ``n<N>`` per grid point, then ``ceiling``."""
    return [f"n{n}" for n in F2_NS] + [CEILING]


def group_rows(group: str) -> tuple[int, list[dict[str, Any]]]:
    """(N, configurations in check order) of one check group."""
    if group == CEILING:
        return F2_NS[-1], ceiling_configs()
    if not group.startswith("n") or not group[1:].isdigit() or int(group[1:]) not in F2_NS:
        raise F2Error(f"unknown F2 group {group!r}; expected one of {groups()}")
    return int(group[1:]), f2_configs()


@dataclass(frozen=True)
class Trajectory:
    """One fine-tuning trajectory: a (condition, N, lr) at the constant lr, checkpointed at ``checkpoints``."""

    condition: str
    n: int
    lr: float
    checkpoints: tuple[int, ...]
    tag: str = ""

    def __post_init__(self) -> None:
        if self.condition not in F2_CONDITIONS:
            raise F2Error(f"unknown condition {self.condition!r}")
        if self.n < F2_HALF or self.n % F2_HALF or self.n > F2_POOL_ROWS:
            raise F2Error(f"N = {self.n} must be a multiple of {F2_HALF} in [{F2_HALF}, {F2_POOL_ROWS}]")
        if not self.checkpoints or list(self.checkpoints) != sorted(set(self.checkpoints)) or self.checkpoints[0] < 1:
            raise F2Error(f"checkpoints must be increasing positive steps, got {self.checkpoints}")

    @property
    def name(self) -> str:
        return f"{self.condition}_n{self.n}_lr{self.lr:g}" + (f"_{self.tag}" if self.tag else "")

    @property
    def steps(self) -> int:
        return self.checkpoints[-1]

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "checkpoints": list(self.checkpoints), "name": self.name}


def f2_trajectories(conditions: Sequence[str] = F2_CONDITIONS) -> list[Trajectory]:
    """Every trajectory of the main job, longest first (makespan), then conditions in order, then larger N first."""
    out = []
    for condition in conditions:
        for n in F2_NS:
            for lr in F2_LRS:
                extra = (F2_CEILING_STEPS,) if n == F2_NS[-1] else ()
                out.append(Trajectory(condition, n, lr, F2_STEPS + extra))
    order = {c: i for i, c in enumerate(conditions)}
    return sorted(out, key=lambda t: (-t.steps, order[t.condition], -t.n, t.lr))


def smoke_trajectories() -> list[Trajectory]:
    """The timing smoke: six concurrent children of 100 or 200 steps; a duplicate (determinism) and a 100-step run
    beside a 200-step one (the 100-step checkpoint equals the 100-step run)."""
    return [
        Trajectory("gaussian_noise", 1024, 1e-4, (100, 200)),
        Trajectory("gaussian_noise", 1024, 1e-4, (100, 200), tag="dup"),
        Trajectory("gaussian_noise", 1024, 1e-4, (100,), tag="short"),
        Trajectory("contrast", 40000, 3e-4, (100, 200)),
        Trajectory("contrast", 64, 1e-5, (100, 200)),
        Trajectory("gaussian_noise", 16384, 3e-5, (100, 200)),
    ]


def checkpoint_dir(trajectory: Trajectory, steps: int) -> str:
    """Relative directory of one checkpoint under a models root."""
    return f"{trajectory.condition}/n{trajectory.n}/lr{trajectory.lr:g}{'_' + trajectory.tag if trajectory.tag else ''}/s{steps}"


# --------------------------------------------------------------------------------------------- data


def f2_dir() -> Path:
    from weightcode.data.store import dataset_dir

    return dataset_dir(F2_DATASET) / F2_DIR


def _file_sha256(path: Path) -> str:
    from weightcode.evalkit.targets import file_sha256

    return file_sha256(path)


def build_f2_data(corrupt_dir: Path, run_info: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``subsets.npz`` and ``f2_data.json`` under ``$WEIGHTCODE_PRIVATE/cifar10_vl7/f2_train`` (refuses to overwrite).

    Checks: the train split against dataset.json; the pool's row ids disjoint from V_evidence, V_check and V_select
    (their ``idx`` arrays only); the corruption record's clean file equal to this train file and its outputs' sha256;
    every corrupted array uint8 [40000, 3, 32, 32].
    """
    from weightcode.data.store import load_split, read_dataset_json, split_path, split_sha256, write_json

    root = f2_dir()
    data_json, subsets_path = root / F2_DATA_JSON, root / F2_SUBSETS
    if data_json.exists() or subsets_path.exists():
        raise F2Error(f"{root} already holds F2 data; nothing written")
    manifest = read_dataset_json(F2_DATASET)
    train = load_split(F2_DATASET, "train")
    if split_sha256(train) != manifest["splits"]["train"]["sha256"] or len(train["x"]) != F2_POOL_ROWS:
        raise F2Error("cifar10_vl7/train does not match its dataset.json or has the wrong size")
    disjoint = {}
    for split in ("V_evidence", "V_check", "V_select"):
        with np.load(split_path(F2_DATASET, split), allow_pickle=False) as archive:
            other = np.asarray(archive["idx"])
        disjoint[split] = not bool(np.intersect1d(train["idx"], other).size)
        if not disjoint[split]:
            raise F2Error(f"train rows share row ids with {split}")
    record_path = Path(corrupt_dir) / "corrupt_train.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    train_file = split_path(F2_DATASET, "train")
    conditions = {}
    for condition in F2_CONDITIONS:
        row = record["rows"][condition]
        path = Path(corrupt_dir) / condition / "train_x.npz"
        if row["clean_file_sha256"] != _file_sha256(train_file) or row["out_file_sha256"] != _file_sha256(path):
            raise F2Error(f"{condition}: the corruption record does not match the files")
        if not record["reproduction"][condition]["equal"]:
            raise F2Error(f"{condition}: the reproduction check failed")
        with np.load(path, allow_pickle=False) as archive:
            x = np.asarray(archive["x"])
        if x.dtype != np.uint8 or x.shape != train["x"].shape:
            raise F2Error(f"{condition}: corrupted x is {x.dtype}{x.shape}")
        conditions[condition] = {"file": str(path.relative_to(root)), "file_sha256": row["out_file_sha256"],
                                 "x_sha256": row["x_sha256"], "seed": row["seed"],
                                 "median_mad_from_clean": row["median_mad_from_clean"]}
    rng = np.random.default_rng(F2_SUBSET_SEED)
    perm_corrupt = rng.permutation(F2_POOL_ROWS).astype(np.int64)
    perm_replay = rng.permutation(F2_POOL_ROWS).astype(np.int64)
    root.mkdir(parents=True, exist_ok=True)
    subsets_sha = save_params(subsets_path, {"perm_corrupt": perm_corrupt, "perm_replay": perm_replay})
    overlap = {n: int(np.intersect1d(perm_corrupt[:n], perm_replay[:n]).size) for n in F2_NS}
    payload = {
        "schema": "weightcode.f2.data.v1",
        "prereg": PREREG_F2,
        "dataset": F2_DATASET,
        "student": {"model_id": F2_STUDENT, "params_sha256": F2_STUDENT_SHA256},
        "pool": {"split": "train", "n": F2_POOL_ROWS, "split_sha256": manifest["splits"]["train"]["sha256"],
                 "file_sha256": _file_sha256(train_file), "disjoint_row_ids_from": disjoint,
                 "note": "E1's 40,000 train rows (the official 50,000 less E1's 10,000 dev rows)"},
        "corrupt": {"dir": str(Path(corrupt_dir).relative_to(root)), "record_sha256": _file_sha256(record_path),
                    "official": record["official"], "libraries": record["libraries"], "severity": record["severity"],
                    "reproduction": record["reproduction"], "conditions": conditions},
        "subsets": {"file": F2_SUBSETS, "params_sha256": subsets_sha, "file_sha256": _file_sha256(subsets_path),
                    "rule": ("perm_corrupt, perm_replay = two permutations of the 40,000 pool rows from "
                             f"numpy.random.default_rng({F2_SUBSET_SEED}) in that order; corrupted rows of N = "
                             "perm_corrupt[:N], clean replay rows of N = perm_replay[:N]; nested in N; the same for "
                             "both conditions"),
                    "seed": F2_SUBSET_SEED, "numpy": np.__version__, "corrupt_replay_shared_images": overlap},
        "ns": list(F2_NS), "n_prereg_top": F2_N_SPEC_TOP,
        "created_by": dict(run_info),
    }
    write_json(data_json, payload)
    return payload


@dataclass
class F2Data:
    """The arrays one trajectory needs, hash-checked against ``f2_data.json``."""

    x_clean: np.ndarray
    y: np.ndarray
    x_corrupt: dict[str, np.ndarray]
    perm_corrupt: np.ndarray
    perm_replay: np.ndarray
    mean: list[float]
    std: list[float]
    data_json_sha256: str
    base_params: dict[str, np.ndarray]


def load_f2_data(conditions: Sequence[str] = F2_CONDITIONS) -> F2Data:
    """Load and check every F2 input from ``$WEIGHTCODE_PRIVATE`` (the cluster job's private copy, or the workstation)."""
    from weightcode.data.store import dataset_dir, load_split, read_dataset_json, split_sha256
    from weightcode.models.e10_export import normalization

    root = f2_dir()
    payload = json.loads((root / F2_DATA_JSON).read_text(encoding="utf-8"))
    manifest = read_dataset_json(F2_DATASET)
    train = load_split(F2_DATASET, "train")
    if split_sha256(train) != payload["pool"]["split_sha256"]:
        raise F2Error("the train split differs from f2_data.json")
    subsets = load_params(root / F2_SUBSETS)
    if params_sha256(subsets) != payload["subsets"]["params_sha256"]:
        raise F2Error("subsets.npz differs from f2_data.json")
    x_corrupt = {}
    for condition in conditions:
        entry = payload["corrupt"]["conditions"][condition]
        path = root / entry["file"]
        if _file_sha256(path) != entry["file_sha256"]:
            raise F2Error(f"{path} differs from f2_data.json")
        with np.load(path, allow_pickle=False) as archive:
            x_corrupt[condition] = np.ascontiguousarray(archive["x"])
    base = load_params(dataset_dir(F2_DATASET) / "models" / F2_STUDENT / "model.npz")
    if params_sha256(base) != F2_STUDENT_SHA256:
        raise F2Error(f"the student's model.npz is not {F2_STUDENT_SHA256}")
    mean, std = normalization(manifest)
    return F2Data(x_clean=np.ascontiguousarray(train["x"]), y=train["y"].astype(np.int64), x_corrupt=x_corrupt,
                  perm_corrupt=subsets["perm_corrupt"], perm_replay=subsets["perm_replay"], mean=mean, std=std,
                  data_json_sha256=_file_sha256(root / F2_DATA_JSON), base_params=base)


# --------------------------------------------------------------------------------------------- training


class BalancedBatchSampler:
    """``steps`` batches of ``half`` rows from [0, n_a) and ``half`` rows from [n_a, n_a + n_b), in that order.

    Each half walks its own set in passes, a fresh ``torch.randperm`` from ``generator`` per pass (set a first when
    both start a pass at the same step). ``n_a`` and ``n_b`` must be multiples of ``half``, so no pass is cut short.
    """

    def __init__(self, n_a: int, n_b: int, half: int, steps: int, generator: Any) -> None:
        if n_a % half or n_b % half or n_a < half or n_b < half:
            raise F2Error(f"set sizes {n_a}, {n_b} must be positive multiples of {half}")
        self.n_a, self.n_b, self.half, self.steps, self.generator = n_a, n_b, half, steps, generator

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        import torch

        perm_a = perm_b = None
        pos_a, pos_b = self.n_a, self.n_b
        for _ in range(self.steps):
            if pos_a == self.n_a:
                perm_a, pos_a = torch.randperm(self.n_a, generator=self.generator).tolist(), 0
            if pos_b == self.n_b:
                perm_b, pos_b = torch.randperm(self.n_b, generator=self.generator).tolist(), 0
            assert perm_a is not None and perm_b is not None
            batch = perm_a[pos_a:pos_a + self.half] + [self.n_a + i for i in perm_b[pos_b:pos_b + self.half]]
            pos_a += self.half
            pos_b += self.half
            yield batch


def trajectory_rows(trajectory: Trajectory, data: F2Data) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The 2N training rows (N corrupted, then N clean) with labels, and their description."""
    n = trajectory.n
    idx_c = data.perm_corrupt[:n]
    idx_r = data.perm_replay[:n]
    x = np.ascontiguousarray(np.concatenate([data.x_corrupt[trajectory.condition][idx_c], data.x_clean[idx_r]]))
    y = np.concatenate([data.y[idx_c], data.y[idx_r]]).astype(np.int64)
    info = {"n_corrupt": n, "n_clean": n, "corrupt_rows": f"perm_corrupt[:{n}] of the pool, {trajectory.condition}_s5",
            "clean_rows": f"perm_replay[:{n}] of the pool, clean",
            "rows_sha256": params_sha256({"x": x, "y": y}),
            "corrupt_class_counts": np.bincount(data.y[idx_c], minlength=10).tolist(),
            "clean_class_counts": np.bincount(data.y[idx_r], minlength=10).tolist()}
    return x, y, info


def train_trajectory(
    trajectory: Trajectory,
    data: F2Data,
    *,
    device: Any,
    seed: int,
    num_workers: int,
    out_root: Path,
    provenance: Mapping[str, Any],
    log_fn: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Fine-tune the student along one trajectory and write each checkpoint; returns one index row per checkpoint.

    The caller sets determinism first (``configure_determinism(seed)``). Each checkpoint goes to
    ``out_root/<checkpoint_dir>/`` as model.npz (float32, the student's names and shapes) and record.json (the
    trajectory, its configuration, the training loss, the device logits on ``F2_CHECK_ROWS`` clean train rows, data and
    code provenance). A checkpoint's eval-mode logits draw no random number, so the trajectory continues unchanged.
    """
    import torch
    from torch.utils.data import DataLoader

    from weightcode.models import vitlite as V

    started = time.perf_counter()
    x, y, rows_info = trajectory_rows(trajectory, data)
    torch.manual_seed(seed)
    model = V.ViTLite(V.ViTLiteConfig(), data.mean, data.std)
    V.load_params_into(model, data.base_params)
    if params_sha256(V.export_params(model)) != F2_STUDENT_SHA256:
        raise F2Error("the model built from the student's weights does not hash to the student")
    model.to(device).train()
    dataset = V.AugmentedImages(x, y, V.train_transform(data.mean, data.std, F2_CROP_PAD))
    sampler = BalancedBatchSampler(trajectory.n, trajectory.n, F2_HALF, trajectory.steps,
                                   torch.Generator().manual_seed(seed))
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=trajectory.lr, betas=F2_BETAS, eps=F2_EPS,
                                  weight_decay=F2_WEIGHT_DECAY)
    check_rows = np.ascontiguousarray(data.x_clean[:F2_CHECK_ROWS])
    wanted = set(trajectory.checkpoints)
    losses: list[float] = []
    correct_corrupt: list[float] = []
    log: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    step = 0
    for images, target in loader:
        step += 1
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model.forward_normalized(images)
        loss = V.label_smoothing_cross_entropy(output, target, F2_SMOOTHING)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"{trajectory.name}: non-finite loss at step {step}")
        losses.append(value)
        correct_corrupt.append(float((output.detach()[:F2_HALF].argmax(1) == target[:F2_HALF]).float().mean()))
        if step % 50 == 0 or step == 1:
            entry = {"step": step, "loss": value, "loss_mean_last50": float(np.mean(losses[-50:])),
                     "train_acc_corrupt_half_last50": float(np.mean(correct_corrupt[-50:])),
                     "elapsed_s": round(time.perf_counter() - started, 2)}
            log.append(entry)
            if log_fn is not None:
                log_fn({"trajectory": trajectory.name, **entry})
        if step in wanted:
            relative = checkpoint_dir(trajectory, step)
            target_dir = Path(out_root) / relative
            staging = target_dir.with_name(f".staging-{target_dir.name}")
            staging.mkdir(parents=True)
            params = V.export_params(model)
            digest = save_params(staging / "model.npz", params)
            device_logits = V.eval_logits(model, check_rows)
            if not model.training:
                raise RuntimeError("eval_logits left the model in eval mode")
            config = {"n": trajectory.n, "lr": trajectory.lr, "steps": step}
            record = {
                "schema": "weightcode.f2.checkpoint.v1", "prereg": PREREG_F2, "method": F2_METHOD,
                "condition": trajectory.condition, "trajectory": trajectory.as_dict(), "config": config,
                "params_sha256": digest, "base_params_sha256": F2_STUDENT_SHA256, "student": F2_STUDENT,
                "recipe_kept": RECIPE_KEPT, "limits": LIMITS_NOTE,
                "optimizer": {"name": "AdamW", "lr": trajectory.lr, "betas": list(F2_BETAS), "eps": F2_EPS,
                              "weight_decay": F2_WEIGHT_DECAY, "schedule": "constant"},
                "loss": {"name": "label_smoothing_cross_entropy", "smoothing": F2_SMOOTHING},
                "batch": {"size": F2_BATCH, "corrupt": F2_HALF, "clean": F2_HALF,
                          "sampler": "BalancedBatchSampler, a fresh permutation per pass of each half"},
                "augment": f"vitlite.train_transform (CIFAR-10 AutoAugment, crop 32 pad {F2_CROP_PAD}, flip, normalise)",
                "model_config": V.ViTLiteConfig().as_dict(), "seed": seed, "num_workers": num_workers,
                "rows": rows_info, "data_json_sha256": data.data_json_sha256,
                "train_loss_mean_last100": float(np.mean(losses[-100:])),
                "train_acc_corrupt_half_last100": float(np.mean(correct_corrupt[-100:])),
                "device_check": {"rows": f"the first {F2_CHECK_ROWS} rows of cifar10_vl7/train (clean)",
                                 "device": str(device),
                                 "logits": [[float(v) for v in row] for row in device_logits]},
                "elapsed_s": round(time.perf_counter() - started, 2), "provenance": dict(provenance),
            }
            (staging / "record.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
            staging.rename(target_dir)
            rows.append({"trajectory": trajectory.name, "condition": trajectory.condition, **config,
                         "tag": trajectory.tag, "path": relative, "params_sha256": digest,
                         "train_loss_mean_last100": record["train_loss_mean_last100"],
                         "elapsed_s": record["elapsed_s"]})
    if step != trajectory.steps:
        raise RuntimeError(f"{trajectory.name}: ran {step} steps, expected {trajectory.steps}")
    (Path(out_root) / Path(checkpoint_dir(trajectory, trajectory.steps)).parent / "trajectory.json").write_text(
        json.dumps({"trajectory": trajectory.as_dict(), "log": log, "rows": rows,
                    "wall_s": round(time.perf_counter() - started, 2)}, indent=1) + "\n")
    return rows


# --------------------------------------------------------------------------------------------- import


def import_f2_run(run_dir: Path, models_subdir: str = "private/f2_models") -> dict[str, Any]:
    """Check a fetched F2 training run and write ``<run_dir>/f2_import.json`` (refuses an existing one).

    Per checkpoint: model.npz's hash equals the index row and its record; the record's base and data manifest hashes
    equal the student's and the local ``f2_data.json``; the store's forward.py (numpy float64) on the record's device
    rows agrees with the job's device logits (max abs at most ``F2_DEVICE_MAX_ABS``, no argmax disagreement).
    """
    from weightcode.data.store import read_json, write_json
    from weightcode.evalkit.targets import load_dataset_forward, read_dataset_meta

    run_dir = Path(run_dir)
    out = run_dir / F2_IMPORT_JSON
    if out.exists():
        raise F2Error(f"{out} exists")
    root = run_dir / models_subdir
    index = read_json(root / "index.json")
    meta = read_dataset_meta(F2_DATASET)
    forward, forward_sha, _ = load_dataset_forward(F2_DATASET, meta)
    data_sha = _file_sha256(f2_dir() / F2_DATA_JSON)
    from weightcode.data.store import load_split

    check_rows = np.ascontiguousarray(load_split(F2_DATASET, "train")["x"][:F2_CHECK_ROWS])
    rows = []
    for row in index["rows"]:
        path = root / row["path"]
        params = load_params(path / "model.npz")
        digest = params_sha256(params)
        record = json.loads((path / "record.json").read_text())
        if digest != row["params_sha256"] or digest != record["params_sha256"]:
            raise F2Error(f"{path}: params hash differs from the index or the record")
        if record["base_params_sha256"] != F2_STUDENT_SHA256 or record["data_json_sha256"] != data_sha:
            raise F2Error(f"{path}: trained from another student or another data manifest")
        local = np.asarray(forward(params, check_rows), dtype=np.float64)
        device = np.asarray(record["device_check"]["logits"], dtype=np.float64)
        check = {"max_abs_diff": float(np.max(np.abs(local - device))),
                 "argmax_disagreements": int(np.count_nonzero(local.argmax(1) != device.argmax(1)))}
        if check["max_abs_diff"] > F2_DEVICE_MAX_ABS or check["argmax_disagreements"]:
            raise F2Error(f"{path}: device check {check}")
        rows.append({**row, "device_check": check, "device": record["device_check"]["device"]})
    payload = {"schema": "weightcode.f2.import.v1", "run_dir": str(run_dir), "models_root": str(root),
               "index_sha256": _file_sha256(root / "index.json"), "data_json_sha256": data_sha,
               "forward_sha256": forward_sha, "n_checkpoints": len(rows), "rows": rows,
               "device_max_abs_diff": max(r["device_check"]["max_abs_diff"] for r in rows)}
    write_json(out, payload)
    return payload


# --------------------------------------------------------------------------------------------- checks


def _spec() -> Any:
    from .runner import MethodSpec

    return MethodSpec(
        method=F2_METHOD,
        label=("F2 label-equivalent amount: full fine-tuning of the student (AdamW, its recipe's conventions) on N "
               "labelled corrupted CIFAR-10 train images mixed 1:1 with N clean train images, trained on NSCC; the "
               f"instance's experience rows are not used; {LIMITS_NOTE}; not training-free; {PREREG_F2}"),
        uses_labels_for_fit=True,
        edit_limits_enforced=False,
    )


def condition_of(instance: Any) -> str:
    """The F2 condition (official function name) of an E11 instance, e.g. ``gaussian_noise`` for gaussian_noise_s5."""
    name = str(instance.private.get("condition"))
    if instance.private.get("dataset") != F2_DATASET or not name.endswith("_s5") or name[:-3] not in F2_CONDITIONS:
        raise F2Error(f"{instance.instance_id}: F2 runs on {F2_DATASET} instances of {F2_CONDITIONS}, got "
                      f"{instance.private.get('dataset')}/{name}")
    if instance.params_sha256 != F2_STUDENT_SHA256:
        raise F2Error(f"{instance.instance_id}: base {instance.params_sha256} is not the seed-100 student")
    return name[:-3]


def f2_generator(instance: Any, import_record: Mapping[str, Any], group: str) -> Any:
    """Candidate generator of one F2 check group: the imported checkpoints of (condition, N) in check order."""
    from .runner import CandidatePatch

    condition = condition_of(instance)
    n, configs = group_rows(group)
    root = Path(import_record["models_root"])

    def generate(evidence: Any) -> tuple[list[Any], dict[str, Any]]:
        candidates = []
        rows = []
        for slot, config in enumerate(configs):
            matches = [r for r in import_record["rows"] if r["condition"] == condition and r["n"] == n
                       and r["lr"] == config["lr"] and r["steps"] == config["steps"] and not r.get("tag")]
            if len(matches) != 1:
                raise F2Error(f"{condition} n {n} {config}: {len(matches)} imported checkpoints, expected 1")
            row = matches[0]
            child = load_params(root / row["path"] / "model.npz")
            if params_sha256(child) != row["params_sha256"]:
                raise F2Error(f"{row['path']}: params hash differs from the import record")
            rows.append(row)
            candidates.append(CandidatePatch(
                slot=slot, child=child, config={"n": n, **config},
                hypothesis=(f"{F2_METHOD}: AdamW lr={config['lr']:g} (constant) for {config['steps']} steps of batch "
                            f"{F2_BATCH} on {n} labelled {condition}_s5 train images and {n} clean train images"),
                predicted_effect=("fewer errors on held-out condition images as N grows; retention is fitted on the "
                                  f"{n} clean replay images"),
                notes={"train_loss_mean_last100": row["train_loss_mean_last100"], "path": row["path"],
                       "device_check": row["device_check"]}))
        info = {"prereg": PREREG_F2, "condition": f"{condition}_s5", "group": group, "n": n,
                "fit_rows": (f"{n} labelled corrupted train images ({condition}_s5, pinned official code) and {n} clean "
                             "train images, trained on NSCC; the instance's experience rows are not used"),
                "deviations": ["the evidence and replay rows described in this record were not used for fitting; F2 "
                               "fits N corrupted + N clean train images (generation.fit_rows)"]
                              + ([f"top grid point {F2_NS[-1]} (the train pool), not PREREG's {F2_N_SPEC_TOP}"]
                                 if n == F2_NS[-1] else []),
                "import": {"run_dir": import_record["run_dir"], "index_sha256": import_record["index_sha256"],
                           "data_json_sha256": import_record["data_json_sha256"]},
                "checkpoints": rows, "recipe_kept": RECIPE_KEPT, "limits": LIMITS_NOTE}
        return candidates, info

    return generate


def run_f2(instance: Any, import_record: Mapping[str, Any], group: str, run_dir: Path, *, run_seed: int = F2_RUN_SEED,
           allow_main: bool = False) -> dict[str, Any]:
    """Check one F2 group on one E11 instance with the references' joint oracle and executor (no size limits)."""
    from .runner import execute_run

    n, configs = group_rows(group)
    config = {"prereg": PREREG_F2, "group": group, "n": n, "candidate_order": configs, "recipe_kept": RECIPE_KEPT,
              "limits": LIMITS_NOTE, "training": "NSCC GPU (scripts/train_f2.py), imported with f2_import.json",
              "run_seed": run_seed}
    return execute_run(instance=instance, spec=_spec(), run_dir=run_dir, run_seed=run_seed, config=config,
                       generate=f2_generator(instance, import_record, group), allow_main=allow_main, joint=True,
                       replay_in_fit=True, e10=True)


# --------------------------------------------------------------------------------------------- N*


def n_star(points: Sequence[Mapping[str, Any]], teacher_joint: float,
           clean_floor: float = F2_CLEAN_FLOOR_POINTS) -> dict[str, Any]:
    """N*: the smallest N whose pick reaches ``teacher_joint`` (V_select joint, points) at clean >= ``clean_floor``,
    log-linearly interpolated in N on the joint between the previous grid point and the first that reaches it.

    ``points`` are the picks in increasing N: ``{"n", "joint", "clean"}`` in points. If the first grid point reaches
    it, N* is at most that N (no lower point to interpolate from). If the previous point's joint already reaches the
    bar but its clean change misses the floor, the joint cannot place N* between them and N* is the first passing N.
    """
    ordered = sorted(points, key=lambda p: p["n"])
    if [p["n"] for p in ordered] != sorted({p["n"] for p in ordered}):
        raise F2Error("one pick per N")
    passing = [i for i, p in enumerate(ordered) if p["joint"] >= teacher_joint and p["clean"] >= clean_floor]
    base = {"teacher_joint": teacher_joint, "clean_floor": clean_floor, "grid": [p["n"] for p in ordered]}
    if not passing:
        return {**base, "reached": False, "n_star": None,
                "text": f"not reached within {ordered[-1]['n']}"}
    k = passing[0]
    hit = ordered[k]
    if k == 0:
        return {**base, "reached": True, "n_star": hit["n"], "bound": "at_most",
                "text": f"<= {hit['n']} (the smallest grid point already reaches it)"}
    prev = ordered[k - 1]
    if prev["joint"] >= teacher_joint:
        return {**base, "reached": True, "n_star": hit["n"], "bound": "grid_point",
                "text": f"{hit['n']} (N = {prev['n']} reaches the joint but misses clean; no interpolation)"}
    t = (teacher_joint - prev["joint"]) / (hit["joint"] - prev["joint"])
    log_n = math.log(prev["n"]) + t * (math.log(hit["n"]) - math.log(prev["n"]))
    value = math.exp(log_n)
    return {**base, "reached": True, "n_star": value, "bound": "interpolated", "between": [prev["n"], hit["n"]],
            "fraction": t, "text": f"{value:.0f} (log-linear between {prev['n']} and {hit['n']})"}


__all__ = [
    "BalancedBatchSampler", "CEILING", "F2Data", "F2Error", "F2_CONDITIONS", "F2_DATASET", "F2_LRS", "F2_METHOD",
    "F2_NS", "F2_STEPS", "F2_TEACHER_JOINT", "Trajectory", "build_f2_data", "ceiling_configs", "checkpoint_dir",
    "condition_of", "f2_configs", "f2_generator", "f2_trajectories", "group_rows", "groups", "import_f2_run",
    "load_f2_data", "n_star", "run_f2", "smoke_trajectories", "train_trajectory", "trajectory_rows",
]
