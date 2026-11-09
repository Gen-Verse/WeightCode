"""Fixed dataset definitions: sources, class names, image shapes and split sizes (contract section 2)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

SPLIT_SEED = 20261103
"""Seed of the stratified split of every official train set."""

SPLIT_NAMES: tuple[str, ...] = ("train", "V_select", "V_evidence", "V_check")
"""Split names on disk, exactly as consumed by the other sessions."""

ALLOCATION_ORDER: tuple[str, ...] = ("V_select", "V_evidence", "V_check", "train")
"""Order in which each class's shuffled indices are dealt to splits; train takes the remainder."""

N_CLASSES = 10


def _frozen(mapping: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True)
class DatasetConfig:
    """Static description of one dataset of the primary task family."""

    name: str
    source: str
    """torchvision dataset class that provides the official train and test sets."""
    class_names: tuple[str, ...]
    image_shape: tuple[int, int, int]
    """Stored image shape [C, H, W] (uint8)."""
    n_official_train: int
    n_official_test: int
    split_sizes: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if tuple(self.split_sizes) != SPLIT_NAMES:
            raise ValueError(f"{self.name}: split sizes must be given for {SPLIT_NAMES} in that order")
        if sum(self.split_sizes.values()) != self.n_official_train:
            raise ValueError(f"{self.name}: split sizes do not sum to {self.n_official_train}")
        if len(self.class_names) != N_CLASSES:
            raise ValueError(f"{self.name}: expected {N_CLASSES} class names")


FASHION_MNIST_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
DIGIT_CLASSES = tuple(str(digit) for digit in range(10))

DATASETS: Mapping[str, DatasetConfig] = MappingProxyType(
    {
        "fashion_mnist": DatasetConfig(
            name="fashion_mnist",
            source="FashionMNIST",
            class_names=FASHION_MNIST_CLASSES,
            image_shape=(1, 28, 28),
            n_official_train=60_000,
            n_official_test=10_000,
            split_sizes=_frozen({"train": 50_000, "V_select": 4_000, "V_evidence": 2_000, "V_check": 4_000}),
        ),
        "cifar10": DatasetConfig(
            name="cifar10",
            source="CIFAR10",
            class_names=CIFAR10_CLASSES,
            image_shape=(3, 32, 32),
            n_official_train=50_000,
            n_official_test=10_000,
            split_sizes=_frozen({"train": 40_000, "V_select": 4_000, "V_evidence": 2_000, "V_check": 4_000}),
        ),
        "colored_mnist": DatasetConfig(
            name="colored_mnist",
            source="MNIST",
            class_names=DIGIT_CLASSES,
            image_shape=(3, 28, 28),
            n_official_train=60_000,
            n_official_test=10_000,
            split_sizes=_frozen({"train": 50_000, "V_select": 4_000, "V_evidence": 2_000, "V_check": 4_000}),
        ),
    }
)


def get_config(name: str) -> DatasetConfig:
    """Return the registered configuration of ``name`` or raise with the list of known datasets."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return DATASETS[name]
