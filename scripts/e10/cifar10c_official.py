"""PREREG E10: the official CIFAR-10-C corruption functions, run as written (hendrycks/robustness, pinned).

Runs in its own environment (Python 3.11 with numpy 1.26.4, scipy 1.11.4, opencv-python-headless 4.8.1.78,
scikit-image 0.22.0, Wand 0.6.13 on ImageMagick 7, Pillow 10.4.0, torch 2.2.2 and torchvision 0.17.2, CPU), not the
project's: the official file needs these libraries and the project environment has none of them.

The file ``ImageNet-C/create_c/make_cifar_c.py`` is read from a local copy whose sha256 must be ``OFFICIAL_SHA256``
(the file at repository commit ``OFFICIAL_COMMIT``; it last changed in ca360085, 2019-01-09). Its text up to the
``END_MARKER`` line (imports, helpers and every distortion function) is executed as is; the rest of the file (the
loop over the test set and the saves) is not run. ``corrupt`` reproduces that loop for one corruption at one
severity on any NCHW uint8 images: each image goes through the file's ``convert_img`` (torchvision ToTensor then
ToPILImage), the distortion, and ``np.uint8`` of the result (truncation toward zero), as at line 458. The functions
draw from numpy's global RandomState, so ``corrupt`` seeds it once (``np.random.seed(seed)``) before the first image
and draws in row order.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

OFFICIAL_REPOSITORY = "https://github.com/hendrycks/robustness"
OFFICIAL_FILE = "ImageNet-C/create_c/make_cifar_c.py"
OFFICIAL_COMMIT = "8190fe329f5f072a06e3a2aea02bb5dda69aed9f"
OFFICIAL_FILE_LAST_CHANGED = "ca360085d57889288c4143fddd8a707bbcae01bd"
OFFICIAL_SHA256 = "f865ae91dffb2b4834292ce05c876bf3d77f36417f70075a2b5259348b0c83a5"
OFFICIAL_URL = f"https://raw.githubusercontent.com/hendrycks/robustness/{OFFICIAL_COMMIT}/{OFFICIAL_FILE}"
END_MARKER = "# /////////////// End Distortions ///////////////"
SEVERITY = 5
N_PER_SEVERITY = 10000  # the official .npy files hold severities 1..5 in order, 10,000 test images each

# E10's conditions (the first corruption of each family) and snow's substitute, by the official function names.
CONDITIONS = ("gaussian_noise", "defocus_blur", "snow", "contrast")
SUBSTITUTE = {"snow": "fog"}
DETERMINISTIC = ("defocus_blur", "contrast")
RANDOM = ("gaussian_noise", "snow", "fog")


class OfficialCodeError(RuntimeError):
    """The official file is missing, altered, or does not define what E10 needs."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official(path: Path) -> dict[str, Any]:
    """Execute the official file's text up to ``END_MARKER`` after checking its sha256; returns its namespace."""
    path = Path(path)
    if file_sha256(path) != OFFICIAL_SHA256:
        raise OfficialCodeError(f"{path}: sha256 differs from {OFFICIAL_SHA256}")
    text = path.read_text(encoding="utf-8")
    if text.count(END_MARKER) != 1:
        raise OfficialCodeError(f"{path}: expected exactly one {END_MARKER!r} line")
    head = text.split(END_MARKER)[0]
    namespace: dict[str, Any] = {"__name__": "make_cifar_c_distortions"}
    exec(compile(head, str(path), "exec"), namespace)
    missing = [name for name in CONDITIONS + tuple(SUBSTITUTE.values()) if not callable(namespace.get(name))]
    if missing:
        raise OfficialCodeError(f"{path} does not define {missing}")
    import torchvision.transforms as trn

    namespace["convert_img"] = trn.Compose([trn.ToTensor(), trn.ToPILImage()])  # line 448 of the official file
    return namespace


def corrupt(official: dict[str, Any], name: str, x: np.ndarray, seed: int, severity: int = SEVERITY,
            progress: Callable[[int], None] | None = None) -> np.ndarray:
    """Corrupt NCHW uint8 images with the official function ``name`` at ``severity`` (module docstring)."""
    if x.dtype != np.uint8 or x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expected NCHW uint8 images with 3 channels, got {x.dtype} {x.shape}")
    function = official[name]
    convert = official["convert_img"]
    np.random.seed(int(seed))
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        image = np.ascontiguousarray(x[i].transpose(1, 2, 0))  # HWC, as torchvision's CIFAR10.data holds it
        out[i] = np.uint8(function(convert(image), severity)).transpose(2, 0, 1)
        if progress is not None and (i + 1) % 1000 == 0:
            progress(i + 1)
    return out


def library_versions() -> dict[str, str]:
    import sys

    import cv2
    import PIL
    import scipy
    import skimage
    import torch
    import torchvision
    import wand.version

    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "opencv": cv2.__version__, "scikit-image": skimage.__version__, "Pillow": PIL.__version__,
            "torch": torch.__version__, "torchvision": torchvision.__version__, "Wand": wand.version.VERSION,
            "ImageMagick": wand.version.MAGICK_VERSION}
