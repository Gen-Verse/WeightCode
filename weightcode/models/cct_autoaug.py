"""CIFAR-10 AutoAugment policy of the Compact-Transformers CIFAR script (PREREG E11's recipe), vendored.

Source: SHI-Labs/Compact-Transformers at commit cd11f0912749d376f7465cb83534e768b6701d7d, files
``examples/utils/autoaug.py`` (sha256 d85cc9da9b2658dae566c2e46c530eded794ad78b93ad08f2d88397d0d0907ae, class
``CIFAR10Policy`` and ``SubPolicy``) and ``examples/utils/transforms.py`` (sha256
1e1f390bebc2c7896e581bf3f925e75d52c401fa7552ca5670c70b6490da0d55), themselves credited to DeepVoltaire/AutoAugment.
Licensed under the Apache License 2.0 (the repository's LICENSE). Copied with the logic unchanged: the 25 CIFAR-10
sub-policies, their probabilities and magnitude indices, the magnitude ranges, the fill colour (128, 128, 128) and
the operations, which draw from Python's ``random`` module as there. Changes: ImageNetPolicy is left out, the
classes are renamed with a leading underscore where they are internal, and ``ranges`` and ``func`` are built once
per sub-policy as in the source.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

FILL = (128, 128, 128)
SOURCE = ("SHI-Labs/Compact-Transformers cd11f0912749d376f7465cb83534e768b6701d7d examples/utils/autoaug.py "
          "CIFAR10Policy (Apache-2.0, after DeepVoltaire/AutoAugment)")


class _ShearX:
    def __init__(self, fillcolor=FILL):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(x.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                           Image.BICUBIC, fillcolor=self.fillcolor)


class _ShearY:
    def __init__(self, fillcolor=FILL):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(x.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                           Image.BICUBIC, fillcolor=self.fillcolor)


class _TranslateX:
    def __init__(self, fillcolor=FILL):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(x.size, Image.AFFINE, (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
                           fillcolor=self.fillcolor)


class _TranslateY:
    def __init__(self, fillcolor=FILL):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
                           fillcolor=self.fillcolor)


class _Rotate:
    def __call__(self, x, magnitude):
        rot = x.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(x.mode)


class _Color:
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))


class _Posterize:
    def __call__(self, x, magnitude):
        return ImageOps.posterize(x, magnitude)


class _Solarize:
    def __call__(self, x, magnitude):
        return ImageOps.solarize(x, magnitude)


class _Contrast:
    def __call__(self, x, magnitude):
        return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))


class _Sharpness:
    def __call__(self, x, magnitude):
        return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class _Brightness:
    def __call__(self, x, magnitude):
        return ImageEnhance.Brightness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class _AutoContrast:
    def __call__(self, x, magnitude):
        return ImageOps.autocontrast(x)


class _Equalize:
    def __call__(self, x, magnitude):
        return ImageOps.equalize(x)


class _Invert:
    def __call__(self, x, magnitude):
        return ImageOps.invert(x)


class SubPolicy:
    """Two operations, each applied with its own probability, at fixed magnitudes (the source's ``SubPolicy``)."""

    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=FILL):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int32),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }
        func = {
            "shearX": _ShearX(fillcolor=fillcolor),
            "shearY": _ShearY(fillcolor=fillcolor),
            "translateX": _TranslateX(fillcolor=fillcolor),
            "translateY": _TranslateY(fillcolor=fillcolor),
            "rotate": _Rotate(),
            "color": _Color(),
            "posterize": _Posterize(),
            "solarize": _Solarize(),
            "contrast": _Contrast(),
            "sharpness": _Sharpness(),
            "brightness": _Brightness(),
            "autocontrast": _AutoContrast(),
            "equalize": _Equalize(),
            "invert": _Invert(),
        }
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class CIFAR10Policy:
    """Randomly choose one of the best 25 sub-policies on CIFAR-10 (the source's ``CIFAR10Policy``)."""

    def __init__(self, fillcolor=FILL):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


__all__ = ["CIFAR10Policy", "FILL", "SOURCE", "SubPolicy"]
