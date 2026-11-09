"""Prepare explicit user-chosen image splits; no automatic sampling or splitting."""

from pathlib import Path

import numpy as np

from weightcode.application.bundles import INPUT, model_bundle


def prepare_data(model: Path, new: Path, old: Path, out: Path) -> dict:
    from PIL import Image
    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, PILToTensor, Resize

    m, _ = model_bundle(model)
    if out.suffix != ".npz":
        raise ValueError("--out must name an explicit .npz file")
    if out.exists():
        raise FileExistsError(out)
    transform = Compose([Resize(INPUT["resize_shorter"], InterpolationMode.BILINEAR, antialias=True),
                         CenterCrop(INPUT["crop"]), PILToTensor()])
    labels = {label: index for index, label in enumerate(m["labels"])}
    data = {}
    for role, directory in (("new", new), ("old", old)):
        if not directory.is_dir():
            raise ValueError(f"missing {role} image directory: {directory}")
        xs, ys = [], []
        for folder in sorted(directory.iterdir()):
            if not folder.is_dir() or folder.name not in labels:
                raise ValueError(f"expected class-name subdirectory: {folder}")
            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    raise ValueError(f"expected image file: {path}")
                with Image.open(path) as image:
                    xs.append(transform(image.convert("RGB")).numpy())
                ys.append(labels[folder.name])
        if not xs:
            raise ValueError(f"empty {role} image directory: {directory}")
        data[f"x_{role}"] = np.stack(xs)
        data[f"y_{role}"] = np.asarray(ys, dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        np.savez(handle, **data)
    return {"out": str(out), "new_images": len(data["y_new"]), "old_images": len(data["y_old"])}
