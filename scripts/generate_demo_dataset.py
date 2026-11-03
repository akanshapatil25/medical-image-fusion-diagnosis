from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]


def add_blob(array, center, radius, intensity):
    yy, xx = np.ogrid[: array.shape[0], : array.shape[1]]
    dist = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    mask = dist <= radius**2
    array[mask] = intensity
    return array


def make_demo_sample(label: str, index: int, size: int = 224):
    rng = np.random.default_rng((abs(hash(label + str(index))) % (2**32)))
    image = np.full((size, size), 30, dtype=np.uint8)
    if label == "glioma":
        image = add_blob(image, (size // 2, size // 2), size // 6, 220)
        image = add_blob(image, (size // 2 + 25, size // 2 - 10), size // 10, 100)
    elif label == "meningioma":
        image = add_blob(image, (size // 2, size // 2), size // 5, 210)
        for dx, dy in [(-18, 12), (18, -12), (0, 25)]:
            image = add_blob(image, (size // 2 + dx, size // 2 + dy), size // 12, 150)
    elif label == "pituitary":
        image = add_blob(image, (size // 2, size // 2), size // 8, 200)
        image = add_blob(image, (size // 2, size // 2), size // 2, 40)
    elif label == "no_tumor":
        image = add_blob(image, (size // 2, size // 2), size // 3, 70)
        image = add_blob(image, (size // 2, size // 2), size // 10, 240)

    noise = rng.normal(0, 8, size=(size, size)).astype(np.int16)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="L")


def main():
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic MRI-like demo dataset")
    parser.add_argument("--root", default="data/demo", help="Output root directory")
    parser.add_argument("--per-class", type=int, default=6, help="Images per class")
    args = parser.parse_args()

    root = Path(args.root)
    for label in CLASSES:
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            img = make_demo_sample(label, i)
            img.save(folder / f"{label}_{i:02d}.png")
    print(f"Created synthetic demo dataset at {root}")
    for label in CLASSES:
        print(f"  {label}: {len(list((root / label).glob('*.png')))} images")


if __name__ == "__main__":
    main()
