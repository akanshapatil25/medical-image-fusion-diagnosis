import hashlib
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from src.preprocessing import SUPPORTED_EXTENSIONS, image_to_float01, load_image


CLASS_ALIASES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "pituitary tumor": "pituitary",
    "no_tumor": "no_tumor",
    "notumor": "no_tumor",
    "no tumor": "no_tumor",
    "healthy": "no_tumor",
}
CLASS_TO_IDX = {"glioma": 0, "meningioma": 1, "pituitary": 2, "no_tumor": 3}


def canonical_label(value):
    key = str(value).strip().lower().replace("-", "_")
    return CLASS_ALIASES.get(key)


def discover_images(root):
    """Scan images whose immediate parent directory identifies a supported class."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    rows = []
    unlabeled = []
    corrupt = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        label = canonical_label(path.parent.name)
        if label is None:
            unlabeled.append(str(path))
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            digest = sha256_file(path)
            rows.append(
                {"path": str(path), "label": label, "width": width, "height": height, "sha256": digest}
            )
        except Exception as exc:
            corrupt.append({"path": str(path), "error": str(exc)})
    return rows, {"unlabeled": unlabeled, "corrupt": corrupt}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_records(records):
    """Deduplicate byte-identical files; conflicting duplicate labels are rejected."""
    by_hash = {}
    for row in records:
        previous = by_hash.get(row["sha256"])
        if previous is not None and previous["label"] != row["label"]:
            raise ValueError(
                f"Identical image has conflicting labels: {previous['path']} and {row['path']}"
            )
        by_hash.setdefault(row["sha256"], row)
    return list(by_hash.values())


def create_split_manifests(records, output_dir, seed=42, ratios=(0.70, 0.12, 0.18)):
    """Make deterministic stratified splits after exact-hash deduplication."""
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("ratios must be three positive values summing to one")
    records = unique_records(records)
    rng = random.Random(seed)
    partitions = {"train": [], "val": [], "test": []}
    for label in CLASS_TO_IDX:
        group = [row for row in records if row["label"] == label]
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * ratios[0] + 0.5)
        n_val = int(n * ratios[1] + 0.5)
        n_val = min(n_val, n - n_train)
        partitions["train"].extend(group[:n_train])
        partitions["val"].extend(group[n_train : n_train + n_val])
        partitions["test"].extend(group[n_train + n_val :])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(records[0]["path"]).parents[0] if records else None
    for name, rows in partitions.items():
        if root:
            saved = [{**r, "path": str(Path(r["path"]).resolve())} for r in rows]
        else:
            saved = rows
        pd.DataFrame(saved, columns=["path", "label", "width", "height", "sha256"]).to_csv(
            output_dir / f"{name}.csv", index=False
        )
    return partitions


class MRIClassificationDataset(Dataset):
    def __init__(self, manifest, image_size=224, augment=False, augmentation_config=None):
        self.frame = pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else pd.DataFrame(manifest)
        if not {"path", "label"}.issubset(self.frame.columns):
            raise ValueError("Manifest must contain path and label columns")
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.augmentation_config = augmentation_config or {}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = load_image(row["path"], grayscale=True)
        if self.augment:
            flip_probability = float(self.augmentation_config.get("horizontal_flip_probability", 0.5))
            rotation_degrees = float(self.augmentation_config.get("rotation_degrees", 8.0))
            brightness_delta = float(self.augmentation_config.get("brightness_delta", 0.0))
            contrast_delta = float(self.augmentation_config.get("contrast_delta", 0.0))
            if random.random() < flip_probability:
                image = ImageOps.mirror(image)
            if rotation_degrees > 0:
                image = image.rotate(
                    random.uniform(-rotation_degrees, rotation_degrees),
                    resample=Image.Resampling.BILINEAR,
                )
            if brightness_delta > 0:
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(1 - brightness_delta, 1 + brightness_delta)
                )
            if contrast_delta > 0:
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(1 - contrast_delta, 1 + contrast_delta)
                )
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(image_to_float01(image).copy()).unsqueeze(0)
        label = canonical_label(row["label"])
        if label not in CLASS_TO_IDX:
            raise ValueError(f"Unsupported class label: {row['label']}")
        return tensor, torch.tensor(CLASS_TO_IDX[label], dtype=torch.long)
