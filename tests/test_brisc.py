import numpy as np
import pandas as pd
from PIL import Image

from src.brisc import prepare_brisc_manifests


def test_brisc_manifests_keep_official_test_partition(tmp_path):
    root = tmp_path / "brisc2025"
    class_root = root / "classification_task"
    classes = ("glioma", "meningioma", "pituitary", "no_tumor")
    rng = np.random.default_rng(42)
    glioma_base = None
    for partition, count in (("train", 5), ("test", 2)):
        for label in classes:
            folder = class_root / partition / label
            folder.mkdir(parents=True)
            for index in range(count):
                if partition == "train" and label == "glioma" and index == 1:
                    array = glioma_base.copy()
                    array[0, 0] ^= 1
                else:
                    array = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
                    if partition == "train" and label == "glioma" and index == 0:
                        glioma_base = array.copy()
                Image.fromarray(array).save(folder / f"{partition}_{label}_{index}.png")

    report = prepare_brisc_manifests(root, tmp_path / "splits", seed=7, validation_fraction=0.2)
    train = pd.read_csv(tmp_path / "splits" / "train.csv")
    validation = pd.read_csv(tmp_path / "splits" / "val.csv")
    test = pd.read_csv(tmp_path / "splits" / "test.csv")

    assert len(train) == 16
    assert len(validation) == 4
    assert len(test) == 8
    assert set(train["official_partition"]) == {"train"}
    assert set(validation["official_partition"]) == {"train"}
    assert set(test["official_partition"]) == {"test"}
    assert not set(train["path"]).intersection(test["path"])
    assert report["near_duplicate_pair_count"] >= 1
    assert report["near_duplicate_edges_crossing_train_validation"] == 0
    near_duplicate_paths = {
        str((class_root / "train" / "glioma" / f"train_glioma_{index}.png").resolve())
        for index in (0, 1)
    }
    assert near_duplicate_paths <= set(train["path"]) or near_duplicate_paths <= set(validation["path"])
    assert report["patient_metadata"]["patient_level_identifiers_available"] is False
