import torch
import pandas as pd
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training import compute_metrics, train_model


def test_compute_metrics_returns_per_class_precision_recall_f1_and_support():
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
    )
    targets = torch.tensor([0, 1, 2, 3])
    loader = DataLoader(TensorDataset(logits, targets), batch_size=4)

    metrics = compute_metrics(nn.Identity(), loader, torch.device("cpu"), nn.CrossEntropyLoss())

    assert metrics["per_class_precision"] == [0.5, 0.0, 1.0, 1.0]
    assert metrics["per_class_recall"] == [1.0, 0.0, 1.0, 1.0]
    assert metrics["per_class_f1"] == pytest.approx([2 / 3, 0.0, 1.0, 1.0])
    assert metrics["per_class_support"] == [1, 1, 1, 1]


def test_training_rejects_official_test_partition():
    test = pd.DataFrame(
        [
            {
                "path": "C:/brisc/classification_task/test/glioma/test.jpg",
                "label": "glioma",
                "official_partition": "test",
                "split": "test",
            }
        ]
    )
    train = pd.DataFrame(
        [
            {
                "path": "C:/brisc/classification_task/train/glioma/train.jpg",
                "label": "glioma",
                "official_partition": "train",
                "split": "val",
            }
        ]
    )
    with pytest.raises(ValueError, match="official training partition"):
        train_model(test, train, config={})


def test_training_rejects_synthetic_demo_paths():
    train = pd.DataFrame(
        [
            {
                "path": "C:/project/data/demo/glioma/demo.jpg",
                "label": "glioma",
                "official_partition": "train",
                "split": "train",
            }
        ]
    )
    validation = train.assign(path="C:/project/data/demo/glioma/demo_val.jpg", split="val")
    with pytest.raises(ValueError, match="Synthetic demo"):
        train_model(train, validation, config={})
