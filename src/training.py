from __future__ import annotations

import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support, recall_score

from models import LiteBrainNet
from src.dataset import CLASS_TO_IDX, MRIClassificationDataset


def build_model(config: dict | None = None, **kwargs) -> nn.Module:
    cfg = config or {}
    model_cfg = cfg.get("model", {})
    model_kwargs = {
        "input_channels": int(model_cfg.get("input_channels", kwargs.pop("input_channels", 1))),
        "dropout": float(model_cfg.get("dropout", kwargs.pop("dropout", 0.25))),
        "channels": tuple(model_cfg.get("channels", kwargs.pop("channels", (16, 32, 64, 128)))),
        "num_classes": len(CLASS_TO_IDX),
    }
    model_kwargs.update(kwargs)
    return LiteBrainNet(**model_kwargs)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def compute_metrics(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += inputs.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
    accuracy = total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(total_samples, 1)
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        all_targets,
        all_preds,
        labels=list(CLASS_TO_IDX.values()),
        average=None,
        zero_division=0,
    )
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": float(f1_score(all_targets, all_preds, labels=list(CLASS_TO_IDX.values()), average="macro", zero_division=0))
        if all_targets
        else 0.0,
        "per_class_recall": recall_score(
            all_targets,
            all_preds,
            labels=list(CLASS_TO_IDX.values()),
            average=None,
            zero_division=0,
        ).tolist()
        if all_targets
        else [0.0] * len(CLASS_TO_IDX),
        "per_class_precision": class_precision.tolist(),
        "per_class_f1": class_f1.tolist(),
        "per_class_support": class_support.tolist(),
        "predictions": np.asarray(all_preds, dtype=np.int64),
        "targets": np.asarray(all_targets, dtype=np.int64),
    }


def train_model(
    train_manifest: str | Path,
    val_manifest: str | Path,
    config: dict,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    history_path: str | Path | None = None,
):
    cfg = config or {}
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = int(train_cfg.get("patience", 6))
    num_workers = int(train_cfg.get("num_workers", 0))
    loss_cfg = train_cfg.get("loss", {})
    augmentation_cfg = train_cfg.get("augmentation", {})
    scheduler_cfg = train_cfg.get("scheduler", {})
    if epochs < 1:
        raise ValueError("training.epochs must be at least one")
    if batch_size < 1:
        raise ValueError("training.batch_size must be at least one")
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {checkpoint_path}")
    if history_path is not None and Path(history_path).exists():
        raise FileExistsError(f"Refusing to overwrite existing training history: {history_path}")
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device is None:
        configured_device = train_cfg.get("device", "auto")
        device = (
            configured_device
            if configured_device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    device = torch.device(device)

    train_frame = pd.read_csv(train_manifest) if isinstance(train_manifest, (str, Path)) else pd.DataFrame(train_manifest)
    val_frame = pd.read_csv(val_manifest) if isinstance(val_manifest, (str, Path)) else pd.DataFrame(val_manifest)

    def validate_manifest(frame, name, expected_split):
        if not {"path", "label"}.issubset(frame.columns):
            raise ValueError(f"{name} manifest must contain path and label columns")
        if "official_partition" in frame and not (
            frame["official_partition"].astype(str).str.lower() == "train"
        ).all():
            raise ValueError(f"{name} manifest must use only the official training partition")
        if "split" in frame and not (frame["split"].astype(str).str.lower() == expected_split).all():
            raise ValueError(f"{name} manifest must contain only split={expected_split!r} rows")
        for image_path in frame["path"].astype(str):
            normalized = image_path.replace("\\", "/").lower()
            parts = normalized.split("/")
            if "demo" in parts:
                raise ValueError("Synthetic demo images are not permitted for model training or validation")
            if "test" in parts:
                raise ValueError(f"{name} manifest contains a test-path image: {image_path}")

    validate_manifest(train_frame, "Training", "train")
    validate_manifest(val_frame, "Validation", "val")
    train_paths = set(train_frame["path"].astype(str))
    val_paths = set(val_frame["path"].astype(str))
    if train_paths.intersection(val_paths):
        raise ValueError("Training and validation manifests contain overlapping image paths")
    if "sha256" in train_frame and "sha256" in val_frame:
        train_hashes = set(train_frame["sha256"].dropna().astype(str))
        val_hashes = set(val_frame["sha256"].dropna().astype(str))
        if train_hashes.intersection(val_hashes):
            raise ValueError("Training and validation manifests contain exact duplicate images")

    train_set = MRIClassificationDataset(
        train_frame,
        image_size=image_size,
        augment=bool(augmentation_cfg.get("enabled", True)),
        augmentation_config=augmentation_cfg,
    )
    val_set = MRIClassificationDataset(val_frame, image_size=image_size, augment=False)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model(cfg).to(device)
    class_counts = train_frame["label"].map(lambda label: CLASS_TO_IDX.get(str(label))).value_counts()
    missing_classes = set(CLASS_TO_IDX.values()) - set(class_counts.index)
    if missing_classes:
        raise ValueError(f"Training manifest is missing classes: {sorted(missing_classes)}")
    class_weights = torch.tensor(
        [
            len(train_frame) / (len(CLASS_TO_IDX) * int(class_counts[index]))
            for index in range(len(CLASS_TO_IDX))
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights if loss_cfg.get("class_weighting", True) else None,
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", max(0, patience // 2))),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
    )

    best_state = None
    best_val_loss = float("inf")
    best_val_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_train_metrics = {"loss": float("inf"), "accuracy": 0.0}
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            running_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_samples += inputs.size(0)

        train_metrics = {
            "loss": running_loss / max(running_samples, 1),
            "accuracy": running_correct / max(running_samples, 1),
        }
        val_metrics = compute_metrics(model, val_loader, device, criterion)
        scheduler.step(val_metrics["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        print(
            f"epoch={epoch:02d} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"lr={current_lr:.2e}",
            flush=True,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class_recall": {
                label: val_metrics["per_class_recall"][index]
                for label, index in CLASS_TO_IDX.items()
            },
            "val_per_class_metrics": {
                label: {
                    "precision": val_metrics["per_class_precision"][index],
                    "recall": val_metrics["per_class_recall"][index],
                    "f1": val_metrics["per_class_f1"][index],
                    "support": val_metrics["per_class_support"][index],
                }
                for label, index in CLASS_TO_IDX.items()
            },
            "learning_rate": current_lr,
        }
        history.append(history_row)
        if history_path is not None:
            history_file = Path(history_path)
            history_file.parent.mkdir(parents=True, exist_ok=True)
            with history_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(history_row) + "\n")

        improved = (
            val_metrics["macro_f1"] > best_val_macro_f1
            or (
                val_metrics["macro_f1"] == best_val_macro_f1
                and val_metrics["loss"] < best_val_loss
            )
        )
        if improved:
            best_val_loss = val_metrics["loss"]
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_train_metrics = train_metrics
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(
                f"Early stopping triggered at epoch {epoch}; best validation macro-F1 was "
                f"{best_val_macro_f1:.4f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_path is not None:
            checkpoint = Path(checkpoint_path)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": best_state,
                    "epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_macro_f1": best_val_macro_f1,
                    "class_weights": class_weights.cpu().tolist(),
                    "seed": seed,
                    "training_config": train_cfg,
                    "history": history,
                },
                checkpoint,
            )
    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_macro_f1": best_val_macro_f1,
        "train_metrics": best_train_metrics,
        "val_metrics": compute_metrics(model, val_loader, device, criterion),
        "history": history,
    }


def load_training_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path is not None else Path("config.yaml")
    import yaml

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return data
