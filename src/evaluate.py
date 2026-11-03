from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import LiteBrainNet
from src.dataset import MRIClassificationDataset
from src.metrics import evaluate_predictions
from src.training import build_model


def evaluate_model(model, loader, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return evaluate_predictions(targets, predictions)


def evaluate_manifest(model_path, manifest, config=None, device=None, batch_size=16):
    config = config or {}
    model = build_model(config)
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    dataset = MRIClassificationDataset(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return evaluate_model(model, loader, device=device)
