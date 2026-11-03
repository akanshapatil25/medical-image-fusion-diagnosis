from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.dataset import CLASS_TO_IDX, canonical_label
from src.preprocessing import image_to_tensor


def predict_image(model, image_source, device=None, label_map=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(image_source, (str, Path)):
        image = Image.open(image_source).convert("L")
    else:
        image = image_source.convert("L")
    tensor = image_to_tensor(image, size=(224, 224), grayscale=True).unsqueeze(0).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    labels = list(label_map or CLASS_TO_IDX.keys())
    ranked = sorted(zip(labels, probs.tolist()), key=lambda item: item[1], reverse=True)
    return ranked


def predict_manifest(model, manifest, device=None):
    frame = pd.read_csv(manifest)
    rows = []
    for _, row in frame.iterrows():
        result = predict_image(model, row["path"], device=device)
        rows.append({"path": row["path"], "true_label": row["label"], "predictions": result})
    return rows
