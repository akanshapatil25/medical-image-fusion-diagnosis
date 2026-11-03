from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.dataset import CLASS_TO_IDX

def evaluate_predictions(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels) if labels is not None else list(CLASS_TO_IDX.values())
    target_names = [next(name for name, index in CLASS_TO_IDX.items() if index == label) for label in labels]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": target_names,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
    }
    return metrics
