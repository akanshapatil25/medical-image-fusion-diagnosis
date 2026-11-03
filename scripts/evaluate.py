from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CLASS_TO_IDX, MRIClassificationDataset
from src.explainability import gradcam_heatmap
from src.metrics import evaluate_predictions
from src.training import build_model, load_training_config


def _assert_official_test_manifest(frame):
    if "official_partition" not in frame or not (frame["official_partition"].astype(str).str.lower() == "test").all():
        raise ValueError("Evaluation requires a manifest containing only BRISC's official test partition")
    if "split" in frame and not (frame["split"].astype(str).str.lower() == "test").all():
        raise ValueError("Evaluation manifest has rows outside split='test'")
    if frame["path"].astype(str).str.replace("\\", "/", regex=False).str.lower().str.contains("/demo/").any():
        raise ValueError("Synthetic demo images are not permitted for evaluation")


def _save_confusion_matrix(matrix, labels, output_path):
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="BRISC official test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = max(matrix.max(), 1) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_gradcam_examples(model, dataset, frame, predictions, output_dir, device, max_per_class):
    import cv2
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    counts = {label: 0 for label in CLASS_TO_IDX}
    for index, row in frame.reset_index(drop=True).iterrows():
        label = str(row["label"])
        if label not in counts or counts[label] >= max_per_class:
            continue
        image_tensor, true_index = dataset[index]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        predicted_index = int(predictions[index])
        heatmap = gradcam_heatmap(model, image_tensor, target_class=predicted_index)[0]
        original = np.asarray(Image.open(row["path"]).convert("L"))
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        color = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        base = np.repeat(original[..., None], 3, axis=2)
        overlay = np.clip(0.55 * base + 0.45 * color, 0, 255).astype(np.uint8)
        predicted_label = next(name for name, idx in CLASS_TO_IDX.items() if idx == predicted_index)
        true_label = next(name for name, idx in CLASS_TO_IDX.items() if idx == int(true_index))
        output_path = output_dir / f"{true_label}_{counts[label] + 1:02d}.png"
        fig, axis = plt.subplots(figsize=(5, 5))
        axis.imshow(overlay)
        axis.set_title(f"True: {true_label} | Predicted: {predicted_label}")
        axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        saved.append(str(output_path))
        counts[label] += 1
    return saved


def main():
    parser = argparse.ArgumentParser(description="Evaluate LiteBrainNet on the untouched BRISC official test split")
    parser.add_argument("--manifest", default="data/splits/brisc/test.csv")
    parser.add_argument("--checkpoint", default="models/checkpoints/brisc_litebrainnet.pt")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--reports", default="outputs/reports/brisc_test")
    parser.add_argument("--gradcam-dir", default="outputs/figures/brisc_test_gradcam")
    parser.add_argument("--gradcam-per-class", type=int, default=3)
    parser.add_argument(
        "--baseline-metrics",
        default="outputs/reports/brisc_test/metrics.json",
        help="Existing pilot metrics JSON used only for the final comparison report",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    _assert_official_test_manifest(frame)
    config = load_training_config(args.config)
    device_name = config.get("training", {}).get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    model.to(device).eval()
    dataset = MRIClassificationDataset(
        frame,
        image_size=int(config.get("data", {}).get("image_size", 224)),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 16)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 0)),
    )
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = evaluate_predictions(y_true, y_pred)
    report_dir = Path(args.reports)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics["dataset"] = "BRISC 2025"
    metrics["evaluated_partition"] = "official_test"
    metrics["test_samples"] = len(frame)
    metrics["checkpoint"] = str(Path(args.checkpoint))
    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            metrics["checkpoint_best_epoch"] = int(checkpoint["epoch"])
        if "best_val_loss" in checkpoint:
            metrics["checkpoint_best_validation_loss"] = float(checkpoint["best_val_loss"])
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    prediction_frame = frame.reset_index(drop=True).copy()
    prediction_frame["true_index"] = y_true
    prediction_frame["predicted_index"] = y_pred
    prediction_frame["predicted_label"] = [
        next(name for name, class_index in CLASS_TO_IDX.items() if class_index == index)
        for index in y_pred
    ]
    prediction_frame["correct"] = prediction_frame["true_index"] == prediction_frame["predicted_index"]
    prediction_frame["plane"] = prediction_frame["filename"].astype(str).str.extract(r"_(ax|co|sa)_", flags=re.I)[0].str.lower()
    prediction_frame.to_csv(report_dir / "predictions.csv", index=False)
    class_report = metrics["classification_report"]
    pd.DataFrame(
        [
            {"class": label, **class_report[label]}
            for label in CLASS_TO_IDX
        ]
    ).to_csv(report_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["labels"],
        columns=metrics["labels"],
    ).to_csv(report_dir / "confusion_matrix.csv")
    plane_rows = []
    for plane, group in prediction_frame.groupby("plane", dropna=False):
        for class_name, class_index in CLASS_TO_IDX.items():
            class_rows = group[group["true_index"] == class_index]
            true_positive = int(
                ((class_rows["true_index"] == class_index) & (class_rows["predicted_index"] == class_index)).sum()
            )
            plane_rows.append(
                {
                    "plane": plane if pd.notna(plane) else "unknown",
                    "class": class_name,
                    "support": len(class_rows),
                    "recall": true_positive / len(class_rows) if len(class_rows) else 0.0,
                }
            )
    pd.DataFrame(plane_rows).to_csv(report_dir / "recall_by_plane.csv", index=False)
    _save_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        metrics["labels"],
        report_dir / "confusion_matrix.png",
    )
    gradcam_files = _save_gradcam_examples(
        model,
        dataset,
        frame,
        y_pred,
        Path(args.gradcam_dir),
        device,
        max(0, args.gradcam_per_class),
    )
    baseline_path = Path(args.baseline_metrics)
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison = {
            "warning": "Descriptive comparison only; the official test set was not used for model selection.",
            "baseline_name": "three_epoch_pilot",
            "final_name": "weighted_augmented_training",
            "baseline_checkpoint": baseline.get("checkpoint"),
            "final_checkpoint": metrics["checkpoint"],
            "baseline_test_samples": baseline.get("test_samples"),
            "final_test_samples": metrics["test_samples"],
            "baseline_accuracy": baseline.get("accuracy"),
            "final_accuracy": metrics["accuracy"],
            "accuracy_delta": (
                metrics["accuracy"] - baseline["accuracy"] if "accuracy" in baseline else None
            ),
            "baseline_macro_f1": baseline.get("macro_f1"),
            "final_macro_f1": metrics["macro_f1"],
            "macro_f1_delta": (
                metrics["macro_f1"] - baseline["macro_f1"] if "macro_f1" in baseline else None
            ),
            "per_class": {},
        }
        for label in CLASS_TO_IDX:
            baseline_class = baseline.get("classification_report", {}).get(label, {})
            final_class = metrics["classification_report"][label]
            comparison["per_class"][label] = {
                "baseline_precision": baseline_class.get("precision"),
                "final_precision": final_class["precision"],
                "baseline_recall": baseline_class.get("recall"),
                "final_recall": final_class["recall"],
                "recall_delta": (
                    final_class["recall"] - baseline_class["recall"]
                    if "recall" in baseline_class
                    else None
                ),
                "baseline_f1": baseline_class.get("f1-score"),
                "final_f1": final_class["f1-score"],
                "f1_delta": (
                    final_class["f1-score"] - baseline_class["f1-score"]
                    if "f1-score" in baseline_class
                    else None
                ),
            }
        (report_dir / "pilot_vs_final_comparison.json").write_text(
            json.dumps(comparison, indent=2),
            encoding="utf-8",
        )
        class_lines = [
            "| Class | Pilot precision | Final precision | Pilot recall | Final recall | Pilot F1 | Final F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, values in comparison["per_class"].items():
            def format_metric(value):
                return "n/a" if value is None else f"{value:.3f}"

            class_lines.append(
                "| {label} | {pilot_p} | {final_p} | {pilot_r} | {final_r} | {pilot_f1} | {final_f1} |".format(
                    label=label,
                    pilot_p=format_metric(values["baseline_precision"]),
                    final_p=format_metric(values["final_precision"]),
                    pilot_r=format_metric(values["baseline_recall"]),
                    final_r=format_metric(values["final_recall"]),
                    pilot_f1=format_metric(values["baseline_f1"]),
                    final_f1=format_metric(values["final_f1"]),
                )
            )
        comparison_markdown = [
            "# BRISC pilot vs final comparison",
            "",
            comparison["warning"],
            "",
            f"- Pilot accuracy: {comparison['baseline_accuracy']:.3f}",
            f"- Final accuracy: {comparison['final_accuracy']:.3f}",
            f"- Accuracy change: {comparison['accuracy_delta']:+.3f}",
            f"- Pilot macro-F1: {comparison['baseline_macro_f1']:.3f}",
            f"- Final macro-F1: {comparison['final_macro_f1']:.3f}",
            f"- Macro-F1 change: {comparison['macro_f1_delta']:+.3f}",
            f"- Pilot checkpoint: `{comparison['baseline_checkpoint']}`",
            f"- Final checkpoint: `{comparison['final_checkpoint']}`",
            "",
            *class_lines,
            "",
            "The pilot and final experiment use different training procedures and checkpoint criteria. "
            "This descriptive comparison is not a paired significance test. Patient-level independence "
            "is unknown because BRISC provides no verified patient identifiers.",
            "",
        ]
        (report_dir / "pilot_vs_final.md").write_text(
            "\n".join(comparison_markdown),
            encoding="utf-8",
        )
    print(json.dumps({key: metrics[key] for key in ("accuracy", "macro_f1", "test_samples")}, indent=2))
    print(f"Reports: {report_dir.resolve()}")
    print(f"Grad-CAM images saved: {len(gradcam_files)}")


if __name__ == "__main__":
    main()
