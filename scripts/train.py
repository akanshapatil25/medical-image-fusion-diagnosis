from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import load_training_config, train_model


def main():
    parser = argparse.ArgumentParser(description="Train the classifier model")
    parser.add_argument("--train", required=True, help="Training manifest CSV file")
    parser.add_argument("--val", required=True, help="Validation manifest CSV file")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration file")
    parser.add_argument("--checkpoint", default="models/checkpoints/litebrainnet.pt", help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count for a short pilot")
    parser.add_argument("--history", help="Write epoch-by-epoch training metrics to a JSONL file")
    args = parser.parse_args()
    config = load_training_config(args.config)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    model, history = train_model(
        args.train,
        args.val,
        config,
        checkpoint_path=args.checkpoint,
        history_path=args.history,
    )
    print(
        f"best_epoch={history['best_epoch']} "
        f"best_val_macro_f1={history['best_val_macro_f1']:.4f}"
    )
    return model, history


if __name__ == "__main__":
    main()
