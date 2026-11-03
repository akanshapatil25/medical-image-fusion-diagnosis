from __future__ import annotations

from pathlib import Path

from src.training import load_training_config, train_model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the MRI classifier")
    parser.add_argument("--train", required=True, help="Train split manifest CSV")
    parser.add_argument("--val", required=True, help="Validation split manifest CSV")
    parser.add_argument("--config", default="config.yaml", help="Project YAML config path")
    parser.add_argument("--checkpoint", default="models/checkpoints/litebrainnet.pt", help="Checkpoint output path")
    args = parser.parse_args()
    config = load_training_config(args.config)
    model, history = train_model(
        args.train,
        args.val,
        config,
        checkpoint_path=args.checkpoint,
    )
    print(f"Training complete. Best epoch: {history['best_epoch']}")
    return model, history


if __name__ == "__main__":
    main()
