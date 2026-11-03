from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import create_split_manifests, discover_images, unique_records


def main():
    parser = argparse.ArgumentParser(description="Create deterministic train/validation/test CSV manifests")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--output", default="data/splits", help="Directory to save split CSVs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducible splitting")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    if (root / "classification_task").is_dir() or next(root.glob("**/classification_task"), None):
        parser.error(
            "BRISC has an official test split; use scripts/prepare_brisc.py so test images remain untouched"
        )
    rows, issues = discover_images(args.root)
    if issues["unlabeled"] or issues["corrupt"]:
        print(f"Found {len(issues['unlabeled'])} unlabeled files and {len(issues['corrupt'])} corrupt files.")
    records = unique_records(rows)
    output_dir = Path(args.output)
    partitions = create_split_manifests(records, output_dir, seed=args.seed)
    print({name: len(rows) for name, rows in partitions.items()})


if __name__ == "__main__":
    main()
