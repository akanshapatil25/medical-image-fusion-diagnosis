from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import discover_images, unique_records


def main():
    parser = argparse.ArgumentParser(description="Inspect a dataset and save a JSON summary")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--report", default="outputs/reports/dataset_inspection.json", help="Report output path")
    parser.add_argument("--max-samples", type=int, default=8, help="Maximum representative sample paths to include")
    args = parser.parse_args()
    rows, issues = discover_images(args.root)
    clean = unique_records(rows)
    summary = {
        "total_files": len(clean),
        "class_counts": {label: sum(1 for row in clean if row["label"] == label) for label in sorted({row["label"] for row in clean})},
        "unlabeled": issues["unlabeled"][:args.max_samples],
        "corrupt": [entry["path"] for entry in issues["corrupt"][:args.max_samples]],
        "sample_paths": [row["path"] for row in clean[:args.max_samples]],
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
