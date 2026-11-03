from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.brisc import prepare_brisc_manifests


def main():
    parser = argparse.ArgumentParser(
        description="Validate BRISC images and create train-only validation plus official test manifests"
    )
    parser.add_argument("--root", default="data/raw", help="BRISC archive extraction root")
    parser.add_argument("--output", default="data/splits/brisc", help="Output directory for BRISC manifests/report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    report = prepare_brisc_manifests(
        args.root,
        args.output,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    summary = {
        key: report[key]
        for key in (
            "images_verified",
            "official_partition_counts",
            "class_counts_by_official_partition",
            "actual_manifest_counts",
            "dimensions",
            "exact_duplicate_group_count",
            "near_duplicate_pair_count",
            "training_images_removed_for_duplicate_count",
            "patient_metadata",
        )
    }
    print(json.dumps(summary, indent=2))
    print(f"Full integrity report: {Path(args.output) / 'brisc_integrity_report.json'}")


if __name__ == "__main__":
    main()
