from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset import CLASS_TO_IDX, canonical_label, sha256_file
from src.preprocessing import SUPPORTED_EXTENSIONS

CLASSIFICATION_DIR = Path("classification_task")
OFFICIAL_ARCHIVE_SIZE = 260175408
OFFICIAL_ARCHIVE_SHA256 = "ce4f46991476214351b160830963cbadd744d0bfe36449243df7e5c4f7a6f216"
OFFICIAL_DATASET_URL = "https://data.mendeley.com/datasets/c7dj4pwhcv/1"
MANIFEST_COLUMNS = [
    "path",
    "label",
    "width",
    "height",
    "sha256",
    "filename",
    "official_partition",
    "split",
]


def find_classification_root(dataset_root):
    root = Path(dataset_root).expanduser().resolve()
    candidates = [root / CLASSIFICATION_DIR, root]
    for candidate in candidates:
        if all((candidate / split / label).is_dir() for split in ("train", "test") for label in CLASS_TO_IDX):
            return candidate
    raise FileNotFoundError(
        f"Could not find BRISC classification_task/train and test class folders under {root}"
    )


def scan_brisc_images(dataset_root):
    classification_root = find_classification_root(dataset_root)
    records = []
    corrupt = []
    unrecognized = []
    for partition in ("train", "test"):
        partition_dir = classification_root / partition
        for path in sorted(partition_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            label = canonical_label(path.parent.name)
            if label is None or path.parent.parent != partition_dir:
                unrecognized.append(str(path))
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                    image.load()
                records.append(
                    {
                        "path": str(path.resolve()),
                        "label": label,
                        "width": int(width),
                        "height": int(height),
                        "sha256": sha256_file(path),
                        "filename": path.name,
                        "official_partition": partition,
                    }
                )
            except Exception as exc:
                corrupt.append({"path": str(path), "error": str(exc)})
    return classification_root, records, {"corrupt": corrupt, "unrecognized": unrecognized}


def _perceptual_hash(path):
    with Image.open(path) as source:
        gray = np.asarray(source.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    dct = cv2.dct(gray.astype(np.float32))
    low = dct[:8, :8].reshape(-1)[1:]
    median = float(np.median(low))
    value = 0
    for bit, coefficient in enumerate(low):
        if coefficient > median:
            value |= 1 << bit
    return value


def find_duplicate_images(records, near_distance=4):
    exact = defaultdict(list)
    for row in records:
        exact[row["sha256"]].append(row)
    exact_groups = [
        {
            "sha256": digest,
            "paths": [row["path"] for row in group],
            "partitions": sorted({row["official_partition"] for row in group}),
        }
        for digest, group in exact.items()
        if len(group) > 1
    ]

    hashes = [_perceptual_hash(row["path"]) for row in records]
    path_to_index = {row["path"]: index for index, row in enumerate(records)}
    # Five disjoint bands ensure every pair within Hamming distance four shares a band.
    bands = [(0, 13), (13, 13), (26, 13), (39, 12), (51, 12)]
    buckets = defaultdict(list)
    candidates = set()
    for index, value in enumerate(hashes):
        for offset, width in bands:
            key = (offset, width, (value >> offset) & ((1 << width) - 1))
            for other in buckets[key]:
                candidates.add((other, index))
            buckets[key].append(index)

    exact_pairs = {
        tuple(sorted((entry["paths"][i], entry["paths"][j])))
        for entry in exact_groups
        for i in range(len(entry["paths"]))
        for j in range(i + 1, len(entry["paths"]))
    }
    near_pairs = []
    near_parent = list(range(len(records)))

    def find_root(index):
        while near_parent[index] != index:
            near_parent[index] = near_parent[near_parent[index]]
            index = near_parent[index]
        return index

    def union(first, second):
        first_root = find_root(first)
        second_root = find_root(second)
        if first_root != second_root:
            near_parent[second_root] = first_root

    for first, second in sorted(candidates):
        distance = (hashes[first] ^ hashes[second]).bit_count()
        if distance <= near_distance and tuple(sorted((records[first]["path"], records[second]["path"]))) not in exact_pairs:
            union(first, second)
            near_pairs.append(
                {
                    "path_a": records[first]["path"],
                    "path_b": records[second]["path"],
                    "hamming_distance": distance,
                    "cross_partition": records[first]["official_partition"]
                    != records[second]["official_partition"],
                }
            )
    cluster_members = defaultdict(set)
    for pair in near_pairs:
        first_index = path_to_index[pair["path_a"]]
        second_index = path_to_index[pair["path_b"]]
        cluster_members[find_root(first_index)].update((first_index, second_index))
    clusters = []
    for members in cluster_members.values():
        member_rows = [records[index] for index in sorted(members)]
        partitions = sorted({row["official_partition"] for row in member_rows})
        clusters.append(
            {
                "member_count": len(member_rows),
                "partitions": partitions,
                "cross_partition": len(partitions) > 1,
                "paths": [row["path"] for row in member_rows],
            }
        )
    clusters.sort(key=lambda group: (-group["member_count"], group["paths"][0]))
    return exact_groups, near_pairs, clusters


def _patient_metadata(classification_root, dataset_root, records):
    column_names = set()
    manifest_paths = list(dict.fromkeys([
        classification_root.parent / "manifest.csv",
        Path(dataset_root).expanduser().resolve() / "manifest.csv",
        classification_root / "manifest.csv",
    ]))
    for manifest in manifest_paths:
        if manifest.is_file():
            try:
                columns = pd.read_csv(manifest, nrows=0).columns
                column_names.update(str(column).strip().lower() for column in columns)
            except Exception:
                continue
    patient_columns = sorted(
        column
        for column in column_names
        if re.search(r"(patient|subject|case|participant).*(id|identifier)|^(patient|subject|case)_?id$", column)
    )
    filename_identifiers = sorted(
        {
            re.search(r"(?:patient|subject|case|participant)[_-]?([a-z0-9]+)", row["filename"], re.I).group(0)
            for row in records
            if re.search(r"(?:patient|subject|case|participant)[_-]?[a-z0-9]+", row["filename"], re.I)
        }
    )
    return {
        "manifest_files_checked": [str(path) for path in manifest_paths],
        "manifest_files_found": [str(path) for path in manifest_paths if path.is_file()],
        "patient_identifier_columns": patient_columns,
        "filename_patient_id_patterns_detected": bool(filename_identifiers),
        "patient_level_identifiers_available": bool(patient_columns or filename_identifiers),
        "note": (
            "Patient/subject-like identifiers were found in available metadata or filenames; verify their semantics with the dataset authors."
            if patient_columns or filename_identifiers
            else "No patient/subject identifier column was found in the available manifest files. "
            "Filename sequence numbers are not assumed to be patient IDs; patient-level independence "
            "cannot be verified from these files alone."
        ),
    }


def _archive_integrity(dataset_root):
    root = Path(dataset_root).expanduser().resolve()
    candidates = [root / "brisc2025.zip", root.parent / "brisc2025.zip"]
    archive = next((path for path in candidates if path.is_file()), None)
    if archive is None:
        return {
            "archive_path": None,
            "archive_size_bytes": None,
            "archive_sha256": None,
            "matches_official_mendeley_metadata": False,
        }
    digest = sha256_file(archive)
    size = archive.stat().st_size
    return {
        "archive_path": str(archive),
        "archive_size_bytes": size,
        "archive_sha256": digest,
        "matches_official_mendeley_metadata": (
            size == OFFICIAL_ARCHIVE_SIZE and digest == OFFICIAL_ARCHIVE_SHA256
        ),
    }


def prepare_brisc_manifests(dataset_root, output_dir, seed=42, validation_fraction=0.1):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    classification_root, records, issues = scan_brisc_images(dataset_root)
    expected_partitions = {"train", "test"}
    present_partitions = {row["official_partition"] for row in records}
    if present_partitions != expected_partitions:
        raise ValueError(f"Expected both official partitions, found {sorted(present_partitions)}")
    if issues["corrupt"] or issues["unrecognized"]:
        raise ValueError(
            f"Dataset contains {len(issues['corrupt'])} corrupt and "
            f"{len(issues['unrecognized'])} unrecognized images; inspect before training"
        )
    for partition in ("train", "test"):
        labels = {row["label"] for row in records if row["official_partition"] == partition}
        if labels != set(CLASS_TO_IDX):
            raise ValueError(f"BRISC {partition} partition class mismatch: found {sorted(labels)}")

    official_train = [row for row in records if row["official_partition"] == "train"]
    official_test = [row for row in records if row["official_partition"] == "test"]
    exact_duplicates, near_duplicates, near_clusters = find_duplicate_images(records)
    test_hashes = {row["sha256"] for row in official_test}
    test_paths = {row["path"] for row in official_test}
    test_duplicate_hashes = {
        digest
        for digest, group in (
            (group["sha256"], group) for group in exact_duplicates
        )
        if test_paths.intersection(group["paths"])
    }
    test_near_paths = {
        pair["path_a"] if pair["path_a"] not in test_paths else pair["path_b"]
        for pair in near_duplicates
        if pair["cross_partition"]
    }
    dropped_train = []
    seen_train_hashes = set()
    filtered_train = []
    for row in official_train:
        reason = None
        if row["sha256"] in test_hashes or row["sha256"] in test_duplicate_hashes:
            reason = "exact_duplicate_of_official_test"
        elif row["path"] in test_near_paths:
            reason = "near_duplicate_of_official_test_phash_distance_le_4"
        elif row["sha256"] in seen_train_hashes:
            reason = "duplicate_within_official_train"
        if reason:
            dropped_train.append({"path": row["path"], "reason": reason})
            continue
        seen_train_hashes.add(row["sha256"])
        filtered_train.append(row)
    official_train = filtered_train
    if not official_train:
        raise ValueError("No official training images remain after duplicate filtering")
    train_paths = {row["path"] for row in official_train}
    group_parent = list(range(len(official_train)))
    train_path_to_index = {row["path"]: index for index, row in enumerate(official_train)}

    def find_group(index):
        while group_parent[index] != index:
            group_parent[index] = group_parent[group_parent[index]]
            index = group_parent[index]
        return index

    def join_groups(first, second):
        first_root = find_group(first)
        second_root = find_group(second)
        if first_root != second_root:
            group_parent[second_root] = first_root

    for pair in near_duplicates:
        if pair["path_a"] in train_paths and pair["path_b"] in train_paths:
            join_groups(train_path_to_index[pair["path_a"]], train_path_to_index[pair["path_b"]])
    near_groups = np.asarray([find_group(index) for index in range(len(official_train))])
    n_splits = int(round(1 / validation_fraction))
    if n_splits < 2 or n_splits > len(set(near_groups.tolist())):
        raise ValueError(
            "validation_fraction cannot be represented by group-aware stratified folds "
            "for the available near-duplicate components"
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_indices, val_indices = next(
        splitter.split(
            np.zeros(len(official_train)),
            [row["label"] for row in official_train],
            near_groups,
        )
    )
    train_rows = [official_train[index] for index in train_indices]
    val_rows = [official_train[index] for index in val_indices]
    train_rows = [dict(row, split="train") for row in train_rows]
    val_rows = [dict(row, split="val") for row in val_rows]
    test_rows = [dict(row, split="test") for row in official_test]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(output / f"{name}.csv", index=False)

    label_counts = {
        partition: dict(Counter(row["label"] for row in records if row["official_partition"] == partition))
        for partition in ("train", "test")
    }
    dimensions = Counter((row["width"], row["height"]) for row in records)
    class_counts_by_manifest = {
        name: dict(Counter(row["label"] for row in rows))
        for name, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows))
    }
    train_manifest_paths = {row["path"] for row in train_rows}
    val_manifest_paths = {row["path"] for row in val_rows}
    near_duplicate_edges_crossing_validation = sum(
        (pair["path_a"] in train_manifest_paths and pair["path_b"] in val_manifest_paths)
        or (pair["path_b"] in train_manifest_paths and pair["path_a"] in val_manifest_paths)
        for pair in near_duplicates
    )
    archive_integrity = _archive_integrity(dataset_root)
    report = {
        "dataset": "BRISC 2025",
        "source": {
            "url": OFFICIAL_DATASET_URL,
            "doi": "10.17632/c7dj4pwhcv.1",
            "archive_filename": "brisc2025.zip",
            "source_reported_total": 6000,
            "source_reported_partition_counts": {"train": 5000, "test": 1000},
            "source_description_claims_balanced_classes": True,
            "archive_integrity": archive_integrity,
        },
        "classification_root": str(classification_root),
        "images_verified": len(records),
        "official_partition_counts": {
            partition: sum(row["official_partition"] == partition for row in records)
            for partition in ("train", "test")
        },
        "class_counts_by_official_partition": label_counts,
        "class_counts_by_generated_manifest": class_counts_by_manifest,
        "near_duplicate_edges_crossing_train_validation": near_duplicate_edges_crossing_validation,
        "train_validation_split_policy": (
            f"StratifiedGroupKFold with {n_splits} folds over connected components of pHash-near-duplicate "
            "edges among eligible official-training images; near-duplicate components are not split across train/validation."
        ),
        "class_distribution_matches_balanced_source_claim": all(
            len(set(counts.values())) == 1 for counts in label_counts.values()
        ),
        "actual_manifest_counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "training_images_removed_for_duplicates": dropped_train,
        "training_images_removed_for_duplicate_count": len(dropped_train),
        "dimensions": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in sorted(dimensions.items())
        ],
        "corrupt_files": issues["corrupt"],
        "unrecognized_files": issues["unrecognized"],
        "exact_duplicate_groups": exact_duplicates,
        "exact_duplicate_group_count": len(exact_duplicates),
        "exact_duplicate_groups_crossing_official_partitions": sum(
            len(group["partitions"]) > 1 for group in exact_duplicates
        ),
        "near_duplicate_pairs_phash_hamming_le_4": near_duplicates,
        "near_duplicate_pair_count": len(near_duplicates),
        "near_duplicate_clusters_phash_hamming_le_4": near_clusters,
        "near_duplicate_cluster_count": len(near_clusters),
        "near_duplicate_clusters_crossing_official_partitions": sum(
            group["cross_partition"] for group in near_clusters
        ),
        "cross_partition_exact_duplicates": [
            group for group in exact_duplicates if len(group["partitions"]) > 1
        ],
        "cross_partition_near_duplicates": [
            pair for pair in near_duplicates if pair["cross_partition"]
        ],
        "patient_metadata": _patient_metadata(classification_root, dataset_root, records),
        "test_partition_policy": (
            "The official test partition is written unchanged to test.csv. Validation is sampled "
            "exclusively from official train. No test image is used for training or validation."
        ),
    }
    report_path = output / "brisc_integrity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
