from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ID = "c7dj4pwhcv"
VERSION = "1"
API_URL = (
    "https://data.mendeley.com/public-api/datasets/"
    f"{DATASET_ID}/files?folder_id=root&version={VERSION}&$start=0&$limit=1000"
)


def fetch_archive_metadata():
    request = urllib.request.Request(
        API_URL,
        headers={
            "Accept": "application/vnd.mendeley-public-dataset.1+json",
            "User-Agent": "Mozilla/5.0 BRISC-reproducible-downloader/1.0",
            "Referer": "https://data.mendeley.com/datasets/c7dj4pwhcv/1",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        files = json.load(response)
    archives = [item for item in files if item.get("filename", "").lower() == "brisc2025.zip"]
    if len(archives) != 1:
        raise RuntimeError(f"Expected one official brisc2025.zip archive, found {len(archives)}")
    archive = archives[0]
    details = archive.get("content_details", {})
    download_url = details.get("download_url", "")
    if not download_url.startswith("https://data.mendeley.com/public-files/"):
        raise RuntimeError("Mendeley returned an unexpected archive download host")
    return {
        "filename": archive["filename"],
        "size": int(details["size"]),
        "sha256": details["sha256_hash"].lower(),
        "url": download_url,
    }


def download_archive(destination: Path, metadata: dict):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    received = 0
    request = urllib.request.Request(
        metadata["url"],
        headers={"User-Agent": "Mozilla/5.0 BRISC-reproducible-downloader/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                print(f"\rDownloaded {received:,} / {metadata['size']:,} bytes", end="", flush=True)
        print()
        if received != metadata["size"]:
            raise RuntimeError(f"Archive size mismatch: expected {metadata['size']}, got {received}")
        actual_hash = digest.hexdigest()
        if actual_hash != metadata["sha256"]:
            raise RuntimeError(
                f"Archive SHA-256 mismatch: expected {metadata['sha256']}, got {actual_hash}"
            )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Verified archive SHA-256: {metadata['sha256']}")


def safe_extract(archive_path: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC check failed for {bad_member}")
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe path in dataset archive: {member.filename}")
            target = (root / Path(*path.parts)).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Archive entry escapes destination: {member.filename}")
        archive.extractall(root)


def main():
    parser = argparse.ArgumentParser(description="Download and verify the official BRISC 2025 archive")
    parser.add_argument("--archive", default="data/raw/brisc2025.zip", help="Archive output path")
    parser.add_argument("--extract-to", default="data/raw", help="Safe extraction directory")
    parser.add_argument("--no-extract", action="store_true", help="Download and verify without extracting")
    args = parser.parse_args()

    archive = Path(args.archive)
    metadata = fetch_archive_metadata()
    if archive.exists():
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if archive.stat().st_size != metadata["size"] or digest.hexdigest() != metadata["sha256"]:
            raise RuntimeError(f"Existing archive failed official size/hash verification: {archive}")
        print(f"Existing archive verified: {archive}")
    else:
        download_archive(archive, metadata)

    if not args.no_extract:
        safe_extract(archive, Path(args.extract_to))
        print(f"Verified and extracted BRISC archive to {args.extract_to}")


if __name__ == "__main__":
    main()
