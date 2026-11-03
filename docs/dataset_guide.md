# BRISC 2025 dataset guide

## Official repository

- Dataset: BRISC 2025, version 1
- Mendeley Data: <https://data.mendeley.com/datasets/c7dj4pwhcv/1>
- DOI: `10.17632/c7dj4pwhcv.1`
- Modality: T1-weighted brain MRI images (not CT)
- Classes: glioma, meningioma, pituitary, no tumor
- Source-reported partitions: 5,000 train and 1,000 test images
- License shown on the Mendeley landing page: CC BY 4.0; check the current terms and attribution requirements before use.

The official page's Steps to reproduce prose contains text referring to chest/abdominal scans despite the dataset title and description identifying brain MRI. This project treats the repository's files, labels, and local checks as authoritative for implementation, and this inconsistency should be disclosed when presenting the source.

## Download and verify

From the project root, run:

```powershell
python scripts\download_brisc.py --archive data\raw\brisc2025.zip --extract-to data\raw
```

The downloader uses Mendeley's public file metadata; no Kaggle account or private API key is required. It verifies the official archive size and SHA-256 and tests the ZIP CRC before safe extraction. The archive is approximately 260 MB compressed. Keep the downloaded data local and do not commit it.

Then inspect the dataset and prepare manifests:

```powershell
python scripts\prepare_brisc.py --root data\raw --output data\splits\brisc --seed 42
```

Expected layout inside the archive:

```text
brisc2025/
  classification_task/
    train/
      glioma/
      meningioma/
      pituitary/
      no_tumor/
    test/
      glioma/
      meningioma/
      pituitary/
      no_tumor/
  segmentation_task/
  manifest.csv
  manifest.json
```

The script searches recursively for `classification_task`. It ignores the segmentation-task copies for classifier manifests.

## Split safety

`train.csv` and `val.csv` are derived only from `classification_task/train`. The default validation fraction is about 10%; stratified group splitting keeps connected components of qualifying pHash near-duplicate edges on one side. `test.csv` contains the official test images and is not resampled or moved. Training checks reject official-test paths/partitions, overlapping train/validation paths or hashes, and any path under `data/demo`.

The integrity report is saved to `data/splits/brisc/brisc_integrity_report.json`. It records the actual local counts, per-class distribution, dimensions, corrupt/unrecognized files, exact SHA-256 duplicates, pHash near-duplicate pairs, exclusions from training, and patient-ID investigation. Do not copy source-reported counts into a presentation as verified local counts until this report exists and has been reviewed.

## Patient identifiers and limitations

The code checks available manifest columns and filenames for patient/subject identifiers. Sequential sample numbers are not treated as patient IDs. If no verified patient identifier exists, it is impossible to guarantee patient-level separation, even when official train/test splits and image-level duplicate checks are respected. Report this limitation; do not claim independent patient-level testing.

The pHash check is a heuristic for near-duplicate images, not proof of distinct or shared patients. Cross-partition matches are documented, and corresponding official-training images are removed from train/validation while all official-test images remain in the test manifest.

## Train and evaluate

```powershell
python scripts\train.py --train data\splits\brisc\train.csv --val data\splits\brisc\val.csv --config config.yaml --checkpoint models\checkpoints\brisc_litebrainnet.pt
python scripts\evaluate.py --manifest data\splits\brisc\test.csv --checkpoint models\checkpoints\brisc_litebrainnet.pt --config config.yaml
```

Only claim performance from the evaluation report produced after a real training run and only for this dataset/split.

## MRI–CT fusion

BRISC contains MRI, not paired MRI and CT. Do not use unrelated MRI and CT scans as fusion inputs. The fusion prototype is a separate research component and needs genuinely paired scans from the same subject to support multimodal claims.
