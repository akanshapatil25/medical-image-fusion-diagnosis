# Implementation reference

## BRISC classification workflow

1. Download, size/hash-check, CRC-check, and extract the official archive:

   ```powershell
   python scripts\download_brisc.py --archive data\raw\brisc2025.zip --extract-to data\raw
   ```

2. Verify images and generate manifests:

   ```powershell
   python scripts\prepare_brisc.py --root data\raw\brisc2025 --output data\splits\brisc
   ```

3. Train from the official training partition only:

   ```powershell
   python scripts\train.py --train data\splits\brisc\train.csv --val data\splits\brisc\val.csv --config config.yaml --checkpoint models\checkpoints\brisc_litebrainnet.pt
   ```

4. Evaluate the unchanged official test partition:

   ```powershell
   python scripts\evaluate.py --manifest data\splits\brisc\test.csv --checkpoint models\checkpoints\brisc_litebrainnet.pt --config config.yaml
   ```

The trainer rejects official-test partition rows, paths containing a test directory, synthetic `data/demo` paths, overlapping train/validation paths, and shared exact hashes. The evaluator accepts only a manifest marked as the official BRISC test partition.

## Important modules

- `src/brisc.py`: BRISC layout discovery, file verification, duplicate analysis, train-only validation splitting, and integrity report generation.
- `scripts/download_brisc.py`: official Mendeley metadata lookup, archive download, checksum verification, and safe extraction.
- `src/dataset.py`: image loading and classification dataset.
- `src/training.py`: LiteBrainNet training and validation.
- `scripts/evaluate.py`: official-test metrics, confusion matrix, and Grad-CAM artifacts.
- `src/registration.py` and `src/fusion.py`: separate MRI–CT research prototype; not validated by BRISC.

## Generated evidence

- Dataset integrity/split report: `data/splits/brisc/brisc_integrity_report.json`
- Evaluation report: `outputs/reports/brisc_test/metrics.json`
- Per-class precision/recall/F1: `outputs/reports/brisc_test/per_class_metrics.csv`
- Confusion matrix: `outputs/reports/brisc_test/confusion_matrix.csv` and `.png`
- Grad-CAM figures: `outputs/figures/brisc_test_gradcam/`

Only artifacts created from a successfully completed run should be presented as results. The synthetic `data/demo` set is not valid for training, evaluation, or medical claims.
