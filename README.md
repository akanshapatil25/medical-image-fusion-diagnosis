# Medical Image Fusion Diagnosis

This project has two separate research components: four-class tumor classification using BRISC 2025 T1-weighted MRI, and an independent MRI–CT registration/fusion prototype. The classifier uses MRI only. MRI–CT fusion is not part of BRISC classification and is not clinically validated by this dataset.

## Project goals

- Download and verify the official BRISC 2025 archive from Mendeley Data.
- Validate its class labels, image integrity, dimensions, and duplicate/near-duplicate images.
- Preserve the official test partition; take validation only from the official training partition.
- Train a compact CNN classifier (LiteBrainNet).
- Generate official-test confusion matrix, per-class precision/recall/F1, accuracy, classification report, and Grad-CAM images after training.
- Keep MRI–CT fusion as a separate prototype that requires genuinely paired scans.

## Dataset strategy

The classification dataset is [BRISC 2025, Mendeley Data version 1](https://data.mendeley.com/datasets/c7dj4pwhcv/1), described by its repository as 6,000 T1-weighted brain MRI images in four classes with official train and test partitions. The downloader obtains the archive metadata directly from Mendeley and verifies the downloaded byte count and SHA-256 against the repository record before extraction. `data/splits/brisc/brisc_integrity_report.json` records local inspection results; do not claim those results until the commands have run successfully.

BRISC contains MRI, not paired CT. The classification experiment uses only its official train partition for training/validation and its official test partition for final evaluation. MRI–CT fusion remains separate and requires valid paired studies from the same subject; unrelated MRI/CT scans must never be treated as pairs.

**Verified local data:** the downloaded archive matched Mendeley's 260,175,408-byte size and SHA-256, and all 6,000 classification images were readable with no corrupt or unrecognized files. The observed partition counts were 5,000 train and 1,000 test, but class counts were not balanced as the repository description claims. Duplicate analysis found 46 exact-duplicate groups and 1,304 pHash-near-duplicate pairs (426 cross-partition). Training candidates matching official test images or duplicate train images were excluded; a stratified, connected-component group split now produces 4,121 train, 454 validation, and all 1,000 official test images, with no qualifying pHash edge crossing train/validation. No patient IDs or manifest files were found, so patient-level independence cannot be verified. See the integrity report for counts, dimensions, and removal reasons.

**Completed three-epoch pilot:** on the 1,000-image official test partition, accuracy was **0.621** and macro-F1 was **0.624**; glioma recall was **0.358**. Pilot artifacts are in `outputs/reports/brisc_test` and `outputs/figures/brisc_test_gradcam`.

**Completed corrected weighted/augmented run:** early stopping ended training at epoch 20; the validation macro-F1-selected checkpoint is from epoch 14 (validation macro-F1 **0.895**). On the 1,000 official test images, accuracy was **0.760** and macro-F1 **0.756**. Per-class precision/recall/F1: glioma **0.938/0.657/0.773**, meningioma **0.661/0.556/0.604**, pituitary **0.836/0.950/0.889**, no tumor **0.616/0.986/0.758**. Accuracy and macro-F1 were higher than the pilot in this descriptive comparison, while meningioma recall decreased. Full metrics, confusion matrix, comparison, and 12 Grad-CAM visualizations are under `outputs/reports/brisc_final` and `outputs/figures/brisc_final_gradcam`. Since the pilot test report had already been inspected before this follow-up was designed, the comparison is exploratory—not an independent, confirmatory estimate or a clinical validation.

An initial longer run was stopped at epoch 11 after discovering 128 near-duplicate pHash edges crossed its random train/validation split; it did not produce a checkpoint or use the official test set. Its history is preserved as `outputs/reports/brisc_final/training_history_random_split_aborted.jsonl` and is not a valid final experiment.

## Quick start

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -U pip
   python -m pip install -r requirements.txt
   ```

2. Download, verify, and extract BRISC from its official Mendeley repository:

   ```bash
   python scripts/download_brisc.py --archive data/raw/brisc2025.zip --extract-to data/raw
   ```

   The official archive is large (about 260 MB compressed). No Kaggle credentials are needed. The script checks the archive SHA-256 and ZIP CRC before extraction.

3. Validate BRISC and create manifests. Validation is a stratified group split of official train (about 10% by default); pHash-connected near-duplicate components cannot cross train/validation, and official test remains unchanged:

   ```bash
   python scripts/prepare_brisc.py --root data/raw/brisc2025 --output data/splits/brisc --seed 42
   ```

4. Train only on the generated BRISC train/validation manifests:

   ```bash
   python scripts/train.py --train data/splits/brisc/train.csv --val data/splits/brisc/val.csv --config config.yaml --checkpoint models/checkpoints/brisc_litebrainnet_reproduction.pt --history outputs/reports/brisc_reproduction/training_history.jsonl
   ```

   The command uses the configured 30 epochs (with early stopping), a fixed seed, train-derived class weights, mild image augmentation, learning-rate scheduling, and validation macro-F1 checkpoint selection. It refuses to overwrite an existing checkpoint or history file.

5. After training completes, evaluate exactly once on BRISC's official test partition:

   ```bash
   python scripts/evaluate.py --manifest data/splits/brisc/test.csv --checkpoint models/checkpoints/brisc_litebrainnet_reproduction.pt --config config.yaml --reports outputs/reports/brisc_reproduction --gradcam-dir outputs/figures/brisc_reproduction_gradcam --baseline-metrics outputs/reports/brisc_test/metrics.json
   ```

   The command above is for a fresh reproduction and uses distinct paths. The completed run's actual outputs are already preserved under `outputs/reports/brisc_final` and `outputs/figures/brisc_final_gradcam`. The test set is for final reporting only and must not be used for additional model selection.

6. Synthetic examples are software-only. If needed, create them separately:

   ```bash
   python scripts/generate_demo_dataset.py --root data/demo --per-class 6
   ```

   `data/demo` is explicitly rejected by the training and evaluation pipelines and must not be used for performance claims.

7. Launch the app:

   ```bash
   streamlit run app.py
   ```

   In the sidebar, choose the BRISC classification folder for dataset exploration.

## Repository layout

- `src/` - dataset, preprocessing, registration, fusion, training, evaluation, explainability, and inference utilities.
- `models/` - model definitions and checkpoints.
- `scripts/` - runnable tools for dataset inspection, preparation, training, and evaluation.
- `data/` - raw images, processed outputs, and split manifests.
- `docs/` - dataset guide, methodology, limitations, and demonstration notes.
- `tests/` - focused validation tests.
- `outputs/` - reports, figures, and predictions.

## Dataset guide

See `docs/dataset_guide.md` for the beginner-friendly dataset instructions and download workflow.

## Methodology

See `docs/methodology.md` for the project rationale and component pipeline.

## Limitations

The project is intentionally explicit about what is validated and what is only illustrative. See `docs/limitations.md`.

## License and data use

This repository contains code, not the downloaded BRISC archive. The Mendeley landing page lists CC BY 4.0; read and follow the current dataset terms, cite BRISC and its associated publication, and retain required attribution. Do not upload the archive to Git.
