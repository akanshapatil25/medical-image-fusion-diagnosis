# Methodology

## Separate research components

This project does **not** make a single MRI–CT fusion-to-diagnosis claim. It has two independent tracks:

1. **MRI tumor classification:** LiteBrainNet is trained and evaluated with the BRISC 2025 T1-weighted brain MRI classification images.
2. **MRI–CT fusion research prototype:** translation registration and weighted fusion utilities are available separately. They require a genuinely paired MRI and CT from the same subject and are not trained or validated using BRISC.

The two tracks must not be joined using unrelated images. The synthetic images under `data/demo` are for demonstrating software functions only; training and evaluation reject them.

## Dataset and split policy

The source is [BRISC 2025, Mendeley Data, version 1](https://data.mendeley.com/datasets/c7dj4pwhcv/1). The repository describes 6,000 T1-weighted MRI slices, four classes (glioma, meningioma, pituitary, no tumor), and official partitions of 5,000 training and 1,000 test images. These are source-reported counts; the local `brisc_integrity_report.json` must be generated to establish actual downloaded file counts, class counts, dimensions, and integrity.

`scripts/download_brisc.py` queries Mendeley's public file metadata, downloads the official ZIP, checks byte count and SHA-256, verifies ZIP CRC, and safely extracts the files. `scripts/prepare_brisc.py` validates every classification image and writes:

- `train.csv`: training subset of official train
- `val.csv`: stratified-group validation subset of official train (about 10% by default)
- `test.csv`: all verified official test images, unchanged

Exact and pHash near-duplicates are reported. Training examples that duplicate or are near-duplicates of official test images are excluded from train/validation manifests; duplicate SHA-256 copies within official train are reduced to one. The remaining training data is split with `StratifiedGroupKFold`, grouping connected components of qualifying pHash edges, so a near-duplicate edge cannot cross train and validation. The integrity report records counts, exclusions, grouping policy, and a cross-split edge count that must be zero. The official test manifest is retained unchanged.

The available repository description/sample filename pattern does not establish a patient identifier. The inspection checks for patient/subject-like columns in available manifests and identifiers in filenames. Sequential image numbers must not be interpreted as patient IDs. If no verified subject ID is provided, subject-level independence and patient leakage cannot be established; duplicate-image checks are not a substitute.

### Local verification observed

The downloaded archive matched Mendeley's recorded 260,175,408-byte size and SHA-256. All 6,000 classification images were readable; no corrupt or unrecognized classification files were found. Observed class counts differ from the repository's claim of balanced classes:

| Official partition | Glioma | Meningioma | Pituitary | No tumor |
|---|---:|---:|---:|---:|
| Train | 1,147 | 1,329 | 1,457 | 1,067 |
| Test | 254 | 306 | 300 | 140 |

Image dimensions are not uniform: 4,881 are 512×512 and the remainder have varied sizes; the classifier resizes to 224×224. Duplicate analysis found 46 exact-duplicate groups and 1,304 pHash-near-duplicate pairs, including 7 exact groups and 426 near-duplicate pairs spanning official train/test. pHash is heuristic, so these are flagged as candidate duplicates, not patient identity. Training examples matching the official test or duplicate training images were excluded, resulting in 4,121 train and 454 validation images; all 1,000 official test images remain in test. The stratified group split had zero qualifying pHash edges crossing train/validation. No manifest file or patient/subject ID was found in the extracted archive.

Five exact-duplicate groups were confined to the official test partition and were retained to preserve that partition. Consequently, reported evaluation is image-level, may overweight repeated images, and cannot be interpreted as patient-independent performance.

## Preprocessing

The image pipeline converts MRI to grayscale, scales intensities to `[0, 1]`, and resizes to the configured 224×224 input. Training augmentation applies small flips/rotations. Evaluation uses no augmentation.

## Classification

LiteBrainNet is a compact four-class CNN. Cross-entropy and AdamW are used. The initial pilot selected its checkpoint by validation loss; the improved experiment uses validation macro-F1 with validation loss as a tie-breaker, and both use early stopping. The official test partition is not provided to the training routine and is used only by the final evaluation command after training completes.

## Evaluation and explainability

`scripts/evaluate.py` accepts only a manifest marked as the official BRISC test partition. It saves:

- overall accuracy and macro-F1
- confusion matrix as CSV and PNG
- per-class precision, recall, and F1/support as CSV
- full scikit-learn classification report as JSON
- Grad-CAM overlays for selected test samples, with true and predicted labels

Grad-CAM illustrates model activation and is not a clinical explanation. In the inspected final examples, some activation extended to image edges/background as well as intracranial regions; this is a qualitative caution, not evidence of a specific causal failure mode.

### Completed pilot

A three-epoch CPU pilot completed with the best checkpoint from epoch 3 (validation loss 0.7285). Evaluation on all 1,000 official test images produced 0.621 accuracy and 0.624 macro-F1. Per-class precision/recall/F1 were: glioma 0.858/0.358/0.506; meningioma 0.443/0.605/0.511; pituitary 0.832/0.757/0.792; no tumor 0.581/0.843/0.688. In particular, glioma recall is weak. These are pilot results from a short run, not a final benchmark or clinical performance claim.

The pilot confusion matrix shows 149 of 254 glioma test images predicted as meningioma and 14 as pituitary; 91 gliomas were correctly classified. Glioma recall was therefore weak despite 972 glioma images in the training manifest, and not explained simply by glioma being the smallest training class. In several inspected pilot Grad-CAM overlays, activation extended into image borders/background rather than being confined to intracranial tissue. This is a qualitative warning about possible framing/background reliance, not a proven causal diagnosis. Pilot epoch logs were not saved, so only checkpoint metadata and the persisted validation/test reports are available for this audit.

### Improved experiment policy

The follow-up run uses class weights computed only from the post-filter training manifest, modest training-only horizontal flips, small rotations, brightness/contrast changes, a validation-loss `ReduceLROnPlateau` scheduler, and early stopping. The best checkpoint is selected by validation macro-F1 (validation loss breaks ties). The official test manifest is not passed to training and is not used for checkpoint selection; however, the prior pilot test report had been reviewed before this follow-up was designed, so test-informed development cannot be ruled out. Epoch-by-epoch training/validation metrics are written separately.

Exact-duplicate groups use SHA-256 equality (identical file bytes). Near-duplicate pairs use a 63-bit DCT perceptual hash after grayscale resizing to 32×32, with Hamming distance ≤4. Five disjoint bit bands generate candidate pairs; accepted pairs are grouped into connected components for inspection. Edges/group sizes overlap and are heuristic, not subject/patient groupings. Before training, any official-training image with an exact or qualifying near-duplicate link to any official test image is excluded; duplicate SHA-256 copies within official train are reduced to one. Here that removed 385 distinct training images by pHash test-neighbor match, 9 exact train files matching official test, and 31 duplicate files within official train. The 1,304 near-duplicate edges formed 686 connected components, 292 of which contain both official partitions; a component is transitive and does not imply every pair within it passes the distance threshold. All official test files remain untouched in the evaluation manifest. The five exact-duplicate groups wholly inside test are preserved and disclosed, not used for model selection.

After test-overlap and exact-duplicate exclusions, there are 4,575 eligible official-training images. The group-aware split produced training counts of glioma 980, meningioma 1,039, pituitary 1,151, and no tumor 951; validation counts are 100, 116, 132, and 106 respectively. These training counts determine class weights. No balancing/resampling is applied to validation or test. Compared with the superseded random split, the train/validation counts differ because validation assignments are made at near-duplicate-component level, not image level.

An initial 30-epoch follow-up attempt used a random split and was stopped at epoch 11 after an audit found 128 qualifying pHash edges crossing train/validation. Its partial epoch log is retained as `outputs/reports/brisc_final/training_history_random_split_aborted.jsonl`; it yielded no checkpoint and the official test set was not accessed. Its validation metrics are excluded from model-selection or performance claims.

### Completed corrected experiment

The corrected group-aware run used seed 42, configured maximum 30 epochs, class-weighted cross-entropy with label smoothing, training-only augmentation, and validation-loss learning-rate scheduling. Early stopping terminated at epoch 20; epoch 14 was selected by validation macro-F1 (0.8954). The saved checkpoint is `models/checkpoints/brisc_litebrainnet_weighted_30ep.pt`. It was evaluated once on all 1,000 images in the official test manifest:

| Metric | Pilot | Corrected run |
|---|---:|---:|
| Accuracy | 0.621 | 0.760 |
| Macro-F1 | 0.624 | 0.756 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 0.938 | 0.657 | 0.773 | 254 |
| Meningioma | 0.661 | 0.556 | 0.604 | 306 |
| Pituitary | 0.836 | 0.950 | 0.889 | 300 |
| No tumor | 0.616 | 0.986 | 0.758 | 140 |

The final confusion matrix (rows are true labels, columns predicted; order glioma, meningioma, pituitary, no tumor) is `[[167, 76, 11, 0], [5, 170, 45, 86], [5, 10, 285, 0], [1, 1, 0, 138]]`. Compared with the pilot, glioma recall increased from 0.358 to 0.657, while meningioma recall declined from 0.605 to 0.556. Results are descriptive, not a paired significance test. Because the pilot test report—including glioma recall—was reviewed before the follow-up was designed, test-informed development cannot be ruled out; these results should be treated as exploratory and confirmed on a genuinely independent cohort. No clinical-validation claim is made. Full-precision metrics and the classification report are in `outputs/reports/brisc_final/metrics.json`; the CSV/PNG confusion matrix, per-class metrics, pilot comparison, predictions, plane recall, and training history are in the same folder. Twelve Grad-CAM overlays are in `outputs/figures/brisc_final_gradcam`.

## MRI–CT fusion

`src/registration.py` estimates translation with phase correlation; `src/fusion.py` performs normalized weighted-average fusion. Neither operation creates a valid pair when images come from different subjects or studies. No BRISC result validates the fusion component. A paired, appropriately licensed MRI–CT dataset and modality-specific evaluation are required for that research track.
