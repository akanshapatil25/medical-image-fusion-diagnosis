# Limitations and validation status

## Dataset-specific limitations

- BRISC 2025 provides T1-weighted MRI classification images, not paired MRI–CT studies.
- The Mendeley source reports 6,000 images (5,000 train / 1,000 test); local counts, labels, dimensions, and integrity are not verified until `scripts/prepare_brisc.py` succeeds and its JSON report is inspected.
- In the downloaded version checked here, there were 6,000 readable images and no corrupt/unrecognized classification files, but the observed class counts (train: 1,147/1,329/1,457/1,067; test: 254/306/300/140) are not balanced despite the repository description saying balanced.
- The repository page's reproduction text refers to chest/abdominal scans even though its title/description and inspected images are brain MRI. This source-metadata inconsistency should be acknowledged.
- Duplicate screening found 46 exact-hash groups and 1,304 pHash-near-duplicate pairs; 426 near-duplicate pairs spanned the official train/test boundary. pHash is heuristic. The generated training manifests exclude candidates matching test images and duplicate training samples, while retaining every official test image.
- A prior random training/validation split had 128 near-duplicate edges crossing the split and was abandoned before test evaluation. The replacement uses stratified grouping over connected components of eligible official-training pHash edges and verifies zero such edges cross train/validation. This image-level heuristic still cannot guarantee subject-level independence.
- Five exact-duplicate groups were found wholly within the official test partition and were retained to preserve it, so image-level test metrics may overweight repeated images.
- Patient-level identifiers may not be provided. When unavailable, patient-level independence cannot be verified; sequential image numbers are not assumed to be subject IDs.
- Exact hashes detect byte-identical copies. Perceptual hashing flags likely near-duplicates but cannot establish patient identity and may produce false matches.
- Inspected final Grad-CAM overlays include some edge/background activation. These visualizations are qualitative and cannot establish why the model predicts a class or whether it localizes pathology.
- This is a research/educational classifier, not a medical device or clinical decision-support system.

## Separation of claims

- Classification is MRI-only and must be reported using the untouched official BRISC test partition.
- MRI–CT fusion is a separate, unvalidated prototype. It requires genuinely paired scans from the same subject; unrelated scans must not be fused or represented as a multimodal patient study.
- Synthetic data in `data/demo` exists only to demonstrate software behavior. Training and evaluation explicitly reject demo paths. It must not be included in metrics, model checkpoints claimed as trained on BRISC, or medical performance statements.

## Training and evaluation status

The three-epoch pilot achieved 0.621 accuracy and 0.624 macro-F1 on 1,000 official test images; glioma recall was 0.358. The corrected weighted/augmented group-aware run stopped early at epoch 20 and selected epoch 14 by validation macro-F1. It achieved 0.760 accuracy and 0.756 macro-F1; glioma recall was 0.657, while meningioma recall was 0.556. These are actual image-level results, not clinical validation.

The official test partition had already been evaluated for the pilot, and its report was reviewed before the follow-up experiment was designed. Although the corrected model's checkpoint selection used only training-derived validation, test-informed development cannot be ruled out. Therefore the pilot/final comparison is exploratory and should not be treated as an independent confirmatory estimate; evaluate future model versions only on a new external cohort. The corrected checkpoint was evaluated once after training. The abandoned random-split attempt stopped at epoch 11 after 128 near-duplicate edges were found across train/validation; its partial history is preserved, and it has no checkpoint or test result.
