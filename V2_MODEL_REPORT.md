# DermoAI V2 Model Report

**Report scope:** local workspace evidence for the refined DermoAI V2 checkpoint selected from run `20261016-152650-e30bec65`.

**Intended use:** research and engineering evaluation only. DermoAI V2 is **not a medical device, diagnostic system, screening test, or substitute for examination by a qualified clinician**.

## 1. Executive summary

DermoAI V2 is a 12-class skin-image classifier built from a pretrained ConvNeXt V2-Base model and partially fine-tuned at 224×224 resolution. The selected artifact is `outputs/dermoai/runtime/20261016-152650-e30bec65/epoch-0002.pt`, SHA-256 `9d0cb9f692397b81ac5527cda06d9d9502056e6edead6440060150b8ce316c21`. It was selected at epoch 2 using validation data only and a composite `balanced_cancer_safety` score; the test split was not used for checkpoint selection (`outputs/dermoai/runtime/20261016-152650-e30bec65/final-selection.json:2`).

On the 4,336-image frozen test split, V2 achieved:

- **Accuracy:** 79.96%
- **Macro-F1:** 75.25%
- **Cancer sensitivity:** 77.71%
- **Cancer specificity:** 91.71%
- **Smartphone subset accuracy:** 72.15% on 316 images

These figures are recorded in `outputs/dermoai/runtime/20261016-152650-e30bec65/final-test/metrics.json:3` and `outputs/dermoai/runtime/20261016-152650-e30bec65/final-test/metrics.json:264`.

Against the original repository's ConvNeXt-Base `ce_ls_best.pth` on the same local split and top-1 rule, refined V2 improved accuracy by 3.37 percentage points, macro-F1 by 1.98 points, cancer sensitivity by 9.44 points, and cancer specificity by 3.02 points (`outputs/dermoai/runtime/20261016-152650-e30bec65/comparison.json:40`). This is the strongest backed same-split comparison in the workspace, but it is not proof of clinical superiority because the original model's historical training membership is unknown and overlap cannot be excluded.

The main safety concern is external generalization. On a sealed, pathology-confirmed 318-image DDI v1 cohort with no byte-identical overlap against the V2 manifest, refined V2 reached only 28.93% accuracy, 20.62% six-class macro-F1, and 25.20% cancer sensitivity, despite 93.33% specificity (`outputs/dermoai/external-eval/ddi/results-20261016-152650-e30bec65/comparison.md:5`). This large in-distribution/external gap is consistent with material domain shift, dataset-source shortcuts, label-taxonomy mismatch, and/or residual non-byte-identical related-image leakage. It rules out any responsible diagnostic or cancer-screening claim.

## 2. Model identity and architecture

### 2.1 Architecture

- **Internal architecture key:** `convnextv2_base`.
- **timm pretrained model:** `convnextv2_base.fcmae_ft_in22k_in1k` (`outputs/dermoai/runtime/20261016-152650-e30bec65/status.json:4`; registry mapping at `outputs/dermoai/dermoai/models.py:8`).
- **Input size:** 224×224 RGB (`outputs/dermoai/runtime/20261016-152650-e30bec65/config.json:49`).
- **Output:** one 12-logit classifier head.
- **Fine-tuning mode:** partial. The classifier, all of stage 3, and the last three blocks of stage 2 were trainable (`outputs/dermoai/runtime/20261016-152650-e30bec65/config.json:34`; `outputs/dermoai/runtime/20261016-152650-e30bec65/config.json:35`).
- **Trainable parameters:** 33,870,348 (`outputs/dermoai/runtime/20261016-152650-e30bec65/status.json:40`).
- **Initialization:** warm-started from epoch 4 of run `20261016-131856-b356b125`, checkpoint hash `a776fb3bf11d13ce3b1abc1507cb66f7d521bd41940dc9df5f57b47717b6a8b9` (`outputs/dermoai/runtime/20261016-152650-e30bec65/status.json:35`). The original CE-LS model was a comparison reference, not an initialization source (`outputs/dermoai/REFINEMENT.md:3`).

### 2.2 Exact canonical class order

Class order is part of the checkpoint contract and must not be changed when loading weights:

| Index | Code | Display name |
|---:|---|---|
| 0 | BCC | Basal Cell Carcinoma |
| 1 | ACK | Actinic Keratosis |
| 2 | NEV | Melanocytic Nevus |
| 3 | SEK | Seborrheic Keratosis |
| 4 | SCC | Squamous Cell Carcinoma |
| 5 | MEL | Melanoma |
| 6 | Acne | Acne and Rosacea |
| 7 | Alopecia | Alopecia and Hair Loss |
| 8 | Nail | Nail Fungus and Disease |
| 9 | Fungal | Fungal Infections |
| 10 | Vascular | Vascular Tumors |
| 11 | Healthy | Healthy Skin |

The checkpoint/config order is recorded at `outputs/dermoai/runtime/20261016-152650-e30bec65/config.json:4` and repeated in final metrics at `outputs/dermoai/runtime/20261016-152650-e30bec65/final-test/metrics.json:250`. Display-name evidence is in `work/dermoai-final/main.py:20`.

The three classes treated as “high risk” for the binary cancer-safety summary are **BCC, SCC, and MEL** (`outputs/dermoai/runtime/20261016-152650-e30bec65/config.json:18`).

## 3. Dataset, provenance, and splits

### 3.1 Frozen manifest

The active manifest is `outputs/dermoai/datasets/training-v2.csv`. Its frozen dataset version is:

`05cdbd7e3dd527f9b6ee434d2e98de6d7258260cf142b3ef0022ec019ccf5ccb`

The split fingerprints are:

- Train: `a253f1ca0b82314600b76543f726b7611d73757434a56b2f6570a718490e2ef0`
- Validation: `abe085e4ddfb309d289362f2c31ec04dbf6bfea025e2c2688f03b729c1780c36`
- Test: `50af5b67b5293a9e628b1669c4f035197c677e38b0955359589ae207eaa25cc4`

Evidence: `outputs/dermoai/runtime/20261016-152650-e30bec65/dataset_snapshot.json:199`.

### 3.2 Image counts

| Split | Images |
|---|---:|
| Train | 28,056 |
| Validation | 4,104 |
| Test | 4,336 |
| **Total** | **36,496** |

The run contract records these sizes at `outputs/dermoai/REFINEMENT.md:17`; the audit records 36,496 total source-labeled images at `outputs/dermoai/runtime/20261016-152650-e30bec65/dataset_snapshot.json:13`.

### 3.3 Source composition

Counts below were computed directly from `outputs/dermoai/datasets/training-v2.csv`:

| Source | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| ISIC-2019 | 18,174 | 2,416 | 2,350 | 22,940 |
| ISIC-2020 | 4,469 | 571 | 591 | 5,631 |
| DermNet-Kaggle | 2,665 | 667 | 1,024 | 4,356 |
| PAD-UFES-20 | 1,606 | 341 | 316 | 2,263 |
| Skin-Types-Kaggle-normal | 1,059 | 87 | 46 | 1,192 |
| MSLD-v2-original-fold1 | 83 | 22 | 9 | 114 |

The manifest schema explicitly carries source, patient, lesion, split, source-split, label-status, and similarity-group fields (`outputs/dermoai/datasets/training-v2.csv:1`).

### 3.4 Test class distribution

| Class | Test support |
|---|---:|
| BCC | 429 |
| ACK | 171 |
| NEV | 1,790 |
| SEK | 273 |
| SCC | 88 |
| MEL | 479 |
| Acne | 310 |
| Alopecia | 38 |
| Nail | 233 |
| Fungal | 322 |
| Vascular | 148 |
| Healthy | 55 |

Evidence: `outputs/dermoai/runtime/20261016-152650-e30bec65/dataset_snapshot.json:17` and `outputs/dermoai/runtime/20261016-152650-e30bec65/final-test/per_class.csv:1`.

### 3.5 Leakage, label, and patient-ID limitations

The dataset audit reported no corrupt rows and no exact byte-duplicate rows crossing the frozen snapshot (`outputs/dermoai/runtime/20261016-152650-e30bec65/dataset_snapshot.json:15`). That is useful but does **not** prove semantic independence: resized, recropped, recompressed, or repeated views can evade exact-hash detection.

The decisive limitations are:

1. **Patient separation cannot be verified for three large sources:** DermNet-Kaggle (4,356 images), ISIC-2019 (22,940), and Skin-Types-Kaggle-normal (1,192). The run marks itself exploratory for this reason (`outputs/dermoai/runtime/20261016-152650-e30bec65/dataset_snapshot.json:4`).
2. **All 36,496 labels are dataset-source labels, not independently human-reviewed annotations** (`outputs/dermoai/runtime/20261016-152650-e30bec65/dataset_snapshot.json:6`). The first manifest records confirm `dataset_source_unreviewed` status (`outputs/dermoai/datasets/training-v2.csv:2`).
3. **The split is not a clean new external test.** The test was examined during earlier comparisons, and the original model may have trained on some of its content (`outputs/dermoai/REFINEMENT.md:21`).
4. **Original CE-LS membership is unknown.** Therefore, same-split comparisons cannot establish that either system was evaluated on wholly unseen examples (`outputs/dermoai/runtime/20261016-152650-e30bec65/comparison.json:69`).
5. **Strong imbalance remains.** NEV alone is 1,790 of 4,336 test images, while SCC has 88, Healthy 55, and Alopecia 38. Training has 13,683 NEV images but only 121 Alopecia and 604 SCC images (`outputs/dermoai/runtime/20261016-152650-e30bec65/dataset_snapshot.json:79`). Macro-F1 is therefore more informative than accuracy, but small-class estimates also have wide uncertainty.
6. **Source and taxonomy heterogeneity:** the model combines lesion-level cancer classes, broad disease families, nail/hair categories, and “Healthy” images from multiple datasets and acquisition pipelines. This creates a plausible opportunity for the model to learn source/capture cues rather than disease morphology. The DDI result supports this as a concern but does not by itself identify the exact causal shortcut.

## 4. Training recipe

The refined run intentionally retained the frozen manifest and changed only the optimization/fine-tuning recipe (`outputs/dermoai/REFINEMENT.md:3`).

| Component | Setting |
|---|---|
| Backbone | ConvNeXt V2-Base, pretrained `fcmae_ft_in22k_in1k` |
| Fine-tuning | Partial: stage 2 blocks 24–26, all stage 3, classifier head |
| Resolution | 224×224 RGB |
| Optimizer | Fused AdamW on CUDA |
| Classifier learning rate | `1e-4` |
| Backbone learning rate | `2e-5` via `backbone_lr_scale=0.2` |
| Weight decay | `0.01` |
| Schedule | Cosine over 12 configured epochs; no warmup |
| Physical batch | 8 |
| Gradient accumulation | 4 steps; effective target batch 32 |
| Loss | Cross-entropy with label smoothing `0.03` |
| Class handling | Square-root inverse sampling; no additional class-loss weights |
| Augmentation | `conservative_skin` |
| Precision/layout | CUDA FP16 autocast + GradScaler; channels-last |
| Gradient clipping | Global norm `1.0` |
| Seed | 42 |
| Early stopping | Minimum 8 epochs, patience 4 |

Configuration evidence is at `outputs/dermoai/runtime/20261016-152650-e30bec65/config.json:31`. AdamW, differential learning rates, and AMP are implemented at `outputs/dermoai/dermoai/training.py:203`; gradient accumulation and clipping are implemented at `outputs/dermoai/dermoai/training.py:332`.

The conservative training transform uses random resized cropping with scale 0.85–1.0, aspect ratio 0.9–1.1, horizontal and vertical flip probabilities of 0.5, color jitter 0.1, and no random erasing or auto-augmentation (`outputs/dermoai/dermoai/models.py:90`). Evaluation uses border cropping with crop percentage 1.0.

The run completed eight epochs and stopped because the validation selection score had not improved within the configured patience after the minimum epoch requirement (`outputs/dermoai/runtime/20261016-152650-e30bec65/status.json:13`; `outputs/dermoai/runtime/20261016-152650-e30bec65/status.json:637`). This is expected early stopping, not a crash.

## 5. Checkpoint selection

The selected checkpoint is epoch 2, not the final epoch. Selection used the validation-only score:

`0.5 × macro-F1 + 0.5 × harmonic_mean(cancer_sensitivity, cancer_specificity)`

The implementation is at `outputs/dermoai/dermoai/training.py:143`. This score is **not accuracy** and should not be reported as accuracy. Epoch 2's best validation selection score was 0.7874314008 (`outputs/dermoai/runtime/20261016-152650-e30bec65/status.json:46`).

The selected checkpoint metadata is:

- Path: `outputs/dermoai/runtime/20261016-152650-e30bec65/epoch-0002.pt`
- Epoch: 2
- SHA-256: `9d0cb9f692397b81ac5527cda06d9d9502056e6edead6440060150b8ce316c21`
- Dataset version: `05cdbd7e3dd527f9b6ee434d2e98de6d7258260cf142b3ef0022ec019ccf5ccb`
- Selection split: validation
- Test used for selection: false

Evidence: `outputs/dermoai/runtime/20261016-152650-e30bec65/final-selection.json:2`.

The checkpoint embeds model weights, optimizer state, AMP scaler state, Python/PyTorch/CUDA RNG state, full config, epoch, best score, and dataset version. This supports exact continuation subject to matching software/hardware behavior. The training code rejects a resume if important config fields or the dataset version differ (`outputs/dermoai/dermoai/training.py:218`).

## 6. Frozen-test results

### 6.1 Overall results

| Metric | Result |
|---|---:|
| Images | 4,336 |
| Accuracy | 79.96% |
| Macro-F1 | 75.25% |
| Cancer sensitivity | 77.71% |
| Cancer specificity | 91.71% |
| Cancer false negatives | 222 |
| Cancer false positives | 277 |
| Smartphone subset | 316 images |
| Smartphone accuracy | 72.15% |

Evidence: `outputs/dermoai/runtime/20261016-152650-e30bec65/final-test/metrics.json:1` and `outputs/dermoai/runtime/20261016-152650-e30bec65/final-test/metrics.json:264`.

### 6.2 Per-class results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| BCC | 71.34% | 77.16% | 74.13% | 429 |
| ACK | 64.10% | 58.48% | 61.16% | 171 |
| NEV | 89.56% | 87.21% | 88.37% | 1,790 |
| SEK | 56.27% | 57.51% | 56.88% | 273 |
| SCC | 35.48% | 25.00% | 29.33% | 88 |
| MEL | 60.95% | 66.81% | 63.75% | 479 |
| Acne | 93.21% | 84.19% | 88.47% | 310 |
| Alopecia | 83.78% | 81.58% | 82.67% | 38 |
| Nail | 96.05% | 93.99% | 95.01% | 233 |
| Fungal | 86.45% | 89.13% | 87.77% | 322 |
| Vascular | 70.69% | 83.11% | 76.40% | 148 |
| Healthy | 98.21% | 100.00% | 99.10% | 55 |

Evidence: `outputs/dermoai/runtime/20261016-152650-e30bec65/final-test/per_class.csv:1`.

### 6.3 Cancer-safety interpretation

The cancer summary converts the 12-way top-1 prediction into a binary decision: a prediction is “cancer/high risk” only when its winning class is BCC, SCC, or MEL. It is not a pathology endpoint, a calibrated risk score, or a clinically selected operating threshold.

The 91.71% specificity is below “near 100%,” and the 77.71% sensitivity means roughly 22% of cancer-class examples were not assigned a cancer top-1 label on this test. SCC is especially weak: only 22 of 88 SCC images were correctly classified as SCC, producing 25% class recall (`outputs/dermoai/runtime/20261016-152650-e30bec65/final-test/metrics.json:31`). Even if some SCC errors land in another high-risk class, the external DDI result demonstrates that the binary sensitivity itself can collapse under domain shift.

## 7. Backed comparisons

### 7.1 Refined V2 versus original CE-LS on the same local test

The “original” comparator is the repository ConvNeXt-Base checkpoint `work/dermoai-final/model_weights/ce_ls_best.pth`, as identified in `outputs/dermoai/benchmarks/original-convnext-base-same-split.json:2`.

| Metric | Original CE-LS | Refined V2 | V2 difference |
|---|---:|---:|---:|
| Accuracy | 76.59% | 79.96% | +3.37 points |
| Macro-F1 | 73.28% | 75.25% | +1.98 points |
| Cancer sensitivity | 68.27% | 77.71% | +9.44 points |
| Cancer specificity | 88.68% | 91.71% | +3.02 points |

Evidence: `outputs/dermoai/runtime/20261016-152650-e30bec65/comparison.json:40`.

This is a favorable result for refined V2 under the local test protocol. It is not a valid claim that V2 is generally or clinically superior because original-model training overlap is unknown, the test was previously inspected, and patient separation is unverified for major sources.

### 7.2 Refined V2 versus the previous V2 checkpoint

| Metric | Previous V2 | Refined V2 | Difference |
|---|---:|---:|---:|
| Accuracy | 75.78% | 79.96% | +4.17 points |
| Macro-F1 | 73.41% | 75.25% | +1.85 points |
| Cancer sensitivity | 80.62% | 77.71% | -2.91 points |
| Cancer specificity | 86.32% | 91.71% | +5.39 points |

Refinement traded some sensitivity for substantially higher specificity and modestly higher accuracy/macro-F1. Evidence: `outputs/dermoai/runtime/20261016-152650-e30bec65/comparison.json:48`.

### 7.3 Historical user-reported baseline

A separate artifact records 82.79% accuracy, 87.30% cancer sensitivity, and 92.27% specificity as historical user-reported results, not measurements from this pipeline. Its dataset, class mapping, and split are unavailable, so direct comparison is invalid (`outputs/dermoai/runtime/20261016-152650-e30bec65/final-test/historical_baseline.json:2`). Those figures must not be used as evidence that V2 regressed or improved.

### 7.4 Sealed DDI external evaluation

The DDI v1 evaluation is the most important generalization evidence currently available:

| Metric | Original CE-LS | Refined V2 |
|---|---:|---:|
| Accuracy | 36.16% | 28.93% |
| Macro-F1, six observed classes | 25.82% | 20.62% |
| Cancer sensitivity | 56.10% | 25.20% |
| Cancer specificity | 67.69% | 93.33% |
| Cancer false negatives | 54 / 123 | 92 / 123 |
| Cancer false positives | 63 / 195 | 13 / 195 |

The 318-image cohort had direct mappings to six lesion classes, pathology-confirmed labels, no byte-identical overlap with the V2 manifest, and no use in training, selection, or threshold tuning (`outputs/dermoai/external-eval/ddi/results-20261016-152650-e30bec65/comparison.md:3`). V2's high external specificity came with critically low sensitivity. It should not be treated as a screening model.

## 8. Failure analysis

1. **SCC is the clearest class failure.** Test F1 is 29.33% and recall is 25.00%. Low support, visual overlap with BCC/ACK/SEK, and source heterogeneity are plausible contributors, but the artifacts do not isolate causality.
2. **Melanoma remains difficult.** MEL F1 is 63.75%; the confusion matrix shows 103 MEL images predicted as NEV and 24 as SEK (`outputs/dermoai/runtime/20261016-152650-e30bec65/final-test/metrics.json:151`).
3. **Common-class accuracy masks weak rare classes.** NEV contributes 1,790 test examples and has 88.37% F1, while SCC contributes only 88 and has 29.33% F1. Overall accuracy therefore overstates balanced performance.
4. **Healthy performance is not yet trustworthy as a broad “normal skin” claim.** Test recall is 100%, but support is only 55 and Healthy images come from narrow source definitions. This does not establish performance across body sites, skin tones, cameras, or benign conditions outside the training taxonomy.
5. **The DDI collapse shows severe domain sensitivity.** The external accuracy gap (79.96% local versus 28.93% DDI) is too large to dismiss as sampling noise. The likely problem is not model capacity but data representativeness, source confounding, taxonomy harmonization, and independent evaluation quality.
6. **Cancer behavior is overly conservative externally.** On DDI, V2 reduced false positives from 63 to 13 but increased false negatives from 54 to 92. This trade-off is incompatible with a sensitivity-first cancer-screening objective.
7. **Unreviewed source labels cap achievable reliability.** Label noise and broad source-category mappings can teach inconsistent decision boundaries, especially among ACK, SEK, SCC, and MEL.

## 9. Confidence, calibration, and inference caveats

The inference implementation applies `softmax` to the 12 logits and returns the largest value as `score` (`outputs/dermoai/dermoai/inference.py:67`). This value is **raw softmax confidence**, not a calibrated probability of disease. A displayed value such as 0.90 does not mean a 90% chance that the patient has that condition.

No temperature-scaling, isotonic-regression, Platt-scaling, expected calibration error, Brier score, reliability diagram, abstention threshold, or clinically validated operating point is recorded for the selected V2 artifact. Until calibration is performed on a truly independent, representative cohort, the UI and API should label outputs as **relative model confidence** and show all 12 scores without diagnostic language.

Additional inference constraints:

- The loader rejects checkpoints whose class order differs from the configured canonical order (`outputs/dermoai/dermoai/inference.py:52`).
- Real-image inference is configured for CUDA; CPU fallback is intentionally disabled outside synthetic tests (`outputs/dermoai/dermoai/models.py:46`).
- Images are EXIF-transposed, converted to RGB, transformed with the checkpoint's evaluation preprocessing, and evaluated under FP16 autocast on CUDA (`outputs/dermoai/dermoai/inference.py:65`).
- The existing backend accepts only registered local images and disables remote fetching (`outputs/dermoai/dermoai/inference.py:29`).
- The model can only choose among its 12 known classes. It has no “unknown,” “uncertain,” image-quality, or out-of-distribution class.
- A top prediction does not rule in that disease or rule out alternatives. Low-quality, cropped, occluded, non-skin, or out-of-taxonomy images may still receive a confident softmax score.

## 10. Reproducibility record

The run is comparatively well recorded for an exploratory experiment:

- Exact model checkpoint and SHA-256 in `outputs/dermoai/runtime/20261016-152650-e30bec65/final-selection.json:2`.
- Full run config in `outputs/dermoai/runtime/20261016-152650-e30bec65/config.json:1`.
- Immutable dataset version and split fingerprints in `outputs/dermoai/runtime/20261016-152650-e30bec65/dataset_snapshot.json:199`.
- Per-epoch checkpoints and validation reports in `outputs/dermoai/runtime/20261016-152650-e30bec65/`.
- Final predictions, confusion matrix, per-class metrics, and aggregate metrics in `outputs/dermoai/runtime/20261016-152650-e30bec65/final-test/`.
- Training seed 42, optimizer/scaler/RNG state embedded in the checkpoint.
- Training entry point documented at `outputs/dermoai/REFINEMENT.md:23`.

To reproduce the selected evaluation, use the existing Python environment and exact checkpoint/config/dataset snapshot. Do not silently regenerate splits, alter class order, replace source labels, or select a different epoch using test results. Software package versions should also be captured before any future archival release; the checkpoint itself does not establish a complete lockfile-level environment specification.

## 11. Prioritized next steps

### Priority 0 — safety and evaluation integrity

1. **Keep V2 research-only and non-diagnostic.** Do not market local test accuracy or specificity as clinical performance.
2. **Freeze the current test permanently.** It has already been inspected. Do not use it for architecture, augmentation, threshold, calibration, ensemble, or stopping decisions.
3. **Create a new, sealed, patient-separated external evaluation set** with verified non-membership, pathology/clinician labels where applicable, and enough BCC/SCC/MEL cases for uncertainty intervals. DDI should remain evaluation-only.
4. **Report uncertainty and subgroup results.** Add confidence intervals and stratify by skin tone, capture device, body site, source, and image quality whenever metadata permits.

### Priority 1 — data quality before larger models

5. **Review and harmonize labels.** Prioritize SCC, ACK, SEK, MEL, and BCC examples; document mapping rules and quarantine ambiguous or multi-diagnosis images.
6. **Recover patient/lesion identifiers or defensible grouping metadata.** Where unavailable, keep those sources out of definitive validation/test claims. Continue perceptual/near-duplicate screening in addition to exact hashes.
7. **Reduce source shortcuts.** Balance acquisition domains within each class, test source-held-out splits, and audit whether class predictions can be made from source/camera cues.
8. **Expand difficult classes with genuinely new patients, not offline copies.** Augmentation improves invariance but does not create new clinical diversity or independent evidence.

### Priority 2 — model and calibration work

9. **Optimize for the actual clinical-research objective.** If cancer sensitivity is primary, define a validation-only operating point and evaluate the resulting specificity; do not optimize for near-100% specificity at the expense of missed cancers.
10. **Add explicit abstention and out-of-distribution handling.** Reject non-skin, poor-quality, and unsupported cases rather than forcing one of 12 labels.
11. **Calibrate after the data split is trustworthy.** Fit temperature scaling or another prespecified calibrator on a clean calibration set, then report ECE, Brier score, reliability curves, and classwise calibration on untouched test data.
12. **Target hard confusions.** Use error-reviewed SCC/ACK/SEK/BCC/MEL cohorts, source-balanced batches, and validation-only ablations. A larger backbone alone is unlikely to solve the DDI gap.

## 12. Bottom line

DermoAI V2 is a technically reproducible exploratory ConvNeXt V2-Base classifier that improves over the local CE-LS checkpoint on the same frozen split. Its best local result is 79.96% accuracy and 75.25% macro-F1, with 77.71% cancer sensitivity and 91.71% specificity. However, the model's weakest cancer class is SCC, major sources lack verifiable patient separation, all training labels are unreviewed dataset-source labels, and sealed DDI performance is poor—especially 25.20% cancer sensitivity. The current evidence supports continued research and data-quality work, not clinical deployment or a claim of reliable diagnosis.
