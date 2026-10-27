# DermoAI V2 — training pipeline

This directory contains the training, evaluation, and inference code that produced the V2
checkpoint in [`../model_weights/dermoai_v2_convnextv2_base.pth`](../model_weights/dermoai_v2_convnextv2_base.pth).

The validated artifact is `convnextv2_base` fine-tuned partially at 224×224, selected at
epoch 2 of run `20261016-152650-e30bec65` by validation-only `balanced_cancer_safety`.
Full methodology and results: [`../V2_MODEL_REPORT.md`](../V2_MODEL_REPORT.md).

> **Research use only.** Not a medical device, diagnostic system, or screening test.
> Outputs are uncalibrated softmax scores, not probabilities of disease.

## What is included

| Path | Contents |
|---|---|
| `dermoai/` | Python package: dataset handling, model factory, training loop, evaluation, inference |
| `configs/refinement-config.json` | Exact configuration of the selected V2 run |
| `configs/config.example.json` | Template for a fresh configuration |
| `datasets/training-v2.csv` | Frozen V2 manifest (36,496 rows, paths + labels + split + similarity group) |
| `datasets/dataset_snapshot.json` | Audit flags, class distribution, dataset version, split fingerprints |
| `datasets/summary.json` | Per-split / per-source counts |
| `scripts/` | PowerShell helpers for environment setup and training |
| `tests/` | Unit tests for dataset preparation and training contracts |
| `REFINEMENT.md` | Original run contract (recipe decisions and evaluation rules) |
| `README.md` | This training guide |
| `PIPELINE_README.md` | Original workspace-level pipeline README, kept verbatim for provenance |

Not included: source images (multi-GB, separately licensed), per-row audit dumps, and
Label Studio export artifacts. The bundled `dataset_snapshot.json` omits only the
36,496-row per-image array; all counts and fingerprints are unchanged.

## Requirements

- Python 3.12 (`>=3.12,<3.13`)
- PyTorch 2.6 + torchvision 0.21, `timm>=1.0.15,<1.1`
- An NVIDIA CUDA device. Real-image training and inference are CUDA-only; CPU fallback is
  intentionally disabled outside synthetic tests (`dermoai/models.py`).

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements-tested.txt"
& ".\.venv\Scripts\python.exe" -m pip install -e .
```

## Dataset and splits

The frozen manifest version is
`05cdbd7e3dd527f9b6ee434d2e98de6d7258260cf142b3ef0022ec019ccf5ccb`.

| Split | Images | Fingerprint |
|---|---:|---|
| Train | 28,056 | `a253f1ca0b82314600b76543f726b7611d73757434a56b2f6570a718490e2ef0` |
| Validation | 4,104 | `abe085e4ddfb309d289362f2c31ec04dbf6bfea025e2c2688f03b729c1780c36` |
| Test | 4,336 | `50af5b67b5293a9e628b1669c4f035197c677e38b0955359589ae207eaa25cc4` |

Manifest columns: `image_path,label,source,patient_id,lesion_id,split,approved,original_validation,smartphone,task_id,source_split,label_status,similarity_group`.

`image_path` values are absolute paths from the original workstation and **must be
rewritten** to your local image root before training. Keep `split`, `similarity_group`, and
`label` untouched — changing them invalidates the dataset version and the training code
will refuse to resume against the expected version.

## Training

`configs/refinement-config.json` reproduces the selected run. Key settings:

| Component | Setting |
|---|---|
| Backbone | `convnextv2_base.fcmae_ft_in22k_in1k` |
| Fine-tuning | Partial: stage 2 blocks 24–26, stage 3, classifier |
| Resolution | 224×224 RGB |
| Optimizer | Fused AdamW, classifier LR `1e-4`, backbone scale `0.2` |
| Weight decay | `0.01` |
| Schedule | Cosine, 12 configured epochs, no warmup |
| Batch | physical 8 × 4 accumulation = 32 effective |
| Loss | Cross-entropy, label smoothing `0.03` |
| Sampling | Square-root inverse, no extra class weights |
| Augmentation | `conservative_skin` |
| Precision | CUDA FP16 autocast + GradScaler, channels-last, grad clip 1.0 |
| Seed | 42 |
| Early stopping | minimum 8 epochs, patience 4 |

```powershell
# from this directory, with the config paths corrected to your local image root
& "..\.venv\Scripts\python.exe" -m dermoai.cli --config configs\refinement-config.json train --approve
```

`--resume <run_id>` continues an interrupted run. The code rejects a resume when config
fields or the dataset version differ from the checkpoint.

For a from-scratch run, copy `configs/config.example.json` and set `warm_start` to `null`.
The published V2 checkpoint was warm-started from epoch 4 of run `20261016-131856-b356b125`,
which is not redistributed here.

Resolving checkpoints are written under `runtime/<run_id>/` as `epoch-NNNN.pt`; the selected
artifact is copied to `final-selection.json` with its SHA-256.

### Checkpoint selection

Checkpoints are selected on validation data only, using:

```
0.5 × macro-F1 + 0.5 × harmonic_mean(cancer_sensitivity, cancer_specificity)
```

This is `balanced_cancer_safety` — a composite score, **not** accuracy. Do not report it as
accuracy. The implementation is `dermoai/training.py` (`selection score`).

## Evaluation

```powershell
& "..\.venv\Scripts\python.exe" -m dermoai.cli --config configs\refinement-config.json evaluate --split test --output evaluation\test.json
```

Produces aggregate metrics, per-class metrics, and a confusion matrix. The test split was
already inspected during development and is not a clean external evaluation.

## Inference

`dermoai/inference.py` loads a checkpoint, enforces the canonical 12-class order, applies
the checkpoint's evaluation transform, and returns the full ranked distribution. Class
order is part of the checkpoint contract and must never be reordered:

`BCC, ACK, NEV, SEK, SCC, MEL, Acne, Alopecia, Nail, Fungal, Vascular, Healthy`

The bundled [`model_tester/`](../model_tester) is the supported end-user application; it
loads the same weights without the training dependencies.

## Reproducibility notes

- Training is deterministic only up to CUDA/cuDNN nondeterminism; the seed and RNG state are
  stored in each checkpoint.
- The dataset version and split fingerprints above must match before a run is considered
  comparable to V2.
- `requirements-tested.txt` records the environment used; no lockfile-level pin exists.
- Patient separation is unverifiable for DermNet-Kaggle, ISIC-2019, and
  Skin-Types-Kaggle-normal, and all labels are dataset-source labels rather than
  independently reviewed annotations. Treat results as exploratory.
