# DermoAI + Label Studio, local research workflow

**Current prepared setup:** see `TRAINING_READY.md`. The user authorized an exploratory 12-class dataset using source labels and a limited missing-patient exception for DermNet. The frozen training version contains 4,356 training, 1,030 validation, and 1,349 test images. The active recipe uses ConvNeXt-Base at its native 224px pretrained resolution. Original strict defaults and raw import provenance remain available; historical statements below describing the initial import are not the current training configuration.

Uses your existing Label Studio interface at http://localhost:8080. No replacement UI, Docker, cloud GPU, or external inference API. Images, training, reports, and inference stay on this computer. Internet is used only for dependencies, source datasets, and pretrained weights.

## Canonical classes

The supplied DermoAI order is preserved everywhere, including checkpoint heads and prediction probabilities:

`0 BCC, 1 ACK, 2 NEV, 3 SEK, 4 SCC, 5 MEL, 6 Acne, 7 Alopecia, 8 Nail, 9 Fungal, 10 Vascular, 11 Healthy`.

This mapping was supplied by the user. The referenced GitHub `main.py` was not provided or independently inspected. Existing external checkpoints must be converted deliberately; the loader expects this pipeline's checkpoint format and refuses a different class order.

## Current environment

The inspected Label Studio service is Community 1.23.2. The original default Python is 3.14 with a CPU-only PyTorch build. This project uses a separate Python 3.12 environment with PyTorch 2.6.0+cu126 and torchvision 0.21.0+cu126. CUDA computation was verified on the RTX 4050 Laptop GPU. The installed ML backend is pinned to official revision `7b0d01f3593e327ad9cd2a92dbd08aa89f6403f4`, with SDK 2.1.2. See `verification.md` and `model-verification.json` for completed checks and their limits.

The already installed environment is under the task's `work/dermoai-venv` directory. The launcher detects it automatically. When moving this folder elsewhere, run the installer to create a new `.venv`; virtual environments are not portable.

## Start and inspect

Open PowerShell in this directory:

```powershell
. .\scripts\Environment.ps1
. .\scripts\Set-Credentials.ps1
& $Python -m dermoai.cli health
& $Python -m dermoai.cli --config config.json inspect
& $Python -m dermoai.cli --config config.json audit --output runtime\dataset-health.json
.\scripts\Start-Backend.ps1
```

The credential helper prompts for the Label Studio token with hidden input and sets it only in that PowerShell session. A separate random local control token is generated automatically, encrypted with Windows DPAPI in `runtime/control-token.dpapi`, and loaded into `DERMOAI_CONTROL_TOKEN` for the current Windows user. Never put plaintext tokens in config, source files, logs, or chat. After sharing a token in chat, rotate it through Label Studio Account & Settings → Access Tokens and enter its replacement locally.

The backend binds only to `127.0.0.1:9090`. Keep it running to display imported images. It serves the original image bytes from an allowlist constructed from the manifest; it does not duplicate images into Label Studio. `/media/<sha256>` is readable by local browser processes so Label Studio can display it. Training control endpoints require a separate `DERMOAI_CONTROL_TOKEN` bearer value. There is no network interface binding or cloud upload.

In Label Studio, open your project's Settings → Model and connect `http://127.0.0.1:9090` when ready. The backend checks the project ID, tag names, and exact class order. An untrained backend will reject prediction requests until a real checkpoint is explicitly selected. It does not use the synthetic test model for real images.

Project 2, **DermoAI — 12-Class Research**, now contains 6,767 unique images across the exact 12 classes. Their source labels are reference metadata only; there are no reviewed annotations and no real-data training has started.

The same connection can be made entirely through the SDK: `& $Python -m dermoai.cli --config config.json connect-backend --approve`. No browser automation is needed.

Label Studio 1.23.2 defaults to blocking localhost ML connections. If it reports `URL resolves to a reserved network address`, the prepared `scripts/Start-LabelStudioLocal.ps1` launcher sets `ML_BLOCK_LOCAL_IP=false` for that process, binds Label Studio to `127.0.0.1:8080`, and leaves `SSRF_PROTECTION_ENABLED=true`. Stop the existing Label Studio process before using it. This is a deliberate operator configuration allowing local/private ML endpoints, not a URL-filter workaround. See the official [1.23.1 release notes](https://github.com/HumanSignal/label-studio/releases/tag/1.23.1). General URL-import/storage restrictions remain enabled.

In this installed version, creation can still return HTTP 403 from automatic webhook registration: webhook validation independently applies general SSRF protection after the ML backend record has been saved. The SDK helper rechecks the registered backend and its live connection state instead of creating a duplicate. Local automatic annotation webhooks remain unavailable; use explicit SDK annotation export/sync and approved training commands. Do not disable general SSRF protection to silence this warning.

To install on a fresh Windows setup with `uv` and Git available:

```powershell
.\scripts\Install.ps1
```

The installer downloads the official CUDA 12.6 wheels; a separately installed CUDA toolkit is not required for those wheels. The NVIDIA driver must support the runtime. Inspect `health` rather than assuming `nvidia-smi` alone proves PyTorch can use CUDA. No silent CPU training fallback exists.

## Dataset sources and provenance

The original reported 20,000+ image dataset and 4,555-image validation set were not available. These Kaggle datasets are new source imports, not a reconstruction of those historical splits:

| Source | Selected labels | Provenance and split handling |
| --- | --- | --- |
| [PAD-UFES-20 on Kaggle](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer) | BCC, ACK, NEV, SEK, SCC, MEL | Preserve patient and lesion IDs. New deterministic 70/15/15 assignment by connected patient/lesion/exact-image groups, recorded as a new split. |
| [DermNet on Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) | Acne, Alopecia, Nail, Fungal, Vascular | Original import was unassigned because patient identities are absent. The authorized exploratory version preserves source test membership and group-splits source training; see `TRAINING_READY.md` for residual risks. |
| [MSLD v2 on Kaggle](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20) | Healthy | Only original Healthy images in fold1. Exclude augmentation folders and copies in other folds. Preserve fold1 split and documented filename patient IDs. |

PAD-UFES-20's original release is [Mendeley DOI 10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1), CC BY 4.0. Cite Pacheco and colleagues (2020). DermNet's Kaggle listing declares CC BY-NC-ND 4.0. MSLD v2 declares CC BY-NC 4.0; cite Ali and colleagues, *A Web-based Mpox Skin Lesion Detection System Using State-of-the-art Deep Learning Models Considering Racial Diversity* (2024). These source restrictions and attribution obligations remain attached to the data. Source metadata files are retained alongside downloads; do not treat third-party mirrors' conflicting license claims as authoritative.

The mappings from DermNet folder names are explicit: `Acne and Rosacea Photos → Acne`; `Hair Loss Photos Alopecia and other Hair Diseases → Alopecia`; `Nail Fungus and other Nail Disease → Nail`; `Tinea Ringworm Candidiasis and other Fungal Infections → Fungal`; `Vascular Tumors → Vascular`. Broad source categories still require review for compatibility with your intended target definitions.

HAM10000 was considered but not added merely to increase the image count: its lesion IDs do not establish patient identity, and combining overlapping sources requires additional leakage checks. There is no invented dataset named “Oncology Set” or presumed MIDAS match.

## Manifest and review workflow

The original `datasets/import-manifest.csv` remains the image-serving allowlist. The active exploratory training manifest is `datasets/training-v1.csv`. The raw `datasets/manifest.csv` and initial audit remain provenance records. The import audit found 114 unique images with conflicting source labels, retained in `import-manifest.quarantine.json` and excluded from import. Same-label duplicates are recorded in `import-manifest.aliases.json`. No downloaded images were deleted. The original import contains 6,767 unique decodable images. The later authorized preparation uses 6,735 after a further similarity screen; it assigns DermNet exploratory splits without inventing patient IDs. See `training-readiness.json` for the frozen split identifiers and limitations.

Required columns are `image_path,label,source,patient_id,lesion_id,split,approved`. Optional fields include `original_validation,smartphone,task_id,source_split`. Paths can be absolute or relative to the manifest. `label` must exactly match a canonical class. `split` is `train`, `validation`, `test`, or `unassigned`. Patient identity is namespaced by source; cross-source identity reconciliation still requires source metadata. Exact image hashes also catch byte-identical copies across sources.

Source labels are imported into task **data** as `reference_label`. They are not fabricated human annotations and are initially `approved=false` in the manifest. Review through Label Studio. Export an explicit JSON list of approved task IDs, then create a new training manifest:

```powershell
& $Python -m dermoai.cli --config config.json export runtime\tasks-with-annotations.json
& $Python -m dermoai.cli --config config.json sync-annotations datasets\reviewed.csv --approved-task-ids approved-task-ids.json
```

The sync command requires one consistent class across non-cancelled annotations. Conflicting annotations fail for adjudication. It preserves the image paths and splits, writes a new manifest, and never modifies annotations in Label Studio. Set `manifest` in your experiment config to the new reviewed file. SDK exports include task data, predictions and annotations.

Bulk import is explicit and skips already-present content hashes:

```powershell
& $Python -m dermoai.cli --config config.json import --approve
```

Do not retry an interrupted import until inspecting the project; the next run uses existing hashes to avoid duplicates. Importing has no annotation overwrite behavior. It preserves data provenance and patient identifiers.

To index a new local folder without assuming folder names are diagnoses:

```powershell
& $Python -m dermoai.cli scan C:\Data\Images datasets\new-images.csv --source MySource
```

Fill labels and identity/split metadata explicitly. `download HANDLE DESTINATION` supports the official KaggleHub client; gated datasets require your own local Kaggle credentials and any applicable terms acceptance. The application never bypasses access restrictions.

Before training, the audit verifies image decoding, file hashes, class distribution, patient and lesion separation, exact duplicates across splits, configured labels, approved status, and protected original-validation flags. It records immutable content snapshots and per-split fingerprints for each experiment. It never automatically repartitions an existing manifest. Corrupt files are reported, not deleted. Within-split exact duplicates are reported for review; they are not silently removed.

**Current real training is deliberately blocked:** DermNet patient IDs are unavailable, source labels are not approved, and all 12 classes must have eligible training examples. Do not invent patient IDs or interpret a random image-level split as patient-safe. Patient IDs or an independently validated grouping scheme are needed to unlock those five classes. The original validation set, if later recovered, must be ingested with `original_validation=true` and `split=validation`; no historical baseline equivalence can be claimed before then.

## Model and training configuration

Copy `config.example.json` to a separate experiment config and use `--config path.json`. Paths resolve relative to that configuration file. Supported architecture keys:

| Key | Registered pretrained checkpoint |
| --- | --- |
| `convnext_base` | `convnext_base.fb_in22k_ft_in1k` |
| `convnext_tiny` | `convnext_tiny.fb_in22k_ft_in1k` |
| `efficientnetv2_s` | `tf_efficientnetv2_s.in21k_ft_in1k` |
| `densenet121` | `densenet121.ra_in1k` |
| `swin_tiny` | `swin_tiny_patch4_window7_224.ms_in1k` |
| `resnet50` | `resnet50.a1_in1k` |

Heads adapt to the configured canonical class count. Normalization comes from each model's pretrained configuration and is used consistently in training, evaluation, and predictions. The strict example config uses frozen-backbone defaults. The active exploratory recipe uses 224px ConvNeXt-Base partial fine-tuning, batch 2, accumulation 8, workers 0, learning rate 0.0001, warmup/cosine decay, class weighting, smoothing, and early stopping; see `TRAINING_READY.md`. FP16 is enabled on CUDA. Windows loaders use guarded CLI entry points.

`mode=frozen` trains the head and keeps backbone batch-normalization statistics frozen. `mode=partial` requires explicit verified parameter prefixes in `unfreeze_prefixes` (for example, inspect `model.named_parameters()` before choosing). `mode=full` trains all parameters and may exceed 6 GB for larger architectures.

On CUDA OOM, the engine rolls back the entire unfinished epoch to its prior checkpoint, halves the microbatch, and maintains the configured effective batch by sample-weighted gradient accumulation. It does not retain partial optimizer updates from the failed epoch. If batch 1 fails, it stops with a recorded error and retained checkpoint. No CPU fallback occurs. Evaluation and single-image inference report errors rather than silently changing the model or device.

The authorized exploratory preparation is complete. `scripts/Start-Training.ps1 -Approve` starts the prepared recipe, capped at 20 epochs with validation-based early stopping. Training itself still requires approval; none was launched during preparation.

```powershell
& $Python -m dermoai.cli --config experiment.json train --approve
& $Python -m dermoai.cli --config experiment.json status RUN_ID
& $Python -m dermoai.cli --config experiment.json stop RUN_ID
& $Python -m dermoai.cli --config experiment.json train --approve --resume C:\Path\epoch-0001.pt
```

Only one training worker can be active per configured runtime directory. A stop request ends at a batch boundary; the incomplete epoch is discarded and the previous complete checkpoint remains resumable. Set `epochs` to the total target count when resuming. Resume creates a new experiment directory and checks class order, model settings, optimizer settings, and dataset fingerprint. Existing checkpoints are never overwritten. The supported resume granularity is a completed epoch, not an arbitrary batch.

Each run stores configuration, dataset snapshot, status, epoch reports, model/optimizer/scaler/RNG state, pretrained identifier, elapsed time, best epoch, memory usage, and checkpoint locations. Read `runtime/RUN_ID/worker.log` for failures. Check `status.json` for structured errors. No TensorBoard or extra service is required.

To train and automatically publish predictions on successful completion, explicitly opt in:

```powershell
& $Python -m dermoai.cli --config experiment.json train --approve --publish-predictions
```

Keep this command running; the parent waits for the worker and then publishes predictions with the best checkpoint through the SDK. Failure or stop does not publish. This creates predictions, not annotations. For the persistent ML backend, select a checkpoint in `config.json` and restart it. Automatic annotation webhooks never launch training: `fit` returns an approval-required response for START_TRAINING and otherwise acknowledges the event.

## Evaluation and comparison

```powershell
& $Python -m dermoai.cli --config experiment.json evaluate C:\Path\epoch-0001.pt --split validation --output runtime\validation-report
& $Python -m dermoai.cli --config experiment.json compare
& $Python -m dermoai.cli --config experiment.json experiments
```

Use the identical manifest and split fingerprint for model comparisons. Hyperparameter/model selection uses validation only. Use `--split test` once a model and configuration are selected; never select hyperparameters using the resulting test metrics. Training never loads the test split into its loaders.

Reports include accuracy; per-class precision/recall/F1; macro F1; confusion matrix PNG; JSON and CSV; per-image predictions; and smartphone accuracy where the metadata explicitly identifies smartphone images. Configure `high_risk_classes` explicitly to enable the binary cancer/high-risk sensitivity, specificity, false positives and false negatives. With an empty list or missing denominator, those rates are null. ACK's role is not inferred. These are research metrics, not validated clinical performance.

The historical 82.79% accuracy, 87.30% cancer sensitivity and 92.27% cancer specificity are stored separately with their provenance. They are not newly measured results and are not directly comparable to the newly sourced datasets.

## Local control API

All `/control/` requests require `Authorization: Bearer <DERMOAI_CONTROL_TOKEN>`. Use `Content-Type: application/json` for POST. Credentials must be supplied through environment variables rather than request examples saved on disk.

| Method | Endpoint | Input |
| --- | --- | --- |
| POST | `/control/start_training` | `{"approved":true,"overrides":{"architecture":"convnext_tiny","epochs":1}}`; optional `resume` |
| POST | `/control/stop_training` | `{"run_id":"..."}` |
| GET | `/control/get_training_status?run_id=...` | none |
| GET | `/control/list_experiments` | none |
| POST | `/control/evaluate_model` | `{"checkpoint":"C:/.../epoch-0001.pt","split":"validation"}` |
| POST | `/control/predict_image` | `{"image":"C:/.../registered-image.png"}` |

Only listed training overrides are accepted. Dataset paths, class order and other sensitive settings must be changed explicitly in the local configuration. Evaluation is synchronous; training starts as a background subprocess. Avoid concurrent GPU evaluation/prediction requests. SDK ML routes `/health`, `/setup`, `/predict`, and `/webhook` use the official package's contracts. The local control API is separate from Label Studio's UI; it does not claim Label Studio provides arbitrary architecture menus or training graphs.

```powershell
$headers = @{ Authorization = "Bearer $env:DERMOAI_CONTROL_TOKEN" }
Invoke-RestMethod http://127.0.0.1:9090/control/list_experiments -Headers $headers
```

## Verification and troubleshooting

```powershell
. .\scripts\Environment.ps1
& $Python -m unittest discover -s tests -p 'test_*.py' -v
& $Python tests\smoke.py
```

The smoke test creates random synthetic images, runs a short GPU training job, verifies duplicate rejection, saves/resumes checkpoints, evaluates a held-out synthetic split, renders reports, and exercises the official backend and image streaming. Its metrics have no medical meaning. `model-verification.json` separately records pretrained checkpoint loading and 12-output GPU forward checks per architecture. `tests/recovery_smoke.py` injects an OOM to verify epoch rollback and tests safe stopping; this is not a claim that every real OOM scenario was exercised. The official ML wrapper's request/body debug hooks are disabled to avoid credential logging and an incompatibility with Flask streaming file responses.

- **CUDA unavailable:** run `health` using this environment's Python, not the system Python. The original system build is CPU-only. Reinstall the paired CUDA wheel versions if necessary.
- **401/403 from Label Studio:** use your local account's current API token. SDK 2.1.2 supports personal refresh tokens and legacy tokens. Never paste the token into diagnostics.
- **Broken images:** start the backend on 9090; it serves registered local files. Keep source paths stable, and restart after changing the manifest. No Label Studio local-file-serving setting change is required.
- **Missing patient IDs / unknown labels / unapproved data:** resolve metadata or annotations; the training refusal is intentional. Do not disable leakage checks to make an experiment run.
- **Out of memory:** close other GPU workloads, prefer Tiny or DenseNet, keep frozen mode, and lower microbatch. Large full fine-tuning is not guaranteed on 6 GB.
- **Interrupted process:** inspect the run status and resume the last complete checkpoint into a new run. There is no checkpoint deletion or replacement command.
- **Source downloads:** retry only failed public downloads; Kaggle-gated datasets require authentication. Network/DNS failures are distinct from package compatibility failures.

Official references: [Label Studio API](https://api.labelstud.io/api-reference/introduction/getting-started), [ML backend interfaces](https://labelstud.io/guide/ml_create), [PyTorch CUDA wheel versions](https://docs.pytorch.org/get-started/previous-versions/), [KaggleHub](https://github.com/Kaggle/kagglehub).
