# ConvNeXt V2 refinement

This run starts from the best checkpoint of `20261016-131856-b356b125` (epoch 4) on the existing frozen dataset. No images, labels or split assignments are changed. The original GitHub model is a comparison reference, not a teacher or initialization source: its training membership is unknown.

## Recipe

- Open the last three blocks of stage 2 as well as stage 3 and the classifier.
- Start a fresh optimizer with classifier learning rate 1e-4 and feature learning rate 2e-5, then cosine decay across 12 epochs.
- Batch 8 with four-step accumulation gives effective batch 32. CUDA memory recovery can reduce the physical batch while retaining the effective target.
- Keep square-root inverse sampling and remove the additional loss weighting. The previous recipe combined both, giving approximately equal total influence per class despite large support differences.
- Use conservative geometric/color augmentation and label smoothing 0.03.
- Allow at least eight epochs before early stopping, with patience four, so the lower learning rates have time to contribute.
- Evaluate the starting checkpoint first and keep it as a fallback if refinement does not improve validation selection score.

## Evaluation contract

Train: 28,056 images. Validation: 4,104 images. Test: 4,336 images.

Select checkpoints only with validation `balanced_cancer_safety`, the mean of macro-F1 and the harmonic mean of BCC/SCC/MEL sensitivity and specificity. This is a composite score, not accuracy. Report accuracy, macro-F1, sensitivity, specificity and per-class results separately. Compare against the original's validation accuracy 78.0458%, macro-F1 80.3014%, sensitivity 74.6114% and specificity 86.8748%, all using the same top-1 evaluation rule. Exceeding one metric does not establish overall superiority.

After selecting and freezing a checkpoint, evaluate test once. The test set has already been inspected in prior comparisons, so it is not a new independent external evaluation. Do not use its new result to tune this run. The original model may have seen some of these images; that overlap is unverified. In particular, its high PAD-UFES-20 result does not establish smartphone generalization. Patient identities remain unavailable for some source datasets.

## Run and monitor

From the workspace root, with the existing environment:

```powershell
& .\work\dermoai-venv\Scripts\python.exe -m dermoai.cli --config .\outputs\dermoai\refinement-config.json train --approve
.\outputs\dermoai\scripts\Watch-Training.ps1
```

The watcher defaults to the newest run. Pass `-RunId` to inspect an older run. A new run records its source checkpoint hash and keeps old checkpoints intact.
