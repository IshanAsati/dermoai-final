# Sealed external DDI v1 comparison

## Protocol

- Dataset: Stanford Diverse Dermatology Images v1, downloaded through the authorized AIMI account.
- Evaluation cohort: 318 images with conservative, direct mappings to BCC, ACK, NEV, SEK, SCC, and MEL.
- Reference labels: supplied pathology-confirmed DDI diagnoses.
- Admission audit: no byte-identical image overlap with the 36,496-image frozen DermoAI v2 training manifest.
- This cohort was excluded from training, validation, checkpoint selection, and threshold tuning.

## Results

| Metric | Original ConvNeXt-Base | Refined ConvNeXt V2-Base | Difference (V2 - original) |
| --- | ---: | ---: | ---: |
| Accuracy | 36.16% | 28.93% | -7.23 points |
| Macro-F1 (six observed classes) | 25.82% | 20.62% | -5.20 points |
| BCC/SCC/MEL sensitivity | 56.10% | 25.20% | -30.89 points |
| BCC/SCC/MEL specificity | 67.69% | 93.33% | +25.64 points |
| Cancer false negatives | 54 / 123 | 92 / 123 | +38 |
| Cancer false positives | 63 / 195 | 13 / 195 | -50 |

## Interpretation

The refined V2 checkpoint is substantially more conservative on DDI: it produces far fewer false positive cancer flags, but misses many more BCC/SCC/MEL cases. This is not acceptable evidence for a clinical cancer-screening model and it does not support a claim that the new model is globally better than the original.

The contrast with the frozen in-distribution test indicates material domain shift and/or residual source overlap in the earlier test. The original model's full historical training membership cannot be reconstructed, so a clean byte-level screen cannot prove historical non-membership for that model. These are research-only results, not clinical performance claims.
