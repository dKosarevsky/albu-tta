# ResNet50 TTA Next Steps

This note is the short read on what to do after looking at the raw per-image gain matrix.

## Files to inspect

- `tables/selector_public_gain_matrix.csv`: one row per public image; `aug_000`...`aug_099` are `clean_nll - aug_nll`.
- `tables/selector_aug_diagnostics.csv`: per-augmentation summary from the raw gain matrix.
- `tables/global_weight_topn_private_metrics.csv`: private readout when keeping only the top-N global aggregation weights.
- `tables/augmentation_impact.csv`: `aug_id` to augmentation name and transform class.

## Main read

The useful signal is real, but the current learned selector is not the strongest way to use it yet. `oracle_topk_uniform` gets private top1 `0.87492`, so there is a large image-specific upside; the current learned top-k gets only `0.80960`, while the simple global weighted ensemble gets `0.81488`.

The class-specific aggregator is not trustworthy as-is. It gets public-val top1 `0.894`, but private top1 only `0.80536`, which looks like classic public-val overfit from too much per-class capacity.

## Practical shortlist result

Global weighted TTA does not need all 100 transforms to keep most of its gain. Keeping only the top global weights gives:

| top_n | private top1 | private nll | compute vs all-100 |
| --- | --- | --- | --- |
| 4 | 0.81184 | 0.84056 | 0.04 |
| 8 | 0.81340 | 0.82281 | 0.08 |
| 16 | 0.81448 | 0.79028 | 0.16 |
| 24 | 0.81504 | 0.77499 | 0.24 |
| 32 | 0.81500 | 0.77127 | 0.32 |
| 100 | 0.81488 | 0.77104 | 1.00 |

For the next run, the cleanest baseline is probably `top-16` or `top-24` global-weighted TTA: most of the gain, much less compute, and no learned per-image selection risk.

Top-16 global shortlist:

`aug_025 aug_011 aug_029 aug_021 aug_020 aug_008 aug_024 aug_048 aug_072 aug_092 aug_051 aug_010 aug_047 aug_069 aug_050 aug_086`

That is mostly scale/rotate/shear plus autocontrast, grayscale, JPEG, equalize, Planckian, CLAHE, and sharpen. This is a much smaller set to inspect by eye.

## What to change next

1. Treat `top-16`/`top-24` global-weighted TTA as the next production-ish baseline and rerun it on another model family before tuning anything on private again.
2. Park the class aggregator for now; it needs regularization, fewer parameters, or a proper validation protocol before it is useful.
3. Fix selector training/evaluation next: the selector heavily picks some transforms with negative mean gain, so the current objective/calibration is not aligned enough with actual private lift.
4. Try selector heads against `helpfulness`, `negative_nll`, and `gain`, but evaluate them against the same top-N shortlist, not the full noisy 100-transform dictionary first.
5. Deprioritize heavy transforms for manual review unless they are clearly image-specific: square rotations, severe downscale, defocus, very low-quality compression, and large safe rotations are mostly harmful on average but have big positive tails.

## Caveat

The top-N table is a private readout of weights learned on public-val, not a new tuning set. Use it to choose the next experiment direction, then confirm on another architecture or a fresh split.
