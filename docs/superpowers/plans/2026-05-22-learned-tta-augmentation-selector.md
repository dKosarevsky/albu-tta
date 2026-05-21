# Learned TTA Augmentation Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end experiment where a small CNN predicts which AlbumentationsX transforms are useful for TTA on each ImageNet validation image, reducing ResNet50 inference cost while preserving or improving metrics.

**Architecture:** The frozen teacher is `timm` ResNet50. Public ImageNet-val samples are used to precompute teacher behavior across 100 deterministic TTA candidates, train a lightweight selector CNN with 100 outputs, and tune selector rules. Private ImageNet-val samples are used only for final comparison against no TTA, fixed TTA, random TTA, all-candidates TTA, learned TTA, and oracle diagnostics.

**Tech Stack:** Python 3.10+, PyTorch, timm, AlbumentationsX, OpenCV headless, numpy, pandas, pyarrow/parquet, tqdm, pytest.

---

## Current Repository State

`/Users/if/Documents/albu-a` is currently an empty git repository with no committed files. This document defines the target project structure from scratch.

## External Choices

- Albumentations library: use AlbumentationsX, installed as `albumentationsx[headless]`.
- Albumentations import style: still `import albumentations as A`, per current AlbumentationsX docs.
- Teacher model: `resnet50.a1_in1k` from `timm`.
- Teacher preprocessing: always resolve from the loaded model with `timm.data.resolve_model_data_config(model)` and `timm.data.create_transform(..., is_training=False)`.
- Main dataset: ImageNet-1k validation set with 50 images per class.
- Split seed: `20260522`.
- Experiment seed for PyTorch, numpy, Python random, and AlbumentationsX Compose: `20260522`.

Rationale:

- AlbumentationsX is the actively maintained line and current docs recommend explicit OpenCV variant installation for reproducible environments.
- `resnet50.a1_in1k` is a strong, public, easy-to-reference ResNet50 checkpoint. The model card documents `timm.create_model('resnet50.a1_in1k', pretrained=True)` and model-specific preprocessing via `resolve_model_data_config`.
- Resolving preprocessing from the model prevents silent metric drift when image size, crop ratio, interpolation, mean, or std differ from checkpoint expectations.

References:

- AlbumentationsX installation: https://albumentations.ai/docs/1-introduction/installation/
- AlbumentationsX reproducibility: https://albumentations.ai/docs/4-advanced-guides/reproducibility/
- ResNet50 timm model card: https://huggingface.co/timm/resnet50.a1_in1k

## Locked Experimental Decisions

### Dataset Splits

Use ImageNet-val only. For every class:

- 25 images go to `public`.
- 25 images go to `private`.
- Public is further split into:
  - 20 images per class for selector training, total 20,000.
  - 5 images per class for selector validation and top-k tuning, total 5,000.

Private remains 25,000 images and is touched only after the selector training procedure and TTA selection rule are frozen.

Rationale:

- The 25/25 class split is exactly stratified because ImageNet-val has 50 images per class.
- A nested public train/val split is needed to tune selector loss, checkpoint choice, top-k, and optional weighting without leaking private results.

### TTA Candidate Set

Use exactly 100 candidates:

- Candidate `aug_000` is identity, meaning the clean image.
- Candidates `aug_001` through `aug_099` are single AlbumentationsX transforms.
- No candidate is a composition of multiple transforms.
- Every candidate is deterministic:
  - `p=1.0`.
  - Numeric ranges are collapsed to a single value, for example `(10, 10)`.
  - AlbumentationsX `Compose(..., seed=20260522)` is still used and serialized for audit.
  - Transforms whose output depends on uncontrolled random placement are excluded from the main candidate set.

Rationale:

- Identity should be included because TTA without the clean prediction can degrade accuracy and makes comparison to no TTA less direct.
- Single-transform candidates make the learned selector interpretable: each output answers "does this one image benefit from this one transform?"
- Deterministic candidates make cached logits, selector targets, and article tables reproducible.

Candidate family allocation:

- 1 identity candidate.
- 18 mild geometric candidates: flips, fixed rotations, safe rotations, small shifts, small scales, small shears.
- 35 color and tone candidates: brightness, contrast, gamma, HSV, PlanckianJitter, grayscale, sepia, CLAHE, equalize, autocontrast, solarize, posterize.
- 16 blur and sharpness candidates: blur, Gaussian blur, median blur, motion blur, defocus, sharpen, unsharp mask, emboss, enhance.
- 12 noise and compression candidates: Gaussian noise, additive channel noise, ISO noise, shot noise, JPEG/WebP compression.
- 10 resolution and resampling candidates: downscale and interpolation variants.
- 8 stress candidates: mild channel dropout, channel shuffle, dithering, halftone, invert, ringing overshoot.

The first implementation must materialize this as `configs/augmentations/imagenet100.yaml`. The config file is the source of truth and must be validated by tests before any expensive inference runs.

Implementation note: AlbumentationsX 2.3 validates `PlanckianJitter` ranges by requiring the range to include its white temperature, `6000K`. The initial registry therefore represents the warm 5000K idea as a deterministic seeded candidate with `temperature_range: [5000, 6000]`, plus adjacent daylight/cool ranges, instead of an invalid point range `[5000, 5000]`.

### Augmentation Order

For every image and every candidate:

1. Decode image as RGB uint8.
2. Apply exactly one AlbumentationsX candidate to the decoded RGB image.
3. Convert the augmented image to PIL or tensor format expected by the `timm` preprocessing transform.
4. Apply `timm` evaluation preprocessing.
5. Run ResNet50 inference.

For geometry candidates:

- Use OpenCV linear interpolation unless the transform requires another interpolation.
- Use reflected borders where the transform supports it, to avoid black triangles becoming a confounder.
- Do not crop away rotated borders in the default candidate set, because timm preprocessing already performs resize/crop and changing field of view would mix rotation with crop.

Rationale:

- Applying AlbumentationsX before model preprocessing matches real TTA usage.
- Applying transforms after normalization would make many image-space transforms invalid or hard to interpret.

### Teacher Cache

For each split image and candidate, store:

- `image_id`
- `class_idx`
- `split`
- `aug_id`
- `logits_fp16` for 1000 classes
- `prob_true_fp32`
- `nll_true_fp32`
- `is_top1`
- `is_top5`

Storage format:

- Metadata in parquet.
- Logits in sharded `.npy` or `.npz` files, fp16.
- One shard per split and candidate group, with manifest files that map rows back to ImageNet image ids.

Rationale:

- The selector trains only on 100 target scores, but storing full logits makes TTA aggregation, ablations, calibration analysis, and article plots much cheaper than rerunning ResNet50.
- fp16 logits are accurate enough for ranking and probability aggregation after converting to fp32 during analysis.

### Selector Target

Raw teacher loss for image `i` and augmentation `a`:

```text
nll(i, a) = -log softmax(logits(i, a))[true_class(i)]
```

Clean loss:

```text
nll_clean(i) = nll(i, aug_000)
```

Selector training target:

```text
gain(i, a) = nll_clean(i) - nll(i, a)
```

Interpretation:

- Positive gain means the augmentation improved the teacher's probability for the true class compared with clean inference.
- Negative gain means the augmentation hurt the teacher.
- `gain(i, aug_000) = 0` by definition.

Before training, standardize each augmentation target using public-train statistics:

```text
target_z(i, a) = (gain(i, a) - mean_gain_train(a)) / std_gain_train(a)
```

Rationale:

- Predicting absolute NLL makes the task dominated by image difficulty. Relative gain focuses the selector on the question we actually care about: which augmentation is useful for this image.
- Per-augmentation standardization prevents high-variance transforms from dominating the regression loss.

### Selector Model

Use a small custom CNN trained from scratch:

- Input: clean image after the same resize/center-crop geometry as the teacher, normalized with ImageNet mean/std.
- Output: 100 real-valued scores, one per candidate.
- Backbone:
  - Stem conv, stride 2.
  - Four depthwise-separable convolution stages with channels 32, 64, 128, 192.
  - Global average pooling.
  - MLP head with dropout and 100 outputs.
- Target size: under 1.5M parameters.

Training objective:

```text
loss = SmoothL1(predicted_target_z, target_z) + 0.2 * pairwise_rank_loss
```

Pairwise rank loss samples augmentation pairs within the same image and penalizes inversions where the model ranks a worse-gain augmentation above a better-gain augmentation.

Rationale:

- The downstream task is ranking candidate augmentations, not classifying ImageNet labels.
- A custom small CNN avoids relying on another heavy pretrained model and keeps the paper claim clean.
- SmoothL1 is stable for noisy teacher targets. The auxiliary ranking term aligns training with top-k selection.

### TTA Selection Rule

For each private image:

1. Run the selector once on the clean image.
2. Convert predicted standardized outputs back to predicted gain.
3. Always include `aug_000` identity.
4. Select top-k non-identity candidates by predicted gain.
5. Run ResNet50 only on identity plus the selected candidates.
6. Aggregate probabilities with uniform averaging.

Tune `k` only on public validation over:

```text
k in {1, 2, 4, 8, 16}
```

The selected `k` is frozen before private evaluation.

Main method name:

```text
learned_topk_uniform
```

Rationale:

- Uniform averaging is less sensitive to score calibration errors than weighting probabilities by predicted gain.
- The selector's ranking should be more reliable than its absolute score scale.
- Tuning `k` on public validation gives a fair compute/quality operating point without private leakage.

Weighted TTA is an ablation, not the main method:

```text
learned_topk_softmax_weighted
```

### TTA Aggregation

Main aggregation:

```text
p_final = mean(softmax(logits_aug), over selected augmentations)
```

Metrics are computed from `p_final`.

Rationale:

- Probability averaging is the standard, stable ensemble baseline for classification TTA.
- Logit averaging can over-sharpen distributions and hurt log loss, which is one of the target metrics.

### Metrics

Primary metrics:

- top-1 accuracy
- top-5 accuracy
- negative log likelihood

Secondary metrics:

- expected calibration error
- average number of ResNet50 forwards per image
- relative compute versus all-100 TTA
- selector recall of oracle top-k augmentations
- per-augmentation mean gain and selection frequency

Rationale:

- Accuracy shows classification improvement.
- NLL shows whether TTA improves confidence on the correct class.
- Compute metrics are needed because the core claim is better TTA under fewer ResNet50 calls.

### Baselines

Final private table must include:

- `clean`: identity only.
- `fixed_light_tta`: identity plus a small fixed hand-picked set, using the same final `k`.
- `random_topk`: identity plus random non-identity candidates, averaged over 5 seeds.
- `all_100_uniform`: all candidates, upper compute bound.
- `learned_topk_uniform`: main method.
- `learned_topk_softmax_weighted`: ablation.
- `oracle_topk_uniform`: private diagnostic only, not a deployable method.

Rationale:

- `clean` shows the base teacher.
- `fixed_light_tta` is the normal "picked from the air" TTA baseline.
- `random_topk` checks whether gains come from selection or just more views.
- `all_100_uniform` shows the expensive upper bound.
- `oracle_topk_uniform` estimates headroom but must not be described as a valid deployable method.

## Target File Structure

Create this structure:

```text
configs/
  experiment/resnet50_a1_in1k.yaml
  augmentations/imagenet100.yaml
src/learned_tta/
  __init__.py
  augmentations.py
  cache.py
  cli.py
  data.py
  imagenet_split.py
  metrics.py
  selector_model.py
  targets.py
  teacher.py
  train_selector.py
  tta_eval.py
tests/
  test_augmentations.py
  test_imagenet_split.py
  test_targets.py
  test_tta_eval.py
artifacts/
  README.md
```

Responsibilities:

- `configs/experiment/resnet50_a1_in1k.yaml`: paths, model name, seeds, batch sizes, split sizes, selector hyperparameters, candidate config path.
- `configs/augmentations/imagenet100.yaml`: exact 100-candidate registry.
- `src/learned_tta/augmentations.py`: load, validate, instantiate, serialize AlbumentationsX candidates.
- `src/learned_tta/imagenet_split.py`: create deterministic stratified public/private/public-train/public-val manifests.
- `src/learned_tta/data.py`: datasets and dataloaders for clean images and augmented teacher inference.
- `src/learned_tta/teacher.py`: load `timm` ResNet50 and model-specific preprocessing.
- `src/learned_tta/cache.py`: write/read metadata parquet and logits shards.
- `src/learned_tta/targets.py`: compute NLL, gain, standardization stats, and selector targets.
- `src/learned_tta/selector_model.py`: define small CNN.
- `src/learned_tta/train_selector.py`: train selector and save checkpoints.
- `src/learned_tta/tta_eval.py`: evaluate clean, fixed, random, all-100, learned, and oracle TTA.
- `src/learned_tta/metrics.py`: top-1, top-5, NLL, ECE, compute summaries.
- `src/learned_tta/cli.py`: command entry points.
- `tests/*`: unit tests for determinism, target math, split math, and aggregation.
- `artifacts/README.md`: documents generated files that stay out of git.

## Work Plan

### Task 1: Bootstrap Project

Files:

- Create `pyproject.toml`
- Create `src/learned_tta/__init__.py`
- Create `configs/experiment/resnet50_a1_in1k.yaml`
- Create `artifacts/README.md`

Steps:

- [ ] Add project metadata and dependencies: PyTorch, timm, albumentationsx headless, opencv-python-headless, numpy, pandas, pyarrow, pillow, tqdm, pydantic, pytest, ruff.
- [ ] Add config with model name `resnet50.a1_in1k`, seed `20260522`, split sizes `25/25` and `20/5`, candidate count `100`, and top-k grid `{1,2,4,8,16}`.
- [ ] Add artifact README explaining that ImageNet images, logits caches, checkpoints, and result tables are not committed.
- [ ] Run `pytest` and confirm the empty suite is discovered cleanly once tests exist in later tasks.

### Task 2: ImageNet Split Manifests

Files:

- Create `src/learned_tta/imagenet_split.py`
- Create `tests/test_imagenet_split.py`

Steps:

- [ ] Implement class-index discovery from ImageNet-val directory layout.
- [ ] Implement deterministic per-class shuffle with seed `20260522`.
- [ ] Emit four manifests: `public_train`, `public_val`, `public`, `private`.
- [ ] Test that each class has 20 public-train, 5 public-val, 25 public total, and 25 private images.
- [ ] Test that splits are disjoint and stable across repeated runs.

### Task 3: Augmentation Registry

Files:

- Create `configs/augmentations/imagenet100.yaml`
- Create `src/learned_tta/augmentations.py`
- Create `tests/test_augmentations.py`

Steps:

- [ ] Define exactly 100 candidate ids, with `aug_000` identity and 99 single AlbumentationsX transforms.
- [ ] Validate every candidate has one transform only, except identity.
- [ ] Validate every non-identity candidate has `p=1.0`.
- [ ] Validate the registry contains no duplicate ids and no duplicate transform specs.
- [ ] Validate applying every candidate twice to a fixed sample image gives byte-identical output.
- [ ] Serialize the loaded AlbumentationsX transforms for audit.

### Task 4: Teacher Inference Cache

Files:

- Create `src/learned_tta/teacher.py`
- Create `src/learned_tta/data.py`
- Create `src/learned_tta/cache.py`

Steps:

- [ ] Load `timm.create_model('resnet50.a1_in1k', pretrained=True)` in eval mode.
- [ ] Resolve teacher preprocessing from the model data config.
- [ ] Build dataloaders that apply one candidate per image and batch tensors for the teacher.
- [ ] Save logits as fp16 shards and metadata as parquet.
- [ ] Add a resume mode that skips completed `(split, aug_id)` shards after validating shape and row count.
- [ ] Run a smoke inference on 2 classes, 2 images per class, and 3 candidates before full ImageNet-val inference.

### Task 5: Target Generation

Files:

- Create `src/learned_tta/targets.py`
- Create `tests/test_targets.py`

Steps:

- [ ] Compute true-class probability and NLL from cached logits.
- [ ] Compute clean NLL from `aug_000`.
- [ ] Compute per-candidate gain as `clean_nll - aug_nll`.
- [ ] Compute public-train mean and std per candidate.
- [ ] Save selector target matrices for public-train and public-val.
- [ ] Test target math with a small hand-written logits tensor where the expected NLL and gain are known.

### Task 6: Selector CNN

Files:

- Create `src/learned_tta/selector_model.py`
- Create `src/learned_tta/train_selector.py`

Steps:

- [ ] Implement the depthwise-separable CNN with under 1.5M parameters.
- [ ] Train on public-train clean images and standardized gain targets.
- [ ] Use SmoothL1 plus pairwise rank loss.
- [ ] Track public-val regression loss, Spearman correlation, oracle top-k recall, and resulting learned TTA metrics from cached logits.
- [ ] Save best checkpoint by public-val NLL of `learned_topk_uniform`, not by raw regression loss.

### Task 7: TTA Evaluation

Files:

- Create `src/learned_tta/tta_eval.py`
- Create `src/learned_tta/metrics.py`
- Create `tests/test_tta_eval.py`

Steps:

- [ ] Implement probability averaging from cached logits.
- [ ] Implement `clean`, `fixed_light_tta`, `random_topk`, `all_100_uniform`, `learned_topk_uniform`, `learned_topk_softmax_weighted`, and `oracle_topk_uniform`.
- [ ] Tune `k` on public-val and freeze the winning value.
- [ ] Evaluate the frozen method on private.
- [ ] Test that aggregation returns correct top-1, top-5, and NLL on a small synthetic logits cache.

### Task 8: Results and Article Artifacts

Files:

- Create `reports/resnet50_a1_in1k/results.md`
- Create `reports/resnet50_a1_in1k/tables/private_metrics.csv`
- Create `reports/resnet50_a1_in1k/tables/augmentation_impact.csv`

Steps:

- [ ] Generate public-val and private metric tables.
- [ ] Generate compute table with average ResNet50 forwards per image.
- [ ] Generate augmentation impact table with mean gain, selection frequency, and oracle frequency.
- [ ] Generate plots for gain distribution and learned versus oracle top-k overlap.
- [ ] Write a short result summary that avoids claiming SOTA until additional architectures are run.

## Full Run Order

After implementation, run in this order:

```bash
python -m learned_tta.cli make-splits --config configs/experiment/resnet50_a1_in1k.yaml
python -m learned_tta.cli validate-augmentations --config configs/experiment/resnet50_a1_in1k.yaml
python -m learned_tta.cli cache-teacher --split public --config configs/experiment/resnet50_a1_in1k.yaml
python -m learned_tta.cli build-targets --config configs/experiment/resnet50_a1_in1k.yaml
python -m learned_tta.cli train-selector --config configs/experiment/resnet50_a1_in1k.yaml
python -m learned_tta.cli tune-tta --split public_val --config configs/experiment/resnet50_a1_in1k.yaml
python -m learned_tta.cli cache-teacher --split private --config configs/experiment/resnet50_a1_in1k.yaml
python -m learned_tta.cli evaluate-private --config configs/experiment/resnet50_a1_in1k.yaml
python -m learned_tta.cli build-report --config configs/experiment/resnet50_a1_in1k.yaml
```

## Expected Compute Shape

Teacher cache:

- Public all candidates: 25,000 images times 100 candidates = 2,500,000 ResNet50 forwards.
- Private all candidates, for full baselines and oracle diagnostics: 25,000 images times 100 candidates = 2,500,000 ResNet50 forwards.

Deployed learned TTA path:

- One selector forward per image.
- `1 + k` ResNet50 forwards per image, because identity is always included.

If tuned `k=4`, learned TTA uses 5 ResNet50 forwards per image instead of 100.

## Risks and Mitigations

- Risk: The selector predicts image difficulty rather than augmentation usefulness.
  - Mitigation: train on relative gain, not raw NLL.
- Risk: The selector's absolute score scale is poorly calibrated.
  - Mitigation: main method uses top-k ranking with uniform averaging.
- Risk: Private leakage through tuning.
  - Mitigation: tune only on public-val; private is used once for final tables.
- Risk: Some transforms silently behave randomly.
  - Mitigation: deterministic byte-output test over all 100 candidates.
- Risk: All-100 private cache is expensive.
  - Mitigation: cache is still needed for fair baselines and oracle analysis; deployed learned path remains cheap.
- Risk: TTA hurts NLL while helping accuracy, or the reverse.
  - Mitigation: report top-1, top-5, and NLL separately.
- Risk: The first ResNet50 result does not justify "SOTA" language.
  - Mitigation: document it as a ResNet50 case study until the method is repeated on additional architectures.

## Completion Criteria

The implementation is complete when:

- The augmentation registry has exactly 100 deterministic candidates.
- Public/private manifests are stratified and reproducible.
- Teacher logits are cached for all public candidates.
- Selector targets are generated from relative gain.
- The selector trains and selects top-k candidates on public-val.
- Private evaluation reports all required baselines and metrics.
- The report includes both quality and compute tables.

## Self-Review Notes

- Spec coverage: the document covers AlbumentationsX choice, 100 unmixed augmentations, ResNet50 teacher, ImageNet-val split, public teacher inference, 100-score selector CNN, learned TTA selection, private evaluation, baselines, metrics, and article artifacts.
- Placeholder scan: no open placeholders are intentionally left in the plan.
- Type consistency: the score is consistently named `gain`; the model output is 100 real-valued selector scores; `aug_000` is consistently identity.
