# Pairwise Augmentation Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve oracle-gap capture by training a selector that scores `(image, augmentation)` pairs with policy-aware targets instead of only predicting a fixed 100-output vector per image.

**Architecture:** Add a small MLP ranker over clean-pass uncertainty features, optional pretrained image features, and augmentation identity features. The ranker emits one score per `(image, aug)` pair; evaluation reshapes scores to `(image, aug)` and reuses existing top-k TTA metrics. Keep the existing image CNN/MLP selector path unchanged.

**Tech Stack:** PyTorch, NumPy/Pandas, existing teacher-cache shards, existing selector feature cache, existing TTA evaluation helpers.

---

### Task 1: Pairwise Ranker Dataset And Model

**Files:**
- Create: `src/learned_tta/pairwise_selector.py`
- Test: `tests/test_pairwise_selector.py`
- Modify: `src/learned_tta/cli.py`

- [x] Write failing tests for pairwise feature matrix construction from manifest, teacher cache, selector targets, clean logits, and augmentation one-hot identity.
- [x] Implement `PairwiseFeatureBundle`, `PairwiseSelectorMLP`, and `build_pairwise_feature_bundle`.
- [x] Add a small CLI command to train/evaluate pairwise selector from artifact paths.
- [x] Run targeted tests for pairwise construction and CLI argument wiring.

### Task 2: Policy-Aware Loss

**Files:**
- Modify: `src/learned_tta/pairwise_selector.py`
- Test: `tests/test_pairwise_selector.py`

- [x] Write failing tests for weighted SmoothL1 gain loss plus useful-augmentation BCE.
- [x] Implement `pairwise_policy_loss` with `usefulness_tau`, `usefulness_weight`, `positive_gain_weight`, and optional per-row weights.
- [x] Verify the loss prioritizes positive/high-gain rows without changing output shape.

### Task 3: Stronger Clean Uncertainty Features

**Files:**
- Modify: `src/learned_tta/selector_features.py`
- Modify: `src/learned_tta/pairwise_selector.py`
- Test: `tests/test_selector_features.py`
- Test: `tests/test_pairwise_selector.py`

- [x] Write failing tests for entropy, top probabilities/logits, probability margin, logit margin, and true-class confidence features.
- [x] Implement `clean_logit_uncertainty_features(logits, class_idxs, top_k=5)`.
- [x] Feed these features into the pairwise bundle while preserving existing `clean_logit_features` behavior.

### Task 4: Separate Top-1 And NLL Objectives

**Files:**
- Modify: `src/learned_tta/pairwise_selector.py`
- Test: `tests/test_pairwise_selector.py`

- [x] Write failing tests for `target_mode="nll_gain"` and `target_mode="top1_delta"`.
- [x] Implement top-1 delta targets from teacher-cache metadata/logits: `aug_is_top1 - clean_is_top1`.
- [x] Select best checkpoint by configurable `selection_metric` (`val_tta_top1` or `val_tta_nll`).
- [x] Emit a comparison CSV for NLL-ranker and top1-ranker variants.

### Task 5: Error Analysis And Report Artifacts

**Files:**
- Create: `src/learned_tta/selector_error_analysis.py`
- Modify: `src/learned_tta/report_builder.py`
- Modify: `src/learned_tta/cli.py`
- Test: `tests/test_selector_error_analysis.py`
- Test: `tests/test_report_builder.py`

- [x] Write failing tests for error buckets: clean wrong fixed, clean right broken, selected-oracle overlap, and confidence buckets.
- [x] Implement CSV output for selector error analysis from predicted scores and teacher cache.
- [x] Add report ingestion/table rendering for the pairwise comparison and error analysis CSVs.
- [x] Run full verification: `uv run ruff check .`, `uv run ty check`, `uv run pytest --cov=learned_tta --cov-report=term-missing --cov-fail-under=98`.
- [ ] Commit and push the branch.
