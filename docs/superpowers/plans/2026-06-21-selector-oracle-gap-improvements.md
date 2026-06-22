# Selector Oracle Gap Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the learned TTA selector so it captures more of the measured ResNet50 oracle top-k headroom while keeping the workflow reproducible and reportable.

**Architecture:** Add lightweight analysis and training primitives around the existing teacher-cache artifacts instead of changing cache generation. The implementation produces stronger selector inputs, richer gain targets, listwise/ranking losses, small-model ablations, compute-policy calibration, and report sections that quantify how much oracle headroom is captured.

**Tech Stack:** Python, PyTorch, NumPy, pytest, existing `learned_tta` CLI/reporting modules.

## 2026-06-22 Follow-Up Status

- [x] Added marginal-logit ensemble contribution targets for pairwise training.
- [x] Added top-k KL listwise loss focused on selected augmentations.
- [x] Added fitted PCA/whitening projection for pretrained selector features.
- [x] Added arbitrary confidence-bucket policy calibration and JSON-backed evaluation.
- [x] Tuned selected-logit softmax temperature on public-val and refreshed private results.

Outcome: the new target/projection/listwise training variants did not beat the
existing public-val-selected `pairwise_top1_delta` checkpoint on top-1, so the
selected model stays unchanged. The useful improvement came from aggregation
tuning: public-val selected `score_temperature=0.25`, which improves private
k=8 softmax-weighted top-1 from 0.82912 to 0.83128 at the same 9 forwards/image.
The confidence-bucket policy is useful as a lower-compute ablation
(0.82760 top-1 at 6.93 forwards/image), not as the primary result.

---

## File Structure

- `src/learned_tta/selector_features.py`: create clean-logit feature extraction from cached clean logits.
- `src/learned_tta/selector_targets.py`: create alternate selector target matrices from clean and augmented logits.
- `src/learned_tta/selector_analysis.py`: create oracle-gap and target diagnostic summaries.
- `src/learned_tta/selector_training.py`: extend selector training/ablation specs for feature modes, target modes, and listwise loss.
- `src/learned_tta/tta_tuning.py`: extend tuning diagnostics with oracle-gap capture metrics.
- `src/learned_tta/report_builder.py`: copy new selector artifacts and render a compact report section.
- `src/learned_tta/cli.py`: add/extend CLI entry points for analysis, feature/target export, and richer ablations.
- `tests/test_selector_features.py`: unit tests for clean-logit features.
- `tests/test_selector_targets.py`: unit tests for target construction.
- `tests/test_selector_analysis.py`: unit tests for oracle-gap summaries.
- `tests/test_selector_training.py`: extend ablation tests.
- `tests/test_tta_tuning.py`: extend diagnostics tests.
- `tests/test_report_builder.py`: extend report-copy/render tests.

## Task 1: Oracle Gap Analysis

**Files:**
- Create: `src/learned_tta/selector_analysis.py`
- Modify: `src/learned_tta/cli.py`
- Test: `tests/test_selector_analysis.py`

- [x] **Step 1: Write failing test for oracle-gap summary**

```python
def test_oracle_gap_summary_quantifies_clean_oracle_and_learned_capture(tmp_path):
    from learned_tta.selector_analysis import summarize_oracle_gap

    summary = summarize_oracle_gap(
        clean_top1=0.80,
        oracle_top1=0.87,
        learned_top1=0.815,
        clean_nll=0.90,
        oracle_nll=0.50,
        learned_nll=0.75,
        forwards_per_image=17.0,
    )

    assert summary["top1_oracle_delta_pp"] == 7.0
    assert summary["top1_learned_delta_pp"] == 1.5
    assert summary["top1_oracle_capture"] == pytest.approx(0.2142857)
    assert summary["nll_oracle_delta"] == pytest.approx(-0.40)
    assert summary["nll_learned_delta"] == pytest.approx(-0.15)
    assert summary["forwards_per_image"] == 17.0
```

- [x] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_selector_analysis.py::test_oracle_gap_summary_quantifies_clean_oracle_and_learned_capture -q`

Expected: fails because `learned_tta.selector_analysis` does not exist.

- [x] **Step 3: Implement minimal oracle-gap summary**

Implement `summarize_oracle_gap(...) -> dict[str, float]` with percentage-point deltas and capture ratios.

- [x] **Step 4: Run targeted test and verify GREEN**

Run: `uv run pytest tests/test_selector_analysis.py -q`

Expected: pass.

## Task 2: Stronger Clean-Pass Features

**Files:**
- Create: `src/learned_tta/selector_features.py`
- Modify: `src/learned_tta/selector_training.py`
- Test: `tests/test_selector_features.py`

- [x] **Step 1: Write failing tests for clean-logit feature extraction**

```python
def test_clean_logit_features_include_margin_entropy_and_topk():
    from learned_tta.selector_features import clean_logit_features

    logits = np.array([[4.0, 2.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    features, names = clean_logit_features(logits, top_k=2)

    assert names[:4] == ["clean_confidence", "clean_margin", "clean_entropy", "clean_pred_class"]
    assert "clean_top1_prob" in names
    assert "clean_top2_prob" in names
    assert features.shape == (2, len(names))
    assert features[0, names.index("clean_margin")] > features[1, names.index("clean_margin")]
```

- [x] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_selector_features.py::test_clean_logit_features_include_margin_entropy_and_topk -q`

Expected: fails because `selector_features` does not exist.

- [x] **Step 3: Implement clean-logit features**

Implement NumPy softmax, confidence, margin, entropy, predicted class index, and top-k probabilities in `selector_features.py`.

- [x] **Step 4: Wire feature mode into selector training spec**

Add `feature_mode: Literal["image", "clean_logits"] = "image"` to selector ablation specs, without changing the existing default behavior.

- [x] **Step 5: Run targeted tests**

Run: `uv run pytest tests/test_selector_features.py tests/test_selector_training.py -q`

Expected: pass.

## Task 3: Richer Selector Targets

**Files:**
- Create: `src/learned_tta/selector_targets.py`
- Modify: `src/learned_tta/selector_training.py`
- Test: `tests/test_selector_targets.py`

- [x] **Step 1: Write failing tests for target modes**

```python
def test_selector_targets_build_gain_logit_margin_and_top1_fix():
    from learned_tta.selector_targets import build_selector_targets

    clean_logits = np.array([[2.0, 0.0], [0.1, 1.0]], dtype=np.float32)
    aug_logits = np.array([[[0.0, 3.0], [3.0, 0.0]], [[0.2, 1.2], [1.5, 0.5]]], dtype=np.float32)
    labels = np.array([1, 0], dtype=np.int64)

    targets = build_selector_targets(clean_logits, aug_logits, labels)

    assert set(targets) >= {"nll_gain", "true_logit_gain", "margin_gain", "top1_fix"}
    assert targets["nll_gain"].shape == (2, 2)
    assert targets["top1_fix"][0, 0] == 1.0
    assert targets["top1_fix"][0, 1] == 0.0
```

- [x] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_selector_targets.py::test_selector_targets_build_gain_logit_margin_and_top1_fix -q`

Expected: fails because `selector_targets` does not exist.

- [x] **Step 3: Implement target construction**

Implement `build_selector_targets(clean_logits, aug_logits, labels)` returning NLL gain, true-class logit gain, true-class margin gain, and top-1 fix targets.

- [x] **Step 4: Add target mode to selector ablation spec**

Add `target_mode: Literal["nll_gain", "true_logit_gain", "margin_gain", "top1_fix"] = "nll_gain"` and keep old behavior as default.

- [x] **Step 5: Run targeted tests**

Run: `uv run pytest tests/test_selector_targets.py tests/test_selector_training.py -q`

Expected: pass.

## Task 4: Loss Closer To Top-K Selection

**Files:**
- Modify: `src/learned_tta/selector_training.py`
- Test: `tests/test_selector_training.py`

- [x] **Step 1: Write failing test for listwise top-k loss**

```python
def test_selector_loss_ablation_includes_listwise_variant():
    from learned_tta.selector_training import DEFAULT_SELECTOR_LOSS_ABLATIONS

    names = {spec.name for spec in DEFAULT_SELECTOR_LOSS_ABLATIONS}
    assert "gain_listwise_topk" in names
```

- [x] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_selector_training.py::test_selector_loss_ablation_includes_listwise_variant -q`

Expected: fails because the listwise variant is not present.

- [x] **Step 3: Implement listwise loss option**

Add `listwise_weight` and `listwise_top_k` to `SelectorLossAblationSpec`, and implement a differentiable loss that compares predicted top-k mass with target top-k membership.

- [x] **Step 4: Run targeted tests**

Run: `uv run pytest tests/test_selector_training.py -q`

Expected: pass.

## Task 5: Compare Multiple Small Models

**Files:**
- Modify: `src/learned_tta/selector_training.py`
- Modify: `src/learned_tta/cli.py`
- Test: `tests/test_selector_training.py`

- [x] **Step 1: Write failing test for feature/model ablation table columns**

```python
def test_selector_ablation_table_records_feature_and_target_modes(tmp_path):
    # Use existing small synthetic artifact helper in tests/test_selector_training.py.
    summary = _run_tiny_selector_ablation(tmp_path)
    text = summary.summary_csv.read_text()
    assert "feature_mode" in text
    assert "target_mode" in text
    assert "model_family" in text
```

- [x] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_selector_training.py::test_selector_ablation_table_records_feature_and_target_modes -q`

Expected: fails because columns are missing.

- [x] **Step 3: Extend ablation variants**

Add variants for image CNN, clean-logit MLP, and gain-listwise clean-logit MLP while keeping the original variants for comparison.

- [x] **Step 4: Run targeted tests**

Run: `uv run pytest tests/test_selector_training.py -q`

Expected: pass.

## Task 6: Compute Policy Calibration

**Files:**
- Modify: `src/learned_tta/tta_tuning.py`
- Modify: `src/learned_tta/report_builder.py`
- Test: `tests/test_tta_tuning.py`

- [x] **Step 1: Write failing test for compute-policy frontier output**

```python
def test_tune_tta_writes_compute_policy_frontier(tmp_path):
    summary = _run_tiny_tta_tuning(tmp_path)
    assert summary.compute_policy_frontier_path is not None
    assert summary.compute_policy_frontier_path.exists()
    assert "top1_oracle_capture" in summary.compute_policy_frontier_path.read_text()
```

- [x] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_tta_tuning.py::test_tune_tta_writes_compute_policy_frontier -q`

Expected: fails because the summary field and file do not exist.

- [x] **Step 3: Implement compute frontier CSV**

Write `public_val_compute_policy_frontier.csv` with strategy, forwards/image, top-1, NLL, top-1 delta, and oracle-capture fields for tuned strategies.

- [x] **Step 4: Run targeted tests**

Run: `uv run pytest tests/test_tta_tuning.py -q`

Expected: pass.

## Task 7: Success Metric And Report Integration

**Files:**
- Modify: `src/learned_tta/report_builder.py`
- Test: `tests/test_report_builder.py`

- [x] **Step 1: Write failing report test**

```python
def test_report_includes_oracle_capture_and_next_goal(tmp_path):
    summary = _build_tiny_report(tmp_path)
    text = summary.results_markdown.read_text()
    assert "Oracle Gap Capture" in text
    assert "next target" in text.lower()
```

- [x] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_report_builder.py::test_report_includes_oracle_capture_and_next_goal -q`

Expected: fails because the section is absent.

- [x] **Step 3: Render report section**

Add an `Oracle Gap Capture` section that states current learned capture, oracle ceiling, and the next target of `+1.5...2.0 pp` at `17 forwards/image`.

- [x] **Step 4: Run targeted tests**

Run: `uv run pytest tests/test_report_builder.py -q`

Expected: pass.

## Task 8: Refresh Artifacts And Final Verification

**Files:**
- Modify: `reports/resnet50_a1_in1k/results.md`
- Modify: `reports/resnet50_a1_in1k/tables/*.csv`
- Test: full suite

- [x] **Step 1: Run refreshed analysis and report build**

Run:

```bash
uv run python -m learned_tta.cli tune-tta --config configs/experiment/resnet50_a1_in1k.yaml --device mps --batch-size 64 --num-workers 4
uv run python -m learned_tta.cli train-selector-ablation --config configs/experiment/resnet50_a1_in1k.yaml --device mps --batch-size 64 --num-workers 4 --epochs 5
uv run python -m learned_tta.cli build-report --config configs/experiment/resnet50_a1_in1k.yaml --device mps --batch-size 64 --num-workers 4
```

- [x] **Step 2: Run full verification**

Run:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src/learned_tta/selector_analysis.py src/learned_tta/selector_features.py src/learned_tta/selector_targets.py src/learned_tta/selector_training.py src/learned_tta/tta_tuning.py src/learned_tta/report_builder.py src/learned_tta/cli.py tests/test_selector_analysis.py tests/test_selector_features.py tests/test_selector_targets.py tests/test_selector_training.py tests/test_tta_tuning.py tests/test_report_builder.py
```

- [x] **Step 3: Commit and push**

Run:

```bash
git add docs/superpowers/plans/2026-06-21-selector-oracle-gap-improvements.md src/learned_tta tests reports configs
git commit -m "Improve selector oracle gap diagnostics"
git push
```

Expected: branch `codex/adaptive-selector-loss` contains implementation, refreshed report, and passing tests.

## Self-Review

- Spec coverage: the eight user-facing goals are covered by Tasks 1-8.
- Placeholder scan: no implementation task depends on an unspecified future artifact.
- Type consistency: feature mode, target mode, model family, listwise loss, and oracle-capture names are defined before report integration.

## 2026-06-21 Pairwise Policy Follow-Up

The follow-up 5-point implementation focused on the pairwise augmentation
ranker, not the earlier image-CNN selector baseline:

- [x] Stronger optional image features: pairwise bundles can append cached
  pretrained image features and deterministically random-project them for compact
  MLP inputs.
- [x] Listwise top-k objective: pairwise training supports an additional
  per-image target-top-k membership cross-entropy.
- [x] Confidence-aware policy/evaluation: evaluation can lower selected k for
  high-confidence clean predictions and reports actual forwards/image.
- [x] Per-image selected-logit weighting: pairwise evaluation now reports
  softmax-weighted aggregation over selected logits in addition to uniform TTA.
- [x] Hard-example mining: pairwise training can upweight clean-wrong or
  low-confidence images.

Empirically, the new listwise/hard-example training variants did not beat the
existing public-val-selected top-1-delta checkpoint. The useful improvement came
from inference policy: the existing checkpoint with softmax-weighted selected
logits reaches 0.82828 private top-1 at k=16, and the best observed private
trade-off is k=8 softmax-weighted with 0.82912 top-1 at 9 forwards/image.
