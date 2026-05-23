# albu-tta

[![pytest](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=pytest&label=pytest)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![ruff](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=ruff&label=ruff)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![ty](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=ty&label=ty)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![coverage](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=coverage&label=coverage)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Learned test-time augmentation selector experiments with AlbumentationsX.

## Methods

The project now keeps two TTA learning layers separate.

- `learned_topk_uniform` and `learned_topk_softmax_weighted`: an image-conditioned
  selector CNN predicts which augmentations are worth running for each image.
- `global_weighted_tta`: a paper-style second-level aggregator learns one sparse
  non-negative weight per augmentation from cached public-val predictions.
- `class_weighted_tta`: the same learned aggregation idea, but with separate
  sparse non-negative augmentation weights per output class.
- `xgboost_multiclass`: an optional tabular stacker over flattened per-policy
  probabilities. It is trained from cached public-val predictions and evaluated
  on private when its artifact is present.

This separation makes the article diagnostics cleaner: selection answers which
AlbumentationsX transforms are useful, while aggregation answers how strongly to
combine predictions once the TTA views are available.

The selector is trained on standardized gain targets, but checkpoints persist
the public-train target mean and std. Inference converts selector outputs back
to the original gain scale before ranking augmentations for top-k TTA. During
training, the best checkpoint is selected by public-validation
`learned_topk_uniform` TTA NLL when the validation teacher cache is available;
regression loss, Spearman correlation, TTA metrics, and oracle top-k recall are
written to `selector/selector_history.csv`.

`build-report` writes the aggregation diagnostics when the learned aggregator
artifacts are present, and copies private clean-vs-TTA diagnostics when
`evaluate-private` has produced them:

```text
reports/resnet50_a1_in1k/tables/aggregation_weights.csv
reports/resnet50_a1_in1k/tables/class_augmentation_weights.csv
reports/resnet50_a1_in1k/tables/xgboost_feature_importance.csv
reports/resnet50_a1_in1k/tables/corrections.csv
reports/resnet50_a1_in1k/tables/selector_history.csv
reports/resnet50_a1_in1k/figures/gain_distribution.svg
reports/resnet50_a1_in1k/figures/oracle_overlap.svg
reports/resnet50_a1_in1k/figures/aggregation_weights.svg
reports/resnet50_a1_in1k/figures/xgboost_feature_importance.svg
reports/resnet50_a1_in1k/figures/corrections.svg
reports/resnet50_a1_in1k/figures/selector_history.svg
```

`aggregation_weights.csv` is the compact table for the article: global weight,
active flag, mean class weight, max class weight, and class activation frequency
per augmentation. The class-level long table is kept for deeper diagnosis of
which ImageNet classes benefit from which AlbumentationsX transforms.
`public_metrics.csv` combines the tuned public-val `learned_topk_uniform` result
with public-val metrics saved inside the optional learned aggregation artifacts,
so global, class-specific, and XGBoost stackers can be compared before private
evaluation.
`compute.csv` keeps compute rows for both `public_val` and `private` with an
explicit `split` column, so public tuning diagnostics and final private compute
costs are not mixed implicitly.
Aggregator training uses the historical `--l1-penalty` CLI option as a sparsity
regularizer and then prunes weights at or below `active_threshold`; this makes
zero-weight TTA candidates explicit in the saved artifact and report tables.
The optional XGBoost stacker is deliberately not a default dependency; install
the stacker extra with `uv sync --extra stackers` before running
`--method xgboost-multiclass`.
When the XGBoost artifact is present, `build-report` writes
`xgboost_feature_importance.csv` and `xgboost_feature_importance.svg` to show
which augmentation candidates the stacker uses most.
`corrections.csv` counts where a strategy fixes a clean ResNet50 mistake and
where it breaks an originally correct clean prediction. This follows the TTA
diagnostic framing from "Better Aggregation in Test-Time Augmentation": average
dataset gain is not enough, because TTA can help and hurt different images.

## Smoke Run

Use the synthetic smoke run before spending GPU time on ImageNet. It creates a tiny
ImageNet-like directory, caches a fake teacher, trains the selector for one epoch,
tunes TTA, trains global and class-specific non-negative aggregators, evaluates
private metrics, and writes `results.md`.

```bash
uv run python -m learned_tta.cli run-smoke \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --output-dir artifacts/smoke \
  --candidate-count 3 \
  --image-size 16 \
  --batch-size 2 \
  --num-workers 0 \
  --epochs 1
```

Expected final artifact:

```text
artifacts/smoke/reports/results.md
```

## Full ImageNet Run

Run the full experiment only after the smoke run passes. `--imagenet-val-dir`
must point to an ImageNet validation directory laid out as
`val/class_name/image.JPEG`.

```bash
uv run python -m learned_tta.cli validate-augmentations \
  --config configs/experiment/resnet50_a1_in1k.yaml

uv run python -m learned_tta.cli make-splits \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --imagenet-val-dir /path/to/imagenet/val

uv run python -m learned_tta.cli cache-teacher --split public_train \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --device cuda

uv run python -m learned_tta.cli cache-teacher --split public_val \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --device cuda

uv run python -m learned_tta.cli build-targets \
  --config configs/experiment/resnet50_a1_in1k.yaml

uv run python -m learned_tta.cli train-selector \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --device cuda

uv run python -m learned_tta.cli tune-tta --split public_val \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --device cuda

uv run python -m learned_tta.cli train-aggregator --method global-nonnegative \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --split public_val \
  --device cuda

uv run python -m learned_tta.cli train-aggregator --method class-nonnegative \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --split public_val \
  --device cuda

uv sync --extra stackers

uv run python -m learned_tta.cli train-aggregator --method xgboost-multiclass \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --split public_val

uv run python -m learned_tta.cli cache-teacher --split private \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --device cuda

uv run python -m learned_tta.cli evaluate-private \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --device cuda

uv run python -m learned_tta.cli build-report \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --device cuda
```

When private evaluation artifacts live outside the report directory, pass
`--corrections /path/to/corrections.csv` to `build-report`.
