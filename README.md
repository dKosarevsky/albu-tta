# albu-tta

[![pytest](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=pytest&label=pytest)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![ruff](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=ruff&label=ruff)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![ty](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=ty&label=ty)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![coverage](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=coverage&label=coverage)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Learned test-time augmentation selector experiments with AlbumentationsX.

Implementation status for the planned lightweight pipeline is tracked in
[`docs/implementation-status.md`](docs/implementation-status.md).

For a Google Colab full ImageNet run, use
[`notebooks/full_imagenet_run_colab.ipynb`](notebooks/full_imagenet_run_colab.ipynb)
and the runbook in [`docs/colab-run.md`](docs/colab-run.md). The notebook is a
resumable GPU entrypoint for the full ImageNet run and uses Google Drive for
persistent `artifacts/` and `reports/`.

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

### Selector Target Formulation

The selector is an augmentation utility predictor: it predicts
100 augmentation utility scores and is not a 50-bin loss classification task.
The teacher cache keeps raw true-class NLL for every image and augmentation,
but the selector target is a relative utility:

```text
gain = clean_nll - aug_nll
```

Positive gain means the augmentation increased the teacher probability assigned
to the correct class compared with clean inference. The deployed decision is
ranking/top-k TTA selection, so the small CNN has one output per configured
candidate rather than assigning an image to a discrete loss bin.

The selector is trained on standardized gain targets, but checkpoints persist
the public-train target mean and std. Inference converts selector outputs back
to the original gain scale before ranking augmentations for top-k TTA. During
training, the best checkpoint is selected by public-validation
`learned_topk_uniform` TTA NLL when the validation teacher cache is available;
regression loss, Spearman correlation, TTA metrics, and oracle top-k recall are
written to `selector/selector_history.csv`.

### Split Contract

The code treats public/private separation as an API contract, not only a
runbook convention. Selector targets are built only from `public_train` and `public_val`.
Selector checkpoint selection, `tune-tta`, and learned aggregation training use `public_val`.
`evaluate-private` accepts only `private` and reads a frozen public-val tuning
artifact. Private oracle rows are diagnostics and upper bounds, not deployable methods or tuning inputs.

`validate-augmentations --audit-output` writes a stable JSON audit of the exact
AlbumentationsX candidate ids, transform classes, parameters, experiment seed,
serialized AlbumentationsX `Compose` payloads, and runtime package versions used
in the run:

```text
artifacts/augmentation_registry_audit.json
```

Each teacher cache shard writes a `.run.json` sidecar next to the parquet
metadata and fp16 logits. The sidecar stores the augmentation id and params,
seed, model name, pretrained flag, timm data config, class count, and storage
format. Resume checks compare this metadata before skipping an existing shard,
so changing the teacher checkpoint, preprocessing, candidate params, or seed
forces recomputation instead of silently reusing stale logits.

`build-report` writes the aggregation diagnostics when the learned aggregator
artifacts are present, and copies private clean-vs-TTA diagnostics when
`evaluate-private` has produced them:

```text
reports/resnet50_a1_in1k/tables/aggregation_weights.csv
reports/resnet50_a1_in1k/tables/class_augmentation_weights.csv
reports/resnet50_a1_in1k/tables/xgboost_feature_importance.csv
reports/resnet50_a1_in1k/tables/private_metric_deltas.csv
reports/resnet50_a1_in1k/tables/corrections.csv
reports/resnet50_a1_in1k/tables/selector_history.csv
reports/resnet50_a1_in1k/tables/transform_class_impact.csv
reports/resnet50_a1_in1k/tables/transform_class_aggregation.csv
reports/resnet50_a1_in1k/figures/gain_distribution.svg
reports/resnet50_a1_in1k/figures/oracle_overlap.svg
reports/resnet50_a1_in1k/figures/aggregation_weights.svg
reports/resnet50_a1_in1k/figures/xgboost_feature_importance.svg
reports/resnet50_a1_in1k/figures/corrections.svg
reports/resnet50_a1_in1k/figures/selector_history.svg
reports/resnet50_a1_in1k/figures/transform_class_impact.svg
reports/resnet50_a1_in1k/figures/transform_class_aggregation.svg
```

`aggregation_weights.csv` is the compact table for the article: global weight,
active flag, mean class weight, max class weight, and class activation frequency
per augmentation. The class-level long table is kept for deeper diagnosis of
which ImageNet classes benefit from which AlbumentationsX transforms.
Augmentation-level report tables include `augmentation_name` and
`transform_class` columns resolved from the registry, so impact and learned
weight tables remain interpretable without a separate id lookup.
`transform_class_impact.csv` groups mean gain, learned selection frequency, and
oracle frequency by AlbumentationsX transform class for a compact article-level
view of which transform families matter.
`results.md` also embeds compact top-N markdown tables for mean-gain
augmentations, learned selector choices, oracle choices, transform classes,
learned aggregation weights, and XGBoost feature importance, so the main
article artifact can be reviewed without opening every CSV first.
`transform_class_aggregation.csv` groups global and class-specific aggregation
weights by AlbumentationsX transform class, complementing the per-augmentation
weight table with a family-level view.
`public_metrics.csv` combines the tuned public-val `learned_topk_uniform` result
with public-val metrics saved inside the optional learned aggregation artifacts,
so global, class-specific, and XGBoost stackers can be compared before private
evaluation.
`private_metric_deltas.csv` compares each private strategy against clean
ResNet50 on top-1, top-5, NLL, ECE, and compute, making the final trade-off
table explicit.
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
which augmentation candidates the stacker uses most. The XGBoost metadata stores
the model path relative to the artifact file when both files are in the same
artifact directory, so selector artifacts can be moved as a bundle.
`corrections.csv` counts where a strategy fixes a clean ResNet50 mistake and
where it breaks an originally correct clean prediction. This follows the TTA
diagnostic framing from "Better Aggregation in Test-Time Augmentation": average
dataset gain is not enough, because TTA can help and hurt different images.
For `random_topk`, correction counts are averaged across the same random seeds
used for the private metric row.

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
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --audit-output artifacts/augmentation_registry_audit.json

uv run python -m learned_tta.cli check-full-run \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --imagenet-val-dir /path/to/imagenet/val

uv run python -m learned_tta.cli full-run-status \
  --config configs/experiment/resnet50_a1_in1k.yaml

uv run python -m learned_tta.cli full-run-status \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --format json

uv run python -m learned_tta.cli full-run-status \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --fail-on-incomplete

uv run python -m learned_tta.cli full-run-status \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --next-command

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

Use `full-run-status` between expensive steps to inspect the configured artifact
locations and print the next missing command without loading ImageNet or GPU
models. The text output separates required steps from the optional XGBoost
stacker, and `--format json` exposes the same required/optional status for
external run scripts.
The status check requires every configured augmentation candidate to have
metadata, logits, and `.run.json` sidecar shard files before a teacher-cache
split is marked complete. full-run-status treats `.run.json` sidecars as required teacher cache outputs.
Incomplete steps show `missing=` and `extra=` counts in text output; JSON
output includes `missing_outputs` and `extra_outputs` path lists for resumable
run scripts.
Use `--fail-on-incomplete` when shell scripts should stop unless all required
full-run steps are complete. Use `--next-command` when a wrapper only needs the
next required command without parsing the full status report.

When private evaluation artifacts live outside the report directory, pass
`--corrections /path/to/corrections.csv` to `build-report`.
