# Full Run

This page is the compact operational map for the end-to-end experiment. Use it
before the detailed provider-specific runbooks:

- [`docs/colab-run.md`](colab-run.md) for Google Colab.
- [`docs/gpu-run.md`](gpu-run.md) for RunPod, Lambda Labs, local CUDA servers,
  and other external GPU workers.
- [`docs/gpu-handoff.md`](gpu-handoff.md) for a short handoff checklist.

The full ImageNet run is resumable. The source of truth is the repository
status CLI, not stale notebook output or manual file counts.

## Smoke Run

Run the synthetic smoke pipeline before spending GPU time:

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

The smoke run exercises manifests, teacher cache, target build, selector
training, public tuning, private evaluation, learned aggregation, and
`build-report`. The expected final artifact is:

```text
artifacts/smoke/reports/results.md
```

The current implementation checklist lives in
[`docs/implementation-status.md`](implementation-status.md).

## ImageNet Preparation

ImageNet preparation is CPU-only and does not require CUDA. It converts local
official ImageNet archives into the required `val/WNID/*.JPEG` layout.

Required inputs:

```text
ILSVRC2012_img_val.tar
ILSVRC2012_devkit_t12.tar.gz
```

Run:

```bash
uv run python -m learned_tta.cli prepare-imagenet-val \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --val-tar /path/to/ILSVRC2012_img_val.tar \
  --devkit /path/to/ILSVRC2012_devkit_t12.tar.gz \
  --output-dir /path/to/imagenet/val
```

`prepare-imagenet-val` does not download ImageNet. If the devkit has already
been extracted, pass that directory with `--devkit`. For official ImageNet-val,
prefer the full devkit archive or extracted devkit directory over a standalone
`ILSVRC2012_validation_ground_truth.txt`: the ground-truth file stores
`ILSVRC2012_ID` labels, and `meta.mat` supplies the required
`ILSVRC2012_ID -> WNID` mapping. Passing `--ground-truth` is only for already
normalized one-based labels that match the configured class index order.

The prepared directory must contain exactly 1,000 WNID directories and 50,000
JPEG files. Check it before launching GPU inference:

```bash
uv run python -m learned_tta.cli check-full-run \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --imagenet-val-dir /path/to/imagenet/val
```

## Augmentation Audit

Before the full cache, validate the registry and write the exact candidate
snapshot:

```bash
uv run python -m learned_tta.cli validate-augmentations \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --audit-output artifacts/augmentation_registry_audit.json
```

The audit records every configured augmentation candidate, serialized
AlbumentationsX `Compose` payloads, transform parameters, deterministic mode,
and runtime package versions. See [`docs/artifacts.md`](artifacts.md) for the
artifact contract.

## Status And Resume

Use `full-run-status` after each expensive step:

```bash
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
```

The human output includes required/optional status plus `missing=` and
`extra=` counts. The JSON output includes `missing_outputs` and `extra_outputs`
for wrappers. full-run-status treats `.run.json` sidecars as required teacher cache outputs
and validates the sidecar metadata; stale Drive shards with a
different teacher model, timm data config, seed, candidate params, row count,
class count, or storage format are incomplete by design.

Prefer the supervisor instead of manually dispatching long steps:

```bash
uv run python -m learned_tta.cli resume-full-run \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --imagenet-val-dir /path/to/imagenet/val \
  --cache-log-dir artifacts/logs
```

`resume-full-run` runs the next missing required step, starts long
`cache-teacher` steps in the background, and refuses duplicate active cache
jobs.

## Clean CenterCrop Baseline

Identity `aug_000` is the clean CenterCrop path: no AlbumentationsX transform is
applied before the standard timm resize, CenterCrop, and normalization.

For a clean-only baseline:

```bash
uv run python -m learned_tta.cli cache-teacher --split public_train \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --candidate-id aug_000 \
  --device cuda

uv run python -m learned_tta.cli cache-teacher --split public_val \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --candidate-id aug_000 \
  --device cuda

uv run python -m learned_tta.cli cache-teacher --split private \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --candidate-id aug_000 \
  --device cuda

uv run python -m learned_tta.cli summarize-clean-baseline \
  --config configs/experiment/resnet50_a1_in1k.yaml
```

The identity shards are reused by the full run; valid shards are skipped by
cache resume.

## Standard TTA Baselines

`evaluate-private` writes two standard cached rows when the corresponding
teacher-cache shards are present:

- `clean_center_crop`: identity `aug_000`, equivalent to the clean CenterCrop
  path.
- `center_crop_hflip`: identity plus `aug_005`, the configured horizontal flip
  candidate.

The classic 10-crop baseline is not one of the 100 AlbumentationsX candidates,
so generate it once as a separate logits artifact:

```bash
uv run python -m learned_tta.cli run-ten-crop-baseline \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --split private \
  --device cuda
```

Then include it in the frozen private table and rebuild the report:

```bash
uv run python -m learned_tta.cli evaluate-private \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --ten-crop-logits reports/resnet50_a1_in1k/tables/private_ten_crop_logits.npz \
  --device cuda

uv run python -m learned_tta.cli build-report \
  --config configs/experiment/resnet50_a1_in1k.yaml
```

## Full GPU Order

The supervisor should drive the required order. Manual commands are useful for
debugging or for a worker that wants explicit control.

Typical sequence:

```text
validate-augmentations
make-splits
cache-teacher --split public_val --candidate-id aug_000
check-clean-baseline
cache-teacher --split public_train
cache-teacher --split public_val
build-targets
train-selector
tune-tta
train-aggregator --method global-nonnegative
train-aggregator --method class-nonnegative
cache-teacher --split private
evaluate-private
build-report
```

Optional XGBoost stacker:

```bash
uv sync --extra stackers

uv run python -m learned_tta.cli train-aggregator \
  --method xgboost-multiclass \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --split public_val
```

In shorthand, this optional command is `train-aggregator --method xgboost-multiclass`.

The report names this strategy `xgboost_multiclass`. Learned aggregation
strategies are reported as `global_weighted_tta` and `class_weighted_tta`.

## Full 5M Cache

For the default ResNet50 experiment, the full teacher cache is:

```text
public_train: 20,000 images * 100 augmentations = 2,000,000 predictions
public_val:    5,000 images * 100 augmentations =   500,000 predictions
private:      25,000 images * 100 augmentations = 2,500,000 predictions
```

That is 300 complete teacher-cache shards: 100 for `public_train`, 100 for
`public_val`, and 100 for `private`. Use `teacher-cache-plan` for a
cache-specific progress report and next command:

```bash
uv run python -m learned_tta.cli teacher-cache-plan \
  --config configs/experiment/resnet50_a1_in1k.yaml
```

After a split cache is complete, run `teacher-cache-diagnostics` for a cheap
augmentation impact readout before training additional selector heads:

```bash
uv run python -m learned_tta.cli teacher-cache-diagnostics \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --split public_val
```

The implemented teacher backend is PyTorch. TensorRT, ONNXRuntime, and OpenVINO
are planned accelerators documented by `teacher-backend-plan`; they are not
enabled teacher-cache backends yet.

## Report Outputs

`build-report` writes `results.md`, top-N markdown tables, SVG figures, and CSV
tables for augmentation impact, private metric deltas, correction counts,
learned aggregation weights, XGBoost feature importance, transform class
impact, and transform class aggregation.

Key report fragments include:

```text
private_metric_deltas.csv
augmentation_name
transform_class
transform_class_impact.csv
transform_class_impact.svg
transform_class_aggregation.csv
transform_class_aggregation.svg
gain_distribution.svg
oracle_overlap.svg
xgboost_feature_importance.csv
xgboost_feature_importance.svg
active_threshold
```

Do not make paper claims from partial runs. Private metrics are meaningful only
after selector training, public-val tuning, learned aggregation training,
private evaluation, and report building are frozen and complete.
