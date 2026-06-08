# External GPU full run

This runbook is for running the full ImageNet validation experiment on a
regular Linux GPU machine outside Colab: RunPod, Lambda Labs, a local CUDA box,
or a shared lab server. Use this when Colab GPU quota is unstable or when a
worker needs a clean handoff without notebook interaction.

The full run is resumable. Completed teacher-cache shards are skipped when the
parquet metadata, fp16 logits, and `.run.json` sidecars match the current
configuration.

For a shorter handoff brief that can be sent directly to a GPU worker, use
[`docs/gpu-handoff.md`](gpu-handoff.md).

## When To Use This

Use an external GPU worker when you need the full ImageNet experiment to finish
without relying on Colab session lifetime. CPU machines are useful for setup,
artifact inspection, and status commands only. Do not run the full teacher cache
on CPU: the required cache covers roughly 5,000,000 ResNet50 forwards.

## Requirements

- Linux machine with an NVIDIA GPU and working CUDA runtime.
- Python 3.10 or newer.
- `git`, `uv`, and enough disk space for ImageNet validation plus run artifacts.
- ImageNet validation data laid out as `val/WNID/image.JPEG`, using the
  ImageNet-1k WNID class directories expected by `timm-imagenet-1k`.
- Persistent storage for `artifacts/`, `reports/`, and logs.

The default experiment config is:

```text
configs/experiment/resnet50_a1_in1k.yaml
```

It writes outputs under repository-relative paths:

```text
artifacts/manifests
artifacts/teacher_cache
artifacts/selector
reports/resnet50_a1_in1k
```

For cloud workers, make `artifacts/` and `reports/` persistent volumes or
symlinks into persistent storage before launching expensive work.

## Recommended Layout

Use explicit paths so reconnects and handoffs stay predictable:

```bash
export WORKDIR="$HOME/albu-tta"
export RUN_ROOT="$HOME/albu-tta-runs/resnet50_a1_in1k"
export IMAGENET_VAL_DIR="$HOME/datasets/imagenet/val"
export CONFIG="configs/experiment/resnet50_a1_in1k.yaml"
export CACHE_LOG_DIR="$RUN_ROOT/logs"
```

Expected persistent layout:

```text
$RUN_ROOT/
  artifacts/
    augmentation_registry_audit.json
    manifests/
      class_to_idx.json
    selector/
    teacher_cache/
  reports/
    resnet50_a1_in1k/
  logs/
    cache_public_train.log
    cache_public_val.log
    cache_private.log
```

## Fresh Machine Setup

```bash
git clone https://github.com/dKosarevsky/albu-tta.git "$WORKDIR"
cd "$WORKDIR"
uv sync --extra stackers
mkdir -p "$RUN_ROOT/artifacts" "$RUN_ROOT/reports" "$CACHE_LOG_DIR"

[ ! -e artifacts ] || [ -L artifacts ] || {
  echo "Refusing to replace non-symlink artifacts directory"
  exit 1
}
[ ! -e reports ] || [ -L reports ] || {
  echo "Refusing to replace non-symlink reports directory"
  exit 1
}
ln -sfn "$RUN_ROOT/artifacts" artifacts
ln -sfn "$RUN_ROOT/reports" reports
```

Check CUDA before spending time on setup:

```bash
nvidia-smi
uv run python - <<'PY'
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## ImageNet Validation Check

`IMAGENET_VAL_DIR` must contain WNID class directories directly under the
validation root:

```text
$IMAGENET_VAL_DIR/n01440764/ILSVRC2012_val_00000293.JPEG
```

Preparing that layout does not require a GPU. Use official ImageNet access,
then copy these local files to the worker:

```text
ILSVRC2012_img_val.tar
ILSVRC2012_devkit_t12.tar.gz
```

The project command streams the validation tar and reads
`ILSVRC2012_validation_ground_truth.txt` from the devkit:

```bash
uv run python -m learned_tta.cli prepare-imagenet-val \
  --config "$CONFIG" \
  --val-tar /path/to/ILSVRC2012_img_val.tar \
  --devkit /path/to/ILSVRC2012_devkit_t12.tar.gz \
  --output-dir "$IMAGENET_VAL_DIR"
```

It writes `val/WNID/*.JPEG` plus `_preparation_audit.json`. It refuses to write
into a non-empty output directory unless `--overwrite` is passed. If the devkit
has already been extracted, pass the extracted devkit directory to `--devkit`;
if only `ILSVRC2012_validation_ground_truth.txt` is available, pass it with
`--ground-truth`.

Verify the count and run the project preflight:

```bash
find "$IMAGENET_VAL_DIR" -name '*.JPEG' | wc -l

uv run python -m learned_tta.cli check-full-run \
  --config "$CONFIG" \
  --imagenet-val-dir "$IMAGENET_VAL_DIR"
```

The preflight must report 1,000 classes, 50,000 images, 100 candidates, and the
configured teacher model. It also verifies that the discovered WNID directories
match the configured `timm-imagenet-1k` class index before any expensive cache
work starts.

## Clean CenterCrop Baseline Only

Use this when the immediate goal is the standard ResNet50 validation baseline
without all 100 TTA candidates. Identity `aug_000` applies no AlbumentationsX
transform; the teacher dataloader still uses timm evaluation preprocessing,
including resize, CenterCrop, and normalization.

```bash
uv run python -m learned_tta.cli cache-teacher \
  --split public_train \
  --config "$CONFIG" \
  --candidate-id aug_000 \
  --device cuda \
  --num-workers 2

uv run python -m learned_tta.cli cache-teacher \
  --split public_val \
  --config "$CONFIG" \
  --candidate-id aug_000 \
  --device cuda \
  --num-workers 2

uv run python -m learned_tta.cli cache-teacher \
  --split private \
  --config "$CONFIG" \
  --candidate-id aug_000 \
  --device cuda \
  --num-workers 2

uv run python -m learned_tta.cli summarize-clean-baseline \
  --config "$CONFIG"
```

The summary goes to:

```text
$RUN_ROOT/reports/resnet50_a1_in1k/tables/clean_center_crop_baseline.json
```

The same identity shards are reused by the later full run; `resume-full-run`
will skip valid shards instead of recomputing them. Each newly written
teacher-cache shard also writes an optional `.benchmark.json` sidecar with
backend, device, elapsed time, and throughput. These benchmark sidecars are
diagnostic only; `.run.json` sidecars remain the source of truth for cache
resume and completeness.

## Restore Progress From Colab Or Another Worker

If another worker already produced persistent outputs, restore them into
`$RUN_ROOT` before linking `artifacts/` and `reports/`. The important invariant
is that the repository sees this layout:

```text
$WORKDIR/artifacts -> $RUN_ROOT/artifacts
$WORKDIR/reports   -> $RUN_ROOT/reports
```

After restoring files, inspect status before launching anything:

```bash
cd "$WORKDIR"

uv run python -m learned_tta.cli full-run-status \
  --config "$CONFIG"

uv run python -m learned_tta.cli full-run-status \
  --config "$CONFIG" \
  --format json

uv run python -m learned_tta.cli teacher-cache-plan \
  --config "$CONFIG"

uv run python -m learned_tta.cli teacher-backend-plan \
  --device cuda
```

Teacher cache progress can be checked with sidecar counts:

```bash
find "$RUN_ROOT/artifacts/teacher_cache" -name 'public_train__*.run.json' | wc -l
find "$RUN_ROOT/artifacts/teacher_cache" -name 'public_val__*.run.json' | wc -l
find "$RUN_ROOT/artifacts/teacher_cache" -name 'private__*.run.json' | wc -l
```

Each split is complete only when it has one matching parquet file, one
`logits.npy` file, and one `.run.json` sidecar for every configured augmentation
candidate.

For the default ResNet50 experiment, the full teacher cache means 5M teacher
predictions:

```text
public_train: 20,000 images * 100 augmentations = 2,000,000 predictions
public_val:    5,000 images * 100 augmentations =   500,000 predictions
private:      25,000 images * 100 augmentations = 2,500,000 predictions
```

On disk that is 300 complete teacher-cache shards: 100 for `public_train`, 100
for `public_val`, and 100 for `private`. Each shard must have the parquet
metadata, fp16 logits, and `.run.json` metadata sidecar. Optional
`.benchmark.json` sidecars help estimate throughput but do not count toward
cache completeness. Use `full-run-status --fail-on-incomplete` as the final
source of truth instead of relying only on file counts.
Use `teacher-cache-plan` for a cache-specific summary: it reports expected
images, teacher forward passes, complete shards, missing files,
stale/malformed shards, fp16 logits size, and the next `cache-teacher --split`
command for `public_train`, `public_val`, and `private`.
The implemented teacher-cache backend is currently PyTorch. `teacher-backend-plan`
documents planned accelerators: TensorRT for CUDA, ONNXRuntime as a portable
exported-model runtime, and OpenVINO for CPU. Do not pass these planned backend
names to `cache-teacher`; the CLI fails early until their inference paths are
implemented and tested.

## One-Step Resume

For interactive use, run one supervised step at a time:

```bash
cd "$WORKDIR"

uv run python -m learned_tta.cli resume-full-run \
  --config "$CONFIG" \
  --imagenet-val-dir "$IMAGENET_VAL_DIR" \
  --cache-log-dir "$CACHE_LOG_DIR"
```

Behavior:

- non-cache steps run in the foreground;
- long `cache-teacher` steps start in the background;
- cache logs are written to `$CACHE_LOG_DIR`;
- duplicate active cache jobs are refused by default;
- already completed cache shards are skipped by cache resume.

After a background cache starts, watch the log and shard counts:

```bash
tail -f "$CACHE_LOG_DIR/cache_public_train.log"

uv run python -m learned_tta.cli full-run-status \
  --config "$CONFIG"
```

Use `cache_public_val.log` or `cache_private.log` instead when the active split
is `public_val` or `private`.

Re-run `resume-full-run` after the active background step finishes. It will
dispatch the next missing required step.

## Fully Automated Driver

On a stable GPU host, this loop can drive the required pipeline end to end while
still respecting the supervisor's duplicate-cache guard:

```bash
cd "$WORKDIR"

while true; do
  uv run python -m learned_tta.cli resume-full-run \
    --config "$CONFIG" \
    --imagenet-val-dir "$IMAGENET_VAL_DIR" \
    --cache-log-dir "$CACHE_LOG_DIR"

  if uv run python -m learned_tta.cli full-run-status \
    --config "$CONFIG" \
    --fail-on-incomplete; then
    break
  fi

  sleep 60
done
```

This is intentionally conservative. If a cache process is already active, the
supervisor prints the active process and exits without starting another copy.
Before launching full all-candidate teacher caching, the status order first
caches `public_val` identity (`aug_000`) and runs `check-clean-baseline`. That
gate checks clean top-1, top-5, and NLL against loose config thresholds so a
broken ImageNet mapping or preprocessing issue fails early.

## Manual Commands

Use manual commands only when you intentionally want to bypass the supervisor.
Run `full-run-status --next-command` first:

```bash
uv run python -m learned_tta.cli full-run-status \
  --config "$CONFIG" \
  --next-command
```

Typical required order:

```bash
uv run python -m learned_tta.cli validate-augmentations \
  --config "$CONFIG" \
  --audit-output artifacts/augmentation_registry_audit.json

uv run python -m learned_tta.cli make-splits \
  --config "$CONFIG" \
  --imagenet-val-dir "$IMAGENET_VAL_DIR"

uv run python -m learned_tta.cli cache-teacher \
  --split public_val \
  --config "$CONFIG" \
  --candidate-id aug_000 \
  --device cuda \
  --num-workers 2

uv run python -m learned_tta.cli check-clean-baseline \
  --config "$CONFIG"

uv run python -m learned_tta.cli cache-teacher \
  --split public_train \
  --config "$CONFIG" \
  --backend pytorch \
  --device cuda \
  --num-workers 2

uv run python -m learned_tta.cli cache-teacher \
  --split public_val \
  --config "$CONFIG" \
  --backend pytorch \
  --device cuda \
  --num-workers 2

uv run python -m learned_tta.cli build-targets \
  --config "$CONFIG"

uv run python -m learned_tta.cli build-targets \
  --config "$CONFIG" \
  --target-kind softmax_weight

uv run python -m learned_tta.cli train-selector \
  --config "$CONFIG" \
  --device cuda

uv run python -m learned_tta.cli tune-tta \
  --split public_val \
  --config "$CONFIG" \
  --device cuda

uv run python -m learned_tta.cli train-aggregator \
  --method global-nonnegative \
  --config "$CONFIG" \
  --split public_val \
  --device cuda

uv run python -m learned_tta.cli train-aggregator \
  --method class-nonnegative \
  --config "$CONFIG" \
  --split public_val \
  --device cuda

uv run python -m learned_tta.cli cache-teacher \
  --split private \
  --config "$CONFIG" \
  --device cuda \
  --num-workers 2

uv run python -m learned_tta.cli evaluate-private \
  --config "$CONFIG" \
  --device cuda

uv run python -m learned_tta.cli build-report \
  --config "$CONFIG" \
  --device cuda
```

Optional XGBoost stacker:

```bash
uv sync --extra stackers

uv run python -m learned_tta.cli train-aggregator \
  --method xgboost-multiclass \
  --config "$CONFIG" \
  --split public_val
```

XGBoost is optional and does not block required `full-run-status` completion.

## Recovery Checklist

Use this checklist after a disconnect, preempted cloud instance, or SSH loss:

1. Reconnect to the same persistent volume.
2. Recreate the repository checkout if needed.
3. Recreate the symlinks:

   ```bash
   cd "$WORKDIR"
   mkdir -p "$RUN_ROOT/artifacts" "$RUN_ROOT/reports" "$CACHE_LOG_DIR"
   [ ! -e artifacts ] || [ -L artifacts ] || {
     echo "Refusing to replace non-symlink artifacts directory"
     exit 1
   }
   [ ! -e reports ] || [ -L reports ] || {
     echo "Refusing to replace non-symlink reports directory"
     exit 1
   }
   ln -sfn "$RUN_ROOT/artifacts" artifacts
   ln -sfn "$RUN_ROOT/reports" reports
   ```

4. Confirm CUDA and ImageNet:

   ```bash
   nvidia-smi
   find "$IMAGENET_VAL_DIR" -name '*.JPEG' | wc -l
   ```

5. Check for an active cache process:

   ```bash
   pgrep -af 'learned_tta.cli cache-teacher' || true
   ```

6. Inspect status:

   ```bash
   uv run python -m learned_tta.cli full-run-status \
     --config "$CONFIG"
   ```

   Treat `.run.json` sidecar mismatches as stale cache. `full-run-status`
   validates split, augmentation id, seed, teacher model, timm data config,
   row count, class count, storage format, and registry candidate parameters
   before it marks a teacher-cache shard complete.
   If a shard is reported incomplete after a code/config update, do not delete
   the whole cache. Re-run `resume-full-run`; valid shards are skipped and
   missing, stale, or malformed shards are regenerated.

7. Resume one step:

   ```bash
   uv run python -m learned_tta.cli resume-full-run \
     --config "$CONFIG" \
     --imagenet-val-dir "$IMAGENET_VAL_DIR" \
     --cache-log-dir "$CACHE_LOG_DIR"
   ```

Do not delete partial teacher-cache files manually unless you intentionally want
to recompute a broken shard. Metadata-aware resume will skip valid shards and
rewrite missing or stale shards.

## Final Gate

The run is complete only when all required steps pass:

```bash
uv run python -m learned_tta.cli full-run-status \
  --config "$CONFIG" \
  --fail-on-incomplete
```

Expected final report:

```text
$RUN_ROOT/reports/resnet50_a1_in1k/results.md
```

Do not make paper claims from partial runs. Private metrics become meaningful
only after `evaluate-private` and `build-report` have both completed.
