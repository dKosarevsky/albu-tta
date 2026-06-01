# External GPU full run

This runbook is for running the full ImageNet validation experiment on a
regular Linux GPU machine outside Colab: RunPod, Lambda Labs, a local CUDA box,
or a shared lab server. Use this when Colab GPU quota is unstable or when a
worker needs a clean handoff without notebook interaction.

The full run is resumable. Completed teacher-cache shards are skipped when the
parquet metadata, fp16 logits, and `.run.json` sidecars match the current
configuration.

## When To Use This

Use an external GPU worker when you need the full ImageNet experiment to finish
without relying on Colab session lifetime. CPU machines are useful for setup,
artifact inspection, and status commands only. Do not run the full teacher cache
on CPU: the required cache covers roughly 5,000,000 ResNet50 forwards.

## Requirements

- Linux machine with an NVIDIA GPU and working CUDA runtime.
- Python 3.10 or newer.
- `git`, `uv`, and enough disk space for ImageNet validation plus run artifacts.
- ImageNet validation data laid out as `val/class_name/image.JPEG`.
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

`IMAGENET_VAL_DIR` must contain class directories directly under the validation
root:

```text
$IMAGENET_VAL_DIR/n01440764/ILSVRC2012_val_00000293.JPEG
```

Verify the count and run the project preflight:

```bash
find "$IMAGENET_VAL_DIR" -name '*.JPEG' | wc -l

uv run python -m learned_tta.cli check-full-run \
  --config "$CONFIG" \
  --imagenet-val-dir "$IMAGENET_VAL_DIR"
```

The preflight must report 1,000 classes, 50,000 images, 100 candidates, and the
configured teacher model.

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
  --split public_train \
  --config "$CONFIG" \
  --device cuda \
  --num-workers 2

uv run python -m learned_tta.cli cache-teacher \
  --split public_val \
  --config "$CONFIG" \
  --device cuda \
  --num-workers 2

uv run python -m learned_tta.cli build-targets \
  --config "$CONFIG"

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
