# GPU Handoff Checklist

Use this one-page checklist when handing the full ImageNet experiment to someone
with a stable NVIDIA GPU machine. The detailed runbook is
[`docs/gpu-run.md`](gpu-run.md); this file is the operational brief.

## What The Worker Needs

- Repository: `https://github.com/dKosarevsky/albu-tta`
- Branch: `main`
- Config: `configs/experiment/resnet50_a1_in1k.yaml`
- ImageNet validation directory in `val/WNID/image.JPEG` layout.
- Persistent storage for `artifacts/`, `reports/`, and logs.
- NVIDIA GPU with CUDA, Python 3.10+, `git`, and `uv`.

Do not run the full teacher cache on CPU. The cache workload is roughly
5,000,000 ResNet50 forwards.

## Setup

```bash
export WORKDIR="$HOME/albu-tta"
export RUN_ROOT="$HOME/albu-tta-runs/resnet50_a1_in1k"
export IMAGENET_VAL_DIR="$HOME/datasets/imagenet/val"
export CONFIG="configs/experiment/resnet50_a1_in1k.yaml"
export CACHE_LOG_DIR="$RUN_ROOT/logs"

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

Check CUDA:

```bash
nvidia-smi
uv run python - <<'PY'
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Start Or Resume

First verify ImageNet and config sanity:

```bash
find "$IMAGENET_VAL_DIR" -name '*.JPEG' | wc -l

uv run python -m learned_tta.cli check-full-run \
  --config "$CONFIG" \
  --imagenet-val-dir "$IMAGENET_VAL_DIR"
```

Then run one resumable step:

```bash
uv run python -m learned_tta.cli resume-full-run \
  --config "$CONFIG" \
  --imagenet-val-dir "$IMAGENET_VAL_DIR" \
  --cache-log-dir "$CACHE_LOG_DIR"
```

Re-run the same `resume-full-run` command after each foreground step completes,
after reconnects, and after background teacher-cache jobs finish. It skips valid
cache shards and refuses duplicate active cache jobs.

## Monitor Progress

Use status as the source of truth:

```bash
uv run python -m learned_tta.cli full-run-status \
  --config "$CONFIG"

uv run python -m learned_tta.cli full-run-status \
  --config "$CONFIG" \
  --next-command
```

For active cache steps:

```bash
pgrep -af 'learned_tta.cli cache-teacher' || true
tail -f "$CACHE_LOG_DIR/cache_public_train.log"
find "$RUN_ROOT/artifacts/teacher_cache" -name 'public_train__*.run.json' | wc -l
find "$RUN_ROOT/artifacts/teacher_cache" -name 'public_val__*.run.json' | wc -l
find "$RUN_ROOT/artifacts/teacher_cache" -name 'private__*.run.json' | wc -l
```

Use the matching log file for the active split:
`cache_public_train.log`, `cache_public_val.log`, or `cache_private.log`.

## Success Criteria

The run is complete only when this passes:

```bash
uv run python -m learned_tta.cli full-run-status \
  --config "$CONFIG" \
  --fail-on-incomplete
```

Expected final artifacts:

```text
$RUN_ROOT/reports/resnet50_a1_in1k/results.md
$RUN_ROOT/reports/resnet50_a1_in1k/tables/private_metrics.csv
$RUN_ROOT/reports/resnet50_a1_in1k/tables/private_metric_deltas.csv
$RUN_ROOT/reports/resnet50_a1_in1k/tables/augmentation_impact.csv
$RUN_ROOT/artifacts/selector/selector_best.pt
$RUN_ROOT/artifacts/selector/public_val_tta_tuning.json
```

Private results must be produced only after selector training, public-val tuning,
and learned aggregation training are frozen. Private oracle rows are diagnostics,
not tuning inputs.

## Return Package

Return the whole `$RUN_ROOT` if possible. If bandwidth is limited, return at
least:

```text
reports/resnet50_a1_in1k/
artifacts/augmentation_registry_audit.json
artifacts/manifests/class_to_idx.json
artifacts/selector/
logs/
```

Keep `artifacts/teacher_cache/` on the GPU worker or persistent volume until the
results are reviewed. It is large, but it is the only way to resume, rerun
evaluation, or debug report anomalies without repeating teacher inference.

## Failure Rules

- No CUDA: stop; do not continue the full cache on CPU.
- ImageNet count is not 50,000: stop and fix the dataset layout.
- `check-clean-baseline` fails: stop; inspect class mapping, labels, and timm
  preprocessing before full caching.
- `full-run-status` reports stale or malformed cache shards: re-run
  `resume-full-run`; do not delete the whole cache.
- Selector target `.npz` files without `image_id` lineage: rebuild with
  `build-targets` before selector training.
