#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

link_persistent_path() {
  local target="$1"
  local link="$2"
  mkdir -p "$(dirname "$target")" "$(dirname "$link")"
  if [ -e "$link" ] && [ ! -L "$link" ]; then
    fail "Refusing to replace non-symlink path: $link"
  fi
  ln -sfn "$target" "$link"
}

count_imagenet_jpegs() {
  if [ ! -d "$IMAGENET_VAL_DIR" ]; then
    printf '0\n'
    return
  fi
  find "$IMAGENET_VAL_DIR" -type f -name '*.JPEG' | wc -l | tr -d ' '
}

check_cuda() {
  if [[ "$DEVICE" != cuda* ]]; then
    log "Skipping CUDA check because DEVICE=$DEVICE"
    return
  fi
  uv run python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(torch.cuda.get_device_name(0))
PY
}

prepare_imagenet_if_needed() {
  local jpeg_count
  jpeg_count="$(count_imagenet_jpegs)"
  if [ "$jpeg_count" = "50000" ]; then
    log "ImageNet validation layout is ready: $IMAGENET_VAL_DIR"
    return
  fi

  if [ -z "${VAL_TAR:-}" ]; then
    fail "ImageNet validation layout has $jpeg_count JPEG files, and VAL_TAR is not set"
  fi
  [ -f "$VAL_TAR" ] || fail "VAL_TAR does not exist: $VAL_TAR"

  local label_args=()
  if [ -n "${GROUND_TRUTH:-}" ]; then
    [ -f "$GROUND_TRUTH" ] || fail "GROUND_TRUTH does not exist: $GROUND_TRUTH"
    label_args=(--ground-truth "$GROUND_TRUTH")
  elif [ -n "${DEVKIT:-}" ]; then
    [ -e "$DEVKIT" ] || fail "DEVKIT does not exist: $DEVKIT"
    label_args=(--devkit "$DEVKIT")
  else
    fail "Set DEVKIT or GROUND_TRUTH when VAL_TAR is used"
  fi

  local overwrite_args=()
  if [ "${OVERWRITE_IMAGENET:-0}" = "1" ]; then
    overwrite_args=(--overwrite)
  fi

  log "Preparing ImageNet validation layout into $IMAGENET_VAL_DIR"
  uv run python -m learned_tta.cli prepare-imagenet-val \
    --config "$CONFIG" \
    --val-tar "$VAL_TAR" \
    "${label_args[@]}" \
    --output-dir "$IMAGENET_VAL_DIR" \
    "${overwrite_args[@]}"

  jpeg_count="$(count_imagenet_jpegs)"
  [ "$jpeg_count" = "50000" ] || fail "Expected 50000 JPEG files after prepare, found $jpeg_count"
}

run_preflight() {
  log "Validating augmentation registry"
  uv run python -m learned_tta.cli validate-augmentations \
    --config "$CONFIG" \
    --audit-output artifacts/augmentation_registry_audit.json

  log "Checking full-run prerequisites"
  uv run python -m learned_tta.cli check-full-run \
    --config "$CONFIG" \
    --imagenet-val-dir "$IMAGENET_VAL_DIR"

  uv run python -m learned_tta.cli teacher-backend-plan --device "$DEVICE"
  uv run python -m learned_tta.cli full-run-status --config "$CONFIG"
  uv run python -m learned_tta.cli teacher-cache-plan --config "$CONFIG"
}

resume_until_complete() {
  local step=0
  local resume_args=()
  if [ "${FOREGROUND_CACHE:-0}" = "1" ]; then
    resume_args+=(--foreground-cache)
  fi
  if [ "${ALLOW_DUPLICATE_CACHE:-0}" = "1" ]; then
    resume_args+=(--allow-duplicate-cache)
  fi

  while ! uv run python -m learned_tta.cli full-run-status \
    --config "$CONFIG" \
    --fail-on-incomplete; do
    step=$((step + 1))
    if [ "$MAX_STEPS" -gt 0 ] && [ "$step" -gt "$MAX_STEPS" ]; then
      fail "MAX_STEPS=$MAX_STEPS reached before full-run completion"
    fi

    log "Running/resuming next full-run step: $step"
    uv run python -m learned_tta.cli resume-full-run \
      --config "$CONFIG" \
      --imagenet-val-dir "$IMAGENET_VAL_DIR" \
      --cache-log-dir "$CACHE_LOG_DIR" \
      "${resume_args[@]}"

    uv run python -m learned_tta.cli teacher-cache-plan --config "$CONFIG" || true
    if [ "$SLEEP_SECONDS" -gt 0 ]; then
      sleep "$SLEEP_SECONDS"
    fi
  done
}

write_post_run_diagnostics() {
  local diagnostics_dir="$RUN_ROOT/reports/resnet50_a1_in1k/tables"
  mkdir -p "$diagnostics_dir"
  for split in public_val private; do
    uv run python -m learned_tta.cli teacher-cache-diagnostics \
      --config "$CONFIG" \
      --split "$split" \
      --output "$diagnostics_dir/${split}_teacher_cache_diagnostics.json"
  done
  uv run python -m learned_tta.cli teacher-cache-plan --config "$CONFIG"
}

main() {
  require_command git
  require_command uv

  WORKDIR="${WORKDIR:-$(pwd)}"
  cd "$WORKDIR"

  CONFIG="${CONFIG:-configs/experiment/resnet50_a1_in1k.yaml}"
  RUN_ROOT="${RUN_ROOT:-$WORKDIR/.runs/resnet50_a1_in1k}"
  IMAGENET_VAL_DIR="${IMAGENET_VAL_DIR:-$RUN_ROOT/imagenet/val}"
  CACHE_LOG_DIR="${CACHE_LOG_DIR:-$RUN_ROOT/logs}"
  DEVICE="${DEVICE:-cuda}"
  MAX_STEPS="${MAX_STEPS:-0}"
  SLEEP_SECONDS="${SLEEP_SECONDS:-60}"

  if [ "${GIT_PULL:-0}" = "1" ]; then
    git pull --ff-only
  fi
  if [ "${SYNC_DEPS:-1}" = "1" ]; then
    uv sync --extra stackers
  fi

  mkdir -p \
    "$RUN_ROOT/artifacts/manifests" \
    "$RUN_ROOT/artifacts/teacher_cache" \
    "$RUN_ROOT/artifacts/selector" \
    "$RUN_ROOT/reports" \
    "$CACHE_LOG_DIR"

  mkdir -p artifacts
  link_persistent_path "$RUN_ROOT/artifacts/manifests" artifacts/manifests
  link_persistent_path "$RUN_ROOT/artifacts/teacher_cache" artifacts/teacher_cache
  link_persistent_path "$RUN_ROOT/artifacts/selector" artifacts/selector
  link_persistent_path "$RUN_ROOT/artifacts/augmentation_registry_audit.json" \
    artifacts/augmentation_registry_audit.json
  link_persistent_path "$RUN_ROOT/reports" reports

  check_cuda
  prepare_imagenet_if_needed
  run_preflight
  resume_until_complete
  write_post_run_diagnostics

  log "Full ImageNet pipeline complete"
}

main "$@"
