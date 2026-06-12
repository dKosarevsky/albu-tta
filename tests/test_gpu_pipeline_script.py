from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_full_imagenet_pipeline.sh"


def test_gpu_pipeline_script_has_resumable_full_run_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    mode = SCRIPT.stat().st_mode

    assert source.startswith("#!/usr/bin/env bash")
    assert "set -Eeuo pipefail" in source
    assert mode & 0o111
    assert "torch.cuda.is_available()" in source
    assert "prepare-imagenet-val" in source
    assert "check-full-run" in source
    assert "validate-augmentations" in source
    assert "teacher-backend-plan" in source
    assert "teacher-cache-plan" in source
    assert "teacher-cache-diagnostics" in source
    assert "resume-full-run" in source
    assert "full-run-status" in source
    assert "--fail-on-incomplete" in source
    assert "--cache-log-dir" in source
    assert "MAX_STEPS" in source
    assert "RUN_ROOT" in source
    assert "CACHE_LOG_DIR" in source
    assert "FOREGROUND_CACHE" in source
    assert "ALLOW_DUPLICATE_CACHE" in source


def test_gpu_pipeline_script_refuses_to_replace_real_artifacts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "Refusing to replace non-symlink path" in source
    assert "link_persistent_path" in source
    assert "artifacts/teacher_cache" in source
    assert "artifacts/manifests" in source
    assert "artifacts/selector" in source
    assert "reports" in source


def test_gpu_pipeline_script_is_documented() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    gpu_runbook = (ROOT / "docs" / "gpu-run.md").read_text(encoding="utf-8")

    assert "scripts/run_full_imagenet_pipeline.sh" in readme
    assert "scripts/run_full_imagenet_pipeline.sh" in gpu_runbook


def test_gpu_pipeline_script_path_constant_is_repo_relative() -> None:
    assert os.fspath(SCRIPT).endswith("scripts/run_full_imagenet_pipeline.sh")
