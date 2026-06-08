# ImageNet Validation Prepare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU-only command that prepares official ImageNet validation images into `val/WNID/*.JPEG` layout before GPU inference.

**Architecture:** Implement a small `imagenet_prepare` module that streams JPEG files from `ILSVRC2012_img_val.tar`, reads validation labels from either the official devkit archive or a direct ground-truth text file, maps one-based labels through the configured class index, and writes an audit JSON. Expose it through `learned_tta.cli prepare-imagenet-val` and document the official-source workflow.

**Tech Stack:** Python standard library `tarfile`, `shutil`, `json`, existing config/class-index helpers, pytest, Ruff, ty.

---

### Task 1: Preparation Module

**Files:**
- Create: `src/learned_tta/imagenet_prepare.py`
- Test: `tests/test_imagenet_prepare.py`

- [ ] Write tests for preparing a fake validation tar with four JPEG files and labels from a fake devkit tar.
- [ ] Implement `prepare_imagenet_val` with non-destructive output handling, tar streaming, label mapping, per-class counts, and audit JSON.
- [ ] Add tests for direct `--ground-truth`, mismatched label/image count, invalid labels, and non-empty output rejection without `overwrite=True`.

### Task 2: CLI

**Files:**
- Modify: `src/learned_tta/cli.py`
- Test: `tests/test_cli.py`

- [ ] Add `prepare-imagenet-val --config --val-tar --output-dir [--devkit|--ground-truth] [--audit-output] [--overwrite]`.
- [ ] Wire the command to `load_experiment_config` and configured `dataset.class_index`.
- [ ] Add a CLI test that prepares a fake tar and asserts command output plus audit file.

### Task 3: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/gpu-run.md`
- Modify: `docs/colab-run.md`
- Modify: `docs/gpu-handoff.md`
- Test: `tests/test_ci_config.py`

- [ ] Document official access URLs, required local files, and the CPU-only prepare command.
- [ ] State that the command does not download ImageNet and does not require GPU.
- [ ] Add doc contract tests for `prepare-imagenet-val`, `ILSVRC2012_img_val.tar`, `ILSVRC2012_devkit_t12.tar.gz`, and CPU-only preparation.

### Task 4: Verification And PR

**Files:**
- No new source files.

- [ ] Run `uv run pytest -q`.
- [ ] Run `uv run pytest --cov=learned_tta --cov-report=term-missing --cov-fail-under=98.5`.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run ty check`.
- [ ] Commit granular changes, push PR, wait for CI, merge after green checks.
