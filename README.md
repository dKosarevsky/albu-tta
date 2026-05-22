# albu-tta

[![CI](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml)
[![pytest](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml/badge.svg?branch=main&job=pytest)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![ruff](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml/badge.svg?branch=main&job=ruff)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![ty](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml/badge.svg?branch=main&job=ty)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![coverage](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml/badge.svg?branch=main&job=coverage)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![pytest](https://img.shields.io/badge/tests-pytest-blue.svg)
![Ruff](https://img.shields.io/badge/lint-ruff-46a2f1.svg)
![ty](https://img.shields.io/badge/types-ty-46a2f1.svg)
![coverage](https://img.shields.io/badge/coverage-ci-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Learned test-time augmentation selector experiments with AlbumentationsX.

## Smoke Run

Use the synthetic smoke run before spending GPU time on ImageNet. It creates a tiny
ImageNet-like directory, caches a fake teacher, trains the selector for one epoch,
tunes TTA, evaluates private metrics, and writes `results.md`.

```bash
uv run python -m learned_tta.cli run-smoke \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --output-dir artifacts/smoke \
  --candidate-count 2 \
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
