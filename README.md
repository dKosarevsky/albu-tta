# albu-tta

[![pytest](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=pytest&label=pytest)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![ruff](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=ruff&label=ruff)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![ty](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=ty&label=ty)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
[![coverage](https://img.shields.io/github/check-runs/dKosarevsky/albu-tta/main?nameFilter=coverage&label=coverage)](https://github.com/dKosarevsky/albu-tta/actions/workflows/ci.yml?query=branch%3Amain)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dKosarevsky/albu-tta/blob/main/notebooks/full_imagenet_run_colab.ipynb)

Learned test-time augmentation selector experiments with
[AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX).

The project trains a small image-conditioned selector that predicts which
single AlbumentationsX candidates are useful for test-time augmentation of a
frozen `timm` teacher. The current default teacher is `resnet50.a1_in1k`.

## Where To Read

- [Implementation status](docs/implementation-status.md): what is implemented
  and what still needs the full ImageNet run.
- [Method](docs/method.md): selector target formulation, split contract,
  aggregation baselines, and leakage rules.
- [Full run](docs/full-run.md): smoke run, ImageNet preparation, clean
  CenterCrop baseline, and command order.
- [Artifacts](docs/artifacts.md): reports, tables, figures, cache sidecars, and
  diagnostics.
- [External GPU runbook](docs/gpu-run.md): running the full pipeline on RunPod,
  Lambda Labs, a local CUDA server, or another non-Colab GPU worker.
- [GPU handoff checklist](docs/gpu-handoff.md): short checklist for delegating
  the experiment to someone with GPU access.
- [Colab runbook](docs/colab-run.md): notebook workflow and Drive-backed
  artifacts.

## GPU Entrypoints

For Google Colab, open
[`notebooks/full_imagenet_run_colab.ipynb`](notebooks/full_imagenet_run_colab.ipynb)
and follow [docs/colab-run.md](docs/colab-run.md).

For a non-Colab GPU machine, use the scripted pipeline:

```bash
WORKDIR="$PWD" \
RUN_ROOT="$HOME/albu-tta-runs/resnet50_a1_in1k" \
IMAGENET_VAL_DIR="$HOME/datasets/imagenet/val" \
scripts/run_full_imagenet_pipeline.sh
```

If ImageNet-val still needs to be prepared, pass the official local archives:

```bash
VAL_TAR=/path/to/ILSVRC2012_img_val.tar \
DEVKIT=/path/to/ILSVRC2012_devkit_t12.tar.gz \
scripts/run_full_imagenet_pipeline.sh
```

Use the full official devkit archive or extracted devkit directory when
preparing ImageNet-val. The validation ground-truth file stores
`ILSVRC2012_ID` labels, and the project needs `meta.mat` from the devkit to map
those labels back to ImageNet-1k WNID class directories.

## Quick Smoke

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

Expected final artifact:

```text
artifacts/smoke/reports/results.md
```

## Current State

The engineering pipeline is implemented and covered by CI. The full 5M teacher
logits cache has not been produced in this repository yet; that requires a full
ImageNet-val layout and a CUDA worker. See [docs/full-run.md](docs/full-run.md)
and [docs/gpu-run.md](docs/gpu-run.md) for the operational path.
