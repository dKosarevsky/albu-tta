# Google Colab full run

This runbook is for launching the full ImageNet validation experiment from
Google Colab. Use it with
`notebooks/full_imagenet_run_colab.ipynb`.

Do not paste API keys into the notebook or repository. Ordinary Colab does not
need a project API key for this workflow. The notebook uses your browser session
to mount Google Drive, and all long-lived outputs are written under Drive so
cache resume can survive Colab disconnects.

## Requirements

- Colab runtime with a GPU selected.
- ImageNet validation data available inside Colab at `IMAGENET_VAL_DIR`.
- ImageNet layout: `val/class_name/image.JPEG`.
- Enough Google Drive space for `artifacts` and `reports`.

The notebook prefers a locally prepared validation folder and falls back to a
Google Drive copy:

```text
LOCAL_IMAGENET_VAL_DIR=/content/imagenet_val_prepare/val
DRIVE_IMAGENET_VAL_DIR=/content/drive/MyDrive/datasets/imagenet/val
IMAGENET_VAL_DIR=LOCAL_IMAGENET_VAL_DIR if present, otherwise DRIVE_IMAGENET_VAL_DIR
DRIVE_RUN_ROOT=/content/drive/MyDrive/albu-tta-runs/resnet50_a1_in1k
```

Keep ImageNet in `/content` when possible; copying 50k small validation files to
Drive can be much slower than downloading and preparing them locally. Change
these paths in the notebook before launching the expensive steps if your layout
differs.

## Execution Model

The notebook clones this repository into `/content/albu-tta`, installs
dependencies with `uv`, and links these repository paths to Google Drive:

```text
/content/albu-tta/artifacts -> DRIVE_RUN_ROOT/artifacts
/content/albu-tta/reports   -> DRIVE_RUN_ROOT/reports
```

All orchestration goes through `full-run-status --next-command`. The notebook
asks the repository which command is next, replaces the ImageNet placeholder
with `IMAGENET_VAL_DIR`, and runs exactly one command at a time. Re-run the
status cell after a disconnect; already complete teacher cache shards are
skipped by cache resume when parquet, logits, and `.run.json` sidecars match the
current run metadata.

## Recommended Flow

1. Open `notebooks/full_imagenet_run_colab.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Mount Google Drive.
4. Set `IMAGENET_VAL_DIR` and `DRIVE_RUN_ROOT`.
5. Run setup and GPU checks.
6. Run the next-command cell until it reports `none`.
7. Inspect `reports/resnet50_a1_in1k/results.md` under `DRIVE_RUN_ROOT`.

The teacher cache is the expensive part. Public and private all-candidate cache
cover roughly 5,000,000 ResNet50 forwards total. Colab can disconnect, so the
safe operating mode is one `full-run-status --next-command` step per cell run,
not a blind shell script that hides intermediate status.

## Troubleshooting

- If GPU is unavailable, change Runtime -> Change runtime type -> GPU.
- If `check-full-run` fails, fix `IMAGENET_VAL_DIR` first and verify that it
  contains exactly 50,000 `*.JPEG` files under class directories.
- If Drive is slow, keep the repository in `/content` and only persist
  `artifacts` and `reports`, which is what the notebook does.
- If Colab disconnects during `cache-teacher`, reconnect and re-run the status
  cell. Complete shards should be skipped.
- If package installation is interrupted, re-run the setup cell; `uv` will
  reuse its cache where possible.
