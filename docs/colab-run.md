# Google Colab full run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dKosarevsky/albu-tta/blob/main/notebooks/full_imagenet_run_colab.ipynb)

This runbook is for launching the full ImageNet validation experiment from
Google Colab. Use it with
`notebooks/full_imagenet_run_colab.ipynb`.

If Colab GPU quota blocks the run, move the same persistent `artifacts/` and
`reports/` directory to an external GPU worker and follow
[`docs/gpu-run.md`](gpu-run.md). The external runbook uses the same
`resume-full-run` supervisor and can continue from completed shards.

Do not paste API keys into the notebook or repository. Ordinary Colab does not
need a project API key for this workflow. The notebook uses your browser session
to mount Google Drive, and all long-lived outputs are written under Drive so
cache resume can survive Colab disconnects.

## Requirements

- Colab runtime with a GPU selected.
- ImageNet validation data available inside Colab at `IMAGENET_VAL_DIR`.
- ImageNet layout: `val/WNID/image.JPEG`, using the ImageNet-1k WNID class
  directories expected by `timm-imagenet-1k`.
- Enough Google Drive space for `artifacts` and `reports`.

CPU runtimes are supported for setup and diagnostics only. They can mount
Drive, inspect existing artifacts, count completed shards, and print status, but
they should not run the full teacher cache. The full public/private
all-candidate cache is roughly 5,000,000 ResNet50 forwards.

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

## GPU Worker Handoff

Use this checklist when handing the run to someone with GPU access:

1. Open the notebook from GitHub with the Colab badge above.
2. Select a GPU runtime before running the GPU check or `resume-full-run`.
3. Mount Google Drive and set `IMAGENET_VAL_DIR` and `DRIVE_RUN_ROOT`.
4. Run setup and the read-only diagnostics cell.
5. Confirm ImageNet has exactly 1,000 WNID directories and 50,000 validation
   JPEG files.
6. Confirm diagnostics show no active duplicate `cache-teacher` process.
7. Run `resume-full-run`; for long teacher-cache steps it starts a background
   process and writes logs under `DRIVE_RUN_ROOT/logs`.
8. Re-run diagnostics after disconnects before launching another resume.
9. Treat `full-run-status --fail-on-incomplete` as the final completion gate.

The notebook in this repository is intentionally output-free. A local Colab
copy may contain old red errors after disconnects or failed path attempts; those
outputs are not the source of truth. Use the diagnostics cell, logs, and
`full-run-status` instead.

## Execution Model

The notebook clones this repository into `/content/albu-tta`, installs
dependencies with `uv`, and links these repository paths to Google Drive:

```text
/content/albu-tta/artifacts -> DRIVE_RUN_ROOT/artifacts
/content/albu-tta/reports   -> DRIVE_RUN_ROOT/reports
```

Interactive orchestration goes through `resume-full-run`. The notebook asks the
repository which command is next, replaces the ImageNet placeholder with
`IMAGENET_VAL_DIR`, and runs exactly one safe step at a time. Long
`cache-teacher` steps start in the background, write logs to Drive, and refuse
to duplicate an already active cache process. Re-run the resume cell after a
disconnect; already complete teacher cache shards are skipped by cache resume
when parquet, logits, and `.run.json` sidecars match the current run metadata.

`full-run-status --next-command` remains available for read-only diagnostics
and external wrappers that only need to print the next command.

Colab runs should keep teacher-cache workers conservative. The notebook pins
`cache-teacher` to `--num-workers 2`, matching the worker limit commonly
reported by T4 Colab runtimes and avoiding warning spam during long resumes.

## Recommended Flow

1. Open `notebooks/full_imagenet_run_colab.ipynb` in Google Colab.
2. Select a GPU runtime.
3. Mount Google Drive.
4. Set `IMAGENET_VAL_DIR` and `DRIVE_RUN_ROOT`.
5. Run setup, read-only diagnostics, and GPU checks.
6. Run the resume cell until it reports no required steps left.
7. Run `full-run-status --fail-on-incomplete`.
8. Inspect `reports/resnet50_a1_in1k/results.md` under `DRIVE_RUN_ROOT`.

The teacher cache is the expensive part. Public and private all-candidate cache
cover roughly 5,000,000 ResNet50 forwards total. Colab can disconnect, so the
safe operating mode is one `resume-full-run` step per cell run, not a blind
shell script that hides intermediate status.

## Troubleshooting

- If GPU is unavailable, change Runtime -> Change runtime type -> GPU.
- If Colab quota blocks GPU allocation, do not continue full teacher caching on
  CPU. Use CPU only for diagnostics, or move the notebook to another GPU
  provider/runtime with [`docs/gpu-run.md`](gpu-run.md).
- If `check-full-run` fails, fix `IMAGENET_VAL_DIR` first and verify that it
  contains exactly 1,000 WNID class directories, 50,000 `*.JPEG` files, and no
  unknown or missing ImageNet-1k class directory names.
- If Drive is slow, keep the repository in `/content` and only persist
  `artifacts` and `reports`, which is what the notebook does.
- If Colab disconnects during `cache-teacher`, reconnect and re-run the resume
  cell. Complete shards should be skipped.
- If terminal output still shows old red notebook errors, check
  `full-run-status --next-command` and the count of
  `artifacts/teacher_cache/*__*.run.json`; stale cell output is not the source
  of truth.
- If package installation is interrupted, re-run the setup cell; `uv` will
  reuse its cache where possible.
