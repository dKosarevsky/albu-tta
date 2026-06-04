# albu-tta implementation status

Status: implementation-complete for the planned lightweight end-to-end pipeline.

Implemented:

- Project bootstrap, CI, pytest, coverage, Ruff, ty, badges, and artifact policy.
- Strict ImageNet-val preflight requiring 1,000 WNID classes, 50,000 images,
  50 images per class, and a configured class index matching `timm-imagenet-1k`.
- Stratified ImageNet-val manifests for `public_train`, `public_val`,
  `public`, and `private`, plus `class_to_idx.json` for label-index audit.
- Clean identity-cache baseline gate on `public_val` before full all-candidate
  teacher caching, with configurable top-1, top-5, and NLL sanity thresholds.
- AlbumentationsX registry with 100 single-transform candidates, explicit
  `fixed` vs `seeded_stochastic` determinism metadata, range validation for
  fixed candidates, and audit JSON including runtime package versions.
- Teacher cache runner for timm ResNet50 preprocessing, fp16 logits, parquet metadata, `.run.json` sidecars, and metadata-aware resume checks.
- Selector target generation from cached logits with clean-vs-augmentation gain targets.
- Selector target formulation documented as a 100-score augmentation utility predictor for ranking/top-k TTA selection, not 50-bin loss classification.
- Public/private split-role guards for target building, selector checkpoint selection, TTA tuning, learned aggregation training, and final private evaluation.
- Small selector CNN training with standardized targets, SmoothL1, pairwise rank loss, public-val diagnostics, and checkpoint selection by public-val learned TTA NLL.
- TTA evaluation for clean, fixed, random, all-candidate, learned uniform, learned softmax-weighted, oracle, global non-negative aggregation, class-specific aggregation, and optional XGBoost stacker.
- Private evaluation artifacts with metrics, compute, clean-vs-TTA corrections, and metric deltas against clean.
- Final report builder with tables, SVG plots, top-N markdown summaries, transform-class impact, transform-class aggregation, selector history, XGBoost importance, and result text that avoids SOTA claims.
- Synthetic smoke run covering manifests, teacher cache, target build, selector training, tuning, aggregation training, private evaluation, and final report generation.
- Full-run artifact status with strict teacher-cache completeness checks across all configured augmentation candidates, `.run.json` metadata validation, and missing/extra artifact diagnostics.
- CI coverage gate fixed at 98.5% minimum for the `learned_tta` package.
- Script-friendly full-run status exit mode via `--fail-on-incomplete`.
- Script-friendly next required command output via `--next-command`.
- Supervised `resume-full-run` CLI for Colab reconnects: it runs the next
  missing required step, starts long `cache-teacher` steps in the background,
  writes cache logs to a persistent directory, and avoids duplicate active cache
  jobs.
- Google Colab runbook and notebook for resumable GPU execution with Drive-backed artifacts and reports.
- External GPU runbook for RunPod, Lambda Labs, local CUDA servers, or other
  non-Colab workers, including persistent artifact layout, one-step resume,
  automated driver, and recovery checklist.

Not part of the implementation-complete status:

- Running the full ImageNet validation experiment.
- Producing paper-ready numeric claims.
- Claiming SOTA or cross-architecture generality.

Next research step:

Run `check-full-run` against the local ImageNet-val directory, confirm the
strict class-count/image-count/WNID-mapping checks pass, let
`resume-full-run` cache `public_val` identity and run `check-clean-baseline`,
then run the full
experiment with `resnet50.a1_in1k`, use `full-run-status` after each expensive
step to confirm the next missing required artifact, use `--format json` if the
GPU run is driven by scripts, use `--fail-on-incomplete` when a shell step
should stop on incomplete required artifacts, use `--next-command` when a
wrapper needs to dispatch only the next required command, prefer
`resume-full-run` for interactive Colab reconnects or external GPU recovery,
follow `docs/gpu-run.md` when Colab quota is the blocker, review
`reports/resnet50_a1_in1k/results.md`, then decide which additional timm
architectures are worth running for the preprint. The optional XGBoost stacker
is tracked separately and does not block the required full-run status.
