# albu-tta implementation status

Status: implementation-complete for the planned lightweight end-to-end pipeline.

Implemented:

- Project bootstrap, CI, pytest, coverage, Ruff, ty, badges, and artifact policy.
- Strict ImageNet-val preflight requiring 1,000 WNID classes, 50,000 images,
  50 images per class, and a configured class index matching `timm-imagenet-1k`.
- CPU-only `prepare-imagenet-val` CLI for converting local official
  `ILSVRC2012_img_val.tar` plus devkit labels into `val/WNID/*.JPEG` layout.
- Stratified ImageNet-val manifests for `public_train`, `public_val`,
  `public`, and `private`, plus `class_to_idx.json` for label-index audit.
- Clean identity-cache baseline gate on `public_val` before full all-candidate
  teacher caching, with configurable top-1, top-5, and NLL sanity thresholds.
- Full-validation clean CenterCrop summary from identity `aug_000` shards across
  `public_train`, `public_val`, and `private`, with optional inference
  throughput benchmark sidecars.
- Teacher-cache planning CLI that reports the expected 5M teacher forward
  passes, 300 shard target, missing files, stale/malformed shards, fp16 logits
  size, and the next split-level `cache-teacher` command.
- Teacher backend planning CLI and explicit `cache-teacher --backend pytorch`
  guard. TensorRT, ONNXRuntime, and OpenVINO are documented as planned
  accelerators and fail early if requested before implementation.
- Metadata-only teacher-cache diagnostics for completed splits: clean NLL/top-1/top-5,
  helpful/harmful augmentation fractions, best non-identity single augmentation,
  oracle best per-image gain, and top augmentation summaries.
- AlbumentationsX registry with 100 single-transform candidates, explicit
  `fixed` vs `seeded_stochastic` determinism metadata, range validation for
  fixed candidates, and audit JSON including runtime package versions.
- Teacher cache runner for timm ResNet50 preprocessing, fp16 logits, parquet metadata, `.run.json` sidecars, and metadata/shape-aware resume checks.
- Selector target generation from cached logits with clean-vs-augmentation gain
  targets, persisted `image_id` lineage, and trainable ablation target kinds:
  `gain`, `negative_nll`, `helpfulness`, `rank`, `softmax_weight`, and
  `true_logit`.
- Selector target formulation documented as a 100-score augmentation utility predictor for ranking/top-k TTA selection, not 50-bin loss classification.
- Public/private split-role guards for target building, selector checkpoint selection, TTA tuning, learned aggregation training, and final private evaluation.
- Small selector CNN training with standardized targets, SmoothL1, pairwise rank loss, optional usefulness BCE head, public-val diagnostics, and checkpoint selection by public-val learned TTA NLL.
- TTA evaluation for clean, fixed, random, all-candidate, learned uniform, learned softmax-weighted, learned adaptive uniform, oracle, global non-negative aggregation, class-specific aggregation, and optional XGBoost stacker.
- Private evaluation artifacts with metrics, compute, clean-vs-TTA corrections, and metric deltas against clean.
- Final report builder with tables, SVG plots, top-N markdown summaries, transform-class impact, transform-class aggregation, selector history, XGBoost importance, and result text that avoids SOTA claims.
- Synthetic smoke run covering manifests, teacher cache, target build, selector training, tuning, aggregation training, private evaluation, and final report generation.
- Full-run artifact status with strict teacher-cache completeness checks across all configured augmentation candidates, `.run.json` metadata validation, row/class-shape validation, and missing/extra artifact diagnostics.
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
- GPU handoff checklist for delegating the full ImageNet run to another worker
  and collecting the resulting artifacts.

Not part of the implementation-complete status:

- Running the full ImageNet validation experiment.
- Producing the full 5M teacher logits cache.
- Producing paper-ready numeric claims.
- Claiming SOTA or cross-architecture generality.
- Running teacher cache through TensorRT, ONNXRuntime, or OpenVINO. PyTorch is
  the only implemented teacher-cache backend today.
- Deciding the final best second-level target formulation. The current primary
  baseline remains the 100-output gain predictor; materialized target variants,
  clean-logits/tabular selector, global aggregation, class-specific aggregation,
  and XGBoost stacking variants are planned ablations after the 5M logits cache
  exists. The target files can be produced now; deciding which one to use is
  still a planned ablation after the 5M logits cache exists. These remain
  ablations after the 5M logits cache exists.

Next research step:

Run `check-full-run` against the local ImageNet-val directory, confirm the
strict class-count/image-count/WNID-mapping checks pass, let
`resume-full-run` cache `public_val` identity and run `check-clean-baseline`,
optionally run the clean CenterCrop-only baseline commands from
`docs/gpu-run.md`, then produce the full 5M logits cache with 300 complete
teacher-cache shards: 100 for `public_train`, 100 for `public_val`, and 100 for
`private`. Use `teacher-cache-plan` to inspect logits-cache progress, use
`full-run-status` after each expensive step to confirm the next missing
required artifact, use `--format json` if the GPU run is driven by scripts, use
`--fail-on-incomplete` when a shell step should stop on incomplete required
artifacts, use `--next-command` when a wrapper needs to dispatch only the next
required command, prefer `resume-full-run` for interactive Colab
reconnects or external GPU recovery, follow `docs/gpu-run.md` when Colab quota
is the blocker, review
`reports/resnet50_a1_in1k/results.md`, then decide which additional timm
architectures are worth running for the preprint. The optional XGBoost stacker
is tracked separately and does not block the required full-run status.
