# albu-tta implementation status

Status: implementation-complete for the planned lightweight end-to-end pipeline.

Implemented:

- Project bootstrap, CI, pytest, coverage, Ruff, ty, badges, and artifact policy.
- Stratified ImageNet-val manifests for `public_train`, `public_val`, `public`, and `private`.
- AlbumentationsX registry with 100 deterministic single-transform candidates and audit JSON including runtime package versions.
- Teacher cache runner for timm ResNet50 preprocessing, fp16 logits, parquet metadata, `.run.json` sidecars, and metadata-aware resume checks.
- Selector target generation from cached logits with clean-vs-augmentation gain targets.
- Selector target formulation documented as a 100-score augmentation utility predictor for ranking/top-k TTA selection, not 50-bin loss classification.
- Public/private split-role guards for target building, selector checkpoint selection, TTA tuning, learned aggregation training, and final private evaluation.
- Small selector CNN training with standardized targets, SmoothL1, pairwise rank loss, public-val diagnostics, and checkpoint selection by public-val learned TTA NLL.
- TTA evaluation for clean, fixed, random, all-candidate, learned uniform, learned softmax-weighted, oracle, global non-negative aggregation, class-specific aggregation, and optional XGBoost stacker.
- Private evaluation artifacts with metrics, compute, clean-vs-TTA corrections, and metric deltas against clean.
- Final report builder with tables, SVG plots, top-N markdown summaries, transform-class impact, transform-class aggregation, selector history, XGBoost importance, and result text that avoids SOTA claims.
- Synthetic smoke run covering manifests, teacher cache, target build, selector training, tuning, aggregation training, private evaluation, and final report generation.
- Full-run artifact status with strict teacher-cache completeness checks across all configured augmentation candidates and missing/extra artifact diagnostics.
- Script-friendly full-run status exit mode via `--fail-on-incomplete`.
- Script-friendly next required command output via `--next-command`.

Not part of the implementation-complete status:

- Running the full ImageNet validation experiment.
- Producing paper-ready numeric claims.
- Claiming SOTA or cross-architecture generality.

Next research step:

Run `check-full-run` against the local ImageNet-val directory, run the full
experiment with `resnet50.a1_in1k`, use `full-run-status` after each expensive
step to confirm the next missing required artifact, use `--format json` if the
GPU run is driven by scripts, use `--fail-on-incomplete` when a shell step
should stop on incomplete required artifacts, use `--next-command` when a
wrapper needs to dispatch only the next required command, review
`reports/resnet50_a1_in1k/results.md`, then decide which additional timm
architectures are worth running for the preprint. The optional XGBoost stacker
is tracked separately and does not block the required full-run status.
