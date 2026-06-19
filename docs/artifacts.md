# Artifacts And Diagnostics

Generated data is intentionally kept out of git. For the default experiment,
the repository expects these generated roots:

```text
artifacts/manifests
artifacts/teacher_cache
artifacts/selector
reports/resnet50_a1_in1k
```

## Augmentation Audit

`validate-augmentations --audit-output` writes a stable JSON audit of the exact
AlbumentationsX candidate ids, transform classes, parameters, experiment seed,
serialized AlbumentationsX `Compose` payloads, and runtime package versions used
in the run:

```text
artifacts/augmentation_registry_audit.json
```

## Teacher Cache

Each teacher cache shard writes a `.run.json` sidecar next to the parquet
metadata and fp16 logits. The sidecar stores the augmentation id and params,
seed, model name, pretrained flag, timm data config, class count, and storage
format. Resume checks compare this metadata before skipping an existing shard,
so changing the teacher checkpoint, preprocessing, candidate params, or seed
forces recomputation instead of silently reusing stale logits.

Teacher cache writes also emit optional `.benchmark.json` sidecars with elapsed
time and throughput per shard; those sidecars are not part of resume
completeness.

Use `teacher-cache-plan` when the question is specifically "how much of the 5M
logits cache is done?". It reports expected images, teacher forward passes,
complete shards, missing files, stale/malformed shards, fp16 logits size, and
the next `cache-teacher --split ...` command for `public_train`, `public_val`,
and `private`.

The current teacher inference backend is explicit:
`cache-teacher --backend pytorch` is the implemented correctness path.
`teacher-backend-plan` documents planned accelerators for future throughput
work: TensorRT for CUDA, ONNXRuntime as a portable exported-model runtime, and
OpenVINO for CPU. Passing a planned backend to `cache-teacher` fails early
instead of silently running PyTorch under a misleading backend name.

After a split cache is complete, `teacher-cache-diagnostics` reads only parquet
metadata and reports clean NLL/top-1/top-5, helpful and harmful augmentation
fractions, best non-identity single augmentation by mean gain, oracle best
per-image mean gain, and a top augmentation table. This is the first cheap
post-logits check before deciding which selector target or aggregation ablation
is worth training.

## Reports

`build-report` writes aggregation diagnostics when learned aggregator artifacts
are present, and copies private clean-vs-TTA diagnostics when `evaluate-private`
has produced them:

```text
reports/resnet50_a1_in1k/tables/aggregation_weights.csv
reports/resnet50_a1_in1k/tables/class_augmentation_weights.csv
reports/resnet50_a1_in1k/tables/xgboost_feature_importance.csv
reports/resnet50_a1_in1k/tables/private_metric_deltas.csv
reports/resnet50_a1_in1k/tables/corrections.csv
reports/resnet50_a1_in1k/tables/global_weight_topn_private_metrics.csv
reports/resnet50_a1_in1k/tables/selector_history.csv
reports/resnet50_a1_in1k/tables/transform_class_impact.csv
reports/resnet50_a1_in1k/tables/transform_class_aggregation.csv
reports/resnet50_a1_in1k/figures/gain_distribution.svg
reports/resnet50_a1_in1k/figures/oracle_overlap.svg
reports/resnet50_a1_in1k/figures/aggregation_weights.svg
reports/resnet50_a1_in1k/figures/xgboost_feature_importance.svg
reports/resnet50_a1_in1k/figures/corrections.svg
reports/resnet50_a1_in1k/figures/selector_history.svg
reports/resnet50_a1_in1k/figures/transform_class_impact.svg
reports/resnet50_a1_in1k/figures/transform_class_aggregation.svg
```

`global_weight_topn_private_metrics.csv` is written by `evaluate-private`
whenever a global aggregator artifact is available. It keeps the highest-weight
global augmentations for each `top_n` value and reports the resulting private
metrics plus the relative forward-pass cost.

`aggregation_weights.csv` is the compact table for the article: global weight,
active flag, mean class weight, max class weight, and class activation
frequency per augmentation. The class-level long table is kept for deeper
diagnosis of which ImageNet classes benefit from which AlbumentationsX
transforms.

Augmentation-level report tables include `augmentation_name` and
`transform_class` columns resolved from the registry, so impact and learned
weight tables remain interpretable without a separate id lookup.
`transform_class_impact.csv` groups mean gain, learned selection frequency, and
oracle frequency by AlbumentationsX transform class for a compact article-level
view of which transform families matter.

`results.md` embeds compact top-N markdown tables for mean-gain augmentations,
learned selector choices, oracle choices, transform classes, learned
aggregation weights, and XGBoost feature importance, so the main article
artifact can be reviewed without opening every CSV first.

`transform_class_aggregation.csv` groups global and class-specific aggregation
weights by AlbumentationsX transform class, complementing the per-augmentation
weight table with a family-level view.

`public_metrics.csv` combines the tuned public-val `learned_topk_uniform`
result with public-val metrics saved inside the optional learned aggregation
artifacts, so global, class-specific, and XGBoost stackers can be compared
before private evaluation.

`private_metric_deltas.csv` compares each private strategy against clean
ResNet50 on top-1, top-5, NLL, ECE, and compute, making the final trade-off
table explicit. `compute.csv` keeps compute rows for both `public_val` and
`private` with an explicit `split` column, so public tuning diagnostics and
final private compute costs are not mixed implicitly.

Aggregator training uses the historical `--l1-penalty` CLI option as a
sparsity regularizer and then prunes weights at or below `active_threshold`;
this makes zero-weight TTA candidates explicit in the saved artifact and
report tables.

The optional XGBoost stacker is deliberately not a default dependency; install
the stacker extra with `uv sync --extra stackers` before running:

```bash
uv run python -m learned_tta.cli train-aggregator \
  --method xgboost-multiclass \
  --config configs/experiment/resnet50_a1_in1k.yaml \
  --split public_val
```

When the XGBoost artifact is present, `build-report` writes
`xgboost_feature_importance.csv` and `xgboost_feature_importance.svg` to show
which augmentation candidates the stacker uses most. The XGBoost metadata stores
the model path relative to the artifact file when both files are in the same
artifact directory, so selector artifacts can be moved as a bundle.

`corrections.csv` counts where a strategy fixes a clean ResNet50 mistake and
where it breaks an originally correct clean prediction. This follows the TTA
diagnostic framing from "Better Aggregation in Test-Time Augmentation": average
dataset gain is not enough, because TTA can help and hurt different images. For
`random_topk`, correction counts are averaged across the same random seeds used
for the private metric row.

When private evaluation artifacts live outside the report directory, pass
`--corrections /path/to/corrections.csv` to `build-report`.
