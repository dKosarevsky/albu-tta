# Method

The project keeps two TTA learning layers separate.

- `learned_topk_uniform` and `learned_topk_softmax_weighted`: an
  image-conditioned selector CNN predicts which augmentations are worth running
  for each image.
- `global_weighted_tta`: a paper-style second-level aggregator learns one
  sparse non-negative weight per augmentation from cached public-val
  predictions.
- `class_weighted_tta`: the same learned aggregation idea, but with separate
  sparse non-negative augmentation weights per output class.
- `xgboost_multiclass`: an optional tabular stacker over flattened per-policy
  probabilities. It is trained from cached public-val predictions and evaluated
  on private when its artifact is present.

This separation keeps article diagnostics clear: selection answers which
AlbumentationsX transforms are useful, while aggregation answers how strongly to
combine predictions once TTA views are available.

Standard non-learned comparison rows are reported separately: clean CenterCrop,
CenterCrop plus horizontal flip, and optionally classic 10-crop when its logits
artifact has been generated. These are deployment baselines, while private
oracle rows remain diagnostic upper bounds.

## Selector Target Formulation

The selector is an augmentation utility predictor: it predicts
100 augmentation utility scores and is not a 50-bin loss classification task.
The teacher cache keeps raw true-class NLL for every image and augmentation,
but the selector target is a relative utility:

```text
gain = clean_nll - aug_nll
```

Positive gain means the augmentation increased the teacher probability assigned
to the correct class compared with clean inference. The deployed decision is
ranking/top-k TTA selection, so the small CNN has one output per configured
candidate rather than assigning an image to a discrete loss bin.

The selector is trained on standardized gain targets, but checkpoints persist
the public-train target mean and std. Inference converts selector outputs back
to the original gain scale before ranking augmentations. The default selector
objective is mixed: SmoothL1 gain regression, pairwise ranking loss, and an
optional usefulness BCE head trained on `gain > usefulness_tau` for non-identity
augmentations. During training, the best checkpoint is selected by
public-validation `learned_topk_uniform` TTA NLL when the validation teacher
cache is available; regression loss, ranking loss, usefulness BCE, Spearman
correlation, TTA metrics, and oracle top-k recall are written to
`selector/selector_history.csv`.

The current primary baseline is the 100-output gain predictor. The
planned ablations after the full 5M teacher logits cache can compare alternative
second-level heads, but the main claim does not depend on those variants. The
usefulness head supports adaptive TTA: `tune-tta` selects a public-val
`useful_prob` threshold and `max_k`, then private evaluation reports
`learned_adaptive_uniform` only from those frozen public-val choices.
`build-targets --target-kind` can currently emit trainable high-is-better
targets for `gain`, `negative_nll`, `helpfulness`, `rank`, `softmax_weight`,
and `true_logit`. Raw `nll` remains diagnostic-only; use `negative_nll` when
the selector should learn a loss-derived utility.

The implemented target variants cover the earlier planned
100 binary helpfulness labels (`gain > 0`) and softmax-weight distillation from
public-train oracle utilities. Second-level model ablations include
clean-logits/tabular selectors, global non-negative aggregation,
class-specific aggregation, and XGBoost stacking. Selector utility scores may
be negative because they can represent predicted gain or negative NLL, but
deployed probability TTA weights should remain non-negative, for example
uniform top-k or softmax over selected utilities.

The pairwise selector ablation is a separate lightweight path for testing the
same signal with a different inductive bias. `train-pairwise-selector` scores
one `(image, augmentation)` row at a time from clean-pass uncertainty features,
optional cached image features, and augmentation identity. `train-pairwise-selector-comparison`
trains the two current objective variants, NLL gain and top-1 delta, and writes
`pairwise_selector_comparison.csv`; `build-report` can render that table plus a
selector error-analysis table when those CSVs are supplied.

Selector target artifacts also persist `image_id` order. Training refuses to
pair a manifest with targets whose rows were generated from a different image
order, which prevents silent label/target drift after interrupted or moved
runs. Migration note: selector target `.npz` files produced before `image_id`
lineage was added are intentionally rejected by training. Rebuild them with:

```bash
uv run python -m learned_tta.cli build-targets \
  --config configs/experiment/resnet50_a1_in1k.yaml
```

## Split Contract

The code treats public/private separation as an API contract, not only a
runbook convention. Selector targets are built only from `public_train` and `public_val`.
Selector checkpoint selection, `tune-tta`, and learned aggregation training use `public_val`.
`evaluate-private` accepts only `private` and reads a frozen public-val tuning
artifact.

Private oracle rows are diagnostics and upper bounds, not deployable methods or tuning inputs.
Private labels must not be used to choose target kind, top-k,
softmax temperature, aggregation weights, or hyperparameters.

The default split shape is:

```text
public_train: 20 images per class, 20,000 images total
public_val:    5 images per class,  5,000 images total
private:      25 images per class, 25,000 images total
```

`public` is the union of `public_train` and `public_val` and is kept for audit
and reporting convenience, not for selector tuning.
