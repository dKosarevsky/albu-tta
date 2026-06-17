# albu-tta ResNet50 Case Study

Tuned k: 16
Public-val oracle top-k recall: 0.1828

This report is a single-architecture ImageNet validation case study. Run additional architectures before making broad leaderboard claims.

## Public Validation Metrics

| strategy | top1 | top5 | nll | ece | forwards_per_image | relative_compute_vs_all |
| --- | --- | --- | --- | --- | --- | --- |
| learned_topk_uniform | 0.8174 | 0.9566 | 0.749915 | 0.0389969 | 17 | 0.17 |
| global_weighted_tta | 0.824 | 0.9596 | 0.703692 | 0.0270465 | 100 | 1 |
| class_weighted_tta | 0.894 | 0.9804 | 0.425668 | 0.0527389 | 100 | 1 |

## Private Metrics

| strategy | top1 | top5 | nll | ece | forwards_per_image | relative_compute_vs_all |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 0.80468 | 0.94592 | 0.939441 | 0.0867522 | 1 | 0.01 |
| fixed_light_tta | 0.80848 | 0.94052 | 0.857318 | 0.0460779 | 17 | 0.17 |
| random_topk | 0.80888 | 0.945448 | 0.821023 | 0.0307059 | 17 | 0.17 |
| all_100_uniform | 0.8102 | 0.94696 | 0.791093 | 0.0265595 | 100 | 1 |
| learned_topk_uniform | 0.80956 | 0.94904 | 0.82498 | 0.0449992 | 17 | 0.17 |
| learned_topk_softmax_weighted | 0.8096 | 0.94912 | 0.824894 | 0.0445476 | 17 | 0.17 |
| oracle_topk_uniform | 0.87492 | 0.96312 | 0.489845 | 0.0135182 | 17 | 0.17 |
| global_weighted_tta | 0.81488 | 0.94984 | 0.771035 | 0.0289746 | 100 | 1 |
| class_weighted_tta | 0.80536 | 0.94128 | 0.839763 | 0.030029 | 100 | 1 |

- Delta table: `tables/private_metric_deltas.csv`

### Private metric deltas vs clean

| strategy | top1_delta_vs_clean | top5_delta_vs_clean | nll_delta_vs_clean | ece_delta_vs_clean | forwards_per_image | relative_compute_vs_all |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 0 | 0 | 0 | 0 | 1 | 0.01 |
| fixed_light_tta | 0.0038 | -0.0054 | -0.0821228 | -0.0406743 | 17 | 0.17 |
| random_topk | 0.0042 | -0.000472 | -0.118418 | -0.0560463 | 17 | 0.17 |
| all_100_uniform | 0.00552 | 0.00104 | -0.148347 | -0.0601927 | 100 | 1 |
| learned_topk_uniform | 0.00488 | 0.00312 | -0.114461 | -0.041753 | 17 | 0.17 |
| learned_topk_softmax_weighted | 0.00492 | 0.0032 | -0.114547 | -0.0422045 | 17 | 0.17 |
| oracle_topk_uniform | 0.07024 | 0.0172 | -0.449596 | -0.073234 | 17 | 0.17 |
| global_weighted_tta | 0.0102 | 0.00392 | -0.168406 | -0.0577776 | 100 | 1 |
| class_weighted_tta | 0.00068 | -0.00464 | -0.0996776 | -0.0567231 | 100 | 1 |

## Compute

| split | strategy | forwards_per_image | relative_compute_vs_all |
| --- | --- | --- | --- |
| public_val | learned_topk_uniform | 17 | 0.17 |
| public_val | global_weighted_tta | 100 | 1 |
| public_val | class_weighted_tta | 100 | 1 |
| private | clean | 1 | 0.01 |
| private | fixed_light_tta | 17 | 0.17 |
| private | random_topk | 17 | 0.17 |
| private | all_100_uniform | 100 | 1 |
| private | learned_topk_uniform | 17 | 0.17 |
| private | learned_topk_softmax_weighted | 17 | 0.17 |
| private | oracle_topk_uniform | 17 | 0.17 |
| private | global_weighted_tta | 100 | 1 |
| private | class_weighted_tta | 100 | 1 |

## Augmentation Impact

- Table: `tables/augmentation_impact.csv`

### Top mean-gain augmentations

| aug_id | augmentation_name | transform_class | mean_gain |
| --- | --- | --- | --- |
| aug_029 | shear_y_plus_5 | Affine | 0.0110863 |
| aug_075 | gaussian_blur_sigma_05 | GaussianBlur | 0.00566818 |
| aug_022 | scale_095 | Affine | 0.00396476 |
| aug_028 | shear_y_minus_5 | Affine | 0.00191463 |
| aug_023 | scale_105 | Affine | 0.00150252 |

### Top learned-selection augmentations

| aug_id | augmentation_name | transform_class | selection_frequency |
| --- | --- | --- | --- |
| aug_010 | rotate_minus_20 | Rotate | 0.9214 |
| aug_078 | median_blur_3 | MedianBlur | 0.8774 |
| aug_009 | rotate_plus_10 | Rotate | 0.7952 |
| aug_066 | planckian_warm_blackbody | PlanckianJitter | 0.7768 |
| aug_069 | planckian_warm_cied | PlanckianJitter | 0.7644 |

### Top oracle-selection augmentations

| aug_id | augmentation_name | transform_class | oracle_frequency |
| --- | --- | --- | --- |
| aug_025 | scale_120 | Affine | 0.2868 |
| aug_005 | square_horizontal | SquareSymmetry | 0.244 |
| aug_024 | scale_110 | Affine | 0.2428 |
| aug_011 | rotate_plus_20 | Rotate | 0.2366 |
| aug_010 | rotate_minus_20 | Rotate | 0.2312 |

![Gain distribution](figures/gain_distribution.svg)

![Learned versus oracle overlap](figures/oracle_overlap.svg)

- Transform-class table: `tables/transform_class_impact.csv`

### Top transform classes by mean gain

| transform_class | candidate_count | mean_gain | selection_frequency | oracle_frequency |
| --- | --- | --- | --- | --- |
| identity | 1 | 0 | 1 | 1 |
| RandomGamma | 4 | -0.00208149 | 0.00045 | 0.1325 |
| UnsharpMask | 1 | -0.00408824 | 0 | 0.073 |
| RGBShift | 6 | -0.00673984 | 0.00126667 | 0.148267 |
| Enhance | 1 | -0.00780114 | 0.0048 | 0.1342 |

![Transform-class impact](figures/transform_class_impact.svg)

## Learned Aggregation Weights

- Table: `tables/aggregation_weights.csv`

### Top global aggregation weights

| aug_id | augmentation_name | transform_class | global_weight |
| --- | --- | --- | --- |
| aug_025 | scale_120 | Affine | 0.182504 |
| aug_011 | rotate_plus_20 | Rotate | 0.085652 |
| aug_029 | shear_y_plus_5 | Affine | 0.0730575 |
| aug_021 | scale_090 | Affine | 0.0667459 |
| aug_020 | scale_080 | Affine | 0.0635984 |

### Top class-mean aggregation weights

| aug_id | augmentation_name | transform_class | class_mean_weight |
| --- | --- | --- | --- |
| aug_072 | gray_weighted | ToGray | 0.0336097 |
| aug_025 | scale_120 | Affine | 0.0311969 |
| aug_011 | rotate_plus_20 | Rotate | 0.0193906 |
| aug_024 | scale_110 | Affine | 0.0187972 |
| aug_098 | downscale_025_area_linear | Downscale | 0.0181198 |

### Top class-active aggregation weights

| aug_id | augmentation_name | transform_class | class_active_frequency |
| --- | --- | --- | --- |
| aug_000 | identity | identity | 1 |
| aug_001 | square_r90 | SquareSymmetry | 1 |
| aug_002 | square_r180 | SquareSymmetry | 1 |
| aug_003 | square_r270 | SquareSymmetry | 1 |
| aug_004 | square_vertical | SquareSymmetry | 1 |

![Aggregation weights](figures/aggregation_weights.svg)

- Transform-class aggregation table: `tables/transform_class_aggregation.csv`

![Transform-class aggregation](figures/transform_class_aggregation.svg)

- Class-specific table: `tables/class_augmentation_weights.csv`

## Corrections and Corruptions

- Table: `tables/corrections.csv`

![TTA corrections and corruptions](figures/corrections.svg)

## Selector Training Diagnostics

- Table: `tables/selector_history.csv`

![Selector training diagnostics](figures/selector_history.svg)
