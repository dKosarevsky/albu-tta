from __future__ import annotations

from pathlib import Path

from learned_tta.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment" / "resnet50_a1_in1k.yaml"


def test_load_experiment_config_resolves_project_relative_paths() -> None:
    config = load_experiment_config(CONFIG_PATH)

    assert config.project_name == "albu-tta"
    assert config.seed == 20260522
    assert config.teacher.model_name == "resnet50.a1_in1k"
    assert config.dataset.class_count == 1000
    assert config.dataset.class_index == "timm-imagenet-1k"
    assert config.dataset.images_per_class == 50
    assert config.clean_baseline.split == "public_val"
    assert config.clean_baseline.min_top1 == 0.70
    assert config.clean_baseline.min_top5 == 0.90
    assert config.clean_baseline.max_nll == 1.60
    assert config.split.public_train_per_class == 20
    assert config.split.public_val_per_class == 5
    assert config.selector.usefulness_head is True
    assert config.selector.usefulness_tau == 0.01
    assert config.selector.usefulness_weight == 0.05
    assert config.selector.adaptive_threshold_grid == [
        0.01,
        0.03,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.5,
        0.75,
    ]
    assert config.selector.adaptive_max_k_grid == [0, 1, 2, 4, 8, 16]
    assert config.augmentations.registry_path == (
        ROOT / "configs" / "augmentations" / "imagenet100.yaml"
    )
    assert config.artifacts.manifests_dir == ROOT / "artifacts" / "manifests"
