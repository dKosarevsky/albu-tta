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
    assert config.split.public_train_per_class == 20
    assert config.split.public_val_per_class == 5
    assert config.augmentations.registry_path == (
        ROOT / "configs" / "augmentations" / "imagenet100.yaml"
    )
    assert config.artifacts.manifests_dir == ROOT / "artifacts" / "manifests"
