"""Experiment configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from learned_tta.imagenet_split import SplitConfig


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    """Teacher model configuration."""

    model_name: str
    pretrained: bool
    data_config: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """ImageNet validation split shape configuration."""

    name: str
    class_count: int
    class_index: str
    images_per_class: int


@dataclass(frozen=True, slots=True)
class CleanBaselineConfig:
    """Sanity thresholds for clean teacher identity-cache metrics."""

    split: str
    min_top1: float
    min_top5: float
    max_nll: float


@dataclass(frozen=True, slots=True)
class AugmentationsConfig:
    """Augmentation registry configuration."""

    registry_path: Path
    candidate_count: int
    identity_id: str


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    """Selector training configuration."""

    output_dim: int
    max_parameters: int
    top_k_grid: list[int]
    usefulness_head: bool
    usefulness_tau: float
    usefulness_weight: float
    adaptive_threshold_grid: list[float]
    adaptive_max_k_grid: list[int]


@dataclass(frozen=True, slots=True)
class ArtifactsConfig:
    """Generated artifact paths."""

    root: Path
    manifests_dir: Path
    teacher_cache_dir: Path
    selector_dir: Path
    reports_dir: Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    path: Path
    project_root: Path
    project_name: str
    seed: int
    teacher: TeacherConfig
    dataset: DatasetConfig
    clean_baseline: CleanBaselineConfig
    split: SplitConfig
    augmentations: AugmentationsConfig
    selector: SelectorConfig
    artifacts: ArtifactsConfig


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment YAML file and resolve project-relative paths."""

    path = Path(path).resolve()
    project_root = _find_project_root(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    dataset = raw["dataset"]
    clean_baseline = raw.get("clean_baseline", {})
    augmentations = raw["augmentations"]
    teacher = raw["teacher"]
    selector = raw["selector"]
    artifacts = raw["artifacts"]

    return ExperimentConfig(
        path=path,
        project_root=project_root,
        project_name=str(raw["project_name"]),
        seed=int(raw["seed"]),
        teacher=TeacherConfig(
            model_name=str(teacher["model_name"]),
            pretrained=bool(teacher["pretrained"]),
            data_config=(
                dict(teacher["data_config"])
                if isinstance(teacher.get("data_config"), dict)
                else None
            ),
        ),
        dataset=DatasetConfig(
            name=str(dataset["name"]),
            class_count=int(dataset["class_count"]),
            class_index=str(dataset["class_index"]),
            images_per_class=int(dataset["images_per_class"]),
        ),
        clean_baseline=CleanBaselineConfig(
            split=str(clean_baseline.get("split", "public_val")),
            min_top1=float(clean_baseline.get("min_top1", 0.70)),
            min_top5=float(clean_baseline.get("min_top5", 0.90)),
            max_nll=float(clean_baseline.get("max_nll", 1.60)),
        ),
        split=SplitConfig(
            seed=int(raw["seed"]),
            public_per_class=int(dataset["public_per_class"]),
            private_per_class=int(dataset["private_per_class"]),
            public_train_per_class=int(dataset["public_train_per_class"]),
            public_val_per_class=int(dataset["public_val_per_class"]),
        ),
        augmentations=AugmentationsConfig(
            registry_path=_resolve_path(project_root, augmentations["registry_path"]),
            candidate_count=int(augmentations["candidate_count"]),
            identity_id=str(augmentations["identity_id"]),
        ),
        selector=SelectorConfig(
            output_dim=int(selector["output_dim"]),
            max_parameters=int(selector["max_parameters"]),
            top_k_grid=[int(k) for k in selector["top_k_grid"]],
            usefulness_head=bool(selector.get("usefulness_head", False)),
            usefulness_tau=float(selector.get("usefulness_tau", 0.01)),
            usefulness_weight=float(selector.get("usefulness_weight", 0.0)),
            adaptive_threshold_grid=[
                float(threshold)
                for threshold in selector.get(
                    "adaptive_threshold_grid",
                    [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75],
                )
            ],
            adaptive_max_k_grid=[
                int(max_k) for max_k in selector.get("adaptive_max_k_grid", selector["top_k_grid"])
            ],
        ),
        artifacts=ArtifactsConfig(
            root=_resolve_path(project_root, artifacts["root"]),
            manifests_dir=_resolve_path(project_root, artifacts["manifests_dir"]),
            teacher_cache_dir=_resolve_path(project_root, artifacts["teacher_cache_dir"]),
            selector_dir=_resolve_path(project_root, artifacts["selector_dir"]),
            reports_dir=_resolve_path(project_root, artifacts["reports_dir"]),
        ),
    )


def _find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return config_path.parent


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path
