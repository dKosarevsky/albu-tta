from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from learned_tta.cache import TeacherShard, write_teacher_shard
from learned_tta.teacher_cache_plan import (
    build_teacher_cache_plan,
    teacher_cache_plan_to_dict,
)


def test_build_teacher_cache_plan_reports_expected_work(tmp_path: Path) -> None:
    config_path = _write_experiment_config(tmp_path)

    plan = build_teacher_cache_plan(config_path)

    assert plan.expected_shards == 300
    assert plan.complete_shards == 0
    assert plan.total_predictions == 800
    assert plan.logits_bytes_estimate == 3_200
    assert plan.complete is False
    assert [split.split for split in plan.splits] == [
        "public_train",
        "public_val",
        "private",
    ]
    assert plan.splits[0].expected_images == 2
    assert plan.splits[1].expected_images == 2
    assert plan.splits[2].expected_images == 4
    assert plan.splits[0].next_command is not None
    assert "cache-teacher --split public_train" in plan.splits[0].next_command
    assert "--device cuda" in plan.splits[0].next_command


def test_build_teacher_cache_plan_counts_complete_and_stale_shards(
    tmp_path: Path,
) -> None:
    config_path = _write_experiment_config(tmp_path)
    cache_dir = tmp_path / "project" / "artifacts" / "teacher_cache"
    _write_complete_teacher_shard(cache_dir, split="public_train", aug_id="aug_000")
    _write_complete_teacher_shard(cache_dir, split="public_train", aug_id="aug_001")
    np.save(cache_dir / "public_train__aug_001.logits.npy", np.zeros((1, 2), dtype=np.float16))

    plan = build_teacher_cache_plan(config_path)
    public_train = plan.splits_by_name["public_train"]

    assert plan.complete_shards == 1
    assert public_train.complete_shards == 1
    assert public_train.stale_or_malformed_shards == 1
    assert public_train.missing_files == 294
    assert public_train.completed_logits_bytes == 8
    assert public_train.complete is False


def test_teacher_cache_plan_to_dict_is_json_serializable(tmp_path: Path) -> None:
    config_path = _write_experiment_config(tmp_path)
    cache_dir = tmp_path / "cache"

    plan = build_teacher_cache_plan(config_path, cache_dir=cache_dir)
    payload = teacher_cache_plan_to_dict(plan)

    assert payload["cache_dir"] == str(cache_dir)
    assert payload["expected_shards"] == 300
    assert payload["complete_shards"] == 0
    assert payload["complete"] is False
    assert payload["splits"][2]["split"] == "private"
    assert payload["splits"][2]["expected_predictions"] == 400
    assert json.loads(json.dumps(payload)) == payload


def _write_experiment_config(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs" / "experiment"
    config_dir.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    config_path = config_dir / "experiment.yaml"
    config_path.write_text(
        """
project_name: albu-tta
seed: 20260522
teacher:
  model_name: resnet50.a1_in1k
  pretrained: true
  data_config:
    input_size: [3, 224, 224]
dataset:
  name: imagenet-val
  class_count: 2
  class_index: timm-imagenet-1k
  images_per_class: 4
  public_per_class: 2
  private_per_class: 2
  public_train_per_class: 1
  public_val_per_class: 1
clean_baseline:
  split: public_val
  min_top1: 0.70
  min_top5: 0.90
  max_nll: 1.60
augmentations:
  registry_path: configs/augmentations/imagenet100.yaml
  candidate_count: 100
  identity_id: aug_000
selector:
  output_dim: 100
  max_parameters: 1500000
  top_k_grid: [1, 2, 4, 8, 16]
artifacts:
  root: artifacts
  manifests_dir: artifacts/manifests
  teacher_cache_dir: artifacts/teacher_cache
  selector_dir: artifacts/selector
  reports_dir: reports/resnet50_a1_in1k
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def _write_complete_teacher_shard(cache_dir: Path, split: str, aug_id: str) -> None:
    rows = 2 if split in {"public_train", "public_val"} else 4
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split=split,
            aug_id=aug_id,
            image_ids=[f"{split}-{row_idx}" for row_idx in range(rows)],
            class_idxs=np.arange(rows, dtype=np.int64) % 2,
            logits=np.zeros((rows, 2), dtype=np.float32),
            run_metadata=_teacher_cache_run_metadata(split=split, aug_id=aug_id),
        ),
    )


def _teacher_cache_run_metadata(split: str, aug_id: str) -> dict[str, object]:
    return {
        "version": 1,
        "split": split,
        "aug_id": aug_id,
        "seed": 20260522,
        "teacher": {
            "model_name": "resnet50.a1_in1k",
            "pretrained": True,
            "num_classes": 2,
            "data_config": {"input_size": [3, 224, 224]},
        },
        "storage": {
            "logits_dtype": "float16",
            "metadata_format": "parquet",
        },
    }
