from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from learned_tta.cache import TeacherShard, write_teacher_shard
from learned_tta.cli import main
from learned_tta.teacher_cache_diagnostics import (
    summarize_teacher_cache_diagnostics,
    teacher_cache_diagnostics_to_dict,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment" / "resnet50_a1_in1k.yaml"


def test_summarize_teacher_cache_diagnostics_reports_oracle_and_single_aug(
    tmp_path: Path,
) -> None:
    cache_dir = _write_cache(tmp_path / "teacher_cache")

    summary = summarize_teacher_cache_diagnostics(
        cache_dir=cache_dir,
        split="public_val",
        aug_ids=["aug_000", "aug_001", "aug_002"],
        identity_aug_id="aug_000",
    )

    assert summary.split == "public_val"
    assert summary.image_count == 2
    assert summary.candidate_count == 3
    assert summary.clean_top1 == pytest.approx(1.0)
    assert summary.clean_top5 == pytest.approx(1.0)
    assert summary.helpful_fraction > 0.0
    assert summary.harmful_fraction > 0.0
    assert summary.oracle_best_mean_gain >= summary.best_single_aug_mean_gain
    assert summary.top_augmentations[0]["aug_id"] in {"aug_001", "aug_002"}


def test_summarize_teacher_cache_diagnostics_rejects_missing_shard(tmp_path: Path) -> None:
    cache_dir = _write_cache(tmp_path / "teacher_cache")

    with pytest.raises(ValueError, match="missing teacher cache metadata"):
        summarize_teacher_cache_diagnostics(
            cache_dir=cache_dir,
            split="public_val",
            aug_ids=["aug_000", "aug_999"],
            identity_aug_id="aug_000",
        )


def test_teacher_cache_diagnostics_to_dict_is_json_serializable(tmp_path: Path) -> None:
    cache_dir = _write_cache(tmp_path / "teacher_cache")

    summary = summarize_teacher_cache_diagnostics(
        cache_dir=cache_dir,
        split="public_val",
        aug_ids=["aug_000", "aug_001", "aug_002"],
        identity_aug_id="aug_000",
    )
    payload = teacher_cache_diagnostics_to_dict(summary)

    assert payload["split"] == "public_val"
    assert payload["candidate_count"] == 3
    assert len(payload["top_augmentations"]) == 2
    assert json.loads(json.dumps(payload)) == payload


def test_cli_teacher_cache_diagnostics_can_emit_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_dir = _write_cache(tmp_path / "teacher_cache")

    main(
        [
            "teacher-cache-diagnostics",
            "--config",
            str(CONFIG_PATH),
            "--cache-dir",
            str(cache_dir),
            "--split",
            "public_val",
            "--candidate-id",
            "aug_000",
            "--candidate-id",
            "aug_001",
            "--candidate-id",
            "aug_002",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["split"] == "public_val"
    assert payload["image_count"] == 2
    assert payload["candidate_count"] == 3


def _write_cache(cache_dir: Path) -> Path:
    image_ids = ["image-0", "image-1"]
    class_idxs = np.array([0, 1], dtype=np.int64)
    for aug_id, logits in {
        "aug_000": np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        "aug_001": np.array([[4.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        "aug_002": np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32),
    }.items():
        write_teacher_shard(
            cache_dir,
            TeacherShard(
                split="public_val",
                aug_id=aug_id,
                image_ids=image_ids,
                class_idxs=class_idxs,
                logits=logits,
            ),
        )
    return cache_dir
