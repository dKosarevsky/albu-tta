from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from learned_tta.cache import TeacherShard, write_teacher_shard
from learned_tta.clean_baseline import check_clean_baseline


def test_check_clean_baseline_writes_metrics_artifact(tmp_path: Path) -> None:
    cache_dir = tmp_path / "teacher_cache"
    output_path = tmp_path / "public_val_clean_baseline.json"
    _write_identity_shard(cache_dir)

    report = check_clean_baseline(
        cache_dir=cache_dir,
        split="public_val",
        identity_aug_id="aug_000",
        min_top1=0.75,
        min_top5=1.0,
        max_nll=1.0,
        output_path=output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert report.passed is True
    assert report.metrics["image_count"] == 4.0
    assert report.metrics["top1"] == pytest.approx(0.75)
    assert report.metrics["top5"] == pytest.approx(1.0)
    assert payload["split"] == "public_val"
    assert payload["identity_aug_id"] == "aug_000"
    assert payload["passed"] is True
    assert payload["thresholds"] == {
        "min_top1": 0.75,
        "min_top5": 1.0,
        "max_nll": 1.0,
    }


def test_check_clean_baseline_rejects_low_top1_without_writing_artifact(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "teacher_cache"
    output_path = tmp_path / "public_val_clean_baseline.json"
    _write_identity_shard(cache_dir)

    with pytest.raises(ValueError, match="clean baseline top1"):
        check_clean_baseline(
            cache_dir=cache_dir,
            split="public_val",
            identity_aug_id="aug_000",
            min_top1=0.80,
            min_top5=1.0,
            max_nll=1.0,
            output_path=output_path,
        )

    assert not output_path.exists()


def _write_identity_shard(cache_dir: Path) -> None:
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="public_val",
            aug_id="aug_000",
            image_ids=["img_0", "img_1", "img_2", "img_3"],
            class_idxs=np.array([0, 1, 2, 1], dtype=np.int64),
            logits=np.array(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [2.0, 0.0, 3.0],
                    [2.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
        ),
    )
