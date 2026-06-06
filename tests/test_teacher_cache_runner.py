from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from learned_tta.augmentations import AugmentationCandidate
from learned_tta.cache import read_teacher_shard, teacher_shard_paths
from learned_tta.data import ManifestRecord
from learned_tta.teacher import TeacherBundle
from learned_tta.teacher_cache import _filter_candidates, _model_num_classes, run_teacher_cache


@pytest.fixture
def manifest_records(tmp_path: Path) -> list[ManifestRecord]:
    records = []
    for index, class_idx in enumerate([0, 1]):
        path = tmp_path / f"image-{index}.png"
        image = np.full((8, 8, 3), fill_value=40 + index, dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(path)
        records.append(
            ManifestRecord(
                split="public",
                image_id=f"image-{index}",
                class_idx=class_idx,
                class_name=f"class-{class_idx}",
                path=path,
            )
        )
    return records


@pytest.fixture
def candidates() -> list[AugmentationCandidate]:
    return [
        AugmentationCandidate(id="aug_000", name="identity", class_name=None),
        AugmentationCandidate(
            id="aug_001",
            name="horizontal_flip",
            class_name="HorizontalFlip",
            params={"p": 1.0},
        ),
    ]


def test_run_teacher_cache_writes_and_resumes_complete_shards(
    tmp_path: Path,
    manifest_records: list[ManifestRecord],
    candidates: list[AugmentationCandidate],
) -> None:
    output_dir = tmp_path / "cache"
    teacher = _fake_teacher_bundle()

    first = run_teacher_cache(
        split="public",
        records=manifest_records,
        candidates=candidates,
        teacher=teacher,
        output_dir=output_dir,
        seed=20260522,
        batch_size=2,
        num_workers=0,
        resume=True,
    )
    second = run_teacher_cache(
        split="public",
        records=manifest_records,
        candidates=candidates,
        teacher=teacher,
        output_dir=output_dir,
        seed=20260522,
        batch_size=2,
        num_workers=0,
        resume=True,
    )

    assert first.written == ["aug_000", "aug_001"]
    assert first.skipped == []
    assert second.written == []
    assert second.skipped == ["aug_000", "aug_001"]

    paths = teacher_shard_paths(output_dir, split="public", aug_id="aug_000")
    loaded = read_teacher_shard(paths.metadata_path, paths.logits_path)
    assert loaded.logits.shape == (2, 3)
    assert loaded.metadata["image_id"].tolist() == ["image-0", "image-1"]
    assert loaded.metadata["class_idx"].tolist() == [0, 1]
    assert loaded.run_metadata["version"] == 1
    assert loaded.run_metadata["seed"] == 20260522
    assert loaded.run_metadata["split"] == "public"
    assert loaded.run_metadata["aug_id"] == "aug_000"
    assert loaded.run_metadata["augmentation"]["name"] == "identity"
    assert loaded.run_metadata["teacher"]["model_name"] == "fake_resnet"
    assert loaded.run_metadata["teacher"]["pretrained"] is False
    assert loaded.run_metadata["teacher"]["data_config"]["input_size"] == [3, 8, 8]
    benchmark = json.loads(paths.benchmark_path.read_text(encoding="utf-8"))
    assert benchmark["version"] == 1
    assert benchmark["split"] == "public"
    assert benchmark["aug_id"] == "aug_000"
    assert benchmark["backend"] == "pytorch"
    assert benchmark["device"] == "cpu"
    assert benchmark["batch_size"] == 2
    assert benchmark["num_workers"] == 0
    assert benchmark["image_count"] == 2
    assert benchmark["elapsed_seconds"] >= 0.0
    assert benchmark["images_per_second"] > 0.0


def test_run_teacher_cache_resume_recomputes_when_run_metadata_changes(
    tmp_path: Path,
    manifest_records: list[ManifestRecord],
    candidates: list[AugmentationCandidate],
) -> None:
    output_dir = tmp_path / "cache"
    teacher = _fake_teacher_bundle()

    first = run_teacher_cache(
        split="public",
        records=manifest_records,
        candidates=[candidates[0]],
        teacher=teacher,
        output_dir=output_dir,
        seed=20260522,
        batch_size=2,
        num_workers=0,
        resume=True,
    )
    second = run_teacher_cache(
        split="public",
        records=manifest_records,
        candidates=[candidates[0]],
        teacher=teacher,
        output_dir=output_dir,
        seed=20260523,
        batch_size=2,
        num_workers=0,
        resume=True,
    )

    assert first.written == ["aug_000"]
    assert second.written == ["aug_000"]
    assert second.skipped == []


def test_cache_teacher_cli_uses_manifest_and_candidate_filter(
    tmp_path: Path,
    manifest_records: list[ManifestRecord],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    manifest_path = tmp_path / "public.csv"
    pd.DataFrame(
        [
            {
                "split": record.split,
                "image_id": record.image_id,
                "class_idx": record.class_idx,
                "class_name": record.class_name,
                "path": str(record.path),
            }
            for record in manifest_records
        ]
    ).to_csv(manifest_path, index=False)
    output_dir = tmp_path / "cache"
    monkeypatch.setattr(
        "learned_tta.teacher_cache.load_teacher",
        lambda model_name, pretrained: _fake_teacher_bundle(),
    )

    main(
        [
            "cache-teacher",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/experiment/resnet50_a1_in1k.yaml"),
            "--split",
            "public",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--candidate-id",
            "aug_000",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
        ]
    )
    captured = capsys.readouterr()

    assert "teacher cache public: wrote 1 shard, skipped 0 shards" in captured.out
    assert teacher_shard_paths(output_dir, "public", "aug_000").metadata_path.exists()
    assert not teacher_shard_paths(output_dir, "public", "aug_001").metadata_path.exists()


def test_filter_candidates_rejects_unknown_candidate_ids(
    candidates: list[AugmentationCandidate],
) -> None:
    with pytest.raises(ValueError, match="unknown augmentation candidate ids: aug_missing"):
        _filter_candidates(candidates, ["aug_000", "aug_missing"])


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (object(), None),
        (SimpleNamespace(num_classes="1000"), None),
        (SimpleNamespace(num_classes=1000), 1000),
    ],
)
def test_model_num_classes_returns_only_integer_values(
    model: object,
    expected: int | None,
) -> None:
    assert _model_num_classes(model) == expected


class _FakeTeacher(torch.nn.Module):
    num_classes = 3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        return torch.tensor(
            [[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]][:batch_size],
            dtype=torch.float32,
        )


def _fake_teacher_bundle() -> TeacherBundle:
    def preprocess(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    return TeacherBundle(
        model=_FakeTeacher(),
        data_config={"input_size": (3, 8, 8)},
        preprocess=preprocess,
        model_name="fake_resnet",
        pretrained=False,
    )
