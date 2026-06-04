from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from learned_tta.cache import (
    TeacherShard,
    _default_run_metadata_path,
    _json_ready,
    read_teacher_shard,
    shard_is_complete,
    write_teacher_shard,
)


@pytest.fixture
def logits() -> np.ndarray:
    return np.array(
        [
            [3.0, 1.0, -1.0],
            [0.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_write_and_read_teacher_shard(tmp_path: Path, logits: np.ndarray) -> None:
    shard = TeacherShard(
        split="public",
        aug_id="aug_001",
        image_ids=["image-a", "image-b"],
        class_idxs=np.array([0, 2], dtype=np.int64),
        logits=logits,
        run_metadata={
            "seed": 20260522,
            "teacher": {"model_name": "resnet50.a1_in1k"},
        },
    )

    paths = write_teacher_shard(tmp_path, shard)
    loaded = read_teacher_shard(paths.metadata_path, paths.logits_path)

    assert paths.metadata_path.name == "public__aug_001.parquet"
    assert paths.logits_path.name == "public__aug_001.logits.npy"
    assert paths.run_metadata_path.name == "public__aug_001.run.json"
    assert loaded.logits.dtype == np.float16
    assert loaded.run_metadata == {
        "seed": 20260522,
        "teacher": {"model_name": "resnet50.a1_in1k"},
    }
    np.testing.assert_allclose(loaded.logits.astype(np.float32), logits, rtol=1e-3)
    assert loaded.metadata["image_id"].to_list() == ["image-a", "image-b"]
    assert loaded.metadata["class_idx"].to_list() == [0, 2]
    assert loaded.metadata["split"].to_list() == ["public", "public"]
    assert loaded.metadata["aug_id"].to_list() == ["aug_001", "aug_001"]
    assert loaded.metadata["is_top1"].to_list() == [True, False]
    assert loaded.metadata["is_top5"].to_list() == [True, True]
    assert loaded.metadata["nll_true"].dtype == "float32"
    assert loaded.metadata["prob_true"].dtype == "float32"


@pytest.mark.parametrize(
    ("expected_rows", "expected_classes", "complete"),
    [
        (2, 3, True),
        (3, 3, False),
        (2, 4, False),
    ],
)
def test_shard_is_complete_validates_shape(
    tmp_path: Path,
    logits: np.ndarray,
    expected_rows: int,
    expected_classes: int,
    complete: bool,
) -> None:
    run_metadata = {
        "seed": 20260522,
        "teacher": {"model_name": "resnet50.a1_in1k"},
    }
    shard = TeacherShard(
        split="public",
        aug_id="aug_001",
        image_ids=["image-a", "image-b"],
        class_idxs=np.array([0, 2], dtype=np.int64),
        logits=logits,
        run_metadata=run_metadata,
    )
    paths = write_teacher_shard(tmp_path, shard)

    assert (
        shard_is_complete(
            metadata_path=paths.metadata_path,
            logits_path=paths.logits_path,
            run_metadata_path=paths.run_metadata_path,
            expected_rows=expected_rows,
            expected_classes=expected_classes,
            expected_run_metadata=run_metadata,
        )
        is complete
    )


def test_shard_is_complete_rejects_run_metadata_mismatch(
    tmp_path: Path,
    logits: np.ndarray,
) -> None:
    shard = TeacherShard(
        split="public",
        aug_id="aug_001",
        image_ids=["image-a", "image-b"],
        class_idxs=np.array([0, 2], dtype=np.int64),
        logits=logits,
        run_metadata={
            "seed": 20260522,
            "teacher": {"model_name": "resnet50.a1_in1k"},
        },
    )
    paths = write_teacher_shard(tmp_path, shard)

    assert not shard_is_complete(
        metadata_path=paths.metadata_path,
        logits_path=paths.logits_path,
        run_metadata_path=paths.run_metadata_path,
        expected_rows=2,
        expected_classes=3,
        expected_run_metadata={
            "seed": 20260523,
            "teacher": {"model_name": "resnet50.a1_in1k"},
        },
    )


def test_shard_is_complete_rejects_missing_or_mismatched_files(tmp_path: Path) -> None:
    metadata_path = tmp_path / "missing.parquet"
    logits_path = tmp_path / "missing.npy"

    assert not shard_is_complete(metadata_path, logits_path, expected_rows=2, expected_classes=3)

    pd.DataFrame({"image_id": ["image-a"]}).to_parquet(metadata_path)
    np.save(logits_path, np.zeros((2, 3), dtype=np.float16))

    assert not shard_is_complete(metadata_path, logits_path, expected_rows=2, expected_classes=3)


@pytest.mark.parametrize(
    ("metadata_bytes", "logits_bytes"),
    [
        (b"not parquet", b"\x93NUMPY"),
        (b"PAR1", b"not numpy"),
    ],
)
def test_shard_is_complete_rejects_unreadable_files(
    tmp_path: Path,
    metadata_bytes: bytes,
    logits_bytes: bytes,
) -> None:
    metadata_path = tmp_path / "public__aug_001.parquet"
    logits_path = tmp_path / "public__aug_001.logits.npy"
    metadata_path.write_bytes(metadata_bytes)
    logits_path.write_bytes(logits_bytes)

    assert not shard_is_complete(
        metadata_path,
        logits_path,
        expected_rows=2,
        expected_classes=3,
    )


@pytest.mark.parametrize(
    ("logits_value", "class_idxs", "match"),
    [
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([0], dtype=np.int64),
            "logits must have shape",
        ),
        (
            np.zeros((2, 3), dtype=np.float32),
            np.array([0], dtype=np.int64),
            "row counts must match",
        ),
    ],
)
def test_write_teacher_shard_rejects_invalid_shapes(
    tmp_path: Path,
    logits_value: np.ndarray,
    class_idxs: np.ndarray,
    match: str,
) -> None:
    shard = TeacherShard(
        split="public",
        aug_id="aug_001",
        image_ids=["image-a"],
        class_idxs=class_idxs,
        logits=logits_value,
    )

    with pytest.raises(ValueError, match=match):
        write_teacher_shard(tmp_path, shard)


def test_shard_is_complete_can_skip_run_metadata_validation(
    tmp_path: Path,
    logits: np.ndarray,
) -> None:
    shard = TeacherShard(
        split="public",
        aug_id="aug_001",
        image_ids=["image-a", "image-b"],
        class_idxs=np.array([0, 2], dtype=np.int64),
        logits=logits,
    )
    paths = write_teacher_shard(tmp_path, shard)

    assert shard_is_complete(
        metadata_path=paths.metadata_path,
        logits_path=paths.logits_path,
        expected_rows=2,
        expected_classes=3,
    )


@pytest.mark.parametrize("metadata_text", ["", "[]"])
def test_shard_is_complete_rejects_invalid_run_metadata(
    tmp_path: Path,
    logits: np.ndarray,
    metadata_text: str,
) -> None:
    shard = TeacherShard(
        split="public",
        aug_id="aug_001",
        image_ids=["image-a", "image-b"],
        class_idxs=np.array([0, 2], dtype=np.int64),
        logits=logits,
        run_metadata={"seed": 1},
    )
    paths = write_teacher_shard(tmp_path, shard)
    paths.run_metadata_path.write_text(metadata_text, encoding="utf-8")

    assert not shard_is_complete(
        metadata_path=paths.metadata_path,
        logits_path=paths.logits_path,
        run_metadata_path=paths.run_metadata_path,
        expected_rows=2,
        expected_classes=3,
        expected_run_metadata={"seed": 1},
    )


def test_shard_is_complete_rejects_missing_run_metadata(
    tmp_path: Path,
    logits: np.ndarray,
) -> None:
    shard = TeacherShard(
        split="public",
        aug_id="aug_001",
        image_ids=["image-a", "image-b"],
        class_idxs=np.array([0, 2], dtype=np.int64),
        logits=logits,
        run_metadata={"seed": 1},
    )
    paths = write_teacher_shard(tmp_path, shard)
    paths.run_metadata_path.unlink()

    assert not shard_is_complete(
        metadata_path=paths.metadata_path,
        logits_path=paths.logits_path,
        expected_rows=2,
        expected_classes=3,
        expected_run_metadata={"seed": 1},
    )


def test_default_run_metadata_path_supports_non_parquet_paths() -> None:
    assert _default_run_metadata_path(Path("cache/public__aug_000.csv")) == Path(
        "cache/public__aug_000.run.json"
    )


def test_json_ready_stringifies_unknown_values() -> None:
    assert _json_ready({"path": Path("artifact")}) == {"path": "PosixPath('artifact')"}
