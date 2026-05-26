from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from learned_tta.run_status import FullRunStatusSummary, FullRunStepStatus
from learned_tta.run_supervisor import run_next_full_run_step


def test_run_next_full_run_step_starts_cache_step_in_background(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    summary = _summary_with_next_step(
        command=(
            "uv run python -m learned_tta.cli cache-teacher --split public_train "
            "--config /content/albu-tta/configs/experiment/resnet50_a1_in1k.yaml "
            "--device cuda --num-workers 2"
        ),
        name="cache_public_train",
    )
    popen_calls: list[dict[str, Any]] = []

    class FakePopen:
        pid = 12345

        def __init__(self, args: list[str], **kwargs: Any) -> None:
            popen_calls.append({"args": args, **kwargs})

    monkeypatch.setattr(
        "learned_tta.run_supervisor.inspect_full_run_status",
        lambda _config_path: summary,
    )
    monkeypatch.setattr(
        "learned_tta.run_supervisor.find_active_cache_teacher_processes",
        lambda split=None: (),
    )
    monkeypatch.setattr("learned_tta.run_supervisor.subprocess.Popen", FakePopen)

    result = run_next_full_run_step(
        config_path=tmp_path / "experiment.yaml",
        cache_log_dir=tmp_path / "logs",
    )

    assert result.status == "started"
    assert result.step_name == "cache_public_train"
    assert result.pid == 12345
    assert result.log_path == tmp_path / "logs" / "cache_public_train.log"
    assert popen_calls[0]["args"][:5] == ["uv", "run", "python", "-m", "learned_tta.cli"]
    assert popen_calls[0]["start_new_session"] is True
    assert result.log_path is not None
    assert result.log_path.exists()


def test_run_next_full_run_step_does_not_duplicate_active_cache_process(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    summary = _summary_with_next_step(
        command=(
            "uv run python -m learned_tta.cli cache-teacher --split public_train "
            "--config /content/albu-tta/configs/experiment/resnet50_a1_in1k.yaml "
            "--device cuda --num-workers 2"
        ),
        name="cache_public_train",
    )
    monkeypatch.setattr(
        "learned_tta.run_supervisor.inspect_full_run_status",
        lambda _config_path: summary,
    )
    monkeypatch.setattr(
        "learned_tta.run_supervisor.find_active_cache_teacher_processes",
        lambda split=None: ("123 python -m learned_tta.cli cache-teacher --split public_train",),
    )

    result = run_next_full_run_step(
        config_path=tmp_path / "experiment.yaml",
        cache_log_dir=tmp_path / "logs",
    )

    assert result.status == "active"
    assert result.pid is None
    assert "123 python" in result.active_processes[0]


def test_run_next_full_run_step_replaces_imagenet_placeholder_and_runs_foreground(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    summary = _summary_with_next_step(
        command=(
            "uv run python -m learned_tta.cli make-splits "
            "--config /content/albu-tta/configs/experiment/resnet50_a1_in1k.yaml "
            "--imagenet-val-dir /path/to/imagenet/val"
        ),
        name="make_splits",
    )
    run_calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        run_calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(
        "learned_tta.run_supervisor.inspect_full_run_status",
        lambda _config_path: summary,
    )
    monkeypatch.setattr("learned_tta.run_supervisor.subprocess.run", fake_run)

    result = run_next_full_run_step(
        config_path=tmp_path / "experiment.yaml",
        imagenet_val_dir=Path("/content/imagenet_val_prepare/val"),
    )

    assert result.status == "completed"
    assert run_calls == [
        [
            "uv",
            "run",
            "python",
            "-m",
            "learned_tta.cli",
            "make-splits",
            "--config",
            "/content/albu-tta/configs/experiment/resnet50_a1_in1k.yaml",
            "--imagenet-val-dir",
            "/content/imagenet_val_prepare/val",
        ]
    ]


def test_run_next_full_run_step_reports_complete_when_no_required_steps_remain(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    summary = FullRunStatusSummary(
        config_path=tmp_path / "experiment.yaml",
        steps=(),
        completed_steps=13,
        total_steps=13,
        completed_required_steps=12,
        total_required_steps=12,
        next_step=None,
    )
    monkeypatch.setattr(
        "learned_tta.run_supervisor.inspect_full_run_status",
        lambda _config_path: summary,
    )

    result = run_next_full_run_step(config_path=tmp_path / "experiment.yaml")

    assert result.status == "complete"
    assert result.command is None


def _summary_with_next_step(command: str, name: str) -> FullRunStatusSummary:
    step = FullRunStepStatus(
        name=name,
        complete=False,
        outputs=(),
        command=command,
        required=True,
        missing_outputs=(),
        extra_outputs=(),
    )
    return FullRunStatusSummary(
        config_path=Path("/content/albu-tta/configs/experiment/resnet50_a1_in1k.yaml"),
        steps=(step,),
        completed_steps=0,
        total_steps=1,
        completed_required_steps=0,
        total_required_steps=1,
        next_step=step,
    )
