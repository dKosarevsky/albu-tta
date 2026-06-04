from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from learned_tta.run_status import FullRunStatusSummary, FullRunStepStatus
from learned_tta.run_supervisor import (
    _cache_log_path,
    _cache_split,
    find_active_cache_teacher_processes,
    prepare_next_command,
    run_next_full_run_step,
)


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


def test_run_next_full_run_step_can_start_duplicate_cache_when_allowed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    summary = _summary_with_next_step(
        command=(
            "uv run python -m learned_tta.cli cache-teacher --split public_train "
            "--config cfg.yaml --device cuda --num-workers 2"
        ),
        name="cache_public_train",
    )
    popen_calls: list[list[str]] = []

    class FakePopen:
        pid = 45678

        def __init__(self, args: list[str], **_kwargs: Any) -> None:
            popen_calls.append(args)

    monkeypatch.setattr(
        "learned_tta.run_supervisor.inspect_full_run_status",
        lambda _config_path: summary,
    )
    monkeypatch.setattr(
        "learned_tta.run_supervisor.find_active_cache_teacher_processes",
        lambda split=None: ("123 python -m learned_tta.cli cache-teacher --split public_train",),
    )
    monkeypatch.setattr("learned_tta.run_supervisor.subprocess.Popen", FakePopen)

    result = run_next_full_run_step(
        config_path=tmp_path / "experiment.yaml",
        cache_log_dir=tmp_path / "logs",
        allow_duplicate_cache=True,
    )

    assert result.status == "started"
    assert result.pid == 45678
    assert popen_calls[0][:5] == ["uv", "run", "python", "-m", "learned_tta.cli"]


def test_run_next_full_run_step_can_report_dry_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    summary = _summary_with_next_step(
        command="uv run python -m learned_tta.cli validate-augmentations --config cfg.yaml",
        name="validate_augmentations",
    )
    monkeypatch.setattr(
        "learned_tta.run_supervisor.inspect_full_run_status",
        lambda _config_path: summary,
    )

    result = run_next_full_run_step(config_path=tmp_path / "experiment.yaml", dry_run=True)

    assert result.status == "dry-run"
    assert result.step_name == "validate_augmentations"
    assert summary.next_step is not None
    assert result.command == summary.next_step.command


def test_run_next_full_run_step_can_run_cache_in_foreground(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    summary = _summary_with_next_step(
        command=(
            "uv run python -m learned_tta.cli cache-teacher --split public_train "
            "--config cfg.yaml --device cuda --num-workers 2"
        ),
        name="cache_public_train",
    )
    run_calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        run_calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(
        "learned_tta.run_supervisor.inspect_full_run_status",
        lambda _config_path: summary,
    )
    monkeypatch.setattr(
        "learned_tta.run_supervisor.find_active_cache_teacher_processes",
        lambda split=None: (),
    )
    monkeypatch.setattr("learned_tta.run_supervisor.subprocess.run", fake_run)

    result = run_next_full_run_step(
        config_path=tmp_path / "experiment.yaml",
        background_cache=False,
    )

    assert result.status == "completed"
    assert run_calls[0][:5] == ["uv", "run", "python", "-m", "learned_tta.cli"]


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


def test_prepare_next_command_requires_imagenet_dir_for_placeholder() -> None:
    with pytest.raises(ValueError, match="next command needs --imagenet-val-dir"):
        prepare_next_command(
            "uv run python -m learned_tta.cli make-splits --imagenet-val-dir /path/to/imagenet/val",
            imagenet_val_dir=None,
        )


def test_prepare_next_command_leaves_commands_without_placeholder_unchanged() -> None:
    command = "uv run python -m learned_tta.cli validate-augmentations --config cfg.yaml"

    assert prepare_next_command(command, imagenet_val_dir=None) == command


def test_find_active_cache_teacher_processes_handles_missing_pgrep(monkeypatch: Any) -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    monkeypatch.setattr("learned_tta.run_supervisor.subprocess.run", fake_run)

    assert find_active_cache_teacher_processes() == ()


def test_find_active_cache_teacher_processes_filters_split_and_ignores_pgrep_errors(
    monkeypatch: Any,
) -> None:
    def fake_run_error(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["pgrep"], returncode=2, stdout="")

    monkeypatch.setattr("learned_tta.run_supervisor.subprocess.run", fake_run_error)
    assert find_active_cache_teacher_processes() == ()

    def fake_run_ok(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["pgrep"],
            returncode=0,
            stdout=(
                "111 python -m learned_tta.cli cache-teacher --split public_train\n"
                "222 python -m learned_tta.cli cache-teacher --split private\n"
                "333 unrelated\n"
            ),
        )

    monkeypatch.setattr("learned_tta.run_supervisor.subprocess.run", fake_run_ok)

    assert find_active_cache_teacher_processes(split="private") == (
        "222 python -m learned_tta.cli cache-teacher --split private",
    )
    assert len(find_active_cache_teacher_processes()) == 2


def test_cache_split_returns_none_for_missing_or_incomplete_split_arg() -> None:
    assert _cache_split("uv run python -m learned_tta.cli cache-teacher") is None
    assert _cache_split("uv run python -m learned_tta.cli cache-teacher --split") is None


def test_cache_log_path_defaults_to_project_artifacts_logs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    config_path = project_root / "configs" / "experiment.yaml"
    config_path.parent.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    assert _cache_log_path(
        cache_log_dir=None,
        step_name="cache_public_train",
        config_path=config_path,
    ) == project_root / "artifacts" / "logs" / "cache_public_train.log"


def test_cache_log_path_falls_back_to_config_parent_outside_project(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "experiment.yaml"
    config_path.parent.mkdir(parents=True)

    assert _cache_log_path(
        cache_log_dir=None,
        step_name="cache_private",
        config_path=config_path,
    ) == config_path.parent / "artifacts" / "logs" / "cache_private.log"


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
