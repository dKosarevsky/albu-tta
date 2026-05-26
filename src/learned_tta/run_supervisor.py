"""Resumable full-run supervisor helpers."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from learned_tta.run_status import inspect_full_run_status

IMAGENET_PLACEHOLDER = "/path/to/imagenet/val"


@dataclass(frozen=True, slots=True)
class FullRunStepRunResult:
    """Result of running or supervising one full-run step."""

    status: str
    step_name: str | None
    command: str | None
    log_path: Path | None = None
    pid: int | None = None
    active_processes: tuple[str, ...] = ()


def run_next_full_run_step(
    config_path: Path,
    *,
    imagenet_val_dir: Path | None = None,
    cache_log_dir: Path | None = None,
    dry_run: bool = False,
    background_cache: bool = True,
    allow_duplicate_cache: bool = False,
) -> FullRunStepRunResult:
    """Run the next missing required full-run step safely.

    Teacher-cache steps are resumable and long-running, so they start in the
    background by default and write logs to a persistent directory. Other steps
    run in the foreground and must finish before the next step is requested.
    """

    summary = inspect_full_run_status(config_path)
    next_step = summary.next_step
    if next_step is None:
        return FullRunStepRunResult(
            status="complete",
            step_name=None,
            command=None,
        )

    command = prepare_next_command(next_step.command, imagenet_val_dir=imagenet_val_dir)
    if dry_run:
        return FullRunStepRunResult(
            status="dry-run",
            step_name=next_step.name,
            command=command,
        )

    if is_cache_teacher_command(command):
        split = _cache_split(command)
        active_processes = find_active_cache_teacher_processes(split=split)
        if active_processes and not allow_duplicate_cache:
            return FullRunStepRunResult(
                status="active",
                step_name=next_step.name,
                command=command,
                active_processes=active_processes,
            )

        if background_cache:
            log_path = _cache_log_path(
                cache_log_dir=cache_log_dir,
                step_name=next_step.name,
                config_path=config_path,
            )
            pid = start_background_command(command, log_path=log_path)
            return FullRunStepRunResult(
                status="started",
                step_name=next_step.name,
                command=command,
                log_path=log_path,
                pid=pid,
            )

    subprocess.run(shlex.split(command), check=True)
    return FullRunStepRunResult(
        status="completed",
        step_name=next_step.name,
        command=command,
    )


def prepare_next_command(command: str, *, imagenet_val_dir: Path | None) -> str:
    """Return a runnable command with runtime-specific placeholders filled."""

    if IMAGENET_PLACEHOLDER not in command:
        return command
    if imagenet_val_dir is None:
        raise ValueError(
            "next command needs --imagenet-val-dir; pass imagenet_val_dir to resume it"
        )
    return command.replace(IMAGENET_PLACEHOLDER, shlex.quote(str(imagenet_val_dir)))


def is_cache_teacher_command(command: str) -> bool:
    """Return whether a command launches teacher-cache inference."""

    args = shlex.split(command)
    return "cache-teacher" in args


def find_active_cache_teacher_processes(split: str | None = None) -> tuple[str, ...]:
    """Return active cache-teacher process lines if `pgrep` is available."""

    try:
        completed = subprocess.run(
            ["pgrep", "-af", "learned_tta.cli cache-teacher"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ()

    if completed.returncode not in (0, 1):
        return ()

    lines = tuple(
        line
        for line in completed.stdout.splitlines()
        if "learned_tta.cli cache-teacher" in line
        and (split is None or f"--split {split}" in line)
    )
    return lines


def start_background_command(command: str, *, log_path: Path) -> int:
    """Start a command detached from the current shell and append output to log."""

    args = shlex.split(command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _append_log_preamble(log_path, command)
    log_handle = log_path.open("ab")
    try:
        process = subprocess.Popen(
            args,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    return int(process.pid)


def _cache_split(command: str) -> str | None:
    args = shlex.split(command)
    try:
        split_idx = args.index("--split")
    except ValueError:
        return None
    if split_idx + 1 >= len(args):
        return None
    return args[split_idx + 1]


def _cache_log_path(
    *,
    cache_log_dir: Path | None,
    step_name: str,
    config_path: Path,
) -> Path:
    log_dir = cache_log_dir if cache_log_dir is not None else _default_log_dir(config_path)
    return log_dir / f"{step_name}.log"


def _default_log_dir(config_path: Path) -> Path:
    project_root = _find_project_root(config_path)
    return project_root / "artifacts" / "logs"


def _find_project_root(config_path: Path) -> Path:
    path = Path(config_path).resolve()
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return path.parent


def _append_log_preamble(log_path: Path, command: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with log_path.open("ab") as handle:
        handle.write(f"\n[{timestamp}] starting: {command}\n".encode())
