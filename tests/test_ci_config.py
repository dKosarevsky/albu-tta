from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def readme_text() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture
def ci_workflow() -> dict[str, Any]:
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    with workflow_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.parametrize(
    "badge_fragment",
    [
        "actions/workflows/ci.yml/badge.svg",
        "python-3.10%2B",
        "lint-ruff",
        "types-ty",
        "license-MIT",
    ],
)
def test_readme_has_project_status_badges(readme_text: str, badge_fragment: str) -> None:
    assert badge_fragment in readme_text


def test_readme_does_not_claim_pypi_status(readme_text: str) -> None:
    assert "pypi" not in readme_text.lower()


def test_ci_workflow_has_separate_test_and_quality_jobs(ci_workflow: dict[str, Any]) -> None:
    jobs = ci_workflow["jobs"]

    assert set(jobs) == {"tests", "quality"}
    assert jobs["tests"]["name"] == "tests"
    assert jobs["quality"]["name"] == "lint-and-types"


@pytest.mark.parametrize(
    ("job_name", "expected_command"),
    [
        ("tests", "uv sync --frozen --extra dev"),
        ("tests", "uv run --frozen pytest -q"),
        ("quality", "uv sync --frozen --extra dev"),
        ("quality", "uv run --frozen ruff check ."),
        ("quality", "uv run --frozen ty check"),
    ],
)
def test_ci_workflow_runs_expected_commands(
    ci_workflow: dict[str, Any],
    job_name: str,
    expected_command: str,
) -> None:
    commands = [
        step["run"]
        for step in ci_workflow["jobs"][job_name]["steps"]
        if "run" in step
    ]

    assert expected_command in commands
