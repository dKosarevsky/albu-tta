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
        "github/check-runs/dKosarevsky/albu-tta/main?nameFilter=pytest",
        "github/check-runs/dKosarevsky/albu-tta/main?nameFilter=ruff",
        "github/check-runs/dKosarevsky/albu-tta/main?nameFilter=ty",
        "github/check-runs/dKosarevsky/albu-tta/main?nameFilter=coverage",
        "python-3.10%2B",
        "license-MIT",
    ],
)
def test_readme_has_project_status_badges(readme_text: str, badge_fragment: str) -> None:
    assert badge_fragment in readme_text


def test_readme_badges_do_not_duplicate_ci_workflow_status(readme_text: str) -> None:
    assert "actions/workflows/ci.yml/badge.svg?branch=main&job=" not in readme_text
    assert readme_text.count("github/check-runs/dKosarevsky/albu-tta/main") == 4
    assert "tests-pytest" not in readme_text
    assert "lint-ruff" not in readme_text
    assert "types-ty" not in readme_text
    assert "coverage-ci" not in readme_text


def test_readme_does_not_claim_pypi_status(readme_text: str) -> None:
    assert "pypi" not in readme_text.lower()


@pytest.mark.parametrize(
    "runbook_fragment",
    [
        "run-smoke",
        "augmentation_registry_audit.json",
        "--imagenet-val-dir",
        "cache-teacher --split public_train",
        "cache-teacher --split public_val",
        "train-aggregator --method global-nonnegative",
        "train-aggregator --method class-nonnegative",
        "train-aggregator --method xgboost-multiclass",
        "uv sync --extra stackers",
        "global_weighted_tta",
        "class_weighted_tta",
        "xgboost_multiclass",
        "xgboost_feature_importance.csv",
        "xgboost_feature_importance.svg",
        "active_threshold",
        "gain_distribution.svg",
        "oracle_overlap.svg",
        "cache-teacher --split private",
        "build-report",
    ],
)
def test_readme_documents_smoke_and_full_run_order(
    readme_text: str,
    runbook_fragment: str,
) -> None:
    assert runbook_fragment in readme_text


def test_ci_workflow_has_separate_pytest_ruff_ty_and_coverage_jobs(
    ci_workflow: dict[str, Any],
) -> None:
    jobs = ci_workflow["jobs"]

    assert set(jobs) == {"pytest", "ruff", "ty", "coverage"}
    assert jobs["pytest"]["name"] == "pytest"
    assert jobs["ruff"]["name"] == "ruff"
    assert jobs["ty"]["name"] == "ty"
    assert jobs["coverage"]["name"] == "coverage"


@pytest.mark.parametrize(
    ("job_name", "expected_command"),
    [
        ("pytest", "uv sync --frozen --extra dev"),
        ("pytest", "uv run --frozen pytest -q"),
        ("ruff", "uv sync --frozen --extra dev"),
        ("ruff", "uv run --frozen ruff check ."),
        ("ty", "uv sync --frozen --extra dev"),
        ("ty", "uv run --frozen ty check"),
        ("coverage", "uv sync --frozen --extra dev"),
        ("coverage", "uv run --frozen pytest --cov=learned_tta --cov-report=term-missing"),
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
