from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import tomli
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def readme_text() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.fixture
def implementation_status_text() -> str:
    return (ROOT / "docs" / "implementation-status.md").read_text(encoding="utf-8")


@pytest.fixture
def colab_runbook_text() -> str:
    return (ROOT / "docs" / "colab-run.md").read_text(encoding="utf-8")


@pytest.fixture
def colab_notebook() -> dict[str, Any]:
    notebook_path = ROOT / "notebooks" / "full_imagenet_run_colab.ipynb"
    with notebook_path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture
def ci_workflow() -> dict[str, Any]:
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    with workflow_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture
def pyproject() -> dict[str, Any]:
    pyproject_path = ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        loaded = tomli.load(handle)
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
    "colab_fragment",
    [
        "docs/colab-run.md",
        "notebooks/full_imagenet_run_colab.ipynb",
        "Google Colab",
        "full ImageNet run",
    ],
)
def test_readme_links_colab_full_run_entrypoint(
    readme_text: str,
    colab_fragment: str,
) -> None:
    assert colab_fragment in readme_text


@pytest.mark.parametrize(
    "runbook_fragment",
    [
        "Google Colab",
        "Do not paste API keys",
        "IMAGENET_VAL_DIR",
        "Google Drive",
        "resume-full-run",
        "cache resume",
        "artifacts",
        "reports",
    ],
)
def test_colab_runbook_documents_resumable_gpu_run(
    colab_runbook_text: str,
    runbook_fragment: str,
) -> None:
    assert runbook_fragment in colab_runbook_text


def test_colab_notebook_is_valid_and_uses_status_orchestration(
    colab_notebook: dict[str, Any],
) -> None:
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in colab_notebook.get("cells", [])
        if isinstance(cell, dict)
    )

    assert colab_notebook["nbformat"] == 4
    assert "drive.mount" in source
    assert "torch.cuda.is_available" in source
    assert "full-run-status" in source
    assert "resume-full-run" in source
    assert "--cache-log-dir" in source
    assert "IMAGENET_VAL_DIR" in source
    assert "API_KEY" not in source
    assert "api_key" not in source


def test_colab_notebook_prefers_local_prepared_imagenet(
    colab_notebook: dict[str, Any],
) -> None:
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in colab_notebook.get("cells", [])
        if isinstance(cell, dict)
    )

    assert "LOCAL_IMAGENET_VAL_DIR = Path('/content/imagenet_val_prepare/val')" in source
    assert "DRIVE_IMAGENET_VAL_DIR = Path('/content/drive/MyDrive/datasets/imagenet/val')" in source
    assert "IMAGENET_VAL_DIR = (" in source
    assert "if LOCAL_IMAGENET_VAL_DIR.exists()" in source
    assert "else DRIVE_IMAGENET_VAL_DIR" in source
    assert "jpeg_count == 50_000" in source


def test_colab_notebook_uses_colab_safe_teacher_cache_workers(
    colab_notebook: dict[str, Any],
) -> None:
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in colab_notebook.get("cells", [])
        if isinstance(cell, dict)
    )

    assert "resume-full-run" in source
    assert "'--cache-log-dir', DRIVE_RUN_ROOT / 'logs'" in source


@pytest.mark.parametrize(
    "target_fragment",
    [
        "augmentation utility predictor",
        "100 augmentation utility scores",
        "not a 50-bin loss classification task",
        "raw true-class NLL",
        "gain = clean_nll - aug_nll",
        "ranking/top-k TTA selection",
    ],
)
def test_readme_documents_selector_target_formulation(
    readme_text: str,
    target_fragment: str,
) -> None:
    assert target_fragment in readme_text


@pytest.mark.parametrize(
    "status_fragment",
    [
        "100-score augmentation utility predictor",
        "not 50-bin loss classification",
        "ranking/top-k TTA selection",
    ],
)
def test_implementation_status_documents_selector_target_choice(
    implementation_status_text: str,
    status_fragment: str,
) -> None:
    assert status_fragment in implementation_status_text


@pytest.mark.parametrize(
    "leakage_fragment",
    [
        "Split Contract",
        "API contract",
        "Selector targets are built only from `public_train` and `public_val`",
        "learned aggregation training use `public_val`",
        "`evaluate-private` accepts only `private`",
        "not deployable methods or tuning inputs",
    ],
)
def test_readme_documents_split_contract(
    readme_text: str,
    leakage_fragment: str,
) -> None:
    assert leakage_fragment in readme_text


def test_implementation_status_documents_split_role_guards(
    implementation_status_text: str,
) -> None:
    assert "Public/private split-role guards" in implementation_status_text
    assert "CI coverage gate fixed at 98% minimum" in implementation_status_text


@pytest.mark.parametrize(
    "runbook_fragment",
    [
        "run-smoke",
        "docs/implementation-status.md",
        "augmentation_registry_audit.json",
        "check-full-run",
        "full-run-status",
        "resume-full-run",
        "--format json",
        "--fail-on-incomplete",
        "--next-command",
        "optional XGBoost",
        "required/optional status",
        "every configured augmentation candidate",
        "missing=",
        "extra=",
        "missing_outputs",
        "extra_outputs",
        "serialized AlbumentationsX `Compose`",
        "runtime package versions",
        "teacher cache shard writes a `.run.json` sidecar",
        "model name, pretrained flag, timm data config",
        "full-run-status treats `.run.json` sidecars as required teacher cache outputs",
        "validates the sidecar metadata",
        "stale Drive shards",
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
        "private_metric_deltas.csv",
        "active_threshold",
        "augmentation_name",
        "transform_class",
        "transform_class_impact.csv",
        "transform_class_impact.svg",
        "transform_class_aggregation.csv",
        "transform_class_aggregation.svg",
        "top-N markdown tables",
        "learned aggregation weights",
        "XGBoost feature importance",
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


def test_ty_excludes_colab_notebooks_from_type_checking(pyproject: dict[str, Any]) -> None:
    assert pyproject["tool"]["ty"]["src"]["exclude"] == ["notebooks/**"]


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
        (
            "coverage",
            "uv run --frozen pytest --cov=learned_tta "
            "--cov-report=term-missing --cov-fail-under=98",
        ),
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
