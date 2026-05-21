from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_project_bootstrap_files_and_config() -> None:
    pyproject = ROOT / "pyproject.toml"
    experiment_config = ROOT / "configs" / "experiment" / "resnet50_a1_in1k.yaml"
    artifacts_readme = ROOT / "artifacts" / "README.md"

    assert pyproject.exists()
    assert experiment_config.exists()
    assert artifacts_readme.exists()

    config_text = experiment_config.read_text(encoding="utf-8")
    assert "model_name: resnet50.a1_in1k" in config_text
    assert "seed: 20260522" in config_text
    assert "candidate_count: 100" in config_text
    assert "- 1" in config_text
    assert "- 16" in config_text

    artifacts_text = artifacts_readme.read_text(encoding="utf-8")
    assert "ImageNet" in artifacts_text
    assert "logits" in artifacts_text
    assert "checkpoints" in artifacts_text


def test_package_imports() -> None:
    sys.path.insert(0, str(SRC))

    import learned_tta

    assert learned_tta.__version__ == "0.1.0"
