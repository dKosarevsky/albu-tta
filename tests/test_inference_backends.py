from __future__ import annotations

import json

import pytest

from learned_tta.cli import main
from learned_tta.inference_backends import (
    IMPLEMENTED_TEACHER_CACHE_BACKENDS,
    build_teacher_backend_plan,
    teacher_backend_plan_to_dict,
    validate_teacher_cache_backend,
)


def test_validate_teacher_cache_backend_accepts_only_implemented_backend() -> None:
    assert IMPLEMENTED_TEACHER_CACHE_BACKENDS == ("pytorch",)
    validate_teacher_cache_backend("pytorch")

    with pytest.raises(ValueError, match="backend 'tensorrt' is planned but not implemented"):
        validate_teacher_cache_backend("tensorrt")

    with pytest.raises(ValueError, match="unknown teacher cache backend 'coreml'"):
        validate_teacher_cache_backend("coreml")


@pytest.mark.parametrize(
    ("device", "accelerator"),
    [
        ("cuda", "tensorrt"),
        ("cpu", "openvino"),
    ],
)
def test_build_teacher_backend_plan_keeps_pytorch_as_active_backend(
    device: str,
    accelerator: str,
) -> None:
    plan = build_teacher_backend_plan(device=device)
    payload = teacher_backend_plan_to_dict(plan)

    assert plan.active_backend == "pytorch"
    assert plan.recommended_accelerator == accelerator
    assert payload["active_backend"] == "pytorch"
    assert payload["recommended_accelerator"] == accelerator
    assert payload["backends"][0]["name"] == "pytorch"
    assert payload["backends"][0]["status"] == "implemented"


def test_cli_teacher_backend_plan_can_emit_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(["teacher-backend-plan", "--device", "cuda", "--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["active_backend"] == "pytorch"
    assert payload["recommended_accelerator"] == "tensorrt"
    assert any(backend["name"] == "tensorrt" for backend in payload["backends"])
