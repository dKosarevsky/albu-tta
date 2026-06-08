"""Teacher inference backend registry and planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

IMPLEMENTED_TEACHER_CACHE_BACKENDS = ("pytorch",)
PLANNED_TEACHER_CACHE_BACKENDS = ("tensorrt", "onnxruntime", "openvino")


@dataclass(frozen=True, slots=True)
class TeacherBackendSpec:
    """Documented teacher inference backend status."""

    name: str
    status: str
    device: str
    role: str
    notes: str


@dataclass(frozen=True, slots=True)
class TeacherBackendPlan:
    """Current and planned backend guidance for a teacher-cache run."""

    device: str
    active_backend: str
    recommended_accelerator: str
    backends: tuple[TeacherBackendSpec, ...]


def validate_teacher_cache_backend(backend: str) -> None:
    """Raise when a requested teacher-cache backend is not implemented yet."""

    if backend in IMPLEMENTED_TEACHER_CACHE_BACKENDS:
        return
    if backend in PLANNED_TEACHER_CACHE_BACKENDS:
        raise ValueError(
            f"backend {backend!r} is planned but not implemented; "
            "use backend='pytorch' for current cache-teacher runs"
        )
    allowed = ", ".join((*IMPLEMENTED_TEACHER_CACHE_BACKENDS, *PLANNED_TEACHER_CACHE_BACKENDS))
    raise ValueError(f"unknown teacher cache backend {backend!r}; expected one of {allowed}")


def build_teacher_backend_plan(device: str = "cuda") -> TeacherBackendPlan:
    """Return backend guidance without importing optional accelerator packages."""

    normalized_device = str(device).lower()
    recommended_accelerator = (
        "tensorrt"
        if normalized_device.startswith(("cuda", "gpu"))
        else "openvino"
        if normalized_device.startswith("cpu")
        else "onnxruntime"
    )
    return TeacherBackendPlan(
        device=device,
        active_backend="pytorch",
        recommended_accelerator=recommended_accelerator,
        backends=(
            TeacherBackendSpec(
                name="pytorch",
                status="implemented",
                device="cpu,cuda",
                role="default teacher-cache backend",
                notes="Uses timm preprocessing and torch inference; this is the correctness path.",
            ),
            TeacherBackendSpec(
                name="tensorrt",
                status="planned",
                device="cuda",
                role="future high-throughput GPU backend",
                notes="Requires explicit export/calibration work before it can replace PyTorch.",
            ),
            TeacherBackendSpec(
                name="onnxruntime",
                status="planned",
                device="cpu,cuda",
                role="future portable exported-model backend",
                notes="Useful for CPU smoke benchmarks and GPU providers without TensorRT.",
            ),
            TeacherBackendSpec(
                name="openvino",
                status="planned",
                device="cpu",
                role="future CPU inference backend",
                notes="Candidate backend for CPU-only throughput comparisons.",
            ),
        ),
    )


def teacher_backend_plan_to_dict(plan: TeacherBackendPlan) -> dict[str, Any]:
    """Return a JSON-serializable teacher backend plan."""

    return {
        "device": plan.device,
        "active_backend": plan.active_backend,
        "recommended_accelerator": plan.recommended_accelerator,
        "backends": [
            {
                "name": backend.name,
                "status": backend.status,
                "device": backend.device,
                "role": backend.role,
                "notes": backend.notes,
            }
            for backend in plan.backends
        ],
    }
