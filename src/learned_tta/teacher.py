"""Teacher model loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import timm
import timm.data


@dataclass(frozen=True, slots=True)
class TeacherBundle:
    """Loaded teacher model plus its evaluation preprocessing."""

    model: Any
    data_config: dict[str, Any]
    preprocess: Any


def load_teacher(model_name: str, pretrained: bool = True) -> TeacherBundle:
    """Load a timm teacher model in eval mode with model-specific preprocessing."""

    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    return TeacherBundle(model=model, data_config=data_config, preprocess=preprocess)
