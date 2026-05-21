from __future__ import annotations

import torch

from learned_tta.selector_model import SelectorCNN, count_trainable_parameters


def test_selector_cnn_outputs_one_score_per_augmentation() -> None:
    model = SelectorCNN(output_dim=100)
    inputs = torch.randn(2, 3, 224, 224)

    outputs = model(inputs)

    assert outputs.shape == (2, 100)


def test_selector_cnn_stays_under_parameter_budget() -> None:
    model = SelectorCNN(output_dim=100)

    assert count_trainable_parameters(model) < 1_500_000
