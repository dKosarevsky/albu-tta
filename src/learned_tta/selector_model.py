"""Small CNN selector model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class SelectorOutputs:
    """Selector model outputs for gain and optional usefulness heads."""

    gain: torch.Tensor
    useful_logits: torch.Tensor | None = None


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable block used by the selector CNN."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SelectorCNN(nn.Module):
    """Lightweight CNN that predicts one TTA score per augmentation candidate."""

    def __init__(
        self,
        output_dim: int = 100,
        dropout: float = 0.1,
        usefulness_head: bool = False,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            DepthwiseSeparableConv(32, 32, stride=1),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 192, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, output_dim),
        )
        self.useful_head = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(192, output_dim),
            )
            if usefulness_head
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_heads(inputs).gain

    def forward_heads(self, inputs: torch.Tensor) -> SelectorOutputs:
        features = self.features(inputs)
        return SelectorOutputs(
            gain=self.head(features),
            useful_logits=(self.useful_head(features) if self.useful_head is not None else None),
        )


class SelectorMLP(nn.Module):
    """Small MLP selector for vector features such as clean-logit summaries."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        usefulness_head: bool = False,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)
        self.useful_head = nn.Linear(hidden_dim, output_dim) if usefulness_head else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_heads(inputs).gain

    def forward_heads(self, inputs: torch.Tensor) -> SelectorOutputs:
        features = self.trunk(inputs)
        return SelectorOutputs(
            gain=self.head(features),
            useful_logits=(self.useful_head(features) if self.useful_head is not None else None),
        )


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
