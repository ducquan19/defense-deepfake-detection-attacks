from __future__ import annotations

import torch
from torch import nn

from core.base import BaseDetector


class TinyCNN(BaseDetector):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).flatten(1)
        return self.classifier(features)
