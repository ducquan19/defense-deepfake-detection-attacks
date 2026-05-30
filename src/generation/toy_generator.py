from __future__ import annotations

import torch

from core.base import BaseDeepfakeGenerator


def apply_toy_deepfake_batch(images: torch.Tensor, strength: float = 0.35) -> torch.Tensor:
    """Apply a lightweight artifact pattern used only for pipeline smoke tests."""
    _, _, height, width = images.shape
    y = torch.linspace(0, 1, height, device=images.device).view(1, 1, height, 1)
    x = torch.linspace(0, 1, width, device=images.device).view(1, 1, 1, width)
    artifact = torch.sin(40 * (x + y)).repeat(images.shape[0], images.shape[1], 1, 1)
    artifact = (artifact + 1) / 2
    return ((1 - strength) * images + strength * artifact).clamp(0, 1)


class ToyDeepfakeGenerator(BaseDeepfakeGenerator):
    name = "toy_blend"

    def __init__(self, strength: float = 0.35) -> None:
        self.strength = strength

    def generate(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        strength = float(kwargs.get("strength", self.strength))
        return apply_toy_deepfake_batch(images, strength=strength)
