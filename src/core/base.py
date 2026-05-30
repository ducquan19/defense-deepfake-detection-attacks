from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


Metadata = dict[str, Any]


@dataclass(frozen=True)
class Batch:
    """Standard batch contract used across data, model, attack, and defense code."""

    images: torch.Tensor
    labels: torch.Tensor
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    """Detector output contract.

    Convention:
    - class 0 = real
    - class 1 = fake
    """

    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_labels: torch.Tensor

    @property
    def fake_scores(self) -> torch.Tensor:
        return self.probabilities[:, 1]


@dataclass(frozen=True)
class AttackResult:
    """Output of an adversarial attack."""

    adversarial_images: torch.Tensor
    perturbations: torch.Tensor
    metadata: Metadata = field(default_factory=dict)


class BaseDataModule(ABC):
    """Dataset/data-loader interface for reproducible experiments."""

    @abstractmethod
    def setup(self) -> None:
        """Prepare datasets, transforms, splits, or metadata."""

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""


class BaseDeepfakeGenerator(ABC):
    """Interface for deepfake or fake-image generation modules."""

    @abstractmethod
    def generate(self, images: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return generated images with the same tensor convention as input.

        Tensor convention is [B, 3, H, W], float32, range [0, 1].
        """


class BaseDetector(nn.Module, ABC):
    """Base class for binary deepfake detectors."""

    num_classes: int = 2

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [B, 2]."""

    def predict(self, images: torch.Tensor) -> Prediction:
        logits = self.forward(images)
        probabilities = logits.softmax(dim=1)
        predicted_labels = probabilities.argmax(dim=1)
        return Prediction(
            logits=logits,
            probabilities=probabilities,
            predicted_labels=predicted_labels,
        )


class BaseAttack(ABC):
    """Interface for authorized robustness attacks against a detector."""

    name: str

    @abstractmethod
    def perturb(
        self,
        model: BaseDetector,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs: Any,
    ) -> AttackResult:
        """Return adversarial images and attack metadata."""


class BaseDefense(ABC):
    """Interface for defenses applied before or during detection."""

    name: str

    @abstractmethod
    def apply(self, images: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Return defended images with shape [B, 3, H, W]."""


class BaseEvaluator(ABC):
    """Interface for clean, attacked, and defended evaluation."""

    @abstractmethod
    def evaluate(
        self,
        model: BaseDetector,
        dataloader: DataLoader,
        device: torch.device,
        attack: BaseAttack | None = None,
        defense: BaseDefense | None = None,
    ) -> dict[str, float]:
        """Return scalar metrics for one evaluation protocol."""
