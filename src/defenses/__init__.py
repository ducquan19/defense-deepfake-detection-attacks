"""Adversarial defense implementations."""

from defenses.preprocessing import JPEGSmoothingDefense, jpeg_smoothing_batch
from defenses.adversarial_training import AdversarialTrainer, adversarial_train_one_epoch

DEFENSE_REGISTRY: dict[str, type] = {
    "jpeg_smoothing": JPEGSmoothingDefense,
}

__all__ = [
    "JPEGSmoothingDefense",
    "jpeg_smoothing_batch",
    "AdversarialTrainer",
    "adversarial_train_one_epoch",
    "DEFENSE_REGISTRY",
]
