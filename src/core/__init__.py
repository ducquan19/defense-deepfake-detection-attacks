"""Core interfaces shared by all research components."""

from core.base import (
    AttackResult,
    BaseAttack,
    BaseDataModule,
    BaseDeepfakeGenerator,
    BaseDefense,
    BaseDetector,
    BaseEvaluator,
    Batch,
    Prediction,
)

__all__ = [
    "AttackResult",
    "BaseAttack",
    "BaseDataModule",
    "BaseDeepfakeGenerator",
    "BaseDefense",
    "BaseDetector",
    "BaseEvaluator",
    "Batch",
    "Prediction",
]
