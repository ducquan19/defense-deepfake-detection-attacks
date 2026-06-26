"""Full-featured evaluation metrics for deepfake detector robustness.

Provides:
- ``compute_full_metrics``  — Accuracy, Precision, Recall, F1, ROC-AUC
- ``compute_asr``           — Attack Success Rate (ASR)
- ``evaluate_detector``     — Single-pass evaluation with optional attack/defense transform
- ``RobustnessEvaluator``   — Multi-attack comparison returning a structured report
- ``StandardEvaluator``     — BaseEvaluator-compatible wrapper
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from core.base import BaseAttack, BaseDefense, BaseDetector, BaseEvaluator


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
BatchTransform = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def compute_full_metrics(
    true_labels: list[int],
    predicted_labels: list[int],
    scores: list[float],
) -> dict[str, float]:
    """Compute classification metrics from accumulated predictions.

    Args:
        true_labels:       Ground-truth class indices.
        predicted_labels:  Hard predictions (argmax).
        scores:            Soft probabilities for class 1 (fake).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc.
    """
    labels_arr = np.asarray(true_labels)
    preds_arr = np.asarray(predicted_labels)
    scores_arr = np.asarray(scores)

    # Handle edge case: only one class present (e.g., small synthetic sets)
    auc = (
        0.5
        if len(np.unique(labels_arr)) < 2
        else float(roc_auc_score(labels_arr, scores_arr))
    )
    return {
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "precision": float(
            precision_score(labels_arr, preds_arr, zero_division=0)
        ),
        "recall": float(
            recall_score(labels_arr, preds_arr, zero_division=0)
        ),
        "f1": float(
            f1_score(labels_arr, preds_arr, zero_division=0)
        ),
        "roc_auc": auc,
    }


def compute_asr(
    true_labels: list[int],
    clean_predictions: list[int],
    adv_predictions: list[int],
) -> float:
    """Compute Attack Success Rate (ASR).

    ASR = fraction of *correctly-classified clean* samples that are
    mis-classified after the attack.

    Args:
        true_labels:        Ground-truth labels.
        clean_predictions:  Model predictions on clean inputs.
        adv_predictions:    Model predictions on adversarial inputs.

    Returns:
        ASR in [0, 1].
    """
    true_arr = np.asarray(true_labels)
    clean_arr = np.asarray(clean_predictions)
    adv_arr = np.asarray(adv_predictions)

    # Only count samples that were correctly classified before the attack
    correctly_classified = clean_arr == true_arr
    n_correct = int(correctly_classified.sum())
    if n_correct == 0:
        return 0.0

    # Among those, how many got fooled?
    fooled = (adv_arr != true_arr) & correctly_classified
    return float(fooled.sum()) / n_correct


# ---------------------------------------------------------------------------
# Single-pass evaluation (supports optional attack/defense transform)
# ---------------------------------------------------------------------------

@torch.enable_grad()
def evaluate_detector(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    transform: BatchTransform | None = None,
) -> dict[str, float]:
    """Run one evaluation pass and return full classification metrics.

    Args:
        model:     Detector (will be set to eval mode internally).
        loader:    DataLoader yielding (images, labels) pairs.
        device:    Compute device.
        transform: Optional function ``(images, labels) → transformed_images``
                   applied before inference (e.g., an attack or defense).

    Returns:
        Dict containing accuracy, precision, recall, f1, roc_auc.
    """
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_scores: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if transform is not None:
            images = transform(images, labels)

        with torch.no_grad():
            logits = model(images)
            probabilities = logits.softmax(dim=1)

        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(
            probabilities.argmax(dim=1).detach().cpu().tolist()
        )
        all_scores.extend(probabilities[:, 1].detach().cpu().tolist())

    return compute_full_metrics(all_labels, all_predictions, all_scores)


# ---------------------------------------------------------------------------
# Robustness Evaluator — multi-attack comparison
# ---------------------------------------------------------------------------

class RobustnessEvaluator:
    """Evaluate a detector against multiple attacks and compute ASR.

    Usage::

        evaluator = RobustnessEvaluator(model, test_loader, device)
        report = evaluator.run(attacks=[fgsm, ifgsm, pgd])
        # report["clean"] → clean metrics dict
        # report["fgsm"]  → attacked metrics dict + "asr" key
    """

    def __init__(
        self,
        model: BaseDetector,
        loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model = model
        self.loader = loader
        self.device = device

    def _collect_predictions(
        self,
        transform: BatchTransform | None = None,
    ) -> tuple[list[int], list[int], list[float]]:
        """Return (true_labels, predicted_labels, fake_scores)."""
        self.model.eval()
        true_labels: list[int] = []
        predicted_labels: list[int] = []
        fake_scores: list[float] = []

        for images, labels in self.loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if transform is not None:
                images = transform(images, labels)

            with torch.no_grad():
                logits = self.model(images)
                probs = logits.softmax(dim=1)

            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(probs.argmax(dim=1).cpu().tolist())
            fake_scores.extend(probs[:, 1].cpu().tolist())

        return true_labels, predicted_labels, fake_scores

    def run(
        self,
        attacks: list[BaseAttack] | None = None,
        defenses: list[BaseDefense] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run clean evaluation + all provided attacks/defenses.

        Args:
            attacks:  List of attack instances to evaluate.
            defenses: List of preprocessing defense instances to evaluate.

        Returns:
            Dict keyed by protocol name (``"clean"``, attack name, defense name).
            Each value is a dict of metric floats; attacked protocols also
            include an ``"asr"`` key.
        """
        report: dict[str, dict[str, Any]] = {}

        # --- 1. Clean evaluation -------------------------------------------
        clean_true, clean_preds, clean_scores = self._collect_predictions()
        clean_metrics = compute_full_metrics(clean_true, clean_preds, clean_scores)
        report["clean"] = clean_metrics
        print(f"[RobustnessEvaluator] clean  → acc={clean_metrics['accuracy']:.4f}  "
              f"f1={clean_metrics['f1']:.4f}  auc={clean_metrics['roc_auc']:.4f}")

        # --- 2. Attack evaluations -----------------------------------------
        for attack in (attacks or []):
            def _transform(
                images: torch.Tensor,
                labels: torch.Tensor,
                _atk: BaseAttack = attack,
            ) -> torch.Tensor:
                return _atk.perturb(self.model, images, labels).adversarial_images

            adv_true, adv_preds, adv_scores = self._collect_predictions(_transform)
            adv_metrics = compute_full_metrics(adv_true, adv_preds, adv_scores)
            asr = compute_asr(adv_true, clean_preds, adv_preds)
            adv_metrics["asr"] = asr

            report[attack.name] = adv_metrics
            print(
                f"[RobustnessEvaluator] {attack.name:<8} → "
                f"acc={adv_metrics['accuracy']:.4f}  "
                f"f1={adv_metrics['f1']:.4f}  "
                f"auc={adv_metrics['roc_auc']:.4f}  "
                f"asr={asr:.4f}"
            )

        # --- 3. Preprocessing defense evaluations -------------------------
        for defense in (defenses or []):
            def _def_transform(
                images: torch.Tensor,
                labels: torch.Tensor,
                _def: BaseDefense = defense,
            ) -> torch.Tensor:
                return _def.apply(images)

            def_true, def_preds, def_scores = self._collect_predictions(_def_transform)
            def_metrics = compute_full_metrics(def_true, def_preds, def_scores)
            report[f"defense_{defense.name}"] = def_metrics
            print(
                f"[RobustnessEvaluator] defense/{defense.name} → "
                f"acc={def_metrics['accuracy']:.4f}  "
                f"f1={def_metrics['f1']:.4f}"
            )

        return report


# ---------------------------------------------------------------------------
# BaseEvaluator-compatible wrapper (backward compatible)
# ---------------------------------------------------------------------------

class StandardEvaluator(BaseEvaluator):
    """Standard evaluator compatible with BaseEvaluator interface."""

    def evaluate(
        self,
        model: BaseDetector,
        dataloader: DataLoader,
        device: torch.device,
        attack: BaseAttack | None = None,
        defense: BaseDefense | None = None,
    ) -> dict[str, float]:
        def transform(
            images: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            transformed = images
            if attack is not None:
                transformed = attack.perturb(
                    model, transformed, labels
                ).adversarial_images
            if defense is not None:
                transformed = defense.apply(transformed)
            return transformed

        return evaluate_detector(
            model=model,
            loader=dataloader,
            device=device,
            transform=transform if attack is not None or defense is not None else None,
        )
