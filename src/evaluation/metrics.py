from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from core.base import BaseAttack, BaseDefense, BaseDetector, BaseEvaluator


BatchTransform = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.enable_grad()
def evaluate_detector(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    transform: BatchTransform | None = None,
) -> dict[str, float]:
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
        all_predictions.extend(probabilities.argmax(dim=1).detach().cpu().tolist())
        all_scores.extend(probabilities[:, 1].detach().cpu().tolist())

    labels_array = np.asarray(all_labels)
    scores_array = np.asarray(all_scores)
    auc = 0.5 if len(np.unique(labels_array)) < 2 else float(roc_auc_score(labels_array, scores_array))
    return {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "roc_auc": auc,
    }


class StandardEvaluator(BaseEvaluator):
    def evaluate(
        self,
        model: BaseDetector,
        dataloader: DataLoader,
        device: torch.device,
        attack: BaseAttack | None = None,
        defense: BaseDefense | None = None,
    ) -> dict[str, float]:
        def transform(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            transformed = images
            if attack is not None:
                transformed = attack.perturb(model, transformed, labels).adversarial_images
            if defense is not None:
                transformed = defense.apply(transformed)
            return transformed

        return evaluate_detector(
            model=model,
            loader=dataloader,
            device=device,
            transform=transform if attack is not None or defense is not None else None,
        )
