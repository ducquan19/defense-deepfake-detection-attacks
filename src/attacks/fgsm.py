from __future__ import annotations

import torch
from torch import nn

from core.base import AttackResult, BaseAttack, BaseDetector


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    model.eval()
    adversarial_images = images.detach().clone().requires_grad_(True)
    logits = model(adversarial_images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()
    perturbed = adversarial_images + epsilon * adversarial_images.grad.sign()
    return perturbed.detach().clamp(0, 1)


class FGSMAttack(BaseAttack):
    name = "fgsm"

    def __init__(self, epsilon: float = 0.03) -> None:
        self.epsilon = epsilon

    def perturb(
        self,
        model: BaseDetector,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> AttackResult:
        epsilon = float(kwargs.get("epsilon", self.epsilon))
        adversarial_images = fgsm_attack(model, images, labels, epsilon=epsilon)
        return AttackResult(
            adversarial_images=adversarial_images,
            perturbations=adversarial_images - images,
            metadata={"attack": self.name, "epsilon": epsilon},
        )
