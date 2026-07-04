from __future__ import annotations

import torch
import torchattacks

from core.base import AttackResult, BaseAttack, BaseDetector


class AutoAttackWrapper(BaseAttack):
    name = "autoattack"

    def __init__(self, epsilon: float = 8/255, version: str = "standard") -> None:
        self.epsilon = epsilon
        self.version = version

    def perturb(
        self,
        model: BaseDetector,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> AttackResult:
        epsilon = float(kwargs.get("epsilon", self.epsilon))
        version = kwargs.get("version", self.version)

        attack = torchattacks.AutoAttack(model, eps=epsilon, version=version)
        adversarial_images = attack(images, labels)
        
        return AttackResult(
            adversarial_images=adversarial_images,
            perturbations=adversarial_images - images,
            metadata={"attack": self.name, "epsilon": epsilon, "version": version},
        )
