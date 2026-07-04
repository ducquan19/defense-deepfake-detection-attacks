from __future__ import annotations

import torch
import torchattacks

from core.base import AttackResult, BaseAttack, BaseDetector


class MIFGSMAttack(BaseAttack):
    name = "mifgsm"

    def __init__(self, epsilon: float = 8/255, steps: int = 10, decay: float = 1.0) -> None:
        self.epsilon = epsilon
        self.steps = steps
        self.decay = decay

    def perturb(
        self,
        model: BaseDetector,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> AttackResult:
        epsilon = float(kwargs.get("epsilon", self.epsilon))
        steps = int(kwargs.get("steps", self.steps))
        decay = float(kwargs.get("decay", self.decay))

        attack = torchattacks.MIFGSM(model, eps=epsilon, steps=steps, decay=decay)
        adversarial_images = attack(images, labels)
        
        return AttackResult(
            adversarial_images=adversarial_images,
            perturbations=adversarial_images - images,
            metadata={"attack": self.name, "epsilon": epsilon, "steps": steps, "decay": decay},
        )
