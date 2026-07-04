from __future__ import annotations

import torch
import torchattacks

from core.base import AttackResult, BaseAttack, BaseDetector


class DeepFoolAttack(BaseAttack):
    name = "deepfool"

    def __init__(self, steps: int = 50, overshoot: float = 0.02) -> None:
        self.steps = steps
        self.overshoot = overshoot

    def perturb(
        self,
        model: BaseDetector,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> AttackResult:
        steps = int(kwargs.get("steps", self.steps))
        overshoot = float(kwargs.get("overshoot", self.overshoot))

        attack = torchattacks.DeepFool(model, steps=steps, overshoot=overshoot)
        adversarial_images = attack(images, labels)
        
        return AttackResult(
            adversarial_images=adversarial_images,
            perturbations=adversarial_images - images,
            metadata={"attack": self.name, "steps": steps, "overshoot": overshoot},
        )
