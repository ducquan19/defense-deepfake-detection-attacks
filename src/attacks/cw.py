from __future__ import annotations

import torch
import torchattacks

from core.base import AttackResult, BaseAttack, BaseDetector


class CWAttack(BaseAttack):
    name = "cw"

    def __init__(self, c: float = 1, kappa: float = 0, steps: int = 50, lr: float = 0.01) -> None:
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def perturb(
        self,
        model: BaseDetector,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> AttackResult:
        c = float(kwargs.get("c", self.c))
        kappa = float(kwargs.get("kappa", self.kappa))
        steps = int(kwargs.get("steps", self.steps))
        lr = float(kwargs.get("lr", self.lr))

        attack = torchattacks.CW(model, c=c, kappa=kappa, steps=steps, lr=lr)
        adversarial_images = attack(images, labels)
        
        return AttackResult(
            adversarial_images=adversarial_images,
            perturbations=adversarial_images - images,
            metadata={"attack": self.name, "c": c, "kappa": kappa, "steps": steps, "lr": lr},
        )
