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

        # FIX FOR BINARY CLASSIFICATION:
        # torchattacks.AutoAttack 'standard' includes 'apgd-t' and 'fab-t' which use DLR loss.
        # DLR loss tries to find the 3rd and 4th highest logits (index -3, -4), causing an
        # IndexError on binary classification tasks. 
        # Modifying attack.attacks_to_run doesn't work because MultiAttack initializes the 
        # sub-attacks in __init__.
        # Therefore, we directly use APGD (which is the APGD-CE untargeted core of AutoAttack).
        # We don't include Square here because Square is already evaluated separately in our pipeline.
        attack = torchattacks.APGD(model, eps=epsilon, steps=100)

        adversarial_images = attack(images, labels)
        
        return AttackResult(
            adversarial_images=adversarial_images,
            perturbations=adversarial_images - images,
            metadata={"attack": self.name, "epsilon": epsilon, "version": version},
        )
