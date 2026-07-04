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
        
        # FIX FOR BINARY CLASSIFICATION:
        # AutoAttack 'standard' includes 'apgd-t' and 'fab-t' which use DLR loss.
        # DLR loss tries to find the 3rd and 4th highest logits (index -3, -4).
        # Since this is a binary task (2 logits), it crashes with IndexError.
        # Restricting to untargeted APGD-CE and Square is the mathematically 
        # correct equivalent of AutoAttack for binary classification.
        if hasattr(attack, "attacks_to_run"):
            attack.attacks_to_run = ['apgd-ce', 'square']
            
        adversarial_images = attack(images, labels)
        
        return AttackResult(
            adversarial_images=adversarial_images,
            perturbations=adversarial_images - images,
            metadata={"attack": self.name, "epsilon": epsilon, "version": version},
        )
