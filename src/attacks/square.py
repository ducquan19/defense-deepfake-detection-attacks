from __future__ import annotations

import torch
import torchattacks

from core.base import AttackResult, BaseAttack, BaseDetector


class SquareAttackWrapper(BaseAttack):
    name = "square"

    def __init__(self, epsilon: float = 8/255, n_queries: int = 100) -> None:
        self.epsilon = epsilon
        self.n_queries = n_queries

    def perturb(
        self,
        model: BaseDetector,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> AttackResult:
        epsilon = float(kwargs.get("epsilon", self.epsilon))
        n_queries = int(kwargs.get("n_queries", self.n_queries))

        # norm='Linf' is default in torchattacks.Square
        attack = torchattacks.Square(model, eps=epsilon, n_queries=n_queries)
        adversarial_images = attack(images, labels)
        
        return AttackResult(
            adversarial_images=adversarial_images,
            perturbations=adversarial_images - images,
            metadata={"attack": self.name, "epsilon": epsilon, "n_queries": n_queries},
        )
