"""PGD (Projected Gradient Descent) attack.

Formula:
    x_0  = x + δ_0  (δ_0 ~ Uniform(-ε, ε))  — random start
    x_{t+1} = Proj_{B(x,ε)} (x_t + α · sign(∇_x L))

Reference: Madry et al. (2018) "Towards Deep Learning Models Resistant to Adversarial Attacks"
"""

from __future__ import annotations

import torch
from torch import nn

from core.base import AttackResult, BaseAttack, BaseDetector


@torch.enable_grad()
def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.01,
    alpha: float = 0.001,
    steps: int = 20,
    random_start: bool = True,
) -> torch.Tensor:
    """Run PGD attack and return adversarial images clamped to [0, 1].

    Args:
        model:        Detector in eval mode.
        images:       Clean images [B, 3, H, W], float32 in [0, 1].
        labels:       Ground-truth class indices [B].
        epsilon:      Maximum L∞ perturbation budget.
        alpha:        Per-step size.
        steps:        Number of iterative steps.
        random_start: If True, initialise from a uniformly random perturbation
                      inside the ε-ball (recommended for PGD).

    Returns:
        Adversarial images [B, 3, H, W] clamped to [0, 1].
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    x_orig = images.detach().clone()

    if random_start:
        delta = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = (x_orig + delta).clamp(0.0, 1.0).detach()
    else:
        x_adv = x_orig.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign
            # Project back into the ε-ball (L∞)
            x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
            x_adv = x_adv.clamp(0.0, 1.0).detach()

    return x_adv


class PGDAttack(BaseAttack):
    """PGD adversarial attack (Madry et al., 2018)."""

    name = "pgd"

    def __init__(
        self,
        epsilon: float = 0.01,
        alpha: float = 0.001,
        steps: int = 20,
        random_start: bool = True,
    ) -> None:
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def perturb(
        self,
        model: BaseDetector,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> AttackResult:
        epsilon = float(kwargs.get("epsilon", self.epsilon))
        alpha = float(kwargs.get("alpha", self.alpha))
        steps = int(kwargs.get("steps", self.steps))
        random_start = bool(kwargs.get("random_start", self.random_start))

        adversarial_images = pgd_attack(
            model, images, labels,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
        )
        return AttackResult(
            adversarial_images=adversarial_images,
            perturbations=adversarial_images - images,
            metadata={
                "attack": self.name,
                "epsilon": epsilon,
                "alpha": alpha,
                "steps": steps,
                "random_start": random_start,
            },
        )
