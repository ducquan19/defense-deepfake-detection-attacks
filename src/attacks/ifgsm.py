"""I-FGSM (Iterative Fast Gradient Sign Method / BIM) attack.

Formula:
    x_{t+1} = clip(x_t + α · sign(∇_x L), x - ε, x + ε)

Reference: Kurakin et al. (2017) "Adversarial examples in the physical world"
"""

from __future__ import annotations

import torch
from torch import nn

from core.base import AttackResult, BaseAttack, BaseDetector


@torch.enable_grad()
def ifgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int = 10,
    alpha: float | None = None,
) -> torch.Tensor:
    """Run iterative FGSM and return adversarial images clamped to [0, 1].

    Args:
        model:   Detector in eval mode.
        images:  Clean images [B, 3, H, W], float32 in [0, 1].
        labels:  Ground-truth class indices [B].
        epsilon: Maximum L∞ perturbation budget.
        steps:   Number of iterative steps.
        alpha:   Per-step size; defaults to epsilon / steps.

    Returns:
        Adversarial images [B, 3, H, W] clamped to [0, 1].
    """
    if alpha is None:
        alpha = epsilon / steps

    model.eval()
    criterion = nn.CrossEntropyLoss()
    x_orig = images.detach().clone()
    x_adv = images.detach().clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign
            # Project back into the ε-ball around the original image
            x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
            x_adv = x_adv.clamp(0.0, 1.0).detach()

    return x_adv


class IFGSMAttack(BaseAttack):
    """Iterative FGSM (Basic Iterative Method) adversarial attack."""

    name = "ifgsm"

    def __init__(
        self,
        epsilon: float = 0.03,
        steps: int = 10,
        alpha: float | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.steps = steps
        # If alpha not set, it is computed dynamically per call
        self.alpha = alpha

    def perturb(
        self,
        model: BaseDetector,
        images: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> AttackResult:
        epsilon = float(kwargs.get("epsilon", self.epsilon))
        steps = int(kwargs.get("steps", self.steps))
        alpha_raw = kwargs.get("alpha", self.alpha)
        alpha = float(alpha_raw) if alpha_raw is not None else None

        adversarial_images = ifgsm_attack(
            model, images, labels, epsilon=epsilon, steps=steps, alpha=alpha
        )
        effective_alpha = alpha if alpha is not None else epsilon / steps
        return AttackResult(
            adversarial_images=adversarial_images,
            perturbations=adversarial_images - images,
            metadata={
                "attack": self.name,
                "epsilon": epsilon,
                "steps": steps,
                "alpha": effective_alpha,
            },
        )
