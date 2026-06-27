"""Adversarial Training defense module.

Strategy:
    At each training step, generate adversarial examples on-the-fly
    (using PGD by default, configurable) and mix them with clean samples.

Loss:
    L = L_clean + λ · L_adv

Reference: Madry et al. (2018) "Towards Deep Learning Models Resistant to
           Adversarial Attacks"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from attacks.pgd import pgd_attack
from attacks.fgsm import fgsm_attack
from core.base import BaseAttack, BaseDetector


# ---------------------------------------------------------------------------
# Adversarial Training Loop
# ---------------------------------------------------------------------------

def adversarial_train_one_epoch(
    model: BaseDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    attack_fn: str = "pgd",
    epsilon: float = 0.01,
    alpha: float = 0.001,
    steps: int = 10,
    adv_lambda: float = 1.0,
    adv_ratio: float = 0.5,
) -> dict[str, float]:
    """Train one epoch with mixed clean + adversarial batches.

    Args:
        model:      Detector being hardened.
        loader:     DataLoader yielding (images, labels) tuples.
        optimizer:  Optimizer for the model parameters.
        device:     Compute device.
        attack_fn:  "pgd" or "fgsm" — which attack to use for training.
        epsilon:    Perturbation budget.
        alpha:      Per-step size (PGD only).
        steps:      Number of attack iterations.
        adv_lambda: Weight λ for the adversarial loss term.
        adv_ratio:  Fraction of each batch to perturb adversarially.

    Returns:
        Dict with keys ``clean_loss``, ``adv_loss``, ``total_loss``.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_clean = 0.0
    total_adv = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # --- Generate adversarial perturbations (no grad update yet) ------
        model.eval()  # Switch to eval for attack generation
        with torch.enable_grad():
            if attack_fn == "fgsm":
                adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
            else:
                adv_images = pgd_attack(
                    model, images, labels,
                    epsilon=epsilon,
                    alpha=alpha,
                    steps=steps,
                    random_start=True,
                )
        # Switch back to train mode for weight update
        model.train()

        optimizer.zero_grad(set_to_none=True)

        # --- Forward clean --------------------------------------------------
        clean_logits = model(images)
        clean_loss = criterion(clean_logits, labels)

        # --- Forward adversarial -------------------------------------------
        adv_logits = model(adv_images)
        adv_loss = criterion(adv_logits, labels)

        # --- Combined loss --------------------------------------------------
        total_loss = clean_loss + adv_lambda * adv_loss
        total_loss.backward()
        optimizer.step()

        total_clean += float(clean_loss.detach().cpu())
        total_adv += float(adv_loss.detach().cpu())
        n_batches += 1

    denom = max(n_batches, 1)
    avg_clean = total_clean / denom
    avg_adv = total_adv / denom
    return {
        "clean_loss": avg_clean,
        "adv_loss": avg_adv,
        "total_loss": avg_clean + adv_lambda * avg_adv,
    }


# ---------------------------------------------------------------------------
# High-Level Adversarial Trainer
# ---------------------------------------------------------------------------

class AdversarialTrainer:
    """High-level trainer that performs adversarial training on a detector.

    Args:
        model:          Detector to be hardened.
        optimizer:      Optimiser (already configured with LR, weight decay).
        device:         Compute device.
        attack_fn:      Attack used during training ("pgd" or "fgsm").
        epsilon:        L∞ perturbation budget.
        alpha:          Per-step size for PGD.
        steps:          PGD step count.
        adv_lambda:     λ weight for adversarial loss.
        checkpoint_dir: Directory to save the robust checkpoint.
    """

    def __init__(
        self,
        model: BaseDetector,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        attack_fn: str = "pgd",
        epsilon: float = 0.01,
        alpha: float = 0.001,
        steps: int = 10,
        adv_lambda: float = 1.0,
        checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.attack_fn = attack_fn
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.adv_lambda = adv_lambda
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Run adversarial training for the specified number of epochs.

        Returns:
            List of per-epoch metric dicts.
        """
        for epoch in range(1, epochs + 1):
            losses = adversarial_train_one_epoch(
                model=self.model,
                loader=train_loader,
                optimizer=self.optimizer,
                device=self.device,
                attack_fn=self.attack_fn,
                epsilon=self.epsilon,
                alpha=self.alpha,
                steps=self.steps,
                adv_lambda=self.adv_lambda,
            )
            record = {"epoch": epoch, **losses}
            self.history.append(record)

            if verbose:
                print(
                    f"[AdversarialTrainer] Epoch {epoch:>3}/{epochs} | "
                    f"clean_loss={losses['clean_loss']:.4f}  "
                    f"adv_loss={losses['adv_loss']:.4f}  "
                    f"total_loss={losses['total_loss']:.4f}"
                )

        return self.history

    def save_checkpoint(self, filename: str = "robust_detector.pth") -> Path:
        """Save the hardened model to ``checkpoints/<filename>``.

        Returns:
            Path to the saved checkpoint.
        """
        save_path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_config": {
                    "attack_fn": self.attack_fn,
                    "epsilon": self.epsilon,
                    "alpha": self.alpha,
                    "steps": self.steps,
                    "adv_lambda": self.adv_lambda,
                },
                "history": self.history,
            },
            save_path,
        )
        print(f"[AdversarialTrainer] Checkpoint saved → {save_path}")
        return save_path

    def save_history(self, filename: str = "adversarial_training_history.json") -> Path:
        """Save training history to JSON for later analysis."""
        out_path = self.checkpoint_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return out_path
