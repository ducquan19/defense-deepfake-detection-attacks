"""Smoke tests for the full adversarial attack & defense pipeline.

Covers:
- Baseline pipeline (FGSM + JPEG defense)
- I-FGSM attack
- PGD attack
- Adversarial training (one epoch)
- Extended metrics (F1, Precision, Recall, ASR, ROC-AUC)
- RobustnessEvaluator multi-attack report
- All components implement correct base interfaces
"""

from __future__ import annotations

import torch
import pytest

from attacks.fgsm import FGSMAttack, fgsm_attack
from attacks.ifgsm import IFGSMAttack, ifgsm_attack
from attacks.pgd import PGDAttack, pgd_attack
from attacks import ATTACK_REGISTRY
from config import load_config
from core.base import AttackResult, BaseAttack, BaseDeepfakeGenerator, BaseDefense, BaseDetector
from defenses.preprocessing import JPEGSmoothingDefense
from defenses.adversarial_training import AdversarialTrainer, adversarial_train_one_epoch
from evaluation.metrics import (
    RobustnessEvaluator,
    StandardEvaluator,
    compute_full_metrics,
    compute_asr,
    evaluate_detector,
)
from generation.toy_generator import ToyDeepfakeGenerator
from models.tiny_cnn import TinyCNN
from pipeline import run_experiment


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_model():
    return TinyCNN()


@pytest.fixture()
def fake_batch():
    """Small batch of (images, labels) for unit-level testing."""
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    return images, labels


@pytest.fixture()
def fake_loader(fake_batch):
    from torch.utils.data import TensorDataset, DataLoader
    images, labels = fake_batch
    return DataLoader(TensorDataset(images, labels), batch_size=4)


# ---------------------------------------------------------------------------
# Tests: Baseline pipeline
# ---------------------------------------------------------------------------

def test_baseline_pipeline_runs(tmp_path):
    """Baseline pipeline should run without error and return clean_accuracy."""
    config = load_config("configs/experiment_baseline.yaml")
    values = dict(config.values)
    values["experiment"] = dict(values["experiment"])
    values["experiment"]["output_dir"] = str(tmp_path / "run")
    values["model"] = dict(values["model"])
    values["model"]["checkpoint_path"] = str(tmp_path / "tiny_cnn.pt")
    values["data"] = dict(values["data"])
    values["data"]["num_samples"] = 16
    values["data"]["batch_size"] = 8
    values["attack"] = {"enabled": False}
    values["defense"] = {"enabled": False}
    values["adversarial_training"] = {"enabled": False}

    metrics = run_experiment(type(config)(values=values, path=None))

    assert "clean_accuracy" in metrics


# ---------------------------------------------------------------------------
# Tests: FGSM Attack
# ---------------------------------------------------------------------------

def test_fgsm_perturbs_images(tiny_model, fake_batch):
    images, labels = fake_batch
    result = FGSMAttack(epsilon=0.01).perturb(tiny_model, images, labels)

    assert isinstance(result, AttackResult)
    assert result.adversarial_images.shape == images.shape
    assert result.adversarial_images.min() >= -1e-6
    assert result.adversarial_images.max() <= 1.0 + 1e-6
    assert not torch.allclose(result.adversarial_images, images)


def test_fgsm_metadata(tiny_model, fake_batch):
    images, labels = fake_batch
    result = FGSMAttack(epsilon=0.05).perturb(tiny_model, images, labels)
    assert result.metadata["attack"] == "fgsm"
    assert result.metadata["epsilon"] == 0.05


# ---------------------------------------------------------------------------
# Tests: I-FGSM Attack
# ---------------------------------------------------------------------------

def test_ifgsm_perturbs_images(tiny_model, fake_batch):
    images, labels = fake_batch
    result = IFGSMAttack(epsilon=0.01, steps=5).perturb(tiny_model, images, labels)

    assert isinstance(result, AttackResult)
    assert result.adversarial_images.shape == images.shape
    assert result.adversarial_images.min() >= -1e-6
    assert result.adversarial_images.max() <= 1.0 + 1e-6


def test_ifgsm_stays_within_epsilon_ball(tiny_model, fake_batch):
    """I-FGSM adversarial images must lie within the L∞ epsilon-ball."""
    images, labels = fake_batch
    epsilon = 0.02
    result = IFGSMAttack(epsilon=epsilon, steps=10).perturb(tiny_model, images, labels)
    diff = (result.adversarial_images - images).abs()
    assert diff.max().item() <= epsilon + 1e-5


def test_ifgsm_metadata(tiny_model, fake_batch):
    images, labels = fake_batch
    result = IFGSMAttack(epsilon=0.01, steps=7).perturb(tiny_model, images, labels)
    assert result.metadata["attack"] == "ifgsm"
    assert result.metadata["steps"] == 7


# ---------------------------------------------------------------------------
# Tests: PGD Attack
# ---------------------------------------------------------------------------

def test_pgd_perturbs_images(tiny_model, fake_batch):
    images, labels = fake_batch
    result = PGDAttack(epsilon=0.01, alpha=0.001, steps=5).perturb(tiny_model, images, labels)

    assert isinstance(result, AttackResult)
    assert result.adversarial_images.shape == images.shape
    assert result.adversarial_images.min() >= -1e-6
    assert result.adversarial_images.max() <= 1.0 + 1e-6


def test_pgd_stays_within_epsilon_ball(tiny_model, fake_batch):
    """PGD adversarial images must lie within the L∞ epsilon-ball."""
    images, labels = fake_batch
    epsilon = 0.01
    result = PGDAttack(epsilon=epsilon, alpha=0.001, steps=10).perturb(tiny_model, images, labels)
    diff = (result.adversarial_images - images).abs()
    assert diff.max().item() <= epsilon + 1e-5


def test_pgd_metadata(tiny_model, fake_batch):
    images, labels = fake_batch
    result = PGDAttack(epsilon=0.01, alpha=0.001, steps=20).perturb(tiny_model, images, labels)
    assert result.metadata["attack"] == "pgd"
    assert result.metadata["steps"] == 20
    assert result.metadata["epsilon"] == 0.01


# ---------------------------------------------------------------------------
# Tests: Attack Registry
# ---------------------------------------------------------------------------

def test_attack_registry_contains_all():
    assert "fgsm" in ATTACK_REGISTRY
    assert "ifgsm" in ATTACK_REGISTRY
    assert "pgd" in ATTACK_REGISTRY


# ---------------------------------------------------------------------------
# Tests: Adversarial Training
# ---------------------------------------------------------------------------

def test_adversarial_train_one_epoch(tiny_model, fake_loader):
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    losses = adversarial_train_one_epoch(
        model=tiny_model,
        loader=fake_loader,
        optimizer=optimizer,
        device=device,
        attack_fn="pgd",
        epsilon=0.01,
        alpha=0.001,
        steps=3,
        adv_lambda=1.0,
    )
    assert "clean_loss" in losses
    assert "adv_loss" in losses
    assert "total_loss" in losses
    assert losses["clean_loss"] >= 0
    assert losses["adv_loss"] >= 0


def test_adversarial_trainer_fit_and_save(tiny_model, fake_loader, tmp_path):
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    trainer = AdversarialTrainer(
        model=tiny_model,
        optimizer=optimizer,
        device=device,
        epsilon=0.01,
        alpha=0.001,
        steps=3,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    history = trainer.fit(train_loader=fake_loader, epochs=2, verbose=False)
    assert len(history) == 2

    ckpt_path = trainer.save_checkpoint("test_robust.pth")
    assert ckpt_path.exists()

    # Checkpoint must contain model_state_dict
    import torch as _torch
    ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "training_config" in ckpt


# ---------------------------------------------------------------------------
# Tests: Full Metrics
# ---------------------------------------------------------------------------

def test_compute_full_metrics():
    true_labels = [0, 1, 0, 1, 0, 1]
    pred_labels = [0, 1, 0, 0, 0, 1]   # one FN
    scores      = [0.1, 0.9, 0.2, 0.4, 0.1, 0.8]

    metrics = compute_full_metrics(true_labels, pred_labels, scores)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_compute_asr_all_fooled():
    true_labels   = [0, 1, 0, 1]
    clean_preds   = [0, 1, 0, 1]   # all correct
    adv_preds     = [1, 0, 1, 0]   # all flipped
    asr = compute_asr(true_labels, clean_preds, adv_preds)
    assert abs(asr - 1.0) < 1e-6


def test_compute_asr_none_fooled():
    true_labels   = [0, 1, 0, 1]
    clean_preds   = [0, 1, 0, 1]
    adv_preds     = [0, 1, 0, 1]   # same as clean
    asr = compute_asr(true_labels, clean_preds, adv_preds)
    assert abs(asr - 0.0) < 1e-6


def test_evaluate_detector_returns_full_metrics(tiny_model, fake_loader):
    device = torch.device("cpu")
    metrics = evaluate_detector(tiny_model, fake_loader, device)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in metrics, f"Missing metric: {key}"


# ---------------------------------------------------------------------------
# Tests: RobustnessEvaluator
# ---------------------------------------------------------------------------

def test_robustness_evaluator_clean(tiny_model, fake_loader):
    device = torch.device("cpu")
    evaluator = RobustnessEvaluator(tiny_model, fake_loader, device)
    report = evaluator.run(attacks=[], defenses=[])
    assert "clean" in report
    assert "accuracy" in report["clean"]


def test_robustness_evaluator_with_attacks(tiny_model, fake_loader):
    device = torch.device("cpu")
    evaluator = RobustnessEvaluator(tiny_model, fake_loader, device)
    attacks = [
        FGSMAttack(epsilon=0.01),
        IFGSMAttack(epsilon=0.01, steps=3),
        PGDAttack(epsilon=0.01, alpha=0.001, steps=3),
    ]
    report = evaluator.run(attacks=attacks)
    assert "clean" in report
    assert "fgsm" in report
    assert "ifgsm" in report
    assert "pgd" in report
    for key in ("fgsm", "ifgsm", "pgd"):
        assert "asr" in report[key]
        assert 0.0 <= report[key]["asr"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: Base interface compliance
# ---------------------------------------------------------------------------

def test_components_follow_base_interfaces():
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    model = TinyCNN()

    generator = ToyDeepfakeGenerator(strength=0.2)
    defense = JPEGSmoothingDefense(quality=80)
    fgsm = FGSMAttack(epsilon=0.01)
    ifgsm = IFGSMAttack(epsilon=0.01, steps=3)
    pgd = PGDAttack(epsilon=0.01, alpha=0.001, steps=3)

    assert isinstance(model, BaseDetector)
    assert isinstance(generator, BaseDeepfakeGenerator)
    assert isinstance(defense, BaseDefense)
    assert isinstance(fgsm, BaseAttack)
    assert isinstance(ifgsm, BaseAttack)
    assert isinstance(pgd, BaseAttack)

    prediction = model.predict(images)
    generated = generator.generate(images)
    defended = defense.apply(images)

    for attack in (fgsm, ifgsm, pgd):
        result = attack.perturb(model, images, labels)
        assert isinstance(result, AttackResult)
        assert result.adversarial_images.shape == images.shape

    assert prediction.logits.shape == (2, 2)
    assert generated.shape == images.shape
    assert defended.shape == images.shape
