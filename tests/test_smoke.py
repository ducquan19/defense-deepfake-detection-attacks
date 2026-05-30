from __future__ import annotations

import torch

from attacks.fgsm import FGSMAttack
from config import load_config
from core.base import AttackResult, BaseAttack, BaseDeepfakeGenerator, BaseDefense, BaseDetector
from defenses.preprocessing import JPEGSmoothingDefense
from generation.toy_generator import ToyDeepfakeGenerator
from models.tiny_cnn import TinyCNN
from pipeline import run_experiment


def test_baseline_pipeline_runs(tmp_path):
    config = load_config("configs/experiment_baseline.yaml")
    values = dict(config.values)
    values["experiment"] = dict(values["experiment"])
    values["experiment"]["output_dir"] = str(tmp_path / "run")
    values["model"] = dict(values["model"])
    values["model"]["checkpoint_path"] = str(tmp_path / "tiny_cnn.pt")
    values["data"] = dict(values["data"])
    values["data"]["num_samples"] = 16
    values["data"]["batch_size"] = 8

    metrics = run_experiment(type(config)(values=values, path=None))

    assert "clean_accuracy" in metrics
    assert "attacked_accuracy" in metrics
    assert "defended_accuracy" in metrics


def test_components_follow_base_interfaces():
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    model = TinyCNN()

    generator = ToyDeepfakeGenerator(strength=0.2)
    defense = JPEGSmoothingDefense(quality=80)
    attack = FGSMAttack(epsilon=0.01)

    assert isinstance(model, BaseDetector)
    assert isinstance(generator, BaseDeepfakeGenerator)
    assert isinstance(defense, BaseDefense)
    assert isinstance(attack, BaseAttack)

    prediction = model.predict(images)
    generated = generator.generate(images)
    defended = defense.apply(images)
    attack_result = attack.perturb(model, images, labels)

    assert prediction.logits.shape == (2, 2)
    assert generated.shape == images.shape
    assert defended.shape == images.shape
    assert isinstance(attack_result, AttackResult)
    assert attack_result.adversarial_images.shape == images.shape
