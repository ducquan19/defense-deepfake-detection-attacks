from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from attacks.fgsm import fgsm_attack
from config import ExperimentConfig
from data.synthetic import build_synthetic_loaders
from defenses.preprocessing import jpeg_smoothing_batch
from evaluation.metrics import evaluate_detector
from generation.toy_generator import apply_toy_deepfake_batch
from models.tiny_cnn import TinyCNN
from training import train_one_epoch
from utils.reproducibility import seed_everything


def run_experiment(config: ExperimentConfig) -> dict[str, float]:
    seed_everything(config.seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.path is not None:
        shutil.copy2(config.path, output_dir / "config.yaml")

    data_cfg = config.values.get("data", {})
    model_cfg = config.values.get("model", {})
    attack_cfg = config.values.get("attack", {})
    defense_cfg = config.values.get("defense", {})
    generation_cfg = config.values.get("generation", {})

    train_loader, test_loader = build_synthetic_loaders(
        num_samples=int(data_cfg.get("num_samples", 64)),
        image_size=int(data_cfg.get("image_size", 128)),
        batch_size=int(data_cfg.get("batch_size", 16)),
        train_split=float(data_cfg.get("train_split", 0.75)),
        seed=config.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg.get("learning_rate", 1e-3)))

    for _ in range(int(model_cfg.get("epochs", 1))):
        train_one_epoch(model, train_loader, optimizer, device)

    clean_metrics = evaluate_detector(model, test_loader, device)
    metrics: dict[str, float] = {f"clean_{key}": value for key, value in clean_metrics.items()}

    if generation_cfg.get("enabled", True):
        metrics["toy_generation_strength"] = float(generation_cfg.get("strength", 0.35))

    if attack_cfg.get("enabled", True):
        attacked_metrics = evaluate_detector(
            model,
            test_loader,
            device,
            transform=lambda images, labels: fgsm_attack(
                model,
                images,
                labels,
                epsilon=float(attack_cfg.get("epsilon", 0.03)),
            ),
        )
        metrics.update({f"attacked_{key}": value for key, value in attacked_metrics.items()})

    if defense_cfg.get("enabled", True):
        defended_metrics = evaluate_detector(
            model,
            test_loader,
            device,
            transform=lambda images, labels: jpeg_smoothing_batch(
                apply_toy_deepfake_batch(images, strength=float(generation_cfg.get("strength", 0.35))),
                quality=int(defense_cfg.get("jpeg_quality", 75)),
            ),
        )
        metrics.update({f"defended_{key}": value for key, value in defended_metrics.items()})

    checkpoint_path = Path(model_cfg.get("checkpoint_path", "models/tiny_cnn.pt"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics
