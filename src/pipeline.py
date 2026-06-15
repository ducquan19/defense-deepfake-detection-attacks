from __future__ import annotations

import datetime
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
from models.dino_mac import DinoMACForDeepfakeDetection
from training import train_one_epoch
from utils.reproducibility import seed_everything


def run_experiment(config: ExperimentConfig) -> dict[str, float]:
    seed_everything(config.seed)
    timestamp = str(int(datetime.datetime.now().timestamp()))
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.path is not None:
        shutil.copy2(config.path, output_dir / f"config_{config.seed}_{timestamp}.yaml")

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
    
    if model_cfg.get("name") == "dino_mac":
        model = DinoMACForDeepfakeDetection(
            model_name=model_cfg.get("huggingface_model_name", "facebook/dinov2-with-registers-large"),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
            use_stochastic_depth=model_cfg.get("use_stochastic_depth", True)
        ).to(device)
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': float(model_cfg.get("backbone_lr", 1e-5)), 'weight_decay': float(model_cfg.get("weight_decay", 1e-4))},
            {'params': model.mac_head.parameters(), 'lr': float(model_cfg.get("head_lr", 1e-4)), 'weight_decay': float(model_cfg.get("weight_decay", 1e-4))}
        ])
    else:
        model = TinyCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg.get("learning_rate", 1e-3)))

    checkpoint_path = Path(model_cfg.get("checkpoint_path", "models/tiny_cnn.pt"))
    
    # 1. Tải trọng số từ file nếu config có flag load_checkpoint = True
    if model_cfg.get("load_checkpoint", False):
        if checkpoint_path.exists():
            print(f"[+] Loading saved model weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"[!] WARNING: load_checkpoint=True nhưng không tìm thấy file {checkpoint_path}. Mô hình sẽ chạy với trọng số ngẫu nhiên.")
            
    # 2. Thực thi huấn luyện nếu config có flag train = True (mặc định là True)
    epochs = int(model_cfg.get("epochs", 1))
    if model_cfg.get("train", True) and epochs > 0:
        print(f"Training model for {epochs} epochs...")
        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, optimizer, device)
            # In log tại epoch đầu tiên, epoch cuối cùng, và sau mỗi 10 epochs
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss:.4f}")
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Thêm seed và timestamp vào tên file khi lưu để phân biệt các lần chạy
        save_path = checkpoint_path.parent / f"{checkpoint_path.stem}_{config.seed}_{timestamp}{checkpoint_path.suffix}"
        torch.save(model.state_dict(), save_path)
        print(f"[+] Model saved to {save_path}")

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

    metrics_file = output_dir / f"metrics_{config.seed}_{timestamp}.json"
    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\n[+] Đã lưu báo cáo thành công vào: {metrics_file}")
    print("[+] KẾT QUẢ ĐÁNH GIÁ (METRICS):")
    print(json.dumps(metrics, indent=2))

    return metrics
