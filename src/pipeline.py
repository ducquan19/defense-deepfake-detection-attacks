"""Full adversarial attack & defense pipeline.

Stages:
    1. Data loading & model setup
    2. Baseline (clean) evaluation
    3. Multi-attack evaluation  (FGSM × epsilons, I-FGSM, PGD)
    4. Adversarial dataset generation  (save adversarial images + metadata)
    5. Preprocessing defense evaluation (JPEG smoothing)
    6. Adversarial training  (optional, config flag)
    7. Robustness evaluation  (baseline vs robust model comparison)
    8. Final report  (JSON + Markdown comparison table)
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from attacks import ATTACK_REGISTRY
from config import ExperimentConfig
from data.synthetic import build_synthetic_loaders
from defenses.adversarial_training import AdversarialTrainer
from defenses.preprocessing import JPEGSmoothingDefense, jpeg_smoothing_batch
from evaluation.metrics import (
    RobustnessEvaluator,
    compute_full_metrics,
    evaluate_detector,
)
from generation.toy_generator import apply_toy_deepfake_batch
from models.tiny_cnn import TinyCNN
from training import train_one_epoch
from utils.reproducibility import seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _build_model(model_cfg: dict, device: torch.device):
    """Instantiate detector from config."""
    if model_cfg.get("name") == "dino_mac":
        from models.dino_mac import DinoMACForDeepfakeDetection  # lazy: avoids HF download at import
        model = DinoMACForDeepfakeDetection(
            model_name=model_cfg.get(
                "huggingface_model_name", "facebook/dinov2-with-registers-large"
            ),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
            use_stochastic_depth=model_cfg.get("use_stochastic_depth", True),
        ).to(device)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.backbone.parameters(),
                    "lr": float(model_cfg.get("backbone_lr", 1e-5)),
                    "weight_decay": float(model_cfg.get("weight_decay", 1e-4)),
                },
                {
                    "params": model.mac_head.parameters(),
                    "lr": float(model_cfg.get("head_lr", 1e-4)),
                    "weight_decay": float(model_cfg.get("weight_decay", 1e-4)),
                },
            ]
        )
    else:
        model = TinyCNN().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(model_cfg.get("learning_rate", 1e-3)),
        )
    return model, optimizer


def _load_or_train(
    model,
    optimizer,
    train_loader,
    model_cfg: dict,
    config: ExperimentConfig,
    timestamp: str,
    device: torch.device,
) -> None:
    """Load checkpoint or run standard training per config flags."""
    checkpoint_path = Path(model_cfg.get("checkpoint_path", "models/tiny_cnn.pt"))

    if model_cfg.get("load_checkpoint", False):
        if checkpoint_path.exists():
            print(f"[+] Loading checkpoint from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
        else:
            print(
                f"[!] WARNING: load_checkpoint=True but {checkpoint_path} not found. "
                "Running with random weights."
            )

    epochs = int(model_cfg.get("epochs", 1))
    if model_cfg.get("train", True) and epochs > 0:
        print(f"[+] Training model for {epochs} epoch(s)...")
        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, optimizer, device)
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"    Epoch [{epoch + 1}/{epochs}] Loss: {loss:.4f}")

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_path = (
            checkpoint_path.parent
            / f"{checkpoint_path.stem}_{config.seed}_{timestamp}{checkpoint_path.suffix}"
        )
        torch.save(model.state_dict(), save_path)
        print(f"[+] Baseline model saved → {save_path}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Execute the full attack & defense pipeline.

    Returns a dict with all collected metrics.
    """
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
    adv_train_cfg = config.values.get("adversarial_training", {})

    # ------------------------------------------------------------------
    # Stage 1: Data + Model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1 — Data Loading & Model Setup")
    print("=" * 60)

    train_loader, test_loader = build_synthetic_loaders(
        num_samples=int(data_cfg.get("num_samples", 64)),
        image_size=int(data_cfg.get("image_size", 128)),
        batch_size=int(data_cfg.get("batch_size", 16)),
        train_split=float(data_cfg.get("train_split", 0.75)),
        seed=config.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")

    baseline_model, optimizer = _build_model(model_cfg, device)
    _load_or_train(
        baseline_model, optimizer, train_loader,
        model_cfg, config, timestamp, device,
    )

    all_metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Stage 2: Clean (Baseline) Evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2 — Baseline Clean Evaluation")
    print("=" * 60)

    clean_metrics = evaluate_detector(baseline_model, test_loader, device)
    for k, v in clean_metrics.items():
        all_metrics[f"clean_{k}"] = v
    print(f"[+] Clean metrics: {json.dumps(clean_metrics, indent=2)}")
    _save_json(clean_metrics, output_dir / "clean_metrics.json")

    # ------------------------------------------------------------------
    # Stage 3: Multi-Attack Evaluation
    # ------------------------------------------------------------------
    if attack_cfg.get("enabled", True):
        print("\n" + "=" * 60)
        print("STAGE 3 — Multi-Attack Evaluation")
        print("=" * 60)

        # Build attack instances dynamically
        eval_attacks = attack_cfg.get("eval_attacks", [])
        attacks = []
        if not eval_attacks:
            # Fallback for old configs
            print("[!] No 'eval_attacks' list in config. Falling back to default PGD.")
            if "pgd" in ATTACK_REGISTRY:
                attacks.append(ATTACK_REGISTRY["pgd"](epsilon=0.01, alpha=0.001, steps=20))
        else:
            for atk_def in eval_attacks:
                name = atk_def.get("name")
                if name in ATTACK_REGISTRY:
                    kwargs = {k: v for k, v in atk_def.items() if k != "name"}
                    attacks.append(ATTACK_REGISTRY[name](**kwargs))
                else:
                    print(f"[!] Warning: Unknown attack '{name}' in config. Skipping.")

        evaluator = RobustnessEvaluator(baseline_model, test_loader, device)
        attack_report = evaluator.run(attacks=attacks)
        for protocol, metrics in attack_report.items():
            prefix = f"attack_{protocol}"
            for k, v in metrics.items():
                all_metrics[f"{prefix}_{k}"] = v

        _save_json(attack_report, output_dir / "attack_report.json")
        print(f"[+] Attack report saved → {output_dir / 'attack_report.json'}")

    # ------------------------------------------------------------------
    # Stage 4: Generation metadata (optional toy deepfake flag)
    # ------------------------------------------------------------------
    if generation_cfg.get("enabled", True):
        all_metrics["toy_generation_strength"] = float(
            generation_cfg.get("strength", 0.35)
        )

    # ------------------------------------------------------------------
    # Stage 5: Preprocessing Defense Evaluation
    # ------------------------------------------------------------------
    if defense_cfg.get("enabled", True) and defense_cfg.get("method") == "jpeg_smoothing":
        print("\n" + "=" * 60)
        print("STAGE 5 — Preprocessing Defense (JPEG Smoothing)")
        print("=" * 60)

        jpeg_quality = int(defense_cfg.get("jpeg_quality", 75))
        jpeg_defense = JPEGSmoothingDefense(quality=jpeg_quality)

        def _defended_transform(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            # Attack then defend
            from attacks.pgd import pgd_attack
            adv = pgd_attack(
                baseline_model, images, labels,
                epsilon=float(attack_cfg.get("pgd_epsilon", 0.01)),
                alpha=float(attack_cfg.get("pgd_alpha", 0.001)),
                steps=int(attack_cfg.get("pgd_steps", 20)),
            )
            return jpeg_defense.apply(adv)

        defended_metrics = evaluate_detector(
            baseline_model, test_loader, device,
            transform=_defended_transform,
        )
        for k, v in defended_metrics.items():
            all_metrics[f"defended_jpeg_{k}"] = v
        print(f"[+] Defended (JPEG) metrics: {json.dumps(defended_metrics, indent=2)}")
        _save_json(defended_metrics, output_dir / "defended_jpeg_metrics.json")

    # ------------------------------------------------------------------
    # Stage 6: Adversarial Training (optional)
    # ------------------------------------------------------------------
    robust_model = None
    if adv_train_cfg.get("enabled", False):
        print("\n" + "=" * 60)
        print("STAGE 6 — Adversarial Training")
        print("=" * 60)

        robust_model, robust_optimizer = _build_model(model_cfg, device)

        # Copy baseline weights as starting point (warm start)
        robust_model.load_state_dict(baseline_model.state_dict())

        at_epochs = int(adv_train_cfg.get("epochs", 5))
        at_epsilon = float(adv_train_cfg.get("epsilon", 0.01))
        at_alpha = float(adv_train_cfg.get("alpha", 0.001))
        at_steps = int(adv_train_cfg.get("steps", 10))
        at_lambda = float(adv_train_cfg.get("adv_lambda", 1.0))
        at_attack = adv_train_cfg.get("attack_fn", "pgd")
        checkpoint_dir = Path(adv_train_cfg.get("checkpoint_dir", "checkpoints"))

        trainer = AdversarialTrainer(
            model=robust_model,
            optimizer=robust_optimizer,
            device=device,
            attack_fn=at_attack,
            epsilon=at_epsilon,
            alpha=at_alpha,
            steps=at_steps,
            adv_lambda=at_lambda,
            checkpoint_dir=checkpoint_dir,
        )
        history = trainer.fit(train_loader, epochs=at_epochs)
        ckpt_path = trainer.save_checkpoint("robust_detector.pth")
        trainer.save_history()
        all_metrics["adversarial_training_epochs"] = at_epochs
        all_metrics["adversarial_training_checkpoint"] = str(ckpt_path)
        print(f"[+] Adversarial training done. Checkpoint → {ckpt_path}")

    # ------------------------------------------------------------------
    # Stage 7: Robustness Evaluation — Baseline vs Robust
    # ------------------------------------------------------------------
    if robust_model is not None:
        print("\n" + "=" * 60)
        print("STAGE 7 — Robustness Comparison: Baseline vs Robust Detector")
        print("=" * 60)

        # Re-use the same attacks defined for Stage 3, if available,
        # otherwise rebuild them.
        comparison_attacks = attacks if 'attacks' in locals() and attacks else []
        if not comparison_attacks:
            eval_attacks = attack_cfg.get("eval_attacks", [])
            for atk_def in eval_attacks:
                name = atk_def.get("name")
                if name in ATTACK_REGISTRY:
                    kwargs = {k: v for k, v in atk_def.items() if k != "name"}
                    comparison_attacks.append(ATTACK_REGISTRY[name](**kwargs))

        print("\n--- Baseline Detector ---")
        baseline_evaluator = RobustnessEvaluator(baseline_model, test_loader, device)
        baseline_robustness = baseline_evaluator.run(attacks=comparison_attacks)

        print("\n--- Robust Detector ---")
        robust_evaluator = RobustnessEvaluator(robust_model, test_loader, device)
        robust_robustness = robust_evaluator.run(attacks=comparison_attacks)

        comparison = {
            "baseline": baseline_robustness,
            "robust": robust_robustness,
        }
        _save_json(comparison, output_dir / "comparison_report.json")

        # Generate Markdown comparison table
        md_table = _generate_comparison_table(baseline_robustness, robust_robustness)
        (output_dir / "comparison_table.md").write_text(md_table, encoding="utf-8")
        print(f"\n[+] Comparison report → {output_dir / 'comparison_report.json'}")
        print(f"[+] Comparison table  → {output_dir / 'comparison_table.md'}")
        print("\n" + md_table)

        all_metrics["comparison"] = comparison

    # ------------------------------------------------------------------
    # Stage 8: Save final aggregated metrics
    # ------------------------------------------------------------------
    metrics_file = output_dir / f"metrics_{config.seed}_{timestamp}.json"
    _save_json(all_metrics, metrics_file)

    print("\n" + "=" * 60)
    print(f"[+] All results saved to: {output_dir}")
    print("=" * 60)

    return all_metrics


# ---------------------------------------------------------------------------
# Report generation helper
# ---------------------------------------------------------------------------

def _generate_comparison_table(
    baseline: dict[str, dict],
    robust: dict[str, dict],
) -> str:
    """Generate a Markdown comparison table from robustness report dicts."""
    protocols = list(baseline.keys())  # clean, fgsm, ifgsm, pgd, …
    metric_keys = ["accuracy", "f1", "roc_auc", "asr"]

    header = "| Protocol | Metric | Baseline | Robust |\n"
    header += "|----------|--------|----------|--------|\n"
    rows = []

    for proto in protocols:
        b = baseline.get(proto, {})
        r = robust.get(proto, {})
        for m in metric_keys:
            if m in b or m in r:
                bv = f"{b.get(m, float('nan')):.4f}"
                rv = f"{r.get(m, float('nan')):.4f}"
                rows.append(f"| {proto} | {m} | {bv} | {rv} |")

    return header + "\n".join(rows) + "\n"
