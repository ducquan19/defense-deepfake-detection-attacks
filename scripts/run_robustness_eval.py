"""Entry point for robustness evaluation — compares baseline vs robust detector.

Loads a pre-saved robust checkpoint and evaluates both models
(baseline + robust) against FGSM, I-FGSM, and PGD attacks, then
generates a Markdown comparison table and JSON report.

Usage:
    uv run scripts/run_robustness_eval.py --baseline-ckpt models/tiny_cnn.pt \
        --robust-ckpt checkpoints/robust_detector.pth \
        --config configs/experiment_adversarial_training.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src/ to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from attacks.fgsm import FGSMAttack
from attacks.ifgsm import IFGSMAttack
from attacks.pgd import PGDAttack
from config import load_config
from data.synthetic import build_synthetic_loaders
from evaluation.metrics import RobustnessEvaluator
from models.tiny_cnn import TinyCNN
from utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustness evaluation: baseline vs robust detector comparison."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_adversarial_training.yaml",
        help="Experiment config used for data and model settings.",
    )
    parser.add_argument(
        "--baseline-ckpt",
        default=None,
        help="Path to the baseline detector checkpoint (optional).",
    )
    parser.add_argument(
        "--robust-ckpt",
        default="checkpoints/robust_detector.pth",
        help="Path to the robust detector checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/runs/robustness_eval",
        help="Directory to write the comparison report.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="ε for attack comparisons.",
    )
    return parser.parse_args()


def load_model(model_cfg: dict, checkpoint_path: str | None, device: torch.device):
    """Load a detector from a checkpoint path."""
    if model_cfg.get("name") == "dino_mac":
        from models.dino_mac import DinoMACForDeepfakeDetection  # lazy: avoids HF download
        model = DinoMACForDeepfakeDetection(
            model_name=model_cfg.get(
                "huggingface_model_name", "facebook/dinov2-with-registers-large"
            ),
        ).to(device)
    else:
        model = TinyCNN().to(device)

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
            print(f"[+] Loaded checkpoint: {ckpt_path}")
        else:
            print(f"[!] Checkpoint not found: {ckpt_path}. Using random weights.")
    return model


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    seed_everything(config.seed)

    data_cfg = config.values.get("data", {})
    model_cfg = config.values.get("model", {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")

    # Build test loader
    _, test_loader = build_synthetic_loaders(
        num_samples=int(data_cfg.get("num_samples", 64)),
        image_size=int(data_cfg.get("image_size", 128)),
        batch_size=int(data_cfg.get("batch_size", 16)),
        train_split=float(data_cfg.get("train_split", 0.75)),
        seed=config.seed,
    )

    # Load models
    baseline_model = load_model(model_cfg, args.baseline_ckpt, device)
    robust_model = load_model(model_cfg, args.robust_ckpt, device)

    eps = args.epsilon
    comparison_attacks = [
        FGSMAttack(epsilon=eps),
        IFGSMAttack(epsilon=eps, steps=10),
        PGDAttack(epsilon=eps, alpha=eps / 10, steps=20),
    ]

    print("\n--- Baseline Detector Robustness ---")
    baseline_eval = RobustnessEvaluator(baseline_model, test_loader, device)
    baseline_report = baseline_eval.run(attacks=comparison_attacks)

    print("\n--- Robust Detector Robustness ---")
    robust_eval = RobustnessEvaluator(robust_model, test_loader, device)
    robust_report = robust_eval.run(attacks=comparison_attacks)

    # Save JSON report
    comparison = {"baseline": baseline_report, "robust": robust_report}
    report_path = output_dir / "robustness_comparison.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[+] JSON report → {report_path}")

    # Generate Markdown table
    md = _build_markdown_table(baseline_report, robust_report)
    md_path = output_dir / "robustness_table.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[+] Markdown table → {md_path}")
    print("\n" + md)


def _build_markdown_table(
    baseline: dict,
    robust: dict,
) -> str:
    protocols = list(baseline.keys())
    metric_keys = ["accuracy", "f1", "roc_auc", "asr"]

    lines = [
        "# Robustness Comparison: Baseline vs Robust Detector\n",
        "| Protocol | Metric | Baseline | Robust | Δ |",
        "|----------|--------|----------|--------|---|",
    ]
    for proto in protocols:
        b = baseline.get(proto, {})
        r = robust.get(proto, {})
        for m in metric_keys:
            if m in b or m in r:
                bv = b.get(m, float("nan"))
                rv = r.get(m, float("nan"))
                delta = rv - bv if not (bv != bv or rv != rv) else float("nan")
                sign = "+" if delta >= 0 else ""
                lines.append(
                    f"| {proto} | {m} | {bv:.4f} | {rv:.4f} | {sign}{delta:.4f} |"
                )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
