"""Entry point for adversarial training (Stage 6) + robustness evaluation (Stage 7).

Usage:
    uv run scripts/run_adversarial_training.py
    uv run scripts/run_adversarial_training.py --config configs/experiment_adversarial_training.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config
from pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarial training pipeline: harden a deepfake detector via AT."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_adversarial_training.yaml",
        help="Path to the YAML experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"[+] Running adversarial training with config: {config_path}")
    metrics = run_experiment(config)
    print(f"\n[+] Done. Robust checkpoint and comparison report saved.")


if __name__ == "__main__":
    main()
