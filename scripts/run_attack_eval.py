"""Entry point for multi-attack evaluation (Stage 2–5).

Usage:
    uv run scripts/run_attack_eval.py
    uv run scripts/run_attack_eval.py --config configs/experiment_attack_eval.yaml
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
        description="Multi-attack evaluation pipeline for deepfake detector robustness."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_attack_eval.yaml",
        help="Path to the YAML experiment config (default: configs/experiment_attack_eval.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"[+] Running attack evaluation with config: {config_path}")
    metrics = run_experiment(config)
    print(f"\n[+] Done. {len(metrics)} metric keys collected.")


if __name__ == "__main__":
    main()
