from __future__ import annotations

import argparse

from config import load_config
from pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deepfake defense research pipeline.")
    parser.add_argument("--config", default="configs/experiment_baseline.yaml", help="Path to YAML config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
