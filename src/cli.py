from __future__ import annotations

import argparse

from config import load_config
from pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deepfake defense research pipeline.")
    parser.add_argument("--config", default="configs/experiment_baseline.yaml", help="Path to YAML config.")
    parser.add_argument("--train-dir", type=str, help="Path to train data directory")
    parser.add_argument("--test-dir", type=str, help="Path to test data directory")
    parser.add_argument("--output-dir", type=str, help="Path to output directory")
    parser.add_argument("--checkpoint-path", type=str, help="Path to pre-trained checkpoint to load")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    
    # Override config with CLI arguments if provided
    if args.train_dir:
        config.values["data"]["train_dir"] = args.train_dir
    if args.test_dir:
        config.values["data"]["test_dir"] = args.test_dir
    if args.output_dir:
        config.values["experiment"]["output_dir"] = args.output_dir
    if args.checkpoint_path:
        config.values["model"]["checkpoint_path"] = args.checkpoint_path
        config.values["model"]["load_checkpoint"] = True

    run_experiment(config)


if __name__ == "__main__":
    main()
