from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    values: dict[str, Any]
    path: Path | None = None

    @property
    def seed(self) -> int:
        return int(self.values.get("experiment", {}).get("seed", 42))

    @property
    def output_dir(self) -> Path:
        return Path(self.values.get("experiment", {}).get("output_dir", "reports/runs/default"))


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    return ExperimentConfig(values=values, path=config_path)
