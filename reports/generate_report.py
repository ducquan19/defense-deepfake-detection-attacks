"""Report generator: aggregate experiment results into a Markdown report.

Reads JSON files from experiment run directories and generates a unified
Markdown report including:
- Baseline clean metrics
- Attack evaluation table (ACC, F1, AUC, ASR per attack × epsilon)
- Baseline vs Robust comparison table

Usage:
    python reports/generate_report.py --runs-dir reports/runs --output reports/final_report.md
    python reports/generate_report.py --comparison reports/runs/adversarial_training/comparison_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Markdown research report from run artifacts.")
    parser.add_argument(
        "--runs-dir",
        default="reports/runs",
        help="Root directory of experiment run folders.",
    )
    parser.add_argument(
        "--output",
        default="reports/final_report.md",
        help="Output Markdown file path.",
    )
    parser.add_argument(
        "--comparison",
        default=None,
        help="Direct path to a comparison_report.json to include in the report.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def fmt(val: float | None) -> str:
    if val is None or val != val:  # NaN check
        return "-"
    return f"{val:.4f}"


def build_attack_table(attack_report: dict) -> str:
    """Build Markdown table from attack_report.json."""
    if not attack_report:
        return "_No attack report found._\n"

    headers = ["Protocol", "ACC", "Precision", "Recall", "F1", "AUC", "ASR"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for proto, metrics in attack_report.items():
        row = [
            proto,
            fmt(metrics.get("accuracy")),
            fmt(metrics.get("precision")),
            fmt(metrics.get("recall")),
            fmt(metrics.get("f1")),
            fmt(metrics.get("roc_auc")),
            fmt(metrics.get("asr")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def build_comparison_table(comparison: dict) -> str:
    """Build Markdown comparison table from comparison_report.json."""
    baseline = comparison.get("baseline", {})
    robust = comparison.get("robust", {})
    protocols = list(baseline.keys())
    metric_keys = ["accuracy", "f1", "roc_auc", "asr"]

    lines = [
        "| Protocol | Metric | Baseline | Robust | Δ |",
        "|----------|--------|----------|--------|---|",
    ]
    for proto in protocols:
        b = baseline.get(proto, {})
        r = robust.get(proto, {})
        for m in metric_keys:
            if m in b or m in r:
                bv = b.get(m)
                rv = r.get(m)
                if bv is not None and rv is not None:
                    delta = rv - bv
                    sign = "+" if delta >= 0 else ""
                    lines.append(
                        f"| {proto} | {m} | {fmt(bv)} | {fmt(rv)} | {sign}{delta:.4f} |"
                    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []

    # Header
    sections.append("# Deepfake Detection — Adversarial Attack & Defense Report\n")
    sections.append(
        "> Auto-generated from experiment artifacts. "
        "Metrics computed on synthetic test set.\n"
    )

    # -------------------------------------------------------------------
    # Section 1: Clean baseline metrics
    # -------------------------------------------------------------------
    sections.append("## 1. Baseline Clean Evaluation\n")
    clean_path = runs_dir / "baseline" / "clean_metrics.json"
    clean = load_json(clean_path)
    if clean:
        rows = "\n".join(f"- **{k}**: {fmt(v)}" for k, v in clean.items())
        sections.append(rows + "\n")
    else:
        # Try attack_eval run
        clean_path2 = runs_dir / "attack_eval" / "clean_metrics.json"
        clean2 = load_json(clean_path2)
        if clean2:
            rows = "\n".join(f"- **{k}**: {fmt(v)}" for k, v in clean2.items())
            sections.append(rows + "\n")
        else:
            sections.append("_Run baseline pipeline first: `python scripts/run_pipeline.py`_\n")

    # -------------------------------------------------------------------
    # Section 2: Attack evaluation
    # -------------------------------------------------------------------
    sections.append("## 2. Multi-Attack Evaluation\n")
    attack_report_path = runs_dir / "attack_eval" / "attack_report.json"
    attack_report = load_json(attack_report_path)
    if attack_report:
        sections.append(build_attack_table(attack_report))
    else:
        sections.append("_Run attack evaluation: `python scripts/run_attack_eval.py`_\n")

    # -------------------------------------------------------------------
    # Section 3: Preprocessing Defense
    # -------------------------------------------------------------------
    sections.append("## 3. Preprocessing Defense (JPEG Smoothing)\n")
    defended_path = runs_dir / "attack_eval" / "defended_jpeg_metrics.json"
    defended = load_json(defended_path)
    if defended:
        rows = "\n".join(f"- **{k}**: {fmt(v)}" for k, v in defended.items())
        sections.append(rows + "\n")
    else:
        sections.append("_No preprocessing defense metrics found._\n")

    # -------------------------------------------------------------------
    # Section 4: Adversarial Training — Baseline vs Robust Comparison
    # -------------------------------------------------------------------
    sections.append("## 4. Adversarial Training — Baseline vs Robust Detector\n")

    comparison: dict | None = None
    if args.comparison:
        comparison = load_json(Path(args.comparison))
    if comparison is None:
        comparison = load_json(runs_dir / "adversarial_training" / "comparison_report.json")
    if comparison is None:
        comparison = load_json(runs_dir / "robustness_eval" / "robustness_comparison.json")

    if comparison:
        sections.append(build_comparison_table(comparison))
    else:
        sections.append(
            "_Run adversarial training: `python scripts/run_adversarial_training.py`_\n"
        )

    # -------------------------------------------------------------------
    # Section 5: Methodology notes
    # -------------------------------------------------------------------
    sections.append("## 5. Methodology\n")
    sections.append(
        "| Component | Details |\n"
        "|-----------|--------|\n"
        "| Baseline Detector | TinyCNN (configurable to DINO-MAC) |\n"
        "| Attack: FGSM | ε ∈ {0.001, 0.003, 0.005, 0.01, 0.03} |\n"
        "| Attack: I-FGSM | ε = 0.03, steps = 10, α = ε/steps |\n"
        "| Attack: PGD | ε = 0.01, α = 0.001, steps = 20, random start |\n"
        "| Defense: Preprocessing | JPEG re-encoding, quality = 75 |\n"
        "| Defense: Adv. Training | PGD-AT, L = L_clean + λ·L_adv, λ = 1.0 |\n"
        "| ASR Definition | Fraction of correctly-classified samples fooled by attack |\n"
    )

    # Write output
    report = "\n".join(sections)
    output_path.write_text(report, encoding="utf-8")
    print(f"[+] Report written → {output_path}")
    print(f"    {output_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
