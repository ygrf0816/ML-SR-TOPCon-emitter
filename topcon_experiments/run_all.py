"""Run all TOPCon experiments in sequence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STEPS = [
    ("exp1 train", ROOT / "exp1_forward" / "train.py"),
    ("exp1 evaluate", ROOT / "exp1_forward" / "evaluate.py"),
    ("exp1 plot", ROOT / "exp1_forward" / "plot.py"),
    ("exp2 de single", ROOT / "exp2_inverse" / "de_single.py"),
    ("exp2 pareto", ROOT / "exp2_inverse" / "pareto_multi.py"),
    ("exp3 train tiers", ROOT / "exp3_classifier" / "train_tiers.py"),
    ("exp3 evaluate tiers", ROOT / "exp3_classifier" / "evaluate_tiers.py"),
    ("exp3 shap tiers", ROOT / "exp3_classifier" / "shap_analysis.py"),
    ("exp4 preprocess", ROOT / "exp4_symbolic" / "preprocess_curves.py"),
    ("exp4 symbolic curves", ROOT / "exp4_symbolic" / "run_symbolic.py"),
    ("exp4 symbolic tabular", ROOT / "exp4_symbolic" / "run_symbolic_tabular.py"),
    ("exp4 FF from predicted IV", ROOT / "exp4_symbolic" / "evaluate_ff_from_predicted_iv.py"),
    ("exp4 SR accuracy plots", ROOT / "exp4_symbolic" / "plot_sr_accuracy.py"),
    ("exp4 llm", ROOT / "exp4_symbolic" / "llm_analysis.py"),
    ("exp4 sr report", ROOT / "exp4_symbolic" / "generate_sr_report.py"),
    ("exp5 correlation", ROOT / "run_exp5_correlation.py"),
]


def main() -> None:
    for name, script in STEPS:
        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        if name in ("exp4 llm", "exp4 sr report"):
            from topcon_experiments.config import DEEPSEEK_API_KEY
            import os
            if not os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY") and not DEEPSEEK_API_KEY:
                print("Skipping LLM analysis (no API key)")
                continue
        result = subprocess.run([PYTHON, str(script)], cwd=str(ROOT.parent))
        if result.returncode != 0:
            print(f"FAILED: {name} (exit {result.returncode})")
            sys.exit(result.returncode)
    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
