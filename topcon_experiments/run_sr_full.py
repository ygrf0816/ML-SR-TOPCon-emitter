"""Run full symbolic-regression pipeline: curves -> tabular -> LLM + MD report.

Usage (from repo root, autogluon env recommended):
    python -m topcon_experiments.run_sr_full

Steps (in order):
  1. exp4_symbolic.run_symbolic        — PySR on doping/defect curves
  2. exp4_symbolic.run_symbolic_tabular — PySR on all tabular targets
  3. exp4_symbolic.llm_analysis        — LLM blind-math/physics + symbolic_regression_report.md

Tuning: edit SR_* and DEEPSEEK_MODEL in topcon_experiments/config.py before running.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PYTHON = sys.executable

STEPS = [
    ("exp4 symbolic curves", ROOT / "exp4_symbolic" / "run_symbolic.py"),
    ("exp4 symbolic tabular", ROOT / "exp4_symbolic" / "run_symbolic_tabular.py"),
    ("exp4 symbolic tabular IV-full", ROOT / "exp4_symbolic" / "run_symbolic_tabular_iv_full.py"),
    ("exp4 FF alternatives", ROOT / "exp4_symbolic" / "evaluate_ff_alternatives.py"),
    ("exp4 FF from predicted IV", ROOT / "exp4_symbolic" / "evaluate_ff_from_predicted_iv.py"),
    ("exp4 SR accuracy plots", ROOT / "exp4_symbolic" / "plot_sr_accuracy.py"),
    ("exp4 llm + report", ROOT / "exp4_symbolic" / "llm_analysis.py"),
]


def _print_sr_config() -> None:
    from topcon_experiments.config import (
        DEEPSEEK_MODEL,
        SR_MAX_FILES,
        SR_MAX_TOTAL_POINTS,
        SR_MODEL_SELECTION,
        SR_NITERATIONS,
        SR_PARSIMONY,
        SR_POPULATIONS,
        SR_POPULATION_SIZE,
        SR_QUICK_VALIDATE,
        SR_TABULAR_MAX_ROWS,
        SR_TOP_FORMULAS_FOR_LLM,
        SR_TOP_FORMULAS_IN_REPORT,
    )

    print("Symbolic regression config:")
    print(f"  SR_NITERATIONS={SR_NITERATIONS}, SR_POPULATIONS={SR_POPULATIONS}, "
          f"SR_POPULATION_SIZE={SR_POPULATION_SIZE}")
    print(f"  SR_MAX_FILES={SR_MAX_FILES}, SR_MAX_TOTAL_POINTS={SR_MAX_TOTAL_POINTS}, "
          f"SR_TABULAR_MAX_ROWS={SR_TABULAR_MAX_ROWS}")
    print(f"  SR_MODEL_SELECTION={SR_MODEL_SELECTION}, SR_PARSIMONY={SR_PARSIMONY}")
    print(f"  SR_TOP_FORMULAS_FOR_LLM={SR_TOP_FORMULAS_FOR_LLM}, "
          f"SR_TOP_FORMULAS_IN_REPORT={SR_TOP_FORMULAS_IN_REPORT}")
    print(f"  SR_QUICK_VALIDATE={SR_QUICK_VALIDATE}")
    print(f"  DEEPSEEK_MODEL={DEEPSEEK_MODEL}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run full symbolic-regression pipeline (curves -> tabular -> LLM + MD report)."
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM analysis; run generate_sr_report.py instead.",
    )
    args = parser.parse_args(argv)

    _print_sr_config()

    for name, script in STEPS:
        if args.skip_llm and name == "exp4 llm + report":
            print(f"\n{'=' * 60}\nSkipping LLM; running generate_sr_report.py\n{'=' * 60}")
            script = ROOT / "exp4_symbolic" / "generate_sr_report.py"
            name = "exp4 SR report (no LLM)"
        print(f"\n{'=' * 60}\nRunning: {name}\n{'=' * 60}")
        if name == "exp4 llm + report" and not args.skip_llm:
            from topcon_experiments.config import DEEPSEEK_API_KEY

            if (
                not os.environ.get("DEEPSEEK_API_KEY")
                and not os.environ.get("OPENAI_API_KEY")
                and not DEEPSEEK_API_KEY
            ):
                print("Skipping LLM step (no API key). Running generate_sr_report.py instead.")
                script = ROOT / "exp4_symbolic" / "generate_sr_report.py"
                name = "exp4 SR report (no LLM)"
        result = subprocess.run([PYTHON, str(script)], cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"FAILED: {name} (exit {result.returncode})")
            sys.exit(result.returncode)

    print("\nSR full pipeline completed.")
    print(f"Report index: {ROOT / 'outputs' / 'exp4_symbolic' / 'symbolic_regression_report.md'}")
    print(f"Per-task reports: {ROOT / 'outputs' / 'exp4_symbolic' / 'sr_reports'}/")


if __name__ == "__main__":
    main()
