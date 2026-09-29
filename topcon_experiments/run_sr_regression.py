"""Run symbolic regression only (no LLM). User confirms formula accuracy before LLM."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PYTHON = sys.executable

STEPS = [
    ("exp4 curve preprocess", ROOT / "exp4_symbolic" / "preprocess_curves.py"),
    ("exp4 symbolic curves", ROOT / "exp4_symbolic" / "run_symbolic.py"),
    ("exp4 symbolic tabular", ROOT / "exp4_symbolic" / "run_symbolic_tabular.py"),
    ("exp4 symbolic tabular IV-full", ROOT / "exp4_symbolic" / "run_symbolic_tabular_iv_full.py"),
    ("exp4 FF alternatives", ROOT / "exp4_symbolic" / "evaluate_ff_alternatives.py"),
    ("exp4 FF from predicted IV", ROOT / "exp4_symbolic" / "evaluate_ff_from_predicted_iv.py"),
    ("exp4 defect SR variants (A)", ROOT / "exp4_symbolic" / "run_defect_sr_variants.py"),
    ("exp4 defect curve variant eval", ROOT / "exp4_symbolic" / "evaluate_defect_curve_variants.py"),
    ("exp4 SR accuracy plots", ROOT / "exp4_symbolic" / "plot_sr_accuracy.py"),
    ("exp4 SR report (no LLM)", ROOT / "exp4_symbolic" / "generate_sr_report.py"),
]


def _print_sr_config() -> None:
    from topcon_experiments.config import (
        SR_CURVE_GRAD_POWER_DOPING,
        SR_CURVE_UNIFORM_FRAC,
        SR_CURVE_UNIFORM_FRAC_DOPING,
        SR_MAX_FILES,
        SR_MAX_POINTS_PER_FILE,
        SR_MAX_TOTAL_POINTS,
        SR_MODEL_SELECTION,
        SR_PARSIMONY,
    )

    print("SR-only pipeline (no LLM):")
    print(f"  SR_MAX_FILES={SR_MAX_FILES}, SR_MAX_POINTS_PER_FILE={SR_MAX_POINTS_PER_FILE}, "
          f"SR_MAX_TOTAL_POINTS={SR_MAX_TOTAL_POINTS}")
    print(f"  SR_CURVE_UNIFORM_FRAC={SR_CURVE_UNIFORM_FRAC} (defect); "
          f"doping={SR_CURVE_UNIFORM_FRAC_DOPING}, grad_power={SR_CURVE_GRAD_POWER_DOPING}")
    print(f"  SR_MODEL_SELECTION={SR_MODEL_SELECTION}, SR_PARSIMONY={SR_PARSIMONY}")
    print("  Formula ranking: test_MSE -> test_MAE -> test_R2")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SR regression + plots + MD report (skip LLM).")
    parser.parse_args(argv)
    _print_sr_config()

    for name, script in STEPS:
        print(f"\n{'=' * 60}\nRunning: {name}\n{'=' * 60}")
        result = subprocess.run([PYTHON, str(script)], cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"FAILED: {name} (exit {result.returncode})")
            sys.exit(result.returncode)

    print("\nSR regression pipeline completed (no LLM).")
    print(f"Formulas: {ROOT / 'outputs' / 'exp4_symbolic'}/sr_formulas_*.csv, sr_tabular_*.csv")
    print(f"Plots:    {ROOT / 'outputs' / 'exp4_symbolic' / 'sr_plots'}/")
    print(f"Reports:  {ROOT / 'outputs' / 'exp4_symbolic' / 'sr_reports'}/")


if __name__ == "__main__":
    main()
