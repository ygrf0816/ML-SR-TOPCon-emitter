"""PySR symbolic regression for doping and defect curves."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import OUTPUT_ROOT, SR_MAX_FILES, SR_MAX_POINTS_PER_FILE, SR_MAX_TOTAL_POINTS
from topcon_experiments.exp4_symbolic.curve_sr_data import build_sr_dataset
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy
from topcon_experiments.exp4_symbolic.sr_eval_utils import run_pysr_curve_eval

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"


def main() -> None:
    setup_runtime()
    EXP4_OUT.mkdir(parents=True, exist_ok=True)
    log(
        f"SR curve config: max_files={SR_MAX_FILES}, max_pts/file={SR_MAX_POINTS_PER_FILE}, "
        f"max_total={SR_MAX_TOTAL_POINTS}, adaptive depth sampling + weighted MSE fit"
    )

    curve_types = ["doping", "defect"]
    for i, curve_type in enumerate(curve_types, start=1):
        log_step(i, len(curve_types), f"PySR curve {curve_type}")
        try:
            X, y, groups, weights = build_sr_dataset(curve_type)
        except FileNotFoundError as e:
            log(f"Skip {curve_type}: {e}")
            continue
        if len(X) < 50:
            log(f"Skip {curve_type}: insufficient samples ({len(X)})")
            continue
        eq_df = run_pysr_curve_eval(X, y, groups, curve_type, sample_weights=weights)
        out_path = EXP4_OUT / f"sr_formulas_{curve_type}.csv"
        save_csv(eq_df, out_path)
        best = sort_formulas_by_accuracy(eq_df).iloc[0]
        log(
            f"Saved {len(eq_df)} equations -> {out_path} "
            f"(best test_MSE={best.get('test_MSE', float('nan')):.6g}, "
            f"test_MAE={best.get('test_MAE', float('nan')):.6g}, "
            f"test_R2={best.get('test_R2', float('nan')):.4f})"
        )


if __name__ == "__main__":
    main()
