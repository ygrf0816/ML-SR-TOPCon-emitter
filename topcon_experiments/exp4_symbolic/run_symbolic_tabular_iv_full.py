"""PySR tabular SR for IV targets using athena + doping + defect descriptors (model2 features)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import preprocess_model2
from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import IV_TARGETS, LOG_FEATURES, OUTPUT_ROOT
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy
from topcon_experiments.exp4_symbolic.sr_eval_utils import run_pysr_tabular_eval

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"


def main() -> None:
    setup_runtime()
    EXP4_OUT.mkdir(parents=True, exist_ok=True)

    X, y_map, _ = preprocess_model2()
    log(
        f"SR tabular IV-full: {len(X)} rows, {len(X.columns)} features, "
        f"ln on {sorted(LOG_FEATURES)}, train/test eval enabled"
    )

    all_eq = []
    for i, target in enumerate(IV_TARGETS, start=1):
        log_step(i, len(IV_TARGETS), f"PySR full -> {target}")
        y = y_map[target].values.astype(float)
        task = f"full_to_{target}"
        eq_df = run_pysr_tabular_eval(X, y, task, "model2_iv_full")
        out = EXP4_OUT / f"sr_tabular_{task}.csv"
        save_csv(eq_df, out)
        best = sort_formulas_by_accuracy(eq_df).iloc[0]
        log(
            f"  best test_MSE={best.get('test_MSE', float('nan')):.6g} "
            f"test_MAE={best.get('test_MAE', float('nan')):.6g} "
            f"test_R2={best['test_R2']:.4f} loss={best['loss']:.6g} -> {out.name}"
        )
        all_eq.append(eq_df.head(5))

    if all_eq:
        import pandas as pd

        merged = pd.concat(all_eq, ignore_index=True)
        save_csv(merged, EXP4_OUT / "sr_tabular_full_iv_summary.csv")
    log(f"IV-full tabular SR done -> {EXP4_OUT}")


if __name__ == "__main__":
    main()
