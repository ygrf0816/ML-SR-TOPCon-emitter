"""Defect curve SR variants: scheme A (mean + shape decomposition)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import OUTPUT_ROOT
from topcon_experiments.exp4_symbolic.curve_sr_data import (
    build_sr_defect_mean_dataset,
    build_sr_defect_shape_dataset,
)

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"


def _warmup_julia() -> None:
    """Avoid cold-start libjulia load failure on some import orders."""
    from topcon_experiments.exp4_symbolic.sr_eval_utils import run_pysr_tabular_eval  # noqa: F401


def _sort_best(eq_df):
    from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy
    return sort_formulas_by_accuracy(eq_df).iloc[0]


def main() -> None:
    setup_runtime()
    EXP4_OUT.mkdir(parents=True, exist_ok=True)
    log("Defect SR scheme A: mean(ln c) tabular + shape residual curve SR")
    _warmup_julia()
    from topcon_experiments.exp4_symbolic.sr_eval_utils import run_pysr_curve_eval, run_pysr_tabular_eval

    # --- mean level: Athena -> mean(ln conc) ---
    log_step(1, 2, "PySR defect mean ln(conc)")
    X_mean, y_mean, g_mean = build_sr_defect_mean_dataset()
    if len(X_mean) < 20:
        log(f"Skip defect mean: only {len(X_mean)} samples")
    else:
        eq_mean = run_pysr_tabular_eval(
            X_mean, y_mean, "defect_mean_ln", "defect_mean_tabular",
        )
        eq_mean["target"] = "defect_mean_ln"
        eq_mean["preprocess_note"] = (
            "方案 A 水平项：X = Athena（athena_c_boron 取 ln）；"
            "y = 每条缺陷曲线 fitted 浓度的 **mean(ln c)**（每样本 1 点）。"
        )
        out_mean = EXP4_OUT / "sr_formulas_defect_mean.csv"
        save_csv(eq_mean, out_mean)
        best = _sort_best(eq_mean)
        log(
            f"Saved {len(eq_mean)} equations -> {out_mean} "
            f"(best test_R2={best.get('test_R2', float('nan')):.4f})"
        )

    # --- shape residual: Athena + depth -> delta ln ---
    log_step(2, 2, "PySR defect shape residual")
    X_shape, y_shape, g_shape, w_shape = build_sr_defect_shape_dataset()
    if len(X_shape) < 50:
        log(f"Skip defect shape: only {len(X_shape)} points")
    else:
        eq_shape = run_pysr_curve_eval(
            X_shape, y_shape, g_shape, "defect_shape", sample_weights=w_shape,
        )
        eq_shape["target"] = "defect_shape_ln"
        eq_shape["preprocess_note"] = (
            "方案 A 形状项：X = Athena + depth_um；"
            "y = **ln(c) - mean(ln c)**（样本内去均值残差）。"
            "推理时 ln(c)_pred = mean_pred(Athena) + shape_pred(Athena, depth)。"
        )
        out_shape = EXP4_OUT / "sr_formulas_defect_shape.csv"
        save_csv(eq_shape, out_shape)
        best = _sort_best(eq_shape)
        log(
            f"Saved {len(eq_shape)} equations -> {out_shape} "
            f"(best test_MSE={best.get('test_MSE', float('nan')):.4g}, "
            f"test_R2={best.get('test_R2', float('nan')):.4f})"
        )


if __name__ == "__main__":
    main()
