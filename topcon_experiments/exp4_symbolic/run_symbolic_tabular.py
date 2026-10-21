"""PySR symbolic regression for tabular targets: athena->descriptors and athena->IV."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import preprocess_model1, preprocess_model2
from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import (
    ATHENA_FEATURES,
    DEFECT_DESCRIPTORS,
    DOPING_DESCRIPTORS,
    IV_TARGETS,
    OUTPUT_ROOT,
    SR_QUICK_TABULAR_TARGETS,
    SR_QUICK_VALIDATE,
    SR_TABULAR_MAX_ROWS,
)
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy
from topcon_experiments.exp4_symbolic.sr_eval_utils import run_pysr_tabular_eval

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"


def main() -> None:
    setup_runtime()
    EXP4_OUT.mkdir(parents=True, exist_ok=True)
    log(f"SR tabular config: max_fit_rows={SR_TABULAR_MAX_ROWS}, train/test split aligned with exp1")

    all_eq = []
    doping_targets = list(DOPING_DESCRIPTORS)
    defect_targets = list(DEFECT_DESCRIPTORS)
    iv_targets = list(IV_TARGETS)
    if SR_QUICK_VALIDATE:
        doping_targets = [t for t in DOPING_DESCRIPTORS if t in SR_QUICK_TABULAR_TARGETS]
        defect_targets = [t for t in DEFECT_DESCRIPTORS if t in SR_QUICK_TABULAR_TARGETS]
        iv_targets = [t for t in IV_TARGETS if t in SR_QUICK_TABULAR_TARGETS]
        log(f"QUICK VALIDATE mode: doping={doping_targets}, defect={defect_targets}, iv={iv_targets}")

    # --- model1-aligned: athena -> doping + defect descriptors ---
    X1, y1_map, _ = preprocess_model1()
    all_tasks = (
        [(t, "doping") for t in doping_targets]
        + [(t, "defect") for t in defect_targets]
        + [(t, "iv") for t in iv_targets]
    )
    task_idx = 0
    for target in doping_targets:
        task_idx += 1
        log_step(task_idx, len(all_tasks), f"PySR tabular doping -> {target}")
        y = y1_map[target].values.astype(float)
        if target in ("doping_N_peak", "doping_dose"):
            task = f"athena_to_{target}_log"
        else:
            task = f"athena_to_{target}"
        eq_df = run_pysr_tabular_eval(X1, y, task, "model1_tabular")
        save_csv(eq_df, EXP4_OUT / f"sr_tabular_{task}.csv")
        best = sort_formulas_by_accuracy(eq_df).iloc[0]
        log(f"  best test_MSE={best.get('test_MSE', float('nan')):.6g} test_MAE={best.get('test_MAE', float('nan')):.6g} test_R2={best['test_R2']:.4f}")
        all_eq.append(eq_df.head(5))

    for target in defect_targets:
        task_idx += 1
        log_step(task_idx, len(all_tasks), f"PySR tabular defect -> {target}")
        y = y1_map[target].values.astype(float)
        if target in ("defect_vac_N_peak", "defect_vac_dose"):
            task = f"athena_to_{target}_log"
        else:
            task = f"athena_to_{target}"
        eq_df = run_pysr_tabular_eval(X1, y, task, "model1_tabular")
        save_csv(eq_df, EXP4_OUT / f"sr_tabular_{task}.csv")
        best = sort_formulas_by_accuracy(eq_df).iloc[0]
        log(f"  best test_MSE={best.get('test_MSE', float('nan')):.6g} test_MAE={best.get('test_MAE', float('nan')):.6g} test_R2={best['test_R2']:.4f}")
        all_eq.append(eq_df.head(5))

    # --- model2 sample filter, athena-only X -> IV ---
    X_full, y2_map, _ = preprocess_model2()
    X_athena = X_full[ATHENA_FEATURES].copy()
    for target in iv_targets:
        task_idx += 1
        log_step(task_idx, len(all_tasks), f"PySR tabular iv -> {target}")
        y = y2_map[target].values.astype(float)
        task = f"athena_to_{target}"
        eq_df = run_pysr_tabular_eval(X_athena, y, task, "model2_iv_athena")
        save_csv(eq_df, EXP4_OUT / f"sr_tabular_{task}.csv")
        best = sort_formulas_by_accuracy(eq_df).iloc[0]
        log(f"  best test_MSE={best.get('test_MSE', float('nan')):.6g} test_MAE={best.get('test_MAE', float('nan')):.6g} test_R2={best['test_R2']:.4f}")
        all_eq.append(eq_df.head(5))

    if all_eq:
        merged = __import__("pandas").concat(all_eq, ignore_index=True)
        save_csv(merged, EXP4_OUT / "sr_tabular_formulas_summary.csv")
    log(f"Tabular SR done -> {EXP4_OUT}")


if __name__ == "__main__":
    main()
