"""PySR + LLM + scatter plot for athena -> doping_R_sheet (front emitter sheet resistance)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.autogluon_utils import safe_predict
from topcon_experiments.common.data import (
    load_meta,
    preprocess_model1,
    save_meta,
    train_test_split_xy,
)
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.metrics import compute_regression_metrics, metrics_to_dataframe
from topcon_experiments.common.plot_utils import plot_regression_scatter, save_csv
from topcon_experiments.config import (
    AUTOGLUON_PRESET,
    META_JSON,
    MODEL1_DIR,
    OUTPUT_ROOT,
)
from topcon_experiments.exp1_forward.evaluate import prioritize_metric_columns
from topcon_experiments.exp4_symbolic.llm_analysis import (
    analyze_formula_rows,
    check_api_connectivity,
    create_client,
    load_top_formulas,
    tabular_task_type,
)
from topcon_experiments.exp4_symbolic.plot_sr_accuracy import (
    _metrics,
    _pred_to_display,
    _to_display_scale,
    best_row_by_test_r2,
    predict_from_row,
)
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy
from topcon_experiments.exp4_symbolic.sr_eval_utils import run_pysr_tabular_eval
from topcon_experiments.exp4_symbolic.sr_split_utils import split_train_test

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
EXP1_OUT = OUTPUT_ROOT / "exp1_forward"
TASK = "athena_to_doping_R_sheet"
TARGET = "doping_R_sheet"
PLOT_OUT = EXP4_OUT / "sr_plots"


def run_sr() -> pd.DataFrame:
    X1, y1_map, _ = preprocess_model1()
    y = y1_map[TARGET].values.astype(float)
    log(f"PySR tabular: {TASK} (n={len(X1)})")
    eq_df = run_pysr_tabular_eval(X1, y, TASK, "model1_tabular")
    save_csv(eq_df, EXP4_OUT / f"sr_tabular_{TASK}.csv")
    best = sort_formulas_by_accuracy(eq_df).iloc[0]
    log(
        f"  best test_MSE={best.get('test_MSE', float('nan')):.6g} "
        f"test_MAE={best.get('test_MAE', float('nan')):.6g} "
        f"test_R2={best['test_R2']:.4f}"
    )
    return eq_df


def run_llm() -> None:
    path = EXP4_OUT / f"sr_tabular_{TASK}.csv"
    df = load_top_formulas(path)
    if df.empty:
        log(f"Skip LLM: no formulas at {path}")
        return
    try:
        check_api_connectivity()
    except RuntimeError as exc:
        log(f"Skip LLM (API unreachable): {exc}")
        return
    client = create_client()
    ttype = tabular_task_type(TASK)
    log(f"LLM tabular analysis: {TASK} ({ttype})")
    analyze_formula_rows(client, df, TASK, ttype, TASK, tabular=True)


def train_autogluon_r_sheet() -> None:
    """Train model1 predictor for doping_R_sheet and patch metrics_model1.csv."""
    X1, y1, _ = preprocess_model1()
    X1_train, _, y1_train, _ = train_test_split_xy(X1, y1)
    train_df = pd.concat(
        [X1_train.reset_index(drop=True), y1_train[TARGET].reset_index(drop=True)],
        axis=1,
    )
    save_path = MODEL1_DIR / f"autogluon_{TARGET}"
    log(f"AutoGluon model1 -> {TARGET}")
    predictor = TabularPredictor(label=TARGET, path=str(save_path))
    predictor.fit(train_df, presets=AUTOGLUON_PRESET)

    meta = load_meta(META_JSON)
    meta.setdefault("model1_dirs", {})[TARGET] = str(save_path)
    meta.setdefault("model1_targets", [])
    if TARGET not in meta["model1_targets"]:
        meta["model1_targets"].append(TARGET)
    save_meta(meta, META_JSON)

    train_idx = meta["train_indices_model1"]
    test_idx = meta["test_indices_model1"]
    rows = []
    for split_name, idx in [("train", train_idx), ("test", test_idx)]:
        X_split = X1.loc[idx]
        y_true = y1[TARGET].loc[idx].values
        y_pred = safe_predict(predictor, X_split)
        m = compute_regression_metrics(y_true, y_pred, target=TARGET)
        rows.append({
            "model": "model1",
            "target": TARGET,
            "target_label": "Front emitter sheet resistance (ohm/sq) [doping_R_sheet]",
            "split": split_name,
            **m,
        })
    new_rows = prioritize_metric_columns(metrics_to_dataframe(rows))
    metrics_path = EXP1_OUT / "metrics_model1.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        old = old[old["target"] != TARGET]
        merged = pd.concat([old, new_rows], ignore_index=True)
    else:
        merged = new_rows
    merged.to_csv(metrics_path, index=False)
    log(f"Updated {metrics_path}")


def plot_scatter() -> Path | None:
    path = EXP4_OUT / f"sr_tabular_{TASK}.csv"
    if not path.exists():
        return None
    X, y_map, _ = preprocess_model1()
    y = y_map[TARGET].values.astype(float)
    _, test_idx = split_train_test(len(X))
    row = best_row_by_test_r2(path)
    pred_all = predict_from_row(row, X.values.astype(float))
    y_true = _to_display_scale(TARGET, y[test_idx], log_target=False)
    y_pred = _pred_to_display(TARGET, pred_all[test_idx], log_target=False)
    m = _metrics(y_true, y_pred)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    prefix = PLOT_OUT / f"regression_scatter_sr_{TASK}"
    plot_regression_scatter(y_true, y_pred, TARGET, prefix, log_scale=False)
    summary = {
        "task": TASK,
        "target": TARGET,
        "split": "test",
        "test_R2_csv": float(row.get("test_R2", np.nan)),
        **m,
        "equation": str(row.get("equation", ""))[:200],
    }
    save_csv(pd.DataFrame([summary]), PLOT_OUT / f"regression_scatter_sr_{TASK}_metrics.csv")
    log(f"Scatter test_R2={m['R2']:.4f} -> {prefix}.png")
    return prefix.with_suffix(".png")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PySR + LLM + scatter for doping_R_sheet")
    parser.add_argument("--skip-sr", action="store_true", help="Skip PySR (reuse existing CSV)")
    args = parser.parse_args(argv)

    setup_runtime()
    EXP4_OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_sr:
        run_sr()
    run_llm()
    try:
        train_autogluon_r_sheet()
    except Exception as exc:
        log(f"AutoGluon train skipped: {exc}")
    plot_scatter()
    log(f"Done -> {EXP4_OUT}")


if __name__ == "__main__":
    main()
