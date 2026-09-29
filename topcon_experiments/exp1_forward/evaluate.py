"""Evaluate Model1 and Model2 on train/test splits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from autogluon.tabular import TabularPredictor

from topcon_experiments.common.autogluon_utils import load_predictor, safe_predict
from topcon_experiments.common.data import (
    inverse_transform_target,
    load_meta,
    preprocess_model1,
    preprocess_model2,
)
from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.metrics import compute_regression_metrics, metrics_to_dataframe
from topcon_experiments.config import CURVE_DESCRIPTORS, IV_TARGETS, LOG_TARGETS_MODEL1, META_JSON, OUTPUT_ROOT


EXP1_OUT = OUTPUT_ROOT / "exp1_forward"


def prioritize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "model",
        "target",
        "target_label",
        "split",
        "RMSE_primary",
        "MAE_primary",
        "error_scale",
        "MAPE_pct",
        "nRMSE_pct",
        "MdAPE_pct",
        "RMSE",
        "MAE",
        "R2",
        "MAPE",
        "RMSE_log",
        "MAE_log",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def evaluate_model(
    model_name: str,
    model_dirs: dict[str, str],
    X: pd.DataFrame,
    y_map: dict[str, pd.Series],
    train_idx: list,
    test_idx: list,
    log_targets: list[str] | None = None,
) -> pd.DataFrame:
    log_targets = log_targets or []
    rows = []
    targets = list(y_map.keys())
    for ti, (target, y_series) in enumerate(y_map.items(), start=1):
        log_step(ti, len(targets), f"{model_name} evaluate {target}")
        predictor = load_predictor(model_dirs[target])
        for split_name, idx in [("train", train_idx), ("test", test_idx)]:
            X_split = X.loc[idx]
            y_true_raw = y_series.loc[idx].values
            y_pred_model = safe_predict(predictor, X_split)

            if target in log_targets:
                y_true = inverse_transform_target(target, y_true_raw)
                y_pred = inverse_transform_target(target, y_pred_model)
            else:
                y_true = y_true_raw
                y_pred = y_pred_model

            m = compute_regression_metrics(y_true, y_pred, target=target)
            rows.append({
                "model": model_name,
                "target": target,
                "target_label": __import__(
                    "topcon_experiments.common.variable_labels", fromlist=["label_for"]
                ).label_for(target),
                "split": split_name,
                **m,
            })
    return prioritize_metric_columns(metrics_to_dataframe(rows))


def main() -> None:
    setup_runtime()
    EXP1_OUT.mkdir(parents=True, exist_ok=True)
    meta = load_meta(META_JSON)
    log("Evaluating model1 and model2...")

    X1, y1, _ = preprocess_model1()
    metrics1 = evaluate_model(
        "model1",
        meta["model1_dirs"],
        X1,
        y1,
        meta["train_indices_model1"],
        meta["test_indices_model1"],
        log_targets=LOG_TARGETS_MODEL1,
    )
    metrics1.to_csv(EXP1_OUT / "metrics_model1.csv", index=False)
    print(metrics1)

    X2, y2, _ = preprocess_model2()
    metrics2 = evaluate_model(
        "model2",
        meta["model2_dirs"],
        X2,
        y2,
        meta["train_indices_model2"],
        meta["test_indices_model2"],
    )
    metrics2.to_csv(EXP1_OUT / "metrics_model2.csv", index=False)
    print(metrics2)


if __name__ == "__main__":
    main()
