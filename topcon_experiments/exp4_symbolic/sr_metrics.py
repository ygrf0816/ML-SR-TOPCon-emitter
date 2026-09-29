"""Regression metrics for SR formula ranking (MSE/MAE primary, R2 secondary)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {"MSE": float("nan"), "RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
    yt, yp = y_true[mask], y_pred[mask]
    mse = float(mean_squared_error(yt, yp))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(yt, yp)),
        "R2": float(r2_score(yt, yp)),
    }


def annotate_equation_metrics(
    model,
    eq_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    out = eq_df.copy()
    cols = ["train_MSE", "train_RMSE", "train_MAE", "train_R2", "test_MSE", "test_RMSE", "test_MAE", "test_R2"]
    for c in cols:
        out[c] = np.nan

    for i in range(len(out)):
        try:
            pred_tr = model.predict(X_train, index=i)
            pred_te = model.predict(X_test, index=i)
            tr = regression_metrics(y_train, pred_tr)
            te = regression_metrics(y_test, pred_te)
            out.at[i, "train_MSE"] = tr["MSE"]
            out.at[i, "train_RMSE"] = tr["RMSE"]
            out.at[i, "train_MAE"] = tr["MAE"]
            out.at[i, "train_R2"] = tr["R2"]
            out.at[i, "test_MSE"] = te["MSE"]
            out.at[i, "test_RMSE"] = te["RMSE"]
            out.at[i, "test_MAE"] = te["MAE"]
            out.at[i, "test_R2"] = te["R2"]
        except Exception:
            continue
    out["dataset_R2"] = out["train_R2"]
    return out


def sort_formulas_by_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Rank by test_MSE (asc), test_MAE (asc), test_R2 (desc). R2 is secondary."""
    out = df.copy()
    if "test_MSE" in out.columns and out["test_MSE"].notna().any():
        sort_cols = ["test_MSE"]
        ascending = [True]
        if "test_MAE" in out.columns:
            sort_cols.append("test_MAE")
            ascending.append(True)
        if "test_R2" in out.columns:
            sort_cols.append("test_R2")
            ascending.append(False)
        if "loss" in out.columns:
            sort_cols.append("loss")
            ascending.append(True)
        return out.sort_values(sort_cols, ascending=ascending, na_position="last")
    if "test_R2" in out.columns and out["test_R2"].notna().any():
        return out.sort_values(["test_R2", "loss"], ascending=[False, True], na_position="last")
    if "loss" in out.columns:
        return out.sort_values("loss")
    return out
