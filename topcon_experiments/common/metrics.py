"""Regression and classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from topcon_experiments.common.variable_labels import LARGE_MAGNITUDE_TARGETS
from topcon_experiments.config import LOG_TARGETS_MODEL1


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0 if target in LOG_TARGETS_MODEL1 else True)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan"), "MAPE": float("nan")}

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    denom = np.maximum(np.abs(y_true), 1e-12)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    out = {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

    out["MAPE_pct"] = mape
    out["MdAPE_pct"] = float(np.median(np.abs((y_true - y_pred) / denom)) * 100.0)
    out["nRMSE_pct"] = float(rmse / np.mean(y_true) * 100.0) if float(np.mean(y_true)) != 0 else float("nan")

    # Log-trained targets: use log-space RMSE/MAE as primary.
    if target and target in LOG_TARGETS_MODEL1:
        yt = np.maximum(y_true, 1e-300)
        yp = np.maximum(y_pred, 1e-300)
        log_true = np.log(yt)
        log_pred = np.log(yp)
        out["RMSE_log"] = float(np.sqrt(mean_squared_error(log_true, log_pred)))
        out["MAE_log"] = float(mean_absolute_error(log_true, log_pred))
        out["RMSE_primary"] = out["RMSE_log"]
        out["MAE_primary"] = out["MAE_log"]
        out["error_scale"] = "log_natural"
    # Large-magnitude, non-log targets: use relative metrics as primary.
    elif target and target in LARGE_MAGNITUDE_TARGETS:
        out["RMSE_primary"] = out["nRMSE_pct"]
        out["MAE_primary"] = out["MdAPE_pct"]
        out["error_scale"] = "relative_pct"
    else:
        out["RMSE_primary"] = rmse
        out["MAE_primary"] = mae
        out["error_scale"] = "original"

    return out


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            pass
    return metrics


def metrics_to_dataframe(rows: list[dict]):
    import pandas as pd
    return pd.DataFrame(rows)
