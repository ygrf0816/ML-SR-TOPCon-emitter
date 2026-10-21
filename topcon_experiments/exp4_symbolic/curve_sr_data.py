"""Build curve SR feature matrix (no PySR import)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.log_utils import log
from topcon_experiments.config import (
    ATHENA_FEATURES,
    LOG_FEATURES,
    MODEL2_FEATURES,
    OUTPUT_ROOT,
    RANDOM_STATE,
    SR_MAX_FILES,
    SR_MAX_POINTS_PER_FILE,
    SR_MAX_TOTAL_POINTS,
    SR_TAIL_USE_ALL_POINTS,
)
from topcon_experiments.exp4_symbolic.curve_sampling import adaptive_sample_indices, fit_sample_weights
from topcon_experiments.exp4_symbolic.preprocess_curves import process_and_save_curves

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"

FeatureMode = str  # "athena" | "full"


def _feature_columns(feature_mode: FeatureMode) -> list[str]:
    if feature_mode == "full":
        return list(MODEL2_FEATURES) + ["depth_um"]
    return list(ATHENA_FEATURES) + ["depth_um"]


def _resolve_feature_row(meta: pd.DataFrame, fb: str, feature_mode: FeatureMode) -> pd.Series | None:
    fb_key = fb
    if fb_key not in meta.index:
        lower_map = {str(i).lower(): i for i in meta.index}
        fb_key = lower_map.get(str(fb).lower())
        if fb_key is None:
            return None
    cols = MODEL2_FEATURES if feature_mode == "full" else ATHENA_FEATURES
    row = meta.loc[fb_key, cols].astype(float).copy()
    for col in LOG_FEATURES:
        if col in row.index and row[col] > 0:
            row[col] = float(np.log(row[col]))
    return row


def build_sr_dataset(
    curve_type: str,
    *,
    processed_suffix: str = "",
    feature_mode: FeatureMode = "athena",
    use_all_points: bool | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return X, y (ln conc), groups (file_base), sample_weights."""
    suffix = processed_suffix or ""
    if use_all_points is None:
        use_all_points = bool(suffix == "_tail" and SR_TAIL_USE_ALL_POINTS)
    curve_csv = EXP4_OUT / f"curve_processed_{curve_type}{suffix}.csv"
    if not curve_csv.exists():
        if suffix:
            raise FileNotFoundError(f"Processed curve CSV not found: {curve_csv}")
        log(f"Missing {curve_csv.name}, running curve preprocess ...")
        curve_df = process_and_save_curves(curve_type)
    else:
        curve_df = pd.read_csv(curve_csv)

    meta = load_raw_dataframe().set_index("file_base")
    feature_names = _feature_columns(feature_mode)
    feat_cols = MODEL2_FEATURES if feature_mode == "full" else ATHENA_FEATURES
    X_rows: list[list[float]] = []
    y_vals: list[float] = []
    groups: list[str] = []
    weights: list[float] = []
    rng = np.random.default_rng(RANDOM_STATE)

    file_bases = curve_df["file_base"].unique()[:SR_MAX_FILES]
    max_total = SR_MAX_FILES * 300 if use_all_points else SR_MAX_TOTAL_POINTS
    for fb in file_bases:
        if len(X_rows) >= max_total:
            break
        feat_row = _resolve_feature_row(meta, str(fb), feature_mode)
        if feat_row is None:
            continue

        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um").reset_index(drop=True)
        if len(sub) == 0:
            continue

        depths = sub["depth_um"].values.astype(float)
        conc = sub["value_fitted"].values.astype(float)
        if use_all_points:
            k = min(len(sub), max_total - len(X_rows))
            idx = np.arange(k, dtype=int)
            w_pts = np.ones(k, dtype=float)
        else:
            k = min(SR_MAX_POINTS_PER_FILE, len(sub), max_total - len(X_rows))
            idx = adaptive_sample_indices(depths, conc, k, rng, curve_type=curve_type)
            w_pts = fit_sample_weights(depths, conc, idx, curve_type=curve_type)

        for j, ridx in enumerate(idx):
            r = sub.iloc[ridx]
            y = float(r["value_fitted"])
            if y <= 0:
                continue
            x = [float(feat_row[c]) for c in feat_cols] + [float(r["depth_um"])]
            X_rows.append(x)
            y_vals.append(float(np.log(y)))
            groups.append(str(fb))
            weights.append(float(w_pts[j]))

    X = pd.DataFrame(X_rows, columns=feature_names)
    y = np.array(y_vals, dtype=float)
    g = np.array(groups)
    w = np.array(weights, dtype=float)
    return X, y, g, w


def _load_curve_df(curve_type: str) -> pd.DataFrame:
    curve_csv = EXP4_OUT / f"curve_processed_{curve_type}.csv"
    if not curve_csv.exists():
        log(f"Missing {curve_csv.name}, running curve preprocess ...")
        return process_and_save_curves(curve_type)
    return pd.read_csv(curve_csv)


def _resolve_athena_row(meta: pd.DataFrame, fb: str) -> pd.Series | None:
    return _resolve_feature_row(meta, fb, "athena")


def build_sr_defect_mean_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One row per sample: Athena -> mean(ln conc) over defect curve."""
    curve_df = _load_curve_df("defect")
    meta = load_raw_dataframe().set_index("file_base")

    X_rows: list[list[float]] = []
    y_vals: list[float] = []
    groups: list[str] = []

    for fb in curve_df["file_base"].unique()[:SR_MAX_FILES]:
        athena_row = _resolve_athena_row(meta, str(fb))
        if athena_row is None:
            continue
        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um")
        conc = sub["value_fitted"].values.astype(float)
        conc = conc[conc > 0]
        if len(conc) == 0:
            continue
        X_rows.append([float(athena_row[c]) for c in ATHENA_FEATURES])
        y_vals.append(float(np.mean(np.log(conc))))
        groups.append(str(fb))

    X = pd.DataFrame(X_rows, columns=list(ATHENA_FEATURES))
    return X, np.array(y_vals, dtype=float), np.array(groups)


def build_sr_defect_shape_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Per depth point: Athena + depth -> ln(c) - mean(ln c) within sample."""
    curve_df = _load_curve_df("defect")
    meta = load_raw_dataframe().set_index("file_base")
    feature_names = ATHENA_FEATURES + ["depth_um"]

    X_rows: list[list[float]] = []
    y_vals: list[float] = []
    groups: list[str] = []
    weights: list[float] = []
    rng = np.random.default_rng(RANDOM_STATE)

    file_bases = curve_df["file_base"].unique()[:SR_MAX_FILES]
    for fb in file_bases:
        if len(X_rows) >= SR_MAX_TOTAL_POINTS:
            break
        athena_row = _resolve_athena_row(meta, str(fb))
        if athena_row is None:
            continue

        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um").reset_index(drop=True)
        if len(sub) == 0:
            continue

        conc = sub["value_fitted"].values.astype(float)
        valid = conc > 0
        if valid.sum() < 2:
            continue
        mean_ln = float(np.mean(np.log(conc[valid])))

        depths = sub["depth_um"].values.astype(float)
        k = min(SR_MAX_POINTS_PER_FILE, len(sub), SR_MAX_TOTAL_POINTS - len(X_rows))
        idx = adaptive_sample_indices(depths, conc, k, rng, curve_type="defect")
        w_pts = fit_sample_weights(depths, conc, idx, curve_type="defect")

        for j, ridx in enumerate(idx):
            y = float(conc[ridx])
            if y <= 0:
                continue
            x = [float(athena_row[c]) for c in ATHENA_FEATURES] + [float(sub.iloc[ridx]["depth_um"])]
            X_rows.append(x)
            y_vals.append(float(np.log(y) - mean_ln))
            groups.append(str(fb))
            weights.append(float(w_pts[j]))

    X = pd.DataFrame(X_rows, columns=feature_names)
    return X, np.array(y_vals, dtype=float), np.array(groups), np.array(weights, dtype=float)
