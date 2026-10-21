"""Data loading and preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from topcon_experiments.config import (
    ATHENA_FEATURES,
    CLASSIFIER_LABEL,
    CURVE_DESCRIPTORS,
    DATA_CSV,
    DOPING_DESCRIPTORS,
    IV_TARGETS,
    LOG_FEATURES,
    LOG_TARGETS_MODEL1,
    META_JSON,
    MODEL2_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


def load_raw_dataframe() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV)


def _positive_mask(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col in df.columns:
            mask &= df[col] > 0
    return mask


def preprocess_model1() -> tuple[pd.DataFrame, dict[str, pd.Series], pd.DataFrame]:
    """Model1: athena features -> curve descriptors."""
    df = load_raw_dataframe()
    positive_cols = list(set(LOG_FEATURES + LOG_TARGETS_MODEL1))
    df = df.loc[_positive_mask(df, positive_cols)].copy()

    X = df[ATHENA_FEATURES].copy()
    for col in LOG_FEATURES:
        if col in X.columns:
            X[col] = np.log(X[col].astype(float))

    y: dict[str, pd.Series] = {}
    for target in CURVE_DESCRIPTORS:
        if target in LOG_TARGETS_MODEL1:
            y[target] = np.log(df[target].astype(float))
        else:
            y[target] = df[target].astype(float).copy()

    meta_df = df[["file_base"]].copy() if "file_base" in df.columns else pd.DataFrame(index=df.index)
    return X, y, meta_df


def preprocess_model2() -> tuple[pd.DataFrame, dict[str, pd.Series], pd.DataFrame]:
    """Model2: athena + curve descriptors -> IV targets."""
    df = load_raw_dataframe()
    positive_cols = list(set(LOG_FEATURES))
    df = df.loc[_positive_mask(df, positive_cols)].copy()

    X = df[MODEL2_FEATURES].copy()
    for col in LOG_FEATURES:
        if col in X.columns:
            X[col] = np.log(X[col].astype(float))

    y = {t: df[t].astype(float).copy() for t in IV_TARGETS}
    meta_df = df[["file_base"]].copy() if "file_base" in df.columns else pd.DataFrame(index=df.index)
    return X, y, meta_df


def preprocess_classifier() -> tuple[pd.DataFrame, pd.Series]:
    """Binary classifier: high_efficiency based on iv_Eff median."""
    X, _, _ = preprocess_model2()
    df = load_raw_dataframe()
    df = df.loc[X.index].copy()
    threshold = df["iv_Eff"].median()
    y = (df["iv_Eff"] >= threshold).astype(int)
    y.name = CLASSIFIER_LABEL
    return X, y


def train_test_split_xy(
    X: pd.DataFrame,
    y_map: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series], dict[str, pd.Series]]:
    first_target = next(iter(y_map))
    X_train, X_test, y_first_train, y_first_test = train_test_split(
        X,
        y_map[first_target],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    y_train = {first_target: y_first_train}
    y_test = {first_target: y_first_test}
    for name, series in y_map.items():
        if name == first_target:
            continue
        y_train[name] = series.loc[X_train.index]
        y_test[name] = series.loc[X_test.index]
    return X_train, X_test, y_train, y_test


def train_test_split_classifier(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )


def inverse_transform_target(target: str, values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if target in LOG_TARGETS_MODEL1:
        return np.exp(arr)
    return arr


def get_athena_bounds(df: pd.DataFrame | None = None, q_low: float = 0.05, q_high: float = 0.95) -> list[tuple[float, float]]:
    if df is None:
        df = load_raw_dataframe()
    bounds = []
    for col in ATHENA_FEATURES:
        lo = float(df[col].quantile(q_low))
        hi = float(df[col].quantile(q_high))
        if lo >= hi:
            lo = float(df[col].min())
            hi = float(df[col].max())
        bounds.append((lo, hi))
    return bounds


def save_meta(meta: dict[str, Any], path: Path | None = None) -> None:
    out = path or META_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta(path: Path | None = None) -> dict[str, Any]:
    out = path or META_JSON
    with open(out, "r", encoding="utf-8") as f:
        return json.load(f)
