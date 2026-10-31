"""COPY of the forward chain pieces the R1-4 rerun needs.

Assembled from `topcon_experiments/common/{forward,data,autogluon_utils}.py`
with one change only: the config import is redirected to the local `config.py`
next to this file, so the patched feature list is the one in effect.
Logic is otherwise copied verbatim.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from config import (
    ATHENA_FEATURES,
    CURVE_DESCRIPTORS,
    IV_TARGETS,
    LOG_FEATURES,
    LOG_TARGETS_MODEL1,
    META_JSON,
    MODEL2_FEATURES,
)

warnings.filterwarnings("ignore", message=".*insecure pickle module.*")


# ---- from common/autogluon_utils.py --------------------------------------- #
def _to_numpy(pred) -> np.ndarray:
    return pred.values if isinstance(pred, pd.Series) else np.asarray(pred)


def safe_predict(predictor: TabularPredictor, X: pd.DataFrame) -> np.ndarray:
    """Predict with the best model; fall back if the CatBoost ensemble fails."""
    try:
        return _to_numpy(predictor.predict(X))
    except Exception:
        pass
    lb = predictor.leaderboard(silent=True)
    skip = ("WeightedEnsemble", "Stacker", "CatBoost")
    for name in lb["model"]:
        if any(name.startswith(p) for p in skip):
            continue
        try:
            return _to_numpy(predictor.predict(X, model=name))
        except Exception:
            continue
    for name in lb["model"]:
        try:
            return _to_numpy(predictor.predict(X, model=name))
        except Exception:
            continue
    raise RuntimeError("All AutoGluon models failed to predict")


# ---- from common/data.py -------------------------------------------------- #
def load_meta(path: Path | None = None) -> dict:
    return json.loads(Path(path or META_JSON).read_text(encoding="utf-8"))


def load_raw_dataframe(csv: Path) -> pd.DataFrame:
    return pd.read_csv(csv)


def get_athena_bounds(df: pd.DataFrame, q_low: float = 0.05,
                      q_high: float = 0.95) -> list[tuple[float, float]]:
    out = []
    for col in ATHENA_FEATURES:
        lo, hi = float(df[col].quantile(q_low)), float(df[col].quantile(q_high))
        if lo >= hi:
            lo, hi = float(df[col].min()), float(df[col].max())
        out.append((lo, hi))
    return out


def inverse_transform_target(target: str, values):
    arr = np.asarray(values, dtype=float)
    return np.exp(arr) if target in LOG_TARGETS_MODEL1 else arr


# ---- from common/forward.py ----------------------------------------------- #
class ForwardChain:
    def __init__(self, meta_path: Path | None = None):
        self.meta = load_meta(meta_path)
        self.model1_dirs: dict[str, str] = self.meta["model1_dirs"]
        self.model2_dirs: dict[str, str] = self.meta["model2_dirs"]
        self._p1: dict[str, TabularPredictor] = {}
        self._p2: dict[str, TabularPredictor] = {}

    def _get1(self, t: str) -> TabularPredictor:
        if t not in self._p1:
            self._p1[t] = TabularPredictor.load(self.model1_dirs[t])
        return self._p1[t]

    def _get2(self, t: str) -> TabularPredictor:
        if t not in self._p2:
            self._p2[t] = TabularPredictor.load(self.model2_dirs[t])
        return self._p2[t]

    def _prepare_athena_frame(self, athena_params) -> pd.DataFrame:
        if isinstance(athena_params, np.ndarray):
            row = {c: float(athena_params[i]) for i, c in enumerate(ATHENA_FEATURES)}
        elif isinstance(athena_params, pd.Series):
            row = {c: float(athena_params[c]) for c in ATHENA_FEATURES}
        else:
            row = {c: float(athena_params[c]) for c in ATHENA_FEATURES}
        df = pd.DataFrame([row])
        for c in LOG_FEATURES:
            if c in df.columns:
                df[c] = np.log(df[c].astype(float))
        return df

    def predict_curve_descriptors(self, athena_params) -> dict[str, float]:
        X = self._prepare_athena_frame(athena_params)
        out: dict[str, float] = {}
        for t in CURVE_DESCRIPTORS:
            v = float(safe_predict(self._get1(t), X)[0])
            out[t] = float(inverse_transform_target(t, v))
        return out

    def _prepare_model2_frame(self, athena_params, curve_desc=None) -> pd.DataFrame:
        if curve_desc is None:
            curve_desc = self.predict_curve_descriptors(athena_params)
        if isinstance(athena_params, np.ndarray):
            row = {c: float(athena_params[i]) for i, c in enumerate(ATHENA_FEATURES)}
        elif isinstance(athena_params, pd.Series):
            row = {c: float(athena_params[c]) for c in ATHENA_FEATURES}
        else:
            row = {c: float(athena_params[c]) for c in ATHENA_FEATURES}
        df = pd.DataFrame([{**row, **curve_desc}])
        for c in LOG_FEATURES:
            if c in df.columns:
                df[c] = np.log(df[c].astype(float))
        return df[MODEL2_FEATURES]

    def predict_iv_single(self, athena_params, metric: str, curve_desc=None) -> float:
        if metric not in IV_TARGETS:
            raise ValueError(f"Unknown IV metric: {metric}")
        return float(safe_predict(self._get2(metric),
                                  self._prepare_model2_frame(athena_params, curve_desc))[0])

    def predict_iv(self, athena_params, curve_desc=None) -> dict[str, float]:
        X = self._prepare_model2_frame(athena_params, curve_desc)
        return {t: float(safe_predict(self._get2(t), X)[0]) for t in IV_TARGETS}

    def predict_all(self, athena_params) -> dict[str, float]:
        curve = self.predict_curve_descriptors(athena_params)
        return {**curve, **self.predict_iv(athena_params, curve)}

    # ---- vectorised helpers (used by the batch scoring / retraining) ------ #
    def predict_curve_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X[ATHENA_FEATURES].copy()
        for c in LOG_FEATURES:
            if c in df.columns:
                df[c] = np.log(df[c].astype(float))
        out = {}
        for t in CURVE_DESCRIPTORS:
            v = safe_predict(self._get1(t), df)
            out[t] = np.exp(v) if t in LOG_TARGETS_MODEL1 else v
        return pd.DataFrame(out, index=X.index)

    def predict_iv_frame(self, X: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
        df = pd.concat([X[ATHENA_FEATURES].reset_index(drop=True),
                        D[CURVE_DESCRIPTORS].reset_index(drop=True)], axis=1)
        for c in LOG_FEATURES:
            if c in df.columns:
                df[c] = np.log(df[c].astype(float))
        return pd.DataFrame({t: safe_predict(self._get2(t), df[MODEL2_FEATURES])
                             for t in IV_TARGETS})


def load_forward_chain(meta_path: Path | None = None) -> ForwardChain:
    return ForwardChain(meta_path)
