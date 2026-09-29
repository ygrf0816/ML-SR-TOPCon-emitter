"""AutoGluon prediction helpers with CatBoost fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from topcon_experiments.common.log_utils import setup_runtime

setup_runtime()


def load_predictor(path: str) -> TabularPredictor:
    return TabularPredictor.load(path)


def _to_numpy(pred) -> np.ndarray:
    if isinstance(pred, pd.Series):
        return pred.values
    return np.asarray(pred)


def safe_predict(predictor: TabularPredictor, X: pd.DataFrame) -> np.ndarray:
    """Predict using best model; fall back if CatBoost ensemble fails."""
    try:
        return _to_numpy(predictor.predict(X))
    except Exception:
        pass

    lb = predictor.leaderboard(silent=True)
    skip_prefixes = ("WeightedEnsemble", "Stacker", "CatBoost")
    for model_name in lb["model"]:
        if any(model_name.startswith(p) for p in skip_prefixes):
            continue
        try:
            return _to_numpy(predictor.predict(X, model=model_name))
        except Exception:
            continue

    for model_name in lb["model"]:
        try:
            return _to_numpy(predictor.predict(X, model=model_name))
        except Exception:
            continue
    raise RuntimeError("All AutoGluon models failed to predict")


def safe_feature_importance(predictor: TabularPredictor, X: pd.DataFrame, label: str | None = None) -> pd.DataFrame:
    label = label or predictor.label
    data = X.copy()
    if label not in data.columns:
        raise ValueError(f"feature_importance requires label column '{label}' in data")
    try:
        imp = predictor.feature_importance(data)
        imp_df = imp.reset_index()
        imp_df.columns = ["feature", "importance"] + list(imp_df.columns[2:])
        return imp_df[["feature", "importance"]]
    except Exception:
        lb = predictor.leaderboard(silent=True)
        for model_name in lb["model"]:
            if model_name.startswith("CatBoost"):
                continue
            try:
                imp = predictor.feature_importance(data, model=model_name)
                imp_df = imp.reset_index()
                imp_df.columns = ["feature", "importance"] + list(imp_df.columns[2:])
                return imp_df[["feature", "importance"]]
            except Exception:
                continue
        raise
