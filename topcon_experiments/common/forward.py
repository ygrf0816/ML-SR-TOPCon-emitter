"""Model1 + Model2 forward prediction chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from topcon_experiments.common.autogluon_utils import safe_predict
from topcon_experiments.common.log_utils import log, log_step

from topcon_experiments.common.data import inverse_transform_target, load_meta
from topcon_experiments.config import (
    ATHENA_FEATURES,
    CURVE_DESCRIPTORS,
    IV_TARGETS,
    LOG_FEATURES,
    LOG_TARGETS_MODEL1,
    META_JSON,
    MODEL1_DIR,
    MODEL2_DIR,
)


class ForwardChain:
    def __init__(self, meta_path: Path | None = None):
        self.meta = load_meta(meta_path)
        self.model1_dirs: dict[str, str] = self.meta["model1_dirs"]
        self.model2_dirs: dict[str, str] = self.meta["model2_dirs"]
        self._predictors1: dict[str, TabularPredictor] = {}
        self._predictors2: dict[str, TabularPredictor] = {}
        self._preload_models()

    def _preload_models(self) -> None:
        all_targets = list(CURVE_DESCRIPTORS) + list(IV_TARGETS)
        for i, target in enumerate(all_targets, start=1):
            log_step(i, len(all_targets), f"preload forward model {target}")
            if target in CURVE_DESCRIPTORS:
                self._get_predictor1(target)
            else:
                self._get_predictor2(target)
        log(f"Preloaded {len(all_targets)} forward models.")

    def _get_predictor1(self, target: str) -> TabularPredictor:
        if target not in self._predictors1:
            self._predictors1[target] = TabularPredictor.load(self.model1_dirs[target])
        return self._predictors1[target]

    def _get_predictor2(self, target: str) -> TabularPredictor:
        if target not in self._predictors2:
            self._predictors2[target] = TabularPredictor.load(self.model2_dirs[target])
        return self._predictors2[target]

    def _prepare_athena_frame(self, athena_params: dict[str, float] | pd.Series | np.ndarray) -> pd.DataFrame:
        if isinstance(athena_params, np.ndarray):
            if athena_params.shape[0] != len(ATHENA_FEATURES):
                raise ValueError(f"Expected {len(ATHENA_FEATURES)} athena params, got {athena_params.shape[0]}")
            row = {col: float(athena_params[i]) for i, col in enumerate(ATHENA_FEATURES)}
        elif isinstance(athena_params, pd.Series):
            row = {col: float(athena_params[col]) for col in ATHENA_FEATURES}
        else:
            row = {col: float(athena_params[col]) for col in ATHENA_FEATURES}

        df = pd.DataFrame([row])
        for col in LOG_FEATURES:
            if col in df.columns and col in ATHENA_FEATURES:
                df[col] = np.log(df[col].astype(float))
        return df

    def predict_curve_descriptors(self, athena_params: dict[str, float] | pd.Series | np.ndarray) -> dict[str, float]:
        X = self._prepare_athena_frame(athena_params)
        out: dict[str, float] = {}
        for target in CURVE_DESCRIPTORS:
            pred = float(safe_predict(self._get_predictor1(target), X)[0])
            out[target] = float(inverse_transform_target(target, pred))
        return out

    def _prepare_model2_frame(
        self,
        athena_params: dict[str, float] | pd.Series | np.ndarray,
        curve_desc: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        if curve_desc is None:
            curve_desc = self.predict_curve_descriptors(athena_params)

        if isinstance(athena_params, np.ndarray):
            athena_row = {col: float(athena_params[i]) for i, col in enumerate(ATHENA_FEATURES)}
        elif isinstance(athena_params, pd.Series):
            athena_row = {col: float(athena_params[col]) for col in ATHENA_FEATURES}
        else:
            athena_row = {col: float(athena_params[col]) for col in ATHENA_FEATURES}

        row = {**athena_row, **curve_desc}
        df = pd.DataFrame([row])
        for col in LOG_FEATURES:
            if col in df.columns:
                df[col] = np.log(df[col].astype(float))
        return df[ATHENA_FEATURES + CURVE_DESCRIPTORS]

    def predict_iv_single(
        self,
        athena_params: dict[str, float] | pd.Series | np.ndarray,
        metric: str,
        curve_desc: dict[str, float] | None = None,
    ) -> float:
        """Predict one IV target (still runs model1 chain once)."""
        if metric not in IV_TARGETS:
            raise ValueError(f"Unknown IV metric: {metric}")
        X = self._prepare_model2_frame(athena_params, curve_desc)
        return float(safe_predict(self._get_predictor2(metric), X)[0])

    def predict_iv(
        self,
        athena_params: dict[str, float] | pd.Series | np.ndarray,
        curve_desc: dict[str, float] | None = None,
    ) -> dict[str, float]:
        X = self._prepare_model2_frame(athena_params, curve_desc)
        out: dict[str, float] = {}
        for target in IV_TARGETS:
            out[target] = float(safe_predict(self._get_predictor2(target), X)[0])
        return out

    def predict_all(
        self,
        athena_params: dict[str, float] | pd.Series | np.ndarray,
    ) -> dict[str, float]:
        curve = self.predict_curve_descriptors(athena_params)
        iv = self.predict_iv(athena_params, curve)
        return {**curve, **iv}


def load_forward_chain(meta_path: Path | None = None) -> ForwardChain:
    return ForwardChain(meta_path=meta_path)
