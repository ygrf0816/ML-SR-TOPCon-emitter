"""Evaluate classifier and export confusion matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_meta, preprocess_classifier
from topcon_experiments.common.metrics import compute_classification_metrics, metrics_to_dataframe
from topcon_experiments.common.plot_utils import plot_confusion_matrix, save_csv
from topcon_experiments.config import CLASSIFIER_LABEL, OUTPUT_ROOT


EXP3_OUT = OUTPUT_ROOT / "exp3_classifier"
CLASSIFIER_META = EXP3_OUT / "classifier_meta.json"


def main() -> None:
    meta = load_meta(CLASSIFIER_META)
    X, y = preprocess_classifier()
    X_test = X.loc[meta["test_indices"]]
    y_test = y.loc[meta["test_indices"]].values

    predictor = TabularPredictor.load(meta["model_dir"])
    y_pred = predictor.predict(X_test).values.astype(int)
    y_prob = predictor.predict_proba(X_test)[1].values

    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    metrics_df = metrics_to_dataframe([{"split": "test", **metrics}])
    save_csv(metrics_df, EXP3_OUT / "classification_metrics.csv")
    print(metrics_df)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plot_confusion_matrix(cm, ["low", "high"], EXP3_OUT / "confusion_matrix")


if __name__ == "__main__":
    main()
