"""Evaluate 4-class efficiency tier classifier."""

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

from topcon_experiments.common.data import load_meta, load_raw_dataframe, preprocess_model2
from topcon_experiments.common.inverse_utils import get_efficiency_tiers
from topcon_experiments.common.metrics import compute_classification_metrics, metrics_to_dataframe
from topcon_experiments.common.plot_utils import plot_confusion_matrix, save_csv
from topcon_experiments.config import OUTPUT_ROOT
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.exp3_classifier.train_tiers import assign_tier_labels

EXP3_OUT = OUTPUT_ROOT / "exp3_classifier"
TIER_META = EXP3_OUT / "tier_classifier_meta.json"
TIER_LABEL_COL = "eff_tier"


def main() -> None:
    setup_runtime()
    meta = load_meta(TIER_META)
    X, _, _ = preprocess_model2()
    raw = load_raw_dataframe().loc[X.index]

    tiers = get_efficiency_tiers(raw)
    tier_df = pd.DataFrame([{k: v for k, v in t.items() if k != "mask"} for t in tiers])
    save_csv(tier_df, EXP3_OUT / "efficiency_tier_definition.csv")

    X_test = X.loc[meta["test_indices"]]
    y_all = assign_tier_labels(raw)
    y_test = y_all.loc[meta["test_indices"]]

    predictor = TabularPredictor.load(meta["model_dir"])
    y_pred = predictor.predict(X_test).astype(str).values

    metrics = compute_classification_metrics(
        pd.Categorical(y_test, categories=[t["label"] for t in tiers]).codes,
        pd.Categorical(y_pred, categories=[t["label"] for t in tiers]).codes,
    )
    metrics_df = metrics_to_dataframe([{"split": "test", **metrics}])
    save_csv(metrics_df, EXP3_OUT / "tier_classification_metrics.csv")
    print(metrics_df)

    labels = [t["label"] for t in tiers]
    cm = confusion_matrix(y_test.values, y_pred, labels=labels)
    plot_confusion_matrix(cm, labels, EXP3_OUT / "tier_confusion_matrix")
    print(f"Tier evaluation saved to {EXP3_OUT}")


if __name__ == "__main__":
    main()
