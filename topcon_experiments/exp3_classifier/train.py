"""Train binary classifier with AutoGluon (deprecated — use train_tiers.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import preprocess_classifier, save_meta, train_test_split_classifier
from topcon_experiments.config import AUTOGLUON_PRESET, CLASSIFIER_DIR, CLASSIFIER_LABEL, OUTPUT_ROOT, RANDOM_STATE, TEST_SIZE


EXP3_OUT = OUTPUT_ROOT / "exp3_classifier"
CLASSIFIER_META = EXP3_OUT / "classifier_meta.json"


def main() -> None:
    EXP3_OUT.mkdir(parents=True, exist_ok=True)
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)

    X, y = preprocess_classifier()
    X_train, X_test, y_train, y_test = train_test_split_classifier(X, y)

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    predictor = TabularPredictor(
        label=CLASSIFIER_LABEL,
        path=str(CLASSIFIER_DIR),
        problem_type="binary",
    )
    predictor.fit(train_df, presets=AUTOGLUON_PRESET)

    meta = {
        "label": CLASSIFIER_LABEL,
        "feature_columns": list(X.columns),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "threshold": "median_iv_Eff",
        "train_indices": X_train.index.tolist(),
        "test_indices": X_test.index.tolist(),
        "model_dir": str(CLASSIFIER_DIR),
    }
    save_meta(meta, CLASSIFIER_META)
    print(f"Classifier saved to {CLASSIFIER_DIR}")


if __name__ == "__main__":
    main()
