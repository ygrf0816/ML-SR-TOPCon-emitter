"""Train 4-class efficiency tier classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import preprocess_model2, save_meta
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.inverse_utils import get_efficiency_tiers
from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.config import (
    AUTOGLUON_PRESET,
    OUTPUT_ROOT,
    RANDOM_STATE,
    TEST_SIZE,
)

EXP3_OUT = OUTPUT_ROOT / "exp3_classifier"
TIER_MODEL_DIR = EXP3_OUT / "models_tier"
TIER_META = EXP3_OUT / "tier_classifier_meta.json"
TIER_LABEL_COL = "eff_tier"


def assign_tier_labels(df: pd.DataFrame) -> pd.Series:
    tiers = get_efficiency_tiers(df)
    labels = pd.Series(index=df.index, dtype=str)
    for t in tiers:
        labels.loc[t["mask"]] = t["label"]
    return labels


def main() -> None:
    setup_runtime()
    EXP3_OUT.mkdir(parents=True, exist_ok=True)
    log("Training 4-class efficiency tier classifier...")
    raw = load_raw_dataframe()
    X, _, _ = preprocess_model2()
    raw = raw.loc[X.index]
    y = assign_tier_labels(raw)
    y.name = TIER_LABEL_COL
    valid = y.notna()
    if not valid.all():
        dropped = int((~valid).sum())
        log(f"Dropping {dropped} samples below lowest tier (22%~23%)")
        X = X.loc[valid]
        raw = raw.loc[valid]
        y = y.loc[valid]

    tier_info = get_efficiency_tiers(raw)
    tier_df = pd.DataFrame([{k: v for k, v in t.items() if k != "mask"} for t in tier_info])
    tier_df.to_csv(EXP3_OUT / "efficiency_tier_definition.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    predictor = TabularPredictor(
        label=TIER_LABEL_COL,
        path=str(TIER_MODEL_DIR),
        problem_type="multiclass",
    )
    predictor.fit(train_df, presets=AUTOGLUON_PRESET)

    meta = {
        "label": TIER_LABEL_COL,
        "feature_columns": list(X.columns),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "tiers": [{k: v for k, v in t.items() if k != "mask"} for t in tier_info],
        "train_indices": X_train.index.tolist(),
        "test_indices": X_test.index.tolist(),
        "model_dir": str(TIER_MODEL_DIR),
    }
    save_meta(meta, TIER_META)
    print(f"Tier classifier saved to {TIER_MODEL_DIR}")
    print(tier_df)


if __name__ == "__main__":
    main()
