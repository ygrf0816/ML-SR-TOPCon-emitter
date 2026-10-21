"""Train AutoGluon Model1 and Model2."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import (
    preprocess_model1,
    preprocess_model2,
    save_meta,
    train_test_split_xy,
)
from topcon_experiments.config import (
    AUTOGLUON_PRESET,
    CURVE_DESCRIPTORS,
    IV_TARGETS,
    META_JSON,
    MODEL1_DIR,
    MODEL2_DIR,
    OUTPUT_ROOT,
    RANDOM_STATE,
    TEST_SIZE,
)


def train_targets(
    X_train: pd.DataFrame,
    y_train: dict[str, pd.Series],
    model_dir: Path,
) -> dict[str, str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    dirs: dict[str, str] = {}
    for target, series in y_train.items():
        train_df = pd.concat([X_train.reset_index(drop=True), series.reset_index(drop=True)], axis=1)
        train_df = train_df.rename(columns={series.name: target})
        save_path = model_dir / f"autogluon_{target}"
        predictor = TabularPredictor(label=target, path=str(save_path))
        predictor.fit(train_df, presets=AUTOGLUON_PRESET)
        dirs[target] = str(save_path)
        print(f"Trained {target} -> {save_path}")
    return dirs


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Model1
    X1, y1, _ = preprocess_model1()
    X1_train, X1_test, y1_train, y1_test = train_test_split_xy(X1, y1)
    model1_dirs = train_targets(X1_train, y1_train, MODEL1_DIR)

    # Model2
    X2, y2, _ = preprocess_model2()
    X2_train, X2_test, y2_train, y2_test = train_test_split_xy(X2, y2)
    model2_dirs = train_targets(X2_train, y2_train, MODEL2_DIR)

    meta = {
        "feature_columns_model1": list(X1.columns),
        "feature_columns_model2": list(X2.columns),
        "model1_targets": CURVE_DESCRIPTORS,
        "model2_targets": IV_TARGETS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "model1_dirs": model1_dirs,
        "model2_dirs": model2_dirs,
        "train_indices_model1": X1_train.index.tolist(),
        "test_indices_model1": X1_test.index.tolist(),
        "train_indices_model2": X2_train.index.tolist(),
        "test_indices_model2": X2_test.index.tolist(),
    }
    save_meta(meta, META_JSON)
    print(f"Saved meta -> {META_JSON}")


if __name__ == "__main__":
    main()
