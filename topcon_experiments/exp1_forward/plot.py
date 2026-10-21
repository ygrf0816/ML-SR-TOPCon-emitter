"""Plot regression scatter and feature importance for Model1/Model2."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from autogluon.tabular import TabularPredictor

from topcon_experiments.common.autogluon_utils import load_predictor, safe_feature_importance, safe_predict
from topcon_experiments.common.data import inverse_transform_target, load_meta, preprocess_model1, preprocess_model2
from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.plot_utils import plot_feature_importance, plot_regression_scatter
from topcon_experiments.config import CURVE_DESCRIPTORS, IV_TARGETS, LOG_TARGETS_MODEL1, META_JSON, OUTPUT_ROOT


EXP1_OUT = OUTPUT_ROOT / "exp1_forward"


def export_scatter_and_importance(
    model_label: str,
    model_dirs: dict[str, str],
    X: pd.DataFrame,
    y_map: dict[str, pd.Series],
    test_idx: list,
    log_targets: list[str] | None = None,
) -> None:
    log_targets = log_targets or []
    X_test = X.loc[test_idx]
    targets = list(y_map.keys())
    for i, (target, y_series) in enumerate(y_map.items(), start=1):
        log_step(i, len(targets), f"{model_label} plot {target}")
        predictor = load_predictor(model_dirs[target])
        y_pred_model = safe_predict(predictor, X_test)
        y_true_raw = y_series.loc[test_idx].values

        if target in log_targets:
            y_true = inverse_transform_target(target, y_true_raw)
            y_pred = inverse_transform_target(target, y_pred_model)
            log_scale = True
        else:
            y_true = y_true_raw
            y_pred = y_pred_model
            log_scale = False

        prefix = EXP1_OUT / f"regression_scatter_{model_label}_{target}"
        plot_regression_scatter(y_true, y_pred, target, prefix, log_scale=log_scale)

        imp_df = safe_feature_importance(predictor, X_test.assign(**{target: y_series.loc[test_idx].values}))
        plot_feature_importance(
            imp_df,
            f"{model_label} feature importance: {target}",
            EXP1_OUT / f"feature_importance_{model_label}_{target}",
        )


def main() -> None:
    setup_runtime()
    EXP1_OUT.mkdir(parents=True, exist_ok=True)
    meta = load_meta(META_JSON)
    log("Exporting regression scatter and feature importance plots...")

    X1, y1, _ = preprocess_model1()
    export_scatter_and_importance(
        "model1",
        meta["model1_dirs"],
        X1,
        y1,
        meta["test_indices_model1"],
        log_targets=LOG_TARGETS_MODEL1,
    )

    X2, y2, _ = preprocess_model2()
    export_scatter_and_importance(
        "model2",
        meta["model2_dirs"],
        X2,
        y2,
        meta["test_indices_model2"],
    )
    log(f"Plots saved to {EXP1_OUT}")

    from topcon_experiments.exp1_forward.merge_exports import main as merge_main
    merge_main()


if __name__ == "__main__":
    main()
