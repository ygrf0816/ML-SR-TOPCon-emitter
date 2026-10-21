"""Merge per-target regression / feature-importance CSVs into summary files."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.variable_labels import label_for
from topcon_experiments.config import CURVE_DESCRIPTORS, IV_TARGETS, OUTPUT_ROOT

EXP1_OUT = OUTPUT_ROOT / "exp1_forward"


def _targets_for_model(model: str) -> list[str]:
    return CURVE_DESCRIPTORS if model == "model1" else IV_TARGETS


def _read_target_scatter(model: str, target: str) -> tuple[pd.Series, pd.Series] | None:
    path = EXP1_OUT / f"regression_scatter_{model}_{target}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    y_true_col = f"y_true_{target}"
    y_pred_col = f"y_pred_{target}"
    if y_true_col not in df.columns and "y_true" in df.columns:
        y_true_col = "y_true"
    if y_pred_col not in df.columns and "y_pred" in df.columns:
        y_pred_col = "y_pred"
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        return None
    return df[y_true_col], df[y_pred_col]


def merge_regression_scatter_wide(model: str) -> pd.DataFrame:
    """Wide format: one row per test sample, one column per target y_true/y_pred."""
    targets = _targets_for_model(model)
    columns: dict[str, pd.Series] = {}
    n_rows: int | None = None

    for target in targets:
        pair = _read_target_scatter(model, target)
        if pair is None:
            continue
        y_true, y_pred = pair
        if n_rows is None:
            n_rows = len(y_true)
        elif len(y_true) != n_rows:
            raise ValueError(f"Row count mismatch for {model}/{target}")
        columns[f"y_true_original_{target}"] = y_true
        columns[f"y_pred_original_{target}"] = y_pred

    if not columns:
        return pd.DataFrame()

    wide = pd.DataFrame(columns)
    wide.insert(0, "model", model)
    wide.insert(1, "split", "test")
    wide.insert(2, "sample_index", range(len(wide)))
    return wide


def merge_regression_scatter_long(model: str) -> pd.DataFrame:
    """Legacy long format kept for compatibility."""
    rows = []
    for target in _targets_for_model(model):
        pair = _read_target_scatter(model, target)
        if pair is None:
            continue
        y_true, y_pred = pair
        part = pd.DataFrame(
            {
                "model": model,
                "target_column": target,
                "target_label": label_for(target),
                "split": "test",
                "y_true_original": y_true.values,
                "y_pred_original": y_pred.values,
            }
        )
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def merge_feature_importance(model: str) -> pd.DataFrame:
    rows = []
    for target in _targets_for_model(model):
        path = EXP1_OUT / f"feature_importance_{model}_{target}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "feature" not in df.columns or "importance" not in df.columns:
            continue
        part = pd.DataFrame(
            {
                "model": model,
                "target_column": target,
                "target_label": label_for(target),
                "feature_column": df["feature"].values,
                "feature_label": df["feature"].map(label_for).values,
                "importance": df["importance"].values,
            }
        )
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    setup_runtime()
    reg_all_long = []
    imp_all = []
    for model in ["model1", "model2"]:
        wide = merge_regression_scatter_wide(model)
        if len(wide):
            out = EXP1_OUT / f"regression_scatter_merged_{model}.csv"
            wide.to_csv(out, index=False)
            log(f"Saved wide {out.name}: {len(wide)} rows x {len(wide.columns)} cols")

        reg_long = merge_regression_scatter_long(model)
        if len(reg_long):
            reg_all_long.append(reg_long)

        imp = merge_feature_importance(model)
        if len(imp):
            imp.to_csv(EXP1_OUT / f"feature_importance_merged_{model}.csv", index=False)
            log(f"Saved feature_importance_merged_{model}.csv ({len(imp)} rows)")
            imp_all.append(imp)

    if reg_all_long:
        reg_merged = pd.concat(reg_all_long, ignore_index=True)
        reg_merged.to_csv(EXP1_OUT / "regression_scatter_merged.csv", index=False)
        log(f"Saved regression_scatter_merged.csv long format ({len(reg_merged)} rows)")
    if imp_all:
        imp_merged = pd.concat(imp_all, ignore_index=True)
        imp_merged.to_csv(EXP1_OUT / "feature_importance_merged.csv", index=False)
        log(f"Saved feature_importance_merged.csv ({len(imp_merged)} rows)")


if __name__ == "__main__":
    main()
