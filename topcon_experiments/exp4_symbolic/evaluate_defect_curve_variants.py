"""Evaluate defect curve SR: baseline vs scheme A (mean+shape) vs scheme B (descriptor reconstruct)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import inverse_transform_target, load_raw_dataframe, preprocess_model1
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import (
    ATHENA_FEATURES,
    DEFECT_DESCRIPTORS,
    LOG_TARGETS_MODEL1,
    OUTPUT_ROOT,
    RANDOM_STATE,
)
from topcon_experiments.exp4_symbolic.curve_sr_data import build_sr_dataset
from topcon_experiments.exp4_symbolic.defect_curve_reconstruct import reconstruct_defect_flat_plus_decay
from topcon_experiments.exp4_symbolic.sr_equation_utils import best_row_by_test_r2, predict_from_row
from topcon_experiments.exp4_symbolic.sr_metrics import regression_metrics
from topcon_experiments.exp4_symbolic.sr_split_utils import split_group_train_test, split_train_test

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
PLOT_OUT = EXP4_OUT / "sr_plots"


def _athena_row(meta: pd.DataFrame, fb: str) -> pd.Series | None:
    from topcon_experiments.config import LOG_FEATURES

    fb_key = fb
    if fb_key not in meta.index:
        lower_map = {str(i).lower(): i for i in meta.index}
        fb_key = lower_map.get(str(fb).lower())
        if fb_key is None:
            return None
    row = meta.loc[fb_key, ATHENA_FEATURES].astype(float).copy()
    for col in LOG_FEATURES:
        if col in row.index and row[col] > 0:
            row[col] = float(np.log(row[col]))
    return row


def _predict_scheme_a(
    mean_row: pd.Series,
    shape_row: pd.Series,
    athena: pd.Series,
    depths: np.ndarray,
) -> np.ndarray:
    X_mean = athena.values.reshape(1, -1).astype(float)
    mean_ln = float(predict_from_row(mean_row, X_mean)[0])
    X_shape = np.column_stack(
        [np.tile(athena[c], len(depths)) for c in ATHENA_FEATURES] + [depths]
    )
    delta_ln = predict_from_row(shape_row, X_shape)
    return np.exp(mean_ln + delta_ln)


def _predict_scheme_b(
    n_peak_row: pd.Series,
    dose_row: pd.Series,
    athena: pd.Series,
    depths: np.ndarray,
) -> np.ndarray:
    X = athena.values.reshape(1, -1).astype(float)
    ln_n_peak = float(predict_from_row(n_peak_row, X)[0])
    ln_dose = float(predict_from_row(dose_row, X)[0])
    n_peak = float(np.exp(ln_n_peak))
    dose = float(np.exp(ln_dose))
    return reconstruct_defect_flat_plus_decay(depths, n_peak, dose)


def _collect_point_metrics(
    curve_df: pd.DataFrame,
    meta: pd.DataFrame,
    test_groups: np.ndarray,
    predict_fn,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_list: list[float] = []
    y_pred_list: list[float] = []
    for fb in test_groups:
        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um")
        if sub.empty:
            continue
        athena = _athena_row(meta, str(fb))
        if athena is None:
            continue
        depths = sub["depth_um"].values.astype(float)
        true_conc = sub["value_fitted"].values.astype(float)
        pred_conc = predict_fn(athena, depths)
        y_true_list.extend(true_conc.tolist())
        y_pred_list.extend(pred_conc.tolist())
    return np.array(y_true_list), np.array(y_pred_list)


def _plot_overlays(
    curve_df: pd.DataFrame,
    meta: pd.DataFrame,
    picked: list[str],
    predictors: dict[str, callable],
    out_stem: Path,
) -> None:
    n = len(picked)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.5 * nrow), squeeze=False)

    for ax, fb in zip(axes.flatten(), picked):
        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um")
        athena = _athena_row(meta, str(fb))
        if sub.empty or athena is None:
            ax.set_visible(False)
            continue
        depths = sub["depth_um"].values.astype(float)
        true_conc = sub["value_fitted"].values.astype(float)
        ax.plot(depths, true_conc, "k-", lw=2, label="true")
        colors = ["C0", "C1", "C2", "C3"]
        for i, (name, fn) in enumerate(predictors.items()):
            pred = fn(athena, depths)
            ax.plot(depths, pred, "--", color=colors[i % len(colors)], lw=1.5, label=name)
        ax.set_yscale("log")
        ax.set_title(str(fb)[-18:])
        ax.legend(fontsize=6)
    for ax in axes.flatten()[len(picked):]:
        ax.set_visible(False)
    fig.suptitle("Defect curve variants (test samples)")
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_runtime()
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    curve_df = pd.read_csv(EXP4_OUT / "curve_processed_defect.csv")
    meta = load_raw_dataframe().set_index("file_base")

    _, _, g_base, _ = build_sr_dataset("defect")
    _, test_idx = split_group_train_test(g_base)
    test_groups = np.unique(g_base[test_idx])
    rng = np.random.default_rng(RANDOM_STATE)
    n_pick = min(6, len(test_groups))
    picked = list(rng.choice(test_groups, size=n_pick, replace=False))

    rows: list[dict] = []
    overlay_fns: dict[str, callable] = {}

    # Baseline full curve SR
    base_path = EXP4_OUT / "sr_formulas_defect.csv"
    if base_path.exists():
        base_row = best_row_by_test_r2(base_path)

        def _pred_base(athena, depths, _row=base_row):
            X = np.column_stack(
                [np.tile(athena[c], len(depths)) for c in ATHENA_FEATURES] + [depths]
            )
            return np.exp(predict_from_row(_row, X))

        yt, yp = _collect_point_metrics(curve_df, meta, test_groups, _pred_base)
        m = regression_metrics(np.log(yt), np.log(np.maximum(yp, 1e-30)))
        rows.append({"variant": "baseline_full_curve_sr", **m})
        overlay_fns["baseline SR"] = _pred_base
        log(f"baseline: test_R2(ln)={m['R2']:.4f} MSE={m['MSE']:.4g}")

    # Scheme A
    mean_path = EXP4_OUT / "sr_formulas_defect_mean.csv"
    shape_path = EXP4_OUT / "sr_formulas_defect_shape.csv"
    if mean_path.exists() and shape_path.exists():
        mean_row = best_row_by_test_r2(mean_path)
        shape_row = best_row_by_test_r2(shape_path)

        def _pred_a(athena, depths, _mr=mean_row, _sr=shape_row):
            return _predict_scheme_a(_mr, _sr, athena, depths)

        yt, yp = _collect_point_metrics(curve_df, meta, test_groups, _pred_a)
        m = regression_metrics(np.log(yt), np.log(np.maximum(yp, 1e-30)))
        rows.append({"variant": "scheme_A_mean_plus_shape", **m})
        overlay_fns["A: mean+shape"] = _pred_a
        log(f"scheme A: test_R2(ln)={m['R2']:.4f} MSE={m['MSE']:.4g}")

    # Scheme B
    peak_path = EXP4_OUT / "sr_tabular_athena_to_defect_vac_N_peak_log.csv"
    dose_path = EXP4_OUT / "sr_tabular_athena_to_defect_vac_dose_log.csv"
    if peak_path.exists() and dose_path.exists():
        peak_row = best_row_by_test_r2(peak_path)
        dose_row = best_row_by_test_r2(dose_path)

        def _pred_b(athena, depths, _pr=peak_row, _dr=dose_row):
            return _predict_scheme_b(_pr, _dr, athena, depths)

        yt, yp = _collect_point_metrics(curve_df, meta, test_groups, _pred_b)
        m = regression_metrics(np.log(yt), np.log(np.maximum(yp, 1e-30)))
        rows.append({"variant": "scheme_B_descriptor_reconstruct", **m})
        overlay_fns["B: desc reconstruct"] = _pred_b
        log(f"scheme B: test_R2(ln)={m['R2']:.4f} MSE={m['MSE']:.4g}")

    if rows:
        summary = pd.DataFrame(rows)
        save_csv(summary, PLOT_OUT / "defect_curve_variants_metrics.csv")

    # Tabular defect descriptor SR accuracy (scheme B building blocks)
    X1, y1_map, _ = preprocess_model1()
    desc_rows: list[dict] = []
    for target in DEFECT_DESCRIPTORS:
        log_target = target in LOG_TARGETS_MODEL1
        task = f"athena_to_{target}_log" if log_target else f"athena_to_{target}"
        path = EXP4_OUT / f"sr_tabular_{task}.csv"
        if not path.exists():
            continue
        _, test_idx_t = split_train_test(len(X1))
        row = best_row_by_test_r2(path)
        pred = predict_from_row(row, X1.iloc[test_idx_t].values.astype(float))
        y_test = y1_map[target].values.astype(float)[test_idx_t]
        if log_target:
            yt_disp = inverse_transform_target(target, y_test)
            yp_disp = np.exp(pred)
        else:
            yt_disp = y_test
            yp_disp = pred
        m = regression_metrics(yt_disp, yp_disp)
        desc_rows.append({"target": target, "test_R2": m["R2"], "test_MSE": m["MSE"]})
        log(f"tabular {target}: test_R2={m['R2']:.4f}")
    if desc_rows:
        save_csv(pd.DataFrame(desc_rows), PLOT_OUT / "defect_descriptor_sr_metrics.csv")

    if overlay_fns:
        _plot_overlays(
            curve_df, meta, picked, overlay_fns,
            PLOT_OUT / "curve_overlay_defect_variants_6samples",
        )
        log(f"overlay -> {PLOT_OUT / 'curve_overlay_defect_variants_6samples.png'}")


if __name__ == "__main__":
    main()
