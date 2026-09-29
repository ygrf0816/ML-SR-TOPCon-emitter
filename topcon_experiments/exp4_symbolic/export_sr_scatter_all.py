# -*- coding: utf-8 -*-
"""Export SR regression-scatter (y_true, y_pred) for ALL SR tasks into one wide CSV.

Fixes three tasks that plot_sr_accuracy.py silently skipped:
  * athena_to_defect_vac_N_peak_log   (stem not in {athena_to_<descriptor>} because of the _log suffix)
  * athena_to_defect_vac_dose_log     (same reason)
  * full_to_iv_FF                     (hard-coded `return None`)

Everything else re-uses the exact same code path as plot_sr_accuracy.py so the
pre-existing scatter CSVs are reproduced bit-for-bit (verified after the run).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import (
    inverse_transform_target,
    preprocess_model1,
    preprocess_model2,
)
from topcon_experiments.config import (
    ATHENA_FEATURES,
    CURVE_DESCRIPTORS,
    LOG_TARGETS_MODEL1,
    OUTPUT_ROOT,
    RANDOM_STATE,
)
from topcon_experiments.exp4_symbolic.curve_sr_data import build_sr_dataset
from topcon_experiments.exp4_symbolic.sr_equation_utils import (
    best_row_by_test_r2,
    predict_from_row,
    predict_from_sr_csv,
)
from topcon_experiments.exp4_symbolic.sr_metrics import (
    regression_metrics,
    sort_formulas_by_metrics,
)
from topcon_experiments.exp4_symbolic.sr_split_utils import (
    split_group_train_test,
    split_train_test,
)

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
PLOT_OUT = EXP4_OUT / "sr_plots"
PLOT_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- label table
# target column -> (paper symbol, unit, display family)
SYM = {
    "doping_N_peak":            ("N_a,peak",  "cm^-3",   "doping descriptor"),
    "doping_x_peak":            ("d_a,peak",  "um",      "doping descriptor"),
    "doping_junction_depth":    ("d_junc",    "um",      "doping descriptor"),
    "doping_FWHM":              ("FWHM",      "um",      "doping descriptor"),
    "doping_gradient_max":      ("N_a,grad",  "cm^-3 um^-1", "doping descriptor"),
    "doping_dose":              ("N_a,eff",   "cm^-3 um", "doping descriptor"),
    "doping_R_sheet":           ("R_sheet",   "ohm/sq",  "doping descriptor"),
    "defect_vac_N_peak":        ("N_t,peak",  "cm^-3",   "defect descriptor"),
    "defect_vac_gradient_max":  ("N_t,grad",  "cm^-3 um^-1", "defect descriptor"),
    "defect_vac_dose":          ("N_t,eff",   "cm^-3 um", "defect descriptor"),
    "iv_Voc":                   ("V_oc",      "V",       "IV"),
    "iv_Jsc":                   ("J_sc",      "mA cm^-2", "IV"),
    "iv_FF":                    ("FF",        "%",       "IV"),
    "iv_Eff":                   ("PCE",       "%",       "IV"),
    "doping_ln_conc":           ("N_a",       "cm^-3",   "doping curve"),
    "defect_ln_conc":           ("N_t",       "cm^-3",   "defect curve"),
}

# task slug -> input-set tag used in the column header
INPUT_TAG = {
    "athena": "process",     # 8 process parameters
    "full":   "descriptor",  # 18-dim process + descriptors
}


def metrics(y_true, y_pred):
    m = regression_metrics(y_true, y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    m["n"] = int(mask.sum())
    return m


# ---------------------------------------------------------------- tabular tasks
def resolve_tabular(slug):
    """Return (X, y_model_space, target, log_target, input_tag) or None."""
    if slug.startswith("athena_to_"):
        X1, y1, _ = preprocess_model1()
        X2, y2, _ = preprocess_model2()
        body = slug[len("athena_to_"):]
        # strip a trailing "_log" (defect tasks keep it, doping tasks too)
        target = body[:-4] if body.endswith("_log") else body
        if target in y1:
            return X1, y1[target].values.astype(float), target, target in LOG_TARGETS_MODEL1, "athena"
        if target in y2:
            return (X2[ATHENA_FEATURES], y2[target].values.astype(float),
                    target, False, "athena")
        return None
    if slug.startswith("full_to_"):
        X2, y2, _ = preprocess_model2()
        target = slug[len("full_to_"):]
        if target not in y2:
            return None
        return X2, y2[target].values.astype(float), target, False, "full"
    return None


def run_tabular():
    rows, data = [], {}
    paths = sorted(EXP4_OUT.glob("sr_tabular_*.csv"))
    skip = {
        "sr_tabular_formulas_summary.csv",
        "sr_tabular_full_iv_summary.csv",
        "sr_tabular_iv_trio_to_FF.csv",
    }
    jobs = [p for p in paths if p.name not in skip]
    for path in jobs:
        slug = path.stem.replace("sr_tabular_", "")
        res = resolve_tabular(slug)
        if res is None:
            print(f"[skip ] {slug}: cannot resolve X/y")
            continue
        X, y, target, log_target, tag = res
        _, test_idx = split_train_test(len(X))
        row = best_row_by_test_r2(path)
        pred_all = predict_from_row(row, X.values.astype(float))
        y_true = inverse_transform_target(target, y[test_idx]) if log_target else y[test_idx]
        y_pred = np.exp(pred_all[test_idx]) if log_target else pred_all[test_idx]
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        m = metrics(y_true, y_pred)
        data[slug] = dict(y_true=y_true, y_pred=y_pred, target=target, tag=tag,
                          family=SYM[target][2], equation=str(row.get("equation", "")))
        rows.append(dict(task=slug, task_type="tabular", target=target, input_set=tag,
                         log_target=log_target, formula_rank=1,
                         test_R2_csv=float(row.get("test_R2", np.nan)), **m,
                         equation=str(row.get("equation", ""))))
        print(f"[ok   ] {slug:42s} n={m['n']:5d} R2={m['R2']:+.4f} RMSE={m['RMSE']:.4g}")
    return rows, data


# ---------------------------------------------------------------- FF identity
def run_ff_identity():
    rows, data = [], {}
    X, y_map, _ = preprocess_model2()
    Xv = X.values.astype(float)
    _, test_idx = split_train_test(len(X))

    voc_p = predict_from_sr_csv(EXP4_OUT / "sr_tabular_full_to_iv_Voc.csv", Xv)
    jsc_p = predict_from_sr_csv(EXP4_OUT / "sr_tabular_full_to_iv_Jsc.csv", Xv)
    eff_p = predict_from_sr_csv(EXP4_OUT / "sr_tabular_full_to_iv_Eff.csv", Xv)
    ff_calc = 100.0 * eff_p / (voc_p * jsc_p)
    ff_true = y_map["iv_FF"].values.astype(float)

    specs = [
        ("FF_from_predicted_IV", ff_true[test_idx], ff_calc[test_idx],
         "FF=100*Eff_pred/(Voc_pred*Jsc_pred)"),
    ]
    robust = (ff_calc >= 75.0) & (ff_calc <= 95.0)
    specs.append((
        "FF_from_predicted_IV_robust",
        ff_true[test_idx][robust[test_idx]],
        ff_calc[test_idx][robust[test_idx]],
        "FF=100*Eff_pred/(Voc_pred*Jsc_pred), FF_calc in [75,95]",
    ))
    for slug, yt, yp, eq in specs:
        yt = np.asarray(yt, dtype=float)
        yp = np.asarray(yp, dtype=float)
        m = metrics(yt, yp)
        data[slug] = dict(y_true=yt, y_pred=yp, target="iv_FF", tag="identity",
                          family="IV", equation=eq)
        rows.append(dict(task=slug, task_type="iv_identity", target="iv_FF",
                         input_set="identity", log_target=False, formula_rank=1,
                         test_R2_csv=float("nan"), **m, equation=eq))
        print(f"[ok   ] {slug:42s} n={m['n']:5d} R2={m['R2']:+.4f} RMSE={m['RMSE']:.4g}")
    return rows, data


# ---------------------------------------------------------------- curve tasks
def _curve_pred_ln(row, pred_raw):
    return pred_raw * np.log(10.0) if str(row.get("target", "ln_curve")) == "log10_curve" else pred_raw


def run_curves():
    rows, data = [], {}
    for curve_type in ("doping", "defect"):
        path = EXP4_OUT / f"sr_formulas_{curve_type}.csv"
        if not path.exists():
            continue
        X, y, groups, _ = build_sr_dataset(curve_type)
        eq_df = pd.read_csv(path)
        if not ("test_MSE" in eq_df.columns and eq_df["test_MSE"].notna().any()):
            train_idx, test_idx = split_group_train_test(groups)
            X_test = X.iloc[test_idx].values.astype(float)
            y_test = y[test_idx]
            t_mse, t_mae, t_r2 = [], [], []
            for i in range(len(eq_df)):
                try:
                    pred = _curve_pred_ln(eq_df.iloc[i], predict_from_row(eq_df.iloc[i], X_test))
                    mm = regression_metrics(y_test, pred)
                    t_mse.append(mm["MSE"]); t_mae.append(mm["MAE"]); t_r2.append(mm["R2"])
                except Exception:
                    t_mse.append(np.nan); t_mae.append(np.nan); t_r2.append(np.nan)
            eq_df["test_MSE"] = t_mse; eq_df["test_MAE"] = t_mae; eq_df["test_R2"] = t_r2
        eq_df = sort_formulas_by_metrics(eq_df)
        top3 = eq_df.head(3).reset_index(drop=True)
        _, test_idx = split_group_train_test(groups)
        X_test = X.iloc[test_idx].values.astype(float)
        y_test = y[test_idx]

        target = f"{curve_type}_ln_conc"
        for rank, (_, erow) in enumerate(top3.iterrows(), start=1):
            pred_ln = _curve_pred_ln(erow, predict_from_row(erow, X_test))
            m = metrics(y_test, pred_ln)
            slug = f"curve_{curve_type}_rank{rank}"
            data[slug] = dict(y_true=np.exp(y_test), y_pred=np.exp(pred_ln), target=target,
                              tag=f"curve #{rank}", family=SYM[target][2],
                              equation=str(erow.get("equation", "")), curve_rank=rank,
                              curve_type=curve_type)
            rows.append(dict(task=slug, task_type="curve", target=target,
                             input_set=f"curve #{rank}", log_target=False,
                             formula_rank=rank,
                             test_R2_csv=float(erow.get("test_R2", np.nan)), **m,
                             equation=str(erow.get("equation", ""))))
            print(f"[ok   ] {slug:42s} n={m['n']:5d} R2={m['R2']:+.4f} RMSE={m['RMSE']:.4g}")
    return rows, data


# ---------------------------------------------------------------- output
def save_all(rows, data):
    df = pd.DataFrame(rows)
    df = df.sort_values("R2", ascending=False).reset_index(drop=True)
    df.to_csv(PLOT_OUT / "sr_scatter_all_metrics.csv", index=False)

    # ---- wide table: True / Predicted column pairs
    order = df["task"].tolist()
    wide = {}
    long_rows = []
    max_n = 0
    for slug in order:
        d = data[slug]
        yt, yp = d["y_true"], d["y_pred"]
        max_n = max(max_n, len(yt))
    for slug in order:
        d = data[slug]
        sym, unit, family = SYM[d["target"]]
        tag = d["tag"]
        yt, yp = np.asarray(d["y_true"], float), np.asarray(d["y_pred"], float)
        ct = f"True {sym} ({unit}) [{tag}]"
        cp = f"Predicted {sym} ({unit}) [{tag}]"
        a = np.full(max_n, np.nan); a[: len(yt)] = yt
        b = np.full(max_n, np.nan); b[: len(yp)] = yp
        wide[ct] = a
        wide[cp] = b
        for i in range(len(yt)):
            long_rows.append(dict(task=slug, target=d["target"], symbol=sym, unit=unit,
                                  family=family, input_set=tag, idx=i,
                                  y_true=yt[i], y_pred=yp[i]))
    wide_df = pd.DataFrame(wide)
    wide_df.insert(0, "Index", np.arange(1, max_n + 1))
    wide_df.to_csv(PLOT_OUT / "sr_scatter_all_wide.csv", index=False)
    pd.DataFrame(long_rows).to_csv(PLOT_OUT / "sr_scatter_all_long.csv", index=False)
    print(f"\nwide  -> {PLOT_OUT/'sr_scatter_all_wide.csv'}  shape={wide_df.shape}")
    print(f"long  -> {PLOT_OUT/'sr_scatter_all_long.csv'}")
    print(f"metrics -> {PLOT_OUT/'sr_scatter_all_metrics.csv'}")
    return df


def patch_missing_per_task_csvs(data):
    """Write the per-task scatter CSVs that plot_sr_accuracy.py never produced."""
    from topcon_experiments.common.variable_labels import label_for
    targets = ["athena_to_defect_vac_N_peak_log", "athena_to_defect_vac_dose_log",
               "full_to_iv_FF"]
    for slug in targets:
        if slug not in data:
            print(f"[warn ] {slug} not computed")
            continue
        d = data[slug]
        t = d["target"]
        out = PLOT_OUT / f"regression_scatter_sr_{slug}.csv"
        pd.DataFrame({
            "target_column": [t] * len(d["y_true"]),
            "target_label": [label_for(t)] * len(d["y_true"]),
            f"y_true_{t}": d["y_true"],
            f"y_pred_{t}": d["y_pred"],
        }).to_csv(out, index=False)
        print(f"[patch] {out.name}")


def main():
    rows, data = [], {}
    r, d = run_tabular();      rows += r; data.update(d)
    r, d = run_ff_identity();  rows += r; data.update(d)
    r, d = run_curves();       rows += r; data.update(d)
    df = save_all(rows, data)
    patch_missing_per_task_csvs(data)
    print("\n=== ranking (R2 desc) ===")
    print(df[["task", "target", "input_set", "n", "R2", "RMSE", "MAE"]].to_string(index=False))


if __name__ == "__main__":
    main()
