"""SR accuracy plots: regression scatter (tabular + IV), curve overlays, summary figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import (
    inverse_transform_target,
    load_raw_dataframe,
    preprocess_model1,
    preprocess_model2,
)
from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.plot_utils import plot_regression_scatter, save_csv
from topcon_experiments.common.variable_labels import label_for
from topcon_experiments.config import (
    ATHENA_FEATURES,
    CURVE_DESCRIPTORS,
    IV_TARGETS,
    LOG_TARGETS_MODEL1,
    OUTPUT_ROOT,
    RANDOM_STATE,
)
from topcon_experiments.exp4_symbolic.curve_sampling import adaptive_sample_indices, curve_change_score
from topcon_experiments.exp4_symbolic.curve_sr_data import build_sr_dataset
from topcon_experiments.exp4_symbolic.sr_metrics import regression_metrics, sort_formulas_by_metrics
from topcon_experiments.exp4_symbolic.sr_equation_utils import (
    best_row_by_test_r2,
    predict_from_row,
    predict_from_sr_csv,
    top_rows_by_test_r2,
)
from topcon_experiments.exp4_symbolic.sr_split_utils import split_group_train_test, split_train_test

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
PLOT_OUT = EXP4_OUT / "sr_plots"

SKIP_CSV = {
    "sr_tabular_formulas_summary.csv",
    "sr_tabular_full_iv_summary.csv",
    "sr_tabular_iv_trio_to_FF.csv",
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = regression_metrics(y_true, y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    m["n"] = int(mask.sum())
    return m


def _resolve_tabular_xy(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, str, bool] | None:
    """Return (X, y_model_space, target, log_target) for a tabular SR CSV."""
    stem = csv_path.stem.replace("sr_tabular_", "")
    if stem.startswith("full_to_iv_FF"):
        return None

    if stem.startswith("athena_to_doping_") or stem in {f"athena_to_{t}" for t in CURVE_DESCRIPTORS}:
        X, y_map, _ = preprocess_model1()
        if stem.endswith("_log"):
            target = stem.replace("athena_to_", "").replace("_log", "")
        else:
            target = stem.replace("athena_to_", "")
        if target not in y_map:
            return None
        log_target = target in LOG_TARGETS_MODEL1
        return X, y_map[target].values.astype(float), target, log_target

    if stem.startswith("athena_to_iv_"):
        X_full, y_map, _ = preprocess_model2()
        target = stem.replace("athena_to_", "")
        X = X_full[ATHENA_FEATURES]
        return X, y_map[target].values.astype(float), target, False

    if stem.startswith("full_to_iv_"):
        X, y_map, _ = preprocess_model2()
        target = stem.replace("full_to_", "")
        if target == "iv_FF":
            return None
        return X, y_map[target].values.astype(float), target, False

    return None


def _to_display_scale(target: str, model_values: np.ndarray, log_target: bool) -> np.ndarray:
    if log_target:
        return inverse_transform_target(target, model_values)
    return model_values


def _pred_to_display(target: str, pred_model: np.ndarray, log_target: bool) -> np.ndarray:
    if log_target:
        return np.exp(pred_model)
    return pred_model


def plot_tabular_scatters() -> list[dict]:
    rows: list[dict] = []
    paths = sorted(EXP4_OUT.glob("sr_tabular_*.csv"))
    jobs = [p for p in paths if p.name not in SKIP_CSV]
    for i, path in enumerate(jobs, start=1):
        resolved = _resolve_tabular_xy(path)
        if resolved is None:
            continue
        X, y, target, log_target = resolved
        _, test_idx = split_train_test(len(X))
        row = best_row_by_test_r2(path)
        pred_all = predict_from_row(row, X.values.astype(float))
        y_true = _to_display_scale(target, y[test_idx], log_target)
        y_pred = _pred_to_display(target, pred_all[test_idx], log_target)
        m = _metrics(y_true, y_pred)
        slug = path.stem.replace("sr_tabular_", "")
        prefix = PLOT_OUT / f"regression_scatter_sr_{slug}"
        plot_regression_scatter(y_true, y_pred, target, prefix, log_scale=log_target)
        rows.append({
            "task": slug,
            "task_type": "tabular",
            "target": target,
            "split": "test",
            "formula_rank": 1,
            "test_R2_csv": float(row.get("test_R2", np.nan)),
            **m,
            "equation": str(row.get("equation", ""))[:120],
        })
        log_step(i, len(jobs), f"tabular scatter {slug} test_R2={m['R2']:.4f}")
    return rows


def plot_iv_ff_identity() -> list[dict]:
    """Voc/Jsc/Eff from SR; FF from predicted IV + identity (not direct SR on FF)."""
    rows: list[dict] = []
    X, y_map, _ = preprocess_model2()
    Xv = X.values.astype(float)
    _, test_idx = split_train_test(len(X))

    for target in ["iv_Voc", "iv_Jsc", "iv_Eff"]:
        path = EXP4_OUT / f"sr_tabular_full_to_{target}.csv"
        if not path.exists():
            continue
        row = best_row_by_test_r2(path)
        pred = predict_from_sr_csv(path, Xv)
        y_true = y_map[target].values[test_idx]
        y_pred = pred[test_idx]
        m = _metrics(y_true, y_pred)
        prefix = PLOT_OUT / f"regression_scatter_sr_full_to_{target}"
        plot_regression_scatter(y_true, y_pred, target, prefix)
        rows.append({
            "task": f"full_to_{target}",
            "task_type": "iv_full",
            "target": target,
            "split": "test",
            "formula_rank": 1,
            "test_R2_csv": float(row.get("test_R2", np.nan)),
            **m,
            "equation": str(row.get("equation", ""))[:120],
        })
        log(f"IV scatter {target}: test_R2={m['R2']:.4f}")

    voc_p = predict_from_sr_csv(EXP4_OUT / "sr_tabular_full_to_iv_Voc.csv", Xv)
    jsc_p = predict_from_sr_csv(EXP4_OUT / "sr_tabular_full_to_iv_Jsc.csv", Xv)
    eff_p = predict_from_sr_csv(EXP4_OUT / "sr_tabular_full_to_iv_Eff.csv", Xv)
    ff_calc = 100.0 * eff_p / (voc_p * jsc_p)
    ff_true = y_map["iv_FF"].values
    y_true = ff_true[test_idx]
    y_pred = ff_calc[test_idx]
    m = _metrics(y_true, y_pred)
    prefix = PLOT_OUT / "regression_scatter_sr_FF_from_predicted_IV"
    plot_regression_scatter(y_true, y_pred, "iv_FF", prefix)
    rows.append({
        "task": "FF_from_predicted_IV",
        "task_type": "iv_identity",
        "target": "iv_FF",
        "split": "test",
        "formula_rank": 1,
        "test_R2_csv": float("nan"),
        **m,
        "equation": "FF=100*Eff_pred/(Voc_pred*Jsc_pred)",
    })
    log(f"FF identity scatter: test_R2={m['R2']:.4f}, MAE={m['MAE']:.3f}%")

    robust = (ff_calc >= 75.0) & (ff_calc <= 95.0)
    y_true_r = ff_true[test_idx][robust[test_idx]]
    y_pred_r = ff_calc[test_idx][robust[test_idx]]
    m_r = _metrics(y_true_r, y_pred_r)
    plot_regression_scatter(
        y_true_r, y_pred_r, "iv_FF",
        PLOT_OUT / "regression_scatter_sr_FF_from_predicted_IV_robust",
    )
    rows.append({
        "task": "FF_from_predicted_IV_robust",
        "task_type": "iv_identity",
        "target": "iv_FF",
        "split": "test",
        "formula_rank": 1,
        "test_R2_csv": float("nan"),
        **m_r,
        "equation": "FF=100*Eff_pred/(Voc_pred*Jsc_pred), FF_calc in [75,95]",
    })
    log(f"FF identity robust scatter: test_R2={m_r['R2']:.4f}, MAE={m_r['MAE']:.3f}%")
    return rows


def _overlay_column_name(file_base: str) -> str:
    s = str(file_base).replace("/", "_")
    return s[-24:] if len(s) > 24 else s


def _save_wide_overlay_csv(
    curve_type: str,
    picked: list[str],
    per_sample: dict[str, dict],
    out_path: Path,
) -> None:
    """Wide table: depth_um + per-sample true/SR1/SR2/SR3 columns (Origin-friendly)."""
    ref_fb = picked[0]
    depths = per_sample[ref_fb]["depth_um"]
    wide: dict[str, np.ndarray] = {"depth_um": depths}
    for fb in picked:
        col = _overlay_column_name(fb)
        data = per_sample[fb]
        wide[f"{col}_true"] = data["conc_true"]
        for rank in (1, 2, 3):
            wide[f"{col}_SR{rank}"] = data.get(f"pred_{rank}", np.full_like(depths, np.nan))
    save_csv(pd.DataFrame(wide), out_path)


def _curve_pred_ln(row: pd.Series, pred_raw: np.ndarray) -> np.ndarray:
    target_kind = str(row.get("target", "ln_curve"))
    if target_kind == "log10_curve":
        return pred_raw * np.log(10.0)
    return pred_raw


def _ensure_curve_test_metrics(path: Path, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "test_MSE" in df.columns and df["test_MSE"].notna().any():
        return sort_formulas_by_metrics(df)
    train_idx, test_idx = split_group_train_test(groups)
    X_test = X.iloc[test_idx].values.astype(float)
    y_test = y[test_idx]
    from topcon_experiments.exp4_symbolic.sr_metrics import regression_metrics

    test_mse, test_mae, test_r2 = [], [], []
    for i in range(len(df)):
        try:
            pred = _curve_pred_ln(df.iloc[i], predict_from_row(df.iloc[i], X_test))
            m = regression_metrics(y_test, pred)
            test_mse.append(m["MSE"])
            test_mae.append(m["MAE"])
            test_r2.append(m["R2"])
        except Exception:
            test_mse.append(float("nan"))
            test_mae.append(float("nan"))
            test_r2.append(float("nan"))
    df["test_MSE"] = test_mse
    df["test_MAE"] = test_mae
    df["test_R2"] = test_r2
    return sort_formulas_by_metrics(df)


def plot_curve_scatters_and_overlays() -> list[dict]:
    rows: list[dict] = []
    rng = np.random.default_rng(RANDOM_STATE)
    for curve_type in ("doping", "defect"):
        path = EXP4_OUT / f"sr_formulas_{curve_type}.csv"
        if not path.exists():
            log(f"Skip curve {curve_type}: no {path.name}")
            continue
        try:
            X, y, groups, _ = build_sr_dataset(curve_type)
        except FileNotFoundError as exc:
            log(f"Skip curve {curve_type}: {exc}")
            continue

        eq_df = _ensure_curve_test_metrics(path, X, y, groups)
        top3 = eq_df.head(3).reset_index(drop=True)
        train_idx, test_idx = split_group_train_test(groups)
        X_test = X.iloc[test_idx].values.astype(float)
        y_test = y[test_idx]

        for rank, (_, erow) in enumerate(top3.iterrows(), start=1):
            pred_ln = _curve_pred_ln(erow, predict_from_row(erow, X_test))
            m = _metrics(y_test, pred_ln)
            y_true_disp = np.exp(y_test)
            y_pred_disp = np.exp(pred_ln)
            prefix = PLOT_OUT / f"regression_scatter_sr_curve_{curve_type}_rank{rank}"
            plot_regression_scatter(
                y_true_disp, y_pred_disp, f"{curve_type}_conc", prefix, log_scale=True,
            )
            rows.append({
                "task": f"curve_{curve_type}",
                "task_type": "curve",
                "target": f"{curve_type}_ln_conc",
                "split": "test",
                "formula_rank": rank,
                "test_R2_csv": float(erow.get("test_R2", np.nan)),
                **m,
                "equation": str(erow.get("equation", ""))[:120],
            })
            log(f"curve {curve_type} rank{rank}: test_MSE={m['MSE']:.4g} test_MAE={m['MAE']:.4g} test_R2(ln)={m['R2']:.4f}")

        _plot_curve_overlays(curve_type, top3, groups, rng)
        _plot_depth_sampling(curve_type, groups, rng)
    return rows


def _plot_curve_overlays(
    curve_type: str,
    top3: pd.DataFrame,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> None:
    curve_csv = EXP4_OUT / f"curve_processed_{curve_type}.csv"
    if not curve_csv.exists():
        return
    curve_df = pd.read_csv(curve_csv)
    meta = load_raw_dataframe().set_index("file_base")

    _, test_idx = split_group_train_test(groups)
    test_groups = np.unique(groups[test_idx])
    n_pick = min(6, len(test_groups))
    picked = list(rng.choice(test_groups, size=n_pick, replace=False))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes_flat = axes.flatten()
    per_sample: dict[str, dict] = {}

    for ax, fb in zip(axes_flat, picked):
        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um")
        if sub.empty:
            continue
        fb_key = fb
        if fb_key not in meta.index:
            lower_map = {str(i).lower(): i for i in meta.index}
            fb_key = lower_map.get(str(fb).lower())
        if fb_key is None:
            continue
        athena = meta.loc[fb_key, ATHENA_FEATURES].astype(float).copy()
        from topcon_experiments.config import LOG_FEATURES
        for col in LOG_FEATURES:
            if col in athena.index and athena[col] > 0:
                athena[col] = float(np.log(athena[col]))

        depths = sub["depth_um"].values.astype(float)
        true_conc = sub["value_fitted"].values.astype(float)
        X_grid = np.column_stack(
            [np.tile(athena[c], len(depths)) for c in ATHENA_FEATURES] + [depths]
        )

        ax.plot(depths, true_conc, "k-", lw=2, label="fitted (true)")
        sample_data: dict = {"depth_um": depths, "conc_true": true_conc}
        colors = ["C0", "C1", "C2"]
        for rank, (_, erow) in enumerate(top3.iterrows(), start=1):
            pred_ln = _curve_pred_ln(erow, predict_from_row(erow, X_grid))
            pred_conc = np.exp(pred_ln)
            mse = float(erow.get("test_MSE", np.nan))
            sample_data[f"pred_{rank}"] = pred_conc
            ax.plot(
                depths, pred_conc, "--", color=colors[rank - 1], lw=1.5,
                label=f"SR#{rank} MSE={mse:.3g}",
            )
        per_sample[str(fb)] = sample_data
        ax.set_yscale("log")
        ax.set_title(str(fb)[-18:])
        ax.set_xlabel("depth (μm)")
        ax.set_ylabel("concentration")
        ax.legend(fontsize=7, loc="best")

    for ax in axes_flat[n_pick:]:
        ax.set_visible(False)
    fig.suptitle(f"{curve_type} curve: true vs top-3 SR formulas (6 random test samples)")
    fig.tight_layout()
    out = PLOT_OUT / f"curve_overlay_{curve_type}_6samples"
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    if per_sample:
        _save_wide_overlay_csv(curve_type, [str(f) for f in picked], per_sample, out.with_suffix(".csv"))
    log(f"curve overlay -> {out.with_suffix('.png')}")


def _plot_depth_sampling(curve_type: str, groups: np.ndarray, rng: np.random.Generator) -> None:
    """Illustrate how depth_um enters SR: full grid vs subsampled training points."""
    curve_csv = EXP4_OUT / f"curve_processed_{curve_type}.csv"
    if not curve_csv.exists():
        return
    from topcon_experiments.config import RANDOM_STATE, SR_MAX_POINTS_PER_FILE

    curve_df = pd.read_csv(curve_csv)
    uniq = np.unique(groups)
    fb = str(rng.choice(uniq))
    sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um").reset_index(drop=True)
    if len(sub) < 2:
        return
    depths = sub["depth_um"].values.astype(float)
    conc = sub["value_fitted"].values.astype(float)
    k = min(SR_MAX_POINTS_PER_FILE, len(sub))
    adapt_rng = np.random.default_rng(RANDOM_STATE)
    adapt_idx = adaptive_sample_indices(depths, conc, k, adapt_rng, curve_type=curve_type)
    uniform_idx = np.linspace(0, len(sub) - 1, k).astype(int)
    score = curve_change_score(depths, conc, curve_type=curve_type)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, conc, "b-", alpha=0.5, lw=2, label="processed grid (256 pts)")
    ax.scatter(
        depths[uniform_idx], conc[uniform_idx],
        c="gray", s=28, alpha=0.5, marker="x", label=f"old uniform ({k} pts)",
    )
    ax.scatter(
        depths[adapt_idx], conc[adapt_idx],
        c=score[adapt_idx], cmap="Reds", s=40, zorder=5,
        label=f"adaptive SR sample ({len(adapt_idx)} pts)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("depth_um (μm) — PySR feature x8")
    ax.set_ylabel("concentration (fitted)")
    ax.set_title(
        f"{curve_type}: adaptive depth sampling (red=high |d ln c / d depth|)\n"
        f"sample={fb[-20:]}"
    )
    ax.legend()
    fig.tight_layout()
    out = PLOT_OUT / f"depth_sampling_{curve_type}"
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    sampled_depths = set(np.round(depths[adapt_idx], 8))
    save_csv(
        pd.DataFrame({
            "file_base": fb,
            "depth_um": depths,
            "conc_fitted": conc,
            "change_score": score,
            "is_adaptive_sr_sample": [round(d, 8) in sampled_depths for d in depths],
        }),
        out.with_suffix(".csv"),
    )


def plot_summary_figures(metric_rows: list[dict]) -> None:
    if not metric_rows:
        return
    df = pd.DataFrame(metric_rows)
    df["label"] = df.apply(
        lambda r: f"{r['task']}#{r.get('formula_rank', 1)}", axis=1,
    )

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(df))))
    plot_df = df.sort_values("R2", ascending=True)
    colors = plot_df["task_type"].map({
        "tabular": "steelblue", "iv_full": "seagreen",
        "iv_identity": "darkorange", "curve": "mediumpurple",
    }).fillna("gray")
    ax.barh(plot_df["label"], plot_df["R2"], color=colors)
    ax.set_xlabel("test R² (scatter evaluation)")
    ax.set_title("SR formula accuracy (test set)")
    ax.axvline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(PLOT_OUT / "sr_test_r2_summary.png", dpi=150)
    plt.close(fig)
    save_csv(plot_df, PLOT_OUT / "sr_test_r2_summary.csv")

    # SR vs AutoGluon for IV
    ag_path = OUTPUT_ROOT / "exp1_forward" / "metrics_model2.csv"
    if ag_path.exists():
        ag = pd.read_csv(ag_path)
        ag_test = ag[(ag["split"] == "test") & (ag["target"].isin(IV_TARGETS))][["target", "R2"]]
        ag_test = ag_test.rename(columns={"R2": "autogluon_R2"})
        iv_sr = df[df["task_type"].isin(["iv_full", "iv_identity"])][["target", "R2"]].rename(columns={"R2": "sr_R2"})
        cmp = iv_sr.merge(ag_test, on="target", how="outer")
        x = np.arange(len(cmp))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w / 2, cmp["sr_R2"], w, label="SR (FF=identity)", color="darkorange")
        ax.bar(x + w / 2, cmp["autogluon_R2"], w, label="AutoGluon model2", color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels([label_for(t) for t in cmp["target"]], rotation=15, ha="right")
        ax.set_ylabel("test R²")
        ax.set_title("IV prediction: SR vs exp1 forward (Model2)")
        ax.legend()
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(PLOT_OUT / "iv_sr_vs_autogluon_r2.png", dpi=150)
        plt.close(fig)
        save_csv(cmp, PLOT_OUT / "iv_sr_vs_autogluon_r2.csv")

    # FF methods MAE
    ff_cmp = EXP4_OUT / "sr_ff_comparison.csv"
    detail = EXP4_OUT / "sr_ff_from_predicted_iv_metrics.csv"
    if ff_cmp.exists():
        fdf = pd.read_csv(ff_cmp)
        labels = {
            "identity_FF_calc": "Identity (measured IV)",
            "sr_iv_trio": "SR(measured IV->FF)",
            "FF_from_SR_Voc_Jsc_Eff": "SR pred IV + identity",
            "sr_full_features": "SR full->FF",
        }
        fdf["label"] = fdf["method"].map(labels).fillna(fdf["method"])
        fig, ax = plt.subplots(figsize=(8, 4))
        if "test_MAE" in fdf.columns:
            ycol = "test_MAE"
        else:
            fdf["test_MAE"] = np.nan
            ycol = "test_MAE"
        if detail.exists():
            ddf = pd.read_csv(detail)
            rob = ddf[(ddf["subset"] == "robust_ff_calc_75_95") & (ddf["split"] == "test")]
            if not rob.empty:
                fdf.loc[fdf["method"] == "FF_from_SR_Voc_Jsc_Eff", "test_MAE"] = rob.iloc[0]["MAE"]
        vals = fdf[ycol].fillna(0)
        ax.bar(fdf["label"], vals, color=["gray", "lightgray", "darkorange", "salmon"][: len(fdf)])
        ax.set_ylabel("test MAE")
        ax.set_title("FF prediction methods (lower MAE is better)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(PLOT_OUT / "ff_methods_comparison.png", dpi=150)
        plt.close(fig)

    # Ablation MAE
    abl_path = EXP4_OUT / "sr_ff_from_predicted_iv_ablation.csv"
    if abl_path.exists():
        abl = pd.read_csv(abl_path)
        abl = abl[abl["split"] == "test"]
        name_map = {
            "all_SR_pred": "All SR pred",
            "true_Voc_Jsc_pred_Eff": "Eff pred only",
            "pred_Voc_true_Jsc_Eff": "Voc pred only",
            "pred_Jsc_true_Voc_Eff": "Jsc pred only",
            "pred_Voc_Jsc_true_Eff": "Voc+Jsc pred",
        }
        abl["label"] = abl["method"].map(name_map)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(abl["label"], abl["MAE"], color="teal", alpha=0.8)
        ax.set_ylabel("test MAE (%)")
        ax.set_title("FF error propagation ablation")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(PLOT_OUT / "ff_ablation_mae.png", dpi=150)
        plt.close(fig)


def main() -> None:
    setup_runtime()
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    log(f"SR accuracy plots -> {PLOT_OUT}")

    all_rows: list[dict] = []
    all_rows.extend(plot_tabular_scatters())
    all_rows.extend(plot_iv_ff_identity())
    all_rows.extend(plot_curve_scatters_and_overlays())
    if all_rows:
        save_csv(pd.DataFrame(all_rows), PLOT_OUT / "sr_scatter_metrics.csv")
    plot_summary_figures(all_rows)
    log("SR accuracy plots done.")


if __name__ == "__main__":
    main()
