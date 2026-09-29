"""Figure-3 extension panels: process-linked & forward/inverse extras.

Produces standalone draft panels under ``outputs/fig3_extension/`` for
selecting which to fold into the paper Fig.3 layout.

Panels
------
A1  High- vs low-efficiency process-parameter distributions (2x4 violin)
A2  Parallel coordinates of process params colored by efficiency tier
B1  Representative GBM train/test RMSE vs boosting iteration (iv_Eff)
B2  Learning curve: test R2 / RMSE vs training-set fraction
C1  Feature importance bar chart (AutoGluon permutation, model2 iv_Eff)
C2  DE-optimal process params as z-score deviations from dataset mean
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe, preprocess_model2
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import (
    ATHENA_FEATURES,
    IV_TARGET_LABELS,
    IV_TARGETS,
    OUTPUT_ROOT,
    RANDOM_STATE,
)

OUT = OUTPUT_ROOT / "fig3_extension"
EXP1 = OUTPUT_ROOT / "exp1_forward"
EXP2 = OUTPUT_ROOT / "exp2_inverse"

# Short axis labels for compact panels
SHORT_LABELS = {
    "athena_thick": "BSG thick (μm)",
    "athena_c_boron": "ln(c_B)",
    "athena_temp1": "T1 (°C)",
    "athena_time1": "t1 (min)",
    "athena_temp2": "T2 (°C)",
    "athena_time2": "t2 (min)",
    "athena_F_N2": "F_N2",
    "athena_F_O2": "F_O2",
}

HIGH_Q, LOW_Q = 0.90, 0.10
COLOR_HIGH = "#2ca02c"
COLOR_LOW = "#d62728"
COLOR_PROC = "#1f77b4"
COLOR_DESC = "#ff7f0e"
DE_COLORS = {
    "iv_Eff": "#d62728",
    "iv_Voc": "#1f77b4",
    "iv_Jsc": "#2ca02c",
    "iv_FF": "#9467bd",
}


def _save_fig(fig: plt.Figure, stem: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  -> {path}")
    return path


def _process_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Athena features with log transform on concentration."""
    X = df[ATHENA_FEATURES].astype(float).copy()
    if "athena_c_boron" in X.columns:
        X["athena_c_boron"] = np.log(X["athena_c_boron"].clip(lower=1e-30))
    return X


# ---------------------------------------------------------------------------
# A1 — high vs low efficiency process distributions
# ---------------------------------------------------------------------------
def panel_high_low_distributions(df: pd.DataFrame) -> None:
    log("A1: high/low efficiency process distributions")
    eff = df["iv_Eff"].astype(float)
    hi_thr, lo_thr = eff.quantile(HIGH_Q), eff.quantile(LOW_Q)
    hi = df.loc[eff >= hi_thr].copy()
    lo = df.loc[eff <= lo_thr].copy()
    X_hi = _process_matrix(hi)
    X_lo = _process_matrix(lo)

    rows = []
    for col in ATHENA_FEATURES:
        for val in X_hi[col].values:
            rows.append({"param": col, "group": "high_eff", "value": float(val)})
        for val in X_lo[col].values:
            rows.append({"param": col, "group": "low_eff", "value": float(val)})
    long = pd.DataFrame(rows)
    save_csv(long, OUT / "A1_high_low_process_long.csv")
    save_csv(
        pd.DataFrame(
            {
                "group": ["high_eff", "low_eff"],
                "quantile": [HIGH_Q, LOW_Q],
                "eff_threshold": [hi_thr, lo_thr],
                "n": [len(hi), len(lo)],
                "eff_mean": [hi["iv_Eff"].mean(), lo["iv_Eff"].mean()],
            }
        ),
        OUT / "A1_high_low_meta.csv",
    )

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))
    for ax, col in zip(axes.ravel(), ATHENA_FEATURES):
        data = [X_lo[col].values, X_hi[col].values]
        parts = ax.violinplot(
            data, positions=[0, 1], showmeans=True, showmedians=False, widths=0.8
        )
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(COLOR_LOW if i == 0 else COLOR_HIGH)
            body.set_alpha(0.55)
            body.set_edgecolor("k")
        for key in ("cbars", "cmins", "cmaxes", "cmeans"):
            if key in parts:
                parts[key].set_color("k")
                parts[key].set_linewidth(0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low 10%", "High 10%"], fontsize=8)
        ax.set_title(SHORT_LABELS.get(col, col), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"Process params: low vs high PCE cohorts "
        f"(n_low={len(lo)}, n_high={len(hi)}; "
        f"thresholds {lo_thr:.2f} / {hi_thr:.2f} %)",
        fontsize=11,
    )
    _save_fig(fig, "A1_high_low_process_violin")


# ---------------------------------------------------------------------------
# A2 — parallel coordinates
# ---------------------------------------------------------------------------
def panel_parallel_coordinates(df: pd.DataFrame) -> None:
    log("A2: parallel coordinates by efficiency tier")
    X = _process_matrix(df)
    # min-max normalize to [0, 1]
    Xn = (X - X.min()) / (X.max() - X.min() + 1e-30)
    eff = df["iv_Eff"].astype(float).values
    # subsample for readability
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Xn), size=min(1500, len(Xn)), replace=False)
    Xn = Xn.iloc[idx]
    eff = eff[idx]

    # tier colors: low / mid / high
    q_lo, q_hi = np.quantile(eff, [LOW_Q, HIGH_Q])
    colors = np.where(
        eff >= q_hi, COLOR_HIGH, np.where(eff <= q_lo, COLOR_LOW, "#bbbbbb")
    )
    alphas = np.where((eff >= q_hi) | (eff <= q_lo), 0.55, 0.08)

    long_rows = []
    for i, (_, row) in enumerate(Xn.iterrows()):
        for j, col in enumerate(ATHENA_FEATURES):
            long_rows.append(
                {
                    "sample_i": int(i),
                    "param": col,
                    "param_idx": j,
                    "value_norm": float(row[col]),
                    "iv_Eff": float(eff[i]),
                    "tier": (
                        "high"
                        if eff[i] >= q_hi
                        else ("low" if eff[i] <= q_lo else "mid")
                    ),
                }
            )
    save_csv(pd.DataFrame(long_rows), OUT / "A2_parallel_coords_long.csv")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = np.arange(len(ATHENA_FEATURES))
    # draw mid first (background), then low/high
    order = np.argsort(alphas)  # low alpha first
    for k in order:
        ax.plot(
            xs,
            Xn.iloc[k].values,
            color=colors[k],
            alpha=float(alphas[k]),
            lw=0.7,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [SHORT_LABELS.get(c, c) for c in ATHENA_FEATURES], rotation=20, ha="right"
    )
    ax.set_ylabel("Min–max normalized value")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Parallel coordinates of process params by PCE tier")
    # legend proxies
    for lab, c in [("Low 10%", COLOR_LOW), ("Mid", "#bbbbbb"), ("High 10%", COLOR_HIGH)]:
        ax.plot([], [], color=c, lw=2, label=lab)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "A2_parallel_coordinates")


# ---------------------------------------------------------------------------
# B1 — GBM train/test RMSE vs iterations
# ---------------------------------------------------------------------------
def panel_training_loss() -> None:
    log("B1: representative GBM train/test RMSE vs iterations")
    X, y_dict, _ = preprocess_model2()
    y = y_dict["iv_Eff"].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # sklearn GradientBoostingRegressor exposes staged_predict for loss curves.
    # Caption must note this is a representative GBM surrogate, not AutoGluon.
    max_iter = 300
    model = GradientBoostingRegressor(
        n_estimators=max_iter,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr, y_tr)

    rows = []
    for i, (p_tr, p_te) in enumerate(
        zip(model.staged_predict(X_tr), model.staged_predict(X_te)), start=1
    ):
        if i % 5 != 0 and i != max_iter:
            continue
        rows.append(
            {
                "iteration": i,
                "train_RMSE": float(np.sqrt(mean_squared_error(y_tr, p_tr))),
                "test_RMSE": float(np.sqrt(mean_squared_error(y_te, p_te))),
                "train_R2": float(r2_score(y_tr, p_tr)),
                "test_R2": float(r2_score(y_te, p_te)),
            }
        )

    curve = pd.DataFrame(rows)
    save_csv(curve, OUT / "B1_gbm_train_test_loss.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        curve["iteration"].to_numpy(),
        curve["train_RMSE"].to_numpy(),
        label="Train RMSE",
        lw=2,
        color=COLOR_PROC,
    )
    ax.plot(
        curve["iteration"].to_numpy(),
        curve["test_RMSE"].to_numpy(),
        label="Test RMSE",
        lw=2,
        color=COLOR_LOW,
    )
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("RMSE (PCE, %)")
    ax.set_title(
        "Representative GBM learning curve (process+descriptors → PCE)\n"
        "Note: surrogate for AutoGluon ensemble; not the production model"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, "B1_gbm_train_test_loss")


# ---------------------------------------------------------------------------
# B2 — learning curve vs sample size
# ---------------------------------------------------------------------------
def panel_learning_curve_n() -> None:
    log("B2: learning curve vs training-set size")
    X, y_dict, _ = preprocess_model2()
    y = y_dict["iv_Eff"].astype(float)
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    fracs = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for frac in fracs:
        n = max(50, int(len(X_tr_full) * frac))
        X_tr = X_tr_full.iloc[:n]
        y_tr = y_tr_full.iloc[:n]
        m = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=max(5, min(20, n // 50)),
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )
        m.fit(X_tr, y_tr)
        p_te = m.predict(X_te)
        rows.append(
            {
                "train_fraction": frac,
                "n_train": n,
                "test_RMSE": float(np.sqrt(mean_squared_error(y_te, p_te))),
                "test_R2": float(r2_score(y_te, p_te)),
            }
        )
    lc = pd.DataFrame(rows)
    save_csv(lc, OUT / "B2_learning_curve_n.csv")

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(
        lc["n_train"].to_numpy(),
        lc["test_R2"].to_numpy(),
        "o-",
        color=COLOR_HIGH,
        lw=2,
        label="Test $R^2$",
    )
    ax1.set_xlabel("Training set size")
    ax1.set_ylabel("Test $R^2$", color=COLOR_HIGH)
    ax1.tick_params(axis="y", labelcolor=COLOR_HIGH)
    ax1.set_ylim(min(0.5, float(lc["test_R2"].min()) - 0.05), 1.01)

    ax2 = ax1.twinx()
    ax2.plot(
        lc["n_train"].to_numpy(),
        lc["test_RMSE"].to_numpy(),
        "s--",
        color=COLOR_LOW,
        lw=2,
        label="Test RMSE",
    )
    ax2.set_ylabel("Test RMSE (PCE, %)", color=COLOR_LOW)
    ax2.tick_params(axis="y", labelcolor=COLOR_LOW)

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("Learning curve vs data size (HistGBM → PCE)")
    ax1.grid(alpha=0.3)
    _save_fig(fig, "B2_learning_curve_n")


# ---------------------------------------------------------------------------
# C1 — feature importance
# ---------------------------------------------------------------------------
def panel_feature_importance() -> None:
    log("C1: feature importance (model2 iv_Eff)")
    src = EXP1 / "feature_importance_model2_iv_Eff.csv"
    fi = pd.read_csv(src).sort_values("importance", ascending=True)
    save_csv(fi, OUT / "C1_feature_importance_iv_Eff.csv")

    colors = [
        COLOR_PROC if f.startswith("athena_") else COLOR_DESC for f in fi["feature"]
    ]
    labels = []
    for f in fi["feature"]:
        if f in SHORT_LABELS:
            labels.append(SHORT_LABELS[f])
        elif f.startswith("doping_"):
            labels.append(f.replace("doping_", "dop."))
        elif f.startswith("defect_"):
            labels.append(f.replace("defect_vac_", "vac.").replace("defect_", "def."))
        else:
            labels.append(f)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(labels, fi["importance"].values, color=colors, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Permutation importance")
    ax.set_title("Model-2 feature importance for PCE (AutoGluon)")
    ax.grid(axis="x", alpha=0.3)
    # legend
    ax.barh([], [], color=COLOR_PROC, label="Process (athena)")
    ax.barh([], [], color=COLOR_DESC, label="Curve descriptor")
    ax.legend(loc="lower right")
    _save_fig(fig, "C1_feature_importance_iv_Eff")


# ---------------------------------------------------------------------------
# C2 — DE optimal parameter deviation (lollipop / z-score)
# ---------------------------------------------------------------------------
def panel_optimal_deviation(df: pd.DataFrame) -> None:
    log("C2: DE-optimal process params as z-score deviations")
    X = _process_matrix(df)
    mu = X.mean()
    sigma = X.std(ddof=0).replace(0, np.nan)

    rows = []
    for metric in IV_TARGETS:
        path = EXP2 / f"de_best_{metric}.csv"
        if not path.exists():
            log(f"  skip missing {path.name}")
            continue
        best = pd.read_csv(path).iloc[0]
        for col in ATHENA_FEATURES:
            raw = float(best[col])
            # match log transform used for dataset stats
            val = np.log(raw) if col == "athena_c_boron" else raw
            z = (val - float(mu[col])) / float(sigma[col])
            rows.append(
                {
                    "metric": metric,
                    "metric_label": IV_TARGET_LABELS.get(metric, metric),
                    "param": col,
                    "param_label": SHORT_LABELS.get(col, col),
                    "value_raw": raw,
                    "value_transformed": float(val),
                    "dataset_mean": float(mu[col]),
                    "dataset_std": float(sigma[col]),
                    "z_score": float(z),
                    "percentile": float((X[col] <= val).mean() * 100.0),
                }
            )
    dev = pd.DataFrame(rows)
    save_csv(dev, OUT / "C2_optimal_param_zscore.csv")

    # focus on PCE-optimal as main lollipop; overlay others as markers
    params = ATHENA_FEATURES
    y_pos = np.arange(len(params))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axvspan(-1, 1, color="0.85", alpha=0.5, label="±1σ band")
    ax.axvline(0, color="k", lw=1.0, ls="--")

    # PCE main lollipop
    sub = dev[dev["metric"] == "iv_Eff"].set_index("param")
    zs = [float(sub.loc[p, "z_score"]) for p in params]
    ax.hlines(y_pos, 0, zs, color=DE_COLORS["iv_Eff"], lw=2.0, alpha=0.85)
    ax.scatter(
        zs,
        y_pos,
        s=80,
        color=DE_COLORS["iv_Eff"],
        zorder=5,
        label=f"DE opt. {IV_TARGET_LABELS['iv_Eff']}",
        edgecolors="k",
        linewidths=0.5,
    )

    # other metrics as open markers offset slightly in y
    offsets = {"iv_Voc": -0.18, "iv_Jsc": 0.0, "iv_FF": 0.18}
    markers = {"iv_Voc": "s", "iv_Jsc": "D", "iv_FF": "^"}
    for metric in ("iv_Voc", "iv_Jsc", "iv_FF"):
        sub_m = dev[dev["metric"] == metric].set_index("param")
        if sub_m.empty:
            continue
        zs_m = [float(sub_m.loc[p, "z_score"]) for p in params]
        ax.scatter(
            zs_m,
            y_pos + offsets[metric],
            s=45,
            marker=markers[metric],
            facecolors="none",
            edgecolors=DE_COLORS[metric],
            linewidths=1.4,
            zorder=4,
            label=f"DE opt. {IV_TARGET_LABELS[metric]}",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT_LABELS.get(p, p) for p in params])
    ax.set_xlabel("Z-score vs dataset mean (log c_B)")
    ax.set_title("DE-optimal process recipes relative to dataset distribution")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(
        min(-3.5, min(dev["z_score"]) - 0.3),
        max(3.5, max(dev["z_score"]) + 0.3),
    )
    _save_fig(fig, "C2_optimal_param_deviation")


# ---------------------------------------------------------------------------
# README summary for layout discussion
# ---------------------------------------------------------------------------
def write_readme(paths: list[str]) -> None:
    text = f"""# Fig.3 extension drafts

Generated by `plot_fig3_extension.py`. Each panel is a standalone PNG + CSV
for Origin / paper selection.

## Panels

| ID | File | Content |
|----|------|---------|
| A1 | `A1_high_low_process_violin.png` | Low vs high PCE (10%) process-param violins (2×4) |
| A2 | `A2_parallel_coordinates.png` | Parallel coords of 8 athena params by PCE tier |
| B1 | `B1_gbm_train_test_loss.png` | HistGBM train/test RMSE vs boosting iteration |
| B2 | `B2_learning_curve_n.png` | Test R² / RMSE vs training-set size |
| C1 | `C1_feature_importance_iv_Eff.png` | AutoGluon permutation importance (model2 PCE) |
| C2 | `C2_optimal_param_deviation.png` | DE-optimal params as z-score vs dataset mean |

## Notes for layout

- **A1 / A2** link Fig.3 to Fig.2-style distribution analysis via process params.
- **B1** is a *representative* HistGBM curve (same features/target as Model-2 PCE),
  not the AutoGluon ensemble itself — caption must say so.
- **B2** shows data-sufficiency; saturates quickly → supports using full cohort.
- **C1** bridges forward model interpretability and process knobs.
- **C2** connects inverse-design optima to the process distribution (Fig.2 link).

## Suggested layouts (for discussion)

1. **Split Fig.3a / 3b**: keep existing 4 scatter + 4 traces as 3a;
   pick A1 + C1 + C2 (+ optional B2) as 3b process/interpretability.
2. **3×4 grid**: replace two weaker traces with A1 (span 2 cols) + C2.
3. **Keep 8 + add 2**: only append C2 and A1 as (i)(j) if page space allows.

Generated files:
{chr(10).join('- ' + p for p in paths)}
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")
    log(f"README -> {OUT / 'README.md'}")


def main() -> None:
    setup_runtime()
    apply_plot_style()
    OUT.mkdir(parents=True, exist_ok=True)

    log("Loading dataset ...")
    df = load_raw_dataframe()
    # drop rows with missing process / efficiency
    need = ATHENA_FEATURES + ["iv_Eff"]
    df = df.dropna(subset=need).copy()
    # positive c_boron for log
    df = df.loc[df["athena_c_boron"] > 0].copy()
    log(f"  n={len(df)} after dropna")

    panel_high_low_distributions(df)
    panel_parallel_coordinates(df)
    panel_training_loss()
    panel_learning_curve_n()
    panel_feature_importance()
    panel_optimal_deviation(df)

    paths = sorted(p.name for p in OUT.glob("*.png"))
    write_readme(paths)
    log(f"Done -> {OUT} ({len(paths)} PNGs)")


if __name__ == "__main__":
    main()
