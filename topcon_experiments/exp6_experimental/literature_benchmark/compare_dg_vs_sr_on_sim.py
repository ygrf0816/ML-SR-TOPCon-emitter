"""Compare literature double-Gaussian vs the 5 SR curve formulas on the SIM set.

Same evaluation protocol as fit_double_gaussian_to_sim.py: for each simulated
doping curve, evaluate the model on the N >= 1e18 segment and report R²_log
(in decades) and RMSE_log. This apples-to-apples comparison tells us whether
the SR formulas (trained on the same sim set) actually beat the 4-parameter
literature double-Gaussian, and where each wins.

Outputs (under outputs/exp6_experimental/dg_vs_sr_on_sim/):
  * data/dg_vs_sr_r2.csv     (file_base x model x r2_log / rmse_log / n_fit)
  * plots/dg_vs_sr_r2_box.png  (R² distribution per model)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import LOG_FEATURES, MODEL2_FEATURES
from topcon_experiments.exp4_symbolic.sr_equation_utils import eval_sympy_expr, sort_formulas_by_accuracy
from topcon_experiments.exp6_experimental.literature_benchmark.fit_double_gaussian_to_sim import (
    BOUNDS,
    N_FLOOR,
    double_gaussian,
    fit_one,
)
from topcon_experiments.exp6_experimental.literature_benchmark.fit_sr_to_literature import (
    VARIANTS,
    CurveVariant,
    _best_expr,
    _full_feature_order,
    _legacy_feature_order,
)

CURVE_CSV = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "exp4_symbolic"
    / "curve_processed_doping.csv"
)
EXP_OUT = Path(__file__).resolve().parents[1] / "outputs" / "dg_vs_sr_on_sim"
DATA_OUT = EXP_OUT / "data"
PLOT_OUT = EXP_OUT / "plots"

LOG_IN_MODEL = set(LOG_FEATURES)


def _sr_predict_curve(
    expr: str, feat_row: pd.Series, z: np.ndarray, variant: CurveVariant
) -> np.ndarray:
    """Predict ln(N) at depths z using one SR formula + this sample's features."""
    order = variant.feature_order
    n = len(z)
    X = np.zeros((n, len(order)), dtype=float)
    for j, col in enumerate(order[:-1]):
        v = float(feat_row[col])
        if col in LOG_IN_MODEL:
            v = np.log(max(v, 1e-30))
        X[:, j] = v
    X[:, -1] = z + variant.depth_offset_um
    ln_pred = eval_sympy_expr(expr, X)
    ln_pred = np.where(np.isfinite(ln_pred), ln_pred, np.nan)
    return np.exp(np.clip(ln_pred, -700, 700))


def _r2_log(N_obs: np.ndarray, N_pred: np.ndarray) -> float:
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() < 4:
        return float("nan")
    y = np.log10(N_obs[mask])
    yhat = np.log10(N_pred[mask])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse_log(N_obs: np.ndarray, N_pred: np.ndarray) -> float:
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.log10(N_obs[mask]) - np.log10(N_pred[mask])) ** 2)))


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(CURVE_CSV)
    meta = load_raw_dataframe().set_index("file_base")
    # cache SR best expressions
    sr_exprs: dict[str, str] = {}
    for label, v in VARIANTS.items():
        if v.formula_path.exists():
            try:
                sr_exprs[label] = _best_expr(v.formula_path)
            except Exception as e:
                print(f"[warn] {label}: cannot load formula: {e}")

    rows = []
    for fb in sorted(curves["file_base"].unique()):
        sub = curves[curves["file_base"] == fb].sort_values("depth_um")
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        keep = N >= N_FLOOR
        z_f, N_f = z[keep], N[keep]
        if z_f.size < 6:
            continue
        feat_row = meta.loc[fb] if fb in meta.index else None

        # double-Gaussian
        dg = fit_one(z, N)
        if "error" not in dg:
            dg_pred = double_gaussian(z_f, dg["N_p_cm3"], dg["z_p_um"], dg["z_f1"], dg["z_f2"])
            rows.append({
                "file_base": fb, "model": "double_gaussian",
                "r2_log": _r2_log(N_f, dg_pred),
                "rmse_log": _rmse_log(N_f, dg_pred),
                "n_fit": int(z_f.size),
            })

        # SR variants
        if feat_row is not None:
            for label, v in VARIANTS.items():
                if label not in sr_exprs:
                    continue
                try:
                    N_pred = _sr_predict_curve(sr_exprs[label], feat_row, z_f, v)
                except Exception as e:
                    rows.append({"file_base": fb, "model": label, "error": str(e),
                                 "n_fit": int(z_f.size)})
                    continue
                rows.append({
                    "file_base": fb, "model": label,
                    "r2_log": _r2_log(N_f, N_pred),
                    "rmse_log": _rmse_log(N_f, N_pred),
                    "n_fit": int(z_f.size),
                })

    df = pd.DataFrame(rows)
    save_csv(df, DATA_OUT / "dg_vs_sr_r2.csv")

    # summary table
    ok = df[df["r2_log"].notna()]
    summary = ok.groupby("model")["r2_log"].agg(
        ["count", "median", "mean", "min", "max",
         lambda s: float((s > 0.9).sum()),
         lambda s: float((s > 0.95).sum())]
    ).rename(columns={"<lambda_0>": "n_r2>0.9", "<lambda_1>": "n_r2>0.95"})
    print(summary.to_string())
    summary.to_csv(DATA_OUT / "dg_vs_sr_summary.csv")

    # box plot
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    models = ["double_gaussian"] + [l for l in VARIANTS.keys() if l in ok["model"].unique()]
    data = [ok[ok["model"] == m]["r2_log"].dropna().values for m in models]
    bp = ax.boxplot(data, labels=models, showfliers=True, patch_artist=True)
    cmap = plt.get_cmap("tab10")
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i % 10))
        patch.set_alpha(0.6)
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
    ax.axhline(0.9, color="0.7", ls=":", lw=0.7)
    ax.set_ylabel("R²_log (decades, on N>=1e18 segment)")
    ax.set_title("Double-Gaussian vs SR curve formulas on the sim training set")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = PLOT_OUT / "dg_vs_sr_r2_box.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Box plot -> {out}")
    print(f"Table -> {DATA_OUT / 'dg_vs_sr_r2.csv'}")


if __name__ == "__main__":
    main()
