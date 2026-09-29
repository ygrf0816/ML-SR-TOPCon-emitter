"""Fully-symbolic chain evaluation on the simulated test set.

Chain: process params --(theta SR formulas)--> (ln N_p, z_p, z_f1, z_f2)
       --> pure DG curve, and DG x exp(r_SR) curve.

Both variants are scored on ALL held-out test samples (R^2_log on the
BSG-excluded, N>=1e18 in-silicon window; same protocol as everywhere else).
The 6 lowest-RMSE test cases are exported as example plots + Origin CSV.

Outputs (under outputs/exp4_symbolic/process_to_theta_sr/):
  * data/symbolic_chain_eval.csv        per-test-sample metrics
  * data/symbolic_chain_summary.csv     aggregate stats
  * data/origin_symbolic_best6_curves.csv / _metadata.csv
  * plots/symbolic_chain_best6.png
"""

from __future__ import annotations

import json
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

from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.exp4_symbolic.run_process_to_dg_chain import (
    DG_TARGETS,
    EXT_TRIM_CSV,
    RESIDUAL_FORMULA_CSV,
    _load_residual_expr,
    _predict_curve,
    _r2_log,
)
from topcon_experiments.exp4_symbolic.run_process_to_theta_sr import OUT_DIR, build_xy
from topcon_experiments.exp4_symbolic.sr_equation_utils import (
    eval_sympy_expr,
    sort_formulas_by_accuracy,
)
from topcon_experiments.exp6_experimental.literature_benchmark.fit_double_gaussian_to_sim import (
    N_FLOOR,
)

DATA_OUT = OUT_DIR / "data"
PLOT_OUT = OUT_DIR / "plots"
N_BEST = 6


def _rmse_log(N_obs: np.ndarray, N_pred: np.ndarray) -> float:
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.log10(N_obs[mask]) - np.log10(N_pred[mask])) ** 2)))


def _load_theta_formulas() -> dict[str, str]:
    out = {}
    for tgt in DG_TARGETS:
        df = pd.read_csv(OUT_DIR / f"sr_formulas_theta_{tgt}.csv")
        best = sort_formulas_by_accuracy(df).iloc[0]
        out[tgt] = str(best.get("sympy_format") or best["equation"])
    return out


def main() -> None:
    setup_runtime()
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)

    X, y, train_fb, test_fb = build_xy()
    theta_expr = _load_theta_formulas()
    resid_expr = _load_residual_expr()
    curves = pd.read_csv(EXT_TRIM_CSV)

    # Predict theta for the whole test set with the SR formulas.
    Xt = X.loc[test_fb].values.astype(float)
    theta_pred = {t: eval_sympy_expr(theta_expr[t], Xt) for t in DG_TARGETS}
    theta_r2 = {
        t: 1.0 - np.nansum((y.loc[test_fb, t].values - theta_pred[t]) ** 2)
        / np.sum((y.loc[test_fb, t].values - y.loc[test_fb, t].values.mean()) ** 2)
        for t in DG_TARGETS
    }
    log("theta SR formulas test R2: "
        + ", ".join(f"{t}={theta_r2[t]:.3f}" for t in DG_TARGETS))

    rows = []
    curve_cache: dict[str, dict[str, np.ndarray]] = {}
    for i, fb in enumerate(test_fb):
        sub = curves[curves["file_base"] == fb].sort_values("depth_um")
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        keep = N >= N_FLOOR
        z_f, N_f = z[keep], N[keep]
        if z_f.size < 6:
            continue
        params = (float(np.exp(theta_pred["ln_N_p"][i])), float(theta_pred["z_p_um"][i]),
                  float(theta_pred["z_f1"][i]), float(theta_pred["z_f2"][i]))
        if not all(np.isfinite(params)):
            continue
        N_dg = _predict_curve(None, z_f, *params)
        N_chain = _predict_curve(resid_expr, z_f, *params)
        rows.append({
            "file_base": fb,
            "r2_dg_symbolic": _r2_log(N_f, N_dg),
            "rmse_dg_symbolic": _rmse_log(N_f, N_dg),
            "r2_chain_symbolic": _r2_log(N_f, N_chain),
            "rmse_chain_symbolic": _rmse_log(N_f, N_chain),
            "n_fit": int(z_f.size),
            "N_p_pred": params[0], "z_p_pred": params[1],
            "z_f1_pred": params[2], "z_f2_pred": params[3],
        })
        curve_cache[str(fb)] = {
            "true_x": z_f, "true_y": N_f, "dg_y": N_dg, "chain_y": N_chain,
        }
    df = pd.DataFrame(rows)
    save_csv(df, DATA_OUT / "symbolic_chain_eval.csv")

    summary = []
    for col in ("r2_dg_symbolic", "r2_chain_symbolic"):
        s = df[col].dropna()
        summary.append({
            "model": col, "count": len(s), "median": s.median(), "mean": s.mean(),
            "min": s.min(), "n_gt_09": int((s > 0.9).sum()),
            "n_gt_095": int((s > 0.95).sum()),
        })
    sdf = pd.DataFrame(summary)
    save_csv(sdf, DATA_OUT / "symbolic_chain_summary.csv")
    print("Fully-symbolic chain on test set (R2_log, N>=1e18 window):")
    print(sdf.to_string(index=False))

    # Best 6 test cases by chained RMSE. Degenerate near-flat windows (very few
    # points barely above the floor) can have tiny RMSE but meaningless shape;
    # require a reasonably wide fit window first.
    eligible = df[df["n_fit"] >= 30]
    best = eligible.sort_values("rmse_chain_symbolic").head(N_BEST).reset_index(drop=True)
    apply_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    wide: dict[str, pd.Series] = {}
    meta_rows = []
    for rank, (ax, (_, row)) in enumerate(zip(axes.ravel(), best.iterrows()), start=1):
        fb = str(row["file_base"])
        cur = curve_cache[fb]
        ax.semilogy(cur["true_x"], cur["true_y"], "ko", ms=3, alpha=0.55, label="sim (in-Si)")
        ax.semilogy(cur["true_x"], cur["dg_y"], "-", color="C3", lw=1.6,
                    label=f"DG(θ_SR) R²={row['r2_dg_symbolic']:.3f}")
        ax.semilogy(cur["true_x"], cur["chain_y"], "--", color="C0", lw=1.6,
                    label=f"DG x SR-resid R²={row['r2_chain_symbolic']:.3f}")
        ax.set_title(fb[-16:], fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        prefix = f"curve{rank:02d}"
        wide[f"{prefix}_true_x"] = pd.Series(cur["true_x"])
        wide[f"{prefix}_true_y"] = pd.Series(cur["true_y"])
        wide[f"{prefix}_dg_x"] = pd.Series(cur["true_x"])
        wide[f"{prefix}_dg_y"] = pd.Series(cur["dg_y"])
        wide[f"{prefix}_pred_x"] = pd.Series(cur["true_x"])
        wide[f"{prefix}_pred_y"] = pd.Series(cur["chain_y"])
        meta_rows.append({
            "curve_id": prefix, "file_base": fb,
            "r2_dg_symbolic": row["r2_dg_symbolic"],
            "r2_chain_symbolic": row["r2_chain_symbolic"],
            "rmse_chain_symbolic": row["rmse_chain_symbolic"],
            "n_fit": row["n_fit"],
        })
    fig.suptitle("Fully symbolic chain: process params -> θ formulas -> curve "
                 "(best 6 of test set)")
    fig.supxlabel("Depth (um)")
    fig.supylabel("Boron concentration (cm-3)")
    fig.tight_layout()
    out_png = PLOT_OUT / "symbolic_chain_best6.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    save_csv(pd.DataFrame(wide), DATA_OUT / "origin_symbolic_best6_curves.csv")
    save_csv(pd.DataFrame(meta_rows), DATA_OUT / "origin_symbolic_best6_metadata.csv")
    log(f"Best-6 plot -> {out_png}")
    log(f"Done -> {DATA_OUT}")


if __name__ == "__main__":
    main()
