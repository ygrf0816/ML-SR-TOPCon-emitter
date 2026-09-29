"""Doping error-analysis figures for the paper package (no model retraining).

Reads existing six-variant test metrics and regenerates overlays for best/worst
cases. Writes under outputs/paper_package_defect_and_doping_extras/01_doping_error_analysis/.
"""

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

from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import OUTPUT_ROOT
from topcon_experiments.exp4_symbolic.run_process_to_dg_chain import (
    DG_EXT_CSV,
    DG_TARGETS,
    EXT_TRIM_CSV,
    _load_residual_expr,
    _predict_curve,
    _r2_log,
)
from topcon_experiments.exp4_symbolic.run_process_to_theta_sr import OUT_DIR as THETA_SR_DIR
from topcon_experiments.exp4_symbolic.run_process_to_theta_sr import build_xy
from topcon_experiments.exp4_symbolic.sr_equation_utils import (
    eval_sympy_expr,
    sort_formulas_by_accuracy,
)
from topcon_experiments.exp6_experimental.literature_benchmark.fit_double_gaussian_to_sim import (
    N_FLOOR,
)

PKG = OUTPUT_ROOT / "paper_package_defect_and_doping_extras"
OUT = PKG / "01_doping_error_analysis"
PLOT = OUT / "plots"
ORIGIN = OUT / "origin_csv"
SUM = OUT / "summaries"
SIX_PER = (
    OUTPUT_ROOT / "paper_package_six_variants" / "04_summaries" / "simulation_test_per_sample.csv"
)

VARIANT_ORDER = [
    "dg_selffit", "dgresid_selffit", "dg_sr", "dgresid_sr", "dg_ag", "dgresid_ag",
]
VARIANT_LABEL = {
    "dg_selffit": "DG self-fit",
    "dgresid_selffit": "DG x resid self-fit",
    "dg_sr": "DG + SR theta",
    "dgresid_sr": "DG x resid + SR",
    "dg_ag": "DG + AG theta",
    "dgresid_ag": "DG x resid + AG",
}
COLORS = {
    "dg_selffit": "C3", "dgresid_selffit": "C1",
    "dg_sr": "C2", "dgresid_sr": "C4",
    "dg_ag": "C0", "dgresid_ag": "C5",
}


def _rmse_log(N_obs, N_pred):
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.log10(N_obs[mask]) - np.log10(N_pred[mask])) ** 2)))


def _load_theta_sr():
    out = {}
    for tgt in DG_TARGETS:
        df = pd.read_csv(THETA_SR_DIR / f"sr_formulas_theta_{tgt}.csv")
        best = sort_formulas_by_accuracy(df).iloc[0]
        out[tgt] = str(best.get("sympy_format") or best["equation"])
    return out


def _load_ag(X, test_fb):
    from autogluon.tabular import TabularPredictor
    pred = {}
    for tgt in DG_TARGETS:
        pred[tgt] = TabularPredictor.load(
            str(THETA_SR_DIR / "autogluon_models" / tgt)
        ).predict(X.loc[test_fb]).values.astype(float)
    return pred


def _boxplot(df: pd.DataFrame) -> None:
    apply_plot_style()
    cols = [f"r2_{k}" for k in VARIANT_ORDER]
    data = [df[c].dropna().clip(-1, 1.05).values for c in cols]  # clip for display
    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(data, labels=[VARIANT_LABEL[k] for k in VARIANT_ORDER],
                    showfliers=False, patch_artist=True)
    for patch, k in zip(bp["boxes"], VARIANT_ORDER):
        patch.set_facecolor(COLORS[k])
        patch.set_alpha(0.55)
    ax.axhline(0.9, color="0.4", ls="--", lw=0.8)
    ax.set_ylabel("R2_log (clipped to [-1, 1] for display)")
    ax.set_title("Six-variant doping curve accuracy on held-out test set")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT / "six_variant_r2_boxplot.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for k in VARIANT_ORDER:
        s = df[f"r2_{k}"].dropna()
        rows.append({
            "model": k, "label": VARIANT_LABEL[k], "n": len(s),
            "p10": float(s.quantile(0.1)), "p25": float(s.quantile(0.25)),
            "median": float(s.median()), "p75": float(s.quantile(0.75)),
            "p90": float(s.quantile(0.9)),
            "n_gt_09": int((s > 0.9).sum()), "n_gt_095": int((s > 0.95).sum()),
        })
    save_csv(pd.DataFrame(rows), SUM / "six_variant_r2_quantiles.csv")


def _theta_propagation(df: pd.DataFrame, X, y, test_fb, sr_pred, ag_pred, dg) -> None:
    """|delta theta| vs curve R2 for SR and AG process layers."""
    rows = []
    fb_to_i = {fb: i for i, fb in enumerate(test_fb)}
    for _, row in df.iterrows():
        fb = str(row["file_base"])
        if fb not in fb_to_i or fb not in dg.index:
            continue
        i = fb_to_i[fb]
        g = dg.loc[fb]
        true = {
            "ln_N_p": float(np.log(g["N_p_cm3"])),
            "z_p_um": float(g["z_p_um"]),
            "z_f1": float(g["z_f1"]),
            "z_f2": float(g["z_f2"]),
        }
        for model, pred in (("sr", sr_pred), ("ag", ag_pred)):
            rows.append({
                "file_base": fb, "model": model,
                "abs_d_ln_Np": abs(float(pred["ln_N_p"][i]) - true["ln_N_p"]),
                "abs_d_zp": abs(float(pred["z_p_um"][i]) - true["z_p_um"]),
                "abs_d_zf1": abs(float(pred["z_f1"][i]) - true["z_f1"]),
                "abs_d_zf2": abs(float(pred["z_f2"][i]) - true["z_f2"]),
                "r2_dg": float(row[f"r2_dg_{model}" if model == "ag" else "r2_dg_sr"]),
                "r2_chain": float(row[f"r2_dgresid_{model}" if model == "ag" else "r2_dgresid_sr"]),
            })
    tdf = pd.DataFrame(rows)
    save_csv(tdf, SUM / "theta_error_vs_curve_r2.csv")

    apply_plot_style()
    err_keys = [("abs_d_ln_Np", "|Δ ln Np|"), ("abs_d_zp", "|Δ zp|"),
                ("abs_d_zf2", "|Δ zf2|")]
    for model in ("sr", "ag"):
        sub = tdf[tdf["model"] == model]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, (ek, lab) in zip(axes, err_keys):
            ax.scatter(sub[ek], sub["r2_dg"].clip(-1, 1.05), s=8, alpha=0.35, c="C0")
            ax.set_xlabel(lab)
            ax.set_ylabel("curve R2_log (DG, clipped)")
            ax.set_title(f"{model.upper()}: {lab} -> curve R2")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Theta prediction error vs doping-curve accuracy ({model})")
        fig.tight_layout()
        fig.savefig(PLOT / f"theta_error_propagation_{model}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def _predict_sample(z_f, N_f, fitted, sr_p, ag_p, resid_expr):
    return {
        "dg_selffit": _predict_curve(None, z_f, *fitted),
        "dgresid_selffit": _predict_curve(resid_expr, z_f, *fitted),
        "dg_sr": _predict_curve(None, z_f, *sr_p),
        "dgresid_sr": _predict_curve(resid_expr, z_f, *sr_p),
        "dg_ag": _predict_curve(None, z_f, *ag_p),
        "dgresid_ag": _predict_curve(resid_expr, z_f, *ag_p),
    }


def _overlays_best_worst(df, curves, dg, test_fb, sr_pred, ag_pred, resid_expr) -> None:
    eligible = df[df["n_fit"] >= 30].copy()
    best = eligible.sort_values("rmse_dg_ag").head(6)
    worst = eligible.sort_values("rmse_dg_ag", ascending=False).head(6)
    fb_to_i = {fb: i for i, fb in enumerate(test_fb)}

    def _pack(subset: pd.DataFrame, tag: str) -> None:
        apply_plot_style()
        n = len(subset)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        wide: dict[str, pd.Series] = {}
        meta = []
        for rank, (ax, (_, row)) in enumerate(zip(axes.ravel(), subset.iterrows()), start=1):
            fb = str(row["file_base"])
            i = fb_to_i[fb]
            sub = curves[curves["file_base"] == fb].sort_values("depth_um")
            z = sub["depth_um"].values.astype(float)
            N = sub["value_fitted"].values.astype(float)
            keep = N >= N_FLOOR
            z_f, N_f = z[keep], N[keep]
            g = dg.loc[fb]
            fitted = (float(g["N_p_cm3"]), float(g["z_p_um"]), float(g["z_f1"]), float(g["z_f2"]))
            sr_p = (float(np.exp(sr_pred["ln_N_p"][i])), float(sr_pred["z_p_um"][i]),
                    float(sr_pred["z_f1"][i]), float(sr_pred["z_f2"][i]))
            ag_p = (float(np.exp(ag_pred["ln_N_p"][i])), float(ag_pred["z_p_um"][i]),
                    float(ag_pred["z_f1"][i]), float(ag_pred["z_f2"][i]))
            preds = _predict_sample(z_f, N_f, fitted, sr_p, ag_p, resid_expr)

            ax.semilogy(z_f, N_f, "ko", ms=2.5, alpha=0.5, label="sim")
            for k in VARIANT_ORDER:
                ax.semilogy(z_f, preds[k], "-", color=COLORS[k], lw=1.1,
                            label=f"{k} R2={row[f'r2_{k}']:.2f}")
            ax.set_title(fb[-14:], fontsize=8)
            ax.grid(True, which="both", alpha=0.25)
            if rank == 1:
                ax.legend(fontsize=5)

            # per-sample figure
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            ax2.semilogy(z_f, N_f, "ko", ms=3, alpha=0.55, label="sim (in-Si)")
            for k in VARIANT_ORDER:
                ax2.semilogy(z_f, preds[k], "-", color=COLORS[k], lw=1.4,
                            label=f"{VARIANT_LABEL[k]} R2={row[f'r2_{k}']:.3f}")
            ax2.set_xlabel("Depth (um)")
            ax2.set_ylabel("Boron (cm-3)")
            ax2.set_title(f"{tag}: {fb}")
            ax2.legend(fontsize=7)
            ax2.grid(True, which="both", alpha=0.25)
            fig2.tight_layout()
            fig2.savefig(PLOT / f"{tag}_{rank:02d}_{fb[-12:]}_overlay.png",
                         dpi=160, bbox_inches="tight")
            plt.close(fig2)

            prefix = f"{tag}{rank:02d}"
            wide[f"{prefix}_true_x"] = pd.Series(z_f)
            wide[f"{prefix}_true_y"] = pd.Series(N_f)
            for k in VARIANT_ORDER:
                wide[f"{prefix}_{k}_x"] = pd.Series(z_f)
                wide[f"{prefix}_{k}_y"] = pd.Series(preds[k])
            meta.append({
                "curve_id": prefix, "file_base": fb, "n_fit": int(row["n_fit"]),
                **{f"r2_{k}": row[f"r2_{k}"] for k in VARIANT_ORDER},
                **{f"rmse_{k}": row[f"rmse_{k}"] for k in VARIANT_ORDER},
            })

        fig.suptitle(f"Doping {tag}-6 overlays (n_fit>=30, ranked by rmse_dg_ag)")
        fig.tight_layout()
        fig.savefig(PLOT / f"{tag}6_panel.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        save_csv(pd.DataFrame(wide), ORIGIN / f"origin_doping_{tag}6_curves.csv")
        save_csv(pd.DataFrame(meta), ORIGIN / f"origin_doping_{tag}6_metadata.csv")

    _pack(best, "best")
    _pack(worst, "worst")


def main() -> None:
    setup_runtime()
    for d in (PKG, OUT, PLOT, ORIGIN, SUM):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SIX_PER)
    log(f"Loaded {len(df)} test samples from six-variant package")
    _boxplot(df)

    X, y, train_fb, test_fb = build_xy()
    # Align test_fb order with df rows that exist in X
    test_fb = [fb for fb in test_fb if fb in set(df["file_base"].astype(str))]
    # rebuild preds on full build_xy test set then map
    X2, y2, tr2, te2 = build_xy()
    theta_sr = _load_theta_sr()
    Xt = X2.loc[te2].values.astype(float)
    sr_pred = {t: eval_sympy_expr(theta_sr[t], Xt) for t in DG_TARGETS}
    ag_pred = _load_ag(X2, te2)
    dg = pd.read_csv(DG_EXT_CSV).set_index("file_base")
    curves = pd.read_csv(EXT_TRIM_CSV)
    resid_expr = _load_residual_expr()

    _theta_propagation(df, X2, y2, te2, sr_pred, ag_pred, dg)
    _overlays_best_worst(df, curves, dg, te2, sr_pred, ag_pred, resid_expr)

    (OUT / "README.md").write_text(
        "# Doping error-analysis extras\n\n"
        "- `six_variant_r2_boxplot.png`: held-out R2_log distributions (display clipped to [-1,1]).\n"
        "- `theta_error_propagation_*.png`: |Δθ| vs curve R2 for SR/AG process layers.\n"
        "- `best6_*` / `worst6_*`: overlays ranked by `rmse_dg_ag` among `n_fit>=30`.\n"
        "- Origin CSVs are wide tables: `*_true_x/y` plus per-variant `*_x/y`.\n",
        encoding="utf-8",
    )
    (PKG / "00_readme.md").write_text(
        "# Doping extras + defect pipeline package\n\n"
        "- `01_doping_error_analysis/`: doping six-variant error figures\n"
        "- `02_defect_pipeline/`: defect shape-prior + process→θ chain\n",
        encoding="utf-8",
    )
    log(f"Doping extras -> {OUT}")


if __name__ == "__main__":
    main()
