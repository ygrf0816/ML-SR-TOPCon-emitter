"""Package the six doping-curve model variants into a paper-ready results folder.

Variants
--------
1. dg_selffit          : pure DG, theta fitted from each curve
2. dgresid_selffit     : DG * exp(r_SR), theta fitted from each curve
3. dg_sr               : pure DG, theta from process-SR formulas
4. dgresid_sr          : DG * exp(r_SR), theta from process-SR formulas
5. dg_ag               : pure DG, theta from AutoGluon
6. dgresid_ag          : DG * exp(r_SR), theta from AutoGluon

Literature D1-D10 has no Athena process parameters, so only variants 1-2 apply.
Simulation uses the held-out test split (same as all prior experiments).

Folder layout (under outputs/paper_package_six_variants/):
  00_readme.md
  01_literature/
  02_simulation_best10/
  03_process_layer_theta/
  04_summaries/
  05_llm_reports/
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

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
    double_gaussian,
)
from topcon_experiments.exp6_experimental.literature_benchmark.fit_prior_sr_to_literature import (
    PRIOR_EXP_DIR,
    _best_expr,
    _fit_dg_to_lit,
)
from topcon_experiments.exp6_experimental.literature_benchmark.fit_sr_to_literature import (
    _load_reference_curve,
)
from topcon_experiments.exp6_experimental.literature_benchmark.parse_literature_params import (
    load_parsed_table,
)

PKG = OUTPUT_ROOT / "paper_package_six_variants"
LIT_DIR = PKG / "01_literature"
SIM_DIR = PKG / "02_simulation_best10"
THETA_DIR = PKG / "03_process_layer_theta"
SUM_DIR = PKG / "04_summaries"
LLM_DIR = PKG / "05_llm_reports"

VARIANT_ORDER = [
    "dg_selffit",
    "dgresid_selffit",
    "dg_sr",
    "dgresid_sr",
    "dg_ag",
    "dgresid_ag",
]
VARIANT_LABEL = {
    "dg_selffit": "Pure DG (self-fit theta)",
    "dgresid_selffit": "DG x SR-resid (self-fit theta)",
    "dg_sr": "Pure DG + SR process layer",
    "dgresid_sr": "DG x SR-resid + SR process layer",
    "dg_ag": "Pure DG + AutoGluon process layer",
    "dgresid_ag": "DG x SR-resid + AutoGluon process layer",
}
N_BEST = 10


def _rmse_log(N_obs: np.ndarray, N_pred: np.ndarray) -> float:
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.log10(N_obs[mask]) - np.log10(N_pred[mask])) ** 2)))


def _load_theta_sr_exprs() -> dict[str, str]:
    out = {}
    for tgt in DG_TARGETS:
        df = pd.read_csv(THETA_SR_DIR / f"sr_formulas_theta_{tgt}.csv")
        best = sort_formulas_by_accuracy(df).iloc[0]
        out[tgt] = str(best.get("sympy_format") or best["equation"])
    return out


def _load_ag_preds(X: pd.DataFrame, test_fb: list[str]) -> dict[str, np.ndarray]:
    from autogluon.tabular import TabularPredictor

    pred = {}
    for tgt in DG_TARGETS:
        path = THETA_SR_DIR / "autogluon_models" / tgt
        predictor = TabularPredictor.load(str(path))
        pred[tgt] = predictor.predict(X.loc[test_fb]).values.astype(float)
    return pred


def _origin_wide(curves: dict[str, dict[str, np.ndarray]], ids: list[str],
                 pred_keys: list[str]) -> pd.DataFrame:
    """Origin-friendly wide table: id_true_x, id_true_y, id_<pred>_x, id_<pred>_y, ..."""
    wide: dict[str, pd.Series] = {}
    for pid in ids:
        cur = curves[pid]
        wide[f"{pid}_true_x"] = pd.Series(cur["true_x"])
        wide[f"{pid}_true_y"] = pd.Series(cur["true_y"])
        for key in pred_keys:
            wide[f"{pid}_{key}_x"] = pd.Series(cur["true_x"])
            wide[f"{pid}_{key}_y"] = pd.Series(cur[key])
    return pd.DataFrame(wide)


def _summary_stats(df: pd.DataFrame, cols: list[str], id_col: str) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = df[col].dropna()
        rows.append({
            "model": col,
            "n": int(len(s)),
            "median_r2_log": float(s.median()) if len(s) else np.nan,
            "mean_r2_log": float(s.mean()) if len(s) else np.nan,
            "min_r2_log": float(s.min()) if len(s) else np.nan,
            "p10_r2_log": float(s.quantile(0.1)) if len(s) else np.nan,
            "n_gt_09": int((s > 0.9).sum()) if len(s) else 0,
            "n_gt_095": int((s > 0.95).sum()) if len(s) else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Literature (self-fit only)
# ---------------------------------------------------------------------------

def package_literature(resid_expr: str) -> pd.DataFrame:
    log("Packaging literature D1-D10 (self-fit variants only)...")
    lit_plot = LIT_DIR / "plots"
    lit_csv = LIT_DIR / "origin_csv"
    lit_plot.mkdir(parents=True, exist_ok=True)
    lit_csv.mkdir(parents=True, exist_ok=True)

    table = load_parsed_table()
    rows = []
    curve_cache: dict[str, dict[str, np.ndarray]] = {}
    apply_plot_style()

    for _, row in table.iterrows():
        pid = str(row["profile_id"])
        z, N = _load_reference_curve(pid)
        keep = N >= N_FLOOR
        z_f, N_f = z[keep], N[keep]
        fit = _fit_dg_to_lit(z, N)
        if "error" in fit:
            continue
        params = (fit["N_p_cm3"], fit["z_p_um"], fit["z_f1"], fit["z_f2"])
        N_dg = _predict_curve(None, z_f, *params)
        N_chain = _predict_curve(resid_expr, z_f, *params)
        r2_dg = _r2_log(N_f, N_dg)
        r2_ch = _r2_log(N_f, N_chain)
        rows.append({
            "profile_id": pid,
            "r2_dg_selffit": r2_dg,
            "rmse_dg_selffit": _rmse_log(N_f, N_dg),
            "r2_dgresid_selffit": r2_ch,
            "rmse_dgresid_selffit": _rmse_log(N_f, N_chain),
            "n_fit": int(z_f.size),
            "note": "literature has no process params; SR/AG process layers N/A",
        })
        curve_cache[pid] = {
            "true_x": z_f, "true_y": N_f,
            "dg_selffit": N_dg, "dgresid_selffit": N_chain,
        }

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(z_f, N_f, "ko", ms=3.5, alpha=0.55, label="literature")
        ax.semilogy(z_f, N_dg, "-", color="C3", lw=1.8,
                    label=f"DG self-fit R2={r2_dg:.4f}")
        ax.semilogy(z_f, N_chain, "--", color="C0", lw=1.8,
                    label=f"DG x SR-resid R2={r2_ch:.4f}")
        ax.set_xlabel("Depth (um)")
        ax.set_ylabel("Boron concentration (cm-3)")
        ax.set_title(f"{pid}: literature overlay (self-fit theta)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(lit_plot / f"{pid}_overlay.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    df = pd.DataFrame(rows).sort_values("profile_id")
    save_csv(df, LIT_DIR / "per_profile_metrics.csv")

    pids = list(df["profile_id"])
    wide = _origin_wide(curve_cache, pids, ["dg_selffit", "dgresid_selffit"])
    save_csv(wide, lit_csv / "origin_literature_all_curves.csv")
    save_csv(df, lit_csv / "origin_literature_all_metadata.csv")

    # README for literature
    (LIT_DIR / "README.md").write_text(
        "# Literature D1-D10\n\n"
        "Literature profiles do **not** publish Athena process parameters, so only "
        "self-fit variants are applicable:\n"
        "- `dg_selffit`, `dgresid_selffit`\n\n"
        "Process-layer variants (`*_sr`, `*_ag`) are marked N/A.\n",
        encoding="utf-8",
    )
    return df


# ---------------------------------------------------------------------------
# Simulation test set: all 6 variants
# ---------------------------------------------------------------------------

def package_simulation(resid_expr: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Evaluating all 6 variants on simulation test set...")
    X, y, train_fb, test_fb = build_xy()
    curves = pd.read_csv(EXT_TRIM_CSV)
    dg = pd.read_csv(DG_EXT_CSV).set_index("file_base")

    # Process-layer predictions
    theta_sr = _load_theta_sr_exprs()
    Xt = X.loc[test_fb].values.astype(float)
    sr_pred = {t: eval_sympy_expr(theta_sr[t], Xt) for t in DG_TARGETS}
    ag_pred = _load_ag_preds(X, test_fb)

    rows = []
    curve_cache: dict[str, dict[str, np.ndarray]] = {}

    for i, fb in enumerate(test_fb):
        if fb not in dg.index:
            continue
        sub = curves[curves["file_base"] == fb].sort_values("depth_um")
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        keep = N >= N_FLOOR
        z_f, N_f = z[keep], N[keep]
        if z_f.size < 6:
            continue
        g = dg.loc[fb]
        fitted = (float(g["N_p_cm3"]), float(g["z_p_um"]), float(g["z_f1"]), float(g["z_f2"]))
        sr_p = (float(np.exp(sr_pred["ln_N_p"][i])), float(sr_pred["z_p_um"][i]),
                float(sr_pred["z_f1"][i]), float(sr_pred["z_f2"][i]))
        ag_p = (float(np.exp(ag_pred["ln_N_p"][i])), float(ag_pred["z_p_um"][i]),
                float(ag_pred["z_f1"][i]), float(ag_pred["z_f2"][i]))
        if not (all(np.isfinite(sr_p)) and all(np.isfinite(ag_p))):
            continue

        preds = {
            "dg_selffit": _predict_curve(None, z_f, *fitted),
            "dgresid_selffit": _predict_curve(resid_expr, z_f, *fitted),
            "dg_sr": _predict_curve(None, z_f, *sr_p),
            "dgresid_sr": _predict_curve(resid_expr, z_f, *sr_p),
            "dg_ag": _predict_curve(None, z_f, *ag_p),
            "dgresid_ag": _predict_curve(resid_expr, z_f, *ag_p),
        }
        row = {"file_base": fb, "n_fit": int(z_f.size), "split": "test"}
        for k, Npred in preds.items():
            row[f"r2_{k}"] = _r2_log(N_f, Npred)
            row[f"rmse_{k}"] = _rmse_log(N_f, Npred)
        rows.append(row)
        curve_cache[str(fb)] = {"true_x": z_f, "true_y": N_f, **preds}

    df = pd.DataFrame(rows)
    save_csv(df, SUM_DIR / "simulation_test_per_sample.csv")

    # Best 10 by AutoGluon DG chain RMSE (strongest process-layer baseline),
    # require enough points for a meaningful profile.
    eligible = df[df["n_fit"] >= 30].copy()
    best = eligible.sort_values("rmse_dg_ag").head(N_BEST).reset_index(drop=True)

    sim_plot = SIM_DIR / "plots"
    sim_csv = SIM_DIR / "origin_csv"
    sim_plot.mkdir(parents=True, exist_ok=True)
    sim_csv.mkdir(parents=True, exist_ok=True)

    apply_plot_style()
    # Combined 2x5 panel
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    wide_all: dict[str, pd.Series] = {}
    meta_rows = []
    for rank, (ax, (_, brow)) in enumerate(zip(axes.ravel(), best.iterrows()), start=1):
        fb = str(brow["file_base"])
        cur = curve_cache[fb]
        ax.semilogy(cur["true_x"], cur["true_y"], "ko", ms=2.5, alpha=0.5, label="sim")
        colors = {"dg_selffit": "C3", "dgresid_selffit": "C1",
                  "dg_sr": "C2", "dgresid_sr": "C4",
                  "dg_ag": "C0", "dgresid_ag": "C5"}
        styles = {"dg_selffit": "-", "dgresid_selffit": "--",
                  "dg_sr": "-", "dgresid_sr": "--",
                  "dg_ag": "-", "dgresid_ag": "--"}
        for k in VARIANT_ORDER:
            ax.semilogy(cur["true_x"], cur[k], styles[k], color=colors[k], lw=1.2,
                        label=f"{k} R2={brow[f'r2_{k}']:.2f}")
        ax.set_title(fb[-14:], fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        if rank == 1:
            ax.legend(fontsize=5, loc="best")

        prefix = f"curve{rank:02d}"
        wide_all[f"{prefix}_true_x"] = pd.Series(cur["true_x"])
        wide_all[f"{prefix}_true_y"] = pd.Series(cur["true_y"])
        for k in VARIANT_ORDER:
            wide_all[f"{prefix}_{k}_x"] = pd.Series(cur["true_x"])
            wide_all[f"{prefix}_{k}_y"] = pd.Series(cur[k])
        meta_rows.append({
            "curve_id": prefix, "file_base": fb, "n_fit": int(brow["n_fit"]),
            **{f"r2_{k}": brow[f"r2_{k}"] for k in VARIANT_ORDER},
            **{f"rmse_{k}": brow[f"rmse_{k}"] for k in VARIANT_ORDER},
        })

        # Per-sample overlay (cleaner legend)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.semilogy(cur["true_x"], cur["true_y"], "ko", ms=3, alpha=0.55, label="sim (in-Si)")
        for k in VARIANT_ORDER:
            ax2.semilogy(cur["true_x"], cur[k], styles[k], color=colors[k], lw=1.5,
                         label=f"{VARIANT_LABEL[k]}\nR2={brow[f'r2_{k}']:.3f}")
        ax2.set_xlabel("Depth (um)")
        ax2.set_ylabel("Boron concentration (cm-3)")
        ax2.set_title(fb)
        ax2.grid(True, which="both", alpha=0.25)
        ax2.legend(fontsize=7)
        fig2.tight_layout()
        fig2.savefig(sim_plot / f"{prefix}_{fb[-12:]}_overlay.png", dpi=160, bbox_inches="tight")
        plt.close(fig2)

    fig.suptitle("Simulation best-10 (by AutoGluon-DG RMSE, n_fit>=30): all 6 variants")
    fig.supxlabel("Depth (um)")
    fig.supylabel("Boron concentration (cm-3)")
    fig.tight_layout()
    fig.savefig(sim_plot / "best10_all_variants_panel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    save_csv(pd.DataFrame(wide_all), sim_csv / "origin_sim_best10_all_variants.csv")
    save_csv(pd.DataFrame(meta_rows), sim_csv / "origin_sim_best10_metadata.csv")
    (SIM_DIR / "README.md").write_text(
        "# Simulation best-10 overlays\n\n"
        "Selected from the held-out test set by lowest `rmse_dg_ag` among samples "
        "with `n_fit >= 30` (N>=1e18 window after BSG valley cut).\n"
        "All six variants are overlaid and exported in Origin wide CSV.\n",
        encoding="utf-8",
    )
    return df, best


# ---------------------------------------------------------------------------
# Process-layer theta diagnostics
# ---------------------------------------------------------------------------

def package_theta_layer() -> pd.DataFrame:
    log("Packaging process-layer theta diagnostics...")
    X, y, train_fb, test_fb = build_xy()
    theta_sr = _load_theta_sr_exprs()
    Xt = X.loc[test_fb].values.astype(float)
    sr_pred = {t: eval_sympy_expr(theta_sr[t], Xt) for t in DG_TARGETS}
    ag_pred = _load_ag_preds(X, test_fb)

    THETA_DIR.mkdir(parents=True, exist_ok=True)
    (THETA_DIR / "plots").mkdir(exist_ok=True)
    (THETA_DIR / "csv").mkdir(exist_ok=True)

    # Per-sample predictions CSV
    rows = []
    for i, fb in enumerate(test_fb):
        row = {"file_base": fb, "split": "test"}
        for t in DG_TARGETS:
            yt = float(y.loc[fb, t])
            row[f"{t}_true"] = yt
            row[f"{t}_sr"] = float(sr_pred[t][i])
            row[f"{t}_ag"] = float(ag_pred[t][i])
        rows.append(row)
    pred_df = pd.DataFrame(rows)
    save_csv(pred_df, THETA_DIR / "csv" / "theta_test_predictions.csv")

    # Summary metrics
    sum_rows = []
    for t in DG_TARGETS:
        yt = pred_df[f"{t}_true"].values
        for model, col in (("sr", f"{t}_sr"), ("autogluon", f"{t}_ag")):
            yp = pred_df[col].values
            mask = np.isfinite(yt) & np.isfinite(yp)
            sum_rows.append({
                "target": t, "model": model, "n": int(mask.sum()),
                "R2": float(r2_score(yt[mask], yp[mask])),
                "RMSE": float(np.sqrt(mean_squared_error(yt[mask], yp[mask]))),
            })
    # Also attach prior GBM numbers if available
    gbm_path = OUTPUT_ROOT / "exp4_symbolic" / "process_to_dg" / "data" / "stage1_metrics.csv"
    if gbm_path.exists():
        gbm = pd.read_csv(gbm_path)
        gbm_test = gbm[(gbm["split"] == "test") & (gbm["features"] == "physics")]
        for _, r in gbm_test.iterrows():
            sum_rows.append({
                "target": r["target"], "model": "gbm_physics", "n": int(r["n"]),
                "R2": float(r["R2"]), "RMSE": float(r["RMSE"]),
            })
    sum_df = pd.DataFrame(sum_rows)
    save_csv(sum_df, SUM_DIR / "process_layer_theta_summary.csv")
    save_csv(sum_df, THETA_DIR / "csv" / "theta_metrics_summary.csv")

    # Scatter plots 2x2 for SR and AG
    apply_plot_style()
    for model, suffix in (("sr", "_sr"), ("autogluon", "_ag")):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for ax, t in zip(axes.ravel(), DG_TARGETS):
            yt = pred_df[f"{t}_true"].values
            yp = pred_df[f"{t}{suffix}"].values
            mask = np.isfinite(yt) & np.isfinite(yp)
            ax.scatter(yt[mask], yp[mask], s=8, alpha=0.35, c="C0")
            lo = float(np.nanmin(yt[mask]))
            hi = float(np.nanmax(yt[mask]))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            r2 = r2_score(yt[mask], yp[mask])
            ax.set_title(f"{t}: {model} R2={r2:.3f}")
            ax.set_xlabel("true")
            ax.set_ylabel("predicted")
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"Process layer: process params -> theta ({model})")
        fig.tight_layout()
        fig.savefig(THETA_DIR / "plots" / f"theta_scatter_{model}.png",
                    dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Feature legend + formulas
    feat_src = THETA_SR_DIR / "feature_columns.json"
    if feat_src.exists():
        shutil.copy2(feat_src, THETA_DIR / "feature_columns.json")
    formulas = []
    for t in DG_TARGETS:
        formulas.append({"target": t, "equation": theta_sr[t]})
    save_csv(pd.DataFrame(formulas), THETA_DIR / "csv" / "theta_sr_formulas.csv")
    return sum_df


# ---------------------------------------------------------------------------
# Summaries + LLM reports
# ---------------------------------------------------------------------------

def package_summaries(lit_df: pd.DataFrame, sim_df: pd.DataFrame) -> None:
    log("Writing curve-prediction summaries...")
    # Literature summary (self-fit only)
    lit_sum = _summary_stats(
        lit_df, ["r2_dg_selffit", "r2_dgresid_selffit"], "profile_id"
    )
    lit_sum.insert(0, "dataset", "literature_D1-D10")
    save_csv(lit_sum, SUM_DIR / "literature_curve_summary.csv")

    sim_cols = [f"r2_{k}" for k in VARIANT_ORDER]
    sim_sum = _summary_stats(sim_df, sim_cols, "file_base")
    sim_sum.insert(0, "dataset", "simulation_test")
    # rename for readability
    sim_sum["model"] = sim_sum["model"].str.replace("^r2_", "", regex=True)
    sim_sum["model_label"] = sim_sum["model"].map(VARIANT_LABEL)
    save_csv(sim_sum, SUM_DIR / "simulation_curve_summary.csv")

    # Combined overview
    overview = pd.concat([
        lit_sum.assign(model_label=lit_sum["model"].map({
            "r2_dg_selffit": VARIANT_LABEL["dg_selffit"],
            "r2_dgresid_selffit": VARIANT_LABEL["dgresid_selffit"],
        })),
        sim_sum,
    ], ignore_index=True)
    save_csv(overview, SUM_DIR / "curve_prediction_overview.csv")


def package_llm() -> None:
    log("Copying LLM reports...")
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    copies = [
        (THETA_SR_DIR / "theta_llm_analysis.md", LLM_DIR / "theta_sr_llm_analysis.md"),
        (THETA_SR_DIR / "theta_llm_analysis.csv", LLM_DIR / "theta_sr_llm_analysis.csv"),
        (PRIOR_EXP_DIR / "prior_residual_full_llm_analysis.md",
         LLM_DIR / "dg_residual_formula_llm_analysis.md"),
        (PRIOR_EXP_DIR / "prior_residual_full_llm_analysis.csv",
         LLM_DIR / "dg_residual_formula_llm_analysis.csv"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)
            log(f"  copied {src.name}")
        else:
            log(f"  MISSING {src}")

    # Pure DG formula note (already simplest form)
    (LLM_DIR / "pure_dg_formula.md").write_text(
        "# Pure double-Gaussian formula\n\n"
        "The literature asymmetric double-Gaussian is already the compact closed form:\n\n"
        "```\n"
        "N(z) = N_p * exp(-((z - z_p) / z_f1)^2)   if z <  z_p\n"
        "N(z) = N_p * exp(-((z - z_p) / z_f2)^2)   if z >= z_p\n"
        "```\n\n"
        "No further LLM simplification is required for the shape layer itself. "
        "LLM analysis for the residual correction `r_SR` and for the process->theta "
        "SR formulas is in the companion reports in this folder.\n",
        encoding="utf-8",
    )


def write_root_readme() -> None:
    text = """# Six-variant doping-curve results package

## Variants

| id | meaning |
|---|---|
| dg_selffit | Pure DG; theta fitted from each curve |
| dgresid_selffit | DG × exp(r_SR); theta fitted from each curve |
| dg_sr | Pure DG; theta from process-SR formulas |
| dgresid_sr | DG × exp(r_SR); theta from process-SR formulas |
| dg_ag | Pure DG; theta from AutoGluon |
| dgresid_ag | DG × exp(r_SR); theta from AutoGluon |

## Folders

- `01_literature/` — D1–D10 overlays + Origin CSV (**self-fit only**; no process params)
- `02_simulation_best10/` — top-10 test-set overlays (all 6 variants) + Origin CSV
- `03_process_layer_theta/` — SR / AutoGluon theta scatter plots + prediction CSV
- `04_summaries/` — curve & process-layer summary CSVs
- `05_llm_reports/` — LLM simplification / physics reviews

## Evaluation window

All R²_log / RMSE_log use the BSG-valley-trimmed, N ≥ 1e18 cm⁻³ in-silicon segment.
"""
    (PKG / "00_readme.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_runtime()
    for d in (PKG, LIT_DIR, SIM_DIR, THETA_DIR, SUM_DIR, LLM_DIR):
        d.mkdir(parents=True, exist_ok=True)
    write_root_readme()

    resid_expr = _load_residual_expr()
    # sanity: residual formula must exist
    _ = _best_expr(PRIOR_EXP_DIR / "sr_formulas_doping_residual_full.csv")

    lit_df = package_literature(resid_expr)
    sim_df, _best = package_simulation(resid_expr)
    package_theta_layer()
    package_summaries(lit_df, sim_df)
    package_llm()

    log(f"Package ready -> {PKG}")
    print(f"\n=== Package written to ===\n{PKG}\n")
    print((SUM_DIR / "simulation_curve_summary.csv").read_text())


if __name__ == "__main__":
    main()
