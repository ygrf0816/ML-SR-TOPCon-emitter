"""Process params -> defect shape-prior theta -> defect curve chain.

Prior (L always > 0):
  constant : N = N0
  offset   : N(z) = Nb + (Ns - Nb) * exp(-z/L),  L > 0

Ns > Nb : surface-high (decay); Ns < Nb : surface-low (rise toward bulk).
This replaces the old N0*exp(-z/L) form that forced L < 0 on rising profiles.

Process layer targets: ln_Ns, ln_Nb, ln_L.
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import ATHENA_FEATURES, OUTPUT_ROOT, RANDOM_STATE
from topcon_experiments.exp4_symbolic.defect_shape_prior import (
    L_FLAT,
    L_MAX,
    L_MIN,
    _r2_ln,
    fit_one,
    predict_prior,
)
from topcon_experiments.exp4_symbolic.preprocess_curves import process_all_curves
from topcon_experiments.exp4_symbolic.run_process_to_dg_chain import _physics_features
from topcon_experiments.exp4_symbolic.sr_equation_utils import (
    eval_sympy_expr,
    sort_formulas_by_accuracy,
)
from topcon_experiments.exp4_symbolic.sr_metrics import annotate_equation_metrics

MAX_FILES = 2000
EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
CURVE_CSV = EXP4_OUT / f"curve_processed_defect_ext{MAX_FILES}.csv"
WORK = EXP4_OUT / "process_to_defect_theta"
PRIOR_CSV = WORK / "data" / "defect_prior_params_v2.csv"  # new schema
AG_DIR = WORK / "autogluon_models_v2"
PKG = OUTPUT_ROOT / "paper_package_defect_and_doping_extras" / "02_defect_pipeline"

THETA_TARGETS = ["ln_Ns", "ln_Nb", "ln_L"]
BASELINE_R2 = 0.801
AG_TIME_S = 300
AG_PRESET = "good_quality"


def _ensure_curves() -> pd.DataFrame:
    if CURVE_CSV.exists():
        return pd.read_csv(CURVE_CSV)
    log(f"Preprocessing {MAX_FILES} defect curves ...")
    df = process_all_curves("defect", max_files=MAX_FILES)
    save_csv(df, CURVE_CSV)
    return df


def _fit_priors(curves: pd.DataFrame) -> pd.DataFrame:
    if PRIOR_CSV.exists():
        return pd.read_csv(PRIOR_CSV)
    PRIOR_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fbs = curves["file_base"].unique()
    for i, fb in enumerate(fbs, start=1):
        sub = curves[curves["file_base"] == fb].sort_values("depth_um")
        res = fit_one(sub["depth_um"].values, sub["value_fitted"].values)
        if "error" in res:
            continue
        res["file_base"] = fb
        rows.append(res)
        if i % 200 == 0 or i == len(fbs):
            log(f"  prior fit {i}/{len(fbs)} ({len(rows)} ok)")
            save_csv(pd.DataFrame(rows), PRIOR_CSV)
    df = pd.DataFrame(rows)
    save_csv(df, PRIOR_CSV)
    n_neg_L = int((df["L_um"] < 0).sum())
    log(
        f"Prior fits: n={len(df)}, median r2_ln={df['r2_ln'].median():.4f}, "
        f"frac_offset={(df['prior_type']=='offset').mean():.2f}, "
        f"n_negative_L={n_neg_L} (must be 0)"
    )
    if n_neg_L:
        raise RuntimeError(f"offset prior still produced {n_neg_L} negative L values")
    return df


def _build_xy(prior: pd.DataFrame):
    prior = prior.set_index("file_base")
    meta = load_raw_dataframe().set_index("file_base")
    ok = [fb for fb in prior.index if fb in meta.index]
    X = meta.loc[ok, ATHENA_FEATURES].astype(float).copy()
    valid = X.notna().all(axis=1)
    X = X.loc[valid]
    ok = list(X.index)
    X["athena_c_boron"] = np.log(X["athena_c_boron"])
    X = _physics_features(X)
    y = pd.DataFrame({
        "ln_Ns": prior.loc[ok, "ln_Ns"].astype(float),
        "ln_Nb": prior.loc[ok, "ln_Nb"].astype(float),
        "ln_L": prior.loc[ok, "ln_L"].astype(float),
    }, index=ok)
    train_fb, test_fb = train_test_split(ok, test_size=0.2, random_state=RANDOM_STATE)
    return X, y, train_fb, test_fb, prior


def _train_ag(X, y, train_fb, test_fb) -> dict[str, np.ndarray]:
    from autogluon.tabular import TabularPredictor

    AG_DIR.mkdir(parents=True, exist_ok=True)
    preds = {}
    metrics = []
    for tgt in THETA_TARGETS:
        path = AG_DIR / tgt
        tr = X.loc[train_fb].copy()
        tr["_y"] = y.loc[train_fb, tgt].values
        if (path / "predictor.pkl").exists():
            predictor = TabularPredictor.load(str(path))
            log(f"loaded AG[{tgt}]")
        else:
            predictor = TabularPredictor(
                label="_y", path=str(path), problem_type="regression", verbosity=1,
            ).fit(tr, presets=AG_PRESET, time_limit=AG_TIME_S)
        pr = predictor.predict(X.loc[test_fb]).values.astype(float)
        preds[tgt] = pr
        r2 = r2_score(y.loc[test_fb, tgt].values, pr)
        metrics.append({
            "target": tgt, "model": "autogluon", "test_R2": r2,
            "RMSE": float(np.sqrt(mean_squared_error(y.loc[test_fb, tgt].values, pr))),
        })
        log(f"AG defect theta[{tgt}]: test R2={r2:.4f}")
    save_csv(pd.DataFrame(metrics), WORK / "data" / "theta_autogluon_metrics_v2.csv")
    return preds


def _train_sr(X, y, train_fb, test_fb) -> dict[str, str]:
    from pysr import PySRRegressor
    from topcon_experiments.exp4_symbolic.sr_utils import (
        EXP4_OUT,
        SR_BINARY_OPERATORS,
        SR_EXTRA_SYMPY_MAPPINGS,
        SR_NESTED_CONSTRAINTS,
        SR_UNARY_OPERATORS,
    )

    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "feature_columns.json").write_text(
        json.dumps(list(X.columns), ensure_ascii=False, indent=2)
    )
    exprs = {}
    summary = []
    for tgt in THETA_TARGETS:
        out_csv = WORK / f"sr_formulas_defect_theta_v2_{tgt}.csv"
        if out_csv.exists():
            df = pd.read_csv(out_csv)
            log(f"skip SR[{tgt}]: exists")
        else:
            X_tr, y_tr = X.loc[train_fb], y.loc[train_fb, tgt].values
            X_te, y_te = X.loc[test_fb], y.loc[test_fb, tgt].values
            log(f"PySR defect theta[{tgt}] (light): fit {len(X_tr)}")
            sr_out = EXP4_OUT / "pysr_checkpoints" / f"defect_theta_v2_{tgt}"
            sr_out.mkdir(parents=True, exist_ok=True)
            model = PySRRegressor(
                populations=12, population_size=80, niterations=120, maxsize=16,
                binary_operators=SR_BINARY_OPERATORS,
                unary_operators=SR_UNARY_OPERATORS,
                extra_sympy_mappings=SR_EXTRA_SYMPY_MAPPINGS,
                nested_constraints=SR_NESTED_CONSTRAINTS,
                model_selection="accuracy", parsimony=0.0, batching=True,
                procs=0, multithreading=True, verbosity=1, progress=True,
                temp_equation_file=str(sr_out / "equations.csv"),
                tempdir=str(sr_out), delete_tempfiles=True,
            )
            model.fit(X_tr.values, y_tr)
            df = model.equations_.copy()
            df = annotate_equation_metrics(model, df, X_tr.values, y_tr, X_te.values, y_te)
            df["target"] = tgt
            save_csv(df, out_csv)
        best = sort_formulas_by_accuracy(df).iloc[0]
        exprs[tgt] = str(best.get("sympy_format") or best["equation"])
        summary.append({
            "target": tgt, "model": "sr",
            "test_R2": float(best["test_R2"]),
            "complexity": int(best["complexity"]),
            "equation": str(best["equation"]),
        })
        log(f"SR defect theta[{tgt}]: test R2={best['test_R2']:.4f}")
    save_csv(pd.DataFrame(summary), WORK / "data" / "theta_sr_summary_v2.csv")
    return exprs


def _L_from_ln(ln_L: float) -> float:
    L = float(np.exp(ln_L))
    if not np.isfinite(L) or L <= 0:
        return L_FLAT
    return float(np.clip(L, L_MIN, L_MAX))


# Empirical bounds from the fitted prior table (ln concentration ~1e16–1e18)
_LN_CLIP = (32.0, 43.0)


def _rebuild(z, ln_Ns, ln_Nb, ln_L) -> np.ndarray:
    """Rebuild offset prior; clip θ to physical range so bad SR preds cannot explode."""
    ln_Ns = float(np.clip(ln_Ns, *_LN_CLIP)) if np.isfinite(ln_Ns) else 38.5
    ln_Nb = float(np.clip(ln_Nb, *_LN_CLIP)) if np.isfinite(ln_Nb) else ln_Ns
    return predict_prior(z, "offset", ln_Ns, ln_Nb, _L_from_ln(ln_L))


def _chain_and_package(curves, prior, X, y, train_fb, test_fb, ag_pred, sr_exprs) -> None:
    PKG.mkdir(parents=True, exist_ok=True)
    plot_dir = PKG / "plots"
    origin_dir = PKG / "origin_csv"
    csv_dir = PKG / "csv"
    sum_dir = PKG / "summaries"
    for d in (plot_dir, origin_dir, csv_dir, sum_dir):
        d.mkdir(parents=True, exist_ok=True)
    # clear old overlays from broken v1
    for p in plot_dir.glob("best10_*.png"):
        p.unlink()
    for p in plot_dir.glob("worst6_*.png"):
        p.unlink()

    Xt = X.loc[test_fb].values.astype(float)
    sr_pred = {t: eval_sympy_expr(sr_exprs[t], Xt) for t in THETA_TARGETS}

    rows_th = []
    for i, fb in enumerate(test_fb):
        row = {"file_base": fb}
        for t in THETA_TARGETS:
            row[f"{t}_true"] = float(y.loc[fb, t])
            row[f"{t}_sr"] = float(sr_pred[t][i])
            row[f"{t}_ag"] = float(ag_pred[t][i])
        rows_th.append(row)
    th_df = pd.DataFrame(rows_th)
    save_csv(th_df, csv_dir / "theta_test_predictions.csv")
    save_csv(prior.reset_index(), csv_dir / "prior_params.csv")

    apply_plot_style()
    for model, sfx in (("sr", "_sr"), ("autogluon", "_ag")):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, tgt in zip(axes, THETA_TARGETS):
            yt = th_df[f"{tgt}_true"].values
            yp = th_df[f"{tgt}{sfx}"].values
            mask = np.isfinite(yt) & np.isfinite(yp)
            ax.scatter(yt[mask], yp[mask], s=10, alpha=0.4, c="C0")
            lo, hi = float(np.nanmin(yt[mask])), float(np.nanmax(yt[mask]))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_title(f"{tgt} {model} R2={r2_score(yt[mask], yp[mask]):.3f}")
            ax.set_xlabel("true")
            ax.set_ylabel("predicted")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Defect process->theta ({model}), L>0 offset prior")
        fig.tight_layout()
        fig.savefig(plot_dir / f"theta_scatter_{model}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    theta_sum = []
    for tgt in THETA_TARGETS:
        for model, sfx in (("sr", "_sr"), ("autogluon", "_ag")):
            yt, yp = th_df[f"{tgt}_true"].values, th_df[f"{tgt}{sfx}"].values
            mask = np.isfinite(yt) & np.isfinite(yp)
            theta_sum.append({
                "target": tgt, "model": model, "n": int(mask.sum()),
                "R2": float(r2_score(yt[mask], yp[mask])),
                "RMSE": float(np.sqrt(mean_squared_error(yt[mask], yp[mask]))),
            })
    save_csv(pd.DataFrame(theta_sum), sum_dir / "process_layer_theta_summary.csv")

    prior_i = prior if prior.index.name == "file_base" else prior.set_index("file_base")

    def _rel_rmse(obs, pred):
        m = (obs > 0) & (pred > 0) & np.isfinite(pred)
        if m.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.mean((np.log(obs[m]) - np.log(pred[m])) ** 2)))

    rows = []
    cache: dict[str, dict] = {}
    pooled = {k: {"y": [], "yhat": []} for k in ("prior_selffit", "prior_sr", "prior_ag")}
    for i, fb in enumerate(test_fb):
        if fb not in prior_i.index:
            continue
        sub = curves[curves["file_base"] == fb].sort_values("depth_um")
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        keep = np.isfinite(N) & (N > 0)
        z_f, N_f = z[keep], N[keep]
        if z_f.size < 6:
            continue
        p = prior_i.loc[fb]
        N_sf = predict_prior(
            z_f, str(p["prior_type"]), float(p["ln_Ns"]), float(p["ln_Nb"]), float(p["L_um"])
        )
        N_sr = _rebuild(z_f, sr_pred["ln_Ns"][i], sr_pred["ln_Nb"][i], sr_pred["ln_L"][i])
        N_ag = _rebuild(z_f, ag_pred["ln_Ns"][i], ag_pred["ln_Nb"][i], ag_pred["ln_L"][i])
        row = {
            "file_base": fb, "n_fit": int(z_f.size),
            "log_range": float(np.log(N_f.max()) - np.log(N_f.min())),
            "r2_prior_selffit": _r2_ln(N_f, N_sf),
            "r2_prior_sr": _r2_ln(N_f, N_sr),
            "r2_prior_ag": _r2_ln(N_f, N_ag),
            "rmse_ln_prior_selffit": _rel_rmse(N_f, N_sf),
            "rmse_ln_prior_sr": _rel_rmse(N_f, N_sr),
            "rmse_ln_prior_ag": _rel_rmse(N_f, N_ag),
        }
        rows.append(row)
        for name, Npred in (("prior_selffit", N_sf), ("prior_sr", N_sr), ("prior_ag", N_ag)):
            m = (N_f > 0) & (Npred > 0) & np.isfinite(Npred)
            pooled[name]["y"].append(np.log(N_f[m]))
            pooled[name]["yhat"].append(np.log(Npred[m]))
        cache[str(fb)] = {
            "true_x": z_f, "true_y": N_f,
            "prior_selffit": N_sf, "prior_sr": N_sr, "prior_ag": N_ag,
        }

    edf = pd.DataFrame(rows)
    save_csv(edf, sum_dir / "defect_curve_per_sample.csv")

    sum_rows = [{
        "model": "baseline_ref_athena_sr",
        "note": "legacy pooled R2 on ln c",
        "n_samples": None, "n_points": None, "pooled_r2_ln": BASELINE_R2,
        "median_per_sample_r2_ln": None, "median_rmse_ln": None, "n_gt_09": None,
    }]
    for name in ("prior_selffit", "prior_sr", "prior_ag"):
        yv = np.concatenate(pooled[name]["y"]) if pooled[name]["y"] else np.array([])
        yhat = np.concatenate(pooled[name]["yhat"]) if pooled[name]["yhat"] else np.array([])
        pooled_r2 = float(r2_score(yv, yhat)) if yv.size >= 4 else float("nan")
        s = edf[f"r2_{name}"].dropna()
        rm = edf[f"rmse_ln_{name}"].dropna()
        sum_rows.append({
            "model": name,
            "note": "offset prior with L>0; pooled R2 comparable to baseline",
            "n_samples": int(len(edf)), "n_points": int(yv.size),
            "pooled_r2_ln": pooled_r2,
            "median_per_sample_r2_ln": float(s.median()) if len(s) else float("nan"),
            "median_rmse_ln": float(rm.median()) if len(rm) else float("nan"),
            "n_gt_09": int((s > 0.9).sum()) if len(s) else 0,
            "n_gt_095": int((s > 0.95).sum()) if len(s) else 0,
        })
    save_csv(pd.DataFrame(sum_rows), sum_dir / "defect_curve_summary.csv")
    print(pd.DataFrame(sum_rows).to_string(index=False))

    eligible = edf[edf["n_fit"] >= 30].copy()
    best = eligible.sort_values("rmse_ln_prior_ag").head(10)
    worst = eligible.sort_values("rmse_ln_prior_ag", ascending=False).head(6)
    variants = ["prior_selffit", "prior_sr", "prior_ag"]
    colors = {"prior_selffit": "C3", "prior_sr": "C2", "prior_ag": "C0"}

    def _pack(subset: pd.DataFrame, tag: str, ncols: int) -> None:
        apply_plot_style()
        n = len(subset)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
        axes = np.atleast_1d(axes).ravel()
        wide: dict[str, pd.Series] = {}
        meta = []
        for rank, (ax, (_, row)) in enumerate(zip(axes, subset.iterrows()), start=1):
            fb = str(row["file_base"])
            cur = cache[fb]
            ax.semilogy(cur["true_x"], cur["true_y"], "ko", ms=2.5, alpha=0.5, label="sim")
            for k in variants:
                ax.semilogy(cur["true_x"], cur[k], "-", color=colors[k], lw=1.3,
                            label=f"{k} R2={row[f'r2_{k}']:.2f}")
            ax.set_title(fb[-14:], fontsize=8)
            ax.grid(True, which="both", alpha=0.25)
            if rank == 1:
                ax.legend(fontsize=5)

            fig2, ax2 = plt.subplots(figsize=(7, 5))
            ax2.semilogy(cur["true_x"], cur["true_y"], "ko", ms=3, alpha=0.55, label="sim defect")
            for k in variants:
                ax2.semilogy(cur["true_x"], cur[k], "-", color=colors[k], lw=1.5,
                            label=f"{k} R2={row[f'r2_{k}']:.3f}")
            ax2.set_xlabel("Depth (um)")
            ax2.set_ylabel("Vacancy conc. (cm-3)")
            ax2.set_title(f"{tag}: {fb}")
            ax2.legend(fontsize=8)
            ax2.grid(True, which="both", alpha=0.25)
            fig2.tight_layout()
            fig2.savefig(plot_dir / f"{tag}_{rank:02d}_{fb[-12:]}_overlay.png",
                         dpi=160, bbox_inches="tight")
            plt.close(fig2)

            prefix = f"{tag}{rank:02d}"
            wide[f"{prefix}_true_x"] = pd.Series(cur["true_x"])
            wide[f"{prefix}_true_y"] = pd.Series(cur["true_y"])
            for k in variants:
                wide[f"{prefix}_{k}_x"] = pd.Series(cur["true_x"])
                wide[f"{prefix}_{k}_y"] = pd.Series(cur[k])
            meta.append({
                "curve_id": prefix, "file_base": fb, "n_fit": int(row["n_fit"]),
                **{f"r2_{k}": row[f"r2_{k}"] for k in variants},
                **{f"rmse_ln_{k}": row[f"rmse_ln_{k}"] for k in variants},
            })

        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"Defect {tag} overlays (offset prior, L>0)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{tag}_panel.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        save_csv(pd.DataFrame(wide), origin_dir / f"origin_defect_{tag}_curves.csv")
        save_csv(pd.DataFrame(meta), origin_dir / f"origin_defect_{tag}_metadata.csv")

    _pack(best, "best10", 5)
    _pack(worst, "worst6", 3)

    (PKG / "README.md").write_text(
        "# Defect pipeline (offset prior, L > 0)\n\n"
        "Prior candidates:\n"
        "- constant: `N = N0`\n"
        "- offset: `N(z) = Nb + (Ns - Nb)*exp(-z/L)` with **L > 0**\n\n"
        "Rising profiles (former negative-L cases) are handled by `Ns < Nb`, "
        "not by a negative length.\n\n"
        "Process layer predicts `ln_Ns`, `ln_Nb`, `ln_L` (SR + AutoGluon).\n\n"
        f"Legacy baseline pooled R2 ≈ {BASELINE_R2}. "
        "Primary metric: `pooled_r2_ln` in `summaries/defect_curve_summary.csv`.\n",
        encoding="utf-8",
    )


def main() -> None:
    setup_runtime()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "data").mkdir(parents=True, exist_ok=True)

    curves = _ensure_curves()
    prior = _fit_priors(curves)
    X, y, train_fb, test_fb, prior_i = _build_xy(prior)
    log(f"defect theta v2: {len(train_fb)} train / {len(test_fb)} test, feats={X.shape[1]}")

    sr_exprs = _train_sr(X, y, train_fb, test_fb)
    ag_pred = _train_ag(X, y, train_fb, test_fb)
    _chain_and_package(curves, prior_i, X, y, train_fb, test_fb, ag_pred, sr_exprs)
    log(f"Defect package -> {PKG}")


if __name__ == "__main__":
    main()
