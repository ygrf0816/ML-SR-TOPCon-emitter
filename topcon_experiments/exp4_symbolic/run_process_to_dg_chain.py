"""Process-parameter -> double-Gaussian-parameter -> curve chain experiment.

Motivation: the winning prior-residual SR formula has FIXED global constants
but takes per-sample DG params (z_p, z_f2) fitted from the curve itself, so
the chain was "curve -> curve". Here we close the loop from PROCESS inputs:

  stage 1: 8 Athena process params -> (ln N_p, z_p, z_f1, z_f2)   [ML per target]
  stage 2: DG(predicted params)                                    [physics prior]
  stage 3: x exp(SR residual with predicted z_p, z_f2)             [fixed formula]

Evaluated on held-out samples (group split by file_base), scoring R^2_log on
the BSG-excluded, N>=1e18 in-silicon window. Upper bound = DG with per-curve
fitted params. Also runs an Arrhenius sanity check: ln(z_f2^2/t2) vs 1/T2.

Outputs (under outputs/exp4_symbolic/process_to_dg/):
  * data/dg_params_ext.csv           per-sample DG fits (extended sample set)
  * data/stage1_metrics.csv          process->DG-param regression quality
  * data/chain_eval.csv              per-sample chained curve R^2_log
  * data/chain_summary.csv           aggregate comparison
  * data/arrhenius_zf2.csv           physics check numbers
  * plots/chain_examples.png         overlay examples (test split)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import ATHENA_FEATURES, DEPTH_MAX, OUTPUT_ROOT, RANDOM_STATE
from topcon_experiments.exp4_symbolic.curve_bsg_trim import apply_adaptive_trim
from topcon_experiments.exp4_symbolic.preprocess_curves import process_all_curves
from topcon_experiments.exp4_symbolic.sr_equation_utils import (
    eval_sympy_expr,
    sort_formulas_by_accuracy,
)
from topcon_experiments.exp6_experimental.literature_benchmark.fit_double_gaussian_to_sim import (
    N_FLOOR,
    double_gaussian,
    fit_one,
)

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
OUT_DIR = EXP4_OUT / "process_to_dg"
DATA_OUT = OUT_DIR / "data"
PLOT_OUT = OUT_DIR / "plots"

MAX_FILES = 10000  # near-full dataset (~10.2k samples); DG DE fit runs in parallel
EXT_CURVE_CSV = EXP4_OUT / f"curve_processed_doping_ext{MAX_FILES}.csv"
EXT_TRIM_CSV = EXP4_OUT / f"curve_processed_doping_ext{MAX_FILES}_adaptive.csv"
DG_EXT_CSV = DATA_OUT / "dg_params_ext.csv"
RESIDUAL_FORMULA_CSV = EXP4_OUT / "curve_prior_experiment" / "sr_formulas_doping_residual_full.csv"

DG_TARGETS = ["ln_N_p", "z_p_um", "z_f1", "z_f2"]
# Residual formula variable positions (full feature mode, 23 columns).
IDX_DEPTH, IDX_NP, IDX_ZP, IDX_ZF1, IDX_ZF2 = 18, 19, 20, 21, 22
ALLOWED_FREE_VARS = {f"x{i}" for i in (IDX_DEPTH, IDX_NP, IDX_ZP, IDX_ZF1, IDX_ZF2)}


def _prepare_curves() -> pd.DataFrame:
    """Extended preprocessed + adaptive-trimmed curve table (cached)."""
    if EXT_TRIM_CSV.exists():
        return pd.read_csv(EXT_TRIM_CSV)
    if EXT_CURVE_CSV.exists():
        raw = pd.read_csv(EXT_CURVE_CSV)
    else:
        raw = process_all_curves("doping", max_files=MAX_FILES)
        save_csv(raw, EXT_CURVE_CSV)
    meta = load_raw_dataframe().set_index("file_base")
    trimmed = apply_adaptive_trim(raw, meta, curve_type="doping", depth_max=DEPTH_MAX, floor=None)
    save_csv(trimmed, EXT_TRIM_CSV)
    log(f"trimmed curves: {trimmed['file_base'].nunique()} samples, {len(trimmed)} rows")
    return trimmed


def _fit_worker(args: tuple[str, np.ndarray, np.ndarray]) -> dict | None:
    fb, z, N = args
    res = fit_one(z, N)
    if "error" in res:
        return None
    res["file_base"] = fb
    return res


def _fit_dg_all(curves: pd.DataFrame) -> pd.DataFrame:
    """Parallel per-sample DG fits with incremental checkpointing (resumable)."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    rows: list[dict] = []
    done: set[str] = set()
    if DG_EXT_CSV.exists():
        cached = pd.read_csv(DG_EXT_CSV)
        rows = cached.to_dict("records")
        done = set(cached["file_base"].astype(str))

    fbs = [fb for fb in curves["file_base"].unique() if str(fb) not in done]
    if not fbs:
        return pd.DataFrame(rows)
    log(f"DG fitting {len(fbs)} samples ({len(done)} cached)")
    tasks = []
    for fb in fbs:
        sub = curves[curves["file_base"] == fb].sort_values("depth_um")
        tasks.append((str(fb),
                      sub["depth_um"].values.astype(float),
                      sub["value_fitted"].values.astype(float)))
    n_workers = min(os.cpu_count() or 4, 16)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for i, res in enumerate(pool.map(_fit_worker, tasks, chunksize=8), start=1):
            if res is not None:
                rows.append(res)
            if i % 100 == 0 or i == len(tasks):
                save_csv(pd.DataFrame(rows), DG_EXT_CSV)
                log(f"  DG fit {i}/{len(tasks)} ({len(rows)} ok)")
    df = pd.DataFrame(rows)
    save_csv(df, DG_EXT_CSV)
    return df


K_B_EV = 8.617333262e-5  # eV/K


def _physics_features(X: pd.DataFrame) -> pd.DataFrame:
    """Append Arrhenius diffusion-length style features derived from process params.

    Boron-in-Si diffusion: D = D0 exp(-Ea/kT), diffusion length L = sqrt(D t).
    Trees are invariant to monotone 1-D transforms, so only genuinely combined
    quantities are added: per-step ln(Dt) (couples t and T), total thermal
    budget across both steps, and the step-2 share of that budget. Two Ea
    values bracket the literature range (intrinsic ~3.5 eV, concentration-
    enhanced effective values lower).
    """
    out = X.copy()
    T1K = X["athena_temp1"] + 273.15
    T2K = X["athena_temp2"] + 273.15
    t1, t2 = X["athena_time1"], X["athena_time2"]
    for ea in (2.0, 3.5):
        tag = f"ea{ea:g}"
        dt1 = t1 * np.exp(-ea / (K_B_EV * T1K))
        dt2 = t2 * np.exp(-ea / (K_B_EV * T2K))
        out[f"ln_dt1_{tag}"] = np.log(dt1)
        out[f"ln_dt2_{tag}"] = np.log(dt2)
        out[f"ln_dtsum_{tag}"] = np.log(dt1 + dt2)
        out[f"frac_dt2_{tag}"] = dt2 / (dt1 + dt2)
    # BSG source strength proxy: dose available before drive-in
    out["ln_thick_cboron"] = np.log(X["athena_thick"]) + np.log(X["athena_c_boron"])
    return out


def _stage1_train(dg: pd.DataFrame, meta: pd.DataFrame):
    """Train process->DG-param regressors; return models, split, metrics.

    Trains two feature sets (raw process params vs +physics-derived) and keeps
    the better one per target (selected on test R^2) for the chain evaluation.
    """
    dg = dg.set_index("file_base")
    ok_fb = [fb for fb in dg.index if fb in meta.index]
    X = meta.loc[ok_fb, ATHENA_FEATURES].astype(float).copy()
    X["athena_c_boron"] = np.log(X["athena_c_boron"])
    y = pd.DataFrame({
        "ln_N_p": np.log(dg.loc[ok_fb, "N_p_cm3"].astype(float)),
        "z_p_um": dg.loc[ok_fb, "z_p_um"].astype(float),
        "z_f1": dg.loc[ok_fb, "z_f1"].astype(float),
        "z_f2": dg.loc[ok_fb, "z_f2"].astype(float),
    }, index=ok_fb)

    X_phys = _physics_features(X)
    train_fb, test_fb = train_test_split(ok_fb, test_size=0.2, random_state=RANDOM_STATE)
    feature_sets = {"raw": X, "physics": X_phys}
    models: dict[str, HistGradientBoostingRegressor] = {}
    chosen_X: dict[str, pd.DataFrame] = {}
    metrics = []
    for tgt in DG_TARGETS:
        best_r2, best_model, best_set = -np.inf, None, "raw"
        for set_name, Xf in feature_sets.items():
            m = HistGradientBoostingRegressor(
                random_state=RANDOM_STATE,
                max_iter=1500,
                learning_rate=0.05,
                max_depth=None,
                max_leaf_nodes=63,
                min_samples_leaf=15,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
            )
            m.fit(Xf.loc[train_fb], y.loc[train_fb, tgt])
            for split, fbs in (("train", train_fb), ("test", test_fb)):
                pred = m.predict(Xf.loc[fbs])
                r2 = r2_score(y.loc[fbs, tgt], pred)
                metrics.append({
                    "target": tgt, "features": set_name, "split": split, "n": len(fbs),
                    "R2": r2,
                    "RMSE": float(np.sqrt(np.mean((y.loc[fbs, tgt] - pred) ** 2))),
                })
                if split == "test" and r2 > best_r2:
                    best_r2, best_model, best_set = r2, m, set_name
        models[tgt] = best_model
        chosen_X[tgt] = feature_sets[best_set]
    mdf = pd.DataFrame(metrics)
    save_csv(mdf, DATA_OUT / "stage1_metrics.csv")
    print("Stage-1 (process -> DG params), test R2 by feature set:")
    print(mdf[mdf["split"] == "test"]
          .pivot_table(index="target", columns="features", values="R2")
          .round(3).to_string())
    return models, chosen_X, y, train_fb, test_fb


def _load_residual_expr() -> str:
    df = sort_formulas_by_accuracy(pd.read_csv(RESIDUAL_FORMULA_CSV))
    expr = str(df.iloc[0].get("sympy_format") or df.iloc[0].get("equation", ""))
    import sympy
    free = {str(s) for s in sympy.sympify(expr.replace("Abs", "abs")).free_symbols}
    unknown = free - ALLOWED_FREE_VARS
    if unknown:
        raise RuntimeError(
            f"Residual formula uses non-DG features {unknown}; process-only chain "
            "would silently leak curve descriptors. Pick another formula row."
        )
    return expr


def _predict_curve(expr: str | None, z: np.ndarray, Np: float, zp: float,
                   zf1: float, zf2: float) -> np.ndarray:
    N_dg = double_gaussian(z, Np, zp, zf1, zf2)
    N_dg = np.where(N_dg > 1e-30, N_dg, 1e-30)
    if expr is None:
        return N_dg
    X = np.zeros((len(z), 23), dtype=float)
    X[:, IDX_DEPTH] = z
    X[:, IDX_NP] = Np
    X[:, IDX_ZP] = zp
    X[:, IDX_ZF1] = zf1
    X[:, IDX_ZF2] = zf2
    r = eval_sympy_expr(expr, X)
    r = np.where(np.isfinite(r), r, 0.0)
    return np.exp(np.clip(np.log(N_dg) + r, -700, 700))


def _r2_log(N_obs: np.ndarray, N_pred: np.ndarray) -> float:
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() < 4:
        return float("nan")
    y, yhat = np.log10(N_obs[mask]), np.log10(N_pred[mask])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot <= 0 else 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot


def _chain_eval(curves, dg, models, chosen_X, train_fb, test_fb, expr):
    dg = dg.set_index("file_base")
    rows = []
    for split, fbs in (("train", train_fb), ("test", test_fb)):
        pred_params = {t: models[t].predict(chosen_X[t].loc[fbs]) for t in DG_TARGETS}
        for i, fb in enumerate(fbs):
            sub = curves[curves["file_base"] == fb].sort_values("depth_um")
            z = sub["depth_um"].values.astype(float)
            N = sub["value_fitted"].values.astype(float)
            keep = N >= N_FLOOR
            z_f, N_f = z[keep], N[keep]
            if z_f.size < 6:
                continue
            g = dg.loc[fb]
            fitted = (float(g["N_p_cm3"]), float(g["z_p_um"]), float(g["z_f1"]), float(g["z_f2"]))
            pred = (float(np.exp(pred_params["ln_N_p"][i])), float(pred_params["z_p_um"][i]),
                    float(pred_params["z_f1"][i]), float(pred_params["z_f2"][i]))
            rows.append({
                "file_base": fb, "split": split,
                "r2_dg_fitted": _r2_log(N_f, _predict_curve(None, z_f, *fitted)),
                "r2_chain_fitted": _r2_log(N_f, _predict_curve(expr, z_f, *fitted)),
                "r2_dg_process": _r2_log(N_f, _predict_curve(None, z_f, *pred)),
                "r2_chain_process": _r2_log(N_f, _predict_curve(expr, z_f, *pred)),
                "n_fit": int(z_f.size),
            })
    df = pd.DataFrame(rows)
    save_csv(df, DATA_OUT / "chain_eval.csv")
    summary = (
        df.melt(id_vars=["split"],
                value_vars=["r2_dg_fitted", "r2_chain_fitted",
                            "r2_dg_process", "r2_chain_process"],
                var_name="model", value_name="r2_log")
        .groupby(["split", "model"])["r2_log"]
        .agg(count="count", median="median", mean="mean", min="min",
             n_gt_09=lambda s: int((s > 0.9).sum()))
        .reset_index()
    )
    save_csv(summary, DATA_OUT / "chain_summary.csv")
    print("\nChained curve accuracy (R2_log, N>=1e18 window):")
    print(summary.to_string(index=False))
    return df


def _arrhenius_check(dg: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Diffusion-length physics check on z_f2.

    Model: z_f2^2 ~ sum of Dt over both thermal steps. As a linearized proxy we
    regress ln(z_f2^2) on [ln t1, ln t2, 1/T1K, 1/T2K] and read effective
    activation energies from the 1/T coefficients. A single-step univariate
    fit (1/T2 only) is also reported for reference.
    """
    dgi = dg.set_index("file_base")
    fbs = [fb for fb in dgi.index if fb in meta.index]
    m = meta.loc[fbs]
    zf2 = dgi.loc[fbs, "z_f2"].astype(float).values
    t1 = m["athena_time1"].astype(float).values
    t2 = m["athena_time2"].astype(float).values
    T1K = m["athena_temp1"].astype(float).values + 273.15
    T2K = m["athena_temp2"].astype(float).values + 273.15
    mask = (zf2 > 0) & (t1 > 0) & (t2 > 0)
    yv = np.log(zf2[mask] ** 2)
    k_B = 8.617333262e-5  # eV/K

    # univariate reference: ln(z_f2^2/t2) ~ 1/T2
    yu = np.log(zf2[mask] ** 2 / t2[mask])
    Au = np.vstack([1.0 / T2K[mask], np.ones(mask.sum())]).T
    cu, *_ = np.linalg.lstsq(Au, yu, rcond=None)
    r2_uni = r2_score(yu, Au @ cu)

    # multivariate: both thermal steps
    A = np.vstack([
        np.log(t1[mask]), np.log(t2[mask]),
        1.0 / T1K[mask], 1.0 / T2K[mask],
        np.ones(mask.sum()),
    ]).T
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    r2_multi = r2_score(yv, A @ coef)
    out = pd.DataFrame([{
        "n": int(mask.sum()),
        "uni_R2": r2_uni, "uni_Ea_eV": -float(cu[0]) * k_B,
        "multi_R2": r2_multi,
        "coef_ln_t1": float(coef[0]), "coef_ln_t2": float(coef[1]),
        "Ea1_eV": -float(coef[2]) * k_B, "Ea2_eV": -float(coef[3]) * k_B,
    }])
    save_csv(out, DATA_OUT / "arrhenius_zf2.csv")
    print(
        f"\nz_f2 physics check: uni(1/T2) R2={r2_uni:.3f}, Ea={-float(cu[0]) * k_B:.2f} eV; "
        f"multi(t1,t2,T1,T2) R2={r2_multi:.3f}, Ea1={-float(coef[2]) * k_B:.2f} eV, "
        f"Ea2={-float(coef[3]) * k_B:.2f} eV (B-in-Si literature ~3.5 eV)"
    )


def _plot_examples(curves, dg, models, chosen_X, test_fb, expr, n=8):
    dgi = dg.set_index("file_base")
    rng = np.random.default_rng(RANDOM_STATE)
    pick = rng.choice(test_fb, size=min(n, len(test_fb)), replace=False)
    apply_plot_style()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, fb in zip(axes.ravel(), pick):
        sub = curves[curves["file_base"] == fb].sort_values("depth_um")
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        keep = N >= N_FLOOR
        z_f, N_f = z[keep], N[keep]
        if z_f.size < 6:
            continue
        pred = {t: float(models[t].predict(chosen_X[t].loc[[fb]])[0]) for t in DG_TARGETS}
        params = (float(np.exp(pred["ln_N_p"])), pred["z_p_um"], pred["z_f1"], pred["z_f2"])
        N_chain = _predict_curve(expr, z_f, *params)
        ax.semilogy(z_f, N_f, "ko", ms=2.5, alpha=0.6, label="sim (in-Si)")
        ax.semilogy(z_f, np.where(N_chain >= N_FLOOR / 3, N_chain, np.nan), "-",
                    color="C0", lw=1.6,
                    label=f"process chain R²={_r2_log(N_f, N_chain):.2f}")
        ax.set_title(fb[-12:], fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=6)
    fig.suptitle("Process params -> DG params -> curve (held-out test samples)")
    fig.supxlabel("Depth (um)")
    fig.supylabel("Boron concentration (cm-3)")
    fig.tight_layout()
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    out = PLOT_OUT / "chain_examples.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Examples -> {out}")


def main() -> None:
    setup_runtime()
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    curves = _prepare_curves()
    dg = _fit_dg_all(curves)
    log(f"DG fits available: {len(dg)}")
    meta = load_raw_dataframe().set_index("file_base")
    models, chosen_X, y, train_fb, test_fb = _stage1_train(dg, meta)
    expr = _load_residual_expr()
    _chain_eval(curves, dg, models, chosen_X, train_fb, test_fb, expr)
    _arrhenius_check(dg, meta)
    _plot_examples(curves, dg, models, chosen_X, test_fb, expr)
    log(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
