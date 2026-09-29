"""Per-sample doping + defect curve metrics for the full overlapping cohort.

One row = one ``file_base`` (Athena sample id) with both doping and defect curves.
Methods evaluated (RMSE + R2):

Doping (N>=1e18 window after adaptive BSG trim, R2/RMSE on log10):
  dg_selffit, dgresid_selffit, dg_sr, dgresid_sr, dg_ag, dgresid_ag

Defect (full depth, R2/RMSE on ln):
  prior_selffit, prior_sr, prior_ag

Outputs under paper_package_defect_and_doping_extras/03_joint_metrics/:
  * joint_metrics_original.csv       raw dataset (all cohort rows, no QC)
  * joint_metrics_preprocessed.csv   after dropping NaN / extreme R2 / RMSE
  * filter_rules.md                  QC rules documentation
  * joint_metrics_summary.csv        aggregate stats before/after filter
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import ATHENA_FEATURES, OUTPUT_ROOT
from topcon_experiments.exp4_symbolic.defect_shape_prior import _r2_ln, predict_prior
from topcon_experiments.exp4_symbolic.run_process_to_dg_chain import (
    DG_EXT_CSV,
    DG_TARGETS,
    _load_residual_expr,
    _physics_features,
    _predict_curve,
)
from topcon_experiments.exp4_symbolic.run_process_to_defect_theta import (
    AG_DIR as DEFECT_AG_DIR,
    PRIOR_CSV as DEFECT_PRIOR_CSV,
    THETA_TARGETS as DEFECT_THETA,
    WORK as DEFECT_WORK,
    _rebuild as defect_rebuild,
)
from topcon_experiments.exp4_symbolic.run_process_to_theta_sr import OUT_DIR as DOPING_THETA_DIR
from topcon_experiments.exp4_symbolic.sr_equation_utils import (
    eval_sympy_expr,
    sort_formulas_by_accuracy,
)
from topcon_experiments.exp6_experimental.literature_benchmark.fit_double_gaussian_to_sim import (
    N_FLOOR,
    fit_one,
)

DOPING_CURVE = OUTPUT_ROOT / "exp4_symbolic" / "curve_processed_doping_ext10000_adaptive.csv"
DEFECT_CURVE = OUTPUT_ROOT / "exp4_symbolic" / "curve_processed_defect_ext2000.csv"
OUT = OUTPUT_ROOT / "paper_package_defect_and_doping_extras" / "03_joint_metrics"

DOPING_METHODS = [
    "dg_selffit", "dgresid_selffit", "dg_sr", "dgresid_sr", "dg_ag", "dgresid_ag",
]
DEFECT_METHODS = ["prior_selffit", "prior_sr", "prior_ag"]

# QC thresholds for the preprocessed table
# Self-fit gates = curve / θ-fit quality (not process-model skill).
SELFFIT_DOP_R2_MIN = 0.5
SELFFIT_DEF_R2_MIN = -2.0
R2_MAX = 1.001
RMSE_DOP_MAX = 2.0          # log10 decades (all doping methods)
RMSE_DEF_MAX = 2.0          # ln units (all defect methods)
RMSE_MIN = 0.0
# Process-model R2 may be largely negative (esp. defect SR); that is method
# failure, not data corruption — do not drop rows for weak process R2.


def _r2_log10(N_obs: np.ndarray, N_pred: np.ndarray) -> float:
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() < 4:
        return float("nan")
    y, yhat = np.log10(N_obs[mask]), np.log10(N_pred[mask])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot <= 0 else 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot


def _rmse_log10(N_obs: np.ndarray, N_pred: np.ndarray) -> float:
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.log10(N_obs[mask]) - np.log10(N_pred[mask])) ** 2)))


def _rmse_ln(N_obs: np.ndarray, N_pred: np.ndarray) -> float:
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.log(N_obs[mask]) - np.log(N_pred[mask])) ** 2)))


def _fit_worker(args):
    fb, z, N = args
    res = fit_one(z, N)
    if "error" in res:
        return None
    res["file_base"] = fb
    return res


def _ensure_dg(curves_dop: pd.DataFrame, fbs: list[str]) -> pd.DataFrame:
    """Load cached DG fits; only attempt new fits when N>=1e18 window has >=6 pts."""
    dg = pd.read_csv(DG_EXT_CSV) if DG_EXT_CSV.exists() else pd.DataFrame()
    have = set(dg["file_base"].astype(str)) if not dg.empty else set()
    missing = [fb for fb in fbs if fb not in have]
    if not missing:
        return dg.set_index("file_base")

    by_fb = {
        str(fb): g.sort_values("depth_um")
        for fb, g in curves_dop.groupby("file_base", sort=False)
        if str(fb) in set(missing)
    }
    tasks = []
    skipped_thin = 0
    for fb in missing:
        sub = by_fb.get(str(fb))
        if sub is None:
            skipped_thin += 1
            continue
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        if (N >= N_FLOOR).sum() < 6:
            skipped_thin += 1
            continue
        tasks.append((str(fb), z, N))

    log(
        f"DG cache miss: {len(missing)}; "
        f"thin window skipped: {skipped_thin}; to fit: {len(tasks)}"
    )
    if not tasks:
        return dg.set_index("file_base") if not dg.empty else dg

    rows = dg.to_dict("records") if not dg.empty else []
    n_workers = min(os.cpu_count() or 4, 16)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for i, res in enumerate(pool.map(_fit_worker, tasks, chunksize=8), start=1):
            if res is not None:
                rows.append(res)
            if i % 100 == 0 or i == len(tasks):
                log(f"  DG fit {i}/{len(tasks)} ({len(rows)} total ok)")
                save_csv(pd.DataFrame(rows), DG_EXT_CSV)
    save_csv(pd.DataFrame(rows), DG_EXT_CSV)
    return pd.DataFrame(rows).set_index("file_base")


def _load_doping_theta_exprs() -> dict[str, str]:
    out = {}
    for tgt in DG_TARGETS:
        df = pd.read_csv(DOPING_THETA_DIR / f"sr_formulas_theta_{tgt}.csv")
        best = sort_formulas_by_accuracy(df).iloc[0]
        out[tgt] = str(best.get("sympy_format") or best["equation"])
    return out


def _load_defect_theta_exprs() -> dict[str, str]:
    out = {}
    for tgt in DEFECT_THETA:
        df = pd.read_csv(DEFECT_WORK / f"sr_formulas_defect_theta_v2_{tgt}.csv")
        best = sort_formulas_by_accuracy(df).iloc[0]
        out[tgt] = str(best.get("sympy_format") or best["equation"])
    return out


def _predict_ag_all(model_dir: Path, targets: list[str], X: pd.DataFrame) -> dict[str, np.ndarray]:
    from autogluon.tabular import TabularPredictor
    pred = {}
    for tgt in targets:
        pred[tgt] = TabularPredictor.load(str(model_dir / tgt)).predict(X).values.astype(float)
    return pred


def _build_X(fbs: list[str]) -> pd.DataFrame:
    meta = load_raw_dataframe().set_index("file_base")
    X = meta.loc[fbs, ATHENA_FEATURES].astype(float).copy()
    X["athena_c_boron"] = np.log(X["athena_c_boron"])
    return _physics_features(X)


def main() -> None:
    setup_runtime()
    OUT.mkdir(parents=True, exist_ok=True)

    log("Loading curves ...")
    curves_dop = pd.read_csv(DOPING_CURVE)
    curves_def = pd.read_csv(DEFECT_CURVE)
    defect_prior = pd.read_csv(DEFECT_PRIOR_CSV).set_index("file_base")

    fbs = sorted(set(curves_def["file_base"].astype(str)) & set(curves_dop["file_base"].astype(str)))
    fbs = [fb for fb in fbs if fb in defect_prior.index]
    log(f"Cohort with both curves + defect prior: {len(fbs)}")

    dg = _ensure_dg(curves_dop, fbs)
    # keep only samples that end up with a DG fit for doping metrics
    fbs_ok = [fb for fb in fbs if fb in dg.index]
    fbs_no_dg = [fb for fb in fbs if fb not in dg.index]
    log(f"With DG fit: {len(fbs_ok)}; without (doping metrics NaN): {len(fbs_no_dg)}")

    # Process features for ALL cohort (including no-DG) for defect; doping SR/AG only for fbs_ok
    X_all = _build_X(fbs)
    valid = X_all.notna().all(axis=1)
    fbs = [fb for fb in fbs if valid.loc[fb]]
    X_all = X_all.loc[fbs]
    fbs_ok = [fb for fb in fbs_ok if fb in fbs]
    log(f"After dropping missing Athena meta: {len(fbs)}")

    resid_expr = _load_residual_expr()
    dop_sr_expr = _load_doping_theta_exprs()
    def_sr_expr = _load_defect_theta_exprs()

    log("Predicting doping theta (SR + AG) on full cohort ...")
    Xv = X_all.values.astype(float)
    dop_sr = {t: eval_sympy_expr(dop_sr_expr[t], Xv) for t in DG_TARGETS}
    dop_ag = _predict_ag_all(DOPING_THETA_DIR / "autogluon_models", DG_TARGETS, X_all)

    log("Predicting defect theta (SR + AG) on full cohort ...")
    def_sr = {t: eval_sympy_expr(def_sr_expr[t], Xv) for t in DEFECT_THETA}
    def_ag = _predict_ag_all(DEFECT_AG_DIR, DEFECT_THETA, X_all)

    dop_by_fb = {
        str(fb): g.sort_values("depth_um")
        for fb, g in curves_dop.groupby("file_base", sort=False)
        if str(fb) in set(fbs)
    }
    def_by_fb = {
        str(fb): g.sort_values("depth_um")
        for fb, g in curves_def.groupby("file_base", sort=False)
        if str(fb) in set(fbs)
    }

    fb_to_i = {fb: i for i, fb in enumerate(fbs)}
    rows = []
    log(f"Scoring {len(fbs)} samples ...")
    for k, fb in enumerate(fbs, start=1):
        i = fb_to_i[fb]
        row: dict = {"file_base": fb}

        # ---- doping ----
        sub = dop_by_fb[fb]
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        keep = N >= N_FLOOR
        z_f, N_f = z[keep], N[keep]
        row["doping_n_fit"] = int(z_f.size)
        if fb in dg.index and z_f.size >= 6:
            g = dg.loc[fb]
            fitted = (float(g["N_p_cm3"]), float(g["z_p_um"]), float(g["z_f1"]), float(g["z_f2"]))
            sr_p = (
                float(np.exp(dop_sr["ln_N_p"][i])), float(dop_sr["z_p_um"][i]),
                float(dop_sr["z_f1"][i]), float(dop_sr["z_f2"][i]),
            )
            ag_p = (
                float(np.exp(dop_ag["ln_N_p"][i])), float(dop_ag["z_p_um"][i]),
                float(dop_ag["z_f1"][i]), float(dop_ag["z_f2"][i]),
            )
            preds = {
                "dg_selffit": _predict_curve(None, z_f, *fitted),
                "dgresid_selffit": _predict_curve(resid_expr, z_f, *fitted),
                "dg_sr": _predict_curve(None, z_f, *sr_p) if all(np.isfinite(sr_p)) else None,
                "dgresid_sr": _predict_curve(resid_expr, z_f, *sr_p) if all(np.isfinite(sr_p)) else None,
                "dg_ag": _predict_curve(None, z_f, *ag_p) if all(np.isfinite(ag_p)) else None,
                "dgresid_ag": _predict_curve(resid_expr, z_f, *ag_p) if all(np.isfinite(ag_p)) else None,
            }
            for m in DOPING_METHODS:
                Np = preds[m]
                if Np is None:
                    row[f"doping_{m}_R2"] = np.nan
                    row[f"doping_{m}_RMSE"] = np.nan
                else:
                    row[f"doping_{m}_R2"] = _r2_log10(N_f, Np)
                    row[f"doping_{m}_RMSE"] = _rmse_log10(N_f, Np)
        else:
            for m in DOPING_METHODS:
                row[f"doping_{m}_R2"] = np.nan
                row[f"doping_{m}_RMSE"] = np.nan

        # ---- defect ----
        sub = def_by_fb[fb]
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        keep = np.isfinite(N) & (N > 0)
        z_f, N_f = z[keep], N[keep]
        row["defect_n_fit"] = int(z_f.size)
        p = defect_prior.loc[fb]
        N_sf = predict_prior(
            z_f, str(p["prior_type"]), float(p["ln_Ns"]), float(p["ln_Nb"]), float(p["L_um"])
        )
        N_sr = defect_rebuild(z_f, def_sr["ln_Ns"][i], def_sr["ln_Nb"][i], def_sr["ln_L"][i])
        N_ag = defect_rebuild(z_f, def_ag["ln_Ns"][i], def_ag["ln_Nb"][i], def_ag["ln_L"][i])
        dpreds = {"prior_selffit": N_sf, "prior_sr": N_sr, "prior_ag": N_ag}
        for m in DEFECT_METHODS:
            Np = dpreds[m]
            row[f"defect_{m}_R2"] = _r2_ln(N_f, Np)
            row[f"defect_{m}_RMSE"] = _rmse_ln(N_f, Np)

        rows.append(row)
        if k % 200 == 0 or k == len(fbs):
            log(f"  scored {k}/{len(fbs)}")


    original = pd.DataFrame(rows)
    orig_path = OUT / "joint_metrics_original.csv"
    save_csv(original, orig_path)
    log(f"Original dataset -> {orig_path} ({len(original)} rows)")

    # ---- preprocess / filter ----
    metric_cols = (
        [f"doping_{m}_R2" for m in DOPING_METHODS]
        + [f"doping_{m}_RMSE" for m in DOPING_METHODS]
        + [f"defect_{m}_R2" for m in DEFECT_METHODS]
        + [f"defect_{m}_RMSE" for m in DEFECT_METHODS]
    )
    filt = original.copy()
    # require finite doping + defect metrics (complete bilateral evaluation)
    filt = filt.dropna(subset=metric_cols)

    r2_cols = [c for c in metric_cols if c.endswith("_R2")]
    rmse_dop = [c for c in metric_cols if c.startswith("doping_") and c.endswith("_RMSE")]
    rmse_def = [c for c in metric_cols if c.startswith("defect_") and c.endswith("_RMSE")]

    ok = np.ones(len(filt), dtype=bool)
    ok &= (filt["doping_n_fit"] >= 6).values & (filt["defect_n_fit"] >= 6).values
    # Self-fit quality: sample is evaluable / prior is trustworthy
    ok &= filt["doping_dg_selffit_R2"].between(
        SELFFIT_DOP_R2_MIN, R2_MAX, inclusive="both"
    ).values
    ok &= filt["defect_prior_selffit_R2"].between(
        SELFFIT_DEF_R2_MIN, R2_MAX, inclusive="both"
    ).values
    # Universal numeric sanity (no impossible R2/RMSE)
    for c in r2_cols:
        ok &= (filt[c] <= R2_MAX).values & np.isfinite(filt[c].values)
    for c in rmse_dop:
        ok &= (filt[c] >= RMSE_MIN).values & (filt[c] <= RMSE_DOP_MAX).values
    for c in rmse_def:
        ok &= (filt[c] >= RMSE_MIN).values & (filt[c] <= RMSE_DEF_MAX).values

    preprocessed = filt.loc[ok].reset_index(drop=True)
    prep_path = OUT / "joint_metrics_preprocessed.csv"
    save_csv(preprocessed, prep_path)
    log(f"Preprocessed dataset -> {prep_path} ({len(preprocessed)} rows; "
        f"removed {len(original) - len(preprocessed)} from original)")

    # summaries
    def _agg(df: pd.DataFrame, tag: str) -> list[dict]:
        out = []
        for m in DOPING_METHODS:
            s_r, s_e = df[f"doping_{m}_R2"], df[f"doping_{m}_RMSE"]
            out.append({
                "table": tag, "task": "doping", "method": m, "n": int(s_r.notna().sum()),
                "median_R2": float(s_r.median()), "mean_R2": float(s_r.mean()),
                "median_RMSE": float(s_e.median()), "mean_RMSE": float(s_e.mean()),
            })
        for m in DEFECT_METHODS:
            s_r, s_e = df[f"defect_{m}_R2"], df[f"defect_{m}_RMSE"]
            out.append({
                "table": tag, "task": "defect", "method": m, "n": int(s_r.notna().sum()),
                "median_R2": float(s_r.median()), "mean_R2": float(s_r.mean()),
                "median_RMSE": float(s_e.median()), "mean_RMSE": float(s_e.mean()),
            })
        return out

    summary = pd.DataFrame(
        _agg(original, "original") + _agg(preprocessed, "preprocessed")
    )
    save_csv(summary, OUT / "joint_metrics_summary.csv")

    (OUT / "filter_rules.md").write_text(
        f"""# Joint metrics filter rules

## Cohort
Samples with both doping (adaptive-trim) and defect curves in the ext2000/ext10000
caches, plus a fitted defect prior (v2). Doping DG params require an
`N >= 1e18` window with at least 6 points; otherwise doping metrics are NaN.

## Original dataset
`joint_metrics_original.csv` — one row per `file_base` (n≈2000). Defect metrics
are always filled; doping metrics are NaN when the high-concentration window is
too thin for a DG fit (~967 samples).

## Preprocessed dataset
`joint_metrics_preprocessed.csv` keeps a row only if **all** of the following hold:

1. Every doping/defect R2 and RMSE column is finite (complete bilateral eval).
2. `doping_n_fit >= 6` and `defect_n_fit >= 6`.
3. **Self-fit quality** (data / θ-fit trustworthiness, not process-model skill):
   - `doping_dg_selffit_R2` in `[{SELFFIT_DOP_R2_MIN}, {R2_MAX}]`
   - `defect_prior_selffit_R2` in `[{SELFFIT_DEF_R2_MIN}, {R2_MAX}]`
4. Numeric sanity for **all** methods:
   - every R2 `<= {R2_MAX}` and finite
   - doping RMSE in `[{RMSE_MIN}, {RMSE_DOP_MAX}]` (log10 decades)
   - defect RMSE in `[{RMSE_MIN}, {RMSE_DEF_MAX}]` (ln units)

Process-model R2 may be largely negative (especially defect PySR). That reflects
method skill, not corrupt samples, so rows are **not** dropped for weak
`*_sr` / `*_ag` R2.

## Metric definitions
- **Doping R2 / RMSE**: on `log10(N)` over the BSG-trimmed, `N>=1e18` segment.
- **Defect R2 / RMSE**: on `ln(N)` over the full positive concentration curve.

## Methods
Doping: `dg_selffit`, `dgresid_selffit`, `dg_sr`, `dgresid_sr`, `dg_ag`, `dgresid_ag`
Defect: `prior_selffit`, `prior_sr`, `prior_ag`
""",
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    log(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
