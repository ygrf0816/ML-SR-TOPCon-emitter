"""Benchmark the prior-injected SR formulas (residual, feature_aug) on the
literature profiles D1-D10.

For each literature curve we:
  1. Generate N(z) from the Table-1 double-Gaussian params (truncated at N>=1e18).
  2. Fit our own 4-param double-Gaussian to that curve to obtain the prior
     (N_p, z_p, z_f1, z_f2) — this is what the SR formula was trained to expect.
  3. Feed (literature features + DG params + ln_N_DG) to the prior SR formula
     to get the final predicted N(z).
  4. Compute R²_log between predicted and literature N(z) on the N>=1e18 segment.

We also score the *raw* double-Gaussian (step 2 only) and the *raw* prior SR
formula using literature-derived descriptors (no DG fit) for comparison.

Outputs (under outputs/exp6_experimental/literature_benchmark/prior_/):
  * data/prior_lit_fit_summary.csv
  * data/origin_literature_all_curves.csv  (Origin wide table, D1-D10)
  * data/origin_literature_all_metadata.csv
  * plots/prior_lit_fit_<pid>.png     (literature vs DG-prior vs prior-SR)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import LOG_FEATURES, MODEL2_FEATURES, ATHENA_FEATURES
from topcon_experiments.exp4_symbolic.sr_equation_utils import eval_sympy_expr, sort_formulas_by_accuracy
from topcon_experiments.exp6_experimental.literature_benchmark.fit_double_gaussian_to_sim import (
    BOUNDS as DG_BOUNDS,
    N_FLOOR,
    double_gaussian,
)
from topcon_experiments.exp6_experimental.literature_benchmark.fit_sr_to_literature import (
    _load_reference_curve,
    extract_all_descriptors,
    load_parsed_table,
)

PRIOR_EXP_DIR = Path(__file__).resolve().parents[2] / "outputs" / "exp4_symbolic" / "curve_prior_experiment"
BASE_OUT = Path(__file__).resolve().parents[1] / "outputs" / "literature_benchmark"
OUT_DIR = BASE_OUT / "prior_"
DATA_OUT = OUT_DIR / "data"
PLOT_OUT = OUT_DIR / "plots"

LOG_IN_MODEL = set(LOG_FEATURES)
DG_FEAT_NAMES = ["dg_N_p", "dg_z_p", "dg_z_f1", "dg_z_f2"]

# prior variants to benchmark
PRIOR_VARIANTS = [
    ("prior_residual_full", "residual", "full"),
    ("prior_residual_athena", "residual", "athena"),
    ("prior_feature_aug_full", "feature_aug", "full"),
    ("prior_feature_aug_athena", "feature_aug", "athena"),
]


def _best_expr(path: Path) -> str | None:
    if not path.exists():
        return None
    df = sort_formulas_by_accuracy(pd.read_csv(path))
    return str(df.iloc[0].get("sympy_format") or df.iloc[0].get("equation", ""))


def _fit_dg_to_lit(z: np.ndarray, N: np.ndarray) -> dict:
    """Fit 4-param DG to a literature curve on N>=N_FLOOR segment."""
    keep = N >= N_FLOOR
    z_f, N_f = z[keep], N[keep]
    if z_f.size < 6:
        return {"error": "too few points above floor"}
    log_obs = np.log10(N_f)
    ipk = int(np.argmax(N_f))
    N_p0, z_p0 = float(N_f[ipk]), float(z_f[ipk])
    local = [
        (max(DG_BOUNDS[0][0], N_p0 * 1e-2), min(DG_BOUNDS[0][1], N_p0 * 1e2)),
        (max(DG_BOUNDS[1][0], z_p0 - 0.15), min(DG_BOUNDS[1][1], z_p0 + 0.15)),
        DG_BOUNDS[2],
        DG_BOUNDS[3],
    ]
    local = [(min(lo, hi), max(lo, hi)) for lo, hi in local]

    def obj(p):
        pred = double_gaussian(z_f, *p)
        if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
            return 1e6
        return float(np.sum((log_obs - np.log10(pred)) ** 2))

    res = differential_evolution(obj, local, seed=1, tol=1e-8, maxiter=400,
                                 popsize=25, polish=True, workers=1)
    Np, zp, zf1, zf2 = (float(v) for v in res.x)
    pred = double_gaussian(z_f, Np, zp, zf1, zf2)
    mask = (N_f > 0) & (pred > 0)
    y = np.log10(N_f[mask]); yhat = np.log10(pred[mask])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {"N_p_cm3": Np, "z_p_um": zp, "z_f1": zf1, "z_f2": zf2,
            "r2_log": r2, "n_fit": int(z_f.size)}


def _prior_feature_vector(feat_vals: dict[str, float], dg_params: dict,
                          z: np.ndarray, *, feature_mode: str, with_ln_dg: bool) -> np.ndarray:
    feat_cols = MODEL2_FEATURES if feature_mode == "full" else ATHENA_FEATURES
    cols = list(feat_cols) + ["depth_um"] + DG_FEAT_NAMES
    if with_ln_dg:
        cols = cols + ["ln_N_dg"]
    n = len(z)
    X = np.zeros((n, len(cols)), dtype=float)
    for j, c in enumerate(feat_cols):
        v = float(feat_vals.get(c, 0.0))
        if c in LOG_IN_MODEL:
            v = np.log(max(v, 1e-30))
        X[:, j] = v
    X[:, len(feat_cols)] = z
    Np = float(dg_params["N_p_cm3"]); zp = float(dg_params["z_p_um"])
    zf1 = float(dg_params["z_f1"]); zf2 = float(dg_params["z_f2"])
    # raw DG params (matches training convention)
    X[:, len(feat_cols) + 1] = Np
    X[:, len(feat_cols) + 2] = zp
    X[:, len(feat_cols) + 3] = zf1
    X[:, len(feat_cols) + 4] = zf2
    if with_ln_dg:
        N_dg = double_gaussian(z, Np, zp, zf1, zf2)
        N_dg = np.where(N_dg > 1e-30, N_dg, 1e-30)
        X[:, len(feat_cols) + 5] = np.log(N_dg)
    return X


def _predict_prior(expr: str, feat_vals: dict, dg_params: dict, z: np.ndarray,
                   *, variant: str, feature_mode: str) -> np.ndarray:
    with_ln_dg = (variant != "residual")
    X = _prior_feature_vector(feat_vals, dg_params, z,
                              feature_mode=feature_mode, with_ln_dg=with_ln_dg)
    out = eval_sympy_expr(expr, X)
    out = np.where(np.isfinite(out), out, np.nan)
    if variant == "residual":
        Np = float(dg_params["N_p_cm3"]); zp = float(dg_params["z_p_um"])
        zf1 = float(dg_params["z_f1"]); zf2 = float(dg_params["z_f2"])
        N_dg = double_gaussian(z, Np, zp, zf1, zf2)
        N_dg = np.where(N_dg > 1e-30, N_dg, 1e-30)
        ln_N = np.log(N_dg) + out
    else:
        ln_N = out
    return np.exp(np.clip(ln_N, -700, 700))


def _r2_log(N_obs, N_pred):
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() < 4:
        return float("nan")
    y = np.log10(N_obs[mask]); yhat = np.log10(N_pred[mask])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def _rmse_log(N_obs, N_pred):
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.log10(N_obs[mask]) - np.log10(N_pred[mask])) ** 2)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(exist_ok=True)
    PLOT_OUT.mkdir(exist_ok=True)

    parsed = load_parsed_table().set_index("profile_id")
    desc_df = extract_all_descriptors(out_dir=DATA_OUT)

    # cache prior expressions
    prior_exprs = {}
    for label, variant, fm in PRIOR_VARIANTS:
        p = PRIOR_EXP_DIR / f"sr_formulas_doping_{variant}_{fm}.csv"
        e = _best_expr(p)
        if e is not None:
            prior_exprs[label] = e

    rows = []
    origin_curves: dict[str, dict[str, np.ndarray]] = {}
    pids = sorted(parsed.index.unique())
    for pid in pids:
        z_ref, N_ref = _load_reference_curve(pid)
        if z_ref.size == 0:
            continue
        # 1) fit DG to literature curve as the prior
        dg = _fit_dg_to_lit(z_ref, N_ref)
        if "error" in dg:
            rows.append({"profile_id": pid, "model": "double_gaussian",
                         "error": dg["error"]})
        else:
            keep = N_ref >= N_FLOOR
            N_dg_pred = double_gaussian(z_ref, dg["N_p_cm3"], dg["z_p_um"],
                                        dg["z_f1"], dg["z_f2"])
            origin_curves[str(pid)] = {
                "true_x": z_ref[keep].copy(),
                "true_y": N_ref[keep].copy(),
                "dg_x": z_ref[keep].copy(),
                "dg_y": N_dg_pred[keep].copy(),
            }
            rows.append({"profile_id": pid, "model": "double_gaussian",
                         "r2_log": _r2_log(N_ref, N_dg_pred),
                         "rmse_log": _rmse_log(N_ref, N_dg_pred),
                         "n_fit": int(keep.sum()),
                         "dg_N_p": dg["N_p_cm3"], "dg_z_p": dg["z_p_um"],
                         "dg_z_f1": dg["z_f1"], "dg_z_f2": dg["z_f2"]})

        # 2) prior SR formulas
        desc_row = desc_df[desc_df["profile_id"] == pid].iloc[0] if pid in desc_df["profile_id"].values else None
        if desc_row is None or "error" in dg:
            continue
        feat_vals = {c: float(desc_row[c]) for c in (MODEL2_FEATURES + ATHENA_FEATURES) if c in desc_row}
        feat_vals["doping_R_sheet"] = float(parsed.loc[pid, "R_sheet_measured"])

        for label, variant, fm in PRIOR_VARIANTS:
            if label not in prior_exprs:
                continue
            try:
                N_pred = _predict_prior(prior_exprs[label], feat_vals, dg, z_ref,
                                        variant=variant, feature_mode=fm)
            except Exception as e:
                rows.append({"profile_id": pid, "model": label, "error": str(e)})
                continue
            rows.append({"profile_id": pid, "model": label,
                         "r2_log": _r2_log(N_ref, N_pred),
                         "rmse_log": _rmse_log(N_ref, N_pred),
                         "n_fit": int((N_ref >= N_FLOOR).sum())})
            if label == "prior_residual_full" and str(pid) in origin_curves:
                keep = N_ref >= N_FLOOR
                origin_curves[str(pid)]["pred_x"] = z_ref[keep].copy()
                origin_curves[str(pid)]["pred_y"] = N_pred[keep].copy()

        # 3) plot
        apply_plot_style()
        fig, ax = plt.subplots(figsize=(8, 5))
        keep = N_ref >= N_FLOOR
        ax.semilogy(z_ref[keep], N_ref[keep], "ko", ms=4, alpha=0.7,
                    label="Literature (double Gaussian)")
        if "error" not in dg:
            N_dg_pred = double_gaussian(z_ref, dg["N_p_cm3"], dg["z_p_um"],
                                        dg["z_f1"], dg["z_f2"])
            N_dg_clip = np.where(N_dg_pred >= N_FLOOR, N_dg_pred, np.nan)
            ax.semilogy(z_ref, N_dg_clip, "--", color="C2", lw=1.5,
                        label=f"DG prior (R²={_r2_log(N_ref, N_dg_pred):.2f})")
        colors = {"prior_residual_full": "C0", "prior_residual_athena": "C1",
                  "prior_feature_aug_full": "C3", "prior_feature_aug_athena": "C4"}
        for label, variant, fm in PRIOR_VARIANTS:
            if label not in prior_exprs:
                continue
            try:
                N_pred = _predict_prior(prior_exprs[label], feat_vals, dg, z_ref,
                                        variant=variant, feature_mode=fm)
            except Exception:
                continue
            r2 = _r2_log(N_ref, N_pred)
            N_pred_clip = np.where((N_pred >= N_FLOOR) & np.isfinite(N_pred), N_pred, np.nan)
            ax.semilogy(z_ref, N_pred_clip, "-", lw=1.8, color=colors.get(label, "C5"),
                        label=f"{label} (R²={r2:.2f})")
        ax.axhline(N_FLOOR, color="0.5", ls="--", lw=0.8)
        y_max = float(np.nanmax(N_ref[keep])) * 3.0 if keep.any() else 1e20
        ax.set_ylim(N_FLOOR / 3.0, y_max)
        ax.set_xlim(0.0, max(float(z_ref.max()), 0.5))
        ax.set_xlabel("Depth (um)")
        ax.set_ylabel("Boron concentration (cm-3)")
        ax.set_title(f"{pid}: prior-SR vs literature (N>=1e18)")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        out = PLOT_OUT / f"prior_lit_fit_{pid}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out.name}")

    df = pd.DataFrame(rows)
    save_csv(df, DATA_OUT / "prior_lit_fit_summary.csv")
    ok = df[df["r2_log"].notna()]
    summary = ok.groupby("model")["r2_log"].agg(
        ["count", "median", "mean", "min", "max",
         lambda s: float((s > 0.9).sum()),
         lambda s: float((s > 0.95).sum())]
    ).rename(columns={"<lambda_0>": "n_r2>0.9", "<lambda_1>": "n_r2>0.95"})
    print(summary.to_string())
    summary.to_csv(DATA_OUT / "prior_lit_summary.csv")

    # Origin export: all D1-D10 literature profiles, horizontally concatenated.
    wide: dict[str, pd.Series] = {}
    meta_rows = []
    for pid in sorted(origin_curves, key=lambda s: int(s[1:]) if s[1:].isdigit() else s):
        curve = origin_curves[pid]
        if "pred_y" not in curve:
            continue
        for suffix in ("true_x", "true_y", "dg_x", "dg_y", "pred_x", "pred_y"):
            wide[f"{pid}_{suffix}"] = pd.Series(curve[suffix])
        sr_row = df[(df["profile_id"] == pid) & (df["model"] == "prior_residual_full")]
        dg_row = df[(df["profile_id"] == pid) & (df["model"] == "double_gaussian")]
        meta_rows.append({
            "profile_id": pid,
            "sr_r2_log": float(sr_row.iloc[0]["r2_log"]) if not sr_row.empty else np.nan,
            "sr_rmse_log": float(sr_row.iloc[0]["rmse_log"]) if not sr_row.empty else np.nan,
            "dg_r2_log": float(dg_row.iloc[0]["r2_log"]) if not dg_row.empty else np.nan,
            "dg_rmse_log": float(dg_row.iloc[0]["rmse_log"]) if not dg_row.empty else np.nan,
            "n_fit": int(sr_row.iloc[0]["n_fit"]) if not sr_row.empty else 0,
        })
    if wide:
        save_csv(pd.DataFrame(wide), DATA_OUT / "origin_literature_all_curves.csv")
        save_csv(pd.DataFrame(meta_rows), DATA_OUT / "origin_literature_all_metadata.csv")
    print(f"Summary -> {DATA_OUT / 'prior_lit_fit_summary.csv'}")


if __name__ == "__main__":
    main()
