"""Evaluate the 3 prior-injected SR variants against the literature DG and the
5 base SR formulas, on the SAME sim-set window (N>=1e18).

Reuses the per-sample DG fits cached by run_curve_prior_sr_experiment.py and
reconstructs each prior-SR variant's full N(z) from its stored target convention:

  * residual      : N(z) = N_DG(z) * exp(SR(feat, z))          (SR target = ln N - ln N_DG)
  * feature_aug   : N(z) = exp(SR(feat, z))                    (SR target = ln N; DG params + ln_N_dg as features)
  * struct_bias   : N(z) = exp(SR(feat, z))                    (same target as feature_aug, custom gauss_kernel op)

The base 5 SR variants and the plain double-Gaussian are also scored for an
apples-to-apples comparison.

Outputs (under outputs/exp6_experimental/prior_sr_eval/):
  * data/prior_sr_r2.csv        (file_base x model x r2_log/rmse_log/n_fit)
  * data/prior_sr_summary.csv   (per-model aggregate)
  * data/origin_sim_best18_curves.csv (Origin wide table, best 18 by RMSE_log)
  * data/origin_sim_best18_metadata.csv
  * plots/prior_sr_r2_box.png   (R² distribution per model)
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
from topcon_experiments.config import LOG_FEATURES, MODEL2_FEATURES, ATHENA_FEATURES
from topcon_experiments.exp4_symbolic.sr_equation_utils import eval_sympy_expr, sort_formulas_by_accuracy
from topcon_experiments.exp4_symbolic.sr_split_utils import split_group_train_test
from topcon_experiments.exp6_experimental.literature_benchmark.fit_double_gaussian_to_sim import (
    N_FLOOR,
    double_gaussian,
)
from topcon_experiments.exp6_experimental.literature_benchmark.fit_sr_to_literature import (
    VARIANTS as BASE_VARIANTS,
    _best_expr,
)

PRIOR_EXP_DIR = Path(__file__).resolve().parents[2] / "outputs" / "exp4_symbolic" / "curve_prior_experiment"
# Use the BSG-removed (adaptive trim) window to match the DG prior fitting
# convention: DG describes the in-silicon diffusion profile, not the BSG layer.
CURVE_CSV = Path(__file__).resolve().parents[2] / "outputs" / "exp4_symbolic" / "curve_processed_doping_adaptive.csv"
FULL_CURVE_CSV = Path(__file__).resolve().parents[2] / "outputs" / "exp4_symbolic" / "curve_processed_doping.csv"
EXP_OUT = Path(__file__).resolve().parents[1] / "outputs" / "prior_sr_eval"
DATA_OUT = EXP_OUT / "data"
PLOT_OUT = EXP_OUT / "plots"
SAMPLE_PLOT_OUT = PLOT_OUT / "sim_curve_checks"

LOG_IN_MODEL = set(LOG_FEATURES)
DG_LOG_FEAT = {"dg_N_p"}
DG_FEAT_NAMES = ["dg_N_p", "dg_z_p", "dg_z_f1", "dg_z_f2"]


def _r2_log(N_obs, N_pred):
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() < 4:
        return float("nan")
    y = np.log10(N_obs[mask])
    yhat = np.log10(N_pred[mask])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def _rmse_log(N_obs, N_pred):
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.log10(N_obs[mask]) - np.log10(N_pred[mask])) ** 2)))


def _prior_feature_vector(feat_row, dg_row, z, *, feature_mode, with_ln_dg):
    feat_cols = MODEL2_FEATURES if feature_mode == "full" else ATHENA_FEATURES
    cols = list(feat_cols) + ["depth_um"] + DG_FEAT_NAMES
    if with_ln_dg:
        cols = cols + ["ln_N_dg"]
    n = len(z)
    X = np.zeros((n, len(cols)), dtype=float)
    for j, c in enumerate(feat_cols):
        v = float(feat_row[c])
        if c in LOG_IN_MODEL:
            v = np.log(max(v, 1e-30))
        X[:, j] = v
    X[:, len(feat_cols)] = z
    N_p = float(dg_row["N_p_cm3"]); z_p = float(dg_row["z_p_um"])
    z_f1 = float(dg_row["z_f1"]); z_f2 = float(dg_row["z_f2"])
    # DG params are passed RAW (no log transform) — matches _build_prior_dataset.
    # Positions: depth at len(feat_cols); DG params at len(feat_cols)+1..+4.
    X[:, len(feat_cols) + 1] = N_p
    X[:, len(feat_cols) + 2] = z_p
    X[:, len(feat_cols) + 3] = z_f1
    X[:, len(feat_cols) + 4] = z_f2
    if with_ln_dg:
        N_dg = double_gaussian(z, N_p, z_p, z_f1, z_f2)
        N_dg = np.where(N_dg > 1e-30, N_dg, 1e-30)
        X[:, len(feat_cols) + 5] = np.log(N_dg)
    return X


def _load_prior_formula(variant, feature_mode):
    p = PRIOR_EXP_DIR / f"sr_formulas_doping_{variant}_{feature_mode}.csv"
    if not p.exists():
        return None
    df = sort_formulas_by_accuracy(pd.read_csv(p))
    return str(df.iloc[0].get("sympy_format") or df.iloc[0].get("equation", ""))


PRIOR_VARIANTS = [
    ("prior_residual_full", "residual", "full", True, "residual"),
    ("prior_residual_athena", "residual", "athena", True, "residual"),
    ("prior_feature_aug_full", "feature_aug", "full", True, "feature_aug"),
    ("prior_feature_aug_athena", "feature_aug", "athena", True, "feature_aug"),
]


def _plot_bsg_checked_sample(
    fb: str,
    full_sub: pd.DataFrame,
    trim_sub: pd.DataFrame,
    dg_row: pd.Series,
    z_prior: np.ndarray,
    N_prior: np.ndarray,
    r2_prior: float,
    split: str,
) -> None:
    """Plot the excluded BSG step and the actual in-silicon fit window."""
    z_all = full_sub["depth_um"].to_numpy(dtype=float)
    N_all = full_sub["value_fitted"].to_numpy(dtype=float)
    z = trim_sub["depth_um"].to_numpy(dtype=float)
    N = trim_sub["value_fitted"].to_numpy(dtype=float)
    cut_depth = float(trim_sub["adaptive_depth_min_um"].iloc[0])
    keep_floor = N >= N_FLOOR

    N_dg = double_gaussian(
        z,
        float(dg_row["N_p_cm3"]),
        float(dg_row["z_p_um"]),
        float(dg_row["z_f1"]),
        float(dg_row["z_f2"]),
    )

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    excluded = z_all < cut_depth - 1e-12
    ax.semilogy(z_all, N_all, color="0.75", lw=1.0, label="Full simulated profile")
    if excluded.any():
        ax.semilogy(
            z_all[excluded],
            N_all[excluded],
            color="C3",
            lw=2.0,
            label="Excluded BSG/cliff",
        )
    ax.semilogy(z[keep_floor], N[keep_floor], "ko", ms=2.5, alpha=0.6,
                label="Fit window (in-Si, N>=1e18)")
    ax.semilogy(z, np.where(N_dg >= N_FLOOR, N_dg, np.nan), "--",
                color="C2", lw=1.6, label="Double Gaussian")
    ax.semilogy(z_prior, np.where(N_prior >= N_FLOOR, N_prior, np.nan), "-",
                color="C0", lw=1.8, label=f"DG × exp(SR), R²={r2_prior:.3f}")
    ax.axvline(cut_depth, color="C3", ls=":", lw=1.2,
               label=f"Adaptive BSG cut={cut_depth:.3f} um")
    ax.axhline(N_FLOOR, color="0.45", ls="--", lw=0.8)
    ax.set_xlim(float(z_all.min()), min(2.0, float(z_all.max())))
    positive = N_all[np.isfinite(N_all) & (N_all > 0)]
    if positive.size:
        ax.set_ylim(max(N_FLOOR / 3.0, float(positive.min()) / 2.0),
                    float(positive.max()) * 2.0)
    ax.set_xlabel("Physical depth (um; not re-zeroed after trimming)")
    ax.set_ylabel("Boron concentration (cm-3)")
    ax.set_title(f"{fb} [{split}] — BSG-excluded prior-SR check")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    SAMPLE_PLOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAMPLE_PLOT_OUT / f"prior_fit_{fb}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    SAMPLE_PLOT_OUT.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(CURVE_CSV)
    full_curves = pd.read_csv(FULL_CURVE_CSV)
    meta = load_raw_dataframe().set_index("file_base")
    dg_cache = PRIOR_EXP_DIR / "data" / "dg_params.csv"
    if not dg_cache.exists():
        raise FileNotFoundError(f"Run run_curve_prior_sr_experiment first: {dg_cache}")
    dg = pd.read_csv(dg_cache).set_index("file_base")
    # Reproduce the exact group split used when the SR formula was trained.
    # Metrics can therefore be reported separately for genuinely unseen curves.
    dg_groups = dg.index.to_numpy(dtype=str)
    train_i, test_i = split_group_train_test(dg_groups)
    split_by_fb = {str(dg_groups[i]): "train" for i in train_i}
    split_by_fb.update({str(dg_groups[i]): "test" for i in test_i})

    # cache base SR exprs (same set as compare_dg_vs_sr_on_sim)
    base_exprs = {}
    for label, v in BASE_VARIANTS.items():
        if v.formula_path.exists():
            try:
                base_exprs[label] = _best_expr(v.formula_path)
            except Exception:
                pass
    # cache prior exprs
    prior_exprs = {}
    for label, variant, fm, _, _ in PRIOR_VARIANTS:
        e = _load_prior_formula(variant, fm)
        if e is not None:
            prior_exprs[label] = e

    rows = []
    origin_curves: dict[str, dict[str, np.ndarray]] = {}
    for fb in sorted(curves["file_base"].unique()):
        sub = curves[curves["file_base"] == fb].sort_values("depth_um")
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        keep = N >= N_FLOOR
        z_f, N_f = z[keep], N[keep]
        if z_f.size < 6:
            continue
        feat_row = meta.loc[fb] if fb in meta.index else None
        dg_row = dg.loc[fb] if fb in dg.index else None
        split = split_by_fb.get(str(fb), "not_used")
        cut_depth = (
            float(sub["adaptive_depth_min_um"].iloc[0])
            if "adaptive_depth_min_um" in sub.columns else float(z.min())
        )
        thick = float(feat_row["athena_thick"]) if feat_row is not None else float("nan")
        common = {
            "file_base": fb,
            "split": split,
            "bsg_cut_depth_um": cut_depth,
            "athena_thick_um": thick,
            "fit_depth_min_um": float(z_f.min()),
            "fit_depth_max_um": float(z_f.max()),
            "n_bsg_excluded": int(
                (full_curves[full_curves["file_base"] == fb]["depth_um"] < cut_depth - 1e-12).sum()
            ),
        }

        # plain DG (use cached params)
        if dg_row is not None:
            Np, zp, zf1, zf2 = (float(dg_row["N_p_cm3"]), float(dg_row["z_p_um"]),
                                float(dg_row["z_f1"]), float(dg_row["z_f2"]))
            N_dg = double_gaussian(z_f, Np, zp, zf1, zf2)
            rows.append({**common, "model": "double_gaussian",
                         "r2_log": _r2_log(N_f, N_dg), "rmse_log": _rmse_log(N_f, N_dg),
                         "n_fit": int(z_f.size)})

        # base SR variants (legacy_tail uses depth_offset)
        if feat_row is not None:
            for label, v in BASE_VARIANTS.items():
                if label not in base_exprs:
                    continue
                order = v.feature_order
                X = np.zeros((len(z_f), len(order)), dtype=float)
                for j, c in enumerate(order[:-1]):
                    val = float(feat_row[c])
                    if c in LOG_IN_MODEL:
                        val = np.log(max(val, 1e-30))
                    X[:, j] = val
                X[:, -1] = z_f + v.depth_offset_um
                try:
                    ln_pred = eval_sympy_expr(base_exprs[label], X)
                    ln_pred = np.where(np.isfinite(ln_pred), ln_pred, np.nan)
                    N_pred = np.exp(np.clip(ln_pred, -700, 700))
                except Exception as e:
                    rows.append({**common, "model": label, "error": str(e),
                                 "n_fit": int(z_f.size)})
                    continue
                rows.append({**common, "model": label,
                             "r2_log": _r2_log(N_f, N_pred), "rmse_log": _rmse_log(N_f, N_pred),
                             "n_fit": int(z_f.size)})

        # prior SR variants
        if feat_row is not None and dg_row is not None:
            for label, variant, fm, with_ln_dg, target_kind in PRIOR_VARIANTS:
                if label not in prior_exprs:
                    continue
                X = _prior_feature_vector(feat_row, dg_row, z_f,
                                          feature_mode=fm, with_ln_dg=(variant != "residual"))
                try:
                    out = eval_sympy_expr(prior_exprs[label], X)
                    out = np.where(np.isfinite(out), out, np.nan)
                    if variant == "residual":
                        Np, zp, zf1, zf2 = (float(dg_row["N_p_cm3"]), float(dg_row["z_p_um"]),
                                            float(dg_row["z_f1"]), float(dg_row["z_f2"]))
                        ln_N_dg = np.log(np.where(double_gaussian(z_f, Np, zp, zf1, zf2) > 1e-30,
                                                  double_gaussian(z_f, Np, zp, zf1, zf2), 1e-30))
                        ln_N = ln_N_dg + out
                    else:
                        ln_N = out
                    N_pred = np.exp(np.clip(ln_N, -700, 700))
                except Exception as e:
                    rows.append({**common, "model": label, "error": str(e),
                                 "n_fit": int(z_f.size)})
                    continue
                r2 = _r2_log(N_f, N_pred)
                rows.append({**common, "model": label,
                             "r2_log": r2, "rmse_log": _rmse_log(N_f, N_pred),
                             "n_fit": int(z_f.size)})
                if label == "prior_residual_full":
                    full_sub = full_curves[full_curves["file_base"] == fb].sort_values("depth_um")
                    _plot_bsg_checked_sample(
                        str(fb), full_sub, sub, dg_row, z_f, N_pred, r2, split
                    )
                    Np, zp, zf1, zf2 = (
                        float(dg_row["N_p_cm3"]),
                        float(dg_row["z_p_um"]),
                        float(dg_row["z_f1"]),
                        float(dg_row["z_f2"]),
                    )
                    origin_curves[str(fb)] = {
                        "true_x": z_f.copy(),
                        "true_y": N_f.copy(),
                        "dg_x": z_f.copy(),
                        "dg_y": double_gaussian(z_f, Np, zp, zf1, zf2),
                        "pred_x": z_f.copy(),
                        "pred_y": N_pred.copy(),
                    }

    df = pd.DataFrame(rows)
    save_csv(df, DATA_OUT / "prior_sr_r2.csv")

    ok = df[df["r2_log"].notna()]
    summary = ok.groupby("model")["r2_log"].agg(
        ["count", "median", "mean", "min", "max",
         lambda s: float((s > 0.9).sum()),
         lambda s: float((s > 0.95).sum())]
    ).rename(columns={"<lambda_0>": "n_r2>0.9", "<lambda_1>": "n_r2>0.95"})
    print(summary.to_string())
    summary.to_csv(DATA_OUT / "prior_sr_summary.csv")

    by_split = (
        ok.groupby(["split", "model"])["r2_log"]
        .agg(
            count="count",
            median="median",
            mean="mean",
            min="min",
            max="max",
            n_r2_gt_09=lambda s: int((s > 0.9).sum()),
            n_r2_gt_095=lambda s: int((s > 0.95).sum()),
        )
        .reset_index()
    )
    save_csv(by_split, DATA_OUT / "prior_sr_summary_by_split.csv")
    print("\nTrain/test-separated summary:")
    print(by_split.to_string(index=False))

    # Relative gain vs pure double-Gaussian on the same samples.
    # This is the honest metric: prior_residual ≈ DG means SR adds little.
    dg_r2 = ok[ok["model"] == "double_gaussian"][["file_base", "split", "r2_log"]].rename(
        columns={"r2_log": "r2_dg"}
    )
    delta_rows = []
    for model in sorted(ok["model"].unique()):
        if model == "double_gaussian":
            continue
        m = ok[ok["model"] == model][["file_base", "split", "r2_log"]].merge(
            dg_r2, on=["file_base", "split"], how="inner"
        )
        if m.empty:
            continue
        m["delta_r2"] = m["r2_log"] - m["r2_dg"]
        for split, g in m.groupby("split"):
            delta_rows.append({
                "split": split,
                "model": model,
                "count": int(len(g)),
                "median_r2": float(g["r2_log"].median()),
                "median_r2_dg": float(g["r2_dg"].median()),
                "median_delta_r2": float(g["delta_r2"].median()),
                "mean_delta_r2": float(g["delta_r2"].mean()),
                "frac_better_than_dg": float((g["delta_r2"] > 0).mean()),
            })
    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        save_csv(delta_df, DATA_OUT / "prior_sr_delta_vs_dg_by_split.csv")
        print("\nΔR² vs double_gaussian (positive = SR better than pure DG):")
        print(delta_df.to_string(index=False))

    # Origin export: best 18 simulated profiles by final prior-SR log-RMSE.
    # Only the strictly BSG-excluded, N>=1e18 in-silicon fit window is exported.
    best = (
        ok[ok["model"] == "prior_residual_full"]
        .sort_values(["rmse_log", "file_base"], ascending=[True, True])
        .head(18)
        .reset_index(drop=True)
    )
    wide: dict[str, pd.Series] = {}
    meta_rows = []
    for rank, row in best.iterrows():
        fb = str(row["file_base"])
        curve = origin_curves.get(fb)
        if curve is None:
            continue
        prefix = f"curve{rank + 1:02d}"
        for suffix in ("true_x", "true_y", "dg_x", "dg_y", "pred_x", "pred_y"):
            wide[f"{prefix}_{suffix}"] = pd.Series(curve[suffix])
        meta_rows.append({
            "curve_id": prefix,
            "file_base": fb,
            "split": row["split"],
            "r2_log": row["r2_log"],
            "rmse_log": row["rmse_log"],
            "bsg_cut_depth_um": row["bsg_cut_depth_um"],
            "athena_thick_um": row["athena_thick_um"],
            "fit_depth_min_um": row["fit_depth_min_um"],
            "fit_depth_max_um": row["fit_depth_max_um"],
            "n_fit": row["n_fit"],
        })
    if wide:
        save_csv(pd.DataFrame(wide), DATA_OUT / "origin_sim_best18_curves.csv")
        save_csv(pd.DataFrame(meta_rows), DATA_OUT / "origin_sim_best18_metadata.csv")

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    order = (["double_gaussian"]
             + [l for l in BASE_VARIANTS.keys() if l in ok["model"].unique()]
             + [l for l, _, _, _, _ in PRIOR_VARIANTS if l in ok["model"].unique()])
    data = [ok[ok["model"] == m]["r2_log"].dropna().values for m in order]
    bp = ax.boxplot(data, labels=order, showfliers=True, patch_artist=True)
    cmap = plt.get_cmap("tab10")
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i % 10))
        patch.set_alpha(0.6)
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
    ax.axhline(0.9, color="0.7", ls=":", lw=0.7)
    ax.set_ylabel("R²_log (decades, on N>=1e18 segment)")
    ax.set_title("Prior-injected SR vs base SR vs literature double-Gaussian (sim set)")
    plt.xticks(rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = PLOT_OUT / "prior_sr_r2_box.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Box plot -> {out}")


if __name__ == "__main__":
    main()
