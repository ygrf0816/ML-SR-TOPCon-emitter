"""Fit the literature asymmetric double-Gaussian to our simulated doping curves.

Validates the literature model (Table 1 of the boron-Gaussian reference) by
fitting its 4 free parameters (N_p, z_p, z_f1, z_f2) to each simulated doping
profile, truncated at N >= N_FLOOR (default 1e18 cm^-3). If the fit R^2 is
good across samples, the literature double-Gaussian is a sound prior for the
downstream SR experiment.

Outputs (under outputs/exp6_experimental/double_gaussian_fit_to_sim/):
  * data/dg_fit_summary.csv        (per-sample 4 params + R^2_log + RMSE)
  * data/dg_fit_all_curves.csv     (long form: file_base, depth_um, N_sim, N_fit)
  * plots/dg_fit_<file_base>.png   (per-sample sim vs fit overlay)
  * plots/dg_fit_r2_histogram.png  (R^2 distribution)
  * plots/dg_fit_overlay_random.png (random subset overlay)
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

CURVE_CSV = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "exp4_symbolic"
    / "curve_processed_doping.csv"
)
EXP_OUT = Path(__file__).resolve().parents[1] / "outputs" / "double_gaussian_fit_to_sim"
DATA_OUT = EXP_OUT / "data"
PLOT_OUT = EXP_OUT / "plots"

N_FLOOR = 1.0e18  # truncate sim curve at this concentration
DEPTH_MAX_UM = 2.0

# Parameter bounds for the DE search.
# N_p  : peak concentration [cm^-3]   ~1e18 .. 1e21
# z_p  : peak depth [um]              0.0 .. 0.5
# z_f1 : left (z<z_p) width [um]      0.005 .. 0.5
# z_f2 : right (z>=z_p) width [um]    0.01 .. 2.0
BOUNDS = [
    (1.0e18, 1.0e21),
    (0.0, 0.5),
    (0.005, 0.5),
    (0.01, 2.0),
]


def double_gaussian(z: np.ndarray, N_p: float, z_p: float, z_f1: float, z_f2: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    left = z < z_p
    right = ~left
    out = np.empty_like(z)
    out[left] = N_p * np.exp(-((z[left] - z_p) / z_f1) ** 2)
    out[right] = N_p * np.exp(-((z[right] - z_p) / z_f2) ** 2)
    return out


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


def fit_one(z: np.ndarray, N: np.ndarray) -> dict:
    """Fit the 4-parameter double-Gaussian on the truncated (N>=N_FLOOR) segment."""
    keep = N >= N_FLOOR
    z_f, N_f = z[keep], N[keep]
    if z_f.size < 6:
        return {"error": "too few points above floor"}

    # Objective in log space (decades) — robust to the wide dynamic range.
    log_obs = np.log10(N_f)

    def obj(p):
        N_p, z_p, z_f1, z_f2 = p
        pred = double_gaussian(z_f, N_p, z_p, z_f1, z_f2)
        if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
            return 1.0e6
        return float(np.sum((log_obs - np.log10(pred)) ** 2))

    # Seed from data: rough peak location & height.
    ipk = int(np.argmax(N_f))
    N_p0 = float(N_f[ipk])
    z_p0 = float(z_f[ipk])
    # Tighter bounds around seed to help DE.
    local_bounds = [
        (max(BOUNDS[0][0], N_p0 * 1e-2), min(BOUNDS[0][1], N_p0 * 1e2)),
        (max(BOUNDS[1][0], z_p0 - 0.15), min(BOUNDS[1][1], z_p0 + 0.15)),
        BOUNDS[2],
        BOUNDS[3],
    ]
    # ensure lower<upper
    local_bounds = [(min(lo, hi), max(lo, hi)) for lo, hi in local_bounds]

    res = differential_evolution(
        obj, local_bounds, seed=1, tol=1e-8, maxiter=400, popsize=25, polish=True, workers=1
    )
    N_p, z_p, z_f1, z_f2 = (float(v) for v in res.x)
    pred = double_gaussian(z_f, N_p, z_p, z_f1, z_f2)
    return {
        "N_p_cm3": N_p,
        "z_p_um": z_p,
        "z_f1": z_f1,
        "z_f2": z_f2,
        "r2_log": _r2_log(N_f, pred),
        "rmse_log_decade": _rmse_log(N_f, pred),
        "n_fit": int(z_f.size),
        "depth_min_um": float(z_f.min()),
        "depth_max_um": float(z_f.max()),
    }


def plot_one(z: np.ndarray, N: np.ndarray, params: dict, fb: str) -> Path:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    keep = N >= N_FLOOR
    # only show the fit-relevant segment (N>=floor); drop below-floor points so
    # the y-axis stays in the [1e18, ~1e22] window where fit quality is visible.
    ax.semilogy(z[keep], N[keep], "ko", ms=2.5, alpha=0.5, label="Simulated (N>=1e18)")
    z_grid = np.linspace(0.0, min(DEPTH_MAX_UM, float(z.max())), 400)
    N_fit = double_gaussian(z_grid, params["N_p_cm3"], params["z_p_um"], params["z_f1"], params["z_f2"])
    # clip the fit line to the visible y-window so divergence below floor does
    # not stretch the axis.
    N_fit_clip = np.where(N_fit >= N_FLOOR, N_fit, np.nan)
    ax.semilogy(z_grid, N_fit_clip, "-", lw=2, color="C3",
                label=f"double-Gaussian (R²_log={params['r2_log']:.3f})")
    ax.axhline(N_FLOOR, color="0.5", ls="--", lw=0.8)
    ax.axvline(params["z_p_um"], color="C3", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("Depth (um)")
    ax.set_ylabel("Boron concentration (cm-3)")
    # force y-axis to the fit window (one decade below floor for headroom)
    y_max = float(np.nanmax(N[keep])) * 3.0
    ax.set_ylim(N_FLOOR / 3.0, y_max)
    ax.set_title(f"{fb}: double-Gaussian fit (N>=1e18)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = PLOT_OUT / f"dg_fit_{fb}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_r2_histogram(summary: pd.DataFrame) -> Path:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    r2 = summary["r2_log"].dropna()
    ax.hist(r2, bins=20, color="C0", alpha=0.8, edgecolor="k")
    ax.axvline(0.0, color="0.5", ls="--", lw=0.8)
    ax.axvline(float(r2.median()), color="C3", ls="-", lw=1.2,
               label=f"median={r2.median():.3f}")
    ax.set_xlabel("R²_log (decades)")
    ax.set_ylabel("# samples")
    ax.set_title(f"Double-Gaussian fit R² over {len(r2)} simulated curves (N>=1e18)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = PLOT_OUT / "dg_fit_r2_histogram.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overlay_random(df: pd.DataFrame, summary: pd.DataFrame, n: int = 8) -> Path:
    apply_plot_style()
    fbs = summary["file_base"].dropna().tolist()
    rng = np.random.default_rng(0)
    pick = rng.choice(fbs, size=min(n, len(fbs)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    cmap = plt.get_cmap("tab10")
    for ax, fb in zip(axes.ravel(), pick):
        sub = df[df["file_base"] == fb].sort_values("depth_um")
        sim = sub["N_sim"].values
        fit = sub["N_fit"].values
        keep = sim >= N_FLOOR
        ax.semilogy(sub["depth_um"].values[keep], sim[keep], "k-", lw=1.4, label="sim")
        fit_clip = np.where(fit >= N_FLOOR, fit, np.nan)
        ax.semilogy(sub["depth_um"].values, fit_clip, "-", color="C3", lw=1.4, label="DG fit")
        ax.axhline(N_FLOOR, color="0.5", ls="--", lw=0.6)
        y_max = float(np.nanmax(sim[keep])) * 3.0 if keep.any() else 1e21
        ax.set_ylim(N_FLOOR / 3.0, y_max)
        row = summary[summary["file_base"] == fb].iloc[0]
        ax.set_title(f"{fb[:18]}\nR²={row['r2_log']:.2f}", fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
    axes.ravel()[0].legend(fontsize=7)
    fig.supxlabel("Depth (um)")
    fig.supylabel("Boron concentration (cm-3)")
    fig.suptitle("Double-Gaussian fit overlay (random 8 samples, N>=1e18)")
    fig.tight_layout()
    out = PLOT_OUT / "dg_fit_overlay_random.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip fitting; replot from cached dg_fit_summary.csv.")
    args = parser.parse_args()
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CURVE_CSV)
    rows = []
    long_rows = []
    if args.replot:
        summary = pd.read_csv(DATA_OUT / "dg_fit_summary.csv")
    else:
        summary = None
    for fb in sorted(df["file_base"].unique()):
        sub = df[df["file_base"] == fb].sort_values("depth_um")
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        if args.replot:
            r = summary[summary["file_base"] == fb]
            if r.empty:
                continue
            res = r.iloc[0].to_dict()
        else:
            res = fit_one(z, N)
            if "error" in res:
                rows.append({"file_base": fb, "error": res["error"]})
                continue
            res["file_base"] = fb
            rows.append(res)
        # long form for overlay
        N_fit_all = double_gaussian(z, res["N_p_cm3"], res["z_p_um"], res["z_f1"], res["z_f2"])
        for zi, Ni, Nfi in zip(z, N, N_fit_all):
            long_rows.append({"file_base": fb, "depth_um": float(zi),
                              "N_sim": float(Ni), "N_fit": float(Nfi)})
        plot_one(z, N, res, fb)

    if args.replot:
        ok = summary[summary["r2_log"].notna()]
    else:
        summary = pd.DataFrame(rows)
        save_csv(summary, DATA_OUT / "dg_fit_summary.csv")
        ok = summary[summary["r2_log"].notna()]
    long_df = pd.DataFrame(long_rows)
    if not args.replot:
        save_csv(long_df, DATA_OUT / "dg_fit_all_curves.csv")

    plot_r2_histogram(summary)
    if not long_df.empty:
        plot_overlay_random(long_df, summary)

    print(f"Fit {len(ok)} / {len(summary)} samples.")
    if not ok.empty:
        print(f"R²_log  median={ok['r2_log'].median():.3f}  "
              f"mean={ok['r2_log'].mean():.3f}  "
              f"min={ok['r2_log'].min():.3f}  max={ok['r2_log'].max():.3f}")
        print(f"RMSE_log (decade) median={ok['rmse_log_decade'].median():.3f}")
        good = (ok["r2_log"] > 0.9).sum()
        print(f"# R²>0.9: {good}/{len(ok)}   # R²>0.95: {(ok['r2_log']>0.95).sum()}")
    print(f"Summary -> {DATA_OUT / 'dg_fit_summary.csv'}")


if __name__ == "__main__":
    main()
