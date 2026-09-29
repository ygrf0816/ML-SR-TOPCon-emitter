"""Q: should the DG-parameter fit use the same points as SR residual training?

SR residual was trained on ALL N>0 points inside the BSG-trimmed window, while
the 4 DG params are fit only on N >= 1e18. This script quantifies the impact of
that mismatch by fitting the double-Gaussian per sample under both point sets:

  * floor1e18 : fit on N >= 1e18 (current strategy)
  * allpoints : fit on all N > 0 inside the trimmed window (down to ~1e15)

and cross-evaluating each fit on BOTH windows (R^2_log on N>=1e18 segment and
on the full window). Run on a random subsample of the ext2000 trimmed curves.

Output: outputs/exp4_symbolic/process_to_dg/data/dg_fit_window_compare.csv
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import OUTPUT_ROOT, RANDOM_STATE
from topcon_experiments.exp6_experimental.literature_benchmark.fit_double_gaussian_to_sim import (
    BOUNDS,
    N_FLOOR,
    double_gaussian,
)

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
TRIM_CSV = EXP4_OUT / "curve_processed_doping_ext2000_adaptive.csv"
OUT_CSV = EXP4_OUT / "process_to_dg" / "data" / "dg_fit_window_compare.csv"

N_SAMPLES = 200


def _fit_dg(z: np.ndarray, N: np.ndarray) -> tuple[float, float, float, float] | None:
    """4-param DG fit in log10 space on the given points (no internal floor)."""
    if z.size < 6:
        return None
    log_obs = np.log10(N)

    def obj(p):
        pred = double_gaussian(z, *p)
        if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
            return 1.0e6
        return float(np.sum((log_obs - np.log10(pred)) ** 2))

    ipk = int(np.argmax(N))
    N_p0, z_p0 = float(N[ipk]), float(z[ipk])
    local_bounds = [
        (max(BOUNDS[0][0], N_p0 * 1e-2), min(BOUNDS[0][1], N_p0 * 1e2)),
        (max(BOUNDS[1][0], z_p0 - 0.15), min(BOUNDS[1][1], z_p0 + 0.15)),
        BOUNDS[2],
        BOUNDS[3],
    ]
    local_bounds = [(min(lo, hi), max(lo, hi)) for lo, hi in local_bounds]
    res = differential_evolution(
        obj, local_bounds, seed=1, tol=1e-8, maxiter=400, popsize=25, polish=True, workers=1
    )
    return tuple(float(v) for v in res.x)


def _r2_log(N_obs: np.ndarray, N_pred: np.ndarray) -> float:
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() < 4:
        return float("nan")
    y, yhat = np.log10(N_obs[mask]), np.log10(N_pred[mask])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot <= 0 else 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot


def _one(args: tuple[str, np.ndarray, np.ndarray]) -> dict | None:
    fb, z, N = args
    hi = N >= N_FLOOR
    if hi.sum() < 6:
        return None
    fits = {
        "floor1e18": _fit_dg(z[hi], N[hi]),
        "allpoints": _fit_dg(z, N),
    }
    row: dict = {"file_base": fb, "n_hi": int(hi.sum()), "n_all": int(z.size)}
    for name, p in fits.items():
        if p is None:
            return None
        pred = double_gaussian(z, *p)
        row[f"{name}_r2_hi"] = _r2_log(N[hi], pred[hi])
        row[f"{name}_r2_all"] = _r2_log(N, pred)
        row[f"{name}_z_p"] = p[1]
        row[f"{name}_z_f2"] = p[3]
    return row


def main() -> None:
    setup_runtime()
    curves = pd.read_csv(TRIM_CSV)
    rng = np.random.default_rng(RANDOM_STATE)
    fbs = curves["file_base"].unique()
    pick = rng.choice(fbs, size=min(N_SAMPLES * 2, len(fbs)), replace=False)

    tasks = []
    for fb in pick:
        sub = curves[curves["file_base"] == fb].sort_values("depth_um")
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        if (N >= N_FLOOR).sum() >= 6:
            tasks.append((str(fb), z, N))
        if len(tasks) >= N_SAMPLES:
            break
    log(f"comparing DG fit windows on {len(tasks)} samples")

    rows = []
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 16)) as pool:
        for res in pool.map(_one, tasks, chunksize=4):
            if res is not None:
                rows.append(res)
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, OUT_CSV)

    print(f"n = {len(df)}")
    for name in ("floor1e18", "allpoints"):
        print(
            f"{name:>10}: R2_log(hi window) median={df[f'{name}_r2_hi'].median():.4f} "
            f"p10={df[f'{name}_r2_hi'].quantile(0.1):.4f} | "
            f"R2_log(full window) median={df[f'{name}_r2_all'].median():.4f} "
            f"p10={df[f'{name}_r2_all'].quantile(0.1):.4f}"
        )
    dzp = (df["allpoints_z_p"] - df["floor1e18_z_p"]).abs()
    print(f"|z_p diff| median={dzp.median():.4f} um, p90={dzp.quantile(0.9):.4f} um")


if __name__ == "__main__":
    main()
