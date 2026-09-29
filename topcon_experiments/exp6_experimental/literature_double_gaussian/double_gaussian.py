"""Literature boron profiles (Table 1) via asymmetric double-Gaussian model."""

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

from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv

MODULE_DIR = Path(__file__).resolve().parent
EXP6_OUT = Path(__file__).resolve().parents[1] / "outputs" / "literature_double_gaussian"
PARAMS_CSV = MODULE_DIR / "params_table1.csv"

# Background / truncation floor (cm^-3)
N_FLOOR = 4.0e18
DEPTH_MIN = 0.0
DEPTH_MAX_UM = 2.0
N_DEPTH_POINTS = 401  # 0~2 µm, step 0.005 µm


def double_gaussian(
    z_um: np.ndarray,
    N_p: float,
    z_p: float,
    z_f1: float,
    z_f2: float,
) -> np.ndarray:
    """Asymmetric double Gaussian (piecewise at z_p).

    N(z) = N_p * exp(-((z - z_p) / z_f1)^2)   for z < z_p
         = N_p * exp(-((z - z_p) / z_f2)^2)   for z >= z_p
    """
    z_um = np.asarray(z_um, dtype=float)
    left = z_um < z_p
    right = ~left
    out = np.empty_like(z_um)
    out[left] = N_p * np.exp(-((z_um[left] - z_p) / z_f1) ** 2)
    out[right] = N_p * np.exp(-((z_um[right] - z_p) / z_f2) ** 2)
    return out


def truncate_floor(z_um: np.ndarray, N_cm3: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep only points with N >= floor (junction-style truncation)."""
    mask = N_cm3 >= floor
    return z_um[mask], N_cm3[mask]


def load_params() -> pd.DataFrame:
    return pd.read_csv(PARAMS_CSV)


def compute_one_profile(row: pd.Series, z_grid: np.ndarray) -> pd.DataFrame:
    N_raw = double_gaussian(
        z_grid,
        float(row["N_p_cm3"]),
        float(row["z_p_um"]),
        float(row["z_f1"]),
        float(row["z_f2"]),
    )
    z_trunc, N_trunc = truncate_floor(z_grid, N_raw, N_FLOOR)
    return pd.DataFrame({
        "profile_id": row["profile_id"],
        "depth_um": z_trunc,
        "N_cm3": N_trunc,
        "R_sheet_ohm_sq": float(row["R_sheet_ohm_sq"]),
        "N_p_cm3": float(row["N_p_cm3"]),
        "z_p_um": float(row["z_p_um"]),
        "z_f1": float(row["z_f1"]),
        "z_f2": float(row["z_f2"]),
        "ref": row["ref"],
        "N_floor_cm3": N_FLOOR,
    })


def compute_all_profiles() -> tuple[pd.DataFrame, list[Path]]:
    z_grid = np.linspace(DEPTH_MIN, DEPTH_MAX_UM, N_DEPTH_POINTS)
    params = load_params()
    data_dir = EXP6_OUT / "data"
    curves_dir = data_dir / "curves"
    data_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    written: list[Path] = []
    for _, row in params.iterrows():
        pid = str(row["profile_id"])
        curve_df = compute_one_profile(row, z_grid)
        frames.append(curve_df)

        csv_path = curves_dir / f"{pid}_doping_curve.csv"
        save_csv(curve_df[["depth_um", "N_cm3"]], csv_path)
        written.append(csv_path)

        txt_path = curves_dir / f"{pid}_doping_curve.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for d, n in zip(curve_df["depth_um"], curve_df["N_cm3"]):
                f.write(f"{d:.6f}\t{n:.6e}\n")
        written.append(txt_path)

    long_df = pd.concat(frames, ignore_index=True)
    save_csv(long_df, data_dir / "all_profiles_long.csv")
    save_csv(params, data_dir / "fitting_parameters_table1.csv")
    written.extend([data_dir / "all_profiles_long.csv", data_dir / "fitting_parameters_table1.csv"])
    return long_df, written


def plot_overlay(long_df: pd.DataFrame, out_png: Path) -> Path:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")
    ids = sorted(long_df["profile_id"].unique())
    for i, pid in enumerate(ids):
        sub = long_df[long_df["profile_id"] == pid].sort_values("depth_um")
        rs = sub["R_sheet_ohm_sq"].iloc[0]
        ax.semilogy(
            sub["depth_um"],
            sub["N_cm3"],
            lw=1.8,
            color=cmap(i % 10),
            label=f"{pid} (Rsh={rs:.0f} ohm/sq)",
        )
    ax.set_xlabel("Depth (um)")
    ax.set_ylabel("Boron concentration (cm-3)")
    ax.set_title(
        "Literature boron profiles (double Gaussian, Table 1)\n"
        f"truncated at N >= {N_FLOOR:.0e} cm-3, depth 0-{DEPTH_MAX_UM} um"
    )
    ax.set_xlim(DEPTH_MIN, DEPTH_MAX_UM)
    ax.axhline(N_FLOOR, color="0.5", ls="--", lw=0.8, label=f"floor = {N_FLOOR:.0e}")
    ax.legend(fontsize=7, loc="upper right", frameon=False, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def run() -> list[Path]:
    long_df, paths = compute_all_profiles()
    plot_dir = EXP6_OUT / "plots"
    png = plot_overlay(long_df, plot_dir / "literature_double_gaussian_overlay.png")
    paths.append(png)
    return paths


def main() -> None:
    paths = run()
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
