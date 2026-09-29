"""Select top-N samples by PCE, plot overlapping doping/defect curves, export Origin-friendly CSV.

Usage:
    python -m topcon_experiments.plot_pce_ranked_curves
    python -m topcon_experiments.plot_pce_ranked_curves --n 50

Origin import tips (wide CSV):
  - Column `depth_um` is the shared X axis (um).
  - Remaining columns are Y values (one curve per column); column names encode rank + PCE.
  - Use `pce_top50_metadata.csv` to map column names to file_base / IV metrics.
  - For log-scale Y in Origin: set Y axis to Log10 after importing.

Long CSV columns are suitable for Origin's "grouped" line plots:
  depth_um, value, curve_type, rank, file_base, iv_Eff, ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import (
    ATHENA_FEATURES,
    DEFECT_CURVE_DIR,
    DOPING_CURVE_DIR,
    IV_TARGETS,
    OUTPUT_ROOT,
)
from topcon_experiments.exp4_symbolic.curve_utils import process_curve_file
from topcon_experiments.exp4_symbolic.preprocess_curves import _resolve_curve_path

OUT_DIR = OUTPUT_ROOT / "pce_top50_curves"
N_DEFAULT = 50
META_COLS = ["file_base", "iv_Eff", "iv_Voc", "iv_Jsc", "iv_FF"] + ATHENA_FEATURES


def _curve_column_name(rank: int, pce: float, file_base: str) -> str:
    short = file_base[-12:] if len(file_base) > 12 else file_base
    return f"rank{rank:02d}_PCE_{pce:.3f}_{short}"


def _eligible_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Samples with both curve types (use dataset point counts, then verify files exist)."""
    out = df.dropna(subset=["iv_Eff"]).copy()
    if "doping_n_points" in out.columns and "defect_n_points" in out.columns:
        out = out[(out["doping_n_points"] > 0) & (out["defect_n_points"] > 0)]
    out = out.sort_values("iv_Eff", ascending=False).reset_index(drop=True)
    return out


def select_samples(n: int, mode: str) -> pd.DataFrame:
    """Pick n samples ordered by PCE (high -> low).

    mode=span: evenly spaced from max PCE to min PCE (default, best color gradient).
    mode=top:  consecutive top-n highest PCE samples.
    """
    eligible = _eligible_samples(load_raw_dataframe())
    if eligible.empty:
        return eligible

    if mode == "top":
        picked = eligible.head(n)
    elif mode == "span":
        if len(eligible) <= n:
            picked = eligible
        else:
            idx = np.linspace(0, len(eligible) - 1, n, dtype=int)
            picked = eligible.iloc[idx]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Verify curve files exist (drop rare mismatches)
    kept: list[pd.Series] = []
    for _, row in picked.iterrows():
        fb = str(row["file_base"])
        if _resolve_curve_path(DOPING_CURVE_DIR, fb) is None:
            continue
        if _resolve_curve_path(DEFECT_CURVE_DIR, fb) is None:
            continue
        kept.append(row)

    if len(kept) < n:
        log(f"Warning: only {len(kept)} samples retained after curve file check (requested {n})")

    selected = pd.DataFrame(kept).reset_index(drop=True)
    selected.insert(0, "rank", range(1, len(selected) + 1))
    return selected


def load_curve_arrays(file_base: str, curve_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curve_dir = DOPING_CURVE_DIR if curve_type == "doping" else DEFECT_CURVE_DIR
    path = _resolve_curve_path(curve_dir, file_base)
    if path is None:
        raise FileNotFoundError(f"Missing {curve_type} curve for {file_base}")
    grid, raw_grid, fitted_grid = process_curve_file(path)
    return grid, raw_grid, fitted_grid


def build_long_table(selected: pd.DataFrame, curve_type: str, value_col: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        fb = str(row["file_base"])
        grid, raw_grid, fitted_grid = load_curve_arrays(fb, curve_type)
        part = pd.DataFrame({
            "depth_um": grid,
            "value_raw": raw_grid,
            "value_fitted": fitted_grid,
            value_col: fitted_grid,
            "curve_type": curve_type,
            "rank": int(row["rank"]),
            "file_base": fb,
            "iv_Eff": float(row["iv_Eff"]),
            "iv_Voc": float(row["iv_Voc"]),
            "iv_Jsc": float(row["iv_Jsc"]),
            "iv_FF": float(row["iv_FF"]),
        })
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_wide_origin_table(selected: pd.DataFrame, curve_type: str, value_col: str) -> pd.DataFrame:
    depth: np.ndarray | None = None
    columns: dict[str, np.ndarray] = {}

    for _, row in selected.iterrows():
        fb = str(row["file_base"])
        rank = int(row["rank"])
        pce = float(row["iv_Eff"])
        grid, _, fitted_grid = load_curve_arrays(fb, curve_type)
        if depth is None:
            depth = grid
        elif not np.allclose(depth, grid):
            fitted_grid = np.interp(depth, grid, fitted_grid)
        col = _curve_column_name(rank, pce, fb)
        columns[col] = fitted_grid

    assert depth is not None
    wide = pd.DataFrame({"depth_um": depth, **columns})
    return wide


def build_metadata(selected: pd.DataFrame, curve_type: str) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        rank = int(row["rank"])
        pce = float(row["iv_Eff"])
        fb = str(row["file_base"])
        rows.append({
            "curve_type": curve_type,
            "wide_column": _curve_column_name(rank, pce, fb),
            "rank": rank,
            "file_base": fb,
            "iv_Eff": pce,
            "iv_Voc": float(row["iv_Voc"]),
            "iv_Jsc": float(row["iv_Jsc"]),
            "iv_FF": float(row["iv_FF"]),
            **{c: row[c] for c in ATHENA_FEATURES if c in row.index},
        })
    return pd.DataFrame(rows)


def plot_overlapping_curves(
    selected: pd.DataFrame,
    curve_type: str,
    value_col: str,
    y_label: str,
    title: str,
    out_png: Path,
) -> None:
    pce = selected["iv_Eff"].astype(float)
    norm = mcolors.Normalize(vmin=pce.min(), vmax=pce.max())
    cmap = mpl.colormaps["plasma"]

    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in selected.iterrows():
        fb = str(row["file_base"])
        grid, _, fitted = load_curve_arrays(fb, curve_type)
        color = cmap(norm(float(row["iv_Eff"])))
        ax.plot(grid, fitted, color=color, linewidth=0.9, alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("Depth (um)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, which="both")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("PCE / iv_Eff (%)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved plot -> {out_png}")


def main(argv: list[str] | None = None) -> None:
    setup_runtime()
    parser = argparse.ArgumentParser(description="Plot and export PCE-ranked doping/defect curves.")
    parser.add_argument("--n", type=int, default=N_DEFAULT, help="Number of samples (default: 50)")
    parser.add_argument(
        "--mode",
        choices=["span", "top"],
        default="span",
        help="span=evenly spaced max->min PCE; top=highest PCE only (default: span)",
    )
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_samples(args.n, args.mode)
    if selected.empty:
        raise RuntimeError("No samples with both doping and defect curves found.")

    log(
        f"Selected {len(selected)} samples by iv_Eff: "
        f"{selected['iv_Eff'].max():.3f}% (rank 1) -> {selected['iv_Eff'].min():.3f}% (rank {len(selected)})"
    )

    sample_cols = ["rank"] + [c for c in META_COLS if c in selected.columns]
    save_csv(selected[sample_cols], OUT_DIR / "pce_top50_samples.csv")

    doping_long = build_long_table(selected, "doping", "doping_conc_cm3")
    defect_long = build_long_table(selected, "defect", "defect_vac_cm3")
    combined_long = pd.concat([doping_long, defect_long], ignore_index=True)

    doping_wide = build_wide_origin_table(selected, "doping", "doping_conc_cm3")
    defect_wide = build_wide_origin_table(selected, "defect", "defect_vac_cm3")

    meta_doping = build_metadata(selected, "doping")
    meta_defect = build_metadata(selected, "defect")
    meta_all = pd.concat([meta_doping, meta_defect], ignore_index=True)

    save_csv(doping_long, OUT_DIR / "pce_top50_doping_long.csv")
    save_csv(defect_long, OUT_DIR / "pce_top50_defect_long.csv")
    save_csv(combined_long, OUT_DIR / "pce_top50_combined_long.csv")
    save_csv(doping_wide, OUT_DIR / "pce_top50_doping_wide_origin.csv")
    save_csv(defect_wide, OUT_DIR / "pce_top50_defect_wide_origin.csv")
    save_csv(meta_all, OUT_DIR / "pce_top50_metadata.csv")

    plot_overlapping_curves(
        selected,
        "doping",
        "doping_conc_cm3",
        "Doping concentration (cm$^{-3}$)",
        f"Top {len(selected)} doping curves colored by PCE (high→low)",
        OUT_DIR / "pce_top50_doping_overlay.png",
    )
    plot_overlapping_curves(
        selected,
        "defect",
        "defect_vac_cm3",
        "Vacancy defect concentration (cm$^{-3}$)",
        f"Top {len(selected)} defect curves colored by PCE (high→low)",
        OUT_DIR / "pce_top50_defect_overlay.png",
    )

    log(f"All outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
