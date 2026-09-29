"""Compare the SR curve-formula variants against literature profiles.

Reads curve_shape_fit_summary.csv from each variant subdirectory under
outputs/literature_benchmark/{adaptive,adaptive_floor1e18,keep_bsg,
keep_bsg_floor1e18,legacy_tail}/
and produces:
  * curve_formula_comparison.csv  (profile_id x variant x R2_log/RMSE/n_fit)
  * curve_formula_comparison_D{1-10}.png  (one literature curve with all
    SR fits overlaid, reconstructed from each variant's stored params)
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

from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.exp6_experimental.literature_benchmark.fit_sr_to_literature import (
    VARIANTS,
    _best_expr,
    _load_reference_curve,
    _predict_ln_curve,
)

BASE_OUT = Path(__file__).resolve().parents[1] / "outputs" / "literature_benchmark"
COMPARE_DIR = BASE_OUT / "curve_formula_comparison"


def _load_summary(variant_label: str) -> pd.DataFrame | None:
    p = BASE_OUT / variant_label / "data" / "curve_shape_fit_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    df["variant"] = variant_label
    return df


def _parse_dict_field(val) -> dict[str, float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return {}
    if isinstance(val, dict):
        return {k: float(v) for k, v in val.items()}
    try:
        import ast

        d = ast.literal_eval(str(val))
        return {k: float(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except Exception:
        return {}


def _variant_curve(variant_label: str, profile_id: str, z_plot: np.ndarray) -> np.ndarray | None:
    """Reconstruct the SR-predicted N(z) for one variant/profile."""
    variant = VARIANTS[variant_label]
    summary = BASE_OUT / variant_label / "data" / "curve_shape_fit_summary.csv"
    if not summary.exists():
        return None
    df = pd.read_csv(summary)
    rows = df[df["profile_id"] == profile_id]
    # Prefer the row with a finite r2; skip pure-error duplicate rows.
    rows = rows[rows["curve_r2_log"].notna()] if "curve_r2_log" in rows.columns else rows
    if rows.empty:
        return None
    r = rows.iloc[0]
    if isinstance(r.get("error"), str):
        return None
    fixed = _parse_dict_field(r.get("fixed_params"))
    opt = _parse_dict_field(r.get("optimized_params"))
    vals = {**fixed, **opt}
    if not vals:
        return None
    expr = _best_expr(variant.formula_path)
    z_sr = z_plot + variant.depth_offset_um
    ln_pred = _predict_ln_curve(expr, vals, z_sr, feature_order=variant.feature_order)
    ln_pred = np.where(np.isfinite(ln_pred), ln_pred, np.nan)
    return np.exp(np.clip(ln_pred, -700, 700))


def main() -> None:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    (COMPARE_DIR / "plots").mkdir(exist_ok=True)

    summaries = []
    for label in VARIANTS.keys():
        s = _load_summary(label)
        if s is not None:
            summaries.append(s)
    if not summaries:
        print("No variant summaries found. Run fit_sr_to_literature for each variant first.")
        return
    all_df = pd.concat(summaries, ignore_index=True)

    cols = ["profile_id", "variant", "curve_r2_log", "curve_rmse_cm3", "n_depth_fit"]
    cols = [c for c in cols if c in all_df.columns]
    pivot = all_df[cols].copy()
    # Keep one row per (profile_id, variant): prefer finite r2, then first.
    pivot = pivot.sort_values(["profile_id", "variant", "curve_r2_log"], na_position="last")
    pivot = pivot.drop_duplicates(subset=["profile_id", "variant"], keep="first").sort_values(["profile_id", "variant"])
    save_csv(pivot, COMPARE_DIR / "curve_formula_comparison.csv")
    print(f"Saved summary table -> {COMPARE_DIR / 'curve_formula_comparison.csv'}")
    print(pivot.to_string(index=False))

    # Per-profile overlay plots
    profile_ids = sorted(all_df["profile_id"].unique())
    apply_plot_style()
    for pid in profile_ids:
        z_ref, N_ref = _load_reference_curve(pid)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(z_ref, N_ref, "ko", ms=3, label="Literature (double Gaussian)", alpha=0.7)
        colors = {
            "adaptive": "C0",
            "adaptive_floor1e18": "C1",
            "legacy_tail": "C2",
            "keep_bsg": "C3",
            "keep_bsg_floor1e18": "C4",
        }
        z_max = 0.0
        plotted = 0
        for label in VARIANTS.keys():
            variant = VARIANTS[label]
            z_max_v = min(variant.fit_depth_max_um, float(z_ref.max()))
            z_plot = np.linspace(0.0, z_max_v, 300)
            N_pred = _variant_curve(label, pid, z_plot)
            if N_pred is None or not np.any(np.isfinite(N_pred)):
                continue
            N_pred = np.where(np.isfinite(N_pred) & (N_pred > 0), N_pred, np.nan)
            z_max = max(z_max, z_max_v)
            r2_row = all_df[(all_df["profile_id"] == pid) & (all_df["variant"] == label)]
            r2_row = r2_row[r2_row["curve_r2_log"].notna()]
            r2 = float(r2_row["curve_r2_log"].iloc[0]) if not r2_row.empty else float("nan")
            ax.semilogy(z_plot, N_pred, "-", lw=1.8, color=colors.get(label, "C5"),
                        label=f"{label} (R²={r2:.2f})")
            ax.axhline(variant.n_floor, color=colors.get(label, "C5"), ls=":", lw=0.6, alpha=0.5)
            plotted += 1
        if plotted == 0:
            plt.close(fig)
            continue
        ax.set_xlim(0.0, max(z_max, 0.5))
        ax.set_xlabel("Depth (um, literature coords)")
        ax.set_ylabel("Boron concentration (cm-3)")
        ax.set_title(f"{pid}: SR curve-formula variant comparison")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        out = COMPARE_DIR / "plots" / f"curve_formula_comparison_{pid}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out.name}")


if __name__ == "__main__":
    main()
