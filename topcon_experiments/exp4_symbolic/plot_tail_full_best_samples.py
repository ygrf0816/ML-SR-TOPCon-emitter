"""Ad-hoc tool: per-sample tail-full SR error, pick 6 best fits, wide CSV + overlay plot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import (
    ATHENA_FEATURES,
    CURVE_TAIL_DEPTH_MIN,
    DEPTH_MAX,
    MODEL2_FEATURES,
    OUTPUT_ROOT,
)
from topcon_experiments.exp4_symbolic.curve_bsg_trim import tail_depth_mask
from topcon_experiments.exp4_symbolic.curve_sr_data import _resolve_feature_row
from topcon_experiments.exp4_symbolic.sr_equation_utils import best_row_by_test_r2, predict_from_row
from topcon_experiments.exp4_symbolic.sr_metrics import regression_metrics

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
EXP_OUT = EXP4_OUT / "curve_tail_experiment"
PLOT_OUT = EXP_OUT / "plots"


def _build_X_grid(
    meta: pd.DataFrame,
    fb: str,
    depths: np.ndarray,
    feature_mode: str,
) -> np.ndarray | None:
    feat_row = _resolve_feature_row(meta, str(fb), feature_mode)
    if feat_row is None:
        return None
    cols = MODEL2_FEATURES if feature_mode == "full" else ATHENA_FEATURES
    return np.column_stack([np.tile(feat_row[c], len(depths)) for c in cols] + [depths])


def _col_name(file_base: str) -> str:
    s = str(file_base).split("_")[-1]
    return s[-20:] if len(s) > 20 else s


def _per_sample_errors(
    curve_type: str,
    formula_row: pd.Series,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    full_df = pd.read_csv(EXP4_OUT / f"curve_processed_{curve_type}.csv")
    rows: list[dict] = []
    for fb in full_df["file_base"].unique():
        sub = full_df[full_df["file_base"] == fb].sort_values("depth_um")
        depths = sub["depth_um"].values.astype(float)
        mask = tail_depth_mask(depths)
        sub = sub.iloc[mask]
        if len(sub) < 2:
            continue
        depths = sub["depth_um"].values.astype(float)
        true_c = sub["value_fitted"].values.astype(float)
        X = _build_X_grid(meta, str(fb), depths, "full")
        if X is None:
            continue
        pred_ln = predict_from_row(formula_row, X)
        true_ln = np.log(np.maximum(true_c, 1e-30))
        m = regression_metrics(true_ln, pred_ln)
        rows.append({
            "file_base": str(fb),
            "n_pts": len(sub),
            "MSE_ln": m["MSE"],
            "MAE_ln": m["MAE"],
            "R2_ln": m["R2"],
            "ln_var": float(np.var(true_ln)),
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["R2_ln", "MSE_ln"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _curve_payload(
    fb: str,
    formula_row: pd.Series,
    meta: pd.DataFrame,
    full_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    sub = full_df[full_df["file_base"] == fb].sort_values("depth_um")
    depths = sub["depth_um"].values.astype(float)
    mask = tail_depth_mask(depths)
    sub = sub.iloc[mask]
    depths = sub["depth_um"].values.astype(float)
    true_c = sub["value_fitted"].values.astype(float)
    X = _build_X_grid(meta, fb, depths, "full")
    if X is None:
        raise ValueError(f"cannot build features for {fb}")
    pred_c = np.exp(predict_from_row(formula_row, X))
    return {"depth_um": depths, "conc_true": true_c, "conc_pred": pred_c}


def _save_wide_csv(picked: list[str], payloads: dict[str, dict], out_path: Path) -> None:
    ref_depths = payloads[picked[0]]["depth_um"]
    wide: dict[str, np.ndarray] = {"depth_um": ref_depths}
    for fb in picked:
        col = _col_name(fb)
        data = payloads[fb]
        d = data["depth_um"]
        if len(d) != len(ref_depths) or not np.allclose(d, ref_depths):
            log(f"Warning: {fb} depth grid differs from reference; aligning by interpolation")
            true_i = np.interp(ref_depths, d, data["conc_true"])
            pred_i = np.interp(ref_depths, d, data["conc_pred"])
            wide[f"{col}_true"] = true_i
            wide[f"{col}_pred"] = pred_i
        else:
            wide[f"{col}_true"] = data["conc_true"]
            wide[f"{col}_pred"] = data["conc_pred"]
    save_csv(pd.DataFrame(wide), out_path)


def _plot_overlay(
    curve_type: str,
    picked: list[str],
    payloads: dict[str, dict],
    err_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    err_map = err_df.set_index("file_base")
    for ax, fb in zip(axes.flatten(), picked):
        data = payloads[fb]
        mse = float(err_map.loc[fb, "MSE_ln"])
        r2 = float(err_map.loc[fb, "R2_ln"])
        ax.plot(data["depth_um"], data["conc_true"], "k-", lw=2, label="true")
        ax.plot(data["depth_um"], data["conc_pred"], "C1--", lw=1.5, label="tail_full SR")
        ax.axvline(CURVE_TAIL_DEPTH_MIN, color="gray", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("depth (μm)")
        ax.set_ylabel("concentration")
        ax.set_title(f"{_col_name(fb)}\nMSE(ln)={mse:.3g} R²={r2:.3f}")
        ax.legend(fontsize=7)
    for ax in axes.flatten()[len(picked):]:
        ax.set_visible(False)
    fig.suptitle(
        f"{curve_type}: tail_full SR — {len(picked)} best-fit samples by R²(ln) "
        f"(depth {CURVE_TAIL_DEPTH_MIN}~{DEPTH_MAX} μm)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_one(curve_type: str, n_pick: int, min_r2: float) -> None:
    formula_path = EXP_OUT / f"sr_formulas_{curve_type}_tail_full.csv"
    if not formula_path.exists():
        log(f"Skip {curve_type}: missing {formula_path}")
        return

    formula_row = best_row_by_test_r2(formula_path)
    meta = load_raw_dataframe().set_index("file_base")
    full_df = pd.read_csv(EXP4_OUT / f"curve_processed_{curve_type}.csv")

    err_df = _per_sample_errors(curve_type, formula_row, meta)
    eligible = err_df[err_df["R2_ln"] >= min_r2]
    if len(eligible) < n_pick:
        log(
            f"{curve_type}: only {len(eligible)} samples with R²(ln)>={min_r2}; "
            f"using all eligible (requested {n_pick})"
        )
    pick_df = eligible.head(n_pick) if len(eligible) else err_df.head(n_pick)
    out_err = EXP_OUT / f"per_sample_error_{curve_type}_tail_full.csv"
    save_csv(err_df, out_err)
    log(f"{curve_type}: ranked {len(err_df)} samples by R²(ln) -> {out_err.name}")

    picked = pick_df["file_base"].tolist()
    payloads = {fb: _curve_payload(fb, formula_row, meta, full_df) for fb in picked}

    out_csv = PLOT_OUT / f"tail_full_best6_{curve_type}_wide.csv"
    out_png = PLOT_OUT / f"tail_full_best6_{curve_type}.png"
    _save_wide_csv(picked, payloads, out_csv)
    _plot_overlay(curve_type, picked, payloads, err_df, out_png)
    log(f"{curve_type}: best-{n_pick} overlay -> {out_png}")
    log(f"  equation: {str(formula_row.get('equation', ''))[:100]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot 6 best tail_full SR curve fits per type.")
    parser.add_argument("--n", type=int, default=6, help="Number of best-fit samples to plot")
    parser.add_argument(
        "--min-r2",
        type=float,
        default=0.0,
        help="Only pick samples with per-sample R²(ln) >= this threshold",
    )
    parser.add_argument(
        "--curve-type",
        choices=("doping", "defect", "both"),
        default="both",
    )
    args = parser.parse_args(argv)

    setup_runtime()
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    types = ["doping", "defect"] if args.curve_type == "both" else [args.curve_type]
    for ct in types:
        run_one(ct, args.n, args.min_r2)
    log(f"Done -> {PLOT_OUT}")


if __name__ == "__main__":
    main()
