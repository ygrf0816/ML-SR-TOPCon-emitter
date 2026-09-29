"""Adaptive BSG-cliff curve SR experiment (preserves pre-peak rising segment).

Two variants are produced in one run:
  * adaptive      : per-sample BSG-cliff cut -> [cliff_end, 2 um]
  * adaptive_floor: same cut + keep only N >= 1e18 cm^-3

Both keep absolute physical depth (no zeroing). Trains PySR for doping
full/athena feature modes. Output mirror of run_curve_tail_sr_experiment.py
under outputs/exp4_symbolic/curve_adaptive_experiment/.
"""

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
from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.plot_utils import plot_regression_scatter, save_csv
from topcon_experiments.config import (
    ATHENA_FEATURES,
    DEPTH_MAX,
    MODEL2_FEATURES,
    OUTPUT_ROOT,
    RANDOM_STATE,
    SR_TAIL_USE_ALL_POINTS,
)
from topcon_experiments.exp4_symbolic.curve_bsg_trim import apply_adaptive_trim, apply_keep_bsg
from topcon_experiments.exp4_symbolic.curve_sr_data import (
    _feature_columns,
    _resolve_feature_row,
    build_sr_dataset,
)
from topcon_experiments.exp4_symbolic.sr_metrics import regression_metrics
from topcon_experiments.exp4_symbolic.sr_split_utils import split_group_train_test

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
EXP_OUT = EXP4_OUT / "curve_adaptive_experiment"
PLOT_OUT = EXP_OUT / "plots"

FLOOR_1E18 = 1.0e18

VARIANTS = [
    ("adaptive", None, "_adaptive", "adaptive"),
    ("adaptive_floor1e18", FLOOR_1E18, "_adaptive_floor1e18", "adaptive"),
    ("keep_bsg", None, "_keep_bsg", "keep_bsg"),
    ("keep_bsg_floor1e18", FLOOR_1E18, "_keep_bsg_floor1e18", "keep_bsg"),
]
SR_JOBS = [("doping", "full"), ("doping", "athena")]


def _warmup_julia() -> None:
    from topcon_experiments.exp4_symbolic.sr_eval_utils import run_pysr_tabular_eval  # noqa: F401


def _predict_from_row(row, X):
    from topcon_experiments.exp4_symbolic.sr_equation_utils import predict_from_row
    return predict_from_row(row, X)


def _best_row(path: Path):
    from topcon_experiments.exp4_symbolic.sr_equation_utils import best_row_by_test_r2
    return best_row_by_test_r2(path)


def _prepare_variant_csvs(variant_name: str, floor: float | None, suffix: str, kind: str = "adaptive") -> None:
    """Build curve_processed_<curve_type><suffix>.csv via the chosen trim strategy."""
    src = EXP4_OUT / "curve_processed_doping.csv"
    if not src.exists():
        raise FileNotFoundError(f"Run preprocess first: {src}")
    df = pd.read_csv(src)
    if kind == "keep_bsg":
        trimmed = apply_keep_bsg(df, curve_type="doping", depth_max=DEPTH_MAX, floor=floor)
    else:
        meta = load_raw_dataframe().set_index("file_base")
        trimmed = apply_adaptive_trim(df, meta, curve_type="doping", depth_max=DEPTH_MAX, floor=floor)
    out = EXP4_OUT / f"curve_processed_doping{suffix}.csv"
    save_csv(trimmed, out)
    log(
        f"doping {variant_name}: {trimmed['file_base'].nunique()} samples, "
        f"{len(trimmed)} rows, cut depths "
        f"[{trimmed.groupby('file_base')['adaptive_depth_min_um'].first().min():.3f}, "
        f"{trimmed.groupby('file_base')['adaptive_depth_min_um'].first().max():.3f}] um -> {out.name}"
    )


def _build_X_grid(meta: pd.DataFrame, fb: str, depths: np.ndarray, feature_mode: str) -> np.ndarray | None:
    feat_row = _resolve_feature_row(meta, str(fb), feature_mode)
    if feat_row is None:
        return None
    cols = MODEL2_FEATURES if feature_mode == "full" else ATHENA_FEATURES
    return np.column_stack([np.tile(feat_row[c], len(depths)) for c in cols] + [depths])


def _eval_on_variant_window(
    formula_row: pd.Series,
    curve_df: pd.DataFrame,
    meta: pd.DataFrame,
    test_groups: np.ndarray,
    feature_mode: str,
    *,
    floor: float | None,
) -> dict:
    y_true: list[float] = []
    y_pred: list[float] = []
    for fb in test_groups:
        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um")
        if sub.empty:
            continue
        depths = sub["depth_um"].values.astype(float)
        true_c = sub["value_fitted"].values.astype(float)
        mask = depths <= DEPTH_MAX + 1e-9
        if floor is not None:
            mask &= true_c >= float(floor) - 1e-9
        sub = sub.iloc[mask]
        if sub.empty:
            continue
        depths = sub["depth_um"].values.astype(float)
        true_c = sub["value_fitted"].values.astype(float)
        X = _build_X_grid(meta, str(fb), depths, feature_mode)
        if X is None:
            continue
        pred_ln = _predict_from_row(formula_row, X)
        y_true.extend(np.log(np.maximum(true_c, 1e-30)).tolist())
        y_pred.extend(pred_ln.tolist())
    return regression_metrics(np.array(y_true), np.array(y_pred))


def _plot_cut_distribution(variant_name: str, curve_df: pd.DataFrame) -> None:
    cuts = curve_df.groupby("file_base")["adaptive_depth_min_um"].first()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cuts.values, bins=20, color="C0", alpha=0.7)
    ax.axvline(0.25, color="r", ls="--", label="legacy fixed cut 0.25 um")
    ax.set_xlabel("adaptive BSG-cliff cut depth (um)")
    ax.set_ylabel("sample count")
    ax.set_title(f"{variant_name}: per-sample cut distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_OUT / f"cut_distribution_{variant_name}.png", dpi=150)
    plt.close(fig)


def _plot_depth_window(variant_name: str, rng: np.random.Generator) -> None:
    src = EXP4_OUT / "curve_processed_doping.csv"
    df = pd.read_csv(src)
    fb = str(rng.choice(df["file_base"].unique()))
    sub = df[df["file_base"] == fb].sort_values("depth_um")
    d = sub["depth_um"].values.astype(float)
    c = sub["value_fitted"].values.astype(float)
    cut = float(sub.iloc[0].get("adaptive_depth_min_um", 0.0)) if "adaptive_depth_min_um" in sub else 0.0
    # recompute cut from meta for display
    meta = load_raw_dataframe().set_index("file_base")
    from topcon_experiments.exp4_symbolic.curve_bsg_trim import detect_bsg_cliff_end
    thick = float(meta.loc[fb, "athena_thick"]) if fb in meta.index else 0.1
    ic = detect_bsg_cliff_end(d, c, thick)
    cut = float(d[ic])

    fig, ax = plt.subplots(figsize=(8, 4))
    keep = np.arange(d.size) >= ic
    ax.plot(d[~keep], c[~keep], "rx", ms=6, label=f"removed BSG cliff (<{cut:.3f} um)")
    ax.plot(d[keep], c[keep], "b-", lw=2, label=f"SR region ({cut:.3f}~{DEPTH_MAX} um)")
    ax.axvline(cut, color="gray", ls="--", label=f"adaptive cut @ {cut:.3f} um")
    ax.axvline(0.25, color="orange", ls=":", label="legacy fixed cut 0.25 um")
    ax.set_yscale("log")
    ax.set_xlabel("depth (um)")
    ax.set_ylabel("concentration")
    ax.set_title(f"{variant_name}: adaptive depth window ({fb[-16:]}, thick={thick:.3f})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_OUT / f"depth_window_{variant_name}.png", dpi=150)
    plt.close(fig)


def _plot_overlays(
    variant_name: str,
    rows_by_mode: dict[str, pd.Series],
    test_groups: np.ndarray,
    rng: np.random.Generator,
    *,
    floor: float | None,
) -> None:
    curve_df = pd.read_csv(EXP4_OUT / f"curve_processed_doping{VARIANT_SUFFIX[variant_name]}.csv")
    meta = load_raw_dataframe().set_index("file_base")
    picked = list(rng.choice(test_groups, size=min(6, len(test_groups)), replace=False))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = {"full": "C1", "athena": "C2"}

    for ax, fb in zip(axes.flatten(), picked):
        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um")
        if sub.empty:
            ax.set_visible(False)
            continue
        d = sub["depth_um"].values.astype(float)
        c = sub["value_fitted"].values.astype(float)
        cut = float(sub.iloc[0]["adaptive_depth_min_um"]) if "adaptive_depth_min_um" in sub else 0.0
        ax.plot(d, c, "k-", lw=2, label="true")
        for mode, erow in rows_by_mode.items():
            fm = "athena" if mode == "athena" else "full"
            X = _build_X_grid(meta, str(fb), d, fm)
            if X is None:
                continue
            pred = np.exp(_predict_from_row(erow, X))
            ax.plot(d, pred, "--", color=colors.get(mode, "C3"), lw=1.3, label=mode)
        ax.axvline(cut, color="gray", ls=":", lw=1)
        ax.set_yscale("log")
        if floor is not None:
            ax.axhline(floor, color="0.5", ls="--", lw=0.6)
        ax.set_title(str(fb)[-16:])
        ax.legend(fontsize=5)
    for ax in axes.flatten()[len(picked):]:
        ax.set_visible(False)
    fig.suptitle(f"doping {variant_name}: adaptive SR (test samples)")
    fig.tight_layout()
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_OUT / f"overlay_doping_{variant_name}_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


VARIANT_SUFFIX = {v[0]: v[2] for v in VARIANTS}
VARIANT_FLOOR = {v[0]: v[1] for v in VARIANTS}
VARIANT_KIND = {v[0]: v[3] for v in VARIANTS}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive BSG-cliff curve SR (preserves pre-peak rising segment).",
    )
    parser.add_argument(
        "--skip-pysr",
        action="store_true",
        help="Only prepare adaptive CSVs and evaluate existing formulas.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["adaptive", "adaptive_floor1e18", "keep_bsg", "keep_bsg_floor1e18"],
        help="Subset of variants to run.",
    )
    args = parser.parse_args(argv)

    setup_runtime()
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    log(
        f"Adaptive curve SR: BSG-cliff cut -> [cut, {DEPTH_MAX}] um (absolute physical depth); "
        f"variants={args.variants}; sampling="
        f"{'all points' if SR_TAIL_USE_ALL_POINTS else 'adaptive subsample'}"
    )

    rng = np.random.default_rng(RANDOM_STATE)
    for v in args.variants:
        floor = VARIANT_FLOOR[v]
        suffix = VARIANT_SUFFIX[v]
        _prepare_variant_csvs(v, floor, suffix, VARIANT_KIND[v])
        _plot_depth_window(v, rng)
        curve_df = pd.read_csv(EXP4_OUT / f"curve_processed_doping{suffix}.csv")
        _plot_cut_distribution(v, curve_df)

    if not args.skip_pysr:
        _warmup_julia()
        from topcon_experiments.exp4_symbolic.sr_eval_utils import run_pysr_curve_eval

    summary_rows: list[dict] = []
    results: dict[tuple[str, str, str], pd.Series] = {}

    for v in args.variants:
        floor = VARIANT_FLOOR[v]
        suffix = VARIANT_SUFFIX[v]
        for step, (curve_type, feature_mode) in enumerate(SR_JOBS, start=1):
            formula_path = EXP_OUT / f"sr_formulas_{curve_type}_{v}_{feature_mode}.csv"
            feat_label = "full (Athena+descriptors)" if feature_mode == "full" else "athena-only"
            key = (v, curve_type, feature_mode)

            if not args.skip_pysr:
                log_step(step, len(SR_JOBS), f"PySR {curve_type} {v} / {feat_label}")
                X, y, groups, weights = build_sr_dataset(
                    curve_type, processed_suffix=suffix, feature_mode=feature_mode,
                )
                if len(X) < 50:
                    log(f"Skip {v}/{curve_type}/{feature_mode}: only {len(X)} points")
                    continue
                n_feat = len(_feature_columns(feature_mode))
                task = f"{curve_type}_{v}_{feature_mode}"
                eq_df = run_pysr_curve_eval(
                    X, y, groups, task, sample_weights=weights, max_fit_rows=0,
                )
                eq_df["target"] = "ln_curve_adaptive"
                eq_df["feature_mode"] = feature_mode
                eq_df["n_features"] = n_feat
                floor_note = f" + N>={FLOOR_1E18:.0e}" if floor is not None else ""
                sampling_note = "全量深度点（无子采样）" if SR_TAIL_USE_ALL_POINTS else "自适应子采样"
                eq_df["preprocess_note"] = (
                    f"自适应截断：BSG 陡崖结束位置起至 {DEPTH_MAX}μm（保留峰前上升段）{floor_note}；"
                    f"特征={feat_label}（{n_feat}维含depth_um）；depth 为绝对物理深度；"
                    f"y=ln(浓度)；{sampling_note}"
                )
                save_csv(eq_df, formula_path)
                best = _best_row(formula_path)
                log(
                    f"  saved {formula_path.name}: test_MSE={best.get('test_MSE', float('nan')):.4g} "
                    f"test_R2={best.get('test_R2', float('nan')):.4f}"
                )
                results[key] = best
            elif formula_path.exists():
                results[key] = _best_row(formula_path)
            else:
                log(f"No formulas at {formula_path}; run without --skip-pysr")

        # Evaluate on the variant window
        suffix = VARIANT_SUFFIX[v]
        floor = VARIANT_FLOOR[v]
        curve_df = pd.read_csv(EXP4_OUT / f"curve_processed_doping{suffix}.csv")
        _, _, groups_ref, _ = build_sr_dataset("doping", processed_suffix=suffix, feature_mode="athena")
        _, test_idx = split_group_train_test(groups_ref)
        test_groups = np.unique(groups_ref[test_idx])
        meta = load_raw_dataframe().set_index("file_base")

        overlay_rows: dict[str, pd.Series] = {}
        for feature_mode in ("full", "athena"):
            key = (v, "doping", feature_mode)
            if key not in results:
                continue
            erow = results[key]
            m = _eval_on_variant_window(
                erow, curve_df, meta, test_groups, feature_mode, floor=floor,
            )
            summary_rows.append({
                "variant": v,
                "curve_type": "doping",
                "feature_mode": feature_mode,
                "test_MSE_native": float(erow.get("test_MSE", np.nan)),
                "test_R2_native": float(erow.get("test_R2", np.nan)),
                **m,
            })
            overlay_rows[feature_mode] = erow
            log(f"  doping {v}_{feature_mode}@window: R2={m['R2']:.4f}")

            y_true_all: list[float] = []
            y_pred_all: list[float] = []
            for fb in test_groups:
                sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um")
                if sub.empty:
                    continue
                depths = sub["depth_um"].values.astype(float)
                true_c = sub["value_fitted"].values.astype(float)
                Xg = _build_X_grid(meta, str(fb), depths, feature_mode)
                if Xg is None:
                    continue
                y_true_all.extend(true_c.tolist())
                y_pred_all.extend(np.exp(_predict_from_row(erow, Xg)).tolist())
            if y_true_all:
                plot_regression_scatter(
                    np.array(y_true_all), np.array(y_pred_all),
                    f"doping_{v}_{feature_mode}",
                    PLOT_OUT / f"scatter_doping_{v}_{feature_mode}",
                    log_scale=True,
                )

        if overlay_rows:
            _plot_overlays(v, overlay_rows, test_groups, rng, floor=floor)

    if summary_rows:
        save_csv(pd.DataFrame(summary_rows), EXP_OUT / "adaptive_sr_metrics.csv")
    log(f"Done -> {EXP_OUT}")


if __name__ == "__main__":
    main()
