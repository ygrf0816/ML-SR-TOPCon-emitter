"""Tail curve SR experiment: depth 0.25~2 um, full vs Athena-only features."""

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
    CURVE_TAIL_DEPTH_MIN,
    DEPTH_MAX,
    MODEL2_FEATURES,
    OUTPUT_ROOT,
    RANDOM_STATE,
    SR_TAIL_USE_ALL_POINTS,
)
from topcon_experiments.exp4_symbolic.curve_bsg_trim import apply_tail_depth_window, tail_depth_mask
from topcon_experiments.exp4_symbolic.curve_sr_data import (
    _feature_columns,
    _resolve_feature_row,
    build_sr_dataset,
)
from topcon_experiments.exp4_symbolic.sr_metrics import regression_metrics
from topcon_experiments.exp4_symbolic.sr_split_utils import split_group_train_test

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
EXP_OUT = EXP4_OUT / "curve_tail_experiment"
PLOT_OUT = EXP_OUT / "plots"
TAIL_SUFFIX = "_tail"

SR_JOBS: list[tuple[str, str]] = [
    ("doping", "full"),
    ("doping", "athena"),
    ("defect", "full"),
    ("defect", "athena"),
]


def _warmup_julia() -> None:
    from topcon_experiments.exp4_symbolic.sr_eval_utils import run_pysr_tabular_eval  # noqa: F401


def _predict_from_row(row, X):
    from topcon_experiments.exp4_symbolic.sr_equation_utils import predict_from_row
    return predict_from_row(row, X)


def _best_row(path: Path):
    from topcon_experiments.exp4_symbolic.sr_equation_utils import best_row_by_test_r2
    return best_row_by_test_r2(path)


def _prepare_tail_csvs() -> None:
    for curve_type in ("doping", "defect"):
        src = EXP4_OUT / f"curve_processed_{curve_type}.csv"
        if not src.exists():
            raise FileNotFoundError(f"Run preprocess first: {src}")
        df = pd.read_csv(src)
        tail = apply_tail_depth_window(df, curve_type)
        out = EXP4_OUT / f"curve_processed_{curve_type}{TAIL_SUFFIX}.csv"
        save_csv(tail, out)
        log(
            f"{curve_type} tail window [{CURVE_TAIL_DEPTH_MIN}, {DEPTH_MAX}] um: "
            f"{tail['file_base'].nunique()} samples, {len(tail)} rows -> {out.name}"
        )


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


def _eval_on_tail_window(
    formula_row: pd.Series,
    curve_df: pd.DataFrame,
    meta: pd.DataFrame,
    test_groups: np.ndarray,
    feature_mode: str,
) -> dict:
    y_true: list[float] = []
    y_pred: list[float] = []
    for fb in test_groups:
        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um")
        if sub.empty:
            continue
        depths = sub["depth_um"].values.astype(float)
        mask = tail_depth_mask(depths)
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


def _plot_depth_window(curve_type: str, rng: np.random.Generator) -> None:
    df = pd.read_csv(EXP4_OUT / f"curve_processed_{curve_type}.csv")
    fb = str(rng.choice(df["file_base"].unique()))
    sub = df[df["file_base"] == fb].sort_values("depth_um")
    d = sub["depth_um"].values.astype(float)
    c = sub["value_fitted"].values.astype(float)
    mask = tail_depth_mask(d)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d[~mask], c[~mask], "rx", ms=6, label=f"removed (<{CURVE_TAIL_DEPTH_MIN} um)")
    ax.plot(d[mask], c[mask], "b-", lw=2, label=f"SR region ({CURVE_TAIL_DEPTH_MIN}~{DEPTH_MAX} um)")
    ax.axvline(CURVE_TAIL_DEPTH_MIN, color="gray", ls="--", label=f"cut @ {CURVE_TAIL_DEPTH_MIN} um")
    ax.set_yscale("log")
    ax.set_xlabel("depth (μm)")
    ax.set_ylabel("concentration")
    ax.set_title(f"{curve_type} depth window ({fb[-16:]})")
    ax.legend()
    fig.tight_layout()
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_OUT / f"depth_window_{curve_type}.png", dpi=150)
    plt.close(fig)


def _plot_overlays(
    curve_type: str,
    rows_by_mode: dict[str, pd.Series],
    test_groups: np.ndarray,
    rng: np.random.Generator,
) -> None:
    full_df = pd.read_csv(EXP4_OUT / f"curve_processed_{curve_type}.csv")
    meta = load_raw_dataframe().set_index("file_base")
    picked = list(rng.choice(test_groups, size=min(6, len(test_groups)), replace=False))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = {"baseline": "C0", "full": "C1", "athena": "C2"}

    for ax, fb in zip(axes.flatten(), picked):
        sub = full_df[full_df["file_base"] == fb].sort_values("depth_um")
        if sub.empty:
            ax.set_visible(False)
            continue
        d = sub["depth_um"].values.astype(float)
        c = sub["value_fitted"].values.astype(float)
        mask = tail_depth_mask(d)
        ax.plot(d, c, "k-", lw=2, label="true")
        ax.plot(d[~mask], c[~mask], "rx", ms=3, alpha=0.4)
        for mode, erow in rows_by_mode.items():
            fm = "athena" if mode in ("baseline", "athena") else "full"
            if mode == "baseline":
                fm = "athena"
            X = _build_X_grid(meta, str(fb), d, fm)
            if X is None:
                continue
            pred = np.exp(_predict_from_row(erow, X))
            ax.plot(d[mask], pred[mask], "--", color=colors.get(mode, "C3"), lw=1.3, label=mode)
        ax.axvline(CURVE_TAIL_DEPTH_MIN, color="gray", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_title(str(fb)[-16:])
        ax.legend(fontsize=5)
    for ax in axes.flatten()[len(picked):]:
        ax.set_visible(False)
    fig.suptitle(f"{curve_type}: tail-window SR (test samples)")
    fig.tight_layout()
    fig.savefig(PLOT_OUT / f"overlay_{curve_type}_tail_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=f"Tail curve SR ({CURVE_TAIL_DEPTH_MIN}~{DEPTH_MAX} um): full vs Athena features.",
    )
    parser.add_argument(
        "--skip-pysr",
        action="store_true",
        help="Only prepare tail CSVs and evaluate existing formulas.",
    )
    args = parser.parse_args(argv)

    setup_runtime()
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    log(
        f"Tail curve SR: depth [{CURVE_TAIL_DEPTH_MIN}, {DEPTH_MAX}] um; "
        f"jobs = full params (Athena+descriptors) then Athena-only; "
        f"sampling={'all tail points' if SR_TAIL_USE_ALL_POINTS else 'adaptive subsample'}"
    )

    _prepare_tail_csvs()
    rng = np.random.default_rng(RANDOM_STATE)
    for ct in ("doping", "defect"):
        _plot_depth_window(ct, rng)

    if not args.skip_pysr:
        _warmup_julia()
        from topcon_experiments.exp4_symbolic.sr_eval_utils import run_pysr_curve_eval

    summary_rows: list[dict] = []
    results: dict[tuple[str, str], pd.Series] = {}

    for step, (curve_type, feature_mode) in enumerate(SR_JOBS, start=1):
        formula_path = EXP_OUT / f"sr_formulas_{curve_type}_tail_{feature_mode}.csv"
        feat_label = "full (Athena+descriptors)" if feature_mode == "full" else "athena-only"

        if not args.skip_pysr:
            log_step(step, len(SR_JOBS), f"PySR {curve_type} tail / {feat_label}")
            X, y, groups, weights = build_sr_dataset(
                curve_type, processed_suffix=TAIL_SUFFIX, feature_mode=feature_mode,
            )
            pts_per_sample = len(X) // max(len(np.unique(groups)), 1)
            log(
                f"  dataset: {len(X)} rows, {len(np.unique(groups))} samples, "
                f"~{pts_per_sample} pts/sample"
            )
            if len(X) < 50:
                log(f"Skip {curve_type}/{feature_mode}: only {len(X)} points")
                continue
            n_feat = len(_feature_columns(feature_mode))
            task = f"{curve_type}_tail_{feature_mode}"
            eq_df = run_pysr_curve_eval(
                X, y, groups, task, sample_weights=weights, max_fit_rows=0,
            )
            eq_df["target"] = "ln_curve_tail"
            eq_df["feature_mode"] = feature_mode
            eq_df["n_features"] = n_feat
            sampling_note = "全量深度点（无子采样）" if SR_TAIL_USE_ALL_POINTS else "自适应子采样"
            eq_df["preprocess_note"] = (
                f"尾部实验 depth∈[{CURVE_TAIL_DEPTH_MIN},{DEPTH_MAX}]μm；"
                f"特征={feat_label}（{n_feat}维含depth_um）；y=ln(浓度)；{sampling_note}"
            )
            save_csv(eq_df, formula_path)
            best = _best_row(formula_path)
            log(
                f"  saved {formula_path.name}: test_MSE={best.get('test_MSE', float('nan')):.4g} "
                f"test_R2={best.get('test_R2', float('nan')):.4f}"
            )
            results[(curve_type, feature_mode)] = best
        elif formula_path.exists():
            results[(curve_type, feature_mode)] = _best_row(formula_path)
        else:
            log(f"No formulas at {formula_path}; run without --skip-pysr")

    # Evaluate all variants on tail-window test points
    for curve_type in ("doping", "defect"):
        _, _, groups_ref, _ = build_sr_dataset(
            curve_type, processed_suffix=TAIL_SUFFIX, feature_mode="athena",
        )
        _, test_idx = split_group_train_test(groups_ref)
        test_groups = np.unique(groups_ref[test_idx])
        full_df = pd.read_csv(EXP4_OUT / f"curve_processed_{curve_type}.csv")
        meta = load_raw_dataframe().set_index("file_base")

        overlay_rows: dict[str, pd.Series] = {}
        base_path = EXP4_OUT / f"sr_formulas_{curve_type}.csv"
        if base_path.exists():
            base_best = _best_row(base_path)
            m = _eval_on_tail_window(base_best, full_df, meta, test_groups, "athena")
            summary_rows.append({
                "curve_type": curve_type, "variant": "baseline_fullcurve_athena",
                "feature_mode": "athena", **m,
            })
            overlay_rows["baseline"] = base_best
            log(f"  {curve_type} baseline@tail: R2={m['R2']:.4f}")

        for feature_mode in ("full", "athena"):
            key = (curve_type, feature_mode)
            if key not in results:
                continue
            erow = results[key]
            m = _eval_on_tail_window(erow, full_df, meta, test_groups, feature_mode)
            summary_rows.append({
                "curve_type": curve_type,
                "variant": f"tail_sr_{feature_mode}",
                "feature_mode": feature_mode,
                "test_MSE_native": float(erow.get("test_MSE", np.nan)),
                "test_R2_native": float(erow.get("test_R2", np.nan)),
                **m,
            })
            overlay_rows[feature_mode] = erow
            log(f"  {curve_type} tail_{feature_mode}@tail: R2={m['R2']:.4f}")

            y_true_all: list[float] = []
            y_pred_all: list[float] = []
            for fb in test_groups:
                sub = full_df[full_df["file_base"] == fb].sort_values("depth_um")
                depths = sub["depth_um"].values.astype(float)
                mask = tail_depth_mask(depths)
                sub = sub.iloc[mask]
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
                    f"{curve_type}_tail_{feature_mode}",
                    PLOT_OUT / f"scatter_{curve_type}_tail_{feature_mode}",
                    log_scale=True,
                )

        if overlay_rows:
            _plot_overlays(curve_type, overlay_rows, test_groups, rng)

    if summary_rows:
        save_csv(pd.DataFrame(summary_rows), EXP_OUT / "tail_sr_metrics.csv")
    log(f"Done -> {EXP_OUT}")


if __name__ == "__main__":
    main()
