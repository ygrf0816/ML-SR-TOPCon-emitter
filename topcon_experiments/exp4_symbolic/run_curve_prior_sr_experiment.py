"""SR curve experiment with the literature double-Gaussian injected as a prior.

Three prior-injection strategies, all trained on the **adaptive BSG-trimmed**
window (in-silicon profile only; absolute physical depth kept; cut placed at
the post-cliff valley — see ``curve_bsg_trim.detect_bsg_cliff_end``). DG
parameters are fit on ``N >= 1e18``; SR residual training historically used
all ``N > 0`` points inside that window.

  * residual      : y = ln(N) - ln(N_DG(z)) ; SR learns the multiplicative
                    correction exp(SR). Final N(z) = N_DG(z) * exp(SR(feat,z)).
                    DG params (N_p, z_p, z_f1, z_f2) are appended as features
                    so SR can see the baseline shape.

  * feature_aug   : y = ln(N) (original target). Features extended with the
                    4 DG params AND ln(N_DG(z)) at the given depth. SR is free
                    to learn "DG + correction" or to invent its own form.

  * struct_bias   : same as feature_aug, but a custom Julia operator
                    gauss_kernel(z, z_p, sigma) = exp(-((z-z_p)/sigma)^2) is
                    registered so PySR can directly assemble a Gaussian peak.

Outputs (under outputs/exp4_symbolic/curve_prior_experiment/):
  * sr_formulas_doping_<variant>_full.csv  (best PySR equations)
  * sr_formulas_doping_<variant>_athena.csv
  * data/dg_params.csv                     (per-sample DG fit used as prior)
  * plots/prior_dg_baseline_samples.png    (DG baseline overlay)
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
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import (
    ATHENA_FEATURES,
    DEPTH_MAX,
    LOG_FEATURES,
    MODEL2_FEATURES,
    OUTPUT_ROOT,
    RANDOM_STATE,
    SR_TAIL_USE_ALL_POINTS,
)
from topcon_experiments.exp4_symbolic.curve_bsg_trim import apply_adaptive_trim
from topcon_experiments.exp4_symbolic.curve_sr_data import _resolve_feature_row
from topcon_experiments.exp4_symbolic.sr_split_utils import split_group_train_test
from topcon_experiments.exp6_experimental.literature_benchmark.fit_double_gaussian_to_sim import (
    N_FLOOR,
    double_gaussian,
    fit_one,
)

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
EXP_OUT = EXP4_OUT / "curve_prior_experiment"
PLOT_OUT = EXP_OUT / "plots"
DATA_OUT = EXP_OUT / "data"

FLOOR = N_FLOOR  # 1e18 — the floor used for DG fitting and SR evaluation
# Use the BSG-removed (adaptive trim) window: the double-Gaussian describes
# the in-silicon diffusion profile only, NOT the BSG glass layer. Fitting DG
# on keep_bsg data forced z_p≈0 degenerate solutions on thick-BSG samples.
SUFFIX = "_adaptive"

# DG feature names appended to the SR input
DG_FEAT_NAMES = ["dg_N_p", "dg_z_p", "dg_z_f1", "dg_z_f2"]
DG_LOG_FEAT = {"dg_N_p"}  # log-transform these (huge dynamic range)

VARIANTS = ["residual", "feature_aug", "struct_bias"]
SR_JOBS = [("doping", "full"), ("doping", "athena")]


def _prepare_window_csv() -> None:
    """Ensure the BSG-removed (adaptive trim) curve CSV exists.

    Uses ``apply_adaptive_trim`` (per-sample cliff_end -> 2 um, absolute
    physical depth, NO concentration floor) so the double-Gaussian is fit on
    the in-silicon diffusion profile only. The 1e18 floor is applied inside
    ``fit_one`` during DG fitting, not here, so low-dose samples keep all
    their points and are simply skipped by ``_fit_dg_for_all_samples``.
    """
    src = EXP4_OUT / "curve_processed_doping.csv"
    out = EXP4_OUT / f"curve_processed_doping{SUFFIX}.csv"
    if out.exists():
        return
    if not src.exists():
        raise FileNotFoundError(f"Run preprocess first: {src}")
    df = pd.read_csv(src)
    meta = load_raw_dataframe().set_index("file_base")
    trimmed = apply_adaptive_trim(df, meta, curve_type="doping", depth_max=DEPTH_MAX, floor=None)
    save_csv(trimmed, out)
    log(f"prepared {out.name}: {trimmed['file_base'].nunique()} samples, {len(trimmed)} rows")


def _fit_dg_for_all_samples(curve_df: pd.DataFrame) -> pd.DataFrame:
    """Fit the 4-param double-Gaussian per sample on the N>=FLOOR segment.

    Samples whose max concentration is below FLOOR (low-dose / shallow-junction
    processes) are skipped — the double-Gaussian is meaningless for them.
    """
    cache = DATA_OUT / "dg_params.csv"
    if cache.exists():
        return pd.read_csv(cache).set_index("file_base")
    rows = []
    for fb in curve_df["file_base"].unique():
        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um")
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        res = fit_one(z, N)
        if "error" in res:
            continue
        res["file_base"] = fb
        rows.append(res)
    df = pd.DataFrame(rows).set_index("file_base")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    save_csv(df.reset_index(), cache)
    log(f"DG fits cached -> {cache.name} ({len(df)} samples)")
    return df


def _plot_dg_baseline(curve_df: pd.DataFrame, dg: pd.DataFrame, n: int = 8) -> None:
    """Sanity overlay of sim curve + DG baseline for a random subset."""
    rng = np.random.default_rng(RANDOM_STATE)
    fbs = rng.choice(dg.index.tolist(), size=min(n, len(dg)), replace=False)
    apply_plot_style = None
    from topcon_experiments.common.mpl_style import apply_plot_style  # noqa: F401
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for ax, fb in zip(axes.ravel(), fbs):
        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um")
        sim = sub["value_fitted"].values
        keep = sim >= FLOOR
        ax.semilogy(sub["depth_um"].values[keep], sim[keep], "k-", lw=1.3, label="sim")
        p = dg.loc[fb]
        z_grid = np.linspace(0.0, DEPTH_MAX, 300)
        N_dg = double_gaussian(z_grid, p["N_p_cm3"], p["z_p_um"], p["z_f1"], p["z_f2"])
        N_dg_clip = np.where(N_dg >= FLOOR, N_dg, np.nan)
        ax.semilogy(z_grid, N_dg_clip, "-", color="C3", lw=1.3, label="DG prior")
        ax.axhline(FLOOR, color="0.5", ls="--", lw=0.6)
        y_max = float(np.nanmax(sim[keep])) * 3.0 if keep.any() else 1e21
        ax.set_ylim(FLOOR / 3.0, y_max)
        ax.set_title(f"{fb[:18]}\nR²={p['r2_log']:.2f}", fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Double-Gaussian prior baseline (random 8 samples, N>=1e18)")
    fig.tight_layout()
    out = PLOT_OUT / "prior_dg_baseline_samples.png"
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved {out.name}")


def _build_prior_dataset(
    curve_df: pd.DataFrame,
    dg: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    feature_mode: str,
    variant: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build the SR training dataset for one prior variant.

    Returns X, y, groups, sample_weights.
    """
    feat_cols = MODEL2_FEATURES if feature_mode == "full" else ATHENA_FEATURES
    base_names = list(feat_cols) + ["depth_um"]
    if variant == "residual" or variant == "feature_aug" or variant == "struct_bias":
        feat_names = base_names + DG_FEAT_NAMES
        if variant in ("feature_aug", "struct_bias"):
            feat_names = feat_names + ["ln_N_dg"]
    else:
        raise ValueError(f"unknown variant {variant}")

    log_in_model = set(LOG_FEATURES) | DG_LOG_FEAT

    X_rows, y_vals, groups, weights = [], [], [], []
    for fb in curve_df["file_base"].unique():
        if fb not in dg.index:
            continue
        feat_row = _resolve_feature_row(meta, str(fb), feature_mode)
        if feat_row is None:
            continue
        p = dg.loc[fb]
        N_p, z_p, z_f1, z_f2 = (float(p["N_p_cm3"]), float(p["z_p_um"]),
                                float(p["z_f1"]), float(p["z_f2"]))
        sub = curve_df[curve_df["file_base"] == fb].sort_values("depth_um")
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        keep = N > 0
        z_k, N_k = z[keep], N[keep]
        ln_N = np.log(N_k)
        N_dg = double_gaussian(z_k, N_p, z_p, z_f1, z_f2)
        N_dg = np.where(N_dg > 1e-30, N_dg, 1e-30)
        ln_N_dg = np.log(N_dg)

        if variant == "residual":
            y_target = ln_N - ln_N_dg
        else:
            y_target = ln_N

        for zi, yi, lndg in zip(z_k, y_target, ln_N_dg):
            x = [float(feat_row[c]) for c in feat_cols] + [float(zi)]
            x += [N_p, z_p, z_f1, z_f2]
            if variant in ("feature_aug", "struct_bias"):
                x += [float(lndg)]
            X_rows.append(x)
            y_vals.append(float(yi))
            groups.append(str(fb))
            weights.append(1.0)

    X = pd.DataFrame(X_rows, columns=feat_names)
    return X, np.array(y_vals, dtype=float), np.array(groups), np.array(weights, dtype=float)


def _build_pysr_with_custom_gauss(task_name: str):
    """PySR regressor with a registered custom operator gauss_kernel(z, z_p, sigma).

    PySR's API for custom operators: pass a Julia definition string inside
    ``binary_operators`` (it has 3 args so it counts as a binary-family op),
    and provide a Python equivalent via ``extra_sympy_mappings`` for export.
    Constants must be Float32 (suffix ``f0``).
    """
    from pysr import PySRRegressor
    from topcon_experiments.exp4_symbolic.sr_utils import (
        SR_BATCHING,
        SR_EXTRA_SYMPY_MAPPINGS,
        SR_MAXSIZE,
        SR_MODEL_SELECTION,
        SR_NESTED_CONSTRAINTS,
        SR_NITERATIONS,
        SR_PARSIMONY,
        SR_POPULATION_SIZE,
        SR_POPULATIONS,
        SR_UNARY_OPERATORS,
        SR_BINARY_OPERATORS,
    )

    sr_out = EXP4_OUT / "pysr_checkpoints" / task_name
    sr_out.mkdir(parents=True, exist_ok=True)

    def _gauss_kernel(z, z_p, sigma):
        z = np.asarray(z, dtype=float)
        return np.exp(-((z - np.asarray(z_p, dtype=float)) / np.asarray(sigma, dtype=float)) ** 2)

    # Julia definition. PySR vectorizes elementwise automatically.
    gauss_julia = "gauss_kernel(z, z_p, sigma) = exp(-((z - z_p) / sigma)^2)"

    # Must keep the existing custom-op mappings (e.g. inv) and add gauss_kernel.
    extra_sympy = dict(SR_EXTRA_SYMPY_MAPPINGS)
    extra_sympy["gauss_kernel"] = _gauss_kernel

    # gauss_kernel is arity-3 => must use the `operators` dict by arity,
    # which fully replaces binary_operators/unary_operators.
    operators = {
        1: SR_UNARY_OPERATORS,
        2: SR_BINARY_OPERATORS,
        3: [gauss_julia],
    }

    return PySRRegressor(
        populations=SR_POPULATIONS,
        population_size=SR_POPULATION_SIZE,
        operators=operators,
        extra_sympy_mappings=extra_sympy,
        nested_constraints=SR_NESTED_CONSTRAINTS,
        maxsize=SR_MAXSIZE,
        model_selection=SR_MODEL_SELECTION,
        parsimony=SR_PARSIMONY,
        niterations=SR_NITERATIONS,
        batching=SR_BATCHING,
        verbosity=1,
        progress=True,
        temp_equation_file=str(sr_out / "equations.csv"),
        tempdir=str(sr_out),
        delete_tempfiles=True,
    )


def _run_pysr(X, y, groups, w, task_name, variant):
    from topcon_experiments.exp4_symbolic.sr_eval_utils import (
        annotate_equation_metrics,
        subsample_train,
    )
    from topcon_experiments.exp4_symbolic.sr_utils import build_pysr_regressor

    train_idx, test_idx = split_group_train_test(groups)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y[train_idx], y[test_idx]
    w_train = w[train_idx]

    X_fit, y_fit, w_fit = subsample_train(X_train, y_train, 0, w_train)
    print(
        f"  PySR {task_name}: fit {len(X_fit)} pts "
        f"(train={len(X_train)}, test={len(X_test)}), variant={variant}"
    )

    if variant == "struct_bias":
        model = _build_pysr_with_custom_gauss(task_name)
        model.fit(X_fit.values, y_fit, weights=w_fit)
    else:
        model = build_pysr_regressor(task_name)
        model.fit(X_fit.values, y_fit, weights=w_fit)

    eq_df = model.equations_.copy()
    eq_df = annotate_equation_metrics(model, eq_df, X_train.values, y_train, X_test.values, y_test)
    eq_df["target"] = "ln_curve_prior" if variant != "residual" else "ln_residual"
    eq_df["n_train"] = len(X_train)
    eq_df["n_test"] = len(X_test)
    eq_df["n_fit"] = len(X_fit)
    eq_df["preprocess_key"] = "curve_prior"
    eq_df["preprocess_note"] = (
        f"adaptive BSG trim + absolute depth；DG fit on N>={FLOOR:.0e}；"
        f"SR points use N>0 inside trimmed window；先验=双高斯；variant={variant}；"
        f"feature_mode={'full' if 'full' in task_name else 'athena'}；"
        f"额外特征: {DG_FEAT_NAMES}"
        + (", ln_N_dg" if variant in ("feature_aug", "struct_bias") else "")
        + f"；目标=y{'(ln N - ln N_DG 残差)' if variant == 'residual' else '(ln N)'}"
    )
    return eq_df


def _best_row(path: Path):
    from topcon_experiments.exp4_symbolic.sr_equation_utils import best_row_by_test_r2
    return best_row_by_test_r2(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SR curve experiment with the literature double-Gaussian as prior.",
    )
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--skip-pysr", action="store_true",
                        help="Only build the prior datasets; do not train PySR.")
    parser.add_argument("--force", action="store_true",
                        help="Re-train even if formula CSV already exists.")
    args = parser.parse_args(argv)

    setup_runtime()
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    log(
        f"Prior SR: window=adaptive BSG trim; DG on N>={FLOOR:.0e}; "
        f"variants={args.variants}; prior=literature double-Gaussian (4 params)"
    )

    _prepare_window_csv()
    curve_df = pd.read_csv(EXP4_OUT / f"curve_processed_doping{SUFFIX}.csv")
    dg = _fit_dg_for_all_samples(curve_df)
    _plot_dg_baseline(curve_df, dg)
    meta = load_raw_dataframe().set_index("file_base")

    results = {}
    for step, variant in enumerate(args.variants, start=1):
        for curve_type, feature_mode in SR_JOBS:
            task = f"{curve_type}_{variant}_{feature_mode}"
            log_step(step * 2 - (1 if feature_mode == "full" else 0),
                     len(args.variants) * 2, f"PySR {variant} / {feature_mode}")
            X, y, g, w = _build_prior_dataset(curve_df, dg, meta,
                                              feature_mode=feature_mode, variant=variant)
            if len(X) < 50:
                log(f"Skip {variant}/{feature_mode}: only {len(X)} points")
                continue
            formula_path = EXP_OUT / f"sr_formulas_{curve_type}_{variant}_{feature_mode}.csv"
            if not args.skip_pysr and formula_path.exists() and not args.force:
                log(f"  {formula_path.name} already exists; skip (use --force to retrain)")
                if formula_path.exists():
                    results[(variant, curve_type, feature_mode)] = _best_row(formula_path)
                continue
            if not args.skip_pysr:
                eq_df = _run_pysr(X, y, g, w, task, variant)
                save_csv(eq_df, formula_path)
                best = _best_row(formula_path)
                log(
                    f"  saved {formula_path.name}: test_MSE={best.get('test_MSE', float('nan')):.4g} "
                    f"test_R2={best.get('test_R2', float('nan')):.4f}"
                )
                results[(variant, curve_type, feature_mode)] = best
            elif formula_path.exists():
                results[(variant, curve_type, feature_mode)] = _best_row(formula_path)
            else:
                log(f"No formulas at {formula_path}; run without --skip-pysr")

    log(f"Done -> {EXP_OUT}")


if __name__ == "__main__":
    main()
