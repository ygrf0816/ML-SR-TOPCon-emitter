"""Plotting helpers with mandatory CSV export."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from topcon_experiments.common.variable_labels import label_for

_CJK_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "WenQuanYi Micro Hei",
    "SimHei",
    "Microsoft YaHei",
)


def setup_matplotlib_cjk() -> str | None:
    """Configure matplotlib for CJK labels; returns chosen font name if any."""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in _CJK_FONT_CANDIDATES if name in available]
    if chosen:
        mpl.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.unicode_minus"] = False
    return chosen[0] if chosen else None


def _np(s: "pd.Series | np.ndarray") -> np.ndarray:
    """Coerce to ndarray before handing data to matplotlib.

    matplotlib 3.4's ``cbook._check_1d`` probes ``x[:, None]``; pandas >= 2.0
    raises ValueError instead of warning, and matplotlib only catches
    AssertionError/IndexError/TypeError, so any ``ax.plot(Series, ...)`` blows up.
    """
    return np.asarray(s)


def save_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def plot_regression_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    out_prefix: Path,
    log_scale: bool = False,
) -> None:
    y_true_col = f"y_true_{target}"
    y_pred_col = f"y_pred_{target}"
    df = pd.DataFrame(
        {
            "target_column": target,
            "target_label": label_for(target),
            y_true_col: y_true,
            y_pred_col: y_pred,
        }
    )
    save_csv(df, out_prefix.with_suffix(".csv"))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, s=12, edgecolors="none")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"True {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"{target}: True vs Predicted")
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_feature_importance(importance_df: pd.DataFrame, title: str, out_prefix: Path) -> None:
    save_csv(importance_df, out_prefix.with_suffix(".csv"))
    plot_df = importance_df.sort_values("importance", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_df))))
    ax.barh(_np(plot_df["feature"]), _np(plot_df["importance"]))
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_trace(trace_df: pd.DataFrame, metric: str, out_prefix: Path) -> None:
    """Plot a DE convergence trace.

    Two different sampling units live in this frame and must not be mixed:

      * ``best``     - cumulative running max over ALL evaluations so far, hence
                       monotone non-decreasing. Hard upper bound on *every*
                       individual ever evaluated (including the current one).
      * ``mean``, ``std``, ``median``, ``p05..p95``, ``gen_best``, ``gen_worst``
                     - statistics of the individuals evaluated *within that
                       generation only*.

    ``mean +/- 1 std`` is a dispersion measure, not an interval. Because a
    converged DE population is strongly left-skewed (most members hug the
    optimum, a few stragglers sit far below), ``mean + std`` can legitimately
    exceed ``best``: by Popoviciu's inequality std <= (max - min) / 2, so
    ``mean + std`` overshoots the max as soon as the mean sits close to it.
    Observed on the archived runs: 25/30 generations for iv_Eff and 29/30 for
    iv_Jsc. Drawing that as a shaded band is therefore misleading.

    The band is consequently drawn from population QUANTILES (hard-bounded by
    the achievable range, so it can never overshoot), while ``std`` is plotted
    separately on the right axis as a population-diversity indicator.
    """
    save_csv(trace_df, out_prefix.with_suffix(".csv"))
    gen = _np(trace_df["gen"])
    best = _np(trace_df["best"])
    mean = _np(trace_df["mean"])
    has_q = {"p05", "p25", "p75", "p95"}.issubset(trace_df.columns)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gen, best, color="C3", lw=2.2, label="best (cumulative)")

    if has_q:
        ax.fill_between(
            gen, _np(trace_df["p05"]), _np(trace_df["p95"]),
            color="C0", alpha=0.15, lw=0, label="population 5-95%",
        )
        ax.fill_between(
            gen, _np(trace_df["p25"]), _np(trace_df["p75"]),
            color="C0", alpha=0.30, lw=0, label="population 25-75%",
        )
        ax.plot(
            gen, _np(trace_df["median"]), color="C0", lw=1.4,
            label="median (per gen)",
        )
        if "gen_best" in trace_df.columns:
            ax.plot(
                gen, _np(trace_df["gen_best"]), color="C1", ls="--", lw=1.4,
                label="best in generation",
            )
    else:
        # Legacy 4-column trace (gen/best/mean/std): quantiles were never stored,
        # so clip the band at the hard upper bound `best`. No evaluated
        # individual can exceed `best`, hence a band reaching above it is
        # provably outside the achievable range.
        std = _np(trace_df["std"])
        upper = np.minimum(mean + std, best)
        ax.fill_between(
            gen, mean - std, upper,
            color="C0", alpha=0.25, lw=0,
            label="mean +/- std (upper clipped at best)",
        )
    ax.plot(gen, mean, color="C2", lw=1.2, alpha=0.9, label="mean (per gen)")

    ax.set_xlabel("Generation")
    ax.set_ylabel(metric)
    ax.set_title(f"DE convergence: maximize {metric}")

    if "std" in trace_df.columns:
        ax2 = ax.twinx()
        ax2.plot(
            gen, _np(trace_df["std"]), color="C7", ls=":", lw=1.6,
            label="std (population diversity)",
        )
        ax2.set_ylabel(f"std of {metric}")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_pareto_2d(df: pd.DataFrame, x_col: str, y_col: str, out_prefix: Path) -> None:
    save_csv(df, out_prefix.with_suffix(".csv"))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(_np(df[x_col]), _np(df[y_col]), alpha=0.6, s=20)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Pareto front: {x_col} vs {y_col}")
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_prefix: Path) -> None:
    df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    save_csv(df.reset_index(), out_prefix.with_suffix(".csv"))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=150)
    plt.close(fig)
