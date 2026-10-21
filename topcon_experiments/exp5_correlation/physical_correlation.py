"""Pearson correlation matrices from simulation dataset (physical baseline)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.common.variable_labels import label_for
from topcon_experiments.config import (
    ATHENA_FEATURES,
    CURVE_DESCRIPTORS,
    EXP5_OUT,
    IV_TARGETS,
    LOG_FEATURES,
    MODEL2_FEATURES,
)


def _display_name(col: str) -> str:
    if col in IV_TARGETS:
        from topcon_experiments.config import IV_TARGET_LABELS

        return IV_TARGET_LABELS.get(col, col)
    return label_for(col).split("[")[0].strip()


def _numeric_frame(columns: list[str], *, log_cols: set[str]) -> pd.DataFrame:
    df = load_raw_dataframe()
    mask = pd.Series(True, index=df.index)
    use_cols = [c for c in columns if c in df.columns]
    for c in use_cols:
        if c in log_cols:
            mask &= df[c] > 0
        else:
            mask &= np.isfinite(df[c].astype(float))
    sub = df.loc[mask, use_cols].astype(float).copy()
    for c in use_cols:
        if c in log_cols:
            sub[c] = np.log(sub[c])
    return sub


def _plot_corr(mat: pd.DataFrame, title: str, out_png: Path) -> None:
    apply_plot_style()
    n = len(mat)
    figsize = (max(8, n * 0.55), max(6, n * 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mat,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "Pearson r"},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_physical_correlation(out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or EXP5_OUT
    plot_dir = out_dir / "plots"
    csv_dir = out_dir / "data"
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, list[str], str]] = [
        ("athena_iv", ATHENA_FEATURES + IV_TARGETS, "Athena vs IV (Pearson)"),
        ("descriptors_iv", CURVE_DESCRIPTORS + IV_TARGETS, "Curve descriptors vs IV (Pearson)"),
        ("model2_iv", MODEL2_FEATURES + IV_TARGETS, "Model2 features + IV (Pearson)"),
    ]

    written: list[Path] = []
    log_cols = set(LOG_FEATURES)
    for slug, cols, title in panels:
        frame = _numeric_frame(cols, log_cols=log_cols)
        corr = frame.corr(method="pearson")
        corr.index = [_display_name(c) for c in corr.index]
        corr.columns = [_display_name(c) for c in corr.columns]
        csv_path = csv_dir / f"pearson_{slug}.csv"
        save_csv(corr.reset_index().rename(columns={"index": "variable"}), csv_path)
        png_path = plot_dir / f"pearson_{slug}.png"
        _plot_corr(corr, title, png_path)
        written.extend([csv_path, png_path])
    return written
