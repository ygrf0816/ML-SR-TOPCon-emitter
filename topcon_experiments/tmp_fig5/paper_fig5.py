"""Paper-ready Fig.5 composite built from joint SR + LLM outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.config import EXP5_OUT
from topcon_experiments.tmp_fig5.utils import TMP_OUT


def _load_sim_csv(name: str) -> pd.DataFrame | None:
    path = EXP5_OUT / "data" / name
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    return df


def _plot_sim_from_csv(
    sim: pd.DataFrame,
    title: str,
    ax: plt.Axes,
    *,
    cmap: str = "coolwarm",
) -> None:
    arr = sim.astype(float).values
    n = len(sim)
    off = arr[~np.eye(n, dtype=bool)] if n > 1 else arr.ravel()
    vmin = max(0.0, float(np.percentile(off, 5))) if off.size else 0.6
    sns.heatmap(
        sim.astype(float),
        xticklabels=sim.columns,
        yticklabels=sim.index,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=1.0,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "cosine sim.", "shrink": 0.85},
    )
    ax.set_title(title, fontsize=10, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)


def plot_composite_paper(out_dir: Path | None = None) -> Path | None:
    out_dir = out_dir or TMP_OUT
    plot_dir = out_dir / "plots"

    paths = {
        "sr_llm_iv": plot_dir / "panel_sr_llm_iv_athena.png",
        "complexity": plot_dir / "panel_complexity_reduction_iv.png",
        "participation": plot_dir / "panel_feature_participation_iv.png",
        "physics": EXP5_OUT / "plots" / "semantic_iv_athena_physics.png",
        "blind": EXP5_OUT / "plots" / "semantic_iv_athena_blind_math.png",
        "pearson": EXP5_OUT / "plots" / "pearson_athena_iv.png",
    }
    if not paths["sr_llm_iv"].exists():
        return None

    apply_plot_style()
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.22)

    # Row 0: optimization story
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(plt.imread(paths["sr_llm_iv"]))
    ax_a.axis("off")
    ax_a.set_title("A  LLM simplification (IV)", fontsize=11, loc="left", fontweight="bold")

    ax_b = fig.add_subplot(gs[0, 1])
    if paths["complexity"].exists():
        ax_b.imshow(plt.imread(paths["complexity"]))
    ax_b.axis("off")
    ax_b.set_title("B  Complexity reduction", fontsize=11, loc="left", fontweight="bold")

    ax_c = fig.add_subplot(gs[0, 2])
    if paths["participation"].exists():
        ax_c.imshow(plt.imread(paths["participation"]))
    ax_c.axis("off")
    ax_c.set_title("C  Feature roles in best formulas", fontsize=11, loc="left", fontweight="bold")

    # Row 1: interpretation
    ax_d = fig.add_subplot(gs[1, 0])
    phys = _load_sim_csv("semantic_iv_athena_physics.csv")
    if phys is not None and not phys.empty:
        _plot_sim_from_csv(phys, "D  LLM physics semantics (IV)", ax_d, cmap="viridis")
    elif paths["physics"].exists():
        ax_d.imshow(plt.imread(paths["physics"]))
        ax_d.axis("off")
        ax_d.set_title("D  LLM physics semantics (IV)", fontsize=11, loc="left", fontweight="bold")
    else:
        ax_d.axis("off")
        ax_d.text(0.5, 0.5, "Run exp5 correlation", ha="center", va="center")

    ax_e = fig.add_subplot(gs[1, 1])
    blind = _load_sim_csv("semantic_iv_athena_blind_math.csv")
    if blind is not None and not blind.empty:
        _plot_sim_from_csv(blind, "E  LLM blind-math semantics (IV)", ax_e, cmap="YlGnBu")
    elif paths["blind"].exists():
        ax_e.imshow(plt.imread(paths["blind"]))
        ax_e.axis("off")
        ax_e.set_title("E  LLM blind-math semantics (IV)", fontsize=11, loc="left", fontweight="bold")
    else:
        ax_e.axis("off")
        ax_e.text(0.5, 0.5, "Run exp5 correlation", ha="center", va="center")

    ax_f = fig.add_subplot(gs[1, 2])
    if paths["pearson"].exists():
        ax_f.imshow(plt.imread(paths["pearson"]))
        ax_f.axis("off")
        ax_f.set_title("F  Physical Pearson (Athena–IV)", fontsize=11, loc="left", fontweight="bold")
    else:
        pearson = _load_sim_csv("pearson_athena_iv.csv")
        if pearson is not None and not pearson.empty:
            sns.heatmap(
                pearson.astype(float),
                ax=ax_f,
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                cbar_kws={"label": "Pearson r", "shrink": 0.85},
            )
            ax_f.set_title("F  Physical Pearson (Athena–IV)", fontsize=10, loc="left")
            plt.setp(ax_f.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            plt.setp(ax_f.get_yticklabels(), fontsize=7)
        else:
            ax_f.axis("off")
            ax_f.text(0.5, 0.5, "Run exp5 correlation", ha="center", va="center")

    fig.suptitle(
        "Symbolic regression optimization with LLM interpretation (joint multivariate formulas)",
        fontsize=13,
        y=0.98,
    )
    out = plot_dir / "fig5_paper_composite.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
