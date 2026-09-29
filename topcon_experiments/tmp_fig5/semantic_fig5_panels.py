"""Supplementary semantic panels from exp5 (task-level, joint SR formulas)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from topcon_experiments.common.log_utils import log
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.config import EXP5_OUT
from topcon_experiments.tmp_fig5.utils import TMP_OUT


def _plot_sim_csv(csv_path: Path, title: str, out_png: Path, *, cmap: str = "coolwarm") -> None:
    if not csv_path.exists():
        return
    sim = pd.read_csv(csv_path, index_col=0).astype(float)
    if sim.shape[0] < 2:
        log(f"  skip {csv_path.name}: only {sim.shape[0]} label(s)")
        return
    apply_plot_style()
    n = len(sim)
    fig, ax = plt.subplots(figsize=(max(5.5, n * 0.75), max(5, n * 0.7)))
    arr = sim.values
    off = arr[~np.eye(n, dtype=bool)] if n > 1 else arr.ravel()
    vmin = max(0.0, float(np.percentile(off, 5))) if off.size else 0.6
    sns.heatmap(
        sim,
        xticklabels=sim.columns,
        yticklabels=sim.index,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=1.0,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "cosine similarity"},
    )
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(out_dir: Path | None = None) -> list[Path]:
    """Copy-style render of exp5 semantic CSVs into tmp_fig5/plots/semantic/."""
    out_dir = out_dir or TMP_OUT
    plot_dir = out_dir / "plots" / "semantic"
    plot_dir.mkdir(parents=True, exist_ok=True)
    exp5_data = EXP5_OUT / "data"

    panels = [
        ("semantic_iv_athena_physics.csv", "IV — LLM physics (joint best)", "iv_physics.png", "viridis"),
        ("semantic_iv_athena_blind_math.csv", "IV — LLM blind math (joint best)", "iv_blind_math.png", "YlGnBu"),
        ("semantic_doping_descriptor_physics.csv", "Doping descriptors — LLM physics", "doping_physics.png", "viridis"),
        ("semantic_defect_descriptor_physics.csv", "Defect descriptors — LLM physics", "defect_physics.png", "viridis"),
        ("semantic_full_iv_physics.csv", "Full model → IV — LLM physics", "full_iv_physics.png", "viridis"),
        ("semantic_curve_physics.csv", "Concentration curves — LLM physics", "curve_physics.png", "YlGnBu"),
    ]

    written: list[Path] = []
    for csv_name, title, png_name, cmap in panels:
        src = exp5_data / csv_name
        dst = plot_dir / png_name
        if not src.exists():
            log(f"  missing exp5 data: {csv_name} (run exp5_correlation first)")
            continue
        _plot_sim_csv(src, title, dst, cmap=cmap)
        if dst.exists():
            written.append(dst)
    return written
